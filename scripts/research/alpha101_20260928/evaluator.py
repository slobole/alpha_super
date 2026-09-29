"""Evaluator with degree tracking and the point-in-time conversion (PREREG section 4; research only).

Every value carries its degree of homogeneity (p in price, v in volume). Raw inputs: open/high/low/close/vwap (1, 0),
volume and adv{d} (0, 1), returns (0, 0), constants (0, 0). Products add degrees, quotients subtract them, constant
powers multiply them; sign, ts_rank, ts_argmax/argmin, correlation are scale-free (0, 0); delay, delta, sum, ts_min,
ts_max, stddev, decay_linear, abs keep the degree; covariance adds; product multiplies by the window.

Conversion to "the value a trader saw that day": x_t x k_t^p x kv_t^v with k_t = Unadjusted Close_t / Close_t and
kv_t = 1 / k_t. It is applied (1) before rank, scale and indneutralize, (2) before any non-homogeneous node: a sum,
difference, comparison, ternary or elementwise min/max of terms of different degree (a constant counts as degree 0), a
non-constant exponent, log of a non-zero-degree quantity, and (3) to the final alpha value. Converted values have
degree (0, 0) and are never rescaled again. Tie-sensitive decisions (comparisons, rank ties, constant windows, a sum or
difference that is zero in traded prices) use the data-precision tolerance of operators.TIE_REL_FLOAT so that they do
not depend on the floating-point path (download date, per-stock rescaling).
"""

from __future__ import annotations

import dataclasses

import numpy as np

from alpha101_20260928 import operators as ops
from alpha101_20260928.parser import Binary, Call, Group, Num, Ternary, Unary, Var, parse

ZERO_DEG = (0.0, 0.0)
PRICE_VAR_TUPLE = ("open", "high", "low", "close", "vwap")


@dataclasses.dataclass
class Value:
    x: object  # (T, S) float64 array or python float (constant)
    deg: tuple  # (price degree, volume degree)

    @property
    def is_const(self) -> bool:
        return not isinstance(self.x, np.ndarray)


class Context:
    """Data needed by the evaluator for one universe (all (T, S) float64 arrays)."""

    def __init__(self, panel_dict: dict, member_arr: np.ndarray, k_arr: np.ndarray, group_dict: dict[str, np.ndarray]):
        self.panel_dict = panel_dict  # open, high, low, close, volume, vwap, returns; adv{d} built lazily from volume
        self.member_arr = member_arr.astype(bool)
        self.k_arr = k_arr
        with np.errstate(divide="ignore", invalid="ignore"):
            self.kv_arr = 1.0 / k_arr
        self.group_dict = group_dict  # level -> int codes per symbol
        self.shape = panel_dict["close"].shape
        self.conversion_count_int = 0

    def variable(self, name_str: str) -> Value:
        if name_str in PRICE_VAR_TUPLE:
            return Value(self.panel_dict[name_str], (1.0, 0.0))
        if name_str == "volume":
            return Value(self.panel_dict["volume"], (0.0, 1.0))
        if name_str == "returns":
            return Value(self.panel_dict["returns"], ZERO_DEG)
        if name_str.startswith("adv") and name_str[3:].isdigit():
            key_str = name_str
            if key_str not in self.panel_dict:
                # mean share Volume over the last d sessions including t, d valid values required (PREREG section 4)
                self.panel_dict[key_str] = ops.ts_sum(self.panel_dict["volume"], int(name_str[3:])) / int(name_str[3:])
            return Value(self.panel_dict[key_str], (0.0, 1.0))
        if name_str == "cap":
            raise ValueError("market cap is not available point-in-time (alpha 56 is excluded)")
        raise NameError(f"unknown input {name_str!r}")

    # ------------------------------------------------------------------ conversion
    def convert(self, value: Value) -> Value:
        if value.deg == ZERO_DEG:
            return value
        if value.is_const:
            return Value(value.x, ZERO_DEG)
        p_float, v_float = value.deg
        factor_arr = np.ones(self.shape)
        with np.errstate(all="ignore"):
            if p_float != 0.0:
                factor_arr = factor_arr * (self.k_arr if p_float == 1.0 else np.power(self.k_arr, p_float))
            if v_float != 0.0:
                factor_arr = factor_arr * (self.kv_arr if v_float == 1.0 else np.power(self.kv_arr, v_float))
            out_arr = value.x * factor_arr
        self.conversion_count_int += 1
        return Value(ops.clean(out_arr), ZERO_DEG)

    def broadcast(self, value: Value) -> np.ndarray:
        return np.full(self.shape, float(value.x)) if value.is_const else value.x


def _homogenize(context: Context, a: Value, b: Value) -> tuple[Value, Value, tuple]:
    """Two operands of a sum / comparison / ternary / elementwise min-max: equal degrees are kept, otherwise both are
    converted (a constant counts as degree 0, so a constant plus a priced quantity converts the quantity)."""
    if a.deg == b.deg:
        return a, b, a.deg
    return context.convert(a), context.convert(b), ZERO_DEG


def _const(node) -> float | None:
    if isinstance(node, Num):
        return node.value
    if isinstance(node, Unary) and node.op == "-" and isinstance(node.x, Num):
        return -node.x.value
    return None


def evaluate(node, context: Context) -> Value:
    if isinstance(node, Num):
        return Value(float(node.value), ZERO_DEG)
    if isinstance(node, Var):
        return context.variable(node.name)
    if isinstance(node, Group):
        raise ValueError("IndClass reference outside indneutralize")
    if isinstance(node, Unary):
        value = evaluate(node.x, context)
        if node.op == "-":
            return Value(-value.x if value.is_const else ops.clean(-value.x), value.deg)
        raise ValueError(node.op)
    if isinstance(node, Ternary):
        cond = context.convert(evaluate(node.cond, context))
        a, b, deg = _homogenize(context, evaluate(node.a, context), evaluate(node.b, context))
        return Value(ops.nan_where(context.broadcast(cond), context.broadcast(a), context.broadcast(b)), deg)
    if isinstance(node, Binary):
        return _binary(node, context)
    if isinstance(node, Call):
        return _call(node, context)
    raise TypeError(type(node))


def _binary(node: Binary, context: Context) -> Value:
    op_str = node.op
    a = evaluate(node.x, context)
    if op_str == "^":
        exponent_const = _const(node.y)
        if exponent_const is not None:
            deg = (a.deg[0] * exponent_const, a.deg[1] * exponent_const)
            if a.is_const:
                return Value(float(np.power(a.x, exponent_const)), ZERO_DEG)
            return Value(ops.power(a.x, exponent_const), deg)
        b = evaluate(node.y, context)
        a, b = context.convert(a), context.convert(b)  # a non-constant exponent is not homogeneous
        return Value(ops.power(context.broadcast(a), context.broadcast(b)), ZERO_DEG)
    b = evaluate(node.y, context)
    if a.is_const and b.is_const:
        with np.errstate(all="ignore"):
            if op_str == "+":
                out = a.x + b.x
            elif op_str == "-":
                out = a.x - b.x
            elif op_str == "*":
                out = a.x * b.x
            elif op_str == "/":
                out = a.x / b.x
            elif op_str in ("<", ">", "=="):
                out = float(ops.nan_compare(a.x, b.x, op_str))
            elif op_str in ("&&", "||"):
                out = float(ops.nan_logical(a.x, b.x, op_str))
            else:
                raise ValueError(op_str)
        return Value(float(out) if np.isfinite(out) else float("nan"), ZERO_DEG)
    if op_str == "*":
        with np.errstate(all="ignore"):
            return Value(ops.clean(a.x * b.x), (a.deg[0] + b.deg[0], a.deg[1] + b.deg[1]))
    if op_str == "/":
        with np.errstate(all="ignore"):
            return Value(ops.clean(a.x / b.x), (a.deg[0] - b.deg[0], a.deg[1] - b.deg[1]))
    if op_str in ("+", "-"):
        a, b, deg = _homogenize(context, a, b)
        with np.errstate(all="ignore"):
            out_arr = a.x + b.x if op_str == "+" else a.x - b.x
        return Value(ops.clean(ops.snap_sum(out_arr, a.x, b.x)), deg)
    if op_str in ("<", ">", "=="):
        a, b, _ = _homogenize(context, a, b)
        return Value(ops.nan_compare(context.broadcast(a), context.broadcast(b), op_str), ZERO_DEG)
    if op_str in ("&&", "||"):
        a, b = context.convert(a), context.convert(b)
        return Value(ops.nan_logical(context.broadcast(a), context.broadcast(b), op_str), ZERO_DEG)
    raise ValueError(op_str)


def _window_arg(node) -> float:
    d_float = _const(node)
    if d_float is None:
        raise ValueError("a window argument must be a numeric constant")
    return d_float


def _panel(context: Context, value: Value) -> np.ndarray:
    return context.broadcast(value)


def _call(node: Call, context: Context) -> Value:
    name_str = node.name
    args = node.args
    if name_str in ("rank", "scale", "indneutralize"):
        x = context.convert(evaluate(args[0], context))
        x_arr = _panel(context, x)
        if name_str == "rank":
            return Value(ops.cs_rank(x_arr, context.member_arr), ZERO_DEG)
        if name_str == "scale":
            a_float = _window_arg(args[1]) if len(args) > 1 else 1.0
            return Value(ops.cs_scale(x_arr, context.member_arr, a_float), ZERO_DEG)
        if not isinstance(args[1], Group):
            raise ValueError("indneutralize needs an IndClass level")
        return Value(ops.cs_indneutralize(x_arr, context.member_arr, context.group_dict[args[1].level]), ZERO_DEG)
    if name_str == "abs":
        x = evaluate(args[0], context)
        return Value(abs(x.x) if x.is_const else np.abs(x.x), x.deg)
    if name_str == "sign":
        x = evaluate(args[0], context)
        return Value(float(np.sign(x.x)) if x.is_const else np.sign(x.x), ZERO_DEG)
    if name_str == "log":
        x = context.convert(evaluate(args[0], context))
        return Value(float(np.log(x.x)) if x.is_const else ops.log(x.x), ZERO_DEG)
    if name_str == "signedpower":
        x = evaluate(args[0], context)
        exponent_const = _const(args[1])
        if exponent_const is not None:
            deg = (x.deg[0] * exponent_const, x.deg[1] * exponent_const)
            return Value(ops.signed_power(_panel(context, x), exponent_const), deg)
        a = context.convert(evaluate(args[1], context))
        x = context.convert(x)
        return Value(ops.signed_power(_panel(context, x), _panel(context, a)), ZERO_DEG)
    if name_str in ("min", "max") and len(args) == 2 and _const(args[1]) is None:
        a, b, deg = _homogenize(context, evaluate(args[0], context), evaluate(args[1], context))
        fn = np.minimum if name_str == "min" else np.maximum
        return Value(ops.clean(fn(_panel(context, a), _panel(context, b))), deg)
    # time-series operators: (x, d) or (x, y, d)
    if name_str in ("correlation", "covariance"):
        x = evaluate(args[0], context)
        y = evaluate(args[1], context)
        d_float = _window_arg(args[2])
        if name_str == "correlation":
            return Value(ops.ts_correlation(_panel(context, x), _panel(context, y), d_float), ZERO_DEG)
        return Value(ops.ts_covariance(_panel(context, x), _panel(context, y), d_float), (x.deg[0] + y.deg[0], x.deg[1] + y.deg[1]))
    x = evaluate(args[0], context)
    d_float = _window_arg(args[1])
    x_arr = _panel(context, x)
    if name_str == "delay":
        return Value(ops.delay(x_arr, d_float), x.deg)
    if name_str == "delta":
        return Value(ops.delta(x_arr, d_float), x.deg)
    if name_str == "sum":
        return Value(ops.ts_sum(x_arr, d_float), x.deg)
    if name_str == "product":
        d_int = ops.floor_window(d_float)
        return Value(ops.ts_product(x_arr, d_float), (x.deg[0] * d_int, x.deg[1] * d_int))
    if name_str in ("ts_min", "min"):
        return Value(ops.ts_min(x_arr, d_float), x.deg)
    if name_str in ("ts_max", "max"):
        return Value(ops.ts_max(x_arr, d_float), x.deg)
    if name_str == "ts_argmax":
        return Value(ops.ts_argmax(x_arr, d_float), ZERO_DEG)
    if name_str == "ts_argmin":
        return Value(ops.ts_argmin(x_arr, d_float), ZERO_DEG)
    if name_str == "ts_rank":
        return Value(ops.ts_rank(x_arr, d_float), ZERO_DEG)
    if name_str == "stddev":
        return Value(ops.ts_stddev(x_arr, d_float), x.deg)
    if name_str == "decay_linear":
        return Value(ops.decay_linear(x_arr, d_float), x.deg)
    raise NameError(f"unknown function {name_str!r}")


def evaluate_formula_with_degree(formula_str: str, context: Context) -> tuple[np.ndarray, tuple]:
    """Final alpha panel in day-t units (the root value converted by its degree) and the root degree before that
    conversion; +-inf -> NaN; non-members NaN."""
    root_value = evaluate(parse(formula_str), context)
    value = context.convert(root_value)
    out_arr = ops.clean(context.broadcast(value)).copy()
    out_arr[~context.member_arr] = np.nan
    return out_arr, root_value.deg


def evaluate_formula(formula_str: str, context: Context) -> np.ndarray:
    return evaluate_formula_with_degree(formula_str, context)[0]
