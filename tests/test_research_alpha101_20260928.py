"""Unit tests for the Alpha101 study (PREREG section 9; research only).

Fast tests (synthetic data) cover the parser, every operator against a naive loop implementation on random panels with
NaNs, the PIT conversion on a synthetic 2:1 split, the composites and the pod's buffer rule. Tests marked `integration`
read the study caches (skipped when they are absent).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT_PATH = Path(__file__).resolve().parents[1]
for path_obj in (REPO_ROOT_PATH, REPO_ROOT_PATH / "scripts" / "research"):
    if str(path_obj) not in sys.path:
        sys.path.insert(0, str(path_obj))

from alpha101_20260928 import formulas  # noqa: E402
from alpha101_20260928 import operators as ops  # noqa: E402
from alpha101_20260928.evaluator import Context, evaluate_formula, evaluate_formula_with_degree  # noqa: E402
from alpha101_20260928.parser import Binary, Call, Group, Num, Ternary, Unary, Var, parse  # noqa: E402

RNG = np.random.default_rng(20260928)


def random_panel(t_int: int = 60, s_int: int = 7, nan_share_float: float = 0.08) -> np.ndarray:
    x_arr = RNG.normal(0, 1, (t_int, s_int)).cumsum(axis=0) + 50.0
    x_arr[RNG.random((t_int, s_int)) < nan_share_float] = np.nan
    x_arr[10:13, 2] = 7.0  # a constant stretch
    x_arr[20, 3] = x_arr[19, 3]  # a tie
    return x_arr


def naive_rolling(x_arr, d_int, fn):
    t_int, s_int = x_arr.shape
    out_arr = np.full((t_int, s_int), np.nan)
    for t in range(d_int - 1, t_int):
        for s in range(s_int):
            w = x_arr[t - d_int + 1 : t + 1, s]
            if np.isfinite(w).all():
                out_arr[t, s] = fn(w)
    return out_arr


def naive_rolling2(x_arr, y_arr, d_int, fn):
    t_int, s_int = x_arr.shape
    out_arr = np.full((t_int, s_int), np.nan)
    for t in range(d_int - 1, t_int):
        for s in range(s_int):
            wx, wy = x_arr[t - d_int + 1 : t + 1, s], y_arr[t - d_int + 1 : t + 1, s]
            if np.isfinite(wx).all() and np.isfinite(wy).all():
                out_arr[t, s] = fn(wx, wy)
    return out_arr


def assert_same(a_arr, b_arr, rel_float=1e-12):
    assert a_arr.shape == b_arr.shape
    assert np.array_equal(np.isnan(a_arr), np.isnan(b_arr)), "NaN pattern differs"
    both_arr = np.isfinite(a_arr) & np.isfinite(b_arr)
    scale_arr = np.maximum(np.abs(a_arr[both_arr]), 1.0)
    assert np.all(np.abs(a_arr[both_arr] - b_arr[both_arr]) <= rel_float * scale_arr)


# ----------------------------------------------------------------------------------------------------------------------
# parser
# ----------------------------------------------------------------------------------------------------------------------
def test_parser_precedence_and_forms():
    assert parse("-x^2") == Unary("-", Binary("^", Var("x"), Num(2.0)))
    assert parse("a + b * c") == Binary("+", Var("a"), Binary("*", Var("b"), Var("c")))
    assert parse("a * b + c") == Binary("+", Binary("*", Var("a"), Var("b")), Var("c"))
    assert parse("a - b - c") == Binary("-", Binary("-", Var("a"), Var("b")), Var("c"))
    assert parse("a < b + c") == Binary("<", Var("a"), Binary("+", Var("b"), Var("c")))
    assert parse("a < b && c == d || e") == Binary("||", Binary("&&", Binary("<", Var("a"), Var("b")), Binary("==", Var("c"), Var("d"))), Var("e"))
    assert parse("a ? b : c ? d : e") == Ternary(Var("a"), Var("b"), Ternary(Var("c"), Var("d"), Var("e")))
    assert parse("(a < 0) ? b : c * 2") == Ternary(Binary("<", Var("a"), Num(0.0)), Var("b"), Binary("*", Var("c"), Num(2.0)))
    assert parse("2. * .001") == Binary("*", Num(2.0), Num(0.001))
    assert parse("Ts_Rank(Close, 5.5)") == Call("ts_rank", (Var("close"), Num(5.5)))
    assert parse("IndNeutralize(vwap, IndClass.Sector)") == Call("indneutralize", (Var("vwap"), Group("sector")))
    assert parse("(-1 * 1)") == Binary("*", Unary("-", Num(1.0)), Num(1.0))
    assert parse("x * -1") == Binary("*", Var("x"), Unary("-", Num(1.0)))
    for n, formula_str in formulas.FORMULA_DICT.items():
        parse(formula_str)


def test_transcription_check_passes():
    if not formulas.PAPER_TEXT_PATH.exists():
        pytest.skip("paper text not available")
    assert formulas.transcription_check()["passed_bool"]


# ----------------------------------------------------------------------------------------------------------------------
# time-series operators against naive loops
# ----------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("d_float", [1.0, 3.0, 5.9, 12.0])
def test_ts_sum_min_max_product_delay_delta(d_float):
    x_arr = random_panel()
    d_int = int(np.floor(d_float))
    assert_same(ops.ts_sum(x_arr, d_float), naive_rolling(x_arr, d_int, np.sum))
    assert_same(ops.ts_min(x_arr, d_float), naive_rolling(x_arr, d_int, np.min))
    assert_same(ops.ts_max(x_arr, d_float), naive_rolling(x_arr, d_int, np.max))
    small_arr = x_arr / 50.0
    assert_same(ops.ts_product(small_arr, d_float), naive_rolling(small_arr, d_int, np.prod), 1e-10)
    delay_arr = np.full_like(x_arr, np.nan)
    delay_arr[d_int:] = x_arr[:-d_int]
    assert_same(ops.delay(x_arr, d_float), delay_arr)
    assert_same(ops.delta(x_arr, d_float), x_arr - delay_arr)


@pytest.mark.parametrize("d_float", [2.0, 4.7, 9.0])
def test_ts_stddev_cov_corr(d_float):
    x_arr = random_panel()
    y_arr = random_panel() * 0.3 + x_arr * 0.5
    d_int = int(np.floor(d_float))

    def sd(w):
        v = np.var(w, ddof=1)
        return np.sqrt(v) if v > 0 else np.nan

    def cov(wx, wy):
        if np.var(wx) == 0 or np.var(wy) == 0:
            return np.nan
        return np.cov(wx, wy, ddof=1)[0, 1]

    def corr(wx, wy):
        if np.var(wx) == 0 or np.var(wy) == 0:
            return np.nan
        return np.corrcoef(wx, wy)[0, 1]

    assert_same(ops.ts_stddev(x_arr, d_float), naive_rolling(x_arr, d_int, sd), 1e-10)
    assert_same(ops.ts_covariance(x_arr, y_arr, d_float), naive_rolling2(x_arr, y_arr, d_int, cov), 1e-10)
    assert_same(ops.ts_correlation(x_arr, y_arr, d_float), naive_rolling2(x_arr, y_arr, d_int, corr), 1e-10)
    # constant window -> NaN
    assert np.isnan(ops.ts_stddev(x_arr, 3.0)[12, 2])
    assert np.isnan(ops.ts_correlation(x_arr, y_arr, 3.0)[12, 2])


@pytest.mark.parametrize("d_float", [1.0, 4.0, 7.3])
def test_ts_rank_argmax_argmin_decay(d_float):
    x_arr = random_panel()
    d_int = int(np.floor(d_float))

    def tsr(w):
        last = w[-1]
        return ((w < last).sum() + ((w == last).sum() + 1) / 2.0) / d_int

    def argmax_days_ago(w):
        idx = np.flatnonzero(w == w.max())
        return float(len(w) - 1 - idx.max())  # most recent occurrence

    def argmin_days_ago(w):
        idx = np.flatnonzero(w == w.min())
        return float(len(w) - 1 - idx.max())

    def decay(w):
        weight_vec = np.arange(1, len(w) + 1, dtype=float)
        return float(np.dot(w, weight_vec) / weight_vec.sum())

    assert_same(ops.ts_rank(x_arr, d_float), naive_rolling(x_arr, d_int, tsr))
    assert_same(ops.ts_argmax(x_arr, d_float), naive_rolling(x_arr, d_int, argmax_days_ago))
    assert_same(ops.ts_argmin(x_arr, d_float), naive_rolling(x_arr, d_int, argmin_days_ago))
    assert_same(ops.decay_linear(x_arr, d_float), naive_rolling(x_arr, d_int, decay), 1e-12)
    # today = max -> 0 days ago; ts_rank of a strictly increasing window = 1
    inc_arr = np.arange(30, dtype=float)[:, None] * np.ones((1, 2))
    assert np.all(ops.ts_argmax(inc_arr, 5)[4:] == 0.0)
    assert np.all(ops.ts_argmin(inc_arr, 5)[4:] == 4.0)
    assert np.allclose(ops.ts_rank(inc_arr, 5)[4:], 1.0)
    tie_arr = np.array([[1.0], [3.0], [3.0], [2.0]])
    assert ops.ts_rank(tie_arr, 4)[3, 0] == pytest.approx(2.0 / 4.0)  # 1 below, 2 above -> rank 2
    assert ops.ts_argmax(np.array([[3.0], [1.0], [3.0]]), 3)[2, 0] == 0.0  # tie -> most recent


def test_floor_window_and_bad_windows():
    assert ops.floor_window(3.92795) == 3
    assert ops.floor_window(17.9282) == 17
    with pytest.raises(ValueError):
        ops.floor_window(0.5)


# ----------------------------------------------------------------------------------------------------------------------
# cross-sectional operators
# ----------------------------------------------------------------------------------------------------------------------
def test_cs_rank_scale_indneutralize():
    x_arr = np.array([[3.0, 1.0, 2.0, 2.0, np.nan, 9.0], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])
    member_arr = np.array([[True, True, True, True, True, False], [True, True, True, False, False, False]])
    rank_arr = ops.cs_rank(x_arr, member_arr)
    assert np.allclose(rank_arr[0, :4], np.array([4.0, 1.0, 2.5, 2.5]) / 4.0)
    assert np.isnan(rank_arr[0, 4]) and np.isnan(rank_arr[0, 5])
    assert np.allclose(rank_arr[1, :3], np.array([1.0, 2.0, 3.0]) / 3.0) and np.isnan(rank_arr[1, 3:]).all()
    scale_arr = ops.cs_scale(x_arr, member_arr)
    assert np.allclose(np.nansum(np.abs(scale_arr[0])), 1.0) and np.isnan(scale_arr[0, 5])
    assert np.allclose(np.nansum(np.abs(ops.cs_scale(x_arr, member_arr, 2.0)[1])), 2.0)
    group_vec = np.array([0, 0, 1, 1, 1, 0])
    neutral_arr = ops.cs_indneutralize(x_arr, member_arr, group_vec)
    assert np.allclose(neutral_arr[0, :2], [1.0, -1.0])  # group 0 members: 3, 1 (9 is not a member)
    assert np.allclose(neutral_arr[0, 2:4], [0.0, 0.0]) and np.isnan(neutral_arr[0, 4])
    assert np.allclose(neutral_arr[1, :2], [-0.5, 0.5]) and neutral_arr[1, 2] == 0.0 and np.isnan(neutral_arr[1, 3:]).all()


def test_boolean_semantics():
    a_arr = np.array([[1.0, np.nan, 2.0]])
    b_arr = np.array([[2.0, 1.0, 2.0]])
    assert np.array_equal(ops.nan_compare(a_arr, b_arr, "<")[0, [0, 2]], [1.0, 0.0]) and np.isnan(ops.nan_compare(a_arr, b_arr, "<")[0, 1])
    assert ops.nan_compare(a_arr, b_arr, "==")[0, 2] == 1.0
    assert ops.nan_logical(np.array([[1.0, 0.0]]), np.array([[0.0, 0.0]]), "||")[0, 0] == 1.0
    assert ops.nan_logical(np.array([[1.0, 0.0]]), np.array([[0.0, 0.0]]), "&&")[0, 0] == 0.0
    out_arr = ops.nan_where(np.array([[1.0, 0.0, np.nan]]), np.array([[5.0, 5.0, 5.0]]), np.array([[7.0, 7.0, 7.0]]))
    assert out_arr[0, 0] == 5.0 and out_arr[0, 1] == 7.0 and np.isnan(out_arr[0, 2])
    assert np.isnan(ops.clean(np.array([np.inf, -np.inf, 1.0]))[:2]).all()
    assert np.isnan(ops.log(np.array([[0.0, -1.0]]))).all()


# ----------------------------------------------------------------------------------------------------------------------
# evaluator: overloads, degrees and the PIT conversion on a synthetic 2:1 split
# ----------------------------------------------------------------------------------------------------------------------
def synthetic_context(split_bool: bool) -> Context:
    t_int, s_int = 80, 6
    rng = np.random.default_rng(7)
    close_arr = np.exp(rng.normal(0, 0.01, (t_int, s_int)).cumsum(axis=0)) * 40.0
    open_arr = close_arr * (1 + rng.normal(0, 0.003, (t_int, s_int)))
    high_arr = np.maximum(open_arr, close_arr) * 1.01
    low_arr = np.minimum(open_arr, close_arr) * 0.99
    volume_arr = rng.lognormal(14, 0.3, (t_int, s_int))
    unadjusted_arr = close_arr.copy()
    k_arr = np.ones((t_int, s_int))
    if split_bool:
        # stock 0 splits 2:1 at row 50: the back-adjusted history before row 50 is half the traded price, and the
        # adjusted volume before the split is twice the traded volume
        k_arr[:50, 0] = 2.0
        unadjusted_arr[:, 0] = close_arr[:, 0] * k_arr[:, 0]
    with np.errstate(all="ignore"):
        returns_arr = np.full_like(close_arr, np.nan)
        returns_arr[1:] = close_arr[1:] / close_arr[:-1] - 1.0
    vwap_arr = (high_arr + low_arr + close_arr) / 3.0
    member_arr = np.ones((t_int, s_int), dtype=bool)
    group_dict = {"sector": np.array([0, 0, 1, 1, 2, 2]), "industry": np.arange(6), "subindustry": np.arange(6)}
    panel_dict = {"open": open_arr, "high": high_arr, "low": low_arr, "close": close_arr, "volume": volume_arr, "vwap": vwap_arr, "returns": returns_arr}
    return Context(panel_dict, member_arr, k_arr, group_dict)


def test_min_max_overloads_and_degrees():
    context = synthetic_context(False)
    ts_arr = evaluate_formula("min(close, 3)", context)
    assert_same(ts_arr, ops.ts_min(context.panel_dict["close"], 3) * 1.0)
    el_arr = evaluate_formula("min(close, open)", context)
    assert_same(el_arr, np.minimum(context.panel_dict["close"], context.panel_dict["open"]))
    _, degree_tuple = evaluate_formula_with_degree("rank(close) * volume / adv20 * (high - close)", context)
    assert degree_tuple == (1.0, 0.0)
    _, degree_tuple = evaluate_formula_with_degree("(close^5) / (open^5) * vwap * volume", context)
    assert degree_tuple == (1.0, 1.0)
    _, degree_tuple = evaluate_formula_with_degree("correlation(close, volume, 5) + sign(delta(close, 2))", context)
    assert degree_tuple == (0.0, 0.0)
    _, degree_tuple = evaluate_formula_with_degree("covariance(close, volume, 5)", context)
    assert degree_tuple == (1.0, 1.0)
    _, degree_tuple = evaluate_formula_with_degree("product(rank(close), 3)", context)
    assert degree_tuple == (0.0, 0.0)
    _, degree_tuple = evaluate_formula_with_degree("(close - open) / ((high - low) + .001)", context)
    assert degree_tuple == (1.0, 0.0)  # denominator converted (constant added), numerator still priced
    _, degree_tuple = evaluate_formula_with_degree("stddev(returns, 5) + close", context)
    assert degree_tuple == (0.0, 0.0)  # non-homogeneous sum converts


def test_ternary_and_booleans_through_evaluator():
    context = synthetic_context(False)
    out_arr = evaluate_formula("(returns < 0) ? 1 : (-1 * 1)", context)
    ret_arr = context.panel_dict["returns"]
    assert np.all(out_arr[1:][ret_arr[1:] < 0] == 1.0) and np.all(out_arr[1:][ret_arr[1:] >= 0] == -1.0) and np.isnan(out_arr[0]).all()
    out_arr = evaluate_formula("((1 < (volume / adv20)) || ((volume / adv20) == 1)) ? 1 : 0", context)
    ratio_arr = context.panel_dict["volume"] / ops.ts_sum(context.panel_dict["volume"], 20) * 20
    assert np.all(out_arr[19:] == (ratio_arr[19:] >= 1.0).astype(float))


def test_pit_conversion_on_synthetic_split():
    """Before the split date the back-adjusted history of stock 0 is half the traded price: every converted value
    must equal what a trader saw (traded prices), and every raw-price rank must use traded prices."""
    plain_ctx = synthetic_context(False)
    split_ctx = synthetic_context(True)
    close_plain = plain_ctx.panel_dict["close"].copy()
    # build the split panel from the same traded prices: adjusted close before row 50 is traded / 2
    for key_str in ("open", "high", "low", "close", "vwap"):
        split_ctx.panel_dict[key_str][:50, 0] = plain_ctx.panel_dict[key_str][:50, 0] / 2.0
    split_ctx.panel_dict["volume"][:50, 0] = plain_ctx.panel_dict["volume"][:50, 0] * 2.0
    split_ctx.k_arr = plain_ctx.panel_dict["close"] / split_ctx.panel_dict["close"]
    split_ctx.kv_arr = 1.0 / split_ctx.k_arr
    with np.errstate(all="ignore"):
        split_ctx.panel_dict["returns"][1:] = split_ctx.panel_dict["close"][1:] / split_ctx.panel_dict["close"][:-1] - 1.0
    # (1) a converted raw price equals the traded price (plain context is the traded-price world with k = 1)
    conv_arr = evaluate_formula("close", split_ctx)
    assert_same(conv_arr, close_plain)
    # (2) rank(close) with the conversion equals the rank of traded prices, not of back-adjusted prices
    assert_same(evaluate_formula("rank(close)", split_ctx), evaluate_formula("rank(close)", plain_ctx))
    # (3) a constant added to a price converts first: close - 1 in traded dollars
    assert_same(evaluate_formula("close - 1", split_ctx), close_plain - 1.0)
    # (4) volume-degree conversion: log(volume) uses traded volume
    assert_same(evaluate_formula("log(volume)", split_ctx), np.log(plain_ctx.panel_dict["volume"]))
    # (5) a scale-free quantity is unchanged by the split and needs no conversion: returns, rank(close / open)
    assert_same(evaluate_formula("rank(close / open)", split_ctx), evaluate_formula("rank(close / open)", plain_ctx))
    # (6) a homogeneous window across the split stays in adjusted units and is converted once: sum(close, 5) in
    #     day-t units = adjusted sum x k_t (no jump for the plain world where k = 1 and no split happened)
    conv_sum = evaluate_formula("sum(close, 5)", split_ctx)
    expected_sum = ops.ts_sum(split_ctx.panel_dict["close"], 5) * split_ctx.k_arr
    assert_same(conv_sum, expected_sum)
    # (7) degree-2 quantity: rank of (close x high) converts with k^2
    assert_same(evaluate_formula("rank(close * high)", split_ctx), evaluate_formula("rank(close * high)", plain_ctx))


def test_download_date_invariance_synthetic():
    """A download made on an earlier date shows prices x k_T0 and volume x kv_T0 (per stock); all alphas must agree
    with the full-history values on the common sessions."""
    full_ctx = synthetic_context(True)
    t0_int = 60
    trunc_ctx = synthetic_context(True)
    scale_vec = full_ctx.k_arr[t0_int]
    for key_str in ("open", "high", "low", "close", "vwap"):
        trunc_ctx.panel_dict[key_str] = (full_ctx.panel_dict[key_str] * scale_vec[None, :])[: t0_int + 1]
    trunc_ctx.panel_dict["volume"] = (full_ctx.panel_dict["volume"] / scale_vec[None, :])[: t0_int + 1]
    trunc_ctx.panel_dict["returns"] = full_ctx.panel_dict["returns"][: t0_int + 1]
    trunc_ctx.k_arr = (full_ctx.k_arr / scale_vec[None, :])[: t0_int + 1]
    trunc_ctx.kv_arr = 1.0 / trunc_ctx.k_arr
    trunc_ctx.member_arr = full_ctx.member_arr[: t0_int + 1]
    trunc_ctx.shape = trunc_ctx.panel_dict["close"].shape
    for alpha_int in (1, 2, 5, 12, 25, 29, 41, 48, 54, 57, 84, 101):
        full_arr = evaluate_formula(formulas.FORMULA_DICT[alpha_int], full_ctx)[: t0_int + 1]
        trunc_arr = evaluate_formula(formulas.FORMULA_DICT[alpha_int], trunc_ctx)
        assert_same(full_arr, trunc_arr, 1e-9)


def test_random_rescaling_invariance_synthetic():
    base_ctx = synthetic_context(True)
    scaled_ctx = synthetic_context(True)
    c_vec = np.exp(np.random.default_rng(3).uniform(np.log(0.1), np.log(10.0), 6))
    for key_str in ("open", "high", "low", "close", "vwap"):
        scaled_ctx.panel_dict[key_str] = base_ctx.panel_dict[key_str] * c_vec[None, :]
    scaled_ctx.panel_dict["volume"] = base_ctx.panel_dict["volume"] / c_vec[None, :]
    scaled_ctx.k_arr = base_ctx.k_arr / c_vec[None, :]
    scaled_ctx.kv_arr = 1.0 / scaled_ctx.k_arr
    for alpha_int in (3, 6, 11, 18, 21, 28, 32, 33, 47, 52, 60, 83, 100):
        assert_same(evaluate_formula(formulas.FORMULA_DICT[alpha_int], base_ctx), evaluate_formula(formulas.FORMULA_DICT[alpha_int], scaled_ctx), 1e-9)


def test_all_formulas_evaluate_on_synthetic_data():
    context = synthetic_context(True)
    for alpha_int in formulas.ALPHA_NUMBER_TUPLE:
        out_arr = evaluate_formula(formulas.FORMULA_DICT[alpha_int], context)
        assert out_arr.shape == context.shape
        assert not (~np.isfinite(out_arr) & ~np.isnan(out_arr)).any()
    with pytest.raises(ValueError):
        evaluate_formula(formulas.FORMULA_DICT[56], context)


# ----------------------------------------------------------------------------------------------------------------------
# composites, the screen and the pod's buffer rule
# ----------------------------------------------------------------------------------------------------------------------
def test_weighted_composite_and_wf_weights():
    from alpha101_20260928 import composites

    z_arr = np.array([[[0.1, np.nan, 0.3]], [[0.2, 0.2, np.nan]], [[np.nan, np.nan, 0.4]]])  # 3 alphas x 1 row x 3 stocks
    eq_arr = composites.weighted_composite(z_arr, np.array([1 / 3, 1 / 3, 1 / 3]), 0.5)
    assert eq_arr[0, 0] == pytest.approx(0.15) and np.isnan(eq_arr[0, 1]) and eq_arr[0, 2] == pytest.approx(0.35)
    w_arr = composites.weighted_composite(z_arr, np.array([0.6, 0.4, 0.0]), 0.5)
    assert w_arr[0, 0] == pytest.approx((0.6 * 0.1 + 0.4 * 0.2) / 1.0) and np.isnan(w_arr[0, 1]) and w_arr[0, 2] == pytest.approx(0.3)
    date_index = pd.bdate_range("2000-01-03", "2004-12-31")
    ic_df = pd.DataFrame({"1": 0.02, "2": -0.01, "3": 0.01}, index=date_index)
    ic_df.loc["2003":, "3"] = -0.05  # only affects the 2005 weights (data through 2004)
    weights_dict = composites.wf_weights_from_ic(ic_df, date_index, [2001, 2003, 2005])
    assert weights_dict["2001"]["mode"] == "C_EQ" and weights_dict["2001"]["weights"]["2"] == pytest.approx(1 / 3)
    assert weights_dict["2003"]["mode"] == "WF" and weights_dict["2003"]["weights"] == pytest.approx({"1": 2 / 3, "2": 0.0, "3": 1 / 3})
    cutoff_ts = pd.Timestamp(weights_dict["2003"]["cutoff_decision_ts"])
    assert cutoff_ts == date_index[date_index.searchsorted(pd.Timestamp("2002-12-31"), side="right") - 3]  # fr of t known by t+2
    assert weights_dict["2005"]["weights"]["3"] == 0.0 and weights_dict["2005"]["mode"] == "WF"
    all_negative_df = pd.DataFrame({"1": -0.02, "2": -0.01}, index=date_index)
    assert composites.wf_weights_from_ic(all_negative_df, date_index, [2004])["2004"]["mode"] == "C_EQ"


def test_daily_screen_deciles_and_turnover():
    from alpha101_20260928 import stage_a

    t_int, s_int = 4, 20
    rng = np.random.default_rng(1)
    score_arr = np.tile(np.arange(s_int, dtype=float), (t_int, 1))  # constant ranking: stock 19 best
    fr_arr = rng.normal(0, 0.01, (t_int, s_int))
    member_arr = np.ones((t_int, s_int), dtype=bool)
    daily_df = stage_a.daily_screen(score_arr, fr_arr, member_arr, stage_a.tie_key_vec(s_int))
    assert daily_df["n"].tolist() == [20.0] * t_int
    assert daily_df["ls"].iloc[0] == pytest.approx(fr_arr[0, 18:].mean() - fr_arr[0, :2].mean())
    assert daily_df["lo"].iloc[0] == pytest.approx(fr_arr[0, 18:].mean() - fr_arr[0].mean())
    assert np.isnan(daily_df["tau"].iloc[0]) and daily_df["tau"].iloc[1:].tolist() == [0.0] * (t_int - 1)  # no turnover
    # a full reversal of the ranking turns the whole book over: 4 dollars traded per 2 dollars of gross -> tau = 2;
    # the long-only top decile sells 1 and buys 1 per dollar of gross -> tau_lo = 2
    score_arr[2:] = -score_arr[2:]
    daily_df = stage_a.daily_screen(score_arr, fr_arr, member_arr, stage_a.tie_key_vec(s_int))
    assert daily_df["tau"].iloc[2] == pytest.approx(2.0) and daily_df["tau_lo"].iloc[2] == pytest.approx(2.0)
    # replacing one of the two top names and one of the two bottom names: 4 x 0.5 dollars traded / 2 -> tau = 1
    score_arr[3] = score_arr[2].copy()
    score_arr[3, [0, 19]] = [-5.0, -10.0]  # stock 0 leaves the top (was best after the flip), 19 leaves the bottom
    daily_df = stage_a.daily_screen(score_arr, fr_arr, member_arr, stage_a.tie_key_vec(s_int))
    assert daily_df["tau"].iloc[3] == pytest.approx(1.0)
    # ties: all scores equal -> the seeded symbol order decides, deciles still hold 2 names
    tie_df = stage_a.daily_screen(np.zeros((t_int, s_int)), fr_arr, member_arr, stage_a.tie_key_vec(s_int))
    assert np.isfinite(tie_df["ls"]).all()
    # IC of a score equal to fr is 1
    assert stage_a.daily_ic(fr_arr, fr_arr, member_arr).tolist() == pytest.approx([1.0] * t_int)


def test_block_stats_break_even_cost():
    from alpha101_20260928 import stage_a

    index = pd.bdate_range("2010-01-04", periods=300)
    daily_df = pd.DataFrame({"ic": 0.02, "n": 500.0, "ls": 0.001, "tau": 0.5, "lo": 0.0005, "tau_lo": 0.5, "ic_overnight": 0.0}, index=index)
    out_dict = stage_a.block_stats(daily_df, "2010-01-01", "2011-12-31")
    assert out_dict["c_star_bps"] == pytest.approx(0.0005 / 0.5 * 1e4)  # 10 bps per side
    assert out_dict["net_mean_ann_3bps"] == pytest.approx((0.001 - 3e-4 * 2 * 0.5) * 252)
    assert out_dict["lo_c_star_bps"] == pytest.approx(0.0005 / 0.5 * 1e4)


class _FakeFeatures:
    def __init__(self, close_arr, unadj_arr, member_arr, symbol_list, month_end_index=None):
        self.close_arr = close_arr
        self.u = {"member_arr": member_arr, "month_end_index": month_end_index if month_end_index is not None else pd.DatetimeIndex([])}
        self.symbol_rank_vec = np.argsort(np.argsort(np.array(symbol_list, dtype=object)))
        self.date_index = pd.bdate_range("2020-01-01", periods=close_arr.shape[0])
        self._unadj = unadj_arr

    def unadjusted_close(self):
        return self._unadj


def test_policy_buffer_rule():
    from alpha101_20260928.policies import PolicyAlpha
    from trend_breakout_20260927.policies import State

    t_int, s_int = 3, 8
    close_arr = np.full((t_int, s_int), 10.0)
    member_arr = np.ones((t_int, s_int), dtype=np.int8)
    score_arr = np.tile(np.array([8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]), (t_int, 1))
    features = _FakeFeatures(close_arr, close_arr.copy(), member_arr, [f"S{i}" for i in range(s_int)])
    policy = PolicyAlpha(features, score_arr, n_int=2, b_int=2)
    state = State(s_int, 1000.0)
    intents = policy.decide(0, state)
    assert [(i.symbol_idx, i.kind_str) for i in intents] == [(0, "value"), (1, "value")]
    assert intents[0].amount_float == pytest.approx(500.0)  # min(V/N, cash / free slots)
    # hold 0 and 1; stock 1 drops to rank 4 = B x N -> kept; stock 0 drops to rank 5 -> exit and refill with the best free name
    state.shares_vec[0] = 50.0
    state.shares_vec[1] = 50.0
    state.cash_float = 0.0
    state.total_value_float = 1000.0
    score_arr[1] = np.array([3.5, 4.5, 8.0, 7.0, 6.0, 5.0, 2.0, 1.0])  # ranks: 2->1, 3->2, 4->3, 5->4, 1->5?  (4.5 is rank 5)
    score_arr[1] = np.array([3.5, 5.5, 8.0, 7.0, 6.0, 4.0, 2.0, 1.0])  # ranks: 2:1, 3:2, 4:3, 1:4 (kept), 5:5, 0:6 (exit)
    intents = policy.decide(1, state)
    kinds = [(i.symbol_idx, i.kind_str, i.reason_str) for i in intents]
    assert (0, "exit", "buffer") in kinds
    entries = [i for i in intents if i.kind_str == "value"]
    assert len(entries) == 1 and entries[0].symbol_idx == 2  # best-ranked free name
    assert entries[0].amount_float == pytest.approx(min(1000.0 / 2, (0.0 + 50.0 * 10.0) / 1))  # exits' value funds the entry
    # a held name that leaves the universe exits with reason 'member'; a missing score exits with 'missing'
    member_arr[2, 1] = 0
    score_arr[2, 2] = np.nan
    state.shares_vec[:] = 0.0
    state.shares_vec[1] = 50.0
    state.shares_vec[2] = 50.0
    intents = policy.decide(2, state)
    reasons = {i.symbol_idx: i.reason_str for i in intents if i.kind_str == "exit"}
    assert reasons == {1: "member", 2: "missing"}


@pytest.mark.integration
def test_cbil_reproduces_section0_table():
    from alpha101_20260928 import checks, common

    if not common.SLEEVE_SERIES_PATH.exists():
        pytest.skip("sleeve series not available")
    report = checks.check_cbil()
    assert report["passed_bool"]


def test_tolerant_ranks_and_snap_sum():
    # two stocks with the same traded volume 509800, one with a 3:1 factor: the two computation paths differ at the
    # last digit but must tie; a third stock 1e-4 apart must not
    volume_a_full, k_a_full = 1529400.0, 3.0000000000000004
    conv_full = volume_a_full * (1.0 / k_a_full)
    conv_trunc = (volume_a_full / 3.0) * (1.0 / (k_a_full / 3.0))
    for conv in (conv_full, conv_trunc):
        rank_arr = ops.tolerant_ranks(np.array([[conv, 509800.0, 509800.0 * (1 + 1e-4), np.nan, 1.0]]))
        assert rank_arr[0].tolist()[:3] == [2.5, 2.5, 4.0] and np.isnan(rank_arr[0, 3]) and rank_arr[0, 4] == 1.0
    # tolerant ranks agree with scipy's average ranks when no near-ties exist
    x_arr = np.array([[3.0, 1.0, 2.0, 2.0, np.nan, 9.0], [5.0, 4.0, 3.0, 2.0, 1.0, np.nan]])
    from scipy import stats as sps

    assert_same(ops.tolerant_ranks(x_arr), sps.rankdata(x_arr, axis=1, nan_policy="omit"))
    # a difference that is zero in traded prices but 3e-8 after back-adjustment is zero
    assert ops.snap_sum(np.array([-3e-8]), np.array([0.45000001]), np.array([0.45]))[0] == 0.0
    assert ops.snap_sum(np.array([0.01]), np.array([0.46]), np.array([0.45]))[0] == 0.01


def test_constant_window_and_comparison_tolerance():
    # rolling sums of the same five closes added in a different order differ by one ulp: the 2-day correlation of such
    # sums must be NaN (constant input) on either computation path
    a_arr = np.array([[300.0], [300.0 + 300.0 * 2e-16]])
    b_arr = np.array([[1.0], [2.0]])
    assert np.isnan(ops.ts_correlation(a_arr, b_arr, 2)[1, 0]) and np.isnan(ops.ts_covariance(a_arr, b_arr, 2)[1, 0]) and np.isnan(ops.ts_stddev(a_arr, 2)[1, 0])
    c_arr = np.array([[300.0], [300.01]])
    assert ops.ts_correlation(c_arr, b_arr, 2)[1, 0] == pytest.approx(1.0)
    # (high + low) / 2 + close == low + open in traded prices, one ulp apart after back-adjustment: not '<' on any path
    x_arr = np.array([[60.300000000000004]])
    y_arr = np.array([[60.3]])
    assert ops.nan_compare(x_arr, y_arr, "<")[0, 0] == 0.0 and ops.nan_compare(x_arr, y_arr, "==")[0, 0] == 1.0
    assert ops.nan_compare(np.array([[60.31]]), y_arr, ">")[0, 0] == 1.0


def synthetic_universe(t_int: int = 320, s_int: int = 12, seed_int: int = 11) -> dict:
    rng = np.random.default_rng(seed_int)
    date_index = pd.bdate_range("2000-01-03", periods=t_int)
    close_arr = np.exp(rng.normal(0.0002, 0.015, (t_int, s_int)).cumsum(axis=0)) * 50.0
    open_arr = close_arr * (1 + rng.normal(0, 0.004, (t_int, s_int)))
    high_arr = np.maximum(open_arr, close_arr) * 1.01
    low_arr = np.minimum(open_arr, close_arr) * 0.99
    volume_arr = rng.lognormal(14, 0.3, (t_int, s_int))
    member_arr = np.ones((t_int, s_int), dtype=np.int8)
    member_arr[200:, 3] = 0  # stock 3 leaves the universe
    close_arr[250:, 5] = np.nan  # stock 5 delists (terminal liquidation)
    open_arr[250:, 5] = np.nan
    month_end_index = pd.DatetimeIndex(pd.Series(date_index, index=date_index.to_period("M")).groupby(level=0).max().to_numpy())
    return {
        "universe_str": "SYN", "date_index": date_index, "symbol_list": [f"S{i:02d}" for i in range(s_int)], "panel_dtype_str": "float64",
        "open_arr": open_arr, "high_arr": high_arr, "low_arr": low_arr, "close_arr": close_arr, "volume_arr": volume_arr,
        "turnover_arr": close_arr * volume_arr, "unadjusted_close_arr": close_arr.copy(), "dividend_arr": np.zeros((t_int, s_int)),
        "member_arr": member_arr, "month_end_index": month_end_index,
    }


def test_pod_end_to_end_synthetic():
    from alpha101_20260928.policies import PolicyAlpha
    from alpha101_20260928.simulate import run_summary, simulate
    from trend_breakout_20260927.features import DailyFeatureBook

    universe_dict = synthetic_universe()
    feature_obj = DailyFeatureBook(universe_dict)
    rng = np.random.default_rng(3)
    score_arr = rng.normal(0, 1, universe_dict["close_arr"].shape)
    score_arr[~np.isfinite(universe_dict["close_arr"])] = np.nan
    policy_obj = PolicyAlpha(feature_obj, score_arr, n_int=4, b_int=2)
    sim_dict = simulate(feature_obj, policy_obj, capital_float=50_000.0)
    assert np.isfinite(sim_dict["total_ser"]).all() and sim_dict["total_ser"].iloc[0] > 0
    assert sim_dict["position_count_ser"].max() <= 4
    assert (sim_dict["cash_weight_ser"] >= 0).all() and (sim_dict["cash_weight_ser"] <= 1.0 + 1e-9).all()
    trade_df = sim_dict["trade_df"]
    assert set(trade_df["reason"]).issubset({"buffer", "member", "missing", "terminal"})
    summary_dict = run_summary(sim_dict, feature_obj.date_index, end_ts=feature_obj.date_index[-1])
    assert summary_dict["turnover_one_way_x_per_year"] > 0 and summary_dict["mean_positions"] > 0
    assert summary_dict["capacity_full"]["orders_int"] > 0
    # the NAV identity: final NAV = capital + gross P&L - commissions - slippage
    assert summary_dict["net_pnl_usd"] == pytest.approx(summary_dict["gross_pnl_usd"] - sim_dict["commission_ser"].sum() - sim_dict["slippage_cost_ser"].sum())
    # HEDGED form on the same universe with SH appended as the last column
    hedged_dict = dict(universe_dict)
    sh_close_vec = 100.0 / np.exp(np.log(universe_dict["close_arr"][:, 0] / universe_dict["close_arr"][0, 0]))  # inverse of stock 0
    sh_close_vec[:30] = np.nan  # SH does not exist yet
    for key_str in ("open_arr", "high_arr", "low_arr", "close_arr", "unadjusted_close_arr"):
        hedged_dict[key_str] = np.column_stack([universe_dict[key_str], sh_close_vec])
    for key_str in ("volume_arr", "turnover_arr"):
        hedged_dict[key_str] = np.column_stack([universe_dict[key_str], np.full(len(sh_close_vec), 1e6)])
    hedged_dict["dividend_arr"] = np.column_stack([universe_dict["dividend_arr"], np.zeros(len(sh_close_vec))])
    hedged_dict["member_arr"] = np.column_stack([universe_dict["member_arr"], np.zeros(len(sh_close_vec), dtype=np.int8)])
    hedged_dict["symbol_list"] = universe_dict["symbol_list"] + ["SH"]
    hedged_features = DailyFeatureBook(hedged_dict)
    hedged_scores = np.column_stack([score_arr, np.full(score_arr.shape[0], np.nan)])
    hedged_policy = PolicyAlpha(hedged_features, hedged_scores, n_int=4, b_int=2, hedged_bool=True, sh_idx=12)
    hedged_sim = simulate(hedged_features, hedged_policy)
    assert np.isfinite(hedged_sim["total_ser"]).all()
    assert (hedged_sim["total_ser"].iloc[:25] == 100_000.0).all()  # inactive before SH exists
    sh_intents = hedged_sim["intent_df"][hedged_sim["intent_df"]["symbol_idx"] == 12]
    assert (sh_intents["kind"] == "target_pct").all() and (sh_intents["amount"] == 0.5).all() and len(sh_intents) >= 10
    assert hedged_sim["position_count_ser"].iloc[-1] <= 5 and hedged_sim["exposure_ser"].iloc[-1] > 0.6


def test_delta_snaps_path_noise_to_zero():
    x_arr = np.array([[50.0], [50.01], [50.02 + 4e-15]])  # second difference zero in traded prices, 4e-15 after scaling
    first_arr = ops.delta(x_arr, 1)
    assert first_arr[2, 0] == pytest.approx(0.01, rel=1e-9)
    assert ops.delta(first_arr, 1)[2, 0] == 0.0
    assert ops.delta(np.array([[50.0], [50.01]]), 1)[1, 0] == pytest.approx(0.01)


def test_indneutralize_snaps_cancellation_noise():
    # two members of one group at the same traded price, back-adjusted with different factors: the demeaned values are
    # zero on either computation path (not +-1e-16)
    x_arr = np.array([[60.3 * 3.0000000000000004 / 3.0, 60.3, 10.0]])
    member_arr = np.ones((1, 3), dtype=bool)
    out_arr = ops.cs_indneutralize(x_arr, member_arr, np.array([0, 0, 1]))
    assert out_arr[0, 0] == 0.0 and out_arr[0, 1] == 0.0 and out_arr[0, 2] == 0.0
    assert np.allclose(ops.cs_indneutralize(np.array([[1.0, 3.0, 10.0]]), member_arr, np.array([0, 0, 1]))[0], [-1.0, 1.0, 0.0])


def test_window_sums_snap_exact_cancellation():
    # decay_linear over 2 days with weights 1/3, 2/3: deltas +0.50 and -0.25 cancel exactly in traded prices; after a
    # per-stock rescaling the float result is 1e-17-ish on one path and must be zero on both
    for c in (1.0, 0.113):
        w_arr = np.array([[0.5 * c], [-0.25 * c]])
        assert ops.decay_linear(w_arr, 2)[1, 0] == 0.0
    assert ops.decay_linear(np.array([[0.5], [-0.2]]), 2)[1, 0] == pytest.approx(0.5 / 3 - 0.4 / 3)
    assert ops.ts_sum(np.array([[0.7], [-0.7 * (1 + 1e-16)]]), 2)[1, 0] == 0.0
    assert ops.ts_sum(np.array([[0.7], [-0.6]]), 2)[1, 0] == pytest.approx(0.1)
