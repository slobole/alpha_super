"""Strategy readiness audit 2026-09-28, DV2 + QPI: factories, recording subclasses and helpers (research-only).

The factories reproduce the production run paths exactly:
- DV2: strategies.dv2.strategy_mr_dv2.run_variant  (DVO2Strategy, $SPX TR benchmark, 2.5 bp, $0.005/sh, $1 min).
- QPI: strategies.qpi.strategy_mr_qpi_ibs_rsi_exit.run_variant (QPIIbsRsiExitStrategy, same costs, default params).

The Rec* subclasses only OBSERVE: before calling the production iterate() they snapshot the pod state that the live
host would be seeded with (positions, cash, previous_total_value, trade ids), and after it they snapshot the orders
the production iterate() created.  Behaviour is unchanged (verified: Rec run == plain run, see aud_full_runs.py).
"""

from __future__ import annotations

import contextlib
import io
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
for path in (REPO, HERE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from alpha.engine.backtest import run_daily  # noqa: E402
from data.norgate_loader import TOTALRETURN_ADJUSTMENT_STR  # noqa: E402
from strategies.dv2.strategy_mr_dv2 import DVO2Strategy, default_trade_id_int  # noqa: E402
from strategies.qpi.strategy_mr_qpi_ibs_rsi_exit import QPIIbsRsiExitStrategy  # noqa: E402
from strategies.qpi.strategy_mr_qpi_ibs_rsi_exit import default_trade_id_int as qpi_default_trade_id_int  # noqa: E402

OUT = REPO / "results" / "research" / "strategy_readiness_audit_20260928" / "mr_dv2_qpi"
OUT.mkdir(parents=True, exist_ok=True)
FULL_START = pd.Timestamp("2004-01-01")
STUDY_END = pd.Timestamp("2026-09-25")


def _order_rows(strategy) -> list[dict]:
    return [{"asset": str(o.asset), "unit": str(o.unit), "target": bool(o.target), "amount": float(o.amount),
             "cls": type(o).__name__} for o in strategy.get_orders()]


class _RecordMixin:
    record_bool = True

    def _rec_init(self):
        if not hasattr(self, "decision_log"):
            self.decision_log = []

    def iterate(self, data, close, open_prices):  # noqa: D401 - observer only
        self._rec_init()
        if not self.record_bool or data is None or close is None:
            return super().iterate(data, close, open_prices)
        positions = {str(k): float(v) for k, v in self.get_positions().items() if abs(float(v)) > 0}
        trade_map = dict(getattr(self, "current_trade_map", None) or getattr(self, "current_trade", {}) or {})
        pre = {"decision_date": pd.Timestamp(self.previous_bar), "fill_date": pd.Timestamp(self.current_bar),
               "positions": positions, "cash": float(self.cash), "prev_total_value": float(self.previous_total_value),
               "trade_id": int(getattr(self, "trade_id_int", getattr(self, "trade_id", 0))),
               "current_trade_map": {str(k): int(v) for k, v in trade_map.items()},
               "n_orders_before": len(self.get_orders())}
        super().iterate(data, close, open_prices)
        pre["orders"] = _order_rows(self)
        self.decision_log.append(pre)


class RecDVO2(_RecordMixin, DVO2Strategy):
    pass


class RecQPI(_RecordMixin, QPIIbsRsiExitStrategy):
    pass


def make_dv2(universe_df: pd.DataFrame, capital: float = 100_000.0, record: bool = False, cls=None):
    cls = cls or (RecDVO2 if record else DVO2Strategy)
    strategy = cls(name="strategy_mr_dv2", benchmarks=["$SPX"], capital_base=capital, slippage=0.00025,
                   commission_per_share=0.005, commission_minimum=1.0,
                   performance_benchmark_adjustment_str=TOTALRETURN_ADJUSTMENT_STR)
    strategy.universe_df = universe_df
    strategy.trade_id = 0
    strategy.current_trade = defaultdict(default_trade_id_int)
    return strategy


def make_qpi(universe_df: pd.DataFrame, capital: float = 100_000.0, record: bool = False, cls=None):
    cls = cls or (RecQPI if record else QPIIbsRsiExitStrategy)
    strategy = cls(name="strategy_mr_qpi_ibs_rsi_exit", benchmarks=["$SPX"], capital_base=capital, slippage=0.00025,
                   commission_per_share=0.005, commission_minimum=1.0)
    strategy.universe_df = universe_df
    strategy.trade_id_int = 0
    strategy.current_trade_map = defaultdict(qpi_default_trade_id_int)
    return strategy


def make(family: str, universe_df, capital: float = 100_000.0, record: bool = False, cls=None):
    return (make_dv2 if family == "dv2" else make_qpi)(universe_df, capital=capital, record=record, cls=cls)


def run(strategy, pricing_df: pd.DataFrame, start, end, quiet: bool = True):
    frame = pricing_df.loc[: pd.Timestamp(end)]
    calendar = frame.index[frame.index >= pd.Timestamp(start)]
    sink = io.StringIO()
    ctx = contextlib.redirect_stdout(sink) if quiet else contextlib.nullcontext()
    with ctx:
        run_daily(strategy, frame, calendar, show_progress=False, show_signal_progress_bool=False)
    strategy._captured_stdout = sink.getvalue() if quiet else ""
    return strategy


def metrics(total_value: pd.Series, start=None, end=None) -> dict:
    nav = total_value.astype(float)
    if start is not None:
        nav = nav.loc[pd.Timestamp(start):]
    if end is not None:
        nav = nav.loc[: pd.Timestamp(end)]
    ret = nav.pct_change().dropna()
    years = (nav.index[-1] - nav.index[0]).days / 365.25
    cagr = (nav.iloc[-1] / nav.iloc[0]) ** (1 / years) - 1
    sharpe = ret.mean() / ret.std() * np.sqrt(252)
    maxdd = (nav / nav.cummax() - 1).min()
    return {"start": nav.index[0].date().isoformat(), "end": nav.index[-1].date().isoformat(), "cagr": float(cagr),
            "sharpe": float(sharpe), "max_dd": float(maxdd), "vol": float(ret.std() * np.sqrt(252))}


def subset_pricing(pricing_df: pd.DataFrame, universe_df: pd.DataFrame, start, end, extra=(), bench=("$SPX",)):
    """Symbols that are PIT members at any time in [start - 20 days, end] plus extras and the benchmark.

    Decision-neutral for DV2/QPI: every cross-sectional step (filters, NATR / Turnover ranking) runs only over
    names that pass the PIT membership filter at T, and held names are always members of the window.
    Used only for the many short invariance/prefix runs.  Full-history runs use the full frame.
    """
    window = universe_df.loc[pd.Timestamp(start) - pd.Timedelta(days=20): pd.Timestamp(end)]
    members = set(window.columns[window.sum(axis=0) > 0].astype(str)) | set(extra) | set(bench)
    keep = [c for c in pricing_df.columns if str(c[0]) in members]
    out = pricing_df.loc[:, keep].copy()
    out.attrs.update(pricing_df.attrs)
    return out


def decisions(strategy) -> pd.DataFrame:
    """Engine fills -> one row per (fill date, asset) with side and dollar notional (share units dropped)."""
    tx = strategy.get_transactions()
    if len(tx) == 0:
        return pd.DataFrame(columns=["date", "asset", "side", "notional"])
    tx = tx.copy()
    tx["notional"] = tx["amount"].astype(float) * tx["price"].astype(float)
    tx["side"] = np.sign(tx["amount"].astype(float)).astype(int)
    grouped = tx.groupby(["bar", "asset"]).agg(side=("side", "first"), notional=("notional", "sum"))
    return grouped.reset_index().rename(columns={"bar": "date"})


def order_decisions(strategy) -> pd.DataFrame:
    """Recorded ORDER intents (not fills): one row per (decision date, asset, kind, value).  Includes orders the
    engine later cancelled (zero shares, missing open)."""
    rows = []
    for rec in getattr(strategy, "decision_log", []):
        for o in rec["orders"]:
            kind = "exit" if (o["target"] and abs(o["amount"]) <= 1e-9) else "entry"
            rows.append({"date": rec["decision_date"], "asset": o["asset"], "kind": kind,
                         "value": o["amount"] if kind == "entry" else 0.0,
                         "weight": (o["amount"] / rec["prev_total_value"]) if kind == "entry" else 0.0})
    return pd.DataFrame(rows, columns=["date", "asset", "kind", "value", "weight"])


def compare_decisions(reference: pd.DataFrame, candidate: pd.DataFrame, end_ts=None, rel_tol: float = 0.02,
                      key=("date", "asset", "side"), value_col: str = "notional") -> dict:
    ref, cand = reference.copy(), candidate.copy()
    if end_ts is not None:
        ref = ref[ref["date"] <= pd.Timestamp(end_ts)]
        cand = cand[cand["date"] <= pd.Timestamp(end_ts)]
    key = list(key)
    merged = ref.merge(cand, on=key, how="outer", suffixes=("_ref", "_cand"), indicator=True)
    only_ref = merged[merged["_merge"] == "left_only"]
    only_cand = merged[merged["_merge"] == "right_only"]
    both = merged[merged["_merge"] == "both"].copy()
    if len(both):
        both["rel_diff"] = ((both[f"{value_col}_cand"] - both[f"{value_col}_ref"]).abs()
                            / both[f"{value_col}_ref"].abs().clip(lower=1e-12))
    first = None
    if len(only_ref) or len(only_cand):
        first = pd.concat([only_ref["date"], only_cand["date"]]).min()
    return {
        "n_reference": int(len(ref)), "n_candidate": int(len(cand)),
        "n_only_reference": int(len(only_ref)), "n_only_candidate": int(len(only_cand)),
        "first_divergence_date": None if first is None else pd.Timestamp(first).date().isoformat(),
        "max_rel_value_diff": float(both["rel_diff"].max()) if len(both) else 0.0,
        "n_value_beyond_tol": int((both["rel_diff"] > rel_tol).sum()) if len(both) else 0,
        "examples_only_reference": only_ref[key].head(5).astype(str).to_dict("records"),
        "examples_only_candidate": only_cand[key].head(5).astype(str).to_dict("records"),
    }


def rescale_symbol_history(pricing_df: pd.DataFrame, symbol_str: str, factor_float: float) -> pd.DataFrame:
    """Protocol A2: one symbol's full history re-based as if a k:1 split happened after the last date.
    OHLC and Dividend / k, Volume * k, 'Unadjusted Close' and 'Turnover' nominal (unchanged)."""
    out_df = pricing_df.copy()
    for field_str in ("Open", "High", "Low", "Close", "Dividend"):
        if (symbol_str, field_str) in out_df.columns:
            out_df[(symbol_str, field_str)] = out_df[(symbol_str, field_str)] / factor_float
    if (symbol_str, "Volume") in out_df.columns:
        out_df[(symbol_str, "Volume")] = out_df[(symbol_str, "Volume")] * factor_float
    out_df.attrs.update(pricing_df.attrs)
    return out_df
