"""Tier B audit helpers for Industry-ETF DV2 (strategies/dv2/strategy_mr_dv2_industry_etf.py). Research-only.

Factories reproduce the production path of ``run_variant`` exactly: ``_load(end)`` (19 ETFs + $SPX from 2009-01-01,
CAPITALSPECIAL, ALLMARKETDAYS padding), ``build_history_universe_df``, ``_new_strategy_obj``, and a calendar starting
at DEFAULT_BACKTEST_START_DATE_STR. Recording subclasses only observe (orders created by the production iterate()).
Adapted from ``mr_dv2_qpi/aud_common.py`` (copied, not imported, so the two audits stay independent).
"""

from __future__ import annotations

import contextlib
import io
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
for _p in (REPO, HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from alpha.engine.backtest import run_daily  # noqa: E402
from data.norgate_loader import TOTALRETURN_ADJUSTMENT_STR  # noqa: E402
from strategies.dv2 import strategy_mr_dv2_industry_etf as etf_module  # noqa: E402
from strategies.dv2.strategy_mr_dv2_liquidity_floor import default_trade_id_int  # noqa: E402

OUT = REPO / "results" / "research" / "strategy_readiness_audit_20260928" / "tierbc_dv2etf_taa2x" / "etf"
OUT.mkdir(parents=True, exist_ok=True)
STUDY_END = "2026-09-25"
START = etf_module.DEFAULT_BACKTEST_START_DATE_STR
CACHE_FILE = OUT / "etf_inputs_full.pkl"


def load_full(end_date_str: str | None = STUDY_END, history_start_str: str | None = None):
    """Production loader (optionally with an earlier history start, only for the pre-2012 splice study)."""
    if history_start_str is None:
        return etf_module._load(end_date_str)
    pricing_df = etf_module.get_prices(list(etf_module.INDUSTRY_ETF_SYMBOL_TUPLE), ["$SPX"],
                                       start_date=history_start_str, end_date=end_date_str)
    return pricing_df, etf_module.build_history_universe_df(pricing_df)


def load_cached():
    if CACHE_FILE.exists():
        with CACHE_FILE.open("rb") as fh:
            return pickle.load(fh)
    pricing_df, universe_df = load_full()
    with CACHE_FILE.open("wb") as fh:
        pickle.dump((pricing_df, universe_df), fh)
    return pricing_df, universe_df


def _order_rows(strategy) -> list[dict]:
    return [{"asset": str(o.asset), "unit": str(o.unit), "target": bool(o.target), "amount": float(o.amount)}
            for o in strategy.get_orders()]


class RecEtf(etf_module.DVO2IndustryEtfStrategy):
    """Observer: records the orders created by the production iterate() at each decision T."""

    def iterate(self, data, close, open_prices):
        if not hasattr(self, "decision_log"):
            self.decision_log = []
        if data is None or close is None:
            return super().iterate(data, close, open_prices)
        pre = {"decision_date": pd.Timestamp(self.previous_bar), "prev_total_value": float(self.previous_total_value),
               "n_before": len(self.get_orders())}
        super().iterate(data, close, open_prices)
        pre["orders"] = _order_rows(self)  # same convention as mr_dv2_qpi/aud_common.py (queue is empty before)
        self.decision_log.append(pre)


def make(universe_df: pd.DataFrame, capital: float = 100_000.0, cls=RecEtf, hsu: bool = False):
    strategy = cls(name=etf_module.STRATEGY_NAME_STR, benchmarks=["$SPX"], capital_base=capital, slippage=0.00025,
                   commission_per_share=0.005, commission_minimum=1.0,
                   performance_benchmark_adjustment_str=TOTALRETURN_ADJUSTMENT_STR)
    strategy.universe_df = universe_df
    strategy.trade_id = 0
    strategy.current_trade = defaultdict(default_trade_id_int)
    strategy.historical_share_units_bool = hsu
    return strategy


def run(strategy, pricing_df: pd.DataFrame, start=START, end=STUDY_END, quiet: bool = True):
    frame = pricing_df.loc[: pd.Timestamp(end)]
    calendar = frame.index[frame.index >= pd.Timestamp(start)]
    sink = io.StringIO()
    with (contextlib.redirect_stdout(sink) if quiet else contextlib.nullcontext()):
        run_daily(strategy, frame, calendar, show_progress=False, show_signal_progress_bool=False)
    return strategy


def metrics(total_value: pd.Series, start=None, end=None) -> dict:
    nav = total_value.astype(float)
    nav.index = pd.to_datetime(nav.index)
    if start is not None:
        nav = nav.loc[pd.Timestamp(start):]
    if end is not None:
        nav = nav.loc[: pd.Timestamp(end)]
    ret = nav.pct_change().dropna()
    years = (nav.index[-1] - nav.index[0]).days / 365.25
    return {"start": nav.index[0].date().isoformat(), "end": nav.index[-1].date().isoformat(),
            "cagr": float((nav.iloc[-1] / nav.iloc[0]) ** (1 / years) - 1),
            "sharpe": float(ret.mean() / ret.std() * np.sqrt(252)),
            "max_dd": float((nav / nav.cummax() - 1).min()), "vol": float(ret.std() * np.sqrt(252))}


def fills(strategy) -> pd.DataFrame:
    tx = strategy.get_transactions()
    if len(tx) == 0:
        return pd.DataFrame(columns=["date", "asset", "side", "notional"])
    tx = tx.copy()
    tx["notional"] = tx["amount"].astype(float) * tx["price"].astype(float)
    tx["side"] = np.sign(tx["amount"].astype(float)).astype(int)
    grouped = tx.groupby(["bar", "asset"]).agg(side=("side", "first"), notional=("notional", "sum"))
    return grouped.reset_index().rename(columns={"bar": "date"})


def order_intents(strategy) -> pd.DataFrame:
    rows = []
    for rec in getattr(strategy, "decision_log", []):
        for o in rec["orders"]:
            kind = "exit" if (o["target"] and abs(o["amount"]) <= 1e-9) else "entry"
            rows.append({"date": rec["decision_date"], "asset": o["asset"], "kind": kind,
                         "weight": (o["amount"] / rec["prev_total_value"]) if kind == "entry" else 0.0})
    return pd.DataFrame(rows, columns=["date", "asset", "kind", "weight"])


def compare(reference: pd.DataFrame, candidate: pd.DataFrame, end_ts=None, rel_tol: float = 0.02,
            key=("date", "asset", "side"), value_col: str = "notional") -> dict:
    ref, cand = reference.copy(), candidate.copy()
    if end_ts is not None:
        ref = ref[pd.to_datetime(ref["date"]) <= pd.Timestamp(end_ts)]
        cand = cand[pd.to_datetime(cand["date"]) <= pd.Timestamp(end_ts)]
    key = list(key)
    merged = ref.merge(cand, on=key, how="outer", suffixes=("_ref", "_cand"), indicator=True)
    only_ref = merged[merged["_merge"] == "left_only"]
    only_cand = merged[merged["_merge"] == "right_only"]
    both = merged[merged["_merge"] == "both"].copy()
    if len(both):
        both["rel_diff"] = ((both[f"{value_col}_cand"] - both[f"{value_col}_ref"]).abs()
                            / both[f"{value_col}_ref"].abs().clip(lower=1e-12))
    return {"n_reference": int(len(ref)), "n_candidate": int(len(cand)),
            "n_only_reference": int(len(only_ref)), "n_only_candidate": int(len(only_cand)),
            "max_rel_value_diff": float(both["rel_diff"].max()) if len(both) else 0.0,
            "n_value_beyond_tol": int((both["rel_diff"] > rel_tol).sum()) if len(both) else 0,
            "examples_only_reference": only_ref[key].head(5).astype(str).to_dict("records"),
            "examples_only_candidate": only_cand[key].head(5).astype(str).to_dict("records")}


def rescale_symbol_history(pricing_df: pd.DataFrame, symbol_str: str, factor_float: float) -> pd.DataFrame:
    """Protocol A2: whole loaded history re-based as if a k:1 split happened after the last date.
    OHLC and Dividend / k, Volume * k; 'Unadjusted Close' and 'Turnover' stay nominal."""
    out_df = pricing_df.copy()
    for field_str in ("Open", "High", "Low", "Close", "Dividend"):
        if (symbol_str, field_str) in out_df.columns:
            out_df[(symbol_str, field_str)] = out_df[(symbol_str, field_str)] / factor_float
    if (symbol_str, "Volume") in out_df.columns:
        out_df[(symbol_str, "Volume")] = out_df[(symbol_str, "Volume")] * factor_float
    out_df.attrs.update(pricing_df.attrs)
    return out_df
