"""HPI readiness audit (2026-09-28): shared helpers. Research-only; no strategy/engine/live file is edited.

Scope: strategies.hpi.strategy_mr_hpi_sp500_2_3_5_vote and strategies.hpi.strategy_mr_hpi_sp500_ibs_rsi_exit
(both run strategies.hpi.stateful_long.run_hpi_variant with Turnover ranking, LIQUIDITY_NONE, 10 slots,
$100k, 2.5 bp slippage, $0.005/share, $1 minimum, start 2004-01-01, pre-start history from 1998-01-01).

The factories here reproduce run_hpi_variant exactly (same constructor arguments, same attrs, same calendar).
RecordingHPIStrategy only records state around iterate(); it calls the production iterate() unchanged.
"""

from __future__ import annotations

import contextlib
import io
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
for path_obj in (REPO, HERE):
    if str(path_obj) not in sys.path:
        sys.path.insert(0, str(path_obj))

from alpha.engine.backtest import run_daily  # noqa: E402
from strategies.hpi import stateful_long as hpi_mod  # noqa: E402

OUT = REPO / "results" / "research" / "strategy_readiness_audit_20260928" / "hpi"
CACHE = OUT / "_cache"
OUT.mkdir(parents=True, exist_ok=True)
CACHE.mkdir(parents=True, exist_ok=True)

BENCH = "$SPXTR"
START = "2004-01-01"
VARIANTS = {
    "vote": ("strategy_mr_hpi_sp500_2_3_5_vote", hpi_mod.ENTRY_HORIZON_VOTE_STR,
             "strategies.hpi.strategy_mr_hpi_sp500_2_3_5_vote"),
    "baseline": ("strategy_mr_hpi_sp500_ibs_rsi_exit", hpi_mod.ENTRY_BASELINE_STR,
                 "strategies.hpi.strategy_mr_hpi_sp500_ibs_rsi_exit"),
}


def load_full_inputs(refresh: bool = False) -> dict:
    path = CACHE / "hpi_inputs_full.pkl"
    if path.exists() and not refresh:
        with path.open("rb") as handle:
            out = pickle.load(handle)
        return out
    t0 = time.time()
    symbols, universe_df, pricing_df = hpi_mod.load_exact_hpi_inputs("S&P 500", BENCH, "1998-01-01", None)
    out = {"symbols": symbols, "universe": universe_df, "pricing_df": pricing_df,
           "load_seconds": time.time() - t0}
    with path.open("wb") as handle:
        pickle.dump(out, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return out


def set_adjustment_attrs(pricing_df: pd.DataFrame) -> pd.DataFrame:
    """Exactly what run_hpi_variant sets before run_daily."""
    pricing_df.attrs["norgate_adjustment_by_symbol_dict"] = {
        str(s): ("TOTALRETURN" if s == BENCH else "CAPITALSPECIAL")
        for s in pricing_df.columns.get_level_values(0).unique().astype(str)
    }
    return pricing_df


class RecordingHPIStrategy(hpi_mod.HPIStatefulLongStrategy):
    """Production iterate(); records the state it saw and the orders it created (for parity and audits)."""

    record_bool = True

    def iterate(self, data_df, close_row_ser, open_price_ser):
        if data_df is None or close_row_ser is None or not self.record_bool:
            return super().iterate(data_df, close_row_ser, open_price_ser)
        position_ser = self.get_positions()
        held = position_ser[position_ser > 0]
        prior_ids = {id(o) for o in self.get_orders()}
        rec = {
            "signal_date": pd.Timestamp(self.previous_bar),
            "exec_date": pd.Timestamp(self.current_bar),
            "positions": {str(k): float(v) for k, v in held.items()},
            "pending_before": sorted(self.pending_exit_symbol_set),
            "trade_id_int": int(self.trade_id_int),
            "current_trade_map": {str(k): int(v) for k, v in self.current_trade_map.items()},
            "prev_total_value": float(self.previous_total_value),
            "cash_before": float(self.cash),
            "missing_open_held": sorted(str(s) for s in held.index
                                        if not np.isfinite(float(open_price_ser.get(s, np.nan)))),
        }
        super().iterate(data_df, close_row_ser, open_price_ser)
        new_orders = [o for o in self.get_orders() if id(o) not in prior_ids]
        rec["exits"] = sorted(str(o.asset) for o in new_orders if o.target and abs(float(o.amount)) <= 1e-9)
        rec["entries"] = [str(o.asset) for o in new_orders if (not o.target) and o.unit == "value"]
        rec["entry_values"] = [float(o.amount) for o in new_orders if (not o.target) and o.unit == "value"]
        rec["pending_after"] = sorted(self.pending_exit_symbol_set)
        self.records.append(rec)


def make_strategy(variant: str, universe_df: pd.DataFrame, capital: float = 100_000.0, start: str = START,
                  cls=RecordingHPIStrategy, hsu: bool = False):
    name, entry_mode, _ = VARIANTS[variant]
    strategy = cls(name=name, benchmarks=[BENCH], ranking_field_str=hpi_mod.TURNOVER_FIELD_STR,
                   capital_base=capital, entry_mode_str=entry_mode, liquidity_mode_str=hpi_mod.LIQUIDITY_NONE_STR,
                   backtest_start_date_str=start)
    strategy.universe_df = universe_df
    strategy.records = []
    strategy.historical_share_units_bool = hsu
    return strategy


def run(strategy, pricing_df: pd.DataFrame, start: str, end=None, quiet: bool = True):
    frame = pricing_df if end is None else pricing_df.loc[: pd.Timestamp(end)]
    if end is not None:
        frame = frame.copy()
        frame.attrs.update(pricing_df.attrs)
    calendar = frame.index[frame.index >= pd.Timestamp(start)]
    sink = io.StringIO()
    ctx = contextlib.redirect_stdout(sink) if quiet else contextlib.nullcontext()
    with ctx:
        run_daily(strategy, frame, calendar, show_progress=False, show_signal_progress_bool=False)
    strategy.captured_stdout = sink.getvalue()
    return strategy


def metrics(nav: pd.Series, start=None, end=None) -> dict:
    nav = nav.astype(float)
    if start is not None:
        nav = nav.loc[pd.Timestamp(start):]
    if end is not None:
        nav = nav.loc[: pd.Timestamp(end)]
    ret = nav.pct_change().dropna()
    years = (nav.index[-1] - nav.index[0]).days / 365.25
    return {"start": str(nav.index[0].date()), "end": str(nav.index[-1].date()),
            "cagr": float((nav.iloc[-1] / nav.iloc[0]) ** (1 / years) - 1),
            "sharpe": float(ret.mean() / ret.std() * np.sqrt(252)),
            "max_dd": float((nav / nav.cummax() - 1).min()), "vol": float(ret.std() * np.sqrt(252))}


def subset_pricing(pricing_df, universe_df, start, end, extra=()):
    """Symbols that are PIT members in [start-30d, end] plus extras and the benchmark (decision-neutral for HPI:
    per-stock time-series features, entries require membership, no cross-sectional statistic over non-members)."""
    window = universe_df.loc[pd.Timestamp(start) - pd.Timedelta(days=30): pd.Timestamp(end)]
    members = set(window.columns[window.sum(axis=0) > 0].astype(str)) | set(extra) | {BENCH}
    keep = [c for c in pricing_df.columns if str(c[0]) in members]
    out = pricing_df.loc[: pd.Timestamp(end), keep].copy()
    return set_adjustment_attrs(out)


def dump_json(obj, name: str) -> Path:
    path = OUT / name
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
    return path
