"""MR capsule leakage hunt: strategy factories and run helpers shared by the mr_* scripts (research-only).

The factories reproduce the production run paths exactly:
- DV2:  strategies.dv2.strategy_mr_dv2.run_variant (DVO2Strategy, $SPX TR benchmark, 2.5 bps, $0.005/sh, $1 min).
- HPI:  strategies.hpi.stateful_long.run_hpi_variant with the 2/3/5 vote wrapper arguments (Turnover rank).
- ETF:  strategies.dv2.strategy_mr_dv2_industry_etf._new_strategy_obj.
Nothing is edited; study variants are subclasses or runtime attributes only.
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
REPO = HERE.parents[2]
for path in (REPO, HERE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from alpha.engine.backtest import run_daily  # noqa: E402
from data.norgate_loader import TOTALRETURN_ADJUSTMENT_STR  # noqa: E402
from strategies.dv2.strategy_mr_dv2 import DVO2Strategy, default_trade_id_int  # noqa: E402
from strategies.dv2 import strategy_mr_dv2_industry_etf as etf_mod  # noqa: E402
from strategies.hpi import stateful_long as hpi_mod  # noqa: E402

OUT = REPO / "results" / "research" / "leakage_hunt_20260927" / "mr"
OUT.mkdir(parents=True, exist_ok=True)
STUDY_END = pd.Timestamp("2026-08-19")
FULL_START = pd.Timestamp("2004-01-01")


def make_dv2(universe_df: pd.DataFrame, capital: float = 100_000.0) -> DVO2Strategy:
    strategy = DVO2Strategy(name="strategy_mr_dv2", benchmarks=["$SPX"], capital_base=capital, slippage=0.00025,
                            commission_per_share=0.005, commission_minimum=1.0,
                            performance_benchmark_adjustment_str=TOTALRETURN_ADJUSTMENT_STR)
    strategy.universe_df = universe_df
    strategy.trade_id = 0
    strategy.current_trade = defaultdict(default_trade_id_int)
    return strategy


def make_hpi(universe_df: pd.DataFrame, capital: float = 100_000.0, start: str = "2004-01-01",
             cls=hpi_mod.HPIStatefulLongStrategy):
    strategy = cls(name="strategy_mr_hpi_sp500_2_3_5_vote", benchmarks=["$SPXTR"],
                   ranking_field_str=hpi_mod.TURNOVER_FIELD_STR, capital_base=capital,
                   entry_mode_str=hpi_mod.ENTRY_HORIZON_VOTE_STR, liquidity_mode_str=hpi_mod.LIQUIDITY_NONE_STR,
                   backtest_start_date_str=start)
    strategy.universe_df = universe_df
    return strategy


def make_etf(universe_df: pd.DataFrame, capital: float = 100_000.0):
    return etf_mod._new_strategy_obj(capital, universe_df)


class HPILiveSlotStrategy(hpi_mod.HPIStatefulLongStrategy):
    """E-03 study arm: an exit frees its slot only after the exit has filled (next decision), as the live host does.

    Exits are still submitted only when Open_(T+1) exists (the engine cannot fill a missing open), but the slot is not
    reused at that same open.  Entries therefore see max_positions - held count at Close_T.
    """

    def iterate(self, data_df, close_row_ser, open_price_ser):
        if data_df is None or close_row_ser is None:
            return
        # Run the production logic with no open prices: no exit orders and no slot freeing (live semantics) ...
        super().iterate(data_df, close_row_ser, pd.Series(dtype=float))
        # ... then submit the pending exits the backtest can fill (same as the live host's post-iterate loop).
        held = self.get_positions()
        held_set = set(held[held > 0].index.astype(str))
        for symbol_str in sorted(self.pending_exit_symbol_set.intersection(held_set)):
            open_float = open_price_ser.get(symbol_str, np.nan)
            if pd.notna(open_float) and np.isfinite(float(open_float)):
                self.order_target_value(symbol_str, 0.0, trade_id=self.current_trade_map[symbol_str])


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
    """Symbols that are PIT members at any time in [start - 10 sessions, end] plus extras and the benchmark.

    Decision-neutral for DV2/HPI (no cross-sectional statistic over non-members; entries require membership), used
    only to make the many short invariance/prefix runs cheap.  Full-history metric runs use the full frame.
    """
    window = universe_df.loc[pd.Timestamp(start) - pd.Timedelta(days=20): pd.Timestamp(end)]
    members = set(window.columns[window.sum(axis=0) > 0].astype(str)) | set(extra) | set(bench)
    keep = [c for c in pricing_df.columns if str(c[0]) in members]
    out = pricing_df.loc[:, keep].copy()
    out.attrs.update(pricing_df.attrs)
    return out


def asof_trimmed_universe(untrimmed_df: pd.DataFrame, asof_ts) -> pd.DataFrame:
    """Universe exactly as data/norgate_loader.build_index_constituent_matrix would build it with data ending at asof:
    each symbol's membership rows up to asof; if its last member row != asof, drop its last 5 member rows."""
    asof_ts = pd.Timestamp(asof_ts)
    frame = untrimmed_df.loc[:asof_ts]
    out = frame.copy()
    for symbol in frame.columns:
        rows = np.flatnonzero(frame[symbol].to_numpy() == 1)
        if len(rows) == 0:
            continue
        if frame.index[rows[-1]] != frame.index[-1]:
            out.iloc[rows[-5:], out.columns.get_loc(symbol)] = 0
    out = out.loc[:, out.sum(axis=0) > 0]
    return out
