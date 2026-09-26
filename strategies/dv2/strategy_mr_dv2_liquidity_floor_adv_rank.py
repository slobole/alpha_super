"""DV2 with the point-in-time liquidity floor, candidates ranked by dollar volume.

Decide after Close_T and fill orders at Open_(T+1). The universe is the
point-in-time S&P 500. The trade and signal price basis is CAPITALSPECIAL;
the $SPX performance benchmark is loaded as TOTALRETURN. At most ten long
positions receive one tenth of prior portfolio value each. Slippage is 2.5
basis points per side; commission is $0.005/share with a $1 minimum.

Everything is identical to strategies/dv2/strategy_mr_dv2_liquidity_floor.py
except the order in which qualifying candidates fill the free slots:

    ADV63_i,t = mean over [t-62, t] of native Norgate Turnover_i
    median_t  = median of available ADV63 over that day's PIT S&P 500 members

    entry_i,t = 1[UnadjustedClose_i,t > 5] * 1[ADV63_i,t > median_t]
        * 1[DV2_i,t < 10] * 1[Close_i,t > SMA200_i,t] * 1[Return126_i,t > 0.05]

    rank: ADV63 descending (the floor module ranks by NATR14 descending)

    exit_i,t = 1[Close_i,t > High_i,t-1]

INVALIDATED as evidence for this version by the 2026-09-26 liquidity-unit
audit: the legacy results below used raw Close times split-adjusted Volume.
They require a native-Turnover rerun; the quantitative history is retained.
Research record: docs/research/DV2_DEEP_RESEARCH_20260925.md (finalist F1).
2000-2026, $1M, engine costs: CAGR 22.3%, Sharpe 1.18, max drawdown -21%
(floor with NATR rank: 22.2%, 1.07, -24%); untouched 1991-1999 holdout Sharpe
2.37 vs 2.34. NATR ranking scored at the 33rd percentile of 200 random
rankings of the same candidates; ADV ranking at the 97.5th.

Bench displays WIRED for this module. That display does not authorize a LIVE
release, broker route, or portfolio allocation; the registry tier stays
RESEARCH until the owner approves a live route.
"""

from __future__ import annotations

from collections import defaultdict

import pandas as pd
from IPython.display import display

from alpha.engine.backtest import run_daily
from alpha.engine.report import save_results
from data.norgate_loader import (
    TOTALRETURN_ADJUSTMENT_STR,
    build_index_constituent_matrix,
)
from strategies.dv2.strategy_mr_dv2_liquidity_floor import (
    RAW_PRICE_MIN_FLOAT,
    DVO2LiquidityFloorStrategy,
    default_trade_id_int,
    get_asof_universe_symbol_list,
    get_prices,
)

STRATEGY_NAME_STR = "strategy_mr_dv2_liquidity_floor_adv_rank"


class DVO2LiquidityFloorAdvRankStrategy(DVO2LiquidityFloorStrategy):
    """Floor-module signals, sizing and exits; only the candidate ranking differs."""

    def get_opportunities(self, close: pd.Series) -> list[str]:
        candidate_df = close.unstack()
        candidate_df = candidate_df[~candidate_df.index.astype(str).str.startswith("$")]
        member_list = get_asof_universe_symbol_list(self.universe_df, pd.Timestamp(self.previous_bar))
        member_df = candidate_df[candidate_df.index.isin(member_list)]
        # *** CRITICAL*** Same PIT median as the floor module: over that day's index
        # members with available ADV63, before requiring complete DV2 rows.
        median_adv_float = float(member_df["adv_63"].dropna().median())
        if not pd.notna(median_adv_float):
            return []
        member_df = member_df.dropna()
        member_df = member_df[(member_df["raw_price"] > RAW_PRICE_MIN_FLOAT) & (member_df["adv_63"] > median_adv_float)]
        member_df = member_df[
            (member_df["dv2"] < 10)
            & (member_df["Close"] > member_df["sma_200"])
            & (member_df["p126d_return"] > 0.05)
        ]
        # Most liquid first; a stable sort keeps ties in universe order.
        return member_df.sort_values("adv_63", ascending=False, kind="stable").index.tolist()


def _new_strategy_obj(capital_base_float: float, universe_df: pd.DataFrame) -> DVO2LiquidityFloorAdvRankStrategy:
    strategy_obj = DVO2LiquidityFloorAdvRankStrategy(
        name=STRATEGY_NAME_STR,
        benchmarks=["$SPX"],
        capital_base=capital_base_float,
        slippage=0.00025,
        commission_per_share=0.005,
        commission_minimum=1.0,
        performance_benchmark_adjustment_str=TOTALRETURN_ADJUSTMENT_STR,
    )
    strategy_obj.universe_df = universe_df
    strategy_obj.trade_id = 0
    strategy_obj.current_trade = defaultdict(default_trade_id_int)
    return strategy_obj


def build_execution_timing_analysis_inputs() -> dict[str, object]:
    """Inputs for ExecutionTimingAnalysis; same calendar and timing grid as the WIRED DV2."""
    symbol_list, universe_df = build_index_constituent_matrix(indexname="S&P 500")
    pricing_df = get_prices(symbol_list, ["$SPX"], start_date="1998-01-01", end_date=None)
    # *** CRITICAL*** Keep the same post-warmup calendar as the WIRED DV2 timing study.
    calendar_idx = pricing_df.index[pricing_df.index.year >= 2004]
    return {
        "strategy_factory_fn": lambda: _new_strategy_obj(100_000.0, universe_df),
        "pricing_data_df": pricing_df,
        "calendar_idx": pd.DatetimeIndex(calendar_idx),
        "order_generation_mode_str": "signal_bar",
        "risk_model_str": "daily_ohlc_signal",
        "entry_timing_str_tuple": ("same_close_moc", "next_open", "next_close"),
        "exit_timing_str_tuple": ("same_close_moc", "next_open", "next_close"),
        "default_entry_timing_str": "next_open",
        "default_exit_timing_str": "next_open",
    }


def build_capacity_analysis_inputs(
    show_display_bool: bool = False,
    backtest_start_date_str: str = "2004-01-01",
    capital_base_float: float = 100_000.0,
    end_date_str: str | None = None,
) -> dict[str, object]:
    """One completed run for CapacityAnalysis, on the same path as run_variant."""
    symbol_list, universe_df = build_index_constituent_matrix(indexname="S&P 500")
    pricing_df = get_prices(symbol_list, ["$SPX"], start_date="1998-01-01", end_date=end_date_str)
    strategy_obj = _new_strategy_obj(capital_base_float, universe_df)
    calendar_idx = pricing_df.index[pricing_df.index >= pd.Timestamp(backtest_start_date_str)]
    run_daily(strategy_obj, pricing_df, calendar_idx, show_progress=show_display_bool, show_signal_progress_bool=show_display_bool)
    strategy_obj.universe_df = None
    strategy_obj._performance_benchmark_symbol_str = "$SPX"
    strategy_obj._performance_benchmark_adjustment_str = "TOTALRETURN"
    return {
        "strategy_obj": strategy_obj,
        "pricing_data_df": pricing_df,
        "execution_policy_str": "MOO",
        "impact_profile_str": "MOO_LARGE_MIXED",
    }


def run_variant(
    show_display_bool: bool = True,
    save_results_bool: bool = True,
    output_dir_str: str = "results",
    backtest_start_date_str: str = "2004-01-01",
    capital_base_float: float = 100_000.0,
    end_date_str: str | None = None,
):
    index_symbol_list, universe_df = build_index_constituent_matrix(indexname="S&P 500")
    pricing_df = get_prices(index_symbol_list, ["$SPX"], start_date="1998-01-01", end_date=end_date_str)
    strategy_obj = _new_strategy_obj(capital_base_float, universe_df)

    # *** CRITICAL*** Keep full pre-start history for indicators; execute only on the
    # configured calendar, as the WIRED DV2 does.
    calendar_idx = pricing_df.index[pricing_df.index >= pd.Timestamp(backtest_start_date_str)]
    run_daily(
        strategy_obj,
        pricing_df,
        calendar_idx,
        show_progress=show_display_bool,
        show_signal_progress_bool=show_display_bool,
    )
    strategy_obj.universe_df = None

    if show_display_bool:
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 1000)
        display(strategy_obj.summary)
        display(strategy_obj.summary_trades)
    if save_results_bool:
        save_results(strategy_obj, output_dir=output_dir_str)
    return strategy_obj


if __name__ == "__main__":
    run_variant()
