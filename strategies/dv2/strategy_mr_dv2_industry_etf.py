"""DV2 mean reversion on liquid US industry ETFs.

Decide after Close_T and fill orders at Open_(T+1). The rules are the WIRED
DV2 rules unchanged; only the universe differs: a fixed list of 19 industry
ETFs, each eligible once it has 252 sessions of its own price history and a
63-day average dollar volume above $50M.

    ADV63_i,t = mean over [t-62, t] of native Norgate Turnover_i
    eligible_i,t = 1[252 sessions of history] * 1[ADV63_i,t > 50,000,000]

    entry_i,t = eligible_i,t * 1[DV2_i,t < 10] * 1[Close_i,t > SMA200_i,t]
        * 1[Return126_i,t > 0.05]
    rank: NATR14 descending (as WIRED DV2)
    exit_i,t = 1[Close_i,t > High_i,t-1]

At most ten long positions of one tenth of prior portfolio value each; slippage
2.5 bps per side, $0.005/share commission with a $1 minimum; CAPITALSPECIAL
prices for signals and fills; $SPX total return as the performance benchmark.
The default backtest starts 2012-01-03: before about 2008 only one to five of
these ETFs were liquid enough, so earlier history is mostly an idle sleeve.

INVALIDATED as evidence for this version by the 2026-09-26 liquidity-unit
audit: the legacy results below used raw Close times split-adjusted Volume.
They require a native-Turnover rerun; the quantitative history is retained.
Research record: docs/research/DV2_DEEP_RESEARCH_20260925.md (ETF block).
2000-2026 with the same rules: CAGR 5.2% while active, Sharpe 0.92, max drawdown
-10%, daily correlation 0.49 with stock DV2. The $50M screen and the 19-ETF list
are research choices (today's ETFs: mild survivorship bias); forward-test first.

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
from data.norgate_loader import TOTALRETURN_ADJUSTMENT_STR
from strategies.dv2.strategy_mr_dv2_liquidity_floor import (
    DVO2LiquidityFloorStrategy,
    default_trade_id_int,
    get_asof_universe_symbol_list,
    get_prices,
)

STRATEGY_NAME_STR = "strategy_mr_dv2_industry_etf"
INDUSTRY_ETF_SYMBOL_TUPLE = (
    "XBI", "IBB", "SMH", "SOXX", "KRE", "KBE", "XHB", "ITB", "XRT", "XOP",
    "OIH", "XME", "GDX", "IYT", "IGV", "ITA", "IHI", "XSD", "XPH",
)
MIN_HISTORY_SESSION_INT = 252
MIN_ADV_DOLLAR_FLOAT = 50_000_000.0
DEFAULT_BACKTEST_START_DATE_STR = "2012-01-03"
DEFAULT_HISTORY_START_DATE_STR = "2009-01-01"


def build_history_universe_df(pricing_df: pd.DataFrame) -> pd.DataFrame:
    """1 once an ETF has MIN_HISTORY_SESSION_INT sessions of closes up to that date."""
    close_df = pricing_df.xs("Close", axis=1, level=1)
    close_df = close_df[[s for s in close_df.columns if not str(s).startswith("$")]]
    # *** CRITICAL*** Cumulative count through date t only; no later close enters.
    return (close_df.notna().cumsum() >= MIN_HISTORY_SESSION_INT).astype(int)


class DVO2IndustryEtfStrategy(DVO2LiquidityFloorStrategy):
    """Floor-module signals, sizing and exits; ETF eligibility and NATR rank."""

    def get_opportunities(self, close: pd.Series) -> list[str]:
        candidate_df = close.unstack()
        candidate_df = candidate_df[~candidate_df.index.astype(str).str.startswith("$")]
        member_list = get_asof_universe_symbol_list(self.universe_df, pd.Timestamp(self.previous_bar))
        member_df = candidate_df[candidate_df.index.isin(member_list)].dropna()
        member_df = member_df[member_df["adv_63"] > MIN_ADV_DOLLAR_FLOAT]
        member_df = member_df[
            (member_df["dv2"] < 10)
            & (member_df["Close"] > member_df["sma_200"])
            & (member_df["p126d_return"] > 0.05)
        ]
        return member_df.sort_values("natr", ascending=False).index.tolist()


def _new_strategy_obj(capital_base_float: float, universe_df: pd.DataFrame) -> DVO2IndustryEtfStrategy:
    strategy_obj = DVO2IndustryEtfStrategy(
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


def _load(end_date_str: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    pricing_df = get_prices(list(INDUSTRY_ETF_SYMBOL_TUPLE), ["$SPX"], start_date=DEFAULT_HISTORY_START_DATE_STR, end_date=end_date_str)
    return pricing_df, build_history_universe_df(pricing_df)


def build_execution_timing_analysis_inputs() -> dict[str, object]:
    """Inputs for ExecutionTimingAnalysis over the default backtest calendar."""
    pricing_df, universe_df = _load()
    calendar_idx = pricing_df.index[pricing_df.index >= pd.Timestamp(DEFAULT_BACKTEST_START_DATE_STR)]
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
    backtest_start_date_str: str = DEFAULT_BACKTEST_START_DATE_STR,
    capital_base_float: float = 100_000.0,
    end_date_str: str | None = None,
) -> dict[str, object]:
    """One completed run for CapacityAnalysis, on the same path as run_variant."""
    pricing_df, universe_df = _load(end_date_str)
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
        "impact_profile_str": "MOO_ETF_PROXY",
    }


def run_variant(
    show_display_bool: bool = True,
    save_results_bool: bool = True,
    output_dir_str: str = "results",
    backtest_start_date_str: str = DEFAULT_BACKTEST_START_DATE_STR,
    capital_base_float: float = 100_000.0,
    end_date_str: str | None = None,
):
    pricing_df, universe_df = _load(end_date_str)
    strategy_obj = _new_strategy_obj(capital_base_float, universe_df)

    # *** CRITICAL*** Indicators use the pre-start history; orders execute only on
    # the configured calendar (first fill = first open on or after the start date).
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
