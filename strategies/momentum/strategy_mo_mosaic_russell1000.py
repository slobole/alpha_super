"""
MOSAIC — monthly Russell 1000 momentum that buys strength which does not move
together.

One-line intuition: a plain top-N momentum basket fills every slot with the
hottest theme and becomes one big trade; MOSAIC keeps the momentum score but
penalizes each next pick by its correlation to the names already chosen, so the
basket is assembled from strong tiles that do not share the same fate.

This is the deployment-reference wrapper around the correlation-penalized
selection engine (`strategy_mo_atr_normalized_ndx_corr_penalty`), with the
research-validated configuration locked:

    universe          Russell 1000 point-in-time members (Norgate)
    regime gate       $RUI close > SMA200, else 100% cash
    stock gates       close > SMA100, trailing 20d median dollar ADV >= $5M
    score             ROC12 / ATR20
    selection         greedy, adjusted = score - 0.75 * avg_corr * |score|,
                      correlation window 126d (robust across 63-252)
    positions         20, equal weight 1/20, shortfall stays in cash
    execution         month-end decision close -> next tradable open (MOO)

Validation record (2026-07-31, results/research/strategy/
mo_atr_normalized_russell_1000_corr_penalty_sweep/): monotone improvement in
lambda, stable across 2000-2012/2013-2026 halves, robust to the correlation
window, replicated on S&P 500, cross-confirmed by a GICS sector-cap arm, and
survives the ADV liquidity gate. No effect on NDX (too homogeneous) or NYSE
Composite (already diverse) — the mechanism needs a concentrated-but-wide pool.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import pandas as pd
from IPython.display import display

from alpha.engine.backtest import run_daily
from alpha.engine.report import save_results
from strategies.momentum.strategy_mo_atr_normalized_ndx import (
    configure_total_return_benchmark_provenance,
    get_atr_normalized_ndx_data,
)
from strategies.momentum.strategy_mo_atr_normalized_ndx_corr_penalty import (
    CorrPenaltyAtrNormalizedNdxConfig,
    CorrPenaltyAtrNormalizedNdxStrategy,
)


@dataclass(frozen=True)
class MosaicRussell1000Config(CorrPenaltyAtrNormalizedNdxConfig):
    # *** CRITICAL*** These defaults are the validated MOSAIC contract. Change
    # them only through a new research cycle — a silent edit here changes what
    # every book allocating to MOSAIC actually holds.
    indexname_str: str = "Russell 1000"
    regime_symbol_str: str = "$RUI"
    max_positions_int: int = 20
    corr_penalty_lambda_float: float = 0.75
    corr_window_int: int = 126
    corr_min_overlap_int: int = 63
    min_dollar_adv_float: float = 5_000_000.0
    adv_window_int: int = 20


DEFAULT_CONFIG = MosaicRussell1000Config()

STRATEGY_NAME_STR = "strategy_mo_mosaic_russell1000"

__all__ = [
    "DEFAULT_CONFIG",
    "MosaicRussell1000Config",
    "MosaicRussell1000Strategy",
    "STRATEGY_NAME_STR",
    "build_capacity_analysis_inputs",
    "build_mosaic_strategy",
    "run_variant",
]


class MosaicRussell1000Strategy(CorrPenaltyAtrNormalizedNdxStrategy):
    """MOSAIC: correlation-penalized Russell 1000 momentum. Logic lives in the
    parent class; this subclass exists so books, releases, and results name the
    strategy explicitly."""


def _resolve_config_obj(
    backtest_start_date_str: str | None,
    capital_base_float: float | None,
    end_date_str: str | None,
) -> MosaicRussell1000Config:
    if (
        backtest_start_date_str is None
        and capital_base_float is None
        and end_date_str is None
    ):
        return DEFAULT_CONFIG
    return replace(
        DEFAULT_CONFIG,
        backtest_start_date_str=(
            DEFAULT_CONFIG.backtest_start_date_str
            if backtest_start_date_str is None
            else backtest_start_date_str
        ),
        capital_base_float=(
            DEFAULT_CONFIG.capital_base_float
            if capital_base_float is None
            else float(capital_base_float)
        ),
        end_date_str=end_date_str,
    )


def build_mosaic_strategy(
    config: MosaicRussell1000Config,
    rebalance_schedule_df: pd.DataFrame,
) -> MosaicRussell1000Strategy:
    return MosaicRussell1000Strategy(
        name=STRATEGY_NAME_STR,
        benchmarks=[config.performance_benchmark_symbol_str],
        rebalance_schedule_df=rebalance_schedule_df,
        regime_symbol_str=config.regime_symbol_str,
        capital_base=config.capital_base_float,
        slippage=config.slippage_float,
        commission_per_share=config.commission_per_share_float,
        commission_minimum=config.commission_minimum_float,
        lookback_month_int=config.lookback_month_int,
        index_trend_window_int=config.index_trend_window_int,
        stock_trend_window_int=config.stock_trend_window_int,
        max_positions_int=config.max_positions_int,
        corr_window_int=config.corr_window_int,
        corr_min_overlap_int=config.corr_min_overlap_int,
        corr_penalty_lambda_float=config.corr_penalty_lambda_float,
        min_dollar_adv_float=config.min_dollar_adv_float,
        adv_window_int=config.adv_window_int,
    )


def run_variant(
    show_display_bool: bool = True,
    save_results_bool: bool = True,
    output_dir_str: str = "results",
    backtest_start_date_str: str | None = None,
    capital_base_float: float | None = None,
    end_date_str: str | None = None,
) -> MosaicRussell1000Strategy:
    config_obj = _resolve_config_obj(
        backtest_start_date_str=backtest_start_date_str,
        capital_base_float=capital_base_float,
        end_date_str=end_date_str,
    )
    pricing_data_df, universe_df, rebalance_schedule_df = get_atr_normalized_ndx_data(
        config_obj,
        include_total_return_benchmark_bool=True,
    )

    strategy_obj = build_mosaic_strategy(
        config=config_obj,
        rebalance_schedule_df=rebalance_schedule_df,
    )
    strategy_obj.universe_df = universe_df
    configure_total_return_benchmark_provenance(
        strategy_obj=strategy_obj,
        config_obj=config_obj,
    )

    # *** CRITICAL*** Deployment-reference backtests keep full pre-start
    # history for monthly ATR, trend, correlation, and ADV features, but the
    # executable calendar starts at the first deployment fill session.
    calendar_idx = pricing_data_df.index[
        pricing_data_df.index >= pd.Timestamp(config_obj.backtest_start_date_str)
    ]
    run_daily(
        strategy_obj,
        pricing_data_df,
        calendar=calendar_idx,
        show_progress=show_display_bool,
        show_signal_progress_bool=show_display_bool,
        audit_override_bool=None,
    )

    if show_display_bool:
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 1000)
        display(strategy_obj.summary)
        display(strategy_obj.summary_trades)

    if save_results_bool:
        save_results(strategy_obj, output_dir=output_dir_str)

    return strategy_obj


def build_capacity_analysis_inputs(
    show_display_bool: bool = False,
    backtest_start_date_str: str | None = None,
    capital_base_float: float | None = None,
    end_date_str: str | None = None,
) -> dict[str, object]:
    config_obj = _resolve_config_obj(
        backtest_start_date_str=backtest_start_date_str,
        capital_base_float=capital_base_float,
        end_date_str=end_date_str,
    )
    pricing_data_df, universe_df, rebalance_schedule_df = get_atr_normalized_ndx_data(
        config_obj,
        include_total_return_benchmark_bool=True,
    )

    strategy_obj = build_mosaic_strategy(
        config=config_obj,
        rebalance_schedule_df=rebalance_schedule_df,
    )
    strategy_obj.universe_df = universe_df
    configure_total_return_benchmark_provenance(
        strategy_obj=strategy_obj,
        config_obj=config_obj,
    )

    # *** CRITICAL *** CapacityAnalysis must assess the same completed order
    # ledger as the deployment-reference MOSAIC backtest. Keep pre-start
    # history for monthly features, but execute only on the configured
    # calendar.
    calendar_idx = pricing_data_df.index[
        pricing_data_df.index >= pd.Timestamp(config_obj.backtest_start_date_str)
    ]
    run_daily(
        strategy_obj,
        pricing_data_df,
        calendar=calendar_idx,
        show_progress=show_display_bool,
        show_signal_progress_bool=show_display_bool,
        audit_override_bool=None,
    )

    strategy_obj.universe_df = None
    return {
        "strategy_obj": strategy_obj,
        "pricing_data_df": pricing_data_df,
        "execution_policy_str": "MOO",
        # Russell 1000 spans NYSE and Nasdaq listings; the mixed large-cap MOO
        # profile is the honest choice over the Nasdaq-only profile.
        "impact_profile_str": "MOO_LARGE_MIXED",
    }


if __name__ == "__main__":
    run_variant()
