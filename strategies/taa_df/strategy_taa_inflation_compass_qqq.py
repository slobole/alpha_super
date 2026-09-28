"""Inflation Compass with QQQ in the growth-up / inflation-off cell.

Identical to ``strategy_taa_inflation_compass`` in every signal, timing, cost and
data rule; only the goldilocks holding changes:

    growth up, inflation on   -> 100% XLE
    growth up, inflation off  -> 100% QQQ   (literal rule: XLK)
    growth down, inflation on -> 100% XLU
    growth down, inflation off -> 50% XLP + 50% IEF

Rationale and evidence (docs/research/INFLATION_COMPASS_DEEP_RESEARCH_20260928.md,
candidate C2, declared before results; the idea itself comes from Allocate
Smartly's "Enhanced" version, published after they had seen the same 2003-2026
history): XLK lost Alphabet and Meta to XLC in
2018, and QQQ trades roughly 15x XLK's dollar volume. On 2003-05..2026-08 the
research replica gave Sharpe 1.138 vs 1.085 (paired block bootstrap
P(dSharpe <= 0) = 0.031, not corrected for seven candidates), better in all
three periods and with doubled costs. QQQ starts 1999-03, so the pre-2003
holdout could not test it. Treat it as a shadow candidate: the parent's
parameter and rebalance-day luck (joint median Sharpe ~0.87) applies unchanged.

Caveats inherited from the parent: current-vintage FRED T5YIE (G-027), no
cash-constrained opening fills (G-028), research-only and absent from LIVE
wiring. The Stress analyzer is not registered for this module.
"""

from __future__ import annotations

from dataclasses import replace

from strategies.taa_df import strategy_taa_inflation_compass as base_module
from strategies.taa_df.strategy_taa_inflation_compass import (
    InflationCompassConfig,
    InflationCompassStrategy,
)


STRATEGY_NAME_STR = "strategy_taa_inflation_compass_qqq"
GOLDILOCKS_ASSET_STR = "QQQ"
TRADEABLE_ASSET_TUPLE = ("XLE", "QQQ", "XLU", "XLP", "IEF")

DEFAULT_CONFIG = replace(
    base_module.DEFAULT_CONFIG,
    tradeable_asset_tuple=TRADEABLE_ASSET_TUPLE,
    goldilocks_asset_str=GOLDILOCKS_ASSET_STR,
    strategy_name_str=STRATEGY_NAME_STR,
)


def run_variant(
    show_display_bool: bool = True,
    save_results_bool: bool = True,
    output_dir_str: str = "results",
    backtest_start_date_str: str | None = None,
    capital_base_float: float = DEFAULT_CONFIG.capital_base_float,
    end_date_str: str | None = None,
    config_obj: InflationCompassConfig = DEFAULT_CONFIG,
) -> InflationCompassStrategy:
    return base_module.run_variant(
        show_display_bool=show_display_bool,
        save_results_bool=save_results_bool,
        output_dir_str=output_dir_str,
        backtest_start_date_str=backtest_start_date_str,
        capital_base_float=capital_base_float,
        end_date_str=end_date_str,
        config_obj=config_obj,
    )


def build_capacity_analysis_inputs(
    show_display_bool: bool = False,
    backtest_start_date_str: str | None = None,
    capital_base_float: float = DEFAULT_CONFIG.capital_base_float,
    end_date_str: str | None = None,
) -> dict[str, object]:
    return base_module.build_capacity_analysis_inputs(
        show_display_bool=show_display_bool,
        backtest_start_date_str=backtest_start_date_str,
        capital_base_float=capital_base_float,
        end_date_str=end_date_str,
        config_obj=DEFAULT_CONFIG,
    )


def build_execution_timing_analysis_inputs() -> dict[str, object]:
    return base_module.build_execution_timing_analysis_inputs(config_obj=DEFAULT_CONFIG)


if __name__ == "__main__":
    run_variant()
