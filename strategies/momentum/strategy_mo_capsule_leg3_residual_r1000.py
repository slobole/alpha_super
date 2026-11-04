"""
Leg 3: Russell 1000, residual after the market and industry peers. Twenty names, a rank buffer, inverse volatility, on two swap days.

The rules, the two books inside the leg and the caveats are in momentum_capsule_legs.py. Closed design and evidence:
docs/research/RESIDUAL_MOMENTUM_20261031.md. Execution: decision at Close_T, fill at the next open.
"""

from __future__ import annotations

import pandas as pd
from IPython.display import display

from alpha.engine.report import save_results
from strategies.momentum.momentum_capsule_legs import (
    LEG3_CONFIG,
    MomentumCapsuleLegStrategy,
    configured_leg_config,
    leg_timing_inputs,
    prepare_leg_strategy,
    run_leg_strategy,
)


STRATEGY_NAME_STR = "strategy_mo_capsule_leg3_residual_r1000"
DEFAULT_CONFIG = LEG3_CONFIG

__all__ = [
    "DEFAULT_CONFIG",
    "MomentumCapsuleLeg3Strategy",
    "build_capacity_analysis_inputs",
    "build_execution_timing_analysis_inputs",
    "run_variant",
]


class MomentumCapsuleLeg3Strategy(MomentumCapsuleLegStrategy):
    """Leg 3 of the momentum capsule."""


def run_variant(
    show_display_bool: bool = True,
    save_results_bool: bool = True,
    output_dir_str: str = "results",
    backtest_start_date_str: str | None = None,
    capital_base_float: float | None = None,
    end_date_str: str | None = None,
) -> MomentumCapsuleLeg3Strategy:
    config_obj = configured_leg_config(DEFAULT_CONFIG, backtest_start_date_str, capital_base_float, end_date_str)
    strategy_obj, pricing_data_df = prepare_leg_strategy(MomentumCapsuleLeg3Strategy, STRATEGY_NAME_STR, config_obj)
    run_leg_strategy(strategy_obj, pricing_data_df, config_obj, show_display_bool)
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
    config_obj = configured_leg_config(DEFAULT_CONFIG, backtest_start_date_str, capital_base_float, end_date_str)
    strategy_obj, pricing_data_df = prepare_leg_strategy(MomentumCapsuleLeg3Strategy, STRATEGY_NAME_STR, config_obj)
    # *** CRITICAL *** CapacityAnalysis must assess the same completed order ledger as the reference backtest.
    run_leg_strategy(strategy_obj, pricing_data_df, config_obj, show_display_bool)
    return {
        "strategy_obj": strategy_obj,
        "pricing_data_df": pricing_data_df,
        "execution_policy_str": "MOO",
        "impact_profile_str": "MOO_LARGE_MIXED",
    }


def build_execution_timing_analysis_inputs() -> dict[str, object]:
    return leg_timing_inputs(MomentumCapsuleLeg3Strategy, STRATEGY_NAME_STR, DEFAULT_CONFIG)


if __name__ == "__main__":
    run_variant()
