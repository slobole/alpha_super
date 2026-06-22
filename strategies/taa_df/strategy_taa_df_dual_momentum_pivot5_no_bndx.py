"""
Dual Momentum Pivot5 variant without BNDX.

This keeps the same Pivot5 rules as strategy_taa_df_dual_momentum_pivot5 and
only removes BNDX from the ETF universe so the real-ETF sample can start before
BNDX history is available.
"""

from __future__ import annotations

from dataclasses import replace

from strategies.taa_df.strategy_taa_df_dual_momentum_pivot5 import (
    DEFAULT_CONFIG,
    DualMomentumPivot5Config,
    DualMomentumPivot5Strategy,
    build_execution_timing_analysis_inputs as _base_build_execution_timing_analysis_inputs,
    build_friction_analysis_inputs as _base_build_friction_analysis_inputs,
    run_friction_analysis as _base_run_friction_analysis,
    run_variant as _base_run_variant,
)


STRATEGY_NAME_STR = "strategy_taa_df_dual_momentum_pivot5_no_bndx"
NO_BNDX_ASSET_TUPLE = tuple(asset_str for asset_str in DEFAULT_CONFIG.asset_list if asset_str != "BNDX")
NO_BNDX_CONFIG = replace(DEFAULT_CONFIG, asset_list=NO_BNDX_ASSET_TUPLE)


def run_variant(
    show_display_bool: bool = True,
    save_results_bool: bool = True,
    output_dir_str: str = "results",
    backtest_start_date_str: str | None = None,
    capital_base_float: float = NO_BNDX_CONFIG.capital_base_float,
    end_date_str: str | None = None,
) -> DualMomentumPivot5Strategy:
    return _base_run_variant(
        show_display_bool=show_display_bool,
        save_results_bool=save_results_bool,
        output_dir_str=output_dir_str,
        backtest_start_date_str=backtest_start_date_str,
        capital_base_float=capital_base_float,
        end_date_str=end_date_str,
        config=NO_BNDX_CONFIG,
        strategy_name_str=STRATEGY_NAME_STR,
    )


def build_friction_analysis_inputs(
    show_display_bool: bool = False,
    backtest_start_date_str: str | None = None,
    capital_base_float: float = NO_BNDX_CONFIG.capital_base_float,
    end_date_str: str | None = None,
) -> dict[str, object]:
    return _base_build_friction_analysis_inputs(
        show_display_bool=show_display_bool,
        backtest_start_date_str=backtest_start_date_str,
        capital_base_float=capital_base_float,
        end_date_str=end_date_str,
        config=NO_BNDX_CONFIG,
        strategy_name_str=STRATEGY_NAME_STR,
    )


def run_friction_analysis(
    save_results_bool: bool = True,
    output_dir_str: str = "results",
    show_display_bool: bool = False,
    backtest_start_date_str: str | None = None,
    capital_base_float: float = NO_BNDX_CONFIG.capital_base_float,
    end_date_str: str | None = None,
):
    return _base_run_friction_analysis(
        save_results_bool=save_results_bool,
        output_dir_str=output_dir_str,
        show_display_bool=show_display_bool,
        backtest_start_date_str=backtest_start_date_str,
        capital_base_float=capital_base_float,
        end_date_str=end_date_str,
        config=NO_BNDX_CONFIG,
        strategy_name_str=STRATEGY_NAME_STR,
    )


def build_execution_timing_analysis_inputs() -> dict[str, object]:
    return _base_build_execution_timing_analysis_inputs(
        config=NO_BNDX_CONFIG,
        strategy_name_str=STRATEGY_NAME_STR,
    )


if __name__ == "__main__":
    run_variant()
