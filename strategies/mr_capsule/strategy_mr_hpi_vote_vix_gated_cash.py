"""MR capsule gate-only stage: idle cash earns 0%; no BIL or SPMO parking.

Uses the same stock rules and Close_T gate as the parked capsule. This separate
identity keeps cash-stage live and reference results distinct from BIL/SPMO.
"""

from __future__ import annotations

from strategies.mr_capsule.hpi_vote_vix_gated import (
    build_hpi_capsule_capacity_analysis_inputs,
    build_hpi_capsule_execution_timing_analysis_inputs,
    run_hpi_capsule_pod,
)

STRATEGY_NAME_STR = "strategy_mr_hpi_vote_vix_gated_cash"


def run_variant(
    show_display_bool: bool = True,
    save_results_bool: bool = True,
    output_dir_str: str = "results",
    backtest_start_date_str: str = "2004-01-01",
    capital_base_float: float = 100_000.0,
    end_date_str: str | None = None,
    slippage_float: float = 0.00025,
):
    return run_hpi_capsule_pod(
        strategy_name_str=STRATEGY_NAME_STR,
        parking_enabled_bool=False,
        spmo_parking_enabled_bool=False,
        show_display_bool=show_display_bool,
        save_results_bool=save_results_bool,
        output_dir_str=output_dir_str,
        backtest_start_date_str=backtest_start_date_str,
        capital_base_float=capital_base_float,
        end_date_str=end_date_str,
        slippage_float=slippage_float,
    )


def build_capacity_analysis_inputs(
    show_display_bool: bool = False,
    backtest_start_date_str: str = "2004-01-01",
    capital_base_float: float = 100_000.0,
    end_date_str: str | None = None,
) -> dict[str, object]:
    return build_hpi_capsule_capacity_analysis_inputs(
        strategy_name_str=STRATEGY_NAME_STR,
        parking_enabled_bool=False,
        spmo_parking_enabled_bool=False,
        show_display_bool=show_display_bool,
        backtest_start_date_str=backtest_start_date_str,
        capital_base_float=capital_base_float,
        end_date_str=end_date_str,
    )


def build_execution_timing_analysis_inputs() -> dict[str, object]:
    return build_hpi_capsule_execution_timing_analysis_inputs(
        strategy_name_str=STRATEGY_NAME_STR,
        parking_enabled_bool=False,
        spmo_parking_enabled_bool=False,
    )


if __name__ == "__main__":
    run_variant()
