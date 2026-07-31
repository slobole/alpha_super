"""
Defense First BTAL 1/n fallback-SSO variant with a month-end VRP cash gate.

Relative to `strategy_taa_df_btal_1n_fallback_tqqq_vix_cash`, SSO changes
both the daily leverage target (3x to 2x) and the fallback index
(Nasdaq-100 to S&P 500). All other strategy semantics are unchanged.

Fallback overlay:
    if rv20_m < VIX_m:
        keep SSO fallback weight
    else:
        set SSO fallback weight to 0 and leave the residual as cash
"""

from __future__ import annotations

from dataclasses import replace

try:
    from strategies.taa_df.strategy_taa_df import get_defense_first_data
    from strategies.taa_df.strategy_taa_df_btal_1n_fallback_sso import DEFAULT_CONFIG as BASE_CONFIG
    from strategies.taa_df.strategy_taa_df_fallback_vix_cash_variant_utils import (
        build_vix_cash_variant_config,
        build_standard_fallback_vix_cash_capacity_analysis_inputs,
        build_standard_fallback_vix_cash_execution_timing_analysis_inputs,
        run_standard_fallback_vix_cash_variant,
    )
except ModuleNotFoundError:
    from strategy_taa_df import get_defense_first_data
    from strategy_taa_df_btal_1n_fallback_sso import DEFAULT_CONFIG as BASE_CONFIG
    from strategy_taa_df_fallback_vix_cash_variant_utils import (
        build_vix_cash_variant_config,
        build_standard_fallback_vix_cash_capacity_analysis_inputs,
        build_standard_fallback_vix_cash_execution_timing_analysis_inputs,
        run_standard_fallback_vix_cash_variant,
    )


STRATEGY_NAME_STR = "strategy_taa_df_btal_1n_fallback_sso_vix_cash"
DEFAULT_CONFIG = build_vix_cash_variant_config(BASE_CONFIG)


def run_variant(
    show_display_bool: bool = True,
    save_results_bool: bool = True,
    output_dir_str: str = "results",
    backtest_start_date_str: str | None = None,
    capital_base_float: float = 100_000.0,
    end_date_str: str | None = None,
):
    config_obj = (
        DEFAULT_CONFIG
        if end_date_str is None
        else replace(DEFAULT_CONFIG, end_date_str=end_date_str)
    )
    return run_standard_fallback_vix_cash_variant(
        strategy_name_str=STRATEGY_NAME_STR,
        config=config_obj,
        base_data_loader_fn=get_defense_first_data,
        show_display_bool=show_display_bool,
        save_results_bool=save_results_bool,
        output_dir_str=output_dir_str,
        backtest_start_date_str=backtest_start_date_str,
        capital_base_float=capital_base_float,
    )


def build_capacity_analysis_inputs(
    show_display_bool: bool = False,
    backtest_start_date_str: str | None = None,
    capital_base_float: float = 100_000.0,
    end_date_str: str | None = None,
) -> dict[str, object]:
    config_obj = (
        DEFAULT_CONFIG
        if end_date_str is None
        else replace(DEFAULT_CONFIG, end_date_str=end_date_str)
    )
    return build_standard_fallback_vix_cash_capacity_analysis_inputs(
        strategy_name_str=STRATEGY_NAME_STR,
        config=config_obj,
        base_data_loader_fn=get_defense_first_data,
        show_display_bool=show_display_bool,
        backtest_start_date_str=backtest_start_date_str,
        capital_base_float=capital_base_float,
    )


def build_execution_timing_analysis_inputs() -> dict[str, object]:
    return build_standard_fallback_vix_cash_execution_timing_analysis_inputs(
        strategy_name_str=STRATEGY_NAME_STR,
        config=DEFAULT_CONFIG,
        base_data_loader_fn=get_defense_first_data,
    )


if __name__ == "__main__":
    run_variant()
