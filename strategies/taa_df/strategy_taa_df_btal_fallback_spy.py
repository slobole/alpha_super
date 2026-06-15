"""
Defense First BTAL fallback variant.

Old behavior:
    fallback_asset = fallback_asset^{base}

New behavior:
    fallback_asset = "SPY"

Start-date guard:

    start_date^{variant} = max(start_date^{base}, first_SPY_date)
"""

from __future__ import annotations

from dataclasses import replace

try:
    from strategies.taa_df.strategy_taa_df_btal import DEFAULT_CONFIG as BASE_CONFIG, get_defense_first_data
    from strategies.taa_df.strategy_taa_df_fallback_variant_utils import (
        build_fallback_variant_config,
        build_standard_fallback_execution_timing_analysis_inputs,
        build_standard_fallback_friction_analysis_inputs,
        run_standard_fallback_friction_analysis,
        run_standard_fallback_variant,
    )
except ModuleNotFoundError:
    from strategy_taa_df_btal import DEFAULT_CONFIG as BASE_CONFIG, get_defense_first_data
    from strategy_taa_df_fallback_variant_utils import (
        build_fallback_variant_config,
        build_standard_fallback_execution_timing_analysis_inputs,
        build_standard_fallback_friction_analysis_inputs,
        run_standard_fallback_friction_analysis,
        run_standard_fallback_variant,
    )


STRATEGY_NAME_STR = "strategy_taa_df_btal_fallback_spy"
fallback_asset_str = "SPY"
DEFAULT_CONFIG = build_fallback_variant_config(BASE_CONFIG, fallback_asset_str)


def run_variant(
    show_display_bool: bool = True,
    save_results_bool: bool = True,
    output_dir_str: str = "results",
    backtest_start_date_str: str | None = None,
    capital_base_float: float = 100_000.0,
    end_date_str: str | None = None,
):
    config = DEFAULT_CONFIG if end_date_str is None else replace(DEFAULT_CONFIG, end_date_str=end_date_str)
    return run_standard_fallback_variant(
        strategy_name_str=STRATEGY_NAME_STR,
        config=config,
        data_loader_fn=get_defense_first_data,
        show_display_bool=show_display_bool,
        save_results_bool=save_results_bool,
        output_dir_str=output_dir_str,
        backtest_start_date_str=backtest_start_date_str,
        capital_base_float=capital_base_float,
    )


def build_friction_analysis_inputs(
    show_display_bool: bool = False,
    backtest_start_date_str: str | None = None,
    capital_base_float: float = 100_000.0,
    end_date_str: str | None = None,
) -> dict[str, object]:
    config = DEFAULT_CONFIG if end_date_str is None else replace(DEFAULT_CONFIG, end_date_str=end_date_str)
    return build_standard_fallback_friction_analysis_inputs(
        strategy_name_str=STRATEGY_NAME_STR,
        config=config,
        data_loader_fn=get_defense_first_data,
        show_display_bool=show_display_bool,
        backtest_start_date_str=backtest_start_date_str,
        capital_base_float=capital_base_float,
    )


def build_execution_timing_analysis_inputs() -> dict[str, object]:
    return build_standard_fallback_execution_timing_analysis_inputs(
        strategy_name_str=STRATEGY_NAME_STR,
        config=DEFAULT_CONFIG,
        data_loader_fn=get_defense_first_data,
    )


def run_friction_analysis(
    save_results_bool: bool = True,
    output_dir_str: str = "results",
    show_display_bool: bool = False,
    backtest_start_date_str: str | None = None,
    capital_base_float: float = 100_000.0,
    end_date_str: str | None = None,
):
    config = DEFAULT_CONFIG if end_date_str is None else replace(DEFAULT_CONFIG, end_date_str=end_date_str)
    return run_standard_fallback_friction_analysis(
        strategy_name_str=STRATEGY_NAME_STR,
        config=config,
        data_loader_fn=get_defense_first_data,
        save_results_bool=save_results_bool,
        output_dir_str=output_dir_str,
        show_display_bool=show_display_bool,
        backtest_start_date_str=backtest_start_date_str,
        capital_base_float=capital_base_float,
    )


if __name__ == "__main__":
    run_variant()
