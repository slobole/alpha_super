"""
Defense First no-BTAL 1/n fallback-TQQQ variant with a month-end VRP cash gate.

This is the no-BTAL counterpart of
`strategy_taa_df_btal_1n_fallback_tqqq_vix_cash`.

Only intended defensive-basket and slot-size changes:

    old defensive assets = ("GLD", "UUP", "TLT", "DBC", "BTAL")
    new defensive assets = ("GLD", "UUP", "TLT", "DBC")

    old slot weight = 1 / 5 = 0.20
    new slot weight = 1 / 4 = 0.25

For each defensive asset i at month-end m:

    if momentum_score_{i,m} > cash_return_m:
        w_{i,m} = 0.25
    else:
        w_{TQQQ,m} += 0.25

Fallback overlay:
    if rv20_m < VIX_m:
        keep TQQQ fallback weight
    else:
        set TQQQ fallback weight to 0 and leave the residual as cash

The momentum signal, default costs, prior-close sizing, and month-end decision
to next-month first-tradable-open execution timing are unchanged.
"""

from __future__ import annotations

from dataclasses import replace

try:
    from strategies.taa_df.strategy_taa_df import (
        DEFAULT_CONFIG as TAA_BASE_CONFIG,
        get_defense_first_data,
    )
    from strategies.taa_df.strategy_taa_df_fallback_variant_utils import (
        build_fallback_variant_config,
    )
    from strategies.taa_df.strategy_taa_df_fallback_vix_cash_variant_utils import (
        build_standard_fallback_vix_cash_capacity_analysis_inputs,
        build_standard_fallback_vix_cash_execution_timing_analysis_inputs,
        build_vix_cash_variant_config,
        run_standard_fallback_vix_cash_variant,
    )
except ModuleNotFoundError:
    from strategy_taa_df import (
        DEFAULT_CONFIG as TAA_BASE_CONFIG,
        get_defense_first_data,
    )
    from strategy_taa_df_fallback_variant_utils import (
        build_fallback_variant_config,
    )
    from strategy_taa_df_fallback_vix_cash_variant_utils import (
        build_standard_fallback_vix_cash_capacity_analysis_inputs,
        build_standard_fallback_vix_cash_execution_timing_analysis_inputs,
        build_vix_cash_variant_config,
        run_standard_fallback_vix_cash_variant,
    )


STRATEGY_NAME_STR = "strategy_taa_df_1n_fallback_tqqq_vix_cash"
NO_BTAL_DEFENSIVE_ASSET_TUPLE = ("GLD", "UUP", "TLT", "DBC")
NO_BTAL_1N_RANK_WEIGHT_TUPLE = (0.25, 0.25, 0.25, 0.25)

NO_BTAL_1N_CONFIG = replace(
    TAA_BASE_CONFIG,
    defensive_asset_list=NO_BTAL_DEFENSIVE_ASSET_TUPLE,
    rank_weight_vec=NO_BTAL_1N_RANK_WEIGHT_TUPLE,
)
TQQQ_FALLBACK_CONFIG = build_fallback_variant_config(
    NO_BTAL_1N_CONFIG,
    "TQQQ",
)
DEFAULT_CONFIG = build_vix_cash_variant_config(TQQQ_FALLBACK_CONFIG)


def run_variant(
    show_display_bool: bool = True,
    save_results_bool: bool = True,
    output_dir_str: str = "results",
    backtest_start_date_str: str | None = None,
    capital_base_float: float = 100_000.0,
    end_date_str: str | None = None,
):
    config_obj = DEFAULT_CONFIG if end_date_str is None else replace(DEFAULT_CONFIG, end_date_str=end_date_str)
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
    config_obj = DEFAULT_CONFIG if end_date_str is None else replace(DEFAULT_CONFIG, end_date_str=end_date_str)
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
