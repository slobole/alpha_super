"""
Defense First fallback-QLD variant with a month-end VRP cash gate.

Fallback overlay:
    if rv20_m < VIX_m:
        keep QLD fallback weight
    else:
        set QLD fallback weight to 0 and leave the residual as cash
"""

from __future__ import annotations

try:
    from strategies.taa_df.strategy_taa_df import get_defense_first_data
    from strategies.taa_df.strategy_taa_df_fallback_qld import DEFAULT_CONFIG as BASE_CONFIG
    from strategies.taa_df.strategy_taa_df_fallback_vix_cash_variant_utils import (
        build_vix_cash_variant_config,
        run_standard_fallback_vix_cash_variant,
    )
except ModuleNotFoundError:
    from strategy_taa_df import get_defense_first_data
    from strategy_taa_df_fallback_qld import DEFAULT_CONFIG as BASE_CONFIG
    from strategy_taa_df_fallback_vix_cash_variant_utils import (
        build_vix_cash_variant_config,
        run_standard_fallback_vix_cash_variant,
    )


DEFAULT_CONFIG = build_vix_cash_variant_config(BASE_CONFIG)


def run_variant(
    show_display_bool: bool = True,
    save_results_bool: bool = True,
    output_dir_str: str = "results",
):
    return run_standard_fallback_vix_cash_variant(
        strategy_name_str="strategy_taa_df_fallback_qld_vix_cash",
        config=DEFAULT_CONFIG,
        base_data_loader_fn=get_defense_first_data,
        show_display_bool=show_display_bool,
        save_results_bool=save_results_bool,
        output_dir_str=output_dir_str,
    )


if __name__ == "__main__":
    run_variant()
