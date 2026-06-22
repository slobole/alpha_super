"""
Defense First fallback-TQQQ VIX-cash variant with ZROZ replacing TLT.

Only intended semantic change versus `strategy_taa_df_fallback_tqqq_vix_cash`:

    defensive_asset_list = ("GLD", "UUP", "ZROZ", "DBC")

The TQQQ fallback sleeve, month-end VIX cash gate, costs, benchmark, and
month-end decision -> next-month first-tradable-open execution timing are
unchanged.
"""

from __future__ import annotations

from dataclasses import replace

try:
    from strategies.taa_df.strategy_taa_df import get_defense_first_data
    from strategies.taa_df.strategy_taa_df_fallback_tqqq import DEFAULT_CONFIG as BASE_CONFIG
    from strategies.taa_df.strategy_taa_df_fallback_vix_cash_variant_utils import (
        build_vix_cash_variant_config,
        run_standard_fallback_vix_cash_variant,
    )
except ModuleNotFoundError:
    from strategy_taa_df import get_defense_first_data
    from strategy_taa_df_fallback_tqqq import DEFAULT_CONFIG as BASE_CONFIG
    from strategy_taa_df_fallback_vix_cash_variant_utils import (
        build_vix_cash_variant_config,
        run_standard_fallback_vix_cash_variant,
    )


zroz_defensive_asset_tuple = tuple(
    "ZROZ" if asset_str == "TLT" else asset_str
    for asset_str in BASE_CONFIG.defensive_asset_list
)
DEFAULT_CONFIG = build_vix_cash_variant_config(
    replace(BASE_CONFIG, defensive_asset_list=zroz_defensive_asset_tuple)
)


def run_variant(
    show_display_bool: bool = True,
    save_results_bool: bool = True,
    output_dir_str: str = "results",
):
    return run_standard_fallback_vix_cash_variant(
        strategy_name_str="strategy_taa_df_fallback_tqqq_vix_cash_zroz",
        config=DEFAULT_CONFIG,
        base_data_loader_fn=get_defense_first_data,
        show_display_bool=show_display_bool,
        save_results_bool=save_results_bool,
        output_dir_str=output_dir_str,
    )


if __name__ == "__main__":
    run_variant()
