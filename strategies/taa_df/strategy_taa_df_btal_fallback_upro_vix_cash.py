"""
Defense First BTAL fallback-UPRO variant with a month-end VRP cash gate.

Old fallback behavior:
    fallback_asset_m = UPRO

New fallback behavior:
    if rv20_m < VIX_m:
        fallback_asset_m = UPRO
    else:
        fallback_asset_m = cash

Signal formulas
---------------
Let:

    ret_spy_t = C_spy_t / C_spy_{t-1} - 1

    rv20_t = std(ret_spy_{t-19:t}) * sqrt(252) * 100

Month-end overlay rule:

    if rv20_m < VIX_m:
        keep UPRO fallback weight
    else:
        set UPRO fallback weight to 0 and leave the residual as cash
"""

from __future__ import annotations

try:
    from strategies.taa_df.strategy_taa_df import get_defense_first_data
    from strategies.taa_df.strategy_taa_df_btal_fallback_upro import DEFAULT_CONFIG as BASE_CONFIG
    from strategies.taa_df.strategy_taa_df_fallback_vix_cash_variant_utils import (
        apply_vrp_cash_gate_to_month_end_weight_df,
        build_vix_cash_variant_config,
        build_vrp_cash_overlay_weight_frames,
        compute_daily_vrp_signal_df,
        get_standard_fallback_vix_cash_data,
        load_helper_close_ser,
        run_standard_fallback_vix_cash_variant,
        sample_month_end_vrp_signal_df,
    )
except ModuleNotFoundError:
    from strategy_taa_df import get_defense_first_data
    from strategy_taa_df_btal_fallback_upro import DEFAULT_CONFIG as BASE_CONFIG
    from strategy_taa_df_fallback_vix_cash_variant_utils import (
        apply_vrp_cash_gate_to_month_end_weight_df,
        build_vix_cash_variant_config,
        build_vrp_cash_overlay_weight_frames,
        compute_daily_vrp_signal_df,
        get_standard_fallback_vix_cash_data,
        load_helper_close_ser,
        run_standard_fallback_vix_cash_variant,
        sample_month_end_vrp_signal_df,
    )


DEFAULT_CONFIG = build_vix_cash_variant_config(BASE_CONFIG)


def get_defense_first_vrp_cash_data(config=DEFAULT_CONFIG):
    return get_standard_fallback_vix_cash_data(
        config=config,
        base_data_loader_fn=get_defense_first_data,
    )


def run_variant(
    show_display_bool: bool = True,
    save_results_bool: bool = True,
    output_dir_str: str = "results",
):
    return run_standard_fallback_vix_cash_variant(
        strategy_name_str="strategy_taa_df_btal_fallback_upro_vix_cash",
        config=DEFAULT_CONFIG,
        base_data_loader_fn=get_defense_first_data,
        show_display_bool=show_display_bool,
        save_results_bool=save_results_bool,
        output_dir_str=output_dir_str,
    )


if __name__ == "__main__":
    run_variant()
