"""
Defense First BTAL fallback-UPRO variant with a diversified month-end VRP cash gate.

Fallback overlay:
    breach_count_m = sum_L 1[rvL_m >= VIX_m],  L in {10, 15, 20}
    cash_frac_m = breach_count_m / 3
    upro_frac_m = 1 - cash_frac_m
"""

from __future__ import annotations

try:
    from strategies.taa_df.strategy_taa_df import get_defense_first_data
    from strategies.taa_df.strategy_taa_df_btal_fallback_upro import DEFAULT_CONFIG as BASE_CONFIG
    from strategies.taa_df.strategy_taa_df_fallback_vix_cash_multi_rv_variant_utils import (
        apply_multi_rv_cash_gate_to_month_end_weight_df,
        build_multi_rv_cash_overlay_weight_frames,
        build_vix_cash_variant_config,
        compute_daily_multi_rv_signal_df,
        get_standard_fallback_vix_cash_multi_rv_data,
        load_helper_close_ser,
        run_standard_fallback_vix_cash_multi_rv_variant,
        rv_lookback_day_tuple,
        sample_month_end_multi_rv_signal_df,
    )
except ModuleNotFoundError:
    from strategy_taa_df import get_defense_first_data
    from strategy_taa_df_btal_fallback_upro import DEFAULT_CONFIG as BASE_CONFIG
    from strategy_taa_df_fallback_vix_cash_multi_rv_variant_utils import (
        apply_multi_rv_cash_gate_to_month_end_weight_df,
        build_multi_rv_cash_overlay_weight_frames,
        build_vix_cash_variant_config,
        compute_daily_multi_rv_signal_df,
        get_standard_fallback_vix_cash_multi_rv_data,
        load_helper_close_ser,
        run_standard_fallback_vix_cash_multi_rv_variant,
        rv_lookback_day_tuple,
        sample_month_end_multi_rv_signal_df,
    )


DEFAULT_CONFIG = build_vix_cash_variant_config(BASE_CONFIG)


def get_defense_first_multi_rv_cash_data(config=DEFAULT_CONFIG):
    return get_standard_fallback_vix_cash_multi_rv_data(
        config=config,
        base_data_loader_fn=get_defense_first_data,
    )


def run_variant(
    show_display_bool: bool = True,
    save_results_bool: bool = True,
    output_dir_str: str = "results",
):
    return run_standard_fallback_vix_cash_multi_rv_variant(
        strategy_name_str="strategy_taa_df_btal_fallback_upro_vix_cash_multi_rv",
        config=DEFAULT_CONFIG,
        base_data_loader_fn=get_defense_first_data,
        show_display_bool=show_display_bool,
        save_results_bool=save_results_bool,
        output_dir_str=output_dir_str,
    )


if __name__ == "__main__":
    run_variant()
