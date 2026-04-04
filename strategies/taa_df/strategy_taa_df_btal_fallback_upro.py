"""
Defense First BTAL fallback variant.

Old behavior:
    fallback_asset = fallback_asset^{base}

New behavior:
    fallback_asset = "UPRO"

Start-date guard:

    start_date^{variant} = max(start_date^{base}, first_UPRO_date)
"""

from __future__ import annotations

try:
    from strategies.taa_df.strategy_taa_df_btal import DEFAULT_CONFIG as BASE_CONFIG, get_defense_first_data
    from strategies.taa_df.strategy_taa_df_fallback_variant_utils import build_fallback_variant_config, run_standard_fallback_variant
except ModuleNotFoundError:
    from strategy_taa_df_btal import DEFAULT_CONFIG as BASE_CONFIG, get_defense_first_data
    from strategy_taa_df_fallback_variant_utils import build_fallback_variant_config, run_standard_fallback_variant


fallback_asset_str = "UPRO"
DEFAULT_CONFIG = build_fallback_variant_config(BASE_CONFIG, fallback_asset_str)


def run_variant(
    show_display_bool: bool = True,
    save_results_bool: bool = True,
    output_dir_str: str = "results",
):
    return run_standard_fallback_variant(
        strategy_name_str="strategy_taa_df_btal_fallback_upro",
        config=DEFAULT_CONFIG,
        data_loader_fn=get_defense_first_data,
        show_display_bool=show_display_bool,
        save_results_bool=save_results_bool,
        output_dir_str=output_dir_str,
    )


if __name__ == "__main__":
    run_variant()
