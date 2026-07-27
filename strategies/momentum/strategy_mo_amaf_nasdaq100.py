"""Research-only Nasdaq-100 Adaptive Moving-Average Factor BENCH variant."""

from __future__ import annotations

from strategies.momentum.adaptive_moving_average_factor import (
    AdaptiveMovingAverageFactorConfig,
    AdaptiveMovingAverageFactorStrategy,
    run_amaf_variant,
)


DEFAULT_CONFIG = AdaptiveMovingAverageFactorConfig(
    strategy_name_str="strategy_mo_amaf_nasdaq100",
    variant_key_str="amaf_nasdaq100_top_quintile_long_only",
    indexname_str="Nasdaq 100",
    source_panel_indexname_str="Russell 3000",
    benchmark_list=("$NDX",),
    min_eligible_count_int=50,
)


def run_variant(
    show_display_bool: bool = True,
    save_results_bool: bool = True,
    output_dir_str: str = "results",
    backtest_start_date_str: str | None = None,
    capital_base_float: float | None = None,
    end_date_str: str | None = None,
    audit_override_bool: bool | None = False,
) -> AdaptiveMovingAverageFactorStrategy:
    return run_amaf_variant(
        config_obj=DEFAULT_CONFIG,
        show_display_bool=show_display_bool,
        save_results_bool=save_results_bool,
        output_dir_str=output_dir_str,
        backtest_start_date_str=backtest_start_date_str,
        capital_base_float=capital_base_float,
        end_date_str=end_date_str,
        audit_override_bool=audit_override_bool,
    )


if __name__ == "__main__":
    run_variant()
