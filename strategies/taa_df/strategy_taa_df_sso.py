"""
Defense First tactical allocation with an SSO fallback sleeve.

This variant preserves the exact signal and execution logic from
`strategy_taa_df.py` and changes only the fallback configuration:

    fallback_asset = "SSO"

Because SSO starts trading on 2006-06-21, the execution dataset must satisfy:

    start_date >= 2006-06-21

Otherwise the backtest would request fills on pre-inception bars with missing
open prices.
"""

from __future__ import annotations

from dataclasses import replace

try:
    from strategies.taa_df.strategy_taa_df import (
        DefenseFirstConfig,
        DefenseFirstStrategy,
        get_defense_first_data,
        run_defense_first_variant,
    )
except ModuleNotFoundError:
    from strategy_taa_df import (
        DefenseFirstConfig,
        DefenseFirstStrategy,
        get_defense_first_data,
        run_defense_first_variant,
    )


sso_inception_date_str = "2006-06-21"

DEFAULT_CONFIG = DefenseFirstConfig(
    fallback_asset="SSO",
    start_date_str=sso_inception_date_str,
)


def run_variant(
    show_display_bool: bool = True,
    save_results_bool: bool = True,
    output_dir_str: str = "results",
    backtest_start_date_str: str | None = None,
    capital_base_float: float = 100_000.0,
    end_date_str: str | None = None,
) -> DefenseFirstStrategy:
    config = DEFAULT_CONFIG if end_date_str is None else replace(DEFAULT_CONFIG, end_date_str=end_date_str)
    return run_defense_first_variant(
        strategy_name_str="strategy_taa_df_sso",
        config=config,
        data_loader_fn=get_defense_first_data,
        show_display_bool=show_display_bool,
        save_results_bool=save_results_bool,
        output_dir_str=output_dir_str,
        backtest_start_date_str=backtest_start_date_str,
        capital_base_float=capital_base_float,
    )


if __name__ == "__main__":
    run_variant()
