"""Research-only HPI 2/3/5 vote with relative liquidity for the PIT S&P 500."""

from __future__ import annotations

from strategies.hpi.stateful_long import (
    ENTRY_HORIZON_VOTE_STR,
    LIQUIDITY_RELATIVE_STR,
    TURNOVER_FIELD_STR,
    run_hpi_variant,
)


STRATEGY_NAME_STR = "strategy_mr_hpi_sp500_2_3_5_vote_relative_liquidity"


def run_variant(
    show_display_bool: bool = True,
    save_results_bool: bool = True,
    output_dir_str: str = "results",
    backtest_start_date_str: str = "2004-01-01",
    capital_base_float: float = 100_000.0,
    end_date_str: str | None = None,
):
    return run_hpi_variant(
        strategy_name_str=STRATEGY_NAME_STR,
        indexname_str="S&P 500",
        benchmark_symbol_str="$SPX",
        ranking_field_str=TURNOVER_FIELD_STR,
        show_display_bool=show_display_bool,
        save_results_bool=save_results_bool,
        output_dir_str=output_dir_str,
        backtest_start_date_str=backtest_start_date_str,
        capital_base_float=capital_base_float,
        end_date_str=end_date_str,
        entry_mode_str=ENTRY_HORIZON_VOTE_STR,
        liquidity_mode_str=LIQUIDITY_RELATIVE_STR,
    )


if __name__ == "__main__":
    run_variant()
