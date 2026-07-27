"""Research-only stateful HPI long strategy for point-in-time Nasdaq-100 members."""

from __future__ import annotations

from strategies.hpi.stateful_long import NATR_FIELD_STR, run_hpi_variant


STRATEGY_NAME_STR = "strategy_mr_hpi_nasdaq100_ibs_rsi_exit"


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
        indexname_str="Nasdaq 100",
        benchmark_symbol_str="$NDX",
        ranking_field_str=NATR_FIELD_STR,
        show_display_bool=show_display_bool,
        save_results_bool=save_results_bool,
        output_dir_str=output_dir_str,
        backtest_start_date_str=backtest_start_date_str,
        capital_base_float=capital_base_float,
        end_date_str=end_date_str,
    )


if __name__ == "__main__":
    run_variant()
