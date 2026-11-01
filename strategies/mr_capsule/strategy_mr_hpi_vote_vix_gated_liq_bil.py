"""HPI-G (MR capsule) with all idle cash in BIL and each new entry sized by the stock's liquidity.

HPI 2/3/5 vote wired rules (HPI < 30 on at least 2 horizons, IBS < 0.10, Close > SMA200, S&P 500 member, Turnover rank,
10 slots; exit IBS > 0.90, RSI2 > 90 or leaving the index); new entries only while the shared VIX stress gate is open;
exits never gated; idle cash in BIL.
Entry = 1/10 of NAV x sqrt(ADV63 over a rolling typical candidate), between half and double; stocks stop at 99% of NAV.
Pod rules and caveats: strategies/mr_capsule/hpi_vote_vix_gated.py. Sizing rule and caveats: strategies/mr_capsule/liquidity_sizing.py.
Evidence: docs/research/GROWTH_MR_PRODUCTS_20261101.md. The 1/N pod stays available: strategy_mr_hpi_vote_vix_gated_bil.py. An account
moves between the two only when flat, under a new pod id: docs/live/MR_CAPSULE_LIQ_SIZING_WIRING_20261101.md.
"""

from __future__ import annotations

from strategies.mr_capsule.hpi_vote_vix_gated_liq import (
    build_hpi_capsule_liquidity_capacity_analysis_inputs,
    build_hpi_capsule_liquidity_execution_timing_analysis_inputs,
    run_hpi_capsule_liquidity_pod,
)

STRATEGY_NAME_STR = "strategy_mr_hpi_vote_vix_gated_liq_bil"


def run_variant(
    show_display_bool: bool = True,
    save_results_bool: bool = True,
    output_dir_str: str = "results",
    backtest_start_date_str: str = "2004-01-01",
    capital_base_float: float = 100_000.0,
    end_date_str: str | None = None,
    slippage_float: float = 0.00025,
):
    return run_hpi_capsule_liquidity_pod(
        strategy_name_str=STRATEGY_NAME_STR,
        show_display_bool=show_display_bool,
        save_results_bool=save_results_bool,
        output_dir_str=output_dir_str,
        backtest_start_date_str=backtest_start_date_str,
        capital_base_float=capital_base_float,
        end_date_str=end_date_str,
        slippage_float=slippage_float,
    )


def build_capacity_analysis_inputs(
    show_display_bool: bool = False,
    backtest_start_date_str: str = "2004-01-01",
    capital_base_float: float = 100_000.0,
    end_date_str: str | None = None,
) -> dict[str, object]:
    return build_hpi_capsule_liquidity_capacity_analysis_inputs(
        strategy_name_str=STRATEGY_NAME_STR,
        show_display_bool=show_display_bool,
        backtest_start_date_str=backtest_start_date_str,
        capital_base_float=capital_base_float,
        end_date_str=end_date_str,
    )


def build_execution_timing_analysis_inputs() -> dict[str, object]:
    return build_hpi_capsule_liquidity_execution_timing_analysis_inputs(strategy_name_str=STRATEGY_NAME_STR)


if __name__ == "__main__":
    run_variant()
