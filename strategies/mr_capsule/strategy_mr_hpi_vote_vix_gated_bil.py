"""HPI-G (MR capsule, HPI 2/3/5 vote) with all idle cash in BIL / T-bills.

HPI 2/3/5 vote rules (PIT S&P 500, HPI < 30 on >= 2 horizons, IBS < 0.10, Close > SMA200, Turnover rank, 10 slots,
exit IBS > 0.90 or RSI2 > 90); new entries only while the shared VIX stress gate is open; exits never gated.
Idle cash: BIL only, whether the gate is open or closed (bought weekly / at gate switches, sold when the day's orders
need the cash). The research "T-bills" parking with BIL's real trades, costs and withholding; Claude's recommended
parking after the 2026-10-04 build.
Pod rules, gate, parking mechanics and the full caveat list: strategies/mr_capsule/hpi_vote_vix_gated.py. Build record and the
SPMO-vs-BIL evidence: docs/research/MR_CAPSULE_20261003.md ("Build record (2026-10-04)").
Variant caveat: BIL dividends carry the house 25% withholding and BIL trades pay the engine's 2.5 bps (spread about
1 bp): conservative, about 0.4 pp/yr of capsule CAGR together (estimate); idle cash earns 0% before 2007-05-30.
"""

from __future__ import annotations

from strategies.mr_capsule.hpi_vote_vix_gated import (
    build_hpi_capsule_capacity_analysis_inputs,
    build_hpi_capsule_execution_timing_analysis_inputs,
    run_hpi_capsule_pod,
)

STRATEGY_NAME_STR = "strategy_mr_hpi_vote_vix_gated_bil"


def run_variant(
    show_display_bool: bool = True,
    save_results_bool: bool = True,
    output_dir_str: str = "results",
    backtest_start_date_str: str = "2004-01-01",
    capital_base_float: float = 100_000.0,
    end_date_str: str | None = None,
    slippage_float: float = 0.00025,
):
    return run_hpi_capsule_pod(
        strategy_name_str=STRATEGY_NAME_STR,
        parking_enabled_bool=True,
        spmo_parking_enabled_bool=False,
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
    return build_hpi_capsule_capacity_analysis_inputs(
        strategy_name_str=STRATEGY_NAME_STR,
        parking_enabled_bool=True,
        spmo_parking_enabled_bool=False,
        show_display_bool=show_display_bool,
        backtest_start_date_str=backtest_start_date_str,
        capital_base_float=capital_base_float,
        end_date_str=end_date_str,
    )


def build_execution_timing_analysis_inputs() -> dict[str, object]:
    return build_hpi_capsule_execution_timing_analysis_inputs(
        strategy_name_str=STRATEGY_NAME_STR,
        parking_enabled_bool=True,
        spmo_parking_enabled_bool=False,
    )


if __name__ == "__main__":
    run_variant()
