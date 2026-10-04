"""DV2-G (MR capsule) with SPMO parking while the VIX gate is closed, BIL otherwise (PM_READY, no live route).

DV2 wired rules (S&P 500, DV2 < 10, Close > SMA200, 126-day return > 5%, NATR rank, 10 slots, exit Close > prior
High); new entries only while the shared VIX stress gate is open; exits never gated.
Idle cash (the capsule spec, owner decision 2026-10-04): while the gate is closed, SPMO at min(1, 8% / 20-day realised
volatility) of the idle value, set weekly and at gate switches, only once SPMO traded on each of the last 20 sessions
(from 2018); the rest, and everything while the gate is open, in BIL (bought weekly / at gate switches, sold when the
day's orders need the cash).
Pod rules, gate, parking mechanics and the full caveat list: strategies/mr_capsule/dv2_vix_gated.py. Build record and the
SPMO-vs-BIL evidence: docs/research/MR_CAPSULE_20261003.md ("Build record (2026-10-04)").
Variant caveat: from 2018 (SPMO tradable) SPMO parking adds about 0.5 pp/yr of CAGR but costs about 0.04 Sharpe and
2 pp of drawdown against BIL only; its research edge came from 2015-11..2018-01, when SPMO did not trade most days.
"""

from __future__ import annotations

from strategies.mr_capsule.dv2_vix_gated import run_dv2_capsule_pod

STRATEGY_NAME_STR = "strategy_mr_dv2_vix_gated_spmo"


def run_variant(
    show_display_bool: bool = True,
    save_results_bool: bool = True,
    output_dir_str: str = "results",
    backtest_start_date_str: str = "2004-01-01",
    capital_base_float: float = 100_000.0,
    end_date_str: str | None = None,
    slippage_float: float = 0.00025,
):
    return run_dv2_capsule_pod(
        strategy_name_str=STRATEGY_NAME_STR,
        parking_enabled_bool=True,
        spmo_parking_enabled_bool=True,
        show_display_bool=show_display_bool,
        save_results_bool=save_results_bool,
        output_dir_str=output_dir_str,
        backtest_start_date_str=backtest_start_date_str,
        capital_base_float=capital_base_float,
        end_date_str=end_date_str,
        slippage_float=slippage_float,
    )


if __name__ == "__main__":
    run_variant()
