"""Write the two RETRO registrations of the P5 re-audition to the Scout ledger (idempotent: existing ids are skipped).

    uv run python scripts/research/scout_p5_reaudition_20261002/register.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from alpha.scout.family import NDX_GRID_DICT, TAA_GRID_DICT
from alpha.scout.ledger import Ledger
from alpha.scout.registration import (
    Registration,
    register,
    registration_rows,
)

REGISTRATION_LIST = [
    Registration(
        registration_id_str="taa_3x_reaudition_20261002",
        family_id_str="tactical_asset_allocation",
        hypothesis_str=(
            "Ranking five defensive asset-class ETFs by blended 1-3-6-12 month momentum, with weak slots moved to TQQQ "
            "and TQQQ held only while SPY realised volatility is below VIX, beats a volatility-targeted equal weight "
            "of the same six ETFs."
        ),
        mechanism_str=(
            "Momentum persistence across asset classes, plus a variance-risk-premium gate: leveraged equity is held "
            "only when implied volatility exceeds realised, which avoids most crash regimes."
        ),
        expected_sign_and_location_str="Positive active return; risk-on months through TQQQ, defensive months through GLD / TLT / UUP / BTAL.",
        hypothesis_class_str="W",
        universe_str="GLD, UUP, TLT, DBC, BTAL; fallback TQQQ",
        horizon_str="one month",
        schedule_str="decision at the last session of the month",
        execution_str="next session's open (parity weights engine, adjusted share units)",
        param_grid_dict={k: tuple(v) for k, v in TAA_GRID_DICT.items()},
        primary_metric_str="S5 MCPT (score SD, A9); S6 T-bill slot",
        kill_criteria_str="S8 CUSUM or Cold Blood Index red; an MCPT p above 0.05 keeps it at WATCHLIST",
        source_str="strategies/taa_df/strategy_taa_df_btal_fallback_tqqq_vix_cash.py (LIVE since 2026-04)",
        retro_bool=True,
        prior_trials_int=103,
        universe_choice_str=(
            "The Defense First ETF list, with BTAL and the TQQQ fallback added by the owner after studying variants "
            "(about 100 variant modules in strategies/taa_df)."
        ),
        universe_chosen_after_results_bool=True,
    ),
    Registration(
        registration_id_str="ndx_vxn_reaudition_20261002",
        family_id_str="equity_cross_sectional_momentum",
        hypothesis_str=(
            "Holding the top Nasdaq-100 members by 12-month return per unit of dollar ATR, above their 100-day average, "
            "sized down by VXN and flat when SPY is below its 200-day average, beats the same overlay on equal-weight members."
        ),
        mechanism_str="Cross-sectional momentum (under-reaction) among large growth stocks, with a market-regime and volatility overlay.",
        expected_sign_and_location_str="Positive active return from stock selection; overlay value concentrated in bear markets.",
        hypothesis_class_str="X",
        universe_str="Nasdaq-100 point-in-time members (exact membership)",
        horizon_str="one month",
        schedule_str="decision at the last session of the month",
        execution_str="next session's open (parity weights engine, historical share units)",
        param_grid_dict={k: tuple(v) for k, v in NDX_GRID_DICT.items()},
        primary_metric_str="S5 MCPT: stock selection (per-asset null, A8) and timing overlay (score SD, A9); S6 T-bill slot",
        kill_criteria_str="S8 CUSUM or Cold Blood Index red; an MCPT p above 0.05 keeps it at WATCHLIST",
        source_str="strategies/momentum/strategy_mo_atr_normalized_ndx_vxn_scaled.py (LIVE since 2026-05)",
        retro_bool=True,
        prior_trials_int=150,
        universe_choice_str=(
            "The Nasdaq-100 was the pod's universe from the start; other universes were explored only later "
            "(2026-09-26 robustness study)."
        ),
        universe_chosen_after_results_bool=False,
    ),
]


def main() -> None:
    ledger = Ledger()
    existing_dict = registration_rows(ledger)
    for registration in REGISTRATION_LIST:
        if registration.registration_id_str in existing_dict:
            print("exists", registration.registration_id_str)
            continue
        row_dict = register(ledger, registration)
        print("registered", registration.registration_id_str, "row", row_dict["row_id_int"])
    print("ledger rows verified:", ledger.verify())


if __name__ == "__main__":
    main()
