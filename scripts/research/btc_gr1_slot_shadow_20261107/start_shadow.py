"""Register the 5% BTC slot beside GR1 and start the S8 shadow of its sleeve (IBIT 20% vol). Runs once.

- Registration btc_vol_sleeve_gr1_slot_20261107 (retro: the idea was studied in Pakal on 2011-2026 BTC history).
- Grade: WATCHLIST (forward shadow; the Scout card stations were not run).
- Shadow from 2026-11-09 (the first session after today). Provider alpha.scout.specs.btc_vol_sleeve:shadow_daily_return_ser.
  Expected process: the same provider's IBIT-era history before the start (2024-04..2026-11, all seen), a yardstick.
  Benchmark Sharpe 0.

    PYTHONDONTWRITEBYTECODE=1 .venv/Scripts/python.exe scripts/research/btc_gr1_slot_shadow_20261107/start_shadow.py [--dry-run]
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_PATH = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT_PATH))

import pandas as pd  # noqa: E402

from alpha.scout.fdr_report import record_station_verdict  # noqa: E402
from alpha.scout.ledger import Ledger  # noqa: E402
from alpha.scout.registration import Registration, register, registration_rows  # noqa: E402
from alpha.scout.shadow import shadow_rows, start_shadow  # noqa: E402
from alpha.scout.specs import btc_vol_sleeve as S  # noqa: E402

START_DATE_STR = "2026-11-09"
REGISTRATION = Registration(
    registration_id_str=S.REGISTRATION_ID_STR,
    family_id_str="optimized_low_risk_allocation",
    hypothesis_str="A Bitcoin sleeve sized to 20% volatility (IBIT + BIL, weekly, 10% band) held at 5% of GR1 raises GR1's excess Sharpe without deepening its drawdown.",
    mechanism_str="A low-correlation asset with a large risk premium; volatility sizing keeps its risk share small and stable.",
    expected_sign_and_location_str="sleeve excess return > 0 forward; GR1 + 5% sleeve Sharpe >= GR1 at the review",
    hypothesis_class_str="W",
    universe_str="IBIT, BIL",
    horizon_str="daily, forward from 2026-11-09",
    schedule_str="last XNYS session of each ISO week; fill next open; 10% no-trade band",
    execution_str="Scout weights engine, adjusted fractional units, slippage 5 bp + $0.005/share (min $1)",
    param_grid_dict={"vol_target": (0.20,), "band": (0.10,)},
    primary_metric_str="forward sleeve excess Sharpe and GR1 + 5% sleeve vs GR1 at review",
    kill_criteria_str="card RED (Cold Blood Index or monthly CUSUM), or at the 12-month review GR1 + 5% sleeve does not beat GR1",
    source_str="pakal-research/reports/btc_sleeve_candidates_study (SIMPLE sleeve, 5% slot) and the 2026-11-07 recheck (btc_sleeve_candidates/followup_gr1_recent.py)",
    retro_bool=True,
    prior_trials_int=40,
    universe_choice_str="BTC chosen by the owner before results; IBIT chosen as the listed vehicle",
    universe_chosen_after_results_bool=False,
)
GRADE_CHECK_LIST = [
    ["grade", "WATCHLIST (no card; forward shadow)", (
        "Pakal btc_sleeve_candidates_study: GR1 + 5% sleeve 1.35 -> 1.47 excess Sharpe 2011-08..2026-08 (P 1.00), ETH P 0.92; "
        "but 2021-26 +0.01 (P 0.46) and since IBIT -0.02. Risk: GR1 vol 12.5% -> 12.1%, maxDD -12.0% -> -12.3%, P(DD < -20%) 3.1% -> 2.2%. "
        "IBIT-era sleeve 2024-04..2026-11: 2.6%/yr at 20% vol. Card stations S3-S6 not run.")],
]


def main() -> None:
    dry_run_bool = "--dry-run" in sys.argv
    ledger = Ledger()
    if S.REGISTRATION_ID_STR in shadow_rows(ledger):
        raise SystemExit(f"{S.REGISTRATION_ID_STR} already has a shadow.")
    daily_ser = S.shadow_daily_return_ser({"frozen_config_dict": S.frozen_config_dict()})
    expected_ser = daily_ser.loc[: pd.Timestamp(START_DATE_STR) - pd.Timedelta(days=1)]
    print(f"expected {expected_ser.index[0].date()} to {expected_ser.index[-1].date()}, {len(expected_ser)} sessions")
    if dry_run_bool:
        REGISTRATION.validate()
        print("dry run: nothing written")
        return
    if S.REGISTRATION_ID_STR not in registration_rows(ledger):
        print("registration row", register(ledger, REGISTRATION)["row_id_int"])
    if not any(r.get("registration_id_str") == S.REGISTRATION_ID_STR and r.get("station_str") == "grade" for r in ledger.rows("station_verdict")):
        print("grade row", record_station_verdict(ledger, S.REGISTRATION_ID_STR, "grade", GRADE_CHECK_LIST, retro_bool=True,
                                                  source_str="pakal-research/reports/btc_sleeve_candidates_study")["row_id_int"])
    row_dict = start_shadow(ledger, S.REGISTRATION_ID_STR, S.frozen_config_dict(), expected_ser,
                            "alpha.scout.specs.btc_vol_sleeve frozen rule before the start (IBIT era, all seen)",
                            start_date_str=START_DATE_STR, benchmark_annual_sharpe_float=0.0,
                            note_str="Slot = 5% of GR1; the book is recombined with GR1 rebuilt on newest data at the 12-month review (Lab ends 2026-08-19).")
    print("shadow row", row_dict["row_id_int"], "start", row_dict["start_date_str"], "MinTRL months",
          None if row_dict["min_trl_months_float"] is None else round(row_dict["min_trl_months_float"], 1),
          "expected Sharpe", round(row_dict["expected_annual_sharpe_float"], 3))


if __name__ == "__main__":
    main()
