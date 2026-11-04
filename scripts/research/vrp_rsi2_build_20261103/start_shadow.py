"""Grade the locked VRP RSI2 rule and start its S8 shadow (SPEC_FROZEN.md). Runs once; refuses a second start.

- Grade: WATCHLIST, because the Scout card stations (S3 to S6 with MCPT) were not run for this retro rule; the evidence
  is the review checks in this folder and the identity gate.
- Shadow from 2026-11-04 (the day after today). Provider alpha.scout.specs.vrp_rsi2:shadow_daily_return_ser, which
  refuses when the rule in code changes. Expected process: the same provider's history before the start (all of it was
  seen, so it is a yardstick, not evidence), checked here against the saved engine run. Benchmark Sharpe 0.

    PYTHONDONTWRITEBYTECODE=1 .venv/Scripts/python.exe scripts/research/vrp_rsi2_build_20261103/start_shadow.py [--dry-run]
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_PATH = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT_PATH))

import pandas as pd  # noqa: E402

from alpha.scout.fdr_report import record_station_verdict  # noqa: E402
from alpha.scout.ledger import Ledger  # noqa: E402
from alpha.scout.shadow import shadow_rows, start_shadow  # noqa: E402
from alpha.scout.specs import vrp_rsi2  # noqa: E402

REGISTRATION_ID_STR = "vrp_rsi2_locked_retro_20261103"
START_DATE_STR = "2026-11-04"
ENGINE_RETURN_PATH = ROOT_PATH / "results" / "research" / "vrp_rsi2_build_20261103" / "engine_ret.parquet"
GRADE_CHECK_LIST = [
    ["grade", "WATCHLIST (no card; forward shadow)", (
        "Owner lock of 2026-11-03. Engine = Scout spec at 4e-16 (gate vrp_rsi2). 2011-10..2026-08, research replica at 5 bp "
        "with T-bills on idle cash: 15.5% / Sharpe 0.93 / -23.2%; engine with whole shares and IBKR fees about 0.90. RSI2 "
        "boost beats 400 of 400 placebo shifts. Threshold 0.96-1.05: Sharpe 0.82-1.01, no plateau. One session late 0.62. "
        "VX futures 2008 +56%. Fund harness, 10 points from CORE5: Growth Plus 20.3/1.48 to 20.9/1.51, share 0.88/0.83/0.46, "
        "house checks pass to 15 bp a side, fail at 20 bp and with the 2020 crash blanked. Card stations S3-S6 not run.")],
]


def main() -> None:
    dry_run_bool = "--dry-run" in sys.argv
    ledger = Ledger()
    if REGISTRATION_ID_STR in shadow_rows(ledger):
        raise SystemExit(f"{REGISTRATION_ID_STR} already has a shadow.")
    daily_ser = vrp_rsi2.shadow_daily_return_ser({"frozen_config_dict": vrp_rsi2.frozen_config_dict()})
    expected_ser = daily_ser.loc[: pd.Timestamp(START_DATE_STR) - pd.Timedelta(days=1)]
    engine_ser = pd.read_parquet(ENGINE_RETURN_PATH)["engine"]
    common_idx = engine_ser.index.intersection(expected_ser.index)
    max_diff_float = float((engine_ser.reindex(common_idx) - expected_ser.reindex(common_idx)).abs().max())
    print(f"expected {expected_ser.index[0].date()} to {expected_ser.index[-1].date()}, {len(expected_ser)} sessions; vs saved engine run "
          f"{len(common_idx)} sessions, max difference {max_diff_float:.1e}")
    if max_diff_float > 1e-9:
        raise SystemExit("The provider does not reproduce the engine run. Not starting.")
    if dry_run_bool:
        print("dry run: nothing written")
        return
    if not any(r.get("registration_id_str") == REGISTRATION_ID_STR and r.get("station_str") == "grade" for r in ledger.rows("station_verdict")):
        row_dict = record_station_verdict(ledger, REGISTRATION_ID_STR, "grade", GRADE_CHECK_LIST, retro_bool=True,
                                          source_str="scripts/research/vrp_rsi2_build_20261103/")
        print("grade row", row_dict["row_id_int"])
    row_dict = start_shadow(ledger, REGISTRATION_ID_STR, vrp_rsi2.frozen_config_dict(), expected_ser,
                            "alpha.scout.specs.vrp_rsi2 frozen rule before the start (engine-equivalent, 0% cash)",
                            start_date_str=START_DATE_STR, benchmark_annual_sharpe_float=0.0,
                            note_str="Owner lock 2026-11-03; adoption rule in scripts/research/vrp_rsi2_build_20261103/SPEC_FROZEN.md")
    print("shadow row", row_dict["row_id_int"], "start", row_dict["start_date_str"], "MinTRL months", round(row_dict["min_trl_months_float"], 1),
          "expected Sharpe", round(row_dict["expected_annual_sharpe_float"], 3))


if __name__ == "__main__":
    main()
