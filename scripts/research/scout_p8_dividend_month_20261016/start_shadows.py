"""P8, S8: start the first shadow, before any forward data exists.

The dividend month premium (row 52) is REJECTED at S4, so by the grade rules it opens no vault and gets no shadow; this
script shows the S7 refusal. The first shadow is the forward line the CORE5 vehicle study left (row 45, retro):
CORE5 with QQQ in the equity sleeve on its own QQQ signal (variant A2). Expected process = its full daily history before
the start (all of it was seen by the study, so it is a yardstick, not evidence); benchmark Sharpe 0.

    PYTHONUTF8=1 .venv/Scripts/python.exe scripts/research/scout_p8_dividend_month_20261016/start_shadows.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import pandas as pd

from alpha.scout.ledger import Ledger
from alpha.scout.shadow import shadow_rows, start_shadow
from alpha.scout.specs import core5_vehicle
from alpha.scout.vault import VaultRefusal, open_vault

START_DATE_STR = "2026-10-17"  # the day after both registrations; the first session is 2026-10-19


def main() -> None:
    ledger = Ledger()
    try:
        open_vault(ledger, "dividend_month_premium_20261016", {"lookback_rule_str": "m3_or_m12", "weighting_str": "EW"})
    except VaultRefusal as refusal:
        print(f"S7 refused as designed: {refusal}")

    if core5_vehicle.REGISTRATION_ID_STR in shadow_rows(ledger):
        print("CORE5 QQQ shadow already started.")
    else:
        daily_ser = core5_vehicle.daily_return_ser("QQQ")
        expected_ser = daily_ser.loc[daily_ser.ne(0).idxmax(): pd.Timestamp(START_DATE_STR) - pd.Timedelta(days=1)]
        row_dict = start_shadow(
            ledger, core5_vehicle.REGISTRATION_ID_STR, {"equity_symbol_str": "QQQ", "signal_str": "own (QQQ)", "core5_config_str": "LIVE_CONFIG"},
            expected_ser, "alpha.scout.specs.core5_vehicle.daily_return_ser('QQQ'), parity weights engine, $1M, 2007-09..2026-10 (seen)",
            start_date_str=START_DATE_STR, benchmark_annual_sharpe_float=0.0,
            note_str=("Forward line from core5_equity_vehicle_v2 (variant A2). Replaces the August QQQ/SPY-signal forward line, "
                      "which failed the 2000-02 check. Research only; no paper or live route."),
        )
        print(f"shadow_start row {row_dict['row_id_int']}: start {row_dict['start_date_str']}, expected Sharpe "
              f"{row_dict['expected_annual_sharpe_float']:.2f}, MinTRL {row_dict['min_trl_months_float']:.0f} months")
    print(f"ledger chain OK: {ledger.verify()} rows")


if __name__ == "__main__":
    main()
