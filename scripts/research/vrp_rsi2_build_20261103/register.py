"""Write the locked VRP RSI2 rule to the Scout ledger, before the engine module's results are read (SPEC_FROZEN.md).

    .venv/Scripts/python.exe scripts/research/vrp_rsi2_build_20261103/register.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from alpha.scout.ledger import Ledger  # noqa: E402
from alpha.scout.registration import Registration, register, registration_rows  # noqa: E402

REGISTRATION = Registration(
    registration_id_str="vrp_rsi2_locked_retro_20261103",
    family_id_str="volatility_premium_and_hedges",
    hypothesis_str=(
        "Short SVXY sized by VIX when the VIX curve is in contango (VIX/VIX3M below 0.98), halved when realised volatility "
        "is above implied, raised a quarter after an S&P 500 RSI(2) dip, and long VIXM when the curve is inverted and "
        "realised volatility is above implied, earns an excess Sharpe near 0.9 before 2026 and adds to the fund's Growth "
        "books when it takes 10 points from CORE5."
    ),
    mechanism_str=(
        "Implied volatility is usually priced above realised volatility, and VIX futures roll down in contango, so short "
        "front-month VIX futures earn a premium while the curve slopes up. An inverted curve with realised volatility above "
        "implied marks a crash in progress, when mid-term VIX futures pay. After a sharp index dip the premium is richest "
        "because implied volatility overshoots and mean-reverts within days."
    ),
    expected_sign_and_location_str=(
        "Positive excess Sharpe in 2011-18, 2019-22 and 2023+; positive in 2008 on the VX futures proxy (long-vol leg); "
        "losses in Feb 2018, Aug 2024 and Apr 2025 smaller than the sleeve's 2020 gain; Sharpe falls by about a third with "
        "one session of delay; the book gain fails when the 2020 crash is replaced by T-bills."
    ),
    hypothesis_class_str="W",
    universe_str="SVXY and VIXM traded; signals $VIX, $VIX3M, $SPX price index, SPY total return; Cboe VX constant-maturity futures for 2006-2011 only",
    horizon_str="days to weeks; about 5.8x NAV turnover a year",
    schedule_str="daily decision after the close, trades only when a target turns on or off or moves 5 points",
    execution_str="Open_(T+1) market on open; Open_(T+2) one-session-late stress",
    param_grid_dict={
        "contango_threshold": (0.90, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1.00, 1.01, 1.0164, 1.02, 1.03, 1.05),
        "rsi_boost": ("off", "rsi2_15", "rsi2_20", "rsi2_25", "rsi2_30", "rsi2_35", "rsi3_20", "rsi3_25", "rsi3_30", "rsi4_25", "rsi4_30"),
        "long_vol_leg": ("vixm", "vixy_half", "none"),
    },
    primary_metric_str=(
        "Paired stationary-bootstrap Sharpe share of the Growth and Growth Plus picks (fund_products_opt_20261102) with 10 "
        "points moved from CORE5, all / 2011+ / 2023+, at 5 bp, +5 bp and 15/35 bp, and with the 2020 crash blanked"
    ),
    kill_criteria_str=(
        "S8 Cold Blood Index or monthly CUSUM RED; measured fill cost above 15 bp a side; the ProShares 1446(f) qualified "
        "notice lapses or VIXM is closed; any of the adoption rules in SPEC_FROZEN.md fails"
    ),
    source_str=(
        "Aziz & Zarattini, SSRN 5316487; Concretum Research 2026-06-14; ledger rows 310-311 and 588; "
        "docs/research/VRP_RSI2_LOCKED_SPEC_AND_GROWTH_CONSTRUCTION_20261103.md; Claude reviews 2026-11-01 to 2026-11-03"
    ),
    source_published_date_str="2025-06-23",
    retro_bool=True,
    prior_trials_int=460,
    parent_id_str="vrp_multipod_scout_retro_20261102",
    grid_justification_str=(
        "429 cells = 13 thresholds x 11 boost forms x 3 long-vol legs: the neighbourhoods already looked at around the "
        "locked rule (Antigravity's RSI table, Claude's threshold grid and long-vol swaps), registered so they count as "
        "trials. Nothing is picked from this grid now; the rule is the owner's lock. Prior 460 = 250 counted by row 588 "
        "+ 144 of row 588's grid + 15 RSI cells + 9 gold cells + about 40 review checks."
    ),
    universe_choice_str=(
        "SVXY and VIXM are the only liquid -0.5x short-term and 1x mid-term VIX futures ETPs with Norgate history from "
        "2011; VIXM was preferred to VIXY after seeing results (VIXY at full size 0.74, -37%)."
    ),
    universe_chosen_after_results_bool=True,
)


def main() -> None:
    ledger = Ledger()
    if REGISTRATION.registration_id_str in registration_rows(ledger):
        print("already registered")
        return
    row_dict = register(ledger, REGISTRATION)
    print("registered row", row_dict["row_id_int"], "hash", REGISTRATION.hash_str())


if __name__ == "__main__":
    main()
