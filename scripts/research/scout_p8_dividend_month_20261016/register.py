"""Station S0 of the first P8 discovery family (2026-10-16): the dividend month premium, registered before any result.

Nothing about this idea had been run in this workspace, Pakal or Scout before this registration (checked:
no Hartzmark-Solomon or dividend-month study; the 2026-08 TradeQuantix "stable dividend" study is a different rule
and produced no backtest). It is therefore the first family whose vault (2023-01-01 on) is clean.

    PYTHONUTF8=1 .venv/Scripts/python.exe scripts/research/scout_p8_dividend_month_20261016/register.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from alpha.scout.ledger import Ledger
from alpha.scout.registration import Registration, register, registration_rows

REGISTRATION_ID_STR = "dividend_month_premium_20261016"
PARAM_GRID_DICT = {
    # Which past ex-dividend months predict an ex-dividend in month m.
    "lookback_rule_str": ("m3", "m12", "m3_or_m12", "m3_and_m12"),
    # Long-leg weights: equal, or proportional to the predicted yield (last dividend / close at the decision).
    "weighting_str": ("EW", "yield"),
}


def registration() -> Registration:
    return Registration(
        registration_id_str=REGISTRATION_ID_STR,
        family_id_str="dividend_seeking_demand",
        hypothesis_str=("S&P 500 members predicted to go ex-dividend in calendar month m (an ex-dividend date in month m-3 "
                        "or m-12) earn a higher total return over month m than the other eligible members."),
        mechanism_str=("Dividend-seeking investors (income funds, retail investors who treat dividends as free income) buy "
                       "ahead of predictable payments and push prices up around the dividend month; the pressure reverses "
                       "after the payment (Hartzmark and Solomon 2013)."),
        expected_sign_and_location_str=(
            "S3: date-level (one date per month) Newey-West t of the predicted payers' month-m total return minus the "
            "eligible mean is positive, about +0.2% to +0.5% a month gross, with the same sign in each of the three eras "
            "and larger for higher predicted yield. S5: active Sharpe over the equal-weight eligible members positive."),
        hypothesis_class_str="X",
        universe_str=("Norgate S&P 500 'Current & Past' point-in-time members (Scout panel 4ecbebd98bf5e8af, sealed at "
                      "2022-12-30), in sample 1998-2022; replication on the Nasdaq-100 panel. Eligible at the decision: "
                      "member that day, >= 252 sessions of history, unadjusted close >= $5, finite turnover."),
        horizon_str="one month (month m)",
        schedule_str="decision at the close of the last session of month m-1, from dividends with ex-dates <= that close",
        execution_str=("Open of the first session of month m, held to the next rebalance (alpha/scout/engines/weights). "
                       "Pod = dollar-neutral: long the predicted payers (gross 1.0), short the other eligible members "
                       "equal weight (gross 1.0). Costs: engine CostModel (2.5 bp, $0.005/share, $1 minimum), 25% "
                       "dividend withholding on long dividends (the owner's tax case; 0% reported as a sensitivity), "
                       "1%/yr borrow on shorts, fund-scale capital $10M."),
        param_grid_dict=PARAM_GRID_DICT,
        primary_metric_str=(
            "S3 date-level Newey-West one-sided p of the month-m total-return excess (primary config m3_or_m12, EW). "
            "S5 gate: MCPT of the whole 8-config search with plateau choice, per-asset null that permutes returns "
            "and keeps the real dividend calendar fixed; score = Sharpe of the daily active return (long payers minus "
            "EW eligible); p <= 0.05. S6: T-bill slot test as a 20% sleeve of the live 60/40 book "
            "(TAA 3x 48% / NDX VXN 32% / candidate 20% vs T-bills), PASS at P >= 0.80."),
        kill_criteria_str=(
            "S3 hard fail (wrong sign under both estimators, or date-level t <= -2) ends the family (REJECTED). "
            "MCPT p > 0.05 or any other soft fail = WATCHLIST, no vault. Vault (opened only for a CANDIDATE, clean): "
            "net return <= T-bills over 2023-01-01..data end, or vault Sharpe below the 5th percentile of the in-sample "
            "stationary-bootstrap Sharpe for windows of the same length = FAIL, final for the family. Shadow: S8 "
            "CUSUM or Cold Blood Index red."),
        source_str=("Hartzmark, S. M. and Solomon, D. H. (2013), 'The dividend month premium', Journal of Financial "
                    "Economics 109(3), 640-660. Scout P8 discovery, first new family (design A20)."),
        source_published_date_str="2013-09-01",
        retro_bool=False,
        prior_trials_int=0,
        universe_choice_str=("The standard Scout point-in-time stock panel, fixed before any result; the Nasdaq-100 "
                             "panel is the registered replication universe."),
        universe_chosen_after_results_bool=False,
    )


def main() -> None:
    ledger = Ledger()
    if REGISTRATION_ID_STR in registration_rows(ledger):
        print(f"{REGISTRATION_ID_STR} already registered.")
        return
    row_dict = register(ledger, registration())
    print(f"registered row {row_dict['row_id_int']} {REGISTRATION_ID_STR} at {row_dict['utc_ts_str']}; chain {ledger.verify()} rows")


if __name__ == "__main__":
    main()
