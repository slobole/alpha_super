"""Station S0 of the DV2 size ladder (2026-10-02): one registration, written before any result of this study.

Question (owner): where does DV2's short-term reversal alpha come from: mega, large, mid, small or micro caps? The rule
is FROZEN at the WIRED live configuration (alpha/scout/specs/dv2.py LIVE_CONFIG); the only axis is the universe.

    PYTHONUTF8=1 uv run python scripts/research/scout_dv2_size_ladder_20261002/register.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from alpha.scout.ledger import Ledger
from alpha.scout.registration import Registration, register, registration_rows

REGISTRATION_ID_STR = "dv2_size_ladder_20261002"
# (Norgate index name, size bucket); informational universes are evaluated but carry no verdict of their own.
UNIVERSE_TUPLE = (
    ("S&P 100", "mega"), ("Russell Top 200", "mega"), ("Dow Jones Industrial Average", "mega (informational, 30 names)"),
    ("S&P 500", "large (reference)"), ("Russell 1000", "large"),
    ("S&P MidCap 400", "mid"), ("Russell Mid Cap", "mid"),
    ("S&P SmallCap 600", "small"), ("Russell 2000", "small"),
    ("Russell Micro Cap", "micro (informational)"),
    ("Russell 3000", "broad"), ("S&P Composite 1500", "broad"),
)
PRIOR_TRIALS_INT = 200 + len(UNIVERSE_TUPLE)  # the 2026-09-25 deep research (200) plus this study's universes


def registration() -> Registration:
    return Registration(
        registration_id_str=REGISTRATION_ID_STR,
        family_id_str="us_equity_short_term_reversal",
        hypothesis_str=("DV2's short-term reversal edge is larger gross in smaller, less liquid stocks (liquidity provision), "
                        "but net of realistic costs it concentrates in large caps."),
        mechanism_str=("Liquidity provision to impatient sellers: the reversal premium should scale with illiquidity (wider "
                       "spreads, thinner books), so gross edge rises down the size ladder while trading costs rise faster."),
        expected_sign_and_location_str=(
            "S3 (h3, excess over same-date eligible members) positive in every size bucket and larger gross in smaller / lower "
            "ADV63 buckets; frozen-rule pod net Sharpe (2x costs + 10 bp and a liquidity-aware half-spread cost) highest in "
            "large caps (S&P 500 / Russell 1000) and lower in small and micro caps."),
        hypothesis_class_str="E",
        universe_str=("Norgate point-in-time 'Current & Past' members, exact flag (no tail trim): "
                      + "; ".join(f"{name_str} [{bucket_str}]" for name_str, bucket_str in UNIVERSE_TUPLE)
                      + ". Size split inside Russell 3000 and S&P Composite 1500 by causal ADV63 tercile and by index membership."),
        horizon_str="days (S3 at h3, the P7 horizon of the S&P 500 pod)",
        schedule_str="daily close decision",
        execution_str="next session's open (fast panel replica of the WIRED rule; costs added per fill)",
        param_grid_dict={"universe_str": tuple(name_str for name_str, _ in UNIVERSE_TUPLE)},
        primary_metric_str=("S3 date-level Newey-West t and mean excess (bp) at h3 per size bucket inside Russell 3000 and S&P "
                            "Composite 1500 (ADV63 terciles, index membership); secondary: per-universe S3 and frozen-rule pod "
                            "net Sharpe at engine costs, 2x costs + 10 bp, and max(2.5 bp, causal Abdi-Ranaldo half-spread)."),
        kill_criteria_str=("Hypothesis rejected if the gross S3 edge is not larger in the small / low-ADV buckets than in the large / "
                           "high-ADV buckets, or if net Sharpe at 2x costs + 10 bp is higher in small or micro caps than in large "
                           "caps. A universe is not a candidate pod unless S3 t >= 2, net Sharpe > 0 at 2x costs + 10 bp, and its "
                           "per-asset panel MCPT p <= 0.05."),
        source_str=("Own follow-up of Scout P7 (dv2_reaudition_20261002); DV2 indicator by D. Varadi (CSS Analytics); rule = "
                    "alpha/scout/specs/dv2.py LIVE_CONFIG (DV2 < 10, Close > SMA200, 126-session return > 0.05, NATR14 rank, "
                    "10 slots, exit Close > High(t-1), next-open fills). No parameter search."),
        retro_bool=False,
        prior_trials_int=PRIOR_TRIALS_INT,
        parent_id_str="dv2_reaudition_20261002",
        universe_choice_str=(
            "Every Norgate US equity index with point-in-time membership, mega to micro, chosen to answer the owner's size "
            "question. Prior results on some of them were seen before this registration (2026-09-25 deep research: MidCap 400 "
            "Sharpe 0.60, SmallCap 600 0.66, a July Russell transfer failure; P7: S&P 500 and Nasdaq-100), so the choice is "
            "marked as made after results. Prior trials: 200 (deep research estimate) + 12 universes = 212. Vault sealed: "
            "data to 2022-12-30 only."),
        universe_chosen_after_results_bool=True,
    )


def main() -> None:
    ledger = Ledger()
    if REGISTRATION_ID_STR in registration_rows(ledger):
        print("already registered:", REGISTRATION_ID_STR)
        return
    row_dict = register(ledger, registration())
    print("registered", REGISTRATION_ID_STR, "row", row_dict.get("row_id_int"), "ledger", ledger.ledger_path)


if __name__ == "__main__":
    main()
