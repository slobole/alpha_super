"""A26: trial-count corrections written as ledger notes (the registration rows are never edited).

1. Book weight grids. The two fund-products studies of 2026-11-02 were registered in tactical_asset_allocation, but
   their 7,611 and 13,746 evaluations are weight grids of whole books and execution-route sizes, not tests of a TAA
   signal. A24 decided not to keep a book family, so they count 0 toward any signal family.
2. Growth-hunt rounds whose frozen grid misstates the search (quant review of A26):
   - hunt_a4_round3 registered a 151,200-cell product of "questions"; its justification states 2,215 simulations read
     through a fixed panel of eight settings: 17,720 evaluations (re-reads count, as in A23).
   - hunt_a4_round2's justification lists 6,639 counted simulations beyond its 3,840-cell grid: 10,479.
   - every other hunt agent that logged more evaluated rows in results/scout/growth_hunt_20261030/<agent>/trials.csv
     than its registered grids add up to: the difference goes to the agent's last registration in each family, split
     in proportion to the family's share of the agent's grids. Agents that logged fewer rows keep their grids
     (conservative).

    PYTHONUTF8=1 .venv/Scripts/python.exe scripts/research/scout_trial_counts_20261107/correct.py [--dry-run]
"""

from __future__ import annotations

import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from alpha.scout.ledger import MAIN_CHECKOUT_ROOT_PATH, Ledger
from alpha.scout.registration import registration_rows
from alpha.scout.trials import record_trial_count_correction, trial_count_correction_dict

HUNT_ROOT_PATH = MAIN_CHECKOUT_ROOT_PATH / "results" / "scout" / "growth_hunt_20261030"
BOOK_REASON_STR = ("Weight grid of whole books and execution-route sizes, not tests of a TAA signal; books are blends of the "
                   "same engines (A24), so these evaluations count toward no signal family (A26).")
FIXED_CORRECTION_LIST = [
    ("fund_products_opt_20261102", 0, BOOK_REASON_STR),
    ("fund_products_size_20261102", 0, BOOK_REASON_STR),
    ("hunt_a4_round3_20261030", 2215 * 8, "The 151,200-cell grid is a product of questions; the justification states 2,215 "
                                          "simulations read through eight fixed settings = 17,720 evaluations (A26 review)."),
    ("hunt_a4_round2_20261030", 3840 + 6639, "3,840-cell grid plus the 6,639 counted simulations its justification lists "
                                             "(T2 630, T5 5,040, T6 945, T7 24) = 10,479 (A26 review)."),
]


def hunt_correction_list(registration_dict: dict, fixed_id_set: set[str]) -> list[tuple[str, int, str]]:
    """Agents whose trials.csv holds more evaluated rows than their registered grids: add the difference."""
    by_agent_dict = defaultdict(list)
    for row_dict in registration_dict.values():
        match = re.match(r"hunt_(a\d+)_", row_dict["registration_id_str"])
        if match and not row_dict.get("retro_bool"):
            by_agent_dict[match.group(1)].append(row_dict)
    correction_list = []
    for agent_str, row_list in sorted(by_agent_dict.items()):
        folder_list = [p for p in HUNT_ROOT_PATH.glob(f"{agent_str}_*") if p.is_dir()]
        if len(folder_list) != 1 or not (folder_list[0] / "trials.csv").exists():
            continue
        with open(folder_list[0] / "trials.csv", encoding="utf-8", newline="") as handle:
            logged_int = sum(1 for _ in csv.DictReader(handle))
        grid_int = sum(int(r.get("grid_size_int") or 1) for r in row_list)
        if logged_int <= grid_int or any(r["registration_id_str"] in fixed_id_set for r in row_list):
            continue
        extra_int = logged_int - grid_int
        family_grid_dict = defaultdict(int)
        for r in row_list:
            family_grid_dict[r["family_id_str"]] += int(r.get("grid_size_int") or 1)
        given_int = 0
        family_list = sorted(family_grid_dict)
        for index_int, family_str in enumerate(family_list):
            share_int = (extra_int - given_int if index_int == len(family_list) - 1
                         else round(extra_int * family_grid_dict[family_str] / grid_int))
            given_int += share_int
            last_row = max((r for r in row_list if r["family_id_str"] == family_str), key=lambda r: r["row_id_int"])
            correction_list.append((last_row["registration_id_str"], int(last_row.get("grid_size_int") or 1) + share_int,
                                    f"{agent_str} logged {logged_int} evaluated rows in trials.csv against {grid_int} registered "
                                    f"grid cells; this row carries its grid plus {share_int} of the {extra_int} extra (A26 review)."))
    return correction_list


def main() -> None:
    ledger = Ledger()
    existing_dict = trial_count_correction_dict(ledger)
    registration_dict = registration_rows(ledger)
    correction_list = FIXED_CORRECTION_LIST + hunt_correction_list(registration_dict, {c[0] for c in FIXED_CORRECTION_LIST})
    for registration_id_str, counted_int, reason_str in correction_list:
        if existing_dict.get(registration_id_str) == counted_int:
            print(f"skip (already corrected)  {registration_id_str}")
        elif "--dry-run" in sys.argv:
            print(f"would correct  {registration_id_str} -> {counted_int:,}  ({reason_str[:80]})")
        else:
            row_dict = record_trial_count_correction(ledger, registration_id_str, counted_int, reason_str)
            print(f"note row {row_dict['row_id_int']}  {registration_id_str} -> {counted_int:,}")
    print(f"ledger chain OK: {ledger.verify()} rows")


if __name__ == "__main__":
    main()
