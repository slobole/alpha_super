"""Station S0, retro: the 77 knowledge-base studies of 2026-10-17..11-07 that no ledger registration cites (design A23).

`python -m alpha.scout kb` listed 77 studies (48,873 evaluations by their own counts) with no citing registration on
2026-11-07: A22's check was report only, so nothing stopped the gap from growing. This script registers them
retroactively from `rows.json` beside it and gives each new registration one grade row, as A22 did. Nothing is re-run.

rows.json holds one object per (study, mechanism family). Seven drafting agents read each study's report, ledger and
frozen spec; the review decisions of A23 are applied by the merge step and recorded in each row's `review_note`.

Counting rules (A19 / A22, prior trials count as independent):
- prior_trials_int = the study's own `actual_total_variants`, minus null draws (random-entry, random-rank, price-noise
  and shuffled-history controls) where the study counted them as variants (8.2). Placebo runs of the same rule on other
  assets are real evaluations and are kept (hy_bond_ma_cross 210).
- One registration per (study, family); each study's rows sum to its counted total (checked in
  tests/test_scout_kb_link.py). Re-runs of another study's evaluations are counted again, as in A19 / A22; an umbrella
  record that only re-lists its children's evaluations counts its own layer (momentum_mr_trend_deep_20261020: 94 of
  368, the rest are its three children's rows).
- An audit of a paper's printed tables with no backtest counts 0 (ssrn_filter_k_replication_20261020, as the GQR audit
  in A22).
- parent_id_str only inside the same family; rows are ordered so a parent is registered first.

Grades: one `grade` row per registration, word + " (outside Scout)", reason = the study's verdict. None is CANDIDATE.

    PYTHONUTF8=1 .venv/Scripts/python.exe scripts/research/scout_kb_link_20261107/register.py [--dry-run]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from alpha.scout.fdr_report import record_station_verdict
from alpha.scout.ledger import Ledger
from alpha.scout.registration import Registration, register, registration_rows

ROWS_PATH = Path(__file__).with_name("rows.json")
PAKAL_REPORTS_STR = "pakal-research/reports"
OUTSIDE_STR = "outside Scout"
KILL_RETRO_STR = "Retro registration of a closed study: its verdict stands; a forward line, if any, is killed by its S8 CUSUM / Cold Blood Index."
GRID_RETRO_STR = "Retro record of the study's declared grid; its trials are counted through prior_trials_int, not by this grid."
METRIC_RETRO_STR = "The study's own primary metric and gates, as frozen in its specification file (retro record)."
UNIVERSE_RETRO_STR = "Fixed by the study's frozen specification before its results (retro record; see the study's SOURCE_RULE_MAP / spec)."


def row_list() -> list[dict]:
    return json.loads(ROWS_PATH.read_text(encoding="utf-8"))


def _grid(grid_dict: dict | None) -> dict:
    out_dict = {}
    for name_str, value in (grid_dict or {}).items():
        value_tuple = tuple(value) if isinstance(value, (list, tuple)) else (value,)
        out_dict[str(name_str)] = tuple(json.dumps(v) if isinstance(v, (dict, list)) else v for v in value_tuple) or ("n/a",)
    return out_dict


def source_str(row: dict) -> str:
    part_str = f": {row['part_label']}" if row.get("part_label") and row["counted_total_int"] != row["trials_int"] else ""
    estimate_str = ", split estimated" if row.get("estimate") else ""
    if row.get("count_note"):
        estimate_str += f"; count: {row['count_note']}"
    note_str = f" Review: {row['review_note']}." if row.get("review_note") else ""
    overlap_str = f" Overlap: {row['overlap_note']}" if row.get("overlap_note") else ""
    return (f"{PAKAL_REPORTS_STR}/{row['folder']} ({row['trials_int']} of the study's {row['counted_total_int']} counted "
            f"evaluations{part_str}{estimate_str}; knowledge base total {row['kb_total_int']}). "
            f"Verdict: {row['grade_reason']}{note_str}{overlap_str}")


def registration_list() -> list[Registration]:
    return [
        Registration(
            registration_id_str=row["registration_id"],
            family_id_str=row["family_id"],
            hypothesis_str=row["hypothesis_str"],
            mechanism_str=row["mechanism_str"],
            expected_sign_and_location_str=row["expected_sign_and_location_str"],
            hypothesis_class_str=row["hypothesis_class_str"],
            universe_str=row["universe_str"],
            horizon_str=row["horizon_str"],
            schedule_str=row["schedule_str"],
            execution_str=row["execution_str"],
            param_grid_dict=_grid(row.get("param_grid_dict")),
            primary_metric_str=METRIC_RETRO_STR,
            kill_criteria_str=KILL_RETRO_STR,
            source_str=source_str(row),
            source_published_date_str=row.get("source_published_date_str") or None,
            retro_bool=True,
            prior_trials_int=int(row["trials_int"]),
            parent_id_str=row.get("parent_id_str") or None,
            grid_justification_str=GRID_RETRO_STR,
            universe_choice_str=UNIVERSE_RETRO_STR,
            universe_chosen_after_results_bool=False,
        )
        for row in row_list()
    ]


def grade_dict() -> dict[str, tuple[str, str, str]]:
    return {row["registration_id"]: (row["grade_word"], row["grade_reason"], f"{PAKAL_REPORTS_STR}/{row['folder']}")
            for row in row_list()}


def main() -> None:
    dry_run_bool = "--dry-run" in sys.argv
    ledger = Ledger()
    existing_dict = registration_rows(ledger)
    graded_set = {row["registration_id_str"] for row in ledger.rows("station_verdict") if row.get("station_str") == "grade"}
    grades = grade_dict()
    for reg in registration_list():
        reg.validate()
        word_str, reason_str, grade_source_str = grades[reg.registration_id_str]
        grade = f"{word_str} ({OUTSIDE_STR})"
        if reg.registration_id_str in existing_dict:
            print(f"skip (exists)  {reg.registration_id_str}")
        elif dry_run_bool:
            print(f"would register {reg.registration_id_str}  family={reg.family_id_str}  prior={reg.prior_trials_int}  "
                  f"parent={reg.parent_id_str}  grade={grade}")
            continue
        else:
            row_dict = register(ledger, reg)
            existing_dict[reg.registration_id_str] = row_dict
            print(f"registered row {row_dict['row_id_int']}  {reg.registration_id_str}  prior={reg.prior_trials_int}")
        if reg.registration_id_str in graded_set:
            print(f"skip (graded)  {reg.registration_id_str}")
            continue
        row_dict = record_station_verdict(ledger, reg.registration_id_str, "grade", [("grade", grade, reason_str)],
                                          retro_bool=True, source_str=grade_source_str)
        print(f"graded row {row_dict['row_id_int']}  {reg.registration_id_str}  {grade}")
    print(f"ledger chain OK: {ledger.verify()} rows")


if __name__ == "__main__":
    main()
