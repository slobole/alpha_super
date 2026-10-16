"""Program-wide false discovery control over the ledger (design section 8.3 "Global count", P8).

Every station result that carries a p-value is written once as a `test_result` row (resamples are not trials, design
8.2). Each registration has at most one PRIMARY test (the first one written is final; a second is refused): the one its
registration names as the primary metric, or for a
re-audition the S5 MCPT gate (when the gate has several components, all must pass, so its p is the largest component
p: an intersection-union test). The report applies Benjamini-Yekutieli at q = 0.05 across the primary p-values of
every family, because families share data and are not independent.

What it does not cover, and says so: registrations with no recorded p (most retro rows from studies run outside
Scout) are listed but are not in the BY count m. Their trials still deflate the DSR through `prior_trials_int`.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from alpha.scout.ledger import Ledger
from alpha.scout.registration import registration_rows
from alpha.stats.fdr import benjamini_yekutieli_q_vec

FDR_Q_FLOAT = 0.05
STATION_TUPLE = ("S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "grade")


def _registration_precondition(registration_id_str: str):
    def precondition(existing_ledger: Ledger) -> dict:
        row_dict = registration_rows(existing_ledger).get(registration_id_str)
        if row_dict is None:
            raise ValueError(f"Registration {registration_id_str!r} does not exist.")
        return {"family_id_str": row_dict["family_id_str"]}

    return precondition


def record_test_result(ledger: Ledger, registration_id_str: str, test_id_str: str, station_str: str, p_value_float: float,
                       statistic_dict: dict | None = None, primary_bool: bool = False, source_str: str = "",
                       retro_bool: bool = False) -> dict:
    """One test_result row. `retro_bool` marks a p-value harvested from an earlier saved result."""
    if station_str not in STATION_TUPLE:
        raise ValueError(f"station_str must be one of {STATION_TUPLE}.")
    if not 0.0 <= float(p_value_float) <= 1.0:
        raise ValueError("p_value_float must lie in [0, 1].")
    precondition_fn = _registration_precondition(registration_id_str)

    def precondition(existing_ledger: Ledger) -> dict:
        # *** CRITICAL*** the first primary p of a registration is final: a re-run cannot replace it in the BY report.
        if primary_bool and any(row.get("registration_id_str") == registration_id_str and row.get("primary_bool")
                                for row in existing_ledger.rows("test_result")):
            raise ValueError(f"{registration_id_str} already has a primary test; a re-run records non-primary rows only.")
        return precondition_fn(existing_ledger)

    return ledger.append("test_result", {
        "registration_id_str": registration_id_str, "test_id_str": test_id_str, "station_str": station_str,
        "p_value_float": float(p_value_float), "statistic_dict": statistic_dict or {}, "primary_bool": bool(primary_bool),
        "source_str": source_str, "retro_bool": bool(retro_bool),
    }, precondition_fn=precondition)


def record_station_verdict(ledger: Ledger, registration_id_str: str, station_str: str, check_list: list) -> dict:
    """One station_verdict row: the station's check rows (name, verdict, detail) and its overall verdict."""
    if station_str not in STATION_TUPLE:
        raise ValueError(f"station_str must be one of {STATION_TUPLE}.")
    if not check_list:
        raise ValueError("A station verdict needs at least one check row.")
    verdict_list = [str(verdict_str) for _, verdict_str, _ in check_list]
    if any(v.startswith("FAIL") for v in verdict_list):
        overall_str = "FAIL"
    elif any(v.startswith("WARN") for v in verdict_list):
        overall_str = "WARN"
    elif verdict_list and station_str != "grade" and all(v == "INFO" for v in verdict_list):
        overall_str = "INFO"
    else:
        overall_str = verdict_list[0] if station_str == "grade" else "PASS"
    return ledger.append("station_verdict", {
        "registration_id_str": registration_id_str, "station_str": station_str, "verdict_str": overall_str,
        "check_list": [[str(n), str(v), str(d)] for n, v, d in check_list],
    }, precondition_fn=_registration_precondition(registration_id_str))


def fdr_table(ledger: Ledger, q_float: float = FDR_Q_FLOAT) -> dict:
    """Per registration: family, retro, prior trials, the latest primary test and its BY q; per family: totals."""
    ledger.verify()
    registration_dict = registration_rows(ledger)
    primary_dict: dict[str, dict] = {}
    test_count_dict: dict[str, int] = defaultdict(int)
    for row_dict in ledger.rows("test_result"):
        test_count_dict[row_dict["registration_id_str"]] += 1
        if row_dict.get("primary_bool"):
            primary_dict.setdefault(row_dict["registration_id_str"], row_dict)  # the first primary row is final
    tested_id_list = [rid for rid in registration_dict if rid in primary_dict]
    q_by_id_dict = {}
    if tested_id_list:
        q_vec = benjamini_yekutieli_q_vec([primary_dict[rid]["p_value_float"] for rid in tested_id_list])
        q_by_id_dict = dict(zip(tested_id_list, q_vec.tolist()))

    row_list = []
    for rid, reg_dict in registration_dict.items():
        test_dict = primary_dict.get(rid)
        row_list.append({
            "registration_id_str": rid, "family_id_str": reg_dict["family_id_str"], "retro_bool": bool(reg_dict.get("retro_bool")),
            "prior_trials_int": int(reg_dict.get("prior_trials_int") or 0), "grid_size_int": int(reg_dict.get("grid_size_int") or 1),
            "test_count_int": test_count_dict.get(rid, 0),
            "primary_test_str": test_dict["test_id_str"] if test_dict else None,
            "p_value_float": test_dict["p_value_float"] if test_dict else None,
            "by_q_float": q_by_id_dict.get(rid),
            "discovery_bool": bool(q_by_id_dict.get(rid) is not None and q_by_id_dict[rid] <= q_float),
        })
    family_dict: dict[str, dict] = {}
    for row in row_list:
        fam = family_dict.setdefault(row["family_id_str"], {"registrations_int": 0, "prior_trials_int": 0, "tested_int": 0, "discoveries_int": 0})
        fam["registrations_int"] += 1
        fam["prior_trials_int"] += row["prior_trials_int"]
        fam["tested_int"] += int(row["p_value_float"] is not None)
        fam["discoveries_int"] += int(row["discovery_bool"])
    return {"q_float": q_float, "test_count_int": len(tested_id_list), "registration_count_int": len(row_list),
            "rows": row_list, "families": dict(sorted(family_dict.items()))}


def render_markdown(table_dict: dict) -> str:
    q_float = table_dict["q_float"]
    tested_list = sorted((r for r in table_dict["rows"] if r["p_value_float"] is not None), key=lambda r: r["p_value_float"])
    untested_list = [r for r in table_dict["rows"] if r["p_value_float"] is None]
    discovery_int = sum(r["discovery_bool"] for r in tested_list)
    line_list = [
        "# Scout ledger: program-wide false discovery report",
        "",
        f"Benjamini-Yekutieli at q = {q_float} over the primary p-value of every registration that has one "
        f"({table_dict['test_count_int']} of {table_dict['registration_count_int']} registrations). "
        f"**{discovery_int} discoveries.** A discovery here means the primary test survives the whole program's multiple "
        "testing; it is not a grade (grades also need S4-S6).",
        "",
        "## Primary tests",
        "",
        "| Registration | Family | Primary test | p | BY q | Discovery |",
        "| --- | --- | --- | ---: | ---: | :---: |",
    ]
    for r in tested_list:
        line_list.append(f"| {r['registration_id_str']} | {r['family_id_str']} | {r['primary_test_str']} | {r['p_value_float']:.4f} | "
                         f"{r['by_q_float']:.4f} | {'yes' if r['discovery_bool'] else 'no'} |")
    line_list += ["", "## Families", "", "| Family | Registrations | Prior trials | With a primary p | Discoveries |",
                  "| --- | ---: | ---: | ---: | ---: |"]
    for fam_str, fam in table_dict["families"].items():
        line_list.append(f"| {fam_str} | {fam['registrations_int']} | {fam['prior_trials_int']:,} | {fam['tested_int']} | {fam['discoveries_int']} |")
    line_list += ["", f"## Registrations without a recorded p ({len(untested_list)})", "",
                  "Not in the BY count. Most are retro rows from studies run outside Scout; their trials still deflate the "
                  "DSR through prior_trials_int.", ""]
    line_list += [f"- {r['registration_id_str']} ({r['family_id_str']}, prior {r['prior_trials_int']:,})" for r in untested_list]
    return "\n".join(line_list) + "\n"


def write_report(ledger: Ledger, output_dir_path) -> dict:
    from pathlib import Path

    output_dir_path = Path(output_dir_path)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    table_dict = fdr_table(ledger)
    (output_dir_path / "fdr_report.md").write_text(render_markdown(table_dict), encoding="utf-8")
    import json

    (output_dir_path / "fdr_report.json").write_text(json.dumps(table_dict, indent=2, default=lambda o: None if o is np.nan else str(o)), encoding="utf-8")
    return table_dict
