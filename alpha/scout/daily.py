"""Scout's daily research run (design A21, gap 2): one command, one status file per day.

    uv run python -m alpha.scout daily [--flex-db PATH | --flex-xml FILE ...]

Five parts, each isolated (one failure is reported and never hides the others):

1. **ledger**: walk the hash chain (`Ledger.verify`).
2. **shadow**: the S8 monitor for every shadow_start row (`alpha.scout.shadow.run_shadow_monitor`).
3. **health**: the live pod health report (`alpha.scout.pod_health.run_pod_health_report`), report only. Data: the
   Flex store if one is given or present at DEFAULT_FLEX_DB_PATH, else every Flex XML matching DEFAULT_FLEX_XML_GLOB.
   The live Flex store sits on the VPS; on the workstation the report reads downloaded XML files and says STALE when
   they are old. Nothing is wired to alerts or trading.
4. **fdr**: the program-wide BY report over the ledger.
5. **kb**: knowledge-base studies with no citing registration (`alpha.scout.kb_link`, A22). Report only: a gap or an
   unreadable knowledge base is shown in the status file but never fails the run.

Output: results/scout/daily/<date>/daily_status.json and daily_status.md (what the Bench page reads), beside each
part's own report folder. Research only: no live path, account or order is read for writing or touched.
"""

from __future__ import annotations

import glob
import json
import os
import traceback
from datetime import date, datetime, timezone
from pathlib import Path

from alpha.scout.ledger import MAIN_CHECKOUT_ROOT_PATH, Ledger

DAILY_ROOT_PATH = MAIN_CHECKOUT_ROOT_PATH / "results" / "scout" / "daily"
DEFAULT_FLEX_DB_PATH = Path(os.environ.get("SCOUT_FLEX_DB", "C:/alpha/live_ops/ibkr_performance.sqlite3"))
DEFAULT_FLEX_XML_GLOB = os.environ.get("SCOUT_FLEX_XML_GLOB", str(Path.home() / "Downloads" / "ALPHA_DAILY_TWR*.xml"))


def _part(name_str: str, fn) -> dict:
    started = datetime.now(timezone.utc)
    try:
        result_dict = fn()
        result_dict.setdefault("ok_bool", True)
    except Exception as error:  # isolate parts: the status file records the failure
        result_dict = {"ok_bool": False, "error_str": f"{type(error).__name__}: {error}", "traceback_str": traceback.format_exc(limit=5)}
    result_dict["seconds_float"] = round((datetime.now(timezone.utc) - started).total_seconds(), 1)
    result_dict["part_str"] = name_str
    return result_dict


def _flex_source(flex_db_path: Path | None, flex_xml_path_list: list[Path] | None) -> tuple[Path | None, list[Path] | None, str]:
    if flex_db_path is not None or flex_xml_path_list:
        return flex_db_path, flex_xml_path_list, "given"
    if DEFAULT_FLEX_DB_PATH.exists():
        return DEFAULT_FLEX_DB_PATH, None, "default Flex store"
    xml_list = sorted(Path(p) for p in glob.glob(DEFAULT_FLEX_XML_GLOB))
    return None, xml_list, f"Flex XML files matching {DEFAULT_FLEX_XML_GLOB}"


def run_daily(as_of_date: date | None = None, flex_db_path: Path | None = None, flex_xml_path_list: list[Path] | None = None,
              ledger: Ledger | None = None, output_root_path: Path = DAILY_ROOT_PATH) -> tuple[Path, dict]:
    as_of_date = as_of_date or date.today()
    ledger = ledger or Ledger()
    part_list = []

    def ledger_part() -> dict:
        return {"rows_int": ledger.verify(), "status_str": "OK"}

    def shadow_part() -> dict:
        from alpha.scout.shadow import run_shadow_monitor

        output_dir_path, report_list = run_shadow_monitor(ledger, as_of_date)
        status_dict = {r["pod_id_str"]: r["status_str"] for r in report_list}
        return {"report_path_str": str(output_dir_path / "shadow.md"), "status_by_registration_dict": status_dict,
                "ok_bool": "ERROR" not in status_dict.values(),
                "errors": {r["pod_id_str"]: r["error_str"] for r in report_list if r.get("error_str")}}

    def health_part() -> dict:
        from alpha.scout.pod_health import run_pod_health_report

        db_path, xml_list, source_str = _flex_source(flex_db_path, flex_xml_path_list)
        if db_path is None and not xml_list:
            return {"status_str": "NO_DATA", "source_str": source_str, "ok_bool": True}
        output_dir_path, report_list = run_pod_health_report(flex_xml_path_list=xml_list, flex_db_path=db_path, as_of_date=as_of_date)
        return {"report_path_str": str(output_dir_path / "pod_health.md"), "source_str": source_str,
                "status_by_pod_dict": {r["label_str"]: r["status_str"] for r in report_list},
                "data_through_by_pod_dict": {r["label_str"]: str((r.get("live") or {}).get("end_str", "")) for r in report_list}}

    def fdr_part() -> dict:
        from alpha.scout.fdr_report import write_report

        output_dir_path = MAIN_CHECKOUT_ROOT_PATH / "results" / "scout" / "fdr" / as_of_date.isoformat()
        table_dict = write_report(ledger, output_dir_path)
        return {"report_path_str": str(output_dir_path / "fdr_report.md"), "registrations_int": table_dict["registration_count_int"],
                "primary_tests_int": table_dict["test_count_int"],
                "discoveries": [r["registration_id_str"] for r in table_dict["rows"] if r["discovery_bool"]]}

    def kb_part() -> dict:
        from alpha.scout.kb_link import kb_link_report, summary_str

        try:
            report_dict = kb_link_report(ledger)
        except Exception as error:  # report only: an unreadable knowledge base never fails the run
            return {"status_str": "UNREADABLE", "summary_str": f"knowledge base not read: {type(error).__name__}: {error}"}
        return {"status_str": "GAPS" if report_dict["gap_count_int"] else "OK", "gap_count_int": report_dict["gap_count_int"],
                "gap_id_list": [e["study_id_str"] for e in report_dict["gaps"]],
                "pre_scout_count_int": len(report_dict["pre_scout"]), "cited_count_int": len(report_dict["cited"]),
                "summary_str": summary_str(report_dict)}

    for name_str, fn in (("ledger", ledger_part), ("shadow", shadow_part), ("health", health_part), ("fdr", fdr_part),
                         ("kb", kb_part)):
        part_list.append(_part(name_str, fn))

    status_dict = {"as_of_str": as_of_date.isoformat(), "written_utc_str": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                   "ok_bool": all(p["ok_bool"] for p in part_list), "parts": {p["part_str"]: p for p in part_list}}
    output_dir_path = Path(output_root_path) / as_of_date.isoformat()
    output_dir_path.mkdir(parents=True, exist_ok=True)
    (output_dir_path / "daily_status.json").write_text(json.dumps(status_dict, indent=2, default=str), encoding="utf-8")
    (output_dir_path / "daily_status.md").write_text(render_markdown(status_dict), encoding="utf-8")
    return output_dir_path, status_dict


def render_markdown(status_dict: dict) -> str:
    line_list = [f"# Scout daily run, {status_dict['as_of_str']}", "",
                 f"Overall: **{'OK' if status_dict['ok_bool'] else 'FAILED PART(S)'}** (written {status_dict['written_utc_str']} UTC)", "",
                 "| Part | OK | Seconds | Summary |", "| --- | :---: | ---: | --- |"]
    for name_str, part in status_dict["parts"].items():
        if not part["ok_bool"] and part.get("error_str"):
            summary_str = part["error_str"]
        elif name_str == "ledger":
            summary_str = f"{part['rows_int']} rows, chain intact"
        elif name_str == "shadow":
            summary_str = ", ".join(f"{k}: {v}" for k, v in part["status_by_registration_dict"].items())
        elif name_str == "health":
            summary_str = part.get("status_str") or ", ".join(f"{k}: {v}" for k, v in part["status_by_pod_dict"].items())
            summary_str += f" ({part['source_str']})"
        elif name_str == "kb":
            summary_str = part["summary_str"]
        else:
            summary_str = f"{len(part['discoveries'])} BY discoveries of {part['primary_tests_int']} primary tests ({', '.join(part['discoveries'])})"
        line_list.append(f"| {name_str} | {'yes' if part['ok_bool'] else '**no**'} | {part['seconds_float']} | {summary_str} |")
    return "\n".join(line_list) + "\n"
