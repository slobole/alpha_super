"""Knowledge base -> ledger link (design A22): which research studies have no Scout registration.

The knowledge base (docs/research/knowledge-base/knowledge_base_registry.json) records every finished study, most of
them run in pakal-research outside Scout. Their trials only reach a family's N, and so the DSR and the BY report, when
a registration counts them. A registration **cites** a study when its `source_str` contains `reports/<folder>`, where
`<folder>` is the path between `reports/` and `/REPORT` in the study's `concise_report` (Windows separators are
normalised first) and the next character does not continue the name, so `reports/x_v2` does not cite `x`. A citation
of a sub-folder (`reports/x/lane`) also cites `x`, as a plain substring match would.

Four groups:
- **cited**: some registration cites the study.
- **gap**: no registration cites it and it is dated on or after SCOUT_LEDGER_START_STR (or has no date). Each one
  needs a retro registration (A19, A21, A22).
- **pre-Scout**: no citing registration, dated before the ledger began. Listed, not counted as gaps.
- **not registrable**: listed in NOT_REGISTRABLE_DICT with the reason (e.g. a book that only combines registered sleeves).

Report only: nothing here writes to the ledger or the knowledge base.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from alpha.scout.ledger import MAIN_CHECKOUT_ROOT_PATH, Ledger

KB_REGISTRY_PATH = MAIN_CHECKOUT_ROOT_PATH / "docs" / "research" / "knowledge-base" / "knowledge_base_registry.json"
SCOUT_LEDGER_START_STR = "2026-10-01"
# Studies that need no registration of their own, with the reason (they only combine registered sleeves).
NOT_REGISTRABLE_DICT: dict[str, str] = {
    "book_picture_20261014": "combines sleeves that are already registered (A19)",
}
_FOLDER_PATTERN = re.compile(r"reports/(.+?)/REPORT")


def study_folder_str(concise_report_str: str | None) -> str | None:
    """`reports/<folder>/REPORT...` -> `<folder>`, or None when the path has no reports folder."""
    match = _FOLDER_PATTERN.search((concise_report_str or "").replace("\\", "/"))
    return match.group(1) if match else None


def cites_bool(source_str: str, folder_str: str) -> bool:
    return re.search(rf"reports/{re.escape(folder_str)}(?![A-Za-z0-9_-])", source_str.replace("\\", "/")) is not None


def kb_link_report(ledger: Ledger, registry_path: Path = KB_REGISTRY_PATH) -> dict:
    """Classify every knowledge-base study as cited, gap, pre-Scout or not registrable."""
    study_list = json.loads(Path(registry_path).read_text(encoding="utf-8"))["studies"]
    source_list = [row_dict.get("source_str", "") for row_dict in ledger.rows("registration")]
    report_dict = {"studies_int": len(study_list), "cited": [], "gaps": [], "pre_scout": [], "not_registrable": [],
                   "ledger_start_str": SCOUT_LEDGER_START_STR, "registry_path_str": str(registry_path)}
    for study_dict in study_list:
        folder_str = study_folder_str(study_dict.get("concise_report"))
        entry_dict = {"study_id_str": study_dict["study_id"], "folder_str": folder_str,
                      "created_str": (study_dict.get("created_at") or "")[:10]}
        if folder_str and any(cites_bool(source_str, folder_str) for source_str in source_list):
            report_dict["cited"].append(entry_dict)
        elif entry_dict["study_id_str"] in NOT_REGISTRABLE_DICT:
            report_dict["not_registrable"].append({**entry_dict, "reason_str": NOT_REGISTRABLE_DICT[entry_dict["study_id_str"]]})
        elif entry_dict["created_str"] and entry_dict["created_str"] < SCOUT_LEDGER_START_STR:
            report_dict["pre_scout"].append(entry_dict)
        else:
            report_dict["gaps"].append(entry_dict)
    report_dict["gap_count_int"] = len(report_dict["gaps"])
    return report_dict


def summary_str(report_dict: dict) -> str:
    gap_str = ", ".join(e["study_id_str"] for e in report_dict["gaps"]) or "none"
    return (f"{report_dict['gap_count_int']} unregistered studies since {report_dict['ledger_start_str']} ({gap_str}); "
            f"{len(report_dict['cited'])} cited, {len(report_dict['pre_scout'])} pre-Scout, "
            f"{len(report_dict['not_registrable'])} not registrable, of {report_dict['studies_int']}")
