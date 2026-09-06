"""Pure projections of saved operator evidence; no command execution or I/O."""

from __future__ import annotations

from datetime import date
import json
import re
from typing import Any
from zoneinfo import ZoneInfo

from alpha.live.ops_report import parse_timestamp_ts


DIAGNOSTIC_VIEW_LABEL_DICT = {
    "status": "Status & evidence",
    "lifecycle": "Execution trace",
    "events": "Logs & events",
    "provenance": "Data & deployment",
    "operations": "Operational actions",
}
DIAGNOSTIC_EVENT_LIMIT_INT = 500
COMMAND_CATALOG_TUPLE = (
    ("status", "CLI status", "Writes job/release metadata; use Saved status for a read-only view."),
    ("compare_reference", "Generate comparison", "Creates artifacts and may update data caches; no order submission."),
    ("tick", "Run one lifecycle cycle", "May sync data, build new plans and send orders when due."),
    ("submit_vplan", "Submit ready plan", "May send orders; existing timing, auto-submit and duplicate guards remain."),
    ("post_execution_reconcile", "Reconcile execution", "Queries broker and writes reconciliation state."),
    ("eod_snapshot", "Capture EOD", "Queries broker and writes account/state observations."),
)


def build_command_catalog_list(target_obj, releases_root_path_str):
    """Display-only fixed argv; never execute a shell or accept command text."""
    release_obj = target_obj.release_obj
    command_list = []
    for action_str, label_str, effects_str in COMMAND_CATALOG_TUPLE:
        argument_list = ["uv", "run", "python", "-m", "alpha.live.runner", action_str,
            "--mode", release_obj.mode_str, "--pod-id", release_obj.pod_id_str,
            "--releases-root", str(releases_root_path_str), "--db-path", target_obj.db_path_str, "--json"]
        command_list.append({"action_str": action_str, "label_str": label_str, "effects_str": effects_str,
            "command_str": "& " + " ".join("'" + str(argument_str).replace("'", "''") + "'" for argument_str in argument_list)})
    return command_list
SECRET_KEY_PART_STR_TUPLE = ("password", "secret", "token", "authorization", "webhook", "api_key", "api-key", "apikey")
STATUS_FIELD_STR_TUPLE = (
    "user_id_str", "pod_id_str", "mode_str", "account_route_str", "release_id_str",
    "db_status_str", "next_action_str", "reason_code_str", "latest_vplan_id_int",
    "latest_vplan_status_str", "latest_reconciliation_status_str",
    "latest_reconciliation_timestamp_str", "latest_event_timestamp_str",
    "latest_pod_state_timestamp_str", "latest_broker_snapshot_timestamp_str",
)
PROVENANCE_FIELD_STR_TUPLE = (
    "db_path_str", "strategy_import_str", "data_profile_str", "release_id_str",
    "session_calendar_id_str", "signal_clock_str", "execution_policy_str",
    "latest_decision_norgate_profile_str", "latest_decision_norgate_snapshot_date_str",
    "latest_live_reference_snapshot_timestamp_str", "latest_live_reference_source_str",
)


def strategy_display_name_str(row_dict: dict[str, Any]) -> str:
    explicit_name_str = str(row_dict.get("display_name_str") or row_dict.get("strategy_name_str") or "").strip()
    if explicit_name_str:
        return explicit_name_str
    import_name_str = str(row_dict.get("strategy_import_str") or "").split(":")[-1]
    if import_name_str and "." not in import_name_str:
        return re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", import_name_str).removesuffix(" Strategy")
    return str(row_dict.get("pod_id_str") or "Unknown strategy").replace("_", " ")


def redact_diagnostic_value(value_obj: Any) -> Any:
    """Do not expose credentials accidentally embedded in logs or exception URLs.

    Only allowlisted saved evidence is fed here, never config.env or process
    environment. This is a second boundary, not permission to export secrets.
    """
    if isinstance(value_obj, dict):
        return {
            str(key_obj): (
                "[redacted]" if any(part_str in str(key_obj).lower() for part_str in SECRET_KEY_PART_STR_TUPLE)
                else redact_diagnostic_value(item_obj)
            )
            for key_obj, item_obj in value_obj.items()
        }
    if isinstance(value_obj, (tuple, list)):
        return [redact_diagnostic_value(item_obj) for item_obj in value_obj]
    if isinstance(value_obj, str):
        value_str = re.sub(r"(?i)\b(Bearer|Basic)\s+[^\s&,;\"']+", r"\1 [redacted]", value_obj)
        value_str = re.sub(
            r"(?i)((?:token|password|secret|authorization|webhook|api[_-]?key)[\w-]*[\"']?\s*[=:]\s*)(?:\"[^\"]*\"|'[^']*'|(?:(?:Basic|Bearer)\s+)?[^\s&,;]+)",
            r"\1[redacted]", value_str,
        )
        return value_str
    return value_obj


def build_diagnostic_payload_dict(
    summary_dict: dict[str, Any],
    row_dict: dict[str, Any],
    detail_dict: dict[str, Any],
    event_dict_list: list[dict[str, Any]],
    *,
    level_str: str = "all",
    from_date_str: str = "",
    to_date_str: str = "",
    search_str: str = "",
) -> dict[str, Any]:
    if level_str not in {"all", "info", "warn", "error"}:
        raise ValueError("Choose all, info, warn or error.")
    for date_str in (from_date_str, to_date_str):
        if date_str:
            date.fromisoformat(date_str)
    if from_date_str and to_date_str and from_date_str > to_date_str:
        raise ValueError("The first date must not be later than the last date.")
    if len(search_str) > 160:
        raise ValueError("Search must be at most 160 characters.")
    matching_event_dict_list = []
    for event_dict in event_dict_list:
        event_level_str = str(event_dict.get("level_str") or event_dict.get("severity_str") or "info").lower()
        event_level_str = {"warning": "warn", "yellow": "warn", "critical": "error", "red": "error"}.get(event_level_str, event_level_str)
        timestamp_str = str(event_dict.get("timestamp_str") or event_dict.get("event_timestamp_str") or event_dict.get("created_timestamp_str") or "")
        event_timestamp_ts = parse_timestamp_ts(timestamp_str)
        event_date_str = (
            event_timestamp_ts.astimezone(ZoneInfo("America/New_York")).date().isoformat()
            if event_timestamp_ts is not None else ""
        )
        if level_str != "all" and event_level_str != level_str:
            continue
        if (from_date_str or to_date_str) and not event_date_str:
            continue
        if (from_date_str and event_date_str < from_date_str) or (to_date_str and event_date_str > to_date_str):
            continue
        redacted_event_dict = redact_diagnostic_value(event_dict)
        if search_str.lower() not in json.dumps(redacted_event_dict, ensure_ascii=False).lower():
            continue
        matching_event_dict_list.append(redacted_event_dict)
    payload_dict = {
        "kind_str": "saved_operator_evidence",
        "view_built_at_str": summary_dict.get("as_of_timestamp_str"),
        "scope_dict": {field_str: row_dict.get(field_str) for field_str in STATUS_FIELD_STR_TUPLE},
        "provenance_dict": {field_str: row_dict.get(field_str) for field_str in PROVENANCE_FIELD_STR_TUPLE},
        "required_action_dict": row_dict.get("required_action_dict") or {},
        "eod_snapshot_dict": row_dict.get("eod_snapshot_dict") or {},
        "freshness_dict": row_dict.get("data_freshness_dict") or {},
        "lifecycle_step_dict_list": detail_dict.get("lifecycle_step_dict_list") or row_dict.get("lifecycle_step_dict_list") or [],
        "latest_decision_plan_dict": detail_dict.get("latest_decision_plan_dict") or {},
        "latest_vplan_dict": detail_dict.get("latest_vplan_dict") or {},
        "event_dict_list": matching_event_dict_list,
        "scanned_event_count_int": len(event_dict_list),
        "event_limit_int": DIAGNOSTIC_EVENT_LIMIT_INT,
        "limitations_str": (
            "Saved evidence only: no active broker or process probe. Build time is not "
            "evidence freshness. Event filtering covers only the latest 500 loaded events."
        ),
    }
    return redact_diagnostic_value(payload_dict)
