from __future__ import annotations

from datetime import UTC, datetime
import json
import socket
from typing import Any
import urllib.error
import urllib.request

from alpha.live import scheduler_utils


OPS_REPORT_SCHEMA_VERSION_STR = "live_ops_inspector.v1"
HEARTBEAT_SCHEMA_VERSION_STR = "live_ops_inspector_heartbeat.v1"
DEFAULT_STALE_AFTER_SECONDS_INT = 900
DEFAULT_HEARTBEAT_TIMEOUT_SECONDS_FLOAT = 3.0
SEVERITY_RANK_DICT = {"red": 0, "yellow": 1, "gray": 2, "green": 3}
VPLAN_EXECUTION_PROVEN_STATUS_SET = {"submitting", "submitted", "completed"}


def utc_now_ts() -> datetime:
    return datetime.now(tz=UTC)


def normalize_severity_str(severity_obj: object) -> str:
    text_str = str(severity_obj or "gray").strip().lower()
    if "red" in text_str or "error" in text_str or "fail" in text_str or "block" in text_str:
        return "red"
    if (
        "yellow" in text_str
        or "warn" in text_str
        or "wait" in text_str
        or "running" in text_str
        or "queued" in text_str
    ):
        return "yellow"
    if text_str == "green":
        return "green"
    return "gray"


def parse_timestamp_ts(timestamp_str: str | None) -> datetime | None:
    if not timestamp_str:
        return None
    normalized_str = str(timestamp_str).replace("Z", "+00:00")
    try:
        timestamp_ts = datetime.fromisoformat(normalized_str)
    except ValueError:
        return None
    if timestamp_ts.tzinfo is None:
        return timestamp_ts.replace(tzinfo=UTC)
    return timestamp_ts.astimezone(UTC)


def build_ops_report_dict(
    summary_dict: dict[str, Any],
    *,
    mode_str: str | None = None,
    generated_at_ts: datetime | None = None,
    stale_after_seconds_int: int = DEFAULT_STALE_AFTER_SECONDS_INT,
    vps_id_str: str | None = None,
) -> dict[str, Any]:
    generated_at_ts = utc_now_ts() if generated_at_ts is None else generated_at_ts.astimezone(UTC)
    source_as_of_ts = parse_timestamp_ts(str(summary_dict.get("as_of_timestamp_str") or ""))
    source_stale_bool = _source_stale_bool(
        source_as_of_ts=source_as_of_ts,
        generated_at_ts=generated_at_ts,
        stale_after_seconds_int=stale_after_seconds_int,
    )
    pod_row_dict_list = [
        row_dict
        for row_dict in summary_dict.get("pod_row_dict_list") or []
        if mode_str is None or str(row_dict.get("mode_str") or "") == mode_str
    ]
    pod_report_dict_list = [
        _build_pod_report_dict(
            row_dict,
            source_stale_bool=source_stale_bool,
            generated_at_ts=generated_at_ts,
        )
        for row_dict in pod_row_dict_list
    ]
    severity_count_dict = _build_severity_count_dict(pod_report_dict_list)
    overall_severity_str = _overall_severity_str(pod_report_dict_list, source_stale_bool)
    return {
        "schema_version_str": OPS_REPORT_SCHEMA_VERSION_STR,
        "vps_id_str": vps_id_str or socket.gethostname(),
        "mode_str": mode_str or "all",
        "generated_at_utc_str": generated_at_ts.isoformat(),
        "source_as_of_utc_str": None if source_as_of_ts is None else source_as_of_ts.isoformat(),
        "source_age_seconds_float": _source_age_seconds_float(source_as_of_ts, generated_at_ts),
        "stale_after_seconds_int": int(stale_after_seconds_int),
        "source_stale_bool": bool(source_stale_bool),
        "overall_severity_str": overall_severity_str,
        "overall_reason_str": _overall_reason_str(
            pod_report_dict_list=pod_report_dict_list,
            source_stale_bool=source_stale_bool,
            source_as_of_ts=source_as_of_ts,
        ),
        "pod_count_int": len(pod_report_dict_list),
        "severity_count_dict": severity_count_dict,
        "pod_report_dict_list": pod_report_dict_list,
        "heartbeat_payload_dict": build_heartbeat_payload_dict(
            generated_at_ts=generated_at_ts,
            vps_id_str=vps_id_str,
        ),
    }


def build_heartbeat_payload_dict(
    *,
    generated_at_ts: datetime | None = None,
    vps_id_str: str | None = None,
) -> dict[str, Any]:
    generated_at_ts = utc_now_ts() if generated_at_ts is None else generated_at_ts.astimezone(UTC)
    return {
        "schema_version_str": HEARTBEAT_SCHEMA_VERSION_STR,
        "vps_id_str": vps_id_str or socket.gethostname(),
        "sent_at_utc_str": generated_at_ts.isoformat(),
    }


def apply_consumer_staleness_dict(
    report_dict: dict[str, Any],
    *,
    consumed_at_ts: datetime | None = None,
    stale_after_seconds_int: int | None = None,
) -> dict[str, Any]:
    consumed_at_ts = utc_now_ts() if consumed_at_ts is None else consumed_at_ts.astimezone(UTC)
    result_dict = dict(report_dict)
    generated_at_ts = parse_timestamp_ts(str(report_dict.get("generated_at_utc_str") or ""))
    effective_stale_after_seconds_int = int(
        stale_after_seconds_int
        if stale_after_seconds_int is not None
        else report_dict.get("stale_after_seconds_int")
        or DEFAULT_STALE_AFTER_SECONDS_INT
    )
    age_seconds_float = _source_age_seconds_float(generated_at_ts, consumed_at_ts)
    consumer_stale_bool = (
        age_seconds_float is None
        or age_seconds_float > float(effective_stale_after_seconds_int)
    )
    result_dict["consumer_checked_at_utc_str"] = consumed_at_ts.isoformat()
    result_dict["consumer_report_age_seconds_float"] = age_seconds_float
    result_dict["consumer_stale_bool"] = bool(consumer_stale_bool)
    if consumer_stale_bool:
        result_dict["overall_severity_str"] = "red"
        result_dict["overall_reason_str"] = (
            "Inspector report is stale at the consumer; stale green is not allowed."
        )
    return result_dict


def post_heartbeat_bool(
    heartbeat_url_str: str,
    payload_dict: dict[str, Any],
    *,
    timeout_seconds_float: float = DEFAULT_HEARTBEAT_TIMEOUT_SECONDS_FLOAT,
) -> bool:
    if not heartbeat_url_str:
        return False
    request_obj = urllib.request.Request(
        heartbeat_url_str,
        data=json.dumps(payload_dict).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request_obj, timeout=timeout_seconds_float) as response_obj:
            return 200 <= response_obj.status < 300
    except (OSError, TimeoutError, urllib.error.URLError):
        return False


def _build_pod_report_dict(
    row_dict: dict[str, Any],
    *,
    source_stale_bool: bool,
    generated_at_ts: datetime,
) -> dict[str, Any]:
    required_action_dict = row_dict.get("required_action_dict") or {}
    debug_summary_dict = row_dict.get("debug_summary_dict") or {}
    base_severity_str = normalize_severity_str(
        debug_summary_dict.get("severity_str")
        or required_action_dict.get("severity_str")
        or row_dict.get("health_str")
    )
    missed_window_bool = _missed_execution_window_bool(row_dict, generated_at_ts=generated_at_ts)
    severity_str = "red" if source_stale_bool or missed_window_bool else base_severity_str
    reason_code_str = _pod_reason_code_str(
        row_dict,
        source_stale_bool=source_stale_bool,
        missed_window_bool=missed_window_bool,
    )
    reason_str = _pod_reason_str(
        row_dict,
        reason_code_str=reason_code_str,
        source_stale_bool=source_stale_bool,
        missed_window_bool=missed_window_bool,
    )
    action_label_str = (
        "Review missed window"
        if missed_window_bool and not source_stale_bool
        else str(required_action_dict.get("label_str") or "Inspect POD")
    )
    inspect_command_name_str = str(required_action_dict.get("inspect_command_name_str") or "status")
    action_detail_str = (
        reason_str
        if missed_window_bool and not source_stale_bool
        else str(required_action_dict.get("detail_str") or reason_str)
    )
    return {
        "pod_id_str": str(row_dict.get("pod_id_str") or ""),
        "mode_str": str(row_dict.get("mode_str") or ""),
        "account_route_str": str(row_dict.get("account_route_str") or ""),
        "strategy_import_str": str(row_dict.get("strategy_import_str") or ""),
        "severity_str": severity_str,
        "status_str": str(
            required_action_dict.get("label_str")
            or row_dict.get("next_action_str")
            or row_dict.get("health_str")
            or "unknown"
        ),
        "reason_code_str": reason_code_str,
        "reason_str": reason_str,
        "next_operator_action_dict": {
            "label_str": action_label_str,
            "inspect_command_name_str": inspect_command_name_str,
            "detail_str": action_detail_str,
            "manual_only_bool": True,
        },
        "evidence_dict": {
            "db_status_str": row_dict.get("db_status_str"),
            "next_action_str": row_dict.get("next_action_str"),
            "reason_code_str": row_dict.get("reason_code_str"),
            "execution_policy_str": row_dict.get("execution_policy_str"),
            "latest_decision_plan_id_int": row_dict.get("latest_decision_plan_id_int"),
            "latest_decision_plan_status_str": row_dict.get("latest_decision_plan_status_str"),
            "latest_decision_execution_policy_str": row_dict.get(
                "latest_decision_execution_policy_str"
            ),
            "latest_decision_plan_target_execution_timestamp_str": row_dict.get(
                "latest_decision_plan_target_execution_timestamp_str"
            ),
            "latest_vplan_id_int": row_dict.get("latest_vplan_id_int"),
            "latest_vplan_status_str": row_dict.get("latest_vplan_status_str"),
            "latest_vplan_target_execution_timestamp_str": row_dict.get(
                "latest_vplan_target_execution_timestamp_str"
            ),
            "latest_pod_state_timestamp_str": row_dict.get("latest_pod_state_timestamp_str"),
            "latest_event_timestamp_str": row_dict.get("latest_event_timestamp_str"),
            "missing_ack_count_int": int(row_dict.get("missing_ack_count_int") or 0),
            "exception_count_int": int(row_dict.get("exception_count_int") or 0),
        },
    }


def _source_age_seconds_float(
    source_as_of_ts: datetime | None,
    generated_at_ts: datetime,
) -> float | None:
    if source_as_of_ts is None:
        return None
    return max(0.0, (generated_at_ts - source_as_of_ts).total_seconds())


def _source_stale_bool(
    *,
    source_as_of_ts: datetime | None,
    generated_at_ts: datetime,
    stale_after_seconds_int: int,
) -> bool:
    age_seconds_float = _source_age_seconds_float(source_as_of_ts, generated_at_ts)
    if age_seconds_float is None:
        return True
    return age_seconds_float > float(stale_after_seconds_int)


def _build_severity_count_dict(
    pod_report_dict_list: list[dict[str, Any]],
) -> dict[str, int]:
    severity_count_dict = {"red": 0, "yellow": 0, "gray": 0, "green": 0}
    for pod_report_dict in pod_report_dict_list:
        severity_str = normalize_severity_str(pod_report_dict.get("severity_str"))
        severity_count_dict[severity_str] = int(severity_count_dict[severity_str]) + 1
    return severity_count_dict


def _overall_severity_str(
    pod_report_dict_list: list[dict[str, Any]],
    source_stale_bool: bool,
) -> str:
    if source_stale_bool:
        return "red"
    if len(pod_report_dict_list) == 0:
        return "gray"
    return min(
        (normalize_severity_str(pod_report_dict.get("severity_str")) for pod_report_dict in pod_report_dict_list),
        key=lambda severity_str: SEVERITY_RANK_DICT.get(severity_str, 99),
    )


def _overall_reason_str(
    *,
    pod_report_dict_list: list[dict[str, Any]],
    source_stale_bool: bool,
    source_as_of_ts: datetime | None,
) -> str:
    if source_as_of_ts is None:
        return "Inspector source summary has no readable as_of timestamp."
    if source_stale_bool:
        return "Inspector source summary is stale; stale green is not allowed."
    if len(pod_report_dict_list) == 0:
        return "No enabled PODs were found for this report scope."
    red_count_int = sum(
        1 for pod_report_dict in pod_report_dict_list if pod_report_dict["severity_str"] == "red"
    )
    if red_count_int > 0:
        return f"{red_count_int} POD(s) need operator action."
    yellow_count_int = sum(
        1 for pod_report_dict in pod_report_dict_list if pod_report_dict["severity_str"] == "yellow"
    )
    if yellow_count_int > 0:
        return f"{yellow_count_int} POD(s) are waiting for expected timing or review."
    gray_count_int = sum(
        1 for pod_report_dict in pod_report_dict_list if pod_report_dict["severity_str"] == "gray"
    )
    if gray_count_int > 0:
        return f"{gray_count_int} POD(s) have unknown or missing evidence."
    return "All enabled PODs in scope are proven fresh and green."


def _pod_reason_code_str(
    row_dict: dict[str, Any],
    *,
    source_stale_bool: bool,
    missed_window_bool: bool,
) -> str:
    if source_stale_bool:
        return "inspector_source_stale"
    if missed_window_bool:
        return "missed_execution_window"
    required_action_dict = row_dict.get("required_action_dict") or {}
    debug_summary_dict = row_dict.get("debug_summary_dict") or {}
    return str(
        row_dict.get("reason_code_str")
        or debug_summary_dict.get("verdict_label_str")
        or required_action_dict.get("reason_code_str")
        or required_action_dict.get("label_str")
        or "unknown"
    )


def _pod_reason_str(
    row_dict: dict[str, Any],
    *,
    reason_code_str: str,
    source_stale_bool: bool,
    missed_window_bool: bool,
) -> str:
    if source_stale_bool:
        return "The source dashboard summary is stale; this POD cannot be shown as green."
    if missed_window_bool:
        return (
            "Execution window is past without proven submitted or completed VPlan evidence; "
            "operator must review manually."
        )
    required_action_dict = row_dict.get("required_action_dict") or {}
    debug_summary_dict = row_dict.get("debug_summary_dict") or {}
    return str(
        required_action_dict.get("reason_str")
        or required_action_dict.get("detail_str")
        or debug_summary_dict.get("primary_reason_str")
        or reason_code_str
    )


def _missed_execution_window_bool(
    row_dict: dict[str, Any],
    *,
    generated_at_ts: datetime,
) -> bool:
    reason_code_str = str(row_dict.get("reason_code_str") or "")
    next_action_str = str(row_dict.get("next_action_str") or "")
    if reason_code_str == "submission_window_expired" or next_action_str == "expire_stale":
        return True

    latest_vplan_status_str = str(row_dict.get("latest_vplan_status_str") or "").lower()
    latest_decision_plan_status_str = str(
        row_dict.get("latest_decision_plan_status_str") or ""
    ).lower()
    latest_decision_target_ts = parse_timestamp_ts(
        row_dict.get("latest_decision_plan_target_execution_timestamp_str")
    )
    latest_vplan_target_ts = parse_timestamp_ts(
        row_dict.get("latest_vplan_target_execution_timestamp_str")
    )
    decision_execution_policy_str = _execution_policy_str(row_dict)

    if (
        latest_decision_plan_status_str == "planned"
        and _execution_window_expired_bool(
            decision_execution_policy_str,
            latest_decision_target_ts,
            generated_at_ts,
        )
        and latest_vplan_status_str not in VPLAN_EXECUTION_PROVEN_STATUS_SET
    ):
        return True
    if (
        latest_vplan_status_str == "ready"
        and _execution_window_expired_bool(
            decision_execution_policy_str,
            latest_vplan_target_ts,
            generated_at_ts,
        )
    ):
        return True
    return False


def _execution_policy_str(row_dict: dict[str, Any]) -> str:
    return str(
        row_dict.get("latest_vplan_execution_policy_str")
        or row_dict.get("latest_decision_execution_policy_str")
        or row_dict.get("execution_policy_str")
        or ""
    )


def _execution_window_expired_bool(
    execution_policy_str: str,
    timestamp_ts: datetime | None,
    generated_at_ts: datetime,
) -> bool:
    if timestamp_ts is None:
        return False
    if not execution_policy_str:
        return timestamp_ts <= generated_at_ts
    return scheduler_utils.is_execution_window_expired_bool(
        execution_policy_str,
        timestamp_ts,
        generated_at_ts,
    )
