"""Plain Activity sentences from scoped, saved facts; no operational actions."""

from datetime import timedelta
from hashlib import sha256
import json
import re
from urllib.parse import quote, urlencode

from alpha.live.dashboard_v3.filters import MARKET_TIMEZONE_OBJ
from alpha.live.ops_report import parse_timestamp_ts


# Wording describes the saved observation, never the current health of a Pod.
EVENT_DICT = {
    "build_decision_plan_created": ("cycles", "done", "Decision saved.", "decision"),
    "build_vplan_created": ("cycles", "done", "Order plan saved.", "plan"),
    "submit_vplan_completed": ("cycles", "done", "Order submission recorded.", "orders"),
    "post_execution_reconcile_completed": ("cycles", "done", "Position check recorded.", "reconcile"),
    "eod_snapshot_completed": ("cycles", "done", "Day closed. Snapshot saved.", "events"),
    "decision_plan_expired": ("cycles", "late", "Decision expired before execution.", "decision"),
    "build_decision_plan_data_dependency_error": ("alerts", "fail", "Decision blocked. Market data unavailable.", "decision"),
    "build_vplan_blocked": ("alerts", "fail", "Order plan blocked.", "plan"),
    "build_vplan_live_reference_fallback_warning": ("alerts", "unk", "Order plan used a fallback price source.", "plan"),
    "build_vplan_position_warning": ("alerts", "unk", "Saved positions need review.", "plan"),
    "submit_vplan_missing_broker_ack": ("alerts", "fail", "Broker acknowledgement missing.", "orders"),
    "exit_residual_detected": ("alerts", "fail", "A position remains after the planned exit.", "reconcile"),
    "execution_exception_parked": ("alerts", "fail", "Execution paused for review.", "events"),
    "post_execution_reconcile_failed": ("alerts", "fail", "Position check failed.", "reconcile"),
    "runner_failed": ("alerts", "fail", "Trading task failed.", "events"),
    "core5_decision_blocked": ("alerts", "fail", "Decision blocked by a safety check.", "decision"),
    "core5_eod_source_untrusted": ("alerts", "fail", "Closing snapshot source could not be verified.", "events"),
    "core5_pre_submit_account_changed": ("alerts", "fail", "Account changed before submission. Orders blocked.", "orders"),
    "scheduler_started": ("system", "done", "Scheduler started.", "events"),
    "scheduler_error_retry": ("system", "fail", "Scheduler error. Retry scheduled.", "events"),
    "operator_action_requested": ("operator", "now", "Operator action requested.", "events"),
    "manual_order_submit_requested": ("operator", "now", "Manual order requested.", "events"),
    "manual_order_submit_completed": ("operator", "now", "Manual order submission recorded.", "events"),
    "manual_order_submit_failed": ("operator", "fail", "Manual order submission failed.", "events"),
    "norgate_snapshot_sync_started": ("system", "now", "Data sync started.", "events"),
    "norgate_snapshot_sync_ready": ("system", "done", "Data sync completed.", "events"),
    "norgate_snapshot_sync_failed": ("system", "fail", "Data sync failed.", "events"),
    "norgate_snapshot_sync_waiting": ("system", "unk", "Data sync is waiting.", "events"),
    "norgate_snapshot_sync_skipped": ("system", "unk", "Data sync skipped.", "events"),
}
FOLDABLE_EVENT_SET = {"build_decision_plan_created", "build_vplan_created", "submit_vplan_completed", "post_execution_reconcile_completed"}
ACTION_LABEL_DICT = {"tick": "Trading check", "submit_vplan": "Order submission", "post_execution_reconcile": "Position check",
    "eod_snapshot": "Closing snapshot", "export_trade_sheet": "Trade sheet export", "doctor": "System check",
    "build_decision_plan": "Decision", "build_vplan": "Order plan", "cancel_vplan": "Order cancellation"}
EVIDENCE_LABEL_DICT = {"decision_plan_id_int": "Decision", "vplan_id_int": "Plan", "job_id_str": "Job",
    "order_count_int": "Orders", "broker_order_count_int": "Orders", "broker_ack_count_int": "Acknowledgements",
    "broker_order_ack_count_int": "Acknowledgements", "initial_status_str": "Initial status", "reconciliation_status_str": "Position check",
    "missing_ack_count_int": "Missing acknowledgements", "fill_count_int": "Fill records", "status_str": "Saved status",
    "reason_code_str": "Reason", "action_name_str": "Action", "delivery_status_str": "Delivery",
    "snapshot_date_str": "Data date", "market_date_str": "Session", "eod_market_date_str": "Session", "source_str": "Source",
    "ticket_id_str": "Ticket", "asset_str": "Symbol", "side_str": "Side", "quantity_int": "Shares",
    "broker_order_type_str": "Order type", "submit_ack_status_str": "Broker acknowledgement"}


def _cycle_matches_bool(event_dict, cycle_dict):
    if event_dict.get("pod_id_str") != cycle_dict.get("pod_id_str"):
        return False
    payload_dict = event_dict.get("payload_dict") or {}
    release_str = payload_dict.get("release_id_str")
    if not release_str or release_str != cycle_dict.get("release_id_str"):
        return False
    identifiers_list = [(event_dict.get(key_str), cycle_dict.get(key_str)) for key_str in ("decision_plan_id_int", "vplan_id_int")]
    return any(left_obj is not None for left_obj, _ in identifiers_list) and all(left_obj is None or left_obj == right_obj for left_obj, right_obj in identifiers_list)


def _event_row_dict(event_dict, pod_map_dict, verified_cycle_list):
    timestamp_ts = parse_timestamp_ts(event_dict.get("timestamp_str"))
    code_str = event_dict.get("event_type_str") or "unknown_event"
    if not isinstance(code_str, str) or not re.fullmatch(r"[A-Za-z0-9_.:-]{1,160}", code_str):
        code_str = "unknown_event"
    pod_id_str = event_dict.get("pod_id_str") or ""
    payload_dict = event_dict.get("payload_dict") or {}
    type_str, state_str, title_str, tab_str = EVENT_DICT.get(code_str, ("system" if not pod_id_str else "cycles", "unk", "Saved event.", "events"))
    level_str = str(event_dict.get("level_str") or "").lower()
    status_set = {str(payload_dict.get(key_str) or "").lower() for key_str in
        ("status_str", "initial_status_str", "reconciliation_status_str", "vplan_status_str", "decision_plan_status_str", "submit_ack_status_str")}
    if level_str in {"critical", "error", "fatal"} or status_set & {"failed", "blocked", "error", "missing_critical"} or (payload_dict.get("missing_ack_count_int") or 0) > 0:
        state_str = "fail"
    elif status_set & {"expired", "late"}:
        state_str = "late"
    elif level_str in {"warning", "warn"} and state_str == "done":
        state_str = "unk"
    if code_str not in EVENT_DICT and (level_str in {"warning", "warn", "critical", "error", "fatal"} or state_str in {"fail", "late"}):
        type_str, title_str = "alerts", "Event needs review."
    if code_str not in EVENT_DICT and code_str.startswith(("notification_", "notification.", "alert_", "discord_")):
        type_str, title_str = "alerts", "Alert event recorded."
    if code_str not in EVENT_DICT and code_str.startswith(("operator_", "manual_order_")):
        type_str, title_str = "operator", "Operator event recorded."
    if code_str == "norgate_snapshot_sync_skipped" and payload_dict.get("reason_code_str") == "sync_failure_cooldown":
        state_str, title_str = "fail", "Data sync waiting after a failure."
    if code_str == "operator_action_requested":
        action_str = payload_dict.get("action_name_str") or payload_dict.get("action_str")
        title_str = ACTION_LABEL_DICT.get(action_str, "Operator action") + " requested."
    if event_dict.get("notification_delivery_str") in {"delivered", "failed", "queued"}:
        delivery_str = event_dict["notification_delivery_str"]
        type_str, state_str = "alerts", {"delivered": "done", "failed": "fail", "queued": "now"}[delivery_str]
        title_str = {"delivered": "Alert delivered.", "failed": "Alert delivery failed.", "queued": "Alert queued."}[delivery_str]
    detail_list = []
    for field_str, suffix_str in (("order_count_int", "order"), ("fill_count_int", "fill record"), ("missing_ack_count_int", "missing acknowledgement")):
        count_int = payload_dict.get(field_str)
        if type(count_int) is int and count_int >= 0:
            detail_list.append(f"{count_int} {suffix_str}{'' if count_int == 1 else 's'}")
    # Source readers allowlist/redact payloads; presentation still exposes only
    # these named scalars. Arbitrary errors, paths and broker accounts stay out.
    evidence_dict = {**payload_dict, **{key_str: event_dict.get(key_str) for key_str in ("decision_plan_id_int", "vplan_id_int")},
        "delivery_status_str": event_dict.get("notification_delivery_str", ""), "source_str": event_dict.get("source_str", "Saved log")}
    evidence_list = [{"label_str": label_str, "value_str": str(evidence_dict[key_str])[:200]} for key_str, label_str in EVIDENCE_LABEL_DICT.items()
        if isinstance(evidence_dict.get(key_str), (str, int, float, bool)) and evidence_dict[key_str] != ""]
    cycle_str = next((f"{prefix_str}:{event_dict[key_str]}" for key_str, prefix_str in (("vplan_id_int", "vplan"), ("decision_plan_id_int", "decision"))
        if type(event_dict.get(key_str)) is int and event_dict[key_str] > 0), "")
    # A numeric plan ID can be reused after a DB replacement. Only link to a
    # currently verified matching release/cycle; otherwise retain inline proof.
    link_verified_bool = any(_cycle_matches_bool(event_dict, cycle_dict) for cycle_dict in verified_cycle_list)
    evidence_url_str = ("/pods/" + quote(pod_id_str, safe="") + "?" + urlencode({"cycle": cycle_str, "tab": tab_str}) + "#evidence") if pod_id_str and cycle_str and link_verified_bool else ""
    market_ts = timestamp_ts.astimezone(MARKET_TIMEZONE_OBJ)
    identity_str = json.dumps(event_dict, sort_keys=True, default=str)
    return {"id_str": "event-" + sha256(identity_str.encode()).hexdigest()[:20], "timestamp_str": timestamp_ts.isoformat(),
        "time_str": market_ts.strftime("%H:%M:%S"), "day_str": market_ts.date().isoformat(), "day_label_str": market_ts.strftime("%a %Y-%m-%d"),
        "pod_id_str": pod_id_str, "pod_name_str": pod_map_dict.get(pod_id_str, "System"), "type_str": type_str,
        "related_pod_id_list": payload_dict.get("related_pod_id_list", []),
        "state_str": state_str, "title_str": title_str, "detail_str": " · ".join(detail_list), "code_str": code_str,
        "evidence_list": evidence_list, "evidence_url_str": evidence_url_str, "child_list": [],
        "other_event_bool": code_str not in EVENT_DICT and level_str not in {"warning", "warn", "critical", "error", "fatal"}
            and state_str == "unk" and type_str not in {"alerts", "operator"}}


def _group_other_events_list(row_list):
    """Keep unclassified routine rows compact without burying warnings/failures."""
    result_list, group_map_dict = [], {}
    for row_dict in row_list:
        if not row_dict.get("other_event_bool"):
            result_list.append(row_dict)
            continue
        key_tuple = (row_dict["day_str"], row_dict["pod_id_str"], tuple(row_dict["related_pod_id_list"]), row_dict["type_str"])
        group_map_dict.setdefault(key_tuple, []).append(row_dict)
    for key_tuple, child_list in group_map_dict.items():
        child_list.sort(key=lambda row_dict: (row_dict["timestamp_str"], row_dict["id_str"]))
        count_int = len(child_list)
        result_list.append({**child_list[-1], "id_str": "other-" + sha256(repr(key_tuple).encode()).hexdigest()[:20],
            "state_str": "now", "title_str": f"Other events ({count_int}).", "detail_str": "",
            "code_str": " · ".join(sorted({row_dict["code_str"] for row_dict in child_list})),
            "evidence_url_str": "", "evidence_list": [{"label_str": "Saved events", "value_str": str(count_int)}],
            "child_list": child_list, "child_label_str": "events"})
    return result_list


def build_activity_page_dict(overview_dict, source_dict, cycle_dict, *, as_of_ts, days_int):
    """Only matching, fully healthy cycles absorb routine log records."""
    pod_map_dict = {pod_dict["pod_id_str"]: pod_dict["name_str"] for pod_dict in overview_dict["pod_list"]}
    from_ts = (as_of_ts.astimezone(MARKET_TIMEZONE_OBJ).replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=days_int - 1))
    event_list = [event_dict for event_dict in source_dict.get("event_list", [])
        if (timestamp_ts := parse_timestamp_ts(event_dict.get("timestamp_str"))) is not None and from_ts <= timestamp_ts <= as_of_ts
        and (not event_dict.get("pod_id_str") or event_dict["pod_id_str"] in pod_map_dict)]
    paired_list = [(event_dict, _event_row_dict(event_dict, pod_map_dict, cycle_dict.get("row_list", []))) for event_dict in event_list]
    row_list, folded_list = [], []
    for row_dict in cycle_dict.get("row_list", []):
        timestamp_ts = parse_timestamp_ts(row_dict.get("timestamp_str"))
        if timestamp_ts is None or not from_ts <= timestamp_ts <= as_of_ts or row_dict.get("pod_id_str") not in pod_map_dict:
            continue
        if row_dict.get("stage_str") == "eod" and any(event_dict.get("pod_id_str") == row_dict["pod_id_str"]
            and event_dict.get("event_type_str") == "eod_snapshot_completed"
            and (event_dict.get("payload_dict") or {}).get("eod_market_date_str") == row_dict["day_str"] for event_dict in event_list):
            continue
        if row_dict.get("child_list"):
            # Later success does not erase a warning, late step or failure in
            # the same saved cycle. Ambiguous identity cannot fold a log row.
            issues_bool = any(_cycle_matches_bool(event_dict, row_dict) and (event_row_dict["state_str"] != "done" or event_row_dict["type_str"] == "alerts")
                for event_dict, event_row_dict in paired_list if event_row_dict["type_str"] != "operator")
            if issues_bool:
                continue
            folded_list.append(row_dict)
        row_list.append(row_dict)
    for event_dict, row_dict in paired_list:
        if row_dict["code_str"] in FOLDABLE_EVENT_SET and row_dict["state_str"] == "done" and any(_cycle_matches_bool(event_dict, group_dict) for group_dict in folded_list):
            continue
        row_list.append(row_dict)
    row_list = _group_other_events_list(row_list)
    row_list.sort(key=lambda row_dict: (row_dict["timestamp_str"], row_dict["id_str"]), reverse=True)
    warning_list = list(dict.fromkeys([*source_dict.get("warning_list", []), *cycle_dict.get("warning_list", [])]))
    if len(row_list) > 500:
        warning_list.append("Showing the newest 500 events. Older records are outside this view.")
        row_list = row_list[:500]
    coverage_dict = source_dict.get("coverage_dict") or {}
    coverage_label_str = ""
    if coverage_dict and not coverage_dict.get("complete_bool"):
        oldest_ts = parse_timestamp_ts(coverage_dict.get("scanned_from_timestamp_str"))
        coverage_label_str = "Partial log history"
        if oldest_ts is not None:
            coverage_label_str += " · scanned to " + oldest_ts.astimezone(MARKET_TIMEZONE_OBJ).strftime("%m-%d %H:%M ET")
    return {"row_list": row_list, "days_int": days_int, "as_of_timestamp_str": as_of_ts.isoformat(),
        "coverage_label_str": coverage_label_str,
        "history_complete_bool": not warning_list and coverage_dict.get("complete_bool", True),
        "feed_available_bool": source_dict.get("feed_available_bool", not source_dict.get("warning_list")),
        "storage_key_str": "alpha.ops.v4.activity.live." + source_dict.get("scope_key_str", "unavailable"),
        "pod_filter_list": [{"pod_id_str": pod_id_str, "name_str": name_str} for pod_id_str, name_str in pod_map_dict.items()],
        "warning_list": warning_list, "load_older_url_str": ""}
