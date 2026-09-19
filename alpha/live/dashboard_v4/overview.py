"""LIVE Overview presentation. Existing readers retain accounting/execution rules."""

from copy import deepcopy
from datetime import datetime

from alpha.live.dashboard_v3.client_operations import SOURCE_MAX_AGE_SECONDS_INT, active_account_list
from alpha.live.dashboard_v3.filters import MARKET_TIMEZONE_OBJ
from alpha.live.dashboard_v3.health import build_health_rollup
from alpha.live.dashboard_v3.operator_tools import strategy_display_name_str
from alpha.live.dashboard_v3.schedule import build_market_status, _operator_action_required_bool
from alpha.live.dashboard_v4.cycle import build_cycle_view_dict
from alpha.live.dashboard_v4.finance import build_financial_overview_dict
from alpha.live.ops_report import parse_timestamp_ts


STATE_CLASS_DICT = {
    "Done": "done", "Now": "now", "Planned": "next", "Late": "late",
    "Failed": "fail", "None": "skip", "Unknown": "unk",
    "green": "done", "blue": "now", "amber": "late", "yellow": "late",
    "red": "fail", "gray": "unk", "neutral": "skip",
}
STATE_RANK_DICT = {"fail": 0, "late": 1, "unk": 2, "now": 3, "next": 4, "done": 5, "skip": 6}


def _state_str(value_str):
    return STATE_CLASS_DICT.get(value_str, value_str if value_str in STATE_RANK_DICT else "unk")


def _duration_str(seconds_float):
    seconds_int = max(0, int(seconds_float))
    hours_int, remaining_int = divmod(seconds_int, 3600)
    minutes_int, seconds_int = divmod(remaining_int, 60)
    return f"{hours_int:02}:{minutes_int:02}:{seconds_int:02}"


def _data_time_str(value_str):
    if len(value_str) == 10:
        return value_str  # A market session date is not a UTC timestamp.
    timestamp_ts = parse_timestamp_ts(value_str)
    return timestamp_ts.astimezone(MARKET_TIMEZONE_OBJ).strftime("%m-%d %H:%M:%S") if timestamp_ts else "unknown"


def _action_required_bool(row_dict, cycle_dict):
    required_dict = row_dict.get("required_action_dict") or {}
    if (required_dict.get("severity_str") == "yellow"
        and (row_dict.get("next_action_str"), required_dict.get("label_str")) in {
            ("submit_vplan", "VPlan ready"), ("build_vplan", "Build VPlan"),
        }):
        submit_dict = cycle_dict["step_dict_list"][3]
        if submit_dict["planned_timestamp_str"] and submit_dict["state_str"] in {"Planned", "Now"}:
            return False
    return _operator_action_required_bool(required_dict)


def build_overview_dict(workspace_dict, snapshot_obj, provider_obj, *, as_of_ts: datetime,
                        period_str="3M", demo_bool=False):
    client_dict = workspace_dict["client_dict"]
    source_dict = workspace_dict.get("summary_dict") or {}
    source_ts = parse_timestamp_ts(source_dict.get("as_of_timestamp_str"))
    fresh_bool = bool(source_ts and 0 <= (as_of_ts - source_ts).total_seconds() <= SOURCE_MAX_AGE_SECONDS_INT)
    fresh_bool = fresh_bool and not bool(workspace_dict.get("operations_error_str"))
    account_list = workspace_dict.get("operations_account_list")
    if account_list is None:
        account_list = active_account_list(client_dict, as_of_ts)
    raw_row_list = [row_dict for row_dict in source_dict.get("pod_row_dict_list") or []
                    if isinstance(row_dict, dict) and row_dict.get("mode_str") == "live"]
    pod_list, attention_list, scoped_row_list, health_row_list = [], [], [], []
    for account_dict in account_list:
        pod_id_str = account_dict["pod_id"]
        matches_list = [row_dict for row_dict in raw_row_list if row_dict.get("pod_id_str") == pod_id_str]
        matched_bool = len(matches_list) == 1 and matches_list[0].get("account_route_str") == account_dict["account_route"]
        row_dict = deepcopy(matches_list[0]) if matched_bool else {
            "mode_str": "live", "pod_id_str": pod_id_str, "db_status_str": "unknown",
        }
        row_dict["source_stale_bool"] = not fresh_bool or not matched_bool
        row_dict.setdefault("as_of_timestamp_str", source_dict.get("as_of_timestamp_str"))
        if fresh_bool and matched_bool and hasattr(provider_obj, "get_cycle_evidence_dict"):
            row_dict["cycle_evidence_dict"] = provider_obj.get_cycle_evidence_dict(row_dict, as_of_ts=source_ts)
        scoped_row_list.append(row_dict)
        cycle_dict = build_cycle_view_dict(row_dict, now_ts=as_of_ts)
        action_required_bool = _action_required_bool(row_dict, cycle_dict)
        # Normalize only the expected current-session EOD wait for the health
        # roll-up. Keep the original saved assessment for all other consumers.
        health_row_dict = deepcopy(row_dict)
        eod_step_dict = cycle_dict["step_dict_list"][-1]
        eod_dict = row_dict.get("eod_snapshot_dict") or {}
        if eod_step_dict["state_str"] == "Now" and eod_dict.get("status_str") in {"due_missing", "blocked_by_execution"}:
            for item_dict in (health_row_dict.get("data_freshness_dict") or {}).get("item_dict_list") or []:
                if item_dict.get("label_str") == "EOD Snapshot" and item_dict.get("severity_str") == "yellow":
                    item_dict.update(severity_str="green", detail_str="Waiting for scheduled capture")
        health_row_list.append(health_row_dict)
        pod_state_str = "skip" if cycle_dict["pill_str"] == "Idle" else _state_str(cycle_dict.get("tone_str"))
        policy_str = row_dict.get("execution_policy_str") or ""
        monthly_bool = policy_str == "next_month_first_open" or row_dict.get("signal_clock_str") == "month_end_snapshot_ready"
        cadence_str = ("Monthly" if monthly_bool else "Daily") if policy_str else "—"
        session_str = "Close" if policy_str == "same_day_moc" else "First open" if monthly_bool else "Open"
        name_str = account_dict.get("display_name") or strategy_display_name_str(row_dict)
        required_dict = row_dict.get("required_action_dict") or {}
        step_list = [{
            "name_str": step_dict["label_str"], "state_str": _state_str(step_dict["state_str"]),
            "detail_str": step_dict.get("fact_str") or "—",
            "time_str": step_dict.get("actual_time_str") or (
                "due " + step_dict["planned_time_str"] if step_dict.get("planned_time_str") else ""),
        } for step_dict in cycle_dict["step_dict_list"]]
        next_time_str = cycle_dict.get("next_time_str") or ""
        next_ts = parse_timestamp_ts(cycle_dict.get("next_timestamp_str"))
        next_detail_str = "in " + _duration_str((next_ts - as_of_ts).total_seconds()) if next_ts and next_ts > as_of_ts else ""
        next_str = cycle_dict.get("next_str") or "—"
        now_str, now_detail_str = cycle_dict.get("now_str") or "Unknown", cycle_dict.get("now_detail_str") or ""
        if pod_state_str in {"fail", "late"}:
            if row_dict.get("latest_submit_ack_status_str") == "missing_critical" or (row_dict.get("missing_ack_count_int") or 0) > 0:
                now_str, now_detail_str = "ACK missing", cycle_dict["now_str"]
            next_str = required_dict.get("label_str") if action_required_bool else "Review " + next_str
            next_time_str, next_detail_str = "", "you · now"
        pod_list.append({
            "pod_id_str": pod_id_str, "name_str": name_str,
            "cadence_str": f"{cadence_str} · {session_str}" if policy_str else "Schedule unknown",
            "cadence_short_str": "M" if monthly_bool else "D" if policy_str else "—",
            "pill_str": cycle_dict["pill_str"], "state_str": pod_state_str,
            "now_str": now_str, "now_detail_str": now_detail_str,
            "next_str": next_str, "next_time_str": next_time_str, "next_detail_str": next_detail_str,
            "step_list": step_list,
        })
        database_failed_bool = row_dict.get("db_status_str") in {"missing", "error"}
        if fresh_bool and matched_bool and (database_failed_bool
            or action_required_bool or pod_state_str in {"fail", "late"}):
            event_ts = parse_timestamp_ts(required_dict.get("timestamp_str") or row_dict.get("latest_event_timestamp_str"))
            age_str = "—" if event_ts is None or event_ts > as_of_ts else _duration_str((as_of_ts - event_ts).total_seconds())
            attention_list.append({
                "state_str": "fail" if database_failed_bool or required_dict.get("severity_str") == "red" or pod_state_str == "fail" else "late",
                "pod_name_str": name_str,
                "title_str": "State DB unavailable." if database_failed_bool else (required_dict.get("label_str") or cycle_dict["now_str"]) if action_required_bool else cycle_dict["now_str"],
                "detail_str": "Cycle evidence cannot be read." if database_failed_bool else (required_dict.get("reason_str") or required_dict.get("detail_str") or "") if action_required_bool else cycle_dict["now_detail_str"],
                "age_str": age_str,
            })

    scoped_summary_dict = {**source_dict, "pod_row_dict_list": scoped_row_list}
    health_obj = build_health_rollup({**source_dict, "pod_row_dict_list": health_row_list}, mode_str="live")
    system_state_str = _state_str(health_obj.severity_str) if fresh_bool and pod_list else "unk"
    if fresh_bool and any(row_dict.get("db_status_str") in {"missing", "error"} for row_dict in scoped_row_list):
        system_state_str = "fail"
    system_label_str = {"done": "System OK", "fail": "System needs action", "late": "System needs review"}.get(system_state_str, "System unknown")
    market_obj = build_market_status(now_dt=as_of_ts)
    transition_ts = parse_timestamp_ts(market_obj.next_transition_timestamp_str)
    market_detail_str = market_obj.reason_label_str
    if market_obj.status_label_str == "Market open" and transition_ts:
        market_detail_str = "closes in " + _duration_str((transition_ts - as_of_ts).total_seconds())
    live_state_str = min([pod_dict["state_str"] for pod_dict in pod_list] + [system_state_str], key=STATE_RANK_DICT.get)
    if not fresh_bool:
        verdict_str, verdict_detail_str = "Status unknown.", "Saved operations are missing or out of date."
    elif not pod_list:
        verdict_str, verdict_detail_str = "No LIVE pods verified.", "Check the local release configuration."
    elif attention_list:
        count_int = len(attention_list)
        verdict_str = f"{count_int} pod{'s' if count_int != 1 else ''} need{'s' if count_int == 1 else ''} action."
        verdict_detail_str = "Review the saved evidence."
    elif system_state_str != "done" or any(pod_dict["state_str"] == "unk" for pod_dict in pod_list):
        verdict_str, verdict_detail_str = "Status needs review.", "Some saved evidence is unavailable."
    else:
        verdict_str, verdict_detail_str = "No action needed.", market_obj.reason_label_str if market_obj.status_label_str == "Market closed" else "Pods are on track."
    data_session_list = []
    for row_dict in scoped_row_list:
        for item_dict in (row_dict.get("data_freshness_dict") or {}).get("item_dict_list", []):
            if item_dict.get("label_str") == "Norgate" and item_dict.get("value_str"):
                data_session_list.append(str(item_dict["value_str"]))
    local_ts = as_of_ts.astimezone(MARKET_TIMEZONE_OBJ)
    overview_dict = {
        "demo_bool": demo_bool, "period_str": period_str,
        "clock_str": local_ts.strftime("%H:%M:%S"), "date_str": local_ts.strftime("%a %m-%d"),
        "updated_str": source_ts.astimezone(MARKET_TIMEZONE_OBJ).strftime("%H:%M:%S") if source_ts else "—",
        "source_fresh_bool": fresh_bool,
        "source_valid_ms_int": max(0, int((SOURCE_MAX_AGE_SECONDS_INT - (as_of_ts - source_ts).total_seconds()) * 1000)) if fresh_bool else 0,
        "market_dict": {"state_str": "now" if market_obj.status_label_str == "Market open" else "skip",
                        "label_str": market_obj.status_label_str, "detail_str": market_detail_str},
        "system_dict": {"state_str": system_state_str, "label_str": system_label_str,
                        "detail_str": "Data " + _data_time_str(min(data_session_list)) if data_session_list else "Data unknown"},
        "live_state_str": live_state_str, "verdict_str": verdict_str, "verdict_detail_str": verdict_detail_str,
        "attention_list": attention_list, "pod_list": pod_list,
    }
    # Reuse the same scoped assessment for cash; foreign mode rows never enter
    # either the operating state or the financial view adapter.
    finance_workspace_dict = {**workspace_dict, "summary_dict": scoped_summary_dict}
    overview_dict.update(build_financial_overview_dict(
        finance_workspace_dict, snapshot_obj, provider_obj, period_str=period_str, as_of_ts=as_of_ts))
    return overview_dict
