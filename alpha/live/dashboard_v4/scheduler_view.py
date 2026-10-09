"""Scheduler evidence in existing operator warnings; no healthy-state widget."""

from alpha.live.dashboard_v3.filters import MARKET_TIMEZONE_OBJ
from alpha.live.ops_report import parse_timestamp_ts


def _time_str(timestamp_str, now_ts):
    timestamp_ts = parse_timestamp_ts(timestamp_str)
    if timestamp_ts is None:
        return "unknown"
    market_ts = timestamp_ts.astimezone(MARKET_TIMEZONE_OBJ)
    format_str = "%H:%M:%S" if market_ts.date() == now_ts.astimezone(MARKET_TIMEZONE_OBJ).date() else "%Y-%m-%d %H:%M:%S"
    return market_ts.strftime(format_str) + " ET"


def scheduler_issue_dict(status_dict, *, now_ts):
    state_str = status_dict.get("state_str", "unknown")
    if state_str not in {"late", "stopped", "error"}:
        return {}
    title_str = {"late": "Scheduler check overdue", "stopped": "Scheduler not responding",
                 "error": "Scheduler error"}[state_str]
    if state_str == "error":
        reason_str = status_dict.get("error_reason_str") or ""
        detail_str = ("Reason: " + reason_str.rstrip(".") + ".") if reason_str else "The scheduler reported an error."
        if status_dict.get("promised_wake_timestamp_str"):
            detail_str += " Retry due " + _time_str(status_dict["promised_wake_timestamp_str"], now_ts) + "."
    else:
        detail_str = "No sign of life since " + _time_str(status_dict.get("last_seen_timestamp_str"), now_ts) + "."
        if status_dict.get("promised_wake_timestamp_str"):
            detail_str += " Expected check " + _time_str(status_dict["promised_wake_timestamp_str"], now_ts) + "."
    return {"state_str": "late" if state_str == "late" else "fail", "title_str": title_str,
            "detail_str": detail_str, "timestamp_str": status_dict.get("last_seen_timestamp_str") or "",
            "console_bool": True}


HOLD_REASON_DICT = {
    "execution_exception_parked": "An execution problem is parked for your review.",
    "manual_review_required": "A plan waits for your review before it can be sent.",
    "mr_capsule_eod_snapshot_untrusted": "The saved end-of-day snapshot is not trusted.",
}


def scheduler_hold_dict(status_dict):
    """A live scheduler that holds a Pod will not retry by itself: the operator must act."""
    if status_dict.get("alive_bool") is not True or status_dict.get("state_str") != "holding":
        return {}
    reason_str = HOLD_REASON_DICT.get(status_dict.get("reason_code_str") or "", "The scheduler holds this Pod.")
    return {"state_str": "late", "title_str": "Waiting for you",
            "detail_str": reason_str + " It will not retry by itself.",
            "timestamp_str": status_dict.get("last_seen_timestamp_str") or "", "console_bool": True}


def scheduler_note_str(status_dict):
    """Only explain a problem from the last verified scheduler state."""
    if status_dict.get("alive_bool") is not True:
        return ""
    if status_dict.get("state_str") == "holding":
        return "The scheduler is alive. It waits for operator action."
    if status_dict.get("waiting_for_data_bool") is True or status_dict.get("reason_code_str") in {"snapshot_not_ready", "snapshot_not_ready_for_session", "waiting_for_data"}:
        return "The scheduler is alive. It waits for data."
    if status_dict.get("next_phase_str") == "post_execution_reconcile":
        return "The scheduler is alive. It is waiting to check execution." if status_dict.get("state_str") == "sleeping" else "The scheduler is alive. It is checking execution."
    return "The scheduler is alive."


def apply_scheduler_to_steps(step_list, status_dict):
    """Retain actual results. Unverified automation cannot promise future work.

    Call only for the current view; historical cycles keep their saved meaning.
    Unknown scheduler evidence does not change the Pod's business-state pill.
    """
    if status_dict.get("alive_bool") is True and status_dict.get("state_str") != "error":
        return
    for step_dict in step_list:
        if step_dict["state_str"] == "Planned":
            step_dict.update(state_str="Unknown", fact_str="Schedule only · Scheduler not verified")
