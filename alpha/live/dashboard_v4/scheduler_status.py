"""Bounded, read-only scheduler evidence; no process or broker probes."""

import json
import math
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path


TAIL_BYTES_INT = 64 * 1024
EVENT_BACKUPS_INT = 10
TRACE_BACKUPS_INT = 5
LATE_AFTER_SECONDS_INT = 60
STOPPED_AFTER_SECONDS_INT = 300
UNPROMISED_FRESH_SECONDS_INT = 120
PHASE_TUPLE = (
    "build_decision_plan", "build_vplan", "eod_snapshot", "expire_stale",
    "idle_probe", "manual_review_pending", "post_execution_reconcile", "submit_vplan",
)
EVENT_NAME_SET = {
    "scheduler_sleeping", "scheduler_due_now", "scheduler.sleeping",
    "scheduler.decision", "scheduler.tick_result", "scheduler.error_retry",
}


def _timestamp_ts(value_obj):
    try:
        timestamp_ts = datetime.fromisoformat(str(value_obj))
        if timestamp_ts.tzinfo is None:
            return None
        return timestamp_ts.astimezone(timezone.utc)
    except (ValueError, TypeError, OverflowError):
        return None


def _unknown_dict(as_of_ts, detail_str="Scheduler evidence unavailable."):
    return {
        "state_str": "unknown", "alive_bool": None,
        "last_seen_timestamp_str": None, "promised_wake_timestamp_str": None,
        "next_phase_str": "", "reason_code_str": "", "detail_str": detail_str,
        "checked_timestamp_str": as_of_ts.isoformat(),
    }


def _contained_path_obj(path_obj, root_path_obj):
    resolved_path_obj = path_obj.resolve()
    resolved_path_obj.relative_to(root_path_obj)
    return resolved_path_obj


def _tail_record_list(path_obj, root_path_obj):
    """Each file costs at most TAIL_BYTES_INT, including a partial first line."""
    resolved_path_obj = _contained_path_obj(path_obj, root_path_obj)
    try:
        with resolved_path_obj.open("rb") as source_file_obj:
            source_file_obj.seek(0, 2)
            size_int = source_file_obj.tell()
            offset_int = max(0, size_int - TAIL_BYTES_INT)
            source_file_obj.seek(offset_int)
            content_bytes = source_file_obj.read(min(size_int, TAIL_BYTES_INT))
            if len(content_bytes) != size_int - offset_int:
                raise ValueError("Log changed while reading")
    except FileNotFoundError:
        return []
    if offset_int:
        content_bytes = content_bytes.partition(b"\n")[2]
    # A writer may be in the middle of appending the final JSON record.
    complete_line_list = content_bytes.split(b"\n")[:-1]
    record_list = []
    for line_bytes in complete_line_list:
        if not line_bytes.strip():
            continue
        record_dict = json.loads(line_bytes.decode("utf-8"))
        if not isinstance(record_dict, dict):
            raise ValueError("Invalid event record")
        record_list.append(record_dict)
    return record_list


def _event_dict(record_dict, pod_id_str):
    event_name_str = record_dict.get("event_name_str")
    if event_name_str not in EVENT_NAME_SET:
        return None
    # log_pod_trace_event's payload is wrapped twice by the structured logger.
    layer_list = [record_dict]
    for _depth_int in range(2):
        payload_obj = layer_list[-1].get("payload_dict")
        if not isinstance(payload_obj, dict):
            break
        layer_list.append(payload_obj)
    field_dict = {}
    for layer_dict in layer_list:
        field_dict.update(layer_dict)
    decision_dict = field_dict.get("scheduler_decision_dict", {})
    if not isinstance(decision_dict, dict):
        raise ValueError("Invalid scheduler decision")
    mode_list = [layer_dict[key_str] for layer_dict in [*layer_list, decision_dict]
        for key_str in ("mode_str", "env_mode_str") if layer_dict.get(key_str) is not None]
    explicit_pod_list = [layer_dict["pod_id_str"] for layer_dict in layer_list
        if layer_dict.get("pod_id_str") is not None]
    related_list_list = [layer_dict["related_pod_id_list"] for layer_dict in [*layer_list, decision_dict]
        if "related_pod_id_list" in layer_dict]
    if any(not isinstance(related_list, list) for related_list in related_list_list):
        raise ValueError("Invalid scheduler Pod identity")
    selected_pod_bool = pod_id_str in explicit_pod_list or any(pod_id_str in related_list for related_list in related_list_list)
    if not selected_pod_bool:
        return None
    if "live" in mode_list and any(mode_str != "live" for mode_str in mode_list):
        raise ValueError("Conflicting scheduler mode")
    if not mode_list or any(mode_str != "live" for mode_str in mode_list):
        return None
    if (any(value_str != pod_id_str for value_str in explicit_pod_list)
            or any(related_list and pod_id_str not in related_list for related_list in related_list_list)):
        raise ValueError("Conflicting scheduler Pod identity")
    event_ts = _timestamp_ts(record_dict.get("event_timestamp_str", record_dict.get("ts_utc")))
    if event_ts is None:
        raise ValueError("Invalid scheduler timestamp")
    if record_dict.get("ts_utc") is not None and _timestamp_ts(record_dict["ts_utc"]) != event_ts:
        raise ValueError("Conflicting scheduler timestamps")
    phase_str = field_dict.get("next_phase_str", decision_dict.get("next_phase_str", ""))
    reason_str = field_dict.get("reason_code_str", decision_dict.get("reason_code_str", ""))
    if phase_str not in PHASE_TUPLE:
        raise ValueError("Invalid scheduler state")
    # Reason codes are diagnostic text, not identity or liveness evidence.
    # Keep the validated event (especially a newer error) if its reason cannot
    # be shown safely; never skip it and revive an older healthy observation.
    if not isinstance(reason_str, str) or not re.fullmatch(r"[a-z0-9_]{0,100}", reason_str):
        reason_str = ""
    return {**decision_dict, **field_dict, "event_name_str": event_name_str,
        "event_ts": event_ts, "next_phase_str": phase_str, "reason_code_str": reason_str}


def _status_dict(event_dict, as_of_ts, event_list):
    result_dict = _unknown_dict(as_of_ts)
    event_ts = event_dict["event_ts"]
    if event_ts > as_of_ts:
        return _unknown_dict(as_of_ts, "Scheduler evidence has a future timestamp.")
    result_dict.update(last_seen_timestamp_str=event_ts.isoformat(),
        next_phase_str=event_dict["next_phase_str"], reason_code_str=event_dict["reason_code_str"])
    event_name_str = event_dict["event_name_str"]
    error_bool = event_name_str == "scheduler.error_retry"
    sleeping_bool = event_name_str in {"scheduler_sleeping", "scheduler.sleeping"}
    if sleeping_bool or error_bool:
        duration_obj = event_dict.get("error_retry_seconds_int" if error_bool else "sleep_seconds_float")
        if isinstance(duration_obj, bool) or not isinstance(duration_obj, (int, float)):
            return _unknown_dict(as_of_ts, "Scheduler wake time is unavailable.")
        duration_float = float(duration_obj)
        if not math.isfinite(duration_float) or duration_float < 0:
            return _unknown_dict(as_of_ts, "Scheduler wake time is unavailable.")
        try:
            wake_ts = event_ts + timedelta(seconds=duration_float)
        except OverflowError:
            return _unknown_dict(as_of_ts, "Scheduler wake time is unavailable.")
        result_dict["promised_wake_timestamp_str"] = wake_ts.isoformat()
        overdue_seconds_float = (as_of_ts - wake_ts).total_seconds()
        # These are display heuristics, not proof that the process has exited.
        if overdue_seconds_float > STOPPED_AFTER_SECONDS_INT:
            result_dict.update(state_str="stopped", alive_bool=False,
                detail_str="No new scheduler evidence after its expected wake. Check the service and logs.")
            return result_dict
        if overdue_seconds_float > LATE_AFTER_SECONDS_INT:
            result_dict.update(state_str="late", alive_bool=None,
                detail_str="No new scheduler evidence after its expected wake.")
            return result_dict
    elif (as_of_ts - event_ts).total_seconds() > UNPROMISED_FRESH_SECONDS_INT:
        result_dict["detail_str"] = "Scheduler evidence is old; no wake time was recorded."
        return result_dict
    else:
        # run_once emits due/decision/tick_result too. These prove a continuous
        # scheduler only when they follow a still-current serve sleep promise.
        prior_sleep_bool = False
        for prior_dict in event_list:
            if prior_dict["event_name_str"] not in {"scheduler_sleeping", "scheduler.sleeping"} or prior_dict["event_ts"] > event_ts:
                continue
            duration_obj = prior_dict.get("sleep_seconds_float")
            if isinstance(duration_obj, bool) or not isinstance(duration_obj, (int, float)) or not math.isfinite(duration_obj) or duration_obj < 0:
                continue
            try:
                prior_wake_ts = prior_dict["event_ts"] + timedelta(seconds=float(duration_obj))
            except OverflowError:
                continue
            if event_ts <= prior_wake_ts + timedelta(seconds=LATE_AFTER_SECONDS_INT):
                prior_sleep_bool = True
                break
        if not prior_sleep_bool:
            result_dict["detail_str"] = "Recent scheduler activity; continuous service is not verified."
            return result_dict
    result_dict["alive_bool"] = True
    if error_bool:
        # Exception strings may include credentials or account data. Never send
        # arbitrary log content to the browser, even if the writer redacted it.
        result_dict.update(state_str="error", detail_str="Scheduler reported an error. Check its log.")
    elif event_dict["next_phase_str"] == "manual_review_pending":
        result_dict.update(state_str="holding",
            detail_str="The scheduler is alive. It holds this pod and will not retry by itself.")
    elif event_dict.get("norgate_snapshot_sync_active_wait_bool") is True or event_dict.get("norgate_snapshot_sync_blocks_decision_plan_bool") is True:
        result_dict.update(state_str="sleeping" if sleeping_bool else "running", waiting_for_data_bool=True,
            detail_str="The scheduler is alive. It waits for data.")
    elif sleeping_bool:
        result_dict.update(state_str="sleeping", detail_str="The scheduler is alive. It checks again at its expected wake.")
    else:
        result_dict.update(state_str="running", detail_str="The scheduler has recent activity.")
    return result_dict


def load_scheduler_status_dict(event_log_path_str, pod_id_str, *, as_of_ts, trace_root_path_str=None):
    """Only configured LIVE sources; max 59 files / 3.7 MiB per Pod, no walks.

    Trace paths match scheduler_service's fixed per-phase run ids. Missing logs
    yield Unknown. A custom trace root must stay inside the event log directory.
    Busy shared logs can push a sleeping Pod outside every bounded tail. Its
    per-Pod trace must then retain the sleep promise; without it return Unknown,
    never infer that the scheduler is stopped or scan unbounded history.
    """
    checked_ts = _timestamp_ts(as_of_ts)
    if checked_ts is None:
        raise ValueError("An aware as_of_ts is required")
    if not isinstance(pod_id_str, str) or not pod_id_str or len(pod_id_str) > 200 or any(
        not character_str.isalnum() and character_str not in "._-" for character_str in pod_id_str
    ) or pod_id_str in {".", ".."}:
        return _unknown_dict(checked_ts)
    if not isinstance(event_log_path_str, str) or not event_log_path_str:
        return _unknown_dict(checked_ts)
    try:
        event_path_obj = Path(event_log_path_str)
        root_path_obj = event_path_obj.parent.resolve()
        trace_root_obj = _contained_path_obj(Path(trace_root_path_str) if trace_root_path_str else root_path_obj / "pods", root_path_obj)
        source_path_list = [event_path_obj.with_name(event_path_obj.name + (f".{index_int}" if index_int else ""))
            for index_int in range(EVENT_BACKUPS_INT + 1)]
        sanitized_pod_str = pod_id_str.strip("._") or "unknown"
        for phase_str in PHASE_TUPLE:
            trace_path_obj = trace_root_obj / sanitized_pod_str / f"live_{pod_id_str}_scheduler_{phase_str}" / "trace_events.jsonl"
            source_path_list.extend(trace_path_obj.with_name(trace_path_obj.name + (f".{index_int}" if index_int else ""))
                for index_int in range(TRACE_BACKUPS_INT + 1))
        event_list = []
        for source_path_obj in source_path_list:
            for record_dict in _tail_record_list(source_path_obj, root_path_obj):
                event_dict = _event_dict(record_dict, pod_id_str)
                if event_dict is not None:
                    event_list.append(event_dict)
        if not event_list:
            return _unknown_dict(checked_ts, "No scheduler evidence found for this pod.")
        newest_event_dict = max(event_list, key=lambda event_dict: (
            event_dict["event_ts"], event_dict["event_name_str"] == "scheduler.error_retry"))
        return _status_dict(newest_event_dict, checked_ts, event_list)
    except (OSError, ValueError, TypeError, RuntimeError, OverflowError, UnicodeError):
        return _unknown_dict(checked_ts)
