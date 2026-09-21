"""Bounded saved-log Activity evidence. Never opens a writer or a broker."""

from datetime import date, datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
import re
from zoneinfo import ZoneInfo


TARGET_LIMIT_INT = 32
EVENT_LIMIT_INT = 1000
TOTAL_BYTES_INT = 4 * 1024 * 1024
FILE_BYTES_INT = 512 * 1024
LINE_BYTES_INT = 32 * 1024
LINE_LIMIT_INT = 20000
BACKUP_COUNT_INT = 10
MARKET_TIMEZONE_OBJ = ZoneInfo("America/New_York")
QUIET_EVENT_SET = {"scheduler_sleeping", "scheduler_woke", "scheduler_due_now", "scheduler_phase_idle",
    "scheduler_tick_invoked", "scheduler.sleeping", "scheduler.decision", "scheduler.tick_result"}
CODE_PATTERN_OBJ = re.compile(r"[A-Za-z][A-Za-z0-9_.:-]{0,159}\Z")
IDENTITY_PATTERN_OBJ = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,199}\Z")
COUNT_FIELD_SET = {"broker_order_count_int", "broker_order_event_count_int", "broker_order_ack_count_int",
    "fill_count_int", "missing_ack_count_int", "terminal_unresolved_count_int", "order_count_int",
    "filled_order_count_int", "diff_count_int", "target_count_int", "eod_snapshot_count_int"}
NUMBER_FIELD_SET = {"ack_coverage_ratio_float", "target_share_float", "broker_share_float", "residual_share_float"}
CODE_FIELD_SET = {"status_str", "reason_code_str", "decision_plan_status_str", "vplan_status_str",
    "submit_ack_status_str", "reconciliation_status_str", "snapshot_stage_str", "next_phase_str",
    "execution_policy_str", "action_name_str", "initial_status_str", "latest_broker_order_status_str"}
TIME_FIELD_SET = {"signal_timestamp_str", "submission_timestamp_str", "target_execution_timestamp_str",
    "broker_snapshot_timestamp_str", "response_timestamp_str"}
POD_FIELD_TUPLE = ("pod_id_str", "pod_str")
ACCOUNT_FIELD_TUPLE = ("account_route_str", "account_id_str", "account_str")
MODE_FIELD_TUPLE = ("mode_str", "env_mode_str", "session_mode_str")


def _timestamp_ts(value_obj):
    if not isinstance(value_obj, str):
        return None
    try:
        timestamp_ts = datetime.fromisoformat(value_obj.replace("Z", "+00:00"))
        return timestamp_ts.astimezone(timezone.utc) if timestamp_ts.tzinfo is not None else None
    except (ValueError, OverflowError):
        return None


def _values_list(layer_list, field_tuple):
    return [layer_dict[field_str] for layer_dict in layer_list for field_str in field_tuple
        if layer_dict.get(field_str) not in (None, "")]


def _scope_dict(provider_obj):
    source_obj = provider_obj
    if not callable(getattr(source_obj, "get_target_list", None)):
        source_obj = provider_obj.app_obj()
    target_list = source_obj.get_target_list()
    if not isinstance(target_list, (list, tuple)):
        raise ValueError("Invalid Activity scope")
    identity_dict, account_set, owner_set = {}, set(), set()
    for target_obj in target_list:
        release_obj = target_obj.release_obj
        if release_obj.mode_str != "live" or release_obj.enabled_bool is not True:
            continue
        row_dict = {field_str: getattr(release_obj, field_str) for field_str in
            ("pod_id_str", "user_id_str", "account_route_str", "release_id_str")}
        if any(not isinstance(value_str, str) or not IDENTITY_PATTERN_OBJ.fullmatch(value_str) for value_str in row_dict.values()):
            raise ValueError("Invalid Activity scope")
        pod_id_str, account_str = row_dict["pod_id_str"], row_dict["account_route_str"]
        if pod_id_str in identity_dict or account_str in account_set or len(identity_dict) >= TARGET_LIMIT_INT:
            raise ValueError("Ambiguous or oversized Activity scope")
        identity_dict[pod_id_str] = row_dict
        account_set.add(account_str)
        owner_set.add(row_dict["user_id_str"])
        if len(owner_set) > 1:
            raise ValueError("Ambiguous Activity owner")
    path_str = getattr(provider_obj, "event_log_path_str", None) or getattr(source_obj, "event_log_path_str", None)
    if not isinstance(path_str, str) or not path_str:
        raise ValueError("Activity path unavailable")
    scope_key_str = hashlib.sha256(json.dumps({"source_str": str(Path(path_str).resolve()),
        "identity_list": sorted(identity_dict.values(), key=lambda row_dict: row_dict["pod_id_str"])},
        sort_keys=True).encode()).hexdigest()
    return identity_dict, Path(path_str), scope_key_str


def _event_dict(record_dict, identity_dict, *, source_str, as_of_ts, from_date_str):
    layer_list = [record_dict]
    for _depth_int in range(2):
        payload_obj = layer_list[-1].get("payload_dict")
        if not isinstance(payload_obj, dict):
            break
        layer_list.append(payload_obj)
    field_dict = {}
    for layer_dict in reversed(layer_list):
        field_dict.update({key_str: value_obj for key_str, value_obj in layer_dict.items() if value_obj is not None})
    journal_bool = source_str == "Operator journal"
    event_type_str = "operator_action_requested" if journal_bool else record_dict.get("event_name_str", record_dict.get("event_type_str"))
    if not isinstance(event_type_str, str) or not CODE_PATTERN_OBJ.fullmatch(event_type_str):
        raise ValueError("Invalid Activity code")
    level_rank_dict = {"DEBUG": 0, "INFO": 1, "WARNING": 2, "ERROR": 3, "CRITICAL": 4}
    level_list = [str(value_obj).upper() for value_obj in _values_list(layer_list, ("level_str", "severity_str"))]
    level_list = ["WARNING" if value_str == "WARN" else value_str for value_str in level_list]
    level_str = max((value_str for value_str in level_list if value_str in level_rank_dict),
        key=level_rank_dict.get, default="INFO")
    if event_type_str in QUIET_EVENT_SET and level_str not in {"WARNING", "ERROR", "CRITICAL"}:
        return None
    mode_list = _values_list(layer_list, MODE_FIELD_TUPLE)
    if any(mode_str != "live" for mode_str in mode_list):
        return None
    pod_list = _values_list(layer_list, POD_FIELD_TUPLE)
    if any(not isinstance(pod_str, str) for pod_str in pod_list) or len(set(pod_list)) > 1:
        return None
    related_list = []
    for layer_dict in layer_list:
        for field_str in ("related_pod_id_list", "pod_id_list", "pod_id_str_list"):
            value_list = layer_dict.get(field_str)
            if value_list is None:
                continue
            if not isinstance(value_list, list) or any(not isinstance(value_str, str) for value_str in value_list):
                raise ValueError("Invalid Activity Pod identity")
            related_list.extend(value_list)
    related_set = set(related_list)
    pod_id_str = pod_list[0] if pod_list else next(iter(related_set)) if len(related_set) == 1 else ""
    selected_set = related_set | ({pod_id_str} if pod_id_str else set())
    if selected_set and (not mode_list or not selected_set <= identity_dict.keys()):
        return None
    if pod_id_str and related_set and related_set != {pod_id_str}:
        return None
    for field_tuple, identity_field_str in ((ACCOUNT_FIELD_TUPLE, "account_route_str"), (("user_id_str",), "user_id_str")):
        value_list = _values_list(layer_list, field_tuple)
        if value_list and (not pod_id_str or any(value_obj != identity_dict[pod_id_str][identity_field_str] for value_obj in value_list)):
            return None
    release_list = _values_list(layer_list, ("release_id_str",))
    if release_list and (not selected_set or any(not isinstance(value_str, str) or
            not IDENTITY_PATTERN_OBJ.fullmatch(value_str) for value_str in release_list) or len(set(release_list)) != 1):
        return None
    # Occurrence time defines the ET activity day. Planned/as-of times do not.
    timestamp_obj = record_dict.get("event_timestamp_str") or record_dict.get("ts_utc") or record_dict.get("timestamp_str") or record_dict.get("created_timestamp_str")
    event_ts = _timestamp_ts(timestamp_obj)
    if event_ts is None or event_ts > as_of_ts:
        raise ValueError("Invalid Activity time")
    if record_dict.get("event_timestamp_str") and record_dict.get("ts_utc") and _timestamp_ts(record_dict["ts_utc"]) != event_ts:
        raise ValueError("Conflicting Activity time")
    if event_ts.astimezone(MARKET_TIMEZONE_OBJ).date().isoformat() < from_date_str:
        return None
    payload_dict = {}
    for field_str in CODE_FIELD_SET:
        value_obj = field_dict.get(field_str)
        if isinstance(value_obj, str) and CODE_PATTERN_OBJ.fullmatch(value_obj):
            payload_dict[field_str] = value_obj
    for field_str in COUNT_FIELD_SET:
        value_obj = field_dict.get(field_str)
        if type(value_obj) is int and 0 <= value_obj <= 1000000000:
            payload_dict[field_str] = value_obj
    for field_str in NUMBER_FIELD_SET:
        value_obj = field_dict.get(field_str)
        if type(value_obj) in (int, float) and math.isfinite(value_obj):
            payload_dict[field_str] = value_obj
    for field_str in TIME_FIELD_SET:
        value_ts = _timestamp_ts(field_dict.get(field_str))
        if value_ts is not None:
            payload_dict[field_str] = value_ts.isoformat()
    for field_str in ("release_id_str", "job_id_str"):
        value_obj = field_dict.get(field_str)
        if isinstance(value_obj, str) and IDENTITY_PATTERN_OBJ.fullmatch(value_obj):
            payload_dict[field_str] = value_obj
    date_str = field_dict.get("eod_market_date_str")
    if isinstance(date_str, str) and re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", date_str):
        try:
            payload_dict["eod_market_date_str"] = date.fromisoformat(date_str).isoformat()
        except ValueError:
            pass
    asset_str = field_dict.get("asset_str")
    if isinstance(asset_str, str) and re.fullmatch(r"[A-Z0-9][A-Z0-9.^_-]{0,19}", asset_str):
        payload_dict["asset_str"] = asset_str
    if related_set:
        payload_dict["related_pod_id_list"] = sorted(related_set)
    for field_str in ("decision_plan_id_int", "vplan_id_int"):
        value_list = _values_list(layer_list, (field_str,))
        if any(type(value_obj) is not int or value_obj <= 0 for value_obj in value_list) or len(set(value_list)) > 1:
            raise ValueError("Invalid Activity cycle identity")
        payload_dict[field_str] = value_list[0] if value_list else None
    delivery_str = ""
    if event_type_str.startswith(("notification_", "notification.", "alert_", "discord_")):
        explicit_str = field_dict.get("notification_delivery_str", field_dict.get("delivery_status_str"))
        if isinstance(explicit_str, str) and explicit_str in {"delivered", "failed", "queued"}:
            delivery_str = explicit_str
        elif type(field_dict.get("delivered_bool")) is bool:
            delivery_str = "delivered" if field_dict["delivered_bool"] else "failed"
    return {"timestamp_str": event_ts.isoformat(), "event_type_str": event_type_str, "level_str": level_str,
        "pod_id_str": pod_id_str, "mode_str": "live" if mode_list else "", "payload_dict": payload_dict,
        "decision_plan_id_int": payload_dict["decision_plan_id_int"], "vplan_id_int": payload_dict["vplan_id_int"],
        "source_str": source_str, "notification_delivery_str": delivery_str}


def load_activity_source_dict(provider_obj, *, as_of_ts, days_int):
    """Read at most 23 files, 4 MiB and 20,000 lines; return at most 1,000 rows.

    The main log contains material runner/scheduler events. Critical is its
    mirror; journal adds operator requests. No trace-directory walk or delivery
    inference from a current notification-state snapshot is performed.
    """
    if not isinstance(as_of_ts, datetime) or as_of_ts.tzinfo is None or type(days_int) is not int or not 1 <= days_int <= 365:
        raise ValueError("Activity needs an aware clock and a bounded day range")
    as_of_ts = as_of_ts.astimezone(timezone.utc)
    result_dict = {"event_list": [], "warning_list": [], "scope_key_str": "", "feed_available_bool": False}
    warning_set = set()
    try:
        identity_dict, event_path_obj, result_dict["scope_key_str"] = _scope_dict(provider_obj)
        if not identity_dict:
            result_dict["warning_list"] = ["No enabled LIVE Pods are available for Activity."]
            return result_dict
        root_path_obj = event_path_obj.parent.resolve()
    except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
        result_dict["warning_list"] = ["Activity scope could not be verified."]
        return result_dict
    critical_path_obj = root_path_obj / "live_critical_events.jsonl"
    source_list = [(event_path_obj, "Event log", True), (critical_path_obj, "Critical log", False),
        (root_path_obj / "operator_journal.jsonl", "Operator journal", False)]
    for index_int in range(1, BACKUP_COUNT_INT + 1):
        source_list.extend((path_obj.with_name(path_obj.name + f".{index_int}"), label_str, False)
            for path_obj, label_str in ((event_path_obj, "Event log"), (critical_path_obj, "Critical log")))
    from_date_str = (as_of_ts.astimezone(MARKET_TIMEZONE_OBJ).date() - timedelta(days=days_int - 1)).isoformat()
    bytes_left_int, lines_left_int, seen_set, main_read_bool = TOTAL_BYTES_INT, LINE_LIMIT_INT, set(), False
    for path_obj, source_str, required_bool in source_list:
        if bytes_left_int <= 0 or lines_left_int <= 0:
            warning_set.add("Activity history reached its scan limit.")
            break
        try:
            resolved_path_obj = path_obj.resolve()
            resolved_path_obj.relative_to(root_path_obj)
            with resolved_path_obj.open("rb") as file_obj:
                file_obj.seek(0, 2)
                size_int = file_obj.tell()
                read_int = min(size_int, FILE_BYTES_INT, bytes_left_int)
                offset_int = size_int - read_int
                file_obj.seek(offset_int)
                content_bytes = file_obj.read(read_int)
                bytes_left_int -= read_int
                if len(content_bytes) != read_int:
                    raise ValueError("Log changed during read")
            if required_bool:
                main_read_bool = True
            if offset_int:
                content_bytes = content_bytes.partition(b"\n")[2]
                warning_set.add("Activity history reached its scan limit.")
                if required_bool and not content_bytes:
                    warning_set.add("Some Activity records are incomplete or invalid.")
            if content_bytes and not content_bytes.endswith(b"\n"):
                warning_set.add("Some Activity records are incomplete or invalid.")
            line_list = content_bytes.split(b"\n")[:-1]
            for line_bytes in reversed(line_list):
                if lines_left_int <= 0:
                    warning_set.add("Activity history reached its scan limit.")
                    break
                lines_left_int -= 1
                if not line_bytes.strip():
                    continue
                try:
                    if len(line_bytes) > LINE_BYTES_INT:
                        raise ValueError("Oversized record")
                    record_dict = json.loads(line_bytes)
                    if not isinstance(record_dict, dict):
                        raise ValueError("Invalid record")
                    identity_str = hashlib.sha256(json.dumps(record_dict, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
                    identity_tuple = (source_str == "Operator journal", identity_str)
                    if identity_tuple in seen_set:
                        continue
                    seen_set.add(identity_tuple)
                    event_dict = _event_dict(record_dict, identity_dict, source_str=source_str, as_of_ts=as_of_ts, from_date_str=from_date_str)
                    if event_dict is not None:
                        result_dict["event_list"].append(event_dict)
                except (ValueError, TypeError, KeyError, UnicodeError, OverflowError, RecursionError):
                    warning_set.add("Some Activity records are incomplete or invalid.")
        except FileNotFoundError:
            if required_bool:
                warning_set.add("The Activity event log is unavailable.")
        except (OSError, ValueError, RuntimeError):
            warning_set.add("An Activity source could not be read safely.")
    result_dict["event_list"].sort(key=lambda event_dict: (event_dict["timestamp_str"], event_dict["event_type_str"]), reverse=True)
    if len(result_dict["event_list"]) > EVENT_LIMIT_INT:
        warning_set.add("Activity history reached its display limit.")
        result_dict["event_list"] = result_dict["event_list"][:EVENT_LIMIT_INT]
    result_dict["warning_list"] = sorted(warning_set)
    result_dict["feed_available_bool"] = main_read_bool and warning_set <= {
        "Activity history reached its scan limit.", "Activity history reached its display limit."}
    return result_dict
