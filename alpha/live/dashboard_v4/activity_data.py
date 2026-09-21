"""Bounded saved-log Activity evidence. Never opens a writer or a broker."""

from datetime import date, datetime, timedelta, timezone
from collections import OrderedDict
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re
from threading import Lock
import time
from zoneinfo import ZoneInfo


TARGET_LIMIT_INT = 32
EVENT_LIMIT_INT = 1000
TOTAL_BYTES_INT = 128 * 1024 * 1024
CHUNK_BYTES_INT = 256 * 1024
LINE_BYTES_INT = 32 * 1024
LINE_LIMIT_INT = 200000
MATERIAL_LIMIT_INT = 4000
SCAN_SECONDS_FLOAT = 5.0
BACKUP_COUNT_INT = 10
CACHE_FILE_LIMIT_INT = 32
CACHE_EVENT_LIMIT_INT = 4000
FILE_CACHE_DICT = OrderedDict()
CACHE_LOCK_OBJ = Lock()
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
    level_list = [{"WARN": "WARNING", "FATAL": "CRITICAL"}.get(value_str, value_str) for value_str in level_list]
    level_str = max((value_str for value_str in level_list if value_str in level_rank_dict),
        key=level_rank_dict.get, default="INFO")
    status_list = _values_list(layer_list, ("status_str", "initial_status_str", "reconciliation_status_str",
        "vplan_status_str", "decision_plan_status_str", "submit_ack_status_str"))
    issue_bool = any(str(value_obj).lower() in {"failed", "blocked", "error", "missing_critical", "expired", "late"}
        for value_obj in status_list) or any(type(value_obj) is int and value_obj > 0
            for value_obj in _values_list(layer_list, ("missing_ack_count_int",)))
    quiet_bool = level_str not in {"WARNING", "ERROR", "CRITICAL"} and not issue_bool
    if event_type_str in QUIET_EVENT_SET and quiet_bool:
        return None
    if event_type_str == "norgate_snapshot_sync_skipped" and quiet_bool and (
        field_dict.get("reason_code_str"), field_dict.get("status_str")) in {
            ("direct_norgate_mode", "direct"), ("no_enabled_releases", "ready"), ("local_snapshot_ready", "ready")}:
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
    if not mode_list and selected_set and event_type_str.startswith("norgate_snapshot_sync_"):
        sync_pod_list = field_dict.get("pod_id_list")
        sync_release_list = field_dict.get("release_id_list")
        if (isinstance(sync_pod_list, list) and isinstance(sync_release_list, list)
            and len(sync_pod_list) == len(sync_release_list) == len(selected_set)
            and set(sync_pod_list) == selected_set and selected_set <= identity_dict.keys()
            and all(layer_dict.get("pod_id_list", sync_pod_list) == sync_pod_list
                and layer_dict.get("release_id_list", sync_release_list) == sync_release_list for layer_dict in layer_list)
            and all(release_str == identity_dict[pod_str]["release_id_str"]
                for pod_str, release_str in zip(sync_pod_list, sync_release_list))
            and all(pod_id_str and release_str == identity_dict[pod_id_str]["release_id_str"]
                for release_str in _values_list(layer_list, ("release_id_str",)))):
            mode_list = ["live"]
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
    for field_str in ("release_id_str", "job_id_str", "ticket_id_str"):
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
    for field_str, allowed_set in (("side_str", {"BUY", "SELL"}),
            ("broker_order_type_str", {"MKT", "LMT", "MOO", "MOC", "LOO", "LOC", "STP", "STP LMT"})):
        if isinstance(field_dict.get(field_str), str) and field_dict[field_str] in allowed_set:
            payload_dict[field_str] = field_dict[field_str]
    quantity_int = field_dict.get("quantity_int")
    if type(quantity_int) is int and 0 < quantity_int <= 1000000000:
        payload_dict["quantity_int"] = quantity_int
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


def _stat_tuple(path_obj):
    stat_obj = path_obj.stat()
    return stat_obj.st_dev, stat_obj.st_ino, stat_obj.st_size, stat_obj.st_mtime_ns, stat_obj.st_ctime_ns


def _cache_file_dict(key_tuple, as_of_ts):
    with CACHE_LOCK_OBJ:
        cached_dict = FILE_CACHE_DICT.get(key_tuple)
        if cached_dict is None or cached_dict["as_of_ts"] > as_of_ts:
            return None
        FILE_CACHE_DICT.move_to_end(key_tuple)
        return deepcopy(cached_dict)


def _save_cache(key_tuple, result_dict):
    if result_dict["reason_set"]:
        return
    with CACHE_LOCK_OBJ:
        # Appends/replacements invalidate the previous entry for this path,
        # scope and selected boundary. There is no age-only cache reuse.
        for old_key_tuple in list(FILE_CACHE_DICT):
            if old_key_tuple[:-1] == key_tuple[:-1]:
                del FILE_CACHE_DICT[old_key_tuple]
        FILE_CACHE_DICT[key_tuple] = deepcopy(result_dict)
        while len(FILE_CACHE_DICT) > CACHE_FILE_LIMIT_INT or sum(len(value_dict["event_list"]) for value_dict in FILE_CACHE_DICT.values()) > CACHE_EVENT_LIMIT_INT:
            FILE_CACHE_DICT.popitem(last=False)


def _limit_str(budget_dict):
    if budget_dict["bytes_int"] >= TOTAL_BYTES_INT:
        return "byte_limit"
    if budget_dict["lines_int"] >= LINE_LIMIT_INT:
        return "line_limit"
    if budget_dict["material_int"] >= MATERIAL_LIMIT_INT:
        return "material_limit"
    if time.monotonic() >= budget_dict["deadline_float"]:
        return "time_limit"
    return ""


def _read_file_dict(path_obj, identity_dict, *, source_str, as_of_ts, from_ts, budget_dict, seen_set):
    """Reverse complete lines in fixed chunks, keeping at most one short prefix."""
    result_dict = {"event_list": [], "reason_set": set(), "oldest_ts": None, "newest_ts": None,
        "boundary_bool": False, "complete_records_int": 0, "as_of_ts": as_of_ts}
    local_seen_set, previous_ts, pending_bytes, discard_bool = set(), None, b"", False
    with path_obj.open("rb") as file_obj:
        file_obj.seek(0, 2)
        offset_int = file_obj.tell()
        first_bool = True
        while offset_int > 0:
            limit_str = _limit_str(budget_dict)
            if limit_str:
                result_dict["reason_set"].add(limit_str)
                break
            read_int = min(offset_int, CHUNK_BYTES_INT, TOTAL_BYTES_INT - budget_dict["bytes_int"])
            offset_int -= read_int
            file_obj.seek(offset_int)
            chunk_bytes = file_obj.read(read_int)
            budget_dict["bytes_int"] += len(chunk_bytes)
            if len(chunk_bytes) != read_int:
                raise ValueError("Log changed during read")
            if discard_bool:
                prefix_bytes, separator_bytes, _suffix_bytes = chunk_bytes.rpartition(b"\n")
                if not separator_bytes:
                    continue
                chunk_bytes, discard_bool = prefix_bytes, False
            part_list = (chunk_bytes + pending_bytes).split(b"\n")
            pending_bytes, line_list = part_list[0], part_list[1:]
            if first_bool:
                if line_list and line_list[-1]:
                    result_dict["reason_set"].add("invalid_records")
                if not line_list and pending_bytes:
                    result_dict["reason_set"].add("invalid_records")
                    discard_bool, pending_bytes = True, b""
                line_list = line_list[:-1]
                first_bool = False
            if offset_int == 0:
                line_list.insert(0, pending_bytes)
                pending_bytes = b""
            if len(pending_bytes) > LINE_BYTES_INT:
                pending_bytes, discard_bool = b"", True
                result_dict["reason_set"].add("invalid_records")
            chunk_time_list = []
            for line_bytes in reversed(line_list):
                # The bytes of this chunk were already read. Only CPU/material
                # limits can interrupt its parsing; a byte limit stops the next read.
                limit_str = _limit_str({**budget_dict, "bytes_int": 0})
                if limit_str:
                    result_dict["reason_set"].add(limit_str)
                    break
                budget_dict["lines_int"] += 1
                if not line_bytes.strip():
                    continue
                try:
                    if len(line_bytes) > LINE_BYTES_INT:
                        raise ValueError("Oversized record")
                    record_dict = json.loads(line_bytes)
                    if not isinstance(record_dict, dict):
                        raise ValueError("Invalid record")
                    event_ts = _timestamp_ts(record_dict.get("event_timestamp_str") or record_dict.get("ts_utc")
                        or record_dict.get("timestamp_str") or record_dict.get("created_timestamp_str"))
                    if event_ts is None or event_ts > as_of_ts or (record_dict.get("event_timestamp_str")
                        and record_dict.get("ts_utc") and _timestamp_ts(record_dict["ts_utc"]) != event_ts):
                        raise ValueError("Invalid occurrence time")
                    result_dict["complete_records_int"] += 1
                    chunk_time_list.append(event_ts)
                    if previous_ts is not None and event_ts > previous_ts:
                        result_dict["reason_set"].add("unordered_records")
                    previous_ts = event_ts
                    result_dict["oldest_ts"] = min(result_dict["oldest_ts"] or event_ts, event_ts)
                    result_dict["newest_ts"] = max(result_dict["newest_ts"] or event_ts, event_ts)
                    if event_ts < from_ts:
                        continue
                    event_dict = _event_dict(record_dict, identity_dict, source_str=source_str,
                        as_of_ts=as_of_ts, from_date_str=from_ts.astimezone(MARKET_TIMEZONE_OBJ).date().isoformat())
                    if event_dict is None:
                        continue
                    identity_str = str(source_str == "Operator journal") + hashlib.sha256(
                        json.dumps(record_dict, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
                    if identity_str in local_seen_set:
                        continue
                    local_seen_set.add(identity_str)
                    result_dict["event_list"].append((identity_str, event_dict))
                    if identity_str not in seen_set:
                        budget_dict["material_int"] += 1
                except (ValueError, TypeError, KeyError, UnicodeError, OverflowError, RecursionError):
                    result_dict["reason_set"].add("invalid_records")
            # *** CRITICAL *** use the logger's append-time occurrence, never
            # a planned/as-of timestamp. A whole old chunk crosses the ET
            # boundary; one delayed/backdated line cannot stop the scan.
            if chunk_time_list and max(chunk_time_list) < from_ts and not result_dict["reason_set"]:
                result_dict["boundary_bool"] = True
                break
            if result_dict["reason_set"] & {"line_limit", "material_limit", "time_limit"}:
                break
        if offset_int == 0 and result_dict["oldest_ts"] is not None and result_dict["oldest_ts"] < from_ts and not result_dict["reason_set"]:
            result_dict["boundary_bool"] = True
    return result_dict


def load_activity_source_dict(provider_obj, *, as_of_ts, days_int):
    """Read saved history to its ET boundary, subject to explicit resource caps.

    At most 23 fixed files, 128 MiB, 200,000 lines, 4,000 material records and
    a cooperative 5-second deadline. Quiet polling uses no material-row budget.
    Unchanged files can reuse a bounded memory cache; nothing writes to disk.
    """
    if not isinstance(as_of_ts, datetime) or as_of_ts.tzinfo is None or type(days_int) is not int or not 1 <= days_int <= 365:
        raise ValueError("Activity needs an aware clock and a bounded day range")
    as_of_ts = as_of_ts.astimezone(timezone.utc)
    from_ts = as_of_ts.astimezone(MARKET_TIMEZONE_OBJ).replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=days_int - 1)
    coverage_dict = {"requested_from_timestamp_str": from_ts.isoformat(), "scanned_from_timestamp_str": None,
        "complete_bool": False, "reason_list": [], "read_bytes_int": 0, "parsed_lines_int": 0, "cache_hit_count_int": 0}
    result_dict = {"event_list": [], "warning_list": [], "scope_key_str": "", "feed_available_bool": False, "coverage_dict": coverage_dict}
    warning_set, reason_set = set(), set()
    try:
        identity_dict, event_path_obj, result_dict["scope_key_str"] = _scope_dict(provider_obj)
        if not identity_dict:
            result_dict["warning_list"] = ["No enabled LIVE Pods are available for Activity."]
            coverage_dict["reason_list"] = ["scope_unavailable"]
            return result_dict
        root_path_obj = event_path_obj.parent.resolve()
    except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
        result_dict["warning_list"] = ["Activity scope could not be verified."]
        coverage_dict["reason_list"] = ["scope_unavailable"]
        return result_dict
    critical_path_obj = root_path_obj / "live_critical_events.jsonl"
    source_list = [(event_path_obj, "Event log", 0), (critical_path_obj, "Critical log", 0),
        (root_path_obj / "operator_journal.jsonl", "Operator journal", 0)]
    for index_int in range(1, BACKUP_COUNT_INT + 1):
        source_list.extend((path_obj.with_name(path_obj.name + f".{index_int}"), label_str, index_int)
            for path_obj, label_str in ((event_path_obj, "Event log"), (critical_path_obj, "Critical log")))
    budget_dict = {"bytes_int": 0, "lines_int": 0, "material_int": 0, "deadline_float": time.monotonic() + SCAN_SECONDS_FLOAT}
    seen_set, finished_set, missing_set = set(), set(), set()
    main_read_bool, main_oldest_ts, main_contiguous_bool = False, None, True
    for path_obj, source_str, rotation_int in source_list:
        if source_str in finished_set:
            continue
        limit_str = _limit_str(budget_dict)
        if limit_str:
            reason_set.add(limit_str)
            break
        required_bool = source_str == "Event log" and rotation_int == 0
        try:
            resolved_path_obj = path_obj.resolve()
            resolved_path_obj.relative_to(root_path_obj)
            stat_tuple = _stat_tuple(resolved_path_obj)
            key_tuple = (result_dict["scope_key_str"], from_ts.isoformat(), source_str, str(resolved_path_obj), stat_tuple)
            file_dict = _cache_file_dict(key_tuple, as_of_ts)
            if file_dict is not None:
                coverage_dict["cache_hit_count_int"] += 1
                budget_dict["material_int"] += sum(identity_str not in seen_set for identity_str, _event_dict in file_dict["event_list"])
                if budget_dict["material_int"] > MATERIAL_LIMIT_INT:
                    reason_set.add("material_limit")
                    break
            else:
                file_dict = _read_file_dict(resolved_path_obj, identity_dict, source_str=source_str,
                    as_of_ts=as_of_ts, from_ts=from_ts, budget_dict=budget_dict, seen_set=seen_set)
                if _stat_tuple(resolved_path_obj) == stat_tuple:
                    _save_cache(key_tuple, file_dict)
                else:
                    file_dict["reason_set"].add("source_changed")
            if required_bool:
                main_read_bool = stat_tuple[2] == 0 or file_dict["complete_records_int"] > 0
            reason_set.update(file_dict["reason_set"])
            if rotation_int and source_str in missing_set:
                reason_set.add("rotation_gap")
            if source_str == "Event log" and main_contiguous_bool and file_dict["oldest_ts"] is not None:
                main_oldest_ts = min(main_oldest_ts or file_dict["oldest_ts"], file_dict["oldest_ts"])
            if file_dict["boundary_bool"]:
                finished_set.add(source_str)
            for identity_str, event_dict in file_dict["event_list"]:
                if identity_str not in seen_set:
                    seen_set.add(identity_str)
                    result_dict["event_list"].append(event_dict)
        except FileNotFoundError:
            missing_set.add(source_str)
            if source_str == "Event log":
                main_contiguous_bool = False
            if required_bool:
                warning_set.add("The Activity event log is unavailable.")
                reason_set.add("source_unavailable")
        except (OSError, ValueError, RuntimeError):
            warning_set.add("An Activity source could not be read safely.")
            reason_set.add("source_unavailable")
    if "Event log" not in finished_set and not reason_set:
        reason_set.add("retained_history")
    if reason_set & {"byte_limit", "line_limit", "material_limit", "time_limit"}:
        warning_set.add("Activity history reached its scan limit.")
    if reason_set & {"invalid_records", "unordered_records", "source_changed", "rotation_gap"}:
        warning_set.add("Some Activity records are incomplete or invalid.")
    result_dict["event_list"].sort(key=lambda event_dict: (event_dict["timestamp_str"], event_dict["event_type_str"]), reverse=True)
    if len(result_dict["event_list"]) > EVENT_LIMIT_INT:
        warning_set.add("Activity history reached its display limit.")
        reason_set.add("display_limit")
        result_dict["event_list"] = result_dict["event_list"][:EVENT_LIMIT_INT]
    result_dict["warning_list"] = sorted(warning_set)
    result_dict["feed_available_bool"] = main_read_bool and not reason_set & {
        "invalid_records", "unordered_records", "source_changed", "source_unavailable", "rotation_gap"}
    coverage_dict.update(scanned_from_timestamp_str=main_oldest_ts.isoformat() if main_oldest_ts else None,
        complete_bool="Event log" in finished_set and not reason_set, reason_list=sorted(reason_set),
        read_bytes_int=budget_dict["bytes_int"], parsed_lines_int=budget_dict["lines_int"])
    return result_dict
