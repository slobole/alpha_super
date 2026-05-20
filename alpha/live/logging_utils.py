from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_LOG_PATH_STR = str(Path(__file__).resolve().parent / "logs" / "live_events.jsonl")
DEFAULT_OPERATOR_LOG_PATH_STR = str(Path(__file__).resolve().parent / "logs" / "live_operator.log")
DEFAULT_CRITICAL_LOG_PATH_STR = str(
    Path(__file__).resolve().parent / "logs" / "live_critical_events.jsonl"
)
DEFAULT_POD_TRACE_LOG_ROOT_PATH_STR = str(Path(__file__).resolve().parent / "logs" / "pods")
DEFAULT_LOG_MAX_BYTES_INT = 50 * 1024 * 1024
DEFAULT_LOG_BACKUP_COUNT_INT = 10
DEFAULT_TRACE_LOG_MAX_BYTES_INT = 10 * 1024 * 1024
DEFAULT_TRACE_LOG_BACKUP_COUNT_INT = 5


def _rotated_log_path_obj(log_path_obj: Path, index_int: int) -> Path:
    return log_path_obj.with_name(f"{log_path_obj.name}.{index_int}")


def _rotate_log_file_if_needed(
    log_path_obj: Path,
    incoming_size_int: int,
    max_bytes_int: int | None,
    backup_count_int: int,
) -> None:
    if max_bytes_int is None or int(max_bytes_int) <= 0:
        return
    if not log_path_obj.exists():
        return
    if log_path_obj.stat().st_size + int(incoming_size_int) <= int(max_bytes_int):
        return

    normalized_backup_count_int = max(int(backup_count_int), 0)
    try:
        if normalized_backup_count_int == 0:
            log_path_obj.unlink()
            return

        oldest_log_path_obj = _rotated_log_path_obj(log_path_obj, normalized_backup_count_int)
        if oldest_log_path_obj.exists():
            oldest_log_path_obj.unlink()

        for index_int in range(normalized_backup_count_int - 1, 0, -1):
            source_log_path_obj = _rotated_log_path_obj(log_path_obj, index_int)
            target_log_path_obj = _rotated_log_path_obj(log_path_obj, index_int + 1)
            if source_log_path_obj.exists():
                source_log_path_obj.replace(target_log_path_obj)

        log_path_obj.replace(_rotated_log_path_obj(log_path_obj, 1))
    except PermissionError:
        return


def _write_line_to_rolling_log(
    log_path_obj: Path,
    line_str: str,
    *,
    max_bytes_int: int | None,
    backup_count_int: int,
) -> None:
    try:
        log_path_obj.parent.mkdir(parents=True, exist_ok=True)
        line_with_newline_str = line_str + "\n"
        incoming_size_int = len(line_with_newline_str.encode("utf-8"))
        _rotate_log_file_if_needed(
            log_path_obj=log_path_obj,
            incoming_size_int=incoming_size_int,
            max_bytes_int=max_bytes_int,
            backup_count_int=backup_count_int,
        )
        with log_path_obj.open("a", encoding="utf-8") as log_file_obj:
            log_file_obj.write(line_with_newline_str)
    except OSError:
        return


def _resolve_level_str(event_payload_dict: dict[str, Any]) -> str:
    level_obj = event_payload_dict.get("level_str")
    if level_obj is not None and str(level_obj).strip() != "":
        return str(level_obj).upper()
    severity_str = str(event_payload_dict.get("severity_str", "")).lower()
    if severity_str == "critical":
        return "CRITICAL"
    if severity_str in {"warning", "warn"}:
        return "WARNING"
    if severity_str == "error":
        return "ERROR"
    return "INFO"


def _first_non_empty_value_obj(event_payload_dict: dict[str, Any], key_tuple: tuple[str, ...]) -> Any:
    for key_str in key_tuple:
        value_obj = event_payload_dict.get(key_str)
        if value_obj is not None and str(value_obj) != "":
            return value_obj
    return None


def _derive_run_id_str(event_payload_dict: dict[str, Any], timestamp_str: str) -> str:
    run_id_obj = _first_non_empty_value_obj(
        event_payload_dict,
        ("run_id_str", "tick_id_str", "cycle_id_str"),
    )
    if run_id_obj is not None:
        return str(run_id_obj)
    as_of_timestamp_obj = _first_non_empty_value_obj(
        event_payload_dict,
        ("as_of_timestamp_str", "timestamp_str", "created_timestamp_str"),
    )
    base_timestamp_str = str(as_of_timestamp_obj or timestamp_str)
    return (
        base_timestamp_str.replace("-", "")
        .replace(":", "")
        .replace("+", "")
        .replace(".", "")
        .replace("T", "T")
    )


def build_structured_event_record_dict(
    event_name_str: str,
    event_payload_dict: dict[str, Any],
    timestamp_obj: datetime | None = None,
) -> dict[str, Any]:
    normalized_timestamp_ts = timestamp_obj or datetime.now(timezone.utc)
    if normalized_timestamp_ts.tzinfo is None:
        normalized_timestamp_ts = normalized_timestamp_ts.replace(tzinfo=timezone.utc)
    normalized_timestamp_ts = normalized_timestamp_ts.astimezone(timezone.utc)
    timestamp_str = normalized_timestamp_ts.isoformat()
    normalized_payload_dict = dict(event_payload_dict)
    account_id_obj = _first_non_empty_value_obj(
        normalized_payload_dict,
        ("account_id_str", "account_route_str", "account_str"),
    )
    structured_event_record_dict = {
        "event_name_str": event_name_str,
        "event_timestamp_str": timestamp_str,
        "ts_utc": timestamp_str,
        "level_str": _resolve_level_str(normalized_payload_dict),
        "mode_str": _first_non_empty_value_obj(
            normalized_payload_dict,
            ("mode_str", "env_mode_str", "session_mode_str"),
        ),
        "pod_id_str": _first_non_empty_value_obj(normalized_payload_dict, ("pod_id_str", "pod_str")),
        "account_id_str": str(account_id_obj) if account_id_obj is not None else None,
        "run_id_str": _derive_run_id_str(normalized_payload_dict, timestamp_str),
        "decision_plan_id_int": normalized_payload_dict.get("decision_plan_id_int"),
        "vplan_id_int": normalized_payload_dict.get("vplan_id_int"),
        "order_request_key_str": normalized_payload_dict.get("order_request_key_str"),
        "broker_order_id_str": normalized_payload_dict.get("broker_order_id_str"),
        "asset_str": normalized_payload_dict.get("asset_str"),
        "payload_dict": normalized_payload_dict,
    }
    structured_event_record_dict.update(normalized_payload_dict)
    structured_event_record_dict["event_name_str"] = event_name_str
    structured_event_record_dict["event_timestamp_str"] = timestamp_str
    structured_event_record_dict["ts_utc"] = timestamp_str
    structured_event_record_dict["level_str"] = _resolve_level_str(normalized_payload_dict)
    structured_event_record_dict["mode_str"] = _first_non_empty_value_obj(
        normalized_payload_dict,
        ("mode_str", "env_mode_str", "session_mode_str"),
    )
    structured_event_record_dict["pod_id_str"] = _first_non_empty_value_obj(
        normalized_payload_dict,
        ("pod_id_str", "pod_str"),
    )
    structured_event_record_dict["account_id_str"] = str(account_id_obj) if account_id_obj is not None else None
    structured_event_record_dict["run_id_str"] = _derive_run_id_str(normalized_payload_dict, timestamp_str)
    structured_event_record_dict["decision_plan_id_int"] = normalized_payload_dict.get("decision_plan_id_int")
    structured_event_record_dict["vplan_id_int"] = normalized_payload_dict.get("vplan_id_int")
    structured_event_record_dict["order_request_key_str"] = normalized_payload_dict.get("order_request_key_str")
    structured_event_record_dict["broker_order_id_str"] = normalized_payload_dict.get("broker_order_id_str")
    structured_event_record_dict["asset_str"] = normalized_payload_dict.get("asset_str")
    structured_event_record_dict["payload_dict"] = normalized_payload_dict
    return structured_event_record_dict


def log_event(
    event_name_str: str,
    event_payload_dict: dict[str, Any],
    log_path_str: str = DEFAULT_LOG_PATH_STR,
    max_bytes_int: int | None = DEFAULT_LOG_MAX_BYTES_INT,
    backup_count_int: int = DEFAULT_LOG_BACKUP_COUNT_INT,
) -> None:
    event_record_dict = build_structured_event_record_dict(event_name_str, event_payload_dict)
    target_log_path_obj_list = [Path(log_path_str)]
    if (
        str(event_payload_dict.get("severity_str", "")).lower() == "critical"
        and str(Path(log_path_str).resolve()) != str(Path(DEFAULT_CRITICAL_LOG_PATH_STR).resolve())
    ):
        target_log_path_obj_list.append(Path(DEFAULT_CRITICAL_LOG_PATH_STR))

    for log_path_obj in target_log_path_obj_list:
        _write_line_to_rolling_log(
            log_path_obj,
            json.dumps(event_record_dict, default=str, sort_keys=True),
            max_bytes_int=max_bytes_int,
            backup_count_int=backup_count_int,
        )


def resolve_operator_log_path_str(
    audit_log_path_str: str = DEFAULT_LOG_PATH_STR,
) -> str:
    audit_log_path_obj = Path(audit_log_path_str).resolve()
    default_log_path_obj = Path(DEFAULT_LOG_PATH_STR).resolve()
    if str(audit_log_path_obj) == str(default_log_path_obj):
        return DEFAULT_OPERATOR_LOG_PATH_STR
    return str(
        audit_log_path_obj.with_name(f"{audit_log_path_obj.stem}_operator.log")
    )


def _sanitize_path_part_str(raw_value_obj: object) -> str:
    sanitized_str = "".join(
        character_str if character_str.isalnum() or character_str in {"-", "_", "."} else "_"
        for character_str in str(raw_value_obj)
    ).strip("._")
    return sanitized_str or "unknown"


def resolve_pod_run_trace_log_path_str(
    *,
    pod_id_str: str,
    run_id_str: str,
    trace_log_root_path_str: str = DEFAULT_POD_TRACE_LOG_ROOT_PATH_STR,
) -> str:
    return str(
        Path(trace_log_root_path_str)
        / _sanitize_path_part_str(pod_id_str)
        / _sanitize_path_part_str(run_id_str)
        / "trace_events.jsonl"
    )


def log_trace_event(
    event_name_str: str,
    event_payload_dict: dict[str, Any],
    *,
    trace_enabled_bool: bool,
    trace_log_path_str: str | None = None,
    trace_log_root_path_str: str = DEFAULT_POD_TRACE_LOG_ROOT_PATH_STR,
    max_bytes_int: int | None = DEFAULT_TRACE_LOG_MAX_BYTES_INT,
    backup_count_int: int = DEFAULT_TRACE_LOG_BACKUP_COUNT_INT,
) -> str | None:
    if not trace_enabled_bool:
        return None
    event_record_dict = build_structured_event_record_dict(event_name_str, event_payload_dict)
    resolved_trace_log_path_str = trace_log_path_str
    if resolved_trace_log_path_str is None:
        resolved_trace_log_path_str = resolve_pod_run_trace_log_path_str(
            pod_id_str=str(event_record_dict.get("pod_id_str") or "unknown"),
            run_id_str=str(event_record_dict.get("run_id_str") or "unknown"),
            trace_log_root_path_str=trace_log_root_path_str,
        )
    trace_log_path_obj = Path(resolved_trace_log_path_str)
    _write_line_to_rolling_log(
        trace_log_path_obj,
        json.dumps(event_record_dict, default=str, sort_keys=True),
        max_bytes_int=max_bytes_int,
        backup_count_int=backup_count_int,
    )
    return str(trace_log_path_obj)


def format_operator_timestamp_str(timestamp_obj: datetime | str | None) -> str:
    if timestamp_obj is None:
        normalized_timestamp_ts = datetime.now(timezone.utc)
    elif isinstance(timestamp_obj, str):
        normalized_timestamp_ts = datetime.fromisoformat(timestamp_obj)
    else:
        normalized_timestamp_ts = timestamp_obj
    if normalized_timestamp_ts.tzinfo is None:
        normalized_timestamp_ts = normalized_timestamp_ts.replace(tzinfo=timezone.utc)
    normalized_timestamp_ts = normalized_timestamp_ts.astimezone(timezone.utc)
    return normalized_timestamp_ts.strftime("%Y-%m-%d %H:%M:%S UTC")


def render_operator_message_str(
    level_str: str,
    phase_action_str: str,
    timestamp_obj: datetime | str | None,
    field_map_dict: dict[str, Any] | None = None,
) -> str:
    normalized_field_map_dict = {
        str(field_name_str): field_value_obj
        for field_name_str, field_value_obj in dict(field_map_dict or {}).items()
        if field_value_obj is not None and str(field_value_obj) != ""
    }
    message_fragment_list = [
        f"[{format_operator_timestamp_str(timestamp_obj)}]",
        str(level_str).upper(),
        str(phase_action_str),
    ]
    message_fragment_list.extend(
        f"{field_name_str}={field_value_obj}"
        for field_name_str, field_value_obj in normalized_field_map_dict.items()
    )
    return " ".join(message_fragment_list)


def log_operator_message(
    *,
    level_str: str,
    phase_action_str: str,
    timestamp_obj: datetime | str | None,
    field_map_dict: dict[str, Any] | None = None,
    audit_log_path_str: str = DEFAULT_LOG_PATH_STR,
    operator_log_path_str: str | None = None,
    print_message_bool: bool = False,
    max_bytes_int: int | None = DEFAULT_LOG_MAX_BYTES_INT,
    backup_count_int: int = DEFAULT_LOG_BACKUP_COUNT_INT,
) -> str:
    message_str = render_operator_message_str(
        level_str=level_str,
        phase_action_str=phase_action_str,
        timestamp_obj=timestamp_obj,
        field_map_dict=field_map_dict,
    )
    target_operator_log_path_obj = Path(
        operator_log_path_str or resolve_operator_log_path_str(audit_log_path_str)
    )
    _write_line_to_rolling_log(
        target_operator_log_path_obj,
        message_str,
        max_bytes_int=max_bytes_int,
        backup_count_int=backup_count_int,
    )
    if print_message_bool:
        print(message_str, flush=True)
    return message_str
