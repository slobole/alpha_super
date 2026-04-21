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


def log_event(
    event_name_str: str,
    event_payload_dict: dict[str, Any],
    log_path_str: str = DEFAULT_LOG_PATH_STR,
) -> None:
    event_record_dict = {
        "event_name_str": event_name_str,
        "event_timestamp_str": datetime.now(timezone.utc).isoformat(),
        **event_payload_dict,
    }
    target_log_path_obj_list = [Path(log_path_str)]
    if (
        str(event_payload_dict.get("severity_str", "")).lower() == "critical"
        and str(Path(log_path_str).resolve()) != str(Path(DEFAULT_CRITICAL_LOG_PATH_STR).resolve())
    ):
        target_log_path_obj_list.append(Path(DEFAULT_CRITICAL_LOG_PATH_STR))

    for log_path_obj in target_log_path_obj_list:
        log_path_obj.parent.mkdir(parents=True, exist_ok=True)
        with log_path_obj.open("a", encoding="utf-8") as log_file_obj:
            log_file_obj.write(json.dumps(event_record_dict, sort_keys=True) + "\n")


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
    target_operator_log_path_obj.parent.mkdir(parents=True, exist_ok=True)
    with target_operator_log_path_obj.open("a", encoding="utf-8") as operator_log_file_obj:
        operator_log_file_obj.write(message_str + "\n")
    if print_message_bool:
        print(message_str, flush=True)
    return message_str
