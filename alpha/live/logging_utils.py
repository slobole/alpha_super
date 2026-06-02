from __future__ import annotations

import json
import re
import shutil
import stat
from datetime import datetime, timedelta, timezone
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
DEFAULT_POD_TRACE_RETENTION_DAYS_INT = 90
TRACE_REDACTED_VALUE_STR = "***REDACTED***"
TRACE_SECRET_KEY_FRAGMENT_TUPLE = (
    "api_key",
    "apikey",
    "authorization",
    "bearer",
    "password",
    "secret",
    "token",
)
TRACE_SECRET_VALUE_PATTERN_TUPLE = (
    re.compile(r"(?i)(token|password|secret|api[_-]?key)\s*[:=]\s*[^\s,;]+"),
    re.compile(r"(?i)bearer\s+[A-Za-z0-9._~+/=-]+"),
)
_POD_TRACE_RETENTION_CLEANUP_DATE_BY_ROOT_DICT: dict[str, str] = {}


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


def _trace_value_contains_secret_key_bool(key_str: str) -> bool:
    normalized_key_str = str(key_str).lower()
    return any(
        secret_fragment_str in normalized_key_str
        for secret_fragment_str in TRACE_SECRET_KEY_FRAGMENT_TUPLE
    )


def redact_trace_secret_value_obj(value_obj: Any, key_str: str | None = None) -> Any:
    if key_str is not None and _trace_value_contains_secret_key_bool(key_str):
        return TRACE_REDACTED_VALUE_STR
    if isinstance(value_obj, str):
        redacted_value_str = value_obj
        for secret_pattern_obj in TRACE_SECRET_VALUE_PATTERN_TUPLE:
            redacted_value_str = secret_pattern_obj.sub(TRACE_REDACTED_VALUE_STR, redacted_value_str)
        return redacted_value_str
    if isinstance(value_obj, dict):
        return {
            str(child_key_obj): redact_trace_secret_value_obj(
                child_value_obj,
                key_str=str(child_key_obj),
            )
            for child_key_obj, child_value_obj in value_obj.items()
        }
    if isinstance(value_obj, (list, tuple, set)):
        return [
            redact_trace_secret_value_obj(child_value_obj)
            for child_value_obj in value_obj
        ]
    return value_obj


def build_pod_trace_run_id_str(
    *,
    mode_str: str,
    pod_id_str: str | None,
    as_of_timestamp_str: str,
) -> str:
    return "_".join(
        _sanitize_path_part_str(value_obj)
        for value_obj in (
            mode_str,
            pod_id_str or "unknown",
            as_of_timestamp_str.replace("+", "_").replace(":", ""),
        )
    )


def build_pod_trace_context_dict(
    *,
    mode_str: str,
    pod_id_str: str | None,
    as_of_timestamp_str: str,
    account_route_str: str | None = None,
    release_id_str: str | None = None,
    run_id_str: str | None = None,
    cycle_id_str: str | None = None,
    decision_plan_id_int: int | None = None,
    vplan_id_int: int | None = None,
) -> dict[str, Any]:
    resolved_run_id_str = run_id_str or build_pod_trace_run_id_str(
        mode_str=mode_str,
        pod_id_str=pod_id_str,
        as_of_timestamp_str=as_of_timestamp_str,
    )
    return {
        "mode_str": str(mode_str),
        "pod_id_str": pod_id_str,
        "account_route_str": account_route_str,
        "release_id_str": release_id_str,
        "run_id_str": resolved_run_id_str,
        "cycle_id_str": cycle_id_str or resolved_run_id_str,
        "as_of_timestamp_str": as_of_timestamp_str,
        "decision_plan_id_int": decision_plan_id_int,
        "vplan_id_int": vplan_id_int,
    }


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


def _safe_resolved_path_obj(path_obj: Path) -> Path | None:
    try:
        return path_obj.resolve()
    except (OSError, RuntimeError):
        return None


def _path_under_root_bool(path_obj: Path, root_path_obj: Path) -> bool:
    try:
        path_obj.relative_to(root_path_obj)
        return True
    except ValueError:
        return False


def _directory_contains_symlink_bool(directory_path_obj: Path) -> bool:
    try:
        for child_path_obj in directory_path_obj.rglob("*"):
            if child_path_obj.is_symlink() or _path_is_windows_reparse_point_bool(child_path_obj):
                return True
    except OSError:
        return True
    return False


def _path_is_windows_reparse_point_bool(path_obj: Path) -> bool:
    try:
        reparse_flag_int = int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0))
        if reparse_flag_int == 0:
            return False
        return bool(int(getattr(path_obj.lstat(), "st_file_attributes", 0)) & reparse_flag_int)
    except OSError:
        return True


def _trace_run_folder_has_marker_bool(run_folder_path_obj: Path) -> bool:
    try:
        for child_path_obj in run_folder_path_obj.iterdir():
            if not child_path_obj.is_file():
                continue
            if child_path_obj.name == "trace_events.jsonl" or child_path_obj.name.startswith(
                "trace_events.jsonl."
            ):
                return True
    except OSError:
        return False
    return False


def _newest_trace_run_folder_mtime_float(run_folder_path_obj: Path) -> float | None:
    newest_mtime_float: float | None = None
    try:
        for child_path_obj in run_folder_path_obj.rglob("*"):
            if child_path_obj.is_symlink() or not child_path_obj.is_file():
                continue
            child_mtime_float = float(child_path_obj.stat().st_mtime)
            if newest_mtime_float is None or child_mtime_float > newest_mtime_float:
                newest_mtime_float = child_mtime_float
        if newest_mtime_float is None:
            newest_mtime_float = float(run_folder_path_obj.stat().st_mtime)
    except OSError:
        return None
    return newest_mtime_float


def cleanup_pod_trace_retention_dict(
    *,
    trace_log_root_path_str: str = DEFAULT_POD_TRACE_LOG_ROOT_PATH_STR,
    active_trace_log_path_str: str | None = None,
    retention_days_int: int = DEFAULT_POD_TRACE_RETENTION_DAYS_INT,
    now_ts: datetime | None = None,
    allow_non_default_trace_root_bool: bool = False,
) -> dict[str, Any]:
    normalized_retention_days_int = int(retention_days_int)
    normalized_now_ts = now_ts or datetime.now(timezone.utc)
    if normalized_now_ts.tzinfo is None:
        normalized_now_ts = normalized_now_ts.replace(tzinfo=timezone.utc)
    normalized_now_ts = normalized_now_ts.astimezone(timezone.utc)
    cutoff_ts = normalized_now_ts - timedelta(days=normalized_retention_days_int)
    trace_root_path_obj = Path(trace_log_root_path_str)
    resolved_trace_root_path_obj = _safe_resolved_path_obj(trace_root_path_obj)
    active_run_folder_path_obj: Path | None = None
    if active_trace_log_path_str is not None:
        active_trace_log_path_obj = Path(active_trace_log_path_str)
        resolved_active_trace_log_path_obj = _safe_resolved_path_obj(active_trace_log_path_obj)
        if resolved_active_trace_log_path_obj is not None:
            active_run_folder_path_obj = resolved_active_trace_log_path_obj.parent
    cleanup_detail_dict: dict[str, Any] = {
        "cleanup_attempted_bool": True,
        "retention_days_int": normalized_retention_days_int,
        "cutoff_timestamp_str": cutoff_ts.isoformat(),
        "trace_log_root_path_str": str(trace_root_path_obj),
        "scanned_run_folder_count_int": 0,
        "deleted_run_folder_count_int": 0,
        "kept_run_folder_count_int": 0,
        "skipped_active_run_folder_count_int": 0,
        "skipped_non_trace_folder_count_int": 0,
        "skipped_symlink_count_int": 0,
        "skipped_unsafe_path_count_int": 0,
        "error_count_int": 0,
        "error_str_list": [],
    }
    if normalized_retention_days_int <= 0:
        cleanup_detail_dict["cleanup_skip_reason_str"] = "retention_disabled"
        return cleanup_detail_dict
    if resolved_trace_root_path_obj is None or not trace_root_path_obj.exists():
        cleanup_detail_dict["cleanup_skip_reason_str"] = "trace_root_missing"
        return cleanup_detail_dict
    default_trace_root_path_obj = _safe_resolved_path_obj(Path(DEFAULT_POD_TRACE_LOG_ROOT_PATH_STR))
    if (
        not allow_non_default_trace_root_bool
        and default_trace_root_path_obj is not None
        and resolved_trace_root_path_obj != default_trace_root_path_obj
    ):
        cleanup_detail_dict["cleanup_skip_reason_str"] = "non_default_trace_root"
        cleanup_detail_dict["skipped_unsafe_path_count_int"] = 1
        return cleanup_detail_dict
    if trace_root_path_obj.is_symlink() or _path_is_windows_reparse_point_bool(trace_root_path_obj):
        cleanup_detail_dict["cleanup_skip_reason_str"] = "trace_root_symlink"
        cleanup_detail_dict["skipped_symlink_count_int"] = 1
        return cleanup_detail_dict

    try:
        pod_folder_path_list = [
            child_path_obj
            for child_path_obj in trace_root_path_obj.iterdir()
            if child_path_obj.is_dir()
        ]
    except OSError as exc:
        cleanup_detail_dict["cleanup_skip_reason_str"] = "trace_root_scan_failed"
        cleanup_detail_dict["error_count_int"] = 1
        cleanup_detail_dict["error_str_list"] = [str(exc)]
        return cleanup_detail_dict

    for pod_folder_path_obj in pod_folder_path_list:
        if pod_folder_path_obj.is_symlink() or _path_is_windows_reparse_point_bool(pod_folder_path_obj):
            cleanup_detail_dict["skipped_symlink_count_int"] += 1
            continue
        resolved_pod_folder_path_obj = _safe_resolved_path_obj(pod_folder_path_obj)
        if (
            resolved_pod_folder_path_obj is None
            or not _path_under_root_bool(resolved_pod_folder_path_obj, resolved_trace_root_path_obj)
        ):
            cleanup_detail_dict["skipped_unsafe_path_count_int"] += 1
            continue
        try:
            run_folder_path_list = [
                child_path_obj
                for child_path_obj in pod_folder_path_obj.iterdir()
                if child_path_obj.is_dir()
            ]
        except OSError as exc:
            cleanup_detail_dict["error_count_int"] += 1
            cleanup_detail_dict["error_str_list"].append(str(exc))
            continue
        for run_folder_path_obj in run_folder_path_list:
            cleanup_detail_dict["scanned_run_folder_count_int"] += 1
            if (
                run_folder_path_obj.is_symlink()
                or _path_is_windows_reparse_point_bool(run_folder_path_obj)
                or _directory_contains_symlink_bool(run_folder_path_obj)
            ):
                cleanup_detail_dict["skipped_symlink_count_int"] += 1
                continue
            resolved_run_folder_path_obj = _safe_resolved_path_obj(run_folder_path_obj)
            if (
                resolved_run_folder_path_obj is None
                or not _path_under_root_bool(resolved_run_folder_path_obj, resolved_trace_root_path_obj)
            ):
                cleanup_detail_dict["skipped_unsafe_path_count_int"] += 1
                continue
            if (
                active_run_folder_path_obj is not None
                and resolved_run_folder_path_obj == active_run_folder_path_obj
            ):
                cleanup_detail_dict["skipped_active_run_folder_count_int"] += 1
                continue
            if not _trace_run_folder_has_marker_bool(run_folder_path_obj):
                cleanup_detail_dict["skipped_non_trace_folder_count_int"] += 1
                continue
            newest_mtime_float = _newest_trace_run_folder_mtime_float(run_folder_path_obj)
            if newest_mtime_float is None:
                cleanup_detail_dict["error_count_int"] += 1
                cleanup_detail_dict["error_str_list"].append(
                    f"could_not_read_mtime:{run_folder_path_obj}"
                )
                continue
            if datetime.fromtimestamp(newest_mtime_float, tz=timezone.utc) >= cutoff_ts:
                cleanup_detail_dict["kept_run_folder_count_int"] += 1
                continue
            try:
                shutil.rmtree(resolved_run_folder_path_obj)
                cleanup_detail_dict["deleted_run_folder_count_int"] += 1
            except OSError as exc:
                cleanup_detail_dict["error_count_int"] += 1
                cleanup_detail_dict["error_str_list"].append(str(exc))
    return cleanup_detail_dict


def cleanup_pod_trace_retention_if_due_dict(
    *,
    trace_log_root_path_str: str = DEFAULT_POD_TRACE_LOG_ROOT_PATH_STR,
    active_trace_log_path_str: str | None = None,
    retention_days_int: int = DEFAULT_POD_TRACE_RETENTION_DAYS_INT,
    now_ts: datetime | None = None,
    allow_non_default_trace_root_bool: bool = False,
) -> dict[str, Any]:
    normalized_now_ts = now_ts or datetime.now(timezone.utc)
    if normalized_now_ts.tzinfo is None:
        normalized_now_ts = normalized_now_ts.replace(tzinfo=timezone.utc)
    normalized_now_ts = normalized_now_ts.astimezone(timezone.utc)
    trace_root_path_obj = Path(trace_log_root_path_str)
    resolved_trace_root_path_obj = _safe_resolved_path_obj(trace_root_path_obj)
    throttle_key_str = str(resolved_trace_root_path_obj or trace_root_path_obj)
    cleanup_date_str = normalized_now_ts.date().isoformat()
    if _POD_TRACE_RETENTION_CLEANUP_DATE_BY_ROOT_DICT.get(throttle_key_str) == cleanup_date_str:
        return {
            "cleanup_attempted_bool": False,
            "cleanup_skip_reason_str": "throttled",
            "trace_log_root_path_str": str(trace_root_path_obj),
        }
    try:
        cleanup_detail_dict = cleanup_pod_trace_retention_dict(
            trace_log_root_path_str=trace_log_root_path_str,
            active_trace_log_path_str=active_trace_log_path_str,
            retention_days_int=retention_days_int,
            now_ts=normalized_now_ts,
            allow_non_default_trace_root_bool=allow_non_default_trace_root_bool,
        )
        if int(cleanup_detail_dict.get("error_count_int") or 0) == 0:
            _POD_TRACE_RETENTION_CLEANUP_DATE_BY_ROOT_DICT[throttle_key_str] = cleanup_date_str
        return cleanup_detail_dict
    except Exception as exc:
        return {
            "cleanup_attempted_bool": False,
            "cleanup_skip_reason_str": "cleanup_error",
            "trace_log_root_path_str": str(trace_root_path_obj),
            "error_str": str(exc),
        }


def log_trace_event(
    event_name_str: str,
    event_payload_dict: dict[str, Any],
    *,
    trace_enabled_bool: bool,
    trace_log_path_str: str | None = None,
    trace_log_root_path_str: str = DEFAULT_POD_TRACE_LOG_ROOT_PATH_STR,
    max_bytes_int: int | None = DEFAULT_TRACE_LOG_MAX_BYTES_INT,
    backup_count_int: int = DEFAULT_TRACE_LOG_BACKUP_COUNT_INT,
    retention_days_int: int = DEFAULT_POD_TRACE_RETENTION_DAYS_INT,
    allow_non_default_trace_root_bool: bool = False,
) -> str | None:
    if not trace_enabled_bool:
        return None
    event_record_dict = build_structured_event_record_dict(
        event_name_str,
        redact_trace_secret_value_obj(event_payload_dict),
    )
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
    cleanup_pod_trace_retention_if_due_dict(
        trace_log_root_path_str=trace_log_root_path_str,
        active_trace_log_path_str=str(trace_log_path_obj),
        retention_days_int=retention_days_int,
        allow_non_default_trace_root_bool=allow_non_default_trace_root_bool,
    )
    return str(trace_log_path_obj)


def log_pod_trace_event(
    event_name_str: str,
    *,
    trace_context_dict: dict[str, Any],
    status_str: str,
    reason_code_str: str,
    payload_dict: dict[str, Any] | None = None,
    trace_enabled_bool: bool,
    trace_log_path_str: str | None = None,
    trace_log_root_path_str: str = DEFAULT_POD_TRACE_LOG_ROOT_PATH_STR,
    max_bytes_int: int | None = DEFAULT_TRACE_LOG_MAX_BYTES_INT,
    backup_count_int: int = DEFAULT_TRACE_LOG_BACKUP_COUNT_INT,
    retention_days_int: int = DEFAULT_POD_TRACE_RETENTION_DAYS_INT,
    allow_non_default_trace_root_bool: bool = False,
) -> str | None:
    if not trace_enabled_bool:
        return None
    try:
        redacted_payload_dict = redact_trace_secret_value_obj(dict(payload_dict or {}))
        event_payload_dict = {
            **dict(trace_context_dict),
            "status_str": str(status_str),
            "reason_code_str": str(reason_code_str),
            "payload_dict": redacted_payload_dict,
        }
        return log_trace_event(
            event_name_str,
            event_payload_dict,
            trace_enabled_bool=True,
            trace_log_path_str=trace_log_path_str,
            trace_log_root_path_str=trace_log_root_path_str,
            max_bytes_int=max_bytes_int,
            backup_count_int=backup_count_int,
            retention_days_int=retention_days_int,
            allow_non_default_trace_root_bool=allow_non_default_trace_root_bool,
        )
    except Exception:
        return None


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
