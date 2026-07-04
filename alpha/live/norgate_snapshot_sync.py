from __future__ import annotations

import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from alpha.live import scheduler_utils
from alpha.live.logging_utils import DEFAULT_LOG_PATH_STR, log_event, log_operator_message
from alpha.live.models import LiveRelease
from alpha.live.release_manifest import load_release_list, select_enabled_release_list_for_mode
from data.norgate_snapshot_store import (
    ALPHA_USE_NORGATE_SNAPSHOT_ENV_STR,
    NORGATE_SNAPSHOT_ROOT_ENV_STR,
    NorgateSnapshotError,
    clear_snapshot_manifest_cache,
    get_snapshot_root_path_obj,
    is_snapshot_mode_enabled_bool,
    load_valid_snapshot_manifest,
)
from scripts.norgate_config_env import norgate_api_url_from_env_str
from scripts.serve_norgate_snapshot_api import NORGATE_API_TOKEN_ENV_STR
from scripts.sync_norgate_snapshots_api import sync_required_snapshots


SYNC_STATUS_FILE_NAME_STR = ".client_sync_status.json"
SYNC_LOCK_FILE_NAME_STR = ".sync.lock"
SYNC_LOCK_TTL_SECONDS_INT = 600
SYNC_FAILURE_COOLDOWN_SECONDS_INT = 60
SYNC_ACTIVE_WAIT_STATUS_SET = {"waiting", "failed"}
SNAPSHOT_STALE_FOR_CYCLE_REASON_CODE_STR = "snapshot_stale_for_cycle"
DEFAULT_PROVIDER_PUBLISH_GRACE_MINUTES_INT = 180
DEFAULT_STALE_ALERT_SUBMISSION_LEAD_MINUTES_INT = 30
SNAPSHOT_STALE_GATE_REASON_SET = {
    "snapshot_not_ready",
    "snapshot_not_ready_for_session",
    "snapshot_window_expired",
}


def _utc_now_str() -> str:
    return datetime.now(tz=UTC).isoformat()


def _snapshot_root_path_obj_or_none() -> Path | None:
    try:
        return get_snapshot_root_path_obj()
    except Exception:
        return None


def _status_path_obj(snapshot_root_path_obj: Path) -> Path:
    return snapshot_root_path_obj / SYNC_STATUS_FILE_NAME_STR


def _lock_path_obj(snapshot_root_path_obj: Path) -> Path:
    return snapshot_root_path_obj / SYNC_LOCK_FILE_NAME_STR


def read_client_sync_status_dict(snapshot_root_path_obj: Path | None = None) -> dict[str, Any]:
    root_path_obj = snapshot_root_path_obj or _snapshot_root_path_obj_or_none()
    if root_path_obj is None:
        return {}
    status_path_obj = _status_path_obj(root_path_obj)
    if not status_path_obj.exists():
        return {}
    try:
        status_obj = json.loads(status_path_obj.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return status_obj if isinstance(status_obj, dict) else {}


def _write_client_sync_status(snapshot_root_path_obj: Path, status_dict: dict[str, Any]) -> None:
    snapshot_root_path_obj.mkdir(parents=True, exist_ok=True)
    tmp_path_obj = snapshot_root_path_obj / f"{SYNC_STATUS_FILE_NAME_STR}.tmp"
    tmp_path_obj.write_text(json.dumps(status_dict, indent=2, sort_keys=True), encoding="utf-8")
    tmp_path_obj.replace(_status_path_obj(snapshot_root_path_obj))


def _emit_sync_event(
    event_name_str: str,
    status_dict: dict[str, Any],
    *,
    log_path_str: str,
    print_operator_bool: bool,
) -> None:
    log_event(event_name_str, status_dict, log_path_str=log_path_str)
    if event_name_str == "norgate_snapshot_sync_started":
        level_str = "INFO"
        phase_action_str = "norgate.sync.start"
    elif event_name_str == "norgate_snapshot_sync_ready":
        level_str = "INFO"
        phase_action_str = "norgate.sync.ready"
    elif event_name_str == "norgate_snapshot_sync_failed":
        level_str = "WARN"
        phase_action_str = "norgate.sync.failed"
    elif event_name_str == "norgate_snapshot_sync_waiting":
        level_str = "WARN"
        phase_action_str = "norgate.sync.waiting"
    else:
        level_str = "INFO"
        phase_action_str = "norgate.sync.skipped"
    log_operator_message(
        level_str=level_str,
        phase_action_str=phase_action_str,
        timestamp_obj=datetime.now(tz=UTC),
        field_map_dict={
            "status": status_dict.get("status_str"),
            "profiles": ",".join(str(item_obj) for item_obj in status_dict.get("required_profile_list", [])),
            "dates": json.dumps(status_dict.get("snapshot_date_by_profile_dict", {}), sort_keys=True),
            "reason": status_dict.get("reason_code_str"),
            "error": status_dict.get("error_str"),
        },
        audit_log_path_str=log_path_str,
        print_message_bool=print_operator_bool and event_name_str != "norgate_snapshot_sync_skipped",
    )


def _selected_release_list(
    releases_root_path_str: str,
    env_mode_str: str,
    pod_id_str: str | None,
) -> list[LiveRelease]:
    release_list = load_release_list(releases_root_path_str)
    selected_release_list = select_enabled_release_list_for_mode(release_list, env_mode_str)
    if pod_id_str is not None:
        selected_release_list = [
            release_obj for release_obj in selected_release_list if release_obj.pod_id_str == pod_id_str
        ]
    return selected_release_list


def _required_profile_list(release_list: list[LiveRelease]) -> list[str]:
    return sorted({str(release_obj.data_profile_str) for release_obj in release_list})


def _local_snapshot_detail_dict(profile_list: list[str]) -> dict[str, Any]:
    clear_snapshot_manifest_cache()
    snapshot_date_by_profile_dict: dict[str, str] = {}
    manifest_hash_by_profile_dict: dict[str, str] = {}
    error_by_profile_dict: dict[str, str] = {}
    for profile_str in profile_list:
        try:
            snapshot_manifest_obj = load_valid_snapshot_manifest(profile_str)
        except Exception as exc:
            error_by_profile_dict[profile_str] = str(exc)
            continue
        snapshot_date_by_profile_dict[profile_str] = snapshot_manifest_obj.snapshot_date_ts.date().isoformat()
        manifest_hash_by_profile_dict[profile_str] = snapshot_manifest_obj.manifest_hash_str
    return {
        "snapshot_date_by_profile_dict": snapshot_date_by_profile_dict,
        "manifest_hash_by_profile_dict": manifest_hash_by_profile_dict,
        "error_by_profile_dict": error_by_profile_dict,
        "all_profiles_ready_bool": len(error_by_profile_dict) == 0,
    }


def _build_gate_reason_by_release_dict(
    release_list: list[LiveRelease],
    as_of_ts: datetime,
) -> dict[str, str]:
    gate_reason_by_release_dict: dict[str, str] = {}
    for release_obj in release_list:
        signal_clock_str = scheduler_utils.normalize_signal_clock_str(release_obj.signal_clock_str)
        if signal_clock_str not in {"eod_snapshot_ready", "month_end_snapshot_ready"}:
            continue
        try:
            gate_dict = scheduler_utils.evaluate_build_gate_dict(release_obj, as_of_ts)
            gate_reason_by_release_dict[release_obj.release_id_str] = str(gate_dict.get("reason_code_str") or "")
        except NorgateSnapshotError as exc:
            gate_reason_by_release_dict[release_obj.release_id_str] = f"snapshot_error:{exc}"
    return gate_reason_by_release_dict


def _required_snapshot_session_by_release_dict(
    release_list: list[LiveRelease],
    as_of_ts: datetime,
) -> dict[str, Any]:
    required_snapshot_session_by_release_dict: dict[str, Any] = {}
    for release_obj in release_list:
        signal_clock_str = scheduler_utils.normalize_signal_clock_str(release_obj.signal_clock_str)
        if signal_clock_str == "eod_snapshot_ready":
            required_session_label_ts = scheduler_utils.get_latest_completed_session_label_ts(
                as_of_ts,
                release_obj.session_calendar_id_str,
            )
        elif signal_clock_str == "month_end_snapshot_ready":
            required_session_label_ts = scheduler_utils.get_latest_completed_month_end_session_label_ts(
                as_of_ts,
                release_obj.session_calendar_id_str,
            )
        else:
            continue
        if required_session_label_ts is not None:
            required_snapshot_session_by_release_dict[release_obj.release_id_str] = required_session_label_ts
    return required_snapshot_session_by_release_dict


def _required_snapshot_date_by_release_dict(
    required_snapshot_session_by_release_dict: dict[str, Any],
) -> dict[str, str]:
    required_snapshot_date_by_release_dict: dict[str, str] = {}
    for release_id_str, required_session_label_ts in required_snapshot_session_by_release_dict.items():
        required_snapshot_date_by_release_dict[release_id_str] = (
            required_session_label_ts.date().isoformat()
        )
    return required_snapshot_date_by_release_dict


def _stale_alert_deadline_by_release_dict(
    release_list: list[LiveRelease],
    required_snapshot_session_by_release_dict: dict[str, Any],
) -> dict[str, str]:
    stale_alert_deadline_by_release_dict: dict[str, str] = {}
    release_by_id_dict = {release_obj.release_id_str: release_obj for release_obj in release_list}
    for release_id_str, required_session_label_ts in required_snapshot_session_by_release_dict.items():
        release_obj = release_by_id_dict.get(release_id_str)
        if release_obj is None:
            continue
        close_deadline_ts = scheduler_utils.get_session_close_timestamp_ts(
            required_session_label_ts,
            release_obj.session_calendar_id_str,
        ) + timedelta(minutes=DEFAULT_PROVIDER_PUBLISH_GRACE_MINUTES_INT)
        submission_deadline_ts = scheduler_utils.build_submission_timestamp_ts(
            required_session_label_ts.to_pydatetime(),
            release_obj,
        ) - timedelta(minutes=DEFAULT_STALE_ALERT_SUBMISSION_LEAD_MINUTES_INT)
        alert_deadline_ts = min(close_deadline_ts, submission_deadline_ts)
        stale_alert_deadline_by_release_dict[release_id_str] = alert_deadline_ts.isoformat()
    return stale_alert_deadline_by_release_dict


def _timestamp_from_iso_or_none(timestamp_str: str | None) -> datetime | None:
    if not timestamp_str:
        return None
    try:
        timestamp_ts = datetime.fromisoformat(str(timestamp_str).replace("Z", "+00:00"))
    except ValueError:
        return None
    if timestamp_ts.tzinfo is None:
        return timestamp_ts.replace(tzinfo=UTC)
    return timestamp_ts.astimezone(UTC)


def _snapshot_stale_past_alert_deadline_bool(
    *,
    release_list: list[LiveRelease],
    local_detail_dict: dict[str, Any],
    as_of_ts: datetime,
) -> bool:
    stale_profile_set = {
        str(profile_obj)
        for profile_obj in local_detail_dict.get("stale_profile_list", [])
    }
    if len(stale_profile_set) == 0:
        return False
    release_by_id_dict = {release_obj.release_id_str: release_obj for release_obj in release_list}
    deadline_by_release_dict = dict(local_detail_dict.get("stale_alert_deadline_by_release_dict", {}))
    as_of_utc_ts = as_of_ts if as_of_ts.tzinfo is not None else as_of_ts.replace(tzinfo=UTC)
    as_of_utc_ts = as_of_utc_ts.astimezone(UTC)
    for release_id_str, deadline_str in deadline_by_release_dict.items():
        release_obj = release_by_id_dict.get(str(release_id_str))
        if release_obj is None or str(release_obj.data_profile_str) not in stale_profile_set:
            continue
        deadline_ts = _timestamp_from_iso_or_none(str(deadline_str))
        if deadline_ts is not None and as_of_utc_ts >= deadline_ts:
            return True
    return False


def _minimum_required_snapshot_date_by_profile_dict(
    release_list: list[LiveRelease],
    required_snapshot_date_by_release_dict: dict[str, str],
) -> dict[str, str]:
    minimum_required_snapshot_date_by_profile_dict: dict[str, str] = {}
    release_by_id_dict = {release_obj.release_id_str: release_obj for release_obj in release_list}
    for release_id_str, required_snapshot_date_str in required_snapshot_date_by_release_dict.items():
        release_obj = release_by_id_dict.get(release_id_str)
        if release_obj is None:
            continue
        profile_str = str(release_obj.data_profile_str)
        current_required_date_str = minimum_required_snapshot_date_by_profile_dict.get(profile_str)
        if current_required_date_str is None or required_snapshot_date_str > current_required_date_str:
            minimum_required_snapshot_date_by_profile_dict[profile_str] = required_snapshot_date_str
    return minimum_required_snapshot_date_by_profile_dict


def _with_cycle_freshness_detail_dict(
    local_detail_dict: dict[str, Any],
    release_list: list[LiveRelease],
    as_of_ts: datetime,
) -> dict[str, Any]:
    detail_dict = dict(local_detail_dict)
    snapshot_date_by_profile_dict = dict(detail_dict.get("snapshot_date_by_profile_dict", {}))
    required_snapshot_session_by_release_dict = _required_snapshot_session_by_release_dict(
        release_list,
        as_of_ts,
    )
    required_snapshot_date_by_release_dict = _required_snapshot_date_by_release_dict(
        required_snapshot_session_by_release_dict
    )
    stale_alert_deadline_by_release_dict = _stale_alert_deadline_by_release_dict(
        release_list,
        required_snapshot_session_by_release_dict,
    )
    minimum_required_snapshot_date_by_profile_dict = (
        _minimum_required_snapshot_date_by_profile_dict(
            release_list,
            required_snapshot_date_by_release_dict,
        )
    )
    stale_profile_list = []
    for profile_str, minimum_required_snapshot_date_str in (
        minimum_required_snapshot_date_by_profile_dict.items()
    ):
        snapshot_date_str = snapshot_date_by_profile_dict.get(profile_str)
        if snapshot_date_str is None or snapshot_date_str < minimum_required_snapshot_date_str:
            stale_profile_list.append(profile_str)

    detail_dict["required_snapshot_date_by_release_dict"] = required_snapshot_date_by_release_dict
    detail_dict["stale_alert_deadline_by_release_dict"] = stale_alert_deadline_by_release_dict
    detail_dict["minimum_required_snapshot_date_by_profile_dict"] = (
        minimum_required_snapshot_date_by_profile_dict
    )
    detail_dict["stale_profile_list"] = sorted(stale_profile_list)
    detail_dict["snapshot_fresh_for_cycle_bool"] = len(stale_profile_list) == 0
    detail_dict["snapshot_stale_past_alert_deadline_bool"] = (
        _snapshot_stale_past_alert_deadline_bool(
            release_list=release_list,
            local_detail_dict=detail_dict,
            as_of_ts=as_of_ts,
        )
    )
    return detail_dict


def _stale_snapshot_error_str(local_detail_dict: dict[str, Any]) -> str | None:
    stale_profile_list = [str(profile_obj) for profile_obj in local_detail_dict.get("stale_profile_list", [])]
    if len(stale_profile_list) == 0:
        return None
    snapshot_date_by_profile_dict = dict(local_detail_dict.get("snapshot_date_by_profile_dict", {}))
    minimum_required_snapshot_date_by_profile_dict = dict(
        local_detail_dict.get("minimum_required_snapshot_date_by_profile_dict", {})
    )
    profile_str = stale_profile_list[0]
    required_snapshot_date_str = str(
        minimum_required_snapshot_date_by_profile_dict.get(profile_str) or "unknown"
    )
    snapshot_date_str = str(snapshot_date_by_profile_dict.get(profile_str) or "missing")
    return (
        "Local Norgate data is too old for the next DecisionPlan. "
        f"Required data date: {required_snapshot_date_str}. "
        f"Local data date: {snapshot_date_str}."
    )


def _operator_message_str(status_dict: dict[str, Any]) -> str:
    stale_error_str = _stale_snapshot_error_str(status_dict)
    if stale_error_str is not None:
        status_str = str(status_dict.get("status_str") or "")
        reason_code_str = str(status_dict.get("reason_code_str") or "")
        hard_stale_bool = status_str in {"failed", "local_snapshot_only"} or reason_code_str == "api_config_missing"
        if (
            not hard_stale_bool
            and not bool(status_dict.get("snapshot_stale_past_alert_deadline_bool", False))
        ):
            return (
                f"Waiting: {stale_error_str} "
                "This is still inside the normal Norgate publish window."
            )
        return f"Blocked: {stale_error_str}"

    status_str = str(status_dict.get("status_str") or "")
    reason_code_str = str(status_dict.get("reason_code_str") or "")
    if status_str == "direct":
        return "Direct Norgate mode is active."
    if status_str == "ready":
        minimum_required_snapshot_date_by_profile_dict = dict(
            status_dict.get("minimum_required_snapshot_date_by_profile_dict", {})
        )
        snapshot_date_by_profile_dict = dict(status_dict.get("snapshot_date_by_profile_dict", {}))
        if minimum_required_snapshot_date_by_profile_dict:
            profile_str = sorted(minimum_required_snapshot_date_by_profile_dict)[0]
            required_snapshot_date_str = minimum_required_snapshot_date_by_profile_dict[profile_str]
            snapshot_date_str = str(snapshot_date_by_profile_dict.get(profile_str) or "missing")
            return (
                "Norgate data is fresh for the next DecisionPlan. "
                f"Required data date: {required_snapshot_date_str}. "
                f"Local data date: {snapshot_date_str}."
            )
        return "Local Norgate snapshot is valid."
    if reason_code_str == "sync_started":
        return "Norgate sync is running. New DecisionPlan waits for fresh data."
    if reason_code_str == "sync_lock_busy":
        return "Another local Norgate sync is already running."
    if reason_code_str == "sync_failure_cooldown":
        return "Norgate sync failed recently. Waiting before retrying."
    if reason_code_str == "sync_waiting_for_newer_snapshot":
        return "Norgate server has not provided newer data yet. Waiting before retrying."
    if reason_code_str == "api_config_missing":
        return "Norgate API is not configured. New DecisionPlan may require local data."
    if status_str == "failed":
        return "Norgate sync failed. New DecisionPlan is blocked until fresh data is available."
    if status_str == "waiting":
        return "Waiting for Norgate snapshot data."
    if status_str == "local_snapshot_only":
        return "Norgate API is not configured; only local snapshot files are available."
    return "Norgate snapshot status is unknown."


def _operator_action_str(status_dict: dict[str, Any]) -> str:
    if _stale_snapshot_error_str(status_dict) is not None:
        status_str = str(status_dict.get("status_str") or "")
        reason_code_str = str(status_dict.get("reason_code_str") or "")
        hard_stale_bool = status_str in {"failed", "local_snapshot_only"} or reason_code_str == "api_config_missing"
        if (
            not hard_stale_bool
            and not bool(status_dict.get("snapshot_stale_past_alert_deadline_bool", False))
        ):
            return "Wait for Norgate provider data; run doctor/sync if this persists."
        return "Check Norgate server or run Norgate doctor/sync."
    reason_code_str = str(status_dict.get("reason_code_str") or "")
    status_str = str(status_dict.get("status_str") or "")
    if reason_code_str == "sync_lock_busy":
        return "Wait for the current sync to finish."
    if reason_code_str == "sync_failure_cooldown":
        return "Wait for the cooldown or check the last sync error."
    if reason_code_str == "sync_waiting_for_newer_snapshot":
        return "Wait for Norgate provider data; run doctor/sync if this persists."
    if status_str == "failed":
        return "Check the Norgate sync error and server/API health."
    if status_str in {"ready", "direct"}:
        return "No action needed."
    return "Inspect Norgate sync status."


def _needs_sync_bool(
    *,
    local_detail_dict: dict[str, Any],
    gate_reason_by_release_dict: dict[str, str],
) -> bool:
    if not bool(local_detail_dict.get("all_profiles_ready_bool")):
        return True
    if len(local_detail_dict.get("stale_profile_list", [])) > 0:
        return True
    return any(
        reason_str in SNAPSHOT_STALE_GATE_REASON_SET or reason_str.startswith("snapshot_error:")
        for reason_str in gate_reason_by_release_dict.values()
    )


def _api_config_dict() -> dict[str, str | None]:
    api_url_str = norgate_api_url_from_env_str()
    token_str = os.getenv(NORGATE_API_TOKEN_ENV_STR, "").strip() or None
    client_id_str = os.getenv("NORGATE_CLIENT_ID", "").strip() or None
    return {
        "api_url_str": api_url_str,
        "token_str": token_str,
        "client_id_str": client_id_str,
    }


def _api_config_missing_error_str(api_config_dict: dict[str, str | None]) -> str | None:
    missing_name_list: list[str] = []
    if not api_config_dict.get("api_url_str"):
        missing_name_list.append("NORGATE_API_URL or NORGATE_API_HOST/NORGATE_API_PORT")
    if not api_config_dict.get("token_str"):
        missing_name_list.append(NORGATE_API_TOKEN_ENV_STR)
    if not api_config_dict.get("client_id_str"):
        missing_name_list.append("NORGATE_CLIENT_ID")
    if len(missing_name_list) == 0:
        return None
    return "Missing Norgate API config: " + ", ".join(missing_name_list) + "."


def _parse_timestamp_or_none(raw_timestamp_obj: object) -> datetime | None:
    if not raw_timestamp_obj:
        return None
    try:
        timestamp_ts = datetime.fromisoformat(str(raw_timestamp_obj))
    except Exception:
        return None
    if timestamp_ts.tzinfo is None:
        return timestamp_ts.replace(tzinfo=UTC)
    return timestamp_ts.astimezone(UTC)


def _cooldown_active_bool(status_dict: dict[str, Any], now_ts: datetime) -> bool:
    if status_dict.get("status_str") not in {"failed", "waiting"}:
        return False
    last_attempt_ts = _parse_timestamp_or_none(status_dict.get("last_attempt_utc_str"))
    if last_attempt_ts is None:
        return False
    return (now_ts - last_attempt_ts).total_seconds() < SYNC_FAILURE_COOLDOWN_SECONDS_INT


def _acquire_sync_lock_bool(snapshot_root_path_obj: Path, now_ts: datetime) -> bool:
    lock_path_obj = _lock_path_obj(snapshot_root_path_obj)
    snapshot_root_path_obj.mkdir(parents=True, exist_ok=True)
    if lock_path_obj.exists():
        lock_age_seconds_float = now_ts.timestamp() - lock_path_obj.stat().st_mtime
        if lock_age_seconds_float <= SYNC_LOCK_TTL_SECONDS_INT:
            return False
        lock_path_obj.unlink(missing_ok=True)
    try:
        file_descriptor_int = os.open(str(lock_path_obj), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    with os.fdopen(file_descriptor_int, "w", encoding="utf-8") as lock_file_obj:
        lock_file_obj.write(
            json.dumps(
                {
                    "created_utc_str": now_ts.isoformat(),
                    "pid_int": os.getpid(),
                },
                sort_keys=True,
            )
        )
    return True


def _release_sync_lock(snapshot_root_path_obj: Path) -> None:
    _lock_path_obj(snapshot_root_path_obj).unlink(missing_ok=True)


def _base_status_dict(
    *,
    status_str: str,
    release_list: list[LiveRelease],
    profile_list: list[str],
    local_detail_dict: dict[str, Any] | None,
    gate_reason_by_release_dict: dict[str, str] | None,
    error_str: str | None,
    reason_code_str: str | None,
    last_attempt_utc_str: str | None,
    last_success_utc_str: str | None,
) -> dict[str, Any]:
    status_dict: dict[str, Any] = {
        "status_str": status_str,
        "data_source_mode_str": "snapshot" if is_snapshot_mode_enabled_bool() else "direct",
        "last_attempt_utc_str": last_attempt_utc_str,
        "last_success_utc_str": last_success_utc_str,
        "required_profile_list": list(profile_list),
        "snapshot_date_by_profile_dict": dict(
            (local_detail_dict or {}).get("snapshot_date_by_profile_dict", {})
        ),
        "manifest_hash_by_profile_dict": dict(
            (local_detail_dict or {}).get("manifest_hash_by_profile_dict", {})
        ),
        "error_by_profile_dict": dict((local_detail_dict or {}).get("error_by_profile_dict", {})),
        "required_snapshot_date_by_release_dict": dict(
            (local_detail_dict or {}).get("required_snapshot_date_by_release_dict", {})
        ),
        "stale_alert_deadline_by_release_dict": dict(
            (local_detail_dict or {}).get("stale_alert_deadline_by_release_dict", {})
        ),
        "minimum_required_snapshot_date_by_profile_dict": dict(
            (local_detail_dict or {}).get("minimum_required_snapshot_date_by_profile_dict", {})
        ),
        "stale_profile_list": list((local_detail_dict or {}).get("stale_profile_list", [])),
        "snapshot_fresh_for_cycle_bool": bool(
            (local_detail_dict or {}).get("snapshot_fresh_for_cycle_bool", True)
        ),
        "snapshot_stale_past_alert_deadline_bool": bool(
            (local_detail_dict or {}).get("snapshot_stale_past_alert_deadline_bool", False)
        ),
        "gate_reason_by_release_id_dict": dict(gate_reason_by_release_dict or {}),
        "release_id_list": [release_obj.release_id_str for release_obj in release_list],
        "pod_id_list": [release_obj.pod_id_str for release_obj in release_list],
        "reason_code_str": reason_code_str,
        "error_str": error_str,
    }
    status_dict["operator_message_str"] = _operator_message_str(status_dict)
    status_dict["operator_action_str"] = _operator_action_str(status_dict)
    return status_dict


def _latest_status_success_utc_str(
    previous_status_dict: dict[str, Any],
    fallback_utc_str: str | None = None,
) -> str | None:
    previous_success_str = previous_status_dict.get("last_success_utc_str")
    if previous_success_str:
        return str(previous_success_str)
    return fallback_utc_str


def ensure_norgate_snapshots_for_live_tick(
    *,
    releases_root_path_str: str,
    env_mode_str: str,
    as_of_ts: datetime,
    log_path_str: str = DEFAULT_LOG_PATH_STR,
    pod_id_str: str | None = None,
    print_operator_bool: bool = False,
) -> dict[str, Any]:
    if not is_snapshot_mode_enabled_bool():
        status_dict = _base_status_dict(
            status_str="direct",
            release_list=[],
            profile_list=[],
            local_detail_dict=None,
            gate_reason_by_release_dict=None,
            error_str=None,
            reason_code_str="direct_norgate_mode",
            last_attempt_utc_str=None,
            last_success_utc_str=None,
        )
        _emit_sync_event(
            "norgate_snapshot_sync_skipped",
            status_dict,
            log_path_str=log_path_str,
            print_operator_bool=print_operator_bool,
        )
        return status_dict

    snapshot_root_path_obj = _snapshot_root_path_obj_or_none()
    if snapshot_root_path_obj is None:
        release_list = _selected_release_list(releases_root_path_str, env_mode_str, pod_id_str)
        profile_list = _required_profile_list(release_list)
        status_dict = _base_status_dict(
            status_str="failed",
            release_list=release_list,
            profile_list=profile_list,
            local_detail_dict=None,
            gate_reason_by_release_dict=None,
            error_str=(
                f"{NORGATE_SNAPSHOT_ROOT_ENV_STR} must be set when "
                f"{ALPHA_USE_NORGATE_SNAPSHOT_ENV_STR}=true."
            ),
            reason_code_str="snapshot_root_missing",
            last_attempt_utc_str=_utc_now_str(),
            last_success_utc_str=None,
        )
        _emit_sync_event(
            "norgate_snapshot_sync_failed",
            status_dict,
            log_path_str=log_path_str,
            print_operator_bool=print_operator_bool,
        )
        return status_dict

    release_list = _selected_release_list(releases_root_path_str, env_mode_str, pod_id_str)
    profile_list = _required_profile_list(release_list)
    previous_status_dict = read_client_sync_status_dict(snapshot_root_path_obj)
    local_detail_dict = _with_cycle_freshness_detail_dict(
        _local_snapshot_detail_dict(profile_list),
        release_list,
        as_of_ts,
    )
    gate_reason_by_release_dict = _build_gate_reason_by_release_dict(release_list, as_of_ts)

    if len(profile_list) == 0:
        status_dict = _base_status_dict(
            status_str="ready",
            release_list=release_list,
            profile_list=profile_list,
            local_detail_dict=local_detail_dict,
            gate_reason_by_release_dict=gate_reason_by_release_dict,
            error_str=None,
            reason_code_str="no_enabled_releases",
            last_attempt_utc_str=None,
            last_success_utc_str=_latest_status_success_utc_str(previous_status_dict),
        )
        _write_client_sync_status(snapshot_root_path_obj, status_dict)
        _emit_sync_event(
            "norgate_snapshot_sync_skipped",
            status_dict,
            log_path_str=log_path_str,
            print_operator_bool=print_operator_bool,
        )
        return status_dict

    if not _needs_sync_bool(
        local_detail_dict=local_detail_dict,
        gate_reason_by_release_dict=gate_reason_by_release_dict,
    ):
        status_dict = _base_status_dict(
            status_str="ready",
            release_list=release_list,
            profile_list=profile_list,
            local_detail_dict=local_detail_dict,
            gate_reason_by_release_dict=gate_reason_by_release_dict,
            error_str=None,
            reason_code_str="local_snapshot_ready",
            last_attempt_utc_str=None,
            last_success_utc_str=_latest_status_success_utc_str(previous_status_dict),
        )
        _write_client_sync_status(snapshot_root_path_obj, status_dict)
        _emit_sync_event(
            "norgate_snapshot_sync_skipped",
            status_dict,
            log_path_str=log_path_str,
            print_operator_bool=print_operator_bool,
        )
        return status_dict

    api_config_dict = _api_config_dict()
    missing_api_error_str = _api_config_missing_error_str(api_config_dict)
    if missing_api_error_str is not None:
        local_ready_bool = bool(local_detail_dict.get("all_profiles_ready_bool"))
        status_dict = _base_status_dict(
            status_str="local_snapshot_only" if local_ready_bool else "waiting",
            release_list=release_list,
            profile_list=profile_list,
            local_detail_dict=local_detail_dict,
            gate_reason_by_release_dict=gate_reason_by_release_dict,
            error_str=missing_api_error_str,
            reason_code_str="api_config_missing",
            last_attempt_utc_str=_utc_now_str(),
            last_success_utc_str=_latest_status_success_utc_str(previous_status_dict),
        )
        _write_client_sync_status(snapshot_root_path_obj, status_dict)
        _emit_sync_event(
            "norgate_snapshot_sync_failed" if not local_ready_bool else "norgate_snapshot_sync_skipped",
            status_dict,
            log_path_str=log_path_str,
            print_operator_bool=print_operator_bool,
        )
        return status_dict

    now_ts = datetime.now(tz=UTC)
    if _cooldown_active_bool(previous_status_dict, now_ts):
        previous_reason_code_str = str(previous_status_dict.get("reason_code_str") or "")
        cooldown_reason_code_str = (
            "sync_waiting_for_newer_snapshot"
            if previous_reason_code_str == SNAPSHOT_STALE_FOR_CYCLE_REASON_CODE_STR
            else "sync_failure_cooldown"
        )
        status_dict = _base_status_dict(
            status_str="waiting",
            release_list=release_list,
            profile_list=profile_list,
            local_detail_dict=local_detail_dict,
            gate_reason_by_release_dict=gate_reason_by_release_dict,
            error_str=str(
                previous_status_dict.get("error_str")
                or (
                    "Previous sync did not produce newer Norgate data yet."
                    if cooldown_reason_code_str == "sync_waiting_for_newer_snapshot"
                    else "Previous sync failed recently."
                )
            ),
            reason_code_str=cooldown_reason_code_str,
            last_attempt_utc_str=str(previous_status_dict.get("last_attempt_utc_str")),
            last_success_utc_str=_latest_status_success_utc_str(previous_status_dict),
        )
        _write_client_sync_status(snapshot_root_path_obj, status_dict)
        _emit_sync_event(
            "norgate_snapshot_sync_skipped",
            status_dict,
            log_path_str=log_path_str,
            print_operator_bool=print_operator_bool,
        )
        return status_dict

    if not _acquire_sync_lock_bool(snapshot_root_path_obj, now_ts):
        status_dict = _base_status_dict(
            status_str="waiting",
            release_list=release_list,
            profile_list=profile_list,
            local_detail_dict=local_detail_dict,
            gate_reason_by_release_dict=gate_reason_by_release_dict,
            error_str="Another local Norgate snapshot sync is already running.",
            reason_code_str="sync_lock_busy",
            last_attempt_utc_str=now_ts.isoformat(),
            last_success_utc_str=_latest_status_success_utc_str(previous_status_dict),
        )
        _write_client_sync_status(snapshot_root_path_obj, status_dict)
        _emit_sync_event(
            "norgate_snapshot_sync_skipped",
            status_dict,
            log_path_str=log_path_str,
            print_operator_bool=print_operator_bool,
        )
        return status_dict

    started_status_dict = _base_status_dict(
        status_str="waiting",
        release_list=release_list,
        profile_list=profile_list,
        local_detail_dict=local_detail_dict,
        gate_reason_by_release_dict=gate_reason_by_release_dict,
        error_str=None,
        reason_code_str="sync_started",
        last_attempt_utc_str=now_ts.isoformat(),
        last_success_utc_str=_latest_status_success_utc_str(previous_status_dict),
    )
    _write_client_sync_status(snapshot_root_path_obj, started_status_dict)
    _emit_sync_event(
        "norgate_snapshot_sync_started",
        started_status_dict,
        log_path_str=log_path_str,
        print_operator_bool=print_operator_bool,
    )

    try:
        promoted_path_list = sync_required_snapshots(
            api_url_str=str(api_config_dict["api_url_str"]),
            token_str=str(api_config_dict["token_str"]),
            client_id_str=str(api_config_dict["client_id_str"]),
            releases_root_path_str=releases_root_path_str,
            local_root_path_str=str(snapshot_root_path_obj),
            mode_str=env_mode_str,
            pod_id_str=pod_id_str,
            overwrite_bool=True,
            skip_valid_existing_bool=True,
        )
        local_detail_dict = _with_cycle_freshness_detail_dict(
            _local_snapshot_detail_dict(profile_list),
            release_list,
            as_of_ts,
        )
        gate_reason_by_release_dict = _build_gate_reason_by_release_dict(release_list, as_of_ts)
        success_utc_str = _utc_now_str()
        post_sync_stale_error_str = _stale_snapshot_error_str(local_detail_dict)
        post_sync_ready_bool = bool(local_detail_dict.get("all_profiles_ready_bool")) and (
            post_sync_stale_error_str is None
        )
        post_sync_error_str = post_sync_stale_error_str
        post_sync_reason_code_str = "sync_ready"
        if not post_sync_ready_bool:
            if post_sync_stale_error_str is not None:
                post_sync_reason_code_str = SNAPSHOT_STALE_FOR_CYCLE_REASON_CODE_STR
            else:
                post_sync_reason_code_str = "snapshot_not_ready"
                post_sync_error_str = json.dumps(
                    local_detail_dict.get("error_by_profile_dict", {}),
                    sort_keys=True,
                )
        status_dict = _base_status_dict(
            status_str="ready" if post_sync_ready_bool else "waiting",
            release_list=release_list,
            profile_list=profile_list,
            local_detail_dict=local_detail_dict,
            gate_reason_by_release_dict=gate_reason_by_release_dict,
            error_str=post_sync_error_str,
            reason_code_str=post_sync_reason_code_str,
            last_attempt_utc_str=started_status_dict["last_attempt_utc_str"],
            last_success_utc_str=(
                success_utc_str
                if post_sync_ready_bool
                else _latest_status_success_utc_str(previous_status_dict)
            ),
        )
        status_dict["promoted_path_list"] = [str(path_obj) for path_obj in promoted_path_list]
        _write_client_sync_status(snapshot_root_path_obj, status_dict)
        _emit_sync_event(
            "norgate_snapshot_sync_ready"
            if post_sync_ready_bool
            else "norgate_snapshot_sync_waiting",
            status_dict,
            log_path_str=log_path_str,
            print_operator_bool=print_operator_bool,
        )
        return status_dict
    except Exception as exc:
        local_detail_dict = _with_cycle_freshness_detail_dict(
            _local_snapshot_detail_dict(profile_list),
            release_list,
            as_of_ts,
        )
        status_dict = _base_status_dict(
            status_str="failed",
            release_list=release_list,
            profile_list=profile_list,
            local_detail_dict=local_detail_dict,
            gate_reason_by_release_dict=gate_reason_by_release_dict,
            error_str=str(exc),
            reason_code_str="sync_failed",
            last_attempt_utc_str=started_status_dict["last_attempt_utc_str"],
            last_success_utc_str=_latest_status_success_utc_str(previous_status_dict),
        )
        _write_client_sync_status(snapshot_root_path_obj, status_dict)
        _emit_sync_event(
            "norgate_snapshot_sync_failed",
            status_dict,
            log_path_str=log_path_str,
            print_operator_bool=print_operator_bool,
        )
        return status_dict
    finally:
        _release_sync_lock(snapshot_root_path_obj)


def norgate_snapshot_sync_active_wait_bool(status_dict: dict[str, Any] | None) -> bool:
    if not status_dict:
        return False
    return str(status_dict.get("status_str") or "") in SYNC_ACTIVE_WAIT_STATUS_SET


def _status_severity_str(
    status_str: str,
    build_gate_reason_code_str: str | None,
    snapshot_fresh_for_cycle_bool: bool = True,
    snapshot_stale_past_alert_deadline_bool: bool = False,
    reason_code_str: str | None = None,
) -> str:
    if not snapshot_fresh_for_cycle_bool:
        if status_str in {"failed", "local_snapshot_only"} or reason_code_str == "api_config_missing":
            return "red"
        if snapshot_stale_past_alert_deadline_bool:
            return "red"
        return "yellow"
    if status_str == "failed":
        return "red"
    if status_str == "ready" and build_gate_reason_code_str == "snapshot_window_expired":
        return "red"
    if status_str == "ready" and build_gate_reason_code_str in SNAPSHOT_STALE_GATE_REASON_SET:
        return "yellow"
    if status_str in {"waiting", "local_snapshot_only"}:
        if build_gate_reason_code_str in {"snapshot_ready", "carry_forward_snapshot_ready"}:
            return "green"
        return "yellow"
    if status_str in {"direct", "ready"}:
        return "green"
    return "gray"


def _sync_stage_label_str(
    *,
    data_source_mode_str: str,
    status_str: str,
    reason_code_str: str | None,
    build_gate_reason_code_str: str | None = None,
    snapshot_fresh_for_cycle_bool: bool = True,
) -> str:
    if data_source_mode_str == "direct":
        return "Direct Norgate"
    reason_label_map_dict = {
        "api_config_missing": "API config missing",
        "direct_norgate_mode": "Direct Norgate",
        "local_snapshot_ready": "Norgate data fresh",
        "no_enabled_releases": "No enabled releases",
        SNAPSHOT_STALE_FOR_CYCLE_REASON_CODE_STR: "Local data too old",
        "snapshot_root_missing": "Snapshot root missing",
        "sync_failed": "Sync failed",
        "sync_failure_cooldown": "Cooldown after failure",
        "sync_waiting_for_newer_snapshot": "Waiting for provider data",
        "sync_lock_busy": "Sync lock busy",
        "sync_ready": "Norgate data fresh",
        "sync_started": "Sync running",
    }
    reason_label_str = reason_label_map_dict.get(str(reason_code_str or ""))
    if reason_label_str is not None and reason_code_str not in {"local_snapshot_ready", "sync_ready"}:
        return reason_label_str
    if not snapshot_fresh_for_cycle_bool:
        return "Local data too old"
    if status_str == "ready" and build_gate_reason_code_str == "snapshot_window_expired":
        return "Snapshot window expired"
    if status_str == "ready" and build_gate_reason_code_str in SNAPSHOT_STALE_GATE_REASON_SET:
        return "Build gate waiting"
    if reason_label_str is not None:
        return reason_label_str
    status_label_map_dict = {
        "failed": "Sync failed",
        "local_snapshot_only": "Local snapshot only",
        "ready": "Norgate data fresh",
        "waiting": "Waiting for snapshot",
    }
    return status_label_map_dict.get(status_str, "Snapshot status unknown")


def _string_list_from_status_file(
    status_file_dict: dict[str, Any],
    key_str: str,
    fallback_list: list[str] | None = None,
) -> list[str]:
    raw_value_obj = status_file_dict.get(key_str)
    if isinstance(raw_value_obj, list):
        return [str(item_obj) for item_obj in raw_value_obj]
    return list(fallback_list or [])


def _string_dict_from_status_file(
    status_file_dict: dict[str, Any],
    key_str: str,
) -> dict[str, str]:
    raw_value_obj = status_file_dict.get(key_str)
    if not isinstance(raw_value_obj, dict):
        return {}
    return {
        str(key_obj): str(value_obj)
        for key_obj, value_obj in raw_value_obj.items()
        if value_obj is not None
    }


def _profile_status_dict_list(
    *,
    profile_list: list[str],
    snapshot_date_by_profile_dict: dict[str, str],
    manifest_hash_by_profile_dict: dict[str, str],
    error_by_profile_dict: dict[str, str],
) -> list[dict[str, Any]]:
    return [
        {
            "profile_str": profile_str,
            "snapshot_date_str": snapshot_date_by_profile_dict.get(profile_str),
            "manifest_hash_str": manifest_hash_by_profile_dict.get(profile_str),
            "manifest_hash_prefix_str": str(manifest_hash_by_profile_dict.get(profile_str) or "")[:12] or None,
            "error_str": error_by_profile_dict.get(profile_str),
        }
        for profile_str in profile_list
    ]


def _release_gate_status_dict_list(
    *,
    release_id_list: list[str],
    pod_id_list: list[str],
    gate_reason_by_release_id_dict: dict[str, str],
    fallback_release_obj: LiveRelease,
    fallback_gate_reason_code_str: str | None,
) -> list[dict[str, Any]]:
    if len(release_id_list) == 0:
        release_id_list = [fallback_release_obj.release_id_str]
        pod_id_list = [fallback_release_obj.pod_id_str]
        if fallback_gate_reason_code_str is not None:
            gate_reason_by_release_id_dict = {
                fallback_release_obj.release_id_str: fallback_gate_reason_code_str
            }
    return [
        {
            "release_id_str": release_id_str,
            "pod_id_str": pod_id_list[index_int] if index_int < len(pod_id_list) else None,
            "gate_reason_code_str": gate_reason_by_release_id_dict.get(release_id_str)
            or (
                fallback_gate_reason_code_str
                if release_id_str == fallback_release_obj.release_id_str
                else None
            ),
        }
        for index_int, release_id_str in enumerate(release_id_list)
    ]


def build_norgate_snapshot_status_dict(
    release_obj: LiveRelease,
    as_of_ts: datetime,
) -> dict[str, Any]:
    if not is_snapshot_mode_enabled_bool():
        return {
            "data_source_mode_str": "direct",
            "status_str": "direct",
            "severity_str": "green",
            "sync_stage_label_str": "Direct Norgate",
            "profile_str": release_obj.data_profile_str,
            "snapshot_date_str": None,
            "reason_code_str": "direct_norgate_mode",
            "required_profile_list": [],
            "snapshot_date_by_profile_dict": {},
            "manifest_hash_by_profile_dict": {},
            "error_by_profile_dict": {},
            "required_snapshot_date_by_release_dict": {},
            "stale_alert_deadline_by_release_dict": {},
            "minimum_required_snapshot_date_by_profile_dict": {},
            "stale_profile_list": [],
            "snapshot_fresh_for_cycle_bool": True,
            "snapshot_stale_past_alert_deadline_bool": False,
            "gate_reason_by_release_id_dict": {},
            "profile_status_dict_list": [],
            "release_gate_status_dict_list": [],
            "last_sync_utc_str": None,
            "last_attempt_utc_str": None,
            "last_error_str": None,
            "operator_message_str": "Direct Norgate mode is active.",
            "operator_action_str": "No action needed.",
            "build_gate_reason_code_str": None,
            "status_file_path_str": None,
            "snapshot_mode_env_str": os.getenv(ALPHA_USE_NORGATE_SNAPSHOT_ENV_STR, ""),
        }

    root_path_obj = _snapshot_root_path_obj_or_none()
    status_file_dict = read_client_sync_status_dict(root_path_obj)
    status_profile_list = [
        str(profile_obj)
        for profile_obj in status_file_dict.get("required_profile_list", [])
    ]
    if status_profile_list and release_obj.data_profile_str not in status_profile_list:
        status_file_dict = {}
    snapshot_date_str = None
    manifest_hash_str = None
    local_error_str = None
    if root_path_obj is None:
        local_error_str = f"{NORGATE_SNAPSHOT_ROOT_ENV_STR} is not set."
    else:
        current_local_detail_dict = _with_cycle_freshness_detail_dict(
            _local_snapshot_detail_dict([release_obj.data_profile_str]),
            [release_obj],
            as_of_ts,
        )
        snapshot_date_str = (
            current_local_detail_dict.get("snapshot_date_by_profile_dict", {}).get(
                release_obj.data_profile_str
            )
        )
        manifest_hash_str = (
            current_local_detail_dict.get("manifest_hash_by_profile_dict", {}).get(
                release_obj.data_profile_str
            )
        )
        local_error_str = (
            current_local_detail_dict.get("error_by_profile_dict", {}).get(
                release_obj.data_profile_str
            )
        )

    build_gate_reason_code_str = None
    signal_clock_str = scheduler_utils.normalize_signal_clock_str(release_obj.signal_clock_str)
    if signal_clock_str in {"eod_snapshot_ready", "month_end_snapshot_ready"}:
        try:
            build_gate_reason_code_str = str(
                scheduler_utils.evaluate_build_gate_dict(release_obj, as_of_ts).get("reason_code_str")
            )
        except Exception as exc:
            build_gate_reason_code_str = "snapshot_status_error"
            if local_error_str is None:
                local_error_str = str(exc)

    current_snapshot_fresh_for_cycle_bool = True
    current_snapshot_stale_past_alert_deadline_bool = False
    current_required_snapshot_date_by_release_dict: dict[str, str] = {}
    current_stale_alert_deadline_by_release_dict: dict[str, str] = {}
    current_minimum_required_snapshot_date_by_profile_dict: dict[str, str] = {}
    current_stale_profile_list: list[str] = []
    if root_path_obj is not None:
        current_snapshot_fresh_for_cycle_bool = bool(
            current_local_detail_dict.get("snapshot_fresh_for_cycle_bool", True)
        )
        current_snapshot_stale_past_alert_deadline_bool = bool(
            current_local_detail_dict.get("snapshot_stale_past_alert_deadline_bool", False)
        )
        current_required_snapshot_date_by_release_dict = dict(
            current_local_detail_dict.get("required_snapshot_date_by_release_dict", {})
        )
        current_stale_alert_deadline_by_release_dict = dict(
            current_local_detail_dict.get("stale_alert_deadline_by_release_dict", {})
        )
        current_minimum_required_snapshot_date_by_profile_dict = dict(
            current_local_detail_dict.get("minimum_required_snapshot_date_by_profile_dict", {})
        )
        current_stale_profile_list = [
            str(profile_obj)
            for profile_obj in current_local_detail_dict.get("stale_profile_list", [])
        ]

    status_str = str(status_file_dict.get("status_str") or ("ready" if snapshot_date_str else "waiting"))
    reason_code_str = str(status_file_dict.get("reason_code_str") or "") or None
    last_error_str = str(status_file_dict.get("error_str") or local_error_str or "") or None
    if snapshot_date_str is not None and current_snapshot_fresh_for_cycle_bool:
        status_str = "ready"
        if reason_code_str in {
            None,
            "api_config_missing",
            SNAPSHOT_STALE_FOR_CYCLE_REASON_CODE_STR,
            "sync_failed",
            "sync_failure_cooldown",
            "sync_lock_busy",
            "sync_started",
        }:
            reason_code_str = "local_snapshot_ready"
        if local_error_str is None:
            last_error_str = None
    profile_list = _string_list_from_status_file(
        status_file_dict,
        "required_profile_list",
        fallback_list=[release_obj.data_profile_str],
    )
    if release_obj.data_profile_str not in profile_list:
        profile_list.append(release_obj.data_profile_str)
    snapshot_date_by_profile_dict = _string_dict_from_status_file(
        status_file_dict,
        "snapshot_date_by_profile_dict",
    )
    manifest_hash_by_profile_dict = _string_dict_from_status_file(
        status_file_dict,
        "manifest_hash_by_profile_dict",
    )
    error_by_profile_dict = _string_dict_from_status_file(
        status_file_dict,
        "error_by_profile_dict",
    )
    if snapshot_date_str is not None:
        snapshot_date_by_profile_dict[release_obj.data_profile_str] = snapshot_date_str
    if manifest_hash_str is not None:
        manifest_hash_by_profile_dict[release_obj.data_profile_str] = manifest_hash_str
    if local_error_str is not None:
        error_by_profile_dict[release_obj.data_profile_str] = local_error_str
    else:
        error_by_profile_dict.pop(release_obj.data_profile_str, None)
    gate_reason_by_release_id_dict = _string_dict_from_status_file(
        status_file_dict,
        "gate_reason_by_release_id_dict",
    )
    status_dict = {
        "data_source_mode_str": "snapshot",
        "status_str": status_str,
        "severity_str": _status_severity_str(
            status_str,
            build_gate_reason_code_str,
            snapshot_fresh_for_cycle_bool=current_snapshot_fresh_for_cycle_bool,
            snapshot_stale_past_alert_deadline_bool=(
                current_snapshot_stale_past_alert_deadline_bool
            ),
            reason_code_str=reason_code_str,
        ),
        "sync_stage_label_str": _sync_stage_label_str(
            data_source_mode_str="snapshot",
            status_str=status_str,
            reason_code_str=reason_code_str,
            build_gate_reason_code_str=build_gate_reason_code_str,
            snapshot_fresh_for_cycle_bool=current_snapshot_fresh_for_cycle_bool,
        ),
        "profile_str": release_obj.data_profile_str,
        "snapshot_date_str": snapshot_date_str,
        "manifest_hash_str": manifest_hash_str,
        "reason_code_str": reason_code_str,
        "required_profile_list": profile_list,
        "snapshot_date_by_profile_dict": snapshot_date_by_profile_dict,
        "manifest_hash_by_profile_dict": manifest_hash_by_profile_dict,
        "error_by_profile_dict": error_by_profile_dict,
        "required_snapshot_date_by_release_dict": current_required_snapshot_date_by_release_dict,
        "stale_alert_deadline_by_release_dict": current_stale_alert_deadline_by_release_dict,
        "minimum_required_snapshot_date_by_profile_dict": (
            current_minimum_required_snapshot_date_by_profile_dict
        ),
        "stale_profile_list": current_stale_profile_list,
        "snapshot_fresh_for_cycle_bool": current_snapshot_fresh_for_cycle_bool,
        "snapshot_stale_past_alert_deadline_bool": (
            current_snapshot_stale_past_alert_deadline_bool
        ),
        "gate_reason_by_release_id_dict": gate_reason_by_release_id_dict,
        "profile_status_dict_list": _profile_status_dict_list(
            profile_list=profile_list,
            snapshot_date_by_profile_dict=snapshot_date_by_profile_dict,
            manifest_hash_by_profile_dict=manifest_hash_by_profile_dict,
            error_by_profile_dict=error_by_profile_dict,
        ),
        "release_gate_status_dict_list": _release_gate_status_dict_list(
            release_id_list=[release_obj.release_id_str],
            pod_id_list=[release_obj.pod_id_str],
            gate_reason_by_release_id_dict=gate_reason_by_release_id_dict,
            fallback_release_obj=release_obj,
            fallback_gate_reason_code_str=build_gate_reason_code_str,
        ),
        "last_sync_utc_str": status_file_dict.get("last_success_utc_str"),
        "last_attempt_utc_str": status_file_dict.get("last_attempt_utc_str"),
        "last_error_str": last_error_str,
        "build_gate_reason_code_str": build_gate_reason_code_str,
        "status_file_path_str": str(_status_path_obj(root_path_obj)) if root_path_obj is not None else None,
        "snapshot_mode_env_str": os.getenv(ALPHA_USE_NORGATE_SNAPSHOT_ENV_STR, ""),
    }
    status_dict["operator_message_str"] = _operator_message_str(status_dict)
    status_dict["operator_action_str"] = _operator_action_str(status_dict)
    return status_dict
