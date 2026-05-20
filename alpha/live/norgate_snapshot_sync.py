from __future__ import annotations

import json
import os
from datetime import UTC, datetime
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


def _needs_sync_bool(
    *,
    local_detail_dict: dict[str, Any],
    gate_reason_by_release_dict: dict[str, str],
) -> bool:
    if not bool(local_detail_dict.get("all_profiles_ready_bool")):
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
        "gate_reason_by_release_id_dict": dict(gate_reason_by_release_dict or {}),
        "release_id_list": [release_obj.release_id_str for release_obj in release_list],
        "pod_id_list": [release_obj.pod_id_str for release_obj in release_list],
        "reason_code_str": reason_code_str,
        "error_str": error_str,
    }
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
    local_detail_dict = _local_snapshot_detail_dict(profile_list)
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
        status_dict = _base_status_dict(
            status_str="waiting",
            release_list=release_list,
            profile_list=profile_list,
            local_detail_dict=local_detail_dict,
            gate_reason_by_release_dict=gate_reason_by_release_dict,
            error_str=str(previous_status_dict.get("error_str") or "Previous sync failed recently."),
            reason_code_str="sync_failure_cooldown",
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
        )
        local_detail_dict = _local_snapshot_detail_dict(profile_list)
        gate_reason_by_release_dict = _build_gate_reason_by_release_dict(release_list, as_of_ts)
        success_utc_str = _utc_now_str()
        status_dict = _base_status_dict(
            status_str="ready",
            release_list=release_list,
            profile_list=profile_list,
            local_detail_dict=local_detail_dict,
            gate_reason_by_release_dict=gate_reason_by_release_dict,
            error_str=None,
            reason_code_str="sync_ready",
            last_attempt_utc_str=started_status_dict["last_attempt_utc_str"],
            last_success_utc_str=success_utc_str,
        )
        status_dict["promoted_path_list"] = [str(path_obj) for path_obj in promoted_path_list]
        _write_client_sync_status(snapshot_root_path_obj, status_dict)
        _emit_sync_event(
            "norgate_snapshot_sync_ready",
            status_dict,
            log_path_str=log_path_str,
            print_operator_bool=print_operator_bool,
        )
        return status_dict
    except Exception as exc:
        local_detail_dict = _local_snapshot_detail_dict(profile_list)
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


def _status_severity_str(status_str: str, build_gate_reason_code_str: str | None) -> str:
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
) -> str:
    if data_source_mode_str == "direct":
        return "Direct Norgate"
    if status_str == "ready" and build_gate_reason_code_str == "snapshot_window_expired":
        return "Snapshot window expired"
    if status_str == "ready" and build_gate_reason_code_str in SNAPSHOT_STALE_GATE_REASON_SET:
        return "Build gate waiting"
    reason_label_map_dict = {
        "api_config_missing": "API config missing",
        "direct_norgate_mode": "Direct Norgate",
        "local_snapshot_ready": "Local snapshot ready",
        "no_enabled_releases": "No enabled releases",
        "snapshot_root_missing": "Snapshot root missing",
        "sync_failed": "Sync failed",
        "sync_failure_cooldown": "Cooldown after failure",
        "sync_lock_busy": "Sync lock busy",
        "sync_ready": "Sync completed",
        "sync_started": "Sync running",
    }
    reason_label_str = reason_label_map_dict.get(str(reason_code_str or ""))
    if reason_label_str is not None:
        return reason_label_str
    status_label_map_dict = {
        "failed": "Sync failed",
        "local_snapshot_only": "Local snapshot only",
        "ready": "Snapshot ready",
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
            "gate_reason_by_release_id_dict": {},
            "profile_status_dict_list": [],
            "release_gate_status_dict_list": [],
            "last_sync_utc_str": None,
            "last_attempt_utc_str": None,
            "last_error_str": None,
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
        try:
            snapshot_manifest_obj = load_valid_snapshot_manifest(release_obj.data_profile_str)
            snapshot_date_str = snapshot_manifest_obj.snapshot_date_ts.date().isoformat()
            manifest_hash_str = snapshot_manifest_obj.manifest_hash_str
        except Exception as exc:
            local_error_str = str(exc)

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

    status_str = str(status_file_dict.get("status_str") or ("ready" if snapshot_date_str else "waiting"))
    reason_code_str = str(status_file_dict.get("reason_code_str") or "") or None
    last_error_str = str(status_file_dict.get("error_str") or local_error_str or "") or None
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
        snapshot_date_by_profile_dict.setdefault(release_obj.data_profile_str, snapshot_date_str)
    if manifest_hash_str is not None:
        manifest_hash_by_profile_dict.setdefault(release_obj.data_profile_str, manifest_hash_str)
    if local_error_str is not None and release_obj.data_profile_str not in error_by_profile_dict:
        error_by_profile_dict[release_obj.data_profile_str] = local_error_str
    gate_reason_by_release_id_dict = _string_dict_from_status_file(
        status_file_dict,
        "gate_reason_by_release_id_dict",
    )
    return {
        "data_source_mode_str": "snapshot",
        "status_str": status_str,
        "severity_str": _status_severity_str(status_str, build_gate_reason_code_str),
        "sync_stage_label_str": _sync_stage_label_str(
            data_source_mode_str="snapshot",
            status_str=status_str,
            reason_code_str=reason_code_str,
            build_gate_reason_code_str=build_gate_reason_code_str,
        ),
        "profile_str": release_obj.data_profile_str,
        "snapshot_date_str": snapshot_date_str,
        "manifest_hash_str": manifest_hash_str,
        "reason_code_str": reason_code_str,
        "required_profile_list": profile_list,
        "snapshot_date_by_profile_dict": snapshot_date_by_profile_dict,
        "manifest_hash_by_profile_dict": manifest_hash_by_profile_dict,
        "error_by_profile_dict": error_by_profile_dict,
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
