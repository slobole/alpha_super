from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
import json
from pathlib import Path
import secrets
import sqlite3
import threading
import traceback
from typing import Any, Callable
from urllib.parse import quote
import uuid

import yaml

from alpha.data import LIVE_FRED_STALE_WARNING_BUSINESS_DAYS_INT
from alpha.live import logging_utils, runner, scheduler_utils
from alpha.live.models import LiveRelease
from alpha.live.norgate_snapshot_sync import build_norgate_snapshot_status_dict
from alpha.live.release_manifest import load_release_list
from alpha.live.state_store_v2 import LiveStateStore
from scripts.norgate_config_env import load_config_env_file  # noqa: F401 - kept for backwards compatibility with tests/imports.


DEFAULT_DASHBOARD_HOST_STR = "127.0.0.1"
DEFAULT_DASHBOARD_PORT_INT = 8765
DEFAULT_RELEASES_ROOT_PATH_STR = str(Path(__file__).resolve().parent / "releases")
DEFAULT_CONFIG_PATH_STR = str(Path(__file__).resolve().parent / "dashboard_config.yaml")
DEFAULT_RESULTS_ROOT_PATH_STR = "results"
DEFAULT_EVENT_LOG_PATH_STR = str(Path(__file__).resolve().parent / "logs" / "live_events.jsonl")
DEFAULT_EVENT_LIMIT_INT = 80
DEFAULT_TRACE_EVENT_LIMIT_INT = 80
DEFAULT_EVENT_LOG_BACKUP_SCAN_COUNT_INT = 10
ALERT_SEVERITY_RANK_DICT = {"red": 0, "yellow": 1, "gray": 2, "green": 3}
COMBINED_BOOK_MODE_ORDER_LIST = ["live", "paper", "incubation"]
SAFE_INSPECT_COMMAND_NAME_LIST = [
    "status",
    "next_due",
    "show_decision_plan",
    "show_vplan",
    "compare_reference",
]
SAFE_ACTION_NAME_LIST = [
    "tick",
    "submit_vplan",
    "post_execution_reconcile",
    "eod_snapshot",
]
ACTION_TOKEN_HEADER_STR = "X-Alpha-Action-Token"
NORGATE_SNAPSHOT_GATED_SIGNAL_CLOCK_SET = {"eod_snapshot_ready", "month_end_snapshot_ready"}
NORGATE_CURRENT_DECISION_STATUS_SET = {"planned", "vplan_ready", "submitted"}


@dataclass(frozen=True)
class DashboardConfig:
    db_override_map_dict: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class DashboardPodTarget:
    release_obj: LiveRelease
    db_path_str: str
    db_override_bool: bool


@dataclass
class DashboardJob:
    job_id_str: str
    pod_id_str: str
    mode_str: str
    status_str: str
    created_timestamp_str: str
    action_name_str: str | None = None
    started_timestamp_str: str | None = None
    completed_timestamp_str: str | None = None
    result_dict: dict[str, Any] | None = None
    error_str: str | None = None
    traceback_str: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id_str": self.job_id_str,
            "pod_id_str": self.pod_id_str,
            "mode_str": self.mode_str,
            "status_str": self.status_str,
            "created_timestamp_str": self.created_timestamp_str,
            "action_name_str": self.action_name_str,
            "started_timestamp_str": self.started_timestamp_str,
            "completed_timestamp_str": self.completed_timestamp_str,
            "result_dict": self.result_dict,
            "error_str": self.error_str,
            "traceback_str": self.traceback_str,
        }


class DiffJobManager:
    def __init__(
        self,
        diff_runner_fn: Callable[[DashboardPodTarget, str, str, datetime], dict[str, Any]],
        pod_job_gate_obj: "DashboardPodJobGate | None" = None,
        max_job_count_int: int = 200,
    ):
        self._diff_runner_fn = diff_runner_fn
        self._job_map_dict: dict[str, DashboardJob] = {}
        self._pod_job_gate_obj = pod_job_gate_obj or DashboardPodJobGate()
        self._max_job_count_int = max_job_count_int
        self._lock_obj = threading.Lock()

    def start_job(
        self,
        pod_target_obj: DashboardPodTarget,
        releases_root_path_str: str,
        results_root_path_str: str,
    ) -> DashboardJob:
        now_ts = datetime.now(UTC)
        pod_id_str = pod_target_obj.release_obj.pod_id_str
        job_obj = DashboardJob(
            job_id_str=uuid.uuid4().hex,
            pod_id_str=pod_id_str,
            mode_str=pod_target_obj.release_obj.mode_str,
            status_str="queued",
            created_timestamp_str=now_ts.isoformat(),
            action_name_str="compare_reference",
        )
        self._pod_job_gate_obj.acquire(pod_id_str)
        with self._lock_obj:
            self._prune_completed_jobs_locked()
            self._job_map_dict[job_obj.job_id_str] = job_obj
        try:
            thread_obj = threading.Thread(
                target=self._run_job,
                args=(
                    job_obj.job_id_str,
                    pod_target_obj,
                    releases_root_path_str,
                    results_root_path_str,
                ),
                daemon=True,
            )
            thread_obj.start()
        except Exception:
            self._pod_job_gate_obj.release(pod_id_str)
            raise
        return job_obj

    def get_job_dict(self, job_id_str: str) -> dict[str, Any] | None:
        with self._lock_obj:
            job_obj = self._job_map_dict.get(job_id_str)
            return None if job_obj is None else job_obj.to_dict()

    def _prune_completed_jobs_locked(self) -> None:
        if len(self._job_map_dict) <= self._max_job_count_int:
            return
        removable_job_id_list = [
            job_id_str
            for job_id_str, job_obj in self._job_map_dict.items()
            if job_obj.status_str not in {"queued", "running"}
        ]
        overflow_count_int = len(self._job_map_dict) - self._max_job_count_int
        for job_id_str in removable_job_id_list[:overflow_count_int]:
            self._job_map_dict.pop(job_id_str, None)

    def _run_job(
        self,
        job_id_str: str,
        pod_target_obj: DashboardPodTarget,
        releases_root_path_str: str,
        results_root_path_str: str,
    ) -> None:
        with self._lock_obj:
            job_obj = self._job_map_dict[job_id_str]
            job_obj.status_str = "running"
            job_obj.started_timestamp_str = datetime.now(UTC).isoformat()
        try:
            result_dict = self._diff_runner_fn(
                pod_target_obj,
                releases_root_path_str,
                results_root_path_str,
                datetime.now(UTC),
            )
        except Exception as exception_obj:  # pragma: no cover - traceback text is environment-specific.
            with self._lock_obj:
                job_obj = self._job_map_dict[job_id_str]
                job_obj.status_str = "failed"
                job_obj.completed_timestamp_str = datetime.now(UTC).isoformat()
                job_obj.error_str = str(exception_obj)
                job_obj.traceback_str = traceback.format_exc()
            return
        else:
            with self._lock_obj:
                job_obj = self._job_map_dict[job_id_str]
                job_obj.status_str = "succeeded"
                job_obj.completed_timestamp_str = datetime.now(UTC).isoformat()
                job_obj.result_dict = result_dict
        finally:
            self._pod_job_gate_obj.release(pod_target_obj.release_obj.pod_id_str)


class DashboardActionJobManager:
    def __init__(
        self,
        action_runner_fn: Callable[[DashboardPodTarget, str, str, str, str, datetime], dict[str, Any]],
        pod_job_gate_obj: "DashboardPodJobGate | None" = None,
        max_job_count_int: int = 200,
    ):
        self._action_runner_fn = action_runner_fn
        self._job_map_dict: dict[str, DashboardJob] = {}
        self._pod_job_gate_obj = pod_job_gate_obj or DashboardPodJobGate()
        self._max_job_count_int = max_job_count_int
        self._lock_obj = threading.Lock()

    def start_job(
        self,
        action_name_str: str,
        pod_target_obj: DashboardPodTarget,
        releases_root_path_str: str,
        results_root_path_str: str,
        event_log_path_str: str,
    ) -> DashboardJob:
        if action_name_str not in SAFE_ACTION_NAME_LIST:
            raise ValueError(f"Unsupported dashboard action '{action_name_str}'.")
        now_ts = datetime.now(UTC)
        pod_id_str = pod_target_obj.release_obj.pod_id_str
        job_obj = DashboardJob(
            job_id_str=uuid.uuid4().hex,
            pod_id_str=pod_id_str,
            mode_str=pod_target_obj.release_obj.mode_str,
            status_str="queued",
            created_timestamp_str=now_ts.isoformat(),
            action_name_str=action_name_str,
        )
        self._pod_job_gate_obj.acquire(pod_id_str)
        with self._lock_obj:
            self._prune_completed_jobs_locked()
            self._job_map_dict[job_obj.job_id_str] = job_obj
        try:
            thread_obj = threading.Thread(
                target=self._run_job,
                args=(
                    job_obj.job_id_str,
                    action_name_str,
                    pod_target_obj,
                    releases_root_path_str,
                    results_root_path_str,
                    event_log_path_str,
                ),
                daemon=True,
            )
            thread_obj.start()
        except Exception:
            self._pod_job_gate_obj.release(pod_id_str)
            raise
        return job_obj

    def get_job_dict(self, job_id_str: str) -> dict[str, Any] | None:
        with self._lock_obj:
            job_obj = self._job_map_dict.get(job_id_str)
            return None if job_obj is None else job_obj.to_dict()

    def _prune_completed_jobs_locked(self) -> None:
        if len(self._job_map_dict) <= self._max_job_count_int:
            return
        removable_job_id_list = [
            job_id_str
            for job_id_str, job_obj in self._job_map_dict.items()
            if job_obj.status_str not in {"queued", "running"}
        ]
        overflow_count_int = len(self._job_map_dict) - self._max_job_count_int
        for job_id_str in removable_job_id_list[:overflow_count_int]:
            self._job_map_dict.pop(job_id_str, None)

    def _run_job(
        self,
        job_id_str: str,
        action_name_str: str,
        pod_target_obj: DashboardPodTarget,
        releases_root_path_str: str,
        results_root_path_str: str,
        event_log_path_str: str,
    ) -> None:
        with self._lock_obj:
            job_obj = self._job_map_dict[job_id_str]
            job_obj.status_str = "running"
            job_obj.started_timestamp_str = datetime.now(UTC).isoformat()
        try:
            result_dict = self._action_runner_fn(
                pod_target_obj,
                action_name_str,
                releases_root_path_str,
                results_root_path_str,
                event_log_path_str,
                datetime.now(UTC),
            )
        except Exception as exception_obj:  # pragma: no cover - traceback text is environment-specific.
            with self._lock_obj:
                job_obj = self._job_map_dict[job_id_str]
                job_obj.status_str = "failed"
                job_obj.completed_timestamp_str = datetime.now(UTC).isoformat()
                job_obj.error_str = str(exception_obj)
                job_obj.traceback_str = traceback.format_exc()
            return
        else:
            with self._lock_obj:
                job_obj = self._job_map_dict[job_id_str]
                job_obj.status_str = "succeeded"
                job_obj.completed_timestamp_str = datetime.now(UTC).isoformat()
                job_obj.result_dict = result_dict
        finally:
            self._pod_job_gate_obj.release(pod_target_obj.release_obj.pod_id_str)


class DashboardActionInFlightError(RuntimeError):
    pass


class DashboardPodJobGate:
    def __init__(self):
        self._active_pod_id_set: set[str] = set()
        self._lock_obj = threading.Lock()

    def acquire(self, pod_id_str: str) -> None:
        with self._lock_obj:
            if pod_id_str in self._active_pod_id_set:
                raise DashboardActionInFlightError(
                    f"Dashboard job already running for pod_id_str '{pod_id_str}'."
                )
            self._active_pod_id_set.add(pod_id_str)

    def release(self, pod_id_str: str) -> None:
        with self._lock_obj:
            self._active_pod_id_set.discard(pod_id_str)


@dataclass
class DashboardApp:
    releases_root_path_str: str = DEFAULT_RELEASES_ROOT_PATH_STR
    config_path_str: str = DEFAULT_CONFIG_PATH_STR
    results_root_path_str: str = DEFAULT_RESULTS_ROOT_PATH_STR
    event_log_path_str: str = DEFAULT_EVENT_LOG_PATH_STR
    action_token_str: str = field(default_factory=lambda: secrets.token_urlsafe(32))
    pod_job_gate_obj: DashboardPodJobGate | None = None
    diff_job_manager_obj: DiffJobManager | None = None
    action_job_manager_obj: DashboardActionJobManager | None = None

    def __post_init__(self) -> None:
        if self.pod_job_gate_obj is None:
            self.pod_job_gate_obj = DashboardPodJobGate()
        if self.diff_job_manager_obj is None:
            self.diff_job_manager_obj = DiffJobManager(_run_reference_diff_for_pod, self.pod_job_gate_obj)
        if self.action_job_manager_obj is None:
            self.action_job_manager_obj = DashboardActionJobManager(
                _run_dashboard_action_for_pod,
                pod_job_gate_obj=self.pod_job_gate_obj,
            )

    def load_config_obj(self) -> DashboardConfig:
        return load_dashboard_config(self.config_path_str)

    def get_target_list(self) -> list[DashboardPodTarget]:
        config_obj = self.load_config_obj()
        release_list = load_release_list(self.releases_root_path_str)
        target_list = [
            DashboardPodTarget(
                release_obj=release_obj,
                db_path_str=resolve_db_path_for_release_str(release_obj, config_obj),
                db_override_bool=release_obj.pod_id_str in config_obj.db_override_map_dict,
            )
            for release_obj in release_list
            if release_obj.enabled_bool
        ]
        return sorted(
            target_list,
            key=lambda target_obj: (
                target_obj.release_obj.mode_str,
                target_obj.release_obj.pod_id_str,
            ),
        )

    def get_target_for_pod(self, pod_id_str: str) -> DashboardPodTarget | None:
        for target_obj in self.get_target_list():
            if target_obj.release_obj.pod_id_str == pod_id_str:
                return target_obj
        return None


def load_dashboard_config(config_path_str: str = DEFAULT_CONFIG_PATH_STR) -> DashboardConfig:
    config_path_obj = Path(config_path_str)
    if not config_path_obj.exists():
        return DashboardConfig()
    raw_config_obj = yaml.safe_load(config_path_obj.read_text(encoding="utf-8")) or {}
    raw_override_obj = raw_config_obj.get("db_overrides", {})
    if not isinstance(raw_override_obj, dict):
        raise ValueError("dashboard config field 'db_overrides' must be a mapping.")
    db_override_map_dict: dict[str, str] = {}
    for pod_id_str, override_obj in raw_override_obj.items():
        if isinstance(override_obj, str):
            db_override_map_dict[str(pod_id_str)] = override_obj
            continue
        if isinstance(override_obj, dict) and "db_path" in override_obj:
            db_override_map_dict[str(pod_id_str)] = str(override_obj["db_path"])
            continue
        raise ValueError(
            "dashboard config db_overrides values must be strings or mappings with db_path."
        )
    return DashboardConfig(db_override_map_dict=db_override_map_dict)


def resolve_db_path_for_release_str(
    release_obj: LiveRelease,
    config_obj: DashboardConfig | None = None,
) -> str:
    config_obj = DashboardConfig() if config_obj is None else config_obj
    override_path_str = config_obj.db_override_map_dict.get(release_obj.pod_id_str)
    if override_path_str is not None:
        return override_path_str
    return runner._resolve_db_path_for_mode_str(
        db_path_str=None,
        env_mode_str=release_obj.mode_str,
        pod_id_str=release_obj.pod_id_str,
    )


def build_dashboard_summary_dict(app_obj: DashboardApp, as_of_ts: datetime | None = None) -> dict[str, Any]:
    if as_of_ts is None:
        as_of_ts = datetime.now(UTC)
    target_list = app_obj.get_target_list()
    pod_row_dict_list = [
        build_pod_row_dict(
            pod_target_obj=target_obj,
            as_of_ts=as_of_ts,
            results_root_path_str=app_obj.results_root_path_str,
            event_log_path_str=app_obj.event_log_path_str,
        )
        for target_obj in target_list
    ]
    alert_dict_list = _build_alert_dict_list(pod_row_dict_list)
    return {
        "as_of_timestamp_str": as_of_ts.isoformat(),
        "pod_row_dict_list": pod_row_dict_list,
        "alert_dict_list": alert_dict_list,
        "alert_summary_dict": _build_alert_summary_dict(alert_dict_list),
        "mode_list": sorted({str(row_dict["mode_str"]) for row_dict in pod_row_dict_list}),
        "combined_book_dict": build_combined_book_dict(
            pod_target_list=target_list,
            as_of_ts=as_of_ts,
        ),
    }


def build_pod_detail_dict(
    app_obj: DashboardApp,
    pod_id_str: str,
    as_of_ts: datetime | None = None,
) -> dict[str, Any]:
    if as_of_ts is None:
        as_of_ts = datetime.now(UTC)
    target_obj = app_obj.get_target_for_pod(pod_id_str)
    if target_obj is None:
        raise KeyError(f"Unknown enabled pod_id_str '{pod_id_str}'.")
    row_dict = build_pod_row_dict(
        pod_target_obj=target_obj,
        as_of_ts=as_of_ts,
        results_root_path_str=app_obj.results_root_path_str,
        event_log_path_str=app_obj.event_log_path_str,
    )
    event_dict_list = load_recent_event_dict_list(
        log_path_str=app_obj.event_log_path_str,
        pod_id_str=pod_id_str,
        limit_int=DEFAULT_EVENT_LIMIT_INT,
    )
    detail_dict = {
        "pod_row_dict": row_dict,
        "required_action_dict": row_dict["required_action_dict"],
        "lifecycle_step_dict_list": row_dict["lifecycle_step_dict_list"],
        "data_freshness_dict": row_dict["data_freshness_dict"],
        "eod_snapshot_dict": row_dict["eod_snapshot_dict"],
        "rehearsal_status_dict": row_dict["rehearsal_status_dict"],
        "debug_story_dict": {},
        "pod_pnl_dict": _empty_pod_pnl_dict(),
        "latest_decision_plan_dict": None,
        "latest_vplan_dict": None,
        "latest_execution_report_dict": None,
        "event_dict_list": event_dict_list,
        "latest_diff_dict": find_latest_diff_artifact_dict(
            results_root_path_str=app_obj.results_root_path_str,
            mode_str=target_obj.release_obj.mode_str,
            pod_id_str=pod_id_str,
        ),
    }
    db_path_obj = Path(target_obj.db_path_str)
    if not db_path_obj.exists():
        return _finalize_pod_detail_debug_story_dict(detail_dict)
    with _connect_readonly_existing_db(db_path_obj) as connection_obj:
        latest_decision_plan_row_dict = _fetch_latest_decision_plan_row_dict(
            connection_obj,
            pod_id_str,
        )
        if latest_decision_plan_row_dict is not None:
            latest_decision_vplan_row_dict = _fetch_latest_vplan_for_decision_row_dict(
                connection_obj,
                int(latest_decision_plan_row_dict["decision_plan_id_int"]),
            )
            detail_dict["latest_decision_plan_dict"] = _build_decision_plan_detail_dict(
                latest_decision_plan_row_dict,
                latest_decision_vplan_row_dict,
            )
        detail_dict["pod_pnl_dict"] = _build_pod_pnl_dict(
            connection_obj=connection_obj,
            release_obj=target_obj.release_obj,
        )
        if not _table_exists_bool(connection_obj, "vplan"):
            return _finalize_pod_detail_debug_story_dict(detail_dict)
        latest_vplan_row_dict = _fetch_latest_vplan_row_dict(connection_obj, pod_id_str)
        if latest_vplan_row_dict is None:
            return _finalize_pod_detail_debug_story_dict(detail_dict)
        vplan_id_int = int(latest_vplan_row_dict["vplan_id_int"])
        latest_vplan_row_dict["vplan_row_dict_list"] = _fetch_all_dict_list(
            connection_obj,
            """
            SELECT *
            FROM vplan_row
            WHERE vplan_id_int = ?
            ORDER BY asset_str ASC, vplan_row_id_int ASC
            """,
            (vplan_id_int,),
        )
        latest_vplan_row_dict["broker_order_row_dict_list"] = _fetch_all_dict_list(
            connection_obj,
            """
            SELECT *
            FROM vplan_broker_order
            WHERE vplan_id_int = ?
            ORDER BY broker_order_record_id_int ASC
            """,
            (vplan_id_int,),
        )
        latest_vplan_row_dict["fill_row_dict_list"] = _fetch_all_dict_list(
            connection_obj,
            """
            SELECT *
            FROM vplan_fill
            WHERE vplan_id_int = ?
            ORDER BY fill_record_id_int ASC
            """,
            (vplan_id_int,),
        )
        latest_vplan_row_dict["broker_ack_row_dict_list"] = _fetch_all_dict_list(
            connection_obj,
            """
            SELECT *
            FROM vplan_broker_ack
            WHERE vplan_id_int = ?
            ORDER BY asset_str ASC, order_request_key_str ASC
            """,
            (vplan_id_int,),
        )
        broker_snapshot_row_dict = None
        if _table_exists_bool(connection_obj, "broker_snapshot_cache"):
            broker_snapshot_row_dict = _fetch_one_dict(
                connection_obj,
                "SELECT * FROM broker_snapshot_cache WHERE account_route_str = ?",
                (latest_vplan_row_dict["account_route_str"],),
            )
        broker_position_map_dict = {}
        if broker_snapshot_row_dict is not None:
            broker_position_map_dict = _json_map_dict(broker_snapshot_row_dict.get("position_json_str"))
        if not broker_position_map_dict:
            broker_position_map_dict = _json_map_dict(
                latest_vplan_row_dict.get("current_broker_position_json_str")
            )
        detail_dict["latest_vplan_dict"] = latest_vplan_row_dict
        detail_dict["latest_execution_report_dict"] = _build_execution_report_from_vplan_dict(
            latest_vplan_row_dict,
            broker_position_map_dict,
        )
    return _finalize_pod_detail_debug_story_dict(detail_dict)


def build_pod_row_dict(
    pod_target_obj: DashboardPodTarget,
    as_of_ts: datetime,
    results_root_path_str: str,
    event_log_path_str: str | None = None,
) -> dict[str, Any]:
    release_obj = pod_target_obj.release_obj
    db_path_obj = Path(pod_target_obj.db_path_str)
    latest_diff_dict = find_latest_diff_artifact_dict(
        results_root_path_str=results_root_path_str,
        mode_str=release_obj.mode_str,
        pod_id_str=release_obj.pod_id_str,
    )
    latest_event_timestamp_str = _latest_event_timestamp_str(
        event_log_path_str,
        release_obj.pod_id_str,
    )
    norgate_snapshot_status_dict = build_norgate_snapshot_status_dict(release_obj, as_of_ts)
    base_row_dict = {
        "release_id_str": release_obj.release_id_str,
        "user_id_str": release_obj.user_id_str,
        "pod_id_str": release_obj.pod_id_str,
        "mode_str": release_obj.mode_str,
        "account_route_str": release_obj.account_route_str,
        "strategy_import_str": release_obj.strategy_import_str,
        "signal_clock_str": release_obj.signal_clock_str,
        "session_calendar_id_str": release_obj.session_calendar_id_str,
        "execution_policy_str": release_obj.execution_policy_str,
        "data_profile_str": release_obj.data_profile_str,
        "as_of_timestamp_str": as_of_ts.isoformat(),
        "auto_submit_enabled_bool": bool(release_obj.auto_submit_enabled_bool),
        "db_path_str": pod_target_obj.db_path_str,
        "db_exists_bool": db_path_obj.exists(),
        "db_override_bool": pod_target_obj.db_override_bool,
        "db_status_str": "ok" if db_path_obj.exists() else "missing",
        "latest_decision_plan_status_str": None,
        "latest_decision_plan_id_int": None,
        "latest_decision_release_id_str": None,
        "latest_decision_execution_policy_str": None,
        "latest_decision_signal_timestamp_str": None,
        "latest_decision_norgate_profile_str": None,
        "latest_decision_norgate_snapshot_date_str": None,
        "latest_decision_plan_submission_timestamp_str": None,
        "latest_decision_plan_target_execution_timestamp_str": None,
        "latest_vplan_status_str": None,
        "latest_vplan_id_int": None,
        "latest_vplan_decision_plan_id_int": None,
        "latest_vplan_is_for_latest_decision_bool": None,
        "latest_vplan_cycle_role_str": "none",
        "latest_vplan_submission_timestamp_str": None,
        "latest_vplan_target_execution_timestamp_str": None,
        "latest_submit_ack_status_str": None,
        "latest_broker_snapshot_timestamp_str": None,
        "latest_live_reference_snapshot_timestamp_str": None,
        "latest_live_reference_source_str": None,
        "latest_open_price_source_str": None,
        "latest_pod_state_timestamp_str": None,
        "latest_pod_state_stage_str": None,
        "latest_pod_state_source_str": None,
        "latest_event_timestamp_str": latest_event_timestamp_str,
        "cash_float": None,
        "equity_float": None,
        "position_count_int": 0,
        "position_exposure_dict_list": [],
        "position_unpriced_count_int": 0,
        "broker_order_count_int": 0,
        "broker_ack_count_int": 0,
        "fill_count_int": 0,
        "completed_rehearsal_cycle_count_int": 0,
        "next_action_str": "no_db",
        "reason_code_str": "db_missing",
        "warning_count_int": 0,
        "missing_ack_count_int": 0,
        "exception_count_int": 0,
        "latest_reconciliation_status_str": None,
        "latest_reconciliation_timestamp_str": None,
        "latest_diff_status_str": str(latest_diff_dict.get("status_str", "not_run")),
        "latest_diff_timestamp_str": latest_diff_dict.get("artifact_timestamp_str"),
        "latest_diff_equity_tracking_error_float": latest_diff_dict.get(
            "equity_tracking_error_float"
        ),
        "latest_diff_open_issue_count_int": latest_diff_dict.get("open_issue_count_int"),
        "latest_diff_artifact_url_str": latest_diff_dict.get("html_url_str"),
        "strategy_family_str": None,
        "dtb3_download_status_str": None,
        "dtb3_latest_observation_date_str": None,
        "dtb3_freshness_business_days_int": None,
        "dtb3_source_name_str": None,
        "dtb3_used_cache_bool": None,
        "dtb3_policy_status_str": None,
        "dtb3_warning_bool": None,
        "dtb3_warn_after_business_days_int": None,
        "norgate_snapshot_status_dict": norgate_snapshot_status_dict,
        "eod_snapshot_dict": _empty_eod_snapshot_dict(),
        "health_str": "gray",
    }
    if not db_path_obj.exists():
        return _finalize_operator_fields_dict(base_row_dict)

    try:
        with _connect_readonly_existing_db(db_path_obj) as connection_obj:
            if not _table_exists_bool(connection_obj, "decision_plan"):
                base_row_dict["db_status_str"] = "empty"
                base_row_dict["reason_code_str"] = "db_empty"
                return _finalize_operator_fields_dict(base_row_dict)
            latest_decision_plan_row_dict = _fetch_latest_decision_plan_row_dict(
                connection_obj,
                release_obj.pod_id_str,
            )
            latest_vplan_row_dict = _fetch_latest_vplan_row_dict(
                connection_obj,
                release_obj.pod_id_str,
            )
            pod_state_row_dict = _fetch_one_dict(
                connection_obj,
                "SELECT * FROM pod_state WHERE pod_id_str = ?",
                (release_obj.pod_id_str,),
            )
            broker_snapshot_row_dict = _fetch_one_dict(
                connection_obj,
                "SELECT * FROM broker_snapshot_cache WHERE account_route_str = ?",
                (release_obj.account_route_str,),
            )
            reconciliation_row_dict = _fetch_latest_reconciliation_row_dict(
                connection_obj,
                release_obj.pod_id_str,
            )
            base_row_dict["eod_snapshot_dict"] = _build_eod_snapshot_dict(
                connection_obj=connection_obj,
                release_obj=release_obj,
                latest_vplan_row_dict=latest_vplan_row_dict,
                as_of_ts=as_of_ts,
            )
            if latest_vplan_row_dict is not None:
                vplan_id_int = int(latest_vplan_row_dict["vplan_id_int"])
                base_row_dict["broker_order_count_int"] = _fetch_count_int(
                    connection_obj,
                    table_name_str="vplan_broker_order",
                    where_sql_str="vplan_id_int = ?",
                    param_tuple=(vplan_id_int,),
                )
                base_row_dict["broker_ack_count_int"] = _fetch_count_int(
                    connection_obj,
                    table_name_str="vplan_broker_ack",
                    where_sql_str="vplan_id_int = ?",
                    param_tuple=(vplan_id_int,),
                )
                base_row_dict["fill_count_int"] = _fetch_count_int(
                    connection_obj,
                    table_name_str="vplan_fill",
                    where_sql_str="vplan_id_int = ?",
                    param_tuple=(vplan_id_int,),
                )
                base_row_dict["latest_open_price_source_str"] = (
                    _fetch_latest_open_price_source_str(connection_obj, vplan_id_int)
                )
                base_row_dict["completed_rehearsal_cycle_count_int"] = (
                    _fetch_completed_rehearsal_cycle_count_int(
                        connection_obj,
                        release_obj.pod_id_str,
                    )
                )
    except sqlite3.DatabaseError as exception_obj:
        base_row_dict["db_status_str"] = "error"
        base_row_dict["reason_code_str"] = "db_error"
        base_row_dict["error_str"] = str(exception_obj)
        base_row_dict["health_str"] = "red"
        return _finalize_operator_fields_dict(base_row_dict)

    if latest_decision_plan_row_dict is not None:
        base_row_dict["latest_decision_plan_id_int"] = latest_decision_plan_row_dict.get(
            "decision_plan_id_int"
        )
        base_row_dict["latest_decision_plan_status_str"] = latest_decision_plan_row_dict.get(
            "status_str"
        )
        base_row_dict["latest_decision_release_id_str"] = latest_decision_plan_row_dict.get(
            "release_id_str"
        )
        base_row_dict["latest_decision_execution_policy_str"] = (
            latest_decision_plan_row_dict.get("execution_policy_str")
        )
        base_row_dict["latest_decision_signal_timestamp_str"] = (
            latest_decision_plan_row_dict.get("signal_timestamp_str")
        )
        base_row_dict["latest_decision_plan_submission_timestamp_str"] = (
            latest_decision_plan_row_dict.get("submission_timestamp_str")
        )
        base_row_dict["latest_decision_plan_target_execution_timestamp_str"] = (
            latest_decision_plan_row_dict.get("target_execution_timestamp_str")
        )
        metadata_dict = _json_map_dict(
            latest_decision_plan_row_dict.get("snapshot_metadata_json_str")
        )
        base_row_dict["latest_decision_norgate_profile_str"] = (
            metadata_dict.get("norgate_data_profile_str")
            or metadata_dict.get("data_profile_str")
        )
        base_row_dict["latest_decision_norgate_snapshot_date_str"] = (
            metadata_dict.get("norgate_snapshot_date_str")
            or metadata_dict.get("snapshot_date_str")
        )
        base_row_dict["strategy_family_str"] = metadata_dict.get("strategy_family_str")
        base_row_dict["dtb3_download_status_str"] = metadata_dict.get(
            "dtb3_download_status_str"
        )
        base_row_dict["dtb3_latest_observation_date_str"] = metadata_dict.get(
            "dtb3_latest_observation_date_str"
        )
        base_row_dict["dtb3_freshness_business_days_int"] = metadata_dict.get(
            "dtb3_freshness_business_days_int"
        )
        base_row_dict["dtb3_source_name_str"] = metadata_dict.get("dtb3_source_name_str")
        base_row_dict["dtb3_used_cache_bool"] = metadata_dict.get("dtb3_used_cache_bool")
        base_row_dict["dtb3_policy_status_str"] = metadata_dict.get("dtb3_policy_status_str")
        base_row_dict["dtb3_warning_bool"] = metadata_dict.get("dtb3_warning_bool")
        base_row_dict["dtb3_warn_after_business_days_int"] = metadata_dict.get(
            "dtb3_warn_after_business_days_int"
        )
    if latest_vplan_row_dict is not None:
        base_row_dict["latest_vplan_status_str"] = latest_vplan_row_dict.get("status_str")
        base_row_dict["latest_vplan_id_int"] = latest_vplan_row_dict.get("vplan_id_int")
        base_row_dict["latest_vplan_decision_plan_id_int"] = latest_vplan_row_dict.get(
            "decision_plan_id_int"
        )
        base_row_dict["latest_vplan_submission_timestamp_str"] = latest_vplan_row_dict.get(
            "submission_timestamp_str"
        )
        base_row_dict["latest_vplan_target_execution_timestamp_str"] = latest_vplan_row_dict.get(
            "target_execution_timestamp_str"
        )
        base_row_dict["latest_submit_ack_status_str"] = latest_vplan_row_dict.get(
            "submit_ack_status_str"
        )
        base_row_dict["latest_live_reference_snapshot_timestamp_str"] = (
            latest_vplan_row_dict.get("live_reference_snapshot_timestamp_str")
        )
        base_row_dict["latest_live_reference_source_str"] = _compact_source_label_str(
            list(
                _json_map_dict(
                    latest_vplan_row_dict.get("live_reference_source_json_str")
                ).values()
            )
            or [latest_vplan_row_dict.get("live_price_source_str")]
        )
        base_row_dict["missing_ack_count_int"] = int(
            latest_vplan_row_dict.get("missing_ack_count_int") or 0
        )
    # Per-asset prices come from the latest VPlan reference snapshot; used to
    # value current positions for the cross-pod exposure view (no extra reads).
    latest_reference_price_map_dict = (
        _json_map_dict(latest_vplan_row_dict.get("live_reference_price_json_str"))
        if latest_vplan_row_dict is not None
        else {}
    )
    if pod_state_row_dict is not None:
        position_map_dict = _json_map_dict(pod_state_row_dict.get("position_json_str"))
        base_row_dict["position_count_int"] = len(
            [
                amount_float
                for amount_float in position_map_dict.values()
                if abs(float(amount_float)) > 1e-9
            ]
        )
        _attach_position_exposure_fields(
            base_row_dict, position_map_dict, latest_reference_price_map_dict
        )
        base_row_dict["cash_float"] = pod_state_row_dict.get("cash_float")
        base_row_dict["equity_float"] = pod_state_row_dict.get("total_value_float")
        base_row_dict["latest_pod_state_timestamp_str"] = pod_state_row_dict.get(
            "updated_timestamp_str"
        )
        base_row_dict["latest_pod_state_stage_str"] = pod_state_row_dict.get(
            "snapshot_stage_str"
        )
        base_row_dict["latest_pod_state_source_str"] = pod_state_row_dict.get(
            "snapshot_source_str"
        )
    elif broker_snapshot_row_dict is not None:
        broker_position_map_dict = _json_map_dict(
            broker_snapshot_row_dict.get("position_json_str")
        )
        base_row_dict["position_count_int"] = len(
            [
                amount_float
                for amount_float in broker_position_map_dict.values()
                if abs(float(amount_float)) > 1e-9
            ]
        )
        _attach_position_exposure_fields(
            base_row_dict, broker_position_map_dict, latest_reference_price_map_dict
        )
        base_row_dict["cash_float"] = broker_snapshot_row_dict.get("cash_float")
        base_row_dict["equity_float"] = broker_snapshot_row_dict.get("net_liq_float")

    if broker_snapshot_row_dict is not None:
        base_row_dict["latest_broker_snapshot_timestamp_str"] = broker_snapshot_row_dict.get(
            "snapshot_timestamp_str"
        )
    if reconciliation_row_dict is not None:
        base_row_dict["latest_reconciliation_status_str"] = reconciliation_row_dict.get(
            "status_str"
        )
        base_row_dict["latest_reconciliation_timestamp_str"] = reconciliation_row_dict.get(
            "created_timestamp_str"
        )
        if reconciliation_row_dict.get("status_str") != "passed":
            base_row_dict["exception_count_int"] = 1

    next_action_dict = _derive_next_action_dict(
        release_obj=release_obj,
        latest_decision_plan_row_dict=latest_decision_plan_row_dict,
        latest_vplan_row_dict=latest_vplan_row_dict,
        as_of_ts=as_of_ts,
    )
    base_row_dict.update(next_action_dict)
    base_row_dict["warning_count_int"] = int(base_row_dict["missing_ack_count_int"]) + int(
        base_row_dict["exception_count_int"]
    )
    base_row_dict["health_str"] = _resolve_health_str(base_row_dict)
    return _finalize_operator_fields_dict(base_row_dict)


def find_latest_diff_artifact_dict(
    results_root_path_str: str,
    mode_str: str,
    pod_id_str: str,
) -> dict[str, Any]:
    artifact_root_path_obj = (
        Path(results_root_path_str)
        / "live_reference_compare"
        / mode_str
        / pod_id_str
    )
    if not artifact_root_path_obj.exists():
        return {"status_str": "not_run"}
    candidate_summary_path_list = sorted(
        artifact_root_path_obj.glob("*/summary.json"),
        key=lambda path_obj: (
            path_obj.stat().st_mtime,
            str(path_obj.parent.name),
        ),
        reverse=True,
    )
    if len(candidate_summary_path_list) == 0:
        return {"status_str": "not_run"}
    summary_path_obj = candidate_summary_path_list[0]
    try:
        summary_dict = json.loads(summary_path_obj.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exception_obj:
        return {
            "status_str": "error",
            "error_str": str(exception_obj),
            "summary_path_str": str(summary_path_obj),
        }
    artifact_dir_path_obj = summary_path_obj.parent
    result_dict = dict(summary_dict)
    result_dict["artifact_timestamp_str"] = artifact_dir_path_obj.name
    result_dict["artifact_dir_path_str"] = str(artifact_dir_path_obj)
    result_dict["summary_json_path_str"] = str(summary_path_obj)
    for filename_str, key_str in (
        ("index.html", "html_url_str"),
        ("summary.json", "summary_url_str"),
        ("equity_compare.png", "equity_png_url_str"),
        ("tracking_error.png", "tracking_png_url_str"),
    ):
        artifact_path_obj = artifact_dir_path_obj / filename_str
        if artifact_path_obj.exists():
            result_dict[key_str] = _artifact_url_str(
                results_root_path_str=results_root_path_str,
                artifact_path_obj=artifact_path_obj,
            )
    return result_dict


def load_recent_event_dict_list(
    log_path_str: str,
    pod_id_str: str,
    limit_int: int = DEFAULT_EVENT_LIMIT_INT,
) -> list[dict[str, Any]]:
    log_path_obj = Path(log_path_str)
    if not log_path_obj.exists():
        return []
    # Stream backwards across the rotated event log set, collecting up to
    # ``limit_int`` events that match the pod. Stops as soon as the cap is hit
    # or all candidate files are exhausted. Beats the previous full-file scan
    # by 10–20× on a large JSONL log.
    collected_dict_list: list[dict[str, Any]] = []
    for candidate_log_path_obj in reversed(_event_log_path_obj_list(log_path_obj)):
        for event_dict in _iter_event_dict_reverse(candidate_log_path_obj):
            if _event_matches_pod_bool(event_dict, pod_id_str):
                collected_dict_list.append(event_dict)
                if len(collected_dict_list) >= int(limit_int):
                    return list(reversed(collected_dict_list))
    return list(reversed(collected_dict_list))


def load_recent_trace_event_dict_list(
    pod_id_str: str,
    limit_int: int = DEFAULT_TRACE_EVENT_LIMIT_INT,
    trace_log_root_path_str: str = logging_utils.DEFAULT_POD_TRACE_LOG_ROOT_PATH_STR,
) -> list[dict[str, Any]]:
    """Read the most recent cycle's structured trace events for one pod.

    The per-cycle trace logger writes one folder per run under
    ``<trace_root>/<pod_id>/<run_id>/trace_events.jsonl``. This locates the
    newest cycle folder (by newest file mtime inside it), reverse-reads its
    trace file, and returns up to ``limit_int`` events oldest-first. Read-only
    and bounded — trace files can be multi-MB, so the reverse reader stops as
    soon as the cap is hit instead of scanning the whole file.
    """
    pod_trace_dir_path_obj = (
        Path(trace_log_root_path_str)
        / logging_utils._sanitize_path_part_str(pod_id_str)
    )
    if not pod_trace_dir_path_obj.is_dir():
        return []
    newest_run_folder_path_obj: Path | None = None
    newest_mtime_float = -1.0
    for child_path_obj in pod_trace_dir_path_obj.iterdir():
        if not child_path_obj.is_dir():
            continue
        run_mtime_float = logging_utils._newest_trace_run_folder_mtime_float(child_path_obj)
        if run_mtime_float is None:
            continue
        if run_mtime_float > newest_mtime_float:
            newest_mtime_float = run_mtime_float
            newest_run_folder_path_obj = child_path_obj
    if newest_run_folder_path_obj is None:
        return []
    trace_log_path_obj = newest_run_folder_path_obj / "trace_events.jsonl"
    if not trace_log_path_obj.exists():
        return []
    collected_dict_list: list[dict[str, Any]] = []
    for event_dict in _iter_event_dict_reverse(trace_log_path_obj):
        collected_dict_list.append(event_dict)
        if len(collected_dict_list) >= int(limit_int):
            break
    return list(reversed(collected_dict_list))


def _event_log_path_obj_list(log_path_obj: Path) -> list[Path]:
    candidate_log_path_obj_list: list[Path] = []
    for backup_index_int in range(DEFAULT_EVENT_LOG_BACKUP_SCAN_COUNT_INT, 0, -1):
        backup_log_path_obj = log_path_obj.with_name(f"{log_path_obj.name}.{backup_index_int}")
        if backup_log_path_obj.exists():
            candidate_log_path_obj_list.append(backup_log_path_obj)
    candidate_log_path_obj_list.append(log_path_obj)
    return candidate_log_path_obj_list


_EVENT_LOG_REVERSE_CHUNK_INT = 16 * 1024


def _iter_event_dict_reverse(log_path_obj: Path):
    """Yield parsed event dicts from a JSONL file in reverse line order.

    Reads the file in chunks from the end, splits on newlines, and parses
    only complete lines. Malformed JSON is skipped silently — matches the
    forward reader's behaviour. This is the hot path for
    ``_latest_event_timestamp_str`` and ``load_recent_event_dict_list``;
    the previous forward scan parsed every line in the file just to find
    the most recent matching event for a single pod.
    """
    try:
        file_obj = log_path_obj.open("rb")
    except OSError:
        return
    try:
        file_obj.seek(0, 2)
        position_int = file_obj.tell()
        leftover_bytes = b""
        while position_int > 0:
            read_size_int = min(_EVENT_LOG_REVERSE_CHUNK_INT, position_int)
            position_int -= read_size_int
            file_obj.seek(position_int)
            chunk_bytes = file_obj.read(read_size_int) + leftover_bytes
            line_bytes_list = chunk_bytes.split(b"\n")
            # The first slice of the next iteration must contain the (possibly
            # truncated) head of the chunk we just read; everything else is a
            # complete line.
            leftover_bytes = line_bytes_list[0] if position_int > 0 else b""
            complete_line_bytes_list = line_bytes_list[1:] if position_int > 0 else line_bytes_list
            for line_bytes in reversed(complete_line_bytes_list):
                stripped_line_bytes = line_bytes.strip()
                if not stripped_line_bytes:
                    continue
                try:
                    event_dict = json.loads(stripped_line_bytes.decode("utf-8"))
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue
                if isinstance(event_dict, dict):
                    yield event_dict
    finally:
        file_obj.close()


def _latest_event_timestamp_str(log_path_str: str | None, pod_id_str: str) -> str | None:
    if log_path_str is None:
        return None
    log_path_obj = Path(log_path_str)
    if not log_path_obj.exists():
        return None
    for candidate_log_path_obj in reversed(_event_log_path_obj_list(log_path_obj)):
        for event_dict in _iter_event_dict_reverse(candidate_log_path_obj):
            if _event_matches_pod_bool(event_dict, pod_id_str):
                return _event_timestamp_str(event_dict)
    return None


def _event_timestamp_str(event_dict: dict[str, Any]) -> str | None:
    for key_str in (
        "timestamp_str",
        "event_timestamp_str",
        "as_of_timestamp_str",
        "created_timestamp_str",
    ):
        value_obj = event_dict.get(key_str)
        if value_obj is not None and str(value_obj) != "":
            return str(value_obj)
    return None


def _event_matches_pod_bool(event_dict: dict[str, Any], pod_id_str: str) -> bool:
    if str(event_dict.get("pod_id_str")) == pod_id_str:
        return True
    related_pod_id_list = event_dict.get("related_pod_id_list", [])
    if isinstance(related_pod_id_list, list) and pod_id_str in {
        str(item_obj) for item_obj in related_pod_id_list
    }:
        return True
    pod_id_list = event_dict.get("pod_id_list", [])
    return isinstance(pod_id_list, list) and pod_id_str in {
        str(item_obj) for item_obj in pod_id_list
    }


def _run_reference_diff_for_pod(
    pod_target_obj: DashboardPodTarget,
    releases_root_path_str: str,
    results_root_path_str: str,
    as_of_ts: datetime,
) -> dict[str, Any]:
    db_path_obj = Path(pod_target_obj.db_path_str)
    if not db_path_obj.exists():
        raise FileNotFoundError(f"POD DB does not exist: {pod_target_obj.db_path_str}")
    state_store_obj = LiveStateStore(str(db_path_obj))
    return runner.get_compare_reference_summary(
        state_store_obj=state_store_obj,
        as_of_ts=as_of_ts,
        releases_root_path_str=releases_root_path_str,
        env_mode_str=pod_target_obj.release_obj.mode_str,
        pod_id_str=pod_target_obj.release_obj.pod_id_str,
        html_output_bool=True,
        output_dir_str=results_root_path_str,
    )


def _run_dashboard_action_for_pod(
    pod_target_obj: DashboardPodTarget,
    action_name_str: str,
    releases_root_path_str: str,
    results_root_path_str: str,
    event_log_path_str: str,
    as_of_ts: datetime,
) -> dict[str, Any]:
    if action_name_str not in SAFE_ACTION_NAME_LIST:
        raise ValueError(f"Unsupported dashboard action '{action_name_str}'.")
    state_store_obj = LiveStateStore(pod_target_obj.db_path_str)
    job_run_id_int = state_store_obj.record_job_start(f"dashboard_{action_name_str}")
    parsed_args_obj = argparse.Namespace(
        command_name_str=action_name_str,
        releases_root_path_str=releases_root_path_str,
        env_mode_str=pod_target_obj.release_obj.mode_str,
        log_path_str=event_log_path_str,
        archive_root_path_str=None,
        json_output_bool=True,
        decision_plan_id_int=None,
        vplan_id_int=None,
        pod_id_str=pod_target_obj.release_obj.pod_id_str,
        broker_host_str=None,
        broker_port_int=None,
        broker_client_id_int=None,
        broker_timeout_seconds_float=None,
        reference_strategy_pickle_path_str=None,
        html_output_bool=False,
        output_dir_str=results_root_path_str,
    )
    try:
        result_dict = runner._execute_runner_command_detail_dict(
            parsed_args_obj=parsed_args_obj,
            state_store_obj=state_store_obj,
            as_of_ts=as_of_ts,
            db_path_str=pod_target_obj.db_path_str,
        )
        state_store_obj.record_job_finish(job_run_id_int, "completed", result_dict)
        return result_dict
    except Exception as exception_obj:
        error_detail_dict = {"error_str": str(exception_obj)}
        state_store_obj.record_job_finish(job_run_id_int, "failed", error_detail_dict)
        raise


def _derive_next_action_dict(
    release_obj: LiveRelease,
    latest_decision_plan_row_dict: dict[str, Any] | None,
    latest_vplan_row_dict: dict[str, Any] | None,
    as_of_ts: datetime,
) -> dict[str, str]:
    build_gate_dict = scheduler_utils.evaluate_build_gate_dict(release_obj, as_of_ts)
    next_action_str = "wait"
    reason_code_str = str(build_gate_dict["reason_code_str"])
    if latest_decision_plan_row_dict is None:
        if bool(build_gate_dict["due_bool"]):
            next_action_str = "build_decision_plan"
            reason_code_str = "ready_to_build_decision_plan"
    elif latest_decision_plan_row_dict.get("status_str") == "planned":
        target_execution_ts = _parse_timestamp_ts(
            latest_decision_plan_row_dict["target_execution_timestamp_str"]
        )
        submission_ts = _parse_timestamp_ts(latest_decision_plan_row_dict["submission_timestamp_str"])
        if scheduler_utils.is_execution_window_expired_bool(
            str(latest_decision_plan_row_dict["execution_policy_str"]),
            target_execution_ts,
            as_of_ts,
        ):
            next_action_str = "expire_stale"
            reason_code_str = "submission_window_expired"
        elif submission_ts <= as_of_ts:
            next_action_str = "build_vplan"
            reason_code_str = "ready_to_build_vplan"
        else:
            reason_code_str = "waiting_for_submission_window"
    elif latest_vplan_row_dict is not None and latest_vplan_row_dict.get("status_str") == "ready":
        next_action_str = "submit_vplan" if release_obj.auto_submit_enabled_bool else "review_vplan"
        reason_code_str = "vplan_ready"
    elif latest_vplan_row_dict is not None and latest_vplan_row_dict.get("status_str") in (
        "submitted",
        "submitting",
    ):
        next_action_str = "post_execution_reconcile"
        reason_code_str = "waiting_for_post_execution_reconcile"
    elif latest_decision_plan_row_dict.get("status_str") in ("completed", "expired", "blocked"):
        if bool(build_gate_dict["due_bool"]):
            next_action_str = "build_decision_plan"
            reason_code_str = "ready_to_build_decision_plan"
    return {
        "next_action_str": next_action_str,
        "reason_code_str": reason_code_str,
    }


def _empty_eod_snapshot_dict(status_str: str = "not_applicable") -> dict[str, Any]:
    return {
        "status_str": status_str,
        "severity_str": "gray",
        "latest_timestamp_str": None,
        "latest_market_date_str": None,
        "recorded_timestamp_str": None,
        "source_str": None,
        "equity_float": None,
        "cash_float": None,
        "position_count_int": 0,
        "expected_due_timestamp_str": None,
        "expected_market_date_str": None,
        "same_session_bool": False,
        "unresolved_execution_bool": False,
        "detail_str": "EOD snapshot is not available for this DB state.",
    }


def _build_eod_snapshot_dict(
    connection_obj: sqlite3.Connection,
    release_obj: LiveRelease,
    latest_vplan_row_dict: dict[str, Any] | None,
    as_of_ts: datetime,
) -> dict[str, Any]:
    expected_due_ts = runner._eod_snapshot_due_timestamp_ts(
        release_obj=release_obj,
        as_of_ts=as_of_ts,
    )
    expected_market_date_str = None
    if expected_due_ts is not None:
        expected_market_date_str = runner._eod_snapshot_market_date_str(
            release_obj=release_obj,
            as_of_ts=as_of_ts,
        )
    eod_row_dict_list = _fetch_all_dict_list(
        connection_obj,
        """
        SELECT *
        FROM pod_state_history
        WHERE pod_id_str = ?
          AND snapshot_stage_str = 'eod'
        ORDER BY updated_timestamp_str DESC, pod_state_history_id_int DESC
        """,
        (release_obj.pod_id_str,),
    )
    latest_eod_row_dict = eod_row_dict_list[0] if len(eod_row_dict_list) > 0 else None
    same_session_eod_row_dict = None
    if expected_market_date_str is not None:
        for eod_row_dict in eod_row_dict_list:
            if (
                _market_date_from_history_row_str(eod_row_dict, release_obj)
                == expected_market_date_str
            ):
                same_session_eod_row_dict = eod_row_dict
                break

    unresolved_execution_bool = (
        latest_vplan_row_dict is not None
        and latest_vplan_row_dict.get("status_str") in ("submitted", "submitting")
    )
    if expected_due_ts is None:
        status_str = "not_applicable"
        severity_str = "gray"
        detail_str = "No active market session for EOD sampling."
    elif same_session_eod_row_dict is not None:
        status_str = "completed"
        severity_str = "green"
        detail_str = "Same-session EOD snapshot exists."
    elif unresolved_execution_bool and as_of_ts >= expected_due_ts:
        status_str = "blocked_by_execution"
        severity_str = "gray"
        detail_str = "EOD snapshot is waiting for submitted execution to reconcile."
    elif as_of_ts < expected_due_ts:
        status_str = "waiting"
        severity_str = "gray"
        detail_str = "EOD snapshot is not due yet."
    elif latest_eod_row_dict is None:
        status_str = "due_missing"
        severity_str = "yellow"
        detail_str = "EOD snapshot is due and no prior EOD snapshot exists."
    else:
        status_str = "due_missing"
        severity_str = "yellow"
        detail_str = "EOD snapshot is due and missing for the current market session."

    result_dict = _empty_eod_snapshot_dict(status_str)
    result_dict.update(
        {
            "severity_str": severity_str,
            "expected_due_timestamp_str": (
                None if expected_due_ts is None else expected_due_ts.isoformat()
            ),
            "expected_market_date_str": expected_market_date_str,
            "same_session_bool": same_session_eod_row_dict is not None,
            "unresolved_execution_bool": unresolved_execution_bool,
            "detail_str": detail_str,
        }
    )
    if latest_eod_row_dict is not None:
        result_dict.update(_eod_snapshot_row_summary_dict(latest_eod_row_dict, release_obj))
    return result_dict


def _eod_snapshot_row_summary_dict(
    eod_row_dict: dict[str, Any],
    release_obj: LiveRelease,
) -> dict[str, Any]:
    position_map_dict = _json_map_dict(eod_row_dict.get("position_json_str"))
    return {
        "latest_timestamp_str": eod_row_dict.get("updated_timestamp_str"),
        "latest_market_date_str": _market_date_from_history_row_str(eod_row_dict, release_obj),
        "recorded_timestamp_str": eod_row_dict.get("recorded_timestamp_str"),
        "source_str": eod_row_dict.get("snapshot_source_str"),
        "equity_float": eod_row_dict.get("total_value_float"),
        "cash_float": eod_row_dict.get("cash_float"),
        "position_count_int": _nonzero_position_count_int(position_map_dict),
    }


def _empty_pod_pnl_dict(status_str: str = "unavailable") -> dict[str, Any]:
    return {
        "status_str": status_str,
        "source_str": "pod_state_history.eod",
        "point_count_int": 0,
        "latest_market_date_str": None,
        "latest_equity_float": None,
        "previous_market_date_str": None,
        "previous_equity_float": None,
        "daily_pnl_float": None,
        "daily_pnl_pct_float": None,
        "since_start_pnl_float": None,
        "since_start_pnl_pct_float": None,
        "equity_point_dict_list": [],
    }


def _build_pod_pnl_dict(
    connection_obj: sqlite3.Connection,
    release_obj: LiveRelease,
) -> dict[str, Any]:
    if not _table_exists_bool(connection_obj, "pod_state_history"):
        return _empty_pod_pnl_dict()
    eod_row_dict_list = _fetch_all_dict_list(
        connection_obj,
        """
        SELECT *
        FROM pod_state_history
        WHERE pod_id_str = ?
          AND snapshot_stage_str = 'eod'
        ORDER BY updated_timestamp_str ASC, pod_state_history_id_int ASC
        """,
        (release_obj.pod_id_str,),
    )
    latest_eod_row_by_market_date_dict: dict[str, dict[str, Any]] = {}
    for eod_row_dict in eod_row_dict_list:
        market_date_str = _market_date_from_history_row_str(eod_row_dict, release_obj)
        if market_date_str is None:
            continue
        latest_eod_row_by_market_date_dict[market_date_str] = eod_row_dict

    equity_point_dict_list: list[dict[str, Any]] = []
    first_equity_float: float | None = None
    previous_equity_float: float | None = None
    # *** CRITICAL*** PnL is computed only from market-date ordered EOD
    # broker NetLiq snapshots. Do not mix intraday/post-execution samples
    # into this series or the close-marked PnL basis becomes inconsistent.
    for market_date_str in sorted(latest_eod_row_by_market_date_dict):
        eod_row_dict = latest_eod_row_by_market_date_dict[market_date_str]
        equity_float = float(eod_row_dict["total_value_float"])
        cash_float = float(eod_row_dict["cash_float"])
        if first_equity_float is None:
            first_equity_float = equity_float
        daily_pnl_float = (
            None
            if previous_equity_float is None
            else equity_float - previous_equity_float
        )
        daily_pnl_pct_float = (
            None
            if previous_equity_float is None or previous_equity_float == 0.0
            else (equity_float / previous_equity_float) - 1.0
        )
        since_start_pnl_float = equity_float - first_equity_float
        since_start_pnl_pct_float = (
            None
            if first_equity_float == 0.0
            else (equity_float / first_equity_float) - 1.0
        )
        equity_point_dict_list.append(
            {
                "market_date_str": market_date_str,
                "equity_float": equity_float,
                "cash_float": cash_float,
                "daily_pnl_float": daily_pnl_float,
                "daily_pnl_pct_float": daily_pnl_pct_float,
                "since_start_pnl_float": since_start_pnl_float,
                "since_start_pnl_pct_float": since_start_pnl_pct_float,
                "snapshot_source_str": eod_row_dict.get("snapshot_source_str"),
                "updated_timestamp_str": eod_row_dict.get("updated_timestamp_str"),
                "recorded_timestamp_str": eod_row_dict.get("recorded_timestamp_str"),
            }
        )
        previous_equity_float = equity_float

    if len(equity_point_dict_list) == 0:
        return _empty_pod_pnl_dict()

    latest_point_dict = equity_point_dict_list[-1]
    previous_point_dict = equity_point_dict_list[-2] if len(equity_point_dict_list) >= 2 else None
    return {
        "status_str": "available",
        "source_str": "pod_state_history.eod",
        "point_count_int": len(equity_point_dict_list),
        "latest_market_date_str": latest_point_dict["market_date_str"],
        "latest_equity_float": latest_point_dict["equity_float"],
        "previous_market_date_str": (
            None if previous_point_dict is None else previous_point_dict["market_date_str"]
        ),
        "previous_equity_float": (
            None if previous_point_dict is None else previous_point_dict["equity_float"]
        ),
        "daily_pnl_float": latest_point_dict["daily_pnl_float"],
        "daily_pnl_pct_float": latest_point_dict["daily_pnl_pct_float"],
        "since_start_pnl_float": latest_point_dict["since_start_pnl_float"],
        "since_start_pnl_pct_float": latest_point_dict["since_start_pnl_pct_float"],
        "equity_point_dict_list": equity_point_dict_list,
    }


def build_combined_book_dict(
    pod_target_list: list[DashboardPodTarget],
    as_of_ts: datetime,
) -> dict[str, Any]:
    target_list_by_mode_dict: dict[str, list[DashboardPodTarget]] = {}
    for pod_target_obj in pod_target_list:
        mode_str = str(pod_target_obj.release_obj.mode_str)
        target_list_by_mode_dict.setdefault(mode_str, []).append(pod_target_obj)
    mode_str_list = sorted(
        set(COMBINED_BOOK_MODE_ORDER_LIST) | set(target_list_by_mode_dict),
        key=_combined_book_mode_sort_tuple,
    )
    return {
        "as_of_timestamp_str": as_of_ts.isoformat(),
        "environment_dict_list": [
            _build_combined_book_environment_dict(
                mode_str=mode_str,
                pod_target_list=target_list_by_mode_dict.get(mode_str, []),
            )
            for mode_str in mode_str_list
        ],
    }


def _combined_book_mode_sort_tuple(mode_str: str) -> tuple[int, str]:
    if mode_str in COMBINED_BOOK_MODE_ORDER_LIST:
        return (COMBINED_BOOK_MODE_ORDER_LIST.index(mode_str), mode_str)
    return (len(COMBINED_BOOK_MODE_ORDER_LIST), mode_str)


def _build_combined_book_environment_dict(
    mode_str: str,
    pod_target_list: list[DashboardPodTarget],
) -> dict[str, Any]:
    pod_book_dict_list = [
        _load_combined_book_pod_dict(pod_target_obj)
        for pod_target_obj in pod_target_list
    ]
    strict_equity_point_dict_list = _build_combined_strict_point_dict_list(
        pod_book_dict_list
    )
    carry_forward_equity_point_dict_list = _build_combined_carry_forward_point_dict_list(
        pod_book_dict_list
    )
    warning_dict_list = _build_combined_book_warning_dict_list(
        mode_str=mode_str,
        pod_book_dict_list=pod_book_dict_list,
        latest_operational_market_date_str=(
            None
            if not carry_forward_equity_point_dict_list
            else carry_forward_equity_point_dict_list[-1]["market_date_str"]
        ),
    )
    contribution_dict_list = _build_combined_contribution_dict_list(
        pod_book_dict_list=pod_book_dict_list,
        strict_equity_point_dict_list=strict_equity_point_dict_list,
        carry_forward_equity_point_dict_list=carry_forward_equity_point_dict_list,
    )
    latest_strict_point_dict = (
        None if not strict_equity_point_dict_list else strict_equity_point_dict_list[-1]
    )
    latest_carry_point_dict = (
        None
        if not carry_forward_equity_point_dict_list
        else carry_forward_equity_point_dict_list[-1]
    )
    return {
        "mode_str": mode_str,
        "status_str": _combined_book_status_str(
            pod_count_int=len(pod_book_dict_list),
            strict_point_count_int=len(strict_equity_point_dict_list),
            carry_forward_point_count_int=len(carry_forward_equity_point_dict_list),
            warning_dict_list=warning_dict_list,
        ),
        "pod_count_int": len(pod_book_dict_list),
        "strict_point_count_int": len(strict_equity_point_dict_list),
        "carry_forward_point_count_int": len(carry_forward_equity_point_dict_list),
        "latest_common_market_date_str": (
            None if latest_strict_point_dict is None else latest_strict_point_dict["market_date_str"]
        ),
        "latest_operational_market_date_str": (
            None if latest_carry_point_dict is None else latest_carry_point_dict["market_date_str"]
        ),
        "strict_latest_equity_float": (
            None if latest_strict_point_dict is None else latest_strict_point_dict["equity_float"]
        ),
        "strict_daily_pnl_float": (
            None if latest_strict_point_dict is None else latest_strict_point_dict["daily_pnl_float"]
        ),
        "strict_daily_pnl_pct_float": (
            None if latest_strict_point_dict is None else latest_strict_point_dict["daily_pnl_pct_float"]
        ),
        "carry_forward_latest_equity_float": (
            None if latest_carry_point_dict is None else latest_carry_point_dict["equity_float"]
        ),
        "warning_count_int": len(warning_dict_list),
        "warning_dict_list": warning_dict_list,
        "contribution_dict_list": contribution_dict_list,
        "strict_equity_point_dict_list": strict_equity_point_dict_list,
        "carry_forward_equity_point_dict_list": carry_forward_equity_point_dict_list,
    }


def _load_combined_book_pod_dict(
    pod_target_obj: DashboardPodTarget,
) -> dict[str, Any]:
    release_obj = pod_target_obj.release_obj
    base_dict: dict[str, Any] = {
        "pod_id_str": release_obj.pod_id_str,
        "mode_str": release_obj.mode_str,
        "strategy_import_str": release_obj.strategy_import_str,
        "account_route_str": release_obj.account_route_str,
        "db_path_str": pod_target_obj.db_path_str,
        "db_status_str": "missing",
        "error_str": None,
        "pod_pnl_dict": _empty_pod_pnl_dict(),
        "point_by_market_date_dict": {},
    }
    db_path_obj = Path(pod_target_obj.db_path_str)
    if not db_path_obj.exists():
        return base_dict
    try:
        with _connect_readonly_existing_db(db_path_obj) as connection_obj:
            if not _table_exists_bool(connection_obj, "pod_state_history"):
                base_dict["db_status_str"] = "empty"
                return base_dict
            pod_pnl_dict = _build_pod_pnl_dict(
                connection_obj=connection_obj,
                release_obj=release_obj,
            )
    except sqlite3.DatabaseError as exception_obj:
        base_dict["db_status_str"] = "error"
        base_dict["error_str"] = str(exception_obj)
        return base_dict
    base_dict["db_status_str"] = "ok"
    base_dict["pod_pnl_dict"] = pod_pnl_dict
    base_dict["point_by_market_date_dict"] = {
        point_dict["market_date_str"]: point_dict
        for point_dict in pod_pnl_dict.get("equity_point_dict_list", [])
    }
    return base_dict


def _build_combined_strict_point_dict_list(
    pod_book_dict_list: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if len(pod_book_dict_list) == 0:
        return []
    common_market_date_set: set[str] | None = None
    for pod_book_dict in pod_book_dict_list:
        market_date_set = set(pod_book_dict["point_by_market_date_dict"])
        common_market_date_set = (
            market_date_set
            if common_market_date_set is None
            else common_market_date_set & market_date_set
        )
    if not common_market_date_set:
        return []
    previous_equity_float: float | None = None
    point_dict_list: list[dict[str, Any]] = []
    # *** CRITICAL*** Combined Book strict PnL uses only market-date
    # aligned EOD snapshots. Do not mix intraday, post-execution, or
    # carry-forward values into this strict accounting series.
    for market_date_str in sorted(common_market_date_set):
        equity_float = sum(
            float(pod_book_dict["point_by_market_date_dict"][market_date_str]["equity_float"])
            for pod_book_dict in pod_book_dict_list
        )
        daily_pnl_float = (
            None
            if previous_equity_float is None
            else equity_float - previous_equity_float
        )
        daily_pnl_pct_float = (
            None
            if previous_equity_float is None or previous_equity_float == 0.0
            else (equity_float / previous_equity_float) - 1.0
        )
        point_dict_list.append(
            {
                "market_date_str": market_date_str,
                "equity_float": equity_float,
                "daily_pnl_float": daily_pnl_float,
                "daily_pnl_pct_float": daily_pnl_pct_float,
                "included_pod_count_int": len(pod_book_dict_list),
                "stale_pod_count_int": 0,
                "missing_pod_count_int": 0,
                "basis_str": "strict_common_eod",
            }
        )
        previous_equity_float = equity_float
    return point_dict_list


def _build_combined_carry_forward_point_dict_list(
    pod_book_dict_list: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    market_date_str_list = sorted(
        {
            market_date_str
            for pod_book_dict in pod_book_dict_list
            for market_date_str in pod_book_dict["point_by_market_date_dict"]
        }
    )
    previous_total_equity_float: float | None = None
    last_point_by_pod_id_dict: dict[str, dict[str, Any]] = {}
    point_dict_list: list[dict[str, Any]] = []
    # *** CRITICAL*** Carry-forward is an operational continuity curve,
    # not the strict accounting headline. It may reuse an older POD EOD
    # mark only when the point is visibly labeled as stale.
    for market_date_str in market_date_str_list:
        total_equity_float = 0.0
        included_pod_count_int = 0
        stale_pod_id_list: list[str] = []
        missing_pod_id_list: list[str] = []
        for pod_book_dict in pod_book_dict_list:
            pod_id_str = str(pod_book_dict["pod_id_str"])
            current_point_dict = pod_book_dict["point_by_market_date_dict"].get(
                market_date_str
            )
            if current_point_dict is not None:
                last_point_by_pod_id_dict[pod_id_str] = current_point_dict
                point_dict = current_point_dict
            else:
                point_dict = last_point_by_pod_id_dict.get(pod_id_str)
                if point_dict is None:
                    missing_pod_id_list.append(pod_id_str)
                    continue
                stale_pod_id_list.append(pod_id_str)
            total_equity_float += float(point_dict["equity_float"])
            included_pod_count_int += 1
        if included_pod_count_int == 0:
            continue
        daily_pnl_float = (
            None
            if previous_total_equity_float is None
            else total_equity_float - previous_total_equity_float
        )
        daily_pnl_pct_float = (
            None
            if previous_total_equity_float is None or previous_total_equity_float == 0.0
            else (total_equity_float / previous_total_equity_float) - 1.0
        )
        point_dict_list.append(
            {
                "market_date_str": market_date_str,
                "equity_float": total_equity_float,
                "daily_pnl_float": daily_pnl_float,
                "daily_pnl_pct_float": daily_pnl_pct_float,
                "included_pod_count_int": included_pod_count_int,
                "stale_pod_count_int": len(stale_pod_id_list),
                "missing_pod_count_int": len(missing_pod_id_list),
                "stale_pod_id_list": stale_pod_id_list,
                "missing_pod_id_list": missing_pod_id_list,
                "basis_str": "carry_forward_eod",
            }
        )
        previous_total_equity_float = total_equity_float
    return point_dict_list


def _build_combined_book_warning_dict_list(
    mode_str: str,
    pod_book_dict_list: list[dict[str, Any]],
    latest_operational_market_date_str: str | None,
) -> list[dict[str, Any]]:
    warning_dict_list: list[dict[str, Any]] = []
    route_to_pod_id_list_dict: dict[str, list[str]] = {}
    for pod_book_dict in pod_book_dict_list:
        pod_id_str = str(pod_book_dict["pod_id_str"])
        route_to_pod_id_list_dict.setdefault(
            str(pod_book_dict["account_route_str"]),
            [],
        ).append(pod_id_str)
        db_status_str = str(pod_book_dict["db_status_str"])
        if db_status_str == "missing":
            warning_dict_list.append(
                _combined_book_warning_dict(
                    "missing_db",
                    "gray",
                    "POD DB missing",
                    f"{pod_id_str} has no readable state DB yet.",
                    pod_id_str,
                )
            )
        elif db_status_str == "empty":
            warning_dict_list.append(
                _combined_book_warning_dict(
                    "empty_db",
                    "gray",
                    "POD DB empty",
                    f"{pod_id_str} has no EOD state history table yet.",
                    pod_id_str,
                )
            )
        elif db_status_str == "error":
            warning_dict_list.append(
                _combined_book_warning_dict(
                    "db_error",
                    "red",
                    "POD DB read error",
                    f"{pod_id_str}: {pod_book_dict.get('error_str') or 'read failed'}",
                    pod_id_str,
                )
            )
        elif len(pod_book_dict["point_by_market_date_dict"]) == 0:
            warning_dict_list.append(
                _combined_book_warning_dict(
                    "missing_eod",
                    "gray",
                    "No EOD snapshots",
                    f"{pod_id_str} has no EOD equity snapshots yet.",
                    pod_id_str,
                )
            )
        if latest_operational_market_date_str is not None:
            latest_market_date_str = _latest_pod_market_date_str(pod_book_dict)
            if (
                latest_market_date_str is not None
                and latest_market_date_str < latest_operational_market_date_str
            ):
                warning_dict_list.append(
                    _combined_book_warning_dict(
                        "stale_eod",
                        "yellow",
                        "Stale EOD snapshot",
                        (
                            f"{pod_id_str} latest EOD is {latest_market_date_str}; "
                            f"operational book date is {latest_operational_market_date_str}."
                        ),
                        pod_id_str,
                    )
                )
    for account_route_str, pod_id_str_list in sorted(route_to_pod_id_list_dict.items()):
        if len(pod_id_str_list) <= 1:
            continue
        warning_dict_list.append(
            _combined_book_warning_dict(
                "duplicate_account_route",
                "red",
                "Duplicate account route",
                (
                    f"{mode_str} account route {account_route_str} is shared by "
                    f"{', '.join(sorted(pod_id_str_list))}."
                ),
                None,
            )
        )
    return warning_dict_list


def _combined_book_warning_dict(
    warning_type_str: str,
    severity_str: str,
    label_str: str,
    detail_str: str,
    pod_id_str: str | None,
) -> dict[str, Any]:
    return {
        "warning_type_str": warning_type_str,
        "severity_str": severity_str,
        "label_str": label_str,
        "detail_str": detail_str,
        "pod_id_str": pod_id_str,
    }


def _build_combined_contribution_dict_list(
    pod_book_dict_list: list[dict[str, Any]],
    strict_equity_point_dict_list: list[dict[str, Any]],
    carry_forward_equity_point_dict_list: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    latest_strict_market_date_str = (
        None
        if not strict_equity_point_dict_list
        else strict_equity_point_dict_list[-1]["market_date_str"]
    )
    previous_strict_market_date_str = (
        None
        if len(strict_equity_point_dict_list) < 2
        else strict_equity_point_dict_list[-2]["market_date_str"]
    )
    latest_operational_market_date_str = (
        None
        if not carry_forward_equity_point_dict_list
        else carry_forward_equity_point_dict_list[-1]["market_date_str"]
    )
    contribution_dict_list = [
        _combined_pod_contribution_dict(
            pod_book_dict=pod_book_dict,
            latest_strict_market_date_str=latest_strict_market_date_str,
            previous_strict_market_date_str=previous_strict_market_date_str,
            latest_operational_market_date_str=latest_operational_market_date_str,
        )
        for pod_book_dict in pod_book_dict_list
    ]
    return sorted(
        contribution_dict_list,
        key=lambda contribution_dict: (
            -abs(float(contribution_dict.get("sort_pnl_float") or 0.0)),
            str(contribution_dict["pod_id_str"]),
        ),
    )


def _combined_pod_contribution_dict(
    pod_book_dict: dict[str, Any],
    latest_strict_market_date_str: str | None,
    previous_strict_market_date_str: str | None,
    latest_operational_market_date_str: str | None,
) -> dict[str, Any]:
    point_by_market_date_dict = pod_book_dict["point_by_market_date_dict"]
    latest_market_date_str = _latest_pod_market_date_str(pod_book_dict)
    latest_point_dict = (
        None
        if latest_market_date_str is None
        else point_by_market_date_dict[latest_market_date_str]
    )
    strict_daily_pnl_float = _strict_pod_daily_pnl_float(
        point_by_market_date_dict=point_by_market_date_dict,
        latest_strict_market_date_str=latest_strict_market_date_str,
        previous_strict_market_date_str=previous_strict_market_date_str,
    )
    latest_daily_pnl_float = pod_book_dict["pod_pnl_dict"].get("daily_pnl_float")
    freshness_warning_str = _combined_pod_freshness_warning_str(
        pod_book_dict=pod_book_dict,
        latest_operational_market_date_str=latest_operational_market_date_str,
    )
    sort_pnl_float = (
        strict_daily_pnl_float
        if strict_daily_pnl_float is not None
        else latest_daily_pnl_float
    )
    return {
        "pod_id_str": pod_book_dict["pod_id_str"],
        "mode_str": pod_book_dict["mode_str"],
        "strategy_import_str": pod_book_dict["strategy_import_str"],
        "account_route_str": pod_book_dict["account_route_str"],
        "db_status_str": pod_book_dict["db_status_str"],
        "latest_market_date_str": latest_market_date_str,
        "latest_equity_float": None if latest_point_dict is None else latest_point_dict["equity_float"],
        "latest_daily_pnl_float": latest_daily_pnl_float,
        "strict_daily_pnl_float": strict_daily_pnl_float,
        "carry_forward_status_str": _combined_pod_carry_forward_status_str(
            pod_book_dict=pod_book_dict,
            latest_operational_market_date_str=latest_operational_market_date_str,
        ),
        "freshness_warning_str": freshness_warning_str,
        "source_str": pod_book_dict["pod_pnl_dict"].get("source_str"),
        "equity_point_dict_list": pod_book_dict["pod_pnl_dict"].get(
            "equity_point_dict_list",
            [],
        ),
        "sort_pnl_float": sort_pnl_float,
    }


def _strict_pod_daily_pnl_float(
    point_by_market_date_dict: dict[str, dict[str, Any]],
    latest_strict_market_date_str: str | None,
    previous_strict_market_date_str: str | None,
) -> float | None:
    if latest_strict_market_date_str is None or previous_strict_market_date_str is None:
        return None
    latest_point_dict = point_by_market_date_dict.get(latest_strict_market_date_str)
    previous_point_dict = point_by_market_date_dict.get(previous_strict_market_date_str)
    if latest_point_dict is None or previous_point_dict is None:
        return None
    return float(latest_point_dict["equity_float"]) - float(previous_point_dict["equity_float"])


def _latest_pod_market_date_str(pod_book_dict: dict[str, Any]) -> str | None:
    market_date_str_list = sorted(pod_book_dict["point_by_market_date_dict"])
    return None if not market_date_str_list else market_date_str_list[-1]


def _combined_pod_carry_forward_status_str(
    pod_book_dict: dict[str, Any],
    latest_operational_market_date_str: str | None,
) -> str:
    if latest_operational_market_date_str is None:
        return "missing"
    latest_market_date_str = _latest_pod_market_date_str(pod_book_dict)
    if latest_market_date_str is None:
        return "missing"
    if latest_market_date_str == latest_operational_market_date_str:
        return "current"
    return "carried_forward"


def _combined_pod_freshness_warning_str(
    pod_book_dict: dict[str, Any],
    latest_operational_market_date_str: str | None,
) -> str:
    db_status_str = str(pod_book_dict["db_status_str"])
    if db_status_str == "missing":
        return "DB missing"
    if db_status_str == "empty":
        return "DB empty"
    if db_status_str == "error":
        return "DB read error"
    latest_market_date_str = _latest_pod_market_date_str(pod_book_dict)
    if latest_market_date_str is None:
        return "No EOD snapshots"
    if (
        latest_operational_market_date_str is not None
        and latest_market_date_str < latest_operational_market_date_str
    ):
        return f"Carried from {latest_market_date_str}"
    return ""


def _combined_book_status_str(
    pod_count_int: int,
    strict_point_count_int: int,
    carry_forward_point_count_int: int,
    warning_dict_list: list[dict[str, Any]],
) -> str:
    if pod_count_int == 0:
        return "no_pods"
    if strict_point_count_int > 0:
        if any(
            warning_dict["severity_str"] in {"red", "yellow"}
            for warning_dict in warning_dict_list
        ):
            return "available_with_warnings"
        return "available"
    if carry_forward_point_count_int > 0:
        return "operational_only"
    return "unavailable"


def _market_date_from_history_row_str(
    history_row_dict: dict[str, Any],
    release_obj: LiveRelease,
) -> str | None:
    timestamp_str = history_row_dict.get("updated_timestamp_str")
    if timestamp_str is None:
        return None
    return runner._market_date_str_from_timestamp_str(
        timestamp_str=str(timestamp_str),
        release_obj=release_obj,
    )


def _nonzero_position_count_int(position_map_dict: dict[str, Any]) -> int:
    return len(
        [
            amount_float
            for amount_float in position_map_dict.values()
            if abs(float(amount_float)) > 1e-9
        ]
    )


def _compact_source_label_str(source_obj_list: list[Any]) -> str | None:
    source_str_list = sorted(
        {
            str(source_obj)
            for source_obj in source_obj_list
            if source_obj is not None and str(source_obj).strip() != ""
        }
    )
    if len(source_str_list) == 0:
        return None
    if len(source_str_list) == 1:
        return source_str_list[0]
    return ", ".join(source_str_list)


def _fetch_latest_open_price_source_str(
    connection_obj: sqlite3.Connection,
    vplan_id_int: int,
) -> str | None:
    if not _table_exists_bool(connection_obj, "vplan_fill"):
        return None
    row_dict_list = _fetch_all_dict_list(
        connection_obj,
        """
        SELECT open_price_source_str
        FROM vplan_fill
        WHERE vplan_id_int = ?
          AND open_price_source_str IS NOT NULL
        ORDER BY fill_record_id_int DESC
        """,
        (int(vplan_id_int),),
    )
    return _compact_source_label_str(
        [row_dict.get("open_price_source_str") for row_dict in row_dict_list]
    )


def _fetch_completed_rehearsal_cycle_count_int(
    connection_obj: sqlite3.Connection,
    pod_id_str: str,
) -> int:
    if not _table_exists_bool(connection_obj, "vplan") or not _table_exists_bool(
        connection_obj,
        "vplan_fill",
    ):
        return 0
    if not _table_exists_bool(connection_obj, "vplan_reconciliation_snapshot"):
        return 0
    row_obj = connection_obj.execute(
        """
        SELECT COUNT(DISTINCT v.vplan_id_int) AS count_int
        FROM vplan v
        JOIN vplan_fill f
          ON f.vplan_id_int = v.vplan_id_int
        WHERE v.pod_id_str = ?
          AND v.status_str = 'completed'
          AND EXISTS (
              SELECT 1
              FROM vplan_reconciliation_snapshot r
              WHERE r.vplan_id_int = v.vplan_id_int
                AND r.stage_str = 'post_execution'
                AND r.status_str = 'passed'
          )
        """,
        (pod_id_str,),
    ).fetchone()
    if row_obj is None:
        return 0
    return int(row_obj["count_int"])


def _resolve_health_str(row_dict: dict[str, Any]) -> str:
    if row_dict.get("db_status_str") in ("missing", "empty"):
        return "gray"
    if row_dict.get("db_status_str") == "error":
        return "red"
    if row_dict.get("latest_decision_plan_status_str") == "blocked":
        return "red"
    if row_dict.get("latest_vplan_status_str") == "blocked":
        return "red"
    if int(row_dict.get("missing_ack_count_int") or 0) > 0:
        return "red"
    if int(row_dict.get("exception_count_int") or 0) > 0:
        return "red"
    if row_dict.get("next_action_str") in {
        "build_decision_plan",
        "build_vplan",
        "submit_vplan",
        "review_vplan",
        "post_execution_reconcile",
        "manual_review",
        "expire_stale",
    }:
        return "yellow"
    if row_dict.get("latest_pod_state_timestamp_str") is None and row_dict.get("latest_vplan_id_int") is None:
        return "gray"
    return "green"


def _rehearsal_cycle_status_str(row_dict: dict[str, Any]) -> str:
    if row_dict.get("latest_decision_plan_status_str") is None:
        return "no_decision"
    if row_dict.get("latest_vplan_id_int") is None:
        return "decision_only"
    if row_dict.get("latest_vplan_status_str") == "blocked":
        return "blocked"
    if int(row_dict.get("fill_count_int") or 0) == 0:
        if row_dict.get("latest_vplan_status_str") in {"submitted", "submitting"}:
            return "awaiting_sim_settlement"
        return "awaiting_vplan_submit"
    if int(row_dict.get("exception_count_int") or 0) > 0:
        return "manual_review"
    if row_dict.get("latest_reconciliation_status_str") == "passed":
        return "cycle_complete"
    return "awaiting_reconcile"


def _rehearsal_gate_status_str(row_dict: dict[str, Any], cycle_status_str: str) -> str:
    if int(row_dict.get("completed_rehearsal_cycle_count_int") or 0) > 0:
        return "complete_one_cycle"
    if cycle_status_str in {"blocked", "manual_review"}:
        return "blocked"
    return "incomplete"


def _build_rehearsal_status_dict(row_dict: dict[str, Any]) -> dict[str, Any]:
    if row_dict.get("mode_str") != "incubation":
        return {
            "status_str": "not_applicable",
            "promotion_gate_status_str": "not_applicable",
            "detail_str": "Rehearsal status applies only to mode=incubation.",
        }

    cycle_status_str = _rehearsal_cycle_status_str(row_dict)
    reference_source_str = row_dict.get("latest_live_reference_source_str")
    open_source_str = row_dict.get("latest_open_price_source_str")
    return {
        "status_str": "active",
        "sim_ledger_status_str": (
            "available" if row_dict.get("latest_pod_state_timestamp_str") else "not_started"
        ),
        "sim_ledger_source_str": row_dict.get("latest_pod_state_source_str"),
        "last_cycle_status_str": cycle_status_str,
        "promotion_gate_status_str": _rehearsal_gate_status_str(row_dict, cycle_status_str),
        "promotion_gate_model_str": "cycle_based",
        "completed_cycle_count_int": int(
            row_dict.get("completed_rehearsal_cycle_count_int") or 0
        ),
        "ibkr_reference_status_str": "available" if reference_source_str else "missing",
        "ibkr_reference_source_str": reference_source_str,
        "ibkr_open_price_status_str": "recorded" if open_source_str else "missing",
        "ibkr_open_price_source_str": open_source_str,
        "paper_probe_status_str": "separate_probe_required",
        "paper_probe_accounting_truth_bool": False,
        "official_accounting_source_str": "incubation_sim_ledger",
        "detail_str": (
            "SIM ledger is the official rehearsal state. Dashboard reads per-POD "
            "incubation state; IBKR price/open sources are evidence. Paper probe "
            "is evidence only; it does not count as SIM ledger P&L."
        ),
    }


def _finalize_operator_fields_dict(row_dict: dict[str, Any]) -> dict[str, Any]:
    row_dict["latest_vplan_is_for_latest_decision_bool"] = (
        _latest_vplan_is_for_latest_decision_bool(row_dict)
    )
    row_dict["latest_vplan_cycle_role_str"] = _latest_vplan_cycle_role_str(row_dict)
    row_dict["rehearsal_status_dict"] = _build_rehearsal_status_dict(row_dict)
    row_dict["required_action_dict"] = _build_required_action_dict(row_dict)
    row_dict["lifecycle_step_dict_list"] = _build_lifecycle_step_dict_list(row_dict)
    row_dict["data_freshness_dict"] = _build_data_freshness_dict(row_dict)
    row_dict["debug_summary_dict"] = _build_debug_summary_dict(row_dict)
    return row_dict


def _finalize_pod_detail_debug_story_dict(detail_dict: dict[str, Any]) -> dict[str, Any]:
    detail_dict["debug_story_dict"] = _build_debug_story_dict(detail_dict)
    return detail_dict


def _latest_vplan_is_for_latest_decision_bool(row_dict: dict[str, Any]) -> bool | None:
    if row_dict.get("latest_vplan_id_int") is None:
        return None
    latest_decision_plan_id_obj = row_dict.get("latest_decision_plan_id_int")
    latest_vplan_decision_plan_id_obj = row_dict.get("latest_vplan_decision_plan_id_int")
    if latest_decision_plan_id_obj is None or latest_vplan_decision_plan_id_obj is None:
        return None
    return int(latest_decision_plan_id_obj) == int(latest_vplan_decision_plan_id_obj)


def _latest_vplan_cycle_role_str(row_dict: dict[str, Any]) -> str:
    if row_dict.get("latest_vplan_id_int") is None:
        return "none"
    match_bool = _latest_vplan_is_for_latest_decision_bool(row_dict)
    if match_bool is False:
        return "previous"
    if match_bool is True:
        return "current"
    return "unknown"


def _build_required_action_dict(row_dict: dict[str, Any]) -> dict[str, Any]:
    action_dict = _build_required_action_base_dict(row_dict)
    action_dict["context_item_dict_list"] = _build_required_action_context_item_dict_list(
        row_dict
    )
    return action_dict


def _build_required_action_base_dict(row_dict: dict[str, Any]) -> dict[str, Any]:
    db_status_str = str(row_dict.get("db_status_str") or "")
    next_action_str = str(row_dict.get("next_action_str") or "")
    reason_code_str = str(row_dict.get("reason_code_str") or "")
    if db_status_str == "missing":
        return _required_action_dict(
            "Setup DB",
            "gray",
            "State DB does not exist yet.",
            "status",
        )
    if db_status_str == "empty":
        return _required_action_dict(
            "No state yet",
            "gray",
            "State DB exists but has no live schema rows yet.",
            "status",
        )
    if db_status_str == "error":
        return _required_action_dict(
            "Manual review",
            "red",
            str(row_dict.get("error_str") or "State DB could not be read."),
            "status",
        )
    if row_dict.get("latest_decision_plan_status_str") == "blocked":
        return _required_action_dict("Manual review", "red", "DecisionPlan is blocked.", "status")
    if row_dict.get("latest_vplan_status_str") == "blocked":
        if row_dict.get("latest_vplan_cycle_role_str") == "previous":
            return _required_action_dict(
                "Review previous VPlan",
                "red",
                "Previous execution cycle VPlan is blocked.",
                "show_vplan",
            )
        return _required_action_dict("Manual review", "red", "VPlan is blocked.", "show_vplan")
    if int(row_dict.get("missing_ack_count_int") or 0) > 0:
        if row_dict.get("latest_vplan_cycle_role_str") == "previous":
            return _required_action_dict(
                "Review previous ACK",
                "red",
                f"Previous execution cycle missing ACK count: {int(row_dict.get('missing_ack_count_int') or 0)}.",
                "show_vplan",
            )
        return _required_action_dict(
            "Review broker ACK",
            "red",
            f"Missing ACK count: {int(row_dict.get('missing_ack_count_int') or 0)}.",
            "show_vplan",
        )
    if int(row_dict.get("exception_count_int") or 0) > 0:
        return _required_action_dict(
            "Review reconcile",
            "red",
            "Latest reconciliation is not passed.",
            "status",
        )
    norgate_gate_dict = _build_norgate_current_cycle_gate_dict(row_dict)
    if str(norgate_gate_dict.get("severity_str") or "") == "red":
        return _required_action_dict(
            "Review DecisionPlan gate",
            "red",
            str(norgate_gate_dict.get("detail_str") or "Norgate DecisionPlan gate needs review."),
            "show_decision_plan",
        )
    if next_action_str == "build_decision_plan":
        return _required_action_dict(
            "Build DecisionPlan",
            "yellow",
            reason_code_str,
            "next_due",
        )
    if next_action_str == "build_vplan":
        return _required_action_dict("Build VPlan", "yellow", reason_code_str, "show_decision_plan")
    if reason_code_str == "waiting_for_submission_window":
        return _required_action_dict(
            "Wait submission window",
            "yellow",
            reason_code_str,
            "show_decision_plan",
        )
    if next_action_str == "submit_vplan":
        return _required_action_dict(
            "VPlan ready",
            "yellow",
            "Auto-submit is enabled; inspect before the service submits.",
            "show_vplan",
        )
    if next_action_str == "review_vplan":
        return _required_action_dict(
            "Review VPlan",
            "yellow",
            "Auto-submit is disabled; inspect the VPlan.",
            "show_vplan",
        )
    if next_action_str == "post_execution_reconcile":
        return _required_action_dict(
            "Waiting reconcile",
            "yellow",
            reason_code_str,
            "show_vplan",
        )
    if next_action_str == "expire_stale":
        return _required_action_dict("Expire stale plan", "yellow", reason_code_str, "status")
    if row_dict.get("latest_pod_state_timestamp_str") is None and row_dict.get("latest_vplan_id_int") is None:
        return _required_action_dict("No state yet", "gray", "No POD state or VPlan was found.", "status")
    return _required_action_dict("No action", "green", "POD is idle or completed.", "status")


def _build_required_action_context_item_dict_list(row_dict: dict[str, Any]) -> list[dict[str, str]]:
    context_item_dict_list: list[dict[str, str]] = []
    next_action_str = str(row_dict.get("next_action_str") or "")
    reason_code_str = str(row_dict.get("reason_code_str") or "")
    clock_relevant_bool = (
        reason_code_str == "waiting_for_submission_window"
        or next_action_str in {"build_vplan", "submit_vplan", "review_vplan"}
    )
    if clock_relevant_bool:
        submission_timestamp_str = (
            row_dict.get("latest_vplan_submission_timestamp_str")
            or row_dict.get("latest_decision_plan_submission_timestamp_str")
        )
        target_execution_timestamp_str = (
            row_dict.get("latest_vplan_target_execution_timestamp_str")
            or row_dict.get("latest_decision_plan_target_execution_timestamp_str")
        )
        context_item_dict_list.append(
            _required_action_context_item_dict(
                "Submission opens",
                submission_timestamp_str,
                "yellow" if submission_timestamp_str else "gray",
                "Earliest allowed VPlan build/submission time.",
            )
        )
        context_item_dict_list.append(
            _required_action_context_item_dict(
                "Target execution",
                target_execution_timestamp_str,
                "yellow" if target_execution_timestamp_str else "gray",
                "Expected execution clock for this DecisionPlan/VPlan.",
            )
        )

    context_item_dict_list.append(
        _required_action_context_item_dict(
            "Broker snapshot",
            row_dict.get("latest_broker_snapshot_timestamp_str"),
            "green" if row_dict.get("latest_broker_snapshot_timestamp_str") else "gray",
            "Latest broker account snapshot seen by this POD.",
        )
    )
    rehearsal_status_dict = row_dict.get("rehearsal_status_dict") or {}
    if rehearsal_status_dict.get("status_str") == "active":
        gate_status_str = str(rehearsal_status_dict.get("promotion_gate_status_str") or "")
        gate_severity_str = (
            "green"
            if gate_status_str == "complete_one_cycle"
            else "red"
            if gate_status_str == "blocked"
            else "yellow"
        )
        context_item_dict_list.append(
            _required_action_context_item_dict(
                "Rehearsal gate",
                gate_status_str,
                gate_severity_str,
                str(rehearsal_status_dict.get("detail_str") or ""),
            )
        )
        context_item_dict_list.append(
            _required_action_context_item_dict(
                "IBKR price",
                (
                    "ref="
                    f"{rehearsal_status_dict.get('ibkr_reference_source_str') or 'missing'}; "
                    "open="
                    f"{rehearsal_status_dict.get('ibkr_open_price_source_str') or 'missing'}"
                ),
                "green"
                if rehearsal_status_dict.get("ibkr_open_price_status_str") == "recorded"
                else "gray",
                "IBKR price/open sources are evidence for the SIM ledger, not broker execution truth.",
            )
        )
        context_item_dict_list.append(
            _required_action_context_item_dict(
                "Paper probe",
                str(rehearsal_status_dict.get("paper_probe_status_str") or "not_run"),
                "gray",
                "Paper probe is evidence only; it does not count as SIM ledger P&L.",
            )
        )
    context_item_dict_list.append(
        _required_action_context_item_dict(
            "Last event",
            row_dict.get("latest_event_timestamp_str"),
            "green" if row_dict.get("latest_event_timestamp_str") else "gray",
            "Latest JSONL event recorded for this POD.",
        )
    )

    eod_snapshot_dict = row_dict.get("eod_snapshot_dict") or _empty_eod_snapshot_dict()
    eod_status_str = str(eod_snapshot_dict.get("status_str") or "not_applicable")
    eod_due_timestamp_str = eod_snapshot_dict.get("expected_due_timestamp_str")
    eod_latest_timestamp_str = eod_snapshot_dict.get("latest_timestamp_str")
    if eod_due_timestamp_str:
        eod_value_str = f"{eod_status_str} / due {eod_due_timestamp_str}"
    elif eod_latest_timestamp_str:
        eod_value_str = f"{eod_status_str} / latest {eod_latest_timestamp_str}"
    else:
        eod_value_str = eod_status_str
    context_item_dict_list.append(
        _required_action_context_item_dict(
            "EOD",
            eod_value_str,
            str(eod_snapshot_dict.get("severity_str") or "gray"),
            str(eod_snapshot_dict.get("detail_str") or ""),
        )
    )

    reconciliation_timestamp_str = row_dict.get("latest_reconciliation_timestamp_str")
    if reconciliation_timestamp_str:
        reconciliation_status_str = str(row_dict.get("latest_reconciliation_status_str") or "")
        context_item_dict_list.append(
            _required_action_context_item_dict(
                "Reconcile",
                reconciliation_timestamp_str,
                _status_to_severity_str(
                    reconciliation_status_str,
                    green_status_set={"passed"},
                    yellow_status_set={"pending", "recorded"},
                ),
                reconciliation_status_str,
            )
        )
    return context_item_dict_list


def _required_action_context_item_dict(
    label_str: str,
    value_obj: Any,
    severity_str: str,
    detail_str: str,
) -> dict[str, str]:
    value_str = "unavailable" if value_obj is None or value_obj == "" else str(value_obj)
    clean_severity_str = severity_str if severity_str in {"green", "yellow", "red", "gray"} else "gray"
    return {
        "label_str": label_str,
        "value_str": value_str,
        "severity_str": clean_severity_str,
        "detail_str": detail_str,
    }


def _required_action_dict(
    label_str: str,
    severity_str: str,
    reason_str: str,
    inspect_command_name_str: str | None,
) -> dict[str, Any]:
    return {
        "label_str": label_str,
        "severity_str": severity_str,
        "reason_str": reason_str,
        "inspect_command_name_str": inspect_command_name_str,
    }


def _build_lifecycle_step_dict_list(row_dict: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        _build_db_step_dict(row_dict),
        _build_decision_step_dict(row_dict),
        _build_vplan_step_dict(row_dict),
        _build_ack_step_dict(row_dict),
        _build_fill_step_dict(row_dict),
        _build_reconcile_step_dict(row_dict),
        _build_eod_step_dict(row_dict),
        _build_diff_step_dict(row_dict),
    ]


def _lifecycle_step_dict(
    step_key_str: str,
    label_str: str,
    status_str: str,
    severity_str: str,
    detail_str: str,
    timestamp_str: str | None = None,
) -> dict[str, Any]:
    return {
        "step_key_str": step_key_str,
        "label_str": label_str,
        "status_str": status_str,
        "severity_str": severity_str,
        "detail_str": detail_str,
        "timestamp_str": timestamp_str,
    }


def _build_db_step_dict(row_dict: dict[str, Any]) -> dict[str, Any]:
    db_status_str = str(row_dict.get("db_status_str") or "missing")
    severity_str = "green"
    if db_status_str in ("missing", "empty"):
        severity_str = "gray"
    elif db_status_str == "error":
        severity_str = "red"
    return _lifecycle_step_dict(
        "db",
        "DB",
        db_status_str,
        severity_str,
        str(row_dict.get("db_path_str") or ""),
    )


def _build_decision_step_dict(row_dict: dict[str, Any]) -> dict[str, Any]:
    status_str = str(row_dict.get("latest_decision_plan_status_str") or "none")
    severity_str = _status_to_severity_str(
        status_str,
        green_status_set={"completed"},
        yellow_status_set={"planned", "vplan_ready", "submitted", "expired"},
    )
    return _lifecycle_step_dict(
        "decision",
        "Decision",
        status_str,
        severity_str,
        str(row_dict.get("reason_code_str") or ""),
    )


def _build_vplan_step_dict(row_dict: dict[str, Any]) -> dict[str, Any]:
    status_str = str(row_dict.get("latest_vplan_status_str") or "none")
    severity_str = _status_to_severity_str(
        status_str,
        green_status_set={"completed"},
        yellow_status_set={"ready", "submitting", "submitted", "expired"},
    )
    detail_str = ""
    if row_dict.get("latest_vplan_id_int") is not None:
        detail_str = f"#{row_dict['latest_vplan_id_int']}"
        if row_dict.get("latest_vplan_cycle_role_str") == "previous":
            detail_str = f"previous cycle {detail_str}"
    return _lifecycle_step_dict("vplan", "VPlan", status_str, severity_str, detail_str)


def _build_ack_step_dict(row_dict: dict[str, Any]) -> dict[str, Any]:
    if row_dict.get("latest_vplan_id_int") is None:
        return _lifecycle_step_dict("ack", "ACK", "none", "gray", "No VPlan.")
    status_str = str(row_dict.get("latest_submit_ack_status_str") or "not_checked")
    missing_ack_count_int = int(row_dict.get("missing_ack_count_int") or 0)
    if missing_ack_count_int > 0 or status_str == "missing_critical":
        return _lifecycle_step_dict(
            "ack",
            "ACK",
            status_str,
            "red",
            f"missing={missing_ack_count_int}",
        )
    if status_str == "complete":
        detail_str = f"rows={int(row_dict.get('broker_ack_count_int') or 0)}"
        if row_dict.get("latest_vplan_cycle_role_str") == "previous":
            detail_str = f"previous cycle, {detail_str}"
        return _lifecycle_step_dict(
            "ack",
            "ACK",
            status_str,
            "green",
            detail_str,
        )
    severity_str = "yellow" if row_dict.get("latest_vplan_status_str") in ("submitted", "submitting") else "gray"
    return _lifecycle_step_dict("ack", "ACK", status_str, severity_str, "Awaiting ACK evidence.")


def _build_fill_step_dict(row_dict: dict[str, Any]) -> dict[str, Any]:
    if row_dict.get("latest_vplan_id_int") is None:
        return _lifecycle_step_dict("fill", "Fill", "none", "gray", "No VPlan.")
    fill_count_int = int(row_dict.get("fill_count_int") or 0)
    if fill_count_int > 0:
        detail_str = f"fill_records={fill_count_int}"
        if row_dict.get("latest_vplan_cycle_role_str") == "previous":
            detail_str = f"previous cycle, {detail_str}"
        return _lifecycle_step_dict("fill", "Fill", "recorded", "green", detail_str)
    severity_str = "yellow" if row_dict.get("latest_vplan_status_str") in ("submitted", "submitting") else "gray"
    return _lifecycle_step_dict("fill", "Fill", "none", severity_str, "No fills recorded.")


def _build_reconcile_step_dict(row_dict: dict[str, Any]) -> dict[str, Any]:
    status_str = str(row_dict.get("latest_reconciliation_status_str") or "none")
    if status_str == "none":
        severity_str = "yellow" if row_dict.get("next_action_str") == "post_execution_reconcile" else "gray"
    elif status_str == "passed":
        severity_str = "green"
    else:
        severity_str = "red"
    return _lifecycle_step_dict(
        "reconcile",
        "Reconcile",
        status_str,
        severity_str,
        str(row_dict.get("reason_code_str") or ""),
        row_dict.get("latest_reconciliation_timestamp_str"),
    )


def _build_eod_step_dict(row_dict: dict[str, Any]) -> dict[str, Any]:
    eod_snapshot_dict = row_dict.get("eod_snapshot_dict") or {}
    status_str = str(eod_snapshot_dict.get("status_str") or "not_applicable")
    severity_str = str(eod_snapshot_dict.get("severity_str") or "gray")
    if severity_str not in {"green", "yellow", "red", "gray"}:
        severity_str = "gray"
    detail_str = str(
        eod_snapshot_dict.get("expected_market_date_str")
        or eod_snapshot_dict.get("latest_market_date_str")
        or eod_snapshot_dict.get("detail_str")
        or ""
    )
    return _lifecycle_step_dict(
        "eod",
        "EOD",
        status_str,
        severity_str,
        detail_str,
        eod_snapshot_dict.get("latest_timestamp_str"),
    )


def _build_diff_step_dict(row_dict: dict[str, Any]) -> dict[str, Any]:
    status_str = str(row_dict.get("latest_diff_status_str") or "not_run")
    severity_str = "green" if status_str == "green" else status_str
    if severity_str not in {"green", "yellow", "red", "gray"}:
        severity_str = "gray" if status_str == "not_run" else "red"
    return _lifecycle_step_dict(
        "diff",
        "DIFF",
        status_str,
        severity_str,
        f"open_issues={row_dict.get('latest_diff_open_issue_count_int')}",
        row_dict.get("latest_diff_timestamp_str"),
    )


def _status_to_severity_str(
    status_str: str,
    green_status_set: set[str],
    yellow_status_set: set[str],
) -> str:
    if status_str in {"blocked", "error", "failed", "fail", "missing_critical"}:
        return "red"
    if status_str in green_status_set:
        return "green"
    if status_str in yellow_status_set:
        return "yellow"
    return "gray"


def _build_data_freshness_dict(row_dict: dict[str, Any]) -> dict[str, Any]:
    eod_snapshot_dict = row_dict.get("eod_snapshot_dict") or _empty_eod_snapshot_dict()
    norgate_snapshot_status_dict = row_dict.get("norgate_snapshot_status_dict") or {}
    norgate_current_cycle_gate_dict = _build_norgate_current_cycle_gate_dict(row_dict)
    norgate_item_severity_str = str(norgate_snapshot_status_dict.get("severity_str") or "gray")

    # Default-path Norgate detail (raw sync surface). When the current cycle
    # gate is *not* allowing continuation, this is what the operator sees:
    # the raw source/status/stage/error fragments that explain the sync state.
    raw_detail_fragment_list = [
        f"source={norgate_snapshot_status_dict.get('data_source_mode_str')}",
        f"status={norgate_snapshot_status_dict.get('status_str')}",
    ]
    if norgate_snapshot_status_dict.get("sync_stage_label_str"):
        raw_detail_fragment_list.append(
            f"stage={norgate_snapshot_status_dict.get('sync_stage_label_str')}"
        )
    if norgate_snapshot_status_dict.get("build_gate_reason_code_str"):
        raw_detail_fragment_list.append(
            f"gate={norgate_snapshot_status_dict.get('build_gate_reason_code_str')}"
        )
    if norgate_snapshot_status_dict.get("last_error_str"):
        raw_detail_fragment_list.append(
            f"error={norgate_snapshot_status_dict.get('last_error_str')}"
        )
    norgate_primary_detail_str = ", ".join(raw_detail_fragment_list)
    norgate_sub_detail_str_list: list[str] = []

    # Current-cycle continuation path: promote the gate verdict to the primary
    # line, demote the gate explanation + any future-risk raw sync info to
    # sub-detail bullets so the operator sees "what does this mean for me
    # right now?" before "what should I plan to fix later?".
    if _norgate_not_blocking_current_cycle_bool(row_dict):
        norgate_item_severity_str = "green"
        gate_status_label_str = str(norgate_current_cycle_gate_dict.get("status_label_str") or "")
        gate_detail_str = str(norgate_current_cycle_gate_dict.get("detail_str") or "")
        norgate_primary_detail_str = gate_status_label_str or gate_detail_str
        if gate_detail_str and gate_detail_str != gate_status_label_str:
            norgate_sub_detail_str_list.append(gate_detail_str)
        raw_status_str = str(norgate_snapshot_status_dict.get("status_str") or "")
        if raw_status_str and raw_status_str not in {"ready", "direct"}:
            raw_future_fragment_list = [f"Next DecisionPlan raw sync: status={raw_status_str}"]
            raw_reason_str = str(norgate_snapshot_status_dict.get("reason_code_str") or "")
            if raw_reason_str:
                raw_future_fragment_list.append(f"reason={raw_reason_str}")
            raw_error_str = str(norgate_snapshot_status_dict.get("last_error_str") or "")
            if raw_error_str:
                raw_future_fragment_list.append(f"error={raw_error_str}")
            norgate_sub_detail_str_list.append(", ".join(raw_future_fragment_list))

    item_dict_list = [
        _freshness_item_dict(
            "Norgate",
            norgate_snapshot_status_dict.get("snapshot_date_str")
            or norgate_snapshot_status_dict.get("last_sync_utc_str"),
            norgate_item_severity_str,
            norgate_primary_detail_str,
            sub_detail_str_list=norgate_sub_detail_str_list,
        ),
        _freshness_item_dict(
            "Pod state",
            row_dict.get("latest_pod_state_timestamp_str"),
            "green" if row_dict.get("latest_pod_state_timestamp_str") else "gray",
            "Latest persisted strategy/broker sleeve state.",
        ),
        _freshness_item_dict(
            "Broker snapshot",
            row_dict.get("latest_broker_snapshot_timestamp_str"),
            "green" if row_dict.get("latest_broker_snapshot_timestamp_str") else "gray",
            "Latest broker account snapshot used by this POD.",
        ),
        _freshness_item_dict(
            "EOD Snapshot",
            eod_snapshot_dict.get("latest_timestamp_str"),
            str(eod_snapshot_dict.get("severity_str") or "gray"),
            str(eod_snapshot_dict.get("detail_str") or ""),
        ),
        _freshness_item_dict(
            "Live reference",
            row_dict.get("latest_live_reference_snapshot_timestamp_str"),
            "green" if row_dict.get("latest_live_reference_snapshot_timestamp_str") else "gray",
            "Latest quote snapshot used for VPlan sizing.",
        ),
        _freshness_item_dict(
            "Event log",
            row_dict.get("latest_event_timestamp_str"),
            "green" if row_dict.get("latest_event_timestamp_str") else "gray",
            "Latest JSONL event for this POD.",
        ),
        _freshness_item_dict(
            "DIFF artifact",
            row_dict.get("latest_diff_timestamp_str"),
            _diff_freshness_severity_str(row_dict),
            str(row_dict.get("latest_diff_status_str") or "not_run"),
        ),
    ]
    if row_dict.get("dtb3_download_status_str") is not None:
        item_dict_list.append(
            _freshness_item_dict(
                "DTB3/FRED",
                row_dict.get("dtb3_latest_observation_date_str"),
                _dtb3_freshness_severity_str(row_dict),
                (
                    f"policy={row_dict.get('dtb3_policy_status_str') or 'legacy'}, "
                    f"status={row_dict.get('dtb3_download_status_str')}, "
                    f"days={row_dict.get('dtb3_freshness_business_days_int')}, "
                    f"cache={row_dict.get('dtb3_used_cache_bool')}"
                ),
            )
        )
    return {
        "pod_state_updated_timestamp_str": row_dict.get("latest_pod_state_timestamp_str"),
        "broker_snapshot_timestamp_str": row_dict.get("latest_broker_snapshot_timestamp_str"),
        "eod_snapshot_dict": eod_snapshot_dict,
        "live_reference_snapshot_timestamp_str": row_dict.get(
            "latest_live_reference_snapshot_timestamp_str"
        ),
        "latest_event_timestamp_str": row_dict.get("latest_event_timestamp_str"),
        "diff_artifact_timestamp_str": row_dict.get("latest_diff_timestamp_str"),
        "norgate_snapshot_status_dict": norgate_snapshot_status_dict,
        "norgate_current_cycle_gate_dict": norgate_current_cycle_gate_dict,
        "norgate_data_source_mode_str": norgate_snapshot_status_dict.get("data_source_mode_str"),
        "norgate_status_str": norgate_snapshot_status_dict.get("status_str"),
        "norgate_sync_stage_label_str": norgate_snapshot_status_dict.get("sync_stage_label_str"),
        "norgate_reason_code_str": norgate_snapshot_status_dict.get("reason_code_str"),
        "norgate_profile_str": norgate_snapshot_status_dict.get("profile_str"),
        "norgate_snapshot_date_str": norgate_snapshot_status_dict.get("snapshot_date_str"),
        "norgate_last_sync_utc_str": norgate_snapshot_status_dict.get("last_sync_utc_str"),
        "norgate_last_attempt_utc_str": norgate_snapshot_status_dict.get("last_attempt_utc_str"),
        "norgate_last_error_str": norgate_snapshot_status_dict.get("last_error_str"),
        "norgate_build_gate_reason_code_str": norgate_snapshot_status_dict.get("build_gate_reason_code_str"),
        "norgate_status_file_path_str": norgate_snapshot_status_dict.get("status_file_path_str"),
        "dtb3_download_status_str": row_dict.get("dtb3_download_status_str"),
        "dtb3_latest_observation_date_str": row_dict.get("dtb3_latest_observation_date_str"),
        "dtb3_freshness_business_days_int": row_dict.get("dtb3_freshness_business_days_int"),
        "dtb3_source_name_str": row_dict.get("dtb3_source_name_str"),
        "dtb3_used_cache_bool": row_dict.get("dtb3_used_cache_bool"),
        "dtb3_policy_status_str": row_dict.get("dtb3_policy_status_str"),
        "dtb3_warning_bool": row_dict.get("dtb3_warning_bool"),
        "dtb3_warn_after_business_days_int": row_dict.get("dtb3_warn_after_business_days_int"),
        "item_dict_list": item_dict_list,
    }


def _build_norgate_current_cycle_gate_dict(row_dict: dict[str, Any]) -> dict[str, Any]:
    signal_clock_str = scheduler_utils.normalize_signal_clock_str(
        str(row_dict.get("signal_clock_str") or "")
    )
    gate_enabled_bool = signal_clock_str in NORGATE_SNAPSHOT_GATED_SIGNAL_CLOCK_SET
    decision_status_str = str(row_dict.get("latest_decision_plan_status_str") or "")
    next_action_str = str(row_dict.get("next_action_str") or "")
    provenance_status_str = _decision_plan_norgate_provenance_status_str(row_dict)
    continuation_allowed_bool = _norgate_not_blocking_current_cycle_bool(row_dict)

    if not gate_enabled_bool:
        status_label_str = "Not Norgate-gated"
        detail_str = "This pod does not use the Norgate snapshot DecisionPlan gate."
        gate_required_bool = False
        blocked_stage_str = "none"
        severity_str = "gray"
    elif continuation_allowed_bool:
        status_label_str = "Not required for current cycle"
        detail_str = (
            "Valid current DecisionPlan exists; VPlan / submit / reconcile continue "
            "without Norgate sync."
        )
        gate_required_bool = False
        blocked_stage_str = "none"
        severity_str = "green"
    elif provenance_status_str in {
        "release_mismatch",
        "execution_policy_mismatch",
        "missing_profile",
        "profile_mismatch",
    }:
        vplan_exists_bool = row_dict.get("latest_vplan_is_for_latest_decision_bool") is True
        status_label_str = "DecisionPlan provenance invalid"
        gate_required_bool = not vplan_exists_bool
        blocked_stage_str = "none" if vplan_exists_bool else "vplan"
        if vplan_exists_bool:
            detail_str = (
                "DecisionPlan provenance is not valid, but a VPlan already exists; "
                "submit/reconcile status remains the active operator truth."
            )
            severity_str = "gray"
        else:
            detail_str = (
                "The latest DecisionPlan does not match the active release/provenance; "
                "VPlan build should stay blocked."
            )
            severity_str = "red"
    elif next_action_str == "build_decision_plan":
        status_label_str = "Required for next DecisionPlan"
        detail_str = "Norgate readiness gates only new DecisionPlan creation."
        gate_required_bool = True
        blocked_stage_str = "decision_plan"
        severity_str = "yellow"
    elif next_action_str == "expire_stale":
        status_label_str = "Stale plan needs expiry"
        detail_str = "The current plan is stale; expire it before building new intent."
        gate_required_bool = False
        blocked_stage_str = "decision_plan"
        severity_str = "yellow"
    elif decision_status_str:
        status_label_str = "Waiting for next DecisionPlan gate"
        detail_str = "Norgate will be checked again only when a new DecisionPlan is due."
        gate_required_bool = False
        blocked_stage_str = "none"
        severity_str = "gray"
    else:
        status_label_str = "Waiting for signal window"
        detail_str = "No DecisionPlan exists yet; Norgate will be checked before one is built."
        gate_required_bool = False
        blocked_stage_str = "none"
        severity_str = "gray"

    return {
        "gate_enabled_bool": gate_enabled_bool,
        "gate_required_bool": gate_required_bool,
        "current_cycle_continuation_allowed_bool": continuation_allowed_bool,
        "blocked_stage_str": blocked_stage_str,
        "severity_str": severity_str,
        "status_label_str": status_label_str,
        "detail_str": detail_str,
        "signal_clock_str": signal_clock_str,
        "decision_plan_status_str": decision_status_str,
        "decision_plan_provenance_status_str": provenance_status_str,
        "expected_profile_str": str(row_dict.get("data_profile_str") or ""),
        "decision_profile_str": str(row_dict.get("latest_decision_norgate_profile_str") or ""),
        "decision_snapshot_date_str": str(
            row_dict.get("latest_decision_norgate_snapshot_date_str") or ""
        ),
    }


def _decision_plan_norgate_provenance_status_str(row_dict: dict[str, Any]) -> str:
    signal_clock_str = scheduler_utils.normalize_signal_clock_str(
        str(row_dict.get("signal_clock_str") or "")
    )
    if signal_clock_str not in NORGATE_SNAPSHOT_GATED_SIGNAL_CLOCK_SET:
        return "not_applicable"
    if row_dict.get("latest_decision_plan_id_int") is None:
        return "missing_decision_plan"
    expected_release_id_str = str(row_dict.get("release_id_str") or "")
    decision_release_id_str = str(row_dict.get("latest_decision_release_id_str") or "")
    if expected_release_id_str and decision_release_id_str != expected_release_id_str:
        return "release_mismatch"
    expected_execution_policy_str = str(row_dict.get("execution_policy_str") or "")
    decision_execution_policy_str = str(
        row_dict.get("latest_decision_execution_policy_str") or ""
    )
    if (
        expected_execution_policy_str
        and decision_execution_policy_str != expected_execution_policy_str
    ):
        return "execution_policy_mismatch"
    expected_profile_str = str(row_dict.get("data_profile_str") or "")
    decision_profile_str = str(row_dict.get("latest_decision_norgate_profile_str") or "")
    if not decision_profile_str:
        return "missing_profile"
    if expected_profile_str and decision_profile_str != expected_profile_str:
        return "profile_mismatch"
    return "matched"


def _norgate_not_blocking_current_cycle_bool(row_dict: dict[str, Any]) -> bool:
    if _decision_plan_norgate_provenance_status_str(row_dict) != "matched":
        return False
    decision_status_str = str(row_dict.get("latest_decision_plan_status_str") or "")
    if decision_status_str in NORGATE_CURRENT_DECISION_STATUS_SET:
        return str(row_dict.get("next_action_str") or "") != "expire_stale"
    if decision_status_str == "completed":
        return _completed_decision_plan_covers_norgate_signal_cycle_bool(row_dict)
    return False


def _completed_decision_plan_covers_norgate_signal_cycle_bool(row_dict: dict[str, Any]) -> bool:
    try:
        signal_timestamp_ts = _parse_timestamp_ts(
            str(row_dict.get("latest_decision_signal_timestamp_str") or "")
        )
        as_of_ts = _parse_timestamp_ts(str(row_dict.get("as_of_timestamp_str") or ""))
    except ValueError:
        return False
    calendar_id_str = str(row_dict.get("session_calendar_id_str") or "")
    signal_clock_str = scheduler_utils.normalize_signal_clock_str(
        str(row_dict.get("signal_clock_str") or "")
    )
    try:
        signal_session_label_ts = scheduler_utils.session_label_from_timestamp_ts(
            signal_timestamp_ts,
            calendar_id_str,
        )
    except ValueError:
        return False
    if signal_session_label_ts is None:
        return False
    if signal_clock_str == "eod_snapshot_ready":
        next_signal_session_label_ts = scheduler_utils.get_next_session_label_ts(
            signal_session_label_ts,
            calendar_id_str,
        )
    elif signal_clock_str == "month_end_snapshot_ready":
        next_signal_session_label_ts = scheduler_utils.get_next_session_label_ts(
            signal_session_label_ts,
            calendar_id_str,
        )
        while not scheduler_utils.is_last_session_of_month_bool(
            next_signal_session_label_ts,
            calendar_id_str,
        ):
            next_signal_session_label_ts = scheduler_utils.get_next_session_label_ts(
                next_signal_session_label_ts,
                calendar_id_str,
            )
    else:
        return True
    next_ready_timestamp_ts = scheduler_utils.get_session_close_timestamp_ts(
        next_signal_session_label_ts,
        calendar_id_str,
    ) + timedelta(minutes=runner.DEFAULT_EOD_SNAPSHOT_BUFFER_MINUTES_INT)
    market_as_of_ts = scheduler_utils.to_market_timestamp_ts(as_of_ts, calendar_id_str)
    return market_as_of_ts < next_ready_timestamp_ts


def _post_sync_data_load_failure_event_dict(
    event_dict_list: list[dict[str, Any]],
    sync_ready_timestamp_str: str | None,
) -> dict[str, Any] | None:
    sync_ready_sort_key_tuple = _debug_timestamp_sort_key_tuple(sync_ready_timestamp_str)
    if sync_ready_sort_key_tuple[0] == 0:
        return None
    for event_dict in reversed(event_dict_list):
        event_name_str = str(event_dict.get("event_name_str") or "")
        phase_str = str(event_dict.get("next_phase_str") or event_dict.get("phase_str") or "")
        error_str = str(event_dict.get("error_str") or event_dict.get("message_str") or "")
        if not error_str:
            continue
        event_sort_key_tuple = _debug_timestamp_sort_key_tuple(_event_timestamp_str(event_dict))
        if event_sort_key_tuple[0] == 0 or event_sort_key_tuple < sync_ready_sort_key_tuple:
            continue
        if event_name_str == "build_decision_plan_data_dependency_error":
            return event_dict
        if "build_decision_plan" in event_name_str or phase_str == "build_decision_plan":
            return event_dict
    return None


def _build_alert_dict_list(pod_row_dict_list: list[dict[str, Any]]) -> list[dict[str, Any]]:
    alert_dict_list: list[dict[str, Any]] = []
    for row_dict in pod_row_dict_list:
        alert_dict_list.extend(_build_alerts_for_row_dict(row_dict))
    return sorted(
        alert_dict_list,
        key=lambda alert_dict: (
            ALERT_SEVERITY_RANK_DICT.get(str(alert_dict["severity_str"]), 99),
            str(alert_dict["pod_id_str"]),
            str(alert_dict["alert_type_str"]),
            str(alert_dict["label_str"]),
        ),
    )


def _build_alert_summary_dict(alert_dict_list: list[dict[str, Any]]) -> dict[str, Any]:
    summary_dict = {
        "total_count_int": len(alert_dict_list),
        "red_count_int": 0,
        "yellow_count_int": 0,
        "gray_count_int": 0,
    }
    for alert_dict in alert_dict_list:
        severity_str = str(alert_dict.get("severity_str") or "gray")
        key_str = f"{severity_str}_count_int"
        if key_str in summary_dict:
            summary_dict[key_str] = int(summary_dict[key_str]) + 1
    return summary_dict


def _build_alerts_for_row_dict(row_dict: dict[str, Any]) -> list[dict[str, Any]]:
    alert_dict_list: list[dict[str, Any]] = []
    db_status_str = str(row_dict.get("db_status_str") or "")
    if db_status_str == "missing":
        alert_dict_list.append(
            _alert_dict(row_dict, "db", "gray", "DB missing", "State DB does not exist yet.", "status")
        )
    elif db_status_str == "empty":
        alert_dict_list.append(
            _alert_dict(row_dict, "db", "gray", "DB empty", "State DB has no live schema rows yet.", "status")
        )
    elif db_status_str == "error":
        alert_dict_list.append(
            _alert_dict(
                row_dict,
                "db",
                "red",
                "DB read error",
                str(row_dict.get("error_str") or "State DB could not be read."),
                "status",
            )
        )

    if int(row_dict.get("missing_ack_count_int") or 0) > 0:
        alert_dict_list.append(
            _alert_dict(
                row_dict,
                "broker_ack",
                "red",
                "Missing broker ACK",
                f"Missing ACK count: {int(row_dict.get('missing_ack_count_int') or 0)}.",
                "show_vplan",
            )
        )
    if int(row_dict.get("exception_count_int") or 0) > 0:
        alert_dict_list.append(
            _alert_dict(row_dict, "reconcile", "red", "Reconcile issue", "Latest reconciliation is not passed.", "status")
        )

    required_action_dict = row_dict.get("required_action_dict") or {}
    required_severity_str = str(required_action_dict.get("severity_str") or "green")
    required_label_str = str(required_action_dict.get("label_str") or "")
    if _should_emit_required_action_alert_bool(row_dict, required_action_dict):
        alert_dict_list.append(
            _alert_dict(
                row_dict,
                "required_action",
                required_severity_str,
                required_label_str,
                str(required_action_dict.get("reason_str") or ""),
                required_action_dict.get("inspect_command_name_str"),
            )
        )

    diff_status_str = str(row_dict.get("latest_diff_status_str") or "not_run")
    if diff_status_str in {"red", "yellow"}:
        alert_dict_list.append(
            _alert_dict(
                row_dict,
                "diff",
                diff_status_str,
                "DIFF needs review",
                f"status={diff_status_str}, open_issues={row_dict.get('latest_diff_open_issue_count_int')}",
                "compare_reference",
            )
        )

    for item_dict in (row_dict.get("data_freshness_dict") or {}).get("item_dict_list", []):
        freshness_alert_dict = _freshness_alert_dict(row_dict, item_dict)
        if freshness_alert_dict is not None:
            alert_dict_list.append(freshness_alert_dict)

    return alert_dict_list


def _should_emit_required_action_alert_bool(
    row_dict: dict[str, Any],
    required_action_dict: dict[str, Any],
) -> bool:
    severity_str = str(required_action_dict.get("severity_str") or "green")
    label_str = str(required_action_dict.get("label_str") or "")
    if severity_str == "green":
        return False
    if label_str in {"Setup DB", "No state yet", "Review broker ACK", "Review reconcile"}:
        return row_dict.get("db_status_str") == "ok" and label_str == "No state yet"
    return True


def _freshness_alert_dict(
    row_dict: dict[str, Any],
    item_dict: dict[str, Any],
) -> dict[str, Any] | None:
    label_str = str(item_dict.get("label_str") or "")
    severity_str = str(item_dict.get("severity_str") or "gray")
    value_obj = item_dict.get("value_str")
    if label_str == "DIFF artifact":
        return None
    if label_str == "Norgate" and _norgate_not_blocking_current_cycle_bool(row_dict):
        return None
    if severity_str in {"red", "yellow"}:
        return _alert_dict(
            row_dict,
            "freshness",
            severity_str,
            f"{label_str} freshness",
            _freshness_item_full_detail_str(item_dict),
            "status",
        )
    if severity_str != "gray" or value_obj not in (None, ""):
        return None
    if label_str == "Pod state" and row_dict.get("db_status_str") == "ok":
        return _alert_dict(row_dict, "freshness", "gray", "Pod state missing", "No persisted POD state yet.", "status")
    if label_str in {"Broker snapshot", "Live reference"} and row_dict.get("latest_vplan_id_int") is not None:
        return _alert_dict(
            row_dict,
            "freshness",
            "gray",
            f"{label_str} missing",
            _freshness_item_full_detail_str(item_dict),
            "show_vplan",
        )
    return None


def _alert_dict(
    row_dict: dict[str, Any],
    alert_type_str: str,
    severity_str: str,
    label_str: str,
    reason_str: str,
    inspect_command_name_str: object,
) -> dict[str, Any]:
    return {
        "alert_type_str": alert_type_str,
        "severity_str": severity_str,
        "pod_id_str": row_dict.get("pod_id_str"),
        "mode_str": row_dict.get("mode_str"),
        "account_route_str": row_dict.get("account_route_str"),
        "label_str": label_str,
        "reason_str": reason_str,
        "inspect_command_name_str": inspect_command_name_str,
    }


def _freshness_item_dict(
    label_str: str,
    value_str: object,
    severity_str: str,
    detail_str: str,
    sub_detail_str_list: list[str] | None = None,
) -> dict[str, Any]:
    """Freshness item shape.

    ``detail_str`` holds the **primary** one-line summary an operator scans
    first (e.g. the gate verdict). ``sub_detail_str_list`` holds optional
    supporting bullet lines (e.g. the gate explanation, plus side facts like
    a failed raw sync status that matters for a future cycle). Callers that
    want the flat composed string for an alert message or debug story should
    use :func:`_freshness_item_full_detail_str` so primary and sub-details
    stay readable together.
    """
    return {
        "label_str": label_str,
        "value_str": value_str,
        "severity_str": severity_str,
        "detail_str": detail_str,
        "sub_detail_str_list": [str(item_obj) for item_obj in (sub_detail_str_list or [])],
    }


def _freshness_item_full_detail_str(item_dict: dict[str, Any]) -> str:
    """Flatten the structured freshness item back into a single human string.

    Used by consumers (alert builder, debug-candidate builder, CLI status
    renderer) that surface freshness items in single-line contexts where the
    structured split would be lost anyway.
    """
    detail_str = str(item_dict.get("detail_str") or "")
    sub_detail_str_list = [
        str(item_obj) for item_obj in (item_dict.get("sub_detail_str_list") or [])
    ]
    fragment_str_list = [fragment_str for fragment_str in [detail_str, *sub_detail_str_list] if fragment_str]
    return " · ".join(fragment_str_list)


def _build_debug_summary_dict(row_dict: dict[str, Any]) -> dict[str, Any]:
    candidate_dict_list = _build_debug_candidate_dict_list(row_dict)
    primary_candidate_dict = candidate_dict_list[0]
    latest_evidence_timestamp_str = (
        primary_candidate_dict.get("timestamp_str")
        or _latest_debug_evidence_timestamp_str(row_dict)
    )
    return {
        "severity_str": primary_candidate_dict["severity_str"],
        "verdict_label_str": primary_candidate_dict["label_str"],
        "primary_reason_str": primary_candidate_dict["reason_str"],
        "primary_evidence_str": primary_candidate_dict["evidence_str"],
        "next_inspect_command_name_str": _safe_inspect_command_name_str(
            primary_candidate_dict.get("inspect_command_name_str")
        ),
        "latest_evidence_timestamp_str": latest_evidence_timestamp_str,
    }


def _build_debug_story_dict(detail_dict: dict[str, Any]) -> dict[str, Any]:
    row_dict = detail_dict.get("pod_row_dict") or {}
    return {
        "verdict_dict": row_dict.get("debug_summary_dict")
        or _build_debug_summary_dict(row_dict),
        "blocker_dict_list": _build_debug_blocker_dict_list(row_dict),
        "timeline_event_dict_list": _build_debug_timeline_event_dict_list(detail_dict),
        "evidence_item_dict_list": _build_debug_evidence_item_dict_list(detail_dict),
        "recommended_command_dict_list": _build_debug_recommended_command_dict_list(
            detail_dict
        ),
    }


def _build_debug_candidate_dict_list(row_dict: dict[str, Any]) -> list[dict[str, Any]]:
    db_status_str = str(row_dict.get("db_status_str") or "")
    mode_str = str(row_dict.get("mode_str") or "")
    if db_status_str in {"missing", "empty", "error"}:
        if db_status_str == "missing":
            if mode_str == "incubation":
                return [
                    _debug_candidate_dict(
                        priority_int=10,
                        severity_str="gray",
                        label_str="Incubation DB missing",
                        reason_str="Incubation SIM ledger DB has not been created yet.",
                        evidence_str=(
                            "Dashboard reads per-POD incubation state from "
                            f"{row_dict.get('db_path_str') or 'the resolved per-POD DB path'}."
                        ),
                        inspect_command_name_str="status",
                    )
                ]
            return [
                _debug_candidate_dict(
                    priority_int=10,
                    severity_str="gray",
                    label_str="DB missing",
                    reason_str="State DB does not exist yet.",
                    evidence_str=str(row_dict.get("db_path_str") or "No DB path."),
                    inspect_command_name_str="status",
                )
            ]
        if db_status_str == "empty":
            if mode_str == "incubation":
                return [
                    _debug_candidate_dict(
                        priority_int=10,
                        severity_str="gray",
                        label_str="Incubation DB empty",
                        reason_str="Incubation DB exists, but no rehearsal cycle has been recorded yet.",
                        evidence_str=(
                            "Dashboard reads per-POD incubation state from "
                            f"{row_dict.get('db_path_str') or 'the resolved per-POD DB path'}."
                        ),
                        inspect_command_name_str="status",
                    )
                ]
            return [
                _debug_candidate_dict(
                    priority_int=10,
                    severity_str="gray",
                    label_str="DB empty",
                    reason_str="State DB exists but has no live schema rows yet.",
                    evidence_str=str(row_dict.get("db_path_str") or "No DB path."),
                    inspect_command_name_str="status",
                )
            ]
        return [
            _debug_candidate_dict(
                priority_int=10,
                severity_str="red",
                label_str="DB read error",
                reason_str="Dashboard could not read the state DB.",
                evidence_str=str(row_dict.get("error_str") or "State DB read failed."),
                inspect_command_name_str="status",
            )
        ]

    candidate_dict_list: list[dict[str, Any]] = []
    if row_dict.get("latest_decision_plan_status_str") == "blocked":
        candidate_dict_list.append(
            _debug_candidate_dict(
                priority_int=20,
                severity_str="red",
                label_str="DecisionPlan blocked",
                reason_str="The latest DecisionPlan is blocked.",
                evidence_str=f"decision_status={row_dict.get('latest_decision_plan_status_str')}",
                inspect_command_name_str="show_decision_plan",
                timestamp_str=row_dict.get("latest_decision_plan_submission_timestamp_str"),
            )
        )
    if row_dict.get("latest_vplan_status_str") == "blocked":
        label_str = "VPlan blocked"
        reason_str = "The latest VPlan is blocked."
        if row_dict.get("latest_vplan_cycle_role_str") == "previous":
            label_str = "Previous VPlan blocked"
            reason_str = "The previous execution cycle VPlan is blocked."
        candidate_dict_list.append(
            _debug_candidate_dict(
                priority_int=21,
                severity_str="red",
                label_str=label_str,
                reason_str=reason_str,
                evidence_str=f"vplan_status={row_dict.get('latest_vplan_status_str')}, vplan_id={row_dict.get('latest_vplan_id_int')}, vplan_cycle={row_dict.get('latest_vplan_cycle_role_str')}",
                inspect_command_name_str="show_vplan",
                timestamp_str=row_dict.get("latest_vplan_submission_timestamp_str"),
            )
        )
    missing_ack_count_int = int(row_dict.get("missing_ack_count_int") or 0)
    if missing_ack_count_int > 0:
        label_str = "Broker ACK missing"
        reason_str = "Broker did not acknowledge all submitted orders."
        if row_dict.get("latest_vplan_cycle_role_str") == "previous":
            label_str = "Previous broker ACK missing"
            reason_str = "Broker ACK is missing for the previous execution cycle."
        candidate_dict_list.append(
            _debug_candidate_dict(
                priority_int=30,
                severity_str="red",
                label_str=label_str,
                reason_str=reason_str,
                evidence_str=f"missing_ack_count={missing_ack_count_int}, submit_ack_status={row_dict.get('latest_submit_ack_status_str')}, vplan_cycle={row_dict.get('latest_vplan_cycle_role_str')}",
                inspect_command_name_str="show_vplan",
                timestamp_str=row_dict.get("latest_vplan_submission_timestamp_str"),
            )
        )
    if int(row_dict.get("exception_count_int") or 0) > 0:
        candidate_dict_list.append(
            _debug_candidate_dict(
                priority_int=40,
                severity_str="red",
                label_str="Reconcile blocked",
                reason_str="Latest reconciliation is not passed.",
                evidence_str=f"reconcile_status={row_dict.get('latest_reconciliation_status_str')}, reason={row_dict.get('reason_code_str')}",
                inspect_command_name_str="status",
                timestamp_str=row_dict.get("latest_reconciliation_timestamp_str"),
            )
        )
    norgate_gate_dict = _build_norgate_current_cycle_gate_dict(row_dict)
    if str(norgate_gate_dict.get("severity_str") or "") == "red":
        candidate_dict_list.append(
            _debug_candidate_dict(
                priority_int=45,
                severity_str="red",
                label_str="DecisionPlan gate blocked",
                reason_str=str(norgate_gate_dict.get("detail_str") or ""),
                evidence_str=(
                    f"provenance={norgate_gate_dict.get('decision_plan_provenance_status_str')}, "
                    f"expected_profile={norgate_gate_dict.get('expected_profile_str')}, "
                    f"decision_profile={norgate_gate_dict.get('decision_profile_str')}"
                ),
                inspect_command_name_str="show_decision_plan",
                timestamp_str=row_dict.get("latest_decision_plan_submission_timestamp_str"),
            )
        )

    diff_status_str = str(row_dict.get("latest_diff_status_str") or "not_run")
    if diff_status_str in {"red", "yellow"}:
        candidate_dict_list.append(
            _debug_candidate_dict(
                priority_int=70,
                severity_str=diff_status_str,
                label_str=f"DIFF {diff_status_str}",
                reason_str="Reference comparison has open issues.",
                evidence_str=f"status={diff_status_str}, open_issues={row_dict.get('latest_diff_open_issue_count_int')}",
                inspect_command_name_str="compare_reference",
                timestamp_str=row_dict.get("latest_diff_timestamp_str"),
            )
        )

    eod_snapshot_dict = row_dict.get("eod_snapshot_dict") or {}
    eod_status_str = str(eod_snapshot_dict.get("status_str") or "not_applicable")
    eod_severity_str = str(eod_snapshot_dict.get("severity_str") or "gray")
    if eod_status_str in {"due_missing", "blocked_by_execution"}:
        candidate_dict_list.append(
            _debug_candidate_dict(
                priority_int=80,
                severity_str=eod_severity_str if eod_severity_str in {"red", "yellow", "gray"} else "gray",
                label_str=f"EOD {eod_status_str}",
                reason_str=str(eod_snapshot_dict.get("detail_str") or "EOD snapshot needs review."),
                evidence_str=f"expected_market_date={eod_snapshot_dict.get('expected_market_date_str')}, due={eod_snapshot_dict.get('expected_due_timestamp_str')}",
                inspect_command_name_str="status",
                timestamp_str=(
                    eod_snapshot_dict.get("latest_timestamp_str")
                    or eod_snapshot_dict.get("expected_due_timestamp_str")
                ),
            )
        )

    for freshness_item_dict in (row_dict.get("data_freshness_dict") or {}).get("item_dict_list", []):
        item_severity_str = str(freshness_item_dict.get("severity_str") or "gray")
        item_label_str = str(freshness_item_dict.get("label_str") or "")
        if item_label_str == "DIFF artifact":
            continue
        if (
            item_label_str == "Norgate"
            and _norgate_not_blocking_current_cycle_bool(row_dict)
        ):
            continue
        if item_severity_str in {"red", "yellow"}:
            candidate_dict_list.append(
                _debug_candidate_dict(
                    priority_int=90,
                    severity_str=item_severity_str,
                    label_str=f"{item_label_str} freshness",
                    reason_str=_freshness_item_full_detail_str(freshness_item_dict),
                    evidence_str=f"latest={freshness_item_dict.get('value_str')}",
                    inspect_command_name_str="status",
                    timestamp_str=freshness_item_dict.get("value_str"),
                )
            )

    required_action_dict = row_dict.get("required_action_dict") or {}
    required_severity_str = str(required_action_dict.get("severity_str") or "green")
    if (
        required_severity_str in {"red", "yellow", "gray"}
        and str(required_action_dict.get("label_str") or "") != "No action"
        and not (
            required_severity_str == "red"
            and any(
                candidate_dict["severity_str"] == "red"
                for candidate_dict in candidate_dict_list
            )
        )
    ):
        candidate_dict_list.append(
            _debug_candidate_dict(
                priority_int=100,
                severity_str=required_severity_str,
                label_str=str(required_action_dict.get("label_str") or "Required action"),
                reason_str=str(required_action_dict.get("reason_str") or ""),
                evidence_str=f"next_action={row_dict.get('next_action_str')}, reason={row_dict.get('reason_code_str')}",
                inspect_command_name_str=required_action_dict.get("inspect_command_name_str"),
                timestamp_str=_latest_debug_evidence_timestamp_str(row_dict),
            )
        )

    if row_dict.get("latest_pod_state_timestamp_str") is None and row_dict.get("latest_vplan_id_int") is None:
        candidate_dict_list.append(
            _debug_candidate_dict(
                priority_int=110,
                severity_str="gray",
                label_str="No POD state",
                reason_str="No POD state or VPlan was found.",
                evidence_str=f"db_status={row_dict.get('db_status_str')}",
                inspect_command_name_str="status",
            )
        )

    if len(candidate_dict_list) == 0:
        candidate_dict_list.append(
            _debug_candidate_dict(
                priority_int=200,
                severity_str="green",
                label_str="No action",
                reason_str="No blocking dashboard condition was found.",
                evidence_str=f"next_action={row_dict.get('next_action_str')}, reason={row_dict.get('reason_code_str')}",
                inspect_command_name_str="status",
                timestamp_str=_latest_debug_evidence_timestamp_str(row_dict),
            )
        )

    return sorted(
        candidate_dict_list,
        key=lambda candidate_dict: (
            ALERT_SEVERITY_RANK_DICT.get(str(candidate_dict["severity_str"]), 99),
            int(candidate_dict["priority_int"]),
        ),
    )


def _build_debug_blocker_dict_list(row_dict: dict[str, Any]) -> list[dict[str, Any]]:
    blocker_dict_list = []
    for candidate_dict in _build_debug_candidate_dict_list(row_dict):
        if candidate_dict["severity_str"] == "green":
            continue
        blocker_dict_list.append(
            {
                "severity_str": candidate_dict["severity_str"],
                "label_str": candidate_dict["label_str"],
                "reason_str": candidate_dict["reason_str"],
                "evidence_str": candidate_dict["evidence_str"],
                "inspect_command_name_str": _safe_inspect_command_name_str(
                    candidate_dict.get("inspect_command_name_str")
                ),
                "timestamp_str": candidate_dict.get("timestamp_str"),
            }
        )
    return blocker_dict_list


def _build_debug_timeline_event_dict_list(detail_dict: dict[str, Any]) -> list[dict[str, Any]]:
    row_dict = detail_dict.get("pod_row_dict") or {}
    timeline_event_dict_list: list[dict[str, Any]] = []

    def add_event(
        source_str: str,
        label_str: str,
        status_str: object,
        severity_str: str,
        timestamp_obj: object,
        detail_str: str,
        decision_plan_id_obj: object = None,
        vplan_id_obj: object = None,
        cycle_role_str: str | None = None,
    ) -> None:
        if status_str is None and timestamp_obj is None and detail_str == "":
            return
        timeline_event_dict_list.append(
            {
                "source_str": source_str,
                "label_str": label_str,
                "status_str": status_str,
                "severity_str": severity_str if severity_str in {"green", "yellow", "red", "gray"} else "gray",
                "timestamp_str": None if timestamp_obj in (None, "") else str(timestamp_obj),
                "detail_str": detail_str,
                "decision_plan_id_int": decision_plan_id_obj,
                "vplan_id_int": vplan_id_obj,
                "cycle_role_str": cycle_role_str,
            }
        )

    add_event(
        "State",
        "POD state",
        row_dict.get("db_status_str"),
        "green" if row_dict.get("latest_pod_state_timestamp_str") else "gray",
        row_dict.get("latest_pod_state_timestamp_str"),
        str(row_dict.get("db_path_str") or ""),
    )
    add_event(
        "State",
        "Broker snapshot",
        "available" if row_dict.get("latest_broker_snapshot_timestamp_str") else "missing",
        "green" if row_dict.get("latest_broker_snapshot_timestamp_str") else "gray",
        row_dict.get("latest_broker_snapshot_timestamp_str"),
        "Latest broker account snapshot.",
    )
    norgate_snapshot_status_dict = row_dict.get("norgate_snapshot_status_dict") or {}
    norgate_timestamp_obj = (
        norgate_snapshot_status_dict.get("last_attempt_utc_str")
        or norgate_snapshot_status_dict.get("last_sync_utc_str")
        or norgate_snapshot_status_dict.get("snapshot_date_str")
    )
    norgate_detail_str = (
        f"stage={norgate_snapshot_status_dict.get('sync_stage_label_str') or '-'}, "
        f"reason={norgate_snapshot_status_dict.get('reason_code_str') or '-'}, "
        f"gate={norgate_snapshot_status_dict.get('build_gate_reason_code_str') or '-'}"
    )
    add_event(
        "Norgate",
        "Norgate sync",
        norgate_snapshot_status_dict.get("status_str"),
        str(norgate_snapshot_status_dict.get("severity_str") or "gray"),
        norgate_timestamp_obj,
        norgate_detail_str,
    )
    post_sync_failure_event_dict = _post_sync_data_load_failure_event_dict(
        list(detail_dict.get("event_dict_list") or []),
        str(
            norgate_snapshot_status_dict.get("last_sync_utc_str")
            or norgate_snapshot_status_dict.get("last_attempt_utc_str")
            or ""
        ),
    )
    if (
        str(norgate_snapshot_status_dict.get("status_str") or "") == "ready"
        and post_sync_failure_event_dict is not None
    ):
        add_event(
            "DecisionPlan",
            "Post-sync data load failed",
            "failed",
            "red",
            _event_timestamp_str(post_sync_failure_event_dict),
            str(post_sync_failure_event_dict.get("error_str") or post_sync_failure_event_dict.get("message_str") or ""),
        )
    rehearsal_status_dict = row_dict.get("rehearsal_status_dict") or {}
    if rehearsal_status_dict.get("status_str") == "active":
        add_event(
            "Rehearsal",
            "Promotion gate",
            rehearsal_status_dict.get("promotion_gate_status_str"),
            (
                "green"
                if rehearsal_status_dict.get("promotion_gate_status_str") == "complete_one_cycle"
                else "red"
                if rehearsal_status_dict.get("promotion_gate_status_str") == "blocked"
                else "yellow"
            ),
            _latest_debug_evidence_timestamp_str(row_dict),
            (
                "cycle="
                f"{rehearsal_status_dict.get('last_cycle_status_str')}, "
                "completed_cycles="
                f"{rehearsal_status_dict.get('completed_cycle_count_int')}"
            ),
        )

    decision_dict = detail_dict.get("latest_decision_plan_dict") or {}
    if decision_dict:
        add_event(
            "DecisionPlan",
            "DecisionPlan",
            decision_dict.get("status_str"),
            _status_to_severity_str(
                str(decision_dict.get("status_str") or ""),
                green_status_set={"completed"},
                yellow_status_set={"planned", "vplan_ready", "submitted", "expired"},
            ),
            decision_dict.get("submission_timestamp_str") or decision_dict.get("signal_timestamp_str"),
            f"book={decision_dict.get('decision_book_type_str')}, execute={decision_dict.get('target_execution_timestamp_str')}",
            decision_dict.get("decision_plan_id_int"),
            decision_dict.get("latest_vplan_id_int"),
            "current",
        )

    vplan_dict = detail_dict.get("latest_vplan_dict") or {}
    if vplan_dict:
        vplan_cycle_role_str = str(row_dict.get("latest_vplan_cycle_role_str") or "unknown")
        add_event(
            "VPlan",
            "VPlan",
            vplan_dict.get("status_str"),
            _status_to_severity_str(
                str(vplan_dict.get("status_str") or ""),
                green_status_set={"completed"},
                yellow_status_set={"ready", "submitting", "submitted", "expired"},
            ),
            vplan_dict.get("submission_timestamp_str"),
            f"vplan_id={vplan_dict.get('vplan_id_int')}, ack={vplan_dict.get('submit_ack_status_str')}",
            vplan_dict.get("decision_plan_id_int"),
            vplan_dict.get("vplan_id_int"),
            vplan_cycle_role_str,
        )
        for ack_row_dict in vplan_dict.get("broker_ack_row_dict_list", []):
            ack_status_str = str(ack_row_dict.get("ack_status_str") or "")
            add_event(
                "ACK",
                str(ack_row_dict.get("asset_str") or "ACK"),
                ack_status_str,
                "green" if bool(ack_row_dict.get("broker_response_ack_bool")) else "red",
                ack_row_dict.get("response_timestamp_str"),
                f"request={ack_row_dict.get('order_request_key_str')}, source={ack_row_dict.get('ack_source_str')}",
                ack_row_dict.get("decision_plan_id_int") or vplan_dict.get("decision_plan_id_int"),
                ack_row_dict.get("vplan_id_int") or vplan_dict.get("vplan_id_int"),
                vplan_cycle_role_str,
            )
        for order_row_dict in vplan_dict.get("broker_order_row_dict_list", []):
            add_event(
                "Broker order",
                str(order_row_dict.get("asset_str") or "Broker order"),
                order_row_dict.get("status_str"),
                _status_to_severity_str(
                    str(order_row_dict.get("status_str") or ""),
                    green_status_set={"Filled"},
                    yellow_status_set={"Submitted", "PendingSubmit", "PreSubmitted"},
                ),
                order_row_dict.get("last_status_timestamp_str") or order_row_dict.get("submitted_timestamp_str"),
                f"type={order_row_dict.get('broker_order_type_str')}, requested={order_row_dict.get('amount_float')}, filled={order_row_dict.get('filled_amount_float')}",
                order_row_dict.get("decision_plan_id_int") or vplan_dict.get("decision_plan_id_int"),
                order_row_dict.get("vplan_id_int") or vplan_dict.get("vplan_id_int"),
                vplan_cycle_role_str,
            )
        for fill_row_dict in vplan_dict.get("fill_row_dict_list", []):
            add_event(
                "Fill",
                str(fill_row_dict.get("asset_str") or "Fill"),
                "recorded",
                "green",
                fill_row_dict.get("fill_timestamp_str"),
                f"shares={fill_row_dict.get('fill_amount_float')}, price={fill_row_dict.get('fill_price_float')}, open={fill_row_dict.get('official_open_price_float')}",
                fill_row_dict.get("decision_plan_id_int") or vplan_dict.get("decision_plan_id_int"),
                fill_row_dict.get("vplan_id_int") or vplan_dict.get("vplan_id_int"),
                vplan_cycle_role_str,
            )

    add_event(
        "Reconcile",
        "Reconcile",
        row_dict.get("latest_reconciliation_status_str") or "none",
        _status_to_severity_str(
            str(row_dict.get("latest_reconciliation_status_str") or "none"),
            green_status_set={"passed"},
            yellow_status_set={"pending", "recorded"},
        ),
        row_dict.get("latest_reconciliation_timestamp_str"),
        str(row_dict.get("reason_code_str") or ""),
    )

    eod_snapshot_dict = row_dict.get("eod_snapshot_dict") or {}
    add_event(
        "EOD",
        "EOD snapshot",
        eod_snapshot_dict.get("status_str"),
        str(eod_snapshot_dict.get("severity_str") or "gray"),
        eod_snapshot_dict.get("latest_timestamp_str") or eod_snapshot_dict.get("expected_due_timestamp_str"),
        str(eod_snapshot_dict.get("detail_str") or ""),
    )

    add_event(
        "Live reference",
        "Live reference",
        "available" if row_dict.get("latest_live_reference_snapshot_timestamp_str") else "missing",
        "green" if row_dict.get("latest_live_reference_snapshot_timestamp_str") else "gray",
        row_dict.get("latest_live_reference_snapshot_timestamp_str"),
        "Latest quote snapshot used for VPlan sizing.",
    )

    add_event(
        "DIFF",
        "Reference DIFF",
        row_dict.get("latest_diff_status_str") or "not_run",
        _diff_freshness_severity_str(row_dict),
        row_dict.get("latest_diff_timestamp_str"),
        f"open_issues={row_dict.get('latest_diff_open_issue_count_int')}",
    )

    if row_dict.get("dtb3_download_status_str") is not None:
        add_event(
            "DTB3/FRED",
            "DTB3/FRED",
            row_dict.get("dtb3_download_status_str"),
            _dtb3_freshness_severity_str(row_dict),
            row_dict.get("dtb3_latest_observation_date_str"),
            (
                f"policy={row_dict.get('dtb3_policy_status_str') or 'legacy'}, "
                f"freshness_days={row_dict.get('dtb3_freshness_business_days_int')}, "
                f"source={row_dict.get('dtb3_source_name_str')}, "
                f"cache={row_dict.get('dtb3_used_cache_bool')}"
            ),
        )

    for event_dict in list(detail_dict.get("event_dict_list") or [])[-8:]:
        add_event(
            "Event log",
            str(event_dict.get("event_name_str") or event_dict.get("command_name_str") or "event"),
            event_dict.get("reason_code_str") or event_dict.get("status_str") or event_dict.get("severity_str"),
            _event_debug_severity_str(event_dict),
            _event_timestamp_str(event_dict),
            str(
                event_dict.get("message_str")
                or event_dict.get("error_str")
                or event_dict.get("reason_code_str")
                or ""
            ),
        )

    return sorted(
        timeline_event_dict_list,
        key=lambda event_dict: _debug_timestamp_sort_key_tuple(
            event_dict.get("timestamp_str")
        ),
        reverse=True,
    )


def _build_debug_evidence_item_dict_list(detail_dict: dict[str, Any]) -> list[dict[str, Any]]:
    row_dict = detail_dict.get("pod_row_dict") or {}
    eod_snapshot_dict = row_dict.get("eod_snapshot_dict") or {}
    rehearsal_status_dict = row_dict.get("rehearsal_status_dict") or {}
    norgate_snapshot_status_dict = row_dict.get("norgate_snapshot_status_dict") or {}
    report_dict = detail_dict.get("latest_execution_report_dict") or {}
    item_dict_list = [
        _debug_evidence_item_dict("DB", row_dict.get("db_status_str"), _status_to_severity_str(str(row_dict.get("db_status_str") or ""), {"ok"}, {"empty"}), str(row_dict.get("db_path_str") or "")),
        _debug_evidence_item_dict(
            "Norgate sync",
            norgate_snapshot_status_dict.get("status_str"),
            str(norgate_snapshot_status_dict.get("severity_str") or "gray"),
            (
                f"stage={norgate_snapshot_status_dict.get('sync_stage_label_str') or '-'}, "
                f"gate={norgate_snapshot_status_dict.get('build_gate_reason_code_str') or '-'}"
            ),
        ),
        _debug_evidence_item_dict("Pod state", row_dict.get("latest_pod_state_timestamp_str"), "green" if row_dict.get("latest_pod_state_timestamp_str") else "gray", "Latest persisted sleeve state."),
        _debug_evidence_item_dict("Broker snapshot", row_dict.get("latest_broker_snapshot_timestamp_str"), "green" if row_dict.get("latest_broker_snapshot_timestamp_str") else "gray", "Broker truth source for account cash and positions."),
        _debug_evidence_item_dict("Live reference", row_dict.get("latest_live_reference_snapshot_timestamp_str"), "green" if row_dict.get("latest_live_reference_snapshot_timestamp_str") else "gray", "Quote snapshot used for VPlan sizing."),
        _debug_evidence_item_dict("Reconcile", row_dict.get("latest_reconciliation_status_str"), _status_to_severity_str(str(row_dict.get("latest_reconciliation_status_str") or ""), {"passed"}, {"pending", "recorded"}), str(row_dict.get("latest_reconciliation_timestamp_str") or "")),
        _debug_evidence_item_dict("EOD snapshot", eod_snapshot_dict.get("status_str"), str(eod_snapshot_dict.get("severity_str") or "gray"), str(eod_snapshot_dict.get("detail_str") or "")),
        _debug_evidence_item_dict("DIFF artifact", row_dict.get("latest_diff_status_str"), _diff_freshness_severity_str(row_dict), f"timestamp={row_dict.get('latest_diff_timestamp_str')}, open_issues={row_dict.get('latest_diff_open_issue_count_int')}"),
    ]
    if rehearsal_status_dict.get("status_str") == "active":
        item_dict_list.append(
            _debug_evidence_item_dict(
                "Rehearsal gate",
                rehearsal_status_dict.get("promotion_gate_status_str"),
                (
                    "green"
                    if rehearsal_status_dict.get("promotion_gate_status_str") == "complete_one_cycle"
                    else "red"
                    if rehearsal_status_dict.get("promotion_gate_status_str") == "blocked"
                    else "yellow"
                ),
                (
                    "SIM ledger truth; paper_probe="
                    f"{rehearsal_status_dict.get('paper_probe_status_str')}; "
                    "Paper probe is evidence only; it does not count as SIM ledger P&L."
                ),
            )
        )
    if row_dict.get("dtb3_download_status_str") is not None:
        item_dict_list.append(
            _debug_evidence_item_dict(
                "DTB3/FRED",
                row_dict.get("dtb3_download_status_str"),
                _dtb3_freshness_severity_str(row_dict),
                (
                    f"policy={row_dict.get('dtb3_policy_status_str') or 'legacy'}, "
                    f"observation={row_dict.get('dtb3_latest_observation_date_str')}, "
                    f"days={row_dict.get('dtb3_freshness_business_days_int')}, "
                    f"cache={row_dict.get('dtb3_used_cache_bool')}"
                ),
            )
        )
    if report_dict:
        item_dict_list.extend(
            [
                _debug_evidence_item_dict(
                    "Model residual rows",
                    report_dict.get("residual_count_int"),
                    "red" if int(report_dict.get("residual_count_int") or 0) > 0 else "green",
                    "Rows where target shares and broker shares differ.",
                ),
                _debug_evidence_item_dict(
                    "Fills",
                    report_dict.get("fill_count_int"),
                    "green" if int(report_dict.get("fill_count_int") or 0) > 0 else "gray",
                    f"official_open_coverage={report_dict.get('fill_with_official_open_count_int')} / {report_dict.get('fill_count_int')}",
                ),
                _debug_evidence_item_dict(
                    "Reference slippage",
                    report_dict.get("vplan_reference_slippage_bps_float"),
                    "gray" if report_dict.get("vplan_reference_slippage_bps_float") is None else "yellow",
                    f"notional={report_dict.get('vplan_reference_slippage_notional_float')}",
                ),
            ]
        )
    return item_dict_list


def _build_debug_recommended_command_dict_list(detail_dict: dict[str, Any]) -> list[dict[str, str]]:
    row_dict = detail_dict.get("pod_row_dict") or {}
    command_name_list: list[str] = []
    primary_command_str = _safe_inspect_command_name_str(
        (row_dict.get("debug_summary_dict") or {}).get("next_inspect_command_name_str")
    )
    for command_name_str in (
        primary_command_str,
        "status",
        "next_due" if row_dict.get("next_action_str") in {"build_decision_plan", "build_vplan"} else None,
        "show_decision_plan" if detail_dict.get("latest_decision_plan_dict") is not None else None,
        "show_vplan" if detail_dict.get("latest_vplan_dict") is not None else None,
        "compare_reference" if str(row_dict.get("latest_diff_status_str") or "not_run") != "not_run" else None,
    ):
        safe_command_str = _safe_inspect_command_name_str(command_name_str)
        if safe_command_str is not None and safe_command_str not in command_name_list:
            command_name_list.append(safe_command_str)
    return [
        {
            "command_name_str": command_name_str,
            "reason_str": _debug_command_reason_str(command_name_str),
        }
        for command_name_str in command_name_list
    ]


def _debug_candidate_dict(
    *,
    priority_int: int,
    severity_str: str,
    label_str: str,
    reason_str: str,
    evidence_str: str,
    inspect_command_name_str: object,
    timestamp_str: object = None,
) -> dict[str, Any]:
    return {
        "priority_int": priority_int,
        "severity_str": severity_str if severity_str in {"green", "yellow", "red", "gray"} else "gray",
        "label_str": label_str,
        "reason_str": reason_str,
        "evidence_str": evidence_str,
        "inspect_command_name_str": _safe_inspect_command_name_str(inspect_command_name_str),
        "timestamp_str": None if timestamp_str in (None, "") else str(timestamp_str),
    }


def _debug_evidence_item_dict(
    label_str: str,
    value_obj: object,
    severity_str: str,
    detail_str: str,
) -> dict[str, Any]:
    return {
        "label_str": label_str,
        "value_str": value_obj,
        "severity_str": severity_str if severity_str in {"green", "yellow", "red", "gray"} else "gray",
        "detail_str": detail_str,
    }


def _safe_inspect_command_name_str(command_name_obj: object) -> str | None:
    if command_name_obj is None:
        return None
    command_name_str = str(command_name_obj)
    if command_name_str in SAFE_INSPECT_COMMAND_NAME_LIST:
        return command_name_str
    return "status"


def _debug_command_reason_str(command_name_str: str) -> str:
    reason_map_dict = {
        "status": "Inspect current POD state and blocker reason.",
        "next_due": "Check scheduler timing without changing live state.",
        "show_decision_plan": "Inspect latest strategy decision intent.",
        "show_vplan": "Inspect VPlan, ACK, order, and fill evidence.",
        "compare_reference": "Inspect live/reference drift artifact.",
    }
    return reason_map_dict.get(command_name_str, "Read-only inspect command.")


def _event_debug_severity_str(event_dict: dict[str, Any]) -> str:
    level_text_str = str(
        event_dict.get("severity_str")
        or event_dict.get("level_str")
        or event_dict.get("status_str")
        or event_dict.get("reason_code_str")
        or ""
    ).lower()
    if any(token_str in level_text_str for token_str in ("error", "fail", "critical", "blocked")):
        return "red"
    if any(token_str in level_text_str for token_str in ("warn", "parked", "waiting")):
        return "yellow"
    return "green" if level_text_str else "gray"


def _latest_debug_evidence_timestamp_str(row_dict: dict[str, Any]) -> str | None:
    eod_snapshot_dict = row_dict.get("eod_snapshot_dict") or {}
    return _latest_timestamp_str_from_value_list(
        [
            row_dict.get("latest_pod_state_timestamp_str"),
            row_dict.get("latest_broker_snapshot_timestamp_str"),
            row_dict.get("latest_live_reference_snapshot_timestamp_str"),
            row_dict.get("latest_event_timestamp_str"),
            row_dict.get("latest_reconciliation_timestamp_str"),
            row_dict.get("latest_diff_timestamp_str"),
            eod_snapshot_dict.get("latest_timestamp_str"),
            eod_snapshot_dict.get("expected_due_timestamp_str"),
            row_dict.get("dtb3_latest_observation_date_str"),
        ]
    )


def _latest_timestamp_str_from_value_list(value_list: list[object]) -> str | None:
    parsed_tuple_list = [
        (_debug_timestamp_sort_key_tuple(value_obj), str(value_obj))
        for value_obj in value_list
        if value_obj not in (None, "")
    ]
    if len(parsed_tuple_list) == 0:
        return None
    return max(parsed_tuple_list, key=lambda item_tuple: item_tuple[0])[1]


def _debug_timestamp_sort_key_tuple(value_obj: object) -> tuple[int, datetime]:
    if value_obj in (None, ""):
        return (0, datetime.min.replace(tzinfo=UTC))
    value_str = str(value_obj)
    try:
        if len(value_str) == 16 and value_str.endswith("Z") and "T" in value_str:
            return (1, datetime.strptime(value_str, "%Y%m%dT%H%M%SZ").replace(tzinfo=UTC))
        return (1, _parse_timestamp_ts(value_str))
    except ValueError:
        return (0, datetime.min.replace(tzinfo=UTC))


def _diff_freshness_severity_str(row_dict: dict[str, Any]) -> str:
    diff_status_str = str(row_dict.get("latest_diff_status_str") or "not_run")
    if diff_status_str in {"green", "yellow", "red"}:
        return diff_status_str
    if diff_status_str == "not_run":
        return "gray"
    return "red"


def _dtb3_freshness_severity_str(row_dict: dict[str, Any]) -> str:
    policy_status_str = str(row_dict.get("dtb3_policy_status_str") or "")
    warning_obj = row_dict.get("dtb3_warning_bool")
    warning_bool = warning_obj is True or str(warning_obj).lower() == "true"
    used_cache_obj = row_dict.get("dtb3_used_cache_bool")
    used_cache_bool = used_cache_obj is True or str(used_cache_obj).lower() == "true"
    download_status_str = str(row_dict.get("dtb3_download_status_str") or "")
    if policy_status_str in {"stale_warning", "cache_warning", "cache_write_warning"} or warning_bool:
        return "yellow"
    if download_status_str == "download_success_cache_write_failed":
        return "yellow"
    if used_cache_bool or download_status_str.startswith("cache_"):
        return "yellow"

    freshness_obj = row_dict.get("dtb3_freshness_business_days_int")
    try:
        freshness_days_int = int(freshness_obj)
    except (TypeError, ValueError):
        return "gray"
    try:
        warn_after_business_days_int = int(
            row_dict.get("dtb3_warn_after_business_days_int")
            or LIVE_FRED_STALE_WARNING_BUSINESS_DAYS_INT
        )
    except (TypeError, ValueError):
        warn_after_business_days_int = LIVE_FRED_STALE_WARNING_BUSINESS_DAYS_INT
    return "green" if freshness_days_int <= warn_after_business_days_int else "yellow"


def _build_execution_report_from_vplan_dict(
    vplan_dict: dict[str, Any],
    broker_position_map_dict: dict[str, Any] | None = None,
) -> dict[str, Any]:
    broker_position_map_dict = {} if broker_position_map_dict is None else broker_position_map_dict
    fill_row_dict_list = runner._enrich_fill_row_dict_list(list(vplan_dict.get("fill_row_dict_list", [])))
    vplan_reference_price_map_dict = _json_map_dict(vplan_dict.get("live_reference_price_json_str"))
    target_share_map_dict = _json_map_dict(vplan_dict.get("target_share_json_str"))
    current_share_map_dict = _json_map_dict(vplan_dict.get("current_broker_position_json_str"))
    order_delta_map_dict = _json_map_dict(vplan_dict.get("order_delta_json_str"))
    vplan_row_map_dict = {
        str(row_dict.get("asset_str")): row_dict
        for row_dict in vplan_dict.get("vplan_row_dict_list", [])
    }
    latest_broker_order_map_dict: dict[str, dict[str, Any]] = {}
    for broker_order_row_dict in vplan_dict.get("broker_order_row_dict_list", []):
        asset_str = str(broker_order_row_dict.get("asset_str"))
        latest_broker_order_map_dict[asset_str] = broker_order_row_dict

    execution_row_dict_list = _build_dashboard_execution_row_dict_list(
        vplan_reference_price_map_dict=vplan_reference_price_map_dict,
        target_share_map_dict=target_share_map_dict,
        current_share_map_dict=current_share_map_dict,
        order_delta_map_dict=order_delta_map_dict,
        broker_position_map_dict=broker_position_map_dict,
        vplan_row_map_dict=vplan_row_map_dict,
        latest_broker_order_map_dict=latest_broker_order_map_dict,
        fill_row_dict_list=fill_row_dict_list,
    )
    official_open_slippage_dict = _aggregate_official_open_slippage_dict(fill_row_dict_list)
    vplan_reference_slippage_dict = _aggregate_vplan_reference_slippage_dict(
        execution_row_dict_list
    )
    return {
        "pod_id_str": vplan_dict.get("pod_id_str"),
        "latest_vplan_id_int": vplan_dict.get("vplan_id_int"),
        "fill_count_int": len(fill_row_dict_list),
        "fill_with_official_open_count_int": sum(
            1 for fill_row_dict in fill_row_dict_list if fill_row_dict.get("official_open_price_float") is not None
        ),
        "broker_order_count_int": len(vplan_dict.get("broker_order_row_dict_list", [])),
        "broker_ack_count_int": len(vplan_dict.get("broker_ack_row_dict_list", [])),
        "residual_count_int": sum(
            1 for execution_row_dict in execution_row_dict_list if bool(execution_row_dict["unresolved_bool"])
        ),
        "official_open_slippage_bps_float": official_open_slippage_dict["slippage_bps_float"],
        "official_open_slippage_notional_float": official_open_slippage_dict["slippage_notional_float"],
        "vplan_reference_slippage_bps_float": vplan_reference_slippage_dict["slippage_bps_float"],
        "vplan_reference_slippage_notional_float": vplan_reference_slippage_dict["slippage_notional_float"],
        "aggregate_official_open_slippage_notional_float": official_open_slippage_dict["slippage_notional_float"],
        "aggregate_vplan_reference_slippage_notional_float": vplan_reference_slippage_dict["slippage_notional_float"],
        "fill_row_dict_list": fill_row_dict_list,
        "execution_row_dict_list": execution_row_dict_list,
    }


def _build_dashboard_execution_row_dict_list(
    *,
    vplan_reference_price_map_dict: dict[str, Any],
    target_share_map_dict: dict[str, Any],
    current_share_map_dict: dict[str, Any],
    order_delta_map_dict: dict[str, Any],
    broker_position_map_dict: dict[str, Any],
    vplan_row_map_dict: dict[str, dict[str, Any]],
    latest_broker_order_map_dict: dict[str, dict[str, Any]],
    fill_row_dict_list: list[dict[str, Any]],
    tolerance_float: float = 1e-9,
) -> list[dict[str, Any]]:
    fill_stat_map_dict = runner._build_fill_stat_by_asset_map_dict(fill_row_dict_list)
    asset_str_set = (
        set(vplan_reference_price_map_dict)
        | set(target_share_map_dict)
        | set(current_share_map_dict)
        | set(order_delta_map_dict)
        | set(broker_position_map_dict)
        | set(vplan_row_map_dict)
        | set(latest_broker_order_map_dict)
        | set(fill_stat_map_dict)
    )
    execution_row_dict_list: list[dict[str, Any]] = []
    for asset_str in sorted(str(asset_obj) for asset_obj in asset_str_set):
        fill_stat_dict = fill_stat_map_dict.get(asset_str, {})
        vplan_row_dict = vplan_row_map_dict.get(asset_str, {})
        planned_share_float = _optional_float(vplan_row_dict.get("order_delta_share_float"))
        if planned_share_float is None:
            planned_share_float = _optional_float(order_delta_map_dict.get(asset_str)) or 0.0
        filled_share_float = float(fill_stat_dict.get("filled_share_float") or 0.0)
        current_share_float = _optional_float(vplan_row_dict.get("current_share_float"))
        if current_share_float is None:
            current_share_float = _optional_float(current_share_map_dict.get(asset_str)) or 0.0
        target_share_float = _optional_float(vplan_row_dict.get("target_share_float"))
        if target_share_float is None:
            target_share_float = _optional_float(target_share_map_dict.get(asset_str))
        if target_share_float is None:
            target_share_float = current_share_float
        broker_share_float = _optional_float(broker_position_map_dict.get(asset_str)) or 0.0
        residual_share_float = target_share_float - broker_share_float
        fill_price_float = _optional_float(fill_stat_dict.get("weighted_avg_fill_price_float"))
        official_open_price_float = _optional_float(fill_stat_dict.get("official_open_price_float"))
        vplan_reference_price_float = _optional_float(vplan_reference_price_map_dict.get(asset_str))
        if vplan_reference_price_float is None:
            vplan_reference_price_float = _optional_float(vplan_row_dict.get("live_reference_price_float"))
        direction_share_float = planned_share_float
        if abs(direction_share_float) <= tolerance_float:
            direction_share_float = filled_share_float
        side_str = _execution_side_str(direction_share_float)
        official_open_slippage_dict = _benchmark_slippage_dict(
            fill_amount_float=filled_share_float,
            fill_price_float=fill_price_float,
            benchmark_price_float=official_open_price_float,
        )
        vplan_reference_slippage_dict = _benchmark_slippage_dict(
            fill_amount_float=filled_share_float,
            fill_price_float=fill_price_float,
            benchmark_price_float=vplan_reference_price_float,
        )
        latest_broker_order_row_dict = latest_broker_order_map_dict.get(asset_str)
        execution_row_dict_list.append(
            {
                "asset_str": asset_str,
                "side_str": side_str,
                "planned_order_delta_share_float": planned_share_float,
                "filled_share_float": filled_share_float,
                "current_share_float": current_share_float,
                "target_share_float": target_share_float,
                "broker_share_float": broker_share_float,
                "residual_share_float": residual_share_float,
                "unresolved_bool": abs(residual_share_float) > tolerance_float,
                "fill_price_float": fill_price_float,
                "official_open_price_float": official_open_price_float,
                "vplan_reference_price_float": vplan_reference_price_float,
                "official_open_slippage_bps_float": official_open_slippage_dict["slippage_bps_float"],
                "official_open_slippage_notional_float": official_open_slippage_dict["slippage_notional_float"],
                "vplan_reference_slippage_bps_float": vplan_reference_slippage_dict["slippage_bps_float"],
                "vplan_reference_slippage_notional_float": vplan_reference_slippage_dict["slippage_notional_float"],
                "latest_broker_order_status_str": (
                    None if latest_broker_order_row_dict is None else latest_broker_order_row_dict.get("status_str")
                ),
            }
        )
    return execution_row_dict_list


def _optional_float(value_obj: Any) -> float | None:
    if value_obj is None or value_obj == "":
        return None
    try:
        return float(value_obj)
    except (TypeError, ValueError):
        return None


def build_position_exposure_dict_list(
    position_map_dict: dict[str, Any] | None,
    price_map_dict: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Value a single pod's current positions for the cross-pod exposure view.

    ``position_map_dict`` is ``asset_str -> shares`` (``+`` long / ``-`` short,
    from pod_state/broker ``position_json_str``). ``price_map_dict`` is
    ``asset_str -> price`` (latest VPlan ``live_reference_price_json_str``).
    Zero-ish positions are skipped. An asset with shares but no usable price is
    kept as ``is_priced_bool=False`` (surfaced as "unpriced", never dropped).
    All values are USD. Read-only — no quant logic.
    """
    exposure_dict_list: list[dict[str, Any]] = []
    for asset_obj, share_obj in (position_map_dict or {}).items():
        share_float = _optional_float(share_obj)
        if share_float is None or abs(share_float) <= 1e-9:
            continue
        price_float = _optional_float((price_map_dict or {}).get(asset_obj))
        is_priced_bool = price_float is not None and price_float > 0
        market_value_float = share_float * price_float if is_priced_bool else None
        exposure_dict_list.append(
            {
                "asset_str": str(asset_obj),
                "share_float": share_float,
                "price_float": price_float if is_priced_bool else None,
                "market_value_float": market_value_float,
                "is_priced_bool": is_priced_bool,
            }
        )
    exposure_dict_list.sort(
        key=lambda item_dict: abs(item_dict["market_value_float"])
        if item_dict["market_value_float"] is not None
        else 0.0,
        reverse=True,
    )
    return exposure_dict_list


def _attach_position_exposure_fields(
    base_row_dict: dict[str, Any],
    position_map_dict: dict[str, Any] | None,
    price_map_dict: dict[str, Any] | None,
) -> None:
    exposure_dict_list = build_position_exposure_dict_list(position_map_dict, price_map_dict)
    base_row_dict["position_exposure_dict_list"] = exposure_dict_list
    base_row_dict["position_unpriced_count_int"] = sum(
        1 for item_dict in exposure_dict_list if not item_dict["is_priced_bool"]
    )


def _benchmark_slippage_dict(
    *,
    fill_amount_float: float,
    fill_price_float: float | None,
    benchmark_price_float: float | None,
) -> dict[str, float | None]:
    if fill_price_float is None or benchmark_price_float is None:
        return {"slippage_bps_float": None, "slippage_notional_float": None}
    if float(benchmark_price_float) <= 0.0:
        return {"slippage_bps_float": None, "slippage_notional_float": None}
    signed_direction_float = runner._signed_execution_direction_float_from_amount(fill_amount_float)
    if signed_direction_float is None:
        return {"slippage_bps_float": None, "slippage_notional_float": None}
    slippage_share_float = signed_direction_float * (
        float(fill_price_float) - float(benchmark_price_float)
    )
    return {
        "slippage_bps_float": runner._compute_execution_cost_bps_float(
            execution_price_float=float(fill_price_float),
            official_open_price_float=float(benchmark_price_float),
            signed_execution_direction_float=signed_direction_float,
        ),
        "slippage_notional_float": abs(float(fill_amount_float)) * slippage_share_float,
    }


def _aggregate_official_open_slippage_dict(
    fill_row_dict_list: list[dict[str, Any]],
) -> dict[str, float | None]:
    slippage_notional_float = 0.0
    benchmark_notional_float = 0.0
    for fill_row_dict in fill_row_dict_list:
        if fill_row_dict.get("slippage_notional_float") is None:
            continue
        official_open_price_float = _optional_float(fill_row_dict.get("official_open_price_float"))
        if official_open_price_float is None or official_open_price_float <= 0.0:
            continue
        fill_amount_float = float(fill_row_dict.get("fill_amount_float") or 0.0)
        slippage_notional_float += float(fill_row_dict["slippage_notional_float"])
        benchmark_notional_float += abs(fill_amount_float) * official_open_price_float
    return _aggregate_slippage_dict(
        slippage_notional_float=slippage_notional_float,
        benchmark_notional_float=benchmark_notional_float,
    )


def _aggregate_vplan_reference_slippage_dict(
    execution_row_dict_list: list[dict[str, Any]],
) -> dict[str, float | None]:
    slippage_notional_float = 0.0
    benchmark_notional_float = 0.0
    for execution_row_dict in execution_row_dict_list:
        if execution_row_dict.get("vplan_reference_slippage_notional_float") is None:
            continue
        vplan_reference_price_float = _optional_float(
            execution_row_dict.get("vplan_reference_price_float")
        )
        if vplan_reference_price_float is None or vplan_reference_price_float <= 0.0:
            continue
        filled_share_float = float(execution_row_dict.get("filled_share_float") or 0.0)
        slippage_notional_float += float(
            execution_row_dict["vplan_reference_slippage_notional_float"]
        )
        benchmark_notional_float += abs(filled_share_float) * vplan_reference_price_float
    return _aggregate_slippage_dict(
        slippage_notional_float=slippage_notional_float,
        benchmark_notional_float=benchmark_notional_float,
    )


def _aggregate_slippage_dict(
    *,
    slippage_notional_float: float,
    benchmark_notional_float: float,
) -> dict[str, float | None]:
    if benchmark_notional_float <= 0.0:
        return {"slippage_bps_float": None, "slippage_notional_float": None}
    return {
        "slippage_bps_float": round(
            10000.0 * slippage_notional_float / benchmark_notional_float,
            10,
        ),
        "slippage_notional_float": slippage_notional_float,
    }


def _execution_side_str(amount_float: float) -> str:
    if amount_float > 0.0:
        return "buy"
    if amount_float < 0.0:
        return "sell"
    return "flat"


def _build_decision_plan_detail_dict(
    decision_plan_row_dict: dict[str, Any],
    latest_vplan_row_dict: dict[str, Any] | None,
) -> dict[str, Any]:
    entry_target_weight_map_dict = _json_map_dict(
        decision_plan_row_dict.get("entry_target_weight_json_str")
    )
    full_target_weight_map_dict = _json_map_dict(
        decision_plan_row_dict.get("full_target_weight_json_str")
    )
    target_weight_map_dict = _json_map_dict(decision_plan_row_dict.get("target_weight_json_str"))
    decision_book_type_str = str(
        decision_plan_row_dict.get("decision_book_type_str") or "incremental_entry_exit_book"
    )
    if decision_book_type_str == "full_target_weight_book":
        display_target_weight_map_dict = (
            full_target_weight_map_dict
            if len(full_target_weight_map_dict) > 0
            else target_weight_map_dict
        )
    else:
        display_target_weight_map_dict = (
            entry_target_weight_map_dict
            if len(entry_target_weight_map_dict) > 0
            else target_weight_map_dict
        )
    return {
        "decision_plan_id_int": decision_plan_row_dict.get("decision_plan_id_int"),
        "release_id_str": decision_plan_row_dict.get("release_id_str"),
        "user_id_str": decision_plan_row_dict.get("user_id_str"),
        "pod_id_str": decision_plan_row_dict.get("pod_id_str"),
        "account_route_str": decision_plan_row_dict.get("account_route_str"),
        "status_str": decision_plan_row_dict.get("status_str"),
        "decision_book_type_str": decision_book_type_str,
        "signal_timestamp_str": decision_plan_row_dict.get("signal_timestamp_str"),
        "submission_timestamp_str": decision_plan_row_dict.get("submission_timestamp_str"),
        "target_execution_timestamp_str": decision_plan_row_dict.get(
            "target_execution_timestamp_str"
        ),
        "execution_policy_str": decision_plan_row_dict.get("execution_policy_str"),
        "target_weight_map_dict": target_weight_map_dict,
        "entry_target_weight_map_dict": entry_target_weight_map_dict,
        "full_target_weight_map_dict": full_target_weight_map_dict,
        "display_target_weight_map_dict": display_target_weight_map_dict,
        "exit_asset_list": _json_list(decision_plan_row_dict.get("exit_asset_json_str")),
        "entry_priority_list": _json_list(decision_plan_row_dict.get("entry_priority_json_str")),
        "decision_base_position_map_dict": _json_map_dict(
            decision_plan_row_dict.get("decision_base_position_json_str")
        ),
        "snapshot_metadata_dict": _json_map_dict(
            decision_plan_row_dict.get("snapshot_metadata_json_str")
        ),
        "strategy_state_dict": _json_map_dict(decision_plan_row_dict.get("strategy_state_json_str")),
        "cash_reserve_weight_float": decision_plan_row_dict.get("cash_reserve_weight_float"),
        "preserve_untouched_positions_bool": bool(
            decision_plan_row_dict.get("preserve_untouched_positions_bool")
        ),
        "rebalance_omitted_assets_to_zero_bool": bool(
            decision_plan_row_dict.get("rebalance_omitted_assets_to_zero_bool")
        ),
        "latest_vplan_id_int": None
        if latest_vplan_row_dict is None
        else latest_vplan_row_dict.get("vplan_id_int"),
        "latest_vplan_status_str": None
        if latest_vplan_row_dict is None
        else latest_vplan_row_dict.get("status_str"),
    }


def _connect_readonly_existing_db(db_path_obj: Path) -> sqlite3.Connection:
    db_uri_path_str = quote(db_path_obj.resolve().as_posix(), safe="/:")
    connection_obj = sqlite3.connect(f"file:{db_uri_path_str}?mode=ro", uri=True)
    connection_obj.row_factory = sqlite3.Row
    return connection_obj


def _table_exists_bool(connection_obj: sqlite3.Connection, table_name_str: str) -> bool:
    row_obj = connection_obj.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table_name_str,),
    ).fetchone()
    return row_obj is not None


def _fetch_latest_decision_plan_row_dict(
    connection_obj: sqlite3.Connection,
    pod_id_str: str,
) -> dict[str, Any] | None:
    if not _table_exists_bool(connection_obj, "decision_plan"):
        return None
    return _fetch_one_dict(
        connection_obj,
        """
        SELECT *
        FROM decision_plan
        WHERE pod_id_str = ?
        ORDER BY decision_plan_id_int DESC
        LIMIT 1
        """,
        (pod_id_str,),
    )


def _fetch_latest_vplan_row_dict(
    connection_obj: sqlite3.Connection,
    pod_id_str: str,
) -> dict[str, Any] | None:
    if not _table_exists_bool(connection_obj, "vplan"):
        return None
    return _fetch_one_dict(
        connection_obj,
        """
        SELECT *
        FROM vplan
        WHERE pod_id_str = ?
        ORDER BY vplan_id_int DESC
        LIMIT 1
        """,
        (pod_id_str,),
    )


def _fetch_latest_vplan_for_decision_row_dict(
    connection_obj: sqlite3.Connection,
    decision_plan_id_int: int,
) -> dict[str, Any] | None:
    if not _table_exists_bool(connection_obj, "vplan"):
        return None
    return _fetch_one_dict(
        connection_obj,
        """
        SELECT *
        FROM vplan
        WHERE decision_plan_id_int = ?
        ORDER BY vplan_id_int DESC
        LIMIT 1
        """,
        (int(decision_plan_id_int),),
    )


def _fetch_latest_reconciliation_row_dict(
    connection_obj: sqlite3.Connection,
    pod_id_str: str,
) -> dict[str, Any] | None:
    if not _table_exists_bool(connection_obj, "vplan_reconciliation_snapshot"):
        return None
    return _fetch_one_dict(
        connection_obj,
        """
        SELECT *
        FROM vplan_reconciliation_snapshot
        WHERE pod_id_str = ?
        ORDER BY vplan_reconciliation_snapshot_id_int DESC
        LIMIT 1
        """,
        (pod_id_str,),
    )


def _fetch_one_dict(
    connection_obj: sqlite3.Connection,
    sql_str: str,
    param_tuple: tuple[Any, ...],
) -> dict[str, Any] | None:
    row_obj = connection_obj.execute(sql_str, param_tuple).fetchone()
    return None if row_obj is None else _row_to_dict(row_obj)


def _fetch_all_dict_list(
    connection_obj: sqlite3.Connection,
    sql_str: str,
    param_tuple: tuple[Any, ...],
) -> list[dict[str, Any]]:
    table_name_str = _extract_table_name_str(sql_str)
    if table_name_str is not None and not _table_exists_bool(connection_obj, table_name_str):
        return []
    return [_row_to_dict(row_obj) for row_obj in connection_obj.execute(sql_str, param_tuple).fetchall()]


def _fetch_count_int(
    connection_obj: sqlite3.Connection,
    table_name_str: str,
    where_sql_str: str,
    param_tuple: tuple[Any, ...],
) -> int:
    if not _table_exists_bool(connection_obj, table_name_str):
        return 0
    row_obj = connection_obj.execute(
        f"SELECT COUNT(*) AS count_int FROM {table_name_str} WHERE {where_sql_str}",
        param_tuple,
    ).fetchone()
    if row_obj is None:
        return 0
    return int(row_obj["count_int"])


def _extract_table_name_str(sql_str: str) -> str | None:
    token_list = sql_str.replace("\n", " ").split()
    lower_token_list = [token_str.lower() for token_str in token_list]
    if "from" not in lower_token_list:
        return None
    from_idx_int = lower_token_list.index("from")
    if from_idx_int + 1 >= len(token_list):
        return None
    return token_list[from_idx_int + 1]


def _row_to_dict(row_obj: sqlite3.Row) -> dict[str, Any]:
    return {key_str: row_obj[key_str] for key_str in row_obj.keys()}


def _json_map_dict(raw_json_obj: object) -> dict[str, Any]:
    if raw_json_obj is None:
        return {}
    try:
        parsed_obj = json.loads(str(raw_json_obj))
    except json.JSONDecodeError:
        return {}
    return parsed_obj if isinstance(parsed_obj, dict) else {}


def _json_list(raw_json_obj: object) -> list[Any]:
    if raw_json_obj is None:
        return []
    try:
        parsed_obj = json.loads(str(raw_json_obj))
    except json.JSONDecodeError:
        return []
    return parsed_obj if isinstance(parsed_obj, list) else []


def _parse_timestamp_ts(timestamp_str: str) -> datetime:
    timestamp_ts = datetime.fromisoformat(str(timestamp_str))
    if timestamp_ts.tzinfo is None:
        return timestamp_ts.replace(tzinfo=UTC)
    return timestamp_ts


def _artifact_url_str(results_root_path_str: str, artifact_path_obj: Path) -> str:
    results_root_path_obj = Path(results_root_path_str).resolve()
    relative_path_obj = artifact_path_obj.resolve().relative_to(results_root_path_obj)
    return "/artifacts/" + relative_path_obj.as_posix()


def _is_relative_to_bool(path_obj: Path, parent_path_obj: Path) -> bool:
    try:
        path_obj.relative_to(parent_path_obj)
    except ValueError:
        return False
    return True




def _jsonable_obj(value_obj: Any) -> Any:
    if isinstance(value_obj, dict):
        return {str(key_obj): _jsonable_obj(item_obj) for key_obj, item_obj in value_obj.items()}
    if isinstance(value_obj, list):
        return [_jsonable_obj(item_obj) for item_obj in value_obj]
    if isinstance(value_obj, tuple):
        return [_jsonable_obj(item_obj) for item_obj in value_obj]
    if isinstance(value_obj, datetime):
        return value_obj.isoformat()
    if isinstance(value_obj, Path):
        return str(value_obj)
    return value_obj
