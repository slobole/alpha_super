from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path
import sqlite3
import threading
import traceback
from typing import Any, Callable
from urllib.parse import quote, unquote, urlparse
import uuid

import yaml

from alpha.live import runner, scheduler_utils
from alpha.live.models import LiveRelease
from alpha.live.release_manifest import load_release_list
from alpha.live.state_store_v2 import LiveStateStore


DEFAULT_DASHBOARD_HOST_STR = "127.0.0.1"
DEFAULT_DASHBOARD_PORT_INT = 8765
DEFAULT_RELEASES_ROOT_PATH_STR = str(Path(__file__).resolve().parent / "releases")
DEFAULT_CONFIG_PATH_STR = str(Path(__file__).resolve().parent / "dashboard_config.yaml")
DEFAULT_RESULTS_ROOT_PATH_STR = "results"
DEFAULT_EVENT_LOG_PATH_STR = str(Path(__file__).resolve().parent / "logs" / "live_events.jsonl")
DEFAULT_EVENT_LIMIT_INT = 80
ALERT_SEVERITY_RANK_DICT = {"red": 0, "yellow": 1, "gray": 2, "green": 3}
SAFE_INSPECT_COMMAND_NAME_LIST = [
    "status",
    "next_due",
    "show_decision_plan",
    "show_vplan",
    "compare_reference",
]


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
    ):
        self._diff_runner_fn = diff_runner_fn
        self._job_map_dict: dict[str, DashboardJob] = {}
        self._lock_obj = threading.Lock()

    def start_job(
        self,
        pod_target_obj: DashboardPodTarget,
        releases_root_path_str: str,
        results_root_path_str: str,
    ) -> DashboardJob:
        now_ts = datetime.now(UTC)
        job_obj = DashboardJob(
            job_id_str=uuid.uuid4().hex,
            pod_id_str=pod_target_obj.release_obj.pod_id_str,
            mode_str=pod_target_obj.release_obj.mode_str,
            status_str="queued",
            created_timestamp_str=now_ts.isoformat(),
        )
        with self._lock_obj:
            self._job_map_dict[job_obj.job_id_str] = job_obj
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
        return job_obj

    def get_job_dict(self, job_id_str: str) -> dict[str, Any] | None:
        with self._lock_obj:
            job_obj = self._job_map_dict.get(job_id_str)
            return None if job_obj is None else job_obj.to_dict()

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
        with self._lock_obj:
            job_obj = self._job_map_dict[job_id_str]
            job_obj.status_str = "succeeded"
            job_obj.completed_timestamp_str = datetime.now(UTC).isoformat()
            job_obj.result_dict = result_dict


@dataclass
class DashboardApp:
    releases_root_path_str: str = DEFAULT_RELEASES_ROOT_PATH_STR
    config_path_str: str = DEFAULT_CONFIG_PATH_STR
    results_root_path_str: str = DEFAULT_RESULTS_ROOT_PATH_STR
    event_log_path_str: str = DEFAULT_EVENT_LOG_PATH_STR
    diff_job_manager_obj: DiffJobManager | None = None

    def __post_init__(self) -> None:
        if self.diff_job_manager_obj is None:
            self.diff_job_manager_obj = DiffJobManager(_run_reference_diff_for_pod)

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
    pod_row_dict_list = [
        build_pod_row_dict(
            pod_target_obj=target_obj,
            as_of_ts=as_of_ts,
            results_root_path_str=app_obj.results_root_path_str,
            event_log_path_str=app_obj.event_log_path_str,
        )
        for target_obj in app_obj.get_target_list()
    ]
    alert_dict_list = _build_alert_dict_list(pod_row_dict_list)
    return {
        "as_of_timestamp_str": as_of_ts.isoformat(),
        "pod_row_dict_list": pod_row_dict_list,
        "alert_dict_list": alert_dict_list,
        "alert_summary_dict": _build_alert_summary_dict(alert_dict_list),
        "mode_list": sorted({str(row_dict["mode_str"]) for row_dict in pod_row_dict_list}),
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
    base_row_dict = {
        "release_id_str": release_obj.release_id_str,
        "user_id_str": release_obj.user_id_str,
        "pod_id_str": release_obj.pod_id_str,
        "mode_str": release_obj.mode_str,
        "account_route_str": release_obj.account_route_str,
        "strategy_import_str": release_obj.strategy_import_str,
        "auto_submit_enabled_bool": bool(release_obj.auto_submit_enabled_bool),
        "db_path_str": pod_target_obj.db_path_str,
        "db_exists_bool": db_path_obj.exists(),
        "db_override_bool": pod_target_obj.db_override_bool,
        "db_status_str": "ok" if db_path_obj.exists() else "missing",
        "latest_decision_plan_status_str": None,
        "latest_decision_plan_submission_timestamp_str": None,
        "latest_decision_plan_target_execution_timestamp_str": None,
        "latest_vplan_status_str": None,
        "latest_vplan_id_int": None,
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
        base_row_dict["latest_decision_plan_status_str"] = latest_decision_plan_row_dict.get(
            "status_str"
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
    if latest_vplan_row_dict is not None:
        base_row_dict["latest_vplan_status_str"] = latest_vplan_row_dict.get("status_str")
        base_row_dict["latest_vplan_id_int"] = latest_vplan_row_dict.get("vplan_id_int")
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
    if pod_state_row_dict is not None:
        position_map_dict = _json_map_dict(pod_state_row_dict.get("position_json_str"))
        base_row_dict["position_count_int"] = len(
            [
                amount_float
                for amount_float in position_map_dict.values()
                if abs(float(amount_float)) > 1e-9
            ]
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
    event_deque: deque[dict[str, Any]] = deque(maxlen=int(limit_int))
    with log_path_obj.open("r", encoding="utf-8") as file_obj:
        for line_str in file_obj:
            line_str = line_str.strip()
            if len(line_str) == 0:
                continue
            try:
                event_dict = json.loads(line_str)
            except json.JSONDecodeError:
                continue
            if _event_matches_pod_bool(event_dict, pod_id_str):
                event_deque.append(event_dict)
    return list(event_deque)


def _latest_event_timestamp_str(log_path_str: str | None, pod_id_str: str) -> str | None:
    if log_path_str is None:
        return None
    event_dict_list = load_recent_event_dict_list(
        log_path_str=log_path_str,
        pod_id_str=pod_id_str,
        limit_int=1,
    )
    if len(event_dict_list) == 0:
        return None
    return _event_timestamp_str(event_dict_list[-1])


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
    return isinstance(related_pod_id_list, list) and pod_id_str in {
        str(item_obj) for item_obj in related_pod_id_list
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


def make_dashboard_handler_class(app_obj: DashboardApp):
    class DashboardRequestHandler(BaseHTTPRequestHandler):
        server_version = "AlphaLiveDashboard/1.0"

        def do_GET(self) -> None:  # noqa: N802 - stdlib handler API.
            self._handle_request("GET")

        def do_POST(self) -> None:  # noqa: N802 - stdlib handler API.
            self._handle_request("POST")

        def log_message(self, format_str: str, *args_obj: object) -> None:
            return

        def _handle_request(self, method_str: str) -> None:
            parsed_url_obj = urlparse(self.path)
            path_str = unquote(parsed_url_obj.path)
            try:
                if method_str == "GET" and path_str == "/":
                    self._send_html(DASHBOARD_HTML_STR)
                    return
                if method_str == "GET" and path_str == "/api/pods":
                    self._send_json(build_dashboard_summary_dict(app_obj))
                    return
                if method_str == "GET" and path_str.startswith("/api/pods/"):
                    self._handle_pod_get(path_str)
                    return
                if method_str == "POST" and path_str.startswith("/api/pods/"):
                    self._handle_pod_post(path_str)
                    return
                if method_str == "GET" and path_str.startswith("/api/jobs/"):
                    self._handle_job_get(path_str)
                    return
                if method_str == "GET" and path_str.startswith("/artifacts/"):
                    self._handle_artifact_get(path_str)
                    return
                self._send_error_json(404, "not_found", f"Unknown route: {path_str}")
            except KeyError as exception_obj:
                self._send_error_json(404, "unknown_pod", str(exception_obj))
            except Exception as exception_obj:  # pragma: no cover - server safety net.
                self._send_error_json(500, "internal_error", str(exception_obj))

        def _handle_pod_get(self, path_str: str) -> None:
            part_list = [part_str for part_str in path_str.split("/") if part_str]
            if len(part_list) == 3:
                self._send_json(build_pod_detail_dict(app_obj, part_list[2]))
                return
            if len(part_list) == 4 and part_list[3] == "events":
                self._send_json(
                    {
                        "pod_id_str": part_list[2],
                        "event_dict_list": load_recent_event_dict_list(
                            log_path_str=app_obj.event_log_path_str,
                            pod_id_str=part_list[2],
                            limit_int=DEFAULT_EVENT_LIMIT_INT,
                        ),
                    }
                )
                return
            if len(part_list) == 5 and part_list[3] == "diff" and part_list[4] == "latest":
                target_obj = app_obj.get_target_for_pod(part_list[2])
                if target_obj is None:
                    raise KeyError(f"Unknown enabled pod_id_str '{part_list[2]}'.")
                self._send_json(
                    find_latest_diff_artifact_dict(
                        results_root_path_str=app_obj.results_root_path_str,
                        mode_str=target_obj.release_obj.mode_str,
                        pod_id_str=part_list[2],
                    )
                )
                return
            self._send_error_json(404, "not_found", f"Unknown POD route: {path_str}")

        def _handle_pod_post(self, path_str: str) -> None:
            part_list = [part_str for part_str in path_str.split("/") if part_str]
            if len(part_list) == 5 and part_list[3] == "diff" and part_list[4] == "run":
                target_obj = app_obj.get_target_for_pod(part_list[2])
                if target_obj is None:
                    raise KeyError(f"Unknown enabled pod_id_str '{part_list[2]}'.")
                assert app_obj.diff_job_manager_obj is not None
                job_obj = app_obj.diff_job_manager_obj.start_job(
                    pod_target_obj=target_obj,
                    releases_root_path_str=app_obj.releases_root_path_str,
                    results_root_path_str=app_obj.results_root_path_str,
                )
                self._send_json(job_obj.to_dict(), status_int=202)
                return
            self._send_error_json(404, "not_found", f"Unknown POD POST route: {path_str}")

        def _handle_job_get(self, path_str: str) -> None:
            part_list = [part_str for part_str in path_str.split("/") if part_str]
            if len(part_list) != 3:
                self._send_error_json(404, "not_found", f"Unknown job route: {path_str}")
                return
            assert app_obj.diff_job_manager_obj is not None
            job_dict = app_obj.diff_job_manager_obj.get_job_dict(part_list[2])
            if job_dict is None:
                self._send_error_json(404, "unknown_job", f"Unknown job_id_str '{part_list[2]}'.")
                return
            self._send_json(job_dict)

        def _handle_artifact_get(self, path_str: str) -> None:
            relative_path_str = path_str.removeprefix("/artifacts/")
            artifact_path_obj = (Path(app_obj.results_root_path_str) / relative_path_str).resolve()
            results_root_path_obj = Path(app_obj.results_root_path_str).resolve()
            if not _is_relative_to_bool(artifact_path_obj, results_root_path_obj):
                self._send_error_json(403, "forbidden", "Artifact path escapes results root.")
                return
            if not artifact_path_obj.exists() or not artifact_path_obj.is_file():
                self._send_error_json(404, "not_found", "Artifact not found.")
                return
            content_type_str = mimetypes.guess_type(str(artifact_path_obj))[0] or "application/octet-stream"
            self.send_response(200)
            self.send_header("Content-Type", content_type_str)
            self.send_header("Content-Length", str(artifact_path_obj.stat().st_size))
            self.end_headers()
            self.wfile.write(artifact_path_obj.read_bytes())

        def _send_html(self, html_text_str: str) -> None:
            payload_bytes = html_text_str.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(payload_bytes)))
            self.end_headers()
            self.wfile.write(payload_bytes)

        def _send_json(self, payload_obj: Any, status_int: int = 200) -> None:
            payload_bytes = json.dumps(_jsonable_obj(payload_obj), indent=2, sort_keys=True).encode("utf-8")
            self.send_response(status_int)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(payload_bytes)))
            self.end_headers()
            self.wfile.write(payload_bytes)

        def _send_error_json(
            self,
            status_int: int,
            error_code_str: str,
            message_str: str,
        ) -> None:
            self._send_json(
                {
                    "error_code_str": error_code_str,
                    "message_str": message_str,
                },
                status_int=status_int,
            )

    return DashboardRequestHandler


def serve_dashboard(
    host_str: str = DEFAULT_DASHBOARD_HOST_STR,
    port_int: int = DEFAULT_DASHBOARD_PORT_INT,
    releases_root_path_str: str = DEFAULT_RELEASES_ROOT_PATH_STR,
    config_path_str: str = DEFAULT_CONFIG_PATH_STR,
    results_root_path_str: str = DEFAULT_RESULTS_ROOT_PATH_STR,
    event_log_path_str: str = DEFAULT_EVENT_LOG_PATH_STR,
) -> None:
    app_obj = DashboardApp(
        releases_root_path_str=releases_root_path_str,
        config_path_str=config_path_str,
        results_root_path_str=results_root_path_str,
        event_log_path_str=event_log_path_str,
    )
    handler_cls = make_dashboard_handler_class(app_obj)
    server_obj = ThreadingHTTPServer((host_str, int(port_int)), handler_cls)
    print(f"Live POD dashboard listening on http://{host_str}:{int(port_int)}")
    server_obj.serve_forever()


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
    row_dict["rehearsal_status_dict"] = _build_rehearsal_status_dict(row_dict)
    row_dict["required_action_dict"] = _build_required_action_dict(row_dict)
    row_dict["lifecycle_step_dict_list"] = _build_lifecycle_step_dict_list(row_dict)
    row_dict["data_freshness_dict"] = _build_data_freshness_dict(row_dict)
    row_dict["debug_summary_dict"] = _build_debug_summary_dict(row_dict)
    return row_dict


def _finalize_pod_detail_debug_story_dict(detail_dict: dict[str, Any]) -> dict[str, Any]:
    detail_dict["debug_story_dict"] = _build_debug_story_dict(detail_dict)
    return detail_dict


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
        return _required_action_dict("Manual review", "red", "VPlan is blocked.", "show_vplan")
    if int(row_dict.get("missing_ack_count_int") or 0) > 0:
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
        return _lifecycle_step_dict(
            "ack",
            "ACK",
            status_str,
            "green",
            f"rows={int(row_dict.get('broker_ack_count_int') or 0)}",
        )
    severity_str = "yellow" if row_dict.get("latest_vplan_status_str") in ("submitted", "submitting") else "gray"
    return _lifecycle_step_dict("ack", "ACK", status_str, severity_str, "Awaiting ACK evidence.")


def _build_fill_step_dict(row_dict: dict[str, Any]) -> dict[str, Any]:
    if row_dict.get("latest_vplan_id_int") is None:
        return _lifecycle_step_dict("fill", "Fill", "none", "gray", "No VPlan.")
    fill_count_int = int(row_dict.get("fill_count_int") or 0)
    if fill_count_int > 0:
        return _lifecycle_step_dict("fill", "Fill", "recorded", "green", f"fills={fill_count_int}")
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
    item_dict_list = [
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
                f"status={row_dict.get('dtb3_download_status_str')}, days={row_dict.get('dtb3_freshness_business_days_int')}",
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
        "dtb3_download_status_str": row_dict.get("dtb3_download_status_str"),
        "dtb3_latest_observation_date_str": row_dict.get("dtb3_latest_observation_date_str"),
        "dtb3_freshness_business_days_int": row_dict.get("dtb3_freshness_business_days_int"),
        "dtb3_source_name_str": row_dict.get("dtb3_source_name_str"),
        "dtb3_used_cache_bool": row_dict.get("dtb3_used_cache_bool"),
        "item_dict_list": item_dict_list,
    }


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
    if severity_str in {"red", "yellow"}:
        return _alert_dict(
            row_dict,
            "freshness",
            severity_str,
            f"{label_str} freshness",
            str(item_dict.get("detail_str") or ""),
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
            str(item_dict.get("detail_str") or ""),
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
) -> dict[str, Any]:
    return {
        "label_str": label_str,
        "value_str": value_str,
        "severity_str": severity_str,
        "detail_str": detail_str,
    }


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
        candidate_dict_list.append(
            _debug_candidate_dict(
                priority_int=21,
                severity_str="red",
                label_str="VPlan blocked",
                reason_str="The latest VPlan is blocked.",
                evidence_str=f"vplan_status={row_dict.get('latest_vplan_status_str')}, vplan_id={row_dict.get('latest_vplan_id_int')}",
                inspect_command_name_str="show_vplan",
                timestamp_str=row_dict.get("latest_vplan_submission_timestamp_str"),
            )
        )
    missing_ack_count_int = int(row_dict.get("missing_ack_count_int") or 0)
    if missing_ack_count_int > 0:
        candidate_dict_list.append(
            _debug_candidate_dict(
                priority_int=30,
                severity_str="red",
                label_str="Broker ACK missing",
                reason_str="Broker did not acknowledge all submitted orders.",
                evidence_str=f"missing_ack_count={missing_ack_count_int}, submit_ack_status={row_dict.get('latest_submit_ack_status_str')}",
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
        if item_severity_str in {"red", "yellow"}:
            candidate_dict_list.append(
                _debug_candidate_dict(
                    priority_int=90,
                    severity_str=item_severity_str,
                    label_str=f"{item_label_str} freshness",
                    reason_str=str(freshness_item_dict.get("detail_str") or ""),
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
        )

    vplan_dict = detail_dict.get("latest_vplan_dict") or {}
    if vplan_dict:
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
            )
        for fill_row_dict in vplan_dict.get("fill_row_dict_list", []):
            add_event(
                "Fill",
                str(fill_row_dict.get("asset_str") or "Fill"),
                "recorded",
                "green",
                fill_row_dict.get("fill_timestamp_str"),
                f"shares={fill_row_dict.get('fill_amount_float')}, price={fill_row_dict.get('fill_price_float')}, open={fill_row_dict.get('official_open_price_float')}",
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
            f"freshness_days={row_dict.get('dtb3_freshness_business_days_int')}, source={row_dict.get('dtb3_source_name_str')}",
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
    report_dict = detail_dict.get("latest_execution_report_dict") or {}
    item_dict_list = [
        _debug_evidence_item_dict("DB", row_dict.get("db_status_str"), _status_to_severity_str(str(row_dict.get("db_status_str") or ""), {"ok"}, {"empty"}), str(row_dict.get("db_path_str") or "")),
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
                f"observation={row_dict.get('dtb3_latest_observation_date_str')}, days={row_dict.get('dtb3_freshness_business_days_int')}",
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
    freshness_obj = row_dict.get("dtb3_freshness_business_days_int")
    try:
        freshness_days_int = int(freshness_obj)
    except (TypeError, ValueError):
        return "gray"
    return "green" if freshness_days_int <= 2 else "red"


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


DASHBOARD_HTML_STR = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Live POD Dashboard</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f7f8fa;
      --panel: #ffffff;
      --line: #e2e7ef;
      --line-strong: #cbd5e1;
      --text: #111827;
      --muted: #667085;
      --green: #247a52;
      --yellow: #936316;
      --red: #b42318;
      --gray: #6b7280;
      --blue: #175cd3;
      --blue-soft: #f3f7ff;
      --green-soft: #ecfdf3;
      --yellow-soft: #fffaeb;
      --red-soft: #fef3f2;
      --gray-soft: #f2f4f7;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font: 14px/1.5 Arial, Helvetica, sans-serif;
    }
    header {
      height: 56px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 0 18px;
      border-bottom: 1px solid var(--line);
      background: var(--panel);
    }
    h1 { font-size: 18px; margin: 0; }
    main {
      max-width: 1540px;
      margin: 0 auto;
      padding: 14px 18px 28px;
    }
    .toolbar {
      display: flex;
      gap: 10px;
      align-items: center;
      margin-bottom: 12px;
      flex-wrap: wrap;
    }
    .dashboard-section {
      --section-accent: var(--line-strong);
      --section-header-bg: #fbfcfe;
      min-width: 0;
      margin-bottom: 16px;
      border: 1px solid var(--line-strong);
      border-left: 5px solid var(--section-accent);
      border-radius: 10px;
      background: var(--panel);
      overflow: hidden;
      box-shadow: 0 4px 14px rgba(15, 23, 42, 0.06);
    }
    .console-section {
      --section-accent: var(--blue);
      --section-header-bg: #f3f7ff;
    }
    .alert-panel {
      --section-accent: var(--yellow);
      --section-header-bg: #fffbeb;
    }
    .attention-panel {
      --section-accent: var(--red);
      --section-header-bg: #fff7ed;
    }
    .table-panel {
      --section-accent: var(--gray);
      --section-header-bg: #f8fafc;
    }
    .detail-workspace {
      --section-accent: var(--green);
      --section-header-bg: #f0fdf4;
    }
    .dashboard-section-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
      padding: 12px 14px;
      border-bottom: 1px solid var(--line-strong);
      background: var(--section-header-bg);
    }
    .dashboard-section-header h2 {
      display: flex;
      align-items: center;
      gap: 8px;
      margin: 0;
      font-size: 14px;
      line-height: 1.25;
    }
    .dashboard-section-header h2::before {
      content: "";
      display: inline-block;
      width: 8px;
      height: 8px;
      border-radius: 999px;
      background: var(--section-accent);
      box-shadow: 0 0 0 3px rgba(255, 255, 255, 0.9);
      flex: 0 0 auto;
    }
    .dashboard-section-subtitle {
      margin-top: 2px;
      color: var(--muted);
      font-size: 12px;
    }
    .dashboard-section-body {
      padding: 14px;
      background: #fff;
    }
    .dashboard-section-body.flush {
      padding: 0;
    }
    .section-status {
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
      white-space: nowrap;
    }
    select, button {
      min-height: 34px;
      border: 1px solid var(--line);
      background: var(--panel);
      color: var(--text);
      padding: 0 10px;
      border-radius: 7px;
    }
    button { cursor: pointer; }
    button.primary { background: var(--blue); color: white; border-color: var(--blue); }
    button.secondary { background: #f9fafb; }
    button:disabled { opacity: 0.55; cursor: not-allowed; }
    .muted { color: var(--muted); }
    .mono { font-family: Consolas, Menlo, monospace; }
    .num { text-align: right; font-variant-numeric: tabular-nums; }
    table {
      width: 100%;
      min-width: 980px;
      border-collapse: collapse;
      background: var(--panel);
      border: 0;
    }
    .pod-table {
      table-layout: fixed;
      min-width: 1120px;
    }
    .pod-table th:nth-child(1),
    .pod-table td:nth-child(1) { width: 92px; }
    .pod-table th:nth-child(2),
    .pod-table td:nth-child(2) { width: 280px; }
    .pod-table th:nth-child(3),
    .pod-table td:nth-child(3) { width: 92px; }
    .pod-table th:nth-child(4),
    .pod-table td:nth-child(4) { width: 190px; }
    .pod-table th:nth-child(5),
    .pod-table td:nth-child(5) { width: 280px; }
    .pod-table th:nth-child(6),
    .pod-table td:nth-child(6) { width: 220px; }
    .pod-table th:nth-child(7),
    .pod-table td:nth-child(7) { width: 150px; }
    th, td {
      border-bottom: 1px solid var(--line);
      padding: 7px 10px;
      text-align: left;
      vertical-align: top;
      white-space: nowrap;
    }
    th {
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
      background: #fbfcfe;
      position: sticky;
      top: 0;
      z-index: 2;
    }
    tr:hover td { background: #fbfcfe; }
    tr.selected td { background: var(--blue-soft); }
    tr.selected td:first-child { box-shadow: inset 3px 0 var(--blue); }
    tr.changed td { box-shadow: inset 0 -2px #bfdbfe; }
    .pod-link { color: var(--blue); cursor: pointer; font-weight: 700; }
    .strategy-text {
      display: inline-block;
      max-width: 420px;
      white-space: normal;
      overflow-wrap: anywhere;
    }
    .debug-cell,
    .action-cell {
      min-width: 260px;
      max-width: 300px;
      white-space: normal;
      overflow-wrap: anywhere;
    }
    .debug-verdict {
      display: grid;
      grid-template-columns: auto minmax(0, 1fr);
      gap: 7px;
      align-items: start;
    }
    .debug-verdict-label {
      font-weight: 800;
      line-height: 1.25;
      overflow-wrap: anywhere;
    }
    .debug-verdict-reason,
    .debug-verdict-evidence {
      color: var(--muted);
      font-size: 12px;
      line-height: 1.25;
      overflow-wrap: anywhere;
    }
    .summary-action {
      display: grid;
      grid-template-columns: auto minmax(0, 1fr);
      gap: 7px;
      align-items: start;
      min-width: 0;
    }
    .summary-action-main { min-width: 0; }
    .summary-action-label {
      font-weight: 800;
      line-height: 1.25;
      overflow-wrap: anywhere;
    }
    .summary-action-reason,
    .summary-action-inspect {
      color: var(--muted);
      font-size: 12px;
      line-height: 1.25;
      overflow-wrap: anywhere;
    }
    .summary-action .pill {
      margin-top: 1px;
    }
    .pill {
      display: inline-block;
      min-width: 48px;
      text-align: center;
      padding: 2px 7px;
      border-radius: 999px;
      border: 1px solid transparent;
      font-size: 12px;
      font-weight: 700;
      white-space: nowrap;
      overflow-wrap: normal;
    }
    .pill.green { color: var(--green); background: var(--green-soft); border-color: #bbf7d0; }
    .pill.yellow { color: var(--yellow); background: var(--yellow-soft); border-color: #fedf89; }
    .pill.red { color: var(--red); background: var(--red-soft); border-color: #fecdca; }
    .pill.gray { color: var(--gray); background: var(--gray-soft); border-color: #e5e7eb; }
    .green { background: var(--green); }
    .yellow { background: var(--yellow); }
    .red { background: var(--red); }
    .gray { background: var(--gray); }
    .state-cell,
    .eod-cell,
    .diff-cell,
    .updated-cell,
    .operator-cell {
      white-space: normal;
      overflow-wrap: anywhere;
    }
    .severity-cell {
      white-space: normal;
    }
    .operator-state {
      font-weight: 800;
      line-height: 1.25;
      overflow-wrap: anywhere;
    }
    .operator-reason,
    .operator-evidence {
      color: var(--muted);
      font-size: 12px;
      line-height: 1.3;
      margin-top: 2px;
      overflow-wrap: anywhere;
    }
    .operator-changed {
      display: inline-block;
      margin-top: 5px;
      color: var(--blue);
      font-size: 11px;
      font-weight: 800;
    }
    .operator-command-cell {
      white-space: normal;
    }
    .operator-command-cell button {
      width: 100%;
      min-height: 30px;
    }
    .state-main { font-weight: 700; }
    .state-sub,
    .cell-sub {
      color: var(--muted);
      font-size: 12px;
      margin-top: 2px;
    }
    .table-panel { min-width: 0; }
    .table-panel .table-scroll {
      border: 0;
      border-radius: 0;
    }
    .table-scroll {
      width: 100%;
      overflow: auto;
      max-height: 52vh;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
    }
    .alert-counts {
      display: flex;
      gap: 6px;
      align-items: center;
      margin-left: auto;
      flex-wrap: wrap;
    }
    .count-pill {
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 3px 8px;
      font-size: 12px;
      color: var(--muted);
      background: var(--panel);
    }
    .count-pill.red { color: var(--red); background: var(--red-soft); }
    .count-pill.yellow { color: var(--yellow); background: var(--yellow-soft); }
    .count-pill.gray { color: var(--gray); background: var(--gray-soft); }
    .alert-panel {
      background: var(--panel);
    }
    .alert-panel.collapsed .alert-header {
      border-bottom: 0;
    }
    .alert-panel.collapsed .alert-list {
      display: none;
    }
    .alert-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
      padding: 10px 12px;
      border-bottom: 1px solid var(--line);
      background: #fbfcfe;
    }
    .alert-header h2 { margin: 0; font-size: 14px; }
    .alert-header-actions {
      display: flex;
      gap: 8px;
      align-items: center;
      flex-wrap: wrap;
      justify-content: flex-end;
    }
    .alert-list {
      display: grid;
      gap: 0;
      background: var(--panel);
    }
    .alert-row {
      display: grid;
      grid-template-columns: 108px minmax(0, 1fr);
      gap: 10px;
      text-align: left;
      min-height: 56px;
      padding: 9px 12px;
      border: 0;
      border-top: 1px solid var(--line);
      border-radius: 0;
      background: var(--panel);
    }
    .alert-row:hover { background: #fbfcfe; }
    .alert-row.red { box-shadow: inset 3px 0 var(--red); }
    .alert-row.yellow { box-shadow: inset 3px 0 var(--yellow); }
    .alert-row.gray { box-shadow: inset 3px 0 var(--gray); }
    .alert-type {
      color: var(--muted);
      font-size: 11px;
      font-weight: 700;
      text-transform: uppercase;
      overflow-wrap: anywhere;
    }
    .alert-main { min-width: 0; }
    .alert-label { font-weight: 800; overflow-wrap: anywhere; }
    .alert-meta,
    .alert-reason {
      color: var(--muted);
      font-size: 12px;
      overflow-wrap: anywhere;
    }
    .console-status-strip {
      display: grid;
      grid-template-columns: repeat(6, minmax(120px, 1fr));
      gap: 8px;
    }
    .console-status-item {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      padding: 8px 10px;
      min-width: 0;
    }
    .console-status-label {
      color: var(--muted);
      font-size: 11px;
      font-weight: 800;
      text-transform: uppercase;
    }
    .console-status-value {
      margin-top: 2px;
      font-weight: 800;
      overflow-wrap: anywhere;
    }
    .attention-panel {
      background: var(--panel);
    }
    .attention-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
      padding: 10px 12px;
      border-bottom: 1px solid var(--line);
      background: #fbfcfe;
    }
    .attention-header h2 { margin: 0; font-size: 14px; }
    .attention-list { display: grid; background: var(--panel); }
    .attention-group-title {
      padding: 7px 12px;
      border-top: 1px solid var(--line);
      color: var(--muted);
      font-size: 11px;
      font-weight: 800;
      text-transform: uppercase;
      background: #fbfcfe;
    }
    .attention-row {
      display: grid;
      grid-template-columns: 92px minmax(0, 1fr) 144px;
      gap: 10px;
      align-items: start;
      padding: 8px 12px;
      border-top: 1px solid var(--line);
      cursor: pointer;
      background: #fff;
      color: var(--text);
    }
    .attention-row:hover { background: #fbfcfe; }
    .attention-row.red {
      border-left: 4px solid var(--red);
      background: #fff8f7;
      color: var(--text);
    }
    .attention-row.yellow {
      border-left: 4px solid var(--yellow);
      background: #fffdf5;
      color: var(--text);
    }
    .attention-row.green {
      border-left: 4px solid var(--green);
      background: #f8fffb;
      color: var(--text);
    }
    .attention-row.gray {
      border-left: 4px solid var(--gray);
      background: #fbfcfe;
      color: var(--text);
    }
    .attention-main { min-width: 0; }
    .attention-title { font-weight: 800; overflow-wrap: anywhere; }
    .attention-meta,
    .attention-reason,
    .attention-evidence {
      color: var(--muted);
      font-size: 12px;
      overflow-wrap: anywhere;
    }
    .attention-command button { width: 100%; }
    .detail-workspace {
      margin-top: 14px;
    }
    .empty-state {
      min-height: 0;
    }
    .empty-state .dashboard-section-body {
      min-height: 160px;
      display: flex;
      flex-direction: column;
      justify-content: center;
      align-items: flex-start;
    }
    .detail-shell-body {
      padding: 14px 16px 16px;
    }
    .detail-header {
      display: flex;
      justify-content: space-between;
      gap: 14px;
      align-items: flex-start;
      border-bottom: 1px solid var(--line);
      padding-bottom: 12px;
      margin-bottom: 14px;
      position: sticky;
      top: 0;
      z-index: 5;
      background: var(--panel);
    }
    .detail-title h2 { margin: 0 0 6px; font-size: 18px; overflow-wrap: anywhere; }
    .detail-subtitle { display: flex; gap: 8px; flex-wrap: wrap; align-items: center; }
    .inspect-actions { display: flex; gap: 8px; flex-wrap: wrap; justify-content: flex-end; }
    .action-banner {
      border: 1px solid var(--line);
      border-left-width: 4px;
      padding: 10px;
      margin-bottom: 14px;
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: flex-start;
      flex-wrap: wrap;
      background: #fff;
      border-radius: 7px;
    }
    .action-banner.green { border-left-color: var(--green); background: #f0fdf4; color: var(--text); }
    .action-banner.yellow { border-left-color: var(--yellow); background: #fffbeb; color: var(--text); }
    .action-banner.red { border-left-color: var(--red); background: #fff7ed; color: var(--text); }
    .action-banner.gray { border-left-color: var(--gray); background: #f9fafb; color: var(--text); }
    .action-main { min-width: 280px; flex: 1 1 520px; }
    .action-title { font-weight: 800; margin-bottom: 3px; }
    .action-context-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
      gap: 6px 12px;
      margin-top: 9px;
    }
    .action-context-item {
      border: 1px solid var(--line);
      border-left: 3px solid #d0d5dd;
      border-radius: 6px;
      padding: 7px 8px;
      min-width: 0;
      background: rgba(255, 255, 255, 0.72);
    }
    .action-context-label {
      color: var(--muted);
      font-size: 11px;
      font-weight: 800;
      text-transform: uppercase;
    }
    .action-context-value {
      margin-top: 2px;
      font-size: 12px;
      font-weight: 700;
      overflow-wrap: anywhere;
    }
    .action-context-detail {
      margin-top: 1px;
      color: var(--muted);
      font-size: 11px;
      overflow-wrap: anywhere;
    }
    .action-context-item.green,
    .action-context-item.yellow,
    .action-context-item.red,
    .action-context-item.gray {
      background: rgba(255, 255, 255, 0.72);
    }
    .action-context-item.green { border-left-color: #16a34a; }
    .action-context-item.yellow { border-left-color: #d97706; }
    .action-context-item.red { border-left-color: #dc2626; }
    .action-context-item.gray { border-left-color: #98a2b3; }
    .action-context-item .action-context-value { color: var(--text); }
    .action-command { white-space: nowrap; }
    .lifecycle-progress {
      display: flex;
      flex-direction: column;
      gap: 10px;
    }
    .lifecycle-current-summary {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      border: 1px solid var(--line);
      background: #fbfcfe;
      border-radius: 8px;
      padding: 8px 10px;
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
    }
    .lifecycle-rail-scroll {
      overflow-x: auto;
      padding-bottom: 3px;
    }
    .lifecycle-rail {
      display: flex;
      align-items: stretch;
      gap: 0;
      min-width: max-content;
    }
    .lifecycle-step-card {
      width: 128px;
      min-height: 96px;
      border: 1px solid var(--line);
      border-left: 4px solid var(--gray);
      background: #fff;
      border-radius: 8px;
      padding: 8px;
      display: flex;
      flex-direction: column;
      gap: 5px;
    }
    .lifecycle-step-card.green { border-left-color: var(--green); background: #f8fffb; }
    .lifecycle-step-card.yellow { border-left-color: var(--yellow); background: #fffdf5; }
    .lifecycle-step-card.red { border-left-color: var(--red); background: #fff8f6; }
    .lifecycle-step-card.gray { border-left-color: var(--gray); background: #fbfcfe; }
    .lifecycle-step-card.current {
      border-color: var(--blue);
      border-left-color: var(--blue);
      box-shadow: 0 0 0 2px rgba(23, 92, 211, 0.12);
      background: var(--blue-soft);
    }
    .lifecycle-step-top {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 6px;
    }
    .lifecycle-step-label {
      color: var(--text);
      font-size: 12px;
      font-weight: 800;
    }
    .lifecycle-current-badge {
      border: 1px solid #bfdbfe;
      background: #fff;
      color: var(--blue);
      border-radius: 999px;
      padding: 1px 6px;
      font-size: 10px;
      font-weight: 800;
      text-transform: uppercase;
      letter-spacing: 0;
      white-space: nowrap;
    }
    .lifecycle-step-status {
      color: var(--text);
      font-size: 11px;
      font-weight: 700;
      overflow-wrap: anywhere;
    }
    .lifecycle-step-sentence,
    .lifecycle-step-detail {
      color: var(--muted);
      font-size: 11px;
      line-height: 1.35;
      overflow-wrap: anywhere;
    }
    .lifecycle-arrow {
      width: 26px;
      min-width: 26px;
      display: flex;
      align-items: center;
      justify-content: center;
      color: var(--muted);
      font-weight: 800;
      font-size: 18px;
    }
    .lifecycle-reference-check {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      border: 1px solid var(--line);
      border-left: 4px solid var(--gray);
      background: #fbfcfe;
      border-radius: 8px;
      padding: 8px 10px;
      min-height: 44px;
    }
    .lifecycle-reference-check.green { border-left-color: var(--green); background: #f8fffb; }
    .lifecycle-reference-check.yellow { border-left-color: var(--yellow); background: #fffdf5; }
    .lifecycle-reference-check.red { border-left-color: var(--red); background: #fff8f6; }
    .lifecycle-reference-check.gray { border-left-color: var(--gray); background: #fbfcfe; }
    .lifecycle-reference-title {
      color: var(--text);
      font-size: 12px;
      font-weight: 800;
    }
    .lifecycle-reference-detail {
      color: var(--muted);
      font-size: 11px;
      margin-top: 2px;
      overflow-wrap: anywhere;
    }
    .detail-tabs {
      display: flex;
      gap: 4px;
      border-bottom: 1px solid var(--line);
      margin: 12px 0 14px;
      overflow-x: auto;
      padding-bottom: 0;
    }
    .tab-button {
      border: 0;
      border-bottom: 2px solid transparent;
      border-radius: 0;
      background: transparent;
      color: var(--muted);
      min-height: 38px;
      padding: 0 12px;
      font-weight: 700;
      white-space: nowrap;
    }
    .tab-button.active {
      color: var(--text);
      border-bottom-color: var(--blue);
    }
    .tab-body {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
    }
    .panel {
      border: 1px solid var(--line);
      background: #fff;
      min-width: 0;
      border-radius: 8px;
      overflow: hidden;
    }
    .panel.full { grid-column: 1 / -1; }
    .panel h3 {
      margin: 0;
      padding: 10px 12px;
      border-bottom: 1px solid var(--line);
      font-size: 14px;
      background: #fbfcfe;
    }
    .panel-body { padding: 12px; }
    .metric-grid {
      display: grid;
      grid-template-columns: repeat(4, minmax(120px, 1fr));
      gap: 8px;
    }
    .metric {
      border: 1px solid var(--line);
      padding: 8px;
      min-width: 0;
      border-radius: 7px;
      background: #fff;
    }
    .metric-label { color: var(--muted); font-size: 12px; }
    .metric-value { margin-top: 3px; font-weight: 700; overflow-wrap: anywhere; }
    .signed-value {
      font-weight: 800;
      font-variant-numeric: tabular-nums;
    }
    .signed-value.positive { color: var(--green); }
    .signed-value.negative { color: var(--red); }
    .signed-value.neutral { color: var(--text); }
    .kv-grid {
      display: grid;
      grid-template-columns: 180px minmax(0, 1fr);
      border-top: 1px solid var(--line);
      border-left: 1px solid var(--line);
    }
    .kv-grid div {
      padding: 7px 8px;
      border-right: 1px solid var(--line);
      border-bottom: 1px solid var(--line);
      overflow-wrap: anywhere;
    }
    .kv-key { color: var(--muted); background: #f9fafb; }
    .mini-table {
      width: 100%;
      min-width: 0;
      border: 1px solid var(--line);
      margin-top: 8px;
    }
    .mini-table th,
    .mini-table td {
      white-space: normal;
      overflow-wrap: anywhere;
      padding: 7px 8px;
    }
    .mini-table .nowrap { white-space: nowrap; }
    .pnl-curve {
      width: 100%;
      max-width: 1120px;
      margin-top: 10px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      padding: 10px 12px 10px;
    }
    .pnl-chart-header {
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      gap: 12px;
      margin-bottom: 8px;
    }
    .pnl-chart-title {
      font-weight: 800;
      line-height: 1.2;
    }
    .pnl-chart-subtitle,
    .pnl-chart-range {
      color: var(--muted);
      font-size: 12px;
    }
    .pnl-chart-range {
      text-align: right;
      font-variant-numeric: tabular-nums;
    }
    .pnl-chart-range strong {
      display: block;
      color: var(--text);
      font-size: 12px;
    }
    .pnl-curve-plot {
      position: relative;
      display: grid;
      grid-template-columns: 96px minmax(0, 980px);
      gap: 10px;
      align-items: stretch;
    }
    .pnl-y-axis {
      display: flex;
      flex-direction: column;
      justify-content: space-between;
      padding: 3px 0 24px;
      color: var(--muted);
      font-size: 12px;
      font-variant-numeric: tabular-nums;
      text-align: right;
    }
    .pnl-chart-area {
      position: relative;
      min-width: 0;
      width: 100%;
      max-width: 980px;
      aspect-ratio: 1000 / 260;
      min-height: 210px;
    }
    .pnl-chart-area svg,
    .pnl-point-layer {
      position: absolute;
      inset: 0;
      width: 100%;
      height: 100%;
    }
    .pnl-chart-area svg {
      display: block;
      overflow: visible;
    }
    .pnl-point-layer {
      pointer-events: none;
    }
    .pnl-x-axis {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      color: var(--muted);
      font-size: 12px;
      font-variant-numeric: tabular-nums;
      margin: 5px 0 0 106px;
      max-width: 980px;
    }
    .pnl-x-axis span {
      max-width: 46%;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .pnl-curve path.pnl-line {
      stroke: var(--blue);
      stroke-width: 2.2;
      fill: none;
      vector-effect: non-scaling-stroke;
    }
    .pnl-curve path.pnl-line.positive { stroke: var(--green); }
    .pnl-curve path.pnl-line.negative { stroke: var(--red); }
    .pnl-point {
      position: absolute;
      width: 10px;
      height: 10px;
      transform: translate(-50%, -50%);
      border: 2px solid #fff;
      border-radius: 999px;
      background: var(--blue);
      box-shadow: 0 0 0 1px rgba(23, 92, 211, 0.28);
    }
    .pnl-point.positive {
      background: var(--green);
      box-shadow: 0 0 0 1px rgba(36, 122, 82, 0.28);
    }
    .pnl-point.negative {
      background: var(--red);
      box-shadow: 0 0 0 1px rgba(180, 35, 24, 0.28);
    }
    .pnl-table .num {
      text-align: right;
      white-space: nowrap;
    }
    .pnl-curve circle {
      fill: var(--blue);
      stroke: #fff;
      stroke-width: 2;
      vector-effect: non-scaling-stroke;
    }
    .pnl-grid-line,
    .pnl-axis-line {
      stroke: #d0d5dd;
      vector-effect: non-scaling-stroke;
    }
    .pnl-grid-line {
      opacity: 0.58;
      stroke-dasharray: 3 5;
    }
    .empty-note {
      color: var(--muted);
      border: 1px dashed var(--line);
      padding: 10px;
      background: #fbfdff;
      border-radius: 7px;
    }
    .link-row { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 8px; }
    .job-status {
      margin-top: 8px;
      color: var(--muted);
      border: 1px solid var(--line);
      padding: 8px;
      min-height: 34px;
      border-radius: 7px;
    }
    .command-list {
      display: grid;
      gap: 6px;
    }
    .command-drawer {
      margin-top: 12px;
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
    }
    .command-drawer summary {
      cursor: pointer;
      padding: 10px 12px;
      font-weight: 700;
      background: #fbfcfe;
      border-bottom: 1px solid var(--line);
    }
    .command-drawer:not([open]) summary { border-bottom: 0; }
    .command-drawer-body { padding: 10px; }
    .command-row {
      display: grid;
      grid-template-columns: 150px minmax(0, 1fr) auto;
      gap: 8px;
      align-items: center;
      border: 1px solid var(--line);
      padding: 7px 8px;
      background: #fff;
      border-radius: 7px;
    }
    .command-name { font-weight: 700; }
    .command-text {
      color: var(--muted);
      overflow-wrap: anywhere;
      white-space: normal;
    }
    .debug-root {
      border: 1px solid var(--line);
      border-left: 4px solid var(--gray);
      border-radius: 8px;
      padding: 14px;
      background: #fff;
    }
    .debug-root.green { border-left-color: var(--green); background: #f8fffb; }
    .debug-root.yellow { border-left-color: var(--yellow); background: #fffdf5; }
    .debug-root.red { border-left-color: var(--red); background: #fff8f6; }
    .debug-root-title {
      display: flex;
      align-items: center;
      gap: 8px;
      flex-wrap: wrap;
      font-weight: 800;
      margin-bottom: 4px;
    }
    .debug-root-sentence {
      font-size: 16px;
      font-weight: 800;
      margin-bottom: 4px;
      overflow-wrap: anywhere;
    }
    .debug-guide-grid {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 8px;
      margin-top: 10px;
    }
    .debug-guide-item {
      border: 1px solid var(--line);
      border-radius: 7px;
      padding: 8px;
      background: rgba(255, 255, 255, 0.72);
      min-width: 0;
    }
    .debug-guide-label {
      color: var(--muted);
      font-size: 11px;
      font-weight: 800;
      text-transform: uppercase;
    }
    .debug-guide-value {
      margin-top: 2px;
      font-weight: 700;
      overflow-wrap: anywhere;
    }
    .audit-detail-drawer {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      overflow: hidden;
    }
    .audit-detail-drawer summary {
      cursor: pointer;
      padding: 9px 10px;
      font-weight: 800;
      background: #fbfcfe;
      border-bottom: 1px solid var(--line);
    }
    .audit-detail-drawer:not([open]) summary {
      border-bottom: 0;
    }
    .audit-detail-body {
      padding: 10px;
    }
    .logger-backdrop {
      position: fixed;
      inset: 0;
      z-index: 20;
      background: rgba(15, 23, 42, 0.22);
      display: none;
    }
    .logger-backdrop.open { display: block; }
    .logger-drawer {
      position: absolute;
      right: 0;
      top: 0;
      width: min(780px, 100vw);
      height: 100%;
      background: var(--panel);
      border-left: 1px solid var(--line-strong);
      box-shadow: -12px 0 30px rgba(15, 23, 42, 0.12);
      display: flex;
      flex-direction: column;
    }
    .logger-header {
      padding: 14px 16px;
      border-bottom: 1px solid var(--line);
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      gap: 12px;
    }
    .logger-title { font-weight: 800; font-size: 16px; overflow-wrap: anywhere; }
    .logger-controls {
      padding: 10px 16px;
      border-bottom: 1px solid var(--line);
      display: flex;
      justify-content: space-between;
      gap: 10px;
      flex-wrap: wrap;
    }
    .logger-filters { display: flex; gap: 6px; flex-wrap: wrap; }
    .logger-filter.active {
      color: var(--blue);
      border-color: var(--blue);
      background: var(--blue-soft);
    }
    .logger-body {
      padding: 12px 16px 16px;
      overflow: auto;
      flex: 1;
    }
    .logger-table-wrap { overflow: auto; }
    .logger-table {
      min-width: 680px;
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
    }
    a { color: var(--blue); }
    @media (max-width: 1180px) {
      th, td { white-space: normal; }
      table { min-width: 0; }
      .console-status-strip { grid-template-columns: repeat(3, minmax(120px, 1fr)); }
      .detail-header { display: block; }
      .inspect-actions { justify-content: flex-start; margin-top: 10px; }
      .tab-body { grid-template-columns: 1fr; }
      .metric-grid { grid-template-columns: repeat(2, minmax(120px, 1fr)); }
      .command-row { grid-template-columns: 1fr; }
    }
    @media (max-width: 680px) {
      main { padding: 10px; }
      header { padding: 0 10px; }
      .toolbar { flex-wrap: wrap; }
      .console-status-strip { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .attention-row { grid-template-columns: 1fr; }
      .metric-grid { grid-template-columns: 1fr; }
      .kv-grid { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <header>
    <h1>Live POD Dashboard</h1>
    <div class="muted" id="last-refresh">loading</div>
  </header>
  <main>
    <div class="toolbar">
      <label>Environment <select id="mode-filter"><option value="all">All</option></select></label>
      <button id="refresh-button">Refresh</button>
    </div>
    <section class="dashboard-section console-section" aria-label="Console status">
      <div class="dashboard-section-header">
        <div>
          <h2>Console Status</h2>
          <div class="dashboard-section-subtitle">Current view, queue pressure, refresh time, and selected POD.</div>
        </div>
        <span class="section-status">read-only / copy-only</span>
      </div>
      <div class="dashboard-section-body">
        <div class="console-status-strip" id="console-status-strip">
          <div class="console-status-item">
            <div class="console-status-label">Environment</div>
            <div class="console-status-value" id="console-mode-value">all</div>
          </div>
          <div class="console-status-item">
            <div class="console-status-label">Visible PODs</div>
            <div class="console-status-value" id="console-visible-pods-value">0</div>
          </div>
          <div class="console-status-item">
            <div class="console-status-label">Needs Action</div>
            <div class="console-status-value" id="console-needs-action-value">0</div>
          </div>
          <div class="console-status-item">
            <div class="console-status-label">Waiting</div>
            <div class="console-status-value" id="console-waiting-value">0</div>
          </div>
          <div class="console-status-item">
            <div class="console-status-label">Last Refresh</div>
            <div class="console-status-value" id="console-last-refresh-value">loading</div>
          </div>
          <div class="console-status-item">
            <div class="console-status-label">Selected POD</div>
            <div class="console-status-value" id="console-selected-pod-value">none</div>
          </div>
        </div>
      </div>
    </section>
    <section class="dashboard-section alert-panel" id="alert-panel">
      <div class="dashboard-section-header alert-header">
        <div>
          <h2>Alert Box</h2>
          <div class="dashboard-section-subtitle">Alert Inbox audit detail, collapsed by default.</div>
        </div>
        <div class="alert-header-actions">
          <div class="alert-counts" id="alert-counts"></div>
          <span class="muted" id="alert-subtitle">loading</span>
          <button class="secondary" id="alert-toggle-button">Show all</button>
        </div>
      </div>
      <div class="alert-list" id="alert-list"></div>
    </section>
    <section class="dashboard-section attention-panel" id="attention-panel">
      <div class="dashboard-section-header attention-header">
        <div>
          <h2>Attention Queue</h2>
          <div class="dashboard-section-subtitle">Primary triage: what is broken, waiting, healthy, or stale.</div>
        </div>
        <span class="section-status" id="attention-subtitle">loading</span>
      </div>
      <div class="attention-list" id="attention-queue-list"></div>
    </section>
    <section class="dashboard-section table-panel">
      <div class="dashboard-section-header">
        <div>
          <h2>POD List</h2>
          <div class="dashboard-section-subtitle">Dense scan table with readable state, reason, evidence, and safe inspect command.</div>
        </div>
        <span class="section-status">safe commands only</span>
      </div>
      <div class="dashboard-section-body flush">
        <div class="table-scroll">
        <table class="pod-table">
          <thead>
            <tr>
              <th>Severity</th>
              <th>POD</th>
              <th>Mode</th>
              <th>Current State</th>
              <th>Why</th>
              <th>Latest Evidence</th>
              <th>Next Inspect</th>
            </tr>
          </thead>
          <tbody id="pod-table-body"></tbody>
        </table>
        </div>
      </div>
    </section>
    <section id="detail-panel" class="dashboard-section detail-workspace empty-state">
      <div class="dashboard-section-header">
        <div>
          <h2>Selected POD / What We Learned</h2>
          <div class="dashboard-section-subtitle">Root cause, blocker chain, evidence, broker/model comparison, and safe commands.</div>
        </div>
        <span class="section-status">none selected</span>
      </div>
      <div class="dashboard-section-body">
        <h2>No POD selected</h2>
        <div class="muted">Select a POD row to inspect details.</div>
      </div>
    </section>
  </main>
  <div class="logger-backdrop" id="logger-backdrop" hidden>
    <aside class="logger-drawer" aria-label="POD logger">
      <div class="logger-header">
        <div>
          <div class="logger-title" id="logger-title">Logger</div>
          <div class="muted" id="logger-subtitle">No POD selected</div>
        </div>
        <button class="secondary" id="logger-close-button">Close</button>
      </div>
      <div class="logger-controls">
        <div class="logger-filters">
          <button class="secondary logger-filter active" data-logger-filter="all">All</button>
          <button class="secondary logger-filter" data-logger-filter="warnings">Warnings</button>
          <button class="secondary logger-filter" data-logger-filter="errors">Errors</button>
        </div>
        <button class="secondary" id="logger-refresh-button">Refresh Logger</button>
      </div>
      <div class="logger-body" id="logger-body">
        <div class="empty-note">Open a POD logger to inspect recent events.</div>
      </div>
    </aside>
  </div>
  <script>
    const detailTabNameList = ['debug', 'overview', 'pnl', 'decision', 'vplan', 'execution', 'broker', 'diff', 'freshness'];
    const detailTabLabelDict = {
      debug: 'Debug',
      overview: 'Overview',
      pnl: 'PnL',
      decision: 'Decision',
      vplan: 'VPlan',
      execution: 'Execution',
      broker: 'Broker',
      diff: 'DIFF',
      freshness: 'Freshness'
    };
    const DASHBOARD_TIME_ZONE_STR = 'America/New_York';
    const DASHBOARD_TIME_ZONE_LABEL_STR = 'NYC';
    const DASHBOARD_TIMESTAMP_PATTERN = /\\d{4}-\\d{2}-\\d{2}T\\d{2}:\\d{2}:\\d{2}(?:\\.\\d+)?(?:Z|[+-]\\d{2}:\\d{2})?/g;
    const DASHBOARD_COMPACT_TIMESTAMP_PATTERN = /\\b\\d{8}T\\d{6}Z\\b/g;
    const safeInspectCommandNameList = ['status', 'next_due', 'show_decision_plan', 'show_vplan', 'compare_reference'];
    const initialUiState = readUiState();
    const state = {
      pods: [],
      alerts: [],
      alertSummary: {},
      podChangeDict: {},
      selectedPod: initialUiState.podId,
      selectedDetail: null,
      jobs: {},
      requestedMode: initialUiState.mode || 'all',
      lastRefreshDisplayStr: 'loading',
      selectedTab: null,
      alertExpanded: false,
      loggerPodId: null,
      loggerEvents: [],
      loggerFilter: 'all',
      loggerLoading: false
    };
    const tbody = document.getElementById('pod-table-body');
    const modeFilter = document.getElementById('mode-filter');
    const detailPanel = document.getElementById('detail-panel');
    const lastRefresh = document.getElementById('last-refresh');
    const alertCounts = document.getElementById('alert-counts');
    const alertList = document.getElementById('alert-list');
    const alertSubtitle = document.getElementById('alert-subtitle');
    const alertToggleButton = document.getElementById('alert-toggle-button');
    const alertPanel = document.getElementById('alert-panel');
    const attentionQueueList = document.getElementById('attention-queue-list');
    const attentionSubtitle = document.getElementById('attention-subtitle');
    const consoleModeValue = document.getElementById('console-mode-value');
    const consoleVisiblePodsValue = document.getElementById('console-visible-pods-value');
    const consoleNeedsActionValue = document.getElementById('console-needs-action-value');
    const consoleWaitingValue = document.getElementById('console-waiting-value');
    const consoleLastRefreshValue = document.getElementById('console-last-refresh-value');
    const consoleSelectedPodValue = document.getElementById('console-selected-pod-value');
    const loggerBackdrop = document.getElementById('logger-backdrop');
    const loggerBody = document.getElementById('logger-body');
    const loggerTitle = document.getElementById('logger-title');
    const loggerSubtitle = document.getElementById('logger-subtitle');
    document.getElementById('refresh-button').addEventListener('click', refreshPods);
    alertToggleButton.addEventListener('click', () => {
      state.alertExpanded = !state.alertExpanded;
      renderAlertInbox();
    });
    document.getElementById('logger-close-button').addEventListener('click', closeLogger);
    document.getElementById('logger-refresh-button').addEventListener('click', refreshLoggerEvents);
    loggerBackdrop.addEventListener('click', event => {
      if (event.target === loggerBackdrop) closeLogger();
    });
    document.querySelectorAll('[data-logger-filter]').forEach(button => {
      button.addEventListener('click', () => {
        state.loggerFilter = button.dataset.loggerFilter || 'all';
        renderLoggerDrawer();
      });
    });
    modeFilter.addEventListener('change', () => {
      state.requestedMode = modeFilter.value;
      const selectedClearedBool = clearSelectedPodIfHidden();
      persistUiState();
      syncUrlState();
      if (selectedClearedBool) renderNoPodSelectedDetail();
      renderPods();
      renderAlertInbox();
      renderAttentionQueue();
      renderConsoleStatusStrip();
    });

    function readUiState() {
      const params = new URLSearchParams(window.location.search);
      const modeFromUrl = params.get('mode');
      window.localStorage.removeItem('dashboard.pod');
      return {
        mode: modeFromUrl || window.localStorage.getItem('dashboard.mode') || 'all',
        podId: null
      };
    }

    function persistUiState() {
      window.localStorage.setItem('dashboard.mode', state.requestedMode || 'all');
      window.localStorage.removeItem('dashboard.pod');
    }

    function syncUrlState() {
      const url = new URL(window.location.href);
      if (state.requestedMode && state.requestedMode !== 'all') {
        url.searchParams.set('mode', state.requestedMode);
      } else {
        url.searchParams.delete('mode');
      }
      url.searchParams.delete('pod');
      window.history.replaceState({}, '', url.toString());
    }

    function getStoredDetailTab(podId) {
      if (!podId) return 'debug';
      const storedTab = window.localStorage.getItem('dashboard.tab.' + podId);
      return detailTabNameList.includes(storedTab) ? storedTab : 'debug';
    }

    function setActiveDetailTab(podId, tabName) {
      const cleanTabName = detailTabNameList.includes(tabName) ? tabName : 'debug';
      state.selectedTab = cleanTabName;
      if (podId) {
        window.localStorage.setItem('dashboard.tab.' + podId, cleanTabName);
      }
      if (state.selectedDetail) {
        renderDetail(state.selectedDetail);
      }
    }

    function activeDetailTab(podId) {
      const cleanTabName = detailTabNameList.includes(state.selectedTab) ? state.selectedTab : getStoredDetailTab(podId);
      return detailTabNameList.includes(cleanTabName) ? cleanTabName : 'debug';
    }

    function fmt(value) {
      if (value === null || value === undefined || value === '') return '-';
      if (typeof value === 'number') {
        if (Math.abs(value) >= 1000) return value.toLocaleString(undefined, { maximumFractionDigits: 2 });
        return value.toLocaleString(undefined, { maximumFractionDigits: 4 });
      }
      if (typeof value === 'string') return formatTimestampText(value);
      return String(value);
    }

    function formatTimestamp(value) {
      if (value === null || value === undefined || value === '') return '-';
      const rawTimestampStr = String(value);
      const normalizedTimestampStr = /(?:Z|[+-]\\d{2}:\\d{2})$/.test(rawTimestampStr)
        ? rawTimestampStr
        : rawTimestampStr + 'Z';
      const timestampDate = new Date(normalizedTimestampStr);
      if (Number.isNaN(timestampDate.getTime())) return rawTimestampStr;
      const partDict = {};
      new Intl.DateTimeFormat('en-CA', {
        timeZone: DASHBOARD_TIME_ZONE_STR,
        year: 'numeric',
        month: '2-digit',
        day: '2-digit',
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit',
        hourCycle: 'h23',
      }).formatToParts(timestampDate).forEach(part => {
        partDict[part.type] = part.value;
      });
      return `${partDict.year}-${partDict.month}-${partDict.day} ${partDict.hour}:${partDict.minute}:${partDict.second} ${DASHBOARD_TIME_ZONE_LABEL_STR}`;
    }

    function formatCompactTimestamp(value) {
      const rawTimestampStr = String(value);
      const isoTimestampStr = `${rawTimestampStr.slice(0, 4)}-${rawTimestampStr.slice(4, 6)}-${rawTimestampStr.slice(6, 8)}T${rawTimestampStr.slice(9, 11)}:${rawTimestampStr.slice(11, 13)}:${rawTimestampStr.slice(13, 15)}Z`;
      return formatTimestamp(isoTimestampStr);
    }

    function formatTimestampText(value) {
      const rawTextStr = String(value);
      return rawTextStr
        .replace(DASHBOARD_COMPACT_TIMESTAMP_PATTERN, timestampStr => formatCompactTimestamp(timestampStr))
        .replace(DASHBOARD_TIMESTAMP_PATTERN, timestampStr => formatTimestamp(timestampStr));
    }

    function fmtPct(value) {
      if (value === null || value === undefined || value === '') return '-';
      return (Number(value) * 100).toLocaleString(undefined, { maximumFractionDigits: 4 }) + '%';
    }

    function fmtBool(value) {
      if (value === null || value === undefined) return '-';
      return value ? 'yes' : 'no';
    }

    function fmtUnavailable(value) {
      if (value === null || value === undefined || value === '') return 'unavailable';
      return fmt(value);
    }

    function formatValue(value) {
      if (value === null || value === undefined || value === '') return '-';
      if (typeof value === 'number') return fmt(value);
      if (typeof value === 'boolean') return fmtBool(value);
      if (Array.isArray(value)) return value.length ? value.map(formatValue).join(', ') : '-';
      if (typeof value === 'object') {
        const entries = Object.entries(value);
        if (!entries.length) return '-';
        return entries.map(([key, item]) => key + '=' + formatValue(item)).join(' | ');
      }
      return formatTimestampText(value);
    }

    function esc(value) {
      return formatValue(value)
        .replaceAll('&', '&amp;')
        .replaceAll('<', '&lt;')
        .replaceAll('>', '&gt;')
        .replaceAll('"', '&quot;')
        .replaceAll("'", '&#39;');
    }

    function pill(label, cls) {
      const cleanCls = ['green', 'yellow', 'red', 'gray'].includes(cls) ? cls : 'gray';
      return `<span class="pill ${cleanCls}">${esc(label || cleanCls)}</span>`;
    }

    function healthClass(status) {
      if (status === 'red') return 'red';
      if (status === 'yellow') return 'yellow';
      if (status === 'green') return 'green';
      return 'gray';
    }

    function renderNoPodSelectedDetail() {
      detailPanel.className = 'dashboard-section detail-workspace empty-state';
      detailPanel.innerHTML = `
        <div class="dashboard-section-header">
          <div>
            <h2>Selected POD / What We Learned</h2>
            <div class="dashboard-section-subtitle">Root cause, blocker chain, evidence, broker/model comparison, and safe commands.</div>
          </div>
          <span class="section-status">none selected</span>
        </div>
        <div class="dashboard-section-body">
          <h2>No POD selected</h2>
          <div class="muted">Select a POD row to inspect details.</div>
        </div>`;
    }

    function renderDetailError(podId, message) {
      detailPanel.className = 'dashboard-section detail-workspace';
      detailPanel.innerHTML = `
        <div class="dashboard-section-header">
          <div>
            <h2>Selected POD / What We Learned</h2>
            <div class="dashboard-section-subtitle">${esc(podId)}</div>
          </div>
          <span class="section-status">detail load failed</span>
        </div>
        <div class="dashboard-section-body">
          <div class="red pill">error</div>
          <div class="job-status">${esc(message)}</div>
        </div>`;
    }

    async function fetchJson(url, options) {
      const response = await fetch(url, options || {});
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.message_str || response.statusText);
      return payload;
    }

    async function refreshPods() {
      try {
        const payload = await fetchJson('/api/pods');
        const nextPodRows = payload.pod_row_dict_list || [];
        state.podChangeDict = buildPodChangeDict(state.pods || [], nextPodRows);
        state.pods = nextPodRows;
        state.alerts = payload.alert_dict_list || [];
        state.alertSummary = payload.alert_summary_dict || {};
        state.lastRefreshDisplayStr = formatTimestamp(new Date().toISOString());
        lastRefresh.textContent = 'Last refresh: ' + state.lastRefreshDisplayStr;
        syncModeFilter(payload.mode_list || []);
        syncUrlState();
        if (clearSelectedPodIfHidden()) {
          persistUiState();
          syncUrlState();
          renderNoPodSelectedDetail();
        }
        renderPods();
        renderAlertInbox();
        renderAttentionQueue();
        renderConsoleStatusStrip();
        if (state.selectedPod && !state.selectedDetail) {
          loadDetail(state.selectedPod);
        }
      } catch (error) {
        lastRefresh.textContent = 'Refresh failed: ' + error.message;
      }
    }

    function syncModeFilter(modeList) {
      const environmentList = Array.from(new Set([...(modeList || []), 'live', 'paper', 'incubation'])).sort();
      modeFilter.innerHTML = '<option value="all">All</option>' + environmentList.map(mode => `<option value="${esc(mode)}">${esc(environmentLabel(mode))}</option>`).join('');
      modeFilter.value = environmentList.includes(state.requestedMode) ? state.requestedMode : 'all';
      state.requestedMode = modeFilter.value;
    }

    function environmentLabel(mode) {
      if (mode === 'live') return 'Live';
      if (mode === 'paper') return 'Paper';
      if (mode === 'incubation') return 'Incubation';
      return mode;
    }

    function buildPodChangeDict(previousPodRows, nextPodRows) {
      const previousSignatureDict = {};
      (previousPodRows || []).forEach(row => {
        previousSignatureDict[row.pod_id_str] = podSignature(row);
      });
      const changeDict = {};
      (nextPodRows || []).forEach(row => {
        if (Object.prototype.hasOwnProperty.call(previousSignatureDict, row.pod_id_str)) {
          changeDict[row.pod_id_str] = previousSignatureDict[row.pod_id_str] !== podSignature(row);
        }
      });
      return changeDict;
    }

    function podSignature(row) {
      const debugSummary = row.debug_summary_dict || {};
      return [
        debugSummary.severity_str || '',
        debugSummary.verdict_label_str || '',
        debugSummary.primary_reason_str || '',
        debugSummary.primary_evidence_str || '',
        debugSummary.latest_evidence_timestamp_str || '',
        row.health_str || '',
        row.next_action_str || '',
        row.reason_code_str || '',
        row.latest_reconciliation_status_str || '',
        (row.eod_snapshot_dict || {}).status_str || '',
        row.latest_diff_status_str || '',
        row.latest_event_timestamp_str || '',
      ].join('||');
    }

    function hasPodChangedSinceLastRefresh(row) {
      return !!state.podChangeDict[row.pod_id_str];
    }

    function safeInspectCommandName(commandName) {
      return safeInspectCommandNameList.includes(commandName) ? commandName : 'status';
    }

    function operatorQueueGroup(row) {
      const debugSummary = row.debug_summary_dict || {};
      const verdictLabel = String(debugSummary.verdict_label_str || '');
      const verdictLower = verdictLabel.toLowerCase();
      const dbStatus = String(row.db_status_str || '');
      const severity = healthClass(debugSummary.severity_str || row.health_str);
      if (dbStatus && dbStatus !== 'ok') return 'Missing / Stale Data';
      if (verdictLower.includes('db ') || verdictLower.includes('stale') || verdictLower.includes('freshness')) return 'Missing / Stale Data';
      if (verdictLower.includes('waiting') || verdictLower.includes('parked') || row.next_action_str === 'wait') return 'Waiting / Parked';
      if (severity === 'green') return 'Healthy';
      if (severity === 'red' || severity === 'yellow') return 'Needs Action';
      return 'Missing / Stale Data';
    }

    function operatorQueuePriority(groupName) {
      const priorityDict = {
        'Needs Action': 0,
        'Waiting / Parked': 1,
        'Missing / Stale Data': 2,
        Healthy: 3,
      };
      return priorityDict[groupName] === undefined ? 9 : priorityDict[groupName];
    }

    function severityPriority(row) {
      const severity = healthClass((row.debug_summary_dict || {}).severity_str || row.health_str);
      if (severity === 'red') return 0;
      if (severity === 'yellow') return 1;
      if (severity === 'gray') return 2;
      return 3;
    }

    function buildAttentionQueueRows(podRows) {
      return (podRows || [])
        .map(row => ({
          row,
          group_name_str: operatorQueueGroup(row),
          state_sentence_str: operatorStateSentence(row),
          reason_sentence_str: operatorReasonSentence(row),
          evidence_sentence_str: operatorEvidenceSentence(row),
          command_name_str: safeInspectCommandName((row.debug_summary_dict || {}).next_inspect_command_name_str),
        }))
        .sort((left, right) => {
          const groupDelta = operatorQueuePriority(left.group_name_str) - operatorQueuePriority(right.group_name_str);
          if (groupDelta !== 0) return groupDelta;
          const severityDelta = severityPriority(left.row) - severityPriority(right.row);
          if (severityDelta !== 0) return severityDelta;
          return String(right.row.latest_event_timestamp_str || '').localeCompare(String(left.row.latest_event_timestamp_str || ''));
        });
    }

    function visibleOperatorRows() {
      const selectedMode = state.requestedMode || modeFilter.value || 'all';
      if (selectedMode === 'all') return state.pods;
      return state.pods.filter(row => rowMatchesSelectedEnvironment(row));
    }

    function rowMatchesSelectedEnvironment(row) {
      const selectedMode = state.requestedMode || modeFilter.value || 'all';
      return selectedMode === 'all' || row.mode_str === selectedMode;
    }

    function renderConsoleStatusStrip() {
      const visibleRows = visibleOperatorRows();
      const needsActionCount = visibleRows.filter(row => operatorQueueGroup(row) === 'Needs Action').length;
      const waitingCount = visibleRows.filter(row => operatorQueueGroup(row) === 'Waiting / Parked').length;
      consoleModeValue.textContent = environmentLabel(state.requestedMode || modeFilter.value || 'all');
      consoleVisiblePodsValue.textContent = String(visibleRows.length);
      consoleNeedsActionValue.textContent = String(needsActionCount);
      consoleWaitingValue.textContent = String(waitingCount);
      consoleLastRefreshValue.textContent = state.lastRefreshDisplayStr || 'loading';
      consoleSelectedPodValue.textContent = state.selectedPod || 'none';
    }

    function renderPods() {
      const rows = visibleOperatorRows();
      if (!rows.length) {
        tbody.innerHTML = `<tr><td colspan="7"><div class="empty-note">${esc(emptyOperatorStateMessage())}</div></td></tr>`;
        return;
      }
      tbody.innerHTML = rows.map(row => `
        <tr data-pod="${esc(row.pod_id_str)}" data-mode="${esc(row.mode_str)}" class="${row.pod_id_str === state.selectedPod ? 'selected' : ''} ${hasPodChangedSinceLastRefresh(row) ? 'changed' : ''}">
          <td class="severity-cell">${renderOperatorSeverityCell(row)}</td>
          <td><span class="pod-link">${esc(row.pod_id_str)}</span><br><span class="muted strategy-text">${esc(row.strategy_import_str)}</span></td>
          <td>${esc(row.mode_str)}</td>
          <td class="operator-cell">${renderOperatorStateCell(row)}</td>
          <td class="operator-cell">${esc(operatorReasonSentence(row))}</td>
          <td class="operator-cell">${esc(operatorEvidenceSentence(row))}</td>
          <td class="operator-command-cell">${renderOperatorCommandCell(row)}</td>
        </tr>`).join('');
      tbody.querySelectorAll('tr[data-pod]').forEach(el => {
        el.addEventListener('click', event => {
          if (event.target.closest('button')) return;
          selectPod(el.dataset.pod, el.dataset.mode);
        });
      });
      tbody.querySelectorAll('[data-copy-command]').forEach(button => {
        button.addEventListener('click', event => {
          event.stopPropagation();
          const podId = button.dataset.copyPod;
          const row = state.pods.find(item => item.pod_id_str === podId);
          if (row) copyCommand(row, button.dataset.copyCommand, button);
        });
      });
    }

    function renderAlertInbox() {
      const summary = state.alertSummary || {};
      const alertRows = state.alerts || [];
      alertCounts.innerHTML = `
        <span class="count-pill red">Red ${esc(summary.red_count_int || 0)}</span>
        <span class="count-pill yellow">Yellow ${esc(summary.yellow_count_int || 0)}</span>
        <span class="count-pill gray">Gray ${esc(summary.gray_count_int || 0)}</span>`;
      alertPanel.classList.toggle('collapsed', !state.alertExpanded);
      alertSubtitle.textContent = state.alertExpanded
        ? (summary.total_count_int || 0) + ' audit alerts shown'
        : (summary.total_count_int || 0) + ' audit alerts collapsed';
      alertToggleButton.textContent = state.alertExpanded ? 'Hide audit alerts' : 'Show audit alerts';
      alertToggleButton.disabled = !alertRows.length;
      if (!state.alertExpanded) {
        alertList.hidden = true;
        alertList.innerHTML = '';
        return;
      }
      alertList.hidden = false;
      if (!alertRows.length) {
        alertList.innerHTML = '<div class="empty-note">No red, yellow, or setup alerts.</div>';
        return;
      }
      alertList.innerHTML = alertRows.map(alert => `
        <button class="alert-row ${healthClass(alert.severity_str)}" data-alert-pod="${esc(alert.pod_id_str)}" data-alert-mode="${esc(alert.mode_str)}">
          <div>
            ${pill(alert.severity_str, healthClass(alert.severity_str))}
            <div class="alert-type">${esc(alert.alert_type_str)}</div>
          </div>
          <div class="alert-main">
            <div class="alert-label">${esc(alert.label_str)}</div>
            <div class="alert-meta">${esc(alert.mode_str)} / ${esc(alert.pod_id_str)} / ${esc(alert.account_route_str)}</div>
            <div class="alert-reason">${esc(alert.reason_str)}${alert.inspect_command_name_str ? ' | inspect: ' + esc(alert.inspect_command_name_str) : ''}</div>
          </div>
        </button>`).join('');
      alertList.querySelectorAll('[data-alert-pod]').forEach(el => {
        el.addEventListener('click', () => selectPod(el.dataset.alertPod, el.dataset.alertMode));
      });
    }

    function renderAttentionQueue() {
      const visibleRows = visibleOperatorRows();
      const queueRows = buildAttentionQueueRows(visibleRows);
      attentionSubtitle.textContent = queueRows.length + ' PODs grouped by operator state';
      if (!queueRows.length) {
        attentionQueueList.innerHTML = `<div class="empty-note">${esc(emptyOperatorStateMessage())}</div>`;
        return;
      }
      let currentGroupName = null;
      const htmlList = [];
      queueRows.forEach(queueRow => {
        if (queueRow.group_name_str !== currentGroupName) {
          currentGroupName = queueRow.group_name_str;
          htmlList.push(`<div class="attention-group-title">${esc(currentGroupName)}</div>`);
        }
        htmlList.push(renderAttentionQueueRow(queueRow));
      });
      attentionQueueList.innerHTML = htmlList.join('');
      attentionQueueList.querySelectorAll('[data-attention-pod]').forEach(el => {
        el.addEventListener('click', event => {
          if (event.target.closest('button')) return;
          selectPod(el.dataset.attentionPod, el.dataset.attentionMode);
        });
      });
      attentionQueueList.querySelectorAll('[data-copy-command]').forEach(button => {
        button.addEventListener('click', event => {
          event.stopPropagation();
          const podId = button.dataset.copyPod;
          const row = state.pods.find(item => item.pod_id_str === podId);
          if (row) copyCommand(row, button.dataset.copyCommand, button);
        });
      });
    }

    function renderAttentionQueueRow(queueRow) {
      const row = queueRow.row;
      const debugSummary = row.debug_summary_dict || {};
      const severity = healthClass(debugSummary.severity_str || row.health_str);
      const changedHtml = hasPodChangedSinceLastRefresh(row) ? '<div class="operator-changed">Changed since last refresh</div>' : '';
      return `
        <div class="attention-row ${severity}" data-attention-pod="${esc(row.pod_id_str)}" data-attention-mode="${esc(row.mode_str)}">
          <div>${pill(debugSummary.severity_str || severity, severity)}</div>
          <div class="attention-main">
            <div class="attention-title">${esc(row.pod_id_str)} - ${esc(queueRow.state_sentence_str)}</div>
            <div class="attention-meta">${esc(row.mode_str)} / ${esc(row.account_route_str)}</div>
            <div class="attention-reason">${esc(queueRow.reason_sentence_str)}</div>
            <div class="attention-evidence">${esc(queueRow.evidence_sentence_str)}</div>
            ${changedHtml}
          </div>
          <div class="attention-command">
            <button class="secondary" data-copy-pod="${esc(row.pod_id_str)}" data-copy-command="${esc(queueRow.command_name_str)}">Copy ${esc(queueRow.command_name_str)}</button>
          </div>
        </div>`;
    }

    function emptyOperatorStateMessage() {
      const selectedMode = state.requestedMode || modeFilter.value || 'all';
      if (selectedMode === 'incubation') {
        return 'No incubation PODs are currently enabled. Dashboard reads per-POD incubation state under alpha/live/state/incubation/{pod_id}.sqlite3.';
      }
      if (selectedMode === 'live') return 'No live PODs are currently enabled.';
      if (selectedMode === 'paper') return 'No paper PODs are currently enabled.';
      return 'No PODs are currently enabled. The dashboard is read-only; use status if you need to inspect config and state paths.';
    }

    function clearSelectedPodState() {
      state.selectedPod = null;
      state.selectedDetail = null;
      state.selectedTab = null;
    }

    function clearSelectedPod() {
      clearSelectedPodState();
      persistUiState();
      syncUrlState();
      renderPods();
      renderAttentionQueue();
      renderConsoleStatusStrip();
      renderNoPodSelectedDetail();
    }

    function clearSelectedPodIfHidden() {
      if (!state.selectedPod) return false;
      const selectedRow = state.pods.find(row => row.pod_id_str === state.selectedPod);
      if (selectedRow && rowMatchesSelectedEnvironment(selectedRow)) return false;
      clearSelectedPodState();
      return true;
    }

    function selectPod(podId, mode) {
      const previousPodId = state.selectedPod;
      state.selectedPod = podId;
      state.selectedDetail = null;
      if (previousPodId !== podId) {
        state.selectedTab = getStoredDetailTab(podId);
      }
      if (mode) {
        state.requestedMode = mode;
        modeFilter.value = mode;
      }
      persistUiState();
      syncUrlState();
      renderPods();
      renderConsoleStatusStrip();
      loadDetail(podId);
    }

    function operatorStateSentence(row) {
      const debugSummary = row.debug_summary_dict || {};
      const verdictLabel = String(debugSummary.verdict_label_str || '');
      const verdictLower = verdictLabel.toLowerCase();
      const eodStatus = String((row.eod_snapshot_dict || {}).status_str || '');
      const rehearsal = row.rehearsal_status_dict || {};
      const gateStatus = String(rehearsal.promotion_gate_status_str || '');
      if (row.mode_str === 'incubation' && row.db_status_str === 'missing') return 'Incubation SIM ledger DB has not been created yet.';
      if (row.mode_str === 'incubation' && row.db_status_str === 'empty') return 'Incubation DB exists, but no rehearsal cycle has been recorded yet.';
      if (row.mode_str === 'incubation' && gateStatus === 'complete_one_cycle') return 'Incubation completed at least one SIM ledger rehearsal cycle.';
      if (row.mode_str === 'incubation' && gateStatus === 'incomplete') return 'Incubation is waiting for one complete SIM ledger rehearsal cycle.';
      if (verdictLower.includes('reconcile blocked')) return 'Reconcile is blocked.';
      if (verdictLower.includes('waiting reconcile')) return 'Reconcile is waiting for broker truth.';
      if (verdictLower.includes('diff red')) return 'Reference DIFF is red.';
      if (verdictLower.includes('db missing')) return 'State DB is missing.';
      if (verdictLower.includes('db empty')) return 'State DB is empty.';
      if (eodStatus === 'blocked_by_execution') return 'EOD is blocked because execution is unresolved.';
      if (verdictLabel) return verdictLabel.endsWith('.') ? verdictLabel : verdictLabel + '.';
      if (row.next_action_str) return 'Next action is ' + row.next_action_str + '.';
      return 'State is unknown.';
    }

    function operatorReasonSentence(row) {
      const debugSummary = row.debug_summary_dict || {};
      const eodStatus = String((row.eod_snapshot_dict || {}).status_str || '');
      const rehearsal = row.rehearsal_status_dict || {};
      const gateStatus = String(rehearsal.promotion_gate_status_str || '');
      if (row.mode_str === 'incubation' && gateStatus === 'complete_one_cycle') return 'Incubation completed at least one SIM ledger rehearsal cycle.';
      if (row.mode_str === 'incubation' && gateStatus === 'incomplete') return 'Incubation is waiting for one complete SIM ledger rehearsal cycle.';
      if (eodStatus === 'blocked_by_execution') return 'EOD is blocked because execution is unresolved.';
      if (debugSummary.primary_reason_str) return debugSummary.primary_reason_str;
      if (row.reason_code_str) return row.reason_code_str;
      return 'No operator reason was reported.';
    }

    function operatorEvidenceSentence(row) {
      const debugSummary = row.debug_summary_dict || {};
      const timestampStr = debugSummary.latest_evidence_timestamp_str || row.latest_event_timestamp_str || row.latest_pod_state_timestamp_str || '-';
      const evidenceStr = debugSummary.primary_evidence_str || row.latest_reconciliation_status_str || row.latest_diff_status_str || row.db_status_str || '-';
      return 'Latest Evidence: ' + formatTimestampText(timestampStr) + ' / ' + evidenceStr;
    }

    function renderOperatorSeverityCell(row) {
      const debugSummary = row.debug_summary_dict || {};
      const severity = healthClass(debugSummary.severity_str || row.health_str);
      return `${pill(debugSummary.severity_str || severity, severity)}${hasPodChangedSinceLastRefresh(row) ? '<div class="operator-changed">Changed since last refresh</div>' : ''}`;
    }

    function renderOperatorStateCell(row) {
      return `
        <div class="operator-state">${esc(operatorStateSentence(row))}</div>
        <div class="operator-reason">${esc((row.debug_summary_dict || {}).verdict_label_str || '-')}</div>`;
    }

    function renderOperatorCommandCell(row) {
      const commandName = safeInspectCommandName((row.debug_summary_dict || {}).next_inspect_command_name_str);
      return `<button class="secondary" data-copy-pod="${esc(row.pod_id_str)}" data-copy-command="${esc(commandName)}">Copy ${esc(commandName)}</button>`;
    }

    function renderRequiredActionCell(action) {
      const severity = healthClass(action.severity_str);
      const inspectHtml = action.inspect_command_name_str
        ? `<div class="summary-action-inspect">inspect: ${esc(action.inspect_command_name_str)}</div>`
        : '';
      return `
        <div class="summary-action">
          ${pill(action.severity_str || severity, severity)}
          <div class="summary-action-main">
            <div class="summary-action-label">${esc(action.label_str || 'Unknown')}</div>
            <div class="summary-action-reason">${esc(action.reason_str)}</div>
            ${inspectHtml}
          </div>
        </div>`;
    }

    function renderDebugVerdictCell(debugSummary) {
      const severity = healthClass(debugSummary.severity_str);
      return `
        <div class="debug-verdict">
          ${pill(debugSummary.severity_str || severity, severity)}
          <div>
            <div class="debug-verdict-label">${esc(debugSummary.verdict_label_str || 'Unknown')}</div>
            <div class="debug-verdict-reason">${esc(debugSummary.primary_reason_str || '-')}</div>
            <div class="debug-verdict-evidence">${esc(debugSummary.primary_evidence_str || '-')}</div>
            <div class="debug-verdict-evidence">evidence: ${esc(debugSummary.latest_evidence_timestamp_str || '-')}</div>
          </div>
        </div>`;
    }

    const lifecycleExecutionStepKeyList = ['db', 'decision', 'vplan', 'ack', 'fill', 'reconcile', 'eod'];
    const lifecycleActionStepKeyByLabel = {
      'Setup DB': 'db',
      'No state yet': 'db',
      'Build DecisionPlan': 'decision',
      'Build VPlan': 'vplan',
      'Wait submission window': 'vplan',
      'VPlan ready': 'vplan',
      'Review VPlan': 'vplan',
      'Review broker ACK': 'ack',
      'Waiting reconcile': 'reconcile',
      'Review reconcile': 'reconcile',
      'Expire stale plan': 'vplan',
    };
    const lifecycleActionStepKeyByCommand = {
      next_due: 'decision',
      show_decision_plan: 'vplan',
      show_vplan: 'vplan',
      status: 'db',
    };

    function lifecycleExecutionStepList(stepList) {
      const stepByKey = new Map((stepList || []).map(step => [step.step_key_str, step]));
      return lifecycleExecutionStepKeyList
        .map(stepKey => stepByKey.get(stepKey))
        .filter(Boolean);
    }

    function isLifecycleActiveWaitingStep(step) {
      const stepKey = step.step_key_str || '';
      const status = String(step.status_str || '').toLowerCase();
      const severity = healthClass(step.severity_str);
      if (severity !== 'gray') return false;
      if (stepKey === 'eod') return ['waiting', 'blocked_by_execution'].includes(status);
      return false;
    }

    function deriveCurrentLifecycleStepKey(stepList, action) {
      const executionStepList = lifecycleExecutionStepList(stepList || []);
      const firstRedStep = executionStepList.find(step => healthClass(step.severity_str) === 'red');
      if (firstRedStep) return firstRedStep.step_key_str;
      const firstYellowStep = executionStepList.find(step => healthClass(step.severity_str) === 'yellow');
      if (firstYellowStep) return firstYellowStep.step_key_str;
      if (action && healthClass(action.severity_str) !== 'green') {
        const actionStepKey = lifecycleActionStepKeyByLabel[action.label_str]
          || lifecycleActionStepKeyByCommand[action.inspect_command_name_str];
        if (actionStepKey && executionStepList.some(step => step.step_key_str === actionStepKey)) {
          return actionStepKey;
        }
      }
      const firstActiveWaitingStep = executionStepList.find(isLifecycleActiveWaitingStep);
      if (firstActiveWaitingStep) return firstActiveWaitingStep.step_key_str;
      return '';
    }

    function operatorLifecycleSentence(step, isCurrent) {
      if (!step) return 'Idle / no blocking lifecycle step.';
      const label = step.label_str || step.step_key_str || 'Lifecycle';
      const severity = healthClass(step.severity_str);
      if (isCurrent) {
        if (severity === 'red') return `${label} is the current blocker.`;
        if (severity === 'yellow') return `${label} is the current waiting or inspect step.`;
        if (severity === 'gray') return `${label} is the current not-yet-started step.`;
        return `${label} is current and passed.`;
      }
      if (severity === 'green') return `${label} completed or passed.`;
      if (severity === 'yellow') return `${label} is waiting or needs inspection.`;
      if (severity === 'red') return `${label} is blocked or failed.`;
      return `${label} is not reached or not applicable.`;
    }

    function renderLifecycleArrow() {
      return '<div class="lifecycle-arrow" aria-hidden="true">&rarr;</div>';
    }

    function renderLifecycleStepCard(step, isCurrent) {
      const severity = healthClass(step.severity_str);
      const currentBadgeHtml = isCurrent ? '<span class="lifecycle-current-badge">Current</span>' : '';
      const timestampHtml = step.timestamp_str ? `<div class="lifecycle-step-detail">Evidence: ${esc(step.timestamp_str)}</div>` : '';
      return `
        <div class="lifecycle-step-card ${severity} ${isCurrent ? 'current' : ''}">
          <div class="lifecycle-step-top">
            <div class="lifecycle-step-label">${esc(step.label_str)}</div>
            ${currentBadgeHtml}
          </div>
          <div class="lifecycle-step-status">${esc(step.status_str || '-')}</div>
          <div class="lifecycle-step-sentence">${esc(operatorLifecycleSentence(step, isCurrent))}</div>
          <div class="lifecycle-step-detail">${esc(step.detail_str || '-')}</div>
          ${timestampHtml}
        </div>`;
    }

    function renderReferenceCheckStep(diffStep) {
      if (!diffStep) {
        return `
          <div class="lifecycle-reference-check gray">
            <div>
              <div class="lifecycle-reference-title">Reference Check</div>
              <div class="lifecycle-reference-detail">No DIFF evidence is available.</div>
            </div>
            ${pill('not_run', 'gray')}
          </div>`;
      }
      const severity = healthClass(diffStep.severity_str);
      const timestampHtml = diffStep.timestamp_str ? ` / ${esc(diffStep.timestamp_str)}` : '';
      return `
        <div class="lifecycle-reference-check ${severity}">
          <div>
            <div class="lifecycle-reference-title">Reference Check</div>
            <div class="lifecycle-reference-detail">${esc(diffStep.detail_str || '-')}${timestampHtml}</div>
          </div>
          ${pill(diffStep.status_str || severity, severity)}
        </div>`;
    }

    function renderLifecycleRail(stepList, action) {
      const executionStepList = lifecycleExecutionStepList(stepList || []);
      const diffStep = (stepList || []).find(step => step.step_key_str === 'diff');
      const currentStepKey = deriveCurrentLifecycleStepKey(stepList || [], action || {});
      const currentStep = executionStepList.find(step => step.step_key_str === currentStepKey);
      const currentSummary = currentStep
        ? `Current: ${currentStep.label_str} / ${currentStep.status_str || '-'}`
        : 'Idle / no blocking lifecycle step';
      if (!executionStepList.length) {
        return `
          <div class="lifecycle-progress">
            <div class="lifecycle-current-summary">
              <span>Idle / no blocking lifecycle step</span>
              <span class="muted">No lifecycle evidence yet.</span>
            </div>
            ${renderReferenceCheckStep(diffStep)}
          </div>`;
      }
      const railHtml = executionStepList.map((step, index) => {
        const stepHtml = renderLifecycleStepCard(step, step.step_key_str === currentStepKey);
        return index === executionStepList.length - 1 ? stepHtml : stepHtml + renderLifecycleArrow();
      }).join('');
      return `
        <div class="lifecycle-progress">
          <div class="lifecycle-current-summary">
            <span>${esc(currentSummary)}</span>
            <span>${esc(currentStep ? operatorLifecycleSentence(currentStep, true) : 'Idle / no blocking lifecycle step.')}</span>
          </div>
          <div class="lifecycle-rail-scroll">
            <div class="lifecycle-rail">${railHtml}</div>
          </div>
          ${renderReferenceCheckStep(diffStep)}
        </div>`;
    }

    function renderDiffCell(row) {
      const status = row.latest_diff_status_str || 'not_run';
      const cls = status === 'red' ? 'red' : status === 'yellow' ? 'yellow' : status === 'green' ? 'green' : 'gray';
      const issueText = row.latest_diff_open_issue_count_int === null || row.latest_diff_open_issue_count_int === undefined
        ? 'issues -'
        : 'issues ' + row.latest_diff_open_issue_count_int;
      return `${pill(status, cls)}<div class="cell-sub">${esc(issueText)}</div>`;
    }

    function renderStateCell(row) {
      return `
        <div class="state-main">${pill(row.health_str, healthClass(row.health_str))}</div>
        <div class="state-sub">${esc(row.next_action_str || '-')}</div>`;
    }

    function renderEodCell(eod) {
      const status = eod.status_str || 'not_applicable';
      const severity = healthClass(eod.severity_str);
      const marketDate = eod.expected_market_date_str || eod.latest_market_date_str || '-';
      return `${pill(status, severity)}<div class="cell-sub">${esc(marketDate)}</div>`;
    }

    function renderUpdatedCell(row) {
      return `
        <div>${esc(row.latest_pod_state_timestamp_str || '-')}</div>
        <div class="cell-sub">event ${esc(row.latest_event_timestamp_str || '-')}</div>`;
    }

    async function loadDetail(podId) {
      const requestedPodId = podId;
      state.selectedPod = podId;
      renderPods();
      detailPanel.className = 'detail-workspace';
      detailPanel.innerHTML = '<h2>' + esc(podId) + '</h2><div class="muted">Loading...</div>';
      try {
        const payload = await fetchJson('/api/pods/' + encodeURIComponent(podId));
        if (state.selectedPod !== requestedPodId) return;
        state.selectedDetail = payload;
        renderDetail(payload);
      } catch (error) {
        if (state.selectedPod !== requestedPodId) return;
        renderDetailError(podId, error.message);
      }
    }

    function renderDetail(payload) {
      const row = payload.pod_row_dict || {};
      const decision = payload.latest_decision_plan_dict || null;
      const vplan = payload.latest_vplan_dict || {};
      const report = payload.latest_execution_report_dict || {};
      const diff = payload.latest_diff_dict || {};
      const action = payload.required_action_dict || row.required_action_dict || {};
      const lifecycle = payload.lifecycle_step_dict_list || row.lifecycle_step_dict_list || [];
      const freshness = payload.data_freshness_dict || row.data_freshness_dict || {};
      const eod = payload.eod_snapshot_dict || row.eod_snapshot_dict || {};
      const rehearsal = payload.rehearsal_status_dict || row.rehearsal_status_dict || {};
      const pnl = payload.pod_pnl_dict || {};
      const debugStory = payload.debug_story_dict || {};
      const tabName = activeDetailTab(row.pod_id_str);
      const artifactButton = diff.html_url_str ? `<a class="button-link" href="${esc(diff.html_url_str)}" target="_blank"><button class="secondary">Open artifact</button></a>` : '';
      detailPanel.className = 'dashboard-section detail-workspace';
      detailPanel.innerHTML = `
        <div class="dashboard-section-header">
          <div>
            <h2>Selected POD / What We Learned</h2>
            <div class="dashboard-section-subtitle">Root cause, blocker chain, evidence, broker/model comparison, and safe commands.</div>
          </div>
          <span class="section-status">${esc(row.mode_str || '-')} / read-only</span>
        </div>
        <div class="dashboard-section-body detail-shell-body">
          <div class="detail-header">
            <div class="detail-title">
              <h2>${esc(row.pod_id_str)}</h2>
              <div class="detail-subtitle">
                ${pill(row.health_str, healthClass(row.health_str))}
                <span>${esc(row.mode_str)}</span>
                <span>${esc(row.account_route_str)}</span>
                <span class="muted">${esc(row.strategy_import_str)}</span>
              </div>
            </div>
            <div class="inspect-actions">
              <button class="secondary" id="refresh-detail-button">Refresh POD</button>
              <button class="secondary" id="open-logger-button">Open Logger</button>
              <button class="secondary" id="clear-selection-button">Clear selection</button>
              ${artifactButton}
            </div>
          </div>
          ${renderActionBanner(action)}
          ${renderLifecycleSection(lifecycle, action)}
          ${renderDetailTabs(row.pod_id_str, tabName)}
          ${renderDetailTabBody(tabName, { row, decision, vplan, report, diff, freshness, eod, rehearsal, pnl, debugStory })}
        </div>`;
      document.getElementById('refresh-detail-button').addEventListener('click', () => loadDetail(row.pod_id_str));
      document.getElementById('open-logger-button').addEventListener('click', () => openLogger(row.pod_id_str));
      document.getElementById('clear-selection-button').addEventListener('click', clearSelectedPod);
      detailPanel.querySelectorAll('[data-detail-tab]').forEach(button => {
        button.addEventListener('click', () => setActiveDetailTab(row.pod_id_str, button.dataset.detailTab));
      });
      detailPanel.querySelectorAll('[data-copy-command]').forEach(button => {
        button.addEventListener('click', () => copyCommand(row, button.dataset.copyCommand, button));
      });
      const runDiffButton = document.getElementById('run-diff-button');
      if (runDiffButton) {
        runDiffButton.addEventListener('click', () => runDiff(row.pod_id_str));
      }
    }

    function renderActionBanner(action) {
      const severity = healthClass(action.severity_str);
      const commandName = action.inspect_command_name_str;
      const commandHtml = commandName ? `<span class="muted">Suggested: ${esc(commandName)}</span>` : '';
      const contextHtml = renderActionContextItems(action.context_item_dict_list || []);
      return `
        <div class="action-banner ${severity}">
          <div class="action-main">
            <div class="action-title">${esc(action.label_str || 'Unknown action')}</div>
            <div>${esc(action.reason_str)}</div>
            ${contextHtml}
          </div>
          <div class="action-command">${commandHtml}</div>
        </div>`;
    }

    function renderActionContextItems(contextItems) {
      if (!contextItems.length) return '';
      return `
        <div class="action-context-grid">
          ${contextItems.map(item => {
            const severity = healthClass(item.severity_str);
            const detailHtml = item.detail_str ? `<div class="action-context-detail">${esc(item.detail_str)}</div>` : '';
            return `
              <div class="action-context-item ${severity}">
                <div class="action-context-label">${esc(item.label_str)}</div>
                <div class="action-context-value">${esc(item.value_str)}</div>
                ${detailHtml}
              </div>`;
          }).join('')}
        </div>`;
    }

    function renderLifecycleSection(stepList, action) {
      return `
        <section class="panel full">
          <h3>Lifecycle Progress</h3>
          <div class="panel-body">${renderLifecycleRail(stepList || [], action || {})}</div>
        </section>`;
    }

    function renderDetailTabs(podId, activeTabName) {
      return `<nav class="detail-tabs" aria-label="POD detail sections">
        ${detailTabNameList.map(tabName => `
          <button class="tab-button ${tabName === activeTabName ? 'active' : ''}" data-detail-tab="${esc(tabName)}">
            ${esc(detailTabLabelDict[tabName])}
          </button>`).join('')}
      </nav>`;
    }

    function renderDetailTabBody(tabName, context) {
      if (tabName === 'debug') {
        return `<div class="tab-body">${renderDebugSection(context.debugStory, context.report, context.row, context.rehearsal)}</div>`;
      }
      if (tabName === 'decision') {
        return `<div class="tab-body">${renderDecisionSection(context.decision)}</div>`;
      }
      if (tabName === 'pnl') {
        return `<div class="tab-body">${renderPnlSection(context.pnl)}</div>`;
      }
      if (tabName === 'vplan') {
        return `<div class="tab-body">${renderVPlanSection(context.vplan)}</div>`;
      }
      if (tabName === 'execution') {
        return `<div class="tab-body">${renderExecutionSection(context.vplan, context.report, context.row)}</div>`;
      }
      if (tabName === 'broker') {
        return `<div class="tab-body">${renderBrokerSection(context.vplan, context.report, context.row)}</div>`;
      }
      if (tabName === 'diff') {
        return `<div class="tab-body">${renderDiffSection(context.diff)}</div>`;
      }
      if (tabName === 'freshness') {
        return `<div class="tab-body">
          ${renderDataFreshnessSection(context.freshness)}
          ${renderEodSnapshotSection(context.eod)}
        </div>`;
      }
      return `<div class="tab-body">
        ${renderRehearsalSection(context.rehearsal)}
        ${renderOverviewSection(context.row)}
        ${renderCommandSection(context.row)}
      </div>`;
    }

    function renderDebugSection(story, report, row, rehearsal) {
      const verdict = story.verdict_dict || row.debug_summary_dict || {};
      const severity = healthClass(verdict.severity_str);
      const rehearsalHtml = renderRehearsalSection(rehearsal);
      const blockerRows = (story.blocker_dict_list || []).map(blocker => [
        blocker.severity_str,
        blocker.label_str,
        blocker.reason_str,
        blocker.evidence_str,
        blocker.inspect_command_name_str || '-'
      ]);
      const timelineRows = (story.timeline_event_dict_list || []).map(event => [
        event.timestamp_str || '-',
        event.source_str,
        event.severity_str,
        event.label_str,
        event.status_str || '-',
        event.detail_str || '-'
      ]);
      const evidenceRows = (story.evidence_item_dict_list || []).map(item => [
        item.label_str,
        item.severity_str,
        item.value_str,
        item.detail_str
      ]);
      const executionRows = (report.execution_row_dict_list || []).map(execution => [
        execution.asset_str,
        fmt(execution.planned_order_delta_share_float),
        fmt(execution.filled_share_float),
        fmt(execution.target_share_float),
        fmt(execution.broker_share_float),
        fmt(execution.residual_share_float),
        execution.latest_broker_order_status_str || '-'
      ]);
      return `
        <section class="panel full">
          <h3>Root Cause</h3>
          <div class="panel-body">
            <div class="debug-root ${severity}">
              <div class="debug-root-title">${pill(verdict.severity_str || severity, severity)} ${esc(verdict.verdict_label_str || 'Unknown')}</div>
              <div class="debug-root-sentence">${esc(operatorStateSentence(row))}</div>
              <div>${esc(operatorReasonSentence(row))}</div>
              <div class="muted">${esc(operatorEvidenceSentence(row))}</div>
              <div class="debug-guide-grid">
                <div class="debug-guide-item">
                  <div class="debug-guide-label">Next inspect</div>
                  <div class="debug-guide-value">${esc(safeInspectCommandName(verdict.next_inspect_command_name_str))}</div>
                </div>
                <div class="debug-guide-item">
                  <div class="debug-guide-label">Latest evidence</div>
                  <div class="debug-guide-value">${esc(formatTimestampText(verdict.latest_evidence_timestamp_str || '-'))}</div>
                </div>
                <div class="debug-guide-item">
                  <div class="debug-guide-label">Changed</div>
                  <div class="debug-guide-value">${hasPodChangedSinceLastRefresh(row) ? 'Changed since last refresh' : 'No change since last refresh'}</div>
                </div>
              </div>
            </div>
          </div>
        </section>
        ${rehearsalHtml}
        <section class="panel full">
          <h3>Blocker Chain</h3>
          <div class="panel-body">${renderTable(['Severity', 'Blocker', 'Reason', 'Evidence', 'Inspect'], blockerRows, 'No blockers were found.')}</div>
        </section>
        <section class="panel full">
          <h3>Model vs Broker Snapshot</h3>
          <div class="panel-body">
            <div class="metric-grid">
              ${metric('Residual rows', report.residual_count_int || 0)}
              ${metric('Fills', report.fill_count_int || 0)}
              ${metric('Broker orders', report.broker_order_count_int || 0)}
              ${metric('Broker ACK rows', report.broker_ack_count_int || 0)}
            </div>
            ${renderTable(['Asset', 'Planned', 'Filled', 'Target', 'Broker', 'Residual', 'Order status'], executionRows, 'No model-vs-broker rows yet.')}
          </div>
        </section>
        <section class="panel full">
          <h3>Evidence Timeline</h3>
          <div class="panel-body">
            <details class="audit-detail-drawer">
              <summary>Show audit evidence timeline</summary>
              <div class="audit-detail-body">
                ${renderTable(['Time', 'Source', 'Severity', 'Label', 'Status', 'Detail'], timelineRows, 'No evidence timeline was found.')}
                ${renderTable(['Evidence', 'Severity', 'Value', 'Detail'], evidenceRows, 'No debug evidence items were found.')}
              </div>
            </details>
          </div>
        </section>
        ${renderSuggestedCommandSection(story, row)}`;
    }

    function renderSuggestedCommandSection(story, row) {
      const commandRows = (story.recommended_command_dict_list || []).map(command => [
        command.command_name_str,
        command.reason_str,
        `<button class="secondary" data-copy-command="${esc(command.command_name_str)}">Copy</button>`
      ]);
      if (!commandRows.length) {
        commandRows.push(['status', 'Inspect current POD state and blocker reason.', '<button class="secondary" data-copy-command="status">Copy</button>']);
      }
      const rowHtml = commandRows.map(rowCells => `
        <tr>
          <td>${esc(rowCells[0])}</td>
          <td>${esc(rowCells[1])}</td>
          <td>${rowCells[2]}</td>
        </tr>`).join('');
      return `
        <section class="panel full">
          <h3>Suggested Copy Commands</h3>
          <div class="panel-body">
            <table class="mini-table">
              <thead><tr><th>Command</th><th>Why</th><th>Copy</th></tr></thead>
              <tbody>${rowHtml}</tbody>
            </table>
          </div>
        </section>`;
    }

    function renderCommandSection(row) {
      const commandNameList = ['status', 'next_due', 'show_decision_plan', 'show_vplan', 'compare_reference'];
      const rows = commandNameList.map(commandName => `
        <div class="command-row">
          <div class="command-name">${esc(commandName)}</div>
          <div class="command-text mono">${esc(buildCommand(row, commandName))}</div>
          <button class="secondary" data-copy-command="${esc(commandName)}">Copy</button>
        </div>`).join('');
      return `
        <section class="panel full">
          <h3>Commands</h3>
          <div class="panel-body">
            <details class="command-drawer">
              <summary>Commands</summary>
              <div class="command-drawer-body">
                <div class="command-list">${rows}</div>
              </div>
            </details>
          </div>
        </section>`;
    }

    function renderRehearsalSection(rehearsal) {
      if (!rehearsal || rehearsal.status_str !== 'active') return '';
      return `
        <section class="panel full">
          <h3>Unified Rehearsal</h3>
          <div class="panel-body">
            <div class="metric-grid">
              ${metric('Promotion gate', rehearsal.promotion_gate_status_str)}
              ${metric('Last cycle', rehearsal.last_cycle_status_str)}
              ${metric('Completed cycles', rehearsal.completed_cycle_count_int || 0)}
              ${metric('SIM ledger', rehearsal.sim_ledger_status_str)}
              ${metric('IBKR reference', rehearsal.ibkr_reference_status_str)}
              ${metric('IBKR open', rehearsal.ibkr_open_price_status_str)}
            </div>
            ${keyValueGrid([
              ['Official accounting source', rehearsal.official_accounting_source_str],
              ['Dashboard state source', 'Dashboard reads per-POD incubation state.'],
              ['SIM ledger source', rehearsal.sim_ledger_source_str],
              ['IBKR reference source', rehearsal.ibkr_reference_source_str],
              ['IBKR open source', rehearsal.ibkr_open_price_source_str],
              ['Paper probe status', rehearsal.paper_probe_status_str],
              ['Paper probe evidence', 'Paper probe is evidence only; it does not count as SIM ledger P&L.'],
              ['Paper fills count as SIM P&L', fmtBool(rehearsal.paper_probe_accounting_truth_bool)],
              ['Detail', rehearsal.detail_str]
            ])}
          </div>
        </section>`;
    }

    function renderOverviewSection(row) {
      return `
        <section class="panel">
          <h3>Overview</h3>
          <div class="panel-body">
            <div class="metric-grid">
              ${metric('Next', row.next_action_str)}
              ${metric('Reason', row.reason_code_str)}
              ${metric('Decision', row.latest_decision_plan_status_str)}
              ${metric('VPlan', row.latest_vplan_status_str ? row.latest_vplan_status_str + (row.latest_vplan_id_int ? ' #' + row.latest_vplan_id_int : '') : '-')}
              ${metric('Equity', row.equity_float)}
              ${metric('Cash', row.cash_float)}
              ${metric('Positions', row.position_count_int)}
              ${metric('Warnings', row.warning_count_int)}
            </div>
            ${keyValueGrid([
              ['DB status', row.db_status_str],
              ['DB override', fmtBool(row.db_override_bool)],
              ['DB path', row.db_path_str],
              ['Pod state updated', row.latest_pod_state_timestamp_str],
              ['Broker snapshot', row.latest_broker_snapshot_timestamp_str],
              ['Reconcile', row.latest_reconciliation_status_str]
            ])}
          </div>
        </section>`;
    }

    function renderPnlSection(pnl) {
      const pointList = pnl.equity_point_dict_list || [];
      if (!pointList.length) {
        return panel('PnL', '<div class="empty-note">No EOD equity snapshots were found for this POD.</div>', true);
      }
      return `
        <section class="panel full">
          <h3>PnL</h3>
          <div class="panel-body">
            <div class="metric-grid">
              ${metric('Latest EOD date', pnl.latest_market_date_str)}
              ${metric('Latest equity', pnl.latest_equity_float)}
              ${metric('Previous EOD equity', pnl.previous_equity_float)}
              ${metricSigned('Daily PnL $', pnl.daily_pnl_float, fmtUnavailable)}
              ${metricSigned('Daily PnL %', pnl.daily_pnl_pct_float, fmtPct)}
              ${metricSigned('Since-start PnL $', pnl.since_start_pnl_float, fmtUnavailable)}
              ${metricSigned('Since-start PnL %', pnl.since_start_pnl_pct_float, fmtPct)}
              ${metric('EOD points', pnl.point_count_int || 0)}
            </div>
            ${renderPnlCurve(pointList)}
            ${renderPnlTable(pointList)}
          </div>
        </section>`;
    }

    function renderPnlCurve(pointList) {
      const equityPointList = (pointList || [])
        .map((point, index) => ({
          index,
          marketDate: point.market_date_str,
          equity: Number(point.equity_float)
        }))
        .filter(point => Number.isFinite(point.equity));
      if (!equityPointList.length) {
        return '<div class="empty-note">No numeric equity points were found.</div>';
      }
      const width = 1000;
      const height = 260;
      const padLeft = 24;
      const padRight = 24;
      const padTop = 18;
      const padBottom = 28;
      const minEquity = Math.min(...equityPointList.map(point => point.equity));
      const maxEquity = Math.max(...equityPointList.map(point => point.equity));
      const rawEquityRange = maxEquity - minEquity;
      const equityPadding = rawEquityRange === 0 ? Math.max(Math.abs(maxEquity) * 0.01, 1) : rawEquityRange * 0.12;
      const yMin = minEquity - equityPadding;
      const yMax = maxEquity + equityPadding;
      const equityRange = yMax === yMin ? 1 : yMax - yMin;
      const xStep = equityPointList.length <= 1 ? 0 : (width - padLeft - padRight) / (equityPointList.length - 1);
      const chartPointList = equityPointList.map((point, index) => {
        const x = equityPointList.length <= 1 ? width / 2 : padLeft + index * xStep;
        const y = padTop + (1 - ((point.equity - yMin) / equityRange)) * (height - padTop - padBottom);
        const xPct = (x / width) * 100;
        const yPct = (y / height) * 100;
        return { ...point, x, y, xPct, yPct };
      });
      const pathStr = chartPointList.map((point, index) => `${index === 0 ? 'M' : 'L'} ${point.x.toFixed(2)} ${point.y.toFixed(2)}`).join(' ');
      const firstPoint = chartPointList[0];
      const latestPoint = chartPointList[chartPointList.length - 1];
      const curveSignedClass = signedValueClass(latestPoint.equity - firstPoint.equity);
      const pointHtml = chartPointList.map(point => {
        const labelStr = `${point.marketDate} ${fmt(point.equity)}`;
        return `<span class="pnl-point ${curveSignedClass}" style="left: ${point.xPct.toFixed(3)}%; top: ${point.yPct.toFixed(3)}%;" title="${esc(labelStr)}" aria-label="${esc(labelStr)}"></span>`;
      }).join('');
      const midY = padTop + (height - padTop - padBottom) / 2;
      const singlePointNoteHtml = equityPointList.length === 1
        ? '<div class="empty-note">Only one EOD equity point is available, so the chart shows a single mark.</div>'
        : '';
      return `
        <div class="pnl-curve">
          <div class="pnl-chart-header">
            <div>
              <div class="pnl-chart-title">EOD equity curve</div>
              <div class="pnl-chart-subtitle">${esc(firstPoint.marketDate)} to ${esc(latestPoint.marketDate)} / ${esc(equityPointList.length)} points</div>
            </div>
            <div class="pnl-chart-range">
              <span>Axis range</span>
              <strong>${esc(fmt(yMin))} - ${esc(fmt(yMax))}</strong>
            </div>
          </div>
          <div class="pnl-curve-plot">
            <div class="pnl-y-axis" aria-hidden="true">
              <span>${esc(fmt(yMax))}</span>
              <span>${esc(fmt((yMax + yMin) / 2))}</span>
              <span>${esc(fmt(yMin))}</span>
            </div>
            <div class="pnl-chart-area">
              <svg viewBox="0 0 ${width} ${height}" preserveAspectRatio="xMidYMid meet" role="img" aria-label="POD EOD equity curve">
                <line class="pnl-grid-line" x1="${padLeft}" y1="${padTop}" x2="${width - padRight}" y2="${padTop}" />
                <line class="pnl-grid-line" x1="${padLeft}" y1="${midY}" x2="${width - padRight}" y2="${midY}" />
                <line class="pnl-axis-line" x1="${padLeft}" y1="${height - padBottom}" x2="${width - padRight}" y2="${height - padBottom}" />
                <path class="pnl-line ${curveSignedClass}" d="${pathStr}" />
              </svg>
              <div class="pnl-point-layer">${pointHtml}</div>
            </div>
          </div>
          <div class="pnl-x-axis" aria-hidden="true">
            <span>${esc(firstPoint.marketDate)}</span>
            <span>${esc(latestPoint.marketDate)}</span>
          </div>
          ${singlePointNoteHtml}
        </div>`;
    }

    function renderPnlTable(pointList) {
      const rowList = (pointList || []).slice().reverse();
      if (!rowList.length) {
        return '<div class="empty-note">No EOD equity points yet.</div>';
      }
      return `
        <table class="mini-table pnl-table">
          <thead>
            <tr>
              <th>Market date</th>
              <th>Equity</th>
              <th>Cash</th>
              <th>Daily PnL $</th>
              <th>Daily PnL %</th>
              <th>Source</th>
              <th>Updated</th>
            </tr>
          </thead>
          <tbody>
            ${rowList.map(point => `
              <tr>
                <td>${esc(point.market_date_str)}</td>
                <td class="num">${esc(fmt(point.equity_float))}</td>
                <td class="num">${esc(fmt(point.cash_float))}</td>
                <td class="num">${signedValueHtml(point.daily_pnl_float, fmtUnavailable)}</td>
                <td class="num">${signedValueHtml(point.daily_pnl_pct_float, fmtPct)}</td>
                <td>${esc(point.snapshot_source_str || '-')}</td>
                <td>${esc(point.updated_timestamp_str || '-')}</td>
              </tr>`).join('')}
          </tbody>
        </table>`;
    }

    function renderDataFreshnessSection(freshness) {
      const rows = (freshness.item_dict_list || []).map(item => [
        item.label_str,
        item.severity_str,
        item.value_str,
        item.detail_str
      ]);
      return `
        <section class="panel">
          <h3>Data Freshness</h3>
          <div class="panel-body">
            ${renderTable(['Source', 'State', 'Latest', 'Detail'], rows, 'No freshness data was found.')}
            ${keyValueGrid([
              ['DTB3 status', freshness.dtb3_download_status_str],
              ['DTB3 observation', freshness.dtb3_latest_observation_date_str],
              ['DTB3 freshness days', freshness.dtb3_freshness_business_days_int],
              ['DTB3 source', freshness.dtb3_source_name_str],
              ['Used cache', freshness.dtb3_used_cache_bool]
            ])}
          </div>
        </section>`;
    }

    function renderEodSnapshotSection(eod) {
      if (!eod || !eod.status_str) {
        return panel('EOD Snapshot', '<div class="empty-note">No EOD snapshot data was found.</div>');
      }
      return `
        <section class="panel">
          <h3>EOD Snapshot</h3>
          <div class="panel-body">
            <div class="metric-grid">
              ${metric('Status', eod.status_str)}
              ${metric('Expected market date', eod.expected_market_date_str)}
              ${metric('Expected due', eod.expected_due_timestamp_str)}
              ${metric('Latest market date', eod.latest_market_date_str)}
              ${metric('Latest timestamp', eod.latest_timestamp_str)}
              ${metric('Recorded', eod.recorded_timestamp_str)}
              ${metric('Source', eod.source_str)}
              ${metric('Positions', eod.position_count_int)}
            </div>
            ${keyValueGrid([
              ['Equity', eod.equity_float],
              ['Cash', eod.cash_float],
              ['Same session', fmtBool(eod.same_session_bool)],
              ['Blocked by execution', fmtBool(eod.unresolved_execution_bool)],
              ['Detail', eod.detail_str]
            ])}
          </div>
        </section>`;
    }

    function renderDecisionSection(decision) {
      if (!decision) {
        return panel('Decision', '<div class="empty-note">No DecisionPlan was found for this POD.</div>');
      }
      const targetRows = Object.entries(decision.display_target_weight_map_dict || {})
        .sort(([left], [right]) => left.localeCompare(right))
        .map(([asset, weight]) => [asset, fmtPct(weight)]);
      const exitRows = (decision.exit_asset_list || []).map(asset => [asset]);
      const baseRows = Object.entries(decision.decision_base_position_map_dict || {})
        .sort(([left], [right]) => left.localeCompare(right))
        .map(([asset, shares]) => [asset, fmt(shares)]);
      const metadata = decision.snapshot_metadata_dict || {};
      return `
        <section class="panel">
          <h3>Decision</h3>
          <div class="panel-body">
            <div class="metric-grid">
              ${metric('Status', decision.status_str)}
              ${metric('Book', decision.decision_book_type_str)}
              ${metric('Signal', decision.signal_timestamp_str)}
              ${metric('Submit', decision.submission_timestamp_str)}
              ${metric('Execute', decision.target_execution_timestamp_str)}
              ${metric('Policy', decision.execution_policy_str)}
              ${metric('Linked VPlan', decision.latest_vplan_id_int ? '#' + decision.latest_vplan_id_int + ' ' + (decision.latest_vplan_status_str || '') : 'none')}
              ${metric('Cash reserve', fmtPct(decision.cash_reserve_weight_float || 0))}
            </div>
            ${renderTable(['Asset', 'Target weight'], targetRows, 'No target weights.')}
            ${renderTable(['Exit asset'], exitRows, 'No exit assets.')}
            ${renderTable(['Base asset', 'Base shares'], baseRows, 'No decision-base positions.')}
            ${keyValueGrid([
              ['Strategy family', metadata.strategy_family_str],
              ['DTB3 status', metadata.dtb3_download_status_str],
              ['DTB3 observation', metadata.dtb3_latest_observation_date_str],
              ['DTB3 freshness days', metadata.dtb3_freshness_business_days_int],
              ['DTB3 source', metadata.dtb3_source_name_str],
              ['Used cache', metadata.dtb3_used_cache_bool]
            ])}
          </div>
        </section>`;
    }

    function renderVPlanSection(vplan) {
      if (!vplan || !vplan.vplan_id_int) {
        return panel('VPlan', '<div class="empty-note">No VPlan was found yet.</div>');
      }
      const rows = (vplan.vplan_row_dict_list || []).map(row => [
        row.asset_str,
        fmt(row.current_share_float),
        fmt(row.target_share_float),
        fmt(row.order_delta_share_float),
        fmt(row.live_reference_price_float),
        row.live_reference_source_str,
        fmt(row.estimated_target_notional_float)
      ]);
      return `
        <section class="panel full">
          <h3>VPlan</h3>
          <div class="panel-body">
            <div class="metric-grid">
              ${metric('VPlan id', '#' + vplan.vplan_id_int)}
              ${metric('Status', vplan.status_str)}
              ${metric('Submit ACK', vplan.submit_ack_status_str)}
              ${metric('ACK coverage', vplan.ack_coverage_ratio_float)}
              ${metric('NetLiq', vplan.net_liq_float)}
              ${metric('Pod budget', vplan.pod_budget_float)}
              ${metric('Price source', vplan.live_price_source_str)}
              ${metric('Missing ACK', vplan.missing_ack_count_int)}
            </div>
            ${renderTable(['Asset', 'Current', 'Target', 'Delta', 'Ref price', 'Ref source', 'Est notional'], rows, 'No VPlan rows.')}
          </div>
        </section>`;
    }

    function renderExecutionSection(vplan, report, row) {
      if (!vplan || !vplan.vplan_id_int) {
        return panel('Execution', '<div class="empty-note">No VPlan execution evidence was found yet.</div>', true);
      }
      const executionRows = (report.execution_row_dict_list || []).map(execution => [
        execution.asset_str,
        execution.side_str,
        fmt(execution.planned_order_delta_share_float),
        fmt(execution.filled_share_float),
        fmt(execution.target_share_float),
        fmt(execution.broker_share_float),
        fmt(execution.residual_share_float),
        fmtUnavailable(execution.fill_price_float),
        fmtUnavailable(execution.official_open_price_float),
        fmtUnavailable(execution.vplan_reference_price_float),
        fmtUnavailable(execution.official_open_slippage_bps_float) + ' / ' + fmtUnavailable(execution.official_open_slippage_notional_float),
        fmtUnavailable(execution.vplan_reference_slippage_bps_float) + ' / ' + fmtUnavailable(execution.vplan_reference_slippage_notional_float),
        execution.latest_broker_order_status_str || '-'
      ]);
      return `
        <section class="panel full">
          <h3>Execution</h3>
          <div class="panel-body">
            <div class="metric-grid">
              ${metric('VPlan status', vplan.status_str)}
              ${metric('Reconcile', row.latest_reconciliation_status_str)}
              ${metric('Residual rows', report.residual_count_int || 0)}
              ${metric('Fills', report.fill_count_int || 0)}
              ${metric('Official open coverage', (report.fill_with_official_open_count_int || 0) + ' / ' + (report.fill_count_int || 0))}
              ${metric('Open slippage bps', fmtUnavailable(report.official_open_slippage_bps_float))}
              ${metric('Open slippage $', fmtUnavailable(report.official_open_slippage_notional_float))}
              ${metric('Reference slippage bps', fmtUnavailable(report.vplan_reference_slippage_bps_float))}
              ${metric('Reference slippage $', fmtUnavailable(report.vplan_reference_slippage_notional_float))}
            </div>
            ${renderTable(
              ['Asset', 'Side', 'Planned', 'Filled', 'Target', 'Broker', 'Residual', 'Fill price', 'Official open', 'VPlan ref', 'Open slippage bps / $', 'Reference slippage bps / $', 'Order status'],
              executionRows,
              'No execution rows yet.'
            )}
          </div>
        </section>`;
    }

    function renderBrokerSection(vplan, report, row) {
      const orderRows = (vplan.broker_order_row_dict_list || []).map(order => [
        order.asset_str,
        order.broker_order_type_str,
        fmt(order.amount_float),
        fmt(order.filled_amount_float),
        fmt(order.remaining_amount_float),
        order.status_str,
        order.last_status_timestamp_str || order.submitted_timestamp_str
      ]);
      const fillRows = (vplan.fill_row_dict_list || []).map(fill => [
        fill.asset_str,
        fmt(fill.fill_amount_float),
        fmt(fill.fill_price_float),
        fmt(fill.official_open_price_float),
        fill.open_price_source_str,
        fill.fill_timestamp_str
      ]);
      const ackRows = (vplan.broker_ack_row_dict_list || []).map(ack => [
        ack.asset_str,
        ack.ack_status_str,
        ack.ack_source_str,
        ack.broker_order_id_str || '-'
      ]);
      return `
        <section class="panel full">
          <h3>Broker</h3>
          <div class="panel-body">
            <div class="metric-grid">
              ${metric('Orders', report.broker_order_count_int || 0)}
              ${metric('Fills', report.fill_count_int || 0)}
              ${metric('ACK rows', report.broker_ack_count_int || 0)}
              ${metric('Reconcile', row.latest_reconciliation_status_str)}
            </div>
            ${renderTable(['Asset', 'Type', 'Requested', 'Filled', 'Remaining', 'Status', 'Time'], orderRows, 'No broker orders yet.')}
            ${renderTable(['Asset', 'Shares', 'Fill price', 'Official open', 'Open source', 'Time'], fillRows, 'No fills yet.')}
            ${renderTable(['Asset', 'ACK status', 'Source', 'Broker order'], ackRows, 'No ACK rows yet.')}
          </div>
        </section>`;
    }

    function renderDiffSection(diff) {
      const status = diff.status_str || 'not_run';
      const links = [
        ['Full artifact', diff.html_url_str],
        ['Summary JSON', diff.summary_url_str],
        ['Equity PNG', diff.equity_png_url_str],
        ['Tracking PNG', diff.tracking_png_url_str]
      ].filter(([, href]) => !!href);
      return `
        <section class="panel">
          <h3>DIFF</h3>
          <div class="panel-body">
            <div class="metric-grid">
              ${metric('Status', status)}
              ${metric('Artifact time', diff.artifact_timestamp_str)}
              ${metric('Equity tracking error', diff.equity_tracking_error_float)}
              ${metric('Open issues', diff.open_issue_count_int)}
            </div>
            <div class="link-row">
              ${links.length ? links.map(([label, href]) => `<a href="${esc(href)}" target="_blank">${esc(label)}</a>`).join('') : '<span class="muted">No artifact links yet.</span>'}
            </div>
            <div class="link-row">
              <button class="primary" id="run-diff-button">Run DIFF</button>
            </div>
            <div class="job-status" id="diff-job-output">No DIFF job running.</div>
          </div>
        </section>`;
    }

    function renderEventsSection(events) {
      const rows = events.map(event => [
        event.timestamp_str || event.event_timestamp_str || event.as_of_timestamp_str || event.created_timestamp_str,
        event.severity_str || event.level_str || '-',
        event.event_name_str || event.command_name_str || '-',
        event.reason_code_str || event.message_str || event.error_str || event.status_str || '-'
      ]);
      return panel(
        'Events',
        renderTable(['Time', 'Severity', 'Event', 'Reason / message'], rows, 'No recent events for this POD.'),
        true
      );
    }

    async function openLogger(podId) {
      state.loggerPodId = podId;
      state.loggerFilter = 'all';
      state.loggerEvents = [];
      loggerBackdrop.hidden = false;
      loggerBackdrop.classList.add('open');
      await refreshLoggerEvents();
    }

    function closeLogger() {
      loggerBackdrop.classList.remove('open');
      loggerBackdrop.hidden = true;
    }

    async function refreshLoggerEvents() {
      if (!state.loggerPodId) return;
      state.loggerLoading = true;
      renderLoggerDrawer();
      try {
        const payload = await fetchJson('/api/pods/' + encodeURIComponent(state.loggerPodId) + '/events');
        state.loggerEvents = payload.event_dict_list || [];
      } catch (error) {
        state.loggerEvents = [{
          timestamp_str: new Date().toISOString(),
          severity_str: 'ERROR',
          event_name_str: 'logger.fetch_failed',
          message_str: error.message
        }];
      } finally {
        state.loggerLoading = false;
        renderLoggerDrawer();
      }
    }

    function renderLoggerDrawer() {
      const podId = state.loggerPodId || '-';
      loggerTitle.textContent = 'Logger';
      loggerSubtitle.textContent = podId;
      document.querySelectorAll('[data-logger-filter]').forEach(button => {
        button.classList.toggle('active', button.dataset.loggerFilter === state.loggerFilter);
      });
      if (state.loggerLoading) {
        loggerBody.innerHTML = '<div class="empty-note">Loading recent POD events...</div>';
        return;
      }
      const filteredEvents = filterLoggerEvents(state.loggerEvents || [], state.loggerFilter);
      if (!filteredEvents.length) {
        loggerBody.innerHTML = '<div class="empty-note">No matching recent events for this POD.</div>';
        return;
      }
      const rows = filteredEvents.map(event => [
        event.timestamp_str || event.event_timestamp_str || event.as_of_timestamp_str || event.created_timestamp_str,
        event.severity_str || event.level_str || '-',
        event.event_name_str || event.command_name_str || '-',
        event.reason_code_str || event.message_str || event.error_str || event.status_str || '-'
      ]);
      loggerBody.innerHTML = `
        <div class="logger-table-wrap">
          ${renderTable(['Time', 'Level', 'Event', 'Reason / Message'], rows, 'No matching recent events for this POD.').replace('mini-table', 'mini-table logger-table')}
        </div>`;
    }

    function filterLoggerEvents(events, filterName) {
      if (filterName === 'errors') {
        return events.filter(event => loggerEventSeverity(event) === 'error');
      }
      if (filterName === 'warnings') {
        return events.filter(event => ['warning', 'error'].includes(loggerEventSeverity(event)));
      }
      return events;
    }

    function loggerEventSeverity(event) {
      const levelText = String(event.severity_str || event.level_str || event.status_str || '').toLowerCase();
      if (levelText.includes('error') || levelText.includes('fail') || levelText.includes('critical')) {
        return 'error';
      }
      if (levelText.includes('warn') || levelText.includes('blocked')) {
        return 'warning';
      }
      return 'info';
    }

    function panel(title, body, full) {
      return `<section class="panel ${full ? 'full' : ''}"><h3>${esc(title)}</h3><div class="panel-body">${body}</div></section>`;
    }

    function metric(label, value) {
      return `<div class="metric"><div class="metric-label">${esc(label)}</div><div class="metric-value">${esc(value)}</div></div>`;
    }

    function metricSigned(label, value, formatterFn) {
      return `<div class="metric"><div class="metric-label">${esc(label)}</div><div class="metric-value">${signedValueHtml(value, formatterFn)}</div></div>`;
    }

    function signedValueClass(value) {
      const numericValue = Number(value);
      if (!Number.isFinite(numericValue) || numericValue === 0) return 'neutral';
      return numericValue > 0 ? 'positive' : 'negative';
    }

    function signedValueHtml(value, formatterFn) {
      const displayValue = formatterFn ? formatterFn(value) : fmtUnavailable(value);
      return `<span class="signed-value ${signedValueClass(value)}">${esc(displayValue)}</span>`;
    }

    function keyValueGrid(rows) {
      const filtered = rows.filter(([, value]) => value !== null && value !== undefined && value !== '');
      if (!filtered.length) return '';
      return `<div class="kv-grid">${filtered.map(([key, value]) => `<div class="kv-key">${esc(key)}</div><div>${esc(value)}</div>`).join('')}</div>`;
    }

    function renderTable(headers, rows, emptyText) {
      if (!rows || !rows.length) {
        return `<div class="empty-note">${esc(emptyText)}</div>`;
      }
      return `
        <table class="mini-table">
          <thead><tr>${headers.map(header => `<th>${esc(header)}</th>`).join('')}</tr></thead>
          <tbody>
            ${rows.map(row => `<tr>${row.map(cell => `<td>${esc(cell)}</td>`).join('')}</tr>`).join('')}
          </tbody>
        </table>`;
    }

    function quoteArg(value) {
      const text = String(value || '');
      if (!text.includes(' ') && !text.includes('"')) return text;
      return '"' + text.replaceAll('"', '`"') + '"';
    }

    function buildCommand(row, commandName) {
      const cleanCommandName = safeInspectCommandName(commandName);
      const moduleName = cleanCommandName === 'next_due' ? 'alpha.live.scheduler_service' : 'alpha.live.runner';
      const parts = ['uv run python -m ' + moduleName, cleanCommandName, '--mode', row.mode_str, '--pod-id', row.pod_id_str];
      if (row.db_override_bool && row.db_path_str) {
        parts.push('--db-path', quoteArg(row.db_path_str));
      }
      return parts.join(' ');
    }

    async function copyCommand(row, commandName, button) {
      const command = buildCommand(row, commandName);
      try {
        if (navigator.clipboard && navigator.clipboard.writeText) {
          await navigator.clipboard.writeText(command);
        } else {
          const textarea = document.createElement('textarea');
          textarea.value = command;
          document.body.appendChild(textarea);
          textarea.select();
          document.execCommand('copy');
          document.body.removeChild(textarea);
        }
        const oldText = button.textContent;
        button.textContent = 'Copied';
        setTimeout(() => { button.textContent = oldText; }, 1200);
      } catch (error) {
        const oldText = button.textContent;
        button.textContent = 'Copy failed';
        setTimeout(() => { button.textContent = oldText; }, 1500);
      }
    }

    function renderJobStatus(job) {
      if (!job) return 'No DIFF job running.';
      const parts = [
        'status=' + formatValue(job.status_str),
        'created=' + formatValue(job.created_timestamp_str),
        'started=' + formatValue(job.started_timestamp_str),
        'completed=' + formatValue(job.completed_timestamp_str)
      ];
      if (job.error_str) parts.push('error=' + formatValue(job.error_str));
      if (job.result_dict && job.result_dict.status_str) parts.push('result=' + formatValue(job.result_dict.status_str));
      return parts.join(' | ');
    }

    async function runDiff(podId) {
      const output = document.getElementById('diff-job-output');
      output.textContent = 'Starting DIFF...';
      try {
        const job = await fetchJson('/api/pods/' + encodeURIComponent(podId) + '/diff/run', { method: 'POST' });
        pollJob(job.job_id_str, podId);
      } catch (error) {
        output.textContent = 'DIFF failed to start: ' + error.message;
      }
    }

    async function pollJob(jobId, podId) {
      const output = document.getElementById('diff-job-output');
      try {
        const job = await fetchJson('/api/jobs/' + encodeURIComponent(jobId));
        output.textContent = renderJobStatus(job);
        if (job.status_str === 'queued' || job.status_str === 'running') {
          setTimeout(() => pollJob(jobId, podId), 1500);
        } else {
          refreshPods();
          loadDetail(podId);
        }
      } catch (error) {
        output.textContent = 'DIFF job poll failed: ' + error.message;
      }
    }

    refreshPods();
    setInterval(refreshPods, 4000);
  </script>
</body>
</html>
"""


def main(argv_list: list[str] | None = None) -> int:
    parser_obj = argparse.ArgumentParser(description="Local read-only live POD dashboard.")
    subparser_obj = parser_obj.add_subparsers(dest="command_name_str", required=True)
    serve_parser_obj = subparser_obj.add_parser("serve")
    serve_parser_obj.add_argument("--host", dest="host_str", default=DEFAULT_DASHBOARD_HOST_STR)
    serve_parser_obj.add_argument("--port", dest="port_int", type=int, default=DEFAULT_DASHBOARD_PORT_INT)
    serve_parser_obj.add_argument("--releases-root", dest="releases_root_path_str", default=DEFAULT_RELEASES_ROOT_PATH_STR)
    serve_parser_obj.add_argument("--config", dest="config_path_str", default=DEFAULT_CONFIG_PATH_STR)
    serve_parser_obj.add_argument("--results-root", dest="results_root_path_str", default=DEFAULT_RESULTS_ROOT_PATH_STR)
    serve_parser_obj.add_argument("--event-log", dest="event_log_path_str", default=DEFAULT_EVENT_LOG_PATH_STR)
    parsed_args_obj = parser_obj.parse_args(argv_list)
    if parsed_args_obj.command_name_str == "serve":
        serve_dashboard(
            host_str=parsed_args_obj.host_str,
            port_int=parsed_args_obj.port_int,
            releases_root_path_str=parsed_args_obj.releases_root_path_str,
            config_path_str=parsed_args_obj.config_path_str,
            results_root_path_str=parsed_args_obj.results_root_path_str,
            event_log_path_str=parsed_args_obj.event_log_path_str,
        )
        return 0
    raise ValueError(f"Unsupported command_name_str '{parsed_args_obj.command_name_str}'.")


if __name__ == "__main__":
    raise SystemExit(main())
