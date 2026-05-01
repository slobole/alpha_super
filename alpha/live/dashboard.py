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
    if release_obj.mode_str == "incubation":
        return runner.DEFAULT_INCUBATION_DB_PATH_STR
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
        )
        for target_obj in app_obj.get_target_list()
    ]
    return {
        "as_of_timestamp_str": as_of_ts.isoformat(),
        "pod_row_dict_list": pod_row_dict_list,
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
    )
    detail_dict = {
        "pod_row_dict": row_dict,
        "latest_decision_plan_dict": None,
        "latest_vplan_dict": None,
        "latest_execution_report_dict": None,
        "event_dict_list": load_recent_event_dict_list(
            log_path_str=app_obj.event_log_path_str,
            pod_id_str=pod_id_str,
            limit_int=DEFAULT_EVENT_LIMIT_INT,
        ),
        "latest_diff_dict": find_latest_diff_artifact_dict(
            results_root_path_str=app_obj.results_root_path_str,
            mode_str=target_obj.release_obj.mode_str,
            pod_id_str=pod_id_str,
        ),
    }
    db_path_obj = Path(target_obj.db_path_str)
    if not db_path_obj.exists():
        return detail_dict
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
        if not _table_exists_bool(connection_obj, "vplan"):
            return detail_dict
        latest_vplan_row_dict = _fetch_latest_vplan_row_dict(connection_obj, pod_id_str)
        if latest_vplan_row_dict is None:
            return detail_dict
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
        detail_dict["latest_vplan_dict"] = latest_vplan_row_dict
        detail_dict["latest_execution_report_dict"] = _build_execution_report_from_vplan_dict(
            latest_vplan_row_dict
        )
    return detail_dict


def build_pod_row_dict(
    pod_target_obj: DashboardPodTarget,
    as_of_ts: datetime,
    results_root_path_str: str,
) -> dict[str, Any]:
    release_obj = pod_target_obj.release_obj
    db_path_obj = Path(pod_target_obj.db_path_str)
    latest_diff_dict = find_latest_diff_artifact_dict(
        results_root_path_str=results_root_path_str,
        mode_str=release_obj.mode_str,
        pod_id_str=release_obj.pod_id_str,
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
        "latest_vplan_status_str": None,
        "latest_vplan_id_int": None,
        "latest_broker_snapshot_timestamp_str": None,
        "latest_pod_state_timestamp_str": None,
        "cash_float": None,
        "equity_float": None,
        "position_count_int": 0,
        "next_action_str": "no_db",
        "reason_code_str": "db_missing",
        "warning_count_int": 0,
        "missing_ack_count_int": 0,
        "exception_count_int": 0,
        "latest_reconciliation_status_str": None,
        "latest_diff_status_str": str(latest_diff_dict.get("status_str", "not_run")),
        "latest_diff_timestamp_str": latest_diff_dict.get("artifact_timestamp_str"),
        "latest_diff_equity_tracking_error_float": latest_diff_dict.get(
            "equity_tracking_error_float"
        ),
        "latest_diff_open_issue_count_int": latest_diff_dict.get("open_issue_count_int"),
        "latest_diff_artifact_url_str": latest_diff_dict.get("html_url_str"),
        "health_str": "gray",
    }
    if not db_path_obj.exists():
        return base_row_dict

    try:
        with _connect_readonly_existing_db(db_path_obj) as connection_obj:
            if not _table_exists_bool(connection_obj, "decision_plan"):
                base_row_dict["db_status_str"] = "empty"
                base_row_dict["reason_code_str"] = "db_empty"
                return base_row_dict
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
    except sqlite3.DatabaseError as exception_obj:
        base_row_dict["db_status_str"] = "error"
        base_row_dict["reason_code_str"] = "db_error"
        base_row_dict["error_str"] = str(exception_obj)
        base_row_dict["health_str"] = "red"
        return base_row_dict

    if latest_decision_plan_row_dict is not None:
        base_row_dict["latest_decision_plan_status_str"] = latest_decision_plan_row_dict.get(
            "status_str"
        )
    if latest_vplan_row_dict is not None:
        base_row_dict["latest_vplan_status_str"] = latest_vplan_row_dict.get("status_str")
        base_row_dict["latest_vplan_id_int"] = latest_vplan_row_dict.get("vplan_id_int")
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
    return base_row_dict


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


def _build_execution_report_from_vplan_dict(vplan_dict: dict[str, Any]) -> dict[str, Any]:
    fill_row_dict_list = list(vplan_dict.get("fill_row_dict_list", []))
    return {
        "pod_id_str": vplan_dict.get("pod_id_str"),
        "latest_vplan_id_int": vplan_dict.get("vplan_id_int"),
        "fill_count_int": len(fill_row_dict_list),
        "broker_order_count_int": len(vplan_dict.get("broker_order_row_dict_list", [])),
        "broker_ack_count_int": len(vplan_dict.get("broker_ack_row_dict_list", [])),
    }


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
      --bg: #f6f7f9;
      --panel: #ffffff;
      --line: #d8dee8;
      --text: #111827;
      --muted: #667085;
      --green: #14804a;
      --yellow: #b7791f;
      --red: #c2410c;
      --gray: #6b7280;
      --blue: #1d4ed8;
      --blue-soft: #eff6ff;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font: 14px/1.45 Arial, Helvetica, sans-serif;
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
    main { padding: 14px 18px 24px; }
    .toolbar {
      display: flex;
      gap: 10px;
      align-items: center;
      margin-bottom: 12px;
    }
    select, button {
      min-height: 32px;
      border: 1px solid var(--line);
      background: var(--panel);
      color: var(--text);
      padding: 0 10px;
      border-radius: 6px;
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
      min-width: 1040px;
      border-collapse: collapse;
      background: var(--panel);
      border: 0;
    }
    th, td {
      border-bottom: 1px solid var(--line);
      padding: 8px 9px;
      text-align: left;
      vertical-align: top;
      white-space: nowrap;
    }
    th { color: var(--muted); font-size: 12px; font-weight: 700; background: #f9fafb; }
    tr:hover td { background: #fbfdff; }
    tr.selected td { background: var(--blue-soft); }
    .pod-link { color: var(--blue); cursor: pointer; font-weight: 700; }
    .strategy-text {
      display: inline-block;
      max-width: 360px;
      white-space: normal;
      overflow-wrap: anywhere;
    }
    .pill {
      display: inline-block;
      min-width: 54px;
      text-align: center;
      padding: 2px 8px;
      border-radius: 999px;
      color: white;
      font-size: 12px;
      font-weight: 700;
    }
    .green { background: var(--green); }
    .yellow { background: var(--yellow); }
    .red { background: var(--red); }
    .gray { background: var(--gray); }
    .table-panel { min-width: 0; }
    .table-scroll {
      width: 100%;
      overflow-x: auto;
      background: var(--panel);
      border: 1px solid var(--line);
    }
    .detail-workspace {
      background: var(--panel);
      border: 1px solid var(--line);
      margin-top: 14px;
      padding: 14px;
    }
    .empty-state {
      min-height: 160px;
      display: flex;
      flex-direction: column;
      justify-content: center;
      align-items: flex-start;
    }
    .detail-header {
      display: flex;
      justify-content: space-between;
      gap: 14px;
      align-items: flex-start;
      border-bottom: 1px solid var(--line);
      padding-bottom: 12px;
      margin-bottom: 12px;
    }
    .detail-title h2 { margin: 0 0 6px; font-size: 18px; overflow-wrap: anywhere; }
    .detail-subtitle { display: flex; gap: 8px; flex-wrap: wrap; align-items: center; }
    .inspect-actions { display: flex; gap: 8px; flex-wrap: wrap; justify-content: flex-end; }
    .detail-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
    }
    .panel {
      border: 1px solid var(--line);
      background: #fff;
      min-width: 0;
    }
    .panel.full { grid-column: 1 / -1; }
    .panel h3 {
      margin: 0;
      padding: 9px 10px;
      border-bottom: 1px solid var(--line);
      font-size: 14px;
      background: #f9fafb;
    }
    .panel-body { padding: 10px; }
    .metric-grid {
      display: grid;
      grid-template-columns: repeat(4, minmax(120px, 1fr));
      gap: 8px;
    }
    .metric {
      border: 1px solid var(--line);
      padding: 8px;
      min-width: 0;
    }
    .metric-label { color: var(--muted); font-size: 12px; }
    .metric-value { margin-top: 3px; font-weight: 700; overflow-wrap: anywhere; }
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
    .empty-note {
      color: var(--muted);
      border: 1px dashed var(--line);
      padding: 10px;
      background: #fbfdff;
    }
    .link-row { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 8px; }
    .job-status {
      margin-top: 8px;
      color: var(--muted);
      border: 1px solid var(--line);
      padding: 8px;
      min-height: 34px;
    }
    a { color: var(--blue); }
    @media (max-width: 1180px) {
      th, td { white-space: normal; }
      table { min-width: 0; }
      .detail-header { display: block; }
      .inspect-actions { justify-content: flex-start; margin-top: 10px; }
      .detail-grid { grid-template-columns: 1fr; }
      .metric-grid { grid-template-columns: repeat(2, minmax(120px, 1fr)); }
    }
    @media (max-width: 680px) {
      main { padding: 10px; }
      header { padding: 0 10px; }
      .toolbar { flex-wrap: wrap; }
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
      <label>Mode <select id="mode-filter"><option value="all">All</option></select></label>
      <button id="refresh-button">Refresh</button>
    </div>
    <section class="table-panel">
      <div class="table-scroll">
        <table>
          <thead>
            <tr>
              <th>Health</th>
              <th>POD</th>
              <th>Mode</th>
              <th>Account</th>
              <th>Next</th>
              <th>Decision</th>
              <th>VPlan</th>
              <th>Equity</th>
              <th>Pos</th>
              <th>Warn</th>
              <th>DIFF</th>
            </tr>
          </thead>
          <tbody id="pod-table-body"></tbody>
        </table>
      </div>
    </section>
    <section id="detail-panel" class="detail-workspace empty-state">
        <h2>No POD selected</h2>
        <div class="muted">Select a POD row to inspect details.</div>
    </section>
  </main>
  <script>
    const state = { pods: [], selectedPod: null, selectedDetail: null, jobs: {} };
    const tbody = document.getElementById('pod-table-body');
    const modeFilter = document.getElementById('mode-filter');
    const detailPanel = document.getElementById('detail-panel');
    const lastRefresh = document.getElementById('last-refresh');
    document.getElementById('refresh-button').addEventListener('click', refreshPods);
    modeFilter.addEventListener('change', renderPods);

    function fmt(value) {
      if (value === null || value === undefined || value === '') return '-';
      if (typeof value === 'number') {
        if (Math.abs(value) >= 1000) return value.toLocaleString(undefined, { maximumFractionDigits: 2 });
        return value.toLocaleString(undefined, { maximumFractionDigits: 4 });
      }
      return String(value);
    }

    function fmtPct(value) {
      if (value === null || value === undefined || value === '') return '-';
      return (Number(value) * 100).toLocaleString(undefined, { maximumFractionDigits: 4 }) + '%';
    }

    function fmtBool(value) {
      if (value === null || value === undefined) return '-';
      return value ? 'yes' : 'no';
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
      return String(value);
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

    async function fetchJson(url, options) {
      const response = await fetch(url, options || {});
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.message_str || response.statusText);
      return payload;
    }

    async function refreshPods() {
      try {
        const payload = await fetchJson('/api/pods');
        state.pods = payload.pod_row_dict_list || [];
        lastRefresh.textContent = 'Last refresh: ' + new Date().toLocaleTimeString();
        syncModeFilter(payload.mode_list || []);
        renderPods();
      } catch (error) {
        lastRefresh.textContent = 'Refresh failed: ' + error.message;
      }
    }

    function syncModeFilter(modeList) {
      const current = modeFilter.value;
      modeFilter.innerHTML = '<option value="all">All</option>' + modeList.map(mode => `<option value="${esc(mode)}">${esc(mode)}</option>`).join('');
      modeFilter.value = modeList.includes(current) ? current : 'all';
    }

    function renderPods() {
      const selectedMode = modeFilter.value;
      const rows = state.pods.filter(row => selectedMode === 'all' || row.mode_str === selectedMode);
      tbody.innerHTML = rows.map(row => `
        <tr data-pod="${esc(row.pod_id_str)}" class="${row.pod_id_str === state.selectedPod ? 'selected' : ''}">
          <td>${pill(row.health_str, healthClass(row.health_str))}</td>
          <td><span class="pod-link">${esc(row.pod_id_str)}</span><br><span class="muted strategy-text">${esc(row.strategy_import_str)}</span></td>
          <td>${esc(row.mode_str)}</td>
          <td>${esc(row.account_route_str)}</td>
          <td>${esc(row.next_action_str)}<br><span class="muted">${esc(row.reason_code_str)}</span></td>
          <td>${esc(row.latest_decision_plan_status_str)}</td>
          <td>${esc(row.latest_vplan_status_str)} ${row.latest_vplan_id_int ? '#' + esc(row.latest_vplan_id_int) : ''}</td>
          <td class="num">${esc(row.equity_float)}<br><span class="muted">cash ${esc(row.cash_float)}</span></td>
          <td>${esc(row.position_count_int)}</td>
          <td>${esc(row.warning_count_int)}</td>
          <td>${renderDiffCell(row)}</td>
        </tr>`).join('');
      tbody.querySelectorAll('tr[data-pod]').forEach(el => el.addEventListener('click', () => loadDetail(el.dataset.pod)));
    }

    function renderDiffCell(row) {
      const status = row.latest_diff_status_str || 'not_run';
      const cls = status === 'red' ? 'red' : status === 'yellow' ? 'yellow' : status === 'green' ? 'green' : 'gray';
      const link = row.latest_diff_artifact_url_str ? `<br><a href="${esc(row.latest_diff_artifact_url_str)}" target="_blank">artifact</a>` : '';
      return `${pill(status, cls)}<br><span class="muted">${esc(row.latest_diff_equity_tracking_error_float)}</span>${link}`;
    }

    async function loadDetail(podId) {
      state.selectedPod = podId;
      renderPods();
      detailPanel.className = 'detail-workspace';
      detailPanel.innerHTML = '<h2>' + esc(podId) + '</h2><div class="muted">Loading...</div>';
      try {
        const payload = await fetchJson('/api/pods/' + encodeURIComponent(podId));
        state.selectedDetail = payload;
        renderDetail(payload);
      } catch (error) {
        detailPanel.innerHTML = '<h2>' + esc(podId) + '</h2><div class="red pill">error</div><div class="job-status">' + esc(error.message) + '</div>';
      }
    }

    function renderDetail(payload) {
      const row = payload.pod_row_dict || {};
      const decision = payload.latest_decision_plan_dict || null;
      const vplan = payload.latest_vplan_dict || {};
      const report = payload.latest_execution_report_dict || {};
      const diff = payload.latest_diff_dict || {};
      const events = payload.event_dict_list || [];
      const artifactButton = diff.html_url_str ? `<a class="button-link" href="${esc(diff.html_url_str)}" target="_blank"><button class="secondary">Open artifact</button></a>` : '';
      detailPanel.innerHTML = `
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
            <button class="secondary" data-copy-command="status">Copy status command</button>
            <button class="secondary" data-copy-command="show_decision_plan">Copy show_decision_plan command</button>
            <button class="secondary" data-copy-command="show_vplan">Copy show_vplan command</button>
            ${artifactButton}
          </div>
        </div>
        <div class="detail-grid">
          ${renderOverviewSection(row)}
          ${renderDecisionSection(decision)}
          ${renderVPlanSection(vplan)}
          ${renderBrokerSection(vplan, report, row)}
          ${renderDiffSection(diff)}
          ${renderEventsSection(events)}
        </div>`;
      document.getElementById('refresh-detail-button').addEventListener('click', () => loadDetail(row.pod_id_str));
      document.querySelectorAll('[data-copy-command]').forEach(button => {
        button.addEventListener('click', () => copyCommand(row, button.dataset.copyCommand, button));
      });
      const runDiffButton = document.getElementById('run-diff-button');
      if (runDiffButton) {
        runDiffButton.addEventListener('click', () => runDiff(row.pod_id_str));
      }
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

    function panel(title, body, full) {
      return `<section class="panel ${full ? 'full' : ''}"><h3>${esc(title)}</h3><div class="panel-body">${body}</div></section>`;
    }

    function metric(label, value) {
      return `<div class="metric"><div class="metric-label">${esc(label)}</div><div class="metric-value">${esc(value)}</div></div>`;
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
      const parts = ['uv run python -m alpha.live.runner', commandName, '--mode', row.mode_str, '--pod-id', row.pod_id_str];
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
