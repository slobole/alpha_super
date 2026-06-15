"""Data provider for Dashboard V3.

Thin wrapper around the existing ``alpha.live.dashboard`` builders. The wrapper
keeps a single ``DashboardApp`` instance per Flask process and exposes only the
read-side functions Phase 1 needs.

A short-lived in-memory cache (default 2s) sits in front of
``build_dashboard_summary_dict`` because the summary is polled by every visible
tab — without it, multiple HTMX fragments would each trigger a full sqlite scan
on every refresh tick. Per-pod detail is *not* cached: it is only fetched when
an operator opens a pod, and freshness matters more than throughput there.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import threading
import time
from typing import Any

from alpha.live import dashboard as dashboard_module
from alpha.live.dashboard import (
    DashboardActionInFlightError,
    DashboardApp,
    DashboardPodTarget,
    DEFAULT_CONFIG_PATH_STR,
    DEFAULT_EVENT_LOG_PATH_STR,
    DEFAULT_EVENT_LIMIT_INT,
    DEFAULT_RELEASES_ROOT_PATH_STR,
    DEFAULT_RESULTS_ROOT_PATH_STR,
    DEFAULT_TRACE_EVENT_LIMIT_INT,
    build_dashboard_summary_dict,
    build_pod_detail_dict,
    load_recent_event_dict_list,
    load_recent_trace_event_dict_list,
)


# Polish-branch decision: the underlying ``build_dashboard_summary_dict`` is now
# ~150 ms for a 5-pod book (down from ~5 s before the reverse-stream event
# reader landed). 8 s of cache absorbs HTMX polling bursts and lets repeated
# tab switches feel instant without hiding a meaningfully stale view.
SUMMARY_CACHE_SECONDS_FLOAT = 8.0


@dataclass
class DashboardDataProvider:
    """Production data provider. Wraps a real ``DashboardApp``."""

    releases_root_path_str: str = DEFAULT_RELEASES_ROOT_PATH_STR
    config_path_str: str = DEFAULT_CONFIG_PATH_STR
    results_root_path_str: str = DEFAULT_RESULTS_ROOT_PATH_STR
    event_log_path_str: str = DEFAULT_EVENT_LOG_PATH_STR
    _app_obj: DashboardApp | None = field(default=None, init=False, repr=False)
    _summary_cache_dict: dict[str, Any] | None = field(default=None, init=False, repr=False)
    _summary_cache_at_float: float = field(default=0.0, init=False, repr=False)
    _summary_cache_lock_obj: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False
    )

    def app_obj(self) -> DashboardApp:
        if self._app_obj is None:
            self._app_obj = DashboardApp(
                releases_root_path_str=self.releases_root_path_str,
                config_path_str=self.config_path_str,
                results_root_path_str=self.results_root_path_str,
                event_log_path_str=self.event_log_path_str,
            )
        return self._app_obj

    def get_summary_dict(self) -> dict[str, Any]:
        with self._summary_cache_lock_obj:
            now_float = time.monotonic()
            cache_age_float = now_float - self._summary_cache_at_float
            if (
                self._summary_cache_dict is not None
                and cache_age_float < SUMMARY_CACHE_SECONDS_FLOAT
            ):
                return self._summary_cache_dict
            summary_dict = build_dashboard_summary_dict(self.app_obj())
            self._summary_cache_dict = summary_dict
            self._summary_cache_at_float = now_float
            return summary_dict

    def get_pod_detail_dict(self, pod_id_str: str) -> dict[str, Any]:
        return build_pod_detail_dict(self.app_obj(), pod_id_str)

    def get_pod_event_dict_list(
        self,
        pod_id_str: str,
        limit_int: int = DEFAULT_EVENT_LIMIT_INT,
    ) -> list[dict[str, Any]]:
        return load_recent_event_dict_list(
            log_path_str=self.app_obj().event_log_path_str,
            pod_id_str=pod_id_str,
            limit_int=limit_int,
        )

    def get_pod_trace_event_dict_list(
        self,
        pod_id_str: str,
        limit_int: int = DEFAULT_TRACE_EVENT_LIMIT_INT,
    ) -> list[dict[str, Any]]:
        # Newest cycle's structured trace, read from logs/pods/<pod>/<run>/.
        return load_recent_trace_event_dict_list(
            pod_id_str=pod_id_str,
            limit_int=limit_int,
        )

    def get_action_token_str(self) -> str:
        return self.app_obj().action_token_str

    def get_target_for_pod(self, pod_id_str: str) -> DashboardPodTarget | None:
        return self.app_obj().get_target_for_pod(pod_id_str)

    def start_diff_job(self, target_obj: DashboardPodTarget) -> dict[str, Any]:
        app_obj = self.app_obj()
        assert app_obj.diff_job_manager_obj is not None
        job_obj = app_obj.diff_job_manager_obj.start_job(
            pod_target_obj=target_obj,
            releases_root_path_str=app_obj.releases_root_path_str,
            results_root_path_str=app_obj.results_root_path_str,
        )
        return job_obj.to_dict()

    def start_action_job(
        self, action_name_str: str, target_obj: DashboardPodTarget
    ) -> dict[str, Any]:
        app_obj = self.app_obj()
        assert app_obj.action_job_manager_obj is not None
        job_obj = app_obj.action_job_manager_obj.start_job(
            action_name_str=action_name_str,
            pod_target_obj=target_obj,
            releases_root_path_str=app_obj.releases_root_path_str,
            results_root_path_str=app_obj.results_root_path_str,
            event_log_path_str=app_obj.event_log_path_str,
        )
        return job_obj.to_dict()

    def submit_manual_order_dict(
        self,
        target_obj: DashboardPodTarget,
        request_body_dict: dict[str, Any],
    ) -> dict[str, Any]:
        from alpha.live.manual_order import submit_manual_order_ticket_dict

        app_obj = self.app_obj()
        assert app_obj.pod_job_gate_obj is not None
        app_obj.pod_job_gate_obj.acquire(target_obj.release_obj.pod_id_str)
        try:
            return submit_manual_order_ticket_dict(
                release_obj=target_obj.release_obj,
                request_body_dict=request_body_dict,
                log_path_str=app_obj.event_log_path_str,
            )
        finally:
            app_obj.pod_job_gate_obj.release(target_obj.release_obj.pod_id_str)

    def export_trade_sheet_path_str(self, target_obj: DashboardPodTarget) -> str:
        """Write the pod's trade sheet xlsx under results_root and return its path.

        Read-only with respect to trading: renders the already-persisted
        DecisionPlan + VPlan from the pod's own DB. Local imports keep the
        provider stub-friendly for route tests.
        """
        from datetime import UTC, datetime
        from pathlib import Path

        from alpha.live.state_store_v2 import LiveStateStore
        from alpha.live.trade_sheet import export_trade_sheet_detail_dict

        # A GET must not create files: LiveStateStore.__init__ would otherwise
        # initialize an empty schema DB at this path if the pod never ran.
        if not Path(target_obj.db_path_str).exists():
            raise ValueError(
                f"Pod DB does not exist yet at '{target_obj.db_path_str}'. "
                "There is nothing to export: the pod has not run in this mode."
            )
        state_store_obj = LiveStateStore(target_obj.db_path_str)
        detail_dict = export_trade_sheet_detail_dict(
            state_store_obj=state_store_obj,
            pod_id_str=target_obj.release_obj.pod_id_str,
            env_mode_str=target_obj.release_obj.mode_str,
            generated_at_ts=datetime.now(UTC),
            output_dir_str=self.results_root_path_str,
        )
        return str(detail_dict["output_path_str"])

    def get_job_dict(self, job_id_str: str) -> dict[str, Any] | None:
        app_obj = self.app_obj()
        if app_obj.diff_job_manager_obj is not None:
            job_dict = app_obj.diff_job_manager_obj.get_job_dict(job_id_str)
            if job_dict is not None:
                return job_dict
        if app_obj.action_job_manager_obj is not None:
            return app_obj.action_job_manager_obj.get_job_dict(job_id_str)
        return None


def get_pod_row_dict_list_for_mode(
    summary_dict: dict[str, Any], mode_str: str
) -> list[dict[str, Any]]:
    pod_row_dict_list = summary_dict.get("pod_row_dict_list") or []
    return [row_dict for row_dict in pod_row_dict_list if row_dict.get("mode_str") == mode_str]


def get_pod_row_dict_by_id(
    summary_dict: dict[str, Any], pod_id_str: str
) -> dict[str, Any] | None:
    for row_dict in summary_dict.get("pod_row_dict_list") or []:
        if row_dict.get("pod_id_str") == pod_id_str:
            return row_dict
    return None


def count_pods_needing_action_int(pod_row_dict_list: list[dict[str, Any]]) -> int:
    return sum(1 for row_dict in pod_row_dict_list if _is_red_severity_bool(row_dict))


def count_pods_waiting_int(pod_row_dict_list: list[dict[str, Any]]) -> int:
    return sum(1 for row_dict in pod_row_dict_list if _is_yellow_severity_bool(row_dict))


def _is_red_severity_bool(row_dict: dict[str, Any]) -> bool:
    return _effective_severity_str(row_dict) == "red"


def _is_yellow_severity_bool(row_dict: dict[str, Any]) -> bool:
    return _effective_severity_str(row_dict) == "yellow"


def _effective_severity_str(row_dict: dict[str, Any]) -> str:
    debug_summary_dict = row_dict.get("debug_summary_dict") or {}
    required_action_dict = row_dict.get("required_action_dict") or {}
    severity_str = (
        debug_summary_dict.get("severity_str")
        or required_action_dict.get("severity_str")
        or row_dict.get("health_str")
        or "gray"
    )
    return _normalize_severity_str(severity_str)


def _normalize_severity_str(severity_str: str | None) -> str:
    text_str = str(severity_str or "gray").lower()
    if "red" in text_str or "error" in text_str or "fail" in text_str:
        return "red"
    if (
        "yellow" in text_str
        or "warn" in text_str
        or "waiting" in text_str
        or "running" in text_str
        or "queued" in text_str
    ):
        return "yellow"
    if (
        "green" in text_str
        or "success" in text_str
        or "complete" in text_str
        or "succeeded" in text_str
    ):
        return "green"
    return "gray"


__all__ = [
    "DashboardDataProvider",
    "count_pods_needing_action_int",
    "count_pods_waiting_int",
    "get_pod_row_dict_by_id",
    "get_pod_row_dict_list_for_mode",
]


# Re-export so tests can monkey-patch via this module instead of reaching into
# alpha.live.dashboard.
_dashboard_module_obj = dashboard_module
