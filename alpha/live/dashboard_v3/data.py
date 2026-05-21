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
    DashboardApp,
    DEFAULT_CONFIG_PATH_STR,
    DEFAULT_EVENT_LOG_PATH_STR,
    DEFAULT_EVENT_LIMIT_INT,
    DEFAULT_RELEASES_ROOT_PATH_STR,
    DEFAULT_RESULTS_ROOT_PATH_STR,
    build_dashboard_summary_dict,
    build_pod_detail_dict,
    load_recent_event_dict_list,
)


SUMMARY_CACHE_SECONDS_FLOAT = 2.0


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

    def get_action_token_str(self) -> str:
        return self.app_obj().action_token_str


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
