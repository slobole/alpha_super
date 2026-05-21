"""Flask application factory for Dashboard V3.

Phase 1 wires the read-only operator console: three mode pages
(``/live``, ``/paper``, ``/incubation``), an HTMX-polled top bar
verdict, an expandable per-pod detail panel rendered as a vertical
timeline of today's cycle, and a polled events tail per pod.

The factory accepts an injectable ``data_provider_obj`` so tests can
swap in a fixture instead of touching the live ``DashboardApp``.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Protocol

from flask import Flask, Response, abort, redirect, render_template, url_for

from alpha.live.dashboard_v3.data import (
    DashboardDataProvider,
    get_pod_row_dict_list_for_mode,
)
from alpha.live.dashboard_v3.filters import FILTER_MAP_DICT
from alpha.live.dashboard_v3.health import build_health_rollup
from alpha.live.dashboard_v3.schedule import build_schedule_entry_list
from alpha.live.dashboard_v3.verdict import resolve_top_bar_verdict


DASHBOARD_V3_VERSION_STR = "0.2.0-phase-2"
SUPPORTED_MODE_STR_LIST = ["live", "paper", "incubation"]
MODE_LABEL_DICT = {"live": "Live", "paper": "Paper", "incubation": "Incubation"}


class DataProviderProtocol(Protocol):
    def get_summary_dict(self) -> dict[str, Any]: ...
    def get_pod_detail_dict(self, pod_id_str: str) -> dict[str, Any]: ...
    def get_pod_event_dict_list(
        self, pod_id_str: str, limit_int: int = 80
    ) -> list[dict[str, Any]]: ...


def create_app(data_provider_obj: DataProviderProtocol | None = None) -> Flask:
    flask_app_obj = Flask(__name__)
    flask_app_obj.config["data_provider_obj"] = (
        data_provider_obj if data_provider_obj is not None else DashboardDataProvider()
    )

    for filter_name_str, filter_fn in FILTER_MAP_DICT.items():
        flask_app_obj.jinja_env.filters[filter_name_str] = filter_fn

    @flask_app_obj.context_processor
    def inject_globals_fn() -> dict[str, Any]:
        return {
            "version_str": DASHBOARD_V3_VERSION_STR,
            "server_time_str": _now_clock_str(),
        }

    # ── plain routes ─────────────────────────────────────────────────────

    @flask_app_obj.route("/healthz")
    def healthz_route_fn() -> tuple[str, int]:
        return (f"dashboard_v3 ok {DASHBOARD_V3_VERSION_STR}", 200)

    @flask_app_obj.route("/")
    def index_route_fn() -> Response:
        return redirect(url_for("mode_page_route_fn", mode_str="live"))

    @flask_app_obj.route("/<mode_str>")
    def mode_page_route_fn(mode_str: str):
        if mode_str not in SUPPORTED_MODE_STR_LIST:
            abort(404)
        provider_obj = flask_app_obj.config["data_provider_obj"]
        summary_dict = provider_obj.get_summary_dict()
        pod_row_dict_list = get_pod_row_dict_list_for_mode(summary_dict, mode_str)
        attention_row_list = [
            row_dict
            for row_dict in pod_row_dict_list
            if _is_attention_row_bool(row_dict)
        ]
        verdict_obj = resolve_top_bar_verdict(summary_dict)
        return render_template(
            "mode_page.html",
            mode_str=mode_str,
            mode_label_str=MODE_LABEL_DICT[mode_str],
            pod_row_dict_list=pod_row_dict_list,
            attention_row_list=attention_row_list,
            verdict_dict=verdict_obj.as_dict(),
            as_of_clock_str=_now_clock_str(),
        )

    # ── HTMX fragments ───────────────────────────────────────────────────

    @flask_app_obj.route("/fragments/top-bar")
    def top_bar_fragment_route_fn():
        provider_obj = flask_app_obj.config["data_provider_obj"]
        summary_dict = provider_obj.get_summary_dict()
        verdict_obj = resolve_top_bar_verdict(summary_dict)
        return render_template(
            "_top_bar_verdict.html",
            verdict_dict=verdict_obj.as_dict(),
            as_of_clock_str=_now_clock_str(),
        )

    @flask_app_obj.route("/fragments/pod-detail/<pod_id_str>")
    def pod_detail_fragment_route_fn(pod_id_str: str):
        provider_obj = flask_app_obj.config["data_provider_obj"]
        try:
            detail_dict = provider_obj.get_pod_detail_dict(pod_id_str)
        except KeyError:
            abort(404)
        return render_template(
            "_pod_detail.html",
            detail_dict=detail_dict,
            as_of_clock_str=_now_clock_str(),
        )

    @flask_app_obj.route("/fragments/events-tail/<pod_id_str>")
    def events_tail_fragment_route_fn(pod_id_str: str):
        provider_obj = flask_app_obj.config["data_provider_obj"]
        event_dict_list = provider_obj.get_pod_event_dict_list(pod_id_str)
        return render_template(
            "_events_tail.html",
            event_dict_list=event_dict_list,
            pod_id_str=pod_id_str,
        )

    @flask_app_obj.route("/fragments/health-strip")
    def health_strip_fragment_route_fn():
        provider_obj = flask_app_obj.config["data_provider_obj"]
        summary_dict = provider_obj.get_summary_dict()
        health_obj = build_health_rollup(summary_dict)
        return render_template(
            "_health_strip.html",
            health_dict=health_obj.as_dict(),
        )

    @flask_app_obj.route("/fragments/schedule-strip")
    def schedule_strip_fragment_route_fn():
        provider_obj = flask_app_obj.config["data_provider_obj"]
        summary_dict = provider_obj.get_summary_dict()
        schedule_entry_obj_list = build_schedule_entry_list(summary_dict)
        return render_template(
            "_schedule_strip.html",
            schedule_entry_dict_list=[entry_obj.as_dict() for entry_obj in schedule_entry_obj_list],
        )

    return flask_app_obj


# ── helpers ──────────────────────────────────────────────────────────────


def _now_clock_str() -> str:
    return datetime.now(timezone.utc).astimezone().strftime("%H:%M:%S")


def _is_attention_row_bool(row_dict: dict[str, Any]) -> bool:
    from alpha.live.dashboard_v3.data import _effective_severity_str

    return _effective_severity_str(row_dict) in ("red", "yellow")
