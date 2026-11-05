"""Bench routes for the Strategy map: the page, its data file, and the refresh job.

The page is static markup plus engine_map.js, which fetches /map/data.json and
draws everything in the browser. The refresh button starts the same command an
operator would type, through the Bench job runner, so it shows on the Jobs page.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime

from flask import Response, redirect, render_template, send_file, url_for

from alpha.bench import engine_map

MAP_JOB_KIND_STR = "map"
MAP_JOB_TARGET_STR = "engine_map"
MAP_JOB_LABEL_STR = "Strategy map · refresh"
MISSING_DATA_TEXT_STR = (
    "The strategy map has no data yet. Start a refresh on /map, or run "
    "python -m alpha.bench.engine_map build."
)


def refresh_command_list() -> list[str]:
    return [sys.executable, "-X", "utf8", "-m", "alpha.bench.engine_map", "refresh"]


def _date_words_str(iso_str: str | None, *, with_time_bool: bool) -> str | None:
    """Oct 30, 2026 or Nov 5, 00:09, the page's own wording. None when unreadable."""
    try:
        moment_obj = datetime.fromisoformat(str(iso_str))
    except (TypeError, ValueError):
        return None
    if with_time_bool:
        return f"{moment_obj:%b} {moment_obj.day}, {moment_obj:%H:%M}"
    return f"{moment_obj:%b} {moment_obj.day}, {moment_obj.year}"


def build_summary_dict() -> dict:
    try:
        return json.loads(engine_map.BUILD_SUMMARY_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def register_routes(app_obj, csrf_failure_fn) -> None:
    def refresh_job_obj():
        for job_obj in app_obj.config["job_manager_obj"].list_jobs():
            if getattr(job_obj, "kind_str", None) == MAP_JOB_KIND_STR and getattr(job_obj, "is_active_bool", False):
                return job_obj
        return None

    @app_obj.route("/map")
    def map_page_fn() -> str:
        summary_dict = build_summary_dict() if engine_map.DATA_PATH.exists() else {}
        return render_template(
            "map.html",
            data_ready_bool=engine_map.DATA_PATH.exists(),
            last_close_str=_date_words_str(summary_dict.get("as_of"), with_time_bool=False),
            built_str=_date_words_str(summary_dict.get("built"), with_time_bool=True),
            warning_list=list(summary_dict.get("warnings") or []),
            refresh_job=refresh_job_obj(),
        )

    @app_obj.route("/map/data.json")
    def map_data_fn() -> Response:
        if not engine_map.DATA_PATH.exists():
            return Response(MISSING_DATA_TEXT_STR, status=404, mimetype="text/plain")
        response_obj = send_file(engine_map.DATA_PATH, mimetype="application/json", max_age=0)
        # A refresh rewrites the file in place: never let the browser keep an old copy.
        response_obj.headers["Cache-Control"] = "no-store"
        return response_obj

    @app_obj.route("/api/map/refresh", methods=["POST"])
    def map_refresh_api_fn() -> Response:
        csrf_failure_obj = csrf_failure_fn()
        if csrf_failure_obj is not None:
            return csrf_failure_obj
        app_obj.config["job_manager_obj"].submit(
            MAP_JOB_LABEL_STR, MAP_JOB_TARGET_STR, MAP_JOB_KIND_STR, refresh_command_list()
        )
        return redirect(url_for("jobs_page_fn"))
