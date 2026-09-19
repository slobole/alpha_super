"""LIVE Overview only. No V3 action routes, executors, journal or notifications."""

from datetime import UTC, datetime
from pathlib import Path

from flask import Flask, Response, abort, jsonify, render_template, request, send_from_directory, url_for

from alpha.live.dashboard_v4.data import LiveDataProvider, load_workspace_snapshot_tuple
from alpha.live.dashboard_v4.overview import build_overview_dict
from alpha.live.ibkr_performance import resolve_performance_db_path_str


PERIOD_TUPLE = ("1M", "3M", "YTD", "All")
ASSET_SET = {
    "htmx.min.js", "fonts/IBMPlexSans-latin.woff2",
    "fonts/IBMPlexMono-400-latin.woff2", "fonts/IBMPlexMono-500-latin.woff2",
}


def create_app(data_provider_obj=None, *, performance_db_path_str=None,
               workspace_snapshot_fn=None, now_fn=None, demo_bool=False) -> Flask:
    flask_app_obj = Flask(__name__)
    provider_obj = data_provider_obj if data_provider_obj is not None else LiveDataProvider()
    clock_fn = now_fn or (lambda: datetime.now(UTC))
    database_path_str = performance_db_path_str or resolve_performance_db_path_str()
    flask_app_obj.config.update(read_only_bool=True, demo_bool=demo_bool)

    @flask_app_obj.before_request
    def read_only_boundary():
        # This runs before route dispatch/provider access, including unknown
        # legacy paths. There is no flag to enable executable actions in V4.
        if request.method not in {"GET", "HEAD", "OPTIONS"}:
            return jsonify(error="read_only", message="This console is read-only."), 403
        if request.endpoint not in {"static", "assets"} and (
            request.remote_addr not in {None, "127.0.0.1", "::1"} and not request.is_secure
        ):
            return Response("HTTPS is required for remote operator access.", status=426)

    @flask_app_obj.after_request
    def response_headers(response_obj):
        response_obj.headers["Cache-Control"] = "no-store"
        response_obj.headers["X-Content-Type-Options"] = "nosniff"
        response_obj.headers["Referrer-Policy"] = "same-origin"
        response_obj.headers["X-Frame-Options"] = "DENY"
        response_obj.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
            "font-src 'self'; img-src 'self' data:; connect-src 'self'; "
            "object-src 'none'; base-uri 'none'; frame-ancestors 'none'; form-action 'none'"
        )
        return response_obj

    def context_dict():
        if set(request.args) - {"period"} or len(request.args.getlist("period")) > 1:
            abort(400)
        period_str = request.args.get("period", "3M")
        if period_str not in PERIOD_TUPLE:
            abort(400)
        acquisition_ts = clock_fn()
        if workspace_snapshot_fn is None:
            workspace_dict, snapshot_obj = load_workspace_snapshot_tuple(
                provider_obj, database_path_str, as_of_ts=acquisition_ts)
        else:
            workspace_dict, snapshot_obj = workspace_snapshot_fn()
        # The provider stamps source assessment during acquisition. Compare
        # freshness to completion, never to the request's earlier start clock.
        overview_dict = build_overview_dict(
            workspace_dict, snapshot_obj, provider_obj,
            as_of_ts=clock_fn(), period_str=period_str, demo_bool=demo_bool,
        )
        overview_dict.update(
            refresh_url_str=url_for("refresh", period=period_str),
            refresh_seconds_int=15,
            period_option_list=[{
                "label_str": option_str, "url_str": url_for("index", period=option_str),
                "selected_bool": option_str == period_str,
            } for option_str in PERIOD_TUPLE],
        )
        return {"overview_dict": overview_dict}

    @flask_app_obj.get("/")
    def index():
        template_str = "_overview.html" if request.headers.get("HX-Request") == "true" else "overview.html"
        return render_template(template_str, **context_dict())

    @flask_app_obj.get("/overview/refresh")
    def refresh():
        return render_template("_overview.html", **context_dict())

    @flask_app_obj.get("/assets/<path:filename>")
    def assets(filename):
        if filename not in ASSET_SET:
            abort(404)
        asset_root_obj = Path(__file__).resolve().parent.parent / "dashboard_v3" / "static"
        return send_from_directory(asset_root_obj, filename)

    @flask_app_obj.get("/healthz")
    def healthz():
        # Process liveness only; this does not assert broker or data health.
        return jsonify(service="dashboard_v4", scope="live", read_only=True)

    return flask_app_obj
