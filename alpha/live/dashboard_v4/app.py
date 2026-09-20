"""LIVE Overview and Pod evidence. No actions, executors or notifications."""

from datetime import UTC, datetime
from pathlib import Path
import re

from flask import Flask, Response, abort, jsonify, render_template, request, send_from_directory, url_for

from alpha.live.dashboard_v4.data import LiveDataProvider, load_workspace_snapshot_tuple
from alpha.live.dashboard_v4.overview import build_overview_dict
from alpha.live.dashboard_v4.pod import TAB_TUPLE, build_pod_page_dict
from alpha.live.dashboard_v4.pod_finance import build_pod_finance_dict
from alpha.live.dashboard_v3.client_operations import SOURCE_MAX_AGE_SECONDS_INT
from alpha.live.dashboard_v3.filters import MARKET_TIMEZONE_OBJ
from alpha.live.ops_report import parse_timestamp_ts
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

    def context_dict(pod_id_str=None):
        allowed_set = {"period", "cycle", "tab"} if pod_id_str is not None else {"period"}
        if set(request.args) - allowed_set or any(len(request.args.getlist(key_str)) > 1 for key_str in allowed_set):
            abort(400)
        period_str = request.args.get("period", "3M")
        if period_str not in PERIOD_TUPLE:
            abort(400)
        tab_str, cycle_str = request.args.get("tab", ""), request.args.get("cycle", "")
        if tab_str and tab_str not in {key_str for key_str, _ in TAB_TUPLE}:
            abort(400)
        cycle_match_obj = re.fullmatch(r"(decision|vplan):([1-9][0-9]{0,9})", cycle_str) if cycle_str else None
        if cycle_str and cycle_match_obj is None:
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
            include_finance_bool=pod_id_str is None,
        )
        overview_dict.update(
            refresh_url_str=url_for("refresh", period=period_str),
            refresh_seconds_int=15,
            period_option_list=[{
                "label_str": option_str, "url_str": url_for("index", period=option_str),
                "selected_bool": option_str == period_str,
            } for option_str in PERIOD_TUPLE],
        )
        if pod_id_str is None:
            return {"overview_dict": overview_dict}
        if not any(item_dict["pod_id_str"] == pod_id_str for item_dict in overview_dict["pod_list"]):
            abort(404)
        source_dict = {"status_str": "unknown", "reason_str": "Saved cycle unavailable"}
        selected_current_bool = not cycle_str
        if overview_dict["source_fresh_bool"]:
            summary_dict = workspace_dict.get("summary_dict") or {}
            matched_list = [item_dict for item_dict in summary_dict.get("pod_row_dict_list") or []
                if isinstance(item_dict, dict) and item_dict.get("pod_id_str") == pod_id_str and item_dict.get("mode_str") == "live"]
            account_list = workspace_dict.get("operations_account_list") or []
            identity_list = [item_dict for item_dict in account_list if item_dict.get("pod_id") == pod_id_str]
            if len(matched_list) == len(identity_list) == 1 and matched_list[0].get("account_route_str") == identity_list[0]["account_route"]:
                row_dict = matched_list[0]
                if cycle_match_obj:
                    selected_current_bool = int(cycle_match_obj[2]) == row_dict.get(
                        "latest_decision_plan_id_int" if cycle_match_obj[1] == "decision" else "latest_vplan_id_int")
                selected_id_dict = {}
                if cycle_match_obj:
                    selected_id_dict["decision_plan_id_int" if cycle_match_obj[1] == "decision" else "vplan_id_int"] = int(cycle_match_obj[2])
                elif row_dict.get("latest_vplan_id_int") and (row_dict.get("latest_vplan_is_for_latest_decision_bool") is not False
                    or row_dict.get("latest_vplan_status_str") in {"ready", "submitting", "submitted", "blocked", "expired"}
                    or (row_dict.get("missing_ack_count_int") or 0) > 0 or row_dict.get("latest_submit_ack_status_str") == "missing_critical"):
                    selected_id_dict["vplan_id_int"] = row_dict["latest_vplan_id_int"]
                elif row_dict.get("latest_decision_plan_id_int"):
                    selected_id_dict["decision_plan_id_int"] = row_dict["latest_decision_plan_id_int"]
                if hasattr(provider_obj, "get_pod_cycles_dict"):
                    source_dict = provider_obj.get_pod_cycles_dict(pod_id_str, as_of_ts=acquisition_ts, **selected_id_dict)
                if source_dict.get("status_str") == "not_found":
                    abort(404)
                selected_row_dict = source_dict.get("pod_row_dict") or {}
                if source_dict.get("status_str") == "ok" and any(selected_row_dict.get(key_str) != row_dict.get(key_str)
                    for key_str in ("pod_id_str", "account_route_str", "mode_str", "user_id_str")):
                    source_dict = {"status_str": "unknown", "reason_str": "Cycle identity not verified"}
                elif source_dict.get("status_str") == "ok" and all(selected_row_dict.get(key_str) == row_dict.get(key_str)
                    for key_str in ("latest_decision_plan_id_int", "latest_vplan_id_int", "release_id_str")):
                    # The latest monthly cycle may still be weeks old. Its Data
                    # stage keeps the decision's saved snapshot; current data
                    # readiness remains in Overview and the Pod header.
                    if "reconcile_read_failure_dict" in row_dict:
                        selected_row_dict["reconcile_read_failure_dict"] = row_dict["reconcile_read_failure_dict"]
        pod_finance_dict = build_pod_finance_dict(workspace_dict, snapshot_obj, provider_obj,
            pod_id_str=pod_id_str, as_of_ts=clock_fn(), period_str=period_str)
        render_ts = clock_fn()
        source_ts = parse_timestamp_ts((workspace_dict.get("summary_dict") or {}).get("as_of_timestamp_str"))
        selected_ts = parse_timestamp_ts((source_dict.get("pod_row_dict") or {}).get("as_of_timestamp_str"))
        final_fresh_bool = source_ts is not None and 0 <= (render_ts - source_ts).total_seconds() <= SOURCE_MAX_AGE_SECONDS_INT
        if not final_fresh_bool:
            overview_dict = build_overview_dict(workspace_dict, snapshot_obj, provider_obj,
                as_of_ts=render_ts, period_str=period_str, demo_bool=demo_bool, include_finance_bool=False)
            overview_dict["refresh_seconds_int"] = 15
        remaining_float = min((SOURCE_MAX_AGE_SECONDS_INT - (render_ts - timestamp_ts).total_seconds()
                              for timestamp_ts in (source_ts, selected_ts) if timestamp_ts is not None), default=0)
        overview_dict.update(source_valid_ms_int=max(0, int(remaining_float * 1000)),
            clock_timestamp_str=render_ts.isoformat(),
            clock_str=render_ts.astimezone(MARKET_TIMEZONE_OBJ).strftime("%H:%M:%S"),
            date_str=render_ts.astimezone(MARKET_TIMEZONE_OBJ).strftime("%a %m-%d"))
        source_dict["selected_explicit_bool"] = bool(cycle_str)
        source_dict["selected_current_bool"] = selected_current_bool
        pod_page_dict = build_pod_page_dict(overview_dict, source_dict, pod_finance_dict,
            pod_id_str=pod_id_str, as_of_ts=render_ts, tab_str=tab_str)
        selected_cycle_str = (source_dict.get("selected_cycle_dict") or {}).get("cycle_key_str") or cycle_str
        def pod_url_str(**options_dict):
            return url_for("pod", pod_id_str=pod_id_str, period=options_dict.get("period", period_str),
                cycle=options_dict.get("cycle", selected_cycle_str), tab=options_dict.get("tab", pod_page_dict["tab_str"]))
        for item_dict in pod_page_dict["tab_list"]:
            item_dict["url_str"] = pod_url_str(tab=item_dict["key_str"])
        for item_dict in pod_page_dict["step_list"]:
            item_dict["url_str"] = pod_url_str(tab=item_dict["tab_str"])
        cycle_list = pod_page_dict["cycle_list"]
        for item_dict in cycle_list:
            item_dict["url_str"] = pod_url_str(cycle=item_dict["cycle_key_str"], tab="plan")
            session_str = "Close" if item_dict.get("execution_policy_str") == "same_day_moc" else "Open"
            date_str = item_dict.get("session_date_str") or "Unknown date"
            item_dict["label_str"] = date_str + " · " + session_str
        selected_index_int = next((index_int for index_int, item_dict in enumerate(cycle_list) if item_dict["cycle_key_str"] == selected_cycle_str), -1)
        pod_page_dict.update(
            previous_cycle_dict=cycle_list[selected_index_int + 1] if 0 <= selected_index_int < len(cycle_list) - 1 else {},
            next_cycle_dict=cycle_list[selected_index_int - 1] if selected_index_int > 0 else {},
            cycle_label_str=cycle_list[selected_index_int]["label_str"] if selected_index_int >= 0 else "Cycle unavailable",
            period_str=period_str, period_option_list=[{"label_str": option_str, "url_str": pod_url_str(period=option_str), "selected_bool": option_str == period_str} for option_str in PERIOD_TUPLE])
        overview_dict["refresh_url_str"] = url_for("pod_refresh", pod_id_str=pod_id_str,
            period=period_str, cycle=cycle_str, tab=tab_str)
        return {"overview_dict": overview_dict, "pod_page_dict": pod_page_dict}

    @flask_app_obj.get("/")
    def index():
        template_str = "_overview.html" if request.headers.get("HX-Request") == "true" else "overview.html"
        return render_template(template_str, **context_dict())

    @flask_app_obj.get("/overview/refresh")
    def refresh():
        return render_template("_overview.html", **context_dict())

    @flask_app_obj.get("/pods/<pod_id_str>")
    def pod(pod_id_str):
        template_str = "_overview.html" if request.headers.get("HX-Request") == "true" else "overview.html"
        return render_template(template_str, **context_dict(pod_id_str))

    @flask_app_obj.get("/pods/<pod_id_str>/refresh")
    def pod_refresh(pod_id_str):
        return render_template("_overview.html", **context_dict(pod_id_str))

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
