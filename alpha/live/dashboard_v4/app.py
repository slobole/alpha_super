"""LIVE operator views. No actions, executors or notifications."""

from datetime import UTC, date, datetime
from pathlib import Path
import re

from flask import Flask, Response, abort, jsonify, render_template, request, send_from_directory, url_for

from alpha.live.dashboard_v4.data import LiveDataProvider, load_workspace_snapshot_tuple
from alpha.live.dashboard_v4.overview import build_overview_dict
from alpha.live.dashboard_v4.pod import TAB_TUPLE, build_pod_page_dict
from alpha.live.dashboard_v4.pod_finance import build_pod_finance_dict
from alpha.live.dashboard_v4.positions import build_positions_page_dict
from alpha.live.dashboard_v4.performance import build_performance_page_dict
from alpha.live.dashboard_v4.performance_exports import export_performance_csv_str, export_performance_pdf_bytes
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

    def context_dict(pod_id_str=None, *, positions_bool=False):
        allowed_set = {"view", "pod"} if positions_bool else ({"period", "cycle", "tab"} if pod_id_str is not None else {"period"})
        if set(request.args) - allowed_set or any(len(request.args.getlist(key_str)) > 1 for key_str in allowed_set):
            abort(400)
        period_str = request.args.get("period", "All")
        if period_str not in PERIOD_TUPLE:
            abort(400)
        tab_str, cycle_str = request.args.get("tab", ""), request.args.get("cycle", "")
        if tab_str and tab_str not in {key_str for key_str, _ in TAB_TUPLE}:
            abort(400)
        cycle_match_obj = re.fullmatch(r"(decision|vplan):([1-9][0-9]{0,9})", cycle_str) if cycle_str else None
        if cycle_str and cycle_match_obj is None:
            abort(400)
        view_str, selected_pod_str = request.args.get("view", "all"), request.args.get("pod", "all")
        if positions_bool and (view_str not in {"all", "changed", "off_target"} or not selected_pod_str or len(selected_pod_str) > 200):
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
            include_finance_bool=pod_id_str is None and not positions_bool,
        )
        overview_dict.update(
            refresh_url_str=url_for("refresh", period=period_str),
            refresh_seconds_int=15,
            period_option_list=[{
                "label_str": option_str, "url_str": url_for("index", period=option_str),
                "selected_bool": option_str == period_str,
            } for option_str in PERIOD_TUPLE],
        )
        if positions_bool:
            positions_page_dict = build_positions_page_dict(workspace_dict, snapshot_obj, provider_obj,
                as_of_ts=clock_fn(), view_str=view_str, pod_str=selected_pod_str)
            pod_option_list = positions_page_dict["pod_filter_list"]
            if selected_pod_str != "all" and not any(item_dict["pod_id_str"] == selected_pod_str for item_dict in pod_option_list):
                abort(404)
            active_pod_set = {item_dict["pod_id_str"] for item_dict in overview_dict["pod_list"]}
            count_available_bool = positions_page_dict["holdings_complete_bool"] or any(
                item_dict["pod_id_str"] == selected_pod_str and item_dict["positions_available_bool"]
                for item_dict in positions_page_dict["pod_row_list"])
            for item_dict in positions_page_dict["pod_row_list"]:
                item_dict["url_str"] = url_for("pod", pod_id_str=item_dict["pod_id_str"]) if item_dict["pod_id_str"] in active_pod_set else ""
            for row_dict in positions_page_dict["row_list"]:
                for item_dict in row_dict["pod_list"]:
                    item_dict["url_str"] = url_for("pod", pod_id_str=item_dict["pod_id_str"])
            positions_page_dict.update(view_str=view_str, pod_str=selected_pod_str, search_str="",
                filter_list=[{
                    "label_str": label_str + (" " + str(positions_page_dict[count_key_str]) if available_bool and count_available_bool else ""),
                    "url_str": url_for("positions", view=key_str, pod=selected_pod_str),
                    "selected_bool": key_str == view_str, "disabled_bool": not available_bool,
                    "detail_str": detail_str,
                } for key_str, label_str, count_key_str, available_bool, detail_str in (
                    ("all", "All", "all_count_int", True, ""),
                    ("changed", "Changed today", "changed_count_int", positions_page_dict["changed_available_bool"], "Today's changes are not verified yet."),
                    ("off_target", "Off target", "off_target_count_int", positions_page_dict["off_target_available_bool"], "Target comparison is not available yet."),
                )],
                pod_filter_list=[{"label_str": "All pods", "selected_bool": selected_pod_str == "all",
                    "url_str": url_for("positions", view=view_str)}] + [{
                    **item_dict, "label_str": item_dict["name_str"], "selected_bool": item_dict["pod_id_str"] == selected_pod_str,
                    "url_str": url_for("positions", view=view_str, pod=item_dict["pod_id_str"]),
                } for item_dict in pod_option_list])
            render_ts = clock_fn()
            # Positions reads may be slow; never renew the header's source lifetime.
            elapsed_float = max(0, (render_ts - parse_timestamp_ts(overview_dict["clock_timestamp_str"])).total_seconds())
            remaining_int = max(0, overview_dict["source_valid_ms_int"] - int(elapsed_float * 1000))
            if remaining_int == 0 and overview_dict["source_fresh_bool"]:
                overview_dict = build_overview_dict(workspace_dict, snapshot_obj, provider_obj,
                    as_of_ts=render_ts, period_str=period_str, demo_bool=demo_bool, include_finance_bool=False)
                overview_dict["refresh_seconds_int"] = 15
            if remaining_int == 0:
                positions_page_dict["source_fresh_bool"] = False
                for row_dict in positions_page_dict["row_list"]:
                    if row_dict["today_pending_bool"]:
                        row_dict["today_detail_str"] = "Unknown"
            overview_dict.update(refresh_url_str=url_for("positions_refresh", view=view_str, pod=selected_pod_str),
                source_valid_ms_int=remaining_int,
                clock_timestamp_str=render_ts.isoformat(),
                clock_str=render_ts.astimezone(MARKET_TIMEZONE_OBJ).strftime("%H:%M:%S"),
                date_str=render_ts.astimezone(MARKET_TIMEZONE_OBJ).strftime("%a %m-%d"))
            return {"overview_dict": overview_dict, "positions_page_dict": positions_page_dict}
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
            pod_id_str=pod_id_str, as_of_ts=clock_fn(), period_str=period_str, performance_db_path_str=database_path_str)
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

    @flask_app_obj.get("/positions")
    def positions():
        template_str = "_overview.html" if request.headers.get("HX-Request") == "true" else "overview.html"
        return render_template(template_str, **context_dict(positions_bool=True))

    @flask_app_obj.get("/positions/refresh")
    def positions_refresh():
        return render_template("_overview.html", **context_dict(positions_bool=True))

    def performance_response(*, refresh_bool=False):
        allowed_set = {"level", "period", "from", "to", "unit"}
        if not refresh_bool:
            allowed_set.update({"download", "expected"})
        if set(request.args) - allowed_set or any(len(request.args.getlist(key_str)) != 1 for key_str in request.args):
            abort(400)
        level_str = request.args.get("level", "portfolio")
        period_str = request.args.get("period", "All")
        unit_str = request.args.get("unit", "pct")
        if level_str not in {"portfolio", "pods"} or period_str not in {"1W", "MTD", "YTD", "All"} or unit_str not in {"pct", "usd"}:
            abort(400)
        acquisition_ts = clock_fn()
        from_date_str, to_date_str = request.args.get("from"), request.args.get("to")
        if "from" in request.args or "to" in request.args:
            try:
                if not all(re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", value_str or "") for value_str in (from_date_str, to_date_str)):
                    raise ValueError("Invalid dates")
                if not date.fromisoformat(from_date_str) <= date.fromisoformat(to_date_str) <= acquisition_ts.astimezone(MARKET_TIMEZONE_OBJ).date():
                    raise ValueError("Invalid interval")
            except ValueError:
                abort(400)
        download_str, expected_str = request.args.get("download"), request.args.get("expected")
        if "download" in request.args or "expected" in request.args:
            if download_str not in {"csv", "pdf"} or re.fullmatch(r"[0-9a-f]{64}", expected_str or "") is None:
                abort(400)
        if workspace_snapshot_fn is None:
            workspace_dict, snapshot_obj = load_workspace_snapshot_tuple(provider_obj, database_path_str, as_of_ts=acquisition_ts)
        else:
            workspace_dict, snapshot_obj = workspace_snapshot_fn()
        performance_page_dict = build_performance_page_dict(workspace_dict, snapshot_obj,
            as_of_ts=acquisition_ts, period_str=period_str, from_date_str=from_date_str,
            to_date_str=to_date_str, unit_str=unit_str, level_str=level_str)
        report_dict = performance_page_dict.get("report_dict") or {}
        if download_str:
            if expected_str != report_dict.get("report_hash_str"):
                return Response("The report changed. Refresh Performance and download again.", status=409)
            content_obj = export_performance_csv_str(report_dict, level_str=level_str) if download_str == "csv" else export_performance_pdf_bytes(report_dict)
            filename_str = f"performance-{level_str if download_str == 'csv' else 'portfolio'}-{performance_page_dict['from_date_str']}-{performance_page_dict['to_date_str']}.{download_str}"
            return Response(content_obj, mimetype="text/csv" if download_str == "csv" else "application/pdf",
                headers={"Content-Disposition": f'attachment; filename="{filename_str}"'})
        # Assess operational freshness after the financial read. A slow report
        # must not renew the header's saved observation lifetime.
        overview_dict = build_overview_dict(workspace_dict, snapshot_obj, provider_obj,
            as_of_ts=clock_fn(), demo_bool=demo_bool, include_finance_bool=False)
        selection_dict = {"level": level_str, "period": period_str, "unit": unit_str}
        if from_date_str is not None:
            selection_dict.update({"from": from_date_str, "to": to_date_str})

        def performance_url_str(**override_dict):
            return url_for("performance", **{**selection_dict, **override_dict})

        performance_page_dict.update(
            level_option_list=[{"label_str": label_str, "selected_bool": level_str == option_str,
                "url_str": performance_url_str(level=option_str)} for option_str, label_str in (("portfolio", "Portfolio"), ("pods", "Pods"))],
            period_option_list=[{"label_str": option_str, "selected_bool": from_date_str is None and period_str == option_str,
                "url_str": url_for("performance", level=level_str, period=option_str, unit=unit_str)} for option_str in ("1W", "MTD", "YTD", "All")],
            unit_option_list=[{"label_str": label_str, "selected_bool": unit_str == option_str,
                "url_str": performance_url_str(unit=option_str)} for option_str, label_str in (("usd", "$"), ("pct", "%"))],
            date_url_str=url_for("performance", level=level_str, unit=unit_str),
            csv_url_str=performance_url_str(download="csv", expected=report_dict["report_hash_str"]) if report_dict else "",
            pdf_url_str=performance_url_str(download="pdf", expected=report_dict["report_hash_str"]) if report_dict else "",
        )
        active_pod_set = {item_dict["pod_id_str"] for item_dict in overview_dict["pod_list"]}
        for row_dict in performance_page_dict["pod_row_list"]:
            row_dict["url_str"] = url_for("pod", pod_id_str=row_dict["pod_id_str"]) if row_dict["pod_id_str"] in active_pod_set else ""
        overview_dict.update(refresh_url_str=url_for("performance_refresh", **selection_dict), refresh_seconds_int=15)
        template_str = "_overview.html" if refresh_bool or request.headers.get("HX-Request") == "true" else "overview.html"
        return render_template(template_str, overview_dict=overview_dict, performance_page_dict=performance_page_dict)

    @flask_app_obj.get("/performance")
    def performance():
        return performance_response()

    @flask_app_obj.get("/performance/refresh")
    def performance_refresh():
        return performance_response(refresh_bool=True)

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
