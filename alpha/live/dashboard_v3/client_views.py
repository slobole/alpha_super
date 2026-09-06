"""Operator-authenticated, read-only client financial screens.

The app's protect_operator_access_fn gates every /clients request before this
blueprint runs, and fails closed when no operator credential is configured.
Only server-configured paths/identities can select financial sources.
"""

from datetime import UTC, date, datetime, timedelta
from io import BytesIO
import json
from zipfile import ZipFile, ZIP_DEFLATED
from zoneinfo import ZoneInfo

from flask import Blueprint, Response, abort, current_app, g, redirect, render_template, request, url_for

from alpha.live.client_reporting import (
    ClientReportingError, build_client_report_dict, load_broker_reporting_snapshot,
    load_client_registry_dict, validate_client_registry_dict,
)
from alpha.live.investor_report import build_investor_snapshot_dict, render_investor_pdf_bytes
from alpha.live.client_benchmark import load_benchmark_snapshot
from alpha.live.dashboard_v3.client_operations import (
    active_account_list, build_client_operations_dict, build_reference_exposure_list,
    load_client_activity_dict, load_operations_summary_dict,
)
from alpha.live.dashboard_v3.operator_tools import redact_diagnostic_value, strategy_display_name_str
from alpha.live.dashboard_v3.client_comparison import saved_comparison_dict


client_blueprint_obj = Blueprint("clients", __name__)
OPERATION_VIEW_SET = {"strategies", "exposure", "activity", "diagnostics"}


@client_blueprint_obj.context_processor
def client_navigation_dict():
    # Carry the user's selection, not dates synthesized by another view's
    # default. Operations can include today; finalized finances still use D+1.
    return {"client_period_query_dict": {
        key_str: request.args[key_str] for key_str in ("from", "to", "window")
        if key_str in request.args
    }}


def _operations_dict(client_dict, as_of_ts):
    summary_dict, error_str = {}, None
    if active_account_list(client_dict, as_of_ts):
        try:
            summary_dict = load_operations_summary_dict(client_dict, current_app.config["data_provider_obj"])
        except (OSError, ValueError, KeyError, TypeError):
            error_str = "Saved operations source could not be read or validated. No alternative client's source was used."
    try:
        # A local provider stamps its assessment during this read, after the
        # request's initial clock. Compare it to acquisition completion, not to
        # the earlier financial selection clock (which would call it future).
        result_dict = build_client_operations_dict(client_dict, summary_dict, as_of_ts=datetime.now(UTC))
    except (ValueError, TypeError, AttributeError):
        result_dict = build_client_operations_dict(client_dict, {}, as_of_ts=as_of_ts)
        error_str = "Saved operations source is invalid. Review the source configuration."
    result_dict["source_error_str"] = error_str
    return result_dict


def _query_dict():
    if set(request.args) - {"from", "to", "window", "download", "expected"} or any(len(request.args.getlist(key_str)) != 1 for key_str in request.args):
        abort(400)
    return request.args.to_dict()


def _activity_dict(client_dict, from_str, to_str):
    return load_client_activity_dict(client_dict, current_app.config["data_provider_obj"],
        from_date_str=from_str, to_date_str=to_str, as_of_ts=datetime.now(UTC))


def _registry_dict():
    registry_dict = current_app.config.get("client_registry_dict")
    if registry_dict is not None:
        return validate_client_registry_dict(registry_dict)
    config_path_str = current_app.config.get("client_reporting_config_path_str")
    if not config_path_str:
        raise ClientReportingError("Set up the client registry. Financial ownership is not inferred from enabled strategies.")
    return load_client_registry_dict(config_path_str)


def local_strategy_name_str(row_dict):
    """Display only: exact current local ownership; never relabel routing keys."""
    fallback_str = strategy_display_name_str(row_dict)
    if row_dict.get("mode_str") != "live":
        return fallback_str
    if not hasattr(g, "local_strategy_name_dict"):
        name_dict = {}
        try:
            registry_dict = _registry_dict()
            as_of_ts = datetime.now(UTC)
            for client_dict in registry_dict["clients"]:
                if client_dict.get("operations_source") != "local":
                    continue
                for account_dict in active_account_list(client_dict, as_of_ts):
                    key_tuple = (account_dict["pod_id"], account_dict["account_route"])
                    name_dict.setdefault(key_tuple, []).append(account_dict["display_name"])
        except (OSError, ValueError, TypeError, KeyError):
            name_dict = {}
        g.local_strategy_name_dict = name_dict
    name_list = g.local_strategy_name_dict.get((row_dict.get("pod_id_str"), row_dict.get("account_route_str")), [])
    return name_list[0] if len(name_list) == 1 else fallback_str


def _snapshot_obj(client_dict):
    snapshot_fn = current_app.config.get("client_reporting_snapshot_fn")
    if snapshot_fn is not None:
        return snapshot_fn(client_dict["client_id"])
    database_path_str = client_dict.get("performance_db_path") or current_app.config["performance_db_path_str"]
    return load_broker_reporting_snapshot(
        database_path_str,
        allowed_account_set={account_dict["account_route"] for account_dict in client_dict["accounts"]},
        query_name_str=client_dict["query_name"],
    )


def _period_tuple(client_dict, as_of_ts, *, operational_bool=False):
    from_str, to_str = request.args.get("from"), request.args.get("to")
    if bool(from_str) != bool(to_str):
        raise ClientReportingError("Choose both the start and end date.")
    if from_str and request.args.get("window"):
        raise ClientReportingError("Use either exact dates or a preset, not both.")
    if from_str:
        return from_str, to_str
    market_day_obj = as_of_ts.astimezone(ZoneInfo("America/New_York")).date()
    # Operational evidence is available today; financial Activity Flex is D+1.
    end_day_obj = market_day_obj if operational_bool else market_day_obj - timedelta(days=1)
    window_str = request.args.get("window", "all")
    if window_str not in {"all", "mtd", "ytd", "1w"}:
        raise ClientReportingError("Unknown reporting period.")
    start_str = client_dict["mandate_start_date"]
    if window_str == "mtd":
        start_str = max(start_str, market_day_obj.replace(day=1).isoformat())
    elif window_str == "ytd":
        start_str = max(start_str, market_day_obj.replace(month=1, day=1).isoformat())
    elif window_str == "1w":
        start_str = max(start_str, (end_day_obj - timedelta(days=6)).isoformat())
    if start_str > end_day_obj.isoformat():
        # Keep a new mandate/current MTD or YTD accessible. The accounting
        # builder explicitly withholds today's unfinalized financial figures.
        return start_str, start_str
    return start_str, end_day_obj.isoformat()


def nav_chart_dict(daily_list):
    nav_list = [row_dict["nav_float"] for row_dict in daily_list if row_dict["nav_float"] is not None]
    if not nav_list:
        return None
    low_float, high_float = min(nav_list), max(nav_list)
    span_float = high_float - low_float
    segment_list, current_list = [], []
    for index_int, daily_dict in enumerate(daily_list):
        value_float = daily_dict["nav_float"]
        if value_float is None:
            if current_list:
                segment_list.append(" ".join(current_list))
                current_list = []
            continue
        horizontal_float = 20 + index_int / max(1, len(daily_list) - 1) * 860
        vertical_float = 110 if span_float == 0 else 195 - (value_float - low_float) / span_float * 170
        current_list.append(f"{horizontal_float:.2f},{vertical_float:.2f}")
    if current_list:
        segment_list.append(" ".join(current_list))
    return {"segment_list": segment_list, "min_float": low_float, "max_float": high_float}


@client_blueprint_obj.get("/clients")
def directory_route_fn():
    try:
        registry_dict = _registry_dict()
    except ClientReportingError as exception_obj:
        return render_template("client_directory.html", client_list=[], error_str=str(exception_obj)), 200
    return render_template("client_directory.html", client_list=registry_dict["clients"], error_str=None)


@client_blueprint_obj.get("/clients/<client_id_str>")
def client_route_fn(client_id_str):
    return redirect(url_for("clients.financial_route_fn", client_id_str=client_id_str, view_str="overview", **_query_dict()))


@client_blueprint_obj.get("/clients/<client_id_str>/<view_str>")
def financial_route_fn(client_id_str, view_str):
    _query_dict()
    if view_str not in {"overview", "performance", "report"} | OPERATION_VIEW_SET:
        abort(404)
    client_dict = None
    try:
        registry_dict = _registry_dict()
        client_dict = next((candidate_dict for candidate_dict in registry_dict["clients"] if candidate_dict["client_id"] == client_id_str), None)
        if client_dict is None:
            abort(404)
        as_of_ts = datetime.now(UTC)
        period_max_date_str = as_of_ts.astimezone(ZoneInfo("America/New_York")).date().isoformat()
        from_str, to_str = _period_tuple(client_dict, as_of_ts, operational_bool=view_str in OPERATION_VIEW_SET)
        if any(date.fromisoformat(value_str).isoformat() != value_str for value_str in (from_str, to_str)):
            raise ClientReportingError("Use YYYY-MM-DD dates.")
        if from_str > to_str or from_str < client_dict["mandate_start_date"] or to_str > period_max_date_str:
            raise ClientReportingError("Choose an ordered period within the client's mandate and through today at most.")
        if view_str in OPERATION_VIEW_SET:
            operations_dict = _operations_dict(client_dict, as_of_ts)
            if request.args.get("download"):
                if view_str != "diagnostics" or request.args["download"] != "json":
                    abort(400)
                response_obj = Response(json.dumps(redact_diagnostic_value(operations_dict), allow_nan=False, indent=2), mimetype="application/json")
                response_obj.headers["Content-Disposition"] = f'attachment; filename="operator-status-{client_id_str}.json"'
                return response_obj
            return render_template(
                "client_operations.html", client_list=registry_dict["clients"], client_dict=client_dict, view_str=view_str,
                report_dict={"requested_from_date_str": from_str, "requested_to_date_str": to_str},
                operations_dict=operations_dict, exposure_list=build_reference_exposure_list(operations_dict),
                activity_dict=_activity_dict(client_dict, from_str, to_str) if view_str == "activity" else None,
                period_max_date_str=period_max_date_str,
            )
    except (ClientReportingError, ValueError) as exception_obj:
        if client_dict is not None:
            if request.args.get("download"):
                return Response("Invalid reporting period. Correct the dates in the client view before exporting.", status=400)
            return render_template(
                "client_period_error.html", client_list=registry_dict["clients"], client_dict=client_dict, view_str=view_str,
                report_dict={"requested_from_date_str": request.args.get("from", ""), "requested_to_date_str": request.args.get("to", "")},
                period_max_date_str=period_max_date_str, error_str=str(exception_obj),
            ), 400
        return render_template("client_directory.html", client_list=[], error_str=str(exception_obj)), 400
    try:
        if current_app.config.get("demo_mode_bool"):
            from alpha.live.dashboard_v3.demo import build_demo_benchmark_snapshot
            benchmark_snapshot_obj = build_demo_benchmark_snapshot()
        else:
            benchmark_snapshot_obj = load_benchmark_snapshot(client_dict.get("benchmark"))
        report_dict = build_client_report_dict(
            client_dict, _snapshot_obj(client_dict),
            from_date_str=from_str, to_date_str=to_str, as_of_ts=as_of_ts,
            benchmark_snapshot_obj=benchmark_snapshot_obj,
        )
    except (ClientReportingError, ValueError, OSError):
        # A financial-source failure must not remove current operations or the
        # selected client's navigation. Do not expose raw paths/XML in errors.
        error_str = "Financial evidence could not be read or validated. No financial figures are available; current operations remain separate."
        if request.args.get("download"):
            return Response(error_str, status=503)
        return render_template(
            "client_unavailable.html", client_list=registry_dict["clients"], client_dict=client_dict, view_str=view_str,
            report_dict={"requested_from_date_str": from_str, "requested_to_date_str": to_str},
            error_str=error_str, operations_dict=_operations_dict(client_dict, as_of_ts) if view_str == "overview" else None,
            activity_dict=_activity_dict(client_dict, from_str, to_str) if view_str == "overview" else None,
        )
    if request.args.get("download"):
        download_str = request.args["download"]
        if download_str not in {"json", "pdf", "bundle"}:
            abort(400)
        if download_str in {"pdf", "bundle"}:
            if view_str != "report":
                abort(400)
            if request.args.get("expected") != report_dict["report_hash_str"]:
                return Response("Reporting evidence changed or preview confirmation is missing. Refresh the report preview before exporting.", status=409)
            investor_dict = build_investor_snapshot_dict(report_dict)
            pdf_bytes = render_investor_pdf_bytes(investor_dict)
            filename_str = f'investment-report-{investor_dict["document_hash_str"][:16]}'
            if download_str == "bundle":
                archive_obj = BytesIO()
                with ZipFile(archive_obj, "w", ZIP_DEFLATED) as zip_obj:
                    zip_obj.writestr(filename_str + ".pdf", pdf_bytes)
                    zip_obj.writestr(filename_str + ".json", json.dumps(investor_dict, indent=2, allow_nan=False))
                response_obj = Response(archive_obj.getvalue(), mimetype="application/zip")
                response_obj.headers["Content-Disposition"] = f'attachment; filename="{filename_str}.zip"'
            else:
                response_obj = Response(pdf_bytes, mimetype="application/pdf")
                response_obj.headers["Content-Disposition"] = f'attachment; filename="{filename_str}.pdf"'
            return response_obj
        # Explicit operator evidence, not an investor statement. Generated in
        # memory after authentication; no SQL/file writes or trading calls.
        response_obj = Response(json.dumps(report_dict, indent=2, allow_nan=False), mimetype="application/json")
        response_obj.headers["Content-Disposition"] = f'attachment; filename="operator-report-{client_id_str}-{report_dict["report_hash_str"][:12]}.json"'
        return response_obj
    return render_template(
        "client_financial.html", client_list=registry_dict["clients"], client_dict=client_dict,
        report_dict=report_dict, view_str=view_str, chart_dict=nav_chart_dict(report_dict["daily_book_list"]),
        client_return_chart_dict=nav_chart_dict([{"nav_float": point_dict["cumulative_return_float"]} for point_dict in report_dict["return_path_list"]]),
        investor_dict=build_investor_snapshot_dict(report_dict) if view_str == "report" else None,
        operations_dict=_operations_dict(client_dict, as_of_ts) if view_str == "overview" else None,
        activity_dict=_activity_dict(client_dict, from_str, to_str) if view_str == "overview" else None,
        period_max_date_str=period_max_date_str,
        performance_chart_list=[nav_chart_dict([{"nav_float": point_dict["cumulative_return_float"]} for point_dict in strategy_dict["performance_dict"]["return_path_list"]]) for strategy_dict in report_dict["strategy_list"]] if view_str == "performance" else [],
        comparison_list=[saved_comparison_dict(next(
            account_dict for account_dict in client_dict["accounts"]
            if (account_dict["pod_id"], account_dict["account_route"]) == (strategy_dict["pod_id_str"], strategy_dict["account_route_str"])
            and account_dict["effective_from"] <= strategy_dict["from_date_str"] <= (account_dict.get("effective_to") or "9999-12-31")
        ), from_date_str=from_str, to_date_str=to_str) for strategy_dict in report_dict["strategy_list"]] if view_str == "performance" else [],
    )
