"""Private-network, read-only client financial screens.

Access is limited by host/tailnet policy, without an application login.
Only server-configured paths/identities can select financial sources.
"""

from datetime import UTC, date, datetime, timedelta
from io import BytesIO
import json
from zipfile import ZipFile, ZIP_DEFLATED
from zoneinfo import ZoneInfo

from flask import Blueprint, Response, abort, current_app, g, redirect, render_template, request, url_for

from alpha.live.client_reporting import (
    BrokerReportingSnapshot, ClientReportingError, build_client_report_dict, load_broker_reporting_snapshot,
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
from alpha.live.dashboard_v3.client_charts import daily_history_list, nav_chart_dict
from alpha.live.dashboard_v3.local_workspace import (
    LocalReportingError, build_local_workspace_dict, local_financial_scope_complete_bool, validate_local_bindings_unchanged,
)


client_blueprint_obj = Blueprint("clients", __name__)
OPERATION_VIEW_SET = {"strategies", "exposure", "activity", "diagnostics"}


def local_workspace_bool():
    return current_app.config.get("client_registry_dict") is None and not current_app.config.get("client_reporting_config_path_str")


def _local_workspace_dict():
    if not hasattr(g, "local_workspace_dict"):
        g.local_workspace_dict = build_local_workspace_dict(
            current_app.config["data_provider_obj"], current_app.config["performance_db_path_str"],
            today_str=datetime.now(UTC).astimezone(ZoneInfo("America/New_York")).date().isoformat(),
        )
    return g.local_workspace_dict


@client_blueprint_obj.context_processor
def client_navigation_dict():
    # Carry the user's selection, not dates synthesized by another view's
    # default. Operations can include today; finalized finances still use D+1.
    return {"local_workspace_bool": local_workspace_bool(), "client_period_query_dict": {
        key_str: request.args[key_str] for key_str in ("from", "to", "window")
        if key_str in request.args
    }}


def _operations_dict(client_dict, as_of_ts):
    if local_workspace_bool():
        workspace_dict = _local_workspace_dict()
        result_dict = build_client_operations_dict(
            client_dict, workspace_dict["summary_dict"], as_of_ts=datetime.now(UTC),
            local_account_list=workspace_dict["operations_account_list"],
        )
        result_dict["source_error_str"] = workspace_dict["operations_error_str"]
        return result_dict
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
    if local_workspace_bool():
        # The selected log window is not an inferred funding/inception date.
        # Existing historical bindings plus current exact release identities
        # bound local events; no cross-account or unscoped event is admitted.
        current_list = _local_workspace_dict()["operations_account_list"]
        current_pair_set = {(account_dict["pod_id"], account_dict["account_route"]) for account_dict in current_list}
        account_list = [account_dict for account_dict in client_dict["accounts"]
            if (account_dict["pod_id"], account_dict["account_route"]) not in current_pair_set]
        account_list.extend({**account_dict, "effective_from": from_str} for account_dict in current_list)
        client_dict = {**client_dict, "accounts": account_list}
    return load_client_activity_dict(client_dict, current_app.config["data_provider_obj"],
        from_date_str=from_str, to_date_str=to_str, as_of_ts=datetime.now(UTC))


def _registry_dict():
    registry_dict = current_app.config.get("client_registry_dict")
    if registry_dict is not None:
        return validate_client_registry_dict(registry_dict)
    config_path_str = current_app.config.get("client_reporting_config_path_str")
    if not config_path_str:
        return {"schema_version": 1, "clients": [_local_workspace_dict()["client_dict"]]}
    return load_client_registry_dict(config_path_str)


def local_strategy_name_str(row_dict):
    """Display only: exact current local ownership; never relabel routing keys."""
    fallback_str = strategy_display_name_str(row_dict)
    if row_dict.get("mode_str") != "live":
        return fallback_str
    if local_workspace_bool():
        # Local release names already feed the projection; don't recursively
        # acquire a summary while the legacy advanced page is rendering one.
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
    if local_workspace_bool():
        if hasattr(g, "local_reporting_snapshot_obj"):
            return g.local_reporting_snapshot_obj
        workspace_dict = _local_workspace_dict()
        if workspace_dict["financial_error_str"]:
            return BrokerReportingSnapshot(unavailable_reason_str=workspace_dict["financial_error_str"])
        try:
            database_path_str = current_app.config["performance_db_path_str"]
            snapshot_obj = load_broker_reporting_snapshot(database_path_str,
                allowed_account_set={account_dict["account_route"] for account_dict in workspace_dict["valuation_account_list"]},
                query_name_str=client_dict["query_name"])
            validate_local_bindings_unchanged(workspace_dict, database_path_str)
        except (ClientReportingError, ValueError, OSError) as exception_obj:
            reason_str = str(exception_obj) if isinstance(exception_obj, LocalReportingError) else "Saved IBKR report failed validation. Check the latest Flex import before using these figures."
            snapshot_obj = BrokerReportingSnapshot(unavailable_reason_str=reason_str)
        g.local_reporting_snapshot_obj = snapshot_obj
        return snapshot_obj
    snapshot_fn = current_app.config.get("client_reporting_snapshot_fn")
    if snapshot_fn is not None:
        return snapshot_fn(client_dict["client_id"])
    database_path_str = client_dict.get("performance_db_path") or current_app.config["performance_db_path_str"]
    snapshot_obj = load_broker_reporting_snapshot(
        database_path_str,
        allowed_account_set={account_dict["account_route"] for account_dict in client_dict["accounts"]},
        query_name_str=client_dict["query_name"],
    )
    return snapshot_obj


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
    if operational_bool and local_workspace_bool() and not client_dict["accounts"]:
        start_str = (end_day_obj - timedelta(days=30)).isoformat()
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


@client_blueprint_obj.get("/clients")
def directory_route_fn():
    if local_workspace_bool():
        return redirect(url_for("clients.financial_route_fn", client_id_str="local", view_str="overview", **_query_dict()))
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
        local_source_unavailable_bool = False
        if local_workspace_bool() and view_str not in OPERATION_VIEW_SET:
            # Available broker NAV may precede the first strategy return. This
            # changes the selectable history, not inferred funding/entry dates.
            snapshot_obj = _snapshot_obj(client_dict)
            local_source_unavailable_bool = bool(snapshot_obj.unavailable_reason_str)
            broker_date_list = [row_obj.market_date_str for row_obj in snapshot_obj.row_tuple
                if row_obj.market_date_str <= period_max_date_str]
            if broker_date_list:
                client_dict["mandate_start_date"] = min(client_dict["mandate_start_date"], min(broker_date_list))
        from_str, to_str = _period_tuple(client_dict, as_of_ts, operational_bool=view_str in OPERATION_VIEW_SET)
        if any(date.fromisoformat(value_str).isoformat() != value_str for value_str in (from_str, to_str)):
            raise ClientReportingError("Use YYYY-MM-DD dates.")
        local_operations_bool = local_workspace_bool() and view_str in OPERATION_VIEW_SET
        if from_str > to_str or (not local_operations_bool and not local_source_unavailable_bool and from_str < client_dict["mandate_start_date"]) or to_str > period_max_date_str:
            raise ClientReportingError("Choose an ordered period within available reporting history and through today at most." if local_workspace_bool() else "Choose an ordered period within the client's mandate and through today at most.")
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
        snapshot_obj = _snapshot_obj(client_dict)
        if local_workspace_bool() and snapshot_obj.unavailable_reason_str:
            raise LocalReportingError(snapshot_obj.unavailable_reason_str)
        if current_app.config.get("demo_mode_bool"):
            from alpha.live.dashboard_v3.demo import build_demo_benchmark_snapshot
            benchmark_snapshot_obj = build_demo_benchmark_snapshot()
        else:
            benchmark_snapshot_obj = load_benchmark_snapshot(client_dict.get("benchmark"))
        report_dict = build_client_report_dict(
            client_dict, snapshot_obj,
            from_date_str=from_str, to_date_str=to_str, as_of_ts=as_of_ts,
            benchmark_snapshot_obj=benchmark_snapshot_obj,
            scope_complete_bool=local_financial_scope_complete_bool(_local_workspace_dict(), from_str, to_str) if local_workspace_bool() else True,
            valuation_account_list=_local_workspace_dict()["valuation_account_list"] if local_workspace_bool() else None,
        )
    except (ClientReportingError, ValueError, OSError) as exception_obj:
        # A financial-source failure must not remove current operations or the
        # selected client's navigation. Do not expose raw paths/XML in errors.
        error_str = "Financial evidence could not be read or validated. No financial figures are available; current operations remain separate."
        if isinstance(exception_obj, LocalReportingError):
            error_str = exception_obj.args[0]
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
        # memory; no SQL/file writes or trading calls.
        response_obj = Response(json.dumps(report_dict, indent=2, allow_nan=False), mimetype="application/json")
        response_obj.headers["Content-Disposition"] = f'attachment; filename="operator-report-{client_id_str}-{report_dict["report_hash_str"][:12]}.json"'
        return response_obj
    return_path_list = report_dict["return_path_list"]
    if not report_dict["client_twr_configured_bool"] and report_dict["twr_float"] is not None and len(report_dict["strategy_list"]) == 1:
        # Same verified one-account fallback as the headline and daily panel.
        return_path_list = report_dict["strategy_list"][0]["performance_dict"]["return_path_list"]
    return render_template(
        "client_financial.html", client_list=registry_dict["clients"], client_dict=client_dict,
        report_dict=report_dict, view_str=view_str, chart_dict=nav_chart_dict(report_dict["daily_book_list"]),
        client_return_chart_dict=nav_chart_dict(return_path_list, value_field_str="cumulative_return_float", unit_str="pct"),
        pnl_unavailable_str="IBKR cash-flow setup required" if not client_dict.get("nav_bridge") else "Incomplete IBKR data",
        return_unavailable_str="Portfolio return setup required" if not report_dict["client_twr_configured_bool"] and len(report_dict["strategy_list"]) > 1 else "Incomplete IBKR return data",
        investor_dict=build_investor_snapshot_dict(report_dict) if view_str == "report" else None,
        operations_dict=_operations_dict(client_dict, as_of_ts) if view_str == "overview" else None,
        activity_dict=_activity_dict(client_dict, from_str, to_str) if view_str == "overview" else None,
        period_max_date_str=period_max_date_str,
        financial_notice_list=_financial_notice_list(report_dict, client_dict),
        performance_chart_list=[nav_chart_dict(strategy_dict["performance_dict"]["return_path_list"], value_field_str="cumulative_return_float", unit_str="pct") for strategy_dict in report_dict["strategy_list"]] if view_str == "performance" else [],
        daily_scope_list=daily_history_list(report_dict) if view_str != "report" else [],
        comparison_list=[saved_comparison_dict(next(
            account_dict for account_dict in client_dict["accounts"]
            if (account_dict["pod_id"], account_dict["account_route"]) == (strategy_dict["pod_id_str"], strategy_dict["account_route_str"])
            and account_dict["effective_from"] <= strategy_dict["from_date_str"] <= (account_dict.get("effective_to") or "9999-12-31")
        ), from_date_str=from_str, to_date_str=to_str) for strategy_dict in report_dict["strategy_list"]] if view_str == "performance" else [],
    )


def _financial_notice_list(report_dict, client_dict):
    """Short visible causes; full diagnostic evidence remains in the report."""
    notice_list = list(report_dict.get("nav_issue_list", []))
    if local_workspace_bool() and not report_dict["scope_complete_bool"]:
        window_dict = {(account_dict["pod_id"], account_dict["account_route"]): account_dict for account_dict in client_dict["accounts"]}
        for identity_dict in _local_workspace_dict()["valuation_account_list"]:
            account_dict = window_dict.get((identity_dict["pod_id"], identity_dict["account_route"]))
            prefix_str = f"{identity_dict['pod_id']} / {identity_dict['account_route']}"
            if account_dict is None:
                notice_list.append(f"{prefix_str}: no verified strategy performance window yet. Account NAV is checked separately.")
            elif account_dict.get("effective_to") and account_dict["effective_to"] < report_dict["requested_to_date_str"]:
                notice_list.append(f"{prefix_str}: strategy history ends {account_dict['effective_to']}; this does not mean the account was closed.")
            elif account_dict["effective_from"] > report_dict["requested_from_date_str"]:
                notice_list.append(f"{prefix_str}: strategy history starts {account_dict['effective_from']}; earlier account NAV is checked separately.")
    if report_dict["pnl_float"] is None and not client_dict.get("nav_bridge"):
        notice_list.append("P&L setup: IBKR capital movements are not mapped. Check the saved Flex fields before configuring the mapping.")
    return list(dict.fromkeys(notice_list))
