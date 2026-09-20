"""One LIVE Pod's saved account facts; no trading or new accounting model."""

from copy import deepcopy
from datetime import datetime
import math
from zoneinfo import ZoneInfo

from alpha.live.client_reporting import ClientReportingError, build_client_report_dict, validate_client_registry_dict
from alpha.live.dashboard_v3.client_cash import load_portfolio_cash_list
from alpha.live.dashboard_v3.client_financial_display import financial_dates_dict
from alpha.live.dashboard_v3.client_operations import build_client_operations_dict
from alpha.live.dashboard_v4.finance import (
    PERIOD_TUPLE, _chart_dict, _empty_chart_dict, _money_str, _months_before_date,
    _percent_str, _tone_str,
)


def _saved_time_str(timestamp_str, as_of_ts):
    try:
        timestamp_ts = datetime.fromisoformat(timestamp_str)
        if timestamp_ts.tzinfo is not None and timestamp_ts <= as_of_ts:
            return timestamp_ts.astimezone(ZoneInfo("America/New_York")).strftime("%Y-%m-%d %H:%M:%S")
    except (TypeError, ValueError):
        pass
    return "—"


def _finite_bool(value_obj):
    return type(value_obj) in {float, int} and math.isfinite(value_obj)


def _scope_tuple(workspace_dict, pod_id_str):
    """Validate the full owner mapping before narrowing the account projection."""
    client_dict = validate_client_registry_dict(
        {"schema_version": 1, "clients": [deepcopy(workspace_dict["client_dict"])]},
        allow_empty_periods_bool=True)["clients"][0]
    operation_list = workspace_dict.get("operations_account_list", client_dict["accounts"])
    selected_list = [account_dict for account_dict in operation_list if account_dict.get("pod_id") == pod_id_str]
    if len(selected_list) != 1:
        raise ValueError("Pod ownership is missing or ambiguous.")
    identity_dict = selected_list[0]
    route_str = identity_dict["account_route"]
    # Narrowing must not hide a route reassignment or an unsupported shared account.
    for account_list in (operation_list, client_dict["accounts"], workspace_dict.get("valuation_account_list", [])):
        if sum(account_dict.get("pod_id") == pod_id_str for account_dict in account_list) > 1:
            raise ValueError("Duplicate Pod ownership.")
        for account_dict in account_list:
            if (account_dict.get("pod_id") == pod_id_str) != (account_dict.get("account_route") == route_str):
                raise ValueError("Conflicting Pod/account ownership.")
    source_list = workspace_dict.get("summary_dict", {}).get("pod_row_dict_list", [])
    if not isinstance(source_list, list) or any(not isinstance(row_dict, dict) for row_dict in source_list):
        raise ValueError("Invalid saved Pod identity list.")
    if any(row_dict.get("account_route_str") == route_str and row_dict.get("pod_id_str") != pod_id_str for row_dict in source_list):
        raise ValueError("The source maps multiple Pods to one account.")
    row_list = [row_dict for row_dict in source_list
        if row_dict.get("pod_id_str") == pod_id_str]
    if len(row_list) != 1 or (row_list[0].get("mode_str"), row_list[0].get("account_route_str")) != ("live", route_str):
        raise ValueError("Current LIVE identity is not established.")
    account_list = [account_dict for account_dict in client_dict["accounts"] if account_dict["pod_id"] == pod_id_str]
    if len(account_list) > 1:
        raise ValueError("Multiple historical periods need an explicit selection.")
    client_dict["accounts"] = account_list
    if account_list:
        client_dict["mandate_start_date"] = max(client_dict["mandate_start_date"], account_list[0]["effective_from"])
    return client_dict, identity_dict


def _positions_dict(evidence_dict, as_of_ts):
    position_time_str = _saved_time_str(evidence_dict.get("latest_pod_state_timestamp_str"), as_of_ts)
    result_dict = {"position_list": [], "position_asof_str": position_time_str,
        "positions_basis_str": "Saved positions", "positions_available_bool": position_time_str != "—"}
    if not result_dict["positions_available_bool"]:
        return result_dict
    position_list = evidence_dict.get("position_exposure_dict_list") or []
    symbol_list = [position_dict.get("asset_str") for position_dict in position_list]
    if len(symbol_list) != len(set(symbol_list)) or any(not isinstance(symbol_str, str) or not symbol_str for symbol_str in symbol_list):
        result_dict["positions_available_bool"] = False
        return result_dict
    for position_dict in position_list:
        share_float = position_dict.get("share_float")
        if _finite_bool(share_float) and share_float == 0:
            continue
        result_dict["position_list"].append({"symbol_str": position_dict["asset_str"],
            "shares_str": f"{share_float:,.6f}".rstrip("0").rstrip(".") if _finite_bool(share_float) else "—"})
    return result_dict


def build_pod_finance_dict(workspace_dict, snapshot_obj, provider_obj, *, pod_id_str, as_of_ts, period_str="3M"):
    """Use official account TWR and the canonical dollar bridge, never NAV delta.

    The complete acquired mapping is checked before selecting one account. A
    missing history boundary can expose finalized NAV but cannot unlock returns.
    Holdings show saved quantities at their own timestamp, independently of the
    financial close. Submission reference prices cannot supply position values.
    """
    if period_str not in PERIOD_TUPLE or as_of_ts.tzinfo is None:
        raise ValueError("Choose a supported period and a timezone-aware clock.")
    result_dict = {"tile_list": [{"label_str": label_str, "value_str": "—", "detail_str": "Data unavailable", "tone_str": ""}
        for label_str in ("Value", "Day", "Month", "Since start")],
        "chart_dict": _empty_chart_dict(), "money_asof_str": "Financial data unavailable",
        "cash_str": "—", "cash_asof_str": "—", "position_list": [], "position_asof_str": "—",
        "positions_basis_str": "Saved positions", "positions_available_bool": False,
        "reference_summary_str": "Live vs backtest unavailable", "financial_error_str": "", "delayed_bool": True}
    try:
        client_dict, identity_dict = _scope_tuple(workspace_dict, pod_id_str)
    except (ClientReportingError, ValueError, TypeError, KeyError):
        result_dict["financial_error_str"] = "Saved Pod/account identity could not be verified."
        return result_dict
    try:
        operations_dict = build_client_operations_dict(workspace_dict["client_dict"], workspace_dict.get("summary_dict", {}),
            as_of_ts=as_of_ts, local_account_list=workspace_dict.get("operations_account_list"))
        strategy_list = [strategy_dict for strategy_dict in operations_dict["strategy_list"]
            if strategy_dict["pod_id_str"] == pod_id_str and strategy_dict["account_route_str"] == identity_dict["account_route"]
            and strategy_dict["matched_bool"]]
        if len(strategy_list) == 1:
            result_dict.update(_positions_dict(strategy_list[0]["evidence_dict"], as_of_ts))
    except (ValueError, TypeError, KeyError):
        operations_dict = {"strategy_list": []}
    if workspace_dict.get("financial_error_str") or snapshot_obj.unavailable_reason_str:
        result_dict["financial_error_str"] = "Saved financial data could not be verified."
        return result_dict
    today_obj = as_of_ts.astimezone(ZoneInfo("America/New_York")).date()
    account_list = client_dict["accounts"]
    if not account_list:
        observed_list = [row_obj.market_date_str for row_obj in snapshot_obj.row_tuple
            if row_obj.account_route_str == identity_dict["account_route"] and row_obj.market_date_str < today_obj.isoformat()]
        client_dict["mandate_start_date"] = min(observed_list, default=today_obj.isoformat())
    try:
        dates_dict = financial_dates_dict(client_dict, snapshot_obj, as_of_ts=as_of_ts, valuation_account_list=[identity_dict])
        closing_str = dates_dict["latest_complete_str"]
        if not closing_str:
            raise ValueError("No finalized account value.")
        report_cache_dict = {}

        def report_dict(from_str):
            from_str = max(from_str, client_dict["mandate_start_date"])
            if from_str > closing_str:
                return None
            if from_str not in report_cache_dict:
                report_cache_dict[from_str] = build_client_report_dict(client_dict, snapshot_obj,
                    from_date_str=from_str, to_date_str=closing_str, as_of_ts=as_of_ts,
                    scope_complete_bool=bool(account_list), valuation_account_list=[identity_dict])
            return report_cache_dict[from_str]

        day_dict = report_dict(closing_str)
        if day_dict is None:
            raise ValueError("No owned reporting interval.")
        month_dict = report_dict(today_obj.replace(day=1).isoformat())
        start_dict = report_dict(client_dict["mandate_start_date"])
        chart_start_str = client_dict["mandate_start_date"] if period_str == "All" else (
            today_obj.replace(month=1, day=1).isoformat() if period_str == "YTD" else
            _months_before_date(today_obj, 1 if period_str == "1M" else 3).isoformat())
        chart_report_dict = report_dict(chart_start_str)
        result_dict["tile_list"][0].update(value_str=_money_str(day_dict["closing_nav_float"]), detail_str="Close " + closing_str)
        for index_int, period_dict in enumerate((day_dict, month_dict, start_dict), start=1):
            if period_dict is None:
                result_dict["tile_list"][index_int]["detail_str"] = "No data in this period"
                continue
            # Official account returns remain independent of book-level TWR policy.
            strategy_list = period_dict["strategy_list"]
            account_dict = strategy_list[0] if len(strategy_list) == 1 and strategy_list[0]["to_date_str"] == closing_str else {}
            return_float, pnl_float = account_dict.get("twr_float"), account_dict.get("pnl_float")
            result_dict["tile_list"][index_int].update(
                value_str=_money_str(pnl_float, signed_bool=True) if index_int == 1 else _percent_str(return_float, signed_bool=True),
                detail_str=_percent_str(return_float, signed_bool=True) if index_int == 1 else (
                    "From " + account_dict["from_date_str"] if index_int == 3 and account_dict else _money_str(pnl_float, signed_bool=True)),
                tone_str=_tone_str(pnl_float if index_int == 1 else return_float))
        chart_account_list = chart_report_dict["strategy_list"] if chart_report_dict else []
        chart_account_dict = chart_account_list[0] if len(chart_account_list) == 1 and chart_account_list[0]["to_date_str"] == closing_str else {}
        return_path_list = chart_account_dict["performance_dict"]["return_path_list"] if chart_account_dict.get("twr_float") is not None else []
        chart_dict = _chart_dict(return_path_list)
        if chart_dict["available_bool"]:
            chart_dict.update(source_str="Demo account TWR" if client_dict.get("is_demo") else "IBKR account TWR",
                detail_str="Cumulative account return for the selected period, adjusted for cash flows.")
        else:
            chart_dict["detail_str"] = "Complete, verified account return history is required."
        result_dict.update(money_asof_str=("Demo · " if client_dict.get("is_demo") else "") + "Close " + closing_str
            + (" · Data delayed" if dates_dict["delayed_bool"] else ""), delayed_bool=dates_dict["delayed_bool"],
            chart_dict=chart_dict)
    except (ClientReportingError, ValueError, TypeError, KeyError, OSError):
        result_dict["financial_error_str"] = "Saved financial data could not be verified."
        return result_dict
    try:
        cash_list = load_portfolio_cash_list(client_dict, day_dict, provider_obj, operations_dict, as_of_ts=as_of_ts)
        if len(cash_list) == 1 and _finite_bool(cash_list[0].get("cash_float")):
            result_dict.update(cash_str=_money_str(cash_list[0]["cash_float"]), cash_asof_str="Close " + closing_str)
    except (ValueError, TypeError, KeyError, OSError, AttributeError):
        pass
    return result_dict
