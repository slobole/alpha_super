"""One LIVE Pod's saved account facts; no trading or new accounting model."""

from copy import deepcopy
from datetime import datetime
from decimal import Decimal
import math
from zoneinfo import ZoneInfo

from alpha.live.client_reporting import ClientReportingError, build_client_report_dict, validate_client_registry_dict
from alpha.live.dashboard_v3.client_cash import load_portfolio_cash_list
from alpha.live.dashboard_v3.client_financial_display import financial_dates_dict
from alpha.live.dashboard_v3.client_operations import build_client_operations_dict
from alpha.live.dashboard_v4.finance import (
    PERIOD_TUPLE, POD_COLOR_TUPLE, _chart_dict, _empty_chart_dict, _money_str, _months_before_date,
    _percent_str, _tone_str,
)
from alpha.live.dashboard_v4.pod_allocation import build_pod_allocation_dict
from alpha.live.dashboard_v4.pod_holdings import load_close_holdings_dict
from alpha.live.dashboard_v4.pod_broker_holdings import load_broker_holdings_dict
from alpha.live.dashboard_v4.positions_data import observed_timestamp_ts, validated_position_map_dict


def _saved_time_str(timestamp_str, as_of_ts):
    try:
        timestamp_ts = datetime.fromisoformat(timestamp_str)
        if timestamp_ts.tzinfo is not None and timestamp_ts <= as_of_ts:
            return timestamp_ts.astimezone(ZoneInfo("America/New_York")).strftime("%Y-%m-%d %H:%M:%S")
    except (TypeError, ValueError):
        pass
    return "—"


def _finite_bool(value_obj):
    try:
        return type(value_obj) in {float, int} and math.isfinite(value_obj)
    except OverflowError:
        return False


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
        "positions_stamp_dict": {"label_str": position_time_str + " ET" if position_time_str != "—" else "—",
            "title_str": "Saved positions · " + position_time_str + " ET" if position_time_str != "—" else "Saved positions time unavailable"},
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


def _build_report_finance_dict(workspace_dict, snapshot_obj, provider_obj, *, pod_id_str, as_of_ts, period_str="3M", performance_db_path_str=None):
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
        "positions_stamp_dict": {"label_str": "—", "title_str": "Saved positions time unavailable"},
        "cash_stamp_dict": {"label_str": "—", "title_str": "Cash close unavailable"},
        "holdings_allocation_dict": build_pod_allocation_dict({}, color_str=POD_COLOR_TUPLE[0]),
        "positions_basis_str": "Saved positions", "positions_available_bool": False,
        "reference_summary_str": "Live vs backtest unavailable", "financial_error_str": "", "delayed_bool": True}
    try:
        client_dict, identity_dict = _scope_tuple(workspace_dict, pod_id_str)
    except (ClientReportingError, ValueError, TypeError, KeyError):
        result_dict["financial_error_str"] = "Saved Pod/account identity could not be verified."
        return result_dict
    position_evidence_dict = {}
    try:
        operations_dict = build_client_operations_dict(workspace_dict["client_dict"], workspace_dict.get("summary_dict", {}),
            as_of_ts=as_of_ts, local_account_list=workspace_dict.get("operations_account_list"))
        strategy_list = [strategy_dict for strategy_dict in operations_dict["strategy_list"]
            if strategy_dict["pod_id_str"] == pod_id_str and strategy_dict["account_route_str"] == identity_dict["account_route"]
            and strategy_dict["matched_bool"]]
        if len(strategy_list) == 1:
            position_evidence_dict = strategy_list[0]["evidence_dict"]
            result_dict.update(_positions_dict(position_evidence_dict, as_of_ts))
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
            # A shared date does not prove these positions and cash came from one snapshot.
            result_dict.update(cash_str=_money_str(cash_list[0]["cash_float"]), cash_asof_str="Close " + closing_str,
                cash_stamp_dict={"label_str": closing_str + " close",
                    "title_str": ("Demo · " if client_dict.get("is_demo") else "")
                        + "Cash · broker end-of-day · " + closing_str
                        + (" · Data delayed" if dates_dict["delayed_bool"] else "")})
            nav_row_list = [row_obj for row_obj in snapshot_obj.row_tuple
                if row_obj.account_route_str == identity_dict["account_route"] and row_obj.market_date_str == closing_str]
            if len(nav_row_list) == 1:
                if client_dict.get("is_demo") and hasattr(provider_obj, "get_close_holdings_dict"):
                    holdings_dict = provider_obj.get_close_holdings_dict(nav_row_list[0],
                        query_name_str=client_dict["query_name"], as_of_ts=as_of_ts)
                elif performance_db_path_str:
                    holdings_dict = load_close_holdings_dict(performance_db_path_str, nav_row_list[0],
                        query_name_str=client_dict["query_name"], as_of_ts=as_of_ts)
                else:
                    holdings_dict = {"available_bool": False, "reason_str": "Closing position values unavailable"}
                # *** CRITICAL *** display time basis: quantities come from the
                # same Flex close as values. Never multiply today's shares by a
                # prior close price; EOD cash must also belong to closing_str.
                holdings_dict.update(nav_float=float(nav_row_list[0].closing_nav_decimal), cash_float=cash_list[0]["cash_float"])
                current_time_str = result_dict["position_asof_str"]
                current_position_list = position_evidence_dict.get("position_exposure_dict_list") or []
                raw_position_list = next(row_dict for row_dict in workspace_dict["summary_dict"]["pod_row_dict_list"]
                    if row_dict["pod_id_str"] == pod_id_str).get("position_exposure_dict_list")
                holdings_note_str = ""
                if (holdings_dict.get("available_bool") and result_dict["positions_available_bool"]
                        and isinstance(raw_position_list, list) and raw_position_list == current_position_list
                        and all(_finite_bool(item_dict.get("share_float")) for item_dict in current_position_list)):
                    current_map_dict = {item_dict["asset_str"]: item_dict["share_float"] for item_dict in current_position_list if item_dict["share_float"] != 0}
                    close_map_dict = {item_dict["symbol_str"]: item_dict["shares_float"]
                        for item_dict in holdings_dict["position_list"] if item_dict["shares_float"] != 0}
                    if current_map_dict != close_map_dict:
                        holdings_dict["holdings_changed_bool"] = current_time_str[:10] > closing_str
                        holdings_note_str = "Holdings changed since this close." if holdings_dict["holdings_changed_bool"] else "Saved holdings differ from this close."
                color_account_list = workspace_dict.get("valuation_account_list") or workspace_dict["client_dict"]["accounts"]
                color_index_int = next((index_int for index_int, account_dict in enumerate(color_account_list)
                    if account_dict["pod_id"] == pod_id_str), 0)
                result_dict["holdings_allocation_dict"] = build_pod_allocation_dict(holdings_dict,
                    color_str=POD_COLOR_TUPLE[color_index_int % len(POD_COLOR_TUPLE)])
                result_dict["holdings_allocation_dict"].update(holdings_note_str=holdings_note_str,
                    comparison_basis_str="Compared with saved positions at " + current_time_str + " ET")
    except (ValueError, TypeError, KeyError, OSError, AttributeError):
        pass
    return result_dict


def build_pod_finance_dict(workspace_dict, snapshot_obj, provider_obj, *, pod_id_str, as_of_ts, period_str="3M", performance_db_path_str=None):
    """Keep reported money intact; display a self-contained saved IBKR portfolio.

    Allocation weights are saved IBKR position value / (sum(values) + saved
    cash), not reported NAV. Observation time is explicit; no prices are fetched
    and these operational marks never feed the official account return/P&L tiles.
    """
    result_dict = _build_report_finance_dict(workspace_dict, snapshot_obj, provider_obj,
        pod_id_str=pod_id_str, as_of_ts=as_of_ts, period_str=period_str,
        performance_db_path_str=performance_db_path_str)
    if (result_dict["holdings_allocation_dict"]["available_bool"]
            or workspace_dict.get("client_dict", {}).get("is_demo")
            or workspace_dict.get("client_dict", {}).get("operations_source") != "local"):
        return result_dict
    try:
        client_dict, identity_dict = _scope_tuple(workspace_dict, pod_id_str)
        if client_dict["base_currency"] != "USD":
            return result_dict
        target_obj = provider_obj.get_target_for_pod(pod_id_str)
        if target_obj is None or (target_obj.release_obj.pod_id_str, target_obj.release_obj.account_route_str) != (
                pod_id_str, identity_dict["account_route"]):
            return result_dict
        holdings_dict = load_broker_holdings_dict(target_obj, as_of_ts=as_of_ts)
        if not holdings_dict["available_bool"]:
            result_dict["holdings_allocation_dict"]["reason_str"] = holdings_dict["reason_str"]
            return result_dict
        timestamp_ts = observed_timestamp_ts(holdings_dict["observed_timestamp_str"], as_of_ts)
        observed_time_str = _saved_time_str(holdings_dict["observed_timestamp_str"], as_of_ts)
        # *** CRITICAL *** observation boundary: all shares, values and cash
        # come from this payload. Never join newer cache quantities to old marks.
        position_list, cash_float = holdings_dict["position_list"], holdings_dict["cash_float"]
        position_map_dict = {row_dict["symbol_str"]: row_dict["shares_float"] for row_dict in position_list}
        total_decimal = sum((Decimal(str(row_dict["value_float"])) for row_dict in position_list), Decimal(str(cash_float)))
        color_account_list = workspace_dict.get("valuation_account_list") or client_dict["accounts"]
        color_index_int = next((index_int for index_int, account_dict in enumerate(color_account_list)
            if account_dict["pod_id"] == pod_id_str), 0)
        allocation_dict = build_pod_allocation_dict({"available_bool": True,
            "close_date_str": observed_time_str[:10], "cash_float": cash_float,
            "nav_float": float(total_decimal), "position_list": position_list},
            color_str=POD_COLOR_TUPLE[color_index_int % len(POD_COLOR_TUPLE)])
        allocation_dict.update(broker_observation_bool=True, source_str="IBKR portfolio",
            observed_timestamp_str=timestamp_ts.isoformat(), observed_time_str=observed_time_str + " ET",
            basis_detail_str="Saved IBKR portfolio values and cash at this observation. "
                "Weights use holdings value plus cash; account NAV and returns remain unchanged.",
            holdings_note_str="", comparison_basis_str="",
            broker_nav_float=holdings_dict["broker_nav_float"], holdings_total_float=float(total_decimal))
        raw_row_dict = next(row_dict for row_dict in workspace_dict["summary_dict"]["pod_row_dict_list"]
            if row_dict["pod_id_str"] == pod_id_str)
        raw_position_list = raw_row_dict.get("position_exposure_dict_list")
        if isinstance(raw_position_list, list):
            current_time_str = _saved_time_str(raw_row_dict.get("latest_pod_state_timestamp_str"), as_of_ts)
            if (current_time_str != "—" and observed_timestamp_ts(raw_row_dict["latest_pod_state_timestamp_str"], as_of_ts) > timestamp_ts
                    and all(isinstance(row_dict, dict) and _finite_bool(row_dict.get("share_float")) for row_dict in raw_position_list)):
                try:
                    current_map_dict = validated_position_map_dict({row_dict["asset_str"]: row_dict["share_float"] for row_dict in raw_position_list})
                except (ValueError, KeyError, TypeError):
                    current_map_dict = None
                if current_map_dict is not None and len(current_map_dict) == len(raw_position_list) and {
                        symbol_str: shares_float for symbol_str, shares_float in current_map_dict.items() if shares_float != 0} != position_map_dict:
                    allocation_dict.update(holdings_changed_bool=True, holdings_note_str="Holdings changed since this snapshot.",
                        comparison_basis_str="Compared with saved positions at " + current_time_str + " ET")
        result_dict["holdings_allocation_dict"] = allocation_dict
    except (ClientReportingError, ValueError, TypeError, KeyError, OSError, AttributeError, ArithmeticError):
        result_dict["holdings_allocation_dict"]["reason_str"] = "Saved IBKR position values could not be verified"
    return result_dict
