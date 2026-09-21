"""LIVE Positions display over saved broker quantities and canonical account facts."""

from copy import deepcopy
import sqlite3
from zoneinfo import ZoneInfo

from alpha.live.client_reporting import ClientReportingError, build_client_report_dict, validate_client_registry_dict
from alpha.live.dashboard_v3.client_cash import load_portfolio_cash_list
from alpha.live.dashboard_v3.client_financial_display import financial_dates_dict
from alpha.live.dashboard_v3.client_operations import active_account_list, build_client_operations_dict
from alpha.live.dashboard_v3.client_presentation import portfolio_allocation_dict
from alpha.live.dashboard_v4.finance import POD_COLOR_TUPLE, _money_str, _share_str
from alpha.live.dashboard_v4.positions_enrichment import enrich_positions_dict
from alpha.live.dashboard_v4.positions_data import (
    IDENTITY_FIELD_TUPLE, load_positions_dict, observed_timestamp_ts, validated_position_map_dict,
)


POD_LIMIT_INT = 64
MARKET_TIMEZONE_OBJ = ZoneInfo("America/New_York")


def _shares_str(share_float):
    return f"{share_float:,.6f}".rstrip("0").rstrip(".")


def _empty_page_dict():
    return {"verdict_str": "Positions unavailable", "verdict_detail_str": "Saved account ownership could not be verified.",
        "asof_str": "Saved positions unavailable", "note_str": "Closing prices and entry P&L are unavailable.",
        "tile_list": [{"label_str": label_str, "value_str": "—", "detail_str": "Data unavailable", "tone_str": ""}
            for label_str in ("Invested", "Open P&L", "Best", "Worst")],
        "row_list": [], "pod_row_list": [], "pod_filter_list": [],
        "total_dict": {"name_str": "Portfolio", "count_str": "—", "invested_str": "—", "cash_str": "—",
            "pnl_str": "—", "pnl_tone_str": "", "weight_str": "—"},
        "empty_str": "Saved positions unavailable", "all_count_int": 0, "changed_count_int": 0,
        "off_target_count_int": 0, "changed_available_bool": False, "off_target_available_bool": False,
        "holdings_complete_bool": False, "source_fresh_bool": False, "missing_pod_list": [],
        "financial_asof_str": "", "financial_basis_str": "", "financial_error_str": "", "financial_delayed_bool": True}


def _owned_accounts_tuple(workspace_dict, as_of_ts):
    client_dict = validate_client_registry_dict(
        {"schema_version": 1, "clients": [deepcopy(workspace_dict["client_dict"])]},
        allow_empty_periods_bool=True)["clients"][0]
    operation_list = workspace_dict.get("operations_account_list")
    if operation_list is None:
        operation_list = active_account_list(client_dict, as_of_ts)
    valuation_list = workspace_dict.get("valuation_account_list", client_dict["accounts"])
    pod_to_route_dict, route_to_pod_dict, owned_dict = {}, {}, {}
    for account_list in (valuation_list, operation_list, client_dict["accounts"]):
        if not isinstance(account_list, list) or len(account_list) > POD_LIMIT_INT:
            raise ValueError("Invalid ownership list")
        pair_set = set()
        for account_dict in account_list:
            pod_str, route_str = account_dict.get("pod_id"), account_dict.get("account_route")
            if any(not isinstance(value_str, str) or not value_str.strip() for value_str in (pod_str, route_str)):
                raise ValueError("Missing account identity")
            if (pod_str, route_str) in pair_set:
                raise ValueError("Duplicate ownership")
            pair_set.add((pod_str, route_str))
            if pod_to_route_dict.get(pod_str, route_str) != route_str or route_to_pod_dict.get(route_str, pod_str) != pod_str:
                raise ValueError("Conflicting ownership")
            pod_to_route_dict[pod_str], route_to_pod_dict[route_str] = route_str, pod_str
            owned_dict.setdefault(pod_str, {"pod_id": pod_str, "account_route": route_str,
                "display_name": account_dict.get("display_name") or pod_str})
    if not owned_dict or len(owned_dict) > POD_LIMIT_INT:
        raise ValueError("No bounded ownership scope")
    return client_dict, operation_list, list(owned_dict.values())


def _saved_positions_dict(provider_obj, account_dict, row_dict, *, demo_bool, as_of_ts):
    target_obj = provider_obj.get_target_for_pod(account_dict["pod_id"])
    release_obj = target_obj.release_obj
    if (release_obj.mode_str != "live" or release_obj.enabled_bool is not True
            or release_obj.pod_id_str != account_dict["pod_id"] or release_obj.account_route_str != account_dict["account_route"]
            or row_dict.get("release_id_str") != release_obj.release_id_str):
        raise ValueError("Current release identity mismatch")
    getter_fn = getattr(provider_obj, "get_positions_dict", None) if demo_bool else None
    source_dict = getter_fn(account_dict["pod_id"], as_of_ts=as_of_ts) if callable(getter_fn) else load_positions_dict(target_obj, as_of_ts=as_of_ts)
    if (not isinstance(source_dict, dict) or source_dict.get("available_bool") is not True
            or source_dict.get("mode_str") != "live"
            or any(source_dict.get(field_str) != getattr(release_obj, field_str) for field_str in IDENTITY_FIELD_TUPLE)
            or (source_dict.get("source_str"), source_dict.get("timestamp_basis_str")) not in {
                ("broker_snapshot", "observed"), ("broker_reconciliation", "recorded")}):
        raise ValueError("Unverified broker positions")
    position_ts = observed_timestamp_ts(source_dict.get("position_timestamp_str"), as_of_ts)
    if account_dict.get("effective_from") and position_ts.astimezone(MARKET_TIMEZONE_OBJ).date().isoformat() < account_dict["effective_from"]:
        raise ValueError("Positions precede ownership")
    return {**source_dict, "position_map_dict": validated_position_map_dict(source_dict.get("position_map_dict")),
        "position_ts": position_ts}


def _financial_dict(workspace_dict, snapshot_obj, provider_obj, client_dict, owned_list, operations_dict, *, as_of_ts):
    result_dict = {"account_dict": {}, "closing_str": "", "basis_str": "", "error_str": "",
        "invested_float": None, "cash_float": None, "cash_weight_float": None, "complete_bool": False, "delayed_bool": True}
    try:
        if workspace_dict.get("financial_error_str") or snapshot_obj.unavailable_reason_str:
            raise ValueError("Financial source unavailable")
        dates_dict = financial_dates_dict(client_dict, snapshot_obj, as_of_ts=as_of_ts, valuation_account_list=owned_list)
        closing_str = dates_dict["latest_complete_str"]
        if not closing_str:
            raise ValueError("No complete finalized close")
        report_client_dict = deepcopy(client_dict)
        report_client_dict["mandate_start_date"] = min(report_client_dict["mandate_start_date"], closing_str)
        report_dict = build_client_report_dict(report_client_dict, snapshot_obj,
            from_date_str=closing_str, to_date_str=closing_str, as_of_ts=as_of_ts,
            scope_complete_bool=workspace_dict.get("financial_scope_complete_bool", False), valuation_account_list=owned_list)
        cash_list = load_portfolio_cash_list(report_client_dict, report_dict, provider_obj, operations_dict, as_of_ts=as_of_ts)
        allocation_dict = portfolio_allocation_dict(report_dict, snapshot_obj, cash_snapshot_list=cash_list)
        for item_dict in allocation_dict["item_list"]:
            cash_float = item_dict.get("cash_float")
            # Canonical allocation authorizes cash only on the same broker EOD
            # equity basis. Never subtract it from separate finalized Flex NAV.
            invested_float = item_dict["value_float"] - cash_float if cash_float is not None else None
            result_dict["account_dict"][item_dict["detail_str"]] = {
                "invested_float": invested_float, "cash_float": cash_float, "weight_float": item_dict["weight_float"]}
        result_dict.update(closing_str=closing_str, basis_str=allocation_dict["basis_str"],
            complete_bool=bool(allocation_dict["item_list"]), delayed_bool=dates_dict["delayed_bool"])
        if allocation_dict.get("cash_complete_bool"):
            result_dict.update(invested_float=sum(item_dict["invested_float"] for item_dict in result_dict["account_dict"].values()),
                cash_float=sum(item_dict["cash_float"] for item_dict in result_dict["account_dict"].values()),
                cash_weight_float=allocation_dict["cash_weight_float"])
    except (ClientReportingError, ValueError, TypeError, KeyError, AttributeError, OSError, sqlite3.Error):
        result_dict["error_str"] = "Saved account totals unavailable."
    return result_dict


def build_positions_page_dict(workspace_dict, snapshot_obj, provider_obj, *, as_of_ts,
                             view_str="all", pod_str="all", search_str=""):
    """Merge saved quantities, attributed broker values and verified daily fills.

    Filters affect the position rows only. The account tiles and By pod table
    retain the complete owned portfolio scope, including unavailable accounts.
    """
    if view_str not in {"all", "changed", "off_target"} or as_of_ts.tzinfo is None or not isinstance(search_str, str):
        raise ValueError("Invalid Positions selection")
    result_dict = _empty_page_dict()
    try:
        client_dict, operation_list, owned_list = _owned_accounts_tuple(workspace_dict, as_of_ts)
        if pod_str != "all" and pod_str not in {account_dict["pod_id"] for account_dict in owned_list}:
            raise ValueError("Unknown Pod selection")
        if workspace_dict.get("operations_error_str"):
            raise ValueError("Operations source unavailable")
        summary_dict = workspace_dict.get("summary_dict", {})
        summary_list = summary_dict.get("pod_row_dict_list", [])
        if not isinstance(summary_list, list) or any(not isinstance(row_dict, dict) for row_dict in summary_list):
            raise ValueError("Invalid saved identity list")
        owned_route_dict = {account_dict["account_route"]: account_dict["pod_id"] for account_dict in owned_list}
        if any(row_dict.get("account_route_str") in owned_route_dict
               and row_dict.get("pod_id_str") != owned_route_dict[row_dict["account_route_str"]] for row_dict in summary_list):
            raise ValueError("Source account ownership conflicts")
        operations_dict = build_client_operations_dict(client_dict, summary_dict, as_of_ts=as_of_ts, local_account_list=operation_list)
    except (ClientReportingError, ValueError, TypeError, KeyError, AttributeError):
        return result_dict
    result_dict["source_fresh_bool"] = operations_dict["source_fresh_bool"]
    operation_by_pod_dict = {account_dict["pod_id"]: account_dict for account_dict in operation_list}
    financial_dict = _financial_dict(workspace_dict, snapshot_obj, provider_obj, client_dict, owned_list, operations_dict, as_of_ts=as_of_ts)
    result_dict.update(financial_asof_str=("Close " + financial_dict["closing_str"]
        + (" · delayed" if financial_dict["delayed_bool"] else "") if financial_dict["closing_str"] else ""),
        financial_delayed_bool=financial_dict["delayed_bool"],
        financial_basis_str=financial_dict["basis_str"], financial_error_str=financial_dict["error_str"])
    symbol_dict, timestamp_list, missing_list, source_by_pod_dict = {}, [], [], {}
    for index_int, account_dict in enumerate(owned_list):
        pod_id_str = account_dict["pod_id"]
        identity_dict = {"name_str": account_dict["display_name"], "pod_id_str": pod_id_str,
            "color_str": POD_COLOR_TUPLE[index_int % len(POD_COLOR_TUPLE)]}
        result_dict["pod_filter_list"].append(dict(identity_dict))
        pod_row_dict = {**identity_dict, "count_str": "—", "invested_str": "—", "cash_str": "—",
            "pnl_str": "—", "pnl_tone_str": "", "weight_str": "—", "positions_available_bool": False}
        finance_dict = financial_dict["account_dict"].get(account_dict["account_route"], {})
        pod_row_dict.update(invested_str=_money_str(finance_dict.get("invested_float")),
            cash_str=_money_str(finance_dict.get("cash_float")), weight_str=_share_str(finance_dict.get("weight_float")))
        try:
            if pod_id_str not in operation_by_pod_dict:
                raise ValueError("No active owned broker source")
            row_list = [row_dict for row_dict in summary_list if row_dict.get("pod_id_str") == pod_id_str]
            if (len(row_list) != 1 or row_list[0].get("mode_str") != "live"
                    or row_list[0].get("account_route_str") != account_dict["account_route"]
                    or row_list[0].get("db_status_str") != "ok"):
                raise ValueError("Current LIVE source identity unavailable")
            source_dict = _saved_positions_dict(provider_obj, operation_by_pod_dict[pod_id_str], row_list[0],
                demo_bool=client_dict.get("is_demo") is True, as_of_ts=as_of_ts)
            position_dict = {symbol_str: share_float for symbol_str, share_float in source_dict["position_map_dict"].items()
                if share_float != 0}
            pod_row_dict.update(count_str=str(len(position_dict)), positions_available_bool=True,
                position_timestamp_str=source_dict["position_timestamp_str"], position_source_str=source_dict["source_str"],
                timestamp_basis_str=source_dict["timestamp_basis_str"],
                position_asof_str=source_dict["position_ts"].astimezone(MARKET_TIMEZONE_OBJ).strftime("%Y-%m-%d %H:%M:%S ET")
                    + (" (recorded)" if source_dict["timestamp_basis_str"] == "recorded" else ""))
            timestamp_list.append(source_dict["position_ts"])
            source_by_pod_dict[pod_id_str] = {**source_dict, "position_map_dict": position_dict,
                "target_obj": provider_obj.get_target_for_pod(pod_id_str), "identity_dict": identity_dict,
                "position_asof_str": pod_row_dict["position_asof_str"]}
            for symbol_str, share_float in position_dict.items():
                symbol_dict.setdefault(symbol_str, []).append({**identity_dict, "share_float": share_float,
                    "share_str": _shares_str(share_float), "position_timestamp_str": source_dict["position_timestamp_str"],
                    "source_str": source_dict["source_str"], "timestamp_basis_str": source_dict["timestamp_basis_str"],
                    "position_asof_str": pod_row_dict["position_asof_str"]})
        except (ValueError, TypeError, KeyError, AttributeError, OSError, sqlite3.Error):
            missing_list.append(pod_id_str)
        result_dict["pod_row_list"].append(pod_row_dict)
    selected_count_int = sum(any(pod_str == "all" or holder_dict["pod_id_str"] == pod_str
        for holder_dict in holder_list) for holder_list in symbol_dict.values())
    complete_bool = not missing_list and bool(owned_list)
    result_dict.update(all_count_int=selected_count_int, missing_pod_list=missing_list, holdings_complete_bool=complete_bool,
        verdict_str=(f"{len(symbol_dict)} saved positions" if complete_bool else "Positions incomplete"),
        verdict_detail_str=(f"{len(missing_list)} pod(s) unavailable." if missing_list else "")
            + (" Refresh unavailable." if not result_dict["source_fresh_bool"] else ""),
        empty_str="No saved positions" if complete_bool else "Saved positions unavailable")
    if pod_str != "all":
        name_str = next(account_dict["display_name"] for account_dict in owned_list if account_dict["pod_id"] == pod_str)
        count_str = f"{selected_count_int} of {len(symbol_dict)} positions" if complete_bool else f"{selected_count_int} saved positions"
        result_dict["verdict_str"] = ("Positions unavailable" if pod_str in missing_list else count_str) + " · " + name_str
    if timestamp_list:
        first_str = min(timestamp_list).astimezone(MARKET_TIMEZONE_OBJ).strftime("%Y-%m-%d %H:%M:%S")
        last_str = max(timestamp_list).astimezone(MARKET_TIMEZONE_OBJ).strftime("%Y-%m-%d %H:%M:%S")
        result_dict["asof_str"] = "Saved positions · " + first_str + (" → " + last_str if last_str != first_str else "") + " ET"
    invested_float = financial_dict["invested_float"]
    if invested_float is not None:
        cash_weight_float = financial_dict["cash_weight_float"]
        result_dict["tile_list"][0].update(value_str=_money_str(invested_float),
            detail_str=_share_str(1 - cash_weight_float) + " · cash " + _share_str(cash_weight_float),
            bar_percent_float=100 * (1 - cash_weight_float))
    result_dict["total_dict"].update(count_str=str(len(symbol_dict)) if complete_bool else "—",
        invested_str=_money_str(invested_float), cash_str=_money_str(financial_dict["cash_float"]),
        weight_str="100.0%" if financial_dict["complete_bool"] else "—")
    all_row_list = enrich_positions_dict(result_dict, source_by_pod_dict, pod_str=pod_str, as_of_ts=as_of_ts)
    search_key_str = search_str.strip().casefold()
    result_dict["row_list"] = [row_dict for row_dict in all_row_list
        if (not search_key_str or search_key_str in row_dict["symbol_str"].casefold())
        and (view_str == "all" or view_str == "changed" and result_dict["changed_available_bool"] and row_dict["changed_bool"])]
    if view_str == "changed" and result_dict["changed_available_bool"]:
        result_dict["empty_str"] = "No verified changes today"
    elif view_str != "all":
        result_dict["empty_str"] = "Changed-today evidence unavailable" if view_str == "changed" else "Target comparison unavailable"
    elif pod_str in missing_list:
        result_dict["empty_str"] = "Saved positions unavailable for this pod"
    elif not result_dict["row_list"] and (search_key_str or pod_str != "all"):
        result_dict["empty_str"] = "No matching positions"
    return result_dict
