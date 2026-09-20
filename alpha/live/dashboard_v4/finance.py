"""V4 display geometry over canonical, saved LIVE financial evidence only."""

import calendar
from copy import deepcopy
from datetime import date
import math
from zoneinfo import ZoneInfo

from alpha.live.client_reporting import ClientReportingError, build_client_report_dict
from alpha.live.dashboard_v3.client_cash import load_portfolio_cash_list
from alpha.live.dashboard_v3.client_charts import nav_chart_dict
from alpha.live.dashboard_v3.client_financial_display import financial_dates_dict
from alpha.live.dashboard_v3.client_operations import build_client_operations_dict
from alpha.live.dashboard_v3.client_presentation import portfolio_allocation_dict


POD_COLOR_TUPLE = ("#2a78d6", "#1baf7a", "#4a3aa7", "#e87ba4")
PERIOD_TUPLE = ("1M", "3M", "YTD", "All")


def _money_str(value_float, *, signed_bool=False):
    if value_float is None:
        return "—"
    prefix_str = "−" if value_float < 0 else "+" if signed_bool and value_float > 0 else ""
    return f"{prefix_str}${abs(value_float):,.2f}"


def _percent_str(value_float, *, signed_bool=False):
    if value_float is None:
        return "—"
    return f"{value_float:+.2%}" if signed_bool and value_float else f"{value_float:.2%}"


def _share_str(value_float):
    return "—" if value_float is None else f"{value_float:.1%}"


def _tone_str(value_float):
    return "pos" if value_float is not None and value_float > 0 else "neg" if value_float is not None and value_float < 0 else ""


def _empty_chart_dict():
    return {"available_bool": False, "empty_str": "Return unavailable", "point_str": "", "area_str": "",
        "segment_list": [], "isolated_point_list": [], "y_tick_list": [], "x_tick_list": [],
        "end_x_float": None, "end_y_float": None, "end_label_str": "—", "zero_y_float": None,
        "source_str": "", "detail_str": ""}


def _chart_dict(daily_list):
    """Plot canonical linked returns; never normalize NAV or remove flows here.

    The reporting layer owns R_t = product(1 + r_d) - 1, including the opening
    zero baseline. V4 only rescales its already-validated percentage geometry.
    """
    source_dict = nav_chart_dict(daily_list, value_field_str="cumulative_return_float", unit_str="pct")
    result_dict = _empty_chart_dict()
    if source_dict is None:
        return result_dict

    def horizontal_float(value_float):
        return 40 + (value_float - 3) / 794 * 456

    def vertical_float(value_float):
        return 18 + (value_float - 10) / 160 * 198

    zero_y_float = vertical_float(source_dict["zero_y_float"]) if source_dict["zero_y_float"] is not None else None
    baseline_float = zero_y_float if zero_y_float is not None else 216
    segment_list = []
    for point_str in source_dict["segment_list"]:
        coordinate_list = [tuple(float(value_str) for value_str in pair_str.split(",")) for pair_str in point_str.split()]
        coordinate_list = [(horizontal_float(horizontal_value_float), vertical_float(vertical_value_float))
            for horizontal_value_float, vertical_value_float in coordinate_list]
        point_str = " ".join(f"{horizontal_value_float:.2f},{vertical_value_float:.2f}"
            for horizontal_value_float, vertical_value_float in coordinate_list)
        segment_list.append({"point_str": point_str,
            "area_str": f"{coordinate_list[0][0]:.2f},{baseline_float:.2f} {point_str} {coordinate_list[-1][0]:.2f},{baseline_float:.2f}"})
    point_list = source_dict["point_list"]
    end_dict = point_list[-1]
    tick_index_list = sorted({0, len(point_list) // 2, len(point_list) - 1})
    result_dict.update(available_bool=True, empty_str="", segment_list=segment_list,
        point_str=segment_list[0]["point_str"] if len(segment_list) == 1 else "",
        area_str=segment_list[0]["area_str"] if len(segment_list) == 1 else "",
        isolated_point_list=[{"x_float": horizontal_float(point_dict["x_float"]),
            "y_float": vertical_float(point_dict["y_float"]), "label_str": point_dict["label_str"]}
            for point_dict in point_list if point_dict["isolated_bool"]],
        end_x_float=horizontal_float(end_dict["x_float"]), end_y_float=vertical_float(end_dict["y_float"]),
        end_label_str=_percent_str(end_dict["value_float"], signed_bool=True), zero_y_float=zero_y_float,
        y_tick_list=[{"y_float": vertical_float(tick_dict["y_float"]), "label_str": tick_dict["label_str"]}
            for tick_dict in source_dict["tick_list"]],
        x_tick_list=[{"x_float": horizontal_float(point_list[index_int]["x_float"]),
            "label_str": date.fromisoformat(point_list[index_int]["market_date_str"][:10]).strftime("%b %d")}
            for index_int in tick_index_list])
    return result_dict


def _ring_path_str(start_float, weight_float, inner_float, outer_float):
    """Annular sector; two arcs also render zero-cash / all-cash full circles."""
    angle_list = [2 * math.pi * fraction_float - math.pi / 2
        for fraction_float in (start_float, start_float + weight_float / 2, start_float + weight_float)]
    outer_list = [(116 + outer_float * math.cos(angle_float), 125 + outer_float * math.sin(angle_float)) for angle_float in angle_list]
    inner_list = [(116 + inner_float * math.cos(angle_float), 125 + inner_float * math.sin(angle_float)) for angle_float in reversed(angle_list)]
    point_fn = lambda coordinate_tuple: f"{coordinate_tuple[0]:.3f},{coordinate_tuple[1]:.3f}"
    return (f"M {point_fn(outer_list[0])} A {outer_float},{outer_float} 0 0 1 {point_fn(outer_list[1])} "
        f"A {outer_float},{outer_float} 0 0 1 {point_fn(outer_list[2])} L {point_fn(inner_list[0])} "
        f"A {inner_float},{inner_float} 0 0 0 {point_fn(inner_list[1])} "
        f"A {inner_float},{inner_float} 0 0 0 {point_fn(inner_list[2])} Z")


def _allocation_dict(source_dict):
    """Regroup validated V3 weights. No new valuation or residual cash estimate."""
    result_dict = {"available_bool": bool(source_dict.get("item_list")),
        "empty_str": source_dict.get("reason_str") or "Allocation unavailable", "slice_list": [], "label_list": [], "row_list": [],
        "cash_percent_str": "—", "total_cash_str": "—", "total_invested_str": "—", "total_value_str": _money_str(source_dict.get("total_float")),
        "cash_complete_bool": source_dict.get("cash_complete_bool", False),
        "invested_heading_str": "Invested", "ring_caption_str": "Outer ring = who holds the cash",
        "basis_str": source_dict.get("basis_str", ""), "source_str": source_dict.get("source_str", ""), "date_str": source_dict.get("date_str", "")}
    if not result_dict["available_bool"]:
        return result_dict
    complete_bool = result_dict["cash_complete_bool"]
    cash_weight_float = source_dict["cash_weight_float"] if complete_bool else None
    if complete_bool:
        result_dict.update(cash_percent_str=_share_str(cash_weight_float), total_cash_str=_share_str(cash_weight_float),
            total_invested_str=_share_str(1 - cash_weight_float), empty_str="")
    else:
        result_dict["basis_str"] += "; cash split unavailable — sectors show total pod value"
        result_dict.update(invested_heading_str="Value", total_invested_str="100.0%",
            ring_caption_str="Pod value · cash split unavailable")

    def add_slice_fn(start_float, weight_float, color_str, label_str, *, outer_bool=False):
        if weight_float <= 0:
            return
        result_dict["slice_list"].append({"path_str": _ring_path_str(start_float, weight_float, 91 if outer_bool else 52, 104 if outer_bool else 86),
            "color_str": color_str, "label_str": label_str})

    if complete_bool:
        add_slice_fn(0, cash_weight_float, "#dfe3e9", "Cash " + _share_str(cash_weight_float))
    invested_offset_float = cash_weight_float if complete_bool else 0
    cash_owner_list = []
    for index_int, item_dict in enumerate(source_dict["item_list"]):
        color_str = POD_COLOR_TUPLE[index_int % len(POD_COLOR_TUPLE)]
        name_str = item_dict["label_str"]
        if item_dict.get("show_identity_bool"):
            name_str += " · " + item_dict["detail_str"]
        result_dict["row_list"].append({"name_str": name_str, "color_str": color_str,
            "invested_str": _share_str(item_dict.get("invested_weight_float") if complete_bool else item_dict["weight_float"]),
            "cash_str": _share_str(item_dict.get("cash_weight_float"))})
        weight_float = item_dict["invested_weight_float"] if complete_bool else item_dict["weight_float"]
        add_slice_fn(invested_offset_float, weight_float, color_str,
            name_str + (" invested " if complete_bool else " value ") + _share_str(weight_float))
        invested_offset_float += weight_float
        if complete_bool:
            cash_owner_list.append({"name_str": name_str, "color_str": color_str, "weight_float": item_dict["cash_weight_float"]})
    cash_offset_float = 0
    for owner_dict in sorted(cash_owner_list, key=lambda owner_dict: -owner_dict["weight_float"]):
        weight_float = owner_dict["weight_float"]
        add_slice_fn(cash_offset_float, weight_float, owner_dict["color_str"],
            owner_dict["name_str"] + " cash " + _share_str(weight_float), outer_bool=True)
        if weight_float >= .03 and len(result_dict["label_list"]) < 3:
            # D labels the largest cash owners outside the ring, never inner slices.
            angle_float = 2 * math.pi * (cash_offset_float + weight_float / 2) - math.pi / 2
            horizontal_float = 116 + 113 * math.cos(angle_float)
            vertical_float = 128.5 + 113 * math.sin(angle_float)
            label_str = owner_dict["name_str"].split()[0] + f" {weight_float:.0%}"
            width_float = len(label_str) * 6
            fits_bool = 0 <= horizontal_float <= 298 - width_float and 12 <= vertical_float <= 246
            overlaps_bool = any(abs(vertical_float - label_dict["y_float"]) < 14
                and horizontal_float < label_dict["x_float"] + len(label_dict["label_str"]) * 6
                and label_dict["x_float"] < horizontal_float + width_float for label_dict in result_dict["label_list"])
            if fits_bool and not overlaps_bool:
                result_dict["label_list"].append({"x_float": horizontal_float, "y_float": vertical_float, "label_str": label_str})
        cash_offset_float += weight_float
    return result_dict


def _months_before_date(today_obj, months_int):
    month_index_int = today_obj.year * 12 + today_obj.month - 1 - months_int
    year_int, month_int = divmod(month_index_int, 12)
    month_int += 1
    return date(year_int, month_int, min(today_obj.day, calendar.monthrange(year_int, month_int)[1]))


def build_financial_overview_dict(workspace_dict, snapshot_obj, provider_obj, *, period_str="3M", as_of_ts):
    """Read-only presentation of one acquired, scoped LIVE reporting snapshot.

    V3 owns NAV, flow-adjusted P&L, TWR and cash validation. Each tile uses its
    own canonical period; missing history in a chart cannot alter the Day tile.
    Current Month/Year stay anchored to the current ET calendar, even if the
    last import is old. No account returns are averaged or NAV changes called P&L.
    """
    if period_str not in PERIOD_TUPLE or as_of_ts.tzinfo is None:
        raise ValueError("Choose a supported period and a timezone-aware clock.")
    client_dict = deepcopy(workspace_dict["client_dict"])
    demo_bool = client_dict.get("is_demo") is True
    result_dict = {"tile_list": [{"label_str": label_str, "value_str": "—", "detail_str": "Data unavailable", "tone_str": ""}
        for label_str in ("Account value", "Day", "Month", "Year")],
        "money_asof_str": ("Demo · " if demo_bool else "") + "Financial data unavailable",
        "chart_dict": _empty_chart_dict(), "allocation_dict": _allocation_dict({}),
        "financial_error_str": "", "delayed_bool": True, "is_demo_bool": demo_bool}
    account_list = workspace_dict.get("valuation_account_list") or client_dict.get("accounts", [])
    identity_dict = {(account_dict["account_route"], account_dict.get("pod_id")): account_dict for account_dict in account_list}
    result_dict["allocation_dict"]["row_list"] = [{"name_str": account_dict.get("display_name") or account_dict.get("pod_id") or account_dict["account_route"],
        "color_str": POD_COLOR_TUPLE[index_int % len(POD_COLOR_TUPLE)], "invested_str": "—", "cash_str": "—"}
        for index_int, account_dict in enumerate(identity_dict.values())]
    if workspace_dict.get("financial_error_str") or snapshot_obj.unavailable_reason_str:
        result_dict["financial_error_str"] = "Saved financial data could not be verified."
        return result_dict
    today_obj = as_of_ts.astimezone(ZoneInfo("America/New_York")).date()
    scope_complete_bool = workspace_dict.get("financial_scope_complete_bool", False)
    valuation_account_list = None
    if not scope_complete_bool or client_dict["mandate_start_date"] > today_obj.isoformat():
        valuation_account_list = workspace_dict.get("valuation_account_list", [])
        client_dict["mandate_start_date"] = min(client_dict["mandate_start_date"], today_obj.isoformat())
        broker_date_list = [row_obj.market_date_str for row_obj in snapshot_obj.row_tuple if row_obj.market_date_str <= today_obj.isoformat()]
        if broker_date_list:
            client_dict["mandate_start_date"] = min(client_dict["mandate_start_date"], min(broker_date_list))
    try:
        dates_dict = financial_dates_dict(client_dict, snapshot_obj, as_of_ts=as_of_ts, valuation_account_list=valuation_account_list)
        closing_str = dates_dict["latest_complete_str"]
        if not closing_str:
            result_dict["financial_error_str"] = "No complete finalized account value is available."
            return result_dict
        report_cache_dict = {}

        def report_dict(from_str):
            from_str = max(from_str, client_dict["mandate_start_date"])
            if from_str > closing_str:
                return None
            if from_str not in report_cache_dict:
                report_cache_dict[from_str] = build_client_report_dict(client_dict, snapshot_obj,
                    from_date_str=from_str, to_date_str=closing_str, as_of_ts=as_of_ts,
                    scope_complete_bool=scope_complete_bool, valuation_account_list=valuation_account_list)
            return report_cache_dict[from_str]

        day_dict = report_dict(closing_str)
        month_dict = report_dict(today_obj.replace(day=1).isoformat())
        year_dict = report_dict(today_obj.replace(month=1, day=1).isoformat())
        chart_start_str = client_dict["mandate_start_date"] if period_str == "All" else (
            today_obj.replace(month=1, day=1).isoformat() if period_str == "YTD" else
            _months_before_date(today_obj, 1 if period_str == "1M" else 3).isoformat())
        chart_report_dict = report_dict(chart_start_str)
        result_dict["tile_list"][0].update(value_str=_money_str(day_dict["closing_nav_float"]), detail_str="Close " + closing_str)
        for index_int, period_dict in enumerate((day_dict, month_dict, year_dict), start=1):
            if period_dict is None:
                result_dict["tile_list"][index_int]["detail_str"] = "No data in this period"
                continue
            return_float, pnl_float = period_dict["twr_float"], period_dict["pnl_float"]
            result_dict["tile_list"][index_int].update(
                value_str=_money_str(pnl_float, signed_bool=True) if index_int == 1 else _percent_str(return_float, signed_bool=True),
                detail_str=_percent_str(return_float, signed_bool=True) if index_int == 1 else _money_str(pnl_float, signed_bool=True),
                tone_str=_tone_str(pnl_float if index_int == 1 else return_float))
        return_path_list = []
        if chart_report_dict and chart_report_dict["twr_float"] is not None:
            return_path_list = chart_report_dict["return_path_list"]
            if not chart_report_dict["client_twr_configured_bool"] and len(chart_report_dict["strategy_list"]) == 1:
                return_path_list = chart_report_dict["strategy_list"][0]["performance_dict"]["return_path_list"]
        chart_dict = _chart_dict(return_path_list)
        if chart_dict["available_bool"]:
            calculated_bool = chart_report_dict["client_twr_configured_bool"]
            chart_dict.update(source_str="Calculated return · End-of-day cash flows" if calculated_bool else
                "Demo account return" if demo_bool else "IBKR account return",
                detail_str="Cumulative return for the selected period. " + chart_report_dict["twr_method_str"])
        elif chart_report_dict:
            chart_dict["detail_str"] = chart_report_dict.get("twr_reason_str") or "Complete, verified return history is required."
        result_dict.update(money_asof_str=("Demo · " if demo_bool else "") + "Close " + closing_str
            + (" · Data delayed" if dates_dict["delayed_bool"] else ""), delayed_bool=dates_dict["delayed_bool"],
            chart_dict=chart_dict)
    except (ClientReportingError, ValueError, OSError, KeyError, TypeError):
        result_dict["financial_error_str"] = "Saved financial data could not be verified."
        return result_dict
    # Optional saved cash cannot change or suppress canonical financial facts.
    try:
        operations_dict = build_client_operations_dict(client_dict, workspace_dict.get("summary_dict", {}),
            as_of_ts=as_of_ts, local_account_list=workspace_dict.get("operations_account_list"))
        cash_list = load_portfolio_cash_list(client_dict, day_dict, provider_obj, operations_dict, as_of_ts=as_of_ts)
    except (ValueError, OSError, KeyError, TypeError, AttributeError):
        cash_list = []
    allocation_dict = _allocation_dict(portfolio_allocation_dict(day_dict, snapshot_obj, cash_snapshot_list=cash_list))
    if not allocation_dict["available_bool"]:
        allocation_dict["row_list"] = result_dict["allocation_dict"]["row_list"]
    result_dict["allocation_dict"] = allocation_dict
    return result_dict
