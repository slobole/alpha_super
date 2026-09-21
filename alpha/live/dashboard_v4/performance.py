"""Read-only Performance presentation over the canonical saved Flex report."""

import calendar
from copy import deepcopy
from datetime import date, timedelta
from decimal import Decimal
import math
import statistics
from zoneinfo import ZoneInfo

from alpha.live.client_reporting import ClientReportingError, build_client_report_dict
from alpha.live.dashboard_v3.client_charts import nav_chart_dict
from alpha.live.dashboard_v3.client_financial_display import financial_dates_dict
from alpha.live.dashboard_v4.finance import POD_COLOR_TUPLE, _money_str, _percent_str, _tone_str
from alpha.live.scheduler_utils import get_exchange_calendar_obj


PERIOD_TUPLE = ("1W", "MTD", "YTD", "All")
MONTH_TUPLE = tuple(calendar.month_abbr[1:])
MARKET_TIMEZONE_OBJ = ZoneInfo("America/New_York")


def _empty_dict(period_str, level_str, unit_str):
    return {"period_str": period_str, "level_str": level_str, "unit_str": unit_str,
        "report_dict": None, "from_date_str": "", "to_date_str": "", "min_date_str": "", "max_date_str": "",
        "latest_close_str": "", "source_str": "Financial data unavailable", "verdict_str": "Performance unavailable",
        "verdict_detail_str": "", "error_str": "", "delayed_bool": True, "is_demo_bool": False,
        "tile_list": [{"label_str": label_str, "value_str": "—", "detail_str": "", "tone_str": ""}
            for label_str in ("Start", "End", "Profit / loss", "Return", "Max drawdown", "End below high")],
        "bridge_row_list": [], "chart_dict": None, "daily_chart_dict": None,
        "chart_title_str": "Portfolio return" if unit_str == "pct" else "Account value",
        "chart_basis_str": "Cumulative TWR" if unit_str == "pct" else "Includes capital movements",
        "pod_row_list": [], "total_row_dict": {field_str: "—" for field_str in
            ("start_str", "end_str", "pnl_str", "return_str", "drawdown_str")},
        "contribution_dict": {"available_bool": False, "row_list": [], "total_str": "—",
            "reason_str": "Complete portfolio P&L is required"},
        "pod_chart_dict": {"available_bool": False, "series_list": [], "tick_list": [],
            "width_int": 800, "height_int": 180, "from_str": "", "to_str": ""},
        "monthly_row_list": [], "month_tuple": MONTH_TUPLE, "risk_row_list": [],
        "daily_basis_str": "Last 30 reporting days · selected period"}


def _risk_dict(path_list, daily_list, session_set):
    result_dict = {"max_float": None, "current_float": None, "max_detail_str": "", "current_detail_str": "",
        "row_list": []}
    if not path_list or not daily_list or any(row_dict.get("return_float") is None for row_dict in daily_list):
        return result_dict
    peak_float, peak_index_int = 1., 0
    max_float, max_peak_str, max_end_str = 0., "", ""
    # *** CRITICAL *** retrospective selected-period risk only. The opening
    # index is 1; DD_t = (1 + canonical cumulative return_t) / peak_t - 1.
    # Capital movements never enter this path and no missing day is filled.
    for index_int, point_dict in enumerate(path_list):
        value_float = 1 + point_dict["cumulative_return_float"]
        if value_float >= peak_float:
            peak_float, peak_index_int = value_float, index_int
        drawdown_float = value_float / peak_float - 1
        if drawdown_float < max_float:
            max_float = drawdown_float
            max_peak_str, max_end_str = path_list[peak_index_int]["market_date_str"][:10], point_dict["market_date_str"][:10]
    below_int = sum(point_dict["market_date_str"] in session_set for point_dict in path_list[peak_index_int + 1:])
    result_dict.update(max_float=max_float, current_float=drawdown_float,
        max_detail_str=f"{max_peak_str} → {max_end_str}" if max_end_str else "Selected period",
        current_detail_str=f"{below_int} {'session' if below_int == 1 else 'sessions'} · selected period" if drawdown_float < 0 else "At the period high")
    session_return_list = [row_dict["return_float"] for row_dict in daily_list if row_dict["market_date_str"] in session_set]
    # Last 20 completed exchange-session returns, sample stdev * sqrt(252).
    # Non-session account activity remains in TWR, but is not a trading session.
    volatility_str = (f"{statistics.stdev(session_return_list[-20:]) * math.sqrt(252):.1%} a year"
        if len(session_return_list) >= 20 else "Needs 20 sessions")
    result_dict["row_list"] = [{"label_str": "Volatility · 20 sessions", "value_str": volatility_str},
        {"label_str": "Sessions below high", "value_str": str(below_int)}]
    if all(row_dict.get("pnl_float") is not None for row_dict in daily_list):
        for label_str, day_dict in (("Best day", max(daily_list, key=lambda row_dict: row_dict["pnl_float"])),
                                   ("Worst day", min(daily_list, key=lambda row_dict: row_dict["pnl_float"]))):
            result_dict["row_list"].append({"label_str": label_str,
                "value_str": _money_str(day_dict["pnl_float"], signed_bool=True) + " · " + day_dict["market_date_str"],
                "tone_str": _tone_str(day_dict["pnl_float"])})
    if session_return_list:
        result_dict["row_list"].append({"label_str": "Up sessions",
            "value_str": f"{sum(return_float > 0 for return_float in session_return_list) / len(session_return_list):.0%}"})
    return result_dict


def _monthly_list(daily_list, *, name_str, pod_id_str, from_str, to_str):
    if not daily_list or any(row_dict.get("return_float") is None for row_dict in daily_list):
        return []
    growth_dict = {}
    # Calendar-month linking of the same complete, canonical daily returns.
    # No resampling, gap filling, account averaging or history extension.
    for day_dict in daily_list:
        month_str = day_dict["market_date_str"][:7]
        growth_dict[month_str] = growth_dict.get(month_str, Decimal(1)) * (1 + Decimal(str(day_dict["return_float"])))
    result_list = []
    for year_str in sorted({month_str[:4] for month_str in growth_dict}):
        year_growth_decimal, cell_list = Decimal(1), []
        for month_int in range(1, 13):
            month_str = f"{year_str}-{month_int:02d}"
            growth_decimal = growth_dict.get(month_str)
            return_float = None if growth_decimal is None else float(growth_decimal - 1)
            month_end_str = f"{month_str}-{calendar.monthrange(int(year_str), month_int)[1]:02d}"
            partial_bool = growth_decimal is not None and (from_str > month_str + "-01" or to_str < month_end_str)
            cell_list.append({"month_int": month_int, "return_float": return_float,
                "value_str": "—" if return_float is None else f"{return_float * 100:+.2f}",
                "partial_bool": partial_bool, "tone_str": _tone_str(return_float)})
            if growth_decimal is not None:
                year_growth_decimal *= growth_decimal
        result_list.append({"name_str": name_str, "pod_id_str": pod_id_str, "year_int": int(year_str),
            "cell_list": cell_list, "year_str": _percent_str(float(year_growth_decimal - 1), signed_bool=True),
            "partial_bool": from_str > year_str + "-01-01" or to_str < year_str + "-12-31"})
    return result_list


def _pod_chart_dict(pod_list):
    result_dict = {"available_bool": False, "series_list": [], "tick_list": [],
        "width_int": 800, "height_int": 180, "from_str": "", "to_str": ""}
    available_list = [pod_dict for pod_dict in pod_list if pod_dict["path_list"]]
    if not available_list:
        return result_dict

    def date_float(date_str):
        # One calendar axis for all Pods; preserve opening SOD and first EOD.
        return date.fromisoformat(date_str[:10]).toordinal() + (0 if date_str.endswith(" SOD") else .75)

    all_point_list = [point_dict for pod_dict in available_list for point_dict in pod_dict["path_list"]]
    start_float = min(date_float(point_dict["market_date_str"]) for point_dict in all_point_list)
    end_float = max(date_float(point_dict["market_date_str"]) for point_dict in all_point_list)
    value_list = [100 * (1 + point_dict["cumulative_return_float"]) for point_dict in all_point_list]
    low_float, high_float = min(value_list), max(value_list)
    if high_float == low_float:
        low_float, high_float = low_float - 1, high_float + 1
    span_float = high_float - low_float
    for pod_dict in available_list:
        point_list = []
        for source_dict in pod_dict["path_list"]:
            value_float = 100 * (1 + source_dict["cumulative_return_float"])
            point_list.append({"x_float": 3 + (date_float(source_dict["market_date_str"]) - start_float) / max(.75, end_float - start_float) * 794,
                "y_float": 170 - (value_float - low_float) / span_float * 160, "value_float": value_float,
                "market_date_str": source_dict["market_date_str"], "label_str": f"{value_float:.2f}"})
        result_dict["series_list"].append({field_str: pod_dict[field_str] for field_str in ("pod_id_str", "name_str", "color_str")}
            | {"point_list": point_list, "segment_list": [" ".join(f"{point_dict['x_float']:.2f},{point_dict['y_float']:.2f}" for point_dict in point_list)],
                "end_str": point_list[-1]["label_str"]})
    result_dict.update(available_bool=True,
        from_str=min(point_dict["market_date_str"] for point_dict in all_point_list)[:10],
        to_str=max(point_dict["market_date_str"] for point_dict in all_point_list)[:10],
        tick_list=[{"value_float": value_float, "label_str": f"{value_float:.1f}", "y_float": 170 - (value_float - low_float) / span_float * 160}
            for value_float in (high_float, (high_float + low_float) / 2, low_float)])
    return result_dict


def build_performance_page_dict(workspace_dict, snapshot_obj, *, as_of_ts, period_str="All",
                                from_date_str=None, to_date_str=None, unit_str="pct", level_str="portfolio"):
    """Project one report; saved LIVE ownership and accounting stay canonical."""
    if (period_str not in PERIOD_TUPLE or unit_str not in {"usd", "pct"} or level_str not in {"portfolio", "pods"}
            or as_of_ts.tzinfo is None or bool(from_date_str) != bool(to_date_str)):
        raise ValueError("Choose a valid period, level, unit and timezone-aware clock.")
    today_obj = as_of_ts.astimezone(MARKET_TIMEZONE_OBJ).date()
    if from_date_str:
        if any(date.fromisoformat(value_str).isoformat() != value_str for value_str in (from_date_str, to_date_str)) or from_date_str > to_date_str or to_date_str > today_obj.isoformat():
            raise ValueError("Choose an ordered date range through today.")
    result_dict = _empty_dict(period_str, level_str, unit_str)
    client_dict = deepcopy(workspace_dict["client_dict"])
    result_dict.update(is_demo_bool=client_dict.get("is_demo") is True,
        min_date_str=client_dict["mandate_start_date"], max_date_str=today_obj.isoformat())
    if workspace_dict.get("financial_error_str") or snapshot_obj.unavailable_reason_str:
        result_dict["error_str"] = "Saved financial data could not be verified."
        return result_dict
    scope_complete_bool = workspace_dict.get("financial_scope_complete_bool", False)
    valuation_list = None
    if not scope_complete_bool or client_dict["mandate_start_date"] > today_obj.isoformat():
        valuation_list = workspace_dict.get("valuation_account_list", [])
        client_dict["mandate_start_date"] = min(client_dict["mandate_start_date"], today_obj.isoformat())
        saved_date_list = [row_obj.market_date_str for row_obj in snapshot_obj.row_tuple if row_obj.market_date_str < today_obj.isoformat()]
        if saved_date_list:
            client_dict["mandate_start_date"] = min(client_dict["mandate_start_date"], min(saved_date_list))
    try:
        dates_dict = financial_dates_dict(client_dict, snapshot_obj, as_of_ts=as_of_ts, valuation_account_list=valuation_list)
        close_str = dates_dict["latest_complete_str"]
        result_dict.update(latest_close_str=close_str or "", delayed_bool=dates_dict["delayed_bool"], min_date_str=client_dict["mandate_start_date"])
        if not close_str and not snapshot_obj.row_tuple:
            result_dict["error_str"] = "No complete finalized account value is available."
            return result_dict
        if not from_date_str:
            from_date_str = client_dict["mandate_start_date"]
            if period_str == "MTD":
                from_date_str = max(from_date_str, today_obj.replace(day=1).isoformat())
            elif period_str == "YTD":
                from_date_str = max(from_date_str, today_obj.replace(month=1, day=1).isoformat())
            elif period_str == "1W":
                from_date_str = max(from_date_str, (today_obj - timedelta(days=7)).isoformat())
            # Old imports cannot move MTD/YTD into a past month/year.
            to_date_str = close_str if close_str and close_str >= from_date_str else max(from_date_str, (today_obj - timedelta(days=1)).isoformat())
        result_dict.update(from_date_str=from_date_str, to_date_str=to_date_str)
        report_dict = build_client_report_dict(client_dict, snapshot_obj, from_date_str=from_date_str,
            to_date_str=to_date_str, as_of_ts=as_of_ts, scope_complete_bool=scope_complete_bool, valuation_account_list=valuation_list)
        session_set = {session_obj.date().isoformat() for session_obj in get_exchange_calendar_obj("XNYS").sessions_in_range(from_date_str, to_date_str)}
    except (ClientReportingError, ValueError, KeyError, TypeError, OSError):
        result_dict["error_str"] = "Saved financial data could not be verified for this period."
        return result_dict
    result_dict["report_dict"] = report_dict
    path_list, daily_return_list = report_dict["return_path_list"], report_dict["twr_daily_list"]
    if not report_dict["client_twr_configured_bool"] and report_dict["twr_float"] is not None and len(report_dict["strategy_list"]) == 1:
        path_list = report_dict["strategy_list"][0]["performance_dict"]["return_path_list"]
        daily_return_list = report_dict["strategy_list"][0]["daily_list"]
    pnl_by_date_dict = {row_dict["market_date_str"]: row_dict["pnl_float"] for row_dict in report_dict["daily_book_list"]}
    daily_list = [{**row_dict, "pnl_float": pnl_by_date_dict.get(row_dict["market_date_str"])} for row_dict in daily_return_list]
    risk_dict = _risk_dict(path_list, daily_list, session_set)
    result_dict["risk_row_list"] = risk_dict["row_list"]
    method_str = ("Calculated TWR · EOD cash flows" if report_dict["client_twr_configured_bool"] else
        "IBKR account TWR" if len(report_dict["strategy_list"]) == 1 else "Portfolio TWR unavailable")
    source_method_str = method_str
    if level_str == "pods":
        source_method_str = "Account TWR" + (" · Portfolio total: calculated" if report_dict["client_twr_configured_bool"]
            else " · Portfolio TWR unavailable" if len(report_dict["strategy_list"]) > 1 else "")
    result_dict["source_str"] = ("Demo" if result_dict["is_demo_bool"] else "IBKR Flex") + " · " + source_method_str
    if not close_str:
        result_dict["source_str"] += " · Portfolio close incomplete"
    elif dates_dict["delayed_bool"]:
        result_dict["source_str"] += " · delayed"
    result_dict["chart_basis_str"] = method_str if unit_str == "pct" else "Includes capital movements"
    tile_value_list = [report_dict["opening_nav_float"], report_dict["closing_nav_float"], report_dict["pnl_float"],
        report_dict["twr_float"], risk_dict["max_float"], risk_dict["current_float"]]
    detail_list = [report_dict["opening_date_str"] or "", report_dict["closing_date_str"] or "", "After reported costs",
        "Calculated TWR" if report_dict["client_twr_configured_bool"] else "Account TWR", risk_dict["max_detail_str"], risk_dict["current_detail_str"]]
    for index_int, tile_dict in enumerate(result_dict["tile_list"]):
        value_float = tile_value_list[index_int]
        tile_dict.update(value_str=_money_str(value_float, signed_bool=index_int == 2) if index_int < 3 else _percent_str(value_float, signed_bool=index_int == 3),
            detail_str=detail_list[index_int], tone_str=_tone_str(value_float) if index_int in {2, 3} else "")
    result_dict["total_row_dict"] = dict(zip(("start_str", "end_str", "pnl_str", "return_str", "drawdown_str"),
        (tile_dict["value_str"] for tile_dict in result_dict["tile_list"][:5])))
    if report_dict["twr_float"] is not None:
        result_dict.update(verdict_str=_percent_str(report_dict["twr_float"], signed_bool=True) + " in this period.",
            verdict_detail_str=_money_str(report_dict["pnl_float"], signed_bool=True) if report_dict["pnl_float"] is not None else "P&L unavailable.")
    else:
        result_dict.update(verdict_str="Return unavailable", verdict_detail_str=report_dict.get("twr_reason_str") or "Complete return history is required.")
    result_dict["bridge_row_list"] = [{"label_str": label_str, "value_str": _money_str(report_dict[field_str], signed_bool=field_str not in {"opening_nav_float", "closing_nav_float"}),
        "total_bool": field_str == "closing_nav_float"} for label_str, field_str in (
        ("Start", "opening_nav_float"), ("Transfers in / out", "capital_movement_float"),
        ("Broker adjustments", "linking_adjustment_float"), ("Pods added / removed", "scope_movement_float"),
        ("Profit / loss", "pnl_float"), ("End", "closing_nav_float"))]
    chart_source_list = path_list if unit_str == "pct" else report_dict["daily_book_list"]
    result_dict["chart_dict"] = nav_chart_dict(chart_source_list, value_field_str="cumulative_return_float" if unit_str == "pct" else "nav_float",
        unit_str=unit_str, daily_fact_list=report_dict["daily_book_list"])
    result_dict["daily_chart_dict"] = nav_chart_dict(report_dict["daily_book_list"][-30:], value_field_str="pnl_float", bars_bool=True)
    color_pair_list = list(dict.fromkeys((account_dict["account_route"], account_dict["pod_id"])
        for account_dict in [*(workspace_dict.get("valuation_account_list") or []), *client_dict["accounts"]]))
    for strategy_dict in report_dict["strategy_list"]:
        pair_tuple = (strategy_dict["account_route_str"], strategy_dict["pod_id_str"])
        color_index_int = color_pair_list.index(pair_tuple)
        result_dict["pod_row_list"].append({"pod_id_str": strategy_dict["pod_id_str"], "name_str": strategy_dict["display_name_str"],
            "color_str": POD_COLOR_TUPLE[color_index_int % len(POD_COLOR_TUPLE)],
            "from_date_str": strategy_dict["from_date_str"], "to_date_str": strategy_dict["to_date_str"],
            "start_str": _money_str(strategy_dict["opening_nav_float"]), "end_str": _money_str(strategy_dict["closing_nav_float"]),
            "pnl_str": _money_str(strategy_dict["pnl_float"], signed_bool=True), "return_str": _percent_str(strategy_dict["twr_float"], signed_bool=True),
            "drawdown_str": _percent_str(strategy_dict["performance_dict"]["max_drawdown_float"]),
            "tone_str": _tone_str(strategy_dict["pnl_float"]), "pnl_float": strategy_dict["pnl_float"],
            "path_list": strategy_dict["performance_dict"]["return_path_list"]})
    result_dict["pod_chart_dict"] = _pod_chart_dict(result_dict["pod_row_list"])
    if report_dict["pnl_float"] is not None and report_dict["flows_complete_bool"] and report_dict["coverage_complete_bool"] and all(
            pod_dict["pnl_float"] is not None for pod_dict in result_dict["pod_row_list"]) and abs(
            sum(Decimal(str(pod_dict["pnl_float"])) for pod_dict in result_dict["pod_row_list"]) - Decimal(str(report_dict["pnl_float"]))) <= Decimal(".01"):
        bound_float = max((abs(pod_dict["pnl_float"]) for pod_dict in result_dict["pod_row_list"]), default=0.) or 1.
        result_dict["contribution_dict"] = {"available_bool": True, "reason_str": "", "total_str": _money_str(report_dict["pnl_float"], signed_bool=True),
            "row_list": [{field_str: pod_dict[field_str] for field_str in ("pod_id_str", "name_str", "color_str")}
                | {"value_str": pod_dict["pnl_str"], "value_float": pod_dict["pnl_float"], "bar_percent_float": 100 * abs(pod_dict["pnl_float"]) / bound_float}
                for pod_dict in sorted(result_dict["pod_row_list"], key=lambda pod_dict: -pod_dict["pnl_float"])]}
    if level_str == "portfolio":
        result_dict["monthly_row_list"] = _monthly_list(daily_return_list, name_str="Portfolio", pod_id_str="",
            from_str=from_date_str, to_str=to_date_str)
    else:
        for strategy_dict in report_dict["strategy_list"]:
            account_dict = next(account_dict for account_dict in client_dict["accounts"]
                if account_dict["pod_id"] == strategy_dict["pod_id_str"] and account_dict["account_route"] == strategy_dict["account_route_str"]
                and account_dict["effective_from"] <= strategy_dict["from_date_str"] <= (account_dict.get("effective_to") or "9999-12-31"))
            result_dict["monthly_row_list"].extend(_monthly_list(strategy_dict["daily_list"], name_str=strategy_dict["display_name_str"],
                pod_id_str=strategy_dict["pod_id_str"], from_str=max(from_date_str, account_dict["effective_from"]),
                to_str=min(to_date_str, account_dict.get("effective_to") or to_date_str)))
    return result_dict
