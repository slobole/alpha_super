"""Display geometry for already-calculated broker facts. No return calculation."""

import math


CHART_WIDTH_INT = 800
CHART_HEIGHT_INT = 180
PLOT_TOP_INT = 10
PLOT_BOTTOM_INT = 170


def _display_date_str(date_str):
    return date_str[:-4] + " · Start" if date_str.endswith(" SOD") else date_str


def _axis_label_str(value_float, unit_str, span_float):
    display_float = value_float * 100 if unit_str == "pct" else value_float
    step_float = span_float / 2 * (100 if unit_str == "pct" else 1)
    precision_int = max(2 if unit_str == "pct" else 0, min(8, -math.floor(math.log10(step_float)) + 1)) if step_float > 0 else 2
    if unit_str == "pct":
        return f"{display_float:.{precision_int}f}%"
    sign_str = "-" if display_float < 0 else ""
    return f"{sign_str}${abs(display_float):,.{precision_int}f}"


def nav_chart_dict(daily_list, *, value_field_str="nav_float", unit_str="usd", bars_bool=False):
    value_list = [row_dict.get(value_field_str) for row_dict in daily_list
        if row_dict.get(value_field_str) is not None and math.isfinite(row_dict[value_field_str])]
    if not value_list:
        return None
    low_float, high_float = min(value_list), max(value_list)
    if bars_bool:
        bound_float = max(abs(low_float), abs(high_float)) or (.01 if unit_str == "pct" else 1)
        low_float, high_float = -bound_float, bound_float
    elif high_float == low_float:
        padding_float = max(abs(high_float) * .01, .001 if unit_str == "pct" else 1)
        low_float, high_float = low_float - padding_float, high_float + padding_float
    span_float = high_float - low_float

    # Geometry only: y(v)=bottom-(v-low)/(high-low)*(bottom-top).
    def vertical_float(value_float):
        return PLOT_BOTTOM_INT - (value_float - low_float) / span_float * (PLOT_BOTTOM_INT - PLOT_TOP_INT)

    tick_list = [{"value_float": value_float, "label_str": _axis_label_str(value_float, unit_str, span_float),
        "y_float": vertical_float(value_float), "top_percent_float": vertical_float(value_float) / CHART_HEIGHT_INT * 100}
        for value_float in (high_float, (high_float + low_float) / 2, low_float)]
    segment_list, current_list, point_list, bar_list = [], [], [], []
    zero_y_float = vertical_float(0) if low_float <= 0 <= high_float else None
    slot_float = CHART_WIDTH_INT / len(daily_list)
    for index_int, row_dict in enumerate(daily_list):
        value_float = row_dict.get(value_field_str)
        horizontal_float = (index_int + .5) * slot_float if bars_bool else 3 + index_int / max(1, len(daily_list) - 1) * (CHART_WIDTH_INT - 6)
        date_str = row_dict.get("market_date_str", "")
        if value_float is None or not math.isfinite(value_float):
            if current_list:
                segment_list.append(" ".join(current_list))
                current_list = []
            if bars_bool:
                bar_list.append({"x_float": horizontal_float, "missing_bool": True, "market_date_str": date_str})
            continue
        point_dict = {"x_float": horizontal_float, "y_float": vertical_float(value_float),
            "value_float": value_float, "market_date_str": date_str,
            "display_date_str": _display_date_str(date_str),
            "label_str": _axis_label_str(value_float, unit_str, min(span_float, .02 if unit_str == "pct" else 2))}
        point_list.append(point_dict)
        current_list.append(f"{horizontal_float:.2f},{point_dict['y_float']:.2f}")
        if bars_bool:
            bar_list.append({**point_dict, "missing_bool": False, "width_float": min(12, slot_float * .72),
                "top_float": min(zero_y_float, point_dict["y_float"]),
                "height_float": abs(point_dict["y_float"] - zero_y_float),
                "sign_str": "positive" if value_float > 0 else "negative" if value_float < 0 else "zero"})
    if current_list:
        segment_list.append(" ".join(current_list))
    isolated_point_set = {segment_str for segment_str in segment_list if " " not in segment_str}
    for point_dict in point_list:
        point_dict["isolated_bool"] = f"{point_dict['x_float']:.2f},{point_dict['y_float']:.2f}" in isolated_point_set
    return {"segment_list": segment_list, "point_list": point_list, "bar_list": bar_list,
        "tick_list": tick_list, "zero_y_float": zero_y_float, "unit_str": unit_str,
        "width_int": CHART_WIDTH_INT, "height_int": CHART_HEIGHT_INT,
        "min_float": low_float, "max_float": high_float,
        "from_str": daily_list[0].get("market_date_str", ""), "to_str": daily_list[-1].get("market_date_str", ""),
        "from_label_str": _display_date_str(daily_list[0].get("market_date_str", "")),
        "to_label_str": _display_date_str(daily_list[-1].get("market_date_str", ""))}


def daily_history_list(report_dict, *, movement_key_set=None):
    """Exact-date display projection; never average accounts or difference NAV."""
    return_by_date_dict = {row_dict["market_date_str"]: row_dict["return_float"] for row_dict in report_dict["twr_daily_list"]}
    if not report_dict["client_twr_configured_bool"] and report_dict["twr_float"] is not None and len(report_dict["strategy_list"]) == 1:
        return_by_date_dict = {row_dict["market_date_str"]: row_dict["return_float"] for row_dict in report_dict["strategy_list"][0]["daily_list"]}
    # *** CRITICAL *** exact retrospective date lookup, not a signal/as-of join.
    # NAV coverage cannot authorize returns; absent daily return stays absent.
    portfolio_list = [{"market_date_str": row_dict["market_date_str"], "pnl_float": row_dict["pnl_float"],
        "return_float": return_by_date_dict.get(row_dict["market_date_str"])} for row_dict in report_dict["daily_book_list"]]
    scope_list = [{"display_name_str": "Portfolio", "daily_list": portfolio_list}]
    scope_list.extend({"display_name_str": strategy_dict["display_name_str"], "daily_list": strategy_dict["daily_list"]}
        for strategy_dict in report_dict["strategy_list"])
    if movement_key_set is not None:
        for index_int, scope_dict in enumerate(scope_list):
            daily_list = []
            for day_dict in scope_dict["daily_list"]:
                date_str = day_dict["market_date_str"]
                if index_int:
                    route_set = {report_dict["strategy_list"][index_int - 1]["account_route_str"]}
                elif "valuation_account_list" in report_dict:
                    route_set = {account_dict["account_route"] for account_dict in report_dict["valuation_account_list"]}
                else:
                    route_set = {strategy_dict["account_route_str"] for strategy_dict in report_dict["strategy_list"]
                        if strategy_dict["from_date_str"] <= date_str <= strategy_dict["to_date_str"]}
                daily_list.append({**day_dict, "capital_movement_bool": any((route_str, date_str) in movement_key_set for route_str in route_set)})
            scope_dict["daily_list"] = daily_list
    return [{**scope_dict,
        "return_chart_dict": nav_chart_dict(scope_dict["daily_list"], value_field_str="return_float", unit_str="pct", bars_bool=True),
        "pnl_chart_dict": nav_chart_dict(scope_dict["daily_list"], value_field_str="pnl_float", bars_bool=True)} for scope_dict in scope_list]
