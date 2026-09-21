"""Selected-period daily P&L presentation from saved report facts only."""

from datetime import date, timedelta
from decimal import Decimal

from alpha.live.dashboard_v4.charts import build_history_chart_dict
from alpha.live.dashboard_v4.finance import _money_str, _percent_str, _tone_str


def _compact_str(value_float):
    if value_float is None:
        return "—"
    if abs(value_float) >= 1000:
        divisor_int, suffix_str = (1000000, "m") if abs(value_float) >= 1000000 else (1000, "k")
        return f"{value_float / divisor_int:+.2f}{suffix_str}"
    return f"{value_float:+.0f}" if value_float else "0"


def build_daily_pnl_dict(book_list, return_list, session_set, *, from_str, to_str, heat_fn):
    # *** CRITICAL *** retrospective close-date display: exact-date lookup uses
    # canonical report returns, never P&L / NAV, forward fill or live prices.
    book_dict = {row_dict["market_date_str"]: row_dict for row_dict in book_list}
    return_dict = {row_dict["market_date_str"]: row_dict.get("return_float") for row_dict in return_list}
    date_set = set(session_set) | {date_str for date_str, row_dict in book_dict.items()
        if row_dict.get("pnl_float") not in (0, None)}
    day_list = []
    for date_str in sorted(date_set):
        if not from_str <= date_str <= to_str:
            continue
        pnl_float = book_dict.get(date_str, {}).get("pnl_float")
        return_float = return_dict.get(date_str)
        day_obj = date.fromisoformat(date_str)
        pnl_str, return_str = _money_str(pnl_float, signed_bool=True), _percent_str(return_float, signed_bool=True)
        day_list.append({"date_str": date_str, "market_date_str": date_str, "pnl_float": pnl_float,
            "return_float": return_float, "pnl_str": pnl_str, "return_str": return_str,
            "compact_str": _compact_str(pnl_float), "tone_str": _tone_str(pnl_float),
            "heat_str": ("" if pnl_float in (None, 0) else
                heat_fn(abs(return_float) * (1 if pnl_float > 0 else -1)) if return_float else
                "heat-pos-1" if pnl_float > 0 else "heat-neg-1"), "session_bool": date_str in session_set,
            "date_label_str": day_obj.strftime("%a %Y-%m-%d"),
            "readout_str": f"{day_obj:%a %Y-%m-%d} · {pnl_str} · {return_str}"})
    known_list = [day_dict for day_dict in day_list if day_dict["pnl_float"] is not None]
    complete_bool = bool(day_list) and len(known_list) == len(day_list)
    # Display totals are sums of the canonical dollar P&L, not linked returns.
    total_float = float(sum((Decimal(str(day_dict["pnl_float"])) for day_dict in known_list), Decimal(0))) if complete_bool else None
    best_dict = max(known_list, key=lambda day_dict: day_dict["pnl_float"]) if complete_bool else None
    worst_dict = min(known_list, key=lambda day_dict: day_dict["pnl_float"]) if complete_bool else None
    last_dict = next((day_dict for day_dict in reversed(day_list) if day_dict["session_bool"]), day_list[-1] if day_list else None)
    chart_dict = build_history_chart_dict(day_list, value_field_str="pnl_float", bars_bool=True)
    drawing_dict = chart_dict["drawing_dict"] if chart_dict else None
    if drawing_dict:
        point_dict = {point_dict["market_date_str"]: point_dict for point_dict in drawing_dict["series_list"][0]["point_list"]}
        first_int, last_int = date.fromisoformat(day_list[0]["date_str"]).toordinal(), date.fromisoformat(day_list[-1]["date_str"]).toordinal()
        for index_int, day_dict in enumerate(day_list):
            x_float = 10 + (date.fromisoformat(day_dict["date_str"]).toordinal() - first_int) / max(.75, last_int - first_int) * 780
            day_dict.update(x_float=x_float, x_percent_float=x_float / 8, index_int=index_int,
                extreme_bool=day_dict is best_dict or day_dict is worst_dict,
                y_percent_float=point_dict.get(day_dict["date_str"], {}).get("y_percent_float", 50))
        for index_int, day_dict in enumerate(day_list):
            left_float = (day_list[index_int - 1]["x_percent_float"] + day_dict["x_percent_float"]) / 2 if index_int else 0
            right_float = (day_list[index_int + 1]["x_percent_float"] + day_dict["x_percent_float"]) / 2 if index_int + 1 < len(day_list) else 100
            day_dict.update(hit_left_float=left_float, hit_width_float=right_float - left_float)
    day_by_date_dict = {day_dict["date_str"]: day_dict for day_dict in day_list}
    week_list = []
    if day_list:
        start_obj, end_obj = date.fromisoformat(from_str), date.fromisoformat(to_str)
        week_obj = start_obj - timedelta(days=start_obj.weekday())
        while week_obj <= end_obj:
            cell_list, extra_list, week_day_list = [], [], []
            for offset_int in range(7):
                day_obj = week_obj + timedelta(days=offset_int)
                date_str = day_obj.isoformat()
                day_dict = day_by_date_dict.get(date_str)
                if day_dict:
                    week_day_list.append(day_dict)
                if offset_int < 5:
                    cell_list.append({"day_dict": day_dict, "date_str": date_str,
                        "empty_str": "·" if not start_obj <= day_obj <= end_obj else "closed"})
                elif day_dict:
                    extra_list.append(day_dict)
            week_total_float = (float(sum((Decimal(str(day_dict["pnl_float"])) for day_dict in week_day_list), Decimal(0)))
                if week_day_list and all(day_dict["pnl_float"] is not None for day_dict in week_day_list) else None)
            week_list.append({"date_str": week_obj.isoformat(), "cell_list": cell_list, "extra_list": extra_list,
                "total_str": _compact_str(week_total_float), "total_detail_str": _money_str(week_total_float, signed_bool=True)})
            week_obj += timedelta(days=7)
    return {"day_list": day_list, "week_list": week_list, "chart_dict": chart_dict, "last_dict": last_dict,
        "total_str": _money_str(total_float, signed_bool=True), "complete_bool": complete_bool,
        "session_count_int": sum(day_dict["session_bool"] for day_dict in day_list),
        "missing_count_int": len(day_list) - len(known_list),
        "extra_count_int": sum(not day_dict["session_bool"] for day_dict in day_list),
        "up_int": sum(day_dict["pnl_float"] > 0 for day_dict in known_list),
        "down_int": sum(day_dict["pnl_float"] < 0 for day_dict in known_list),
        "flat_int": sum(day_dict["pnl_float"] == 0 for day_dict in known_list),
        "best_dict": best_dict, "worst_dict": worst_dict}
