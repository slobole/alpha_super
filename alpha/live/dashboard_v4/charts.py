"""One V4 drawing style for canonical financial facts; no return calculation."""

from datetime import date, timedelta
import math

from alpha.live.dashboard_v3.client_charts import nav_chart_dict


WIDTH_INT = 800
NICE_STEP_TUPLE = (1., 2., 2.5, 5., 10.)


def _finite_bool(value_obj):
    return type(value_obj) in {int, float} and math.isfinite(value_obj)


def _date_float(date_str):
    # SOD and that day's EOD are distinct display points, not an as-of join.
    return date.fromisoformat(date_str[:10]).toordinal() + (0. if date_str.endswith(" SOD") else .75)


def _axis_str(value_float, step_float, unit_str):
    display_float = value_float * 100 if unit_str == "pct" else value_float
    display_step_float = step_float * 100 if unit_str == "pct" else step_float
    suffix_str = "%" if unit_str == "pct" else ""
    if unit_str == "usd" and abs(display_float) >= 1_000:
        divisor_float, suffix_str = (1_000_000., "m") if abs(display_float) >= 1_000_000 else (1_000., "k")
        display_float, display_step_float = display_float / divisor_float, display_step_float / divisor_float
    precision_int = max(0, min(10, -math.floor(math.log10(display_step_float)))) if display_step_float else 0
    while precision_int < 10 and abs(round(display_step_float, precision_int) - display_step_float) > abs(display_step_float) * 1e-8:
        precision_int += 1
    if abs(display_float) < abs(display_step_float) * 1e-8:
        display_float = 0.
    return f"{display_float:.{precision_int}f}" + suffix_str


def _value_str(value_float, unit_str):
    if unit_str == "pct":
        return f"{value_float:+.2%}" if value_float else "0.00%"
    if unit_str == "usd":
        return ("−" if value_float < 0 else "") + f"${abs(value_float):,.2f}"
    return f"{value_float:.2f}"


def _scale_tuple(value_list, *, unit_str, bars_bool):
    multiplier_float = 100 if unit_str == "pct" else 1
    low_float, high_float = min(value_list) * multiplier_float, max(value_list) * multiplier_float
    if bars_bool:
        bound_float = max(abs(low_float), abs(high_float)) or 1.
        exponent_int = math.floor(math.log10(bound_float / 2))
        step_float = next(step_float * 10 ** exponent_int for step_float in NICE_STEP_TUPLE if step_float * 10 ** exponent_int >= bound_float / 2)
        return [index_int * step_float / multiplier_float for index_int in range(-2, 3)], step_float / multiplier_float
    if high_float == low_float:
        padding_float = max(abs(high_float) * .02, 1. if unit_str != "pct" else .1)
        low_float, high_float = low_float - padding_float, high_float + padding_float
    span_float = high_float - low_float
    exponent_int = math.floor(math.log10(span_float / 5))
    candidate_list = []
    for power_int in range(exponent_int - 1, exponent_int + 2):
        for coefficient_float in NICE_STEP_TUPLE:
            step_float = coefficient_float * 10 ** power_int
            first_int = math.floor(low_float / step_float + 1e-10)
            last_int = math.ceil(high_float / step_float - 1e-10)
            count_int = last_int - first_int + 1
            if not 4 <= count_int <= 8:
                continue
            padding_float = ((last_int - first_int) * step_float - span_float) / span_float
            score_tuple = (0 if 5 <= count_int <= 6 else 1, abs(count_int - 6), padding_float)
            candidate_list.append((score_tuple, first_int, last_int, step_float))
    _, first_int, last_int, step_float = min(candidate_list)
    return [index_int * step_float / multiplier_float for index_int in range(first_int, last_int + 1)], step_float / multiplier_float


def _x_ticks_list(start_float, end_float, *, bars_bool):
    start_obj, end_obj = date.fromordinal(math.floor(start_float)), date.fromordinal(math.floor(end_float))
    month_bool = not bars_bool and (end_obj - start_obj).days >= 45
    date_list = [start_obj]
    if month_bool:
        tick_obj = (start_obj.replace(day=28) + timedelta(days=4)).replace(day=1)
        while tick_obj <= end_obj:
            date_list.append(tick_obj)
            tick_obj = (tick_obj.replace(day=28) + timedelta(days=4)).replace(day=1)
    else:
        tick_obj = start_obj + timedelta(days=7 - start_obj.weekday())
        while tick_obj <= end_obj:
            if (tick_obj - date_list[-1]).days >= 4:
                date_list.append(tick_obj)
            tick_obj += timedelta(days=7)
    if len(date_list) == 1 and end_obj != start_obj:
        date_list.append(end_obj)
    stride_int = max(1, math.ceil(len(date_list) / 9))
    date_list = date_list[::stride_int]
    # A late-month start and the next month's first day can be only one day
    # apart. Keep the real start, omit its near-neighbor before mobile thinning.
    if month_bool and start_obj.day != 1:
        while len(date_list) > 1 and (date_list[1] - start_obj).days / max(1, (end_obj - start_obj).days) < .20:
            date_list.pop(1)
    compact_stride_int = max(1, math.ceil(len(date_list) / 4))
    return [{"x_percent_float": max(0., min(100., (tick_obj.toordinal() - start_float) / max(.75, end_float - start_float) * 100)),
        "label_str": tick_obj.strftime("%b %y" if (end_obj - start_obj).days > 365 else "%b %d" if month_bool and index_int == 0 and tick_obj.day != 1 else "%b" if month_bool else "%m-%d"),
        "compact_bool": index_int % compact_stride_int == 0 or index_int == len(date_list) - 1,
        "first_bool": index_int == 0, "last_bool": index_int == len(date_list) - 1 and tick_obj == end_obj}
        for index_int, tick_obj in enumerate(date_list)]


def _drawing_dict(series_list, date_list, *, unit_str, bars_bool=False, pods_bool=False):
    result_dict = {"available_bool": False, "series_list": [], "tick_list": [], "x_tick_list": [], "bar_list": [],
        "height_int": 235 if pods_bool else 216, "width_int": WIDTH_INT, "bars_bool": bars_bool,
        "pods_bool": pods_bool, "baseline_y_float": None, "end_dict": None, "gutter_int": 40}
    all_point_list = [point_dict for series_dict in series_list for segment_list in series_dict["segment_list"] for point_dict in segment_list]
    if not all_point_list:
        return result_dict
    start_float, end_float = min(map(_date_float, date_list)), max(map(_date_float, date_list))
    tick_value_list, step_float = _scale_tuple([point_dict["value_float"] for point_dict in all_point_list], unit_str=unit_str, bars_bool=bars_bool)
    low_float, high_float = tick_value_list[0], tick_value_list[-1]
    height_int = result_dict["height_int"]
    bottom_float = height_int - 10.

    def vertical_float(value_float):
        return bottom_float - (value_float - low_float) / (high_float - low_float) * (height_int - 20)

    baseline_float = 100. if pods_bool else 0.
    baseline_y_float = vertical_float(baseline_float) if low_float <= baseline_float <= high_float else None
    area_y_float = baseline_y_float if baseline_y_float is not None else (10. if high_float < 0 else bottom_float)
    result_dict.update(available_bool=True, baseline_y_float=baseline_y_float,
        tick_list=[{"y_float": vertical_float(value_float), "y_percent_float": vertical_float(value_float) / height_int * 100,
            "value_float": value_float, "label_str": _axis_str(value_float, step_float, unit_str)} for value_float in reversed(tick_value_list)],
        x_tick_list=_x_ticks_list(start_float, end_float, bars_bool=bars_bool))
    result_dict["gutter_int"] = max(40, 7 * max(len(tick_dict["label_str"]) for tick_dict in result_dict["tick_list"]) + 5)
    for source_dict in series_list:
        series_dict = {"name_str": source_dict.get("name_str", ""), "color_str": source_dict.get("color_str", ""),
            "segment_list": [], "point_list": []}
        for segment_list in source_dict["segment_list"]:
            point_list = []
            for point_dict in segment_list:
                fraction_float = (_date_float(point_dict["market_date_str"]) - start_float) / max(.75, end_float - start_float)
                x_float = 10 + fraction_float * (WIDTH_INT - 20) if bars_bool else fraction_float * WIDTH_INT
                y_float = vertical_float(point_dict["value_float"])
                point_list.append({**point_dict, "x_float": x_float, "y_float": y_float,
                    "x_percent_float": x_float / WIDTH_INT * 100, "y_percent_float": y_float / height_int * 100,
                    "label_str": _value_str(point_dict["value_float"], unit_str), "isolated_bool": len(segment_list) == 1})
            if not point_list:
                continue
            point_str = " ".join(f"{point_dict['x_float']:.3f},{point_dict['y_float']:.3f}" for point_dict in point_list)
            series_dict["segment_list"].append({"point_str": point_str,
                "area_str": f"{point_list[0]['x_float']:.3f},{area_y_float:.3f} {point_str} {point_list[-1]['x_float']:.3f},{area_y_float:.3f}"})
            series_dict["point_list"].extend(point_list)
        result_dict["series_list"].append(series_dict)
    if bars_bool:
        point_list = result_dict["series_list"][0]["point_list"]
        width_float = min(18., WIDTH_INT / max(len(date_list), 1) * .72)
        for point_dict in point_list:
            if point_dict["value_float"] == 0:
                continue  # Zero has no bar height; do not invent a minimum gain.
            left_float, right_float = point_dict["x_float"] - width_float / 2, point_dict["x_float"] + width_float / 2
            end_y_float = point_dict["y_float"]
            radius_float = min(3., width_float / 2, abs(end_y_float - baseline_y_float))
            direction_int = 1 if end_y_float > baseline_y_float else -1
            corner_y_float = end_y_float - direction_int * radius_float
            path_str = (f"M{left_float:.3f},{baseline_y_float:.3f} V{corner_y_float:.3f} "
                f"Q{left_float:.3f},{end_y_float:.3f} {left_float + radius_float:.3f},{end_y_float:.3f} "
                f"H{right_float - radius_float:.3f} Q{right_float:.3f},{end_y_float:.3f} {right_float:.3f},{corner_y_float:.3f} "
                f"V{baseline_y_float:.3f} Z")
            result_dict["bar_list"].append({**point_dict, "path_str": path_str, "positive_bool": point_dict["value_float"] > 0})
    elif not pods_bool:
        end_dict = result_dict["series_list"][0]["point_list"][-1]
        result_dict["end_dict"] = dict(end_dict, detail_str=end_dict["label_str"])
        if unit_str == "usd" and abs(end_dict["value_float"]) >= 1_000:
            value_float = end_dict["value_float"]
            divisor_float, suffix_str = (1_000_000., "m") if abs(value_float) >= 1_000_000 else (1_000., "k")
            result_dict["end_dict"]["label_str"] = ("−" if value_float < 0 else "") + f"${abs(value_float) / divisor_float:.1f}" + suffix_str
    return result_dict


def build_history_chart_dict(daily_list, *, value_field_str="nav_float", unit_str="usd", bars_bool=False, daily_fact_list=None):
    """Keep V3 facts/geometry intact; attach the shared V4 drawing geometry."""
    source_dict = nav_chart_dict(daily_list, value_field_str=value_field_str, unit_str=unit_str,
        bars_bool=bars_bool, daily_fact_list=daily_fact_list)
    if source_dict is None:
        return None
    segment_list, current_list = [], []
    for row_dict in daily_list:
        if not _finite_bool(row_dict.get(value_field_str)):
            if current_list:
                segment_list.append(current_list)
                current_list = []
            continue
        current_list.append({"market_date_str": row_dict["market_date_str"], "value_float": row_dict[value_field_str]})
    if current_list:
        segment_list.append(current_list)
    source_dict["drawing_dict"] = _drawing_dict([{"segment_list": segment_list}],
        [row_dict["market_date_str"] for row_dict in daily_list], unit_str=unit_str, bars_bool=bars_bool)
    return source_dict


def add_pod_drawing_dict(chart_dict):
    """Reuse existing dated Pod values and their exact segment boundaries."""
    series_list, date_list = [], []
    for source_dict in chart_dict.get("series_list", []):
        # Dated points are evidence; rounded SVG coordinates are never identity.
        segment_list = source_dict["segment_point_list"]
        series_list.append({"segment_list": segment_list, "name_str": source_dict["name_str"], "color_str": source_dict["color_str"]})
        date_list.extend(point_dict["market_date_str"] for point_dict in source_dict["point_list"])
    return {**chart_dict, "drawing_dict": _drawing_dict(series_list, date_list, unit_str="index", pods_bool=True)}
