"""V4 chart drawing changes presentation, never canonical financial facts."""

from copy import deepcopy
from datetime import date, timedelta
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined
import pytest

from alpha.live.dashboard_v3.client_charts import nav_chart_dict
from alpha.live.dashboard_v4.charts import add_pod_drawing_dict, build_history_chart_dict
from alpha.live.dashboard_v4.finance import _chart_dict


def _daily_list(value_list, *, start_str="2026-06-01", step_int=1):
    return [{"market_date_str": (date.fromisoformat(start_str) + timedelta(days=index_int * step_int)).isoformat(), "nav_float": value_float}
        for index_int, value_float in enumerate(value_list)]


def _drawing_dict(value_list, **option_dict):
    return build_history_chart_dict(_daily_list(value_list), **option_dict)["drawing_dict"]


@pytest.mark.parametrize("value_list,unit_str,bars_bool", [([112668, 125480], "usd", False),
    ([-.001, .024, .050434], "pct", False), ([-349, 0, 170], "usd", True), ([0, 0], "pct", False)])
def test_canonical_chart_facts_remain_identical(value_list, unit_str, bars_bool):
    daily_list = _daily_list(value_list)
    before_list = deepcopy(daily_list)
    result_dict = build_history_chart_dict(daily_list, unit_str=unit_str, bars_bool=bars_bool)
    drawing_dict = result_dict.pop("drawing_dict")
    assert result_dict == nav_chart_dict(daily_list, unit_str=unit_str, bars_bool=bars_bool)
    assert daily_list == before_list
    assert drawing_dict["available_bool"]
    assert 5 <= len(drawing_dict["tick_list"]) <= 6
    assert all(0 <= point_dict["y_percent_float"] <= 100 for series_dict in drawing_dict["series_list"] for point_dict in series_dict["point_list"])


def test_round_grid_does_not_use_raw_extremes_and_keeps_values_in_range():
    result_dict = _drawing_dict([112668.2, 125480.12])
    value_list = [tick_dict["value_float"] for tick_dict in result_dict["tick_list"]]
    assert max(value_list) >= 125480.12
    assert min(value_list) <= 112668.2
    assert 112668.2 not in value_list and 125480.12 not in value_list
    assert all(tick_dict["label_str"].endswith("k") for tick_dict in result_dict["tick_list"])
    assert result_dict["gutter_int"] == 40
    assert result_dict["end_dict"]["label_str"] == "$125.5k"
    assert result_dict["end_dict"]["detail_str"] == "$125,480.12"


def test_million_dollar_end_label_is_compact_but_exact_value_is_preserved():
    drawing_dict = _drawing_dict([1_200_000., 1_234_567.89])
    assert drawing_dict["end_dict"]["label_str"] == "$1.2m"
    assert drawing_dict["end_dict"]["detail_str"] == "$1,234,567.89"
    assert drawing_dict["end_dict"]["value_float"] == 1_234_567.89


def test_breaks_and_isolated_points_are_not_connected_or_filled():
    source_dict = build_history_chart_dict(_daily_list([1., None, 2., 3., None, 4., None]))
    drawing_dict = source_dict["drawing_dict"]
    series_dict = drawing_dict["series_list"][0]
    assert len(series_dict["segment_list"]) == 3
    assert [point_dict["isolated_bool"] for point_dict in series_dict["point_list"]] == [True, False, False, True]
    assert drawing_dict["end_dict"]["market_date_str"] == "2026-06-06"
    assert drawing_dict["end_dict"]["x_percent_float"] < 100
    assert all("nan" not in segment_dict["point_str"] for segment_dict in series_dict["segment_list"])


def test_zero_and_negative_bars_keep_signed_height_and_rounded_data_ends():
    drawing_dict = _drawing_dict([-349, 0, 170], bars_bool=True)
    assert [tick_dict["value_float"] for tick_dict in drawing_dict["tick_list"]] == [400., 200., 0., -200., -400.]
    assert len(drawing_dict["bar_list"]) == 2
    negative_dict, positive_dict = drawing_dict["bar_list"]
    assert negative_dict["y_float"] > drawing_dict["baseline_y_float"] > positive_dict["y_float"]
    assert not negative_dict["positive_bool"] and positive_dict["positive_bool"]
    assert all(bar_dict["path_str"].count("Q") == 2 for bar_dict in drawing_dict["bar_list"])
    assert all(0 < bar_dict["x_float"] < drawing_dict["width_int"] for bar_dict in drawing_dict["bar_list"])
    zero_dict = _drawing_dict([0, 0, 0], bars_bool=True)
    assert zero_dict["bar_list"] == []
    assert zero_dict["baseline_y_float"] is not None


def test_month_axis_and_weekly_bar_axis():
    line_dict = build_history_chart_dict(_daily_list(list(range(130))))["drawing_dict"]
    assert [tick_dict["label_str"] for tick_dict in line_dict["x_tick_list"]] == ["Jun", "Jul", "Aug", "Sep", "Oct"]
    bar_dict = build_history_chart_dict(_daily_list(list(range(30))), bars_bool=True)["drawing_dict"]
    assert [tick_dict["label_str"] for tick_dict in bar_dict["x_tick_list"]] == ["06-01", "06-08", "06-15", "06-22", "06-29"]
    assert sum(tick_dict["compact_bool"] for tick_dict in bar_dict["x_tick_list"]) <= 4


def test_sod_and_same_day_close_remain_distinct_points():
    source_list = [{"market_date_str": "2026-06-01 SOD", "nav_float": 100}, {"market_date_str": "2026-06-01", "nav_float": 90}]
    point_list = build_history_chart_dict(source_list)["drawing_dict"]["series_list"][0]["point_list"]
    assert point_list[0]["x_float"] == 0
    assert point_list[1]["x_float"] == 800
    assert point_list[0]["value_float"] == 100 and point_list[1]["value_float"] == 90


@pytest.mark.parametrize("start_str, label_str", [("2026-07-31", "Jul 31"), ("2026-07-24", "Jul 24")])
def test_late_month_start_does_not_overlap_the_next_month_on_phone(start_str, label_str):
    source_list = [{"market_date_str": start_str + " SOD", "nav_float": 0},
        {"market_date_str": "2026-09-18", "nav_float": .05}]
    tick_list = build_history_chart_dict(source_list, unit_str="pct")["drawing_dict"]["x_tick_list"]
    assert [tick_dict["label_str"] for tick_dict in tick_list] == [label_str, "Sep"]
    assert tick_list[1]["x_percent_float"] - tick_list[0]["x_percent_float"] >= 20
    compact_list = [tick_dict for tick_dict in tick_list if tick_dict["compact_bool"]]
    assert compact_list == tick_list


def test_pod_series_use_dates_not_rounded_coordinates_and_preserve_segments():
    first_list = [{"market_date_str": "2026-06-01 SOD", "value_float": 100., "x_float": 0., "y_float": 100.},
        {"market_date_str": "2026-06-01", "value_float": 100.001, "x_float": .001, "y_float": 100.001}]
    last_list = [{"market_date_str": "2026-09-01", "value_float": 110., "x_float": 800., "y_float": 10.}]
    chart_dict = {"available_bool": True, "series_list": [{"name_str": "Pod A", "color_str": "#2a78d6",
        "point_list": first_list + last_list, "segment_list": ["0.00,100.00 0.00,100.00", "800.00,10.00"],
        "segment_point_list": [first_list, last_list]}]}
    before_dict = deepcopy(chart_dict)
    drawing_dict = add_pod_drawing_dict(chart_dict)["drawing_dict"]
    assert chart_dict == before_dict
    assert drawing_dict["height_int"] == 235
    assert drawing_dict["baseline_y_float"] is not None
    series_dict = drawing_dict["series_list"][0]
    assert len(series_dict["segment_list"]) == 2
    assert [point_dict["value_float"] for point_dict in series_dict["point_list"]] == [100., 100.001, 110.]
    assert series_dict["point_list"][0]["x_float"] < series_dict["point_list"][1]["x_float"]
    assert drawing_dict["end_dict"] is None


def test_unavailable_drawing_stays_unavailable():
    assert build_history_chart_dict([]) is None
    assert build_history_chart_dict(_daily_list([None, None])) is None
    assert not add_pod_drawing_dict({"available_bool": False, "series_list": []})["drawing_dict"]["available_bool"]


def test_overview_and_pod_legacy_contract_includes_shared_drawing():
    chart_dict = _chart_dict([{ "market_date_str": "2026-06-01 SOD", "cumulative_return_float": 0.},
        {"market_date_str": "2026-06-01", "cumulative_return_float": -.05}])
    assert chart_dict["end_x_float"] == 496  # retained old returned-dict contract
    assert chart_dict["drawing_dict"]["end_dict"]["value_float"] == -.05
    assert chart_dict["drawing_dict"]["height_int"] == 216


def test_shared_template_uses_unscaled_html_labels_and_soft_area():
    template_root_obj = Path(__file__).parents[1] / "alpha/live/dashboard_v4/templates"
    environment_obj = Environment(loader=FileSystemLoader(template_root_obj), undefined=StrictUndefined)
    macro_obj = environment_obj.get_template("_chart.html").module.financial_chart
    html_str = macro_obj(_drawing_dict([0, .051], unit_str="pct"), "Portfolio return")
    assert 'class="area"' in html_str
    assert 'class="v4-chart-end"' in html_str
    assert 'class="v4-chart-yaxis"' in html_str
    assert '<text' not in html_str
    assert 'preserveAspectRatio="none"' in html_str
    assert "+5.10%" in html_str
