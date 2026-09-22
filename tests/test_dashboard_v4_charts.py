"""V4 chart drawing changes presentation, never canonical financial facts."""

from copy import deepcopy
from datetime import date, timedelta
from html.parser import HTMLParser
import json
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


def _render_str(drawing_dict, label_str="Portfolio return", daily_dict=None):
    template_root_obj = Path(__file__).parents[1] / "alpha/live/dashboard_v4/templates"
    environment_obj = Environment(loader=FileSystemLoader(template_root_obj), undefined=StrictUndefined, autoescape=True)
    return environment_obj.get_template("_chart.html").module.financial_chart(drawing_dict, label_str, daily_dict)


def _elements_list(html_str):
    class ChartParser(HTMLParser):
        def handle_starttag(self, tag_str, attribute_list):
            element_list.append((tag_str, dict(attribute_list)))

    element_list = []
    ChartParser().feed(html_str)
    return element_list


@pytest.mark.parametrize("unit_str,value_float,label_str,name_str", [
    ("usd", 1234567.891, "$1,234,567.89", "Value"),
    ("usd", -10.126, "−$10.13", "Value"),
    ("pct", .1234567, "+12.35%", "Return"),
    ("index", 112.34567, "112.35", "Index")])
def test_interaction_retains_exact_saved_point_and_existing_unit_format(unit_str, value_float, label_str, name_str):
    drawing_dict = _drawing_dict([value_float], unit_str=unit_str)
    day_dict = drawing_dict["default_dict"]
    point_dict = drawing_dict["series_list"][0]["point_list"][0]
    value_dict = day_dict["value_list"][0]
    assert point_dict["value_float"] == value_float
    assert value_dict == {"name_str": name_str, "color_str": "var(--blue)", "label_str": label_str,
        "x_percent_float": point_dict["x_percent_float"], "y_percent_float": point_dict["y_percent_float"], "available_bool": True}
    assert day_dict["hit_left_float"] == 0 and day_dict["hit_width_float"] == 100


def test_all_saved_dates_remain_selectable_including_internal_and_latest_gaps():
    drawing_dict = _drawing_dict([10., None, 20., None])
    day_list = drawing_dict["interaction_list"]
    assert [day_dict["date_str"] for day_dict in day_list] == ["2026-06-01", "2026-06-02", "2026-06-03", "2026-06-04"]
    assert [day_dict["value_list"][0]["label_str"] for day_dict in day_list] == ["$10.00", "—", "$20.00", "—"]
    assert drawing_dict["default_dict"] == day_list[-1]
    assert drawing_dict["end_dict"]["market_date_str"] == "2026-06-03"
    assert len(drawing_dict["series_list"][0]["segment_list"]) == 2
    for day_dict in (day_list[1], day_list[3]):
        assert day_dict["value_list"][0]["available_bool"] is False
        assert day_dict["value_list"][0]["y_percent_float"] is None
    assert sum(day_dict["hit_width_float"] for day_dict in day_list) == pytest.approx(100.)
    for left_dict, right_dict in zip(day_list, day_list[1:]):
        assert left_dict["hit_left_float"] + left_dict["hit_width_float"] == pytest.approx(right_dict["hit_left_float"])


def test_interaction_sorts_unique_date_keys_without_merging_start_and_close():
    daily_list = [{"market_date_str": "2026-06-01 SOD", "nav_float": 100.},
        {"market_date_str": "2026-06-01", "nav_float": 90.}]
    drawing_dict = build_history_chart_dict(daily_list)["drawing_dict"]
    start_dict, close_dict = drawing_dict["interaction_list"]
    assert start_dict["date_label_str"] == "2026-06-01 · Start"
    assert close_dict["date_label_str"] == "2026-06-01"
    assert start_dict["x_percent_float"] == 0 and close_dict["x_percent_float"] == 100
    assert start_dict["value_list"][0]["label_str"] == "$100.00"
    assert close_dict["value_list"][0]["label_str"] == "$90.00"
    assert start_dict["hit_width_float"] == close_dict["hit_width_float"] == 50


def test_pod_date_union_preserves_individual_gaps_and_full_date_input():
    first_list = [{"market_date_str": "2026-06-01 SOD", "value_float": 100.},
        {"market_date_str": "2026-06-03", "value_float": 110.}]
    second_list = [{"market_date_str": "2026-06-02", "value_float": 100.},
        {"market_date_str": "2026-06-03", "value_float": 95.}]
    chart_dict = {"date_list": ["2026-06-04", "2026-06-03", "2026-06-02"], "series_list": [
        {"name_str": "Long Pod A name", "color_str": "#228855", "point_list": first_list, "segment_point_list": [[first_list[0]], [first_list[1]]]},
        {"name_str": "Pod B", "color_str": "#225588", "point_list": second_list, "segment_point_list": [second_list]}]}
    before_dict = deepcopy(chart_dict)
    drawing_dict = add_pod_drawing_dict(chart_dict)["drawing_dict"]
    assert chart_dict == before_dict
    day_list = drawing_dict["interaction_list"]
    assert [day_dict["date_str"] for day_dict in day_list] == ["2026-06-01 SOD", "2026-06-02", "2026-06-03", "2026-06-04"]
    assert [[value_dict["label_str"] for value_dict in day_dict["value_list"]] for day_dict in day_list] == [
        ["100.00", "—"], ["—", "100.00"], ["110.00", "95.00"], ["—", "—"]]
    assert len(drawing_dict["series_list"][0]["segment_list"]) == 2
    assert [value_dict["color_str"] for value_dict in day_list[-1]["value_list"]] == ["#228855", "#225588"]
    assert [value_dict["name_str"] for value_dict in day_list[-1]["value_list"]] == ["Long Pod A name", "Pod B"]


def test_line_markup_has_exact_date_columns_reusable_markers_and_safe_series_names():
    name_str = 'Pod "A" <script>alert(1)</script> & a very long identity'
    point_list = [{"market_date_str": "2026-06-01 SOD", "value_float": 100.},
        {"market_date_str": "2026-06-02", "value_float": 105.}]
    chart_dict = {"series_list": [{"name_str": name_str, "color_str": "#228855", "point_list": point_list, "segment_point_list": [point_list]}]}
    drawing_dict = add_pod_drawing_dict(chart_dict)["drawing_dict"]
    html_str = _render_str(drawing_dict, 'Return by pod "selected"')
    element_list = _elements_list(html_str)
    root_dict = next(attribute_dict for _, attribute_dict in element_list if "data-history-chart" in attribute_dict)
    assert root_dict["role"] == "group"
    assert root_dict["data-chart-id"] == 'Return by pod "selected"'
    assert root_dict["data-chart-default"] == "2026-06-02"
    button_list = [attribute_dict for tag_str, attribute_dict in element_list if tag_str == "button"]
    assert [attribute_dict["tabindex"] for attribute_dict in button_list] == ["-1", "0"]
    assert [attribute_dict["data-chart-day"] for attribute_dict in button_list] == ["2026-06-01 SOD", "2026-06-02"]
    for button_dict, day_dict in zip(button_list, drawing_dict["interaction_list"]):
        assert json.loads(button_dict["data-chart-values"]) == day_dict["value_list"]
        assert button_dict["aria-label"] == day_dict["date_label_str"] + " · " + name_str + " " + day_dict["value_list"][0]["label_str"]
        assert float(button_dict["data-chart-x"]) == day_dict["x_percent_float"]
    marker_list = [attribute_dict for _, attribute_dict in element_list if "data-chart-marker" in attribute_dict]
    assert len(marker_list) == 1 and marker_list[0]["data-series-index"] == "0" and "hidden" in marker_list[0]
    assert any("data-chart-crosshair" in attribute_dict and "hidden" in attribute_dict for _, attribute_dict in element_list)
    assert any("data-chart-value" in attribute_dict and attribute_dict["data-series-index"] == "0" for _, attribute_dict in element_list)
    assert not any(tag_str == "script" for tag_str, _ in element_list)
    assert not any(attribute_dict.get("class") == "chart-hit" for _, attribute_dict in element_list)
    assert any(attribute_dict.get("class") == "v4-chart-dot" for _, attribute_dict in element_list)
    assert 'data-chart-number>105.00' in html_str


def test_empty_and_bar_drawing_do_not_add_line_interaction_markup():
    empty_dict = add_pod_drawing_dict({"series_list": []})["drawing_dict"]
    assert empty_dict["interaction_list"] == [] and empty_dict["default_dict"] is None
    assert "data-history-chart" not in _render_str(empty_dict)
    drawing_dict = _drawing_dict([-12.3, 20.], bars_bool=True)
    assert drawing_dict["interaction_list"] == [] and drawing_dict["default_dict"] is None
    day_list = [{"date_str": "2026-06-01", "date_label_str": "2026-06-01", "pnl_str": "−$12.30", "return_str": "−1.00%",
        "tone_str": "negative", "readout_str": "2026-06-01 · −$12.30 · −1.00%", "hit_left_float": 0., "hit_width_float": 50., "extreme_bool": False},
        {"date_str": "2026-06-02", "date_label_str": "2026-06-02", "pnl_str": "+$20.00", "return_str": "+2.00%",
        "tone_str": "positive", "readout_str": "2026-06-02 · +$20.00 · +2.00%", "hit_left_float": 50., "hit_width_float": 50., "extreme_bool": False}]
    html_str = _render_str(drawing_dict, "Daily P&L", {"day_list": day_list, "last_dict": day_list[-1]})
    assert "data-history-chart" not in html_str and "data-chart-day" not in html_str
    button_list = [attribute_dict for tag_str, attribute_dict in _elements_list(html_str) if tag_str == "button"]
    assert len(button_list) == 2
    for button_dict, day_dict in zip(button_list, day_list):
        assert button_dict["data-daily-day"] == day_dict["date_str"]
        assert button_dict["data-pnl"] == day_dict["pnl_str"]
        assert button_dict["data-return"] == day_dict["return_str"]
        assert button_dict["data-tone"] == day_dict["tone_str"]
        assert button_dict["aria-label"] == day_dict["readout_str"]
    assert [button_dict["tabindex"] for button_dict in button_list] == ["-1", "0"]
