"""Every history chart exposes its saved observations to the shared interaction UI."""

from html.parser import HTMLParser
import json

import pytest
from flask import template_rendered

from alpha.live.dashboard_v4.app import create_app
from alpha.live.dashboard_v4.demo import DEMO_NOW_TS, build_demo_workspace_tuple


class ChartParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.chart_list = []
        self.day_list = []

    def handle_starttag(self, tag_str, attribute_list):
        attribute_dict = dict(attribute_list)
        if "data-history-chart" in attribute_dict:
            self.chart_list.append(attribute_dict)
        if "data-chart-day" in attribute_dict:
            self.day_list.append(attribute_dict)


@pytest.fixture(scope="module")
def chart_app_obj():
    workspace_dict, snapshot_obj, provider_obj = build_demo_workspace_tuple()
    app_obj = create_app(provider_obj, demo_bool=True, now_fn=lambda: DEMO_NOW_TS,
        workspace_snapshot_fn=lambda: (workspace_dict, snapshot_obj))
    app_obj.config["TESTING"] = True
    try:
        yield app_obj
    finally:
        provider_obj.close()


@pytest.mark.parametrize("asset_str,marker_str,mimetype_str", [
    ("chart_interaction.js", "data-history-chart", "javascript"),
    ("donut_interaction.js", "data-donut-inspector", "javascript"),
    ("donut_interaction.css", ".donut-inspector", "text/css"),
])
def test_interaction_assets_are_served(chart_app_obj, asset_str, marker_str, mimetype_str):
    response_obj = chart_app_obj.test_client().get(f"/static/{asset_str}")
    assert response_obj.status_code == 200
    assert mimetype_str in response_obj.mimetype
    assert marker_str in response_obj.get_data(as_text=True)


@pytest.mark.parametrize("path_str,context_key_str,chart_key_str", [
    ("/", "overview_dict", "chart_dict"),
    ("/pods/demo_1_0", "pod_page_dict", "chart_dict"),
    ("/performance", "performance_page_dict", "chart_dict"),
    ("/performance?unit=usd", "performance_page_dict", "chart_dict"),
    ("/performance?level=pods", "performance_page_dict", "pod_chart_dict"),
])
def test_each_history_chart_uses_exact_saved_points_and_one_keyboard_entry(
        chart_app_obj, path_str, context_key_str, chart_key_str):
    context_list = []

    def capture_context(sender_obj, **signal_dict):
        context_list.append(signal_dict["context"])

    with template_rendered.connected_to(capture_context, chart_app_obj):
        response_obj = chart_app_obj.test_client().get(path_str)
    assert response_obj.status_code == 200
    html_str = response_obj.get_data(as_text=True)
    parser_obj = ChartParser()
    parser_obj.feed(html_str)
    assert len(parser_obj.chart_list) == 1
    assert "/static/chart_interaction.js" in html_str
    assert "/static/donut_interaction.js" in html_str
    drawing_dict = context_list[-1][context_key_str][chart_key_str]["drawing_dict"]
    assert len(parser_obj.day_list) == len(drawing_dict["interaction_list"])
    assert sum(day_dict["tabindex"] == "0" for day_dict in parser_obj.day_list) == 1
    assert parser_obj.chart_list[0]["data-chart-default"] == drawing_dict["default_dict"]["date_str"]
    for attribute_dict, day_dict in zip(parser_obj.day_list, drawing_dict["interaction_list"], strict=True):
        assert attribute_dict["data-chart-day"] == day_dict["date_str"]
        assert json.loads(attribute_dict["data-chart-values"]) == day_dict["value_list"]
    if "level=pods" in path_str:
        assert "indexed to 100" in parser_obj.chart_list[0]["aria-label"]
        assert len(drawing_dict["default_dict"]["value_list"]) > 1
        assert [value_dict["color_str"] for value_dict in drawing_dict["default_dict"]["value_list"]] == [
            series_dict["color_str"] for series_dict in drawing_dict["series_list"]]


def test_status_refresh_does_not_replace_financial_charts(chart_app_obj):
    response_obj = chart_app_obj.test_client().get("/performance/status")
    assert response_obj.status_code == 200
    html_str = response_obj.get_data(as_text=True)
    assert "data-history-chart" not in html_str
    assert "data-chart-day" not in html_str


@pytest.mark.parametrize("path_str", ["/overview/refresh", "/pods/demo_1_0/refresh"])
def test_full_page_refresh_keeps_chart_selection_contract(chart_app_obj, path_str):
    response_obj = chart_app_obj.test_client().get(path_str)
    assert response_obj.status_code == 200
    parser_obj = ChartParser()
    parser_obj.feed(response_obj.get_data(as_text=True))
    assert parser_obj.chart_list and parser_obj.day_list
