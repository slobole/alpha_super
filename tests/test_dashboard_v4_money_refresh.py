"""Money panels leave the 15 s frame refresh: kept in place, refreshed by their own route."""

from html.parser import HTMLParser

import pytest

from alpha.live.dashboard_v4 import app as app_module
from alpha.live.dashboard_v4 import overview as overview_module
from alpha.live.dashboard_v4.demo import create_demo_app


class _ElementParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.element_dict = {}

    def handle_starttag(self, tag_str, attribute_list):
        attribute_dict = dict(attribute_list)
        if attribute_dict.get("id"):
            self.element_dict[attribute_dict["id"]] = attribute_dict


def _element_dict(html_str):
    parser_obj = _ElementParser()
    parser_obj.feed(html_str)
    return parser_obj.element_dict


def _pod_block_tuple(html_str, mode_str):
    match_list = [(id_str, attribute_dict) for id_str, attribute_dict in _element_dict(html_str).items()
        if id_str.startswith("pod-money-" + mode_str + "-")]
    assert len(match_list) == 1, match_list
    return match_list[0]


@pytest.fixture(scope="module")
def client_obj():
    return create_demo_app().test_client()


def _forbidden(*args, **kwargs):
    raise AssertionError("A frame refresh must not rebuild money panels")


def test_overview_refresh_sends_preserved_placeholders_and_builds_no_finance(client_obj, monkeypatch):
    page_str = client_obj.get("/").get_data(as_text=True)
    monkeypatch.setattr(overview_module, "build_financial_overview_dict", _forbidden)
    refresh_str = client_obj.get("/overview/refresh?period=All").get_data(as_text=True)
    element_dict = _element_dict(refresh_str)
    for id_str in ("overview-money-tiles", "overview-money-asof", "overview-money-panels"):
        assert element_dict[id_str]["hx-preserve"] == "true"
    assert "data-own-placeholder" in element_dict["overview-money-tiles"]
    assert element_dict["overview-money-tiles"]["hx-get"] == "/overview/money?period=All"
    assert element_dict["overview-money-tiles"]["hx-trigger"] == "v4own"
    assert element_dict["overview-money-tiles"]["data-own-refresh-ms"] == "300000"
    # Explicit, or htmx inherits the frame's hx-target and replaces the whole frame.
    assert element_dict["overview-money-tiles"]["hx-target"] == "this"
    assert "data-history-chart" not in refresh_str and "Portfolio return" not in refresh_str
    assert len(refresh_str) < 0.4 * len(page_str)


def test_full_page_renders_money_without_preserve(client_obj):
    element_dict = _element_dict(client_obj.get("/").get_data(as_text=True))
    for id_str in ("overview-money-tiles", "overview-money-asof", "overview-money-panels"):
        assert "hx-preserve" not in element_dict[id_str] and "hx-swap-oob" not in element_dict[id_str]
    assert "data-own-placeholder" not in element_dict["overview-money-tiles"]


def test_overview_money_route_replaces_tiles_and_sends_the_rest_out_of_band(client_obj):
    money_str = client_obj.get("/overview/money?period=3M").get_data(as_text=True)
    element_dict = _element_dict(money_str)
    assert "hx-swap-oob" not in element_dict["overview-money-tiles"]
    assert element_dict["overview-money-tiles"]["hx-get"] == "/overview/money?period=3M"
    assert element_dict["overview-money-asof"]["hx-swap-oob"] == "outerHTML"
    assert element_dict["overview-money-panels"]["hx-swap-oob"] == "outerHTML"
    assert all("hx-preserve" not in attribute_dict for attribute_dict in element_dict.values())
    assert "Portfolio return" in money_str and "overview-shell" not in money_str
    assert client_obj.get("/overview/money?period=bad").status_code == 400
    assert client_obj.post("/overview/money").status_code == 403


@pytest.mark.parametrize("pod_id_str,mode_str", [("demo_1_0", "normal"), ("demo_1_1", "issue")])
def test_pod_refresh_keeps_a_mode_specific_money_block(client_obj, monkeypatch, pod_id_str, mode_str):
    page_id_str, page_dict = _pod_block_tuple(client_obj.get("/pods/" + pod_id_str).get_data(as_text=True), mode_str)
    assert "hx-preserve" not in page_dict
    monkeypatch.setattr(app_module, "build_pod_finance_dict", _forbidden)
    refresh_str = client_obj.get(f"/pods/{pod_id_str}/refresh").get_data(as_text=True)
    block_id_str, block_dict = _pod_block_tuple(refresh_str, mode_str)
    assert block_id_str == page_id_str  # Same id, so htmx keeps the block in place.
    assert block_dict["hx-preserve"] == "true" and "data-own-placeholder" in block_dict
    assert block_dict["hx-get"] == f"/pods/{pod_id_str}/money?period=All" and block_dict["hx-target"] == "this"
    assert "data-pod-allocation" not in refresh_str


@pytest.mark.parametrize("pod_id_str,mode_str", [("demo_1_0", "normal"), ("demo_1_1", "issue")])
def test_pod_money_route_returns_only_the_block(client_obj, pod_id_str, mode_str):
    money_str = client_obj.get(f"/pods/{pod_id_str}/money").get_data(as_text=True)
    assert "hx-preserve" not in _pod_block_tuple(money_str, mode_str)[1]
    assert "overview-shell" not in money_str and "Pod money at last close" in money_str
    assert ("pod-money-fold" in money_str) is (mode_str == "issue")
    assert client_obj.get("/pods/unknown_pod/money").status_code == 404


def test_historical_cycle_money_block_matches_its_refresh_and_keeps_links(client_obj):
    # The latest demo_1_1 cycle is an issue; vplan:1 is an older, normal cycle.
    query_str = "?period=1M&cycle=vplan:1&tab=events"
    page_str = client_obj.get("/pods/demo_1_1" + query_str).get_data(as_text=True)
    page_id_str, page_dict = _pod_block_tuple(page_str, "normal")
    assert page_dict["hx-get"] == "/pods/demo_1_1/money?period=1M&cycle=vplan:1&tab=events"
    refresh_id_str, _ = _pod_block_tuple(client_obj.get("/pods/demo_1_1/refresh" + query_str).get_data(as_text=True), "normal")
    money_str = client_obj.get(page_dict["hx-get"]).get_data(as_text=True)
    money_id_str, _ = _pod_block_tuple(money_str, "normal")
    assert page_id_str == refresh_id_str == money_id_str
    assert 'href="/pods/demo_1_1?period=3M&amp;cycle=vplan:1&amp;tab=events"' in money_str


def test_new_saved_pod_state_changes_the_block_id_so_it_reloads():
    from alpha.live.dashboard_v4.app import create_app
    from alpha.live.dashboard_v4.demo import DEMO_NOW_TS, build_demo_workspace_tuple
    workspace_dict, snapshot_obj, provider_obj = build_demo_workspace_tuple()
    app_obj = create_app(provider_obj, demo_bool=True, now_fn=lambda: DEMO_NOW_TS,
        workspace_snapshot_fn=lambda: (workspace_dict, snapshot_obj))
    try:
        client_obj = app_obj.test_client()
        before_id_str, _ = _pod_block_tuple(client_obj.get("/pods/demo_1_0/refresh").get_data(as_text=True), "normal")
        row_dict = next(item_dict for item_dict in workspace_dict["summary_dict"]["pod_row_dict_list"]
            if item_dict["pod_id_str"] == "demo_1_0")
        row_dict["latest_pod_state_timestamp_str"] = "2026-09-08T13:40:00+00:00"  # fills saved
        after_id_str, _ = _pod_block_tuple(client_obj.get("/pods/demo_1_0/refresh").get_data(as_text=True), "normal")
        assert before_id_str != after_id_str
    finally:
        provider_obj.close()


def test_activity_timeline_says_when_its_data_is_from(client_obj):
    html_str = client_obj.get("/activity/body?days=7").get_data(as_text=True)
    assert 'data-own-status data-own-updated="' in html_str and "· updated " in html_str
