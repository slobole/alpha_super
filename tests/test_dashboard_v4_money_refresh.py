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
    page_dict = _element_dict(client_obj.get("/pods/" + pod_id_str).get_data(as_text=True))
    assert "hx-preserve" not in page_dict["pod-money-" + mode_str]
    monkeypatch.setattr(app_module, "build_pod_finance_dict", _forbidden)
    refresh_str = client_obj.get(f"/pods/{pod_id_str}/refresh").get_data(as_text=True)
    block_dict = _element_dict(refresh_str)["pod-money-" + mode_str]
    assert block_dict["hx-preserve"] == "true" and "data-own-placeholder" in block_dict
    assert block_dict["hx-get"] == f"/pods/{pod_id_str}/money?period=All" and block_dict["hx-target"] == "this"
    assert "data-pod-allocation" not in refresh_str


@pytest.mark.parametrize("pod_id_str,mode_str", [("demo_1_0", "normal"), ("demo_1_1", "issue")])
def test_pod_money_route_returns_only_the_block(client_obj, pod_id_str, mode_str):
    money_str = client_obj.get(f"/pods/{pod_id_str}/money").get_data(as_text=True)
    element_dict = _element_dict(money_str)
    assert "hx-preserve" not in element_dict["pod-money-" + mode_str]
    assert "overview-shell" not in money_str and "Pod money at last close" in money_str
    assert ("pod-money-fold" in money_str) is (mode_str == "issue")
    assert client_obj.get("/pods/unknown_pod/money").status_code == 404
