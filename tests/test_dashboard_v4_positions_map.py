"""Positions map data: only a complete valuation is drawn; labels and exposure are explicit."""

import json
from html import unescape
import re

import pytest

from alpha.live.dashboard_v4.app import create_app
from alpha.live.dashboard_v4.demo import DEMO_NOW_TS, build_demo_workspace_tuple
from alpha.live.dashboard_v4.positions import build_positions_page_dict
from alpha.live.dashboard_v4.positions_map import (
    build_now_map_dict, compact_money_str, exposure_multiplier_float, now_side_dict, signed_percent_str,
)


IDENTITY_LIST = [{"pod_id_str": "a", "name_str": "Alpha", "color_str": "#111111"},
    {"pod_id_str": "b", "name_str": "Beta", "color_str": "#222222"}]


def _row(symbol_str, shares_float, value_float, pnl_float, cost_float):
    return {"symbol_str": symbol_str, "shares_float": shares_float, "shares_str": str(shares_float), "value_float": value_float,
        "value_str": f"${value_float:,.2f}", "pnl_float": pnl_float, "cost_float": cost_float,
        "pnl_str": "—" if pnl_float is None else f"{pnl_float:+.2f}"}


SOURCE_LIST = [
    {"pod_id_str": "a", "cash_float": 500.0, "row_list": [_row("TQQQ", 10, 900.0, 90.0, 810.0), _row("AAA", 5, 600.0, -60.0, 660.0)]},
    {"pod_id_str": "b", "cash_float": 0.0, "row_list": [_row("BBB", -4, -400.0, 20.0, 420.0), _row("ZERO", 0, 0.0, 0.0, 0.0)]},
]


def test_formatting_and_exposure_rules():
    assert compact_money_str(1234.4) == "$1,234"
    assert compact_money_str(-12.4, signed_bool=True) == "−$12"
    assert compact_money_str(12.6, signed_bool=True) == "+$13"
    assert compact_money_str(None) == "—"
    assert signed_percent_str(0.0414) == "+4.1%" and signed_percent_str(-0.0001) == "0.0%" and signed_percent_str(None) == "—"
    assert exposure_multiplier_float("TQQQ") == 3.0 and exposure_multiplier_float("SQQQ") == -3.0 and exposure_multiplier_float("AAPL") == 1.0


def test_incomplete_valuation_draws_nothing():
    for source_list, book_float in (([], 2000.0), (SOURCE_LIST, None), (SOURCE_LIST, 0.0)):
        map_dict = build_now_map_dict(source_list, IDENTITY_LIST, pod_str="all", book_value_float=book_float, asof_str="x")
        assert map_dict["available_bool"] is False and map_dict["pod_list"] == []
        assert "every Pod" in map_dict["reason_str"]
        assert now_side_dict(map_dict) == {}


def test_complete_valuation_builds_tiles_with_exposure_shorts_and_cash():
    map_dict = build_now_map_dict(SOURCE_LIST, IDENTITY_LIST, pod_str="all", book_value_float=1600.0, asof_str="IBKR values")
    assert map_dict["available_bool"] is True and map_dict["mode_str"] == "now"
    alpha_dict, beta_dict = map_dict["pod_list"]
    assert alpha_dict["nav_float"] == 2000.0 and alpha_dict["cash_float"] == 500.0
    assert alpha_dict["pnl_float"] == 30.0 and alpha_dict["return_str"] == "+2.0%"
    tqqq_dict = alpha_dict["item_list"][0]
    assert tqqq_dict["key_str"] == "TQQQ|a" and tqqq_dict["multiplier_float"] == 3.0
    assert ["Exposure", "$2,700 (3x)"] in tqqq_dict["tip_list"]
    assert ["Weight in book", "56.2%"] in tqqq_dict["tip_list"]
    # Shorts size by absolute value; a zero-share row is not a tile.
    assert [item_dict["symbol_str"] for item_dict in beta_dict["item_list"]] == ["BBB"]
    assert beta_dict["item_list"][0]["value_float"] == 400.0 and beta_dict["item_list"][0]["status_str"] == "short"
    assert ["Side", "Short"] in beta_dict["item_list"][0]["tip_list"]


def test_pod_filter_and_side_column():
    map_dict = build_now_map_dict(SOURCE_LIST, IDENTITY_LIST, pod_str="a", book_value_float=1600.0, asof_str="x")
    assert [pod_dict["pod_id_str"] for pod_dict in map_dict["pod_list"]] == ["a"]
    side_dict = now_side_dict(map_dict)
    assert side_dict["big_str"] == "+$30" and "+2.0% on cost" in side_dict["big_detail_str"]
    assert [mover_dict["symbol_str"] for mover_dict in side_dict["best_list"]] == ["TQQQ"]
    assert [mover_dict["symbol_str"] for mover_dict in side_dict["worst_list"]] == ["AAA"]
    assert side_dict["pod_bar_list"][0]["width_percent_float"] == 50.0


def test_unknown_pnl_keeps_tiles_but_says_pnl_is_not_verified():
    source_list = [{"pod_id_str": "a", "cash_float": 0.0, "row_list": [_row("AAA", 5, 600.0, None, None)]}]
    map_dict = build_now_map_dict(source_list, IDENTITY_LIST, pod_str="all", book_value_float=600.0, asof_str="x")
    assert map_dict["available_bool"] is True and map_dict["pod_list"][0]["pnl_str"] == ""
    assert map_dict["pod_list"][0]["item_list"][0]["return_float"] is None
    assert now_side_dict(map_dict)["big_detail_str"] == "Open P&L not verified for every Pod"


@pytest.fixture(scope="module")
def demo_app():
    workspace_dict, snapshot_obj, provider_obj = build_demo_workspace_tuple(include_holdings_bool=True)
    app_obj = create_app(provider_obj, demo_bool=True, now_fn=lambda: DEMO_NOW_TS,
        workspace_snapshot_fn=lambda: (workspace_dict, snapshot_obj))
    app_obj.config["TESTING"] = True
    try:
        yield app_obj
    finally:
        provider_obj.close()


def _map_json_dict(html_str):
    match_obj = re.search(r"data-map-json='([^']*)'", html_str)
    assert match_obj, "map data attribute missing"
    return json.loads(unescape(match_obj[1]))


def test_routes_render_now_and_month_maps_and_period_links(demo_app):
    client_obj = demo_app.test_client()
    now_str = client_obj.get("/positions").get_data(as_text=True)
    now_dict = _map_json_dict(now_str)
    assert now_dict["mode_str"] == "now" and now_dict["available_bool"] is True
    assert 'data-sortable-table="positions-now"' in now_str and 'data-sort-key="value"' in now_str
    assert re.findall(r'<nav class="positions-period".*?</nav>', now_str, re.S)
    for month_str in ("2026-09", "2026-08", "2026-07", "2026-06"):
        assert f"month={month_str}" in now_str
    month_str = client_obj.get("/positions?month=2026-08&pod=demo_1_2").get_data(as_text=True)
    month_dict = _map_json_dict(month_str)
    assert month_dict["mode_str"] == "month" and [pod_dict["pod_id_str"] for pod_dict in month_dict["pod_list"]] == ["demo_1_2"]
    assert 'data-sortable-table="positions-month"' in month_str
    assert "/positions/refresh?month=2026-08&amp;pod=demo_1_2" in month_str
    assert "view=changed" not in month_str
    assert "August 2026" in month_str and 'aria-current="page">Aug<' in month_str


@pytest.mark.parametrize(("url_str", "status_int"), [
    ("/positions?month=", 400), ("/positions/refresh?month=", 400),
    ("/positions?month=2026-13", 400), ("/positions?month=26-08", 400), ("/positions?month=2026-08&view=changed", 400),
    ("/positions?month=2026-08&month=2026-07", 400), ("/positions?month=2026-01", 404), ("/positions/refresh?month=2025-12", 404),
    ("/positions?month=2026-08&pod=nope", 404),
])
def test_month_parameter_is_strict(demo_app, url_str, status_int):
    assert demo_app.test_client().get(url_str).status_code == status_int


def test_non_finite_values_and_empty_views_never_reach_the_browser():
    nan_list = [{"pod_id_str": "a", "cash_float": 0.0, "row_list": [_row("AAA", 5, float("nan"), 1.0, 2.0)]}]
    assert build_now_map_dict(nan_list, IDENTITY_LIST, pod_str="all", book_value_float=600.0, asof_str="x")["reason_str"] == "Map data could not be verified."
    empty_list = [{"pod_id_str": "a", "cash_float": 0.0, "row_list": []}, *SOURCE_LIST[1:]]
    map_dict = build_now_map_dict(empty_list, IDENTITY_LIST, pod_str="a", book_value_float=600.0, asof_str="x")
    assert map_dict["available_bool"] is False and map_dict["reason_str"] == "No current holdings or cash in this view."


def _app_tuple(workspace_dict=None):
    base_workspace_dict, snapshot_obj, provider_obj = build_demo_workspace_tuple(include_holdings_bool=True)
    workspace_dict = workspace_dict(base_workspace_dict) if callable(workspace_dict) else base_workspace_dict
    app_obj = create_app(provider_obj, demo_bool=True, now_fn=lambda: DEMO_NOW_TS,
        workspace_snapshot_fn=lambda: (workspace_dict, snapshot_obj))
    app_obj.config["TESTING"] = True
    return app_obj, provider_obj


def _visible_str(html_str):
    return re.sub(r"\s+", " ", unescape(re.sub(r"<[^>]+>", " ", html_str)))


def test_unavailable_month_renders_its_reason_and_no_numbers():
    app_obj, provider_obj = _app_tuple()
    provider_obj.get_day_mtm_dict = lambda nav_row_obj, **_: {"available_bool": False, "reason_str": "x", "missing_section_bool": True, "row_list": []}
    try:
        client_obj = app_obj.test_client()
        response_obj = client_obj.get("/positions?month=2026-08&pod=demo_1_2")
        html_str = response_obj.get_data(as_text=True)
        assert response_obj.status_code == 200
        visible_str = _visible_str(html_str)
        assert "August 2026 unavailable" in visible_str and "no per-symbol P&L" in visible_str
        for absent_str in ("data-map-json", 'aria-label="Month by pod"', "positions-month-table", "positions-month-tile"):
            assert absent_str not in html_str
        # The period links keep the Pod filter.
        assert 'href="/positions?pod=demo_1_2"' in html_str
        assert client_obj.get("/positions/refresh?month=2026-08&pod=demo_1_2").status_code == 200
    finally:
        provider_obj.close()


def test_month_pod_filter_links_keep_the_month(demo_app):
    html_str = demo_app.test_client().get("/positions?month=2026-08&pod=demo_1_2").get_data(as_text=True)
    assert 'href="/positions?month=2026-08"' in html_str and 'href="/positions?month=2026-08&amp;pod=demo_1_0"' in html_str
    assert 'href="/positions?pod=demo_1_2"' in html_str


def test_unreadable_reports_show_the_reason_for_a_valid_month():
    app_obj, provider_obj = _app_tuple(lambda workspace_dict: {**workspace_dict, "financial_error_str": "Saved financial data could not be verified."})
    try:
        response_obj = app_obj.test_client().get("/positions?month=2026-08")
        assert response_obj.status_code == 200
        assert "Saved IBKR reports unavailable." in _visible_str(response_obj.get_data(as_text=True))
    finally:
        provider_obj.close()


def test_ownership_failure_gives_one_reason_in_month_mode():
    workspace_dict, snapshot_obj, provider_obj = build_demo_workspace_tuple(include_holdings_bool=True)
    try:
        broken_dict = {**workspace_dict, "valuation_account_list": workspace_dict["valuation_account_list"] * 2}
        page_dict = build_positions_page_dict(broken_dict, snapshot_obj, provider_obj, as_of_ts=DEMO_NOW_TS, month_str="2026-08")
        assert page_dict["mode_str"] == "month" and page_dict["month_known_bool"] is True
        assert page_dict["verdict_detail_str"] == page_dict["map_dict"]["reason_str"] == page_dict["month_dict"]["reason_str"]
    finally:
        provider_obj.close()


def test_now_map_says_why_when_one_pod_has_no_values(monkeypatch):
    from alpha.live.dashboard_v4 import positions_enrichment
    original_fn = positions_enrichment.load_broker_holdings_dict

    def one_missing(target_obj, *, as_of_ts):
        if target_obj.release_obj.pod_id_str == "demo_1_1":
            return {"available_bool": False, "missing_str": "IBKR sent no prices at the 16:10 capture"}
        return original_fn(target_obj, as_of_ts=as_of_ts)
    monkeypatch.setattr(positions_enrichment, "load_broker_holdings_dict", one_missing)
    workspace_dict, snapshot_obj, provider_obj = build_demo_workspace_tuple(include_holdings_bool=True)
    try:
        page_dict = build_positions_page_dict(workspace_dict, snapshot_obj, provider_obj, as_of_ts=DEMO_NOW_TS)
        assert page_dict["map_dict"]["available_bool"] is False and page_dict["side_dict"] == {}
        # The message names the pod and why.
        assert page_dict["map_dict"]["reason_str"] == ("The map needs IBKR values for every pod. "
            "Missing: QPI (IBKR sent no prices at the 16:10 capture).")
    finally:
        provider_obj.close()


@pytest.mark.parametrize("activity_dict,shown_bool", [
    ({"available_bool": True, "symbol_dict": {}}, False),  # nothing changed today: no empty column
    ({"available_bool": False, "symbol_dict": {}}, True),  # unknown stays visible as "Unknown"
])
def test_today_column_shows_only_when_it_says_something(monkeypatch, activity_dict, shown_bool):
    from alpha.live.dashboard_v4 import positions_enrichment
    monkeypatch.setattr(positions_enrichment, "load_position_activity_dict", lambda target_obj, *, as_of_ts: dict(activity_dict))
    workspace_dict, snapshot_obj, provider_obj = build_demo_workspace_tuple(include_holdings_bool=True)
    try:
        page_dict = build_positions_page_dict(workspace_dict, snapshot_obj, provider_obj, as_of_ts=DEMO_NOW_TS)
        assert page_dict["today_column_bool"] is shown_bool
    finally:
        provider_obj.close()
