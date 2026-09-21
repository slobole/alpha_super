"""Positions routes preserve read-only scope and render saved quantities honestly."""

from copy import deepcopy
from datetime import timedelta
from html import unescape
import re

from flask import template_rendered
import pytest

from alpha.live.dashboard_v4.app import create_app
from alpha.live.dashboard_v4.demo import DEMO_NOW_TS, build_demo_workspace_tuple
from alpha.live.dashboard_v4.positions import build_positions_page_dict


@pytest.fixture
def positions_fixture_tuple():
    workspace_dict, snapshot_obj, provider_obj = build_demo_workspace_tuple()
    app_obj = create_app(provider_obj, demo_bool=True, now_fn=lambda: DEMO_NOW_TS,
        workspace_snapshot_fn=lambda: (workspace_dict, snapshot_obj))
    app_obj.config["TESTING"] = True
    try:
        yield workspace_dict, provider_obj, app_obj
    finally:
        provider_obj.close()


def _render_tuple(app_obj, path_str, *, headers_dict=None):
    context_list = []

    def capture_context(sender_obj, **signal_dict):
        context_list.append(signal_dict["context"])

    with template_rendered.connected_to(capture_context, app_obj):
        response_obj = app_obj.test_client().get(path_str, headers=headers_dict)
    assert response_obj.status_code == 200
    return response_obj, context_list[-1]["positions_page_dict"]


def _visible_text_str(html_str):
    return re.sub(r"\s+", " ", unescape(re.sub(r"<[^>]+>", " ", html_str))).strip()


@pytest.mark.parametrize("method_str", ["POST", "PUT", "PATCH", "DELETE"])
@pytest.mark.parametrize("path_str", ["/positions", "/positions/refresh"])
def test_positions_writes_are_rejected_before_provider_access(method_str, path_str):
    app_obj = create_app(object(), now_fn=lambda: DEMO_NOW_TS,
        workspace_snapshot_fn=lambda: pytest.fail("Write request reached provider"))
    response_obj = app_obj.test_client().open(path_str, method=method_str, json={"action": "submit"})
    assert response_obj.status_code == 403
    assert response_obj.get_json()["error"] == "read_only"


@pytest.mark.parametrize("query_str", [
    "mode=paper", "period=All", "q=AMD", "cycle=vplan:2", "view=unknown", "view=",
    "view=all&view=changed", "pod=all&pod=demo_1_0", "pod=", "pod=" + "a" * 201,
    "action=submit", "export=csv",
])
@pytest.mark.parametrize("path_str", ["/positions", "/positions/refresh"])
def test_positions_invalid_queries_fail_before_provider_access(query_str, path_str):
    app_obj = create_app(object(), now_fn=lambda: DEMO_NOW_TS,
        workspace_snapshot_fn=lambda: pytest.fail("Invalid selection reached provider"))
    assert app_obj.test_client().get(path_str + "?" + query_str).status_code == 400


def test_positions_shell_security_and_local_search_contract(positions_fixture_tuple):
    _, _, app_obj = positions_fixture_tuple
    response_obj, page_dict = _render_tuple(app_obj, "/positions")
    html_str = response_obj.get_data(as_text=True)
    assert "<html" in html_str and "ALPHA / OPS V4 · Positions" in html_str
    assert 'class="page calm positions-page"' in html_str
    assert 'data-selection-scope="positions:all:all"' in html_str
    assert html_str.count('hx-get="') == 1 and "positions/refresh?view=all&amp;pod=all" in html_str
    assert 'data-positions-search type="search" name="q"' in html_str and 'maxlength="80"' in html_str
    assert "<form" not in html_str and 'type="hidden"' not in html_str
    assert 'data-position-search-empty hidden' in html_str
    assert 'aria-label="Book at last close"' in html_str
    assert "READ-ONLY" not in html_str and "Free cash" not in html_str and "Show all" not in html_str
    assert 'title="PAPER is not available in V4 yet"' in html_str
    assert 'title="INCUBATION is not available in V4 yet"' in html_str
    assert len(page_dict["pod_row_list"]) == 4
    assert response_obj.headers["Cache-Control"] == "no-store"
    assert response_obj.headers["X-Content-Type-Options"] == "nosniff"
    assert response_obj.headers["X-Frame-Options"] == "DENY"
    assert response_obj.headers["Content-Security-Policy"] == (
        "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
        "font-src 'self'; img-src 'self' data:; connect-src 'self'; "
        "object-src 'none'; base-uri 'none'; frame-ancestors 'none'; form-action 'none'")


def test_positions_displays_only_connected_fields(positions_fixture_tuple):
    _, _, app_obj = positions_fixture_tuple
    response_obj, page_dict = _render_tuple(app_obj, "/positions")
    html_str = response_obj.get_data(as_text=True)
    main_str = re.search(r"<main\b[^>]*>(.*?)</main>", html_str, re.S)[1]
    holdings_str = re.search(r'<table\b[^>]*data-selection-key="positions-table"[^>]*>(.*?)</table>', main_str, re.S)[1]
    account_str = re.search(r'<table\b[^>]*data-selection-key="positions-by-pod"[^>]*>(.*?)</table>', main_str, re.S)[1]
    assert [_visible_text_str(header_str) for header_str in re.findall(r"<th\b[^>]*>(.*?)</th>", holdings_str, re.S)] == ["Symbol", "Pods", "Shares"]
    assert [_visible_text_str(header_str) for header_str in re.findall(r"<th\b[^>]*>(.*?)</th>", account_str, re.S)] == ["Pod", "Positions", "Invested $", "Cash $", "Book"]
    assert re.findall(r'data-selection-key="positions-tile:([^"]+)"', main_str) == ["Invested"]
    visible_str = _visible_text_str(main_str)
    for label_str in ("Value $", "Weight", "P&L since entry", "Today", "Open P&L", "Best", "Worst",
                      "Changed today", "Off target", "Closing prices", "entry P&L are unavailable"):
        assert label_str not in visible_str
    assert "view=changed" not in main_str and "view=off_target" not in main_str
    assert page_dict["financial_asof_str"] in visible_str
    assert page_dict["total_dict"]["invested_str"] in visible_str
    assert page_dict["total_dict"]["cash_str"] in visible_str


@pytest.mark.parametrize("path_str,headers_dict", [
    ("/positions?view=all&pod=demo_1_0", {"HX-Request": "true"}),
    ("/positions/refresh?view=all&pod=demo_1_0", None),
])
def test_positions_partial_preserves_filter_scope(positions_fixture_tuple, path_str, headers_dict):
    _, _, app_obj = positions_fixture_tuple
    response_obj, page_dict = _render_tuple(app_obj, path_str, headers_dict=headers_dict)
    html_str = response_obj.get_data(as_text=True)
    assert "<html" not in html_str
    assert 'data-selection-scope="positions:all:demo_1_0"' in html_str
    assert 'hx-get="/positions/refresh?view=all&amp;pod=demo_1_0"' in html_str
    assert page_dict["pod_str"] == "demo_1_0"
    assert sum(item_dict["selected_bool"] for item_dict in page_dict["pod_filter_list"]) == 1


def test_pod_filter_narrows_shares_without_changing_portfolio_totals(positions_fixture_tuple):
    workspace_dict, provider_obj, app_obj = positions_fixture_tuple
    pod_id_str = workspace_dict["operations_account_list"][0]["pod_id"]
    _, full_dict = _render_tuple(app_obj, "/positions")
    response_obj, selected_dict = _render_tuple(app_obj, "/positions?pod=" + pod_id_str)
    expected_dict = provider_obj.get_positions_dict(pod_id_str, as_of_ts=DEMO_NOW_TS)["position_map_dict"]
    assert {row_dict["symbol_str"] for row_dict in selected_dict["row_list"]} == set(expected_dict)
    for row_dict in selected_dict["row_list"]:
        assert {holder_dict["pod_id_str"] for holder_dict in row_dict["pod_list"]} == {pod_id_str}
        assert float(row_dict["share_str"].replace(",", "")) == pytest.approx(expected_dict[row_dict["symbol_str"]], abs=1e-6)
    full_sgov_dict = next(row_dict for row_dict in full_dict["row_list"] if row_dict["symbol_str"] == "SGOV")
    selected_sgov_dict = next(row_dict for row_dict in selected_dict["row_list"] if row_dict["symbol_str"] == "SGOV")
    assert len(full_sgov_dict["pod_list"]) == 4
    assert full_sgov_dict["share_str"] != selected_sgov_dict["share_str"]
    assert selected_dict["total_dict"] == full_dict["total_dict"]
    assert selected_dict["tile_list"] == full_dict["tile_list"]
    assert selected_dict["total_dict"]["count_str"] == "8"
    assert "3 of 8" in selected_dict["verdict_str"]
    verdict_str = selected_dict["verdict_str"] + " " + selected_dict["verdict_detail_str"]
    assert "DVO2" in verdict_str
    header_str = re.search(r'<p\b[^>]*data-selection-key="positions-verdict"[^>]*>(.*?)</p>', response_obj.get_data(as_text=True), re.S)[1]
    assert "3 of 8" in _visible_text_str(header_str) and "DVO2" in _visible_text_str(header_str)


def test_shared_symbol_quantities_are_visible_without_hover(positions_fixture_tuple):
    workspace_dict, provider_obj, app_obj = positions_fixture_tuple
    response_obj, page_dict = _render_tuple(app_obj, "/positions")
    row_str = re.search(r'<tr\b[^>]*data-position-symbol="SGOV"[^>]*>(.*?)</tr>', response_obj.get_data(as_text=True), re.S)[1]
    chip_list = [_visible_text_str(chip_str) for chip_str in re.findall(r'<a\b[^>]*class="podchip"[^>]*>(.*?)</a>', row_str, re.S)]
    assert len(chip_list) == 4
    for account_dict in workspace_dict["operations_account_list"]:
        share_float = provider_obj.get_positions_dict(account_dict["pod_id"], as_of_ts=DEMO_NOW_TS)["position_map_dict"]["SGOV"]
        chip_str = next(chip_str for chip_str in chip_list if account_dict["display_name"] in chip_str)
        number_list = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", chip_str)
        assert number_list, "The quantity must be visible text, not only a title attribute"
        assert float(number_list[-1].replace(",", "")) == pytest.approx(share_float, abs=1e-6)
    assert next(row_dict for row_dict in page_dict["row_list"] if row_dict["symbol_str"] == "SGOV")["offset_bool"] is False


def test_foreign_pod_is_not_a_queryable_source(positions_fixture_tuple, monkeypatch):
    _, provider_obj, app_obj = positions_fixture_tuple
    monkeypatch.setattr(provider_obj, "get_positions_dict", lambda *args, **kwargs: pytest.fail("Foreign selection reached source"))
    assert app_obj.test_client().get("/positions?pod=foreign_paper_pod").status_code == 404


@pytest.mark.parametrize("view_str,reason_str", [
    ("changed", "Changed-today evidence unavailable"), ("off_target", "Target comparison unavailable"),
])
def test_unconnected_views_have_no_visible_controls_or_false_zero_claim(positions_fixture_tuple, view_str, reason_str):
    _, _, app_obj = positions_fixture_tuple
    response_obj, page_dict = _render_tuple(app_obj, "/positions")
    html_str = response_obj.get_data(as_text=True)
    assert "Changed today" not in html_str and "Off target" not in html_str
    assert "view=changed" not in html_str and "view=off_target" not in html_str
    _, unavailable_dict = _render_tuple(app_obj, "/positions?view=" + view_str)
    assert unavailable_dict["row_list"] == []
    assert unavailable_dict["empty_str"] == reason_str


@pytest.mark.parametrize("available_bool", [True, False])
def test_empty_holdings_are_distinct_from_unavailable(positions_fixture_tuple, monkeypatch, available_bool):
    _, provider_obj, app_obj = positions_fixture_tuple
    original_fn = provider_obj.get_positions_dict

    def empty_positions_dict(pod_id_str, *, as_of_ts):
        source_dict = original_fn(pod_id_str, as_of_ts=as_of_ts)
        return {**source_dict, "available_bool": available_bool, "position_map_dict": {}}

    monkeypatch.setattr(provider_obj, "get_positions_dict", empty_positions_dict)
    response_obj, page_dict = _render_tuple(app_obj, "/positions")
    html_str = response_obj.get_data(as_text=True)
    assert page_dict["row_list"] == []
    assert page_dict["holdings_complete_bool"] is available_bool
    assert page_dict["total_dict"]["count_str"] == ("0" if available_bool else "—")
    assert ("No saved positions" if available_bool else "Saved positions unavailable") in html_str
    assert page_dict["filter_list"][0]["label_str"] == ("All 0" if available_bool else "All")


@pytest.mark.parametrize("available_bool", [True, False])
def test_selected_empty_pod_count_is_distinct_from_missing_source(positions_fixture_tuple, monkeypatch, available_bool):
    workspace_dict, provider_obj, app_obj = positions_fixture_tuple
    selected_pod_str = workspace_dict["operations_account_list"][0]["pod_id"]
    original_fn = provider_obj.get_positions_dict

    def selected_positions_dict(pod_id_str, *, as_of_ts):
        source_dict = original_fn(pod_id_str, as_of_ts=as_of_ts)
        if pod_id_str == selected_pod_str:
            return {**source_dict, "available_bool": available_bool, "position_map_dict": {}}
        return source_dict

    monkeypatch.setattr(provider_obj, "get_positions_dict", selected_positions_dict)
    _, page_dict = _render_tuple(app_obj, "/positions?pod=" + selected_pod_str)
    assert page_dict["row_list"] == []
    if available_bool:
        assert "0 of 6" in page_dict["verdict_str"]
        assert "DVO2" in page_dict["verdict_str"] + " " + page_dict["verdict_detail_str"]
        assert page_dict["total_dict"]["count_str"] == "6"
    else:
        assert "unavailable" in (page_dict["verdict_str"] + page_dict["verdict_detail_str"]).lower()
        assert "0 of" not in page_dict["verdict_str"]
        assert page_dict["total_dict"]["count_str"] == "—"


def test_selected_healthy_pod_does_not_claim_complete_portfolio_count_when_another_source_is_missing(positions_fixture_tuple, monkeypatch):
    workspace_dict, provider_obj, app_obj = positions_fixture_tuple
    selected_pod_str = workspace_dict["operations_account_list"][0]["pod_id"]
    missing_pod_str = workspace_dict["operations_account_list"][1]["pod_id"]
    original_fn = provider_obj.get_positions_dict

    def partial_positions_dict(pod_id_str, *, as_of_ts):
        source_dict = original_fn(pod_id_str, as_of_ts=as_of_ts)
        return {**source_dict, "available_bool": False} if pod_id_str == missing_pod_str else source_dict

    monkeypatch.setattr(provider_obj, "get_positions_dict", partial_positions_dict)
    _, page_dict = _render_tuple(app_obj, "/positions?pod=" + selected_pod_str)
    assert len(page_dict["row_list"]) == 3
    assert page_dict["total_dict"]["count_str"] == "—"
    verdict_str = (page_dict["verdict_str"] + " " + page_dict["verdict_detail_str"]).lower()
    assert "3 of 7" not in verdict_str and "3 of 8" not in verdict_str
    assert "unavailable" in verdict_str or "incomplete" in verdict_str


def test_saved_symbols_are_escaped_in_text_and_filter_attributes(positions_fixture_tuple, monkeypatch):
    _, provider_obj, app_obj = positions_fixture_tuple
    original_fn = provider_obj.get_positions_dict
    symbol_str = 'BAD"><script>alert(1)</script>'

    def changed_positions_dict(pod_id_str, *, as_of_ts):
        source_dict = deepcopy(original_fn(pod_id_str, as_of_ts=as_of_ts))
        source_dict["position_map_dict"] = {symbol_str: 1.0}
        return source_dict

    monkeypatch.setattr(provider_obj, "get_positions_dict", changed_positions_dict)
    response_obj, page_dict = _render_tuple(app_obj, "/positions")
    html_str = response_obj.get_data(as_text=True)
    assert page_dict["row_list"][0]["symbol_str"] == symbol_str
    assert '<script>alert(1)</script>' not in html_str
    assert 'data-position-symbol="BAD&#34;&gt;&lt;script&gt;alert(1)&lt;/script&gt;"' in html_str
    assert 'data-position-row' in html_str


@pytest.mark.parametrize("path_str", ["/positions", "/positions/refresh"])
def test_slow_positions_read_expires_server_html_before_javascript(monkeypatch, path_str):
    workspace_dict, snapshot_obj, provider_obj = build_demo_workspace_tuple()
    clock_dict = {"now_ts": DEMO_NOW_TS}
    context_list = []

    def slow_positions_dict(*args, **kwargs):
        result_dict = build_positions_page_dict(*args, **kwargs)
        clock_dict["now_ts"] += timedelta(seconds=121)
        return result_dict

    def capture_context(sender_obj, **signal_dict):
        context_list.append(signal_dict["context"])

    monkeypatch.setattr("alpha.live.dashboard_v4.app.build_positions_page_dict", slow_positions_dict)
    app_obj = create_app(provider_obj, demo_bool=True, now_fn=lambda: clock_dict["now_ts"],
        workspace_snapshot_fn=lambda: (workspace_dict, snapshot_obj))
    try:
        with template_rendered.connected_to(capture_context, app_obj):
            response_obj = app_obj.test_client().get(path_str)
        assert response_obj.status_code == 200
        overview_dict = context_list[-1]["overview_dict"]
        assert overview_dict["source_fresh_bool"] is False
        assert overview_dict["source_valid_ms_int"] == 0
        assert all(pod_dict["state_str"] == "unk" for pod_dict in overview_dict["pod_list"])
        assert 'data-source-valid-ms="0"' in response_obj.get_data(as_text=True)
        assert 'positions/refresh' in overview_dict["refresh_url_str"]
    finally:
        provider_obj.close()


def test_inactive_owned_pod_keeps_financial_row_without_broken_link_or_zero_claim(positions_fixture_tuple):
    workspace_dict, _, app_obj = positions_fixture_tuple
    inactive_dict = workspace_dict["operations_account_list"][-1]
    workspace_dict["operations_account_list"] = workspace_dict["operations_account_list"][:-1]
    pod_id_str = inactive_dict["pod_id"]
    response_obj, page_dict = _render_tuple(app_obj, "/positions")
    account_row_dict = next(row_dict for row_dict in page_dict["pod_row_list"] if row_dict["pod_id_str"] == pod_id_str)
    assert account_row_dict["count_str"] == "—" and account_row_dict["url_str"] == ""
    assert account_row_dict["cash_str"] != "—"
    assert f'href="/pods/{pod_id_str}"' not in response_obj.get_data(as_text=True)
    assert page_dict["filter_list"][0]["label_str"] == "All"
    _, selected_dict = _render_tuple(app_obj, "/positions?pod=" + pod_id_str)
    assert selected_dict["empty_str"] == "Saved positions unavailable for this pod"
    assert selected_dict["filter_list"][0]["label_str"] == "All"
