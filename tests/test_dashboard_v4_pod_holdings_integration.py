"""Pod closing-value integration stays separate from current cycle/quantity data."""

from copy import deepcopy
from decimal import Decimal
from html.parser import HTMLParser

from flask import template_rendered
import pytest

from alpha.live.dashboard_v4.app import create_app
from alpha.live.dashboard_v4 import demo
from alpha.live.dashboard_v4.demo import DEMO_NOW_TS, build_demo_workspace_tuple
from alpha.live.dashboard_v4.pod_finance import build_pod_finance_dict


@pytest.fixture
def full_fixture_tuple():
    workspace_dict, snapshot_obj, provider_obj = build_demo_workspace_tuple(include_holdings_bool=True)
    app_obj = create_app(provider_obj, demo_bool=True, workspace_snapshot_fn=lambda: (workspace_dict, snapshot_obj),
        now_fn=lambda: DEMO_NOW_TS)
    app_obj.config["TESTING"] = True
    try:
        yield workspace_dict, snapshot_obj, provider_obj, app_obj
    finally:
        provider_obj.close()


def _finance_dict(fixture_tuple, index_int=0, **option_dict):
    workspace_dict, snapshot_obj, provider_obj, _ = fixture_tuple
    return build_pod_finance_dict(workspace_dict, snapshot_obj, provider_obj,
        pod_id_str=workspace_dict["operations_account_list"][index_int]["pod_id"], as_of_ts=DEMO_NOW_TS, **option_dict)


def _source_dict(fixture_tuple, index_int=0):
    workspace_dict, snapshot_obj, provider_obj, _ = fixture_tuple
    route_str = workspace_dict["operations_account_list"][index_int]["account_route"]
    nav_row_obj = next(row_obj for row_obj in snapshot_obj.row_tuple
        if row_obj.account_route_str == route_str and row_obj.market_date_str == "2026-09-04")
    return provider_obj.get_close_holdings_dict(nav_row_obj, query_name_str=workspace_dict["client_dict"]["query_name"], as_of_ts=DEMO_NOW_TS)


def _render_tuple(app_obj, path_str, **option_dict):
    context_list = []

    def capture_context(sender_obj, **signal_dict):
        context_list.append(signal_dict["context"])

    with template_rendered.connected_to(capture_context, app_obj):
        response_obj = app_obj.test_client().get(path_str, **option_dict)
    assert response_obj.status_code == 200
    return response_obj, context_list[-1]["pod_page_dict"]


def _allocation_html_str(response_obj):
    return response_obj.get_data(as_text=True).split('<section class="panel pod-holdings"', 1)[1].split("</section>", 1)[0]


def _set_current_to_close(fixture_tuple):
    source_dict = _source_dict(fixture_tuple)
    row_dict = fixture_tuple[0]["summary_dict"]["pod_row_dict_list"][0]
    row_dict["position_exposure_dict_list"] = [{"asset_str": position_dict["symbol_str"], "share_float": position_dict["shares_float"]}
        for position_dict in source_dict["position_list"]]
    return row_dict


@pytest.mark.parametrize("index_int,count_int", [(0, 10), (2, 4)])
def test_full_preview_has_verified_close_rows_and_cash_without_changing_account_finance(full_fixture_tuple, index_int, count_int):
    workspace_dict, snapshot_obj, provider_obj, app_obj = full_fixture_tuple
    source_dict = _source_dict(full_fixture_tuple, index_int)
    view_dict = _finance_dict(full_fixture_tuple, index_int)
    allocation_dict = view_dict["holdings_allocation_dict"]
    route_str = workspace_dict["operations_account_list"][index_int]["account_route"]
    nav_row_obj = next(row_obj for row_obj in snapshot_obj.row_tuple if row_obj.account_route_str == route_str and row_obj.market_date_str == "2026-09-04")
    assert allocation_dict["available_bool"] and allocation_dict["donut_available_bool"]
    assert len(allocation_dict["row_list"]) == count_int
    assert len(allocation_dict["slice_list"]) == count_int + 1
    total_decimal = sum((Decimal(str(row_dict["value_float"])) for row_dict in allocation_dict["row_list"]),
        Decimal(str(allocation_dict["cash_row_dict"]["value_float"])))
    assert abs(total_decimal - nav_row_obj.closing_nav_decimal) <= Decimal(".01")
    assert {row_dict["symbol_str"]: row_dict["shares_float"] for row_dict in allocation_dict["row_list"]} == {
        row_dict["symbol_str"]: row_dict["shares_float"] for row_dict in source_dict["position_list"]}
    response_obj, page_dict = _render_tuple(app_obj, "/pods/" + workspace_dict["operations_account_list"][index_int]["pod_id"])
    allocation_html_str = _allocation_html_str(response_obj)
    assert '>Value $</th>' in allocation_html_str and '>Weight</th>' in allocation_html_str
    assert 'Close 2026-09-04' in allocation_html_str and '<b>Cash</b>' in allocation_html_str
    assert page_dict["tile_list"] == view_dict["tile_list"]
    assert all(row_dict["target_percent_float"] is None and row_dict["new_bool"] is False for row_dict in allocation_dict["row_list"])


def test_default_fixture_remains_quantities_only_with_existing_finance():
    workspace_dict, snapshot_obj, provider_obj = build_demo_workspace_tuple()
    try:
        view_dict = _finance_dict((workspace_dict, snapshot_obj, provider_obj, None))
        assert not hasattr(provider_obj, "get_close_holdings_dict")
        assert not view_dict["holdings_allocation_dict"]["available_bool"]
        assert view_dict["position_list"] and view_dict["tile_list"][0]["value_str"] != "—"
    finally:
        provider_obj.close()


def test_create_demo_app_enables_full_visual_preview(monkeypatch):
    original_fn = demo.build_demo_workspace_tuple
    provider_list = []

    def capture_fixture(**option_dict):
        assert option_dict == {"include_holdings_bool": True}
        result_tuple = original_fn(**option_dict)
        provider_list.append(result_tuple[2])
        return result_tuple

    monkeypatch.setattr(demo, "build_demo_workspace_tuple", capture_fixture)
    try:
        response_obj = demo.create_demo_app().test_client().get("/pods/demo_1_0")
        assert response_obj.status_code == 200 and "data-pod-allocation" in response_obj.get_data(as_text=True)
    finally:
        for provider_obj in provider_list:
            provider_obj.close()


def test_nav_mismatch_falls_back_to_dated_quantities_without_empty_value_columns(full_fixture_tuple, monkeypatch):
    _, _, provider_obj, app_obj = full_fixture_tuple
    baseline_dict = _finance_dict(full_fixture_tuple)
    source_dict = _source_dict(full_fixture_tuple)
    source_dict["position_list"][0]["value_float"] += 1
    monkeypatch.setattr(provider_obj, "get_close_holdings_dict", lambda *args, **kwargs: deepcopy(source_dict))
    response_obj, page_dict = _render_tuple(app_obj, "/pods/demo_1_0")
    html_str = response_obj.get_data(as_text=True)
    assert not page_dict["holdings_allocation_dict"]["available_bool"]
    assert 'data-selection-key="positions"' in html_str
    assert '>Value $</th>' not in html_str and '>Weight</th>' not in html_str and "data-pod-allocation" not in html_str
    assert "Closing values do not match NAV" in html_str
    assert page_dict["position_list"] == baseline_dict["position_list"]
    assert page_dict["tile_list"] == baseline_dict["tile_list"] and page_dict["chart_dict"] == baseline_dict["chart_dict"]


def test_missing_close_cash_cannot_create_residual_cash_or_allocation(full_fixture_tuple, monkeypatch):
    baseline_dict = _finance_dict(full_fixture_tuple)
    monkeypatch.setattr("alpha.live.dashboard_v4.pod_finance.load_portfolio_cash_list", lambda *args, **kwargs: [])
    view_dict = _finance_dict(full_fixture_tuple)
    assert view_dict["cash_str"] == "—"
    assert not view_dict["holdings_allocation_dict"]["available_bool"]
    assert view_dict["holdings_allocation_dict"]["cash_row_dict"] == {}
    assert view_dict["tile_list"] == baseline_dict["tile_list"]


def test_newer_current_share_change_keeps_close_shares_and_adds_notice(full_fixture_tuple):
    row_dict = _set_current_to_close(full_fixture_tuple)
    baseline_dict = _finance_dict(full_fixture_tuple)["holdings_allocation_dict"]
    assert not baseline_dict["holdings_note_str"]
    row_dict["position_exposure_dict_list"][0]["share_float"] += 13
    result_dict = _finance_dict(full_fixture_tuple)["holdings_allocation_dict"]
    assert result_dict["row_list"] == baseline_dict["row_list"]
    assert result_dict["slice_list"] == baseline_dict["slice_list"]
    assert result_dict["holdings_changed_bool"] is True
    assert result_dict["holdings_note_str"] == "Holdings changed since this close."


def test_unknown_current_quantity_does_not_claim_holdings_changed(full_fixture_tuple):
    row_dict = _set_current_to_close(full_fixture_tuple)
    row_dict["position_exposure_dict_list"][0]["share_float"] = None
    result_dict = _finance_dict(full_fixture_tuple)["holdings_allocation_dict"]
    assert result_dict["available_bool"]
    assert not result_dict["holdings_changed_bool"] and not result_dict["holdings_note_str"]


def test_missing_or_sanitized_current_position_list_does_not_prove_flat_holdings(full_fixture_tuple):
    row_dict = _set_current_to_close(full_fixture_tuple)
    valid_list = deepcopy(row_dict["position_exposure_dict_list"])
    for raw_obj in (None, {}, [None], [valid_list[0], None], "missing"):
        if raw_obj == "missing":
            row_dict.pop("position_exposure_dict_list", None)
        else:
            row_dict["position_exposure_dict_list"] = raw_obj
        result_dict = _finance_dict(full_fixture_tuple)["holdings_allocation_dict"]
        assert result_dict["available_bool"]
        assert not result_dict["holdings_changed_bool"] and not result_dict["holdings_note_str"]
    row_dict["position_exposure_dict_list"] = []
    result_dict = _finance_dict(full_fixture_tuple)["holdings_allocation_dict"]
    assert result_dict["holdings_changed_bool"]
    assert result_dict["holdings_note_str"] == "Holdings changed since this close."


def test_same_day_different_saved_quantities_do_not_claim_later_trade(full_fixture_tuple):
    row_dict = _set_current_to_close(full_fixture_tuple)
    row_dict["latest_pod_state_timestamp_str"] = "2026-09-04T19:59:00+00:00"
    row_dict["position_exposure_dict_list"][0]["share_float"] += 1
    result_dict = _finance_dict(full_fixture_tuple)["holdings_allocation_dict"]
    assert result_dict["available_bool"] and not result_dict["holdings_changed_bool"]
    assert result_dict["holdings_note_str"] == "Saved holdings differ from this close."


def test_selected_cycle_does_not_change_current_close_or_infer_new_and_target(full_fixture_tuple):
    app_obj = full_fixture_tuple[3]
    current_response_obj, current_dict = _render_tuple(app_obj, "/pods/demo_1_0?cycle=vplan:2")
    historical_response_obj, historical_dict = _render_tuple(app_obj, "/pods/demo_1_0?cycle=vplan:1")
    assert historical_dict["cycle_label_str"] != current_dict["cycle_label_str"]
    assert historical_dict["holdings_allocation_dict"] == current_dict["holdings_allocation_dict"]
    assert historical_dict["tile_list"] == current_dict["tile_list"]
    assert _allocation_html_str(historical_response_obj) == _allocation_html_str(current_response_obj)
    assert '>New</span>' not in _allocation_html_str(historical_response_obj)
    assert '<em ' not in _allocation_html_str(historical_response_obj)


def test_symbol_text_svg_labels_and_cross_highlight_keys_are_escaped(full_fixture_tuple, monkeypatch):
    provider_obj, app_obj = full_fixture_tuple[2:]
    source_dict = _source_dict(full_fixture_tuple, 2)
    symbol_str = 'AAA" onfocus="alert(1)<svg>&'
    source_dict["position_list"][0]["symbol_str"] = symbol_str
    monkeypatch.setattr(provider_obj, "get_close_holdings_dict", lambda *args, **kwargs: deepcopy(source_dict))
    response_obj, _ = _render_tuple(app_obj, "/pods/demo_1_2")
    allocation_html_str = _allocation_html_str(response_obj)
    element_list = []

    class ElementParser(HTMLParser):
        def handle_starttag(self, tag_str, attribute_list):
            element_list.append((tag_str, dict(attribute_list)))

    ElementParser().feed(allocation_html_str)
    assert "<svg>&" not in allocation_html_str
    assert not any("onfocus" in attribute_dict for _, attribute_dict in element_list)
    matched_list = [(tag_str, attribute_dict) for tag_str, attribute_dict in element_list
        if attribute_dict.get("data-allocation-key") == "position:" + symbol_str]
    assert {tag_str for tag_str, _ in matched_list} == {"path", "tr"}
    assert all(attribute_dict["tabindex"] == "0" for _, attribute_dict in matched_list)


def test_real_scope_uses_saved_reader_and_never_demo_provider_override(full_fixture_tuple, monkeypatch):
    workspace_dict, _, provider_obj, _ = full_fixture_tuple
    source_dict = _source_dict(full_fixture_tuple)
    workspace_dict["client_dict"]["is_demo"] = False
    monkeypatch.setattr(provider_obj, "get_close_holdings_dict", lambda *args, **kwargs: pytest.fail("Real scope invoked demo override"))
    call_list = []

    def saved_reader(database_path_str, nav_row_obj, **option_dict):
        call_list.append((database_path_str, nav_row_obj, option_dict))
        return deepcopy(source_dict)

    monkeypatch.setattr("alpha.live.dashboard_v4.pod_finance.load_close_holdings_dict", saved_reader)
    result_dict = _finance_dict(full_fixture_tuple, performance_db_path_str="owned-report.sqlite3")
    assert result_dict["holdings_allocation_dict"]["available_bool"]
    assert len(call_list) == 1 and call_list[0][0] == "owned-report.sqlite3"
    assert call_list[0][1].account_route_str == workspace_dict["operations_account_list"][0]["account_route"]
    assert call_list[0][1].market_date_str == "2026-09-04"
    assert call_list[0][2] == {"query_name_str": workspace_dict["client_dict"]["query_name"], "as_of_ts": DEMO_NOW_TS}


def test_holdings_full_and_refresh_keep_security_headers_and_existing_routes(full_fixture_tuple):
    app_obj = full_fixture_tuple[3]
    for path_str in ("/pods/demo_1_0", "/pods/demo_1_0/refresh"):
        response_obj, _ = _render_tuple(app_obj, path_str)
        assert "data-pod-allocation" in response_obj.get_data(as_text=True)
        assert response_obj.headers["Cache-Control"] == "no-store"
        assert response_obj.headers["X-Frame-Options"] == "DENY"
        assert "script-src 'self'" in response_obj.headers["Content-Security-Policy"]
        assert "form-action 'none'" in response_obj.headers["Content-Security-Policy"]
        assert "<form" not in response_obj.get_data(as_text=True)
    assert not any("holdings" in rule_obj.rule for rule_obj in app_obj.url_map.iter_rules())
    assert app_obj.test_client().get("/pods/demo_1_0/holdings").status_code == 404
    assert app_obj.test_client().get("/pods/demo_1_0?holdings=true").status_code == 400


def test_holdings_requests_cannot_enable_writes_or_reach_source_on_post():
    app_obj = create_app(object(), workspace_snapshot_fn=lambda: pytest.fail("Write reached source"), now_fn=lambda: DEMO_NOW_TS)
    for method_str in ("POST", "PUT", "PATCH", "DELETE"):
        response_obj = app_obj.test_client().open("/pods/demo_1_0", method=method_str, json={"action": "holdings"})
        assert response_obj.status_code == 403 and response_obj.get_json()["error"] == "read_only"
