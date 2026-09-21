"""Market closing allocation works independently of Flex account reporting."""

from copy import deepcopy
from dataclasses import replace

import pytest

from alpha.live.dashboard_v4 import pod_finance
from alpha.live.dashboard_v4.app import create_app
from alpha.live.dashboard_v4.demo import DEMO_NOW_TS, build_demo_workspace_tuple


@pytest.fixture
def market_fixture_tuple(monkeypatch):
    workspace_dict, snapshot_obj, provider_obj = build_demo_workspace_tuple()
    workspace_dict["client_dict"]["is_demo"] = False
    workspace_dict["client_dict"]["operations_source"] = "local"
    pod_id_str = workspace_dict["operations_account_list"][0]["pod_id"]
    eod_dict = {"available_bool": True, "reason_str": "", "close_date_str": "2026-09-04",
        "position_map_dict": {"AMD": 31, "CRM": 17}, "cash_float": 500.0,
        "broker_nav_float": 7100.0, "observed_timestamp_str": "2026-09-04T20:10:00+00:00"}
    prices_dict = {"available_bool": True, "reason_str": "", "close_date_str": "2026-09-04",
        "price_map_dict": {"AMD": 100.0, "CRM": 200.0}, "source_str": "Norgate unadjusted close"}
    call_list = []

    def eod_reader(target_obj, **option_dict):
        assert target_obj.release_obj.pod_id_str == pod_id_str
        assert option_dict == {"close_date_str": None, "as_of_ts": DEMO_NOW_TS}
        call_list.append("eod")
        return deepcopy(eod_dict)

    def prices_reader(symbol_list, **option_dict):
        assert symbol_list == sorted(symbol_str for symbol_str, shares_float in eod_dict["position_map_dict"].items() if shares_float)
        assert option_dict["close_date_str"] == eod_dict["close_date_str"]
        call_list.append("prices")
        return deepcopy(prices_dict)

    monkeypatch.setattr(pod_finance, "load_eod_holdings_dict", eod_reader)
    monkeypatch.setattr(pod_finance, "load_close_prices_dict", prices_reader)
    try:
        yield workspace_dict, snapshot_obj, provider_obj, pod_id_str, eod_dict, prices_dict, call_list
    finally:
        provider_obj.close()


def _view_dict(fixture_tuple):
    workspace_dict, snapshot_obj, provider_obj, pod_id_str = fixture_tuple[:4]
    return pod_finance.build_pod_finance_dict(workspace_dict, snapshot_obj, provider_obj,
        pod_id_str=pod_id_str, as_of_ts=DEMO_NOW_TS)


def test_estimate_uses_actual_eod_cash_and_raw_marks_without_modifying_reported_money(market_fixture_tuple):
    workspace_dict, snapshot_obj, provider_obj, pod_id_str = market_fixture_tuple[:4]
    baseline_dict = pod_finance._build_report_finance_dict(workspace_dict, snapshot_obj, provider_obj,
        pod_id_str=pod_id_str, as_of_ts=DEMO_NOW_TS)
    view_dict = _view_dict(market_fixture_tuple)
    allocation_dict = view_dict["holdings_allocation_dict"]
    assert allocation_dict["available_bool"] and allocation_dict["donut_available_bool"]
    assert allocation_dict["estimated_bool"] is True
    assert allocation_dict["estimated_total_float"] == 7000
    assert allocation_dict["broker_nav_float"] == 7100
    assert [row_dict["value_float"] for row_dict in allocation_dict["row_list"]] == [3100, 3400]
    assert allocation_dict["row_list"][0]["weight_float"] == pytest.approx(3100 / 7000)
    assert allocation_dict["cash_row_dict"]["value_float"] == 500
    for field_str in ("tile_list", "chart_dict", "money_asof_str", "cash_str"):
        assert view_dict[field_str] == baseline_dict[field_str]


def test_missing_flex_finance_does_not_hide_market_allocation(market_fixture_tuple):
    fixture_list = list(market_fixture_tuple)
    fixture_list[0]["financial_error_str"] = "No Flex report"
    fixture_list[1] = replace(fixture_list[1], row_tuple=(), unavailable_reason_str="No Flex report")
    view_dict = _view_dict(fixture_list)
    assert view_dict["holdings_allocation_dict"]["available_bool"]
    assert all(tile_dict["value_str"] == "—" for tile_dict in view_dict["tile_list"])
    assert not view_dict["chart_dict"]["available_bool"]


@pytest.mark.parametrize("mutation_str", ["date", "missing", "extra", "zero", "nan"])
def test_mixed_missing_or_bad_marks_never_render_partial_pie(market_fixture_tuple, mutation_str):
    prices_dict = market_fixture_tuple[5]
    if mutation_str == "date":
        prices_dict["close_date_str"] = "2026-09-03"
    elif mutation_str == "missing":
        prices_dict["price_map_dict"].pop("CRM")
    elif mutation_str == "extra":
        prices_dict["price_map_dict"]["MSFT"] = 100
    else:
        prices_dict["price_map_dict"]["CRM"] = 0 if mutation_str == "zero" else float("nan")
    assert not _view_dict(market_fixture_tuple)["holdings_allocation_dict"]["available_bool"]


def test_unavailable_prices_keep_quantities_and_explain_reason(market_fixture_tuple):
    market_fixture_tuple[5].update(available_bool=False, reason_str="Closing prices unavailable for this date.")
    view_dict = _view_dict(market_fixture_tuple)
    assert view_dict["position_list"]
    assert not view_dict["holdings_allocation_dict"]["available_bool"]
    assert view_dict["holdings_allocation_dict"]["reason_str"] == "Closing prices unavailable for this date."


def test_unavailable_eod_does_not_call_prices(market_fixture_tuple):
    market_fixture_tuple[4].update(available_bool=False, reason_str="Saved broker EOD unavailable")
    assert not _view_dict(market_fixture_tuple)["holdings_allocation_dict"]["available_bool"]
    assert market_fixture_tuple[6] == ["eod"]


def test_cash_only_needs_no_market_price_source(market_fixture_tuple):
    market_fixture_tuple[4]["position_map_dict"] = {}
    view_dict = _view_dict(market_fixture_tuple)
    allocation_dict = view_dict["holdings_allocation_dict"]
    assert allocation_dict["available_bool"] and allocation_dict["donut_available_bool"]
    assert allocation_dict["cash_percent_str"] == "100.0%"
    assert allocation_dict["row_list"] == []
    assert market_fixture_tuple[6] == ["eod"]


@pytest.mark.parametrize("cash_float,crm_shares_float", [(-100.0, 17.0), (500.0, -1.0)])
def test_signed_positions_and_cash_keep_value_table_without_donut(market_fixture_tuple, cash_float, crm_shares_float):
    market_fixture_tuple[4].update(cash_float=cash_float)
    market_fixture_tuple[4]["position_map_dict"]["CRM"] = crm_shares_float
    allocation_dict = _view_dict(market_fixture_tuple)["holdings_allocation_dict"]
    assert allocation_dict["available_bool"] and not allocation_dict["donut_available_bool"]


def test_current_holdings_do_not_replace_valued_eod_quantities(market_fixture_tuple):
    row_dict = market_fixture_tuple[0]["summary_dict"]["pod_row_dict_list"][0]
    row_dict["position_exposure_dict_list"] = [{"asset_str": "AMD", "share_float": 999}]
    allocation_dict = _view_dict(market_fixture_tuple)["holdings_allocation_dict"]
    assert allocation_dict["row_list"][0]["shares_float"] == 31
    assert allocation_dict["holdings_note_str"] == "Holdings changed since this close."
    row_dict["position_exposure_dict_list"] = None
    assert not _view_dict(market_fixture_tuple)["holdings_allocation_dict"]["holdings_note_str"]
    row_dict["position_exposure_dict_list"] = [{"asset_str": None, "share_float": 2}]
    allocation_dict = _view_dict(market_fixture_tuple)["holdings_allocation_dict"]
    assert allocation_dict["available_bool"] and not allocation_dict["holdings_note_str"]


def test_wrong_owner_never_reaches_optional_valuation_sources(market_fixture_tuple):
    market_fixture_tuple[0]["summary_dict"]["pod_row_dict_list"][0]["account_route_str"] = "FOREIGN"
    assert not _view_dict(market_fixture_tuple)["holdings_allocation_dict"]["available_bool"]
    assert not market_fixture_tuple[6]


def test_demo_and_remote_snapshot_scopes_never_access_local_market_sources(market_fixture_tuple):
    client_dict = market_fixture_tuple[0]["client_dict"]
    client_dict["is_demo"] = True
    _view_dict(market_fixture_tuple)
    client_dict.update(is_demo=False, operations_source="snapshot")
    _view_dict(market_fixture_tuple)
    assert not market_fixture_tuple[6]


def test_estimated_basis_and_date_are_visible_on_full_and_refresh_pages(market_fixture_tuple):
    workspace_dict, snapshot_obj, provider_obj, pod_id_str = market_fixture_tuple[:4]
    app_obj = create_app(provider_obj, workspace_snapshot_fn=lambda: (workspace_dict, snapshot_obj), now_fn=lambda: DEMO_NOW_TS)
    for path_str in ("/pods/" + pod_id_str, "/pods/" + pod_id_str + "/refresh"):
        response_obj = app_obj.test_client().get(path_str)
        assert response_obj.status_code == 200
        html_str = response_obj.get_data(as_text=True)
        assert "Estimated · " in html_str and "Close 2026-09-04" in html_str
        assert "Norgate unadjusted close" in html_str and "data-pod-allocation" in html_str
        assert "form-action 'none'" in response_obj.headers["Content-Security-Policy"]


def test_complete_official_holdings_do_not_read_market_estimate_sources(market_fixture_tuple, monkeypatch):
    workspace_dict, snapshot_obj, provider_obj, pod_id_str = market_fixture_tuple[:4]
    monkeypatch.setattr(pod_finance, "load_portfolio_cash_list", lambda *args, **kwargs: [{"cash_float": 500.0}])
    route_str = workspace_dict["operations_account_list"][0]["account_route"]
    nav_obj = next(row_obj for row_obj in snapshot_obj.row_tuple if row_obj.account_route_str == route_str and row_obj.market_date_str == "2026-09-04")
    monkeypatch.setattr(pod_finance, "load_close_holdings_dict", lambda *args, **kwargs: {
        "available_bool": True, "close_date_str": "2026-09-04",
        "position_list": [{"symbol_str": "AMD", "shares_float": 31, "value_float": float(nav_obj.closing_nav_decimal) - 500}]})
    result_dict = pod_finance.build_pod_finance_dict(workspace_dict, snapshot_obj, provider_obj,
        pod_id_str=pod_id_str, as_of_ts=DEMO_NOW_TS, performance_db_path_str="report.sqlite3")
    assert result_dict["holdings_allocation_dict"]["available_bool"]
    assert not result_dict["holdings_allocation_dict"].get("estimated_bool")
    assert not market_fixture_tuple[6]
