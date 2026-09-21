"""Saved IBKR portfolio allocation is independent of finalized Flex reporting."""

from copy import deepcopy
from contextlib import closing
from dataclasses import replace
import json
import sqlite3

import pytest

from alpha.live.dashboard_v4 import pod_finance
from alpha.live.dashboard_v4.app import create_app
from alpha.live.dashboard_v4.demo import DEMO_NOW_TS, build_demo_workspace_tuple
from alpha.live.dashboard_v4.pod_broker_holdings import load_broker_holdings_dict


@pytest.fixture
def broker_fixture_tuple(monkeypatch):
    workspace_dict, snapshot_obj, provider_obj = build_demo_workspace_tuple()
    workspace_dict["client_dict"].update(is_demo=False, operations_source="local")
    pod_id_str = workspace_dict["operations_account_list"][0]["pod_id"]
    holdings_dict = {"available_bool": True, "reason_str": "", "cash_float": 500.0,
        "broker_nav_float": 7100.0, "observed_timestamp_str": "2026-09-04T20:10:00+00:00",
        "source_str": "IBKR portfolio", "position_list": [
            {"symbol_str": "AMD", "shares_float": 31.0, "value_float": 3100.0},
            {"symbol_str": "CRM", "shares_float": 17.0, "value_float": 3400.0}]}
    call_list = []

    def broker_reader(target_obj, **option_dict):
        assert target_obj.release_obj.pod_id_str == pod_id_str
        assert option_dict == {"as_of_ts": DEMO_NOW_TS}
        call_list.append("saved_ibkr")
        return deepcopy(holdings_dict)

    monkeypatch.setattr(pod_finance, "load_broker_holdings_dict", broker_reader)
    try:
        yield workspace_dict, snapshot_obj, provider_obj, pod_id_str, holdings_dict, call_list
    finally:
        provider_obj.close()


def _view_dict(fixture_tuple):
    workspace_dict, snapshot_obj, provider_obj, pod_id_str = fixture_tuple[:4]
    return pod_finance.build_pod_finance_dict(workspace_dict, snapshot_obj, provider_obj,
        pod_id_str=pod_id_str, as_of_ts=DEMO_NOW_TS)


def test_saved_values_cash_and_denominator_do_not_modify_reported_money(broker_fixture_tuple):
    workspace_dict, snapshot_obj, provider_obj, pod_id_str = broker_fixture_tuple[:4]
    baseline_dict = pod_finance._build_report_finance_dict(workspace_dict, snapshot_obj, provider_obj,
        pod_id_str=pod_id_str, as_of_ts=DEMO_NOW_TS)
    view_dict = _view_dict(broker_fixture_tuple)
    allocation_dict = view_dict["holdings_allocation_dict"]
    assert allocation_dict["available_bool"] and allocation_dict["donut_available_bool"]
    assert allocation_dict["broker_observation_bool"] is True
    assert allocation_dict["holdings_total_float"] == 7000
    assert allocation_dict["broker_nav_float"] == 7100
    assert [row_dict["value_float"] for row_dict in allocation_dict["row_list"]] == [3100, 3400]
    assert allocation_dict["row_list"][0]["weight_float"] == pytest.approx(3100 / 7000)
    assert allocation_dict["cash_row_dict"]["value_float"] == 500
    for field_str in ("tile_list", "chart_dict", "money_asof_str", "cash_str"):
        assert view_dict[field_str] == baseline_dict[field_str]


def test_missing_flex_does_not_hide_saved_broker_allocation(broker_fixture_tuple):
    fixture_list = list(broker_fixture_tuple)
    fixture_list[0]["financial_error_str"] = "No Flex report"
    fixture_list[1] = replace(fixture_list[1], row_tuple=(), unavailable_reason_str="No Flex report")
    view_dict = _view_dict(fixture_list)
    assert view_dict["holdings_allocation_dict"]["available_bool"]
    assert all(tile_dict["value_str"] == "—" for tile_dict in view_dict["tile_list"])
    assert not view_dict["chart_dict"]["available_bool"]


def test_unavailable_broker_values_keep_quantities_and_explain_reason(broker_fixture_tuple):
    broker_fixture_tuple[4].update(available_bool=False, reason_str="IBKR position values not saved yet")
    view_dict = _view_dict(broker_fixture_tuple)
    assert view_dict["position_list"]
    assert not view_dict["holdings_allocation_dict"]["available_bool"]
    assert view_dict["holdings_allocation_dict"]["reason_str"] == "IBKR position values not saved yet"
    assert broker_fixture_tuple[5] == ["saved_ibkr"]


def test_cash_only_is_a_complete_portfolio(broker_fixture_tuple):
    broker_fixture_tuple[4]["position_list"] = []
    allocation_dict = _view_dict(broker_fixture_tuple)["holdings_allocation_dict"]
    assert allocation_dict["available_bool"] and allocation_dict["donut_available_bool"]
    assert allocation_dict["cash_percent_str"] == "100.0%"
    assert allocation_dict["row_list"] == []


@pytest.mark.parametrize("cash_float,crm_shares_float,crm_value_float", [(-100.0, 17.0, 3400.0), (500.0, -1.0, -200.0)])
def test_signed_positions_and_cash_keep_table_without_donut(broker_fixture_tuple, cash_float, crm_shares_float, crm_value_float):
    broker_fixture_tuple[4]["cash_float"] = cash_float
    broker_fixture_tuple[4]["position_list"][1].update(shares_float=crm_shares_float, value_float=crm_value_float)
    allocation_dict = _view_dict(broker_fixture_tuple)["holdings_allocation_dict"]
    assert allocation_dict["available_bool"] and not allocation_dict["donut_available_bool"]


def test_newer_holdings_do_not_replace_valued_snapshot_quantities(broker_fixture_tuple):
    row_dict = broker_fixture_tuple[0]["summary_dict"]["pod_row_dict_list"][0]
    row_dict["position_exposure_dict_list"] = [{"asset_str": "AMD", "share_float": 999}]
    row_dict["latest_pod_state_timestamp_str"] = "2026-09-04T20:10:01+00:00"
    allocation_dict = _view_dict(broker_fixture_tuple)["holdings_allocation_dict"]
    assert allocation_dict["row_list"][0]["shares_float"] == 31
    assert allocation_dict["holdings_note_str"] == "Holdings changed since this snapshot."
    row_dict["position_exposure_dict_list"] = None
    assert not _view_dict(broker_fixture_tuple)["holdings_allocation_dict"]["holdings_note_str"]
    row_dict["position_exposure_dict_list"] = [{"asset_str": None, "share_float": 2}]
    allocation_dict = _view_dict(broker_fixture_tuple)["holdings_allocation_dict"]
    assert allocation_dict["available_bool"] and not allocation_dict["holdings_note_str"]


@pytest.mark.parametrize("timestamp_str", ["2026-09-04T20:09:59+00:00", "2026-09-04T20:10:00+00:00", "bad"])
def test_old_simultaneous_or_invalid_comparison_does_not_claim_later_change(broker_fixture_tuple, timestamp_str):
    row_dict = broker_fixture_tuple[0]["summary_dict"]["pod_row_dict_list"][0]
    row_dict.update(position_exposure_dict_list=[{"asset_str": "AMD", "share_float": 999}],
        latest_pod_state_timestamp_str=timestamp_str)
    allocation_dict = _view_dict(broker_fixture_tuple)["holdings_allocation_dict"]
    assert allocation_dict["available_bool"] and not allocation_dict["holdings_note_str"]


def test_invalid_newer_quantity_comparison_cannot_hide_valid_saved_values(broker_fixture_tuple):
    row_dict = broker_fixture_tuple[0]["summary_dict"]["pod_row_dict_list"][0]
    row_dict.update(position_exposure_dict_list=[{"asset_str": "AMD", "share_float": 10 ** 500}],
        latest_pod_state_timestamp_str="2026-09-04T20:10:01+00:00")
    allocation_dict = _view_dict(broker_fixture_tuple)["holdings_allocation_dict"]
    assert allocation_dict["available_bool"] and not allocation_dict["holdings_note_str"]


def test_wrong_owner_never_reaches_optional_valuation_source(broker_fixture_tuple):
    broker_fixture_tuple[0]["summary_dict"]["pod_row_dict_list"][0]["account_route_str"] = "FOREIGN"
    assert not _view_dict(broker_fixture_tuple)["holdings_allocation_dict"]["available_bool"]
    assert not broker_fixture_tuple[5]


def test_demo_and_remote_snapshot_scopes_never_access_local_state(broker_fixture_tuple):
    client_dict = broker_fixture_tuple[0]["client_dict"]
    client_dict["is_demo"] = True
    _view_dict(broker_fixture_tuple)
    client_dict.update(is_demo=False, operations_source="snapshot")
    _view_dict(broker_fixture_tuple)
    assert not broker_fixture_tuple[5]


def test_broker_source_and_actual_observed_time_visible_on_full_and_refresh_pages(broker_fixture_tuple):
    workspace_dict, snapshot_obj, provider_obj, pod_id_str = broker_fixture_tuple[:4]
    app_obj = create_app(provider_obj, workspace_snapshot_fn=lambda: (workspace_dict, snapshot_obj), now_fn=lambda: DEMO_NOW_TS)
    for path_str in ("/pods/" + pod_id_str, "/pods/" + pod_id_str + "/refresh"):
        response_obj = app_obj.test_client().get(path_str)
        assert response_obj.status_code == 200
        panel_str = response_obj.get_data(as_text=True).split('<section class="panel pod-holdings"', 1)[1].split("</section>", 1)[0]
        assert "IBKR · 2026-09-04 16:10:00 ET" in panel_str
        assert "Close " not in panel_str and "Estimated" not in panel_str and "Norgate" not in panel_str
        assert 'data-close-date="2026-09-04T20:10:00+00:00"' in panel_str
        assert "form-action 'none'" in response_obj.headers["Content-Security-Policy"]


def test_complete_official_holdings_keep_priority(broker_fixture_tuple, monkeypatch):
    workspace_dict, snapshot_obj, provider_obj, pod_id_str = broker_fixture_tuple[:4]
    monkeypatch.setattr(pod_finance, "load_portfolio_cash_list", lambda *args, **kwargs: [{"cash_float": 500.0}])
    route_str = workspace_dict["operations_account_list"][0]["account_route"]
    nav_obj = next(row_obj for row_obj in snapshot_obj.row_tuple if row_obj.account_route_str == route_str and row_obj.market_date_str == "2026-09-04")
    monkeypatch.setattr(pod_finance, "load_close_holdings_dict", lambda *args, **kwargs: {
        "available_bool": True, "close_date_str": "2026-09-04",
        "position_list": [{"symbol_str": "AMD", "shares_float": 31, "value_float": float(nav_obj.closing_nav_decimal) - 500}]})
    result_dict = pod_finance.build_pod_finance_dict(workspace_dict, snapshot_obj, provider_obj,
        pod_id_str=pod_id_str, as_of_ts=DEMO_NOW_TS, performance_db_path_str="report.sqlite3")
    assert result_dict["holdings_allocation_dict"]["available_bool"]
    assert not result_dict["holdings_allocation_dict"].get("broker_observation_bool")
    assert not broker_fixture_tuple[5]


def test_public_page_reads_real_saved_payload_without_price_service(broker_fixture_tuple, monkeypatch):
    workspace_dict, snapshot_obj, provider_obj, pod_id_str, holdings_dict, _ = broker_fixture_tuple
    target_obj = provider_obj.get_target_for_pod(pod_id_str)
    payload_dict = deepcopy(holdings_dict)
    payload_dict.update(schema_version_int=1, currency_str="USD",
        account_route_str=target_obj.release_obj.account_route_str,
        owner_dict={field_str: getattr(target_obj.release_obj, field_str)
            for field_str in ("release_id_str", "user_id_str", "pod_id_str", "account_route_str", "mode_str")})
    for conid_int, position_dict in enumerate(payload_dict["position_list"], 1):
        position_dict.update(conid_int=conid_int, currency_str="USD",
            market_price_float=position_dict["value_float"] / position_dict["shares_float"])
    with closing(sqlite3.connect(target_obj.db_path_str)) as connection_obj:
        connection_obj.execute("UPDATE broker_snapshot_cache SET portfolio_valuation_json_str=? WHERE account_route_str=?",
            (json.dumps(payload_dict), target_obj.release_obj.account_route_str))
        connection_obj.commit()
    monkeypatch.setattr(pod_finance, "load_broker_holdings_dict", load_broker_holdings_dict)
    workspace_dict["financial_error_str"] = "No Flex report"
    snapshot_obj = replace(snapshot_obj, row_tuple=(), unavailable_reason_str="No Flex report")
    app_obj = create_app(provider_obj, workspace_snapshot_fn=lambda: (workspace_dict, snapshot_obj), now_fn=lambda: DEMO_NOW_TS)
    response_obj = app_obj.test_client().get("/pods/" + pod_id_str)
    assert response_obj.status_code == 200
    panel_str = response_obj.get_data(as_text=True).split('<section class="panel pod-holdings"', 1)[1].split("</section>", 1)[0]
    assert "IBKR · 2026-09-04 16:10:00 ET" in panel_str
    assert "3,100.00" in panel_str and "3,400.00" in panel_str and "500.00" in panel_str
