"""Pod finance retains official account facts and exact LIVE ownership."""

from copy import deepcopy
from contextlib import closing
from dataclasses import replace
from datetime import UTC, datetime
import sqlite3

import pytest

from alpha.live.client_reporting import build_client_report_dict
from alpha.live.dashboard_v4.demo import build_demo_workspace_tuple, DEMO_NOW_TS
from alpha.live.dashboard_v4.pod_finance import build_pod_finance_dict
from test_dashboard_operator_access import ForbiddenProvider


def _fixture_tuple():
    workspace_dict, snapshot_obj, provider_obj = build_demo_workspace_tuple()
    pod_id_str = workspace_dict["operations_account_list"][0]["pod_id"]
    return workspace_dict, snapshot_obj, provider_obj, pod_id_str


def _view(workspace_dict, snapshot_obj, provider_obj, pod_id_str, **option_dict):
    return build_pod_finance_dict(workspace_dict, snapshot_obj, provider_obj,
        pod_id_str=pod_id_str, as_of_ts=option_dict.pop("as_of_ts", DEMO_NOW_TS), **option_dict)


def test_one_pod_values_and_returns_match_canonical_account_not_book():
    workspace_dict, snapshot_obj, provider_obj, pod_id_str = _fixture_tuple()
    before_dict = deepcopy(workspace_dict)
    result_dict = _view(workspace_dict, snapshot_obj, provider_obj, pod_id_str)
    selected_dict = deepcopy(workspace_dict["client_dict"])
    selected_dict["accounts"] = [selected_dict["accounts"][0]]
    for index_int, from_str in ((1, "2026-09-04"), (2, "2026-09-01"), (3, "2026-06-01")):
        account_dict = build_client_report_dict(selected_dict, snapshot_obj,
            from_date_str=from_str, to_date_str="2026-09-04", as_of_ts=DEMO_NOW_TS)["strategy_list"][0]
        if index_int == 1:
            assert result_dict["tile_list"][0]["value_str"] == f"${account_dict['closing_nav_float']:,.2f}"
            sign_str = "−" if account_dict["pnl_float"] < 0 else "+"
            assert result_dict["tile_list"][1]["value_str"] == f"{sign_str}${abs(account_dict['pnl_float']):,.2f}"
            assert result_dict["tile_list"][1]["detail_str"] == f"{account_dict['twr_float']:+.2%}"
        else:
            assert result_dict["tile_list"][index_int]["value_str"] == f"{account_dict['twr_float']:+.2%}"
    assert result_dict["tile_list"][3]["detail_str"] == "From 2026-06-01"
    assert result_dict["chart_dict"]["available_bool"] is True
    assert result_dict["reference_summary_str"] == "Live vs backtest unavailable"
    assert workspace_dict == before_dict


def test_other_account_missing_history_does_not_replace_selected_account_facts():
    workspace_dict, snapshot_obj, provider_obj, pod_id_str = _fixture_tuple()
    baseline_dict = _view(workspace_dict, snapshot_obj, provider_obj, pod_id_str)
    route_str = workspace_dict["operations_account_list"][0]["account_route"]
    snapshot_obj = replace(snapshot_obj, row_tuple=tuple(row_obj for row_obj in snapshot_obj.row_tuple if row_obj.account_route_str == route_str))
    result_dict = _view(workspace_dict, snapshot_obj, provider_obj, pod_id_str)
    assert result_dict["tile_list"] == baseline_dict["tile_list"]
    assert result_dict["chart_dict"] == baseline_dict["chart_dict"]


@pytest.mark.parametrize("period_str", ["1M", "3M", "YTD", "All"])
def test_chart_range_does_not_change_period_tiles(period_str):
    workspace_dict, snapshot_obj, provider_obj, pod_id_str = _fixture_tuple()
    result_dict = _view(workspace_dict, snapshot_obj, provider_obj, pod_id_str, period_str=period_str)
    assert result_dict["tile_list"] == _view(workspace_dict, snapshot_obj, provider_obj, pod_id_str)["tile_list"]
    assert result_dict["chart_dict"]["end_x_float"] == 496


@pytest.mark.parametrize("conflict_str", ["mode", "route", "duplicate_row", "duplicate_owner", "duplicate_valuation", "source_shared_route", "shared_route", "history_route"])
def test_ambiguous_or_nonlive_identity_withholds_everything_without_io(conflict_str):
    workspace_dict, snapshot_obj, _, pod_id_str = _fixture_tuple()
    row_list = workspace_dict["summary_dict"]["pod_row_dict_list"]
    if conflict_str == "mode":
        row_list[0]["mode_str"] = "paper"
    elif conflict_str == "route":
        row_list[0]["account_route_str"] = row_list[1]["account_route_str"]
    elif conflict_str == "duplicate_row":
        row_list.append(deepcopy(row_list[0]))
    elif conflict_str == "duplicate_owner":
        workspace_dict["operations_account_list"] = [*workspace_dict["operations_account_list"], deepcopy(workspace_dict["operations_account_list"][0])]
    elif conflict_str == "duplicate_valuation":
        workspace_dict["valuation_account_list"] = [*workspace_dict["valuation_account_list"], deepcopy(workspace_dict["valuation_account_list"][0])]
    elif conflict_str == "source_shared_route":
        row_list[1]["account_route_str"] = row_list[0]["account_route_str"]
    elif conflict_str == "shared_route":
        workspace_dict["valuation_account_list"] = deepcopy(workspace_dict["valuation_account_list"])
        workspace_dict["valuation_account_list"][1]["account_route"] = row_list[0]["account_route_str"]
    else:
        workspace_dict["client_dict"]["accounts"] = deepcopy(workspace_dict["client_dict"]["accounts"])
        workspace_dict["client_dict"]["accounts"][0]["account_route"] = "OTHER_ROUTE"
    result_dict = _view(workspace_dict, snapshot_obj, ForbiddenProvider(), pod_id_str)
    assert all(tile_dict["value_str"] == "—" for tile_dict in result_dict["tile_list"])
    assert result_dict["position_list"] == []
    assert result_dict["cash_str"] == "—"
    assert result_dict["financial_error_str"]


def test_unknown_start_keeps_saved_account_nav_but_no_returns():
    workspace_dict, snapshot_obj, provider_obj, pod_id_str = _fixture_tuple()
    workspace_dict["client_dict"]["accounts"] = workspace_dict["client_dict"]["accounts"][1:]
    workspace_dict["financial_scope_complete_bool"] = False
    result_dict = _view(workspace_dict, snapshot_obj, provider_obj, pod_id_str)
    assert result_dict["tile_list"][0]["value_str"] != "—"
    assert all(tile_dict["value_str"] == "—" for tile_dict in result_dict["tile_list"][1:])
    assert result_dict["chart_dict"]["available_bool"] is False


def test_closed_history_does_not_present_a_partial_month_as_current_period():
    workspace_dict, snapshot_obj, provider_obj, pod_id_str = _fixture_tuple()
    workspace_dict["client_dict"]["accounts"] = deepcopy(workspace_dict["client_dict"]["accounts"])
    workspace_dict["client_dict"]["accounts"][0]["effective_to"] = "2026-09-02"
    result_dict = _view(workspace_dict, snapshot_obj, provider_obj, pod_id_str)
    assert result_dict["tile_list"][0]["value_str"] != "—"
    assert all(tile_dict["value_str"] == "—" for tile_dict in result_dict["tile_list"][1:])


def test_saved_share_quantities_keep_their_own_date_and_no_price_claims():
    workspace_dict, snapshot_obj, provider_obj, pod_id_str = _fixture_tuple()
    row_dict = workspace_dict["summary_dict"]["pod_row_dict_list"][0]
    row_dict.update(latest_pod_state_timestamp_str="2026-09-08T13:36:12+00:00",
        position_exposure_dict_list=[{"asset_str": "AMD", "share_float": 31, "price_float": 999},
            {"asset_str": "CRM", "share_float": 17, "price_float": None}])
    result_dict = _view(workspace_dict, snapshot_obj, provider_obj, pod_id_str)
    assert result_dict["position_list"] == [{"symbol_str": "AMD", "shares_str": "31"}, {"symbol_str": "CRM", "shares_str": "17"}]
    assert result_dict["positions_basis_str"] == "Saved positions"
    assert result_dict["position_asof_str"] == "2026-09-08 09:36:12"
    assert "price_asof_str" not in result_dict
    assert result_dict["money_asof_str"] == "Demo · Money as of close 2026-09-04"
    assert result_dict["cash_asof_str"] == "Close 2026-09-04"
    assert result_dict["cash_str"] != "—"


@pytest.mark.parametrize("timestamp_str", [None, "not-time", "2027-01-01T00:00:00+00:00", "2026-09-04T20:10:00"])
def test_old_or_invalid_reference_prices_cannot_change_quantities_or_finance(timestamp_str):
    workspace_dict, snapshot_obj, provider_obj, pod_id_str = _fixture_tuple()
    baseline_dict = _view(workspace_dict, snapshot_obj, provider_obj, pod_id_str)
    row_dict = workspace_dict["summary_dict"]["pod_row_dict_list"][0]
    row_dict["latest_live_reference_snapshot_timestamp_str"] = timestamp_str
    for position_dict in row_dict["position_exposure_dict_list"]:
        position_dict["price_float"] = float("nan")
    result_dict = _view(workspace_dict, snapshot_obj, provider_obj, pod_id_str)
    assert result_dict["position_list"] == baseline_dict["position_list"]
    assert result_dict["tile_list"] == baseline_dict["tile_list"]
    assert all(set(position_dict) == {"symbol_str", "shares_str"} for position_dict in result_dict["position_list"])


def test_unpriced_and_short_quantities_remain_visible_without_allocations():
    workspace_dict, snapshot_obj, provider_obj, pod_id_str = _fixture_tuple()
    workspace_dict["summary_dict"]["pod_row_dict_list"][0]["position_exposure_dict_list"] = [
        {"asset_str": "SHORT", "share_float": -1.25, "price_float": 100},
        {"asset_str": "UNPRICED", "share_float": 3, "price_float": None},
        {"asset_str": "EXITED", "share_float": 0, "price_float": 999}]
    result_dict = _view(workspace_dict, snapshot_obj, provider_obj, pod_id_str)
    assert result_dict["position_list"] == [{"symbol_str": "SHORT", "shares_str": "-1.25"},
        {"symbol_str": "UNPRICED", "shares_str": "3"}]


@pytest.mark.parametrize("timestamp_str", [None, "not-time", "2027-01-01T00:00:00+00:00", "2026-09-04T20:10:00"])
def test_unknown_or_future_position_time_withholds_quantities(timestamp_str):
    workspace_dict, snapshot_obj, provider_obj, pod_id_str = _fixture_tuple()
    workspace_dict["summary_dict"]["pod_row_dict_list"][0]["latest_pod_state_timestamp_str"] = timestamp_str
    result_dict = _view(workspace_dict, snapshot_obj, provider_obj, pod_id_str)
    assert result_dict["position_list"] == []
    assert not result_dict["positions_available_bool"]


def test_duplicate_symbols_are_ambiguous_and_future_state_is_hidden():
    workspace_dict, snapshot_obj, provider_obj, pod_id_str = _fixture_tuple()
    row_dict = workspace_dict["summary_dict"]["pod_row_dict_list"][0]
    row_dict["position_exposure_dict_list"] *= 2
    assert _view(workspace_dict, snapshot_obj, provider_obj, pod_id_str)["position_list"] == []
    row_dict["position_exposure_dict_list"] = row_dict["position_exposure_dict_list"][:1]
    row_dict["latest_pod_state_timestamp_str"] = "2027-01-01T00:00:00+00:00"
    assert _view(workspace_dict, snapshot_obj, provider_obj, pod_id_str)["position_list"] == []


def test_cash_does_not_borrow_another_date_and_old_import_keeps_current_month_blank():
    workspace_dict, snapshot_obj, provider_obj, pod_id_str = _fixture_tuple()
    workspace_dict["summary_dict"]["pod_row_dict_list"][0]["eod_snapshot_dict"]["latest_market_date_str"] = "2026-09-03"
    # The SQLite-backed demo also has authoritative local EOD history. Remove
    # the selected date from that synthetic source, not only from its summary.
    with closing(sqlite3.connect(provider_obj.get_target_for_pod(pod_id_str).db_path_str)) as connection_obj, connection_obj:
        connection_obj.execute("UPDATE pod_state_history SET updated_timestamp_str='2026-09-03T20:10:01+00:00' WHERE snapshot_stage_str='eod'")
    result_dict = _view(workspace_dict, snapshot_obj, provider_obj, pod_id_str,
        as_of_ts=datetime(2026, 10, 10, 12, tzinfo=UTC))
    assert result_dict["cash_str"] == "—"
    assert result_dict["tile_list"][2]["value_str"] == "—"
    assert result_dict["tile_list"][0]["value_str"] != "—"
    assert result_dict["delayed_bool"] is True


def test_today_and_duplicate_nav_are_not_finalized_financial_facts():
    workspace_dict, snapshot_obj, provider_obj, pod_id_str = _fixture_tuple()
    route_str = workspace_dict["operations_account_list"][0]["account_route"]
    latest_obj = next(row_obj for row_obj in snapshot_obj.row_tuple if row_obj.account_route_str == route_str and row_obj.market_date_str == "2026-09-04")
    future_obj = replace(latest_obj, market_date_str="2026-09-08")
    updated_obj = replace(snapshot_obj, row_tuple=(*snapshot_obj.row_tuple, future_obj))
    assert _view(workspace_dict, updated_obj, provider_obj, pod_id_str)["tile_list"] == _view(workspace_dict, snapshot_obj, provider_obj, pod_id_str)["tile_list"]
    duplicate_obj = replace(snapshot_obj, row_tuple=(*snapshot_obj.row_tuple, latest_obj))
    assert all(tile_dict["value_str"] == "—" for tile_dict in _view(workspace_dict, duplicate_obj, provider_obj, pod_id_str)["tile_list"])
