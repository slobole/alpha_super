"""Positions never substitutes reference prices or account profit for holdings."""

from copy import deepcopy
from dataclasses import replace
from datetime import timedelta
from types import SimpleNamespace

import pytest

from alpha.live.client_reporting import build_client_report_dict
from alpha.live.dashboard_v3.client_cash import load_portfolio_cash_list
from alpha.live.dashboard_v3.client_operations import build_client_operations_dict
from alpha.live.dashboard_v3.client_presentation import portfolio_allocation_dict
from alpha.live.dashboard_v4.demo import DEMO_NOW_TS, build_demo_workspace_tuple
from alpha.live.dashboard_v4.positions import build_positions_page_dict
from alpha.live.dashboard_v4.positions_data import load_positions_dict


@pytest.fixture(scope="module")
def base_tuple():
    workspace_dict, snapshot_obj, provider_obj = build_demo_workspace_tuple()
    source_dict = {account_dict["pod_id"]: load_positions_dict(provider_obj.get_target_for_pod(account_dict["pod_id"]), as_of_ts=DEMO_NOW_TS)
        for account_dict in workspace_dict["operations_account_list"]}
    yield workspace_dict, snapshot_obj, provider_obj, source_dict
    provider_obj.close()


@pytest.fixture
def fixture_tuple(base_tuple):
    workspace_dict, snapshot_obj, provider_obj, original_dict = base_tuple
    source_dict = deepcopy(original_dict)
    wrapper_obj = SimpleNamespace(get_target_for_pod=provider_obj.get_target_for_pod,
        get_positions_dict=lambda pod_id_str, **kwargs: deepcopy(source_dict[pod_id_str]))
    workspace_dict = deepcopy(workspace_dict)
    workspace_dict["operations_account_list"] = deepcopy(workspace_dict["operations_account_list"])
    workspace_dict["valuation_account_list"] = deepcopy(workspace_dict["valuation_account_list"])
    return workspace_dict, snapshot_obj, wrapper_obj, source_dict


def _page(fixture_tuple, **option_dict):
    return build_positions_page_dict(*fixture_tuple[:3], as_of_ts=option_dict.pop("as_of_ts", DEMO_NOW_TS), **option_dict)


def test_saved_demo_rows_have_shares_but_no_reference_values_or_position_pnl(fixture_tuple):
    workspace_dict, _, _, source_dict = fixture_tuple
    before_dict = deepcopy(workspace_dict)
    result_dict = _page(fixture_tuple)
    assert result_dict["holdings_complete_bool"] is True
    assert result_dict["row_list"]
    assert result_dict["all_count_int"] == len({symbol_str for source_row_dict in source_dict.values()
        for symbol_str, share_float in source_row_dict["position_map_dict"].items() if share_float})
    for row_dict in result_dict["row_list"]:
        assert row_dict["value_str"] == row_dict["weight_str"] == row_dict["pl_str"] == row_dict["pl_percent_str"] == "—"
        assert row_dict["weight_percent_float"] is None
        assert row_dict["today_str"] == ""
        assert row_dict["name_str"] == ""
        assert all("ET" in holder_dict["position_asof_str"] for holder_dict in row_dict["pod_list"])
    assert workspace_dict == before_dict


def test_merge_keeps_offsetting_open_legs_even_with_zero_net_shares(fixture_tuple):
    source_dict = fixture_tuple[3]
    for source_row_dict in source_dict.values():
        source_row_dict["position_map_dict"] = {}
    pod_list = list(source_dict)
    source_dict[pod_list[0]]["position_map_dict"] = {"SHARED": 10}
    source_dict[pod_list[1]]["position_map_dict"] = {"SHARED": -10}
    result_dict = _page(fixture_tuple)
    assert result_dict["all_count_int"] == 1
    row_dict = result_dict["row_list"][0]
    assert row_dict["share_str"] == "0"
    assert row_dict["offset_bool"] is True
    assert [holder_dict["share_str"] for holder_dict in row_dict["pod_list"]] == ["10", "-10"]
    assert [row_dict["count_str"] for row_dict in result_dict["pod_row_list"]] == ["1", "1", "0", "0"]
    selected_dict = _page(fixture_tuple, pod_str=pod_list[1])
    assert selected_dict["row_list"][0]["share_str"] == "-10"
    assert [holder_dict["pod_id_str"] for holder_dict in selected_dict["row_list"][0]["pod_list"]] == [pod_list[1]]
    assert selected_dict["row_list"][0]["offset_bool"] is False
    assert selected_dict["tile_list"] == result_dict["tile_list"]
    assert selected_dict["pod_row_list"] == result_dict["pod_row_list"]
    assert selected_dict["total_dict"] == result_dict["total_dict"]


def test_all_filter_count_follows_selected_pod_but_portfolio_count_does_not(fixture_tuple):
    pod_id_str = next(iter(fixture_tuple[3]))
    full_dict = _page(fixture_tuple)
    selected_dict = _page(fixture_tuple, pod_str=pod_id_str)
    assert selected_dict["all_count_int"] == len(selected_dict["row_list"])
    assert selected_dict["all_count_int"] < full_dict["all_count_int"]
    assert selected_dict["total_dict"] == full_dict["total_dict"]
    assert selected_dict["verdict_str"] == "3 of 8 positions · DVO2"
    assert full_dict["verdict_str"] == "8 saved positions"


def test_account_totals_use_exact_canonical_eod_allocation_independent_of_share_dates(fixture_tuple):
    workspace_dict, snapshot_obj, provider_obj, source_dict = fixture_tuple
    for source_row_dict in source_dict.values():
        source_row_dict.update(position_timestamp_str=DEMO_NOW_TS.isoformat(), source_str="broker_snapshot", timestamp_basis_str="observed")
    operations_dict = build_client_operations_dict(workspace_dict["client_dict"], workspace_dict["summary_dict"],
        as_of_ts=DEMO_NOW_TS, local_account_list=workspace_dict["operations_account_list"])
    report_dict = build_client_report_dict(workspace_dict["client_dict"], snapshot_obj,
        from_date_str="2026-09-04", to_date_str="2026-09-04", as_of_ts=DEMO_NOW_TS,
        scope_complete_bool=True, valuation_account_list=workspace_dict["valuation_account_list"])
    cash_list = load_portfolio_cash_list(workspace_dict["client_dict"], report_dict, provider_obj, operations_dict, as_of_ts=DEMO_NOW_TS)
    allocation_dict = portfolio_allocation_dict(report_dict, snapshot_obj, cash_snapshot_list=cash_list)
    result_dict = _page(fixture_tuple)
    assert result_dict["financial_asof_str"] == "Close 2026-09-04"
    assert "2026-09-08" in result_dict["asof_str"]
    assert result_dict["financial_basis_str"] == allocation_dict["basis_str"]
    expected_invested_float = sum(item_dict["value_float"] - item_dict["cash_float"] for item_dict in allocation_dict["item_list"])
    assert result_dict["tile_list"][0]["value_str"] == f"${expected_invested_float:,.2f}"
    assert result_dict["total_dict"]["invested_str"] == result_dict["tile_list"][0]["value_str"]
    assert all(row_dict["pnl_str"] == "—" for row_dict in result_dict["pod_row_list"])


def test_missing_account_finance_does_not_hide_saved_broker_positions(fixture_tuple):
    workspace_dict, snapshot_obj, provider_obj, source_dict = fixture_tuple
    missing_route_str = workspace_dict["valuation_account_list"][0]["account_route"]
    snapshot_obj = replace(snapshot_obj, row_tuple=tuple(row_obj for row_obj in snapshot_obj.row_tuple if row_obj.account_route_str != missing_route_str))
    result_dict = _page((workspace_dict, snapshot_obj, provider_obj, source_dict))
    assert result_dict["holdings_complete_bool"] is True
    assert result_dict["row_list"]
    assert result_dict["total_dict"]["weight_str"] == "—"
    assert result_dict["tile_list"][0]["value_str"] == "—"
    assert result_dict["financial_error_str"]


def test_incomplete_broker_cash_cannot_be_subtracted_from_flex_nav(fixture_tuple, monkeypatch):
    monkeypatch.setattr("alpha.live.dashboard_v4.positions.load_portfolio_cash_list", lambda *args, **kwargs: [])
    result_dict = _page(fixture_tuple)
    assert result_dict["holdings_complete_bool"] is True
    assert result_dict["total_dict"]["weight_str"] == "100.0%"
    assert result_dict["financial_basis_str"] == "Finalized IBKR account value"
    assert result_dict["total_dict"]["invested_str"] == result_dict["total_dict"]["cash_str"] == "—"
    assert all(row_dict["cash_str"] == row_dict["invested_str"] == "—" for row_dict in result_dict["pod_row_list"])


def test_delayed_financial_close_stays_explicit_in_date_label(fixture_tuple):
    result_dict = _page(fixture_tuple, as_of_ts=DEMO_NOW_TS + timedelta(days=1))
    assert result_dict["financial_delayed_bool"] is True
    assert result_dict["financial_asof_str"] == "Close 2026-09-04 · delayed"


def test_missing_position_source_cannot_prove_flat_book_or_erase_other_pods(fixture_tuple):
    source_dict = fixture_tuple[3]
    missing_pod_str = next(iter(source_dict))
    source_dict[missing_pod_str]["available_bool"] = False
    result_dict = _page(fixture_tuple)
    assert result_dict["holdings_complete_bool"] is False
    assert result_dict["missing_pod_list"] == [missing_pod_str]
    assert result_dict["pod_row_list"][0]["count_str"] == "—"
    assert result_dict["total_dict"]["count_str"] == "—"
    assert result_dict["row_list"]
    assert "incomplete" in result_dict["verdict_str"]


def test_inactive_owned_account_is_a_visible_coverage_gap_and_not_read(fixture_tuple):
    workspace_dict, _, provider_obj, _ = fixture_tuple
    missing_pod_str = workspace_dict["operations_account_list"].pop()["pod_id"]
    getter_fn = provider_obj.get_positions_dict
    def scoped_getter(pod_id_str, **option_dict):
        assert pod_id_str != missing_pod_str
        return getter_fn(pod_id_str, **option_dict)
    provider_obj.get_positions_dict = scoped_getter
    result_dict = _page(fixture_tuple)
    assert result_dict["missing_pod_list"] == [missing_pod_str]
    assert result_dict["pod_row_list"][-1]["pod_id_str"] == missing_pod_str
    assert result_dict["pod_row_list"][-1]["count_str"] == "—"
    assert len(result_dict["pod_filter_list"]) == 4


@pytest.mark.parametrize("source_change_dict", [{"position_map_dict": {"SPY": True}}, {"position_map_dict": {"SPY": float("nan")}},
    {"position_timestamp_str": "2026-09-08T09:35:00"}, {"position_timestamp_str": "2027-01-01T00:00:00+00:00"},
    {"account_route_str": "other"}, {"user_id_str": "other"}, {"release_id_str": "other"}, {"mode_str": "paper"},
    {"source_str": "pod_state"}, {"timestamp_basis_str": "invented"}])
def test_provider_results_are_revalidated_without_partial_maps(fixture_tuple, source_change_dict):
    source_dict = fixture_tuple[3]
    pod_id_str = next(iter(source_dict))
    source_dict[pod_id_str].update(source_change_dict)
    result_dict = _page(fixture_tuple)
    assert pod_id_str in result_dict["missing_pod_list"]
    assert all(holder_dict["pod_id_str"] != pod_id_str for row_dict in result_dict["row_list"] for holder_dict in row_dict["pod_list"])


@pytest.mark.parametrize("conflict_str", ["duplicate_account", "shared_route", "reassigned_history", "source_shared_route"])
def test_ambiguous_ownership_withholds_page_before_provider_io(fixture_tuple, conflict_str):
    workspace_dict, _, provider_obj, _ = fixture_tuple
    if conflict_str == "duplicate_account":
        workspace_dict["operations_account_list"].append(deepcopy(workspace_dict["operations_account_list"][0]))
    elif conflict_str == "shared_route":
        workspace_dict["valuation_account_list"][1]["account_route"] = workspace_dict["valuation_account_list"][0]["account_route"]
    elif conflict_str == "reassigned_history":
        workspace_dict["client_dict"]["accounts"][0]["account_route"] = "other"
    else:
        workspace_dict["summary_dict"]["pod_row_dict_list"][1]["account_route_str"] = workspace_dict["summary_dict"]["pod_row_dict_list"][0]["account_route_str"]
    provider_obj.get_target_for_pod = lambda *args: pytest.fail("Read ambiguous account")
    result_dict = _page(fixture_tuple)
    assert result_dict["row_list"] == []
    assert result_dict["holdings_complete_bool"] is False


def test_stale_page_keeps_dated_saved_positions_without_current_health_claim(fixture_tuple):
    result_dict = _page(fixture_tuple, as_of_ts=DEMO_NOW_TS + timedelta(seconds=121))
    assert result_dict["source_fresh_bool"] is False
    assert result_dict["row_list"]
    assert "Refresh unavailable" in result_dict["verdict_detail_str"]
    assert "Saved positions" in result_dict["asof_str"]


def test_search_is_literal_and_unproved_filters_do_not_claim_zero_events(fixture_tuple):
    source_dict = fixture_tuple[3]
    pod_id_str = next(iter(source_dict))
    source_dict[pod_id_str]["position_map_dict"] = {'<unsafe "symbol">': 1.25}
    result_dict = _page(fixture_tuple, search_str="UNSAFE")
    assert [row_dict["symbol_str"] for row_dict in result_dict["row_list"]] == ['<unsafe "symbol">']
    for view_str in ("changed", "off_target"):
        result_dict = _page(fixture_tuple, view_str=view_str)
        assert result_dict["row_list"] == []
        assert "unavailable" in result_dict["empty_str"]
        assert result_dict["changed_available_bool"] is result_dict["off_target_available_bool"] is False
