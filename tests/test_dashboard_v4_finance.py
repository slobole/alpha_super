"""V4 finance must display the saved V3 accounting facts without changing them."""

from copy import deepcopy
from dataclasses import replace
from datetime import UTC, datetime

import pytest

from alpha.live.client_reporting import BrokerReportingSnapshot, build_client_report_dict
from alpha.live.dashboard_v3.client_presentation import portfolio_allocation_dict
from alpha.live.dashboard_v3.demo import DemoOperationsProvider, build_demo_fixture_tuple
from alpha.live.dashboard_v4.finance import build_financial_overview_dict, _allocation_dict, _chart_dict
from test_client_cash import allocation_fixture_tuple
from test_client_reporting import client_config_dict, nav_attributes_dict, snapshot_obj
from test_dashboard_operator_access import ForbiddenProvider


AS_OF_TS = datetime(2026, 9, 5, 12, tzinfo=UTC)


def demo_fixture_tuple():
    registry_dict, snapshot_dict = build_demo_fixture_tuple()
    client_dict = next(client_dict for client_dict in registry_dict["clients"] if client_dict["client_id"] == "demo-owner")
    provider_obj = DemoOperationsProvider({"schema_version": 1, "clients": [client_dict]})
    workspace_dict = {"client_dict": client_dict, "financial_scope_complete_bool": True,
        "valuation_account_list": client_dict["accounts"], "summary_dict": provider_obj.get_summary_dict(),
        "operations_account_list": client_dict["accounts"], "financial_error_str": None}
    workspace_dict["summary_dict"]["as_of_timestamp_str"] = AS_OF_TS.isoformat()
    return workspace_dict, snapshot_dict["demo-owner"], provider_obj


def test_live_demo_tiles_match_canonical_periods_and_do_not_mutate_source():
    workspace_dict, source_obj, provider_obj = demo_fixture_tuple()
    before_dict = deepcopy(workspace_dict)
    result_dict = build_financial_overview_dict(workspace_dict, source_obj, provider_obj, as_of_ts=AS_OF_TS)
    client_dict = workspace_dict["client_dict"]
    day_dict = build_client_report_dict(client_dict, source_obj, from_date_str="2026-09-04", to_date_str="2026-09-04", as_of_ts=AS_OF_TS)
    month_dict = build_client_report_dict(client_dict, source_obj, from_date_str="2026-09-01", to_date_str="2026-09-04", as_of_ts=AS_OF_TS)
    year_dict = build_client_report_dict(client_dict, source_obj, from_date_str="2026-06-01", to_date_str="2026-09-04", as_of_ts=AS_OF_TS)
    assert result_dict["tile_list"][0]["value_str"] == f"${day_dict['closing_nav_float']:,.2f}"
    assert result_dict["tile_list"][1]["value_str"].replace("−", "-") == f"{'-' if day_dict['pnl_float'] < 0 else '+'}${abs(day_dict['pnl_float']):,.2f}"
    for index_int, period_dict in ((2, month_dict), (3, year_dict)):
        assert result_dict["tile_list"][index_int]["value_str"] == f"{period_dict['twr_float']:+.2%}"
    assert result_dict["money_asof_str"] == "Demo · Close 2026-09-04"
    assert result_dict["allocation_dict"]["cash_percent_str"] == "30.0%"
    assert result_dict["allocation_dict"]["total_invested_str"] == "70.0%"
    assert len(result_dict["allocation_dict"]["row_list"]) == 2
    assert not any("Free cash" in row_dict["name_str"] for row_dict in result_dict["allocation_dict"]["row_list"])
    assert workspace_dict == before_dict


@pytest.mark.parametrize("period_str", ["1M", "3M", "YTD", "All"])
def test_chart_period_changes_neither_tiles_nor_allocation(period_str):
    workspace_dict, source_obj, provider_obj = demo_fixture_tuple()
    result_dict = build_financial_overview_dict(workspace_dict, source_obj, provider_obj, period_str=period_str, as_of_ts=AS_OF_TS)
    baseline_dict = build_financial_overview_dict(workspace_dict, source_obj, provider_obj, period_str="All", as_of_ts=AS_OF_TS)
    assert result_dict["tile_list"] == baseline_dict["tile_list"]
    assert result_dict["allocation_dict"] == baseline_dict["allocation_dict"]
    assert result_dict["chart_dict"]["available_bool"] is True
    assert result_dict["chart_dict"]["end_x_float"] == 496
    assert all(18 <= tick_dict["y_float"] <= 216 for tick_dict in result_dict["chart_dict"]["y_tick_list"])
    if period_str == "1M":
        assert result_dict["chart_dict"]["x_tick_list"][0]["label_str"] == "Aug 05"
        assert result_dict["chart_dict"]["segment_list"] != baseline_dict["chart_dict"]["segment_list"]


def test_old_import_does_not_turn_current_month_into_previous_month():
    workspace_dict, source_obj, provider_obj = demo_fixture_tuple()
    result_dict = build_financial_overview_dict(workspace_dict, source_obj, provider_obj,
        as_of_ts=datetime(2026, 10, 10, 12, tzinfo=UTC), period_str="1M")
    assert result_dict["tile_list"][2]["value_str"] == "—"
    assert result_dict["tile_list"][2]["detail_str"] == "No data in this period"
    assert result_dict["tile_list"][0]["value_str"] != "—"
    assert result_dict["delayed_bool"] is True
    assert "Data delayed" in result_dict["money_asof_str"]
    assert result_dict["chart_dict"]["available_bool"] is False


@pytest.mark.parametrize("error_str", ["mapping", "snapshot", "missing"])
def test_missing_finance_withholds_every_total_without_provider_access(error_str):
    workspace_dict, source_obj, _ = demo_fixture_tuple()
    if error_str == "mapping":
        workspace_dict["financial_error_str"] = "Private path must not leak"
    elif error_str == "snapshot":
        source_obj = replace(source_obj, unavailable_reason_str="Private file cannot be loaded")
    else:
        source_obj = BrokerReportingSnapshot()
    result_dict = build_financial_overview_dict(workspace_dict, source_obj, ForbiddenProvider(), as_of_ts=AS_OF_TS)
    assert all(tile_dict["value_str"] == "—" for tile_dict in result_dict["tile_list"])
    assert result_dict["financial_error_str"]
    assert "Private" not in str(result_dict)
    assert result_dict["chart_dict"]["available_bool"] is False
    assert result_dict["allocation_dict"]["available_bool"] is False
    assert len(result_dict["allocation_dict"]["row_list"]) == 2
    assert all(row_dict["cash_str"] == "—" for row_dict in result_dict["allocation_dict"]["row_list"])


def test_unknown_local_scope_keeps_nav_but_never_unlocks_returns():
    workspace_dict, source_obj, provider_obj = demo_fixture_tuple()
    workspace_dict["financial_scope_complete_bool"] = False
    result_dict = build_financial_overview_dict(workspace_dict, source_obj, provider_obj, as_of_ts=AS_OF_TS)
    assert result_dict["tile_list"][0]["value_str"] != "—"
    assert all(tile_dict["value_str"] == "—" for tile_dict in result_dict["tile_list"][1:])
    assert result_dict["chart_dict"]["available_bool"] is False


def test_today_is_withheld_and_newer_partial_book_does_not_replace_last_common_close():
    client_dict = client_config_dict(second_bool=True)
    source_obj = snapshot_obj([
        nav_attributes_dict(account_str=route_str, date_str="2026-09-03") for route_str in ("U_TEST_A", "U_TEST_B")
    ] + [nav_attributes_dict(date_str="2026-09-04", opening_str="1010", closing_str="1020"),
        nav_attributes_dict(date_str="2026-09-05", opening_str="1020", closing_str="1030")])
    workspace_dict = {"client_dict": client_dict, "financial_scope_complete_bool": True, "summary_dict": {}}
    result_dict = build_financial_overview_dict(workspace_dict, source_obj, ForbiddenProvider(), as_of_ts=AS_OF_TS)
    assert result_dict["tile_list"][0]["value_str"] == "$2,020.00"
    assert result_dict["tile_list"][0]["detail_str"] == "Close 2026-09-03"
    assert result_dict["delayed_bool"] is True


def test_old_history_gap_withholds_month_but_does_not_hide_valid_day():
    client_dict = client_config_dict()
    source_obj = snapshot_obj([nav_attributes_dict(date_str="2026-09-01"),
        nav_attributes_dict(date_str="2026-09-03", opening_str="1020", closing_str="1030"),
        nav_attributes_dict(date_str="2026-09-04", opening_str="1030", closing_str="1040")])
    workspace_dict = {"client_dict": client_dict, "financial_scope_complete_bool": True, "summary_dict": {}}
    result_dict = build_financial_overview_dict(workspace_dict, source_obj, ForbiddenProvider(), as_of_ts=AS_OF_TS)
    assert result_dict["tile_list"][1]["value_str"] == "+$10.00"
    assert result_dict["tile_list"][2]["value_str"] == "—"
    assert result_dict["chart_dict"]["available_bool"] is False
    assert result_dict["chart_dict"]["segment_list"] == []


def test_cash_fallback_never_changes_flex_headline_or_invents_cash():
    workspace_dict, source_obj, provider_obj = demo_fixture_tuple()
    baseline_dict = build_financial_overview_dict(workspace_dict, source_obj, provider_obj, as_of_ts=AS_OF_TS)
    workspace_dict["summary_dict"]["pod_row_dict_list"][0]["eod_snapshot_dict"]["equity_float"] = None
    result_dict = build_financial_overview_dict(workspace_dict, source_obj, provider_obj, as_of_ts=AS_OF_TS)
    assert result_dict["tile_list"] == baseline_dict["tile_list"]
    allocation_dict = result_dict["allocation_dict"]
    assert allocation_dict["source_str"] == "flex_nav"
    assert allocation_dict["cash_complete_bool"] is False
    assert allocation_dict["cash_percent_str"] == "—"
    assert allocation_dict["total_cash_str"] == "—"
    assert allocation_dict["invested_heading_str"] == "Value"
    assert all(row_dict["cash_str"] == "—" for row_dict in allocation_dict["row_list"])
    assert len(allocation_dict["slice_list"]) == 2


def test_broker_and_flex_difference_uses_v3_whole_ring_basis():
    report_dict, source_obj, cash_list = allocation_fixture_tuple()
    report_dict["closing_nav_float"] = 200
    cash_list[0].update(equity_float=110, cash_float=11)
    cash_list[1].update(equity_float=95, cash_float=19)
    source_dict = portfolio_allocation_dict(report_dict, source_obj, cash_snapshot_list=cash_list)
    result_dict = _allocation_dict(source_dict)
    assert result_dict["source_str"] == "broker_eod"
    assert result_dict["total_value_str"] == "$205.00"
    assert result_dict["cash_percent_str"] == f"{30 / 205:.1%}"
    assert result_dict["row_list"][0]["invested_str"] == f"{99 / 205:.1%}"
    assert result_dict["row_list"][1]["cash_str"] == f"{19 / 205:.1%}"
    assert len(result_dict["slice_list"]) == 5
    assert report_dict["closing_nav_float"] == 200


@pytest.mark.parametrize("cash_float", [0, 100, None, -1, float("nan"), float("inf")])
def test_cash_zero_all_missing_and_invalid_never_generate_invalid_geometry(cash_float):
    report_dict, source_obj, cash_list = allocation_fixture_tuple()
    for cash_dict in cash_list:
        cash_dict["cash_float"] = cash_float
    result_dict = _allocation_dict(portfolio_allocation_dict(report_dict, source_obj, cash_snapshot_list=cash_list))
    assert all("nan" not in slice_dict["path_str"] and "inf" not in slice_dict["path_str"] for slice_dict in result_dict["slice_list"])
    if cash_float == 0:
        assert result_dict["cash_percent_str"] == "0.0%"
        assert len(result_dict["slice_list"]) == 2
    elif cash_float == 100:
        assert result_dict["cash_percent_str"] == "100.0%"
        assert len(result_dict["slice_list"]) == 3
        assert result_dict["slice_list"][0]["path_str"].count(" A ") == 4
    else:
        assert result_dict["cash_percent_str"] == "—"
        assert result_dict["total_cash_str"] == "—"


def test_chart_missing_flat_single_and_zero_values_are_finite_and_visible():
    assert _chart_dict([])["available_bool"] is False
    for value_float in (0, -.1, .1, .0001):
        result_dict = _chart_dict([{"market_date_str": "2026-09-04", "cumulative_return_float": value_float}])
        assert result_dict["available_bool"] is True
        assert result_dict["isolated_point_list"]
        assert len({tick_dict["label_str"] for tick_dict in result_dict["y_tick_list"]}) == 3
        assert "nan" not in str(result_dict) and "inf" not in str(result_dict)


def test_allocation_matches_d_cash_label_order_without_changing_pod_order():
    source_dict = {"total_float": 400, "cash_complete_bool": True, "cash_weight_float": .249,
        "source_str": "broker_eod", "item_list": [{"label_str": name_str, "value_float": 100,
            "weight_float": .25, "cash_weight_float": share_float, "invested_weight_float": .25 - share_float}
            for name_str, share_float in (("DVO2", .028), ("QPI", .021), ("NDX Momentum", .12), ("TAA BTAL", .08))]}
    before_dict = deepcopy(source_dict)
    result_dict = _allocation_dict(source_dict)
    assert result_dict["cash_percent_str"] == "24.9%"
    assert result_dict["slice_list"][0]["color_str"] == "#dfe3e9"
    assert [row_dict["name_str"] for row_dict in result_dict["row_list"]] == ["DVO2", "QPI", "NDX Momentum", "TAA BTAL"]
    cash_slice_list = [slice_dict for slice_dict in result_dict["slice_list"] if " cash " in slice_dict["label_str"]]
    assert [slice_dict["label_str"] for slice_dict in cash_slice_list] == [
        "NDX Momentum cash 12.0%", "TAA BTAL cash 8.0%", "DVO2 cash 2.8%", "QPI cash 2.1%"]
    assert cash_slice_list[0]["color_str"] == "#4a3aa7"
    assert [label_dict["label_str"] for label_dict in result_dict["label_list"]] == ["NDX 12%", "TAA 8%"]
    assert result_dict["label_list"][0]["x_float"] == pytest.approx(157.6, abs=.1)
    assert result_dict["label_list"][0]["y_float"] == pytest.approx(23.4, abs=.1)
    assert result_dict["ring_caption_str"] == "Outer ring = who holds the cash"
    assert source_dict == before_dict


def test_unknown_period_and_naive_clock_are_rejected():
    workspace_dict, source_obj, provider_obj = demo_fixture_tuple()
    with pytest.raises(ValueError):
        build_financial_overview_dict(workspace_dict, source_obj, provider_obj, period_str="PAPER", as_of_ts=AS_OF_TS)
    with pytest.raises(ValueError):
        build_financial_overview_dict(workspace_dict, source_obj, provider_obj, as_of_ts=AS_OF_TS.replace(tzinfo=None))
