"""Returned Performance facts retain the canonical reporting boundary."""

from copy import deepcopy
from dataclasses import replace
from datetime import UTC, datetime
from decimal import Decimal
import math
import statistics

import pytest

from alpha.live.client_reporting import BrokerReportingSnapshot, build_client_report_dict
from alpha.live.dashboard_v3.demo import build_demo_fixture_tuple
from alpha.live.dashboard_v4 import performance
from alpha.live.dashboard_v4.performance import build_performance_page_dict
from test_client_reporting import client_config_dict, nav_attributes_dict, snapshot_obj


AS_OF_TS = datetime(2026, 9, 8, 13, 41, tzinfo=UTC)


def _demo_tuple():
    registry_dict, snapshot_dict = build_demo_fixture_tuple()
    client_dict = registry_dict["clients"][1]
    return {"client_dict": client_dict, "financial_scope_complete_bool": True,
        "valuation_account_list": client_dict["accounts"]}, snapshot_dict[client_dict["client_id"]]


def _workspace_dict(client_dict, *, complete_bool=True):
    return {"client_dict": client_dict, "financial_scope_complete_bool": complete_bool,
        "valuation_account_list": client_dict["accounts"]}


def _configured_dict(*, second_bool=False):
    client_dict = client_config_dict(second_bool=second_bool)
    client_dict["client_twr"] = {"method": "daily_nav_eod_v1", "reviewed_by": "Synthetic fixture",
        "evidence_ref": "tests/test_dashboard_v4_performance.py"}
    return client_dict


def _view(workspace_dict, source_obj, **options_dict):
    return build_performance_page_dict(workspace_dict, source_obj, as_of_ts=options_dict.pop("as_of_ts", AS_OF_TS), **options_dict)


def test_default_all_uses_one_canonical_report_and_does_not_mutate(monkeypatch):
    workspace_dict, source_obj = _demo_tuple()
    before_dict, call_list = deepcopy(workspace_dict), []
    original_fn = performance.build_client_report_dict

    def recording_fn(*args_tuple, **kwargs_dict):
        call_list.append(kwargs_dict)
        return original_fn(*args_tuple, **kwargs_dict)

    monkeypatch.setattr(performance, "build_client_report_dict", recording_fn)
    result_dict = _view(workspace_dict, source_obj)
    assert len(call_list) == 1
    assert result_dict["period_str"] == "All"
    assert (result_dict["from_date_str"], result_dict["to_date_str"]) == ("2026-06-01", "2026-09-04")
    expected_dict = original_fn(workspace_dict["client_dict"], source_obj, from_date_str="2026-06-01", to_date_str="2026-09-04", as_of_ts=AS_OF_TS)
    assert result_dict["report_dict"] == expected_dict
    assert result_dict["tile_list"][3]["value_str"] == f"{expected_dict['twr_float']:+.2%}"
    assert result_dict["chart_dict"]["point_list"][-1]["value_float"] == expected_dict["twr_float"]
    assert result_dict["contribution_dict"]["available_bool"]
    assert sum(row_dict["value_float"] for row_dict in result_dict["contribution_dict"]["row_list"]) == pytest.approx(expected_dict["pnl_float"])
    assert workspace_dict == before_dict


@pytest.mark.parametrize("period_str,from_str", [("1W", "2026-09-01"), ("MTD", "2026-09-01"), ("YTD", "2026-06-01"), ("All", "2026-06-01")])
def test_presets_keep_current_calendar_start(period_str, from_str):
    workspace_dict, source_obj = _demo_tuple()
    result_dict = _view(workspace_dict, source_obj, period_str=period_str)
    assert result_dict["from_date_str"] == from_str
    assert result_dict["to_date_str"] == "2026-09-04"


@pytest.mark.parametrize("period_str,as_of_ts,from_str", [
    ("MTD", datetime(2026, 10, 10, 13, tzinfo=UTC), "2026-10-01"),
    ("YTD", datetime(2027, 1, 10, 13, tzinfo=UTC), "2027-01-01")])
def test_delayed_import_does_not_relabel_old_month_or_year(period_str, as_of_ts, from_str):
    workspace_dict, source_obj = _demo_tuple()
    result_dict = _view(workspace_dict, source_obj, period_str=period_str, as_of_ts=as_of_ts)
    assert result_dict["from_date_str"] == from_str
    assert result_dict["delayed_bool"]
    assert result_dict["report_dict"]["twr_float"] is None
    assert result_dict["chart_dict"] is None
    assert not result_dict["monthly_row_list"]


def test_custom_dates_are_exact_and_today_is_unfinalized():
    workspace_dict, source_obj = _demo_tuple()
    result_dict = _view(workspace_dict, source_obj, from_date_str="2026-09-01", to_date_str="2026-09-08")
    assert result_dict["to_date_str"] == "2026-09-08"
    assert result_dict["latest_close_str"] == "2026-09-04"
    assert result_dict["report_dict"]["pnl_float"] is None
    assert result_dict["report_dict"]["twr_float"] is None


@pytest.mark.parametrize("options_dict", [{"period_str": "3M"}, {"level_str": "paper"}, {"unit_str": "shares"},
    {"from_date_str": "2026-09-01"}, {"from_date_str": "2026-09-04", "to_date_str": "2026-09-01"},
    {"from_date_str": "2026-09-01", "to_date_str": "2027-09-01"}, {"as_of_ts": datetime(2026, 9, 8)}])
def test_invalid_inputs_are_rejected(options_dict):
    workspace_dict, source_obj = _demo_tuple()
    with pytest.raises(ValueError):
        _view(workspace_dict, source_obj, **options_dict)


@pytest.mark.parametrize("failure_str", ["workspace", "snapshot", "empty"])
def test_financial_failure_has_full_safe_empty_shape(failure_str):
    workspace_dict, source_obj = _demo_tuple()
    if failure_str == "workspace":
        workspace_dict["financial_error_str"] = "Private C:/secret/source.xml"
    elif failure_str == "snapshot":
        source_obj = replace(source_obj, unavailable_reason_str="Private XML")
    else:
        source_obj = BrokerReportingSnapshot()
    result_dict = _view(workspace_dict, source_obj)
    assert result_dict["error_str"]
    assert result_dict["report_dict"] is None
    assert len(result_dict["tile_list"]) == 6
    assert all(tile_dict["value_str"] == "—" for tile_dict in result_dict["tile_list"])
    assert not result_dict["pod_chart_dict"]["available_bool"]
    assert not result_dict["contribution_dict"]["available_bool"]
    assert "Private" not in str(result_dict)


def test_unknown_history_retains_nav_without_portfolio_return_or_contribution():
    workspace_dict, source_obj = _demo_tuple()
    workspace_dict["financial_scope_complete_bool"] = False
    result_dict = _view(workspace_dict, source_obj, unit_str="usd")
    assert result_dict["tile_list"][0]["value_str"] != "—"
    assert result_dict["tile_list"][1]["value_str"] != "—"
    assert result_dict["chart_dict"] is not None
    assert result_dict["tile_list"][2]["value_str"] == "—"
    assert result_dict["tile_list"][3]["value_str"] == "—"
    assert not result_dict["contribution_dict"]["available_bool"]


def test_gap_withholds_path_and_risk_but_keeps_independent_nav_and_other_pod():
    workspace_dict, source_obj = _demo_tuple()
    route_str = workspace_dict["client_dict"]["accounts"][0]["account_route"]
    source_obj = replace(source_obj, row_tuple=tuple(row_obj for row_obj in source_obj.row_tuple
        if (row_obj.account_route_str, row_obj.market_date_str) != (route_str, "2026-07-02")))
    result_dict = _view(workspace_dict, source_obj, level_str="pods")
    assert result_dict["tile_list"][1]["value_str"] != "—"
    assert result_dict["chart_dict"] is None
    assert not result_dict["risk_row_list"]
    assert result_dict["pod_row_list"][0]["return_str"] == "—"
    assert len(result_dict["pod_chart_dict"]["series_list"]) == 3
    assert not result_dict["contribution_dict"]["available_bool"]
    assert all(row_dict["pod_id_str"] != result_dict["pod_row_list"][0]["pod_id_str"]
        for row_dict in result_dict["monthly_row_list"])


def test_no_complete_portfolio_close_keeps_other_pod_official_history():
    workspace_dict, source_obj = _demo_tuple()
    missing_route_str = workspace_dict["client_dict"]["accounts"][0]["account_route"]
    source_obj = replace(source_obj, row_tuple=tuple(row_obj for row_obj in source_obj.row_tuple if row_obj.account_route_str != missing_route_str))
    result_dict = _view(workspace_dict, source_obj, level_str="pods")
    assert result_dict["latest_close_str"] == ""
    assert result_dict["report_dict"] is not None
    assert result_dict["tile_list"][1]["value_str"] == "—"
    assert result_dict["tile_list"][3]["value_str"] == "—"
    assert result_dict["pod_row_list"][0]["return_str"] == "—"
    assert len(result_dict["pod_chart_dict"]["series_list"]) == 3
    assert not result_dict["contribution_dict"]["available_bool"]


def test_first_day_loss_uses_opening_baseline_and_legacy_account_twr():
    source_obj = snapshot_obj([nav_attributes_dict(closing_str="900", twr_str="-10", mtm="-100")])
    result_dict = _view(_workspace_dict(client_config_dict()), source_obj)
    assert result_dict["tile_list"][4]["value_str"] == "-10.00%"
    assert result_dict["tile_list"][5]["value_str"] == "-10.00%"
    assert result_dict["tile_list"][5]["detail_str"] == "1 session · selected period"
    assert result_dict["chart_dict"]["point_list"][0]["value_float"] == 0
    assert result_dict["risk_row_list"][0]["value_str"] == "Needs 20 sessions"
    assert result_dict["monthly_row_list"][0]["cell_list"][8]["value_str"] == "-10.00"


def test_deposit_changes_nav_not_return_drawdown_or_profit_contribution():
    client_dict = _configured_dict()
    source_obj = snapshot_obj([nav_attributes_dict(opening_str="1000", closing_str="2010", depositsWithdrawals="1000")])
    result_dict = _view(_workspace_dict(client_dict), source_obj)
    assert result_dict["tile_list"][1]["value_str"] == "$2,010.00"
    assert result_dict["tile_list"][2]["value_str"] == "+$10.00"
    assert result_dict["tile_list"][3]["value_str"] == "+1.00%"
    assert result_dict["tile_list"][4]["value_str"] == "0.00%"
    assert result_dict["contribution_dict"]["total_str"] == "+$10.00"
    assert next(row_dict["value_str"] for row_dict in result_dict["bridge_row_list"] if row_dict["label_str"] == "Transfers in / out") == "+$1,000.00"


def test_midperiod_entry_and_retirement_keep_scope_capital_and_actual_date_axis():
    client_dict = _configured_dict(second_bool=True)
    client_dict["accounts"][0]["effective_to"] = "2026-09-02"
    client_dict["accounts"][1]["effective_from"] = "2026-09-02"
    source_obj = snapshot_obj([
        nav_attributes_dict(date_str="2026-09-01"),
        nav_attributes_dict(date_str="2026-09-02", opening_str="1010", closing_str="1020"),
        nav_attributes_dict(account_str="U_TEST_B", date_str="2026-09-02", opening_str="2000", closing_str="2010", twr_str=".5"),
        nav_attributes_dict(account_str="U_TEST_B", date_str="2026-09-03", opening_str="2010", closing_str="2020", twr_str=".4975124378")])
    result_dict = _view(_workspace_dict(client_dict), source_obj, level_str="pods")
    assert result_dict["report_dict"]["scope_movement_float"] == 980
    assert result_dict["total_row_dict"]["start_str"] == "$1,000.00"
    assert result_dict["total_row_dict"]["end_str"] == "$2,020.00"
    assert result_dict["contribution_dict"]["total_str"] == "+$40.00"
    assert len(result_dict["pod_row_list"]) == 2
    first_dict, second_dict = result_dict["pod_chart_dict"]["series_list"]
    assert first_dict["point_list"][0]["value_float"] == second_dict["point_list"][0]["value_float"] == 100
    assert first_dict["point_list"][0]["x_float"] < second_dict["point_list"][0]["x_float"]
    first_eod_dict = first_dict["point_list"][-1]
    second_eod_dict = second_dict["point_list"][1]
    assert first_eod_dict["market_date_str"] == second_eod_dict["market_date_str"] == "2026-09-02"
    assert first_eod_dict["x_float"] == second_eod_dict["x_float"]
    assert first_dict["point_list"][-1]["x_float"] < second_dict["point_list"][-1]["x_float"]


def test_legacy_multiaccount_never_invents_portfolio_twr_from_account_returns():
    client_dict = client_config_dict(second_bool=True)
    source_obj = snapshot_obj([nav_attributes_dict(account_str=route_str) for route_str in ("U_TEST_A", "U_TEST_B")])
    result_dict = _view(_workspace_dict(client_dict), source_obj, level_str="pods")
    assert result_dict["report_dict"]["pnl_float"] == 20
    assert result_dict["report_dict"]["twr_float"] is None
    assert result_dict["chart_dict"] is None
    assert len(result_dict["pod_chart_dict"]["series_list"]) == 2
    assert "Portfolio TWR unavailable" in result_dict["source_str"]


def test_missing_bridge_does_not_erase_official_account_return_or_invent_pnl():
    client_dict = client_config_dict(bridge_bool=False)
    result_dict = _view(_workspace_dict(client_dict), snapshot_obj([nav_attributes_dict()]))
    assert result_dict["tile_list"][3]["value_str"] == "+1.00%"
    assert result_dict["chart_dict"] is not None
    assert result_dict["pod_row_list"][0]["return_str"] == "+1.00%"
    assert result_dict["tile_list"][2]["value_str"] == "—"
    assert result_dict["daily_chart_dict"] is None
    assert not result_dict["contribution_dict"]["available_bool"]
    assert not any(row_dict["label_str"] in {"Best day", "Worst day"} for row_dict in result_dict["risk_row_list"])


def test_unmatched_internal_transfer_blocks_configured_twr_not_valid_dollars():
    client_dict = _configured_dict()
    source_obj = snapshot_obj([nav_attributes_dict(opening_str="1000", closing_str="1110", internalCashTransfers="100")])
    result_dict = _view(_workspace_dict(client_dict), source_obj)
    assert result_dict["tile_list"][3]["value_str"] == "—"
    assert result_dict["tile_list"][2]["value_str"] == "+$10.00"
    assert result_dict["contribution_dict"]["total_str"] == "+$10.00"
    assert result_dict["pod_row_list"][0]["return_str"] == "+1.00%"
    assert not result_dict["risk_row_list"]
    assert "transfer" in result_dict["verdict_detail_str"].lower()


def test_unfinalized_latest_row_does_not_replace_previous_close():
    client_dict = _configured_dict()
    source_obj = snapshot_obj([nav_attributes_dict(date_str="2026-09-04"),
        nav_attributes_dict(date_str="2026-09-08", opening_str="1010", closing_str="1020")])
    client_dict["mandate_start_date"] = client_dict["accounts"][0]["effective_from"] = "2026-09-04"
    result_dict = _view(_workspace_dict(client_dict), source_obj)
    assert result_dict["latest_close_str"] == result_dict["to_date_str"] == "2026-09-04"
    assert result_dict["tile_list"][1]["value_str"] == "$1,010.00"
    assert result_dict["report_dict"]["twr_float"] == pytest.approx(.01)


def test_complete_calendar_month_with_weekend_end_is_not_marked_partial():
    workspace_dict, source_obj = _demo_tuple()
    result_dict = _view(workspace_dict, source_obj, level_str="pods", from_date_str="2026-08-01", to_date_str="2026-08-31")
    assert len(result_dict["monthly_row_list"]) == 4
    for month_row_dict in result_dict["monthly_row_list"]:
        assert month_row_dict["cell_list"][7]["partial_bool"] is False
        assert month_row_dict["partial_bool"] is True


def test_negative_pod_contributions_remain_signed_and_equal_portfolio_profit():
    client_dict = _configured_dict(second_bool=True)
    source_obj = snapshot_obj([nav_attributes_dict(closing_str="990", mtm="-10", twr_str="-1"),
        nav_attributes_dict(account_str="U_TEST_B", closing_str="1005", mtm="5", twr_str=".5")])
    result_dict = _view(_workspace_dict(client_dict), source_obj)
    contribution_dict = result_dict["contribution_dict"]
    assert contribution_dict["available_bool"]
    assert contribution_dict["total_str"] == "−$5.00"
    assert [row_dict["value_float"] for row_dict in contribution_dict["row_list"]] == [5., -10.]
    assert [row_dict["bar_percent_float"] for row_dict in contribution_dict["row_list"]] == [50., 100.]
    zero_float = contribution_dict["zero_percent_float"]
    gain_dict, loss_dict = contribution_dict["row_list"]
    assert gain_dict["bar_left_percent_float"] == pytest.approx(zero_float)
    assert gain_dict["bar_left_percent_float"] + gain_dict["bar_width_percent_float"] == pytest.approx(100.)
    assert loss_dict["bar_left_percent_float"] == 0
    assert loss_dict["bar_left_percent_float"] + loss_dict["bar_width_percent_float"] == pytest.approx(zero_float)
    assert loss_dict["bar_width_percent_float"] == pytest.approx(2 * gain_dict["bar_width_percent_float"])


def test_months_compound_across_years_and_mark_selected_partial_months():
    daily_list = [{"market_date_str": date_str, "return_float": return_float} for date_str, return_float in
        (("2025-12-31", .1), ("2026-01-02", .1), ("2026-01-05", -.1))]
    row_list = performance._monthly_list(daily_list, name_str="Portfolio", pod_id_str="", from_str="2025-12-31", to_str="2026-01-05")
    assert [row_dict["year_int"] for row_dict in row_list] == [2025, 2026]
    assert row_list[1]["cell_list"][0]["return_float"] == pytest.approx(-.01)
    assert row_list[1]["cell_list"][0]["partial_bool"]
    assert row_list[1]["cell_list"][1]["return_float"] is None
    assert row_list[1]["cell_list"][1]["value_str"] == "·"
    assert row_list[1]["cell_list"][1]["heat_class_str"] == ""
    assert row_list[1]["year_str"] == "-1.00%"


@pytest.mark.parametrize("return_float,heat_str", [
    (None, ""), (0., ""), (.00001, "heat-pos-1"), (.0099999, "heat-pos-1"),
    (.01, "heat-pos-2"), (.0199999, "heat-pos-2"), (.02, "heat-pos-3"),
    (-.00001, "heat-neg-1"), (-.0099999, "heat-neg-1"), (-.01, "heat-neg-2"), (-.2, "heat-neg-2"),
])
def test_monthly_heat_uses_mockup_magnitude_boundaries(return_float, heat_str):
    assert performance._heat_class_str(return_float) == heat_str
    if return_float is not None:
        row_list = performance._monthly_list([{"market_date_str": "2026-09-01", "return_float": return_float}],
            name_str="Test", pod_id_str="test", from_str="2026-09-01", to_str="2026-09-30")
        assert row_list[0]["cell_list"][8]["heat_class_str"] == heat_str
        assert row_list[0]["heat_class_str"] == heat_str


def test_pod_verdict_counts_verified_returns_and_verified_dollar_leader():
    workspace_dict, source_obj = _demo_tuple()
    result_dict = _view(workspace_dict, source_obj, level_str="pods")
    assert result_dict["verdict_str"] == "All 4 pods are up in this period."
    leader_dict = result_dict["contribution_dict"]["row_list"][0]
    assert result_dict["verdict_detail_str"] == leader_dict["name_str"] + " adds the most: " + leader_dict["value_str"] + "."
    assert result_dict["has_multiple_years_bool"] is False


def test_pod_verdict_does_not_promote_surviving_returns_after_missing_history():
    workspace_dict, source_obj = _demo_tuple()
    missing_route_str = workspace_dict["client_dict"]["accounts"][0]["account_route"]
    source_obj = replace(source_obj, row_tuple=tuple(row_obj for row_obj in source_obj.row_tuple
        if (row_obj.account_route_str, row_obj.market_date_str) != (missing_route_str, "2026-07-02")))
    result_dict = _view(workspace_dict, source_obj, level_str="pods")
    assert result_dict["verdict_str"] == "Pod returns incomplete."
    assert result_dict["verdict_detail_str"] == "3 of 4 returns verified."
    assert not result_dict["contribution_dict"]["available_bool"]


def test_pod_return_verdict_never_invents_a_dollar_leader_without_flow_coverage():
    client_dict = _configured_dict()
    attributes_dict = nav_attributes_dict()
    attributes_dict.pop("depositsWithdrawals")
    result_dict = _view(_workspace_dict(client_dict), snapshot_obj([attributes_dict]), level_str="pods")
    assert result_dict["pod_row_list"][0]["return_float"] == .01
    assert result_dict["verdict_str"] == "1 pod is up in this period."
    assert result_dict["verdict_detail_str"] == ""
    assert not result_dict["contribution_dict"]["available_bool"]


def test_equal_positive_contributions_do_not_select_arbitrary_pod_leader():
    client_dict = _configured_dict(second_bool=True)
    source_obj = snapshot_obj([nav_attributes_dict(account_str=route_str) for route_str in ("U_TEST_A", "U_TEST_B")])
    result_dict = _view(_workspace_dict(client_dict), source_obj, level_str="pods")
    assert result_dict["verdict_str"] == "All 2 pods are up in this period."
    assert result_dict["verdict_detail_str"] == "2 pods share the lead: +$10.00 each."


def test_multiple_years_are_identified_without_extending_pod_history():
    client_dict = _configured_dict()
    client_dict["mandate_start_date"] = client_dict["accounts"][0]["effective_from"] = "2025-12-31"
    source_obj = snapshot_obj([nav_attributes_dict(date_str="2025-12-31"),
        nav_attributes_dict(date_str="2026-01-02", opening_str="1010", closing_str="1020", twr_str=".9900990099")])
    result_dict = _view(_workspace_dict(client_dict), source_obj, level_str="pods")
    assert result_dict["has_multiple_years_bool"] is True
    assert [row_dict["year_int"] for row_dict in result_dict["monthly_row_list"]] == [2025, 2026]
    assert result_dict["monthly_row_list"][0]["cell_list"][0]["value_str"] == "·"


def test_twenty_session_risk_uses_sample_stdev_and_does_not_drop_first_return():
    workspace_dict, source_obj = _demo_tuple()
    result_dict = _view(workspace_dict, source_obj)
    report_dict = result_dict["report_dict"]
    calendar_obj = performance.get_exchange_calendar_obj("XNYS")
    session_set = {session_obj.date().isoformat() for session_obj in calendar_obj.sessions_in_range("2026-06-01", "2026-09-04")}
    return_list = [row_dict["return_float"] for row_dict in report_dict["twr_daily_list"] if row_dict["market_date_str"] in session_set]
    expected_str = f"{statistics.stdev(return_list[-20:]) * math.sqrt(252):.1%} a year"
    assert result_dict["risk_row_list"][0]["value_str"] == expected_str
    assert sum(cell_dict["return_float"] is not None for cell_dict in result_dict["monthly_row_list"][0]["cell_list"]) == 4


def test_unit_and_level_change_display_only_not_report_or_totals():
    workspace_dict, source_obj = _demo_tuple()
    base_dict = _view(workspace_dict, source_obj)
    result_dict = _view(workspace_dict, source_obj, unit_str="usd", level_str="pods")
    assert result_dict["report_dict"] == base_dict["report_dict"]
    assert result_dict["tile_list"] == base_dict["tile_list"]
    assert result_dict["total_row_dict"] == base_dict["total_row_dict"]
    assert result_dict["chart_dict"]["point_list"][-1]["value_float"] == base_dict["report_dict"]["closing_nav_float"]
    assert len(result_dict["monthly_row_list"]) == 4


def test_historical_drawdown_is_labeled_at_selected_period_end():
    workspace_dict, source_obj = _demo_tuple()
    result_dict = _view(workspace_dict, source_obj, from_date_str="2026-07-01", to_date_str="2026-07-31")
    assert result_dict["tile_list"][5]["label_str"] == "End below high"
    assert "period" in result_dict["tile_list"][5]["detail_str"]
    assert result_dict["to_date_str"] == "2026-07-31"
