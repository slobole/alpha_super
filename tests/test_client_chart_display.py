"""Axes and daily history are projections of existing financial facts, not math shortcuts."""
from copy import deepcopy
from datetime import UTC, datetime

import pytest

from alpha.live.client_reporting import build_client_report_dict
from alpha.live.dashboard_v3.client_charts import daily_history_list, nav_chart_dict
from test_client_reporting import client_config_dict, nav_attributes_dict, snapshot_obj
from test_client_twr import twr_config_dict


def test_axes_map_high_zero_low_and_include_percentage_units():
    chart_dict = nav_chart_dict([{"nav_float": value_float} for value_float in (-.1, 0, .1)], unit_str="pct")
    assert [tick_dict["label_str"] for tick_dict in chart_dict["tick_list"]] == ['10.00%', '0.00%', '-10.00%']
    assert [tick_dict["y_float"] for tick_dict in chart_dict["tick_list"]] == [10, 90, 170]
    assert [point_dict["y_float"] for point_dict in chart_dict["point_list"]] == [170, 90, 10]


@pytest.mark.parametrize("value_list", [[100], [100, 100], [-50, -10], [1e12, 1e12 + 100], [0, 1e-7]])
def test_flat_single_negative_large_and_tiny_charts_have_distinct_finite_ticks(value_list):
    chart_dict = nav_chart_dict([{"nav_float": value_float} for value_float in value_list])
    assert len({tick_dict["label_str"] for tick_dict in chart_dict["tick_list"]}) == 3
    assert all('$' in tick_dict["label_str"] for tick_dict in chart_dict["tick_list"])
    assert all(10 <= point_dict["y_float"] <= 170 for point_dict in chart_dict["point_list"])


def test_daily_bars_have_real_zero_line_and_missing_days_are_not_zero():
    row_list = [{"market_date_str": f"2026-09-0{index_int + 1}", "pnl_float": value_float}
        for index_int, value_float in enumerate((10, -5, 0, None))]
    chart_dict = nav_chart_dict(row_list, value_field_str="pnl_float", bars_bool=True)
    assert chart_dict["zero_y_float"] == 90
    positive_dict, negative_dict, zero_dict, missing_dict = chart_dict["bar_list"]
    assert positive_dict["top_float"] == 10 and positive_dict["height_float"] == 80
    assert negative_dict["top_float"] == 90 and negative_dict["height_float"] == 40
    assert zero_dict["sign_str"] == 'zero' and zero_dict["height_float"] == 0
    assert missing_dict["missing_bool"] and "value_float" not in missing_dict
    assert row_list[-1]["pnl_float"] is None


def test_zero_only_daily_history_stays_visible():
    chart_dict = nav_chart_dict([{"return_float": 0}], value_field_str="return_float", unit_str="pct", bars_bool=True)
    assert chart_dict is not None and chart_dict["bar_list"][0]["sign_str"] == "zero"


def test_isolated_values_around_gap_have_visible_markers_not_invisible_single_point_lines():
    chart_dict = nav_chart_dict([{"nav_float": value_float} for value_float in (100, None, 120)])
    assert len(chart_dict["segment_list"]) == 2
    assert all(point_dict["isolated_bool"] for point_dict in chart_dict["point_list"])
    mixed_dict = nav_chart_dict([{"nav_float": value_float} for value_float in (100, 110, None, 120)])
    assert [point_dict["isolated_bool"] for point_dict in mixed_dict["point_list"]] == [False, False, True]


def build_report_dict(attribute_list, config_dict, *, to_str="2026-09-01", as_of_ts=None):
    return build_client_report_dict(config_dict, snapshot_obj(attribute_list), from_date_str="2026-09-01",
        to_date_str=to_str, as_of_ts=as_of_ts or datetime(2026, 9, 8, tzinfo=UTC))


def test_daily_first_return_is_official_not_cumulative_difference_or_nav_delta():
    report_dict = build_report_dict([
        nav_attributes_dict(opening_str="100", closing_str="110", twr_str="10"),
        nav_attributes_dict(date_str="2026-09-02", opening_str="110", closing_str="99", twr_str="-10", mtm="-11")
    ], twr_config_dict(False), to_str="2026-09-02")
    scope_list = daily_history_list(report_dict)
    assert report_dict["twr_float"] == pytest.approx(-.01)
    for scope_dict in scope_list:
        assert [row_dict["return_float"] for row_dict in scope_dict["daily_list"]] == [.1, -.1]
        assert [row_dict["pnl_float"] for row_dict in scope_dict["daily_list"]] == [10, -11]
        assert [row_dict["market_date_str"] for row_dict in scope_dict["daily_list"]] == ["2026-09-01", "2026-09-02"]


def test_flow_sensitive_account_and_portfolio_returns_remain_distinct():
    report_dict = build_report_dict([nav_attributes_dict(opening_str="100", closing_str="210", depositsWithdrawals="100", twr_str="5")], twr_config_dict(False))
    original_dict = deepcopy(report_dict)
    portfolio_dict, strategy_dict = daily_history_list(report_dict)
    assert portfolio_dict["daily_list"][0]["return_float"] == .1
    assert strategy_dict["daily_list"][0]["return_float"] == .05
    assert portfolio_dict["daily_list"][0]["pnl_float"] == strategy_dict["daily_list"][0]["pnl_float"] == 10
    assert report_dict == original_dict  # No chart or toggle mutates accounting/hash.


def test_client_daily_return_is_not_equal_weight_average():
    report_dict = build_report_dict([nav_attributes_dict(), nav_attributes_dict("U_TEST_B", opening_str="10000", closing_str="10050", twr_str=".5", mtm="50")], twr_config_dict())
    portfolio_dict = daily_history_list(report_dict)[0]
    assert portfolio_dict["daily_list"][0]["return_float"] == pytest.approx(60 / 11000)
    assert portfolio_dict["daily_list"][0]["return_float"] != .0075


def test_missing_capital_keeps_official_daily_return_but_never_invents_profit():
    report_dict = build_report_dict([nav_attributes_dict()], client_config_dict(bridge_bool=False))
    portfolio_dict, strategy_dict = daily_history_list(report_dict)
    assert strategy_dict["daily_list"][0] == {"market_date_str": "2026-09-01", "return_float": .01, "pnl_float": None}
    assert portfolio_dict["daily_list"][0]["return_float"] == .01  # Existing one-account official fallback.
    assert strategy_dict["pnl_chart_dict"] is None and portfolio_dict["pnl_chart_dict"] is None


def test_incomplete_period_does_not_unlock_daily_returns_or_profit():
    report_dict = build_report_dict([nav_attributes_dict()], twr_config_dict(False), to_str="2026-09-02")
    for scope_dict in daily_history_list(report_dict):
        assert len(scope_dict["daily_list"]) == 2
        assert all(row_dict["return_float"] is None and row_dict["pnl_float"] is None for row_dict in scope_dict["daily_list"])
    assert report_dict["status_str"] == "draft"


def test_today_does_not_enter_daily_chart_as_finalized_fact():
    report_dict = build_report_dict([nav_attributes_dict()], twr_config_dict(False), as_of_ts=datetime(2026, 9, 1, 23, tzinfo=UTC))
    assert all(scope_dict["return_chart_dict"] is None and scope_dict["pnl_chart_dict"] is None for scope_dict in daily_history_list(report_dict))
