"""Capital movements must never become performance in V4 charts."""

from datetime import UTC, datetime

import pytest

from alpha.live.dashboard_v4.finance import build_financial_overview_dict, _chart_dict
from alpha.live.dashboard_v4.pod_finance import build_pod_finance_dict
from test_client_reporting import nav_attributes_dict, snapshot_obj
from test_client_twr import twr_config_dict
from test_dashboard_operator_access import ForbiddenProvider


AS_OF_TS = datetime(2026, 9, 8, 23, tzinfo=UTC)


def _workspace_dict(client_dict):
    return {"client_dict": client_dict, "financial_scope_complete_bool": True,
        "operations_account_list": client_dict["accounts"], "valuation_account_list": client_dict["accounts"],
        "summary_dict": {"as_of_timestamp_str": AS_OF_TS.isoformat(), "pod_row_dict_list": [
            {"pod_id_str": account_dict["pod_id"], "account_route_str": account_dict["account_route"], "mode_str": "live"}
            for account_dict in client_dict["accounts"]]}}


def _overview_dict(row_list, client_dict):
    return build_financial_overview_dict(_workspace_dict(client_dict), snapshot_obj(row_list),
        ForbiddenProvider(), period_str="All", as_of_ts=AS_OF_TS)


@pytest.mark.parametrize("flow_str,close_str", [("100", "210"), ("-50", "60")])
def test_cash_flows_keep_portfolio_and_official_pod_returns_distinct(flow_str, close_str):
    client_dict = twr_config_dict(False)
    row_list = [nav_attributes_dict(opening_str="100", closing_str=close_str,
        depositsWithdrawals=flow_str, twr_str="5")]
    overview_dict = _overview_dict(row_list, client_dict)
    pod_dict = build_pod_finance_dict(_workspace_dict(client_dict), snapshot_obj(row_list), ForbiddenProvider(),
        pod_id_str="strategy_a", period_str="All", as_of_ts=AS_OF_TS)
    assert overview_dict["chart_dict"]["end_label_str"] == "+10.00%"
    assert pod_dict["chart_dict"]["end_label_str"] == "+5.00%"
    for view_dict in (overview_dict, pod_dict):
        chart_dict = view_dict["chart_dict"]
        assert view_dict["tile_list"][0]["value_str"] == f"${float(close_str):,.2f}"
        assert chart_dict["point_str"].startswith(f'40.00,{chart_dict["zero_y_float"]:.2f}')
        assert all(tick_dict["label_str"].endswith("%") for tick_dict in chart_dict["y_tick_list"])
        assert chart_dict["x_tick_list"][0]["label_str"] == "Sep 01"


@pytest.mark.parametrize("membership_str", ["join", "exit"])
def test_pod_membership_capital_does_not_reset_or_jump_the_return(membership_str):
    client_dict = twr_config_dict()
    row_list = [nav_attributes_dict(), nav_attributes_dict(date_str="2026-09-02", opening_str="1010", closing_str="1020")]
    if membership_str == "join":
        client_dict["accounts"][1]["effective_from"] = "2026-09-02"
        row_list.append(nav_attributes_dict("U_TEST_B", date_str="2026-09-02", opening_str="500", closing_str="550", mtm="50", twr_str="10"))
        expected_return_float, expected_nav_str = 1.01 * (1 + 60 / 1510) - 1, "$1,570.00"
    else:
        client_dict["accounts"][1]["effective_to"] = "2026-09-01"
        row_list.append(nav_attributes_dict("U_TEST_B", closing_str="1100", mtm="100", twr_str="10"))
        expected_return_float, expected_nav_str = 1.055 * (1 + 10 / 1010) - 1, "$1,020.00"
    view_dict = _overview_dict(row_list, client_dict)
    assert view_dict["chart_dict"]["end_label_str"] == f"{expected_return_float:+.2%}"
    assert len(view_dict["chart_dict"]["point_str"].split()) == 3  # Opening zero and both sessions.
    assert view_dict["chart_dict"]["x_tick_list"][0]["label_str"] == "Sep 01"
    assert view_dict["tile_list"][0]["value_str"] == expected_nav_str


def test_deposit_without_profit_keeps_both_return_charts_flat_at_zero():
    client_dict = twr_config_dict(False)
    row_list = [nav_attributes_dict(opening_str="100", closing_str="200", depositsWithdrawals="100", mtm="0", twr_str="0")]
    pod_dict = build_pod_finance_dict(_workspace_dict(client_dict), snapshot_obj(row_list), ForbiddenProvider(),
        pod_id_str="strategy_a", period_str="All", as_of_ts=AS_OF_TS)
    for view_dict in (_overview_dict(row_list, client_dict), pod_dict):
        chart_dict = view_dict["chart_dict"]
        assert chart_dict["available_bool"] is True
        assert chart_dict["end_label_str"] == "0.00%"
        assert len({point_str.split(",")[1] for point_str in chart_dict["point_str"].split()}) == 1
        assert view_dict["tile_list"][0]["value_str"] == "$200.00"


def test_matched_internal_transfer_keeps_the_portfolio_return_available():
    row_list = [
        nav_attributes_dict(opening_str="100", closing_str="60", internalCashTransfers="-50", twr_str="10"),
        nav_attributes_dict("U_TEST_B", opening_str="100", closing_str="150", internalCashTransfers="50", mtm="0", twr_str="0")]
    view_dict = _overview_dict(row_list, twr_config_dict())
    assert view_dict["chart_dict"]["end_label_str"] == "+5.00%"
    assert view_dict["tile_list"][0]["value_str"] == "$210.00"


@pytest.mark.parametrize("case_str", ["missing_field", "missing_day", "transfer"])
def test_invalid_portfolio_history_never_falls_back_to_a_nav_curve(case_str):
    client_dict = twr_config_dict(False)
    row_list = [nav_attributes_dict()]
    if case_str == "missing_field":
        del row_list[0]["billPay"]
    elif case_str == "missing_day":
        row_list.append(nav_attributes_dict(date_str="2026-09-03", opening_str="1020", closing_str="1030"))
    else:
        row_list[0].update(endingValue="960", internalCashTransfers="-50")
        row_list.append(nav_attributes_dict(date_str="2026-09-02", opening_str="960", closing_str="1020", internalCashTransfers="50"))
    view_dict = _overview_dict(row_list, client_dict)
    assert view_dict["tile_list"][0]["value_str"] != "—"
    assert view_dict["chart_dict"]["available_bool"] is False
    assert view_dict["chart_dict"]["segment_list"] == []
    assert view_dict["chart_dict"]["empty_str"] == "Return unavailable"
    if case_str == "missing_field":
        pod_dict = build_pod_finance_dict(_workspace_dict(client_dict), snapshot_obj(row_list), ForbiddenProvider(),
            pod_id_str="strategy_a", period_str="All", as_of_ts=AS_OF_TS)
        assert pod_dict["chart_dict"]["end_label_str"] == "+1.00%"  # Independent official account return.


@pytest.mark.parametrize("second_bool", [False, True])
def test_legacy_fallback_requires_one_account_covering_the_period(second_bool):
    client_dict = twr_config_dict(second_bool)
    del client_dict["client_twr"]
    row_list = [nav_attributes_dict()]
    if second_bool:
        row_list.append(nav_attributes_dict("U_TEST_B"))
    view_dict = _overview_dict(row_list, client_dict)
    assert view_dict["chart_dict"]["available_bool"] is not second_bool
    assert view_dict["chart_dict"]["end_label_str"] == ("—" if second_bool else "+1.00%")


def test_linked_losses_and_period_baseline_are_preserved():
    client_dict = twr_config_dict(False)
    row_list = [nav_attributes_dict(opening_str="100", closing_str="110", twr_str="10"),
        nav_attributes_dict(date_str="2026-09-02", opening_str="110", closing_str="99", twr_str="-10", mtm="-11")]
    full_dict = _overview_dict(row_list, client_dict)
    assert full_dict["chart_dict"]["end_label_str"] == "-1.00%"
    client_dict["mandate_start_date"] = client_dict["accounts"][0]["effective_from"] = "2026-09-02"
    day_dict = _overview_dict(row_list, client_dict)
    assert day_dict["chart_dict"]["end_label_str"] == "-10.00%"
    assert day_dict["chart_dict"]["zero_y_float"] == 18
    assert day_dict["chart_dict"]["area_str"].endswith(",18.00")  # Negative return shades from zero.


def test_missing_geometry_points_are_not_connected():
    chart_dict = _chart_dict([
        {"market_date_str": "2026-09-01 SOD", "cumulative_return_float": 0},
        {"market_date_str": "2026-09-01", "cumulative_return_float": None},
        {"market_date_str": "2026-09-02", "cumulative_return_float": .01}])
    assert len(chart_dict["segment_list"]) == 2
    assert len(chart_dict["isolated_point_list"]) == 2
