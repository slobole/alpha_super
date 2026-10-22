"""A role handover preserves physical account ownership across closed markets."""

from datetime import UTC, date, datetime, timedelta

import pytest

from alpha.live.client_reporting import build_client_report_dict
from test_client_reporting import client_config_dict, nav_attributes_dict, snapshot_obj


def _handover_fixture_tuple(session_date_tuple=("2026-09-03", "2026-09-04", "2026-09-08")):
    first_str, _middle_str, pod_start_str = session_date_tuple
    reserve_end_str = (date.fromisoformat(pod_start_str) - timedelta(days=1)).isoformat()
    client_dict = client_config_dict()
    client_dict["mandate_start_date"] = first_str
    client_dict["accounts"][0]["effective_from"] = first_str
    client_dict["client_twr"] = {
        "method": "daily_nav_eod_v1", "reviewed_by": "Independent synthetic reviewer",
        "evidence_ref": "tests/test_other_account_handover_calendar.py",
    }
    client_dict["accounts"].extend([
        {"account_route": "U_TEST_B", "role": "reserve", "display_name": "Other",
         "effective_from": first_str, "effective_to": reserve_end_str},
        {"account_route": "U_TEST_B", "pod_id": "pod_b", "display_name": "Pod B",
         "effective_from": pod_start_str, "effective_to": None},
    ])
    attribute_list = []
    for index_int, date_str in enumerate(session_date_tuple):
        opening_int = 1000 + 10 * index_int
        attribute_list.extend([
            nav_attributes_dict(date_str=date_str, opening_str=str(opening_int),
                closing_str=str(opening_int + 10), twr_str=str(1000 / opening_int)),
            nav_attributes_dict("U_TEST_B", date_str=date_str, opening_str="5000",
                closing_str="5000", mtm="0", twr_str="0"),
        ])
    valuation_list = [
        {"account_route": "U_TEST_A", "pod_id": "strategy_a", "display_name": "Strategy A"},
        {"account_route": "U_TEST_B", "pod_id": "pod_b", "display_name": "Pod B"},
    ]
    return client_dict, attribute_list, valuation_list


def _report_dict(client_dict, attribute_list, valuation_list, *, to_date_str="2026-09-08"):
    return build_client_report_dict(client_dict, snapshot_obj(attribute_list),
        from_date_str=client_dict["mandate_start_date"], to_date_str=to_date_str,
        as_of_ts=datetime.combine(date.fromisoformat(to_date_str) + timedelta(days=1),
            datetime.min.time(), tzinfo=UTC) + timedelta(hours=14),
        valuation_account_list=valuation_list)


@pytest.mark.parametrize("session_date_tuple", [
    ("2026-08-06", "2026-08-07", "2026-08-10"),
    ("2026-09-03", "2026-09-04", "2026-09-08"),
])
def test_weekend_or_holiday_handover_does_not_demand_an_artificial_nav_day(session_date_tuple):
    client_dict, attribute_list, valuation_list = _handover_fixture_tuple(session_date_tuple)
    result_dict = _report_dict(client_dict, attribute_list, valuation_list,
        to_date_str=session_date_tuple[-1])
    assert result_dict["status_str"] == "ready", result_dict["issue_list"]
    assert result_dict["twr_float"] == pytest.approx(.005)
    assert result_dict["pnl_float"] == 30
    assert result_dict["scope_movement_float"] == 0
    assert [row_dict["market_date_str"] for row_dict in result_dict["daily_book_list"]] == list(session_date_tuple)


@pytest.mark.parametrize("transfer_date_str", ["2026-09-06", "2026-09-07"])
def test_observed_non_session_internal_transfer_is_retained_across_handover(transfer_date_str):
    client_dict, attribute_list, valuation_list = _handover_fixture_tuple()
    attribute_list.extend([
        nav_attributes_dict(date_str=transfer_date_str, opening_str="1020", closing_str="920",
            mtm="0", twr_str="0", internalCashTransfers="-100"),
        nav_attributes_dict("U_TEST_B", date_str=transfer_date_str, opening_str="5000",
            closing_str="5100", mtm="0", twr_str="0", internalCashTransfers="100"),
    ])
    for attribute_dict in attribute_list:
        if attribute_dict["toDate"] == "20260908":
            attribute_dict.update(startingValue="920" if attribute_dict["accountId"] == "U_TEST_A" else "5100",
                endingValue="930" if attribute_dict["accountId"] == "U_TEST_A" else "5100",
                twr=str(1000 / 920) if attribute_dict["accountId"] == "U_TEST_A" else "0")
    result_dict = _report_dict(client_dict, attribute_list, valuation_list)
    assert result_dict["status_str"] == "ready", result_dict["issue_list"]
    assert result_dict["twr_float"] == pytest.approx(.005)
    assert result_dict["capital_movement_float"] == 0
    assert transfer_date_str in [row_dict["market_date_str"] for row_dict in result_dict["daily_book_list"]]


@pytest.mark.parametrize("failure_str", ["missing_session", "continuity", "unmatched_sunday"])
def test_handover_keeps_missing_sessions_continuity_and_transfer_gates(failure_str):
    client_dict, attribute_list, valuation_list = _handover_fixture_tuple()
    if failure_str == "missing_session":
        attribute_list = [attribute_dict for attribute_dict in attribute_list
            if (attribute_dict["accountId"], attribute_dict["toDate"]) != ("U_TEST_B", "20260904")]
    elif failure_str == "continuity":
        for attribute_dict in attribute_list:
            if (attribute_dict["accountId"], attribute_dict["toDate"]) == ("U_TEST_B", "20260908"):
                attribute_dict.update(startingValue="5001", endingValue="5001")
    else:
        attribute_list.extend([
            nav_attributes_dict(date_str="2026-09-06", opening_str="1020", closing_str="1020", mtm="0", twr_str="0"),
            nav_attributes_dict("U_TEST_B", date_str="2026-09-06", opening_str="5000",
                closing_str="5100", mtm="0", twr_str="0", internalCashTransfers="100"),
        ])
        for attribute_dict in attribute_list:
            if (attribute_dict["accountId"], attribute_dict["toDate"]) == ("U_TEST_B", "20260908"):
                attribute_dict.update(startingValue="5100", endingValue="5100")
    result_dict = _report_dict(client_dict, attribute_list, valuation_list)
    assert result_dict["status_str"] == "draft"
    assert result_dict["twr_float"] is None


@pytest.mark.parametrize("boundary_str", ["real_entry", "real_exit", "ownership_gap"])
def test_real_non_session_membership_boundaries_are_not_suppressed(boundary_str):
    client_dict, attribute_list, _valuation_list = _handover_fixture_tuple()
    reserve_dict = client_dict["accounts"][1]
    if boundary_str == "real_entry":
        reserve_dict["effective_from"] = "2026-09-06"
    elif boundary_str == "real_exit":
        client_dict["accounts"].pop()
    else:
        reserve_dict["effective_to"] = "2026-09-04"
    result_dict = _report_dict(client_dict, attribute_list, None)
    assert result_dict["status_str"] == "draft"
    assert result_dict["twr_float"] is None


@pytest.mark.parametrize("boundary_str", ["entry", "exit"])
def test_one_day_period_does_not_confuse_real_boundary_with_role_handover(boundary_str):
    client_dict, attribute_list, _valuation_list = _handover_fixture_tuple()
    if boundary_str == "entry":
        # The real entry is Labor Day, also the last day of the reserve role.
        client_dict["accounts"][1]["effective_from"] = "2026-09-07"
    else:
        # The final, one-day Pod role begins and ends Labor Day. Its real exit
        # still needs that day's last owned NAV; Friday cannot stand in for it.
        client_dict["accounts"][1]["effective_to"] = "2026-09-06"
        client_dict["accounts"][2].update(effective_from="2026-09-07", effective_to="2026-09-07")
    result_dict = _report_dict(client_dict, attribute_list, None)
    assert result_dict["status_str"] == "draft", result_dict["issue_list"]
    assert result_dict["twr_float"] is None
    assert any("Missing broker days: 2026-09-07" in issue_str for issue_str in result_dict["issue_list"])
