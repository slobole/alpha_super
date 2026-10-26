"""Hand-calculated subset returns under the unchanged daily EOD convention."""

from copy import deepcopy
from dataclasses import replace

import pytest

from alpha.live.client_reporting import build_client_report_dict
from test_client_reporting import AS_OF_TS, client_config_dict, nav_attributes_dict, snapshot_obj


def _parent_dict():
    parent_dict = client_config_dict(second_bool=True)
    parent_dict["client_twr"] = {"method": "daily_nav_eod_v1", "reviewed_by": "Synthetic fixture",
        "evidence_ref": "Hand-calculated subset transfer cases; no real accounts"}
    return parent_dict


def _transfer_row_list():
    return [nav_attributes_dict(opening_str="100", closing_str="60", internalCashTransfers="-50", twr_str="10"),
        nav_attributes_dict("U_TEST_B", opening_str="100", closing_str="155",
            internalCashTransfers="50", mtm="5", twr_str="3")]


def _report_dict(parent_dict, row_list, *, route_set=None, parent_context_bool=True, to_str="2026-09-01"):
    selected_dict = deepcopy(parent_dict)
    if route_set is not None:
        selected_dict["accounts"] = [account_dict for account_dict in selected_dict["accounts"]
            if account_dict["account_route"] in route_set]
    return build_client_report_dict(selected_dict, snapshot_obj(row_list), from_date_str="2026-09-01",
        to_date_str=to_str, as_of_ts=AS_OF_TS,
        transfer_scope_client_dict=parent_dict if parent_context_bool else None)


def test_cross_program_flow_is_external_only_to_subset_and_never_counted_twice():
    parent_dict, row_list = _parent_dict(), _transfer_row_list()
    original_row_list, original_parent_dict = deepcopy(row_list), deepcopy(parent_dict)
    parent_report_dict = _report_dict(parent_dict, row_list, parent_context_bool=False)
    source_obj = snapshot_obj(row_list)
    snapshot_before_obj = deepcopy(source_obj)
    for route_str, expected_return_float, expected_pnl_float, expected_capital_float in (
            ("U_TEST_A", .1, 10., -50.), ("U_TEST_B", .05, 5., 50.)):
        subset_dict = deepcopy(parent_dict)
        subset_dict["accounts"] = [account_dict for account_dict in subset_dict["accounts"]
            if account_dict["account_route"] == route_str]
        default_dict = build_client_report_dict(subset_dict, source_obj, from_date_str="2026-09-01",
            to_date_str="2026-09-01", as_of_ts=AS_OF_TS)
        result_dict = build_client_report_dict(subset_dict, source_obj, from_date_str="2026-09-01",
            to_date_str="2026-09-01", as_of_ts=AS_OF_TS, transfer_scope_client_dict=parent_dict)
        assert default_dict["twr_float"] is None and "transit" in default_dict["twr_reason_str"]
        assert result_dict["status_str"] == "ready"
        assert result_dict["twr_float"] == pytest.approx(expected_return_float)
        assert result_dict["pnl_float"] == expected_pnl_float
        assert result_dict["capital_movement_float"] == expected_capital_float
        assert result_dict["twr_daily_list"][0]["opening_nav_float"] == 100
        assert result_dict["strategy_list"] == default_dict["strategy_list"]  # Official account TWR is untouched.
        assert result_dict["transfer_scope_dict"]["report_hash_str"] == parent_report_dict["report_hash_str"]
        assert result_dict["report_hash_str"] != default_dict["report_hash_str"]
    assert parent_report_dict["twr_float"] == pytest.approx(.075)
    assert parent_report_dict["capital_movement_float"] == 0
    assert row_list == original_row_list and parent_dict == original_parent_dict
    assert source_obj == snapshot_before_obj


def test_within_scope_transfer_and_no_transfer_keep_default_results_identical():
    parent_dict = _parent_dict()
    for row_list in (_transfer_row_list(), [nav_attributes_dict(), nav_attributes_dict("U_TEST_B")]):
        default_dict = _report_dict(parent_dict, row_list, parent_context_bool=False)
        context_dict = _report_dict(parent_dict, row_list)
        assert context_dict == default_dict
        assert "transfer_scope_dict" not in context_dict


def test_reserve_to_pod_transfer_uses_full_ownership_not_strategy_only_scope():
    parent_dict = _parent_dict()
    parent_dict["accounts"][1].pop("pod_id")
    parent_dict["accounts"][1]["role"] = "reserve"
    row_list = [nav_attributes_dict(opening_str="100", closing_str="160", internalCashTransfers="50", twr_str="10"),
        nav_attributes_dict("U_TEST_B", opening_str="100", closing_str="55",
            internalCashTransfers="-50", mtm="5", twr_str="5")]
    result_dict = _report_dict(parent_dict, row_list, route_set={"U_TEST_A"})
    assert result_dict["twr_float"] == pytest.approx(.1)
    assert result_dict["capital_movement_float"] == 50 and result_dict["pnl_float"] == 10
    incomplete_parent_dict = deepcopy(parent_dict)
    incomplete_parent_dict["accounts"] = incomplete_parent_dict["accounts"][:1]
    assert _report_dict(incomplete_parent_dict, row_list)["twr_float"] is None


@pytest.mark.parametrize("case_str", ["unequal", "missing_row", "missing_field", "bad_bridge", "linking", "source_unavailable"])
def test_parent_failure_cannot_authorize_subset_transfer(case_str):
    parent_dict, row_list = _parent_dict(), _transfer_row_list()
    if case_str == "unequal":
        row_list[1].update(internalCashTransfers="49", endingValue="154")
    elif case_str == "missing_row":
        row_list.pop()
    elif case_str == "missing_field":
        row_list[1].pop("billPay")
    elif case_str == "bad_bridge":
        row_list[1]["mtm"] = "6"
    elif case_str == "linking":
        row_list[1].update(linkingAdjustments="2", endingValue="157")
    selected_dict = deepcopy(parent_dict)
    selected_dict["accounts"] = selected_dict["accounts"][:1]
    source_obj = snapshot_obj(row_list)
    if case_str == "source_unavailable":
        source_obj = replace(source_obj, unavailable_reason_str="Source checksum verification failed.")
    result_dict = build_client_report_dict(selected_dict, source_obj, from_date_str="2026-09-01",
        to_date_str="2026-09-01", as_of_ts=AS_OF_TS, transfer_scope_client_dict=parent_dict)
    assert result_dict["twr_float"] is None
    assert result_dict["return_path_list"] == result_dict["twr_daily_list"] == []
    assert result_dict["status_str"] == "draft"
    assert not result_dict["transfer_scope_dict"]["verified_bool"]
    assert "Containing transfer scope" in result_dict["twr_reason_str"]
    assert result_dict["pnl_float"] == 10 and result_dict["capital_movement_float"] == -50


def test_different_day_transfer_remains_unavailable_even_when_period_net_is_zero():
    row_list = [nav_attributes_dict(opening_str="100", closing_str="50", internalCashTransfers="-50", mtm="0", twr_str="0"),
        nav_attributes_dict("U_TEST_B", opening_str="100", closing_str="100", mtm="0", twr_str="0"),
        nav_attributes_dict(date_str="2026-09-02", opening_str="50", closing_str="55", mtm="5", twr_str="10"),
        nav_attributes_dict("U_TEST_B", date_str="2026-09-02", opening_str="100", closing_str="150",
            internalCashTransfers="50", mtm="0", twr_str="0")]
    result_dict = _report_dict(_parent_dict(), row_list, route_set={"U_TEST_A"}, to_str="2026-09-02")
    assert result_dict["twr_float"] is None and "transit" in result_dict["twr_reason_str"]
    assert result_dict["pnl_float"] == 5


@pytest.mark.parametrize("case_str", ["client", "query", "bridge", "method", "missing_owner", "date_gap"])
def test_foreign_or_incomplete_ownership_context_is_not_accepted(case_str):
    parent_dict = _parent_dict()
    selected_dict = deepcopy(parent_dict)
    selected_dict["accounts"] = selected_dict["accounts"][:1]
    if case_str == "client":
        parent_dict["client_id"] = "another_client"
    elif case_str == "query":
        parent_dict["query_name"] = "ANOTHER_QUERY"
    elif case_str == "bridge":
        parent_dict["nav_bridge"]["profile_id"] = "ANOTHER_PROFILE"
    elif case_str == "method":
        parent_dict.pop("client_twr")
    elif case_str == "missing_owner":
        parent_dict["accounts"] = parent_dict["accounts"][1:]
    else:
        parent_dict["accounts"][0]["effective_from"] = "2026-09-02"
    result_dict = build_client_report_dict(selected_dict, snapshot_obj(_transfer_row_list()),
        from_date_str="2026-09-01", to_date_str="2026-09-01", as_of_ts=AS_OF_TS,
        transfer_scope_client_dict=parent_dict)
    assert result_dict["twr_float"] is None and not result_dict["transfer_scope_dict"]["verified_bool"]
    assert result_dict["pnl_float"] == 10


def test_parent_role_change_preserves_ownership_and_linked_subset_history():
    parent_dict = _parent_dict()
    reserve_dict = deepcopy(parent_dict["accounts"][1])
    reserve_dict.pop("pod_id")
    reserve_dict.update(role="reserve", effective_to="2026-09-01")
    parent_dict["accounts"][1]["effective_from"] = "2026-09-02"
    parent_dict["accounts"].insert(1, reserve_dict)
    row_list = _transfer_row_list() + [
        nav_attributes_dict(date_str="2026-09-02", opening_str="60", closing_str="66", mtm="6", twr_str="10"),
        nav_attributes_dict("U_TEST_B", date_str="2026-09-02", opening_str="155", closing_str="160", mtm="5", twr_str="1")]
    result_dict = _report_dict(parent_dict, row_list, route_set={"U_TEST_A"}, to_str="2026-09-02")
    assert result_dict["twr_float"] == pytest.approx(1.1 * 1.1 - 1)
    assert result_dict["pnl_float"] == 16 and result_dict["scope_movement_float"] == 0


def test_parent_correction_changes_subset_provenance_even_if_subset_numbers_do_not():
    parent_dict, row_list = _parent_dict(), _transfer_row_list()
    first_dict = _report_dict(parent_dict, row_list, route_set={"U_TEST_A"})
    row_list[1].update(mtm="6", endingValue="156")
    corrected_dict = _report_dict(parent_dict, row_list, route_set={"U_TEST_A"})
    assert first_dict["twr_float"] == corrected_dict["twr_float"] == .1
    assert first_dict["transfer_scope_dict"]["report_hash_str"] != corrected_dict["transfer_scope_dict"]["report_hash_str"]
    assert first_dict["report_hash_str"] != corrected_dict["report_hash_str"]
