"""Local broker valuations are not strategy membership or return calculations."""

from copy import deepcopy
from datetime import UTC, datetime

import pytest

from alpha.live.client_reporting import ClientReportingError, build_client_report_dict, validate_client_registry_dict
from alpha.live.investor_report import build_investor_snapshot_dict
from test_client_reporting import client_config_dict, nav_attributes_dict, snapshot_obj


def fixture_tuple():
    client_dict = client_config_dict(second_bool=True)
    identity_list = [{field_str: account_dict[field_str] for field_str in ("account_route", "pod_id", "display_name")}
        for account_dict in client_dict["accounts"]]
    attribute_list = [nav_attributes_dict(account_str=account_str, date_str=f"2026-09-0{day_int}",
        opening_str=str(1000 + (day_int - 1) * 10), closing_str=str(1000 + day_int * 10),
        twr_str=str(1000 / (1000 + (day_int - 1) * 10)))
        for day_int in (1, 2, 3) for account_str in ("U_TEST_A", "U_TEST_B")]
    return client_dict, identity_list, attribute_list


def local_report_dict(client_dict, identity_list, attribute_list, *, to_str="2026-09-03", as_of_ts=None):
    return build_client_report_dict(client_dict, snapshot_obj(attribute_list),
        valuation_account_list=identity_list, from_date_str="2026-09-01", to_date_str=to_str,
        as_of_ts=as_of_ts or datetime(2026, 9, 8, 23, tzinfo=UTC))


def test_retired_strategy_still_contributes_broker_account_nav_not_assumed_exit():
    client_dict, identity_list, attribute_list = fixture_tuple()
    client_dict["accounts"][1]["effective_to"] = "2026-09-01"
    report_dict = local_report_dict(client_dict, identity_list, attribute_list)
    assert report_dict["opening_nav_float"] == 2000
    assert report_dict["closing_nav_float"] == 2060
    assert report_dict["nav_coverage_complete_bool"] is True
    assert report_dict["coverage_complete_bool"] is False
    assert report_dict["scope_movement_float"] is None and report_dict["pnl_float"] is None
    assert report_dict["twr_float"] is None
    assert [row_dict["twr_float"] for row_dict in report_dict["strategy_list"]] == pytest.approx([.03, .01])
    investor_dict = build_investor_snapshot_dict(report_dict)
    assert investor_dict["closing_nav_float"] == 2060 and investor_dict["document_status_str"] == "draft"


def test_account_nav_before_first_strategy_return_is_not_discarded():
    client_dict, identity_list, attribute_list = fixture_tuple()
    client_dict["accounts"][1]["effective_from"] = "2026-09-03"
    report_dict = local_report_dict(client_dict, identity_list, attribute_list, to_str="2026-09-01")
    assert report_dict["closing_nav_float"] == 2020
    assert len(report_dict["strategy_list"]) == 1
    assert report_dict["twr_float"] is None
    assert report_dict["pnl_float"] is None


def test_missing_retired_account_is_unknown_never_zero():
    client_dict, identity_list, attribute_list = fixture_tuple()
    client_dict["accounts"][1]["effective_to"] = "2026-09-01"
    attribute_list = [row_dict for row_dict in attribute_list if not (row_dict["accountId"] == "U_TEST_B" and row_dict["fromDate"] == "20260903")]
    report_dict = local_report_dict(client_dict, identity_list, attribute_list)
    assert report_dict["opening_nav_float"] == 2000 and report_dict["closing_nav_float"] is None
    assert report_dict["nav_issue_list"] == ["strategy_b / U_TEST_B: missing IBKR NAV for 2026-09-03."]


def test_missing_middle_date_keeps_endpoints_and_chart_gap():
    client_dict, identity_list, attribute_list = fixture_tuple()
    attribute_list = [row_dict for row_dict in attribute_list if row_dict["fromDate"] != "20260902"]
    report_dict = local_report_dict(client_dict, identity_list, attribute_list)
    assert report_dict["opening_nav_float"] == 2000 and report_dict["closing_nav_float"] == 2060
    assert [row_dict["nav_float"] for row_dict in report_dict["daily_book_list"]] == [2020, None, 2060]
    assert report_dict["nav_coverage_complete_bool"] is False
    assert report_dict["pnl_float"] is None and report_dict["twr_float"] is None


@pytest.mark.parametrize("empty_bool", [False, True])
def test_nav_only_identities_do_not_unlock_partial_returns_or_finality(empty_bool):
    client_dict, identity_list, attribute_list = fixture_tuple()
    client_dict["accounts"] = [] if empty_bool else client_dict["accounts"][:1]
    report_dict = local_report_dict(client_dict, identity_list, attribute_list)
    assert report_dict["closing_nav_float"] == 2060
    assert report_dict["coverage_complete_bool"] is False and report_dict["flows_complete_bool"] is False
    assert report_dict["twr_float"] is None and report_dict["status_str"] == "draft"
    assert report_dict["source_list"]
    assert len(report_dict["strategy_list"]) == (0 if empty_bool else 1)
    if empty_bool:
        with pytest.raises(ClientReportingError, match="at least one account period"):
            validate_client_registry_dict({"schema_version": 1, "clients": [client_dict]})


def test_today_nav_is_pending_even_with_complete_rows():
    client_dict, identity_list, attribute_list = fixture_tuple()
    report_dict = local_report_dict(client_dict, identity_list, attribute_list, as_of_ts=datetime(2026, 9, 3, 23, tzinfo=UTC))
    assert report_dict["opening_nav_float"] == 2000 and report_dict["closing_nav_float"] is None
    assert "2026-09-03 is not finalized" in " ".join(report_dict["nav_issue_list"])


def test_nav_only_revision_and_scope_are_hashed():
    client_dict, identity_list, attribute_list = fixture_tuple()
    client_dict["accounts"] = []
    report_dict = local_report_dict(client_dict, identity_list, attribute_list)
    corrected_list = deepcopy(attribute_list)
    corrected_list[-1]["endingValue"] = "1040"
    corrected_dict = local_report_dict(client_dict, identity_list, corrected_list)
    assert corrected_dict["closing_nav_float"] == 2070
    assert corrected_dict["report_hash_str"] != report_dict["report_hash_str"]
    single_dict = local_report_dict(client_dict, identity_list[:1], attribute_list)
    assert single_dict["scope_hash_str"] != report_dict["scope_hash_str"]


def test_local_valuation_does_not_change_complete_strategy_math_or_names():
    client_dict, identity_list, attribute_list = fixture_tuple()
    original_dict = build_client_report_dict(client_dict, snapshot_obj(attribute_list), from_date_str="2026-09-01",
        to_date_str="2026-09-03", as_of_ts=datetime(2026, 9, 8, 23, tzinfo=UTC))
    report_dict = local_report_dict(client_dict, identity_list, attribute_list)
    for field_str in ("strategy_list", "pnl_float", "twr_float", "capital_movement_float", "scope_movement_float", "status_str"):
        assert report_dict[field_str] == original_dict[field_str]


def test_duplicate_valuation_identity_rejected():
    client_dict, identity_list, attribute_list = fixture_tuple()
    with pytest.raises(ClientReportingError, match="Duplicate local valuation"):
        local_report_dict(client_dict, [*identity_list, identity_list[0]], attribute_list)


def test_configured_client_twr_is_unchanged_by_local_valuation():
    from test_client_twr import twr_config_dict
    _, identity_list, attribute_list = fixture_tuple()
    client_dict = twr_config_dict()
    original_dict = build_client_report_dict(client_dict, snapshot_obj(attribute_list), from_date_str="2026-09-01",
        to_date_str="2026-09-03", as_of_ts=datetime(2026, 9, 8, 23, tzinfo=UTC))
    report_dict = local_report_dict(client_dict, identity_list, attribute_list)
    assert report_dict["twr_float"] == pytest.approx(.03)
    for field_str in ("twr_float", "twr_daily_list", "return_path_list", "twr_method_id_str", "pnl_float", "strategy_list"):
        assert report_dict[field_str] == original_dict[field_str]
