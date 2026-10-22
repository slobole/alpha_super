"""Independent synthetic accounting for reserves without a strategy identity."""

from copy import deepcopy
from datetime import UTC, datetime
from decimal import Decimal
import hashlib

import pytest

from alpha.live.client_reporting import (
    BrokerReportingSnapshot, ClientReportingError, build_client_report_dict,
    load_broker_reporting_snapshot, parse_broker_nav_import, validate_client_registry_dict,
)
from alpha.live.ibkr_performance import PerformanceStore
from alpha.live.investor_report import build_investor_snapshot_dict
from test_client_reporting import AS_OF_TS, client_config_dict, insert_import, nav_attributes_dict, xml_text_str


def reserve_client_dict():
    client_dict = client_config_dict(second_bool=True)
    client_dict["accounts"][1].pop("pod_id")
    client_dict["accounts"][1].update(role="reserve", display_name="Reserve cash")
    client_dict["client_twr"] = {
        "method": "daily_nav_eod_v1", "reviewed_by": "Synthetic accounting fixture",
        "evidence_ref": "tests/test_client_reserve_accounts.py",
    }
    return client_dict


def reserve_report_dict(attribute_list, *, client_dict=None, to_date_str="2026-09-01",
                        as_of_ts=AS_OF_TS, valuation_account_list=None):
    raw_xml_str = xml_text_str(attribute_list)
    checksum_str = hashlib.sha256(raw_xml_str.encode()).hexdigest()
    row_list = parse_broker_nav_import(raw_xml_str,
        allowed_account_set={attribute_dict["accountId"] for attribute_dict in attribute_list},
        query_name_str="TEST_NAV", source_import_id_int=1, source_checksum_str=checksum_str)
    snapshot_obj = BrokerReportingSnapshot(tuple(row_list), ({
        "import_id_int": 1, "checksum_str": checksum_str,
        "imported_timestamp_str": AS_OF_TS.isoformat(), "query_name_str": "TEST_NAV",
    },))
    return build_client_report_dict(client_dict or reserve_client_dict(), snapshot_obj,
        from_date_str="2026-09-01", to_date_str=to_date_str, as_of_ts=as_of_ts,
        valuation_account_list=valuation_account_list)


def reserve_row_dict(**attribute_dict):
    return nav_attributes_dict("U_TEST_B", opening_str="9000", closing_str="9000",
                               twr_str="0", mtm="0", **attribute_dict)


def test_reserve_enters_client_capital_but_never_becomes_strategy():
    result_dict = reserve_report_dict([nav_attributes_dict(), reserve_row_dict()])
    assert result_dict["opening_nav_float"] == 10000
    assert result_dict["closing_nav_float"] == 10010
    assert result_dict["pnl_float"] == 10
    assert result_dict["twr_float"] == pytest.approx(.001)
    assert len(result_dict["strategy_list"]) == 1
    assert result_dict["strategy_list"][0]["twr_float"] == pytest.approx(.01)
    assert result_dict["status_str"] == "ready"
    assert result_dict["daily_book_list"][0]["account_count_int"] == 2
    snapshot_dict = build_investor_snapshot_dict(result_dict)
    assert snapshot_dict["closing_nav_float"] == 10010
    assert len(snapshot_dict["strategy_list"]) == 1


@pytest.mark.parametrize("transfer_int", [-500, 500])
def test_reserve_transfer_is_neither_client_profit_nor_external_capital(transfer_int):
    result_dict = reserve_report_dict([
        nav_attributes_dict(closing_str=str(1010 + transfer_int), internalCashTransfers=str(transfer_int)),
        nav_attributes_dict("U_TEST_B", opening_str="9000", closing_str=str(9000 - transfer_int),
                            twr_str="0", mtm="0", internalCashTransfers=str(-transfer_int)),
    ])
    assert result_dict["pnl_float"] == 10
    assert result_dict["capital_movement_float"] == 0
    assert result_dict["twr_float"] == pytest.approx(.001)


def test_unmatched_transfer_to_excluded_account_withholds_client_return():
    result_dict = reserve_report_dict([nav_attributes_dict(),
        nav_attributes_dict("U_TEST_B", opening_str="9000", closing_str="8500", twr_str="0",
                            mtm="0", internalCashTransfers="-500"),
        nav_attributes_dict("U_EXCLUDED", opening_str="0", closing_str="500", twr_str="0",
                            mtm="0", internalCashTransfers="500")])
    assert result_dict["closing_nav_float"] == 9510
    assert result_dict["twr_float"] is None
    assert "transit" in result_dict["twr_reason_str"]
    assert len(result_dict["strategy_list"]) == 1


@pytest.mark.parametrize("flow_int", [500, -500])
def test_reserve_external_cash_movement_is_not_profit(flow_int):
    result_dict = reserve_report_dict([nav_attributes_dict(),
        nav_attributes_dict("U_TEST_B", opening_str="9000", closing_str=str(9000 + flow_int),
                            twr_str="0", mtm="0", depositsWithdrawals=str(flow_int))])
    assert result_dict["pnl_float"] == 10
    assert result_dict["capital_movement_float"] == flow_int
    assert result_dict["twr_float"] == pytest.approx(.001)


def test_reserve_interest_accrual_and_fee_are_economic_profit_once():
    client_dict = reserve_client_dict()
    client_dict["nav_bridge"]["economic_fields"] += ["interest", "changeInInterestAccruals", "commissions"]
    row_list = [nav_attributes_dict(), nav_attributes_dict("U_TEST_B", opening_str="9000",
        closing_str="9002", mtm="0", twr_str=".0222222222222222")]
    for attribute_dict in row_list:
        attribute_dict.update(interest="0", changeInInterestAccruals="0", commissions="0")
    row_list[1].update(interest="5", changeInInterestAccruals="-2", commissions="-1")
    result_dict = reserve_report_dict(row_list, client_dict=client_dict)
    assert result_dict["pnl_float"] == 12
    assert result_dict["twr_float"] == pytest.approx(.0012)


def test_observed_zero_reserve_is_valid_when_client_opening_is_positive():
    result_dict = reserve_report_dict([nav_attributes_dict(),
        nav_attributes_dict("U_TEST_B", opening_str="0", closing_str="0", mtm="0", twr_str="0")])
    assert result_dict["twr_float"] == pytest.approx(.01)
    assert result_dict["coverage_complete_bool"]


@pytest.mark.parametrize("missing_str", ["row", "billPay", "bad_bridge", "twr"])
def test_incomplete_reserve_never_disappears_from_aggregate_gates(missing_str):
    row_list = [nav_attributes_dict(), reserve_row_dict()]
    if missing_str == "row":
        row_list.pop()
    elif missing_str == "bad_bridge":
        row_list[1]["mtm"] = "1"
    else:
        row_list[1].pop(missing_str)
    if missing_str == "twr":
        with pytest.raises(ClientReportingError):
            reserve_report_dict(row_list)
        return
    result_dict = reserve_report_dict(row_list)
    assert result_dict["twr_float"] is None
    assert result_dict["pnl_float"] is None
    assert result_dict["strategy_list"][0]["twr_float"] == pytest.approx(.01)
    assert result_dict["status_str"] == "draft"


def test_one_strategy_with_reserve_cannot_fall_back_to_strategy_account_return():
    client_dict = reserve_client_dict()
    client_dict.pop("client_twr")
    result_dict = reserve_report_dict([nav_attributes_dict(), reserve_row_dict()], client_dict=client_dict)
    assert result_dict["twr_float"] is None


def test_reserve_only_book_is_not_a_strategy():
    client_dict = reserve_client_dict()
    client_dict["accounts"] = client_dict["accounts"][1:]
    result_dict = reserve_report_dict([reserve_row_dict()], client_dict=client_dict)
    assert result_dict["twr_float"] == 0
    assert result_dict["strategy_list"] == []


def test_reserve_late_scope_entry_is_capital_and_uses_its_opening_nav():
    client_dict = reserve_client_dict()
    client_dict["accounts"][1]["effective_from"] = "2026-09-02"
    result_dict = reserve_report_dict([nav_attributes_dict(),
        nav_attributes_dict(date_str="2026-09-02", opening_str="1010", closing_str="1020"),
        nav_attributes_dict("U_TEST_B", date_str="2026-09-02", opening_str="9000", closing_str="9000",
                            twr_str="0", mtm="0")], client_dict=client_dict, to_date_str="2026-09-02")
    assert result_dict["scope_movement_float"] == 9000
    assert result_dict["twr_float"] == pytest.approx(float(Decimal("1.01") * (1 + Decimal(10) / Decimal(10010)) - 1))


def test_current_day_reserve_does_not_unlock_final_return():
    result_dict = reserve_report_dict([nav_attributes_dict(), reserve_row_dict()],
        as_of_ts=datetime(2026, 9, 1, 23, tzinfo=UTC))
    assert result_dict["twr_float"] is None
    assert result_dict["closing_nav_float"] is None


def test_valuation_and_performance_include_the_same_reserve_identity():
    client_dict = reserve_client_dict()
    identity_list = [{field_str: account_dict[field_str]
        for field_str in ("account_route", "pod_id", "display_name", "role", "effective_from", "effective_to") if field_str in account_dict}
        for account_dict in client_dict["accounts"]]
    result_dict = reserve_report_dict([nav_attributes_dict(), reserve_row_dict()],
        client_dict=client_dict, valuation_account_list=identity_list)
    assert result_dict["scope_complete_bool"]
    assert result_dict["twr_float"] == pytest.approx(.001)
    assert result_dict["closing_nav_float"] == 10010


@pytest.mark.parametrize("case_str", ["pod", "role", "duplicate", "early_date"])
def test_reserve_identity_cannot_bypass_ownership_or_impersonate_strategy(case_str):
    client_dict = reserve_client_dict()
    if case_str == "pod":
        client_dict["accounts"][1]["pod_id"] = "pretend_strategy"
    elif case_str == "role":
        client_dict["accounts"][1]["role"] = "whatever"
    elif case_str == "duplicate":
        client_dict["accounts"].append(deepcopy(client_dict["accounts"][1]))
    else:
        client_dict["accounts"][1]["effective_from"] = "2026-08-31"
    with pytest.raises(ClientReportingError):
        validate_client_registry_dict({"schema_version": 1, "clients": [client_dict]})


def test_legacy_strategy_facts_match_after_reserve_role_separates_presentation():
    legacy_dict = reserve_report_dict([nav_attributes_dict()], client_dict=client_config_dict())
    reserve_dict = reserve_report_dict([nav_attributes_dict(), reserve_row_dict()])
    assert legacy_dict["strategy_list"] == reserve_dict["strategy_list"]
    assert legacy_dict["scope_hash_str"] != reserve_dict["scope_hash_str"]


@pytest.mark.parametrize("scope_str", ["enters", "exits"])
def test_nav_fallback_obeys_reserve_dates_without_trimming_legacy_strategy_valuation(scope_str):
    client_dict = reserve_client_dict()
    if scope_str == "enters":
        client_dict["accounts"][1]["effective_from"] = "2026-09-02"
    else:
        client_dict["accounts"][1]["effective_to"] = "2026-09-01"
    identity_list = deepcopy(client_dict["accounts"])
    result_dict = reserve_report_dict([nav_attributes_dict(), reserve_row_dict(),
        nav_attributes_dict(date_str="2026-09-02", opening_str="1010", closing_str="1020"),
        nav_attributes_dict("U_TEST_B", date_str="2026-09-02", opening_str="9000", closing_str="9000", twr_str="0", mtm="0")],
        client_dict=client_dict, to_date_str="2026-09-02", valuation_account_list=identity_list)
    assert result_dict["scope_complete_bool"]
    assert result_dict["twr_float"] is not None
    assert [day_dict["nav_float"] for day_dict in result_dict["daily_book_list"]] == (
        [1010, 10020] if scope_str == "enters" else [10010, 1020])


def test_unowned_extra_account_never_enters_nav_or_return_denominator():
    result_dict = reserve_report_dict([nav_attributes_dict(), reserve_row_dict(),
        nav_attributes_dict("U_EXCLUDED", opening_str="1000000", closing_str="1100000", mtm="100000", twr_str="10")])
    assert result_dict["opening_nav_float"] == 10000
    assert result_dict["closing_nav_float"] == 10010
    assert result_dict["twr_float"] == pytest.approx(.001)


def test_reserve_valuation_dates_cannot_override_financial_ownership():
    client_dict = reserve_client_dict()
    identity_list = deepcopy(client_dict["accounts"])
    identity_list[1]["effective_from"] = "2026-09-02"
    with pytest.raises(ClientReportingError, match="dates disagree"):
        reserve_report_dict([nav_attributes_dict(), reserve_row_dict()],
            client_dict=client_dict, valuation_account_list=identity_list)


def test_source_correction_omitting_reserve_never_resurrects_previous_nav(tmp_path):
    database_path_obj = tmp_path / "reporting.sqlite3"
    PerformanceStore(str(database_path_obj)).initialize()
    insert_import(database_path_obj, [nav_attributes_dict(), reserve_row_dict()],
                  from_str="2026-09-01", to_str="2026-09-01")
    first_snapshot_obj = load_broker_reporting_snapshot(str(database_path_obj),
        allowed_account_set={"U_TEST_A", "U_TEST_B"}, query_name_str="TEST_NAV")
    first_report_dict = build_client_report_dict(reserve_client_dict(), first_snapshot_obj,
        from_date_str="2026-09-01", to_date_str="2026-09-01", as_of_ts=AS_OF_TS)
    assert first_report_dict["twr_float"] == pytest.approx(.001)
    insert_import(database_path_obj, [nav_attributes_dict()],
                  from_str="2026-09-01", to_str="2026-09-01")
    corrected_snapshot_obj = load_broker_reporting_snapshot(str(database_path_obj),
        allowed_account_set={"U_TEST_A", "U_TEST_B"}, query_name_str="TEST_NAV")
    corrected_report_dict = build_client_report_dict(reserve_client_dict(), corrected_snapshot_obj,
        from_date_str="2026-09-01", to_date_str="2026-09-01", as_of_ts=AS_OF_TS)
    assert corrected_report_dict["twr_float"] is None
    assert corrected_report_dict["closing_nav_float"] is None
    assert corrected_report_dict["pnl_float"] is None
    assert corrected_report_dict["scope_hash_str"] == first_report_dict["scope_hash_str"]
    assert corrected_report_dict["report_hash_str"] != first_report_dict["report_hash_str"]


@pytest.mark.parametrize("opening_int,gain_int", [(0, 0), (100, 5)])
def test_multiple_reserves_count_once_without_additional_strategy(opening_int, gain_int):
    client_dict = reserve_client_dict()
    client_dict["accounts"].append({
        "account_route": "U_TEST_C", "role": "reserve", "display_name": "Second reserve",
        "effective_from": "2026-09-01", "effective_to": None,
    })
    result_dict = reserve_report_dict([nav_attributes_dict(), reserve_row_dict(),
        nav_attributes_dict("U_TEST_C", opening_str=str(opening_int), closing_str=str(opening_int + gain_int),
                            mtm=str(gain_int), twr_str="5" if gain_int else "0")],
        client_dict=client_dict, valuation_account_list=deepcopy(client_dict["accounts"]))
    assert result_dict["opening_nav_float"] == 10000 + opening_int
    assert result_dict["closing_nav_float"] == 10010 + opening_int + gain_int
    assert result_dict["twr_float"] == pytest.approx((10 + gain_int) / (10000 + opening_int))
    assert len(result_dict["financial_account_list"]) == 3
    assert len(result_dict["strategy_list"]) == 1


def test_cross_day_reserve_transfer_keeps_whole_return_unavailable():
    result_dict = reserve_report_dict([nav_attributes_dict(),
        nav_attributes_dict("U_TEST_B", opening_str="9000", closing_str="8500", twr_str="0", mtm="0", internalCashTransfers="-500"),
        nav_attributes_dict(date_str="2026-09-02", opening_str="1010", closing_str="1520", internalCashTransfers="500"),
        nav_attributes_dict("U_TEST_B", date_str="2026-09-02", opening_str="8500", closing_str="8500", twr_str="0", mtm="0")],
        to_date_str="2026-09-02")
    assert result_dict["pnl_float"] == 20
    assert result_dict["twr_float"] is None
    assert result_dict["return_path_list"] == []
    assert "transit" in result_dict["twr_reason_str"]


def test_pre_scope_weekend_reserve_row_does_not_create_a_false_missing_strategy_day():
    client_dict = reserve_client_dict()
    client_dict["accounts"][1]["effective_from"] = "2026-09-08"
    row_list = [nav_attributes_dict(date_str=f"2026-09-{day_int:02d}",
        opening_str=str(1000 + 10 * index_int), closing_str=str(1010 + 10 * index_int))
        for index_int, day_int in enumerate((1, 2, 3, 4, 8))]
    row_list.extend(nav_attributes_dict("U_TEST_B", date_str=f"2026-09-{day_int:02d}",
        opening_str="9000", closing_str="9000", twr_str="0", mtm="0") for day_int in (6, 8))
    result_dict = reserve_report_dict(row_list, client_dict=client_dict,
        to_date_str="2026-09-08", as_of_ts=datetime(2026, 9, 9, 20, tzinfo=UTC),
        valuation_account_list=deepcopy(client_dict["accounts"]))
    assert result_dict["nav_coverage_complete_bool"]
    assert result_dict["twr_float"] is not None
    assert "2026-09-06" not in {day_dict["market_date_str"] for day_dict in result_dict["daily_book_list"]}
