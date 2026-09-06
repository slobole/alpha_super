"""Synthetic accounting fixtures, never evidence for an actual investor NAV."""

from copy import deepcopy
from datetime import UTC, datetime
import hashlib
import sqlite3
import xml.etree.ElementTree as ElementTree

import pytest

from alpha.live.client_reporting import (
    BrokerReportingSnapshot, CAPITAL_FIELD_TUPLE, ClientReportingError,
    build_client_report_dict, load_broker_reporting_snapshot,
    parse_broker_nav_import, validate_client_registry_dict,
)
from alpha.live.ibkr_performance import PerformanceStore


AS_OF_TS = datetime(2026, 9, 8, 23, tzinfo=UTC)


def client_config_dict(*, second_bool=False, bridge_bool=True):
    client_dict = {
        "client_id": "sample", "display_name": "Sample client", "base_currency": "USD",
        "mandate_start_date": "2026-09-01", "fee_basis": "Synthetic test: broker-recorded charges only",
        "query_name": "TEST_NAV", "accounts": [{
            "account_route": "U_TEST_A", "pod_id": "strategy_a", "display_name": "Strategy A",
            "effective_from": "2026-09-01", "effective_to": None,
        }],
    }
    if second_bool:
        client_dict["accounts"].append({
            "account_route": "U_TEST_B", "pod_id": "strategy_b", "display_name": "Strategy B",
            "effective_from": "2026-09-01", "effective_to": None,
        })
    if bridge_bool:
        client_dict["nav_bridge"] = {
            "profile_id": "SYNTHETIC_TEST_ONLY", "evidence_ref": "tests/test_client_reporting.py",
            "reviewed_by": "synthetic fixture", "mode": "MTM", "nonoverlap_confirmed": True,
            "economic_fields": ["mtm"], "informational_fields": [],
        }
    return client_dict


def nav_attributes_dict(account_str="U_TEST_A", date_str="2026-09-01", opening_str="1000", closing_str="1010", twr_str="1", **extra_dict):
    attribute_dict = {
        "accountId": account_str, "currency": "USD", "fromDate": date_str.replace("-", ""),
        "toDate": date_str.replace("-", ""), "startingValue": opening_str,
        "endingValue": closing_str, "twr": twr_str, "mtm": "10",
        "linkingAdjustments": "0", **{field_str: "0" for field_str in CAPITAL_FIELD_TUPLE},
    }
    attribute_dict.update(extra_dict)
    return attribute_dict


def xml_text_str(attribute_list):
    root_obj = ElementTree.Element("FlexQueryResponse", queryName="TEST_NAV")
    statements_obj = ElementTree.SubElement(root_obj, "FlexStatements")
    for account_str in sorted({attribute_dict["accountId"] for attribute_dict in attribute_list}):
        statement_obj = ElementTree.SubElement(statements_obj, "FlexStatement", accountId=account_str)
        ElementTree.SubElement(statement_obj, "AccountInformation", accountId=account_str, currency="USD")
        for attribute_dict in attribute_list:
            if attribute_dict["accountId"] == account_str:
                ElementTree.SubElement(statement_obj, "ChangeInNAV", **attribute_dict)
    return ElementTree.tostring(root_obj, encoding="unicode")


def snapshot_obj(attribute_list, import_id_int=1):
    raw_xml_str = xml_text_str(attribute_list)
    checksum_str = hashlib.sha256(raw_xml_str.encode()).hexdigest()
    row_list = parse_broker_nav_import(
        raw_xml_str, allowed_account_set={"U_TEST_A", "U_TEST_B"}, query_name_str="TEST_NAV",
        source_import_id_int=import_id_int, source_checksum_str=checksum_str,
    )
    return BrokerReportingSnapshot(tuple(row_list), ({
        "import_id_int": import_id_int, "checksum_str": checksum_str,
        "imported_timestamp_str": "2026-09-08T22:00:00+00:00", "query_name_str": "TEST_NAV",
    },))


def report_dict(attribute_list, *, config_dict=None, to_str="2026-09-01"):
    return build_client_report_dict(
        config_dict or client_config_dict(), snapshot_obj(attribute_list),
        from_date_str="2026-09-01", to_date_str=to_str, as_of_ts=AS_OF_TS,
    )


def test_single_account_bridge_and_official_return():
    result_dict = report_dict([nav_attributes_dict()])
    assert result_dict["status_str"] == "ready"
    assert result_dict["opening_nav_float"] == 1000
    assert result_dict["closing_nav_float"] == 1010
    assert result_dict["pnl_float"] == 10
    assert result_dict["twr_float"] == pytest.approx(0.01)
    assert result_dict["source_list"][0]["import_id_int"] == 1


def test_missing_flow_profile_does_not_destroy_official_twr_or_invent_profit():
    result_dict = report_dict([nav_attributes_dict()], config_dict=client_config_dict(bridge_bool=False))
    assert result_dict["twr_float"] == pytest.approx(0.01)
    assert result_dict["closing_nav_float"] == 1010
    assert result_dict["pnl_float"] is None
    assert result_dict["capital_movement_float"] is None
    assert result_dict["status_str"] == "draft"


def test_deposit_is_not_profit_and_twr_is_not_nav_growth():
    result_dict = report_dict([nav_attributes_dict(closing_str="1510", depositsWithdrawals="500")])
    assert result_dict["pnl_float"] == 10
    assert result_dict["capital_movement_float"] == 500
    assert result_dict["twr_float"] == pytest.approx(.01)


def test_missing_zero_field_is_unknown_not_zero():
    attribute_dict = nav_attributes_dict()
    del attribute_dict["billPay"]
    result_dict = report_dict([attribute_dict])
    assert result_dict["pnl_float"] is None
    assert "billPay" in " ".join(result_dict["issue_list"])


@pytest.mark.parametrize("extra_dict", [{"grantActivity": "2"}, {"depositsWithdrawals": "NaN"}, {"mtm": "11"}])
def test_unclassified_nonfinite_or_nonreconciling_components_withhold_pnl(extra_dict):
    result_dict = report_dict([nav_attributes_dict(**extra_dict)])
    assert result_dict["pnl_float"] is None
    assert result_dict["twr_float"] == pytest.approx(.01)


def test_linking_adjustment_is_separate_from_capital_and_profit():
    result_dict = report_dict([nav_attributes_dict(closing_str="1510", linkingAdjustments="500")])
    assert result_dict["linking_adjustment_float"] == 500
    assert result_dict["capital_movement_float"] == 0
    assert result_dict["pnl_float"] == 10


def test_two_accounts_have_dollar_bridge_but_no_invented_combined_twr():
    result_dict = report_dict([
        nav_attributes_dict(), nav_attributes_dict("U_TEST_B", opening_str="10000", closing_str="10050", twr_str=".5", mtm="50"),
    ], config_dict=client_config_dict(second_bool=True))
    assert result_dict["opening_nav_float"] == 11000
    assert result_dict["closing_nav_float"] == 11060
    assert result_dict["pnl_float"] == 60
    assert result_dict["twr_float"] is None
    assert [strategy_dict["twr_float"] for strategy_dict in result_dict["strategy_list"]] == [.01, .005]


def test_strategy_entry_is_scope_capital_not_profit():
    config_dict = client_config_dict(second_bool=True)
    config_dict["accounts"][1]["effective_from"] = "2026-09-02"
    result_dict = report_dict([
        nav_attributes_dict(), nav_attributes_dict(date_str="2026-09-02", opening_str="1010", closing_str="1020", twr_str=".990099"),
        nav_attributes_dict("U_TEST_B", "2026-09-02", "500", "505", "1", mtm="5"),
    ], config_dict=config_dict, to_str="2026-09-02")
    assert result_dict["opening_nav_float"] == 1000
    assert result_dict["closing_nav_float"] == 1525
    assert result_dict["scope_movement_float"] == 500
    assert result_dict["pnl_float"] == 25
    assert result_dict["twr_float"] is None


def test_strategy_retirement_keeps_historical_profit_and_removes_scope_nav():
    config_dict = client_config_dict(second_bool=True)
    config_dict["accounts"][1]["effective_to"] = "2026-09-01"
    result_dict = report_dict([
        nav_attributes_dict(), nav_attributes_dict(date_str="2026-09-02", opening_str="1010", closing_str="1020"),
        nav_attributes_dict("U_TEST_B", "2026-09-01", "500", "505", "1", mtm="5"),
    ], config_dict=config_dict, to_str="2026-09-02")
    assert result_dict["opening_nav_float"] == 1500
    assert result_dict["closing_nav_float"] == 1020
    assert result_dict["scope_movement_float"] == -505
    assert result_dict["pnl_float"] == 25
    assert len(result_dict["strategy_list"]) == 2


def test_equal_opposite_account_transfers_do_not_prove_consolidated_twr():
    result_dict = report_dict([
        nav_attributes_dict(closing_str="910", internalCashTransfers="-100"),
        nav_attributes_dict("U_TEST_B", closing_str="1110", internalCashTransfers="100"),
    ], config_dict=client_config_dict(second_bool=True))
    assert result_dict["capital_movement_float"] == 0
    assert result_dict["pnl_float"] == 20
    assert result_dict["twr_float"] is None


def test_missing_middle_or_last_session_does_not_silently_shorten_period():
    result_dict = report_dict([nav_attributes_dict()], to_str="2026-09-03")
    assert result_dict["closing_date_str"] == "2026-09-03"
    assert result_dict["closing_nav_float"] is None
    assert result_dict["pnl_float"] is None
    assert result_dict["twr_float"] is None
    assert result_dict["strategy_list"][0]["expected_day_count_int"] == 3


def test_nav_continuity_break_is_not_hidden_by_twr_linking():
    result_dict = report_dict([
        nav_attributes_dict(), nav_attributes_dict(date_str="2026-09-02", opening_str="1110", closing_str="1120"),
    ], to_str="2026-09-02")
    assert result_dict["pnl_float"] is None
    assert "continuity" in " ".join(result_dict["issue_list"])


def test_non_session_source_activity_is_preserved_not_discarded():
    attribute_list = [nav_attributes_dict(date_str=f"2026-09-0{day_int}", opening_str=str(990 + 10 * day_int), closing_str=str(1000 + 10 * day_int)) for day_int in range(1, 6)]
    result_dict = report_dict(attribute_list, to_str="2026-09-07")
    assert result_dict["closing_date_str"] == "2026-09-05"  # Sat actual; Mon Labor Day.
    assert result_dict["pnl_float"] == 50
    assert result_dict["strategy_list"][0]["observed_day_count_int"] == 5


def test_refresh_hash_stable_and_source_correction_changes_hash():
    config_dict = client_config_dict()
    source_obj = snapshot_obj([nav_attributes_dict()])
    first_dict = build_client_report_dict(config_dict, source_obj, from_date_str="2026-09-01", to_date_str="2026-09-01", as_of_ts=AS_OF_TS)
    refreshed_dict = build_client_report_dict(config_dict, source_obj, from_date_str="2026-09-01", to_date_str="2026-09-01", as_of_ts=AS_OF_TS.replace(minute=1))
    revised_dict = report_dict([nav_attributes_dict(closing_str="1011", mtm="11")])
    assert first_dict["report_hash_str"] == refreshed_dict["report_hash_str"]
    assert first_dict["report_hash_str"] != revised_dict["report_hash_str"]


def test_other_client_rows_never_enter_projection():
    result_dict = report_dict([nav_attributes_dict(), nav_attributes_dict("U_TEST_B")])
    assert result_dict["closing_nav_float"] == 1010
    assert len(result_dict["strategy_list"]) == 1


@pytest.mark.parametrize("mutation_fn", [
    lambda client_dict: client_dict.update(base_currency="EUR"),
    lambda client_dict: client_dict["accounts"].append(deepcopy(client_dict["accounts"][0])),
    lambda client_dict: client_dict["accounts"][0].update(effective_from="2026-08-01"),
    lambda client_dict: client_dict["accounts"][0].update(effective_to="2026-08-01"),
    lambda client_dict: client_dict["nav_bridge"].update(economic_fields=["endingValue"]),
    lambda client_dict: client_dict["nav_bridge"].update(informational_fields=["mtm"]),
])
def test_invalid_scope_or_overlapping_bridge_rejected(mutation_fn):
    config_dict = client_config_dict()
    mutation_fn(config_dict)
    with pytest.raises(ClientReportingError):
        validate_client_registry_dict({"schema_version": 1, "clients": [config_dict]})


def test_cross_client_account_overlap_rejected():
    first_dict = client_config_dict()
    second_dict = client_config_dict()
    second_dict["client_id"] = "other"
    with pytest.raises(ClientReportingError, match="Overlapping account"):
        validate_client_registry_dict({"schema_version": 1, "clients": [first_dict, second_dict]})


def insert_import(db_path_obj, attribute_list, from_str="2026-09-01", to_str="2026-09-02"):
    raw_xml_str = xml_text_str(attribute_list)
    checksum_str = hashlib.sha256(raw_xml_str.encode()).hexdigest()
    with sqlite3.connect(db_path_obj) as connection_obj:
        connection_obj.execute(
            "INSERT INTO flex_import(imported_timestamp_str, request_from_date_str, request_to_date_str, query_name_str, checksum_str, raw_xml_str) VALUES (?, ?, ?, ?, ?, ?)",
            (AS_OF_TS.isoformat(), from_str, to_str, "TEST_NAV", checksum_str, raw_xml_str),
        )


def test_raw_reader_ignores_operational_clipping_and_is_read_only(tmp_path):
    db_path_obj = tmp_path / "performance.sqlite3"
    PerformanceStore(str(db_path_obj)).initialize()
    insert_import(db_path_obj, [nav_attributes_dict()])
    before_bytes = db_path_obj.read_bytes()
    before_mtime_int = db_path_obj.stat().st_mtime_ns
    source_obj = load_broker_reporting_snapshot(str(db_path_obj), allowed_account_set={"U_TEST_A"}, query_name_str="TEST_NAV")
    assert len(source_obj.row_tuple) == 1  # normalized daily_performance is empty.
    assert db_path_obj.read_bytes() == before_bytes
    assert db_path_obj.stat().st_mtime_ns == before_mtime_int
    assert list(tmp_path.iterdir()) == [db_path_obj]


def test_corrected_range_omission_does_not_resurrect_older_fact(tmp_path):
    db_path_obj = tmp_path / "performance.sqlite3"
    PerformanceStore(str(db_path_obj)).initialize()
    insert_import(db_path_obj, [nav_attributes_dict(), nav_attributes_dict(date_str="2026-09-02", opening_str="1010", closing_str="1020")])
    insert_import(db_path_obj, [nav_attributes_dict(date_str="2026-09-02", opening_str="1010", closing_str="1021", mtm="11")])
    source_obj = load_broker_reporting_snapshot(str(db_path_obj), allowed_account_set={"U_TEST_A"}, query_name_str="TEST_NAV")
    assert [row_obj.market_date_str for row_obj in source_obj.row_tuple] == ["2026-09-02"]
    assert source_obj.row_tuple[0].source_import_id_int == 2


def test_missing_database_never_created(tmp_path):
    db_path_obj = tmp_path / "missing.sqlite3"
    source_obj = load_broker_reporting_snapshot(str(db_path_obj), allowed_account_set={"U_TEST_A"}, query_name_str="TEST_NAV")
    assert source_obj.unavailable_reason_str
    assert not db_path_obj.exists()


def test_bad_checksum_rejected():
    with pytest.raises(ClientReportingError, match="checksum"):
        parse_broker_nav_import(xml_text_str([nav_attributes_dict()]), allowed_account_set={"U_TEST_A"}, query_name_str="TEST_NAV", source_import_id_int=1, source_checksum_str="bad")


def test_duplicate_broker_day_rejected():
    with pytest.raises(ClientReportingError, match="Duplicate"):
        snapshot_obj([nav_attributes_dict(), nav_attributes_dict()])


def test_same_account_strategy_boundary_cannot_hide_nav_jump():
    config_dict = client_config_dict()
    config_dict["accounts"][0]["effective_to"] = "2026-09-01"
    next_dict = deepcopy(config_dict["accounts"][0])
    next_dict.update(pod_id="successor", display_name="Successor", effective_from="2026-09-02", effective_to=None)
    config_dict["accounts"].append(next_dict)
    result_dict = report_dict([
        nav_attributes_dict(), nav_attributes_dict(date_str="2026-09-02", opening_str="2000", closing_str="2010"),
    ], config_dict=config_dict, to_str="2026-09-02")
    assert result_dict["pnl_float"] is None
    assert result_dict["status_str"] == "draft"
    assert all(daily_dict["pnl_float"] is None for daily_dict in result_dict["daily_book_list"])


@pytest.mark.parametrize("malformed_obj", [None, [], "bad", False, 0])
@pytest.mark.parametrize("field_str", ["client", "account", "bridge", "effective_to"])
def test_malformed_registry_members_raise_domain_error(field_str, malformed_obj):
    if malformed_obj is None and field_str in {"bridge", "effective_to"}:
        return  # Null is the explicitly supported absent value.
    config_dict = client_config_dict()
    registry_dict = {"schema_version": 1, "clients": [config_dict]}
    if field_str == "client":
        registry_dict["clients"] = [malformed_obj]
    elif field_str == "account":
        config_dict["accounts"] = [malformed_obj]
    elif field_str == "bridge":
        config_dict["nav_bridge"] = malformed_obj
    else:
        config_dict["accounts"][0]["effective_to"] = malformed_obj
    with pytest.raises(ClientReportingError):
        validate_client_registry_dict(registry_dict)


def test_boolean_schema_is_not_version_one():
    with pytest.raises(ClientReportingError):
        validate_client_registry_dict({"schema_version": True, "clients": [client_config_dict()]})


def test_report_detaches_provenance_and_excludes_unscoped_sync_detail():
    source_obj = snapshot_obj([nav_attributes_dict()])
    source_obj.import_tuple[0]["private_column"] = "another client"
    source_obj = BrokerReportingSnapshot(source_obj.row_tuple, source_obj.import_tuple, {"detail_str": "another client token=secret"})
    result_dict = build_client_report_dict(client_config_dict(), source_obj, from_date_str="2026-09-01", to_date_str="2026-09-01", as_of_ts=AS_OF_TS)
    result_dict["source_list"][0]["checksum_str"] = "changed"
    assert source_obj.import_tuple[0]["checksum_str"] != "changed"
    assert "another client" not in str(result_dict)
    assert "latest_sync_attempt_dict" not in result_dict


def test_empty_replacement_is_in_report_provenance(tmp_path):
    db_path_obj = tmp_path / "performance.sqlite3"
    PerformanceStore(str(db_path_obj)).initialize()
    insert_import(db_path_obj, [nav_attributes_dict()])
    insert_import(db_path_obj, [])
    source_obj = load_broker_reporting_snapshot(str(db_path_obj), allowed_account_set={"U_TEST_A"}, query_name_str="TEST_NAV")
    result_dict = build_client_report_dict(client_config_dict(), source_obj, from_date_str="2026-09-01", to_date_str="2026-09-02", as_of_ts=AS_OF_TS)
    assert not source_obj.row_tuple
    assert [source_dict["import_id_int"] for source_dict in result_dict["source_list"]] == [1, 2]
    assert result_dict["status_str"] == "draft"


def test_weekend_account_entry_cannot_disappear_from_scope():
    config_dict = client_config_dict(second_bool=True)
    config_dict["accounts"][1]["effective_from"] = "2026-09-05"
    attribute_list = [nav_attributes_dict(date_str=f"2026-09-0{day_int}", opening_str=str(990 + 10 * day_int), closing_str=str(1000 + 10 * day_int)) for day_int in range(1, 5)]
    result_dict = report_dict(attribute_list, config_dict=config_dict, to_str="2026-09-06")
    assert result_dict["status_str"] == "draft"
    assert result_dict["closing_nav_float"] is None
    assert result_dict["twr_float"] is None
    assert len(result_dict["strategy_list"]) == 2


def test_current_day_activity_flex_is_pending_even_when_row_exists():
    result_dict = build_client_report_dict(client_config_dict(), snapshot_obj([nav_attributes_dict()]), from_date_str="2026-09-01", to_date_str="2026-09-01", as_of_ts=AS_OF_TS.replace(day=1, hour=14))
    assert result_dict["status_str"] == "draft"
    assert result_dict["pnl_float"] is None
    assert result_dict["twr_float"] is None
    assert "D+1" in " ".join(result_dict["issue_list"])
    assert result_dict["strategy_list"][0]["twr_float"] is None
    assert result_dict["strategy_list"][0]["pnl_float"] is None
    assert result_dict["strategy_list"][0]["coverage_complete_bool"] is False


@pytest.mark.parametrize("return_list,expected_return_float,max_dd_float,current_dd_float", [
    ([-10], -.1, -.1, -.1), ([10, -20, 25], .1, -.2, 0), ([0, 0, 0], 0, 0, 0),
])
def test_account_return_index_and_drawdown_math(return_list, expected_return_float, max_dd_float, current_dd_float):
    from alpha.live.client_reporting import account_performance_dict
    source_obj = snapshot_obj([nav_attributes_dict(date_str=f"2026-09-0{index_int + 1}", twr_str=str(value_float)) for index_int, value_float in enumerate(return_list)])
    performance_dict = account_performance_dict(list(source_obj.row_tuple), complete_bool=True, from_date_str="2026-09-01", to_date_str="2026-09-03", session_date_set={"2026-09-01", "2026-09-02", "2026-09-03"})
    assert performance_dict["return_path_list"][-1]["cumulative_return_float"] == pytest.approx(expected_return_float)
    assert performance_dict["max_drawdown_float"] == pytest.approx(max_dd_float)
    assert performance_dict["current_drawdown_float"] == pytest.approx(current_dd_float)
    assert performance_dict["monthly_return_list"][0]["partial_bool"] is True


def test_another_accounts_weekend_activity_does_not_invalidate_account_return():
    config_dict = client_config_dict(second_bool=True)
    attribute_list = []
    for account_str in ("U_TEST_A", "U_TEST_B"):
        for day_int in range(1, 5):
            attribute_list.append(nav_attributes_dict(account_str=account_str, date_str=f"2026-09-0{day_int}", opening_str=str(990 + 10 * day_int), closing_str=str(1000 + 10 * day_int)))
    attribute_list.append(nav_attributes_dict(account_str="U_TEST_B", date_str="2026-09-05", opening_str="1040", closing_str="1050"))
    result_dict = report_dict(attribute_list, config_dict=config_dict, to_str="2026-09-05")
    assert result_dict["status_str"] == "draft"  # Strict client NAV coverage.
    assert all(strategy_dict["twr_float"] is not None for strategy_dict in result_dict["strategy_list"])
    assert result_dict["strategy_list"][1]["performance_dict"]["non_session_count_int"] == 1
    assert len(result_dict["strategy_list"][1]["performance_dict"]["return_path_list"]) == 6


def test_missing_account_day_withholds_risk_not_only_total_return():
    result_dict = report_dict([nav_attributes_dict()], to_str="2026-09-02")
    performance_dict = result_dict["strategy_list"][0]["performance_dict"]
    assert performance_dict["return_path_list"] == []
    assert performance_dict["max_drawdown_float"] is None
