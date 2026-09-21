"""Synthetic Flex holdings: exact source/date identity, never live IO or marks."""

from contextlib import closing
from dataclasses import replace
from datetime import UTC, datetime
from decimal import Decimal
import hashlib
import sqlite3
import xml.etree.ElementTree as ElementTree

import pytest

from alpha.live.client_reporting import parse_broker_nav_import
from alpha.live.dashboard_v4 import pod_holdings


AS_OF_TS = datetime(2026, 9, 8, 15, tzinfo=UTC)


def _position_dict(**override_dict):
    return {"accountId": "U_TEST_A", "reportDate": "20260904", "currency": "USD", "assetCategory": "STK",
        "levelOfDetail": "SUMMARY", "multiplier": "1", "fxRateToBase": "1", "conid": "101", "symbol": "AAA",
        "position": "2", "markPrice": "100", "positionValue": "200", "percentOfNAV": "99", **override_dict}


def _source_tuple(position_list=None, *, section_bool=True):
    root_obj = ElementTree.Element("FlexQueryResponse", queryName="TEST_NAV")
    statements_obj = ElementTree.SubElement(root_obj, "FlexStatements", count="1")
    statement_obj = ElementTree.SubElement(statements_obj, "FlexStatement", accountId="U_TEST_A")
    ElementTree.SubElement(statement_obj, "AccountInformation", accountId="U_TEST_A", currency="USD")
    ElementTree.SubElement(statement_obj, "ChangeInNAV", accountId="U_TEST_A", currency="USD",
        fromDate="20260904", toDate="20260904", startingValue="1000", endingValue="1050", twr="5")
    if section_bool:
        section_obj = ElementTree.SubElement(statement_obj, "OpenPositions")
        for position_dict in ([_position_dict()] if position_list is None else position_list):
            ElementTree.SubElement(section_obj, "OpenPosition", **position_dict)
    raw_xml_str = ElementTree.tostring(root_obj, encoding="unicode")
    nav_row_obj = parse_broker_nav_import(raw_xml_str, allowed_account_set={"U_TEST_A"}, query_name_str="TEST_NAV",
        source_import_id_int=1, source_checksum_str=hashlib.sha256(raw_xml_str.encode()).hexdigest())[0]
    return raw_xml_str, nav_row_obj


def _parse_dict(source_tuple, **override_dict):
    return pod_holdings.parse_close_holdings_dict(*source_tuple,
        **{"query_name_str": "TEST_NAV", "as_of_ts": AS_OF_TS, **override_dict})


def _database_path(tmp_path, source_tuple):
    database_path_obj = tmp_path / "performance.sqlite3"
    with closing(sqlite3.connect(database_path_obj)) as connection_obj, connection_obj:
        connection_obj.execute("CREATE TABLE flex_import (import_id_int INTEGER PRIMARY KEY, imported_timestamp_str TEXT, "
            "request_from_date_str TEXT, request_to_date_str TEXT, query_name_str TEXT, checksum_str TEXT, raw_xml_str TEXT)")
        connection_obj.execute("INSERT INTO flex_import VALUES (1, ?, ?, ?, ?, ?, ?)",
            ("2026-09-05T12:00:00+00:00", "2026-09-04", "2026-09-04", "TEST_NAV", source_tuple[1].source_checksum_str, source_tuple[0]))
    return database_path_obj


def _load_dict(database_path_obj, nav_row_obj, **override_dict):
    return pod_holdings.load_close_holdings_dict(str(database_path_obj), nav_row_obj,
        **{"query_name_str": "TEST_NAV", "as_of_ts": AS_OF_TS, **override_dict})


def test_saved_close_values_and_shares_are_returned_without_asset_class_percent_of_nav():
    result_dict = _parse_dict(_source_tuple())
    assert result_dict == {"available_bool": True, "reason_str": "", "close_date_str": "2026-09-04",
        "position_list": [{"symbol_str": "AAA", "shares_float": 2.0, "value_float": 200.0}],
        "source_str": "IBKR Flex closing positions"}


def test_short_values_are_kept_signed_for_builder_to_handle():
    result_dict = _parse_dict(_source_tuple([_position_dict(position="-2", positionValue="-200")]))
    assert result_dict["position_list"] == [{"symbol_str": "AAA", "shares_float": -2.0, "value_float": -200.0}]


def test_missing_section_is_unknown_but_explicit_empty_section_is_available():
    missing_dict = _parse_dict(_source_tuple(section_bool=False))
    assert not missing_dict["available_bool"]
    assert missing_dict["reason_str"] == "The saved report has no closing positions."
    assert _parse_dict(_source_tuple([]))["available_bool"]
    assert _parse_dict(_source_tuple([]))["position_list"] == []


@pytest.mark.parametrize("override_dict", [
    {"accountId": "OTHER_ACCOUNT_SECRET"}, {"reportDate": "20260903"}, {"model": "MODEL"}, {"currency": "EUR"},
    {"assetCategory": "OPT"}, {"levelOfDetail": "LOT"}, {"levelOfDetail": ""}, {"multiplier": "100"},
    {"fxRateToBase": "1.1"}, {"conid": ""}, {"conid": "0"}, {"symbol": ""}, {"symbol": " AAA"},
    {"position": "NaN"}, {"position": "1e999"}, {"markPrice": "-1"}, {"positionValue": "Infinity"},
    {"positionValue": "200.02"}, {"positionValue": "-200"}])
def test_unsupported_ambiguous_or_invalid_rows_withhold_whole_snapshot(override_dict):
    result_dict = _parse_dict(_source_tuple([_position_dict(**override_dict)]))
    assert not result_dict["available_bool"]
    assert result_dict["position_list"] == []
    assert "SECRET" not in result_dict["reason_str"]


@pytest.mark.parametrize("field_str", ["position", "positionValue", "markPrice", "multiplier", "reportDate", "currency", "conid"])
def test_missing_required_fields_are_not_zero_or_inferred(field_str):
    position_dict = _position_dict()
    del position_dict[field_str]
    assert not _parse_dict(_source_tuple([position_dict]))["available_bool"]


@pytest.mark.parametrize("second_dict", [_position_dict(conid="102"), _position_dict(symbol="BBB")])
def test_duplicate_symbol_or_contract_is_not_aggregated(second_dict):
    assert not _parse_dict(_source_tuple([_position_dict(), second_dict]))["available_bool"]


@pytest.mark.parametrize("clock_ts", [datetime(2026, 9, 4, 23, tzinfo=UTC), datetime(2026, 9, 5, 2, tzinfo=UTC),
    datetime(2026, 9, 8, 12)])
def test_current_et_date_or_naive_clock_cannot_expose_finalized_holdings(clock_ts):
    assert not _parse_dict(_source_tuple(), as_of_ts=clock_ts)["available_bool"]


def test_same_checksum_does_not_allow_wrong_nav_account_date_or_values():
    raw_xml_str, nav_row_obj = _source_tuple()
    for override_dict in ({"account_route_str": "OTHER"}, {"market_date_str": "2026-09-03"},
            {"closing_nav_decimal": Decimal("9999")}, {"source_checksum_str": "bad"}):
        assert not _parse_dict((raw_xml_str, replace(nav_row_obj, **override_dict)))["available_bool"]
    assert not _parse_dict((raw_xml_str, nav_row_obj), query_name_str="OTHER")["available_bool"]


@pytest.mark.parametrize("raw_xml_str", ["<not-xml", '<!DOCTYPE a [<!ENTITY secret "data">]><FlexQueryResponse/>'])
def test_malformed_or_entity_xml_is_sanitized(raw_xml_str):
    _, nav_row_obj = _source_tuple()
    nav_row_obj = replace(nav_row_obj, source_checksum_str=hashlib.sha256(raw_xml_str.encode()).hexdigest())
    assert not _parse_dict((raw_xml_str, nav_row_obj))["available_bool"]


def test_other_accounts_cannot_add_positions_and_duplicate_selected_statements_fail():
    raw_xml_str, nav_row_obj = _source_tuple()
    root_obj = ElementTree.fromstring(raw_xml_str)
    statements_obj = root_obj.find("FlexStatements")
    foreign_obj = ElementTree.SubElement(statements_obj, "FlexStatement", accountId="OTHER")
    ElementTree.SubElement(ElementTree.SubElement(foreign_obj, "OpenPositions"), "OpenPosition", symbol="SECRET")
    raw_xml_str = ElementTree.tostring(root_obj, encoding="unicode")
    nav_row_obj = replace(nav_row_obj, source_checksum_str=hashlib.sha256(raw_xml_str.encode()).hexdigest())
    assert len(_parse_dict((raw_xml_str, nav_row_obj))["position_list"]) == 1
    statements_obj.append(ElementTree.fromstring(ElementTree.tostring(statements_obj[0], encoding="unicode")))
    raw_xml_str = ElementTree.tostring(root_obj, encoding="unicode")
    nav_row_obj = replace(nav_row_obj, source_checksum_str=hashlib.sha256(raw_xml_str.encode()).hexdigest())
    assert not _parse_dict((raw_xml_str, nav_row_obj))["available_bool"]


def test_same_account_close_positions_split_across_statements_are_ambiguous():
    raw_xml_str, nav_row_obj = _source_tuple()
    root_obj = ElementTree.fromstring(raw_xml_str)
    second_obj = ElementTree.SubElement(root_obj.find("FlexStatements"), "FlexStatement", accountId="U_TEST_A")
    ElementTree.SubElement(ElementTree.SubElement(second_obj, "OpenPositions"), "OpenPosition", **_position_dict(symbol="BBB", conid="102"))
    raw_xml_str = ElementTree.tostring(root_obj, encoding="unicode")
    nav_row_obj = replace(nav_row_obj, source_checksum_str=hashlib.sha256(raw_xml_str.encode()).hexdigest())
    assert not _parse_dict((raw_xml_str, nav_row_obj))["available_bool"]


def test_cached_results_are_detached_and_current_bytes_are_rechecked(monkeypatch):
    monkeypatch.setattr(pod_holdings, "CACHE_LIMIT_INT", 1)
    source_tuple = _source_tuple()
    result_dict = _parse_dict(source_tuple)
    result_dict["position_list"][0]["value_float"] = 9999
    assert _parse_dict(source_tuple)["position_list"][0]["value_float"] == 200
    assert not _parse_dict((source_tuple[0] + " ", source_tuple[1]))["available_bool"]
    _parse_dict(_source_tuple([_position_dict(symbol="BBB")]))
    assert len(pod_holdings._cache_dict) <= 1


def test_loader_reads_exact_source_without_modifying_database(tmp_path):
    source_tuple = _source_tuple()
    database_path_obj = _database_path(tmp_path, source_tuple)
    before_bytes = database_path_obj.read_bytes()
    before_int = database_path_obj.stat().st_mtime_ns
    assert _load_dict(database_path_obj, source_tuple[1]) == _parse_dict(source_tuple)
    assert database_path_obj.read_bytes() == before_bytes
    assert database_path_obj.stat().st_mtime_ns == before_int


def test_missing_database_is_not_created(tmp_path):
    database_path_obj = tmp_path / "missing.sqlite3"
    assert not _load_dict(database_path_obj, _source_tuple()[1])["available_bool"]
    assert not database_path_obj.exists()


@pytest.mark.parametrize("column_str,value_str", [
    ("checksum_str", "bad"), ("query_name_str", "OTHER"), ("request_from_date_str", "2026-09-05"),
    ("request_to_date_str", "2026-09-03"), ("imported_timestamp_str", "2027-01-01T00:00:00+00:00"),
    ("imported_timestamp_str", "2026-09-05T12:00:00")])
def test_loader_rechecks_import_identity_range_and_observation_time(tmp_path, column_str, value_str):
    source_tuple = _source_tuple()
    database_path_obj = _database_path(tmp_path, source_tuple)
    with closing(sqlite3.connect(database_path_obj)) as connection_obj, connection_obj:
        connection_obj.execute(f"UPDATE flex_import SET {column_str}=?", (value_str,))
    assert not _load_dict(database_path_obj, source_tuple[1])["available_bool"]


@pytest.mark.parametrize("from_str,to_str,expected_bool", [
    ("2026-09-04", "2026-09-04", False), ("2026-09-01", "2026-09-30", False),
    ("2026-09-05", "2026-09-06", True), ("bad", "2026-09-06", False)])
def test_newer_import_replacement_never_resurrects_cached_old_positions(tmp_path, from_str, to_str, expected_bool):
    source_tuple = _source_tuple()
    database_path_obj = _database_path(tmp_path, source_tuple)
    assert _load_dict(database_path_obj, source_tuple[1])["available_bool"]
    with closing(sqlite3.connect(database_path_obj)) as connection_obj, connection_obj:
        connection_obj.execute("INSERT INTO flex_import VALUES (2, ?, ?, ?, 'TEST_NAV', 'new', '<no-position-section/>')",
            ("2026-09-08T12:00:00+00:00", from_str, to_str))
    assert _load_dict(database_path_obj, source_tuple[1])["available_bool"] is expected_bool


def test_sql_caps_utf8_bytes_before_parser_even_with_embedded_nul(tmp_path, monkeypatch):
    source_tuple = _source_tuple()
    database_path_obj = _database_path(tmp_path, source_tuple)
    oversized_str = "\0" + "א" * 200
    with closing(sqlite3.connect(database_path_obj)) as connection_obj, connection_obj:
        connection_obj.execute("UPDATE flex_import SET raw_xml_str=?", (oversized_str,))
    monkeypatch.setattr(pod_holdings, "XML_BYTE_LIMIT_INT", 300)
    seen_list = []
    monkeypatch.setattr(pod_holdings, "parse_close_holdings_dict", lambda raw_obj, *args, **kwargs: seen_list.append(raw_obj) or {"available_bool": False})
    assert not _load_dict(database_path_obj, source_tuple[1])["available_bool"]
    assert seen_list == [None]


def test_position_count_limit_is_enforced(tmp_path, monkeypatch):
    monkeypatch.setattr(pod_holdings, "POSITION_LIMIT_INT", 1)
    source_tuple = _source_tuple([_position_dict(), _position_dict(symbol="BBB", conid="102")])
    assert not _load_dict(_database_path(tmp_path, source_tuple), source_tuple[1])["available_bool"]
