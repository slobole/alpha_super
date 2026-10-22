"""Automatic import evidence survives promotions, duplicates and corrections."""

from contextlib import closing
import json
import sqlite3

import pytest

from alpha.live.client_reporting import ClientReportingError
from alpha.live.ibkr_performance import PerformanceStore, PodPerformanceBinding
from alpha.live.reporting_accounts import discover_other_account_list
from test_other_accounts_discovery import _binding_list, _statement_str, _store_path_str


def _discovered_list(database_path_str):
    return discover_other_account_list(database_path_str, query_name_str="ALPHA_DAILY_TWR",
        strategy_start_by_route_dict={"U100": "2026-08-07", "U900": "2026-08-10"})


@pytest.mark.parametrize("legacy_bool", [False, True])
def test_same_xml_promotion_and_later_correction_keep_original_roles(tmp_path, legacy_bool):
    database_path_str = _store_path_str(tmp_path / "performance.sqlite3",
        _statement_str("U900", [("20260807", 500, 500), ("20260810", 500, 500)]))
    with closing(sqlite3.connect(database_path_str)) as connection_obj, connection_obj:
        raw_xml_str = connection_obj.execute("SELECT raw_xml_str FROM flex_import").fetchone()[0]
        if legacy_bool:
            connection_obj.execute("ALTER TABLE flex_import DROP COLUMN strategy_account_routes_json_str")
    binding_list = [*_binding_list(), PodPerformanceBinding("U900", "pod_new", "2026-08-10", None, True)]
    store_obj = PerformanceStore(database_path_str)
    import_kwargs_dict = dict(query_name_str="ALPHA_DAILY_TWR", request_from_date_str="2026-08-07",
        request_to_date_str="2026-08-10", binding_obj_list=binding_list)
    assert not store_obj.replace_range(xml_text_str=raw_xml_str, **import_kwargs_dict)
    expected_list = [{"role": "reserve", "account_route": "U900", "display_name": "Other account …U900",
        "effective_from": "2026-08-07", "effective_to": "2026-08-09"}]
    assert _discovered_list(database_path_str) == expected_list
    # A new raw source removes all mutable daily rows of the first import.
    assert store_obj.replace_range(xml_text_str=raw_xml_str.replace('count="2"', 'count="02"'), **import_kwargs_dict)
    assert _discovered_list(database_path_str) == expected_list
    with closing(sqlite3.connect(database_path_str)) as connection_obj:
        role_list = [json.loads(row_tuple[0]) for row_tuple in connection_obj.execute(
            "SELECT strategy_account_routes_json_str FROM flex_import ORDER BY import_id_int")]
        assert role_list == [["U100"], ["U100", "U900"]]
        assert connection_obj.execute("SELECT DISTINCT account_route_str FROM pod_binding ORDER BY account_route_str").fetchall() == [
            ("U100",), ("U900",)]


@pytest.mark.parametrize("role_json_str", ["bad-json", "null", '"U100"', "[1]", '[""]'])
def test_invalid_import_role_evidence_fails_closed(tmp_path, role_json_str):
    database_path_str = _store_path_str(tmp_path / "performance.sqlite3",
        _statement_str("U900", [("20260807", 500, 500)]))
    with closing(sqlite3.connect(database_path_str)) as connection_obj, connection_obj:
        connection_obj.execute("UPDATE flex_import SET strategy_account_routes_json_str = ?", (role_json_str,))
    with pytest.raises(ClientReportingError, match="account-role evidence is invalid"):
        _discovered_list(database_path_str)


@pytest.mark.parametrize("field_str", ["startingValue", "endingValue", "twr"])
@pytest.mark.parametrize("value_str", ["", "NaN", "Infinity"])
def test_unreadable_required_facts_never_prove_an_empty_account(tmp_path, field_str, value_str):
    statement_str = _statement_str("U900", [("20260807", 0, 0)]).replace(f'{field_str}="0"', f'{field_str}="{value_str}"')
    database_path_str = _store_path_str(tmp_path / "performance.sqlite3", statement_str)
    assert _discovered_list(database_path_str)[0]["account_route"] == "U900"
