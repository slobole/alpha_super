"""Synthetic decode-cache parity: current bytes/SQL remain authoritative."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from copy import deepcopy
from hashlib import sha256
import json
import os
from pathlib import Path
import sqlite3
from unittest.mock import patch

import pandas as pd
import pytest

from alpha.live import client_benchmark, client_reporting
from alpha.live.ibkr_performance import PerformanceStore
from test_client_benchmark import write_source_dict
from test_client_reporting import (
    AS_OF_TS, client_config_dict, insert_import, nav_attributes_dict, xml_text_str,
)


@pytest.fixture(autouse=True)
def clear_decode_caches():
    client_benchmark._benchmark_cache_dict.clear()
    client_reporting._nav_cache_dict.clear()
    yield
    client_benchmark._benchmark_cache_dict.clear()
    client_reporting._nav_cache_dict.clear()


def parse_rows(attribute_list=None, **override_dict):
    xml_str = xml_text_str(attribute_list if attribute_list is not None else [nav_attributes_dict()])
    argument_dict = dict(allowed_account_set={"U_TEST_A"}, query_name_str="TEST_NAV",
                         source_import_id_int=1, source_checksum_str=sha256(xml_str.encode()).hexdigest())
    argument_dict.update(override_dict)
    return client_reporting.parse_broker_nav_import(xml_str, **argument_dict)


def test_benchmark_reuses_decode_but_revalidates_exact_current_bytes(tmp_path):
    config_dict = write_source_dict(tmp_path)
    price_path = Path(config_dict["snapshot_directory"]) / "prices.parquet"
    with patch.object(pd, "read_parquet", wraps=pd.read_parquet) as decode_mock:
        first_obj = client_benchmark.load_benchmark_snapshot(config_dict)
        assert first_obj.unavailable_reason_str is None
        assert client_benchmark.load_benchmark_snapshot(config_dict) == first_obj
        assert decode_mock.call_count == 1
        stat_obj = price_path.stat()
        original_bytes = price_path.read_bytes()
        price_path.write_bytes(b"X" + original_bytes[1:])
        os.utime(price_path, ns=(stat_obj.st_atime_ns, stat_obj.st_mtime_ns))
        assert client_benchmark.load_benchmark_snapshot(config_dict).unavailable_reason_str
        assert decode_mock.call_count == 1
        price_path.write_bytes(original_bytes)
        assert client_benchmark.load_benchmark_snapshot(config_dict) == first_obj
        price_path.unlink()
        assert client_benchmark.load_benchmark_snapshot(config_dict).unavailable_reason_str


def test_benchmark_revision_and_symbol_never_reuse_wrong_values(tmp_path):
    config_dict = write_source_dict(tmp_path)
    original_obj = client_benchmark.load_benchmark_snapshot(config_dict)
    assert client_benchmark.load_benchmark_snapshot(dict(config_dict, symbol="$SPXTR")).unavailable_reason_str
    directory_path = Path(config_dict["snapshot_directory"])
    price_path = directory_path / "prices.parquet"
    price_df = pd.read_parquet(price_path)
    price_df["Close"] = price_df["Close"] * 2
    price_df.to_parquet(price_path, index=False)
    manifest_path = directory_path / "manifest.json"
    manifest_dict = json.loads(manifest_path.read_text())
    manifest_dict["files"]["prices.parquet"]["sha256"] = sha256(price_path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest_dict))
    revised_obj = client_benchmark.load_benchmark_snapshot(config_dict)
    assert revised_obj.unavailable_reason_str is None
    assert revised_obj.close_tuple[0][1] == original_obj.close_tuple[0][1] * 2
    assert revised_obj.price_hash_str != original_obj.price_hash_str
    manifest_dict["snapshot_market_session_date_str"] = "2026-09-01"
    manifest_path.write_text(json.dumps(manifest_dict))
    assert client_benchmark.load_benchmark_snapshot(config_dict).unavailable_reason_str
    manifest_path.write_text("malformed")
    assert client_benchmark.load_benchmark_snapshot(config_dict).unavailable_reason_str


def test_flex_reuses_decode_and_detaches_mutable_attributes():
    with patch.object(client_reporting.ElementTree, "fromstring", wraps=client_reporting.ElementTree.fromstring) as decode_mock:
        first_list = parse_rows()
        original_list = deepcopy(first_list)
        first_list[0].attribute_dict["mtm"] = "999999"
        second_list = parse_rows()
        assert second_list == original_list
        second_list[0].attribute_dict.clear()
        assert parse_rows() == original_list
        assert decode_mock.call_count == 1


def test_flex_checksum_scope_query_and_import_identity_on_warm_cache():
    first_list = parse_rows()
    with pytest.raises(client_reporting.ClientReportingError, match="checksum"):
        parse_rows([nav_attributes_dict(closing_str="9999")], source_checksum_str=first_list[0].source_checksum_str)
    assert parse_rows(allowed_account_set={"U_TEST_B"}) == []
    with pytest.raises(client_reporting.ClientReportingError, match="identity"):
        parse_rows(query_name_str="OTHER")
    assert parse_rows(source_import_id_int=2)[0].source_import_id_int == 2
    assert parse_rows([nav_attributes_dict(closing_str="1011", mtm="11")])[0].closing_nav_decimal == 1011


def test_fresh_sql_revisions_empty_replacements_deletion_and_report_parity(tmp_path):
    database_path = tmp_path / "source.sqlite3"
    PerformanceStore(str(database_path)).initialize()
    insert_import(database_path, [nav_attributes_dict()])
    def load_source():
        return client_reporting.load_broker_reporting_snapshot(str(database_path), allowed_account_set={"U_TEST_A"}, query_name_str="TEST_NAV")
    def build_report(source_obj, as_of_ts=AS_OF_TS):
        return client_reporting.build_client_report_dict(client_config_dict(), source_obj,
            from_date_str="2026-09-01", to_date_str="2026-09-01", as_of_ts=as_of_ts)
    first_obj = load_source()
    report_dict = build_report(first_obj)
    assert build_report(load_source()) == report_dict
    client_reporting._nav_cache_dict.clear()
    assert build_report(load_source()) == report_dict
    assert build_report(load_source(), AS_OF_TS.replace(day=1))["pnl_float"] is None
    insert_import(database_path, [nav_attributes_dict(closing_str="1011", mtm="11")])
    assert build_report(load_source())["pnl_float"] == 11
    insert_import(database_path, [])
    assert load_source().row_tuple == ()
    assert len(load_source().import_tuple) == 3
    with closing(sqlite3.connect(database_path)) as connection_obj:
        with connection_obj:
            connection_obj.execute("DELETE FROM flex_import")
    assert load_source().unavailable_reason_str
    # insert_import's legacy test helper leaves its connection for GC on
    # Windows. Simulate a missing source without relying on deleting its handle.
    assert client_reporting.load_broker_reporting_snapshot(str(tmp_path / "missing.sqlite3"),
        allowed_account_set={"U_TEST_A"}, query_name_str="TEST_NAV").unavailable_reason_str


def test_cache_bounds_bypass_and_concurrent_access_preserve_results(tmp_path, monkeypatch):
    monkeypatch.setattr(client_reporting, "_NAV_CACHE_LIMIT_INT", 2)
    monkeypatch.setattr(client_benchmark, "_BENCHMARK_CACHE_LIMIT_INT", 1)
    config_dict = write_source_dict(tmp_path)
    expected_obj = client_benchmark.load_benchmark_snapshot(config_dict)
    with ThreadPoolExecutor(max_workers=4) as executor_obj:
        source_list = list(executor_obj.map(lambda ignored_int: client_benchmark.load_benchmark_snapshot(config_dict), range(12)))
        row_list = list(executor_obj.map(lambda import_int: parse_rows(source_import_id_int=import_int), range(20)))
    assert all(source_obj == expected_obj for source_obj in source_list)
    assert [item_list[0].source_import_id_int for item_list in row_list] == list(range(20))
    assert len(client_reporting._nav_cache_dict) <= 2
    client_benchmark._benchmark_cache_dict.clear()
    client_reporting._nav_cache_dict.clear()
    monkeypatch.setattr(client_benchmark, "_BENCHMARK_CACHE_ROW_LIMIT_INT", 0)
    monkeypatch.setattr(client_reporting, "_NAV_CACHE_XML_LIMIT_INT", 0)
    assert client_benchmark.load_benchmark_snapshot(config_dict) == expected_obj
    assert parse_rows()[0].closing_nav_decimal == 1010
    assert not client_benchmark._benchmark_cache_dict
    assert not client_reporting._nav_cache_dict
