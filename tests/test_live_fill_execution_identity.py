"""The owner-restored 54b417f tuple fill key and synthetic rollback boundary."""
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import json
import shutil
import sqlite3

import pytest

from alpha.live.models import BrokerOrderFill
from alpha.live.state_store_v2 import LiveStateStore


FILL_TIMESTAMP_TS = datetime(2026, 10, 5, 13, 30, tzinfo=timezone.utc)
BASELINE_FILL_SCHEMA_STR = """
CREATE TABLE vplan_fill (
    fill_record_id_int INTEGER PRIMARY KEY AUTOINCREMENT,
    broker_order_id_str TEXT NOT NULL,
    decision_plan_id_int INTEGER,
    vplan_id_int INTEGER NOT NULL,
    account_route_str TEXT NOT NULL,
    asset_str TEXT NOT NULL,
    fill_amount_float REAL NOT NULL,
    fill_price_float REAL NOT NULL,
    official_open_price_float REAL,
    open_price_source_str TEXT,
    fill_timestamp_str TEXT NOT NULL,
    raw_payload_json_str TEXT NOT NULL
);
CREATE UNIQUE INDEX vplan_fill_unique_event_idx ON vplan_fill (
    vplan_id_int, broker_order_id_str, fill_timestamp_str, fill_amount_float, fill_price_float
);
"""
BASELINE_FILL_COLUMN_LIST = [
    "fill_record_id_int", "broker_order_id_str", "decision_plan_id_int", "vplan_id_int",
    "account_route_str", "asset_str", "fill_amount_float", "fill_price_float",
    "official_open_price_float", "open_price_source_str", "fill_timestamp_str", "raw_payload_json_str",
]
BASELINE_KEY_COLUMN_LIST = [
    "vplan_id_int", "broker_order_id_str", "fill_timestamp_str", "fill_amount_float", "fill_price_float",
]


def make_fill(execution_id_str=None, **override_dict):
    return replace(BrokerOrderFill(
        broker_order_id_str="order-1", decision_plan_id_int=11, vplan_id_int=13,
        account_route_str="DU_TEST", asset_str="AAPL", fill_amount_float=5.0,
        fill_price_float=100.0, fill_timestamp_ts=FILL_TIMESTAMP_TS,
        raw_payload_dict={} if execution_id_str is None else {"exec_id_str": execution_id_str},
    ), **override_dict)


def _seed_baseline_fill_table(database_path_obj):
    with sqlite3.connect(database_path_obj) as connection_obj:
        connection_obj.executescript(BASELINE_FILL_SCHEMA_STR)
        connection_obj.execute("""INSERT INTO vplan_fill VALUES
            (7, 'order-1', 11, 13, 'DU_TEST', 'AAPL', 5, 100, 99, 'official', ?, ?)""",
            (FILL_TIMESTAMP_TS.isoformat(), json.dumps({"exec_id_str": "exec-1"})))


def _fill_rows(database_path_obj):
    with sqlite3.connect(database_path_obj) as connection_obj:
        return connection_obj.execute("SELECT * FROM vplan_fill ORDER BY fill_record_id_int").fetchall()


def _assert_baseline_fill_schema(database_path_obj):
    with sqlite3.connect(database_path_obj) as connection_obj:
        assert [row_tuple[1] for row_tuple in connection_obj.execute("PRAGMA table_info(vplan_fill)")] == BASELINE_FILL_COLUMN_LIST
        index_list = connection_obj.execute("PRAGMA index_list(vplan_fill)").fetchall()
        assert [(row_tuple[1], row_tuple[2], row_tuple[4]) for row_tuple in index_list] == [("vplan_fill_unique_event_idx", 1, 0)]
        assert [row_tuple[2] for row_tuple in connection_obj.execute("PRAGMA index_info(vplan_fill_unique_event_idx)")] == BASELINE_KEY_COLUMN_LIST


def test_fresh_database_has_exact_baseline_fill_columns_and_tuple_key(tmp_path):
    database_path_obj = tmp_path / "fresh.sqlite3"
    LiveStateStore(str(database_path_obj))
    _assert_baseline_fill_schema(database_path_obj)


def test_baseline_database_copy_preserves_rows_ids_and_old_reader_contract(tmp_path):
    original_path_obj, copy_path_obj = tmp_path / "baseline.sqlite3", tmp_path / "copy.sqlite3"
    _seed_baseline_fill_table(original_path_obj)
    original_row_list = _fill_rows(original_path_obj)
    shutil.copyfile(original_path_obj, copy_path_obj)
    store_obj = LiveStateStore(str(copy_path_obj))
    assert _fill_rows(copy_path_obj) == original_row_list
    _assert_baseline_fill_schema(copy_path_obj)
    store_obj.upsert_vplan_fill_list([make_fill("exec-2")])
    LiveStateStore(str(copy_path_obj))
    row_dict, = store_obj.get_fill_row_dict_list_for_vplan(13)
    assert row_dict == {
        "asset_str": "AAPL", "fill_amount_float": 5.0, "fill_price_float": 100.0,
        "official_open_price_float": 99.0, "open_price_source_str": "official",
        "fill_timestamp_str": FILL_TIMESTAMP_TS.isoformat(),
    }
    saved_row_tuple, = _fill_rows(copy_path_obj)
    assert saved_row_tuple[0] == 7
    assert json.loads(saved_row_tuple[-1]) == {"exec_id_str": "exec-2"}
    assert _fill_rows(original_path_obj) == original_row_list


@pytest.mark.parametrize("amount_float", [5.0, -5.0])
def test_equal_execution_tuples_keep_baseline_single_row_even_with_distinct_ids(tmp_path, amount_float):
    database_path_obj = tmp_path / "equal.sqlite3"
    store_obj = LiveStateStore(str(database_path_obj))
    fill_list = [make_fill(execution_id_str, fill_amount_float=amount_float) for execution_id_str in ("exec-1", "exec-2")]
    store_obj.upsert_vplan_fill_list(fill_list)
    store_obj = LiveStateStore(str(database_path_obj))
    store_obj.upsert_vplan_fill_list(fill_list)
    fill_dict, = store_obj.get_fill_row_dict_list_for_vplan(13)
    assert fill_dict["fill_amount_float"] == amount_float
    assert json.loads(_fill_rows(database_path_obj)[0][-1]) == {"exec_id_str": "exec-2"}


@pytest.mark.parametrize("override_dict", [
    {"broker_order_id_str": "order-2"}, {"fill_amount_float": 6.0},
    {"fill_price_float": 101.0}, {"fill_timestamp_ts": FILL_TIMESTAMP_TS + timedelta(seconds=1)},
    {"vplan_id_int": 14},
])
def test_same_execution_id_does_not_replace_distinct_baseline_tuple(tmp_path, override_dict):
    database_path_obj = tmp_path / "distinct.sqlite3"
    store_obj = LiveStateStore(str(database_path_obj))
    store_obj.upsert_vplan_fill_list([make_fill("exec-1"), make_fill("exec-1", **override_dict)])
    assert len(_fill_rows(database_path_obj)) == 2


@pytest.mark.parametrize("execution_id_str", [None, "", "   "])
def test_no_execution_id_keeps_baseline_replay_and_open_price_enrichment(tmp_path, execution_id_str):
    store_obj = LiveStateStore(str(tmp_path / "legacy.sqlite3"))
    fill_obj = make_fill(execution_id_str)
    store_obj.upsert_vplan_fill_list([fill_obj, replace(fill_obj, official_open_price_float=99.0, open_price_source_str="official"), fill_obj])
    fill_dict, = store_obj.get_fill_row_dict_list_for_vplan(13)
    assert fill_dict["official_open_price_float"] == 99.0
    assert fill_dict["open_price_source_str"] == "official"


def test_external_transaction_still_rolls_fill_recording_back(tmp_path):
    database_path_obj = tmp_path / "atomic.sqlite3"
    store_obj = LiveStateStore(str(database_path_obj))
    with pytest.raises(RuntimeError, match="injected"):
        with store_obj._connect() as connection_obj:
            store_obj.upsert_vplan_fill_list([make_fill("exec-1")], connection_obj=connection_obj)
            raise RuntimeError("injected")
    assert _fill_rows(database_path_obj) == []


def test_pre_open_price_schema_only_gets_baseline_open_price_migration(tmp_path):
    database_path_obj = tmp_path / "old.sqlite3"
    with sqlite3.connect(database_path_obj) as connection_obj:
        connection_obj.executescript(BASELINE_FILL_SCHEMA_STR.replace(
            "    official_open_price_float REAL,\n    open_price_source_str TEXT,\n", ""))
    LiveStateStore(str(database_path_obj))
    with sqlite3.connect(database_path_obj) as connection_obj:
        column_list = [row_tuple[1] for row_tuple in connection_obj.execute("PRAGMA table_info(vplan_fill)")]
    assert set(column_list) == set(BASELINE_FILL_COLUMN_LIST)


def test_experimental_duplicate_copy_fails_without_discarding_either_fill(tmp_path):
    original_path_obj, copy_path_obj = tmp_path / "experimental.sqlite3", tmp_path / "copy.sqlite3"
    _seed_baseline_fill_table(original_path_obj)
    with sqlite3.connect(original_path_obj) as connection_obj:
        connection_obj.execute("ALTER TABLE vplan_fill ADD COLUMN broker_execution_id_str TEXT")
        connection_obj.execute("DROP INDEX vplan_fill_unique_event_idx")
        connection_obj.execute("CREATE UNIQUE INDEX vplan_fill_unique_execution_idx ON vplan_fill(account_route_str,broker_execution_id_str) WHERE broker_execution_id_str IS NOT NULL")
        connection_obj.execute("UPDATE vplan_fill SET broker_execution_id_str='exec-1'")
        connection_obj.execute("INSERT INTO vplan_fill SELECT 8,broker_order_id_str,decision_plan_id_int,vplan_id_int,account_route_str,asset_str,fill_amount_float,fill_price_float,official_open_price_float,open_price_source_str,fill_timestamp_str,?,? FROM vplan_fill",
            (json.dumps({"exec_id_str": "exec-2"}), "exec-2"))
    original_row_list = _fill_rows(original_path_obj)
    shutil.copyfile(original_path_obj, copy_path_obj)
    with pytest.raises(sqlite3.IntegrityError, match="UNIQUE constraint failed: vplan_fill"):
        LiveStateStore(str(copy_path_obj))
    assert _fill_rows(copy_path_obj) == original_row_list
    assert _fill_rows(original_path_obj) == original_row_list
