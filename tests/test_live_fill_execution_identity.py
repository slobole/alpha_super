"""Broker execution identity survives retries, migration and equal-sized fills."""
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import json
import sqlite3

import pytest

from alpha.live.models import BrokerOrderFill
from alpha.live.state_store_v2 import LiveStateStore


FILL_TIMESTAMP_TS = datetime(2026, 10, 5, 13, 30, tzinfo=timezone.utc)


def make_fill(execution_id_str=None, **override_dict):
    return replace(BrokerOrderFill(
        broker_order_id_str="order-1", decision_plan_id_int=11, vplan_id_int=13,
        account_route_str="DU_TEST", asset_str="AAPL", fill_amount_float=5.0,
        fill_price_float=100.0, fill_timestamp_ts=FILL_TIMESTAMP_TS,
        raw_payload_dict={} if execution_id_str is None else {"exec_id_str": execution_id_str},
    ), **override_dict)


@pytest.mark.parametrize("amount_float", [5.0, -5.0])
def test_identical_executions_are_distinct_and_replays_are_idempotent(tmp_path, amount_float):
    database_path_str = str(tmp_path / "fills.sqlite3")
    store_obj = LiveStateStore(database_path_str)
    fill_list = [make_fill(execution_id_str, fill_amount_float=amount_float)
                 for execution_id_str in ("exec-1", "exec-2")]
    store_obj.upsert_vplan_fill_list(fill_list)
    # A process restart must preserve the ID index, not reinstate tuple uniqueness.
    store_obj = LiveStateStore(database_path_str)
    store_obj.upsert_vplan_fill_list(fill_list * 2)
    fill_row_list = store_obj.get_fill_row_dict_list_for_vplan(13)
    assert len(fill_row_list) == 2
    assert sum(row_dict["fill_amount_float"] for row_dict in fill_row_list) == 2 * amount_float
    with store_obj._connect() as connection_obj:
        assert [row_obj[0] for row_obj in connection_obj.execute(
            "SELECT broker_execution_id_str FROM vplan_fill ORDER BY fill_record_id_int"
        )] == ["exec-1", "exec-2"]


def test_execution_replay_enriches_open_price_without_making_another_fill(tmp_path):
    store_obj = LiveStateStore(str(tmp_path / "fills.sqlite3"))
    fill_obj = make_fill("exec-1")
    store_obj.upsert_vplan_fill_list([fill_obj])
    enriched_fill_obj = replace(fill_obj, official_open_price_float=99.0, open_price_source_str="official")
    store_obj.upsert_vplan_fill_list([enriched_fill_obj, fill_obj])
    fill_row_dict, = store_obj.get_fill_row_dict_list_for_vplan(13)
    assert fill_row_dict["official_open_price_float"] == 99.0
    assert fill_row_dict["open_price_source_str"] == "official"


@pytest.mark.parametrize("override_dict", [
    {"broker_order_id_str": "different-order"}, {"fill_amount_float": 6.0},
    {"fill_price_float": 101.0}, {"fill_timestamp_ts": FILL_TIMESTAMP_TS + timedelta(seconds=1)},
])
def test_execution_id_conflicting_order_or_economics_fails_without_overwrite(tmp_path, override_dict):
    store_obj = LiveStateStore(str(tmp_path / "fills.sqlite3"))
    original_obj = make_fill("exec-1")
    store_obj.upsert_vplan_fill_list([original_obj])
    with pytest.raises(ValueError, match="order or execution economics"):
        store_obj.upsert_vplan_fill_list([replace(original_obj, **override_dict,
            raw_payload_dict={"exec_id_str": "exec-1", "conflicting_observation_bool": True})])
    with store_obj._connect() as connection_obj:
        row_obj, = connection_obj.execute("SELECT * FROM vplan_fill").fetchall()
    assert row_obj["broker_order_id_str"] == original_obj.broker_order_id_str
    assert row_obj["fill_amount_float"] == original_obj.fill_amount_float
    assert row_obj["fill_price_float"] == original_obj.fill_price_float
    assert row_obj["fill_timestamp_str"] == original_obj.fill_timestamp_ts.isoformat()
    assert json.loads(row_obj["raw_payload_json_str"]) == original_obj.raw_payload_dict


def test_execution_replay_accepts_same_instant_in_another_timezone(tmp_path):
    store_obj = LiveStateStore(str(tmp_path / "fills.sqlite3"))
    original_obj = make_fill("exec-1")
    store_obj.upsert_vplan_fill_list([original_obj])
    store_obj.upsert_vplan_fill_list([replace(original_obj,
        fill_timestamp_ts=FILL_TIMESTAMP_TS.astimezone(timezone(timedelta(hours=-4))),
        official_open_price_float=99.0, open_price_source_str="official")])
    row_dict, = store_obj.get_fill_row_dict_list_for_vplan(13)
    assert row_dict["fill_timestamp_str"] == FILL_TIMESTAMP_TS.isoformat()
    assert row_dict["official_open_price_float"] == 99.0


@pytest.mark.parametrize("known_first_bool", [False, True])
def test_identified_and_legacy_equal_fill_observations_are_ambiguous(tmp_path, known_first_bool):
    store_obj = LiveStateStore(str(tmp_path / "fills.sqlite3"))
    first_obj, second_obj = (make_fill("exec-1"), make_fill()) if known_first_bool else (make_fill(), make_fill("exec-1"))
    store_obj.upsert_vplan_fill_list([first_obj])
    second_obj = replace(second_obj, fill_timestamp_ts=FILL_TIMESTAMP_TS.astimezone(timezone(timedelta(hours=-4))))
    with pytest.raises(ValueError, match="Ambiguous identified and legacy"):
        store_obj.upsert_vplan_fill_list([second_obj])
    with store_obj._connect() as connection_obj:
        row_obj, = connection_obj.execute("SELECT * FROM vplan_fill").fetchall()
    assert json.loads(row_obj["raw_payload_json_str"]) == first_obj.raw_payload_dict


@pytest.mark.parametrize("override_dict", [
    {"vplan_id_int": 14}, {"decision_plan_id_int": 12}, {"asset_str": "MSFT"},
])
def test_execution_cannot_be_reassigned_to_another_lineage(tmp_path, override_dict):
    store_obj = LiveStateStore(str(tmp_path / "fills.sqlite3"))
    store_obj.upsert_vplan_fill_list([make_fill("exec-1")])
    with pytest.raises(ValueError, match="lineage"):
        store_obj.upsert_vplan_fill_list([make_fill("exec-1", **override_dict)])
    with store_obj._connect() as connection_obj:
        fill_row_obj, = connection_obj.execute("SELECT * FROM vplan_fill").fetchall()
    assert (fill_row_obj["vplan_id_int"], fill_row_obj["decision_plan_id_int"], fill_row_obj["asset_str"]) == (13, 11, "AAPL")


def test_execution_identity_is_scoped_to_the_broker_account(tmp_path):
    store_obj = LiveStateStore(str(tmp_path / "fills.sqlite3"))
    store_obj.upsert_vplan_fill_list([
        make_fill("exec-1"), make_fill("exec-1", account_route_str="DU_OTHER", vplan_id_int=14),
    ])
    assert len(store_obj.get_fill_row_dict_list_for_vplan(13)) == 1
    assert len(store_obj.get_fill_row_dict_list_for_vplan(14)) == 1


@pytest.mark.parametrize("execution_id_str", [None, "", "   "])
def test_fills_without_broker_execution_id_keep_legacy_idempotency(tmp_path, execution_id_str):
    store_obj = LiveStateStore(str(tmp_path / "fills.sqlite3"))
    fill_obj = make_fill(execution_id_str)
    store_obj.upsert_vplan_fill_list([fill_obj, fill_obj])
    assert len(store_obj.get_fill_row_dict_list_for_vplan(13)) == 1


def create_legacy_fill_table(database_path_str, fill_list):
    with sqlite3.connect(database_path_str) as connection_obj:
        connection_obj.executescript("""
            CREATE TABLE vplan_fill (
                fill_record_id_int INTEGER PRIMARY KEY AUTOINCREMENT,
                broker_order_id_str TEXT NOT NULL, decision_plan_id_int INTEGER,
                vplan_id_int INTEGER NOT NULL, account_route_str TEXT NOT NULL,
                asset_str TEXT NOT NULL, fill_amount_float REAL NOT NULL,
                fill_price_float REAL NOT NULL, fill_timestamp_str TEXT NOT NULL,
                raw_payload_json_str TEXT NOT NULL
            );
            CREATE UNIQUE INDEX vplan_fill_unique_event_idx ON vplan_fill
                (vplan_id_int, broker_order_id_str, fill_timestamp_str, fill_amount_float, fill_price_float);
        """)
        for fill_obj in fill_list:
            connection_obj.execute("""INSERT INTO vplan_fill
                (broker_order_id_str, decision_plan_id_int, vplan_id_int, account_route_str,
                 asset_str, fill_amount_float, fill_price_float, fill_timestamp_str, raw_payload_json_str)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""", (
                    fill_obj.broker_order_id_str, fill_obj.decision_plan_id_int, fill_obj.vplan_id_int,
                    fill_obj.account_route_str, fill_obj.asset_str, fill_obj.fill_amount_float,
                    fill_obj.fill_price_float, fill_obj.fill_timestamp_ts.isoformat(),
                    json.dumps(fill_obj.raw_payload_dict),
                ))


def test_old_schema_migration_preserves_rows_and_recovers_saved_execution_ids(tmp_path):
    database_path_str = str(tmp_path / "old.sqlite3")
    historical_fill_list = [make_fill("exec-1"), make_fill(fill_amount_float=-7.0)]
    create_legacy_fill_table(database_path_str, historical_fill_list)
    store_obj = LiveStateStore(database_path_str)
    with store_obj._connect() as connection_obj:
        fill_row_list = connection_obj.execute("SELECT * FROM vplan_fill ORDER BY fill_record_id_int").fetchall()
    assert [(row_obj["fill_record_id_int"], row_obj["broker_execution_id_str"]) for row_obj in fill_row_list] == [(1, "exec-1"), (2, None)]
    assert [json.loads(row_obj["raw_payload_json_str"]) for row_obj in fill_row_list] == [
        fill_obj.raw_payload_dict for fill_obj in historical_fill_list]
    store_obj.upsert_vplan_fill_list([*historical_fill_list, make_fill("exec-2")])
    assert len(store_obj.get_fill_row_dict_list_for_vplan(13)) == 3
    restarted_store_obj = LiveStateStore(database_path_str)
    restarted_store_obj.upsert_vplan_fill_list([*historical_fill_list, make_fill("exec-2")])
    assert len(restarted_store_obj.get_fill_row_dict_list_for_vplan(13)) == 3


def test_conflicting_legacy_execution_ids_fail_without_losing_or_reassigning_rows(tmp_path):
    database_path_str = str(tmp_path / "old_conflict.sqlite3")
    create_legacy_fill_table(database_path_str, [make_fill("exec-1"), make_fill("exec-1", vplan_id_int=14)])
    with pytest.raises(RuntimeError, match="Duplicate broker execution IDs"):
        LiveStateStore(database_path_str)
    with sqlite3.connect(database_path_str) as connection_obj:
        assert connection_obj.execute("SELECT vplan_id_int FROM vplan_fill ORDER BY fill_record_id_int").fetchall() == [(13,), (14,)]
        assert "broker_execution_id_str" not in {row_tuple[1] for row_tuple in connection_obj.execute("PRAGMA table_info(vplan_fill)")}
        index_sql_str, = connection_obj.execute("SELECT sql FROM sqlite_master WHERE name='vplan_fill_unique_event_idx'").fetchone()
        assert "WHERE" not in index_sql_str


def test_migration_rejects_mixed_identity_overlap_across_timezone_representations(tmp_path):
    database_path_str = str(tmp_path / "ambiguous.sqlite3")
    create_legacy_fill_table(database_path_str, [make_fill(), make_fill("exec-1",
        fill_timestamp_ts=FILL_TIMESTAMP_TS.astimezone(timezone(timedelta(hours=-4))))])
    with pytest.raises(RuntimeError, match="Ambiguous identified and legacy"):
        LiveStateStore(database_path_str)
    with sqlite3.connect(database_path_str) as connection_obj:
        assert connection_obj.execute("SELECT COUNT(*) FROM vplan_fill").fetchone()[0] == 2
        assert "broker_execution_id_str" not in {row_tuple[1] for row_tuple in connection_obj.execute("PRAGMA table_info(vplan_fill)")}
