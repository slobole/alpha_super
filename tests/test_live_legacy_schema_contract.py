"""Pinned production schema and legacy decision behavior survive daily-only additions."""
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sqlite3
import subprocess

import pytest

from alpha.live import state_store_v2
from alpha.live.core5_adapter import CORE5_STRATEGY_IMPORT_STR
from alpha.live.models import BrokerOrderFill
from alpha.live.state_store_v2 import LiveStateStore
from test_live_mr_capsule_target_shares import _capsule_inputs


@pytest.fixture(scope="module")
def baseline_store_class():
    # The contract is the owner's pinned production source, not a reconstructed
    # schema fixture that could accidentally repeat a new implementation mistake.
    repository_path_obj = Path(__file__).resolve().parents[1]
    source_str = subprocess.check_output([
        "git", "-c", f"safe.directory={repository_path_obj.as_posix()}", "-c", "core.fsmonitor=false",
        "show", "54b417f:alpha/live/state_store_v2.py",
    ], cwd=repository_path_obj, text=True, encoding="utf-8")
    namespace_dict = {"__name__": "baseline_54b417f_state_store"}
    exec(compile(source_str, "54b417f:alpha/live/state_store_v2.py", "exec"), namespace_dict)
    return namespace_dict["LiveStateStore"]


def _schema_dict(database_path_obj):
    with sqlite3.connect(database_path_obj) as connection_obj:
        return {(row_tuple[0], row_tuple[1]): (row_tuple[2], row_tuple[3]) for row_tuple in connection_obj.execute(
            "SELECT type,name,tbl_name,sql FROM sqlite_master WHERE name NOT LIKE 'sqlite_%' ORDER BY type,name")}


def _table_rows(database_path_obj, table_str):
    with sqlite3.connect(database_path_obj) as connection_obj:
        return connection_obj.execute(f'SELECT * FROM "{table_str}" ORDER BY rowid').fetchall()


def test_copied_production_database_keeps_every_existing_schema_and_row(tmp_path, baseline_store_class):
    original_path_obj, copy_path_obj = tmp_path / "production-copy.sqlite3", tmp_path / "candidate-copy.sqlite3"
    baseline_obj = baseline_store_class(str(original_path_obj))
    release_obj, decision_obj, _, _ = _capsule_inputs()
    baseline_obj.upsert_release(release_obj)
    decision_obj = baseline_obj.insert_decision_plan(replace(decision_obj, target_share_map_dict={}))
    baseline_obj.upsert_vplan_fill_list([BrokerOrderFill(
        broker_order_id_str="order-1", account_route_str=release_obj.account_route_str,
        asset_str="AAPL", fill_amount_float=5.0, fill_price_float=100.0,
        fill_timestamp_ts=decision_obj.target_execution_timestamp_ts,
        raw_payload_dict={"exec_id_str": "exec-1"},
        decision_plan_id_int=decision_obj.decision_plan_id_int, vplan_id_int=13)])
    baseline_schema_dict = _schema_dict(original_path_obj)
    baseline_row_dict = {name_str: _table_rows(original_path_obj, name_str)
        for (kind_str, name_str) in baseline_schema_dict if kind_str == "table"}
    shutil.copyfile(original_path_obj, copy_path_obj)
    for _restart_int in range(2):
        store_obj = LiveStateStore(str(copy_path_obj))
        candidate_schema_dict = _schema_dict(copy_path_obj)
        assert {key_tuple: candidate_schema_dict[key_tuple] for key_tuple in baseline_schema_dict} == baseline_schema_dict
        assert {name_str: _table_rows(copy_path_obj, name_str) for name_str in baseline_row_dict} == baseline_row_dict
        assert all(table_str not in baseline_row_dict for key_tuple, (table_str, _) in candidate_schema_dict.items()
            if key_tuple not in baseline_schema_dict)
        assert store_obj.get_decision_plan_by_id(decision_obj.decision_plan_id_int) == decision_obj
    assert _schema_dict(original_path_obj) == baseline_schema_dict
    assert {name_str: _table_rows(original_path_obj, name_str) for name_str in baseline_row_dict} == baseline_row_dict


@pytest.mark.parametrize("strategy_str", [
    "strategies.momentum.strategy_mo_atr_normalized_ndx:AtrNormalizedNdxStrategy",
    "strategies.taa_df.strategy_taa_df_btal_fallback_tqqq_vix_cash",
])
def test_monthly_insert_payload_and_uniqueness_match_pinned_production(tmp_path, baseline_store_class, monkeypatch, strategy_str):
    release_obj, decision_obj, _, _ = _capsule_inputs()
    release_obj = replace(release_obj, strategy_import_str=strategy_str)
    decision_obj = replace(decision_obj, target_share_map_dict={}, snapshot_metadata_dict={"strategy_import_str": strategy_str})
    fixed_timestamp_ts = datetime(2026, 10, 9, tzinfo=timezone.utc)
    monkeypatch.setattr(state_store_v2, "_utc_now_ts", lambda: fixed_timestamp_ts)
    monkeypatch.setitem(baseline_store_class.insert_decision_plan.__globals__, "_utc_now_ts", lambda: fixed_timestamp_ts)
    database_path_list = [tmp_path / "baseline.sqlite3", tmp_path / "candidate.sqlite3"]
    restored_list = []
    for store_class, database_path_obj in zip((baseline_store_class, LiveStateStore), database_path_list):
        store_obj = store_class(str(database_path_obj))
        store_obj.upsert_release(release_obj)
        stored_obj = store_obj.insert_decision_plan(decision_obj)
        restored_list.append(store_obj.get_decision_plan_by_id(stored_obj.decision_plan_id_int))
        with pytest.raises(sqlite3.IntegrityError, match="UNIQUE constraint failed"):
            store_obj.insert_decision_plan(decision_obj)
    assert _table_rows(database_path_list[0], "decision_plan") == _table_rows(database_path_list[1], "decision_plan")
    assert restored_list[0] == restored_list[1]


@pytest.mark.parametrize("strategy_str", [None, CORE5_STRATEGY_IMPORT_STR])
def test_daily_targets_roundtrip_in_single_atomic_metadata_insert(tmp_path, strategy_str):
    release_obj, decision_obj, _, _ = _capsule_inputs()
    if strategy_str is not None:
        release_obj = replace(release_obj, strategy_import_str=strategy_str)
    decision_obj = replace(decision_obj, snapshot_metadata_dict={**decision_obj.snapshot_metadata_dict,
        "daily_target_strategy_import_str": "untrusted caller marker"})
    database_path_obj = tmp_path / "daily.sqlite3"
    store_obj = LiveStateStore(str(database_path_obj))
    store_obj.upsert_release(release_obj)
    stored_obj = store_obj.insert_decision_plan(decision_obj)
    restarted_obj = LiveStateStore(str(database_path_obj))
    restored_obj = restarted_obj.get_decision_plan_by_id(stored_obj.decision_plan_id_int)
    assert restored_obj == stored_obj
    assert restored_obj.target_share_map_dict == {"BIL": 400.0}
    assert restored_obj.snapshot_metadata_dict["daily_target_share_map_dict"] == {"BIL": 400.0}
    assert restored_obj.snapshot_metadata_dict["daily_target_strategy_import_str"] == release_obj.strategy_import_str
    assert "daily_target_share_map_dict" not in decision_obj.snapshot_metadata_dict
    with store_obj._connect() as connection_obj:
        assert "target_share_json_str" not in {row_obj["name"] for row_obj in connection_obj.execute("PRAGMA table_info(decision_plan)")}
        connection_obj.execute("CREATE TRIGGER fail_decision AFTER INSERT ON decision_plan BEGIN SELECT RAISE(ABORT,'injected crash'); END")
    next_decision_obj = replace(stored_obj, signal_timestamp_ts=datetime(2026, 10, 2, 20, tzinfo=timezone.utc), target_share_map_dict={})
    with pytest.raises(sqlite3.IntegrityError, match="injected crash"):
        store_obj.insert_decision_plan(next_decision_obj)
    assert len(_table_rows(database_path_obj, "decision_plan")) == 1
    with store_obj._connect() as connection_obj:
        connection_obj.execute("DROP TRIGGER fail_decision")
    cleared_obj = store_obj.insert_decision_plan(next_decision_obj)
    assert LiveStateStore(str(database_path_obj)).get_decision_plan_by_id(cleared_obj.decision_plan_id_int).target_share_map_dict == {}


@pytest.mark.parametrize("mutation_dict", [
    {"strategy_import_str": "strategies.taa_df.strategy_taa_df_btal_fallback_tqqq_vix_cash"},
    {"pod_id_str": "another-pod"}, {"account_route_str": "another-account"}, {"user_id_str": "another-user"},
])
def test_daily_targets_cannot_use_another_strategy_or_identity(tmp_path, mutation_dict):
    release_obj, decision_obj, _, _ = _capsule_inputs()
    store_obj = LiveStateStore(str(tmp_path / "guard.sqlite3"))
    store_obj.upsert_release(replace(release_obj, **mutation_dict))
    with pytest.raises(ValueError, match="matching CORE5/capsule release"):
        store_obj.insert_decision_plan(decision_obj)
    assert _table_rows(store_obj.db_path_str, "decision_plan") == []
