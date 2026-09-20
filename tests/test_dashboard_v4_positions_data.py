"""Saved broker positions preserve identity, quantities and read-only access."""

from dataclasses import replace
from datetime import timedelta
import json
import sqlite3

import pytest

from alpha.live.dashboard_v4.positions_data import POSITION_LIMIT_INT, load_positions_dict
from test_dashboard_v4_evidence import FILL_TS, NOW_TS, SUBMIT_TS, build_fixture_tuple, update_db
from test_dashboard_v4_pod_data import _reconcile


def _cache(target_obj, position_dict=None, *, timestamp_ts=FILL_TS):
    with sqlite3.connect(target_obj.db_path_str) as connection_obj:
        connection_obj.execute("""INSERT INTO broker_snapshot_cache
            (account_route_str,snapshot_timestamp_str,cash_float,total_value_float,net_liq_float,
             position_json_str,open_order_id_json_str,updated_timestamp_str)
            VALUES ('U111',?,9800,10000,10000,?,'[]',?)""",
            (timestamp_ts.isoformat(), json.dumps({"SPY": 2} if position_dict is None else position_dict), timestamp_ts.isoformat()))


def test_saved_cache_is_exact_read_only_broker_quantity_not_model_or_target(tmp_path):
    target_obj, _ = build_fixture_tuple(tmp_path)
    _cache(target_obj, {"SPY": 3.125, "SHORT": -4, "FLAT": 0})
    update_db(target_obj, "UPDATE vplan SET target_share_json_str='{\"SPY\":999}'")
    path_obj = tmp_path / "pod.sqlite3"
    before_bytes, before_mtime_int = path_obj.read_bytes(), path_obj.stat().st_mtime_ns
    result_dict = load_positions_dict(target_obj, as_of_ts=NOW_TS)
    assert result_dict["available_bool"] is True
    assert result_dict["position_map_dict"] == {"SPY": 3.125, "SHORT": -4, "FLAT": 0}
    assert result_dict["position_timestamp_str"] == FILL_TS.isoformat()
    assert result_dict["source_str"] == "broker_snapshot"
    assert result_dict["timestamp_basis_str"] == "observed"
    assert result_dict["account_route_str"] == "U111"
    assert path_obj.read_bytes() == before_bytes
    assert path_obj.stat().st_mtime_ns == before_mtime_int


def test_explicit_empty_broker_map_is_distinct_from_no_broker_evidence(tmp_path):
    target_obj, _ = build_fixture_tuple(tmp_path)
    assert load_positions_dict(target_obj, as_of_ts=NOW_TS)["available_bool"] is False
    _cache(target_obj, {})
    result_dict = load_positions_dict(target_obj, as_of_ts=NOW_TS)
    assert result_dict["available_bool"] is True
    assert result_dict["position_map_dict"] == {}


def test_model_pod_state_without_broker_observation_does_not_prove_positions(tmp_path):
    target_obj, _ = build_fixture_tuple(tmp_path)
    update_db(target_obj, """INSERT INTO pod_state
        (pod_id_str,user_id_str,account_route_str,position_json_str,cash_float,total_value_float,
         strategy_state_json_str,snapshot_stage_str,snapshot_source_str,updated_timestamp_str)
        VALUES ('pod','owner','U111','{"SPY":99}',10,100,'{}','post_execution','pod_state',?)""", (FILL_TS.isoformat(),))
    result_dict = load_positions_dict(target_obj, as_of_ts=NOW_TS)
    assert result_dict["available_bool"] is False
    assert result_dict["position_map_dict"] == {}


def test_newer_failed_reconciliation_still_exposes_actual_broker_not_model_map(tmp_path):
    target_obj, _ = build_fixture_tuple(tmp_path)
    _cache(target_obj, {"SPY": 8})
    _reconcile(target_obj, status_str="failed")
    update_db(target_obj, "UPDATE vplan_reconciliation_snapshot SET model_position_json_str='{\"SPY\":900}',broker_position_json_str='{\"SPY\":3}'")
    result_dict = load_positions_dict(target_obj, as_of_ts=NOW_TS)
    assert result_dict["available_bool"] is True
    assert result_dict["position_map_dict"] == {"SPY": 3}
    assert result_dict["source_str"] == "broker_reconciliation"
    assert result_dict["timestamp_basis_str"] == "recorded"


def test_newer_cache_supersedes_reconciliation(tmp_path):
    target_obj, _ = build_fixture_tuple(tmp_path)
    _reconcile(target_obj)
    _cache(target_obj, {"SPY": 9}, timestamp_ts=NOW_TS)
    assert load_positions_dict(target_obj, as_of_ts=NOW_TS)["position_map_dict"] == {"SPY": 9}


@pytest.mark.parametrize("json_str", ["", "null", "[]", "{broken}", '{"SPY":2,"SPY":3}',
    '{"SPY":NaN}', '{"SPY":Infinity}', '{"SPY":true}', '{"SPY":"2"}', '{" SPY":2}', '{"":2}',
    json.dumps({str(index_int): 1 for index_int in range(POSITION_LIMIT_INT + 1)}), " " * 262145],
    ids=["empty", "null", "list", "malformed", "duplicate", "nan", "infinite", "boolean", "string", "whitespace", "blank-symbol", "too-many", "too-large"])
def test_entire_invalid_map_is_unavailable_never_partially_filtered(tmp_path, json_str):
    target_obj, _ = build_fixture_tuple(tmp_path)
    _cache(target_obj)
    update_db(target_obj, "UPDATE broker_snapshot_cache SET position_json_str=?", (json_str,))
    result_dict = load_positions_dict(target_obj, as_of_ts=NOW_TS)
    assert result_dict["available_bool"] is False
    assert result_dict["position_map_dict"] == {}


@pytest.mark.parametrize("timestamp_str", ["bad", "2026-09-18T13:30:00", (NOW_TS + timedelta(seconds=1)).isoformat()])
def test_invalid_or_future_cache_time_cannot_fall_back_to_old_reconciliation(tmp_path, timestamp_str):
    target_obj, _ = build_fixture_tuple(tmp_path)
    _reconcile(target_obj)
    _cache(target_obj)
    update_db(target_obj, "UPDATE broker_snapshot_cache SET snapshot_timestamp_str=?", (timestamp_str,))
    assert load_positions_dict(target_obj, as_of_ts=NOW_TS)["available_bool"] is False


@pytest.mark.parametrize("sql_str", [
    "UPDATE live_release SET mode_str='paper'", "UPDATE live_release SET user_id_str='other'",
    "UPDATE live_release SET release_id_str='other'", "UPDATE live_release SET account_route_str='other'",
    "UPDATE vplan SET account_route_str='other'", "UPDATE decision_plan SET user_id_str='other'",
    "UPDATE vplan_reconciliation_snapshot SET decision_plan_id_int=999",
    "UPDATE vplan_reconciliation_snapshot SET vplan_id_int=999",
    "UPDATE vplan_reconciliation_snapshot SET broker_position_json_str='[]'",
])
def test_saved_release_or_reconciliation_identity_mismatch_is_unavailable(tmp_path, sql_str):
    target_obj, _ = build_fixture_tuple(tmp_path)
    _reconcile(target_obj)
    update_db(target_obj, sql_str)
    assert load_positions_dict(target_obj, as_of_ts=NOW_TS)["available_bool"] is False


def test_conflicting_same_time_positions_cannot_choose_a_convenient_source(tmp_path):
    target_obj, _ = build_fixture_tuple(tmp_path)
    _reconcile(target_obj, timestamp_ts=FILL_TS)
    _cache(target_obj, {"SPY": 19})
    assert load_positions_dict(target_obj, as_of_ts=NOW_TS)["available_bool"] is False


def test_reconcile_before_target_execution_is_unavailable(tmp_path):
    target_obj, _ = build_fixture_tuple(tmp_path)
    _reconcile(target_obj, timestamp_ts=SUBMIT_TS)
    assert load_positions_dict(target_obj, as_of_ts=NOW_TS)["available_bool"] is False


@pytest.mark.parametrize("change_dict", [{"mode_str": "paper"}, {"enabled_bool": False}, {"mode_str": "incubation"}])
def test_non_live_or_disabled_target_never_opens_database(tmp_path, monkeypatch, change_dict):
    target_obj, _ = build_fixture_tuple(tmp_path)
    target_obj = replace(target_obj, release_obj=replace(target_obj.release_obj, **change_dict))
    monkeypatch.setattr("alpha.live.dashboard_v4.positions_data.sqlite3.connect", lambda *args, **kwargs: pytest.fail("Opened excluded target"))
    assert load_positions_dict(target_obj, as_of_ts=NOW_TS)["available_bool"] is False


def test_missing_database_is_not_created(tmp_path):
    target_obj, _ = build_fixture_tuple(tmp_path)
    missing_path_obj = tmp_path / "missing.sqlite3"
    target_obj = replace(target_obj, db_path_str=str(missing_path_obj))
    assert load_positions_dict(target_obj, as_of_ts=NOW_TS)["available_bool"] is False
    assert not missing_path_obj.exists()
