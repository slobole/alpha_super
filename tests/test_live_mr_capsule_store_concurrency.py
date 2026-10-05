"""MR capsule state store on a real file database: schema upgrade and cross-connection submission races.

The in-memory race tests share one connection, so they cannot show that BEGIN IMMEDIATE serializes two processes.
These tests use separate connections to one SQLite file, as two scheduler workers on the VPS would.
"""
import json
import sqlite3
import threading
import time
from dataclasses import replace
from datetime import datetime, timezone

import pytest

from alpha.live.models import DecisionPlan, VPlan, VPlanRow
from alpha.live.mr_capsule_adapter import MR_CAPSULE_CONTRACT_STR
from alpha.live.state_store_v2 import LiveStateStore

IDENTITY_DICT = dict(release_id_str="capsule.v1", user_id_str="owner", pod_id_str="capsule", account_route_str="DU123")
SIGNAL_TS = datetime(2024, 1, 12, 21, tzinfo=timezone.utc)
EXECUTION_TS = datetime(2024, 1, 16, 14, 30, tzinfo=timezone.utc)
TIMING_DICT = dict(signal_timestamp_ts=SIGNAL_TS, submission_timestamp_ts=EXECUTION_TS,
                   target_execution_timestamp_ts=EXECUTION_TS, execution_policy_str="next_open_moo")


def _capsule_decision(target_share_map_dict=None, metadata_dict=None) -> DecisionPlan:
    return DecisionPlan(
        **IDENTITY_DICT, **TIMING_DICT, decision_base_position_map={}, strategy_state_dict={"trade_id_int": 3},
        snapshot_metadata_dict={"sizing_contract_str": MR_CAPSULE_CONTRACT_STR} if metadata_dict is None else metadata_dict,
        target_share_map_dict={"BIL": 10} if target_share_map_dict is None else target_share_map_dict,
    )


def _vplan_for(decision_obj: DecisionPlan) -> VPlan:
    return VPlan(
        **IDENTITY_DICT, **TIMING_DICT, decision_plan_id_int=decision_obj.decision_plan_id_int,
        broker_snapshot_timestamp_ts=EXECUTION_TS, live_reference_snapshot_timestamp_ts=EXECUTION_TS,
        live_price_source_str="synthetic", net_liq_float=10_000.0, available_funds_float=None,
        excess_liquidity_float=None, pod_budget_fraction_float=1.0, pod_budget_float=10_000.0,
        current_broker_position_map={}, live_reference_price_map={"BIL": 100.0},
        target_share_map={"BIL": 10.0}, order_delta_map={"BIL": 10.0},
        vplan_row_list=[VPlanRow(
            asset_str="BIL", current_share_float=0.0, target_share_float=10.0, order_delta_share_float=10.0,
            live_reference_price_float=100.0, estimated_target_notional_float=1000.0, broker_order_type_str="MOO",
        )],
    )


@pytest.fixture
def ready_cycle(tmp_path):
    db_path_str = str(tmp_path / "capsule_pod.sqlite3")
    store_obj = LiveStateStore(db_path_str)
    decision_obj = store_obj.insert_decision_plan(_capsule_decision())
    vplan_obj = store_obj.insert_vplan(_vplan_for(decision_obj))
    assert store_obj.get_decision_plan_by_id(decision_obj.decision_plan_id_int).status_str == "vplan_ready"
    return db_path_str, decision_obj, vplan_obj


def _status_pair(db_path_str: str, decision_obj: DecisionPlan, vplan_obj: VPlan) -> tuple[str, str, dict]:
    with sqlite3.connect(db_path_str) as connection_obj:
        decision_row = connection_obj.execute(
            "SELECT status_str, snapshot_metadata_json_str FROM decision_plan WHERE decision_plan_id_int = ?",
            (decision_obj.decision_plan_id_int,),
        ).fetchone()
        vplan_status_str = connection_obj.execute(
            "SELECT status_str FROM vplan WHERE vplan_id_int = ?", (vplan_obj.vplan_id_int,),
        ).fetchone()[0]
    return decision_row[0], vplan_status_str, json.loads(decision_row[1])


def test_upgraded_database_gains_the_target_share_column_before_any_write(tmp_path):
    """Every pod's decision insert writes target_share_json_str; an upgraded DB must get the column first."""
    db_path_str = str(tmp_path / "pre_capsule.sqlite3")
    LiveStateStore(db_path_str)
    with sqlite3.connect(db_path_str) as connection_obj:
        connection_obj.execute("ALTER TABLE decision_plan DROP COLUMN target_share_json_str")
    upgraded_store_obj = LiveStateStore(db_path_str)
    with sqlite3.connect(db_path_str) as connection_obj:
        column_list = [row for row in connection_obj.execute("PRAGMA table_info(decision_plan)") if row[1] == "target_share_json_str"]
    # (cid, name, type, notnull, default, pk): NOT NULL DEFAULT '{}' keeps old INSERTs valid after a rollback.
    assert len(column_list) == 1 and column_list[0][3] == 1 and column_list[0][4] == "'{}'"
    capsule_obj = upgraded_store_obj.insert_decision_plan(_capsule_decision(target_share_map_dict={"BIL": 400}))
    dv2_obj = upgraded_store_obj.insert_decision_plan(replace(
        _capsule_decision(target_share_map_dict={}, metadata_dict={}),
        pod_id_str="dv2_pod", release_id_str="dv2.v1", entry_target_weight_map_dict={"AAPL": 0.1},
    ))
    assert upgraded_store_obj.get_decision_plan_by_id(capsule_obj.decision_plan_id_int).target_share_map_dict == {"BIL": 400.0}
    restored_dv2_obj = upgraded_store_obj.get_decision_plan_by_id(dv2_obj.decision_plan_id_int)
    assert restored_dv2_obj.target_share_map_dict == {} and restored_dv2_obj.entry_target_weight_map_dict == {"AAPL": 0.1}


def test_abandonment_waits_for_an_in_flight_claim_and_never_overwrites_it(ready_cycle):
    db_path_str, decision_obj, vplan_obj = ready_cycle
    # Open worker B's store first: the constructor's schema statements also take the write lock.
    abandon_store_obj = LiveStateStore(db_path_str)
    # Worker A is inside its submission claim: it holds the write lock and has moved the VPlan to 'submitting'.
    claim_connection_obj = sqlite3.connect(db_path_str, isolation_level=None)
    claim_connection_obj.execute("BEGIN IMMEDIATE")
    claim_connection_obj.execute(
        "UPDATE vplan SET status_str = 'submitting' WHERE vplan_id_int = ? AND status_str = 'ready'", (vplan_obj.vplan_id_int,),
    )
    result_dict = {}
    abandon_thread_obj = threading.Thread(target=lambda: result_dict.update(
        abandoned_bool=abandon_store_obj.abandon_unsubmitted_mr_capsule_cycle(decision_obj.decision_plan_id_int, "blocked")
    ))
    abandon_thread_obj.start()
    time.sleep(0.4)
    assert abandon_thread_obj.is_alive(), "worker B must wait for worker A's lock, not read the uncommitted 'ready'"
    claim_connection_obj.execute("COMMIT")
    claim_connection_obj.close()
    abandon_thread_obj.join(timeout=10)
    assert result_dict == {"abandoned_bool": False}
    decision_status_str, vplan_status_str, metadata_dict = _status_pair(db_path_str, decision_obj, vplan_obj)
    assert (decision_status_str, vplan_status_str) == ("vplan_ready", "submitting")
    assert "mr_capsule_unsubmitted_cycle_abandoned_bool" not in metadata_dict


def test_claim_waits_for_an_in_flight_abandonment_and_then_fails_closed(ready_cycle):
    db_path_str, decision_obj, vplan_obj = ready_cycle
    # Open worker A's store first: the constructor's schema statements also take the write lock.
    claim_store_obj = LiveStateStore(db_path_str)
    # Worker B is inside an abandonment: it holds the write lock and has marked both plans terminal.
    abandon_connection_obj = sqlite3.connect(db_path_str, isolation_level=None)
    abandon_connection_obj.execute("BEGIN IMMEDIATE")
    abandon_connection_obj.execute("UPDATE vplan SET status_str = 'blocked' WHERE vplan_id_int = ?", (vplan_obj.vplan_id_int,))
    abandon_connection_obj.execute(
        "UPDATE decision_plan SET status_str = 'blocked', snapshot_metadata_json_str = ? WHERE decision_plan_id_int = ?",
        (json.dumps({"sizing_contract_str": MR_CAPSULE_CONTRACT_STR, "mr_capsule_unsubmitted_cycle_abandoned_bool": True}),
         decision_obj.decision_plan_id_int),
    )
    result_dict = {}
    claim_thread_obj = threading.Thread(target=lambda: result_dict.update(
        claimed_bool=claim_store_obj.claim_vplan_for_submission(vplan_obj.vplan_id_int)
    ))
    claim_thread_obj.start()
    time.sleep(0.4)
    assert claim_thread_obj.is_alive(), "the claim must wait for the abandonment lock"
    abandon_connection_obj.execute("COMMIT")
    abandon_connection_obj.close()
    claim_thread_obj.join(timeout=10)
    assert result_dict == {"claimed_bool": False}
    assert _status_pair(db_path_str, decision_obj, vplan_obj)[:2] == ("blocked", "blocked")


def test_abandonment_undoes_itself_if_the_vplan_left_ready_inside_its_transaction(ready_cycle, monkeypatch):
    """Defence in depth: the guarded UPDATEs never overwrite a VPlan that is no longer 'ready'."""
    db_path_str, decision_obj, vplan_obj = ready_cycle
    store_obj = LiveStateStore(db_path_str)
    original_connect_fn = store_obj._connect

    class _ClaimDuringAbandonConnection:
        """Moves the VPlan to 'submitting' right after the abandonment's checks, as an unlocked writer would."""

        def __init__(self):
            self._connection_obj = original_connect_fn()

        def __enter__(self):
            self._connection_obj.__enter__()
            return self

        def __exit__(self, *exc_info):
            return self._connection_obj.__exit__(*exc_info)

        def rollback(self):
            self._connection_obj.rollback()

        def execute(self, sql_str, parameter_tuple=()):
            if sql_str.startswith("UPDATE vplan SET status_str = ?"):
                self._connection_obj.execute("UPDATE vplan SET status_str = 'submitting' WHERE vplan_id_int = ?", (vplan_obj.vplan_id_int,))
            return self._connection_obj.execute(sql_str, parameter_tuple)

    monkeypatch.setattr(store_obj, "_connect", _ClaimDuringAbandonConnection)
    assert store_obj.abandon_unsubmitted_mr_capsule_cycle(decision_obj.decision_plan_id_int, "expired") is False
    decision_status_str, _, metadata_dict = _status_pair(db_path_str, decision_obj, vplan_obj)
    assert decision_status_str == "vplan_ready" and "mr_capsule_unsubmitted_cycle_abandoned_bool" not in metadata_dict



def test_two_workers_can_upgrade_the_same_pre_capsule_database(tmp_path, monkeypatch):
    """Force both old-schema readers to overlap; real SQLite must serialize the upgrade."""
    db_path_str = str(tmp_path / "simultaneous_upgrade.sqlite3")
    LiveStateStore(db_path_str)
    with sqlite3.connect(db_path_str) as connection_obj:
        connection_obj.execute("ALTER TABLE decision_plan DROP COLUMN target_share_json_str")
    schema_barrier_obj = threading.Barrier(2)
    second_worker_event_obj = threading.Event()
    counter_lock_obj = threading.Lock()
    counter_dict = {"begin": 0, "read": 0}
    error_list = []

    class SchemaCursor:
        def __init__(self, row_list):
            self.row_list = row_list

        def fetchall(self):
            return self.row_list

    class UpgradeConnection(sqlite3.Connection):
        def execute(self, statement_str, *argument_tuple):
            # Both constructors finish their existing table/index DDL before
            # synchronizing at the read immediately preceding this migration.
            if statement_str == "PRAGMA table_info(live_release)":
                schema_barrier_obj.wait(timeout=5)
            if statement_str.strip() == "BEGIN IMMEDIATE":
                with counter_lock_obj:
                    counter_dict["begin"] += 1
                    if counter_dict["begin"] == 2:
                        second_worker_event_obj.set()
            cursor_obj = super().execute(statement_str, *argument_tuple)
            if statement_str == "PRAGMA table_info(decision_plan)":
                row_list = cursor_obj.fetchall()
                with counter_lock_obj:
                    counter_dict["read"] += 1
                    first_reader_bool = counter_dict["read"] == 1
                if first_reader_bool:
                    assert second_worker_event_obj.wait(timeout=5), "Second migration worker never reached the upgrade"
                else:
                    second_worker_event_obj.set()
                return SchemaCursor(row_list)
            return cursor_obj

    def connect_store(store_obj):
        connection_obj = sqlite3.connect(store_obj.db_path_str, factory=UpgradeConnection)
        connection_obj.row_factory = sqlite3.Row
        return connection_obj

    def upgrade_worker():
        try:
            LiveStateStore(db_path_str)
        except Exception as exception_obj:
            error_list.append(exception_obj)

    with monkeypatch.context() as patch_obj:
        patch_obj.setattr(LiveStateStore, "_connect", connect_store)
        worker_list = [threading.Thread(target=upgrade_worker) for _ in range(2)]
        for worker_obj in worker_list:
            worker_obj.start()
        for worker_obj in worker_list:
            worker_obj.join(timeout=15)
        assert not any(worker_obj.is_alive() for worker_obj in worker_list)
    assert error_list == []
    upgraded_store_obj = LiveStateStore(db_path_str)
    decision_obj = upgraded_store_obj.insert_decision_plan(_capsule_decision(target_share_map_dict={"BIL": 400}))
    assert upgraded_store_obj.get_decision_plan_by_id(decision_obj.decision_plan_id_int).target_share_map_dict == {"BIL": 400.0}
