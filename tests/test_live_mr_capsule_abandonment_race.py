"""Deterministic SQLite interleavings for capsule build/abandon/claim races."""
import json
import sqlite3
from datetime import datetime, timezone

import pytest

from alpha.live.models import DecisionPlan, VPlan, VPlanRow
from alpha.live.mr_capsule_adapter import MR_CAPSULE_CONTRACT_STR
from alpha.live.state_store_v2 import LiveStateStore


@pytest.fixture
def capsule_cycle(monkeypatch):
    connection_obj = sqlite3.connect(":memory:")
    connection_obj.row_factory = sqlite3.Row
    store_obj = LiveStateStore.__new__(LiveStateStore)
    monkeypatch.setattr(store_obj, "_connect", lambda: connection_obj)
    store_obj._initialize_schema()
    store_obj._initialize_v2_schema()
    signal_ts = datetime(2024, 1, 12, 21, tzinfo=timezone.utc)
    execution_ts = datetime(2024, 1, 16, 14, 30, tzinfo=timezone.utc)
    identity_dict = dict(release_id_str="capsule.v1", user_id_str="owner", pod_id_str="capsule", account_route_str="DU123")
    timing_dict = dict(signal_timestamp_ts=signal_ts, submission_timestamp_ts=execution_ts,
                       target_execution_timestamp_ts=execution_ts, execution_policy_str="next_open_moo")
    decision_obj = store_obj.insert_decision_plan(DecisionPlan(
        **identity_dict, **timing_dict, decision_base_position_map={}, strategy_state_dict={"trade_id_int": 99},
        snapshot_metadata_dict={"sizing_contract_str": MR_CAPSULE_CONTRACT_STR}, target_share_map_dict={"BIL": 10},
    ))
    candidate_obj = VPlan(
        **identity_dict, **timing_dict, decision_plan_id_int=decision_obj.decision_plan_id_int,
        broker_snapshot_timestamp_ts=execution_ts, live_reference_snapshot_timestamp_ts=execution_ts,
        live_price_source_str="synthetic", net_liq_float=10_000.0, available_funds_float=None,
        excess_liquidity_float=None, pod_budget_fraction_float=1.0, pod_budget_float=10_000.0,
        current_broker_position_map={}, live_reference_price_map={"BIL": 100.0},
        target_share_map={"BIL": 10.0}, order_delta_map={"BIL": 10.0},
        vplan_row_list=[VPlanRow(
            asset_str="BIL", current_share_float=0.0, target_share_float=10.0, order_delta_share_float=10.0,
            live_reference_price_float=100.0, estimated_target_notional_float=1000.0, broker_order_type_str="MOO",
        )],
    )
    yield store_obj, decision_obj, candidate_obj
    connection_obj.close()


@pytest.mark.parametrize("status_str", ["blocked", "expired"])
def test_candidate_built_before_abandonment_cannot_resurrect_cycle(capsule_cycle, status_str):
    store_obj, decision_obj, candidate_obj = capsule_cycle
    # Worker A has built candidate_obj; worker B abandons before A persists it.
    assert store_obj.abandon_unsubmitted_mr_capsule_cycle(decision_obj.decision_plan_id_int, status_str)
    with pytest.raises(ValueError, match="active planned decision"):
        store_obj.insert_vplan(candidate_obj)
    assert store_obj.get_latest_vplan_for_decision(decision_obj.decision_plan_id_int) is None
    persisted_obj = store_obj.get_decision_plan_by_id(decision_obj.decision_plan_id_int)
    assert persisted_obj.status_str == status_str
    assert persisted_obj.snapshot_metadata_dict["mr_capsule_unsubmitted_cycle_abandoned_bool"] is True


def test_insert_and_decision_ready_transition_roll_back_together(capsule_cycle):
    store_obj, decision_obj, candidate_obj = capsule_cycle
    with store_obj._connect() as connection_obj:
        connection_obj.execute("""CREATE TRIGGER fail_ready_transition BEFORE UPDATE OF status_str ON decision_plan
            WHEN NEW.status_str = 'vplan_ready' BEGIN SELECT RAISE(ABORT, 'simulated build crash'); END""")
    with pytest.raises(sqlite3.IntegrityError, match="simulated build crash"):
        store_obj.insert_vplan(candidate_obj)
    assert store_obj.get_latest_vplan_for_decision(decision_obj.decision_plan_id_int) is None
    assert store_obj.get_decision_plan_by_id(decision_obj.decision_plan_id_int).status_str == "planned"
    with store_obj._connect() as connection_obj:
        assert connection_obj.execute("SELECT COUNT(*) FROM vplan_row").fetchone()[0] == 0


def test_first_builder_wins_and_claimed_no_ack_cycle_cannot_be_abandoned(capsule_cycle):
    store_obj, decision_obj, candidate_obj = capsule_cycle
    inserted_obj = store_obj.insert_vplan(candidate_obj)
    assert store_obj.get_decision_plan_by_id(decision_obj.decision_plan_id_int).status_str == "vplan_ready"
    # A second worker resumes a candidate built from the same earlier planned state.
    with pytest.raises(ValueError, match="active planned decision"):
        store_obj.insert_vplan(candidate_obj)
    assert store_obj.claim_vplan_for_submission(inserted_obj.vplan_id_int)
    assert not store_obj.abandon_unsubmitted_mr_capsule_cycle(decision_obj.decision_plan_id_int, "expired")
    assert store_obj.get_latest_vplan_for_decision(decision_obj.decision_plan_id_int).status_str == "submitting"
    assert not store_obj.claim_vplan_for_submission(inserted_obj.vplan_id_int)


def test_abandonment_after_insert_wins_over_later_claim(capsule_cycle):
    store_obj, decision_obj, candidate_obj = capsule_cycle
    inserted_obj = store_obj.insert_vplan(candidate_obj)
    assert store_obj.abandon_unsubmitted_mr_capsule_cycle(decision_obj.decision_plan_id_int, "blocked")
    assert not store_obj.claim_vplan_for_submission(inserted_obj.vplan_id_int)
    assert store_obj.get_latest_vplan_for_decision(decision_obj.decision_plan_id_int).status_str == "blocked"


@pytest.mark.parametrize("status_str,abandoned_bool", [
    ("planned", False), ("blocked", False), ("expired", False), ("completed", False), ("vplan_ready", True),
])
def test_claim_refuses_ready_vplan_with_inactive_or_abandoned_parent(capsule_cycle, status_str, abandoned_bool):
    store_obj, decision_obj, candidate_obj = capsule_cycle
    inserted_obj = store_obj.insert_vplan(candidate_obj)
    # Represents a stale ready row left by an old writer, without calling the fixed abandonment path.
    metadata_dict = {**decision_obj.snapshot_metadata_dict, "mr_capsule_unsubmitted_cycle_abandoned_bool": abandoned_bool}
    with store_obj._connect() as connection_obj:
        connection_obj.execute("UPDATE decision_plan SET status_str = ?, snapshot_metadata_json_str = ? WHERE decision_plan_id_int = ?",
                               (status_str, json.dumps(metadata_dict), decision_obj.decision_plan_id_int))
    assert not store_obj.claim_vplan_for_submission(inserted_obj.vplan_id_int)
    assert store_obj.get_latest_vplan_for_decision(decision_obj.decision_plan_id_int).status_str == "ready"


def test_non_capsule_insert_and_claim_keep_existing_status_contract(capsule_cycle):
    store_obj, decision_obj, candidate_obj = capsule_cycle
    with store_obj._connect() as connection_obj:
        connection_obj.execute("UPDATE decision_plan SET snapshot_metadata_json_str = '{}' WHERE decision_plan_id_int = ?",
                               (decision_obj.decision_plan_id_int,))
    inserted_obj = store_obj.insert_vplan(candidate_obj)
    assert store_obj.get_decision_plan_by_id(decision_obj.decision_plan_id_int).status_str == "planned"
    assert store_obj.claim_vplan_for_submission(inserted_obj.vplan_id_int)
