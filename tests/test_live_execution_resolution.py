"""Offline absence proof and durable local dispatch evidence regressions."""
from dataclasses import replace
from datetime import timedelta

import pytest

from alpha.live.execution_engine import build_broker_order_request_list_from_vplan
from alpha.live.execution_resolution import (
    load_request_resolution_dict, record_never_dispatched_requests, resolve_never_sent_requests,
)
from alpha.live.models import BrokerOrderRecord
from alpha.live.mr_capsule_recovery import claim_capsule_request
from alpha.live.mr_capsule_reconcile import classify_capsule_execution
from test_live_mr_capsule_recovery import capsule_case, RECONCILE_TIMESTAMP_TS
from test_live_mr_capsule_execution_policy import reconcile_case, seed_execution


def _empty_evidence(vplan_obj, as_of_ts):
    return {"account_route_str": vplan_obj.account_route_str, "source_str": "offline.complete_broker_query",
        "refresh_started_timestamp_str": as_of_ts.isoformat(), "refreshed_timestamp_str": as_of_ts.isoformat(),
        "coverage_since_timestamp_str": vplan_obj.submission_timestamp_ts.isoformat(),
        "open_orders_complete_bool": True, "completed_orders_complete_bool": True, "executions_complete_bool": True,
        "order_row_list": [], "execution_row_list": []}


def _claimed_empty_cycle(case_tuple, as_of_ts):
    store_obj, broker_obj, release_obj, vplan_obj, _, _ = case_tuple
    assert store_obj.claim_vplan_for_submission(vplan_obj.vplan_id_int)
    snapshot_obj = broker_obj.get_account_snapshot(release_obj.account_route_str)
    broker_obj._snapshot_map[release_obj.account_route_str] = replace(snapshot_obj, snapshot_timestamp_ts=as_of_ts)
    return build_broker_order_request_list_from_vplan(vplan_obj)


def test_claimed_empty_cycle_completes_after_close_with_audited_fresh_absence(capsule_case, monkeypatch):
    store_obj, broker_obj, release_obj, vplan_obj, _, _ = capsule_case
    as_of_ts = RECONCILE_TIMESTAMP_TS.replace(hour=16, minute=5)
    request_list = _claimed_empty_cycle(capsule_case, as_of_ts)
    call_list = []
    def refreshed_evidence(account_route_str, since_timestamp_ts):
        call_list.append((account_route_str, since_timestamp_ts))
        return _empty_evidence(vplan_obj, as_of_ts)
    monkeypatch.setattr(broker_obj, "get_refreshed_order_evidence", refreshed_evidence)
    result_tuple = reconcile_case(capsule_case, as_of_ts)
    assert result_tuple[0].passed_bool and result_tuple[1] == "accepted_residual"
    assert store_obj.get_vplan_by_id(vplan_obj.vplan_id_int).status_str == "completed"
    assert store_obj.get_decision_plan_by_id(vplan_obj.decision_plan_id_int).status_str == "completed"
    assert broker_obj.submitted_order_request_list == []
    assert len(call_list) == 1
    resolution_dict = load_request_resolution_dict(store_obj, vplan_obj)
    assert set(resolution_dict) == {request_obj.order_request_key_str for request_obj in request_list}
    assert {row_dict["resolution_str"] for row_dict in resolution_dict.values()} == {"never_sent"}
    assert store_obj.get_broker_order_row_dict_list_for_vplan(vplan_obj.vplan_id_int) == []
    assert store_obj.get_pod_state(release_obj.pod_id_str).position_amount_map == vplan_obj.current_broker_position_map


def test_before_close_never_claims_broker_absence(capsule_case, monkeypatch):
    store_obj, broker_obj, _, vplan_obj, _, _ = capsule_case
    _claimed_empty_cycle(capsule_case, RECONCILE_TIMESTAMP_TS)
    def forbidden_refresh(*argument_tuple):
        raise AssertionError("Before-close absence cannot resolve a claim.")
    monkeypatch.setattr(broker_obj, "get_refreshed_order_evidence", forbidden_refresh)
    assert reconcile_case(capsule_case)[1] == "awaiting_evidence"
    assert load_request_resolution_dict(store_obj, vplan_obj) == {}


@pytest.mark.parametrize("fault_str", ["open_incomplete", "completed_incomplete", "executions_incomplete", "stale", "short_history", "wrong_account", "unidentified_execution"])
def test_incomplete_or_stale_broker_proof_cannot_terminalize_claim(capsule_case, monkeypatch, fault_str):
    store_obj, broker_obj, _, vplan_obj, _, _ = capsule_case
    as_of_ts = RECONCILE_TIMESTAMP_TS.replace(hour=16, minute=5)
    _claimed_empty_cycle(capsule_case, as_of_ts)
    evidence_dict = _empty_evidence(vplan_obj, as_of_ts)
    if fault_str.endswith("_incomplete"):
        field_dict = {"open_incomplete": "open_orders_complete_bool", "completed_incomplete": "completed_orders_complete_bool",
            "executions_incomplete": "executions_complete_bool"}
        evidence_dict[field_dict[fault_str]] = False
    elif fault_str == "stale":
        evidence_dict["refresh_started_timestamp_str"] = (as_of_ts - timedelta(minutes=1)).isoformat()
    elif fault_str == "short_history":
        evidence_dict["coverage_since_timestamp_str"] = vplan_obj.target_execution_timestamp_ts.isoformat()
    elif fault_str == "wrong_account":
        evidence_dict["account_route_str"] = "OTHER"
    else:
        evidence_dict["execution_row_list"] = [{}]
    monkeypatch.setattr(broker_obj, "get_refreshed_order_evidence", lambda *argument_tuple: evidence_dict)
    with pytest.raises(ValueError, match="proof"):
        reconcile_case(capsule_case, as_of_ts)
    assert load_request_resolution_dict(store_obj, vplan_obj) == {}
    assert store_obj.get_vplan_by_id(vplan_obj.vplan_id_int).status_str == "submitting"


@pytest.mark.parametrize("observation_str", ["order_ref", "execution_ref", "unknown_ref_execution"])
def test_order_or_execution_evidence_blocks_absence_for_its_asset(capsule_case, monkeypatch, observation_str):
    store_obj, broker_obj, release_obj, vplan_obj, _, _ = capsule_case
    as_of_ts = RECONCILE_TIMESTAMP_TS.replace(hour=16, minute=5)
    request_list = _claimed_empty_cycle(capsule_case, as_of_ts)
    request_obj = next(request_obj for request_obj in request_list if request_obj.asset_str == "BIL")
    evidence_dict = _empty_evidence(vplan_obj, as_of_ts)
    row_dict = {"asset_str": "BIL", "broker_order_id_str": "broker-bil",
        "order_request_key_str": "" if observation_str == "unknown_ref_execution" else request_obj.order_request_key_str}
    evidence_dict["order_row_list" if observation_str == "order_ref" else "execution_row_list"] = [row_dict]
    monkeypatch.setattr(broker_obj, "get_refreshed_order_evidence", lambda *argument_tuple: evidence_dict)
    result_dict = resolve_never_sent_requests(store_obj, broker_obj, release_obj, vplan_obj, request_list, as_of_ts)
    assert request_obj.order_request_key_str not in result_dict
    assert {row_dict["asset_str"] for row_dict in result_dict.values()} == {"AAPL"}


@pytest.mark.parametrize("original_execution_has_ref_bool", [False, True])
def test_completion_claim_that_never_reached_broker_resolves_after_close(capsule_case, monkeypatch, original_execution_has_ref_bool):
    store_obj, broker_obj, _, vplan_obj, _, _ = seed_execution(capsule_case)
    request_obj = next(request_obj for request_obj in build_broker_order_request_list_from_vplan(vplan_obj)
        if request_obj.asset_str == "BIL")
    recovery_obj = replace(request_obj, amount_float=-60.0, broker_order_type_str="MKT",
        order_request_key_str=f"{vplan_obj.submission_key_str}:late:BIL")
    assert claim_capsule_request(store_obj, vplan_obj, recovery_obj, "recovery", RECONCILE_TIMESTAMP_TS)
    as_of_ts = RECONCILE_TIMESTAMP_TS.replace(hour=16, minute=5)
    snapshot_obj = broker_obj.get_account_snapshot(vplan_obj.account_route_str)
    broker_obj._snapshot_map[vplan_obj.account_route_str] = replace(snapshot_obj, snapshot_timestamp_ts=as_of_ts)
    evidence_dict = _empty_evidence(vplan_obj, as_of_ts)
    evidence_dict["execution_row_list"] = [{"asset_str": "BIL", "broker_order_id_str": "original:BIL",
        "order_request_key_str": request_obj.order_request_key_str if original_execution_has_ref_bool else "",
        "broker_execution_id_str": "original-execution", "fill_timestamp_str": vplan_obj.target_execution_timestamp_ts.isoformat()}]
    monkeypatch.setattr(broker_obj, "get_refreshed_order_evidence", lambda *argument_tuple: evidence_dict)
    result_tuple = reconcile_case(capsule_case, as_of_ts)
    assert result_tuple[0].passed_bool and result_tuple[1] == "accepted_residual"
    assert broker_obj.submitted_order_request_list == []
    assert load_request_resolution_dict(store_obj, vplan_obj)[recovery_obj.order_request_key_str]["resolution_str"] == "never_sent"


def test_local_never_dispatched_proof_is_terminal_before_close_and_audited(capsule_case):
    store_obj, broker_obj, release_obj, vplan_obj, _, _ = capsule_case
    request_list = _claimed_empty_cycle(capsule_case, RECONCILE_TIMESTAMP_TS)
    result_dict = record_never_dispatched_requests(store_obj, release_obj, vplan_obj, request_list,
        vplan_obj.submission_timestamp_ts, "Dispatch cutoff reached before placeOrder.")
    assert {row_dict["resolution_str"] for row_dict in result_dict.values()} == {"never_dispatched"}
    result_tuple = classify_capsule_execution(vplan_obj, broker_obj.get_account_snapshot(release_obj.account_route_str),
        [], [], resolved_request_dict=result_dict)
    assert result_tuple[0].passed_bool and result_tuple[1] == "accepted_residual"
    assert broker_obj.submitted_order_request_list == []


def test_after_close_proof_needs_holdings_observed_after_that_proof(capsule_case, monkeypatch):
    store_obj, broker_obj, _, vplan_obj, _, _ = capsule_case
    _claimed_empty_cycle(capsule_case, RECONCILE_TIMESTAMP_TS)
    as_of_ts = RECONCILE_TIMESTAMP_TS.replace(hour=16, minute=5)
    monkeypatch.setattr(broker_obj, "get_refreshed_order_evidence", lambda *argument_tuple: _empty_evidence(vplan_obj, as_of_ts))
    result_tuple = reconcile_case(capsule_case, as_of_ts)
    assert result_tuple[1] == "awaiting_evidence" and not result_tuple[0].passed_bool
    assert store_obj.get_vplan_by_id(vplan_obj.vplan_id_int).status_str == "submitting"
    assert len(load_request_resolution_dict(store_obj, vplan_obj)) == 2
    snapshot_obj = broker_obj.get_account_snapshot(vplan_obj.account_route_str)
    broker_obj._snapshot_map[vplan_obj.account_route_str] = replace(snapshot_obj, snapshot_timestamp_ts=as_of_ts)
    assert reconcile_case(capsule_case, as_of_ts)[0].passed_bool


def test_local_proof_rejects_changed_intent_and_other_strategies(capsule_case):
    store_obj, _, release_obj, vplan_obj, _, _ = capsule_case
    request_obj = build_broker_order_request_list_from_vplan(vplan_obj)[0]
    with pytest.raises(ValueError, match="persisted intent"):
        record_never_dispatched_requests(store_obj, release_obj, vplan_obj,
            [replace(request_obj, amount_float=request_obj.amount_float + 1)], RECONCILE_TIMESTAMP_TS, "test")
    with pytest.raises(ValueError, match="restricted"):
        record_never_dispatched_requests(store_obj, replace(release_obj, strategy_import_str="ndx"), vplan_obj,
            [request_obj], RECONCILE_TIMESTAMP_TS, "test")
    assert load_request_resolution_dict(store_obj, vplan_obj) == {}


def test_local_proof_joins_callers_transaction_and_rolls_back_atomically(capsule_case):
    store_obj, _, release_obj, vplan_obj, _, _ = capsule_case
    request_list = build_broker_order_request_list_from_vplan(vplan_obj)
    with pytest.raises(RuntimeError, match="injected"):
        with store_obj._connect() as connection_obj:
            connection_obj.execute("BEGIN IMMEDIATE")
            connection_obj.execute("UPDATE vplan SET status_str='submitting' WHERE vplan_id_int=?", (vplan_obj.vplan_id_int,))
            result_dict = record_never_dispatched_requests(store_obj, release_obj, vplan_obj,
                request_list, vplan_obj.submission_timestamp_ts, "deadline", connection_obj=connection_obj)
            assert len(result_dict) == len(request_list)
            raise RuntimeError("injected rollback")
    assert load_request_resolution_dict(store_obj, vplan_obj) == {}
    assert store_obj.get_vplan_by_id(vplan_obj.vplan_id_int).status_str == "ready"


def test_late_observed_order_overrides_local_never_dispatched_proof(capsule_case):
    store_obj, broker_obj, release_obj, vplan_obj, _, _ = capsule_case
    request_list = _claimed_empty_cycle(capsule_case, RECONCILE_TIMESTAMP_TS)
    record_never_dispatched_requests(store_obj, release_obj, vplan_obj, request_list,
        vplan_obj.submission_timestamp_ts, "test local proof")
    request_obj = next(request_obj for request_obj in request_list if request_obj.asset_str == "BIL")
    broker_obj.seed_broker_order_state(BrokerOrderRecord(broker_order_id_str="late-observed", decision_plan_id_int=None,
        vplan_id_int=None, account_route_str=vplan_obj.account_route_str, asset_str="BIL",
        order_request_key_str=request_obj.order_request_key_str, broker_order_type_str="MOO", unit_str="shares",
        amount_float=request_obj.amount_float, filled_amount_float=0, status_str="Submitted",
        submitted_timestamp_ts=vplan_obj.submission_timestamp_ts, submission_key_str=vplan_obj.submission_key_str,
        raw_payload_dict={"snapshot_source_str": "open_order"}))
    assert reconcile_case(capsule_case)[1] == "awaiting_evidence"
    assert store_obj.get_vplan_by_id(vplan_obj.vplan_id_int).status_str == "submitting"
    assert broker_obj.submitted_order_request_list == []
