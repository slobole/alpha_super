"""CORE5 signed execution accounting and terminal evidence without a broker."""
from dataclasses import replace
from datetime import timedelta

import pytest

from alpha.live.core5_recovery import classify_core5_cycle_evidence, park_core5_cycle, refresh_core5_cycle_evidence
from alpha.live.execution_engine import build_broker_order_request_list_from_vplan
from alpha.live.models import BrokerOrderFill, BrokerOrderRecord
from alpha.live.runner import post_execution_reconcile
from test_live_core5_wiring import _prepared_cycle, _release_for_mode, core5_price_df, qualified_release_obj


@pytest.fixture
def recovery_case(tmp_path, monkeypatch, qualified_release_obj, core5_price_df):
    release_obj = _release_for_mode(qualified_release_obj, "paper")
    return (release_obj, *_prepared_cycle(tmp_path, monkeypatch, release_obj, core5_price_df, "long_to_short"))


def _seed_complete_cycle(case_tuple):
    release_obj, store_obj, broker_obj, _, vplan_obj, _ = case_tuple
    request_list = build_broker_order_request_list_from_vplan(vplan_obj)
    position_dict = dict(vplan_obj.current_broker_position_map)
    for request_index_int, request_obj in enumerate(request_list):
        record_obj = BrokerOrderRecord(broker_order_id_str=f"core5:{request_index_int}",
            decision_plan_id_int=None, vplan_id_int=None, account_route_str=release_obj.account_route_str,
            asset_str=request_obj.asset_str, order_request_key_str=request_obj.order_request_key_str,
            broker_order_type_str="MOO", unit_str="shares", amount_float=request_obj.amount_float,
            filled_amount_float=abs(request_obj.amount_float), status_str="Filled",
            submitted_timestamp_ts=vplan_obj.submission_timestamp_ts, submission_key_str=vplan_obj.submission_key_str,
            raw_payload_dict={"snapshot_source_str": "completed_order"})
        fill_obj = BrokerOrderFill(broker_order_id_str=record_obj.broker_order_id_str,
            decision_plan_id_int=None, vplan_id_int=None, account_route_str=release_obj.account_route_str,
            asset_str=request_obj.asset_str, fill_amount_float=request_obj.amount_float,
            fill_price_float=request_obj.sizing_reference_price_float, fill_timestamp_ts=vplan_obj.target_execution_timestamp_ts)
        broker_obj.seed_broker_order_state(record_obj, broker_order_fill_list=[fill_obj])
        position_dict[request_obj.asset_str] = position_dict.get(request_obj.asset_str, 0.0) + request_obj.amount_float
    as_of_ts = vplan_obj.target_execution_timestamp_ts + timedelta(minutes=10)
    broker_obj.seed_account_snapshot(release_obj.account_route_str, 1000.0, 100000.0, position_dict,
        snapshot_timestamp_ts=as_of_ts, session_mode_str=release_obj.mode_str)
    store_obj.mark_vplan_status(vplan_obj.vplan_id_int, "submitted")
    store_obj.mark_decision_plan_status(vplan_obj.decision_plan_id_int, "submitted")
    return as_of_ts


def _persisted_evidence(case_tuple):
    release_obj, store_obj, broker_obj, _, vplan_obj, _ = case_tuple
    return (broker_obj.get_core5_account_snapshot(release_obj.account_route_str),
        store_obj.get_broker_order_row_dict_list_for_vplan(vplan_obj.vplan_id_int, include_evidence_bool=True),
        store_obj.get_fill_row_dict_list_for_vplan(vplan_obj.vplan_id_int,
            include_order_identity_bool=True, include_evidence_bool=True))


def test_refresh_preserves_signed_dbc_round_trip_and_reads_required_sql_evidence(recovery_case):
    release_obj, store_obj, broker_obj, _, vplan_obj, _ = recovery_case
    as_of_ts = _seed_complete_cycle(recovery_case)
    result_dict = refresh_core5_cycle_evidence(store_obj, broker_obj, release_obj, vplan_obj, as_of_ts)
    assert result_dict["terminal_bool"] and result_dict["complete_bool"]
    assert result_dict["expected_position_map_dict"]["DBC"] == vplan_obj.target_share_map["DBC"] < 0
    dbc_order_list = [row_dict for row_dict in _persisted_evidence(recovery_case)[1] if row_dict["asset_str"] == "DBC"]
    assert len(dbc_order_list) == 2
    assert len({row_dict["order_request_key_str"] for row_dict in dbc_order_list}) == 2


def test_identical_distinct_executions_complete_core5_and_commit_memory(recovery_case):
    release_obj, store_obj, broker_obj, decision_obj, vplan_obj, runner_kwarg_dict = recovery_case
    as_of_ts = _seed_complete_cycle(recovery_case)
    original_fill_list = broker_obj._fill_map[release_obj.account_route_str]
    original_fill_obj = next(fill_obj for fill_obj in original_fill_list
        if abs(fill_obj.fill_amount_float) >= 2 and fill_obj.fill_amount_float % 2 == 0)
    split_fill_list = [replace(original_fill_obj, fill_amount_float=original_fill_obj.fill_amount_float / 2,
        raw_payload_dict={"exec_id_str": f"core5-identical-execution-{execution_index_int}"})
        for execution_index_int in (1, 2)]
    # Both executions have the same broker order, second, signed quantity and
    # price. The unchanged order total and actual holdings require both rows.
    broker_obj._fill_map[release_obj.account_route_str] = [fill_obj for fill_obj in original_fill_list
        if fill_obj is not original_fill_obj] + split_fill_list
    prior_memory_dict = dict(store_obj.get_pod_state(release_obj.pod_id_str).strategy_state_dict)
    assert prior_memory_dict != decision_obj.strategy_state_dict
    result_dict = post_execution_reconcile(store_obj, broker_obj, as_of_ts, release_obj.mode_str, **runner_kwarg_dict)
    assert result_dict["completed_vplan_count_int"] == 1
    assert store_obj.get_vplan_by_id(vplan_obj.vplan_id_int).status_str == "completed"
    assert store_obj.get_decision_plan_by_id(decision_obj.decision_plan_id_int).status_str == "completed"
    assert store_obj.get_pod_state(release_obj.pod_id_str).strategy_state_dict == decision_obj.strategy_state_dict
    with store_obj._connect() as connection_obj:
        split_row_list = connection_obj.execute("""SELECT broker_execution_id_str, fill_amount_float,
            fill_price_float, fill_timestamp_str FROM vplan_fill
            WHERE vplan_id_int=? AND broker_order_id_str=?""",
            (vplan_obj.vplan_id_int, original_fill_obj.broker_order_id_str)).fetchall()
    assert len(split_row_list) == 2
    assert {row_obj["broker_execution_id_str"] for row_obj in split_row_list} == {
        "core5-identical-execution-1", "core5-identical-execution-2"}
    assert len({(row_obj["fill_amount_float"], row_obj["fill_price_float"], row_obj["fill_timestamp_str"])
        for row_obj in split_row_list}) == 1
    assert sum(row_obj["fill_amount_float"] for row_obj in split_row_list) == original_fill_obj.fill_amount_float


@pytest.mark.parametrize("refresh_error_bool", [False, True])
def test_claimed_restart_with_missing_evidence_is_durable_and_critical(recovery_case, monkeypatch, refresh_error_bool):
    release_obj, store_obj, broker_obj, decision_obj, vplan_obj, runner_kwarg_dict = recovery_case
    assert store_obj.claim_vplan_for_submission(vplan_obj.vplan_id_int)
    prior_state_obj = store_obj.get_pod_state(release_obj.pod_id_str)
    as_of_ts = vplan_obj.target_execution_timestamp_ts + timedelta(minutes=10)
    snapshot_obj = broker_obj.get_core5_account_snapshot(release_obj.account_route_str)
    broker_obj._snapshot_map[release_obj.account_route_str] = replace(snapshot_obj, snapshot_timestamp_ts=as_of_ts)
    if refresh_error_bool:
        def failed_snapshot_fn(*_argument_tuple):
            raise TimeoutError("Synthetic connection failure during recovery")
        monkeypatch.setattr(broker_obj, "get_core5_account_snapshot", failed_snapshot_fn)
    result_dict = post_execution_reconcile(store_obj, broker_obj, as_of_ts, "paper", **runner_kwarg_dict)
    assert result_dict["completed_vplan_count_int"] == 0
    assert store_obj.get_vplan_by_id(vplan_obj.vplan_id_int).status_str == "submitted"
    assert store_obj.get_decision_plan_by_id(decision_obj.decision_plan_id_int).snapshot_metadata_dict["core5_pending_execution_dict"]["severity_str"] == "critical"
    assert store_obj.get_pod_state(release_obj.pod_id_str).strategy_state_dict == prior_state_obj.strategy_state_dict
    with store_obj._connect() as connection_obj:
        assert connection_obj.execute("SELECT COUNT(*) FROM mr_capsule_execution_alert WHERE vplan_id_int=? AND alert_kind_str='dispatch_failed'", (vplan_obj.vplan_id_int,)).fetchone()[0] == 1


@pytest.mark.parametrize("fault_str", ["future_fill", "pre_submission_fill", "wrong_fill_account", "wrong_fill_side",
    "nonfinite_fill", "nonfinite_position", "nonfinite_cash", "nonfinite_nav", "open_order"])
def test_invalid_execution_or_account_observation_never_becomes_terminal(recovery_case, fault_str):
    release_obj, store_obj, broker_obj, _, vplan_obj, _ = recovery_case
    as_of_ts = _seed_complete_cycle(recovery_case)
    refresh_core5_cycle_evidence(store_obj, broker_obj, release_obj, vplan_obj, as_of_ts)
    snapshot_obj, order_list, fill_list = _persisted_evidence(recovery_case)
    if fault_str == "future_fill":
        fill_list[0]["fill_timestamp_str"] = (as_of_ts + timedelta(seconds=1)).isoformat()
    elif fault_str == "pre_submission_fill":
        fill_list[0]["fill_timestamp_str"] = (vplan_obj.submission_timestamp_ts - timedelta(seconds=1)).isoformat()
    elif fault_str == "wrong_fill_account":
        fill_list[0]["account_route_str"] = "OTHER"
    elif fault_str == "wrong_fill_side":
        fill_list[0]["fill_amount_float"] *= -1
    elif fault_str == "nonfinite_fill":
        fill_list[0]["fill_amount_float"] = float("nan")
    elif fault_str == "nonfinite_position":
        snapshot_obj = replace(snapshot_obj, position_amount_map={**snapshot_obj.position_amount_map, "DBC": float("nan")})
    elif fault_str == "nonfinite_cash":
        snapshot_obj = replace(snapshot_obj, cash_float=float("inf"))
    elif fault_str == "nonfinite_nav":
        snapshot_obj = replace(snapshot_obj, net_liq_float=float("nan"))
    else:
        snapshot_obj = replace(snapshot_obj, open_order_id_list=[order_list[0]["broker_order_id_str"]])
    result_dict = classify_core5_cycle_evidence(vplan_obj, snapshot_obj, order_list, fill_list, {})
    assert not result_dict["terminal_bool"] and not result_dict["complete_bool"]


@pytest.mark.parametrize("fault_str", ["wrong_kind", "wrong_asset", "snapshot_before_proof"])
def test_resolution_requires_recognized_proof_for_this_asset_and_fresh_holdings(recovery_case, fault_str):
    release_obj, _, broker_obj, _, vplan_obj, _ = recovery_case
    snapshot_obj = replace(broker_obj.get_core5_account_snapshot(release_obj.account_route_str),
        snapshot_timestamp_ts=vplan_obj.target_execution_timestamp_ts + timedelta(minutes=10))
    resolution_dict = {request_obj.order_request_key_str: {"asset_str": request_obj.asset_str,
        "resolution_str": "never_dispatched", "created_timestamp_str": vplan_obj.submission_timestamp_ts.isoformat()}
        for request_obj in build_broker_order_request_list_from_vplan(vplan_obj)}
    first_resolution_dict = next(iter(resolution_dict.values()))
    if fault_str == "wrong_kind":
        first_resolution_dict["resolution_str"] = "unknown"
    elif fault_str == "wrong_asset":
        first_resolution_dict["asset_str"] = "OTHER"
    else:
        first_resolution_dict["evidence_dict"] = {"refreshed_timestamp_str": (
            snapshot_obj.snapshot_timestamp_ts + timedelta(seconds=1)).isoformat()}
    result_dict = classify_core5_cycle_evidence(vplan_obj, snapshot_obj, [], [], resolution_dict)
    assert not result_dict["terminal_bool"] and not result_dict["complete_bool"]


def test_claimed_never_sent_core5_parks_after_fresh_post_close_proof_without_advancing_memory(recovery_case, monkeypatch):
    release_obj, store_obj, broker_obj, decision_obj, vplan_obj, _ = recovery_case
    prior_memory_dict = dict(store_obj.get_pod_state(release_obj.pod_id_str).strategy_state_dict)
    assert store_obj.claim_vplan_for_submission(vplan_obj.vplan_id_int)
    as_of_ts = vplan_obj.target_execution_timestamp_ts.replace(hour=16, minute=5)
    snapshot_obj = broker_obj.get_core5_account_snapshot(release_obj.account_route_str)
    broker_obj._snapshot_map[release_obj.account_route_str] = replace(snapshot_obj, snapshot_timestamp_ts=as_of_ts)
    evidence_dict = {"account_route_str": release_obj.account_route_str, "source_str": "offline.complete_query",
        "refresh_started_timestamp_str": as_of_ts.isoformat(), "refreshed_timestamp_str": as_of_ts.isoformat(),
        "coverage_since_timestamp_str": vplan_obj.submission_timestamp_ts.isoformat(),
        "open_orders_complete_bool": True, "completed_orders_complete_bool": True, "executions_complete_bool": True,
        "order_row_list": [], "execution_row_list": []}
    monkeypatch.setattr(broker_obj, "get_refreshed_order_evidence", lambda *argument_tuple: evidence_dict)
    result_dict = refresh_core5_cycle_evidence(store_obj, broker_obj, release_obj, vplan_obj, as_of_ts)
    assert result_dict["terminal_bool"] and not result_dict["complete_bool"]
    park_core5_cycle(store_obj, release_obj, vplan_obj, as_of_ts, result_dict)
    assert store_obj.get_vplan_by_id(vplan_obj.vplan_id_int).status_str == "parked"
    assert store_obj.get_decision_plan_by_id(decision_obj.decision_plan_id_int).status_str == "blocked"
    assert store_obj.get_pod_state(release_obj.pod_id_str).strategy_state_dict == prior_memory_dict
    with store_obj._connect() as connection_obj:
        alert_row_obj = connection_obj.execute("SELECT * FROM mr_capsule_execution_alert").fetchone()
    assert alert_row_obj["alert_kind_str"] == "dispatch_failed"
    assert broker_obj.submitted_order_request_list == []


def test_observed_correlated_orders_override_older_never_dispatched_resolution(recovery_case):
    release_obj, store_obj, broker_obj, _, vplan_obj, _ = recovery_case
    as_of_ts = _seed_complete_cycle(recovery_case)
    refresh_core5_cycle_evidence(store_obj, broker_obj, release_obj, vplan_obj, as_of_ts)
    snapshot_obj, order_list, fill_list = _persisted_evidence(recovery_case)
    resolution_dict = {request_obj.order_request_key_str: {"asset_str": request_obj.asset_str,
        "resolution_str": "never_dispatched", "created_timestamp_str": vplan_obj.submission_timestamp_ts.isoformat()}
        for request_obj in build_broker_order_request_list_from_vplan(vplan_obj)}
    result_dict = classify_core5_cycle_evidence(vplan_obj, snapshot_obj, order_list, fill_list, resolution_dict)
    assert result_dict["terminal_bool"] and result_dict["complete_bool"]


@pytest.mark.parametrize("fault_str", ["account", "stale", "cash", "nav", "position"])
def test_recovery_rejects_invalid_snapshot_before_runner_can_overwrite_pod_state(recovery_case, monkeypatch, fault_str):
    release_obj, store_obj, broker_obj, _, vplan_obj, runner_kwarg_dict = recovery_case
    as_of_ts = _seed_complete_cycle(recovery_case)
    prior_state_obj = store_obj.get_pod_state(release_obj.pod_id_str)
    snapshot_obj = broker_obj.get_core5_account_snapshot(release_obj.account_route_str)
    change_dict = {"account": {"account_route_str": "OTHER"},
        "stale": {"snapshot_timestamp_ts": vplan_obj.submission_timestamp_ts},
        "cash": {"cash_float": float("nan")}, "nav": {"net_liq_float": -1.0},
        "position": {"position_amount_map": {"DBC": float("inf")}}}[fault_str]
    broken_snapshot_obj = replace(snapshot_obj, **change_dict)
    # A valid initial cache read is followed by the malformed recovery refresh.
    # The runner must not save the latter as authoritative pod holdings.
    snapshot_call_count_list = [0]
    def account_snapshot(account_route_str, **keyword_dict):
        snapshot_call_count_list[0] += 1
        return snapshot_obj if snapshot_call_count_list[0] == 1 else broken_snapshot_obj
    monkeypatch.setattr(broker_obj, "get_core5_account_snapshot", account_snapshot)
    result_dict = post_execution_reconcile(store_obj, broker_obj, as_of_ts, release_obj.mode_str, **runner_kwarg_dict)
    assert result_dict["completed_vplan_count_int"] == 0
    assert store_obj.get_pod_state(release_obj.pod_id_str) == prior_state_obj
    assert store_obj.get_vplan_by_id(vplan_obj.vplan_id_int).status_str == "submitted"
