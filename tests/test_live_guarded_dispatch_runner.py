"""Runner dispatch recovery uses real temporary SQLite and offline broker doubles."""
from dataclasses import replace
from datetime import timedelta
import json
from types import SimpleNamespace

import pytest

from alpha.live import logging_utils, runner
from alpha.live.execution_engine import build_broker_order_request_list_from_vplan
from alpha.live.execution_resolution import load_request_resolution_dict
from alpha.live.state_store_v2 import LiveStateStore
from test_live_core5_wiring import (
    core5_price_df, qualified_release_obj, _prepared_cycle, _release_for_mode,
)
from test_live_ibkr_socket_client import _ExpirySubmitIB, _expiry_client_obj
from test_live_mr_capsule_recovery import capsule_case


def _freeze_dispatch_clock(monkeypatch, clock_list):
    class DispatchClock:
        @classmethod
        def now(cls, timezone_obj):
            return clock_list[0].astimezone(timezone_obj)
    monkeypatch.setattr("alpha.live.dispatch_state.datetime", DispatchClock)


@pytest.fixture
def core5_case(tmp_path, monkeypatch, qualified_release_obj, core5_price_df):
    release_obj = _release_for_mode(qualified_release_obj, "paper")
    monkeypatch.setattr(logging_utils, "DEFAULT_CRITICAL_LOG_PATH_STR", str(tmp_path / "critical.jsonl"))
    store_obj, broker_obj, decision_obj, vplan_obj, option_dict = _prepared_cycle(
        tmp_path, monkeypatch, release_obj, core5_price_df, "long_to_short")
    clock_list = [decision_obj.submission_timestamp_ts]
    _freeze_dispatch_clock(monkeypatch, clock_list)
    return SimpleNamespace(store_obj=store_obj, broker_obj=broker_obj, release_obj=release_obj,
        decision_obj=decision_obj, vplan_obj=vplan_obj, option_dict=option_dict, clock_list=clock_list)


def _submit(case_obj, as_of_ts=None):
    return runner.submit_ready_vplans(case_obj.store_obj, case_obj.broker_obj,
        as_of_ts or case_obj.decision_obj.submission_timestamp_ts, "paper", False,
        vplan_id_int=case_obj.vplan_obj.vplan_id_int, **case_obj.option_dict)


def _saved_failure(case_obj):
    saved_decision_obj = case_obj.store_obj.get_decision_plan_by_id(case_obj.decision_obj.decision_plan_id_int)
    return saved_decision_obj.snapshot_metadata_dict["opening_dispatch_result_dict"]


def _assert_critical_terminal_requests(case_obj, expected_key_set):
    resolution_dict = load_request_resolution_dict(case_obj.store_obj, case_obj.vplan_obj)
    assert set(resolution_dict) == expected_key_set
    assert all(row_dict["resolution_str"] == "never_dispatched" for row_dict in resolution_dict.values())
    saved_decision_obj = case_obj.store_obj.get_decision_plan_by_id(case_obj.decision_obj.decision_plan_id_int)
    assert saved_decision_obj.snapshot_metadata_dict["opening_dispatch_parked_bool"] is True
    with case_obj.store_obj._connect() as connection_obj:
        alert_row_obj, = connection_obj.execute("SELECT * FROM mr_capsule_execution_alert WHERE vplan_id_int=?",
            (case_obj.vplan_obj.vplan_id_int,)).fetchall()
    assert alert_row_obj["alert_kind_str"] == "dispatch_failed"
    assert json.loads(alert_row_obj["payload_json_str"])["severity_str"] == "critical"


@pytest.mark.parametrize("stage_str", ["account", "funding", "trace"])
def test_transient_errors_before_place_order_remain_retryable_until_cutoff(core5_case, monkeypatch, stage_str):
    case_obj = core5_case
    if stage_str == "account":
        original_fn = case_obj.broker_obj.get_core5_account_snapshot
        owner_obj, name_str = case_obj.broker_obj, "get_core5_account_snapshot"
    elif stage_str == "funding":
        original_fn = case_obj.broker_obj.get_core5_funding_evidence
        owner_obj, name_str = case_obj.broker_obj, "get_core5_funding_evidence"
    else:
        original_fn = runner._emit_live_trace_event
        owner_obj, name_str = runner, "_emit_live_trace_event"
    def failing_fn(*argument_tuple, **argument_dict):
        if stage_str != "trace" or argument_tuple[0] == "vplan.submit_request":
            raise TimeoutError("synthetic transient before placeOrder")
        return original_fn(*argument_tuple, **argument_dict)
    monkeypatch.setattr(owner_obj, name_str, failing_fn)
    first_dict = _submit(case_obj)
    assert first_dict["submitted_vplan_count_int"] == 0
    assert _saved_failure(case_obj)["reason_code_str"] == "dispatch_retry_pending"
    assert case_obj.store_obj.get_vplan_by_id(case_obj.vplan_obj.vplan_id_int).status_str == "ready"
    assert load_request_resolution_dict(case_obj.store_obj, case_obj.vplan_obj) == {}
    assert not case_obj.broker_obj.submitted_order_request_list
    monkeypatch.setattr(owner_obj, name_str, original_fn)
    second_dict = _submit(case_obj)
    assert second_dict["submitted_vplan_count_int"] == 1
    assert len(case_obj.broker_obj.submitted_order_request_list) == len(build_broker_order_request_list_from_vplan(case_obj.vplan_obj))


def test_trace_failure_between_claim_and_send_marks_all_legs_terminal(core5_case, monkeypatch):
    case_obj = core5_case
    original_fn = runner._emit_live_trace_event
    def trace_failure_fn(*argument_tuple, **argument_dict):
        if argument_tuple[0] == "vplan.submit_request":
            assert case_obj.store_obj.get_vplan_by_id(case_obj.vplan_obj.vplan_id_int).status_str == "submitting"
            raise ValueError("synthetic audit serialization failure")
        return original_fn(*argument_tuple, **argument_dict)
    monkeypatch.setattr(runner, "_emit_live_trace_event", trace_failure_fn)
    assert _submit(case_obj)["submitted_vplan_count_int"] == 0
    request_key_set = {request_obj.order_request_key_str for request_obj in build_broker_order_request_list_from_vplan(case_obj.vplan_obj)}
    _assert_critical_terminal_requests(case_obj, request_key_set)
    assert case_obj.store_obj.get_vplan_by_id(case_obj.vplan_obj.vplan_id_int).status_str == "submitted"
    assert not case_obj.broker_obj.submitted_order_request_list
    case_obj.store_obj = LiveStateStore(case_obj.store_obj.db_path_str)
    assert _submit(case_obj)["submitted_vplan_count_int"] == 0
    assert not case_obj.broker_obj.submitted_order_request_list


def test_typed_socket_presend_timeout_releases_claim_for_one_safe_retry(core5_case, monkeypatch):
    case_obj = core5_case
    socket_broker_obj = _ExpirySubmitIB(case_obj.clock_list)
    socket_obj = _expiry_client_obj(monkeypatch, socket_broker_obj)
    qualification_fn = socket_broker_obj.qualifyContracts
    def timeout_fn(*contract_list):
        raise TimeoutError("qualification timeout before any dispatch")
    monkeypatch.setattr(socket_broker_obj, "qualifyContracts", timeout_fn)
    monkeypatch.setattr(case_obj.broker_obj, "submit_order_request_list", socket_obj.submit_order_request_list)
    assert _submit(case_obj)["submitted_vplan_count_int"] == 0
    assert case_obj.store_obj.get_vplan_by_id(case_obj.vplan_obj.vplan_id_int).status_str == "ready"
    assert _saved_failure(case_obj)["error_type_str"] == "TimeoutError"
    assert _saved_failure(case_obj)["attempted_request_key_list"] == []
    assert load_request_resolution_dict(case_obj.store_obj, case_obj.vplan_obj) == {}
    assert socket_broker_obj.placed_order_list == []
    monkeypatch.setattr(socket_broker_obj, "qualifyContracts", qualification_fn)
    assert _submit(case_obj)["submitted_vplan_count_int"] == 1
    assert len(socket_broker_obj.placed_order_list) == len(build_broker_order_request_list_from_vplan(case_obj.vplan_obj))


def test_partial_socket_send_persists_progress_and_never_replays_after_restart(core5_case, monkeypatch):
    case_obj = core5_case
    socket_broker_obj = _ExpirySubmitIB(case_obj.clock_list)
    socket_obj = _expiry_client_obj(monkeypatch, socket_broker_obj)
    place_fn = socket_broker_obj.placeOrder
    def fail_second_fn(contract_obj, order_obj):
        if socket_broker_obj.placed_order_list:
            raise TimeoutError("second order may have reached broker")
        return place_fn(contract_obj, order_obj)
    monkeypatch.setattr(socket_broker_obj, "placeOrder", fail_second_fn)
    monkeypatch.setattr(case_obj.broker_obj, "submit_order_request_list", socket_obj.submit_order_request_list)
    assert _submit(case_obj)["submitted_vplan_count_int"] == 0
    request_list = build_broker_order_request_list_from_vplan(case_obj.vplan_obj)
    _assert_critical_terminal_requests(case_obj, {request_obj.order_request_key_str for request_obj in request_list[2:]})
    assert _saved_failure(case_obj)["attempted_request_key_list"] == [request_obj.order_request_key_str for request_obj in request_list[:2]]
    assert case_obj.store_obj.count_broker_orders_for_vplan(case_obj.vplan_obj.vplan_id_int) == 1
    assert case_obj.store_obj.get_vplan_by_id(case_obj.vplan_obj.vplan_id_int).status_str == "submitted"
    assert len(socket_broker_obj.placed_order_list) == 1
    case_obj.store_obj = LiveStateStore(case_obj.store_obj.db_path_str)
    assert _submit(case_obj)["submitted_vplan_count_int"] == 0
    assert len(socket_broker_obj.placed_order_list) == 1


@pytest.mark.parametrize("typed_bool", [False, True])
def test_wall_clock_crossing_cutoff_makes_transient_failure_terminal(core5_case, monkeypatch, typed_bool):
    case_obj = core5_case
    cutoff_ts = case_obj.vplan_obj.target_execution_timestamp_ts - timedelta(minutes=2)
    def slow_timeout_fn(*argument_tuple, **argument_dict):
        case_obj.clock_list[:] = [cutoff_ts]
        raise TimeoutError("preflight crossed dispatch deadline")
    if typed_bool:
        socket_broker_obj = _ExpirySubmitIB(case_obj.clock_list)
        socket_obj = _expiry_client_obj(monkeypatch, socket_broker_obj)
        monkeypatch.setattr(socket_broker_obj, "qualifyContracts", slow_timeout_fn)
        monkeypatch.setattr(case_obj.broker_obj, "submit_order_request_list", socket_obj.submit_order_request_list)
    else:
        monkeypatch.setattr(case_obj.broker_obj, "get_core5_funding_evidence", slow_timeout_fn)
    assert _submit(case_obj)["submitted_vplan_count_int"] == 0
    _assert_critical_terminal_requests(case_obj,
        {request_obj.order_request_key_str for request_obj in build_broker_order_request_list_from_vplan(case_obj.vplan_obj)})
    assert _saved_failure(case_obj)["reason_code_str"] == "opening_dispatch_parked"
    assert case_obj.store_obj.get_vplan_by_id(case_obj.vplan_obj.vplan_id_int).status_str == ("submitted" if typed_bool else "blocked")


def test_core5_cutoff_prevents_preflight_and_terminalizes_every_unsent_request(core5_case, monkeypatch):
    case_obj = core5_case
    cutoff_ts = case_obj.vplan_obj.target_execution_timestamp_ts - timedelta(minutes=2)
    monkeypatch.setattr(case_obj.broker_obj, "get_core5_funding_evidence", lambda *argument_tuple: pytest.fail("cutoff reached funding"))
    monkeypatch.setattr(case_obj.store_obj, "claim_vplan_for_submission", lambda *argument_tuple: pytest.fail("cutoff claimed dispatch"))
    assert _submit(case_obj, cutoff_ts)["submitted_vplan_count_int"] == 0
    _assert_critical_terminal_requests(case_obj,
        {request_obj.order_request_key_str for request_obj in build_broker_order_request_list_from_vplan(case_obj.vplan_obj)})
    assert case_obj.broker_obj.submitted_order_request_list == []


def test_capsule_cutoff_uses_same_deadline_and_no_broker_dispatch(capsule_case, monkeypatch):
    store_obj, broker_obj, release_obj, vplan_obj, option_dict, _ = capsule_case
    decision_obj = store_obj.get_decision_plan_by_id(vplan_obj.decision_plan_id_int)
    case_obj = SimpleNamespace(store_obj=store_obj, broker_obj=broker_obj, release_obj=release_obj,
        vplan_obj=vplan_obj, decision_obj=decision_obj, option_dict=option_dict)
    cutoff_ts = vplan_obj.target_execution_timestamp_ts - timedelta(minutes=2)
    _freeze_dispatch_clock(monkeypatch, [cutoff_ts])
    monkeypatch.setattr(broker_obj, "get_capsule_funding_evidence", lambda *argument_tuple: pytest.fail("cutoff reached funding"))
    assert _submit(case_obj, cutoff_ts)["submitted_vplan_count_int"] == 0
    _assert_critical_terminal_requests(case_obj,
        {request_obj.order_request_key_str for request_obj in build_broker_order_request_list_from_vplan(vplan_obj)})
    assert broker_obj.submitted_order_request_list == []


def test_capsule_transient_funding_retry_retains_buy_and_sell(capsule_case, monkeypatch):
    store_obj, broker_obj, release_obj, vplan_obj, option_dict, _ = capsule_case
    decision_obj = store_obj.get_decision_plan_by_id(vplan_obj.decision_plan_id_int)
    case_obj = SimpleNamespace(store_obj=store_obj, broker_obj=broker_obj, release_obj=release_obj,
        vplan_obj=vplan_obj, decision_obj=decision_obj, option_dict=option_dict)
    _freeze_dispatch_clock(monkeypatch, [decision_obj.submission_timestamp_ts])
    funding_fn = broker_obj.get_capsule_funding_evidence
    def transient_fn(*argument_tuple):
        raise TimeoutError("temporary margin preview timeout")
    monkeypatch.setattr(broker_obj, "get_capsule_funding_evidence", transient_fn)
    assert _submit(case_obj)["submitted_vplan_count_int"] == 0
    assert store_obj.get_vplan_by_id(vplan_obj.vplan_id_int).status_str == "ready"
    assert load_request_resolution_dict(store_obj, vplan_obj) == {}
    monkeypatch.setattr(broker_obj, "get_capsule_funding_evidence", funding_fn)
    assert _submit(case_obj)["submitted_vplan_count_int"] == 1
    assert [(request_obj.asset_str, request_obj.amount_float) for request_obj in broker_obj.submitted_order_request_list] == [("BIL", -100.0), ("AAPL", 80.0)]


def test_core5_quotes_and_changed_nav_cash_do_not_block_or_resize_frozen_targets(
        tmp_path, monkeypatch, qualified_release_obj, core5_price_df):
    from alpha.live.order_clerk import StubBrokerAdapter

    monkeypatch.setattr(StubBrokerAdapter, "get_live_price_snapshot",
        lambda *argument_tuple, **argument_dict: pytest.fail("CORE5 requested a blocking live quote"))
    release_obj = _release_for_mode(qualified_release_obj, "paper")
    store_obj, broker_obj, decision_obj, vplan_obj, option_dict = _prepared_cycle(
        tmp_path, monkeypatch, release_obj, core5_price_df)
    expected_request_list = build_broker_order_request_list_from_vplan(vplan_obj)
    original_target_dict = dict(vplan_obj.target_share_map)
    original_snapshot_obj = broker_obj.get_account_snapshot(release_obj.account_route_str)
    broker_obj._snapshot_map[release_obj.account_route_str] = replace(original_snapshot_obj,
        cash_float=original_snapshot_obj.cash_float + 1_000_000.0,
        net_liq_float=original_snapshot_obj.net_liq_float * 7,
        total_value_float=original_snapshot_obj.total_value_float * 7)
    _freeze_dispatch_clock(monkeypatch, [decision_obj.submission_timestamp_ts])
    case_obj = SimpleNamespace(store_obj=store_obj, broker_obj=broker_obj, release_obj=release_obj,
        vplan_obj=vplan_obj, decision_obj=decision_obj, option_dict=option_dict)
    assert _submit(case_obj)["submitted_vplan_count_int"] == 1
    assert [replace(request_obj, submission_deadline_timestamp_str=None) for request_obj in broker_obj.submitted_order_request_list] == expected_request_list
    assert store_obj.get_vplan_by_id(vplan_obj.vplan_id_int).target_share_map == original_target_dict
    assert vplan_obj.live_price_source_str == "core5.frozen_close_t"


@pytest.mark.parametrize("cutoff_phase_str", ["after_qualification", "between_orders"])
def test_capsule_socket_checks_deadline_immediately_before_every_order(capsule_case, monkeypatch, cutoff_phase_str):
    store_obj, broker_obj, release_obj, vplan_obj, option_dict, _ = capsule_case
    decision_obj = store_obj.get_decision_plan_by_id(vplan_obj.decision_plan_id_int)
    case_obj = SimpleNamespace(store_obj=store_obj, broker_obj=broker_obj, release_obj=release_obj,
        vplan_obj=vplan_obj, decision_obj=decision_obj, option_dict=option_dict)
    cutoff_ts = vplan_obj.target_execution_timestamp_ts - timedelta(minutes=2)
    clock_list = [decision_obj.submission_timestamp_ts]
    _freeze_dispatch_clock(monkeypatch, clock_list)
    socket_broker_obj = _ExpirySubmitIB(clock_list)
    socket_obj = _expiry_client_obj(monkeypatch, socket_broker_obj)
    if cutoff_phase_str == "after_qualification":
        socket_broker_obj.after_qualification_ts = cutoff_ts
    else:
        place_fn = socket_broker_obj.placeOrder
        def cross_cutoff_fn(contract_obj, order_obj):
            trade_obj = place_fn(contract_obj, order_obj)
            clock_list[:] = [cutoff_ts]
            return trade_obj
        monkeypatch.setattr(socket_broker_obj, "placeOrder", cross_cutoff_fn)
    monkeypatch.setattr(broker_obj, "submit_order_request_list", socket_obj.submit_order_request_list)
    assert _submit(case_obj)["submitted_vplan_count_int"] == 0
    request_list = build_broker_order_request_list_from_vplan(vplan_obj)
    sent_key_set = {order_obj.orderRef for order_obj in socket_broker_obj.placed_order_list}
    _assert_critical_terminal_requests(case_obj,
        {request_obj.order_request_key_str for request_obj in request_list} - sent_key_set)
    assert store_obj.count_broker_orders_for_vplan(vplan_obj.vplan_id_int) == len(sent_key_set)
    if cutoff_phase_str == "after_qualification":
        assert sent_key_set == set()
    else:
        assert sent_key_set == {request_obj.order_request_key_str for request_obj in request_list if request_obj.asset_str == "BIL"}
        assert socket_broker_obj.placed_order_list[0].tif == "OPG"
    assert _submit(case_obj)["submitted_vplan_count_int"] == 0
    assert {order_obj.orderRef for order_obj in socket_broker_obj.placed_order_list} == sent_key_set
