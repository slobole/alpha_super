"""CORE5 deployment gates and lifecycle with fake accounts only; no broker access."""
from dataclasses import asdict, replace
from datetime import UTC, datetime, timedelta
import json

import pytest
import yaml

from alpha.live import reference_compare, runner, scheduler_service, scheduler_utils
from alpha.live.execution_engine import build_broker_order_request_list_from_vplan
from alpha.live.order_clerk import BrokerAdapter
from alpha.live.release_manifest import load_release_list, validate_release_list, validate_release_manifest
from alpha.live.state_store_v2 import LiveStateStore
import test_live_core5_adapter as core5_helper
from test_live_ibkr_socket_client import _ExpirySubmitIB, _expiry_client_obj


@pytest.fixture(autouse=True)
def historical_dispatch_clock(monkeypatch):
    class DispatchClock:
        @classmethod
        def now(cls, timezone_obj):
            return datetime(2026, 9, 14, 13, 22, tzinfo=UTC).astimezone(timezone_obj)
    monkeypatch.setattr("alpha.live.dispatch_state.datetime", DispatchClock)


@pytest.fixture
def core5_price_df():
    return core5_helper.price_df.__wrapped__()


@pytest.fixture
def qualified_release_obj():
    release_obj = replace(core5_helper.release_obj.__wrapped__(),
        mode_str="live", account_route_str="U_CORE5_TEST", broker_port_int=7496)
    qualification_dict = {
        "release_id_str": release_obj.release_id_str,
        "account_route_str": release_obj.account_route_str,
        "evidence_reference_str": "synthetic-test-only:no-real-account-qualified",
        "approved_at_str": "2026-01-01T00:00:00+00:00",
        "expires_at_str": "2099-01-01T00:00:00+00:00",
        "margin_account_confirmed_bool": True,
        "dbc_borrow_and_recall_policy_confirmed_bool": True,
        "forward_execution_qualified_bool": True,
        "operator_approved_bool": True,
    }
    return replace(release_obj, params_dict={"core5_live_qualification_dict": qualification_dict})


def _changed_qualification(release_obj, **change_dict):
    qualification_dict = dict(release_obj.params_dict["core5_live_qualification_dict"])
    qualification_dict.update(change_dict)
    return replace(release_obj, params_dict={"core5_live_qualification_dict": qualification_dict})


def _release_for_mode(release_obj, mode_str):
    return release_obj if mode_str == "live" else replace(release_obj,
        mode_str="paper", account_route_str="DU_CORE5_TEST", params_dict={}, broker_port_int=7497)


def _write_release_yaml(tmp_path, release_obj):
    release_dir_path = tmp_path / "releases"
    release_dir_path.mkdir()
    payload_dict = asdict(release_obj)
    for source_str, target_str in (("release_id_str", "release_id"), ("user_id_str", "user_id"),
            ("pod_id_str", "pod_id"), ("account_route_str", "account_route"),
            ("mode_str", "mode"), ("params_dict", "params")):
        payload_dict[target_str] = payload_dict.pop(source_str)
    (release_dir_path / "core5.yaml").write_text(yaml.safe_dump(payload_dict), encoding="utf-8")
    return release_dir_path


def _unqualified_release(release_obj, qualification_state_str):
    return _changed_qualification(release_obj, **(
        {"approved_at_str": "2000-01-01T00:00:00+00:00", "expires_at_str": "2001-01-01T00:00:00+00:00"}
        if qualification_state_str == "expired" else {"operator_approved_bool": False}))


def _prepared_cycle(tmp_path, monkeypatch, release_obj, price_df, cycle_str="initialization"):
    change_dict = {}
    if cycle_str != "initialization":
        previous_long_float = 1.0 if cycle_str == "long_to_short" else 0.0
        for date_str, long_float in (("2026-09-09", previous_long_float),
                ("2026-09-10", previous_long_float), ("2026-09-11", 1.0 - previous_long_float)):
            change_dict[(date_str, "DBC", "long_state_ser")] = long_float
            change_dict[(date_str, "DBC", "short_state_ser")] = 1.0 - long_float
    core5_helper._controlled_signals(monkeypatch, price_df, change_dict)
    state_obj = core5_helper._state(release_obj, "2026-09-11")
    if cycle_str != "initialization":
        prior_decision_obj = core5_helper._build(release_obj, price_df, "2026-09-10",
            core5_helper._state(release_obj, "2026-09-10"))
        state_obj = core5_helper._state(release_obj, "2026-09-11",
            prior_decision_obj.snapshot_metadata_dict["fixed_target_share_map_dict"],
            prior_decision_obj.strategy_state_dict)
    store_obj, broker_obj = core5_helper._store_and_broker(tmp_path, release_obj, state_obj)
    decision_obj = store_obj.insert_decision_plan(core5_helper._build(
        release_obj, price_df, "2026-09-11", state_obj))
    broker_obj.seed_account_snapshot(release_obj.account_route_str, state_obj.cash_float,
        100_000.0, state_obj.position_amount_map, decision_obj.submission_timestamp_ts,
        session_mode_str=release_obj.mode_str)
    runner_kwarg_dict = {"log_path_str": str(tmp_path / "ops.log"), "trace_enabled_bool": False}
    result_dict = runner.build_vplans(store_obj, broker_obj, decision_obj.submission_timestamp_ts,
        release_obj.mode_str, **runner_kwarg_dict)
    assert result_dict["created_vplan_count_int"] == 1
    vplan_obj = store_obj.get_latest_vplan_for_pod(release_obj.pod_id_str)
    return store_obj, broker_obj, decision_obj, vplan_obj, runner_kwarg_dict


def test_disabled_live_template_is_reviewable_but_cannot_build(qualified_release_obj, core5_price_df):
    disabled_release_obj = replace(qualified_release_obj, enabled_bool=False, params_dict={})
    validate_release_manifest(disabled_release_obj)
    with pytest.raises(ValueError):
        core5_helper._build(disabled_release_obj, core5_price_df, "2026-09-11",
            core5_helper._state(disabled_release_obj, "2026-09-11"))


def test_enabled_live_requires_qualification(qualified_release_obj):
    validate_release_manifest(qualified_release_obj)
    with pytest.raises(ValueError):
        validate_release_manifest(replace(qualified_release_obj, params_dict={}))


@pytest.mark.parametrize("field_str,value_obj", [
    ("release_id_str", "another.release"), ("account_route_str", "U_ANOTHER"),
    ("evidence_reference_str", "  "), ("margin_account_confirmed_bool", False),
    ("dbc_borrow_and_recall_policy_confirmed_bool", False),
    ("forward_execution_qualified_bool", False), ("operator_approved_bool", False),
    ("operator_approved_bool", "true"), ("operator_approved_bool", 1),
    ("approved_at_str", "2026-01-01T00:00:00"),
    ("approved_at_str", "2100-01-01T00:00:00+00:00"),
    ("expires_at_str", "not-a-timestamp"), ("expires_at_str", "2099-01-01T00:00:00"),
    ("expires_at_str", "2000-01-01T00:00:00+00:00"),
])
def test_live_qualification_fails_closed(qualified_release_obj, field_str, value_obj):
    with pytest.raises(ValueError):
        validate_release_manifest(_changed_qualification(qualified_release_obj, **{field_str: value_obj}))


def test_qualification_does_not_admit_wrong_account_state(qualified_release_obj, core5_price_df):
    state_obj = replace(core5_helper._state(qualified_release_obj, "2026-09-11"),
        account_route_str="U_ANOTHER")
    with pytest.raises(ValueError):
        core5_helper._build(qualified_release_obj, core5_price_df, "2026-09-11", state_obj)


@pytest.mark.parametrize("cycle_str", ["initialization", "long_to_short", "short_to_long"])
@pytest.mark.parametrize("mode_str", ["paper", "live"])
def test_qualified_live_fake_cycle_preserves_intent_and_commits_after_fills(
        qualified_release_obj, core5_price_df, tmp_path, monkeypatch, cycle_str, mode_str):
    release_obj = _release_for_mode(qualified_release_obj, mode_str)
    store_obj, broker_obj, decision_obj, vplan_obj, runner_kwarg_dict = _prepared_cycle(
        tmp_path, monkeypatch, release_obj, core5_price_df, cycle_str)
    expected_request_list = build_broker_order_request_list_from_vplan(vplan_obj)
    phase_list = []
    original_funding_fn = broker_obj.get_core5_funding_evidence
    original_claim_fn = store_obj.claim_vplan_for_submission

    def funding_fn(account_route_str, request_list, current_position_dict):
        phase_list.append("funding")
        assert account_route_str == release_obj.account_route_str
        assert request_list == expected_request_list
        assert current_position_dict == vplan_obj.current_broker_position_map
        assert store_obj.get_vplan_by_id(vplan_obj.vplan_id_int).status_str == "ready"
        return original_funding_fn(account_route_str, request_list, current_position_dict)

    def claim_fn(vplan_id_int):
        phase_list.append("claim")
        return original_claim_fn(vplan_id_int)

    monkeypatch.setattr(broker_obj, "get_core5_funding_evidence", funding_fn)
    monkeypatch.setattr(store_obj, "claim_vplan_for_submission", claim_fn)
    prior_state_dict = dict(store_obj.get_pod_state(release_obj.pod_id_str).strategy_state_dict)
    result_dict = runner.submit_ready_vplans(store_obj, broker_obj, decision_obj.submission_timestamp_ts,
        mode_str, False, vplan_id_int=vplan_obj.vplan_id_int, **runner_kwarg_dict)
    assert result_dict["submitted_vplan_count_int"] == 1
    assert phase_list == ["funding", "claim"]
    deadline_str = (vplan_obj.target_execution_timestamp_ts - timedelta(minutes=2)).isoformat()
    assert broker_obj.submitted_order_request_list == [replace(request_obj,
        submission_deadline_timestamp_str=deadline_str) for request_obj in expected_request_list]
    assert all(request_obj.submission_deadline_timestamp_str == deadline_str
        for request_obj in broker_obj.submitted_order_request_list)
    assert all(request_obj.broker_order_type_str == "MOO" and request_obj.unit_str == "shares"
        and request_obj.target_bool is False for request_obj in expected_request_list)
    assert store_obj.get_pod_state(release_obj.pod_id_str).strategy_state_dict == prior_state_dict
    if cycle_str != "initialization":
        dbc_request_list = [request_obj for request_obj in expected_request_list if request_obj.asset_str == "DBC"]
        assert len(dbc_request_list) == 2
        assert dbc_request_list[0].amount_float == -vplan_obj.current_broker_position_map["DBC"]
        assert dbc_request_list[1].amount_float == vplan_obj.target_share_map["DBC"]
        assert dbc_request_list[0].order_request_key_str != dbc_request_list[1].order_request_key_str
    restarted_store_obj = LiveStateStore(str(tmp_path / "core5.sqlite3"))
    reconcile_ts = decision_obj.target_execution_timestamp_ts + timedelta(minutes=10)
    broker_obj._snapshot_map[release_obj.account_route_str] = replace(
        broker_obj.get_account_snapshot(release_obj.account_route_str), snapshot_timestamp_ts=reconcile_ts)
    result_dict = runner.post_execution_reconcile(restarted_store_obj, broker_obj,
        reconcile_ts, mode_str, **runner_kwarg_dict)
    assert result_dict["completed_vplan_count_int"] == 1
    assert restarted_store_obj.get_pod_state(release_obj.pod_id_str).strategy_state_dict == decision_obj.strategy_state_dict
    assert restarted_store_obj.get_latest_decision_plan_for_pod(release_obj.pod_id_str).status_str == "completed"


@pytest.mark.parametrize("error_obj", [ValueError("Borrow unavailable"),
    TimeoutError("Margin preview timed out"), NotImplementedError("Funding unsupported")])
@pytest.mark.parametrize("mode_str", ["paper", "live"])
def test_unverified_funding_never_claims_or_submits(
        qualified_release_obj, core5_price_df, tmp_path, monkeypatch, error_obj, mode_str):
    release_obj = _release_for_mode(qualified_release_obj, mode_str)
    store_obj, broker_obj, decision_obj, vplan_obj, runner_kwarg_dict = _prepared_cycle(
        tmp_path, monkeypatch, release_obj, core5_price_df, "long_to_short")
    prior_state_dict = dict(store_obj.get_pod_state(release_obj.pod_id_str).strategy_state_dict)

    def failed_funding_fn(*argument_tuple):
        raise error_obj

    monkeypatch.setattr(broker_obj, "get_core5_funding_evidence", failed_funding_fn)
    monkeypatch.setattr(store_obj, "claim_vplan_for_submission",
        lambda *_argument_tuple: pytest.fail("Unverified funding claimed a batch"))
    result_dict = runner.submit_ready_vplans(store_obj, broker_obj, decision_obj.submission_timestamp_ts,
        mode_str, False, vplan_id_int=vplan_obj.vplan_id_int, **runner_kwarg_dict)
    assert result_dict["submitted_vplan_count_int"] == 0
    assert not broker_obj.submitted_order_request_list
    assert store_obj.get_vplan_by_id(vplan_obj.vplan_id_int).status_str == (
        "ready" if isinstance(error_obj, TimeoutError) else "blocked")
    assert store_obj.get_pod_state(release_obj.pod_id_str).strategy_state_dict == prior_state_dict


def test_base_adapter_cannot_invent_core5_funding_evidence():
    with pytest.raises(NotImplementedError):
        BrokerAdapter.get_core5_funding_evidence(object(), "U_CORE5_TEST", [], {})


def test_failed_paper_preflight_cannot_downgrade_another_process_submission_claim(
        qualified_release_obj, core5_price_df, tmp_path, monkeypatch):
    release_obj = _release_for_mode(qualified_release_obj, "paper")
    store_obj, broker_obj, decision_obj, vplan_obj, runner_kwarg_dict = _prepared_cycle(
        tmp_path, monkeypatch, release_obj, core5_price_df)
    other_store_obj = LiveStateStore(str(tmp_path / "core5.sqlite3"))
    prior_state_dict = dict(store_obj.get_pod_state(release_obj.pod_id_str).strategy_state_dict)

    def competing_claim_then_failure_fn(*argument_tuple):
        assert other_store_obj.claim_vplan_for_submission(vplan_obj.vplan_id_int)
        raise TimeoutError("This worker lost its margin preview after another worker claimed")

    monkeypatch.setattr(broker_obj, "get_core5_funding_evidence", competing_claim_then_failure_fn)
    monkeypatch.setattr(store_obj, "claim_vplan_for_submission",
        lambda *_argument_tuple: pytest.fail("Failed worker tried to claim"))
    result_dict = runner.submit_ready_vplans(store_obj, broker_obj, decision_obj.submission_timestamp_ts,
        "paper", False, vplan_id_int=vplan_obj.vplan_id_int, **runner_kwarg_dict)
    assert result_dict["submitted_vplan_count_int"] == 0
    assert other_store_obj.get_vplan_by_id(vplan_obj.vplan_id_int).status_str == "submitting"
    assert other_store_obj.get_decision_plan_by_id(decision_obj.decision_plan_id_int).status_str == "vplan_ready"
    assert other_store_obj.get_pod_state(release_obj.pod_id_str).strategy_state_dict == prior_state_dict
    assert not broker_obj.submitted_order_request_list


def test_paper_restart_after_claim_reconciles_without_expiring_the_pending_cycle(
        qualified_release_obj, core5_price_df, tmp_path, monkeypatch):
    release_obj = _release_for_mode(qualified_release_obj, "paper")
    store_obj, broker_obj, decision_obj, vplan_obj, runner_kwarg_dict = _prepared_cycle(
        tmp_path, monkeypatch, release_obj, core5_price_df)
    assert store_obj.claim_vplan_for_submission(vplan_obj.vplan_id_int)
    restarted_store_obj = LiveStateStore(str(tmp_path / "core5.sqlite3"))
    after_open_ts = decision_obj.target_execution_timestamp_ts + timedelta(minutes=10)
    assert restarted_store_obj.get_decision_plan_by_id(decision_obj.decision_plan_id_int).status_str == "vplan_ready"
    assert restarted_store_obj.get_vplan_by_id(vplan_obj.vplan_id_int).status_str == "submitting"

    result_dict = runner.expire_stale_decision_plans(restarted_store_obj, after_open_ts,
        "unused", "paper", **runner_kwarg_dict)
    assert result_dict["expired_decision_plan_count_int"] == 0
    result_dict = runner.submit_ready_vplans(restarted_store_obj, broker_obj, after_open_ts,
        "paper", False, vplan_id_int=vplan_obj.vplan_id_int, **runner_kwarg_dict)
    assert result_dict["submitted_vplan_count_int"] == 0
    assert not broker_obj.submitted_order_request_list
    assert restarted_store_obj.get_decision_plan_by_id(decision_obj.decision_plan_id_int).status_str == "vplan_ready"
    assert restarted_store_obj.get_vplan_by_id(vplan_obj.vplan_id_int).status_str == "submitting"

    monkeypatch.setattr(scheduler_service, "_load_release_list_and_sync",
        lambda *_argument_tuple, **_argument_dict: [release_obj])
    monkeypatch.setattr(scheduler_utils, "evaluate_build_gate_dict",
        lambda *_argument_tuple, **_argument_dict: {"due_bool": True})
    schedule_obj = scheduler_service.get_scheduler_decision(restarted_store_obj,
        after_open_ts, "unused", "paper")
    assert schedule_obj.next_phase_str == "post_execution_reconcile"
    assert schedule_obj.reason_code_str == "ready_to_reconcile"


@pytest.mark.parametrize("evidence_phase_str", ["submission", "cutoff", "after_cutoff"])
def test_paper_funding_completion_must_precede_opening_order_cutoff(
        qualified_release_obj, core5_price_df, tmp_path, monkeypatch, evidence_phase_str):
    release_obj = _release_for_mode(qualified_release_obj, "paper")
    store_obj, broker_obj, decision_obj, vplan_obj, runner_kwarg_dict = _prepared_cycle(
        tmp_path, monkeypatch, release_obj, core5_price_df)
    cutoff_ts = decision_obj.target_execution_timestamp_ts - timedelta(minutes=2)
    checked_ts = decision_obj.submission_timestamp_ts if evidence_phase_str == "submission" else (
        cutoff_ts if evidence_phase_str == "cutoff" else cutoff_ts + timedelta(seconds=1))
    monkeypatch.setattr(broker_obj, "get_core5_funding_evidence", lambda *_argument_tuple: {
        "simulated_bool": True, "checked_at_str": checked_ts.isoformat()})
    if evidence_phase_str != "submission":
        monkeypatch.setattr(store_obj, "claim_vplan_for_submission",
            lambda *_argument_tuple: pytest.fail("Late funding evidence claimed a batch"))
    result_dict = runner.submit_ready_vplans(store_obj, broker_obj, decision_obj.submission_timestamp_ts,
        "paper", False, vplan_id_int=vplan_obj.vplan_id_int, **runner_kwarg_dict)
    assert result_dict["submitted_vplan_count_int"] == int(evidence_phase_str == "submission")
    assert bool(broker_obj.submitted_order_request_list) == (evidence_phase_str == "submission")
    if evidence_phase_str != "submission":
        assert store_obj.get_vplan_by_id(vplan_obj.vplan_id_int).status_str == "blocked"


def test_paper_submit_at_cutoff_never_requests_funding_or_claims(
        qualified_release_obj, core5_price_df, tmp_path, monkeypatch):
    release_obj = _release_for_mode(qualified_release_obj, "paper")
    store_obj, broker_obj, decision_obj, vplan_obj, runner_kwarg_dict = _prepared_cycle(
        tmp_path, monkeypatch, release_obj, core5_price_df)
    monkeypatch.setattr(broker_obj, "get_core5_funding_evidence",
        lambda *_argument_tuple: pytest.fail("Cutoff guard ran after funding"))
    monkeypatch.setattr(store_obj, "claim_vplan_for_submission",
        lambda *_argument_tuple: pytest.fail("Cutoff guard allowed a submission claim"))
    result_dict = runner.submit_ready_vplans(store_obj, broker_obj,
        decision_obj.target_execution_timestamp_ts - timedelta(minutes=2),
        "paper", False, vplan_id_int=vplan_obj.vplan_id_int, **runner_kwarg_dict)
    assert result_dict["submitted_vplan_count_int"] == 0
    assert not broker_obj.submitted_order_request_list


@pytest.mark.parametrize("qualification_state_str", ["expired", "revoked"])
def test_actual_yaml_with_invalid_qualification_still_recovers_sent_cycle(
        qualified_release_obj, core5_price_df, tmp_path, monkeypatch, qualification_state_str):
    store_obj, broker_obj, decision_obj, vplan_obj, runner_kwarg_dict = _prepared_cycle(
        tmp_path, monkeypatch, qualified_release_obj, core5_price_df)
    result_dict = runner.submit_ready_vplans(store_obj, broker_obj, decision_obj.submission_timestamp_ts,
        "live", False, vplan_id_int=vplan_obj.vplan_id_int, **runner_kwarg_dict)
    assert result_dict["submitted_vplan_count_int"] == 1
    release_dir_path = _write_release_yaml(tmp_path,
        _unqualified_release(qualified_release_obj, qualification_state_str))
    loaded_release_obj = load_release_list(str(release_dir_path))[0]
    assert loaded_release_obj.enabled_bool and loaded_release_obj.mode_str == "live"
    with pytest.raises(ValueError):
        validate_release_manifest(loaded_release_obj)
    with pytest.raises(ValueError):
        core5_helper._build(loaded_release_obj, core5_price_df, "2026-09-11",
            core5_helper._state(loaded_release_obj, "2026-09-11"))
    monkeypatch.setattr(scheduler_utils, "evaluate_build_gate_dict",
        lambda *_argument_tuple, **_argument_dict: {"due_bool": True})
    after_open_ts = decision_obj.target_execution_timestamp_ts + timedelta(minutes=10)
    restarted_store_obj = LiveStateStore(str(tmp_path / "core5.sqlite3"))
    schedule_obj = scheduler_service.get_scheduler_decision(restarted_store_obj,
        after_open_ts, str(release_dir_path), "live")
    assert schedule_obj.next_phase_str == "post_execution_reconcile"
    broker_obj._snapshot_map[qualified_release_obj.account_route_str] = replace(
        broker_obj.get_account_snapshot(qualified_release_obj.account_route_str), snapshot_timestamp_ts=after_open_ts)
    result_dict = runner.post_execution_reconcile(restarted_store_obj, broker_obj, after_open_ts,
        "live", releases_root_path_str=str(release_dir_path), **runner_kwarg_dict)
    assert result_dict["completed_vplan_count_int"] == 1
    assert restarted_store_obj.get_pod_state(loaded_release_obj.pod_id_str).strategy_state_dict == decision_obj.strategy_state_dict
    assert restarted_store_obj.get_latest_decision_plan_for_pod(loaded_release_obj.pod_id_str).status_str == "completed"


@pytest.mark.parametrize("qualification_state_str", ["expired", "revoked"])
def test_actual_yaml_with_invalid_qualification_cannot_submit_ready_plan(
        qualified_release_obj, core5_price_df, tmp_path, monkeypatch, qualification_state_str):
    store_obj, broker_obj, decision_obj, vplan_obj, runner_kwarg_dict = _prepared_cycle(
        tmp_path, monkeypatch, qualified_release_obj, core5_price_df)
    release_dir_path = _write_release_yaml(tmp_path,
        _unqualified_release(qualified_release_obj, qualification_state_str))
    monkeypatch.setattr(broker_obj, "get_core5_account_snapshot",
        lambda *_argument_tuple: pytest.fail("Invalid qualification reached account preflight"))
    result_dict = runner.submit_ready_vplans(store_obj, broker_obj, decision_obj.submission_timestamp_ts,
        "live", False, vplan_id_int=vplan_obj.vplan_id_int,
        releases_root_path_str=str(release_dir_path), **runner_kwarg_dict)
    assert result_dict["submitted_vplan_count_int"] == 0
    assert not broker_obj.submitted_order_request_list
    assert store_obj.get_vplan_by_id(vplan_obj.vplan_id_int).status_str == "blocked"


@pytest.mark.parametrize("cutoff_phase_str", ["after_qualification", "between_dbc_legs"])
def test_socket_cutoff_stops_dispatch_and_preserves_claimed_cycle_for_recovery(
        qualified_release_obj, core5_price_df, tmp_path, monkeypatch, cutoff_phase_str):
    release_obj = _release_for_mode(qualified_release_obj, "paper")
    store_obj, broker_obj, decision_obj, vplan_obj, runner_kwarg_dict = _prepared_cycle(
        tmp_path, monkeypatch, release_obj, core5_price_df, "long_to_short")
    cutoff_ts = vplan_obj.target_execution_timestamp_ts - timedelta(minutes=2)
    ib_obj = _ExpirySubmitIB([decision_obj.submission_timestamp_ts])
    socket_obj = _expiry_client_obj(monkeypatch, ib_obj)
    if cutoff_phase_str == "after_qualification":
        ib_obj.after_qualification_ts = cutoff_ts
    else:
        original_place_fn = ib_obj.placeOrder

        def place_then_cross_cutoff_fn(contract_obj, order_obj):
            trade_obj = original_place_fn(contract_obj, order_obj)
            if contract_obj.symbol == "DBC":
                ib_obj.clock_list[:] = [cutoff_ts]
            return trade_obj

        monkeypatch.setattr(ib_obj, "placeOrder", place_then_cross_cutoff_fn)
    monkeypatch.setattr(broker_obj, "submit_order_request_list", socket_obj.submit_order_request_list)
    result_dict = runner.submit_ready_vplans(store_obj, broker_obj, decision_obj.submission_timestamp_ts,
        "paper", False, vplan_id_int=vplan_obj.vplan_id_int, **runner_kwarg_dict)
    assert result_dict["submitted_vplan_count_int"] == 0
    assert result_dict["reason_count_map_dict"]["opening_dispatch_parked"] == 1
    if cutoff_phase_str == "after_qualification":
        assert not ib_obj.placed_order_list
    else:
        expected_dbc_request_list = [request_obj for request_obj in build_broker_order_request_list_from_vplan(vplan_obj)
            if request_obj.asset_str == "DBC"]
        placed_dbc_order_list = [order_obj for order_obj in ib_obj.placed_order_list if ":DBC:" in order_obj.orderRef]
        assert len(placed_dbc_order_list) == 1
        assert placed_dbc_order_list[0].orderRef == expected_dbc_request_list[0].order_request_key_str
    restarted_store_obj = LiveStateStore(str(tmp_path / "core5.sqlite3"))
    assert restarted_store_obj.get_vplan_by_id(vplan_obj.vplan_id_int).status_str == "submitted"
    persisted_decision_obj = restarted_store_obj.get_decision_plan_by_id(decision_obj.decision_plan_id_int)
    assert persisted_decision_obj.status_str == "submitted"
    assert persisted_decision_obj.snapshot_metadata_dict["opening_dispatch_parked_bool"]
    placed_key_set = {order_obj.orderRef for order_obj in ib_obj.placed_order_list}
    all_key_set = {request_obj.order_request_key_str for request_obj in build_broker_order_request_list_from_vplan(vplan_obj)}
    with restarted_store_obj._connect() as connection_obj:
        resolved_key_set = {row_obj[0] for row_obj in connection_obj.execute(
            "SELECT order_request_key_str FROM vplan_execution_resolution WHERE vplan_id_int=? AND resolution_str='never_dispatched'",
            (vplan_obj.vplan_id_int,))}
        alert_row_obj, = connection_obj.execute("SELECT payload_json_str FROM mr_capsule_execution_alert").fetchall()
    assert resolved_key_set == all_key_set - placed_key_set
    assert json.loads(alert_row_obj[0])["severity_str"] == "critical"
    assert restarted_store_obj.count_broker_orders_for_vplan(vplan_obj.vplan_id_int) == len(ib_obj.placed_order_list)
    placed_count_int = len(ib_obj.placed_order_list)
    retry_dict = runner.submit_ready_vplans(restarted_store_obj, broker_obj, decision_obj.submission_timestamp_ts,
        "paper", False, vplan_id_int=vplan_obj.vplan_id_int, **runner_kwarg_dict)
    assert retry_dict["submitted_vplan_count_int"] == 0
    assert len(ib_obj.placed_order_list) == placed_count_int
    after_open_ts = decision_obj.target_execution_timestamp_ts + timedelta(minutes=10)
    result_dict = runner.expire_stale_decision_plans(restarted_store_obj, after_open_ts,
        "unused", "paper", **runner_kwarg_dict)
    assert result_dict["expired_decision_plan_count_int"] == 0
    monkeypatch.setattr(scheduler_service, "_load_release_list_and_sync",
        lambda *_argument_tuple, **_argument_dict: [release_obj])
    monkeypatch.setattr(scheduler_utils, "evaluate_build_gate_dict",
        lambda *_argument_tuple, **_argument_dict: {"due_bool": True})
    schedule_obj = scheduler_service.get_scheduler_decision(restarted_store_obj, after_open_ts, "unused", "paper")
    assert schedule_obj.next_phase_str == "post_execution_reconcile"


@pytest.mark.parametrize("mutation_str", ["other_account", "other_client_order", "changed_position"])
@pytest.mark.parametrize("mode_str", ["paper", "live"])
def test_final_account_read_blocks_changed_account_before_funding(
        qualified_release_obj, core5_price_df, tmp_path, monkeypatch, mutation_str, mode_str):
    release_obj = _release_for_mode(qualified_release_obj, mode_str)
    store_obj, broker_obj, decision_obj, vplan_obj, runner_kwarg_dict = _prepared_cycle(
        tmp_path, monkeypatch, release_obj, core5_price_df)
    snapshot_obj = broker_obj.get_account_snapshot(release_obj.account_route_str)
    mutation_dict = {"account_route_str": "U_ANOTHER"} if mutation_str == "other_account" else (
        {"open_order_id_list": ["manual:other_client"]} if mutation_str == "other_client_order"
        else {"position_amount_map": {"SPY": 1.0}})
    monkeypatch.setattr(broker_obj, "get_core5_account_snapshot", lambda *_argument_tuple: replace(snapshot_obj, **mutation_dict))
    monkeypatch.setattr(broker_obj, "get_core5_funding_evidence",
        lambda *_argument_tuple: pytest.fail("Account validation did not precede funding"))
    result_dict = runner.submit_ready_vplans(store_obj, broker_obj, decision_obj.submission_timestamp_ts,
        mode_str, False, vplan_id_int=vplan_obj.vplan_id_int, **runner_kwarg_dict)
    assert result_dict["submitted_vplan_count_int"] == 0
    assert not broker_obj.submitted_order_request_list


def test_disabling_a_prepared_release_prevents_direct_submit(
        qualified_release_obj, core5_price_df, tmp_path, monkeypatch):
    store_obj, broker_obj, decision_obj, vplan_obj, runner_kwarg_dict = _prepared_cycle(
        tmp_path, monkeypatch, qualified_release_obj, core5_price_df)
    store_obj.upsert_release(replace(qualified_release_obj, enabled_bool=False))
    result_dict = runner.submit_ready_vplans(store_obj, broker_obj, decision_obj.submission_timestamp_ts,
        "live", False, vplan_id_int=vplan_obj.vplan_id_int, **runner_kwarg_dict)
    assert result_dict["submitted_vplan_count_int"] == 0
    assert not broker_obj.submitted_order_request_list


@pytest.mark.parametrize("core5_first_bool", [False, True])
def test_core5_account_is_dedicated_in_either_release_order(qualified_release_obj, core5_first_bool):
    other_release_obj = replace(qualified_release_obj, release_id_str="other.v1", pod_id_str="other_pod",
        account_route_str=" u_core5_test ", strategy_import_str="strategies.dv2.strategy_mr_dv2:DVO2Strategy",
        data_profile_str="norgate_eod_sp500_pit", params_dict={})
    release_list = [qualified_release_obj, other_release_obj] if core5_first_bool else [other_release_obj, qualified_release_obj]
    with pytest.raises(ValueError, match="dedicated account"):
        validate_release_list(release_list)


def test_core5_auto_reference_rejects_incompatible_accounting(qualified_release_obj, tmp_path, monkeypatch):
    support_dict = reference_compare.inspect_auto_reference_support_dict(qualified_release_obj)
    assert support_dict["supported_bool"] is False
    assert support_dict["reason_str"] == "core5_accounting_bridge_required"
    monkeypatch.setattr(core5_helper.core5_module, "run_variant",
        lambda **_argument_dict: pytest.fail("Incompatible accounting reference was run"))
    with pytest.raises((ValueError, AttributeError), match="CORE5|core5"):
        reference_compare.run_auto_reference_strategy(release_obj=qualified_release_obj,
            deployment_start_date_str="2026-09-11", reference_end_date_str="2026-09-14",
            deployment_initial_cash_float=100_000.0, output_dir_path_obj=tmp_path)


@pytest.mark.parametrize("cycle_str,action_str", [("long_to_short", "SELL"), ("short_to_long", "BUY")])
def test_dbc_two_legs_map_to_distinct_signed_market_on_open_orders(
        qualified_release_obj, core5_price_df, tmp_path, monkeypatch, cycle_str, action_str):
    _, _, decision_obj, vplan_obj, _ = _prepared_cycle(
        tmp_path, monkeypatch, qualified_release_obj, core5_price_df, cycle_str)
    deadline_str = (vplan_obj.target_execution_timestamp_ts - timedelta(minutes=2)).isoformat()
    request_list = [replace(request_obj, submission_deadline_timestamp_str=deadline_str)
        for request_obj in build_broker_order_request_list_from_vplan(vplan_obj)
        if request_obj.asset_str == "DBC"]
    ib_obj = _ExpirySubmitIB([decision_obj.submission_timestamp_ts])
    socket_obj = _expiry_client_obj(monkeypatch, ib_obj)
    result_obj = socket_obj.submit_order_request_list(qualified_release_obj.account_route_str,
        request_list, decision_obj.submission_timestamp_ts)
    assert result_obj.submit_ack_status_str == "complete"
    assert len(ib_obj.placed_order_list) == 2
    for request_obj, order_obj in zip(request_list, ib_obj.placed_order_list):
        assert (order_obj.orderType, order_obj.tif, order_obj.action) == ("MKT", "OPG", action_str)
        assert order_obj.totalQuantity == abs(request_obj.amount_float)
        assert order_obj.account == qualified_release_obj.account_route_str
        assert order_obj.orderRef == request_obj.order_request_key_str
    assert ib_obj.placed_order_list[0].orderRef != ib_obj.placed_order_list[1].orderRef
