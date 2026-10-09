"""Capsule submission policy; all account and order operations are synthetic."""
from dataclasses import replace

import pytest

from alpha.live import runner as runner_module
from alpha.live.daily_broker import DailyExecutionSnapshot
from alpha.live.execution_engine import build_broker_order_request_list_from_vplan
from alpha.live.daily_reconcile import DailyReconcileResult
from alpha.live.order_clerk import BrokerAdapter, IBKRGatewayBrokerAdapter
from test_live_mr_capsule_recovery import capsule_case, SUBMIT_TIMESTAMP_TS, RECONCILE_TIMESTAMP_TS


@pytest.fixture(autouse=True)
def historical_dispatch_clock(monkeypatch):
    class DispatchClock:
        @classmethod
        def now(cls, timezone_obj):
            return SUBMIT_TIMESTAMP_TS.astimezone(timezone_obj)
    monkeypatch.setattr("alpha.live.dispatch_state.datetime", DispatchClock)


def test_capsule_funding_precedes_claim_and_dispatch_preserves_request_ids(capsule_case, monkeypatch):
    state_store_obj, broker_adapter_obj, release_obj, vplan_obj, runner_kwarg_dict, _ = capsule_case
    original_request_list = build_broker_order_request_list_from_vplan(vplan_obj)
    # The persisted fixture has BUY before BIL SELL, opposite desired dispatch.
    assert [request_obj.asset_str for request_obj in original_request_list] == ["AAPL", "BIL"]
    phase_list = []
    original_funding_fn = broker_adapter_obj.get_capsule_funding_evidence
    original_claim_fn = state_store_obj.claim_vplan_for_submission
    original_submit_fn = broker_adapter_obj.submit_order_request_list
    def funding_fn(account_route_str, request_list):
        phase_list.append("funding")
        assert request_list == original_request_list
        assert state_store_obj.get_vplan_by_id(vplan_obj.vplan_id_int).status_str == "ready"
        return original_funding_fn(account_route_str, request_list)
    def claim_fn(vplan_id_int):
        phase_list.append("claim")
        return original_claim_fn(vplan_id_int)
    def submit_fn(**submit_kwarg_dict):
        phase_list.append("submit")
        request_list = submit_kwarg_dict["broker_order_request_list"]
        assert [request_obj.asset_str for request_obj in request_list] == ["BIL", "AAPL"]
        assert {request_obj.order_request_key_str for request_obj in request_list} == {request_obj.order_request_key_str for request_obj in original_request_list}
        return original_submit_fn(**submit_kwarg_dict)
    monkeypatch.setattr(broker_adapter_obj, "get_capsule_funding_evidence", funding_fn)
    monkeypatch.setattr(state_store_obj, "claim_vplan_for_submission", claim_fn)
    monkeypatch.setattr(broker_adapter_obj, "submit_order_request_list", submit_fn)
    result_dict = runner_module.submit_ready_vplans(state_store_obj, broker_adapter_obj,
        SUBMIT_TIMESTAMP_TS, "paper", False, **runner_kwarg_dict)
    assert result_dict["submitted_vplan_count_int"] == 1
    assert phase_list == ["funding", "claim", "submit"]
    assert build_broker_order_request_list_from_vplan(state_store_obj.get_vplan_by_id(vplan_obj.vplan_id_int)) == original_request_list


def test_other_sales_dispatch_between_bil_and_buys_without_reassigning_ids(capsule_case, monkeypatch):
    state_store_obj, broker_adapter_obj, release_obj, vplan_obj, runner_kwarg_dict, _ = capsule_case
    original_request_list = build_broker_order_request_list_from_vplan(vplan_obj)
    stock_sale_obj = replace(original_request_list[1], asset_str="MSFT", amount_float=-5.0,
        order_request_key_str="fixed:MSFT:3")
    input_request_list = [*original_request_list, stock_sale_obj]
    monkeypatch.setattr(runner_module, "build_broker_order_request_list_from_vplan", lambda _: list(input_request_list))
    runner_module.submit_ready_vplans(state_store_obj, broker_adapter_obj,
        SUBMIT_TIMESTAMP_TS, "paper", False, **runner_kwarg_dict)
    assert [request_obj.asset_str for request_obj in broker_adapter_obj.submitted_order_request_list] == ["BIL", "MSFT", "AAPL"]
    assert broker_adapter_obj.submitted_order_request_list[1].order_request_key_str == "fixed:MSFT:3"


@pytest.mark.parametrize("error_obj", [ValueError("Insufficient broker buying power"), TimeoutError("Account preview timeout"), NotImplementedError("Unsupported adapter")])
def test_capsule_funding_failure_retries_transient_or_preserves_sales(capsule_case, monkeypatch, error_obj):
    state_store_obj, broker_adapter_obj, release_obj, vplan_obj, runner_kwarg_dict, _ = capsule_case
    def fail_funding(*_):
        raise error_obj
    monkeypatch.setattr(broker_adapter_obj, "get_capsule_funding_evidence", fail_funding)
    transient_bool = isinstance(error_obj, TimeoutError)
    if transient_bool:
        monkeypatch.setattr(state_store_obj, "claim_vplan_for_submission", lambda *_: pytest.fail("Transient funding failure claimed a batch"))
    result_dict = runner_module.submit_ready_vplans(state_store_obj, broker_adapter_obj,
        SUBMIT_TIMESTAMP_TS, "paper", False, **runner_kwarg_dict)
    assert result_dict["submitted_vplan_count_int"] == int(not transient_bool)
    assert result_dict["reason_count_map_dict"]["mr_capsule_funding_not_verified"] == 1
    metadata_dict = state_store_obj.get_decision_plan_by_id(vplan_obj.decision_plan_id_int).snapshot_metadata_dict
    if transient_bool:
        assert result_dict["reason_count_map_dict"]["dispatch_retry_pending"] == 1
        assert not broker_adapter_obj.submitted_order_request_list
        assert state_store_obj.get_vplan_by_id(vplan_obj.vplan_id_int).status_str == "ready"
        assert state_store_obj.get_decision_plan_by_id(vplan_obj.decision_plan_id_int).status_str == "vplan_ready"
        assert not metadata_dict.get("funding_buys_dropped_bool")
    else:
        sale_request_obj, = broker_adapter_obj.submitted_order_request_list
        assert (sale_request_obj.asset_str, sale_request_obj.amount_float) == ("BIL", -100.0)
        assert state_store_obj.get_vplan_by_id(vplan_obj.vplan_id_int).status_str == "submitted"
        assert state_store_obj.get_decision_plan_by_id(vplan_obj.decision_plan_id_int).status_str == "submitted"
        assert metadata_dict["funding_buys_dropped_bool"] is True


def test_capsule_checks_all_client_open_orders_before_funding(capsule_case, monkeypatch):
    state_store_obj, broker_adapter_obj, release_obj, vplan_obj, runner_kwarg_dict, _ = capsule_case
    snapshot_obj = broker_adapter_obj.get_account_snapshot(release_obj.account_route_str)
    monkeypatch.setattr(broker_adapter_obj, "get_capsule_account_snapshot", lambda _: replace(snapshot_obj, open_order_id_list=["manual:other_client"]))
    monkeypatch.setattr(broker_adapter_obj, "get_capsule_funding_evidence", lambda *_: pytest.fail("Open-order guard skipped"))
    result_dict = runner_module.submit_ready_vplans(state_store_obj, broker_adapter_obj,
        SUBMIT_TIMESTAMP_TS, "paper", False, **runner_kwarg_dict)
    assert result_dict["reason_count_map_dict"] == {"mr_capsule_pre_submit_account_changed": 1}
    assert not broker_adapter_obj.submitted_order_request_list


@pytest.mark.parametrize("operation_str", ["submit", "reconcile", "build"])
def test_persisted_capsule_live_refused_before_any_broker_resolution(capsule_case, monkeypatch, operation_str):
    state_store_obj, broker_adapter_obj, release_obj, vplan_obj, runner_kwarg_dict, _ = capsule_case
    state_store_obj.upsert_release(replace(release_obj, mode_str="live", account_route_str="U_TEST"))
    monkeypatch.setattr(runner_module, "_coerce_broker_adapter_resolver_obj", lambda **_: pytest.fail("LIVE lock happened after broker resolution"))
    with pytest.raises(ValueError, match="MR capsule LIVE trading is locked"):
        if operation_str == "submit":
            runner_module.submit_ready_vplans(state_store_obj, broker_adapter_obj, SUBMIT_TIMESTAMP_TS, "live", False, **runner_kwarg_dict)
        elif operation_str == "reconcile":
            runner_module.post_execution_reconcile(state_store_obj, broker_adapter_obj, RECONCILE_TIMESTAMP_TS, "live", **runner_kwarg_dict)
        else:
            runner_module.build_vplans(state_store_obj, broker_adapter_obj, SUBMIT_TIMESTAMP_TS, "live", **runner_kwarg_dict)


def test_capsule_daily_reconcile_runs_even_when_fill_reporting_is_unavailable(capsule_case, monkeypatch):
    state_store_obj, broker_adapter_obj, release_obj, vplan_obj, runner_kwarg_dict, _ = capsule_case
    state_store_obj.mark_vplan_status(vplan_obj.vplan_id_int, "submitted")
    state_store_obj.mark_decision_plan_status(vplan_obj.decision_plan_id_int, "submitted")
    snapshot_obj = replace(broker_adapter_obj.get_account_snapshot(release_obj.account_route_str), snapshot_timestamp_ts=RECONCILE_TIMESTAMP_TS)
    daily_snapshot_obj = DailyExecutionSnapshot(snapshot_obj, [], RECONCILE_TIMESTAMP_TS, RECONCILE_TIMESTAMP_TS)
    called_vplan_list = []
    def recovery_fn(store_obj, adapter_obj, passed_release_obj, decision_obj, as_of_ts, *, vplan_obj, **_kwarg_dict):
        assert store_obj is state_store_obj and adapter_obj is broker_adapter_obj
        assert passed_release_obj == release_obj and as_of_ts == RECONCILE_TIMESTAMP_TS
        called_vplan_list.append(vplan_obj.vplan_id_int)
        return DailyReconcileResult("pending", snapshot_obj)
    def unavailable_reporting_fn(**_kwarg_dict):
        raise TimeoutError("Synthetic fill history outage")
    monkeypatch.setattr(runner_module, "reconcile_daily_cycle", recovery_fn)
    monkeypatch.setattr(broker_adapter_obj, "get_daily_execution_snapshot", lambda _: daily_snapshot_obj)
    monkeypatch.setattr(broker_adapter_obj, "get_account_snapshot", lambda *_: pytest.fail("Legacy observation ran"))
    monkeypatch.setattr(broker_adapter_obj, "get_recent_order_state_snapshot", unavailable_reporting_fn)
    result_dict = runner_module.post_execution_reconcile(state_store_obj, broker_adapter_obj,
        RECONCILE_TIMESTAMP_TS, "paper", **runner_kwarg_dict)
    assert result_dict["completed_vplan_count_int"] == 0
    assert called_vplan_list == [vplan_obj.vplan_id_int]
    assert not runner_module.is_vplan_execution_exception_parked(state_store_obj, vplan_obj)


def test_base_adapter_refuses_unimplemented_funding_and_gateway_uses_opt_in_methods():
    with pytest.raises(NotImplementedError, match="margin verification"):
        BrokerAdapter.get_capsule_funding_evidence(object(), "DU_TEST", [])
    called_method_list = []
    class FakeSocket:
        def get_capsule_account_snapshot(self, account_route_str):
            called_method_list.append(("snapshot", account_route_str))
            return "account snapshot"
        def get_capsule_order_state_snapshot(self, account_route_str, since_timestamp_ts, **kwarg_dict):
            called_method_list.append(("orders", account_route_str, since_timestamp_ts, kwarg_dict))
            return [], [], []
        def get_capsule_funding_evidence(self, account_route_str, request_list):
            called_method_list.append(("funding", account_route_str, request_list))
            return {"required_bool": False}
    adapter_obj = IBKRGatewayBrokerAdapter.__new__(IBKRGatewayBrokerAdapter)
    adapter_obj.socket_client_obj = FakeSocket()
    assert adapter_obj.get_capsule_account_snapshot("DU_TEST") == "account snapshot"
    assert adapter_obj.get_capsule_order_state_snapshot("DU_TEST", SUBMIT_TIMESTAMP_TS, submission_key_str="plan", allowed_broker_order_id_set={"one"}) == ([], [], [])
    assert adapter_obj.get_capsule_funding_evidence("DU_TEST", []) == {"required_bool": False}
    assert [entry_tuple[0] for entry_tuple in called_method_list] == ["snapshot", "orders", "funding"]


def test_funded_retry_keeps_buy_drop_and_withholds_daily_bil_completion(capsule_case, monkeypatch):
    store_obj, broker_obj, release_obj, plan_obj, option_dict, _ = capsule_case
    original_funding_fn = broker_obj.get_capsule_funding_evidence
    original_trace_fn = runner_module._emit_live_trace_event
    funding_count_int = 0
    trace_count_int = 0

    def funding_fn(*arg_tuple, **kwarg_dict):
        nonlocal funding_count_int
        funding_count_int += 1
        if funding_count_int == 1:
            raise ValueError("Insufficient broker buying power")
        return original_funding_fn(*arg_tuple, **kwarg_dict)

    def trace_fn(*arg_tuple, **kwarg_dict):
        nonlocal trace_count_int
        if arg_tuple[0] == "vplan.submit_request":
            trace_count_int += 1
            if trace_count_int == 1:
                raise TimeoutError("Trace failed before any send")
        return original_trace_fn(*arg_tuple, **kwarg_dict)

    monkeypatch.setattr(broker_obj, "get_capsule_funding_evidence", funding_fn)
    monkeypatch.setattr(runner_module, "_emit_live_trace_event", trace_fn)
    first_dict = runner_module.submit_ready_vplans(store_obj, broker_obj,
        SUBMIT_TIMESTAMP_TS, "paper", False, **option_dict)
    assert first_dict["reason_count_map_dict"]["dispatch_retry_pending"] == 1
    assert store_obj.get_vplan_by_id(plan_obj.vplan_id_int).status_str == "ready"
    assert broker_obj.submitted_order_request_list == []
    assert store_obj.get_decision_plan_by_id(plan_obj.decision_plan_id_int).snapshot_metadata_dict["funding_buys_dropped_bool"] is True

    assert runner_module.submit_ready_vplans(store_obj, broker_obj,
        SUBMIT_TIMESTAMP_TS, "paper", False, **option_dict)["submitted_vplan_count_int"] == 1
    assert [request_obj.asset_str for request_obj in broker_obj.submitted_order_request_list] == ["BIL"]
    assert store_obj.get_decision_plan_by_id(plan_obj.decision_plan_id_int).snapshot_metadata_dict["funding_buys_dropped_bool"] is True

    # No supplemental BIL sale is allowed after this cycle's buys were dropped.
    snapshot_obj = broker_obj.get_account_snapshot(release_obj.account_route_str)
    broker_obj.seed_account_snapshot(release_obj.account_route_str, snapshot_obj.cash_float,
        snapshot_obj.net_liq_float, {**snapshot_obj.position_amount_map, "BIL": plan_obj.target_share_map["BIL"] + 60},
        RECONCILE_TIMESTAMP_TS, session_mode_str="paper")
    runner_module.post_execution_reconcile(store_obj, broker_obj, RECONCILE_TIMESTAMP_TS, "paper", **option_dict)
    assert len(broker_obj.submitted_order_request_list) == 1
    assert runner_module.submit_ready_vplans(store_obj, broker_obj,
        SUBMIT_TIMESTAMP_TS, "paper", False, **option_dict)["submitted_vplan_count_int"] == 0
    assert len(broker_obj.submitted_order_request_list) == 1


def test_funding_drop_survives_crash_before_claim_and_reopened_store(capsule_case, monkeypatch):
    from alpha.live.state_store_v2 import LiveStateStore

    store_obj, broker_obj, _, plan_obj, option_dict, _ = capsule_case
    original_funding_fn = broker_obj.get_capsule_funding_evidence

    def funding_failure_fn(*argument_tuple):
        raise ValueError("Insufficient broker buying power")

    def crash_fn(vplan_id_int):
        assert store_obj.get_decision_plan_by_id(plan_obj.decision_plan_id_int).snapshot_metadata_dict["funding_buys_dropped_bool"]
        raise KeyboardInterrupt("Synthetic process stop before claim")

    monkeypatch.setattr(broker_obj, "get_capsule_funding_evidence", funding_failure_fn)
    monkeypatch.setattr(store_obj, "claim_vplan_for_submission", crash_fn)
    with pytest.raises(KeyboardInterrupt, match="Synthetic"):
        runner_module.submit_ready_vplans(store_obj, broker_obj, SUBMIT_TIMESTAMP_TS, "paper", False, **option_dict)
    assert not broker_obj.submitted_order_request_list
    restarted_store_obj = LiveStateStore(store_obj.db_path_str)

    def funding_recovered_fn(account_route_str, request_list):
        assert all(request_obj.amount_float < 0 for request_obj in request_list)
        return original_funding_fn(account_route_str, request_list)

    monkeypatch.setattr(broker_obj, "get_capsule_funding_evidence", funding_recovered_fn)
    assert runner_module.submit_ready_vplans(restarted_store_obj, broker_obj,
        SUBMIT_TIMESTAMP_TS, "paper", False, **option_dict)["submitted_vplan_count_int"] == 1
    assert [request_obj.asset_str for request_obj in broker_obj.submitted_order_request_list] == ["BIL"]
    assert restarted_store_obj.get_decision_plan_by_id(plan_obj.decision_plan_id_int).snapshot_metadata_dict["funding_buys_dropped_bool"]


def test_drop_from_concurrent_preflight_is_reloaded_after_claim(capsule_case, monkeypatch):
    from alpha.live.capsule_funding import persist_capsule_buy_drop

    store_obj, broker_obj, release_obj, plan_obj, option_dict, _ = capsule_case
    original_claim_fn = store_obj.claim_vplan_for_submission

    def concurrent_claim_fn(vplan_id_int):
        assert persist_capsule_buy_drop(store_obj, release_obj, plan_obj)
        return original_claim_fn(vplan_id_int)

    monkeypatch.setattr(store_obj, "claim_vplan_for_submission", concurrent_claim_fn)
    assert runner_module.submit_ready_vplans(store_obj, broker_obj,
        SUBMIT_TIMESTAMP_TS, "paper", False, **option_dict)["submitted_vplan_count_int"] == 1
    assert [request_obj.asset_str for request_obj in broker_obj.submitted_order_request_list] == ["BIL"]


def test_losing_funding_preflight_cannot_change_already_claimed_batch(capsule_case):
    from alpha.live.capsule_funding import persist_capsule_buy_drop

    store_obj, _, release_obj, plan_obj, _, _ = capsule_case
    assert store_obj.claim_vplan_for_submission(plan_obj.vplan_id_int)
    assert not persist_capsule_buy_drop(store_obj, release_obj, plan_obj)
    assert not store_obj.get_decision_plan_by_id(plan_obj.decision_plan_id_int).snapshot_metadata_dict.get("funding_buys_dropped_bool")


def test_failed_funding_drop_write_never_claims_or_aborts_the_run(capsule_case, monkeypatch):
    store_obj, broker_obj, _, plan_obj, option_dict, _ = capsule_case
    original_funding_fn = broker_obj.get_capsule_funding_evidence
    with store_obj._connect() as connection_obj:
        connection_obj.execute("""CREATE TRIGGER reject_funding_drop BEFORE UPDATE OF snapshot_metadata_json_str
            ON decision_plan BEGIN SELECT RAISE(ABORT,'synthetic funding metadata failure'); END""")

    def funding_failure_fn(*argument_tuple):
        raise ValueError("Insufficient broker buying power")

    monkeypatch.setattr(broker_obj, "get_capsule_funding_evidence", funding_failure_fn)
    monkeypatch.setattr(store_obj, "claim_vplan_for_submission", lambda *_args:
        pytest.fail("Cannot claim without durable funding suppression"))
    result_dict = runner_module.submit_ready_vplans(store_obj, broker_obj,
        SUBMIT_TIMESTAMP_TS, "paper", False, **option_dict)
    assert result_dict["reason_count_map_dict"]["mr_capsule_funding_drop_persist_failed"] == 1
    assert not broker_obj.submitted_order_request_list
    assert store_obj.get_vplan_by_id(plan_obj.vplan_id_int).status_str == "blocked"
    with store_obj._connect() as connection_obj:
        connection_obj.execute("DROP TRIGGER reject_funding_drop")
    monkeypatch.setattr(broker_obj, "get_capsule_funding_evidence", original_funding_fn)
    retry_result_dict = runner_module.submit_ready_vplans(store_obj, broker_obj,
        SUBMIT_TIMESTAMP_TS, "paper", False, **option_dict)
    assert retry_result_dict["submitted_vplan_count_int"] == 0
    assert not broker_obj.submitted_order_request_list
