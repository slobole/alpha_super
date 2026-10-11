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


def test_capsule_account_read_precedes_claim_and_dispatch_preserves_request_ids(capsule_case, monkeypatch):
    state_store_obj, broker_adapter_obj, release_obj, vplan_obj, runner_kwarg_dict, _ = capsule_case
    original_request_list = build_broker_order_request_list_from_vplan(vplan_obj)
    # The persisted fixture has BUY before BIL SELL, opposite desired dispatch.
    assert [request_obj.asset_str for request_obj in original_request_list] == ["AAPL", "BIL"]
    phase_list = []
    original_account_fn = broker_adapter_obj.get_capsule_account_snapshot
    original_claim_fn = state_store_obj.claim_vplan_for_submission
    original_submit_fn = broker_adapter_obj.submit_order_request_list
    def account_fn(account_route_str):
        phase_list.append("account_read")
        assert state_store_obj.get_vplan_by_id(vplan_obj.vplan_id_int).status_str == "ready"
        return original_account_fn(account_route_str)
    def claim_fn(vplan_id_int):
        phase_list.append("claim")
        return original_claim_fn(vplan_id_int)
    def submit_fn(**submit_kwarg_dict):
        phase_list.append("submit")
        request_list = submit_kwarg_dict["broker_order_request_list"]
        assert [request_obj.asset_str for request_obj in request_list] == ["BIL", "AAPL"]
        assert {request_obj.order_request_key_str for request_obj in request_list} == {request_obj.order_request_key_str for request_obj in original_request_list}
        return original_submit_fn(**submit_kwarg_dict)
    monkeypatch.setattr(broker_adapter_obj, "get_capsule_account_snapshot", account_fn)
    # Owner decision 2026-10-11: no funding preview exists any more.
    assert not hasattr(broker_adapter_obj, "get_capsule_funding_evidence")
    monkeypatch.setattr(state_store_obj, "claim_vplan_for_submission", claim_fn)
    monkeypatch.setattr(broker_adapter_obj, "submit_order_request_list", submit_fn)
    result_dict = runner_module.submit_ready_vplans(state_store_obj, broker_adapter_obj,
        SUBMIT_TIMESTAMP_TS, "paper", False, **runner_kwarg_dict)
    assert result_dict["submitted_vplan_count_int"] == 1
    assert phase_list == ["account_read", "claim", "submit"]
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


def test_capsule_checks_all_client_open_orders_before_claim(capsule_case, monkeypatch):
    state_store_obj, broker_adapter_obj, release_obj, vplan_obj, runner_kwarg_dict, _ = capsule_case
    snapshot_obj = broker_adapter_obj.get_account_snapshot(release_obj.account_route_str)
    monkeypatch.setattr(broker_adapter_obj, "get_capsule_account_snapshot", lambda _: replace(snapshot_obj, open_order_id_list=["manual:other_client"]))
    monkeypatch.setattr(state_store_obj, "claim_vplan_for_submission", lambda *_: pytest.fail("Open-order guard skipped"))
    result_dict = runner_module.submit_ready_vplans(state_store_obj, broker_adapter_obj,
        SUBMIT_TIMESTAMP_TS, "paper", False, **runner_kwarg_dict)
    assert result_dict["reason_count_map_dict"] == {"mr_capsule_pre_submit_account_changed": 1}
    assert not broker_adapter_obj.submitted_order_request_list


class _ReachedBrokerResolution(Exception):
    pass


def _run_live_operation(operation_str, state_store_obj, broker_adapter_obj, runner_kwarg_dict):
    if operation_str == "submit":
        runner_module.submit_ready_vplans(state_store_obj, broker_adapter_obj, SUBMIT_TIMESTAMP_TS, "live", False, **runner_kwarg_dict)
    elif operation_str == "reconcile":
        runner_module.post_execution_reconcile(state_store_obj, broker_adapter_obj, RECONCILE_TIMESTAMP_TS, "live", **runner_kwarg_dict)
    else:
        runner_module.build_vplans(state_store_obj, broker_adapter_obj, SUBMIT_TIMESTAMP_TS, "live", **runner_kwarg_dict)


@pytest.mark.parametrize("params_dict", [{}, {"margin_account_confirmed_bool": True}])
@pytest.mark.parametrize("operation_str", ["submit", "reconcile", "build"])
def test_persisted_capsule_live_reaches_broker_resolution(capsule_case, monkeypatch, operation_str, params_dict):
    # LIVE is allowed with no YAML attestation (owner decisions 2026-10-10/11).
    state_store_obj, broker_adapter_obj, release_obj, vplan_obj, runner_kwarg_dict, _ = capsule_case
    state_store_obj.upsert_release(replace(release_obj, mode_str="live", account_route_str="U_TEST", params_dict=params_dict))
    def reached_fn(**_):
        raise _ReachedBrokerResolution()
    monkeypatch.setattr(runner_module, "_coerce_broker_adapter_resolver_obj", reached_fn)
    with pytest.raises(_ReachedBrokerResolution):
        _run_live_operation(operation_str, state_store_obj, broker_adapter_obj, runner_kwarg_dict)


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


def test_gateway_uses_the_capsule_opt_in_methods():
    called_method_list = []
    class FakeSocket:
        def get_capsule_account_snapshot(self, account_route_str):
            called_method_list.append(("snapshot", account_route_str))
            return "account snapshot"
        def get_capsule_order_state_snapshot(self, account_route_str, since_timestamp_ts, **kwarg_dict):
            called_method_list.append(("orders", account_route_str, since_timestamp_ts, kwarg_dict))
            return [], [], []
    adapter_obj = IBKRGatewayBrokerAdapter.__new__(IBKRGatewayBrokerAdapter)
    adapter_obj.socket_client_obj = FakeSocket()
    assert adapter_obj.get_capsule_account_snapshot("DU_TEST") == "account snapshot"
    assert adapter_obj.get_capsule_order_state_snapshot("DU_TEST", SUBMIT_TIMESTAMP_TS, submission_key_str="plan", allowed_broker_order_id_set={"one"}) == ([], [], [])
    assert [entry_tuple[0] for entry_tuple in called_method_list] == ["snapshot", "orders"]
    assert not hasattr(BrokerAdapter, "get_capsule_funding_evidence")
