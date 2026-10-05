"""Capsule submission policy; all account and order operations are synthetic."""
from dataclasses import replace

import pytest

from alpha.live import runner as runner_module
from alpha.live.execution_engine import build_broker_order_request_list_from_vplan
from alpha.live.order_clerk import BrokerAdapter, IBKRGatewayBrokerAdapter
from alpha.live.reconcile import reconcile_account_state
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
    from alpha.live.execution_resolution import load_request_resolution_dict

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
    resolution_dict = load_request_resolution_dict(state_store_obj, vplan_obj)
    if transient_bool:
        assert result_dict["reason_count_map_dict"]["dispatch_retry_pending"] == 1
        assert not broker_adapter_obj.submitted_order_request_list
        assert state_store_obj.get_vplan_by_id(vplan_obj.vplan_id_int).status_str == "ready"
        assert state_store_obj.get_decision_plan_by_id(vplan_obj.decision_plan_id_int).status_str == "vplan_ready"
        assert resolution_dict == {}
    else:
        sale_request_obj, = broker_adapter_obj.submitted_order_request_list
        assert (sale_request_obj.asset_str, sale_request_obj.amount_float) == ("BIL", -100.0)
        assert state_store_obj.get_vplan_by_id(vplan_obj.vplan_id_int).status_str == "submitted"
        assert state_store_obj.get_decision_plan_by_id(vplan_obj.decision_plan_id_int).status_str == "submitted"
        resolution_obj, = resolution_dict.values()
        assert resolution_obj["asset_str"] == "AAPL"
        assert resolution_obj["resolution_str"] == "never_dispatched"
        assert resolution_obj["reason_str"] == "funding_check_suppressed_buy"


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


def test_capsule_reconcile_delegates_before_legacy_observation_and_keeps_automatic_refresh(capsule_case, monkeypatch):
    state_store_obj, broker_adapter_obj, release_obj, vplan_obj, runner_kwarg_dict, _ = capsule_case
    state_store_obj.mark_vplan_status(vplan_obj.vplan_id_int, "submitted")
    state_store_obj.mark_decision_plan_status(vplan_obj.decision_plan_id_int, "submitted")
    snapshot_obj = replace(broker_adapter_obj.get_account_snapshot(release_obj.account_route_str), snapshot_timestamp_ts=RECONCILE_TIMESTAMP_TS)
    reconciliation_obj = reconcile_account_state(snapshot_obj.position_amount_map, snapshot_obj.cash_float, snapshot_obj)
    called_vplan_list = []
    def recovery_fn(store_obj, adapter_obj, passed_release_obj, passed_vplan_obj, decision_obj, as_of_ts):
        assert store_obj is state_store_obj and adapter_obj is broker_adapter_obj
        assert passed_release_obj == release_obj and as_of_ts == RECONCILE_TIMESTAMP_TS
        called_vplan_list.append(passed_vplan_obj.vplan_id_int)
        return reconciliation_obj, "accepted_residual", [], snapshot_obj
    monkeypatch.setattr(runner_module, "reconcile_capsule_cycle", recovery_fn)
    monkeypatch.setattr(broker_adapter_obj, "get_account_snapshot", lambda *_: pytest.fail("Legacy observation ran"))
    monkeypatch.setattr(broker_adapter_obj, "get_recent_order_state_snapshot", lambda **_: pytest.fail("Legacy order normalization ran"))
    result_dict = runner_module.post_execution_reconcile(state_store_obj, broker_adapter_obj,
        RECONCILE_TIMESTAMP_TS, "paper", **runner_kwarg_dict)
    assert result_dict["completed_vplan_count_int"] == 1
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
