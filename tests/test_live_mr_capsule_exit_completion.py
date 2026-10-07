"""Offline recovery boundaries: stock exits, parking targets, sessions and claims."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime, timedelta
from threading import Barrier

import pytest

from alpha.live.execution_engine import build_broker_order_request_list_from_vplan
from alpha.live.models import BrokerOrderFill, BrokerOrderRecord, SubmitBatchResult, VPlanRow
from alpha.live.daily_reconcile import claim_daily_completion_request
from alpha.live.order_clerk import StubBrokerAdapter
from alpha.live.state_store_v2 import LiveStateStore
from test_live_mr_capsule_recovery import capsule_case, MARKET_TIMEZONE_OBJ, RECONCILE_TIMESTAMP_TS
from test_live_mr_capsule_execution_policy import alert_rows, reconcile_case, seed_execution, close_case
from daily_broker_fakes import install_daily_broker_stub


def _exit_case(case_tuple, asset_str, target_share_float=0.0, target_timestamp_ts=None):
    source_store_obj, _, source_release_obj, source_vplan_obj, runner_dict, temporary_path = case_tuple
    source_decision_obj = source_store_obj.get_decision_plan_by_id(source_vplan_obj.decision_plan_id_int)
    target_timestamp_ts = target_timestamp_ts or source_vplan_obj.target_execution_timestamp_ts
    submission_timestamp_ts = target_timestamp_ts - timedelta(minutes=8)
    signal_timestamp_ts = (datetime(2024, 11, 27, 16, tzinfo=MARKET_TIMEZONE_OBJ)
                           if target_timestamp_ts.month == 11 else source_decision_obj.signal_timestamp_ts)
    strategy_import_str = source_release_obj.strategy_import_str.replace("_bil", "_spmo") if asset_str == "SPMO" else source_release_obj.strategy_import_str
    release_obj = replace(source_release_obj, strategy_import_str=strategy_import_str)
    store_obj = LiveStateStore(str(temporary_path / "exit_policy.sqlite3"))
    store_obj.upsert_release(release_obj)
    store_obj.upsert_pod_state(replace(source_store_obj.get_pod_state(release_obj.pod_id_str),
        position_amount_map={asset_str: 10.0}, updated_timestamp_ts=signal_timestamp_ts))
    decision_obj = store_obj.insert_decision_plan(replace(source_decision_obj,
        decision_plan_id_int=None, status_str="planned", signal_timestamp_ts=signal_timestamp_ts,
        submission_timestamp_ts=submission_timestamp_ts, target_execution_timestamp_ts=target_timestamp_ts,
        decision_base_position_map={asset_str: 10.0}, entry_target_weight_map_dict={}, target_weight_map={},
        entry_priority_list=[], exit_asset_set={asset_str} if target_share_float == 0 else set(),
        target_share_map_dict={} if target_share_float == 0 else {asset_str: target_share_float},
        snapshot_metadata_dict={**source_decision_obj.snapshot_metadata_dict,
            "strategy_import_str": strategy_import_str,
            "mr_capsule_parking_mode_str": "spmo" if asset_str == "SPMO" else "bil"}))
    vplan_obj = store_obj.insert_vplan(replace(source_vplan_obj, vplan_id_int=None,
        decision_plan_id_int=decision_obj.decision_plan_id_int, signal_timestamp_ts=signal_timestamp_ts,
        submission_timestamp_ts=submission_timestamp_ts, target_execution_timestamp_ts=target_timestamp_ts,
        broker_snapshot_timestamp_ts=submission_timestamp_ts, live_reference_snapshot_timestamp_ts=submission_timestamp_ts,
        current_broker_position_map={asset_str: 10.0}, live_reference_price_map={asset_str: 100.0},
        target_share_map={asset_str: target_share_float}, order_delta_map={asset_str: target_share_float - 10.0},
        live_reference_source_map_dict={asset_str: "offline"}, submission_key_str=f"exit-policy:{asset_str}",
        vplan_row_list=[VPlanRow(asset_str, 10.0, target_share_float, target_share_float - 10.0,
            100.0, target_share_float * 100.0, "MOO", "offline")]))
    broker_obj = StubBrokerAdapter()
    install_daily_broker_stub(broker_obj)
    broker_obj.seed_live_price_snapshot(account_route_str=release_obj.account_route_str,
        asset_reference_price_map={asset_str: 100.0}, snapshot_timestamp_ts=submission_timestamp_ts)
    return store_obj, broker_obj, release_obj, vplan_obj, runner_dict, temporary_path


@pytest.mark.parametrize("asset_str,target_share_float,expected_recovery_float", [
    ("MSFT", 0.0, -6.0), ("MSFT", 3.0, None), ("SPMO", 0.0, -6.0), ("SPMO", 4.0, None),
])
def test_only_full_stock_or_spmo_exits_receive_market_completion(capsule_case, asset_str, target_share_float, expected_recovery_float):
    # A nonzero stock reduction is a defensive synthetic VPlan case, not a new
    # strategy signal rule. Recovery must not broaden an exit-to-zero policy.
    case_tuple = _exit_case(capsule_case, asset_str, target_share_float)
    store_obj, broker_obj, release_obj, vplan_obj, _, _ = seed_execution(case_tuple, {asset_str: -4.0})
    result_tuple = reconcile_case(case_tuple)
    assert not result_tuple[0].passed_bool
    final_result_tuple = close_case(case_tuple)
    assert final_result_tuple[0].passed_bool
    if expected_recovery_float is None:
        assert broker_obj.submitted_order_request_list == []
        assert final_result_tuple[1] == "completed_with_exceptions"
        assert store_obj.get_pod_state(release_obj.pod_id_str).position_amount_map[asset_str] == 6.0
        assert final_result_tuple[2][0]["side_str"] == "SELL"
        assert final_result_tuple[2][0]["residual_amount_float"] == target_share_float - 6.0
    else:
        request_obj, = broker_obj.submitted_order_request_list
        assert (request_obj.asset_str, request_obj.amount_float, request_obj.broker_order_type_str) == (asset_str, expected_recovery_float, "MKT")
        assert final_result_tuple[1] == "completed"
        assert store_obj.get_pod_state(release_obj.pod_id_str).position_amount_map.get(asset_str, 0.0) == 0.0
        late_fill_list = [row_dict for row_dict in store_obj.get_fill_row_dict_list_for_vplan(vplan_obj.vplan_id_int)
                         if row_dict["open_price_source_str"] == "late_execution"]
        assert len(late_fill_list) == 1
        assert late_fill_list[0]["official_open_price_float"] is None
    assert store_obj.get_latest_vplan_for_pod(release_obj.pod_id_str).target_share_map == {asset_str: target_share_float}


def test_partially_filled_stock_market_completion_is_not_retried_after_restart(capsule_case, monkeypatch):
    case_tuple = _exit_case(capsule_case, "MSFT")
    store_obj, broker_obj, release_obj, vplan_obj, _, temporary_path = seed_execution(case_tuple, {"MSFT": -4.0})
    request_list = []

    def partial_completion(**argument_dict):
        request_obj, = argument_dict["broker_order_request_list"]
        request_list.append(request_obj)
        record_obj = BrokerOrderRecord(broker_order_id_str="late-stock-partial", decision_plan_id_int=None,
            vplan_id_int=None, account_route_str=release_obj.account_route_str, asset_str="MSFT",
            order_request_key_str=request_obj.order_request_key_str, broker_order_type_str="MKT", unit_str="shares",
            amount_float=-6.0, filled_amount_float=2.0, status_str="Cancelled", submitted_timestamp_ts=RECONCILE_TIMESTAMP_TS,
            submission_key_str=vplan_obj.submission_key_str, raw_payload_dict={"snapshot_source_str": "completed_order"})
        fill_obj = BrokerOrderFill(broker_order_id_str=record_obj.broker_order_id_str, decision_plan_id_int=None,
            vplan_id_int=None, account_route_str=release_obj.account_route_str, asset_str="MSFT",
            fill_amount_float=-2.0, fill_price_float=99.5, fill_timestamp_ts=RECONCILE_TIMESTAMP_TS)
        broker_obj.seed_broker_order_state(record_obj, broker_order_fill_list=[fill_obj])
        snapshot_obj = broker_obj.get_account_snapshot(release_obj.account_route_str)
        broker_obj._snapshot_map[release_obj.account_route_str] = replace(snapshot_obj, position_amount_map={"MSFT": 4.0})
        return SubmitBatchResult(broker_order_record_list=[record_obj], broker_order_fill_list=[fill_obj])

    monkeypatch.setattr(broker_obj, "submit_order_request_list", partial_completion)
    result_tuple = reconcile_case(case_tuple)
    restarted_tuple = (LiveStateStore(str(temporary_path / "exit_policy.sqlite3")), *case_tuple[1:])
    reconcile_case(restarted_tuple)
    assert len(request_list) == 1
    assert not result_tuple[0].passed_bool and result_tuple[1] == "pending"
    final_result_tuple = close_case(restarted_tuple)
    assert final_result_tuple[2][0]["residual_amount_float"] == -4.0
    assert final_result_tuple[2][0]["side_str"] == "SELL"
    assert {row_dict["alert_kind_str"] for row_dict in alert_rows(store_obj)} == {"daily_exception"}


@pytest.mark.parametrize("hour_int,minute_int,recovery_expected_bool", [(12, 59, True), (13, 0, False), (13, 1, False)])
def test_market_completion_respects_actual_early_close(capsule_case, hour_int, minute_int, recovery_expected_bool):
    target_timestamp_ts = datetime(2024, 11, 29, 9, 30, tzinfo=MARKET_TIMEZONE_OBJ)
    as_of_ts = target_timestamp_ts.replace(hour=hour_int, minute=minute_int)
    case_tuple = _exit_case(capsule_case, "MSFT", target_timestamp_ts=target_timestamp_ts)
    store_obj, broker_obj, release_obj, _, _, _ = seed_execution(case_tuple, {"MSFT": -4.0})
    snapshot_obj = broker_obj.get_account_snapshot(release_obj.account_route_str)
    broker_obj._snapshot_map[release_obj.account_route_str] = replace(snapshot_obj, snapshot_timestamp_ts=as_of_ts)
    result_tuple = reconcile_case(case_tuple, as_of_ts)
    assert result_tuple[0].passed_bool == (not recovery_expected_bool)
    assert len(broker_obj.submitted_order_request_list) == int(recovery_expected_bool)
    assert result_tuple[1] == ("pending" if recovery_expected_bool else "completed_with_exceptions")
    if recovery_expected_bool:
        assert broker_obj.submitted_order_request_list[0].execution_deadline_timestamp_str == "2024-11-29T13:00:00-05:00"
        # Stub send reports only ACKs. A later normal poll records its fill;
        # neither the send nor lifecycle completion waits on fill reporting.
        reconcile_case(case_tuple, as_of_ts)
        assert len(broker_obj.submitted_order_request_list) == 1
        late_fill_list = [row_dict for row_dict in store_obj.get_fill_row_dict_list_for_vplan(case_tuple[3].vplan_id_int)
                         if row_dict["open_price_source_str"] == "late_execution"]
        assert datetime.fromisoformat(late_fill_list[0]["fill_timestamp_str"]) == as_of_ts


@pytest.mark.parametrize("same_request_key_bool", [False, True])
def test_concurrent_workers_can_claim_only_one_recovery_per_asset(capsule_case, same_request_key_bool):
    case_tuple = _exit_case(capsule_case, "MSFT")
    store_obj, _, release_obj, vplan_obj, _, temporary_path = seed_execution(case_tuple, {"MSFT": -4.0})
    decision_obj = store_obj.get_decision_plan_by_id(vplan_obj.decision_plan_id_int)
    original_request_obj, = build_broker_order_request_list_from_vplan(vplan_obj)
    worker_count_int = 6
    worker_store_list = [LiveStateStore(str(temporary_path / "exit_policy.sqlite3")) for _ in range(worker_count_int)]
    barrier_obj = Barrier(worker_count_int)

    def claim_one(worker_index_int):
        request_obj = replace(original_request_obj, amount_float=-6.0, broker_order_type_str="MKT",
            order_request_key_str=f"{vplan_obj.submission_key_str}:late:{0 if same_request_key_bool else worker_index_int}")
        barrier_obj.wait(timeout=10)
        return claim_daily_completion_request(worker_store_list[worker_index_int], release_obj, decision_obj,
            vplan_obj, request_obj, RECONCILE_TIMESTAMP_TS)

    with ThreadPoolExecutor(max_workers=worker_count_int) as executor_obj:
        result_list = list(executor_obj.map(claim_one, range(worker_count_int)))
    assert sum(result_list) == 1
    with store_obj._connect() as connection_obj:
        assert connection_obj.execute("SELECT COUNT(*) FROM daily_completion_request").fetchone()[0] == 1
