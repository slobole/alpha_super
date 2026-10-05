"""Accepted residuals permit tomorrow's cycle using persisted broker holdings."""
from datetime import datetime

import pytest

from alpha.live import runner as runner_module
from alpha.live.models import BrokerOrderRecord, DecisionPlan, SubmitBatchResult
from alpha.live.runner import build_decision_plans, eod_snapshot
from alpha.live.state_store_v2 import LiveStateStore
from test_live_mr_capsule_execution_policy import alert_rows, reconcile_case, seed_execution
from test_live_mr_capsule_recovery import (
    MARKET_TIMEZONE_OBJ, RECONCILE_TIMESTAMP_TS, capsule_case,
)


@pytest.mark.parametrize("outcome_str", ["missed_buy", "partial_buy", "rejected_bil_recovery"])
def test_accepted_residual_advances_through_eod_with_actual_holdings(
    capsule_case, monkeypatch, outcome_str,
):
    buy_fill_float = {"missed_buy": 0.0, "partial_buy": 30.0, "rejected_bil_recovery": 80.0}[outcome_str]
    bil_fill_float = -40.0 if outcome_str == "rejected_bil_recovery" else -100.0
    state_store_obj, broker_obj, release_obj, vplan_obj, runner_kwarg_dict, tmp_path = seed_execution(
        capsule_case, {"AAPL": buy_fill_float, "BIL": bil_fill_float})
    original_decision_obj = state_store_obj.get_decision_plan_by_id(vplan_obj.decision_plan_id_int)
    recovery_request_list = []

    def reject_recovery(**submission_kwarg_dict):
        request_obj, = submission_kwarg_dict["broker_order_request_list"]
        recovery_request_list.append(request_obj)
        record_obj = BrokerOrderRecord(
            broker_order_id_str="late-rejected", decision_plan_id_int=None, vplan_id_int=None,
            account_route_str=release_obj.account_route_str, asset_str=request_obj.asset_str,
            order_request_key_str=request_obj.order_request_key_str, broker_order_type_str="MKT",
            unit_str="shares", amount_float=request_obj.amount_float, filled_amount_float=0.0,
            status_str="Inactive", submitted_timestamp_ts=RECONCILE_TIMESTAMP_TS,
            submission_key_str=vplan_obj.submission_key_str,
            raw_payload_dict={"snapshot_source_str": "completed_order"})
        broker_obj.seed_broker_order_state(record_obj)
        return SubmitBatchResult(broker_order_record_list=[record_obj])

    monkeypatch.setattr(broker_obj, "submit_order_request_list", reject_recovery)
    result_tuple = reconcile_case(capsule_case)
    assert result_tuple[0].passed_bool and result_tuple[1] == "accepted_residual"
    expected_position_dict = {"AAPL": buy_fill_float, "BIL": 980.0 + bil_fill_float, "MSFT": 5.0}
    assert state_store_obj.get_pod_state(release_obj.pod_id_str).position_amount_map == expected_position_dict
    assert state_store_obj.get_latest_vplan_for_pod(release_obj.pod_id_str).status_str == "completed"
    assert state_store_obj.get_decision_plan_by_id(vplan_obj.decision_plan_id_int).status_str == "completed"

    # Restart cannot resend the accepted missed buy or the rejected recovery.
    restarted_store_obj = LiveStateStore(str(tmp_path / "recovery.sqlite3"))
    restarted_case_tuple = (restarted_store_obj, *capsule_case[1:])
    assert reconcile_case(restarted_case_tuple)[0].passed_bool
    expected_recovery_count_int = int(outcome_str == "rejected_bil_recovery")
    assert len(recovery_request_list) == expected_recovery_count_int
    if recovery_request_list:
        assert (recovery_request_list[0].asset_str, recovery_request_list[0].amount_float) == ("BIL", -60.0)
    alert_row_dict, = alert_rows(restarted_store_obj)
    assert alert_row_dict["alert_kind_str"] == "accepted_residual"

    # The next completed signal session is February 1; its orders would execute
    # on February 2. Exercise the real EOD gate and SQL state write before build.
    next_close_ts = datetime(2024, 2, 1, 16, 0, tzinfo=MARKET_TIMEZONE_OBJ)
    next_decision_ts = datetime(2024, 2, 1, 16, 10, tzinfo=MARKET_TIMEZONE_OBJ)
    broker_snapshot_obj = broker_obj.get_account_snapshot(release_obj.account_route_str)
    broker_obj.seed_account_snapshot(
        account_route_str=release_obj.account_route_str,
        cash_float=broker_snapshot_obj.cash_float, total_value_float=broker_snapshot_obj.net_liq_float,
        position_amount_map=expected_position_dict, snapshot_timestamp_ts=next_close_ts,
        session_mode_str="paper")
    snapshot_result_dict = eod_snapshot(
        restarted_store_obj, broker_obj, next_decision_ts, "paper", **runner_kwarg_dict)
    assert snapshot_result_dict["eod_snapshot_count_int"] == 1
    assert snapshot_result_dict["blocked_action_count_int"] == 0
    eod_state_obj = restarted_store_obj.get_pod_state(release_obj.pod_id_str)
    assert eod_state_obj.snapshot_stage_str == "eod"
    assert eod_state_obj.snapshot_source_str == "broker"
    assert eod_state_obj.updated_timestamp_ts == next_close_ts
    assert eod_state_obj.position_amount_map == expected_position_dict
    builder_state_list = []

    def next_decision(pod_state_obj, **_kwarg_dict):
        builder_state_list.append(pod_state_obj)
        # Signal generation is intentionally stubbed; this test checks actual
        # holdings reach the next decision, not strategy signals or sizing.
        return DecisionPlan(
            release_id_str=release_obj.release_id_str, user_id_str=release_obj.user_id_str,
            pod_id_str=release_obj.pod_id_str, account_route_str=release_obj.account_route_str,
            signal_timestamp_ts=next_close_ts,
            submission_timestamp_ts=datetime(2024, 2, 2, 9, 22, tzinfo=MARKET_TIMEZONE_OBJ),
            target_execution_timestamp_ts=datetime(2024, 2, 2, 9, 30, tzinfo=MARKET_TIMEZONE_OBJ),
            execution_policy_str="next_open_moo",
            decision_base_position_map=dict(pod_state_obj.position_amount_map),
            snapshot_metadata_dict={
                "strategy_import_str": release_obj.strategy_import_str,
                "norgate_snapshot_date_str": "2024-02-01",
                "decision_nav_float": pod_state_obj.total_value_float,
                "decision_cash_float": pod_state_obj.cash_float,
            },
            strategy_state_dict=dict(pod_state_obj.strategy_state_dict))

    monkeypatch.setattr(runner_module, "_load_release_list_validate_and_sync", lambda *_args, **_kwargs: [release_obj])
    monkeypatch.setattr(runner_module.scheduler_utils, "select_due_release_list", lambda release_list, _as_of_ts: release_list)
    monkeypatch.setattr(runner_module.strategy_host, "build_decision_plan_for_release", next_decision)
    next_store_obj = LiveStateStore(str(tmp_path / "recovery.sqlite3"))
    build_result_dict = build_decision_plans(
        next_store_obj, next_decision_ts, str(tmp_path),
        auto_sync_norgate_snapshots_bool=False, **runner_kwarg_dict)
    assert build_result_dict["created_decision_plan_count_int"] == 1
    assert build_result_dict["skipped_decision_plan_count_int"] == 0
    assert builder_state_list == [eod_state_obj]
    next_plan_obj = next_store_obj.get_latest_decision_plan_for_pod(release_obj.pod_id_str)
    assert next_plan_obj.decision_plan_id_int != original_decision_obj.decision_plan_id_int
    assert next_plan_obj.signal_timestamp_ts == next_close_ts
    assert next_plan_obj.decision_base_position_map == expected_position_dict
    residual_asset_str = "BIL" if outcome_str == "rejected_bil_recovery" else "AAPL"
    assert next_plan_obj.decision_base_position_map[residual_asset_str] != vplan_obj.target_share_map[residual_asset_str]
    assert next_plan_obj.snapshot_metadata_dict["decision_cash_float"] == broker_snapshot_obj.cash_float
    assert len(recovery_request_list) == expected_recovery_count_int
    assert broker_obj.submitted_order_request_list == []
    assert next_store_obj.get_latest_vplan_for_pod(release_obj.pod_id_str).target_share_map == vplan_obj.target_share_map
    assert next_store_obj.get_decision_plan_by_id(vplan_obj.decision_plan_id_int).target_share_map_dict == original_decision_obj.target_share_map_dict
