"""Identical execution tuples must reconcile through the real capsule persistence path."""
from alpha.live.execution_engine import build_broker_order_request_list_from_vplan
from alpha.live.models import BrokerOrderFill, BrokerOrderRecord
from alpha.live.runner import post_execution_reconcile
from test_live_mr_capsule_recovery import capsule_case, CLOSE_TIMESTAMP_TS


def test_identical_partial_executions_complete_capsule_cycle(capsule_case):
    store_obj, broker_obj, release_obj, vplan_obj, runner_kwarg_dict, _ = capsule_case
    store_obj.mark_vplan_status(vplan_obj.vplan_id_int, "submitted")
    store_obj.mark_decision_plan_status(vplan_obj.decision_plan_id_int, "submitted")
    for request_obj in build_broker_order_request_list_from_vplan(vplan_obj):
        order_id_str = f"original:{request_obj.asset_str}"
        order_obj = BrokerOrderRecord(
            broker_order_id_str=order_id_str, decision_plan_id_int=None, vplan_id_int=None,
            account_route_str=release_obj.account_route_str, asset_str=request_obj.asset_str,
            order_request_key_str=request_obj.order_request_key_str, broker_order_type_str="MOO", unit_str="shares",
            amount_float=request_obj.amount_float, filled_amount_float=abs(request_obj.amount_float),
            remaining_amount_float=0.0, status_str="Filled", submitted_timestamp_ts=vplan_obj.submission_timestamp_ts,
            submission_key_str=vplan_obj.submission_key_str,
            raw_payload_dict={"snapshot_source_str": "completed_order", "open_order_observed_bool": False},
        )
        fill_list = [BrokerOrderFill(
            broker_order_id_str=order_id_str, decision_plan_id_int=None, vplan_id_int=None,
            account_route_str=release_obj.account_route_str, asset_str=request_obj.asset_str,
            fill_amount_float=request_obj.amount_float / 2, fill_price_float=request_obj.sizing_reference_price_float,
            fill_timestamp_ts=vplan_obj.target_execution_timestamp_ts,
            raw_payload_dict={"exec_id_str": f"{request_obj.asset_str}:{execution_int}"},
        ) for execution_int in (1, 2)]
        broker_obj.seed_broker_order_state(order_obj, broker_order_fill_list=fill_list)
    broker_obj.seed_account_snapshot(
        account_route_str=release_obj.account_route_str, cash_float=1000.0, total_value_float=100000.0,
        position_amount_map={"AAPL": 80.0, "BIL": 880.0, "MSFT": 5.0},
        snapshot_timestamp_ts=CLOSE_TIMESTAMP_TS, session_mode_str="paper",
    )
    result_dict = post_execution_reconcile(store_obj, broker_obj, CLOSE_TIMESTAMP_TS, **runner_kwarg_dict)
    assert result_dict["completed_vplan_count_int"] == 1
    assert len(store_obj.get_fill_row_dict_list_for_vplan(vplan_obj.vplan_id_int)) == 4
    assert store_obj.get_vplan_by_id(vplan_obj.vplan_id_int).status_str == "completed"
    assert store_obj.get_pod_state(release_obj.pod_id_str).position_amount_map == {"AAPL": 80.0, "BIL": 880.0, "MSFT": 5.0}
    assert broker_obj.submitted_order_request_list == []
