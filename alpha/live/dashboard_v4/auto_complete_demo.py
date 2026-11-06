"""Disposable real-reader recovery UI fixture; never opens live config or a broker."""
from contextlib import closing
from copy import deepcopy
from dataclasses import asdict
from datetime import timedelta
import json
import sqlite3

from alpha.live import auto_complete
from alpha.live.dashboard_v4.app import create_app
from alpha.live.dashboard_v4.demo import build_demo_workspace_tuple, DEMO_NOW_TS
from alpha.live.dashboard_v4.orders_demo import DemoOrderBroker
from alpha.live.dashboard_v4.pod_demo import _DemoStateStore
from alpha.live.models import BrokerOrderAck, BrokerOrderEvent, BrokerOrderFill, BrokerOrderRecord, BrokerOrderRequest, BrokerSnapshot


def create_auto_completion_demo_app(case_str="planned"):
    """Cases: planned, sent, unconfirmed, completed, partial, not_completed, not_automatic."""
    if case_str not in {"planned", "sent", "unconfirmed", "completed", "partial", "not_completed", "not_automatic"}:
        raise ValueError("Unknown synthetic recovery case")
    workspace_dict, finance_snapshot_obj, provider_obj = build_demo_workspace_tuple()
    pod_id_str, asset_str = "demo_1_1", "MSFT"
    target_obj = provider_obj.get_target_for_pod(pod_id_str)
    open_ts = DEMO_NOW_TS.replace(hour=13, minute=30, second=0)
    now_ts = open_ts + (timedelta(minutes=4, seconds=30) if case_str in {"planned", "not_automatic"}
                       else timedelta(minutes=31) if case_str in {"partial", "not_completed"}
                       else timedelta(minutes=5, seconds=10))
    with closing(_DemoStateStore(target_obj.db_path_str)) as store_obj:
        vplan_obj = store_obj.get_latest_vplan_for_pod(pod_id_str)
        original_dict = next(row_dict for row_dict in store_obj.get_broker_order_row_dict_list_for_vplan(vplan_obj.vplan_id_int)
                             if row_dict["asset_str"] == asset_str)
        identity_dict = {"decision_plan_id_int": vplan_obj.decision_plan_id_int, "vplan_id_int": vplan_obj.vplan_id_int,
            "account_route_str": target_obj.release_obj.account_route_str, "asset_str": asset_str}
        message_str = "Not enough buying power" if case_str == "not_automatic" else "Order would cross related resting order"
        with closing(sqlite3.connect(target_obj.db_path_str)) as connection_obj, connection_obj:
            connection_obj.execute("UPDATE vplan_broker_order SET status_str='Cancelled' WHERE broker_order_id_str=?",
                                   (original_dict["broker_order_id_str"],))
            connection_obj.execute("UPDATE vplan_broker_ack SET broker_response_ack_bool=1,ack_status_str='broker_acked',response_timestamp_str=? WHERE vplan_id_int=?",
                                   ((open_ts - timedelta(minutes=6)).isoformat(), vplan_obj.vplan_id_int))
            connection_obj.execute("UPDATE vplan_broker_ack SET ack_source_str='broker_error' WHERE broker_order_id_str=?",
                                   (original_dict["broker_order_id_str"],))
            connection_obj.execute("UPDATE vplan SET status_str='submitted',submit_ack_status_str='complete',missing_ack_count_int=0,updated_timestamp_str=? WHERE vplan_id_int=?",
                                   ((open_ts + timedelta(seconds=3)).isoformat(), vplan_obj.vplan_id_int))
            connection_obj.execute("UPDATE decision_plan SET status_str='submitted',updated_timestamp_str=? WHERE decision_plan_id_int=?",
                                   ((open_ts + timedelta(seconds=3)).isoformat(), vplan_obj.decision_plan_id_int))
            connection_obj.execute("DELETE FROM broker_snapshot_cache")
        store_obj.insert_vplan_broker_order_event_list([BrokerOrderEvent(**identity_dict,
            broker_order_id_str=original_dict["broker_order_id_str"], order_request_key_str=original_dict["order_request_key_str"],
            status_str="Cancelled", filled_amount_float=0, event_timestamp_ts=open_ts - timedelta(minutes=6),
            message_str=message_str, raw_payload_dict={"error_code_int": 202})])
        quantity_float = abs(original_dict["amount_float"])
        filled_float = quantity_float if case_str == "completed" else 7.0 if case_str == "partial" else 0.0
        position_dict = dict(vplan_obj.target_share_map)
        position_dict[asset_str] = filled_float
        store_obj.upsert_broker_snapshot_cache(BrokerSnapshot(target_obj.release_obj.account_route_str,
            now_ts - timedelta(seconds=1), 10000, 20000, position_dict), release_obj=target_obj.release_obj)
        if case_str not in {"planned", "not_automatic"}:
            submission_key_str = vplan_obj.submission_key_str or f"vplan:{vplan_obj.decision_plan_id_int}"
            reference_str = f"{submission_key_str}:auto-complete:{asset_str}"
            order_id_str = None if case_str == "unconfirmed" else "DEMO-AUTO-1"
            terminal_bool = case_str in {"completed", "partial", "not_completed"}
            ended_str = (now_ts - timedelta(seconds=2)).isoformat() if terminal_bool else None
            request_obj = BrokerOrderRequest(target_obj.release_obj.release_id_str, pod_id_str,
                target_obj.release_obj.account_route_str, submission_key_str, reference_str, asset_str,
                "MKT", "MarketOrder", "shares", original_dict["amount_float"], False, None, 220.0, 20000.0,
                decision_plan_id_int=vplan_obj.decision_plan_id_int, vplan_id_int=vplan_obj.vplan_id_int,
                execution_deadline_timestamp_str=(open_ts + timedelta(minutes=30)).isoformat(),
                submission_deadline_timestamp_str=(open_ts + timedelta(minutes=6)).isoformat())
            with closing(sqlite3.connect(target_obj.db_path_str)) as connection_obj, connection_obj:
                auto_complete._ensure_schema(connection_obj)
                connection_obj.execute("""INSERT INTO auto_completion_attempt
                    (vplan_id_int,asset_str,decision_plan_id_int,pod_id_str,account_route_str,original_broker_order_id_str,
                     original_message_str,attempted_timestamp_str,updated_timestamp_str,ended_timestamp_str,last_checked_timestamp_str,
                     order_request_key_str,status_str,reason_str,quantity_float,remaining_quantity_float,broker_order_id_str,request_json_str)
                     VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""", (vplan_obj.vplan_id_int, asset_str,
                    vplan_obj.decision_plan_id_int, pod_id_str, target_obj.release_obj.account_route_str,
                    original_dict["broker_order_id_str"], message_str, (open_ts + timedelta(minutes=5, seconds=2)).isoformat(),
                    ended_str or now_ts.isoformat(), ended_str, ended_str, reference_str,
                    "completed" if case_str == "completed" else "not_completed" if terminal_bool else "working",
                    "Partly filled; order ended" if case_str == "partial" else "IBKR cancelled it again",
                    quantity_float, quantity_float - filled_float, order_id_str, json.dumps(asdict(request_obj))))
            if order_id_str:
                order_status_str = "Filled" if case_str == "completed" else "Cancelled" if terminal_bool else "Submitted"
                store_obj.upsert_vplan_broker_order_record_list([BrokerOrderRecord(**identity_dict,
                    broker_order_id_str=order_id_str, order_request_key_str=reference_str, broker_order_type_str="MKT", unit_str="shares",
                    amount_float=quantity_float, filled_amount_float=filled_float, remaining_amount_float=quantity_float-filled_float,
                    avg_fill_price_float=220.2 if filled_float else None, status_str=order_status_str,
                    submitted_timestamp_ts=open_ts + timedelta(minutes=5, seconds=2), last_status_timestamp_ts=now_ts - timedelta(seconds=2),
                    submission_key_str=submission_key_str)])
                store_obj.upsert_vplan_broker_ack_list([BrokerOrderAck(**identity_dict, broker_order_id_str=order_id_str,
                    order_request_key_str=reference_str, broker_order_type_str="MKT", local_submit_ack_bool=True,
                    broker_response_ack_bool=True, ack_status_str="broker_acked", ack_source_str="open_order",
                    response_timestamp_ts=open_ts + timedelta(minutes=5, seconds=3))])
                if filled_float:
                    store_obj.upsert_vplan_fill_list([BrokerOrderFill(**identity_dict, broker_order_id_str=order_id_str,
                        fill_amount_float=filled_float, fill_price_float=220.2, fill_timestamp_ts=now_ts-timedelta(seconds=2),
                        official_open_price_float=220.0, open_price_source_str="DEMO official open",
                        raw_payload_dict={"exec_id_str": "DEMO-AUTO-FILL"})])
    row_dict = next(row_dict for row_dict in provider_obj.row_list if row_dict["pod_id_str"] == pod_id_str)
    row_dict.update(as_of_timestamp_str=now_ts.isoformat(), latest_submit_ack_status_str="complete", missing_ack_count_int=0,
        broker_ack_count_int=3, required_action_dict={}, next_action_str="wait", reason_code_str="waiting_for_post_execution_reconcile")
    workspace_dict["operations_account_list"] = [item_dict for item_dict in workspace_dict["operations_account_list"] if item_dict["pod_id"] == pod_id_str]
    workspace_dict["summary_dict"].update(as_of_timestamp_str=now_ts.isoformat(), pod_row_dict_list=[row_dict])
    original_activity_fn = provider_obj.get_activity_source_dict
    recovery_event_list = []
    if case_str != "not_automatic":
        event_type_list = ["auto_complete_planned"]
        if case_str not in {"planned", "unconfirmed"}:
            event_type_list.append("auto_complete_sent")
        if case_str in {"completed", "partial", "not_completed"}:
            event_type_list.append("auto_complete_done" if case_str == "completed" else "auto_complete_not_completed")
        for index_int, event_type_str in enumerate(event_type_list):
            recovery_event_list.append({"timestamp_str": (now_ts - timedelta(seconds=4-index_int)).isoformat(),
                "event_type_str": event_type_str, "level_str": "warning" if event_type_str.endswith("not_completed") else "info",
                "mode_str": "live", "pod_id_str": pod_id_str, "decision_plan_id_int": vplan_obj.decision_plan_id_int,
                "vplan_id_int": vplan_obj.vplan_id_int, "source_str": "Demo event log", "payload_dict": {
                    "release_id_str": target_obj.release_obj.release_id_str, "asset_str": asset_str, "quantity_float": quantity_float}})

    def recovery_activity_source_dict(*, as_of_ts, days_int):
        source_dict = original_activity_fn(as_of_ts=as_of_ts, days_int=days_int)
        source_dict["event_list"].extend(deepcopy(recovery_event_list))
        return source_dict

    provider_obj.get_activity_source_dict = recovery_activity_source_dict
    broker_obj = DemoOrderBroker(provider_obj)
    app_obj = create_app(provider_obj, demo_bool=True, orders_enabled_bool=True,
        order_broker_factory_fn=lambda release_obj: broker_obj,
        workspace_snapshot_fn=lambda: (deepcopy(workspace_dict), finance_snapshot_obj),
        operations_workspace_fn=lambda: deepcopy(workspace_dict), now_fn=lambda: now_ts)
    app_obj.config["auto_complete_demo_provider_obj"] = provider_obj
    return app_obj
