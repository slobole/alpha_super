"""Explicit complete offline broker state for daily reconciliation regressions."""
from dataclasses import replace

from alpha.live.daily_broker import DailyExecutionSnapshot


def install_daily_broker_stub(broker_obj):
    broker_obj.daily_extra_open_row_list = []
    broker_obj.daily_cancel_ref_list = []

    def get_daily_execution_snapshot(account_route_str):
        snapshot_obj = broker_obj.get_account_snapshot(account_route_str)
        order_list = list(broker_obj._broker_order_record_map.get(account_route_str, {}).values())
        row_list = [{"account_route_str": account_route_str, "asset_str": order_obj.asset_str,
            "order_ref_str": order_obj.order_request_key_str or "", "order_id_int": index_int + 1,
            "perm_id_int": index_int + 1000, "client_id_int": 31, "side_str": "BUY" if order_obj.amount_float > 0 else "SELL",
            "amount_float": abs(order_obj.amount_float), "remaining_amount_float": order_obj.remaining_amount_float,
            "status_str": order_obj.status_str, "test_broker_order_id_str": order_obj.broker_order_id_str}
            for index_int, order_obj in enumerate(order_list)
            if order_obj.status_str not in {"Filled", "Cancelled", "ApiCancelled", "Rejected", "Expired", "Inactive"}]
        row_list.extend(dict(row_dict) for row_dict in broker_obj.daily_extra_open_row_list)
        known_id_set = {order_obj.broker_order_id_str for order_obj in order_list}
        known_id_set.update(row_dict.get("test_broker_order_id_str") for row_dict in row_list)
        if set(snapshot_obj.open_order_id_list) - known_id_set:
            raise ValueError("Synthetic open order lacks its complete symbol observation.")
        return DailyExecutionSnapshot(snapshot_obj, row_list,
            snapshot_obj.snapshot_timestamp_ts, snapshot_obj.snapshot_timestamp_ts)

    def cancel_daily_owned_orders(account_route_str, owned_order_ref_set, *, session_close_timestamp_ts):
        snapshot_obj = broker_obj.get_account_snapshot(account_route_str)
        assert snapshot_obj.snapshot_timestamp_ts >= session_close_timestamp_ts
        broker_obj.daily_cancel_ref_list.append(set(owned_order_ref_set))
        cancelled_id_set = set()
        for order_id_str, order_obj in broker_obj._broker_order_record_map.get(account_route_str, {}).items():
            if order_obj.order_request_key_str in owned_order_ref_set:
                broker_obj._broker_order_record_map[account_route_str][order_id_str] = replace(order_obj, status_str="Cancelled")
                cancelled_id_set.add(order_id_str)
        broker_obj.daily_extra_open_row_list = [row_dict for row_dict in broker_obj.daily_extra_open_row_list
            if row_dict.get("order_ref_str") not in owned_order_ref_set]
        broker_obj._snapshot_map[account_route_str] = replace(snapshot_obj,
            open_order_id_list=[order_id_str for order_id_str in snapshot_obj.open_order_id_list if order_id_str not in cancelled_id_set])
        return get_daily_execution_snapshot(account_route_str)

    broker_obj.get_daily_execution_snapshot = get_daily_execution_snapshot
    broker_obj.cancel_daily_owned_orders = cancel_daily_owned_orders
    return broker_obj
