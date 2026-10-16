"""Synthetic broker for the V4 demo order panel (--demo --demo-tools). Never contacts IBKR.

Positions come from the demo pod's own saved snapshot. A demo order stays "open"
until the demo restarts, so a second ticket for the same symbol shows the
"order already open" stop.
"""

from __future__ import annotations

from datetime import UTC, datetime
import threading
from types import SimpleNamespace

from alpha.live.daily_broker import DailyExecutionSnapshot
from alpha.live.dashboard_authorization import _ReadOnlyConfirmationStore
from alpha.live.models import BrokerSnapshot, SubmitBatchResult


class DemoOrderBroker:
    def __init__(self, provider_obj):
        self.provider_obj = provider_obj
        self.open_row_list = []
        self._lock_obj = threading.Lock()
        self._order_id_int = 9000

    def _saved_snapshot(self, account_route_str):
        for target_obj in self.provider_obj.get_target_list():
            if target_obj.release_obj.account_route_str != account_route_str:
                continue
            store_obj = _ReadOnlyConfirmationStore(target_obj.db_path_str)
            try:
                return store_obj.get_latest_broker_snapshot_for_account(account_route_str)
            finally:
                store_obj.connection_obj.close()
        raise ConnectionError("Demo account not found.")

    def get_daily_execution_snapshot(self, account_route_str):
        saved_obj = self._saved_snapshot(account_route_str)
        now_ts = datetime.now(UTC)
        snapshot_obj = BrokerSnapshot(account_route_str=account_route_str, snapshot_timestamp_ts=now_ts,
            cash_float=saved_obj.cash_float if saved_obj else 0.0, total_value_float=saved_obj.total_value_float if saved_obj else 0.0,
            position_amount_map=dict(saved_obj.position_amount_map) if saved_obj else {})
        with self._lock_obj:
            open_list = [dict(row_dict) for row_dict in self.open_row_list if row_dict["account_route_str"] == account_route_str]
        return DailyExecutionSnapshot(snapshot_obj, open_list, now_ts, now_ts)

    def submit_order_request_list(self, *, account_route_str, broker_order_request_list, submitted_timestamp_ts):
        record_list = []
        with self._lock_obj:
            for request_obj in broker_order_request_list:
                self._order_id_int += 1
                order_id_str = f"DEMO-{self._order_id_int}"
                self.open_row_list.append({"account_route_str": account_route_str, "asset_str": request_obj.asset_str,
                    "order_ref_str": request_obj.order_request_key_str, "order_id_int": self._order_id_int,
                    "perm_id_int": self._order_id_int, "client_id_int": 99,
                    "side_str": "BUY" if request_obj.amount_float > 0 else "SELL",
                    "amount_float": abs(request_obj.amount_float), "remaining_amount_float": abs(request_obj.amount_float),
                    "status_str": "PreSubmitted"})
                record_list.append(SimpleNamespace(broker_order_id_str=order_id_str))
        return SubmitBatchResult(broker_order_record_list=record_list, ack_coverage_ratio_float=1.0,
            submit_ack_status_str="complete")
