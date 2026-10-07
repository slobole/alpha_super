"""Offline complete-account snapshots and selective cross-client cancellation."""
from contextlib import contextmanager
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from alpha.live import daily_broker
from alpha.live.ibkr_socket_client import IBKRSocketClient


class Event:
    def __init__(self):
        self.handler_list = []
    def __iadd__(self, handler_fn):
        self.handler_list.append(handler_fn)
        return self
    def __isub__(self, handler_fn):
        self.handler_list.remove(handler_fn)
        return self
    def emit(self, *arg_tuple):
        for handler_fn in list(self.handler_list):
            handler_fn(*arg_tuple)


def _trade(ref_str="owned", client_int=31, account_str="DU1", order_int=8, perm_int=80, asset_str="SPY"):
    return SimpleNamespace(order=SimpleNamespace(account=account_str, clientId=client_int,
        orderId=order_int, permId=perm_int, action="SELL", totalQuantity=4.0, orderRef=ref_str),
        orderStatus=SimpleNamespace(permId=perm_int, remaining=4.0, status="Submitted"),
        contract=SimpleNamespace(symbol=asset_str))


class FakeIB:
    def __init__(self, state_obj, client_int):
        self.state_obj = state_obj
        self.client_int = client_int
        self.errorEvent = Event()
        self.accountSummaryEvent = Event()
        self.connected_bool = True
        self.summary_active_set = set()
        self.summary_request_int = 100
        self.wrapper = SimpleNamespace(startReq=lambda request_int: request_int)
        self.client = SimpleNamespace(getReqId=self._next_request_int,
            reqAccountSummary=self._request_summary, cancelAccountSummary=self._cancel_summary)
    def _next_request_int(self):
        self.summary_request_int += 1
        return self.summary_request_int
    def _request_summary(self, request_int, group_str, tags_str):
        assert group_str == "All"
        self.summary_active_set.add(request_int)
        assert len(self.summary_active_set) <= 2
        self.reqAccountSummary()
    def _cancel_summary(self, request_int):
        self.summary_active_set.remove(request_int)
        self.state_obj.summary_cancel_list.append(request_int)
    def _run(self, future_obj):
        return None
    def isConnected(self):
        return self.connected_bool
    def managedAccounts(self):
        return ["DU1", "DU2"]
    def _call(self, name_str):
        self.state_obj.call_list.append((self.client_int, name_str))
        if self.state_obj.failure_str == name_str:
            raise TimeoutError("incomplete " + name_str)
    def reqAllOpenOrders(self):
        self._call("all_orders")
        if self.state_obj.order_hook_fn:
            self.state_obj.order_hook_fn(self)
        return deepcopy(self.state_obj.trade_list)
    def reqPositions(self):
        self._call("positions")
        return deepcopy(self.state_obj.position_list)
    def reqAccountSummary(self):
        self._call("account_summary")
        for value_obj in self.state_obj.account_value_list:
            self.accountSummaryEvent.emit(value_obj)
        if self.state_obj.error_code_int:
            self.errorEvent.emit(-1, self.state_obj.error_code_int, "synthetic")
        if self.state_obj.disconnect_bool:
            self.connected_bool = False
    def cancelOrder(self, order_obj):
        self._call("cancel")
        assert self.client_int == order_obj.clientId
        self.state_obj.cancel_list.append((self.client_int, order_obj.account, order_obj.orderRef, order_obj.orderId))
        if self.state_obj.confirm_bool:
            self.state_obj.trade_list = [trade_obj for trade_obj in self.state_obj.trade_list
                if (trade_obj.order.account, trade_obj.order.clientId, trade_obj.order.orderId, trade_obj.order.permId)
                != (order_obj.account, order_obj.clientId, order_obj.orderId, order_obj.permId)]
    def sleep(self, duration_float):
        pass


@pytest.fixture
def socket_obj():
    state_obj = SimpleNamespace(trade_list=[], call_list=[], cancel_list=[], connection_list=[],
        summary_cancel_list=[], position_list=[SimpleNamespace(account="DU1", position=-3.0,
            contract=SimpleNamespace(symbol="DBC", secType="STK", currency="USD"))],
        account_value_list=[SimpleNamespace(account="DU1", tag=tag_str, currency="USD", value=value_str)
            for tag_str, value_str in [("TotalCashValue", "12500"), ("NetLiquidation", "12000")]],
        failure_str=None, error_code_int=None, disconnect_bool=False, confirm_bool=True,
        order_hook_fn=None, connect_hook_fn=None)
    class Socket:
        timeout_seconds_float = 0.01
        @contextmanager
        def daily_connection(self, client_id_int=None):
            client_int = 31 if client_id_int is None else client_id_int
            state_obj.connection_list.append(client_int)
            if state_obj.connect_hook_fn:
                state_obj.connect_hook_fn(client_int)
            broker_obj = FakeIB(state_obj, client_int)
            try:
                yield broker_obj
            finally:
                assert not broker_obj.summary_active_set
                assert not broker_obj.errorEvent.handler_list
                assert not broker_obj.accountSummaryEvent.handler_list
    socket_obj = Socket()
    socket_obj.state_obj = state_obj
    return socket_obj


def test_snapshot_refreshes_complete_all_client_orders_and_account_scoped_holdings(socket_obj):
    socket_obj.state_obj.trade_list = [_trade(), _trade("other_client", 77, perm_int=81),
        _trade("foreign_account", 88, "DU2")]
    result_obj = daily_broker.get_daily_execution_snapshot(socket_obj, "DU1")
    assert result_obj.complete_bool is True
    assert result_obj.broker_snapshot_obj.position_amount_map == {"DBC": -3.0}
    assert result_obj.broker_snapshot_obj.cash_float == 12500.0
    assert result_obj.broker_snapshot_obj.net_liq_float == 12000.0
    assert {row_dict["client_id_int"] for row_dict in result_obj.open_order_row_list} == {31, 77}
    assert all(row_dict["account_route_str"] == "DU1" for row_dict in result_obj.open_order_row_list)
    assert result_obj.refresh_started_timestamp_ts <= result_obj.broker_snapshot_obj.snapshot_timestamp_ts <= result_obj.refreshed_timestamp_ts
    assert socket_obj.state_obj.call_list == [(31, "all_orders"), (31, "positions"), (31, "account_summary"), (31, "all_orders")]


@pytest.mark.parametrize("failure_str", ["all_orders", "positions", "account_summary"])
def test_incomplete_requests_never_return_an_empty_complete_snapshot(socket_obj, failure_str):
    socket_obj.state_obj.failure_str = failure_str
    with pytest.raises(TimeoutError):
        daily_broker.get_daily_execution_snapshot(socket_obj, "DU1")


@pytest.mark.parametrize("mutation_str", ["missing_cash", "currency", "disconnected", "broker_error", "missing_account", "nonfinite"])
def test_untrusted_account_or_order_response_fails_closed(socket_obj, mutation_str):
    if mutation_str == "missing_cash":
        socket_obj.state_obj.account_value_list.pop(0)
    elif mutation_str == "currency":
        socket_obj.state_obj.account_value_list[0].currency = "EUR"
    elif mutation_str == "disconnected":
        socket_obj.state_obj.disconnect_bool = True
    elif mutation_str == "broker_error":
        socket_obj.state_obj.error_code_int = 1100
    elif mutation_str == "missing_account":
        socket_obj.state_obj.trade_list = [_trade(account_str="")]
    else:
        socket_obj.state_obj.position_list[0].position = float("nan")
    with pytest.raises((ValueError, ConnectionError)):
        daily_broker.get_daily_execution_snapshot(socket_obj, "DU1")


def test_orders_changing_during_account_refresh_require_retry(socket_obj):
    def change_fn(broker_obj):
        if len(socket_obj.state_obj.call_list) == 4:
            socket_obj.state_obj.trade_list = [_trade()]
    socket_obj.state_obj.order_hook_fn = change_fn
    with pytest.raises(RuntimeError, match="changed"):
        daily_broker.get_daily_execution_snapshot(socket_obj, "DU1")


def test_after_close_cancel_reconnects_to_owners_and_preserves_foreign_orders(socket_obj):
    socket_obj.state_obj.trade_list = [_trade(), _trade("owned_other_client", 77, perm_int=81),
        _trade("manual", 77, order_int=9, perm_int=82), _trade("owned", 88, "DU2")]
    result_obj = daily_broker.cancel_daily_owned_orders(socket_obj, "DU1", {"owned", "owned_other_client"},
        session_close_timestamp_ts=datetime.now(UTC) - timedelta(minutes=1))
    assert socket_obj.state_obj.cancel_list == [(31, "DU1", "owned", 8), (77, "DU1", "owned_other_client", 8)]
    assert socket_obj.state_obj.connection_list == [31, 31, 77, 31]
    assert [row_dict["order_ref_str"] for row_dict in result_obj.open_order_row_list] == ["manual"]
    assert any(trade_obj.order.account == "DU2" for trade_obj in socket_obj.state_obj.trade_list)


@pytest.mark.parametrize("case_str", ["before_close", "client_zero", "client_busy", "identity_changed", "unconfirmed"])
def test_cancellation_errors_never_claim_confirmation(socket_obj, case_str):
    socket_obj.state_obj.trade_list = [_trade(client_int=77)]
    close_ts = datetime.now(UTC) - timedelta(minutes=1)
    if case_str == "before_close":
        close_ts += timedelta(minutes=2)
    elif case_str == "client_zero":
        socket_obj.state_obj.trade_list[0].order.clientId = 0
    elif case_str == "client_busy":
        def hook_fn(client_int):
            if client_int == 77:
                raise ConnectionError("client ID already connected")
        socket_obj.state_obj.connect_hook_fn = hook_fn
    elif case_str == "identity_changed":
        def hook_fn(client_int):
            if client_int == 77:
                socket_obj.state_obj.trade_list[0].order.orderId = 999
        socket_obj.state_obj.connect_hook_fn = hook_fn
    else:
        socket_obj.state_obj.confirm_bool = False
    with pytest.raises((ValueError, ConnectionError, TimeoutError)):
        daily_broker.cancel_daily_owned_orders(socket_obj, "DU1", {"owned"}, session_close_timestamp_ts=close_ts)
    assert len(socket_obj.state_obj.trade_list) == 1
    if case_str != "unconfirmed":
        assert not socket_obj.state_obj.cancel_list


def test_daily_connection_skips_history_and_does_not_retry_body_errors(monkeypatch):
    instance_list = []
    class ConnectionIB:
        def __init__(self):
            self.connected_bool = False
            instance_list.append(self)
        def connect(self, *arg_tuple, **kwarg_dict):
            self.kwarg_dict = kwarg_dict
            self.connected_bool = True
        def isConnected(self):
            return self.connected_bool
        def disconnect(self):
            self.connected_bool = False
    monkeypatch.setattr("alpha.live.ibkr_socket_client.IB", ConnectionIB)
    client_obj = IBKRSocketClient()
    with pytest.raises(TimeoutError, match="cancel failed"):
        with client_obj.daily_connection(client_id_int=77):
            raise TimeoutError("cancel failed")
    assert len(instance_list) == 1
    assert instance_list[0].kwarg_dict["clientId"] == 77
    assert instance_list[0].kwarg_dict["raiseSyncErrors"] is True
    assert instance_list[0].kwarg_dict["readonly"] is True
    assert instance_list[0].kwarg_dict["fetchFields"].value == 0
    assert not instance_list[0].connected_bool
    with pytest.raises(ValueError, match="client zero"):
        with client_obj.daily_connection(client_id_int=0):
            pytest.fail("Client zero must never connect")


@pytest.mark.parametrize("conflict_str", ["account", "perm_id"])
def test_conflicting_client_order_namespace_prevents_cancellation(socket_obj, conflict_str):
    conflicting_obj = _trade(account_str="DU2" if conflict_str == "account" else "DU1", perm_int=99)
    socket_obj.state_obj.trade_list = [_trade(), conflicting_obj]
    with pytest.raises(ValueError, match="ambiguous client/order"):
        daily_broker.cancel_daily_owned_orders(socket_obj, "DU1", {"owned"},
            session_close_timestamp_ts=datetime.now(UTC) - timedelta(minutes=1))
    assert not socket_obj.state_obj.cancel_list



def test_unbound_manual_orders_remain_visible_by_permanent_identity(socket_obj):
    socket_obj.state_obj.trade_list = [_trade("", 0, order_int=0, perm_int=80),
        _trade("", 0, order_int=0, perm_int=81)]
    snapshot_obj = daily_broker.get_daily_execution_snapshot(socket_obj, "DU1")
    assert {row_dict["perm_id_int"] for row_dict in snapshot_obj.open_order_row_list} == {80, 81}
    assert len(snapshot_obj.open_order_row_list) == 2


def test_cancel_waits_through_a_changing_pending_cancel_snapshot(socket_obj):
    socket_obj.state_obj.trade_list = [_trade()]
    socket_obj.state_obj.confirm_bool = False
    socket_obj.timeout_seconds_float = 1.0
    def change_fn(broker_obj):
        call_count_int = sum(name_str == "all_orders" for _, name_str in socket_obj.state_obj.call_list)
        if socket_obj.state_obj.cancel_list and call_count_int == 6:
            socket_obj.state_obj.trade_list = []
    socket_obj.state_obj.order_hook_fn = change_fn
    result_obj = daily_broker.cancel_daily_owned_orders(socket_obj, "DU1", {"owned"},
        session_close_timestamp_ts=datetime.now(UTC) - timedelta(minutes=1))
    assert result_obj.open_order_row_list == []
    assert len(socket_obj.state_obj.cancel_list) == 1



def test_summary_subscription_is_cancelled_after_success_and_request_failure(socket_obj):
    daily_broker.get_daily_execution_snapshot(socket_obj, "DU1")
    assert len(socket_obj.state_obj.summary_cancel_list) == 1
    socket_obj.state_obj.failure_str = "account_summary"
    with pytest.raises(TimeoutError):
        daily_broker.get_daily_execution_snapshot(socket_obj, "DU1")
    assert len(socket_obj.state_obj.summary_cancel_list) == 2
