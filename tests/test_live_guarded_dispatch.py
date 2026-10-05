"""Deadline batches retain safe retry and partial-send evidence without a broker."""
from dataclasses import replace
from datetime import UTC, datetime

import pytest

from alpha.live.guarded_dispatch import DispatchFailure
from alpha.live.ibkr_socket_client import IBKRSocketClient
from test_live_ibkr_socket_client import (
    _ExpirySubmitIB, _expiry_client_obj, _funding_request_obj, _FakeContract, _freeze_tick_clock,
)


BEFORE_TS = datetime(2026, 10, 5, 13, 27, tzinfo=UTC)
DEADLINE_TS = datetime(2026, 10, 5, 13, 28, tzinfo=UTC)


def request_list():
    return [replace(_funding_request_obj(asset_str=asset_str),
        submission_deadline_timestamp_str=DEADLINE_TS.isoformat()) for asset_str in ("AAPL", "BIL", "SPY")]


def test_presend_timeout_retains_typed_transient_failure_with_no_attempted_legs(monkeypatch):
    broker_obj = _ExpirySubmitIB([BEFORE_TS])
    client_obj = _expiry_client_obj(monkeypatch, broker_obj)
    def fail_qualification(*contract_list):
        raise TimeoutError("contract request timed out")
    monkeypatch.setattr(broker_obj, "qualifyContracts", fail_qualification)
    with pytest.raises(DispatchFailure) as error_info:
        client_obj.submit_order_request_list("SIM_pod", request_list(), BEFORE_TS)
    error_obj = error_info.value
    assert error_obj.error_type_str == "TimeoutError" and error_obj.transient_bool
    assert isinstance(error_obj.__cause__, TimeoutError)
    assert error_obj.attempted_key_list == []
    assert error_obj.never_dispatched_request_list == request_list()
    assert error_obj.partial_result_obj.broker_order_record_list == []
    assert broker_obj.placed_order_list == []


def test_partial_place_order_exception_keeps_attempted_and_never_sent_legs_distinct(monkeypatch):
    broker_obj = _ExpirySubmitIB([BEFORE_TS])
    client_obj = _expiry_client_obj(monkeypatch, broker_obj)
    place_fn = broker_obj.placeOrder
    def fail_second(contract_obj, order_obj):
        if broker_obj.placed_order_list:
            raise TimeoutError("second order may have reached broker")
        return place_fn(contract_obj, order_obj)
    monkeypatch.setattr(broker_obj, "placeOrder", fail_second)
    with pytest.raises(DispatchFailure) as error_info:
        client_obj.submit_order_request_list("SIM_pod", request_list(), BEFORE_TS)
    error_obj = error_info.value
    assert error_obj.attempted_key_list == [request_obj.order_request_key_str for request_obj in request_list()[:2]]
    assert error_obj.never_dispatched_request_list == request_list()[2:]
    assert len(error_obj.partial_result_obj.broker_order_record_list) == 1
    assert error_obj.partial_result_obj.submit_ack_status_str == "missing_critical"
    assert len(broker_obj.placed_order_list) == 1


def test_deadline_after_contract_qualification_prevents_first_dispatch(monkeypatch):
    broker_obj = _ExpirySubmitIB([BEFORE_TS])
    broker_obj.after_qualification_ts = DEADLINE_TS
    client_obj = _expiry_client_obj(monkeypatch, broker_obj)
    with pytest.raises(DispatchFailure) as error_info:
        client_obj.submit_order_request_list("SIM_pod", request_list(), BEFORE_TS)
    assert not error_info.value.transient_bool
    assert error_info.value.attempted_key_list == []
    assert error_info.value.never_dispatched_request_list == request_list()
    assert broker_obj.placed_order_list == []


def test_deadline_between_legs_retains_first_order_and_stops_remainder(monkeypatch):
    broker_obj = _ExpirySubmitIB([BEFORE_TS])
    client_obj = _expiry_client_obj(monkeypatch, broker_obj)
    place_fn = broker_obj.placeOrder
    def place_at_cutoff(contract_obj, order_obj):
        trade_obj = place_fn(contract_obj, order_obj)
        broker_obj.clock_list[:] = [DEADLINE_TS]
        return trade_obj
    monkeypatch.setattr(broker_obj, "placeOrder", place_at_cutoff)
    with pytest.raises(DispatchFailure) as error_info:
        client_obj.submit_order_request_list("SIM_pod", request_list(), BEFORE_TS)
    assert error_info.value.attempted_key_list == [request_list()[0].order_request_key_str]
    assert error_info.value.never_dispatched_request_list == request_list()[1:]
    assert len(error_info.value.partial_result_obj.broker_order_record_list) == 1
    assert len(broker_obj.placed_order_list) == 1
    assert broker_obj.placed_order_list[0].tif == "OPG"


def test_ack_poll_timeout_never_labels_dispatched_orders_as_never_sent(monkeypatch):
    broker_obj = _ExpirySubmitIB([BEFORE_TS])
    client_obj = _expiry_client_obj(monkeypatch, broker_obj)
    def fail_ack(*argument_tuple, **argument_dict):
        raise TimeoutError("ack poll timed out")
    monkeypatch.setattr(client_obj, "_get_recent_order_state_snapshot_from_connection", fail_ack)
    with pytest.raises(DispatchFailure) as error_info:
        client_obj.submit_order_request_list("SIM_pod", request_list(), BEFORE_TS)
    assert len(error_info.value.attempted_key_list) == 3
    assert error_info.value.never_dispatched_request_list == []
    assert len(error_info.value.partial_result_obj.broker_order_record_list) == 3
    assert len(broker_obj.placed_order_list) == 3


@pytest.mark.parametrize("connect_timeout_bool", [False, True])
def test_connection_fallback_never_reconnects_after_uncertain_send(monkeypatch, connect_timeout_bool):
    clock_list = [BEFORE_TS]
    broker_list = []
    connect_host_list = []
    class GuardedIB(_ExpirySubmitIB):
        def __init__(self):
            super().__init__(clock_list)
            self.connected_bool = False
            broker_list.append(self)
        def connect(self, host_str, *argument_tuple, **argument_dict):
            connect_host_list.append(host_str)
            if connect_timeout_bool and len(connect_host_list) == 1:
                raise TimeoutError("initial connection timeout before any order")
            self.connected_bool = True
        def isConnected(self):
            return self.connected_bool
        def disconnect(self):
            self.connected_bool = False
        def placeOrder(self, contract_obj, order_obj):
            if self.placed_order_list:
                raise TimeoutError("ambiguous send must not trigger host fallback")
            return super().placeOrder(contract_obj, order_obj)
    monkeypatch.setattr("alpha.live.ibkr_socket_client.IB", GuardedIB)
    monkeypatch.setattr("alpha.live.ibkr_socket_client.Stock",
        lambda symbol_str, exchange_str, currency_str: _FakeContract(symbol_str))
    _freeze_tick_clock(monkeypatch, clock_list)
    with pytest.raises(DispatchFailure) as error_info:
        IBKRSocketClient().submit_order_request_list("SIM_pod", request_list(), BEFORE_TS)
    assert connect_host_list == (["127.0.0.1", "localhost"] if connect_timeout_bool else ["127.0.0.1"])
    assert sum(len(broker_obj.placed_order_list) for broker_obj in broker_list) == 1
    assert len(error_info.value.attempted_key_list) == 2
    assert error_info.value.never_dispatched_request_list == request_list()[2:]
