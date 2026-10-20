from __future__ import annotations

from contextlib import contextmanager
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from alpha.live import core5_broker as broker_module
from alpha.live.ibkr_socket_client import IBKRSocketClient


ROUTE_STR = "U_CORE5_TEST"


def _account_value_obj(tag_str, value_obj, currency_str="USD", account_str=ROUTE_STR, model_str=""):
    return SimpleNamespace(tag=tag_str, value=value_obj, currency=currency_str, account=account_str, modelCode=model_str)


def _position_obj(asset_str="DBC", shares_float=-5.0, account_str=ROUTE_STR, **contract_dict):
    return SimpleNamespace(account=account_str, position=shares_float,
        contract=SimpleNamespace(**{"symbol": asset_str, "secType": "STK", "currency": "USD", "multiplier": "",
                                   **contract_dict}))


class _FakeBroker:
    def __init__(self):
        self.account_list = [ROUTE_STR]
        self.wrapper = SimpleNamespace(_futures={})
        self.summary_list = [_account_value_obj("TotalCashValue", "12000"), _account_value_obj("NetLiquidation", "100000")]
        self.position_list = []
        self.open_trade_list = []
        self.open_read_count_int = 0

    def managedAccounts(self):
        return self.account_list

    def accountSummary(self, account):
        assert account == ROUTE_STR
        return self.summary_list

    def positions(self, account):
        assert account == ROUTE_STR
        return self.position_list

    def reqAllOpenOrders(self):
        self.open_read_count_int += 1
        return self.open_trade_list

    def reqOpenOrders(self):
        raise AssertionError("CORE5 must inspect every client, not only current-client open orders")

    def placeOrder(self, *argument_tuple, **keyword_dict):
        raise AssertionError("CORE5 account reads must never submit orders")

    def cancelOrder(self, *argument_tuple, **keyword_dict):
        raise AssertionError("CORE5 account reads must never cancel orders")


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    monkeypatch.setattr(IBKRSocketClient, "connect", lambda *argument_tuple: pytest.fail("Real broker forbidden"))
    monkeypatch.setattr(IBKRSocketClient, "daily_connection", lambda *argument_tuple: pytest.fail("Real broker forbidden"))


@pytest.fixture
def broker_case_tuple(monkeypatch):
    broker_obj = _FakeBroker()
    socket_obj = IBKRSocketClient()
    socket_obj.test_connect_count_int = 0

    @contextmanager
    def connect_fake():
        socket_obj.test_connect_count_int += 1
        yield broker_obj

    monkeypatch.setattr(socket_obj, "connect", connect_fake)
    monkeypatch.setattr(socket_obj, "daily_connection", connect_fake)
    return socket_obj, broker_obj


def test_snapshot_requires_one_connection_and_keeps_signed_cash_and_other_client_orders(broker_case_tuple):
    socket_obj, broker_obj = broker_case_tuple
    broker_obj.summary_list[0].value = "-12000"
    broker_obj.summary_list.extend([
        _account_value_obj("TotalCashValue", "999999", "EUR"),
        _account_value_obj("TotalCashValue", "999999", account_str="OTHER"),
        _account_value_obj("TotalCashValue", "999999", model_str="MODEL"),
    ])
    broker_obj.position_list = [_position_obj(), _position_obj("SPY", 10),
                               _position_obj("UNSUPPORTED", 20, account_str="OTHER")]
    broker_obj.open_trade_list = [SimpleNamespace(order=SimpleNamespace(account=ROUTE_STR, orderId=91, clientId=99)),
                                  SimpleNamespace(order=SimpleNamespace(account="OTHER", orderId=92))]
    before_ts = datetime.now(tz=UTC)
    snapshot_obj = broker_module.get_core5_account_snapshot(socket_obj, ROUTE_STR)
    assert socket_obj.test_connect_count_int == 1
    assert broker_obj.open_read_count_int == 1 and broker_obj.RequestTimeout == 4
    assert snapshot_obj.cash_float == -12000 and snapshot_obj.net_liq_float == snapshot_obj.total_value_float == 100000
    assert snapshot_obj.position_amount_map == {"DBC": -5, "SPY": 10}
    assert snapshot_obj.open_order_id_list == ["91"]
    assert snapshot_obj.available_funds_float is None
    assert before_ts <= snapshot_obj.snapshot_timestamp_ts <= datetime.now(tz=UTC)


def test_snapshot_optional_valuation_uses_same_connection_and_original_observation(broker_case_tuple, monkeypatch):
    socket_obj, broker_obj = broker_case_tuple
    valuation_dict = {"available_bool": True}
    monkeypatch.setattr(socket_obj, "_capture_portfolio_valuation_dict",
        lambda current_broker_obj, route_str, position_dict: valuation_dict
        if current_broker_obj is broker_obj and route_str == ROUTE_STR and position_dict == {} else pytest.fail("Wrong capture"))
    snapshot_obj = broker_module.get_core5_account_snapshot(socket_obj, ROUTE_STR, True)
    assert snapshot_obj.portfolio_valuation_dict is valuation_dict
    assert socket_obj.test_connect_count_int == 1


@pytest.mark.parametrize("include_bool", [False, True])
def test_only_core5_eod_valuation_selects_lean_connection(broker_case_tuple, monkeypatch, include_bool):
    socket_obj, _ = broker_case_tuple
    forbidden_str = "connect" if include_bool else "daily_connection"
    monkeypatch.setattr(socket_obj, forbidden_str, lambda: pytest.fail("Wrong connection profile"))
    monkeypatch.setattr(socket_obj, "_capture_portfolio_valuation_dict",
        lambda *argument_tuple: {"available_bool": True})
    snapshot_obj = broker_module.get_core5_account_snapshot(socket_obj, ROUTE_STR, include_bool)
    assert snapshot_obj.cash_float == 12000
    assert (snapshot_obj.portfolio_valuation_dict is not None) is include_bool


@pytest.mark.parametrize("tag_str,value_obj", [
    ("TotalCashValue", None), ("TotalCashValue", ""), ("TotalCashValue", "nan"), ("TotalCashValue", "inf"),
    ("TotalCashValue", "1.7976931348623157e308"), ("TotalCashValue", True),
    ("NetLiquidation", "0"), ("NetLiquidation", "-1"), ("NetLiquidation", "nan"),
])
def test_snapshot_rejects_missing_invalid_or_sentinel_totals(broker_case_tuple, tag_str, value_obj):
    socket_obj, broker_obj = broker_case_tuple
    next(value_row_obj for value_row_obj in broker_obj.summary_list if value_row_obj.tag == tag_str).value = value_obj
    with pytest.raises(ValueError):
        broker_module.get_core5_account_snapshot(socket_obj, ROUTE_STR)


@pytest.mark.parametrize("mutation_str", ["missing_cash", "ambiguous_cash", "wrong_currency", "incomplete_positions", "unsupported_sdk", "invisible"])
def test_snapshot_fails_closed_on_incomplete_account_evidence(broker_case_tuple, mutation_str):
    socket_obj, broker_obj = broker_case_tuple
    if mutation_str == "missing_cash":
        broker_obj.summary_list.pop(0)
    elif mutation_str == "ambiguous_cash":
        broker_obj.summary_list.append(_account_value_obj("TotalCashValue", "12000"))
    elif mutation_str == "wrong_currency":
        broker_obj.summary_list[0].currency = "BASE"
    elif mutation_str == "incomplete_positions":
        broker_obj.wrapper._futures["positions"] = object()
    elif mutation_str == "unsupported_sdk":
        del broker_obj.wrapper
    else:
        broker_obj.account_list = ["OTHER"]
    with pytest.raises(ValueError):
        broker_module.get_core5_account_snapshot(socket_obj, ROUTE_STR)


@pytest.mark.parametrize("position_obj", [
    _position_obj(shares_float=-.5), _position_obj(shares_float=float("nan")),
    _position_obj("SPY", -1), _position_obj("QQQ", 1),
    _position_obj(currency="EUR"), _position_obj(secType="OPT"), _position_obj(multiplier="100"),
])
def test_snapshot_rejects_unsupported_or_fractional_positions(broker_case_tuple, position_obj):
    socket_obj, broker_obj = broker_case_tuple
    broker_obj.position_list = [position_obj]
    with pytest.raises(ValueError):
        broker_module.get_core5_account_snapshot(socket_obj, ROUTE_STR)


def test_snapshot_rejects_duplicate_symbol_contracts(broker_case_tuple):
    socket_obj, broker_obj = broker_case_tuple
    broker_obj.position_list = [_position_obj(), _position_obj()]
    with pytest.raises(ValueError, match="unique"):
        broker_module.get_core5_account_snapshot(socket_obj, ROUTE_STR)
