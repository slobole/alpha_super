from __future__ import annotations

from contextlib import nullcontext
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from alpha.live.ibkr_socket_client import IBKRSocketClient, IBKR_TICK_OPEN_SOURCE_STR


class _FakeIB:
    attempted_host_str_list: list[str] = []

    def __init__(self):
        self.connected_bool = False
        self.connected_host_str: str | None = None

    def connect(self, host_str, port_int, clientId, timeout):
        del port_int, clientId, timeout
        self.__class__.attempted_host_str_list.append(str(host_str))
        if str(host_str) == "127.0.0.1":
            raise ConnectionRefusedError(1225, "refused")
        self.connected_bool = True
        self.connected_host_str = str(host_str)

    def isConnected(self) -> bool:
        return bool(self.connected_bool)

    def disconnect(self) -> None:
        self.connected_bool = False


def test_ibkr_socket_client_retries_alternate_loopback_hosts_on_connection_refused(monkeypatch):
    _FakeIB.attempted_host_str_list = []
    monkeypatch.setattr("alpha.live.ibkr_socket_client.IB", _FakeIB)

    socket_client_obj = IBKRSocketClient(
        host_str="127.0.0.1",
        port_int=7496,
        client_id_int=31,
        timeout_seconds_float=4.0,
    )

    with socket_client_obj.connect() as ib_obj:
        assert ib_obj.connected_host_str == "localhost"

    assert _FakeIB.attempted_host_str_list == ["127.0.0.1", "localhost"]


class _FakeIBTimeoutOnLocalhost:
    attempted_host_str_list: list[str] = []

    def __init__(self):
        self.connected_bool = False
        self.connected_host_str: str | None = None

    def connect(self, host_str, port_int, clientId, timeout):
        del port_int, clientId, timeout
        self.__class__.attempted_host_str_list.append(str(host_str))
        if str(host_str) == "localhost":
            raise TimeoutError()
        self.connected_bool = True
        self.connected_host_str = str(host_str)

    def isConnected(self) -> bool:
        return bool(self.connected_bool)

    def disconnect(self) -> None:
        self.connected_bool = False


def test_ibkr_socket_client_retries_alternate_loopback_hosts_on_timeout(monkeypatch):
    _FakeIBTimeoutOnLocalhost.attempted_host_str_list = []
    monkeypatch.setattr("alpha.live.ibkr_socket_client.IB", _FakeIBTimeoutOnLocalhost)

    socket_client_obj = IBKRSocketClient(
        host_str="localhost",
        port_int=7496,
        client_id_int=31,
        timeout_seconds_float=4.0,
    )

    with socket_client_obj.connect() as ib_obj:
        assert ib_obj.connected_host_str == "127.0.0.1"

    assert _FakeIBTimeoutOnLocalhost.attempted_host_str_list == ["localhost", "127.0.0.1"]


class _FakeIBUnqualifiedContract:
    def qualifyContracts(self, *contract_list):
        return [contract_obj if contract_obj.symbol != "CTRA" else None for contract_obj in contract_list]


class _FakeContract:
    def __init__(self, symbol: str):
        self.symbol = str(symbol)


def test_ibkr_socket_client_contract_map_reports_unqualified_symbols(monkeypatch):
    monkeypatch.setattr(
        "alpha.live.ibkr_socket_client.Stock",
        lambda symbol_str, exchange_str, currency_str: _FakeContract(symbol_str),
    )

    with pytest.raises(ValueError, match="IBKR contract qualification failed for assets: CTRA"):
        IBKRSocketClient._build_stock_contract_map(
            _FakeIBUnqualifiedContract(),
            ["AAPL", "CTRA"],
        )


class _FakeTicker:
    def __init__(self, symbol: str, open_price_float):
        self.contract = _FakeContract(symbol)
        self.open = open_price_float


class _FakeIBTickOpen:
    connected_client_id_int: int | None = None
    historical_call_count_int: int = 0
    open_price_map_dict = {
        "AAPL": 123.45,
        "MSFT": None,
        "NEG": -1.0,
    }

    def __init__(self):
        self.connected_bool = False

    def connect(self, host_str, port_int, clientId, timeout):
        del host_str, port_int, timeout
        self.__class__.connected_client_id_int = int(clientId)
        self.connected_bool = True

    def isConnected(self) -> bool:
        return bool(self.connected_bool)

    def disconnect(self) -> None:
        self.connected_bool = False

    def qualifyContracts(self, *contract_list):
        return list(contract_list)

    def reqTickers(self, *contract_list):
        return [
            _FakeTicker(
                contract_obj.symbol,
                self.open_price_map_dict.get(contract_obj.symbol),
            )
            for contract_obj in contract_list
        ]

    def reqHistoricalData(self, *args, **kwargs):
        del args, kwargs
        self.__class__.historical_call_count_int += 1
        raise AssertionError("tick-open provider must not use historical bars")


def _freeze_tick_clock(monkeypatch, timestamp_list):
    class ClockDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            timestamp_ts = timestamp_list.pop(0) if len(timestamp_list) > 1 else timestamp_list[0]
            return timestamp_ts.astimezone(tz or UTC)
    monkeypatch.setattr("alpha.live.ibkr_socket_client.datetime", ClockDateTime)


def test_ibkr_socket_client_tick_open_reads_only_ticker_open(monkeypatch):
    _freeze_tick_clock(monkeypatch, [datetime(2024, 1, 3, 14, 35, tzinfo=UTC)])
    _FakeIBTickOpen.connected_client_id_int = None
    _FakeIBTickOpen.historical_call_count_int = 0
    monkeypatch.setattr("alpha.live.ibkr_socket_client.IB", _FakeIBTickOpen)
    monkeypatch.setattr(
        "alpha.live.ibkr_socket_client.Stock",
        lambda symbol_str, exchange_str, currency_str: _FakeContract(symbol_str),
    )

    socket_client_obj = IBKRSocketClient(
        host_str="127.0.0.1",
        port_int=7497,
        client_id_int=91,
        timeout_seconds_float=4.0,
    )
    session_open_price_list = socket_client_obj.get_tick_open_price_list(
        account_route_str="SIM_pod",
        asset_str_list=["MSFT", "AAPL", "NEG"],
        session_open_timestamp_ts=datetime(2024, 1, 3, 14, 30, tzinfo=UTC),
        session_calendar_id_str="XNYS",
    )

    session_open_price_by_asset_map_dict = {
        session_open_price_obj.asset_str: session_open_price_obj
        for session_open_price_obj in session_open_price_list
    }
    assert _FakeIBTickOpen.connected_client_id_int == 91
    assert _FakeIBTickOpen.historical_call_count_int == 0
    assert session_open_price_by_asset_map_dict["AAPL"].official_open_price_float == 123.45
    assert session_open_price_by_asset_map_dict["AAPL"].open_price_source_str == IBKR_TICK_OPEN_SOURCE_STR
    assert session_open_price_by_asset_map_dict["MSFT"].official_open_price_float is None
    assert session_open_price_by_asset_map_dict["MSFT"].open_price_source_str is None
    assert session_open_price_by_asset_map_dict["NEG"].official_open_price_float is None
    assert session_open_price_by_asset_map_dict["NEG"].open_price_source_str is None


@pytest.mark.parametrize("observed_ts", [
    datetime(2024, 1, 3, 14, 29, tzinfo=UTC),
    datetime(2024, 1, 4, 15, 0, tzinfo=UTC),
    datetime(2024, 1, 2, 15, 0, tzinfo=UTC),
])
def test_uncached_tick_open_rejects_wrong_day_or_preopen_before_connect(monkeypatch, observed_ts):
    _freeze_tick_clock(monkeypatch, [observed_ts])
    def unexpected_connect(*args, **kwargs):
        raise AssertionError("Invalid historical/pre-open request must not connect")
    monkeypatch.setattr(IBKRSocketClient, "connect", unexpected_connect)
    with pytest.raises(RuntimeError, match="Uncached IBKR tick-open"):
        IBKRSocketClient().get_tick_open_price_list(
            "SIM_pod", ["AAPL"], datetime(2024, 1, 3, 14, 30, tzinfo=UTC), "XNYS")


def test_tick_open_rejects_date_rollover_during_fetch(monkeypatch):
    _freeze_tick_clock(monkeypatch, [datetime(2024, 1, 4, 4, 59, tzinfo=UTC),
                                    datetime(2024, 1, 4, 5, 0, tzinfo=UTC)])
    monkeypatch.setattr("alpha.live.ibkr_socket_client.IB", _FakeIBTickOpen)
    monkeypatch.setattr("alpha.live.ibkr_socket_client.Stock",
                        lambda symbol_str, exchange_str, currency_str: _FakeContract(symbol_str))
    with pytest.raises(RuntimeError, match="Uncached IBKR tick-open"):
        IBKRSocketClient().get_tick_open_price_list(
            "SIM_pod", ["AAPL"], datetime(2024, 1, 3, 14, 30, tzinfo=UTC), "XNYS")


@pytest.mark.parametrize("target_ts", [datetime(2024, 1, 3, 9, 30, tzinfo=UTC),
                                       datetime(2024, 1, 6, 14, 30, tzinfo=UTC)])
def test_tick_open_requires_canonical_trading_session_open(monkeypatch, target_ts):
    def unexpected_connect(*args, **kwargs):
        raise AssertionError("Invalid session request must not connect")
    monkeypatch.setattr(IBKRSocketClient, "connect", unexpected_connect)
    with pytest.raises(ValueError, match="session"):
        IBKRSocketClient().get_tick_open_price_list("SIM_pod", ["AAPL"], target_ts, "XNYS")


def _snapshot_trade_obj(order_id_int, *, status_str="Cancelled", account_str="SIM_pod", order_ref_str="batch:stock:1", log_bool=True):
    return SimpleNamespace(
        order=SimpleNamespace(account=account_str, orderId=order_id_int, permId=1000 + order_id_int,
                              orderRef=order_ref_str, orderType="MKT", tif="OPG", totalQuantity=10, action="BUY"),
        orderStatus=SimpleNamespace(orderId=order_id_int, permId=1000 + order_id_int,
                                    status=status_str, filled=4, remaining=6, avgFillPrice=100.0),
        contract=SimpleNamespace(symbol="AAPL"),
        log=[SimpleNamespace(time=datetime(2024, 1, 3, 15, 0, tzinfo=UTC), status=status_str,
                             message="broker observation", errorCode=0)] if log_bool else [],
    )


def _snapshot_client_obj(monkeypatch, open_trade_list, completed_trade_list):
    class SnapshotIB:
        def reqOpenOrders(self):
            return open_trade_list

        def reqCompletedOrders(self, apiOnly):
            assert apiOnly is False
            return completed_trade_list

        def reqExecutions(self, execution_filter_obj):
            assert execution_filter_obj.acctCode == "SIM_pod"
            return []

    socket_client_obj = IBKRSocketClient()
    monkeypatch.setattr(socket_client_obj, "connect", lambda: nullcontext(SnapshotIB()))

    def unexpected_shadowed_helper(*args, **kwargs):
        raise AssertionError("This regression must exercise the effective public refresh implementation")

    monkeypatch.setattr(socket_client_obj, "_get_recent_order_state_snapshot_from_connection", unexpected_shadowed_helper)
    return socket_client_obj


@pytest.mark.parametrize("source_str", ["open_order", "completed_order"])
def test_public_order_refresh_preserves_record_and_event_source(monkeypatch, source_str):
    trade_obj = _snapshot_trade_obj(1)
    socket_client_obj = _snapshot_client_obj(
        monkeypatch, [trade_obj] if source_str == "open_order" else [],
        [trade_obj] if source_str == "completed_order" else [],
    )
    record_list, event_list, fill_list = socket_client_obj.get_recent_order_state_snapshot(
        "SIM_pod", datetime(2024, 1, 3, 14, 0, tzinfo=UTC), submission_key_str="batch",
    )

    assert len(record_list) == len(event_list) == 1
    assert fill_list == []
    record_obj = record_list[0]
    assert record_obj.raw_payload_dict["snapshot_source_str"] == source_str
    assert record_obj.raw_payload_dict["open_order_observed_bool"] is (source_str == "open_order")
    assert record_obj.raw_payload_dict["order_ref_str"] == "batch:stock:1"
    assert record_obj.broker_order_id_str == "1001"
    assert record_obj.broker_order_type_str == "MOO"
    assert record_obj.amount_float == 10.0
    assert record_obj.filled_amount_float == 4.0
    assert event_list[0].raw_payload_dict["snapshot_source_str"] == source_str
    assert event_list[0].raw_payload_dict["error_code_int"] == 0


@pytest.mark.parametrize("log_bool", [False, True])
def test_public_order_refresh_duplicate_open_evidence_survives_completed_precedence(monkeypatch, log_bool):
    open_trade_list = [_snapshot_trade_obj(1, status_str="Submitted", log_bool=log_bool)]
    completed_trade_list = [_snapshot_trade_obj(1, log_bool=log_bool)]
    socket_client_obj = _snapshot_client_obj(monkeypatch, open_trade_list, completed_trade_list)
    record_list, event_list, _ = socket_client_obj.get_recent_order_state_snapshot(
        "SIM_pod", datetime(2024, 1, 3, 14, 0, tzinfo=UTC), submission_key_str="batch",
    )

    assert len(record_list) == 1
    assert record_list[0].status_str == "Cancelled"  # Existing completed-record precedence is preserved.
    assert record_list[0].raw_payload_dict["snapshot_source_str"] == "completed_order"
    assert record_list[0].raw_payload_dict["open_order_observed_bool"] is True
    assert [event_obj.raw_payload_dict["snapshot_source_str"] for event_obj in event_list] == (
        ["open_order", "completed_order"] if log_bool else []
    )

    open_trade_list.clear()
    later_record_list, _, _ = socket_client_obj.get_recent_order_state_snapshot(
        "SIM_pod", datetime(2024, 1, 3, 14, 0, tzinfo=UTC), submission_key_str="batch",
    )
    assert later_record_list[0].raw_payload_dict["open_order_observed_bool"] is False
    assert later_record_list[0].raw_payload_dict["snapshot_source_str"] == "completed_order"


@pytest.mark.parametrize("filter_dict", [
    {"submission_key_str": "batch"},
    {"allowed_broker_order_id_set": {"1"}},
    {"allowed_broker_order_id_set": {"1001"}},
])
def test_public_order_refresh_keeps_account_and_correlation_filters(monkeypatch, filter_dict):
    socket_client_obj = _snapshot_client_obj(monkeypatch, [], [
        _snapshot_trade_obj(1),
        _snapshot_trade_obj(2, account_str="OTHER"),
        _snapshot_trade_obj(3, order_ref_str="other:stock:1"),
    ])
    record_list, event_list, fill_list = socket_client_obj.get_recent_order_state_snapshot(
        "SIM_pod", datetime(2024, 1, 3, 14, 0, tzinfo=UTC), **filter_dict,
    )
    assert [record_obj.broker_order_id_str for record_obj in record_list] == ["1001"]
    assert [event_obj.broker_order_id_str for event_obj in event_list] == ["1001"]
    assert fill_list == []


def test_public_order_refresh_keeps_uncorrelated_event_time_filter(monkeypatch):
    socket_client_obj = _snapshot_client_obj(monkeypatch, [], [_snapshot_trade_obj(1)])
    assert socket_client_obj.get_recent_order_state_snapshot(
        "SIM_pod", datetime(2024, 1, 3, 15, 1, tzinfo=UTC),
    ) == ([], [], [])


@pytest.mark.parametrize("asset_str,broker_symbol_str", [("BRK.B", "BRK B"), ("BF.B", "BF B"), ("AAPL", "AAPL"), ("ABC.X", "ABC.X")])
def test_share_class_aliases_roundtrip_qualification_prices_records_and_fills(monkeypatch, asset_str, broker_symbol_str):
    class AliasIB:
        def qualifyContracts(self, *contract_list):
            assert [contract_obj.symbol for contract_obj in contract_list] == [broker_symbol_str]
            return list(contract_list)

        def reqTickers(self, *contract_list):
            return [SimpleNamespace(contract=contract_obj, marketPrice=lambda: 100.0)
                    for contract_obj in contract_list]

        def reqExecutions(self, execution_filter_obj):
            assert execution_filter_obj.acctCode == "SIM_pod"
            return [SimpleNamespace(contract=_FakeContract(broker_symbol_str),
                time=datetime(2024, 1, 3, 15, 0, tzinfo=UTC),
                execution=SimpleNamespace(permId=1001, orderId=1, side="SLD", shares=3, price=100, execId="e1"))]

        def accountSummary(self, account):
            return [SimpleNamespace(account=account, tag="NetLiquidation", value="10000"),
                    SimpleNamespace(account=account, tag="TotalCashValue", value="1000")]

        def positions(self, account):
            return [SimpleNamespace(contract=_FakeContract(broker_symbol_str), position=7)]

        def reqOpenOrders(self):
            return []

    monkeypatch.setattr("alpha.live.ibkr_socket_client.Stock",
                        lambda symbol_str, exchange_str, currency_str: _FakeContract(symbol_str))
    ib_obj = AliasIB()
    client_obj = IBKRSocketClient()
    monkeypatch.setattr(client_obj, "connect", lambda: nullcontext(ib_obj))
    contract_map_dict = client_obj._build_stock_contract_map(ib_obj, [asset_str])
    assert list(contract_map_dict) == [asset_str]
    quote_obj = client_obj.get_live_price_snapshot("SIM_pod", [asset_str])
    assert quote_obj.asset_reference_price_map == {asset_str: 100.0}
    assert client_obj.get_account_snapshot("SIM_pod").position_amount_map == {asset_str: 7.0}
    trade_obj = _snapshot_trade_obj(1)
    trade_obj.contract.symbol = broker_symbol_str
    record_obj = client_obj._build_broker_order_record_obj(trade_obj, "SIM_pod", datetime(2024, 1, 3, tzinfo=UTC), None, None, asset_str)
    assert record_obj.asset_str == asset_str
    assert client_obj._build_broker_order_event_list(trade_obj, "SIM_pod", None, None, asset_str)[0].asset_str == asset_str
    for fill_list in [client_obj.get_recent_fill_list("SIM_pod", datetime(2024, 1, 3, tzinfo=UTC)),
                      client_obj._get_recent_fill_list_from_connection(ib_obj, "SIM_pod", datetime(2024, 1, 3, tzinfo=UTC))]:
        assert fill_list[0].asset_str == asset_str
        assert fill_list[0].fill_amount_float == -3.0


def test_share_class_aliases_in_official_open_and_portfolio_valuation(monkeypatch):
    _freeze_tick_clock(monkeypatch, [datetime(2024, 1, 3, 14, 35, tzinfo=UTC)])
    monkeypatch.setattr("alpha.live.ibkr_socket_client.IB", _FakeIBTickOpen)
    monkeypatch.setattr("alpha.live.ibkr_socket_client.Stock",
                        lambda symbol_str, exchange_str, currency_str: _FakeContract(symbol_str))
    monkeypatch.setitem(_FakeIBTickOpen.open_price_map_dict, "BRK B", 100.0)
    price_list = IBKRSocketClient().get_tick_open_price_list(
        "SIM_pod", ["BRK.B"], datetime(2024, 1, 3, 14, 30, tzinfo=UTC), "XNYS")
    assert price_list[0].asset_str == "BRK.B"
    assert price_list[0].official_open_price_float == 100.0
    valuation_dict = IBKRSocketClient._build_portfolio_valuation_values_dict(
        [SimpleNamespace(contract=SimpleNamespace(symbol="BRK B", conId=1, secType="STK", currency="USD", multiplier="1"),
                         position=7, marketPrice=100, marketValue=700)],
        [SimpleNamespace(tag="TotalCashValue", currency="USD", value="300", modelCode=""),
         SimpleNamespace(tag="NetLiquidation", currency="USD", value="1000", modelCode="")],
        {"BRK.B": 7.0},
    )
    assert valuation_dict["available_bool"] is True
    assert valuation_dict["position_list"][0]["symbol_str"] == "BRK.B"


def _funding_request_obj(asset_str="AAPL", amount_float=10, price_float=100):
    from alpha.live.models import BrokerOrderRequest
    return BrokerOrderRequest(
        release_id_str="capsule", pod_id_str="capsule", account_route_str="SIM_pod",
        submission_key_str="batch", order_request_key_str=f"batch:{asset_str}:1", asset_str=asset_str,
        broker_order_type_str="MOO", order_class_str="order_target", unit_str="shares",
        amount_float=amount_float, target_bool=False, trade_id_int=None,
        sizing_reference_price_float=price_float, portfolio_value_float=100000,
    )


class _FundingIB:
    def __init__(self, trading_type_str="STKNOPT", buying_power_str="2000", currency_str="USD"):
        self.account_value_list = [
            SimpleNamespace(account="SIM_pod", tag="TradingType-S", value=trading_type_str, currency=""),
            SimpleNamespace(account="SIM_pod", tag="BuyingPower", value=buying_power_str, currency=currency_str),
            # AvailableFunds is not a notional buying-power measure.
            SimpleNamespace(account="SIM_pod", tag="AvailableFunds", value="500", currency="USD"),
            SimpleNamespace(account="OTHER", tag="BuyingPower", value="999999", currency="USD"),
        ]
        self.preview_list = []
        self.downloaded_bool = False
        self.preview_obj = SimpleNamespace(status="PreSubmitted", initMarginBefore="2000", initMarginAfter="2250",
            maintMarginBefore="1500", maintMarginAfter="1750", equityWithLoanBefore="10000", equityWithLoanAfter="9999", warningText="")

    def managedAccounts(self):
        return ["SIM_pod", "OTHER"]

    def reqAccountUpdates(self, account):
        assert account == "SIM_pod"
        self.downloaded_bool = True

    def accountValues(self, account):
        assert self.downloaded_bool
        return self.account_value_list

    def qualifyContracts(self, *contract_list):
        return list(contract_list)

    def whatIfOrder(self, contract_obj, order_obj):
        assert order_obj.whatIf is True
        assert order_obj.action == "BUY"
        assert order_obj.account == "SIM_pod"
        assert order_obj.tif == "OPG"
        self.preview_list.append((contract_obj.symbol, order_obj.totalQuantity))
        return self.preview_obj

    def placeOrder(self, *args):
        raise AssertionError("Funding must never send an executable order")


def _funding_client_obj(monkeypatch, ib_obj):
    client_obj = IBKRSocketClient()
    monkeypatch.setattr(client_obj, "connect", lambda: nullcontext(ib_obj))
    return client_obj


@pytest.mark.parametrize("trading_type_str", ["STKNOPT", "STKMRGN", "PMRGN", "GPMRGN"])
def test_capsule_funding_uses_real_buying_power_and_individual_previews_without_sell_credit(monkeypatch, trading_type_str):
    ib_obj = _FundingIB(trading_type_str=trading_type_str)
    evidence_dict = _funding_client_obj(monkeypatch, ib_obj).get_capsule_funding_evidence(
        "SIM_pod", [_funding_request_obj("BIL", -1000), _funding_request_obj("AAPL"), _funding_request_obj("BRK.B")])
    assert evidence_dict["buying_power_float"] == evidence_dict["total_buy_notional_float"] == 2000
    assert evidence_dict["pending_sell_credit_float"] == 0
    assert evidence_dict["basket_margin_guaranteed_bool"] is False
    assert ib_obj.preview_list == [("AAPL", 10.0), ("BRK B", 10.0)]
    assert ib_obj.RequestTimeout == 4.0


@pytest.mark.parametrize("trading_type_str", ["", "STKCASH", "IRAMRGN", "INDIVIDUAL", "UNRECOGNIZED"])
def test_capsule_funding_rejects_unverified_or_limited_margin_accounts(monkeypatch, trading_type_str):
    ib_obj = _FundingIB(trading_type_str=trading_type_str)
    with pytest.raises(ValueError, match="verified margin account"):
        _funding_client_obj(monkeypatch, ib_obj).get_capsule_funding_evidence("SIM_pod", [_funding_request_obj()])
    assert ib_obj.preview_list == []


@pytest.mark.parametrize("buying_power_str,currency_str", [("nan", "USD"), ("inf", "USD"), ("1.7976931348623157e308", "USD"), ("-1", "USD"), ("bad", "USD"), ("2000", "EUR"), ("2000", "BASE")])
def test_capsule_funding_requires_finite_unambiguous_usd_buying_power(monkeypatch, buying_power_str, currency_str):
    ib_obj = _FundingIB(buying_power_str=buying_power_str, currency_str=currency_str)
    with pytest.raises(ValueError, match="BuyingPower"):
        _funding_client_obj(monkeypatch, ib_obj).get_capsule_funding_evidence("SIM_pod", [_funding_request_obj()])


def test_capsule_funding_checks_combined_buys_without_netting_pending_sales(monkeypatch):
    ib_obj = _FundingIB(buying_power_str="1500")
    with pytest.raises(ValueError, match="require 2000.00 USD before sells"):
        _funding_client_obj(monkeypatch, ib_obj).get_capsule_funding_evidence(
            "SIM_pod", [_funding_request_obj("BIL", -100), _funding_request_obj("AAPL"), _funding_request_obj("MSFT")])
    assert ib_obj.preview_list == []


@pytest.mark.parametrize("field_str,value_str", [("status", "Inactive"), ("initMarginAfter", ""), ("initMarginAfter", "nan"), ("initMarginAfter", "12000"), ("maintMarginAfter", "-1")])
def test_capsule_funding_requires_valid_individual_margin_preview(monkeypatch, field_str, value_str):
    ib_obj = _FundingIB()
    setattr(ib_obj.preview_obj, field_str, value_str)
    with pytest.raises(ValueError, match="preview"):
        _funding_client_obj(monkeypatch, ib_obj).get_capsule_funding_evidence("SIM_pod", [_funding_request_obj()])


def test_capsule_sell_only_funding_does_not_block_risk_reduction(monkeypatch):
    client_obj = IBKRSocketClient()
    monkeypatch.setattr(client_obj, "connect", lambda: pytest.fail("No funding preview for sell-only batch"))
    assert client_obj.get_capsule_funding_evidence("SIM_pod", [_funding_request_obj("BIL", -10)]) == {
        "required_bool": False, "total_buy_notional_float": 0.0}


@pytest.mark.parametrize("pending_download_bool", [False, True])
def test_capsule_funding_single_account_uses_completed_startup_without_duplicate_subscription(monkeypatch, pending_download_bool):
    ib_obj = _FundingIB()
    ib_obj.downloaded_bool = True
    ib_obj.wrapper = SimpleNamespace(_futures={"accountValues": object()} if pending_download_bool else {})
    monkeypatch.setattr(ib_obj, "managedAccounts", lambda: ["SIM_pod"])
    monkeypatch.setattr(ib_obj, "reqAccountUpdates", lambda **kwargs: pytest.fail("Duplicate subscription"))
    client_obj = _funding_client_obj(monkeypatch, ib_obj)
    if pending_download_bool:
        with pytest.raises(ValueError, match="download is incomplete"):
            client_obj.get_capsule_funding_evidence("SIM_pod", [_funding_request_obj()])
    else:
        assert client_obj.get_capsule_funding_evidence("SIM_pod", [_funding_request_obj()])["required_bool"] is True


class _CapsuleEvidenceIB:
    def __init__(self, completed_trade_list, fill_list=(), open_trade_list=()):
        self.completed_trade_list = completed_trade_list
        self.fill_list = list(fill_list)
        self.open_trade_list = list(open_trade_list)
        self.all_open_call_count_int = 0

    def reqAllOpenOrders(self):
        self.all_open_call_count_int += 1
        return self.open_trade_list

    def reqOpenOrders(self):
        return []  # Other-client/manual orders are deliberately absent here.

    def reqCompletedOrders(self, apiOnly):
        assert apiOnly is False
        return self.completed_trade_list

    def reqExecutions(self, execution_filter_obj):
        assert execution_filter_obj.acctCode == "SIM_pod"
        return self.fill_list

    def accountSummary(self, account):
        return [SimpleNamespace(account=account, tag="NetLiquidation", value="10000"),
                SimpleNamespace(account=account, tag="TotalCashValue", value="1000")]

    def positions(self, account):
        return [SimpleNamespace(contract=_FakeContract("BRK B"), position=6)]


def _completed_trade_obj(order_id_int=1, filled_quantity_float=4, order_ref_str="batch:stock:1"):
    # Match ib_async.wrapper.completedOrder: default-zero status quantities and
    # empty log; decoded execution quantity is on Order.filledQuantity instead.
    from ib_async import Order, OrderStatus, Stock, Trade
    return Trade(Stock("BRK B", "SMART", "USD"),
        Order(account="SIM_pod", orderId=order_id_int, permId=1000 + order_id_int,
              orderRef=order_ref_str, action="SELL", orderType="MKT", tif="OPG", totalQuantity=10,
              filledQuantity=filled_quantity_float),
        OrderStatus(orderId=order_id_int, status="Cancelled"), [], [])


def _completed_fill_obj(order_id_int=1, account_str="SIM_pod", timestamp_ts=None):
    return SimpleNamespace(contract=_FakeContract("BRK B"),
        time=timestamp_ts or datetime(2024, 1, 3, 15, 0, tzinfo=UTC),
        execution=SimpleNamespace(acctNumber=account_str, orderId=order_id_int, permId=1000 + order_id_int,
            side="SLD", shares=4, price=100, execId=f"fill:{order_id_int}"))


def test_capsule_completed_feed_uses_authoritative_quantity_not_default_zero_status(monkeypatch):
    ib_obj = _CapsuleEvidenceIB([_completed_trade_obj()], [_completed_fill_obj()])
    client_obj = _funding_client_obj(monkeypatch, ib_obj)
    record_list, event_list, fill_list = client_obj.get_capsule_order_state_snapshot(
        "SIM_pod", datetime(2024, 1, 3, 14, 30, tzinfo=UTC), submission_key_str="batch")
    assert len(record_list) == len(fill_list) == 1
    assert event_list == []
    assert record_list[0].asset_str == fill_list[0].asset_str == "BRK.B"
    assert record_list[0].filled_amount_float == 4
    assert record_list[0].remaining_amount_float == 6
    assert record_list[0].raw_payload_dict["completed_quantity_verified_bool"] is True
    assert record_list[0].raw_payload_dict["filled_quantity_source_str"] == "completed_order.filledQuantity"
    assert fill_list[0].fill_amount_float == -4
    # Default refresh retains the existing monthly behavior.
    legacy_record_list, _, _ = client_obj.get_recent_order_state_snapshot(
        "SIM_pod", datetime(2024, 1, 3, 14, 30, tzinfo=UTC), submission_key_str="batch")
    assert legacy_record_list[0].filled_amount_float == 0
    assert "completed_quantity_verified_bool" not in legacy_record_list[0].raw_payload_dict


def test_capsule_discovers_no_log_manual_completion_only_with_recent_account_execution(monkeypatch):
    ib_obj = _CapsuleEvidenceIB(
        [_completed_trade_obj(order_id_int, order_ref_str="") for order_id_int in (1, 2, 3, 4)],
        [_completed_fill_obj(1),
         _completed_fill_obj(2, timestamp_ts=datetime(2024, 1, 2, 15, 0, tzinfo=UTC)),
         _completed_fill_obj(3, account_str="OTHER")])
    client_obj = _funding_client_obj(monkeypatch, ib_obj)
    record_list, event_list, fill_list = client_obj.get_capsule_order_state_snapshot(
        "SIM_pod", datetime(2024, 1, 3, 14, 30, tzinfo=UTC))
    assert [record_obj.broker_order_id_str for record_obj in record_list] == ["1001"]
    assert [fill_obj.broker_order_id_str for fill_obj in fill_list] == ["1001"]
    assert event_list == []
    assert record_list[0].order_request_key_str is None


@pytest.mark.parametrize("filled_quantity_float", [float("nan"), 1.7976931348623157e308, -1, 11, None])
def test_capsule_does_not_treat_missing_or_invalid_completed_quantity_as_verified_zero(monkeypatch, filled_quantity_float):
    client_obj = _funding_client_obj(monkeypatch,
        _CapsuleEvidenceIB([_completed_trade_obj(filled_quantity_float=filled_quantity_float)]))
    record_list, _, _ = client_obj.get_capsule_order_state_snapshot(
        "SIM_pod", datetime(2024, 1, 3, 14, 30, tzinfo=UTC), submission_key_str="batch")
    assert record_list[0].raw_payload_dict["completed_quantity_verified_bool"] is False


def test_capsule_zero_fill_completed_sale_is_explicitly_verified(monkeypatch):
    client_obj = _funding_client_obj(monkeypatch,
        _CapsuleEvidenceIB([_completed_trade_obj(filled_quantity_float=0)]))
    record_list, _, fill_list = client_obj.get_capsule_order_state_snapshot(
        "SIM_pod", datetime(2024, 1, 3, 14, 30, tzinfo=UTC), submission_key_str="batch")
    assert record_list[0].filled_amount_float == 0
    assert record_list[0].remaining_amount_float == 10
    assert record_list[0].raw_payload_dict["completed_quantity_verified_bool"] is True
    assert fill_list == []


def test_capsule_account_snapshot_includes_other_client_orders_without_changing_default(monkeypatch):
    ib_obj = _CapsuleEvidenceIB([], open_trade_list=[
        _snapshot_trade_obj(7, order_ref_str="", status_str="Submitted"),
        _snapshot_trade_obj(8, account_str="OTHER", status_str="Submitted")])
    client_obj = _funding_client_obj(monkeypatch, ib_obj)
    assert client_obj.get_account_snapshot("SIM_pod").open_order_id_list == []
    snapshot_obj = client_obj.get_capsule_account_snapshot("SIM_pod")
    assert snapshot_obj.open_order_id_list == ["7"]
    assert snapshot_obj.position_amount_map == {"BRK.B": 6.0}
    assert ib_obj.all_open_call_count_int == 1


def test_capsule_completed_quantity_cannot_hide_simultaneous_other_client_open_order(monkeypatch):
    open_trade_obj = _snapshot_trade_obj(1, status_str="Submitted", log_bool=False)
    client_obj = _funding_client_obj(monkeypatch,
        _CapsuleEvidenceIB([_completed_trade_obj()], [_completed_fill_obj()], [open_trade_obj]))
    record_list, _, _ = client_obj.get_capsule_order_state_snapshot(
        "SIM_pod", datetime(2024, 1, 3, 14, 30, tzinfo=UTC), submission_key_str="batch")
    assert record_list[0].raw_payload_dict["open_order_observed_bool"] is True
    assert record_list[0].raw_payload_dict["completed_quantity_verified_bool"] is True


class _ExpirySubmitIB:
    def __init__(self, clock_list, reject_bool=False, raise_bool=False):
        self.clock_list = clock_list
        self.reject_bool = reject_bool
        self.raise_bool = raise_bool
        self.trade_list = []
        self.placed_order_list = []
        self.after_qualification_ts = None
        self.place_timestamp_ts = None

    def qualifyContracts(self, *contract_list):
        if self.after_qualification_ts is not None:
            self.clock_list[:] = [self.after_qualification_ts]
        return list(contract_list)

    def placeOrder(self, contract_obj, order_obj):
        self.placed_order_list.append(order_obj)
        self.place_timestamp_ts = self.clock_list[0]
        if self.raise_bool:
            raise ValueError("Simulated unsupported MKT/GTD order")
        order_obj.orderId = len(self.placed_order_list)
        order_obj.permId = 1000 + order_obj.orderId
        order_obj.filledQuantity = 0
        status_str = "Inactive" if self.reject_bool else "Submitted"
        trade_obj = SimpleNamespace(contract=contract_obj, order=order_obj,
            orderStatus=SimpleNamespace(orderId=order_obj.orderId, permId=order_obj.permId,
                status=status_str, filled=0.0, remaining=order_obj.totalQuantity, avgFillPrice=0.0),
            log=[SimpleNamespace(time=self.clock_list[0], status=status_str, message="", errorCode=0)], fills=[])
        self.trade_list.append(trade_obj)
        return trade_obj

    def reqOpenOrders(self):
        return [] if self.reject_bool else self.trade_list

    def reqCompletedOrders(self, apiOnly=False):
        return self.trade_list if self.reject_bool else []

    def reqExecutions(self, filter_obj):
        return []


def _expiry_client_obj(monkeypatch, ib_obj):
    monkeypatch.setattr("alpha.live.ibkr_socket_client.Stock",
        lambda symbol_str, exchange_str, currency_str: _FakeContract(symbol_str))
    _freeze_tick_clock(monkeypatch, ib_obj.clock_list)
    client_obj = IBKRSocketClient()
    monkeypatch.setattr(client_obj, "connect", lambda: nullcontext(ib_obj))
    return client_obj


@pytest.mark.parametrize("deadline_str,expected_gtd_str", [
    ("2024-10-04T16:00:00-04:00", "20241004-20:00:00"),
    ("2024-11-29T13:00:00-05:00", "20241129-18:00:00"),
    ("2024-11-29T18:00:00Z", "20241129-18:00:00"),
])
def test_recovery_mkt_uses_fixed_utc_broker_expiry(monkeypatch, deadline_str, expected_gtd_str):
    from dataclasses import asdict, replace
    from alpha.live.models import BrokerOrderRequest
    clock_list = [datetime(2024, 10, 4, 19, 30, tzinfo=UTC)]
    ib_obj = _ExpirySubmitIB(clock_list)
    client_obj = _expiry_client_obj(monkeypatch, ib_obj)
    request_obj = replace(_funding_request_obj(amount_float=-6), broker_order_type_str="MKT",
        execution_deadline_timestamp_str=deadline_str)
    # The durable recovery request remains JSON-serializable and reloadable.
    import json
    assert BrokerOrderRequest(**json.loads(json.dumps(asdict(request_obj)))) == request_obj
    result_obj = client_obj.submit_order_request_list("SIM_pod", [request_obj], clock_list[0])
    assert result_obj.submit_ack_status_str == "complete"
    assert len(ib_obj.placed_order_list) == 1
    order_obj = ib_obj.placed_order_list[0]
    assert (order_obj.orderType, order_obj.tif, order_obj.goodTillDate, order_obj.outsideRth) == (
        "MKT", "GTD", expected_gtd_str, False)
    assert order_obj.action == "SELL"
    assert order_obj.totalQuantity == 6


@pytest.mark.parametrize("deadline_str", ["", "bad-date", "2024-11-29", "2024-11-29T18:00:00", 123])
def test_recovery_expiry_rejects_invalid_or_naive_timestamp_before_connect(monkeypatch, deadline_str):
    from dataclasses import replace
    client_obj = IBKRSocketClient()
    monkeypatch.setattr(client_obj, "connect", lambda: pytest.fail("Invalid expiry must not connect"))
    request_obj = replace(_funding_request_obj(), broker_order_type_str="MKT",
        execution_deadline_timestamp_str=deadline_str)
    with pytest.raises(ValueError, match="aware ISO timestamp"):
        client_obj.submit_order_request_list("SIM_pod", [request_obj], datetime(2024, 11, 29, tzinfo=UTC))


@pytest.mark.parametrize("order_type_str", ["MOO", "MOC", "LMT"])
def test_recovery_expiry_rejects_non_mkt_before_connect(monkeypatch, order_type_str):
    from dataclasses import replace
    client_obj = IBKRSocketClient()
    monkeypatch.setattr(client_obj, "connect", lambda: pytest.fail("Invalid expiry must not connect"))
    request_obj = replace(_funding_request_obj(), broker_order_type_str=order_type_str,
        execution_deadline_timestamp_str="2024-11-29T18:00:00Z")
    with pytest.raises(ValueError, match="only for MKT"):
        client_obj.submit_order_request_list("SIM_pod", [request_obj], datetime(2024, 11, 29, tzinfo=UTC))


def test_recovery_qualification_delay_cannot_roll_expiry_into_next_session(monkeypatch):
    from dataclasses import replace
    clock_list = [datetime(2024, 11, 29, 17, 59, tzinfo=UTC)]
    ib_obj = _ExpirySubmitIB(clock_list, reject_bool=True)
    ib_obj.after_qualification_ts = datetime(2024, 11, 29, 18, 1, tzinfo=UTC)
    client_obj = _expiry_client_obj(monkeypatch, ib_obj)
    request_obj = replace(_funding_request_obj(amount_float=-6), broker_order_type_str="MKT",
        execution_deadline_timestamp_str="2024-11-29T13:00:00-05:00")
    result_obj = client_obj.submit_order_request_list("SIM_pod", [request_obj], clock_list[0])
    assert ib_obj.place_timestamp_ts == datetime(2024, 11, 29, 18, 1, tzinfo=UTC)
    assert len(ib_obj.placed_order_list) == 1
    assert ib_obj.placed_order_list[0].goodTillDate == "20241129-18:00:00"
    assert ib_obj.placed_order_list[0].tif == "GTD"
    assert all(record_obj.status_str == "Inactive" for record_obj in result_obj.broker_order_record_list)


def test_recovery_expiry_submission_error_never_falls_back_to_day(monkeypatch):
    from dataclasses import replace
    clock_list = [datetime(2024, 11, 29, 17, 59, tzinfo=UTC)]
    ib_obj = _ExpirySubmitIB(clock_list, raise_bool=True)
    client_obj = _expiry_client_obj(monkeypatch, ib_obj)
    request_obj = replace(_funding_request_obj(amount_float=-6), broker_order_type_str="MKT",
        execution_deadline_timestamp_str="2024-11-29T18:00:00Z")
    with pytest.raises(ValueError, match="Simulated unsupported"):
        client_obj.submit_order_request_list("SIM_pod", [request_obj], clock_list[0])
    assert len(ib_obj.placed_order_list) == 1
    assert ib_obj.placed_order_list[0].tif == "GTD"


@pytest.mark.parametrize("order_type_str,broker_type_str,tif_str", [
    ("MOO", "MKT", "OPG"), ("MKT", "MKT", "DAY"), ("MOC", "MOC", "DAY"), ("LMT", "LMT", "DAY")])
def test_orders_without_recovery_deadline_preserve_all_broker_fields(monkeypatch, order_type_str, broker_type_str, tif_str):
    from dataclasses import asdict, replace
    from ib_async.order import Order
    clock_list = [datetime(2024, 11, 29, 17, 59, tzinfo=UTC)]
    ib_obj = _ExpirySubmitIB(clock_list)
    client_obj = _expiry_client_obj(monkeypatch, ib_obj)
    request_obj = replace(_funding_request_obj(), broker_order_type_str=order_type_str,
        limit_price_float=123.45 if order_type_str == "LMT" else None)
    assert request_obj.execution_deadline_timestamp_str is None
    client_obj.submit_order_request_list("SIM_pod", [request_obj], clock_list[0])
    legacy_kwargs_dict = {"lmtPrice": 123.45} if order_type_str == "LMT" else {}
    expected_order_obj = Order(action="BUY", totalQuantity=10.0, orderType=broker_type_str, tif=tif_str,
        account="SIM_pod", orderRef=request_obj.order_request_key_str,
        orderId=1, permId=1001, filledQuantity=0, **legacy_kwargs_dict)
    assert asdict(ib_obj.placed_order_list[0]) == asdict(expected_order_obj)
