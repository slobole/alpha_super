from __future__ import annotations

from datetime import UTC, datetime

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


class _FakeContract:
    def __init__(self, symbol: str):
        self.symbol = str(symbol)


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


def test_ibkr_socket_client_tick_open_reads_only_ticker_open(monkeypatch):
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
        session_open_timestamp_ts=datetime(2024, 1, 3, 9, 30, tzinfo=UTC),
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
