from __future__ import annotations

from contextlib import contextmanager
from datetime import UTC, datetime
from types import SimpleNamespace

from alpha.live.execution_engine import build_broker_order_request_list_from_vplan, build_vplan
from alpha.live.ibkr_socket_client import IBKRSocketClient
from alpha.live.models import (
    BrokerOrderRequest,
    BrokerSnapshot,
    DecisionPlan,
    LivePriceSnapshot,
    LiveRelease,
    VPlan,
    VPlanRow,
)
from alpha.live.order_clerk import StubBrokerAdapter, infer_ibkr_account_mode_str


def test_stub_broker_adapter_roundtrips_live_price_snapshot():
    broker_adapter_obj = StubBrokerAdapter()
    broker_adapter_obj.seed_account_snapshot(
        account_route_str="DU1",
        cash_float=10000.0,
        total_value_float=10000.0,
        position_amount_map={},
    )
    broker_adapter_obj.seed_live_price_snapshot(
        account_route_str="DU1",
        asset_reference_price_map={"AAPL": 101.0, "MSFT": 202.0},
    )

    live_price_snapshot_obj = broker_adapter_obj.get_live_price_snapshot("DU1", ["AAPL"])

    assert live_price_snapshot_obj.asset_reference_price_map == {"AAPL": 101.0}
    assert live_price_snapshot_obj.asset_reference_source_map_dict == {"AAPL": "stub"}
    assert live_price_snapshot_obj.price_source_str == "stub"


def test_stub_broker_adapter_updates_positions_from_vplan_requests():
    broker_adapter_obj = StubBrokerAdapter()
    broker_adapter_obj.seed_account_snapshot(
        account_route_str="DU1",
        cash_float=10000.0,
        total_value_float=10000.0,
        position_amount_map={},
    )
    broker_adapter_obj.seed_live_price_snapshot(
        account_route_str="DU1",
        asset_reference_price_map={"AAPL": 50.0},
    )
    broker_order_request_list = [
        BrokerOrderRequest(
            decision_plan_id_int=7,
            vplan_id_int=9,
            release_id_str="release_001",
            pod_id_str="pod_001",
            account_route_str="DU1",
            submission_key_str="vplan:9",
            order_request_key_str="vplan:9:AAPL:1",
            asset_str="AAPL",
            broker_order_type_str="MOO",
            order_class_str="MarketOrder",
            unit_str="shares",
            amount_float=10.0,
            target_bool=False,
            trade_id_int=None,
            sizing_reference_price_float=50.0,
            portfolio_value_float=5000.0,
        )
    ]

    submit_batch_result_obj = broker_adapter_obj.submit_order_request_list(
        account_route_str="DU1",
        broker_order_request_list=broker_order_request_list,
        submitted_timestamp_ts=datetime.now(UTC),
    )
    refreshed_broker_order_record_list, refreshed_broker_order_event_list, refreshed_broker_fill_list = (
        broker_adapter_obj.get_recent_order_state_snapshot(
            account_route_str="DU1",
            since_timestamp_ts=datetime.min.replace(tzinfo=UTC),
        )
    )
    broker_snapshot_obj = broker_adapter_obj.get_account_snapshot("DU1")

    assert len(submit_batch_result_obj.broker_order_record_list) == 1
    assert len(submit_batch_result_obj.broker_order_event_list) == 1
    assert submit_batch_result_obj.broker_order_event_list[0].status_str == "PendingSubmit"
    assert submit_batch_result_obj.broker_order_record_list[0].decision_plan_id_int == 7
    assert submit_batch_result_obj.broker_order_record_list[0].vplan_id_int == 9
    assert submit_batch_result_obj.broker_order_fill_list == []
    assert len(submit_batch_result_obj.broker_order_ack_list) == 1
    assert submit_batch_result_obj.broker_order_ack_list[0].broker_response_ack_bool is True
    assert submit_batch_result_obj.submit_ack_status_str == "complete"
    assert len(refreshed_broker_order_record_list) == 1
    assert refreshed_broker_order_record_list[0].status_str == "Filled"
    assert len(refreshed_broker_order_event_list) == 2
    assert len(refreshed_broker_fill_list) == 1
    assert refreshed_broker_fill_list[0].decision_plan_id_int == 7
    assert refreshed_broker_fill_list[0].vplan_id_int == 9
    assert broker_snapshot_obj.position_amount_map["AAPL"] == 10.0


def test_stub_broker_adapter_filters_snapshot_by_submission_key():
    broker_adapter_obj = StubBrokerAdapter()
    broker_adapter_obj.seed_account_snapshot(
        account_route_str="DU1",
        cash_float=10000.0,
        total_value_float=10000.0,
        position_amount_map={},
    )
    broker_adapter_obj.seed_live_price_snapshot(
        account_route_str="DU1",
        asset_reference_price_map={"AAPL": 50.0, "MSFT": 25.0},
    )
    broker_adapter_obj.submit_order_request_list(
        account_route_str="DU1",
        broker_order_request_list=[
            BrokerOrderRequest(
                decision_plan_id_int=7,
                vplan_id_int=9,
                release_id_str="release_001",
                pod_id_str="pod_001",
                account_route_str="DU1",
                submission_key_str="vplan:9",
                order_request_key_str="vplan:9:AAPL:1",
                asset_str="AAPL",
                broker_order_type_str="MOO",
                order_class_str="MarketOrder",
                unit_str="shares",
                amount_float=10.0,
                target_bool=False,
                trade_id_int=None,
                sizing_reference_price_float=50.0,
                portfolio_value_float=5000.0,
            )
        ],
        submitted_timestamp_ts=datetime.now(UTC),
    )
    broker_adapter_obj.submit_order_request_list(
        account_route_str="DU1",
        broker_order_request_list=[
            BrokerOrderRequest(
                decision_plan_id_int=8,
                vplan_id_int=10,
                release_id_str="release_001",
                pod_id_str="pod_001",
                account_route_str="DU1",
                submission_key_str="vplan:10",
                order_request_key_str="vplan:10:MSFT:1",
                asset_str="MSFT",
                broker_order_type_str="MOO",
                order_class_str="MarketOrder",
                unit_str="shares",
                amount_float=4.0,
                target_bool=False,
                trade_id_int=None,
                sizing_reference_price_float=25.0,
                portfolio_value_float=5000.0,
            )
        ],
        submitted_timestamp_ts=datetime.now(UTC),
    )

    broker_order_record_list, broker_order_event_list, broker_order_fill_list = (
        broker_adapter_obj.get_recent_order_state_snapshot(
            account_route_str="DU1",
            since_timestamp_ts=datetime.min.replace(tzinfo=UTC),
            submission_key_str="vplan:9",
        )
    )

    assert [broker_order_record_obj.asset_str for broker_order_record_obj in broker_order_record_list] == ["AAPL"]
    assert {broker_order_event_obj.asset_str for broker_order_event_obj in broker_order_event_list} == {"AAPL"}
    assert [broker_order_fill_obj.asset_str for broker_order_fill_obj in broker_order_fill_list] == ["AAPL"]


def test_stub_broker_adapter_session_open_price_roundtrip():
    broker_adapter_obj = StubBrokerAdapter()
    broker_adapter_obj.seed_account_snapshot(
        account_route_str="DU1",
        cash_float=10000.0,
        total_value_float=10000.0,
        position_amount_map={},
    )
    broker_adapter_obj.seed_session_open_price(
        account_route_str="DU1",
        session_date_str="2024-02-01",
        asset_str="AAPL",
        official_open_price_float=101.25,
    )

    session_open_price_list = broker_adapter_obj.get_session_open_price_list(
        account_route_str="DU1",
        asset_str_list=["AAPL"],
        session_open_timestamp_ts=datetime(2024, 2, 1, 9, 30, tzinfo=UTC),
        session_calendar_id_str="XNYS",
    )

    assert len(session_open_price_list) == 1
    assert session_open_price_list[0].official_open_price_float == 101.25
    assert session_open_price_list[0].open_price_source_str == "stub.seeded_open"


def test_infer_ibkr_account_mode_str_uses_du_prefix_for_paper():
    assert infer_ibkr_account_mode_str("DU1234567") == "paper"
    assert infer_ibkr_account_mode_str("U1234567") == "live"


def test_ibkr_socket_client_live_price_snapshot_keeps_account_route(monkeypatch):
    class DummyContract:
        def __init__(self, symbol_str: str):
            self.symbol = symbol_str

    class DummyTicker:
        def __init__(self, symbol_str: str, market_price_float: float):
            self.contract = DummyContract(symbol_str)
            self._market_price_float = market_price_float

        def marketPrice(self) -> float:
            return self._market_price_float

    class DummyIB:
        def reqTickers(self, *contract_obj_tuple):
            del contract_obj_tuple
            return [DummyTicker("AAPL", 101.5)]

    ibkr_socket_client_obj = IBKRSocketClient()

    @contextmanager
    def fake_connect():
        yield DummyIB()

    monkeypatch.setattr(ibkr_socket_client_obj, "connect", fake_connect)
    monkeypatch.setattr(
        ibkr_socket_client_obj,
        "_build_stock_contract_map",
        lambda ib_obj, asset_str_list: {asset_str: DummyContract(asset_str) for asset_str in asset_str_list},
    )

    live_price_snapshot_obj = ibkr_socket_client_obj.get_live_price_snapshot("DU1", ["AAPL"])

    assert live_price_snapshot_obj.account_route_str == "DU1"
    assert live_price_snapshot_obj.asset_reference_price_map == {"AAPL": 101.5}
    assert live_price_snapshot_obj.asset_reference_source_map_dict == {
        "AAPL": "ib_async.reqTickers.marketPrice"
    }
    assert live_price_snapshot_obj.price_source_str == "ib_async.reqTickers.marketPrice"


def test_ibkr_socket_client_live_price_snapshot_uses_auction_first_for_next_open_moo(monkeypatch):
    class DummyContract:
        def __init__(self, symbol_str: str):
            self.symbol = symbol_str

    class DummyTicker:
        def __init__(
            self,
            symbol_str: str,
            market_price_float: float | None,
            auction_price_float: float | None,
        ):
            self.contract = DummyContract(symbol_str)
            self._market_price_float = market_price_float
            self.auctionPrice = auction_price_float

        def marketPrice(self) -> float | None:
            return self._market_price_float

    class DummyIB:
        def __init__(self):
            self.cancelled_symbol_list: list[str] = []
            self.sleep_seconds_float: float | None = None
            self.req_tickers_call_count_int = 0

        def reqMktData(self, contract_obj, genericTickList: str = "", snapshot: bool = False):
            assert genericTickList == "225"
            assert snapshot is False
            return DummyTicker(contract_obj.symbol, market_price_float=100.5, auction_price_float=101.25)

        def sleep(self, seconds_float: float):
            self.sleep_seconds_float = seconds_float
            return True

        def cancelMktData(self, contract_obj):
            self.cancelled_symbol_list.append(contract_obj.symbol)
            return True

        def reqTickers(self, *contract_obj_tuple):
            self.req_tickers_call_count_int += 1
            return []

    dummy_ib_obj = DummyIB()
    ibkr_socket_client_obj = IBKRSocketClient()

    @contextmanager
    def fake_connect():
        yield dummy_ib_obj

    monkeypatch.setattr(ibkr_socket_client_obj, "connect", fake_connect)
    monkeypatch.setattr(
        ibkr_socket_client_obj,
        "_build_stock_contract_map",
        lambda ib_obj, asset_str_list: {asset_str: DummyContract(asset_str) for asset_str in asset_str_list},
    )

    live_price_snapshot_obj = ibkr_socket_client_obj.get_live_price_snapshot(
        "DU1",
        ["AAPL"],
        execution_policy_str="next_open_moo",
    )

    assert live_price_snapshot_obj.asset_reference_price_map == {"AAPL": 101.25}
    assert live_price_snapshot_obj.asset_reference_source_map_dict == {
        "AAPL": "ib_async.reqMktData.225.auctionPrice"
    }
    assert live_price_snapshot_obj.price_source_str == "auction_225"
    assert dummy_ib_obj.req_tickers_call_count_int == 0
    assert dummy_ib_obj.cancelled_symbol_list == ["AAPL"]
    assert dummy_ib_obj.sleep_seconds_float == 2.0


def test_ibkr_socket_client_live_price_snapshot_uses_req_tickers_fallback_for_unresolved_assets(monkeypatch):
    class DummyContract:
        def __init__(self, symbol_str: str):
            self.symbol = symbol_str

    class DummyTicker:
        def __init__(
            self,
            symbol_str: str,
            market_price_float: float | None,
            auction_price_float: float | None,
        ):
            self.contract = DummyContract(symbol_str)
            self._market_price_float = market_price_float
            self.auctionPrice = auction_price_float

        def marketPrice(self) -> float | None:
            return self._market_price_float

    class DummyIB:
        def __init__(self):
            self.cancelled_symbol_list: list[str] = []
            self.req_tickers_requested_symbol_tuple_list: list[tuple[str, ...]] = []

        def reqMktData(self, contract_obj, genericTickList: str = "", snapshot: bool = False):
            assert genericTickList == "225"
            assert snapshot is False
            if contract_obj.symbol == "AAPL":
                return DummyTicker("AAPL", market_price_float=None, auction_price_float=None)
            return DummyTicker("MSFT", market_price_float=200.0, auction_price_float=201.0)

        def sleep(self, seconds_float: float):
            del seconds_float
            return True

        def cancelMktData(self, contract_obj):
            self.cancelled_symbol_list.append(contract_obj.symbol)
            return True

        def reqTickers(self, *contract_obj_tuple):
            self.req_tickers_requested_symbol_tuple_list.append(
                tuple(contract_obj.symbol for contract_obj in contract_obj_tuple)
            )
            return [DummyTicker("AAPL", market_price_float=99.5, auction_price_float=None)]

    dummy_ib_obj = DummyIB()
    ibkr_socket_client_obj = IBKRSocketClient()

    @contextmanager
    def fake_connect():
        yield dummy_ib_obj

    monkeypatch.setattr(ibkr_socket_client_obj, "connect", fake_connect)
    monkeypatch.setattr(
        ibkr_socket_client_obj,
        "_build_stock_contract_map",
        lambda ib_obj, asset_str_list: {asset_str: DummyContract(asset_str) for asset_str in asset_str_list},
    )

    live_price_snapshot_obj = ibkr_socket_client_obj.get_live_price_snapshot(
        "DU1",
        ["AAPL", "MSFT"],
        execution_policy_str="next_open_moo",
    )

    assert live_price_snapshot_obj.asset_reference_price_map == {
        "AAPL": 99.5,
        "MSFT": 201.0,
    }
    assert live_price_snapshot_obj.asset_reference_source_map_dict == {
        "AAPL": "ib_async.reqTickers.marketPrice.fallback",
        "MSFT": "ib_async.reqMktData.225.auctionPrice",
    }
    assert live_price_snapshot_obj.price_source_str == "mixed"
    assert dummy_ib_obj.req_tickers_requested_symbol_tuple_list == [("AAPL",)]
    assert dummy_ib_obj.cancelled_symbol_list == ["AAPL", "MSFT"]


def test_build_broker_order_request_list_from_vplan_assigns_unique_order_request_keys():
    vplan_obj = VPlan(
        release_id_str="release_001",
        user_id_str="user_001",
        pod_id_str="pod_001",
        account_route_str="DU1",
        decision_plan_id_int=7,
        signal_timestamp_ts=datetime.now(UTC),
        submission_timestamp_ts=datetime.now(UTC),
        target_execution_timestamp_ts=datetime.now(UTC),
        execution_policy_str="next_open_moo",
        broker_snapshot_timestamp_ts=datetime.now(UTC),
        live_reference_snapshot_timestamp_ts=datetime.now(UTC),
        live_price_source_str="stub",
        net_liq_float=10000.0,
        available_funds_float=9000.0,
        excess_liquidity_float=9000.0,
        pod_budget_fraction_float=0.5,
        pod_budget_float=5000.0,
        current_broker_position_map={},
        live_reference_price_map={"AAPL": 100.0, "MSFT": 50.0},
        target_share_map={"AAPL": 10.0, "MSFT": 20.0},
        order_delta_map={"AAPL": 10.0, "MSFT": 20.0},
        vplan_row_list=[
            VPlanRow(
                asset_str="AAPL",
                current_share_float=0.0,
                target_share_float=10.0,
                order_delta_share_float=10.0,
                live_reference_price_float=100.0,
                estimated_target_notional_float=1000.0,
                broker_order_type_str="MOO",
            ),
            VPlanRow(
                asset_str="MSFT",
                current_share_float=0.0,
                target_share_float=20.0,
                order_delta_share_float=20.0,
                live_reference_price_float=50.0,
                estimated_target_notional_float=1000.0,
                broker_order_type_str="MOO",
            ),
        ],
        submission_key_str="vplan:7",
        vplan_id_int=9,
    )

    broker_order_request_list = build_broker_order_request_list_from_vplan(vplan_obj)

    assert [broker_order_request_obj.order_request_key_str for broker_order_request_obj in broker_order_request_list] == [
        "vplan:7:AAPL:1",
        "vplan:7:MSFT:2",
    ]
    assert len(
        {
            broker_order_request_obj.order_request_key_str
            for broker_order_request_obj in broker_order_request_list
        }
    ) == 2


def test_next_open_market_vplan_uses_plain_mkt_broker_order_type():
    timestamp_ts = datetime(2024, 2, 1, 9, 30)
    release_obj = LiveRelease(
        release_id_str="release_market_001",
        user_id_str="user_001",
        pod_id_str="pod_market",
        account_route_str="DU1",
        strategy_import_str="strategies.dv2.strategy_mr_dv2:DVO2Strategy",
        mode_str="paper",
        session_calendar_id_str="XNYS",
        signal_clock_str="eod_snapshot_ready",
        execution_policy_str="next_open_market",
        data_profile_str="norgate_eod_sp500_pit",
        params_dict={},
        risk_profile_str="standard",
        enabled_bool=True,
        source_path_str="manifest.yaml",
        pod_budget_fraction_float=0.5,
    )
    decision_plan_obj = DecisionPlan(
        release_id_str=release_obj.release_id_str,
        user_id_str=release_obj.user_id_str,
        pod_id_str=release_obj.pod_id_str,
        account_route_str=release_obj.account_route_str,
        signal_timestamp_ts=timestamp_ts,
        submission_timestamp_ts=timestamp_ts,
        target_execution_timestamp_ts=timestamp_ts,
        execution_policy_str="next_open_market",
        decision_base_position_map={},
        snapshot_metadata_dict={},
        strategy_state_dict={},
        decision_book_type_str="incremental_entry_exit_book",
        entry_target_weight_map_dict={"AAPL": 0.2},
        entry_priority_list=["AAPL"],
        decision_plan_id_int=7,
    )
    broker_snapshot_obj = BrokerSnapshot(
        account_route_str="DU1",
        snapshot_timestamp_ts=timestamp_ts,
        cash_float=10000.0,
        total_value_float=10000.0,
        net_liq_float=10000.0,
        position_amount_map={},
    )
    live_price_snapshot_obj = LivePriceSnapshot(
        account_route_str="DU1",
        snapshot_timestamp_ts=timestamp_ts,
        price_source_str="stub",
        asset_reference_price_map={"AAPL": 100.0},
    )

    vplan_obj = build_vplan(
        release_obj=release_obj,
        decision_plan_obj=decision_plan_obj,
        broker_snapshot_obj=broker_snapshot_obj,
        live_price_snapshot_obj=live_price_snapshot_obj,
    )
    broker_order_request_list = build_broker_order_request_list_from_vplan(vplan_obj)

    assert vplan_obj.vplan_row_list[0].broker_order_type_str == "MKT"
    assert broker_order_request_list[0].broker_order_type_str == "MKT"


def test_ibkr_socket_client_submit_ack_builder_requires_broker_correlated_response():
    broker_order_request_obj = BrokerOrderRequest(
        decision_plan_id_int=7,
        vplan_id_int=9,
        release_id_str="release_001",
        pod_id_str="pod_001",
        account_route_str="DU1",
        submission_key_str="vplan:9",
        order_request_key_str="vplan:9:AAPL:1",
        asset_str="AAPL",
        broker_order_type_str="MOO",
        order_class_str="MarketOrder",
        unit_str="shares",
        amount_float=10.0,
        target_bool=False,
        trade_id_int=None,
        sizing_reference_price_float=50.0,
        portfolio_value_float=5000.0,
    )
    ibkr_socket_client_obj = IBKRSocketClient()

    broker_order_ack_list = ibkr_socket_client_obj._build_broker_order_ack_list(
        broker_order_request_list=[broker_order_request_obj],
        broker_order_record_list=[],
        broker_order_event_list=[],
        broker_order_fill_list=[],
        local_broker_order_id_alias_set_by_request_key_map_dict={
            "vplan:9:AAPL:1": {"local_order_id_123"}
        },
    )

    assert len(broker_order_ack_list) == 1
    assert broker_order_ack_list[0].local_submit_ack_bool is True
    assert broker_order_ack_list[0].broker_response_ack_bool is False
    assert broker_order_ack_list[0].ack_status_str == "missing_critical"
    assert broker_order_ack_list[0].ack_source_str == "missing"


def test_stub_broker_adapter_submit_ack_is_order_type_agnostic():
    broker_adapter_obj = StubBrokerAdapter()
    broker_adapter_obj.seed_account_snapshot(
        account_route_str="DU1",
        cash_float=10000.0,
        total_value_float=10000.0,
        position_amount_map={},
    )
    broker_adapter_obj.seed_live_price_snapshot(
        account_route_str="DU1",
        asset_reference_price_map={
            "AAPL": 100.0,
            "MSFT": 200.0,
            "TLT": 90.0,
            "GLD": 180.0,
        },
    )
    broker_order_request_list = [
        BrokerOrderRequest(
            decision_plan_id_int=7,
            vplan_id_int=9,
            release_id_str="release_001",
            pod_id_str="pod_001",
            account_route_str="DU1",
            submission_key_str="vplan:9",
            order_request_key_str="vplan:9:AAPL:1",
            asset_str="AAPL",
            broker_order_type_str="MOO",
            order_class_str="MarketOrder",
            unit_str="shares",
            amount_float=1.0,
            target_bool=False,
            trade_id_int=None,
            sizing_reference_price_float=100.0,
            portfolio_value_float=5000.0,
        ),
        BrokerOrderRequest(
            decision_plan_id_int=7,
            vplan_id_int=9,
            release_id_str="release_001",
            pod_id_str="pod_001",
            account_route_str="DU1",
            submission_key_str="vplan:9",
            order_request_key_str="vplan:9:MSFT:2",
            asset_str="MSFT",
            broker_order_type_str="MOC",
            order_class_str="MarketOrder",
            unit_str="shares",
            amount_float=1.0,
            target_bool=False,
            trade_id_int=None,
            sizing_reference_price_float=200.0,
            portfolio_value_float=5000.0,
        ),
        BrokerOrderRequest(
            decision_plan_id_int=7,
            vplan_id_int=9,
            release_id_str="release_001",
            pod_id_str="pod_001",
            account_route_str="DU1",
            submission_key_str="vplan:9",
            order_request_key_str="vplan:9:TLT:3",
            asset_str="TLT",
            broker_order_type_str="MKT",
            order_class_str="MarketOrder",
            unit_str="shares",
            amount_float=1.0,
            target_bool=False,
            trade_id_int=None,
            sizing_reference_price_float=90.0,
            portfolio_value_float=5000.0,
        ),
        BrokerOrderRequest(
            decision_plan_id_int=7,
            vplan_id_int=9,
            release_id_str="release_001",
            pod_id_str="pod_001",
            account_route_str="DU1",
            submission_key_str="vplan:9",
            order_request_key_str="vplan:9:GLD:4",
            asset_str="GLD",
            broker_order_type_str="LOO",
            order_class_str="LimitOrder",
            unit_str="shares",
            amount_float=1.0,
            target_bool=False,
            trade_id_int=None,
            sizing_reference_price_float=180.0,
            portfolio_value_float=5000.0,
        ),
    ]

    submit_batch_result_obj = broker_adapter_obj.submit_order_request_list(
        account_route_str="DU1",
        broker_order_request_list=broker_order_request_list,
        submitted_timestamp_ts=datetime.now(UTC),
    )

    assert submit_batch_result_obj.submit_ack_status_str == "complete"
    assert submit_batch_result_obj.missing_ack_asset_list == []
    assert submit_batch_result_obj.ack_coverage_ratio_float == 1.0
    assert {
        broker_order_ack_obj.broker_order_type_str
        for broker_order_ack_obj in submit_batch_result_obj.broker_order_ack_list
    } == {"MOO", "MOC", "MKT", "LOO"}


def test_ibkr_socket_client_submits_plain_mkt_without_opg_tif(monkeypatch):
    class DummyContract:
        def __init__(self, symbol_str: str):
            self.symbol = symbol_str

    class DummyTrade:
        def __init__(self, contract_obj, order_obj):
            self.contract = contract_obj
            self.order = order_obj
            self.orderStatus = SimpleNamespace(
                orderId=int(order_obj.orderId),
                permId=0,
                status="Submitted",
                filled=0.0,
                remaining=float(order_obj.totalQuantity),
                avgFillPrice=0.0,
            )
            self.log = []
            self.fills = []

    class DummyIB:
        def __init__(self):
            self.trade_list = []
            self.placed_order_list = []

        def placeOrder(self, contract_obj, order_obj):
            order_obj.orderId = len(self.trade_list) + 1
            order_obj.permId = 0
            trade_obj = DummyTrade(contract_obj, order_obj)
            self.trade_list.append(trade_obj)
            self.placed_order_list.append(order_obj)
            return trade_obj

        def reqOpenOrders(self):
            return list(self.trade_list)

        def reqCompletedOrders(self, apiOnly: bool = False):
            del apiOnly
            return []

        def reqExecutions(self, filter_obj):
            del filter_obj
            return []

        def sleep(self, seconds_float: float):
            del seconds_float
            return True

    dummy_ib_obj = DummyIB()
    ibkr_socket_client_obj = IBKRSocketClient()

    @contextmanager
    def fake_connect():
        yield dummy_ib_obj

    monkeypatch.setattr(ibkr_socket_client_obj, "connect", fake_connect)
    monkeypatch.setattr(
        ibkr_socket_client_obj,
        "_build_stock_contract_map",
        lambda ib_obj, asset_str_list: {asset_str: DummyContract(asset_str) for asset_str in asset_str_list},
    )
    broker_order_request_obj = BrokerOrderRequest(
        decision_plan_id_int=7,
        vplan_id_int=9,
        release_id_str="release_001",
        pod_id_str="pod_001",
        account_route_str="DU1",
        submission_key_str="vplan:9",
        order_request_key_str="vplan:9:AAPL:1",
        asset_str="AAPL",
        broker_order_type_str="MKT",
        order_class_str="MarketOrder",
        unit_str="shares",
        amount_float=1.0,
        target_bool=False,
        trade_id_int=None,
        sizing_reference_price_float=100.0,
        portfolio_value_float=5000.0,
    )

    submit_batch_result_obj = ibkr_socket_client_obj.submit_order_request_list(
        account_route_str="DU1",
        broker_order_request_list=[broker_order_request_obj],
        submitted_timestamp_ts=datetime.now(UTC),
    )

    submitted_order_obj = dummy_ib_obj.placed_order_list[0]
    assert submit_batch_result_obj.submit_ack_status_str == "complete"
    assert submitted_order_obj.orderType == "MKT"
    assert str(getattr(submitted_order_obj, "tif", "")).upper() != "OPG"
