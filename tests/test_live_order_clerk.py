from __future__ import annotations

from datetime import UTC, datetime

from alpha.live.models import BrokerOrderRequest
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

    broker_order_record_list, broker_fill_list = broker_adapter_obj.submit_order_request_list(
        account_route_str="DU1",
        broker_order_request_list=broker_order_request_list,
        submitted_timestamp_ts=datetime.now(UTC),
    )
    broker_snapshot_obj = broker_adapter_obj.get_account_snapshot("DU1")

    assert len(broker_order_record_list) == 1
    assert broker_order_record_list[0].decision_plan_id_int == 7
    assert broker_order_record_list[0].vplan_id_int == 9
    assert len(broker_fill_list) == 1
    assert broker_fill_list[0].decision_plan_id_int == 7
    assert broker_fill_list[0].vplan_id_int == 9
    assert broker_snapshot_obj.position_amount_map["AAPL"] == 10.0


def test_infer_ibkr_account_mode_str_uses_du_prefix_for_paper():
    assert infer_ibkr_account_mode_str("DU1234567") == "paper"
    assert infer_ibkr_account_mode_str("U1234567") == "live"
