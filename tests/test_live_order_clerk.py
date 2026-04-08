from __future__ import annotations

from alpha.live.models import FrozenOrderIntent
from alpha.live.order_clerk import (
    StubBrokerAdapter,
    build_broker_order_request_list,
    infer_ibkr_account_mode_str,
)


def test_order_clerk_builds_requests_for_open_and_close_policies():
    order_intent_list = [
        FrozenOrderIntent(
            asset_str="AAPL",
            order_class_str="MarketOrder",
            unit_str="shares",
            amount_float=10.0,
            target_bool=False,
            trade_id_int=1,
            broker_order_type_str="MOO",
            sizing_reference_price_float=100.0,
            portfolio_value_float=10000.0,
        ),
        FrozenOrderIntent(
            asset_str="MSFT",
            order_class_str="MarketOrder",
            unit_str="shares",
            amount_float=5.0,
            target_bool=False,
            trade_id_int=2,
            broker_order_type_str="MOC",
            sizing_reference_price_float=200.0,
            portfolio_value_float=10000.0,
        ),
    ]

    broker_order_request_list = build_broker_order_request_list(
        order_plan_id_int=1,
        release_id_str="release_001",
        pod_id_str="pod_001",
        account_route_str="U1",
        submission_key_str="order_plan:1",
        order_intent_list=order_intent_list,
        order_intent_id_list=[11, 12],
    )

    assert broker_order_request_list[0].broker_order_type_str == "MOO"
    assert broker_order_request_list[1].broker_order_type_str == "MOC"


def test_stub_broker_adapter_updates_positions_from_order_requests():
    broker_adapter_obj = StubBrokerAdapter()
    broker_adapter_obj.seed_account_snapshot(
        account_route_str="U1",
        cash_float=10000.0,
        total_value_float=10000.0,
        position_amount_map={},
    )
    broker_order_request_list = build_broker_order_request_list(
        order_plan_id_int=1,
        release_id_str="release_001",
        pod_id_str="pod_001",
        account_route_str="U1",
        submission_key_str="order_plan:1",
        order_intent_list=[
            FrozenOrderIntent(
                asset_str="AAPL",
                order_class_str="MarketOrder",
                unit_str="shares",
                amount_float=10.0,
                target_bool=False,
                trade_id_int=1,
                broker_order_type_str="MOO",
                sizing_reference_price_float=50.0,
                portfolio_value_float=10000.0,
            )
        ],
        order_intent_id_list=[101],
    )

    broker_order_record_list, broker_fill_list = broker_adapter_obj.submit_order_request_list(
        account_route_str="U1",
        broker_order_request_list=broker_order_request_list,
        submitted_timestamp_ts=broker_adapter_obj.get_account_snapshot("U1").snapshot_timestamp_ts,
    )
    broker_snapshot_obj = broker_adapter_obj.get_account_snapshot("U1")

    assert len(broker_order_record_list) == 1
    assert len(broker_fill_list) == 1
    assert broker_snapshot_obj.position_amount_map["AAPL"] == 10.0


def test_infer_ibkr_account_mode_str_uses_du_prefix_for_paper():
    assert infer_ibkr_account_mode_str("DU1234567") == "paper"
    assert infer_ibkr_account_mode_str("U1234567") == "live"
