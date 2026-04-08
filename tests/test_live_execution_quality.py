from __future__ import annotations

from alpha.live.execution_quality import build_execution_quality_snapshot


def test_execution_quality_snapshot_prices_buy_slippage_as_positive_cash_cost():
    execution_quality_snapshot_obj = build_execution_quality_snapshot(
        order_plan_id_int=17,
        pod_id_str="pod_001",
        execution_fill_input_row_list=[
            {
                "fill_amount_float": 10.0,
                "reference_price_float": 100.0,
                "fill_price_float": 101.0,
            }
        ],
    )

    assert execution_quality_snapshot_obj.reference_notional_float == 1000.0
    assert execution_quality_snapshot_obj.actual_notional_float == 1010.0
    assert execution_quality_snapshot_obj.slippage_cash_float == 10.0
    assert execution_quality_snapshot_obj.slippage_bps_float == 100.0


def test_execution_quality_snapshot_prices_sell_slippage_as_positive_cash_cost():
    execution_quality_snapshot_obj = build_execution_quality_snapshot(
        order_plan_id_int=18,
        pod_id_str="pod_002",
        execution_fill_input_row_list=[
            {
                "fill_amount_float": -10.0,
                "reference_price_float": 100.0,
                "fill_price_float": 99.0,
            }
        ],
    )

    assert execution_quality_snapshot_obj.reference_notional_float == 1000.0
    assert execution_quality_snapshot_obj.actual_notional_float == 990.0
    assert execution_quality_snapshot_obj.slippage_cash_float == 10.0
    assert execution_quality_snapshot_obj.slippage_bps_float == 100.0
