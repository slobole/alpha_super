from __future__ import annotations

from datetime import UTC, datetime

from alpha.live.models import ExecutionQualitySnapshot


def build_execution_quality_snapshot(
    order_plan_id_int: int,
    pod_id_str: str,
    execution_fill_input_row_list: list[dict],
) -> ExecutionQualitySnapshot:
    reference_notional_float = 0.0
    actual_notional_float = 0.0
    slippage_cash_float = 0.0

    for execution_fill_input_row_dict in execution_fill_input_row_list:
        fill_amount_float = float(execution_fill_input_row_dict["fill_amount_float"])
        reference_price_float = float(execution_fill_input_row_dict["reference_price_float"])
        fill_price_float = float(execution_fill_input_row_dict["fill_price_float"])

        reference_notional_float += abs(fill_amount_float) * reference_price_float
        actual_notional_float += abs(fill_amount_float) * fill_price_float

        # Positive slippage_cash_float is worse than the frozen model price for both buys and sells:
        #
        # slippage_cash = (fill_price - reference_price) * signed_fill_amount
        #
        # buy:  q > 0, worse if fill > ref  -> positive
        # sell: q < 0, worse if fill < ref  -> positive
        slippage_cash_float += (fill_price_float - reference_price_float) * fill_amount_float

    if reference_notional_float == 0.0:
        slippage_bps_float = 0.0
    else:
        slippage_bps_float = 10_000.0 * slippage_cash_float / reference_notional_float

    return ExecutionQualitySnapshot(
        order_plan_id_int=int(order_plan_id_int),
        pod_id_str=str(pod_id_str),
        reference_notional_float=float(reference_notional_float),
        actual_notional_float=float(actual_notional_float),
        slippage_cash_float=float(slippage_cash_float),
        slippage_bps_float=float(slippage_bps_float),
        fill_count_int=len(execution_fill_input_row_list),
        computed_timestamp_ts=datetime.now(UTC),
    )
