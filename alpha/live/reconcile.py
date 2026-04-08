from __future__ import annotations

from alpha.live.models import BrokerPositionSnapshot, ReconciliationResult


def reconcile_account_state(
    model_position_map: dict[str, float],
    model_cash_float: float,
    broker_snapshot_obj: BrokerPositionSnapshot,
    tolerance_float: float = 1e-9,
) -> ReconciliationResult:
    compared_symbol_set = set(model_position_map) | set(broker_snapshot_obj.position_amount_map)
    mismatch_dict: dict[str, dict[str, float]] = {}

    for asset_str in sorted(compared_symbol_set):
        model_amount_float = float(model_position_map.get(asset_str, 0.0))
        broker_amount_float = float(broker_snapshot_obj.position_amount_map.get(asset_str, 0.0))
        if abs(model_amount_float - broker_amount_float) > tolerance_float:
            mismatch_dict[asset_str] = {
                "model_amount_float": model_amount_float,
                "broker_amount_float": broker_amount_float,
            }

    cash_mismatch_float = float(model_cash_float) - float(broker_snapshot_obj.cash_float)
    if abs(cash_mismatch_float) > tolerance_float:
        mismatch_dict["__cash__"] = {
            "model_cash_float": float(model_cash_float),
            "broker_cash_float": float(broker_snapshot_obj.cash_float),
        }

    passed_bool = len(mismatch_dict) == 0
    return ReconciliationResult(
        passed_bool=passed_bool,
        status_str="passed" if passed_bool else "blocked",
        mismatch_dict=mismatch_dict,
        model_position_map={asset_str: float(amount_float) for asset_str, amount_float in model_position_map.items()},
        broker_position_map={
            asset_str: float(amount_float)
            for asset_str, amount_float in broker_snapshot_obj.position_amount_map.items()
        },
        model_cash_float=float(model_cash_float),
        broker_cash_float=float(broker_snapshot_obj.cash_float),
    )
