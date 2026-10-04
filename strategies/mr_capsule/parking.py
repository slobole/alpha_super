"""Idle-cash parking for the MR capsule pods: SPMO while the gate is closed, BIL otherwise (PM_READY, no live route).

Decided 2026-10-04 (docs/research/MR_CAPSULE_20261003.md, "Final parking decision"). After the pod has placed its
stock orders for the decision close T (fills at Open_(T+1)):

    idle_value     = max(pod_value_T - stock_value_after_orders_T - buffer x pod_value_T, 0)
    re-target close  the last session of the ISO week, or a close at which the gate switched (opened or shut)
    gate open      SPMO target 0 shares; BIL holds the idle value
    gate closed    on a re-target close: SPMO target = floor(spmo_weight_T x idle_value / SPMO_Close_T)
                   on other closes SPMO is left untouched
    BIL            on a re-target close: BIL holds the rest of the idle value (bought or sold)
                   on other closes BIL is only SOLD, when it holds more than the rest of the idle value, so the
                   day's stock orders are funded without borrowing; cash from exits waits for the next re-target
                   close (weekly sweep, build amendment B2: re-targeting BIL on every trading close turned the
                   pod over about 20 times a year in BIL alone, which cost about what BIL yields)
    band           BIL is traded only when it is off target by more than the band (or must go to zero)

All sizing uses Close_T (the engine's sizing price) and whole shares; fills are the engine's next-open fills with its
slippage and commissions. The engine applies no cash check: same-open sells fund same-open buys, and the cash buffer
absorbs gaps, slippage, commissions and rounding (negative cash stays an engine diagnostic, G-023).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from strategies.mr_capsule.vix_stress_gate import BIL_SYMBOL_STR, SPMO_SYMBOL_STR

PARKING_SYMBOL_TUPLE = (SPMO_SYMBOL_STR, BIL_SYMBOL_STR)


def require_parking_dividends(pricing_data_df) -> None:
    """Fail loud if a parking ETF has no Dividend column: its distributions are most of its return."""
    missing_list = [s for s in PARKING_SYMBOL_TUPLE if (s, "Dividend") not in pricing_data_df.columns]
    if missing_list:
        raise RuntimeError(f"Parking ETFs without a Dividend column: {missing_list}; parked cash would earn ~0%.")
CASH_BUFFER_FRACTION_FLOAT = 0.01
BIL_REBALANCE_BAND_FRACTION_FLOAT = 0.01


@dataclass(frozen=True)
class ParkingOrderPlan:
    """Target whole shares for this decision; None = place no order for that asset."""

    spmo_target_share_int: int | None
    bil_target_share_int: int | None
    idle_value_float: float
    retarget_bool: bool


def _finite_positive(value_float: float) -> bool:
    return value_float is not None and np.isfinite(value_float) and value_float > 0.0


def plan_parking_orders(
    pod_value_float: float,
    stock_value_after_orders_float: float,
    spmo_held_share_int: int,
    bil_held_share_int: int,
    spmo_close_float: float,
    bil_close_float: float,
    gate_open_bool: bool,
    gate_switched_bool: bool,
    week_end_bool: bool,
    spmo_weight_float: float,
    cash_buffer_fraction_float: float = CASH_BUFFER_FRACTION_FLOAT,
    bil_band_fraction_float: float = BIL_REBALANCE_BAND_FRACTION_FLOAT,
) -> ParkingOrderPlan:
    """Parking targets after the stock orders of one decision close (pure function; see module formulas)."""
    if not _finite_positive(pod_value_float):
        raise ValueError("pod_value_float must be finite and positive.")
    if not np.isfinite(stock_value_after_orders_float):
        # A held stock without a finite Close_T: the idle value is unknown, so leave the parking untouched today
        # (neither park money that may not exist nor sell parking that may still be idle).
        return ParkingOrderPlan(None, None, float("nan"), False)
    idle_value_float = max(pod_value_float - stock_value_after_orders_float - cash_buffer_fraction_float * pod_value_float, 0.0)
    spmo_tradable_bool = _finite_positive(spmo_close_float)
    bil_tradable_bool = _finite_positive(bil_close_float)
    retarget_bool = bool(week_end_bool or gate_switched_bool)

    # ---- SPMO: zero while the gate is open; set on re-target closes while it is closed, untouched otherwise
    spmo_target_share_int: int | None = None
    if gate_open_bool or not spmo_tradable_bool or not (spmo_weight_float > 0.0):
        if spmo_held_share_int != 0:
            spmo_target_share_int = 0
    elif retarget_bool:
        spmo_target_share_int = int(np.floor(spmo_weight_float * idle_value_float / spmo_close_float))
        if spmo_target_share_int == spmo_held_share_int:
            spmo_target_share_int = None
    spmo_share_after_int = spmo_held_share_int if spmo_target_share_int is None else spmo_target_share_int
    spmo_value_after_float = spmo_share_after_int * spmo_close_float if spmo_tradable_bool else 0.0

    # ---- BIL: the rest of the idle value; bought only on re-target closes, sold whenever it is above target
    bil_target_share_int: int | None = None
    if bil_tradable_bool:
        bil_wanted_share_int = int(np.floor(max(idle_value_float - spmo_value_after_float, 0.0) / bil_close_float))
        off_target_value_float = abs(bil_wanted_share_int - bil_held_share_int) * bil_close_float
        if bil_wanted_share_int == 0 and bil_held_share_int > 0:
            bil_target_share_int = 0
        elif (
            bil_wanted_share_int != bil_held_share_int
            and off_target_value_float > bil_band_fraction_float * pod_value_float
            and (retarget_bool or bil_wanted_share_int < bil_held_share_int)
        ):
            bil_target_share_int = bil_wanted_share_int
    elif bil_held_share_int != 0:
        bil_target_share_int = 0
    return ParkingOrderPlan(spmo_target_share_int, bil_target_share_int, float(idle_value_float), retarget_bool)
