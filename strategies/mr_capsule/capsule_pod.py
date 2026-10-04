"""Shared gate-and-parking behaviour of the two MR capsule pods (research-only mixin).

The mixin adds, to an existing stock-reversal pod:
- the shared VIX stress gate (strategies/mr_capsule/vix_stress_gate.py), read at the decision close;
- the SPMO / BIL parking of idle cash (strategies/mr_capsule/parking.py), placed after the pod's stock orders.
The pod keeps its own stock rules; the subclass only filters the parking symbols out of its slot and exit loops and
skips entries while the gate is closed. Trade statistics and exposure time are reported on stock trades only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from alpha.engine.metrics import generate_trades_metrics
from strategies.mr_capsule.parking import PARKING_SYMBOL_TUPLE, plan_parking_orders
from strategies.mr_capsule.vix_stress_gate import (
    BIL_SYMBOL_STR,
    GATE_MEMORY_SESSION_INT,
    SPMO_SYMBOL_STR,
    SPMO_VOL_TARGET_FLOAT,
    gate_state_at,
    spmo_weight_from_history,
    stress_gate_open_ser,
)

PARKING_TRADE_ID_BASE_INT = 900_000_000


class CapsulePodMixin:
    """Gate + parking state; call _prepare_capsule_state in compute_signals and _place_parking_orders in iterate."""

    vix_close_ser: pd.Series | None = None
    parking_enabled_bool: bool = True
    spmo_parking_enabled_bool: bool = True  # False: the research reference "BIL only" (idle cash all in BIL)
    gate_override_ser: pd.Series | None = None  # tests only: replaces the VIX gate

    def _prepare_capsule_state(self) -> None:
        # The gate depends on the VIX history only (not on the pricing frame), so a truncated signal-audit
        # recompute of compute_signals rebuilds the same gate; nothing here is derived from the pricing frame.
        if self.gate_override_ser is not None:
            self.gate_open_ser = pd.Series(self.gate_override_ser, copy=True).astype(bool).sort_index()
        else:
            if self.vix_close_ser is None:
                raise RuntimeError("The MR capsule gate needs vix_close_ser (load_vix_close_ser) before the run.")
            self.gate_open_ser = stress_gate_open_ser(self.vix_close_ser)
        self._last_gate_open_bool: bool | None = None
        self._parking_trade_id_map: dict[str, int] = {}
        self._parking_trade_counter_int = PARKING_TRADE_ID_BASE_INT
        self._parking_order_count_dict = {SPMO_SYMBOL_STR: 0, BIL_SYMBOL_STR: 0}
        self._gate_closed_free_slot_session_int = 0
        self._accounting_policy_dict.update(
            {
                "capsule_gate_str": f"VIX > expanding mean since 1990 (min 500); open >= {GATE_MEMORY_SESSION_INT} sessions",
                "capsule_parking_str": (
                    "disabled (cash at 0%)"
                    if not self.parking_enabled_bool
                    else "BIL only (bought weekly / at gate switches, sold when needed)"
                    if not self.spmo_parking_enabled_bool
                    else f"gate closed: SPMO at min(1, {SPMO_VOL_TARGET_FLOAT:.0%} / 20d vol) if traded on each of the last 20 "
                    "sessions, set weekly and at gate switches; rest BIL (bought weekly / at switches, sold when "
                    "needed); gate open: BIL"
                ),
                "parking_dividend_withholding_note_str": "BIL/SPMO dividends use the house 25% withholding (conservative for BIL, G-029)",
            }
        )

    def _stock_position_ser(self) -> pd.Series:
        position_ser = self.get_positions()
        position_ser = position_ser[position_ser > 0]
        return position_ser.drop(labels=[s for s in PARKING_SYMBOL_TUPLE if s in position_ser.index])

    def _gate_open_at_decision(self) -> bool:
        # *** CRITICAL*** the decision is taken after Close(previous_bar); read the gate state of that close only.
        return gate_state_at(self.gate_open_ser, pd.Timestamp(self.previous_bar))

    def _parking_trade_id(self, symbol_str: str, target_share_int: int) -> int:
        held_share_int = int(self.get_position(symbol_str))
        if held_share_int == 0 and target_share_int > 0:
            self._parking_trade_counter_int += 1
            self._parking_trade_id_map[symbol_str] = self._parking_trade_counter_int
        return self._parking_trade_id_map.get(symbol_str, PARKING_TRADE_ID_BASE_INT)

    def _place_parking_orders(
        self, data_df: pd.DataFrame, close_row_ser: pd.Series, stock_value_after_orders_float: float, gate_open_bool: bool
    ) -> None:
        previous_gate_open_bool = self._last_gate_open_bool
        self._last_gate_open_bool = gate_open_bool
        if not self.parking_enabled_bool:
            return
        decision_ts, execution_ts = pd.Timestamp(self.previous_bar), pd.Timestamp(self.current_bar)
        # *** CRITICAL*** "last session of the ISO week" is read from the trading calendar (the execution session is
        # known in advance), never from prices after Close(previous_bar).
        week_end_bool = tuple(decision_ts.isocalendar())[:2] != tuple(execution_ts.isocalendar())[:2]
        # The first decision of the run re-targets too, so the parking is deployed without waiting for a week end.
        gate_switched_bool = previous_gate_open_bool is None or previous_gate_open_bool != gate_open_bool
        spmo_close_key, spmo_volume_key = (SPMO_SYMBOL_STR, "Close"), (SPMO_SYMBOL_STR, "Volume")
        spmo_weight_float = 0.0
        if self.spmo_parking_enabled_bool and spmo_close_key in data_df.columns and spmo_volume_key in data_df.columns:
            # *** CRITICAL*** data_df ends at Close(previous_bar): the weight uses closes and volumes up to T only.
            spmo_weight_float = spmo_weight_from_history(data_df[spmo_close_key], data_df[spmo_volume_key])
        plan = plan_parking_orders(
            pod_value_float=float(self.previous_total_value),
            stock_value_after_orders_float=float(stock_value_after_orders_float),
            spmo_held_share_int=int(self.get_position(SPMO_SYMBOL_STR)),
            bil_held_share_int=int(self.get_position(BIL_SYMBOL_STR)),
            spmo_close_float=float(close_row_ser.get((SPMO_SYMBOL_STR, "Close"), np.nan)),
            bil_close_float=float(close_row_ser.get((BIL_SYMBOL_STR, "Close"), np.nan)),
            gate_open_bool=gate_open_bool,
            gate_switched_bool=gate_switched_bool,
            week_end_bool=week_end_bool,
            spmo_weight_float=spmo_weight_float,
        )
        for symbol_str, target_share_int in ((SPMO_SYMBOL_STR, plan.spmo_target_share_int), (BIL_SYMBOL_STR, plan.bil_target_share_int)):
            if target_share_int is None:
                continue
            self.order_target(symbol_str, int(target_share_int), trade_id=self._parking_trade_id(symbol_str, int(target_share_int)))
            self._parking_order_count_dict[symbol_str] += 1

    def _record_capsule_diagnostics(self) -> None:
        diagnostic_dict = {
            "parking_order_count_dict": dict(self._parking_order_count_dict),
            "gate_closed_free_slot_session_count_int": int(self._gate_closed_free_slot_session_int),
        }
        transaction_df = self.get_transactions()
        if len(transaction_df) > 0 and len(self.results) > 0:
            parking_df = transaction_df[transaction_df["asset"].isin(PARKING_SYMBOL_TUPLE)]
            year_float = len(self.results) / 252.0
            mean_value_float = float(self.results["total_value"].astype(float).mean())
            traded_value_ser = (parking_df["amount"].abs() * parking_df["price"]).groupby(parking_df["asset"]).sum()
            diagnostic_dict["parking_one_way_turnover_per_year_dict"] = {
                str(k): float(v / mean_value_float / year_float) for k, v in traded_value_ser.items()
            }
            diagnostic_dict["parking_commission_total_dict"] = {
                str(k): float(v) for k, v in parking_df.groupby("asset")["commission"].sum().items()
            }
        self._accounting_policy_dict.update(diagnostic_dict)

    def summarize(self, include_benchmarks=True):
        super().summarize(include_benchmarks=include_benchmarks)
        trade_df = self._trades
        if trade_df is None or len(trade_df) == 0:
            return
        # Trade statistics and exposure on stock trades only: SPMO / BIL parking trades (ids from
        # PARKING_TRADE_ID_BASE_INT) would otherwise dominate trade counts, holding periods and exposure time.
        # NAV, returns and costs keep every trade.
        stock_trade_df = trade_df[np.asarray(trade_df.index, dtype=float) < PARKING_TRADE_ID_BASE_INT]
        self.summary_trades = generate_trades_metrics(stock_trade_df, self.results.index)
        calendar_idx = pd.to_datetime(self.results.index)
        event_arr = np.zeros(len(calendar_idx) + 1)
        np.add.at(event_arr, calendar_idx.searchsorted(pd.to_datetime(stock_trade_df["start"])), 1.0)
        np.add.at(event_arr, calendar_idx.searchsorted(pd.to_datetime(stock_trade_df["end"]), side="right"), -1.0)
        exposed_session_int = int((np.cumsum(event_arr)[:-1] > 0).sum())
        if "Exposure Time [%]" in self.summary.index and "Strategy" in self.summary.columns:
            self.summary.loc["Exposure Time [%]", "Strategy"] = exposed_session_int / len(calendar_idx) * 100
