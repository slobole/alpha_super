"""Decision policies: order intents at the decision close t for fills at the next open (research only).

A policy sees features up to and including the close of t and the replica's own state (ledger shares, highest close
since entry, cash, total value). It returns order intents; the simulator (simulate.py) and the real engine (through the
research subclass in engine_parity.py) execute the same intents, so the selection / stop logic is one code path.

Intent kinds (the three engine calls the parity subclass makes):
    exit        -> order_target(asset, 0)                  sell the whole position
    target_pct  -> order_target_percent(asset, w)          trunc(w x V_t / U_t) raw shares, mapped to ledger units
    value       -> order_value(asset, D)                   trunc(D / U_t) raw shares, mapped to ledger units
"""

from __future__ import annotations

import dataclasses

import numpy as np

import ndx_param_robustness_core as core  # noqa: E402

from trend_breakout_20260927.cells import ACell, BCell, StopSpec
from trend_breakout_20260927.features import DailyFeatureBook

L_SLOT_COUNT_INT = 10
L_STOCK_FILTER_INT = 100
L_REGIME_STR = "SPY"
L_VXN_TARGET_FLOAT = 22.0
L_VXN_FLOOR_FLOAT = 0.25


@dataclasses.dataclass
class Intent:
    symbol_idx: int
    kind_str: str  # exit | target_pct | value
    amount_float: float  # weight (target_pct), dollars (value), 0 (exit)
    reason_str: str  # rebalance | stop | refill | entry | regime
    stop_level_float: float = float("nan")


class State:
    """Replica state visible to a policy at the close of t."""

    def __init__(self, symbol_count_int: int, capital_float: float):
        self.shares_vec = np.zeros(symbol_count_int)  # ledger (adjusted) units
        self.hwm_vec = np.full(symbol_count_int, np.nan)  # highest close since entry
        self.entry_pos_vec = np.full(symbol_count_int, -1, dtype=np.int64)
        self.cash_float = float(capital_float)
        self.total_value_float = float(capital_float)

    def held_mask(self) -> np.ndarray:
        return self.shares_vec > 0


def rank_indices(score_vec: np.ndarray, eligible_vec: np.ndarray, symbol_rank_vec: np.ndarray) -> np.ndarray:
    """Eligible indices ordered by score descending, then symbol ascending (the repo's mergesort order)."""
    idx_vec = np.flatnonzero(eligible_vec & np.isfinite(score_vec))
    order_vec = np.lexsort((symbol_rank_vec[idx_vec], -score_vec[idx_vec]))
    return idx_vec[order_vec]


def raw_share_count_int(dollars_float: float, unadjusted_close_float: float) -> int:
    """Whole raw shares the engine would buy for D dollars at the decision close (int truncates toward zero)."""
    if not np.isfinite(unadjusted_close_float) or unadjusted_close_float <= 0.0 or not np.isfinite(dollars_float):
        return 0
    return int(dollars_float / unadjusted_close_float)


# ----------------------------------------------------------------------------------------------------------------------
# family A: the live pod L with a per-position stop
# ----------------------------------------------------------------------------------------------------------------------
class PolicyA:
    """L's month-end selection (score ROC12 / ATR20$, SMA100, SPY gate, top 10, VXN scale) plus a daily stop."""

    def __init__(self, feature_obj: DailyFeatureBook, cell: ACell):
        self.f = feature_obj
        self.cell = cell
        self.schedule_dict = core.build_schedule(feature_obj.u, cell.offset_int)
        l_cell = dataclasses.replace(core.L_CELL, offset_int=cell.offset_int)
        # *** CRITICAL *** the monthly L score (ROC12 over decision-day dollar ATR) on the (possibly shifted) schedule.
        self.score_arr = core.score_table(feature_obj, self.schedule_dict, l_cell)
        traded_row_vec = np.flatnonzero(self.schedule_dict["trade_vec"])
        decision_pos_vec = self.schedule_dict["decision_pos_vec"][traded_row_vec]
        self.row_by_pos_dict = {int(pos_int): int(row_int) for row_int, pos_int in zip(traded_row_vec, decision_pos_vec)}
        scale_vec = feature_obj.vxn_scale_vec(decision_pos_vec, L_VXN_TARGET_FLOAT, L_VXN_FLOOR_FLOAT)
        self.scale_by_pos_dict = {int(pos_int): float(scale_float) for pos_int, scale_float in zip(decision_pos_vec, scale_vec)}
        self.trend_arr = feature_obj.trend_pass(L_STOCK_FILTER_INT)
        self.regime_vec = feature_obj.regime_pass(L_REGIME_STR)
        self.member_arr = feature_obj.u["member_arr"]
        self.atr_arr = feature_obj.atr_cs(20)
        self.unadj_arr = feature_obj.unadjusted_close()
        self.daily_score_arr = feature_obj.daily_l_score(self.schedule_dict) if cell.policy_str == "REFILL" else None
        self.current_scale_float = float("nan")  # VXN scale of the current month's decision (s_m)
        self.selection_log_dict: dict[int, list[int]] = {}

    def decide(self, pos_int: int, state: State) -> list[Intent]:
        close_vec = self.f.close_arr[pos_int]
        held_vec = state.held_mask()
        atr_vec = self.atr_arr[pos_int]
        fires_vec = held_vec & self.cell.stop.fires_vec(close_vec, state.hwm_vec, atr_vec)
        level_vec = self.cell.stop.stop_level_vec(state.hwm_vec, atr_vec)
        intent_list: list[Intent] = []

        if pos_int in self.row_by_pos_dict:
            # month-end decision: a name that fires at this close is excluded from this decision only
            row_int = self.row_by_pos_dict[pos_int]
            scale_float = self.scale_by_pos_dict[pos_int]
            self.current_scale_float = scale_float
            score_vec = self.score_arr[row_int]
            eligible_vec = (self.member_arr[pos_int] == 1) & self.trend_arr[pos_int] & np.isfinite(score_vec) & ~fires_vec
            if not self.regime_vec[pos_int]:
                eligible_vec[:] = False
            selected_vec = rank_indices(score_vec, eligible_vec, self.f.symbol_rank_vec)[:L_SLOT_COUNT_INT]
            selected_set = set(int(i) for i in selected_vec)
            self.selection_log_dict[pos_int] = sorted(selected_set)
            weight_float = scale_float / L_SLOT_COUNT_INT
            for symbol_idx in np.flatnonzero(held_vec):
                if int(symbol_idx) not in selected_set:
                    reason_str = "stop" if fires_vec[symbol_idx] else "rebalance"
                    intent_list.append(Intent(int(symbol_idx), "exit", 0.0, reason_str, float(level_vec[symbol_idx]) if fires_vec[symbol_idx] else float("nan")))
            for symbol_idx in selected_vec:
                intent_list.append(Intent(int(symbol_idx), "target_pct", weight_float, "rebalance"))
            return intent_list

        fired_idx_vec = np.flatnonzero(fires_vec)
        for symbol_idx in fired_idx_vec:
            intent_list.append(Intent(int(symbol_idx), "exit", 0.0, "stop", float(level_vec[symbol_idx])))
        if self.cell.policy_str == "REFILL" and len(fired_idx_vec) > 0 and self.regime_vec[pos_int] and np.isfinite(self.current_scale_float):
            daily_score_vec = self.daily_score_arr[pos_int]
            candidate_vec = (self.member_arr[pos_int] == 1) & self.trend_arr[pos_int] & np.isfinite(daily_score_vec) & ~held_vec
            ranked_vec = rank_indices(daily_score_vec, candidate_vec, self.f.symbol_rank_vec)
            refill_count_int = min(len(fired_idx_vec), len(ranked_vec))
            if refill_count_int > 0:
                exiting_value_float = float(np.dot(state.shares_vec[fired_idx_vec], close_vec[fired_idx_vec]))
                slot_float = state.total_value_float * self.current_scale_float / L_SLOT_COUNT_INT
                budget_float = min(slot_float, (state.cash_float + exiting_value_float) / refill_count_int)
                if budget_float > 0:
                    for symbol_idx in ranked_vec[:refill_count_int]:
                        if raw_share_count_int(budget_float, float(self.unadj_arr[pos_int, symbol_idx])) >= 1:
                            intent_list.append(Intent(int(symbol_idx), "value", budget_float, "refill"))
        return intent_list


# ----------------------------------------------------------------------------------------------------------------------
# monthly target lists from the 26 Sep replica (family C, the L / A0 legs on other universes)
# ----------------------------------------------------------------------------------------------------------------------
class PolicyMonthlyTargets:
    """Replays core.build_target_list targets: exits for names leaving, target weights for the selected names."""

    def __init__(self, target_list: list[dict]):
        self.target_by_pos_dict = {int(target_dict["decision_pos"]): target_dict for target_dict in target_list}

    def decide(self, pos_int: int, state: State) -> list[Intent]:
        target_dict = self.target_by_pos_dict.get(pos_int)
        if target_dict is None:
            return []
        selected_set = set(int(i) for i in target_dict["symbol_idx_vec"])
        intent_list = [
            Intent(int(symbol_idx), "exit", 0.0, "rebalance")
            for symbol_idx in np.flatnonzero(state.held_mask())
            if int(symbol_idx) not in selected_set
        ]
        for symbol_idx, weight_float in zip(target_dict["symbol_idx_vec"], target_dict["weight_vec"]):
            intent_list.append(Intent(int(symbol_idx), "target_pct", float(weight_float), "rebalance"))
        return intent_list


# ----------------------------------------------------------------------------------------------------------------------
# family B: daily breakout entries, chandelier exits
# ----------------------------------------------------------------------------------------------------------------------
class PolicyB:
    def __init__(self, feature_obj: DailyFeatureBook, cell: BCell):
        self.f = feature_obj
        self.cell = cell
        self.member_arr = feature_obj.u["member_arr"]
        self.rel25_arr = feature_obj.rel25_pass()
        self.sma200_arr = feature_obj.trend_pass(200)
        self.breakout_arr = feature_obj.breakout_pass(cell.n_int)
        self.rank_arr = feature_obj.rank_score(cell.rank_str)
        self.regime_vec = feature_obj.regime_pass("SPY")
        self.atr_arr = feature_obj.atr_cs(20)
        self.unadj_arr = feature_obj.unadjusted_close()
        self.scale_vec = feature_obj.vxn_scale_all() if cell.vxn_bool else None

    def decide(self, pos_int: int, state: State) -> list[Intent]:
        close_vec = self.f.close_arr[pos_int]
        held_vec = state.held_mask()
        atr_vec = self.atr_arr[pos_int]
        fires_vec = held_vec & self.cell.stop.fires_vec(close_vec, state.hwm_vec, atr_vec)
        level_vec = self.cell.stop.stop_level_vec(state.hwm_vec, atr_vec)
        regime_bool = bool(self.regime_vec[pos_int])
        exit_vec = fires_vec.copy()
        if self.cell.regime_exit_bool and not regime_bool:
            exit_vec |= held_vec
        intent_list = [
            Intent(int(symbol_idx), "exit", 0.0, "stop" if fires_vec[symbol_idx] else "regime", float(level_vec[symbol_idx]) if fires_vec[symbol_idx] else float("nan"))
            for symbol_idx in np.flatnonzero(exit_vec)
        ]
        if not regime_bool:
            return intent_list
        # *** CRITICAL *** every entry condition is read at the close of pos_int; the fill is the next open.
        signal_vec = (
            (self.member_arr[pos_int] == 1)
            & self.rel25_arr[pos_int]
            & self.sma200_arr[pos_int]
            & self.breakout_arr[pos_int]
            & np.isfinite(self.rank_arr[pos_int])
            & ~held_vec
        )
        free_int = self.cell.slots_int - int(held_vec.sum() - exit_vec.sum())
        admit_int = min(free_int, int(signal_vec.sum()))
        if admit_int <= 0:
            return intent_list
        ranked_vec = rank_indices(self.rank_arr[pos_int], signal_vec, self.f.symbol_rank_vec)[:admit_int]
        exiting_value_float = float(np.dot(state.shares_vec[exit_vec], close_vec[exit_vec])) if exit_vec.any() else 0.0
        slot_float = state.total_value_float / self.cell.slots_int
        if self.scale_vec is not None:
            slot_float *= float(self.scale_vec[pos_int])
        budget_float = min(slot_float, (state.cash_float + exiting_value_float) / admit_int)
        if budget_float <= 0:
            return intent_list
        for symbol_idx in ranked_vec:
            if raw_share_count_int(budget_float, float(self.unadj_arr[pos_int, symbol_idx])) >= 1:
                intent_list.append(Intent(int(symbol_idx), "value", budget_float, "entry"))
        return intent_list
