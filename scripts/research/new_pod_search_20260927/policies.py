"""Decision policies for the new-pod search (research only): order intents at the close of t for fills at open t+1.

Intent kinds are the trend study's (exit -> order_target(asset, 0); target_pct -> order_target_percent; value ->
order_value), so the same intents are executed by the replica and replayed by the real engine.
"""

from __future__ import annotations

import numpy as np

import ndx_param_robustness_core as core  # noqa: E402

from new_pod_search_20260927.cells import MCell, SCell
from new_pod_search_20260927.features import PodFeatureBook
from trend_breakout_20260927.policies import Intent, State, rank_indices, raw_share_count_int


class PolicyM:
    """Merger-arbitrage candidates queued FIFO by confirmation date; K slots; break / time stops."""

    def __init__(self, feature_obj: PodFeatureBook, cell: MCell):
        self.f = feature_obj
        self.cell = cell
        conf_dict = feature_obj.confirmation(cell.jump_float, cell.theta_float, cell.window_int)
        self.conf_arr = conf_dict["conf"]
        self.pin_vol_arr = conf_dict["pin_vol"]
        self.pin_ref_arr = conf_dict["pin_ref"]
        self.event_pos_arr = conf_dict["event_pos"]
        self.unadj_arr = feature_obj.unadjusted_close()
        self.queue_list: list[tuple[int, float, int, int]] = []  # (confirmation pos, pin_vol, symbol rank, symbol idx)
        self.pin_ref_by_symbol_dict: dict[int, float] = {}
        self.event_pos_by_symbol_dict: dict[int, int] = {}
        self.dropped_zero_share_int = 0
        self.dropped_delisted_int = 0
        self.confirmations_int = 0
        self.admissions_int = 0

    def decide(self, pos_int: int, state: State) -> list[Intent]:
        close_vec = self.f.close_arr[pos_int]
        held_vec = state.held_mask()
        intent_list: list[Intent] = []
        exit_vec = np.zeros(len(close_vec), dtype=bool)
        # exits: break stop first, then the time stop (labels)
        for symbol_idx in np.flatnonzero(held_vec):
            pin_ref_float = self.pin_ref_by_symbol_dict.get(int(symbol_idx), np.nan)
            close_float = float(close_vec[symbol_idx])
            if self.cell.break_float is not None and np.isfinite(close_float) and np.isfinite(pin_ref_float) and close_float <= self.cell.break_float * pin_ref_float:
                intent_list.append(Intent(int(symbol_idx), "exit", 0.0, "break", float(self.cell.break_float * pin_ref_float)))
                exit_vec[symbol_idx] = True
            elif pos_int - int(state.entry_pos_vec[symbol_idx]) >= self.cell.time_stop_int - 1:
                # held for time_stop sessions (entry day counts as the first): sold at the next open
                intent_list.append(Intent(int(symbol_idx), "exit", 0.0, "time"))
                exit_vec[symbol_idx] = True
        # new confirmations at this close join the queue (one entry per symbol; held names are ignored)
        queued_set = {item[3] for item in self.queue_list}
        for symbol_idx in np.flatnonzero(self.conf_arr[pos_int]):
            symbol_idx = int(symbol_idx)
            self.confirmations_int += 1
            if held_vec[symbol_idx] or symbol_idx in queued_set:
                continue
            self.queue_list.append((pos_int, float(self.pin_vol_arr[pos_int, symbol_idx]), int(self.f.symbol_rank_vec[symbol_idx]), symbol_idx))
            self.pin_ref_by_symbol_dict[symbol_idx] = float(self.pin_ref_arr[pos_int, symbol_idx])
            self.event_pos_by_symbol_dict[symbol_idx] = int(self.event_pos_arr[pos_int, symbol_idx])
            queued_set.add(symbol_idx)
        # drop queued names whose price series has ended or that are somehow held
        kept_list = []
        for item in self.queue_list:
            if not np.isfinite(close_vec[item[3]]):
                self.dropped_delisted_int += 1
                continue
            if held_vec[item[3]]:
                continue
            kept_list.append(item)
        # *** CRITICAL *** FIFO by confirmation date, then lowest pin_vol, then symbol.
        self.queue_list = sorted(kept_list)
        free_int = self.cell.slots_int - int(held_vec.sum() - exit_vec.sum())
        admit_int = min(free_int, len(self.queue_list))
        if admit_int <= 0:
            return intent_list
        exiting_value_float = float(np.dot(state.shares_vec[exit_vec], close_vec[exit_vec])) if exit_vec.any() else 0.0
        budget_float = min(state.total_value_float / self.cell.slots_int, (state.cash_float + exiting_value_float) / admit_int)
        admitted_list = self.queue_list[:admit_int]
        self.queue_list = self.queue_list[admit_int:]
        if budget_float <= 0:
            return intent_list
        for item in admitted_list:
            symbol_idx = item[3]
            if raw_share_count_int(budget_float, float(self.unadj_arr[pos_int, symbol_idx])) >= 1:
                intent_list.append(Intent(symbol_idx, "value", budget_float, "entry"))
                self.admissions_int += 1
            else:
                self.dropped_zero_share_int += 1
        return intent_list


class PolicyS:
    """Seasonality top-N at each schedule decision: GATED (SPY > SMA200 else cash) or HEDGED (50% stocks + 50% SH)."""

    def __init__(self, feature_obj: PodFeatureBook, cell: SCell, score_arr: np.ndarray, sh_idx: int | None = None):
        self.f = feature_obj
        self.cell = cell
        self.score_arr = score_arr  # schedule rows (month-end index) x symbols
        self.schedule_dict = core.build_schedule(feature_obj.u, cell.offset_int)
        traded_row_vec = np.flatnonzero(self.schedule_dict["trade_vec"])
        decision_pos_vec = self.schedule_dict["decision_pos_vec"][traded_row_vec]
        self.row_by_pos_dict = {int(pos_int): int(row_int) for row_int, pos_int in zip(traded_row_vec, decision_pos_vec)}
        self.member_arr = feature_obj.u["member_arr"]
        self.rel25_arr = feature_obj.rel25_pass()
        self.regime_vec = feature_obj.regime_pass("SPY")
        self.sh_idx = sh_idx
        if cell.form_str == "HEDGED" and sh_idx is None:
            raise ValueError("HEDGED form needs the SH symbol index")
        self.selection_log_dict: dict[int, list[int]] = {}

    def decide(self, pos_int: int, state: State) -> list[Intent]:
        if pos_int not in self.row_by_pos_dict:
            return []
        row_int = self.row_by_pos_dict[pos_int]
        held_vec = state.held_mask()
        hedged_bool = self.cell.form_str == "HEDGED"
        if hedged_bool and not np.isfinite(self.f.close_arr[pos_int, self.sh_idx]):
            return []  # SH does not exist yet: nothing is traded (HEDGED cells start 2006-07-03)
        if not hedged_bool and not self.regime_vec[pos_int]:
            self.selection_log_dict[pos_int] = []
            return [Intent(int(symbol_idx), "exit", 0.0, "gate") for symbol_idx in np.flatnonzero(held_vec)]
        score_vec = self.score_arr[row_int]
        eligible_vec = (self.member_arr[pos_int] == 1) & self.rel25_arr[pos_int] & np.isfinite(score_vec)
        if self.sh_idx is not None:
            eligible_vec[self.sh_idx] = False
        selected_vec = rank_indices(score_vec, eligible_vec, self.f.symbol_rank_vec)[: self.cell.n_int]
        selected_set = set(int(i) for i in selected_vec)
        self.selection_log_dict[pos_int] = sorted(selected_set)
        stock_weight_float = (0.5 if hedged_bool else 1.0) / self.cell.n_int
        intent_list: list[Intent] = []
        for symbol_idx in np.flatnonzero(held_vec):
            symbol_idx = int(symbol_idx)
            if symbol_idx in selected_set or (hedged_bool and symbol_idx == self.sh_idx):
                continue
            intent_list.append(Intent(symbol_idx, "exit", 0.0, "rebalance"))
        for symbol_idx in selected_vec:
            intent_list.append(Intent(int(symbol_idx), "target_pct", stock_weight_float, "rebalance"))
        if hedged_bool:
            intent_list.append(Intent(int(self.sh_idx), "target_pct", 0.5, "hedge"))
        return intent_list
