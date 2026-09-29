"""Family V policy: confirmed takeover candidates in a persistent FIFO queue with drop rules; K slots; break / time
stops (PREREG section 4). Intents are the trend study's (exit, target_pct, value)."""

from __future__ import annotations

import numpy as np

from merger_arb_v2_20260927.cells import QUEUE_MAX_AGE_INT, TIME_STOP_INT, VCell
from merger_arb_v2_20260927.features import V2FeatureBook
from trend_breakout_20260927.policies import Intent, State, raw_share_count_int


class PolicyV:
    def __init__(self, feature_obj: V2FeatureBook, cell: VCell):
        self.f = feature_obj
        self.cell = cell
        conf_dict = feature_obj.confirmation(cell.jump_float, cell.theta_float, cell.window_int, cell.half_str)
        self.conf_arr = conf_dict["conf"]
        # sparse confirmations grouped by decision row: t -> list of (symbol, pin_stat, pin_ref, event_pos)
        self.conf_by_t_dict: dict[int, list[tuple[int, float, float, int]]] = {}
        for t_int, symbol_idx, pin_stat_float, pin_ref_float, event_pos_int in zip(conf_dict["t_vec"], conf_dict["symbol_vec"], conf_dict["pin_stat_vec"], conf_dict["pin_ref_vec"], conf_dict["event_pos_vec"]):
            self.conf_by_t_dict.setdefault(int(t_int), []).append((int(symbol_idx), float(pin_stat_float), float(pin_ref_float), int(event_pos_int)))
        self.unadj_arr = feature_obj.unadjusted_close()
        # queue items: (confirmation pos, pin_stat, symbol rank, symbol idx, pin_ref)
        self.queue_list: list[tuple[int, float, int, int, float]] = []
        self.pin_ref_by_symbol_dict: dict[int, float] = {}
        self.event_pos_by_symbol_dict: dict[int, int] = {}
        self.confirmations_int = 0
        self.admissions_int = 0
        self.dropped_dict = {"delisted": 0, "break_while_queued": 0, "aged_out": 0, "zero_share": 0}

    def decide(self, pos_int: int, state: State) -> list[Intent]:
        close_vec = self.f.close_arr[pos_int]
        held_vec = state.held_mask()
        intent_list: list[Intent] = []
        exit_vec = np.zeros(len(close_vec), dtype=bool)
        for symbol_idx in np.flatnonzero(held_vec):
            symbol_idx = int(symbol_idx)
            pin_ref_float = self.pin_ref_by_symbol_dict.get(symbol_idx, np.nan)
            close_float = float(close_vec[symbol_idx])
            if self.cell.break_float is not None and np.isfinite(close_float) and np.isfinite(pin_ref_float) and close_float <= self.cell.break_float * pin_ref_float:
                intent_list.append(Intent(symbol_idx, "exit", 0.0, "break", float(self.cell.break_float * pin_ref_float)))
                exit_vec[symbol_idx] = True
            elif pos_int - int(state.entry_pos_vec[symbol_idx]) >= TIME_STOP_INT - 1:
                intent_list.append(Intent(symbol_idx, "exit", 0.0, "time"))
                exit_vec[symbol_idx] = True
        # new confirmations join the queue (held names and names already queued are ignored)
        queued_set = {item[3] for item in self.queue_list}
        for symbol_idx, pin_stat_float, pin_ref_float, event_pos_int in self.conf_by_t_dict.get(pos_int, []):
            self.confirmations_int += 1
            if held_vec[symbol_idx] or symbol_idx in queued_set:
                continue
            self.queue_list.append((pos_int, pin_stat_float, int(self.f.symbol_rank_vec[symbol_idx]), symbol_idx, pin_ref_float))
            self.event_pos_by_symbol_dict[symbol_idx] = event_pos_int
            queued_set.add(symbol_idx)
        # *** CRITICAL *** drop rules, checked at every close while queued: series ended; close <= 0.95 x pin_ref;
        # 60 sessions since confirmation.
        kept_list = []
        for item in self.queue_list:
            conf_pos_int, _, _, symbol_idx, pin_ref_float = item
            close_float = float(close_vec[symbol_idx])
            if not np.isfinite(close_float):
                self.dropped_dict["delisted"] += 1
                continue
            if close_float <= 0.95 * pin_ref_float:
                self.dropped_dict["break_while_queued"] += 1
                continue
            if pos_int - conf_pos_int >= QUEUE_MAX_AGE_INT:
                self.dropped_dict["aged_out"] += 1
                continue
            if held_vec[symbol_idx]:
                continue
            kept_list.append(item)
        self.queue_list = sorted(kept_list)  # confirmation date, then median |r|, then symbol
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
                self.pin_ref_by_symbol_dict[symbol_idx] = item[4]
                self.admissions_int += 1
            else:
                self.dropped_dict["zero_share"] += 1
        return intent_list
