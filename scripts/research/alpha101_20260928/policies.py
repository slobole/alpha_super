"""The Alpha101 pod policy (PREREG section 7; research only): order intents at the close of t for fills at open t+1.

Rule at each close t on members with a valid composite score (rank 1 = best score; ties by symbol):
    exit   a held name whose rank is worse than B x N, that is no longer a member, or whose score is missing
    enter  free slots are filled with the best-ranked names not held that rank within the top N
    size   b = min(V_t / N, (C_t + P_t) / n_t); C_t cash, P_t the exits' value at the close of t, n_t free slots;
           raw shares trunc(b / U_t) (a `value` intent); no re-sizing; terminal liquidation by the simulator
HEDGED form: stock budget b = min(0.5 x V_t / N, available cash / n_t) with the SH re-size cash flow deducted from the
available cash; SH held long at 50% of NAV and re-sized to 50% at each month-end close (`target_pct` intent); the policy
is inactive before the first month-end close at which SH has a price (first fills 2006-07-03).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from trend_breakout_20260927.policies import Intent, State, rank_indices, raw_share_count_int


class PolicyAlpha:
    def __init__(self, feature_obj, score_arr: np.ndarray, n_int: int, b_int: int, hedged_bool: bool = False, sh_idx: int | None = None):
        self.f = feature_obj
        self.score_arr = score_arr  # (T, S) composite score, NaN = missing
        self.n_int = int(n_int)
        self.b_int = int(b_int)
        self.hedged_bool = bool(hedged_bool)
        self.sh_idx = sh_idx
        if hedged_bool and sh_idx is None:
            raise ValueError("HEDGED form needs the SH symbol index")
        self.member_arr = feature_obj.u["member_arr"]
        self.unadj_arr = feature_obj.unadjusted_close()
        self.close_arr = feature_obj.close_arr
        self.month_end_pos_set: set[int] = set()
        if hedged_bool:
            month_end_index = pd.DatetimeIndex(feature_obj.u["month_end_index"])
            self.month_end_pos_set = set(int(p) for p in feature_obj.date_index.get_indexer(month_end_index) if p >= 0)
        self.active_bool = not hedged_bool
        self.decisions_int = 0
        self.no_slot_int = 0
        self.zero_share_skips_int = 0

    def decide(self, pos_int: int, state: State) -> list[Intent]:
        if pos_int < 0:
            return []
        close_vec = self.close_arr[pos_int]
        held_vec = state.held_mask()
        intent_list: list[Intent] = []
        sh_cash_flow_float = 0.0
        if self.hedged_bool:
            held_vec = held_vec.copy()
            held_vec[self.sh_idx] = False  # SH is never a stock slot
            sh_close_float = float(close_vec[self.sh_idx])
            if not self.active_bool:
                if pos_int in self.month_end_pos_set and np.isfinite(sh_close_float):
                    self.active_bool = True
                else:
                    return []
            if pos_int in self.month_end_pos_set and np.isfinite(sh_close_float):
                intent_list.append(Intent(int(self.sh_idx), "target_pct", 0.5, "hedge"))
                sh_cash_flow_float = 0.5 * state.total_value_float - float(state.shares_vec[self.sh_idx] * sh_close_float)
        score_vec = self.score_arr[pos_int]
        member_vec = self.member_arr[pos_int] == 1
        if self.sh_idx is not None:
            member_vec = member_vec.copy()
            member_vec[self.sh_idx] = False
        eligible_vec = member_vec & np.isfinite(score_vec)
        ranked_vec = rank_indices(score_vec, eligible_vec, self.f.symbol_rank_vec)  # best first, ties by symbol
        rank_vec = np.full(len(score_vec), np.inf)
        rank_vec[ranked_vec] = np.arange(1, len(ranked_vec) + 1)
        self.decisions_int += 1
        # exits
        exit_vec = held_vec & (~eligible_vec | (rank_vec > self.b_int * self.n_int))
        for symbol_idx in np.flatnonzero(exit_vec):
            reason_str = "member" if not member_vec[symbol_idx] else ("missing" if not np.isfinite(score_vec[symbol_idx]) else "buffer")
            intent_list.append(Intent(int(symbol_idx), "exit", 0.0, reason_str))
        # entries
        free_int = self.n_int - int(held_vec.sum() - exit_vec.sum())
        if free_int <= 0:
            return intent_list
        top_vec = ranked_vec[: self.n_int]
        candidate_vec = top_vec[~held_vec[top_vec]]
        admit_int = min(free_int, len(candidate_vec))
        if admit_int <= 0:
            self.no_slot_int += 1
            return intent_list
        exiting_value_float = float(np.dot(state.shares_vec[exit_vec], np.nan_to_num(close_vec[exit_vec]))) if exit_vec.any() else 0.0
        available_float = state.cash_float + exiting_value_float - sh_cash_flow_float
        slot_float = (0.5 if self.hedged_bool else 1.0) * state.total_value_float / self.n_int
        budget_float = min(slot_float, available_float / free_int)
        if budget_float <= 0:
            return intent_list
        for symbol_idx in candidate_vec[:admit_int]:
            if raw_share_count_int(budget_float, float(self.unadj_arr[pos_int, symbol_idx])) >= 1:
                intent_list.append(Intent(int(symbol_idx), "value", budget_float, "entry"))
            else:
                self.zero_share_skips_int += 1
        return intent_list
