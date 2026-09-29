"""Daily replica with a cash log and a capital parameter (research only). Same accounting as the new-pod study's
simulator (engine parity at machine precision); the capital parameter serves the $25k small-account sensitivity."""

from __future__ import annotations

import numpy as np
import pandas as pd

from merger_arb_v2_20260927 import common
from trend_breakout_20260927.features import last_finite_close_before
from trend_breakout_20260927.policies import Intent, State
from trend_breakout_20260927.simulate import commission_float, ledger_target_shares_float

PHANTOM_RAW_SHARE_FLOAT = 1e-3
DIVIDEND_NET_RATE_FLOAT = common.trend_common.DIVIDEND_NET_RATE_FLOAT


def simulate(feature_obj, policy, slippage_float: float = common.ENGINE_SLIPPAGE_FLOAT, terminal_factor_float: float = 1.0,
             capital_float: float = common.CAPITAL_BASE_FLOAT, record_positions_bool: bool = False, phantom_cancel_bool: bool = False) -> dict:
    date_index = feature_obj.date_index
    open_arr, close_arr, unadj_arr, dividend_arr, adv_arr = feature_obj.open64(), feature_obj.close_arr, feature_obj.unadjusted_close(), feature_obj.dividend64(), feature_obj.adv20_dollar()
    date_count_int, symbol_count_int = close_arr.shape
    start_pos_int = int(date_index.searchsorted(common.TRADING_START_TS, side="left"))
    state = State(symbol_count_int, capital_float)
    cost_vec, proceeds_vec = np.zeros(symbol_count_int), np.zeros(symbol_count_int)
    total_vec, cash_vec, exposure_vec = np.full(date_count_int, np.nan), np.full(date_count_int, np.nan), np.full(date_count_int, np.nan)
    traded_vec, commission_vec = np.zeros(date_count_int), np.zeros(date_count_int)
    order_frac_list, trade_list, intent_log_list, position_log_list = [], [], [], []
    skipped_missing_close_int, phantom_fill_int = 0, 0
    previous_total_float = capital_float

    def log_intents(decision_pos_int: int, intent_list: list[Intent]) -> None:
        for intent in intent_list:
            intent_log_list.append({"decision_pos": int(decision_pos_int), "decision_ts": date_index[decision_pos_int], "symbol_idx": int(intent.symbol_idx), "kind": intent.kind_str,
                                    "amount": float(intent.amount_float), "reason": intent.reason_str, "stop_level": float(intent.stop_level_float)})

    def close_episode(symbol_idx: int, exit_pos_int: int, reason_str: str, price_float: float, stop_level_float: float) -> None:
        trade_list.append({"symbol_idx": int(symbol_idx), "entry_pos": int(state.entry_pos_vec[symbol_idx]), "exit_pos": int(exit_pos_int), "holding_sessions": int(exit_pos_int - state.entry_pos_vec[symbol_idx]),
                           "reason": reason_str, "cost": float(cost_vec[symbol_idx]), "proceeds": float(proceeds_vec[symbol_idx]), "pnl": float(proceeds_vec[symbol_idx] - cost_vec[symbol_idx]),
                           "exit_open": float(price_float), "stop_level": float(stop_level_float),
                           "fill_vs_stop": float(price_float / stop_level_float - 1.0) if np.isfinite(stop_level_float) and stop_level_float > 0 else float("nan")})
        state.shares_vec[symbol_idx] = 0.0
        state.hwm_vec[symbol_idx] = np.nan
        state.entry_pos_vec[symbol_idx] = -1
        cost_vec[symbol_idx] = 0.0
        proceeds_vec[symbol_idx] = 0.0

    pending_list = policy.decide(start_pos_int - 1, state)
    log_intents(start_pos_int - 1, pending_list)
    for pos_int in range(start_pos_int, date_count_int):
        held_idx_vec = np.flatnonzero(state.shares_vec != 0.0)
        if len(held_idx_vec) > 0:
            gross_vec = state.shares_vec[held_idx_vec] * np.nan_to_num(dividend_arr[pos_int - 1, held_idx_vec])
            state.cash_float += float(np.sum(np.where(gross_vec > 0, gross_vec * DIVIDEND_NET_RATE_FLOAT, gross_vec)))
        liquidated_set: set[int] = set()
        if len(held_idx_vec) > 0:
            missing_vec = ~np.isfinite(open_arr[pos_int, held_idx_vec]) | ~np.isfinite(close_arr[pos_int, held_idx_vec])
            for symbol_idx in held_idx_vec[missing_vec]:
                anchor_int, last_close_float = last_finite_close_before(close_arr, pos_int, int(symbol_idx))
                scale_float = unadj_arr[anchor_int, symbol_idx] / close_arr[anchor_int, symbol_idx]
                shares_float = state.shares_vec[symbol_idx]
                fee_float = commission_float(-shares_float, scale_float, True)
                proceeds_float = shares_float * last_close_float * terminal_factor_float
                state.cash_float += proceeds_float - fee_float
                traded_vec[pos_int] += abs(proceeds_float)
                commission_vec[pos_int] += fee_float
                proceeds_vec[symbol_idx] += proceeds_float
                close_episode(int(symbol_idx), pos_int, "terminal", last_close_float * terminal_factor_float, float("nan"))
                liquidated_set.add(int(symbol_idx))
        for intent in pending_list:
            symbol_idx = intent.symbol_idx
            if symbol_idx in liquidated_set:
                continue
            open_float = float(open_arr[pos_int, symbol_idx])
            if not np.isfinite(open_float):
                continue
            if intent.kind_str == "exit":
                delta_float = -state.shares_vec[symbol_idx]
            elif intent.kind_str == "target_pct":
                delta_float = ledger_target_shares_float(intent.amount_float * previous_total_float, unadj_arr[pos_int - 1, symbol_idx], close_arr[pos_int - 1, symbol_idx], True) - state.shares_vec[symbol_idx]
            elif intent.kind_str == "value":
                delta_float = ledger_target_shares_float(intent.amount_float, unadj_arr[pos_int - 1, symbol_idx], close_arr[pos_int - 1, symbol_idx], True)
            else:
                raise ValueError(intent.kind_str)
            if delta_float == 0.0:
                continue
            if not np.isfinite(close_arr[pos_int, symbol_idx]):
                skipped_missing_close_int += 1
                continue
            execution_scale_float = unadj_arr[pos_int, symbol_idx] / close_arr[pos_int, symbol_idx]
            if abs(delta_float) / execution_scale_float < PHANTOM_RAW_SHARE_FLOAT:
                phantom_fill_int += 1
                if phantom_cancel_bool:
                    continue
            price_float = open_float * (1.0 + np.sign(delta_float) * slippage_float)
            fee_float = commission_float(delta_float, execution_scale_float, True)
            state.cash_float -= delta_float * price_float + fee_float
            traded_vec[pos_int] += abs(delta_float * price_float)
            commission_vec[pos_int] += fee_float
            order_frac_list.append((pos_int, abs(delta_float * price_float) / previous_total_float, float(adv_arr[pos_int - 1, symbol_idx])))
            if delta_float > 0:
                if state.shares_vec[symbol_idx] == 0.0:
                    state.entry_pos_vec[symbol_idx] = pos_int
                    state.hwm_vec[symbol_idx] = np.nan
                cost_vec[symbol_idx] += delta_float * price_float
                state.shares_vec[symbol_idx] += delta_float
            else:
                proceeds_vec[symbol_idx] += -delta_float * price_float
                state.shares_vec[symbol_idx] += delta_float
                if state.shares_vec[symbol_idx] == 0.0:
                    close_episode(symbol_idx, pos_int, intent.reason_str, open_float, intent.stop_level_float)
        held_idx_vec = np.flatnonzero(state.shares_vec != 0.0)
        portfolio_value_float = float(np.dot(state.shares_vec[held_idx_vec], close_arr[pos_int, held_idx_vec])) if len(held_idx_vec) else 0.0
        total_float = state.cash_float + portfolio_value_float
        total_vec[pos_int], cash_vec[pos_int] = total_float, state.cash_float
        exposure_vec[pos_int] = portfolio_value_float / total_float if total_float > 0 else np.nan
        previous_total_float = total_float
        state.total_value_float = total_float
        if len(held_idx_vec) > 0:
            close_held_vec = close_arr[pos_int, held_idx_vec]
            state.hwm_vec[held_idx_vec] = np.where(np.isnan(state.hwm_vec[held_idx_vec]), close_held_vec, np.maximum(state.hwm_vec[held_idx_vec], close_held_vec))
        if record_positions_bool:
            position_log_list.append((pos_int, held_idx_vec.copy(), state.entry_pos_vec[held_idx_vec].copy()))
        if pos_int + 1 < date_count_int:
            pending_list = policy.decide(pos_int, state)
            log_intents(pos_int, pending_list)
        else:
            pending_list = []
    index = date_index[start_pos_int:]
    total_ser = pd.Series(total_vec[start_pos_int:], index=index, name="total_value")
    cash_ser = pd.Series(cash_vec[start_pos_int:], index=index, name="cash")
    return {"total_ser": total_ser, "return_ser": total_ser.pct_change().iloc[1:], "cash_weight_ser": (cash_ser.clip(lower=0.0) / total_ser).rename("cash_weight"), "cash_ser": cash_ser,
            "exposure_ser": pd.Series(exposure_vec[start_pos_int:], index=index), "traded_notional_ser": pd.Series(traded_vec[start_pos_int:], index=index),
            "commission_ser": pd.Series(commission_vec[start_pos_int:], index=index), "order_frac_arr": np.array(order_frac_list, dtype=np.float64).reshape(-1, 3),
            "trade_df": pd.DataFrame(trade_list), "intent_df": pd.DataFrame(intent_log_list), "position_log": position_log_list,
            "skipped_missing_close_int": skipped_missing_close_int, "phantom_fill_int": phantom_fill_int, "final_total_float": float(total_vec[-1]), "start_pos_int": start_pos_int,
            "capital_float": capital_float}
