"""Daily replica of alpha/engine/strategy.py with a cash log and a capital-base parameter (research only).

Verbatim accounting of scripts/research/new_pod_search_20260927/simulate.py (engine parity at machine precision in
the 27 Sep studies), copied only to add `capital_float` for the $10k / $25k / $1M label runs (PREREG section 7). Per
session p (the decision was taken at the close of p - 1):
    1. dividend cash: 75% of shares x Dividend(p - 1) for names held before the open
    2. a held name without an Open or a Close on p is liquidated at its last close before p (commission, no slippage)
    3. the intents decided at the close of p - 1 fill at Open_p x (1 +/- slippage): exit / target_pct / value; a zero
       delta is cancelled without commission; commission = max($1, $0.005 x raw-equivalent shares)
    4. V_p = cash + sum shares x Close_p
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from alpha101_20260928 import common
from trend_breakout_20260927.features import last_finite_close_before
from trend_breakout_20260927.policies import Intent, State
from trend_breakout_20260927.simulate import capacity_stats, commission_float, ledger_target_shares_float  # noqa: F401

PHANTOM_RAW_SHARE_FLOAT = 1e-3


def simulate(
    feature_obj,
    policy,
    slippage_float: float = common.ENGINE_SLIPPAGE_FLOAT,
    capital_float: float = common.CAPITAL_BASE_FLOAT,
    record_positions_bool: bool = False,
    start_ts: pd.Timestamp = common.TRADING_START_TS,
) -> dict:
    date_index = feature_obj.date_index
    open_arr = feature_obj.open64()
    close_arr = feature_obj.close_arr
    unadj_arr = feature_obj.unadjusted_close()
    dividend_arr = feature_obj.dividend64()
    adv_arr = feature_obj.adv20_dollar()
    date_count_int, symbol_count_int = close_arr.shape
    start_pos_int = int(date_index.searchsorted(start_ts, side="left"))
    historical_bool = True

    state = State(symbol_count_int, capital_float)
    cost_vec = np.zeros(symbol_count_int)
    proceeds_vec = np.zeros(symbol_count_int)
    total_vec = np.full(date_count_int, np.nan)
    cash_vec = np.full(date_count_int, np.nan)
    exposure_vec = np.full(date_count_int, np.nan)
    position_count_vec = np.zeros(date_count_int)
    traded_vec = np.zeros(date_count_int)
    commission_vec = np.zeros(date_count_int)
    slippage_cost_vec = np.zeros(date_count_int)
    order_frac_list: list[tuple[int, float, float]] = []
    trade_list: list[dict] = []
    intent_log_list: list[dict] = []
    position_log_list: list[tuple[int, np.ndarray, np.ndarray]] = []
    skipped_missing_close_int = 0
    phantom_fill_int = 0
    previous_total_float = capital_float

    def log_intents(decision_pos_int: int, intent_list: list[Intent]) -> None:
        for intent in intent_list:
            intent_log_list.append({"decision_pos": int(decision_pos_int), "decision_ts": date_index[decision_pos_int], "symbol_idx": int(intent.symbol_idx),
                                    "kind": intent.kind_str, "amount": float(intent.amount_float), "reason": intent.reason_str, "stop_level": float(intent.stop_level_float)})

    def close_episode(symbol_idx: int, exit_pos_int: int, reason_str: str, price_float: float) -> None:
        trade_list.append({"symbol_idx": int(symbol_idx), "entry_pos": int(state.entry_pos_vec[symbol_idx]), "exit_pos": int(exit_pos_int),
                           "holding_sessions": int(exit_pos_int - state.entry_pos_vec[symbol_idx]), "reason": reason_str, "cost": float(cost_vec[symbol_idx]),
                           "proceeds": float(proceeds_vec[symbol_idx]), "pnl": float(proceeds_vec[symbol_idx] - cost_vec[symbol_idx]), "exit_open": float(price_float)})
        state.shares_vec[symbol_idx] = 0.0
        state.hwm_vec[symbol_idx] = np.nan
        state.entry_pos_vec[symbol_idx] = -1
        cost_vec[symbol_idx] = 0.0
        proceeds_vec[symbol_idx] = 0.0

    pending_list = policy.decide(start_pos_int - 1, state)
    log_intents(start_pos_int - 1, pending_list)

    for pos_int in range(start_pos_int, date_count_int):
        held_idx_vec = np.flatnonzero(state.shares_vec != 0.0)
        if len(held_idx_vec) > 0:  # dividends of the entitlement session p-1, credited before the open of p
            gross_vec = state.shares_vec[held_idx_vec] * np.nan_to_num(dividend_arr[pos_int - 1, held_idx_vec])
            state.cash_float += float(np.sum(np.where(gross_vec > 0, gross_vec * common.DIVIDEND_NET_RATE_FLOAT, gross_vec)))
        liquidated_set: set[int] = set()
        if len(held_idx_vec) > 0:  # terminal liquidation at the last available close before p (engine gap G-014)
            missing_vec = ~np.isfinite(open_arr[pos_int, held_idx_vec]) | ~np.isfinite(close_arr[pos_int, held_idx_vec])
            for symbol_idx in held_idx_vec[missing_vec]:
                anchor_int, last_close_float = last_finite_close_before(close_arr, pos_int, int(symbol_idx))
                scale_float = unadj_arr[anchor_int, symbol_idx] / close_arr[anchor_int, symbol_idx]
                shares_float = state.shares_vec[symbol_idx]
                fee_float = commission_float(-shares_float, scale_float, historical_bool)
                proceeds_float = shares_float * last_close_float
                state.cash_float += proceeds_float - fee_float
                traded_vec[pos_int] += abs(proceeds_float)
                commission_vec[pos_int] += fee_float
                proceeds_vec[symbol_idx] += proceeds_float
                close_episode(int(symbol_idx), pos_int, "terminal", last_close_float)
                liquidated_set.add(int(symbol_idx))
        for intent in pending_list:  # fills of the intents decided at the close of p-1
            symbol_idx = intent.symbol_idx
            if symbol_idx in liquidated_set:
                continue
            open_float = float(open_arr[pos_int, symbol_idx])
            if not np.isfinite(open_float):
                continue
            if intent.kind_str == "exit":
                delta_float = -state.shares_vec[symbol_idx]
            elif intent.kind_str == "target_pct":
                delta_float = ledger_target_shares_float(intent.amount_float * previous_total_float, unadj_arr[pos_int - 1, symbol_idx], close_arr[pos_int - 1, symbol_idx], historical_bool) - state.shares_vec[symbol_idx]
            elif intent.kind_str == "value":
                delta_float = ledger_target_shares_float(intent.amount_float, unadj_arr[pos_int - 1, symbol_idx], close_arr[pos_int - 1, symbol_idx], historical_bool)
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
            price_float = open_float * (1.0 + np.sign(delta_float) * slippage_float)
            fee_float = commission_float(delta_float, execution_scale_float, historical_bool)
            state.cash_float -= delta_float * price_float + fee_float
            traded_vec[pos_int] += abs(delta_float * price_float)
            commission_vec[pos_int] += fee_float
            slippage_cost_vec[pos_int] += abs(delta_float) * open_float * slippage_float
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
                    close_episode(symbol_idx, pos_int, intent.reason_str, open_float)
        held_idx_vec = np.flatnonzero(state.shares_vec != 0.0)
        portfolio_value_float = float(np.dot(state.shares_vec[held_idx_vec], close_arr[pos_int, held_idx_vec])) if len(held_idx_vec) else 0.0
        total_float = state.cash_float + portfolio_value_float
        total_vec[pos_int] = total_float
        cash_vec[pos_int] = state.cash_float
        exposure_vec[pos_int] = portfolio_value_float / total_float if total_float > 0 else np.nan
        position_count_vec[pos_int] = len(held_idx_vec)
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
    return {
        "total_ser": total_ser,
        "return_ser": total_ser.pct_change().iloc[1:],
        # *** CRITICAL *** cash share at the close of t; the sweep applies BIL's return of t+1 to it.
        "cash_weight_ser": (cash_ser.clip(lower=0.0) / total_ser).rename("cash_weight"),
        "cash_ser": cash_ser,
        "exposure_ser": pd.Series(exposure_vec[start_pos_int:], index=index),
        "position_count_ser": pd.Series(position_count_vec[start_pos_int:], index=index),
        "traded_notional_ser": pd.Series(traded_vec[start_pos_int:], index=index),
        "commission_ser": pd.Series(commission_vec[start_pos_int:], index=index),
        "slippage_cost_ser": pd.Series(slippage_cost_vec[start_pos_int:], index=index),
        "order_frac_arr": np.array(order_frac_list, dtype=np.float64).reshape(-1, 3),
        "trade_df": pd.DataFrame(trade_list),
        "intent_df": pd.DataFrame(intent_log_list),
        "position_log": position_log_list,
        "skipped_missing_close_int": skipped_missing_close_int,
        "phantom_fill_int": phantom_fill_int,
        "final_total_float": float(total_vec[-1]),
        "start_pos_int": start_pos_int,
        "capital_float": capital_float,
    }


def run_summary(sim_dict: dict, date_index: pd.DatetimeIndex, end_ts: pd.Timestamp = common.END_TS) -> dict:
    """Turnover, holding, positions, costs and capacity per run (PREREG section 8)."""
    total_ser = sim_dict["total_ser"].loc[:end_ts]
    years_float = len(total_ser) / 252.0
    mean_nav_float = float(total_ser.mean())
    trade_df = sim_dict["trade_df"]
    if len(trade_df) > 0:
        trade_df = trade_df[date_index[trade_df["exit_pos"].to_numpy()] <= end_ts]
    terminal_df = trade_df[trade_df["reason"] == "terminal"] if len(trade_df) else trade_df
    traded_total_float = float(sim_dict["traded_notional_ser"].loc[:end_ts].sum())
    commission_total_float = float(sim_dict["commission_ser"].loc[:end_ts].sum())
    slippage_total_float = float(sim_dict["slippage_cost_ser"].loc[:end_ts].sum())
    net_pnl_float = float(total_ser.iloc[-1] - sim_dict["capital_float"])
    gross_pnl_float = net_pnl_float + commission_total_float + slippage_total_float
    return {
        "turnover_one_way_x_per_year": traded_total_float / 2.0 / mean_nav_float / years_float,
        "turnover_x_per_year": traded_total_float / mean_nav_float / years_float,
        "commission_pct_nav_per_year": commission_total_float / mean_nav_float / years_float,
        "slippage_pct_nav_per_year": slippage_total_float / mean_nav_float / years_float,
        "cost_share_of_gross_pnl": (commission_total_float + slippage_total_float) / gross_pnl_float if gross_pnl_float > 0 else float("nan"),
        "gross_pnl_usd": gross_pnl_float,
        "net_pnl_usd": net_pnl_float,
        "mean_exposure": float(sim_dict["exposure_ser"].loc[:end_ts].mean()),
        "mean_positions": float(sim_dict["position_count_ser"].loc[:end_ts].mean()),
        "round_trips_int": int(len(trade_df)),
        "round_trips_per_year": float(len(trade_df) / years_float),
        "holding_sessions_mean": float(trade_df["holding_sessions"].mean()) if len(trade_df) else float("nan"),
        "holding_sessions_median": float(trade_df["holding_sessions"].median()) if len(trade_df) else float("nan"),
        "exits_by_reason": {str(k): int(v) for k, v in trade_df["reason"].value_counts().items()} if len(trade_df) else {},
        "terminal_liquidations_int": int(len(terminal_df)),
        "terminal_pnl_usd": float(terminal_df["pnl"].sum()) if len(terminal_df) else 0.0,
        "skipped_missing_close_int": int(sim_dict["skipped_missing_close_int"]),
        "phantom_fill_int": int(sim_dict["phantom_fill_int"]),
        "final_nav_usd": float(total_ser.iloc[-1]),
        "capacity_full": capacity_with_orders(sim_dict["order_frac_arr"], date_index, None, end_ts),
        "capacity_2021_2026": capacity_with_orders(sim_dict["order_frac_arr"], date_index, pd.Timestamp("2021-01-01"), end_ts),
    }


def capacity_with_orders(order_frac_arr: np.ndarray, date_index: pd.DatetimeIndex, start_ts, end_ts) -> dict:
    """capacity_stats (AUM at which the p95 / largest order reaches 1% / 5% of ADV20) plus the largest and p95 order as a
    share of ADV20 at the run's own capital base."""
    out_dict = capacity_stats(order_frac_arr, date_index, start_ts, end_ts)
    if len(order_frac_arr) == 0 or out_dict.get("orders_int", 0) == 0:
        return out_dict
    pos_vec = order_frac_arr[:, 0].astype(int)
    keep_vec = np.isfinite(order_frac_arr[:, 2]) & (order_frac_arr[:, 2] > 0) & (date_index[pos_vec] <= end_ts)
    if start_ts is not None:
        keep_vec &= date_index[pos_vec] >= start_ts
    per_dollar_vec = order_frac_arr[keep_vec, 1] / order_frac_arr[keep_vec, 2]
    out_dict["order_share_of_adv20_at_capital_max"] = float(per_dollar_vec.max() * common.CAPITAL_BASE_FLOAT)
    out_dict["order_share_of_adv20_at_capital_p95"] = float(np.quantile(per_dollar_vec, 0.95) * common.CAPITAL_BASE_FLOAT)
    return out_dict
