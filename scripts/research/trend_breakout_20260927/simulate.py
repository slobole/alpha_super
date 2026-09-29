"""Daily replica of alpha/engine/strategy.py for path-dependent policies (research only).

Per session p (the engine's current_bar; the decision was taken at the close of p - 1 = previous_bar):
    1. dividend cash: cash += 0.75 x shares_i x Dividend_i(p - 1) for names held before the open (negative gross
       dividends are not withheld; longs only here)
    2. a held name without an Open or a Close on p is liquidated at its last available close before p: no slippage,
       commission on raw-equivalent shares at that anchor bar (x terminal_haircut on the price for the sensitivity)
    3. the intents decided at the close of p - 1 fill at Open_p x (1 +/- slippage):
           exit        delta = -shares
           target_pct  target = trunc(w x V_{p-1} / U_{p-1}) x U_{p-1} / Close_{p-1};   delta = target - shares
           value       delta  = trunc(D / U_{p-1}) x U_{p-1} / Close_{p-1}
       a zero delta is cancelled without commission; commission = max($1, $0.005 x |delta| x Close_p / U_p)
    4. V_p = cash + sum_i shares_i x Close_p
    5. highest close since entry updated with Close_p; the policy decides the intents for p + 1

Legacy mode (historical_share_bool=False) reproduces the 26 Sep replica's whole-adjusted-share sizing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from trend_breakout_20260927 import common
from trend_breakout_20260927.features import DailyFeatureBook, last_finite_close_before
from trend_breakout_20260927.policies import Intent, State


def ledger_target_shares_float(dollars_float: float, unadjusted_close_float: float, adjusted_close_float: float, historical_bool: bool) -> float:
    """Engine sizing at the decision close (Strategy.historical_share_amount_float / legacy int rounding)."""
    if not np.isfinite(adjusted_close_float) or adjusted_close_float <= 0.0:
        raise RuntimeError("invalid decision close for a sized order")
    if not historical_bool:
        return float(int(dollars_float / adjusted_close_float))
    if not np.isfinite(unadjusted_close_float) or unadjusted_close_float <= 0.0:
        raise RuntimeError("invalid decision Unadjusted Close for a sized order")
    price_scale_float = unadjusted_close_float / adjusted_close_float
    # *** CRITICAL *** whole raw shares fixed from the decision close, then expressed in adjusted ledger units.
    return float(int(dollars_float / unadjusted_close_float) * price_scale_float)


def commission_float(ledger_delta_float: float, price_scale_float: float, historical_bool: bool) -> float:
    share_float = abs(ledger_delta_float)
    if historical_bool:
        share_float = share_float / price_scale_float  # raw-equivalent shares at the execution (or anchor) bar
    return max(common.COMMISSION_MINIMUM_FLOAT, common.COMMISSION_PER_SHARE_FLOAT * share_float)


def simulate(
    feature_obj: DailyFeatureBook,
    policy,
    slippage_float: float = common.ENGINE_SLIPPAGE_FLOAT,
    historical_share_bool: bool = True,
    terminal_haircut_float: float = 1.0,
    record_positions_bool: bool = False,
    phantom_cancel_bool: bool = False,
) -> dict:
    """phantom_cancel_bool=True (invariance diagnostic only) cancels orders below 1e-3 raw-equivalent shares instead of
    filling them at the $1 minimum commission, which is what the engine does; the study's runs keep the engine's rule."""
    date_index = feature_obj.date_index
    open_arr = feature_obj.open64()
    close_arr = feature_obj.close_arr
    unadj_arr = feature_obj.unadjusted_close()
    dividend_arr = feature_obj.dividend64()
    adv_arr = feature_obj.adv20_dollar()
    date_count_int, symbol_count_int = close_arr.shape
    start_pos_int = int(date_index.searchsorted(common.TRADING_START_TS, side="left"))

    state = State(symbol_count_int, common.CAPITAL_BASE_FLOAT)
    cost_vec = np.zeros(symbol_count_int)  # cost of the open episode (fill notional incl. slippage, excl. fees)
    proceeds_vec = np.zeros(symbol_count_int)  # sale proceeds of the open episode
    total_vec = np.full(date_count_int, np.nan)
    exposure_vec = np.full(date_count_int, np.nan)
    traded_vec = np.zeros(date_count_int)
    commission_vec = np.zeros(date_count_int)
    order_frac_list: list[tuple[int, float, float]] = []
    trade_list: list[dict] = []
    intent_log_list: list[dict] = []
    position_log_list: list[tuple[int, np.ndarray, np.ndarray]] = []
    skipped_missing_close_int = 0
    phantom_fill_int = 0  # |delta| < 1e-3 RAW-equivalent shares: an engine artifact (per-bar rounding of the adjusted close makes U/Close drift ~1e-8 between days, so a re-size to an unchanged raw count is a ~1e-5-share order) that costs the $1 minimum commission
    previous_total_float = common.CAPITAL_BASE_FLOAT

    def log_intents(decision_pos_int: int, intent_list: list[Intent]) -> None:
        for intent in intent_list:
            intent_log_list.append(
                {
                    "decision_pos": int(decision_pos_int),
                    "decision_ts": date_index[decision_pos_int],
                    "symbol_idx": int(intent.symbol_idx),
                    "kind": intent.kind_str,
                    "amount": float(intent.amount_float),
                    "reason": intent.reason_str,
                    "stop_level": float(intent.stop_level_float),
                }
            )

    def close_episode(symbol_idx: int, exit_pos_int: int, reason_str: str, price_float: float, stop_level_float: float) -> None:
        trade_list.append(
            {
                "symbol_idx": int(symbol_idx),
                "entry_pos": int(state.entry_pos_vec[symbol_idx]),
                "exit_pos": int(exit_pos_int),
                "holding_sessions": int(exit_pos_int - state.entry_pos_vec[symbol_idx]),
                "reason": reason_str,
                "cost": float(cost_vec[symbol_idx]),
                "proceeds": float(proceeds_vec[symbol_idx]),
                "pnl": float(proceeds_vec[symbol_idx] - cost_vec[symbol_idx]),
                "exit_open": float(price_float),
                "stop_level": float(stop_level_float),
                "fill_vs_stop": float(price_float / stop_level_float - 1.0) if np.isfinite(stop_level_float) and stop_level_float > 0 else float("nan"),
            }
        )
        state.shares_vec[symbol_idx] = 0.0
        state.hwm_vec[symbol_idx] = np.nan
        state.entry_pos_vec[symbol_idx] = -1
        cost_vec[symbol_idx] = 0.0
        proceeds_vec[symbol_idx] = 0.0

    pending_list = policy.decide(start_pos_int - 1, state)
    log_intents(start_pos_int - 1, pending_list)

    for pos_int in range(start_pos_int, date_count_int):
        held_idx_vec = np.flatnonzero(state.shares_vec != 0.0)
        # 1. dividends: Norgate stamps the Dividend on the entitlement session p-1; credited before the open of p.
        if len(held_idx_vec) > 0:
            gross_vec = state.shares_vec[held_idx_vec] * np.nan_to_num(dividend_arr[pos_int - 1, held_idx_vec])
            state.cash_float += float(np.sum(np.where(gross_vec > 0, gross_vec * common.DIVIDEND_NET_RATE_FLOAT, gross_vec)))

        # 2. terminal liquidation at the last available close before p (engine gap G-014)
        liquidated_set: set[int] = set()
        if len(held_idx_vec) > 0:
            missing_vec = ~np.isfinite(open_arr[pos_int, held_idx_vec]) | ~np.isfinite(close_arr[pos_int, held_idx_vec])
            for symbol_idx in held_idx_vec[missing_vec]:
                anchor_int, last_close_float = last_finite_close_before(close_arr, pos_int, int(symbol_idx))
                scale_float = unadj_arr[anchor_int, symbol_idx] / close_arr[anchor_int, symbol_idx]
                shares_float = state.shares_vec[symbol_idx]
                fee_float = commission_float(-shares_float, scale_float, historical_share_bool)
                proceeds_float = shares_float * last_close_float * terminal_haircut_float
                state.cash_float += proceeds_float - fee_float
                traded_vec[pos_int] += abs(proceeds_float)
                commission_vec[pos_int] += fee_float
                proceeds_vec[symbol_idx] += proceeds_float
                close_episode(int(symbol_idx), pos_int, "terminal", last_close_float * terminal_haircut_float, float("nan"))
                liquidated_set.add(int(symbol_idx))

        # 3. fills of the intents decided at the close of p-1
        for intent in pending_list:
            symbol_idx = intent.symbol_idx
            if symbol_idx in liquidated_set:
                continue  # the engine clears the asset's pending orders after a terminal liquidation
            open_float = float(open_arr[pos_int, symbol_idx])
            if not np.isfinite(open_float):
                continue  # engine: order cancelled (missing open)
            if intent.kind_str == "exit":
                delta_float = -state.shares_vec[symbol_idx]
            elif intent.kind_str == "target_pct":
                target_float = ledger_target_shares_float(
                    intent.amount_float * previous_total_float, unadj_arr[pos_int - 1, symbol_idx], close_arr[pos_int - 1, symbol_idx], historical_share_bool
                )
                delta_float = target_float - state.shares_vec[symbol_idx]
            elif intent.kind_str == "value":
                delta_float = ledger_target_shares_float(
                    intent.amount_float, unadj_arr[pos_int - 1, symbol_idx], close_arr[pos_int - 1, symbol_idx], historical_share_bool
                )
            else:
                raise ValueError(intent.kind_str)
            if delta_float == 0.0:
                continue  # engine cancels a zero-share fill without commission
            if not np.isfinite(close_arr[pos_int, symbol_idx]):
                skipped_missing_close_int += 1  # the engine would raise here; never observed, counted if it happens
                continue
            execution_scale_float = unadj_arr[pos_int, symbol_idx] / close_arr[pos_int, symbol_idx]
            if abs(delta_float) / execution_scale_float < 1e-3:  # raw-equivalent shares: no genuine order is below one raw share
                phantom_fill_int += 1
                if phantom_cancel_bool:
                    continue
            price_float = open_float * (1.0 + np.sign(delta_float) * slippage_float)
            fee_float = commission_float(delta_float, execution_scale_float, historical_share_bool)
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

        # 4. valuation at the close of p
        held_idx_vec = np.flatnonzero(state.shares_vec != 0.0)
        portfolio_value_float = float(np.dot(state.shares_vec[held_idx_vec], close_arr[pos_int, held_idx_vec])) if len(held_idx_vec) else 0.0
        total_float = state.cash_float + portfolio_value_float
        total_vec[pos_int] = total_float
        exposure_vec[pos_int] = portfolio_value_float / total_float if total_float > 0 else np.nan
        previous_total_float = total_float
        state.total_value_float = total_float

        # 5. highest close since entry (the entry day's close counts)
        if len(held_idx_vec) > 0:
            close_held_vec = close_arr[pos_int, held_idx_vec]
            state.hwm_vec[held_idx_vec] = np.where(
                np.isnan(state.hwm_vec[held_idx_vec]), close_held_vec, np.maximum(state.hwm_vec[held_idx_vec], close_held_vec)
            )
        if record_positions_bool:
            position_log_list.append((pos_int, held_idx_vec.copy(), state.entry_pos_vec[held_idx_vec].copy()))

        # 6. decisions for the next open
        if pos_int + 1 < date_count_int:
            pending_list = policy.decide(pos_int, state)
            log_intents(pos_int, pending_list)
        else:
            pending_list = []

    total_ser = pd.Series(total_vec[start_pos_int:], index=date_index[start_pos_int:], name="total_value")
    trade_df = pd.DataFrame(trade_list)
    intent_df = pd.DataFrame(intent_log_list)
    return {
        "total_ser": total_ser,
        "return_ser": total_ser.pct_change().iloc[1:],
        "exposure_ser": pd.Series(exposure_vec[start_pos_int:], index=total_ser.index),
        "traded_notional_ser": pd.Series(traded_vec[start_pos_int:], index=total_ser.index),
        "commission_ser": pd.Series(commission_vec[start_pos_int:], index=total_ser.index),
        "order_frac_arr": np.array(order_frac_list, dtype=np.float64).reshape(-1, 3),
        "trade_df": trade_df,
        "intent_df": intent_df,
        "position_log": position_log_list,
        "open_positions_end": {int(i): float(state.shares_vec[i]) for i in np.flatnonzero(state.shares_vec != 0.0)},
        "skipped_missing_close_int": skipped_missing_close_int,
        "phantom_fill_int": phantom_fill_int,
        "final_total_float": float(total_vec[-1]),
        "start_pos_int": start_pos_int,
    }


# ----------------------------------------------------------------------------------------------------------------------
# per-run summaries (PREREG section 5)
# ----------------------------------------------------------------------------------------------------------------------
def capacity_stats(order_frac_arr: np.ndarray, date_index: pd.DatetimeIndex, start_ts: pd.Timestamp | None, end_ts: pd.Timestamp) -> dict:
    """Participation per $1 of pod AUM: p = (|order| / V) / ADV20. AUM at which p reaches x = x / p."""
    if len(order_frac_arr) == 0:
        return {}
    pos_vec = order_frac_arr[:, 0].astype(int)
    frac_vec = order_frac_arr[:, 1]
    adv_vec = order_frac_arr[:, 2]
    keep_vec = np.isfinite(adv_vec) & (adv_vec > 0) & (date_index[pos_vec] <= end_ts)
    if start_ts is not None:
        keep_vec &= date_index[pos_vec] >= start_ts
    if not keep_vec.any():
        return {"orders_int": 0}
    per_dollar_vec = frac_vec[keep_vec] / adv_vec[keep_vec]
    q95_float = float(np.quantile(per_dollar_vec, 0.95))
    max_float = float(per_dollar_vec.max())
    return {
        "orders_int": int(keep_vec.sum()),
        "orders_without_adv_int": int((~(np.isfinite(adv_vec) & (adv_vec > 0))).sum()),
        "aum_p95_1pct_usd": 0.01 / q95_float,
        "aum_p95_5pct_usd": 0.05 / q95_float,
        "aum_max_1pct_usd": 0.01 / max_float,
        "aum_max_5pct_usd": 0.05 / max_float,
    }


def run_summary(sim_dict: dict, date_index: pd.DatetimeIndex, end_ts: pd.Timestamp = common.END_TS) -> dict:
    total_ser = sim_dict["total_ser"].loc[:end_ts]
    years_float = len(total_ser) / 252.0
    trade_df = sim_dict["trade_df"]
    if len(trade_df) > 0:
        trade_df = trade_df[date_index[trade_df["exit_pos"].to_numpy()] <= end_ts]
    stop_df = trade_df[trade_df["reason"] == "stop"] if len(trade_df) else trade_df
    terminal_df = trade_df[trade_df["reason"] == "terminal"] if len(trade_df) else trade_df
    total_pnl_float = float(total_ser.iloc[-1] - common.CAPITAL_BASE_FLOAT)
    traded_total_float = float(sim_dict["traded_notional_ser"].loc[:end_ts].sum())
    summary_dict = {
        "turnover_x_per_year": traded_total_float / float(total_ser.mean()) / years_float,
        "commission_pct_nav_per_year": float(sim_dict["commission_ser"].loc[:end_ts].sum()) / float(total_ser.mean()) / years_float,
        "mean_exposure": float(sim_dict["exposure_ser"].loc[:end_ts].mean()),
        "round_trips_int": int(len(trade_df)),
        "round_trips_per_year": float(len(trade_df) / years_float),
        "holding_sessions_median": float(trade_df["holding_sessions"].median()) if len(trade_df) else float("nan"),
        "holding_sessions_mean": float(trade_df["holding_sessions"].mean()) if len(trade_df) else float("nan"),
        "exits_by_reason": {str(k): int(v) for k, v in trade_df["reason"].value_counts().items()} if len(trade_df) else {},
        "stop_exits_int": int(len(stop_df)),
        "stop_fill_vs_level_mean": float(stop_df["fill_vs_stop"].mean()) if len(stop_df) else float("nan"),
        "stop_fill_vs_level_p5": float(stop_df["fill_vs_stop"].quantile(0.05)) if len(stop_df) else float("nan"),
        "stop_fill_below_minus2pct_share": float((stop_df["fill_vs_stop"] < -0.02).mean()) if len(stop_df) else float("nan"),
        "stop_gap_through_level_share": float((stop_df["fill_vs_stop"] < 0).mean()) if len(stop_df) else float("nan"),
        "terminal_liquidations_int": int(len(terminal_df)),
        "terminal_pnl_usd": float(terminal_df["pnl"].sum()) if len(terminal_df) else 0.0,
        "total_pnl_usd": total_pnl_float,
        "terminal_pnl_share": float(terminal_df["pnl"].sum() / total_pnl_float) if len(terminal_df) and total_pnl_float != 0 else 0.0,
        "terminal_notional_share": float(terminal_df["proceeds"].sum() / traded_total_float) if len(terminal_df) and traded_total_float > 0 else 0.0,
        "skipped_missing_close_int": int(sim_dict["skipped_missing_close_int"]),
        "phantom_fill_int": int(sim_dict["phantom_fill_int"]),
        "capacity_full": capacity_stats(sim_dict["order_frac_arr"], date_index, None, end_ts),
        "capacity_2021_2026": capacity_stats(sim_dict["order_frac_arr"], date_index, pd.Timestamp("2021-01-01"), end_ts),
    }
    return summary_dict
