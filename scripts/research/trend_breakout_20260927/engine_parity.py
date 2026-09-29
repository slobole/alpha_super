"""Engine parity gate (PREREG section 6): the real repo engine executes the replica's order intents.

G1: the live pod's own run (run_variant of strategy_mo_atr_normalized_ndx_vxn_scaled) versus the replica's L in
    historical-share mode and versus the stored ndx_atrfix sleeve.
G2: daily configurations (stops, refills, breakouts) replayed by the real engine through ResearchOrderIntentStrategy,
    which only places order_target(asset, 0), order_target_percent(asset, w) and order_value(asset, D) on the bar whose
    previous_bar is the intent's decision close. Every other piece of accounting is the engine's own code.
"""

from __future__ import annotations

import dataclasses
import time
from collections import defaultdict

import numpy as np
import pandas as pd

from alpha.engine.backtest import run_daily  # noqa: E402
from alpha.engine.strategy import Strategy  # noqa: E402

from trend_breakout_20260927 import common

CORR_THRESHOLD_FLOAT = 0.9999
CAGR_TOLERANCE_FLOAT = 0.0005


def default_trade_id_int() -> int:
    return -1


class ResearchOrderIntentStrategy(Strategy):
    """Replays recorded intents; historical share units on; no selection logic of its own."""

    enable_signal_audit = False

    def __init__(self, name: str, intents_by_decision_dict: dict, capital_base: float = common.CAPITAL_BASE_FLOAT,
                 slippage: float = common.ENGINE_SLIPPAGE_FLOAT, commission_per_share: float = common.COMMISSION_PER_SHARE_FLOAT,
                 commission_minimum: float = common.COMMISSION_MINIMUM_FLOAT):
        super().__init__(name=name, benchmarks=[], capital_base=capital_base, slippage=slippage,
                         commission_per_share=commission_per_share, commission_minimum=commission_minimum)
        self.historical_share_units_bool = True
        self.intents_by_decision_dict = intents_by_decision_dict
        self.trade_id_int = 0
        self.current_trade_map = defaultdict(default_trade_id_int)

    def compute_signals(self, pricing_data: pd.DataFrame) -> pd.DataFrame:
        return pricing_data

    def iterate(self, data: pd.DataFrame, close: pd.Series, open_prices: pd.Series):
        if close is None or data is None:
            return
        # *** CRITICAL *** intents are keyed by the decision close = previous_bar; fills happen at this bar's open.
        for symbol_str, kind_str, amount_float in self.intents_by_decision_dict.get(pd.Timestamp(self.previous_bar), []):
            if kind_str == "exit":
                self.order_target(symbol_str, 0.0, trade_id=self.current_trade_map[symbol_str])
                continue
            if self.get_position(symbol_str) == 0.0:
                self.trade_id_int += 1
                self.current_trade_map[symbol_str] = self.trade_id_int
            if kind_str == "target_pct":
                self.order_target_percent(symbol_str, float(amount_float), trade_id=self.current_trade_map[symbol_str])
            elif kind_str == "value":
                self.order_value(symbol_str, float(amount_float), trade_id=self.current_trade_map[symbol_str])
            else:
                raise ValueError(kind_str)


def load_engine_pricing_data(universe_str: str):
    from strategies.momentum.strategy_mo_atr_normalized_ndx_vxn_scaled import DEFAULT_CONFIG, get_vxn_scaled_atr_normalized_ndx_data

    config_obj = dataclasses.replace(DEFAULT_CONFIG, indexname_str=common.UNIVERSE_INDEXNAME_DICT[universe_str])
    pricing_data_df, _universe_df, _schedule_df, _vxn_df = get_vxn_scaled_atr_normalized_ndx_data(config_obj, include_total_return_benchmark_bool=False)
    calendar_idx = pricing_data_df.index[pricing_data_df.index >= pd.Timestamp(config_obj.backtest_start_date_str)]
    return pricing_data_df, calendar_idx


def intents_from_intent_df(intent_df: pd.DataFrame, symbol_list: list[str]) -> dict:
    intents_by_decision_dict: dict = defaultdict(list)
    for row in intent_df.itertuples(index=False):
        intents_by_decision_dict[pd.Timestamp(row.decision_ts)].append((symbol_list[int(row.symbol_idx)], str(row.kind), float(row.amount)))
    return dict(intents_by_decision_dict)


def run_engine_replay(universe_str: str, intents_by_decision_dict: dict, name_str: str, slippage_float: float = common.ENGINE_SLIPPAGE_FLOAT):
    pricing_data_df, calendar_idx = load_engine_pricing_data(universe_str)
    strategy_obj = ResearchOrderIntentStrategy(name_str, intents_by_decision_dict, slippage=slippage_float)
    start_float = time.perf_counter()
    run_daily(strategy_obj, pricing_data_df, calendar=calendar_idx, show_progress=False, show_signal_progress_bool=False, audit_override_bool=False)
    common.log_progress(f"engine replay {name_str}: {time.perf_counter() - start_float:.0f}s, {len(strategy_obj.get_transactions())} transactions")
    return strategy_obj


def run_live_pod() -> Strategy:
    from strategies.momentum.strategy_mo_atr_normalized_ndx_vxn_scaled import run_variant

    start_float = time.perf_counter()
    strategy_obj = run_variant(show_display_bool=False, save_results_bool=False)
    common.log_progress(f"engine live pod run: {time.perf_counter() - start_float:.0f}s, {len(strategy_obj.get_transactions())} transactions")
    return strategy_obj


def engine_nav_ser(strategy_obj: Strategy) -> pd.Series:
    nav_ser = strategy_obj.results["total_value"].astype(float)
    nav_ser.index = pd.to_datetime(nav_ser.index)
    return nav_ser


def engine_daily_positions(strategy_obj: Strategy, date_index: pd.DatetimeIndex) -> dict:
    """Set of symbols with a non-zero ledger position at the close of every session (from the transaction ledger)."""
    transaction_df = strategy_obj.get_transactions().copy()
    transaction_df["bar"] = pd.to_datetime(transaction_df["bar"])
    position_by_bar_dict: dict = {}
    running_dict: dict[str, float] = defaultdict(float)
    grouped = {bar_ts: group_df for bar_ts, group_df in transaction_df.groupby("bar")} if len(transaction_df) else {}
    for bar_ts in date_index:
        group_df = grouped.get(bar_ts)
        if group_df is not None:
            for asset_str, amount_float in zip(group_df["asset"], group_df["amount"]):
                running_dict[str(asset_str)] += float(amount_float)
        position_by_bar_dict[bar_ts] = frozenset(asset_str for asset_str, share_float in running_dict.items() if share_float != 0.0)
    return position_by_bar_dict


def replica_daily_positions(position_log: list, date_index: pd.DatetimeIndex, symbol_list: list[str]) -> dict:
    return {date_index[pos_int]: frozenset(symbol_list[int(i)] for i in held_idx_vec) for pos_int, held_idx_vec, _ in position_log}


def compare_nav(a_nav_ser: pd.Series, b_nav_ser: pd.Series, end_ts: pd.Timestamp = common.END_TS) -> dict:
    a_ret_ser = a_nav_ser.pct_change().iloc[1:]
    b_ret_ser = b_nav_ser.pct_change().iloc[1:]
    common_index = a_ret_ser.index.intersection(b_ret_ser.index)
    common_index = common_index[common_index <= end_ts]
    a_vec = a_ret_ser.reindex(common_index).to_numpy()
    b_vec = b_ret_ser.reindex(common_index).to_numpy()

    def cagr(return_vec: np.ndarray) -> float:
        return float(np.prod(1.0 + return_vec) ** (252.0 / len(return_vec)) - 1.0)

    report_dict = {
        "sessions_compared_int": int(len(common_index)),
        "first_date": str(common_index[0].date()),
        "last_date": str(common_index[-1].date()),
        "daily_return_corr_float": float(np.corrcoef(a_vec, b_vec)[0, 1]),
        "max_abs_daily_diff_float": float(np.max(np.abs(a_vec - b_vec))),
        "cagr_a_float": cagr(a_vec),
        "cagr_b_float": cagr(b_vec),
        "final_nav_a_float": float(a_nav_ser.loc[:end_ts].iloc[-1]),
        "final_nav_b_float": float(b_nav_ser.loc[:end_ts].iloc[-1]),
    }
    report_dict["cagr_gap_pp_float"] = abs(report_dict["cagr_a_float"] - report_dict["cagr_b_float"]) * 100.0
    report_dict["passed_bool"] = bool(
        report_dict["daily_return_corr_float"] >= CORR_THRESHOLD_FLOAT and abs(report_dict["cagr_a_float"] - report_dict["cagr_b_float"]) <= CAGR_TOLERANCE_FLOAT
    )
    return report_dict


def compare_positions(a_position_dict: dict, b_position_dict: dict, end_ts: pd.Timestamp = common.END_TS) -> dict:
    common_dates = sorted(set(a_position_dict) & set(b_position_dict))
    common_dates = [d for d in common_dates if d <= end_ts]
    mismatch_list = [str(d.date()) for d in common_dates if a_position_dict[d] != b_position_dict[d]]
    return {
        "sessions_compared_int": len(common_dates),
        "position_mismatch_sessions_int": len(mismatch_list),
        "first_mismatches": mismatch_list[:10],
        "identical_bool": len(mismatch_list) == 0,
    }
