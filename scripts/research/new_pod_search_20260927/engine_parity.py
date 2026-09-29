"""Engine parity gate for the new-pod search (PREREG section 6; research only).

The real repo engine replays the replica's intents through the trend study's ResearchOrderIntentStrategy (historical
share units). The pricing frame passed to the engine holds the symbols the run trades (plus SH and SPY), loaded through
the repo loader from 1999-01-01, i.e. the same Norgate data as the caches; that is enough because the subclass has no
selection logic of its own and the engine's ledger only touches held names.
"""

from __future__ import annotations

import time

import pandas as pd

from alpha.engine.backtest import run_daily  # noqa: E402

from new_pod_search_20260927 import common
from trend_breakout_20260927.engine_parity import (  # noqa: F401
    ResearchOrderIntentStrategy,
    compare_nav,
    compare_positions,
    engine_daily_positions,
    engine_nav_ser,
    intents_from_intent_df,
    replica_daily_positions,
)


def load_subset_pricing_data(symbol_list: list[str]) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    from data.norgate_loader import load_raw_prices

    pricing_data_df = load_raw_prices(symbols=list(dict.fromkeys(symbol_list)), benchmarks=[], start_date="1999-01-01", end_date=None)
    calendar_idx = pricing_data_df.index[pricing_data_df.index >= common.TRADING_START_TS]
    return pricing_data_df, calendar_idx


def run_engine_replay(intents_by_decision_dict: dict, symbol_list: list[str], name_str: str, slippage_float: float = common.ENGINE_SLIPPAGE_FLOAT):
    pricing_data_df, calendar_idx = load_subset_pricing_data(symbol_list)
    strategy_obj = ResearchOrderIntentStrategy(name_str, intents_by_decision_dict, slippage=slippage_float)
    start_float = time.perf_counter()
    run_daily(strategy_obj, pricing_data_df, calendar=calendar_idx, show_progress=False, show_signal_progress_bool=False, audit_override_bool=False)
    common.log_progress(f"engine replay {name_str}: {len(symbol_list)} symbols in the frame, {time.perf_counter() - start_float:.0f}s, {len(strategy_obj.get_transactions())} transactions")
    return strategy_obj


def parity_report(replica_sim: dict, engine_obj, date_index: pd.DatetimeIndex, symbol_list: list[str]) -> dict:
    engine_pos_dict = engine_daily_positions(engine_obj, date_index)
    replica_pos_dict = replica_daily_positions(replica_sim["position_log"], date_index, symbol_list)
    report_dict = compare_nav(engine_nav_ser(engine_obj), replica_sim["total_ser"]) | compare_positions(engine_pos_dict, replica_pos_dict)
    report_dict["intents_int"] = int(len(replica_sim["intent_df"]))
    report_dict["engine_transactions_int"] = int(len(engine_obj.get_transactions()))
    report_dict["passed_bool"] = bool(report_dict["passed_bool"] and report_dict["identical_bool"])
    return report_dict
