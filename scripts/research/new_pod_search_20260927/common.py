"""Shared constants, series, the idle-cash sweep and the books for the new-pod search (research only).

Frozen plan: docs/research/NEW_POD_SEARCH_PREREG_20260927.md (sections 2, 3, 5).
"""

from __future__ import annotations

import datetime as dt
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT_PATH = Path(__file__).resolve().parents[3]
SCRIPTS_RESEARCH_PATH = REPO_ROOT_PATH / "scripts" / "research"
for path_obj in (REPO_ROOT_PATH, SCRIPTS_RESEARCH_PATH):
    if str(path_obj) not in sys.path:
        sys.path.insert(0, str(path_obj))

from trend_breakout_20260927 import common as trend_common  # noqa: E402

MAIN_CHECKOUT_PATH = trend_common.MAIN_CHECKOUT_PATH
RESULTS_DIR_PATH = Path(os.environ.get("NEW_POD_SEARCH_OUT", str(MAIN_CHECKOUT_PATH / "results" / "research" / "new_pod_search_20260927")))
CACHE_DIR_PATH = RESULTS_DIR_PATH / "_cache"
CHART_DIR_PATH = RESULTS_DIR_PATH / "charts"
TREND_RESULTS_PATH = trend_common.RESULTS_DIR_PATH
TREND_CACHE_PATH = trend_common.CACHE_DIR_PATH
L_KEY = "A|none|CASH|k+0"  # the replica's live pod in the trend study's parquet files (= ndx_atrfix, trend G1)

END_TS = trend_common.END_TS
TRADING_START_TS = trend_common.TRADING_START_TS
BLOCK_DICT = dict(trend_common.BLOCK_DICT)
HEDGED_START_STR = "2006-07-03"  # first session with an SH position (SH exists from 2006-06-21)
BLOCK_DICT_HEDGED = {
    "P1": (HEDGED_START_STR, "2011-12-31"),
    "P2": ("2012-01-01", "2021-12-31"),
    "P3": ("2022-01-01", "2026-08-19"),
    "FULL": (HEDGED_START_STR, "2026-08-19"),
}
BOOK_BLOCK_DICT = dict(trend_common.BOOK_BLOCK_DICT)
RULE_BLOCK_TUPLE = trend_common.RULE_BLOCK_TUPLE
DD_TOLERANCE_FLOAT = trend_common.DD_TOLERANCE_FLOAT
OWNER_GATE_SHARPE_FLOAT = trend_common.OWNER_GATE_SHARPE_FLOAT
OWNER_GATE_MAX_DD_FLOAT = trend_common.OWNER_GATE_MAX_DD_FLOAT
ENGINE_SLIPPAGE_FLOAT = trend_common.ENGINE_SLIPPAGE_FLOAT
STRESS_SLIPPAGE_FLOAT = trend_common.STRESS_SLIPPAGE_FLOAT
CAPITAL_BASE_FLOAT = trend_common.CAPITAL_BASE_FLOAT
CAPACITY_MIN_AUM_FLOAT = trend_common.CAPACITY_MIN_AUM_FLOAT
SEED_INT = 20260927
BOOT_N_INT = 2000
BOOT_BLOCK_FLOAT = 21.0
N_TRIALS_INT = 327

CANDIDATE_WEIGHT_DICT = {"taa": 0.5, "L": 0.25, "X": 0.25}
G3_WEIGHT_DICT = {"taa": 0.5, "L": 0.5}

metric_dict = trend_common.metric_dict
window_metrics = trend_common.window_metrics
book_metrics_by_block = trend_common.book_metrics_by_block
book_window_return_ser = trend_common.book_window_return_ser
load_taa_ser = trend_common.load_taa_ser


def log_progress(message_str: str) -> None:
    RESULTS_DIR_PATH.mkdir(parents=True, exist_ok=True)
    stamp_str = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(RESULTS_DIR_PATH / "progress_log.txt", "a", encoding="utf-8") as file_obj:
        file_obj.write(f"{stamp_str} {message_str}\n")
    print(f"[{stamp_str}] {message_str}", flush=True)


def write_json(name_str: str, payload_obj) -> Path:
    RESULTS_DIR_PATH.mkdir(parents=True, exist_ok=True)
    path_obj = RESULTS_DIR_PATH / name_str
    path_obj.write_text(json.dumps(payload_obj, indent=1, default=trend_common._json_default), encoding="utf-8")
    return path_obj


def append_note(note_dict: dict) -> None:
    path_obj = RESULTS_DIR_PATH / "amendments_and_bugs.json"
    note_list = json.loads(path_obj.read_text()) if path_obj.exists() else []
    note_list.append({"date": dt.datetime.now().strftime("%Y-%m-%d %H:%M +03:00"), **note_dict})
    path_obj.write_text(json.dumps(note_list, indent=1), encoding="utf-8")


# ----------------------------------------------------------------------------------------------------------------------
# series
# ----------------------------------------------------------------------------------------------------------------------
def load_total_return_ret_ser(symbol_str: str, name_str: str) -> pd.Series:
    """Daily total-return series of an ETF (Norgate TOTALRETURN close), 1999-01-05..END."""
    from data.norgate_loader import load_price_timeseries
    from data.norgate_snapshot_store import TOTALRETURN_ADJUSTMENT_STR

    close_ser = load_price_timeseries(symbol_str, adjustment_str=TOTALRETURN_ADJUSTMENT_STR, start_date_str="1999-01-01")["Close"].astype(float)
    ret_ser = close_ser.pct_change().dropna().loc[:END_TS]
    ret_ser.name = name_str
    return ret_ser


def load_bil_ret_ser() -> pd.Series:
    return load_total_return_ret_ser("BIL", "BIL")


def load_spy_tr_ret_ser() -> pd.Series:
    return load_total_return_ret_ser("SPY", "SPY_TR")


def load_l_ret_ser(cost_str: str = "engine") -> pd.Series:
    """The replica's live pod L (historical-share mode) from the trend study; equal to ndx_atrfix (trend G1)."""
    frame_df = pd.read_parquet(TREND_RESULTS_PATH / f"returns_A_NDX_{cost_str}.parquet")
    frame_df.index = pd.to_datetime(frame_df.index)
    l_ser = frame_df[L_KEY].astype(float).loc[:END_TS]
    l_ser.name = "L"
    return l_ser


# ----------------------------------------------------------------------------------------------------------------------
# idle-cash sweep (PREREG section 2)
# ----------------------------------------------------------------------------------------------------------------------
def sweep_return_ser(engine_return_ser: pd.Series, cash_weight_ser: pd.Series, bil_ret_ser: pd.Series) -> pd.Series:
    """pod return_t = engine return_t + max(cash_{t-1}, 0) / V_{t-1} x BIL total return_t.

    cash_weight_ser holds max(cash, 0) / V at the close of every session (from the replica). Before BIL's first return
    (2007-05-31) the sweep rate is 0, i.e. the standalone windows that start in 2000 understate a true T-bill sweep
    there (note N1); no book window is affected (they start 2008-03-04).
    """
    weight_prev_ser = cash_weight_ser.shift(1).reindex(engine_return_ser.index)
    bil_aligned_ser = bil_ret_ser.reindex(engine_return_ser.index).fillna(0.0)
    # *** CRITICAL *** the weight is the previous close's cash share; BIL's return of day t accrues on that cash.
    return (engine_return_ser + weight_prev_ser.fillna(0.0) * bil_aligned_ser).rename(engine_return_ser.name)


# ----------------------------------------------------------------------------------------------------------------------
# books (official pod model, each window its own run)
# ----------------------------------------------------------------------------------------------------------------------
def candidate_book_blocks(taa_ser: pd.Series, l_ser: pd.Series, x_ser: pd.Series, daily_bool: bool = False) -> dict:
    return book_metrics_by_block({"taa": taa_ser, "L": l_ser, "X": x_ser}, CANDIDATE_WEIGHT_DICT, daily_bool)


def candidate_book_series(taa_ser: pd.Series, l_ser: pd.Series, x_ser: pd.Series, block_str: str = "G-FULL") -> pd.Series:
    start_str, end_str = BOOK_BLOCK_DICT[block_str]
    return book_window_return_ser({"taa": taa_ser, "L": l_ser, "X": x_ser}, CANDIDATE_WEIGHT_DICT, start_str, end_str)


def control_books(taa_ser: pd.Series, l_ser: pd.Series, bil_ser: pd.Series, spy_ser: pd.Series) -> dict:
    zero_ser = pd.Series(0.0, index=l_ser.index)
    return {
        "C_BIL": candidate_book_blocks(taa_ser, l_ser, bil_ser),
        "C_SPY": candidate_book_blocks(taa_ser, l_ser, spy_ser),
        "C_CASH0": candidate_book_blocks(taa_ser, l_ser, zero_ser),
        "G3": book_metrics_by_block({"taa": taa_ser, "L": l_ser}, G3_WEIGHT_DICT),
    }
