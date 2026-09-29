"""Shared constants, series and books for merger arbitrage v2 (research only).

Frozen plan: docs/research/MERGER_ARB_V2_PREREG_20260927.md. Reuses the new-pod study's windows, sweep and book code.
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

from new_pod_search_20260927 import common as pod_common  # noqa: E402
from trend_breakout_20260927 import common as trend_common  # noqa: E402

MAIN_CHECKOUT_PATH = trend_common.MAIN_CHECKOUT_PATH
RESULTS_DIR_PATH = Path(os.environ.get("MERGER_ARB_V2_OUT", str(MAIN_CHECKOUT_PATH / "results" / "research" / "merger_arb_v2_20260927")))
CACHE_DIR_PATH = RESULTS_DIR_PATH / "_cache"
CHART_DIR_PATH = RESULTS_DIR_PATH / "charts"
POD_RESULTS_PATH = pod_common.RESULTS_DIR_PATH
V1_M0_KEY = "M|J15|th1|W5|K20|brk95"  # the version-1 anchor (reference label)

END_TS = pod_common.END_TS
TRADING_START_TS = pod_common.TRADING_START_TS
HISTORY_START_STR = "1999-01-01"
BLOCK_DICT = dict(pod_common.BLOCK_DICT)
BOOK_BLOCK_DICT = dict(pod_common.BOOK_BLOCK_DICT)
RULE_BLOCK_TUPLE = pod_common.RULE_BLOCK_TUPLE
DD_TOLERANCE_FLOAT = pod_common.DD_TOLERANCE_FLOAT
OWNER_GATE_SHARPE_FLOAT = pod_common.OWNER_GATE_SHARPE_FLOAT
OWNER_GATE_MAX_DD_FLOAT = pod_common.OWNER_GATE_MAX_DD_FLOAT
ENGINE_SLIPPAGE_FLOAT = pod_common.ENGINE_SLIPPAGE_FLOAT
STRESS_SLIPPAGE_FLOAT = pod_common.STRESS_SLIPPAGE_FLOAT
CAPITAL_BASE_FLOAT = pod_common.CAPITAL_BASE_FLOAT
SMALL_CAPITAL_FLOAT = 25_000.0
CAPACITY_MIN_AUM_FLOAT = pod_common.CAPACITY_MIN_AUM_FLOAT
SEED_INT = 20260927
BOOT_N_INT = 2000
BOOT_BLOCK_FLOAT = 21.0
N_TRIALS_INT = 377
MNA_START_STR = "2009-11-17"
CANDIDATE_WEIGHT_DICT = pod_common.CANDIDATE_WEIGHT_DICT
G3_WEIGHT_DICT = pod_common.G3_WEIGHT_DICT

metric_dict = pod_common.metric_dict
window_metrics = pod_common.window_metrics
book_metrics_by_block = pod_common.book_metrics_by_block
book_window_return_ser = pod_common.book_window_return_ser
load_taa_ser = pod_common.load_taa_ser
load_bil_ret_ser = pod_common.load_bil_ret_ser
load_spy_tr_ret_ser = pod_common.load_spy_tr_ret_ser
load_l_ret_ser = pod_common.load_l_ret_ser
sweep_return_ser = pod_common.sweep_return_ser
candidate_book_blocks = pod_common.candidate_book_blocks
candidate_book_series = pod_common.candidate_book_series
control_books = pod_common.control_books


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


def load_mna_ret_ser() -> pd.Series:
    """MNA (IQ Merger Arbitrage ETF) total-return daily returns from 2009-11-17 (reference slot row)."""
    ret_ser = pod_common.load_total_return_ret_ser("MNA", "MNA")
    return ret_ser.loc[MNA_START_STR:]


def load_v1_m0_ret_ser(bil_ser: pd.Series) -> pd.Series | None:
    """Version-1 anchor M0 with the sweep (reference label), from the new-pod study's outputs."""
    ret_path = POD_RESULTS_PATH / "returns_M_R1000_engine.parquet"
    cashw_path = POD_RESULTS_PATH / "cashw_M_R1000_engine.parquet"
    if not ret_path.exists() or not cashw_path.exists():
        return None
    ret_df = pd.read_parquet(ret_path)
    ret_df.index = pd.to_datetime(ret_df.index)
    cashw_df = pd.read_parquet(cashw_path)
    cashw_df.index = pd.to_datetime(cashw_df.index)
    return sweep_return_ser(ret_df[V1_M0_KEY].loc[:END_TS], cashw_df[V1_M0_KEY], bil_ser).rename("M0_v1")
