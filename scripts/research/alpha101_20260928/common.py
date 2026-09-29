"""Shared constants, paths, notes and the reused book / sweep helpers for the Alpha101 study (research only).

Frozen plan: docs/research/ALPHA101_PREREG_20260928.md. Books, the idle-cash sweep, the controls and the metric
definitions are imported unchanged from the new-pod-search package so that C_BIL / C_SPY / G3 agree by construction.
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

from new_pod_search_20260927 import common as nps_common  # noqa: E402
from trend_breakout_20260927 import common as trend_common  # noqa: E402

MAIN_CHECKOUT_PATH = trend_common.MAIN_CHECKOUT_PATH
RESULTS_DIR_PATH = Path(os.environ.get("ALPHA101_OUT", str(MAIN_CHECKOUT_PATH / "results" / "research" / "alpha101_20260928")))
CACHE_DIR_PATH = RESULTS_DIR_PATH / "_cache"
CHART_DIR_PATH = RESULTS_DIR_PATH / "charts"
SLEEVE_SERIES_PATH = trend_common.SLEEVE_SERIES_PATH
PREREG_REL_PATH = "docs/research/ALPHA101_PREREG_20260928.md"

END_TS = trend_common.END_TS  # 2026-08-19
TRADING_START_TS = pd.Timestamp("2000-01-03")
SEED_INT = 20260928
BOOT_N_INT = 2000
BOOT_BLOCK_FLOAT = 21.0
N_TRIALS_INT = 503  # DSR lower bound: 377 earlier + 100 alphas + 24 cells + 2 HEDGED (PREREG section 10)
CAPACITY_MIN_AUM_FLOAT = trend_common.CAPACITY_MIN_AUM_FLOAT
CAPACITY_PARTICIPATION_FLOAT = 0.05

# standalone blocks (PREREG section 3); labels POST and PAPER
BLOCK_DICT = {
    "P1": ("2000-01-03", "2011-12-31"),
    "P2": ("2012-01-01", "2021-12-31"),
    "P3": ("2022-01-01", "2026-08-19"),
    "FULL": ("2000-01-03", "2026-08-19"),
}
LABEL_BLOCK_DICT = {"POST": ("2016-01-01", "2026-08-19"), "PAPER": ("2010-01-04", "2013-12-31")}
STAGE_A_BLOCK_DICT = {**BLOCK_DICT, **LABEL_BLOCK_DICT}
HEDGED_START_STR = nps_common.HEDGED_START_STR  # 2006-07-03
BLOCK_DICT_HEDGED = dict(nps_common.BLOCK_DICT_HEDGED)
BOOK_BLOCK_DICT = dict(trend_common.BOOK_BLOCK_DICT)
RULE_BLOCK_TUPLE = trend_common.RULE_BLOCK_TUPLE
DD_TOLERANCE_FLOAT = trend_common.DD_TOLERANCE_FLOAT
OWNER_GATE_SHARPE_FLOAT = trend_common.OWNER_GATE_SHARPE_FLOAT
OWNER_GATE_MAX_DD_FLOAT = trend_common.OWNER_GATE_MAX_DD_FLOAT

ENGINE_SLIPPAGE_FLOAT = trend_common.ENGINE_SLIPPAGE_FLOAT
STRESS_SLIPPAGE_FLOAT = trend_common.STRESS_SLIPPAGE_FLOAT
COMMISSION_PER_SHARE_FLOAT = trend_common.COMMISSION_PER_SHARE_FLOAT
COMMISSION_MINIMUM_FLOAT = trend_common.COMMISSION_MINIMUM_FLOAT
DIVIDEND_NET_RATE_FLOAT = trend_common.DIVIDEND_NET_RATE_FLOAT
CAPITAL_BASE_FLOAT = trend_common.CAPITAL_BASE_FLOAT
SMALL_ACCOUNT_CAPITAL_TUPLE = (10_000.0, 25_000.0, 1_000_000.0)
COST_DICT = {"engine": ENGINE_SLIPPAGE_FLOAT, "stress": STRESS_SLIPPAGE_FLOAT}
COST_BPS_TUPLE = (3.0, 8.0)  # Stage A net Sharpe / break-even comparison points (bps per side)

# Stage B grid (PREREG section 7)
N_TUPLE = (10, 20, 40)
B_TUPLE = (1, 2, 4, 8)
COMPOSITE_TUPLE = ("C_EQ", "C_WF")
A0_N_INT, A0_B_INT, A0_COMPOSITE_STR = 20, 2, "C_EQ"
LABEL_N_INT, LABEL_B_INT = 20, 2
MIN_ALPHAS_C_EQ_INT = 50
WF_FIRST_YEAR_INT = 2003

CANDIDATE_WEIGHT_DICT = nps_common.CANDIDATE_WEIGHT_DICT
G3_WEIGHT_DICT = nps_common.G3_WEIGHT_DICT

# reused unchanged
metric_dict = nps_common.metric_dict
window_metrics = nps_common.window_metrics
book_metrics_by_block = nps_common.book_metrics_by_block
book_window_return_ser = nps_common.book_window_return_ser
candidate_book_blocks = nps_common.candidate_book_blocks
candidate_book_series = nps_common.candidate_book_series
control_books = nps_common.control_books
sweep_return_ser = nps_common.sweep_return_ser
load_taa_ser = nps_common.load_taa_ser
load_bil_ret_ser = nps_common.load_bil_ret_ser
load_spy_tr_ret_ser = nps_common.load_spy_tr_ret_ser
load_l_ret_ser = nps_common.load_l_ret_ser


def cell_key(composite_str: str, n_int: int, b_int: int, form_str: str = "LONG") -> str:
    return f"{composite_str}|N{n_int}|B{b_int}|{form_str}"


def parse_cell_key(key_str: str) -> dict:
    composite_str, n_str, b_str, form_str = key_str.split("|")
    return {"composite": composite_str, "n_int": int(n_str[1:]), "b_int": int(b_str[1:]), "form": form_str}


def log_progress(message_str: str, log_name_str: str = "progress_log.txt") -> None:
    RESULTS_DIR_PATH.mkdir(parents=True, exist_ok=True)
    stamp_str = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(RESULTS_DIR_PATH / log_name_str, "a", encoding="utf-8") as file_obj:
        file_obj.write(f"{stamp_str} {message_str}\n")
    print(f"[{stamp_str}] {message_str}", flush=True)


def write_json(name_str: str, payload_obj) -> Path:
    RESULTS_DIR_PATH.mkdir(parents=True, exist_ok=True)
    path_obj = RESULTS_DIR_PATH / name_str
    path_obj.write_text(json.dumps(payload_obj, indent=1, default=trend_common._json_default), encoding="utf-8")
    return path_obj


def read_json(name_str: str, default_obj=None):
    path_obj = RESULTS_DIR_PATH / name_str
    return json.loads(path_obj.read_text(encoding="utf-8")) if path_obj.exists() else default_obj


def append_note(note_id_str: str, what_str: str, why_str: str, effect_str: str, stage_str: str = "") -> None:
    """Numbered implementation note (N1, N2, ...) in amendments_and_bugs.json; an existing id is replaced."""
    RESULTS_DIR_PATH.mkdir(parents=True, exist_ok=True)
    path_obj = RESULTS_DIR_PATH / "amendments_and_bugs.json"
    note_list = json.loads(path_obj.read_text(encoding="utf-8")) if path_obj.exists() else []
    note_list = [note for note in note_list if note.get("id") != note_id_str]
    note_list.append({"id": note_id_str, "date": dt.datetime.now().strftime("%Y-%m-%d %H:%M +03:00"), "stage": stage_str, "what": what_str, "why": why_str, "effect": effect_str})
    note_list.sort(key=lambda note: int(note["id"][1:]) if note["id"][1:].isdigit() else 999)
    path_obj.write_text(json.dumps(note_list, indent=1), encoding="utf-8")


def load_dv2_ser() -> pd.Series:
    """The live short-horizon mean-reversion sleeve `dv2` (2008-03-04..END) for the reference slot row."""
    sleeve_df = pd.read_csv(SLEEVE_SERIES_PATH, index_col=0, parse_dates=True)
    dv2_ser = sleeve_df["dv2"].astype(float).loc[:END_TS]
    dv2_ser.name = "dv2"
    return dv2_ser


def newey_west_t(x_vec: np.ndarray, lag_int: int = 5) -> float:
    """t-statistic of the mean with a Bartlett-kernel HAC variance (lag_int lags)."""
    x_vec = np.asarray(x_vec, dtype=np.float64)
    x_vec = x_vec[np.isfinite(x_vec)]
    n_int = len(x_vec)
    if n_int < lag_int + 2:
        return float("nan")
    d_vec = x_vec - x_vec.mean()
    gamma0_float = float(np.dot(d_vec, d_vec) / n_int)
    lrv_float = gamma0_float
    for lag in range(1, lag_int + 1):
        gamma_float = float(np.dot(d_vec[lag:], d_vec[:-lag]) / n_int)
        lrv_float += 2.0 * (1.0 - lag / (lag_int + 1.0)) * gamma_float
    if lrv_float <= 0:
        return float("nan")
    return float(x_vec.mean() / np.sqrt(lrv_float / n_int))
