"""Shared constants, paths, sleeves, metrics and the official pod-model book (research only).

Frozen plan: docs/research/TREND_BREAKOUT_PREREG_20260927.md (sections 3, 5).
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

MAIN_CHECKOUT_PATH = Path(r"C:/Users/User/Documents/workspace/alpha_super")
RESULTS_DIR_PATH = Path(
    os.environ.get("TREND_BREAKOUT_OUT", str(MAIN_CHECKOUT_PATH / "results" / "research" / "trend_breakout_20260927"))
)
CACHE_DIR_PATH = RESULTS_DIR_PATH / "_cache"
CHART_DIR_PATH = RESULTS_DIR_PATH / "charts"
SLEEVE_SERIES_PATH = MAIN_CHECKOUT_PATH / "results/research/portfolio/portfolio_refresh_20260927/sleeve_series_incl_2008.csv.gz"
MOMENTUM_SERIES_PATH = MAIN_CHECKOUT_PATH / "results/research/portfolio/portfolio_refresh_20260927/momentum_series_full.csv.gz"

UNIVERSE_INDEXNAME_DICT = {"NDX": "Nasdaq 100", "SP500": "S&P 500", "R1000": "Russell 1000"}
TRADING_START_TS = pd.Timestamp("2000-01-01")  # first engine calendar session is 2000-01-03
END_TS = pd.Timestamp("2026-08-19")  # last TAA return (PREREG section 5)
BLOCK_DICT = {
    "P1": ("2000-01-04", "2011-12-31"),
    "P2": ("2012-01-01", "2021-12-31"),
    "P3": ("2022-01-01", "2026-08-19"),
    "FULL": ("2000-01-04", "2026-08-19"),
}
BOOK_BLOCK_DICT = {
    "G-P1": ("2008-03-04", "2011-12-31"),
    "G-P2": ("2012-10-02", "2021-12-31"),
    "G-P3": ("2022-01-01", "2026-08-19"),
    "G-FULL": ("2012-10-02", "2026-08-19"),
    "G-LONG": ("2008-03-04", "2026-08-19"),
}
RULE_BLOCK_TUPLE = ("G-P1", "G-P2", "G-P3")
DD_TOLERANCE_FLOAT = 0.02
OWNER_GATE_SHARPE_FLOAT = 1.35
OWNER_GATE_MAX_DD_FLOAT = -0.20

ENGINE_SLIPPAGE_FLOAT = 0.00025
STRESS_SLIPPAGE_FLOAT = 0.00075  # 2.5 bps engine + 5 bps stress, per side
COMMISSION_PER_SHARE_FLOAT = 0.005
COMMISSION_MINIMUM_FLOAT = 1.0
DIVIDEND_NET_RATE_FLOAT = 0.75  # engine: 25% withholding on long dividends
CAPITAL_BASE_FLOAT = 100_000.0
TERMINAL_HAIRCUT_FLOAT = 0.75  # distress-haircut sensitivity (PREREG section 5)

SEED_INT = 20260927
BOOT_N_INT = 2000
BOOT_BLOCK_FLOAT = 21.0
N_TRIALS_INT = 293
CAPACITY_MIN_AUM_FLOAT = 5_000_000.0

ROLE_WEIGHT_DICT = {
    "G3": {"taa": 0.5, "L": 0.5},
    "replacement": {"taa": 0.5, "T": 0.5},
    "addition": {"taa": 0.5, "L": 0.25, "T": 0.25},
}


def log_progress(message_str: str) -> None:
    RESULTS_DIR_PATH.mkdir(parents=True, exist_ok=True)
    stamp_str = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(RESULTS_DIR_PATH / "progress_log.txt", "a", encoding="utf-8") as file_obj:
        file_obj.write(f"{stamp_str} {message_str}\n")
    print(f"[{stamp_str}] {message_str}", flush=True)


def write_json(name_str: str, payload_obj) -> Path:
    RESULTS_DIR_PATH.mkdir(parents=True, exist_ok=True)
    path_obj = RESULTS_DIR_PATH / name_str
    path_obj.write_text(json.dumps(payload_obj, indent=1, default=_json_default), encoding="utf-8")
    return path_obj


def _json_default(value_obj):
    if isinstance(value_obj, (np.integer,)):
        return int(value_obj)
    if isinstance(value_obj, (np.floating,)):
        return float(value_obj)
    if isinstance(value_obj, (np.bool_,)):
        return bool(value_obj)
    if isinstance(value_obj, (pd.Timestamp, dt.datetime, dt.date)):
        return str(value_obj)
    if isinstance(value_obj, np.ndarray):
        return value_obj.tolist()
    return str(value_obj)


# ----------------------------------------------------------------------------------------------------------------------
# metrics
# ----------------------------------------------------------------------------------------------------------------------
def metric_dict(return_ser: pd.Series) -> dict:
    """CAGR (252 d/yr), Sharpe (mean / sd x sqrt 252, no risk-free rate), max drawdown on the daily NAV, Calmar."""
    return_ser = return_ser.dropna()
    if len(return_ser) < 2:
        return {"cagr": float("nan"), "sharpe": float("nan"), "max_dd": float("nan"), "calmar": float("nan"), "n": int(len(return_ser))}
    wealth_vec = np.concatenate([[1.0], (1.0 + return_ser.to_numpy(dtype=float)).cumprod()])
    max_dd_float = float((wealth_vec / np.maximum.accumulate(wealth_vec) - 1.0).min())
    cagr_float = float(wealth_vec[-1] ** (252.0 / len(return_ser)) - 1.0)
    sd_float = float(return_ser.std())
    sharpe_float = float(return_ser.mean() / sd_float * np.sqrt(252.0)) if sd_float > 0 else float("nan")
    return {
        "cagr": cagr_float,
        "sharpe": sharpe_float,
        "max_dd": max_dd_float,
        "calmar": cagr_float / abs(max_dd_float) if max_dd_float < 0 else float("nan"),
        "n": int(len(return_ser)),
    }


def window_metrics(return_ser: pd.Series, block_dict: dict) -> dict:
    return {block_str: metric_dict(return_ser.loc[start_str:end_str]) for block_str, (start_str, end_str) in block_dict.items()}


# ----------------------------------------------------------------------------------------------------------------------
# sleeves
# ----------------------------------------------------------------------------------------------------------------------
def load_taa_ser() -> pd.Series:
    """TAA leg = taa_btal_tqqq daily returns (synthetic proxy before 2012-10-02), 2008-03-04..END."""
    sleeve_df = pd.read_csv(SLEEVE_SERIES_PATH, index_col=0, parse_dates=True)
    taa_ser = sleeve_df["taa_btal_tqqq"].astype(float).loc[:END_TS]
    if taa_ser.isna().any():
        raise RuntimeError("TAA sleeve has missing returns.")
    taa_ser.name = "taa"
    return taa_ser


def load_stored_l_ser() -> pd.Series:
    """The corrected live pod L from the real engine (ndx_atrfix), 2000-01-03..END; NaN before its first fill = cash (0)."""
    momentum_df = pd.read_csv(MOMENTUM_SERIES_PATH, index_col=0, parse_dates=True)
    l_ser = momentum_df["ndx_atrfix"].astype(float).loc[:END_TS].fillna(0.0)
    l_ser.name = "L_stored"
    return l_ser


# ----------------------------------------------------------------------------------------------------------------------
# official pod-model book (verbatim logic of scripts/research/fund_menu_20260923/common.py::book_return_ser)
# ----------------------------------------------------------------------------------------------------------------------
def book_return_ser(
    sleeve_return_df: pd.DataFrame,
    weight_by_alias_dict: dict[str, float],
    rebalance_str: str = "none",
) -> tuple[pd.Series, pd.DataFrame]:
    """Sum of independently compounded pods, optionally reset to target weights.

    Pod model: E_book,t = sum_i E_i,t with E_i,0 = w_i * capital, each pod compounding its own daily return. With
    rebalance "annual", the pod values are reset to w_i * E_book at the last session of each calendar year (a
    frictionless reallocation between pod accounts). Returns the book daily return and the prior-close pod weights.

    *** CRITICAL*** the reset happens AFTER the year-end close is booked, so the first return that uses the new
    weights is the first session of the new year. Copied from the fund-menu study so the two agree by construction.
    """
    alias_list = list(weight_by_alias_dict)
    weight_arr = np.array([weight_by_alias_dict[alias_str] for alias_str in alias_list], dtype=float)
    if abs(weight_arr.sum() - 1.0) > 1e-9:
        raise ValueError("Book weights must sum to 1.")
    window_return_df = sleeve_return_df[alias_list]
    if window_return_df.isna().any().any():
        raise ValueError("Book window has missing sleeve returns; slice to a common window first.")
    return_mat = window_return_df.to_numpy(dtype=float)
    index = window_return_df.index
    pod_value_arr = weight_arr.copy()
    book_return_list = []
    prior_weight_list = []
    for position_int in range(len(index)):
        book_value_before_float = pod_value_arr.sum()
        prior_weight_list.append(pod_value_arr / book_value_before_float)
        pod_value_arr = pod_value_arr * (1.0 + return_mat[position_int])
        book_return_list.append(pod_value_arr.sum() / book_value_before_float - 1.0)
        is_year_end_bool = position_int + 1 < len(index) and index[position_int + 1].year != index[position_int].year
        if rebalance_str == "annual" and is_year_end_bool:
            pod_value_arr = weight_arr * pod_value_arr.sum()
        elif rebalance_str not in ("none", "annual"):
            raise ValueError(f"Unsupported rebalance_str {rebalance_str!r}.")
    book_ser = pd.Series(book_return_list, index=index, name="book")
    prior_weight_df = pd.DataFrame(prior_weight_list, index=index, columns=alias_list)
    return book_ser, prior_weight_df


def book_window_return_ser(
    leg_return_dict: dict[str, pd.Series],
    weight_by_alias_dict: dict[str, float],
    start_str: str,
    end_str: str,
    daily_bool: bool = False,
) -> pd.Series:
    """One book window = its own run from target weights at the window's first session (PREREG section 5).

    daily_bool=True is the sensitivity: weighted sum of same-day returns (the 26 Sep convention).
    """
    frame_df = pd.DataFrame({alias_str: leg_return_dict[alias_str] for alias_str in weight_by_alias_dict})
    frame_df = frame_df.loc[start_str:end_str].dropna()
    if len(frame_df) == 0:
        return pd.Series(dtype=float)
    if daily_bool:
        weight_ser = pd.Series(weight_by_alias_dict)
        return (frame_df * weight_ser).sum(axis=1)
    return book_return_ser(frame_df, weight_by_alias_dict, "annual")[0]


def book_metrics_by_block(
    leg_return_dict: dict[str, pd.Series],
    weight_by_alias_dict: dict[str, float],
    daily_bool: bool = False,
) -> dict:
    return {
        block_str: metric_dict(book_window_return_ser(leg_return_dict, weight_by_alias_dict, start_str, end_str, daily_bool))
        for block_str, (start_str, end_str) in BOOK_BLOCK_DICT.items()
    }
