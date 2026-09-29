"""Universes, inputs, GICS groups and the evaluator context for the Alpha101 study (research only).

U1 = the trend study's PIT S&P 500 cache (float64, native Turnover, Unadjusted Close, Dividend, PIT member_arr).
U2 = the new-pod-search Russell 1000 ex S&P 500 (float64, same fields). Inputs follow PREREG section 4:

    returns = Close_t / Close_{t-1} - 1
    vwap    = Turnover / Volume clipped into [Low, High]; (High + Low + Close) / 3 when Turnover or Volume is missing
              or not positive (both counts reported)
    k_t     = Unadjusted Close_t / Close_t (price factor), kv_t = 1 / k_t (volume factor)
    IndClass: last-known GICS from Norgate (8-digit code: sector = 2 digits, industry = 6, sub-industry = 8);
              symbols without a class form one 'unclassified' group per level (not point-in-time; disclosed).
"""

from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd

from alpha101_20260928 import common
from alpha101_20260928.evaluator import Context
from new_pod_search_20260927 import data as nps_data
from trend_breakout_20260927 import data as trend_data

UNIVERSE_SOURCE_DICT = {"U1": "SP500", "U2": "R1000_EX"}
GROUP_LEVEL_DIGITS_DICT = {"sector": 2, "industry": 6, "subindustry": 8}
UNCLASSIFIED_STR = "unclassified"


def load_universe(universe_str: str, sh_bool: bool = False) -> dict:
    """U1 (PIT S&P 500) or U2 (PIT Russell 1000 ex S&P 500); sh_bool appends SH as an extra non-member symbol."""
    source_str = UNIVERSE_SOURCE_DICT[universe_str]
    universe_dict = dict(trend_data.load_universe("SP500")) if source_str == "SP500" else nps_data.get_universe("R1000_EX")
    universe_dict["universe_str"] = universe_str
    if sh_bool:
        universe_dict = nps_data.with_sh(universe_dict)
    return universe_dict


# ----------------------------------------------------------------------------------------------------------------------
# GICS
# ----------------------------------------------------------------------------------------------------------------------
def gics_cache_path(universe_str: str):
    return common.CACHE_DIR_PATH / f"gics_{universe_str}.json"


def fetch_gics_codes(symbol_list: list[str]) -> dict[str, str | None]:
    """Last-known 8-digit GICS sub-industry code per symbol from Norgate (None when unclassified)."""
    import norgatedata

    logging.disable(logging.CRITICAL)
    code_dict: dict[str, str | None] = {}
    for symbol_str in symbol_list:
        try:
            code_obj = norgatedata.classification_at_level(symbol_str, "GICS", "ClassificationId", 4)
        except Exception:  # noqa: BLE001 - Norgate raises on unknown symbols
            code_obj = None
        code_str = str(code_obj) if code_obj is not None else None
        code_dict[symbol_str] = code_str if code_str is not None and len(code_str) == 8 and code_str.isdigit() else None
    logging.disable(logging.NOTSET)
    return code_dict


def load_gics_codes(universe_str: str, symbol_list: list[str]) -> dict[str, str | None]:
    path_obj = gics_cache_path(universe_str)
    if path_obj.exists():
        cached_dict = json.loads(path_obj.read_text(encoding="utf-8"))
        if all(symbol_str in cached_dict for symbol_str in symbol_list):
            return {symbol_str: cached_dict[symbol_str] for symbol_str in symbol_list}
    code_dict = fetch_gics_codes(symbol_list)
    common.CACHE_DIR_PATH.mkdir(parents=True, exist_ok=True)
    path_obj.write_text(json.dumps(code_dict, indent=0), encoding="utf-8")
    return code_dict


def group_codes(symbol_list: list[str], code_dict: dict[str, str | None]) -> tuple[dict[str, np.ndarray], dict]:
    """Integer group codes per level (0..G-1 per symbol) and a small coverage summary."""
    group_dict: dict[str, np.ndarray] = {}
    summary_dict: dict = {"symbols_int": len(symbol_list), "unclassified_int": int(sum(1 for s in symbol_list if code_dict.get(s) is None))}
    for level_str, digits_int in GROUP_LEVEL_DIGITS_DICT.items():
        label_list = [code_dict[s][:digits_int] if code_dict.get(s) is not None else UNCLASSIFIED_STR for s in symbol_list]
        unique_list = sorted(set(label_list))
        index_dict = {label_str: idx for idx, label_str in enumerate(unique_list)}
        group_dict[level_str] = np.array([index_dict[label_str] for label_str in label_list], dtype=np.int64)
        summary_dict[f"groups_{level_str}_int"] = len(unique_list)
    return group_dict, summary_dict


# ----------------------------------------------------------------------------------------------------------------------
# inputs and the evaluator context
# ----------------------------------------------------------------------------------------------------------------------
def input_panels(universe_dict: dict) -> tuple[dict, dict]:
    """open/high/low/close/volume/vwap/returns panels (float64) plus the vwap source counts."""
    open_arr = universe_dict["open_arr"].astype(np.float64)
    high_arr = universe_dict["high_arr"].astype(np.float64)
    low_arr = universe_dict["low_arr"].astype(np.float64)
    close_arr = universe_dict["close_arr"].astype(np.float64)
    volume_arr = universe_dict["volume_arr"].astype(np.float64)
    turnover_arr = universe_dict["turnover_arr"].astype(np.float64)
    with np.errstate(all="ignore"):
        returns_arr = np.full_like(close_arr, np.nan)
        returns_arr[1:] = close_arr[1:] / close_arr[:-1] - 1.0
        ok_arr = np.isfinite(turnover_arr) & (turnover_arr > 0) & np.isfinite(volume_arr) & (volume_arr > 0)
        raw_vwap_arr = np.where(ok_arr, turnover_arr / np.where(ok_arr, volume_arr, 1.0), np.nan)
        clipped_arr = np.minimum(np.maximum(raw_vwap_arr, low_arr), high_arr)
        typical_arr = (high_arr + low_arr + close_arr) / 3.0
        # *** CRITICAL *** Turnover / Volume is already in the adjusted price basis (Turnover is nominal dollars and the
        # CAPITALSPECIAL Volume carries the inverse price factor), so it is comparable with Low / High and Close.
        vwap_arr = np.where(ok_arr, clipped_arr, typical_arr)
        finite_close_arr = np.isfinite(close_arr)
    count_dict = {
        "finite_close_cells_int": int(finite_close_arr.sum()),
        "vwap_from_turnover_int": int((ok_arr & finite_close_arr).sum()),
        "vwap_typical_price_fallback_int": int((~ok_arr & finite_close_arr).sum()),
        "vwap_clipped_to_low_high_int": int((ok_arr & finite_close_arr & (raw_vwap_arr != clipped_arr)).sum()),
    }
    panel_dict ={"open": open_arr, "high": high_arr, "low": low_arr, "close": close_arr, "volume": volume_arr, "vwap": vwap_arr, "returns": returns_arr}
    for key_str, panel_arr in panel_dict.items():
        bad_arr = ~np.isfinite(panel_arr) & ~np.isnan(panel_arr)
        if bad_arr.any():
            panel_arr[bad_arr] = np.nan
    return panel_dict, count_dict


def price_factor(universe_dict: dict) -> np.ndarray:
    with np.errstate(all="ignore"):
        k_arr = universe_dict["unadjusted_close_arr"].astype(np.float64) / universe_dict["close_arr"].astype(np.float64)
    return np.where(np.isfinite(k_arr) & (k_arr > 0), k_arr, np.nan)


def build_context(universe_dict: dict, code_dict: dict[str, str | None] | None = None) -> tuple[Context, dict]:
    symbol_list = list(universe_dict["symbol_list"])
    if code_dict is None:
        code_dict = load_gics_codes(universe_dict["universe_str"], symbol_list)
    group_dict, group_summary_dict = group_codes(symbol_list, code_dict)
    panel_dict, count_dict = input_panels(universe_dict)
    member_arr = universe_dict["member_arr"] == 1
    context = Context(panel_dict, member_arr, price_factor(universe_dict), group_dict)
    return context, {"vwap": count_dict, "gics": group_summary_dict}


def forward_return_panel(open_arr: np.ndarray) -> np.ndarray:
    """fr_t = Open_{t+2} / Open_{t+1} - 1 (the next-open to next-next-open return of a decision at the close of t)."""
    open_arr = np.asarray(open_arr, dtype=np.float64)
    out_arr = np.full_like(open_arr, np.nan)
    with np.errstate(all="ignore"):
        out_arr[:-2] = open_arr[2:] / open_arr[1:-1] - 1.0
    out_arr[~np.isfinite(out_arr)] = np.nan
    return out_arr


def overnight_panel(open_arr: np.ndarray, close_arr: np.ndarray) -> np.ndarray:
    """Open_{t+1} / Close_t - 1: the part a next-open fill cannot capture."""
    out_arr = np.full(np.asarray(close_arr).shape, np.nan)
    with np.errstate(all="ignore"):
        out_arr[:-1] = np.asarray(open_arr)[1:] / np.asarray(close_arr)[:-1] - 1.0
    out_arr[~np.isfinite(out_arr)] = np.nan
    return out_arr


def date_index_of(universe_dict: dict) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(universe_dict["date_index"])
