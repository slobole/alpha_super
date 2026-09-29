"""Universes for the new-pod search (research only).

- NDX / SP500 reuse the trend study's caches (float64, native Turnover); R1000 is rebuilt here in float64 so that
  engine parity on the Russell 1000 is exact (the trend cache is float32).
- The two disjoint Russell 1000 halves use the S&P 500 cache's PIT membership aligned to the R1000 symbols.
- SH (ProShares Short S&P 500, from 2006-06-21) is appended to a universe as one extra tradeable symbol with
  membership 0, so policies can hold it but never select it.
- Month-end closes from 1990-01-01 (SPX sessions before 1999, the repo schedule after) feed the seasonality scores only.
"""

from __future__ import annotations

import dataclasses
import pickle
import time

import numpy as np
import pandas as pd

from new_pod_search_20260927 import common
from trend_breakout_20260927 import data as trend_data

SH_SYMBOL_STR = "SH"
MONTHLY_HISTORY_START_STR = "1990-01-01"
PANEL_KEY_TUPLE = ("open_arr", "high_arr", "low_arr", "close_arr", "volume_arr", "turnover_arr", "unadjusted_close_arr", "dividend_arr")
BASE_UNIVERSE_DICT = {"NDX": "NDX", "SP500": "SP500", "R1000": "R1000", "R1000_SP": "R1000", "R1000_EX": "R1000"}


def r1000_f64_cache_path():
    return common.CACHE_DIR_PATH / "universe_R1000_f64.pkl"


def prepare_r1000_f64() -> dict:
    """The trend study's prepare_universe with float64 panels for the Russell 1000 (cached in this study's folder)."""
    from data.norgate_loader import CAPITALSPECIAL_ADJUSTMENT_STR, load_price_timeseries
    from strategies.momentum.strategy_mo_atr_normalized_ndx import get_monthly_decision_close_df
    from strategies.momentum.strategy_mo_atr_normalized_ndx_vxn_scaled import DEFAULT_CONFIG, get_vxn_scaled_atr_normalized_ndx_data

    start_float = time.perf_counter()
    config_obj = dataclasses.replace(DEFAULT_CONFIG, indexname_str="Russell 1000")
    pricing_data_df, universe_df, rebalance_schedule_df, vxn_scale_signal_df = get_vxn_scaled_atr_normalized_ndx_data(config_obj, include_total_return_benchmark_bool=False)
    date_index = pd.DatetimeIndex(pricing_data_df.index)
    symbol_list = [str(symbol_str) for symbol_str in universe_df.columns]
    universe_dict = {"universe_str": "R1000", "date_index": date_index, "symbol_list": symbol_list, "panel_dtype_str": "float64"}
    for field_str, key_str in trend_data.PANEL_KEY_DICT.items():
        universe_dict[key_str] = np.column_stack([pricing_data_df[(symbol_str, field_str)].to_numpy(dtype=np.float64) for symbol_str in symbol_list])
    universe_dict["member_arr"] = universe_df.reindex(date_index).fillna(0).to_numpy(dtype=np.int8)
    universe_dict["spy_close_ser"] = pricing_data_df[(config_obj.regime_symbol_str, "Close")].astype(float)
    universe_dict["vxn_close_ser"] = vxn_scale_signal_df["vxn_close"].astype(float)
    universe_dict["repo_schedule_df"] = rebalance_schedule_df.copy()
    universe_dict["month_end_index"] = pd.DatetimeIndex(get_monthly_decision_close_df(pd.DataFrame({"x": universe_dict["close_arr"][:, 0]}, index=date_index)).index)
    qqq_df = load_price_timeseries("QQQ", adjustment_str=CAPITALSPECIAL_ADJUSTMENT_STR, start_date_str=config_obj.history_start_date_str, end_date_str=None)
    universe_dict["qqq_close_ser"] = qqq_df["Close"].astype(float).reindex(date_index)
    del pricing_data_df
    common.CACHE_DIR_PATH.mkdir(parents=True, exist_ok=True)
    with open(r1000_f64_cache_path(), "wb") as file_obj:
        pickle.dump(universe_dict, file_obj, protocol=pickle.HIGHEST_PROTOCOL)
    common.log_progress(f"cache R1000 float64: {len(symbol_list)} symbols, {len(date_index)} sessions, {time.perf_counter() - start_float:.0f}s")
    return universe_dict


def load_base_universe(base_str: str) -> dict:
    if base_str == "R1000":
        with open(r1000_f64_cache_path(), "rb") as file_obj:
            return pickle.load(file_obj)
    return trend_data.load_universe(base_str)


def sp500_member_panel(r1000_dict: dict) -> np.ndarray:
    """PIT S&P 500 membership (from the S&P 500 cache) aligned to the Russell 1000 symbols; 0 for symbols never in it."""
    sp500_dict = trend_data.load_universe("SP500")
    sp_index_dict = {symbol_str: idx for idx, symbol_str in enumerate(sp500_dict["symbol_list"])}
    if not pd.DatetimeIndex(sp500_dict["date_index"]).equals(pd.DatetimeIndex(r1000_dict["date_index"])):
        raise RuntimeError("SP500 and R1000 caches have different date indexes")
    panel_arr = np.zeros(r1000_dict["member_arr"].shape, dtype=np.int8)
    for idx, symbol_str in enumerate(r1000_dict["symbol_list"]):
        if symbol_str in sp_index_dict:
            panel_arr[:, idx] = sp500_dict["member_arr"][:, sp_index_dict[symbol_str]]
    return panel_arr


def load_sh_panel_dict(date_index: pd.DatetimeIndex) -> dict[str, np.ndarray]:
    from data.norgate_loader import CAPITALSPECIAL_ADJUSTMENT_STR, load_price_timeseries

    sh_df = load_price_timeseries(SH_SYMBOL_STR, adjustment_str=CAPITALSPECIAL_ADJUSTMENT_STR, start_date_str="1999-01-01").reindex(date_index)
    return {key_str: sh_df[field_str].to_numpy(dtype=np.float64) for field_str, key_str in trend_data.PANEL_KEY_DICT.items()}


def with_sh(universe_dict: dict) -> dict:
    """Append SH as the last symbol (membership 0). Returns a new dict; panels are copied."""
    if SH_SYMBOL_STR in universe_dict["symbol_list"]:
        return universe_dict
    sh_dict = load_sh_panel_dict(pd.DatetimeIndex(universe_dict["date_index"]))
    out_dict = dict(universe_dict)
    for key_str in PANEL_KEY_TUPLE:
        panel_arr = universe_dict[key_str]
        out_dict[key_str] = np.column_stack([panel_arr, sh_dict[key_str].astype(panel_arr.dtype)])
    out_dict["member_arr"] = np.column_stack([universe_dict["member_arr"], np.zeros(len(universe_dict["date_index"]), dtype=np.int8)])
    out_dict["symbol_list"] = list(universe_dict["symbol_list"]) + [SH_SYMBOL_STR]
    out_dict["sh_idx"] = len(universe_dict["symbol_list"])
    return out_dict


def get_universe(universe_str: str, sh_bool: bool = False) -> dict:
    """NDX | SP500 | R1000 | R1000_SP (R1000 members that are S&P 500 members) | R1000_EX (R1000 members not in it)."""
    base_dict = load_base_universe(BASE_UNIVERSE_DICT[universe_str])
    universe_dict = dict(base_dict)
    universe_dict["universe_str"] = universe_str
    if universe_str in ("R1000_SP", "R1000_EX"):
        sp_arr = sp500_member_panel(base_dict)
        member_arr = base_dict["member_arr"]
        universe_dict["member_arr"] = np.where((member_arr == 1) & ((sp_arr == 1) if universe_str == "R1000_SP" else (sp_arr == 0)), 1, 0).astype(np.int8)
    if sh_bool:
        universe_dict = with_sh(universe_dict)
    return universe_dict


# ----------------------------------------------------------------------------------------------------------------------
# month-end closes from 1990 (seasonality scores only)
# ----------------------------------------------------------------------------------------------------------------------
def monthly_close_1990_path(base_str: str):
    return common.CACHE_DIR_PATH / f"monthly_close_1990_{base_str}.pkl"


def prepare_monthly_closes_1990(base_str: str) -> pd.DataFrame:
    from data.norgate_loader import CAPITALSPECIAL_ADJUSTMENT_STR, load_price_timeseries

    start_float = time.perf_counter()
    universe_dict = load_base_universe(base_str)
    spx_df = load_price_timeseries("$SPX", adjustment_str=CAPITALSPECIAL_ADJUSTMENT_STR, start_date_str=MONTHLY_HISTORY_START_STR)
    early_index = pd.DatetimeIndex(pd.Series(spx_df.index, index=spx_df.index.to_period("M")).groupby(level=0).max().to_numpy())
    cache_month_end_index = pd.DatetimeIndex(universe_dict["month_end_index"])
    early_index = early_index[early_index.to_period("M") < cache_month_end_index[0].to_period("M")]
    # *** CRITICAL *** SPX sessions give the last session of each month before 1999; from 1999 the repo schedule is used.
    month_end_index = early_index.append(cache_month_end_index)
    close_dict = {}
    spy_df = load_price_timeseries("SPY", adjustment_str=CAPITALSPECIAL_ADJUSTMENT_STR, start_date_str=MONTHLY_HISTORY_START_STR)
    close_dict["SPY"] = spy_df["Close"].astype(float).reindex(month_end_index)
    for symbol_str in universe_dict["symbol_list"]:
        price_df = load_price_timeseries(symbol_str, adjustment_str=CAPITALSPECIAL_ADJUSTMENT_STR, start_date_str=MONTHLY_HISTORY_START_STR)
        close_dict[symbol_str] = price_df["Close"].astype(float).reindex(month_end_index) if price_df is not None and len(price_df) else pd.Series(np.nan, index=month_end_index)
    monthly_close_df = pd.DataFrame(close_dict, index=month_end_index)
    monthly_close_df.index.name = "month_end_ts"
    common.CACHE_DIR_PATH.mkdir(parents=True, exist_ok=True)
    monthly_close_df.to_pickle(monthly_close_1990_path(base_str))
    common.log_progress(f"monthly closes 1990 {base_str}: {monthly_close_df.shape[1]} columns x {len(month_end_index)} month-ends {month_end_index[0].date()}..{month_end_index[-1].date()}, {time.perf_counter() - start_float:.0f}s")
    return monthly_close_df


def load_monthly_closes_1990(universe_str: str) -> pd.DataFrame:
    return pd.read_pickle(monthly_close_1990_path(BASE_UNIVERSE_DICT[universe_str]))
