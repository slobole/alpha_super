"""Universe caches with native Turnover, and the extended monthly closes for family C (research only).

PREREG section 3: universes are loaded through the repo's own NDX data function with the index name changed, exactly
as the 26 Sep study did, plus the native `Turnover` field. NDX and SP500 are stored in float64 (engine parity needs
the engine's own float64 prices); R1000 in float32 (memory).
"""

from __future__ import annotations

import dataclasses
import pickle
import time

import numpy as np
import pandas as pd

from trend_breakout_20260927 import common

import ndx_param_robustness_core as core  # noqa: E402  (26 Sep replica, imported only)

PANEL_FIELD_TUPLE = ("Open", "High", "Low", "Close", "Volume", "Turnover", "Unadjusted Close", "Dividend")
PANEL_KEY_DICT = {
    "Open": "open_arr",
    "High": "high_arr",
    "Low": "low_arr",
    "Close": "close_arr",
    "Volume": "volume_arr",
    "Turnover": "turnover_arr",
    "Unadjusted Close": "unadjusted_close_arr",
    "Dividend": "dividend_arr",
}
MONTHLY_HISTORY_START_STR = "1996-01-01"  # family C regression inputs only (PREREG section 4)


def universe_cache_path(universe_str: str):
    return common.CACHE_DIR_PATH / f"universe_{universe_str}.pkl"


def monthly_close_cache_path(universe_str: str):
    return common.CACHE_DIR_PATH / f"monthly_close_ext_{universe_str}.pkl"


def prepare_universe(universe_str: str) -> dict:
    """Load one PIT universe through the repo's NDX data function (only the index name changes) and cache the panels."""
    from data.norgate_loader import CAPITALSPECIAL_ADJUSTMENT_STR, load_price_timeseries
    from strategies.momentum.strategy_mo_atr_normalized_ndx import get_monthly_decision_close_df
    from strategies.momentum.strategy_mo_atr_normalized_ndx_vxn_scaled import (
        DEFAULT_CONFIG,
        get_vxn_scaled_atr_normalized_ndx_data,
    )

    start_float = time.perf_counter()
    config_obj = dataclasses.replace(DEFAULT_CONFIG, indexname_str=common.UNIVERSE_INDEXNAME_DICT[universe_str])
    pricing_data_df, universe_df, rebalance_schedule_df, vxn_scale_signal_df = get_vxn_scaled_atr_normalized_ndx_data(
        config_obj,
        include_total_return_benchmark_bool=False,
    )
    date_index = pd.DatetimeIndex(pricing_data_df.index)
    symbol_list = [str(symbol_str) for symbol_str in universe_df.columns]
    panel_dtype_obj = np.float32 if universe_str == "R1000" else np.float64

    def panel_arr(field_str: str) -> np.ndarray:
        return np.column_stack(
            [pricing_data_df[(symbol_str, field_str)].to_numpy(dtype=np.float64) for symbol_str in symbol_list]
        ).astype(panel_dtype_obj)

    universe_dict = {
        "universe_str": universe_str,
        "date_index": date_index,
        "symbol_list": symbol_list,
        "panel_dtype_str": str(np.dtype(panel_dtype_obj)),
    }
    for field_str in PANEL_FIELD_TUPLE:
        universe_dict[PANEL_KEY_DICT[field_str]] = panel_arr(field_str)
    # *** CRITICAL *** audited PIT membership is forward-filled onto the price index (never back-filled): row t is the
    # latest constituent row on or before t, which is exactly the engine's as-of lookup.
    universe_dict["member_arr"] = universe_df.reindex(date_index).fillna(0).to_numpy(dtype=np.int8)
    universe_dict["spy_close_ser"] = pricing_data_df[(config_obj.regime_symbol_str, "Close")].astype(float)
    universe_dict["vxn_close_ser"] = vxn_scale_signal_df["vxn_close"].astype(float)
    universe_dict["repo_schedule_df"] = rebalance_schedule_df.copy()
    universe_dict["month_end_index"] = pd.DatetimeIndex(
        get_monthly_decision_close_df(pd.DataFrame({"x": universe_dict["close_arr"][:, 0]}, index=date_index)).index
    )
    qqq_df = load_price_timeseries(
        "QQQ", adjustment_str=CAPITALSPECIAL_ADJUSTMENT_STR, start_date_str=config_obj.history_start_date_str, end_date_str=None
    )
    universe_dict["qqq_close_ser"] = qqq_df["Close"].astype(float).reindex(date_index)
    del pricing_data_df
    common.CACHE_DIR_PATH.mkdir(parents=True, exist_ok=True)
    with open(universe_cache_path(universe_str), "wb") as file_obj:
        pickle.dump(universe_dict, file_obj, protocol=pickle.HIGHEST_PROTOCOL)
    common.log_progress(
        f"cache {universe_str}: {len(symbol_list)} symbols, {len(date_index)} sessions "
        f"{date_index[0].date()}..{date_index[-1].date()}, dtype {universe_dict['panel_dtype_str']}, "
        f"{time.perf_counter() - start_float:.0f}s"
    )
    return universe_dict


def load_universe(universe_str: str) -> dict:
    with open(universe_cache_path(universe_str), "rb") as file_obj:
        return pickle.load(file_obj)


def prepare_monthly_closes(universe_str: str) -> pd.DataFrame:
    """Month-end CAPITALSPECIAL closes of the universe's symbols and SPY from 1996-01-01 (family C regression inputs).

    Months before 1999-01 take the last SPY session of the month; months from 1999-01 use the cache's month-end index
    (the repo schedule), so the two schedules agree where they overlap.
    """
    from data.norgate_loader import CAPITALSPECIAL_ADJUSTMENT_STR, load_price_timeseries

    start_float = time.perf_counter()
    universe_dict = load_universe(universe_str)
    spy_df = load_price_timeseries("SPY", adjustment_str=CAPITALSPECIAL_ADJUSTMENT_STR, start_date_str=MONTHLY_HISTORY_START_STR)
    spy_close_ser = spy_df["Close"].astype(float)
    early_month_end_index = pd.DatetimeIndex(
        pd.Series(spy_close_ser.index, index=spy_close_ser.index.to_period("M")).groupby(level=0).max().to_numpy()
    )
    cache_month_end_index = pd.DatetimeIndex(universe_dict["month_end_index"])
    first_cache_period = cache_month_end_index[0].to_period("M")
    early_month_end_index = early_month_end_index[early_month_end_index.to_period("M") < first_cache_period]
    month_end_index = early_month_end_index.append(cache_month_end_index)

    close_dict = {"SPY": spy_close_ser.reindex(month_end_index)}
    missing_list = []
    for symbol_str in universe_dict["symbol_list"]:
        price_df = load_price_timeseries(symbol_str, adjustment_str=CAPITALSPECIAL_ADJUSTMENT_STR, start_date_str=MONTHLY_HISTORY_START_STR)
        if price_df is None or len(price_df) == 0:
            missing_list.append(symbol_str)
            close_dict[symbol_str] = pd.Series(np.nan, index=month_end_index)
            continue
        close_dict[symbol_str] = price_df["Close"].astype(float).reindex(month_end_index)
    monthly_close_df = pd.DataFrame(close_dict, index=month_end_index)
    monthly_close_df.index.name = "month_end_ts"
    common.CACHE_DIR_PATH.mkdir(parents=True, exist_ok=True)
    monthly_close_df.to_pickle(monthly_close_cache_path(universe_str))
    common.log_progress(
        f"monthly closes {universe_str}: {monthly_close_df.shape[1]} columns x {len(month_end_index)} month-ends "
        f"{month_end_index[0].date()}..{month_end_index[-1].date()}, {len(missing_list)} symbols without data, "
        f"{time.perf_counter() - start_float:.0f}s"
    )
    return monthly_close_df


def load_monthly_closes(universe_str: str) -> pd.DataFrame:
    return pd.read_pickle(monthly_close_cache_path(universe_str))
