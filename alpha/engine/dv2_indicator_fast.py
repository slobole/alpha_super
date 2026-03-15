"""
Fast DV2 indicator backed by Numba kernels.

Core formulas
-------------

    HL2_t = (H_t + L_t) / 2

    DV1_t = (C_t / HL2_t) - 1

    DV_t = (DV1_t + DV1_{t-1}) / 2

    DV2_t = 100 * PctRank(DV_{t-L+1}, ..., DV_t)

The implementation preserves the current reference semantics:
- trailing-only windows,
- average rank for ties,
- NaN output until a full valid trailing window exists.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numba import njit


@njit(cache=True)
def _compute_dv_value_arr(
    close_price_arr: np.ndarray,
    high_price_arr: np.ndarray,
    low_price_arr: np.ndarray,
) -> np.ndarray:
    observation_count_int = len(close_price_arr)
    dv1_value_arr = np.empty(observation_count_int, dtype=np.float64)
    dv_value_arr = np.empty(observation_count_int, dtype=np.float64)
    dv1_value_arr[:] = np.nan
    dv_value_arr[:] = np.nan

    for idx_int in range(observation_count_int):
        close_price_float = close_price_arr[idx_int]
        high_price_float = high_price_arr[idx_int]
        low_price_float = low_price_arr[idx_int]

        if np.isnan(close_price_float) or np.isnan(high_price_float) or np.isnan(low_price_float):
            continue

        hl2_price_float = (high_price_float + low_price_float) / 2.0
        if np.isnan(hl2_price_float) or hl2_price_float == 0.0:
            continue

        dv1_value_arr[idx_int] = (close_price_float / hl2_price_float) - 1.0

    for idx_int in range(1, observation_count_int):
        # *** CRITICAL*** DV_t must use only DV1_t and DV1_{t-1}.
        prior_dv1_float = dv1_value_arr[idx_int - 1]
        current_dv1_float = dv1_value_arr[idx_int]
        if np.isnan(prior_dv1_float) or np.isnan(current_dv1_float):
            continue
        dv_value_arr[idx_int] = (prior_dv1_float + current_dv1_float) / 2.0

    return dv_value_arr


@njit(cache=True)
def _rolling_last_percent_rank(value_arr: np.ndarray, window_int: int) -> np.ndarray:
    observation_count_int = len(value_arr)
    percent_rank_arr = np.empty(observation_count_int, dtype=np.float64)
    percent_rank_arr[:] = np.nan

    for end_idx_int in range(window_int - 1, observation_count_int):
        start_idx_int = end_idx_int - window_int + 1
        last_value_float = value_arr[end_idx_int]
        if np.isnan(last_value_float):
            continue

        less_count_int = 0
        equal_count_int = 0
        has_nan_bool = False

        for idx_int in range(start_idx_int, end_idx_int + 1):
            window_value_float = value_arr[idx_int]
            if np.isnan(window_value_float):
                has_nan_bool = True
                break
            if window_value_float < last_value_float:
                less_count_int += 1
            elif window_value_float == last_value_float:
                equal_count_int += 1

        if has_nan_bool:
            continue

        average_rank_float = less_count_int + ((equal_count_int + 1.0) / 2.0)
        percent_rank_arr[end_idx_int] = average_rank_float / window_int

    return percent_rank_arr


def dv2_indicator_fast(
    close_ser: pd.Series,
    high_ser: pd.Series,
    low_ser: pd.Series,
    length_int: int = 126,
) -> pd.Series:
    """
    Fast DV2 implementation.

    Parameters
    ----------
    close_ser : pd.Series
        Close price series.
    high_ser : pd.Series
        High price series.
    low_ser : pd.Series
        Low price series.
    length_int : int, default 126
        Lookback window for the trailing percent rank.
    """
    if length_int <= 0:
        raise ValueError("length_int must be positive.")

    close_price_arr = close_ser.to_numpy(dtype=np.float64, copy=False)
    high_price_arr = high_ser.to_numpy(dtype=np.float64, copy=False)
    low_price_arr = low_ser.to_numpy(dtype=np.float64, copy=False)

    dv_value_arr = _compute_dv_value_arr(close_price_arr, high_price_arr, low_price_arr)
    percent_rank_arr = _rolling_last_percent_rank(dv_value_arr, length_int)
    dv2_value_arr = percent_rank_arr * 100.0
    return pd.Series(dv2_value_arr, index=close_ser.index, name=close_ser.name, dtype=float)
