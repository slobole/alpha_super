"""
Fast QPI indicator backed by Numba kernels.

Core formulas
-------------

    r_t^{(w)} = (C_t / C_{t-w}) - 1

    p_down,t = (1 / L) * sum_{k=t-L+1}^{t} 1[r_k <= 0]

    p_up,t = 1 - p_down,t

    rank_t = PctRank(r_{t-L+1}, ..., r_t)

    QPI_t =
        100 * rank_t / p_down,t         if r_t <= 0
        100 * (1 - rank_t) / p_up,t     if r_t > 0

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
def _pct_change_periods(close_price_arr: np.ndarray, window_int: int) -> np.ndarray:
    observation_count_int = len(close_price_arr)
    return_value_arr = np.empty(observation_count_int, dtype=np.float64)
    return_value_arr[:] = np.nan

    for idx_int in range(window_int, observation_count_int):
        # *** CRITICAL*** r_t^{(w)} must use only C_t and C_{t-w}.
        current_close_float = close_price_arr[idx_int]
        prior_close_float = close_price_arr[idx_int - window_int]
        if np.isnan(current_close_float) or np.isnan(prior_close_float):
            continue
        return_value_arr[idx_int] = (current_close_float / prior_close_float) - 1.0

    return return_value_arr


@njit(cache=True)
def _rolling_rank_and_down_count(
    return_value_arr: np.ndarray,
    lookback_window_int: int,
) -> tuple[np.ndarray, np.ndarray]:
    observation_count_int = len(return_value_arr)
    percent_rank_arr = np.empty(observation_count_int, dtype=np.float64)
    down_count_arr = np.empty(observation_count_int, dtype=np.float64)
    percent_rank_arr[:] = np.nan
    down_count_arr[:] = np.nan

    for end_idx_int in range(lookback_window_int - 1, observation_count_int):
        start_idx_int = end_idx_int - lookback_window_int + 1
        last_return_float = return_value_arr[end_idx_int]
        if np.isnan(last_return_float):
            continue

        less_count_int = 0
        equal_count_int = 0
        down_count_int = 0
        has_nan_bool = False

        for idx_int in range(start_idx_int, end_idx_int + 1):
            window_return_float = return_value_arr[idx_int]
            if np.isnan(window_return_float):
                has_nan_bool = True
                break
            if window_return_float < last_return_float:
                less_count_int += 1
            elif window_return_float == last_return_float:
                equal_count_int += 1
            if window_return_float <= 0.0:
                down_count_int += 1

        if has_nan_bool:
            continue

        average_rank_float = less_count_int + ((equal_count_int + 1.0) / 2.0)
        percent_rank_arr[end_idx_int] = average_rank_float / lookback_window_int
        down_count_arr[end_idx_int] = float(down_count_int)

    return percent_rank_arr, down_count_arr


@njit(cache=True)
def _compute_qpi_value_arr(
    return_value_arr: np.ndarray,
    percent_rank_arr: np.ndarray,
    down_count_arr: np.ndarray,
    lookback_window_int: int,
) -> np.ndarray:
    observation_count_int = len(return_value_arr)
    qpi_value_arr = np.empty(observation_count_int, dtype=np.float64)
    qpi_value_arr[:] = np.nan

    for idx_int in range(observation_count_int):
        current_return_float = return_value_arr[idx_int]
        current_rank_float = percent_rank_arr[idx_int]
        current_down_count_float = down_count_arr[idx_int]
        if (
            np.isnan(current_return_float)
            or np.isnan(current_rank_float)
            or np.isnan(current_down_count_float)
        ):
            continue

        probability_down_float = current_down_count_float / lookback_window_int
        probability_up_float = 1.0 - probability_down_float

        if current_return_float <= 0.0:
            qpi_value_arr[idx_int] = 100.0 * (current_rank_float / probability_down_float)
        else:
            qpi_value_arr[idx_int] = 100.0 * ((1.0 - current_rank_float) / probability_up_float)

    return qpi_value_arr


def qp_indicator_fast(
    close_ser: pd.Series,
    window_int: int = 3,
    lookback_years_int: int = 5,
) -> pd.Series:
    """
    Fast QPI implementation.

    Parameters
    ----------
    close_ser : pd.Series
        Close price series.
    window_int : int, default 3
        Period length for the trailing return.
    lookback_years_int : int, default 5
        Lookback depth, converted to trading days via 252 * years.
    """
    if window_int <= 0:
        raise ValueError("window_int must be positive.")
    if lookback_years_int <= 0:
        raise ValueError("lookback_years_int must be positive.")

    lookback_window_int = lookback_years_int * 252
    close_price_arr = close_ser.to_numpy(dtype=np.float64, copy=False)
    return_value_arr = _pct_change_periods(close_price_arr, window_int)
    percent_rank_arr, down_count_arr = _rolling_rank_and_down_count(
        return_value_arr,
        lookback_window_int,
    )
    qpi_value_arr = _compute_qpi_value_arr(
        return_value_arr,
        percent_rank_arr,
        down_count_arr,
        lookback_window_int,
    )
    return pd.Series(qpi_value_arr, index=close_ser.index, name=close_ser.name, dtype=float)
