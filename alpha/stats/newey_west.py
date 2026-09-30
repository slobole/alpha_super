"""Newey–West (HAC) t-statistic for the mean of a time series.

Used where observations overlap or cluster in time, for example the daily mean
excess return of 5-day event holds: consecutive days share four of five return
days, so an i.i.d. standard error is far too small.

For a series x_1..x_T with mean x̄ and lag L (Bartlett kernel):

    γ_j      = (1/T) · Σ_{t=j+1..T} (x_t − x̄)(x_{t−j} − x̄)
    w_j      = 1 − j / (L + 1)
    Var(x̄)   = (γ_0 + 2 · Σ_{j=1..L} w_j · γ_j) / T
    t        = x̄ / sqrt(Var(x̄))

This is the estimator statsmodels reports for OLS on a constant with
cov_type="HAC", kernel Bartlett, use_correction=False.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class NeweyWestResult:
    mean_float: float
    standard_error_float: float
    t_stat_float: float
    observation_count_int: int
    lag_int: int


def newey_west_mean_t_stat(value_vec, lag_int: int) -> NeweyWestResult:
    """Mean, HAC standard error and t-statistic of a 1-D series.

    NaN values are dropped first, so the series must already be in time order
    with gaps that are acceptable to close (for example, dates with no events).
    """
    value_arr = np.asarray(value_vec, dtype=float)
    value_arr = value_arr[np.isfinite(value_arr)]
    observation_count_int = int(value_arr.size)
    if lag_int < 0:
        raise ValueError("lag_int must be >= 0.")
    if observation_count_int < 2:
        raise ValueError("Need at least two finite observations.")
    if lag_int >= observation_count_int:
        raise ValueError("lag_int must be smaller than the number of observations.")

    mean_float = float(value_arr.mean())
    demeaned_arr = value_arr - mean_float

    long_run_variance_float = float(demeaned_arr @ demeaned_arr) / observation_count_int
    for lag_idx_int in range(1, lag_int + 1):
        autocovariance_float = (
            float(demeaned_arr[lag_idx_int:] @ demeaned_arr[:-lag_idx_int]) / observation_count_int
        )
        bartlett_weight_float = 1.0 - lag_idx_int / (lag_int + 1.0)
        long_run_variance_float += 2.0 * bartlett_weight_float * autocovariance_float

    if long_run_variance_float <= 0.0:
        raise ValueError("Non-positive long-run variance; the series is degenerate.")

    standard_error_float = float(np.sqrt(long_run_variance_float / observation_count_int))
    return NeweyWestResult(
        mean_float=mean_float,
        standard_error_float=standard_error_float,
        t_stat_float=mean_float / standard_error_float,
        observation_count_int=observation_count_int,
        lag_int=int(lag_int),
    )
