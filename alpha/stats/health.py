"""Pod-health statistics: is a live or shadow pod still behaving like its expected process?

Two detectors, used together (a red on either one is a red for the pod):

1. Cold Blood Index, block version (after Lotter, Financial Hacker, 2015).
   The pod has been in a drawdown of depth D for L sessions, after M sessions of
   live or shadow trading (M >= L). The index is the probability that the
   expected return process produces, somewhere in M sessions, a drawdown of at
   least D that is reached within at most L sessions of its peak:

       equity_t        = Π_{s<=t} (1 + r_s),  equity_0 = 1
       dd_within_L(t)  = equity_t / max(equity_{t−L}, ..., equity_t) − 1
       CBI             = P( min_t dd_within_L(t) <= −D )  over M-session paths

   Paths come from a stationary block bootstrap of the expected daily returns
   (normally the worst-rebalance-offset backtest). Lotter counts overlapping
   backtest windows as independent samples, which overstates confidence; the
   bootstrap does not. A small CBI means the current drawdown is rare for the
   process we believe we are running.

2. Lower one-sided CUSUM on standardised monthly returns (mean-shift detector):

       z_t = (r_t − μ) / σ           μ, σ from the expected monthly returns
       S_0 = 0,  S_t = min(0, S_{t−1} + z_t + k)
       alarm when S_t <= −h

   k (default 0.5) sets the smallest shift worth detecting (about 2k σ); h is
   calibrated by bootstrap so that a pod that IS behaving as expected alarms in a
   12-month window with probability `false_alarm_float` (default 5%).

Monitoring caveats (from the P0 review; P1 must calibrate on the procedure):
- `cold_blood_index` is a single-evaluation probability. Re-evaluating it every
  day on the drawdown that the data chose makes a fixed "RED below 5%" cut far too
  trigger-happy (a healthy pod goes RED at least once in a year about a third of
  the time in simulation, most often early when M is small). Alert thresholds must
  be set by simulating the whole daily monitoring procedure on healthy paths.
- The CUSUM alarm is the minimum over the whole history, so it latches: with a 5%
  per-12-month rate, a healthy pod's chance of EVER alarming grows with time
  (about 9% by 24 months, 24% by 60 months). The card must say so.
- The calibrated h moves by about ±10% with the block length and the sample
  autocorrelation of the expected returns.

These are report-only statistics. Retiring a pod stays the owner's decision.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import maximum_filter1d

from alpha.stats.bootstrap import stationary_bootstrap_path_mat


@dataclass(frozen=True)
class DrawdownState:
    depth_float: float
    length_int: int


def current_drawdown_state(return_vec) -> DrawdownState:
    """Depth (positive fraction) and sessions since the last equity peak.

    The starting equity 1.0 counts as a peak, so a pod that has lost from day one
    is in a drawdown whose length is the whole history.
    """
    return_arr = np.asarray(return_vec, dtype=float)
    if return_arr.size == 0 or not np.all(np.isfinite(return_arr)):
        raise ValueError("return_vec must be a non-empty finite series.")
    equity_arr = np.concatenate(([1.0], np.cumprod(1.0 + return_arr)))
    peak_idx_int = int(np.flatnonzero(equity_arr >= np.max(equity_arr))[-1])
    depth_float = 1.0 - equity_arr[-1] / equity_arr[peak_idx_int]
    return DrawdownState(depth_float=float(depth_float), length_int=int(equity_arr.size - 1 - peak_idx_int))


def _min_windowed_drawdown_vec(return_path_mat: np.ndarray, window_int: int) -> np.ndarray:
    """For each path: the worst drawdown measured from the highest equity of the trailing window_int sessions."""
    path_count_int = return_path_mat.shape[0]
    equity_mat = np.concatenate((np.ones((path_count_int, 1)), np.cumprod(1.0 + return_path_mat, axis=1)), axis=1)
    filter_size_int = window_int + 1
    # *** CRITICAL*** trailing window [t − window_int, t]: origin shifts the filter so it
    # covers only the current and earlier columns; mode="nearest" repeats equity_0 = 1.
    trailing_max_mat = maximum_filter1d(
        equity_mat, size=filter_size_int, axis=1, mode="nearest", origin=(filter_size_int - 1) - filter_size_int // 2
    )
    return np.min(equity_mat / trailing_max_mat - 1.0, axis=1)


def cold_blood_index(
    expected_return_vec,
    drawdown_depth_float: float,
    drawdown_length_int: int,
    observation_length_int: int,
    path_count_int: int = 5000,
    mean_block_length_float: float = 20.0,
    random_seed_int: int = 0,
) -> float:
    """Probability of a drawdown at least this deep and this fast within the observation period."""
    if drawdown_depth_float <= 0.0:
        return 1.0
    if drawdown_length_int < 1 or observation_length_int < drawdown_length_int:
        raise ValueError("Need 1 <= drawdown_length_int <= observation_length_int.")
    return_path_mat = stationary_bootstrap_path_mat(
        expected_return_vec,
        path_count_int=path_count_int,
        mean_block_length_float=mean_block_length_float,
        path_length_int=observation_length_int,
        random_seed_int=random_seed_int,
    )
    worst_drawdown_vec = _min_windowed_drawdown_vec(return_path_mat, drawdown_length_int)
    return float(np.mean(worst_drawdown_vec <= -drawdown_depth_float))


def lower_cusum_path(standardized_vec, reference_k_float: float = 0.5) -> np.ndarray:
    """S_t = min(0, S_{t−1} + z_t + k), S_0 = 0. Returns S_1..S_T."""
    standardized_arr = np.asarray(standardized_vec, dtype=float)
    if not np.all(np.isfinite(standardized_arr)):
        raise ValueError("standardized_vec must be finite.")
    cusum_arr = np.empty(standardized_arr.size)
    running_float = 0.0
    for step_idx_int, z_float in enumerate(standardized_arr):
        running_float = min(0.0, running_float + z_float + reference_k_float)
        cusum_arr[step_idx_int] = running_float
    return cusum_arr


@dataclass(frozen=True)
class CusumCalibration:
    threshold_h_float: float
    reference_k_float: float
    expected_mean_float: float
    expected_std_float: float
    horizon_periods_int: int
    false_alarm_float: float


def calibrate_cusum_threshold(
    expected_period_return_vec,
    reference_k_float: float = 0.5,
    horizon_periods_int: int = 12,
    false_alarm_float: float = 0.05,
    path_count_int: int = 20000,
    mean_block_length_float: float = 3.0,
    random_seed_int: int = 0,
) -> CusumCalibration:
    """Choose h so that P(alarm within `horizon_periods_int`) = `false_alarm_float` under the expected process."""
    expected_arr = np.asarray(expected_period_return_vec, dtype=float)
    expected_mean_float = float(expected_arr.mean())
    expected_std_float = float(expected_arr.std(ddof=1))
    if expected_std_float <= 0.0:
        raise ValueError("Expected returns have zero dispersion.")

    path_mat = stationary_bootstrap_path_mat(
        expected_arr,
        path_count_int=path_count_int,
        mean_block_length_float=mean_block_length_float,
        path_length_int=horizon_periods_int,
        random_seed_int=random_seed_int,
    )
    standardized_mat = (path_mat - expected_mean_float) / expected_std_float
    running_vec = np.zeros(path_count_int)
    worst_vec = np.zeros(path_count_int)
    for step_idx_int in range(horizon_periods_int):
        running_vec = np.minimum(0.0, running_vec + standardized_mat[:, step_idx_int] + reference_k_float)
        worst_vec = np.minimum(worst_vec, running_vec)

    threshold_h_float = float(np.quantile(-worst_vec, 1.0 - false_alarm_float))
    if threshold_h_float <= 0.0:
        raise ValueError("Calibrated threshold is not positive; lower reference_k_float or lengthen the horizon.")
    return CusumCalibration(
        threshold_h_float=threshold_h_float,
        reference_k_float=reference_k_float,
        expected_mean_float=expected_mean_float,
        expected_std_float=expected_std_float,
        horizon_periods_int=horizon_periods_int,
        false_alarm_float=false_alarm_float,
    )


def cusum_alarm_bool(realized_period_return_vec, calibration: CusumCalibration) -> bool:
    """True when the lower CUSUM of the realised returns has crossed −h."""
    realized_arr = np.asarray(realized_period_return_vec, dtype=float)
    if realized_arr.size == 0:
        return False
    if not np.all(np.isfinite(realized_arr)):
        # A NaN would reset the running sum to 0 (min(0, nan) is 0 in Python) and erase a near-alarm.
        raise ValueError("Realised returns must be finite; fill or drop missing periods explicitly.")
    standardized_arr = (realized_arr - calibration.expected_mean_float) / calibration.expected_std_float
    cusum_arr = lower_cusum_path(standardized_arr, calibration.reference_k_float)
    return bool(np.min(cusum_arr) <= -calibration.threshold_h_float)
