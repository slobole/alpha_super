"""Daily pod-health monitoring with thresholds calibrated on the monitoring procedure itself.

Why this module exists: `health.cold_blood_index` answers "how rare is THIS
drawdown for the expected process?" once. A monitor asks that question every
day, on whatever drawdown the data happened to produce. Asked every day with a
fixed "RED below 5%" cut, a healthy pod goes RED at least once in a year about a
third of the time (P0 review). So the cut must come from simulating the whole
monitor on healthy pods:

    for each simulated healthy pod path (bootstrap of the expected returns):
        for each session M from `min_observation_int` to `horizon_int`:
            D_M, L_M = depth and age of the current drawdown
            CBI_M    = P_expected( a drawdown >= D_M, reached within L_M sessions,
                                   somewhere in M sessions )
        min_CBI = min over M of CBI_M
    RED   = the 5% quantile of min_CBI    ->  P(healthy pod ever RED in horizon)   <= 5%
    AMBER = the 15% quantile of min_CBI   ->  P(healthy pod ever AMBER in horizon) <= 15%

A pod is RED on a day when CBI < RED (strictly), AMBER when CBI < AMBER.

Speed: CBI is looked up in a precomputed table instead of re-simulated. For a
grid of window lengths L and checkpoints M, the table stores, for every
reference path, the worst L-window drawdown seen up to M. Lookups round L UP to
the next grid length and M DOWN to the previous checkpoint:
- rounding L up can only raise CBI (a longer window finds more drawdowns), which
  makes alarms slightly LESS likely;
- rounding M down can only lower CBI by at most one checkpoint step of history,
  which makes alarms slightly MORE likely.
Both are the same in calibration and in use, so the calibrated false-alarm rate
holds for the rounded procedure that actually runs.

Resolution: with P reference paths the smallest non-zero CBI is 1/P. A RED cut
that rests on fewer than 5 reference paths is table noise, so calibration refuses
it and asks for a larger table instead.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import maximum_filter1d

from alpha.stats.bootstrap import stationary_bootstrap_path_mat

MIN_TABLE_HITS_INT = 5
# Reference and monitored paths are resampled from the same sample, so exact ties (the same worst day,
# the same copied block) are common. Different cumulative products and float32 storage move a tie by
# ~1e-9, which would turn "as bad as a reference path" into "worse than every path" (CBI = 0).
# A drawdown within this tolerance of a reference drawdown counts as a tie (counted as a hit).
TIE_TOLERANCE_FLOAT = 1e-7
# Simulated live paths (calibration, detection power) are bootstrap copies of the same sample as the
# reference paths, so they would tie with reference drawdowns far more often than real, continuous live
# returns ever do; ties count as hits and would make simulated pods look healthier than live ones (P1
# review: TAA RED false alarms 5% on exact copies vs 10% on jittered ones). A tiny relative jitter makes
# the simulated paths as tie-free as live data without changing their distribution.
SIMULATION_JITTER_FLOAT = 1e-4
DEFAULT_WINDOW_GRID_TUPLE = (1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597)


def drawdown_state_mat(return_path_mat) -> tuple[np.ndarray, np.ndarray]:
    """Depth and age (sessions since the last peak) of the drawdown after each session, for every path row.

    The starting equity 1.0 counts as a peak. Ties with the running peak reset the age.
    """
    return_arr = np.atleast_2d(np.asarray(return_path_mat, dtype=float))
    if return_arr.shape[1] == 0 or not np.all(np.isfinite(return_arr)):
        raise ValueError("Returns must be a non-empty finite series.")
    path_count_int = return_arr.shape[0]
    equity_mat = np.concatenate((np.ones((path_count_int, 1)), np.cumprod(1.0 + return_arr, axis=1)), axis=1)
    running_peak_mat = np.maximum.accumulate(equity_mat, axis=1)
    step_mat = np.broadcast_to(np.arange(equity_mat.shape[1]), equity_mat.shape)
    last_peak_step_mat = np.maximum.accumulate(np.where(equity_mat >= running_peak_mat, step_mat, 0), axis=1)
    depth_mat = 1.0 - equity_mat / running_peak_mat
    age_mat = step_mat - last_peak_step_mat
    return depth_mat[:, 1:], age_mat[:, 1:]


def drawdown_state_path(return_vec) -> tuple[np.ndarray, np.ndarray]:
    """Single-path version of `drawdown_state_mat`."""
    depth_mat, age_mat = drawdown_state_mat(np.asarray(return_vec, dtype=float)[None, :])
    return depth_mat[0], age_mat[0]


@dataclass(frozen=True)
class CbiTable:
    window_grid_vec: np.ndarray  # L values
    checkpoint_vec: np.ndarray  # M values (sessions)
    sorted_worst_mat: np.ndarray  # [window, checkpoint, path], ascending along paths
    path_count_int: int
    mean_block_length_float: float

    def probability(self, depth_float: float, age_int: int, observation_int: int) -> float:
        """CBI for a drawdown of depth D and age L after M sessions (see module docstring for rounding)."""
        if depth_float <= 0.0:
            return 1.0
        window_idx_int = min(int(np.searchsorted(self.window_grid_vec, max(age_int, 1), side="left")), self.window_grid_vec.size - 1)
        if observation_int > int(self.checkpoint_vec[-1]):
            raise ValueError(f"Observation length {observation_int} exceeds the table length {int(self.checkpoint_vec[-1])}.")
        checkpoint_idx_int = int(np.searchsorted(self.checkpoint_vec, observation_int, side="right")) - 1
        checkpoint_idx_int = max(checkpoint_idx_int, 0)
        worst_vec = self.sorted_worst_mat[window_idx_int, checkpoint_idx_int]
        hit_count_int = int(np.searchsorted(worst_vec, -depth_float + TIE_TOLERANCE_FLOAT, side="right"))
        return hit_count_int / self.path_count_int


def build_cbi_table(
    expected_return_vec,
    max_observation_int: int,
    path_count_int: int = 2000,
    mean_block_length_float: float = 20.0,
    checkpoint_step_int: int = 5,
    random_seed_int: int = 0,
    window_grid_tuple: tuple[int, ...] = DEFAULT_WINDOW_GRID_TUPLE,
) -> CbiTable:
    window_list = [window_int for window_int in window_grid_tuple if window_int < max_observation_int]
    window_list.append(max_observation_int)
    checkpoint_vec = np.arange(checkpoint_step_int, max_observation_int + 1, checkpoint_step_int)
    if checkpoint_vec.size == 0 or checkpoint_vec[-1] != max_observation_int:
        checkpoint_vec = np.append(checkpoint_vec, max_observation_int)

    return_path_mat = stationary_bootstrap_path_mat(
        expected_return_vec,
        path_count_int=path_count_int,
        mean_block_length_float=mean_block_length_float,
        path_length_int=max_observation_int,
        random_seed_int=random_seed_int,
    )
    equity_mat = np.concatenate((np.ones((path_count_int, 1)), np.cumprod(1.0 + return_path_mat, axis=1)), axis=1)

    sorted_worst_mat = np.empty((len(window_list), checkpoint_vec.size, path_count_int), dtype=np.float32)
    for window_idx_int, window_int in enumerate(window_list):
        filter_size_int = window_int + 1
        # *** CRITICAL*** trailing window [t − L, t] only (same filter as health._min_windowed_drawdown_vec).
        trailing_max_mat = maximum_filter1d(
            equity_mat, size=filter_size_int, axis=1, mode="nearest", origin=(filter_size_int - 1) - filter_size_int // 2
        )
        worst_so_far_mat = np.minimum.accumulate(equity_mat / trailing_max_mat - 1.0, axis=1)
        sorted_worst_mat[window_idx_int] = np.sort(worst_so_far_mat[:, checkpoint_vec].T, axis=1)

    return CbiTable(
        window_grid_vec=np.array(window_list),
        checkpoint_vec=checkpoint_vec,
        sorted_worst_mat=sorted_worst_mat,
        path_count_int=path_count_int,
        mean_block_length_float=mean_block_length_float,
    )


def simulated_live_path_mat(
    expected_return_vec,
    path_count_int: int,
    mean_block_length_float: float,
    path_length_int: int,
    random_seed_int: int,
    daily_drift_shift_float: float = 0.0,
) -> np.ndarray:
    """Bootstrap paths of the expected process, shifted by a daily drift, with a tie-breaking jitter.

        r_sim = r_boot * (1 + j * eps) + shift,   eps ~ N(0, 1),  j = SIMULATION_JITTER_FLOAT
    """
    path_mat = stationary_bootstrap_path_mat(
        expected_return_vec,
        path_count_int=path_count_int,
        mean_block_length_float=mean_block_length_float,
        path_length_int=path_length_int,
        random_seed_int=random_seed_int,
    )
    jitter_mat = np.random.default_rng(random_seed_int + 1_000_003).standard_normal(path_mat.shape)
    return path_mat * (1.0 + SIMULATION_JITTER_FLOAT * jitter_mat) + daily_drift_shift_float


def cbi_mat(return_path_mat, table: CbiTable, min_observation_int: int = 21) -> np.ndarray:
    """CBI after each session for every path row; NaN before `min_observation_int` sessions.

    Same lookup rule as `CbiTable.probability`, vectorised across paths: at session M
    every path uses the same checkpoint, so only the window differs.
    """
    depth_mat, age_mat = drawdown_state_mat(return_path_mat)
    path_count_int, session_count_int = depth_mat.shape
    if session_count_int > int(table.checkpoint_vec[-1]):
        raise ValueError(f"Paths of {session_count_int} sessions exceed the table length {int(table.checkpoint_vec[-1])}.")
    cbi_arr = np.full((path_count_int, session_count_int), np.nan)
    last_window_idx_int = table.window_grid_vec.size - 1
    for step_idx_int in range(min_observation_int - 1, session_count_int):
        observation_int = step_idx_int + 1
        checkpoint_idx_int = max(int(np.searchsorted(table.checkpoint_vec, observation_int, side="right")) - 1, 0)
        depth_vec = depth_mat[:, step_idx_int]
        window_idx_vec = np.minimum(
            np.searchsorted(table.window_grid_vec, np.maximum(age_mat[:, step_idx_int], 1), side="left"), last_window_idx_int
        )
        step_cbi_vec = np.ones(path_count_int)
        in_drawdown_mask = depth_vec > 0.0
        for window_idx_int in np.unique(window_idx_vec[in_drawdown_mask]):
            member_mask = in_drawdown_mask & (window_idx_vec == window_idx_int)
            worst_vec = table.sorted_worst_mat[window_idx_int, checkpoint_idx_int]
            step_cbi_vec[member_mask] = (
                np.searchsorted(worst_vec, -depth_vec[member_mask] + TIE_TOLERANCE_FLOAT, side="right") / table.path_count_int
            )
        cbi_arr[:, step_idx_int] = step_cbi_vec
    return cbi_arr


def cbi_path(return_vec, table: CbiTable, min_observation_int: int = 21) -> np.ndarray:
    """CBI after each session for one path; NaN before `min_observation_int` sessions."""
    return cbi_mat(np.asarray(return_vec, dtype=float)[None, :], table, min_observation_int)[0]


@dataclass(frozen=True)
class CbiThresholds:
    red_float: float
    amber_float: float
    horizon_int: int
    min_observation_int: int
    red_rate_float: float
    amber_rate_float: float
    healthy_path_count_int: int


def _min_cbi_vec(return_path_mat: np.ndarray, table: CbiTable, min_observation_int: int) -> np.ndarray:
    return np.nanmin(cbi_mat(return_path_mat, table, min_observation_int), axis=1)


def calibrate_cbi_thresholds(
    expected_return_vec,
    table: CbiTable,
    horizon_int: int = 252,
    min_observation_int: int = 21,
    red_rate_float: float = 0.05,
    amber_rate_float: float = 0.15,
    healthy_path_count_int: int = 1000,
    random_seed_int: int = 1,
) -> CbiThresholds:
    """RED / AMBER cuts such that a healthy pod crosses them within `horizon_int` sessions at the given rates."""
    if horizon_int > int(table.checkpoint_vec[-1]):
        raise ValueError("horizon_int exceeds the table's max observation length.")
    healthy_path_mat = simulated_live_path_mat(
        expected_return_vec, healthy_path_count_int, table.mean_block_length_float, horizon_int, random_seed_int
    )
    min_cbi_vec = _min_cbi_vec(healthy_path_mat, table, min_observation_int)
    # "lower" picks an observed value, and alarms use a strict "<", so P(min < cut) <= rate.
    red_float = float(np.quantile(min_cbi_vec, red_rate_float, method="lower"))
    amber_float = float(np.quantile(min_cbi_vec, amber_rate_float, method="lower"))
    # The RED cut must rest on at least MIN_TABLE_HITS_INT reference paths, or it is table noise.
    if red_float * table.path_count_int < MIN_TABLE_HITS_INT:
        raise ValueError(
            f"The RED cut ({red_float:.5f}) rests on fewer than {MIN_TABLE_HITS_INT} of {table.path_count_int} "
            "reference paths: the table is too coarse. Increase path_count_int of the CBI table."
        )
    return CbiThresholds(
        red_float=red_float,
        amber_float=amber_float,
        horizon_int=horizon_int,
        min_observation_int=min_observation_int,
        red_rate_float=red_rate_float,
        amber_rate_float=amber_rate_float,
        healthy_path_count_int=healthy_path_count_int,
    )


@dataclass(frozen=True)
class DetectionResult:
    detected_share_float: float
    median_sessions_to_red_float: float
    scenario_str: str


def detection_delay(
    expected_return_vec,
    table: CbiTable,
    thresholds: CbiThresholds,
    daily_drift_shift_float: float,
    scenario_str: str,
    path_count_int: int = 500,
    random_seed_int: int = 2,
) -> DetectionResult:
    """How often, and how fast, a pod whose daily mean moved by `daily_drift_shift_float` goes RED within the horizon.

    Example scenarios: shift = −mean (the edge died: same volatility, zero drift),
    shift = −2·mean (the pod now loses what it used to make).
    """
    shifted_path_mat = simulated_live_path_mat(
        expected_return_vec, path_count_int, table.mean_block_length_float, thresholds.horizon_int, random_seed_int,
        daily_drift_shift_float,
    )
    red_mask_mat = cbi_mat(shifted_path_mat, table, thresholds.min_observation_int) < thresholds.red_float
    first_red_vec = np.where(red_mask_mat.any(axis=1), red_mask_mat.argmax(axis=1) + 1, np.nan).astype(float)
    detected_mask = np.isfinite(first_red_vec)
    return DetectionResult(
        detected_share_float=float(detected_mask.mean()),
        median_sessions_to_red_float=float(np.median(first_red_vec[detected_mask])) if detected_mask.any() else float("nan"),
        scenario_str=scenario_str,
    )


def compound_period_returns(return_vec, period_length_int: int = 21) -> np.ndarray:
    """Non-overlapping compounded returns of consecutive `period_length_int`-session blocks (a partial tail is dropped)."""
    return_arr = np.asarray(return_vec, dtype=float)
    block_count_int = return_arr.size // period_length_int
    block_mat = return_arr[: block_count_int * period_length_int].reshape(block_count_int, period_length_int)
    return np.prod(1.0 + block_mat, axis=1) - 1.0
