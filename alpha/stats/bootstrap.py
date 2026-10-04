"""Stationary block bootstrap (Politis & Romano, 1994), vectorised.

Same algorithm as `alpha.engine.risk_analysis.stationary_bootstrap_index_mat`,
which is left unchanged: each simulated step starts a new block at a uniformly
random observation with probability p = 1 / L (L = mean block length) and
otherwise continues to the next observation, wrapping at the sample end. The
first step of every path starts a new block.

This version is vectorised because the health monitor and the calibration study
draw thousands of paths. It does not reproduce RiskAnalysis's random stream for
the same seed; only the sampling law is the same.
"""

from __future__ import annotations

import numpy as np


def stationary_bootstrap_index_mat(
    sample_size_int: int,
    path_count_int: int,
    mean_block_length_float: float,
    path_length_int: int,
    random_seed_int: int,
) -> np.ndarray:
    """Return a (path_count, path_length) matrix of indices into the sample."""
    if sample_size_int <= 0 or path_count_int <= 0 or path_length_int <= 0:
        raise ValueError("sample_size_int, path_count_int and path_length_int must be positive.")
    if mean_block_length_float < 1.0:
        raise ValueError("mean_block_length_float must be >= 1.")

    rng_obj = np.random.default_rng(int(random_seed_int))
    restart_probability_float = 1.0 / float(mean_block_length_float)

    block_start_mat = rng_obj.integers(0, sample_size_int, size=(path_count_int, path_length_int))
    restart_mask_mat = rng_obj.random((path_count_int, path_length_int)) < restart_probability_float
    restart_mask_mat[:, 0] = True

    step_idx_mat = np.broadcast_to(np.arange(path_length_int), (path_count_int, path_length_int))
    # Position of the most recent block start at or before each step.
    last_restart_step_mat = np.maximum.accumulate(np.where(restart_mask_mat, step_idx_mat, 0), axis=1)
    last_restart_index_mat = np.take_along_axis(block_start_mat, last_restart_step_mat, axis=1)

    return (last_restart_index_mat + (step_idx_mat - last_restart_step_mat)) % sample_size_int


def stationary_bootstrap_path_mat(
    value_vec,
    path_count_int: int,
    mean_block_length_float: float,
    path_length_int: int,
    random_seed_int: int,
) -> np.ndarray:
    """Resample a 1-D series into (path_count, path_length) bootstrap paths."""
    value_arr = np.asarray(value_vec, dtype=float)
    if not np.all(np.isfinite(value_arr)):
        raise ValueError("value_vec must be finite; drop NaN before resampling.")
    index_mat = stationary_bootstrap_index_mat(
        sample_size_int=int(value_arr.size),
        path_count_int=path_count_int,
        mean_block_length_float=mean_block_length_float,
        path_length_int=path_length_int,
        random_seed_int=random_seed_int,
    )
    return value_arr[index_mat]
