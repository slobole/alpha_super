"""False discovery rate q-values across many tests.

Benjamini–Hochberg (1995) for independent or positively dependent tests, and
Benjamini–Yekutieli (2001) for arbitrary dependence. Scout uses BY across the
ledger because families and trials share data and are not independent.

For sorted p-values p_(1) <= ... <= p_(m):

    q_(i) = min_{k >= i} min(1, c(m) · m · p_(k) / k)
    c(m)  = 1                     (BH)
    c(m)  = Σ_{j=1..m} 1/j        (BY)

These are the adjusted p-values statsmodels reports for "fdr_bh" and "fdr_by".
"""

from __future__ import annotations

import numpy as np


def _step_up_q_vec(p_value_vec, dependence_factor_float: float) -> np.ndarray:
    p_value_arr = np.asarray(p_value_vec, dtype=float)
    if p_value_arr.ndim != 1 or p_value_arr.size == 0:
        raise ValueError("p_value_vec must be a non-empty 1-D sequence.")
    if not np.all((p_value_arr >= 0.0) & (p_value_arr <= 1.0)):
        raise ValueError("p-values must lie in [0, 1].")

    test_count_int = p_value_arr.size
    order_idx_arr = np.argsort(p_value_arr)
    rank_arr = np.arange(1, test_count_int + 1)
    raw_q_arr = dependence_factor_float * test_count_int * p_value_arr[order_idx_arr] / rank_arr
    # Step-up: enforce monotonicity from the largest p-value downwards.
    sorted_q_arr = np.minimum(1.0, np.minimum.accumulate(raw_q_arr[::-1])[::-1])

    q_arr = np.empty(test_count_int)
    q_arr[order_idx_arr] = sorted_q_arr
    return q_arr


def benjamini_hochberg_q_vec(p_value_vec) -> np.ndarray:
    return _step_up_q_vec(p_value_vec, dependence_factor_float=1.0)


def benjamini_yekutieli_q_vec(p_value_vec) -> np.ndarray:
    test_count_int = len(p_value_vec)
    harmonic_float = float(np.sum(1.0 / np.arange(1, test_count_int + 1)))
    return _step_up_q_vec(p_value_vec, dependence_factor_float=harmonic_float)
