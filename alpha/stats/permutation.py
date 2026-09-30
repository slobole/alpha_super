"""Permutation-test helpers.

Empirical p-value with the +1 correction (Phipson & Smyth, 2010), so a test with
n draws can never report p = 0:

    p = (1 + #{null_i >= observed}) / (1 + n)

`shuffle_within_groups_index` is the one shuffle used everywhere: within-date
label shuffles in edge studies (group = date) and volatility-stratified date
shuffles in the MCPT null (group = volatility stratum).
"""

from __future__ import annotations

import numpy as np


def permutation_p_value(observed_float: float, null_vec, alternative_str: str = "greater") -> float:
    """One-sided empirical p-value of `observed_float` against its null draws."""
    null_arr = np.asarray(null_vec, dtype=float)
    if null_arr.size == 0 or not np.all(np.isfinite(null_arr)):
        raise ValueError("null_vec must be a non-empty finite sample.")
    if not np.isfinite(observed_float):
        raise ValueError("observed_float must be finite.")
    if alternative_str == "greater":
        extreme_count_int = int(np.sum(null_arr >= observed_float))
    elif alternative_str == "less":
        extreme_count_int = int(np.sum(null_arr <= observed_float))
    else:
        raise ValueError("alternative_str must be 'greater' or 'less'.")
    return (1.0 + extreme_count_int) / (1.0 + null_arr.size)


def shuffle_within_groups_index(group_vec, rng_obj: np.random.Generator) -> np.ndarray:
    """Permutation index that shuffles positions only among members of the same group.

    Result `perm_idx` satisfies group_vec[perm_idx] == group_vec, so applying it
    to any array moves values only between positions of the same group.
    """
    group_arr = np.asarray(group_vec)
    perm_idx_arr = np.arange(group_arr.size)
    for group_value in np.unique(group_arr):
        member_idx_arr = np.flatnonzero(group_arr == group_value)
        perm_idx_arr[member_idx_arr] = rng_obj.permutation(member_idx_arr)
    return perm_idx_arr
