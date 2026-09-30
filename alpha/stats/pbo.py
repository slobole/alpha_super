"""Probability of Backtest Overfitting by combinatorially symmetric cross-validation (CSCV).

Bailey, Borwein, López de Prado & Zhu, "The Probability of Backtest Overfitting" (2017).

1. Split the T x N matrix of configuration returns into S contiguous blocks of rows.
2. For each of the C(S, S/2) ways to pick half the blocks as the training set:
   - compute every configuration's Sharpe on the training half and on the other half;
   - pick a configuration with `select_fn` on the training Sharpes (default: the highest);
   - its relative out-of-sample rank  w = rank_OOS / (N + 1)   (rank 1 = worst, N = best);
   - logit  λ = ln( w / (1 − w) ).
3. PBO = share of splits with λ <= 0, i.e. the chosen configuration is at or below
   the out-of-sample median.

Pure noise gives PBO near 0.5; a real, stable edge in the chosen region gives PBO
near 0. Block Sharpes come from block sums, so the whole test costs one pass over
the data plus C(S, S/2) cheap combinations (S = 10: 252).
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Callable

import numpy as np
from scipy.stats import rankdata


@dataclass(frozen=True)
class PboResult:
    pbo_float: float
    logit_vec: np.ndarray
    block_count_int: int


def probability_of_backtest_overfitting(
    config_return_mat,
    block_count_int: int = 10,
    select_fn: Callable[[np.ndarray], int] | None = None,
) -> PboResult:
    return_arr = np.asarray(config_return_mat, dtype=float)
    if return_arr.ndim != 2 or return_arr.shape[1] < 2:
        raise ValueError("config_return_mat must be T x N with N >= 2 configurations.")
    if block_count_int < 2 or block_count_int % 2:
        raise ValueError("block_count_int must be an even number >= 2.")
    if not np.all(np.isfinite(return_arr)):
        raise ValueError("config_return_mat must be finite (fill flat days with 0.0).")
    select_fn = select_fn or (lambda sharpe_vec: int(np.nanargmax(sharpe_vec)))

    config_count_int = return_arr.shape[1]
    block_edge_vec = np.linspace(0, return_arr.shape[0], block_count_int + 1).astype(int)
    block_sum_mat = np.array([return_arr[a:b].sum(axis=0) for a, b in zip(block_edge_vec[:-1], block_edge_vec[1:])])
    block_square_mat = np.array([(return_arr[a:b] ** 2).sum(axis=0) for a, b in zip(block_edge_vec[:-1], block_edge_vec[1:])])
    block_count_vec = np.diff(block_edge_vec).astype(float)

    def half_sharpe_vec(block_idx_tuple) -> np.ndarray:
        idx_arr = np.asarray(block_idx_tuple)
        count_float = block_count_vec[idx_arr].sum()
        mean_vec = block_sum_mat[idx_arr].sum(axis=0) / count_float
        variance_vec = (block_square_mat[idx_arr].sum(axis=0) - count_float * mean_vec**2) / (count_float - 1.0)
        std_vec = np.sqrt(np.clip(variance_vec, 0.0, None))
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(std_vec > 0, mean_vec / std_vec, np.nan)

    all_block_set = set(range(block_count_int))
    logit_list = []
    for train_block_tuple in combinations(range(block_count_int), block_count_int // 2):
        test_block_tuple = tuple(sorted(all_block_set - set(train_block_tuple)))
        train_sharpe_vec = half_sharpe_vec(train_block_tuple)
        test_sharpe_vec = half_sharpe_vec(test_block_tuple)
        chosen_int = select_fn(train_sharpe_vec)
        test_rank_vec = rankdata(np.where(np.isfinite(test_sharpe_vec), test_sharpe_vec, -np.inf))
        relative_rank_float = test_rank_vec[chosen_int] / (config_count_int + 1.0)
        logit_list.append(np.log(relative_rank_float / (1.0 - relative_rank_float)))

    logit_vec = np.array(logit_list)
    return PboResult(pbo_float=float(np.mean(logit_vec <= 0.0)), logit_vec=logit_vec, block_count_int=block_count_int)
