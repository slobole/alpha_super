"""Plateau selection and PBO (alpha/stats/selection.py, alpha/stats/pbo.py)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alpha.stats.pbo import probability_of_backtest_overfitting
from alpha.stats.selection import column_sharpe_vec, make_plateau_selector, neighbourhood_median_vec, plateau_choice


def test_neighbourhood_median_by_hand():
    sharpe_vec = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])  # grid 2 x 3
    median_vec = neighbourhood_median_vec(sharpe_vec, (2, 3))
    # (0,0): self 1, right 2, down 4 -> 2 ; (1,1): self 5, up 2, left 4, right 6 -> 4.5
    assert median_vec[0] == 2.0 and median_vec[4] == 4.5


def test_plateau_beats_a_lone_peak():
    sharpe_grid = np.full((5, 5), 0.1)
    sharpe_grid[0, 0] = 2.0  # lone lucky peak
    sharpe_grid[2:5, 2:5] = 0.8  # broad plateau
    sharpe_grid[3, 3] = 0.9
    choice = plateau_choice(sharpe_grid.reshape(-1), (5, 5))
    assert np.unravel_index(choice.flat_index_int, (5, 5)) == (3, 3)
    assert choice.peak_sharpe_float == 2.0
    assert choice.plateau_ratio_float == pytest.approx(0.8 / 2.0)


def test_plateau_tie_breaks_and_errors():
    choice = plateau_choice(np.array([1.0, 1.0, 1.0]), (3,))
    assert choice.flat_index_int == 0
    with pytest.raises(ValueError):
        plateau_choice(np.array([1.0, 2.0]), (3,))
    with pytest.raises(ValueError):
        plateau_choice(np.array([np.nan, np.nan]), (2,))


def test_column_sharpe_and_selector():
    rng_obj = np.random.default_rng(0)
    return_mat = rng_obj.normal(0.0, 0.01, (1000, 6))
    return_mat[:, 4] += 0.002
    return_mat[:, 5] += 0.002
    expected_vec = return_mat.mean(axis=0) / return_mat.std(axis=0, ddof=1) * np.sqrt(252)
    np.testing.assert_allclose(column_sharpe_vec(return_mat), expected_vec)
    train_df = pd.DataFrame(return_mat, columns=[f"c{i}" for i in range(6)])
    assert make_plateau_selector((6,), min_observation_int=100)(train_df) in ("c4", "c5")


def test_pbo_on_noise_is_near_one_half():
    pbo_list = [
        probability_of_backtest_overfitting(np.random.default_rng(seed_int).normal(0, 0.01, (2000, 20))).pbo_float
        for seed_int in range(30)
    ]
    assert np.mean(pbo_list) == pytest.approx(0.5, abs=0.1)


def test_pbo_on_a_stable_edge_is_near_zero():
    return_mat = np.random.default_rng(1).normal(0, 0.01, (2500, 20))
    return_mat[:, 7] += 0.0015  # annual Sharpe about 2.4 in one configuration
    result = probability_of_backtest_overfitting(return_mat)
    assert result.pbo_float < 0.05
    assert result.logit_vec.size == 252


def test_pbo_input_errors():
    with pytest.raises(ValueError):
        probability_of_backtest_overfitting(np.zeros((100, 1)))
    with pytest.raises(ValueError):
        probability_of_backtest_overfitting(np.zeros((100, 3)), block_count_int=5)
    bad_mat = np.zeros((100, 3))
    bad_mat[5, 1] = np.nan
    with pytest.raises(ValueError):
        probability_of_backtest_overfitting(bad_mat)
