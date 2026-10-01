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


def test_a_configuration_without_its_own_sharpe_is_never_chosen():
    sharpe_grid = np.full((3, 3), 0.2)
    sharpe_grid[1, 1] = np.nan  # neighbours are strong, own Sharpe undefined
    sharpe_grid[0, 1] = sharpe_grid[1, 0] = sharpe_grid[1, 2] = sharpe_grid[2, 1] = 1.0
    choice = plateau_choice(sharpe_grid.reshape(-1), (3, 3))
    assert choice.flat_index_int != 4 and np.isfinite(choice.own_sharpe_float)


def test_vectorised_plateau_matches_the_scalar_rule():
    from alpha.stats.selection import plateau_choice_index_mat

    draw_mat = np.random.default_rng(3).normal(0, 1, (2000, 21))
    draw_mat[:50] = np.round(draw_mat[:50], 1)  # force ties
    vector_idx = plateau_choice_index_mat(draw_mat, (7, 3))
    scalar_idx = np.array([plateau_choice(row, (7, 3)).flat_index_int for row in draw_mat])
    np.testing.assert_array_equal(vector_idx, scalar_idx)


def test_selector_ignores_configurations_with_too_little_history():
    rng_obj = np.random.default_rng(0)
    frame = pd.DataFrame(rng_obj.normal(0, 0.01, (400, 3)), columns=["a", "b", "c"])
    frame["c"] = np.nan
    frame.loc[frame.index[-50:], "c"] = 0.05  # spectacular but only 50 days
    assert make_plateau_selector((3,), min_observation_int=252)(frame) != "c"


def test_null_benchmark_known_cases():
    from alpha.stats.psr_dsr import exact_expected_max_z, null_selected_sharpe_benchmark

    observation_count_int = 1001
    scale_float = 1 / np.sqrt(observation_count_int - 1)
    independent_float = null_selected_sharpe_benchmark(np.eye(10), observation_count_int, draw_count_int=100_000)
    assert independent_float == pytest.approx(exact_expected_max_z(10) * scale_float, rel=0.01)
    identical_float = null_selected_sharpe_benchmark(np.ones((10, 10)), observation_count_int, draw_count_int=100_000)
    assert identical_float == pytest.approx(0.0, abs=0.01 * scale_float)
    with_prior_float = null_selected_sharpe_benchmark(np.ones((10, 10)), observation_count_int, prior_independent_trial_count_int=9)
    assert with_prior_float == pytest.approx(exact_expected_max_z(10) * scale_float, rel=0.02)
    # Highly correlated but not identical configurations still deflate (the clustered N_eff would say 1).
    correlated_mat = np.full((24, 24), 0.9) + 0.1 * np.eye(24)
    assert null_selected_sharpe_benchmark(correlated_mat, observation_count_int) > 0.3 * independent_float


def test_pbo_uses_the_selector_and_counts_the_median_as_overfit():
    # Two blocks -> two splits. Three configurations with fixed per-block means.
    block_mean_mat = np.array([[0.01, 0.00, -0.01], [-0.01, 0.00, 0.01]])  # config 0 wins block 0, config 2 wins block 1
    rng_obj = np.random.default_rng(0)
    return_mat = np.vstack([block_mean_mat[b] + rng_obj.normal(0, 1e-4, (200, 3)) for b in range(2)])
    # Max selection: the in-sample winner is the out-of-sample loser in both splits -> PBO = 1.
    assert probability_of_backtest_overfitting(return_mat, 2).pbo_float == 1.0
    # Always choosing config 1 (the middle in both halves) gives w = 2/4 = 0.5, logit 0, counted as overfit.
    middle_result = probability_of_backtest_overfitting(return_mat, 2, select_fn=lambda sharpe_vec: 1)
    assert middle_result.pbo_float == 1.0 and np.allclose(middle_result.logit_vec, 0.0)
    # Choosing the out-of-sample winner (config 2 on block 1, config 0 on block 0) is never overfit.
    oracle_result = probability_of_backtest_overfitting(return_mat, 2, select_fn=lambda sharpe_vec: int(np.argmin(sharpe_vec)))
    assert oracle_result.pbo_float == 0.0
    assert np.allclose(oracle_result.logit_vec, np.log(0.75 / 0.25))


def test_null_p_value_and_matrix_validation():
    from alpha.stats.psr_dsr import null_selected_sharpe_draws, null_selected_sharpe_p_value

    null_vec = null_selected_sharpe_draws(np.eye(5), 1001, draw_count_int=50_000)
    median_float = float(np.median(null_vec))
    assert null_selected_sharpe_p_value(median_float, null_vec) == pytest.approx(0.5, abs=0.01)
    assert null_selected_sharpe_p_value(1.0, null_vec) == pytest.approx(1 / 50_001)
    bad_mat = np.eye(3)
    bad_mat[0, 1] = bad_mat[1, 0] = np.nan
    with pytest.raises(ValueError, match="finite square"):
        null_selected_sharpe_draws(bad_mat, 1001)
    # A non-PSD pairwise matrix is repaired to a valid unit-diagonal correlation.
    from alpha.stats.psr_dsr import _unit_correlation_root

    root_mat = _unit_correlation_root(np.array([[1.0, 0.9, -0.9], [0.9, 1.0, 0.9], [-0.9, 0.9, 1.0]]))
    np.testing.assert_allclose(np.diag(root_mat @ root_mat.T), 1.0, atol=1e-9)


def test_plateau_refuses_lone_configurations_and_nan_rows():
    from alpha.stats.selection import plateau_choice_index_mat

    sharpe_grid = np.array([[np.nan, 3.0, np.nan], [np.nan, np.nan, np.nan], [0.5, 0.6, 0.4]])
    choice = plateau_choice(sharpe_grid.reshape(-1), (3, 3))  # the lone 3.0 has no finite neighbour
    assert choice.flat_index_int in (6, 7, 8)
    with pytest.raises(ValueError, match="no plateau"):
        plateau_choice(np.array([np.nan, 3.0, np.nan, 2.0]), (4,))
    with pytest.raises(ValueError, match="finite"):
        plateau_choice_index_mat(np.array([[0.1, np.nan, 0.2]]), (3,))


def test_mcpt_refuses_zero_filled_listings_when_the_mask_is_given():
    from alpha.stats.mcpt import mcpt

    return_mat = np.random.default_rng(0).normal(0, 0.01, (200, 2))
    return_mat[:100, 1] = 0.0  # zero-filled before listing: invisible without a mask
    mcpt(lambda mat: 0.0, return_mat, permutation_count_int=3, random_seed_int=0)
    mask_mat = np.ones_like(return_mat, dtype=bool)
    mask_mat[:100, 1] = False
    with pytest.raises(ValueError, match="mcpt_live_spans"):
        mcpt(lambda mat: 0.0, return_mat, permutation_count_int=3, random_seed_int=0, availability_mask_mat=mask_mat)
