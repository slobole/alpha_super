"""Shared statistics (alpha/stats): each estimator against a published example or an independent implementation."""

from __future__ import annotations

import numpy as np
import pytest
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.multitest import multipletests

from alpha.stats.bootstrap import stationary_bootstrap_index_mat
from alpha.stats.fdr import benjamini_hochberg_q_vec, benjamini_yekutieli_q_vec
from alpha.stats.newey_west import newey_west_mean_t_stat
from alpha.stats.permutation import permutation_p_value, shuffle_within_groups_index
from alpha.stats.psr_dsr import (
    bailey_approx_expected_max_z,
    deflated_sharpe_ratio,
    exact_expected_max_z,
    expected_max_sharpe_ratio,
    minimum_track_record_length,
    probabilistic_sharpe_ratio,
    sharpe_moments,
)


# ---------------------------------------------------------------- Newey–West
@pytest.mark.parametrize("lag_int", [0, 1, 4, 9])
def test_newey_west_matches_statsmodels_hac(lag_int):
    rng_obj = np.random.default_rng(7)
    noise_vec = rng_obj.normal(0.001, 0.01, 800)
    overlapping_vec = np.convolve(noise_vec, np.ones(5) / 5, mode="valid")  # 5-day overlapping means

    result = newey_west_mean_t_stat(overlapping_vec, lag_int)
    ols_result = sm.OLS(overlapping_vec, np.ones(overlapping_vec.size)).fit(
        cov_type="HAC", cov_kwds={"maxlags": lag_int, "use_correction": False}
    )
    assert result.mean_float == pytest.approx(ols_result.params[0])
    assert result.standard_error_float == pytest.approx(ols_result.bse[0], rel=1e-10)
    assert result.t_stat_float == pytest.approx(ols_result.tvalues[0], rel=1e-10)


def test_newey_west_overlap_shrinks_t_stat():
    rng_obj = np.random.default_rng(3)
    overlapping_vec = np.convolve(rng_obj.normal(0.001, 0.01, 2000), np.ones(5), mode="valid")
    naive_t_float = newey_west_mean_t_stat(overlapping_vec, 0).t_stat_float
    hac_t_float = newey_west_mean_t_stat(overlapping_vec, 4).t_stat_float
    assert abs(hac_t_float) < abs(naive_t_float)


def test_newey_west_drops_nan_and_rejects_bad_lag():
    result = newey_west_mean_t_stat([0.01, np.nan, 0.02, -0.01, 0.03], 1)
    assert result.observation_count_int == 4
    with pytest.raises(ValueError):
        newey_west_mean_t_stat([0.01, 0.02], 2)


# ---------------------------------------------------------------- bootstrap
def test_stationary_bootstrap_block_law():
    sample_size_int = 50
    index_mat = stationary_bootstrap_index_mat(sample_size_int, 400, 10.0, 300, random_seed_int=1)
    assert index_mat.shape == (400, 300)
    assert index_mat.min() >= 0 and index_mat.max() < sample_size_int
    # A step either continues its block (index + 1, wrapping) or restarts; restarts happen with p = 1/L.
    continued_mask = index_mat[:, 1:] == (index_mat[:, :-1] + 1) % sample_size_int
    assert 1.0 - continued_mask.mean() == pytest.approx(0.1 * (1 - 1 / sample_size_int), abs=0.003)
    assert set(np.unique(index_mat)) == set(range(sample_size_int))


def test_stationary_bootstrap_is_deterministic_per_seed():
    first_mat = stationary_bootstrap_index_mat(30, 5, 4.0, 20, random_seed_int=11)
    second_mat = stationary_bootstrap_index_mat(30, 5, 4.0, 20, random_seed_int=11)
    np.testing.assert_array_equal(first_mat, second_mat)


# ---------------------------------------------------------------- permutation
def test_permutation_p_value_has_plus_one_correction():
    assert permutation_p_value(10.0, np.zeros(99)) == pytest.approx(0.01)
    assert permutation_p_value(-1.0, np.zeros(99)) == pytest.approx(1.0)
    assert permutation_p_value(-1.0, np.zeros(99), "less") == pytest.approx(0.01)
    # Ties count as at least as extreme.
    assert permutation_p_value(0.0, np.zeros(99)) == pytest.approx(1.0)
    assert permutation_p_value(1.0, [1, 1, 0, 0]) == pytest.approx(3 / 5)


def test_shuffle_within_groups_keeps_group_membership():
    group_vec = np.array([0, 1, 0, 2, 1, 0, 2, 2, 1, 0])
    rng_obj = np.random.default_rng(0)
    for _ in range(20):
        perm_idx = shuffle_within_groups_index(group_vec, rng_obj)
        np.testing.assert_array_equal(group_vec[perm_idx], group_vec)
        assert sorted(perm_idx) == list(range(group_vec.size))


# ---------------------------------------------------------------- PSR / DSR / MinTRL
def test_dsr_matches_bailey_lopez_de_prado_2014_example():
    """Numerical example of "The Deflated Sharpe Ratio" (2014): N = 100 trials, V[SR] = 0.5 (annualised),
    selected SR = 2.5 (annualised), T = 1250 daily observations, skew = -3, kurtosis = 10.
    The paper reports SR*_0 = 0.1132 (per period) and DSR = 0.9004."""
    periods_per_year_int = 250
    benchmark_sharpe_float = np.sqrt(0.5 / periods_per_year_int) * bailey_approx_expected_max_z(100)
    assert benchmark_sharpe_float == pytest.approx(0.1132, abs=5e-5)
    # Production uses the exact E[max]; at N = 100 it is within 1% of the paper's approximation.
    assert expected_max_sharpe_ratio(0.5 / periods_per_year_int, 100) == pytest.approx(benchmark_sharpe_float, rel=0.01)

    deflated_sharpe_float = probabilistic_sharpe_ratio(
        sharpe_float=2.5 / np.sqrt(periods_per_year_int),
        observation_count_int=1250,
        skewness_float=-3.0,
        kurtosis_float=10.0,
        benchmark_sharpe_float=benchmark_sharpe_float,
    )
    assert deflated_sharpe_float == pytest.approx(0.9004, abs=5e-4)


def test_psr_normal_case_equals_classical_z_test():
    sharpe_float, observation_count_int = 0.08, 500
    expected_float = stats.norm.cdf(sharpe_float * np.sqrt(observation_count_int - 1) / np.sqrt(1 + 0.5 * sharpe_float**2))
    assert probabilistic_sharpe_ratio(sharpe_float, observation_count_int, 0.0, 3.0) == pytest.approx(expected_float)


def test_single_trial_has_no_selection_penalty():
    assert expected_max_sharpe_ratio(0.01, 1) == pytest.approx(0.0, abs=1e-12)


def test_exact_expected_max_z_known_values_and_small_fractional_n():
    assert exact_expected_max_z(2) == pytest.approx(1.0 / np.sqrt(np.pi), abs=1e-9)  # E[max of 2] = 1/sqrt(pi)
    simulated_float = np.random.default_rng(0).normal(size=(200_000, 5)).max(axis=1).mean()
    assert exact_expected_max_z(5) == pytest.approx(simulated_float, abs=0.01)
    # The paper's approximation is negative for N < 1.28; the exact value never is, and grows with N.
    assert bailey_approx_expected_max_z(1.1) < 0.0
    grid_vec = [exact_expected_max_z(n) for n in (1.0, 1.01, 1.1, 1.5, 2.0, 10.0, 100.0)]
    assert grid_vec[0] == pytest.approx(0.0, abs=1e-9)
    assert all(later > earlier for earlier, later in zip(grid_vec, grid_vec[1:]))
    assert expected_max_sharpe_ratio(0.01, 1.1) > 0.0


def test_sharpe_moments_rejects_infinity():
    with pytest.raises(ValueError, match="infinity"):
        sharpe_moments([0.01, np.inf, -0.02, 0.005])


def test_min_track_record_length_hits_the_confidence_level():
    sharpe_float, skewness_float, kurtosis_float = 0.1, -0.5, 5.0
    min_length_float = minimum_track_record_length(sharpe_float, skewness_float, kurtosis_float, 0.02, 0.95)
    psr_at_length_float = probabilistic_sharpe_ratio(
        sharpe_float, int(np.ceil(min_length_float)), skewness_float, kurtosis_float, 0.02
    )
    assert psr_at_length_float >= 0.95
    assert probabilistic_sharpe_ratio(
        sharpe_float, int(np.floor(min_length_float)) - 1, skewness_float, kurtosis_float, 0.02
    ) < 0.95
    assert minimum_track_record_length(0.01, 0.0, 3.0, 0.02) == float("inf")


def test_sharpe_moments_exact_small_vector():
    return_vec = np.array([0.01, -0.02, 0.015, 0.003, -0.004, 0.02])
    moments = sharpe_moments(return_vec)
    assert moments.sharpe_float == pytest.approx(return_vec.mean() / return_vec.std(ddof=1))
    assert moments.skewness_float == pytest.approx(stats.skew(return_vec))
    assert moments.kurtosis_float == pytest.approx(stats.kurtosis(return_vec, fisher=False))
    assert moments.observation_count_int == 6


def test_deflated_sharpe_ratio_wiring():
    return_vec = np.random.default_rng(3).normal(0.0006, 0.01, 1500)
    result = deflated_sharpe_ratio(return_vec, trial_sharpe_variance_float=4e-4, effective_trial_count_float=25)
    moments = sharpe_moments(return_vec)
    expected_benchmark_float = expected_max_sharpe_ratio(4e-4, 25)
    assert result.benchmark_sharpe_float == pytest.approx(expected_benchmark_float)
    assert result.deflated_sharpe_float == pytest.approx(
        probabilistic_sharpe_ratio(
            moments.sharpe_float, moments.observation_count_int, moments.skewness_float, moments.kurtosis_float,
            expected_benchmark_float,
        )
    )
    undeflated_float = deflated_sharpe_ratio(return_vec, 4e-4, 1).deflated_sharpe_float
    assert result.deflated_sharpe_float < undeflated_float


def test_psr_and_dsr_input_errors():
    with pytest.raises(ValueError):
        expected_max_sharpe_ratio(0.01, 0.5)
    with pytest.raises(ValueError):
        expected_max_sharpe_ratio(-0.01, 5)
    with pytest.raises(ValueError, match="variance factor"):
        probabilistic_sharpe_ratio(1.0, 100, 5.0, 3.0)
    with pytest.raises(ValueError):
        minimum_track_record_length(0.1, 0.0, 3.0, 0.0, 1.5)
    with pytest.raises(ValueError):
        newey_west_mean_t_stat(np.ones(10), 1)
    with pytest.raises(ValueError):
        newey_west_mean_t_stat(np.arange(10.0), -1)


def test_sharpe_moments_use_pearson_kurtosis():
    rng_obj = np.random.default_rng(5)
    moments = sharpe_moments(rng_obj.normal(0.0005, 0.01, 200_000))
    assert moments.kurtosis_float == pytest.approx(3.0, abs=0.05)
    assert moments.skewness_float == pytest.approx(0.0, abs=0.03)


# ---------------------------------------------------------------- FDR
def test_fdr_q_values_match_statsmodels():
    rng_obj = np.random.default_rng(9)
    p_value_vec = np.concatenate([rng_obj.uniform(0, 0.01, 5), rng_obj.uniform(0, 1, 40)])
    np.testing.assert_allclose(benjamini_hochberg_q_vec(p_value_vec), multipletests(p_value_vec, method="fdr_bh")[1])
    np.testing.assert_allclose(benjamini_yekutieli_q_vec(p_value_vec), multipletests(p_value_vec, method="fdr_by")[1])


def test_fdr_rejects_invalid_p_values():
    with pytest.raises(ValueError):
        benjamini_yekutieli_q_vec([0.1, 1.2])
