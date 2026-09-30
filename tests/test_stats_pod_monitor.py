"""Daily pod monitor with procedure-calibrated thresholds (alpha/stats/pod_monitor.py)."""

from __future__ import annotations

import numpy as np
import pytest

from alpha.stats.bootstrap import stationary_bootstrap_path_mat
from alpha.stats.health import cold_blood_index, current_drawdown_state
from alpha.stats.pod_monitor import (
    build_cbi_table,
    cbi_mat,
    simulated_live_path_mat,
    calibrate_cbi_thresholds,
    cbi_path,
    compound_period_returns,
    detection_delay,
    drawdown_state_path,
)

EXPECTED_VEC = np.random.default_rng(0).normal(0.0006, 0.012, 3000)


@pytest.fixture(scope="module")
def table():
    return build_cbi_table(EXPECTED_VEC, 252, path_count_int=10_000, random_seed_int=0)


def test_drawdown_state_path_matches_prefix_evaluation():
    return_vec = np.random.default_rng(3).normal(0.0, 0.02, 120)
    return_vec[40:43] = 0.0  # flat stretch, including exact ties with the peak
    depth_arr, age_arr = drawdown_state_path(return_vec)
    for step_int in range(1, return_vec.size + 1):
        state = current_drawdown_state(return_vec[:step_int])
        assert depth_arr[step_int - 1] == pytest.approx(state.depth_float, abs=1e-12)
        assert age_arr[step_int - 1] == state.length_int


def test_table_matches_direct_cold_blood_index_on_grid_points(table):
    for depth_float, age_int, observation_int in ((0.05, 21, 60), (0.10, 55, 120), (0.15, 144, 250)):
        table_float = table.probability(depth_float, age_int, observation_int)
        direct_float = cold_blood_index(
            EXPECTED_VEC, depth_float, age_int, observation_int, path_count_int=10_000, random_seed_int=5
        )
        standard_error_float = np.sqrt(max(direct_float * (1 - direct_float), 1e-4) * 2 / 10_000)
        assert abs(table_float - direct_float) < 4 * standard_error_float


def test_table_rounding_directions(table):
    # Depth: deeper is rarer.
    assert table.probability(0.05, 21, 100) >= table.probability(0.10, 21, 100) >= table.probability(0.2, 21, 100)
    # Age rounds UP to the next grid window: age 22 uses window 34, which can only find more drawdowns.
    assert table.probability(0.08, 22, 100) == table.probability(0.08, 34, 100)
    assert table.probability(0.08, 22, 100) >= table.probability(0.08, 21, 100)
    # Observation rounds DOWN to the previous checkpoint (step 5).
    assert table.probability(0.08, 21, 104) == table.probability(0.08, 21, 100)
    assert table.probability(0.0, 5, 50) == 1.0


def test_calibrated_cuts_hold_their_false_alarm_rates(table):
    thresholds = calibrate_cbi_thresholds(EXPECTED_VEC, table, healthy_path_count_int=3000, random_seed_int=1)
    assert 0.0 < thresholds.red_float < thresholds.amber_float
    # In sample, the "lower" quantile with a strict "<" never exceeds the target rate.
    calibration_path_mat = simulated_live_path_mat(EXPECTED_VEC, 3000, 20.0, 252, 1)
    in_sample_min_vec = np.nanmin(cbi_mat(calibration_path_mat, table), axis=1)
    assert np.mean(in_sample_min_vec < thresholds.red_float) <= 0.05
    assert np.mean(in_sample_min_vec < thresholds.amber_float) <= 0.15
    independent_path_mat = simulated_live_path_mat(EXPECTED_VEC, 3000, 20.0, 252, 77)
    min_cbi_vec = np.nanmin(cbi_mat(independent_path_mat, table), axis=1)
    assert np.mean(min_cbi_vec < thresholds.red_float) == pytest.approx(0.05, abs=0.015)
    assert np.mean(min_cbi_vec < thresholds.amber_float) == pytest.approx(0.15, abs=0.025)


def test_lumpy_process_false_alarms_hold_on_tie_free_live_paths():
    """A process with a few extreme crash blocks invites exact ties; calibration must not rely on them."""
    rng_obj = np.random.default_rng(4)
    lumpy_vec = rng_obj.normal(0.0008, 0.01, 3000)
    for start_int in (500, 1500, 2500):
        lumpy_vec[start_int : start_int + 5] = [-0.09, -0.06, 0.04, -0.07, 0.05]
    lumpy_table = build_cbi_table(lumpy_vec, 252, path_count_int=10_000, random_seed_int=3)
    thresholds = calibrate_cbi_thresholds(lumpy_vec, lumpy_table, healthy_path_count_int=3000, random_seed_int=5)
    # Live data never copy a historical block exactly: judge the cut on paths with a larger jitter.
    live_like_path_mat = stationary_bootstrap_path_mat(lumpy_vec, 3000, 20.0, 252, 91)
    live_like_path_mat = live_like_path_mat * (1.0 + 1e-3 * np.random.default_rng(92).standard_normal(live_like_path_mat.shape))
    min_cbi_vec = np.nanmin(cbi_mat(live_like_path_mat, lumpy_table), axis=1)
    assert np.mean(min_cbi_vec < thresholds.red_float) == pytest.approx(0.05, abs=0.02)


def test_tie_tolerance_is_small(table):
    deepest_float = float(table.sorted_worst_mat[0, -1, 0])  # worst 1-session drawdown of all reference paths
    assert table.probability(-deepest_float, 1, 252) > 0.0
    assert table.probability(-deepest_float + 1e-5, 1, 252) == 0.0


def test_lookups_beyond_the_table_raise(table):
    with pytest.raises(ValueError, match="exceeds"):
        table.probability(0.05, 21, 253)
    with pytest.raises(ValueError, match="exceed"):
        cbi_path(np.zeros(300), table)


def test_a_steadily_losing_pod_goes_red_quickly(table):
    thresholds = calibrate_cbi_thresholds(EXPECTED_VEC, table, healthy_path_count_int=2000)
    cbi_arr = cbi_path(np.full(80, -0.005), table)
    assert np.isnan(cbi_arr[:20]).all()
    first_red_int = int(np.flatnonzero(cbi_arr < thresholds.red_float)[0]) + 1
    assert first_red_int <= 60
    broken_result = detection_delay(EXPECTED_VEC, table, thresholds, -0.006, "broken", path_count_int=200)
    assert broken_result.detected_share_float > 0.9


def test_coarse_table_is_refused():
    coarse_table = build_cbi_table(EXPECTED_VEC, 252, path_count_int=100)
    with pytest.raises(ValueError, match="too coarse"):
        calibrate_cbi_thresholds(EXPECTED_VEC, coarse_table, healthy_path_count_int=500)


def test_horizon_longer_than_table_is_refused(table):
    with pytest.raises(ValueError, match="horizon"):
        calibrate_cbi_thresholds(EXPECTED_VEC, table, horizon_int=500)


def test_compound_period_returns():
    np.testing.assert_allclose(compound_period_returns([0.1, 0.1, -0.5, 0.2, 0.3], 2), [0.21, -0.4])


def test_vectorised_cbi_matches_the_scalar_lookup(table):
    from alpha.stats.pod_monitor import cbi_mat

    path_mat = np.random.default_rng(9).normal(-0.001, 0.015, (25, 252))
    vector_mat = cbi_mat(path_mat, table, 21)
    for path_idx_int in range(path_mat.shape[0]):
        depth_arr, age_arr = drawdown_state_path(path_mat[path_idx_int])
        for step_idx_int in range(20, 252, 7):
            scalar_float = table.probability(depth_arr[step_idx_int], int(age_arr[step_idx_int]), step_idx_int + 1)
            assert vector_mat[path_idx_int, step_idx_int] == scalar_float
    assert np.isnan(vector_mat[:, :20]).all()


def test_a_healthy_path_that_repeats_the_worst_sample_day_is_not_off_the_table(table):
    """Ties with reference drawdowns must count as hits, whatever the storage precision."""
    worst_day_float = float(EXPECTED_VEC.min())
    path_vec = np.concatenate([np.full(25, 0.001), [worst_day_float], np.full(10, 0.0005)])
    cbi_arr = cbi_path(path_vec, table)
    single_day_depth_float = -worst_day_float
    # Every reference path that contains the worst day at a peak has at least this drawdown in a 1-day window.
    assert cbi_arr[25] >= table.probability(single_day_depth_float + 1e-6, 1, 26)
    assert cbi_arr[25] > 0.0
