"""MCPT, walk-forward and pod-health statistics (alpha/stats)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from alpha.stats.health import (
    _min_windowed_drawdown_vec,
    calibrate_cusum_threshold,
    cold_blood_index,
    current_drawdown_state,
    cusum_alarm_bool,
    lower_cusum_path,
)
from alpha.stats.mcpt import mcpt, volatility_strata_vec
from alpha.stats.walk_forward import (
    DESIGN_GRID_TUPLE,
    WalkForwardDesign,
    annualized_sharpe_float,
    design_sensitivity_df,
    run_walk_forward,
    select_max_sharpe,
)


# ---------------------------------------------------------------- MCPT
def _momentum_search_score(return_mat: np.ndarray) -> float:
    """A small timing search: hold the asset tomorrow if its trailing k-day return is positive; best mean over k."""
    return_vec = return_mat[:, 0]
    best_float = -np.inf
    for lookback_int in (1, 2, 5, 10):
        trailing_vec = pd.Series(return_vec).rolling(lookback_int).sum().to_numpy()
        # *** CRITICAL*** position for day t uses returns up to t-1 only.
        position_vec = np.concatenate(([0.0], (trailing_vec[:-1] > 0).astype(float)))
        best_float = max(best_float, float(np.mean(position_vec * return_vec)))
    return best_float


def test_mcpt_detects_planted_serial_structure():
    rng_obj = np.random.default_rng(1)
    shock_vec = rng_obj.normal(0, 0.01, 1500)
    trending_vec = np.empty_like(shock_vec)
    trending_vec[0] = shock_vec[0]
    for t_int in range(1, shock_vec.size):
        trending_vec[t_int] = 0.3 * trending_vec[t_int - 1] + shock_vec[t_int]
    result = mcpt(_momentum_search_score, trending_vec[:, None], permutation_count_int=200, random_seed_int=2)
    assert result.p_value_float <= 0.01


def test_mcpt_p_values_are_roughly_uniform_on_noise():
    p_value_list = []
    for seed_int in range(40):
        noise_vec = np.random.default_rng(100 + seed_int).normal(0, 0.01, 400)
        result = mcpt(_momentum_search_score, noise_vec[:, None], permutation_count_int=60, random_seed_int=seed_int)
        p_value_list.append(result.p_value_float)
    assert stats.kstest(p_value_list, "uniform").pvalue > 0.01
    assert np.mean(np.array(p_value_list) <= 0.05) <= 0.2


def test_mcpt_keeps_cross_sectional_rows_and_strata():
    rng_obj = np.random.default_rng(4)
    return_mat = rng_obj.normal(0, 0.01, (300, 3))
    seen_row_set = {tuple(row) for row in return_mat}
    strata_vec = volatility_strata_vec(return_mat, window_int=20)

    captured_list = []

    def capture_fn(permuted_mat):
        captured_list.append(permuted_mat.copy())
        return 0.0

    mcpt(capture_fn, return_mat, permutation_count_int=3, random_seed_int=0, strata_vec=strata_vec)
    for permuted_mat in captured_list[1:]:
        assert {tuple(row) for row in permuted_mat} == seen_row_set
        # Each position still holds a day from the same volatility stratum.
        row_lookup = {tuple(row): idx for idx, row in enumerate(return_mat)}
        source_idx_vec = np.array([row_lookup[tuple(row)] for row in permuted_mat])
        np.testing.assert_array_equal(strata_vec[source_idx_vec], strata_vec)


def test_volatility_strata_match_a_trailing_rolling_std():
    return_vec = np.concatenate(
        [np.random.default_rng(0).normal(0, 0.005, 300), np.random.default_rng(1).normal(0, 0.03, 300)]
    )
    strata_vec = volatility_strata_vec(return_vec, window_int=20, stratum_count_int=2)
    trailing_vol_vec = pd.Series(return_vec).rolling(20).std(ddof=0).bfill().to_numpy()
    expected_vec = np.searchsorted(np.quantile(trailing_vol_vec, [0.5]), trailing_vol_vec, side="right")
    np.testing.assert_array_equal(strata_vec, expected_vec)


def test_mcpt_refuses_a_moving_nan_pattern():
    return_mat = np.random.default_rng(0).normal(0, 0.01, (200, 3))
    return_mat[:100, 2] = np.nan  # asset 2 lists halfway through
    with pytest.raises(ValueError, match="NaN pattern"):
        mcpt(lambda mat: 0.0, return_mat, permutation_count_int=5, random_seed_int=0)
    return_mat[:, 2] = np.nan  # a column that is missing everywhere is a fixed pattern
    mcpt(lambda mat: 0.0, return_mat, permutation_count_int=5, random_seed_int=0)


def test_mcpt_input_errors():
    with pytest.raises(ValueError):
        mcpt(lambda mat: 0.0, np.zeros((50, 1)), permutation_count_int=0, random_seed_int=0)
    with pytest.raises(ValueError):
        mcpt(lambda mat: 0.0, np.zeros((50, 1)), permutation_count_int=5, random_seed_int=0, strata_vec=np.zeros(10))
    with pytest.raises(ValueError):
        mcpt(lambda mat: float("nan"), np.zeros((50, 1)), permutation_count_int=5, random_seed_int=0)
    with pytest.raises(ValueError):
        volatility_strata_vec(np.zeros(30), window_int=63)


def test_volatility_strata_are_trailing_and_balanced():
    calm_vec = np.random.default_rng(0).normal(0, 0.005, 300)
    wild_vec = np.random.default_rng(1).normal(0, 0.03, 300)
    strata_vec = volatility_strata_vec(np.concatenate([calm_vec, wild_vec]), window_int=20, stratum_count_int=2)
    assert np.mean(strata_vec[100:280] == 0) > 0.95
    assert np.mean(strata_vec[340:] == 1) > 0.95


# ---------------------------------------------------------------- walk-forward
def _config_returns(date_count_int: int = 252 * 14, seed_int: int = 0) -> pd.DataFrame:
    date_index = pd.bdate_range("2000-01-03", periods=date_count_int)
    rng_obj = np.random.default_rng(seed_int)
    return pd.DataFrame(rng_obj.normal(0.0002, 0.01, (date_count_int, 4)), index=date_index, columns=list("abcd"))


def test_walk_forward_never_sees_the_test_window():
    config_return_df = _config_returns()
    # Config "d" is the worst before 2010 and spectacular after; a look-ahead selector would pick it early.
    config_return_df.loc[:"2009-12-31", "d"] -= 0.002
    config_return_df.loc["2010-01-01":, "d"] += 0.01
    result = run_walk_forward(config_return_df, select_max_sharpe, WalkForwardDesign(True, 5, 12))
    refit_df = result.refit_df
    assert "d" not in set(refit_df.loc[refit_df["test_start"] < "2010-01-01", "chosen_config"])
    # Once the boom is in the training data, the selector finds it.
    assert "d" in set(refit_df.loc[refit_df["test_start"] >= "2012-01-01", "chosen_config"])


def test_walk_forward_training_slices_respect_the_boundary():
    config_return_df = _config_returns()
    for design in DESIGN_GRID_TUPLE[:2] + DESIGN_GRID_TUPLE[6:8]:
        seen_list = []

        def spy_selector(train_return_df):
            seen_list.append((train_return_df.index.min(), train_return_df.index.max()))
            return select_max_sharpe(train_return_df)

        result = run_walk_forward(config_return_df, spy_selector, design)
        for (train_min, train_max), test_start in zip(seen_list, result.refit_df["test_start"]):
            assert train_max < test_start
            window_start = test_start - pd.DateOffset(years=design.train_years_int)
            if design.anchored_bool:
                assert train_min == config_return_df.index[0]
            else:
                assert window_start <= train_min < window_start + pd.Timedelta(days=7)


def test_walk_forward_picks_up_a_spike_only_after_it_happens():
    base_df = _config_returns()
    base_refit_df = run_walk_forward(base_df, select_max_sharpe).refit_df
    boom_start, boom_end = base_refit_df["test_start"].iloc[3], base_refit_df["test_start"].iloc[4]
    boomed_df = base_df.copy()
    # Config "c" booms for exactly one test window that starts on a refit day.
    boomed_df.loc[(boomed_df.index >= boom_start) & (boomed_df.index < boom_end), "c"] += 0.02
    boomed_refit_df = run_walk_forward(boomed_df, select_max_sharpe).refit_df
    up_to_boom_mask = boomed_refit_df["test_start"] <= boom_start
    # Choices made up to and including the boom's first day cannot change; the next refit must see it.
    pd.testing.assert_series_equal(
        boomed_refit_df.loc[up_to_boom_mask, "chosen_config"], base_refit_df.loc[up_to_boom_mask, "chosen_config"]
    )
    assert boomed_refit_df.loc[boomed_refit_df["test_start"] == boom_end, "chosen_config"].item() == "c"


def test_walk_forward_schedule_and_efficiency():
    config_return_df = _config_returns(252 * 8)
    first_date = config_return_df.index[0]
    for design in (WalkForwardDesign(True, 5, 12), WalkForwardDesign(False, 5, 6)):
        result = run_walk_forward(config_return_df, select_max_sharpe, design)
        # Data starts 2000-01-03, so five years of history exist from 2005-01-03: the first calendar
        # anchor on or after that is 2005-07-01 (6-month) or 2006-01-01 (12-month).
        assert first_date == pd.Timestamp("2000-01-03")
        first_anchor_str = "2005-07-01" if design.refit_months_int == 6 else "2006-01-01"
        expected_anchor_list = list(
            pd.date_range(first_anchor_str, config_return_df.index[-1], freq=f"{design.refit_months_int}MS")
        )
        expected_start_list = [
            config_return_df.index[config_return_df.index.searchsorted(anchor)] for anchor in expected_anchor_list
        ]
        assert list(result.refit_df["test_start"]) == expected_start_list
        assert all(start.month in (1, 7) for start in result.refit_df["test_start"])
        for _, refit_row in result.refit_df.iterrows():
            train_mask = config_return_df.index < refit_row["test_start"]
            if not design.anchored_bool:
                train_mask &= config_return_df.index >= refit_row["test_start"] - pd.DateOffset(years=5)
            train_ser = config_return_df.loc[train_mask, refit_row["chosen_config"]]
            assert refit_row["is_sharpe_float"] == pytest.approx(annualized_sharpe_float(train_ser))
        assert result.oos_sharpe_float == pytest.approx(annualized_sharpe_float(result.oos_return_ser))
        if result.mean_is_sharpe_float > 0:
            assert result.efficiency_float == pytest.approx(result.oos_sharpe_float / result.mean_is_sharpe_float)
    assert np.isnan(run_walk_forward(config_return_df - 0.01, select_max_sharpe).efficiency_float)


def test_walk_forward_warm_up_and_nan_handling():
    config_return_df = _config_returns()
    # A long-lookback config with only two (spectacular) finite days in the window must not be selectable.
    config_return_df["warmup"] = np.nan
    config_return_df.loc[config_return_df.index[1200:1202], "warmup"] = [0.05, 0.06]
    assert select_max_sharpe(config_return_df.iloc[:1250]) != "warmup"

    gap_df = _config_returns()
    gap_df.loc[:, ["b", "c", "d"]] -= 0.01  # "a" is always the choice
    gap_df.loc[gap_df.index[-10], "a"] = np.nan
    with pytest.raises(ValueError, match="NaN returns in the test window"):
        run_walk_forward(gap_df, select_max_sharpe)


def test_walk_forward_input_errors():
    with pytest.raises(ValueError, match="shorter"):
        run_walk_forward(_config_returns(252 * 3), select_max_sharpe)
    with pytest.raises(ValueError, match="sorted"):
        run_walk_forward(_config_returns().iloc[::-1], select_max_sharpe)


def test_walk_forward_stitches_the_chosen_columns():
    config_return_df = _config_returns()
    result = run_walk_forward(config_return_df, select_max_sharpe, WalkForwardDesign(False, 5, 6))
    for row_idx_int, refit_row in result.refit_df.iterrows():
        start = refit_row["test_start"]
        end = result.refit_df["test_start"].iloc[row_idx_int + 1] if row_idx_int + 1 < len(result.refit_df) else None
        piece_ser = result.oos_return_ser.loc[start:] if end is None else result.oos_return_ser.loc[start : end - pd.Timedelta(days=1)]
        expected_ser = config_return_df.loc[piece_ser.index, refit_row["chosen_config"]]
        np.testing.assert_allclose(piece_ser.to_numpy(), expected_ser.to_numpy())
    assert result.oos_return_ser.index.is_unique
    assert result.oos_return_ser.index[0] >= config_return_df.index[0] + pd.DateOffset(years=5)


def test_design_grid_has_eight_distinct_designs_and_skips_too_short_history():
    assert len(DESIGN_GRID_TUPLE) == 8
    assert len({design.label_str for design in DESIGN_GRID_TUPLE}) == 8
    sensitivity_df = design_sensitivity_df(_config_returns(252 * 8 + 100), select_max_sharpe)
    assert len(sensitivity_df) == 8
    too_long_mask = sensitivity_df["design_str"].str.contains("rolling_8y")
    assert sensitivity_df.loc[too_long_mask, "oos_sharpe_float"].isna().all()
    assert sensitivity_df.loc[~too_long_mask, "oos_sharpe_float"].notna().all()


# ---------------------------------------------------------------- health
def test_current_drawdown_state():
    state = current_drawdown_state([0.10, -0.05, 0.02, -0.10])
    equity_vec = np.cumprod([1.10, 0.95, 1.02, 0.90])
    assert state.depth_float == pytest.approx(1 - equity_vec[-1] / 1.10)
    assert state.length_int == 3
    assert current_drawdown_state([0.01, 0.02]).length_int == 0


def test_windowed_drawdown_matches_brute_force():
    rng_obj = np.random.default_rng(8)
    return_path_mat = rng_obj.normal(0, 0.02, (30, 60))
    for window_int in (1, 4, 7, 20, 60, 100):
        fast_vec = _min_windowed_drawdown_vec(return_path_mat, window_int)
        for path_idx_int in range(return_path_mat.shape[0]):
            equity_vec = np.concatenate(([1.0], np.cumprod(1 + return_path_mat[path_idx_int])))
            brute_float = min(
                equity_vec[t_int] / equity_vec[max(0, t_int - window_int) : t_int + 1].max() - 1
                for t_int in range(equity_vec.size)
            )
            assert fast_vec[path_idx_int] == pytest.approx(brute_float)


def test_cold_blood_index_matches_iid_resampling_of_the_same_sample():
    """With block length 1 the bootstrap is i.i.d. resampling, so the CBI must match a direct i.i.d. simulation."""
    sample_vec = np.random.default_rng(12).normal(0.0004, 0.01, 5_000)
    path_count_int = 20_000
    cbi_float = cold_blood_index(
        sample_vec, drawdown_depth_float=0.08, drawdown_length_int=40, observation_length_int=120,
        path_count_int=path_count_int, mean_block_length_float=1.0, random_seed_int=3,
    )
    iid_path_mat = np.random.default_rng(99).choice(sample_vec, size=(path_count_int, 120))
    direct_float = float(np.mean(_min_windowed_drawdown_vec(iid_path_mat, 40) <= -0.08))
    standard_error_float = np.sqrt(2 * direct_float * (1 - direct_float) / path_count_int)
    assert abs(cbi_float - direct_float) < 4 * standard_error_float


def test_cold_blood_index_is_monotone():
    sample_vec = np.random.default_rng(2).normal(0.0005, 0.01, 3000)
    shallow_float = cold_blood_index(sample_vec, 0.05, 30, 250, path_count_int=2000)
    deep_float = cold_blood_index(sample_vec, 0.12, 30, 250, path_count_int=2000)
    longer_observation_float = cold_blood_index(sample_vec, 0.12, 30, 500, path_count_int=2000)
    assert shallow_float > deep_float
    assert longer_observation_float >= deep_float
    assert cold_blood_index(sample_vec, 0.0, 30, 250) == 1.0


def test_lower_cusum_path():
    np.testing.assert_allclose(lower_cusum_path([-1.0, -1.0, 2.0, -0.2], 0.5), [-0.5, -1.0, 0.0, 0.0])
    with pytest.raises(ValueError):
        lower_cusum_path([-2.0, -2.0, np.nan, -0.1])


def test_cusum_false_alarm_rate_and_detection():
    rng_obj = np.random.default_rng(21)
    expected_monthly_vec = rng_obj.normal(0.01, 0.04, 300)
    calibration = calibrate_cusum_threshold(expected_monthly_vec, mean_block_length_float=1.0, random_seed_int=1)

    # Healthy pod = the same (i.i.d.) resampling law as the calibration, fresh draws.
    healthy_path_mat = np.random.default_rng(1000).choice(expected_monthly_vec, size=(4000, 12))
    healthy_alarm_vec = [cusum_alarm_bool(path_vec, calibration) for path_vec in healthy_path_mat]
    assert np.mean(healthy_alarm_vec) == pytest.approx(0.05, abs=0.012)
    assert cusum_alarm_bool([], calibration) is False

    dead_alarm_vec = [
        cusum_alarm_bool(np.random.default_rng(5000 + i).normal(-0.03, 0.04, 12), calibration) for i in range(500)
    ]
    assert np.mean(dead_alarm_vec) > 0.8
    with pytest.raises(ValueError, match="finite"):
        cusum_alarm_bool([-0.05, np.nan, -0.02], calibration)
