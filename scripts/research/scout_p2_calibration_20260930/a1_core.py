"""Amendment 1 (PROTOCOL_AMENDMENT_1.md): data, two search families, and every candidate gate."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from alpha.scout.trials import effective_trial_count
from alpha.stats.mcpt import mcpt, volatility_strata_vec
from alpha.stats.pbo import probability_of_backtest_overfitting
from alpha.stats.psr_dsr import (
    deflated_sharpe_ratio,
    null_selected_sharpe_benchmark,
    probabilistic_sharpe_ratio,
    sharpe_moments,
)
from alpha.stats.selection import column_sharpe_vec, make_plateau_selector, plateau_choice, plateau_choice_index_mat
from alpha.stats.walk_forward import REGISTERED_DESIGN, design_sensitivity_df, run_walk_forward

SESSION_COUNT_INT = 5040
WARM_UP_INT = 250
MOMENTUM_WINDOW_INT = 20
MCPT_PERMUTATION_COUNT_INT = 200
BENCHMARK_DRAW_COUNT_INT = 20000


@dataclass(frozen=True)
class Family:
    name_str: str
    lookback_tuple: tuple[int, ...]
    threshold_tuple: tuple[float, ...]

    @property
    def grid_shape_tuple(self) -> tuple[int, int]:
        return (len(self.lookback_tuple), len(self.threshold_tuple))

    @property
    def true_index_int(self) -> int:
        return self.lookback_tuple.index(20) * len(self.threshold_tuple) + self.threshold_tuple.index(0.0)


FAMILY_A = Family("A_varied", (5, 10, 20, 40, 60, 120, 250), (0.0, 0.25, 0.5))
FAMILY_B = Family("B_near_duplicates", (16, 18, 20, 22, 24), (0.0, 0.05, 0.1))


def garch_returns(session_count_int: int, rng_obj: np.random.Generator, momentum_a_float: float = 0.0) -> np.ndarray:
    """GARCH(1,1) on r^2_{t-1} (as the protocol states), Student-t(5) shocks, 16% annual vol, planted momentum."""
    alpha_float, beta_float, student_df_int = 0.08, 0.90, 5
    target_variance_float = 0.16**2 / 252.0
    omega_float = target_variance_float * (1.0 - alpha_float - beta_float)
    shock_vec = rng_obj.standard_t(student_df_int, session_count_int) / np.sqrt(student_df_int / (student_df_int - 2.0))
    return_vec = np.zeros(session_count_int)
    return_prev_float, variance_prev_float, running_sum_float = 0.0, target_variance_float, 0.0
    for t_int in range(session_count_int):
        variance_float = omega_float + alpha_float * return_prev_float**2 + beta_float * variance_prev_float
        # *** CRITICAL*** the momentum term uses only returns before t.
        momentum_float = running_sum_float / MOMENTUM_WINDOW_INT if t_int >= MOMENTUM_WINDOW_INT else 0.0
        return_vec[t_int] = momentum_a_float * momentum_float + np.sqrt(variance_float) * shock_vec[t_int]
        running_sum_float += return_vec[t_int]
        if t_int >= MOMENTUM_WINDOW_INT:
            running_sum_float -= return_vec[t_int - MOMENTUM_WINDOW_INT]
        return_prev_float, variance_prev_float = return_vec[t_int], variance_float
    return return_vec


def config_return_mat(return_vec, family: Family) -> np.ndarray:
    """(T − 250) × configurations, grid order (lookback outer, threshold inner). Same rule as the first run."""
    return_arr = np.asarray(return_vec, dtype=float)
    session_count_int = return_arr.size
    cum_vec = np.concatenate(([0.0], np.cumsum(return_arr)))
    cum_square_vec = np.concatenate(([0.0], np.cumsum(return_arr**2)))
    column_list = []
    for lookback_int in family.lookback_tuple:
        end_idx = np.arange(lookback_int, session_count_int + 1)
        window_sum_vec = cum_vec[end_idx] - cum_vec[end_idx - lookback_int]
        window_square_vec = cum_square_vec[end_idx] - cum_square_vec[end_idx - lookback_int]
        variance_vec = np.clip((window_square_vec - window_sum_vec**2 / lookback_int) / (lookback_int - 1), 1e-18, None)
        z_vec = np.full(session_count_int, np.nan)
        z_vec[lookback_int - 1 :] = window_sum_vec / (np.sqrt(variance_vec) * np.sqrt(lookback_int))
        for threshold_float in family.threshold_tuple:
            position_vec = np.where(np.abs(z_vec) > threshold_float, np.sign(z_vec), 0.0)
            # *** CRITICAL*** position decided on day t earns day t+1's return.
            strategy_vec = np.zeros(session_count_int)
            strategy_vec[1:] = np.nan_to_num(position_vec[:-1]) * return_arr[1:]
            column_list.append(strategy_vec[WARM_UP_INT:])
    return np.column_stack(column_list)


def mean_true_sharpe(momentum_a_float: float, path_count_int: int = 8, years_int: int = 100) -> float:
    sharpe_list = []
    for path_idx_int in range(path_count_int):
        return_vec = garch_returns(252 * years_int, np.random.default_rng(424_242 + path_idx_int), momentum_a_float)
        strategy_vec = config_return_mat(return_vec, FAMILY_A)[:, FAMILY_A.true_index_int]
        sharpe_list.append(strategy_vec.mean() / strategy_vec.std(ddof=1) * np.sqrt(252))
    return float(np.mean(sharpe_list))


def calibrate_momentum_a(target_sharpe_float: float) -> float:
    low_float, high_float = 0.0, 0.6
    while high_float - low_float > 2e-4:
        middle_float = 0.5 * (low_float + high_float)
        if mean_true_sharpe(middle_float) < target_sharpe_float:
            low_float = middle_float
        else:
            high_float = middle_float
    return 0.5 * (low_float + high_float)


def evaluate_history(return_vec, family: Family, seed_int: int) -> dict:
    return_arr = np.asarray(return_vec, dtype=float)
    grid_shape_tuple = family.grid_shape_tuple
    config_mat = config_return_mat(return_arr, family)
    sharpe_vec = column_sharpe_vec(config_mat)
    choice = plateau_choice(sharpe_vec, grid_shape_tuple)
    chosen_vec = config_mat[:, choice.flat_index_int]
    observation_count_int = chosen_vec.size
    moments = sharpe_moments(chosen_vec)

    # DSR-corr: benchmark = expected plateau-selected Sharpe under the null, from this family's correlation matrix.
    correlation_mat = np.corrcoef(config_mat, rowvar=False)
    benchmark_float = null_selected_sharpe_benchmark(
        correlation_mat, observation_count_int, lambda draw_mat: plateau_choice_index_mat(draw_mat, grid_shape_tuple),
        draw_count_int=BENCHMARK_DRAW_COUNT_INT, random_seed_int=seed_int,
    )
    dsr_corr_float = probabilistic_sharpe_ratio(
        moments.sharpe_float, observation_count_int, moments.skewness_float, moments.kurtosis_float, benchmark_float
    )
    # Old clustered DSR, for comparison only.
    config_df = pd.DataFrame(config_mat, index=pd.bdate_range("2000-01-03", periods=observation_count_int))
    n_eff_float = effective_trial_count(config_df)
    dsr_cluster_float = deflated_sharpe_ratio(chosen_vec, float(np.var(sharpe_vec / np.sqrt(252), ddof=1)), n_eff_float).deflated_sharpe_float

    def search_fn(return_mat: np.ndarray) -> float:
        search_sharpe_vec = column_sharpe_vec(config_return_mat(return_mat[:, 0], family))
        return float(search_sharpe_vec[plateau_choice(search_sharpe_vec, grid_shape_tuple).flat_index_int])

    plain = mcpt(search_fn, return_arr[:, None], MCPT_PERMUTATION_COUNT_INT, seed_int)
    stratified = mcpt(search_fn, return_arr[:, None], MCPT_PERMUTATION_COUNT_INT, seed_int, volatility_strata_vec(return_arr[:, None], 63, 3))

    selector_fn = make_plateau_selector(grid_shape_tuple)
    registered = run_walk_forward(config_df, selector_fn, REGISTERED_DESIGN)
    positive_count_int = int(design_sensitivity_df(config_df, selector_fn)["oos_positive_bool"].fillna(False).astype(bool).sum())
    wf_pass_bool = bool(
        np.isfinite(registered.efficiency_float) and registered.efficiency_float >= 0.5
        and registered.oos_sharpe_float > 0 and positive_count_int >= 6
    )
    pbo_float = probability_of_backtest_overfitting(
        config_mat, 10, select_fn=lambda train_vec: plateau_choice(train_vec, grid_shape_tuple).flat_index_int
    ).pbo_float

    return {
        "chosen_index_int": choice.flat_index_int,
        "chosen_sharpe_float": choice.own_sharpe_float,
        "true_config_sharpe_float": float(sharpe_vec[family.true_index_int]),
        "benchmark_annual_float": benchmark_float * np.sqrt(252),
        "n_eff_float": n_eff_float,
        "dsr_corr_float": dsr_corr_float,
        "dsr_cluster_float": dsr_cluster_float,
        "mcpt_plain_p_float": plain.p_value_float,
        "mcpt_stratified_p_float": stratified.p_value_float,
        "wf_pass_bool": wf_pass_bool,
        "pbo_float": pbo_float,
        "naive_t_float": choice.own_sharpe_float * np.sqrt(observation_count_int / 252),
    }
