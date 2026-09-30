"""Synthetic data and the registered 21-configuration search family (PROTOCOL.md, Part A).

Research-only. Returns are daily, zero drift. The planted edge is time-series momentum:

    r_t = a · m_{t−1} + u_t,   m_{t−1} = mean(r_{t−20}, ..., r_{t−1}),   u_t = σ_t · ε_t (GARCH(1,1), Student-t(5))

Search family: z_t = S_L(t) / (σ̂_L(t) · √L), with S_L the sum and σ̂_L the sample std of the last L returns;
position for day t+1 = sign(z_t) if |z_t| > θ else 0.
"""

from __future__ import annotations

import numpy as np

SESSION_COUNT_INT = 5040
LOOKBACK_TUPLE = (5, 10, 20, 40, 60, 120, 250)
THRESHOLD_TUPLE = (0.0, 0.25, 0.5)
GRID_SHAPE_TUPLE = (len(LOOKBACK_TUPLE), len(THRESHOLD_TUPLE))
WARM_UP_INT = max(LOOKBACK_TUPLE)
TRUE_CONFIG_INDEX_INT = LOOKBACK_TUPLE.index(20) * len(THRESHOLD_TUPLE) + THRESHOLD_TUPLE.index(0.0)
MOMENTUM_WINDOW_INT = 20


def iid_returns(session_count_int: int, rng_obj: np.random.Generator, daily_vol_float: float = 0.01) -> np.ndarray:
    return rng_obj.normal(0.0, daily_vol_float, session_count_int)


def garch_returns(
    session_count_int: int,
    rng_obj: np.random.Generator,
    momentum_a_float: float = 0.0,
    annual_vol_float: float = 0.16,
    alpha_float: float = 0.08,
    beta_float: float = 0.90,
    student_df_int: int = 5,
) -> np.ndarray:
    target_variance_float = annual_vol_float**2 / 252.0
    omega_float = target_variance_float * (1.0 - alpha_float - beta_float)
    shock_vec = rng_obj.standard_t(student_df_int, session_count_int) / np.sqrt(student_df_int / (student_df_int - 2.0))
    return_vec = np.zeros(session_count_int)
    noise_prev_float, variance_prev_float, running_sum_float = 0.0, target_variance_float, 0.0
    for t_int in range(session_count_int):
        variance_float = omega_float + alpha_float * noise_prev_float**2 + beta_float * variance_prev_float
        noise_float = np.sqrt(variance_float) * shock_vec[t_int]
        # *** CRITICAL*** the momentum term uses only returns before t.
        momentum_float = running_sum_float / MOMENTUM_WINDOW_INT if t_int >= MOMENTUM_WINDOW_INT else 0.0
        return_vec[t_int] = momentum_a_float * momentum_float + noise_float
        running_sum_float += return_vec[t_int]
        if t_int >= MOMENTUM_WINDOW_INT:
            running_sum_float -= return_vec[t_int - MOMENTUM_WINDOW_INT]
        noise_prev_float, variance_prev_float = noise_float, variance_float
    return return_vec


def config_return_mat(return_vec) -> np.ndarray:
    """(T − WARM_UP) × 21 strategy returns, columns in grid order (lookback outer, threshold inner)."""
    return_arr = np.asarray(return_vec, dtype=float)
    session_count_int = return_arr.size
    cum_vec = np.concatenate(([0.0], np.cumsum(return_arr)))
    cum_square_vec = np.concatenate(([0.0], np.cumsum(return_arr**2)))
    column_list = []
    for lookback_int in LOOKBACK_TUPLE:
        end_idx = np.arange(lookback_int, session_count_int + 1)  # window [end − L, end) covers r_{end−L..end−1}
        window_sum_vec = cum_vec[end_idx] - cum_vec[end_idx - lookback_int]
        window_square_vec = cum_square_vec[end_idx] - cum_square_vec[end_idx - lookback_int]
        variance_vec = np.clip((window_square_vec - window_sum_vec**2 / lookback_int) / (lookback_int - 1), 1e-18, None)
        z_vec = np.full(session_count_int, np.nan)
        z_vec[lookback_int - 1 :] = window_sum_vec / (np.sqrt(variance_vec) * np.sqrt(lookback_int))  # z known after close t
        for threshold_float in THRESHOLD_TUPLE:
            position_vec = np.where(np.abs(z_vec) > threshold_float, np.sign(z_vec), 0.0)
            # *** CRITICAL*** position decided on day t earns day t+1's return.
            strategy_vec = np.zeros(session_count_int)
            strategy_vec[1:] = np.nan_to_num(position_vec[:-1]) * return_arr[1:]
            column_list.append(strategy_vec[WARM_UP_INT:])
    return np.column_stack(column_list)


def true_config_sharpe(momentum_a_float: float, years_int: int = 400, seed_int: int = 12345) -> float:
    return_vec = garch_returns(252 * years_int, np.random.default_rng(seed_int), momentum_a_float)
    strategy_vec = config_return_mat(return_vec)[:, TRUE_CONFIG_INDEX_INT]
    return float(strategy_vec.mean() / strategy_vec.std(ddof=1) * np.sqrt(252))


def calibrate_momentum_a(target_sharpe_float: float, tolerance_float: float = 0.01) -> float:
    low_float, high_float = 0.0, 0.6
    for _ in range(40):
        middle_float = 0.5 * (low_float + high_float)
        if true_config_sharpe(middle_float) < target_sharpe_float:
            low_float = middle_float
        else:
            high_float = middle_float
        if high_float - low_float < 1e-4:
            break
    result_float = 0.5 * (low_float + high_float)
    achieved_float = true_config_sharpe(result_float)
    if abs(achieved_float - target_sharpe_float) > tolerance_float * 5:
        raise RuntimeError(f"Calibration missed: target {target_sharpe_float}, achieved {achieved_float}.")
    return result_float
