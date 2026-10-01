"""P5 calibration (PROTOCOL.md): which MCPT score keeps size for volatility-timed ETF families.

All four scores are computed on the same real and shuffled histories (one null, four statistics). The factor f is a
column of the shuffled matrix too, because the F1 gate reads its realised volatility.

    uv run python scripts/research/scout_p5_calibration_20261002/run_score_calibration.py
"""

from __future__ import annotations

import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from alpha.scout.ledger import MAIN_CHECKOUT_ROOT_PATH  # noqa: E402
from alpha.stats.selection import plateau_choice  # noqa: E402

OUTPUT_DIR_PATH = MAIN_CHECKOUT_ROOT_PATH / "results" / "scout" / "p5_calibration"
SESSION_COUNT_INT, MONTH_INT, WARM_UP_MONTH_INT = 3780, 21, 13
BETA_VEC = np.array([0.3, 0.5, 0.7, 0.9, 1.1, 3.0])
DRIFT_FLOAT, EDGE_COEFFICIENT_FLOAT = 0.0003, 0.30
PERMUTATION_COUNT_INT, NULL_SEED_COUNT_INT, EDGE_SEED_COUNT_INT = 200, 200, 100
F1_GRID = (((1, 3), (1, 3, 6), (1, 3, 6, 12)), (10, 20, 40))
F2_GRID = ((100, 150, 200, 250), (16.0, 20.0, 24.0))
SCORE_TUPLE = ("SA", "SB", "SC", "SD")


def garch_path(session_count_int: int, rng_obj: np.random.Generator, annual_vol_float: float) -> tuple[np.ndarray, np.ndarray]:
    target_variance_float = annual_vol_float**2 / 252.0
    omega_float = target_variance_float * (1.0 - 0.08 - 0.90)
    shock_vec = rng_obj.standard_t(5, session_count_int) / np.sqrt(5 / 3)
    return_vec, sigma_vec = np.zeros(session_count_int), np.zeros(session_count_int)
    noise_prev_float, variance_prev_float = 0.0, target_variance_float
    for t_int in range(session_count_int):
        variance_float = omega_float + 0.08 * noise_prev_float**2 + 0.90 * variance_prev_float
        sigma_vec[t_int] = np.sqrt(variance_float)
        return_vec[t_int] = sigma_vec[t_int] * shock_vec[t_int]
        noise_prev_float, variance_prev_float = return_vec[t_int], variance_float
    return return_vec, sigma_vec


def make_matrix(seed_int: int, edge_bool: bool, family_str: str = "F1", paired_bool: bool = False) -> np.ndarray:
    """Columns: r1..r6, f, V (implied-volatility proxy, annualised %).

    Planted edges (amendment 1), neither changing the average drift: F1 = relative momentum, each asset's drift gains
    c × (its own 126-session mean − the six assets' mean); F2 = time-series momentum of asset 6, c × (its own
    126-session mean − the base drift). Means use returns up to t − 2.
    """
    rng_obj = np.random.default_rng(5_000_000 + seed_int + (100_000 if edge_bool and not paired_bool else 0))
    factor_vec, factor_sigma_vec = garch_path(SESSION_COUNT_INT, rng_obj, 0.16)
    idio_mat = np.column_stack([garch_path(SESSION_COUNT_INT, rng_obj, 0.10)[0] for _ in range(BETA_VEC.size)])
    return_mat = DRIFT_FLOAT + factor_vec[:, None] * BETA_VEC[None, :] + idio_mat
    if edge_bool:
        # *** CRITICAL*** the planted drift uses returns up to t − 2 (126-session mean, lagged one session).
        running_sum_vec = np.zeros(BETA_VEC.size)
        for t_int in range(SESSION_COUNT_INT):
            if t_int >= 128:
                mean_vec = running_sum_vec / 126.0
                if family_str == "F1":
                    return_mat[t_int] += EDGE_COEFFICIENT_FLOAT * (mean_vec - mean_vec.mean())
                else:
                    return_mat[t_int, 5] += EDGE_COEFFICIENT_FLOAT * (mean_vec[5] - DRIFT_FLOAT)
            if t_int >= 1:
                running_sum_vec += return_mat[t_int - 1]
            if t_int >= 127:
                running_sum_vec -= return_mat[t_int - 127]
    implied_vec = 1.15 * factor_sigma_vec * np.sqrt(252.0) * 100.0 * np.exp(rng_obj.normal(0.0, 0.1, SESSION_COUNT_INT))
    return np.column_stack([return_mat, factor_vec, implied_vec])


def _sharpe(daily_vec: np.ndarray) -> float:
    window_vec = daily_vec[WARM_UP_MONTH_INT * MONTH_INT :]
    sd_float = window_vec.std(ddof=1)
    return float(window_vec.mean() / sd_float * np.sqrt(252.0)) if sd_float > 0 else 0.0


def _hold(weight_by_decision_mat: np.ndarray, decision_vec: np.ndarray, return_mat: np.ndarray) -> np.ndarray:
    """Daily return of weights decided at the close of each decision session, held from the close of T+1."""
    daily_weight_mat = np.zeros_like(return_mat)
    for idx_int, t_int in enumerate(decision_vec):
        end_int = decision_vec[idx_int + 1] + 2 if idx_int + 1 < decision_vec.size else return_mat.shape[0]
        daily_weight_mat[t_int + 2 : end_int] = weight_by_decision_mat[idx_int]
    return (daily_weight_mat * return_mat).sum(axis=1)


def _vol_target_vec(daily_vec: np.ndarray) -> np.ndarray:
    realized_vec = pd.Series(daily_vec).rolling(20).std().shift(1).to_numpy() * np.sqrt(252.0)
    scale_vec = np.where(np.isfinite(realized_vec) & (realized_vec > 0), np.minimum(1.0, 0.10 / realized_vec), 0.0)
    return scale_vec * daily_vec


def _four_scores(config_daily_list: list[np.ndarray], grid_shape_tuple: tuple, baseline_vec: np.ndarray) -> np.ndarray:
    targeted_vec = _vol_target_vec(baseline_vec)
    own_vec = np.array([_sharpe(v) for v in config_daily_list])
    chosen_own = plateau_choice(own_vec, grid_shape_tuple).own_sharpe_float
    active_ew_vec = np.array([_sharpe(v - baseline_vec) for v in config_daily_list])
    active_vt_vec = np.array([_sharpe(v - targeted_vec) for v in config_daily_list])
    return np.array([
        chosen_own - _sharpe(baseline_vec),
        chosen_own - _sharpe(targeted_vec),
        plateau_choice(active_ew_vec, grid_shape_tuple).own_sharpe_float,
        plateau_choice(active_vt_vec, grid_shape_tuple).own_sharpe_float,
    ])


def f1_scores(matrix: np.ndarray) -> np.ndarray:
    return_mat, factor_vec, implied_vec = matrix[:, :6], matrix[:, 6], matrix[:, 7]
    price_mat = np.cumprod(1.0 + return_mat, axis=0)
    decision_vec = np.arange(MONTH_INT - 1, SESSION_COUNT_INT - 2, MONTH_INT)
    monthly_price_mat = price_mat[decision_vec, :5]
    daily_list = []
    for k_tuple in F1_GRID[0]:
        score_mat = np.full(monthly_price_mat.shape, np.nan)
        score_mat[12:] = np.mean([monthly_price_mat[12:] / monthly_price_mat[12 - k : -k] - 1.0 for k in k_tuple], axis=0)
        for window_int in F1_GRID[1]:
            realized_vec = pd.Series(factor_vec).rolling(window_int).std(ddof=0).to_numpy() * np.sqrt(252.0) * 100.0
            weight_mat = np.zeros((decision_vec.size, 6))
            for m_int in range(12, decision_vec.size):
                order_vec = np.argsort(-score_mat[m_int], kind="stable")
                for slot_int, asset_int in enumerate(order_vec):
                    rank_weight_float = (5 - slot_int) / 15.0
                    if score_mat[m_int, asset_int] > 0:
                        weight_mat[m_int, asset_int] = rank_weight_float
                    else:
                        weight_mat[m_int, 5] += rank_weight_float
                if not realized_vec[decision_vec[m_int]] < implied_vec[decision_vec[m_int]]:
                    weight_mat[m_int, 5] = 0.0
            daily_list.append(_hold(weight_mat, decision_vec, return_mat))
    return _four_scores(daily_list, (len(F1_GRID[0]), len(F1_GRID[1])), return_mat.mean(axis=1))


def f2_scores(matrix: np.ndarray) -> np.ndarray:
    asset_vec, implied_vec = matrix[:, 5], matrix[:, 7]
    price_vec = np.cumprod(1.0 + asset_vec)
    decision_vec = np.arange(MONTH_INT - 1, SESSION_COUNT_INT - 2, MONTH_INT)
    daily_list = []
    for sma_int in F2_GRID[0]:
        sma_vec = pd.Series(price_vec).rolling(sma_int).mean().to_numpy()
        for reference_float in F2_GRID[1]:
            weight_mat = np.zeros((decision_vec.size, 1))
            for m_int, t_int in enumerate(decision_vec):
                if np.isfinite(sma_vec[t_int]) and price_vec[t_int] > sma_vec[t_int]:
                    weight_mat[m_int, 0] = min(1.0, max(0.25, reference_float / implied_vec[t_int]))
            daily_list.append(_hold(weight_mat, decision_vec, asset_vec[:, None]))
    return _four_scores(daily_list, (len(F2_GRID[0]), len(F2_GRID[1])), asset_vec)


FAMILY_DICT = {"F1": f1_scores, "F2": f2_scores}


def _task(task_tuple) -> dict:
    family_str, seed_int, edge_bool = task_tuple
    matrix = make_matrix(seed_int, edge_bool, family_str)
    score_fn = FAMILY_DICT[family_str]
    observed_vec = score_fn(matrix)
    rng_obj = np.random.default_rng(6_000_000 + seed_int)
    null_mat = np.array([score_fn(matrix[rng_obj.permutation(matrix.shape[0])]) for _ in range(PERMUTATION_COUNT_INT)])
    p_vec = (1 + (null_mat >= observed_vec[None, :]).sum(axis=0)) / (1 + PERMUTATION_COUNT_INT)
    row_dict = {"family_str": family_str, "case_str": "edge" if edge_bool else "null", "seed_int": seed_int}
    for idx_int, score_str in enumerate(SCORE_TUPLE):
        row_dict[f"{score_str}_p_float"] = float(p_vec[idx_int])
        row_dict[f"{score_str}_observed_float"] = float(observed_vec[idx_int])
    return row_dict


def main() -> None:
    OUTPUT_DIR_PATH.mkdir(parents=True, exist_ok=True)
    started_float = time.time()
    task_list = [
        (family_str, seed_int, edge_bool)
        for family_str in FAMILY_DICT
        for edge_bool, count_int in ((False, NULL_SEED_COUNT_INT), (True, EDGE_SEED_COUNT_INT))
        for seed_int in range(count_int)
    ]
    with Pool(14) as pool_obj:
        row_list = []
        for row_dict in pool_obj.imap_unordered(_task, task_list):
            row_list.append(row_dict)
            if len(row_list) % 100 == 0:
                print(len(row_list), "of", len(task_list), f"{time.time() - started_float:.0f}s", flush=True)
    frame = pd.DataFrame(row_list)
    frame.to_parquet(OUTPUT_DIR_PATH / "score_calibration.parquet")
    summary_df = frame.groupby(["family_str", "case_str"]).agg(
        **{f"{s}_pass": (f"{s}_p_float", lambda p: float(np.mean(p <= 0.05))) for s in SCORE_TUPLE}
    )
    print(summary_df.round(3).to_string())
    print("done", f"{time.time() - started_float:.0f}s")


if __name__ == "__main__":
    main()
