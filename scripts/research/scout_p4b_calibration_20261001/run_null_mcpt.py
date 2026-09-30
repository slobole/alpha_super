"""Part B (PROTOCOL.md): size and power of the per-asset MCPT null on synthetic point-in-time panels.

Two nulls on the same histories: N1 = `mcpt_live_spans` with membership strata, N2 = without strata.
Two families: short-term reversal (27-configuration grid) and momentum (9), plateau selection, score = the Sharpe of
the daily active return of the selected configuration over the equal-weight member portfolio (amendment 2).
Panel as amended: market drift +0.04% a day and 50 index members of 150 stocks (member tenure near the S&P 500's).

    uv run python scripts/research/scout_p4b_calibration_20261001/run_null_mcpt.py
"""

from __future__ import annotations

import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR_PATH = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR_PATH.parent / "scout_p2_calibration_20260930"))
sys.path.insert(0, str(SCRIPT_DIR_PATH.parents[2]))

from synth import garch_returns

from alpha.scout.ledger import MAIN_CHECKOUT_ROOT_PATH
from alpha.stats.mcpt import mcpt_live_spans
from alpha.stats.selection import plateau_choice

OUTPUT_DIR_PATH = MAIN_CHECKOUT_ROOT_PATH / "results" / "scout" / "p4b_calibration"
SESSION_COUNT_INT, STOCK_COUNT_INT, SECTOR_COUNT_INT, MEMBER_COUNT_INT = 2520, 150, 5, 50
DRIFT_PER_DAY_FLOAT = 0.0004
RECONSTITUTION_INT, WARM_UP_INT = 63, 260
PERMUTATION_COUNT_INT, NULL_SEED_COUNT_INT, EDGE_SEED_COUNT_INT = 200, 200, 100
REVERSAL_GRID = ((2, 3, 5), (0.05, 0.10, 0.20), (3, 5, 10))  # lookback, bottom quantile, hold
MOMENTUM_GRID = ((63, 126, 252), (5, 10, 20))  # lookback, top K
FAMILY_SEED_OFFSET_DICT = {"reversal": 0, "momentum": 500_000}


# ---------------------------------------------------------------- synthetic point-in-time panel
def make_panel(family_str: str, seed_int: int, edge_bool: bool) -> tuple[np.ndarray, np.ndarray]:
    """(returns with NaN where unlisted, membership mask)."""
    rng_obj = np.random.default_rng(2_000_000 + FAMILY_SEED_OFFSET_DICT[family_str] + seed_int + (100_000 if edge_bool else 0))
    T, N = SESSION_COUNT_INT, STOCK_COUNT_INT
    market_vec = garch_returns(T, rng_obj)
    sector_mat = np.column_stack([garch_returns(T, rng_obj, annual_vol_float=0.12) for _ in range(SECTOR_COUNT_INT)])
    sector_of_stock_vec = rng_obj.integers(0, SECTOR_COUNT_INT, N)
    beta_vec = rng_obj.uniform(0.5, 1.5, N)
    idio_mat = 0.02 * rng_obj.standard_t(5, (T, N)) / np.sqrt(5 / 3)
    return_mat = DRIFT_PER_DAY_FLOAT + market_vec[:, None] * beta_vec[None, :] + sector_mat[:, sector_of_stock_vec] + idio_mat

    start_vec = np.zeros(N, dtype=int)
    late_vec = rng_obj.choice(N, 50, replace=False)
    start_vec[late_vec] = rng_obj.integers(0, 2000, late_vec.size)
    end_vec = np.full(N, T)
    for stock_idx_int in rng_obj.choice(N, 30, replace=False):
        if start_vec[stock_idx_int] + 500 < T:
            end_vec[stock_idx_int] = rng_obj.integers(start_vec[stock_idx_int] + 500, T)
    date_vec = np.arange(T)[:, None]
    listed_mat = (date_vec >= start_vec[None, :]) & (date_vec < end_vec[None, :])

    if edge_bool and family_str == "momentum":
        # *** CRITICAL*** the planted drift uses returns up to t − 2 only (126-session mean, lagged one session), and
        # only listed returns (pre-listing returns are never observable).
        return_mat = np.where(listed_mat, return_mat, 0.0)
        running_sum_vec = np.zeros(N)
        for t_int in range(T):
            if t_int >= 128:
                return_mat[t_int] += 0.15 * running_sum_vec / 126.0 * listed_mat[t_int]
            if t_int >= 1:
                running_sum_vec += return_mat[t_int - 1]
            if t_int >= 127:
                running_sum_vec -= return_mat[t_int - 127]
    return_mat = np.where(listed_mat, return_mat, np.nan)

    price_mat = np.cumprod(1.0 + np.nan_to_num(return_mat), axis=0)
    cap_mat = np.where(listed_mat, price_mat * rng_obj.lognormal(0.0, 1.0, N)[None, :], np.nan)
    member_mat = np.zeros((T, N), dtype=bool)
    current_vec = np.zeros(N, dtype=bool)
    for t_int in range(T):
        if t_int % RECONSTITUTION_INT == 0:
            order_vec = np.argsort(-np.nan_to_num(cap_mat[t_int], nan=-np.inf))
            current_vec = np.zeros(N, dtype=bool)
            current_vec[order_vec[:MEMBER_COUNT_INT]] = True
            current_vec &= listed_mat[t_int]
        member_mat[t_int] = current_vec & listed_mat[t_int]

    if edge_bool and family_str == "reversal":
        rank_df = _trailing_return_df(return_mat, 3).where(member_mat).rank(axis=1, pct=True)
        trigger_mat = (rank_df <= 0.05).to_numpy()
        for offset_int in range(1, 6):
            # *** CRITICAL*** the planted edge pays on t+1..t+5, after the trigger day t.
            return_mat[offset_int:] += 0.0002 * trigger_mat[:-offset_int]
        return_mat = np.where(listed_mat, return_mat, np.nan)
    return return_mat, member_mat


# ---------------------------------------------------------------- searches
def _trailing_return_df(return_mat: np.ndarray, window_int: int, skip_int: int = 0) -> pd.DataFrame:
    log_price_mat = np.cumsum(np.log1p(np.nan_to_num(return_mat)), axis=0)
    log_price_mat = np.where(np.isfinite(return_mat), log_price_mat, np.nan)
    log_price_df = pd.DataFrame(log_price_mat)
    return np.exp(log_price_df.shift(skip_int) - log_price_df.shift(window_int)) - 1.0


def _sharpe(daily_vec: np.ndarray) -> float:
    window_vec = daily_vec[WARM_UP_INT:]
    standard_deviation_float = window_vec.std(ddof=1)
    return float(window_vec.mean() / standard_deviation_float * np.sqrt(252.0)) if standard_deviation_float > 0 else 0.0


def _baseline_daily_vec(return_mat: np.ndarray, member_mat: np.ndarray) -> np.ndarray:
    """Equal-weight members, decided at close t and earning from t+2, like the strategies."""
    held_mat = np.zeros_like(member_mat)
    held_mat[2:] = member_mat[:-2]
    count_vec = held_mat.sum(axis=1)
    return np.where(count_vec > 0, (np.nan_to_num(return_mat) * held_mat).sum(axis=1) / np.maximum(count_vec, 1), 0.0)


def reversal_search(return_mat: np.ndarray, member_mat: np.ndarray) -> float:
    filled_mat = np.nan_to_num(return_mat)
    baseline_vec = _baseline_daily_vec(return_mat, member_mat)
    sharpe_list = []
    for lookback_int in REVERSAL_GRID[0]:
        rank_mat = _trailing_return_df(return_mat, lookback_int).where(member_mat).rank(axis=1, pct=True).to_numpy()
        for quantile_float in REVERSAL_GRID[1]:
            event_mat = np.nan_to_num(rank_mat, nan=2.0) <= quantile_float
            cohort_list = []
            for lag_int in range(2, max(REVERSAL_GRID[2]) + 2):  # decided at close t, held from close t+1
                cohort_vec = np.zeros(return_mat.shape[0])
                count_vec = event_mat[:-lag_int].sum(axis=1)
                cohort_vec[lag_int:] = np.where(count_vec > 0, (event_mat[:-lag_int] * filled_mat[lag_int:]).sum(axis=1) / np.maximum(count_vec, 1), 0.0)
                cohort_list.append(cohort_vec)
            for hold_int in REVERSAL_GRID[2]:
                sharpe_list.append(_sharpe(np.mean(cohort_list[:hold_int], axis=0) - baseline_vec))
    choice = plateau_choice(np.array(sharpe_list), tuple(len(axis) for axis in REVERSAL_GRID))
    return choice.own_sharpe_float


def momentum_search(return_mat: np.ndarray, member_mat: np.ndarray) -> float:
    filled_mat = np.nan_to_num(return_mat)
    baseline_vec = _baseline_daily_vec(return_mat, member_mat)
    T = return_mat.shape[0]
    rebalance_vec = np.arange(max(MOMENTUM_GRID[0]) + 1, T - 2, 21)
    sharpe_list = []
    for lookback_int in MOMENTUM_GRID[0]:
        score_mat = _trailing_return_df(return_mat, lookback_int, skip_int=5).where(member_mat).to_numpy()
        for top_int in MOMENTUM_GRID[1]:
            weight_mat = np.zeros_like(filled_mat)
            for rebalance_idx_int, t_int in enumerate(rebalance_vec):
                score_vec = np.nan_to_num(score_mat[t_int], nan=-np.inf)
                chosen_vec = np.argsort(-score_vec)[:top_int]
                chosen_vec = chosen_vec[np.isfinite(score_vec[chosen_vec])]
                end_int = rebalance_vec[rebalance_idx_int + 1] + 2 if rebalance_idx_int + 1 < rebalance_vec.size else T
                weight_mat[t_int + 2 : end_int, chosen_vec] = 1.0 / top_int
            sharpe_list.append(_sharpe((weight_mat * filled_mat).sum(axis=1) - baseline_vec))
    choice = plateau_choice(np.array(sharpe_list), tuple(len(axis) for axis in MOMENTUM_GRID))
    return choice.own_sharpe_float


SEARCH_DICT = {"reversal": reversal_search, "momentum": momentum_search}


def _task(task_tuple) -> dict:
    family_str, seed_int, edge_bool = task_tuple
    return_mat, member_mat = make_panel(family_str, seed_int, edge_bool)
    search_fn = SEARCH_DICT[family_str]
    row_dict = {"family_str": family_str, "case_str": "edge" if edge_bool else "null", "seed_int": seed_int}
    for null_str, strata_mat in (("N1", member_mat.astype(int)), ("N2", None)):
        result = mcpt_live_spans(lambda r: search_fn(r, member_mat), return_mat, PERMUTATION_COUNT_INT, 3_000_000 + seed_int, strata_mat)
        row_dict[f"{null_str}_p_float"] = result.p_value_float
        row_dict["observed_score_float"] = result.observed_score_float
        row_dict[f"{null_str}_null_median_float"] = float(np.median(result.null_score_vec))
    row_dict["member_share_float"] = float(member_mat.sum() / np.isfinite(return_mat).sum())
    return row_dict


def main() -> None:
    OUTPUT_DIR_PATH.mkdir(parents=True, exist_ok=True)
    started_float = time.time()
    task_list = [
        (family_str, seed_int, edge_bool)
        for family_str in SEARCH_DICT
        for edge_bool, count_int in ((False, NULL_SEED_COUNT_INT), (True, EDGE_SEED_COUNT_INT))
        for seed_int in range(count_int)
    ]
    with Pool(14) as pool_obj:
        row_list = []
        for row_dict in pool_obj.imap_unordered(_task, task_list):
            row_list.append(row_dict)
            if len(row_list) % 50 == 0:
                print(len(row_list), "of", len(task_list), f"{time.time() - started_float:.0f}s", flush=True)
    frame = pd.DataFrame(row_list)
    frame.to_parquet(OUTPUT_DIR_PATH / "null_mcpt.parquet")
    summary_df = frame.groupby(["family_str", "case_str"]).agg(
        seeds_int=("seed_int", "size"),
        n1_pass=("N1_p_float", lambda p: float(np.mean(p <= 0.05))),
        n2_pass=("N2_p_float", lambda p: float(np.mean(p <= 0.05))),
        observed_median=("observed_score_float", "median"),
        n1_null_median=("N1_null_median_float", "median"),
        n2_null_median=("N2_null_median_float", "median"),
    )
    print(summary_df.round(4).to_string())
    print("done", f"{time.time() - started_float:.0f}s")


if __name__ == "__main__":
    main()
