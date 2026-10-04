"""Part B, EXPLORATORY (designed after seeing Part B): a panel with the two dependence sources the frozen panel lacked.

1. Sector factors: 10 sectors of 20 stocks, each with its own GARCH factor (1% daily), so excess returns of stocks in
   the same sector are correlated on the same date even after removing the equal-weight panel.
2. Persistent signal: an event on (stock, t) repeats on (stock, t+1) with probability 0.6, as oversold readings do,
   so the 5-day holds of one stock overlap.
Events still carry no information about future returns in `x_null`; `x_edge` adds +0.2% per event as before.
Clustering by date (after market down days) is kept. 100 seeds each. Not a decision input under PROTOCOL.md.
"""

from __future__ import annotations

import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

SCRIPT_DIR_PATH = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR_PATH))
sys.path.insert(0, str(SCRIPT_DIR_PATH.parents[2]))

from alpha.scout.ledger import MAIN_CHECKOUT_ROOT_PATH  # noqa: E402
from alpha.stats.newey_west import newey_west_mean_t_stat  # noqa: E402

from synth import garch_returns  # noqa: E402

OUTPUT_DIR_PATH = MAIN_CHECKOUT_ROOT_PATH / "results" / "scout" / "p2_calibration"
STOCK_COUNT_INT, SECTOR_COUNT_INT, SESSION_COUNT_INT, HOLD_INT = 200, 10, 2520, 5
EVENT_RATE_FLOAT, REPEAT_FLOAT, EDGE_PER_DAY_FLOAT, PERMUTATION_COUNT_INT = 0.01, 0.6, 0.0004, 300


def _panel(seed_int: int, edge_bool: bool) -> tuple[np.ndarray, np.ndarray]:
    rng_obj = np.random.default_rng(800_000 + seed_int + (50_000 if edge_bool else 0))
    market_vec = garch_returns(SESSION_COUNT_INT, rng_obj)
    sector_mat = np.column_stack(
        [garch_returns(SESSION_COUNT_INT, rng_obj, annual_vol_float=0.16) for _ in range(SECTOR_COUNT_INT)]
    )
    sector_of_stock_vec = np.repeat(np.arange(SECTOR_COUNT_INT), STOCK_COUNT_INT // SECTOR_COUNT_INT)
    beta_vec = rng_obj.uniform(0.5, 1.5, STOCK_COUNT_INT)
    idio_mat = 0.02 * rng_obj.standard_t(5, (SESSION_COUNT_INT, STOCK_COUNT_INT)) / np.sqrt(5 / 3)
    return_mat = market_vec[:, None] * beta_vec[None, :] + sector_mat[:, sector_of_stock_vec] + idio_mat

    cluster_vec = np.where(market_vec <= np.quantile(market_vec, 0.10), 5.0, 1.0)
    start_probability_vec = EVENT_RATE_FLOAT * cluster_vec / cluster_vec.mean()
    event_mat = np.zeros((SESSION_COUNT_INT, STOCK_COUNT_INT), dtype=bool)
    for t_int in range(SESSION_COUNT_INT):
        fresh_vec = rng_obj.random(STOCK_COUNT_INT) < start_probability_vec[t_int]
        repeat_vec = (rng_obj.random(STOCK_COUNT_INT) < REPEAT_FLOAT) & (event_mat[t_int - 1] if t_int else False)
        event_mat[t_int] = fresh_vec | repeat_vec
    event_mat[-HOLD_INT - 1 :] = False
    if edge_bool:
        for day_offset_int in range(1, HOLD_INT + 1):
            return_mat[day_offset_int:] += EDGE_PER_DAY_FLOAT * event_mat[:-day_offset_int]

    cum_mat = np.vstack([np.zeros(STOCK_COUNT_INT), np.cumsum(return_mat, axis=0)])
    forward_mat = np.zeros((SESSION_COUNT_INT, STOCK_COUNT_INT))
    forward_mat[: SESSION_COUNT_INT - HOLD_INT] = cum_mat[1 + HOLD_INT : SESSION_COUNT_INT + 1] - cum_mat[1 : SESSION_COUNT_INT - HOLD_INT + 1]
    excess_mat = forward_mat - forward_mat.mean(axis=1, keepdims=True)
    return excess_mat, event_mat


def _task(task_tuple) -> dict:
    seed_int, edge_bool = task_tuple
    excess_mat, event_mat = _panel(seed_int, edge_bool)
    event_excess_vec = excess_mat[event_mat]
    naive_t_float = float(event_excess_vec.mean() / (event_excess_vec.std(ddof=1) / np.sqrt(event_excess_vec.size)))
    event_count_vec = event_mat.sum(axis=1)
    date_mean_vec = np.where(event_count_vec > 0, (excess_mat * event_mat).sum(axis=1) / np.maximum(event_count_vec, 1), np.nan)
    nw_t_float = newey_west_mean_t_stat(date_mean_vec, HOLD_INT - 1).t_stat_float
    observed_float = float(event_excess_vec.mean())
    rng_obj = np.random.default_rng(950_000 + seed_int)
    null_vec = np.array([excess_mat[rng_obj.permuted(event_mat, axis=1)].mean() for _ in range(PERMUTATION_COUNT_INT)])
    permutation_p_float = (1.0 + np.sum(null_vec >= observed_float)) / (1.0 + PERMUTATION_COUNT_INT)
    cut_float = stats.norm.ppf(0.975)
    return {
        "case_str": "x_edge" if edge_bool else "x_null",
        "seed_int": seed_int,
        "naive_t_float": naive_t_float,
        "nw_t_float": nw_t_float,
        "reject_naive_bool": abs(naive_t_float) > cut_float,
        "reject_nw_bool": abs(nw_t_float) > cut_float,
        "reject_permutation_bool": permutation_p_float <= 0.05,
    }


def main() -> None:
    started_float = time.time()
    task_list = [(seed_int, edge_bool) for edge_bool in (False, True) for seed_int in range(100)]
    with Pool(14) as pool_obj:
        row_list = list(pool_obj.imap_unordered(_task, task_list, chunksize=2))
    frame = pd.DataFrame(row_list)
    frame.to_parquet(OUTPUT_DIR_PATH / "s3_inference_exploratory.parquet")
    print(frame.groupby("case_str")[["reject_naive_bool", "reject_nw_bool", "reject_permutation_bool"]].mean().round(3).to_string())
    print(frame.groupby("case_str")[["naive_t_float", "nw_t_float"]].std().round(2).to_string())
    print("done", f"{time.time() - started_float:.0f}s", flush=True)


if __name__ == "__main__":
    main()
