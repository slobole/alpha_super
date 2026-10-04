"""Part B: which inference unit keeps S3's error rate honest on an event panel (PROTOCOL.md, Part B).

Panel: 200 stocks x 2,520 sessions. r_{i,t} = beta_i * m_t + e_{i,t}; m_t GARCH(1,1) Student-t market, beta_i ~ U(0.5, 1.5),
e_{i,t} = 2% * standardised Student-t(5). Events: an oversold-style signal that fires more often after market down
days (so events cluster by date), independent of future returns. Specification detail (within the protocol's
"signal independent of future returns"): P(event on day t) = 2% * k_t / mean(k), k_t = 5 when m_t is in its lowest
decile, else 1. Hold = 5 sessions from t+1. Excess = forward 5-day return minus the equal-weight panel's.
`s3_edge` adds +0.04% per day on the 5 held days (+0.2% per event).

    uv run python scripts/research/scout_p2_calibration_20260930/run_s3_inference.py
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
STOCK_COUNT_INT, SESSION_COUNT_INT, HOLD_INT = 200, 2520, 5
EVENT_RATE_FLOAT, EDGE_PER_DAY_FLOAT, PERMUTATION_COUNT_INT = 0.02, 0.0004, 500
SEED_COUNT_INT = 200


def _panel(seed_int: int, edge_bool: bool) -> tuple[np.ndarray, np.ndarray]:
    rng_obj = np.random.default_rng(700_000 + seed_int + (50_000 if edge_bool else 0))
    market_vec = garch_returns(SESSION_COUNT_INT, rng_obj)
    beta_vec = rng_obj.uniform(0.5, 1.5, STOCK_COUNT_INT)
    idio_mat = 0.02 * rng_obj.standard_t(5, (SESSION_COUNT_INT, STOCK_COUNT_INT)) / np.sqrt(5 / 3)
    return_mat = market_vec[:, None] * beta_vec[None, :] + idio_mat

    cluster_vec = np.where(market_vec <= np.quantile(market_vec, 0.10), 5.0, 1.0)
    event_probability_vec = EVENT_RATE_FLOAT * cluster_vec / cluster_vec.mean()
    event_mat = rng_obj.random((SESSION_COUNT_INT, STOCK_COUNT_INT)) < event_probability_vec[:, None]
    event_mat[-HOLD_INT - 1 :] = False  # no forward window at the end
    if edge_bool:
        for day_offset_int in range(1, HOLD_INT + 1):
            # *** CRITICAL*** the planted edge pays on t+1..t+5, after the event day t.
            return_mat[day_offset_int:] += EDGE_PER_DAY_FLOAT * event_mat[:-day_offset_int]

    cum_mat = np.vstack([np.zeros(STOCK_COUNT_INT), np.cumsum(return_mat, axis=0)])
    forward_mat = np.full((SESSION_COUNT_INT, STOCK_COUNT_INT), np.nan)
    forward_mat[: SESSION_COUNT_INT - HOLD_INT] = cum_mat[1 + HOLD_INT : SESSION_COUNT_INT + 1] - cum_mat[1 : SESSION_COUNT_INT - HOLD_INT + 1]
    excess_mat = forward_mat - np.nanmean(forward_mat, axis=1, keepdims=True)
    return np.nan_to_num(excess_mat), event_mat


def _task(task_tuple) -> dict:
    seed_int, edge_bool = task_tuple
    excess_mat, event_mat = _panel(seed_int, edge_bool)
    event_excess_vec = excess_mat[event_mat]

    naive_t_float = float(event_excess_vec.mean() / (event_excess_vec.std(ddof=1) / np.sqrt(event_excess_vec.size)))
    event_count_vec = event_mat.sum(axis=1)
    date_mean_vec = np.where(event_count_vec > 0, (excess_mat * event_mat).sum(axis=1) / np.maximum(event_count_vec, 1), np.nan)
    nw_t_float = newey_west_mean_t_stat(date_mean_vec, HOLD_INT - 1).t_stat_float

    observed_float = float(event_excess_vec.mean())
    rng_obj = np.random.default_rng(900_000 + seed_int)
    null_vec = np.empty(PERMUTATION_COUNT_INT)
    for permutation_idx_int in range(PERMUTATION_COUNT_INT):
        permuted_event_mat = rng_obj.permuted(event_mat, axis=1)  # shuffle labels within each date
        null_vec[permutation_idx_int] = excess_mat[permuted_event_mat].mean()
    permutation_p_float = (1.0 + np.sum(null_vec >= observed_float)) / (1.0 + PERMUTATION_COUNT_INT)

    two_sided_cut_float = stats.norm.ppf(0.975)
    return {
        "case_str": "s3_edge" if edge_bool else "s3_null",
        "seed_int": seed_int,
        "event_count_int": int(event_mat.sum()),
        "naive_t_float": naive_t_float,
        "nw_t_float": nw_t_float,
        "permutation_p_float": float(permutation_p_float),
        "reject_naive_bool": abs(naive_t_float) > two_sided_cut_float,
        "reject_nw_bool": abs(nw_t_float) > two_sided_cut_float,
        "reject_permutation_bool": permutation_p_float <= 0.05,
    }


def main() -> None:
    OUTPUT_DIR_PATH.mkdir(parents=True, exist_ok=True)
    started_float = time.time()
    task_list = [(seed_int, edge_bool) for edge_bool in (False, True) for seed_int in range(SEED_COUNT_INT)]
    with Pool(14) as pool_obj:
        row_list = list(pool_obj.imap_unordered(_task, task_list, chunksize=4))
    pd.DataFrame(row_list).to_parquet(OUTPUT_DIR_PATH / "s3_inference.parquet")
    print("done", f"{time.time() - started_float:.0f}s", flush=True)


if __name__ == "__main__":
    main()
