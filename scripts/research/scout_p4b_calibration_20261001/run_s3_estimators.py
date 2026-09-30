"""Part A (PROTOCOL.md): size and power of S3's per-event estimator and of the shift placebo, on the two P2 S3 panels.

    uv run python scripts/research/scout_p4b_calibration_20261001/run_s3_estimators.py
"""

from __future__ import annotations

import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR_PATH = Path(__file__).resolve().parent
P2_DIR_PATH = SCRIPT_DIR_PATH.parent / "scout_p2_calibration_20260930"
sys.path.insert(0, str(P2_DIR_PATH))
sys.path.insert(0, str(SCRIPT_DIR_PATH.parents[2]))

import run_s3_inference as standard_module
import run_s3_inference_exploratory as dependent_module

from alpha.scout.ledger import MAIN_CHECKOUT_ROOT_PATH
from alpha.scout.stations.s3_edge import _per_event_nw
from alpha.stats.newey_west import newey_west_mean_t_stat

OUTPUT_DIR_PATH = MAIN_CHECKOUT_ROOT_PATH / "results" / "scout" / "p4b_calibration"
HOLD_INT, SHIFT_COUNT_INT, MIN_SHIFT_INT = 5, 200, 252
NULL_SEED_COUNT_INT, EDGE_SEED_COUNT_INT = 400, 200
PANEL_DICT = {"standard": standard_module._panel, "dependent": dependent_module._panel}


def _date_mean_vec(excess_mat: np.ndarray, event_mat: np.ndarray) -> np.ndarray:
    count_vec = event_mat.sum(axis=1)
    return np.where(count_vec > 0, (excess_mat * event_mat).sum(axis=1) / np.maximum(count_vec, 1), np.nan)


def _task(task_tuple) -> dict:
    panel_str, seed_int, edge_bool = task_tuple
    excess_mat, event_mat = PANEL_DICT[panel_str](seed_int, edge_bool)
    date_mean_vec = _date_mean_vec(excess_mat, event_mat)
    date_t_float = newey_west_mean_t_stat(date_mean_vec, HOLD_INT - 1).t_stat_float
    per_event_dict = _per_event_nw(pd.DataFrame(excess_mat), pd.DataFrame(event_mat), HOLD_INT)

    observed_float = float(np.nanmean(date_mean_vec))
    rng_obj = np.random.default_rng(1_000_000 + seed_int + (50_000 if edge_bool else 0))
    shift_vec = rng_obj.integers(MIN_SHIFT_INT, event_mat.shape[0] - MIN_SHIFT_INT, SHIFT_COUNT_INT)
    null_vec = np.array([np.nanmean(_date_mean_vec(excess_mat, np.roll(event_mat, shift_int, axis=0))) for shift_int in shift_vec])
    placebo_p_float = float((1 + np.sum(null_vec >= observed_float)) / (1 + SHIFT_COUNT_INT))
    return {
        "panel_str": panel_str,
        "case_str": "edge" if edge_bool else "null",
        "seed_int": seed_int,
        "date_t_float": date_t_float,
        "per_event_t_float": per_event_dict["t_float"],
        "placebo_p_float": placebo_p_float,
    }


def main() -> None:
    OUTPUT_DIR_PATH.mkdir(parents=True, exist_ok=True)
    started_float = time.time()
    task_list = [
        (panel_str, seed_int, edge_bool)
        for panel_str in PANEL_DICT
        for edge_bool, count_int in ((False, NULL_SEED_COUNT_INT), (True, EDGE_SEED_COUNT_INT))
        for seed_int in range(count_int)
    ]
    with Pool(14) as pool_obj:
        row_list = list(pool_obj.imap_unordered(_task, task_list, chunksize=4))
    frame = pd.DataFrame(row_list)
    frame.to_parquet(OUTPUT_DIR_PATH / "s3_estimators.parquet")
    summary_df = frame.groupby(["panel_str", "case_str"]).agg(
        seeds_int=("seed_int", "size"),
        date_two_sided=("date_t_float", lambda t: float(np.mean(np.abs(t) > 1.96))),
        per_event_two_sided=("per_event_t_float", lambda t: float(np.mean(np.abs(t) > 1.96))),
        placebo_one_sided=("placebo_p_float", lambda p: float(np.mean(p <= 0.05))),
    )
    print(summary_df.round(4).to_string())
    print("done", f"{time.time() - started_float:.0f}s")


if __name__ == "__main__":
    main()
