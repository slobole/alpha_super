"""Part A runner: every case x seed through every gate, in parallel. Writes one parquet row per history.

    uv run python scripts/research/scout_p2_calibration_20260930/run_synthetic.py
"""

from __future__ import annotations

import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR_PATH = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR_PATH))
sys.path.insert(0, str(SCRIPT_DIR_PATH.parents[2]))

from alpha.scout.ledger import MAIN_CHECKOUT_ROOT_PATH  # noqa: E402

from gates import evaluate_history  # noqa: E402
from synth import SESSION_COUNT_INT, calibrate_momentum_a, garch_returns, iid_returns  # noqa: E402

OUTPUT_DIR_PATH = MAIN_CHECKOUT_ROOT_PATH / "results" / "scout" / "p2_calibration"
CASE_SEED_COUNT_DICT = {"noise_iid": 100, "noise_garch": 200, "edge_030": 100, "edge_050": 100, "edge_080": 100}
EDGE_TARGET_DICT = {"edge_030": 0.3, "edge_050": 0.5, "edge_080": 0.8}


def _task(task_tuple) -> dict:
    case_str, seed_int, momentum_a_float = task_tuple
    rng_obj = np.random.default_rng(10_000 * (list(CASE_SEED_COUNT_DICT).index(case_str) + 1) + seed_int)
    if case_str == "noise_iid":
        return_vec = iid_returns(SESSION_COUNT_INT, rng_obj)
    else:
        return_vec = garch_returns(SESSION_COUNT_INT, rng_obj, momentum_a_float)
    result_dict = evaluate_history(return_vec, seed_int)
    result_dict.update({"case_str": case_str, "seed_int": seed_int, "momentum_a_float": momentum_a_float})
    return result_dict


def main() -> None:
    OUTPUT_DIR_PATH.mkdir(parents=True, exist_ok=True)
    started_float = time.time()
    momentum_a_dict = {case_str: calibrate_momentum_a(target_float) for case_str, target_float in EDGE_TARGET_DICT.items()}
    (OUTPUT_DIR_PATH / "momentum_a.json").write_text(json.dumps(momentum_a_dict, indent=2), encoding="utf-8")
    print("calibrated a:", momentum_a_dict, f"({time.time() - started_float:.0f}s)", flush=True)

    task_list = [
        (case_str, seed_int, momentum_a_dict.get(case_str, 0.0))
        for case_str, seed_count_int in CASE_SEED_COUNT_DICT.items()
        for seed_int in range(seed_count_int)
    ]
    with Pool(14) as pool_obj:
        row_list = []
        for row_idx_int, row_dict in enumerate(pool_obj.imap_unordered(_task, task_list, chunksize=2), start=1):
            row_list.append(row_dict)
            if row_idx_int % 50 == 0:
                print(f"{row_idx_int}/{len(task_list)} ({time.time() - started_float:.0f}s)", flush=True)
    pd.DataFrame(row_list).to_parquet(OUTPUT_DIR_PATH / "synthetic.parquet")
    print("done", f"{time.time() - started_float:.0f}s", flush=True)


if __name__ == "__main__":
    main()
