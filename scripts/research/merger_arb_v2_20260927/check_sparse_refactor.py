"""Compare the grid outputs of the sparse-confirmation code with the set-aside outputs of the first (dense) run.
Research tooling only: the two runs must agree bit for bit (same rules, memory-only refactor). Writes
results/research/merger_arb_v2_20260927/_cache/dense_run_check/comparison.json and prints it."""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from merger_arb_v2_20260927 import common  # noqa: E402

OUT_DIR = Path(common.RESULTS_DIR_PATH)
DENSE_DIR = OUT_DIR / "_cache" / "dense_run_check"


def frame_equal(a_df: pd.DataFrame, b_df: pd.DataFrame) -> dict:
    same_shape = a_df.shape == b_df.shape and list(a_df.columns) == list(b_df.columns) and a_df.index.equals(b_df.index)
    if not same_shape:
        return {"identical_bool": False, "reason": f"shape/columns/index differ: {a_df.shape} vs {b_df.shape}"}
    a_arr, b_arr = a_df.to_numpy(dtype=np.float64), b_df.to_numpy(dtype=np.float64)
    both_nan = np.isnan(a_arr) & np.isnan(b_arr)
    diff_arr = np.where(both_nan, 0.0, np.abs(a_arr - b_arr))
    nan_mismatch_int = int((np.isnan(a_arr) != np.isnan(b_arr)).sum())
    return {"identical_bool": bool(nan_mismatch_int == 0 and np.nanmax(diff_arr) == 0.0), "max_abs_diff_float": float(np.nanmax(diff_arr)), "nan_mismatch_int": nan_mismatch_int}


def main() -> None:
    report = {}
    for name_str in ("returns_V_ALL_engine.parquet", "returns_V_ALL_stress.parquet", "cashw_V_ALL_engine.parquet", "cashw_V_ALL_stress.parquet", "exposure_V_ALL.parquet"):
        report[name_str] = frame_equal(pd.read_parquet(OUT_DIR / name_str), pd.read_parquet(DENSE_DIR / name_str))
    meta_new = json.loads((OUT_DIR / "meta_V_ALL.json").read_text(encoding="utf-8"))
    meta_old = json.loads((DENSE_DIR / "meta_V_ALL.json").read_text(encoding="utf-8"))
    report["meta_V_ALL.json"] = {"identical_bool": meta_new == meta_old, "keys_int": len(meta_new)}
    if meta_new != meta_old:
        diff_keys = [k for k in meta_new if meta_new.get(k) != meta_old.get(k)]
        report["meta_V_ALL.json"]["differing_cells"] = diff_keys[:20]
    with open(OUT_DIR / "trades_V_ALL.pkl", "rb") as fh:
        trades_new = pickle.load(fh)
    with open(DENSE_DIR / "trades_V_ALL.pkl", "rb") as fh:
        trades_old = pickle.load(fh)
    trade_report = {"identical_bool": True, "cells_int": len(trades_new)}
    for key_str, obj_new in trades_new.items():
        obj_old = trades_old.get(key_str)
        same_bool = False
        if isinstance(obj_new, pd.DataFrame) and isinstance(obj_old, pd.DataFrame):
            try:
                pd.testing.assert_frame_equal(obj_new.reset_index(drop=True), obj_old.reset_index(drop=True), check_exact=True)
                same_bool = True
            except AssertionError:
                same_bool = False
        elif isinstance(obj_new, dict) and isinstance(obj_old, dict):
            same_bool = all(str(obj_new.get(k)) == str(obj_old.get(k)) for k in set(obj_new) | set(obj_old))
        else:
            same_bool = str(obj_new) == str(obj_old)
        if not same_bool:
            trade_report["identical_bool"] = False
            trade_report.setdefault("differing_cells", []).append(key_str)
    report["trades_V_ALL.pkl"] = trade_report
    report["all_identical_bool"] = all(v["identical_bool"] for v in report.values() if isinstance(v, dict))
    (DENSE_DIR / "comparison.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
