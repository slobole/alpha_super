"""Evaluate the 100 alphas on a universe and store their percentile-rank scores on disk (research only).

Per alpha i and session t: a_{i,t} = the converted alpha panel (evaluator.evaluate_formula), z_{i,t} = percentile rank
of a among members with finite values minus 0.5 (PREREG section 6). z is stored as float32 in
_cache/z_<U>.npy (shape 100 x T x S, .npy memmap); for U1 the converted alpha values are also stored as float64
(_cache/alpha_raw_U1.npy) for the invariance checks. A meta JSON records per-alpha seconds, degree, finite share and
serves as the checkpoint (finished alphas are skipped on restart).
"""

from __future__ import annotations

import json
import time

import numpy as np

from alpha101_20260928 import common
from alpha101_20260928 import data as data_module
from alpha101_20260928 import formulas
from alpha101_20260928 import operators as ops
from alpha101_20260928.evaluator import Context, evaluate_formula_with_degree

ALPHA_INDEX_DICT = {n: i for i, n in enumerate(formulas.ALPHA_NUMBER_TUPLE)}


def z_path(universe_str: str):
    return common.CACHE_DIR_PATH / f"z_{universe_str}.npy"


def raw_path(universe_str: str):
    return common.CACHE_DIR_PATH / f"alpha_raw_{universe_str}.npy"


def meta_path(universe_str: str):
    return common.CACHE_DIR_PATH / f"alphas_meta_{universe_str}.json"


def signal_z(alpha_arr: np.ndarray, member_arr: np.ndarray) -> np.ndarray:
    """Percentile rank among members with finite values minus 0.5 (NaN elsewhere)."""
    return ops.cs_rank(alpha_arr, member_arr) - 0.5


def open_store(path_obj, shape_tuple: tuple, dtype_obj, create_bool: bool):
    if create_bool and not path_obj.exists():
        common.CACHE_DIR_PATH.mkdir(parents=True, exist_ok=True)
        store_arr = np.lib.format.open_memmap(path_obj, mode="w+", dtype=dtype_obj, shape=shape_tuple)
        store_arr[:] = np.nan
        store_arr.flush()
        del store_arr
    return np.load(path_obj, mmap_mode="r+" if create_bool else "r")


def evaluate_alpha(alpha_int: int, context: Context) -> tuple[np.ndarray, tuple]:
    return evaluate_formula_with_degree(formulas.FORMULA_DICT[alpha_int], context)


def compute_all(universe_str: str, store_raw_bool: bool | None = None, alpha_list: list[int] | None = None) -> dict:
    """Evaluate all alphas (or a subset) and write z (and raw for U1) to the on-disk stores; resumable."""
    store_raw_bool = universe_str == "U1" if store_raw_bool is None else store_raw_bool
    universe_dict = data_module.load_universe(universe_str)
    context, context_meta_dict = data_module.build_context(universe_dict)
    t_int, s_int = context.shape
    shape_tuple = (len(formulas.ALPHA_NUMBER_TUPLE), t_int, s_int)
    z_store = open_store(z_path(universe_str), shape_tuple, np.float32, True)
    raw_store = open_store(raw_path(universe_str), shape_tuple, np.float64, True) if store_raw_bool else None
    meta_dict = json.loads(meta_path(universe_str).read_text(encoding="utf-8")) if meta_path(universe_str).exists() else {}
    meta_dict.setdefault("universe", universe_str)
    meta_dict["symbols_int"], meta_dict["sessions_int"] = s_int, t_int
    meta_dict["context"] = context_meta_dict
    meta_dict.setdefault("alphas", {})
    member_arr = context.member_arr
    member_count_int = int(member_arr.sum())
    common.log_progress(f"alphas {universe_str}: {s_int} symbols x {t_int} sessions; vwap {context_meta_dict['vwap']}; gics {context_meta_dict['gics']}", f"log_alphas_{universe_str}.txt")
    for alpha_int in alpha_list or formulas.ALPHA_NUMBER_TUPLE:
        key_str = str(alpha_int)
        if key_str in meta_dict["alphas"] and meta_dict["alphas"][key_str].get("done_bool") and alpha_list is None:
            continue
        start_float = time.perf_counter()
        conversions_before_int = context.conversion_count_int
        alpha_arr, degree_tuple = evaluate_alpha(alpha_int, context)
        z_arr = signal_z(alpha_arr, member_arr)
        idx_int = ALPHA_INDEX_DICT[alpha_int]
        z_store[idx_int] = z_arr.astype(np.float32)
        if raw_store is not None:
            raw_store[idx_int] = alpha_arr
        finite_member_int = int((np.isfinite(alpha_arr) & member_arr).sum())
        seconds_float = time.perf_counter() - start_float
        meta_dict["alphas"][key_str] = {
            "done_bool": True,
            "seconds": round(seconds_float, 2),
            "root_degree": list(degree_tuple),
            "conversions_int": context.conversion_count_int - conversions_before_int,
            "finite_share_of_member_days": finite_member_int / member_count_int,
            "first_finite_session": str(data_module.date_index_of(universe_dict)[int(np.argmax((np.isfinite(alpha_arr) & member_arr).any(axis=1)))].date()) if finite_member_int else None,
        }
        z_store.flush()
        if raw_store is not None:
            raw_store.flush()
        meta_path(universe_str).write_text(json.dumps(meta_dict, indent=1), encoding="utf-8")
        common.log_progress(f"alpha {alpha_int:3d} {universe_str}: {seconds_float:6.1f}s, degree {degree_tuple}, finite share {finite_member_int / member_count_int:.3f}", f"log_alphas_{universe_str}.txt")
        del alpha_arr, z_arr
    meta_dict["all_done_bool"] = all(meta_dict["alphas"].get(str(n), {}).get("done_bool") for n in formulas.ALPHA_NUMBER_TUPLE)
    meta_path(universe_str).write_text(json.dumps(meta_dict, indent=1), encoding="utf-8")
    return meta_dict


def load_z(universe_str: str) -> np.ndarray:
    return np.load(z_path(universe_str), mmap_mode="r")


def load_raw(universe_str: str) -> np.ndarray:
    return np.load(raw_path(universe_str), mmap_mode="r")


def load_meta(universe_str: str) -> dict:
    return json.loads(meta_path(universe_str).read_text(encoding="utf-8"))


def control_signals(context: Context) -> dict[str, np.ndarray]:
    """REV1 = -returns_t; REV5 = -(Close_t / Close_{t-5} - 1) (PREREG section 5); NaN outside members."""
    close_arr = context.panel_dict["close"]
    with np.errstate(all="ignore"):
        rev5_arr = np.full_like(close_arr, np.nan)
        rev5_arr[5:] = -(close_arr[5:] / close_arr[:-5] - 1.0)
    rev1_arr = -context.panel_dict["returns"]
    out_dict = {}
    for name_str, arr in (("REV1", rev1_arr), ("REV5", rev5_arr)):
        arr = ops.clean(arr).copy()
        arr[~context.member_arr] = np.nan
        out_dict[name_str] = arr
    return out_dict
