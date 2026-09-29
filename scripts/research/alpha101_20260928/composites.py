"""Controls, daily IC of the alphas, walk-forward weights and the composite panels (PREREG sections 5-6; research only).

    z_{i,t}     percentile rank of alpha i among members with finite values, minus 0.5 (stored by alphas.py)
    C_EQ        mean of z over the alphas available for the stock-day; at least 50 of the 100 must be available
    C_WF        sum_i w_{i,Y} z_i / sum_i w_{i,Y} over available alphas; at least half of the total weight available;
                w_{i,Y} = max(m_{i,Y}, 0) / sum_j max(m_{j,Y}, 0) for calendar year Y >= 2003, m_{i,Y} = mean daily IC
                of alpha i over decisions from 2000-01-03 whose fr is known by the last session of Y-1 (fr_t needs
                Open_{t+2}); years 2000-2002 and years where every m <= 0 use C_EQ
    C_EQ_noInd  C_EQ without the 18 IndNeutralize alphas (sensitivity; at least half of the 82 available)
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from alpha101_20260928 import alphas as alphas_module
from alpha101_20260928 import common
from alpha101_20260928 import data as data_module
from alpha101_20260928 import formulas
from alpha101_20260928 import stage_a

CHUNK_ROWS_INT = 400


def controls_path(universe_str: str):
    return common.CACHE_DIR_PATH / f"controls_{universe_str}.npz"


def ic_path(universe_str: str):
    return common.CACHE_DIR_PATH / f"ic_alphas_{universe_str}.parquet"


def composite_path(universe_str: str, name_str: str):
    return common.CACHE_DIR_PATH / f"composite_{universe_str}_{name_str}.npy"


def composite_meta_path(universe_str: str):
    return common.CACHE_DIR_PATH / f"composite_meta_{universe_str}.json"


# ----------------------------------------------------------------------------------------------------------------------
def build_controls(universe_str: str) -> dict:
    universe_dict = data_module.load_universe(universe_str)
    context, _ = data_module.build_context(universe_dict)
    raw_dict = alphas_module.control_signals(context)
    out_dict = {}
    for name_str, arr in raw_dict.items():
        out_dict[f"{name_str}_raw"] = arr
        out_dict[f"{name_str}_z"] = alphas_module.signal_z(arr, context.member_arr).astype(np.float32)
    common.CACHE_DIR_PATH.mkdir(parents=True, exist_ok=True)
    np.savez(controls_path(universe_str), **out_dict)
    common.log_progress(f"controls {universe_str}: REV1 / REV5 stored")
    return {k: v.shape for k, v in out_dict.items()}


def load_controls(universe_str: str) -> dict:
    with np.load(controls_path(universe_str)) as npz_obj:
        return {k: npz_obj[k] for k in npz_obj.files}


# ----------------------------------------------------------------------------------------------------------------------
def build_alpha_ic(universe_str: str) -> pd.DataFrame:
    """Daily Spearman IC (decision t vs fr_t) of every alpha, 2000-01-03..END; the input of the walk-forward weights."""
    universe_dict = data_module.load_universe(universe_str)
    date_index = data_module.date_index_of(universe_dict)
    fr_arr = data_module.forward_return_panel(universe_dict["open_arr"])
    member_arr = universe_dict["member_arr"] == 1
    z_store = alphas_module.load_z(universe_str)
    start_pos_int, end_pos_int = stage_a.decision_range(date_index)
    ic_dict = {}
    for alpha_int in formulas.ALPHA_NUMBER_TUPLE:
        z_arr = np.asarray(z_store[alphas_module.ALPHA_INDEX_DICT[alpha_int], start_pos_int:end_pos_int], dtype=np.float64)
        ic_dict[str(alpha_int)] = stage_a.daily_ic(z_arr, fr_arr[start_pos_int:end_pos_int], member_arr[start_pos_int:end_pos_int])
    ic_df = pd.DataFrame(ic_dict, index=date_index[start_pos_int:end_pos_int])
    ic_df.index.name = "decision_ts"
    ic_df.to_parquet(ic_path(universe_str))
    common.log_progress(f"IC {universe_str}: {ic_df.shape[0]} sessions x {ic_df.shape[1]} alphas; mean IC over alphas {ic_df.mean().mean():.5f}")
    return ic_df


def load_alpha_ic(universe_str: str) -> pd.DataFrame:
    ic_df = pd.read_parquet(ic_path(universe_str))
    ic_df.index = pd.to_datetime(ic_df.index)
    return ic_df


# ----------------------------------------------------------------------------------------------------------------------
def known_by_cutoff_ts(date_index: pd.DatetimeIndex, year_int: int) -> pd.Timestamp | None:
    """Latest decision date t whose fr_t (= Open_{t+2} / Open_{t+1} - 1) is known by the last session of year - 1."""
    last_session_pos_int = int(date_index.searchsorted(pd.Timestamp(f"{year_int - 1}-12-31"), side="right")) - 1
    decision_pos_int = last_session_pos_int - 2
    return date_index[decision_pos_int] if decision_pos_int >= 0 else None


def wf_weights_from_ic(ic_df: pd.DataFrame, date_index: pd.DatetimeIndex, year_list: list[int]) -> dict:
    """Per calendar year: the weights (or the C_EQ fallback), the IC means m and the cutoff (PREREG section 6)."""
    weights_dict: dict = {}
    alpha_list = list(ic_df.columns)
    for year_int in year_list:
        entry = {"year": year_int, "mode": "C_EQ", "cutoff_decision_ts": None, "sessions_int": 0, "m": None, "weights": None, "positive_m_int": 0}
        if year_int >= common.WF_FIRST_YEAR_INT:
            cutoff_ts = known_by_cutoff_ts(date_index, year_int)
            window_df = ic_df.loc[:cutoff_ts] if cutoff_ts is not None else ic_df.iloc[0:0]
            m_ser = window_df.mean()
            entry["cutoff_decision_ts"] = str(cutoff_ts.date()) if cutoff_ts is not None else None
            entry["sessions_int"] = int(len(window_df))
            entry["m"] = {a: (float(v) if np.isfinite(v) else None) for a, v in m_ser.items()}
            positive_ser = m_ser.clip(lower=0.0).fillna(0.0)
            entry["positive_m_int"] = int((positive_ser > 0).sum())
            if positive_ser.sum() > 0:
                weight_ser = positive_ser / positive_ser.sum()
                entry["mode"] = "WF"
                entry["weights"] = {a: float(weight_ser[a]) for a in alpha_list}
        if entry["weights"] is None:
            entry["weights"] = {a: 1.0 / len(alpha_list) for a in alpha_list}
        weights_dict[str(year_int)] = entry
    return weights_dict


def weighted_composite(z_chunk_arr: np.ndarray, weight_vec: np.ndarray, min_weight_float: float) -> np.ndarray:
    """sum_i w_i z_i / sum_{available} w_i for one time chunk (A x rows x S); NaN below the weight floor."""
    finite_arr = np.isfinite(z_chunk_arr)
    z_filled_arr = np.where(finite_arr, z_chunk_arr, 0.0)
    numerator_arr = np.tensordot(weight_vec, z_filled_arr, axes=(0, 0))
    available_arr = np.tensordot(weight_vec, finite_arr.astype(np.float64), axes=(0, 0))
    with np.errstate(all="ignore"):
        out_arr = np.where(available_arr >= min_weight_float, numerator_arr / available_arr, np.nan)
    return out_arr


def build_composites(universe_str: str) -> dict:
    universe_dict = data_module.load_universe(universe_str)
    date_index = data_module.date_index_of(universe_dict)
    z_store = alphas_module.load_z(universe_str)
    alpha_count_int, t_int, s_int = z_store.shape
    ic_df = load_alpha_ic(universe_str)
    year_list = list(range(int(date_index[0].year), int(date_index[-1].year) + 1))
    weights_dict = wf_weights_from_ic(ic_df, date_index, year_list)
    alpha_list = [str(n) for n in formulas.ALPHA_NUMBER_TUPLE]
    eq_weight_vec = np.full(alpha_count_int, 1.0 / alpha_count_int)
    noind_mask_vec = np.array([n not in formulas.INDNEUTRALIZE_ALPHA_TUPLE for n in formulas.ALPHA_NUMBER_TUPLE], dtype=np.float64)
    noind_weight_vec = noind_mask_vec / noind_mask_vec.sum()
    year_of_row_vec = date_index.year.to_numpy()
    out_dict = {name_str: np.lib.format.open_memmap(composite_path(universe_str, name_str), mode="w+", dtype=np.float64, shape=(t_int, s_int)) for name_str in ("C_EQ", "C_WF", "C_EQ_noInd")}
    member_arr = universe_dict["member_arr"] == 1
    available_count_list = []
    for row_start_int in range(0, t_int, CHUNK_ROWS_INT):
        row_end_int = min(t_int, row_start_int + CHUNK_ROWS_INT)
        z_chunk_arr = np.asarray(z_store[:, row_start_int:row_end_int, :], dtype=np.float64)
        available_count_list.append(np.isfinite(z_chunk_arr).sum(axis=0)[member_arr[row_start_int:row_end_int]])
        out_dict["C_EQ"][row_start_int:row_end_int] = weighted_composite(z_chunk_arr, eq_weight_vec, common.MIN_ALPHAS_C_EQ_INT / alpha_count_int)
        out_dict["C_EQ_noInd"][row_start_int:row_end_int] = weighted_composite(z_chunk_arr, noind_weight_vec, 0.5)
        wf_chunk_arr = np.full((row_end_int - row_start_int, s_int), np.nan)
        for year_int in np.unique(year_of_row_vec[row_start_int:row_end_int]):
            row_mask_vec = year_of_row_vec[row_start_int:row_end_int] == year_int
            weight_vec = np.array([weights_dict[str(int(year_int))]["weights"][a] for a in alpha_list])
            wf_chunk_arr[row_mask_vec] = weighted_composite(z_chunk_arr[:, row_mask_vec, :], weight_vec, 0.5)
        out_dict["C_WF"][row_start_int:row_end_int] = wf_chunk_arr
    for name_str, store_arr in out_dict.items():
        store_arr[~member_arr] = np.nan
        store_arr.flush()
        del store_arr
    available_vec = np.concatenate(available_count_list)
    meta_dict = {
        "universe": universe_str,
        "alphas_available_per_member_day": {"mean": float(available_vec.mean()), "p05": float(np.quantile(available_vec, 0.05)), "min": int(available_vec.min()),
                                            "share_below_50": float((available_vec < common.MIN_ALPHAS_C_EQ_INT).mean())},
        "wf_years_using_c_eq": [y for y, e in weights_dict.items() if e["mode"] == "C_EQ"],
        "wf_effective_alpha_count_by_year": {y: (1.0 / float(np.sum(np.square(list(e["weights"].values()))))) for y, e in weights_dict.items()},
        "wf_positive_m_by_year": {y: e["positive_m_int"] for y, e in weights_dict.items()},
    }
    composite_meta_path(universe_str).write_text(json.dumps(meta_dict, indent=1), encoding="utf-8")
    existing_dict = common.read_json("wf_weights.json", {}) or {}
    existing_dict[universe_str] = weights_dict
    common.write_json("wf_weights.json", existing_dict)
    common.log_progress(f"composites {universe_str}: available alphas per member-day mean {available_vec.mean():.1f}, C_EQ years {meta_dict['wf_years_using_c_eq']}")
    return meta_dict


def load_composite(universe_str: str, name_str: str) -> np.ndarray:
    return np.load(composite_path(universe_str, name_str), mmap_mode="r")


def load_wf_weights(universe_str: str) -> dict:
    return (common.read_json("wf_weights.json", {}) or {})[universe_str]
