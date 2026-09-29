"""Invariance checks V1 / V2 / V3 (PREREG section 9; research only).

V1  download-date invariance: for T0 in {2005-06-30, 2012-03-30, 2019-09-30} U1 is rebuilt as a download made on T0
    would show it (rows after T0 dropped; prices x k_T0, Volume x kv_T0 per stock, with k_T0 the stock's last price
    factor at or before T0; Turnover and Unadjusted Close unchanged); all 100 alphas over the 300 sessions up to T0
    must equal the full-panel values (relative 1e-9, same NaN pattern). Every mismatch is listed.
V2  random per-stock rescaling: OHLC x c_i, Volume / c_i, c_i log-uniform in [0.1, 10] (seed 20260928); Unadjusted
    Close unchanged; all alphas unchanged (relative 1e-9) over the full history.
V3  walk-forward look-ahead: the C_WF weights of each year recomputed from data truncated at the end of the prior year
    (alpha panels and opens cut there, the IC recomputed with an independent per-day Spearman loop) must equal the
    stored full-run weights.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from scipy import stats

from alpha101_20260928 import alphas as alphas_module
from alpha101_20260928 import checks
from alpha101_20260928 import common
from alpha101_20260928 import composites
from alpha101_20260928 import data as data_module
from alpha101_20260928 import formulas
from alpha101_20260928.evaluator import evaluate_formula

V1_DATE_TUPLE = ("2005-06-30", "2012-03-30", "2019-09-30")
V1_SESSIONS_INT = 300
REL_TOL_FLOAT = 1e-9
MAX_LISTED_INT = 200
PRICE_KEY_TUPLE = ("open_arr", "high_arr", "low_arr", "close_arr")


def compare_panels(full_arr: np.ndarray, other_arr: np.ndarray, rel_tol_float: float = REL_TOL_FLOAT) -> dict:
    """Mismatch mask: different NaN pattern, or |a - b| > rel x max(|a|, |b|) + 1e-12 x max|window| (float floor)."""
    nan_full_arr = np.isnan(full_arr)
    nan_other_arr = np.isnan(other_arr)
    pattern_arr = nan_full_arr != nan_other_arr
    both_arr = ~nan_full_arr & ~nan_other_arr
    scale_float = float(np.nanmax(np.abs(full_arr))) if both_arr.any() else 0.0
    with np.errstate(all="ignore"):
        diff_arr = np.abs(full_arr - other_arr)
        tol_arr = rel_tol_float * np.maximum(np.abs(full_arr), np.abs(other_arr)) + 1e-12 * scale_float
        value_arr = both_arr & (diff_arr > tol_arr)
    return {"pattern": pattern_arr, "value": value_arr, "compared_int": int(both_arr.sum())}


def truncated_universe(universe_dict: dict, t0_ts: pd.Timestamp) -> tuple[dict, int]:
    """The U1 panels as a download on t0 would show them (PREREG V1)."""
    date_index = data_module.date_index_of(universe_dict)
    pos_int = int(date_index.searchsorted(t0_ts, side="right")) - 1
    k_arr = data_module.price_factor(universe_dict)[: pos_int + 1]
    # last known price factor at or before T0 per stock (a delisted stock keeps its final factor)
    k0_vec = np.full(k_arr.shape[1], np.nan)
    for symbol_idx in range(k_arr.shape[1]):
        finite_vec = np.flatnonzero(np.isfinite(k_arr[:, symbol_idx]))
        if len(finite_vec):
            k0_vec[symbol_idx] = k_arr[finite_vec[-1], symbol_idx]
    out_dict = dict(universe_dict)
    out_dict["date_index"] = date_index[: pos_int + 1]
    for key_str in PRICE_KEY_TUPLE:
        out_dict[key_str] = universe_dict[key_str][: pos_int + 1].astype(np.float64) * k0_vec[None, :]
    out_dict["volume_arr"] = universe_dict["volume_arr"][: pos_int + 1].astype(np.float64) / k0_vec[None, :]
    for key_str in ("turnover_arr", "unadjusted_close_arr", "dividend_arr", "member_arr"):
        out_dict[key_str] = universe_dict[key_str][: pos_int + 1]
    return out_dict, pos_int


def rescaled_universe(universe_dict: dict, seed_int: int = common.SEED_INT) -> tuple[dict, np.ndarray]:
    rng = np.random.default_rng(seed_int)
    scale_vec = np.exp(rng.uniform(np.log(0.1), np.log(10.0), len(universe_dict["symbol_list"])))
    out_dict = dict(universe_dict)
    for key_str in PRICE_KEY_TUPLE:
        out_dict[key_str] = universe_dict[key_str].astype(np.float64) * scale_vec[None, :]
    out_dict["volume_arr"] = universe_dict["volume_arr"].astype(np.float64) / scale_vec[None, :]
    return out_dict, scale_vec


def _mismatch_rows(alpha_int: int, date_index, symbol_list, full_arr, other_arr, mask_arr, kind_str: str, limit_int: int) -> list[dict]:
    row_list = []
    for t_int, s_int in zip(*np.nonzero(mask_arr)):
        if len(row_list) >= limit_int:
            break
        row_list.append({"alpha": alpha_int, "kind": kind_str, "date": str(date_index[t_int].date()), "symbol": symbol_list[s_int],
                         "full": None if np.isnan(full_arr[t_int, s_int]) else float(full_arr[t_int, s_int]),
                         "rebuilt": None if np.isnan(other_arr[t_int, s_int]) else float(other_arr[t_int, s_int])})
    return row_list


def run_v1() -> dict:
    universe_dict = data_module.load_universe("U1")
    symbol_list = list(universe_dict["symbol_list"])
    code_dict = data_module.load_gics_codes("U1", symbol_list)
    raw_store = alphas_module.load_raw("U1")
    report_dict: dict = {"dates": {}, "rel_tol": REL_TOL_FLOAT, "sessions_compared_int": V1_SESSIONS_INT}
    total_value_mismatch_int = 0
    total_pattern_mismatch_int = 0
    for date_str in V1_DATE_TUPLE:
        start_float = time.perf_counter()
        trunc_dict, pos_int = truncated_universe(universe_dict, pd.Timestamp(date_str))
        context, _ = data_module.build_context(trunc_dict, code_dict)
        window_slice = slice(pos_int - V1_SESSIONS_INT + 1, pos_int + 1)
        date_window = data_module.date_index_of(trunc_dict)[window_slice]
        per_alpha_dict = {}
        listed_list = []
        for alpha_int in formulas.ALPHA_NUMBER_TUPLE:
            rebuilt_arr = evaluate_formula(formulas.FORMULA_DICT[alpha_int], context)[window_slice]
            full_arr = np.asarray(raw_store[alphas_module.ALPHA_INDEX_DICT[alpha_int], window_slice])
            cmp_dict = compare_panels(full_arr, rebuilt_arr)
            value_int = int(cmp_dict["value"].sum())
            pattern_int = int(cmp_dict["pattern"].sum())
            member_window_arr = trunc_dict["member_arr"][window_slice] == 1
            per_alpha_dict[str(alpha_int)] = {"compared_int": cmp_dict["compared_int"], "value_mismatch_int": value_int, "nan_pattern_mismatch_int": pattern_int,
                                              "nan_pattern_mismatch_members_int": int((cmp_dict["pattern"] & member_window_arr).sum()),
                                              "max_rel_diff": float(np.nanmax(np.abs(full_arr - rebuilt_arr) / np.maximum(np.maximum(np.abs(full_arr), np.abs(rebuilt_arr)), 1e-300))) if cmp_dict["compared_int"] else 0.0}
            total_value_mismatch_int += value_int
            total_pattern_mismatch_int += pattern_int
            if value_int or pattern_int:
                listed_list += _mismatch_rows(alpha_int, date_window, symbol_list, full_arr, rebuilt_arr, cmp_dict["value"], "value", MAX_LISTED_INT)
                listed_list += _mismatch_rows(alpha_int, date_window, symbol_list, full_arr, rebuilt_arr, cmp_dict["pattern"], "nan_pattern", MAX_LISTED_INT)
        report_dict["dates"][date_str] = {
            "truncation_pos": pos_int, "window": [str(date_window[0].date()), str(date_window[-1].date())], "seconds": round(time.perf_counter() - start_float, 1),
            "alphas_with_any_mismatch": [a for a, v in per_alpha_dict.items() if v["value_mismatch_int"] or v["nan_pattern_mismatch_int"]],
            "value_mismatch_cells_int": int(sum(v["value_mismatch_int"] for v in per_alpha_dict.values())),
            "nan_pattern_mismatch_cells_int": int(sum(v["nan_pattern_mismatch_int"] for v in per_alpha_dict.values())),
            "max_rel_diff_over_alphas": float(max(v["max_rel_diff"] for v in per_alpha_dict.values())),
            "per_alpha": per_alpha_dict, "listed_mismatches": listed_list[: 3 * MAX_LISTED_INT],
        }
        common.log_progress(f"V1 {date_str}: value mismatches {report_dict['dates'][date_str]['value_mismatch_cells_int']}, NaN-pattern mismatches {report_dict['dates'][date_str]['nan_pattern_mismatch_cells_int']}, "
                            f"max rel diff {report_dict['dates'][date_str]['max_rel_diff_over_alphas']:.2e}, {report_dict['dates'][date_str]['seconds']}s")
        del context, trunc_dict
    report_dict["value_mismatch_cells_total_int"] = total_value_mismatch_int
    report_dict["nan_pattern_mismatch_cells_total_int"] = total_pattern_mismatch_int
    report_dict["passed_strict_bool"] = total_value_mismatch_int == 0 and total_pattern_mismatch_int == 0
    checks.update_checks("v1_download_date_invariance", report_dict)
    return report_dict


def run_v2() -> dict:
    universe_dict = data_module.load_universe("U1")
    symbol_list = list(universe_dict["symbol_list"])
    code_dict = data_module.load_gics_codes("U1", symbol_list)
    scaled_dict, scale_vec = rescaled_universe(universe_dict)
    context, _ = data_module.build_context(scaled_dict, code_dict)
    raw_store = alphas_module.load_raw("U1")
    date_index = data_module.date_index_of(universe_dict)
    per_alpha_dict = {}
    listed_list = []
    start_float = time.perf_counter()
    for alpha_int in formulas.ALPHA_NUMBER_TUPLE:
        rebuilt_arr = evaluate_formula(formulas.FORMULA_DICT[alpha_int], context)
        full_arr = np.asarray(raw_store[alphas_module.ALPHA_INDEX_DICT[alpha_int]])
        cmp_dict = compare_panels(full_arr, rebuilt_arr)
        value_int, pattern_int = int(cmp_dict["value"].sum()), int(cmp_dict["pattern"].sum())
        per_alpha_dict[str(alpha_int)] = {"compared_int": cmp_dict["compared_int"], "value_mismatch_int": value_int, "nan_pattern_mismatch_int": pattern_int,
                                          "max_rel_diff": float(np.nanmax(np.abs(full_arr - rebuilt_arr) / np.maximum(np.maximum(np.abs(full_arr), np.abs(rebuilt_arr)), 1e-300))) if cmp_dict["compared_int"] else 0.0}
        if value_int or pattern_int:
            listed_list += _mismatch_rows(alpha_int, date_index, symbol_list, full_arr, rebuilt_arr, cmp_dict["value"], "value", MAX_LISTED_INT)
            listed_list += _mismatch_rows(alpha_int, date_index, symbol_list, full_arr, rebuilt_arr, cmp_dict["pattern"], "nan_pattern", MAX_LISTED_INT)
        del rebuilt_arr, full_arr
    report_dict = {
        "scale_range": [float(scale_vec.min()), float(scale_vec.max())], "seconds": round(time.perf_counter() - start_float, 1), "rel_tol": REL_TOL_FLOAT,
        "alphas_with_any_mismatch": [a for a, v in per_alpha_dict.items() if v["value_mismatch_int"] or v["nan_pattern_mismatch_int"]],
        "value_mismatch_cells_int": int(sum(v["value_mismatch_int"] for v in per_alpha_dict.values())),
        "nan_pattern_mismatch_cells_int": int(sum(v["nan_pattern_mismatch_int"] for v in per_alpha_dict.values())),
        "max_rel_diff_over_alphas": float(max(v["max_rel_diff"] for v in per_alpha_dict.values())),
        "per_alpha": per_alpha_dict, "listed_mismatches": listed_list[: 3 * MAX_LISTED_INT],
    }
    report_dict["passed_strict_bool"] = report_dict["value_mismatch_cells_int"] == 0 and report_dict["nan_pattern_mismatch_cells_int"] == 0
    checks.update_checks("v2_random_rescaling", report_dict)
    common.log_progress(f"V2: value mismatches {report_dict['value_mismatch_cells_int']}, NaN-pattern mismatches {report_dict['nan_pattern_mismatch_cells_int']}, max rel diff {report_dict['max_rel_diff_over_alphas']:.2e}, {report_dict['seconds']}s")
    return report_dict


def independent_daily_ic(z_arr: np.ndarray, fr_arr: np.ndarray, member_arr: np.ndarray) -> np.ndarray:
    """Per-day scipy Spearman on the eligible names (an independent implementation of stage_a.daily_ic)."""
    out_vec = np.full(z_arr.shape[0], np.nan)
    for t_int in range(z_arr.shape[0]):
        eligible_vec = member_arr[t_int] & np.isfinite(z_arr[t_int]) & np.isfinite(fr_arr[t_int])
        if eligible_vec.sum() >= 3:
            out_vec[t_int] = stats.spearmanr(z_arr[t_int][eligible_vec], fr_arr[t_int][eligible_vec]).statistic
    return out_vec


def run_v3(universe_str: str = "U1") -> dict:
    universe_dict = data_module.load_universe(universe_str)
    date_index = data_module.date_index_of(universe_dict)
    open_arr = universe_dict["open_arr"].astype(np.float64)
    member_arr = universe_dict["member_arr"] == 1
    z_store = alphas_module.load_z(universe_str)
    stored_dict = composites.load_wf_weights(universe_str)
    from alpha101_20260928 import stage_a

    start_pos_int, _ = stage_a.decision_range(date_index)
    year_list = sorted(int(y) for y in stored_dict if int(y) >= common.WF_FIRST_YEAR_INT and int(y) <= common.END_TS.year)
    report_dict = {"years": {}, "max_abs_weight_diff": 0.0, "mode_mismatch_years": [], "ic_implementation_max_abs_diff": 0.0}
    stored_ic_df = composites.load_alpha_ic(universe_str)
    start_float = time.perf_counter()
    for year_int in year_list:
        # data truncated at the last session of the prior year: opens beyond it are unknown
        last_pos_int = int(date_index.searchsorted(pd.Timestamp(f"{year_int - 1}-12-31"), side="right")) - 1
        open_trunc_arr = open_arr[: last_pos_int + 1]
        fr_trunc_arr = data_module.forward_return_panel(open_trunc_arr)  # NaN on the last two sessions
        rows_slice = slice(start_pos_int, last_pos_int + 1)
        m_dict = {}
        for alpha_int in formulas.ALPHA_NUMBER_TUPLE:
            z_arr = np.asarray(z_store[alphas_module.ALPHA_INDEX_DICT[alpha_int], rows_slice], dtype=np.float64)
            ic_vec = independent_daily_ic(z_arr, fr_trunc_arr[rows_slice], member_arr[rows_slice])
            m_dict[str(alpha_int)] = float(np.nanmean(ic_vec)) if np.isfinite(ic_vec).any() else float("nan")
            if year_int == year_list[-1]:
                stored_vec = stored_ic_df[str(alpha_int)].reindex(date_index[rows_slice]).to_numpy()
                both_vec = np.isfinite(stored_vec) & np.isfinite(ic_vec)
                report_dict["ic_implementation_max_abs_diff"] = max(report_dict["ic_implementation_max_abs_diff"], float(np.max(np.abs(stored_vec[both_vec] - ic_vec[both_vec]))) if both_vec.any() else 0.0)
        m_ser = pd.Series(m_dict)
        positive_ser = m_ser.clip(lower=0.0).fillna(0.0)
        mode_str = "WF" if positive_ser.sum() > 0 else "C_EQ"
        weight_ser = positive_ser / positive_ser.sum() if mode_str == "WF" else pd.Series(1.0 / len(m_ser), index=m_ser.index)
        stored_entry = stored_dict[str(year_int)]
        stored_weight_ser = pd.Series(stored_entry["weights"]).reindex(weight_ser.index)
        diff_float = float(np.max(np.abs(weight_ser.to_numpy() - stored_weight_ser.to_numpy())))
        report_dict["years"][str(year_int)] = {"mode_recomputed": mode_str, "mode_stored": stored_entry["mode"], "max_abs_weight_diff": diff_float, "truncated_at": str(date_index[last_pos_int].date()),
                                               "sessions_int": int(np.isfinite(fr_trunc_arr[rows_slice]).any(axis=1).sum())}
        report_dict["max_abs_weight_diff"] = max(report_dict["max_abs_weight_diff"], diff_float)
        if mode_str != stored_entry["mode"]:
            report_dict["mode_mismatch_years"].append(year_int)
    report_dict["seconds"] = round(time.perf_counter() - start_float, 1)
    report_dict["passed_bool"] = bool(report_dict["max_abs_weight_diff"] <= 1e-10 and not report_dict["mode_mismatch_years"])
    checks.update_checks("v3_walk_forward_weights", report_dict)
    common.log_progress(f"V3: max |weight diff| {report_dict['max_abs_weight_diff']:.2e}, mode mismatches {report_dict['mode_mismatch_years']}, IC impl max diff {report_dict['ic_implementation_max_abs_diff']:.2e}, passed {report_dict['passed_bool']}")
    return report_dict
