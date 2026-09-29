"""Download-date invariance checks (PREREG section 7; research only).

V1  Every stock's OHLC and per-share Dividend are multiplied by a random constant (log-uniform 0.01..100, seed
    20260927); Turnover and Unadjusted Close are unchanged. Decisions (intents), positions and the pre-fee NAV must be
    identical; NAV is also compared with the engine's phantom re-size fills cancelled (trend study note N6).
V2  Decision-day restatement from Norgate NONE bars, X_t(d) = X_NONE(d) x R(t) / R(d), for all month-ends plus 200
    random sessions: family M event / pin features (jump return, pin_vol, hold ratio, pin_ref, the M0 confirmation
    flag; the Turnover parts are nominal and unchanged) on the Russell 1000, and the SE_1_10 seasonality score on the
    S&P 500 restricted to the look-back years the NONE panel covers (1999 onward).
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

import ndx_param_robustness_core as core  # noqa: E402

from new_pod_search_20260927 import cells as cells_module
from new_pod_search_20260927 import common
from new_pod_search_20260927 import data as data_module
from new_pod_search_20260927.features import HOLD_FLOOR_FLOAT, PodFeatureBook, seasonality_score_tables
from new_pod_search_20260927.policies import PolicyM, PolicyS
from new_pod_search_20260927.simulate import simulate
from trend_breakout_20260927.invariance import load_none_panel

RANDOM_SESSION_COUNT_INT = 200


def scaled_universe(universe_dict: dict, seed_int: int = common.SEED_INT) -> dict:
    rng = np.random.default_rng(seed_int)
    scale_vec = np.exp(rng.uniform(np.log(0.01), np.log(100.0), len(universe_dict["symbol_list"])))
    scaled_dict = dict(universe_dict)
    for field_str in ("open_arr", "high_arr", "low_arr", "close_arr", "dividend_arr"):
        scaled_dict[field_str] = (universe_dict[field_str].astype(np.float64) * scale_vec[None, :]).astype(universe_dict[field_str].dtype)
    scaled_dict["volume_arr"] = (universe_dict["volume_arr"].astype(np.float64) / scale_vec[None, :]).astype(universe_dict["volume_arr"].dtype)
    return scaled_dict


def _context(family_str: str, universe_dict: dict) -> dict:
    feature_obj = PodFeatureBook(universe_dict)
    context_dict = {"features": feature_obj, "sh_idx": universe_dict.get("sh_idx")}
    if family_str == "S":
        context_dict["scores"] = seasonality_score_tables(universe_dict, data_module.load_monthly_closes_1990(universe_dict["universe_str"]), list(cells_module.HORIZON_TUPLE))
    return context_dict


def _policy(context_dict: dict, cell):
    if isinstance(cell, cells_module.MCell):
        return PolicyM(context_dict["features"], cell)
    return PolicyS(context_dict["features"], cell, context_dict["scores"][cell.horizon_str], context_dict["sh_idx"])


def _compare_runs(base_sim: dict, scaled_sim: dict) -> dict:
    position_diff_int = sum(1 for (p0, h0, _), (p1, h1, _) in zip(base_sim["position_log"], scaled_sim["position_log"]) if p0 != p1 or set(h0) != set(h1))
    b_df, s_df = base_sim["intent_df"], scaled_sim["intent_df"]
    col_list = ["decision_pos", "symbol_idx", "kind", "reason"]
    intents_same_bool = len(b_df) == len(s_df) and (len(b_df) == 0 or bool((b_df[col_list].to_numpy() == s_df[col_list].to_numpy()).all()))
    pre_fee_b = base_sim["total_ser"].to_numpy() + base_sim["commission_ser"].cumsum().to_numpy()
    pre_fee_s = scaled_sim["total_ser"].to_numpy() + scaled_sim["commission_ser"].cumsum().to_numpy()
    return {"position_mismatch_sessions_int": int(position_diff_int), "intents_identical_bool": intents_same_bool,
            "max_rel_nav_diff_float": float(np.max(np.abs(scaled_sim["total_ser"].to_numpy() / base_sim["total_ser"].to_numpy() - 1.0))),
            "max_rel_nav_diff_before_commissions_float": float(np.max(np.abs(pre_fee_s / pre_fee_b - 1.0))),
            "phantom_fills_base_int": int(base_sim["phantom_fill_int"]), "phantom_fills_scaled_int": int(scaled_sim["phantom_fill_int"])}


def run_v1() -> dict:
    report_dict: dict = {}
    for family_str, universe_str, cell_list in (("M", "R1000", cells_module.family_m_cells(include_sensitivities_bool=False)), ("S", "SP500", cells_module.family_s_cells())):
        universe_dict = data_module.get_universe(universe_str, sh_bool=family_str == "S")
        base_ctx = _context(family_str, universe_dict)
        scaled_ctx = _context(family_str, scaled_universe(universe_dict))
        cell_report_dict: dict = {}
        for cell in cell_list:
            base_sim = simulate(base_ctx["features"], _policy(base_ctx, cell), record_positions_bool=True)
            scaled_sim = simulate(scaled_ctx["features"], _policy(scaled_ctx, cell), record_positions_bool=True)
            cell_report = _compare_runs(base_sim, scaled_sim)
            base_pf = simulate(base_ctx["features"], _policy(base_ctx, cell), phantom_cancel_bool=True)
            scaled_pf = simulate(scaled_ctx["features"], _policy(scaled_ctx, cell), phantom_cancel_bool=True)
            cell_report["max_rel_nav_diff_phantom_free_float"] = float(np.max(np.abs(scaled_pf["total_ser"].to_numpy() / base_pf["total_ser"].to_numpy() - 1.0)))
            cell_report_dict[cell.key_str] = cell_report
            print(f"V1 {universe_str} {cell.key_str}: pos mismatches {cell_report['position_mismatch_sessions_int']}, intents same {cell_report['intents_identical_bool']}, "
                  f"nav {cell_report['max_rel_nav_diff_float']:.2e}, pre-fee {cell_report['max_rel_nav_diff_before_commissions_float']:.2e}, phantom-free {cell_report['max_rel_nav_diff_phantom_free_float']:.2e}", flush=True)
        report_dict[f"{family_str}_{universe_str}"] = {
            "cells": cell_report_dict,
            "cells_with_position_or_decision_difference": [k for k, v in cell_report_dict.items() if v["position_mismatch_sessions_int"] or not v["intents_identical_bool"]],
            "max_rel_nav_diff_phantom_free_over_cells_float": float(max(v["max_rel_nav_diff_phantom_free_float"] for v in cell_report_dict.values())),
            "max_rel_nav_diff_engine_rule_over_cells_float": float(max(v["max_rel_nav_diff_float"] for v in cell_report_dict.values())),
        }
        common.log_progress(f"V1 {family_str} {universe_str}: differing cells {report_dict[f'{family_str}_{universe_str}']['cells_with_position_or_decision_difference']}, "
                            f"phantom-free NAV rel diff {report_dict[f'{family_str}_{universe_str}']['max_rel_nav_diff_phantom_free_over_cells_float']:.2e}")
    common.write_json("invariance_v1.json", report_dict)
    return report_dict


def _rel(a_vec: np.ndarray, b_vec: np.ndarray) -> float:
    both_vec = np.isfinite(a_vec) & np.isfinite(b_vec)
    if not both_vec.any():
        return 0.0
    scale_vec = np.maximum(np.maximum(np.abs(a_vec[both_vec]), np.abs(b_vec[both_vec])), 1e-9)
    return float(np.max(np.abs(a_vec[both_vec] - b_vec[both_vec]) / scale_vec))


def run_v2() -> dict:
    rng = np.random.default_rng(common.SEED_INT)
    report_dict: dict = {}
    # ---------------------------------------------------------------- family M on the Russell 1000
    universe_dict = data_module.get_universe("R1000")
    feature_obj = PodFeatureBook(universe_dict)
    none_dict = load_none_panel(universe_dict)
    cs_close_arr = feature_obj.close_arr
    with np.errstate(invalid="ignore", divide="ignore"):
        factor_arr = none_dict["Close"] / cs_close_arr
    factor_arr = np.where(np.isfinite(factor_arr) & (factor_arr > 0), factor_arr, np.nan)
    schedule_dict = core.build_schedule(universe_dict, 0)
    decision_pos_vec = schedule_dict["decision_pos_vec"][schedule_dict["trade_vec"]]
    other_pos_vec = np.array(sorted(set(range(300, len(feature_obj.date_index) - 1)) - set(decision_pos_vec.tolist())))
    sampled_pos_vec = np.sort(np.concatenate([decision_pos_vec, rng.choice(other_pos_vec, RANDOM_SESSION_COUNT_INT, replace=False)]))
    cell = cells_module.M0_CELL
    conf_dict = feature_obj.confirmation(cell.jump_float, cell.theta_float, cell.window_int)
    pin_dict = feature_obj.pin_features(cell.window_int)
    member_arr = universe_dict["member_arr"]
    max_rel = {"jump_ret": 0.0, "pin_vol": 0.0, "pin_ref_day_t_units": 0.0, "hold_ratio": 0.0}
    mismatch = {"jump_flag": 0, "conf_flag": 0, "pin_ok": 0}
    checked_int = 0
    w_int = cell.window_int
    for t_int in sampled_pos_vec:
        t_int = int(t_int)
        d_int = t_int - w_int
        tradable_vec = (member_arr[d_int] == 1) & np.isfinite(cs_close_arr[t_int]) & np.isfinite(cs_close_arr[d_int])
        # restate rows d-1 .. t into day-t units
        scale_arr = factor_arr[t_int][None, :] / factor_arr[d_int - 1 : t_int + 1]
        r_close = none_dict["Close"][d_int - 1 : t_int + 1] * scale_arr
        r_high = none_dict["High"][d_int - 1 : t_int + 1] * scale_arr
        r_low = none_dict["Low"][d_int - 1 : t_int + 1] * scale_arr
        with np.errstate(invalid="ignore", divide="ignore"):
            jump_vec = r_close[1] / r_close[0] - 1.0  # event day d = row 1
            tr_arr = core.true_range_arr(r_high, r_low, r_close)  # row s uses row s-1 as the prior close
            pin_vol_vec = (tr_arr[2:] / r_close[2:]).mean(axis=0)
            hold_vec = r_close[2:].min(axis=0) / r_close[1]
            pin_ref_vec = np.median(r_close[2:], axis=0)
        # CS-based counterparts
        cs_jump_vec = feature_obj.jump_ret()[d_int]
        cs_pin_vol_vec = pin_dict["pin_vol"][d_int]
        with np.errstate(invalid="ignore", divide="ignore"):
            cs_hold_vec = np.fmin.reduce([cs_close_arr[d_int + k] for k in range(1, w_int + 1)]) / cs_close_arr[d_int]
        cs_pin_ref_day_t_vec = pin_dict["pin_ref"][d_int] * factor_arr[t_int]
        max_rel["jump_ret"] = max(max_rel["jump_ret"], _rel(jump_vec[tradable_vec], cs_jump_vec[tradable_vec]))
        max_rel["pin_vol"] = max(max_rel["pin_vol"], _rel(pin_vol_vec[tradable_vec], cs_pin_vol_vec[tradable_vec]))
        max_rel["hold_ratio"] = max(max_rel["hold_ratio"], _rel(hold_vec[tradable_vec], cs_hold_vec[tradable_vec]))
        max_rel["pin_ref_day_t_units"] = max(max_rel["pin_ref_day_t_units"], _rel(pin_ref_vec[tradable_vec], cs_pin_ref_day_t_vec[tradable_vec]))
        with np.errstate(invalid="ignore"):
            jump_flag_r = jump_vec >= cell.jump_float
            jump_flag_cs = cs_jump_vec >= cell.jump_float
            pin_ok_r = pin_dict["traded"][d_int] & np.isfinite(pin_vol_vec) & (hold_vec >= HOLD_FLOOR_FLOAT) & np.isfinite(pin_ref_vec)
            conf_r = (member_arr[d_int] == 1) & jump_flag_r & feature_obj.volume_shock()[d_int] & feature_obj.rel25_prev_pass()[d_int] & pin_ok_r & (pin_vol_vec <= cell.theta_float)
        mismatch["jump_flag"] += int(((jump_flag_r != jump_flag_cs) & tradable_vec).sum())
        mismatch["pin_ok"] += int(((pin_ok_r != pin_dict["pin_ok"][d_int]) & tradable_vec).sum())
        mismatch["conf_flag"] += int(((conf_r != conf_dict["conf"][t_int]) & tradable_vec).sum())
        checked_int += 1
    report_dict["M_R1000"] = {"sessions_checked_int": checked_int, "feature_max_rel_diff": max_rel, "flag_mismatches_tradable": mismatch}
    common.log_progress(f"V2 M R1000: {json.dumps(report_dict['M_R1000'])}")
    del none_dict, feature_obj, universe_dict

    # ---------------------------------------------------------------- family S on the S&P 500 (years covered by the NONE panel)
    universe_dict = data_module.get_universe("SP500")
    feature_obj = PodFeatureBook(universe_dict)
    none_dict = load_none_panel(universe_dict)
    cs_close_arr = feature_obj.close_arr
    with np.errstate(invalid="ignore", divide="ignore"):
        factor_arr = none_dict["Close"] / cs_close_arr
    factor_arr = np.where(np.isfinite(factor_arr) & (factor_arr > 0), factor_arr, np.nan)
    monthly_close_df = data_module.load_monthly_closes_1990("SP500")
    date_index = feature_obj.date_index
    month_end_index = pd.DatetimeIndex(universe_dict["month_end_index"])
    me_pos_by_period = {pd.Timestamp(ts).to_period("M"): int(date_index.get_loc(ts)) for ts in month_end_index}
    # consistency of the two CAPITALSPECIAL sources on their overlap
    overlap_index = monthly_close_df.index.intersection(month_end_index)
    monthly_arr = monthly_close_df.reindex(overlap_index)[universe_dict["symbol_list"]].to_numpy(dtype=np.float64)
    daily_arr = cs_close_arr[date_index.get_indexer(overlap_index)]
    consistency_float = _rel(monthly_arr.ravel(), daily_arr.ravel())
    offset_tuple, min_count_int = cells_module.HORIZON_DICT["SE_1_10"]
    schedule_dict = core.build_schedule(universe_dict, 0)
    traded_decision_vec = schedule_dict["decision_pos_vec"][schedule_dict["trade_vec"]]
    score_max_rel_float = 0.0
    return_max_rel_float = 0.0
    months_checked_int = 0
    for t_int in traded_decision_vec:
        t_int = int(t_int)
        target_period = pd.Timestamp(date_index[t_int]).to_period("M") + 1
        r_list, cs_list = [], []
        for k_int in offset_tuple:
            past_period = target_period - 12 * k_int
            prev_period = past_period - 1
            if past_period not in me_pos_by_period or prev_period not in me_pos_by_period:
                continue
            p2, p1 = me_pos_by_period[past_period], me_pos_by_period[prev_period]
            with np.errstate(invalid="ignore", divide="ignore"):
                restated_ret = (none_dict["Close"][p2] * factor_arr[t_int] / factor_arr[p2]) / (none_dict["Close"][p1] * factor_arr[t_int] / factor_arr[p1]) - 1.0
                cs_ret = cs_close_arr[p2] / cs_close_arr[p1] - 1.0
            r_list.append(restated_ret)
            cs_list.append(cs_ret)
        if not r_list:
            continue
        r_arr, cs_arr = np.stack(r_list), np.stack(cs_list)
        member_vec = universe_dict["member_arr"][t_int] == 1
        return_max_rel_float = max(return_max_rel_float, _rel(r_arr[:, member_vec].ravel(), cs_arr[:, member_vec].ravel()))
        with np.errstate(invalid="ignore"):
            count_vec = np.isfinite(r_arr).sum(axis=0)
            score_r = np.where(count_vec >= min_count_int, np.nanmean(r_arr, axis=0), np.nan)
            score_cs = np.where(count_vec >= min_count_int, np.nanmean(cs_arr, axis=0), np.nan)
        score_max_rel_float = max(score_max_rel_float, _rel(score_r[member_vec], score_cs[member_vec]))
        months_checked_int += 1
    report_dict["S_SP500"] = {"month_ends_checked_int": months_checked_int, "monthly_vs_daily_capitalspecial_close_max_rel_diff": consistency_float,
                              "calendar_month_return_max_rel_diff": return_max_rel_float, "SE_1_10_score_max_rel_diff_same_years": score_max_rel_float,
                              "note": "look-back years before 1999 are outside the NONE panel and are not restated; the score comparison uses the same year set on both sides"}
    common.log_progress(f"V2 S SP500: {json.dumps(report_dict['S_SP500'])}")
    common.write_json("invariance_v2.json", report_dict)
    return report_dict
