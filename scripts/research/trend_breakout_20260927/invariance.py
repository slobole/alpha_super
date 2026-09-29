"""Download-date invariance checks (PREREG section 7; research only).

V1  Every stock's OHLC (and its per-share Dividend) is multiplied by a random constant c_i (log-uniform 0.01..100,
    seed 20260927); Unadjusted Close and Turnover are unchanged. Entry lists, stop decisions and position lists must be
    identical and the NAV path identical before commissions. The only tolerated NAV difference is the engine's own
    phantom-fill artifact: U / Close is not bit-stable across days once prices are rescaled, so a re-size to an
    unchanged raw share count becomes a ~1e-13-share order charged the $1 minimum commission (the replica mirrors the
    engine; the count of such fills is reported).
V2  On every month-end decision date and on 200 random other sessions, features are recomputed from Norgate NONE
    (unadjusted) bars restated into day-t units, X_t(d) = X_NONE(d) x R(t) / R(d), R(x) = U_x / Close_x: what a
    download made on day t shows. Compared: SMA100 / SMA200 passes, NATR20, ATR20$, ROC252, the breakout flags
    HH_50 / HH_100 / HH_250, the daily refill score, and for every name held that day in the CAPITALSPECIAL run of the
    V2 cells: the highest close since entry and the stop decision.
"""

from __future__ import annotations

import json
import pickle

import numpy as np
import pandas as pd

import ndx_param_robustness_core as core  # noqa: E402

from trend_breakout_20260927 import cells as cells_module
from trend_breakout_20260927 import common
from trend_breakout_20260927 import data as data_module
from trend_breakout_20260927.features import DailyFeatureBook
from trend_breakout_20260927.policies import PolicyA, PolicyB
from trend_breakout_20260927.simulate import simulate

V1_CELL_DICT = {
    "NDX": [cells_module.L_REFERENCE_CELL] + cells_module.family_a_cells(),
    "SP500": cells_module.family_b_cells(),
}
V2_CELL_DICT = {
    "NDX": [cells_module.ACell(cells_module.StopSpec("CH", 3.0), "CASH"), cells_module.ACell(cells_module.StopSpec("PT", 0.15), "REFILL")],
    "SP500": [cells_module.B0_CELL],
}
RANDOM_SESSION_COUNT_INT = 200
RESTATE_WINDOW_INT = 320  # covers SMA200, HH_250 (251 closes) and ROC252 (253 closes)


def build_policy(feature_obj: DailyFeatureBook, cell) -> object:
    return PolicyA(feature_obj, cell) if isinstance(cell, cells_module.ACell) else PolicyB(feature_obj, cell)


def scaled_universe(universe_dict: dict, seed_int: int = common.SEED_INT) -> tuple[dict, np.ndarray]:
    rng = np.random.default_rng(seed_int)
    scale_vec = np.exp(rng.uniform(np.log(0.01), np.log(100.0), len(universe_dict["symbol_list"])))
    scaled_dict = dict(universe_dict)
    for field_str in ("open_arr", "high_arr", "low_arr", "close_arr", "dividend_arr"):
        scaled_dict[field_str] = (universe_dict[field_str].astype(np.float64) * scale_vec[None, :]).astype(universe_dict[field_str].dtype)
    scaled_dict["volume_arr"] = (universe_dict["volume_arr"].astype(np.float64) / scale_vec[None, :]).astype(universe_dict["volume_arr"].dtype)
    return scaled_dict, scale_vec


def run_v1() -> dict:
    report_dict: dict = {}
    for universe_str, cell_list in V1_CELL_DICT.items():
        universe_dict = data_module.load_universe(universe_str)
        scaled_dict, _ = scaled_universe(universe_dict)
        base_feature_obj = DailyFeatureBook(universe_dict)
        scaled_feature_obj = DailyFeatureBook(scaled_dict)
        universe_report: dict = {"cells": {}, "cells_checked_int": 0, "cells_with_position_or_decision_difference": []}
        for cell in cell_list:
            base_sim = simulate(base_feature_obj, build_policy(base_feature_obj, cell), record_positions_bool=True)
            scaled_sim = simulate(scaled_feature_obj, build_policy(scaled_feature_obj, cell), record_positions_bool=True)
            position_diff_int = sum(
                1 for (p0, h0, _), (p1, h1, _) in zip(base_sim["position_log"], scaled_sim["position_log"]) if p0 != p1 or set(h0) != set(h1)
            )
            base_intent_df, scaled_intent_df = base_sim["intent_df"], scaled_sim["intent_df"]
            intent_same_bool = len(base_intent_df) == len(scaled_intent_df) and bool(
                (base_intent_df[["decision_pos", "symbol_idx", "kind", "reason"]].to_numpy() == scaled_intent_df[["decision_pos", "symbol_idx", "kind", "reason"]].to_numpy()).all()
            ) if len(base_intent_df) == len(scaled_intent_df) else False
            pre_fee_base_vec = base_sim["total_ser"].to_numpy() + base_sim["commission_ser"].cumsum().to_numpy()
            pre_fee_scaled_vec = scaled_sim["total_ser"].to_numpy() + scaled_sim["commission_ser"].cumsum().to_numpy()
            cell_report = {
                "position_mismatch_sessions_int": int(position_diff_int),
                "intents_identical_bool": intent_same_bool,
                "max_rel_nav_diff_float": float(np.max(np.abs(scaled_sim["total_ser"].to_numpy() / base_sim["total_ser"].to_numpy() - 1.0))),
                "max_rel_nav_diff_before_commissions_float": float(np.max(np.abs(pre_fee_scaled_vec / pre_fee_base_vec - 1.0))),
                "commission_total_base_usd": float(base_sim["commission_ser"].sum()),
                "commission_total_scaled_usd": float(scaled_sim["commission_ser"].sum()),
                "phantom_fills_base_int": int(base_sim["phantom_fill_int"]),
                "phantom_fills_scaled_int": int(scaled_sim["phantom_fill_int"]),
            }
            universe_report["cells"][cell.key_str] = cell_report
            universe_report["cells_checked_int"] += 1
            if position_diff_int or not intent_same_bool:
                universe_report["cells_with_position_or_decision_difference"].append(cell.key_str)
            print(f"V1 {universe_str} {cell.key_str}: pos mismatches {position_diff_int}, intents same {intent_same_bool}, "
                  f"nav rel diff {cell_report['max_rel_nav_diff_float']:.2e}, pre-fee {cell_report['max_rel_nav_diff_before_commissions_float']:.2e}, "
                  f"phantom {cell_report['phantom_fills_base_int']}/{cell_report['phantom_fills_scaled_int']}", flush=True)
        universe_report["max_rel_nav_diff_before_commissions_over_cells_float"] = float(
            max(c["max_rel_nav_diff_before_commissions_float"] for c in universe_report["cells"].values())
        )
        report_dict[universe_str] = universe_report
        common.log_progress(
            f"V1 {universe_str}: {universe_report['cells_checked_int']} cells, differing cells {universe_report['cells_with_position_or_decision_difference']}, "
            f"max pre-fee NAV rel diff {universe_report['max_rel_nav_diff_before_commissions_over_cells_float']:.2e}"
        )
    common.write_json("invariance_v1.json", report_dict)
    return report_dict


def run_v1_phantom_free() -> dict:
    """Diagnostic for note N1: the same V1 comparison with phantom fills (|delta| < 1e-6 ledger shares) cancelled in
    both runs. If the phantom-fill commissions are the whole source of the V1 NAV differences, NAV must now agree to
    ~1e-9 relative on every cell."""
    report_dict: dict = {}
    for universe_str, cell_list in V1_CELL_DICT.items():
        universe_dict = data_module.load_universe(universe_str)
        scaled_dict, _ = scaled_universe(universe_dict)
        base_feature_obj = DailyFeatureBook(universe_dict)
        scaled_feature_obj = DailyFeatureBook(scaled_dict)
        cell_report_dict = {}
        for cell in cell_list:
            base_sim = simulate(base_feature_obj, build_policy(base_feature_obj, cell), record_positions_bool=True, phantom_cancel_bool=True)
            scaled_sim = simulate(scaled_feature_obj, build_policy(scaled_feature_obj, cell), record_positions_bool=True, phantom_cancel_bool=True)
            position_diff_int = sum(1 for (p0, h0, _), (p1, h1, _) in zip(base_sim["position_log"], scaled_sim["position_log"]) if p0 != p1 or set(h0) != set(h1))
            cell_report_dict[cell.key_str] = {
                "max_rel_nav_diff_float": float(np.max(np.abs(scaled_sim["total_ser"].to_numpy() / base_sim["total_ser"].to_numpy() - 1.0))),
                "position_mismatch_sessions_int": int(position_diff_int),
                "phantom_fills_cancelled_base_int": int(base_sim["phantom_fill_int"]),
                "phantom_fills_cancelled_scaled_int": int(scaled_sim["phantom_fill_int"]),
            }
            print(f"V1 phantom-free {universe_str} {cell.key_str}: nav rel diff {cell_report_dict[cell.key_str]['max_rel_nav_diff_float']:.2e}", flush=True)
        report_dict[universe_str] = {"cells": cell_report_dict, "max_rel_nav_diff_over_cells_float": float(max(c["max_rel_nav_diff_float"] for c in cell_report_dict.values()))}
        common.log_progress(f"V1 phantom-free {universe_str}: max NAV rel diff {report_dict[universe_str]['max_rel_nav_diff_over_cells_float']:.2e}")
    common.write_json("invariance_v1_phantom_free.json", report_dict)
    return report_dict


# ----------------------------------------------------------------------------------------------------------------------
# V2
# ----------------------------------------------------------------------------------------------------------------------
def load_none_panel(universe_dict: dict) -> dict[str, np.ndarray]:
    import norgatedata as norgatedata_module

    date_index = pd.DatetimeIndex(universe_dict["date_index"])
    panel_dict = {k: np.full((len(date_index), len(universe_dict["symbol_list"])), np.nan) for k in ("Open", "High", "Low", "Close")}
    for symbol_idx, symbol_str in enumerate(universe_dict["symbol_list"]):
        price_df = norgatedata_module.price_timeseries(
            symbol_str,
            stock_price_adjustment_setting=norgatedata_module.StockPriceAdjustmentType.NONE,
            padding_setting=norgatedata_module.PaddingType.ALLMARKETDAYS,
            start_date="1999-01-01",
            end_date=None,
            timeseriesformat="pandas-dataframe",
        )
        if price_df is None or len(price_df) == 0:
            continue
        price_df = price_df.reindex(date_index)
        for field_str in panel_dict:
            panel_dict[field_str][:, symbol_idx] = price_df[field_str].to_numpy(dtype=np.float64)
    return panel_dict


def restated_window(none_dict: dict, factor_arr: np.ndarray, start_int: int, end_int: int, t_int: int) -> dict[str, np.ndarray]:
    """Bars d in [start, end] restated into day-t units: X_NONE(d) x R(t) / R(d). Rows after t are never touched."""
    scale_arr = factor_arr[t_int][None, :] / factor_arr[start_int : end_int + 1]
    return {k: none_dict[k][start_int : end_int + 1] * scale_arr for k in none_dict}


def window_features(open_arr, high_arr, low_arr, close_arr, unadj_vec) -> dict[str, np.ndarray]:
    """Features at the LAST row of a restated window (same formulas as DailyFeatureBook)."""
    tr_arr = core.true_range_arr(high_arr, low_arr, close_arr)
    out_dict: dict[str, np.ndarray] = {}
    atr_vec = tr_arr[-20:].mean(axis=0)
    out_dict["NATR20"] = atr_vec / close_arr[-1]
    out_dict["ATR20_dollar"] = atr_vec * unadj_vec / close_arr[-1]
    with np.errstate(invalid="ignore"):
        for n_int in (100, 200):
            out_dict[f"SMA{n_int}_pass"] = close_arr[-1] > close_arr[-n_int:].mean(axis=0)
        for n_int in (50, 100, 250):
            out_dict[f"HH{n_int}_pass"] = close_arr[-1] > np.max(close_arr[-n_int - 1 : -1], axis=0)
    out_dict["ROC252"] = close_arr[-1] / close_arr[-253] - 1.0
    return out_dict


def run_v2() -> dict:
    rng = np.random.default_rng(common.SEED_INT)
    report_dict: dict = {}
    for universe_str, cell_list in V2_CELL_DICT.items():
        universe_dict = data_module.load_universe(universe_str)
        feature_obj = DailyFeatureBook(universe_dict)
        none_dict = load_none_panel(universe_dict)
        cs_close_arr = feature_obj.close_arr
        with np.errstate(invalid="ignore", divide="ignore"):
            factor_arr = none_dict["Close"] / cs_close_arr
        factor_arr = np.where(np.isfinite(factor_arr) & (factor_arr > 0), factor_arr, np.nan)
        unadj_arr = feature_obj.unadjusted_close()
        schedule_dict = core.build_schedule(universe_dict, 0)
        decision_pos_vec = schedule_dict["decision_pos_vec"][schedule_dict["trade_vec"]]
        other_pos_vec = np.array(sorted(set(range(RESTATE_WINDOW_INT, len(feature_obj.date_index) - 1)) - set(decision_pos_vec.tolist())))
        sampled_pos_vec = np.sort(np.concatenate([decision_pos_vec, rng.choice(other_pos_vec, RANDOM_SESSION_COUNT_INT, replace=False)]))
        member_arr = universe_dict["member_arr"]

        # CS-run positions of the V2 cells
        run_dict = {}
        for cell in cell_list:
            sim = simulate(feature_obj, build_policy(feature_obj, cell), record_positions_bool=True)
            run_dict[cell.key_str] = {"cell": cell, "positions": {p: (h, e) for p, h, e in sim["position_log"]}}

        max_rel_dict: dict[str, float] = {}
        mismatch_dict: dict[str, int] = {}
        tradable_nan_mismatch_dict: dict[str, int] = {}
        daily_score_arr = feature_obj.daily_l_score(schedule_dict)
        anchor_vec = feature_obj.me12_anchor_pos(schedule_dict)
        stop_mismatch_dict = {key: 0 for key in run_dict}
        hwm_max_rel_float = 0.0
        held_checked_int = 0

        def update(key_str: str, a_vec: np.ndarray, b_vec: np.ndarray, tradable_vec: np.ndarray, bool_flag: bool = False) -> None:
            if bool_flag:
                mismatch_dict[key_str] = mismatch_dict.get(key_str, 0) + int(((a_vec != b_vec) & tradable_vec).sum())
                return
            both_vec = np.isfinite(a_vec) & np.isfinite(b_vec)
            # relative difference scaled by the larger magnitude; an exact-zero CAPITALSPECIAL value (e.g. a return of
            # exactly 0) against a 1e-17 restated value is float rounding, not a discrepancy, so the scale is floored.
            scale_vec = np.maximum(np.maximum(np.abs(a_vec[both_vec]), np.abs(b_vec[both_vec])), 1e-9)
            rel_float = float(np.max(np.abs(a_vec[both_vec] - b_vec[both_vec]) / scale_vec)) if both_vec.any() else 0.0
            max_rel_dict[key_str] = max(max_rel_dict.get(key_str, 0.0), rel_float)
            tradable_nan_mismatch_dict[key_str] = tradable_nan_mismatch_dict.get(key_str, 0) + int(((np.isfinite(a_vec) != np.isfinite(b_vec)) & tradable_vec).sum())

        for t_int in sampled_pos_vec:
            t_int = int(t_int)
            start_int = max(0, t_int - RESTATE_WINDOW_INT + 1)
            restated = restated_window(none_dict, factor_arr, start_int, t_int, t_int)
            feat_dict = window_features(restated["Open"], restated["High"], restated["Low"], restated["Close"], unadj_arr[t_int])
            tradable_vec = (member_arr[t_int] == 1) & np.isfinite(cs_close_arr[t_int])
            update("NATR20", feat_dict["NATR20"], feature_obj.natr(20)[t_int], tradable_vec)
            update("ATR20_dollar", feat_dict["ATR20_dollar"], feature_obj.atr_dollar()[t_int], tradable_vec)
            update("ROC252", feat_dict["ROC252"], feature_obj.roc_daily(252)[t_int], tradable_vec)
            for n_int in (100, 200):
                update(f"SMA{n_int}_pass", feat_dict[f"SMA{n_int}_pass"], feature_obj.trend_pass(n_int)[t_int], tradable_vec, bool_flag=True)
            for n_int in (50, 100, 250):
                update(f"HH{n_int}_pass", feat_dict[f"HH{n_int}_pass"], feature_obj.breakout_pass(n_int)[t_int], tradable_vec, bool_flag=True)
            # daily refill score: anchor close restated into day-t units
            a_int = int(anchor_vec[t_int])
            if a_int >= 0:
                anchor_close_vec = none_dict["Close"][a_int] * factor_arr[t_int] / factor_arr[a_int]
                with np.errstate(invalid="ignore", divide="ignore"):
                    restated_score_vec = (restated["Close"][-1] / anchor_close_vec - 1.0) / feat_dict["ATR20_dollar"]
                update("daily_refill_score", restated_score_vec, daily_score_arr[t_int], tradable_vec)
            # held names: highest close since entry and the stop decision, restated from the entry day
            for key_str, run in run_dict.items():
                held_idx_vec, entry_vec = run["positions"].get(t_int, (np.array([], dtype=int), np.array([], dtype=int)))
                cell = run["cell"]
                for symbol_idx, entry_int in zip(held_idx_vec, entry_vec):
                    symbol_idx, entry_int = int(symbol_idx), int(entry_int)
                    restated_close_vec = none_dict["Close"][entry_int : t_int + 1, symbol_idx] * factor_arr[t_int, symbol_idx] / factor_arr[entry_int : t_int + 1, symbol_idx]
                    hwm_restated_float = float(np.nanmax(restated_close_vec))  # day-t nominal units
                    hwm_cs_float = float(np.nanmax(cs_close_arr[entry_int : t_int + 1, symbol_idx]))  # today's adjusted units
                    # *** CRITICAL *** compare in the same units: the adjusted HWM x R(t) is the HWM a day-t download shows.
                    hwm_max_rel_float = max(hwm_max_rel_float, abs(hwm_restated_float / (hwm_cs_float * factor_arr[t_int, symbol_idx]) - 1.0))
                    fires_restated = bool(cell.stop.fires_vec(restated["Close"][-1][[symbol_idx]], np.array([hwm_restated_float]), feat_dict["ATR20_dollar"][[symbol_idx]] * restated["Close"][-1][[symbol_idx]] / unadj_arr[t_int, symbol_idx])[0])
                    fires_cs = bool(cell.stop.fires_vec(cs_close_arr[t_int][[symbol_idx]], np.array([hwm_cs_float]), feature_obj.atr_cs(20)[t_int][[symbol_idx]])[0])
                    held_checked_int += 1
                    if fires_restated != fires_cs:
                        stop_mismatch_dict[key_str] += 1
        report_dict[universe_str] = {
            "sessions_checked_int": int(len(sampled_pos_vec)),
            "feature_max_rel_diff": max_rel_dict,
            "feature_nan_mismatch_tradable": tradable_nan_mismatch_dict,
            "bool_feature_mismatches_tradable": mismatch_dict,
            "held_names_checked_int": held_checked_int,
            "hwm_max_rel_diff_float": hwm_max_rel_float,
            "stop_decision_mismatches_by_cell": stop_mismatch_dict,
        }
        common.log_progress(f"V2 {universe_str}: {json.dumps(report_dict[universe_str], default=str)[:800]}")
    common.write_json("invariance_v2.json", report_dict)
    return report_dict
