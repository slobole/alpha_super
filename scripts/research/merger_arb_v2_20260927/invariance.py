"""Invariance checks (PREREG section 7; research only).

V1  Random per-stock OHLC and Dividend constants (Turnover and Unadjusted Close unchanged) on the confirmed panel: events,
    confirmations, decisions and positions must be identical, and the NAV identical before commissions (the engine's
    phantom re-size fill artifact is documented in the trend study; a phantom-free NAV comparison is also reported).
    Symbols outside the confirmed panel cannot gain an event under rescaling (the jump is a ratio and Turnover is
    nominal), so the panel is the full universe for this check.
V2  Event and pin features recomputed from Norgate NONE closes restated to day t (X_t(d) = X_NONE(d) x R(t) / R(d)) for
    all V0 confirmations plus 200 random candidate events: the jump return, the median and maximum |close-to-close| move,
    the min-close ratio, pin_ref (in day-t units) and the confirmation flag must match up to float rounding.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from merger_arb_v2_20260927 import cells as cells_module
from merger_arb_v2_20260927 import common
from merger_arb_v2_20260927 import data as data_module
from merger_arb_v2_20260927.cells import HOLD_FLOOR_FLOAT, MAX_MOVE_FLOAT
from merger_arb_v2_20260927.features import V2FeatureBook
from merger_arb_v2_20260927.policies import PolicyV
from merger_arb_v2_20260927.simulate import simulate

RANDOM_EVENT_COUNT_INT = 200


def scaled_panel(panel_dict: dict, seed_int: int = common.SEED_INT) -> dict:
    rng = np.random.default_rng(seed_int)
    scale_vec = np.exp(rng.uniform(np.log(0.01), np.log(100.0), len(panel_dict["symbol_list"])))
    scaled_dict = dict(panel_dict)
    for key_str in ("open_arr", "close_arr", "dividend_arr"):
        scaled_dict[key_str] = panel_dict[key_str] * scale_vec[None, :]
    return scaled_dict


def _rel(a_vec: np.ndarray, b_vec: np.ndarray) -> float:
    both_vec = np.isfinite(a_vec) & np.isfinite(b_vec)
    if not both_vec.any():
        return 0.0
    scale_vec = np.maximum(np.maximum(np.abs(a_vec[both_vec]), np.abs(b_vec[both_vec])), 1e-9)
    return float(np.max(np.abs(a_vec[both_vec] - b_vec[both_vec]) / scale_vec))


def run_v1() -> dict:
    panel_dict = data_module.load_panel()
    base_obj = V2FeatureBook(panel_dict)
    scaled_obj = V2FeatureBook(scaled_panel(panel_dict))
    report_dict: dict = {"cells": {}}
    for cell in cells_module.grid_cells():
        base_conf = base_obj.confirmation(cell.jump_float, cell.theta_float, cell.window_int)["conf"]
        scaled_conf = scaled_obj.confirmation(cell.jump_float, cell.theta_float, cell.window_int)["conf"]
        event_same = bool(np.array_equal(base_obj.event(cell.jump_float), scaled_obj.event(cell.jump_float)))
        conf_same = bool(np.array_equal(base_conf, scaled_conf))
        base_sim = simulate(base_obj, PolicyV(base_obj, cell), record_positions_bool=True)
        scaled_sim = simulate(scaled_obj, PolicyV(scaled_obj, cell), record_positions_bool=True)
        position_diff_int = sum(1 for (p0, h0, _), (p1, h1, _) in zip(base_sim["position_log"], scaled_sim["position_log"]) if p0 != p1 or set(h0) != set(h1))
        col_list = ["decision_pos", "symbol_idx", "kind", "reason"]
        b_df, s_df = base_sim["intent_df"], scaled_sim["intent_df"]
        intents_same = len(b_df) == len(s_df) and (len(b_df) == 0 or bool((b_df[col_list].to_numpy() == s_df[col_list].to_numpy()).all()))
        pre_b = base_sim["total_ser"].to_numpy() + base_sim["commission_ser"].cumsum().to_numpy()
        pre_s = scaled_sim["total_ser"].to_numpy() + scaled_sim["commission_ser"].cumsum().to_numpy()
        base_pf = simulate(base_obj, PolicyV(base_obj, cell), phantom_cancel_bool=True)
        scaled_pf = simulate(scaled_obj, PolicyV(scaled_obj, cell), phantom_cancel_bool=True)
        report_dict["cells"][cell.key_str] = {
            "events_identical_bool": event_same, "confirmations_identical_bool": conf_same, "intents_identical_bool": intents_same, "position_mismatch_sessions_int": int(position_diff_int),
            "max_rel_nav_diff_float": float(np.max(np.abs(scaled_sim["total_ser"].to_numpy() / base_sim["total_ser"].to_numpy() - 1.0))),
            "max_rel_nav_diff_before_commissions_float": float(np.max(np.abs(pre_s / pre_b - 1.0))),
            "max_rel_nav_diff_phantom_free_float": float(np.max(np.abs(scaled_pf["total_ser"].to_numpy() / base_pf["total_ser"].to_numpy() - 1.0))),
            "phantom_fills_base_int": int(base_sim["phantom_fill_int"]), "phantom_fills_scaled_int": int(scaled_sim["phantom_fill_int"]),
        }
        print(f"V1 {cell.key_str}: events {event_same}, conf {conf_same}, intents {intents_same}, pos mismatches {position_diff_int}, pre-fee {report_dict['cells'][cell.key_str]['max_rel_nav_diff_before_commissions_float']:.2e}, "
              f"phantom-free {report_dict['cells'][cell.key_str]['max_rel_nav_diff_phantom_free_float']:.2e}", flush=True)
        # memory: each cell's confirmation set is used once per book
        for book_obj in (base_obj, scaled_obj):
            book_obj._cache_dict.pop(("confirmation", cell.jump_float, cell.theta_float, cell.window_int, "ALL"), None)
    report_dict["cells_with_any_decision_difference"] = [k for k, v in report_dict["cells"].items() if not (v["events_identical_bool"] and v["confirmations_identical_bool"] and v["intents_identical_bool"] and v["position_mismatch_sessions_int"] == 0)]
    report_dict["max_rel_nav_diff_phantom_free_over_cells_float"] = float(max(v["max_rel_nav_diff_phantom_free_float"] for v in report_dict["cells"].values()))
    report_dict["max_rel_nav_diff_engine_rule_over_cells_float"] = float(max(v["max_rel_nav_diff_float"] for v in report_dict["cells"].values()))
    common.log_progress(f"V1: differing cells {report_dict['cells_with_any_decision_difference']}, phantom-free NAV rel diff {report_dict['max_rel_nav_diff_phantom_free_over_cells_float']:.2e}, "
                        f"engine-rule {report_dict['max_rel_nav_diff_engine_rule_over_cells_float']:.2e}")
    common.write_json("invariance_v1.json", report_dict)
    return report_dict


def load_none_close_panel(panel_dict: dict) -> np.ndarray:
    import norgatedata as nd

    date_index = pd.DatetimeIndex(panel_dict["date_index"])
    out_arr = np.full((len(date_index), len(panel_dict["symbol_list"])), np.nan)
    for idx, symbol_str in enumerate(panel_dict["symbol_list"]):
        price_df = nd.price_timeseries(symbol_str, stock_price_adjustment_setting=nd.StockPriceAdjustmentType.NONE, padding_setting=nd.PaddingType.ALLMARKETDAYS,
                                       start_date=common.HISTORY_START_STR, end_date=None, timeseriesformat="pandas-dataframe")
        if price_df is None or len(price_df) == 0:
            continue
        out_arr[:, idx] = price_df["Close"].reindex(date_index).to_numpy(dtype=np.float64)
    return out_arr


def run_v2() -> dict:
    rng = np.random.default_rng(common.SEED_INT)
    panel_dict = data_module.load_panel()
    feature_obj = V2FeatureBook(panel_dict)
    none_close_arr = load_none_close_panel(panel_dict)
    cs_close_arr = feature_obj.close_arr
    with np.errstate(invalid="ignore", divide="ignore"):
        factor_arr = none_close_arr / cs_close_arr
    factor_arr = np.where(np.isfinite(factor_arr) & (factor_arr > 0), factor_arr, np.nan)
    cell = cells_module.V0_CELL
    w_int = cell.window_int
    conf_dict = feature_obj.confirmation(cell.jump_float, cell.theta_float, w_int)
    pin_dict = feature_obj.pin(w_int)
    event_arr = feature_obj.event(cell.jump_float)
    conf_pairs = np.argwhere(conf_dict["conf"])  # (t, symbol)
    event_pairs = np.argwhere(event_arr)
    event_pairs = event_pairs[event_pairs[:, 0] + w_int < len(feature_obj.date_index)]
    sample_idx = rng.choice(len(event_pairs), min(RANDOM_EVENT_COUNT_INT, len(event_pairs)), replace=False)
    check_list = [(int(t) - w_int, int(s)) for t, s in conf_pairs] + [(int(d), int(s)) for d, s in event_pairs[sample_idx]]
    max_rel = {"jump_ret": 0.0, "median_abs": 0.0, "max_abs": 0.0, "hold_ratio": 0.0, "pin_ref_day_t_units": 0.0}
    mismatch = {"jump_flag": 0, "pin_ok": 0, "confirmation_flag": 0}
    checked_int = 0
    for d_int, s_int in check_list:
        t_int = d_int + w_int
        if not np.isfinite(factor_arr[t_int, s_int]):
            continue
        # restate rows d-1..t of this symbol into day-t units
        scale_vec = factor_arr[t_int, s_int] / factor_arr[d_int - 1 : t_int + 1, s_int]
        r_close_vec = none_close_arr[d_int - 1 : t_int + 1, s_int] * scale_vec
        with np.errstate(invalid="ignore", divide="ignore"):
            jump_float = r_close_vec[1] / r_close_vec[0] - 1.0
            move_vec = r_close_vec[2:] / r_close_vec[1:-1] - 1.0
            abs_vec = np.abs(move_vec)
            median_abs_float, max_abs_float = float(np.median(abs_vec)), float(np.max(abs_vec))
            hold_float = float(r_close_vec[2:].min() / r_close_vec[1])
            pin_ref_float = float(np.median(r_close_vec[2:]))
        cs_jump_float = feature_obj.close_arr[d_int, s_int] / feature_obj.close_arr[d_int - 1, s_int] - 1.0
        cs_median_float = float(pin_dict["median_abs"][d_int, s_int])
        cs_max_float = float(pin_dict["max_abs"][d_int, s_int])
        cs_hold_float = float(cs_close_arr[d_int + 1 : t_int + 1, s_int].min() / cs_close_arr[d_int, s_int])
        cs_pin_ref_day_t_float = float(pin_dict["pin_ref"][d_int, s_int] * factor_arr[t_int, s_int])
        max_rel["jump_ret"] = max(max_rel["jump_ret"], _rel(np.array([jump_float]), np.array([cs_jump_float])))
        max_rel["median_abs"] = max(max_rel["median_abs"], _rel(np.array([median_abs_float]), np.array([cs_median_float])))
        max_rel["max_abs"] = max(max_rel["max_abs"], _rel(np.array([max_abs_float]), np.array([cs_max_float])))
        max_rel["hold_ratio"] = max(max_rel["hold_ratio"], _rel(np.array([hold_float]), np.array([cs_hold_float])))
        max_rel["pin_ref_day_t_units"] = max(max_rel["pin_ref_day_t_units"], _rel(np.array([pin_ref_float]), np.array([cs_pin_ref_day_t_float])))
        traded_bool = bool(pin_dict["traded"][d_int, s_int])
        pin_ok_r = traded_bool and np.isfinite(abs_vec).all() and max_abs_float <= MAX_MOVE_FLOAT and hold_float >= HOLD_FLOOR_FLOAT and np.isfinite(pin_ref_float)
        conf_r = bool(event_arr[d_int, s_int]) and (jump_float >= cell.jump_float) and pin_ok_r and (median_abs_float <= cell.theta_float)
        mismatch["jump_flag"] += int((jump_float >= cell.jump_float) != (cs_jump_float >= cell.jump_float))
        mismatch["pin_ok"] += int(pin_ok_r != bool(pin_dict["pin_ok"][d_int, s_int]))
        mismatch["confirmation_flag"] += int(conf_r != bool(conf_dict["conf"][t_int, s_int]))
        checked_int += 1
    report_dict = {"confirmations_checked_int": int(len(conf_pairs)), "random_events_checked_int": int(len(sample_idx)), "checked_int": checked_int,
                   "feature_max_rel_diff": max_rel, "flag_mismatches": mismatch}
    common.log_progress(f"V2: {json.dumps(report_dict)}")
    common.write_json("invariance_v2.json", report_dict)
    return report_dict
