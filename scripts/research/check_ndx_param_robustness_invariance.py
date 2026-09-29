"""Download-date invariance checks for the NDX parameter-robustness study (PREREG section 7; research only).

V1  Every stock's OHLC is multiplied by a random constant c_i (log-uniform 0.01..100; volume divided by c_i).
    Every scale-free NDX cell must produce the same list (and weights) on every decision date.
V2  On every month-end decision date t, the features are recomputed from Norgate unadjusted (NONE) bars restated
    into day-t units, X_t(d) = X_NONE(d) * R(t) / R(d), R(x) = Close_NONE(x) / Close_CS(x): exactly what a download
    made on day t shows. R(t)/R(d) contains only capital events in (d, t], all known at t.
    Reports the largest relative feature difference versus the CAPITALSPECIAL computation and whether the stage-1
    lists (plus L and B) are identical.

    uv run python scripts/research/check_ndx_param_robustness_invariance.py
"""

from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ndx_param_robustness_core as core  # noqa: E402

WINDOW_INT = 300  # sessions of history restated per decision date (covers SMA200 and the 12-month ROC anchor)


def selection_signature(target_list: list[dict]) -> dict[int, tuple]:
    return {t["decision_pos"]: (t["symbol_idx_vec"], t["weight_vec"]) for t in target_list}


def same_selection_bool(a_tuple, b_tuple) -> bool:
    """Same names exactly, same weights to 1e-9 relative (float rounding of rescaled prices is ~1e-16)."""
    if b_tuple is None:
        return False
    a_order, b_order = np.argsort(a_tuple[0]), np.argsort(b_tuple[0])
    if not np.array_equal(a_tuple[0][a_order], b_tuple[0][b_order]):
        return False
    return bool(np.allclose(a_tuple[1][a_order], b_tuple[1][b_order], rtol=1e-9, atol=0.0))


def run_v1(universe_dict: dict) -> dict:
    rng = np.random.default_rng(20260926)
    scale_vec = np.exp(rng.uniform(np.log(0.01), np.log(100.0), len(universe_dict["symbol_list"])))
    scaled_dict = dict(universe_dict)
    for field_str in ("open_arr", "high_arr", "low_arr", "close_arr"):
        scaled_dict[field_str] = universe_dict[field_str] * scale_vec[None, :]
    scaled_dict["volume_arr"] = universe_dict["volume_arr"] / scale_vec[None, :]
    base_feature_obj = core.FeatureBook(universe_dict)
    scaled_feature_obj = core.FeatureBook(scaled_dict)
    result_dict: dict = {"cells_checked_int": 0, "cells_with_any_difference": [], "decision_dates_int": 0}
    for cell in core.all_cell_list():
        if not cell.scale_free_bool:
            continue
        base_sig = selection_signature(core.build_target_list(base_feature_obj, cell))
        scaled_sig = selection_signature(core.build_target_list(scaled_feature_obj, cell))
        diff_list = [pos for pos in base_sig if not same_selection_bool(base_sig[pos], scaled_sig.get(pos))]
        result_dict["cells_checked_int"] += 1
        result_dict["decision_dates_int"] += len(base_sig)
        if diff_list:
            result_dict["cells_with_any_difference"].append({"cell": cell.key_str, "dates_int": len(diff_list)})
    # contrast: the dollar-unit reference B changes under the same rescaling
    base_b = selection_signature(core.build_target_list(base_feature_obj, core.B_CELL))
    scaled_b = selection_signature(core.build_target_list(scaled_feature_obj, core.B_CELL))
    result_dict["contrast_B_dates_changed_share"] = float(
        np.mean([not same_selection_bool(base_b[p], scaled_b.get(p)) for p in base_b if len(base_b[p][0])])
    )
    return result_dict


def load_none_panel(universe_dict: dict) -> dict[str, np.ndarray]:
    import norgatedata as nd

    date_index = universe_dict["date_index"]
    panel_dict = {k: np.full((len(date_index), len(universe_dict["symbol_list"])), np.nan) for k in ("Open", "High", "Low", "Close")}
    for symbol_idx, symbol_str in enumerate(universe_dict["symbol_list"]):
        price_df = nd.price_timeseries(
            symbol_str,
            stock_price_adjustment_setting=nd.StockPriceAdjustmentType.NONE,
            padding_setting=nd.PaddingType.ALLMARKETDAYS,
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


def window_features(open_arr, high_arr, low_arr, close_arr) -> dict[str, np.ndarray]:
    """Features at the LAST row of a restated window (same formulas as the core FeatureBook)."""
    tr_arr = core.true_range_arr(high_arr, low_arr, close_arr)
    out_dict = {}
    for n_int in (20, 63):
        out_dict[f"NATR{n_int}"] = tr_arr[-n_int:].mean(axis=0) / close_arr[-1]
    for n_int in (50, 100, 200):
        with np.errstate(invalid="ignore"):
            out_dict[f"SMA{n_int}"] = close_arr[-1] > close_arr[-n_int:].mean(axis=0)
    return_arr = close_arr[-63:] / close_arr[-64:-1] - 1.0
    valid_count_vec = np.isfinite(return_arr).sum(axis=0)
    sigma_vec = pd.DataFrame(return_arr).std().to_numpy()
    out_dict["sigma63"] = np.where(valid_count_vec >= 60, sigma_vec, np.nan)
    out_dict["ATR20_dollars_day_t"] = tr_arr[-20:].mean(axis=0)
    return out_dict


def run_v2(universe_dict: dict) -> dict:
    feature_obj = core.FeatureBook(universe_dict)
    none_dict = load_none_panel(universe_dict)
    cs_close_arr = feature_obj.close_arr
    # *** CRITICAL *** R(x) embeds events AFTER x; it is only ever used as the ratio R(t)/R(d) with d <= t below.
    factor_arr = none_dict["Close"] / cs_close_arr
    factor_arr = np.where(np.isfinite(factor_arr) & (factor_arr > 0), factor_arr, np.nan)
    schedule_dict = core.build_schedule(universe_dict, 0)
    traded_row_vec = np.flatnonzero(schedule_dict["trade_vec"])
    decision_pos_vec = schedule_dict["decision_pos_vec"]

    stage1_cell_list = [cell for _, cell in core.stage_grid_dict()["S1_score"]] + [core.L_CELL, core.B_CELL]
    cs_list_dict = {
        cell.key_str: {t["decision_pos"]: set(t["symbol_idx_vec"].tolist()) for t in core.build_target_list(feature_obj, cell)}
        for cell in stage1_cell_list
    }
    numerator_cs_dict = {n: core.roc_table(cs_close_arr, schedule_dict, n) for n in core.STAGE1_NUMERATOR_TUPLE}
    member_arr = universe_dict["member_arr"]
    regime_vec = feature_obj.regime_pass("SPY")
    max_rel_diff_dict: dict[str, float] = {}
    trend_mismatch_int = 0
    list_mismatch_dict = {cell.key_str: 0 for cell in stage1_cell_list}
    dates_int = 0

    # restated decision-row panels, later swapped into a FeatureBook to rebuild every k = 0 cell (all stages)
    restated_row_dict: dict[str, np.ndarray] = {
        name: np.full(cs_close_arr.shape, np.nan) for name in ("NATR20", "NATR63", "sigma63", "ATR20_dollars_day_t")
    }
    restated_trend_dict = {n: feature_obj.trend_pass(n).copy() for n in (50, 100, 200)}
    restated_numerator_dict = {n: numerator_cs_dict[n].copy() for n in core.STAGE1_NUMERATOR_TUPLE}
    tradable_t_mask_list: list[np.ndarray] = []

    def update_max(key_str: str, a_vec: np.ndarray, b_vec: np.ndarray) -> None:
        both_vec = np.isfinite(a_vec) & np.isfinite(b_vec)
        nan_mismatch_int = int((np.isfinite(a_vec) != np.isfinite(b_vec)).sum())
        # NaN mismatches among names that could be selected on day t (PIT member with a close on day t)
        tradable_vec = tradable_t_mask_list[-1]
        tradable_mismatch_int = int(((np.isfinite(a_vec) != np.isfinite(b_vec)) & tradable_vec).sum())
        max_rel_diff_dict[key_str + "_nan_mismatch_tradable"] = max_rel_diff_dict.get(key_str + "_nan_mismatch_tradable", 0) + tradable_mismatch_int
        rel_float = float(np.max(np.abs(a_vec[both_vec] / b_vec[both_vec] - 1.0))) if both_vec.any() else 0.0
        max_rel_diff_dict[key_str] = max(max_rel_diff_dict.get(key_str, 0.0), rel_float)
        max_rel_diff_dict[key_str + "_nan_mismatch"] = max_rel_diff_dict.get(key_str + "_nan_mismatch", 0) + nan_mismatch_int

    for row_int in traded_row_vec:
        t_pos = int(decision_pos_vec[row_int])
        window_slice = slice(max(0, t_pos - WINDOW_INT + 1), t_pos + 1)
        # *** CRITICAL *** restate bars d <= t into day-t units; rows after t are never touched.
        scale_arr = factor_arr[t_pos][None, :] / factor_arr[window_slice]
        restated = {k: none_dict[k][window_slice] * scale_arr for k in none_dict}
        feat_dict = window_features(restated["Open"], restated["High"], restated["Low"], restated["Close"])
        dates_int += 1
        tradable_t_mask_list.append((member_arr[t_pos] == 1) & np.isfinite(cs_close_arr[t_pos]))
        for name in restated_row_dict:
            restated_row_dict[name][t_pos] = feat_dict[name]
        for n_int in (50, 100, 200):
            restated_trend_dict[n_int][t_pos] = feat_dict[f"SMA{n_int}"]
        for n_int in (20, 63):
            update_max(f"NATR{n_int}", feat_dict[f"NATR{n_int}"], feature_obj.natr(n_int)[t_pos])
        update_max("sigma63", feat_dict["sigma63"], feature_obj.sigma63()[t_pos])
        for n_int in (50, 100, 200):
            trend_mismatch_int += int((feat_dict[f"SMA{n_int}"] != feature_obj.trend_pass(n_int)[t_pos]).sum())
        # restated numerators: anchor closes NONE(a) * R(t) / R(a)
        numerator_dict = {}
        anchor_close_dict = {}
        for back_int in (0, 1, 3, 6, 9, 12):
            a_pos = int(decision_pos_vec[row_int - back_int])
            anchor_close_dict[back_int] = none_dict["Close"][a_pos] * factor_arr[t_pos] / factor_arr[a_pos]
        roc = lambda b, s=0: anchor_close_dict[s] / anchor_close_dict[b] - 1.0  # noqa: E731
        numerator_dict.update({"ROC3": roc(3), "ROC6": roc(6), "ROC9": roc(9), "ROC12": roc(12), "ROC12-1": roc(12, 1)})
        numerator_dict["B3612"] = (roc(3) + roc(6) + roc(12)) / 3.0
        numerator_dict["B612"] = (roc(6) + roc(12)) / 2.0
        for n_str, num_vec in numerator_dict.items():
            update_max(f"num_{n_str}", num_vec, numerator_cs_dict[n_str][row_int])
            restated_numerator_dict[n_str][row_int] = num_vec
        # lists
        for cell in stage1_cell_list:
            num_vec = numerator_dict[cell.numerator_str]
            with np.errstate(divide="ignore", invalid="ignore"):
                if cell.denominator_str == "none":
                    score_vec = num_vec
                elif cell.denominator_str.startswith("NATR"):
                    score_vec = num_vec / feat_dict[cell.denominator_str]
                else:  # L and B: what a day-t download computes is dollar ATR in day-t dollars for both
                    score_vec = num_vec / feat_dict["ATR20_dollars_day_t"]
            score_vec = np.where(np.isfinite(score_vec), score_vec, np.nan)
            eligible_vec = (member_arr[t_pos] == 1) & feat_dict["SMA100"] & np.isfinite(score_vec) & bool(regime_vec[t_pos])
            idx_vec = np.flatnonzero(eligible_vec)
            order_vec = np.lexsort((feature_obj.symbol_rank_vec[idx_vec], -score_vec[idx_vec]))
            day_t_set = set(idx_vec[order_vec][:10].tolist())
            if day_t_set != cs_list_dict[cell.key_str].get(t_pos, set()):
                list_mismatch_dict[cell.key_str] += 1
    # Rebuild every k = 0 cell of every stage (plus L and B) from the day-t view and compare lists and weights.
    restated_obj = core.FeatureBook(universe_dict)
    restated_obj._cache_dict[("natr", 20)] = restated_row_dict["NATR20"]
    restated_obj._cache_dict[("natr", 63)] = restated_row_dict["NATR63"]
    restated_obj._cache_dict[("sigma63",)] = restated_row_dict["sigma63"]
    restated_obj._cache_dict[("atr_cs", 20)] = restated_row_dict["ATR20_dollars_day_t"]  # day-t dollars
    restated_obj._cache_dict[("split_factor",)] = np.ones_like(cs_close_arr)  # day-t units already
    for n_int in (50, 100, 200):
        restated_obj._cache_dict[("trend", n_int)] = restated_trend_dict[n_int]
    restated_obj.numerator_override_dict = restated_numerator_dict
    all_cell_mismatch_dict = {}
    for cell in core.all_cell_list():
        if cell.offset_int != 0:
            continue
        base_sig = selection_signature(core.build_target_list(feature_obj, cell))
        day_t_sig = selection_signature(core.build_target_list(restated_obj, cell))
        all_cell_mismatch_dict[cell.key_str] = int(sum(not same_selection_bool(base_sig[p], day_t_sig.get(p)) for p in base_sig))
    scale_free_mismatch_list = [k for k, v in all_cell_mismatch_dict.items() if v and "ATR20_CS" not in k]
    return {
        "all_k0_cells_checked_int": len(all_cell_mismatch_dict),
        "all_k0_cells_with_list_or_weight_mismatch_excluding_B": scale_free_mismatch_list,
        "B_mismatch_dates_int": all_cell_mismatch_dict.get(core.B_CELL.key_str),
        "decision_dates_int": dates_int,
        "feature_max_rel_diff": max_rel_diff_dict,
        "trend_filter_mismatch_count_int": trend_mismatch_int,
        "list_mismatch_dates_by_cell": list_mismatch_dict,
        "note": "B mismatches are expected: B ranks on today's split-adjusted dollars, the day-t view does not.",
    }


def main() -> None:
    universe_dict = core.load_universe("NDX")
    report_dict = {"V1_random_rescaling": run_v1(universe_dict)}
    print(json.dumps(report_dict["V1_random_rescaling"], indent=1), flush=True)
    report_dict["V2_decision_day_restatement"] = run_v2(universe_dict)
    print(json.dumps(report_dict["V2_decision_day_restatement"], indent=1))
    (core.RESULTS_DIR_PATH / "invariance_checks.json").write_text(json.dumps(report_dict, indent=2))


if __name__ == "__main__":
    main()
