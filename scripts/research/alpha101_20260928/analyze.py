"""Analysis for the Alpha101 study (PREREG sections 8 and 10; research only): applied mechanically.

Reads the pod outputs written by run_pods.py, builds the pod returns with the idle-cash sweep, the candidate books
{TAA 0.5, L 0.25, X 0.25} and the controls (C_BIL, C_SPY, G3, the dv2 slot) in the official pod model, the plateau
candidates, the rule R1-R5 versus C_BIL (neighbourhood medians, as in the earlier new-pod studies), every label, the
confidence labels, capacity and small-account economics, the charts and summary_for_lead.json.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from alpha101_20260928 import checks
from alpha101_20260928 import common
from alpha101_20260928 import run_pods
from trend_breakout_20260927 import analyze as trend_analyze

OUT_PATH = common.RESULTS_DIR_PATH
CHART_PATH = common.CHART_DIR_PATH
POST_BLOCK_TUPLE = ("2016-01-01", "2026-08-19")
BOOK_START_STR = "2008-03-04"
ORDER_LIST = ["C_BIL", "C_SPY", "G3", "dv2_slot"]


# ----------------------------------------------------------------------------------------------------------------------
def blocks_for(key_str: str) -> dict:
    return common.BLOCK_DICT_HEDGED if key_str.endswith("|HEDGED") else common.BLOCK_DICT


def beta_to(x_ser: pd.Series, y_ser: pd.Series, start_str: str, end_str: str) -> float:
    frame_df = pd.concat([x_ser, y_ser], axis=1).loc[start_str:end_str].dropna()
    if len(frame_df) < 30:
        return float("nan")
    x_vec, y_vec = frame_df.iloc[:, 0].to_numpy(), frame_df.iloc[:, 1].to_numpy()
    return float(np.cov(x_vec, y_vec)[0, 1] / np.var(y_vec, ddof=1))


def alpha_vs_market(x_ser: pd.Series, spy_ser: pd.Series, start_str: str, end_str: str) -> dict:
    """OLS of the daily pod return on SPY total return: annualized intercept and its Newey-West (5 lags) t."""
    frame_df = pd.concat([x_ser, spy_ser], axis=1).loc[start_str:end_str].dropna()
    if len(frame_df) < 60:
        return {"alpha_ann": float("nan"), "t_nw5": float("nan"), "beta": float("nan"), "n": int(len(frame_df))}
    x_vec, m_vec = frame_df.iloc[:, 0].to_numpy(), frame_df.iloc[:, 1].to_numpy()
    beta_float = float(np.cov(x_vec, m_vec)[0, 1] / np.var(m_vec, ddof=1))
    residual_vec = x_vec - beta_float * m_vec
    return {"alpha_ann": float(residual_vec.mean() * 252.0), "t_nw5": common.newey_west_t(residual_vec), "beta": beta_float, "n": int(len(frame_df))}


def corr_from(x_ser: pd.Series, y_ser: pd.Series, start_str: str, end_str: str = "2026-08-19") -> float:
    frame_df = pd.concat([x_ser, y_ser], axis=1).loc[start_str:end_str].dropna()
    return float(frame_df.iloc[:, 0].corr(frame_df.iloc[:, 1])) if len(frame_df) > 30 else float("nan")


def neigh_median(key_list: list[str], getter) -> float:
    value_list = []
    for key_str in key_list:
        try:
            value_obj = getter(key_str)
        except (KeyError, TypeError):
            continue
        if value_obj is not None and np.isfinite(value_obj):
            value_list.append(float(value_obj))
    return float(np.median(value_list)) if value_list else float("nan")


def margins(cell_res: dict, neigh: list[str], book_str: str, control: dict) -> dict:
    margin_dict = {b: neigh_median(neigh, lambda k: cell_res[k]["books"][book_str][b]["sharpe"]) - control[b]["sharpe"] for b in common.RULE_BLOCK_TUPLE}
    dd_gap_dict = {b: neigh_median(neigh, lambda k: cell_res[k]["books"][book_str][b]["max_dd"]) - control[b]["max_dd"] for b in ("G-FULL", "G-LONG")}
    return {"sharpe_margin": margin_dict, "min_margin": float(min(margin_dict.values())), "dd_gap_pp": {b: v * 100 for b, v in dd_gap_dict.items()},
            "R1": bool(all(np.isfinite(m) and m > 0 for m in margin_dict.values())), "R2": bool(all(np.isfinite(v) and v >= -common.DD_TOLERANCE_FLOAT for v in dd_gap_dict.values())),
            "g_full_margin": neigh_median(neigh, lambda k: cell_res[k]["books"][book_str]["G-FULL"]["sharpe"]) - control["G-FULL"]["sharpe"],
            "neigh_median_sharpe": {b: neigh_median(neigh, lambda k: cell_res[k]["books"][book_str][b]["sharpe"]) for b in common.BOOK_BLOCK_DICT},
            "neigh_median_max_dd": {b: neigh_median(neigh, lambda k: cell_res[k]["books"][book_str][b]["max_dd"]) for b in common.BOOK_BLOCK_DICT},
            "control_sharpe": {b: control[b]["sharpe"] for b in common.BOOK_BLOCK_DICT}, "control_max_dd": {b: control[b]["max_dd"] for b in common.BOOK_BLOCK_DICT}}


def load_meta(tag_str: str) -> dict:
    return common.read_json(f"meta_{tag_str}.json", {}) or {}


# ----------------------------------------------------------------------------------------------------------------------
def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    CHART_PATH.mkdir(parents=True, exist_ok=True)
    seq_cmap = LinearSegmentedColormap.from_list("seq", trend_analyze.BLUE_RAMP)
    div_cmap = LinearSegmentedColormap.from_list("div", trend_analyze.DIVERGING)
    common.log_progress("analysis start")
    series_dict = checks.control_series()
    taa_ser, bil_ser, spy_ser, dv2_ser = series_dict["taa"], series_dict["bil"], series_dict["spy"], series_dict["dv2"]
    l_engine_ser, l_stress_ser = series_dict["L_engine"], series_dict["L_stress"]
    controls = common.control_books(taa_ser, l_engine_ser, bil_ser, spy_ser)
    controls["dv2_slot"] = common.candidate_book_blocks(taa_ser, l_engine_ser, dv2_ser)
    controls_stress = {"C_BIL": common.candidate_book_blocks(taa_ser, l_stress_ser, bil_ser)}
    c_bil, c_bil_stress = controls["C_BIL"], controls_stress["C_BIL"]
    control_series_dict = {
        "C_BIL": common.candidate_book_series(taa_ser, l_engine_ser, bil_ser, "G-LONG"),
        "C_SPY": common.candidate_book_series(taa_ser, l_engine_ser, spy_ser, "G-LONG"),
        "G3": common.book_window_return_ser({"taa": taa_ser, "L": l_engine_ser}, common.G3_WEIGHT_DICT, *common.BOOK_BLOCK_DICT["G-LONG"]),
        "dv2_slot": common.candidate_book_series(taa_ser, l_engine_ser, dv2_ser, "G-LONG"),
    }
    control_post_dict = {name_str: common.metric_dict(common.book_window_return_ser({"taa": taa_ser, "L": l_engine_ser, "X": x_ser}, common.CANDIDATE_WEIGHT_DICT, *POST_BLOCK_TUPLE))
                         for name_str, x_ser in (("C_BIL", bil_ser), ("C_SPY", spy_ser), ("dv2_slot", dv2_ser))}
    control_post_dict["G3"] = common.metric_dict(common.book_window_return_ser({"taa": taa_ser, "L": l_engine_ser}, common.G3_WEIGHT_DICT, *POST_BLOCK_TUPLE))

    # ------------------------------------------------------------------ pod returns and per-cell metrics
    frames = {(cost_str, sweep_bool): run_pods.load_returns("U1", cost_str, sweep_bool, bil_ser) for cost_str in ("engine", "stress") for sweep_bool in (True, False)}
    u2_frames = {(cost_str, sweep_bool): run_pods.load_returns("U2", cost_str, sweep_bool, bil_ser) for cost_str in ("engine", "stress") for sweep_bool in (True, False)}
    meta_u1 = load_meta("U1")
    meta_u2 = load_meta("U2")
    cell_res: dict = {}
    book_g_full_dict: dict[str, pd.Series] = {}
    book_g_long_dict: dict[str, pd.Series] = {}
    sweep_engine_df = frames[("engine", True)]
    for key_str in sweep_engine_df.columns:
        x_ser = sweep_engine_df[key_str]
        block_dict = blocks_for(key_str)
        full_start_str, full_end_str = block_dict["FULL"]
        entry = {
            "standalone_sweep": common.window_metrics(x_ser, block_dict),
            "standalone_nosweep": common.window_metrics(frames[("engine", False)][key_str], block_dict),
            "standalone_sweep_stress": common.window_metrics(frames[("stress", True)][key_str], block_dict) if frames[("stress", True)] is not None and key_str in frames[("stress", True)] else None,
            "standalone_nosweep_stress": common.window_metrics(frames[("stress", False)][key_str], block_dict) if frames[("stress", False)] is not None and key_str in frames[("stress", False)] else None,
            "standalone_sweep_POST": common.metric_dict(x_ser.loc[POST_BLOCK_TUPLE[0]:POST_BLOCK_TUPLE[1]]),
            "beta_spy": {b: beta_to(x_ser, spy_ser, s, e) for b, (s, e) in block_dict.items()},
            "alpha_vs_market": {b: alpha_vs_market(x_ser, spy_ser, s, e) for b, (s, e) in block_dict.items()},
            "corr_taa_2008_2026": corr_from(x_ser, taa_ser, BOOK_START_STR),
            "corr_L_full": corr_from(x_ser, l_engine_ser, full_start_str),
            "corr_dv2_2008_2026": corr_from(x_ser, dv2_ser, BOOK_START_STR),
            "corr_bil_2007_2026": corr_from(x_ser, bil_ser, "2007-05-31"),
            "meta_engine": {k: meta_u1.get(key_str, {}).get("engine", {}).get(k) for k in ("turnover_one_way_x_per_year", "holding_sessions_mean", "holding_sessions_median", "mean_positions", "mean_exposure",
                                                                                          "commission_pct_nav_per_year", "slippage_pct_nav_per_year", "cost_share_of_gross_pnl", "round_trips_per_year",
                                                                                          "terminal_liquidations_int", "phantom_fill_int", "exits_by_reason", "final_nav_usd")},
            "meta_stress": {k: meta_u1.get(key_str, {}).get("stress", {}).get(k) for k in ("commission_pct_nav_per_year", "slippage_pct_nav_per_year", "cost_share_of_gross_pnl", "final_nav_usd")},
            "capacity_2021_2026": meta_u1.get(key_str, {}).get("engine", {}).get("capacity_2021_2026", {}),
            "capacity_full": meta_u1.get(key_str, {}).get("engine", {}).get("capacity_full", {}),
            "books": {
                "sweep_engine": common.candidate_book_blocks(taa_ser, l_engine_ser, x_ser),
                "sweep_stress": common.candidate_book_blocks(taa_ser, l_stress_ser, frames[("stress", True)][key_str]) if frames[("stress", True)] is not None and key_str in frames[("stress", True)] else None,
                "nosweep_engine": common.candidate_book_blocks(taa_ser, l_engine_ser, frames[("engine", False)][key_str]),
            },
            "book_POST_sweep_engine": common.metric_dict(common.book_window_return_ser({"taa": taa_ser, "L": l_engine_ser, "X": x_ser}, common.CANDIDATE_WEIGHT_DICT, *POST_BLOCK_TUPLE)),
            "u2": None,
        }
        if u2_frames[("engine", True)] is not None and key_str in u2_frames[("engine", True)]:
            u2_ser = u2_frames[("engine", True)][key_str]
            entry["u2"] = {"standalone_sweep": common.window_metrics(u2_ser, block_dict), "books_sweep_engine": common.candidate_book_blocks(taa_ser, l_engine_ser, u2_ser),
                           "books_sweep_stress": common.candidate_book_blocks(taa_ser, l_stress_ser, u2_frames[("stress", True)][key_str]) if u2_frames[("stress", True)] is not None and key_str in u2_frames[("stress", True)] else None,
                           "meta_engine": {k: meta_u2.get(key_str, {}).get("engine", {}).get(k) for k in ("turnover_one_way_x_per_year", "holding_sessions_mean", "mean_positions", "cost_share_of_gross_pnl", "final_nav_usd")},
                           "capacity_2021_2026": meta_u2.get(key_str, {}).get("engine", {}).get("capacity_2021_2026", {})}
        book_g_full_dict[key_str] = common.candidate_book_series(taa_ser, l_engine_ser, x_ser, "G-FULL")
        book_g_long_dict[key_str] = common.candidate_book_series(taa_ser, l_engine_ser, x_ser, "G-LONG")
        cell_res[key_str] = entry
    grid_key_list = [k for k in run_pods.grid_keys() if k in cell_res]

    # ------------------------------------------------------------------ candidates and the rule
    candidate_dict = run_pods.load_candidates()
    candidate_list: list[dict] = []
    for composite_str in common.COMPOSITE_TUPLE:
        long_entry = candidate_dict["long_only"][composite_str]
        candidate_list.append({"name": f"{composite_str}_LONG", "composite": composite_str, "form": "LONG", "centre": long_entry["centre"], "neighbourhood": [k for k in long_entry["neighbourhood"] if k in cell_res],
                               "plateau_full_sharpe_sweep": long_entry["plateau_value"], "grid_pos": long_entry["grid_pos"]})
        hedged_entry = candidate_dict["hedged"][composite_str]
        if hedged_entry["centre"] in cell_res:
            candidate_list.append({"name": f"{composite_str}_HEDGED", "composite": composite_str, "form": "HEDGED", "centre": hedged_entry["centre"], "neighbourhood": [hedged_entry["centre"]],
                                   "plateau_full_sharpe_sweep": cell_res[hedged_entry["centre"]]["standalone_sweep"]["FULL"]["sharpe"], "grid_pos": None, "built_on": hedged_entry["built_on"]})
    label_pod_keys = {"REV1": common.cell_key("REV1", common.LABEL_N_INT, common.LABEL_B_INT), "REV5": common.cell_key("REV5", common.LABEL_N_INT, common.LABEL_B_INT)}
    noind_key = next((k for k in cell_res if k.startswith("C_EQ_noInd|")), None)
    for c in candidate_list:
        neigh = c["neighbourhood"]
        centre = c["centre"]
        engine_rule = margins(cell_res, neigh, "sweep_engine", c_bil)
        stress_rule = margins(cell_res, neigh, "sweep_stress", c_bil_stress)
        centre_rule = margins(cell_res, [centre], "sweep_engine", c_bil)
        u2_entry = cell_res[centre].get("u2")
        u2_sharpe = u2_entry["books_sweep_engine"]["G-FULL"]["sharpe"] if u2_entry else float("nan")
        r4_dict = {"u2_book_g_full_sharpe": u2_sharpe, "C_BIL_g_full_sharpe": c_bil["G-FULL"]["sharpe"], "pass": bool(np.isfinite(u2_sharpe) and u2_sharpe > c_bil["G-FULL"]["sharpe"]),
                   "u2_book_blocks": u2_entry["books_sweep_engine"] if u2_entry else None, "u2_standalone_sweep": u2_entry["standalone_sweep"] if u2_entry else None, "available": u2_entry is not None}
        capacity = cell_res[centre]["capacity_2021_2026"]
        r5_dict = {"aum_max_5pct_usd_2021_2026": capacity.get("aum_max_5pct_usd"), "aum_p95_5pct_usd_2021_2026": capacity.get("aum_p95_5pct_usd"), "aum_p95_1pct_usd_2021_2026": capacity.get("aum_p95_1pct_usd"),
                   "largest_order_share_of_adv20_at_100k": capacity.get("order_share_of_adv20_at_capital_max"), "p95_order_share_of_adv20_at_100k": capacity.get("order_share_of_adv20_at_capital_p95"),
                   "pass": bool((capacity.get("aum_max_5pct_usd") or 0.0) >= common.CAPACITY_MIN_AUM_FLOAT)}
        centre_book = cell_res[centre]["books"]["sweep_engine"]
        c["rule"] = {
            "engine": engine_rule, "stress": stress_rule, "centre_cell_engine": centre_rule,
            "R1": engine_rule["R1"], "R2": engine_rule["R2"], "R3": bool(stress_rule["R1"] and stress_rule["R2"]), "R4": r4_dict, "R5": r5_dict,
            "passes": bool(engine_rule["R1"] and engine_rule["R2"] and stress_rule["R1"] and stress_rule["R2"] and r4_dict["pass"] and r5_dict["pass"]),
        }
        c["labels"] = {
            "vs_G3": margins(cell_res, neigh, "sweep_engine", controls["G3"]),
            "vs_C_SPY": margins(cell_res, neigh, "sweep_engine", controls["C_SPY"]),
            "vs_dv2_slot": margins(cell_res, neigh, "sweep_engine", controls["dv2_slot"]),
            "nosweep_vs_C_CASH0": margins(cell_res, neigh, "nosweep_engine", controls["C_CASH0"]),
            "owner_gate_centre": {"sharpe": centre_book["G-FULL"]["sharpe"], "max_dd": centre_book["G-FULL"]["max_dd"], "cagr": centre_book["G-FULL"]["cagr"],
                                  "pass": bool(centre_book["G-FULL"]["sharpe"] >= common.OWNER_GATE_SHARPE_FLOAT and centre_book["G-FULL"]["max_dd"] >= common.OWNER_GATE_MAX_DD_FLOAT)},
            "owner_gate_neighbourhood_median": {"sharpe": engine_rule["neigh_median_sharpe"]["G-FULL"], "max_dd": engine_rule["neigh_median_max_dd"]["G-FULL"],
                                                "pass": bool(engine_rule["neigh_median_sharpe"]["G-FULL"] >= common.OWNER_GATE_SHARPE_FLOAT and engine_rule["neigh_median_max_dd"]["G-FULL"] >= common.OWNER_GATE_MAX_DD_FLOAT)},
            "POST": {"standalone_sweep": cell_res[centre]["standalone_sweep_POST"], "book_sweep_engine": cell_res[centre]["book_POST_sweep_engine"], "controls_book": control_post_dict,
                     "beats_C_BIL_book": bool(cell_res[centre]["book_POST_sweep_engine"]["sharpe"] > control_post_dict["C_BIL"]["sharpe"])},
        }
        c["centre_standalone_sweep"] = cell_res[centre]["standalone_sweep"]
        c["centre_standalone_nosweep"] = cell_res[centre]["standalone_nosweep"]
        c["centre_standalone_sweep_stress"] = cell_res[centre]["standalone_sweep_stress"]
        c["centre_books_sweep_engine"] = centre_book
        c["centre_books_sweep_stress"] = cell_res[centre]["books"]["sweep_stress"]
        c["centre_alpha_vs_market"] = cell_res[centre]["alpha_vs_market"]
        c["centre_meta_engine"] = cell_res[centre]["meta_engine"]
        c["centre_correlations"] = {k: cell_res[centre][k] for k in ("corr_taa_2008_2026", "corr_L_full", "corr_dv2_2008_2026", "corr_bil_2007_2026")}
        c["centre_beta_spy"] = cell_res[centre]["beta_spy"]
        c["neigh_median_standalone_sweep"] = {b: {m: neigh_median(neigh, lambda k: cell_res[k]["standalone_sweep"][b][m]) for m in ("cagr", "sharpe", "max_dd")} for b in blocks_for(centre)}
        # walk-forward: re-chosen on P1 standalone Sharpe (long-only grid only)
        if c["form"] == "LONG":
            wf_entry = candidate_dict["walk_forward_P1"][c["composite"]]
            wf_neigh = [k for k in wf_entry["neighbourhood"] if k in cell_res]
            c["walk_forward"] = {"centre_by_P1": wf_entry["centre"], "plateau_P1_sharpe": wf_entry["plateau_value"],
                                 "standalone_sweep_neigh_median_sharpe": {b: neigh_median(wf_neigh, lambda k: cell_res[k]["standalone_sweep"][b]["sharpe"]) for b in ("P2", "P3")},
                                 "standalone_sweep_centre_sharpe": {b: cell_res[wf_entry["centre"]]["standalone_sweep"][b]["sharpe"] for b in ("P2", "P3")},
                                 "book_neigh_median_sharpe": {b: neigh_median(wf_neigh, lambda k: cell_res[k]["books"]["sweep_engine"][b]["sharpe"]) for b in ("G-P2", "G-P3")},
                                 "C_BIL": {b: c_bil[b]["sharpe"] for b in ("G-P2", "G-P3")}, "BIL_standalone": {b: common.metric_dict(bil_ser.loc[common.BLOCK_DICT[b][0]:common.BLOCK_DICT[b][1]])["sharpe"] for b in ("P2", "P3")}}
    label_pods = {name_str: {"key": key_str, "standalone_sweep": cell_res[key_str]["standalone_sweep"], "books_sweep_engine": cell_res[key_str]["books"]["sweep_engine"],
                             "margin_vs_C_BIL": {b: cell_res[key_str]["books"]["sweep_engine"][b]["sharpe"] - c_bil[b]["sharpe"] for b in common.BOOK_BLOCK_DICT}, "meta_engine": cell_res[key_str]["meta_engine"]}
                  for name_str, key_str in label_pod_keys.items() if key_str in cell_res}
    if noind_key is not None:
        label_pods["C_EQ_noInd"] = {"key": noind_key, "standalone_sweep": cell_res[noind_key]["standalone_sweep"], "books_sweep_engine": cell_res[noind_key]["books"]["sweep_engine"],
                                    "margin_vs_C_BIL": {b: cell_res[noind_key]["books"]["sweep_engine"][b]["sharpe"] - c_bil[b]["sharpe"] for b in common.BOOK_BLOCK_DICT}, "meta_engine": cell_res[noind_key]["meta_engine"],
                                    "compared_with": candidate_dict["long_only"]["C_EQ"]["centre"],
                                    "delta_vs_C_EQ_centre": {b: cell_res[noind_key]["books"]["sweep_engine"][b]["sharpe"] - cell_res[candidate_dict["long_only"]["C_EQ"]["centre"]]["books"]["sweep_engine"][b]["sharpe"] for b in common.BOOK_BLOCK_DICT}}
    passing_list = [c for c in candidate_list if c["rule"]["passes"]]
    ranked_by_margin = sorted(candidate_list, key=lambda c: c["rule"]["engine"]["min_margin"], reverse=True)
    decision_dict = {
        "prereg": common.PREREG_REL_PATH, "end_ts": str(common.END_TS.date()),
        "rule_convention": "R1-R3 on the neighbourhood median of the clipped 3x3 (N, B) box around the centre (long-only), the cell itself (HEDGED); the centre-cell values are reported alongside",
        "verdict": "recommend a forward paper line for the passing candidate(s)" if passing_list else "no candidate passes R1-R5; no forward paper line; Stage A stays descriptive",
        "passing": [c["name"] for c in passing_list],
        "best_min_margin_candidate": {"name": ranked_by_margin[0]["name"], "centre": ranked_by_margin[0]["centre"], "min_margin": ranked_by_margin[0]["rule"]["engine"]["min_margin"]} if ranked_by_margin else None,
        "controls": {"C_BIL": c_bil, "C_BIL_stress": c_bil_stress, "C_SPY": controls["C_SPY"], "G3": controls["G3"], "dv2_slot": controls["dv2_slot"], "C_CASH0": controls["C_CASH0"], "POST": control_post_dict},
        "candidates": candidate_list, "label_pods": label_pods,
    }
    common.write_json("decision.json", decision_dict)

    # ------------------------------------------------------------------ confidence
    book_full_df = pd.DataFrame({**{k: book_g_full_dict[k] for k in grid_key_list + [c["centre"] for c in candidate_list if c["form"] == "HEDGED"]}, "C_BIL": common.candidate_book_series(taa_ser, l_engine_ser, bil_ser, "G-FULL")}).dropna()
    candidate_keys = [c["centre"] for c in candidate_list]
    confidence_dict = {"reality_check_book_g_full_vs_C_BIL": trend_analyze.reality_check(book_full_df, "C_BIL", candidate_keys, common.SEED_INT),
                       "reality_check_note": f"{len(book_full_df.columns) - 1} configurations (24 long-only cells + HEDGED cells), stationary bootstrap block {common.BOOT_BLOCK_FLOAT:.0f}, {common.BOOT_N_INT} draws, seed {common.SEED_INT}",
                       "deflated_sharpe_excess_of_bil": {}, "walk_forward": {c["name"]: c.get("walk_forward") for c in candidate_list if c["form"] == "LONG"}}
    bil_full_ser = bil_ser.reindex(sweep_engine_df.index).fillna(0.0)
    excess_df = sweep_engine_df.sub(bil_full_ser, axis=0)
    for c in candidate_list:
        start_str, end_str = blocks_for(c["centre"])["FULL"]
        family_keys = [k for k in grid_key_list + [cc["centre"] for cc in candidate_list if cc["form"] == "HEDGED"] if k in excess_df]
        family_df = excess_df[family_keys].loc[start_str:end_str]
        family_sharpe_vec = (family_df.mean() / family_df.std()).to_numpy()
        dsr = trend_analyze.deflated_sharpe(excess_df[c["centre"]].loc[start_str:end_str], family_sharpe_vec, common.N_TRIALS_INT)
        dsr["window"] = [start_str, end_str]
        dsr["note"] = "excess of BIL total return (0 before BIL's first return on 2007-05-31)"
        dsr_2007 = trend_analyze.deflated_sharpe(excess_df[c["centre"]].loc["2007-05-31":end_str], (excess_df[family_keys].loc["2007-05-31":end_str].mean() / excess_df[family_keys].loc["2007-05-31":end_str].std()).to_numpy(), common.N_TRIALS_INT)
        confidence_dict["deflated_sharpe_excess_of_bil"][c["name"]] = {"full": dsr, "from_2007_05_31": dsr_2007}
    common.write_json("confidence.json", confidence_dict)

    # ------------------------------------------------------------------ capacity and small accounts
    capacity_dict: dict = {"capital_base_usd": common.CAPITAL_BASE_FLOAT, "candidates": {}, "small_accounts": {}}
    for c in candidate_list:
        capacity_dict["candidates"][c["name"]] = {"centre": c["centre"], "capacity_2021_2026": cell_res[c["centre"]]["capacity_2021_2026"], "capacity_full": cell_res[c["centre"]]["capacity_full"],
                                                  "R5_pass": c["rule"]["R5"]["pass"], "u2_capacity_2021_2026": (cell_res[c["centre"]].get("u2") or {}).get("capacity_2021_2026")}
    for capital_float in common.SMALL_ACCOUNT_CAPITAL_TUPLE:
        tag_str = f"U1_cap{int(capital_float)}"
        cap_sweep_df = run_pods.load_returns(tag_str, "engine", True, bil_ser)
        cap_meta = load_meta(tag_str)
        if cap_sweep_df is None:
            continue
        for c in candidate_list:
            key_str = c["centre"]
            if key_str not in cap_sweep_df:
                continue
            x_ser = cap_sweep_df[key_str]
            block_dict = blocks_for(key_str)
            capacity_dict["small_accounts"].setdefault(c["name"], {})[f"{int(capital_float)}"] = {
                "standalone_sweep": common.window_metrics(x_ser, block_dict), "books_sweep_engine": common.candidate_book_blocks(taa_ser, l_engine_ser, x_ser),
                "book_g_full_margin_vs_C_BIL": common.candidate_book_blocks(taa_ser, l_engine_ser, x_ser)["G-FULL"]["sharpe"] - c_bil["G-FULL"]["sharpe"],
                "meta_engine": {k: cap_meta.get(key_str, {}).get("engine", {}).get(k) for k in ("commission_pct_nav_per_year", "slippage_pct_nav_per_year", "cost_share_of_gross_pnl", "mean_positions", "turnover_one_way_x_per_year", "final_nav_usd", "phantom_fill_int")},
                "zero_share_skips_int": cap_meta.get(key_str, {}).get("engine", {}).get("policy", {}).get("zero_share_skips_int"),
            }
        for c in candidate_list:
            capacity_dict["small_accounts"].setdefault(c["name"], {})["100000"] = {"standalone_sweep": c["centre_standalone_sweep"], "books_sweep_engine": c["centre_books_sweep_engine"],
                                                                                   "book_g_full_margin_vs_C_BIL": c["centre_books_sweep_engine"]["G-FULL"]["sharpe"] - c_bil["G-FULL"]["sharpe"], "meta_engine": c["centre_meta_engine"]}
    common.write_json("capacity_small_account.json", capacity_dict)

    # ------------------------------------------------------------------ grid results (json + flat csv)
    grid_dict = {"cells": cell_res, "candidates_json": candidate_dict}
    common.write_json("grid_results.json", grid_dict)
    row_list = []
    for key_str, entry in cell_res.items():
        row = {"key": key_str, **common.parse_cell_key(key_str)}
        for b, m in entry["standalone_sweep"].items():
            row.update({f"sa_sweep_{b}_sharpe": m["sharpe"], f"sa_sweep_{b}_cagr": m["cagr"], f"sa_sweep_{b}_dd": m["max_dd"]})
        row["sa_sweep_POST_sharpe"] = entry["standalone_sweep_POST"]["sharpe"]
        row["sa_nosweep_FULL_sharpe"] = entry["standalone_nosweep"]["FULL"]["sharpe"]
        row["sa_sweep_stress_FULL_sharpe"] = entry["standalone_sweep_stress"]["FULL"]["sharpe"] if entry["standalone_sweep_stress"] else None
        for b, m in entry["books"]["sweep_engine"].items():
            row.update({f"book_{b}_sharpe": m["sharpe"], f"book_{b}_dd": m["max_dd"], f"book_{b}_margin_vs_CBIL": m["sharpe"] - c_bil[b]["sharpe"]})
        if entry["books"]["sweep_stress"]:
            row["book_stress_G-FULL_sharpe"] = entry["books"]["sweep_stress"]["G-FULL"]["sharpe"]
            row["book_stress_min_margin_vs_CBIL"] = min(entry["books"]["sweep_stress"][b]["sharpe"] - c_bil_stress[b]["sharpe"] for b in common.RULE_BLOCK_TUPLE)
        row["book_min_margin_vs_CBIL"] = min(entry["books"]["sweep_engine"][b]["sharpe"] - c_bil[b]["sharpe"] for b in common.RULE_BLOCK_TUPLE)
        row["book_POST_sharpe"] = entry["book_POST_sweep_engine"]["sharpe"]
        row.update({f"beta_spy_FULL": entry["beta_spy"]["FULL"], "corr_taa": entry["corr_taa_2008_2026"], "corr_L": entry["corr_L_full"], "corr_dv2": entry["corr_dv2_2008_2026"]})
        row.update({k: v for k, v in entry["meta_engine"].items() if k != "exits_by_reason"})
        row.update({"cap21_max_5pct_usd": entry["capacity_2021_2026"].get("aum_max_5pct_usd"), "cap21_p95_1pct_usd": entry["capacity_2021_2026"].get("aum_p95_1pct_usd")})
        row["alpha_vs_spy_FULL_ann"] = entry["alpha_vs_market"]["FULL"]["alpha_ann"]
        row["alpha_vs_spy_FULL_t"] = entry["alpha_vs_market"]["FULL"]["t_nw5"]
        if entry["u2"]:
            row["u2_book_G-FULL_sharpe"] = entry["u2"]["books_sweep_engine"]["G-FULL"]["sharpe"]
            row["u2_sa_FULL_sharpe"] = entry["u2"]["standalone_sweep"]["FULL"]["sharpe"]
        row_list.append(row)
    pd.DataFrame(row_list).to_csv(OUT_PATH / "grid_results.csv", index=False)

    # ------------------------------------------------------------------ daily return series
    daily_df = pd.DataFrame({**{f"{c['name']}|sweep_engine": sweep_engine_df[c["centre"]] for c in candidate_list},
                             **{f"{c['name']}|sweep_stress": frames[("stress", True)][c["centre"]] for c in candidate_list if frames[("stress", True)] is not None},
                             **{f"{c['name']}|nosweep_engine": frames[("engine", False)][c["centre"]] for c in candidate_list},
                             **{f"{c['name']}|book_G-LONG": book_g_long_dict[c["centre"]] for c in candidate_list},
                             **{f"{name_str}|sweep_engine": sweep_engine_df[entry["key"]] for name_str, entry in label_pods.items()},
                             **{f"{name_str}|book_G-LONG": ser for name_str, ser in control_series_dict.items()},
                             "BIL": bil_ser, "SPY_TR": spy_ser, "L": l_engine_ser, "TAA": taa_ser, "dv2": dv2_ser})
    daily_df.loc[:common.END_TS].to_parquet(OUT_PATH / "daily_returns_candidates_controls.parquet")

    # ------------------------------------------------------------------ charts
    for composite_str in common.COMPOSITE_TUPLE:
        grid = candidate_dict["long_only"][composite_str]
        key_arr = np.array(grid["keys"], dtype=object)
        value_arr = np.array(grid["values"], dtype=float)
        book_arr = np.vectorize(lambda k: cell_res[k]["books"]["sweep_engine"]["G-FULL"]["sharpe"] - c_bil["G-FULL"]["sharpe"] if k in cell_res else np.nan, otypes=[float])(key_arr)
        margin_arr = np.vectorize(lambda k: min(cell_res[k]["books"]["sweep_engine"][b]["sharpe"] - c_bil[b]["sharpe"] for b in common.RULE_BLOCK_TUPLE) if k in cell_res else np.nan, otypes=[float])(key_arr)
        fig, axes = plt.subplots(1, 3, figsize=(15, 3.4), facecolor=trend_analyze.SURFACE)
        row_labels = [f"N {n}" for n in grid["rows_N"]]
        col_labels = [f"B {b}" for b in grid["cols_B"]]
        trend_analyze.heatmap(axes[0], value_arr, row_labels, col_labels, "Standalone Sharpe 2000-26 (with sweep, engine costs)", seq_cmap, np.nanmin(value_arr), np.nanmax(value_arr), tuple(grid["a0_pos"]), tuple(grid["grid_pos"]))
        trend_analyze.heatmap(axes[1], book_arr, row_labels, col_labels, f"Book G-FULL Sharpe minus C_BIL ({c_bil['G-FULL']['sharpe']:.3f})", div_cmap, -0.15, 0.15, tuple(grid["a0_pos"]), tuple(grid["grid_pos"]), fmt="{:+.3f}")
        trend_analyze.heatmap(axes[2], margin_arr, row_labels, col_labels, "Worst block margin vs C_BIL (G-P1, G-P2, G-P3)", div_cmap, -0.3, 0.3, tuple(grid["a0_pos"]), tuple(grid["grid_pos"]), fmt="{:+.3f}")
        fig.suptitle(f"{composite_str}: black box = A0 (C_EQ N20 B2 position), dashed = plateau centre; rows N, columns B", fontsize=10, color=trend_analyze.TEXT_SECONDARY, x=0.01, ha="left")
        fig.tight_layout()
        fig.savefig(CHART_PATH / f"heatmap_{composite_str}.png", dpi=130)
        plt.close(fig)
    palette = ["#0b0b0b", "#52514e", "#9ec5f4", "#8f2423", "#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
    fig, axes = plt.subplots(2, 1, figsize=(14, 7.5), facecolor=trend_analyze.SURFACE, sharex=True)
    series_list = [("C_BIL", control_series_dict["C_BIL"]), ("G3", control_series_dict["G3"]), ("C_SPY", control_series_dict["C_SPY"]), ("dv2 slot", control_series_dict["dv2_slot"])] + [(f"{c['name']}: {c['centre']}", book_g_long_dict[c["centre"]]) for c in candidate_list]
    for (label_str, ser), color in zip(series_list, palette):
        wealth_ser = (1 + ser.dropna()).cumprod()
        axes[0].plot(wealth_ser.index, wealth_ser, color=color, lw=1.6 if label_str in ("C_BIL", "G3") else 1.1, label=label_str)
        axes[1].plot(wealth_ser.index, wealth_ser / wealth_ser.cummax() - 1, color=color, lw=1.6 if label_str in ("C_BIL", "G3") else 1.1)
    axes[0].set_yscale("log")
    axes[0].set_title("Candidate books {TAA 0.5, L 0.25, X 0.25} versus C_BIL, G3, C_SPY and the dv2 slot, 2008-03-04..2026-08-19 (official pod model, annual reset)", fontsize=9.5, loc="left", color=trend_analyze.TEXT_PRIMARY)
    axes[1].set_title("Drawdown", fontsize=9.5, loc="left", color=trend_analyze.TEXT_PRIMARY)
    axes[0].legend(fontsize=7, frameon=False, ncol=2)
    for ax in axes:
        trend_analyze.style_axis(ax)
    fig.tight_layout()
    fig.savefig(CHART_PATH / "equity_dd_candidate_books.png", dpi=130)
    plt.close(fig)
    fig, axes = plt.subplots(2, 1, figsize=(14, 7.5), facecolor=trend_analyze.SURFACE, sharex=True)
    series_list = [("BIL total return", bil_ser.loc["2000-01-03":common.END_TS]), ("SPY total return", spy_ser.loc["2000-01-03":common.END_TS])] + [(f"{c['name']}: {c['centre']} (sweep, engine)", sweep_engine_df[c["centre"]].loc[blocks_for(c["centre"])["FULL"][0]:]) for c in candidate_list] + [(f"{n}: {e['key']}", sweep_engine_df[e["key"]]) for n, e in label_pods.items()]
    for (label_str, ser), color in zip(series_list, palette):
        wealth_ser = (1 + ser.dropna()).cumprod()
        axes[0].plot(wealth_ser.index, wealth_ser, color=color, lw=1.6 if label_str.startswith(("BIL", "SPY")) else 1.1, label=label_str)
        axes[1].plot(wealth_ser.index, wealth_ser / wealth_ser.cummax() - 1, color=color, lw=1.6 if label_str.startswith(("BIL", "SPY")) else 1.1)
    axes[0].set_yscale("log")
    axes[0].set_title("Standalone pods with the idle-cash sweep (engine costs) versus BIL and SPY total return, 2000-01-03..2026-08-19", fontsize=9.5, loc="left", color=trend_analyze.TEXT_PRIMARY)
    axes[1].set_title("Drawdown", fontsize=9.5, loc="left", color=trend_analyze.TEXT_PRIMARY)
    axes[0].legend(fontsize=7, frameon=False, ncol=2)
    for ax in axes:
        trend_analyze.style_axis(ax)
    fig.tight_layout()
    fig.savefig(CHART_PATH / "equity_dd_candidates_standalone.png", dpi=130)
    plt.close(fig)
    stage_a_path = OUT_PATH / "stageA_alpha_table.csv"
    if stage_a_path.exists():
        table_df = pd.read_csv(stage_a_path)
        alpha_df = table_df[table_df["is_alpha"]].copy()
        alpha_df["alpha"] = alpha_df["signal"].astype(int)
        fig, axes = plt.subplots(2, 1, figsize=(15, 7), facecolor=trend_analyze.SURFACE, sharex=True)
        block_colors = {"P1": "#2a78d6", "P2": "#eb6834", "P3": "#1baf7a", "POST": "#52514e"}
        for block_str, color in block_colors.items():
            block_df = alpha_df[alpha_df["block"] == block_str].sort_values("alpha")
            axes[0].scatter(block_df["alpha"], block_df["ic_mean"], s=14, color=color, label=f"{block_str} (median {block_df['ic_mean'].median():+.4f})", alpha=0.85)
            axes[1].scatter(block_df["alpha"], block_df["c_star_bps"].clip(-30, 30), s=14, color=color, label=f"{block_str} (median {block_df['c_star_bps'].median():+.1f} bps)", alpha=0.85)
        axes[0].axhline(0, color=trend_analyze.TEXT_SECONDARY, lw=0.8, ls=":")
        axes[1].axhline(0, color=trend_analyze.TEXT_SECONDARY, lw=0.8, ls=":")
        axes[1].axhline(3, color="#8f2423", lw=0.8, ls="--")
        axes[1].axhline(8, color="#8f2423", lw=0.8, ls=":")
        axes[0].set_title("Stage A: mean daily Spearman IC (next-open to next-next-open return) per alpha and block, U1 = PIT S&P 500", fontsize=9.5, loc="left", color=trend_analyze.TEXT_PRIMARY)
        axes[1].set_title("Break-even cost per side c* (bps, clipped to +-30); dashed 3 bps engine, dotted 8 bps stress", fontsize=9.5, loc="left", color=trend_analyze.TEXT_PRIMARY)
        axes[1].set_xlabel("alpha number (paper Appendix A; 56 excluded)", fontsize=8, color=trend_analyze.TEXT_SECONDARY)
        axes[1].set_xticks(range(0, 102, 5))
        for ax in axes:
            ax.legend(fontsize=7, frameon=False, ncol=4)
            trend_analyze.style_axis(ax)
        fig.tight_layout()
        fig.savefig(CHART_PATH / "stageA_ic_by_block.png", dpi=130)
        plt.close(fig)

    # ------------------------------------------------------------------ summary for the lead
    family_summary = common.read_json("stageA_family_summary.json", {}) or {}
    composites_summary = common.read_json("stageA_composites.json", {}) or {}
    tests_and_checks = common.read_json("tests_and_checks.json", {}) or {}
    summary_dict = {
        "prereg": common.PREREG_REL_PATH, "verdict": decision_dict["verdict"], "passing": decision_dict["passing"],
        "checks": {k: (v.get("passed_bool", v.get("passed_strict_bool")) if isinstance(v, dict) else v) for k, v in tests_and_checks.items() if k != "updated_at"},
        "controls_by_block": {name_str: {b: {"sharpe": m["sharpe"], "max_dd": m["max_dd"], "cagr": m["cagr"]} for b, m in blocks.items()} for name_str, blocks in decision_dict["controls"].items() if name_str != "POST"},
        "controls_POST": control_post_dict,
        "stage_a_family_by_block": {b: {k: d.get(k) for k in ("median_ic", "median_c_star_bps", "median_ls_sharpe", "median_net_sharpe_3bps", "median_net_sharpe_8bps", "count_ic_t_above_2", "count_ic_t_below_minus_2", "count_c_star_above_3bps", "median_tau")} for b, d in family_summary.get("blocks", {}).items()},
        "stage_a_paper_window_mean_pairwise_ls_correlation": family_summary.get("paper_window_mean_pairwise_ls_correlation"),
        "composites_by_block": {name_str: {b: {k: d.get(k) for k in ("ic_mean", "ic_t_nw5", "ic_positive_share", "c_star_bps", "ls_sharpe", "net_sharpe_3bps", "net_sharpe_8bps", "tau_mean", "lo_mean_ann", "lo_c_star_bps", "ic_overnight_mean")} for b, d in blocks.items()} for name_str, blocks in composites_summary.items()},
        "candidates": {c["name"]: {
            "centre": c["centre"], "plateau_full_sharpe_sweep": c["plateau_full_sharpe_sweep"],
            "standalone_sweep_by_block": {b: {"sharpe": m["sharpe"], "cagr": m["cagr"], "max_dd": m["max_dd"]} for b, m in c["centre_standalone_sweep"].items()},
            "standalone_sweep_POST": c["labels"]["POST"]["standalone_sweep"],
            "standalone_nosweep_FULL": c["centre_standalone_nosweep"]["FULL"],
            "standalone_sweep_stress_FULL": c["centre_standalone_sweep_stress"]["FULL"] if c["centre_standalone_sweep_stress"] else None,
            "book_sweep_engine_by_block": {b: {"sharpe": m["sharpe"], "max_dd": m["max_dd"], "cagr": m["cagr"], "margin_vs_C_BIL": m["sharpe"] - c_bil[b]["sharpe"], "margin_vs_G3": m["sharpe"] - controls["G3"][b]["sharpe"],
                                               "margin_vs_C_SPY": m["sharpe"] - controls["C_SPY"][b]["sharpe"], "margin_vs_dv2_slot": m["sharpe"] - controls["dv2_slot"][b]["sharpe"]} for b, m in c["centre_books_sweep_engine"].items()},
            "book_POST": c["labels"]["POST"]["book_sweep_engine"],
            "neighbourhood_median_book_sharpe": c["rule"]["engine"]["neigh_median_sharpe"],
            "rule": {"R1": c["rule"]["R1"], "R2": c["rule"]["R2"], "R3": c["rule"]["R3"], "R4": c["rule"]["R4"]["pass"], "R5": c["rule"]["R5"]["pass"], "passes": c["rule"]["passes"],
                     "sharpe_margin_neigh": c["rule"]["engine"]["sharpe_margin"], "dd_gap_pp_neigh": c["rule"]["engine"]["dd_gap_pp"], "stress_sharpe_margin_neigh": c["rule"]["stress"]["sharpe_margin"],
                     "u2_book_g_full_sharpe": c["rule"]["R4"]["u2_book_g_full_sharpe"], "aum_max_5pct_usd_2021_2026": c["rule"]["R5"]["aum_max_5pct_usd_2021_2026"]},
            "labels": {"beats_G3_all_rule_blocks": c["labels"]["vs_G3"]["R1"], "beats_C_SPY_all_rule_blocks": c["labels"]["vs_C_SPY"]["R1"], "beats_dv2_slot_all_rule_blocks": c["labels"]["vs_dv2_slot"]["R1"],
                       "owner_gate_centre": c["labels"]["owner_gate_centre"]["pass"], "POST_beats_C_BIL_book": c["labels"]["POST"]["beats_C_BIL_book"]},
            "alpha_vs_spy_by_block": {b: {"alpha_ann": d["alpha_ann"], "t_nw5": d["t_nw5"], "beta": d["beta"]} for b, d in c["centre_alpha_vs_market"].items()},
            "correlations": c["centre_correlations"], "meta_engine": c["centre_meta_engine"],
            "costs": {"commission_pct_nav_per_year": c["centre_meta_engine"]["commission_pct_nav_per_year"], "slippage_pct_nav_per_year": c["centre_meta_engine"]["slippage_pct_nav_per_year"], "cost_share_of_gross_pnl": c["centre_meta_engine"]["cost_share_of_gross_pnl"]},
            "confidence": {"reality_check_paired": confidence_dict["reality_check_book_g_full_vs_C_BIL"]["paired"].get(c["centre"]), "dsr_excess_of_bil_full": confidence_dict["deflated_sharpe_excess_of_bil"][c["name"]]["full"], "walk_forward": c.get("walk_forward")},
            "small_accounts": {cap: {"sa_FULL_sharpe": v["standalone_sweep"]["FULL"]["sharpe"], "book_G-FULL_sharpe": v["books_sweep_engine"]["G-FULL"]["sharpe"], "commission_pct_nav_per_year": v["meta_engine"].get("commission_pct_nav_per_year"), "cost_share_of_gross_pnl": v["meta_engine"].get("cost_share_of_gross_pnl"), "final_nav_usd": v["meta_engine"].get("final_nav_usd")}
                               for cap, v in capacity_dict["small_accounts"].get(c["name"], {}).items()},
        } for c in candidate_list},
        "label_pods": {n: {"key": e["key"], "sa_FULL_sharpe": e["standalone_sweep"]["FULL"]["sharpe"], "book_by_block_sharpe": {b: m["sharpe"] for b, m in e["books_sweep_engine"].items()}, "margin_vs_C_BIL": e["margin_vs_C_BIL"], "turnover_one_way_x_per_year": e["meta_engine"]["turnover_one_way_x_per_year"]} for n, e in label_pods.items()},
        "reality_check": {k: v for k, v in confidence_dict["reality_check_book_g_full_vs_C_BIL"].items() if k != "paired"},
        "grid_share_of_cells_beating_C_BIL_g_full": float(np.mean([cell_res[k]["books"]["sweep_engine"]["G-FULL"]["sharpe"] > c_bil["G-FULL"]["sharpe"] for k in grid_key_list])),
        "grid_best_cell_by_book_g_full": max(grid_key_list, key=lambda k: cell_res[k]["books"]["sweep_engine"]["G-FULL"]["sharpe"]),
        "data_diagnostics": common.read_json("data_diagnostics.json", {}),
        "amendments_and_bugs": common.read_json("amendments_and_bugs.json", []),
    }
    common.write_json("summary_for_lead.json", summary_dict)
    common.log_progress(f"analysis done: {decision_dict['verdict']}; " + json.dumps({c["name"]: {k: c["rule"][k] for k in ("R1", "R2", "R3")} | {"R4": c["rule"]["R4"]["pass"], "R5": c["rule"]["R5"]["pass"], "min_margin": round(c["rule"]["engine"]["min_margin"], 3)} for c in candidate_list}))


if __name__ == "__main__":
    main()
