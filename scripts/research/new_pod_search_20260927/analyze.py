"""Analysis for the new-pod search (research only): PREREG section 8 applied mechanically.

Builds the pod returns with the idle-cash sweep, the candidate books {TAA 0.5, L 0.25, X 0.25} and the controls
(C_BIL, C_SPY, C_CASH0, G3) in the official pod model, plateaus, candidates, the rule R1-R5 versus C_BIL, the labels,
the confidence labels, walk-forward, timing luck and charts; writes results.json, table_cells.csv, table_rule.csv.
"""

from __future__ import annotations

import json
import pickle

import numpy as np
import pandas as pd
from scipy import stats

from new_pod_search_20260927 import cells as cells_module
from new_pod_search_20260927 import common
from trend_breakout_20260927 import analyze as trend_analyze

OUT_PATH = common.RESULTS_DIR_PATH
CHART_PATH = common.CHART_DIR_PATH
PRIMARY_DICT = {"M": "R1000", "S": "SP500"}
CROSS_DICT = {"M": ("R1000_SP", "R1000_EX"), "S": ("R1000_EX", "NDX")}
M0_KEY = cells_module.M0_CELL.key_str
S_ANCHOR_KEY_DICT = {form_str: cell.key_str for form_str, cell in cells_module.S_ANCHOR_DICT.items()}


# ----------------------------------------------------------------------------------------------------------------------
def load_frame(name_str: str) -> pd.DataFrame | None:
    path = OUT_PATH / f"{name_str}.parquet"
    if not path.exists():
        return None
    frame_df = pd.read_parquet(path)
    frame_df.index = pd.to_datetime(frame_df.index)
    return frame_df


def load_meta(tag_str: str) -> dict:
    path = OUT_PATH / f"meta_{tag_str}.json"
    return json.loads(path.read_text()) if path.exists() else {}


def pod_returns(tag_str: str, cost_str: str, bil_ser: pd.Series, sweep_bool: bool) -> pd.DataFrame | None:
    ret_df = load_frame(f"returns_{tag_str}_{cost_str}")
    if ret_df is None:
        return None
    ret_df = ret_df.loc[:common.END_TS]
    if not sweep_bool:
        return ret_df
    cashw_df = load_frame(f"cashw_{tag_str}_{cost_str}")
    return pd.DataFrame({key_str: common.sweep_return_ser(ret_df[key_str], cashw_df[key_str], bil_ser) for key_str in ret_df.columns})


def family_of(key_str: str) -> str:
    return key_str.split("|")[0]


def blocks_for(key_str: str) -> dict:
    return common.BLOCK_DICT_HEDGED if "|HEDGED|" in key_str else common.BLOCK_DICT


def beta_to(x_ser: pd.Series, y_ser: pd.Series, start_str: str, end_str: str) -> float:
    frame_df = pd.concat([x_ser, y_ser], axis=1).loc[start_str:end_str].dropna()
    if len(frame_df) < 30:
        return float("nan")
    x_vec, y_vec = frame_df.iloc[:, 0].to_numpy(), frame_df.iloc[:, 1].to_numpy()
    return float(np.cov(x_vec, y_vec)[0, 1] / np.var(y_vec, ddof=1))


def plateau_matrix(key_arr: np.ndarray, value_fn, use_row_bool: bool, use_col_bool: bool):
    return trend_analyze.plateau_matrix(key_arr, value_fn, use_row_bool, use_col_bool)


def stage_matrix(stage_list: list) -> tuple[list, list, np.ndarray]:
    row_list = list(dict.fromkeys(rc[0] for rc, _ in stage_list))
    col_list = list(dict.fromkeys(rc[1] for rc, _ in stage_list))
    key_arr = np.empty((len(row_list), len(col_list)), dtype=object)
    for (row_label, col_label), cell in stage_list:
        key_arr[row_list.index(row_label), col_list.index(col_label)] = cell.key_str
    return row_list, col_list, key_arr


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    CHART_PATH.mkdir(parents=True, exist_ok=True)
    seq_cmap = LinearSegmentedColormap.from_list("seq", trend_analyze.BLUE_RAMP)
    div_cmap = LinearSegmentedColormap.from_list("div", trend_analyze.DIVERGING)
    common.log_progress("analysis start")
    taa_ser = common.load_taa_ser()
    bil_ser = common.load_bil_ret_ser()
    spy_ser = common.load_spy_tr_ret_ser()
    l_engine_ser = common.load_l_ret_ser("engine")
    l_stress_ser = common.load_l_ret_ser("stress")
    res: dict = {"prereg": "docs/research/NEW_POD_SEARCH_PREREG_20260927.md", "end_ts": str(common.END_TS.date()),
                 "sweep_note": "sweep rate = BIL total return from 2007-05-31; 0 before (affects only standalone windows starting in 2000, not the book windows)"}
    controls = common.control_books(taa_ser, l_engine_ser, bil_ser, spy_ser)
    controls_stress = {"C_BIL": common.candidate_book_blocks(taa_ser, l_stress_ser, bil_ser), "C_CASH0": common.candidate_book_blocks(taa_ser, l_stress_ser, pd.Series(0.0, index=l_stress_ser.index))}
    res["controls"] = controls
    res["controls_stress"] = controls_stress
    res["controls_daily_rebalanced"] = {"C_BIL": common.candidate_book_blocks(taa_ser, l_engine_ser, bil_ser, daily_bool=True)}
    c_bil = controls["C_BIL"]
    c_bil_stress = controls_stress["C_BIL"]
    c_cash0 = controls["C_CASH0"]
    c_bil_full_ser = common.candidate_book_series(taa_ser, l_engine_ser, bil_ser)
    res["legs"] = {"BIL": common.window_metrics(bil_ser, common.BOOK_BLOCK_DICT), "SPY_TR": common.window_metrics(spy_ser, common.BOOK_BLOCK_DICT),
                   "L": common.window_metrics(l_engine_ser, common.BLOCK_DICT), "TAA": common.window_metrics(taa_ser, common.BOOK_BLOCK_DICT)}

    # ---------------------------------------------------------------- returns and per-cell metrics
    sweep_dict: dict = {}   # (family, universe, cost) -> DataFrame with sweep
    nosweep_dict: dict = {}
    for family_str in ("M", "S"):
        for universe_str in (PRIMARY_DICT[family_str],) + CROSS_DICT[family_str]:
            for cost_str in ("engine", "stress"):
                sweep_df = pod_returns(f"{family_str}_{universe_str}", cost_str, bil_ser, True)
                if sweep_df is not None:
                    sweep_dict[(family_str, universe_str, cost_str)] = sweep_df
                    nosweep_dict[(family_str, universe_str, cost_str)] = pod_returns(f"{family_str}_{universe_str}", cost_str, bil_ser, False)
    meta_dict = {(f, u): load_meta(f"{f}_{u}") for f in ("M", "S") for u in (PRIMARY_DICT[f],) + CROSS_DICT[f]}
    cell_res: dict = {}
    book_full_dict: dict[str, pd.Series] = {"C_BIL": c_bil_full_ser}
    for family_str in ("M", "S"):
        primary_str = PRIMARY_DICT[family_str]
        sweep_df = sweep_dict.get((family_str, primary_str, "engine"))
        if sweep_df is None:
            continue
        stress_df = sweep_dict.get((family_str, primary_str, "stress"))
        nosweep_df = nosweep_dict[(family_str, primary_str, "engine")]
        for key_str in sweep_df.columns:
            x_ser = sweep_df[key_str]
            block_dict = blocks_for(key_str)
            full_start_str, full_end_str = block_dict["FULL"]
            entry = {
                "family": family_str,
                "standalone_sweep": common.window_metrics(x_ser, block_dict),
                "standalone_nosweep": common.window_metrics(nosweep_df[key_str], block_dict),
                "standalone_sweep_stress": common.window_metrics(stress_df[key_str], block_dict) if stress_df is not None else None,
                "beta_spy_full": beta_to(x_ser, spy_ser, full_start_str, full_end_str),
                "corr_taa_2008_2026": float(x_ser.loc["2008-03-04":].corr(taa_ser.loc["2008-03-04":])),
                "corr_L_full": float(x_ser.loc[full_start_str:].corr(l_engine_ser.loc[full_start_str:])),
                "books": {
                    "sweep_engine": common.candidate_book_blocks(taa_ser, l_engine_ser, x_ser),
                    "sweep_stress": common.candidate_book_blocks(taa_ser, l_stress_ser, stress_df[key_str]) if stress_df is not None else None,
                    "nosweep_engine": common.candidate_book_blocks(taa_ser, l_engine_ser, nosweep_df[key_str]),
                },
                "cross_universe_books_sweep_engine": {},
                "cross_universe_standalone_full_sweep": {},
            }
            for universe_str in CROSS_DICT[family_str]:
                other_df = sweep_dict.get((family_str, universe_str, "engine"))
                if other_df is not None and key_str in other_df:
                    entry["cross_universe_books_sweep_engine"][universe_str] = common.candidate_book_blocks(taa_ser, l_engine_ser, other_df[key_str])
                    entry["cross_universe_standalone_full_sweep"][universe_str] = common.metric_dict(other_df[key_str].loc[full_start_str:full_end_str])
            book_full_dict[key_str] = common.candidate_book_series(taa_ser, l_engine_ser, x_ser)
            cell_res[key_str] = entry
    res["cells"] = cell_res
    res["cell_meta_primary"] = {k: meta_dict[(v["family"], PRIMARY_DICT[v["family"]])].get(k, {}) for k, v in cell_res.items()}
    res["cell_meta_cross"] = {f"{f}_{u}": meta_dict[(f, u)] for f in ("M", "S") for u in CROSS_DICT[f] if meta_dict.get((f, u))}

    # ---------------------------------------------------------------- plateaus and candidates
    def sa_full(key_str: str) -> float:
        return cell_res[key_str]["standalone_sweep"]["FULL"]["sharpe"] if key_str in cell_res else float("nan")

    def sa_p1(key_str: str) -> float:
        return cell_res[key_str]["standalone_sweep"]["P1"]["sharpe"] if key_str in cell_res else float("nan")

    grids: dict = {}
    candidate_list: list[dict] = []

    def add_candidate(family_str: str, stage_str: str, stage_list: list, anchor_key: str, use_row_bool: bool, use_col_bool: bool) -> None:
        row_list, col_list, key_arr = stage_matrix(stage_list)
        anchor_pos = tuple(int(x) for x in np.argwhere(key_arr == anchor_key)[0])
        value_arr, plateau_arr = plateau_matrix(key_arr, sa_full, use_row_bool, use_col_bool)
        _, plateau_p1_arr = plateau_matrix(key_arr, sa_p1, use_row_bool, use_col_bool)

        def best_pos(plat_arr):
            return max(((i, j) for i in range(key_arr.shape[0]) for j in range(key_arr.shape[1])),
                       key=lambda p: (round(float(np.nan_to_num(plat_arr[p], nan=-9.0)), 12), -(abs(p[0] - anchor_pos[0]) + abs(p[1] - anchor_pos[1]))))

        centre_pos, wf_pos = best_pos(plateau_arr), best_pos(plateau_p1_arr)
        grids[stage_str] = {"rows": [str(r) for r in row_list], "cols": [str(c) for c in col_list], "keys": key_arr.tolist(), "standalone_full_sharpe_sweep": value_arr.tolist(),
                            "plateau_full_sharpe_sweep": plateau_arr.tolist(), "anchor_pos": anchor_pos, "centre_pos": centre_pos}
        candidate_list.append({"family": family_str, "stage": stage_str, "centre": key_arr[centre_pos], "anchor": anchor_key,
                               "neighbourhood": [key_arr[p] for p in trend_analyze.neighbour_list(*centre_pos, key_arr.shape, use_row_bool, use_col_bool)],
                               "plateau_full_sharpe": float(plateau_arr[centre_pos]), "walk_forward_centre": key_arr[wf_pos],
                               "walk_forward_neighbourhood": [key_arr[p] for p in trend_analyze.neighbour_list(*wf_pos, key_arr.shape, use_row_bool, use_col_bool)], "grid_pos": centre_pos})

    if any(k.startswith("M|") for k in cell_res):
        m_stages = cells_module.family_m_stage_dict()
        add_candidate("M", "M1_event_pin", m_stages["M1_event_pin"], M0_KEY, True, True)
        add_candidate("M", "M2_window_slots", m_stages["M2_window_slots"], M0_KEY, True, True)
    if any(k.startswith("S|") for k in cell_res):
        for stage_str, stage_list in cells_module.family_s_stage_dict().items():
            add_candidate("S", stage_str, stage_list, S_ANCHOR_KEY_DICT[stage_str.split("_")[1]], True, True)
    res["grids"] = grids

    # ---------------------------------------------------------------- the rule
    def neigh_median(key_list: list[str], getter) -> float:
        value_list = [getter(k) for k in key_list if k in cell_res]
        value_list = [v for v in value_list if v is not None and np.isfinite(v)]
        return float(np.median(value_list)) if value_list else float("nan")

    def margins(neigh: list[str], book_str: str, control: dict) -> dict:
        margin_dict = {b: neigh_median(neigh, lambda k: cell_res[k]["books"][book_str][b]["sharpe"]) - control[b]["sharpe"] for b in common.RULE_BLOCK_TUPLE}
        dd_gap_dict = {b: neigh_median(neigh, lambda k: cell_res[k]["books"][book_str][b]["max_dd"]) - control[b]["max_dd"] for b in ("G-FULL", "G-LONG")}
        return {"sharpe_margin": margin_dict, "min_margin": float(min(margin_dict.values())), "dd_gap_pp": {b: v * 100 for b, v in dd_gap_dict.items()},
                "R1": bool(all(m > 0 for m in margin_dict.values())), "R2": bool(all(v >= -common.DD_TOLERANCE_FLOAT for v in dd_gap_dict.values())),
                "g_full_margin": neigh_median(neigh, lambda k: cell_res[k]["books"][book_str]["G-FULL"]["sharpe"]) - control["G-FULL"]["sharpe"]}

    for c in candidate_list:
        neigh = [k for k in c["neighbourhood"] if k in cell_res]
        family_str = c["family"]
        engine_rule = margins(neigh, "sweep_engine", c_bil)
        stress_rule = margins(neigh, "sweep_stress", c_bil_stress)
        r4_dict = {}
        for universe_str in CROSS_DICT[family_str]:
            value_float = neigh_median(neigh, lambda k: cell_res[k]["cross_universe_books_sweep_engine"].get(universe_str, {}).get("G-FULL", {}).get("sharpe"))
            r4_dict[universe_str] = {"neigh_median_book_g_full_sharpe": value_float, "C_BIL_g_full_sharpe": c_bil["G-FULL"]["sharpe"], "pass": bool(np.isfinite(value_float) and value_float > c_bil["G-FULL"]["sharpe"])}
        capacity = res["cell_meta_primary"].get(c["centre"], {}).get("engine", {}).get("capacity_2021_2026", {})
        r5_bool = bool(capacity.get("aum_max_5pct_usd", 0.0) >= common.CAPACITY_MIN_AUM_FLOAT)
        centre_book = cell_res[c["centre"]]["books"]["sweep_engine"]
        neigh_full_sharpe = neigh_median(neigh, lambda k: cell_res[k]["books"]["sweep_engine"]["G-FULL"]["sharpe"])
        neigh_full_dd = neigh_median(neigh, lambda k: cell_res[k]["books"]["sweep_engine"]["G-FULL"]["max_dd"])
        c["rule"] = {
            "engine": engine_rule, "stress": stress_rule, "R3": bool(stress_rule["R1"] and stress_rule["R2"]), "R4": r4_dict,
            "R5": {"aum_max_5pct_usd_2021_2026": capacity.get("aum_max_5pct_usd"), "aum_p95_1pct_usd_2021_2026": capacity.get("aum_p95_1pct_usd"), "pass": r5_bool},
            "passes": bool(engine_rule["R1"] and engine_rule["R2"] and stress_rule["R1"] and stress_rule["R2"] and all(v["pass"] for v in r4_dict.values()) and r5_bool),
            "labels": {
                "vs_G3": margins(neigh, "sweep_engine", controls["G3"]),
                "vs_C_SPY": margins(neigh, "sweep_engine", controls["C_SPY"]),
                "nosweep_vs_C_CASH0": margins(neigh, "nosweep_engine", c_cash0),
                "owner_gate_centre": {"sharpe": centre_book["G-FULL"]["sharpe"], "max_dd": centre_book["G-FULL"]["max_dd"], "cagr": centre_book["G-FULL"]["cagr"],
                                      "pass": bool(centre_book["G-FULL"]["sharpe"] >= common.OWNER_GATE_SHARPE_FLOAT and centre_book["G-FULL"]["max_dd"] >= common.OWNER_GATE_MAX_DD_FLOAT)},
                "owner_gate_neighbourhood_median": {"sharpe": neigh_full_sharpe, "max_dd": neigh_full_dd,
                                                    "pass": bool(neigh_full_sharpe >= common.OWNER_GATE_SHARPE_FLOAT and neigh_full_dd >= common.OWNER_GATE_MAX_DD_FLOAT)},
            },
            "neigh_median_books_sweep_engine": {b: {m: neigh_median(neigh, lambda k: cell_res[k]["books"]["sweep_engine"][b][m]) for m in ("cagr", "sharpe", "max_dd")} for b in common.BOOK_BLOCK_DICT},
            "centre_books_sweep_engine": centre_book,
        }
        c["centre_standalone_sweep"] = cell_res[c["centre"]]["standalone_sweep"]
        c["neigh_median_standalone_sweep"] = {b: {m: neigh_median(neigh, lambda k: cell_res[k]["standalone_sweep"][b][m]) for m in ("cagr", "sharpe", "max_dd")} for b in blocks_for(c["centre"])}
        wf_neigh = [k for k in c["walk_forward_neighbourhood"] if k in cell_res]
        c["walk_forward"] = {"centre_by_P1": c["walk_forward_centre"],
                             "standalone_sweep_neigh_median_sharpe": {b: neigh_median(wf_neigh, lambda k: cell_res[k]["standalone_sweep"][b]["sharpe"]) for b in ("P2", "P3")},
                             "book_neigh_median_sharpe": {b: neigh_median(wf_neigh, lambda k: cell_res[k]["books"]["sweep_engine"][b]["sharpe"]) for b in ("G-P2", "G-P3")},
                             "C_BIL": {b: c_bil[b]["sharpe"] for b in ("G-P2", "G-P3")}}
    res["candidates"] = candidate_list
    passing_list = [c for c in candidate_list if c["rule"]["passes"]]
    if passing_list:
        best = max(passing_list, key=lambda c: c["rule"]["engine"]["min_margin"])
        res["decision"] = {"decision": "recommend", "recommended": {"stage": best["stage"], "centre": best["centre"], "min_margin": best["rule"]["engine"]["min_margin"]},
                           "passing": [(c["stage"], c["centre"]) for c in passing_list], "shadow_line": None}
    else:
        eligible = [c for c in candidate_list if c["rule"]["engine"]["R2"] and c["rule"]["R5"]["pass"]]
        pool = eligible if eligible else candidate_list
        best = max(pool, key=lambda c: c["rule"]["engine"]["min_margin"]) if pool else None
        res["decision"] = {"decision": "no new pod recommended", "passing": [],
                           "shadow_line": {"stage": best["stage"], "centre": best["centre"], "min_margin": best["rule"]["engine"]["min_margin"], "flagged_no_candidate_satisfies_R2_and_R5": not eligible} if best else None}

    # ---------------------------------------------------------------- confidence labels
    book_full_df = pd.DataFrame(book_full_dict).dropna()
    grid_key_set = {c.key_str for c in cells_module.family_m_cells(include_sensitivities_bool=False)} | {c.key_str for c in cells_module.family_s_cells()}
    grid_key_list = [k for k in book_full_df.columns if k in grid_key_set]  # the 36 frozen configurations (PREREG section 8)
    all_key_list = [k for k in book_full_df.columns if k != "C_BIL"]  # plus the 3 M sensitivity cells (reported, not part of the frozen count)
    candidate_keys = [c["centre"] for c in candidate_list]
    res["multiplicity"] = {"book_configs_int": len(grid_key_list),
                           "reality_check_book_g_full_vs_C_BIL": trend_analyze.reality_check(book_full_df[grid_key_list + ["C_BIL"]], "C_BIL", candidate_keys, common.SEED_INT),
                           "reality_check_incl_sensitivity_cells": trend_analyze.reality_check(book_full_df[all_key_list + ["C_BIL"]], "C_BIL", candidate_keys, common.SEED_INT)}
    dsr_dict = {}
    for c in candidate_list:
        family_keys = [k for k, v in cell_res.items() if v["family"] == c["family"] and blocks_for(k) == blocks_for(c["centre"])]
        start_str, end_str = blocks_for(c["centre"])["FULL"]
        family_df = sweep_dict[(c["family"], PRIMARY_DICT[c["family"]], "engine")][family_keys].loc[start_str:end_str]
        family_sharpe_vec = (family_df.mean() / family_df.std()).to_numpy()
        dsr_dict[c["stage"]] = trend_analyze.deflated_sharpe(family_df[c["centre"]], family_sharpe_vec, common.N_TRIALS_INT)
    res["multiplicity"]["deflated_sharpe_standalone_sweep"] = dsr_dict

    # ---------------------------------------------------------------- timing luck (S anchors)
    luck_dict: dict = {}
    offsets_df = pod_returns("S_SP500_offsets", "engine", bil_ser, True)
    base_df = sweep_dict.get(("S", "SP500", "engine"))
    if offsets_df is not None and base_df is not None:
        all_df = pd.concat([base_df, offsets_df], axis=1)
        for form_str, anchor_key in S_ANCHOR_KEY_DICT.items():
            prefix_str = anchor_key.rsplit("|k", 1)[0]
            key_by_offset = {k: f"{prefix_str}|k{k:+d}" for k in cells_module.OFFSET_TUPLE}
            start_str, end_str = blocks_for(anchor_key)["FULL"]
            sa_vec = np.array([common.metric_dict(all_df[key_by_offset[k]].loc[start_str:end_str])["sharpe"] for k in cells_module.OFFSET_TUPLE])
            g_vec = np.array([common.candidate_book_blocks(taa_ser, l_engine_ser, all_df[key_by_offset[k]])["G-FULL"]["sharpe"] for k in cells_module.OFFSET_TUPLE])
            k0 = cells_module.OFFSET_TUPLE.index(0)
            luck_dict[form_str] = {"anchor_key": anchor_key,
                                   "standalone_full": {"k0": float(sa_vec[k0]), "min": float(sa_vec.min()), "median": float(np.median(sa_vec)), "max": float(sa_vec.max()), "k0_percentile": float(np.mean(sa_vec <= sa_vec[k0])), "by_offset": sa_vec.tolist()},
                                   "book_g_full": {"k0": float(g_vec[k0]), "min": float(g_vec.min()), "median": float(np.median(g_vec)), "max": float(g_vec.max()), "k0_percentile": float(np.mean(g_vec <= g_vec[k0])),
                                                   "share_above_C_BIL": float(np.mean(g_vec > c_bil["G-FULL"]["sharpe"])), "by_offset": g_vec.tolist()}}
    res["timing_luck_S"] = luck_dict

    # ---------------------------------------------------------------- cross-universe tables
    res["cross_universe"] = {}
    for family_str in ("M", "S"):
        primary_str = PRIMARY_DICT[family_str]
        primary_df = sweep_dict.get((family_str, primary_str, "engine"))
        if primary_df is None:
            continue
        fam = {"primary": primary_str, "by_universe": {}, "spearman_vs_primary": {}}
        for universe_str in (primary_str,) + CROSS_DICT[family_str]:
            frame_df = sweep_dict.get((family_str, universe_str, "engine"))
            if frame_df is None:
                continue
            fam["by_universe"][universe_str] = {k: {"standalone_full_sweep": common.metric_dict(frame_df[k].loc[blocks_for(k)["FULL"][0]:blocks_for(k)["FULL"][1]]),
                                                    "book_g_full_sweep": common.candidate_book_blocks(taa_ser, l_engine_ser, frame_df[k])["G-FULL"]} for k in frame_df.columns}
        for universe_str in CROSS_DICT[family_str]:
            table = fam["by_universe"].get(universe_str)
            if table is None:
                continue
            keys = [k for k in primary_df.columns if k in table]
            a = [fam["by_universe"][primary_str][k]["standalone_full_sweep"]["sharpe"] for k in keys]
            b = [table[k]["standalone_full_sweep"]["sharpe"] for k in keys]
            fam["spearman_vs_primary"][universe_str] = float(stats.spearmanr(a, b).statistic) if len(keys) > 2 else float("nan")
        res["cross_universe"][family_str] = fam

    # ---------------------------------------------------------------- CSV tables
    row_list = []
    for key_str, entry in cell_res.items():
        row = {"key": key_str, "family": entry["family"], "beta_spy": entry["beta_spy_full"], "corr_taa": entry["corr_taa_2008_2026"], "corr_L": entry["corr_L_full"]}
        for b, m in entry["standalone_sweep"].items():
            row.update({f"sa_sweep_{b}_sharpe": m["sharpe"], f"sa_sweep_{b}_cagr": m["cagr"], f"sa_sweep_{b}_dd": m["max_dd"]})
        row["sa_nosweep_FULL_sharpe"] = entry["standalone_nosweep"]["FULL"]["sharpe"]
        for b, m in entry["books"]["sweep_engine"].items():
            row.update({f"book_{b}_sharpe": m["sharpe"], f"book_{b}_cagr": m["cagr"], f"book_{b}_dd": m["max_dd"], f"book_{b}_margin_vs_CBIL": m["sharpe"] - c_bil[b]["sharpe"]})
        if entry["books"]["sweep_stress"]:
            row["book_stress_G-FULL_sharpe"] = entry["books"]["sweep_stress"]["G-FULL"]["sharpe"]
        row["book_nosweep_G-FULL_sharpe"] = entry["books"]["nosweep_engine"]["G-FULL"]["sharpe"]
        for u, m in entry["cross_universe_books_sweep_engine"].items():
            row[f"book_G-FULL_sharpe_{u}"] = m["G-FULL"]["sharpe"]
        meta_e = res["cell_meta_primary"].get(key_str, {}).get("engine", {})
        row.update({k: meta_e.get(k) for k in ("turnover_x_per_year", "mean_exposure", "round_trips_per_year", "holding_sessions_median", "terminal_liquidations_int", "terminal_pnl_share", "phantom_fill_int")})
        cap = meta_e.get("capacity_2021_2026", {})
        row.update({"cap21_max_5pct_usd": cap.get("aum_max_5pct_usd"), "cap21_p95_1pct_usd": cap.get("aum_p95_1pct_usd")})
        if "m" in meta_e:
            row.update({"m_entries_per_year": meta_e["m"]["entries_per_year"], "m_precision": meta_e["m"]["precision_terminal_within_252"], "m_events_per_year": meta_e["m"]["events_per_year"]})
        if "s" in meta_e:
            row["s_score_coverage"] = meta_e["s"]["score_coverage_mean"]
        row_list.append(row)
    pd.DataFrame(row_list).to_csv(OUT_PATH / "table_cells.csv", index=False)
    rule_rows = []
    for c in candidate_list:
        r = c["rule"]
        rule_rows.append({"stage": c["stage"], "centre": c["centre"], "plateau_full_sharpe_sweep": c["plateau_full_sharpe"],
                          **{f"margin_{b}": v for b, v in r["engine"]["sharpe_margin"].items()}, "min_margin": r["engine"]["min_margin"],
                          **{f"dd_gap_pp_{b}": v for b, v in r["engine"]["dd_gap_pp"].items()}, "R1": r["engine"]["R1"], "R2": r["engine"]["R2"], "R3": r["R3"],
                          **{f"R4_{u}": v["pass"] for u, v in r["R4"].items()}, "R5": r["R5"]["pass"], "passes": r["passes"],
                          "beats_G3_all_blocks": r["labels"]["vs_G3"]["R1"], "beats_C_SPY_all_blocks": r["labels"]["vs_C_SPY"]["R1"], "nosweep_beats_C_CASH0_all_blocks": r["labels"]["nosweep_vs_C_CASH0"]["R1"],
                          "owner_gate_centre": r["labels"]["owner_gate_centre"]["pass"], "centre_book_sharpe": r["labels"]["owner_gate_centre"]["sharpe"], "centre_book_dd": r["labels"]["owner_gate_centre"]["max_dd"]})
    pd.DataFrame(rule_rows).to_csv(OUT_PATH / "table_rule.csv", index=False)

    # ---------------------------------------------------------------- charts
    for stage_str, grid in grids.items():
        key_arr = np.array(grid["keys"], dtype=object)
        value_arr = np.array(grid["standalone_full_sharpe_sweep"])
        book_arr = np.vectorize(lambda k: cell_res[k]["books"]["sweep_engine"]["G-FULL"]["sharpe"] - c_bil["G-FULL"]["sharpe"] if k in cell_res else np.nan, otypes=[float])(key_arr)
        fig, axes = plt.subplots(1, 2, figsize=(11, 0.6 * len(grid["rows"]) + 2.2), facecolor=trend_analyze.SURFACE)
        trend_analyze.heatmap(axes[0], value_arr, grid["rows"], grid["cols"], "Standalone Sharpe FULL (with sweep)", seq_cmap, np.nanmin(value_arr), np.nanmax(value_arr), tuple(grid["anchor_pos"]), tuple(grid["centre_pos"]))
        trend_analyze.heatmap(axes[1], book_arr, grid["rows"], grid["cols"], f"Book G-FULL Sharpe minus C_BIL ({c_bil['G-FULL']['sharpe']:.3f})", div_cmap, -0.15, 0.15, tuple(grid["anchor_pos"]), tuple(grid["centre_pos"]), fmt="{:+.3f}")
        fig.suptitle(f"{stage_str}: black box = anchor, dashed = plateau centre", fontsize=10, color=trend_analyze.TEXT_SECONDARY, x=0.01, ha="left")
        fig.tight_layout()
        fig.savefig(CHART_PATH / f"heatmap_{stage_str}.png", dpi=130)
        plt.close(fig)
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), facecolor=trend_analyze.SURFACE, sharex=True)
    series_list = [("C_BIL", c_bil_full_ser), ("G3", common.book_window_return_ser({"taa": taa_ser, "L": l_engine_ser}, common.G3_WEIGHT_DICT, *common.BOOK_BLOCK_DICT["G-FULL"])),
                   ("C_SPY", common.candidate_book_series(taa_ser, l_engine_ser, spy_ser))] + [(f"{c['stage']}: {c['centre']}", book_full_dict[c["centre"]]) for c in candidate_list]
    for (label_str, ser), color in zip(series_list, ["#0b0b0b", "#52514e", "#9ec5f4", "#2a78d6", "#eb6834", "#1baf7a", "#eda100"]):
        wealth_ser = (1 + ser.dropna()).cumprod()
        axes[0].plot(wealth_ser.index, wealth_ser, color=color, lw=1.6 if label_str in ("C_BIL", "G3") else 1.0, label=label_str)
        axes[1].plot(wealth_ser.index, wealth_ser / wealth_ser.cummax() - 1, color=color, lw=1.6 if label_str in ("C_BIL", "G3") else 1.0)
    axes[0].set_yscale("log")
    axes[0].set_title("Candidate books {TAA 0.5, L 0.25, X 0.25} versus C_BIL, G3 and C_SPY, 2012-10-02..2026-08-19", fontsize=9.5, loc="left", color=trend_analyze.TEXT_PRIMARY)
    axes[1].set_title("Drawdown", fontsize=9.5, loc="left", color=trend_analyze.TEXT_PRIMARY)
    axes[0].legend(fontsize=7, frameon=False, ncol=2)
    for ax in axes:
        trend_analyze.style_axis(ax)
    fig.tight_layout()
    fig.savefig(CHART_PATH / "equity_dd_candidates.png", dpi=130)
    plt.close(fig)

    for name_str in ("parity_gate", "invariance_v1", "invariance_v2", "amendments_and_bugs"):
        path = OUT_PATH / f"{name_str}.json"
        res[name_str] = json.loads(path.read_text()) if path.exists() else None
    common.write_json("results.json", res)
    common.log_progress(f"analysis done: {json.dumps(res['decision'], default=str)[:500]}")
    for c in candidate_list:
        r = c["rule"]
        print(c["stage"], "| centre", c["centre"], "| margins vs C_BIL", {b: round(v, 3) for b, v in r["engine"]["sharpe_margin"].items()}, "| dd", {b: round(v, 2) for b, v in r["engine"]["dd_gap_pp"].items()},
              "| R3", r["R3"], "| R4", {u: v["pass"] for u, v in r["R4"].items()}, "| R5", r["R5"]["pass"], "| passes", r["passes"])


if __name__ == "__main__":
    main()
