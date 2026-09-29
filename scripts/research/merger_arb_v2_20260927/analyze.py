"""Analysis for merger arbitrage v2 (research only): PREREG section 8 applied mechanically; results.json, tables, charts."""

from __future__ import annotations

import json
import pickle

import numpy as np
import pandas as pd

from merger_arb_v2_20260927 import cells as cells_module
from merger_arb_v2_20260927 import common
from trend_breakout_20260927 import analyze as trend_analyze

OUT_PATH = common.RESULTS_DIR_PATH
CHART_PATH = common.CHART_DIR_PATH
V0_KEY = cells_module.V0_CELL.key_str


def load_frame(name_str: str) -> pd.DataFrame | None:
    path = OUT_PATH / f"{name_str}.parquet"
    if not path.exists():
        return None
    frame_df = pd.read_parquet(path)
    frame_df.index = pd.to_datetime(frame_df.index)
    return frame_df


def pod_returns(tag_str: str, cost_str: str, bil_ser: pd.Series, sweep_bool: bool) -> pd.DataFrame | None:
    ret_df = load_frame(f"returns_{tag_str}_{cost_str}")
    if ret_df is None:
        return None
    ret_df = ret_df.loc[:common.END_TS]
    if not sweep_bool:
        return ret_df
    cashw_df = load_frame(f"cashw_{tag_str}_{cost_str}")
    return pd.DataFrame({k: common.sweep_return_ser(ret_df[k], cashw_df[k], bil_ser) for k in ret_df.columns})


def beta_to(x_ser: pd.Series, y_ser: pd.Series, start_str: str, end_str: str) -> float:
    frame_df = pd.concat([x_ser, y_ser], axis=1).loc[start_str:end_str].dropna()
    return float(np.cov(frame_df.iloc[:, 0], frame_df.iloc[:, 1])[0, 1] / np.var(frame_df.iloc[:, 1], ddof=1)) if len(frame_df) > 30 else float("nan")


def stage_matrix(stage_list: list) -> tuple[list, list, np.ndarray]:
    row_list = list(dict.fromkeys(rc[0] for rc, _ in stage_list))
    col_list = list(dict.fromkeys(rc[1] for rc, _ in stage_list))
    key_arr = np.empty((len(row_list), len(col_list)), dtype=object)
    for (row_label, col_label), cell in stage_list:
        key_arr[row_list.index(row_label), col_list.index(col_label)] = cell.key_str
    return row_list, col_list, key_arr


def half_key(key_str: str, half_str: str) -> str:
    return f"{key_str}|{half_str}"


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    CHART_PATH.mkdir(parents=True, exist_ok=True)
    seq_cmap = LinearSegmentedColormap.from_list("seq", trend_analyze.BLUE_RAMP)
    div_cmap = LinearSegmentedColormap.from_list("div", trend_analyze.DIVERGING)
    common.log_progress("analysis start")
    taa_ser, bil_ser, spy_ser = common.load_taa_ser(), common.load_bil_ret_ser(), common.load_spy_tr_ret_ser()
    l_engine_ser, l_stress_ser = common.load_l_ret_ser("engine"), common.load_l_ret_ser("stress")
    mna_ser = common.load_mna_ret_ser()
    m0_v1_ser = common.load_v1_m0_ret_ser(bil_ser)
    res: dict = {"prereg": "docs/research/MERGER_ARB_V2_PREREG_20260927.md", "end_ts": str(common.END_TS.date())}
    controls = common.control_books(taa_ser, l_engine_ser, bil_ser, spy_ser)
    controls["C_MNA"] = {b: v for b, v in common.candidate_book_blocks(taa_ser, l_engine_ser, mna_ser).items() if b in ("G-P2", "G-P3", "G-FULL")}
    if m0_v1_ser is not None:
        controls["C_M0_v1"] = common.candidate_book_blocks(taa_ser, l_engine_ser, m0_v1_ser)
    controls_stress = {"C_BIL": common.candidate_book_blocks(taa_ser, l_stress_ser, bil_ser)}
    res["controls"], res["controls_stress"] = controls, controls_stress
    res["legs"] = {"MNA_standalone": common.window_metrics(mna_ser, {"2009-11..END": (common.MNA_START_STR, "2026-08-19"), "G-P2": common.BOOK_BLOCK_DICT["G-P2"], "G-P3": common.BOOK_BLOCK_DICT["G-P3"], "G-FULL": common.BOOK_BLOCK_DICT["G-FULL"]}),
                   "MNA_beta_spy_2012_2026": beta_to(mna_ser, spy_ser, "2012-10-02", "2026-08-19"), "BIL": common.window_metrics(bil_ser, common.BOOK_BLOCK_DICT)}
    c_bil, c_bil_stress, c_cash0 = controls["C_BIL"], controls_stress["C_BIL"], controls["C_CASH0"]
    c_bil_full_ser = common.candidate_book_series(taa_ser, l_engine_ser, bil_ser)

    sweep_all = pod_returns("V_ALL", "engine", bil_ser, True)
    stress_all = pod_returns("V_ALL", "stress", bil_ser, True)
    nosweep_all = pod_returns("V_ALL", "engine", bil_ser, False)
    halves = {h: pod_returns(f"V_{h}", "engine", bil_ser, True) for h in cells_module.HALF_TUPLE}
    meta_all = json.loads((OUT_PATH / "meta_V_ALL.json").read_text())
    meta_halves = {h: json.loads((OUT_PATH / f"meta_V_{h}.json").read_text()) if (OUT_PATH / f"meta_V_{h}.json").exists() else {} for h in cells_module.HALF_TUPLE}
    cell_res: dict = {}
    book_full_dict: dict[str, pd.Series] = {"C_BIL": c_bil_full_ser}
    for key_str in sweep_all.columns:
        x_ser = sweep_all[key_str]
        entry = {
            "standalone_sweep": common.window_metrics(x_ser, common.BLOCK_DICT), "standalone_nosweep": common.window_metrics(nosweep_all[key_str], common.BLOCK_DICT),
            "standalone_sweep_stress": common.window_metrics(stress_all[key_str], common.BLOCK_DICT) if stress_all is not None and key_str in stress_all else None,
            "beta_spy_full": beta_to(x_ser, spy_ser, *common.BLOCK_DICT["FULL"]), "beta_spy_nosweep_full": beta_to(nosweep_all[key_str], spy_ser, *common.BLOCK_DICT["FULL"]),
            "corr_taa_2008_2026": float(x_ser.loc["2008-03-04":].corr(taa_ser.loc["2008-03-04":])), "corr_L_full": float(x_ser.corr(l_engine_ser.reindex(x_ser.index))),
            "corr_mna_2009_2026": float(x_ser.loc[common.MNA_START_STR:].corr(mna_ser)),
            "books": {"sweep_engine": common.candidate_book_blocks(taa_ser, l_engine_ser, x_ser),
                      "sweep_stress": common.candidate_book_blocks(taa_ser, l_stress_ser, stress_all[key_str]) if stress_all is not None and key_str in stress_all else None,
                      "nosweep_engine": common.candidate_book_blocks(taa_ser, l_engine_ser, nosweep_all[key_str])},
            "half_books_sweep_engine": {}, "half_standalone_full_sweep": {},
        }
        for half_str, frame_df in halves.items():
            hk = half_key(key_str, half_str)
            if frame_df is not None and hk in frame_df:
                entry["half_books_sweep_engine"][half_str] = common.candidate_book_blocks(taa_ser, l_engine_ser, frame_df[hk])
                entry["half_standalone_full_sweep"][half_str] = common.metric_dict(frame_df[hk].loc[common.BLOCK_DICT["FULL"][0]:])
        book_full_dict[key_str] = common.candidate_book_series(taa_ser, l_engine_ser, x_ser)
        cell_res[key_str] = entry
    res["cells"] = cell_res
    res["cell_meta"] = meta_all
    res["cell_meta_halves"] = meta_halves

    # ---------------------------------------------------------------- plateaus and candidates
    def sa_full(k):
        return cell_res[k]["standalone_sweep"]["FULL"]["sharpe"] if k in cell_res else float("nan")

    def sa_p1(k):
        return cell_res[k]["standalone_sweep"]["P1"]["sharpe"] if k in cell_res else float("nan")

    grids, candidate_list = {}, []
    for stage_str, stage_list in cells_module.stage_dict().items():
        row_list, col_list, key_arr = stage_matrix(stage_list)
        anchor_pos = tuple(int(x) for x in np.argwhere(key_arr == V0_KEY)[0])
        # Stage P neighbourhood: within one theta step, in both J rows (rows = J, cols = theta) -> row+col box; stage S: 3 x 3 box
        value_arr, plateau_arr = trend_analyze.plateau_matrix(key_arr, sa_full, True, True)
        _, plateau_p1_arr = trend_analyze.plateau_matrix(key_arr, sa_p1, True, True)

        def best_pos(plat_arr):
            return max(((i, j) for i in range(key_arr.shape[0]) for j in range(key_arr.shape[1])),
                       key=lambda p: (round(float(np.nan_to_num(plat_arr[p], nan=-9.0)), 12), -(abs(p[0] - anchor_pos[0]) + abs(p[1] - anchor_pos[1]))))

        centre_pos, wf_pos = best_pos(plateau_arr), best_pos(plateau_p1_arr)
        grids[stage_str] = {"rows": [str(r) for r in row_list], "cols": [str(c) for c in col_list], "keys": key_arr.tolist(), "standalone_full_sharpe_sweep": value_arr.tolist(),
                            "plateau_full_sharpe_sweep": plateau_arr.tolist(), "anchor_pos": anchor_pos, "centre_pos": centre_pos}
        candidate_list.append({"stage": stage_str, "centre": key_arr[centre_pos], "neighbourhood": [key_arr[p] for p in trend_analyze.neighbour_list(*centre_pos, key_arr.shape, True, True)],
                               "plateau_full_sharpe": float(plateau_arr[centre_pos]), "walk_forward_centre": key_arr[wf_pos],
                               "walk_forward_neighbourhood": [key_arr[p] for p in trend_analyze.neighbour_list(*wf_pos, key_arr.shape, True, True)], "grid_pos": centre_pos})
    res["grids"] = grids

    def neigh_median(key_list, getter):
        vals = [getter(k) for k in key_list if k in cell_res]
        vals = [v for v in vals if v is not None and np.isfinite(v)]
        return float(np.median(vals)) if vals else float("nan")

    def margins(neigh, book_str, control, blocks=common.RULE_BLOCK_TUPLE):
        margin_dict = {b: neigh_median(neigh, lambda k: cell_res[k]["books"][book_str][b]["sharpe"]) - control[b]["sharpe"] for b in blocks if b in control}
        dd_gap_dict = {b: neigh_median(neigh, lambda k: cell_res[k]["books"][book_str][b]["max_dd"]) - control[b]["max_dd"] for b in ("G-FULL", "G-LONG") if b in control}
        return {"sharpe_margin": margin_dict, "min_margin": float(min(margin_dict.values())) if margin_dict else float("nan"), "dd_gap_pp": {b: v * 100 for b, v in dd_gap_dict.items()},
                "R1": bool(margin_dict and all(m > 0 for m in margin_dict.values())), "R2": bool(dd_gap_dict and all(v >= -common.DD_TOLERANCE_FLOAT for v in dd_gap_dict.values())),
                "g_full_margin": neigh_median(neigh, lambda k: cell_res[k]["books"][book_str]["G-FULL"]["sharpe"]) - control["G-FULL"]["sharpe"] if "G-FULL" in control else float("nan")}

    for c in candidate_list:
        neigh = [k for k in c["neighbourhood"] if k in cell_res]
        engine_rule, stress_rule = margins(neigh, "sweep_engine", c_bil), margins(neigh, "sweep_stress", c_bil_stress)
        r4 = {}
        for half_str in cells_module.HALF_TUPLE:
            v = neigh_median(neigh, lambda k: cell_res[k]["half_books_sweep_engine"].get(half_str, {}).get("G-FULL", {}).get("sharpe"))
            r4[half_str] = {"neigh_median_book_g_full_sharpe": v, "C_BIL_g_full_sharpe": c_bil["G-FULL"]["sharpe"], "pass": bool(np.isfinite(v) and v > c_bil["G-FULL"]["sharpe"])}
        cap = meta_all.get(c["centre"], {}).get("engine", {}).get("capacity_2021_2026", {})
        r5_bool = bool(cap.get("aum_max_5pct_usd", 0.0) >= common.CAPACITY_MIN_AUM_FLOAT)
        centre_book = cell_res[c["centre"]]["books"]["sweep_engine"]
        neigh_full_sharpe = neigh_median(neigh, lambda k: cell_res[k]["books"]["sweep_engine"]["G-FULL"]["sharpe"])
        neigh_full_dd = neigh_median(neigh, lambda k: cell_res[k]["books"]["sweep_engine"]["G-FULL"]["max_dd"])
        c["rule"] = {"engine": engine_rule, "stress": stress_rule, "R3": bool(stress_rule["R1"] and stress_rule["R2"]), "R4": r4,
                     "R5": {"aum_max_5pct_usd_2021_2026": cap.get("aum_max_5pct_usd"), "aum_p95_1pct_usd_2021_2026": cap.get("aum_p95_1pct_usd"), "pass": r5_bool},
                     "passes": bool(engine_rule["R1"] and engine_rule["R2"] and stress_rule["R1"] and stress_rule["R2"] and all(v["pass"] for v in r4.values()) and r5_bool),
                     "labels": {"vs_G3": margins(neigh, "sweep_engine", controls["G3"]), "vs_C_SPY": margins(neigh, "sweep_engine", controls["C_SPY"]),
                                "vs_C_MNA": margins(neigh, "sweep_engine", controls["C_MNA"], blocks=("G-P2", "G-P3")), "nosweep_vs_C_CASH0": margins(neigh, "nosweep_engine", c_cash0),
                                "owner_gate_centre": {"sharpe": centre_book["G-FULL"]["sharpe"], "max_dd": centre_book["G-FULL"]["max_dd"], "cagr": centre_book["G-FULL"]["cagr"],
                                                      "pass": bool(centre_book["G-FULL"]["sharpe"] >= common.OWNER_GATE_SHARPE_FLOAT and centre_book["G-FULL"]["max_dd"] >= common.OWNER_GATE_MAX_DD_FLOAT)},
                                "owner_gate_neighbourhood_median": {"sharpe": neigh_full_sharpe, "max_dd": neigh_full_dd, "pass": bool(neigh_full_sharpe >= common.OWNER_GATE_SHARPE_FLOAT and neigh_full_dd >= common.OWNER_GATE_MAX_DD_FLOAT)}},
                     "neigh_median_books_sweep_engine": {b: {m: neigh_median(neigh, lambda k: cell_res[k]["books"]["sweep_engine"][b][m]) for m in ("cagr", "sharpe", "max_dd")} for b in common.BOOK_BLOCK_DICT},
                     "centre_books_sweep_engine": centre_book}
        c["centre_standalone_sweep"] = cell_res[c["centre"]]["standalone_sweep"]
        c["neigh_median_standalone_sweep"] = {b: {m: neigh_median(neigh, lambda k: cell_res[k]["standalone_sweep"][b][m]) for m in ("cagr", "sharpe", "max_dd")} for b in common.BLOCK_DICT}
        wf_neigh = [k for k in c["walk_forward_neighbourhood"] if k in cell_res]
        c["walk_forward"] = {"centre_by_P1": c["walk_forward_centre"], "standalone_sweep_neigh_median_sharpe": {b: neigh_median(wf_neigh, lambda k: cell_res[k]["standalone_sweep"][b]["sharpe"]) for b in ("P2", "P3")},
                             "book_neigh_median_sharpe": {b: neigh_median(wf_neigh, lambda k: cell_res[k]["books"]["sweep_engine"][b]["sharpe"]) for b in ("G-P2", "G-P3")}, "C_BIL": {b: c_bil[b]["sharpe"] for b in ("G-P2", "G-P3")}}
    res["candidates"] = candidate_list
    # terminal-proceeds and stop sensitivities of V0 (labels)
    res["sensitivities_V0"] = {}
    for label_str, cell in cells_module.SENSITIVITY_DICT.items():
        if cell.key_str in cell_res:
            b = cell_res[cell.key_str]["books"]["sweep_engine"]
            res["sensitivities_V0"][label_str] = {"key": cell.key_str, "standalone_sweep_FULL": cell_res[cell.key_str]["standalone_sweep"]["FULL"], "standalone_nosweep_FULL": cell_res[cell.key_str]["standalone_nosweep"]["FULL"],
                                                  "book_margins_vs_C_BIL": {bk: b[bk]["sharpe"] - c_bil[bk]["sharpe"] for bk in common.BOOK_BLOCK_DICT}, "meta_v": meta_all.get(cell.key_str, {}).get("engine", {}).get("v")}
    passing = [c for c in candidate_list if c["rule"]["passes"]]
    if passing:
        best = max(passing, key=lambda c: c["rule"]["engine"]["min_margin"])
        res["decision"] = {"decision": "recommend a forward paper line first (follow-up to a seen version)", "recommended": {"stage": best["stage"], "centre": best["centre"], "min_margin": best["rule"]["engine"]["min_margin"]},
                           "passing": [(c["stage"], c["centre"]) for c in passing]}
    else:
        res["decision"] = {"decision": "no candidate passes; no further merger-arbitrage variant in this session", "passing": [],
                           "best_min_margin": max(((c["stage"], c["centre"], c["rule"]["engine"]["min_margin"]) for c in candidate_list), key=lambda x: x[2])}

    # ---------------------------------------------------------------- confidence labels
    book_full_df = pd.DataFrame(book_full_dict).dropna()
    grid_keys = [c.key_str for c in cells_module.grid_cells() if c.key_str in book_full_df]
    candidate_keys = [c["centre"] for c in candidate_list]
    res["multiplicity"] = {"book_configs_int": len(grid_keys), "reality_check_book_g_full_vs_C_BIL": trend_analyze.reality_check(book_full_df[grid_keys + ["C_BIL"]], "C_BIL", candidate_keys, common.SEED_INT)}
    fam_df = sweep_all[grid_keys].loc[common.BLOCK_DICT["FULL"][0]:common.BLOCK_DICT["FULL"][1]]
    fam_sharpe_vec = (fam_df.mean() / fam_df.std()).to_numpy()
    res["multiplicity"]["deflated_sharpe_standalone_sweep"] = {c["stage"]: trend_analyze.deflated_sharpe(fam_df[c["centre"]], fam_sharpe_vec, common.N_TRIALS_INT) for c in candidate_list}
    nosweep_fam_df = nosweep_all[grid_keys].loc[common.BLOCK_DICT["FULL"][0]:common.BLOCK_DICT["FULL"][1]]
    res["multiplicity"]["deflated_sharpe_standalone_nosweep"] = {c["stage"]: trend_analyze.deflated_sharpe(nosweep_fam_df[c["centre"]], (nosweep_fam_df.mean() / nosweep_fam_df.std()).to_numpy(), common.N_TRIALS_INT) for c in candidate_list}

    # ---------------------------------------------------------------- small account
    small_meta_path = OUT_PATH / "meta_V_ALL_25k.json"
    if small_meta_path.exists():
        small_meta = json.loads(small_meta_path.read_text())[V0_KEY]["engine"]
        big_meta = meta_all[V0_KEY]["engine"]
        res["small_account_V0"] = {"capital_usd": common.SMALL_CAPITAL_FLOAT,
                                   "commission_total_usd": small_meta["v"]["commission_total_usd"], "total_pnl_usd": small_meta["v"]["total_pnl_usd"],
                                   "commission_share_of_gross_pnl": small_meta["v"]["commission_total_usd"] / (small_meta["v"]["total_pnl_usd"] + small_meta["v"]["commission_total_usd"]) if (small_meta["v"]["total_pnl_usd"] + small_meta["v"]["commission_total_usd"]) != 0 else float("nan"),
                                   "commission_pct_nav_per_year": small_meta["commission_pct_nav_per_year"], "entries_total_int": small_meta["v"]["entries_total_int"], "queue_drops": small_meta["v"]["queue_drops"],
                                   "at_100k": {"commission_total_usd": big_meta["v"]["commission_total_usd"], "total_pnl_usd": big_meta["v"]["total_pnl_usd"],
                                               "commission_share_of_gross_pnl": big_meta["v"]["commission_total_usd"] / (big_meta["v"]["total_pnl_usd"] + big_meta["v"]["commission_total_usd"]), "commission_pct_nav_per_year": big_meta["commission_pct_nav_per_year"]}}
        small_ret = pod_returns("V_ALL_25k", "engine", bil_ser, False)
        if small_ret is not None:
            res["small_account_V0"]["standalone_nosweep_FULL_25k"] = common.metric_dict(small_ret[V0_KEY].loc[common.BLOCK_DICT["FULL"][0]:])

    # ---------------------------------------------------------------- tables
    rows = []
    for key_str, e in cell_res.items():
        m = meta_all.get(key_str, {}).get("engine", {})
        row = {"key": key_str, "beta_spy": e["beta_spy_full"], "corr_taa": e["corr_taa_2008_2026"], "corr_L": e["corr_L_full"], "corr_mna": e["corr_mna_2009_2026"]}
        for b, v in e["standalone_sweep"].items():
            row.update({f"sa_sweep_{b}_sharpe": v["sharpe"], f"sa_sweep_{b}_cagr": v["cagr"], f"sa_sweep_{b}_dd": v["max_dd"]})
        row["sa_nosweep_FULL_sharpe"], row["sa_nosweep_FULL_cagr"] = e["standalone_nosweep"]["FULL"]["sharpe"], e["standalone_nosweep"]["FULL"]["cagr"]
        for b, v in e["books"]["sweep_engine"].items():
            row.update({f"book_{b}_sharpe": v["sharpe"], f"book_{b}_dd": v["max_dd"], f"book_{b}_margin_vs_CBIL": v["sharpe"] - c_bil[b]["sharpe"]})
        for h, v in e["half_books_sweep_engine"].items():
            row[f"book_G-FULL_sharpe_{h}"] = v["G-FULL"]["sharpe"]
        vd = m.get("v", {})
        row.update({k: vd.get(k) for k in ("events_per_year", "confirmations_per_year", "entries_per_year", "entries_total_int", "positions_mean", "positions_max", "precision_terminal_within_252", "worst_episode_return")})
        row.update({k: m.get(k) for k in ("turnover_x_per_year", "mean_exposure", "holding_sessions_median", "terminal_pnl_share")})
        cap = m.get("capacity_2021_2026", {})
        row.update({"cap21_max_5pct_usd": cap.get("aum_max_5pct_usd"), "cap21_p95_1pct_usd": cap.get("aum_p95_1pct_usd")})
        rows.append(row)
    pd.DataFrame(rows).to_csv(OUT_PATH / "table_cells.csv", index=False)
    rule_rows = []
    for c in candidate_list:
        r = c["rule"]
        rule_rows.append({"stage": c["stage"], "centre": c["centre"], "plateau_full_sharpe_sweep": c["plateau_full_sharpe"], **{f"margin_{b}": v for b, v in r["engine"]["sharpe_margin"].items()}, "min_margin": r["engine"]["min_margin"],
                          **{f"dd_gap_pp_{b}": v for b, v in r["engine"]["dd_gap_pp"].items()}, "R1": r["engine"]["R1"], "R2": r["engine"]["R2"], "R3": r["R3"], **{f"R4_{h}": v["pass"] for h, v in r["R4"].items()},
                          "R5": r["R5"]["pass"], "passes": r["passes"], "beats_G3_all_blocks": r["labels"]["vs_G3"]["R1"], "beats_C_SPY_all_blocks": r["labels"]["vs_C_SPY"]["R1"], "beats_C_MNA_G-P2_G-P3": r["labels"]["vs_C_MNA"]["R1"],
                          "nosweep_beats_C_CASH0_all_blocks": r["labels"]["nosweep_vs_C_CASH0"]["R1"], "owner_gate_centre": r["labels"]["owner_gate_centre"]["pass"], "centre_book_sharpe": r["labels"]["owner_gate_centre"]["sharpe"]})
    pd.DataFrame(rule_rows).to_csv(OUT_PATH / "table_rule.csv", index=False)

    # ---------------------------------------------------------------- charts
    for stage_str, grid in grids.items():
        key_arr = np.array(grid["keys"], dtype=object)
        value_arr = np.array(grid["standalone_full_sharpe_sweep"])
        book_arr = np.vectorize(lambda k: cell_res[k]["books"]["sweep_engine"]["G-FULL"]["sharpe"] - c_bil["G-FULL"]["sharpe"] if k in cell_res else np.nan, otypes=[float])(key_arr)
        fig, axes = plt.subplots(1, 2, figsize=(11, 0.7 * len(grid["rows"]) + 2.2), facecolor=trend_analyze.SURFACE)
        trend_analyze.heatmap(axes[0], value_arr, grid["rows"], grid["cols"], "Standalone Sharpe FULL (with sweep)", seq_cmap, np.nanmin(value_arr), np.nanmax(value_arr), tuple(grid["anchor_pos"]), tuple(grid["centre_pos"]))
        trend_analyze.heatmap(axes[1], book_arr, grid["rows"], grid["cols"], f"Book G-FULL Sharpe minus C_BIL ({c_bil['G-FULL']['sharpe']:.3f})", div_cmap, -0.05, 0.05, tuple(grid["anchor_pos"]), tuple(grid["centre_pos"]), fmt="{:+.3f}")
        fig.suptitle(f"{stage_str}: black box = V0, dashed = plateau centre", fontsize=10, color=trend_analyze.TEXT_SECONDARY, x=0.01, ha="left")
        fig.tight_layout()
        fig.savefig(CHART_PATH / f"heatmap_{stage_str}.png", dpi=130)
        plt.close(fig)
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), facecolor=trend_analyze.SURFACE, sharex=True)
    series_list = [("C_BIL", c_bil_full_ser), ("G3", common.book_window_return_ser({"taa": taa_ser, "L": l_engine_ser}, common.G3_WEIGHT_DICT, *common.BOOK_BLOCK_DICT["G-FULL"])),
                   ("MNA slot", common.candidate_book_series(taa_ser, l_engine_ser, mna_ser))] + [(f"{c['stage']}: {c['centre']}", book_full_dict[c["centre"]]) for c in candidate_list]
    for (label_str, ser), color in zip(series_list, ["#0b0b0b", "#52514e", "#9ec5f4", "#2a78d6", "#eb6834"]):
        wealth_ser = (1 + ser.dropna()).cumprod()
        axes[0].plot(wealth_ser.index, wealth_ser, color=color, lw=1.6 if label_str in ("C_BIL", "G3") else 1.0, label=label_str)
        axes[1].plot(wealth_ser.index, wealth_ser / wealth_ser.cummax() - 1, color=color, lw=1.6 if label_str in ("C_BIL", "G3") else 1.0)
    axes[0].set_yscale("log")
    axes[0].set_title("Candidate books {TAA 0.5, L 0.25, X 0.25} versus C_BIL, G3 and the MNA slot, 2012-10-02..2026-08-19", fontsize=9.5, loc="left", color=trend_analyze.TEXT_PRIMARY)
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
    for name_str in ("pass1", "detection_stats"):
        path = common.CACHE_DIR_PATH / f"{name_str}.json"
        if path.exists():
            payload = json.loads(path.read_text())
            payload.pop("candidate_symbols", None)
            res[f"detection_{name_str}"] = payload
    common.write_json("results.json", res)
    common.log_progress(f"analysis done: {json.dumps(res['decision'], default=str)[:400]}")
    for c in candidate_list:
        r = c["rule"]
        print(c["stage"], "| centre", c["centre"], "| margins", {b: round(v, 4) for b, v in r["engine"]["sharpe_margin"].items()}, "| dd", {b: round(v, 2) for b, v in r["engine"]["dd_gap_pp"].items()},
              "| R3", r["R3"], "| R4", {h: (round(v["neigh_median_book_g_full_sharpe"], 4), v["pass"]) for h, v in r["R4"].items()}, "| R5", r["R5"]["pass"], "| passes", r["passes"])


if __name__ == "__main__":
    main()
