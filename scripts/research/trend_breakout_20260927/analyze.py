"""Analysis for the trend / breakout study (research only): applies PREREG section 8 mechanically.

Reads the grid outputs written by run_study.py, builds the official pod-model books (annual reset, each window its own
run), computes plateaus, candidates, the frozen rule R1-R5 per candidate-role, confidence labels, walk-forward, timing
luck, capacity, stop realism, terminal economics and the daily-rebalanced sensitivity; writes results.json, CSV tables
and charts/*.png.
"""

from __future__ import annotations

import dataclasses
import json
import pickle

import numpy as np
import pandas as pd
from scipy import stats

from trend_breakout_20260927 import cells as cells_module
from trend_breakout_20260927 import common

OUT_PATH = common.RESULTS_DIR_PATH
CHART_PATH = common.CHART_DIR_PATH
L_KEY = cells_module.L_REFERENCE_CELL.key_str
B0_KEY = cells_module.B0_CELL.key_str
C0_KEY = cells_module.C0_CELL.key_str
FAMILY_PRIMARY_DICT = {"A": "NDX", "B": "SP500", "C": "NDX"}
FAMILY_OTHER_DICT = {"A": ("SP500", "R1000"), "B": ("NDX", "R1000"), "C": ("SP500", "R1000")}
FAMILY_ROLE_DICT = {"A": ("replacement",), "B": ("replacement", "addition"), "C": ("replacement", "addition")}
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
DIVERGING = ["#8f2423", "#c93b3a", "#e34948", "#f3a9a8", "#f0efec", "#9ec5f4", "#3987e5", "#256abf", "#184f95"]
TEXT_PRIMARY, TEXT_SECONDARY, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"


# ----------------------------------------------------------------------------------------------------------------------
# loading
# ----------------------------------------------------------------------------------------------------------------------
def load_returns(family_str: str, universe_str: str, cost_str: str, offsets_bool: bool = False) -> pd.DataFrame | None:
    path = OUT_PATH / f"returns_{family_str}_{universe_str}{'_offsets' if offsets_bool else ''}_{cost_str}.parquet"
    if not path.exists():
        return None
    frame_df = pd.read_parquet(path)
    frame_df.index = pd.to_datetime(frame_df.index)
    return frame_df.loc[:common.END_TS]


def load_meta(family_str: str, universe_str: str) -> dict:
    path = OUT_PATH / f"meta_{family_str}_{universe_str}.json"
    return json.loads(path.read_text()) if path.exists() else {}


def load_trades(family_str: str, universe_str: str) -> dict:
    path = OUT_PATH / f"trades_{family_str}_{universe_str}.pkl"
    if not path.exists():
        return {}
    with open(path, "rb") as file_obj:
        return pickle.load(file_obj)


# ----------------------------------------------------------------------------------------------------------------------
# books
# ----------------------------------------------------------------------------------------------------------------------
def role_legs(role_str: str, taa_ser: pd.Series, l_ser: pd.Series, t_ser: pd.Series | None) -> tuple[dict, dict]:
    weight_dict = dict(common.ROLE_WEIGHT_DICT[role_str])
    leg_dict = {"taa": taa_ser, "L": l_ser}
    if "T" in weight_dict:
        leg_dict["T"] = t_ser
    return leg_dict, weight_dict


def book_blocks(role_str: str, taa_ser: pd.Series, l_ser: pd.Series, t_ser: pd.Series | None, daily_bool: bool = False) -> dict:
    leg_dict, weight_dict = role_legs(role_str, taa_ser, l_ser, t_ser)
    return common.book_metrics_by_block(leg_dict, weight_dict, daily_bool)


def book_series(role_str: str, taa_ser: pd.Series, l_ser: pd.Series, t_ser: pd.Series | None, block_str: str = "G-FULL", daily_bool: bool = False) -> pd.Series:
    leg_dict, weight_dict = role_legs(role_str, taa_ser, l_ser, t_ser)
    start_str, end_str = common.BOOK_BLOCK_DICT[block_str]
    return common.book_window_return_ser(leg_dict, weight_dict, start_str, end_str, daily_bool)


# ----------------------------------------------------------------------------------------------------------------------
# plateaus
# ----------------------------------------------------------------------------------------------------------------------
def neighbour_list(i: int, j: int, shape: tuple, use_row_bool: bool, use_col_bool: bool) -> list[tuple[int, int]]:
    row_range = range(max(0, i - 1), min(shape[0], i + 2)) if use_row_bool else [i]
    col_range = range(max(0, j - 1), min(shape[1], j + 2)) if use_col_bool else [j]
    return [(r, c) for r in row_range for c in col_range]


def plateau_matrix(key_arr: np.ndarray, value_fn, use_row_bool: bool, use_col_bool: bool) -> tuple[np.ndarray, np.ndarray]:
    value_arr = np.vectorize(value_fn, otypes=[float])(key_arr)
    plateau_arr = np.full(key_arr.shape, np.nan)
    for i in range(key_arr.shape[0]):
        for j in range(key_arr.shape[1]):
            plateau_arr[i, j] = np.nanmedian([value_arr[r, c] for r, c in neighbour_list(i, j, key_arr.shape, use_row_bool, use_col_bool)])
    return value_arr, plateau_arr


def family_a_matrix() -> tuple[list, list, np.ndarray]:
    row_list = [f"{type_str}|{policy_str}" for type_str, policy_str in cells_module.A_ROW_TUPLE]
    col_list = [1, 2, 3, 4]  # level index, tight to loose
    key_arr = np.empty((len(row_list), 4), dtype=object)
    for i, (type_str, policy_str) in enumerate(cells_module.A_ROW_TUPLE):
        for j, cell in enumerate(cells_module.family_a_row_cells(type_str, policy_str)):
            key_arr[i, j] = cell.key_str
    return row_list, col_list, key_arr


def family_b_matrix(stage_str: str) -> tuple[list, list, np.ndarray]:
    stage_list = cells_module.family_b_stage_dict()[stage_str]
    row_list = list(dict.fromkeys(row_col[0] for row_col, _ in stage_list))
    col_list = list(dict.fromkeys(row_col[1] for row_col, _ in stage_list))
    key_arr = np.empty((len(row_list), len(col_list)), dtype=object)
    for (row_label, col_label), cell in stage_list:
        key_arr[row_list.index(row_label), col_list.index(col_label)] = cell.key_str
    return row_list, col_list, key_arr


# ----------------------------------------------------------------------------------------------------------------------
# multiplicity
# ----------------------------------------------------------------------------------------------------------------------
def stationary_boot_index(n_int: int, rng: np.random.Generator, draws_int: int) -> np.ndarray:
    idx_arr = np.empty((draws_int, n_int), dtype=np.int32)
    idx_arr[:, 0] = rng.integers(0, n_int, draws_int)
    new_block_arr = rng.random((draws_int, n_int)) < 1.0 / common.BOOT_BLOCK_FLOAT
    start_arr = rng.integers(0, n_int, (draws_int, n_int))
    for j in range(1, n_int):
        idx_arr[:, j] = np.where(new_block_arr[:, j], start_arr[:, j], (idx_arr[:, j - 1] + 1) % n_int)
    return idx_arr


def sharpe_cols(arr: np.ndarray) -> np.ndarray:
    return arr.mean(axis=0) / arr.std(axis=0, ddof=1) * np.sqrt(252.0)


def reality_check(return_df: pd.DataFrame, benchmark_key: str, paired_key_list: list[str], seed_int: int) -> dict:
    """White's Reality Check on annualised Sharpe differences versus the benchmark column, plus paired p-values."""
    key_list = [k for k in return_df.columns if k != benchmark_key]
    x_arr = return_df[key_list].to_numpy()
    b_vec = return_df[benchmark_key].to_numpy()
    obs_diff_vec = sharpe_cols(x_arr) - sharpe_cols(b_vec[:, None])[0]
    idx_arr = stationary_boot_index(len(return_df), np.random.default_rng(seed_int), common.BOOT_N_INT)
    max_centered_vec = np.empty(common.BOOT_N_INT)
    col_pos_dict = {k: key_list.index(k) for k in paired_key_list if k in key_list}
    paired_diff_dict = {k: np.empty(common.BOOT_N_INT) for k in col_pos_dict}
    for d in range(common.BOOT_N_INT):
        idx_vec = idx_arr[d]
        diff_vec = sharpe_cols(x_arr[idx_vec]) - sharpe_cols(b_vec[idx_vec][:, None])[0]
        max_centered_vec[d] = np.max(diff_vec - obs_diff_vec)
        for k, pos in col_pos_dict.items():
            paired_diff_dict[k][d] = diff_vec[pos]
    best_pos = int(np.argmax(obs_diff_vec))
    return {
        "configs_int": len(key_list),
        "best_config": key_list[best_pos],
        "best_obs_sharpe_diff": float(obs_diff_vec[best_pos]),
        "reality_check_p": float(np.mean(max_centered_vec >= obs_diff_vec[best_pos])),
        "share_configs_above_benchmark": float(np.mean(obs_diff_vec > 0)),
        "paired": {
            k: {"obs_diff": float(obs_diff_vec[col_pos_dict[k]]), "p_one_sided": float(np.mean(v <= 0)), "ci90": [float(np.quantile(v, 0.05)), float(np.quantile(v, 0.95))]}
            for k, v in paired_diff_dict.items()
        },
    }


def deflated_sharpe(return_ser: pd.Series, sharpe_daily_all_vec: np.ndarray, trials_int: int = common.N_TRIALS_INT) -> dict:
    gamma_float = 0.5772156649
    var_float = float(np.var(sharpe_daily_all_vec, ddof=1)) if len(sharpe_daily_all_vec) > 1 else float("nan")
    sr0_float = np.sqrt(var_float) * ((1 - gamma_float) * stats.norm.ppf(1 - 1 / trials_int) + gamma_float * stats.norm.ppf(1 - 1 / (trials_int * np.e)))
    x_vec = return_ser.dropna().to_numpy()
    sr_float = x_vec.mean() / x_vec.std(ddof=1)
    skew_float, kurt_float = float(stats.skew(x_vec)), float(stats.kurtosis(x_vec, fisher=False))
    z_float = (sr_float - sr0_float) * np.sqrt(len(x_vec) - 1) / np.sqrt(1 - skew_float * sr_float + (kurt_float - 1) / 4 * sr_float**2)
    return {"sharpe_ann": float(sr_float * np.sqrt(252)), "sr0_ann": float(sr0_float * np.sqrt(252)), "dsr_prob": float(stats.norm.cdf(z_float)),
            "trials_int": trials_int, "cross_trial_sd_ann": float(np.sqrt(var_float) * np.sqrt(252))}


# ----------------------------------------------------------------------------------------------------------------------
# charts
# ----------------------------------------------------------------------------------------------------------------------
def heatmap(ax, value_arr, row_list, col_list, title_str, cmap, vmin, vmax, anchor_pos=None, centre_pos=None, fmt="{:.2f}", fontsize=8):
    from matplotlib.patches import Rectangle

    image = ax.imshow(value_arr.astype(float), cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    for i in range(value_arr.shape[0]):
        for j in range(value_arr.shape[1]):
            v = float(value_arr[i, j])
            if not np.isfinite(v):
                continue
            rgba = image.cmap(image.norm(v))
            lum = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
            ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=fontsize, color="white" if lum < 0.5 else TEXT_PRIMARY)
    ax.set_xticks(range(len(col_list)), [str(c) for c in col_list], fontsize=8, color=TEXT_SECONDARY)
    ax.set_yticks(range(len(row_list)), [str(r) for r in row_list], fontsize=8, color=TEXT_SECONDARY)
    ax.set_title(title_str, fontsize=9.5, color=TEXT_PRIMARY, loc="left")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    if anchor_pos is not None:
        ax.add_patch(Rectangle((anchor_pos[1] - 0.5, anchor_pos[0] - 0.5), 1, 1, fill=False, ec=TEXT_PRIMARY, lw=2.0))
    if centre_pos is not None:
        ax.add_patch(Rectangle((centre_pos[1] - 0.42, centre_pos[0] - 0.42), 0.84, 0.84, fill=False, ec="#eb6834", lw=2.0, ls="--"))
    return image


def style_axis(ax) -> None:
    ax.grid(axis="y", color="#e4e3df", lw=0.6)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(labelsize=8, colors=TEXT_SECONDARY)


# ----------------------------------------------------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------------------------------------------------
def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    CHART_PATH.mkdir(parents=True, exist_ok=True)
    seq_cmap = LinearSegmentedColormap.from_list("seq", BLUE_RAMP)
    div_cmap = LinearSegmentedColormap.from_list("div", DIVERGING)
    common.log_progress("analysis start")

    taa_ser = common.load_taa_ser()
    stored_l_ser = common.load_stored_l_ser()
    ret_dict: dict = {}
    for family_str in ("A", "B", "C"):
        for universe_str in ("NDX", "SP500", "R1000"):
            for cost_str in ("engine", "stress", "haircut"):
                frame_df = load_returns(family_str, universe_str, cost_str)
                if frame_df is not None:
                    ret_dict[(family_str, universe_str, cost_str)] = frame_df
    meta_dict = {(f, u): load_meta(f, u) for f in ("A", "B", "C") for u in ("NDX", "SP500", "R1000")}
    a_ndx_engine_df = ret_dict[("A", "NDX", "engine")]
    a_ndx_stress_df = ret_dict[("A", "NDX", "stress")]
    l_engine_ser = a_ndx_engine_df[L_KEY].rename("L")
    l_stress_ser = a_ndx_stress_df[L_KEY].rename("L")
    res: dict = {"prereg": "docs/research/TREND_BREAKOUT_PREREG_20260927.md", "end_ts": str(common.END_TS.date())}

    # ---------------------------------------------------------------------------------------------- baseline G3
    res["g3_baseline_replica_L"] = {
        "engine": book_blocks("G3", taa_ser, l_engine_ser, None),
        "stress": book_blocks("G3", taa_ser, l_stress_ser, None),
        "daily_rebalanced_engine": book_blocks("G3", taa_ser, l_engine_ser, None, daily_bool=True),
    }
    res["g3_reference_stored_L"] = {"engine": book_blocks("G3", taa_ser, stored_l_ser, None)}
    res["legs"] = {
        "L_replica_standalone": common.window_metrics(l_engine_ser, common.BLOCK_DICT),
        "L_stored_standalone": common.window_metrics(stored_l_ser, common.BLOCK_DICT),
        "L_replica_stress_standalone": common.window_metrics(l_stress_ser, common.BLOCK_DICT),
        "TAA_standalone_book_blocks": common.window_metrics(taa_ser, common.BOOK_BLOCK_DICT),
        "corr_L_replica_vs_stored": float(l_engine_ser.corr(stored_l_ser.reindex(l_engine_ser.index))),
    }
    g3_engine = res["g3_baseline_replica_L"]["engine"]
    g3_stress = res["g3_baseline_replica_L"]["stress"]

    # ---------------------------------------------------------------------------------------------- per-cell metrics
    cell_res: dict = {}
    book_full_series_dict: dict[str, pd.Series] = {"G3": book_series("G3", taa_ser, l_engine_ser, None)}
    for family_str in ("A", "B", "C"):
        primary_str = FAMILY_PRIMARY_DICT[family_str]
        engine_df = ret_dict.get((family_str, primary_str, "engine"))
        stress_df = ret_dict.get((family_str, primary_str, "stress"))
        haircut_df = ret_dict.get((family_str, primary_str, "haircut"))
        if engine_df is None:
            continue
        for key_str in engine_df.columns:
            if key_str == L_KEY or key_str.startswith("C|A0_ref"):
                continue
            t_engine_ser = engine_df[key_str]
            entry = {
                "family": family_str,
                "standalone": common.window_metrics(t_engine_ser, common.BLOCK_DICT),
                "standalone_stress": common.window_metrics(stress_df[key_str], common.BLOCK_DICT) if stress_df is not None and key_str in stress_df else None,
                "standalone_haircut": common.window_metrics(haircut_df[key_str], common.BLOCK_DICT) if haircut_df is not None and key_str in haircut_df else None,
                "corr_with_taa_2012_2026": float(t_engine_ser.loc["2012-10-02":].corr(taa_ser.loc["2012-10-02":])),
                "corr_with_L_2000_2026": float(t_engine_ser.corr(l_engine_ser)),
                "other_universes_standalone_full": {},
                "books": {},
            }
            for universe_str in FAMILY_OTHER_DICT[family_str]:
                other_df = ret_dict.get((family_str, universe_str, "engine"))
                if other_df is not None and key_str in other_df:
                    entry["other_universes_standalone_full"][universe_str] = common.metric_dict(other_df[key_str].loc[common.BLOCK_DICT["FULL"][0]:])
            for role_str in FAMILY_ROLE_DICT[family_str]:
                entry["books"][role_str] = {
                    "engine": book_blocks(role_str, taa_ser, l_engine_ser, t_engine_ser),
                    "stress": book_blocks(role_str, taa_ser, l_stress_ser, stress_df[key_str]) if stress_df is not None and key_str in stress_df else None,
                }
                book_full_series_dict[f"{key_str}||{role_str}"] = book_series(role_str, taa_ser, l_engine_ser, t_engine_ser)
            cell_res[key_str] = entry
    res["cells"] = cell_res
    meta_primary = {k: meta_dict[(v["family"], FAMILY_PRIMARY_DICT[v["family"]])].get(k, {}) for k, v in cell_res.items()}
    res["cell_meta_primary"] = meta_primary
    res["cell_meta_other_universes"] = {(f"{f}_{u}"): meta_dict[(f, u)] for f in ("A", "B", "C") for u in ("NDX", "SP500", "R1000") if meta_dict[(f, u)]}

    # ---------------------------------------------------------------------------------------------- plateaus and candidates
    def sa_full(key_str: str) -> float:
        return cell_res[key_str]["standalone"]["FULL"]["sharpe"] if key_str in cell_res else float("nan")

    def sa_p1(key_str: str) -> float:
        return cell_res[key_str]["standalone"]["P1"]["sharpe"] if key_str in cell_res else float("nan")

    candidate_list: list[dict] = []
    # family A: per row, along the level axis; ties -> looser level
    a_rows, a_cols, a_key_arr = family_a_matrix()
    a_value_arr, a_plateau_arr = plateau_matrix(a_key_arr, sa_full, False, True)
    _, a_plateau_p1_arr = plateau_matrix(a_key_arr, sa_p1, False, True)
    res["family_A_grid"] = {"rows": a_rows, "cols": [f"level {c}" for c in a_cols], "keys": a_key_arr.tolist(), "standalone_full_sharpe": a_value_arr.tolist(),
                            "plateau_full_sharpe": a_plateau_arr.tolist(), "L_standalone_full_sharpe": res["legs"]["L_replica_standalone"]["FULL"]["sharpe"]}
    for i, row_str in enumerate(a_rows):
        j_best = int(max(range(4), key=lambda j: (round(a_plateau_arr[i, j], 12), j)))
        j_wf = int(max(range(4), key=lambda j: (round(a_plateau_p1_arr[i, j], 12), j)))
        candidate_list.append({"family": "A", "stage": f"A_row_{row_str}", "centre": a_key_arr[i, j_best], "neighbourhood": [a_key_arr[i, c] for c in range(max(0, j_best - 1), min(4, j_best + 2))],
                               "plateau_full_sharpe": float(a_plateau_arr[i, j_best]), "walk_forward_centre": a_key_arr[i, j_wf],
                               "walk_forward_neighbourhood": [a_key_arr[i, c] for c in range(max(0, j_wf - 1), min(4, j_wf + 2))], "grid_pos": (i, j_best), "wf_grid_pos": (i, j_wf)})
    # family B stages; ties -> nearest B0
    res["family_B_grids"] = {}
    for stage_str, (use_row_bool, use_col_bool) in (("B1_entry_exit", (True, True)), ("B2_slots_rank", (False, True))):
        row_list, col_list, key_arr = family_b_matrix(stage_str)
        b0_pos = tuple(int(x) for x in np.argwhere(key_arr == B0_KEY)[0])
        value_arr, plateau_arr = plateau_matrix(key_arr, sa_full, use_row_bool, use_col_bool)
        _, plateau_p1_arr = plateau_matrix(key_arr, sa_p1, use_row_bool, use_col_bool)

        def best_pos(plat_arr):
            return max(((i, j) for i in range(key_arr.shape[0]) for j in range(key_arr.shape[1])),
                       key=lambda p: (round(plat_arr[p], 12), -(abs(p[0] - b0_pos[0]) + abs(p[1] - b0_pos[1]))))

        centre_pos = best_pos(plateau_arr)
        wf_pos = best_pos(plateau_p1_arr)
        res["family_B_grids"][stage_str] = {"rows": [str(r) for r in row_list], "cols": [str(c) for c in col_list], "keys": key_arr.tolist(),
                                            "standalone_full_sharpe": value_arr.tolist(), "plateau_full_sharpe": plateau_arr.tolist(), "b0_pos": b0_pos}
        candidate_list.append({"family": "B", "stage": stage_str, "centre": key_arr[centre_pos], "neighbourhood": [key_arr[p] for p in neighbour_list(*centre_pos, key_arr.shape, use_row_bool, use_col_bool)],
                               "plateau_full_sharpe": float(plateau_arr[centre_pos]), "walk_forward_centre": key_arr[wf_pos],
                               "walk_forward_neighbourhood": [key_arr[p] for p in neighbour_list(*wf_pos, key_arr.shape, use_row_bool, use_col_bool)], "grid_pos": centre_pos, "wf_grid_pos": wf_pos})
    # family C: C0 pre-declared
    if C0_KEY in cell_res:
        candidate_list.append({"family": "C", "stage": "C_anchor", "centre": C0_KEY, "neighbourhood": [c.key_str for c in cells_module.C0_NEIGHBOURHOOD_TUPLE],
                               "plateau_full_sharpe": float(np.nanmedian([sa_full(c.key_str) for c in cells_module.C0_NEIGHBOURHOOD_TUPLE])),
                               "walk_forward_centre": C0_KEY, "walk_forward_neighbourhood": [c.key_str for c in cells_module.C0_NEIGHBOURHOOD_TUPLE], "grid_pos": None, "wf_grid_pos": None})

    # ---------------------------------------------------------------------------------------------- the rule
    def other_universe_reference(family_str: str, universe_str: str) -> float:
        frame_df = ret_dict.get((family_str, universe_str, "engine"))
        if frame_df is None:
            return float("nan")
        if family_str == "A":
            ref_key = L_KEY
        elif family_str == "B":
            ref_key = B0_KEY
        else:
            ref_key = "C|A0_ref|REL25"
        return common.metric_dict(frame_df[ref_key].loc[common.BLOCK_DICT["FULL"][0]:])["sharpe"] if ref_key in frame_df else float("nan")

    def apply_rule(candidate: dict, role_str: str) -> dict:
        neigh = [k for k in candidate["neighbourhood"] if k in cell_res]
        family_str = candidate["family"]

        def margins(cost_str: str, g3_blocks: dict) -> dict:
            margin_dict = {b: float(np.nanmedian([cell_res[k]["books"][role_str][cost_str][b]["sharpe"] for k in neigh]) - g3_blocks[b]["sharpe"]) for b in common.RULE_BLOCK_TUPLE}
            dd_gap_dict = {b: float(np.nanmedian([cell_res[k]["books"][role_str][cost_str][b]["max_dd"] for k in neigh]) - g3_blocks[b]["max_dd"]) for b in ("G-FULL", "G-LONG")}
            return {"sharpe_margin_vs_G3": margin_dict, "min_margin": min(margin_dict.values()), "dd_gap_vs_G3_pp": {b: v * 100 for b, v in dd_gap_dict.items()},
                    "R1": bool(all(m > 0 for m in margin_dict.values())), "R2": bool(all(v >= -common.DD_TOLERANCE_FLOAT for v in dd_gap_dict.values()))}

        engine_rule = margins("engine", g3_engine)
        stress_rule = margins("stress", g3_stress)
        r4_dict = {}
        for universe_str in FAMILY_OTHER_DICT[family_str]:
            values = [cell_res[k]["other_universes_standalone_full"].get(universe_str, {}).get("sharpe", np.nan) for k in neigh]
            neigh_float = float(np.nanmedian(values)) if np.isfinite(values).any() else float("nan")
            ref_float = other_universe_reference(family_str, universe_str)
            trivial_bool = family_str == "B" and candidate["centre"] == B0_KEY
            r4_dict[universe_str] = {"neigh_median_sharpe": neigh_float, "reference_sharpe": ref_float, "pass": bool(trivial_bool or (np.isfinite(neigh_float) and np.isfinite(ref_float) and neigh_float >= ref_float))}
        capacity = meta_primary.get(candidate["centre"], {}).get("engine", {}).get("capacity_2021_2026", {})
        r5_bool = bool(capacity.get("aum_max_5pct_usd", 0.0) >= common.CAPACITY_MIN_AUM_FLOAT)
        centre_book = cell_res[candidate["centre"]]["books"][role_str]["engine"]["G-FULL"]
        neigh_full_sharpe = float(np.nanmedian([cell_res[k]["books"][role_str]["engine"]["G-FULL"]["sharpe"] for k in neigh]))
        neigh_full_dd = float(np.nanmedian([cell_res[k]["books"][role_str]["engine"]["G-FULL"]["max_dd"] for k in neigh]))
        passes_bool = bool(engine_rule["R1"] and engine_rule["R2"] and stress_rule["R1"] and stress_rule["R2"] and all(v["pass"] for v in r4_dict.values()) and r5_bool)
        return {
            "role": role_str, "engine": engine_rule, "stress": stress_rule, "R3": bool(stress_rule["R1"] and stress_rule["R2"]), "R4": r4_dict,
            "R5": {"aum_max_5pct_usd_2021_2026": capacity.get("aum_max_5pct_usd"), "aum_p95_1pct_usd_2021_2026": capacity.get("aum_p95_1pct_usd"), "pass": r5_bool},
            "passes": passes_bool,
            "owner_gate_centre": {"sharpe": centre_book["sharpe"], "max_dd": centre_book["max_dd"], "cagr": centre_book["cagr"],
                                  "pass": bool(centre_book["sharpe"] >= common.OWNER_GATE_SHARPE_FLOAT and centre_book["max_dd"] >= common.OWNER_GATE_MAX_DD_FLOAT)},
            "owner_gate_neighbourhood_median": {"sharpe": neigh_full_sharpe, "max_dd": neigh_full_dd,
                                                "pass": bool(neigh_full_sharpe >= common.OWNER_GATE_SHARPE_FLOAT and neigh_full_dd >= common.OWNER_GATE_MAX_DD_FLOAT)},
            "neigh_median_books_engine": {b: {m: float(np.nanmedian([cell_res[k]["books"][role_str]["engine"][b][m] for k in neigh])) for m in ("cagr", "sharpe", "max_dd")} for b in common.BOOK_BLOCK_DICT},
            "centre_books_engine": cell_res[candidate["centre"]]["books"][role_str]["engine"],
        }

    for candidate in candidate_list:
        candidate["rule_by_role"] = {role_str: apply_rule(candidate, role_str) for role_str in FAMILY_ROLE_DICT[candidate["family"]]}
        candidate["centre_standalone"] = cell_res[candidate["centre"]]["standalone"]
        candidate["neigh_median_standalone"] = {b: {m: float(np.nanmedian([cell_res[k]["standalone"][b][m] for k in candidate["neighbourhood"] if k in cell_res])) for m in ("cagr", "sharpe", "max_dd")} for b in common.BLOCK_DICT}
        # walk-forward: candidate re-chosen on P1 plateau; P2, P3 standalone and G-P2, G-P3 books (neighbourhood medians)
        wf_neigh = [k for k in candidate["walk_forward_neighbourhood"] if k in cell_res]
        candidate["walk_forward"] = {
            "centre_by_P1": candidate["walk_forward_centre"],
            "standalone_neigh_median_sharpe": {b: float(np.nanmedian([cell_res[k]["standalone"][b]["sharpe"] for k in wf_neigh])) for b in ("P2", "P3")},
            "book_neigh_median_sharpe": {role_str: {b: float(np.nanmedian([cell_res[k]["books"][role_str]["engine"][b]["sharpe"] for k in wf_neigh])) for b in ("G-P2", "G-P3")} for role_str in FAMILY_ROLE_DICT[candidate["family"]]},
            "G3": {b: g3_engine[b]["sharpe"] for b in ("G-P2", "G-P3")},
            "L_standalone": {b: res["legs"]["L_replica_standalone"][b]["sharpe"] for b in ("P2", "P3")},
        }
    res["candidates"] = candidate_list

    # decision
    passing_list = [(c["stage"], role_str) for c in candidate_list for role_str, rule in c["rule_by_role"].items() if rule["passes"]]
    recommended_list = []
    for c in candidate_list:
        roles_passed = [r for r, rule in c["rule_by_role"].items() if rule["passes"]]
        if roles_passed:
            recommended_list.append({"stage": c["stage"], "centre": c["centre"], "role": "addition" if "addition" in roles_passed else roles_passed[0], "roles_passed": roles_passed})
    shadow = None
    if not passing_list:
        pair_list = [(c, role_str, rule) for c in candidate_list for role_str, rule in c["rule_by_role"].items()]
        eligible_list = [p for p in pair_list if p[2]["engine"]["R2"] and p[2]["R5"]["pass"]]
        flagged_bool = len(eligible_list) == 0
        pool_list = eligible_list if eligible_list else pair_list
        c, role_str, rule = max(pool_list, key=lambda p: p[2]["engine"]["min_margin"])
        shadow = {"stage": c["stage"], "centre": c["centre"], "role": role_str, "min_margin": rule["engine"]["min_margin"], "flagged_no_pair_satisfies_R2_and_R5": flagged_bool}
    res["decision"] = {"passing_stage_roles": passing_list, "recommended": recommended_list, "decision": "keep G3" if not passing_list else "recommend the passing candidate(s)", "shadow_line": shadow}

    # ---------------------------------------------------------------------------------------------- confidence labels
    book_full_df = pd.DataFrame(book_full_series_dict).dropna()
    config_key_list = [k for k in book_full_df.columns if k != "G3"]
    candidate_pair_keys = [f"{c['centre']}||{role_str}" for c in candidate_list for role_str in FAMILY_ROLE_DICT[c["family"]]]
    res["multiplicity"] = {
        "book_configs_int": len(config_key_list),
        "reality_check_book_g_full_all": reality_check(book_full_df, "G3", candidate_pair_keys, common.SEED_INT),
        "reality_check_book_g_full_family_A": reality_check(book_full_df[[k for k in config_key_list if k.startswith("A|")] + ["G3"]], "G3", [k for k in candidate_pair_keys if k.startswith("A|")], common.SEED_INT + 1),
    }
    a_sa_df = a_ndx_engine_df.loc[common.BLOCK_DICT["FULL"][0]:].dropna()
    res["multiplicity"]["reality_check_standalone_A_vs_L"] = reality_check(a_sa_df, L_KEY, [c["centre"] for c in candidate_list if c["family"] == "A"], common.SEED_INT + 2)
    book_sharpe_daily_vec = (book_full_df[config_key_list].mean() / book_full_df[config_key_list].std()).to_numpy()
    dsr_dict = {}
    for c in candidate_list:
        family_keys = [k for k, v in cell_res.items() if v["family"] == c["family"]]
        family_df = ret_dict[(c["family"], FAMILY_PRIMARY_DICT[c["family"]], "engine")][family_keys].loc[common.BLOCK_DICT["FULL"][0]:]
        family_sharpe_daily_vec = (family_df.mean() / family_df.std()).to_numpy()
        dsr_dict[c["stage"]] = {
            "standalone": deflated_sharpe(family_df[c["centre"]], family_sharpe_daily_vec),
            "book": {role_str: deflated_sharpe(book_full_df[f"{c['centre']}||{role_str}"], book_sharpe_daily_vec) for role_str in FAMILY_ROLE_DICT[c["family"]]},
        }
    dsr_dict["L_standalone_same_haircut_A"] = deflated_sharpe(a_sa_df[L_KEY], (a_sa_df[[k for k in a_sa_df.columns if k != L_KEY]].mean() / a_sa_df[[k for k in a_sa_df.columns if k != L_KEY]].std()).to_numpy())
    dsr_dict["G3_book_same_haircut"] = deflated_sharpe(book_full_df["G3"], book_sharpe_daily_vec)
    res["multiplicity"]["deflated_sharpe"] = dsr_dict

    # ---------------------------------------------------------------------------------------------- timing luck (A)
    offsets_df = load_returns("A", "NDX", "engine", offsets_bool=True)
    luck_dict: dict = {}
    if offsets_df is not None:
        all_df = pd.concat([a_ndx_engine_df, offsets_df], axis=1)
        for label_str, base_key in [("L", L_KEY)] + [(c["stage"], c["centre"]) for c in candidate_list if c["family"] == "A"]:
            prefix_str = base_key.rsplit("|k", 1)[0]
            key_by_offset = {k_int: f"{prefix_str}|k{k_int:+d}" for k_int in cells_module.OFFSET_TUPLE}
            sa_vec = np.array([common.metric_dict(all_df[key_by_offset[k]].loc[common.BLOCK_DICT["FULL"][0]:])["sharpe"] for k in cells_module.OFFSET_TUPLE])
            g_vec = np.array([book_blocks("replacement", taa_ser, l_engine_ser, all_df[key_by_offset[k]])["G-FULL"]["sharpe"] if base_key != L_KEY
                              else book_blocks("G3", taa_ser, all_df[key_by_offset[k]], None)["G-FULL"]["sharpe"] for k in cells_module.OFFSET_TUPLE])
            k0 = cells_module.OFFSET_TUPLE.index(0)
            luck_dict[label_str] = {
                "centre_key": base_key,
                "standalone_full": {"k0": float(sa_vec[k0]), "min": float(sa_vec.min()), "median": float(np.median(sa_vec)), "max": float(sa_vec.max()), "k0_percentile": float(np.mean(sa_vec <= sa_vec[k0])), "by_offset": sa_vec.tolist()},
                "book_g_full": {"k0": float(g_vec[k0]), "min": float(g_vec.min()), "median": float(np.median(g_vec)), "max": float(g_vec.max()), "k0_percentile": float(np.mean(g_vec <= g_vec[k0])), "by_offset": g_vec.tolist()},
            }
        # stop-vs-L at the same offset: share of offsets where the candidate beats L standalone / G3 in the book
        for label_str in list(luck_dict):
            if label_str == "L":
                continue
            luck_dict[label_str]["beats_L_share_standalone"] = float(np.mean(np.array(luck_dict[label_str]["standalone_full"]["by_offset"]) > np.array(luck_dict["L"]["standalone_full"]["by_offset"])))
            luck_dict[label_str]["beats_G3_share_book"] = float(np.mean(np.array(luck_dict[label_str]["book_g_full"]["by_offset"]) > np.array(luck_dict["L"]["book_g_full"]["by_offset"])))
    res["timing_luck_A"] = luck_dict

    # ---------------------------------------------------------------------------------------------- cross-universe
    cross_dict = {}
    for family_str in ("A", "B", "C"):
        primary_str = FAMILY_PRIMARY_DICT[family_str]
        primary_df = ret_dict.get((family_str, primary_str, "engine"))
        if primary_df is None:
            continue
        fam_dict = {"primary": primary_str, "by_universe": {}, "spearman_vs_primary": {}}
        for universe_str in ("NDX", "SP500", "R1000"):
            frame_df = ret_dict.get((family_str, universe_str, "engine"))
            if frame_df is None:
                continue
            fam_dict["by_universe"][universe_str] = {k: common.metric_dict(frame_df[k].loc[common.BLOCK_DICT["FULL"][0]:]) for k in frame_df.columns}
        primary_keys = [k for k in primary_df.columns]
        for universe_str, table in fam_dict["by_universe"].items():
            if universe_str == primary_str:
                continue
            common_keys = [k for k in primary_keys if k in table and k in fam_dict["by_universe"][primary_str]]
            a = [fam_dict["by_universe"][primary_str][k]["sharpe"] for k in common_keys]
            b = [table[k]["sharpe"] for k in common_keys]
            fam_dict["spearman_vs_primary"][universe_str] = float(stats.spearmanr(a, b).statistic) if len(common_keys) > 2 else float("nan")
        cross_dict[family_str] = fam_dict
    res["cross_universe"] = cross_dict

    # ---------------------------------------------------------------------------------------------- daily-rebalanced sensitivity
    daily_dict = {"G3": res["g3_baseline_replica_L"]["daily_rebalanced_engine"]}
    for c in candidate_list:
        t_ser = ret_dict[(c["family"], FAMILY_PRIMARY_DICT[c["family"]], "engine")][c["centre"]]
        daily_dict[c["stage"]] = {role_str: book_blocks(role_str, taa_ser, l_engine_ser, t_ser, daily_bool=True) for role_str in FAMILY_ROLE_DICT[c["family"]]}
    res["daily_rebalanced_sensitivity"] = daily_dict

    # ---------------------------------------------------------------------------------------------- diagnostics: stops vs L
    stop_diag = {}
    for key_str, entry in cell_res.items():
        if entry["family"] != "A":
            continue
        m = meta_primary.get(key_str, {}).get("engine", {})
        stop_diag[key_str] = {
            "standalone_full_sharpe_minus_L": entry["standalone"]["FULL"]["sharpe"] - res["legs"]["L_replica_standalone"]["FULL"]["sharpe"],
            "standalone_full_dd_minus_L_pp": (entry["standalone"]["FULL"]["max_dd"] - res["legs"]["L_replica_standalone"]["FULL"]["max_dd"]) * 100,
            "standalone_full_cagr_minus_L_pp": (entry["standalone"]["FULL"]["cagr"] - res["legs"]["L_replica_standalone"]["FULL"]["cagr"]) * 100,
            "book_g_full_sharpe_minus_G3": entry["books"]["replacement"]["engine"]["G-FULL"]["sharpe"] - g3_engine["G-FULL"]["sharpe"],
            "book_g_full_dd_minus_G3_pp": (entry["books"]["replacement"]["engine"]["G-FULL"]["max_dd"] - g3_engine["G-FULL"]["max_dd"]) * 100,
            "stop_exits_per_year": m.get("stop_exits_int", 0) / (len(a_ndx_engine_df) / 252.0),
            "stop_fill_vs_level_mean": m.get("stop_fill_vs_level_mean"), "stop_fill_vs_level_p5": m.get("stop_fill_vs_level_p5"),
            "stop_gap_through_level_share": m.get("stop_gap_through_level_share"), "stop_fill_below_minus2pct_share": m.get("stop_fill_below_minus2pct_share"),
            "turnover_x_per_year": m.get("turnover_x_per_year"), "terminal_pnl_share": m.get("terminal_pnl_share"), "phantom_fill_int": m.get("phantom_fill_int"),
        }
    res["stops_vs_L"] = stop_diag
    res["L_meta"] = meta_dict[("A", "NDX")].get(L_KEY, {})

    # ---------------------------------------------------------------------------------------------- CSV tables
    row_list = []
    for key_str, entry in cell_res.items():
        row = {"key": key_str, "family": entry["family"]}
        for b, m in entry["standalone"].items():
            row.update({f"sa_{b}_sharpe": m["sharpe"], f"sa_{b}_cagr": m["cagr"], f"sa_{b}_dd": m["max_dd"]})
        for role_str, books in entry["books"].items():
            for b, m in books["engine"].items():
                row.update({f"{role_str}_{b}_sharpe": m["sharpe"], f"{role_str}_{b}_cagr": m["cagr"], f"{role_str}_{b}_dd": m["max_dd"]})
            if books["stress"]:
                for b, m in books["stress"].items():
                    row.update({f"{role_str}_stress_{b}_sharpe": m["sharpe"], f"{role_str}_stress_{b}_dd": m["max_dd"]})
        for u, m in entry["other_universes_standalone_full"].items():
            row[f"sa_FULL_sharpe_{u}"] = m["sharpe"]
        row["corr_taa"] = entry["corr_with_taa_2012_2026"]
        row["corr_L"] = entry["corr_with_L_2000_2026"]
        meta_e = meta_primary.get(key_str, {}).get("engine", {})
        row.update({k: meta_e.get(k) for k in ("turnover_x_per_year", "mean_exposure", "round_trips_per_year", "holding_sessions_median", "stop_exits_int", "stop_fill_vs_level_mean",
                                               "stop_gap_through_level_share", "terminal_liquidations_int", "terminal_pnl_share", "terminal_notional_share", "phantom_fill_int")})
        cap = meta_e.get("capacity_2021_2026", {})
        row.update({"cap21_max_5pct_usd": cap.get("aum_max_5pct_usd"), "cap21_p95_1pct_usd": cap.get("aum_p95_1pct_usd")})
        if entry["standalone_haircut"]:
            row["sa_FULL_sharpe_haircut"] = entry["standalone_haircut"]["FULL"]["sharpe"]
            row["sa_FULL_cagr_haircut"] = entry["standalone_haircut"]["FULL"]["cagr"]
        row_list.append(row)
    pd.DataFrame(row_list).to_csv(OUT_PATH / "table_cells.csv", index=False)
    rule_rows = []
    for c in candidate_list:
        for role_str, rule in c["rule_by_role"].items():
            rule_rows.append({"stage": c["stage"], "centre": c["centre"], "role": role_str, "plateau_full_sharpe": c["plateau_full_sharpe"],
                              **{f"margin_{b}": v for b, v in rule["engine"]["sharpe_margin_vs_G3"].items()}, "min_margin": rule["engine"]["min_margin"],
                              **{f"dd_gap_pp_{b}": v for b, v in rule["engine"]["dd_gap_vs_G3_pp"].items()}, "R1": rule["engine"]["R1"], "R2": rule["engine"]["R2"], "R3": rule["R3"],
                              **{f"R4_{u}": v["pass"] for u, v in rule["R4"].items()}, "R5": rule["R5"]["pass"], "passes": rule["passes"],
                              "owner_gate_centre": rule["owner_gate_centre"]["pass"], "centre_book_sharpe": rule["owner_gate_centre"]["sharpe"], "centre_book_dd": rule["owner_gate_centre"]["max_dd"]})
    pd.DataFrame(rule_rows).to_csv(OUT_PATH / "table_rule.csv", index=False)

    # ---------------------------------------------------------------------------------------------- charts
    # A heatmaps
    def book_val(key_str, role_str, block_str, metric_str="sharpe"):
        return cell_res[key_str]["books"][role_str]["engine"][block_str][metric_str] if key_str in cell_res else np.nan

    def worst_margin(key_str, role_str):
        return min(cell_res[key_str]["books"][role_str]["engine"][b]["sharpe"] - g3_engine[b]["sharpe"] for b in common.RULE_BLOCK_TUPLE) if key_str in cell_res else np.nan

    a_book_arr = np.vectorize(lambda k: book_val(k, "replacement", "G-FULL"), otypes=[float])(a_key_arr)
    a_margin_arr = np.vectorize(lambda k: worst_margin(k, "replacement"), otypes=[float])(a_key_arr)
    a_centres = {c["grid_pos"] for c in candidate_list if c["family"] == "A"}
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.6), facecolor=SURFACE)
    level_labels = ["CH 2 / PT 10%", "CH 3 / PT 15%", "CH 4 / PT 20%", "CH 5 / PT 25%"]
    for ax, arr, title, cmap, lim in zip(axes, (a_value_arr, a_book_arr, a_margin_arr),
                                          (f"Standalone Sharpe 2000-26 (L = {res['legs']['L_replica_standalone']['FULL']['sharpe']:.2f})",
                                           f"Book Sharpe 2012-26, 0.5 TAA + 0.5 leg (G3 = {g3_engine['G-FULL']['sharpe']:.2f})",
                                           "Worst G3-block margin vs G3 (blue = beats G3 in all 3 blocks)"),
                                          (seq_cmap, seq_cmap, div_cmap), ((np.nanmin(a_value_arr), np.nanmax(a_value_arr)), (np.nanmin(a_book_arr), np.nanmax(a_book_arr)), (-0.35, 0.35))):
        heatmap(ax, arr, a_rows, level_labels, title, cmap, lim[0], lim[1])
        from matplotlib.patches import Rectangle
        for (i, j) in a_centres:
            ax.add_patch(Rectangle((j - 0.42, i - 0.42), 0.84, 0.84, fill=False, ec="#eb6834", lw=2.0, ls="--"))
    fig.suptitle("Family A: stop overlays on the live NDX pod (dashed = row plateau centre)", fontsize=10, color=TEXT_SECONDARY, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(CHART_PATH / "heatmap_A_NDX.png", dpi=130)
    plt.close(fig)
    # B heatmaps
    for stage_str in ("B1_entry_exit", "B2_slots_rank"):
        grid = res["family_B_grids"][stage_str]
        key_arr = np.array(grid["keys"], dtype=object)
        centre_pos = next(c["grid_pos"] for c in candidate_list if c["stage"] == stage_str)
        value_arr = np.array(grid["standalone_full_sharpe"])
        rep_arr = np.vectorize(lambda k: book_val(k, "replacement", "G-FULL"), otypes=[float])(key_arr)
        add_arr = np.vectorize(lambda k: book_val(k, "addition", "G-FULL"), otypes=[float])(key_arr)
        margin_arr = np.vectorize(lambda k: min(worst_margin(k, "replacement"), worst_margin(k, "addition")) if k in cell_res else np.nan, otypes=[float])(key_arr)
        fig, axes = plt.subplots(1, 4, figsize=(18, 3.2), facecolor=SURFACE)
        titles = [f"Standalone Sharpe 2000-26 S&P 500 (L = {res['legs']['L_replica_standalone']['FULL']['sharpe']:.2f})",
                  f"Book replacement 0.5 TAA + 0.5 B (G3 = {g3_engine['G-FULL']['sharpe']:.2f})", "Book addition 0.5 TAA + 0.25 L + 0.25 B", "Worst G3-block margin (best role)"]
        for ax, arr, title, cmap, lim in zip(axes, (value_arr, rep_arr, add_arr, margin_arr), titles, (seq_cmap, seq_cmap, seq_cmap, div_cmap),
                                              ((np.nanmin(value_arr), np.nanmax(value_arr)), (np.nanmin(rep_arr), np.nanmax(rep_arr)), (np.nanmin(add_arr), np.nanmax(add_arr)), (-0.35, 0.35))):
            heatmap(ax, arr, grid["rows"], grid["cols"], title, cmap, lim[0], lim[1], tuple(grid["b0_pos"]), tuple(centre_pos))
        fig.suptitle(f"Family B {stage_str}: black box = B0 anchor, dashed = plateau centre", fontsize=10, color=TEXT_SECONDARY, x=0.01, ha="left")
        fig.tight_layout()
        fig.savefig(CHART_PATH / f"heatmap_{stage_str}_SP500.png", dpi=130)
        plt.close(fig)
        # universes
        fig, axes = plt.subplots(1, 3, figsize=(15, 3.2), facecolor=SURFACE)
        for ax, universe_str in zip(axes, ("SP500", "NDX", "R1000")):
            table = cross_dict.get("B", {}).get("by_universe", {}).get(universe_str)
            if table is None:
                continue
            arr = np.vectorize(lambda k: table[k]["sharpe"] if k in table else np.nan, otypes=[float])(key_arr)
            heatmap(ax, arr, grid["rows"], grid["cols"], f"{universe_str}: standalone Sharpe 2000-26", seq_cmap, np.nanmin(arr), np.nanmax(arr), tuple(grid["b0_pos"]), tuple(centre_pos))
        fig.tight_layout()
        fig.savefig(CHART_PATH / f"heatmap_{stage_str}_universes.png", dpi=130)
        plt.close(fig)
    # timing luck
    if luck_dict:
        fig, axes = plt.subplots(1, 2, figsize=(14, 4.2), facecolor=SURFACE)
        colors = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
        for ax, metric_str, title_str in zip(axes, ("standalone_full", "book_g_full"), ("Standalone Sharpe 2000-26", "Book Sharpe 2012-26 (0.5 TAA + 0.5 leg)")):
            for (label_str, entry), color in zip(luck_dict.items(), colors):
                ax.plot(cells_module.OFFSET_TUPLE, entry[metric_str]["by_offset"], color=color, lw=2 if label_str == "L" else 1.4, marker="o", ms=4, label=f"{label_str}: {entry['centre_key']}")
            ax.axvline(0, color=TEXT_SECONDARY, lw=0.8, ls=":")
            ax.set_xticks(range(-10, 11, 2))
            ax.set_title(title_str + " by rebalance day", fontsize=9.5, loc="left", color=TEXT_PRIMARY)
            ax.set_xlabel("rebalance offset k (0 = live: last session of the month)", fontsize=8, color=TEXT_SECONDARY)
            style_axis(ax)
        axes[0].legend(fontsize=7, frameon=False, loc="lower left")
        fig.tight_layout()
        fig.savefig(CHART_PATH / "timing_luck_A.png", dpi=130)
        plt.close(fig)
    # stop slippage histograms
    trade_dict = {**{k: v for k, v in load_trades("A", "NDX").items()}, **{k: v for k, v in load_trades("B", "SP500").items()}}
    hist_keys = [c["centre"] for c in candidate_list if c["centre"] in trade_dict] + ([B0_KEY] if B0_KEY in trade_dict else [])
    if hist_keys:
        fig, axes = plt.subplots(1, len(hist_keys), figsize=(3.6 * len(hist_keys), 3.2), facecolor=SURFACE, squeeze=False)
        for ax, key_str in zip(axes[0], hist_keys):
            trade_df = trade_dict[key_str]
            slip_vec = trade_df.loc[trade_df["reason"] == "stop", "fill_vs_stop"].dropna().to_numpy() * 100
            if len(slip_vec):
                ax.hist(np.clip(slip_vec, -15, 15), bins=40, color="#3987e5")
                ax.axvline(0, color=TEXT_SECONDARY, lw=0.8, ls=":")
            ax.set_title(f"{key_str}\nfill vs stop level, % (n={len(slip_vec)}, mean {slip_vec.mean() if len(slip_vec) else float('nan'):.2f})", fontsize=7.5, loc="left", color=TEXT_PRIMARY)
            style_axis(ax)
        fig.tight_layout()
        fig.savefig(CHART_PATH / "stop_slippage_hist.png", dpi=130)
        plt.close(fig)
    # exposure paths and equity / drawdown of candidate books
    exposure_frames = []
    for family_str, universe_str in (("A", "NDX"), ("B", "SP500"), ("C", "NDX")):
        path = OUT_PATH / f"exposure_{family_str}_{universe_str}.parquet"
        if path.exists():
            exposure_frames.append(pd.read_parquet(path))
    if exposure_frames:
        exposure_df = pd.concat(exposure_frames, axis=1)
        exposure_df.index = pd.to_datetime(exposure_df.index)
        fig, ax = plt.subplots(figsize=(14, 3.6), facecolor=SURFACE)
        for (label_str, key_str), color in zip([("L", L_KEY)] + [(c["stage"], c["centre"]) for c in candidate_list], ["#0b0b0b", "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#8f2423", "#256abf"]):
            if key_str in exposure_df:
                ax.plot(exposure_df.index, exposure_df[key_str].rolling(63).mean(), color=color, lw=1.2, label=f"{label_str}: {key_str}")
        ax.set_title("Invested share of NAV, 63-session mean", fontsize=9.5, loc="left", color=TEXT_PRIMARY)
        ax.legend(fontsize=7, frameon=False, ncol=2)
        style_axis(ax)
        fig.tight_layout()
        fig.savefig(CHART_PATH / "exposure_paths.png", dpi=130)
        plt.close(fig)
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), facecolor=SURFACE, sharex=True)
    for (label_str, ser), color in zip([("G3 (0.5 TAA + 0.5 L)", book_full_df["G3"])] + [(f"{c['stage']} {role_str}: {c['centre']}", book_full_df[f"{c['centre']}||{role_str}"]) for c in candidate_list for role_str in FAMILY_ROLE_DICT[c["family"]]],
                                       ["#0b0b0b", "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#8f2423", "#256abf", "#6da7ec", "#c93b3a", "#184f95"]):
        wealth_ser = (1 + ser).cumprod()
        axes[0].plot(wealth_ser.index, wealth_ser, color=color, lw=1.6 if label_str.startswith("G3") else 1.0, label=label_str)
        axes[1].plot(wealth_ser.index, wealth_ser / wealth_ser.cummax() - 1, color=color, lw=1.6 if label_str.startswith("G3") else 1.0)
    axes[0].set_yscale("log")
    axes[0].set_title("Candidate books versus G3, 2012-10-02..2026-08-19 (official pod model, annual reset)", fontsize=9.5, loc="left", color=TEXT_PRIMARY)
    axes[1].set_title("Drawdown", fontsize=9.5, loc="left", color=TEXT_PRIMARY)
    axes[0].legend(fontsize=7, frameon=False, ncol=2)
    for ax in axes:
        style_axis(ax)
    fig.tight_layout()
    fig.savefig(CHART_PATH / "equity_dd_candidates.png", dpi=130)
    plt.close(fig)

    amendments_path = OUT_PATH / "amendments_and_bugs.json"
    res["amendments_and_bugs"] = json.loads(amendments_path.read_text()) if amendments_path.exists() else []
    common.write_json("results.json", res)
    common.log_progress(f"analysis done: decision {json.dumps(res['decision'], default=str)[:600]}")
    for c in candidate_list:
        for role_str, rule in c["rule_by_role"].items():
            print(c["stage"], role_str, "| centre", c["centre"], "| margins", {b: round(v, 3) for b, v in rule["engine"]["sharpe_margin_vs_G3"].items()},
                  "| dd", {b: round(v, 2) for b, v in rule["engine"]["dd_gap_vs_G3_pp"].items()}, "| R3", rule["R3"], "| R4", {u: v["pass"] for u, v in rule["R4"].items()},
                  "| R5", rule["R5"]["pass"], "| passes", rule["passes"], "| owner gate", rule["owner_gate_centre"]["pass"])


if __name__ == "__main__":
    main()
