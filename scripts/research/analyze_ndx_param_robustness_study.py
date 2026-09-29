"""Analysis for the NDX momentum pod parameter-robustness study (research only).

Applies the frozen rule of docs/research/NDX_PARAM_ROBUSTNESS_PREREG_20260926.md mechanically to the grid returns
written by run_ndx_param_robustness_study.py, then the confidence labels and diagnostics. Writes results.json,
tables.md and charts/*.png into results/research/ndx_param_robustness_20260926/.

    uv run python scripts/research/analyze_ndx_param_robustness_study.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ndx_param_robustness_core as core  # noqa: E402

OUT_PATH = core.RESULTS_DIR_PATH
CHART_PATH = OUT_PATH / "charts"
TAA_CSV_PATH = Path(r"C:/Users/User/Documents/workspace/0_papers/index/review/stage3/rescreen_true_g3/book_daily_returns_true.csv")
TAA_PROXY_PATH = Path(
    r"C:/Users/User/Documents/workspace/alpha_super/results/research/portfolio/growth_shelf_v2_20260926/"
    r"proxy_runs/splice_scaled/taa_btal_tqqq__path.csv.gz"
)
END_TS = pd.Timestamp("2026-07-24")
BLOCK_DICT = {
    "P1": ("2000-01-04", "2011-12-31"),
    "P2": ("2012-01-01", "2021-12-31"),
    "P3": ("2022-01-01", "2026-07-24"),
    "FULL": ("2000-01-04", "2026-07-24"),
}
G3_BLOCK_DICT = {
    "G-P1": ("2008-03-04", "2011-12-31"),
    "G-P2": ("2012-10-02", "2021-12-31"),
    "G-P3": ("2022-01-01", "2026-07-24"),
    "G-FULL": ("2012-10-02", "2026-07-24"),
    "G-LONG": ("2008-03-04", "2026-07-24"),
}
RULE_BLOCK_TUPLE = ("G-P1", "G-P2", "G-P3")
DD_TOLERANCE_FLOAT = 0.02
N_TRIALS_INT = 159
BOOT_N_INT, BOOT_BLOCK_FLOAT, BOOT_SEED_INT = 2000, 21.0, 20260926
STAGE_TUPLE = ("S1_score", "S2_n_weight", "S3_filters", "S4_buffer_offset", "S5_vxn")
# neighbourhood axes per stage: (use row neighbours, use column neighbours)
NEIGHBOUR_AXIS_DICT = {
    "S1_score": (True, True),
    "S2_n_weight": (False, True),
    "S3_filters": (False, True),
    "S5_vxn": (True, True),
}
A0_KEY = core.ANCHOR_CELL.key_str
L_KEY = core.L_CELL.key_str
B_KEY = core.B_CELL.key_str


# ----------------------------------------------------------------------------------------------------------------------
# metrics
# ----------------------------------------------------------------------------------------------------------------------
def metric_dict(return_ser: pd.Series) -> dict:
    return_ser = return_ser.dropna()
    wealth_vec = np.concatenate([[1.0], (1.0 + return_ser).cumprod().to_numpy()])
    max_dd_float = float((wealth_vec / np.maximum.accumulate(wealth_vec) - 1.0).min())
    cagr_float = float(wealth_vec[-1] ** (252.0 / len(return_ser)) - 1.0)
    return {
        "cagr": cagr_float,
        "sharpe": float(return_ser.mean() / return_ser.std() * np.sqrt(252.0)),
        "max_dd": max_dd_float,
        "calmar": cagr_float / abs(max_dd_float) if max_dd_float < 0 else float("nan"),
    }


def load_taa_long_ser() -> pd.Series:
    taa_ser = pd.read_csv(TAA_CSV_PATH, index_col=0, parse_dates=True)["taa_rank_tqqq"].dropna()
    proxy_ret_ser = pd.read_csv(TAA_PROXY_PATH, index_col="date", parse_dates=True)["total_value_float"].pct_change()
    g3_start_ts = pd.Timestamp("2012-10-02")
    return pd.concat([proxy_ret_ser.loc["2008-03-04": g3_start_ts - pd.Timedelta(days=1)], taa_ser.loc[g3_start_ts:]])


def g3_frame(return_df: pd.DataFrame, taa_long_ser: pd.Series) -> pd.DataFrame:
    leg_df = return_df.reindex(taa_long_ser.index).loc["2008-03-04":END_TS]
    # *** CRITICAL *** 50/50 rebalanced daily: same-day returns of the TAA leg and the NDX leg.
    return 0.5 * leg_df.add(taa_long_ser.loc["2008-03-04":END_TS], axis=0)


def all_metrics(return_df: pd.DataFrame, taa_long_ser: pd.Series) -> tuple[dict, dict]:
    standalone_dict = {
        key: {b: metric_dict(return_df[key].loc[s:e]) for b, (s, e) in BLOCK_DICT.items()} for key in return_df.columns
    }
    g3_df = g3_frame(return_df, taa_long_ser)
    g3_dict = {key: {b: metric_dict(g3_df[key].loc[s:e]) for b, (s, e) in G3_BLOCK_DICT.items()} for key in g3_df.columns}
    return standalone_dict, g3_dict


# ----------------------------------------------------------------------------------------------------------------------
# grids and plateaus
# ----------------------------------------------------------------------------------------------------------------------
def stage_matrix(stage_map_list: list[dict]) -> tuple[list, list, np.ndarray]:
    row_list = list(dict.fromkeys(item["row"] for item in stage_map_list))
    col_list = list(dict.fromkeys(item["col"] for item in stage_map_list))
    key_arr = np.empty((len(row_list), len(col_list)), dtype=object)
    for item in stage_map_list:
        key_arr[row_list.index(item["row"]), col_list.index(item["col"])] = item["key"]
    return row_list, col_list, key_arr


def neighbour_list(i: int, j: int, shape: tuple, use_row_bool: bool, use_col_bool: bool) -> list[tuple[int, int]]:
    row_range = range(max(0, i - 1), min(shape[0], i + 2)) if use_row_bool else [i]
    col_range = range(max(0, j - 1), min(shape[1], j + 2)) if use_col_bool else [j]
    return [(r, c) for r in row_range for c in col_range]


def stage_candidate(stage_str: str, key_arr: np.ndarray, value_fn, a0_pos: tuple) -> dict:
    """Plateau values of a scalar metric and the argmax cell (ties -> nearest A0)."""
    if stage_str == "S4_buffer_offset":
        # per buffer: median over the 21 offsets, then +-1 buffer neighbourhood; candidate at k = 0
        k0_col_int = list(core.STAGE4_OFFSET_TUPLE).index(0)
        buffer_value_vec = np.array([np.median([value_fn(k) for k in key_arr[r]]) for r in range(key_arr.shape[0])])
        plateau_vec = np.array(
            [np.median(buffer_value_vec[max(0, r - 1): r + 2]) for r in range(len(buffer_value_vec))]
        )
        best_r = int(max(range(len(plateau_vec)), key=lambda r: (round(plateau_vec[r], 12), -abs(r - a0_pos[0]))))
        neighbourhood = [(r, k0_col_int) for r in range(max(0, best_r - 1), min(len(plateau_vec), best_r + 2))]
        plateau_arr = np.tile(plateau_vec[:, None], (1, key_arr.shape[1]))
        return {"centre": (best_r, k0_col_int), "neighbourhood": neighbourhood, "plateau_arr": plateau_arr,
                "buffer_median_over_offsets": buffer_value_vec.tolist()}
    use_row_bool, use_col_bool = NEIGHBOUR_AXIS_DICT[stage_str]
    value_arr = np.vectorize(value_fn, otypes=[float])(key_arr)
    plateau_arr = np.full(key_arr.shape, np.nan)
    for i in range(key_arr.shape[0]):
        for j in range(key_arr.shape[1]):
            plateau_arr[i, j] = np.median([value_arr[r, c] for r, c in neighbour_list(i, j, key_arr.shape, use_row_bool, use_col_bool)])
    best_pos = max(
        ((i, j) for i in range(key_arr.shape[0]) for j in range(key_arr.shape[1])),
        key=lambda p: (round(plateau_arr[p], 12), -(abs(p[0] - a0_pos[0]) + abs(p[1] - a0_pos[1]))),
    )
    return {"centre": best_pos, "neighbourhood": neighbour_list(*best_pos, key_arr.shape, use_row_bool, use_col_bool),
            "plateau_arr": plateau_arr}


def apply_rule(neigh_key_list: list[str], g3_dict: dict, g3_stress_dict: dict, uni_sharpe_dict: dict,
               centre_is_a0_bool: bool = False) -> dict:
    def r1_r2(g_dict: dict) -> dict:
        margin_dict = {
            b: float(np.median([g_dict[k][b]["sharpe"] for k in neigh_key_list]) - g_dict[L_KEY][b]["sharpe"])
            for b in RULE_BLOCK_TUPLE
        }
        dd_gap_dict = {
            b: float(np.median([g_dict[k][b]["max_dd"] for k in neigh_key_list]) - g_dict[L_KEY][b]["max_dd"])
            for b in ("G-FULL", "G-LONG")
        }
        return {
            "sharpe_margin_vs_L": margin_dict,
            "min_margin": min(margin_dict.values()),
            "dd_gap_vs_L_pp": {b: v * 100 for b, v in dd_gap_dict.items()},
            "R1": all(m > 0 for m in margin_dict.values()),
            "R2": all(v >= -DD_TOLERANCE_FLOAT for v in dd_gap_dict.values()),
        }

    engine_dict = r1_r2(g3_dict)
    stress_dict = r1_r2(g3_stress_dict)
    r4_dict = {}
    for universe_str, sharpe_dict in uni_sharpe_dict.items():
        neigh_float = float(np.median([sharpe_dict[k] for k in neigh_key_list]))
        # PREREG section 8: if the plateau centre is A0 itself, R4 holds trivially (the live value is on the plateau);
        # the neighbourhood comparison is still reported.
        r4_dict[universe_str] = {"neigh_median_sharpe": neigh_float, "a0_sharpe": sharpe_dict[A0_KEY],
                                 "neigh_ge_a0": bool(neigh_float >= sharpe_dict[A0_KEY]),
                                 "pass": bool(centre_is_a0_bool or neigh_float >= sharpe_dict[A0_KEY])}
    passes_bool = bool(
        engine_dict["R1"] and engine_dict["R2"] and stress_dict["R1"] and stress_dict["R2"] and all(v["pass"] for v in r4_dict.values())
    )
    return {"engine": engine_dict, "stress5": stress_dict, "R3": bool(stress_dict["R1"] and stress_dict["R2"]),
            "R4": r4_dict, "passes": passes_bool}


# ----------------------------------------------------------------------------------------------------------------------
# multiplicity
# ----------------------------------------------------------------------------------------------------------------------
def stationary_boot_index(n_int: int, rng: np.random.Generator, draws_int: int) -> np.ndarray:
    """Politis-Romano stationary bootstrap indices (draws x n)."""
    idx_arr = np.empty((draws_int, n_int), dtype=np.int32)
    idx_arr[:, 0] = rng.integers(0, n_int, draws_int)
    new_block_arr = rng.random((draws_int, n_int)) < 1.0 / BOOT_BLOCK_FLOAT
    start_arr = rng.integers(0, n_int, (draws_int, n_int))
    for j in range(1, n_int):
        idx_arr[:, j] = np.where(new_block_arr[:, j], start_arr[:, j], (idx_arr[:, j - 1] + 1) % n_int)
    return idx_arr


def sharpe_cols(arr: np.ndarray) -> np.ndarray:
    return arr.mean(axis=0) / arr.std(axis=0, ddof=1) * np.sqrt(252.0)


def reality_check(return_df: pd.DataFrame, candidate_key_list: list[str], seed_int: int) -> dict:
    """White's Reality Check on annualised Sharpe differences versus L, plus paired p for chosen centres."""
    key_list = [k for k in return_df.columns if k != L_KEY]
    x_arr = return_df[key_list].to_numpy()
    l_vec = return_df[L_KEY].to_numpy()
    obs_diff_vec = sharpe_cols(x_arr) - sharpe_cols(l_vec[:, None])[0]
    idx_arr = stationary_boot_index(len(return_df), np.random.default_rng(seed_int), BOOT_N_INT)
    max_centered_vec = np.empty(BOOT_N_INT)
    paired_diff_dict = {k: np.empty(BOOT_N_INT) for k in candidate_key_list}
    col_pos_dict = {k: key_list.index(k) for k in candidate_key_list if k in key_list}
    for b in range(BOOT_N_INT):
        idx_vec = idx_arr[b]
        diff_vec = sharpe_cols(x_arr[idx_vec]) - sharpe_cols(l_vec[idx_vec][:, None])[0]
        max_centered_vec[b] = np.max(diff_vec - obs_diff_vec)
        for k, pos in col_pos_dict.items():
            paired_diff_dict[k][b] = diff_vec[pos]
    best_pos = int(np.argmax(obs_diff_vec))
    return {
        "configs_int": len(key_list),
        "best_cell": key_list[best_pos],
        "best_obs_diff_vs_L": float(obs_diff_vec[best_pos]),
        "reality_check_p": float(np.mean(max_centered_vec >= obs_diff_vec[best_pos])),
        "share_cells_above_L": float(np.mean(obs_diff_vec > 0)),
        "paired": {
            k: {"obs_diff": float(obs_diff_vec[col_pos_dict[k]]), "p_one_sided": float(np.mean(v <= 0)),
                "ci90": [float(np.quantile(v, 0.05)), float(np.quantile(v, 0.95))]}
            for k, v in paired_diff_dict.items() if k in col_pos_dict
        },
    }


def deflated_sharpe(return_ser: pd.Series, sharpe_daily_all_vec: np.ndarray) -> dict:
    gamma_float = 0.5772156649
    var_float = float(np.var(sharpe_daily_all_vec, ddof=1))
    sr0_float = np.sqrt(var_float) * (
        (1 - gamma_float) * stats.norm.ppf(1 - 1 / N_TRIALS_INT) + gamma_float * stats.norm.ppf(1 - 1 / (N_TRIALS_INT * np.e))
    )
    x_vec = return_ser.dropna().to_numpy()
    sr_float = x_vec.mean() / x_vec.std(ddof=1)
    skew_float, kurt_float = float(stats.skew(x_vec)), float(stats.kurtosis(x_vec, fisher=False))
    z_float = (sr_float - sr0_float) * np.sqrt(len(x_vec) - 1) / np.sqrt(1 - skew_float * sr_float + (kurt_float - 1) / 4 * sr_float**2)
    return {"sharpe_ann": sr_float * np.sqrt(252), "sr0_ann": float(sr0_float * np.sqrt(252)), "dsr_prob": float(stats.norm.cdf(z_float)),
            "trials_int": N_TRIALS_INT, "cross_trial_sd_ann": float(np.sqrt(var_float) * np.sqrt(252))}


# ----------------------------------------------------------------------------------------------------------------------
# charts
# ----------------------------------------------------------------------------------------------------------------------
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
DIVERGING = ["#8f2423", "#c93b3a", "#e34948", "#f3a9a8", "#f0efec", "#9ec5f4", "#3987e5", "#256abf", "#184f95"]
TEXT_PRIMARY, TEXT_SECONDARY, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"


def heatmap(ax, value_arr, row_list, col_list, title_str, cmap, vmin, vmax, a0_pos=None, centre_pos=None, fmt="{:.2f}", fontsize=8):
    from matplotlib.patches import Rectangle

    image = ax.imshow(value_arr.astype(float), cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    for i in range(value_arr.shape[0]):
        for j in range(value_arr.shape[1]):
            v = float(value_arr[i, j])
            rgba = image.cmap(image.norm(v))
            lum = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
            ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=fontsize, color="white" if lum < 0.5 else TEXT_PRIMARY)
    ax.set_xticks(range(len(col_list)), [str(c) for c in col_list], fontsize=8, color=TEXT_SECONDARY)
    ax.set_yticks(range(len(row_list)), [str(r) for r in row_list], fontsize=8, color=TEXT_SECONDARY)
    ax.set_title(title_str, fontsize=9.5, color=TEXT_PRIMARY, loc="left")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    if a0_pos is not None:
        ax.add_patch(Rectangle((a0_pos[1] - 0.5, a0_pos[0] - 0.5), 1, 1, fill=False, ec=TEXT_PRIMARY, lw=2.0))
    if centre_pos is not None:
        ax.add_patch(Rectangle((centre_pos[1] - 0.42, centre_pos[0] - 0.42), 0.84, 0.84, fill=False, ec="#eb6834", lw=2.0, ls="--"))
    return image


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    CHART_PATH.mkdir(parents=True, exist_ok=True)
    seq_cmap = LinearSegmentedColormap.from_list("seq", BLUE_RAMP)
    div_cmap = LinearSegmentedColormap.from_list("div", DIVERGING)
    stage_map_dict = json.loads((OUT_PATH / "stage_map.json").read_text())
    taa_long_ser = load_taa_long_ser()
    ret_dict = {c: pd.read_parquet(OUT_PATH / f"returns_NDX_{c}.parquet") for c in ("engine", "stress5", "commfix")}
    meta_dict = json.loads((OUT_PATH / "cell_meta_NDX.json").read_text())
    selections_dict = json.loads((OUT_PATH / "selections_NDX.json").read_text())
    standalone, g3 = all_metrics(ret_dict["engine"], taa_long_ser)
    _, g3_stress = all_metrics(ret_dict["stress5"], taa_long_ser)
    _, g3_cf = all_metrics(ret_dict["commfix"], taa_long_ser)
    other_dict = {}
    for universe_str in ("SP500", "R1000"):
        path = OUT_PATH / f"returns_{universe_str}_engine.parquet"
        if path.exists():
            other_ret_df = pd.read_parquet(path)
            other_dict[universe_str] = {
                k: {b: metric_dict(other_ret_df[k].loc[s:e]) for b, (s, e) in BLOCK_DICT.items()} for k in other_ret_df.columns
            }
    uni_sharpe_dict = {u: {k: v["FULL"]["sharpe"] for k, v in d.items()} for u, d in other_dict.items()}
    res: dict = {"L": {"standalone": standalone[L_KEY], "g3": g3[L_KEY]}, "A0": {"standalone": standalone[A0_KEY], "g3": g3[A0_KEY]},
                 "B": {"standalone": standalone[B_KEY], "g3": g3[B_KEY]},
                 "taa_alone": {b: metric_dict(taa_long_ser.loc[s:e]) for b, (s, e) in G3_BLOCK_DICT.items()}}
    scale_free_key_list = [k for k, m in meta_dict.items() if m["scale_free_bool"]]

    # ------------------------------------------------------------------------------------------------ stages
    stage_res: dict = {}
    for stage_str in STAGE_TUPLE:
        row_list, col_list, key_arr = stage_matrix(stage_map_dict[stage_str])
        a0_pos = tuple(int(x) for x in np.argwhere(key_arr == A0_KEY)[0])
        cand = stage_candidate(stage_str, key_arr, lambda k: standalone[k]["FULL"]["sharpe"], a0_pos)
        wf = stage_candidate(stage_str, key_arr, lambda k: standalone[k]["P1"]["sharpe"], a0_pos)
        centre_key = key_arr[cand["centre"]]
        neigh_key_list = [key_arr[p] for p in cand["neighbourhood"]]
        rule = apply_rule(neigh_key_list, g3, g3_stress, uni_sharpe_dict, centre_key == A0_KEY)
        rule_cf = apply_rule(neigh_key_list, g3_cf, g3_cf, {})["engine"]
        all_keys = list(key_arr.flatten())
        wf_centre_key = key_arr[wf["centre"]]
        wf_neigh = [key_arr[p] for p in wf["neighbourhood"]]
        stage_res[stage_str] = {
            "rows": [str(r) for r in row_list], "cols": [str(c) for c in col_list],
            "a0_pos": a0_pos, "candidate_centre": centre_key, "candidate_is_A0": centre_key == A0_KEY,
            "candidate_neighbourhood": neigh_key_list,
            "candidate_plateau_full_sharpe": float(cand["plateau_arr"][cand["centre"]]),
            "a0_plateau_full_sharpe": float(cand["plateau_arr"][a0_pos]),
            "a0_rank_full_sharpe": int(1 + sum(standalone[k]["FULL"]["sharpe"] > standalone[A0_KEY]["FULL"]["sharpe"] for k in set(all_keys))),
            "cells_int": len(set(all_keys)),
            "share_cells_beating_L_g3": {b: float(np.mean([g3[k][b]["sharpe"] > g3[L_KEY][b]["sharpe"] for k in all_keys])) for b in RULE_BLOCK_TUPLE + ("G-FULL",)},
            "share_cells_beating_L_g3_all_blocks": float(np.mean([all(g3[k][b]["sharpe"] > g3[L_KEY][b]["sharpe"] for b in RULE_BLOCK_TUPLE) for k in all_keys])),
            "rule": rule, "rule_commission_fixed": rule_cf,
            "centre_standalone": standalone[centre_key], "centre_g3": g3[centre_key],
            "neigh_median_standalone": {b: {m: float(np.median([standalone[k][b][m] for k in neigh_key_list])) for m in ("cagr", "sharpe", "max_dd")} for b in BLOCK_DICT},
            "neigh_median_g3": {b: {m: float(np.median([g3[k][b][m] for k in neigh_key_list])) for m in ("cagr", "sharpe", "max_dd")} for b in G3_BLOCK_DICT},
            "walk_forward": {
                "centre_by_P1": wf_centre_key,
                "neigh_median": {b: float(np.median([standalone[k][b]["sharpe"] for k in wf_neigh])) for b in ("P2", "P3")}
                | {b: float(np.median([g3[k][b]["sharpe"] for k in wf_neigh])) for b in ("G-P2", "G-P3")},
                "a0": {b: standalone[A0_KEY][b]["sharpe"] for b in ("P2", "P3")} | {b: g3[A0_KEY][b]["sharpe"] for b in ("G-P2", "G-P3")},
                "L": {b: standalone[L_KEY][b]["sharpe"] for b in ("P2", "P3")} | {b: g3[L_KEY][b]["sharpe"] for b in ("G-P2", "G-P3")},
            },
        }
        if "buffer_median_over_offsets" in cand:
            stage_res[stage_str]["buffer_median_full_sharpe_over_offsets"] = cand["buffer_median_over_offsets"]

        # ---- heatmaps: standalone FULL Sharpe, G3 FULL Sharpe, min-block margin vs L
        def arr_of(fn):
            return np.vectorize(fn, otypes=[float])(key_arr)

        sa_arr = arr_of(lambda k: standalone[k]["FULL"]["sharpe"])
        g3_arr = arr_of(lambda k: g3[k]["G-FULL"]["sharpe"])
        margin_arr = arr_of(lambda k: min(g3[k][b]["sharpe"] - g3[L_KEY][b]["sharpe"] for b in RULE_BLOCK_TUPLE))
        wide_bool = stage_str == "S4_buffer_offset"
        fig, axes = plt.subplots(3 if wide_bool else 1, 1 if wide_bool else 3,
                                 figsize=(15, 8.5) if wide_bool else (15, 0.55 * len(row_list) + 2.2), facecolor=SURFACE)
        titles = [f"Standalone Sharpe 2000-26 (L = {standalone[L_KEY]['FULL']['sharpe']:.2f})",
                  f"G3 Sharpe 2012-26 (L = {g3[L_KEY]['G-FULL']['sharpe']:.2f})",
                  "Worst G3 block margin vs L (blue = beats L in all 3 blocks)"]
        for ax, arr, title, cmap, lim in zip(axes, (sa_arr, g3_arr, margin_arr), titles, (seq_cmap, seq_cmap, div_cmap),
                                              ((np.nanmin(sa_arr), np.nanmax(sa_arr)), (np.nanmin(g3_arr), np.nanmax(g3_arr)), (-0.35, 0.35))):
            heatmap(ax, arr, row_list, col_list, title, cmap, lim[0], lim[1], a0_pos, cand["centre"], fontsize=7 if wide_bool else 8)
        fig.suptitle(f"{stage_str}: black box = live setting (A0), dashed = plateau centre", fontsize=10, color=TEXT_SECONDARY, x=0.01, ha="left")
        fig.tight_layout()
        fig.savefig(CHART_PATH / f"heatmap_{stage_str}_NDX.png", dpi=130)
        plt.close(fig)

        # ---- other universes
        if other_dict:
            fig, axes = plt.subplots(3 if wide_bool else 1, 1 if wide_bool else 3,
                                     figsize=(15, 8.5) if wide_bool else (15, 0.55 * len(row_list) + 2.2), facecolor=SURFACE)
            for ax, universe_str in zip(axes, ("NDX", "SP500", "R1000")):
                src = standalone if universe_str == "NDX" else other_dict.get(universe_str)
                if src is None:
                    continue
                arr = arr_of(lambda k: src[k]["FULL"]["sharpe"])
                heatmap(ax, arr, row_list, col_list, f"{universe_str}: standalone Sharpe 2000-26", seq_cmap,
                        np.nanmin(arr), np.nanmax(arr), a0_pos, cand["centre"], fontsize=7 if wide_bool else 8)
            fig.tight_layout()
            fig.savefig(CHART_PATH / f"heatmap_{stage_str}_universes.png", dpi=130)
            plt.close(fig)
    res["stages"] = stage_res

    # ------------------------------------------------------------------------------------------------ S5 reference
    no_vxn_key = stage_map_dict["S5_reference"][0]["key"]
    res["no_vxn_reference"] = {"standalone": standalone[no_vxn_key], "g3": g3[no_vxn_key]}

    # ------------------------------------------------------------------------------------------------ timing luck
    offset_list = list(core.STAGE4_OFFSET_TUPLE)
    row_list, col_list, key_arr = stage_matrix(stage_map_dict["S4_buffer_offset"])
    l_offset_key_list = [k for k in ret_dict["engine"].columns if k.startswith("ROC12/ATR20_LIVE")]
    l_offset_key_list = sorted(l_offset_key_list, key=lambda k: int(k.split("|k")[1].split("|")[0]))
    luck_dict = {}
    series_for_plot = {"L (live score)": l_offset_key_list}
    for r, buffer_int in enumerate(row_list):
        series_for_plot[f"A0 score, buffer {buffer_int}"] = list(key_arr[r])
    for label_str, key_list in series_for_plot.items():
        luck_dict[label_str] = {
            metric: {"k0": None, "min": None, "median": None, "max": None} for metric in ("standalone_FULL", "G-FULL", "G-P1", "G-P2", "G-P3")
        }
        for metric in luck_dict[label_str]:
            vec = np.array([standalone[k]["FULL"]["sharpe"] if metric == "standalone_FULL" else g3[k][metric]["sharpe"] for k in key_list])
            luck_dict[label_str][metric] = {"k0": float(vec[offset_list.index(0)]), "min": float(vec.min()),
                                            "median": float(np.median(vec)), "max": float(vec.max()),
                                            "k0_percentile": float(np.mean(vec <= vec[offset_list.index(0)]))}
    a0_offsets = list(key_arr[0])
    luck_dict["L_beats_A0_same_offset_share"] = {
        b: float(np.mean([g3[l][b]["sharpe"] > g3[a][b]["sharpe"] for l, a in zip(l_offset_key_list, a0_offsets)]))
        for b in ("G-FULL",) + RULE_BLOCK_TUPLE
    }
    luck_dict["L_beats_A0_same_offset_share"]["standalone_FULL"] = float(
        np.mean([standalone[l]["FULL"]["sharpe"] > standalone[a]["FULL"]["sharpe"] for l, a in zip(l_offset_key_list, a0_offsets)])
    )
    # Sensitivity (amendment A3): k > 0 schedules first trade in mid-February 2000, k <= 0 in January, so the
    # PREREG standalone window compares different starts. Re-measure from a common start: the day after the latest
    # first execution across all offsets.
    common_start_ts = max(
        ret_dict["engine"][key].ne(0).idxmax() for key in l_offset_key_list + list(key_arr[0])
    ) + pd.Timedelta(days=1)
    luck_dict["common_start_date"] = str(common_start_ts.date())
    for label_str, key_list in series_for_plot.items():
        vec = np.array([metric_dict(ret_dict["engine"][k].loc[common_start_ts:END_TS])["sharpe"] for k in key_list])
        luck_dict[label_str]["standalone_common_start"] = {
            "k0": float(vec[offset_list.index(0)]), "min": float(vec.min()), "median": float(np.median(vec)),
            "max": float(vec.max()), "k0_percentile": float(np.mean(vec <= vec[offset_list.index(0)]))}
    res["timing_luck"] = luck_dict
    # timing luck chart
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.2), facecolor=SURFACE)
    colors = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
    for ax, metric, title in zip(axes, ("standalone", "G-FULL"), ("Standalone Sharpe 2000-26", "G3 Sharpe 2012-26")):
        for (label_str, key_list), color in zip(series_for_plot.items(), colors):
            vec = [standalone[k]["FULL"]["sharpe"] if metric == "standalone" else g3[k]["G-FULL"]["sharpe"] for k in key_list]
            ax.plot(offset_list, vec, color=color, lw=2 if label_str.startswith("L") else 1.4, marker="o", ms=4, label=label_str)
        ax.axvline(0, color=TEXT_SECONDARY, lw=0.8, ls=":")
        ax.set_xticks(range(-10, 11, 2))
        ax.set_title(title + " by rebalance day (sessions from month-end)", fontsize=9.5, loc="left", color=TEXT_PRIMARY)
        ax.set_xlabel("rebalance offset k (0 = live: last session of the month)", fontsize=8, color=TEXT_SECONDARY)
        ax.grid(axis="y", color="#e4e3df", lw=0.6)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.tick_params(labelsize=8, colors=TEXT_SECONDARY)
    axes[0].legend(fontsize=7.5, frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(CHART_PATH / "timing_luck_NDX.png", dpi=130)
    plt.close(fig)

    # ------------------------------------------------------------------------------------------------ multiplicity
    candidate_key_list = sorted({stage_res[s]["candidate_centre"] for s in STAGE_TUPLE} | {A0_KEY})
    sf_keys = scale_free_key_list
    g3_full_df = g3_frame(ret_dict["engine"][sf_keys + [L_KEY]], taa_long_ser).loc[G3_BLOCK_DICT["G-FULL"][0]:END_TS]
    sa_full_df = ret_dict["engine"][sf_keys + [L_KEY]].loc[BLOCK_DICT["FULL"][0]:END_TS]
    res["multiplicity"] = {
        "reality_check_g3_full": reality_check(g3_full_df, candidate_key_list, BOOT_SEED_INT),
        "reality_check_standalone_full": reality_check(sa_full_df, candidate_key_list, BOOT_SEED_INT + 1),
    }
    sharpe_daily_vec = (sa_full_df[sf_keys].mean() / sa_full_df[sf_keys].std()).to_numpy()
    common_ts = pd.Timestamp(luck_dict["common_start_date"])
    res["multiplicity"]["reality_check_standalone_common_start"] = reality_check(
        ret_dict["engine"][sf_keys + [L_KEY]].loc[common_ts:END_TS], candidate_key_list, BOOT_SEED_INT + 2)
    res["multiplicity"]["deflated_sharpe"] = {k: deflated_sharpe(sa_full_df[k], sharpe_daily_vec) for k in candidate_key_list}
    res["multiplicity"]["deflated_sharpe_L_same_haircut"] = deflated_sharpe(sa_full_df[L_KEY], sharpe_daily_vec)

    # ------------------------------------------------------------------------------------------------ cross-universe
    cross_dict = {}
    for universe_str, d in other_dict.items():
        keys = [k for k in sf_keys if k in d]
        a = [standalone[k]["FULL"]["sharpe"] for k in keys]
        b = [d[k]["FULL"]["sharpe"] for k in keys]
        cross_dict[universe_str] = {"spearman_all_cells": float(stats.spearmanr(a, b).statistic),
                                    "A0": d[A0_KEY]["FULL"], "L": d[L_KEY]["FULL"], "B": d[B_KEY]["FULL"],
                                    "by_stage": {}}
        for stage_str in STAGE_TUPLE:
            ks = list(dict.fromkeys(item["key"] for item in stage_map_dict[stage_str]))
            cross_dict[universe_str]["by_stage"][stage_str] = float(
                stats.spearmanr([standalone[k]["FULL"]["sharpe"] for k in ks], [d[k]["FULL"]["sharpe"] for k in ks]).statistic
            )
    res["cross_universe"] = cross_dict

    # ------------------------------------------------------------------------------------------------ turnover / capacity
    def overlap_with_l(key_str: str) -> float:
        l_sel, k_sel = selections_dict[L_KEY], selections_dict[key_str]
        vals = [len(set(l_sel[d]) & set(k_sel[d])) for d in l_sel if l_sel[d] and d in k_sel]
        return float(np.mean(vals)) if vals else float("nan")

    def replaced_share(key_str: str) -> float:
        sel = list(selections_dict[key_str].values())
        vals = [1 - len(set(a) & set(b)) / max(len(a), 1) for a, b in zip(sel[:-1], sel[1:]) if a and b]
        return float(np.mean(vals))

    res["turnover_capacity"] = {
        k: {"turnover_x_per_year": meta_dict[k]["turnover_x_per_year"], "commission_pct_nav": meta_dict[k]["commission_pct_nav_per_year"],
            "share_replaced_per_rebalance": replaced_share(k), "mean_overlap_with_L": overlap_with_l(k),
            "capacity_full": meta_dict[k]["capacity_full"], "capacity_2021_2026": meta_dict[k]["capacity_2021_2026"],
            "mean_target_exposure": meta_dict[k]["mean_target_exposure"]}
        for k in candidate_key_list + [L_KEY]
    }

    # ------------------------------------------------------------------------------------------------ post-hoc
    # NOT pre-registered; descriptive only, added after the rule was applied. Changes no verdict.
    no_gate_key = "ROC12/NATR20|N10EW|F0|Gnone|b0|k+0|V22-0.25"
    sma200_key = "ROC12/NATR20|N10EW|F200|GSPY|b0|k+0|V22-0.25"
    eng_df = ret_dict["engine"].loc[:END_TS]
    year_df = (1 + eng_df[[L_KEY, A0_KEY, sma200_key, no_gate_key]]).groupby(eng_df.index.year).prod() - 1
    year_df.columns = ["L", "A0", "A0_SMA200", "A0_no_filter_no_regime"]
    a0_offset_key_list = [key_arr[0][c] for c in range(len(offset_list))]
    tranche_df = pd.DataFrame({
        "L_21_day_tranches": eng_df[l_offset_key_list].mean(axis=1),
        "A0_21_day_tranches": eng_df[a0_offset_key_list].mean(axis=1),
    })
    tranche_g3_df = g3_frame(tranche_df, taa_long_ser)
    g3_p3_df = g3_frame(eng_df[[L_KEY, A0_KEY, sma200_key, no_gate_key]], taa_long_ser).loc["2022-01-01":END_TS]
    res["posthoc_not_preregistered"] = {
        "calendar_year_returns": {str(y): {c: float(v) for c, v in row.items()} for y, row in year_df.iterrows()},
        "a0_beats_L_years_int": int((year_df["A0"] > year_df["L"]).sum()),
        "years_int": int(len(year_df)),
        "g3_p3_sharpe_ex_2025": {c: metric_dict(g3_p3_df[c][g3_p3_df.index.year != 2025])["sharpe"] for c in g3_p3_df},
        "tranche_average_over_21_rebalance_days": {
            c: {"standalone": {b: metric_dict(tranche_df[c].loc[s:e]) for b, (s, e) in BLOCK_DICT.items()},
                "g3": {b: metric_dict(tranche_g3_df[c].loc[s:e]) for b, (s, e) in G3_BLOCK_DICT.items()}}
            for c in tranche_df
        },
        "no_filter_no_regime_cell": {"standalone": standalone[no_gate_key], "g3": g3[no_gate_key], "g3_stress": g3_stress[no_gate_key]},
    }

    # ------------------------------------------------------------------------------------------------ decision
    # PREREG section 8 read literally: an A0-centred stage can pass too (that would mean switching L -> A0), and the
    # shadow line is chosen over ALL stage candidates, A0-centred ones included.
    passing_list = [s for s in STAGE_TUPLE if stage_res[s]["rule"]["passes"]]
    if passing_list:
        decision_str = "combine passing stages (evaluate once, post-selection): " + ", ".join(passing_list)
        shadow_key = None
    else:
        decision_str = "keep L"
        shadow_stage = max(STAGE_TUPLE, key=lambda s: stage_res[s]["rule"]["engine"]["min_margin"])
        shadow_key = stage_res[shadow_stage]["candidate_centre"]
    res["decision"] = {"passing_stages": passing_list, "decision": decision_str, "shadow_line": shadow_key}
    (OUT_PATH / "results.json").write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps(res["decision"], indent=1))
    for s in STAGE_TUPLE:
        r = stage_res[s]
        print(s, "| centre:", r["candidate_centre"], "| A0 rank", r["a0_rank_full_sharpe"], "/", r["cells_int"],
              "| margins", {b: round(v, 3) for b, v in r["rule"]["engine"]["sharpe_margin_vs_L"].items()},
              "| dd", {b: round(v, 2) for b, v in r["rule"]["engine"]["dd_gap_vs_L_pp"].items()},
              "| R4", {u: v["pass"] for u, v in r["rule"]["R4"].items()}, "| passes", r["rule"]["passes"])


if __name__ == "__main__":
    main()
