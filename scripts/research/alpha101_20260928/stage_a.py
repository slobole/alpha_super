"""Stage A screen (PREREG section 5; research only): IC, decile long-short, turnover, break-even cost, long-only top
decile, per block, for the 100 alphas, the composites and the controls.

Per decision day t (close of t, members with finite score and finite fr_t = Open_{t+2} / Open_{t+1} - 1):
    IC        Spearman correlation of the score and fr among the eligible names
    deciles   n_dec = floor(n / 10) names in the top and in the bottom decile; ties broken by a fixed pseudo-random
              symbol order (seed 20260928)
    LS_t      mean fr of the top decile minus mean fr of the bottom decile (P&L per $1 long and $1 short; gross 2)
    tau_t     sum |w_t - w_{t-1}| / 2 with w = +1/n_dec (top) and -1/n_dec (bottom): dollars traded per dollar of gross
    c*        mean(LS_t / 2) / mean(tau_t): the cost per dollar traded (per side) at which the gross P&L is spent
    net(c)    LS_t - c x sum |w_t - w_{t-1}|
    LO_t      mean fr of the top decile minus the eligible names' equal-weight fr (gross 1); tau_LO = sum |dw|
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy import stats

from alpha101_20260928 import alphas as alphas_module
from alpha101_20260928 import common
from alpha101_20260928 import data as data_module
from alpha101_20260928 import formulas

TIE_EPS_FLOAT = 1e-12
DAILY_COLUMN_TUPLE = ("ic", "n", "ls", "tau", "lo", "tau_lo", "ic_overnight")


def decision_range(date_index: pd.DatetimeIndex) -> tuple[int, int]:
    """Decision positions [start, end): 2000-01-03..END with fr_t available (t + 2 within the cache)."""
    start_pos_int = int(date_index.searchsorted(common.TRADING_START_TS, side="left"))
    end_pos_int = min(int(date_index.searchsorted(common.END_TS, side="right")), len(date_index) - 2)
    return start_pos_int, end_pos_int


def tie_key_vec(symbol_count_int: int, seed_int: int = common.SEED_INT) -> np.ndarray:
    return np.random.default_rng(seed_int).permutation(symbol_count_int).astype(np.float64) / symbol_count_int


def row_pearson(a_arr: np.ndarray, b_arr: np.ndarray) -> np.ndarray:
    """Row-wise Pearson correlation over cells finite in both (NaN with fewer than 3 cells or a constant row)."""
    both_arr = np.isfinite(a_arr) & np.isfinite(b_arr)
    n_vec = both_arr.sum(axis=1).astype(np.float64)
    a_arr = np.where(both_arr, a_arr, 0.0)
    b_arr = np.where(both_arr, b_arr, 0.0)
    with np.errstate(all="ignore"):
        ma_vec = a_arr.sum(axis=1) / n_vec
        mb_vec = b_arr.sum(axis=1) / n_vec
        da_arr = np.where(both_arr, a_arr - ma_vec[:, None], 0.0)
        db_arr = np.where(both_arr, b_arr - mb_vec[:, None], 0.0)
        cov_vec = (da_arr * db_arr).sum(axis=1)
        denominator_vec = np.sqrt((da_arr * da_arr).sum(axis=1) * (db_arr * db_arr).sum(axis=1))
        out_vec = np.where((n_vec >= 3) & (denominator_vec > 0), cov_vec / denominator_vec, np.nan)
    return out_vec


def daily_ic(score_arr: np.ndarray, fr_arr: np.ndarray, member_arr: np.ndarray) -> np.ndarray:
    eligible_arr = member_arr & np.isfinite(score_arr) & np.isfinite(fr_arr)
    score_masked_arr = np.where(eligible_arr, score_arr, np.nan)
    fr_masked_arr = np.where(eligible_arr, fr_arr, np.nan)
    rank_score_arr = stats.rankdata(score_masked_arr, axis=1, nan_policy="omit")
    rank_fr_arr = stats.rankdata(fr_masked_arr, axis=1, nan_policy="omit")
    return row_pearson(rank_score_arr, rank_fr_arr)


def daily_screen(score_arr: np.ndarray, fr_arr: np.ndarray, member_arr: np.ndarray, tie_vec: np.ndarray, overnight_arr: np.ndarray | None = None) -> pd.DataFrame:
    """All daily series of the screen for one score panel (rows = decision sessions)."""
    eligible_arr = member_arr & np.isfinite(score_arr) & np.isfinite(fr_arr)
    n_vec = eligible_arr.sum(axis=1)
    ndec_vec = n_vec // 10
    score_masked_arr = np.where(eligible_arr, score_arr, np.nan)
    fr_masked_arr = np.where(eligible_arr, fr_arr, np.nan)
    ic_vec = row_pearson(stats.rankdata(score_masked_arr, axis=1, nan_policy="omit"), stats.rankdata(fr_masked_arr, axis=1, nan_policy="omit"))
    key_arr = score_masked_arr + TIE_EPS_FLOAT * tie_vec[None, :]
    order_arr = np.argsort(key_arr, axis=1, kind="stable")  # NaN last
    position_arr = np.empty_like(order_arr)
    np.put_along_axis(position_arr, order_arr, np.arange(order_arr.shape[1])[None, :].repeat(order_arr.shape[0], axis=0), axis=1)
    bottom_arr = eligible_arr & (position_arr < ndec_vec[:, None])
    top_arr = eligible_arr & (position_arr >= (n_vec - ndec_vec)[:, None])
    valid_vec = ndec_vec > 0
    fr_filled_arr = np.where(eligible_arr, fr_arr, 0.0)
    with np.errstate(all="ignore"):
        top_mean_vec = (fr_filled_arr * top_arr).sum(axis=1) / ndec_vec
        bottom_mean_vec = (fr_filled_arr * bottom_arr).sum(axis=1) / ndec_vec
        all_mean_vec = fr_filled_arr.sum(axis=1) / n_vec
        weight_arr = (top_arr.astype(np.float64) - bottom_arr.astype(np.float64)) / np.where(ndec_vec > 0, ndec_vec, 1)[:, None]
        weight_lo_arr = top_arr.astype(np.float64) / np.where(ndec_vec > 0, ndec_vec, 1)[:, None]
    ls_vec = np.where(valid_vec, top_mean_vec - bottom_mean_vec, np.nan)
    lo_vec = np.where(valid_vec, top_mean_vec - all_mean_vec, np.nan)
    tau_vec = np.full(len(n_vec), np.nan)
    tau_lo_vec = np.full(len(n_vec), np.nan)
    tau_vec[1:] = np.abs(np.diff(weight_arr, axis=0)).sum(axis=1) / 2.0
    tau_lo_vec[1:] = np.abs(np.diff(weight_lo_arr, axis=0)).sum(axis=1)
    both_valid_vec = valid_vec.copy()
    both_valid_vec[1:] &= valid_vec[:-1]
    tau_vec[~both_valid_vec] = np.nan
    tau_lo_vec[~both_valid_vec] = np.nan
    out_dict = {"ic": ic_vec, "n": n_vec.astype(np.float64), "ls": ls_vec, "tau": tau_vec, "lo": lo_vec, "tau_lo": tau_lo_vec}
    if overnight_arr is not None:
        out_dict["ic_overnight"] = daily_ic(score_arr, overnight_arr, member_arr)
    else:
        out_dict["ic_overnight"] = np.full(len(n_vec), np.nan)
    return pd.DataFrame(out_dict)


def block_stats(daily_df: pd.DataFrame, start_str: str, end_str: str) -> dict:
    window_df = daily_df.loc[start_str:end_str]
    ic_vec = window_df["ic"].to_numpy()
    ls_vec = window_df["ls"].to_numpy()
    tau_vec = window_df["tau"].to_numpy()
    lo_vec = window_df["lo"].to_numpy()
    tau_lo_vec = window_df["tau_lo"].to_numpy()
    ok_ic_vec = np.isfinite(ic_vec)
    ok_ls_vec = np.isfinite(ls_vec)
    ok_tau_vec = np.isfinite(tau_vec) & ok_ls_vec
    ok_lo_vec = np.isfinite(lo_vec) & np.isfinite(tau_lo_vec)

    def sharpe(x_vec: np.ndarray) -> float:
        x_vec = x_vec[np.isfinite(x_vec)]
        return float(x_vec.mean() / x_vec.std(ddof=1) * np.sqrt(252.0)) if len(x_vec) > 2 and x_vec.std(ddof=1) > 0 else float("nan")

    out_dict = {
        "days_int": int(ok_ic_vec.sum()),
        "ic_mean": float(np.nanmean(ic_vec)) if ok_ic_vec.any() else float("nan"),
        "ic_t_nw5": common.newey_west_t(ic_vec[ok_ic_vec]) if ok_ic_vec.sum() > 10 else float("nan"),
        "ic_positive_share": float((ic_vec[ok_ic_vec] > 0).mean()) if ok_ic_vec.any() else float("nan"),
        "ic_overnight_mean": float(np.nanmean(window_df["ic_overnight"].to_numpy())) if np.isfinite(window_df["ic_overnight"].to_numpy()).any() else float("nan"),
        "ls_mean_ann": float(np.nanmean(ls_vec) * 252.0) if ok_ls_vec.any() else float("nan"),
        "ls_sharpe": sharpe(ls_vec),
        "tau_mean": float(np.nanmean(tau_vec[ok_tau_vec])) if ok_tau_vec.any() else float("nan"),
    }
    if ok_tau_vec.any() and np.nanmean(tau_vec[ok_tau_vec]) > 0:
        out_dict["c_star_bps"] = float(np.nanmean(ls_vec[ok_tau_vec] / 2.0) / np.nanmean(tau_vec[ok_tau_vec]) * 1e4)
    else:
        out_dict["c_star_bps"] = float("nan")
    for cost_bps in common.COST_BPS_TUPLE:
        net_vec = ls_vec[ok_tau_vec] - cost_bps * 1e-4 * 2.0 * tau_vec[ok_tau_vec]
        out_dict[f"net_sharpe_{int(cost_bps)}bps"] = sharpe(net_vec)
        out_dict[f"net_mean_ann_{int(cost_bps)}bps"] = float(net_vec.mean() * 252.0) if len(net_vec) else float("nan")
    out_dict["lo_mean_ann"] = float(np.nanmean(lo_vec) * 252.0) if np.isfinite(lo_vec).any() else float("nan")
    out_dict["lo_tau_mean"] = float(np.nanmean(tau_lo_vec[ok_lo_vec])) if ok_lo_vec.any() else float("nan")
    out_dict["lo_c_star_bps"] = float(np.nanmean(lo_vec[ok_lo_vec]) / np.nanmean(tau_lo_vec[ok_lo_vec]) * 1e4) if ok_lo_vec.any() and np.nanmean(tau_lo_vec[ok_lo_vec]) > 0 else float("nan")
    return out_dict


def residual_score(score_arr: np.ndarray, control_arr: np.ndarray, member_arr: np.ndarray) -> np.ndarray:
    """Residual of a daily cross-sectional regression of rank(score) on rank(control) (with intercept), members only."""
    eligible_arr = member_arr & np.isfinite(score_arr) & np.isfinite(control_arr)
    y_arr = stats.rankdata(np.where(eligible_arr, score_arr, np.nan), axis=1, nan_policy="omit")
    x_arr = stats.rankdata(np.where(eligible_arr, control_arr, np.nan), axis=1, nan_policy="omit")
    n_vec = eligible_arr.sum(axis=1).astype(np.float64)
    x_filled_arr = np.where(eligible_arr, x_arr, 0.0)
    y_filled_arr = np.where(eligible_arr, y_arr, 0.0)
    with np.errstate(all="ignore"):
        mx_vec = x_filled_arr.sum(axis=1) / n_vec
        my_vec = y_filled_arr.sum(axis=1) / n_vec
        dx_arr = np.where(eligible_arr, x_arr - mx_vec[:, None], 0.0)
        dy_arr = np.where(eligible_arr, y_arr - my_vec[:, None], 0.0)
        beta_vec = (dx_arr * dy_arr).sum(axis=1) / (dx_arr * dx_arr).sum(axis=1)
        residual_arr = np.where(eligible_arr, dy_arr - beta_vec[:, None] * dx_arr, np.nan)
    return residual_arr


# ----------------------------------------------------------------------------------------------------------------------
def signal_panels(universe_str: str, start_pos_int: int, end_pos_int: int, member_arr: np.ndarray) -> dict[str, np.ndarray]:
    """All non-alpha score panels of the screen for the decision window."""
    from alpha101_20260928 import composites

    control_dict = composites.load_controls(universe_str)
    panel_dict = {
        "C_EQ": np.asarray(composites.load_composite(universe_str, "C_EQ")[start_pos_int:end_pos_int]),
        "C_WF": np.asarray(composites.load_composite(universe_str, "C_WF")[start_pos_int:end_pos_int]),
        "C_EQ_noInd": np.asarray(composites.load_composite(universe_str, "C_EQ_noInd")[start_pos_int:end_pos_int]),
        "REV1": control_dict["REV1_raw"][start_pos_int:end_pos_int],
        "REV5": control_dict["REV5_raw"][start_pos_int:end_pos_int],
    }
    panel_dict["C_EQ_resid_REV5"] = residual_score(panel_dict["C_EQ"], panel_dict["REV5"], member_arr)
    return panel_dict


def run(universe_str: str) -> dict:
    universe_dict = data_module.load_universe(universe_str)
    date_index = data_module.date_index_of(universe_dict)
    start_pos_int, end_pos_int = decision_range(date_index)
    decision_index = date_index[start_pos_int:end_pos_int]
    member_arr = universe_dict["member_arr"][start_pos_int:end_pos_int] == 1
    fr_arr = data_module.forward_return_panel(universe_dict["open_arr"])[start_pos_int:end_pos_int]
    overnight_arr = data_module.overnight_panel(universe_dict["open_arr"], universe_dict["close_arr"])[start_pos_int:end_pos_int]
    tie_vec = tie_key_vec(len(universe_dict["symbol_list"]))
    z_store = alphas_module.load_z(universe_str)
    daily_dict: dict[str, pd.DataFrame] = {}
    for alpha_int in formulas.ALPHA_NUMBER_TUPLE:
        z_arr = np.asarray(z_store[alphas_module.ALPHA_INDEX_DICT[alpha_int], start_pos_int:end_pos_int], dtype=np.float64)
        daily_dict[str(alpha_int)] = daily_screen(z_arr, fr_arr, member_arr, tie_vec, overnight_arr)
    for name_str, panel_arr in signal_panels(universe_str, start_pos_int, end_pos_int, member_arr).items():
        daily_dict[name_str] = daily_screen(panel_arr, fr_arr, member_arr, tie_vec, overnight_arr)
    for name_str, frame_df in daily_dict.items():
        frame_df.index = decision_index
    daily_df = pd.concat(daily_dict, axis=1)
    daily_df.columns = [f"{a}|{b}" for a, b in daily_df.columns]
    daily_df.to_parquet(common.CACHE_DIR_PATH / f"stageA_daily_{universe_str}.parquet")
    # per-signal per-block table
    row_list = []
    for name_str, frame_df in daily_dict.items():
        for block_str, (start_str, end_str) in common.STAGE_A_BLOCK_DICT.items():
            row = {"universe": universe_str, "signal": name_str, "block": block_str, **block_stats(frame_df, start_str, end_str)}
            row["is_alpha"] = name_str.isdigit()
            if name_str.isdigit():
                n = int(name_str)
                row["indneutralize"] = n in formulas.INDNEUTRALIZE_ALPHA_TUPLE
                row["delay0"] = n in formulas.DELAY0_ALPHA_TUPLE
                row["ts_arg"] = n in formulas.TS_ARG_ALPHA_TUPLE
            row_list.append(row)
    table_df = pd.DataFrame(row_list)
    table_path = common.RESULTS_DIR_PATH / (f"stageA_alpha_table.csv" if universe_str == "U1" else f"stageA_alpha_table_{universe_str}.csv")
    table_df.to_csv(table_path, index=False)
    # family summary
    alpha_df = table_df[table_df["is_alpha"]]
    summary_dict: dict = {"universe": universe_str, "alphas_int": int(alpha_df["signal"].nunique()), "blocks": {}}
    for block_str in common.STAGE_A_BLOCK_DICT:
        block_df = alpha_df[alpha_df["block"] == block_str]
        summary_dict["blocks"][block_str] = {
            "median_ic": float(block_df["ic_mean"].median()),
            "mean_ic": float(block_df["ic_mean"].mean()),
            "median_abs_ic": float(block_df["ic_mean"].abs().median()),
            "median_c_star_bps": float(block_df["c_star_bps"].median()),
            "median_ls_sharpe": float(block_df["ls_sharpe"].median()),
            "median_net_sharpe_3bps": float(block_df["net_sharpe_3bps"].median()),
            "median_net_sharpe_8bps": float(block_df["net_sharpe_8bps"].median()),
            "median_tau": float(block_df["tau_mean"].median()),
            "count_ic_t_above_2": int((block_df["ic_t_nw5"] > 2).sum()),
            "count_ic_t_below_minus_2": int((block_df["ic_t_nw5"] < -2).sum()),
            "count_c_star_above_3bps": int((block_df["c_star_bps"] > 3).sum()),
            "count_c_star_above_8bps": int((block_df["c_star_bps"] > 8).sum()),
            "alphas_with_ic_t_above_2": [int(s) for s in block_df.loc[block_df["ic_t_nw5"] > 2, "signal"]],
            "expected_false_positives_at_t2_one_sided": round(100 * (1 - stats.norm.cdf(2.0)), 2),
        }
    paper_start_str, paper_end_str = common.LABEL_BLOCK_DICT["PAPER"]
    ls_paper_df = pd.DataFrame({name_str: frame_df.loc[paper_start_str:paper_end_str, "ls"] for name_str, frame_df in daily_dict.items() if name_str.isdigit()})
    corr_arr = ls_paper_df.corr().to_numpy()
    off_diag_vec = corr_arr[np.triu_indices_from(corr_arr, k=1)]
    summary_dict["paper_window_mean_pairwise_ls_correlation"] = float(np.nanmean(off_diag_vec))
    summary_dict["paper_reported_mean_pairwise_correlation"] = 0.159
    summary_dict["composites"] = {name_str: {block_str: block_stats(daily_dict[name_str], s, e) for block_str, (s, e) in common.STAGE_A_BLOCK_DICT.items()}
                                  for name_str in ("C_EQ", "C_WF", "C_EQ_noInd", "C_EQ_resid_REV5", "REV1", "REV5")}
    summary_dict["composite_ic_correlation_full"] = {
        f"{a}_vs_{b}": float(daily_dict[a]["ic"].corr(daily_dict[b]["ic"])) for a, b in (("C_EQ", "REV5"), ("C_EQ", "REV1"), ("C_EQ", "C_WF"), ("C_WF", "REV5"))
    }
    summary_dict["composite_ls_correlation_full"] = {
        f"{a}_vs_{b}": float(daily_dict[a]["ls"].corr(daily_dict[b]["ls"])) for a, b in (("C_EQ", "REV5"), ("C_EQ", "REV1"), ("C_EQ", "C_WF"), ("C_WF", "REV5"))
    }
    summary_path = common.RESULTS_DIR_PATH / ("stageA_family_summary.json" if universe_str == "U1" else f"stageA_family_summary_{universe_str}.json")
    summary_path.write_text(json.dumps(summary_dict, indent=1, default=float), encoding="utf-8")
    if universe_str == "U1":
        common.write_json("stageA_composites.json", summary_dict["composites"])
    common.log_progress(f"stage A {universe_str}: " + json.dumps({b: {k: round(v, 4) if isinstance(v, float) else v for k, v in d.items() if k in ("median_ic", "median_c_star_bps", "count_ic_t_above_2")} for b, d in summary_dict["blocks"].items()}))
    return summary_dict
