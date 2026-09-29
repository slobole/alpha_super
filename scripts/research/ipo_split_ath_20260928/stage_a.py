"""Stage A - the event study (PREREG section 4).

    X_h = R_h - M_h,   M_h = SPY_TR Open_{exit} / SPY_TR Open_{entry} - 1   (Close at exit when the stock's exit is a close)
    D_m = mean X_20 (population, entered in month m) - mean X_20 (control, entered in month m)
    t   = mean(D) / NW-se(D), Newey-West with 2 lags, over months present in both populations

Usage: python stage_a.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE_PATH = Path(__file__).resolve().parent
if str(HERE_PATH) not in sys.path:
    sys.path.insert(0, str(HERE_PATH))

import common  # noqa: E402
import events  # noqa: E402
import features  # noqa: E402

BLOCK_DICT = {
    "P0": ("1993-01-04", "1999-12-31"),
    "P1": ("2000-01-01", "2011-12-31"),
    "P2": ("2012-01-01", "2021-12-31"),
    "P3": ("2022-01-01", "2026-08-19"),
    "FULL": ("1993-01-04", "2026-08-19"),
}
COMPARISON_LIST = [("IPO_ATH", "BASE_ATH"), ("SPLIT_ATH", "BASE_ATH"), ("IPO_ATH", "IPO_ALL"), ("SPLIT_ATH", "SPLIT_ALL"),
                   ("IPO_ATH_SP", "BASE_ATH")]
NW_LAG_INT = 2


def add_market_excess(rows_df: pd.DataFrame) -> pd.DataFrame:
    spy_df = pd.read_parquet(common.CACHE_DIR_PATH / "spy_tr.parquet")
    spy_open_arr = spy_df["Open"].to_numpy(dtype=float)
    spy_close_arr = spy_df["Close"].to_numpy(dtype=float)
    entry_pos_arr = rows_df["entry_cal_pos"].to_numpy()
    valid_entry_arr = entry_pos_arr >= 0
    safe_entry_arr = np.where(valid_entry_arr, entry_pos_arr, 0)
    for horizon_int in features.HORIZON_LIST:
        exit_pos_arr = rows_df[f"exit_cal_pos{horizon_int}"].to_numpy()
        exit_is_close_arr = rows_df[f"exit_is_close{horizon_int}"].to_numpy()
        spy_exit_arr = np.where(exit_is_close_arr, spy_close_arr[exit_pos_arr], spy_open_arr[exit_pos_arr])
        market_arr = np.where(valid_entry_arr, spy_exit_arr / spy_open_arr[safe_entry_arr] - 1.0, np.nan)
        rows_df[f"M{horizon_int}"] = market_arr
        rows_df[f"X{horizon_int}"] = rows_df[f"R{horizon_int}"].to_numpy() - market_arr
    return rows_df


def newey_west_t(value_vec: np.ndarray, lag_int: int = NW_LAG_INT) -> float:
    value_vec = value_vec[np.isfinite(value_vec)]
    n_int = len(value_vec)
    if n_int < 5:
        return float("nan")
    dev_vec = value_vec - value_vec.mean()
    gamma0_float = float(dev_vec @ dev_vec) / n_int
    long_run_float = gamma0_float
    for lag in range(1, lag_int + 1):
        gamma_float = float(dev_vec[lag:] @ dev_vec[:-lag]) / n_int
        long_run_float += 2.0 * (1.0 - lag / (lag_int + 1)) * gamma_float
    return float(value_vec.mean() / np.sqrt(long_run_float / n_int))


def describe(sub_df: pd.DataFrame) -> dict:
    out_dict = {"count": int(len(sub_df))}
    if len(sub_df) == 0:
        return out_dict
    years_float = max((sub_df["date"].max() - sub_df["date"].min()).days / 365.25, 1e-9)
    out_dict["per_year"] = float(len(sub_df) / years_float)
    out_dict["symbols"] = int(sub_df["sym_id"].nunique())
    r_vec = sub_df["R20"].dropna().to_numpy()
    x_vec = sub_df["X20"].dropna().to_numpy()
    out_dict.update({
        "R20_mean": float(r_vec.mean()), "R20_median": float(np.median(r_vec)), "R20_pos": float((r_vec > 0).mean()),
        "R20_avg_win": float(r_vec[r_vec > 0].mean()) if (r_vec > 0).any() else float("nan"),
        "R20_avg_loss": float(r_vec[r_vec <= 0].mean()) if (r_vec <= 0).any() else float("nan"),
        "X20_mean": float(x_vec.mean()), "X20_median": float(np.median(x_vec)), "X20_pos": float((x_vec > 0).mean()),
        "CC20_mean": float(sub_df["CC20"].dropna().mean()),
        "exit_is_close20_share": float(sub_df["exit_is_close20"].mean()),
    })
    for horizon_int in [1, 5, 10]:
        out_dict[f"R{horizon_int}_mean"] = float(sub_df[f"R{horizon_int}"].dropna().mean())
        out_dict[f"X{horizon_int}_mean"] = float(sub_df[f"X{horizon_int}"].dropna().mean())
    return out_dict


def compare(a_df: pd.DataFrame, b_df: pd.DataFrame) -> dict:
    a_month_ser = a_df.groupby(a_df["date"].dt.to_period("M"))["X20"].mean()
    b_month_ser = b_df.groupby(b_df["date"].dt.to_period("M"))["X20"].mean()
    both_idx = a_month_ser.index.intersection(b_month_ser.index)
    d_vec = (a_month_ser.reindex(both_idx) - b_month_ser.reindex(both_idx)).to_numpy()
    a_vec = a_df["X20"].dropna().to_numpy()
    b_vec = b_df["X20"].dropna().to_numpy()
    welch = stats.ttest_ind(a_vec, b_vec, equal_var=False) if len(a_vec) > 1 and len(b_vec) > 1 else None
    return {
        "months_int": int(len(d_vec)),
        "event_mean_diff": float(a_vec.mean() - b_vec.mean()) if len(a_vec) and len(b_vec) else float("nan"),
        "D_mean": float(np.nanmean(d_vec)) if len(d_vec) else float("nan"),
        "D_share_positive": float(np.mean(d_vec > 0)) if len(d_vec) else float("nan"),
        "clustered_nw_t": newey_west_t(d_vec),
        "naive_welch_t": float(welch.statistic) if welch is not None else float("nan"),
        "naive_welch_p": float(welch.pvalue) if welch is not None else float("nan"),
    }


def main() -> None:
    rows_df = events.load_rows()
    rows_df = add_market_excess(rows_df)
    mask_dict = events.population_mask_dict(rows_df, 1000)
    result_dict = {"populations": {}, "comparisons": {}, "stage_a_answer": {}}
    by_year_list = []
    for pop_str in events.POPULATION_LIST:
        pop_df = rows_df.loc[mask_dict[pop_str]]
        result_dict["populations"][pop_str] = {
            block_str: describe(pop_df[(pop_df["date"] >= start_str) & (pop_df["date"] <= end_str)])
            for block_str, (start_str, end_str) in BLOCK_DICT.items()}
        year_df = pop_df.groupby(pop_df["date"].dt.year).agg(count=("X20", "size"), X20_mean=("X20", "mean"), R20_mean=("R20", "mean"))
        year_df["population"] = pop_str
        by_year_list.append(year_df.reset_index())
    for a_str, b_str in COMPARISON_LIST:
        a_df = rows_df.loc[mask_dict[a_str]]
        b_df = rows_df.loc[mask_dict[b_str]]
        block_out_dict = {}
        for block_str, (start_str, end_str) in BLOCK_DICT.items():
            block_out_dict[block_str] = compare(a_df[(a_df["date"] >= start_str) & (a_df["date"] <= end_str)],
                                                b_df[(b_df["date"] >= start_str) & (b_df["date"] <= end_str)])
        result_dict["comparisons"][f"{a_str}_vs_{b_str}"] = block_out_dict
        if b_str == "BASE_ATH" and a_str in ("IPO_ATH", "SPLIT_ATH"):
            positive_blocks_int = sum(block_out_dict[b]["D_mean"] > 0 for b in ["P0", "P1", "P2", "P3"])
            result_dict["stage_a_answer"][a_str] = {
                "full_clustered_t": block_out_dict["FULL"]["clustered_nw_t"],
                "positive_blocks_of_4": int(positive_blocks_int),
                "edge_present_bool": bool(block_out_dict["FULL"]["clustered_nw_t"] > 2 and positive_blocks_int >= 3),
            }
    pd.concat(by_year_list, ignore_index=True).to_csv(common.RESULTS_DIR_PATH / "stage_a_by_year.csv", index=False)
    common.write_json("stage_a.json", result_dict)
    common.log_progress("stage_a: done")


if __name__ == "__main__":
    main()
