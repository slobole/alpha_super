"""V1 download-date invariance and V2 random rescaling on real Norgate bars (PREREG section 7).

V1: for T0 in {2001-06-29, 2010-06-30, 2020-06-30} rebuild each sampled symbol as a download on T0 would show it:
    bars after T0 dropped; OHLC x k_T0; Volume / k_T0; Turnover and Unadjusted Close unchanged.
    Every event flag (ath, is_split, in_split_window, in_ipo_window, age, sessions_since_split) over the last 300
    sessions up to T0 must equal the full-panel value; ADV must match to relative 1e-9.
V2: OHLC x c, Volume / c for a random c in [0.1, 10]; the same fields must be unchanged over the whole history.

Usage: python invariance.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE_PATH = Path(__file__).resolve().parent
if str(HERE_PATH) not in sys.path:
    sys.path.insert(0, str(HERE_PATH))

import common  # noqa: E402
import data_build  # noqa: E402
import events  # noqa: E402
import features  # noqa: E402

TRUNCATION_LIST = ["2001-06-29", "2010-06-30", "2020-06-30"]
FLAG_LIST = ["ath", "is_split", "in_split_window", "in_ipo_window", "age", "sessions_since_split"]
SEED_INT = 20260928
SAMPLE_EVENT_INT = 200
SAMPLE_OTHER_INT = 100


def compare_feature_df(a_df: pd.DataFrame, b_df: pd.DataFrame) -> list[str]:
    problem_list = []
    for col_str in FLAG_LIST:
        if not (a_df[col_str].to_numpy() == b_df[col_str].to_numpy()).all():
            problem_list.append(col_str)
    if not np.allclose(a_df["adv"].to_numpy(), b_df["adv"].to_numpy(), rtol=1e-9, atol=0.0, equal_nan=True):
        problem_list.append("adv")
    return problem_list


def main() -> None:
    rng = np.random.default_rng(SEED_INT)
    sym_df = pd.read_parquet(common.CACHE_DIR_PATH / "symbols.parquet")
    rows_df = pd.read_parquet(common.CACHE_DIR_PATH / "rows.parquet", columns=["sym_id", "in_ipo_window", "in_split_window"])
    event_sym_arr = rows_df.loc[rows_df["in_ipo_window"] | rows_df["in_split_window"], "sym_id"].unique()
    other_sym_arr = np.setdiff1d(sym_df["sym_id"].to_numpy(), event_sym_arr)
    sample_arr = np.concatenate([rng.choice(event_sym_arr, SAMPLE_EVENT_INT, replace=False),
                                 rng.choice(other_sym_arr, SAMPLE_OTHER_INT, replace=False)])
    calendar_idx = events.load_calendar_idx()
    report_dict = {"symbols": int(len(sample_arr)), "v1_checks": 0, "v1_failures": [], "v2_checks": 0, "v2_failures": []}
    for sym_id_int in sample_arr:
        record = sym_df.loc[sym_df["sym_id"] == sym_id_int].iloc[0]
        raw_df = data_build.load_symbol_bars_df(record["symbol"])
        bars_df, _ = features.clean_bars_df(raw_df, calendar_idx)
        if len(bars_df) < 3:
            continue
        new_listing_bool = bool(record["new_listing_incl_spac"])
        full_df = features.symbol_feature_df(bars_df, calendar_idx, record["first_quoted"], new_listing_bool)
        for t0_str in TRUNCATION_LIST:
            t0_ts = pd.Timestamp(t0_str)
            cut_df = bars_df.loc[:t0_ts].copy()
            if len(cut_df) < 3:
                continue
            k_t0_float = float(cut_df["Unadjusted Close"].iloc[-1] / cut_df["Close"].iloc[-1])
            cut_df[features.PRICE_FIELD_LIST] = cut_df[features.PRICE_FIELD_LIST] * k_t0_float
            cut_df["Volume"] = cut_df["Volume"] / k_t0_float
            cut_feat_df = features.symbol_feature_df(cut_df, calendar_idx, record["first_quoted"], new_listing_bool)
            last_pos_int = int(calendar_idx.searchsorted(t0_ts, side="right")) - 1
            window_mask_arr = cut_feat_df["cal_pos"].to_numpy() > last_pos_int - 300
            problem_list = compare_feature_df(full_df.loc[cut_df.index].loc[window_mask_arr], cut_feat_df.loc[window_mask_arr])
            report_dict["v1_checks"] += 1
            if problem_list:
                report_dict["v1_failures"].append({"symbol": record["symbol"], "t0": t0_str, "fields": problem_list})
        scale_float = float(rng.uniform(0.1, 10.0))
        scaled_df = bars_df.copy()
        scaled_df[features.PRICE_FIELD_LIST] = scaled_df[features.PRICE_FIELD_LIST] * scale_float
        scaled_df["Volume"] = scaled_df["Volume"] / scale_float
        scaled_feat_df = features.symbol_feature_df(scaled_df, calendar_idx, record["first_quoted"], new_listing_bool)
        report_dict["v2_checks"] += 1
        problem_list = compare_feature_df(full_df, scaled_feat_df)
        if problem_list:
            report_dict["v2_failures"].append({"symbol": record["symbol"], "fields": problem_list})
    report_dict["passed_bool"] = not report_dict["v1_failures"] and not report_dict["v2_failures"]
    common.write_json("invariance.json", report_dict)
    common.log_progress(f"invariance: v1 {report_dict['v1_checks']} checks, {len(report_dict['v1_failures'])} failures; "
                        f"v2 {report_dict['v2_checks']} checks, {len(report_dict['v2_failures'])} failures")


if __name__ == "__main__":
    main()
