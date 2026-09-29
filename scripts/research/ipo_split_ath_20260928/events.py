"""Eligibility (liquidity rank) and the event populations (PREREG section 3).

    rank_t(i)   = position of ADV_t(i) in descending order among all study securities with a finite ADV on t
    eligible_K  = rank_t(i) <= K  and  Unadjusted Close_t(i) >= $5

Populations at the close of t (all require eligible_K):
    IPO_ATH     new listing (no SPAC), 1 <= age <= 89, ATH
    IPO_ATH_SP  as IPO_ATH but SPAC listings allowed (label S-SPAC)
    SPLIT_ATH   in a forward-split window (0..89 sessions), age >= 252, ATH
    BASE_ATH    ATH, age >= 90, not in a split window
    IPO_ALL     new listing (no SPAC), 1 <= age <= 89, any day
    SPLIT_ALL   in a split window, age >= 252, any day
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
import features  # noqa: E402

PRICE_FLOOR_FLOAT = 5.0
POPULATION_LIST = ["IPO_ATH", "IPO_ATH_SP", "SPLIT_ATH", "BASE_ATH", "IPO_ALL", "SPLIT_ALL"]


def load_calendar_idx() -> pd.DatetimeIndex:
    return pd.DatetimeIndex(np.load(common.CACHE_DIR_PATH / "calendar.npy"))


def rank_frame(tag_str: str = "") -> pd.DataFrame:
    """Daily ADV rank of every study security (1 = most liquid). Cached."""
    out_path = common.CACHE_DIR_PATH / f"rank{tag_str}.parquet"
    if out_path.exists():
        return pd.read_parquet(out_path)
    panel_df = pd.read_parquet(common.CACHE_DIR_PATH / f"rank_panel{tag_str}.parquet")
    panel_df = panel_df[np.isfinite(panel_df["adv"])]
    # *** CRITICAL*** the rank is cross-sectional within the same session t only (ADV_t is itself trailing).
    panel_df = panel_df.sort_values(["cal_pos", "adv", "sym_id"], ascending=[True, False, True], kind="mergesort")
    panel_df["rank"] = panel_df.groupby("cal_pos").cumcount().astype(np.int32) + 1
    panel_df = panel_df[["cal_pos", "sym_id", "rank"]].reset_index(drop=True)
    panel_df.to_parquet(out_path)
    return panel_df


def load_rows(tag_str: str = "") -> pd.DataFrame:
    rows_df = pd.read_parquet(common.CACHE_DIR_PATH / f"rows{tag_str}.parquet")
    sym_df = pd.read_parquet(common.CACHE_DIR_PATH / "symbols.parquet")[["sym_id", "symbol", "new_listing", "new_listing_incl_spac", "data_start"]]
    rows_df = rows_df.merge(sym_df, on="sym_id", how="left", validate="many_to_one")
    rank_df = rank_frame(tag_str)
    rows_df = rows_df.merge(rank_df, on=["cal_pos", "sym_id"], how="left", validate="one_to_one")
    rows_df["rank"] = rows_df["rank"].fillna(10**9).astype(np.int64)
    calendar_idx = load_calendar_idx()
    rows_df["date"] = calendar_idx[rows_df["cal_pos"].to_numpy()]
    return rows_df


def population_mask_dict(rows_df: pd.DataFrame, top_k_int: int = 1000) -> dict:
    eligible_arr = (rows_df["rank"].to_numpy() <= top_k_int) & (rows_df["uclose"].to_numpy() >= PRICE_FLOOR_FLOAT)
    ath_arr = rows_df["ath"].to_numpy()
    age_arr = rows_df["age"].to_numpy()
    ipo_window_sp_arr = rows_df["in_ipo_window"].to_numpy()  # built with SPAC-inclusive new listings
    ipo_window_arr = ipo_window_sp_arr & rows_df["new_listing"].to_numpy()
    split_window_arr = rows_df["in_split_window"].to_numpy() & (age_arr >= features.SPLIT_MIN_AGE_INT)
    return {
        "IPO_ATH": eligible_arr & ipo_window_arr & ath_arr,
        "IPO_ATH_SP": eligible_arr & ipo_window_sp_arr & ath_arr,
        "SPLIT_ATH": eligible_arr & split_window_arr & ath_arr,
        "BASE_ATH": eligible_arr & ath_arr & (age_arr > features.IPO_WINDOW_MAX_AGE_INT) & ~rows_df["in_split_window"].to_numpy(),
        "IPO_ALL": eligible_arr & ipo_window_arr,
        "SPLIT_ALL": eligible_arr & split_window_arr,
    }


def candidates_by_pos(rows_df: pd.DataFrame, mask_arr: np.ndarray) -> dict:
    """cal_pos -> [(symbol, adv)] ranked by ADV descending, ties by symbol (PREREG section 5, step 2)."""
    sub_df = rows_df.loc[mask_arr, ["cal_pos", "symbol", "adv"]].sort_values(["cal_pos", "adv", "symbol"], ascending=[True, False, True], kind="mergesort")
    out_dict: dict = {}
    for cal_pos_int, symbol_str, adv_float in sub_df.itertuples(index=False):
        out_dict.setdefault(int(cal_pos_int), []).append((symbol_str, float(adv_float)))
    return out_dict


def load_bar_store_dict(symbol_set: set) -> dict:
    bars_dict = {}
    for path_obj in sorted((common.CACHE_DIR_PATH / "bars").glob("chunk_*.pkl")):
        chunk_dict = pd.read_pickle(path_obj)
        for symbol_str, bar_df in chunk_dict.items():
            if symbol_str in symbol_set:
                bars_dict[symbol_str] = bar_df
    return bars_dict
