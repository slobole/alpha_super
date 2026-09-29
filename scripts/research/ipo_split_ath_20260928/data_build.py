"""Build the study caches from Norgate (research only; PREREG section 3).

Outputs in results/research/ipo_split_ath_20260928/_cache/:
    calendar.npy            market sessions ($SPX), datetime64
    symbols.parquet         sym_id -> symbol, new_listing, data_start, first_quoted, blank_check_first
    rank_panel.parquet      (cal_pos, sym_id, adv) for every bar from 1992-11-02 (liquidity rank input)
    rows.parquet            bars that are an ATH, in an IPO window or in a split window, 1993-01-04..END,
                            with features and forward returns
    splits.parquet          every detected forward-split ex-date (for counts and the spot check)
    bars/<chunk>.pkl        OHLC / Unadjusted Close / Dividend of symbols with any IPO-ATH or SPLIT-ATH row
    spy_tr.parquet          SPY TOTALRETURN Open / Close on the calendar

Usage: python data_build.py [workers]
"""

from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

HERE_PATH = Path(__file__).resolve().parent
if str(HERE_PATH) not in sys.path:
    sys.path.insert(0, str(HERE_PATH))

import common  # noqa: E402
import features  # noqa: E402

START_ROWS_TS = pd.Timestamp("1993-01-04")
START_RANK_TS = pd.Timestamp("1992-11-02")
END_TS = pd.Timestamp("2026-08-19")
ALLOWED_SUBTYPE2_PATH = common.CACHE_DIR_PATH / "allowed_subtype2.json"
BAR_FIELD_LIST = ["Open", "High", "Low", "Close", "Unadjusted Close", "Dividend"]


def load_calendar_idx() -> pd.DatetimeIndex:
    import norgatedata as nd

    spx_df = nd.price_timeseries("$SPX", start_date="1990-01-01", timeseriesformat="pandas-dataframe")
    return pd.DatetimeIndex(spx_df.index)


def load_symbol_bars_df(symbol_str: str) -> pd.DataFrame:
    import norgatedata as nd

    return nd.price_timeseries(
        symbol_str,
        stock_price_adjustment_setting=nd.StockPriceAdjustmentType.CAPITALSPECIAL,
        padding_setting=nd.PaddingType.NONE,
        start_date="1990-01-01",
        timeseriesformat="pandas-dataframe",
    )


def process_chunk(task_tuple) -> dict:
    chunk_int, symbol_record_list, calendar_arr = task_tuple
    calendar_idx = pd.DatetimeIndex(calendar_arr)
    start_rows_pos_int = int(calendar_idx.searchsorted(START_ROWS_TS))
    end_pos_int = int(calendar_idx.searchsorted(END_TS, side="right")) - 1
    start_rank_pos_int = int(calendar_idx.searchsorted(START_RANK_TS))
    rank_list, row_list, split_list, bars_dict, count_list = [], [], [], {}, []
    for record_dict in symbol_record_list:
        symbol_str = record_dict["symbol"]
        raw_df = load_symbol_bars_df(symbol_str)
        if raw_df is None or len(raw_df) == 0:
            count_list.append({"symbol": symbol_str, "raw": 0})
            continue
        bars_df, count_dict = features.clean_bars_df(raw_df, calendar_idx)
        count_dict["symbol"] = symbol_str
        count_list.append(count_dict)
        if len(bars_df) < 2:
            continue
        feat_df = features.symbol_feature_df(bars_df, calendar_idx, record_dict["first_quoted"], record_dict["new_listing"])
        cal_pos_arr = feat_df["cal_pos"].to_numpy()
        sym_id_int = record_dict["sym_id"]

        rank_mask_arr = cal_pos_arr >= start_rank_pos_int
        rank_list.append(pd.DataFrame({"cal_pos": cal_pos_arr[rank_mask_arr], "sym_id": np.int32(sym_id_int),
                                       "adv": feat_df["adv"].to_numpy()[rank_mask_arr].astype(np.float32)}))

        split_mask_arr = feat_df["is_split"].to_numpy()
        if split_mask_arr.any():
            split_list.append(pd.DataFrame({"sym_id": np.int32(sym_id_int), "symbol": symbol_str,
                                            "date": bars_df.index[split_mask_arr],
                                            "ratio": feat_df["split_ratio"].to_numpy()[split_mask_arr],
                                            "uclose": feat_df["uclose"].to_numpy()[split_mask_arr]}))

        in_range_arr = (cal_pos_arr >= start_rows_pos_int) & (cal_pos_arr <= end_pos_int)
        keep_arr = in_range_arr & (feat_df["ath"].to_numpy() | feat_df["in_ipo_window"].to_numpy() | feat_df["in_split_window"].to_numpy())
        if not keep_arr.any():
            continue
        row_pos_arr = np.nonzero(keep_arr)[0]
        kept_df = feat_df.iloc[row_pos_arr].reset_index(drop=True)
        fwd_df = features.forward_return_df(bars_df, row_pos_arr, cal_pos_arr)
        kept_df = pd.concat([kept_df, fwd_df], axis=1)
        kept_df.insert(0, "sym_id", np.int32(sym_id_int))
        row_list.append(kept_df)

        tradable_arr = kept_df["ath"].to_numpy() & (kept_df["in_ipo_window"].to_numpy() | kept_df["in_split_window"].to_numpy())
        if tradable_arr.any():
            bar_df = bars_df[BAR_FIELD_LIST].copy()
            bar_df.insert(0, "cal_pos", cal_pos_arr)
            bars_dict[symbol_str] = bar_df.reset_index(drop=True)
    out_dir_path = common.CACHE_DIR_PATH / "bars"
    out_dir_path.mkdir(parents=True, exist_ok=True)
    pd.to_pickle(bars_dict, out_dir_path / f"chunk_{chunk_int:03d}.pkl")
    return {
        "rank": pd.concat(rank_list, ignore_index=True) if rank_list else None,
        "rows": pd.concat(row_list, ignore_index=True) if row_list else None,
        "splits": pd.concat(split_list, ignore_index=True) if split_list else None,
        "counts": count_list,
    }


def select_symbols(meta_df: pd.DataFrame) -> pd.DataFrame:
    allowed_list = json.loads(ALLOWED_SUBTYPE2_PATH.read_text())
    sel_df = meta_df[(meta_df["subtype1"] == "Equity") & meta_df["subtype2"].isin(allowed_list)].copy()
    # *** CRITICAL*** Norgate returns no last quoted date for ACTIVE symbols (NaT). Dropping NaT rows would keep only
    # delisted names - an inverted survivorship bias. Active symbols are kept; delisted ones only if they traded
    # after the rank warm-up start.
    sel_df = sel_df[sel_df["last_quoted"].isna() | (sel_df["last_quoted"] >= START_RANK_TS)]
    sel_df = sel_df.drop_duplicates("symbol").reset_index(drop=True)
    database_count_dict = sel_df["database"].value_counts().to_dict()
    if database_count_dict.get("US Equities", 0) < 1000 or database_count_dict.get("US Equities Delisted", 0) < 1000:
        raise RuntimeError(f"Universe must contain active and delisted symbols; got {database_count_dict}.")
    sel_df["data_start"] = sel_df["first_quoted"] < features.NEW_LISTING_MIN_DATE_TS
    sel_df["new_listing"] = (~sel_df["data_start"]) & (sel_df["major_listed_first"] == 1) & (sel_df["blank_check_first"] != 1)
    sel_df["new_listing_incl_spac"] = (~sel_df["data_start"]) & (sel_df["major_listed_first"] == 1)
    sel_df["sym_id"] = np.arange(len(sel_df), dtype=np.int32)
    return sel_df


def main(worker_int: int, new_listing_col_str: str = "new_listing_incl_spac", tag_str: str = "") -> None:
    started_float = time.time()
    meta_df = pd.read_parquet(common.CACHE_DIR_PATH / "symbol_meta.parquet")
    sym_df = select_symbols(meta_df)
    calendar_idx = load_calendar_idx()
    common.CACHE_DIR_PATH.mkdir(parents=True, exist_ok=True)
    np.save(common.CACHE_DIR_PATH / "calendar.npy", calendar_idx.values.astype("datetime64[ns]"))
    sym_df.to_parquet(common.CACHE_DIR_PATH / "symbols.parquet")
    common.log_progress(f"data_build: {len(sym_df)} symbols, {worker_int} workers, new-listing column {new_listing_col_str}")

    record_list = [{"symbol": r.symbol, "sym_id": int(r.sym_id), "first_quoted": r.first_quoted,
                    "new_listing": bool(getattr(r, new_listing_col_str))} for r in sym_df.itertuples()]
    chunk_size_int = 400
    task_list = [(i // chunk_size_int, record_list[i:i + chunk_size_int], calendar_idx.values) for i in range(0, len(record_list), chunk_size_int)]
    rank_list, row_list, split_list, count_list = [], [], [], []
    with ProcessPoolExecutor(max_workers=worker_int) as pool:
        for done_int, result_dict in enumerate(pool.map(process_chunk, task_list)):
            for key_str, target_list in (("rank", rank_list), ("rows", row_list), ("splits", split_list)):
                if result_dict[key_str] is not None:
                    target_list.append(result_dict[key_str])
            count_list += result_dict["counts"]
            if done_int % 10 == 0:
                common.log_progress(f"data_build: chunk {done_int + 1}/{len(task_list)}")
    pd.concat(rank_list, ignore_index=True).to_parquet(common.CACHE_DIR_PATH / f"rank_panel{tag_str}.parquet")
    pd.concat(row_list, ignore_index=True).to_parquet(common.CACHE_DIR_PATH / f"rows{tag_str}.parquet")
    pd.concat(split_list, ignore_index=True).to_parquet(common.CACHE_DIR_PATH / "splits.parquet")
    pd.DataFrame(count_list).to_parquet(common.CACHE_DIR_PATH / "bar_counts.parquet")

    import norgatedata as nd

    spy_df = nd.price_timeseries("SPY", stock_price_adjustment_setting=nd.StockPriceAdjustmentType.TOTALRETURN,
                                 padding_setting=nd.PaddingType.NONE, start_date="1990-01-01", timeseriesformat="pandas-dataframe")
    spy_df[["Open", "Close"]].reindex(calendar_idx).to_parquet(common.CACHE_DIR_PATH / "spy_tr.parquet")
    common.log_progress(f"data_build: done in {time.time() - started_float:.0f}s")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 6)
