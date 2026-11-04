"""Is Norgate's Turnover field stable between reads of the same symbol? Reads every Russell 1000 name a few times in one process and reports the cells that change. Direct Norgate; no write to any data store.

Usage: turnover_flake_probe.py [read count] [output parquet]
"""
from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd


def read_all_df(symbol_list: list[str], norgatedata_module) -> pd.DataFrame:
    frame_dict = {}
    for symbol_str in symbol_list:
        price_df = norgatedata_module.price_timeseries(
            symbol_str,
            stock_price_adjustment_setting=norgatedata_module.StockPriceAdjustmentType.CAPITALSPECIAL,
            padding_setting=norgatedata_module.PaddingType.ALLMARKETDAYS,
            start_date="1998-01-01",
            timeseriesformat="pandas-dataframe",
        )
        if price_df is not None and len(price_df.index) and "Turnover" in price_df.columns:
            frame_dict[symbol_str] = price_df[["Close", "Volume", "Turnover", "Unadjusted Close"]]
    return pd.concat(frame_dict, axis=1)


def main(read_count_int: int, output_str: str | None) -> None:
    import norgatedata as norgatedata_module

    symbol_list = sorted(norgatedata_module.watchlist_symbols("Russell 1000 Current & Past"))
    print(len(symbol_list), "symbols", flush=True)
    frame_list = []
    for read_int in range(read_count_int):
        t0_float = time.time()
        frame_list.append(read_all_df(symbol_list, norgatedata_module))
        print("read", read_int, frame_list[-1].shape, f"{time.time() - t0_float:.0f}s", flush=True)
    base_df = frame_list[0]
    row_list = []
    for read_int, other_df in enumerate(frame_list[1:], start=1):
        other_df = other_df.reindex(index=base_df.index, columns=base_df.columns)
        for field_str in ("Close", "Volume", "Turnover", "Unadjusted Close"):
            base_mat = base_df.xs(field_str, axis=1, level=1).to_numpy(dtype=float)
            other_mat = other_df.xs(field_str, axis=1, level=1).to_numpy(dtype=float)
            differ_mat = ~((base_mat == other_mat) | (np.isnan(base_mat) & np.isnan(other_mat)))
            print(f"read {read_int} {field_str}: {int(differ_mat.sum())} cells differ of {int(np.isfinite(base_mat).sum())}", flush=True)
            for row_int, column_int in zip(*np.nonzero(differ_mat)):
                row_list.append({
                    "read_int": read_int, "field_str": field_str, "date": base_df.index[row_int],
                    "symbol_str": base_df.xs(field_str, axis=1, level=1).columns[column_int],
                    "first_float": base_mat[row_int, column_int], "other_float": other_mat[row_int, column_int],
                })
    differ_df = pd.DataFrame(row_list)
    if len(differ_df):
        differ_df["ratio_float"] = differ_df["other_float"] / differ_df["first_float"]
        print(differ_df.groupby("field_str").agg(cells=("ratio_float", "size"), symbols=("symbol_str", "nunique"), median_ratio=("ratio_float", "median"), low=("ratio_float", "min"), high=("ratio_float", "max")).to_string(), flush=True)
        print(differ_df.head(30).to_string(), flush=True)
        if output_str:
            differ_df.to_parquet(output_str)
    print("FLAKE PROBE DONE", flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 3, sys.argv[2] if len(sys.argv) > 2 else None)
