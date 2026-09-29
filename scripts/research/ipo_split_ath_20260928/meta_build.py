"""Symbol metadata for the IPO / split all-time-high study (research only, data plumbing, no returns).

For every symbol in Norgate's 'US Equities' and 'US Equities Delisted' databases: asset id, security type
(subtype1 / subtype2), exchange, first and last quoted date, and whether the symbol was a blank-check company (SPAC)
or not major-exchange listed on its first quoted date.

Usage: python meta_build.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

HERE_PATH = Path(__file__).resolve().parent
if str(HERE_PATH) not in sys.path:
    sys.path.insert(0, str(HERE_PATH))

import common  # noqa: E402


def _first_flag(frame_df: pd.DataFrame) -> float:
    if frame_df is None or len(frame_df) == 0:
        return float("nan")
    return float(frame_df.iloc[0, 0])


def main() -> None:
    import norgatedata as nd

    started_float = time.time()
    row_list = []
    for database_str in ["US Equities", "US Equities Delisted"]:
        symbol_list = nd.database_symbols(database_str)
        common.log_progress(f"meta: {database_str} {len(symbol_list)} symbols")
        for count_int, symbol_str in enumerate(symbol_list):
            row_dict = {
                "symbol": symbol_str,
                "database": database_str,
                "assetid": nd.assetid(symbol_str),
                "subtype1": nd.subtype1(symbol_str),
                "subtype2": nd.subtype2(symbol_str),
                "exchange": nd.exchange_name(symbol_str),
                "name": nd.security_name(symbol_str),
                "first_quoted": pd.Timestamp(nd.first_quoted_date(symbol_str)) if nd.first_quoted_date(symbol_str) else pd.NaT,
                "last_quoted": pd.Timestamp(nd.last_quoted_date(symbol_str)) if nd.last_quoted_date(symbol_str) else pd.NaT,
            }
            is_equity_bool = row_dict["subtype1"] == "Equity"
            if is_equity_bool and pd.notna(row_dict["first_quoted"]):
                first_str = row_dict["first_quoted"].strftime("%Y-%m-%d")
                blank_df = nd.blank_check_company_timeseries(symbol_str, start_date=first_str, end_date=first_str,
                                                             timeseriesformat="pandas-dataframe")
                major_df = nd.major_exchange_listed_timeseries(symbol_str, start_date=first_str, end_date=first_str,
                                                               timeseriesformat="pandas-dataframe")
                row_dict["blank_check_first"] = _first_flag(blank_df)
                row_dict["major_listed_first"] = _first_flag(major_df)
            row_list.append(row_dict)
            if count_int % 5000 == 0:
                common.log_progress(f"meta: {database_str} {count_int}/{len(symbol_list)}")
    meta_df = pd.DataFrame(row_list)
    common.CACHE_DIR_PATH.mkdir(parents=True, exist_ok=True)
    meta_df.to_parquet(common.CACHE_DIR_PATH / "symbol_meta.parquet")
    common.log_progress(f"meta: done {len(meta_df)} rows in {time.time() - started_float:.0f}s")


if __name__ == "__main__":
    main()
