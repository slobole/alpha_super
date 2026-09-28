"""
Audit probe (read-only, local Norgate): how often would ALLMARKETDAYS padding in the
non-CORE5 live snapshot profiles carry a stale value into a month-end decision row?

The TAA (norgate_eod_etf_plus_vix_helper) and NDX VXN (norgate_eod_ndx_pit_plus_vxn_helper)
profiles are exported with padding ALLMARKETDAYS and, unlike CORE5, with no per-symbol
unpadded-endpoint check (scripts/export_norgate_snapshot.py:309-328 is CORE5-only).
Here we count XNYS sessions (from the $SPX calendar) on which each input symbol has NO
unpadded Norgate row, overall and on month-end sessions, since 2008.

Output: results/research/strategy_readiness_audit_20260928/live_path/helper_padding_gaps.csv
"""
from __future__ import annotations

from pathlib import Path

import norgatedata
import pandas as pd

OUTPUT_DIR_PATH_OBJ = Path("results/research/strategy_readiness_audit_20260928/live_path")
SYMBOL_LIST = ["$VIX", "$VXN", "SPY", "QQQ", "TQQQ", "BTAL", "GLD", "UUP", "TLT", "DBC"]
START_DATE_STR = "2008-01-01"


def main() -> None:
    OUTPUT_DIR_PATH_OBJ.mkdir(parents=True, exist_ok=True)
    spx_index = pd.DatetimeIndex(
        norgatedata.price_timeseries("$SPX", timeseriesformat="pandas-dataframe").index
    )
    spx_index = spx_index[spx_index >= pd.Timestamp(START_DATE_STR)]
    month_end_index = pd.DatetimeIndex(
        pd.Series(spx_index, index=spx_index.to_period("M")).groupby(level=0).max().to_numpy()
    )
    row_list: list[dict[str, object]] = []
    for symbol_str in SYMBOL_LIST:
        raw_df = norgatedata.price_timeseries(
            symbol_str,
            stock_price_adjustment_setting=norgatedata.StockPriceAdjustmentType.CAPITALSPECIAL,
            padding_setting=norgatedata.PaddingType.NONE,
            timeseriesformat="pandas-dataframe",
        )
        symbol_index = pd.DatetimeIndex(raw_df.index)
        first_ts = max(symbol_index[0], spx_index[0])
        in_range_session_index = spx_index[spx_index >= first_ts]
        missing_index = in_range_session_index.difference(symbol_index)
        missing_month_end_index = missing_index.intersection(month_end_index)
        row_list.append(
            {
                "symbol_str": symbol_str,
                "first_date_str": first_ts.date().isoformat(),
                "last_date_str": symbol_index[-1].date().isoformat(),
                "session_count_int": len(in_range_session_index),
                "missing_session_count_int": len(missing_index),
                "missing_month_end_count_int": len(missing_month_end_index),
                "missing_month_end_list_str": ";".join(d.date().isoformat() for d in missing_month_end_index),
                "missing_session_sample_str": ";".join(d.date().isoformat() for d in missing_index[:15]),
            }
        )
    out_df = pd.DataFrame(row_list)
    out_df.to_csv(OUTPUT_DIR_PATH_OBJ / "helper_padding_gaps.csv", index=False)
    print(out_df.drop(columns=["missing_session_sample_str"]).to_string())
    print(out_df[["symbol_str", "missing_session_sample_str"]].to_string())


if __name__ == "__main__":
    main()
