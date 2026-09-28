"""
Audit probe (read-only, local Norgate): bound for A-LIVE-06 (NetLiq marked before the open
vs auction-priced targets). |Open_t / Close_{t-1} - 1| for TAA assets, all sessions and
first session of each month (the monthly MOO execution day).

Output: results/research/strategy_readiness_audit_20260928/live_path/overnight_gap_bounds.csv
"""
from __future__ import annotations

from pathlib import Path

import norgatedata
import pandas as pd

OUTPUT_PATH_OBJ = Path("results/research/strategy_readiness_audit_20260928/live_path/overnight_gap_bounds.csv")


def main() -> None:
    row_list = []
    for symbol_str in ["TQQQ", "BTAL", "GLD", "UUP", "TLT", "DBC", "QQQ", "SPY"]:
        price_df = norgatedata.price_timeseries(
            symbol_str,
            stock_price_adjustment_setting=norgatedata.StockPriceAdjustmentType.CAPITALSPECIAL,
            timeseriesformat="pandas-dataframe",
        )
        gap_ser = (price_df["Open"] / price_df["Close"].shift(1) - 1.0).dropna().abs()
        first_session_ser = gap_ser.groupby(gap_ser.index.to_period("M")).head(1)
        row_list.append({
            "symbol_str": symbol_str,
            "all_p50": gap_ser.median(), "all_p99": gap_ser.quantile(0.99), "all_max": gap_ser.max(),
            "first_session_p50": first_session_ser.median(), "first_session_p99": first_session_ser.quantile(0.99),
            "first_session_max": first_session_ser.max(),
        })
    out_df = pd.DataFrame(row_list)
    OUTPUT_PATH_OBJ.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUTPUT_PATH_OBJ, index=False)
    print(out_df.round(4).to_string())


if __name__ == "__main__":
    main()
