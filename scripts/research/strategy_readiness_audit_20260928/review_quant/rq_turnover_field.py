"""Is Norgate's native Turnover a nominal (vintage-invariant) dollar volume?

HPI and QPI rank candidates by Turnover_T; every audit's split harness ASSUMES Turnover stays nominal after a
future corporate action. Here Turnover is compared with three reconstructions, year by year:
  nominal  = Unadjusted Close x raw Volume  (raw Volume = adjusted Volume / k, k = Unadj/Adj close)
  cs       = CAPITALSPECIAL Close x CAPITALSPECIAL Volume
  tr       = TOTALRETURN Close x TOTALRETURN Volume
If Turnover tracked `tr`, it would embed FUTURE ordinary dividends (look-ahead in the ranking).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT_PATH = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT_PATH))

from data.norgate_loader import load_price_timeseries  # noqa: E402

OUT_DIR_PATH = REPO_ROOT_PATH / "results/research/strategy_readiness_audit_20260928/review_quant"
SYMBOL_LIST = ["XOM", "T", "MO", "PFE", "AAPL", "MSFT", "NVDA", "COST", "KO", "JNJ"]


def main() -> None:
    row_list = []
    for symbol_str in SYMBOL_LIST:
        cs_df = load_price_timeseries(symbol_str, adjustment_str="CAPITALSPECIAL", start_date_str="2000-01-01", end_date_str="2026-09-25")
        tr_df = load_price_timeseries(symbol_str, adjustment_str="TOTALRETURN", start_date_str="2000-01-01", end_date_str="2026-09-25")
        cs_df = cs_df[cs_df["Volume"] > 0]
        k_ser = cs_df["Unadjusted Close"] / cs_df["Close"]
        nominal_ser = cs_df["Unadjusted Close"] * (cs_df["Volume"] / k_ser)
        cs_ser = cs_df["Close"] * cs_df["Volume"]
        tr_ser = (tr_df["Close"] * tr_df["Volume"]).reindex(cs_df.index)
        turnover_ser = cs_df["Turnover"].astype(float)
        for year_int, idx in cs_df.groupby(cs_df.index.year).groups.items():
            t = turnover_ser.loc[idx]
            row_list.append({
                "symbol": symbol_str, "year": int(year_int),
                "median_turnover_over_nominal": float(np.nanmedian(t / nominal_ser.loc[idx])),
                "median_turnover_over_cs": float(np.nanmedian(t / cs_ser.loc[idx])),
                "median_turnover_over_tr": float(np.nanmedian(t / tr_ser.loc[idx])),
                "median_tr_over_cs_close": float(np.nanmedian(tr_df["Close"].reindex(idx) / cs_df["Close"].loc[idx])),
            })
    df = pd.DataFrame(row_list)
    df.to_csv(OUT_DIR_PATH / "rq_turnover_field.csv", index=False)
    summary = df.groupby("symbol")[["median_turnover_over_nominal", "median_turnover_over_cs", "median_turnover_over_tr"]].agg(["min", "max"])
    print(summary.to_string())
    print(df[df.symbol.isin(["XOM", "MO"])].to_string())


if __name__ == "__main__":
    main()
