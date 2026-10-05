"""MR pods: cash vs BIL share through time (from raw path + transactions)."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import ir_core as c
for a in ["dv2_g", "hpi_g"]:
    p = c.read_path(c.NEW_SRC / f"{a}__path.csv.gz")
    t = c.read_tx(c.NEW_SRC / f"{a}__transactions.csv.gz")
    q = p.loc[c.LONG_START:c.END]
    cw = q["cash_float"] / q["total_value_float"]
    print(a, "LONG days", len(q), "cash>0.9:", int((cw > 0.9).sum()), "cash>0.5:", int((cw > 0.5).sum()), "cash>0.2:", int((cw > 0.2).sum()), "cash<0:", int((cw < 0).sum()), "cash< -0.02:", int((cw < -0.02).sum()))
    hi = cw[cw > 0.9]
    print("   first/last dates cash>0.9:", list(hi.index[:8].strftime("%Y-%m-%d")), "...", list(hi.index[-5:].strftime("%Y-%m-%d")))
    by = (cw > 0.9).groupby(cw.index.year).sum()
    print("   cash>0.9 days by year:", by[by > 0].to_dict())
    bil = t[t["asset_str"] == "BIL"].copy()
    print("   BIL fills:", len(bil), "of", len(t), "first", bil["date"].min(), "last", bil["date"].max(), "BIL gross notional share of all:", round(bil["signed_notional_float"].abs().sum() / t["signed_notional_float"].abs().sum(), 4))
    tl = t[(t["date"] >= c.LONG_START) & (t["date"] <= c.END)]
    navm = q["total_value_float"].mean()
    yrs = len(q) / 252
    g = tl.assign(a=tl["signed_notional_float"].abs())
    # turnover relative to prior-day NAV
    daily = g.groupby("date")["a"].sum().reindex(q.index).fillna(0) / p["total_value_float"].shift(1).reindex(q.index)
    bil_daily = g[g["asset_str"] == "BIL"].groupby("date")["a"].sum().reindex(q.index).fillna(0) / p["total_value_float"].shift(1).reindex(q.index)
    print(f"   LONG turnover: {daily.sum()/yrs:.1f}x NAV/yr all fills, of which BIL {bil_daily.sum()/yrs:.1f}x; trading days/yr {int((daily>0).sum()/yrs)}")
    # BIL position: cumulative shares
    pos = bil.groupby("date")["amount_float"].sum().cumsum()
    print("   BIL share count min", pos.min(), "end", pos.iloc[-1])
