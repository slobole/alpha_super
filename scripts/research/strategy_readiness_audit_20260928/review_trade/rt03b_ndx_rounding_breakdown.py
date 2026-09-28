"""Review R3b: NDX VXN whole-share lost return by asset and year (is it one name?)."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
from data.norgate_loader import load_price_timeseries  # noqa: E402
TRD = REPO / "results/research/strategy_readiness_audit_20260928/tradability"
OUT = REPO / "results/research/strategy_readiness_audit_20260928/review_trade"
END = "2026-09-25"
ws = pd.read_csv(TRD / "ndx_vxn/whole_share_since_2016.csv", parse_dates=["decision_date"])
ws = ws[ws["capital_usd"] == 30000.0][["decision_date", "asset", "target_weight", "nominal_price"]].copy()
dates = sorted(ws["decision_date"].unique()); nxt = {d: (dates[i+1] if i+1 < len(dates) else pd.Timestamp(END)) for i, d in enumerate(dates)}
cache = {}
def px(a, d):
    if a not in cache: cache[a] = load_price_timeseries(a, start_date_str="2015-06-01", end_date_str=END)["Close"].dropna()
    s = cache[a].loc[:d]; return float(s.iloc[-1])
ws["r"] = [px(a, nxt[d]) / px(a, d) - 1 for a, d in zip(ws["asset"], ws["decision_date"])]
rows = []
for C in (12_000.0, 18_000.0, 30_000.0):
    sh = np.floor(ws["target_weight"] * C / ws["nominal_price"]); err = ws["target_weight"] - sh * ws["nominal_price"] / C
    contrib = err * ws["r"]
    by_asset = contrib.groupby(ws["asset"]).sum().sort_values(ascending=False)
    by_year = contrib.groupby(ws["decision_date"].dt.year).sum()
    print(f"C={C:.0f} total lost {100*contrib.sum():.2f}% (sum of period returns); top assets:", (100*by_asset.head(6)).round(2).to_dict())
    print("  by year:", (100*by_year).round(2).to_dict())
    print("  ex-SNDK lost pp/yr:", round(100*contrib[ws['asset']!='SNDK'].sum()/10.49, 3))
    print("  mean per-name budget USD:", round(float((ws['target_weight']*C).mean()),0), " share of name-decisions priced > budget/2:", round(float((ws['nominal_price'] > ws['target_weight']*C/2).mean()),3))
    rows.append({"capital": C, "lost_pp_per_yr": 100*contrib.sum()/10.49, "lost_pp_per_yr_ex_SNDK": 100*contrib[ws['asset']!='SNDK'].sum()/10.49})
pd.DataFrame(rows).to_csv(OUT / "r3b_ndx_rounding_breakdown.csv", index=False)
last = ws[ws["decision_date"] == dates[-1]]
print("latest decision", dates[-1].date(), last[["asset","target_weight","nominal_price"]].to_string(index=False))
