"""Review R6: buy notional beyond pre-trade cash on each rebalance bar, % of prior NAV.
cash_prev = NAV_(t-1) - sum(shares_(t-1) * AdjClose_(t-1)); need = max(0, buys_t - max(cash_prev, 0)).
At MOO submission (09:23 ET) no same-auction sell has printed, so this is buying power beyond cash a cash account lacks.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
from data.norgate_loader import load_price_timeseries  # noqa: E402
LED = REPO / "results/research/strategy_readiness_audit_20260928/review_trade/ledgers"
OUT = REPO / "results/research/strategy_readiness_audit_20260928/review_trade"
out = {}
for key in ["taa3x", "taa1n", "btal_qqq", "ndx_vxn"]:
    nav = pd.read_csv(LED / f"{key}_nav.csv", index_col=0, parse_dates=True)["nav"]
    tx = pd.read_csv(LED / f"{key}_tx.csv", parse_dates=["bar"])
    sh = tx.pivot_table(index="bar", columns="asset", values="amount", aggfunc="sum").reindex(nav.index).fillna(0).cumsum()
    px = pd.DataFrame({a: load_price_timeseries(a, start_date_str="1998-01-01", end_date_str="2026-09-25")["Close"] for a in sh.columns}).reindex(nav.index).ffill()
    posval = (sh * px.fillna(0)).sum(axis=1)
    cash = nav - posval
    tx["notional"] = tx["amount"] * tx["price"]
    buys = tx[tx["notional"] > 0].groupby("bar")["notional"].sum()
    cash_prev = cash.shift(1).reindex(buys.index); nav_prev = nav.shift(1).reindex(buys.index)
    need = (buys - cash_prev.clip(lower=0)).clip(lower=0) / nav_prev
    need = need.dropna()
    l3 = need[need.index >= "2023-09-25"]
    out[key] = {"rebalance_days": int(len(need)), "share_days_needing_credit": round(float((need > 0.001).mean()), 3),
                "median_pct_nav": round(100 * float(need.median()), 1), "p95_pct_nav": round(100 * float(need.quantile(.95)), 1),
                "max_pct_nav": round(100 * float(need.max()), 1), "max_date": str(need.idxmax().date()),
                "last3y_median_pct_nav": round(100 * float(l3.median()), 1), "last3y_max_pct_nav": round(100 * float(l3.max()), 1),
                "eod_cash_min_pct_nav": round(100 * float((cash / nav).min()), 2),
                "eod_cash_days_negative": int((cash < -1).sum()), "eod_cash_mean_pct_nav": round(100 * float((cash / nav).mean()), 2)}
    print(key, out[key], flush=True)
(OUT / "r6_buying_power_beyond_cash.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
