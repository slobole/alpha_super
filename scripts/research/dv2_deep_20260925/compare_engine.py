"""Compare real-engine finalist runs with the replica sources (fills and NAV)."""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
OUT = Path(__file__).resolve().parents[3] / "results/research/dv2_deep_20260925"
MAP = {"F1": "F1_floor_adv", "F2": "F2_floor_adv_s15", "F3": "F3_E_vote", "F4": "F4_w252", "ETF": "etf_industries"}
res = {}
for w, alias in MAP.items():
    e = OUT / "engine" / f"{w}__path.csv"
    if not e.exists():
        continue
    en = pd.read_csv(e, index_col=0, parse_dates=True)["total_value"]
    et = pd.read_csv(OUT / "engine" / f"{w}__transactions.csv", parse_dates=["bar"])
    rn = pd.read_csv(OUT / "sources" / f"{alias}__path.csv.gz", index_col="date", parse_dates=True)["total_value_float"]
    rt = pd.read_csv(OUT / "sources" / f"{alias}__transactions.csv.gz", parse_dates=["date"])
    key = lambda d, a, q: set(zip(pd.to_datetime(d).dt.strftime("%Y-%m-%d"), a, np.round(q.astype(float), 6)))
    A, B = key(rt.date, rt.asset_str, rt.amount_float), key(et.bar, et.asset, et.amount)
    both = pd.concat([en.pct_change(), rn.pct_change()], axis=1).dropna()
    d = (both.iloc[:, 0] - both.iloc[:, 1]).abs()
    cagr = lambda s: (s.iloc[-1] / s.iloc[0]) ** (365.25 / (s.index[-1] - s.index[0]).days) - 1
    res[w] = {"fills_engine": len(B), "fills_replica": len(A), "only_engine": len(B - A), "only_replica": len(A - B),
              "max_abs_daily_diff": float(d.max()), "days_diff_gt_1e-8": int((d > 1e-8).sum()),
              "cagr_engine": float(cagr(en)), "cagr_replica": float(cagr(rn.reindex(en.index)))}
print(json.dumps(res, indent=1))
(OUT / "phase5_engine_vs_replica.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
