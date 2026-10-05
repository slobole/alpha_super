"""Timing lens 13: bootstrap blocks / horizons / EXACT frame, variance ratio, rolling and reset numbers by independent code."""
import json, pickle, sys
from pathlib import Path
import numpy as np, pandas as pd
WT = Path(__file__).resolve().parents[5]
OUT = WT / "results/research/portfolio/fund_products_20261005/audit/review/timing_lens"
c = pickle.load(open(OUT / "cache.pkl", "rb")); F = c["frames"]
bat = json.loads((WT / "results/research/portfolio/fund_products_20261005/report/battery.json").read_text())
LONG, EXACT, END = pd.Timestamp("2008-03-04"), pd.Timestamp("2012-10-02"), pd.Timestamp("2026-08-19")
def book(fr, w, start=LONG, pid=lambda ix: ix.year.to_numpy()):
    x = fr.loc[start:END, list(w)]; tgt = np.array([w[k] for k in w]); pods = tgt.copy(); out = []; p = pid(x.index)
    for i, row in enumerate(x.to_numpy()):
        if i > 0 and p[i] != p[i - 1]: pods = tgt * pods.sum()
        v0 = pods.sum(); pods = pods * (1 + row); out.append(pods.sum() / v0 - 1)
    return pd.Series(out, index=x.index)
def idx_mat(n, reps, block, seed):
    rng = np.random.default_rng(seed); idx = np.empty((reps, n), dtype=np.int64); idx[:, 0] = rng.integers(0, n, size=reps)
    restart = rng.random((reps, n)) < 1 / block; fresh = rng.integers(0, n, size=(reps, n))
    for j in range(1, n): idx[:, j] = np.where(restart[:, j], fresh[:, j], (idx[:, j - 1] + 1) % n)
    return idx
def pdd(r, limit, block=63.0, horizon=None):
    x = r.to_numpy(); n = len(x); out = []
    for s in range(10):
        idx = idx_mat(n, 2000, block, 20260929 + s)
        if horizon: idx = idx[:, :horizon]
        nav = np.c_[np.ones(len(idx)), np.cumprod(1 + x[idx], axis=1)]
        out.append(((nav / np.maximum.accumulate(nav, axis=1) - 1).min(axis=1) < limit).mean())
    return float(np.mean(out))
main = F["main"][0]; rf = main["tbill"]
GR1 = {"taa3x": 1 / 3, "ndx_atr_cap": 1 / 6, "ndx_natr_cap": 1 / 6, "dv2_g": 1 / 6, "hpi_g": 1 / 6}
r = book(main, GR1)
B = bat["bootstrap"]
print("block 1:", pdd(r, -0.20, 1.0), "battery", B["blocks"]["1"]["GR1"]["p20"])
print("block 21:", pdd(r, -0.20, 21.0), "battery", B["blocks"]["21"]["GR1"]["p20"])
print("block 252:", pdd(r, -0.20, 252.0), "battery", B["blocks"]["252"]["GR1"]["p20"])
print("horizon 756 p15:", pdd(r, -0.15, 63.0, 756), "battery", B["horizons"]["756"]["GR1"]["p15"])
print("horizon 1260 p20:", pdd(r, -0.20, 63.0, 1260), "battery", B["horizons"]["1260"]["GR1"]["p20"])
rx = book(F["s6_exact"][0], GR1, EXACT)
print("EXACT p20:", pdd(rx, -0.20), "battery", B["frames"]["s6_exact"]["GR1"]["p20"])
lr = np.log1p(r.to_numpy())
for q in (21, 63):
    cs = np.cumsum(np.r_[0, lr]); agg = cs[q:] - cs[:-q]
    print(f"VR({q}) =", agg.var(ddof=1) / (q * lr.var(ddof=1)), "battery", B["variance_ratio"]["GR1"][str(q)])
print("lag-1 autocorrelation of daily returns:", float(pd.Series(lr).autocorr(1)), " lags 1-5 sum:", float(sum(pd.Series(lr).autocorr(k) for k in range(1, 6))))
# rolling 3y
x = r - rf.reindex(r.index); xs = (x.rolling(756).mean() / x.rolling(756).std(ddof=1) * 252 ** .5).dropna()
print("rolling 3y xs min / p10 / median:", float(xs.min()), float(xs.quantile(.1)), float(xs.median()), "battery", bat["rolling"]["GR1"]["3y"]["xs_min"], bat["rolling"]["GR1"]["3y"]["xs_p10"], bat["rolling"]["GR1"]["3y"]["xs_median"])
# reset: quarterly, annual-7, none
st = lambda s: (float(np.prod(1 + s.to_numpy()) ** (252 / len(s)) - 1), float((s - rf.reindex(s.index)).mean() / (s - rf.reindex(s.index)).std(ddof=1) * 252 ** .5))
for pol, pid in (("quarterly", lambda ix: (ix.year * 4 + (ix.month - 1) // 3).to_numpy()), ("annual-7", lambda ix: (ix.year + (ix.month >= 7)).to_numpy()), ("none", lambda ix: np.zeros(len(ix))), ("monthly", lambda ix: (ix.year * 12 + ix.month).to_numpy())):
    ref = bat["reset"]["GR1"]["policies"][pol]["free"]
    print(pol, st(book(main, GR1, pid=pid)), "battery", (ref["cagr"], ref["xs"]))
