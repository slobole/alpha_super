"""Timing lens 10: paired bootstrap shares by independent vectorised code (seeds 0-9 pooled), GR1 vs S9 / T1 / T2; block tests."""
import json, pickle, sys
from pathlib import Path
import numpy as np, pandas as pd
WT = Path(__file__).resolve().parents[5]
OUT = WT / "results/research/portfolio/fund_products_20261005/audit/review/timing_lens"
c = pickle.load(open(OUT / "cache.pkl", "rb")); F = c["frames"]
study = json.loads((WT / "results/research/portfolio/fund_products_20261005/report/study.json").read_text())
LONG, END = pd.Timestamp("2008-03-04"), pd.Timestamp("2026-08-19")
def book(fr, w):
    x = fr.loc[LONG:END, list(w)]; tgt = np.array([w[k] for k in w]); pods = tgt.copy(); out = []; yr = x.index[0].year
    for t, row in zip(x.index, x.to_numpy()):
        if t.year != yr: pods = tgt * pods.sum(); yr = t.year
        v0 = pods.sum(); pods = pods * (1 + row); out.append(pods.sum() / v0 - 1)
    return pd.Series(out, index=x.index)
def idx_mat(n, reps, block, seed):
    rng = np.random.default_rng(seed); idx = np.empty((reps, n), dtype=np.int64); idx[:, 0] = rng.integers(0, n, size=reps)
    restart = rng.random((reps, n)) < 1 / block; fresh = rng.integers(0, n, size=(reps, n))
    for j in range(1, n): idx[:, j] = np.where(restart[:, j], fresh[:, j], (idx[:, j - 1] + 1) % n)
    return idx
def shares(a, b, rf, seeds=range(10)):
    A, B, Rf = a.to_numpy(), b.to_numpy(), rf.reindex(a.index).to_numpy(); n = len(A); gx, gc = [], []
    for s in seeds:
        idx = idx_mat(n, 2000, 63.0, 20260929 + s)
        xa, xb = A[idx] - Rf[idx], B[idx] - Rf[idx]
        gx.append(xa.mean(1) / xa.std(1, ddof=1) - xb.mean(1) / xb.std(1, ddof=1)); gc.append(np.log1p(A[idx]).sum(1) - np.log1p(B[idx]).sum(1))
    gx, gc = np.concatenate(gx) * 252 ** .5, np.concatenate(gc)
    return float((gx > 0).mean()), float((gc > 0).mean()), [float(np.percentile(gx, q)) for q in (5, 50, 95)]
main, p5 = F["main"][0], F["s3_plus_5bps"][0]; rf = main["tbill"]
MOM = {"ndx_atr_cap": 1 / 6, "ndx_natr_cap": 1 / 6}; MR = {"dv2_g": 1 / 6, "hpi_g": 1 / 6}
GR1 = {"taa3x": 1 / 3, **MOM, **MR}
S9 = {"taa3x_1n": 0.384, "ndx_vxn": 0.256, "core5": 0.18, "btal_qqq": 0.18}
T1 = {"taa3x": 1 / 3, "tbill": 1 / 3, **MR}; T2 = {"taa3x": 1 / 3, **MOM, "tbill": 1 / 3}
r1 = book(main, GR1)
rev = next(x for x in study["challenges_reverse"] if x["default"] == "S9 incumbent launch")
print("GR1 vs S9 (xs share, cagr share, gap p5/50/95):", shares(r1, book(main, S9), rf), "| study", rev["share_xs"], rev["share_cagr"], rev["gap_xs_p5_50_95"])
for nm, w in (("T1 MOM -> BIL", T1), ("T2 MR -> BIL (GR1-L)", T2)):
    s0 = shares(r1, book(main, w), rf)
    a5, b5 = book(p5, GR1), book(p5, w)
    s5 = shares(a5, b5, rf)
    a10, b10 = r1 + 2 * (a5 - r1), book(main, w) + 2 * (b5 - book(main, w))
    s10 = shares(a10, b10, rf)
    ref = study["slots"][nm]["frames"]
    print(nm, "main", s0[:2], "| +5", s5[:2], "| +10", s10[:2], "|| study", [(ref[k]["share_xs"], ref[k]["share_cagr"]) for k in ("main", "plus5", "plus10")])
