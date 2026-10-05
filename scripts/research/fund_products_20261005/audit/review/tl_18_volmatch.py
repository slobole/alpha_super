"""Timing lens 18: L that matches realised volatility of the levered book (annual reset, drifting leverage) vs the study's L = ratio of unlevered vols."""
import pickle
from pathlib import Path
import numpy as np, pandas as pd
WT = Path(__file__).resolve().parents[5]
c = pickle.load(open(WT / "results/research/portfolio/fund_products_20261005/audit/review/timing_lens/cache.pkl", "rb")); F, d = c["frames"], c["data"]
LONG, END = pd.Timestamp("2008-03-04"), pd.Timestamp("2026-08-19")
main = F["main"][0]; idx = main.loc[LONG:END].index
days = pd.Series(d["index"], index=d["index"]).diff().dt.days
dr = ((d["dtb3"] + 0.015) * days / 360).loc[LONG:END].to_numpy()
def lev(w, L):
    x = main.loc[LONG:END, list(w)].to_numpy(); tgt = np.array([w[k] for k in w]); E = 1.0; out = []; yr = idx[0].year; pods = L * tgt; debt = L - 1.0; levs = []
    for i, t in enumerate(idx):
        if t.year != yr: pods = L * E * tgt; debt = (L - 1) * E; yr = t.year
        levs.append(pods.sum() / E); pods = pods * (1 + x[i]); debt *= 1 + dr[i]; E1 = pods.sum() - debt; out.append(E1 / E - 1); E = E1
    r = np.array(out); nav = np.r_[1, np.cumprod(1 + r)]
    return dict(cagr=nav[-1] ** (252 / len(r)) - 1, vol=r.std(ddof=1) * 252 ** .5, dd=(nav / np.maximum.accumulate(nav) - 1).min(), mean_lev=float(np.mean(levs)))
P = {"GR1": {"taa3x": 1 / 3, "ndx_atr_cap": 1 / 6, "ndx_natr_cap": 1 / 6, "dv2_g": 1 / 6, "hpi_g": 1 / 6},
     "GR2": {"taa3x_1n": 1 / 3, "ndx_atr_cap": 1 / 6, "ndx_natr_cap": 1 / 6, "dv2_g": 1 / 6, "hpi_g": 1 / 6},
     "GR3": {"taa3x_1n": 1 / 2, "ndx_atr_cap": 1 / 8, "ndx_natr_cap": 1 / 8, "dv2_g": 1 / 8, "hpi_g": 1 / 8}}
base = {k: lev(w, 1.0) for k, w in P.items()}
for b, t, Ls in (("GR1", "GR2", 1.19), ("GR1", "GR3", 1.32), ("GR2", "GR3", 1.11)):
    Lm = next(L for L in np.arange(1.0, 2.0, 0.005) if lev(P[b], float(L))["vol"] >= base[t]["vol"])
    a, m = lev(P[b], Ls), lev(P[b], float(Lm))
    print(f"{b}->{t}: study L {Ls}: vol {a['vol']:.4f} (target {base[t]['vol']:.4f}), mean leverage {a['mean_lev']:.3f}, cagr {a['cagr']:.4f}, dd {a['dd']:.4f} | realised-vol-matched L {Lm:.3f}: vol {m['vol']:.4f} cagr {m['cagr']:.4f} dd {m['dd']:.4f} | target cagr {base[t]['cagr']:.4f} dd {base[t]['dd']:.4f}")
