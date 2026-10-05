"""Timing lens 15: margin rows by independent arithmetic (equity = L x pods - (L-1) x debt, annual reset), L solves, spreads, daily-constant."""
import json, pickle
from pathlib import Path
import numpy as np, pandas as pd
WT = Path(__file__).resolve().parents[5]
OUT = WT / "results/research/portfolio/fund_products_20261005/audit/review/timing_lens"
c = pickle.load(open(OUT / "cache.pkl", "rb")); F, d = c["frames"], c["data"]
study = json.loads((WT / "results/research/portfolio/fund_products_20261005/report/study.json").read_text())
LONG, END = pd.Timestamp("2008-03-04"), pd.Timestamp("2026-08-19")
main = F["main"][0]; rf = main["tbill"]; idx = main.loc[LONG:END].index
days = pd.Series(d["index"], index=d["index"]).diff().dt.days
def debt_r(spread): return ((d["dtb3"] + spread) * days / 360).loc[LONG:END].to_numpy()
def lev(w, L, spread=0.015):
    x = main.loc[LONG:END, list(w)].to_numpy(); tgt = np.array([w[k] for k in w]); dr = debt_r(spread)
    E = 1.0; out = []; yr = idx[0].year; pods = L * E * tgt; debt = (L - 1) * E; peak = 0
    for i, t in enumerate(idx):
        if t.year != yr:
            pods = L * E * tgt; debt = (L - 1) * E; yr = t.year
        peak = max(peak, pods.sum() / E)
        pods = pods * (1 + x[i]); debt = debt * (1 + dr[i]); E1 = pods.sum() - debt; out.append(E1 / E - 1); E = E1
    return pd.Series(out, index=idx), peak
def st(r):
    x = r.to_numpy(); nav = np.r_[1, np.cumprod(1 + x)]; f = rf.reindex(r.index).to_numpy()
    return dict(cagr=nav[-1] ** (252 / len(x)) - 1, vol=x.std(ddof=1) * 252 ** .5, dd=(nav / np.maximum.accumulate(nav) - 1).min(), xs=(x - f).mean() / (x - f).std(ddof=1) * 252 ** .5)
P = {"GR1": {"taa3x": 1 / 3, "ndx_atr_cap": 1 / 6, "ndx_natr_cap": 1 / 6, "dv2_g": 1 / 6, "hpi_g": 1 / 6},
     "GR2": {"taa3x_1n": 1 / 3, "ndx_atr_cap": 1 / 6, "ndx_natr_cap": 1 / 6, "dv2_g": 1 / 6, "hpi_g": 1 / 6},
     "GR3": {"taa3x_1n": 1 / 2, "ndx_atr_cap": 1 / 8, "ndx_natr_cap": 1 / 8, "dv2_g": 1 / 8, "hpi_g": 1 / 8}}
base = {k: st(lev(w, 1.0)[0]) for k, w in P.items()}
for b, t in (("GR1", "GR2"), ("GR1", "GR3"), ("GR2", "GR3")):
    Lv = round(base[t]["vol"] / base[b]["vol"], 2)
    Lc = next(round(float(L), 2) for L in np.arange(1.0, 3.001, 0.01) if st(lev(P[b], float(L))[0])["cagr"] >= base[t]["cagr"])
    m = study["margin"][f"{b} -> {t}"]
    r, pk = lev(P[b], Lv); s = st(r)
    print(f"{b}->{t}: L_vol {Lv} (study {m['vol_matched']['L']}), L_cagr {Lc} (study {m['cagr_matched']['L']}); vol-matched cagr {s['cagr']:.4f} xs {s['xs']:.3f} dd {s['dd']:.4f} vol {s['vol']:.4f} peak lev {pk:.3f}"
          f" | study {m['vol_matched']['q']['cagr']:.4f} {m['vol_matched']['q']['xs']:.3f} {m['vol_matched']['q']['dd']:.4f} {m['vol_matched']['q']['vol']:.4f} {m['vol_matched']['peak_leverage']:.3f}"
          f" | spreads 0.5% {st(lev(P[b], Lv, 0.005)[0])['cagr']:.4f} (study {m['vol_matched']['spread_050']['cagr']:.4f}) 2.5% {st(lev(P[b], Lv, 0.025)[0])['cagr']:.4f} (study {m['vol_matched']['spread_250']['cagr']:.4f})"
          f" | target vol {base[t]['vol']:.4f} cagr {base[t]['cagr']:.4f} dd {base[t]['dd']:.4f}")
