"""Reviewer scratch (secondary lens): margin arithmetic, headline stats, exposure look-through, worst days.
Plain pandas / numpy on the dumped frames; no study book functions."""
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

WT = Path(r"C:\Users\User\Documents\workspace\alpha_super\.claude\worktrees\nervous-colden-784bbf")
OUT = WT / "results" / "research" / "portfolio" / "fund_products_20261005" / "audit" / "review" / "secondary"
D = pickle.load(open(OUT / "dump.pkl", "rb"))
END = pd.Timestamp("2026-08-19")
LONG, EXACT = pd.Timestamp("2008-03-04"), pd.Timestamp("2012-10-02")
main = D["frames"]["main"][0]
plus5 = D["frames"]["s3_plus_5bps"][0]
exact = D["frames"]["s6_exact"][0]
rf = main["tbill"]

MOM = {"ndx_atr_cap": 0.5, "ndx_natr_cap": 0.5}
MR = {"dv2_g": 0.5, "hpi_g": 0.5}


def three(taa, t, m, r):
    w = {taa: t}
    for k, v in MOM.items():
        w[k] = m * v
    for k, v in MR.items():
        w[k] = r * v
    return w


GR = {"GR1": three("taa3x", 1 / 3, 1 / 3, 1 / 3), "GR2": three("taa3x_1n", 1 / 3, 1 / 3, 1 / 3), "GR3": three("taa3x_1n", .5, .25, .25)}
S9 = {"taa3x_1n": 0.384, "ndx_vxn": 0.256, "core5": 0.18, "btal_qqq": 0.18}


def book(frame, w, start, end=END, extra=None, want_weights=False):
    """Pods compound independently; reset to target weights at the first session of each calendar year."""
    cols = list(w)
    R = frame.loc[start:end, [c for c in cols if not (extra and c in extra)]].copy()
    if extra is not None:
        for k, s in extra.items():
            R[k] = s.reindex(R.index)
    R = R[cols]
    assert not R.isna().any().any()
    tw = np.array([w[c] for c in cols])
    assert abs(tw.sum() - 1) < 1e-9
    yrs = R.index.year.to_numpy()
    vals = R.to_numpy()
    out = np.empty(len(R))
    wts = np.empty_like(vals)
    equity = 1.0
    pods = tw * equity
    for i in range(len(R)):
        if i == 0 or yrs[i] != yrs[i - 1]:
            pods = tw * equity
        wts[i] = pods / equity                      # prior-close weights
        pods = pods * (1.0 + vals[i])
        new = pods.sum()
        out[i] = new / equity - 1.0
        equity = new
    r = pd.Series(out, index=R.index)
    if want_weights:
        return r, pd.DataFrame(wts, index=R.index, columns=cols)
    return r


def stats(r):
    x = r.to_numpy()
    f = rf.reindex(r.index).to_numpy()
    nav = np.r_[1.0, np.cumprod(1 + x)]
    return {"cagr": nav[-1] ** (252 / len(x)) - 1, "vol": x.std(ddof=1) * np.sqrt(252), "xs": (x - f).mean() / (x - f).std(ddof=1) * np.sqrt(252),
            "dd": (nav / np.maximum.accumulate(nav) - 1).min(), "n": len(x)}


res = {}
print("=== headline (MAIN frame, my own book loop)")
for n, w in {**GR, "S9": S9}.items():
    s = stats(book(main, w, LONG))
    res[n] = s
    print(n, {k: round(v, 5) for k, v in s.items()})

# ---- margin: debt pod built from DTB3 myself
idx = main.index
days = pd.Series(idx, index=idx).diff().dt.days
dtb3 = D["dtb3"]
debt = {sp: ((dtb3 + sp) * days / 360.0).reindex(idx) for sp in (0.015, 0.005, 0.025)}
print("debt col vs mine max abs diff", float((main["debt"] - debt[0.015]).abs().max()), float((main["debt_250"] - debt[0.025]).abs().max()))
print("mean DTB3 over LONG", float(dtb3.loc[LONG:END].mean()))


def lev(w, L):
    out = {k: L * v for k, v in w.items()}
    out["DEBT"] = -(L - 1.0)
    return out


print("=== margin")
for base, L, tgt in (("GR1", 1.19, "GR2"), ("GR1", 1.17, "GR2"), ("GR1", 1.32, "GR3"), ("GR1", 1.31, "GR3"), ("GR2", 1.11, "GR3"), ("GR2", 1.12, "GR3")):
    wl = lev(GR[base], L)
    r, wt = book(main, wl, LONG, extra={"DEBT": debt[0.015]}, want_weights=True)
    s = stats(r)
    gross = 1.0 - wt["DEBT"]
    r5 = book(plus5, wl, LONG, extra={"DEBT": debt[0.015]})
    rex = book(exact, wl, EXACT, extra={"DEBT": debt[0.015]})
    r05 = book(main, wl, LONG, extra={"DEBT": debt[0.005]})
    r25 = book(main, wl, LONG, extra={"DEBT": debt[0.025]})
    regt = L * sum(v * (0.75 if k in ("taa3x", "taa3x_1n") else 0.5) for k, v in GR[base].items())
    # Reg-T with drifted weights at the worst day (TAA fully in TQQQ assumed)
    regt_drift = (wt.drop(columns="DEBT") * np.array([0.75 if k in ("taa3x", "taa3x_1n") else 0.5 for k in wt.columns if k != "DEBT"])).sum(axis=1)
    print(f"{base} x{L}: cagr {s['cagr']:.5f} vol {s['vol']:.5f} xs {s['xs']:.4f} dd {s['dd']:.5f} | +5 {stats(r5)['cagr']:.5f} | exact {stats(rex)['cagr']:.5f} "
          f"| sp0.5 {stats(r05)['cagr']:.5f} sp2.5 {stats(r25)['cagr']:.5f} | peak lev {gross.max():.4f} on {gross.idxmax().date()} mean {gross.mean():.4f} "
          f"| regT {regt:.4f} drift-max {regt_drift.max():.4f} on {regt_drift.idxmax().date()}")
    res[f"{base} x{L}"] = {**s, "plus5": stats(r5)["cagr"], "exact": stats(rex)["cagr"], "sp05": stats(r05)["cagr"], "sp25": stats(r25)["cagr"],
                           "peak_lev": float(gross.max()), "regt": regt, "regt_drift_max": float(regt_drift.max())}
for n in ("GR2", "GR3"):
    w = GR[n]
    print(n, "MAIN", {k: round(v, 5) for k, v in stats(book(main, w, LONG)).items()}, "+5", round(stats(book(plus5, w, LONG))["cagr"], 5),
          "exact", round(stats(book(exact, w, EXACT))["cagr"], 5))
print("GR1 +5", round(stats(book(plus5, GR["GR1"], LONG))["cagr"], 5), "exact", {k: round(v, 5) for k, v in stats(book(exact, GR["GR1"], EXACT)).items()})

# netting view: how much of the levered book is T-bill-like while it borrows
# ---- worst days
print("=== worst single days (MAIN frame)")
qqq = D["bench"]["QQQ"]
spx = D["bench"]["SPXTR"]
worst = {}
for n, w in {**GR, "S9": S9}.items():
    r, wt = book(main, w, LONG, want_weights=True)
    o = r.nsmallest(5)
    worst[n] = [(str(d.date()), float(v), float(qqq.loc[d]), float(spx.loc[d])) for d, v in o.items()]
    print(n, [(d, round(v, 4), round(q, 4), round(s_, 4)) for d, v, q, s_ in worst[n]])
    oe = r.loc[EXACT:].nsmallest(3)
    print("   EXACT-era:", [(str(d.date()), round(float(v), 4), round(float(qqq.loc[d]), 4)) for d, v in oe.items()])
res["worst_days"] = worst
print("QQQ worst 8 days in LONG:", [(str(d.date()), round(float(v), 4)) for d, v in qqq.loc[LONG:END].nsmallest(8).items()])
json.dump(res, open(OUT / "sec_01_margin.json", "w"), indent=1, default=float)
