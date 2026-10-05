"""Timing lens 3: headline numbers from the source files with independent arithmetic; S7 weights; MR gate breakeven; slot shares."""
import json, pickle, sys
from pathlib import Path
import numpy as np, pandas as pd
WT = Path(__file__).resolve().parents[5]
OUT = WT / "results/research/portfolio/fund_products_20261005/audit/review/timing_lens"
c = pickle.load(open(OUT / "cache.pkl", "rb"))
d, F = c["data"], c["frames"]
study = json.loads((WT / "results/research/portfolio/fund_products_20261005/report/study.json").read_text())
LONG, EXACT, END = pd.Timestamp("2008-03-04"), pd.Timestamp("2012-10-02"), pd.Timestamp("2026-08-19")

def book(fr, w, start=LONG, end=END):
    """independent loop implementation: pods compound, reset at first session of each calendar year"""
    x = fr.loc[start:end, list(w)]
    assert not x.isna().any().any()
    tgt = np.array([w[k] for k in w]); pods = tgt.copy(); out = []; yr = x.index[0].year
    for t, row in zip(x.index, x.to_numpy()):
        if t.year != yr:
            pods = tgt * pods.sum(); yr = t.year
        v0 = pods.sum(); pods = pods * (1 + row); out.append(pods.sum() / v0 - 1)
    return pd.Series(out, index=x.index)

def st(r, rf):
    x = r.to_numpy(); f = rf.reindex(r.index).to_numpy(); nav = np.r_[1, np.cumprod(1 + x)]
    return dict(cagr=nav[-1] ** (252 / len(x)) - 1, xs=(x - f).mean() / (x - f).std(ddof=1) * 252 ** 0.5, dd=(nav / np.maximum.accumulate(nav) - 1).min(), vol=x.std(ddof=1) * 252 ** .5,
                cagr_cal=nav[-1] ** (365.25 / (r.index[-1] - d["index"][d["index"].get_loc(r.index[0]) - 1]).days) - 1)

def boot_p(r, limit=-0.20, reps=2000, block=63.0, seeds=range(10), seed0=20260929):
    x = r.to_numpy(); n = len(x); res = []
    for s in seeds:
        rng = np.random.default_rng(seed0 + s)
        idx = np.empty((reps, n), dtype=np.int64); idx[:, 0] = rng.integers(0, n, size=reps)
        restart = rng.random((reps, n)) < 1 / block; fresh = rng.integers(0, n, size=(reps, n))
        for j in range(1, n):
            idx[:, j] = np.where(restart[:, j], fresh[:, j], (idx[:, j - 1] + 1) % n)
        nav = np.cumprod(1 + x[idx], axis=1); nav = np.c_[np.ones(reps), nav]
        dd = (nav / np.maximum.accumulate(nav, axis=1) - 1).min(axis=1)
        res.append((dd < limit).mean())
    return float(np.mean(res)), float(np.max(res))

main, rf = F["main"][0], F["main"][0]["tbill"]
P = {"GR1": {"taa3x": 1 / 3, "ndx_atr_cap": 1 / 6, "ndx_natr_cap": 1 / 6, "dv2_g": 1 / 6, "hpi_g": 1 / 6},
     "GR2": {"taa3x_1n": 1 / 3, "ndx_atr_cap": 1 / 6, "ndx_natr_cap": 1 / 6, "dv2_g": 1 / 6, "hpi_g": 1 / 6},
     "GR3": {"taa3x_1n": 1 / 2, "ndx_atr_cap": 1 / 8, "ndx_natr_cap": 1 / 8, "dv2_g": 1 / 8, "hpi_g": 1 / 8},
     "S9": {"taa3x_1n": 0.384, "ndx_vxn": 0.256, "core5": 0.18, "btal_qqq": 0.18}}
for n, w in P.items():
    r = book(main, w); s = st(r, rf); p = boot_p(r)
    r5 = book(F["s3_plus_5bps"][0], w); rh = book(F["s1_house_cash"][0], w); rx = book(F["s6_exact"][0], w, EXACT)
    print(f"{n}: MAIN CAGR {s['cagr']:.4%} (calendar-time {s['cagr_cal']:.4%}) xs {s['xs']:.3f} dd {s['dd']:.4%} vol {s['vol']:.4%} P(DD<-20%) mean {p[0]:.4f} worst seed {p[1]:.4f}"
          f" | +5bps CAGR {st(r5, rf)['cagr']:.4%} xs {st(r5, rf)['xs']:.3f} dd {st(r5, rf)['dd']:.4%} | house CAGR {st(rh, rf)['cagr']:.4%} | EXACT CAGR {st(rx, rf)['cagr']:.4%} xs {st(rx, rf)['xs']:.3f} dd {st(rx, rf)['dd']:.4%}")
    key = n if n != "S9" else "S9 incumbent launch"
    q = study["books"][key]["q"]; fr = study["books"][key]["frames"]
    print(f"     study.json: CAGR {q['cagr']:.4%} xs {q['xs']:.3f} dd {q['dd']:.4%} p20 {study['books'][key]['tails']['p20']:.4f} | +5 {fr['s3_plus_5bps']['cagr']:.4%} {fr['s3_plus_5bps']['xs']:.3f} | house {fr['s1_house_cash']['cagr']:.4%} | exact {fr['s6_exact']['cagr']:.4%} {fr['s6_exact']['xs']:.3f}")
print("\nS7 weights:")
for row in study["s7_weights"]:
    print("  ", row)
print("\nMR gate breakeven:", json.dumps(study["mr_gate_breakeven"], indent=0))
sl = study["slots"]["T2 MR -> BIL (GR1-L)"]["frames"]
for fk in ("main", "plus5", "plus10"):
    print(fk, "GR1 xs", round(sl[fk]["gr1"]["xs"], 4), "cagr", round(sl[fk]["gr1"]["cagr"], 5), "| GR1-L xs", round(sl[fk]["slot"]["xs"], 4), "cagr", round(sl[fk]["slot"]["cagr"], 5), "share_xs", sl[fk]["share_xs"], "share_cagr", sl[fk]["share_cagr"])
for t, row in study["slots"].items():
    print(t, {fk: (round(v["share_xs"], 3), round(v["share_cagr"], 3), round(v["gr1"]["xs"], 3), round(v["slot"]["xs"], 3)) for fk, v in row["frames"].items()}, "adds_return_not_sharpe", row["adds_return_not_sharpe"],
          "blocks", {k: (round(v["share_xs"], 3)) for k, v in row["blocks"].items()})
print("\nmargin:")
for k, v in study["margin"].items():
    for kind in ("vol_matched", "cagr_matched"):
        m = v[kind]
        print(k, kind, "L", m["L"], "cagr", round(m["q"]["cagr"], 4), "xs", round(m["q"]["xs"], 3), "dd", round(m["q"]["dd"], 4), "vol", round(m["q"]["vol"], 4), "tgt", round(v["target"]["q"]["cagr"], 4), round(v["target"]["q"]["vol"], 4), round(v["target"]["q"]["dd"], 4),
              "daily_const cagr", round(m["daily_constant"]["cagr"], 4), "peak lev", round(m["peak_leverage"], 3), "regT", round(m["reg_t"], 3), "sp050", round(m["spread_050"]["cagr"], 4), "sp250", round(m["spread_250"]["cagr"], 4))
print("\nproducts:", json.dumps({k: {kk: v[kk] for kk in ("final", "cash_added", "fits_rung", "strictest_rung", "step_over_product_below", "offered")} for k, v in study["products"].items()}))
