"""+10 bps: pod-level exact drag vs the study's book-level linear extrapolation; per-pod cash add and drag sizes; proxy sanity."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import ir_core as c

main = pd.read_pickle(c.OUT / "main_frame.pkl"); plus5 = pd.read_pickle(c.OUT / "plus5_frame.pkl"); house = pd.read_pickle(c.OUT / "house_frame.pkl")
bil = main["BIL"]
study = json.loads((c.REPORT / "study.json").read_text(encoding="utf-8"))
plus10 = main - 2.0 * (main - plus5); plus10["BIL"] = bil
B = dict(c.BOOKS)
B["T2 MR -> BIL (GR1-L)"] = {"taa3x": 1/3, "ndx_atr_cap": 1/6, "ndx_natr_cap": 1/6, "BIL": 1/3}
print("== +10 bps: pod-level exact vs book-level extrapolation")
for name in ["GR1", "GR2", "GR3", "S9 incumbent launch", "T2 MR -> BIL (GR1-L)"]:
    r0 = c.book_returns(main, B[name]); r5 = c.book_returns(plus5, B[name]); r10p = c.book_returns(plus10, B[name])
    r10b = r0 + 2.0 * (r5 - r0)
    mp, mb = c.metrics(r10p, bil), c.metrics(r10b, bil)
    ref = study["books"][name]["frames"].get("plus_10bps") if name in study["books"] else None
    print(f"{name:24s} pod-level cagr {mp['cagr']:.6f} xs {mp['xs']:.6f} dd {mp['dd']:.6f} | book-extrap cagr {mb['cagr']:.6f} xs {mb['xs']:.6f} dd {mb['dd']:.6f} | study {ref and (ref['cagr'], ref['xs'], ref['dd'])}")
g0 = c.metrics(c.book_returns(main, B["GR1"]), bil); l0 = c.metrics(c.book_returns(main, B["T2 MR -> BIL (GR1-L)"]), bil)
for lab, f in (("pod-level", plus10),):
    g10 = c.metrics(c.book_returns(f, B["GR1"]), bil); l10 = c.metrics(c.book_returns(f, B["T2 MR -> BIL (GR1-L)"]), bil)
    for m in ("xs", "cagr"):
        d0 = g0[m] - l0[m]; d10 = g10[m] - l10[m]
        print(f"breakeven {m} [{lab}]: gap0 {d0:.6f} gap10 {d10:.6f} breakeven {10*d0/(d0-d10):.3f} bps | study {study['mr_gate_breakeven'][m]}")

print("\n== per-pod: fair-cash add and +5 bps drag (pp of CAGR, LONG and last 3y)")
last3 = main.index >= (main.index[-1] - pd.DateOffset(years=3))
def cagr(x): return float(np.prod(1 + x.values) ** (252 / len(x)) - 1)
for p in ["taa3x", "taa3x_1n", "btal_qqq", "ndx_vxn", "core5", "ndx_atr_cap", "ndx_natr_cap", "dv2_g", "hpi_g"]:
    print(f"{p:13s} house {cagr(house[p])*100:6.2f}% main {cagr(main[p])*100:6.2f}% add {100*(cagr(main[p])-cagr(house[p])):+.3f}pp (last3y {100*(cagr(main[p][last3])-cagr(house[p][last3])):+.3f}) | +5bps {cagr(plus5[p])*100:6.2f}% drag {100*(cagr(plus5[p])-cagr(main[p])):+.3f}pp | vol {main[p].std()*np.sqrt(252)*100:.2f}% xs {c.metrics(main[p], bil)['xs']:.3f}")

print("\n== proxy path vs real path after 2012-10-02 (house returns)")
for a in ["taa3x", "taa3x_1n", "btal_qqq"]:
    pr = house[a + "__proxyfull"].loc[c.EXACT_START:]; rr = house[a + "__real"].loc[c.EXACT_START:]
    print(f"{a:9s} corr {np.corrcoef(pr, rr)[0,1]:.6f} cagr proxy-run {cagr(pr)*100:.3f}% real {cagr(rr)*100:.3f}% max|diff| {np.abs(pr-rr).max():.5f}")
    pp = house[a].loc[:c.EXACT_START - pd.Timedelta(days=1)]
    print(f"          proxy era: n {len(pp)} cagr {cagr(pp)*100:.2f}% vol {pp.std()*np.sqrt(252)*100:.2f}% | exact era cagr {cagr(rr)*100:.2f}% vol {rr.std()*np.sqrt(252)*100:.2f}%")

print("\n== raw path integrity: total = portfolio + cash; cash weights in LONG")
import itertools
for a, src in itertools.chain(((x, c.OLD_SRC) for x in ["taa3x", "taa3x_1n", "btal_qqq", "ndx_vxn", "core5"]), ((x, c.NEW_SRC) for x in c.NEW_ALIASES), ((x, c.PROXY_SRC) for x in c.PROXY_ALIASES)):
    p = c.read_path(src / f"{a}__path.csv.gz")
    err = (p["total_value_float"] - p["portfolio_value_float"] - p["cash_float"]).abs().max()
    q = p.loc[c.LONG_START:c.END]
    cw = q["cash_float"] / q["total_value_float"]
    dup = p.index.duplicated().sum()
    print(f"{a:13s} {src.name:14s} rows {len(p)} dup {dup} max|total-(pv+cash)| {err:.4f} | LONG cash weight mean {cw.mean():+.4f} min {cw.min():+.4f} max {cw.max():+.4f} | nonpos NAV {(p['total_value_float']<=0).sum()}")
