"""GR1 vs S9 by block and cost frame; benchmark row; BIL sanity; 40/30/30 dial point."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ir_core as c

main = pd.read_pickle(c.OUT / "main_frame.pkl")
plus5 = pd.read_pickle(c.OUT / "plus5_frame.pkl")
bil = main["BIL"]
study = json.loads((c.REPORT / "study.json").read_text(encoding="utf-8"))
B = dict(c.BOOKS)
BLOCKS = {"A": ("2008-03-04", "2012-10-01"), "B": ("2012-10-02", "2021-12-31"), "C": ("2022-01-01", "2026-08-19"),
          "RECENT": ("2023-08-21", "2026-08-19")}

print("== GR1 vs S9 by block (book sliced from the full window)")
for fn, f in (("MAIN", main), ("+5bps", plus5)):
    g = c.book_returns(f, B["GR1"])
    s = c.book_returns(f, B["S9 incumbent launch"])
    for k, (lo, hi) in BLOCKS.items():
        mg = c.metrics(g.loc[lo:hi], bil)
        ms = c.metrics(s.loc[lo:hi], bil)
        print(f"   {fn:6s} block {k:6s} n {mg['n']:5d}: GR1 cagr {mg['cagr']*100:6.2f}% xs {mg['xs']:.3f} dd {mg['dd']*100:6.2f}% | S9 cagr {ms['cagr']*100:6.2f}% xs {ms['xs']:.3f} dd {ms['dd']*100:6.2f}% | xs gap GR1-S9 {mg['xs']-ms['xs']:+.3f}")
print("   study GR1 blocks:", {k: (v["cagr"], v["xs"], v["n"]) for k, v in study["books"]["GR1"]["blocks"].items()})
print("   study S9 blocks:", {k: (v["cagr"], v["xs"], v["n"]) for k, v in study["books"]["S9 incumbent launch"]["blocks"].items()})
ch = [x for x in study["challenges"] if x["challenger"].startswith("S9")][0]
print("   study block_gap_xs (S9 - GR1):", ch["block_gap_xs"])

print("")
print("== benchmarks on LONG (Norgate total return)")
for sym in ("$SPXTR", "QQQ"):
    r = c.load_tr(sym, main.index)
    m = c.metrics(r, bil)
    print(f"   {sym}: cagr {m['cagr']:.6f} xs {m['xs']:.4f} vol {m['vol']:.4f} dd {m['dd']:.4f} nan {int(r.isna().sum())}")
print("   study bench:", {k: (v.get("cagr"), v.get("xs"), v.get("dd")) for k, v in study["bench"].items()})

print("")
print("== BIL sanity on LONG")
print(f"   BIL cagr {float(np.prod(1+bil.values)**(252/len(bil))-1):.5f} max {bil.max():.5f} min {bil.min():.5f} n|r|>0.002 {int((bil.abs()>0.002).sum())}")
y = c.dtb3_rate(main.index)
days = pd.Series(main.index, index=main.index).diff().dt.days.fillna(1.0)
tb = (y * days / 360.0)
print(f"   DTB3 accrual cagr {float(np.prod(1+tb.values)**(252/len(tb))-1):.5f}; mean DTB3 {y.mean():.4f}")

print("")
print("== GR3 dial guard: TAA risk share and the 40/30/30 alternative (taa3x_1n)")


def contributions(ret_df, weights):
    cols = list(weights)
    w = np.array([weights[k] for k in cols])
    R = ret_df[cols].to_numpy()
    yrs = ret_df.index.year.values
    out = np.empty_like(R)
    pv = w.copy()
    tot = 1.0
    for i in range(len(R)):
        if i == 0 or yrs[i] != yrs[i - 1]:
            pv = w * tot
        out[i] = pv * R[i] / tot
        pv = pv * (1.0 + R[i])
        tot = pv.sum()
    return pd.DataFrame(out, index=ret_df.index, columns=cols)


for lab, w in (("GR3 50/25/25", B["GR3"]),
               ("dial 1N 40/30/30", {"taa3x_1n": 0.4, "ndx_atr_cap": 0.15, "ndx_natr_cap": 0.15, "dv2_g": 0.15, "hpi_g": 0.15})):
    con = contributions(main, w)
    book = con.sum(axis=1)
    rs = float(np.cov(con["taa3x_1n"], book, ddof=1)[0, 1] / np.var(book, ddof=1))
    m = c.metrics(c.book_returns(main, w), bil)
    print(f"   {lab}: TAA risk share {rs:.4f} (2/3 = 0.6667) | cagr {m['cagr']:.4f} xs {m['xs']:.3f} dd {m['dd']:.4f}")
