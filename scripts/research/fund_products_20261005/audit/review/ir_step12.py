"""Unscaled-proxy frame: GR1 frame check and the knife-edge first-half check (c5) for S8 / S5."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ir_core as c

main = pd.read_pickle(c.OUT / "main_frame.pkl")
bil = main["BIL"]
study = json.loads((c.REPORT / "study.json").read_text(encoding="utf-8"))
UNS = c.PROXY_SRC.parent / "splice_unscaled"
f2 = main.copy()
for a in c.PROXY_ALIASES:
    p = c.read_path(UNS / f"{a}__path.csv.gz")
    t = c.read_tx(UNS / f"{a}__transactions.csv.gz")
    fr = c.pod_frames(p, t)
    s = (fr["r"] + fr["add"]).reindex(main.index)
    pre = main.index < c.EXACT_START
    f2.loc[pre, a] = s[pre]
B = dict(c.BOOKS)
B["S5 MR tilt"] = {"taa3x": 0.5, "ndx_atr_cap": 0.075, "ndx_natr_cap": 0.075, "dv2_g": 0.175, "hpi_g": 0.175}
B["S8 GR1 75 / DEF 25"] = {"taa3x": 0.25, "ndx_atr_cap": 0.125, "ndx_natr_cap": 0.125, "dv2_g": 0.125, "hpi_g": 0.125, "core5": 0.15, "btal_qqq": 0.10}
m = c.metrics(c.book_returns(f2, B["GR1"]), bil)
r = study["books"]["GR1"]["frames"]["s2_proxy_unscaled"]
print(f"GR1 unscaled proxy: mine cagr {m['cagr']:.6f} xs {m['xs']:.6f} dd {m['dd']:.6f} | study {r['cagr']} {r['xs']} {r['dd']}")
cut = pd.Timestamp("2017-06-30")
for lab, f in (("MAIN (scaled proxy)", main), ("unscaled proxy", f2)):
    h = {}
    for name in ("GR1", "S5 MR tilt", "S8 GR1 75 / DEF 25", "S9 incumbent launch"):
        rr = c.book_returns(f, B[name])
        h[name] = (c.metrics(rr.loc[:cut], bil)["xs"], c.metrics(rr, bil)["xs"])
    print(lab, {k: (round(v[0], 4), round(v[1], 4)) for k, v in h.items()}, "| H1 gap S8-GR1", round(h["S8 GR1 75 / DEF 25"][0] - h["GR1"][0], 4), "S5-GR1", round(h["S5 MR tilt"][0] - h["GR1"][0], 4))
# alternative cut dates for the halves (sensitivity of a knife-edge check)
for cd in ("2016-12-30", "2017-06-30", "2017-12-29"):
    ct = pd.Timestamp(cd)
    g = c.book_returns(main, B["GR1"])
    s8 = c.book_returns(main, B["S8 GR1 75 / DEF 25"])
    print(f"cut {cd}: H1 GR1 {c.metrics(g.loc[:ct], bil)['xs']:.4f} S8 {c.metrics(s8.loc[:ct], bil)['xs']:.4f}")
