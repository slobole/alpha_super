"""+5 bps excluding BIL fills (s3b) for GR1; variance ratio; write a compact summary of the review numbers."""
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
battery = json.loads((c.REPORT / "battery.json").read_text(encoding="utf-8"))
B = dict(c.BOOKS)

f3b = plus5.copy()
for a in ("dv2_g", "hpi_g"):
    p = c.read_path(c.NEW_SRC / f"{a}__path.csv.gz")
    t = c.read_tx(c.NEW_SRC / f"{a}__transactions.csv.gz")
    t = t[t["asset_str"] != "BIL"]
    fr = c.pod_frames(p, t).reindex(main.index)
    f3b[a] = fr["r"] + fr["add"] - fr["drag"]
m = c.metrics(c.book_returns(f3b, B["GR1"]), bil)
r = study["books"]["GR1"]["frames"]["s3b_plus_5bps_ex_bil"]
print(f"GR1 +5bps ex BIL fills: mine cagr {m['cagr']:.6f} xs {m['xs']:.6f} dd {m['dd']:.6f} | study {r['cagr']} {r['xs']} {r['dd']}")

g = c.book_returns(main, B["GR1"])
lg = np.log1p(g.to_numpy())
for q in (21, 63):
    k = np.convolve(lg, np.ones(q), mode="valid")
    vr = k.var(ddof=1) / (q * lg.var(ddof=1))
    print(f"variance ratio GR1 q={q}: overlapping log-return VR {vr:.3f}")
print("battery variance_ratio:", json.dumps(battery["bootstrap"]["variance_ratio"])[:400])
ac = pd.Series(g.to_numpy()).autocorr(1)
print("GR1 daily autocorr lag 1:", round(ac, 4))
