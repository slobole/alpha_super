"""Steps 2, 3, 5: books in MAIN and +5 bps, compared with study.json / battery.json."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ir_core as c

main = pd.read_pickle(c.OUT / "main_frame.pkl")
plus5 = pd.read_pickle(c.OUT / "plus5_frame.pkl")
house = pd.read_pickle(c.OUT / "house_frame.pkl")
bil = main["BIL"]
study = json.loads((c.REPORT / "study.json").read_text(encoding="utf-8"))
battery = json.loads((c.REPORT / "battery.json").read_text(encoding="utf-8"))

rows = []


def cmp(label, mine, theirs, tol=1e-6):
    d = mine - theirs
    flag = "" if abs(d) <= tol else "  <-- DIFF"
    rows.append({"item": label, "mine": mine, "study": theirs, "diff": d})
    print(f"{label:58s} mine {mine: .6f}  study {theirs: .6f}  diff {d: .2e}{flag}")


for name, w in c.BOOKS.items():
    r = c.book_returns(main, w)
    m = c.metrics(r, bil)
    sb = study["books"][name]
    q = sb["q"]
    print(f"=== {name}  (study weights {sb['weights']})")
    for k in ("cagr", "vol", "xs", "sharpe0", "dd"):
        cmp(f"{name} MAIN {k}", m[k], q[k])
    assert m["n"] == q["n"], (m["n"], q["n"])
    yr = c.year_returns(r)
    if "years" in sb:
        for y, v in sb["years"].items():
            cmp(f"{name} MAIN year {y}", yr[int(y)], v)
    cr = c.crisis_returns(r)
    for k, v in q["crises"].items():
        mine_incl, mine_excl = cr[k]["incl"], cr[k]["excl"]
        best = mine_incl if abs(mine_incl - v) <= abs(mine_excl - v) else mine_excl
        which = "incl" if best == mine_incl else "excl"
        cmp(f"{name} MAIN crisis {k} [{which}]", best, v)
    if "crises_dd" in sb:
        for k, v in sb["crises_dd"].items():
            cmp(f"{name} MAIN crisis dd {k}", cr[k]["dd_in_window"], v)
    # +5 bps
    r5 = c.book_returns(plus5, w)
    m5 = c.metrics(r5, bil)
    f5 = sb.get("frames", {}).get("s3_plus_5bps")
    if f5:
        for k in ("cagr", "vol", "xs", "sharpe0", "dd"):
            cmp(f"{name} +5bps {k}", m5[k], f5[k])
    # house cash
    rh = c.book_returns(house, w)
    mh = c.metrics(rh, bil)
    fh = sb.get("frames", {}).get("s1_house_cash")
    if fh:
        for k in ("cagr", "xs", "dd"):
            cmp(f"{name} house {k}", mh[k], fh[k])
    # EXACT window (book restarted 2012-10-02 on real paths)
    fe = sb.get("frames", {}).get("s6_exact")
    if fe:
        ex = main.loc[c.EXACT_START:]
        re_ = c.book_returns(ex, w)
        me = c.metrics(re_, bil)
        for k in ("cagr", "xs", "dd"):
            cmp(f"{name} EXACT(restart) {k}", me[k], fe[k])
        assert me["n"] == fe["n"], (me["n"], fe["n"])

# edge decay: TAA dead for GR1
print("=== edge decay TAA dead (GR1)")
mu = float((main["taa3x"] - bil).mean())
dead = main.copy()
dead["taa3x"] = dead["taa3x"] - mu
r = c.book_returns(dead, c.BOOKS["GR1"])
m = c.metrics(r, bil)
sc = battery["edge_decay"]["scenarios"]["TAA dead"]["GR1"]
print("battery scenario record:", sc)
for k in ("cagr", "xs"):
    if k in sc:
        cmp(f"GR1 TAA dead {k}", m[k], sc[k])
print("TAA dead single-path dd (mine):", m["dd"])

pd.DataFrame(rows).to_csv(c.OUT / "step2_3_5_compare.csv", index=False)
bad = [r for r in rows if abs(r["diff"]) > 1e-6]
print(f"\n{len(rows)} comparisons, {len(bad)} above 1e-6")
for b in bad:
    print(b)
