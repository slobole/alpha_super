"""Extra independent checks: planning / floor, risk shares, halves, paired shares by frame, slot shares per seed, net 2/20."""
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
plus10 = main - 2.0 * (main - plus5)  # drag is linear in bps
plus10["BIL"] = main["BIL"]
bil = main["BIL"]
study = json.loads((c.REPORT / "study.json").read_text(encoding="utf-8"))
battery = json.loads((c.REPORT / "battery.json").read_text(encoding="utf-8"))

B = dict(c.BOOKS)
B["S5 MR tilt"] = {"taa3x": 0.5, "ndx_atr_cap": 0.075, "ndx_natr_cap": 0.075, "dv2_g": 0.175, "hpi_g": 0.175}
B["S8 GR1 75 / DEF 25"] = {"taa3x": 0.25, "ndx_atr_cap": 0.125, "ndx_natr_cap": 0.125, "dv2_g": 0.125, "hpi_g": 0.125,
                           "core5": 0.15, "btal_qqq": 0.10}
B["T2 MR -> BIL (GR1-L)"] = {"taa3x": 1 / 3, "ndx_atr_cap": 1 / 6, "ndx_natr_cap": 1 / 6, "BIL": 1 / 3}
B["T3 TAA -> BIL"] = {"BIL": 1 / 3, "ndx_atr_cap": 1 / 6, "ndx_natr_cap": 1 / 6, "dv2_g": 1 / 6, "hpi_g": 1 / 6}

print("study weights S5:", study["books"]["S5 MR tilt"]["weights"])
print("study weights S8:", study["books"]["S8 GR1 75 / DEF 25"]["weights"])


def show(label, mine, theirs):
    d = mine - theirs
    print(f"{label:55s} mine {mine: .6f} study {theirs: .6f} diff {d: .2e}{'  <-- DIFF' if abs(d) > 1e-6 else ''}")


# (a) planning / floor: k = 0.75 / 0.5 on every pod, +5 bps frame, mu from the frame itself
print("\n(a) planning / floor")
for name in ["GR1", "GR2", "GR3", "S9 incumbent launch"]:
    w = B[name]
    for lab, k in (("planning", 0.75), ("floor", 0.5)):
        f = plus5.copy()
        for p in w:
            if p == "BIL":
                continue
            mu = float((plus5[p] - bil).mean())
            f[p] = plus5[p] - (1.0 - k) * mu
        m = c.metrics(c.book_returns(f, w), bil)
        ref = battery["edge_decay"]["headline"].get(name, {}).get(lab)
        if ref:
            show(f"{name} {lab} cagr", m["cagr"], ref["cagr"])
            show(f"{name} {lab} xs", m["xs"], ref["xs"])
        else:
            print(name, lab, m)

# (b) risk shares per capsule: Cov(contribution, book) / Var(book), contributions with drifting weights
print("\n(b) risk shares")


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


CAPS = {"TAA": ("taa3x", "taa3x_1n"), "MOM": ("ndx_atr_cap", "ndx_natr_cap"), "MR": ("dv2_g", "hpi_g")}
for name in ["GR1", "GR2", "GR3"]:
    con = contributions(main, B[name])
    book = con.sum(axis=1)
    ref = battery["dependence"]["products"][name]["risk_share"]
    for cap, pods in CAPS.items():
        cc = con[[p for p in pods if p in con.columns]].sum(axis=1)
        rs = float(np.cov(cc, book, ddof=1)[0, 1] / np.var(book, ddof=1))
        show(f"{name} risk share {cap}", rs, ref[cap])

# (c) halves, sliced from the full-window book
print("\n(c) halves (cut 2017-06-30)")
cut = pd.Timestamp("2017-06-30")
ref_by = {ch["challenger"]: ch for ch in study["challenges"]}
for name in ["GR1", "S5 MR tilt", "S8 GR1 75 / DEF 25", "S9 incumbent launch", "S1 no momentum"]:
    r = c.book_returns(main, B[name])
    h1 = c.metrics(r.loc[:cut], bil)["xs"]
    h2 = c.metrics(r.loc[cut + pd.Timedelta(days=1):], bil)["xs"]
    if name == "GR1":
        ref = study["books"]["GR1"]["halves_xs"]
    else:
        ref = ref_by[name]["halves_xs"][0]
    show(f"{name} h1 xs", h1, ref[0])
    show(f"{name} h2 xs", h2, ref[1])
    if name != "GR1":
        m = c.metrics(r, bil)
        show(f"{name} xs", m["xs"], ref_by[name]["xs"][0])
        m5 = c.metrics(c.book_returns(plus5, B[name]), bil)
        show(f"{name} +5 xs", m5["xs"], ref_by[name]["plus5_xs"][0])
        print(f"     {name}: MAIN dd {m['dd']:.6f}  +5bps dd {m5['dd']:.6f}")

# (d) paired shares by frame and per seed
print("\n(d) paired shares")
n = len(bil)
frames = {"main": main, "plus5": plus5, "plus10": plus10}
series = {fn: {k: c.book_returns(f, B[k]).to_numpy() for k in ["GR1", "S9 incumbent launch", "T1 MOM -> BIL", "S1 no momentum",
                                                                 "S5 MR tilt", "S8 GR1 75 / DEF 25", "T2 MR -> BIL (GR1-L)"]}
          for fn, f in frames.items()}
bil_a = bil.to_numpy()
acc = {}
for s in range(10):
    idx = c.sb_index(n, 2000, 63.0, 20260929 + s)
    bs = bil_a[idx]
    for fn in frames:
        xs = {}
        for k, arr in series[fn].items():
            ex = arr[idx] - bs
            xs[k] = ex.mean(axis=1) / ex.std(axis=1, ddof=1) * np.sqrt(252.0)
        for other in series[fn]:
            if other == "GR1":
                continue
            acc.setdefault((fn, other), []).append(float((xs["GR1"] > xs[other]).mean()))
for (fn, other), v in sorted(acc.items()):
    print(f"  GR1 xs > {other:24s} [{fn:6s}] pooled {np.mean(v):.5f}  per-seed min {min(v):.4f} max {max(v):.4f}")
print("study: GR1>S9 main", [x for x in study["challenges_reverse"] if x["default"].startswith("S9")][0]["share_xs"])
for k in ("T1 MOM -> BIL", "T2 MR -> BIL (GR1-L)"):
    fr = study["slots"][k]["frames"]
    print("study slot", k, {f: fr[f]["share_xs"] for f in fr})
print("study: S1 over GR1", ref_by["S1 no momentum"]["share_xs"], " S5 over GR1", ref_by["S5 MR tilt"]["share_xs"], " S8 over GR1", ref_by["S8 GR1 75 / DEF 25"]["share_xs"])

# (e) net of 2/20: own simple implementation (daily accrual of 2%/252 on NAV, 20% over HWM crystallised at each year end)
print("\n(e) net 2/20 (own convention; expect approximate agreement only)")


def net_nav(r: pd.Series) -> pd.Series:
    gross = 1.0  # gross NAV after paid fees, before accrued perf fee
    hwm = 1.0
    out = []
    yrs = r.index.year.values
    for i, x in enumerate(r.to_numpy()):
        gross *= (1.0 + x)
        gross *= (1.0 - 0.02 / 252.0)
        accr = 0.2 * max(gross - hwm, 0.0)
        net = gross - accr
        last = i == len(r) - 1 or yrs[i + 1] != yrs[i]
        if last:
            gross = net
            hwm = max(hwm, net)
        out.append(net)
    return pd.Series(out, index=r.index)


for name in ["GR1", "GR2", "GR3", "S9 incumbent launch"]:
    r = c.book_returns(main, B[name])
    nn = net_nav(r)
    cg = float(nn.iloc[-1] ** (252.0 / len(nn)) - 1.0)
    ref = study["books"][name].get("net", {}).get("cagr")
    print(f"  {name}: net CAGR mine {cg:.6f}  study {ref}")
