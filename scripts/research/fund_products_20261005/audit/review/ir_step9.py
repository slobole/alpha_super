"""More independent checks: S9 'TAA dead' mapping, MR fair-cash frame, block lengths, reset months, GR2 vs old plus, cash add arithmetic."""
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
B = dict(c.BOOKS)
B["old growth plus"] = {"taa3x_1n": 0.574, "ndx_vxn": 0.246, "core5": 0.09, "btal_qqq": 0.09}
print("study old plus weights:", study["books"]["old growth plus"]["weights"])


def dead(frame, pods):
    f = frame.copy()
    for p in pods:
        f[p] = f[p] - float((frame[p] - bil).mean())
    return f


print("== S9 'TAA dead' under different capsule mappings; report prints 6.8% / 0.45")
ref = battery["edge_decay"]["scenarios"]["TAA dead"]["S9 incumbent launch"]
print("   battery:", {k: ref[k] for k in ("cagr", "xs")})
for lab, pods in (("taa3x_1n only", ["taa3x_1n"]), ("taa3x_1n + btal_qqq", ["taa3x_1n", "btal_qqq"]),
                  ("taa3x_1n + btal_qqq + core5", ["taa3x_1n", "btal_qqq", "core5"])):
    m = c.metrics(c.book_returns(dead(main, pods), B["S9 incumbent launch"]), bil)
    print(f"   {lab:30s} cagr {m['cagr']:.6f} xs {m['xs']:.6f}")
refm = battery["edge_decay"]["scenarios"]["MOM dead"]["S9 incumbent launch"]
m = c.metrics(c.book_returns(dead(main, ["ndx_vxn"]), B["S9 incumbent launch"]), bil)
print(f"   MOM dead (ndx_vxn): mine cagr {m['cagr']:.6f} xs {m['xs']:.6f} | battery {refm['cagr']} {refm['xs']}")
for name, pods in (("GR2", ["taa3x_1n"]), ("GR3", ["taa3x_1n"])):
    m = c.metrics(c.book_returns(dead(main, pods), B[name]), bil)
    r = battery["edge_decay"]["scenarios"]["TAA dead"][name]
    print(f"   {name} TAA dead mine {m['cagr']:.6f} / {m['xs']:.6f} | battery {r['cagr']} / {r['xs']}")
for name, pods, sc in (("GR1", ["ndx_atr_cap", "ndx_natr_cap"], "MOM dead"), ("GR1", ["dv2_g", "hpi_g"], "MR dead")):
    m = c.metrics(c.book_returns(dead(main, pods), B[name]), bil)
    r = battery["edge_decay"]["scenarios"][sc][name]
    print(f"   {name} {sc} mine {m['cagr']:.6f} / {m['xs']:.6f} | battery {r['cagr']} / {r['xs']}")

print("")
print("== MR fair-cash frame (parking-off runs + fair cash) and MR cash 0% for GR1")
mr = {}
for a in ("dv2_g_cash", "hpi_g_cash"):
    p = c.read_path(c.NEW_SRC / f"{a}__path.csv.gz")
    t = c.read_tx(c.NEW_SRC / f"{a}__transactions.csv.gz")
    mr[a] = c.pod_frames(p, t).reindex(main.index)
f9 = main.copy()
f10 = main.copy()
f9["dv2_g"] = mr["dv2_g_cash"]["r"] + mr["dv2_g_cash"]["add"]
f9["hpi_g"] = mr["hpi_g_cash"]["r"] + mr["hpi_g_cash"]["add"]
f10["dv2_g"] = mr["dv2_g_cash"]["r"]
f10["hpi_g"] = mr["hpi_g_cash"]["r"]
for lab, f in (("s9_mr_fair_cash", f9), ("s10_mr_cash_0", f10)):
    m = c.metrics(c.book_returns(f, B["GR1"]), bil)
    r = study["books"]["GR1"]["frames"][lab]
    print(f"   GR1 {lab}: mine cagr {m['cagr']:.6f} xs {m['xs']:.6f} dd {m['dd']:.6f} | study {r['cagr']} {r['xs']} {r['dd']}")
cap_mr = {"dv2_g": 0.5, "hpi_g": 0.5}
m = c.metrics(c.book_returns(main, cap_mr), bil)
m5 = c.metrics(c.book_returns(plus5, cap_mr), bil)
mh = c.metrics(c.book_returns(house, cap_mr), bil)
m9 = c.metrics(c.book_returns(f9, cap_mr), bil)
q = study["books"]["capsule MR"]["q"]
print(f"   MR capsule: MAIN cagr {m['cagr']:.4f} xs {m['xs']:.3f} dd {m['dd']:.4f} | house {mh['cagr']:.4f} | +5 {m5['cagr']:.4f} | parking-off fair cash {m9['cagr']:.4f} xs {m9['xs']:.3f} | study q {q['cagr']} {q['xs']} {q['dd']}")
mm = c.metrics(c.book_returns(main, {"ndx_atr_cap": 0.5, "ndx_natr_cap": 0.5}), bil)
q = study["books"]["capsule MOM"]["q"]
print(f"   MOM capsule: MAIN cagr {mm['cagr']:.4f} xs {mm['xs']:.3f} dd {mm['dd']:.4f} | study q {q['cagr']} {q['xs']} {q['dd']}")

print("")
print("== cash add: arithmetic (mean daily add x 252) vs CAGR difference, LONG")
pods = c.build_pods()
for p in ["taa3x", "taa3x_1n", "ndx_atr_cap", "ndx_natr_cap", "dv2_g", "hpi_g"]:
    add = pods[p]["add"].reindex(main.index)
    last3 = add.index >= (add.index[-1] - pd.DateOffset(years=3))
    print(f"   {p:13s} arithmetic add {add.mean()*252*100:+.3f} pp/yr (last 3y {add[last3].mean()*252*100:+.3f})")

print("")
print("== reset month spread, GR1 (annual reset at first session of month m)")


def book_reset_month(ret_df, weights, month):
    cols = list(weights)
    w = np.array([weights[k] for k in cols])
    R = ret_df[cols].to_numpy()
    idx = ret_df.index
    out = np.empty(len(idx))
    pv = w.copy()
    tot = 1.0
    for i in range(len(idx)):
        newper = i == 0 or (idx[i].month == month and idx[i - 1].month != month)
        if newper:
            pv = w * tot
        pv = pv * (1 + R[i])
        t2 = pv.sum()
        out[i] = t2 / tot - 1
        tot = t2
    return pd.Series(out, index=idx)


vals = []
for mth in range(1, 13):
    mt = c.metrics(book_reset_month(main, B["GR1"], mth), bil)
    vals.append((mth, mt["cagr"], mt["xs"]))
print("   ", [(a, round(b, 5), round(x, 4)) for a, b, x in vals])
cg = sorted(v[1] for v in vals)
print("    CAGR min/median/max", cg[0], float(np.median(cg)), cg[-1], "| Jan", vals[0][1], "| quartiles", float(np.percentile(cg, 25)), float(np.percentile(cg, 75)))
print("    battery reset GR1:", json.dumps(battery["reset"]["GR1"])[:900])

print("")
print("== bootstrap: other block lengths (GR1, S9, limit -20%), seeds 0-9; GR1 vs S9 share at +4 bps; GR2 vs old plus")
n = len(bil)
bil_a = bil.to_numpy()
r_gr1 = c.book_returns(main, B["GR1"]).to_numpy()
r_s9 = c.book_returns(main, B["S9 incumbent launch"]).to_numpy()
plus4 = main - 0.8 * (main - plus5)
plus4["BIL"] = bil
g4 = c.book_returns(plus4, B["GR1"]).to_numpy()
s4 = c.book_returns(plus4, B["S9 incumbent launch"]).to_numpy()
g2 = c.book_returns(main, B["GR2"]).to_numpy()
op = c.book_returns(main, B["old growth plus"]).to_numpy()


def ddp(s):
    nav = np.cumprod(1 + s, axis=1)
    pk = np.maximum(np.maximum.accumulate(nav, axis=1), 1.0)
    return (nav / pk - 1).min(axis=1)


def xs(s, b):
    e = s - b
    return e.mean(axis=1) / e.std(axis=1, ddof=1)


for blk in (1.0, 21.0, 126.0, 252.0):
    a, b_ = [], []
    for s in range(10):
        idx = c.sb_index(n, 2000, blk, 20260929 + s)
        a.append((ddp(r_gr1[idx]) < -0.2).mean())
        b_.append((ddp(r_s9[idx]) < -0.2).mean())
    print(f"   block {blk:5.0f}: GR1 p20 {np.mean(a):.4f}  S9 p20 {np.mean(b_):.4f}")
print("   battery bootstrap blocks:", json.dumps(battery["bootstrap"]["blocks"])[:1200])
sh4, sh2, p25a, p25b = [], [], [], []
for s in range(10):
    idx = c.sb_index(n, 2000, 63.0, 20260929 + s)
    bs = bil_a[idx]
    sh4.append((xs(g4[idx], bs) > xs(s4[idx], bs)).mean())
    sh2.append((xs(g2[idx], bs) > xs(op[idx], bs)).mean())
    p25a.append((ddp(g2[idx]) < -0.25).mean())
    p25b.append((ddp(op[idx]) < -0.25).mean())
print(f"   GR1 > S9 share at +4 bps: {np.mean(sh4):.4f}")
ref2 = study["gr2_vs_old_plus"]
print(f"   GR2 > old plus share: {np.mean(sh2):.4f} (study {ref2['share_xs']}); p25 GR2 {np.mean(p25a):.5f} old plus {np.mean(p25b):.5f} (study {ref2['breach']})")
mo = c.metrics(pd.Series(op, index=main.index), bil)
qo = study["books"]["old growth plus"]["q"]
print("   old plus mine:", {k: round(v, 6) for k, v in mo.items()}, "| study", qo["cagr"], qo["xs"], qo["dd"])
