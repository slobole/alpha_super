"""Review v5.2, second pass: blends p10/p15 + diluted rows, versus blocks, monthly edge margin, grid claims."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
import g_lib as g
lab = g.Lab()
main, start = lab.frames["main"]; f5 = lab.frames["s3_plus_5bps"][0]
M = main.loc[start:g.END]; F5 = f5.loc[start:g.END]; rf = M[g.TBILL]
REP = g.OUT
S = json.loads((REP / "study.json").read_text()); V = json.loads((REP / "versus.json").read_text()); MO = json.loads((REP / "monthly.json").read_text()); B = json.loads((REP / "battery.json").read_text())

def book(fr, w):
    cols = list(w); ww = np.array([w[c] for c in cols]); assert abs(ww.sum() - 1) < 1e-9
    out = []
    for y, blk in fr[cols].groupby(fr.index.year):
        lvl = ((1 + blk).cumprod() * ww).sum(axis=1); out.append(lvl / lvl.shift(1).fillna(1.0) - 1)
    return pd.concat(out)
def st(r):
    x = r.to_numpy(); e = x - rf.reindex(r.index).to_numpy(); nav = np.cumprod(1 + x); pk = np.maximum.accumulate(np.r_[1.0, nav])[1:]
    return dict(cagr=nav[-1] ** (252 / len(x)) - 1, vol=x.std(ddof=1) * 252 ** .5, xs=e.mean() / e.std(ddof=1) * 252 ** .5, dd=(nav / pk - 1).min())
IDX = {}
def idx(s, n):
    if (s, n) not in IDX: IDX[(s, n)] = g.lib.evaluation.stationary_bootstrap_index_mat(n, 2000, 63.0, g.SEED0 + s).astype(np.int32)
    return IDX[(s, n)]
def tails(r, limits):
    x = r.to_numpy(); res = {L: [] for L in limits}
    for s in range(10):
        nav = np.cumprod(1 + x[idx(s, len(x))], axis=1); dd = (nav / np.maximum(np.maximum.accumulate(nav, axis=1), 1.0) - 1).min(axis=1)
        for L in limits: res[L].append((dd < L).mean())
    return {L: float(np.mean(v)) for L, v in res.items()}
def paired(ra, rb):
    A = np.column_stack([ra.to_numpy(), rb.to_numpy(), rf.reindex(ra.index).to_numpy()]); n = len(A); wx = wc = tot = 0
    for s in range(10):
        ii = idx(s, n)
        for a in range(0, 2000, 250):
            smp = A[ii[a:a + 250]]; xa = smp[..., 0] - smp[..., 2]; xb = smp[..., 1] - smp[..., 2]
            wx += ((xa.mean(1) / xa.std(1, ddof=1) - xb.mean(1) / xb.std(1, ddof=1)) > 0).sum()
            wc += ((np.log1p(smp[..., 0]).sum(1) - np.log1p(smp[..., 1]).sum(1)) > 0).sum(); tot += smp.shape[0]
    return round(wx / tot, 4), round(wc / tot, 4)
GR1 = {"taa3x": .40, "ndx_atr_cap": .15, "ndx_natr_cap": .15, "dv2_g": .15, "hpi_g": .15}
GR2 = {"taa3x_1n": .40, "ndx_atr_cap": .15, "ndx_natr_cap": .15, "dv2_g": .15, "hpi_g": .15}
MON = {"taa3x_1n": .50, "core5": .50}; MONP = {"taa3x_1n": .65, "core5": .35}
OLDM = {"taa3x_1n": 0.384, "ndx_vxn": 0.256, "core5": 0.18, "btal_qqq": 0.18}
DEFL = {"core5": .54, "btal_qqq": .36, g.TBILL: .10}
print("== blends: own p10/p15/p20 and diluted rows vs study.json")
keys = {0.2: "GR1 20 / defensive 80", 0.4: "GR1 40 / defensive 60", 0.6: "GR1 60 / defensive 40", 0.8: "GR1 80 / defensive 19"}
vol1 = st(book(M, GR1))["vol"]
for sh in (0.0, .2, .4, .6, .8, 1.0):
    w = {}
    for k, v in GR1.items(): w[k] = w.get(k, 0) + sh * v
    for k, v in DEFL.items(): w[k] = w.get(k, 0) + (1 - sh) * v
    w = {k: v for k, v in w.items() if v > 1e-12}
    r = book(M, w); s_ = st(r); t = tails(r, (-0.10, -0.15, -0.20))
    line = [sh, {k: round(v, 4) for k, v in s_.items()}, {k: round(v, 4) for k, v in t.items()}]
    if sh in keys:
        jb = S["blends"][keys[sh]]; jt = S["books"][keys[sh]]["tails"]
        c = max(0.0, round(1 - s_["vol"] / vol1, 2))
        wd = {k: v * (1 - c) for k, v in GR1.items()} | {g.TBILL: c}; sd = st(book(M, wd))
        line += ["json tails", {k: jt[k] for k in ("p10", "p15", "p20")}, "dil c", c, jb["diluted"]["cash"], {k: round(v, 4) for k, v in sd.items()},
                 "json dil", {k: round(jb["diluted"]["q"][k], 4) for k in ("cagr", "vol", "xs", "dd")},
                 "gap cagr/xs/dd", round(s_["cagr"] - sd["cagr"], 4), round(s_["xs"] - sd["xs"], 3), round(s_["dd"] - sd["dd"], 4)]
    else:
        nm = "defensive launch" if sh == 0 else "GR1"; jt = S["books"][nm]["tails"]; line += ["json tails", {k: jt[k] for k in ("p10", "p15", "p20")}]
    print(line, flush=True)
print("corr GR1 vs defensive launch", round(float(book(M, GR1).corr(book(M, DEFL))), 4), S["corr_gr1_defensive"])
print("== versus blocks (own) vs versus.json")
for a, b, wa, wb in (("GR1", "S9 incumbent launch", GR1, MON), ("S9 incumbent launch", "S13 old monthly (2026-10-01)", MON, OLDM), ("GR2", "GR1", GR2, GR1), ("GR2", "old growth plus", GR2, MONP)):
    ra, rb = book(M, wa), book(M, wb)
    for blk, (lo, hi) in g.BLOCK_DICT.items():
        xa, xb = g.lib.window(ra, lo, hi), g.lib.window(rb, lo, hi)
        j = V[f"{a} | {b}"]["blocks"][blk]["main"]
        print(a, "|", b, blk, lo, hi, len(xa), "own", paired(xa, xb), "json", round(j["share_xs"], 4), round(j["share_cagr"], 4),
              "own stats a/b", {k: round(v, 4) for k, v in st(xa).items()}, {k: round(v, 4) for k, v in st(xb).items()}, flush=True)
print("== edge margin for the monthly books (own): breach at the rung limit by k, MAIN")
def decay(fr, k):
    out = fr.copy()
    for a_ in g.CAPSULE_OF:
        if a_ in out.columns: out[a_] = fr[a_] - (1 - k) * float((fr[a_] - fr[g.TBILL]).mean())
    return out
for k in (1.0, .95, .9, .85, .8, .75, .7):
    fr = decay(M, k)
    print(k, "Monthly p20", round(tails(book(fr, MON), (-0.20,))[-0.20], 4), "MonthlyPlus p25", round(tails(book(fr, MONP), (-0.25,))[-0.25], 4),
          "GR1 p20", round(tails(book(fr, GR1), (-0.20,))[-0.20], 4), "GR2 p25", round(tails(book(fr, GR2), (-0.25,))[-0.25], 4), flush=True)
print("json edge margin", {n: B["edge_decay"]["edge_margin"][n]["lowest_passing_k"] for n in B["edge_decay"]["edge_margin"]})
print("== grid claims from monthly.json")
G = MO["grid"]; rows = []
for name, x in G.items():
    if x["momentum"] > 0:
        base = G[f"{x['taa']} {x['taa_to_core5']:.0%}:{1 - x['taa_to_core5']:.0%} mom 0%"]
        rows.append((name, x["vs_no_momentum"]["share_xs"], x["q"]["cagr"] - base["q"]["cagr"], x["q"]["dd"] - base["q"]["dd"], x["q"]["crises"], base["q"]["crises"]))
print("n", len(rows), "share min/max", min(r[1] for r in rows), max(r[1] for r in rows))
print("1N below 50%:", sum(1 for r in rows if "1n" in r[0] and r[1] < 0.5), "of", sum(1 for r in rows if "1n" in r[0]))
print("cagr diff", [round(r[2] * 100, 2) for r in rows], "<=0:", sum(1 for r in rows if r[2] <= 0), "rounded 1dp <=0:", sum(1 for r in rows if round(r[2] * 100, 1) <= 0))
print("dd diff pp", sorted(round(r[3] * 100, 2) for r in rows))
ck = list(rows[0][4].keys()); print("crisis keys", ck)
for key in ck: print(key, "worse in", sum(1 for r in rows if r[4][key] < r[5][key]), "of 16")
print("== GR1 vs S9 reverse challenge"); 
for c in S["challenges_reverse"]:
    if c["default"] == "S9 incumbent launch": print(c["checks"], c["passed"], c["share_xs"], c["share_cagr"], c["breach"], c["exact_xs"], c["plus5_xs"], c["halves_xs"])
for c in S["challenges"]:
    if c["challenger"] == "S9 incumbent launch": print("S9 vs GR1", c["checks"], c["share_xs"], c["share_cagr"])
