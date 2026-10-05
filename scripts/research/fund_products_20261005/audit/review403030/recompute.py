"""Independent recomputation of GR1 40/30/30 (review lens). Reads frames via g_lib.Lab; own book/bootstrap arithmetic."""
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
import g_lib as g
from g_lib import Lab, LONG_START, END, TBILL, BLOCK_DICT
t0 = time.time()
lab = Lab()
main = lab.frames["main"][0]; f5 = lab.frames["s3_plus_5bps"][0]
print("load", round(time.time() - t0), "s; start", lab.start, LONG_START, END, "n", lab.n)
print("tbill same in f5:", float((main[TBILL] - f5[TBILL]).loc[LONG_START:END].abs().max()))
rf = main[TBILL].loc[LONG_START:END]

GR1 = {"taa3x": .40, "ndx_atr_cap": .15, "ndx_natr_cap": .15, "dv2_g": .15, "hpi_g": .15}
S0 = {"taa3x": 1/3, "ndx_atr_cap": 1/6, "ndx_natr_cap": 1/6, "dv2_g": 1/6, "hpi_g": 1/6}
S9 = {"taa3x_1n": 0.384, "ndx_vxn": 0.256, "core5": 0.18, "btal_qqq": 0.18}
S8 = {**{k: .75 * v for k, v in GR1.items()}, "core5": .15, "btal_qqq": .10}
SLOT = {"T1 MOM -> BIL": {"taa3x": .40, TBILL: .30, "dv2_g": .15, "hpi_g": .15},
        "T2 MR -> BIL (GR1-L)": {"taa3x": .40, "ndx_atr_cap": .15, "ndx_natr_cap": .15, TBILL: .30},
        "T3 TAA -> BIL": {TBILL: .40, "ndx_atr_cap": .15, "ndx_natr_cap": .15, "dv2_g": .15, "hpi_g": .15},
        "T4 MOM -> QQQ": {"taa3x": .40, g.QQQ: .30, "dv2_g": .15, "hpi_g": .15}}

def book(frame, w):
    """Own arithmetic: each calendar year start with target weights, pods compound, no intra-year rebalance."""
    df = frame.loc[LONG_START:END, list(w)]
    assert not df.isna().any().any()
    wv = pd.Series(w)
    nav_prev, out = 1.0, []
    for y, blk in df.groupby(df.index.year):
        pods = (1.0 + blk).cumprod() * wv          # value of each pod per 1 of start-of-year NAV
        lvl = pods.sum(axis=1) * nav_prev
        out.append(lvl)
        nav_prev = float(lvl.iloc[-1])
    lvl = pd.concat(out)
    return lvl / lvl.shift(1).fillna(1.0) - 1.0

def st(r, rf_=rf):
    x = r.to_numpy(); f = rf_.reindex(r.index).to_numpy()
    nav = np.concatenate([[1.0], np.cumprod(1 + x)])
    ex = x - f
    return dict(cagr=nav[-1] ** (252 / len(x)) - 1, vol=x.std(ddof=1) * 252 ** .5, xs=ex.mean() / ex.std(ddof=1) * 252 ** .5,
                dd=(nav / np.maximum.accumulate(nav) - 1).min())

def breach(r, limit=-0.20, own_idx=False):
    x = r.to_numpy(); n = len(x); res = []
    for s in range(10):
        if own_idx:
            rng = np.random.default_rng(777 + s)
            idx = np.empty((n, 2000), dtype=np.int64)
            idx[0] = rng.integers(0, n, 2000)
            for t in range(1, n):
                new = rng.random(2000) < 1 / 63.0
                idx[t] = np.where(new, rng.integers(0, n, 2000), (idx[t - 1] + 1) % n)
        else:
            idx = lab.idx(s)                         # (days, paths) the study's index convention
        nav = np.cumprod(1 + x[idx], axis=0)
        peak = np.maximum(np.maximum.accumulate(nav, axis=0), 1.0)
        dd = (nav / peak - 1).min(axis=0)
        res.append((dd < limit).mean())
    return float(np.mean(res)), float(np.max(res))

def paired(ra, rb):
    A = pd.concat([ra, rb, rf.reindex(ra.index)], axis=1).dropna().to_numpy(); n = len(A)
    gx, gc = [], []
    for s in range(10):
        idx = lab.idx(s, g.BOOT_BLOCK, n)
        a, b, f = A[:, 0][idx], A[:, 1][idx], A[:, 2][idx]
        xa, xb = a - f, b - f
        gx.append(xa.mean(0) / xa.std(0, ddof=1) - xb.mean(0) / xb.std(0, ddof=1))
        gc.append(np.log1p(a).sum(0) - np.log1p(b).sum(0))
    gx, gc = np.concatenate(gx), np.concatenate(gc)
    return float((gx > 0).mean()), float((gc > 0).mean())

def decay(frame, k, only=None):
    out = frame.copy(); win = frame.loc[LONG_START:END]
    for a, cap in g.CAPSULE_OF.items():
        if a in out.columns and (only is None or cap in only):
            out[a] = frame[a] - (1 - k) * float((win[a] - win[TBILL]).mean())
    return out

res = {}
r_main, r_5 = book(main, GR1), book(f5, GR1)
res["gr1_main"] = st(r_main); res["gr1_p5"] = st(r_5)
res["gr1_main_breach"] = breach(r_main); res["gr1_p5_breach"] = breach(r_5)
res["gr1_main_breach_ownidx"] = breach(r_main, own_idx=True)
print(json.dumps(res, indent=0, default=float), flush=True)
for lab_, k in (("planning", .75), ("floor", .5)):
    r = book(decay(f5, k), GR1); res[lab_] = {**st(r), "p20": breach(r)[0]}
    r = book(decay(f5, k), S0); res[lab_ + "_S0"] = {**st(r), "p20": breach(r)[0]}
r = book(decay(main, 0.0, only=("TAA",)), GR1); res["taa_dead"] = {**st(r), "p20": breach(r)[0]}
r = book(decay(main, 0.0, only=("MOM",)), GR1); res["mom_dead"] = st(r)
r = book(decay(main, 0.0, only=("MR",)), GR1); res["mr_dead"] = st(r)
r = book(decay(main, 0.0, only=("TAA",)), S0); res["taa_dead_S0"] = st(r)
print(json.dumps({k: res[k] for k in ("planning", "floor", "planning_S0", "floor_S0", "taa_dead", "mom_dead", "mr_dead", "taa_dead_S0")}, indent=0, default=float), flush=True)
for nm, w in (("S9", S9), ("S0", S0), ("S8", S8)):
    a, b = book(main, w), book(f5, w)
    res[nm] = {"main": st(a), "p5": st(b), "p20": breach(a)[0]}
    for fk, (x, y) in {"main": (r_main, a), "plus5": (r_5, b), "plus10": (r_main + 2 * (r_5 - r_main), a + 2 * (b - a))}.items():
        res[f"GR1|{nm} {fk}"] = paired(x, y)
    print(nm, json.dumps({k: v for k, v in res.items() if nm in k}, default=float), flush=True)
for nm, w in SLOT.items():
    a, b = book(main, w), book(f5, w)
    res[nm] = {"main": st(a), "p5": st(b), "p20": breach(a)[0],
               "pair_main": paired(r_main, a), "pair_p5": paired(r_5, b), "pair_p10": paired(r_main + 2 * (r_5 - r_main), a + 2 * (b - a))}
    print(nm, json.dumps(res[nm], default=float), flush=True)
(HERE / "recompute_out.json").write_text(json.dumps(res, indent=1, default=float))
print("done", round(time.time() - t0), "s")
