"""Independent recomputation, review v5.2 (own book arithmetic; g_lib.Lab only loads frames)."""
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
import g_lib as g
t0 = time.time()
lab = g.Lab()
main, start = lab.frames["main"]
f5 = lab.frames["s3_plus_5bps"][0]
END = g.END
M = main.loc[start:END]; F5 = f5.loc[start:END]
rf = M[g.TBILL]
print("loaded", round(time.time() - t0, 1), "s", M.shape, M.index[0], M.index[-1], flush=True)

def book(fr, w):
    """annual reset on the first session of each calendar year; pods compound in between."""
    cols = list(w); ww = np.array([w[c] for c in cols]); assert abs(ww.sum() - 1) < 1e-9
    X = fr[cols]; assert not X.isna().any().any()
    out = []; 
    for y, blk in X.groupby(X.index.year):
        lvl = ((1 + blk).cumprod() * ww).sum(axis=1)
        prev = lvl.shift(1).fillna(1.0)
        out.append(lvl / prev - 1)
    return pd.concat(out)

def st(r, rf_=None):
    rf_ = rf if rf_ is None else rf_
    x = r.to_numpy(); f = rf_.reindex(r.index).to_numpy()
    nav = np.cumprod(1 + x); pk = np.maximum.accumulate(np.r_[1.0, nav])[1:]
    e = x - f
    return dict(cagr=nav[-1] ** (252 / len(x)) - 1, vol=x.std(ddof=1) * 252 ** .5, xs=e.mean() / e.std(ddof=1) * 252 ** .5, dd=(nav / pk - 1).min())

IDX = {}
def idx(s, n):
    if (s, n) not in IDX:
        IDX[(s, n)] = g.lib.evaluation.stationary_bootstrap_index_mat(n, 2000, 63.0, g.SEED0 + s).astype(np.int32)  # (paths, days)
    return IDX[(s, n)]

def tails(r, limits=(-0.20, -0.25, -0.30)):
    x = r.to_numpy(); res = {L: [] for L in limits}
    for s in range(10):
        ii = idx(s, len(x))
        nav = np.cumprod(1 + x[ii], axis=1)
        pk = np.maximum(np.maximum.accumulate(nav, axis=1), 1.0)
        dd = (nav / pk - 1).min(axis=1)
        for L in limits: res[L].append((dd < L).mean())
    return {f"p{int(-L*100)}": float(np.mean(v)) for L, v in res.items()} | {f"p{int(-L*100)}_max": float(np.max(v)) for L, v in res.items()}

def own_tails(r, seed=777, reps=4000):
    """fully independent stationary bootstrap (own RNG) as a sanity check on the index generator"""
    x = r.to_numpy(); n = len(x); rng = np.random.default_rng(seed)
    startp = rng.integers(0, n, size=(reps, n)); new = rng.random((reps, n)) < 1 / 63.0; new[:, 0] = True
    ii = np.empty((reps, n), dtype=np.int64)
    cur = startp[:, 0]
    for t in range(n):
        cur = np.where(new[:, t], startp[:, t], (cur + 1) % n); ii[:, t] = cur
    nav = np.cumprod(1 + x[ii], axis=1); pk = np.maximum(np.maximum.accumulate(nav, axis=1), 1.0); dd = (nav / pk - 1).min(axis=1)
    return {"p20": float((dd < -0.20).mean()), "p25": float((dd < -0.25).mean())}

def paired(ra, rb):
    A = np.column_stack([ra.to_numpy(), rb.to_numpy(), rf.reindex(ra.index).to_numpy()]); n = len(A); wx = wc = tot = 0
    for s in range(10):
        ii = idx(s, n)
        for a in range(0, 2000, 250):
            smp = A[ii[a:a + 250]]
            xa = smp[..., 0] - smp[..., 2]; xb = smp[..., 1] - smp[..., 2]
            gx = xa.mean(1) / xa.std(1, ddof=1) - xb.mean(1) / xb.std(1, ddof=1)
            gc = np.log1p(smp[..., 0]).sum(1) - np.log1p(smp[..., 1]).sum(1)
            wx += (gx > 0).sum(); wc += (gc > 0).sum(); tot += len(gx)
    return dict(share_xs=wx / tot, share_cagr=wc / tot)

def decay(fr, k_of):
    out = fr.copy()
    for a, cap in g.CAPSULE_OF.items():
        if a in out.columns and cap in k_of:
            out[a] = fr[a] - (1 - k_of[cap]) * float((fr[a] - fr[g.TBILL]).mean())
    return out

GR1 = {"taa3x": .40, "ndx_atr_cap": .15, "ndx_natr_cap": .15, "dv2_g": .15, "hpi_g": .15}
GR2 = {"taa3x_1n": .40, "ndx_atr_cap": .15, "ndx_natr_cap": .15, "dv2_g": .15, "hpi_g": .15}
GR3 = {"taa3x_1n": .50, "ndx_atr_cap": .125, "ndx_natr_cap": .125, "dv2_g": .125, "hpi_g": .125}
MON = {"taa3x_1n": .50, "core5": .50}; MONP = {"taa3x_1n": .65, "core5": .35}
OLDM = {"taa3x_1n": 0.384, "ndx_vxn": 0.256, "core5": 0.18, "btal_qqq": 0.18}
OLDP = {"taa3x_1n": 0.574, "ndx_vxn": 0.246, "core5": 0.09, "btal_qqq": 0.09}
B = {"GR1": GR1, "GR2": GR2, "GR3": GR3, "S9 incumbent launch": MON, "old growth plus": MONP, "S13 old monthly (2026-10-01)": OLDM, "old monthly plus (2026-10-01)": OLDP}
ALLK = lambda k: dict.fromkeys(("TAA", "MOM", "MR", "DEF"), k)
Mc = decay(M, ALLK(0.75)); F5s = decay(F5, ALLK(0.5)); Mt = decay(M, {"TAA": 0.0})
out = {"books": {}}
for n, w in B.items():
    r0, r5 = book(M, w), book(F5, w)
    row = {"main": st(r0) | tails(r0), "plus5": st(r5) | tails(r5), "plus10": st(r0 + 2 * (r5 - r0)),
           "cons": st(book(Mc, w)) | tails(book(Mc, w)), "stress": st(book(F5s, w)) | tails(book(F5s, w)), "taa_dead": st(book(Mt, w))}
    if n in ("GR2", "S9 incumbent launch", "old growth plus"): row["own_boot"] = own_tails(r0)
    out["books"][n] = row
    print(n, {k: {a: round(float(b), 4) for a, b in v.items()} for k, v in row.items()}, flush=True)
out["paired"] = {}
for a, b in (("GR1", "S9 incumbent launch"), ("GR2", "GR1"), ("S9 incumbent launch", "S13 old monthly (2026-10-01)"), ("GR2", "old growth plus")):
    row = {}
    for fk in ("main", "plus5", "plus10"):
        f = {"main": lambda w: book(M, w), "plus5": lambda w: book(F5, w), "plus10": lambda w: book(M, w) + 2 * (book(F5, w) - book(M, w))}[fk]
        row[fk] = paired(f(B[a]), f(B[b]))
    out["paired"][f"{a} | {b}"] = row; print(a, "|", b, row, flush=True)
# blends
DEFL = {"core5": .54, "btal_qqq": .36, g.TBILL: .10}
out["blends"] = {}
for sh in (0.0, .2, .4, .6, .8, 1.0):
    w = {}
    for k, v in GR1.items(): w[k] = w.get(k, 0) + sh * v
    for k, v in DEFL.items(): w[k] = w.get(k, 0) + (1 - sh) * v
    w = {k: v for k, v in w.items() if v > 1e-12}
    r = book(M, w); out["blends"][str(sh)] = st(r) | tails(r); print("blend", sh, {a: round(float(b), 4) for a, b in out["blends"][str(sh)].items()}, flush=True)
# monthly grid spot checks + levered rows
out["grid"] = {}
def cell(taa, t, m):
    w = {taa: (1 - m) * t, "core5": (1 - m) * (1 - t)}
    if m: w |= {"ndx_atr_cap": m / 2, "ndx_natr_cap": m / 2}
    return w
for taa, t, m in (("taa3x", .4, .15), ("taa3x_1n", .5, .30), ("taa3x", .7, .30), ("taa3x_1n", .6, .15)):
    w = cell(taa, t, m); r = book(M, w); r5 = book(F5, w)
    row = {"w": w, "main": st(r) | tails(r), "plus5": st(r5), "vs0": paired(r, book(M, cell(taa, t, 0)))}
    out["grid"][f"{taa} {t:.0%}:{1-t:.0%} mom {m:.0%}"] = row; print("grid", taa, t, m, row, flush=True)
out["lev"] = {}
for nm, w, L in (("1N 50 / CORE5 50 x1.25", MON, 1.25), ("1N 40 / CORE5 60 x1.50", {"taa3x_1n": .4, "core5": .6}, 1.5)):
    wl = {k: L * v for k, v in w.items()} | {"debt": -(L - 1)}
    r = book(M, wl); r5 = book(F5, wl); out["lev"][nm] = {"main": st(r) | tails(r), "plus5": st(r5)}; print("lev", nm, out["lev"][nm], flush=True)
    # own debt check
days = pd.Series(M.index, index=M.index).diff().dt.days
print("debt col head", M["debt"].head(3).to_dict(), "mean ann", float(M["debt"].mean() * 252), "rf mean ann", float(rf.mean() * 252))
def cv(o):
    if isinstance(o, dict): return {str(k): cv(v) for k, v in o.items()}
    if isinstance(o, (np.floating, float)): return float(o)
    return o
(HERE / "rc1_out.json").write_text(json.dumps(cv(out), indent=1))
print("done", round(time.time() - t0, 1))
