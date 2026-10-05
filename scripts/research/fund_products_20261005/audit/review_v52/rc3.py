"""Review v5.2, third pass: levered monthly rows at +5 bps (rung), own debt column, monthly edge margin incl. worst seed and +5 bps."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
import g_lib as g
lab = g.Lab()
main, start = lab.frames["main"]; f5 = lab.frames["s3_plus_5bps"][0]
M = main.loc[start:g.END]; F5 = f5.loc[start:g.END]
def book(fr, w):
    cols = list(w); ww = np.array([w[c] for c in cols]); out = []
    for y, blk in fr[cols].groupby(fr.index.year):
        lvl = ((1 + blk).cumprod() * ww).sum(axis=1); out.append(lvl / lvl.shift(1).fillna(1.0) - 1)
    return pd.concat(out)
IDX = {}
def idx(s, n):
    if (s, n) not in IDX: IDX[(s, n)] = g.lib.evaluation.stationary_bootstrap_index_mat(n, 2000, 63.0, g.SEED0 + s).astype(np.int32)
    return IDX[(s, n)]
def tails(r, L):
    x = r.to_numpy(); v = []
    for s in range(10):
        nav = np.cumprod(1 + x[idx(s, len(x))], axis=1); dd = (nav / np.maximum(np.maximum.accumulate(nav, axis=1), 1.0) - 1).min(axis=1); v.append((dd < L).mean())
    return round(float(np.mean(v)), 4), round(float(np.max(v)), 4)
def dd(r):
    nav = np.cumprod(1 + r.to_numpy()); return round(float((nav / np.maximum.accumulate(np.r_[1.0, nav])[1:] - 1).min()), 4)
# own debt column: prior DTB3 observation + 1.5%, ACT/360 (compare with the frame's)
full = main.index; days = pd.Series(full, index=full).diff().dt.days
own_debt = ((lab.data["dtb3"] + 0.015) * days / 360.0).reindex(M.index)
print("debt col max abs diff vs own", float((own_debt - M["debt"]).abs().max()))
for nm, w, L in (("x1.25", {"taa3x_1n": .5, "core5": .5}, 1.25), ("x1.50", {"taa3x_1n": .4, "core5": .6}, 1.5)):
    wl = {k: L * v for k, v in w.items()} | {"debt": -(L - 1)}
    r0, r5 = book(M, wl), book(F5, wl)
    print(nm, "main dd", dd(r0), "p20", tails(r0, -0.20), "p25", tails(r0, -0.25), "| +5 dd", dd(r5), "p20", tails(r5, -0.20), "p25", tails(r5, -0.25), flush=True)
def decay(fr, k):
    out = fr.copy()
    for a_ in g.CAPSULE_OF:
        if a_ in out.columns: out[a_] = fr[a_] - (1 - k) * float((fr[a_] - fr[g.TBILL]).mean())
    return out
MON = {"taa3x_1n": .50, "core5": .50}; MONP = {"taa3x_1n": .65, "core5": .35}
for k in (1.0, 0.95, 0.9):
    a, b = decay(M, k), decay(F5, k)
    print("k", k, "Monthly main dd/p20(mean,worst)", dd(book(a, MON)), tails(book(a, MON), -0.20), "+5", dd(book(b, MON)), tails(book(b, MON), -0.20),
          "| MonthlyPlus main dd/p25", dd(book(a, MONP)), tails(book(a, MONP), -0.25), "+5", dd(book(b, MONP)), tails(book(b, MONP), -0.25), flush=True)
