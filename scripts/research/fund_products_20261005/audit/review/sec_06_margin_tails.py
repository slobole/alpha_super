"""Reviewer scratch: breach figures of the margin rows with an independent stationary bootstrap, and the differences
against the plan's tolerance (0.3 pp CAGR, 0.5 pp drawdown / breach)."""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

WT = Path(r"C:\Users\User\Documents\workspace\alpha_super\.claude\worktrees\nervous-colden-784bbf")
OUT = WT / "results" / "research" / "portfolio" / "fund_products_20261005" / "audit" / "review" / "secondary"
D = pickle.load(open(OUT / "dump.pkl", "rb"))
END, LONG, EXACT = pd.Timestamp("2026-08-19"), pd.Timestamp("2008-03-04"), pd.Timestamp("2012-10-02")
main, plus5, exact = (D["frames"][k][0] for k in ("main", "s3_plus_5bps", "s6_exact"))


def three(taa, t, m, r):
    return {taa: t, "ndx_atr_cap": m / 2, "ndx_natr_cap": m / 2, "dv2_g": r / 2, "hpi_g": r / 2}


GR = {"GR1": three("taa3x", 1 / 3, 1 / 3, 1 / 3), "GR2": three("taa3x_1n", 1 / 3, 1 / 3, 1 / 3), "GR3": three("taa3x_1n", .5, .25, .25)}


def book(frame, w, start, debt="debt"):
    cols = list(w)
    R = frame.loc[start:END, cols]
    tw = np.array([w[c] for c in cols])
    yrs, v = R.index.year.to_numpy(), R.to_numpy()
    out = np.empty(len(R))
    eq, pods = 1.0, tw.copy()
    for i in range(len(R)):
        if i > 0 and yrs[i] != yrs[i - 1]:
            pods = tw * eq
        pods = pods * (1 + v[i])
        out[i] = pods.sum() / eq - 1
        eq = pods.sum()
    return pd.Series(out, index=R.index)


def lev(w, L, col="debt"):
    o = {k: L * v for k, v in w.items()}
    o[col] = -(L - 1)
    return o


def st(r):
    x = r.to_numpy()
    nav = np.r_[1.0, np.cumprod(1 + x)]
    return nav[-1] ** (252 / len(x)) - 1, (nav / np.maximum.accumulate(nav) - 1).min(), x.std(ddof=1) * np.sqrt(252)


def sb_index(n, reps, block, seed):
    rng = np.random.default_rng(seed)
    idx = np.empty((n, reps), dtype=np.int32)
    cur = rng.integers(0, n, size=reps)
    idx[0] = cur
    jump = rng.random((n, reps)) < 1.0 / block
    fresh = rng.integers(0, n, size=(n, reps))
    for t in range(1, n):
        cur = np.where(jump[t], fresh[t], (cur + 1) % n)
        idx[t] = cur
    return idx


N = len(main.loc[LONG:END])
IDX = [sb_index(N, 2000, 63.0, s) for s in (2101, 2102, 2103, 2104, 2105)]


def p_dd(r, limit):
    x = r.to_numpy()
    ps = []
    for idx in IDX:
        nav = np.cumprod(1 + x[idx], axis=0)
        dd = (nav / np.maximum.accumulate(np.maximum(nav, 1.0), axis=0) - 1).min(axis=0)
        ps.append(float((dd < limit).mean()))
    return float(np.mean(ps))


for base, L, tgt, lim in (("GR1", 1.19, "GR2", -0.25), ("GR1", 1.32, "GR3", -0.30), ("GR2", 1.11, "GR3", -0.30)):
    wl, wt = lev(GR[base], L), GR[tgt]
    a, b = book(main, wl, LONG), book(main, wt, LONG)
    ca, da, va = st(a)
    cb, db, vb = st(b)
    pa, pb = p_dd(a, lim), p_dd(b, lim)
    c5a, c5b = st(book(plus5, wl, LONG))[0], st(book(plus5, wt, LONG))[0]
    cea, ceb = st(book(exact, wl, EXACT))[0], st(book(exact, wt, EXACT))[0]
    c25 = st(book(main, lev(GR[base], L, "debt_250"), LONG))[0]
    flag = lambda d, tol: "OUT" if abs(d) > tol else "in"  # noqa: E731
    print(f"{base} x{L} vs {tgt}: vol {va:.4f} vs {vb:.4f}")
    print(f"   CAGR MAIN {ca:.4f} vs {cb:.4f}: {100 * (ca - cb):+.2f} pp [{flag(ca - cb, 0.003)}] | Max DD {da:.4f} vs {db:.4f}: {100 * (da - db):+.2f} pp [{flag(da - db, 0.005)}]"
          f" | breach P(DD<{lim:.0%}) own bootstrap {pa:.4f} vs {pb:.4f}: {100 * (pa - pb):+.2f} pp [{flag(pa - pb, 0.005)}]")
    print(f"   CAGR +5 bps {c5a:.4f} vs {c5b:.4f}: {100 * (c5a - c5b):+.2f} pp [{flag(c5a - c5b, 0.003)}] | CAGR 2012+ {cea:.4f} vs {ceb:.4f}: {100 * (cea - ceb):+.2f} pp [{flag(cea - ceb, 0.003)}]"
          f" | CAGR at 2.5% spread {c25:.4f} vs {cb:.4f}: {100 * (c25 - cb):+.2f} pp [{flag(c25 - cb, 0.003)}]")
