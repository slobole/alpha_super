"""Reviewer scratch (secondary lens): the DEFENSIVE refresh rows with my own book loop, crisis windows and an
independent stationary bootstrap (own generator, own seeds)."""
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

WT = Path(r"C:\Users\User\Documents\workspace\alpha_super\.claude\worktrees\nervous-colden-784bbf")
MAIN = Path(r"C:\Users\User\Documents\workspace\alpha_super")
OUT = WT / "results" / "research" / "portfolio" / "fund_products_20261005" / "audit" / "review" / "secondary"
D = pickle.load(open(OUT / "dump.pkl", "rb"))
END, LONG, EXACT = pd.Timestamp("2026-08-19"), pd.Timestamp("2008-03-04"), pd.Timestamp("2012-10-02")
main, plus5, exact = (D["frames"][k][0] for k in ("main", "s3_plus_5bps", "s6_exact"))
rf = main["tbill"].loc[LONG:END]
CR = D["crisis"]


def book(frame, w, start=LONG):
    cols = list(w)
    R = frame.loc[start:END, cols]
    assert not R.isna().any().any()
    tw = np.array([w[c] for c in cols])
    assert abs(tw.sum() - 1) < 1e-9, tw.sum()
    yrs = R.index.year.to_numpy()
    v = R.to_numpy()
    out = np.empty(len(R))
    eq, pods = 1.0, tw.copy()
    for i in range(len(R)):
        if i > 0 and yrs[i] != yrs[i - 1]:
            pods = tw * eq
        pods = pods * (1 + v[i])
        new = pods.sum()
        out[i] = new / eq - 1
        eq = new
    return pd.Series(out, index=R.index)


def stats(r):
    x = r.to_numpy()
    f = main["tbill"].reindex(r.index).to_numpy()
    nav = np.r_[1.0, np.cumprod(1 + x)]
    return {"cagr": nav[-1] ** (252 / len(x)) - 1, "vol": x.std(ddof=1) * np.sqrt(252), "xs": (x - f).mean() / (x - f).std(ddof=1) * np.sqrt(252),
            "dd": (nav / np.maximum.accumulate(nav) - 1).min()}


def crisis(r, incl_start=False):
    out = {}
    for k, (lo, hi) in CR.items():
        lo, hi = pd.Timestamp(lo), pd.Timestamp(hi)
        w = r[(r.index >= lo if incl_start else r.index > lo) & (r.index <= hi)]
        out[k] = float((1 + w).prod() - 1)
    return out


def blend(*parts):
    out = {}
    for share, w in parts:
        tot = sum(w.values())
        for k, v in w.items():
            out[k] = out.get(k, 0.0) + share * v / tot
    return out


def with_cash(w, c):
    return blend((1 - c, w), (c, {"tbill": 1.0})) if c > 0 else dict(w)


C_L = {"core5": 0.6, "btal_qqq": 0.4}
GR1 = {"taa3x": 1 / 3, "ndx_atr_cap": 1 / 6, "ndx_natr_cap": 1 / 6, "dv2_g": 1 / 6, "hpi_g": 1 / 6}
S9 = {"taa3x_1n": 0.384, "ndx_vxn": 0.256, "core5": 0.18, "btal_qqq": 0.18}
MR = {"dv2_g": 0.5, "hpi_g": 0.5}

# ---- independent stationary bootstrap (Politis-Romano), own generator
N = len(rf)


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


SEEDS = [911, 912, 913, 914, 915]
IDX = [sb_index(N, 2000, 63.0, s) for s in SEEDS]


def p_dd(r, limit):
    x = r.to_numpy()
    ps = []
    for idx in IDX:
        nav = np.cumprod(1 + x[idx], axis=0)
        peak = np.maximum.accumulate(np.maximum(nav, 1.0), axis=0)
        dd = (nav / peak - 1).min(axis=0)
        ps.append(float((dd < limit).mean()))
    return float(np.mean(ps)), float(np.max(ps))


def paired_share(ra, rb):
    a, b, f = ra.to_numpy(), rb.to_numpy(), rf.to_numpy()
    wins, tot, wins_c = 0, 0, 0
    for idx in IDX:
        xa, xb = a[idx] - f[idx], b[idx] - f[idx]
        sa = xa.mean(axis=0) / xa.std(axis=0, ddof=1)
        sb = xb.mean(axis=0) / xb.std(axis=0, ddof=1)
        wins += int((sa > sb).sum())
        wins_c += int((np.log1p(a[idx]).sum(axis=0) > np.log1p(b[idx]).sum(axis=0)).sum())
        tot += idx.shape[1]
    return wins / tot, wins_c / tot


def full(name, w, show=True):
    r0, r5 = book(main, w), book(plus5, w)
    s0, s5 = stats(r0), stats(r5)
    c = crisis(r0)
    c2 = crisis(r0, incl_start=True)
    p0, p5 = p_dd(r0, -0.10), p_dd(r5, -0.10)
    ok_dd = s0["dd"] >= -0.07 and s5["dd"] >= -0.07
    ok_floor = c["gfc"] >= -0.01 and c["bear_2022"] >= -0.01 and min(c.values()) >= -0.05
    ok_p = max(p0) <= 0.10 and max(p5) <= 0.10
    if show:
        print(f"{name}: CAGR {s0['cagr']:.5f} xs {s0['xs']:.4f} vol {s0['vol']:.4f} DD {s0['dd']:.5f} | +5: CAGR {s5['cagr']:.5f} DD {s5['dd']:.5f} | GFC {c['gfc']:+.5f} 2022 {c['bear_2022']:+.5f} "
              f"worst {min(c.values()):+.5f} ({min(c, key=c.get)}) [start day included: worst {min(c2.values()):+.5f}] | P(DD<-10%) own bootstrap mean/worst seed: MAIN {p0[0]:.4f}/{p0[1]:.4f}, +5 {p5[0]:.4f}/{p5[1]:.4f}"
              f" | DEF dd {ok_dd} floor {ok_floor} tail {ok_p}")
    return {"r0": r0, "r5": r5, "s0": s0, "s5": s5, "crisis": c, "p0": p0, "p5": p5, "pass": ok_dd and ok_floor and ok_p}


old = json.loads((WT / "results/research/portfolio/fund_products_20260930/report/a6d.json").read_text(encoding="utf-8"))
st = old["defensive"]["rich"]
print("stored rich slot:", st["name"], "cagr", st["q"]["cagr"], "dd", st["q"]["dd"], "p10", st["tails"]["p10"], "weights", st["weights"])

print("\n=== launch and parity")
launch = full("launch C_L|c0.10", with_cash(C_L, 0.10))
par = full("parity RICH S9 g0.25 c0.20", with_cash(blend((0.75, C_L), (0.25, S9)), 0.20))
print("   parity CAGR diff vs stored a6d (stored uses its own annualisation):", par["s0"]["cagr"] - st["q"]["cagr"], "| DD diff", par["s0"]["dd"] - st["q"]["dd"])

print("\n=== GR1-based more-return: the pick and its neighbours")
res = {}
for gs, c in ((0.45, 0.30), (0.35, 0.25), (0.15, 0.10), (0.50, 0.30), (0.45, 0.25), (0.50, 0.35), (0.40, 0.25), (0.40, 0.30), (0.30, 0.20), (0.25, 0.15), (0.20, 0.10), (0.25, 0.20), (0.10, 0.10)):
    w = with_cash(blend((1 - gs, C_L), (gs, GR1)), c)
    res[(gs, c)] = full(f"g{gs:.2f} c{c:.2f}", w)
pick_w = with_cash(blend((0.55, C_L), (0.45, GR1)), 0.30)
print("pick weights", {k: round(v, 4) for k, v in pick_w.items()})
# risk share of the growth slice inside the pick (covariance of weighted pod returns with the book, target weights)
cols = list(pick_w)
Rm = main.loc[LONG:END, cols]
contrib = Rm * np.array([pick_w[c] for c in cols])
bookr = contrib.sum(axis=1)
share = {c: float(np.cov(contrib[c], bookr)[0, 1] / bookr.var(ddof=1)) for c in cols}
g_cols = [c for c in cols if c in GR1]
print("risk share: growth slice %.3f | defensive pair %.3f | cash %.3f ; capital: growth %.3f pair %.3f cash %.3f ; Defense First capital (taa3x + btal_qqq) %.3f"
      % (sum(share[c] for c in g_cols), share["core5"] + share["btal_qqq"], share["tbill"], sum(pick_w[c] for c in g_cols), pick_w["core5"] + pick_w["btal_qqq"], pick_w["tbill"],
         pick_w["taa3x"] + pick_w["btal_qqq"]))
w15 = with_cash(blend((0.85, C_L), (0.15, GR1)), 0.10)
cols = list(w15)
contrib = main.loc[LONG:END, cols] * np.array([w15[c] for c in cols])
bookr = contrib.sum(axis=1)
print("g0.15 c0.10 risk share growth slice %.3f" % sum(float(np.cov(contrib[c], bookr)[0, 1] / bookr.var(ddof=1)) for c in cols if c in GR1))

print("\n=== gated upgrade: 60/40 at 90% + MR capsule 10%")
for c in (0.0, 0.05, 0.10):
    w = with_cash(blend((0.9, C_L), (0.1, MR)), c)
    g = full(f"C_L 90 + MR 10 | cash {c:.2f}", w)
    if c == 0.05:
        r10 = g["r0"] + 2 * (g["r5"] - g["r0"])
        l10 = launch["r0"] + 2 * (launch["r5"] - launch["r0"])
        print("   paired share (xs, cagr) vs launch: MAIN", paired_share(g["r0"], launch["r0"]), "+5", paired_share(g["r5"], launch["r5"]), "+10", paired_share(r10, l10))
        ex_c, ex_l = stats(book(exact, w, EXACT)), stats(book(exact, with_cash(C_L, 0.10), EXACT))
        cut = pd.Timestamp("2017-06-30")
        h = lambda r: (stats(r[r.index <= cut])["xs"], stats(r[r.index > cut])["xs"])  # noqa: E731
        print("   EXACT xs", ex_c["xs"], "vs", ex_l["xs"], "| +5 xs", g["s5"]["xs"], "vs", launch["s5"]["xs"], "| halves", h(g["r0"]), "vs", h(launch["r0"]),
              "| breach", g["p0"][0], "vs", launch["p0"][0])
w = with_cash(blend((0.9, C_L), (0.1, {"hpi_vote": 1.0})), 0.10)
g = full("stored row C_L 90 + hpi_vote 10 | cash 0.10", w)
r10 = g["r0"] + 2 * (g["r5"] - g["r0"])
l10 = launch["r0"] + 2 * (launch["r5"] - launch["r0"])
print("   paired share vs launch: MAIN", paired_share(g["r0"], launch["r0"]), "+5", paired_share(g["r5"], launch["r5"]), "+10", paired_share(r10, l10))
