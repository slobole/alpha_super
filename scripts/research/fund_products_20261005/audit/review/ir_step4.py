"""Step 4: own stationary bootstrap. P(max DD < L) for GR1 / S9 (+GR2, GR3) and paired share GR1 xs > S9 xs."""
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
bil = main["BIL"].to_numpy()
study = json.loads((c.REPORT / "study.json").read_text(encoding="utf-8"))
cache = json.loads((c.REPORT / "tail_cache.json").read_text(encoding="utf-8"))

names = ["GR1", "GR2", "GR3", "S9 incumbent launch"]
R = {k: c.book_returns(main, c.BOOKS[k]).to_numpy() for k in names}
R5 = {k: c.book_returns(plus5, c.BOOKS[k]).to_numpy() for k in names}
n = len(bil)
LIMITS = [-0.10, -0.15, -0.17, -0.20, -0.22, -0.25, -0.27, -0.30, -0.35]


def cache_rows(weights: dict, frame: str):
    target = sorted((k if k != "BIL" else "tbill", round(v, 12)) for k, v in weights.items())
    for key, val in cache.items():
        k = json.loads(key)
        if k[2] != frame:
            continue
        got = sorted((a, round(b, 12)) for a, b in k[1])
        if got == target:
            return val
    return None


def path_dd(sample: np.ndarray, with_base: bool) -> np.ndarray:
    nav = np.cumprod(1.0 + sample, axis=1)
    peak = np.maximum.accumulate(nav, axis=1)
    if with_base:
        peak = np.maximum(peak, 1.0)
    return (nav / peak - 1.0).min(axis=1)


def xs_paths(sample: np.ndarray, bil_s: np.ndarray) -> np.ndarray:
    ex = sample - bil_s
    return ex.mean(axis=1) / ex.std(axis=1, ddof=1) * np.sqrt(252.0)


def cagr_paths(sample: np.ndarray) -> np.ndarray:
    return np.exp(np.log1p(sample).sum(axis=1) * 252.0 / sample.shape[1]) - 1.0


res = {k: {"base": [], "nobase": []} for k in names}
res5 = {k: [] for k in names}
share_xs, share_cagr, gap_xs = [], [], []
seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 10
for s in range(seeds):
    idx = c.sb_index(n, 2000, 63.0, 20260929 + s)
    bil_s = bil[idx]
    xs = {}
    cg = {}
    for k in names:
        smp = R[k][idx]
        dd_b = path_dd(smp, True)
        dd_n = path_dd(smp, False)
        res[k]["base"].append({str(L): float((dd_b < L).mean()) for L in LIMITS})
        res[k]["nobase"].append({str(L): float((dd_n < L).mean()) for L in LIMITS})
        xs[k] = xs_paths(smp, bil_s)
        cg[k] = cagr_paths(smp)
        res5[k].append(float((path_dd(R5[k][idx], False) < -0.20).mean()))
    share_xs.append(xs["GR1"] > xs["S9 incumbent launch"])
    share_cagr.append(cg["GR1"] > cg["S9 incumbent launch"])
    gap_xs.append(xs["GR1"] - xs["S9 incumbent launch"])
    print(f"seed {s}: GR1 p20 base {res['GR1']['base'][-1]['-0.2']:.4f} nobase {res['GR1']['nobase'][-1]['-0.2']:.4f} | "
          f"S9 p20 base {res['S9 incumbent launch']['base'][-1]['-0.2']:.4f} nobase {res['S9 incumbent launch']['nobase'][-1]['-0.2']:.4f} | "
          f"share xs GR1>S9 {share_xs[-1].mean():.4f}  share cagr {share_cagr[-1].mean():.4f}", flush=True)

print()
for k in names:
    cr = cache_rows(c.BOOKS[k], "main")
    print(f"=== {k}: cache rows found: {cr is not None}")
    for L in LIMITS:
        mine_b = [r[str(L)] for r in res[k]["base"]]
        mine_n = [r[str(L)] for r in res[k]["nobase"]]
        theirs = [r[str(L)] for r in cr][:seeds] if cr else None
        md_b = max(abs(a - b) for a, b in zip(mine_b, theirs)) if theirs else float("nan")
        md_n = max(abs(a - b) for a, b in zip(mine_n, theirs)) if theirs else float("nan")
        print(f"  L {L:+.2f}: mine mean(base) {np.mean(mine_b):.5f} mean(nobase) {np.mean(mine_n):.5f}  cache mean {np.mean(theirs):.5f}"
              f"  max|per-seed diff| base {md_b:.5f} nobase {md_n:.5f}  worst seed mine {max(mine_n):.4f} cache {max(theirs):.4f}")
    t = study["books"][k].get("tails")
    if t:
        print("  study tails p20", t["p20"], "p20_max", t["p20_max"])
    t5 = study["books"][k].get("tails_plus5")
    if t5:
        print(f"  +5bps p20: mine mean {np.mean(res5[k]):.5f} worst {max(res5[k]):.4f} | study {t5['p20']} worst {t5['p20_max']}")

sx = np.concatenate(share_xs)
sc_ = np.concatenate(share_cagr)
gx = np.concatenate(gap_xs)
print()
print(f"paired share GR1 xs > S9 xs: seed0 {share_xs[0].mean():.4f}; pooled {sx.mean():.4f} ({len(sx)} paths)")
print(f"paired share GR1 CAGR > S9 CAGR: seed0 {share_cagr[0].mean():.4f}; pooled {sc_.mean():.4f}")
print("gap xs p5/50/95:", np.percentile(gx, [5, 50, 95]).round(6))
for ch in study["challenges_reverse"]:
    if ch["challenger"] == "GR1" and ch["default"].startswith("S9"):
        print("study challenges_reverse:", {k: ch[k] for k in ("share_xs", "share_cagr", "gap_xs_p5_50_95", "paths")})
for ch in study["challenges"]:
    if ch["challenger"].startswith("S9"):
        print("study challenges:", {k: ch[k] for k in ("share_xs", "share_cagr", "gap_xs_p5_50_95", "paths")})

json.dump({"res": res, "res5": res5, "share_xs_pooled": float(sx.mean()), "share_xs_seed0": float(share_xs[0].mean()),
           "share_cagr_pooled": float(sc_.mean())}, open(c.OUT / "step4_bootstrap.json", "w"), indent=1)
