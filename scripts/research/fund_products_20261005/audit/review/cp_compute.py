"""Compliance reviewer: independent recomputation of the headline rows and of plan items the study left out.

Read-only: uses g_lib.Lab for the frames and bootstrap indices, never calls lab.tails / run_tails / g.ledger
(those write to the study's tail cache and ledger). Output: audit/review/compliance/cp_compute.json.
"""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd

WT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(WT / "scripts" / "research" / "fund_products_20261005"))
import g_lib as g
from g_lib import LONG_START, END, TBILL, Book, lib
import battery as bt

OUTD = WT / "results/research/portfolio/fund_products_20261005/audit/review/compliance"
lab = g.Lab()
lab._disk_path = OUTD / "never_written_tail_cache.json"     # safety: the study cache can not be touched
frame, rf = lab.frame, lab.rf
out = {}

# 1. own book model: pods compound independently, reset to target at the first session of each calendar year
def own_book(fr, w, start=LONG_START):
    win = fr.loc[start:END, list(w)]
    assert not win.isna().any().any()
    parts = []
    level = 1.0
    for _, grp in win.groupby(win.index.year):
        pods = (1.0 + grp).cumprod() * pd.Series(w)
        tot = pods.sum(axis=1) * level
        parts.append(tot)
        level = float(tot.iloc[-1])
    nav = pd.concat(parts)
    return nav.pct_change().fillna(nav.iloc[0] - 1.0)

def own_stats(r):
    x = r.to_numpy(); f = rf.reindex(r.index).to_numpy()
    nav = np.cumprod(1 + x)
    dd = (np.r_[1.0, nav] / np.maximum.accumulate(np.r_[1.0, nav]) - 1).min()
    e = x - f
    return {"cagr": float(nav[-1] ** (252 / len(x)) - 1), "xs": float(e.mean() / e.std(ddof=1) * np.sqrt(252)), "dd": float(dd), "n": len(x)}

def own_breach(r, limit, block=63.0):
    x = r.to_numpy()
    ps = []
    for s in range(10):
        idx = lib.evaluation.stationary_bootstrap_index_mat(len(x), 2000, block, g.SEED0 + s)   # (paths, days)
        nav = np.cumprod(1.0 + x[idx], axis=1)
        nav = np.concatenate([np.ones((nav.shape[0], 1)), nav], axis=1)
        dd = (nav / np.maximum.accumulate(nav, axis=1) - 1.0).min(axis=1)
        ps.append(float((dd < limit).mean()))
    return {"mean": float(np.mean(ps)), "worst": float(np.max(ps))}

books = {**g.PRODUCTS, "S9": g.INCUMBENT}
f5 = lab.frames["s3_plus_5bps"][0]
out["headline"] = {}
for n, w in books.items():
    r, r5 = own_book(frame, w), own_book(f5, w)
    lim = {"GR1": -0.20, "GR2": -0.25, "GR3": -0.30, "S9": -0.20}[n]
    out["headline"][n] = {"main": own_stats(r), "plus5": own_stats(r5), "breach_main": own_breach(r, lim), "breach_plus5": own_breach(r5, lim), "limit": lim,
                          "breach_m20_main": own_breach(r, -0.20) if lim != -0.20 else None}
    print("HEAD", n, out["headline"][n], flush=True)

# 2. GR3 at the rung it is labelled with (GROWTH PLUS, -25%): edge-margin curve and worst seed, MAIN and +5 bps
curve = {}
for k in np.round(np.arange(1.0, 0.449, -0.05), 2):
    fr = bt.decay_frame(frame, dict.fromkeys(bt.ALL_CAPS, float(k)))
    R = np.column_stack([lab.ret(g.PRODUCTS[n], frame=fr).to_numpy() for n in ("GR2", "GR3")])
    tm = lab.tail_matrix(R, limits=(-0.25, -0.30))
    curve[float(k)] = {"GR2_p25": float(tm[-0.25][:, 0].mean()), "GR3_p25": float(tm[-0.25][:, 1].mean()), "GR3_p30": float(tm[-0.30][:, 1].mean())}
    print("EDGE", k, curve[float(k)], flush=True)
out["edge_curve_m25"] = curve

# 3. S7 under the one-engine-dead scenarios (annual reset; weights from the MAIN frame's capsule history as the plan fixes)
caps = {"TAA": {"taa3x": 1.0}, "MOM": g.MOM, "MR": g.MR}
def capsule_frame(fr):
    return pd.DataFrame({k: g.book_returns(fr, g.blend((1.0, w)), LONG_START) for k, w in caps.items()})
src = capsule_frame(frame)
s7 = {}
for cap in ("TAA", "MOM", "MR"):
    fr = bt.decay_frame(frame, {cap: 0.0})
    r = lib.book_returns(capsule_frame(fr), Book("S7", ("TAA", "MOM", "MR"), "IV"), LONG_START, weight_source=src)
    s7[f"{cap} dead"] = g.stats(r, rf)
    r1 = g.book_returns(fr, g.PRODUCTS["GR1"], LONG_START)
    s7[f"{cap} dead GR1"] = g.stats(r1, rf)
out["s7_dead"] = {k: {"xs": v["xs"], "cagr": v["cagr"]} for k, v in s7.items()}
print("S7", out["s7_dead"], flush=True)

# 4. rolling windows: share above each capsule and S9, 756 and 1260 sessions, excess Sharpe and CAGR (plan 5.6)
def roll(r, n):
    x = r - rf.reindex(r.index)
    xs = (x.rolling(n).mean() / x.rolling(n).std(ddof=1) * np.sqrt(252)).dropna()
    cg = np.expm1(np.log1p(r).rolling(n).sum() * 252 / n).dropna()
    return xs, cg
others = {"TAA 3x": {"taa3x": 1.0}, "TAA 3x 1N": {"taa3x_1n": 1.0}, "MOM": g.MOM, "MR": g.MR, "S9": g.INCUMBENT}
ro = {}
for n in g.PRODUCTS:
    ro[n] = {}
    for win in (756, 1260):
        xs, cg = roll(own_book(frame, g.PRODUCTS[n]), win)
        ro[n][win] = {}
        for o, w in others.items():
            oxs, ocg = roll(lab.ret(g.blend((1.0, w))), win)
            ro[n][win][o] = {"xs_share": float((xs > oxs.reindex(xs.index)).mean()), "cagr_share": float((cg > ocg.reindex(cg.index)).mean())}
out["rolling_share"] = ro
print("ROLL", json.dumps(ro)[:1500], flush=True)

# 5. gap table by the plan's wording: Nasdaq look-through at mean / P90 / peak plus MR at its mean / peak stock weight
X = json.loads((g.OUT / "exposure.json").read_text(encoding="utf-8"))
gp = {}
for n in g.PRODUCTS:
    b = X["books"][n]
    w_mr = g.PRODUCTS[n]["dv2_g"] + g.PRODUCTS[n]["hpi_g"]
    nq, mr = b["nasdaq_lookthrough"], X["mr_stock_weight"]
    gp[n] = {"joint_peak": b["equity_exposure"]["max"], "sum_of_peaks": nq["max"] + w_mr * mr["max"], "joint_p90": b["equity_exposure"]["p90"],
             "nq_p90_plus_mr_peak": nq["p90"] + w_mr * mr["max"], "mean": nq["mean"] + w_mr * mr["mean"]}
out["gap_exposure"] = gp
print("GAP", gp, flush=True)

(OUTD / "cp_compute.json").write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
print("done")
