"""Self-calibrating DV2 stress gate (SPEC_FROZEN.md). Usage: python run_selfcal.py"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "mr_gate_definition_20261003"))
import run_gates as rg  # noqa: E402

rs, q, ll, rp, npc, tbc = rg.rs, rg.q, rg.ll, rg.rp, rg.npc, rg.tbc
OUT = q.REPO / "results/research/mr_gate_selfcal_20261003"
HOLD = ("1995-01-03", "1999-12-31")
BLK = ("G-P1", "G-P2", "G-P3")
GRID = [(t, m) for t in (18, 20, 22, 25) for m in (5, 10, 15, 20)]


def gate_mem(vix: pd.Series, thr, mem):
    """C3 semantics with (possibly time-varying) threshold and memory; memory is fixed at each opening."""
    v = vix.to_numpy()
    th = np.broadcast_to(np.asarray(thr, dtype=float), v.shape) if np.ndim(thr) == 0 else np.asarray(thr, dtype=float)
    mm = np.broadcast_to(np.asarray(mem, dtype=float), v.shape) if np.ndim(mem) == 0 else np.asarray(mem, dtype=float)
    out = np.zeros(len(v), dtype=bool)
    state, held, need = False, 0, 0
    for i in range(len(v)):
        if not (np.isfinite(v[i]) and np.isfinite(th[i]) and np.isfinite(mm[i])):
            out[i] = state
            continue
        c = v[i] > th[i]
        if not state and c:
            state, held, need = True, 0, int(mm[i])
        elif state:
            held += 1
            if not c and held >= need:
                state = False
        out[i] = state
    return out


def selfcal_params(vix: pd.Series):
    x = vix.dropna()
    thr_mean = x.expanding(min_periods=500).mean()
    thr_med = x.expanding(min_periods=500).median()
    lx = np.log(x)
    dev = lx - lx.expanding(min_periods=500).mean()
    a, b = dev.shift(1), dev
    ok = a.notna() & b.notna()
    a0, b0 = a.where(ok, 0.0), b.where(ok, 0.0)
    n = ok.astype(float).cumsum()
    sa, sb, sab, saa = a0.cumsum(), b0.cumsum(), (a0 * b0).cumsum(), (a0 * a0).cumsum()
    phi = (sab - sa * sb / n) / (saa - sa * sa / n)
    phi = phi.where(n >= 500)
    hl = (np.log(0.5) / np.log(phi)).round().clip(5, 40)
    return thr_mean.reindex(vix.index), thr_med.reindex(vix.index), hl.reindex(vix.index)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    p = rp.Panel("sp500")
    vix = rg.inputs(p)[0]
    rate = rs.cash_rate(p.dates)
    DV = ll.dv2_masks(p, rp.Rule())
    taa, L, Ls, bil = tbc.load_taa_ser(), npc.load_l_ret_ser("engine"), npc.load_l_ret_ser("stress"), npc.load_bil_ret_ser()
    thr_mean, thr_med, hl = selfcal_params(vix)
    G = {"C3_20_10": gate_mem(vix, 20.0, 10), "R_20_15": gate_mem(vix, 20.0, 15),
         "S1_mean_halflife": gate_mem(vix, thr_mean.to_numpy(), hl.to_numpy()),
         "S2_median_halflife": gate_mem(vix, thr_med.to_numpy(), hl.to_numpy()),
         "ANY_OFF": ((vix > 20) | rg.inputs(p)[3]).fillna(False).to_numpy()}
    # S3 walk-forward on standalone DV2 Sharpe
    grid_gate = {gm: gate_mem(vix, float(gm[0]), gm[1]) for gm in GRID}
    grid_ret = {gm: rs.swept(ll.run(p, rg.spec(DV, "off", g), "1991-01-02", rs.END), rate) for gm, g in grid_gate.items()}
    picks, s3 = {}, np.zeros(len(p.dates), dtype=bool)
    for yr in range(1995, int(rs.END[:4]) + 1):
        past = {gm: r.loc[: f"{yr - 1}-12-31"] for gm, r in grid_ret.items()}
        best = max(GRID, key=lambda gm: past[gm].mean() / past[gm].std())
        picks[yr] = best
        m = p.dates.year == yr
        s3[m] = grid_gate[best][m]
    G["S3_walkforward"] = s3
    rep = {"picks": {str(k): list(v) for k, v in picks.items()},
           "paths": {"threshold_mean": {str(y): float(thr_mean[thr_mean.index.year == y].iloc[-1]) for y in range(1993, 2027) if (thr_mean.index.year == y).any()},
                     "threshold_median": {str(y): float(thr_med[thr_med.index.year == y].iloc[-1]) for y in range(1993, 2027) if (thr_med.index.year == y).any()},
                     "half_life": {str(y): float(hl[hl.index.year == y].iloc[-1]) for y in range(1993, 2027) if (hl.index.year == y).any()}}}
    u = rs.swept(ll.run(p, ll.Spec("u", *DV), rs.MAIN0, rs.END), rate)
    us = rs.swept(ll.run(p, ll.Spec("u", *DV, slip_extra_bps=5.0), "2007-01-03", rs.END), rate)
    rep["controls"] = {"ungated": npc.candidate_book_blocks(taa, L, u), "ungated_stress": npc.candidate_book_blocks(taa, Ls, us),
                       "C_BIL": npc.candidate_book_blocks(taa, L, bil), "C_BIL_stress": npc.candidate_book_blocks(taa, Ls, bil)}
    rows = {}
    for n, g in G.items():
        r = rs.swept(ll.run(p, rg.spec(DV, "off", g), rs.MAIN0, rs.END), rate)
        rst = rs.swept(ll.run(p, rg.spec(DV, "off", g, 5.0), "2007-01-03", rs.END), rate)
        sw, op = rg.switches("off", g, p.dates)
        d = {"book": npc.candidate_book_blocks(taa, L, r), "book_stress": npc.candidate_book_blocks(taa, Ls, rst), "standalone": rs.stats(r),
             "holdout": rs.stats(rs.swept(ll.run(p, rg.spec(DV, "off", g), *HOLD), rate)), "switches_per_year": sw, "open_share": op}
        rep[n] = d
        rows[n] = {"G-FULL": d["book"]["G-FULL"]["sharpe"], "G-LONG": d["book"]["G-LONG"]["sharpe"], **{b: d["book"][b]["sharpe"] for b in BLK},
                   "S_FULL": d["book_stress"]["G-FULL"]["sharpe"], "S_P3": d["book_stress"]["G-P3"]["sharpe"], "pod_sh": d["standalone"]["sharpe"],
                   "pod_cagr": d["standalone"]["cagr"], "hold95_99": d["holdout"]["sharpe"], "sw_yr": sw, "open": op}
        print(n, {k: round(x, 3) for k, x in rows[n].items()}, flush=True)
    ref = rep["C3_20_10"]
    dec = {}
    for n in ("S1_mean_halflife", "S2_median_halflife", "S3_walkforward"):
        d = rep[n]
        ok = (all(d["book"][b]["sharpe"] >= ref["book"][b]["sharpe"] - 0.02 for b in ("G-FULL", "G-LONG"))
              and all(d["book"][b]["sharpe"] >= ref["book"][b]["sharpe"] - 0.03 for b in BLK)
              and all(d["book_stress"][b]["sharpe"] >= ref["book_stress"][b]["sharpe"] - 0.02 for b in ("G-FULL", "G-LONG"))
              and all(d["book_stress"][b]["sharpe"] >= ref["book_stress"][b]["sharpe"] - 0.03 for b in BLK)
              and d["holdout"]["sharpe"] >= ref["holdout"]["sharpe"] - 0.05)
        dec[n] = bool(ok)
    rep["decision"] = dec
    (OUT / "results.json").write_text(json.dumps(rep, indent=2, default=float), encoding="utf-8")
    pd.DataFrame(rows).T.to_csv(OUT / "summary.csv")
    print("decision", dec)
    print("S3 picks", rep["picks"])
    print("paths", {k: {y: round(v, 1) for y, v in d.items() if int(y) % 4 == 0 or y == "2026"} for k, d in rep["paths"].items()})


if __name__ == "__main__":
    main()
