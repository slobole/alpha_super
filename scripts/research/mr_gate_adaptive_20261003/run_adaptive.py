"""Adaptive stress measures for the DV2 gate (SPEC_FROZEN.md). Usage: python run_adaptive.py"""

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
OUT = q.REPO / "results/research/mr_gate_adaptive_20261003"
HOLD = ("1995-01-03", "1999-12-31")
BLK = ("G-P1", "G-P2", "G-P3")


def memory(cond: pd.Series, min_hold=10):
    """C3 semantics on any boolean condition: open on the first True; close on the first False after >= min_hold."""
    out = np.zeros(len(cond), dtype=bool)
    state, held = False, 0
    for i, c in enumerate(cond.fillna(False).to_numpy()):
        if not state and c:
            state, held = True, 0
        elif state:
            held += 1
            if not c and held >= min_hold:
                state = False
        out[i] = state
    return out


def measures(p):
    d = pd.DatetimeIndex(np.load(rs.ETFX / "dates.npy"))
    vix = pd.Series(np.load(rs.ETFX / "vix_close.npy"), index=d).reindex(p.dates).ffill(limit=3)
    spx = pd.Series(np.load(rs.ETFX / "spx_close.npy"), index=d).reindex(p.dates).ffill(limit=3)
    # PR: percentrank of 1/SMA10(VIX) in the last 2000 sessions (low D = VIX high versus its own history)
    D = (1.0 / vix.rolling(10).mean()).rolling(2000, min_periods=1000).rank(pct=True)
    # AV: adaptive volatility (EMA of squared returns, SC from 20-day R^2 of log price vs time)
    lp = np.log(spx)
    tt = pd.Series(np.arange(len(lp), dtype=float), index=lp.index)
    r2 = lp.rolling(20).corr(tt) ** 2
    sc = np.minimum(np.exp(-10.0 * (1.0 - r2)), 0.5)
    r = spx.pct_change()
    v = np.full(len(r), np.nan)
    rv_seed = r.rolling(20).var()
    for i in range(len(r)):
        if not np.isfinite(r.iloc[i]) or not np.isfinite(sc.iloc[i]):
            v[i] = v[i - 1] if i and np.isfinite(v[i - 1]) else np.nan
            continue
        prev = v[i - 1] if i and np.isfinite(v[i - 1]) else rv_seed.iloc[i]
        v[i] = sc.iloc[i] * r.iloc[i] ** 2 + (1 - sc.iloc[i]) * prev if np.isfinite(prev) else np.nan
    av = pd.Series(np.sqrt(252 * v), index=r.index)
    rv = r.rolling(20).std() * np.sqrt(252)
    return vix, D, av, rv


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    p = rp.Panel("sp500")
    vix, D, av, rv = measures(p)
    rate = rs.cash_rate(p.dates)
    DV = ll.dv2_masks(p, rp.Rule())
    taa, L, Ls, bil = tbc.load_taa_ser(), npc.load_l_ret_ser("engine"), npc.load_l_ret_ser("stress"), npc.load_bil_ret_ser()
    V = {"C3_ref": memory(vix > 20), "ANY_OFF": ((vix > 20) | rg.inputs(p)[3]).fillna(False).to_numpy()}
    for dd in (0.3, 0.4, 0.5):
        V[f"PR{dd}"] = (D < dd).fillna(False).to_numpy()
        V[f"PR{dd}_m10"] = memory(D < dd)
    for a in (0.14, 0.16, 0.18, 0.20):
        V[f"AV{int(a*100)}"] = (av > a).fillna(False).to_numpy()
        V[f"AV{int(a*100)}_m10"] = memory(av > a)
        V[f"RV{int(a*100)}_m10"] = memory(rv > a)
    votes = (vix > 20).astype(int) + (D < 0.4).astype(int) + (av > 0.16).astype(int)
    V["VOTE2_m10"] = memory(votes >= 2)
    rep, rows = {}, {}
    u = rs.swept(ll.run(p, ll.Spec("u", *DV), rs.MAIN0, rs.END), rate)
    us = rs.swept(ll.run(p, ll.Spec("u", *DV, slip_extra_bps=5.0), "2007-01-03", rs.END), rate)
    rep["controls"] = {"ungated": npc.candidate_book_blocks(taa, L, u), "ungated_stress": npc.candidate_book_blocks(taa, Ls, us),
                       "ungated_hold": rs.stats(rs.swept(ll.run(p, ll.Spec("u", *DV), *HOLD), rate)),
                       "C_BIL": npc.candidate_book_blocks(taa, L, bil), "C_BIL_stress": npc.candidate_book_blocks(taa, Ls, bil)}
    for n, g in V.items():
        r = rs.swept(ll.run(p, rg.spec(DV, "off", g), rs.MAIN0, rs.END), rate)
        rst = rs.swept(ll.run(p, rg.spec(DV, "off", g, 5.0), "2007-01-03", rs.END), rate)
        sw, op = rg.switches("off", g, p.dates)
        d = {"book": npc.candidate_book_blocks(taa, L, r), "book_stress": npc.candidate_book_blocks(taa, Ls, rst),
             "standalone": rs.stats(r), "holdout": rs.stats(rs.swept(ll.run(p, rg.spec(DV, "off", g), *HOLD), rate)),
             "switches_per_year": sw, "open_share": op}
        rep[n] = d
        rows[n] = {"G-FULL": d["book"]["G-FULL"]["sharpe"], "G-LONG": d["book"]["G-LONG"]["sharpe"], **{b: d["book"][b]["sharpe"] for b in BLK},
                   "S_FULL": d["book_stress"]["G-FULL"]["sharpe"], "S_P3": d["book_stress"]["G-P3"]["sharpe"], "pod_sh": d["standalone"]["sharpe"],
                   "hold95_99": d["holdout"]["sharpe"], "sw_yr": sw, "open": op}
        print(n, {k: round(x, 3) for k, x in rows[n].items()}, flush=True)
    ref = rep["C3_ref"]
    fam = {"PR": ["PR0.3", "PR0.4", "PR0.5"], "PRm": ["PR0.3_m10", "PR0.4_m10", "PR0.5_m10"],
           "AV": ["AV14", "AV16", "AV18", "AV20"], "AVm": ["AV14_m10", "AV16_m10", "AV18_m10", "AV20_m10"],
           "RVm": ["RV14_m10", "RV16_m10", "RV18_m10", "RV20_m10"], "VOTE": ["VOTE2_m10"]}
    dec = {}
    for f, names in fam.items():
        for i, n in enumerate(names):
            d = rep[n]
            nbrs = [names[j] for j in (i - 1, i + 1) if 0 <= j < len(names)]
            c1 = all(d["book"][b]["sharpe"] >= ref["book"][b]["sharpe"] for b in ("G-FULL", "G-LONG")) and all(d["book"][b]["sharpe"] >= ref["book"][b]["sharpe"] - 0.03 for b in BLK)
            c2 = all(d["book_stress"][b]["sharpe"] >= ref["book_stress"][b]["sharpe"] for b in ("G-FULL", "G-LONG")) and all(d["book_stress"][b]["sharpe"] >= ref["book_stress"][b]["sharpe"] - 0.03 for b in BLK)
            c3 = all(rep[x]["book"]["G-FULL"]["sharpe"] >= ref["book"]["G-FULL"]["sharpe"] - 0.03 for x in nbrs)
            c4 = d["holdout"]["sharpe"] >= ref["holdout"]["sharpe"] - 0.05
            dec[n] = {"c1": bool(c1), "c2": bool(c2), "c3": bool(c3), "c4": bool(c4), "pass": bool(c1 and c2 and c3 and c4)}
    rep["decision"] = dec
    rep["av_beats_rv"] = {f"{a}": rep[f"AV{a}_m10"]["book"]["G-FULL"]["sharpe"] > rep[f"RV{a}_m10"]["book"]["G-FULL"]["sharpe"] for a in (14, 16, 18, 20)}
    (OUT / "results.json").write_text(json.dumps(rep, indent=2, default=float), encoding="utf-8")
    pd.DataFrame(rows).T.to_csv(OUT / "summary.csv")
    print("decision", {k: v["pass"] for k, v in dec.items()}, "av_beats_rv", rep["av_beats_rv"])


if __name__ == "__main__":
    main()
