"""VIX-trend gates (KAMA / SMA band) for DV2 (SPEC_FROZEN.md). Usage: python run_kama.py"""
from __future__ import annotations
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "mr_gate_selfcal_20261003"))
import run_selfcal as sc  # noqa: E402
rg, rs, ll, rp, npc, tbc = sc.rg, sc.rs, sc.ll, sc.rp, sc.npc, sc.tbc
q = rg.q
OUT = q.REPO / "results/research/mr_gate_kama_20261003"
HOLD, BLK = ("1995-01-03", "1999-12-31"), ("G-P1", "G-P2", "G-P3")


def kama(v: pd.Series, n=10, fast=2, slow=30):
    x = v.to_numpy()
    out = np.full(len(x), np.nan)
    fa, sa = 2 / (fast + 1), 2 / (slow + 1)
    for i in range(n, len(x)):
        if not np.isfinite(x[i]):
            out[i] = out[i - 1]
            continue
        w = x[i - n:i + 1]
        if not np.all(np.isfinite(w)):
            out[i] = out[i - 1] if np.isfinite(out[i - 1]) else x[i]
            continue
        vol = np.abs(np.diff(w)).sum()
        er = abs(w[-1] - w[0]) / vol if vol > 0 else 0.0
        s = (er * (fa - sa) + sa) ** 2
        prev = out[i - 1] if np.isfinite(out[i - 1]) else x[i - 1]
        out[i] = prev + s * (x[i] - prev)
    return pd.Series(out, index=v.index)


def band(v, upper, lower):
    o = np.zeros(len(v), dtype=bool)
    state = False
    for i, (x, u, l) in enumerate(zip(v.to_numpy(), upper.to_numpy(), lower.to_numpy())):
        if np.isfinite(x) and np.isfinite(u) and np.isfinite(l):
            if not state and x > u:
                state = True
            elif state and x < l:
                state = False
        o[i] = state
    return o


def main():
    p = rp.Panel("sp500")
    vix = rg.inputs(p)[0]
    rate = rs.cash_rate(p.dates)
    DV = ll.dv2_masks(p, rp.Rule())
    taa, L, Ls = tbc.load_taa_ser(), npc.load_l_ret_ser("engine"), npc.load_l_ret_ser("stress")
    k10, k20, s10, s20 = kama(vix, 10), kama(vix, 20), vix.rolling(10).mean(), vix.rolling(20).mean()
    thr_mean = sc.selfcal_params(vix)[0]
    G = {"C3_ref": sc.gate_mem(vix, 20.0, 10)}
    for b in (0.0, 0.05, 0.10):
        G[f"K10_b{int(b*100)}"] = band(vix, k10 * (1 + b), k10)
        G[f"S10_b{int(b*100)}"] = band(vix, s10 * (1 + b), s10)
    G["K20_b5"] = band(vix, k20 * 1.05, k20)
    G["S20_b5"] = band(vix, s20 * 1.05, s20)
    G["K10_b5_AND_level"] = G["K10_b5"] & (vix > thr_mean).fillna(False).to_numpy()
    G["K10_b5_OR_C3"] = G["K10_b5"] | G["C3_ref"]
    rep, rows = {}, {}
    for n, g in G.items():
        r = rs.swept(ll.run(p, rg.spec(DV, "off", g), rs.MAIN0, rs.END), rate)
        rst = rs.swept(ll.run(p, rg.spec(DV, "off", g, 5.0), "2007-01-03", rs.END), rate)
        sw, op = rg.switches("off", g, p.dates)
        d = {"book": npc.candidate_book_blocks(taa, L, r), "book_stress": npc.candidate_book_blocks(taa, Ls, rst), "standalone": rs.stats(r),
             "holdout": rs.stats(rs.swept(ll.run(p, rg.spec(DV, "off", g), *HOLD), rate)), "switches_per_year": sw, "open_share": op}
        # how much of the open time is in calm (VIX <= 20) markets
        on = g[p.dates >= rs.MAIN0]
        d["open_days_with_vix_le_20"] = float(((vix.to_numpy()[p.dates >= rs.MAIN0] <= 20) & on).sum() / max(on.sum(), 1))
        rep[n] = d
        rows[n] = {"G-FULL": d["book"]["G-FULL"]["sharpe"], "G-LONG": d["book"]["G-LONG"]["sharpe"], **{b: d["book"][b]["sharpe"] for b in BLK},
                   "S_FULL": d["book_stress"]["G-FULL"]["sharpe"], "S_P3": d["book_stress"]["G-P3"]["sharpe"], "pod_sh": d["standalone"]["sharpe"],
                   "hold95_99": d["holdout"]["sharpe"], "sw_yr": sw, "open": op, "open_calm": d["open_days_with_vix_le_20"]}
        print(n, {k: round(x, 3) for k, x in rows[n].items()}, flush=True)
    ref = rep["C3_ref"]
    nb = {"K10_b0": ["K10_b5"], "K10_b5": ["K10_b0", "K10_b10", "K20_b5"], "K10_b10": ["K10_b5"], "K20_b5": ["K10_b5"],
          "S10_b0": ["S10_b5"], "S10_b5": ["S10_b0", "S10_b10", "S20_b5"], "S10_b10": ["S10_b5"], "S20_b5": ["S10_b5"],
          "K10_b5_AND_level": [], "K10_b5_OR_C3": []}
    dec = {}
    for n, nbrs in nb.items():
        d = rep[n]
        ok = (all(d["book"][b]["sharpe"] >= ref["book"][b]["sharpe"] for b in ("G-FULL", "G-LONG"))
              and all(d["book"][b]["sharpe"] >= ref["book"][b]["sharpe"] - 0.03 for b in BLK)
              and all(d["book_stress"][b]["sharpe"] >= ref["book_stress"][b]["sharpe"] for b in ("G-FULL", "G-LONG"))
              and all(d["book_stress"][b]["sharpe"] >= ref["book_stress"][b]["sharpe"] - 0.03 for b in BLK)
              and all(rep[x]["book"]["G-FULL"]["sharpe"] >= ref["book"]["G-FULL"]["sharpe"] - 0.03 for x in nbrs)
              and d["holdout"]["sharpe"] >= ref["holdout"]["sharpe"] - 0.05)
        dec[n] = bool(ok)
    rep["decision"] = dec
    (OUT / "results.json").write_text(json.dumps(rep, indent=2, default=float), encoding="utf-8")
    pd.DataFrame(rows).T.to_csv(OUT / "summary.csv")
    print("decision", dec)


if __name__ == "__main__":
    main()
