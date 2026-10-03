"""Check A: long rolling VIX mean as the gate threshold (SPEC_FROZEN.md)."""
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "mr_gate_selfcal_20261003"))
import run_selfcal as sc  # noqa: E402
rg, rs, ll, rp, npc, tbc = sc.rg, sc.rs, sc.ll, sc.rp, sc.npc, sc.tbc
OUT = rg.q.REPO / "results/research/mr_gate_final_checks_20261003"
BLK = ("G-P1", "G-P2", "G-P3")


def main():
    p = rp.Panel("sp500")
    vix = rg.inputs(p)[0]
    rate = rs.cash_rate(p.dates)
    DV = ll.dv2_masks(p, rp.Rule())
    taa, L, Ls = tbc.load_taa_ser(), npc.load_l_ret_ser("engine"), npc.load_l_ret_ser("stress")
    x = vix.dropna()
    thr = {"expanding": x.expanding(min_periods=500).mean()}
    for w in (10, 15, 20):
        thr[f"rolling_{w}y"] = x.rolling(w * 252, min_periods=500).mean()
    rep, rows = {}, {}
    for n, t in thr.items():
        t = t.reindex(vix.index)
        g = sc.gate_mem(vix, t.to_numpy(), 15)
        r = rs.swept(ll.run(p, rg.spec(DV, "off", g), rs.MAIN0, rs.END), rate)
        rst = rs.swept(ll.run(p, rg.spec(DV, "off", g, 5.0), "2007-01-03", rs.END), rate)
        h = rs.stats(rs.swept(ll.run(p, rg.spec(DV, "off", g), "1995-01-03", "1999-12-31"), rate))
        b, bs = npc.candidate_book_blocks(taa, L, r), npc.candidate_book_blocks(taa, Ls, rst)
        sw, op = rg.switches("off", g, p.dates)
        rep[n] = {"book": b, "book_stress": bs, "standalone": rs.stats(r), "holdout": h, "switches": sw, "open": op,
                  "threshold_by_year": {str(y): float(t[t.index.year == y].iloc[-1]) for y in range(2000, 2027, 2) if (t.index.year == y).any()}}
        rows[n] = {"G-FULL": b["G-FULL"]["sharpe"], "G-LONG": b["G-LONG"]["sharpe"], **{k: b[k]["sharpe"] for k in BLK},
                   "S_FULL": bs["G-FULL"]["sharpe"], "S_P3": bs["G-P3"]["sharpe"], "pod_sh": rs.stats(r)["sharpe"], "hold": h["sharpe"], "sw": sw, "open": op}
        print(n, {k: round(v, 3) for k, v in rows[n].items()}, "thr", {y: round(v, 1) for y, v in rep[n]["threshold_by_year"].items()}, flush=True)
    ref = rep["expanding"]
    names = ["rolling_10y", "rolling_15y", "rolling_20y"]
    dec = {}
    for i, n in enumerate(names):
        d = rep[n]
        nb = [names[j] for j in (i - 1, i + 1) if 0 <= j < len(names)]
        dec[n] = bool(all(d["book"][k]["sharpe"] >= ref["book"][k]["sharpe"] for k in ("G-FULL", "G-LONG"))
                      and all(d["book"][k]["sharpe"] >= ref["book"][k]["sharpe"] - 0.03 for k in BLK)
                      and all(d["book_stress"][k]["sharpe"] >= ref["book_stress"][k]["sharpe"] for k in ("G-FULL", "G-LONG"))
                      and all(d["book_stress"][k]["sharpe"] >= ref["book_stress"][k]["sharpe"] - 0.03 for k in BLK)
                      and all(rep[m]["book"]["G-FULL"]["sharpe"] >= ref["book"]["G-FULL"]["sharpe"] - 0.03 for m in nb))
    rep["decision"] = dec
    (OUT / "check_a.json").write_text(json.dumps(rep, indent=2, default=float), encoding="utf-8")
    print("decision", dec)


if __name__ == "__main__":
    main()
