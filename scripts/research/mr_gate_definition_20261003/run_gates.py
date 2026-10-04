"""Gate-definition study for stress-gated DV2 (SPEC_FROZEN.md). Usage: python run_gates.py"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "mr_stress_regime_20261002"))
import run_stress as rs  # noqa: E402

q, ll, rp, npc, tbc = rs.q, rs.ll, rs.rp, rs.npc, rs.tbc
OUT = q.REPO / "results/research/mr_gate_definition_20261003"
BLK = ("G-P1", "G-P2", "G-P3")


def inputs(p):
    d = pd.DatetimeIndex(np.load(rs.ETFX / "dates.npy"))
    vix = pd.Series(np.load(rs.ETFX / "vix_close.npy"), index=d).reindex(p.dates).ffill(limit=3)
    spx = pd.Series(np.load(rs.ETFX / "spx_close.npy"), index=d).reindex(p.dates).ffill(limit=3)
    from data.norgate_loader import load_price_timeseries
    v3 = load_price_timeseries("$VIX3M", start_date_str="2002-01-01")["Close"].astype(float)
    v3.index = pd.to_datetime(v3.index)
    vix3m = v3.reindex(p.dates).ffill(limit=3)
    rv = spx.pct_change().rolling(20).std() * np.sqrt(252)
    mkt = spx < spx.rolling(200, min_periods=200).mean()
    return vix, vix3m, rv, mkt


def hysteresis(vix, up, down, min_hold=0):
    out = np.zeros(len(vix), dtype=bool)
    state, held = False, 0
    for i, v in enumerate(vix.to_numpy()):
        if not np.isfinite(v):
            out[i] = state
            continue
        if not state and v > up:
            state, held = True, 0
        elif state:
            held += 1
            if (down is not None and v < down) or (down is None and v <= up and held >= min_hold):
                state = False
        out[i] = state
    return out


def variants(vix, vix3m, rv, mkt):
    V = {}
    for k in (16, 18, 20, 22, 25):
        V[f"A_VIX{k}"] = ("off", (vix > k).fillna(False).to_numpy())
        V[f"B_ANY{k}"] = ("off", ((vix > k) | mkt).fillna(False).to_numpy())
    V["C1_hyst20_17"] = ("off", hysteresis(vix, 20, 17))
    V["C2_hyst20_18"] = ("off", hysteresis(vix, 20, 18))
    V["C3_open20_min10"] = ("off", hysteresis(vix, 20, None, min_hold=10))
    for a, b in ((14, 25), (16, 22), (12, 30)):
        V[f"D_grade{a}_{b}"] = ("grade", ((vix - a) / (b - a)).clip(0, 1).fillna(0.0).to_numpy())
    ts = (vix / vix3m > 1).fillna(False)
    V["E1_TS"] = ("off", ts.to_numpy())
    V["E2_TS_or_VIX20"] = ("off", (ts | (vix > 20)).fillna(False).to_numpy())
    V["F_RV15"] = ("off", (rv > 0.15).fillna(False).to_numpy())
    return V


def spec(base, kind, g, stress=0.0):
    e, s, x = base
    if kind == "off":
        return ll.Spec("g", e & g[:, None], s, x, slip_extra_bps=stress)
    return ll.Spec("g", e & (g > 0)[:, None], s, x, size_mult=g, slip_extra_bps=stress)


def switches(kind, g, dates):
    on = (g > 0) if kind == "grade" else g
    m = dates >= rs.MAIN0
    o = on[m].astype(int)
    return float(np.abs(np.diff(o)).sum() / (m.sum() / 252)), float(o.mean())


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    p = rp.Panel("sp500")
    vix, vix3m, rv, mkt = inputs(p)
    rate = rs.cash_rate(p.dates)
    fin = q.Features(p)
    turn = p.feat("turn", lambda: np.asarray(p.RAW) * np.asarray(p.V))
    D = ll.dv2_masks(p, rp.Rule())
    Q = (fin.entry, turn, fin.exit)
    taa, L, Ls, bil = tbc.load_taa_ser(), npc.load_l_ret_ser("engine"), npc.load_l_ret_ser("stress"), npc.load_bil_ret_ser()
    V = variants(vix, vix3m, rv, mkt)
    rep = {"controls": {"C_BIL": npc.candidate_book_blocks(taa, L, bil), "C_BIL_stress": npc.candidate_book_blocks(taa, Ls, bil)}}
    u = rs.swept(ll.run(p, ll.Spec("u", *D), rs.MAIN0, rs.END), rate)
    us = rs.swept(ll.run(p, ll.Spec("u", *D, slip_extra_bps=5.0), "2007-01-03", rs.END), rate)
    rep["ungated"] = {"book": npc.candidate_book_blocks(taa, L, u), "book_stress": npc.candidate_book_blocks(taa, Ls, us),
                      "standalone": rs.stats(u), "holdout": rs.stats(rs.swept(ll.run(p, ll.Spec("u", *D), *rs.HOLD_DV2), rate))}
    rows = {}
    for name, (kind, g) in V.items():
        r = rs.swept(ll.run(p, spec(D, kind, g), rs.MAIN0, rs.END), rate)
        rst = rs.swept(ll.run(p, spec(D, kind, g, 5.0), "2007-01-03", rs.END), rate)
        sw, op = switches(kind, g, p.dates)
        d = {"book": npc.candidate_book_blocks(taa, L, r), "book_stress": npc.candidate_book_blocks(taa, Ls, rst),
             "standalone": rs.stats(r.loc["2002-01-02":] if name.startswith("E") else r), "switches_per_year": sw, "open_share": op}
        if not name.startswith("E"):
            d["holdout"] = rs.stats(rs.swept(ll.run(p, spec(D, kind, g), *rs.HOLD_DV2), rate))
        rep[name] = d
        rows[name] = {"G-FULL": d["book"]["G-FULL"]["sharpe"], "G-LONG": d["book"]["G-LONG"]["sharpe"], **{b: d["book"][b]["sharpe"] for b in BLK},
                      "S_FULL": d["book_stress"]["G-FULL"]["sharpe"], "S_P3": d["book_stress"]["G-P3"]["sharpe"],
                      "pod_sh": d["standalone"]["sharpe"], "pod_cagr": d["standalone"]["cagr"], "hold": d.get("holdout", {}).get("sharpe", np.nan),
                      "sw_yr": sw, "open": op}
        print(name, {k: round(v, 3) for k, v in rows[name].items()}, flush=True)
    # ---- decision rule
    ref = rep["B_ANY20"]
    nb = {"A_VIX16": ["A_VIX18"], "A_VIX18": ["A_VIX16", "A_VIX20"], "A_VIX20": ["A_VIX18", "A_VIX22"], "A_VIX22": ["A_VIX20", "A_VIX25"], "A_VIX25": ["A_VIX22"],
          "B_ANY16": ["B_ANY18"], "B_ANY18": ["B_ANY16", "B_ANY20"], "B_ANY22": ["B_ANY20", "B_ANY25"], "B_ANY25": ["B_ANY22"],
          "C1_hyst20_17": ["C2_hyst20_18"], "C2_hyst20_18": ["C1_hyst20_17"], "C3_open20_min10": [],
          "D_grade14_25": ["D_grade16_22", "D_grade12_30"], "D_grade16_22": ["D_grade14_25"], "D_grade12_30": ["D_grade14_25"],
          "E1_TS": ["E2_TS_or_VIX20"], "E2_TS_or_VIX20": ["E1_TS"], "F_RV15": []}
    dec = {}
    for name, nbrs in nb.items():
        d = rep[name]
        c1 = all(d["book"][b]["sharpe"] >= ref["book"][b]["sharpe"] for b in ("G-FULL", "G-LONG")) and all(d["book"][b]["sharpe"] >= ref["book"][b]["sharpe"] - 0.03 for b in BLK)
        c2 = all(d["book_stress"][b]["sharpe"] >= ref["book_stress"][b]["sharpe"] for b in ("G-FULL", "G-LONG")) and all(d["book_stress"][b]["sharpe"] >= ref["book_stress"][b]["sharpe"] - 0.03 for b in BLK)
        c3 = all(rep[n]["book"]["G-FULL"]["sharpe"] >= ref["book"]["G-FULL"]["sharpe"] - 0.03 for n in nbrs)
        dec[name] = {"c1": bool(c1), "c2": bool(c2), "c3_neighbours": bool(c3), "pass": bool(c1 and c2 and c3)}
    rep["decision"] = dec
    ug = rep["ungated"]["book"]
    rep["plateau"] = {fam: all(rep[f"{fam}{k}"]["book"][b]["sharpe"] > ug[b]["sharpe"] for k in (16, 18, 20, 22, 25) for b in ("G-FULL", "G-LONG"))
                      for fam in ("A_VIX", "B_ANY")}
    # QPI cross-check for passing variants and the A/B k=20 references
    fins = [n for n, v in dec.items() if v["pass"]] + ["A_VIX20", "B_ANY20"]
    qb = rs.stats(rs.swept(ll.run(p, ll.Spec("q", *Q), rs.QPI0, rs.END), rate))["sharpe"]
    rep["qpi_crosscheck"] = {"ungated": qb}
    for n in dict.fromkeys(fins):
        kind, g = V[n]
        rep["qpi_crosscheck"][n] = rs.stats(rs.swept(ll.run(p, spec(Q, kind, g), rs.QPI0, rs.END), rate))["sharpe"]
    (OUT / "results.json").write_text(json.dumps(rep, indent=2, default=float), encoding="utf-8")
    pd.DataFrame(rows).T.to_csv(OUT / "summary.csv")
    print("decision", {k: v["pass"] for k, v in dec.items()})
    print("plateau", rep["plateau"], "qpi", {k: round(v, 3) for k, v in rep["qpi_crosscheck"].items()})


if __name__ == "__main__":
    main()
