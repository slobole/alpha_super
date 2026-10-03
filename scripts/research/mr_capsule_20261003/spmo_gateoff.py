"""EXPLORATORY (after the rebalance results, 2026-10-03): hold SPMO only while the gate is closed.

Weekly vol-target re-weight as spmo_rebalance.weekly, but at every close where the gate is open the SPMO weight goes to
0 (T-bills only); on the close where the gate shuts, the weight goes straight to the target. Same 2.5 bps per side on
SPMO trades. Printed only; reproduces the numbers quoted to the owner on 2026-10-03.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import spmo_rebalance as sr  # noqa: E402

ev, cp, fp, rs, npc, tbc, rp, sp = sr.ev, sr.cp, sr.fp, sr.rs, sr.npc, sr.tbc, sr.rp, sr.sp


def main():
    comp = pd.read_parquet(cp.OUT / "components.parquet")
    idx = pd.DatetimeIndex(comp.index)
    rate = rs.cash_rate(idx)
    gate = pd.Series(cp.gate_on(idx), index=idx)
    spmo = npc.load_total_return_ret_ser("SPMO", "SPMO").reindex(idx)
    pdp = npc.load_total_return_ret_ser("PDP", "PDP").reindex(idx)
    syn = sp.synth_momentum(rp.Panel("sp500")).reindex(idx)
    taa, L = tbc.load_taa_ser(), npc.load_l_ret_ser("engine")

    def sleeve_gateoff(s, b):
        rv = s.rolling(20).std() * np.sqrt(252)
        target = (0.08 / rv).clip(upper=1).fillna(0.0).to_numpy()
        wk = idx.isocalendar().week.to_numpy()
        wk_end = np.r_[wk[1:] != wk[:-1], True]
        sv, bv, g = s.fillna(0.0).to_numpy(), b.fillna(0.0).to_numpy(), gate.to_numpy()
        w, held, ret = 0.0, np.zeros(len(idx)), np.zeros(len(idx))
        for t in range(len(idx)):
            held[t] = w
            p = w * sv[t] + (1 - w) * bv[t]
            ret[t] = p
            w = w * (1 + sv[t]) / (1 + p)
            if g[t]:
                w = 0.0
            elif wk_end[t] or (t > 0 and g[t - 1]):
                w = target[t]
        return pd.Series(ret, index=idx), pd.Series(held, index=idx)

    def pod(name, pr, hw):
        cw = comp[f"{name}|engine|cw"]
        e = cw * hw
        return comp[f"{name}|engine|base"] + cw * pr - sr.COST * e.diff().abs().fillna(0.0), float(e.diff().abs().sum() / (len(e) / 252))

    tb = {n: comp[f"{n}|engine|base"] + comp[f"{n}|engine|cw"] * rate for n in ("DV2-G", "HPI-G")}
    cases = (("SPMO real 2015-11+", spmo, "2015-11-02", sr.CRISES_REAL),
             ("PDP proxy (spliced)", pdp.where(idx < pd.Timestamp("2015-11-02"), spmo), "2007-04-02", [("GFC", "2007-10-09", "2009-03-09"), ("Calendar 2008", "2008-01-01", "2008-12-31")]),
             ("Synthetic proxy (spliced)", syn.where(idx < pd.Timestamp("2015-11-02"), spmo), "2004-01-05", [("GFC", "2007-10-09", "2009-03-09"), ("Calendar 2008", "2008-01-01", "2008-12-31")]))
    for sname, s, start, crises in cases:
        print(f"\n=== {sname}")
        for mode in ("weekly", "weekly + SPMO only while gate closed"):
            pr, hw = sr.sleeve(s.fillna(0.0), rate, "weekly") if mode == "weekly" else sleeve_gateoff(s.fillna(0.0), rate)
            dv, to = pod("DV2-G", pr, hw)
            hp, _ = pod("HPI-G", pr, hw)
            for lab, (a, b) in (("chosen", (dv, tb["HPI-G"])), ("both SPMO", (dv, hp))):
                cap = ev.capsule({"DV2": a, "HPI": b}, {"DV2": .5, "HPI": .5}, start=start, end="2026-09-24")
                bk = tbc.book_window_return_ser({"taa": taa, "L": L, "X": cap}, npc.CANDIDATE_WEIGHT_DICT, max(start, "2008-03-04"), "2026-08-19")
                cs, bm = fp.stats(cap), tbc.metric_dict(bk)
                cr = "  ".join(f"{n}: {((1 + cap.loc[a0:b0]).prod() - 1) * 100:5.1f}%" for n, a0, b0 in crises)
                print(f"  {mode:<38} {lab:<10} cap ${100000 * (1 + cap).prod():>10,.0f} Sh {cs['sharpe']:.3f} DD {cs['max_dd'] * 100:5.1f}% | "
                      f"book ${100000 * (1 + bk).prod():>10,.0f} Sh {bm['sharpe']:.3f} DD {bm['max_dd'] * 100:5.1f}% | turnover {to:5.2f}")
                print(f"      {cr}")


if __name__ == "__main__":
    main()
