"""SPMO parking: daily vs weekly vs weekly-with-band vs monthly re-weighting (owner question 2026-10-03; exploratory).

Parking sleeve = SPMO (or a 2008 proxy) + T-bills. Target SPMO weight = min(1, 8% / 20-day realised volatility).
  daily          re-weight every session to the target known at the previous close (the simulation used so far)
  weekly         re-weight at the last session of each week (target known at that close); between, the weights drift
  weekly_band10  as weekly, but only when the drifted weight is more than 10 points from the target
  monthly        re-weight at the last session of each month
Costs: every change in the pod's SPMO holding pays 2.5 bps per side (the engine's slippage), approximated as
|cw_t x w_t - cw_(t-1) x w_(t-1)| of pod value; this includes the trades forced by the pod's own entries and exits.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import evaluate as ev  # noqa: E402
import spmo_2008_proxy as sp  # noqa: E402

cp, fp, rs, npc, tbc, rp = ev.cp, ev.fp, ev.rs, ev.npc, ev.tbc, ev.cp.rp
OUT = cp.OUT
MODES = ("daily", "weekly", "weekly_band10", "monthly")
COST = 0.00025
CRISES_REAL = [c for c in sp.fp.CRISES if c[1] >= "2015-11-02"]


def sleeve(s: pd.Series, b: pd.Series, mode: str):
    """Daily parking-sleeve return and the SPMO weight held during each session."""
    rv = s.rolling(20).std() * np.sqrt(252)
    target = (0.08 / rv).clip(upper=1).fillna(0.0).to_numpy()
    idx = s.index
    wk_end = np.r_[idx.isocalendar().week.to_numpy()[1:] != idx.isocalendar().week.to_numpy()[:-1], True]
    mo_end = np.r_[idx.month[1:] != idx.month[:-1], True]
    sv, bv = s.fillna(0.0).to_numpy(), b.reindex(idx).fillna(0.0).to_numpy()
    w = 0.0
    held = np.zeros(len(idx))
    ret = np.zeros(len(idx))
    for t in range(len(idx)):
        held[t] = w
        p = w * sv[t] + (1 - w) * bv[t]
        ret[t] = p
        w = w * (1 + sv[t]) / (1 + p) if (1 + p) != 0 else w      # drift through the session
        tgt = target[t]                                             # known at this close, traded at the next session
        if mode == "daily" or (mode == "weekly" and wk_end[t]) or (mode == "monthly" and mo_end[t]) or \
                (mode == "weekly_band10" and wk_end[t] and abs(w - tgt) > 0.10):
            w = tgt
    return pd.Series(ret, index=idx), pd.Series(held, index=idx)


def main():
    comp = pd.read_parquet(OUT / "components.parquet")
    idx = pd.DatetimeIndex(comp.index)
    rate = rs.cash_rate(idx)
    spmo = npc.load_total_return_ret_ser("SPMO", "SPMO").reindex(idx)
    pdp = npc.load_total_return_ret_ser("PDP", "PDP").reindex(idx)
    syn = sp.synth_momentum(rp.Panel("sp500")).reindex(idx)
    taa, L = tbc.load_taa_ser(), npc.load_l_ret_ser("engine")
    series = {"SPMO (real, 2015-11+)": (spmo, "2015-11-02"),
              "PDP proxy, spliced to SPMO": (pdp.where(idx < pd.Timestamp("2015-11-02"), spmo), "2007-04-02"),
              "Synthetic proxy, spliced to SPMO": (syn.where(idx < pd.Timestamp("2015-11-02"), spmo), "2004-01-05")}

    def pod(name, park_ret, held_w):
        cw = comp[f"{name}|engine|cw"]
        expo = cw * held_w
        cost = COST * expo.diff().abs().fillna(0.0)
        return comp[f"{name}|engine|base"] + cw * park_ret - cost, float(expo.diff().abs().sum() / (len(expo) / 252))

    rep = {}
    for sname, (s, start) in series.items():
        rep[sname] = {}
        for mode in MODES:
            pr, hw = sleeve(s.fillna(0.0), rate, mode)
            dv2_m, to_dv2 = pod("DV2-G", pr, hw)
            hpi_m, to_hpi = pod("HPI-G", pr, hw)
            dv2_t = comp["DV2-G|engine|base"] + comp["DV2-G|engine|cw"] * rate
            hpi_t = comp["HPI-G|engine|base"] + comp["HPI-G|engine|cw"] * rate
            out = {"spmo_turnover_per_year_dv2": to_dv2, "avg_spmo_weight": float(hw.loc[start:].mean())}
            for lab, (a, b) in {"chosen (DV2 SPMO, HPI T-bills)": (dv2_m, hpi_t), "both SPMO": (dv2_m, hpi_m), "T-bills both": (dv2_t, hpi_t)}.items():
                cap = ev.capsule({"DV2": a, "HPI": b}, {"DV2": 0.5, "HPI": 0.5}, start=start, end="2026-09-24")
                bstart = max(start, "2008-03-04")
                bk = tbc.book_window_return_ser({"taa": taa, "L": L, "X": cap}, npc.CANDIDATE_WEIGHT_DICT, bstart, "2026-08-19")
                d = {"capsule": fp.stats(cap), "capsule_money": float(100_000 * (1 + cap).prod()), "book": tbc.metric_dict(bk), "book_money": float(100_000 * (1 + bk).prod())}
                crises = CRISES_REAL if sname.startswith("SPMO") else [("GFC", "2007-10-09", "2009-03-09"), ("Calendar 2008", "2008-01-01", "2008-12-31"), ("Sep-Nov 2008", "2008-09-01", "2008-11-30")]
                for cn, a0, b0 in crises:
                    seg = cap.loc[a0:b0]
                    nav = (1 + seg).cumprod()
                    d[cn] = {"ret": float(nav.iloc[-1] - 1), "dd": float((nav / nav.cummax() - 1).min())}
                out[lab] = d
            rep[sname][mode] = out
    (OUT / "spmo_rebalance.json").write_text(json.dumps(rep, indent=2, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
