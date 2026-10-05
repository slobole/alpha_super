"""A6-b (post-result, labelled; reported, replaces nothing): four questions the A6 results raised.

1. Is the next-step book better than the launch winner (A6 compared each slot only with its own default)?
2. When DV2-IND arrives, should HPI stay (C_N + HPI 10%; the launch book + DV2-IND)?
3. Does the HPI launch still beat 60/40 + cash if the HPI live gap is never fixed (frame s5)? DV2 instead?
4. Downshock 5% in the target passed with a paired share of 0.801: how stable is that share across seeds 1-4?

Usage: python a6b.py   Writes <study>/report/a6b.json.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

import fp_lib as fp
from fp_lib import TBILL, lib
import evaluate as ev
from a6 import BASES, C_L, CASH_STEPS, Lab, pro_rata, with_cash


def share_seed(lab: Lab, a: str, b: str, seed: int, frame_key: str = "main") -> float:
    """Paired excess-Sharpe share on another seed's 2,000 paths (and optionally another frame)."""
    fr = lab.frames[frame_key][0]
    ra = lab.ret(lab.cands[a]["w"], lab.cands[a]["L"], frame_key)
    rb = lab.ret(lab.cands[b]["w"], lab.cands[b]["L"], frame_key)
    A = pd.concat([ra, rb, fr[TBILL].reindex(ra.index)], axis=1).dropna().to_numpy()
    idx = lib.evaluation.stationary_bootstrap_index_mat(len(A), 2000, 63.0, fp.SEED0 + seed)
    wins = 0
    for k in range(2000):
        s = A[idx[k]]
        wins += ev.xsharpe(s[:, 0], s[:, 2]) > ev.xsharpe(s[:, 1], s[:, 2])
    return wins / 2000


def main() -> int:
    lab = Lab()
    for base, (stage, w) in BASES.items():
        for c in CASH_STEPS:
            lab.add(f"{base}|c{c:.2f}", with_cash(w, c), base=base, stage=stage, cash=c)
    hpi_core = pro_rata(C_L, "hpi_vote", 0.10)
    extra = {"N+HPI_10": pro_rata(BASES["C_N"][1], "hpi_vote", 0.10),
             "L+IND_25": pro_rata(hpi_core, "etf_dv2", 0.25),
             "L+IND_33": pro_rata(hpi_core, "etf_dv2", 1 / 3)}
    for base, w in extra.items():
        BASES[base] = ("a6b", w)
        for c in CASH_STEPS:
            lab.add(f"{base}|c{c:.2f}", with_cash(w, c), base=base, stage="a6b", cash=c)
    out: dict = {}
    launch, default_l, nxt, target, target_def = "C_L+HPI_10|c0.10", "C_L|c0.05", "C_N|c0.00", "C_T+DS_05|c0.00", "C_T|c0.00"
    dv2_launch = "C_L+DV2_10|c0.10"
    lab.run_tails([launch, default_l, nxt, target, target_def, dv2_launch])

    # 1. Next step vs launch, head to head (both directions of the same paired test).
    out["next_vs_launch"] = lab.challenge(nxt, launch, "p10", "xs")
    print("Q1 next vs launch", round(out["next_vs_launch"]["share"], 3), out["next_vs_launch"]["checks"], flush=True)

    # 2. HPI with DV2-IND.
    out["hpi_with_ind"] = []
    for base in extra:
        n = lab.min_cash(base, "DEF")
        if n is None:
            out["hpi_with_ind"].append({"base": base, "passed": False, "reason": "no passing cash level"})
            print("Q2", base, "no passing cash level", flush=True)
            continue
        vs_next = lab.challenge(n, nxt, "p10", "xs")
        q, t = lab.cands[n]["q"], lab.tails[n]
        out["hpi_with_ind"].append({"base": base, "name": n, "weights": lab.cands[n]["w"], "q": q, "tails": t, "vs_next": vs_next})
        print("Q2", n, {k: round(v, 4) for k, v in q.items() if isinstance(v, float)}, "p10", round(t["p10"], 4),
              "vs next share", round(vs_next["share"], 3), "passed", vs_next["passed"], vs_next["checks"], flush=True)

    # 3. HPI live gap (frame s5): the launch winner, the DV2 alternative and the 60/40 default.
    out["hpi_gap"] = {}
    for n in (launch, dv2_launch, default_l):
        out["hpi_gap"][n] = {"s5": lab.frame_stats(n, "s5_hpi_live_gap"),
                             "base": {"cagr": lab.cands[n]["q"]["cagr"], "xs": lab.cands[n]["q"]["xs"]}}
    out["hpi_gap_share_vs_default"] = share_seed(lab, launch, default_l, 0, "s5_hpi_live_gap")
    out["hpi_gap_share_hpi_vs_dv2"] = share_seed(lab, launch, dv2_launch, 0, "s5_hpi_live_gap")
    out["share_hpi_vs_dv2_main"] = share_seed(lab, launch, dv2_launch, 0)
    print("Q3", {n: {k: round(v, 4) for k, v in d["s5"].items()} for n, d in out["hpi_gap"].items()},
          "share HPI-launch vs default under gap", out["hpi_gap_share_vs_default"],
          "HPI vs DV2 under gap", out["hpi_gap_share_hpi_vs_dv2"], "HPI vs DV2 main", out["share_hpi_vs_dv2_main"], flush=True)

    # 4. Downshock 5% in the target: paired share on seeds 0-4.
    out["ds5_target_shares"] = [share_seed(lab, target, target_def, s) for s in range(5)]
    print("Q4 shares", out["ds5_target_shares"], flush=True)
    (fp.STUDY / "report" / "a6b.json").write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    fp.ledger("a6b_finished")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
