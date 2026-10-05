"""A5 follow-up (post-result, labelled): margin on the high-Sharpe defensive books as a route to ~22%.

Usage: python round2b.py   Adds entries to <study>/report/round2.json under "growth".
"""

from __future__ import annotations

import json

import numpy as np

import fp_lib as fp
from fp_lib import TBILL, Book, ga, lib
from round2 import beats, describe, lever

BASES = {"margin_d4down": {"core5": 0.25, "btal_qqq": 0.25, "etf_dv2": 0.25, "downshock": 0.25},
         "margin_deomdown": {"core5": 0.25, "btal_qqq": 0.25, "eom_flow": 0.25, "downshock": 0.25},
         "margin_mix": {"taa3x_1n": 0.192, "ndx_vxn": 0.128, "core5": 0.215, "btal_qqq": 0.215, "eom_flow": 0.125, "etf_dv2": 0.125}}


def main() -> int:
    data = lib.load_inputs()
    frames = ga.frames(data)
    frame, start = frames["main"]
    rf = frame[TBILL]
    path = fp.STUDY / "report" / "round2.json"
    out = json.loads(path.read_text(encoding="utf-8"))
    g = {"taa3x_1n": 0.384, "ndx_vxn": 0.256, "core5": 0.18, "btal_qqq": 0.18}
    r_g = lib.book_returns(frame, Book("g", tuple(g), "EQ", g), start)
    for route, w in BASES.items():
        base = lib.book_returns(frame, Book(route, tuple(w), "EQ", w), start)
        for T in (0.20, 0.22):
            for L in np.arange(1.0, 3.51, 0.05):
                r = lever(base, rf, float(L))
                if np.prod(1 + r.to_numpy()) ** (252 / len(r)) - 1 >= T:
                    break
            row = describe(r, rf, data, frames, w, float(L))
            row.update({"route": route, "weights": w, "lever": round(float(L), 2), "target": T, "beats_growth_cagr": beats(r, r_g, rf, "cagr")})
            out["growth"][f"{route}|{T:.2f}"] = row
            print("GRO", route, T, "L", round(float(L), 2), {a: round(row[a], 4) for a in ("cagr", "net", "dd", "xsharpe", "vol")},
                  {f"p{p}": round(row["tail"][f"p{p}"], 3) for p in (20, 25, 30)}, "5bps", round(row["s3_plus_5bps"]["cagr"], 4),
                  "exact", round(row["s6_exact"]["cagr"], 4), "recent", round(row["recent_cagr"], 3), "beats", row["beats_growth_cagr"],
                  {k: round(v, 3) for k, v in row["crises"].items()}, flush=True)
    path.write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    fp.ledger("round2b_finished")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
