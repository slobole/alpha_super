"""A7 (descriptive, selects nothing): the price of returns above 22%, with and without leverage.

Targets 24/26/28/30/35% gross CAGR. Routes: more TAA3x-1N inside the verdict structure (no leverage), and daily margin
(DTB3 + 1.5%, ACT/360) on the launch, plus, the growth/core mixes, the target book and the MR target.

Usage: python a7.py   Writes <study>/report/a7.json (own tail cache, extended loss limits).
"""

from __future__ import annotations

import json

import numpy as np

import fp_lib as fp
import a6
from a6 import BASES, GROWTH, GROWTH_PLUS, GROWTH_MR, Lab, blend

a6.LIMITS = (-0.10, -0.15, -0.20, -0.25, -0.30, -0.35, -0.40)       # deeper limits for this exploration
TARGETS = (0.24, 0.26, 0.28, 0.30, 0.35)
TARGET_SLOT = {"core5": 0.2375, "btal_qqq": 0.2375, "eom_flow": 0.2375, "etf_dv2": 0.2375, "downshock": 0.05}
LEV_ROUTES = {
    "launch_x": ("השקת הצמיחה, במינוף", GROWTH),
    "plus_x": ("\"יותר תשואה\" בצמיחה, במינוף", GROWTH_PLUS),
    "next_mix50_x": ("חצי צמיחה + חצי ליבת הצעד הבא, במינוף", blend((0.5, GROWTH), (0.5, BASES["C_N"][1]))),
    "target_mix50_x": ("חצי צמיחה + חצי ליבת היעד, במינוף", blend((0.5, GROWTH), (0.5, BASES["C_T"][1]))),
    "target_mix33_x": ("שליש צמיחה + שני שלישים ליבת היעד, במינוף", blend((1 / 3, GROWTH), (2 / 3, BASES["C_T"][1]))),
    "target_x": ("היעד ההגנתי לבדו, במינוף", TARGET_SLOT),
    "mr_x": ("יעד ההיפוך לממוצע, במינוף", GROWTH_MR),
}
REST = {"ndx_vxn": 0.256, "core5": 0.18, "btal_qqq": 0.18}


def main() -> int:
    lab = Lab()
    lab._disk_path = fp.STUDY / "report" / "a7_tail_cache.json"     # separate cache: extended limits
    lab._disk = json.loads(lab._disk_path.read_text(encoding="utf-8")) if lab._disk_path.exists() else {}
    out: dict = {"targets": TARGETS, "unlevered": {}, "levered": {}, "ceiling": {}}
    names = []
    # Unlevered ceiling: the single highest-return pods.
    for pod in ("taa3x_1n", "taa3x"):
        names.append(lab.add(f"PURE_{pod}", {pod: 1.0}, base="pure"))
    # No leverage: TAA3x-1N share t inside the verdict structure, the first t reaching each target.
    path = []
    for t in np.round(np.arange(0.40, 1.0001, 0.01), 2):
        w = {"taa3x_1n": float(t)} | ({k: v / sum(REST.values()) * (1 - t) for k, v in REST.items()} if t < 1 else {})
        path.append(lab.add(f"MORE_TAA|{t:.2f}", w, base="more_taa", t=float(t)))
    for T in TARGETS:
        hit = next((n for n in path if lab.cands[n]["q"]["cagr"] >= T), None)
        out["unlevered"][f"{T:.2f}"] = hit
        if hit:
            names.append(hit)
    # Leverage: the smallest L on a 0.01 grid reaching each target.
    for key, (label, w) in LEV_ROUTES.items():
        out["levered"][key] = {"label": label}
        for T in TARGETS:
            try:
                L = lab.solve_L(w, T)
            except ValueError:
                out["levered"][key][f"{T:.2f}"] = None
                continue
            n = lab.add(f"{key}|{T:.2f}", w, L, base=key)
            out["levered"][key][f"{T:.2f}"] = n
            names.append(n)
    print("books:", len(names), flush=True)
    lab.run_tails(names)
    T_ = lab.tails

    def row(n: str) -> dict:
        c, q = lab.cands[n], lab.cands[n]["q"]
        r0 = lab.ret(c["w"], c["L"])
        r5 = lab.ret(c["w"], c["L"], "s3_plus_5bps").reindex(r0.index)
        r10 = r0 + 2 * (r5 - r0)
        cg = lambda r: float(np.prod(1 + r.to_numpy()) ** (252 / len(r)) - 1)  # noqa: E731
        fin25 = cg(lab.ret(c["w"], c["L"], spread=0.025)) if c["L"] != 1.0 else q["cagr"]
        return {"name": n, "weights": c["w"], "L": c["L"], "cagr": q["cagr"], "plus5": cg(r5), "plus10": cg(r10), "fin25": fin25,
                "dd": q["dd"], "vol": q["vol"], "xs": q["xs"], "worst_year": q["worst_year"], "gfc": q["crises"]["gfc"],
                "bear_2022": q["crises"]["bear_2022"], "gfc_dd": q["crises_dd"]["gfc"], "dd_2022": q["crises_dd"]["bear_2022"],
                "tails": T_[n], "reg_t": lab.reg_t(c["w"], c["L"]), "t": c.get("t")}

    out["rows"] = {n: row(n) for n in names}
    for n in names:
        r = out["rows"][n]
        print(n, "L", r["L"], "cagr", round(r["cagr"], 4), "+5", round(r["plus5"], 4), "+10", round(r["plus10"], 4), "dd", round(r["dd"], 4),
              {k: round(v, 3) for k, v in r["tails"].items() if not k.endswith("_max") and k in ("p20", "p25", "p30", "p35", "p40")},
              "wy", round(r["worst_year"], 3), "gfc", round(r["gfc"], 3), "2022", round(r["bear_2022"], 3), "regT", round(r["reg_t"], 2), flush=True)
    (fp.STUDY / "report" / "a7.json").write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    fp.ledger("a7_finished")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
