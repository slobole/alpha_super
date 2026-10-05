"""Fund products, final pass: the two DEFENSIVE rows that depend on growth or on stock MR (SPEC 6).

The defensive slots keep their A6-d books. Shown beside them as target-stage rows, replacing nothing:
  1. "more return" (60/40 + growth slice + cash) re-scanned by the A6-d rule with the growth slice = unscaled GR1,
     after a parity run of the same code with the old growth book S9 reproduces the stored slot;
  2. the gated upgrade restated with the capsule: 60/40 at 90% + MR capsule 10%, smallest cash passing the DEF double
     test, tested against the launch slot, paired shares at 0 / +5 / +10 bps.

A6-d rule (a6d.py): DEF = historical max DD >= -7% and P(max DD < -10%) <= 10% for the 10-seed mean and the worst seed,
in the MAIN frame and at +5 bps; crisis floor in MAIN: GFC window >= -1%, 2022 window >= -1%, worst named crisis >= -5%.

Usage: PYTHONDONTWRITEBYTECODE=1 python defensive.py   Writes <study>/report/defensive.json.
"""

from __future__ import annotations

import json

import g_lib as g
from g_lib import TBILL, Lab

C_L = {"core5": 0.6, "btal_qqq": 0.4}
C_N = {"core5": 1 / 3, "btal_qqq": 1 / 3, "etf_dv2": 1 / 3}
DEF_RULE = (-0.07, "p10", 0.10)
FLOOR = {"gfc": -0.01, "bear_2022": -0.01, "worst": -0.05}
PLUS5 = "s3_plus_5bps"


def main() -> int:
    lab = Lab()
    old = json.loads((g.WT_REPO / "results/research/portfolio/fund_products_20260930/report/a6d.json").read_text(encoding="utf-8"))

    def floor_ok(n: str) -> bool:
        q = lab.cands[n]["q"]
        return q["crises"]["gfc"] >= FLOOR["gfc"] and q["crises"]["bear_2022"] >= FLOOR["bear_2022"] and q["worst_crisis"] >= FLOOR["worst"]

    def dd_ok(n: str) -> bool:
        return lab.cands[n]["q"]["dd"] >= DEF_RULE[0] and lab.frame_stats(n, PLUS5)["dd"] >= DEF_RULE[0]

    def double_ok(n: str) -> bool:
        if not (dd_ok(n) and floor_ok(n)):
            return False
        for fk in ("main", PLUS5):
            t = lab.tails(n, fk)
            if t["p10"] > DEF_RULE[2] or t["p10_max"] > DEF_RULE[2]:
                return False
        return True

    def row(n: str) -> dict:
        c = lab.cands[n]
        return {"name": n, "weights": c["w"], "q": c["q"], "tails": lab.tails(n), "tails_plus5": lab.tails(n, PLUS5),
                "floor_ok": floor_ok(n), "frames": {f: lab.frame_stats(n, f) for f in (PLUS5, "s6_exact", "s1_house_cash")},
                "halves_xs": list(lab.halves(n))}

    launch = lab.add("launch C_L|c0.10", g.with_cash(C_L, 0.10))
    assert double_ok(launch)
    out: dict = {"launch": row(launch), "more_return": {}, "gated_upgrade": {}}

    def richer(core: dict, growth: dict, tag: str) -> list[str]:
        """a6d.richer: walk down the CAGR ranking of core + growth slice + cash; the first three passing the double test."""
        grid = []
        for gs in [round(0.05 * k, 2) for k in range(1, 11)]:
            for c in [round(0.05 * k, 2) for k in range(11)]:
                n = lab.add(f"RICH_{tag}|g{gs:.2f}|c{c:.2f}", g.with_cash(g.blend((1 - gs, core), (gs, growth)), c), gs=gs, cash=c)
                if dd_ok(n) and floor_ok(n):
                    grid.append(n)
        grid.sort(key=lambda n: -lab.cands[n]["q"]["cagr"])
        passing = []
        for i in range(0, len(grid), 8):
            chunk = grid[i:i + 8]
            for fk in ("main", PLUS5):
                lab.run_tails(chunk, fk)
            passing += [n for n in chunk if double_ok(n)]
            if len(passing) >= 3:
                break
        return passing[:3]

    def frontier(names: list[str]) -> list[dict]:
        rows = []
        for n in names:
            q, t, t5 = lab.cands[n]["q"], lab.tails(n), lab.tails(n, PLUS5)
            rows.append({"name": n, "g": lab.cands[n]["gs"], "cash": lab.cands[n]["cash"], "weights": lab.cands[n]["w"], "cagr": q["cagr"], "xs": q["xs"],
                         "dd": q["dd"], "gfc": q["crises"]["gfc"], "bear_2022": q["crises"]["bear_2022"], "worst": q["worst_crisis"],
                         "p10": t["p10"], "p10_max": t["p10_max"], "p10_plus5_max": t5["p10_max"],
                         "slack": {"dd": q["dd"] - DEF_RULE[0], "bear_2022": q["crises"]["bear_2022"] - FLOOR["bear_2022"],
                                   "p10": DEF_RULE[2] - max(t["p10_max"], t5["p10_max"])}})
        return rows

    launch_cagr = lab.cands[launch]["q"]["cagr"]
    for tag, growth in (("s9_parity", g.INCUMBENT), ("gr1", g.PRODUCTS["GR1"])):
        front = richer(C_L, growth, f"{tag}_launch")
        rec = {"frontier_launch": frontier(front), "stage": "launch"}
        pick = front[0] if front else None
        if pick is None or lab.cands[pick]["q"]["cagr"] - launch_cagr < 0.010:
            rec["launch_candidate"] = pick
            front = richer(C_N, growth, f"{tag}_next")
            rec["frontier_next"], rec["stage"] = frontier(front), "next"
            pick = front[0] if front else None
        rec["pick"] = None if pick is None else row(pick)
        if pick is not None:
            rec["pick"]["g"], rec["pick"]["cash"] = lab.cands[pick]["gs"], lab.cands[pick]["cash"]
            rec["gain_over_launch_pp"] = (lab.cands[pick]["q"]["cagr"] - launch_cagr) * 100
        out["more_return"][tag] = rec
        print("MORE RETURN", tag, rec["stage"], None if pick is None else (pick, round(lab.cands[pick]["q"]["cagr"], 4)), flush=True)
    # Parity: the same code with the old growth book must reproduce the stored A6-d slot.
    stored = old["defensive"]["rich"]
    par = out["more_return"]["s9_parity"]["pick"]
    par_name = f"RICH_launch|g{par['g']:.2f}|c{par['cash']:.2f}"
    out["more_return"]["parity"] = {"stored_name": stored["name"], "recomputed_name": par_name, "stored_cagr": stored["q"]["cagr"], "recomputed_cagr": par["q"]["cagr"],
                                    "stored_p10": stored["tails"]["p10"], "recomputed_p10": par["tails"]["p10"]}
    assert stored["name"] == par_name and abs(stored["q"]["cagr"] - par["q"]["cagr"]) < 1e-12 and abs(stored["tails"]["p10"] - par["tails"]["p10"]) < 1e-12, \
        out["more_return"]["parity"]

    # Gated upgrade with the capsule: 60/40 at 90% + MR capsule 10%, smallest passing cash.
    for tag, base in (("mr_capsule_10", g.blend((0.9, C_L), (0.1, g.MR))), ("hpi_g_10", g.blend((0.9, C_L), (0.1, {"hpi_g": 1.0}))),
                      ("hpi_vote_10 (stored A6-d row)", g.blend((0.9, C_L), (0.1, {"hpi_vote": 1.0})))):
        found = None
        for c in [round(0.05 * k, 2) for k in range(13)]:
            n = lab.add(f"C_L+{tag}|c{c:.2f}", g.with_cash(base, c), cash=c)
            if double_ok(n):
                found = n
                break
        rec = {"pick": None}
        if found:
            ch = lab.challenge(found, launch, breach_key="p10", rung=None)
            shares = {"main": ch["share_xs"], "plus5": lab.paired(lab.r(found, PLUS5), lab.r(launch, PLUS5))["share_xs"],
                      "plus10": lab.paired(lab.plus10(found), lab.plus10(launch))["share_xs"]}
            rec = {"pick": row(found), "cash": lab.cands[found]["cash"], "vs_launch": ch, "shares_vs_launch": shares}
        out["gated_upgrade"][tag] = rec
        print("GATED", tag, found, None if not found else rec["shares_vs_launch"], flush=True)
    out["stored"] = {k: (None if v is None else {"name": v["name"], "weights": v["weights"], "q": v["q"], "tails": v["tails"], "tails_plus5": v.get("tails_plus5")})
                     for k, v in old["defensive"].items()}
    out["stored_notes"] = {"launch_gated_shares": old["notes"].get("launch_gated_shares")}
    (g.OUT / "defensive.json").write_text(json.dumps(g.r6(out), indent=1, default=str), encoding="utf-8")
    g.ledger("defensive_finished")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
