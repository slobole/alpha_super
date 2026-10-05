"""A6-d (post-review 2, labelled): every defensive slot must pass its rule at model costs AND at +5 bps.

Re-runs A6's defensive slot logic (defaults, challenges, the A6-c stock-MR gate, very defensive, more return with
its 1.0 pp fallback) with that double test, and reports downshock as a launch upgrade (60/40 + 5% / 10%).
Growth is unchanged and stays in a6.json.

Usage: python a6d.py   Writes <study>/report/a6d.json.
"""

from __future__ import annotations

import json

import fp_lib as fp
from fp_lib import lib  # noqa: F401  (keeps the import order a6 expects)
from a6 import BASES, C_L, CASH_STEPS, DEFAULT, GATED_PODS, GROWTH, RULES, Lab, blend, pro_rata, with_cash

PLUS5 = "s3_plus_5bps"


def main() -> int:
    lab = Lab()
    bases = dict(BASES)
    bases["C_L+DS_05"] = ("ds_upgrade", pro_rata(C_L, "downshock", 0.05))
    bases["C_L+DS_10"] = ("ds_upgrade", pro_rata(C_L, "downshock", 0.10))
    for base, (stage, w) in bases.items():
        for c in CASH_STEPS:
            lab.add(f"{base}|c{c:.2f}", with_cash(w, c), base=base, stage=stage, cash=c)
            lab.add(f"{base}|c{c:.2f}@5", with_cash(w, c), frame=PLUS5, base=base, stage=stage, cash=c)

    def both_ok(n: str, rule: str) -> bool:
        return lab.rule_ok(n, rule) and lab.rule_ok(n + "@5", rule)

    def scan(names_by_base: dict[str, list[str]], rule: str) -> dict[str, str | None]:
        """First name per base (in the given order) passing the rule at model costs (with the floor) and at +5 bps.
        Exact and batched: a name whose seed 0 breaks the cap in either frame fails for certain."""
        build, key, cap = RULES[rule]
        lv = {b: [n for n in ns if lab.cands[n]["q"]["dd"] >= build and lab.floor_ok(n) and lab.cands[n + "@5"]["q"]["dd"] >= build]
              for b, ns in names_by_base.items()}
        pos = {b: 0 for b in lv}
        found: dict[str, str] = {}
        while True:
            cur = {b: lv[b][pos[b]] for b in lv if b not in found and pos[b] < len(lv[b])}
            if not cur:
                break
            lab.run_seeds([x for n in cur.values() for x in (n, n + "@5")], [0])
            live = {b: n for b, n in cur.items() if lab.seed0(n, key) <= cap and lab.seed0(n + "@5", key) <= cap}
            lab.run_tails([x for n in live.values() for x in (n, n + "@5")])
            for b, n in cur.items():
                if b in live and both_ok(n, rule):
                    found[b] = n
                else:
                    pos[b] += 1
        return {b: found.get(b) for b in names_by_base}

    levels = lambda b: [f"{b}|c{c:.2f}" for c in CASH_STEPS]  # noqa: E731
    first = scan({b: levels(b) for b in bases}, "DEF")
    print("DEF double-test levels:", first, flush=True)
    out: dict = {"spec": "A6-d", "defensive": {}, "challenges": {}, "notes": {}}

    def pick(stage: str) -> tuple[str, list, str | None]:
        de = first[DEFAULT[stage]]
        if de is None:
            raise ValueError(f"default {DEFAULT[stage]} fails the double test")
        log, best, best_share, gated, gated_share = [], de, -1.0, None, -1.0
        for base, (st, w) in bases.items():
            if st != stage or base == DEFAULT[stage]:
                continue
            ch = first[base]
            if ch is None:
                log.append({"challenger": base, "default": de, "passed": False, "reason": "no passing cash level"})
                continue
            res = lab.challenge(ch, de, "p10", "xs")
            res["gated"] = bool(set(w) & GATED_PODS)
            log.append(res)
            if res["passed"] and res["gated"]:
                if res["share"] > gated_share:
                    gated, gated_share = ch, res["share"]
            elif res["passed"] and res["share"] > best_share:
                best, best_share = ch, res["share"]
        return best, log, gated

    launch, log_l, gated = pick("launch")
    nxt, log_n, _ = pick("next")
    target, log_t, _ = pick("target")
    out["challenges"].update({"launch": log_l, "next": log_n, "target": log_t})
    calm_levels = scan({"calm": levels(lab.cands[launch]["base"]), "calm_next": levels("C_N")}, "CALM")
    calm, calm_next = calm_levels["calm"], calm_levels["calm_next"]

    # More return: walk down the CAGR ranking of core + growth slice + cash; first three passing the double test.
    def richer(core: dict, tag: str) -> list[str]:
        grid = []
        for g in [round(0.05 * k, 2) for k in range(1, 11)]:
            for c in [round(0.05 * k, 2) for k in range(11)]:
                w = with_cash(blend((1 - g, core), (g, GROWTH)), c)
                n = lab.add(f"RICH_{tag}|g{g:.2f}|c{c:.2f}", w, base=f"RICH_{tag}", stage="rich", cash=c, g=g)
                lab.add(n + "@5", w, frame=PLUS5, base=f"RICH_{tag}", stage="rich", cash=c, g=g)
                if lab.cands[n]["q"]["dd"] >= RULES["DEF"][0] and lab.floor_ok(n) and lab.cands[n + "@5"]["q"]["dd"] >= RULES["DEF"][0]:
                    grid.append(n)
        grid.sort(key=lambda n: -lab.cands[n]["q"]["cagr"])
        cap = RULES["DEF"][2]
        passing = []
        for i in range(0, len(grid), 8):
            chunk = grid[i:i + 8]
            lab.run_seeds([x for n in chunk for x in (n, n + "@5")], [0])
            live = [n for n in chunk if lab.seed0(n, "p10") <= cap and lab.seed0(n + "@5", "p10") <= cap]
            lab.run_tails([x for n in live for x in (n, n + "@5")])
            passing += [n for n in chunk if n in live and both_ok(n, "DEF")]
            if len(passing) >= 3:
                break
        return passing[:3]

    def frontier_rows(names: list[str]) -> list[dict]:
        T = lab.tails
        rows = []
        for n in names:
            q = lab.cands[n]["q"]
            rows.append({"name": n, "g": lab.cands[n]["g"], "cash": lab.cands[n]["cash"], "cagr": q["cagr"], "dd": q["dd"],
                         "bear_2022": q["crises"]["bear_2022"], "gfc": q["crises"]["gfc"], "worst": q["worst_crisis"],
                         "p10": T[n]["p10"], "p10_max": T[n]["p10_max"], "p10_plus5_max": T[n + "@5"]["p10_max"],
                         "slack": {"dd": q["dd"] - RULES["DEF"][0], "bear_2022": q["crises"]["bear_2022"] + 0.01,
                                   "p10": RULES["DEF"][2] - max(T[n]["p10_max"], T[n + "@5"]["p10_max"])}})
        return rows

    core_l = blend((1.0, bases[lab.cands[launch]["base"]][1]))
    front = richer(core_l, "launch")
    out["notes"]["rich_frontier_launch"] = frontier_rows(front)
    rich, rich_stage = (front[0] if front else None), "launch"
    if rich is None or lab.cands[rich]["q"]["cagr"] - lab.cands[launch]["q"]["cagr"] < 0.010:
        out["notes"]["rich_launch_candidate"] = rich
        front = richer(blend((1.0, bases["C_N"][1])), "next")
        out["notes"]["rich_frontier_next"] = frontier_rows(front)
        rich, rich_stage = (front[0] if front else None), "next"
    out["notes"]["rich_stage"] = rich_stage

    # Head-to-heads and the reported upgrades.
    out["next_vs_launch"] = lab.challenge(nxt, launch, "p10", "xs")
    if gated:
        out["notes"]["launch_gated_shares"] = {"main": lab.share(gated, launch), "plus5": lab.share(gated, launch, PLUS5),
                                               "plus10": lab.share_plus10(gated, launch)}
    ds = {}
    for b in ("C_L+DS_05", "C_L+DS_10"):
        n = first[b]
        if n is None:
            ds[b] = None
            continue
        ds[b] = {"name": n, "vs_launch": lab.challenge(n, launch, "p10", "xs"), "next_vs_this": lab.challenge(nxt, n, "p10", "xs"),
                 "shares_vs_launch": {"plus5": lab.share(n, launch, PLUS5), "plus10": lab.share_plus10(n, launch)}}
        if gated:
            ds[b]["vs_gated_hpi"] = lab.share(n, gated)
    out["notes"]["ds_upgrade"] = ds

    T = lab.tails
    slots = {"launch": launch, "next": nxt, "calm": calm, "rich": rich, "target": target, "calm_next": calm_next, "launch_gated": gated,
             "ds_upgrade_05": first["C_L+DS_05"], "ds_upgrade_10": first["C_L+DS_10"]}
    for slot, n in slots.items():
        if n is None:
            out["defensive"][slot] = None
            continue
        c = lab.cands[n]
        out["defensive"][slot] = {"name": n, "weights": c["w"], "lever": 1.0, "cash": c.get("cash"), "g": c.get("g"), "q": c["q"],
                                  "tails": T[n], "tails_plus5": T.get(n + "@5"), "floor_ok": lab.floor_ok(n),
                                  "frames": {f: lab.frame_stats(n, f) for f in (PLUS5, "s6_exact", "s1_house_cash")},
                                  "halves_xs": list(lab.halves(n))}
        print("SLOT", slot, n, {k: round(v, 4) for k, v in c["q"].items() if isinstance(v, float)}, "p10", round(T[n]["p10"], 4),
              "max", round(T[n]["p10_max"], 4), "| +5 p10 max", round(T[n + "@5"]["p10_max"], 4) if n + "@5" in T else "-",
              "p7", round(T[n]["p7"], 4), "| +5 p7 max", round(T[n + "@5"]["p7_max"], 4) if n + "@5" in T else "-", flush=True)
    for stage, log in out["challenges"].items():
        for r in log:
            print("CHALLENGE", stage, r.get("challenger"), "vs", r.get("default"), "share", round(r["share"], 3) if "share" in r else "-",
                  "passed", r["passed"], "gated", r.get("gated"), r.get("checks", r.get("reason")), flush=True)
    print("NEXT_VS_LAUNCH", round(out["next_vs_launch"]["share"], 3), out["next_vs_launch"]["passed"], flush=True)
    print("GATED", gated, out["notes"].get("launch_gated_shares"), flush=True)
    for b, d in ds.items():
        if d:
            print("DS", b, d["name"], "vs launch", round(d["vs_launch"]["share"], 3), d["vs_launch"]["passed"], d["vs_launch"]["checks"],
                  "| next vs this", round(d["next_vs_this"]["share"], 3), d["next_vs_this"]["passed"], "| shares", d["shares_vs_launch"],
                  "| vs HPI", d.get("vs_gated_hpi"), flush=True)
    print("RICH", out["notes"]["rich_stage"], [(r["name"], round(r["cagr"], 4)) for r in out["notes"].get("rich_frontier_" + out["notes"]["rich_stage"], [])], flush=True)
    (fp.STUDY / "report" / "a6d.json").write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    fp.ledger("a6d_finished", slots={k: (v["name"] if v else None) for k, v in out["defensive"].items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
