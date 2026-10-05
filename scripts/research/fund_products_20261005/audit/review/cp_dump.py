"""Compliance reviewer: dump the stored JSON values the frozen plan's rules read (no recomputation)."""
import json
from pathlib import Path
REP = Path("results/research/portfolio/fund_products_20261005/report")
S = json.loads((REP / "study.json").read_text(encoding="utf-8"))
B = json.loads((REP / "battery.json").read_text(encoding="utf-8"))
C = json.loads((REP / "capacity.json").read_text(encoding="utf-8"))
X = json.loads((REP / "exposure.json").read_text(encoding="utf-8"))
D = json.loads((REP / "defensive.json").read_text(encoding="utf-8"))
print("study keys", list(S)); print("battery keys", list(B)); print("capacity keys", list(C)); print("exposure keys", list(X)); print("def keys", list(D))
print("\n== products")
for n, p in S["products"].items():
    print(n, json.dumps(p, indent=None)[:1200])
print("\n== books list", list(S["books"]))
print("\n== book block keys", list(S["books"]["GR1"]))
print("blocks keys", list(S["books"]["GR1"]["blocks"]))
for n in ("GR1", "GR2", "GR3", "S9 incumbent launch"):
    b = S["books"][n]
    print(n, "q", {k: round(v, 4) if isinstance(v, float) else v for k, v in b["q"].items() if k not in ("crises",)})
    print("   tails", b["tails"]); print("   tails+5", b["tails_plus5"])
    print("   frames", {f: (round(v["cagr"], 4), round(v["xs"], 3), round(v["dd"], 4)) for f, v in b["frames"].items()})
    print("   rungs", {r: (v["pass"], {fk: (round(v[fk]["dd"], 4), v[fk]["breach_mean"], v[fk]["breach_worst_seed"]) for fk in ("main", "s3_plus_5bps")}) for r, v in b["rungs"].items()}, "strictest", b["strictest_rung"])
    print("   net", b["net"], "halves", b["halves_xs"])
print("\n== mr gate", S["mr_gate_breakeven"])
print("\n== slots")
for t, r in S["slots"].items():
    print(t, "adds_return_not_sharpe", r["adds_return_not_sharpe"], {fk: (v["share_xs"], v["share_cagr"], round(v["gr1"]["xs"], 3), round(v["slot"]["xs"], 3), round(v["gr1"]["cagr"], 4), round(v["slot"]["cagr"], 4)) for fk, v in r["frames"].items()})
    print("   blocks", {k: (v["share_xs"], v["share_cagr"], v["paths"]) for k, v in r["blocks"].items()})
print("\n== challenges")
for c in S["challenges"]:
    print(c["challenger"], "share", c["share_xs"], "cagr share", c["share_cagr"], "borderline", c["borderline"], "rung", c["rung_pass"], "breach", c["breach"], "passed", c["passed"], "wo c2", c["passed_without_c2"], "gap", [round(x, 3) for x in c["gap_xs_p5_50_95"]], "blk", {k: round(v, 3) for k, v in c["block_gap_xs"].items()}, "xsm", [round(x, 3) for x in c["xs_monthly"]], "xs", [round(x, 3) for x in c["xs"]])
print("\n== reverse")
for c in S["challenges_reverse"]:
    print(c["default"], "share", c["share_xs"], "cagr share", c["share_cagr"], "checks", c["checks"], "passed", c["passed"])
print("gr2 vs old plus", {k: S["gr2_vs_old_plus"][k] for k in ("share_xs", "share_cagr", "checks", "passed", "breach", "borderline")})
print("old plus vs gr2", {k: S["old_plus_vs_gr2"][k] for k in ("share_xs", "share_cagr", "checks", "passed", "breach")})
print("\n== standins")
for n, v in S["standins"].items():
    print(n, v["rung"], "vs s9 share", v["vs_s9"]["share_xs"], v["vs_s9"]["checks"], v["vs_s9"]["passed"], v["corr_with_gr1"])
print("\n== margin")
for k, m in S["margin"].items():
    for kind in ("vol_matched", "cagr_matched"):
        r = m[kind]
        print(k, kind, "L", r["L"], "cagr", round(r["q"]["cagr"], 4), "dd", round(r["q"]["dd"], 4), "vol", round(r["q"]["vol"], 4), "xs", round(r["q"]["xs"], 3), "breach", r["tails"][r["breach_key"]], "+5", round(r["plus5"]["cagr"], 4), "exact", round(r["exact"]["cagr"], 4), "s050", round(r["spread_050"]["cagr"], 4), "s250", round(r["spread_250"]["cagr"], 4), "daily", round(r["daily_constant"]["cagr"], 4), "peak", r["peak_leverage"], "regt", r["reg_t"], r["needs_portfolio_margin"], "worst yr", r["q"]["worst_year"])
    t = m["target"]; print("   target", round(t["q"]["cagr"], 4), round(t["q"]["dd"], 4), round(t["q"]["vol"], 4), t["tails"][t["breach_key"]], round(t["plus5"]["cagr"], 4), round(t["exact"]["cagr"], 4), t["reg_t"])
print("\n== blends")
for k, v in S["blends"].items():
    print(k, {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in v["q"].items() if kk in ("cagr", "vol", "xs", "dd")}, v.get("defense_first_capital"), (v.get("diluted") or {}).get("name"), {kk: round(vv, 4) for kk, vv in ((v.get("diluted") or {}).get("q") or {}).items() if kk in ("cagr", "vol", "xs", "dd")})
print("\n== construction", S["construction"])
print("\n== dial", S["dial"])
print("\n== s7 weights", S["s7_weights"][:4])
