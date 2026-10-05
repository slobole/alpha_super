"""Compliance reviewer: dump battery / capacity / exposure / defensive values the plan's reading rules use."""
import json
from pathlib import Path
REP = Path("results/research/portfolio/fund_products_20261005/report")
S = json.loads((REP / "study.json").read_text(encoding="utf-8"))
B = json.loads((REP / "battery.json").read_text(encoding="utf-8"))
C = json.loads((REP / "capacity.json").read_text(encoding="utf-8"))
X = json.loads((REP / "exposure.json").read_text(encoding="utf-8"))
D = json.loads((REP / "defensive.json").read_text(encoding="utf-8"))
ED = B["edge_decay"]
print("== scenarios", list(ED["scenarios"]))
print("books in scenarios", list(ED["scenarios"]["TAA dead"]))
print("== minimax", ED["minimax"])
wc = ED["worst_case"]
for n, v in sorted(wc.items(), key=lambda kv: -kv[1]["xs"])[:12]:
    print("  worst", n, v)
for n in ("GR1", "GR2", "GR3", "S9 incumbent launch"):
    print("  worst", n, wc[n])
print("S7 in worst_case:", any("S7" in n for n in wc))
print("== edge margin")
for n, v in ED["edge_margin"].items():
    print(n, "lowest k", v["lowest_passing_k"], "only full", v["passes_only_at_full_edge"], {k: v["curve"][k] for k in ("1.0", "0.9", "0.8", "0.75", "0.7", "0.6", "0.5") if k in v["curve"]})
print("== headline")
for n in ("GR1", "GR2", "GR3", "S9 incumbent launch"):
    h = ED["headline"][n]
    print(n, {lab: (round(h[lab]["cagr"], 4), round(h[lab]["xs"], 3), round(h[lab]["dd"], 4), h[lab].get("p20"), h[lab].get("p25"), h[lab].get("p30")) for lab in ("backtest", "planning", "floor")})
print("== decay breach by scenario (target rung)")
for sc, d in ED["scenarios"].items():
    print(sc, {n: (round(d[n]["cagr"], 4), round(d[n]["xs"], 3), d[n]["p20"], d[n]["p25"], d[n]["p30"]) for n in ("GR1", "GR2", "GR3", "S9 incumbent launch")})
print("== plateau")
for n, p in B["plateau"].items():
    print(n, "n", p["n"], "robust", p["rung_robust"], "med dd", p["neighbour_median_dd"], "med breach", p["neighbour_median_breach"], "flags", p["flags"])
    print("   ", {k: (round(v["min"], 4), round(v["median"], 4), round(v["max"], 4), round(v["product"], 4), v["rank_from_best"]) for k, v in p["summary"].items()})
print("== reset")
for n, r in B["reset"].items():
    print(n, r["start_month_spread"])
    print("   ", {p: (round(v["free"]["cagr"], 4), round(v["free"]["xs"], 3), round(v["charged"]["cagr"], 4)) for p, v in r["policies"].items() if p in ("annual", "none", "quarterly", "monthly")})
    print("    max share", r["policies"]["annual"]["max_capsule_share"])
print("== rolling share above", B["rolling_share_above"])
print("== bootstrap blocks")
for b, d in B["bootstrap"]["blocks"].items():
    print(b, d)
print("horizons", B["bootstrap"]["horizons"])
print("frames", B["bootstrap"]["frames"])
print("VR", B["bootstrap"]["variance_ratio"])
print("cagr pct", B["bootstrap"]["cagr"])
print("== believe")
for n, v in B["believe"].items():
    print(n, v)
dp = B["dependence"]
print("== dependence keys", list(dp))
for n, v in dp["products"].items():
    print(n, "risk", v["risk_share"], "cluster", v["cluster_risk_share"], "ret", v["excess_return_share"], "DR", v["diversification_ratio"], "pred", v["vol_predicted_vs_realised"], "flags", v["flags"], "enb", v["enb"], "months", v["months_all_three_lost"], "maxdd", v["at_max_dd"])
print("tail dep", dp["tail_dependence"])
print("gate share", dp["gate_open_share"])
print("== after", B["after_window"])
print("\n== capacity")
for n, e in C["books"].items():
    print(n, "excess", round(e["exact_excess_cagr"], 4), "uncovered", e["orders_uncovered"], "/", e["orders_3y"], "limited", e["capacity_limited"], "btal wall", e["btal_wall"], "part", {k: (round(v / 1e6, 2) if isinstance(v, float) else v) for k, v in e["participation"].items() if k != "top_decile_symbols"}, "top", e["participation"]["top_decile_symbols"])
    print("    ", {r: (v["recommended"], v["first_fail"], {lv: (round(x["cost"], 5), x["gates_ok"]) for lv, x in v["levels"].items()}) for r, v in e["routes"].items()})
    print("     ease", e["ease"], "fee", e["fee_income"])
print("\n== exposure")
print(X["meta"], X["tqqq_weight"], X["mom_invested"], X["mr_stock_weight"], X["gate_open_share"])
for n in ("GR1", "GR2", "GR3"):
    b = X["books"][n]
    print(n, "nasdaq", {k: v for k, v in b["nasdaq_lookthrough"].items() if not k.startswith("by_year")}, "equity", b["equity_exposure"], "gap", b["gap_table"], "tbill", b["tbill_like_share"])
print("\n== defensive")
print("launch", D["launch"]["q"]["cagr"], D["launch"]["tails"])
for k, v in D["more_return"].items():
    if k == "parity":
        print("parity", v); continue
    print(k, v["stage"], None if v["pick"] is None else (v["pick"]["name"], v["pick"]["weights"], round(v["pick"]["q"]["cagr"], 4), v["pick"]["tails"]["p10"], v["pick"]["tails_plus5"]["p10_max"]), v.get("gain_over_launch_pp"), [(f["name"], round(f["cagr"], 4)) for f in v["frontier_launch"]])
for k, v in D["gated_upgrade"].items():
    print(k, None if not v.get("pick") else (v["pick"]["name"], v["pick"]["weights"], round(v["pick"]["q"]["cagr"], 4), round(v["pick"]["q"]["xs"], 3), v["pick"]["tails"]["p10"], v["shares_vs_launch"], v["vs_launch"]["checks"], v["vs_launch"]["passed"]))
print("stored keys", list(D["stored"]))
for k, v in D["stored"].items():
    if v: print("  stored", k, v["name"], v["weights"], round(v["q"]["cagr"], 4), v["tails"].get("p10"))
print("stored notes", D["stored_notes"])
