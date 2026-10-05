import json, os, time
from pathlib import Path
R = Path("results/research/portfolio/fund_products_20261005/report")
s = json.loads((R/"study.json").read_text(encoding="utf-8"))
b = json.loads((R/"battery.json").read_text(encoding="utf-8"))
v = json.loads((R/"versus.json").read_text(encoding="utf-8"))
print("study keys", list(s))
print("products GR1", json.dumps(s["products"]["GR1"], indent=0)[:1500])
bk = s["books"]["GR1"]
print("weights", bk["weights"]); print("q", {k: bk["q"][k] for k in ("cagr","vol","xs","dd","n")})
print("tails", bk["tails"]); print("tails5", bk["tails_plus5"])
print("frames", {k:{m:x[m] for m in ("cagr","vol","xs","dd")} for k,x in bk["frames"].items()})
for n in s["books"]:
    if n.startswith(("S0","S8","T1","T2","T3","T4","S9")): print(n, s["books"][n]["weights"], {k: s["books"][n]["q"][k] for k in ("cagr","vol","xs","dd")}, s["books"][n]["tails"]["p20"])
for t,row in s["slots"].items():
    print(t, {fk:{k: (x[k] if not isinstance(x[k],dict) else {m:x[k][m] for m in ("cagr","xs","dd")}) for k in ("gr1","slot","share_xs","share_cagr")} for fk,x in row["frames"].items()}, row["adds_return_not_sharpe"])
print("mr gate", s["mr_gate_breakeven"])
print("battery keys", list(b), list(b["edge_decay"]))
h = b["edge_decay"]["headline"]
for n in ("GR1", "S0 equal capital (registered default)","S9 incumbent launch","S8 GR1 75 / DEF 25"):
    print(n, {lab:{k:h[n][lab].get(k) for k in ("cagr","vol","xs","dd","p20")} for lab in ("backtest","planning","floor")})
for sc in b["edge_decay"]["scenarios"]:
    x=b["edge_decay"]["scenarios"][sc]["GR1"]; print(sc, {k:x[k] for k in ("cagr","xs","dd","p20")})
print("worst", b["edge_decay"]["worst_case"]["GR1"], b["edge_decay"]["minimax"])
print("edge margin", b["edge_decay"]["edge_margin"]["GR1"]["lowest_passing_k"], b["edge_decay"]["edge_margin"]["GR1"]["curve"])
p = b["plateau"]["GR1"]; print("plateau n", p["n"], [(r["t"],r["m"],r["r"]) for r in p["rows"]]); print(p["summary"], p["flags"], p["rung_robust"], p["neighbour_median_dd"], p["neighbour_median_breach"])
print("nb names", [n for n in b["edge_decay"]["worst_case"] if n.startswith("nb ")])
for k in ("GR1 | S9 incumbent launch", "GR1 | S0 equal capital (registered default)", "S8 GR1 75 / DEF 25 | GR1"):
    print(k, {fk:{m: x[m] for m in ("share_xs","share_cagr","gap_xs_p5_50_95","gap_cagr_p5_50_95")} for fk,x in v[k]["frames"].items()})
    print("  blocks", {blk:{fk:(x["share_xs"],x["share_cagr"]) for fk,x in d.items()} for blk,d in v[k]["blocks"].items()})
    print("  a/b main", {m:v[k]["frames"]["main"]["a"][m] for m in ("cagr","xs","dd")},{m:v[k]["frames"]["main"]["b"][m] for m in ("cagr","xs","dd")})
print(json.dumps(json.loads((R/"pm_confirm.json").read_text()), indent=0))
e = json.loads((R/"exposure.json").read_text(encoding="utf-8"))
x = e["books"]["GR1"]; print("exp", x["nasdaq_lookthrough"]["max"], x["nasdaq_lookthrough"]["mean"], x["nasdaq_lookthrough"]["p90"], x["nasdaq_peak_date"], x["nasdaq_lookthrough_drift"]["max"], x["equity_exposure"], x["gap_table"], x["tbill_like_share"])
print(list(e["books"]))
c = json.loads((R/"capacity.json").read_text(encoding="utf-8")); print("cap keys", list(c))
d = json.loads((R/"defensive.json").read_text(encoding="utf-8")); print("def keys", list(d))
for ch in s["challenges"]:
    print(ch["challenger"], round(ch["share_xs"],4), ch["passed"], ch["checks"])
