import json, os, datetime
from pathlib import Path
R = Path("results/research/portfolio/fund_products_20261005")
def load(tag, f): return json.loads((R/tag/f).read_text(encoding="utf-8"))
for tag in ("report", "report_equal_capital_snapshot"):
    s, b, e, d, v = (load(tag, f) for f in ("study.json", "battery.json", "exposure.json", "defensive.json", "versus.json"))
    print("=====", tag)
    rev = [c for c in s["challenges_reverse"] if c["default"].startswith("S9")][0]
    print("GR1 vs S9 reverse:", rev["share_xs"], rev["passed"], rev["checks"], "plus5 share", v["GR1 | S9 incumbent launch"]["frames"]["plus5"]["share_xs"])
    rev0 = [c for c in s["challenges_reverse"] if c["default"].startswith("S0")]
    if rev0: print("GR1 vs S0 reverse:", rev0[0]["share_xs"], rev0[0]["passed"], rev0[0]["checks"])
    s8 = [c for c in s["challenges"] if c["challenger"].startswith("S8")][0]; print("S8 halves", s8["halves_xs"], s8["halves_xs"][0][0]-s8["halves_xs"][1][0], s8["cagr"])
    for n in ("GR1","GR2","GR3"):
        x = e["books"][n]; print(n, "drift", x["nasdaq_lookthrough_drift"]["max"], x["nasdaq_lookthrough_drift"]["peak_date"], x["nasdaq_lookthrough_drift"]["max_since_2015"], "target since 2015", x["nasdaq_max_since_2015"])
    for k, m in s["margin"].items():
        for kind in ("vol_matched","cagr_matched"):
            r = m[kind]; print(k, kind, r["name"], {a: round(r["q"][a],5) for a in ("cagr","vol","dd","xs")}, r["tails"][r["breach_key"]], "p5", round(r["plus5"]["cagr"],5), "s250", round(r["spread_250"]["cagr"],5), "meanL", round(r["mean_leverage"],4), "peakL", round(r["peak_leverage"],4), "regT", r["reg_t"])
        print("   target", {a: round(m["target"]["q"][a],5) for a in ("cagr","vol","dd")}, m["target"]["tails"][m["target"]["breach_key"]], round(m["target"]["plus5"]["cagr"],5))
    sc = b["edge_decay"]["scenarios"]
    for n in ("GR1", "S0 equal capital (registered default)", "dial taa3x 33"):
        if n in sc["TAA dead"]: print(n, {k: round(sc[k][n]["xs"],3) for k in ("TAA dead","Defense First dead (TAA and BTAL_QQQ)","MOM dead","MR dead","all at 0.75","all at 0.5")})
    print("def more_return", json.dumps(d["more_return"])[:3000])
    al = [a for a in b["alpha"] if a["book"]=="GR1" and a["window"]=="long" and a["model"]=="M2"]; print([(a["basis"], a["alpha_ann"], a["alpha_t"]) for a in al])
    sy = b["start_years"]["GR1"]; print("start years cagr", min(x["cagr"] for x in sy.values()), max(x["cagr"] for x in sy.values()), "xs", min(x["xs"] for x in sy.values()), max(x["xs"] for x in sy.values()))
    print("blocks GR1", {k: (round(x["cagr"],4), round(x["xs"],3), round(x["dd"],4)) for k, x in s["books"]["GR1"]["blocks"].items()})
    print("slot blocks", {t: {k: round(x["share_xs"],3) for k, x in r["blocks"].items()} for t, r in s["slots"].items()})
    print("T2 planning", b["edge_decay"]["headline"]["T2 MR -> BIL (GR1-L)"]["planning"]["cagr"], b["edge_decay"]["headline"]["T2 MR -> BIL (GR1-L)"]["floor"]["cagr"])
c = load("report", "capacity.json")
for n in ("GR1", "T2 MR -> BIL (GR1-L)", "dial taa3x 40", "dial taa3x 33"):
    x = c["books"][n]; print(n, x["turnover_x_nav"], x["btal_wall"], x["participation_by_leg"]["aum_p99_at_5pct"], x["routes"]["MOO"]["recommended"], x["routes"]["worked+blocks"]["recommended"])
for p in ["scripts/research/fund_products_20261005/g_lib.py", "portfolios/fund_growth.yaml", "results/research/portfolio/fund_growth/vanilla_backtest/2026-10-05_084534/fund_growth.pkl"] + [str(R/"report"/f) for f in os.listdir(R/"report")]:
    print(datetime.datetime.fromtimestamp(os.path.getmtime(p)).strftime("%H:%M:%S"), p)
import subprocess
print(open(R/"pm_write.log").read()); print(open(R/"pm_fund_growth.log", encoding="utf-8", errors="replace").read()[:1500])
for l in open(R/"experiment_ledger.jsonl", encoding="utf-8"):
    j = json.loads(l); print(j["recorded_at_utc_str"][:19], j["event_str"], j["spec_sha256_str"][:10], {k: (str(v)[:150]) for k, v in j.items() if k not in ("recorded_at_utc_str","event_str","spec_sha256_str","checks")})
import hashlib; print("spec now", hashlib.sha256(open("scripts/research/fund_products_20261005/SPEC_FROZEN.md","rb").read()).hexdigest()[:10])
