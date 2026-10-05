import json, hashlib
from pathlib import Path
R = Path("results/research/portfolio/fund_products_20261005")
for tag in ("report", "report_equal_capital_snapshot"):
    d = json.loads((R/tag/"defensive.json").read_text(encoding="utf-8"))
    g1 = d["more_return"]["gr1"]
    print(tag, [(f["name"], f["weights"], round(f["cagr"],5), round(f["xs"],3), round(f["dd"],4), round(f["worst"],5), f["slack"]) for f in g1["frontier_launch"]])
    print("  pick", g1["pick"]["name"], g1["pick"]["weights"], g1["pick"]["q"]["crises"], g1["pick"]["q"]["worst_crisis"], {k: v for k, v in g1.items() if k not in ("frontier_launch", "pick")})
    print("  other keys", list(d["gated_upgrade"]) if isinstance(d["gated_upgrade"], dict) else "")
for l in open(R/"experiment_ledger.jsonl", encoding="utf-8"):
    j = json.loads(l)
    if j["recorded_at_utc_str"] > "2026-10-05T05:0":
        print(j["recorded_at_utc_str"][:19], j["event_str"], j.get("spec_sha256_str", "")[:10], {k: (str(v)[:400]) for k, v in j.items() if k not in ("recorded_at_utc_str","event_str","spec_sha256_str","checks")})
print("spec now", hashlib.sha256(open("scripts/research/fund_products_20261005/SPEC_FROZEN.md","rb").read()).hexdigest()[:10])
for f in ("g_lib.py","study.py","battery.py","versus.py","exposure.py","capacity.py","defensive.py","pm_confirm.py","build_report.py","build_record.py","report_texts.py"):
    print(f, hashlib.sha256(open("scripts/research/fund_products_20261005/"+f,"rb").read()).hexdigest())
