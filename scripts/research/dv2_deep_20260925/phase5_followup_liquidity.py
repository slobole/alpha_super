"""Post-hoc follow-up (labelled exploratory; the holdout is already spent): stricter liquidity for fund capacity.

Stocks: floor at the 75th / 90th percentile of same-day member ADV63 instead of the median, NATR or ADV rank.
ETFs: industry list with ADV63 > $50M (a $25M pod, 10 slots, <= 5% of daily volume per order).
"""
from pathlib import Path
import json
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import replica as rp
import phase4_universes as p4
import phase5_finalists as f5
import phase5_book as pb
import phase5_capacity_alone as pc

V = {"L1_q75_natr": rp.Rule(floor=True, floor_q=0.75), "L2_q75_adv": rp.Rule(floor=True, floor_q=0.75, rank="adv"),
     "L3_q90_adv": rp.Rule(floor=True, floor_q=0.90, rank="adv")}
p = rp.Panel("sp500")
out = {}
for a, r in V.items():
    res = rp.run(p, r)
    f5.write_source(a, res)
    out[a] = {"main": rp.summarize(res), "stress": rp.summarize(rp.run(p, r.with_(slippage=0.00075)))}
pe = p4.etf_panel(p4.GROUPS["industries"])
r = rp.Rule(adv_min=50e6)
res = rp.run(pe, r)
f5.write_source("etf_ind_adv50", res)
out["etf_ind_adv50"] = {"main": rp.summarize(res), "stress": rp.summarize(rp.run(pe, r.with_(slippage=0.0005)))}
(pb.batch.OUT / "phase5_followup_liquidity.json").write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
for a, d in out.items():
    print(a, {k: round(d["main"][k], 3) for k in ("cagr", "sharpe", "maxdd", "P1_sharpe", "P2_sharpe", "P3_sharpe")}, "stress", round(d["stress"]["sharpe"], 3))
pb.VARIANTS = list(V) + ["etf_ind_adv50"]
pc.pb.VARIANTS = pb.VARIANTS
pc.main()
