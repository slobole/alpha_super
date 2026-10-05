"""Compliance reviewer: plan item 5.12 for whichever PortfolioManager runs have finished (read-only; no ledger, no json in the study folder)."""
import json, pickle, sys
from pathlib import Path
import pandas as pd
WT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(WT / "scripts" / "research" / "fund_products_20261005"))
import g_lib as g
OUTD = WT / "results/research/portfolio/fund_products_20261005/audit/review/compliance"
lab = g.Lab()
lab._disk_path = OUTD / "never_written_tail_cache.json"
out = {}
for code, stem in {"GR1": "fund_growth", "GR2": "fund_growth_plus", "GR3": "fund_growth_aggressive"}.items():
    runs = sorted((WT / "results/research/portfolio" / stem / "vanilla_backtest").glob("*/" + stem + ".pkl"))
    if not runs:
        out[code] = "no finished run"; print(code, "no finished run"); continue
    with open(runs[-1], "rb") as fh:
        pf = pickle.load(fh)
    tv = pf.results["total_value"].astype(float)
    tv.index = pd.to_datetime(tv.index).normalize()
    house = lab.ret(g.PRODUCTS[code], "s1_house_cash").loc["2013-01-02":]
    pm = tv.pct_change().loc["2013-01-02":g.END].reindex(house.index)
    a, b = g.stats(pm, lab.rf), g.stats(house, lab.rf)
    corr = float(pm.corr(house))
    d = (pm - house).abs()
    out[code] = {"run": runs[-1].parent.name, "engine": a, "research": b, "corr": corr, "max_abs_daily_diff": float(d.max()), "worst_day": str(d.idxmax().date()),
                 "cagr_diff_pp": (a["cagr"] - b["cagr"]) * 100, "dd_diff_pp": (a["dd"] - b["dd"]) * 100,
                 "accepted": bool(abs(a["cagr"] - b["cagr"]) <= 0.003 and abs(a["dd"] - b["dd"]) <= 0.01 and corr >= 0.995), "nan_days": int(pm.isna().sum())}
    print(code, json.dumps(out[code], default=str))
(OUTD / "cp_pm.json").write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
