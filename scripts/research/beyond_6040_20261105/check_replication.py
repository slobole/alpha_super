import time, json, sys
sys.path.insert(0, "scripts/research/beyond_6040_20261105")
import pandas as pd, numpy as np
from data import load
from engine import Params, run, fixed_mix, metrics
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30)
panel = load()
def show(name, res, s, e):
    m = metrics(res["ret"].loc[s:e], res["cash_daily"].loc[s:e], res["exposure"].loc[s:e], res["turnover"].loc[s:e])
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items() if k in ("cagr","vol","sharpe_raw","sharpe","max_dd","avg_exposure","pct_fully_invested","turnover_per_year")} | {"name": name}
rows = []
t0 = time.time()
for lag in (1, 2):
    for cost in (0.0, 5.0):
        p = Params(lag=lag, cost_bps=cost)
        r = run(panel, p)
        rows.append(show(f"B0 lag{lag} c{cost}", r, "1968-07-01", "2026-10-23") | {"span": "1968-2026"})
        rows.append(show(f"B0 lag{lag} c{cost}", r, "2005-01-01", "2026-10-23") | {"span": "2005-2026"})
        rows.append(show(f"B0 lag{lag} c{cost}", r, "2007-06-01", "2026-08-07") | {"span": "PAKAL 2007-06..2026-08"})
print("elapsed per run", (time.time()-t0)/4)
for nm, w in [("60/40", {"r_stock": .6, "r_bond": .4}), ("EW3", {"r_stock": 1/3, "r_bond": 1/3, "r_gold": 1/3}), ("stocks", {"r_stock": 1.0})]:
    r = fixed_mix(panel, w)
    rows.append(show(nm, r, "1968-07-01", "2026-10-23") | {"span": "1968-2026"})
    rows.append(show(nm, r, "2005-01-01", "2026-10-23") | {"span": "2005-2026"})
df = pd.DataFrame(rows).set_index(["name","span"])
print(df.to_string())
