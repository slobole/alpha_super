import pickle, sys, json
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
import g_lib as g
p = g.WT_REPO / "results/research/portfolio/fund_growth/vanilla_backtest/2026-10-05_084534/fund_growth.pkl"
pf = pickle.load(open(p, "rb"))
for a in dir(pf):
    if "weight" in a.lower() or "pod" in a.lower() or "config" in a.lower():
        try: print(a, str(getattr(pf, a))[:600])
        except Exception as e: print(a, "ERR", e)
tv = pf.results["total_value"].astype(float); tv.index = pd.to_datetime(tv.index).normalize()
pm = tv.pct_change().loc["2013-01-02":g.END]
lab = g.Lab(); fr = lab.frames["s1_house_cash"][0]
w = pd.Series({"taa3x": .40, "ndx_atr_cap": .15, "ndx_natr_cap": .15, "dv2_g": .15, "hpi_g": .15})
df = fr.loc[g.LONG_START:g.END, list(w.index)]; prev = 1.0; out = []
for y, blk in df.groupby(df.index.year):
    lvl = ((1 + blk).cumprod() * w).sum(axis=1) * prev; out.append(lvl); prev = float(lvl.iloc[-1])
lvl = pd.concat(out); r = (lvl / lvl.shift(1).fillna(1.0) - 1).loc["2013-01-02":]
pm = pm.reindex(r.index)
c = lambda x: float(np.prod(1 + x.to_numpy()) ** (252 / len(x)) - 1)
dd = lambda x: float(((1 + x).cumprod() / np.maximum((1 + x).cumprod().cummax(), 1) - 1).min())
print("engine", c(pm), dd(pm), "own house", c(r), dd(r), "corr", pm.corr(r), "maxdiff", float((pm - r).abs().max()), "n", len(r), "nan", int(pm.isna().sum()))
