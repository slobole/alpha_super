"""Reviewer scratch (report lens): BIL pod vs BIL total return; $1M MR capsule vs the stored capsule. Read-only."""
import json
from pathlib import Path
import numpy as np, pandas as pd
WT = Path(__file__).resolve().parents[5]
ST = WT / "results/research/portfolio/fund_products_20261005"
A = ST / "audit"
def path(p):
    d = pd.read_csv(p, parse_dates=["date"]).set_index("date")
    return d
def cagr(nav, lo, hi):
    n = nav.loc[lo:hi]
    return float((n.iloc[-1] / n.iloc[0]) ** (252 / (len(n) - 1)) - 1)
for name in ("bil_pod", "tbill"):
    d = path(A / "rerun_taa_def" / f"{name}__path.csv.gz")
    nav = d["total_value_float"]
    print(name, "first", nav.index[0].date(), "last", nav.index[-1].date(), "CAGR 2012-10-02..2026-08-19:", round(cagr(nav, "2012-10-02", "2026-08-19"), 5),
          "| 2012-01-03..2026-08-19:", round(cagr(nav, "2012-01-03", "2026-08-19"), 5), "| full:", round(cagr(nav, nav.index[0], nav.index[-1]), 5))
    md = A / "rerun_taa_def" / f"{name}__metadata.json"
    if md.exists():
        m = json.loads(md.read_text(encoding="utf-8")); print("   module:", m.get("strategy_import_str") or m.get("module_path_str"))
# $1M MR capsule vs stored $100K capsule
old = pd.read_csv(A / "rerun_mr" / "mr_capsule_daily.csv", index_col=0, parse_dates=True)
new = {}
for a in ("dv2_g", "hpi_g"):
    nav = path(ST / "sources" / f"{a}__path.csv.gz")["total_value_float"]
    new[a] = nav.pct_change()
caps_new = (0.5 * new["dv2_g"] + 0.5 * new["hpi_g"]).dropna()
# book-model capsule (pods compound, annual reset) for both
def book(df):
    out = []; idx = df.index; val = 1.0
    for y, blk in df.groupby(idx.year):
        pv = (1 + blk).cumprod() * 0.5
        lvl = val * pv.sum(axis=1)
        out.append(lvl); val = float(lvl.iloc[-1])
    lv = pd.concat(out); return lv.pct_change().fillna(lv.iloc[0] - 1)
bn = book(pd.DataFrame(new).dropna())
bo = book(old[["dv2_bil_ret", "hpi_bil_ret"]].dropna())
j = pd.concat([bn.rename("n"), bo.rename("o"), old["capsule_bil_ret"].rename("stored")], axis=1).dropna()
for lab, lo, hi in (("full", j.index[0], j.index[-1]), ("LONG", "2008-03-04", "2026-08-19")):
    x = j.loc[lo:hi]
    cg = lambda s: float((1 + s).prod() ** (252 / len(s)) - 1)
    print(lab, "new $1M capsule CAGR", round(cg(x["n"]), 5), "stored-pod capsule", round(cg(x["o"]), 5), "stored capsule_bil_ret", round(cg(x["stored"]), 5),
          "corr new vs stored", round(float(x["n"].corr(x["stored"])), 5), "max abs daily diff", round(float((x["n"] - x["stored"]).abs().max()), 5))
