"""Read-only: schema of the sleeve source files the v4 pipeline reads (MAIN shelf_rebuild sources) + tiers."""
import glob
import json
import sys

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
S = r"C:/Users/User/Documents/workspace/alpha_super/results/research/portfolio/shelf_rebuild_20260929/sources"
p = pd.read_csv(S + "/ndx_vxn__path.csv.gz", index_col="date", parse_dates=True)
print("path cols", list(p.columns), p.index[0].date(), p.index[-1].date(), len(p))
t = pd.read_csv(S + "/ndx_vxn__transactions.csv.gz", parse_dates=["date"])
print("tx cols", list(t.columns), len(t))
print(t.head(2).to_string())
for f in sorted(glob.glob(S + "/*__metadata.json")):
    m = json.load(open(f, encoding="utf-8"))
    print(f"{m['alias_str']:20s} | {m.get('tier_str'):9s} | first_invested {m.get('first_invested_date_str')} | end {m.get('end_date_str')} | {m.get('strategy_import_str')}")
