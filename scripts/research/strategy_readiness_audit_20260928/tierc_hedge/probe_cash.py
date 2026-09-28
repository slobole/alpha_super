import pickle, pandas as pd, numpy as np
import tc_common as c
for name in ("vixm","ctc","trin"):
    b = pickle.load(open(c.CACHE/f"{name}_baseline.pkl","rb"))
    r = b["results"]
    cash = r["cash"].astype(float); tv = r["total_value"].astype(float)
    frac = cash/tv
    print(name, "min cash frac", frac.min(), frac.idxmin(), "share days cash<0", (cash< -1e-6).mean(), "median cash frac", frac.median())
    d = frac.idxmin()
    tx = b["tx"]; tx["bar"]=pd.to_datetime(tx["bar"])
    print(tx[tx["bar"].between(d-pd.Timedelta(days=3), d)].to_string()[:3000])
