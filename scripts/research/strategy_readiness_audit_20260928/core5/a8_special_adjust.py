"""A8: CAPITALSPECIAL adjustment events (k = Unadjusted/Adjusted changes) vs Dividend rows: any double count?
Also: dividend-ledger rows paid by a DBC short in the baseline."""
import json, pickle
import numpy as np, pandas as pd
import core5_common as c
df = c.load_pricing()
out = {}
for sym in c.TRADEABLES:
    k = (df[(sym, "Unadjusted Close")].astype(np.float64) / df[(sym, "Close")].astype(np.float64)).dropna()
    chg = k.pct_change().abs() > 2e-4
    ev = k.index[chg.fillna(False).to_numpy()]
    div = df[(sym, "Dividend")].astype(float)
    rows = []
    for d in ev:
        prev = df.index[df.index.get_loc(d) - 1]
        rows.append({"date": str(d.date()), "k_before": round(float(k.loc[prev]), 5), "k_after": round(float(k.loc[d]), 5),
                     "ratio": round(float(k.loc[d] / k.loc[prev]), 5),
                     "dividend_on_prev_row": float(div.loc[prev]) if np.isfinite(div.loc[prev]) else None,
                     "unadj_close_prev": float(df.loc[prev, (sym, "Unadjusted Close")])})
    out[sym] = rows
with (c.CACHE / "baseline_strategy.pkl").open("rb") as fh:
    base = pickle.load(fh)
dl = base["div"]
out["dividend_ledger_short_rows"] = dl[dl["position_share_float"] < 0].astype(str).to_dict("records")
out["dividend_ledger_by_asset_gross"] = dl.groupby("asset_str")["gross_dividend_cash_float"].sum().round(2).to_dict()
print(json.dumps(out, indent=1))
c.dump(out, "a8_special_adjust.json")
