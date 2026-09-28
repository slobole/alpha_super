"""A8: commission on adjusted-unit shares (model) vs raw shares actually traded at the fill date (current vintage).
Also borrow collateral on adjusted units vs raw. Bound on CAGR via fee difference / mean NAV / years."""
import pickle, json
import numpy as np, pandas as pd
import core5_common as c
df = c.load_pricing()
with (c.CACHE / "baseline_strategy.pkl").open("rb") as fh:
    base = pickle.load(fh)
tx = base["tx"].copy(); tx["bar"] = pd.to_datetime(tx["bar"])
k = np.array([float(df.loc[b, (a, "Unadjusted Close")]) / float(df.loc[b, (a, "Close")]) for b, a in zip(tx["bar"], tx["asset"])])
raw = tx["amount"].astype(float).abs().to_numpy() / k
fee_raw = np.maximum(1.0, 0.005 * raw)
tx["fee_raw"] = fee_raw
by = tx.groupby("asset")[["commission", "fee_raw"]].sum().round(2)
tv = base["results"]["total_value"].astype(float)
years = (tv.index[-1] - tv.index[0]).days / 365.25
d = float(tx["commission"].sum() - fee_raw.sum())
out = {"fee_model_total": float(tx["commission"].sum()), "fee_raw_total": float(fee_raw.sum()), "model_minus_raw": d,
       "approx_cagr_bias_pp": 100 * d / tv.mean() / years, "by_asset": by.to_dict(orient="index")}
print(json.dumps(out, indent=1)); c.dump(out, "a8_fee_units.json")
