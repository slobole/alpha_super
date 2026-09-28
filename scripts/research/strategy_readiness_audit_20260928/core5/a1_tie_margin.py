"""A1/B4: how close do SMA10 and AMA come (relative gap) on decision days? A float32 re-adjustment of Norgate's
TOTALRETURN history (new dividend) perturbs both by ~1e-7 relative; a gap below that could flip a committed
T-1 long state and trip the adapter's 'historical revision' halt. Also: cash-at-0% conservative bound."""
import pickle, json
import numpy as np, pandas as pd
import core5_common as c
core5 = c.core5
with (c.CACHE / "baseline_strategy.pkl").open("rb") as fh:
    base = pickle.load(fh)
diag = base["signal_diag"]
cal_start = pd.Timestamp("2007-08-31")
out = {}
for a in core5.RISK_ASSET_TUPLE:
    g = (diag[f"{a}_filtered_price_ser"] / diag[f"{a}_adaptive_moving_average_ser"] - 1.0).abs()
    g = g[g.index >= cal_start].dropna()
    out[a] = {"n_days": int(len(g)), "min_rel_gap": float(g.min()), "n_gap_lt_1e-6": int((g < 1e-6).sum()),
              "n_gap_lt_1e-5": int((g < 1e-5).sum()), "n_gap_lt_1e-4": int((g < 1e-4).sum()),
              "min_gap_date": str(g.idxmin().date())}
res = base["results"]
cash = res["cash"].astype(float); tv = res["total_value"].astype(float)
out["cash_weight_mean_pct"] = float(100 * (cash / tv).mean())
out["cash_weight_positive_mean_pct"] = float(100 * (cash.clip(lower=0) / tv).mean())
print(json.dumps(out, indent=1))
c.dump(out, "a1_tie_margin.json")
