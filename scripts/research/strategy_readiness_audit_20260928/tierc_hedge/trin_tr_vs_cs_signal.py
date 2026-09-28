"""Trinity A1/A8 role check: the vol signals use CAPITALSPECIAL (price) returns, so ex-dividend drops enter the
63-day volatility. Measure the desired-exposure difference vs TOTALRETURN returns (design sensitivity, not a leak)."""
import numpy as np
import pandas as pd
import tc_common as c
from data.norgate_loader import load_price_timeseries

full = c.load_trin()
risk = list(c.trin.RISK_ASSET_TUPLE)
s = c.trin.TrinityVolControlStrategy(name="x", benchmarks=["$SPX"])
sig = s.compute_signals(full)
tr = pd.DataFrame({a: load_price_timeseries(a, adjustment_str="TOTALRETURN", start_date_str="1995-01-01")["Close"] for a in risk}).reindex(full.index)
tr_ret = tr.pct_change(fill_method=None)
cs_ret = pd.DataFrame({a: sig[(a, "return_ser")] for a in risk})
bw = pd.DataFrame({a: sig[(a, "base_weight_ser")] for a in risk})
days = bw.dropna().index
days = days[days >= "2007-06-01"]
rows = []
for T in days:
    w = bw.loc[T]
    m_cs = c.b6040.compute_gross_exposure_float((cs_ret.loc[:T].iloc[-63:] * w).sum(axis=1), 63, 0.08, 0.085)
    m_tr = c.b6040.compute_gross_exposure_float((tr_ret.loc[:T].iloc[-63:] * w).sum(axis=1), 63, 0.08, 0.085)
    rows.append((T, m_cs, m_tr))
d = pd.DataFrame(rows, columns=["T", "m_cs", "m_tr"]).set_index("T")
diff = (d["m_cs"] - d["m_tr"]).abs()
out = {"days": int(len(d)), "mean_abs_m_diff_pp": float(diff.mean() * 100), "max_abs_m_diff_pp": float(diff.max() * 100),
       "share_days_diff_gt_1pp": float((diff > 0.01).mean()), "share_days_diff_ge_5pp": float((diff >= 0.05).mean()),
       "mean_m_cs": float(d["m_cs"].mean()), "mean_m_tr": float(d["m_tr"].mean()),
       "share_days_m_lt_1": float((d["m_cs"] < 1).mean())}
c.dump(out, "trin/a1_tr_vs_cs_vol_signal.json")
print(out)
