"""VIXM A5 (signal timing) and cost sensitivity.

1. Causal replay with one extra session of lag: the state at Close_T uses VIX/VIX3M closes of T-1 (engine run).
2. One-day-stale single helper: state flips if only VIX3M (or only VIX) is one session stale.
3. VIXM-leg slippage sensitivity: +15 bp and +40 bp per side on VIXM fills only (analytic on the baseline ledger).
"""
import pickle
import numpy as np
import pandas as pd
import tc_common as c

df = c.load_vixm()
NS = c.vixm.VIX_SIGNAL_NAMESPACE_STR
out = {}


class Lag1(c.vixm.VixmBackwardationStrategy):
    def compute_signals(self, pricing_data_df):
        pdf = pricing_data_df.copy()
        for f in ("vix_close_float", "vix3m_close_float"):
            pdf[(NS, f)] = pdf[(NS, f)].shift(1)
        return super().compute_signals(pdf)


base = pickle.load(open(c.CACHE / "vixm_baseline.pkl", "rb"))
tv0 = base["results"]["total_value"].astype(float)
r0 = tv0.pct_change().dropna()
cfg_start = "2011-03-02"
s1 = c.run_vixm(df, start=cfg_start, strategy_cls=Lag1)
r1 = c.strat_returns(s1)
common = r0.index.intersection(r1.index)
out["lag1_extra_session"] = {"base": c.metrics(r0.loc[common]), "lag1": c.metrics(r1.loc[common]),
                             "base_last3y": c.metrics(r0.loc[common], start="2023-09-25"),
                             "lag1_last3y": c.metrics(r1.loc[common], start="2023-09-25")}
out["lag1_extra_session"]["cagr_delta_pp"] = out["lag1_extra_session"]["lag1"]["cagr_pct"] - out["lag1_extra_session"]["base"]["cagr_pct"]

v = df[(NS, "vix_close_float")].astype(float)
v3 = df[(NS, "vix3m_close_float")].astype(float)
st = (v > v3).astype(int)
st_v3stale = (v > v3.shift(1)).astype(int)
st_vstale = (v.shift(1) > v3).astype(int)
m = st.index >= "2011-03-01"
out["stale_single_helper_flips"] = {
    "days": int(m.sum()),
    "vix3m_stale_flips": int((st[m] != st_v3stale[m]).sum()),
    "vix_stale_flips": int((st[m] != st_vstale[m]).sum()),
    "vix3m_stale_flips_last3y": int((st[st.index >= "2023-09-25"] != st_v3stale[st.index >= "2023-09-25"]).sum()),
}

tx = base["tx"].copy()
tx["bar"] = pd.to_datetime(tx["bar"])
navprev = tv0.shift(1)
vx = tx[tx["asset"] == "VIXM"].copy()
vx["frac"] = (vx["amount"] * vx["price"]).abs() / vx["bar"].map(navprev)
for extra_bp in (15, 40):
    cost = vx.groupby("bar")["frac"].sum() * extra_bp / 1e4
    r = r0 - cost.reindex(r0.index).fillna(0.0)
    out[f"vixm_leg_plus_{extra_bp}bp"] = {"cagr_pct": c.metrics(r)["cagr_pct"],
                                          "delta_pp": c.metrics(r)["cagr_pct"] - c.metrics(r0)["cagr_pct"],
                                          "delta_pp_last3y": c.metrics(r, start="2023-09-25")["cagr_pct"]
                                          - c.metrics(r0, start="2023-09-25")["cagr_pct"]}
out["vixm_leg_nav_turnover_per_year"] = float(vx["frac"].sum() / ((tv0.index[-1] - tv0.index[0]).days / 365.25))
# Daily range proxy for spread cost: median (High-Low)/Close for VIXM last 3y, and share of days High==Low
rec = df[df.index >= "2023-09-25"]
out["vixm_median_daily_range_pct_last3y"] = float(((rec[("VIXM", "High")] - rec[("VIXM", "Low")]) / rec[("VIXM", "Close")]).median() * 100)
c.dump(out, "vixm/a5_lag_and_cost.json")
import json; print(json.dumps(out, indent=1, default=str))
