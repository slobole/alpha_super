"""VIXM A1/A5/A7 data probe: which series, padding/staleness, reverse splits, state stats, gaps, liquidity."""
import os
import pickle
import numpy as np
import pandas as pd
import tc_common as c
from data.norgate_loader import load_price_timeseries

out = {}
# Raw index series straight from Norgate (ALLMARKETDAYS padding as the loader uses)
for sym in ("$VIX", "$VIX3M"):
    d = load_price_timeseries(sym, start_date_str="2006-01-01")
    close = d["Close"].astype(float)
    rep = close.eq(close.shift(1))
    out[sym] = {"first": str(d.index[0].date()), "last": str(d.index[-1].date()), "n": int(len(d)),
                "columns": list(d.columns), "repeated_close_days": int(rep.sum()),
                "repeated_close_days_since_2011": int(rep[rep.index >= "2011-03-01"].sum()),
                "repeated_dates_since_2011": [str(x.date()) for x in rep[rep & (rep.index >= "2011-03-01")].index[:20]],
                "open_eq_close_days": int((d["Open"] == d["Close"]).sum()) if "Open" in d else None}
# are repeated closes on the same dates for both (padding) or random?
v = load_price_timeseries("$VIX", start_date_str="2011-01-01")["Close"].astype(float)
v3 = load_price_timeseries("$VIX3M", start_date_str="2011-01-01")["Close"].astype(float)
xnys = c.xnys_sessions("2011-01-01", "2026-09-25")
out["vix_dates_not_xnys"] = [str(x.date()) for x in v.index.difference(xnys)][:10]
out["xnys_dates_missing_in_vix"] = [str(x.date()) for x in xnys.difference(v.index)][:10]
out["xnys_dates_missing_in_vix3m"] = [str(x.date()) for x in xnys.difference(v3.index)][:10]

df = c.load_vixm()
vx = df[("VIXM", "Close")].astype(float)
un = df[("VIXM", "Unadjusted Close")].astype(float)
k = (un / vx)
jumps = k[(k / k.shift(1) - 1).abs() > 0.05]
out["vixm_price_scale_changes"] = [{"date": str(d.date()), "k_before": float(k.shift(1).loc[d]), "k_after": float(k.loc[d]),
                                     "ratio": float(k.loc[d] / k.shift(1).loc[d])} for d in jumps.index]
out["vixm_unadj_close_last"] = float(un.iloc[-1])
out["vixm_adj_close_first"] = float(vx.iloc[0])
out["vixm_unadj_close_first"] = float(un.iloc[0])
vol = df[("VIXM", "Volume")].astype(float)
out["vixm_zero_volume_days_total"] = int((vol == 0).sum())
out["vixm_zero_volume_days_since_2023"] = int((vol[vol.index >= "2023-09-25"] == 0).sum())

# state statistics
s = c.vixm.VixmBackwardationStrategy()
sig = s.compute_signals(df)
st = sig[(c.vixm.VIX_SIGNAL_NAMESPACE_STR, c.vixm.STATE_FIELD_STR)].astype(float)
st = st[st.index >= "2011-02-28"]
ch = st.diff().abs() > 0
yrs = (st.index[-1] - st.index[0]).days / 365.25
out["state_share_backwardation"] = float(st.mean())
out["state_changes_total"] = int(ch.sum())
out["state_changes_per_year"] = float(ch.sum() / yrs)
st3 = st[st.index >= "2023-09-25"]
out["state_share_last3y"] = float(st3.mean())
out["state_changes_last3y_per_year"] = float((st3.diff().abs() > 0).sum() / 3.0)
# episode lengths
ep = []
run = 0
for x in st.to_numpy():
    if x == 1:
        run += 1
    elif run:
        ep.append(run); run = 0
if run:
    ep.append(run)
out["episodes"] = {"n": len(ep), "median_len": float(np.median(ep)), "share_len_1": float(np.mean(np.array(ep) == 1)),
                   "share_len_le_3": float(np.mean(np.array(ep) <= 3))}
# near-ties: |VIX/VIX3M - 1| < 0.5% at state decisions
ratio = sig[(c.vixm.VIX_SIGNAL_NAMESPACE_STR, c.vixm.TERM_RATIO_FIELD_STR)].astype(float)
out["share_days_ratio_within_0p5pct_of_1"] = float(((ratio - 1).abs() < 0.005).mean())

# overnight gap on VIXM entry fills: open(T+1)/close(T)
base = pickle.load(open(c.CACHE / "vixm_baseline.pkl", "rb"))
tx = base["tx"].copy(); tx["bar"] = pd.to_datetime(tx["bar"])
buys = tx[(tx["asset"] == "VIXM") & (tx["amount"] > 0)]
idx = df.index
gaps = []
for _, r in buys.iterrows():
    d = r["bar"]; p = idx[idx.get_loc(d) - 1]
    gaps.append(float(df.loc[d, ("VIXM", "Open")] / df.loc[p, ("VIXM", "Close")] - 1))
gaps = np.array(gaps)
out["vixm_entry_gap_open_vs_prior_close"] = {"n": int(len(gaps)), "median_pct": float(np.median(gaps) * 100),
                                             "p90_pct": float(np.quantile(gaps, 0.9) * 100),
                                             "max_pct": float(gaps.max() * 100), "mean_pct": float(gaps.mean() * 100)}
res = base["results"]
cash = res["cash"].astype(float); tv = res["total_value"].astype(float)
f = cash / tv
out["cash_frac"] = {"min": float(f.min()), "min_date": str(f.idxmin().date()), "share_days_neg": float((f < 0).mean()),
                    "mean_neg_when_neg": float(f[f < 0].mean()), "share_days_below_minus_2pct": float((f < -0.02).mean())}
# trades per year and one-way turnover (share of NAV)
tx["notional"] = (tx["amount"] * tx["price"]).abs()
navprev = tv.shift(1)
tx["frac"] = tx["notional"] / tx["bar"].map(navprev)
out["orders_per_year"] = float(len(tx) / yrs)
out["turnover_nav_per_year"] = float(tx["frac"].sum() / yrs)
div = base["div_ledger"]
out["dividends_net_total"] = float(div["net_dividend_cash_float"].sum()) if div is not None and len(div) else 0.0
out["dividend_assets"] = sorted(set(div["asset_str"])) if div is not None and len(div) else []
c.dump(out, "vixm/data_probe.json")
import json; print(json.dumps(out, indent=1, default=str)[:6000])
