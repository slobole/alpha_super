"""VIXM: where does the edge live? Frictionless analytic timing decomposition of the one-hot VIXM/SHY rule.

r(fill convention) for the state decided at Close_T:
  next_open     : Open_(T+1) -> Open_(T+2)   (the backtest's convention)
  next_close    : Close_(T+1) -> Close_(T+2) (MOC one session later)
  day1_intraday : VIXM Open_(T+1) -> Close_(T+1) on entry days (mean, bp)
Prices are CAPITALSPECIAL; SHY dividends ignored (same in all conventions).
"""
import numpy as np
import pandas as pd
import tc_common as c

df = c.load_vixm()
NS = c.vixm.VIX_SIGNAL_NAMESPACE_STR
st = (df[(NS, "vix_close_float")] > df[(NS, "vix3m_close_float")]).astype(float)
o = {a: df[(a, "Open")].astype(float) for a in ("VIXM", "SHY")}
cl = {a: df[(a, "Close")].astype(float) for a in ("VIXM", "SHY")}


def cagr(r):
    r = r.dropna()
    r = r[r.index >= "2011-03-01"]
    return float(((1 + r).prod()) ** (252 / len(r)) - 1) * 100


out = {}
# open-to-open: position decided at T held from Open_(T+1) to Open_(T+2); stamp the return at T+2
oo = {a: o[a].shift(-1) / o[a] - 1 for a in o}  # stamped at T+1: Open_(T+1)->Open_(T+2)
w = st.shift(1)  # state decided at T applies at T+1
r_open = w * oo["VIXM"] + (1 - w) * oo["SHY"]
out["next_open_cagr_frictionless"] = cagr(r_open)
cc = {a: cl[a].shift(-1) / cl[a] - 1 for a in cl}  # Close_(T+1)->Close_(T+2) stamped at T+1
r_close = w * cc["VIXM"] + (1 - w) * cc["SHY"]
out["next_close_moc_cagr_frictionless"] = cagr(r_close)
w2 = st.shift(2)
out["next_open_one_extra_session_lag_cagr_frictionless"] = cagr(w2 * oo["VIXM"] + (1 - w2) * oo["SHY"])
out["shy_only_cagr_price"] = cagr(oo["SHY"])
out["vixm_buy_hold_cagr"] = cagr(oo["VIXM"])
# day-1 intraday on entry days
entry = (st.shift(1) == 1) & (st.shift(2) == 0)
intraday = (cl["VIXM"] / o["VIXM"] - 1)
overnight_next = (o["VIXM"].shift(-1) / cl["VIXM"] - 1)
out["entry_days"] = int(entry[entry.index >= "2011-03-01"].sum())
out["entry_day_vixm_open_to_close_mean_bp"] = float(intraday[entry].mean() * 1e4)
out["entry_day_vixm_open_to_close_median_bp"] = float(intraday[entry].median() * 1e4)
out["entry_day_vixm_close_to_next_open_mean_bp"] = float(overnight_next[entry].mean() * 1e4)
out["all_days_vixm_open_to_close_mean_bp"] = float(intraday.mean() * 1e4)
# how much of the frictionless open-to-open P&L of VIXM legs comes from day 1 of an episode
hold = w == 1
day1 = hold & (w.shift(1) != 1)
lr = np.log1p(oo["VIXM"])
out["vixm_log_pnl_share_day1"] = float(lr[day1].sum() / lr[hold].sum()) if lr[hold].sum() != 0 else None
out["vixm_log_pnl_day1_sum"] = float(lr[day1].sum())
out["vixm_log_pnl_later_days_sum"] = float(lr[hold & ~day1].sum())
# open-print sanity: share of VIXM days with Open == prior Close
out["vixm_open_eq_prev_close_share"] = float((o["VIXM"] == cl["VIXM"].shift(1)).mean())
out["vixm_open_eq_high_or_low_share"] = float(((o["VIXM"] == df[("VIXM", "High")]) | (o["VIXM"] == df[("VIXM", "Low")])).mean())
c.dump(out, "vixm/timing_decomposition.json")
import json; print(json.dumps(out, indent=1))
