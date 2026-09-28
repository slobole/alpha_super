"""CTC A1 probes: last-row month-end flag, band-trigger frequency, reserve/eligibility timeline."""
import pickle
import numpy as np
import pandas as pd
import tc_common as c

df = c.load_ctc_workaround()
s = c.ctc.CrisisTrendCoreStrategy()
sig = s.compute_signals(df)
me = sig[(c.ctc.PORTFOLIO_NAMESPACE_STR, c.ctc.MONTH_END_FIELD_STR)].astype(bool)
risk = list(c.ctc.UNIVERSE_ASSET_TUPLE)
des = pd.DataFrame({a: sig[(c.ctc.signal_namespace_str(a), c.ctc.DESIRED_WEIGHT_FIELD_STR)] for a in risk})
last = df.index[-1]
prev_me = me[me & (me.index < last)].index[-1]
out = {"last_row": str(last.date()), "last_row_flagged_month_end": bool(me.loc[last]),
       "previous_true_month_end": str(prev_me.date()),
       "last_row_target_vs_true_month_end_target_max_abs_diff": float((des.loc[last] - des.loc[prev_me]).abs().max()),
       "last_row_target_vs_true_month_end_gross_diff": float((des.loc[last] - des.loc[prev_me]).abs().sum())}
base = pickle.load(open(c.CACHE / "ctc_baseline.pkl", "rb"))
rb = base["rebal"]
out["last_recorded_rebalance_decision"] = str(rb.index[-1].date())
xnys = c.xnys_sessions("2003-01-01", "2026-12-31")
me_x = pd.Series(xnys, index=xnys).groupby(xnys.to_period("M")).max()
me_set = set(me_x.tolist())
rbd = pd.DatetimeIndex(rb.index)
out["rebalance_decisions"] = int(len(rbd))
out["rebalance_decisions_on_month_end"] = int(sum(d in me_set for d in rbd))
out["rebalance_decisions_mid_month_band"] = int(sum(d not in me_set for d in rbd))
# first eligible / first non-zero target
elig = pd.DataFrame({a: sig[(c.ctc.signal_namespace_str(a), c.ctc.ELIGIBLE_FIELD_STR)] for a in risk}).astype(bool)
out["first_eligible_by_asset"] = {a: str(elig[a].idxmax().date()) if elig[a].any() else None for a in risk}
nz = des.abs().sum(axis=1) > 0
out["first_nonzero_target"] = str(nz.idxmax().date())
# data start per asset in the HEAD loader frame (2002-01-01)
full = c.load_ctc()
out["first_valid_tr_close"] = {a: str(full[(c.ctc.signal_namespace_str(a), "Close")].first_valid_index().date()) for a in c.ctc.TRADEABLE_ASSET_TUPLE}
c.dump(out, "ctc/a1_design_probe.json")
import json; print(json.dumps(out, indent=1))
