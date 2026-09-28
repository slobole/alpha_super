"""D-check: how much of the live NDX score is nominal share price?

score_i = ROC12_i / ATR20$_i with ATR20$_i = NATR20_i * P_i (P = nominal decision close), so
log score = log ROC - log NATR - log P for positive ROC. We report, at each of the last 60
decisions and across eligible positive-ROC members, the cross-sectional variance share of
log P and the Spearman correlation of the score rank with 1/P, and compare the selected set with
the NATR (price-free) ranking of the same eligible pool.
"""
from __future__ import annotations
import json, sys
from dataclasses import replace
from pathlib import Path
import numpy as np, pandas as pd
REPO = Path(__file__).resolve().parents[4]; sys.path.insert(0, str(REPO)); sys.path.insert(0, str(Path(__file__).resolve().parent))
import strategies.momentum.strategy_mo_atr_normalized_ndx as atr_module
import strategies.momentum.strategy_mo_atr_normalized_ndx_vxn_scaled as vxn_module
from ndx_live_parity_replay import _cached_build_universe
atr_module.build_index_constituent_matrix = _cached_build_universe
cfg = replace(vxn_module.DEFAULT_CONFIG, end_date_str="2026-09-25")
px, uni, sched, vxn = vxn_module.get_vxn_scaled_atr_normalized_ndx_data(cfg)
s = vxn_module.VxnScaledAtrNormalizedNdxStrategy(name="a", benchmarks=["$SPX"], rebalance_schedule_df=sched, vxn_scale_signal_df=vxn)
s.universe_df = uni
sig = s.compute_signals(px.copy())
rows = []
for d in pd.DatetimeIndex(sched["decision_date_ts"].iloc[-60:]):
    s.previous_bar = d
    ranked = s.get_ranked_candidate_feature_df(sig.loc[d])
    if len(ranked) < 15:
        continue
    syms = ranked.index.tolist()
    P = np.array([float(px.loc[d, (x, "Unadjusted Close")]) for x in syms])
    roc = ranked["monthly_roc_12_ser"].astype(float).to_numpy()
    atr = ranked["atr_20_ser"].astype(float).to_numpy()
    natr = atr / P
    pos = roc > 0
    lp, ls = np.log(P[pos]), np.log(roc[pos] / atr[pos])
    var_share = float(np.var(lp) / np.var(ls)) if np.var(ls) > 0 else np.nan
    corr_invp = float(pd.Series(roc / atr).rank().corr(pd.Series(1 / P).rank()))
    natr_rank = pd.Series(roc / natr, index=syms).sort_values(ascending=False)
    sel_live = set(syms[:10]); sel_natr = set(natr_rank.index[:10])
    rows.append({"decision": d.date().isoformat(), "eligible": len(syms), "logP_var_over_logscore_var": var_share,
                 "spearman_score_vs_invprice": corr_invp, "overlap_live_vs_price_free_top10": len(sel_live & sel_natr),
                 "median_price_selected": float(np.median(P[:10])), "median_price_eligible": float(np.median(P))})
df = pd.DataFrame(rows)
out = REPO / "results/research/strategy_readiness_audit_20260928/ndx/nominal_price_dependence.csv"
df.to_csv(out, index=False)
print(df.describe().T[["mean", "50%", "min", "max"]].to_string())
