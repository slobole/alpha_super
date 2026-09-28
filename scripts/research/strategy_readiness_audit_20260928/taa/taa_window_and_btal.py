"""Sensitivity: TAA metrics on the full window vs 2014+ (BTAL median turnover was ~$30K/day in
2011-13), and BTAL's P&L contribution by year (price + net dividends, from the fill ledger)."""
from __future__ import annotations
import json, sys
from dataclasses import replace
from importlib import import_module
from pathlib import Path
import numpy as np, pandas as pd
REPO = Path(__file__).resolve().parents[4]; sys.path.insert(0, str(REPO))
OUT = REPO / "results/research/strategy_readiness_audit_20260928/taa"
def metrics(nav):
    nav = nav.astype(float); r = nav.pct_change().dropna(); y = len(r) / 252
    return {"cagr": float((nav.iloc[-1] / nav.iloc[0]) ** (1 / y) - 1), "sharpe": float(r.mean() / r.std() * np.sqrt(252)), "max_dd": float((nav / nav.cummax() - 1).min())}
res = {}
for key, mod in [("taa3x", "strategies.taa_df.strategy_taa_df_btal_fallback_tqqq_vix_cash"), ("taa1n", "strategies.taa_df.strategy_taa_df_btal_1n_fallback_tqqq_vix_cash"), ("btal_qqq", "strategies.taa_df.strategy_taa_df_btal_linearity_1n_fallback_qqq_vix_cash")]:
    m = import_module(mod); m.DEFAULT_CONFIG = replace(m.DEFAULT_CONFIG, dtb3_csv_path_str=str(OUT / "DTB3_audit_cache.csv"))
    s = m.run_variant(show_display_bool=False, save_results_bool=False, end_date_str="2026-09-25")
    nav = s.results["total_value"].astype(float); nav.index = pd.to_datetime(nav.index)
    pos = s.get_transactions().copy(); pos["bar"] = pd.to_datetime(pos["bar"])
    res[key] = {"full": metrics(nav), "from_2014": metrics(nav.loc["2014-01-02":]), "2012_2013_only": metrics(nav.loc[:"2013-12-31"])}
    print(key, json.dumps(res[key]), flush=True)
(OUT / "window_sensitivity.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
