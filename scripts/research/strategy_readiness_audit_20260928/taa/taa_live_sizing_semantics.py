"""A-LIVE-06 quantification for TAA: backtest sizes target shares from Close_T; live sizes from the
opening-auction price (approximated by Open_(T+1)) with the pre-open NAV (approximated by NAV at
Close_T). Replica: override the engine's sizing price with the current open. Study code only."""
from __future__ import annotations
import json, sys
from dataclasses import replace
from importlib import import_module
from pathlib import Path
import numpy as np, pandas as pd
REPO = Path(__file__).resolve().parents[4]; sys.path.insert(0, str(REPO))
from alpha.engine.strategy import Strategy
OUT = REPO / "results/research/strategy_readiness_audit_20260928/taa"
def metrics(nav):
    nav = nav.astype(float); r = nav.pct_change().dropna(); y = len(r) / 252
    return {"cagr": float((nav.iloc[-1] / nav.iloc[0]) ** (1 / y) - 1), "sharpe": float(r.mean() / r.std() * np.sqrt(252)), "max_dd": float((nav / nav.cummax() - 1).min()), "min_cash_frac": None}
res = {}
real_fn = Strategy._get_order_sizing_price_float
for key, mod in [("taa3x", "strategies.taa_df.strategy_taa_df_btal_fallback_tqqq_vix_cash"), ("taa1n", "strategies.taa_df.strategy_taa_df_btal_1n_fallback_tqqq_vix_cash"), ("btal_qqq", "strategies.taa_df.strategy_taa_df_btal_linearity_1n_fallback_qqq_vix_cash")]:
    m = import_module(mod); m.DEFAULT_CONFIG = replace(m.DEFAULT_CONFIG, dtb3_csv_path_str=str(OUT / "DTB3_audit_cache.csv"))
    out = {}
    for label, patch in (("backtest_close_sizing", False), ("live_open_sizing", True)):
        Strategy._get_order_sizing_price_float = (lambda self, prices, asset_str, current_open_float: float(current_open_float)) if patch else real_fn
        try:
            s = m.run_variant(show_display_bool=False, save_results_bool=False, end_date_str="2026-09-25")
        finally:
            Strategy._get_order_sizing_price_float = real_fn
        nav = s.results["total_value"].astype(float); mt = metrics(nav)
        mt["min_cash_frac"] = float((s.results["cash"].astype(float) / nav).min())
        out[label] = mt
    res[key] = out; print(key, json.dumps(out), flush=True)
(OUT / "live_sizing_semantics.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
