"""Review (tradability lens): rerun the four monthly pods as published and save the engine ledgers
(transactions with commission, daily NAV) so that commission drag can be re-priced at owner size.
Read-only on production code. Output: results/.../review_trade/ledgers/<key>_{tx,nav}.csv
"""
from __future__ import annotations
import sys
from dataclasses import replace as dataclass_replace
from importlib import import_module
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
OUT = REPO / "results/research/strategy_readiness_audit_20260928/review_trade/ledgers"
OUT.mkdir(parents=True, exist_ok=True)
END = "2026-09-25"
MODULES = {
    "taa3x": "strategies.taa_df.strategy_taa_df_btal_fallback_tqqq_vix_cash",
    "taa1n": "strategies.taa_df.strategy_taa_df_btal_1n_fallback_tqqq_vix_cash",
    "btal_qqq": "strategies.taa_df.strategy_taa_df_btal_linearity_1n_fallback_qqq_vix_cash",
    "ndx_vxn": "strategies.momentum.strategy_mo_atr_normalized_ndx_vxn_scaled",
}
for key in sys.argv[1:] or list(MODULES):
    mod = import_module(MODULES[key])
    if hasattr(mod.DEFAULT_CONFIG, "dtb3_csv_path_str"):
        mod.DEFAULT_CONFIG = dataclass_replace(mod.DEFAULT_CONFIG, dtb3_csv_path_str=str(OUT / "DTB3_review_cache.csv"))
    strat = mod.run_variant(show_display_bool=False, save_results_bool=False, end_date_str=END)
    strat.get_transactions().to_csv(OUT / f"{key}_tx.csv", index=False)
    strat.total_value_series.rename("nav").to_csv(OUT / f"{key}_nav.csv")
    print(key, "final", float(strat.total_value), "fills", len(strat.get_transactions()), flush=True)
