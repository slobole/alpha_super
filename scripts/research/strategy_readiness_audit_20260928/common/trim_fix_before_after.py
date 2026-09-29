"""Fix #7 evidence: exact membership (new default) vs the legacy 5-session tail trim.

Runs each module's own run_variant twice: new default (exact) and legacy (trim patched back in
through the loader flag). Reports full-history and last-3-year metrics and the fills that differ.
Expected, from the audit reviewers' untrimmed arms: NDX VXN full about +0.004 pp, last 3y about
-0.47 pp; DV2 full about +0.02 pp, last 3y about -0.22 pp.
"""

from __future__ import annotations

import functools
import json
import sys
from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT_PATH = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT_PATH))

import data.norgate_loader as loader_module  # noqa: E402

OUTPUT_PATH = REPO_ROOT_PATH / "results/research/strategy_readiness_audit_20260928/trim_fix_before_after.json"
END_DATE_STR = "2026-09-25"
LAST3Y_START_STR = "2023-09-25"

MODULE_DICT = {
    "ndx_vxn": "strategies.momentum.strategy_mo_atr_normalized_ndx_vxn_scaled",
    "ndx_atr": "strategies.momentum.strategy_mo_atr_normalized_ndx",
    "dv2": "strategies.dv2.strategy_mr_dv2",
}


def _metrics(nav_ser: pd.Series) -> dict:
    out = {}
    for label_str, window_ser in (("full", nav_ser), ("last3y", nav_ser.loc[LAST3Y_START_STR:])):
        ret_ser = window_ser.pct_change().dropna()
        years_float = len(ret_ser) / 252.0
        out[label_str] = {
            "cagr_pct": round(((window_ser.iloc[-1] / window_ser.iloc[0]) ** (1 / years_float) - 1) * 100, 3),
            "sharpe": round(float(ret_ser.mean() / ret_ser.std() * np.sqrt(252)), 4),
            "max_dd_pct": round(float((window_ser / window_ser.cummax() - 1).min()) * 100, 2),
        }
    return out


def _run(key_str: str, legacy_trim_bool: bool):
    module_obj = import_module(MODULE_DICT[key_str])
    real_fn = loader_module.build_index_constituent_matrix
    patched_fn = functools.partial(real_fn, trim_past_member_tail_bool=legacy_trim_bool)
    patch_target_list = [m for m in (module_obj, import_module("strategies.momentum.strategy_mo_atr_normalized_ndx"))
                         if hasattr(m, "build_index_constituent_matrix")]
    original_dict = {id(m): m.build_index_constituent_matrix for m in patch_target_list}
    for m in patch_target_list:
        m.build_index_constituent_matrix = patched_fn
    try:
        if key_str == "dv2":
            strategy_obj = module_obj.run_variant(show_display_bool=False, save_results_bool=False, end_date_str=END_DATE_STR) \
                if "end_date_str" in module_obj.run_variant.__code__.co_varnames else \
                module_obj.run_variant(show_display_bool=False, save_results_bool=False)
        else:
            strategy_obj = module_obj.run_variant(show_display_bool=False, save_results_bool=False, end_date_str=END_DATE_STR)
    finally:
        for m in patch_target_list:
            m.build_index_constituent_matrix = original_dict[id(m)]
    nav_ser = strategy_obj.results["total_value"].astype(float)
    nav_ser.index = pd.to_datetime(nav_ser.index)
    tx_df = strategy_obj.get_transactions()[["bar", "asset", "amount"]].copy()
    tx_df["bar"] = pd.to_datetime(tx_df["bar"])
    return nav_ser, tx_df


def main() -> None:
    out = {}
    for key_str in (sys.argv[1:] or list(MODULE_DICT)):
        exact_nav, exact_tx = _run(key_str, legacy_trim_bool=False)
        legacy_nav, legacy_tx = _run(key_str, legacy_trim_bool=True)
        exact_buys = set(map(tuple, exact_tx[exact_tx["amount"] > 0][["bar", "asset"]].astype(str).to_numpy()))
        legacy_buys = set(map(tuple, legacy_tx[legacy_tx["amount"] > 0][["bar", "asset"]].astype(str).to_numpy()))
        m_exact, m_legacy = _metrics(exact_nav), _metrics(legacy_nav)
        out[key_str] = {
            "exact_new_default": m_exact,
            "legacy_trim": m_legacy,
            "delta_full_cagr_pp": round(m_exact["full"]["cagr_pct"] - m_legacy["full"]["cagr_pct"], 3),
            "delta_last3y_cagr_pp": round(m_exact["last3y"]["cagr_pct"] - m_legacy["last3y"]["cagr_pct"], 3),
            "buys_only_exact": sorted(exact_buys - legacy_buys)[:25],
            "buys_only_legacy": sorted(legacy_buys - exact_buys)[:25],
            "n_buys_only_exact": len(exact_buys - legacy_buys),
            "n_buys_only_legacy": len(legacy_buys - exact_buys),
        }
        print(key_str, json.dumps({k: v for k, v in out[key_str].items() if not k.startswith("buys_only")}), flush=True)
    OUTPUT_PATH.write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
