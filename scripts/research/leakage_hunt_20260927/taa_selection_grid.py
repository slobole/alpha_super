"""TAA leakage hunt - (e) selection/hindsight context: where do the chosen variants sit in the sibling grid?

Run:  uv run python scripts/research/leakage_hunt_20260927/taa_selection_grid.py

Re-runs the 4 bases x 6 fallbacks x {plain, vix_cash} = 48 sibling modules that exist in strategies/taa_df
(the run_taa_df_fallback*_variant_suite.py grids) on the SAME book window (2012-10-02 .. 2026-08-19, default
engine, $100k) and reports each variant's CAGR/Sharpe and the rank of the three chosen modules.  This is
selection context, not leakage: every choice (BTAL, fallback, VIX gate, weights) was made on this history.

Output: results/research/leakage_hunt_20260927/taa/taa_selection_grid.csv
"""

from __future__ import annotations

import importlib
import sys
import time
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd

from taa_common import BOOK_END_STR, BOOK_START_STR, OUT_DIR, install_patches, metrics_from_strategy, patched

BASES = {"taa_df": "strategy_taa_df", "btal": "strategy_taa_df_btal", "btal_1n": "strategy_taa_df_btal_1n",
         "btal_linearity_1n": "strategy_taa_df_btal_linearity_1n"}
FALLBACKS = ("spy", "sso", "upro", "qqq", "qld", "tqqq")
CHOSEN = {"strategy_taa_df_btal_fallback_tqqq_vix_cash", "strategy_taa_df_btal_1n_fallback_tqqq_vix_cash",
          "strategy_taa_df_btal_linearity_1n_fallback_qqq_vix_cash"}


def run_module(mod_name: str) -> dict:
    import strategies.taa_df.strategy_taa_df as base
    import strategies.taa_df.strategy_taa_df_btal_linearity_1n as lin
    import strategies.taa_df.strategy_taa_df_fallback_variant_utils as fu
    import strategies.taa_df.strategy_taa_df_fallback_vix_cash_variant_utils as vix
    mod = importlib.import_module(f"strategies.taa_df.{mod_name}")
    cfg = replace(mod.DEFAULT_CONFIG, end_date_str=BOOK_END_STR)
    is_lin = "linearity" in mod_name
    if mod_name.endswith("_vix_cash"):
        if is_lin:
            epx, _, _, _, _, rb, _ = vix.get_linearity_1n_fallback_vix_cash_data(cfg, lin.get_defense_first_linearity_1n_data)
        else:
            epx, _, _, _, rb, _ = vix.get_standard_fallback_vix_cash_data(cfg, base.get_defense_first_data)
    else:
        if is_lin:
            epx, _, _, _, rb = lin.get_defense_first_linearity_1n_data(cfg)
        else:
            epx, _, _, rb = base.get_defense_first_data(cfg)
    strat = fu._build_defense_first_strategy(strategy_name_str=mod_name, config=cfg, rebalance_weight_df=rb)
    fu._run_strategy_from_weight_df(strategy=strat, execution_price_df=epx, rebalance_weight_df=rb,
                                    backtest_start_date_str=BOOK_START_STR)
    m = metrics_from_strategy(strat)
    m["module"] = mod_name
    return m


def main():
    install_patches()
    rows = []
    for bkey, bname in BASES.items():
        for fb in FALLBACKS:
            for suffix in ("", "_vix_cash"):
                name = f"{bname}_fallback_{fb}{suffix}"
                if not (Path(__file__).resolve().parents[3] / "strategies" / "taa_df" / f"{name}.py").exists():
                    continue
                t = time.time()
                with patched():
                    try:
                        m = run_module(name)
                    except Exception as exc:  # keep going; record
                        m = {"module": name, "error": str(exc)[:200]}
                m.update({"base": bkey, "fallback": fb, "overlay": suffix or "plain", "chosen": name in CHOSEN,
                          "runtime_sec": round(time.time() - t, 1)})
                print({k: m.get(k) for k in ("module", "cagr_pct_calc", "sharpe_calc", "summary_max_dd_pct")}, flush=True)
                rows.append(m)
    df = pd.DataFrame(rows)
    df["sharpe_rank_of_48"] = df["sharpe_calc"].rank(ascending=False, method="min")
    df["cagr_rank_of_48"] = df["cagr_pct_calc"].rank(ascending=False, method="min")
    df.to_csv(OUT_DIR / "taa_selection_grid.csv", index=False)
    print(df.sort_values("sharpe_calc", ascending=False)[["module", "cagr_pct_calc", "sharpe_calc", "summary_max_dd_pct",
                                                           "sharpe_rank_of_48", "cagr_rank_of_48", "chosen"]].to_string(index=False))


if __name__ == "__main__":
    main()
