"""Audit scratch (fund products 2026-10-05, task rerun_e2): the live NDX VXN rule alone, for reference.

Runs strategies.momentum.strategy_mo_atr_normalized_ndx_vxn_scaled.run_variant (alias ndx_vxn in
scripts/research/shelf_rebuild_20260929/run_sleeves.py SLEEVE_DICT) at HEAD, $1M, from 2000-01-01 to the latest
Norgate bar, nothing saved by the engine. Writes the daily path and the fills as CSV. Read-only on the repo.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
OUT = REPO / "results/research/portfolio/fund_products_20261005/audit/rerun_e2"


def main() -> None:
    from strategies.momentum import strategy_mo_atr_normalized_ndx_vxn_scaled as mod

    OUT.mkdir(parents=True, exist_ok=True)
    strategy_obj = mod.run_variant(
        show_display_bool=False,
        save_results_bool=False,
        output_dir_str=str(OUT / "scratch_unused"),
        backtest_start_date_str="2000-01-01",
        capital_base_float=1_000_000.0,
        end_date_str=None,
    )
    res = strategy_obj.results.copy()
    res.index = pd.to_datetime(res.index)
    keep = [c for c in ("total_value", "portfolio_value", "cash", "daily_returns") if c in res.columns]
    res[keep].to_csv(OUT / "ndx_vxn_live_daily.csv", index_label="date")
    strategy_obj._transactions.to_csv(OUT / "ndx_vxn_live_transactions.csv", index=False)
    print("columns:", list(res.columns))
    print("rows:", len(res), res.index[0].date(), res.index[-1].date(), "final", float(res["total_value"].iloc[-1]))
    print(strategy_obj.summary.to_string())


if __name__ == "__main__":
    main()
