"""NDX ATR-VXN trim follow-up: WHERE does the last-3y CAGR gap between the trimmed (production) and untrimmed
(live-semantics) universes come from? Saves month-end target sets of both arms and daily NAV, then splits the gap into
months with a different target set vs months with the same set (rounding / NAV-path residue)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rq_ndx_untrimmed as base  # noqa: E402

atr_module, vxn_module = base.atr_module, base.vxn_module


def run_arm(trim_bool: bool):
    atr_module.build_index_constituent_matrix = lambda indexname="S&P 500", _t=trim_bool: base.build_universe(indexname, _t)
    strategy_obj = vxn_module.run_variant(show_display_bool=False, save_results_bool=False, end_date_str=base.END_DATE_STR)
    tx = strategy_obj.get_transactions().copy()
    tx["bar"] = pd.to_datetime(tx["bar"])
    pos = tx.pivot_table(index="bar", columns="asset", values="amount", aggfunc="sum").fillna(0.0).cumsum()
    held = {d.date().isoformat(): sorted(pos.columns[pos.loc[d].abs() > 1e-9].astype(str)) for d in pos.index}
    return strategy_obj.results["total_value"].astype(float), held


def main() -> None:
    tv_t, held_t = run_arm(True)
    tv_u, held_u = run_arm(False)
    start = tv_t.index[-1] - pd.DateOffset(years=3)
    r_t = tv_t.pct_change().loc[start:].dropna()
    r_u = tv_u.pct_change().loc[start:].dropna()
    common_dates = sorted(set(held_t) & set(held_u))
    diff_dates = [d for d in common_dates if held_t[d] != held_u[d]]
    diff_recent = [d for d in diff_dates if pd.Timestamp(d) >= start]
    monthly_t = (1 + r_t).resample("ME").prod() - 1
    monthly_u = (1 + r_u).resample("ME").prod() - 1
    gap = (monthly_u - monthly_t)
    out = {
        "last3y_daily_return_corr": float(r_t.corr(r_u)),
        "last3y_cum_return_trimmed": float((1 + r_t).prod() - 1),
        "last3y_cum_return_untrimmed": float((1 + r_u).prod() - 1),
        "rebalance_bars_with_different_holdings_total": len(diff_dates),
        "rebalance_bars_with_different_holdings_last3y": diff_recent,
        "holdings_detail_last3y": {d: {"trimmed_only": sorted(set(held_t[d]) - set(held_u[d])),
                                       "untrimmed_only": sorted(set(held_u[d]) - set(held_t[d]))} for d in diff_recent},
        "largest_monthly_gaps_last3y": {str(k.date()): float(v) for k, v in gap.abs().nlargest(6).items()},
    }
    print(json.dumps(out, indent=1))
    (base.OUT_DIR_PATH / "rq_ndx_untrimmed_detail.json").write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
