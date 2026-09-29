"""Tactical FI: bound the whole-share-in-adjusted-units effect (E-02 analogue; commission is zero here).

The invariance runs showed identical month-end weight tables but different small top-up trades when an ETF's price
scale changes (int(target/Close_adj) == current shares decides whether a drift top-up trade happens).  Measure the
NAV effect: rescaled runs and historical_share_units_bool=True vs baseline; report the real-data k range.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from def_common import BOOK_END, BOOK_START, OUT, dump_json, harness, metric_rows
import def_tfi as t

import pandas as pd

from strategies.taa_beyond_6040 import strategy_taa_tactical_fixed_income_ief_lqd as m


def main():
    px, yield_df, snaps = t.load_inputs()
    sessions = pd.DatetimeIndex(px.index)
    sig, w = t.weights_from(yield_df, sessions, m.DEFAULT_CONFIG.last_complete_signal_month_str)
    cash = m.build_causal_cash_return_ser(sessions, snaps[t.SERIES.index("DGS3MO")].value_ser)
    windows = {"full": (None, None), "book": (BOOK_START, BOOK_END)}
    base = t.run_engine(px, sig, w, cash, snaps)
    rows = metric_rows("baseline", base.results["daily_returns"], windows)
    for a, k in (("IEF", 40.0), ("IEF", 0.1), ("LQD", 40.0), ("LQD", 0.1)):
        s = t.run_engine(harness.rescale_symbol_history(px, a, k), sig, w, cash, snaps)
        rows += metric_rows(f"{a}_rescaled_k{k}", s.results["daily_returns"], windows)
    # historical units: need an un-run strategy object; replicate _run_strategy with the flag set
    orig = m._build_strategy_obj

    def flagged(*args, **kw):
        obj = orig(*args, **kw)
        obj.historical_share_units_bool = True
        return obj
    m._build_strategy_obj = flagged
    try:
        h = t.run_engine(px, sig, w, cash, snaps)
    finally:
        m._build_strategy_obj = orig
    rows += metric_rows("historical_share_units", h.results["daily_returns"], windows)
    kr = {a: {"k_min": float((px[(a, "Unadjusted Close")] / px[(a, "Close")]).min()),
              "k_max": float((px[(a, "Unadjusted Close")] / px[(a, "Close")]).max())} for a in ("IEF", "LQD")}
    tx = base.get_transactions()
    nav = base.results["total_value"].astype(float)
    tx_small = tx.assign(notional=(tx["amount"].astype(float) * tx["price"].astype(float)).abs(),
                         nav=pd.to_datetime(tx["bar"]).map(nav).astype(float))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "tfi_units_metrics.csv", index=False)
    dump_json({"k_range": kr, "baseline_n_fills": int(len(tx)),
               "baseline_fills_below_1pct_nav": int((tx_small["notional"] < 0.01 * tx_small["nav"]).sum())},
              "tfi_units_summary.json")
    print(df.to_string())
    print(kr)


if __name__ == "__main__":
    main()
