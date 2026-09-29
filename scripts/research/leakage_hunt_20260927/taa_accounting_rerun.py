"""TAA leakage hunt - (f) TAA-01 adjusted share units: re-run backtests default engine vs
``historical_share_units_bool=True`` (runtime attribute on the strategy object; engine code untouched).

Run:  uv run python scripts/research/leakage_hunt_20260927/taa_accounting_rerun.py

Window: book window backtest_start 2012-10-02 .. end_date 2026-08-19 for the three BTAL variants (as the book
builder uses them); Inflation Compass on its own default start (2003 warm-up) to 2026-08-19.

Also a P&L-level future-action test: rescale TQQQ (or QQQ) history by 40 / 0.1 and re-run.  With adjusted share
units the modelled commissions and rounding move (non-invariant accounting); with historical share units the run
should be (near) invariant because order size and fees are expressed in nominal shares via Unadjusted Close.

Outputs: results/research/leakage_hunt_20260927/taa/taa01_accounting_rerun.csv (+ .json)
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd

from taa_common import (BOOK_END_STR, BOOK_START_STR, OUT_DIR, compute_decisions, metrics_from_strategy, patched,
                        run_backtest_from_weights)


def run_compass(decisions: dict, hsu: bool):
    import strategies.taa_df.strategy_taa_inflation_compass as compass
    from alpha.engine.backtest import run_daily
    cfg = decisions["config"]
    strat = compass._build_strategy_obj(config_obj=cfg, rebalance_weight_df=decisions["rebalance_weight_df"])
    if hsu:
        strat.historical_share_units_bool = True
    strat.show_taa_weights_report = True
    epx = decisions["execution_price_df"]
    strat.daily_target_weights = decisions["rebalance_weight_df"].reindex(epx.index).ffill().dropna()
    cal = compass._execution_calendar_index(epx, decisions["rebalance_weight_df"], None)
    run_daily(strat, epx, calendar=cal, show_progress=False, show_signal_progress_bool=False, audit_override_bool=None)
    return strat


def one(key: str, hsu: bool, rescale: dict | None = None) -> dict:
    t = time.time()
    with patched(rescale=rescale or {}):
        dec = compute_decisions(key, end_date_str=BOOK_END_STR)
        if key == "compass":
            strat = run_compass(dec, hsu)
        else:
            strat = run_backtest_from_weights(key, dec, start_str=BOOK_START_STR, historical_share_units_bool=hsu)
    m = metrics_from_strategy(strat)
    m.update({"strategy": key, "historical_share_units": hsu, "rescale": json.dumps(rescale or {}),
              "runtime_sec": round(time.time() - t, 1)})
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()}, flush=True)
    return m


def main():
    rows = []
    for key in ("taa3x_rank", "taa3x_1n", "btal_qqq_linearity", "compass"):
        for hsu in (False, True):
            rows.append(one(key, hsu))
    # P&L-level future-action invariance probes
    for key, sym in (("taa3x_rank", "TQQQ"), ("taa3x_1n", "TQQQ"), ("btal_qqq_linearity", "QQQ")):
        for k in (40.0, 0.1):
            for hsu in (False, True):
                rows.append(one(key, hsu, {sym: k}))
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "taa01_accounting_rerun.csv", index=False)
    (OUT_DIR / "taa01_accounting_rerun.json").write_text(df.to_json(orient="records", indent=2), encoding="utf-8")
    print(df[["strategy", "historical_share_units", "rescale", "cagr_pct_calc", "sharpe_calc", "summary_max_dd_pct",
              "total_commissions", "n_transactions", "final_value"]].to_string(index=False))


if __name__ == "__main__":
    main()
