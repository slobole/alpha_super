"""Inflation Compass - causal T5YIE re-run (follow-up to taa_macro_vintage.py).

Run:  uv run python scripts/research/leakage_hunt_20260927/taa_compass_causal.py

The backtest aligns T5YIE by observation date with allow_exact_matches=True (strategy_taa_inflation_compass.py:185)
so 282 of 283 decisions use T5YIE_T at Close_T.  T5YIE_T is first published on T+1 (H.15 / FRED; ALFRED vintage
as of T ends at T-1; ALFRED shows 0 revisions in 660,697 overlapping observations since the 2014-01 first vintage).
The causal equivalent of "the vintage available at T" is therefore the current series with observations dated < T.

Variants (all on a COMMON execution window starting 2003-05-01 so CAGR/Sharpe are comparable):
  base    : code as-is (same-date T5YIE)
  causal  : for each decision T, features recomputed with T5YIE observations dated < T only (T-60 value unchanged)
  lag1_all: every observation made available one XNYS session after its date (also lags the T-60 value)
Also cross-checks causal vs ALFRED for the 2014-01..2026-08 decisions.
Output: results/research/leakage_hunt_20260927/taa/taa_compass_causal.json / .csv
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from taa_accounting_rerun import run_compass
from taa_common import BOOK_END_STR, OUT_DIR, compute_decisions, load_norgate_full, metrics_from_strategy, patched, read_frozen_series

COMMON_START = "2003-05-01"


def run_from_weights(month_end_weight_df, dec):
    import strategies.taa_df.strategy_taa_df as base
    from alpha.engine.backtest import run_daily
    import strategies.taa_df.strategy_taa_inflation_compass as compass
    rb = base.map_month_end_weights_to_rebalance_open_df(month_end_weight_df, dec["execution_price_df"].index)
    strat = compass._build_strategy_obj(config_obj=dec["config"], rebalance_weight_df=rb)
    epx = dec["execution_price_df"]
    strat.daily_target_weights = rb.reindex(epx.index).ffill().dropna()
    cal = compass._execution_calendar_index(epx, rb, COMMON_START)
    run_daily(strat, epx, calendar=cal, show_progress=False, show_signal_progress_bool=False, audit_override_bool=None)
    return strat


def main():
    import strategies.taa_df.strategy_taa_inflation_compass as compass
    spy = load_norgate_full("SPY", "CAPITALSPECIAL")
    sessions = pd.DatetimeIndex(spy.index[spy["Volume"].fillna(0) > 0])
    t5 = read_frozen_series("T5YIE")
    with patched():
        dec = compute_decisions("compass", end_date_str=BOOK_END_STR)
        sig = compass.load_signal_close_df(symbol_list=compass.DEFAULT_CONFIG.signal_asset_tuple,
                                           start_date_str=compass.DEFAULT_CONFIG.start_date_str,
                                           end_date_str=BOOK_END_STR)
    ref = dec["month_end_weight_df"]
    causal = ref.copy()
    flips = []
    for T in ref.index:
        T = pd.Timestamp(T)
        s = t5[t5.index < T]
        try:
            _, w = compass.compute_month_end_signal_and_weight_df(sig.loc[:T], s, compass.DEFAULT_CONFIG)
        except RuntimeError as exc:
            flips.append({"label": str(T.date()), "error": str(exc)[:160]})
            continue
        if T in w.index:
            causal.loc[T] = w.loc[T].reindex(causal.columns).values
            if not np.allclose(w.loc[T].reindex(ref.columns).values, ref.loc[T].values):
                flips.append({"label": str(T.date()), "base": {c: v for c, v in ref.loc[T].items() if v},
                              "causal": {c: v for c, v in w.loc[T].items() if v}})
        else:
            flips.append({"label": str(T.date()), "error": "no row at T (warm-up)"})
    # lag1_all variant
    pos = sessions.searchsorted(t5.index, side="right")
    ok = pos < len(sessions)
    lag = pd.Series(t5.values[ok], index=sessions[pos[ok]], name="T5YIE")
    lag = lag[~lag.index.duplicated(keep="last")]
    with patched(macro_override={"T5YIE": lag}):
        dec_lag = compute_decisions("compass", end_date_str=BOOK_END_STR)

    rows = []
    with patched():
        for name, w in (("base", ref), ("causal_obs_before_T", causal), ("lag1_all_obs", dec_lag["month_end_weight_df"])):
            strat = run_from_weights(w, dec)
            m = metrics_from_strategy(strat)
            m["variant"] = name
            rows.append(m)
            print(m, flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "taa_compass_causal.csv", index=False)
    alf = pd.read_csv(OUT_DIR / "taa_t5yie_alfred_table.csv", parse_dates=["T"]) if (OUT_DIR / "taa_t5yie_alfred_table.csv").exists() else None
    xcheck = None
    if alf is not None and "same_weights" in alf:
        a = alf.dropna(subset=["same_weights"])
        alf_flip = set(a.loc[~a["same_weights"].astype(bool), "T"].dt.date.astype(str))
        causal_flip_2014 = {f["label"] for f in flips if "causal" in f and f["label"] >= "2014-01-31"}
        xcheck = {"alfred_flips": sorted(alf_flip), "causal_flips_since_2014": sorted(causal_flip_2014),
                  "identical": alf_flip == causal_flip_2014}
    out = {"n_decisions": int(len(ref)), "n_flips_causal": len([f for f in flips if "causal" in f]),
           "flips": flips, "alfred_crosscheck": xcheck, "metrics": rows}
    (OUT_DIR / "taa_compass_causal.json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "metrics"}, indent=1, default=str))
    print(df[["variant", "start", "end", "cagr_pct_calc", "sharpe_calc", "summary_max_dd_pct", "n_transactions"]].to_string(index=False))


if __name__ == "__main__":
    main()
