"""Baseline production runs, determinism (A10) and capital scaling (A9) for the five Tier B pods.

Usage: uv run python tb_baseline.py [key ...]
Outputs: baseline_<key>.json, equity_<key>.parquet, fills_<key>.parquet
"""

from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd

import tb_common as tc


def run_key(key: str) -> dict:
    t0 = time.time()
    pricing = tc.load_pricing(key)
    dtypes = sorted({str(t) for t in pricing.dtypes})
    cfg = tc.config_for(key)
    a = tc.run(key, pricing)
    b = tc.run(key, pricing)
    ta, tb = a.total_value_series.astype(float), b.total_value_series.astype(float)
    deterministic = bool(ta.equals(tb)) and tc.fills(a).equals(tc.fills(b))
    tv = ta.copy()
    tv.index = pd.to_datetime(tv.index)
    tv.to_frame("total_value").to_parquet(tc.OUT / f"equity_{key}.parquet")
    tx = a.get_transactions().copy()
    tx.to_parquet(tc.OUT / f"fills_{key}.parquet")
    # A9 capital scaling
    scale = {}
    for cap in (30_000.0, 10_000_000.0):
        s = tc.run(key, pricing, cfg=tc.config_for(key, capital_base_float=cap))
        rv = s.total_value_series.astype(float).pct_change().dropna()
        r0 = ta.pct_change().dropna()
        idx = rv.index.intersection(r0.index)
        scale[str(int(cap))] = {**tc.metric_pair(s),
                                "daily_ret_corr_vs_100k": float(np.corrcoef(rv.loc[idx], r0.loc[idx])[0, 1]),
                                "n_fills": int(len(s.get_transactions()))}
    policy = {k: v for k, v in a._accounting_policy_dict.items() if "dividend" in k or "short" in k or "borrow" in k}
    out = {
        "key": key, "dtypes": dtypes, "rows": int(len(pricing)),
        "first_row": str(pricing.index[0].date()), "last_row": str(pricing.index[-1].date()),
        "calendar_start": str(tc.calendar_for(key, pricing, cfg)[0].date()),
        "capital_base": cfg.capital_base_float,
        "costs": {"slippage": a._slippage, "commission_per_share": a._commission_per_share,
                  "commission_minimum": a._commission_minimum},
        "metrics_100k": tc.metric_pair(a), "n_fills": int(len(tx)),
        "commission_total": float(tx["commission"].sum()),
        "dividend_policy": policy,
        "dividend_net_total": float(getattr(a, "dividend_cash_net_total_float", 0.0)),
        "dividend_withholding_total": float(getattr(a, "dividend_withholding_total_float", 0.0)),
        "deterministic_bit_identical": deterministic,
        "capital_scaling": scale,
        "min_cash_frac_nav": float((a.results["cash"].astype(float) / a.results["total_value"].astype(float)).min()),
        "days_negative_cash": int((a.results["cash"].astype(float) < 0).sum()),
        "days": int(len(a.results)),
        "runtime_s": round(time.time() - t0, 1),
    }
    if key == "eom":
        out["borrow_fee_total"] = float(a.borrow_fee_total_float)
        a.month_table_df.to_csv(tc.OUT / "eom_month_table.csv", index=False)
        a.decision_df.to_csv(tc.OUT / "eom_decisions.csv", index=False)
        a.borrow_fee_df.to_csv(tc.OUT / "eom_borrow_ledger.csv", index=False)
    tc.write_json(f"baseline_{key}.json", out)
    print(key, out["metrics_100k"], "det", deterministic, "t", out["runtime_s"], flush=True)
    return out


if __name__ == "__main__":
    for key in (sys.argv[1:] or tc.ALL_KEYS):
        run_key(key)
