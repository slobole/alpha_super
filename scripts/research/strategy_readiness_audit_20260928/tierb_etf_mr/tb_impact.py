"""House square-root impact drag (alpha/engine/capacity_analysis.py) on the audited fills, and C2 whole-share error.

  impact_bps_i(C) = lambda * sqrt(part_i(C) / 1%)     part = order notional at pod size C / ADV20 (native Turnover)
  house cost_i = max(2.5 bp, impact_bps_i)   (capacity_implicit_cost_bps_float); the backtest already pays 2.5 bp, so
  incremental drag(C) [pp/yr] = sum_i order_frac_i * (cost_i - 2.5) / 1e4 / years * 100
lambda: MOO_ETF_PROXY central 40 bp (stress 66.4); MOC central 8.2 bp (stress 17.8).  Pre-TCA house model, not TCA.

C2: for each decision close P (nominal) in the last 3 years, whole-share error at 30K = (w * 30K mod P) / 30K.

Usage: uv run python tb_impact.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import tb_common as tc


def main() -> None:
    out = {}
    for key in tc.ALL_KEYS:
        path = tc.OUT / f"{key}_participation.csv"
        if not path.exists():
            continue
        part = pd.read_csv(path, parse_dates=["bar"])
        lam_c, lam_s = (8.2, 17.8) if key == "eom" else (40.0, 66.4)
        eq = pd.read_parquet(tc.OUT / f"equity_{key}.parquet")["total_value"]
        rec = {}
        for window, frame, start in (("full", part, eq.index[0]),
                                     ("last3y", part[part["bar"] >= pd.Timestamp(tc.LAST3Y_START_STR)],
                                      pd.Timestamp(tc.LAST3Y_START_STR))):
            years = (eq.index[-1] - start).days / 365.25
            for cap in (30_000, 100_000, 300_000, 1_000_000, 10_000_000):
                # participation is linear in pod size: part(C) = part(30K) * C / 30K
                p = frame["part_30000"].replace([np.inf], np.nan) * cap / 30_000.0
                ok = p.notna()
                bps_c = np.maximum(2.5, lam_c * np.sqrt(p[ok] / 0.01)) - 2.5
                bps_s = np.maximum(2.5, lam_s * np.sqrt(p[ok] / 0.01)) - 2.5
                rec[f"{window}_{cap}"] = {
                    "central_pp_per_yr": float((frame.loc[ok, "order_frac_of_nav"] * bps_c).sum() / 1e4 / years * 100),
                    "stress_pp_per_yr": float((frame.loc[ok, "order_frac_of_nav"] * bps_s).sum() / 1e4 / years * 100),
                    "median_incremental_order_bps_central": float(bps_c.median()) if ok.any() else None}
        out[key] = {"impact": rec}
        # C2 whole-share error at 30K over the last 3 years of decision closes
        pricing = tc.load_pricing(key)
        if key == "eom":
            weights = {"SPY": 1.0, "TLT": 1.0}
        else:
            strat = tc.make_strategy(key)
            weights = {s: strat.target_weight_float for s in tc.symbols_for(key)}
        c2 = {}
        for s, w in weights.items():
            px = pricing[(s, "Unadjusted Close")].astype(float).loc[tc.LAST3Y_START_STR:].dropna()
            err = ((w * 30_000.0) % px) / 30_000.0 * 100
            c2[s] = {"max_err_pct_nav": float(err.max()), "p95_err_pct_nav": float(err.quantile(0.95)),
                     "median_err_pct_nav": float(err.median()), "zero_share_days": int((px > w * 30_000.0).sum())}
        out[key]["whole_share_30k_last3y"] = c2
        print(key, out[key], flush=True)
    tc.write_json("impact_and_whole_share.json", out)


if __name__ == "__main__":
    main()
