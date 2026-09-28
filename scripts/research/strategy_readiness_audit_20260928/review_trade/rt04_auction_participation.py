"""Review R4: auction-aware participation and the house MOO impact model at owner and fund sizes.

Uses the audit's participation_by_fill.csv (order_frac_of_nav, native-Turnover ADV20 shifted by one session).
(a) House MOO guardrails (alpha/engine/capacity_analysis.py:91-92): soft 0.05%, hard 0.10% of ADV.
(b) Order as a share of the opening auction if the auction is A% of ADV. A = 1.1% (external anchor: NYSE+Nasdaq
    listed average 2011-2017, Greenwich/Tethys; NOT measured here) and a thin-ETF stress A = 0.5% (assumption).
(c) House square-root impact above the modelled 2.5 bp: lambda * sqrt(q/1%) - 2.5, floored at 0, with
    MOO_ETF_PROXY 40/66.4 for ETF pods and MOO_NASDAQ_LARGE 66.4/114 for NDX (capacity_analysis.py:49-68).
    Drag = sum(frac_i * extra_bp_i) / years  [bp of NAV per year].
(d) Opening-print noise per ETF, last 3y: Roll-type estimate sqrt(-cov(gap_t, intraday_t)) where
    gap = ln(O_t/C_{t-1}), intraday = ln(C_t/O_t); positive cov -> reported as 0. Plus the one-tick size in bp.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
from data.norgate_loader import load_price_timeseries  # noqa: E402
TRD = REPO / "results/research/strategy_readiness_audit_20260928/tradability"
OUT = REPO / "results/research/strategy_readiness_audit_20260928/review_trade"
END = pd.Timestamp("2026-09-25"); L3Y = pd.Timestamp("2023-09-25")
LAM = {"taa3x": (40.0, 66.4), "taa1n": (40.0, 66.4), "btal_qqq": (40.0, 66.4), "ndx_vxn": (66.4, 114.0)}
out = {}
for key, (lc, ls) in LAM.items():
    df = pd.read_csv(TRD / key / "participation_by_fill.csv", parse_dates=["bar"])
    res = {}
    for wname, w in (("full", df), ("last3y", df[df["bar"] >= L3Y])):
        years = (END - w["bar"].min()).days / 365.25
        r = {}
        for C in (12_000, 18_000, 30_000, 1_000_000, 10_000_000):
            q = w["order_frac_of_nav"] * C / w["adv20_usd"]
            extra_c = np.maximum(0.0, lc * np.sqrt(q / 0.01) - 2.5)
            extra_s = np.maximum(0.0, ls * np.sqrt(q / 0.01) - 2.5)
            worst = w.loc[q.idxmax()]
            r[f"C{C}"] = {
                "share_over_moo_soft_0.05pct": round(float((q > 0.0005).mean()), 4),
                "share_over_moo_hard_0.10pct": round(float((q > 0.0010).mean()), 4),
                "p99_order_pct_of_open_auction_A1.1": round(float(q.quantile(0.99) / 0.011 * 100), 2),
                "max_order_pct_of_open_auction_A1.1": round(float(q.max() / 0.011 * 100), 1),
                "p99_order_pct_of_open_auction_A0.5": round(float(q.quantile(0.99) / 0.005 * 100), 2),
                "worst_order": f"{worst['asset']} {worst['bar'].date()} ADV20 ${worst['adv20_usd']:,.0f}",
                "extra_impact_central_bp_per_yr": round(float((w["order_frac_of_nav"] * extra_c).sum() / years), 2),
                "extra_impact_stress_bp_per_yr": round(float((w["order_frac_of_nav"] * extra_s).sum() / years), 2),
            }
        res[wname] = r
    out[key] = res
# BTAL liquidity history
bt = load_price_timeseries("BTAL", start_date_str="2011-09-01", end_date_str=str(END.date()))
yr = bt["Turnover"].groupby(bt.index.year).median()
out["btal_median_turnover_by_year_usd"] = {int(k): round(float(v), 0) for k, v in yr.items()}
out["btal_zero_volume_days_by_year"] = {int(k): int(v) for k, v in (bt["Volume"] == 0).groupby(bt.index.year).sum().items()}
# opening print noise
noise = {}
for s in ["TQQQ", "QQQ", "BTAL", "DBC", "UUP", "GLD", "TLT", "SPY"]:
    p = load_price_timeseries(s, start_date_str="2023-06-01", end_date_str=str(END.date()))
    p = p[p.index >= L3Y]
    gap = np.log(p["Open"] / p["Close"].shift(1)); intr = np.log(p["Close"] / p["Open"])
    cov = float(pd.concat([gap, intr], axis=1).dropna().cov().iloc[0, 1])
    noise[s] = {"open_noise_bp": round(1e4 * np.sqrt(max(0.0, -cov)), 1), "cov_sign": "neg" if cov < 0 else "pos",
                "tick_bp_median": round(float((0.01 / p["Unadjusted Close"]).median() * 1e4), 1),
                "median_adv_usd": round(float(p["Turnover"].median()), 0)}
out["opening_print_noise_last3y"] = noise
(OUT / "r4_auction_participation.json").write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
print(json.dumps(out, indent=1, default=str))
