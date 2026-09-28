"""Review R3: return impact of whole-share rounding at owner pod sizes (12K / 18K / 30K), since 2016.

Input: the audit's whole_share_since_2016.csv (decision_date, asset, target_weight, nominal_price).
shares = floor(w*C/P); err = w - shares*P/C (uninvested, earns 0 like the engine and like a <USD 100K IBKR account).
Holding return r = AdjClose(next decision)/AdjClose(decision) - 1 (close-to-close proxy for open-to-open).
Lost return per period = sum(err * r). Reported annualised (x periods per year) and as mean cash.
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
END = "2026-09-25"
_C: dict[str, pd.Series] = {}


def close(a):
    if a not in _C:
        _C[a] = load_price_timeseries(a, start_date_str="2015-06-01", end_date_str=END)["Close"].astype(float).dropna()
    return _C[a]


def px_on_or_before(a, d):
    s = close(a).loc[:d]
    return float(s.iloc[-1]) if len(s) else np.nan


out = {}
for key in sys.argv[1:] or ["taa3x", "taa1n", "btal_qqq", "ndx_vxn"]:
    ws = pd.read_csv(TRD / key / "whole_share_since_2016.csv", parse_dates=["decision_date"])
    ws = ws[ws["capital_usd"] == 30000.0][["decision_date", "asset", "target_weight", "nominal_price"]]
    dates = sorted(ws["decision_date"].unique())
    nxt = {d: (dates[i + 1] if i + 1 < len(dates) else pd.Timestamp(END)) for i, d in enumerate(dates)}
    ws["r"] = [px_on_or_before(a, nxt[d]) / px_on_or_before(a, d) - 1.0 for a, d in zip(ws["asset"], ws["decision_date"])]
    years = (pd.Timestamp(END) - dates[0]).days / 365.25
    res = {"decisions": len(dates), "years": round(years, 2)}
    for C in (12_000.0, 18_000.0, 30_000.0):
        sh = np.floor(ws["target_weight"] * C / ws["nominal_price"])
        err = ws["target_weight"] - sh * ws["nominal_price"] / C
        per = (err * ws["r"]).groupby(ws["decision_date"]).sum()
        cash = err.groupby(ws["decision_date"]).sum()
        zero = ws[(sh == 0) & (ws["target_weight"] > 0)]
        res[f"C{int(C)}"] = {
            "mean_rounding_cash_pct_nav": round(100 * float(cash.mean()), 2),
            "p95_rounding_cash_pct_nav": round(100 * float(cash.quantile(0.95)), 2),
            "max_rounding_cash_pct_nav": round(100 * float(cash.max()), 2),
            "max_name_error_pct_nav": round(100 * float(err.max()), 2),
            "name_decisions_err_over_2pct": int((err > 0.02).sum()),
            "name_decisions_err_over_5pct": int((err > 0.05).sum()),
            "zero_share_name_decisions": int(len(zero)),
            "decisions_with_zero_share_name": int(zero["decision_date"].nunique()),
            "zero_share_assets": sorted(zero["asset"].unique().tolist()),
            "lost_return_pp_per_yr": round(100 * float(per.sum()) / years, 3),
        }
    out[key] = res
    print(key, json.dumps(res, indent=1), flush=True)
(OUT / "r3_whole_share_return_impact.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
