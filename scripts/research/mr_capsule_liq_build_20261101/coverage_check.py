"""How close real history comes to the live coverage rule of validate_liquidity_sizing_ready: usable Turnover for at least 80% of members at T. On the frozen sign-off inputs, every session from 2004.

Usage: coverage_check.py. Writes results/research/mr_capsule_liq_build_20261101/coverage.json. Reads saved parquet files only.
"""

from __future__ import annotations

import gzip
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

MAIN = Path(r"C:\Users\User\Documents\workspace\alpha_super")
BASE = MAIN / "results" / "review" / "wired_signoff_20261015" / "baselines"
OUT = Path(__file__).resolve().parents[3] / "results" / "research" / "mr_capsule_liq_build_20261101"
SOURCE = {"dv2": "strategy_mr_dv2_vix_gated_bil", "hpi": "strategy_mr_hpi_vote_vix_gated_bil"}


def main() -> None:
    out = {}
    for pod, source in SOURCE.items():
        pricing = pd.read_parquet(BASE / source / "pricing.parquet")
        universe = pickle.load(gzip.open(BASE / source / "auxiliary.pkl.gz"))["universe_df"]
        stocks = sorted({str(s) for s, f in pricing.columns if f == "Turnover" and not str(s).startswith("$") and str(s) not in ("BIL", "SPMO")})
        turnover = pd.DataFrame({s: pd.to_numeric(pricing[(s, "Turnover")], errors="coerce") for s in stocks}, index=pricing.index).astype(float)
        member = universe.sort_index().reindex(columns=stocks).fillna(0).reindex(pricing.index, method="ffill").fillna(0).eq(1)
        positive = turnover.where(np.isfinite(turnover) & (turnover > 0.0))
        usable = (positive.notna().rolling(63, min_periods=1).sum() >= 42) & positive.notna()
        coverage = (usable & member).sum(axis=1) / member.sum(axis=1).replace(0, np.nan)
        window = coverage.loc["2004-01-01":]
        worst = window.nsmallest(5)
        out[pod] = {"member_days_zero_or_inf": int(((~np.isfinite(turnover) | (turnover <= 0.0)) & turnover.notna() & member).loc["2004-01-01":].to_numpy().sum()), "sessions": int(window.notna().sum()), "min": float(window.min()), "p1": float(window.quantile(0.01)), "median": float(window.median()), "sessions_under_80": int((window < 0.8).sum()), "sessions_under_90": int((window < 0.9).sum()), "worst": {str(d.date()): float(v) for d, v in worst.items()}, "since_2020_min": float(window.loc["2020":].min())}
        print(pod, out[pod], flush=True)
    (OUT / "coverage.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
