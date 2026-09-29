"""Phase 1 acceptance: replica vs the engine run dv2_check (WIRED DV2, $1M, 2000-01-03 -> 2026-08-19)."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import replica as rp  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
REF = REPO / "results/research/portfolio/growth_shelf_20260924/dv2_sources"
OUT = REPO / "results/research/dv2_deep_20260925"


def feature_parity(p: rp.Panel) -> dict:
    out = {}
    for name, mine in [("dv2", rp._pct_rank(rp.dvk(p, 2), 126)), ("sma_200", pd.DataFrame(p.C).rolling(200).mean().to_numpy()),
                       ("p126d_return", (pd.DataFrame(p.C) / pd.DataFrame(p.C).shift(126) - 1).to_numpy())]:
        eng = p.eng[name]
        both = np.isfinite(mine) & np.isfinite(eng)
        out[name] = {"nan_pattern_mismatch_int": int((np.isfinite(mine) != np.isfinite(eng)).sum()),
                     "max_abs_diff_float": float(np.nanmax(np.abs(mine[both] - eng[both])))}
    return out


def main():
    t = time.time()
    p = rp.Panel("sp500")
    parity = feature_parity(p)
    res = rp.run(p, rp.Rule())
    ref_path = pd.read_csv(REF / "dv2_check__path.csv.gz", index_col="date", parse_dates=True)
    ref_tx = pd.read_csv(REF / "dv2_check__transactions.csv.gz", parse_dates=["date"])
    nav = pd.Series(res.nav, index=res.dates)
    ref_nav = ref_path["total_value_float"].reindex(nav.index)
    ret_diff = (nav.pct_change() - ref_nav.pct_change()).abs()
    mine_tx = res.trades.rename(columns={"asset": "asset_str", "amount": "amount_float"})
    key = lambda df: set(zip(df["date"].dt.strftime("%Y-%m-%d"), df["asset_str"], df["amount_float"].round(6)))
    a, b = key(mine_tx), key(ref_tx)
    report = {"runtime_seconds": round(time.time() - t, 1), "feature_parity": parity,
              "replica_fill_count": len(mine_tx), "engine_fill_count": len(ref_tx),
              "fills_only_in_replica": len(a - b), "fills_only_in_engine": len(b - a),
              "max_abs_daily_return_diff": float(ret_diff.max()), "final_nav_replica": float(nav.iloc[-1]), "final_nav_engine": float(ref_nav.iloc[-1]),
              "replica_stats": rp.summarize(res)}
    first_bad = ret_diff[ret_diff > 1e-8]
    report["first_divergence_date"] = str(first_bad.index[0].date()) if len(first_bad) else None
    if len(a ^ b):
        diff = sorted(list(a - b))[:5], sorted(list(b - a))[:5]
        report["example_diffs"] = diff
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "phase1_replica_validation.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
