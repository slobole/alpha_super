"""Fast (numba) vs reference (pandas) indicator equivalence on real Norgate data (research-only).

QPI: alpha.indicators.qp_indicator (-> alpha/engine/qp_indicator_fast.py) vs alpha.indicators.qp_indicator_reference
     (-> alpha/engine/indicators.py:3-24).
DV2: alpha.indicators.dv2_indicator (-> alpha/engine/dv2_indicator_fast.py) vs dv2_indicator_reference
     (-> alpha/engine/indicators.py:27-37).
Symbols: every symbol traded by the production QPI/DV2 backtests (if on disk) plus a seeded random sample, over the
full 1998-2026 history exactly as the strategies load it (CAPITALSPECIAL, ALLMARKETDAYS).
Norgate returns float32 prices.  The fast kernels cast to float64 before computing returns; the pandas reference
computes pct_change / rolling mean in the input dtype (float32), so near-ties can resolve differently.  The script
counts value differences AND threshold flips (QPI < 30, DV2 < 10) that could change a decision.
Also records, per QPI value, whether the 1,260-return window includes the current bar (it does by construction:
window = returns [t-1259, t]) and the share of zero 3-day returns created by padded bars.

Usage: uv run python aud_indicator_equiv.py [n_random]
"""

from __future__ import annotations

import json
import sys
import time

import numpy as np
import pandas as pd

import aud_common as ac
import aud_data
from alpha.indicators import dv2_indicator, dv2_indicator_reference, qp_indicator, qp_indicator_reference


def _cmp(a: pd.Series, b: pd.Series, threshold: float) -> dict:
    a, b = a.astype(float).to_numpy(), b.astype(float).to_numpy()
    flips = int(((a < threshold) != (b < threshold))[~np.isnan(a) & ~np.isnan(b)].sum())
    nan_mismatch = int((np.isnan(a) != np.isnan(b)).sum())
    both = ~np.isnan(a) & ~np.isnan(b)
    diff = np.abs(a[both] - b[both])
    return {"n_valid": int(both.sum()), "nan_mismatch": nan_mismatch,
            "max_abs_diff": float(diff.max()) if diff.size else 0.0, "n_diff_gt_1e9": int((diff > 1e-9).sum()),
            "threshold_flips": flips}


def main(n_random: int = 150) -> None:
    data = aud_data.load()
    pricing = data["pricing_df"]
    symbols = sorted({str(s) for s in pricing.columns.get_level_values(0) if not str(s).startswith("$")})
    traded = set()
    for arm in ("qpi_base", "dv2_base"):
        path = ac.OUT / "full_runs" / arm / "transactions.csv.gz"
        if path.exists():
            traded |= set(pd.read_csv(path, usecols=["asset"])["asset"].astype(str))
    rng = np.random.default_rng(20260928)
    sample = sorted(set(rng.choice(symbols, size=min(n_random, len(symbols)), replace=False)) | (traded & set(symbols)))
    rows = []
    t0 = time.time()
    for i, s in enumerate(sample):
        c, h, l = (pricing[(s, f)] for f in ("Close", "High", "Low"))
        if c.notna().sum() < 300:
            continue
        q_fast, q_ref = qp_indicator(c), qp_indicator_reference(c)
        d_fast, d_ref = dv2_indicator(c, h, l), dv2_indicator_reference(c, h, l)
        vol = pricing[(s, "Volume")]
        r3 = c.pct_change(3, fill_method=None)
        rows.append({"symbol": s, "traded": s in traded, **{f"qpi_{k}": v for k, v in _cmp(q_fast, q_ref, 30.0).items()},
                     **{f"dv2_{k}": v for k, v in _cmp(d_fast, d_ref, 10.0).items()},
                     "close_dtype": str(c.dtype),
                     "n_padded_bars": int(((vol == 0) & c.notna()).sum()),
                     "share_zero_3d_returns": float((r3 == 0).sum() / max(1, r3.notna().sum()))})
        if i % 25 == 0:
            print(i, len(sample), s, rows[-1]["qpi_max_abs_diff"], rows[-1]["dv2_max_abs_diff"], round(time.time() - t0), flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(ac.OUT / "indicator_equivalence.csv", index=False)
    summary = {
        "n_symbols": int(len(df)), "n_traded_symbols": int(df["traded"].sum()),
        "qpi_values_compared": int(df["qpi_n_valid"].sum()), "qpi_nan_mismatch": int(df["qpi_nan_mismatch"].sum()),
        "qpi_max_abs_diff": float(df["qpi_max_abs_diff"].max()), "qpi_n_diff_gt_1e9": int(df["qpi_n_diff_gt_1e9"].sum()),
        "qpi_threshold30_flips": int(df["qpi_threshold_flips"].sum()),
        "dv2_threshold10_flips": int(df["dv2_threshold_flips"].sum()),
        "close_dtypes": sorted(df["close_dtype"].unique().tolist()),
        "dv2_values_compared": int(df["dv2_n_valid"].sum()), "dv2_nan_mismatch": int(df["dv2_nan_mismatch"].sum()),
        "dv2_max_abs_diff": float(df["dv2_max_abs_diff"].max()), "dv2_n_diff_gt_1e9": int(df["dv2_n_diff_gt_1e9"].sum()),
        "median_share_zero_3d_returns": float(df["share_zero_3d_returns"].median()),
        "symbols_with_padded_bars": int((df["n_padded_bars"] > 0).sum()),
        "runtime_s": round(time.time() - t0, 1),
    }
    (ac.OUT / "indicator_equivalence_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 150)
