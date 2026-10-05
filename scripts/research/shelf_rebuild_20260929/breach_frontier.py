"""Post-hoc, descriptive (labelled in the report): the probability that each family book breaks the owner's hard
drawdown limit on resampled histories (the same stationary bootstrap as SPEC 9: 2,000 paths, mean block 63 sessions,
seed 20260929), next to its LONG CAGR. Nothing here selects a book; it shows the CAGR / breach-risk frontier.

Usage: python breach_frontier.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import lib

OUT = lib.STUDY / "report"


def breach(returns: pd.DataFrame, limits: tuple[float, ...]) -> pd.DataFrame:
    idx = lib.boot_index(len(returns))
    R = returns.to_numpy()
    dd = np.empty((idx.shape[0], R.shape[1]))
    for k in range(idx.shape[0]):
        nav = np.cumprod(1.0 + R[idx[k]], axis=0)
        nav = np.vstack([np.ones((1, R.shape[1])), nav])
        dd[k] = (nav / np.maximum.accumulate(nav, axis=0) - 1.0).min(axis=0)
    out = {f"p_worse_{int(abs(l) * 100)}": (dd < l).mean(axis=0) for l in limits}
    out["boot_dd_p50"] = np.percentile(dd, 50, axis=0)
    return pd.DataFrame(out, index=returns.columns)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    d_r = pd.read_csv(lib.STUDY / "part_d" / "d_long_returns.csv.gz", index_col=0, parse_dates=True)
    g_r = pd.read_csv(lib.STUDY / "part_g" / "g_long_returns.csv.gz", index_col=0, parse_dates=True)
    d_tab = pd.read_csv(lib.STUDY / "part_d" / "d_books.csv", index_col=0)
    g_tab = pd.read_csv(lib.STUDY / "part_g" / "g_books.csv", index_col=0)
    d_b = breach(d_r, (-0.07, -0.10)).join(d_tab[["long_cagr", "long_maxdd", "gates_pass", "pods", "trade_days_per_year"]])
    g_b = breach(g_r, (-0.15, -0.20)).join(g_tab[["long_cagr", "long_maxdd", "long_sharpe", "gates_pass", "pods",
                                                   "any_daily", "wired_share", "mr_option", "third_leg"]])
    d_b.to_csv(OUT / "breach_frontier_d.csv", float_format="%.4f")
    g_b.to_csv(OUT / "breach_frontier_g.csv", float_format="%.4f")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 100)
    ok_g = g_b[g_b["gates_pass"] & (g_b["p_worse_20"] <= 0.15)].sort_values("long_cagr", ascending=False)
    print("Growth books passing the gates with P(DD worse than -20%) <= 15%, by LONG CAGR:")
    print(ok_g.round(3).head(15).to_string())
    ok_d = d_b[d_b["gates_pass"] & (d_b["p_worse_10"] <= 0.15)].sort_values("long_cagr", ascending=False)
    print("Defensive books passing the gates with P(DD worse than -10%) <= 15%, by LONG CAGR:")
    print(ok_d.round(3).head(15).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
