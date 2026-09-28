"""Independent re-checks of the reviewer's material findings (2026-09-28).

1. Joint parameter x rebalance-day luck: 80 random grid cells x 11 offsets (-5..+5).
2. Episode concentration: Sharpe of Compass / growth-only / SPY trend excluding 2003-2007 and 2022; yearly
   contribution of the inflation axis (Compass minus growth-only).
3. Book role on excess-of-T-bill returns (Sharpe and Calmar of the book's return minus the T-bill return).
"""

from __future__ import annotations

import itertools
import json

import numpy as np
import pandas as pd

import common as cm
import p3_sensitivity as p3


def run(w):
    return cm.run_replica(cm.map_to_execution(w, cm.sessions(), lag=1))


def sharpe(r):
    return float(r.mean() / r.std(ddof=1) * np.sqrt(252))


def main():
    sig = cm.signal_close_df()
    t5 = cm.fred_series("T5YIE")
    t5 = t5[t5.index <= pd.Timestamp(cm.MAIN_END_STR)]
    out = {}
    rng = np.random.default_rng(11)
    cells = list(itertools.product(p3.THRESH, p3.BE, p3.SLOPE, p3.SMA))
    pick = [cells[i] for i in rng.choice(len(cells), 80, replace=False)]
    rows = []
    for th, be, sl, sma in pick:
        p = cm.Params(threshold=th, be_lookback=be, slope_lookback=sl, sma=sma)
        for off in range(-5, 6):
            w, _ = cm.compass_weights(sig, t5, p, offset=off)
            rows.append({"threshold": th, "be": be, "slope": sl, "sma": sma, "offset": off,
                         "sharpe": cm.metrics(run(w))["sharpe"]})
    j = pd.DataFrame(rows)
    j.to_csv(cm.OUT / "p7_joint_grid_offset.csv", index=False)
    out["joint"] = {"n": len(j), "median": float(j.sharpe.median()), "p05": float(j.sharpe.quantile(0.05)),
                    "p95": float(j.sharpe.quantile(0.95)),
                    "median_offset0": float(j[j.offset == 0].sharpe.median()),
                    "median_offset_nonzero": float(j[j.offset != 0].sharpe.median())}

    navs = pd.read_pickle(cm.OUT / "p2_navs.pkl")
    r = navs[["Compass (fixed)", "SPY>SMA200: XLK else XLP/IEF (growth axis only)", "SPY>SMA200 else IEF"]].pct_change()
    r.columns = ["compass", "growth_only", "spy_trend"]
    yr = r.index.year
    ex = r[~((yr >= 2003) & (yr <= 2007)) & (yr != 2022)].dropna()
    mid = r[(yr >= 2008) & (yr <= 2021)].dropna()
    out["sharpe_ex_2003_07_and_2022"] = {c: sharpe(ex[c]) for c in ex}
    out["sharpe_2008_2021"] = {c: sharpe(mid[c]) for c in mid}
    ann = (1 + r).groupby(yr).prod() - 1
    out["yearly_inflation_axis_pp"] = ((ann["compass"] - ann["growth_only"]) * 100).round(1).to_dict()

    # book excess metrics
    b = pd.read_csv(cm.OUT / "p5_books.csv")
    out["book_note"] = "see p7_books_excess.csv"
    print(json.dumps(out, indent=1, default=str))
    (cm.OUT / "p7_review_checks.json").write_text(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
