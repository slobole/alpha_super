"""Descriptive extras for the portfolio refresh report (reads refresh_books.py outputs; no new selection).

- calendar-year returns of the NDX variants, TAA, the main books and QQQ / SPX total return;
- a TAA-share curve for the TAA 3x rank + NDX pair (descriptive sensitivity, not a pick);
- sleeve correlation matrix (monthly, exact window);
- rolling 3-year Sharpe of G3 variants;
- NDX variants regressed on QQQ total return (monthly alpha, beta).

Usage: python extras.py
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
for path in (REPO, REPO / "scripts" / "research" / "fund_menu_20260923"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import common  # noqa: E402

OUT = REPO / "results" / "research" / "portfolio" / "portfolio_refresh_20260927"
CUT, END = pd.Timestamp("2012-10-02"), pd.Timestamp("2026-08-19")


def maxdd(r: pd.Series) -> float:
    v = (1 + r).cumprod()
    return float((v / v.cummax() - 1).min())


def sharpe(r: pd.Series) -> float:
    return float(r.mean() / r.std() * np.sqrt(252))


def main() -> int:
    sleeves = pd.read_csv(OUT / "sleeve_series_incl_2008.csv.gz", index_col=0, parse_dates=True)
    books = pd.read_csv(OUT / "growth_series_incl_2008.csv.gz", index_col=0, parse_dates=True)
    bench = pd.read_csv(OUT / "bench_incl_2008.csv.gz", index_col=0, parse_dates=True)
    qqq_close = common.load_total_return_close_ser("QQQ", "2007-12-01", END.strftime("%Y-%m-%d"))
    qqq = qqq_close.pct_change().reindex(sleeves.index)
    out = {}

    # Calendar years.
    cols = {"TAA 3x rank": sleeves["taa_btal_tqqq"], "NDX legacy (leaky)": sleeves["ndx_vxn"],
            "NDX NATR20": sleeves["ndx_natr20"], "NDX ATR$ fixed": sleeves["ndx_atrfix"],
            "G3 legacy": books["G3 | legacy"], "G3 NATR20": books["G3 | NATR20"], "G3 ATR$ fixed": books["G3 | ATR$ fixed"],
            "G3+capsule NATR20": books["G3 + capsule | NATR20"], "G3+capsule ATR$ fixed": books["G3 + capsule | ATR$ fixed"],
            "QQQ TR": qqq, "S&P 500 TR": bench["SPXTR"].reindex(sleeves.index)}
    years = pd.DataFrame({k: (1 + v.fillna(0.0)).groupby(v.index.year).prod() - 1 for k, v in cols.items()})
    years.to_csv(OUT / "calendar_years.csv", float_format="%.5g")

    # TAA-share curve (annual reset), incl. 2008 and exact window.
    curve = []
    for ndx in ("ndx_natr20", "ndx_atrfix", "ndx_vxn"):
        for w in np.round(np.arange(0.0, 1.0001, 0.1), 2):
            weights = {"taa_btal_tqqq": w, ndx: 1 - w} if 0 < w < 1 else ({"taa_btal_tqqq": 1.0} if w == 1 else {ndx: 1.0})
            r = common.book_return_ser(sleeves[list(weights)], weights, "annual")[0]
            ex = r.loc[CUT:]
            curve.append({"ndx": ndx, "taa_share": w, "sharpe_exact": sharpe(ex), "cagr_exact": float((1 + ex).prod() ** (252 / len(ex)) - 1),
                          "maxdd_incl_2008": maxdd(r), "maxdd_exact": maxdd(ex), "sharpe_incl_2008": sharpe(r),
                          "sharpe_2022_26": sharpe(ex.loc["2022":])})
    pd.DataFrame(curve).to_csv(OUT / "taa_share_curve.csv", index=False, float_format="%.5g")

    # Monthly correlation, exact window.
    corr_cols = {"TAA 3x rank": "taa_btal_tqqq", "NDX NATR20": "ndx_natr20", "NDX ATR$ fixed": "ndx_atrfix",
                 "MOSAIC fixed": "mosaic_fix", "DV2": "dv2", "HPI vote": "hpi_vote", "ETF DV2": "etf_ind_fix",
                 "CORE5": "core5", "BTAL_QQQ": "taa_btal_lin_qqq"}
    monthly = (1 + sleeves.loc[CUT:, list(corr_cols.values())]).resample("ME").prod() - 1
    monthly.columns = list(corr_cols)
    monthly["QQQ"] = ((1 + qqq.loc[CUT:]).resample("ME").prod() - 1)
    monthly.corr().to_csv(OUT / "corr_monthly.csv", float_format="%.3f")

    # Rolling 3y Sharpe.
    roll = pd.DataFrame({k: books[k].rolling(756).apply(lambda x: x.mean() / x.std() * np.sqrt(252), raw=True)
                         for k in ("G3 | legacy", "G3 | NATR20", "G3 | ATR$ fixed", "G3 + capsule | NATR20", "G3 + capsule | ATR$ fixed")})
    roll["TAA 3x rank alone"] = sleeves["taa_btal_tqqq"].rolling(756).apply(lambda x: x.mean() / x.std() * np.sqrt(252), raw=True)
    roll.loc[CUT:].iloc[::5].to_csv(OUT / "rolling_3y_sharpe_weekly.csv", float_format="%.4f")

    # NDX variants vs QQQ (monthly OLS, exact window and 2022-26).
    reg = {}
    for name in ("NDX legacy (leaky)", "NDX NATR20", "NDX ATR$ fixed", "TAA 3x rank"):
        y = (1 + cols[name].loc[CUT:]).resample("ME").prod() - 1
        x = monthly["QQQ"]
        for label, sl in (("2012_26", slice(None)), ("2022_26", slice("2022", None))):
            yy, xx = y.loc[sl], x.loc[sl]
            beta = float(np.cov(yy, xx)[0, 1] / xx.var())
            alpha = float((yy.mean() - beta * xx.mean()) * 12)
            resid = yy - beta * xx
            t = float(resid.mean() / (resid.std() / np.sqrt(len(resid))))
            reg[f"{name} | {label}"] = {"alpha_ann": alpha, "beta": beta, "t_alpha": t, "corr": float(yy.corr(xx))}
    out["vs_qqq"] = reg
    q = qqq.loc[CUT:].dropna()
    out["qqq_exact"] = {"cagr": float((1 + q).prod() ** (252 / len(q)) - 1), "sharpe": sharpe(q), "maxdd": maxdd(q)}
    (OUT / "extras.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    pd.set_option("display.width", 220)
    print(years.round(3).to_string())
    print(pd.DataFrame(curve).round(3).to_string())
    print(monthly.corr().round(2).to_string())
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
