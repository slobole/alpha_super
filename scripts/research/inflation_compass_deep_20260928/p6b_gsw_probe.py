"""Amendment A3 (post-hoc, EXPLORATORY): 1999-2002 with real sector SPDRs and the Fed's GSW 5-year breakeven.

BKEVEN05 = zero-coupon 5y inflation compensation from Gurkaynak-Sack-Wright (feds200805), a later fitted
reconstruction. Value dated T is treated as usable from T+1 (like T5YIE). Execution and costs as the holdout
(next close, 5 bps per side, total-return levels, synthetic bond before IEF exists).
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

import common as cm
import p6_holdout as ph

RAW = cm.OUT / "_cache" / "holdout" / "feds200805.csv"


def gsw_bkeven05() -> pd.Series:
    lines = RAW.read_text(encoding="utf-8").splitlines()
    head = next(i for i, l in enumerate(lines) if l.startswith("Date,"))
    df = pd.read_csv(RAW, skiprows=head)
    ser = pd.to_numeric(df["BKEVEN05"], errors="coerce")
    ser.index = pd.to_datetime(df["Date"])
    return ser.dropna().sort_index().rename("BKEVEN05")


def main():
    g = gsw_bkeven05()
    t5 = cm.fred_series("T5YIE")
    both = pd.concat([g, t5], axis=1).dropna()
    out = {"gsw_start": str(g.index[0].date()),
           "overlap_level_corr": float(both.corr().iloc[0, 1]),
           "overlap_mean_gsw_minus_t5yie": float((both.iloc[:, 0] - both.iloc[:, 1]).mean()),
           "overlap_60d_change_corr": float(both.diff(60).corr().iloc[0, 1]),
           "share_months_above_2_1999_2002": float((g.loc["1999":"2002"].resample("ME").last() > 2.0).mean())}
    etf = cm.signal_close_df(["SPY", "XLE", "XLK", "XLU", "XLP", "XLI", "XLF", "XLB", "XLV", "IEF"])
    import holdout_data as hd
    bond = hd.holdout_daily_inputs("1982-01-01", "2002-12-31")["tr_index"]["IEF"]
    etf = etf.loc["1998-01-01":cm.MAIN_END_STR].dropna(subset=["SPY"])
    b = bond.reindex(etf.index).ffill()
    first = etf["IEF"].first_valid_index()
    etf["IEF"] = b.where(etf.index < first, etf["IEF"] / etf["IEF"].loc[first] * b.loc[first])
    etf = etf.dropna()
    # published semantics: value dated < T
    infl = cm._align(g, pd.DatetimeIndex(etf.index), same_date=False)
    rows = []
    for (a, e), lab in (((("1999-06-01", "2002-12-31")), "1999-06..2002 GSW gate"),
                        ((("2003-05-01", cm.MAIN_END_STR)), "2003-05..2026-08 GSW instead of T5YIE (comparability)")):
        reg = ph.regimes(etf, infl)
        reg = reg[reg.index >= pd.Timestamp(a) - pd.Timedelta(days=40)]
        nav = ph.nav_from_weights(cm.weights_from_regimes(reg).reindex(columns=ph.TRADE).fillna(0.0), etf, a, e)
        m = cm.metrics(nav)
        tir = {f"g={gg},i={ii}": float(((reg.growth == gg) & (reg.infl == ii)).mean()) for gg in (True, False)
               for ii in (True, False)}
        rows.append({"segment": lab, **{k: m[k] for k in ("start", "end", "cagr", "sharpe", "maxdd")}, **tir})
        if lab.startswith("1999"):
            out["regimes_1999_2002"] = {str(k.date()): v for k, v in reg.assign(level=infl.reindex(reg.index)).astype(str).to_dict(orient="index").items()}
    out["rows"] = rows
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    print(json.dumps({k: v for k, v in out.items() if k not in ("rows", "regimes_1999_2002")}, indent=1))
    (cm.OUT / "p6b_gsw_probe.json").write_text(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
