"""A7/A8 accounting probe on the baseline ledgers (100K, legacy units) for CTC, VIXM and Trinity.

Measures, per strategy:
  - negative cash (unfinanced in the engine): frequency, depth, and the financing bound at DTB3 + spread
    (IBKR Pro-style benchmark + 1.5% below USD 100K; + 1.0% to 1M; + 0.5% above), and the omitted credit interest
    on positive cash at max(DTB3 - 0.5%, 0) (conservative omission);
  - dividends credited (withholding used by the strategy) and the bound for 25% US withholding on long dividends
    (Israeli-resident treaty rate), plus manufactured dividends paid on shorts;
  - CTC short borrow charged and gross long > 100% of NAV days;
  - orders per year, NAV turnover, fills on zero-volume bars;
  - small-account commission drag (IBKR Fixed USD 0.005/sh min USD 1, cap 1% of value; Tiered USD 0.0035/sh min
    USD 0.35) at USD 30K using raw (unadjusted) share counts at the fill date.

DTB3 is read from a private copy in this audit's _cache (never the shared FRED cache), lagged one session.
"""
from __future__ import annotations

import pickle

import numpy as np
import pandas as pd

import tc_common as c

dtb3 = pd.read_csv(c.CACHE / "DTB3_tierc_copy.csv", parse_dates=["observation_date"], index_col="observation_date")
dtb3 = pd.to_numeric(dtb3["DTB3"], errors="coerce").ffill() / 100.0

LOADERS = {"vixm": c.load_vixm, "ctc": c.load_ctc_workaround, "trin": c.load_trin}
BACKTEST_START = {"vixm": "2011-03-01", "ctc": "2004-01-01", "trin": "2007-06-01"}


def cagr(r: pd.Series) -> float:
    eq = (1 + r).cumprod()
    yrs = (r.index[-1] - r.index[0]).days / 365.25
    return float(eq.iloc[-1] ** (1 / yrs) - 1) * 100


def ibkr_fee(shares: float, value: float, plan: str) -> float:
    shares = abs(shares)
    value = abs(value)
    if shares == 0:
        return 0.0
    if plan == "fixed":
        fee = max(1.0, 0.005 * shares)
        return min(fee, 0.01 * value) if value > 0 else fee
    fee = max(0.35, 0.0035 * shares)
    return min(fee, 0.01 * value) if value > 0 else fee


def probe(name: str) -> dict:
    base = pickle.load(open(c.CACHE / f"{name}_baseline.pkl", "rb"))
    df = LOADERS[name]()
    res = base["results"]
    cash = res["cash"].astype(float)
    tv = res["total_value"].astype(float)
    r = tv.pct_change().dropna()
    rf = dtb3.reindex(tv.index, method="ffill").shift(1).fillna(0.0)
    frac = (cash / tv)
    out = {"cagr_pct": cagr(r)}
    out["cash_frac"] = {"min": float(frac.min()), "min_date": str(frac.idxmin().date()),
                        "share_days_negative": float((frac < -1e-9).mean()),
                        "mean_when_negative": float(frac[frac < -1e-9].mean()) if (frac < -1e-9).any() else 0.0,
                        "share_days_below_minus_5pct": float((frac < -0.05).mean()),
                        "mean_cash_frac": float(frac.mean())}
    # financing bound: charge debit interest on prior-close negative cash at DTB3 + 1.5%; credit omitted at DTB3-0.5%
    prev_cash = cash.shift(1)
    prev_tv = tv.shift(1)
    debit = (-prev_cash).clip(lower=0) * (rf + 0.015) / 252.0 / prev_tv
    credit = prev_cash.clip(lower=0) * (rf - 0.005).clip(lower=0) / 252.0 / prev_tv
    r_fin = (r - debit.reindex(r.index).fillna(0.0))
    r_both = (r_fin + credit.reindex(r.index).fillna(0.0))
    out["financing"] = {
        "cagr_with_debit_interest_pct": cagr(r_fin),
        "debit_interest_impact_pp": cagr(r_fin) - cagr(r),
        "cagr_with_debit_and_credit_pct": cagr(r_both),
        "credit_interest_omitted_pp": cagr(r_both) - cagr(r_fin),
        "last3y_debit_impact_pp": cagr(r_fin[r_fin.index >= "2023-09-25"]) - cagr(r[r.index >= "2023-09-25"]),
    }
    # dividends
    div = base["div_ledger"]
    if div is not None and len(div):
        div = div.copy()
        div["ex_date"] = pd.to_datetime(div["ex_date"])
        long_gross = div.loc[div["gross_dividend_cash_float"] > 0, "gross_dividend_cash_float"]
        short_paid = div.loc[div["gross_dividend_cash_float"] < 0, "gross_dividend_cash_float"]
        wh_used = float(div["withholding_cash_float"].sum())
        # 25% withholding bound: subtract 25% of each long dividend (minus what was already withheld) on its ex-date
        extra = div.copy()
        extra["extra_wh"] = extra["gross_dividend_cash_float"].clip(lower=0) * 0.25 - extra["withholding_cash_float"]
        extra_by_day = extra.groupby("ex_date")["extra_wh"].sum()
        r_wh = r - (extra_by_day.reindex(r.index).fillna(0.0) / prev_tv.reindex(r.index))
        out["dividends"] = {"events": int(len(div)), "long_gross_total": float(long_gross.sum()),
                            "short_manufactured_paid_total": float(short_paid.sum()),
                            "withholding_used_total": wh_used,
                            "assets": sorted(set(div["asset_str"])),
                            "cagr_with_25pct_withholding_pct": cagr(r_wh),
                            "withholding_25pct_impact_pp": cagr(r_wh) - cagr(r),
                            "last3y_withholding_25pct_impact_pp": cagr(r_wh[r_wh.index >= "2023-09-25"])
                            - cagr(r[r.index >= "2023-09-25"])}
    # trades, turnover, zero-volume fills, small-account fees
    tx = base["tx"].copy()
    tx["bar"] = pd.to_datetime(tx["bar"])
    tx = tx[tx["bar"] >= pd.Timestamp(BACKTEST_START[name])]
    yrs = (tv.index[-1] - tv.index[0]).days / 365.25
    tx["navprev"] = tx["bar"].map(prev_tv)
    tx["frac"] = (tx["amount"] * tx["price"]).abs() / tx["navprev"]
    zero_vol = []
    fee_rows = []
    for _, row in tx.iterrows():
        a, d = row["asset"], row["bar"]
        vol = float(df.loc[d, (a, "Volume")]) if (a, "Volume") in df.columns else np.nan
        if vol == 0:
            zero_vol.append(f"{d.date()} {a}")
        unadj_open = float(df.loc[d, (a, "Open")] * df.loc[d, (a, "Unadjusted Close")] / df.loc[d, (a, "Close")])
        for cap in (30_000.0, 100_000.0):
            value = row["frac"] * cap
            sh = np.floor(value / unadj_open) if unadj_open > 0 else 0
            fee_rows.append({"bar": d, "asset": a, "cap": cap, "value": value, "shares": sh,
                             "fixed": ibkr_fee(sh, value, "fixed") if sh > 0 else 0.0,
                             "tiered": ibkr_fee(sh, value, "tiered") if sh > 0 else 0.0,
                             "sub_one_share": bool(sh == 0 and value > 0)})
    fees = pd.DataFrame(fee_rows)
    out["orders"] = {"n": int(len(tx)), "per_year": float(len(tx) / yrs), "nav_turnover_per_year": float(tx["frac"].sum() / yrs),
                     "fills_on_zero_volume_bars": len(zero_vol), "zero_volume_examples": zero_vol[:10],
                     "order_value_median_frac_nav": float(tx["frac"].median()),
                     "order_value_p10_frac_nav": float(tx["frac"].quantile(0.1))}
    backtest_comm = float(tx["commission"].sum())
    for cap in (30_000.0, 100_000.0):
        f = fees[fees["cap"] == cap]
        out[f"fees_{int(cap)}"] = {
            "fixed_pp_per_year": float(f["fixed"].sum() / yrs / cap * 100),
            "tiered_pp_per_year": float(f["tiered"].sum() / yrs / cap * 100),
            "orders_below_one_share": int(f["sub_one_share"].sum()),
            "median_order_value_usd": float(f["value"].median()),
            "share_orders_min_fee_binding_fixed": float(((f["shares"] * 0.005) < 1.0).mean()),
        }
    out["backtest_commission_pp_per_year_at_100k"] = backtest_comm / yrs / 100_000.0 * 100
    if name == "ctc":
        b = base["borrow"]
        out["borrow_total_usd"] = float(b["borrow_fee_float"].sum()) if b is not None and len(b) else 0.0
        out["borrow_pp_per_year_approx"] = out["borrow_total_usd"] / yrs / float(tv.mean()) * 100
        tw = base["daily_target"]
        risk = list(c.ctc.UNIVERSE_ASSET_TUPLE)
        longg = tw[risk].clip(lower=0).sum(axis=1)
        shortg = -tw[risk].clip(upper=0).sum(axis=1)
        out["targets"] = {"share_days_long_gross_gt_1": float((longg > 1 + 1e-9).mean()),
                          "max_long_gross": float(longg.max()),
                          "share_days_with_shorts": float((shortg > 1e-9).mean()),
                          "max_short_gross": float(shortg.max()), "mean_short_gross": float(shortg.mean()),
                          "mean_long_risk_gross": float(longg.mean()), "mean_shy": float(tw["SHY"].mean()),
                          "last3y_mean_short_gross": float(shortg[shortg.index >= "2023-09-25"].mean()),
                          "short_assets_ever": sorted([a for a in risk if (tw[a] < -1e-9).any()]),
                          "long_assets_ever": sorted([a for a in risk if (tw[a] > 1e-9).any()])}
    return out


if __name__ == "__main__":
    import json
    import sys
    names = sys.argv[1:] or ["vixm", "ctc", "trin"]
    for n in names:
        o = probe(n)
        c.dump(o, f"{n}/a8_accounting_probe.json")
        print(n, json.dumps(o, indent=1, default=str)[:4000], flush=True)
