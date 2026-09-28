"""Review R5: (a) how much of the TAA record sits in the BTAL-illiquid era (2012-10..2018-12, BTAL median
turnover USD 10-55K/day, 55-79 zero-volume days/yr 2013-15); BTAL fills on zero-volume bars; BTAL weight then.
(b) TQQQ nominal-price/split history and worst daily return in the data. (c) buy notional per rebalance as % of
prior NAV (buying power needed at 09:23 ET before same-auction sells print) for the four pods.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
from data.norgate_loader import load_price_timeseries  # noqa: E402
LED = REPO / "results/research/strategy_readiness_audit_20260928/review_trade/ledgers"
OUT = REPO / "results/research/strategy_readiness_audit_20260928/review_trade"
END = "2026-09-25"


def stats(nav):
    r = nav.pct_change().dropna()
    yrs = (nav.index[-1] - nav.index[0]).days / 365.25
    dd = (nav / nav.cummax() - 1).min()
    return {"cagr_pct": round(100 * ((nav.iloc[-1] / nav.iloc[0]) ** (1 / yrs) - 1), 2),
            "sharpe": round(float(r.mean() / r.std() * np.sqrt(252)), 3), "max_dd_pct": round(100 * float(dd), 1)}


out = {}
btal = load_price_timeseries("BTAL", start_date_str="2011-09-01", end_date_str=END)
zero_vol_days = set(btal.index[btal["Volume"] == 0])
for key in ["taa3x", "taa1n", "btal_qqq", "ndx_vxn"]:
    nav = pd.read_csv(LED / f"{key}_nav.csv", index_col=0, parse_dates=True)["nav"]
    tx = pd.read_csv(LED / f"{key}_tx.csv", parse_dates=["bar"])
    res = {"full": stats(nav)}
    if key != "ndx_vxn":
        res["2012-10_to_2018-12 (BTAL illiquid)"] = stats(nav.loc[:"2018-12-31"])
        res["2019-01_to_2026-09 (BTAL >= USD 1M/day)"] = stats(nav.loc["2018-12-31":])
        bt = tx[tx["asset"] == "BTAL"]
        res["btal_fills"] = int(len(bt))
        res["btal_fills_on_zero_volume_bar"] = int(bt["bar"].isin(zero_vol_days).sum())
        res["btal_fills_2012_2018"] = int((bt["bar"] <= "2018-12-31").sum())
        # BTAL position weight at month-ends pre-2019 (from cumulative shares * adjusted close / NAV)
        pos = bt.set_index("bar")["amount"].groupby(level=0).sum().cumsum()
        pos = pos.reindex(nav.index, method="ffill").fillna(0.0)
        w = pos * btal["Close"].reindex(nav.index).ffill() / nav
        res["btal_mean_weight_pct_2012_2018"] = round(100 * float(w.loc[:"2018-12-31"].mean()), 1)
        res["btal_mean_weight_pct_2019_2026"] = round(100 * float(w.loc["2019-01-01":].mean()), 1)
    # buy notional on each rebalance bar / prior NAV
    tx["notional"] = tx["amount"] * tx["price"]
    nav_prev = nav.shift(1)
    g = tx.groupby("bar")
    buys = g["notional"].apply(lambda s: s[s > 0].sum())
    sells = -g["notional"].apply(lambda s: s[s < 0].sum())
    bp = (buys / buys.index.map(nav_prev)).dropna()
    res["buy_notional_pct_nav_per_rebalance"] = {"median": round(100 * float(bp.median()), 1),
                                                  "p95": round(100 * float(bp.quantile(0.95)), 1),
                                                  "max": round(100 * float(bp.max()), 1),
                                                  "max_date": str(bp.idxmax().date())}
    net = ((buys - sells) / buys.index.map(nav_prev)).dropna()
    res["net_buy_minus_sell_pct_nav_max"] = round(100 * float(net.max()), 2)
    out[key] = res
tq = load_price_timeseries("TQQQ", start_date_str="2010-01-01", end_date_str=END)
k = tq["Unadjusted Close"] / tq["Close"]
chg = k.pct_change().abs()
out["tqqq"] = {"split_like_scale_changes": [str(d.date()) for d in chg[chg > 0.2].index],
               "worst_daily_return_pct": round(100 * float(tq["Close"].pct_change().min()), 1),
               "worst_day": str(tq["Close"].pct_change().idxmin().date()),
               "nominal_close_min": round(float(tq["Unadjusted Close"].min()), 2),
               "nominal_close_last": round(float(tq["Unadjusted Close"].iloc[-1]), 2)}
qqq = load_price_timeseries("QQQ", start_date_str="1999-03-10", end_date_str=END)["Close"].pct_change()
out["qqq_worst_daily_pct_since_1999"] = round(100 * float(qqq.min()), 1)
out["qqq_days_below_minus_10pct"] = int((qqq < -0.10).sum())
(OUT / "r5_btal_era_tqqq_buying_power.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
print(json.dumps(out, indent=1))
