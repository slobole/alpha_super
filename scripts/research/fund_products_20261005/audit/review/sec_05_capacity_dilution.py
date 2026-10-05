"""Reviewer scratch: does pooling the MR pods' BIL orders dilute the ETF one-day gate of the book (close-auction route)?
Re-uses the order table logic of sec_03 (own implementation)."""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import norgatedata

WT = Path(r"C:\Users\User\Documents\workspace\alpha_super\.claude\worktrees\nervous-colden-784bbf")
OUT = WT / "results" / "research" / "portfolio" / "fund_products_20261005" / "audit" / "review" / "secondary"
D = pickle.load(open(OUT / "dump.pkl", "rb"))
END, START = pd.Timestamp("2026-08-19"), pd.Timestamp("2023-08-21")
house = D["sleeve"]
BOOKS = {"GR1": {"taa3x": 1 / 3, "ndx_atr_cap": 1 / 6, "ndx_natr_cap": 1 / 6, "dv2_g": 1 / 6, "hpi_g": 1 / 6},
         "GR2": {"taa3x_1n": 1 / 3, "ndx_atr_cap": 1 / 6, "ndx_natr_cap": 1 / 6, "dv2_g": 1 / 6, "hpi_g": 1 / 6},
         "GR3": {"taa3x_1n": .5, "ndx_atr_cap": .125, "ndx_natr_cap": .125, "dv2_g": .125, "hpi_g": .125}}
ETF = {"BIL", "BTAL", "DBC", "GLD", "IEF", "QQQ", "SPY", "TLT", "TQQQ", "UUP"}
adv = {}
for tk in ETF:
    p = norgatedata.price_timeseries(tk, stock_price_adjustment_setting=norgatedata.StockPriceAdjustmentType.CAPITALSPECIAL, padding_setting=norgatedata.PaddingType.ALLMARKETDAYS,
                                     start_date="2023-03-01", end_date="2026-08-19", timeseriesformat="pandas-dataframe")
    p.index = pd.to_datetime(p.index).normalize()
    adv[tk] = (p["Close"] * p["Volume"]).replace(0.0, np.nan).rolling(60, min_periods=20).median().shift(1)


def drift(w):
    cols = list(w)
    R = house.loc[START - pd.Timedelta(days=10):END, cols]
    tw = np.array([w[c] for c in cols])
    yrs, v = R.index.year.to_numpy(), R.to_numpy()
    out, pods = np.empty_like(v), tw.copy()
    for i in range(len(R)):
        if i > 0 and yrs[i] != yrs[i - 1]:
            pods = tw * pods.sum()
        out[i] = pods / pods.sum()
        pods = pods * (1 + v[i])
    return pd.DataFrame(out, index=R.index, columns=cols)


for name, w in BOOKS.items():
    W = drift(w)
    rows = []
    for a in w:
        t = D["tx"][a]
        t = t[(t["date"] >= START) & (t["date"] <= END) & t["asset_str"].isin(ETF)].copy()
        t["bf"] = t["signed_notional_float"].abs().to_numpy() / D["nav"][a].shift(1).reindex(t["date"]).to_numpy() * W[a].reindex(t["date"]).to_numpy()
        t["alias"] = a
        t["urgent"] = a in ("dv2_g", "hpi_g")
        rows.append(t[["date", "asset_str", "bf", "alias", "urgent"]])
    o = pd.concat(rows)
    d = o.groupby(["date", "asset_str", "urgent"], as_index=False)["bf"].sum()
    d["adv60"] = [adv[t].get(dt, np.nan) for dt, t in zip(d["date"], d["asset_str"])]
    d = d.dropna()
    k = (d["bf"] / d["adv60"]).to_numpy()
    is_bil_mr = d["urgent"].to_numpy()
    print(f"{name}: ETF order rows {len(d)} of which BIL parking rows from the MR pods {int(is_bil_mr.sum())}")
    for lab, m in (("all ETF rows (as published)", np.ones(len(d), bool)), ("without the MR pods' BIL rows", ~is_bil_mr)):
        kk = k[m]
        a95, amax = 0.05 / np.percentile(kk, 95), 0.20 / kk.max()
        print(f"   {lab:32s}: p95 gate binds at ${a95 / 1e6:.2f}M, max gate at ${amax / 1e6:.2f}M -> ETF one-day gate holds to ${min(a95, amax) / 1e6:.2f}M")
    taa = [a for a in w if a.startswith("taa")][0]
    print(f"   leg-consistent figure: TAA leg alone holds to ${0.05 / np.percentile((d[~d['urgent']]['bf'] / d[~d['urgent']]['adv60']).to_numpy(), 95) * w[taa] / 1e6:.2f}M of pod AUM = the line above")
