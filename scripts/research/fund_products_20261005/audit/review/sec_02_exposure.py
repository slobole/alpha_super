"""Reviewer scratch (secondary lens): TQQQ weights, Nasdaq look-through, gap-table sanity, realised betas."""
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import norgatedata

WT = Path(r"C:\Users\User\Documents\workspace\alpha_super\.claude\worktrees\nervous-colden-784bbf")
OUT = WT / "results" / "research" / "portfolio" / "fund_products_20261005" / "audit" / "review" / "secondary"
D = pickle.load(open(OUT / "dump.pkl", "rb"))
END = pd.Timestamp("2026-08-19")
LONG, EXACT = pd.Timestamp("2008-03-04"), pd.Timestamp("2012-10-02")
main = D["frames"]["main"][0]
house = D["frames"]["s1_house_cash"][0]


def px(sym, adj):
    p = norgatedata.price_timeseries(sym, stock_price_adjustment_setting=adj, padding_setting=norgatedata.PaddingType.NONE,
                                     start_date="2010-01-01", end_date="2026-08-19", timeseriesformat="pandas-dataframe")
    p.index = pd.to_datetime(p.index).normalize()
    return p


A = norgatedata.StockPriceAdjustmentType
tq_cs = px("TQQQ", A.CAPITALSPECIAL)
tq_none = px("TQQQ", A.NONE)
tq_tr = px("TQQQ", A.TOTALRETURN)
ndx = px("$NDX", A.NONE)["Close"].pct_change()
qqq = D["bench"]["QQQ"]

print("=== 1. TQQQ weight inside the TAA pods (shares from fills x CAPITALSPECIAL close / NAV)")
tqw = {}
for a in ("taa3x", "taa3x_1n"):
    tx = D["tx"][a]
    nav = D["nav"][a]
    t = tx[tx["asset_str"] == "TQQQ"]
    # fill price against the CAPITALSPECIAL open of the fill date
    op = tq_cs["Open"].reindex(t["date"]).to_numpy()
    ratio = t["fill_price_float"].to_numpy() / op
    print(a, "TQQQ fills", len(t), "first", t["date"].min().date(), "| fill/CAPITALSPECIAL open: min %.5f med %.5f max %.5f" % (np.nanmin(ratio), np.nanmedian(ratio), np.nanmax(ratio)),
          "| fill/TOTALRETURN open med %.4f min %.4f" % (np.nanmedian(t["fill_price_float"].to_numpy() / tq_tr["Open"].reindex(t["date"]).to_numpy()),
                                                       np.nanmin(t["fill_price_float"].to_numpy() / tq_tr["Open"].reindex(t["date"]).to_numpy())))
    shares = t.groupby("date")["amount_float"].sum().reindex(nav.index).fillna(0.0).cumsum()
    w = (shares * tq_cs["Close"].reindex(nav.index) / nav).loc[EXACT:END]
    tqw[a] = w
    me = w.groupby([w.index.year, w.index.month]).last()
    print(a, "daily mean %.4f p90 %.4f max %.4f (on %s) | month-end mean %.4f p90 %.4f max %.4f | >50%% days %.4f | <1%% days %.4f | min shares %.2f"
          % (w.mean(), w.quantile(.9), w.max(), w.idxmax().date(), me.mean(), me.quantile(.9), me.max(), (w > .5).mean(), (w < .01).mean(), shares.min()))
    path = D["path_old"][a]
    inv = (path["portfolio_value_float"] / path["total_value_float"]).reindex(w.index)
    print(a, "  TQQQ weight above the pod's invested weight on", int((w > inv + 1e-6).sum()), "days; max excess %.6f" % float((w - inv).max()))
    for d in ("2013-11-29", "2021-11-30", "2025-12-31"):
        d = pd.Timestamp(d)
        last = t[t["date"] <= d].tail(3)[["date", "amount_float", "fill_price_float"]]
        print(f"   {d.date()} shares {shares.loc[d]:.2f} x close {tq_cs['Close'].loc[d]:.4f} (unadjusted close {tq_none['Close'].loc[d]:.2f}) = {shares.loc[d] * tq_cs['Close'].loc[d]:,.0f}"
              f" / NAV {nav.loc[d]:,.0f} = {w.loc[d]:.4f} | pod invested {inv.loc[d]:.4f} | last fills: "
              + "; ".join(f"{r.date.date()} {r.amount_float:+.1f}@{r.fill_price_float:.3f}" for r in last.itertuples()))

print("=== 2. look-through")
idx = tqw["taa3x"].index
inv = lambda p: (p["portfolio_value_float"] / p["total_value_float"]).reindex(idx)  # noqa: E731
P = D["path_new"]
mom_inv = 0.5 * inv(P["ndx_atr_cap"]) + 0.5 * inv(P["ndx_natr_cap"])
mr_stock = 0.5 * inv(P["dv2_g_cash"]) + 0.5 * inv(P["hpi_g_cash"])
# stock weight inside the BIL-parking runs themselves: (portfolio value - BIL value) / NAV
mr_stock_bil = {}
for a in ("dv2_g", "hpi_g"):
    tx = D["tx"][a]
    nav = D["nav"][a]
    b = tx[tx["asset_str"] == "BIL"].groupby("date")["amount_float"].sum().reindex(nav.index).fillna(0.0).cumsum()
    bil_close = px("BIL", A.CAPITALSPECIAL)["Close"].reindex(nav.index).ffill()
    path = P[a]
    mr_stock_bil[a] = ((path["portfolio_value_float"] - b * bil_close) / path["total_value_float"]).reindex(idx)
mr_stock2 = 0.5 * mr_stock_bil["dv2_g"] + 0.5 * mr_stock_bil["hpi_g"]
print("MR stock weight: cash runs mean %.4f p90 %.4f max %.4f | BIL runs (pv - BIL) mean %.4f p90 %.4f max %.4f | max abs diff %.4f"
      % (mr_stock.mean(), mr_stock.quantile(.9), mr_stock.max(), mr_stock2.mean(), mr_stock2.quantile(.9), mr_stock2.max(), (mr_stock - mr_stock2).abs().max()))
print("MOM invested mean %.4f p90 %.4f max %.4f" % (mom_inv.mean(), mom_inv.quantile(.9), mom_inv.max()))

SH = {"GR1": ("taa3x", 1 / 3, 1 / 3, 1 / 3), "GR2": ("taa3x_1n", 1 / 3, 1 / 3, 1 / 3), "GR3": ("taa3x_1n", .5, .25, .25)}


def drift_weights(shape):
    """capsule weights at the prior close under the annual reset (house frame pods)."""
    taa, t, m, r = shape
    w = {taa: t, "ndx_atr_cap": m / 2, "ndx_natr_cap": m / 2, "dv2_g": r / 2, "hpi_g": r / 2}
    cols = list(w)
    R = main.loc[LONG:END, cols]
    tw = np.array([w[c] for c in cols])
    yrs = R.index.year.to_numpy()
    v = R.to_numpy()
    out = np.empty_like(v)
    pods = tw.copy()
    for i in range(len(R)):
        if i == 0 or yrs[i] != yrs[i - 1]:
            pods = tw * pods.sum()
        out[i] = pods / pods.sum()
        pods = pods * (1 + v[i])
    # weights AFTER session i's returns (close of i) = prior-close weights of i+1 without a reset
    W = pd.DataFrame(out, index=R.index, columns=cols)
    return W


res = {}
for n, shape in SH.items():
    taa, t, m, r = shape
    N = t * 3 * tqw[taa] + m * mom_inv
    E = N + r * mr_stock
    W = drift_weights(shape).shift(-1).reindex(idx)        # close-of-day weights (same timing as the close exposures)
    Wt, Wm, Wr = W[taa], W["ndx_atr_cap"] + W["ndx_natr_cap"], W["dv2_g"] + W["hpi_g"]
    inv_atr, inv_natr = inv(P["ndx_atr_cap"]), inv(P["ndx_natr_cap"])
    Nd = Wt * 3 * tqw[taa] + W["ndx_atr_cap"] * inv_atr + W["ndx_natr_cap"] * inv_natr
    Ed = Nd + W["dv2_g"] * inv(P["dv2_g_cash"]) + W["hpi_g"] * inv(P["hpi_g_cash"])
    dN, dE = N.idxmax(), E.idxmax()
    print(f"{n}: N target-weights mean {N.mean():.4f} p90 {N.quantile(.9):.4f} max {N.max():.4f} on {dN.date()} (TQQQ w {tqw[taa].loc[dN]:.3f}, MOM inv {mom_inv.loc[dN]:.3f}, MR stock {mr_stock.loc[dN]:.3f})")
    print(f"     E mean {E.mean():.4f} p90 {E.quantile(.9):.4f} max {E.max():.4f} on {dE.date()} (TQQQ w {tqw[taa].loc[dE]:.3f}, MOM inv {mom_inv.loc[dE]:.3f}, MR stock {mr_stock.loc[dE]:.3f})")
    print(f"     with drifted pod weights: N mean {Nd.mean():.4f} p90 {Nd.quantile(.9):.4f} max {Nd.max():.4f} on {Nd.idxmax().date()} (TAA share {Wt.loc[Nd.idxmax()]:.3f}) | E max {Ed.max():.4f} on {Ed.idxmax().date()}")
    top = N.nlargest(10)
    print("     N top-10 dates:", [str(d.date()) for d in top.index][:10])
    print(f"     days N > 1.2: {(N > 1.2).sum()}, N > 1.0: {(N > 1.0).sum()} of {len(N)}; by year max since 2015: {N.loc['2015':].max():.3f}")
    full = tqw[taa] > 0.9
    print(f"     MR stock weight when TQQQ weight > 0.9: mean {mr_stock[full].mean():.3f} (days {int(full.sum())}); when TQQQ weight < 0.1: {mr_stock[tqw[taa] < 0.1].mean():.3f}; corr(TQQQ w, MR stock w) {tqw[taa].corr(mr_stock):.3f}")
    res[n] = {"N": [float(N.mean()), float(N.quantile(.9)), float(N.max()), str(dN.date())], "E": [float(E.mean()), float(E.quantile(.9)), float(E.max()), str(dE.date())],
              "N_drift": [float(Nd.mean()), float(Nd.quantile(.9)), float(Nd.max()), str(Nd.idxmax().date())], "E_drift_max": [float(Ed.max()), str(Ed.idxmax().date())]}
    res[n]["series"] = {"N": N, "E": E, "Nd": Nd, "Ed": Ed}

print("=== 3. realised betas to the Nasdaq-100 (house-frame pod returns; exposure = prior-close and same-close weight)")
x = ndx.reindex(idx)
mom_r = 0.5 * house["ndx_atr_cap"].reindex(idx) + 0.5 * house["ndx_natr_cap"].reindex(idx)
mr_r = 0.5 * house["dv2_g"].reindex(idx) + 0.5 * house["hpi_g"].reindex(idx)
bil = main["tbill"].reindex(idx)


def beta(y, xx, mask):
    yy, xv = y[mask].to_numpy(), xx[mask].to_numpy()
    return float(np.cov(yy, xv)[0, 1] / xv.var(ddof=1)), int(mask.sum())


for lab_, mask in (("all days", x.notna()), ("NDX <= -2%", x <= -0.02), ("NDX <= -3%", x <= -0.03)):
    # MOM: return per unit of invested weight (prior close)
    mi = mom_inv.shift(1)
    ok = mask & (mi > 0.5)
    b_mom, k1 = beta(mom_r / mi, x, ok)
    ratio_mom = float((mom_r[ok]).sum() / (mi[ok] * x[ok]).sum())
    ms = mr_stock                      # the MR pods trade at the open: same-day close weight is the day's exposure
    ok2 = mask & (ms > 0.5)
    mr_ex = mr_r - (1 - ms.clip(upper=1)) * bil
    b_mr, k2 = beta(mr_ex / ms, x, ok2)
    ratio_mr = float(mr_ex[ok2].sum() / (ms[ok2] * x[ok2]).sum())
    tq_r = tq_tr["Close"].pct_change().reindex(idx)
    b_tq, k3 = beta(tq_r, x, mask & tq_r.notna())
    print(f"{lab_:12s} MOM beta per invested unit {b_mom:.2f} (n {k1}), loss ratio {ratio_mom:.2f} | MR stocks beta per stock unit {b_mr:.2f} (n {k2}), loss ratio {ratio_mr:.2f} | TQQQ beta {b_tq:.2f} (n {k3})")

print("=== 4. the realised worst days against the gap arithmetic (E at the prior close x NDX move)")
BOOKS = {"GR1": {"taa3x": 1 / 3, "ndx_atr_cap": 1 / 6, "ndx_natr_cap": 1 / 6, "dv2_g": 1 / 6, "hpi_g": 1 / 6},
         "GR2": {"taa3x_1n": 1 / 3, "ndx_atr_cap": 1 / 6, "ndx_natr_cap": 1 / 6, "dv2_g": 1 / 6, "hpi_g": 1 / 6},
         "GR3": {"taa3x_1n": .5, "ndx_atr_cap": .125, "ndx_natr_cap": .125, "dv2_g": .125, "hpi_g": .125}}
for n, w in BOOKS.items():
    cols = list(w)
    R = main.loc[LONG:END, cols]
    tw = np.array([w[c] for c in cols])
    yrs = R.index.year.to_numpy()
    v = R.to_numpy()
    r = np.empty(len(R))
    contrib = np.empty_like(v)
    pods = tw.copy()
    for i in range(len(R)):
        if i == 0 or yrs[i] != yrs[i - 1]:
            pods = tw * pods.sum()
        eq = pods.sum()
        contrib[i] = pods * v[i] / eq
        pods = pods * (1 + v[i])
        r[i] = pods.sum() / eq - 1
    r = pd.Series(r, index=R.index)
    C = pd.DataFrame(contrib, index=R.index, columns=cols)
    taa = cols[0]
    for d in list(r.loc[EXACT:].nsmallest(4).index) + [pd.Timestamp("2020-03-16"), pd.Timestamp("2020-03-12"), pd.Timestamp("2025-04-04")]:
        prev = idx[idx.get_loc(d) - 1]
        Eprev, Etoday = res[n]["series"]["E"].loc[prev], res[n]["series"]["E"].loc[d]
        print(f"{n} {d.date()}: book {r.loc[d]:+.4f} | NDX {x.loc[d]:+.4f} QQQ {qqq.loc[d]:+.4f} | pods: TAA {C.loc[d, taa]:+.4f} MOM {C.loc[d, ['ndx_atr_cap', 'ndx_natr_cap']].sum():+.4f} MR {C.loc[d, ['dv2_g', 'hpi_g']].sum():+.4f}"
              f" | TQQQ w prev {tqw[taa].loc[prev]:.2f} MOM inv prev {mom_inv.loc[prev]:.2f} MR stock prev {mr_stock.loc[prev]:.2f} today {mr_stock.loc[d]:.2f}"
              f" | E prev {Eprev:.2f} -> arithmetic {Eprev * x.loc[d]:+.4f} (E today {Etoday:.2f} -> {Etoday * x.loc[d]:+.4f}) | realised / NDX = {r.loc[d] / x.loc[d]:.2f}")
    res[n].pop("series")
json.dump(res, open(OUT / "sec_02_exposure.json", "w"), indent=1, default=str)
