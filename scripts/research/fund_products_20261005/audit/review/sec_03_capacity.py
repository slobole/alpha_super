"""Reviewer scratch (secondary lens): independent rebuild of the route model for GR1 / GR2 / GR3 / S9 and the legs.
Own order table, own drifted weights, own liquidity pull (norgatedata), own gate arithmetic, fine AUM search."""
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
START = pd.Timestamp("2023-08-21")
YEARS = (END - START).days / 365.25
house = D["sleeve"]
main = D["frames"]["main"][0]
exact = D["frames"]["s6_exact"][0]

ETF_PODS = {"core5", "btal_qqq", "taa3x", "taa3x_1n"}
NASDAQ_PODS = {"ndx_vxn", "ndx_atr_cap", "ndx_natr_cap"}
URGENT_PODS = {"dv2", "hpi_vote", "dv2_g", "hpi_g"}
THIN = {"BTAL", "UUP", "DBC"}
MOO = dict(soft=0.0005, hard=0.0010, lam_n=66.4, lam_o=40.0)
MOC = dict(soft=0.0025, hard=0.0050, lam_n=8.2, lam_o=8.2)
BLOCK_COST = 0.0015

BOOKS = {
    "GR1": {"taa3x": 1 / 3, "ndx_atr_cap": 1 / 6, "ndx_natr_cap": 1 / 6, "dv2_g": 1 / 6, "hpi_g": 1 / 6},
    "GR2": {"taa3x_1n": 1 / 3, "ndx_atr_cap": 1 / 6, "ndx_natr_cap": 1 / 6, "dv2_g": 1 / 6, "hpi_g": 1 / 6},
    "GR3": {"taa3x_1n": .5, "ndx_atr_cap": .125, "ndx_natr_cap": .125, "dv2_g": .125, "hpi_g": .125},
    "S9": {"taa3x_1n": 0.384, "ndx_vxn": 0.256, "core5": 0.18, "btal_qqq": 0.18},
    "leg MR": {"dv2_g": .5, "hpi_g": .5},
    "leg MOM": {"ndx_atr_cap": .5, "ndx_natr_cap": .5},
    "leg TAA 3x": {"taa3x": 1.0},
    "leg TAA 3x 1N": {"taa3x_1n": 1.0},
}

# ---- orders
aliases = sorted({a for w in BOOKS.values() for a in w})
rows = []
for a in aliases:
    t = D["tx"][a]
    t = t[(t["date"] >= START) & (t["date"] <= END)].copy()
    prior = D["nav"][a].shift(1)
    t["frac"] = t["signed_notional_float"].abs().to_numpy() / prior.reindex(t["date"]).to_numpy()
    t["alias"] = a
    rows.append(t[["date", "alias", "asset_str", "frac", "signed_notional_float"]])
orders = pd.concat(rows, ignore_index=True)
etf_tickers = set(orders.loc[orders["alias"].isin(ETF_PODS), "asset_str"]) | {"BIL"}
nasdaq_tickers = set(orders.loc[orders["alias"].isin(NASDAQ_PODS), "asset_str"])
print("orders", len(orders), "tickers", orders["asset_str"].nunique(), "| ETF tickers", sorted(etf_tickers))
mr_tickers = set(orders.loc[orders["alias"].isin({"dv2_g", "hpi_g"}), "asset_str"])
print("MR tickers that are tagged ETF:", sorted(mr_tickers & etf_tickers), "| MR tickers tagged Nasdaq (cost lambda only):", len(mr_tickers & nasdaq_tickers), "of", len(mr_tickers))

# ---- liquidity (own pull)
liq = {}
miss = []
for tk in sorted(orders["asset_str"].unique()):
    try:
        p = norgatedata.price_timeseries(tk, stock_price_adjustment_setting=norgatedata.StockPriceAdjustmentType.CAPITALSPECIAL,
                                         padding_setting=norgatedata.PaddingType.ALLMARKETDAYS, start_date="2023-03-01", end_date="2026-08-19",
                                         timeseriesformat="pandas-dataframe")
    except Exception:
        miss.append(tk)
        continue
    if p is None or len(p) == 0:
        miss.append(tk)
        continue
    p.index = pd.to_datetime(p.index).normalize()
    dv = (p["Close"] * p["Volume"]).replace(0.0, np.nan)
    liq[tk] = pd.DataFrame({"adv20": dv.rolling(20, min_periods=10).median().shift(1), "adv60": dv.rolling(60, min_periods=20).median().shift(1),
                            "sigma": p["Close"].pct_change(fill_method=None).rolling(60, min_periods=20).std().shift(1)})
print("missing price tickers:", miss)


def look(d, tk, col):
    f = liq.get(tk)
    if f is None or d not in f.index:
        return np.nan
    return f.at[d, col]


for c in ("adv20", "adv60", "sigma"):
    orders[c] = [look(d, t, c) for d, t in zip(orders["date"], orders["asset_str"])]


def drift(w, start):
    """prior-close pod weights, annual reset, book started at `start` (house returns)."""
    cols = list(w)
    R = house.loc[start:END, cols]
    tw = np.array([w[c] for c in cols])
    tw = tw / tw.sum()
    yrs = R.index.year.to_numpy()
    v = R.to_numpy()
    out = np.empty_like(v)
    pods = tw.copy()
    for i in range(len(R)):
        if i > 0 and yrs[i] != yrs[i - 1]:
            pods = tw * pods.sum()
        out[i] = pods / pods.sum()
        pods = pods * (1 + v[i])
    return pd.DataFrame(out, index=R.index, columns=cols)


def gates(df, route, aum):
    """df: date, asset_str, bf (book fraction), urgent, etf, nasdaq, adv20, adv60, sigma. Returns (ok, cost, detail)."""
    dol = df["bf"].to_numpy() * aum
    p20, p60, sig = dol / df["adv20"].to_numpy(), dol / df["adv60"].to_numpy(), df["sigma"].to_numpy()
    etf, urg, thin, nq = df["etf"].to_numpy(), df["urgent"].to_numpy(), df["asset_str"].isin(THIN).to_numpy(), df["nasdaq"].to_numpy()
    sym = df["asset_str"].to_numpy()
    cost = np.zeros(len(df))
    det = {}
    ok = True

    def auction(m, P, lab):
        nonlocal ok
        if not m.any():
            return
        lam = np.where(nq[m], P["lam_n"], P["lam_o"]) / 1e4
        cost[m] = dol[m] * lam * np.sqrt(p20[m] / 0.01)
        a, b = np.percentile(p20[m], 95), np.percentile(p20[m], 99)
        g = a <= P["soft"] and b <= P["hard"]
        det[lab] = (g, float(a), float(b), sym[m][np.argmax(p20[m])], "p95" if a > P["soft"] else ("p99" if b > P["hard"] else ""))
        ok = ok and g

    def worked(m, maxday, lab):
        nonlocal ok
        if not m.any():
            return
        days = np.clip(np.ceil(p60[m] / 0.10), 1.0, maxday)
        dp = p60[m] / days
        cost[m] = dol[m] * sig[m] * np.sqrt(dp)
        a, b = np.percentile(dp, 95), dp.max()
        g = a <= 0.05 and b <= 0.20
        det[lab] = (g, float(a), float(b), sym[m][np.argmax(dp)], "p95" if a > 0.05 else ("max" if b > 0.20 else ""))
        ok = ok and g

    if route in ("MOO", "MOC"):
        auction(~etf, MOO if route == "MOO" else MOC, "stock_auction")
        worked(etf, 1.0, "etf_one_day")
    else:
        auction(~etf & urg, MOC, "urgent_stock_moc")
        worked(~urg & ~thin, 5.0, "worked")
        cost[thin & ~urg] = dol[thin & ~urg] * BLOCK_COST
        worked(etf & urg, 1.0, "urgent_etf_one_day")
    return ok, float(cost.sum()) / aum / YEARS, det


def max_aum(df, route):
    grid = 10 ** np.arange(5.0, 9.5, 0.01)
    last = None
    for a in grid:
        ok, _, det = gates(df, route, a)
        if not ok:
            return last, a, {k: v for k, v in det.items() if not v[0]}
        last = a
    return last, None, {}


def stats_cagr(frame, w, start):
    cols = list(w)
    R = frame.loc[start:END, cols]
    tw = np.array([w[c] for c in cols])
    yrs = R.index.year.to_numpy()
    v = R.to_numpy()
    eq, pods = 1.0, tw.copy()
    for i in range(len(R)):
        if i > 0 and yrs[i] != yrs[i - 1]:
            pods = tw * pods.sum()
        pods = pods * (1 + v[i])
    n = len(R)
    bil = frame.loc[start:END, "tbill"].to_numpy()
    return pods.sum() ** (252 / n) - 1 - (np.prod(1 + bil) ** (252 / n) - 1)


res = {}
for name, w in BOOKS.items():
    W_study = drift(w, START - pd.Timedelta(days=10))          # as the study: book started ten days before the window
    W_true = drift(w, pd.Timestamp("2012-10-02"))              # drift carried from the running book
    out = {}
    for tag, W in (("study_start", W_study), ("running_book", W_true)):
        bo = orders[orders["alias"].isin(w)].copy()
        wv = W.reindex(bo["date"])
        bo["bf"] = bo["frac"].to_numpy() * np.array([wv.iloc[i][a] for i, a in enumerate(bo["alias"])])
        bo["urgent"] = bo["alias"].isin(URGENT_PODS)
        daily = bo.groupby(["date", "asset_str", "urgent"], as_index=False).agg(bf=("bf", "sum"), adv20=("adv20", "first"), adv60=("adv60", "first"), sigma=("sigma", "first"))
        daily["etf"] = daily["asset_str"].isin(etf_tickers)
        daily["nasdaq"] = daily["asset_str"].isin(nasdaq_tickers)
        cov = daily.dropna(subset=["adv20", "adv60", "sigma"])
        rec = {"orders": int(len(daily)), "uncovered": int(len(daily) - len(cov)), "raw_fills_per_year": float(len(bo) / YEARS)}
        for route in ("MOO", "MOC", "worked+blocks"):
            hold, fail, det = max_aum(cov, route)
            grid_levels = {f"{a / 1e6:g}M": gates(cov, route, a)[:2] for a in (2.5e6, 5e6, 1e7, 2.5e7)}
            rec[route] = {"gates_hold_to": hold, "first_fail": fail, "failing": {k: (v[1], v[2], v[3], v[4]) for k, v in det.items()}, "levels": grid_levels}
        k = (cov["bf"] / cov["adv60"]).to_numpy()
        rec["participation"] = {"p90": 0.05 / np.percentile(k, 90), "p99": 0.05 / np.percentile(k, 99), "max": 0.05 / k.max(), "max_symbol": cov["asset_str"].to_numpy()[np.argmax(k)]}
        for aum in (2.5e6, 1e7, 3.45e7):
            part = k * aum
            rec["participation"][f"orders_per_year_above_5pct_at_{aum / 1e6:g}M"] = float((part > 0.05).sum() / YEARS)
            rec["participation"][f"largest_order_share_of_day_at_{aum / 1e6:g}M"] = float(part.max())
        # per-leg screen expressed in book AUM
        legs = {}
        bo2 = bo.dropna(subset=["adv60"])
        capsule = {"taa3x": "TAA", "taa3x_1n": "TAA", "ndx_atr_cap": "MOM", "ndx_natr_cap": "MOM", "ndx_vxn": "MOM", "dv2_g": "MR", "hpi_g": "MR", "core5": "DEF", "btal_qqq": "DEF"}
        bo2 = bo2.assign(cap=bo2["alias"].map(capsule))
        for c, sub in bo2.groupby("cap"):
            d2 = sub.groupby(["date", "asset_str"], as_index=False).agg(bf=("bf", "sum"), adv60=("adv60", "first"))
            kk = (d2["bf"] / d2["adv60"]).to_numpy()
            legs[c] = {"n": int(len(kk)), "book_aum_p99_at_5pct": float(0.05 / np.percentile(kk, 99)), "book_aum_max_at_5pct": float(0.05 / kk.max()),
                       "symbol": d2["asset_str"].to_numpy()[np.argmax(kk)]}
        rec["per_leg_screen_in_book_aum"] = legs
        out[tag] = rec
    out["exact_excess_cagr"] = float(stats_cagr(exact, w, pd.Timestamp("2012-10-02"))) if all(c in exact.columns for c in w) else None
    res[name] = out
    r = out["study_start"]
    print(f"\n== {name}: orders {r['orders']} uncovered {r['uncovered']} fills/yr {r['raw_fills_per_year']:.0f} | EXACT excess CAGR {out['exact_excess_cagr']:.4f} (25% cap = {0.25 * out['exact_excess_cagr']:.4f})")
    for route in ("MOO", "MOC", "worked+blocks"):
        a, b = r[route], out["running_book"][route]
        print(f"   {route:14s} gates hold to ${a['gates_hold_to'] / 1e6:8.2f}M (running-book weights ${b['gates_hold_to'] / 1e6:8.2f}M) fail: {a['failing']} | cost %NAV/yr at 2.5/5/10/25M: "
              + " / ".join(f"{a['levels'][k][1] * 100:.2f}{'' if a['levels'][k][0] else 'x'}" for k in ("2.5M", "5M", "10M", "25M")))
    p = r["participation"]
    print(f"   screen: P90 ${p['p90'] / 1e6:.1f}M P99 ${p['p99'] / 1e6:.1f}M max ${p['max'] / 1e6:.2f}M ({p['max_symbol']}) | orders/yr above 5% of a median day at $2.5M / $10M / $34.5M: "
          f"{p['orders_per_year_above_5pct_at_2.5M']:.1f} / {p['orders_per_year_above_5pct_at_10M']:.1f} / {p['orders_per_year_above_5pct_at_34.5M']:.1f}; largest order as share of a day: "
          f"{p['largest_order_share_of_day_at_2.5M']:.2f} / {p['largest_order_share_of_day_at_10M']:.2f} / {p['largest_order_share_of_day_at_34.5M']:.2f}")
    print("   per-leg P99 / max screen in book AUM ($M):", {c: (round(v["book_aum_p99_at_5pct"] / 1e6, 1), round(v["book_aum_max_at_5pct"] / 1e6, 1), v["symbol"], v["n"]) for c, v in r["per_leg_screen_in_book_aum"].items()})
json.dump(res, open(OUT / "sec_03_capacity.json", "w"), indent=1, default=str)

# BTAL wall
w_btal = {"taa3x": 0.33634, "taa3x_1n": 0.22352, "btal_qqq": 0.21684}
print("\n== BTAL 10% ownership wall (0.10 x $317M / book BTAL weight)")
for name, w in BOOKS.items():
    bw = sum(w.get(a, 0) * x for a, x in w_btal.items())
    if bw > 0:
        print(f"   {name}: BTAL weight {bw:.4f} -> ${0.10 * 317e6 / bw / 1e6:.1f}M")
# BTAL max weight inside each TAA pod, last 3 years and since 2012, from fills
btal_cs = norgatedata.price_timeseries("BTAL", stock_price_adjustment_setting=norgatedata.StockPriceAdjustmentType.CAPITALSPECIAL,
                                       padding_setting=norgatedata.PaddingType.NONE, start_date="2011-01-01", end_date="2026-08-19", timeseriesformat="pandas-dataframe")
btal_cs.index = pd.to_datetime(btal_cs.index).normalize()
for a in ("taa3x", "taa3x_1n", "btal_qqq"):
    tx, nav = D["tx"][a], D["nav"][a]
    sh = tx[tx["asset_str"] == "BTAL"].groupby("date")["amount_float"].sum().reindex(nav.index).fillna(0).cumsum()
    wb = (sh * btal_cs["Close"].reindex(nav.index) / nav)
    print(f"   {a}: BTAL weight max last 3y {wb.loc[START:END].max():.4f}, max since 2012-10 {wb.loc['2012-10-02':END].max():.4f}, mean last 3y {wb.loc[START:END].mean():.4f}")
