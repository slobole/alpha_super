"""Timing lens 11: does classing the MR pods' BIL orders as ETF orders dilute the pooled etf_one_day P95 gate (MOC route)?"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import g_lib as g
from g_lib import END, TBILL, ga, lib
lab = g.Lab(); data = lab.data
sys.path.insert(1, str(ga.MAIN_REPO / "scripts" / "research" / "growth_shelf_v2_20260926"))
import shelf_books as sb
ETF_PODS = {"core5", "btal_qqq", "taa3x", "taa3x_1n"}; NASDAQ_PODS = {"ndx_vxn", "ndx_atr_cap", "ndx_natr_cap"}; URGENT = {"dv2", "hpi_vote", "dv2_g", "hpi_g"}
books = {"GR1": g.PRODUCTS["GR1"], "GR2": g.PRODUCTS["GR2"], "GR3": g.PRODUCTS["GR3"], "GR1-L": g.SLOT_TESTS[g.GR1_L], "S9": g.INCUMBENT}
start = sb.CAPACITY_START; years = (END - start).days / 365.25
alias_all = sorted({a for w in books.values() for a in w if a != TBILL})
frames = []
for a in alias_all:
    t = data["tx"][a]; t = t[(t.date >= start) & (t.date <= END)]; pn = data["nav"][a].shift(1)
    frames.append(t.assign(alias_str=a, fraction_float=t.signed_notional_float.abs().to_numpy() / pn.reindex(t.date).to_numpy())[["date", "alias_str", "asset_str", "fraction_float"]])
orders = pd.concat(frames, ignore_index=True)
etf_tickers = set(orders.loc[orders.alias_str.isin(ETF_PODS), "asset_str"]) | {"BIL"}
nasdaq_tickers = set(orders.loc[orders.alias_str.isin(NASDAQ_PODS), "asset_str"])
liq = {"adv20": {}, "adv60": {}, "sigma": {}}
for tk in sorted(orders.asset_str.unique()):
    try: px = sb.load_price_timeseries(tk, start_date_str="2023-03-01", end_date_str=END.strftime("%Y-%m-%d"))
    except Exception: continue
    px.index = pd.to_datetime(px.index).normalize(); dollar = (px.Close * px.Volume).replace(0.0, np.nan)
    liq["adv20"][tk] = dollar.rolling(20, min_periods=10).median().shift(1); liq["adv60"][tk] = dollar.rolling(60, min_periods=20).median().shift(1)
    liq["sigma"][tk] = px.Close.pct_change(fill_method=None).rolling(60, min_periods=20).std().shift(1)
liq = {k: pd.DataFrame(v) for k, v in liq.items()}
def covered(w):
    wp = {a: v for a, v in w.items() if a != TBILL}
    cols = list(w) if abs(sum(wp.values()) - 1) > 1e-9 else list(wp)
    pw = lib.common.book_return_ser(data["sleeve"].loc[start - pd.Timedelta(days=10):END, cols], {c: w[c] for c in cols}, "annual")[1]
    bo = orders[orders.alias_str.isin(wp)].copy(); warr = pw.reindex(bo.date).to_numpy(); col = {a: i for i, a in enumerate(pw.columns)}
    bo["book_fraction_float"] = bo.fraction_float.to_numpy() * np.array([warr[i, col[a]] for i, a in enumerate(bo.alias_str)])
    bo["is_urgent"] = bo.alias_str.isin(URGENT)
    d = bo.groupby(["date", "asset_str", "is_urgent"], as_index=False)["book_fraction_float"].sum()
    d["is_etf"] = d.asset_str.isin(etf_tickers); d["is_nasdaq"] = d.asset_str.isin(nasdaq_tickers)
    for c, fr in liq.items():
        d[c] = [fr.at[x, t] if (t in fr.columns and x in fr.index) else np.nan for x, t in zip(d.date, d.asset_str)]
    return d.dropna(subset=list(liq))
for name, w in books.items():
    cov = covered(w)
    print(f"\n{name}: ETF-classed orders {int(cov.is_etf.sum())} of which BIL {(cov.asset_str == 'BIL').sum()}, BTAL {(cov.asset_str == 'BTAL').sum()}")
    for route in ("MOC", "MOO"):
        for aum in (5e6, 1e7, 2.5e7):
            _, gates = sb.route_cost_and_gates_v2(cov, route, aum)
            _, gates_nb = sb.route_cost_and_gates_v2(cov[cov.asset_str != "BIL"], route, aum)
            f = lambda gt: {k: (round(v, 4) if isinstance(v, float) else v) for k, v in gt.items() if k.startswith("etf_one_day") or k == "stock_auction_ok"}
            print(f"   {route} ${aum / 1e6:g}M  as published (BIL pooled with the ETF orders): {f(gates)}")
            if (cov.asset_str == "BIL").any():
                print(f"   {route} ${aum / 1e6:g}M  BIL orders left out of the pooled gate:        {f(gates_nb)}")
