"""Timing lens 7: capacity.py - participation screen dilution, liquidity lag, pod-weight path, BIL as ETF, cost cap, min size."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import g_lib as g
from g_lib import END, TBILL, ga, lib
lab = g.Lab(); data = lab.data
sys.path.insert(1, str(ga.MAIN_REPO / "scripts" / "research" / "growth_shelf_v2_20260926"))
import shelf_books as sb
print("shelf_books from", sb.__file__, "| load_price_timeseries from", sb.load_price_timeseries.__module__, sys.modules[sb.load_price_timeseries.__module__].__file__)
import inspect
src = inspect.getsource(sb.load_price_timeseries); print(src[:900])
ETF_PODS = {"core5", "btal_qqq", "taa3x", "taa3x_1n"}; NASDAQ_PODS = {"ndx_vxn", "ndx_atr_cap", "ndx_natr_cap"}; URGENT = {"dv2", "hpi_vote", "dv2_g", "hpi_g"}
books = {"GR1": g.PRODUCTS["GR1"], "GR2": g.PRODUCTS["GR2"], "GR3": g.PRODUCTS["GR3"], "S9": g.INCUMBENT, "GR1-L": g.SLOT_TESTS[g.GR1_L],
         "leg TAA 3x": {"taa3x": 1.0}, "leg TAA 3x 1N": {"taa3x_1n": 1.0}, "leg MOM": g.MOM, "leg MR": g.MR}
start = sb.CAPACITY_START
alias_all = sorted({a for w in books.values() for a in w if a != TBILL})
frames = []
for a in alias_all:
    t = data["tx"][a]; t = t[(t.date >= start) & (t.date <= END)]
    pn = data["nav"][a].shift(1)
    frames.append(t.assign(alias_str=a, fraction_float=t.signed_notional_float.abs().to_numpy() / pn.reindex(t.date).to_numpy())[["date", "alias_str", "asset_str", "fraction_float"]])
orders = pd.concat(frames, ignore_index=True)
adv60 = {}
for tk in sorted(orders.asset_str.unique()):
    try:
        px = sb.load_price_timeseries(tk, start_date_str="2023-03-01", end_date_str=END.strftime("%Y-%m-%d"))
    except Exception as e:
        continue
    px.index = pd.to_datetime(px.index).normalize()
    adv60[tk] = (px.Close * px.Volume).replace(0.0, np.nan).rolling(60, min_periods=20).median().shift(1)
adv60 = pd.DataFrame(adv60)
cap = json.loads((g.OUT / "capacity.json").read_text())["books"]
res = {}
for name, w in books.items():
    wp = {a: v for a, v in w.items() if a != TBILL}
    cols = list(w) if abs(sum(wp.values()) - 1) > 1e-9 else list(wp)
    pw = lib.common.book_return_ser(data["sleeve"].loc[start - pd.Timedelta(days=10):END, cols], {c: w[c] for c in cols}, "annual")[1]
    bo = orders[orders.alias_str.isin(wp)].copy()
    warr = pw.reindex(bo.date).to_numpy(); col = {a: i for i, a in enumerate(pw.columns)}
    bo["bf"] = bo.fraction_float.to_numpy() * np.array([warr[i, col[a]] for i, a in enumerate(bo.alias_str)])
    bo["urgent"] = bo.alias_str.isin(URGENT)
    daily = bo.groupby(["date", "asset_str", "urgent"], as_index=False)["bf"].sum()
    daily["adv"] = [adv60.at[d, t] if (t in adv60.columns and d in adv60.index) else np.nan for d, t in zip(daily.date, daily.asset_str)]
    cov = daily.dropna(subset=["adv"]).copy(); cov["k"] = cov.bf / cov.adv
    p99 = 0.05 / np.percentile(cov.k, 99); p90 = 0.05 / np.percentile(cov.k, 90); mx = 0.05 / cov.k.max()
    key = {"S9": "S9 incumbent launch", "GR1-L": g.GR1_L}.get(name, name)
    ref = cap[key]["participation"]
    n_btal = int((cov.asset_str == "BTAL").sum()); thin = cov.asset_str.isin(["BTAL", "UUP", "DBC"])
    over = cov[cov.k * p99 > 0.05]
    part_btal = (cov[cov.asset_str == "BTAL"].k * p99)
    res[name] = dict(p99=p99, mx=mx)
    print(f"\n{name}: orders {len(cov)} (BTAL {n_btal}, thin-ETF {int(thin.sum())}, BIL {(cov.asset_str == 'BIL').sum()}); P90 ${p90 / 1e6:.1f}M P99 ${p99 / 1e6:.1f}M max ${mx / 1e6:.2f}M | capacity.json P99 ${ref['aum_p99_at_5pct'] / 1e6:.1f}M max ${ref['aum_max_at_5pct'] / 1e6:.2f}M")
    print(f"   at the P99 AUM: orders above 5% of a median day: {len(over)} ({over.asset_str.value_counts().head(5).to_dict()}); BTAL orders: median participation {part_btal.median():.1%}, max {part_btal.max():.1%}; share of BTAL orders above 5%: {(part_btal > 0.05).mean():.0%}")
    # screen on the binding symbol only, and per-symbol P99
    for sym in ("BTAL",):
        ks = cov[cov.asset_str == sym].k
        if len(ks):
            print(f"   {sym}-only screen: AUM at which its P90 order = 5%: ${0.05 / np.percentile(ks, 90) / 1e6:.2f}M; median order = 5%: ${0.05 / ks.median() / 1e6:.2f}M")
print("\nper-leg P99 scaled to the product (leg P99 / capital weight):")
for n, w in (("GR1", {"leg TAA 3x": 1 / 3, "leg MOM": 1 / 3, "leg MR": 1 / 3}), ("GR2", {"leg TAA 3x 1N": 1 / 3, "leg MOM": 1 / 3, "leg MR": 1 / 3}), ("GR3", {"leg TAA 3x 1N": 1 / 2, "leg MOM": 1 / 4, "leg MR": 1 / 4})):
    print(" ", n, {l: f"${res[l]['p99'] / s / 1e6:.1f}M" for l, s in w.items()}, "-> binding leg min:", f"${min(res[l]['p99'] / s for l, s in w.items()) / 1e6:.1f}M", "| book-level P99 reported:", f"${res[n]['p99'] / 1e6:.1f}M")
# pod-weight path: initialised at target 10 days before the window vs the true annual-reset drift from January 2023
for name in ("GR1", "GR3"):
    w = books[name]
    pw0 = lib.common.book_return_ser(data["sleeve"].loc[start - pd.Timedelta(days=10):END, list(w)], w, "annual")[1]
    pw1 = lib.common.book_return_ser(data["sleeve"].loc["2023-01-03":END, list(w)], w, "annual")[1]
    d = (pw0 - pw1.reindex(pw0.index)).loc[:"2023-12-31"].abs().max()
    print(name, "max abs pod-weight difference in 2023 (window-start init vs January reset):", d.round(4).to_dict())
