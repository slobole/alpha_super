"""Timing lens 1: inputs and frames (reindex, END cut, cash add, drag, ex-BIL sign, MR frames, debt pod, QQQ)."""
import pickle, sys
from pathlib import Path
import numpy as np, pandas as pd
WT = Path(__file__).resolve().parents[5]
OUT = WT / "results/research/portfolio/fund_products_20261005/audit/review/timing_lens"
SRC = WT / "results/research/portfolio/fund_products_20261005/sources"
c = pickle.load(open(OUT / "cache.pkl", "rb"))
d, F = c["data"], c["frames"]
idx = d["index"]
LONG, EXACT, END = pd.Timestamp("2008-03-04"), pd.Timestamp("2012-10-02"), pd.Timestamp("2026-08-19")
print("index", idx[0].date(), idx[-1].date(), len(idx), "LONG sessions", len(idx[(idx >= LONG) & (idx <= END)]))
dtb3 = d["dtb3"]
# independent DTB3
raw = pd.read_csv(r"C:\Users\User\Documents\workspace\alpha_super\scripts\research\fund_menu_20260923\..\..\..\data\fred\DTB3.csv", parse_dates=["observation_date"], na_values=["."]) if False else None
main = F["main"][0]
for a in ["ndx_atr_cap", "ndx_natr_cap", "dv2_g", "hpi_g", "dv2_g_cash", "hpi_g_cash"]:
    p = pd.read_csv(SRC / f"{a}__path.csv.gz", index_col="date", parse_dates=True)
    w = p.loc[LONG:END]
    sess = idx[(idx >= LONG) & (idx <= END)]
    print(f"\n== {a}: path {p.index[0].date()}..{p.index[-1].date()} n={len(p)}; LONG rows {len(w)} vs index {len(sess)}; same dates {w.index.equals(sess)}; dup {p.index.duplicated().sum()}")
    nav, cash = p["total_value_float"], p["cash_float"]
    r = nav.pct_change()
    # independent cash add: prior-close cash/NAV, lagged DTB3 (frame's dtb3), ACT/360
    days = pd.Series(p.index, index=p.index).diff().dt.days
    y = dtb3.reindex(p.index)
    add = ((cash.shift(1).clip(lower=0) * (y - 0.005).clip(lower=0) - (-cash.shift(1)).clip(lower=0) * (y + 0.015)) * days / 360 / nav.shift(1))
    if a.endswith("_cash"):
        rr, aa = d["mr_cash_run"][a], d["mr_cash_add"][a]
        print("  cash-run ret diff", float((rr.loc[LONG:END] - r.loc[LONG:END]).abs().max()), "add diff", float((aa.loc[LONG:END] - add.loc[LONG:END]).abs().max()))
        print("  cash weight LONG mean", float((cash / nav).loc[LONG:END].mean()), "invested mean", float((p['portfolio_value_float'] / nav).loc[LONG:END].mean()))
        continue
    tx = pd.read_csv(SRC / f"{a}__transactions.csv.gz", parse_dates=["date"])
    print("  tx cols", list(tx.columns))
    print("  tx dates outside path index:", int((~tx["date"].isin(p.index)).sum()), "of", len(tx), "; tx time component:", bool((tx["date"] != tx["date"].dt.normalize()).any()))
    trad = tx.assign(n=tx["signed_notional_float"].abs()).groupby("date")["n"].sum()
    drag = (trad * 0.0005).reindex(p.index).fillna(0) / nav.shift(1)
    trad_x = tx[tx["asset_str"] != "BIL"].assign(n=lambda t: t["signed_notional_float"].abs()).groupby("date")["n"].sum()
    drag_x = (trad_x * 0.0005).reindex(p.index).fillna(0) / nav.shift(1)
    house = F["s1_house_cash"][0][a]
    print("  house ret diff", float((house.loc[LONG:END] - r.loc[LONG:END]).abs().max()))
    print("  main  diff vs r+add", float((main[a].loc[LONG:END] - (r + add).loc[LONG:END]).abs().max()))
    print("  +5    diff vs r+add-drag", float((F["s3_plus_5bps"][0][a].loc[LONG:END] - (r + add - drag).loc[LONG:END]).abs().max()))
    print("  +5exB diff vs r+add-drag_x", float((F["s3b_plus_5bps_ex_bil"][0][a].loc[LONG:END] - (r + add - drag_x).loc[LONG:END]).abs().max()))
    print("  exact diff vs r+add", float((F["s6_exact"][0][a].loc[EXACT:END] - (r + add).loc[EXACT:END]).abs().max()))
    print("  unscaled diff vs r+add", float((F["s2_proxy_unscaled"][0][a].loc[LONG:END] - (r + add).loc[LONG:END]).abs().max()))
    ann = lambda s: float(s.loc[LONG:END].sum() / len(s.loc[LONG:END]) * 252)
    print(f"  annualised (LONG): cash add {ann(add):.4%}  drag all {ann(drag):.4%}  drag ex-BIL {ann(drag_x):.4%}  turnover x/yr {ann(drag)/0.0005:.1f}")
    print("  cash weight LONG mean", float((cash / nav).loc[LONG:END].mean()), "min", float((cash / nav).loc[LONG:END].min()), "invested mean", float((p['portfolio_value_float'] / nav).loc[LONG:END].mean()))
    if a in ("dv2_g", "hpi_g"):
        ca = a + "_cash"
        print("  s9 fair diff", float((F["s9_mr_fair_cash"][0][a] - (d["mr_cash_run"][ca] + d["mr_cash_add"][ca])).loc[LONG:END].abs().max()),
              " s10 zero diff", float((F["s10_mr_cash_0"][0][a] - d["mr_cash_run"][ca]).loc[LONG:END].abs().max()))
        bil = tx[tx["asset_str"] == "BIL"]
        print("  BIL fills", len(bil), "first", bil["date"].min().date(), "share of traded notional LONG", float(bil[(bil.date >= LONG) & (bil.date <= END)]["signed_notional_float"].abs().sum() / tx[(tx.date >= LONG) & (tx.date <= END)]["signed_notional_float"].abs().sum()))

# debt pod
days = pd.Series(idx, index=idx).diff().dt.days
for name, sp in (("debt", 0.015), ("debt_050", 0.005), ("debt_250", 0.025)):
    mine = (dtb3 + sp) * days / 360
    for k, (fr, s) in F.items():
        assert float((fr[name] - mine).abs().max()) < 1e-15, (k, name)
print("\ndebt pod equals (DTB3_lag + spread) x calendar days / 360 in every frame; mean annual (LONG) 1.5%:",
      float(((dtb3 + 0.015) * days / 360).loc[LONG:END].sum() / ((END - LONG).days / 365.25)))
# DTB3 lag check: raw csv
import importlib.util
dtb3_csv = None
for cand in Path(r"C:\Users\User\Documents\workspace\alpha_super").rglob("DTB3*.csv"):
    dtb3_csv = cand; break
print("DTB3 csv", dtb3_csv)
raw = pd.read_csv(dtb3_csv, parse_dates=["observation_date"], na_values=["."]).set_index("observation_date")["DTB3"].dropna() / 100
for t in ["2020-03-16", "2020-03-17", "2022-06-15", "2022-06-16", "2008-03-04", "2026-08-19"]:
    t = pd.Timestamp(t)
    prev_obs = raw.loc[:t - pd.Timedelta(days=1)].iloc[-1]
    same = raw.loc[:t].iloc[-1]
    print(f"  {t.date()} frame rate {dtb3.loc[t]:.4f} | last obs strictly before {prev_obs:.4f} | last obs <= t {same:.4f}")
# TBILL untouched across frames, QQQ passive
for k, (fr, s) in F.items():
    assert fr["tbill"].equals(main["tbill"]) or float((fr["tbill"] - main["tbill"]).abs().max()) == 0, k
    assert float((fr["qqq_tr"] - d["bench"]["QQQ"].reindex(idx)).abs().max()) == 0, k
print("tbill and qqq_tr identical in all frames; qqq LONG NaN:", int(main["qqq_tr"].loc[LONG:END].isna().sum()))
