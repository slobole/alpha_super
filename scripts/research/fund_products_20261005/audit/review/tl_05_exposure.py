"""Timing lens 5: exposure.py - TQQQ weight basis, hand checks, drifted-weight peak, gap table arithmetic, T-bill-like share."""
import json, pickle, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import g_lib as g
import norgatedata
lib = g.lib
lab = g.Lab(); data = lab.data
END, EX = g.END, g.EXACT_START
def px(sym, adj):
    p = norgatedata.price_timeseries(sym, stock_price_adjustment_setting=adj, padding_setting=norgatedata.PaddingType.NONE, start_date="2010-01-01",
                                     end_date="2026-10-02", timeseriesformat="pandas-dataframe")
    p.index = pd.to_datetime(p.index).normalize(); return p
cs = px("TQQQ", norgatedata.StockPriceAdjustmentType.CAPITALSPECIAL)
tr = px("TQQQ", norgatedata.StockPriceAdjustmentType.TOTALRETURN)
raw = px("TQQQ", norgatedata.StockPriceAdjustmentType.NONE)
print("TQQQ close 2012-10-02: CAPITALSPECIAL", round(cs.Close.loc['2012-10-02'], 4), "TOTALRETURN", round(tr.Close.loc['2012-10-02'], 4), "NONE", round(raw.Close.loc['2012-10-02'], 4),
      "| ratio TR/CS 2012:", round(tr.Close.loc['2012-10-02'] / cs.Close.loc['2012-10-02'], 4), " 2026-08-19:", round(tr.Close.loc['2026-08-19'] / cs.Close.loc['2026-08-19'], 4))
for a in ("taa3x", "taa3x_1n"):
    tx = data["tx"][a]; t = tx[tx.asset_str == "TQQQ"].copy()
    nav = data["nav"][a]; path = lib.read_path(lib.SOURCE, a)
    t = t[(t.date >= EX) & (t.date <= END)]
    m = t.merge(cs[["Open", "Close"]].rename(columns={"Open": "cs_open", "Close": "cs_close"}), left_on="date", right_index=True).merge(
        tr[["Open"]].rename(columns={"Open": "tr_open"}), left_on="date", right_index=True).merge(raw[["Open"]].rename(columns={"Open": "raw_open"}), left_on="date", right_index=True)
    m["vs_cs_open"] = m.fill_price_float / m.cs_open - 1; m["vs_tr_open"] = m.fill_price_float / m.tr_open - 1; m["vs_raw_open"] = m.fill_price_float / m.raw_open - 1
    print(f"\n{a}: TQQQ fills {len(m)}; fill/CAPITALSPECIAL open-1: median {m.vs_cs_open.median():.5f} min {m.vs_cs_open.min():.5f} max {m.vs_cs_open.max():.5f} | vs TOTALRETURN open: median {m.vs_tr_open.median():.5f} min {m.vs_tr_open.min():.4f} max {m.vs_tr_open.max():.4f} | vs unadjusted open: median {m.vs_raw_open.median():.4f} max {m.vs_raw_open.max():.3f}")
    print(m[["date", "amount_float", "fill_price_float", "cs_open", "tr_open", "raw_open"]].iloc[[0, 1, len(m) // 2, -2, -1]].to_string())
    sh = tx[tx.asset_str == "TQQQ"].groupby("date")["amount_float"].sum().reindex(nav.index).fillna(0).cumsum()
    w = (sh * cs.Close.reindex(nav.index) / nav).loc[EX:END]
    inv = (path["portfolio_value_float"] / path["total_value_float"]).loc[EX:END]
    print("  shares min", float(sh.loc[EX:END].min()), "| weight max", round(float(w.max()), 4), "| days weight > invested weight + 0.5%:", int((w > inv + 0.005).sum()), "| max (w - inv)", round(float((w - inv).max()), 5))
    # other holdings on the same dates: full position reconstruction for a hand check on 3 dates
    allsh = tx.pivot_table(index="date", columns="asset_str", values="amount_float", aggfunc="sum").reindex(nav.index).fillna(0).cumsum()
    for dte in ("2013-06-28", "2020-02-19", "2021-11-19", "2024-07-10"):
        dte = pd.Timestamp(dte); held = allsh.loc[dte]; held = held[held.abs() > 1e-6]
        val = {s: float(held[s] * px(s, norgatedata.StockPriceAdjustmentType.CAPITALSPECIAL).Close.loc[dte]) for s in held.index}
        print(f"  {dte.date()} NAV {nav.loc[dte]:,.0f} positions(CS close) {{" + ", ".join(f"{k}: {v:,.0f}" for k, v in val.items()) + f"}} sum+cash {sum(val.values()) + path.cash_float.loc[dte]:,.0f}  port_value {path.portfolio_value_float.loc[dte]:,.0f}  TQQQ w {w.loc[dte]:.4f}")
    print("  TQQQ weight: mean", round(float(w.mean()), 4), "p90", round(float(w.quantile(.9)), 4), "max", round(float(w.max()), 4), "on", w.idxmax().date())
exp = json.loads((g.OUT / "exposure.json").read_text())
print("\nexposure.json tqqq", json.dumps(exp["tqqq_weight"]), "\nmom", exp["mom_invested"], "\nmr", exp["mr_stock_weight"], "gate_open_share", exp["gate_open_share"])
for n in ("GR1", "GR2", "GR3"):
    b = exp["books"][n]
    print(n, "nasdaq", {k: round(v, 3) for k, v in b["nasdaq_lookthrough"].items() if k in ("mean", "p90", "max")}, "equity", {k: round(v, 3) for k, v in b["equity_exposure"].items()}, "gap", b["gap_table"]["10%"], "tbill-like", {k: round(v, 3) for k, v in b["tbill_like_share"].items()})
pickle.dump({"cs_close": cs.Close}, open(g.STUDY / "audit/review/timing_lens/tqqq.pkl", "wb"))
