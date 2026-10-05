"""Audit follow-up on orders_last3y.csv.gz: stock vs BIL split, today's-ADV variants, breach counts on an AUM grid,
and a Turnover-vs-Close*Volume sanity check. Rough participation screen, not a market-impact model."""
from pathlib import Path
import numpy as np
import pandas as pd
from data.norgate_loader import load_price_timeseries

OUT = Path(__file__).resolve().parents[4] / "results" / "research" / "portfolio" / "fund_products_20261005" / "audit" / "capacity"
o = pd.read_csv(OUT / "orders_last3y.csv.gz", parse_dates=["date"])
meta_windows = {"taa3x": ("2023-08-20", "2026-08-19"), "taa3x_1n": ("2023-08-20", "2026-08-19"), "core5": ("2023-08-20", "2026-08-19"),
                "btal_qqq": ("2023-08-20", "2026-08-19"), "ndx_vxn": ("2023-08-20", "2026-08-19")}
rows, grid_rows = [], []
GRID = (1e6, 5e6, 1e7, 2.5e7, 5e7, 1e8, 2.5e8)
for leg, g in o.groupby("leg"):
    lo, hi = meta_windows.get(leg, ("2023-10-03", "2026-10-02"))
    g = g[(g["date"] >= lo) & (g["date"] <= hi)]
    years = (pd.Timestamp(hi) - pd.Timestamp(lo)).days / 365.25
    for part, sub in (("all", g), ("ex_BIL", g[g["asset"] != "BIL"]), ("BIL_only", g[g["asset"] == "BIL"])):
        if sub.empty or (part != "all" and (g["asset"] == "BIL").sum() == 0):
            continue
        k = sub["k"].replace([np.inf, -np.inf], np.nan).dropna()
        kt = sub["k_today"].replace([np.inf, -np.inf], np.nan).dropna()
        f = sub["frac"].abs()
        r = {"leg": leg, "part": part, "orders": len(sub), "orders_per_year": len(sub) / years, "trade_days_per_year": sub["date"].nunique() / years,
             "turnover_x_nav": f.sum() / years, "frac_med": f.median(), "frac_p90": f.quantile(0.9), "frac_max": f.max()}
        for nm, kk in (("adv_at_order", k), ("adv_today", kt)):
            for q, lab in ((0.9, "p90"), (0.99, "p99"), (1.0, "max")):
                for x in (0.01, 0.05, 0.10):
                    r[f"aumM_{lab}_{int(x*100)}pct_{nm}"] = x / kk.quantile(q) / 1e6
        rows.append(r)
        if part == "all":
            for aum in GRID:
                p = aum * k
                above5 = sub.loc[k.index][p > 0.05]
                grid_rows.append({"leg": leg, "aum_musd": aum / 1e6, "share_orders_gt_1pct": float((p > 0.01).mean()), "share_orders_gt_5pct": float((p > 0.05).mean()),
                                  "share_orders_gt_10pct": float((p > 0.10).mean()), "orders_gt_5pct_per_year": float((p > 0.05).sum() / years),
                                  "orders_gt_20pct_per_year": float((p > 0.20).sum() / years), "max_participation": float(p.max()),
                                  "symbols_gt_5pct": "; ".join(f"{s} {n}" for s, n in above5["asset"].value_counts().head(6).items())})
t = pd.DataFrame(rows)
t.to_csv(OUT / "participation_screen_split.csv", index=False, float_format="%.6g")
gdf = pd.DataFrame(grid_rows)
gdf.to_csv(OUT / "participation_breaches_by_aum.csv", index=False, float_format="%.6g")
pd.set_option("display.width", 330); pd.set_option("display.max_columns", 60); pd.set_option("display.max_colwidth", 60)
cols = ["leg", "part", "orders", "orders_per_year", "trade_days_per_year", "turnover_x_nav", "frac_med", "frac_p90", "frac_max",
        "aumM_p90_1pct_adv_at_order", "aumM_p90_5pct_adv_at_order", "aumM_p90_10pct_adv_at_order", "aumM_p99_5pct_adv_at_order", "aumM_max_5pct_adv_at_order",
        "aumM_p90_5pct_adv_today", "aumM_p99_5pct_adv_today", "aumM_max_5pct_adv_today"]
print(t[cols].round(3).to_string())
print(gdf[gdf["aum_musd"].isin([10, 25, 50, 100])].round(4).to_string())
# Uncovered orders
print("uncovered:", o[o["adv60"].isna()][["leg", "date", "asset", "frac"]].to_string())
# Turnover vs Close*Volume sanity (unadjusted-equivalent): ratio of 60-session medians at the end of data.
for s in ("BTAL", "TQQQ", "BIL", "CCEP", "FOX", "NWS", "DBC", "UUP", "FER"):
    px = load_price_timeseries(s, start_date_str="2026-05-01", end_date_str="2026-10-02")
    cv = (px["Close"] * px["Volume"]).iloc[-60:].median()
    tv = px["Turnover"].iloc[-60:].median()
    print(s, "Turnover med60 $M", round(tv / 1e6, 2), "Close*Volume med60 $M", round(cv / 1e6, 2), "ratio", round(tv / cv, 3), "cols", [c for c in px.columns][:8])
# Capsule negative cash across the three parking variants (doc cross-check)
MAIN = Path(r"C:\Users\User\Documents\workspace\alpha_super\results\research\mr_capsule_build_20261004")
for tag in ("dv2_bil", "dv2_parked", "dv2_cash", "hpi_bil", "hpi_parked", "hpi_cash"):
    n = pd.read_csv(MAIN / f"{tag}_nav.csv", index_col=0, parse_dates=True)
    w = n["cash"] / n["total_value"]
    print(tag, "neg-cash sessions", int((n["cash"] < 0).sum()), "min cash/NAV", round(float(w.min()), 4), w.idxmin().date(), "sessions", len(n),
          "mean cash w", round(float(w.mean()), 4))
