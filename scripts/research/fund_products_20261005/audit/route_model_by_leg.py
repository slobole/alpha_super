"""Audit (inputs only): the earlier report's house route model applied to each leg ALONE, at pod level.

This is a replica of the gate and cost arithmetic in
  scripts/research/growth_shelf_20260924/growth_capacity.py  (route_cost_and_gates)
  scripts/research/growth_shelf_v2_20260926/shelf_books.py   (route_cost_and_gates_v2)
as called by scripts/research/growth_aggressive_20260930/report_data.py (capacity). It is written out here instead of
imported so nothing under the main checkout is executed. Same liquidity inputs as the original:
  dollar = Close x Volume (zeros -> NaN); adv20 = rolling(20, min 10).median().shift(1);
  adv60 = rolling(60, min 20).median().shift(1); sigma = Close.pct_change().rolling(60, min 20).std().shift(1).

Routes (per order of size p20 = $ / adv20, p60 = $ / adv60):
  MOO   stocks: opening auction, gate P95(p20) <= 0.05% and P99(p20) <= 0.10%, cost = $ x lambda x sqrt(p20 / 1%)
               (lambda 66.4 bps for Nasdaq-100 pods, 40 bps otherwise); ETFs: one day, gate P95(p60) <= 5% and
               max(p60) <= 20%, cost = $ x sigma x sqrt(p60).
  MOC   as MOO with stock limits 0.25% / 0.50% and lambda 8.2 bps.
  worked+blocks  urgent stock orders (daily MR pods) stay in the close auction (MOC limits); urgent ETF orders are worked
               within one day (5% / 20%); every other order is worked over up to 5 days at <= 10% of volume a day
               (gate on the daily participation: P95 <= 5%, max <= 20%); BTAL / UUP / DBC are blocks at 15 bps, no gate.
Not reproduced: the cost gate (cost <= 25% of the book's excess return over T-bills), because it needs a book. The
yearly cost is printed at fixed pod sizes instead.

Classification used here (the original takes it from alias sets): ETF pods = taa3x, taa3x_1n, core5, btal_qqq;
Nasdaq pods = ndx_vxn and the two E2 pods; urgent pods = DV2-G, HPI-G. BIL orders of the capsule pods are shown two
ways: `bil_etf` (BIL is an ETF, urgent: one-day 5% / 20% gate) and `bil_stock` (what the original code does when no
ETF pod of the book trades BIL: BIL falls under the single-stock close-auction limits).

Usage: PYTHONPATH=. .venv/Scripts/python.exe scripts/research/fund_products_20261005/audit/route_model_by_leg.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from data.norgate_loader import load_price_timeseries

OUT = Path(__file__).resolve().parents[4] / "results" / "research" / "portfolio" / "fund_products_20261005" / "audit" / "capacity"
THIN = {"BTAL", "UUP", "DBC"}
MOO = {"soft": 0.0005, "hard": 0.0010, "lam_nasdaq": 66.4e-4, "lam_other": 40.0e-4}
MOC = {"soft": 0.0025, "hard": 0.0050, "lam_nasdaq": 8.2e-4, "lam_other": 8.2e-4}
BLOCK = 0.0015
GRID = (5e5, 1e6, 2.5e6, 5e6, 1e7, 2.5e7, 5e7, 1e8, 2.5e8)          # the original AUM grid
FINE = np.round(10 ** np.arange(5.0, 9.51, 0.05), 0)                 # 100K .. ~3.2B, 20 steps per decade
COST_AT = (1e7, 2.5e7, 5e7, 1e8)
ETF_PODS = {"taa3x", "taa3x_1n", "core5", "btal_qqq"}
NASDAQ_PODS = {"ndx_vxn", "e2_atr_cap", "e2_natr20_cap", "e2_book_5050_net"}
URGENT_PODS = {"dv2_g_bil", "hpi_g_bil", "mr_capsule_bil_5050_gross"}
WINDOW = {"taa3x": ("2023-08-21", "2026-08-19"), "taa3x_1n": ("2023-08-21", "2026-08-19"), "core5": ("2023-08-21", "2026-08-19"),
          "btal_qqq": ("2023-08-21", "2026-08-19"), "ndx_vxn": ("2023-08-21", "2026-08-19")}
DEFAULT_WINDOW = ("2023-10-03", "2026-10-02")
LEGS = ["taa3x", "taa3x_1n", "core5", "btal_qqq", "ndx_vxn", "e2_atr_cap", "e2_natr20_cap", "e2_book_5050_net", "dv2_g_bil", "hpi_g_bil",
        "mr_capsule_bil_5050_gross"]


def route(o: pd.DataFrame, route_str: str, aum: float) -> tuple[float, dict]:
    dollars = o["frac"].abs().to_numpy() * aum
    is_etf, urgent, thin, nasdaq = (o[c].to_numpy() for c in ("is_etf", "is_urgent", "is_thin", "is_nasdaq"))
    p20, p60, sigma = dollars / o["adv20"].to_numpy(), dollars / o["adv60"].to_numpy(), o["sigma"].to_numpy()
    cost, gates = np.zeros_like(dollars), {}
    names = o["asset"].to_numpy()

    def auction(mask, par, label):
        if not mask.any():
            return
        lam = np.where(nasdaq[mask], par["lam_nasdaq"], par["lam_other"])
        cost[mask] = dollars[mask] * lam * np.sqrt(p20[mask] / 0.01)
        a, b = float(np.percentile(p20[mask], 95)), float(np.percentile(p20[mask], 99))
        gates[label] = (a <= par["soft"] and b <= par["hard"], names[mask][int(np.argmax(p20[mask]))])

    def worked(mask, max_day, label):
        if not mask.any():
            return
        day = np.clip(np.ceil(p60[mask] / 0.10), 1.0, max_day)
        daily = p60[mask] / day
        cost[mask] = dollars[mask] * sigma[mask] * np.sqrt(daily)
        gates[label] = (float(np.percentile(daily, 95)) <= 0.05 and float(daily.max()) <= 0.20, names[mask][int(np.argmax(daily))])

    if route_str in ("MOO", "MOC"):
        auction(~is_etf, MOO if route_str == "MOO" else MOC, "stock_auction")
        worked(is_etf, 1.0, "etf_one_day")
    else:
        urgent_etf = is_etf & urgent
        auction(~is_etf & urgent, MOC, "urgent_stock_moc")
        worked(~urgent & ~thin, 5.0, "worked")
        cost[thin & ~urgent] = dollars[thin & ~urgent] * BLOCK
        if urgent_etf.any():
            part = p60[urgent_etf]
            cost[urgent_etf] = dollars[urgent_etf] * sigma[urgent_etf] * np.sqrt(part)
            gates["urgent_etf_one_day"] = (float(np.percentile(part, 95)) <= 0.05 and float(part.max()) <= 0.20,
                                           names[urgent_etf][int(np.argmax(part))])
    return float(cost.sum()), gates


def main() -> int:
    orders = pd.read_csv(OUT / "orders_last3y.csv.gz", parse_dates=["date"])
    orders = orders[orders["leg"].isin(LEGS)]
    liq = {"adv20": {}, "adv60": {}, "sigma": {}}
    for i, sym in enumerate(sorted(orders["asset"].unique())):
        px = load_price_timeseries(sym, start_date_str="2023-03-01", end_date_str="2026-10-02")
        px.index = pd.to_datetime(px.index).normalize()
        dollar = (px["Close"] * px["Volume"]).replace(0.0, np.nan)
        # *** CRITICAL*** shift(1): only liquidity known before the trade.
        liq["adv20"][sym] = dollar.rolling(20, min_periods=10).median().shift(1)
        liq["adv60"][sym] = dollar.rolling(60, min_periods=20).median().shift(1)
        liq["sigma"][sym] = px["Close"].pct_change(fill_method=None).rolling(60, min_periods=20).std().shift(1)
        if (i + 1) % 100 == 0:
            print("loaded", i + 1, flush=True)
    for col, by in liq.items():
        orders[col] = [by[s].get(d, np.nan) for d, s in zip(orders["date"], orders["asset"])]
    rows = []
    for leg in LEGS:
        lo, hi = WINDOW.get(leg, DEFAULT_WINDOW)
        base = orders[(orders["leg"] == leg) & (orders["date"] >= lo) & (orders["date"] <= hi)].copy()
        years = (pd.Timestamp(hi) - pd.Timestamp(lo)).days / 365.25
        variants = [("", None)] if leg not in URGENT_PODS else [("bil_etf", True), ("bil_stock", False), ("stocks_only", None)]
        for tag, bil_is_etf in variants:
            o = base.copy()
            if tag == "stocks_only":
                o = o[o["asset"] != "BIL"]
            o["is_etf"] = leg in ETF_PODS
            if bil_is_etf:
                o.loc[o["asset"] == "BIL", "is_etf"] = True
            o["is_urgent"], o["is_nasdaq"], o["is_thin"] = leg in URGENT_PODS, leg in NASDAQ_PODS, o["asset"].isin(THIN)
            cov = o.dropna(subset=["adv20", "adv60", "sigma"])
            for r in ("MOO", "MOC", "worked+blocks"):
                def scan(grid):
                    rec, fail = None, ""
                    for aum in grid:
                        _, g = route(cov, r, aum)
                        ok = all(v[0] for v in g.values())
                        if ok and not fail:
                            rec = aum
                        elif not fail:
                            fail = f"${aum / 1e6:g}M: " + "; ".join(f"{k} ({v[1]})" for k, v in g.items() if not v[0])
                    return rec, fail
                rec, fail = scan(GRID)
                rec_fine, fail_fine = scan(FINE)
                row = {"leg": leg, "variant": tag, "route": r, "orders": len(o), "uncovered": len(o) - len(cov),
                       "gates_ok_up_to_grid": rec, "first_fail_grid": fail, "gates_ok_up_to_fine": rec_fine, "first_fail_fine": fail_fine}
                for aum in COST_AT:
                    row[f"cost_pct_yr_at_{aum / 1e6:g}M"] = 100.0 * route(cov, r, aum)[0] / aum / years
                rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "route_model_by_leg.csv", index=False, float_format="%.6g")
    pd.set_option("display.width", 330)
    pd.set_option("display.max_columns", 40)
    pd.set_option("display.max_colwidth", 70)
    show = table.copy()
    for c in ("gates_ok_up_to_grid", "gates_ok_up_to_fine"):
        show[c] = (show[c] / 1e6).round(2)
    print(show.round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
