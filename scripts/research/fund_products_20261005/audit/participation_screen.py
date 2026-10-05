"""Audit (inputs only): rough today's-liquidity participation screen per fund-product leg, last 3 years of fills.

NOT a market-impact model. For every backtest order i of a leg, filled on session t in symbol s:

    frac_i   = |signed notional_i| / NAV_(t-1)                 (order as a fraction of the pod's prior-close NAV)
    ADV60_i  = median(Turnover_s over sessions t-60 .. t-1)    (Norgate native Turnover, USD; shift(1): known before t)
    k_i      = frac_i / ADV60_i                                 (participation per dollar of pod AUM)
    part_i(A)= A * k_i                                          (share of a median day's dollar volume at pod AUM A)

AUM at which the 90th-percentile order reaches x of a median day's volume:  A_x = x / P90(k).
Orders in one symbol on one session are summed (signed) inside a pod first. Book legs (E2 50/50, MR capsule 50/50)
combine the two pods at fixed 0.5 / 0.5: `gross` adds the two pods' absolute orders (separate accounts hitting one
auction), `net` adds the signed orders (one account). Forced liquidations (order_id == -1) are not orders.

Sources (all read-only, under the main checkout):
  taa3x, taa3x_1n, core5, btal_qqq, ndx_vxn  results/research/portfolio/shelf_rebuild_20260929/sources (end 2026-08-19)
  E2 pods                                     results/research/portfolio/ndx_e2_sector_cap_5050/vanilla_backtest/2026-10-04_095907/pods
  MR capsule pods                             results/research/mr_capsule_build_20261004/{dv2,hpi}_bil_{transactions,nav}.csv

Usage: PYTHONPATH=. .venv/Scripts/python.exe scripts/research/fund_products_20261005/audit/participation_screen.py
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from data.norgate_loader import load_price_timeseries

MAIN = Path(r"C:\Users\User\Documents\workspace\alpha_super")
WT = Path(__file__).resolve().parents[4]
OUT = WT / "results" / "research" / "portfolio" / "fund_products_20261005" / "audit" / "capacity"
SHELF = MAIN / "results" / "research" / "portfolio" / "shelf_rebuild_20260929" / "sources"
E2 = MAIN / "results" / "research" / "portfolio" / "ndx_e2_sector_cap_5050" / "vanilla_backtest" / "2026-10-04_095907" / "pods"
CAPS = MAIN / "results" / "research" / "mr_capsule_build_20261004"
ADV_WINDOW, ADV_MIN = 60, 20
LEVELS = (0.01, 0.05, 0.10)
COMMON_LO, COMMON_HI = pd.Timestamp("2023-08-21"), pd.Timestamp("2026-08-19")


def shelf_leg(alias: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    tx = pd.read_csv(SHELF / f"{alias}__transactions.csv.gz", parse_dates=["date"])
    tx = tx.rename(columns={"asset_str": "asset", "amount_float": "amount", "signed_notional_float": "notional"})
    tx["forced"] = False                      # the shelf export carries no order id; rows are fills
    path = pd.read_csv(SHELF / f"{alias}__path.csv.gz", index_col="date", parse_dates=True)
    path = path.rename(columns={"total_value_float": "nav", "cash_float": "cash"})
    return tx[["date", "asset", "amount", "notional", "forced"]], path[["nav", "cash"]]


def engine_tx(path: Path) -> pd.DataFrame:
    tx = pd.read_csv(path, parse_dates=["bar"]).rename(columns={"bar": "date"})
    tx["notional"] = tx["amount"].astype(float) * tx["price"].astype(float)
    tx["forced"] = tx["order_id"] == -1
    return tx[["date", "asset", "amount", "notional", "forced"]]


def e2_leg(pod: str, module: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    tx = engine_tx(E2 / pod / "transactions.csv")
    with open(E2 / pod / f"{module}.pkl", "rb") as handle:
        res = pickle.load(handle).results
    path = pd.DataFrame({"nav": res["total_value"].astype(float), "cash": res["cash"].astype(float)})
    path.index = pd.to_datetime(path.index)
    return tx, path


def capsule_leg(tag: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    tx = engine_tx(CAPS / f"{tag}_bil_transactions.csv")
    nav = pd.read_csv(CAPS / f"{tag}_bil_nav.csv", index_col=0, parse_dates=True)
    return tx, nav.rename(columns={"total_value": "nav"})[["nav", "cash"]]


def last3y(path: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp]:
    hi = path.index[-1]
    return hi - pd.DateOffset(years=3) + pd.Timedelta(days=1), hi


def orders_of(tx: pd.DataFrame, path: pd.DataFrame, leg: str) -> pd.DataFrame:
    """One row per (session, symbol): signed order as a fraction of the prior close's NAV."""
    live = tx[~tx["forced"]]
    day = live.groupby(["date", "asset"], as_index=False)["notional"].sum()
    prior = path["nav"].shift(1)
    day["frac"] = day["notional"].to_numpy() / prior.reindex(day["date"]).to_numpy()
    day["leg"] = leg
    return day.dropna(subset=["frac"])[["leg", "date", "asset", "frac"]]


def positions_count(tx: pd.DataFrame, path: pd.DataFrame, lo: pd.Timestamp, hi: pd.Timestamp) -> tuple[float, float, float]:
    """Mean / max open positions per session (all, and without BIL), from the cumulated fills."""
    wide = tx.pivot_table(index="date", columns="asset", values="amount", aggfunc="sum").fillna(0.0).cumsum()
    wide = wide.reindex(path.index).ffill().fillna(0.0).loc[lo:hi]
    # Residual dust from fractional historical-share units is not a position.
    open_ = wide.abs() > 1e-6
    n_all = open_.sum(axis=1)
    n_ex = open_.drop(columns=[c for c in open_.columns if c == "BIL"]).sum(axis=1)
    return float(n_all.mean()), float(n_all.max()), float(n_ex.mean())


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    legs: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {a: shelf_leg(a) for a in ("taa3x", "taa3x_1n", "core5", "btal_qqq", "ndx_vxn")}
    legs["e2_atr_cap"] = e2_leg("pod_ndx_atr_vxn_sector_cap", "strategy_mo_atr_normalized_ndx_vxn_scaled_sector_cap")
    legs["e2_natr20_cap"] = e2_leg("pod_ndx_natr20_vxn_sector_cap", "strategy_mo_natr20_ndx_vxn_scaled_sector_cap")
    legs["dv2_g_bil"] = capsule_leg("dv2")
    legs["hpi_g_bil"] = capsule_leg("hpi")

    windows = {leg: last3y(path) for leg, (_, path) in legs.items()}
    all_orders = pd.concat([orders_of(tx, path, leg) for leg, (tx, path) in legs.items()], ignore_index=True)

    # Book legs at fixed 50/50.
    def book(name: str, a: str, b: str) -> None:
        nonlocal all_orders
        sub = all_orders[all_orders["leg"].isin([a, b])].copy()
        sub["half"] = 0.5 * sub["frac"]
        net = sub.groupby(["date", "asset"], as_index=False)["half"].sum().rename(columns={"half": "frac"})
        gross = sub.assign(half=sub["half"].abs()).groupby(["date", "asset"], as_index=False)["half"].sum().rename(columns={"half": "frac"})
        all_orders = pd.concat([all_orders, net.assign(leg=f"{name}_net"), gross.assign(leg=f"{name}_gross")], ignore_index=True)
        lo = max(windows[a][0], windows[b][0])
        windows[f"{name}_net"] = windows[f"{name}_gross"] = (lo, min(windows[a][1], windows[b][1]))

    book("e2_book_5050", "e2_atr_cap", "e2_natr20_cap")
    book("mr_capsule_bil_5050", "dv2_g_bil", "hpi_g_bil")

    lo_all = min(lo for lo, _ in windows.values())
    hi_all = max(hi for _, hi in windows.values())
    keep = pd.Series(False, index=all_orders.index)
    for leg, (lo, hi) in windows.items():
        keep |= (all_orders["leg"] == leg) & (all_orders["date"] >= min(lo, COMMON_LO)) & (all_orders["date"] <= hi)
    all_orders = all_orders[keep & (all_orders["frac"].abs() > 0)].reset_index(drop=True)

    # Liquidity: native Turnover, median of the 60 sessions before the order; "today" = last 60 sessions of the data.
    symbols = sorted(all_orders["asset"].unique()) + ["BIL", "SPMO"]
    adv, adv_today, src, missing = {}, {}, {}, []
    start = (lo_all - pd.Timedelta(days=150)).strftime("%Y-%m-%d")
    for i, sym in enumerate(sorted(set(symbols))):
        try:
            px = load_price_timeseries(sym, start_date_str=start, end_date_str=hi_all.strftime("%Y-%m-%d"))
        except Exception as exc:  # noqa: BLE001 - reported below, the order is then uncovered
            missing.append(f"{sym}: {type(exc).__name__}")
            continue
        px.index = pd.to_datetime(px.index).normalize()
        if "Turnover" in px.columns and float(px["Turnover"].fillna(0).sum()) > 0:
            dollar, src[sym] = px["Turnover"].astype(float), "Turnover"
        else:
            col = "Unadjusted Close" if "Unadjusted Close" in px.columns else "Close"
            dollar, src[sym] = px[col].astype(float) * px["Volume"].astype(float), f"{col} x Volume"
        # *** CRITICAL*** shift(1): only volume known before the order's session.
        adv[sym] = dollar.rolling(ADV_WINDOW, min_periods=ADV_MIN).median().shift(1)
        adv_today[sym] = float(dollar.dropna().iloc[-ADV_WINDOW:].median()) if len(dollar.dropna()) else np.nan
        if (i + 1) % 100 == 0:
            print("loaded", i + 1, "symbols", flush=True)
    all_orders["adv60"] = [adv[s].get(d, np.nan) if s in adv else np.nan for d, s in zip(all_orders["date"], all_orders["asset"])]
    all_orders["adv60_today"] = all_orders["asset"].map(adv_today)
    all_orders["k"] = all_orders["frac"].abs() / all_orders["adv60"]
    all_orders["k_today"] = all_orders["frac"].abs() / all_orders["adv60_today"]
    all_orders.to_csv(OUT / "orders_last3y.csv.gz", index=False, float_format="%.8g")

    rows, bind_rows = [], []
    for leg, (lo, hi) in windows.items():
        for wname, (a, b) in (("last3y_own_end", (lo, hi)), ("common_2023-08-21_2026-08-19", (COMMON_LO, COMMON_HI))):
            o = all_orders[(all_orders["leg"] == leg) & (all_orders["date"] >= a) & (all_orders["date"] <= b)]
            if o.empty:
                continue
            years = (b - a).days / 365.25
            f = o["frac"].abs()
            cov = o.dropna(subset=["k"])
            cov = cov[np.isfinite(cov["k"]) & (cov["adv60"] > 0)]
            row = {"leg": leg, "window": wname, "start": a.date(), "end": b.date(), "orders": len(o),
                   "orders_per_year": len(o) / years, "trade_days_per_year": o["date"].nunique() / years,
                   "turnover_x_nav_per_year": float(f.sum() / years),
                   "order_frac_median": float(f.median()), "order_frac_p90": float(f.quantile(0.90)), "order_frac_max": float(f.max()),
                   "orders_uncovered": int(len(o) - len(cov)), "k_p50": float(cov["k"].median()), "k_p90": float(cov["k"].quantile(0.90)),
                   "k_p99": float(cov["k"].quantile(0.99)), "k_max": float(cov["k"].max())}
            for x in LEVELS:
                row[f"aum_p90_at_{int(x * 100)}pct"] = x / row["k_p90"]
                row[f"aum_p99_at_{int(x * 100)}pct"] = x / row["k_p99"]
                row[f"aum_max_at_{int(x * 100)}pct"] = x / row["k_max"]
                row[f"aum_p90_at_{int(x * 100)}pct_today_adv"] = x / float(o["k_today"].replace([np.inf], np.nan).dropna().quantile(0.90))
            worst = cov.loc[cov["k"].idxmax()]
            row.update({"worst_symbol": worst["asset"], "worst_date": worst["date"].date(), "worst_frac": float(abs(worst["frac"])),
                        "worst_adv60_usd": float(worst["adv60"])})
            top = cov[cov["k"] >= row["k_p90"]]
            share = top["asset"].value_counts(normalize=True)
            row["binding_symbols_top_decile"] = "; ".join(f"{s} {v:.0%}" for s, v in share.head(6).items())
            rows.append(row)
            if wname == "last3y_own_end":
                g = top.groupby("asset").agg(orders_in_top_decile=("k", "size"), median_frac=("frac", lambda s: float(s.abs().median())),
                                             max_frac=("frac", lambda s: float(s.abs().max())), median_adv60_usd=("adv60", "median"),
                                             min_adv60_usd=("adv60", "min"), max_k=("k", "max")).reset_index()
                g["adv60_today_usd"] = g["asset"].map(adv_today)
                g["aum_at_5pct_worst_order"] = 0.05 / g["max_k"]
                g.insert(0, "leg", leg)
                bind_rows.append(g.sort_values("max_k", ascending=False).head(12))
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "participation_screen_by_leg.csv", index=False, float_format="%.8g")
    pd.concat(bind_rows, ignore_index=True).to_csv(OUT / "binding_symbols_by_leg.csv", index=False, float_format="%.8g")

    # Ease facts from the same runs: positions held, negative cash (margin need), over the leg's own last 3 years and full history.
    ease = []
    for leg, (tx, path) in legs.items():
        lo, hi = windows[leg]
        w = path["cash"] / path["nav"]
        n_mean, n_max, n_ex = positions_count(tx, path, lo, hi)
        w3 = w.loc[lo:hi]
        ease.append({"leg": leg, "run_start": path.index[0].date(), "run_end": path.index[-1].date(),
                     "positions_mean_3y": n_mean, "positions_max_3y": n_max, "positions_mean_3y_ex_bil": n_ex,
                     "cash_weight_mean_3y": float(w3.mean()), "cash_weight_min_3y": float(w3.min()),
                     "neg_cash_day_share_3y": float((w3 < -1e-9).mean()), "cash_weight_min_full": float(w.min()),
                     "cash_weight_min_full_date": w.idxmin().date(), "neg_cash_day_share_full": float((w < -1e-9).mean()),
                     "neg_cash_below_minus5pct_days_full": int((w < -0.05).sum()), "forced_liquidations_3y": int(tx[(tx["forced"]) & (tx["date"] >= lo)].shape[0])})
    pd.DataFrame(ease).to_csv(OUT / "ease_facts_by_leg.csv", index=False, float_format="%.6g")

    parking = {s: {"adv60_today_usd": adv_today.get(s), "source": src.get(s)} for s in ("BIL", "SPMO", "BTAL", "DBC", "UUP", "TQQQ", "QQQ", "GLD", "TLT", "IEF", "SPY")}
    for s, v in parking.items():
        if v["adv60_today_usd"]:
            v.update({f"position_usd_at_{int(x * 100)}pct_of_median_day": x * v["adv60_today_usd"] for x in LEVELS})
    meta = {"adv_window_sessions": ADV_WINDOW, "adv_min_sessions": ADV_MIN, "levels": LEVELS, "missing_symbols": missing,
            "liquidity_source_counts": pd.Series(src).value_counts().to_dict(), "today_adv_end": str(hi_all.date()),
            "windows": {k: [str(a.date()), str(b.date())] for k, (a, b) in windows.items()}, "reference_etfs_today": parking,
            "label": "rough participation screen, not a market-impact model"}
    (OUT / "participation_screen_meta.json").write_text(json.dumps(meta, indent=1, default=str), encoding="utf-8")

    pd.set_option("display.width", 320)
    pd.set_option("display.max_columns", 60)
    pd.set_option("display.max_colwidth", 70)
    main_rows = table[table["window"] == "last3y_own_end"]
    print(main_rows[["leg", "start", "end", "orders", "orders_per_year", "trade_days_per_year", "turnover_x_nav_per_year", "order_frac_median",
                     "order_frac_p90", "order_frac_max", "orders_uncovered"]].round(4).to_string())
    show = main_rows[["leg", "k_p90", "aum_p90_at_1pct", "aum_p90_at_5pct", "aum_p90_at_10pct", "aum_p99_at_5pct", "aum_max_at_5pct",
                      "aum_p90_at_5pct_today_adv", "worst_symbol", "worst_date", "worst_adv60_usd", "binding_symbols_top_decile"]].copy()
    for c in [c for c in show.columns if c.startswith("aum_")]:
        show[c] = (show[c] / 1e6).round(2)
    print(show.to_string())
    print(pd.DataFrame(ease).round(4).to_string())
    print(json.dumps(meta, indent=1, default=str)[:3000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
