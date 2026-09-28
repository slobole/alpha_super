"""Review (tradability lens) R2: re-price the engine's commission at a constant owner pod size in NOMINAL shares.

For every engine fill i on bar t:
  frac_i   = |amount_i * price_i| / NAV_(t-1)             (engine's adjusted units; notional is unit-free)
  k_t      = UnadjustedClose_t / AdjustedClose_t            (nominal price scale on the fill bar)
  P_nom    = price_i * k_t                                  (nominal fill price incl. 2.5 bp slippage)
  sh_i(C)  = round(frac_i * C / P_nom)                      (0 -> no order at that size)
Commission models per order:
  engine_nominal = max(1, 0.005 sh)                         (the audit's model, but nominal shares)
  ibkr_fixed     = min(max(1, 0.005 sh), 1% notional) + sell reg fees (FINRA TAF 0.000166/sh, SEC 27.8 per $1M)
  ibkr_tiered    = min(max(0.35, 0.0035 sh), 1% notional) + 0.0002 clearing + 0.0015 auction fee (upper bound) per sh
                   + same sell reg fees
  per_share_only = 0.005 sh                                 (large-account marginal cost, no minimum)
Backtest's own drag = engine commission / NAV_(t-1) (adjusted shares, NAV compounding from USD 100K).
Drags reported in bp of NAV per year (sum over fills in the window / years).
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
from data.norgate_loader import load_price_timeseries  # noqa: E402

LED = REPO / "results/research/strategy_readiness_audit_20260928/review_trade/ledgers"
OUT = REPO / "results/research/strategy_readiness_audit_20260928/review_trade"
END = "2026-09-25"
L3Y = pd.Timestamp("2023-09-25")
CAPS = (12_000.0, 18_000.0, 30_000.0, 100_000.0, 1_000_000.0)
_K: dict[str, pd.Series] = {}


def k_ser(asset):
    if asset not in _K:
        df = load_price_timeseries(asset, start_date_str="1998-01-01", end_date_str=END)
        _K[asset] = (df["Unadjusted Close"] / df["Close"]).astype(float)
    return _K[asset]


def sell_reg(sh, notional, is_sell):
    return np.where(is_sell, np.maximum(0.01, 0.000166 * sh) * (sh > 0) + 27.8e-6 * notional, 0.0)


def analyse(key):
    tx = pd.read_csv(LED / f"{key}_tx.csv", parse_dates=["bar"])
    tx = tx[tx["order_id"] != -1].copy() if "order_id" in tx else tx
    nav = pd.read_csv(LED / f"{key}_nav.csv", index_col=0, parse_dates=True)["nav"]
    nav_prev = nav.shift(1).fillna(100_000.0)
    tx["nav_prev"] = tx["bar"].map(nav_prev)
    tx["notional_bt"] = (tx["amount"] * tx["price"]).abs()
    tx["frac"] = tx["notional_bt"] / tx["nav_prev"]
    tx["k"] = [float(k_ser(a).reindex([b]).iloc[0]) if b in k_ser(a).index else np.nan for a, b in zip(tx["asset"], tx["bar"])]
    tx["k"] = tx["k"].fillna(1.0)
    tx["p_nom"] = tx["price"] * tx["k"]
    tx["is_sell"] = tx["amount"] < 0
    tx["bt_drag"] = tx["commission"] / tx["nav_prev"]
    # engine's per-share part in adjusted vs nominal shares, at the backtest's own NAV
    tx["bt_nominal_comm"] = np.maximum(1.0, 0.005 * (tx["notional_bt"] / tx["p_nom"]).round())
    tx["bt_nominal_drag"] = tx["bt_nominal_comm"] / tx["nav_prev"]
    first, last = tx["bar"].min(), pd.Timestamp(END)
    res = {"key": key, "first_fill": str(first.date()), "fills": int(len(tx))}
    for wname, mask, years in (("full", tx["bar"] >= first, (last - first).days / 365.25),
                               ("last3y", tx["bar"] >= L3Y, (last - L3Y).days / 365.25)):
        w = tx[mask]
        row = {"years": round(years, 2), "fills_per_year": round(len(w) / years, 1),
               "backtest_engine_bp_per_yr": 1e4 * w["bt_drag"].sum() / years,
               "backtest_nominal_units_bp_per_yr": 1e4 * w["bt_nominal_drag"].sum() / years}
        for C in CAPS:
            notional = w["frac"] * C
            sh = (notional / w["p_nom"]).round()
            live = sh > 0
            notional_r = sh * w["p_nom"]
            reg = sell_reg(sh, notional_r, w["is_sell"].values)
            eng = np.where(live, np.maximum(1.0, 0.005 * sh), 0.0)
            fixed = np.where(live, np.minimum(np.maximum(1.0, 0.005 * sh), 0.01 * notional_r) + reg, 0.0)
            tiered = np.where(live, np.minimum(np.maximum(0.35, 0.0035 * sh), 0.01 * notional_r) + 0.0017 * sh + reg, 0.0)
            pso = np.where(live, 0.005 * sh, 0.0)
            c = int(C)
            row[f"C{c}"] = {
                "orders_per_yr": round(float(live.sum()) / years, 1),
                "orders_rounding_to_zero_share": int((~live).sum()),
                "median_order_usd": round(float(notional[live].median()), 0),
                "p10_order_usd": round(float(notional[live].quantile(0.10)), 0),
                "share_orders_at_1usd_min": round(float((0.005 * sh[live] < 1.0).mean()), 3),
                "engine_nominal_bp_per_yr": round(1e4 * eng.sum() / C / years, 1),
                "ibkr_fixed_bp_per_yr": round(1e4 * fixed.sum() / C / years, 1),
                "ibkr_tiered_bp_per_yr": round(1e4 * tiered.sum() / C / years, 1),
                "per_share_only_bp_per_yr": round(1e4 * pso.sum() / C / years, 1),
            }
        res[wname] = row
    return res


if __name__ == "__main__":
    out = {}
    for key in sys.argv[1:] or ["taa3x", "taa1n", "btal_qqq", "ndx_vxn"]:
        out[key] = analyse(key)
        print(json.dumps(out[key], indent=1, default=float), flush=True)
    (OUT / "r2_commission_owner_size.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
