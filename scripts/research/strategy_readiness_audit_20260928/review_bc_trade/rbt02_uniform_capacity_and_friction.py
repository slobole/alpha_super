"""Review BC-trade RBT02: one capacity and one small-account-friction method applied to every Tier B/C pod.

Inputs are the auditors' own per-fill participation files (order notional as a fraction of NAV_(t-1) and the
20-session median native-Turnover ADV at the fill bar). Nothing is re-simulated; the question is whether the TR
verdicts hold when the SAME rules are applied to every pod.

CAPACITY (trade as if from today, last 3 years = AM-01 window, 2023-09-25..2026-09-25):
  p            = frac * C / ADV20
  impact_bp    = lambda * sqrt(p / 1%)          (house square-root model, alpha/engine/capacity_analysis.py:185-219)
  extra_bp     = max(model_slip_bp, impact_bp) - model_slip_bp   (only the part the backtest does not already charge)
  drag pp/yr   = sum(frac * extra_bp) / years
  lambda       : MOO_ETF_PROXY 40 / 66.4 bp for ETFs, MOO_NASDAQ_LARGE 66.4 / 114 for NDX stocks, MOC 8.2 / 17.8 (EOM)
  guardrail    : MOO hard 0.10% of ADV (MOC 0.50%)  (capacity_analysis.py:91-94)
  capacity_025 : smallest C at which the central extra drag reaches 0.25 pp/yr (protocol materiality)
  capacity_p99 : C at which the last-3y p99 order reaches 5% of ADV (protocol C1)

SMALL-ACCOUNT FRICTION at C in {12K, 18K, 30K} (constant pod size, nominal shares = round(frac*C/P_nom),
P_nom = Norgate Unadjusted Close on the fill bar), same fee formulas as review_trade/rt02:
  ibkr_fixed  = min(max(1, 0.005 sh), 1% notional) + sell reg fees
  ibkr_tiered = min(max(0.35, 0.0035 sh), 1% notional) + 0.0017 sh + sell reg fees   (upper-bound exchange/clearing)
  model       = the pod's own backtest commission rule applied to the same nominal orders
  gap         = ibkr - model (bp of NAV per year): what the backtest leaves out at that pod size
Sell side is unknown in the participation files, so reg fees are charged on half of the orders' notional
(SEC 27.8 per USD 1M + FINRA TAF); they are < 1 bp/yr everywhere and do not move any conclusion.
Whole shares: orders that round to 0 shares, and the largest single-share price as % of NAV (bound on the per-name
floor-rounding error).
"""
from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
from data.norgate_loader import load_price_timeseries  # noqa: E402

RES = REPO / "results/research/strategy_readiness_audit_20260928"
OUT = RES / "review_bc_trade"
END = pd.Timestamp("2026-09-25")
L3Y = pd.Timestamp("2023-09-25")
YEARS = (END - L3Y).days / 365.25
OWNER_CAPS = (12_000.0, 18_000.0, 30_000.0)
CAP_CAPS = (12_000.0, 30_000.0, 100_000.0, 300_000.0, 1_000_000.0, 10_000_000.0)

ETF = (40.0, 66.4, 0.0010)
NDXL = (66.4, 114.0, 0.0010)
MOC = (8.2, 17.8, 0.0050)

# key: (file, format, (lam_c, lam_s, hard), model_slip_bp, (per_share, minimum), last-3y CAGR % from the findings)
PODS = {
    # Tier A reference (same method; Tier A review numbers came from rt04)
    "taa3x": ("tradability/taa3x/participation_by_fill.csv", "A", ETF, 2.5, (0.005, 1.0), None),
    "ndx_vxn": ("tradability/ndx_vxn/participation_by_fill.csv", "A", NDXL, 2.5, (0.005, 1.0), None),
    # Tier B
    "compass_xlk": ("tierb_macro/tradability_compass_xlk/participation_by_fill.csv", "A", ETF, 5.0, (0.0, 0.0), None),
    "compass_qqq": ("tierb_macro/tradability_compass_qqq/participation_by_fill.csv", "A", ETF, 5.0, (0.0, 0.0), None),
    "tactical_fi": ("tierb_macro/tradability_tactical_fi/participation_by_fill.csv", "A", ETF, 5.0, (0.0, 0.0), 4.71),
    "eom": ("tierb_etf_mr/eom_participation.csv", "A", MOC, 2.5, (0.005, 1.0), 2.47),
    "vox_iyr": ("tierb_etf_mr/vox_participation.csv", "A", ETF, 2.5, (0.005, 1.0), 6.66),
    "disp_kie_ihi_xlc": ("tierb_etf_mr/xlc_participation.csv", "A", ETF, 2.5, (0.00525, 0.0), 8.76),
    "disp_xlc_sma200": ("tierb_etf_mr/xlc200_participation.csv", "A", ETF, 2.5, (0.00525, 0.0), 6.65),
    "disp_kie_ihi_sma200": ("tierb_etf_mr/kie200_participation.csv", "A", ETF, 2.5, (0.00525, 0.0), 6.86),
    "etf_dv2": ("tierbc_dv2etf_taa2x/etf/participation_by_fill.csv", "B", ETF, 2.5, (0.005, 1.0), 9.85),
    # Tier C
    "qld_1n": ("tierbc_dv2etf_taa2x/taa2x/participation_by_fill_qld_1n.csv", "B", ETF, 2.5, (0.005, 1.0), 27.27),
    "sso_1n": ("tierbc_dv2etf_taa2x/taa2x/participation_by_fill_sso_1n.csv", "B", ETF, 2.5, (0.005, 1.0), 22.07),
    "btal_qld_1n": ("tierbc_dv2etf_taa2x/taa2x/participation_by_fill_btal_qld_1n.csv", "B", ETF, 2.5, (0.005, 1.0), 27.51),
    "lin_qqq": ("tierbc_dv2etf_taa2x/taa2x/participation_by_fill_lin_qqq.csv", "B", ETF, 2.5, (0.005, 1.0), 20.84),
    "ctc": ("tierc_hedge/ctc/c1_participation_orders.csv.gz", "A", ETF, 10.0, (0.0, 0.0), -5.42),
    "vixm": ("tierc_hedge/vixm/c1_participation_orders.csv.gz", "A", ETF, 10.0, (0.0, 0.0), -5.30),
    "trinity": ("tierc_hedge/trin/c1_participation_orders.csv.gz", "A", ETF, 1.0, (0.005, 1.0), 12.90),
}
_PX: dict[str, pd.Series] = {}


def nominal_close(asset: str) -> pd.Series:
    if asset not in _PX:
        try:
            df = load_price_timeseries(asset, start_date_str="2015-01-01", end_date_str=str(END.date()))
            _PX[asset] = df["Unadjusted Close"].astype(float)
        except Exception:  # delisted NDX names etc.
            _PX[asset] = pd.Series(dtype=float)
    return _PX[asset]


def load(path: str, fmt: str) -> pd.DataFrame:
    p = RES / path
    df = pd.read_csv(p, compression="gzip" if p.suffix == ".gz" else None, parse_dates=["bar"])
    if fmt == "B":
        df = df.rename(columns={"frac": "order_frac_of_nav", "adv20": "adv20_usd"})
    df = df[["bar", "asset", "order_frac_of_nav", "adv20_usd"]].copy()
    df["order_frac_of_nav"] = df["order_frac_of_nav"].abs()
    return df.dropna(subset=["order_frac_of_nav", "adv20_usd"])


def extra_drag_pp(w: pd.DataFrame, cap: float, lam: float, slip_bp: float) -> float:
    p = w["order_frac_of_nav"] * cap / w["adv20_usd"]
    imp = lam * np.sqrt(p / 0.01)
    extra = np.maximum(slip_bp, imp) - slip_bp
    return float((w["order_frac_of_nav"] * extra).sum() / 1e4 / YEARS * 100.0)


def solve_cap(fun, target: float) -> float | None:
    grid = np.exp(np.linspace(np.log(1e3), np.log(1e9), 400))
    for c in grid:
        if fun(c) >= target:
            return float(c)
    return None


def fees(w: pd.DataFrame, cap: float, per_share: float, minimum: float) -> dict:
    notional = w["order_frac_of_nav"] * cap
    sh = (notional / w["p_nom"]).round()
    live = sh > 0
    nr = sh * w["p_nom"]
    reg = 0.5 * (27.8e-6 * nr + np.maximum(0.01, 0.000166 * sh) * live)
    fixed = np.where(live, np.minimum(np.maximum(1.0, 0.005 * sh), 0.01 * nr) + reg, 0.0)
    tiered = np.where(live, np.minimum(np.maximum(0.35, 0.0035 * sh), 0.01 * nr) + 0.0017 * sh + reg, 0.0)
    model = np.where(live, np.maximum(minimum, per_share * sh) if minimum > 0 else per_share * sh, 0.0)
    f = lambda x: round(1e4 * float(np.sum(x)) / cap / YEARS, 1)  # noqa: E731
    return {
        "orders_per_yr": round(float(live.sum()) / YEARS, 1),
        "orders_zero_share": int((~live).sum()),
        "median_order_usd": round(float(notional[live].median()), 0) if live.any() else None,
        "share_at_1usd_min_fixed": round(float((0.005 * sh[live] < 1.0).mean()), 3) if live.any() else None,
        "ibkr_fixed_bp_yr": f(fixed), "ibkr_tiered_bp_yr": f(tiered), "model_bp_yr": f(model),
        "gap_fixed_pp_yr": round((f(fixed) - f(model)) / 100.0, 2),
        "gap_tiered_pp_yr": round((f(tiered) - f(model)) / 100.0, 2),
        "max_share_price_pct_nav": round(100.0 * float(w["p_nom"].max()) / cap, 2),
    }


def main() -> None:
    out = {"window": [str(L3Y.date()), str(END.date())], "years": round(YEARS, 3), "pods": {}}
    for key, (path, fmt, (lc, ls, hard), slip, (ps, mn), cagr3) in PODS.items():
        df = load(path, fmt)
        w = df[(df["bar"] >= L3Y) & (df["bar"] <= END)].copy()
        r = {"fills_last3y": int(len(w)), "turnover_x_nav_per_yr": round(float(w["order_frac_of_nav"].sum()) / YEARS, 2),
             "model_slippage_bp": slip, "lambda_central_bp": lc, "last3y_cagr_pct": cagr3}
        if len(w) == 0:
            out["pods"][key] = r
            continue
        cap_rows = {}
        for C in CAP_CAPS:
            p = w["order_frac_of_nav"] * C / w["adv20_usd"]
            cap_rows[f"C{int(C)}"] = {
                "p50_pct_adv": round(100 * float(p.median()), 4), "p99_pct_adv": round(100 * float(p.quantile(0.99)), 3),
                "share_orders_over_hard_guardrail": round(float((p > hard).mean()), 3),
                "extra_drag_central_pp_yr": round(extra_drag_pp(w, C, lc, slip), 3),
                "extra_drag_stress_pp_yr": round(extra_drag_pp(w, C, ls, slip), 3),
            }
        r["capacity"] = cap_rows
        r["capacity_usd_extra_drag_025pp"] = solve_cap(lambda c: extra_drag_pp(w, c, lc, slip), 0.25)
        r["capacity_usd_extra_drag_025pp_stress"] = solve_cap(lambda c: extra_drag_pp(w, c, ls, slip), 0.25)
        p99_1 = float((w["order_frac_of_nav"] / w["adv20_usd"]).quantile(0.99))
        r["capacity_usd_p99_5pct_adv"] = round(0.05 / p99_1, 0)
        r["binding_asset_by_extra_drag_at_1m"] = (
            w.assign(x=w["order_frac_of_nav"] * (np.maximum(slip, lc * np.sqrt(w["order_frac_of_nav"] * 1e6 / w["adv20_usd"] / 0.01)) - slip))
            .groupby("asset")["x"].sum().sort_values(ascending=False).head(3).div(1e4 * YEARS / 100).round(3).to_dict())
        if key != "ndx_vxn":
            w["p_nom"] = [float(nominal_close(a).asof(b)) if len(nominal_close(a)) else np.nan for a, b in zip(w["asset"], w["bar"])]
            w2 = w.dropna(subset=["p_nom"])
            r["fees"] = {f"C{int(C)}": fees(w2, C, ps, mn) for C in OWNER_CAPS}
            r["fee_price_coverage"] = round(len(w2) / len(w), 3)
        out["pods"][key] = r
        print(key, json.dumps({k: v for k, v in r.items() if k not in ("capacity",)}, default=float)[:900], flush=True)
    (OUT / "rbt02_uniform_capacity_and_friction.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
