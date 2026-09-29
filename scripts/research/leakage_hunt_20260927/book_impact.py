"""Book-level effect of the two NEEDS FIX findings of the 2026-09-27 leakage hunt (research only).

Fixes applied as daily-return deltas to the sleeves the books already use (so everything else stays identical):
- HPI vote (C-3): live slot semantics.  delta_t = r(hpi_liveslot)_t - r(hpi_base)_t, both $100k engine runs from
  2004 (results/research/leakage_hunt_20260927/mr/full_runs); added to the book's hpi_vote sleeve.
- Inflation Compass (C-1): T5YIE observations dated before T.  delta_t = r(causal)_t - r(base)_t from a fresh run of
  taa_compass_causal's two variants; added to the book's infl_compass sleeve.

Baseline sleeves: the fund-menu inventory (shelf_books.load_inputs(fixed=False)) with the momentum sleeves swapped
to the fixed versions exactly as portfolio_refresh_20260927 did (ndx_vxn -> ndx_atrfix, mosaic -> mosaic_fix,
etf_ind -> etf_ind_fix).  Book model: common.book_return_ser (independent pods, annual reset).
Window: the books' exact window 2012-10-02 .. 2026-08-19.

Usage: uv run python scripts/research/leakage_hunt_20260927/book_impact.py
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
for path in (REPO, HERE, REPO / "scripts" / "research" / "growth_shelf_v2_20260926"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import shelf_books as sb  # noqa: E402
import common  # noqa: E402

OUT = REPO / "results" / "research" / "leakage_hunt_20260927"
REFRESH = REPO / "results" / "research" / "portfolio" / "portfolio_refresh_20260927"
START, END = pd.Timestamp("2012-10-02"), pd.Timestamp("2026-08-19")
FIXED_MOMENTUM = {"ndx_vxn": "ndx_atrfix", "mosaic": "mosaic_fix", "etf_ind": "etf_ind_fix"}
MENU_ALIAS = {  # fund-menu yaml import -> inventory alias
    "strategies.taa_beyond_6040.strategy_taa_adaptive_macro_core5": "core5",
    "strategies.taa_beyond_6040.strategy_taa_tactical_fixed_income_ief_lqd": "tactical_fi",
    "strategies.taa_beyond_6040.strategy_taa_month_end_rebalancing_flow": "eom_flow",
    "strategies.mean_reversion.strategy_mr_us_sector_etf_ibs_downshock_vox_iyr": "sector_vox_iyr",
    "strategies.dv2.strategy_mr_dv2:DVO2Strategy": "dv2",
    "strategies.hpi.strategy_mr_hpi_sp500_2_3_5_vote": "hpi_vote",
    "strategies.mean_reversion.strategy_mr_sector_dispersion_ibs_kie_ihi_asset_sma200": "disp_kie_ihi_sma",
    "strategies.taa_df.strategy_taa_df_btal_fallback_tqqq_vix_cash": "taa_btal_tqqq",
    "strategies.momentum.strategy_mo_mosaic_russell1000:MosaicRussell1000Strategy": "mosaic",
    "strategies.momentum.strategy_mo_atr_normalized_ndx_vxn_scaled:VxnScaledAtrNormalizedNdxStrategy": "ndx_vxn",
    "strategies.taa_df.strategy_taa_inflation_compass": "infl_compass",
}


def nav_returns(nav: pd.Series) -> pd.Series:
    return nav.astype(float).pct_change()


def hpi_delta() -> pd.Series:
    runs = OUT / "mr" / "full_runs"
    base = pd.read_csv(runs / "hpi_base" / "daily.csv.gz", index_col=0, parse_dates=True)["total_value"]
    live = pd.read_csv(runs / "hpi_liveslot" / "daily.csv.gz", index_col=0, parse_dates=True)["total_value"]
    return (nav_returns(live) - nav_returns(base)).fillna(0.0)


def compass_delta() -> pd.Series:
    cache = OUT / "book_impact_compass_nav.csv"
    if not cache.exists():
        import taa_compass_causal as tc
        from taa_common import BOOK_END_STR, compute_decisions, patched, read_frozen_series
        import strategies.taa_df.strategy_taa_inflation_compass as compass
        t5 = read_frozen_series("T5YIE")
        with patched():
            dec = compute_decisions("compass", end_date_str=BOOK_END_STR)
            sig = compass.load_signal_close_df(symbol_list=compass.DEFAULT_CONFIG.signal_asset_tuple,
                                               start_date_str=compass.DEFAULT_CONFIG.start_date_str,
                                               end_date_str=BOOK_END_STR)
        ref = dec["month_end_weight_df"]
        causal = ref.copy()
        for T in ref.index:
            # *** CRITICAL*** decision T may only use T5YIE observations dated before T (published at T+1).
            _, w = compass.compute_month_end_signal_and_weight_df(sig.loc[:T], t5[t5.index < T], compass.DEFAULT_CONFIG)
            if T in w.index:
                causal.loc[T] = w.loc[T].reindex(causal.columns).values
        navs = {}
        with patched():
            for name, weights in (("base", ref), ("causal", causal)):
                navs[name] = tc.run_from_weights(weights, dec).results["total_value"]
        pd.DataFrame(navs).to_csv(cache)
    nav = pd.read_csv(cache, index_col=0, parse_dates=True)
    return (nav_returns(nav["causal"]) - nav_returns(nav["base"])).fillna(0.0)


def metrics(r: pd.Series) -> dict:
    v = (1 + r).cumprod()
    years = (r.index[-1] - r.index[0]).days / 365.25
    return {"cagr": float(v.iloc[-1] ** (1 / years) - 1), "sharpe": float(r.mean() / r.std() * np.sqrt(252)),
            "maxdd": float((v / v.cummax() - 1).min())}


def main() -> None:
    sleeve = sb.load_inputs(fixed=False)["sleeve"]
    refresh = pd.read_csv(REFRESH / "sleeve_series_incl_2008.csv.gz", index_col=0, parse_dates=True)
    for new in FIXED_MOMENTUM.values():
        sleeve[new] = refresh[new].reindex(sleeve.index)
    inventory = pd.read_csv(common.STUDY_DIR_PATH / "inventory" / "sleeve_returns.csv.gz", index_col=0, parse_dates=True)
    for alias in ("infl_compass", "disp_kie_ihi_sma"):
        sleeve[alias] = inventory[alias].reindex(sleeve.index)
    window = sleeve.loc[START:END]
    fixed = window.copy()
    fixed["hpi_vote"] = window["hpi_vote"] + hpi_delta().reindex(window.index).fillna(0.0)
    fixed["infl_compass"] = window["infl_compass"] + compass_delta().reindex(window.index).fillna(0.0)

    books = {}
    for name, (weights, policy) in {**sb.growth_books(), **sb.defensive_books()}.items():
        books[name] = ({FIXED_MOMENTUM.get(a, a): w for a, w in weights.items()}, policy)
    books["loren 60/40 (drift)"] = ({"ndx_atrfix": 0.4, "taa_btal_tqqq": 0.6}, "none")
    for path in sorted((REPO / "portfolios").glob("fund_menu_*.yaml")):
        spec = yaml.safe_load(path.read_text(encoding="utf-8"))
        pods = spec.get("pods") or spec.get("pod_list") or []
        weights = {}
        for pod in pods:
            alias = FIXED_MOMENTUM.get(MENU_ALIAS[pod["strategy_import_str"]], MENU_ALIAS[pod["strategy_import_str"]])
            weights[alias] = weights.get(alias, 0.0) + float(pod["weight_float"])
        total = sum(weights.values())
        books[f"menu {path.stem.replace('fund_menu_', '')}"] = ({a: w / total for a, w in weights.items()}, "annual")

    rows = []
    for name, (weights, policy) in books.items():
        touched = weights.get("hpi_vote", 0.0) + weights.get("infl_compass", 0.0)
        before, _ = common.book_return_ser(window, weights, policy)
        after, _ = common.book_return_ser(fixed, weights, policy)
        m0, m1 = metrics(before), metrics(after)
        rows.append({"book": name, "hpi_weight": weights.get("hpi_vote", 0.0),
                     "compass_weight": weights.get("infl_compass", 0.0),
                     "cagr_before": m0["cagr"], "cagr_after": m1["cagr"], "d_cagr_pp": 100 * (m1["cagr"] - m0["cagr"]),
                     "sharpe_before": m0["sharpe"], "sharpe_after": m1["sharpe"],
                     "d_sharpe": m1["sharpe"] - m0["sharpe"], "d_maxdd_pp": 100 * (m1["maxdd"] - m0["maxdd"]),
                     "touched_bool": touched > 0})
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "book_impact.csv", index=False, float_format="%.6g")
    for key, label in (("hpi_vote", "hpi"), ("infl_compass", "compass")):
        m0, m1 = metrics(window[key]), metrics(fixed[key])
        print(f"sleeve {label}: CAGR {m0['cagr']:.4f} -> {m1['cagr']:.4f}, Sharpe {m0['sharpe']:.3f} -> {m1['sharpe']:.3f}")
    print(table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
