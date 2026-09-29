"""Portfolio refresh after the corporate-action leakage fix (fb81e86, 2026-09-27).

Reuses the growth shelf v2 pipeline (shelf_books.load_inputs / book_rows: same sleeves, same 2008 proxy for the
BTAL TAA sleeves, same pod model with annual reset, same metrics and gates) and only swaps the sleeves the fix
changed. The legacy sleeves stay in the frame under their old aliases so old and new books sit side by side.

Swapped sleeves (all end 2026-09-25 and are cut to the study end 2026-08-19):
- ndx_natr20  : NDX NATR20 + VXN scaling (strategies.momentum.strategy_mo_natr20_ndx_vxn_scaled), leak-free by
                construction (scale-free ranking). Codex leakage audit arm vxn/natr20_corrected.
- ndx_atrfix  : the live NDX-VXN rule (ROC12 / dollar ATR20) with ATR rebased to decision-date nominal dollars.
                Audit arm vxn/asof_atr_corrected. Same as the fixed module in the repo.
- mosaic_fix  : MOSAIC R1000 with as-of dollar ATR and native Turnover liquidity. Audit arm mosaic/asof_atr_corrected.
- etf_ind_fix : industry-ETF DV2 with native Turnover liquidity (BENCH run 2026-09-27, from 2012-01-03). Before
                that date it is filled with the legacy research run (see the overlap check in the output).

Parity check: the audit's legacy_vintage_diagnostic arm must reproduce the inventory ndx_vxn sleeve.

Usage: python refresh_books.py
"""

from __future__ import annotations

import json
from pathlib import Path
import pickle
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
SHELF_DIR = REPO / "scripts" / "research" / "growth_shelf_v2_20260926"
for path in (REPO, SHELF_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import shelf_books as sb  # noqa: E402  (also puts the fund-menu helpers on sys.path)
import common  # noqa: E402
import evaluation  # noqa: E402

OUT = REPO / "results" / "research" / "portfolio" / "portfolio_refresh_20260927"
AUDIT = Path(r"C:\Users\User\Documents\Codex\2026-09-26\new-chat\outputs\leakage-audit\runs")
ETF_PKL = (REPO / "results/research/strategy/strategy_mr_dv2_industry_etf/vanilla_backtest/2026-09-27_020616"
           / "strategy_mr_dv2_industry_etf.pkl")
END, CUT, LONG = sb.END, sb.CUT, sb.LONG
T = 1.0 / 3.0

AUDIT_ARM = {"ndx_natr20": ("vxn", "natr20_corrected"), "ndx_atrfix": ("vxn", "asof_atr_corrected"),
             "mosaic_fix": ("mosaic", "asof_atr_corrected"), "ndx_legacy_check": ("vxn", "legacy_vintage_diagnostic"),
             "mosaic_legacy_check": ("mosaic", "legacy_vintage_diagnostic"), "mosaic_natr20": ("mosaic", "natr20_corrected")}


# ─── corrected sleeves ───────────────────────────────────────────────────────


def nav_to_returns(nav: pd.Series, invested: pd.Series) -> pd.Series:
    """Returns from the day before the first invested session (same rule as common.sleeve_nav_df)."""
    first = invested[invested].index[0]
    position = max(nav.index.get_loc(first) - 1, 0)
    return nav.iloc[position:].pct_change().iloc[1:]


def load_audit_arm(family: str, arm: str) -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    folder = AUDIT / family / arm
    daily = pd.read_csv(folder / "daily_results.csv", index_col=0, parse_dates=True)
    tx = pd.read_csv(folder / "transactions.csv", parse_dates=["bar"])
    nav = daily["total_value"].astype(float)
    ret = nav_to_returns(nav, daily["portfolio_value"].abs() > 1e-9)
    tx = pd.DataFrame({"date": tx["bar"], "asset_str": tx["asset"], "signed_notional_float": tx["amount"] * tx["price"]})
    return ret, nav, tx


def load_etf_fix() -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    with ETF_PKL.open("rb") as handle:
        strategy = pickle.load(handle)
    results = strategy.results
    nav = results["total_value"].astype(float)
    ret = nav_to_returns(nav, results["portfolio_value"].abs() > 1e-9)
    tx = strategy.get_transactions() if hasattr(strategy, "get_transactions") else strategy.transactions
    tx = tx.reset_index() if "bar" not in tx.columns else tx
    tx = pd.DataFrame({"date": pd.to_datetime(tx["bar"]), "asset_str": tx["asset"],
                       "signed_notional_float": tx["amount"] * tx["price"]})
    return ret, nav, tx


def add_sleeve(data: dict, alias: str, ret: pd.Series, nav: pd.Series, tx: pd.DataFrame, pre_fill: pd.Series | None = None) -> None:
    index = data["sleeve"].index
    ser = ret.reindex(index)
    if pre_fill is not None:
        early = index < ret.index[0]
        ser[early] = pre_fill.reindex(index)[early]
    drag = evaluation.extra_slippage_cost_ser(tx, nav, 0.0005).reindex(index).fillna(0.0)
    data["sleeve"][alias] = ser
    data["stressed"][alias] = ser - drag.where(ser.notna())
    for frame in data["long"].values():
        frame[alias] = ser
    data["tx"][alias] = tx
    data["nav"][alias] = nav.loc[:END]
    data["alias_set"].add(alias)


# ─── books ───────────────────────────────────────────────────────────────────


def swap(weights: dict, mapping: dict) -> dict:
    out = {}
    for alias, w in weights.items():
        out[mapping.get(alias, alias)] = out.get(mapping.get(alias, alias), 0.0) + w
    return out


NEW_MAP = {"ndx_vxn": "ndx_natr20", "mosaic": "mosaic_fix", "etf_ind": "etf_ind_fix"}
ATRFIX_MAP = {"ndx_vxn": "ndx_atrfix", "mosaic": "mosaic_fix", "etf_ind": "etf_ind_fix"}


def growth_books() -> dict:
    """The declared growth shelf v2 grid, in three versions: legacy (old sleeves), NATR20 (new default), ATR$-fixed."""
    books = {}
    for name, (weights, policy) in sb.growth_books().items():
        books[f"{name} | legacy"] = (weights, policy)
        books[f"{name} | NATR20"] = (swap(weights, NEW_MAP), policy)
        books[f"{name} | ATR$ fixed"] = (swap(weights, ATRFIX_MAP), policy)
    # The owner's own book (portfolios/loren.yaml): TAA 60 / NDX 40, no rebalance.
    loren = {"taa_btal_tqqq": 0.6, "ndx_vxn": 0.4}
    books["loren 60/40 (drift) | legacy"] = (loren, "none")
    books["loren 60/40 (drift) | NATR20"] = (swap(loren, NEW_MAP), "none")
    books["loren 60/40 (drift) | ATR$ fixed"] = (swap(loren, ATRFIX_MAP), "none")
    # References, not candidates: each sleeve family alone.
    books["TAA 3x rank alone | ref"] = ({"taa_btal_tqqq": 1.0}, "annual")
    books["NDX NATR20 alone | ref"] = ({"ndx_natr20": 1.0}, "annual")
    books["NDX ATR$ fixed alone | ref"] = ({"ndx_atrfix": 1.0}, "annual")
    return books


def defensive_books() -> dict:
    books = {}
    for name, (weights, policy) in sb.defensive_books().items():
        books[f"{name} | legacy"] = (weights, policy)
        if any(a in weights for a in NEW_MAP):
            books[f"{name} | NATR20"] = (swap(weights, NEW_MAP), policy)
            books[f"{name} | ATR$ fixed"] = (swap(weights, ATRFIX_MAP), policy)
    return books


def product_books() -> dict:
    """The seven fund-menu product YAMLs (exact window only: some sleeves start after 2008)."""
    import yaml
    from run_sources import SLEEVE_ALIAS_BY_IMPORT_DICT
    alias_by_import = {k.split(":")[0]: v for k, v in SLEEVE_ALIAS_BY_IMPORT_DICT.items()}
    books = {}
    for path in sorted((REPO / "portfolios").glob("fund_menu_*.yaml")):
        spec = yaml.safe_load(path.read_text(encoding="utf-8"))
        weights = {alias_by_import[p["strategy_import_str"].split(":")[0]]: float(p["weight_float"]) for p in spec["pods"]}
        policy = "annual" if (spec.get("rebalance") or {}).get("frequency_str") == "annually" else "none"
        name = path.stem.replace("fund_menu_", "")
        books[f"{name} | legacy"] = (weights, policy)
        books[f"{name} | NATR20"] = (swap(weights, NEW_MAP), policy)
        books[f"{name} | ATR$ fixed"] = (swap(weights, ATRFIX_MAP), policy)
    return books


def product_rows(books: dict, data: dict) -> pd.DataFrame:
    sleeve, bench = data["sleeve"], data["bench"]
    rows = []
    for name, (weights, policy) in books.items():
        exact = common.book_return_ser(sleeve.loc[CUT:END, list(weights)], weights, policy)[0]
        base = sleeve.index[sleeve.index.get_loc(exact.index[0]) - 1]
        m = common.metric_dict(exact, bench["SPXTR"], bench["TBILL"], base)
        rows.append({"book": name, "cagr": m["cagr_float"], "vol": m["volatility_float"], "sharpe": m["sharpe_rf0_float"],
                     "maxdd": m["max_drawdown_float"], "calmar": m["cagr_float"] / abs(m["max_drawdown_float"]),
                     "sharpe_2022_26": sb.sharpe(exact.loc[sb.H2_START:]), "worst_year": m["worst_year_float"],
                     "affected_weight": float(sum(w for a, w in weights.items() if a in ("ndx_vxn", "mosaic")))})
    return pd.DataFrame(rows).set_index("book")


# ─── extra series for the report ─────────────────────────────────────────────


def book_series(books: dict, frame: pd.DataFrame, start: pd.Timestamp) -> pd.DataFrame:
    out = {}
    for name, (weights, policy) in books.items():
        out[name] = common.book_return_ser(frame.loc[start:END, list(weights)], weights, policy)[0]
    return pd.DataFrame(out)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    sb.WIRED_SET |= {"ndx_atrfix"}
    sb.NASDAQ_SET |= {"ndx_natr20", "ndx_atrfix"}
    sb.NEEDS_DICT.update({"ndx_natr20": "switch NDX pod to NATR20 (research tier, not wired)",
                          "mosaic_fix": "wire MOSAIC (PM_READY)", "etf_ind_fix": "wire industry-ETF DV2 (research tier)"})
    sb.DAILY_MR_SET |= {"etf_ind_fix"}
    sb.ETF_POD_SET |= {"etf_ind_fix"}
    sb.URGENT_SET |= {"etf_ind_fix"}

    data = sb.load_inputs(fixed=False)
    checks = {}
    extra = {}
    for alias, (family, arm) in AUDIT_ARM.items():
        extra[alias] = load_audit_arm(family, arm)
    etf_ret, etf_nav, etf_tx = load_etf_fix()

    # Parity: the audit's legacy arm vs the inventory sleeve (different capital: $100k vs $1M, so rounding differs).
    for alias, legacy in (("ndx_legacy_check", "ndx_vxn"), ("mosaic_legacy_check", "mosaic")):
        a = extra[alias][0].loc[:END]
        b = data["sleeve"][legacy].reindex(a.index)
        both = a.notna() & b.notna()
        checks[f"{alias}_vs_inventory"] = {"daily_corr": float(a[both].corr(b[both])),
                                          "cagr_audit": float((1 + a[both]).prod() ** (252 / both.sum()) - 1),
                                          "cagr_inventory": float((1 + b[both]).prod() ** (252 / both.sum()) - 1)}
    # Overlap: legacy vs corrected industry-ETF DV2 (decides whether the legacy research run may fill pre-2012).
    legacy_etf = data["sleeve"]["etf_ind"]
    both = etf_ret.index[(etf_ret.index > sb.ETF_ENGINE_START) & (etf_ret.index <= END)]
    checks["etf_ind_overlap_2012_2026"] = {
        "daily_corr": float(etf_ret.reindex(both).corr(legacy_etf.reindex(both))),
        "cagr_fixed": float((1 + etf_ret.reindex(both)).prod() ** (252 / len(both)) - 1),
        "cagr_legacy": float((1 + legacy_etf.reindex(both)).prod() ** (252 / len(both)) - 1)}

    for alias in ("ndx_natr20", "ndx_atrfix", "mosaic_fix", "mosaic_natr20"):
        add_sleeve(data, alias, *extra[alias])
    add_sleeve(data, "etf_ind_fix", etf_ret, etf_nav, etf_tx, pre_fill=legacy_etf)
    (OUT / "checks.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
    print(json.dumps(checks, indent=2))

    growth, defensive = growth_books(), defensive_books()
    growth_table, prior_weights = sb.book_rows(growth, data)
    defensive_table, _ = sb.book_rows(defensive, data)
    growth_table.to_csv(OUT / "growth_books.csv", float_format="%.6g")
    defensive_table.to_csv(OUT / "defensive_books.csv", float_format="%.6g")
    product_table = product_rows(product_books(), data)
    product_table.to_csv(OUT / "product_books.csv", float_format="%.6g")
    print(product_table.round(3).to_string())

    # Sleeve-level table on the exact window and incl. 2008.
    sleeve_rows = []
    for alias in ["taa_btal_tqqq", "taa_btal_1n_tqqq", "ndx_vxn", "ndx_natr20", "ndx_atrfix", "mosaic", "mosaic_fix",
                  "mosaic_natr20", "dv2", "hpi_vote", "etf_ind", "etf_ind_fix", "core5", "taa_btal_lin_qqq", "tactical_fi", "eom_flow"]:
        for window, frame, start in (("exact", data["sleeve"], CUT), ("incl_2008", data["long"]["new"], LONG), ("full", data["sleeve"], None)):
            ser = frame[alias].loc[start:END].dropna() if start is not None else frame[alias].loc[:END].dropna()
            if alias in ("etf_ind_fix",) and window == "full":
                ser = ser.loc[ser.index >= sb.ETF_ENGINE_START]
            base = frame.index[frame.index.get_loc(ser.index[0]) - 1]
            m = common.metric_dict(ser, data["bench"]["SPXTR"], data["bench"]["TBILL"], base)
            sleeve_rows.append({"alias": alias, "window": window, "start": ser.index[0].date().isoformat(),
                                "cagr": m["cagr_float"], "vol": m["volatility_float"], "sharpe": m["sharpe_rf0_float"],
                                "maxdd": m["max_drawdown_float"], "worst_year": m["worst_year_float"], "beta": m["beta_spx_float"]})
    pd.DataFrame(sleeve_rows).to_csv(OUT / "sleeves.csv", index=False, float_format="%.6g")

    # Daily series for charts.
    book_series(growth, data["long"]["new"], LONG).to_csv(OUT / "growth_series_incl_2008.csv.gz", float_format="%.8g")
    book_series(defensive, data["long"]["new"], LONG).to_csv(OUT / "defensive_series_incl_2008.csv.gz", float_format="%.8g")
    data["long"]["new"].loc[LONG:END, ["taa_btal_tqqq", "taa_btal_1n_tqqq", "ndx_vxn", "ndx_natr20", "ndx_atrfix", "mosaic",
                                       "mosaic_fix", "dv2", "hpi_vote", "etf_ind_fix", "core5", "taa_btal_lin_qqq"]].to_csv(
        OUT / "sleeve_series_incl_2008.csv.gz", float_format="%.8g")
    data["sleeve"][["ndx_vxn", "ndx_natr20", "ndx_atrfix", "mosaic", "mosaic_fix", "mosaic_natr20"]].loc[:END].to_csv(
        OUT / "momentum_series_full.csv.gz", float_format="%.8g")
    data["bench"].loc[LONG:END].to_csv(OUT / "bench_incl_2008.csv.gz", float_format="%.8g")

    pd.set_option("display.width", 260)
    pd.set_option("display.max_columns", 40)
    show = ["cagr", "sharpe", "maxdd", "calmar", "long_cagr", "long_sharpe", "long_maxdd", "long_calmar", "gfc",
            "sharpe_2012_21", "sharpe_2022_26", "calmar_plus5", "worst_year", "trade_days_per_year", "gates_pass"]
    print(growth_table[show].round(3).sort_values("long_calmar", ascending=False).to_string())
    print(defensive_table[["cagr", "sharpe", "maxdd", "long_maxdd", "gfc", "long_calmar", "sharpe_2022_26", "worst_year"]].round(3).to_string())
    print(pd.DataFrame(sleeve_rows).round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
