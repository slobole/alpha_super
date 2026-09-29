"""Industry-ETF DV2 checks (research-only).

1. Corrected engine run from 2000-01-03 (production DVO2IndustryEtfStrategy, native Turnover ADV, pricing from 1998)
   vs the legacy research path etf_ind_adv50 (dv2_deep_20260925 replica, ADV = Unadjusted Close x adjusted Volume)
   that portfolio_refresh_20260927/refresh_books.py (via growth_shelf_v2 shelf_books.load_inputs) splices in for
   every date <= 2012-01-03.
2. The book's engine run results/research/strategy/strategy_mr_dv2_industry_etf/vanilla_backtest/2026-09-27_020616
   vs this study's etf_base arm (same module, same data) on 2012-01-04..2026-08-19.
3. Per-ETF first price date and first date with ADV63 > $50M, native Turnover vs legacy mixed-unit ADV.

Writes mr/etf_checks.json and mr/etf_eligibility.csv.
"""

from __future__ import annotations

import json
import pickle

import numpy as np
import pandas as pd

import mr_common as mc
from strategies.dv2 import strategy_mr_dv2_industry_etf as etf

LEGACY = mc.REPO / "results/research/dv2_deep_20260925/sources/etf_ind_adv50__path.csv.gz"
BOOK_PKL = (mc.REPO / "results/research/strategy/strategy_mr_dv2_industry_etf/vanilla_backtest/2026-09-27_020616"
            / "strategy_mr_dv2_industry_etf.pkl")


def stats(ret: pd.Series) -> dict:
    ret = ret.dropna()
    if len(ret) < 2:
        return {}
    nav = (1 + ret).cumprod()
    years = len(ret) / 252
    return {"start": ret.index[0].date().isoformat(), "end": ret.index[-1].date().isoformat(),
            "cagr": float(nav.iloc[-1] ** (1 / years) - 1), "sharpe": float(ret.mean() / ret.std() * np.sqrt(252)),
            "max_dd": float((nav / nav.cummax() - 1).min())}


def main() -> None:
    pricing = etf.get_prices(list(etf.INDUSTRY_ETF_SYMBOL_TUPLE), ["$SPX"], start_date="1998-01-01", end_date=None)
    universe = etf.build_history_universe_df(pricing)
    strategy = mc.run(mc.make_etf(universe), pricing, "2000-01-03", mc.STUDY_END)
    nav = strategy.results["total_value"].astype(float)
    corrected = nav.pct_change()
    tx = strategy.get_transactions()
    out_dir = mc.OUT / "full_runs" / "etf_2000_corrected"
    out_dir.mkdir(parents=True, exist_ok=True)
    strategy.results[["total_value", "portfolio_value", "cash"]].to_csv(out_dir / "daily.csv.gz")
    tx.to_csv(out_dir / "transactions.csv.gz", index=False)

    legacy = pd.read_csv(LEGACY, index_col="date", parse_dates=True)["total_value_float"].pct_change()
    pre = (corrected.index > pd.Timestamp("2000-01-03")) & (corrected.index <= pd.Timestamp("2012-01-03"))
    idx = corrected.index[pre].intersection(legacy.index)
    result = {
        "pre2012_corrected_engine": stats(corrected.reindex(idx)),
        "pre2012_legacy_research_splice": stats(legacy.reindex(idx)),
        "pre2012_daily_corr": float(corrected.reindex(idx).corr(legacy.reindex(idx))),
        "pre2012_fills_corrected": int((tx["bar"] <= pd.Timestamp("2012-01-03")).sum()),
        "full_2000_2026_corrected": stats(corrected.loc["2000-01-04":]),
        "post_2012_10_02_corrected_2000start": stats(corrected.loc["2012-10-02":]),
    }
    idx2 = legacy.index[(legacy.index > pd.Timestamp("2012-01-03")) & (legacy.index <= mc.STUDY_END)]
    result["post2012_legacy_research"] = stats(legacy.reindex(idx2))
    result["post2012_corrected_2000start"] = stats(corrected.reindex(idx2))

    with BOOK_PKL.open("rb") as handle:
        book = pickle.load(handle)
    book_ret = book.results["total_value"].astype(float).pct_change()
    arm = pd.read_csv(mc.OUT / "full_runs" / "etf_base" / "daily.csv.gz", index_col=0, parse_dates=True)
    arm_ret = arm["total_value"].pct_change()
    idx3 = arm_ret.index[(arm_ret.index > pd.Timestamp("2012-01-03")) & (arm_ret.index <= mc.STUDY_END)]
    result["book_run_vs_study_etf_base"] = {
        "max_abs_daily_diff": float((book_ret.reindex(idx3) - arm_ret.reindex(idx3)).abs().max()),
        "book": stats(book_ret.reindex(idx3)), "study": stats(arm_ret.reindex(idx3)),
        "book_n_tx_to_study_end": int((book.get_transactions()["bar"] <= mc.STUDY_END).sum()),
    }

    rows = []
    for symbol in etf.INDUSTRY_ETF_SYMBOL_TUPLE:
        close = pricing[(symbol, "Close")].dropna()
        adv_native = pricing[(symbol, "Turnover")].rolling(63, min_periods=63).mean()
        adv_legacy = (pricing[(symbol, "Unadjusted Close")] * pricing[(symbol, "Volume")]).rolling(63, min_periods=63).mean()
        first = lambda ser: (ser[ser > 50e6].index[0].date().isoformat() if (ser > 50e6).any() else None)  # noqa: E731
        ratio = (pricing[(symbol, "Unadjusted Close")] / pricing[(symbol, "Close")]).dropna()
        rows.append({"symbol": symbol, "first_price": close.index[0].date().isoformat(),
                     "first_adv50_native": first(adv_native), "first_adv50_legacy": first(adv_legacy),
                     "legacy_over_native_adv_2005": float((adv_legacy / adv_native).loc["2005"].median())
                     if len(adv_native.loc["2005"].dropna()) else None,
                     "split_factor_min_max": [float(ratio.min()), float(ratio.max())],
                     "days_eligible_native_pre2012": int((adv_native.loc[:"2012-01-03"] > 50e6).sum()),
                     "days_eligible_legacy_pre2012": int((adv_legacy.loc[:"2012-01-03"] > 50e6).sum())})
    pd.DataFrame(rows).to_csv(mc.OUT / "etf_eligibility.csv", index=False)
    (mc.OUT / "etf_checks.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
