"""The real modules against the research pods of growth_mr_products_20261101: m cell by cell, then the engine trade by trade, on the same frozen sign-off inputs.

Usage: parity.py dv2|hpi [f32]. Writes results/research/mr_capsule_liq_build_20261101/parity_<pod>[_f32].json and engine files beside it. Run alone: it loads the whole frame.
f32 rounds m to float32 as the research matrices were saved, so that the two runs must agree share for share; without it the gaps are the rounding's.
Run from this checkout; the research pods and sign-off inputs are read from the main folder (git-ignored results).
"""

from __future__ import annotations

import gzip
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
MAIN = Path(r"C:\Users\User\Documents\workspace\alpha_super")
BASE = MAIN / "results" / "review" / "wired_signoff_20261015" / "baselines"
STUDY = MAIN / "results" / "scout" / "growth_mr_products_20261101"
OUT = REPO / "results" / "research" / "mr_capsule_liq_build_20261101"
END = "2026-10-14"
SOURCE = {"dv2": "strategy_mr_dv2_vix_gated_bil", "hpi": "strategy_mr_hpi_vote_vix_gated_bil"}


def inputs(pod: str) -> tuple[pd.DataFrame, dict]:
    pricing_df = pd.read_parquet(BASE / SOURCE[pod] / "pricing.parquet").loc[:END]
    auxiliary = pickle.load(gzip.open(BASE / SOURCE[pod] / "auxiliary.pkl.gz"))
    benchmark = "$SPX" if pod == "dv2" else "$SPXTR"
    symbols = pricing_df.columns.get_level_values(0).unique().astype(str)
    pricing_df.attrs["norgate_adjustment_by_symbol_dict"] = {s: ("TOTALRETURN" if s == benchmark else "CAPITALSPECIAL") for s in symbols}
    return pricing_df, auxiliary


def build(pod: str, auxiliary: dict):
    kwargs = dict(strategy_name_str=f"parity_{pod}", parking_enabled_bool=True, spmo_parking_enabled_bool=False, universe_df=auxiliary["universe_df"], vix_close_ser=auxiliary["vix_close_ser"], capital_base_float=1_000_000.0)
    if pod == "dv2":
        from strategies.mr_capsule.dv2_vix_gated_liq import build_dv2_capsule_liquidity_strategy

        return build_dv2_capsule_liquidity_strategy(**kwargs)
    from strategies.mr_capsule.hpi_vote_vix_gated_liq import build_hpi_capsule_liquidity_strategy

    return build_hpi_capsule_liquidity_strategy(**kwargs)


def rounded(strategy):
    from strategies.mr_capsule.liquidity_sizing import LIQUIDITY_SIZE_FIELD_STR

    class _Rounded(type(strategy)):
        def compute_signals(self, pricing_data):
            signal_df = super().compute_signals(pricing_data)
            size_list = [col for col in signal_df.columns if col[1] == LIQUIDITY_SIZE_FIELD_STR]
            signal_df[size_list] = signal_df[size_list].astype("float32").astype(float)
            return signal_df.copy()

    strategy.__class__ = _Rounded
    return strategy


def main(pod: str, f32: bool = False) -> None:
    from alpha.engine.backtest import run_daily
    from strategies.mr_capsule.liquidity_sizing import LIQUIDITY_SIZE_FIELD_STR

    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    pricing_df, auxiliary = inputs(pod)
    strategy = build(pod, auxiliary)
    res: dict = {"pod": pod, "float32_m": f32}
    tag = f"{pod}_f32" if f32 else pod

    # 1. m, cell by cell, against the research matrix (saved as float32).
    signal_df = strategy.compute_signals(pricing_df)
    size_df = signal_df.xs(LIQUIDITY_SIZE_FIELD_STR, axis=1, level=1)
    research_df = pd.read_parquet(STUDY / "m" / f"{pod}_today.parquet")
    common = research_df.columns.intersection(size_df.columns)
    gap = (size_df.loc[research_df.index, common].astype("float32") - research_df[common]).abs()
    res["m"] = {"rows": len(research_df), "columns_research": len(research_df.columns), "columns_common": len(common), "max_gap": float(gap.max().max()), "cells_over_1e-5": int((gap > 1e-5).sum().sum())}
    if res["m"]["cells_over_1e-5"]:
        worst = gap.stack().sort_values(ascending=False).head(10)
        res["m"]["worst"] = {f"{d.date()} {s}": float(v) for (d, s), v in worst.items()}
    print(json.dumps(res["m"]), round(time.time() - t0), flush=True)
    del signal_df, size_df, research_df, gap

    # 2. The engine on the same inputs, against the research pod.
    strategy = rounded(build(pod, auxiliary)) if f32 else build(pod, auxiliary)
    calendar = pricing_df.index[pricing_df.index >= pd.Timestamp("2004-01-01")]
    run_daily(strategy, pricing_df, calendar, show_progress=False, show_signal_progress_bool=False)
    results = strategy.results
    nav = results["total_value"].astype(float)
    nav.index = pd.to_datetime(nav.index).normalize()
    tx = strategy.get_transactions()
    pd.DataFrame({"total_value_float": nav, "cash_float": results["cash"].astype(float).to_numpy()}).to_csv(OUT / f"{tag}_path.csv.gz", index_label="date", compression="gzip")
    new_tx = pd.DataFrame({"date": pd.to_datetime(tx["bar"]).dt.normalize(), "asset_str": tx["asset"].astype(str), "amount_float": pd.to_numeric(tx["amount"]), "price_float": pd.to_numeric(tx["price"])})
    new_tx.to_csv(OUT / f"{tag}_tx.csv.gz", index=False, compression="gzip")
    research_path = pd.read_csv(STUDY / "pods" / f"{pod}_g_l__path.csv.gz", index_col="date", parse_dates=True)
    research_tx = pd.read_csv(STUDY / "pods" / f"{pod}_g_l__transactions.csv.gz", parse_dates=["date"])
    both = pd.concat([nav, research_path["total_value_float"].astype(float)], axis=1, keys=["new", "research"]).dropna()
    daily_gap = (both["new"].pct_change() - both["research"].pct_change()).abs()
    res["nav"] = {"sessions": len(both), "final_new": float(both["new"].iloc[-1]), "final_research": float(both["research"].iloc[-1]), "final_gap_pct": float(both["new"].iloc[-1] / both["research"].iloc[-1] - 1) * 100, "max_daily_gap": float(daily_gap.max()), "corr": float(both["new"].pct_change().corr(both["research"].pct_change()))}
    key = ["date", "asset_str"]
    new_group = new_tx.groupby(key)["amount_float"].sum()
    research_group = research_tx.groupby(key)["amount_float"].sum()
    merged = pd.concat([new_group, research_group], axis=1, keys=["new", "research"])
    stock = ~merged.index.get_level_values("asset_str").isin(["BIL", "SPMO"])
    only_new, only_research = merged["research"].isna(), merged["new"].isna()
    share_gap = (merged["new"] - merged["research"]).abs()
    res["trades"] = {"rows_new": len(new_group), "rows_research": len(research_group), "only_new": int(only_new.sum()), "only_research": int(only_research.sum()), "stock_only_new": int((only_new & stock).sum()), "stock_only_research": int((only_research & stock).sum()),
                     "same_shares": int((share_gap == 0).sum()), "stock_max_share_gap": float(share_gap[stock].max()), "stock_rows_share_gap_over_1": int((share_gap[stock] > 1).sum()), "bil_rows_share_gap_over_1": int((share_gap[~stock] > 1).sum())}
    res["seconds"] = round(time.time() - t0)
    (OUT / f"parity_{tag}.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    print(json.dumps(res, indent=1, default=float), flush=True)


if __name__ == "__main__":
    np.seterr(all="ignore")
    main(sys.argv[1], len(sys.argv) > 2 and sys.argv[2] == "f32")
