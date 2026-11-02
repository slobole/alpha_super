"""Fresh engine runs through the panel's last session, of the 1/N pods, the sized pods, TAA 1N and CORE5, for the monthly comparison. Descriptive.

Usage: fresh_runs.py dv2|hpi|etf. One family per process; the two pods of a family share one loaded frame. A flag file is written once the frame is loaded, so the next family can start without loading Norgate at the same time.
Writes results/research/mr_capsule_liq_build_20261101/fresh/<alias>__path.csv.gz and __tx.csv.gz.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
OUT = REPO / "results" / "research" / "mr_capsule_liq_build_20261101" / "fresh"
CAPITAL = 1_000_000.0


def last_session() -> str:
    from data.norgate_loader import load_price_timeseries

    return str(pd.Timestamp(load_price_timeseries("$SPX", start_date_str="2026-01-01", end_date_str=str(pd.Timestamp.today().date())).index[-1]).date())


def save(alias: str, strategy) -> None:
    results = strategy.results
    path_df = pd.DataFrame({"total_value_float": results["total_value"].astype(float), "cash_float": results["cash"].astype(float)})
    path_df.index = pd.to_datetime(path_df.index).normalize()
    path_df.to_csv(OUT / f"{alias}__path.csv.gz", index_label="date", compression="gzip")
    tx = strategy.get_transactions()
    pd.DataFrame({"date": pd.to_datetime(tx["bar"]).dt.normalize(), "asset_str": tx["asset"].astype(str), "amount_float": pd.to_numeric(tx["amount"]), "price_float": pd.to_numeric(tx["price"]),
                  "signed_notional_float": pd.to_numeric(tx["total_value"]), "commission_float": pd.to_numeric(tx["commission"])}).to_csv(OUT / f"{alias}__tx.csv.gz", index=False, compression="gzip")
    print("done", alias, path_df.index[-1].date(), round(path_df["total_value_float"].iloc[-1]), len(tx), flush=True)


def run_family(family: str) -> None:
    from alpha.engine.backtest import run_daily
    from strategies.mr_capsule.vix_stress_gate import load_vix_close_ser

    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    end_str = last_session()
    print("last session", end_str, flush=True)
    if family == "dv2":
        from strategies.mr_capsule.dv2_vix_gated import build_dv2_capsule_strategy, load_pricing_data
        from strategies.mr_capsule.dv2_vix_gated_liq import build_dv2_capsule_liquidity_strategy

        pricing_df, universe_df = load_pricing_data(end_str)
        builders = (("dv2_g_n", build_dv2_capsule_strategy), ("dv2_g_l", build_dv2_capsule_liquidity_strategy))
    else:
        from strategies.mr_capsule.hpi_vote_vix_gated import build_hpi_capsule_strategy, load_hpi_capsule_pricing_data
        from strategies.mr_capsule.hpi_vote_vix_gated_liq import build_hpi_capsule_liquidity_strategy

        pricing_df, universe_df = load_hpi_capsule_pricing_data(end_str)
        builders = (("hpi_g_n", build_hpi_capsule_strategy), ("hpi_g_l", build_hpi_capsule_liquidity_strategy))
    vix_close_ser = load_vix_close_ser(end_str)
    (OUT / f"loaded_{family}.flag").write_text(end_str, encoding="utf-8")
    print("loaded", pricing_df.shape, round(time.time() - t0), flush=True)
    calendar_idx = pricing_df.index[pricing_df.index >= pd.Timestamp("2004-01-01")]
    for alias, builder in builders:
        strategy = builder(strategy_name_str=f"fresh_{alias}", parking_enabled_bool=True, spmo_parking_enabled_bool=False, universe_df=universe_df, vix_close_ser=vix_close_ser, capital_base_float=CAPITAL)
        run_daily(strategy, pricing_df, calendar_idx, show_progress=False, show_signal_progress_bool=False)
        save(alias, strategy)
        print(alias, round(time.time() - t0), flush=True)


def run_etf() -> None:
    import importlib

    OUT.mkdir(parents=True, exist_ok=True)
    end_str = last_session()
    for alias, module_str in (("taa3x_1n", "strategies.taa_df.strategy_taa_df_btal_1n_fallback_tqqq_vix_cash"), ("core5", "strategies.taa_beyond_6040.strategy_taa_adaptive_macro_core5")):
        module_obj = importlib.import_module(module_str)
        strategy = module_obj.run_variant(show_display_bool=False, save_results_bool=False, capital_base_float=CAPITAL, end_date_str=end_str)
        save(alias, strategy)


if __name__ == "__main__":
    if sys.argv[1] == "etf":
        run_etf()
    else:
        run_family(sys.argv[1])
