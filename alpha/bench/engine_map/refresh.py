"""Refresh every engine's daily returns through the last close, for the Engine Map.

Each engine runs through its own run_variant from RUN_START, results kept in memory only (save_results_bool=False).
History up to the frozen fund frame's end stays exactly as saved; only later sessions are appended, with the same
fair-cash add the fund study used (scripts/research/shelf_rebuild_20260929/lib.py cash_realism_add). A short run
is enough because every rule's state settles within weeks; the overlap check below proves it against the frame.

Writes only under results/bench/engine_map/_work. No tracked file changes and nothing touches a live path.

Usage: .venv/Scripts/python.exe -X utf8 -m alpha.bench.engine_map refresh [--workers 5] [--only alias ...] [--check]
(refresh, then build). Read-only check: the same with --check.
"""
from __future__ import annotations

import argparse
import contextlib
import gc
import importlib
import io
import json
import os
import pickle
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

import numpy as np
import pandas as pd

from alpha.bench.engine_map import DAILY_DIR_PATH as DAILY, REPO_ROOT_PATH as REPO, WORK_DIR_PATH as WORK

SCRATCH = WORK / "engine_scratch"
FROZEN_MAIN = REPO / "results/scout/growth_hunt_20261030/shared/frame_main.parquet"
FROZEN_HOUSE = REPO / "results/scout/growth_hunt_20261030/shared/frame_house.parquet"
DTB3 = REPO / "results/scout/growth_hunt_20261030/shared/dtb3.parquet"
RESEARCH_RUNS = REPO / "results/research/strategy"
RUN_START = "2024-01-02"
CAPITAL = 1_000_000.0

RUNS: dict[str, tuple[str, dict]] = {
    "taa3x": ("strategies.taa_df.strategy_taa_df_btal_fallback_tqqq_vix_cash", {}),
    "taa3x_1n": ("strategies.taa_df.strategy_taa_df_btal_1n_fallback_tqqq_vix_cash", {}),
    "taa2x_1n": ("strategies.taa_df.strategy_taa_df_btal_1n_fallback_qld_vix_cash", {}),
    "taa_1n_qld": ("strategies.taa_df.strategy_taa_df_1n_fallback_qld_vix_cash", {}),
    "taa_1n_sso": ("strategies.taa_df.strategy_taa_df_1n_fallback_sso_vix_cash", {}),
    "btal_qqq": ("strategies.taa_df.strategy_taa_df_btal_linearity_1n_fallback_qqq_vix_cash", {}),
    "taa_lin_qqq": ("strategies.taa_df.strategy_taa_df_linearity_1n_fallback_qqq_vix_cash", {}),
    "ndx_vxn": ("strategies.momentum.strategy_mo_atr_normalized_ndx_vxn_scaled", {}),
    "ndx_atr": ("strategies.momentum.strategy_mo_atr_normalized_ndx", {}),
    "ndx_atr_cap": ("strategies.momentum.strategy_mo_atr_normalized_ndx_vxn_scaled_sector_cap", {}),
    "ndx_natr_cap": ("strategies.momentum.strategy_mo_natr20_ndx_vxn_scaled_sector_cap", {}),
    "dv2": ("strategies.dv2.strategy_mr_dv2", {}),
    "dv2_g": ("strategies.mr_capsule.strategy_mr_dv2_vix_gated_bil", {}),
    "dv2_g_spmo": ("strategies.mr_capsule.strategy_mr_dv2_vix_gated_spmo", {}),
    "hpi_vote": ("strategies.hpi.strategy_mr_hpi_sp500_2_3_5_vote", {}),
    "hpi_g": ("strategies.mr_capsule.strategy_mr_hpi_vote_vix_gated_bil", {}),
    "hpi_g_spmo": ("strategies.mr_capsule.strategy_mr_hpi_vote_vix_gated_spmo", {}),
    "hpi_ibs_rsi": ("strategies.hpi.strategy_mr_hpi_sp500_ibs_rsi_exit", {}),
    # No column in the frozen frame: build.py takes their history from the newest saved research run, as for the Flow Vote.
    "dv2_g_liq": ("strategies.mr_capsule.strategy_mr_dv2_vix_gated_liq_bil", {}),
    "hpi_g_liq": ("strategies.mr_capsule.strategy_mr_hpi_vote_vix_gated_liq_bil", {}),
    "etf_dv2": ("strategies.dv2.strategy_mr_dv2_industry_etf", {}),
    "downshock": ("strategies.mean_reversion.strategy_mr_us_sector_etf_ibs_downshock_vox_iyr", {}),
    "disp": ("strategies.mean_reversion.strategy_mr_sector_dispersion_ibs_kie_ihi_asset_sma200", {}),
    "disp_xlc": ("strategies.mean_reversion.strategy_mr_sector_dispersion_ibs_kie_ihi_xlc", {}),
    "disp_xlc_sma": ("strategies.mean_reversion.strategy_mr_sector_dispersion_ibs_kie_ihi_xlc_asset_sma200", {}),
    "core5": ("strategies.taa_beyond_6040.strategy_taa_adaptive_macro_core5", {}),
    "compass": ("strategies.taa_df.strategy_taa_inflation_compass", {}),
    "compass_qqq": ("strategies.taa_df.strategy_taa_inflation_compass_qqq", {}),
    "eom_flow": ("strategies.taa_beyond_6040.strategy_taa_month_end_rebalancing_flow", {}),
    # No column in the frozen frame, here and for the two SPMO versions above: build.py takes the history from the newest saved research run that covers its start.
    # A short run is safe, its drift history loads from 2002 whatever the start.
    "eom_vote": ("strategies.taa_beyond_6040.strategy_taa_month_end_flow_vote", {}),
    # No column in the frozen frame either: history from the saved research run of Nov 5, 2026. Leg 3 is the heaviest run on the page.
    "mo_leg1": ("strategies.momentum.strategy_mo_capsule_leg1_natr_ndx", {}),
    "mo_leg2": ("strategies.momentum.strategy_mo_capsule_leg2_residual_ndx", {}),
    "mo_leg3": ("strategies.momentum.strategy_mo_capsule_leg3_residual_r1000", {}),
    "vrp": ("strategies.vix_stuff.strategy_vrp_svxy_vixm_rsi2", {}),
    "t1": ("strategies.taa_beyond_6040.strategy_taa_beyond_6040_trend_tilt", {}),
}
HEAVY = ("hpi_vote", "hpi_g", "hpi_g_spmo", "hpi_ibs_rsi", "dv2", "dv2_g", "dv2_g_spmo", "dv2_g_liq", "hpi_g_liq", "ndx_vxn", "ndx_atr", "ndx_atr_cap", "ndx_natr_cap")
RETRY_UNTIL_SECONDS = 1200  # no retry starts this long after the first run started, so one refresh stays near 30 minutes


def run_one(alias: str, run_start: str) -> dict:
    """Worker: run one engine, store its path (NAV, invested value, cash), return a status row."""
    os.chdir(REPO)
    sys.path.insert(0, str(REPO))
    DAILY.mkdir(parents=True, exist_ok=True)
    SCRATCH.mkdir(parents=True, exist_ok=True)
    started = time.time()
    module_str, extra = RUNS[alias]
    log_path = DAILY / f"{alias}.log"
    with log_path.open("w", encoding="utf-8") as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        module = importlib.import_module(module_str)
        strategy = module.run_variant(
            show_display_bool=False, save_results_bool=False, output_dir_str=str(SCRATCH), backtest_start_date_str=run_start,
            capital_base_float=CAPITAL, end_date_str=None, **extra,
        )
    results = path_of(strategy.results)
    results.to_parquet(DAILY / f"{alias}.parquet")
    last = str(results.index[-1].date())
    try:  # what it holds is a drawer extra: a run without a weight frame just has none
        (DAILY / f"{alias}.holdings.json").write_text(json.dumps(holdings_of(strategy)), encoding="utf-8")
    except Exception:
        (DAILY / f"{alias}.holdings.json").unlink(missing_ok=True)
    del strategy
    gc.collect()
    return {"alias": alias, "last": last, "rows": int(len(results)), "seconds": round(time.time() - started, 1)}


def holdings_of(strategy) -> dict:
    """Weights at the last close and on average over the run, from the run's realized weight frame (or its daily
    target weights when it keeps no realized frame). Weights are shares of NAV; Cash is the idle part, negative when borrowed."""
    weights = getattr(strategy, "realized_weight_df", None)
    if weights is None or not len(weights):
        weights = getattr(strategy, "daily_target_weights", None)
    weights = weights.astype(float).fillna(0.0)
    weights.index = pd.to_datetime(weights.index).normalize()
    held = weights.abs() > 1e-4
    last = weights.iloc[-1]
    return {
        "date": str(weights.index[-1].date()), "start": str(weights.index[0].date()),
        "now": {asset: round(float(w), 4) for asset, w in last[last.abs() > 1e-4].items()},
        # name: [average weight, percent of sessions held]
        "avg": {asset: [round(float(weights[asset].mean()), 5), round(float(held[asset].mean() * 100), 1)] for asset in weights.columns[held.any()]},
    }


def path_of(results: pd.DataFrame) -> pd.DataFrame:
    """A run's NAV, invested value and cash, under the names the return arithmetic below reads."""
    path = results.loc[:, ["total_value", "portfolio_value", "cash"]].copy()
    path.index = pd.to_datetime(path.index).normalize()
    path.columns = ["total_value_float", "portfolio_value_float", "cash_float"]
    return path.astype(float)


class _Stub:
    """Stands in for a strategy class inside a saved run, so reading one imports no strategy code and no norgatedata."""

    def __init__(self, *args, **kwargs) -> None:
        pass

    def __setstate__(self, state) -> None:
        self.__dict__.update(state if isinstance(state, dict) else {"state": state})

    # a saved dict or list subclass (the momentum legs keep one) is rebuilt item by item: keep the items, read none of them
    def __setitem__(self, key, value) -> None:
        self.__dict__.setdefault("_items", {})[key] = value

    def append(self, value) -> None:
        self.__dict__.setdefault("_list", []).append(value)

    def extend(self, values) -> None:
        self.__dict__.setdefault("_list", []).extend(values)


class _StubUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        if module.split(".")[0] in ("strategies", "alpha", "data", "norgatedata"):
            return _Stub
        return super().find_class(module, name)


def saved_run(stem: str, covers: str | None = None) -> tuple[str, dict]:
    """The newest saved research run of a strategy module, or the newest that starts by the date in covers: its folder name and the saved object's fields. Reads only."""
    runs = [run for run in sorted((RESEARCH_RUNS / stem / "vanilla_backtest").glob("*")) if (run / f"{stem}.pkl").exists()]
    for run in reversed(runs):
        state = _StubUnpickler(io.BytesIO((run / f"{stem}.pkl").read_bytes())).load().__dict__
        if covers is None or pd.Timestamp(state["results"].index[0]) <= pd.Timestamp(covers):
            return run.name, state
    raise FileNotFoundError(f"no saved run of {stem}" + (f" covers {covers}" if covers else ""))


def house_returns(path_df: pd.DataFrame) -> pd.Series:
    nav = path_df["total_value_float"]
    invested = path_df["portfolio_value_float"].abs() > 1e-9
    first = nav.index.get_loc(invested[invested].index[0])
    return nav.iloc[max(first - 1, 0):].pct_change(fill_method=None).iloc[1:]


def cash_add(path_df: pd.DataFrame, rate: pd.Series) -> pd.Series:
    """Fair cash, same arithmetic as lib.cash_realism_add: cash and NAV of the prior close, DTB3 before the session."""
    cash = path_df["cash_float"].shift(1)
    nav = path_df["total_value_float"].shift(1)
    days = pd.Series(path_df.index, index=path_df.index).diff().dt.days
    y = rate.reindex(path_df.index)
    earn = cash.clip(lower=0.0) * (y - 0.005).clip(lower=0.0)
    pay = (-cash).clip(lower=0.0) * (y + 0.015)
    return ((earn - pay) * days / 360.0 / nav).fillna(0.0)


def dtb3_rate(index: pd.DatetimeIndex) -> pd.Series:
    """The frame's saved DTB3 column, already a decimal rate on the prior observation; the last value carries forward."""
    rate = pd.read_parquet(DTB3)["dtb3"].dropna().sort_index()
    return rate.reindex(rate.index.union(index)).ffill().reindex(index)


# A new run must reproduce the frozen frame where they overlap, or the map keeps its last build: a silent gap would reach the page as history.
OVERLAP_LIMITS = {"min_days": 200, "min_corr": 0.999, "max_gap_bp": 5.0, "max_mean_gap_bp_year": 25.0}


def overlap_failures(checks: list[dict]) -> list[str]:
    """One line per check outside the limits, empty when every run reproduces its frozen history."""
    out = []
    for c in checks:
        why = []
        if c["days"] < OVERLAP_LIMITS["min_days"]:
            why.append(f"{c['days']} days overlap")
        if c["corr"] < OVERLAP_LIMITS["min_corr"]:
            why.append(f"corr {c['corr']}")
        if c["max_gap_bp"] > OVERLAP_LIMITS["max_gap_bp"]:
            why.append(f"max gap {c['max_gap_bp']} bp")
        if abs(c["mean_gap_bp_year"]) > OVERLAP_LIMITS["max_mean_gap_bp_year"]:
            why.append(f"mean gap {c['mean_gap_bp_year']} bp a year")
        if why:
            out.append(f"{c['alias']}: {', '.join(why)}")
    return out


def overlap_check(alias: str, frozen_house: pd.DataFrame) -> dict:
    """Does the short run reproduce the frozen frame where they overlap? Judged on the last year before the frame ends."""
    path = pd.read_parquet(DAILY / f"{alias}.parquet")
    mine = house_returns(path)
    theirs = frozen_house[alias].dropna()
    start = theirs.index[-1] - pd.Timedelta(days=365)
    common = mine.loc[start:].index.intersection(theirs.index)
    a, b = mine.reindex(common), theirs.reindex(common)
    return {
        "alias": alias, "days": int(len(common)), "corr": round(float(a.corr(b)), 5), "max_gap_bp": round(float((a - b).abs().max() * 1e4), 1),
        "mean_gap_bp_year": round(float((a - b).mean() * 252 * 1e4), 1), "mine_year": round(float((1 + a).prod() - 1) * 100, 2), "frame_year": round(float((1 + b).prod() - 1) * 100, 2),
    }


def write_bench(run_start: str) -> str:
    """QQQ, BIL and the other factor funds, total returns from Norgate, for the sessions after the frozen frames."""
    sys.path.insert(0, str(REPO))
    import norgatedata as nd

    closes = {}
    for symbol in ("QQQ", "BIL", "IEF", "GLD", "DBC", "UUP"):
        frame = nd.price_timeseries(symbol, stock_price_adjustment_setting=nd.StockPriceAdjustmentType.TOTALRETURN, padding_setting=nd.PaddingType.NONE,
                                    start_date=run_start, timeseriesformat="pandas-dataframe")
        closes[symbol] = frame["Close"].astype(float)
    returns = pd.DataFrame(closes).pct_change(fill_method=None).dropna(how="all")
    returns.to_parquet(DAILY / "_bench.parquet")
    return str(returns.index[-1].date())


def run_pool(aliases: list[str], run_start: str, workers: int, status: dict) -> list[str]:
    """One pass over these engines, status rows written as they finish. Returns the ones that failed, in the order given."""
    failed = set()
    with ProcessPoolExecutor(max_workers=workers, max_tasks_per_child=1) as pool:
        futures = {pool.submit(run_one, alias, run_start): alias for alias in aliases}
        for future in as_completed(futures):
            alias = futures[future]
            try:
                row = future.result()
                status["runs"][alias] = {**row, "ok": True, "at": datetime.now().isoformat(timespec="seconds")}
                print("done", alias, row["last"], row["seconds"], flush=True)
            except Exception as exc:  # keep the last good series for this engine and say so
                previous = status["runs"].get(alias, {})
                status["runs"][alias] = {**previous, "alias": alias, "ok": False, "error": repr(exc)[:300], "traceback": traceback.format_exc()[-800:], "at": datetime.now().isoformat(timespec="seconds")}
                print("FAILED", alias, repr(exc)[:200], flush=True)
                failed.add(alias)
    return [alias for alias in aliases if alias in failed]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m alpha.bench.engine_map refresh", description=__doc__)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument("--start", default=RUN_START)
    parser.add_argument("--check", action="store_true", help="only print the overlap check for runs already on disk")
    args = parser.parse_args(argv)
    aliases = [a for a in sorted(RUNS, key=lambda x: (x not in HEAVY, x)) if not args.only or a in args.only]
    frozen_house = pd.read_parquet(FROZEN_HOUSE)
    status_path = WORK / "refresh_status.json"
    status = json.loads(status_path.read_text(encoding="utf-8")) if status_path.exists() else {"runs": {}}
    if not args.check:
        WORK.mkdir(parents=True, exist_ok=True)
        started = time.time()
        # A busy night can lose an engine to memory: once the pool has drained, each failed one runs once more, alone.
        for alias in run_pool(aliases, args.start, args.workers, status):
            if time.time() - started > RETRY_UNTIL_SECONDS:
                print("no time left to retry", alias, flush=True)
                continue
            first_error = status["runs"][alias]["error"]
            print("retry", alias, flush=True)
            if not run_pool([alias], args.start, 1, status):
                status["runs"][alias]["first_error"] = first_error
        try:
            status["bench_last"] = write_bench(args.start)
            status.pop("bench_error", None)
        except Exception as exc:  # the engines' rows still get written; the refresh command stops before the build on this
            status["bench_error"] = repr(exc)[:300]
            print("FAILED bench", repr(exc)[:200], flush=True)
        status["seconds"] = round(time.time() - started, 1)
        status["finished"] = datetime.now().isoformat(timespec="seconds")
        status_path.write_text(json.dumps(status, indent=1), encoding="utf-8")
    checks = []
    for alias in aliases:
        if (DAILY / f"{alias}.parquet").exists() and alias in frozen_house.columns:
            checks.append(overlap_check(alias, frozen_house))
            print(json.dumps(checks[-1]), flush=True)
    overlap_bad = overlap_failures(checks)
    for line in overlap_bad:
        print("FAILED overlap", line, flush=True)
    if not args.check:
        status["overlap_failed"] = overlap_bad
        status_path.write_text(json.dumps(status, indent=1), encoding="utf-8")
    failed = [a for a, r in status["runs"].items() if not r.get("ok")]
    print("failed:", failed, "| seconds:", status.get("seconds"), flush=True)
    return 1 if failed or overlap_bad or status.get("bench_error") else 0


if __name__ == "__main__":
    raise SystemExit(main())
