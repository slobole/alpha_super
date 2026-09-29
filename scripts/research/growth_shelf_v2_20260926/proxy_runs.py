"""Parts A3 + A4 of SPEC_FROZEN.md: the BTAL TAA sleeves through the real engine with synthetic TQQQ / BTAL.

Modes (per sleeve):
  real              unpatched, the strategy's own config (sanity: must equal the fund-menu inventory run)
  syn_scaled        synthetic TQQQ and the scaled synthetic BTAL for the WHOLE history, start 2006 (A3, main)
  syn_unscaled      same with the unscaled BTAL replica (A3, sensitivity; amendment A3)
  splice_scaled     synthetic only before each fund's first bar, real bars after, start 2006 (A4: the tables)
  splice_unscaled   same with the unscaled replica (sensitivity)

The patch swaps `load_price_timeseries` in every loaded module for a wrapper that serves TQQQ and BTAL from the
synthetic bars (proxy_instruments.py); every other symbol still comes from Norgate unchanged. The strategies' own
code and parameters are untouched. Outputs: results/research/portfolio/growth_shelf_v2_20260926/proxy_runs/.

Usage: python proxy_runs.py
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import contextlib
from dataclasses import replace
import importlib
import io
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
for path in (REPO, REPO / "scripts" / "research" / "fund_menu_20260923"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

PROXY_DIR = REPO / "results" / "research" / "portfolio" / "growth_shelf_v2_20260926" / "proxy"
RUN_DIR = REPO / "results" / "research" / "portfolio" / "growth_shelf_v2_20260926" / "proxy_runs"
END_STR = "2026-08-19"
PROXY_START_STR = "2006-01-01"
CUT = pd.Timestamp("2012-10-02")
SLEEVE_DICT = {
    "taa_btal_tqqq": ("strategies.taa_df.strategy_taa_df_btal_fallback_tqqq_vix_cash", "strategy_taa_df_btal_fallback_tqqq_vix_cash", "standard"),
    "taa_btal_1n_tqqq": ("strategies.taa_df.strategy_taa_df_btal_1n_fallback_tqqq_vix_cash", "strategy_taa_df_btal_1n_fallback_tqqq_vix_cash", "standard"),
    "taa_btal_lin_qqq": ("strategies.taa_df.strategy_taa_df_btal_linearity_1n_fallback_qqq_vix_cash", "strategy_taa_df_btal_linearity_1n_fallback_qqq_vix_cash", "linearity"),
}
MODE_LIST = ["real", "syn_scaled", "syn_unscaled", "splice_scaled", "splice_unscaled"]


def synthetic_bars(symbol_str: str, btal_label_str: str) -> pd.DataFrame:
    file_str = "synthetic_TQQQ_bars.csv.gz" if symbol_str == "TQQQ" else f"synthetic_BTAL_bars_{btal_label_str}.csv.gz"
    frame = pd.read_csv(PROXY_DIR / file_str, index_col=0, parse_dates=True)
    frame.index.name = "Date"
    return frame


def splice(real_df: pd.DataFrame, syn_df: pd.DataFrame) -> pd.DataFrame:
    """Synthetic rows strictly before the real fund's first bar, rescaled so the synthetic close on that first bar
    equals the real close; real rows unchanged from the first bar on.
    *** CRITICAL*** real data is never altered; the synthetic part only fills dates the fund did not exist."""
    first_ts = real_df.index[0]
    scale = float(real_df["Close"].iloc[0] / syn_df.loc[first_ts, "Close"])
    pre_df = syn_df.loc[syn_df.index < first_ts].copy()
    for field in ("Open", "High", "Low", "Close", "Unadjusted Close"):
        pre_df[field] = pre_df[field] * scale
    pre_df["Turnover"] = pre_df["Close"] * pre_df["Volume"]
    pre_df = pre_df.reindex(columns=real_df.columns)
    return pd.concat([pre_df, real_df])


def make_patched_loader(original_fn, mode_str: str, bars_fn=synthetic_bars):
    """Wrapper for load_price_timeseries: TQQQ / BTAL from synthetic bars (whole history for syn_* modes, only before
    the fund's first real bar for splice_* modes); every other symbol passes through unchanged."""
    btal_label_str = "unscaled" if mode_str.endswith("unscaled") else "scaled"
    full_bool = mode_str.startswith("syn_")
    cache: dict = {}

    def patched(symbol_str, *args, **kwargs):
        if symbol_str not in ("TQQQ", "BTAL"):
            return original_fn(symbol_str, *args, **kwargs)
        start_str, end_str = kwargs.get("start_date_str"), kwargs.get("end_date_str")
        key = (symbol_str, kwargs.get("adjustment_str"))
        if key not in cache:
            syn_df = bars_fn(symbol_str, btal_label_str)
            if full_bool:
                cache[key] = syn_df
            else:
                # Norgate returns an empty frame when start_date is None, so ask for the full history explicitly.
                real_df = original_fn(symbol_str, *args, **{**kwargs, "start_date_str": "1990-01-01", "end_date_str": None})
                real_df.index = pd.to_datetime(real_df.index).normalize()
                cache[key] = splice(real_df, syn_df)
        out_df = cache[key]
        if start_str is not None:
            out_df = out_df.loc[pd.Timestamp(start_str):]
        if end_str is not None:
            out_df = out_df.loc[: pd.Timestamp(end_str)]
        return out_df.copy()

    return patched


def install_patch(mode_str: str) -> None:
    import data.norgate_loader as loader

    original_fn = loader.load_price_timeseries
    patched = make_patched_loader(original_fn, mode_str)
    for module_obj in list(sys.modules.values()):
        if getattr(module_obj, "load_price_timeseries", None) is original_fn:
            module_obj.load_price_timeseries = patched


def run_one(alias_str: str, mode_str: str) -> dict:
    from scripts.research import run_ladder4_candidate_value_add_study as ladder_runner

    module_str, name_str, kind_str = SLEEVE_DICT[alias_str]
    module = importlib.import_module(module_str)
    utils = importlib.import_module("strategies.taa_df.strategy_taa_df_fallback_vix_cash_variant_utils")
    if mode_str != "real":
        install_patch(mode_str)
    config = replace(module.DEFAULT_CONFIG, end_date_str=END_STR)
    if mode_str != "real":
        config = replace(config, start_date_str=PROXY_START_STR)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        if kind_str == "standard":
            loader_fn = importlib.import_module("strategies.taa_df.strategy_taa_df").get_defense_first_data
            strategy = utils.run_standard_fallback_vix_cash_variant(
                strategy_name_str=name_str, config=config, base_data_loader_fn=loader_fn,
                show_display_bool=False, save_results_bool=False, capital_base_float=1_000_000.0)
        else:
            loader_fn = importlib.import_module("strategies.taa_df.strategy_taa_df_btal_linearity_1n").get_defense_first_linearity_1n_data
            strategy = utils.run_linearity_1n_fallback_vix_cash_variant(
                strategy_name_str=name_str, config=config, base_data_loader_fn=loader_fn,
                show_display_bool=False, save_results_bool=False, capital_base_float=1_000_000.0)
    out_dir = RUN_DIR / mode_str
    out_dir.mkdir(parents=True, exist_ok=True)
    path_df = ladder_runner.extract_source_result_df(strategy)
    tx_df = ladder_runner.extract_source_transaction_df(strategy, alias_str)
    ladder_runner.write_csv_gzip(path_df, out_dir / f"{alias_str}__path.csv.gz", index_bool=True, index_label_str="date")
    ladder_runner.write_csv_gzip(tx_df, out_dir / f"{alias_str}__transactions.csv.gz", index_bool=False)
    weight_df = strategy.rebalance_weight_df.copy()
    weight_df.index = pd.to_datetime(weight_df.index).normalize()
    weight_df.to_csv(out_dir / f"{alias_str}__rebalance_weights.csv")
    return {"alias": alias_str, "mode": mode_str, "first": str(path_df.index[0].date()), "last": str(path_df.index[-1].date()),
            "fills": int(len(tx_df))}


def compare(alias_str: str, mode_str: str, inventory_ser: pd.Series) -> dict:
    """A3: a sleeve run on synthetic instruments vs the real run on 2012-10-02 -> end."""
    nav = pd.read_csv(RUN_DIR / mode_str / f"{alias_str}__path.csv.gz", index_col="date", parse_dates=True)["total_value_float"]
    real_nav = pd.read_csv(RUN_DIR / "real" / f"{alias_str}__path.csv.gz", index_col="date", parse_dates=True)["total_value_float"]
    both = pd.concat([nav.pct_change(), real_nav.pct_change()], axis=1, keys=["syn", "real"]).loc[CUT:END_STR].dropna()
    monthly = (1 + both).resample("ME").prod() - 1
    years = (both.index[-1] - both.index[0]).days / 365.25

    def cagr(r):
        return float((1 + r).prod() ** (1 / years) - 1)

    def maxdd(r):
        v = (1 + r).cumprod()
        return float((v / v.cummax() - 1).min())

    w_syn = pd.read_csv(RUN_DIR / mode_str / f"{alias_str}__rebalance_weights.csv", index_col=0, parse_dates=True)
    w_real = pd.read_csv(RUN_DIR / "real" / f"{alias_str}__rebalance_weights.csv", index_col=0, parse_dates=True)
    common_idx = w_syn.index.intersection(w_real.index)
    common_idx = common_idx[common_idx >= CUT]
    columns = sorted(set(w_syn.columns) | set(w_real.columns))
    overlap = 1 - 0.5 * (w_syn.reindex(index=common_idx, columns=columns).fillna(0) - w_real.reindex(index=common_idx, columns=columns).fillna(0)).abs().sum(axis=1)
    inv = inventory_ser.loc[CUT:END_STR].dropna()
    real_vs_inventory = float((real_nav.pct_change().reindex(inv.index) - inv).abs().max())
    row = {"alias": alias_str, "mode": mode_str, "monthly_corr": float(monthly.corr().iloc[0, 1]), "daily_corr": float(both.corr().iloc[0, 1]),
           "cagr_syn": cagr(both["syn"]), "cagr_real": cagr(both["real"]), "maxdd_syn": maxdd(both["syn"]), "maxdd_real": maxdd(both["real"]),
           "allocation_overlap_mean": float(overlap.mean()), "months_identical_share": float((overlap > 0.999).mean()),
           "real_run_vs_inventory_max_abs_daily_diff": real_vs_inventory}
    row["cagr_diff_pp"] = 100 * (row["cagr_syn"] - row["cagr_real"])
    row["maxdd_diff_pp"] = 100 * (row["maxdd_syn"] - row["maxdd_real"])
    row["pass"] = bool(row["monthly_corr"] >= 0.90 and abs(row["cagr_diff_pp"]) <= 2.0 and abs(row["maxdd_diff_pp"]) <= 5.0)
    return row


def main() -> int:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    job_list = [(a, m) for a in SLEEVE_DICT for m in MODE_LIST]
    # *** CRITICAL*** one job per worker process: the patch rewrites module globals, and a reused worker would
    # leak one mode's synthetic data into the next job (seen in the first run: a "real" run was contaminated).
    with ProcessPoolExecutor(max_workers=6, max_tasks_per_child=1) as pool:
        for result in pool.map(run_one, *zip(*job_list)):
            print(json.dumps(result))
    import common  # noqa: PLC0415 (fund menu study helpers)
    inventory_df = pd.read_csv(common.STUDY_DIR_PATH / "inventory" / "sleeve_returns.csv.gz", index_col=0, parse_dates=True)
    row_list = [compare(a, m, inventory_df[a]) for a in SLEEVE_DICT for m in ("syn_scaled", "syn_unscaled")]
    result_df = pd.DataFrame(row_list)
    result_df.to_csv(RUN_DIR / "a3_strategy_validation.csv", index=False, float_format="%.6g")
    pd.set_option("display.width", 250)
    print(result_df.round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
