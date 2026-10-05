"""SPEC_FROZEN.md 2.4: the four BTAL TAA sleeves re-run at HEAD with synthetic TQQQ / BTAL before 2012.

Reuses the validated synthetic instruments of growth_shelf_v2_20260926 unchanged (file hashes are written to the
ledger) and its loader patch: `load_price_timeseries` is swapped, in every loaded module, for a wrapper that serves
TQQQ and BTAL from the synthetic bars; every other symbol still comes from Norgate. Strategy code and parameters are
untouched.

Modes per sleeve:
  real             unpatched, the module's own config (parity: must equal this study's sleeve inventory run)
  syn_scaled       synthetic TQQQ + scaled synthetic BTAL for the whole history, start 2006 (A3 check)
  syn_unscaled     same with the unscaled BTAL replica (A3 bracket)
  splice_scaled    synthetic only before each fund's first real bar, start 2006 (the LONG-window series, main)
  splice_unscaled  same with the unscaled replica (sensitivity)

Outputs: results/research/portfolio/shelf_rebuild_20260929/proxy_runs/<mode>/.

Usage: python run_proxies.py
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import contextlib
from dataclasses import replace
import hashlib
import importlib
import io
import json
from pathlib import Path
import sys

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
GROWTH_V2_DIR = REPO / "scripts" / "research" / "growth_shelf_v2_20260926"
for path in (REPO, GROWTH_V2_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import proxy_runs as v2  # noqa: E402  (growth_shelf_v2: synthetic bars, splice rule, loader patch)

STUDY_DIR = REPO / "results" / "research" / "portfolio" / "shelf_rebuild_20260929"
RUN_DIR = STUDY_DIR / "proxy_runs"
END_STR = "2026-08-19"
PROXY_START_STR = "2006-01-01"
CUT = pd.Timestamp("2012-10-02")
SLEEVE_DICT = {
    "taa3x": ("strategies.taa_df.strategy_taa_df_btal_fallback_tqqq_vix_cash", "standard"),
    "taa3x_1n": ("strategies.taa_df.strategy_taa_df_btal_1n_fallback_tqqq_vix_cash", "standard"),
    "taa2x_1n": ("strategies.taa_df.strategy_taa_df_btal_1n_fallback_qld_vix_cash", "standard"),
    "btal_qqq": ("strategies.taa_df.strategy_taa_df_btal_linearity_1n_fallback_qqq_vix_cash", "linearity"),
}
MODE_LIST = ["real", "syn_scaled", "syn_unscaled", "splice_scaled", "splice_unscaled"]


def install_patch(mode_str: str) -> None:
    import data.norgate_loader as loader

    original_fn = loader.load_price_timeseries
    patched = v2.make_patched_loader(original_fn, mode_str)
    for module_obj in list(sys.modules.values()):
        if getattr(module_obj, "load_price_timeseries", None) is original_fn:
            module_obj.load_price_timeseries = patched


def run_one(alias_str: str, mode_str: str) -> dict:
    from scripts.research import run_ladder4_candidate_value_add_study as ladder_runner

    module_str, kind_str = SLEEVE_DICT[alias_str]
    module = importlib.import_module(module_str)
    utils = importlib.import_module("strategies.taa_df.strategy_taa_df_fallback_vix_cash_variant_utils")
    if mode_str != "real":
        install_patch(mode_str)
    config = replace(module.DEFAULT_CONFIG, end_date_str=END_STR)
    if mode_str != "real":
        config = replace(config, start_date_str=PROXY_START_STR)
    name_str = module_str.rsplit(".", 1)[1]
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        if kind_str == "standard":
            loader_fn = importlib.import_module("strategies.taa_df.strategy_taa_df").get_defense_first_data
            strategy = utils.run_standard_fallback_vix_cash_variant(
                strategy_name_str=name_str, config=config, base_data_loader_fn=loader_fn,
                show_display_bool=False, save_results_bool=False, capital_base_float=1_000_000.0)
        else:
            loader_fn = importlib.import_module(
                "strategies.taa_df.strategy_taa_df_btal_linearity_1n").get_defense_first_linearity_1n_data
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
    return {"alias": alias_str, "mode": mode_str, "first": str(path_df.index[0].date()),
            "last": str(path_df.index[-1].date()), "fills": int(len(tx_df))}


def a3_row(alias_str: str, mode_str: str) -> dict:
    """A3: a sleeve run on synthetic instruments for its whole history vs the real run, 2012-10-02 -> end."""
    def nav(mode: str) -> pd.Series:
        return pd.read_csv(RUN_DIR / mode / f"{alias_str}__path.csv.gz", index_col="date",
                           parse_dates=True)["total_value_float"]

    both = pd.concat([nav(mode_str).pct_change(), nav("real").pct_change()], axis=1,
                     keys=["syn", "real"]).loc[CUT:END_STR].dropna()
    monthly = (1 + both).resample("ME").prod() - 1
    years = (both.index[-1] - both.index[0]).days / 365.25

    def cagr(r: pd.Series) -> float:
        return float((1 + r).prod() ** (1 / years) - 1)

    def maxdd(r: pd.Series) -> float:
        v = (1 + r).cumprod()
        return float((v / v.cummax() - 1).min())

    row = {"alias": alias_str, "mode": mode_str, "monthly_corr": float(monthly.corr().iloc[0, 1]),
           "daily_corr": float(both.corr().iloc[0, 1]), "cagr_syn": cagr(both["syn"]), "cagr_real": cagr(both["real"]),
           "maxdd_syn": maxdd(both["syn"]), "maxdd_real": maxdd(both["real"])}
    row["cagr_diff_pp"] = 100 * (row["cagr_syn"] - row["cagr_real"])
    row["maxdd_diff_pp"] = 100 * (row["maxdd_syn"] - row["maxdd_real"])
    row["pass"] = bool(row["monthly_corr"] >= 0.90 and abs(row["cagr_diff_pp"]) <= 2.0
                       and abs(row["maxdd_diff_pp"]) <= 5.0)
    return row


def main() -> int:
    from scripts.research import run_ladder4_candidate_value_add_study as ladder_runner

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    ladder_runner.append_jsonl(STUDY_DIR / "experiment_ledger.jsonl", {
        "event_str": "proxy_runs_started", "recorded_at_utc_str": ladder_runner.utc_now_str(),
        "synthetic_file_sha256_dict": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                       for p in sorted(v2.PROXY_DIR.glob("synthetic_*bars*.csv.gz"))}})
    job_list = [(a, m) for a in SLEEVE_DICT for m in MODE_LIST]
    # *** CRITICAL*** one job per worker process: the patch rewrites module globals, and a reused worker would
    # leak one mode's synthetic data into the next job.
    with ProcessPoolExecutor(max_workers=5, max_tasks_per_child=1) as pool:
        for result in pool.map(run_one, *zip(*job_list)):
            print(json.dumps(result), flush=True)
    a3_df = pd.DataFrame([a3_row(a, m) for a in SLEEVE_DICT for m in ("syn_scaled", "syn_unscaled")])
    a3_df.to_csv(RUN_DIR / "a3_strategy_validation.csv", index=False, float_format="%.6g")
    pd.set_option("display.width", 250)
    print(a3_df.round(4).to_string(index=False))
    ladder_runner.append_jsonl(STUDY_DIR / "experiment_ledger.jsonl", {
        "event_str": "proxy_runs_finished", "recorded_at_utc_str": ladder_runner.utc_now_str(),
        "a3_rows": a3_df.to_dict(orient="records")})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
