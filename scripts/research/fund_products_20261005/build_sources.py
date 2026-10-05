"""Fresh standalone runs of the new sleeves at HEAD (SPEC_FROZEN.md section 1), in the shelf-rebuild source format.

Each sleeve runs once through its own ``run_variant`` at the same reference capital as the shelf-rebuild sleeves
($1M), from its module's default start to the latest session, from this clean worktree. The study's frames cut the
series at the common END (2026-08-19); the sessions after it feed the "after the window" appendix only.

Sleeves:
    ndx_atr_cap, ndx_natr_cap   the two pods of the momentum capsule (E2 with the 40% sector cap)
    dv2_g, hpi_g                the two pods of the MR capsule, idle cash in BIL (the PM_READY Bench entry points)
    dv2_g_cash, hpi_g_cash      the same pods with parking disabled (idle cash at 0%), sensitivity only
    plus any shelf-rebuild alias given with --only (taa3x, core5, ...), re-run to the latest session.

Research authority only. No strategy, engine or live file is modified.

Usage: python build_sources.py [--only alias ...] [--workers 1]
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import contextlib
import gc
import importlib
import inspect
import json
from pathlib import Path
import subprocess
import sys
import time

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO_ROOT_PATH = HERE.parents[2]
if str(REPO_ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_PATH))

from alpha import strategy_registry  # noqa: E402
from scripts.research import run_ladder4_candidate_value_add_study as ladder_runner  # noqa: E402

STUDY_DIR_PATH = REPO_ROOT_PATH / "results" / "research" / "portfolio" / "fund_products_20261005"
SOURCE_DIR_PATH = STUDY_DIR_PATH / "sources"
LOG_DIR_PATH = STUDY_DIR_PATH / "source_logs"
REFERENCE_CAPITAL_FLOAT = 1_000_000.0
LATEST_END_DATE_STR = "2026-10-02"

# alias -> (import string, runner). "variant" = the module's run_variant; "dv2_cash" / "hpi_cash" = the capsule pod
# builders with parking disabled (no Bench entry point exists for that reference).
NEW_SLEEVE_DICT: dict[str, tuple[str, str]] = {
    "ndx_atr_cap": ("strategies.momentum.strategy_mo_atr_normalized_ndx_vxn_scaled_sector_cap"
                    ":SectorCapVxnScaledAtrNormalizedNdxStrategy", "variant"),
    "ndx_natr_cap": ("strategies.momentum.strategy_mo_natr20_ndx_vxn_scaled_sector_cap"
                     ":SectorCapNatr20VxnScaledNdxStrategy", "variant"),
    "dv2_g": ("strategies.mr_capsule.strategy_mr_dv2_vix_gated_bil", "variant"),
    "hpi_g": ("strategies.mr_capsule.strategy_mr_hpi_vote_vix_gated_bil", "variant"),
    "dv2_g_cash": ("strategies.mr_capsule.dv2_vix_gated", "dv2_cash"),
    "hpi_g_cash": ("strategies.mr_capsule.hpi_vote_vix_gated", "hpi_cash"),
}
# Shelf-rebuild sleeves that may be re-run to the latest session (same import strings as run_sleeves.SLEEVE_DICT).
OLD_SLEEVE_DICT: dict[str, tuple[str, str]] = {
    "taa3x": ("strategies.taa_df.strategy_taa_df_btal_fallback_tqqq_vix_cash", "variant"),
    "taa3x_1n": ("strategies.taa_df.strategy_taa_df_btal_1n_fallback_tqqq_vix_cash", "variant"),
    "core5": ("strategies.taa_beyond_6040.strategy_taa_adaptive_macro_core5", "variant"),
    "btal_qqq": ("strategies.taa_df.strategy_taa_df_btal_linearity_1n_fallback_qqq_vix_cash", "variant"),
    "ndx_vxn": ("strategies.momentum.strategy_mo_atr_normalized_ndx_vxn_scaled:VxnScaledAtrNormalizedNdxStrategy", "variant"),
    "dv2": ("strategies.dv2.strategy_mr_dv2:DVO2Strategy", "variant"),
    "hpi_vote": ("strategies.hpi.strategy_mr_hpi_sp500_2_3_5_vote", "variant"),
}
SLEEVE_DICT = {**NEW_SLEEVE_DICT, **OLD_SLEEVE_DICT}
SHELF_REQUESTED_START_STR = "2000-01-03"   # the shelf rebuild's requested start for its sleeves


def git_state_dict() -> dict:
    def run_git(arg_list: list[str]) -> str:
        return subprocess.run(["git", *arg_list], cwd=REPO_ROOT_PATH, capture_output=True, text=True, check=False).stdout.strip()

    return {"head_commit_str": run_git(["rev-parse", "HEAD"]),
            "dirty_path_list": [line for line in run_git(["status", "--porcelain"]).splitlines() if line]}


def run_strategy(alias_str: str):
    strategy_import_str, runner_str = SLEEVE_DICT[alias_str]
    module_obj = importlib.import_module(strategy_import_str.split(":", maxsplit=1)[0])
    common_kwargs = dict(show_display_bool=False, save_results_bool=False,
                         output_dir_str=str(STUDY_DIR_PATH / "scratch_unused"),
                         capital_base_float=REFERENCE_CAPITAL_FLOAT, end_date_str=LATEST_END_DATE_STR)
    if runner_str == "dv2_cash":
        return module_obj.run_dv2_capsule_pod(strategy_name_str="strategy_mr_dv2_vix_gated_cash",
                                              parking_enabled_bool=False, spmo_parking_enabled_bool=False, **common_kwargs), "module default"
    if runner_str == "hpi_cash":
        return module_obj.run_hpi_capsule_pod(strategy_name_str="strategy_mr_hpi_vote_vix_gated_cash",
                                              parking_enabled_bool=False, spmo_parking_enabled_bool=False, **common_kwargs), "module default"
    run_variant_fn = module_obj.run_variant
    if alias_str in OLD_SLEEVE_DICT:
        # Same start rule as the shelf rebuild: the common requested start, module default as the fallback.
        try:
            return run_variant_fn(backtest_start_date_str=SHELF_REQUESTED_START_STR, **common_kwargs), SHELF_REQUESTED_START_STR
        except Exception as first_exc:  # noqa: BLE001 - data may not reach the requested start
            default_start_obj = inspect.signature(run_variant_fn).parameters["backtest_start_date_str"].default
            print(f"Requested start failed ({first_exc!r}); retrying with module default {default_start_obj!r}.")
            return run_variant_fn(backtest_start_date_str=default_start_obj, **common_kwargs), str(default_start_obj)
    # New sleeves keep their module's own default start (the capsule's gate and the E2 book were built on it).
    return run_variant_fn(**common_kwargs), "module default"


def run_one_sleeve(alias_str: str) -> dict:
    SOURCE_DIR_PATH.mkdir(parents=True, exist_ok=True)
    LOG_DIR_PATH.mkdir(parents=True, exist_ok=True)
    started_float = time.time()
    strategy_import_str, runner_str = SLEEVE_DICT[alias_str]
    with (LOG_DIR_PATH / f"{alias_str}.log").open("w", encoding="utf-8") as log_file_obj, \
            contextlib.redirect_stdout(log_file_obj), contextlib.redirect_stderr(log_file_obj):
        strategy_obj, start_used_str = run_strategy(alias_str)
    source_result_df = ladder_runner.extract_source_result_df(strategy_obj)
    transaction_df = ladder_runner.extract_source_transaction_df(strategy_obj, alias_str)
    if source_result_df.index[-1] != pd.Timestamp(LATEST_END_DATE_STR):
        raise RuntimeError(f"{alias_str} ended {source_result_df.index[-1].date()}, not {LATEST_END_DATE_STR}.")
    invested_mask_ser = source_result_df["portfolio_value_float"].abs() > 1e-9
    path_file_path = SOURCE_DIR_PATH / f"{alias_str}__path.csv.gz"
    transaction_file_path = SOURCE_DIR_PATH / f"{alias_str}__transactions.csv.gz"
    ladder_runner.write_csv_gzip(source_result_df, path_file_path, index_bool=True, index_label_str="date")
    ladder_runner.write_csv_gzip(transaction_df, transaction_file_path, index_bool=False)
    accounting_policy_dict = dict(getattr(strategy_obj, "_accounting_policy_dict", {}))
    cash_ser, nav_ser = source_result_df["cash_float"], source_result_df["total_value_float"]
    module_path = Path(importlib.import_module(strategy_import_str.split(":", maxsplit=1)[0]).__file__).resolve()
    tier_str = (strategy_registry.tier_label_for(strategy_import_str) if runner_str == "variant" else "research")
    metadata_dict = {
        "alias_str": alias_str, "strategy_import_str": strategy_import_str, "runner_str": runner_str, "tier_str": tier_str,
        "strategy_name_str": str(strategy_obj.name), "reference_capital_float": REFERENCE_CAPITAL_FLOAT,
        "start_used_str": start_used_str, "engine_first_date_str": source_result_df.index[0].date().isoformat(),
        "first_invested_date_str": invested_mask_ser[invested_mask_ser].index[0].date().isoformat(),
        "end_date_str": source_result_df.index[-1].date().isoformat(),
        "observation_count_int": int(len(source_result_df)), "transaction_count_int": int(len(transaction_df)),
        "gross_transaction_notional_float": float(transaction_df["signed_notional_float"].abs().sum()),
        "total_commission_float": float(transaction_df["commission_float"].sum()),
        "negative_cash_day_count_int": int((cash_ser < 0.0).sum()),
        "minimum_cash_nav_weight_float": float((cash_ser / nav_ser).min()),
        "mean_cash_nav_weight_float": float((cash_ser / nav_ser).mean()),
        "positive_cash_rate_policy_str": str(accounting_policy_dict.get("positive_cash_rate_policy_str", "missing")),
        "negative_cash_financing_policy_str": str(accounting_policy_dict.get("negative_cash_financing_policy_str", "missing")),
        "module_path_str": module_path.relative_to(REPO_ROOT_PATH).as_posix(),
        "module_sha256_str": ladder_runner.sha256_file_str(module_path),
        "path_sha256_str": ladder_runner.sha256_file_str(path_file_path),
        "transaction_sha256_str": ladder_runner.sha256_file_str(transaction_file_path),
        "runtime_seconds_float": round(time.time() - started_float, 1),
    }
    ladder_runner.write_json(SOURCE_DIR_PATH / f"{alias_str}__metadata.json", metadata_dict)
    del strategy_obj
    gc.collect()
    return metadata_dict


def main(arg_list: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--only", nargs="*", default=None)
    args = parser.parse_args(arg_list)
    alias_list = args.only if args.only else list(NEW_SLEEVE_DICT)
    STUDY_DIR_PATH.mkdir(parents=True, exist_ok=True)
    ledger_path = STUDY_DIR_PATH / "experiment_ledger.jsonl"
    vintage_start_dict = ladder_runner.norgate_database_vintage_dict()
    ladder_runner.append_jsonl(ledger_path, {
        "event_str": "source_runs_started", "recorded_at_utc_str": ladder_runner.utc_now_str(), "alias_list": alias_list,
        "end_date_str": LATEST_END_DATE_STR, "reference_capital_float": REFERENCE_CAPITAL_FLOAT,
        "git_state_dict": git_state_dict(), "norgate_vintage_dict": vintage_start_dict})
    failure_list = []
    with ProcessPoolExecutor(max_workers=args.workers, max_tasks_per_child=1) as executor_obj:
        future_map = {executor_obj.submit(run_one_sleeve, alias_str): alias_str for alias_str in alias_list}
        for future_obj in as_completed(future_map):
            alias_str = future_map[future_obj]
            try:
                row_dict = future_obj.result()
                print(f"done {alias_str:<14} first_invested={row_dict['first_invested_date_str']} "
                      f"obs={row_dict['observation_count_int']} {row_dict['runtime_seconds_float']}s", flush=True)
                ladder_runner.append_jsonl(ledger_path, {"event_str": "source_run_completed",
                                                         "recorded_at_utc_str": ladder_runner.utc_now_str(), **row_dict})
            except Exception as exc:  # noqa: BLE001 - record and continue; the exit code fails the run
                failure_list.append({"alias_str": alias_str, "error_str": repr(exc)})
                print(f"FAILED {alias_str}: {exc!r}", flush=True)
    vintage_end_dict = ladder_runner.norgate_database_vintage_dict()
    ladder_runner.append_jsonl(ledger_path, {
        "event_str": "source_runs_finished", "recorded_at_utc_str": ladder_runner.utc_now_str(),
        "failure_list": failure_list, "norgate_vintage_changed_bool": vintage_end_dict != vintage_start_dict})
    print(json.dumps({"failed": failure_list, "norgate_vintage_changed": vintage_end_dict != vintage_start_dict}, indent=2))
    return 1 if failure_list or vintage_end_dict != vintage_start_dict else 0


if __name__ == "__main__":
    raise SystemExit(main())
