"""Fresh standalone run of every sleeve in SPEC_FROZEN.md section 2.2 at the current HEAD.

Each sleeve runs once through its own ``run_variant`` at the same reference capital, from the earliest date its
data allows (requested 2000-01-03, module default as fallback) to 2026-08-19, on one Norgate database vintage.
Nothing here combines sleeves; the output (daily NAV, invested value, cash, fills, lineage) is the raw material
for books.py.

Research authority only. No strategy, engine or live file is modified.

Usage: python run_sleeves.py [--workers 5] [--only alias ...]
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import contextlib
import gc
import hashlib
import importlib
import inspect
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO_ROOT_PATH = HERE.parents[2]
if str(REPO_ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_PATH))

from alpha import strategy_registry  # noqa: E402
from scripts.research import run_ladder4_candidate_value_add_study as ladder_runner  # noqa: E402

STUDY_DIR_PATH = REPO_ROOT_PATH / "results" / "research" / "portfolio" / "shelf_rebuild_20260929"
SOURCE_DIR_PATH = STUDY_DIR_PATH / "sources"
LOG_DIR_PATH = STUDY_DIR_PATH / "source_logs"
SPEC_PATH = HERE / "SPEC_FROZEN.md"

REFERENCE_CAPITAL_FLOAT = 1_000_000.0
REQUESTED_START_DATE_STR = "2000-01-03"
COMMON_END_DATE_STR = "2026-08-19"

# alias -> (import string, extra run_variant keyword arguments). The registry's PM_READY + WIRED set must be
# covered exactly (checked at start); the shadow candidates are the only extras allowed.
SLEEVE_DICT: dict[str, tuple[str, dict]] = {
    "core5": ("strategies.taa_beyond_6040.strategy_taa_adaptive_macro_core5", {}),
    "btal_qqq": ("strategies.taa_df.strategy_taa_df_btal_linearity_1n_fallback_qqq_vix_cash", {}),
    "tactical_fi": ("strategies.taa_beyond_6040.strategy_taa_tactical_fixed_income_ief_lqd",
                    {"fred_data_mode_str": "alfred_point_in_time", "alfred_vintage_policy_str": "decision_date",
                     "stale_input_policy_str": "block_and_hold"}),
    "trinity": ("strategies.taa_beyond_6040.strategy_taa_trinity_vol_control_8_bil", {}),
    "eom_flow": ("strategies.taa_beyond_6040.strategy_taa_month_end_rebalancing_flow", {}),
    "downshock": ("strategies.mean_reversion.strategy_mr_us_sector_etf_ibs_downshock_vox_iyr", {}),
    "disp": ("strategies.mean_reversion.strategy_mr_sector_dispersion_ibs_kie_ihi_asset_sma200", {}),
    "disp_xlc": ("strategies.mean_reversion.strategy_mr_sector_dispersion_ibs_kie_ihi_xlc", {}),
    "disp_xlc_sma": ("strategies.mean_reversion.strategy_mr_sector_dispersion_ibs_kie_ihi_xlc_asset_sma200", {}),
    "taa_lin_qqq": ("strategies.taa_df.strategy_taa_df_linearity_1n_fallback_qqq_vix_cash", {}),
    "taa3x": ("strategies.taa_df.strategy_taa_df_btal_fallback_tqqq_vix_cash", {}),
    "taa3x_1n": ("strategies.taa_df.strategy_taa_df_btal_1n_fallback_tqqq_vix_cash", {}),
    "taa2x_1n": ("strategies.taa_df.strategy_taa_df_btal_1n_fallback_qld_vix_cash", {}),
    "taa_1n_qld": ("strategies.taa_df.strategy_taa_df_1n_fallback_qld_vix_cash", {}),
    "taa_1n_sso": ("strategies.taa_df.strategy_taa_df_1n_fallback_sso_vix_cash", {}),
    "ndx_vxn": ("strategies.momentum.strategy_mo_atr_normalized_ndx_vxn_scaled:VxnScaledAtrNormalizedNdxStrategy", {}),
    "ndx_atr": ("strategies.momentum.strategy_mo_atr_normalized_ndx:AtrNormalizedNdxStrategy", {}),
    "ndx_natr20": ("strategies.momentum.strategy_mo_natr20_ndx_vxn_scaled", {}),
    "compass_qqq": ("strategies.taa_df.strategy_taa_inflation_compass_qqq", {}),
    "compass": ("strategies.taa_df.strategy_taa_inflation_compass", {}),
    "dv2": ("strategies.dv2.strategy_mr_dv2:DVO2Strategy", {}),
    "dv2_adv": ("strategies.dv2.strategy_mr_dv2_liquidity_floor_adv_rank", {}),
    "dv2_floor": ("strategies.dv2.strategy_mr_dv2_liquidity_floor", {}),
    "hpi_vote": ("strategies.hpi.strategy_mr_hpi_sp500_2_3_5_vote", {}),
    "hpi_ibs_rsi": ("strategies.hpi.strategy_mr_hpi_sp500_ibs_rsi_exit", {}),
    "etf_dv2": ("strategies.dv2.strategy_mr_dv2_industry_etf", {}),
    # Sensitivity only (SPEC 9): Tactical FI in its governed frozen current-vintage FRED mode.
    "tactical_fi_frozen": ("strategies.taa_beyond_6040.strategy_taa_tactical_fixed_income_ief_lqd", {}),
}
SHADOW_ALIAS_SET = {"ndx_natr20", "dv2_adv", "dv2_floor", "etf_dv2"}
SENSITIVITY_ALIAS_SET = {"tactical_fi_frozen"}

# Single-stock universes are slow and memory-heavy; start them first.
HEAVY_ALIAS_TUPLE = ("hpi_vote", "hpi_ibs_rsi", "dv2", "dv2_adv", "dv2_floor", "ndx_atr", "ndx_vxn", "ndx_natr20")


def git_state_dict() -> dict:
    def run_git(arg_list: list[str]) -> str:
        return subprocess.run(["git", *arg_list], cwd=REPO_ROOT_PATH, capture_output=True, text=True,
                              check=False).stdout.strip()

    return {"head_commit_str": run_git(["rev-parse", "HEAD"]),
            "dirty_path_list": [line for line in run_git(["status", "--porcelain"]).splitlines() if line]}


def spec_sha256_str() -> str:
    return hashlib.sha256(SPEC_PATH.read_bytes()).hexdigest()


def run_fresh_strategy(strategy_module_obj, extra_kwarg_dict: dict) -> tuple:
    """Run from the common requested start; fall back to the module's own default start."""
    run_variant_fn = getattr(strategy_module_obj, "run_variant")
    call_kwargs_dict = dict(show_display_bool=False, save_results_bool=False,
                            output_dir_str=str(STUDY_DIR_PATH / "scratch_unused"),
                            capital_base_float=REFERENCE_CAPITAL_FLOAT, end_date_str=COMMON_END_DATE_STR,
                            **extra_kwarg_dict)
    try:
        strategy_obj = run_variant_fn(backtest_start_date_str=REQUESTED_START_DATE_STR, **call_kwargs_dict)
        return strategy_obj, REQUESTED_START_DATE_STR, ""
    except Exception as first_exc:  # data may not reach the requested start
        default_start_obj = inspect.signature(run_variant_fn).parameters["backtest_start_date_str"].default
        print(f"Requested start failed ({first_exc!r}); retrying with module default {default_start_obj!r}.")
        traceback.print_exc()
        strategy_obj = run_variant_fn(backtest_start_date_str=default_start_obj, **call_kwargs_dict)
        return strategy_obj, str(default_start_obj), f"requested {REQUESTED_START_DATE_STR} failed: {first_exc!r}"


def run_one_sleeve(alias_str: str) -> dict:
    """Worker: run one sleeve, write its compact artifacts, return a summary row."""
    strategy_import_str, extra_kwarg_dict = SLEEVE_DICT[alias_str]
    SOURCE_DIR_PATH.mkdir(parents=True, exist_ok=True)
    LOG_DIR_PATH.mkdir(parents=True, exist_ok=True)
    started_float = time.time()
    module_import_str = strategy_import_str.split(":", maxsplit=1)[0]
    with (LOG_DIR_PATH / f"{alias_str}.log").open("w", encoding="utf-8") as log_file_obj, \
            contextlib.redirect_stdout(log_file_obj), contextlib.redirect_stderr(log_file_obj):
        strategy_module_obj = importlib.import_module(module_import_str)
        strategy_obj, requested_start_used_str, start_fallback_note_str = run_fresh_strategy(
            strategy_module_obj, extra_kwarg_dict)

    source_result_df = ladder_runner.extract_source_result_df(strategy_obj)
    transaction_df = ladder_runner.extract_source_transaction_df(strategy_obj, alias_str)
    if source_result_df.index[-1] != pd.Timestamp(COMMON_END_DATE_STR):
        raise RuntimeError(f"{alias_str} ended {source_result_df.index[-1].date()}, not {COMMON_END_DATE_STR}.")

    # First day the sleeve holds anything: leading all-cash warmup days are trimmed later, never filled.
    invested_mask_ser = source_result_df["portfolio_value_float"].abs() > 1e-9
    first_invested_date_str = (invested_mask_ser[invested_mask_ser].index[0].date().isoformat()
                               if invested_mask_ser.any() else None)

    path_file_path = SOURCE_DIR_PATH / f"{alias_str}__path.csv.gz"
    transaction_file_path = SOURCE_DIR_PATH / f"{alias_str}__transactions.csv.gz"
    ladder_runner.write_csv_gzip(source_result_df, path_file_path, index_bool=True, index_label_str="date")
    ladder_runner.write_csv_gzip(transaction_df, transaction_file_path, index_bool=False)

    accounting_policy_dict = dict(getattr(strategy_obj, "_accounting_policy_dict", {}))
    data_adjustment_policy_dict = dict(getattr(strategy_obj, "_data_adjustment_policy_dict", {}))
    cash_ser = source_result_df["cash_float"]
    nav_ser = source_result_df["total_value_float"]
    module_path = Path(strategy_module_obj.__file__).resolve()
    metadata_dict = {
        "alias_str": alias_str,
        "strategy_import_str": strategy_import_str,
        "extra_kwarg_dict": extra_kwarg_dict,
        "tier_str": ("shadow" if alias_str in SHADOW_ALIAS_SET
                     else strategy_registry.tier_label_for(strategy_import_str)),
        "strategy_name_str": str(strategy_obj.name),
        "reference_capital_float": REFERENCE_CAPITAL_FLOAT,
        "requested_start_date_str": requested_start_used_str,
        "start_fallback_note_str": start_fallback_note_str,
        "engine_first_date_str": source_result_df.index[0].date().isoformat(),
        "first_invested_date_str": first_invested_date_str,
        "end_date_str": source_result_df.index[-1].date().isoformat(),
        "observation_count_int": int(len(source_result_df)),
        "transaction_count_int": int(len(transaction_df)),
        "gross_transaction_notional_float": float(transaction_df["signed_notional_float"].abs().sum())
        if len(transaction_df) else 0.0,
        "total_commission_float": float(transaction_df["commission_float"].sum()) if len(transaction_df) else 0.0,
        "negative_cash_day_count_int": int((cash_ser < 0.0).sum()),
        "minimum_cash_nav_weight_float": float((cash_ser / nav_ser).min()),
        "mean_cash_nav_weight_float": float((cash_ser / nav_ser).mean()),
        "positive_cash_rate_policy_str": str(accounting_policy_dict.get("positive_cash_rate_policy_str", "missing")),
        "negative_cash_financing_policy_str": str(accounting_policy_dict.get("negative_cash_financing_policy_str",
                                                                             "missing")),
        "execution_adjustment_str": str(data_adjustment_policy_dict.get("execution_and_marks_adjustment_str",
                                                                        "missing")),
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


def check_alias_table() -> None:
    registered_set = set(strategy_registry.pm_ready_import_tuple())
    table_set = {import_str for alias_str, (import_str, _) in SLEEVE_DICT.items()
                 if alias_str not in SHADOW_ALIAS_SET | SENSITIVITY_ALIAS_SET}
    if registered_set != table_set:
        raise RuntimeError(f"Alias table differs from the registry: missing={sorted(registered_set - table_set)} "
                           f"extra={sorted(table_set - registered_set)}")


def main(arg_list: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--only", nargs="*", default=None)
    args = parser.parse_args(arg_list)
    check_alias_table()

    STUDY_DIR_PATH.mkdir(parents=True, exist_ok=True)
    ledger_path = STUDY_DIR_PATH / "experiment_ledger.jsonl"
    vintage_start_dict = ladder_runner.norgate_database_vintage_dict()
    ladder_runner.append_jsonl(ledger_path, {
        "event_str": "sleeve_runs_started", "recorded_at_utc_str": ladder_runner.utc_now_str(),
        "spec_sha256_str": spec_sha256_str(), "requested_start_date_str": REQUESTED_START_DATE_STR,
        "common_end_date_str": COMMON_END_DATE_STR, "reference_capital_float": REFERENCE_CAPITAL_FLOAT,
        "only_list": args.only, "git_state_dict": git_state_dict(), "norgate_vintage_dict": vintage_start_dict,
        "shared_execution_dependency_hash_dict": ladder_runner.shared_execution_dependency_hash_dict()})

    alias_list = sorted(SLEEVE_DICT, key=lambda a: (a not in HEAVY_ALIAS_TUPLE,
                                                    HEAVY_ALIAS_TUPLE.index(a) if a in HEAVY_ALIAS_TUPLE else 0))
    if args.only:
        alias_list = [a for a in alias_list if a in set(args.only)]

    summary_row_list, failure_list = [], []
    with ProcessPoolExecutor(max_workers=args.workers, max_tasks_per_child=1) as executor_obj:
        future_map = {executor_obj.submit(run_one_sleeve, alias_str): alias_str for alias_str in alias_list}
        for future_obj in as_completed(future_map):
            alias_str = future_map[future_obj]
            try:
                row_dict = future_obj.result()
                summary_row_list.append({k: v for k, v in row_dict.items() if not isinstance(v, dict)})
                print(f"done {alias_str:<20} first_invested={row_dict['first_invested_date_str']} "
                      f"obs={row_dict['observation_count_int']} {row_dict['runtime_seconds_float']}s", flush=True)
                ladder_runner.append_jsonl(ledger_path, {"event_str": "sleeve_run_completed",
                                                         "recorded_at_utc_str": ladder_runner.utc_now_str(), **row_dict})
            except Exception as exc:  # noqa: BLE001 - record and continue; the summary fails the run below
                failure_list.append({"alias_str": alias_str, "error_str": repr(exc)})
                print(f"FAILED {alias_str}: {exc!r}", flush=True)
                ladder_runner.append_jsonl(ledger_path, {"event_str": "sleeve_run_failed",
                                                         "recorded_at_utc_str": ladder_runner.utc_now_str(),
                                                         "alias_str": alias_str, "error_str": repr(exc)})

    vintage_end_dict = ladder_runner.norgate_database_vintage_dict()
    summary_name_str = "sleeve_run_summary.csv" if not args.only else f"sleeve_run_summary_{'_'.join(args.only)}.csv"
    pd.DataFrame(summary_row_list).sort_values("alias_str").to_csv(STUDY_DIR_PATH / summary_name_str, index=False,
                                                                   lineterminator="\n")
    ladder_runner.append_jsonl(ledger_path, {
        "event_str": "sleeve_runs_finished", "recorded_at_utc_str": ladder_runner.utc_now_str(),
        "completed_count_int": len(summary_row_list), "failure_list": failure_list,
        "norgate_vintage_changed_bool": vintage_end_dict != vintage_start_dict,
        "norgate_vintage_end_dict": vintage_end_dict})
    print(json.dumps({"completed": len(summary_row_list), "failed": failure_list,
                      "norgate_vintage_changed": vintage_end_dict != vintage_start_dict}, indent=2))
    return 1 if failure_list or vintage_end_dict != vintage_start_dict else 0


if __name__ == "__main__":
    raise SystemExit(main())
