"""Audit re-run of the TAA / defensive / NDX-VXN sleeves and T-bills at the current HEAD.

Same call pattern as scripts/research/shelf_rebuild_20260929/run_sleeves.py (run_variant at $1M reference capital,
requested start 2000-01-03 with the module default as fallback), for a handful of aliases and a chosen end date.
Each alias writes <alias>__path.csv.gz, <alias>__transactions.csv.gz and <alias>__metadata.json (same columns and
fields as the original, plus git HEAD and the Norgate vintage).

T-bills: the shelf rebuild never ran an engine sleeve; it used BIL TOTALRETURN closes (lib.load_inputs). Here
`tbill` stores that same series as a path file (NAV = $1M * Close / first Close), and `bil_pod` is the PM_READY
engine pod strategies.portfolio_controls.strategy_passive_bil (registered after the shelf rebuild).

Research audit only. Nothing outside the output folder is written.

Usage: PYTHONPATH=. .venv/Scripts/python.exe scripts/research/fund_products_20261005/audit/rerun_taa_def_run.py
           [--end 2026-10-02] [--subdir .] [--workers 4] [--only alias ...]
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

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO_ROOT_PATH = HERE.parents[3]
if str(REPO_ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_PATH))

from scripts.research import run_ladder4_candidate_value_add_study as ladder_runner  # noqa: E402

AUDIT_DIR_PATH = (REPO_ROOT_PATH / "results" / "research" / "portfolio" / "fund_products_20261005" / "audit"
                  / "rerun_taa_def")
REFERENCE_CAPITAL_FLOAT = 1_000_000.0
REQUESTED_START_DATE_STR = "2000-01-03"
TBILL_ALIAS_STR = "tbill"

# alias -> (import string, extra run_variant kwargs); the first five rows are copied from the shelf rebuild table.
SLEEVE_DICT: dict[str, tuple[str, dict]] = {
    "core5": ("strategies.taa_beyond_6040.strategy_taa_adaptive_macro_core5", {}),
    "btal_qqq": ("strategies.taa_df.strategy_taa_df_btal_linearity_1n_fallback_qqq_vix_cash", {}),
    "taa3x": ("strategies.taa_df.strategy_taa_df_btal_fallback_tqqq_vix_cash", {}),
    "taa3x_1n": ("strategies.taa_df.strategy_taa_df_btal_1n_fallback_tqqq_vix_cash", {}),
    "ndx_vxn": ("strategies.momentum.strategy_mo_atr_normalized_ndx_vxn_scaled:VxnScaledAtrNormalizedNdxStrategy", {}),
    "bil_pod": ("strategies.portfolio_controls.strategy_passive_bil", {}),
}
ALIAS_ORDER_TUPLE = ("ndx_vxn", "core5", "btal_qqq", "taa3x", "taa3x_1n", "bil_pod", TBILL_ALIAS_STR)


def git_state_dict() -> dict:
    def run_git(arg_list: list[str]) -> str:
        return subprocess.run(["git", *arg_list], cwd=REPO_ROOT_PATH, capture_output=True, text=True,
                              check=False).stdout.strip()

    return {"head_commit_str": run_git(["rev-parse", "HEAD"]),
            "head_tree_str": run_git(["rev-parse", "HEAD^{tree}"]),
            "dirty_path_list": [line for line in run_git(["status", "--porcelain"]).splitlines() if line]}


def lf_sha256_str(file_path: Path) -> str:
    """Hash with CRLF folded to LF, so checkouts with different line endings compare equal."""
    return hashlib.sha256(file_path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def run_fresh_strategy(strategy_module_obj, extra_kwarg_dict: dict, end_date_str: str, scratch_dir_path: Path):
    """Run from the common requested start; fall back to the module's own default start (as run_sleeves.py)."""
    run_variant_fn = getattr(strategy_module_obj, "run_variant")
    call_kwargs_dict = dict(show_display_bool=False, save_results_bool=False, output_dir_str=str(scratch_dir_path),
                            capital_base_float=REFERENCE_CAPITAL_FLOAT, end_date_str=end_date_str,
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


def run_tbill(end_date_str: str, out_dir_path: Path) -> dict:
    """BIL TOTALRETURN closes exactly as shelf_rebuild lib.load_inputs loads them (start 2007-01-01)."""
    from data.norgate_loader import TOTALRETURN_ADJUSTMENT_STR, load_price_timeseries

    started_float = time.time()
    price_df = load_price_timeseries("BIL", adjustment_str=TOTALRETURN_ADJUSTMENT_STR, start_date_str="2007-01-01",
                                     end_date_str=end_date_str)
    close_ser = price_df["Close"].astype(float)
    close_ser.index = pd.to_datetime(close_ser.index).normalize()
    nav_ser = REFERENCE_CAPITAL_FLOAT * close_ser / float(close_ser.iloc[0])
    path_df = pd.DataFrame({"total_value_float": nav_ser, "portfolio_value_float": nav_ser, "cash_float": 0.0,
                            "bil_totalreturn_close_float": close_ser})
    path_df.index.name = "date"
    path_file_path = out_dir_path / f"{TBILL_ALIAS_STR}__path.csv.gz"
    ladder_runner.write_csv_gzip(path_df, path_file_path, index_bool=True, index_label_str="date")
    transaction_df = pd.DataFrame(columns=["source_id_str", "date", "asset_str", "amount_float", "fill_price_float",
                                           "signed_notional_float", "commission_float"])
    transaction_file_path = out_dir_path / f"{TBILL_ALIAS_STR}__transactions.csv.gz"
    ladder_runner.write_csv_gzip(transaction_df, transaction_file_path, index_bool=False)
    metadata_dict = {
        "alias_str": TBILL_ALIAS_STR,
        "strategy_import_str": "data.norgate_loader.load_price_timeseries('BIL', TOTALRETURN)",
        "note_str": "Not an engine run: BIL TOTALRETURN closes, the shelf rebuild's T-bill series (lib.load_inputs).",
        "reference_capital_float": REFERENCE_CAPITAL_FLOAT,
        "engine_first_date_str": path_df.index[0].date().isoformat(),
        "first_invested_date_str": path_df.index[0].date().isoformat(),
        "end_date_str": path_df.index[-1].date().isoformat(),
        "observation_count_int": int(len(path_df)),
        "transaction_count_int": 0,
        "path_sha256_str": ladder_runner.sha256_file_str(path_file_path),
        "transaction_sha256_str": ladder_runner.sha256_file_str(transaction_file_path),
        "runtime_seconds_float": round(time.time() - started_float, 1),
    }
    return metadata_dict


def run_one_sleeve(alias_str: str, end_date_str: str, out_dir_str: str) -> dict:
    """Worker: run one sleeve, write its compact artifacts, return its metadata."""
    out_dir_path = Path(out_dir_str)
    log_dir_path = out_dir_path / "source_logs"
    out_dir_path.mkdir(parents=True, exist_ok=True)
    log_dir_path.mkdir(parents=True, exist_ok=True)
    vintage_start_dict = ladder_runner.norgate_database_vintage_dict()
    if alias_str == TBILL_ALIAS_STR:
        metadata_dict = run_tbill(end_date_str, out_dir_path)
    else:
        strategy_import_str, extra_kwarg_dict = SLEEVE_DICT[alias_str]
        started_float = time.time()
        module_import_str = strategy_import_str.split(":", maxsplit=1)[0]
        with (log_dir_path / f"{alias_str}.log").open("w", encoding="utf-8") as log_file_obj, \
                contextlib.redirect_stdout(log_file_obj), contextlib.redirect_stderr(log_file_obj):
            strategy_module_obj = importlib.import_module(module_import_str)
            strategy_obj, requested_start_used_str, start_fallback_note_str = run_fresh_strategy(
                strategy_module_obj, extra_kwarg_dict, end_date_str, out_dir_path / "scratch_unused")

        source_result_df = ladder_runner.extract_source_result_df(strategy_obj)
        transaction_df = ladder_runner.extract_source_transaction_df(strategy_obj, alias_str)
        if source_result_df.index[-1] != pd.Timestamp(end_date_str):
            raise RuntimeError(f"{alias_str} ended {source_result_df.index[-1].date()}, not {end_date_str}.")

        invested_mask_ser = source_result_df["portfolio_value_float"].abs() > 1e-9
        first_invested_date_str = (invested_mask_ser[invested_mask_ser].index[0].date().isoformat()
                                   if invested_mask_ser.any() else None)
        path_file_path = out_dir_path / f"{alias_str}__path.csv.gz"
        transaction_file_path = out_dir_path / f"{alias_str}__transactions.csv.gz"
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
            "positive_cash_rate_policy_str": str(accounting_policy_dict.get("positive_cash_rate_policy_str",
                                                                            "missing")),
            "negative_cash_financing_policy_str": str(accounting_policy_dict.get(
                "negative_cash_financing_policy_str", "missing")),
            "execution_adjustment_str": str(data_adjustment_policy_dict.get("execution_and_marks_adjustment_str",
                                                                            "missing")),
            "module_path_str": module_path.relative_to(REPO_ROOT_PATH).as_posix(),
            "module_sha256_str": ladder_runner.sha256_file_str(module_path),
            "module_lf_sha256_str": lf_sha256_str(module_path),
            "path_sha256_str": ladder_runner.sha256_file_str(path_file_path),
            "transaction_sha256_str": ladder_runner.sha256_file_str(transaction_file_path),
            "runtime_seconds_float": round(time.time() - started_float, 1),
        }
        del strategy_obj
        gc.collect()
    metadata_dict["requested_end_date_str"] = end_date_str
    metadata_dict["git_state_dict"] = git_state_dict()
    metadata_dict["norgate_vintage_dict"] = vintage_start_dict
    metadata_dict["norgate_vintage_changed_during_run_bool"] = (
        ladder_runner.norgate_database_vintage_dict() != vintage_start_dict)
    metadata_dict["recorded_at_utc_str"] = ladder_runner.utc_now_str()
    ladder_runner.write_json(out_dir_path / f"{alias_str}__metadata.json", metadata_dict)
    return metadata_dict


def main(arg_list: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--end", default="2026-10-02")
    parser.add_argument("--subdir", default=".")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--only", nargs="*", default=None)
    args = parser.parse_args(arg_list)

    out_dir_path = (AUDIT_DIR_PATH / args.subdir).resolve()
    out_dir_path.mkdir(parents=True, exist_ok=True)
    alias_list = [a for a in ALIAS_ORDER_TUPLE if not args.only or a in set(args.only)]
    print(f"end={args.end} out={out_dir_path} aliases={alias_list}", flush=True)
    print(json.dumps({"git_state_dict": git_state_dict(),
                      "norgate_vintage_dict": ladder_runner.norgate_database_vintage_dict()}), flush=True)

    failure_list = []
    with ProcessPoolExecutor(max_workers=args.workers, max_tasks_per_child=1) as executor_obj:
        future_map = {executor_obj.submit(run_one_sleeve, alias_str, args.end, str(out_dir_path)): alias_str
                      for alias_str in alias_list}
        for future_obj in as_completed(future_map):
            alias_str = future_map[future_obj]
            try:
                row_dict = future_obj.result()
                print(f"done {alias_str:<10} first={row_dict['engine_first_date_str']} "
                      f"end={row_dict['end_date_str']} obs={row_dict['observation_count_int']} "
                      f"tx={row_dict['transaction_count_int']} {row_dict['runtime_seconds_float']}s", flush=True)
            except Exception as exc:  # noqa: BLE001 - report every failure, exit non-zero below
                failure_list.append({"alias_str": alias_str, "error_str": repr(exc)})
                print(f"FAILED {alias_str}: {exc!r}", flush=True)
    print(json.dumps({"failed": failure_list}), flush=True)
    return 1 if failure_list else 0


if __name__ == "__main__":
    raise SystemExit(main())
