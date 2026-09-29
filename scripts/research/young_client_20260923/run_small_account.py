"""Run TAA_DF variants and CORE5 at small-account sizes (and at $1M for reference).

Each sleeve runs through its own run_variant at the exact dollar size its pod
would hold, so whole-share rounding and the $1 minimum commission are part of
the result. Outputs: results/research/portfolio/young_client_taa_core5_20260923/sources/.
Research authority only; no strategy or engine file is modified.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import contextlib
import hashlib
import importlib
import json
from pathlib import Path
import sys
import time

import pandas as pd
import yaml

REPO_ROOT_PATH = Path(__file__).resolve().parents[3]
if str(REPO_ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_PATH))

from scripts.research import run_ladder4_candidate_value_add_study as ladder_runner  # noqa: E402
from scripts.research.fund_menu_20260923.run_sources import SLEEVE_ALIAS_BY_IMPORT_DICT  # noqa: E402

PLAN_PATH = Path(__file__).resolve().parent / "plan_frozen.yaml"
STUDY_DIR_PATH = REPO_ROOT_PATH / "results" / "research" / "portfolio" / "young_client_taa_core5_20260923"
SOURCE_DIR_PATH = STUDY_DIR_PATH / "sources"
IMPORT_BY_ALIAS_DICT = {alias_str: import_str for import_str, alias_str in SLEEVE_ALIAS_BY_IMPORT_DICT.items()}


def run_job(alias_str: str, capital_float: float, end_date_str: str) -> dict:
    SOURCE_DIR_PATH.mkdir(parents=True, exist_ok=True)
    tag_str = f"{alias_str}__{int(round(capital_float))}"
    started_float = time.time()
    module_obj = importlib.import_module(IMPORT_BY_ALIAS_DICT[alias_str].split(":")[0])
    with (SOURCE_DIR_PATH / f"{tag_str}.log").open("w", encoding="utf-8") as log_obj, \
            contextlib.redirect_stdout(log_obj), contextlib.redirect_stderr(log_obj):
        strategy_obj = module_obj.run_variant(
            show_display_bool=False, save_results_bool=False, output_dir_str=str(STUDY_DIR_PATH / "scratch_unused"),
            backtest_start_date_str="2000-01-03", capital_base_float=float(capital_float), end_date_str=end_date_str,
        )
    path_df = ladder_runner.extract_source_result_df(strategy_obj)
    transaction_df = ladder_runner.extract_source_transaction_df(strategy_obj, tag_str)
    ladder_runner.write_csv_gzip(path_df, SOURCE_DIR_PATH / f"{tag_str}__path.csv.gz", index_bool=True, index_label_str="date")
    ladder_runner.write_csv_gzip(transaction_df, SOURCE_DIR_PATH / f"{tag_str}__transactions.csv.gz", index_bool=False)
    invested_ser = path_df["portfolio_value_float"].abs() > 1e-9
    return {
        "tag_str": tag_str, "alias_str": alias_str, "capital_float": float(capital_float),
        "first_invested_date_str": invested_ser[invested_ser].index[0].date().isoformat(),
        "end_date_str": path_df.index[-1].date().isoformat(),
        "transaction_count_int": int(len(transaction_df)),
        "total_commission_float": float(transaction_df["commission_float"].sum()) if len(transaction_df) else 0.0,
        "min_cash_weight_float": float((path_df["cash_float"] / path_df["total_value_float"]).min()),
        "runtime_seconds_float": round(time.time() - started_float, 1),
    }


def main() -> int:
    plan_dict = yaml.safe_load(PLAN_PATH.read_text(encoding="utf-8"))
    STUDY_DIR_PATH.mkdir(parents=True, exist_ok=True)
    ledger_path = STUDY_DIR_PATH / "experiment_ledger.jsonl"
    with ledger_path.open("a", encoding="utf-8") as ledger_obj:
        ledger_obj.write(json.dumps({"event_str": "plan_frozen_before_runs", "recorded_at_utc_str": ladder_runner.utc_now_str(),
                                     "plan_sha256_str": hashlib.sha256(PLAN_PATH.read_bytes()).hexdigest()}) + "\n")
    end_date_str = plan_dict["end_date_str"]
    job_list = []
    for alias_str in plan_dict["candidates"]["taa_variants"]:
        for capital_float in plan_dict["execution_realism"]["pod_capitals_usd"]["taa"] + [plan_dict["execution_realism"]["reference_capital_usd"]]:
            job_list.append((alias_str, capital_float))
    for capital_float in plan_dict["execution_realism"]["pod_capitals_usd"]["core5"] + [plan_dict["execution_realism"]["reference_capital_usd"]]:
        job_list.append(("core5", capital_float))
    row_list = []
    with ProcessPoolExecutor(max_workers=5) as executor_obj:
        future_map = {executor_obj.submit(run_job, a, c, end_date_str): (a, c) for a, c in job_list}
        for future_obj in as_completed(future_map):
            alias_str, capital_float = future_map[future_obj]
            try:
                row_list.append(future_obj.result())
                print(f"done {alias_str} @ {capital_float:,.0f}", flush=True)
            except Exception as exc:  # report, keep going
                print(f"FAILED {alias_str} @ {capital_float:,.0f}: {exc!r}", flush=True)
                row_list.append({"tag_str": f"{alias_str}__{int(round(capital_float))}", "alias_str": alias_str,
                                 "capital_float": capital_float, "error_str": repr(exc)})
    pd.DataFrame(row_list).sort_values("tag_str").to_csv(STUDY_DIR_PATH / "run_summary.csv", index=False)
    with ledger_path.open("a", encoding="utf-8") as ledger_obj:
        ledger_obj.write(json.dumps({"event_str": "runs_finished", "recorded_at_utc_str": ladder_runner.utc_now_str(),
                                     "job_count_int": len(job_list), "failed_count_int": sum("error_str" in r for r in row_list)}) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
