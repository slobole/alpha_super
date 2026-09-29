"""Fresh standalone run of every PM_READY + WIRED sleeve for the fund product menu study.

Each sleeve runs once through its own ``run_variant`` at the same reference
capital, from the earliest date its data allows up to one common end date, on
one Norgate database vintage. Nothing here combines sleeves or ranks them; the
output is the raw material (daily NAV, invested value, cash, transactions,
lineage) that the later, separately frozen construction step reads.

Research authority only. No strategy, engine or live file is modified.
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
import pickle
import subprocess
import sys
import time
import traceback

import numpy as np
import pandas as pd

REPO_ROOT_PATH = Path(__file__).resolve().parents[3]
if str(REPO_ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_PATH))

from alpha import strategy_registry  # noqa: E402
from scripts.research import run_ladder4_candidate_value_add_study as ladder_runner  # noqa: E402

STUDY_DIR_PATH = REPO_ROOT_PATH / "results" / "research" / "portfolio" / "fund_product_menu_20260923"
SOURCE_DIR_PATH = STUDY_DIR_PATH / "sources"
LOG_DIR_PATH = STUDY_DIR_PATH / "source_logs"

REFERENCE_CAPITAL_FLOAT = 1_000_000.0
REQUESTED_START_DATE_STR = "2000-01-03"
# Tactical Fixed Income is contractually frozen through 2026-08-19; one common
# end date keeps every sleeve and every book on the same window.
COMMON_END_DATE_STR = "2026-08-19"

# Short, stable aliases used throughout the study. Every registered sleeve at or
# above PM_READY must appear exactly once (checked at start).
SLEEVE_ALIAS_BY_IMPORT_DICT: dict[str, str] = {
    "strategies.dv2.strategy_mr_dv2:DVO2Strategy": "dv2",
    "strategies.qpi.strategy_mr_qpi_ibs_rsi_exit:QPIIbsRsiExitStrategy": "qpi",
    "strategies.hpi.strategy_mr_hpi_sp500_2_3_5_vote": "hpi_vote",
    "strategies.hpi.strategy_mr_hpi_sp500_ibs_rsi_exit": "hpi_ibs_rsi",
    "strategies.taa_df.strategy_taa_df_btal_fallback_tqqq_vix_cash": "taa_btal_tqqq",
    "strategies.taa_df.strategy_taa_df_btal_1n_fallback_tqqq_vix_cash": "taa_btal_1n_tqqq",
    "strategies.taa_df.strategy_taa_df_btal_linearity_1n_fallback_qqq_vix_cash": "taa_btal_lin_qqq",
    "strategies.momentum.strategy_mo_atr_normalized_ndx:AtrNormalizedNdxStrategy": "ndx_atr",
    "strategies.momentum.strategy_mo_atr_normalized_ndx_vxn_scaled:VxnScaledAtrNormalizedNdxStrategy": "ndx_vxn",
    "strategies.taa_beyond_6040.strategy_taa_month_end_rebalancing_flow": "eom_flow",
    "strategies.tail_hedge.strategy_crisis_trend_core": "crisis_trend",
    "strategies.tail_hedge.strategy_vixm_backwardation": "vixm",
    "strategies.taa_df.strategy_taa_df_1n_fallback_qld_vix_cash": "taa_1n_qld",
    "strategies.taa_df.strategy_taa_df_1n_fallback_sso_vix_cash": "taa_1n_sso",
    "strategies.taa_df.strategy_taa_df_btal_1n_fallback_qld_vix_cash": "taa_btal_1n_qld",
    "strategies.mean_reversion.strategy_mr_us_sector_etf_ibs_downshock_vox_iyr": "sector_vox_iyr",
    "strategies.taa_df.strategy_taa_df_linearity_1n_fallback_qqq_vix_cash": "taa_lin_qqq",
    "strategies.mean_reversion.strategy_mr_sector_dispersion_ibs_kie_ihi_xlc": "disp_kie_ihi_xlc",
    "strategies.mean_reversion.strategy_mr_sector_dispersion_ibs_kie_ihi_xlc_asset_sma200": "disp_kie_ihi_xlc_sma",
    "strategies.mean_reversion.strategy_mr_sector_dispersion_ibs_kie_ihi_asset_sma200": "disp_kie_ihi_sma",
    "strategies.taa_beyond_6040.strategy_taa_trinity_vol_control_8_bil": "trinity",
    "strategies.taa_beyond_6040.strategy_taa_adaptive_macro_core5": "core5",
    "strategies.taa_df.strategy_taa_inflation_compass": "infl_compass",
    "strategies.taa_beyond_6040.strategy_taa_tactical_fixed_income_ief_lqd": "tactical_fi",
    "strategies.momentum.strategy_mo_mosaic_russell1000:MosaicRussell1000Strategy": "mosaic",
}

# Single-stock universes are memory-heavy; start them first so the long tail of
# light ETF sleeves fills the remaining workers.
HEAVY_ALIAS_TUPLE = ("mosaic", "dv2", "qpi", "hpi_vote", "hpi_ibs_rsi", "ndx_atr", "ndx_vxn")

# *** CRITICAL*** Tactical Fixed Income refuses a fresh run: its frozen Norgate
# price fingerprint (IEF/LQD/$SPXTR) no longer matches the 2026-09-23 vendor
# vintage, and the module says a human must review the revision before the
# governed snapshot changes. We do not bypass that guard. The sleeve is read
# from its last governed run instead (same frozen rule, same end date), and the
# overlap is checked against the 2026-08-30 Foundry fresh run downstream.
SAVED_ARTIFACT_BY_ALIAS_DICT: dict[str, Path] = {
    "tactical_fi": REPO_ROOT_PATH
    / "results/research/strategy/strategy_taa_tactical_fixed_income_ief_lqd/vanilla_backtest"
    / "2026-08-20_214948/strategy_taa_tactical_fixed_income_ief_lqd.pkl",
    # Crisis Trend Core (hedge-overlay test only) no longer runs fresh: its SHY
    # total-return endpoint check fails at Close_2003-01-13 on the 2026-09-23
    # code/data vintage (SHY launched 2002-07, inside the 252-day lookback). The
    # governed 2026-09-04 run is used and trimmed to the common end date.
    "crisis_trend": REPO_ROOT_PATH
    / "results/research/strategy/strategy_crisis_trend_core/vanilla_backtest"
    / "2026-09-04_232043/strategy_crisis_trend_core.pkl",
}


SAVED_ARTIFACT_REASON_BY_ALIAS_DICT: dict[str, str] = {
    "tactical_fi": "fresh run blocked by the module's frozen Norgate fingerprint ($SPXTR benchmark revised; IEF/LQD unchanged)",
    "crisis_trend": "fresh run fails the SHY total-return endpoint check at Close_2003-01-13 on the current vintage",
}


def git_state_dict() -> dict:
    def run_git(arg_list: list[str]) -> str:
        return subprocess.run(
            ["git", *arg_list], cwd=REPO_ROOT_PATH, capture_output=True, text=True, check=False
        ).stdout.strip()

    return {
        "head_commit_str": run_git(["rev-parse", "HEAD"]),
        "dirty_path_list": [line for line in run_git(["status", "--porcelain"]).splitlines() if line],
    }


def load_saved_strategy(saved_artifact_path: Path):
    with saved_artifact_path.open("rb") as pickle_file_obj:
        return pickle.load(pickle_file_obj)


def run_fresh_strategy(strategy_module_obj) -> tuple:
    """Run from the common requested start; fall back to the module's own default start."""
    run_variant_fn = getattr(strategy_module_obj, "run_variant")
    call_kwargs_dict = dict(
        show_display_bool=False,
        save_results_bool=False,
        output_dir_str=str(STUDY_DIR_PATH / "scratch_unused"),
        capital_base_float=REFERENCE_CAPITAL_FLOAT,
        end_date_str=COMMON_END_DATE_STR,
    )
    try:
        strategy_obj = run_variant_fn(backtest_start_date_str=REQUESTED_START_DATE_STR, **call_kwargs_dict)
        return strategy_obj, REQUESTED_START_DATE_STR, ""
    except Exception as first_exc:  # data may not reach the requested start
        default_start_obj = inspect.signature(run_variant_fn).parameters["backtest_start_date_str"].default
        print(f"Requested start failed ({first_exc!r}); retrying with module default {default_start_obj!r}.")
        traceback.print_exc()
        strategy_obj = run_variant_fn(backtest_start_date_str=default_start_obj, **call_kwargs_dict)
        return strategy_obj, str(default_start_obj), f"requested {REQUESTED_START_DATE_STR} failed: {first_exc!r}"


def run_one_sleeve(strategy_import_str: str, alias_str: str) -> dict:
    """Worker: run one sleeve, write its compact artifacts, return a summary row."""
    SOURCE_DIR_PATH.mkdir(parents=True, exist_ok=True)
    LOG_DIR_PATH.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR_PATH / f"{alias_str}.log"
    started_float = time.time()
    module_import_str = strategy_import_str.split(":", maxsplit=1)[0]
    requested_start_used_str = REQUESTED_START_DATE_STR
    start_fallback_note_str = ""

    saved_artifact_path = SAVED_ARTIFACT_BY_ALIAS_DICT.get(alias_str)
    with log_path.open("w", encoding="utf-8") as log_file_obj, contextlib.redirect_stdout(
        log_file_obj
    ), contextlib.redirect_stderr(log_file_obj):
        strategy_module_obj = importlib.import_module(module_import_str)
        if saved_artifact_path is not None:
            strategy_obj = load_saved_strategy(saved_artifact_path)
            requested_start_used_str = "saved_governed_artifact"
            start_fallback_note_str = (
                f"{SAVED_ARTIFACT_REASON_BY_ALIAS_DICT[alias_str]}; loaded "
                f"{saved_artifact_path.relative_to(REPO_ROOT_PATH).as_posix()} "
                f"sha256={ladder_runner.sha256_file_str(saved_artifact_path)}"
            )
            print(start_fallback_note_str)
        else:
            strategy_obj, requested_start_used_str, start_fallback_note_str = run_fresh_strategy(
                strategy_module_obj
            )

    source_result_df = ladder_runner.extract_source_result_df(strategy_obj)
    transaction_df = ladder_runner.extract_source_transaction_df(strategy_obj, alias_str)
    if saved_artifact_path is not None:
        # *** CRITICAL*** a saved run may extend past the common end date; the
        # path up to that date is unchanged by later bars, so trimming is causal.
        source_result_df = source_result_df.loc[: pd.Timestamp(COMMON_END_DATE_STR)]
        transaction_df = transaction_df[transaction_df["date"] <= pd.Timestamp(COMMON_END_DATE_STR)]
    if source_result_df.index[-1] != pd.Timestamp(COMMON_END_DATE_STR):
        raise RuntimeError(
            f"{alias_str} ended {source_result_df.index[-1].date()}, not {COMMON_END_DATE_STR}."
        )

    # First day the sleeve holds anything: leading all-cash warmup days carry no
    # information about the strategy and are trimmed later, never filled.
    invested_mask_ser = source_result_df["portfolio_value_float"].abs() > 1e-9
    first_invested_date_str = (
        invested_mask_ser[invested_mask_ser].index[0].date().isoformat() if invested_mask_ser.any() else None
    )
    first_transaction_date_str = (
        pd.Timestamp(transaction_df["date"].min()).date().isoformat() if len(transaction_df) else None
    )

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
        "tier_str": strategy_registry.tier_label_for(strategy_import_str),
        "strategy_name_str": str(strategy_obj.name),
        "reference_capital_float": REFERENCE_CAPITAL_FLOAT,
        "first_nav_float": float(source_result_df["total_value_float"].iloc[0]),
        "requested_start_date_str": requested_start_used_str,
        "start_fallback_note_str": start_fallback_note_str,
        "engine_first_date_str": source_result_df.index[0].date().isoformat(),
        "first_invested_date_str": first_invested_date_str,
        "first_transaction_date_str": first_transaction_date_str,
        "end_date_str": source_result_df.index[-1].date().isoformat(),
        "observation_count_int": int(len(source_result_df)),
        "transaction_count_int": int(len(transaction_df)),
        "gross_transaction_notional_float": float(transaction_df["signed_notional_float"].abs().sum())
        if len(transaction_df)
        else 0.0,
        "total_commission_float": float(transaction_df["commission_float"].sum()) if len(transaction_df) else 0.0,
        "negative_cash_day_count_int": int((cash_ser < 0.0).sum()),
        "minimum_cash_float": float(cash_ser.min()),
        "minimum_cash_nav_weight_float": float((cash_ser / nav_ser).min()),
        "slippage_per_side_float": float(getattr(strategy_obj, "_slippage", np.nan)),
        "commission_per_share_float": float(getattr(strategy_obj, "_commission_per_share", np.nan)),
        "positive_cash_rate_policy_str": str(accounting_policy_dict.get("positive_cash_rate_policy_str", "missing")),
        "negative_cash_financing_policy_str": str(
            accounting_policy_dict.get("negative_cash_financing_policy_str", "missing")
        ),
        "execution_adjustment_str": str(
            data_adjustment_policy_dict.get("execution_and_marks_adjustment_str", "missing")
        ),
        "dividend_cash_net_total_float": float(getattr(strategy_obj, "dividend_cash_net_total_float", 0.0)),
        "module_path_str": str(module_path.relative_to(REPO_ROOT_PATH)),
        "module_sha256_str": ladder_runner.sha256_file_str(module_path),
        "path_sha256_str": ladder_runner.sha256_file_str(path_file_path),
        "transaction_sha256_str": ladder_runner.sha256_file_str(transaction_file_path),
        "runtime_seconds_float": round(time.time() - started_float, 1),
        "accounting_policy_dict": accounting_policy_dict,
        "data_adjustment_policy_dict": data_adjustment_policy_dict,
    }
    ladder_runner.write_json(SOURCE_DIR_PATH / f"{alias_str}__metadata.json", metadata_dict)
    del strategy_obj
    gc.collect()
    return {key_str: value_obj for key_str, value_obj in metadata_dict.items() if not key_str.endswith("_dict")}


def main(arg_list: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--only", nargs="*", default=None, help="Run only these aliases (debug).")
    args = parser.parse_args(arg_list)

    registered_import_set = set(strategy_registry.pm_ready_import_tuple())
    if registered_import_set != set(SLEEVE_ALIAS_BY_IMPORT_DICT):
        raise RuntimeError(
            "Alias table differs from the registry: "
            f"missing={sorted(registered_import_set - set(SLEEVE_ALIAS_BY_IMPORT_DICT))} "
            f"extra={sorted(set(SLEEVE_ALIAS_BY_IMPORT_DICT) - registered_import_set)}"
        )

    STUDY_DIR_PATH.mkdir(parents=True, exist_ok=True)
    ledger_path = STUDY_DIR_PATH / "experiment_ledger.jsonl"
    vintage_start_dict = ladder_runner.norgate_database_vintage_dict()
    ladder_runner.append_jsonl(
        ledger_path,
        {
            "event_str": "source_runs_started",
            "recorded_at_utc_str": ladder_runner.utc_now_str(),
            "requested_start_date_str": REQUESTED_START_DATE_STR,
            "common_end_date_str": COMMON_END_DATE_STR,
            "reference_capital_float": REFERENCE_CAPITAL_FLOAT,
            "git_state_dict": git_state_dict(),
            "norgate_vintage_dict": vintage_start_dict,
            "shared_execution_dependency_hash_dict": ladder_runner.shared_execution_dependency_hash_dict(),
        },
    )

    job_list = sorted(
        SLEEVE_ALIAS_BY_IMPORT_DICT.items(),
        key=lambda item: (item[1] not in HEAVY_ALIAS_TUPLE, HEAVY_ALIAS_TUPLE.index(item[1]) if item[1] in HEAVY_ALIAS_TUPLE else 0),
    )
    if args.only:
        job_list = [job for job in job_list if job[1] in set(args.only)]

    summary_row_list: list[dict] = []
    failure_list: list[dict] = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor_obj:
        future_map = {
            executor_obj.submit(run_one_sleeve, import_str, alias_str): alias_str for import_str, alias_str in job_list
        }
        for future_obj in as_completed(future_map):
            alias_str = future_map[future_obj]
            try:
                row_dict = future_obj.result()
                summary_row_list.append(row_dict)
                print(
                    f"done {alias_str:<22} first_invested={row_dict['first_invested_date_str']} "
                    f"obs={row_dict['observation_count_int']} {row_dict['runtime_seconds_float']}s",
                    flush=True,
                )
                ladder_runner.append_jsonl(
                    ledger_path,
                    {"event_str": "source_run_completed", "recorded_at_utc_str": ladder_runner.utc_now_str(), **row_dict},
                )
            except Exception as exc:
                failure_list.append({"alias_str": alias_str, "error_str": repr(exc)})
                print(f"FAILED {alias_str}: {exc!r}", flush=True)
                ladder_runner.append_jsonl(
                    ledger_path,
                    {
                        "event_str": "source_run_failed",
                        "recorded_at_utc_str": ladder_runner.utc_now_str(),
                        "alias_str": alias_str,
                        "error_str": repr(exc),
                    },
                )

    vintage_end_dict = ladder_runner.norgate_database_vintage_dict()
    summary_path = STUDY_DIR_PATH / ("source_run_summary.csv" if not args.only else "source_run_summary_partial.csv")
    pd.DataFrame(summary_row_list).sort_values("alias_str").to_csv(summary_path, index=False, lineterminator="\n")
    ladder_runner.append_jsonl(
        ledger_path,
        {
            "event_str": "source_runs_finished",
            "recorded_at_utc_str": ladder_runner.utc_now_str(),
            "completed_count_int": len(summary_row_list),
            "failure_list": failure_list,
            "norgate_vintage_changed_bool": vintage_end_dict != vintage_start_dict,
            "norgate_vintage_end_dict": vintage_end_dict,
        },
    )
    print(json.dumps({"completed": len(summary_row_list), "failed": failure_list,
                      "norgate_vintage_changed": vintage_end_dict != vintage_start_dict}, indent=2))
    return 1 if failure_list else 0


if __name__ == "__main__":
    raise SystemExit(main())
