"""Compare CORE5 with an independently compounded 50/50 CORE5/TFI sleeve.

The study starts at the earliest clean CORE5 execution date.  It preserves
each strategy's native accounting and execution contract, anchors both
products in cash on 2007-08-31, and performs no outer rebalance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
from typing import Any, Iterable

import numpy as np
import pandas as pd

from scripts.research import run_defensive_sleeve_v3_phase0 as phase0_runner
from scripts.research import run_ladder4_candidate_value_add_study as source_runner


REPO_ROOT_PATH = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR_PATH = (
    REPO_ROOT_PATH
    / "results"
    / "research"
    / "portfolio"
    / "defensive_sleeve_certification"
    / "2026-08-30_core5_vs_tfi_equal_earliest_rerun2"
)
CORE5_IMPORT_STR = (
    "strategies.taa_beyond_6040.strategy_taa_adaptive_macro_core5"
)
TACTICAL_FI_IMPORT_STR = (
    "strategies.taa_beyond_6040.strategy_taa_tactical_fixed_income_ief_lqd"
)
CAPITAL_ANCHOR_DATE_STR = "2007-08-31"
EFFECTIVE_EXECUTION_START_DATE_STR = "2007-09-04"
END_DATE_STR = "2026-08-19"
TOTAL_CAPITAL_FLOAT = 750_000.0
HALF_CAPITAL_FLOAT = TOTAL_CAPITAL_FLOAT / 2.0
TRADING_DAY_COUNT_PER_YEAR_INT = 252
INHERITED_HISTORICAL_TRIAL_COUNT_FLOOR_INT = 73


def sha256_bytes_str(payload_bytes: bytes) -> str:
    return hashlib.sha256(payload_bytes).hexdigest()


def build_source_contract_dict() -> dict[str, Any]:
    return {
        "authority_str": "research_bench_only",
        "portfolio_contract": {
            "capital_base_float": TOTAL_CAPITAL_FLOAT,
            "requested_start_date_str": "2002-07-26",
            "capital_anchor_date_str": CAPITAL_ANCHOR_DATE_STR,
            "effective_execution_start_date_str": (
                EFFECTIVE_EXECUTION_START_DATE_STR
            ),
            "end_date_str": END_DATE_STR,
            "outer_rebalance": None,
            "allocation_semantics_str": (
                "fixed_initial_capital_then_independent_drift"
            ),
            "decision_execution_timing_str": "Close_T_to_Open_T_plus_1",
        },
        "lineage_contract": {
            "native_history_request_start_by_strategy_import": {
                CORE5_IMPORT_STR: "1990-01-01",
                TACTICAL_FI_IMPORT_STR: "2002-07-26",
            }
        },
        "source_runs": {
            "core5_375000": {
                "strategy_import_str": CORE5_IMPORT_STR,
                "allocated_capital_float": HALF_CAPITAL_FLOAT,
                "run_variant_kwargs_dict": {},
            },
            "core5_750000": {
                "strategy_import_str": CORE5_IMPORT_STR,
                "allocated_capital_float": TOTAL_CAPITAL_FLOAT,
                "run_variant_kwargs_dict": {},
            },
            "tactical_fi_375000": {
                "strategy_import_str": TACTICAL_FI_IMPORT_STR,
                "allocated_capital_float": HALF_CAPITAL_FLOAT,
                "run_variant_kwargs_dict": {},
            },
        },
    }


def build_product_value_df(
    source_path_by_id_dict: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    source_id_tuple = (
        "core5_375000",
        "core5_750000",
        "tactical_fi_375000",
    )
    global_idx = pd.DatetimeIndex(
        source_path_by_id_dict[source_id_tuple[0]].index
    )
    for source_id_str in source_id_tuple[1:]:
        global_idx = global_idx.intersection(
            pd.DatetimeIndex(source_path_by_id_dict[source_id_str].index)
        )
    global_idx = global_idx.sort_values()
    if len(global_idx) < 2:
        raise RuntimeError("No usable exact source intersection.")
    if global_idx[0] != pd.Timestamp(CAPITAL_ANCHOR_DATE_STR):
        raise RuntimeError("The source intersection has the wrong cash anchor.")
    if global_idx[1] != pd.Timestamp(EFFECTIVE_EXECUTION_START_DATE_STR):
        raise RuntimeError("The source intersection has the wrong first session.")
    if global_idx[-1] != pd.Timestamp(END_DATE_STR):
        raise RuntimeError("The source intersection has the wrong endpoint.")

    core5_full_value_ser = source_path_by_id_dict["core5_750000"].reindex(
        global_idx
    )["total_value_float"].astype(float)
    core5_half_value_ser = source_path_by_id_dict["core5_375000"].reindex(
        global_idx
    )["total_value_float"].astype(float)
    tactical_fi_value_ser = source_path_by_id_dict[
        "tactical_fi_375000"
    ].reindex(global_idx)["total_value_float"].astype(float)
    product_value_df = pd.DataFrame(
        {
            "CORE5": core5_full_value_ser,
            "CORE5_50_TACTICAL_FI_50": (
                core5_half_value_ser + tactical_fi_value_ser
            ),
        },
        index=global_idx,
    )
    if product_value_df.isna().any().any():
        raise RuntimeError("A product contains missing values.")
    if not np.allclose(
        product_value_df.iloc[0].to_numpy(dtype=float),
        TOTAL_CAPITAL_FLOAT,
        rtol=0.0,
        atol=1e-6,
    ):
        raise RuntimeError("Products do not start at the same capital.")
    product_value_df.index.name = "date"
    return product_value_df


def drawdown_detail_dict(total_value_ser: pd.Series) -> dict[str, Any]:
    clean_value_ser = total_value_ser.astype(float).dropna()
    running_peak_ser = clean_value_ser.cummax()
    drawdown_ser = clean_value_ser.div(running_peak_ser).sub(1.0)
    trough_ts = pd.Timestamp(drawdown_ser.idxmin())
    peak_ts = pd.Timestamp(clean_value_ser.loc[:trough_ts].idxmax())
    peak_value_float = float(clean_value_ser.loc[peak_ts])
    recovery_value_ser = clean_value_ser.loc[trough_ts:]
    recovered_bool_ser = recovery_value_ser.ge(peak_value_float)
    recovery_ts = (
        pd.Timestamp(recovered_bool_ser[recovered_bool_ser].index[0])
        if recovered_bool_ser.any()
        else None
    )
    return {
        "max_drawdown_float": float(drawdown_ser.loc[trough_ts]),
        "peak_date_str": peak_ts.date().isoformat(),
        "trough_date_str": trough_ts.date().isoformat(),
        "recovery_date_str": (
            recovery_ts.date().isoformat() if recovery_ts is not None else ""
        ),
    }


def headline_metric_df(product_value_df: pd.DataFrame) -> pd.DataFrame:
    row_list: list[dict[str, Any]] = []
    for product_id_str in product_value_df.columns:
        product_value_ser = product_value_df[product_id_str]
        metric_value_dict = phase0_runner.metric_dict(product_value_ser)
        max_drawdown_float = float(metric_value_dict["max_drawdown_float"])
        row_list.append(
            {
                "product_id_str": product_id_str,
                **metric_value_dict,
                "mar_float": (
                    float(metric_value_dict["cagr_float"])
                    / abs(max_drawdown_float)
                    if max_drawdown_float < 0.0
                    else float("nan")
                ),
                "ending_value_float": float(product_value_ser.iloc[-1]),
                **drawdown_detail_dict(product_value_ser),
            }
        )
    return pd.DataFrame(row_list)


def calendar_return_df(product_value_df: pd.DataFrame) -> pd.DataFrame:
    year_end_value_df = product_value_df.resample("YE").last()
    # *** CRITICAL*** retrospective return calculation only: the previous
    # calendar endpoint is used solely for reporting and never feeds a trade.
    calendar_return_df = year_end_value_df.pct_change(fill_method=None)
    first_year_int = int(product_value_df.index[0].year)
    final_year_int = int(product_value_df.index[-1].year)
    calendar_return_df.loc[calendar_return_df.index[0]] = (
        year_end_value_df.iloc[0].div(product_value_df.iloc[0]).sub(1.0)
    )
    calendar_return_df.index = calendar_return_df.index.year
    calendar_return_df.index.name = "year_int"
    calendar_return_df.insert(
        0,
        "window_status_str",
        [
            (
                "partial_from_2007_08_31"
                if year_int == first_year_int
                else "partial_through_2026_08_19"
                if year_int == final_year_int
                else "full_year"
            )
            for year_int in calendar_return_df.index
        ],
    )
    calendar_return_df["equal_minus_core5_float"] = (
        calendar_return_df["CORE5_50_TACTICAL_FI_50"]
        - calendar_return_df["CORE5"]
    )
    return calendar_return_df.reset_index()


def component_diagnostic_df(
    source_path_by_id_dict: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    core5_value_ser = source_path_by_id_dict["core5_375000"][
        "total_value_float"
    ].astype(float)
    tactical_fi_value_ser = source_path_by_id_dict["tactical_fi_375000"][
        "total_value_float"
    ].astype(float)
    common_idx = core5_value_ser.index.intersection(tactical_fi_value_ser.index)
    core5_value_ser = core5_value_ser.reindex(common_idx)
    tactical_fi_value_ser = tactical_fi_value_ser.reindex(common_idx)
    combined_value_ser = core5_value_ser + tactical_fi_value_ser
    # *** CRITICAL*** retrospective return diagnostic only: these component
    # returns do not feed portfolio weights, signals, or execution.
    core5_return_ser = core5_value_ser.pct_change(fill_method=None).iloc[1:]
    tactical_fi_return_ser = tactical_fi_value_ser.pct_change(
        fill_method=None
    ).iloc[1:]
    return pd.DataFrame(
        [
            {
                "daily_return_correlation_float": float(
                    core5_return_ser.corr(tactical_fi_return_ser)
                ),
                "initial_core5_weight_float": float(
                    core5_value_ser.iloc[0] / combined_value_ser.iloc[0]
                ),
                "ending_core5_weight_float": float(
                    core5_value_ser.iloc[-1] / combined_value_ser.iloc[-1]
                ),
                "ending_tactical_fi_weight_float": float(
                    tactical_fi_value_ser.iloc[-1] / combined_value_ser.iloc[-1]
                ),
            }
        ]
    )


def execution_reality_df(
    source_path_by_id_dict: dict[str, pd.DataFrame],
    source_summary_df: pd.DataFrame,
) -> pd.DataFrame:
    summary_by_id_df = source_summary_df.set_index("source_id_str")
    row_list: list[dict[str, Any]] = []
    for source_id_str, source_path_df in source_path_by_id_dict.items():
        source_summary_ser = summary_by_id_df.loc[source_id_str]
        cash_weight_ser = source_path_df["cash_float"].astype(float).div(
            source_path_df["total_value_float"].astype(float)
        )
        row_list.append(
            {
                "source_id_str": source_id_str,
                "negative_cash_day_count_int": int(
                    source_summary_ser["negative_cash_day_count_int"]
                ),
                "minimum_cash_float": float(
                    source_summary_ser["minimum_cash_float"]
                ),
                "minimum_cash_weight_float": float(cash_weight_ser.min()),
                "negative_cash_financing_policy_str": str(
                    source_summary_ser["negative_cash_financing_policy_str"]
                ),
                "slippage_per_side_float": float(
                    source_summary_ser["slippage_per_side_float"]
                ),
                "commission_per_share_float": float(
                    source_summary_ser["commission_per_share_float"]
                ),
                "commission_minimum_float": float(
                    source_summary_ser["commission_minimum_float"]
                ),
            }
        )
    return pd.DataFrame(row_list)


def write_report(
    headline_df: pd.DataFrame,
    calendar_df: pd.DataFrame,
    component_df: pd.DataFrame,
    execution_reality_df: pd.DataFrame,
    output_dir_path: Path,
) -> Path:
    report_headline_df = headline_df.copy()
    for column_str in (
        "cagr_float",
        "annualized_volatility_float",
        "max_drawdown_float",
        "worst_252d_return_float",
    ):
        report_headline_df[column_str] = report_headline_df[column_str].map(
            lambda value_float: f"{100.0 * float(value_float):.2f}%"
        )
    report_headline_df["sharpe_float"] = report_headline_df[
        "sharpe_float"
    ].map(lambda value_float: f"{float(value_float):.3f}")
    report_headline_df["mar_float"] = report_headline_df["mar_float"].map(
        lambda value_float: f"{float(value_float):.3f}"
    )
    report_headline_df["ending_value_float"] = report_headline_df[
        "ending_value_float"
    ].map(lambda value_float: f"${float(value_float):,.0f}")

    calendar_report_df = calendar_df.copy()
    for column_str in (
        "CORE5",
        "CORE5_50_TACTICAL_FI_50",
        "equal_minus_core5_float",
    ):
        calendar_report_df[column_str] = calendar_report_df[column_str].map(
            lambda value_float: f"{100.0 * float(value_float):.2f}%"
        )
    component_report_df = component_df.copy()
    for column_str in component_report_df.columns:
        component_report_df[column_str] = component_report_df[column_str].map(
            lambda value_float: f"{100.0 * float(value_float):.2f}%"
        )
    execution_report_df = execution_reality_df.copy()
    execution_report_df["minimum_cash_float"] = execution_report_df[
        "minimum_cash_float"
    ].map(lambda value_float: f"${float(value_float):,.0f}")
    execution_report_df["minimum_cash_weight_float"] = execution_report_df[
        "minimum_cash_weight_float"
    ].map(lambda value_float: f"{100.0 * float(value_float):.2f}%")
    execution_report_df["slippage_per_side_float"] = execution_report_df[
        "slippage_per_side_float"
    ].map(lambda value_float: f"{10_000.0 * float(value_float):.2f} bps")

    report_path = output_dir_path / "REPORT.md"
    report_text_str = f"""# CORE5 vs 50/50 CORE5 and Tactical FI

## Authority

Research-only comparison. It does not authorize allocation, PAPER, or LIVE.

## Frozen contract

- Cash anchor: {CAPITAL_ANCHOR_DATE_STR}
- First executable session: {EFFECTIVE_EXECUTION_START_DATE_STR}
- End: {END_DATE_STR}
- Capital: ${TOTAL_CAPITAL_FLOAT:,.0f}
- Comparison: CORE5 at $750,000 versus CORE5 at $375,000 plus Tactical FI at $375,000
- Outer rebalance: none; the two subaccounts compound independently
- Timing: Close_T decision to Open_(T+1) execution
- Costs and accounting: each strategy's existing native contract, unchanged
- This runner executes one fixed path, but the 50/50 candidate is historically
  preselected and belongs to a family with at least
  {INHERITED_HISTORICAL_TRIAL_COUNT_FLOOR_INT} prior/countable trials
- Untouched holdout: no

## Headline metrics

{phase0_runner.markdown_table_str(report_headline_df)}

## Component diagnostic

{phase0_runner.markdown_table_str(component_report_df)}

## Execution and financing diagnostic

{phase0_runner.markdown_table_str(execution_report_df)}

## Calendar returns

{phase0_runner.markdown_table_str(calendar_report_df)}

## Interpretation limits

CORE5 and Tactical FI retain different native cash, macro-vintage, and execution
assumptions. Tactical FI uses current-vintage FRED history with modeled release
lags. CORE5's DBC short uses a fixed 1% annual borrow baseline. Both products
can realize negative cash after next-open gaps, slippage, and whole-share
sizing; negative-cash financing is not modeled and therefore overstates NAV
when a deficit exists. The result is historically conditioned and is a
controlled comparison of the current research implementations, not a fresh
holdout or final offered-account accounting certification.
"""
    report_path.write_text(report_text_str, encoding="utf-8", newline="\n")
    return report_path


def run_study(output_dir_path: Path = DEFAULT_OUTPUT_DIR_PATH) -> Path:
    if output_dir_path.exists() and any(output_dir_path.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir_path}")
    output_dir_path.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(
        Path(__file__).resolve(),
        output_dir_path / "executed_runner_frozen.py",
    )
    shutil.copyfile(
        Path(source_runner.__file__).resolve(),
        output_dir_path / "source_execution_runner_frozen.py",
    )
    source_contract_dict = build_source_contract_dict()
    source_runner.write_json(
        output_dir_path / "source_contract.json", source_contract_dict
    )
    norgate_start_dict = source_runner.norgate_database_vintage_dict()
    source_runner.write_json(
        output_dir_path / "norgate_database_vintage_start.json",
        norgate_start_dict,
    )
    source_summary_df = source_runner.execute_source_runs(
        source_contract_dict,
        output_dir_path,
        resume_bool=False,
    )
    norgate_end_dict = source_runner.norgate_database_vintage_dict()
    source_runner.write_json(
        output_dir_path / "norgate_database_vintage_end.json",
        norgate_end_dict,
    )
    if norgate_end_dict != norgate_start_dict:
        raise RuntimeError("Norgate database vintage changed during the study.")

    source_path_by_id_dict = source_runner.load_all_source_path_dict(
        source_contract_dict, output_dir_path
    )
    product_value_df = build_product_value_df(source_path_by_id_dict)
    headline_df = headline_metric_df(product_value_df)
    calendar_df = calendar_return_df(product_value_df)
    component_df = component_diagnostic_df(source_path_by_id_dict)
    execution_df = execution_reality_df(
        source_path_by_id_dict, source_summary_df
    )

    source_runner.write_csv_gzip(
        product_value_df,
        output_dir_path / "product_paths.csv.gz",
        index_bool=True,
        index_label_str="date",
    )
    headline_df.to_csv(output_dir_path / "headline_metrics.csv", index=False)
    calendar_df.to_csv(output_dir_path / "calendar_returns.csv", index=False)
    component_df.to_csv(
        output_dir_path / "component_diagnostics.csv", index=False
    )
    execution_df.to_csv(
        output_dir_path / "execution_reality_diagnostics.csv", index=False
    )
    source_summary_df.to_csv(
        output_dir_path / "source_run_summary.csv", index=False
    )
    report_path = write_report(
        headline_df,
        calendar_df,
        component_df,
        execution_df,
        output_dir_path,
    )
    global_index_sha256_str = sha256_bytes_str(
        "\n".join(
            date_ts.date().isoformat() for date_ts in product_value_df.index
        ).encode("utf-8")
    )
    source_runner.write_json(
        output_dir_path / "run_manifest.json",
        {
            "study_id_str": "core5_vs_tactical_fi_equal_earliest",
            "authority_str": "research_bench_only",
            "runner_sha256_str": source_runner.sha256_file_str(
                Path(__file__).resolve()
            ),
            "frozen_runner_sha256_str": source_runner.sha256_file_str(
                output_dir_path / "executed_runner_frozen.py"
            ),
            "source_execution_runner_sha256_str": source_runner.sha256_file_str(
                Path(source_runner.__file__).resolve()
            ),
            "global_index_sha256_str": global_index_sha256_str,
            "observation_count_int": int(len(product_value_df.index) - 1),
            "execution_path_count_int": 1,
            "inherited_historical_trial_count_floor_int": (
                INHERITED_HISTORICAL_TRIAL_COUNT_FLOOR_INT
            ),
            "untouched_holdout_bool": False,
            "report_path_str": str(report_path),
        },
    )
    return report_path


def parse_args(arg_list: Iterable[str] | None = None) -> argparse.Namespace:
    parser_obj = argparse.ArgumentParser(
        description="Compare CORE5 with 50/50 CORE5 and Tactical FI."
    )
    parser_obj.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR_PATH
    )
    return parser_obj.parse_args(list(arg_list) if arg_list is not None else None)


def main(arg_list: Iterable[str] | None = None) -> int:
    args_obj = parse_args(arg_list)
    report_path = run_study(args_obj.output_dir)
    print(f"study completed: {report_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
