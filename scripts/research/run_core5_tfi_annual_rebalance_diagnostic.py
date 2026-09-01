"""Diagnose annual outer rebalancing for the CORE5/Tactical FI sleeve.

The diagnostic consumes the frozen, exact-capital source paths from the
earliest-history comparison.  It resets the two notional component equities to
50/50 after the final close of each calendar year and before applying the first
common-session return of the next year.  It does not simulate cash transfers,
share rounding, or transfer-driven trades inside the source strategies.
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
SOURCE_ARTIFACT_DIR_PATH = (
    REPO_ROOT_PATH
    / "results"
    / "research"
    / "portfolio"
    / "defensive_sleeve_certification"
    / "2026-08-30_core5_vs_tfi_equal_earliest_rerun2"
)
DEFAULT_OUTPUT_DIR_PATH = (
    REPO_ROOT_PATH
    / "results"
    / "research"
    / "portfolio"
    / "defensive_sleeve_certification"
    / "2026-08-31_core5_tfi_annual_rebalance_diagnostic_rerun2"
)
CAPITAL_ANCHOR_DATE_STR = "2007-08-31"
FIRST_RETURN_DATE_STR = "2007-09-04"
END_DATE_STR = "2026-08-19"
TOTAL_CAPITAL_FLOAT = 750_000.0
TARGET_COMPONENT_WEIGHT_FLOAT = 0.50
INHERITED_HISTORICAL_TRIAL_COUNT_FLOOR_INT = 74


def sha256_file_str(file_path: Path) -> str:
    return hashlib.sha256(file_path.read_bytes()).hexdigest()


def consumed_source_sha256_by_relative_path_dict(
    source_artifact_dir_path: Path,
) -> dict[str, str]:
    consumed_source_relative_path_list = [
        Path("source_paths") / "core5_375000.csv.gz",
        Path("source_paths") / "core5_750000.csv.gz",
        Path("source_paths") / "tactical_fi_375000.csv.gz",
        Path("source_metadata") / "core5_375000.json",
        Path("source_metadata") / "core5_750000.json",
        Path("source_metadata") / "tactical_fi_375000.json",
        Path("norgate_database_vintage_start.json"),
        Path("norgate_database_vintage_end.json"),
    ]
    return {
        relative_path.as_posix(): sha256_file_str(
            source_artifact_dir_path / relative_path
        )
        for relative_path in consumed_source_relative_path_list
    }


def load_validated_source_path_by_id_dict(
    source_artifact_dir_path: Path,
) -> dict[str, pd.DataFrame]:
    norgate_start_dict = json.loads(
        (source_artifact_dir_path / "norgate_database_vintage_start.json").read_text(
            encoding="utf-8"
        )
    )
    norgate_end_dict = json.loads(
        (source_artifact_dir_path / "norgate_database_vintage_end.json").read_text(
            encoding="utf-8"
        )
    )
    if norgate_start_dict != norgate_end_dict:
        raise RuntimeError("The source artifact mixes Norgate vintages.")

    source_path_by_id_dict: dict[str, pd.DataFrame] = {}
    dependency_hash_payload_str_set: set[str] = set()
    for source_id_str in (
        "core5_375000",
        "core5_750000",
        "tactical_fi_375000",
    ):
        metadata_path = (
            source_artifact_dir_path / "source_metadata" / f"{source_id_str}.json"
        )
        source_path = (
            source_artifact_dir_path / "source_paths" / f"{source_id_str}.csv.gz"
        )
        metadata_dict = json.loads(metadata_path.read_text(encoding="utf-8"))
        if sha256_file_str(source_path) != str(
            metadata_dict["source_path_sha256_str"]
        ):
            raise RuntimeError(f"Source hash changed for {source_id_str}.")
        dependency_hash_payload_str_set.add(
            json.dumps(
                metadata_dict["shared_execution_dependency_hash_dict"],
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        source_path_df = source_runner.read_source_path_df(source_path)
        if (
            source_path_df.index[0] != pd.Timestamp(CAPITAL_ANCHOR_DATE_STR)
            or source_path_df.index[1] != pd.Timestamp(FIRST_RETURN_DATE_STR)
            or source_path_df.index[-1] != pd.Timestamp(END_DATE_STR)
        ):
            raise RuntimeError(f"Source dates changed for {source_id_str}.")
        source_path_by_id_dict[source_id_str] = source_path_df
    if len(dependency_hash_payload_str_set) != 1:
        raise RuntimeError("Source runs used different execution-code trees.")
    return source_path_by_id_dict


def build_annual_rebalanced_path_tuple(
    core5_value_ser: pd.Series,
    tactical_fi_value_ser: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    common_idx = core5_value_ser.index.intersection(tactical_fi_value_ser.index)
    common_idx = pd.DatetimeIndex(common_idx).sort_values()
    core5_value_ser = core5_value_ser.reindex(common_idx).astype(float)
    tactical_fi_value_ser = tactical_fi_value_ser.reindex(common_idx).astype(float)
    if core5_value_ser.isna().any() or tactical_fi_value_ser.isna().any():
        raise RuntimeError("Component path intersection contains missing values.")
    if not math.isclose(
        float(core5_value_ser.iloc[0] + tactical_fi_value_ser.iloc[0]),
        TOTAL_CAPITAL_FLOAT,
        rel_tol=0.0,
        abs_tol=1e-6,
    ):
        raise RuntimeError("Component source capital is not $750,000.")

    # *** CRITICAL*** retrospective source-return construction: return at T
    # uses only NAV_T and NAV_(T-1). The annual reset decision is based on the
    # fully known prior close and is applied before this return at T.
    core5_return_ser = core5_value_ser.pct_change(fill_method=None)
    tactical_fi_return_ser = tactical_fi_value_ser.pct_change(fill_method=None)

    core5_notional_float = float(core5_value_ser.iloc[0])
    tactical_fi_notional_float = float(tactical_fi_value_ser.iloc[0])
    row_list: list[dict[str, Any]] = [
        {
            "date": common_idx[0],
            "core5_notional_float": core5_notional_float,
            "tactical_fi_notional_float": tactical_fi_notional_float,
            "total_value_float": core5_notional_float + tactical_fi_notional_float,
        }
    ]
    rebalance_row_list: list[dict[str, Any]] = []
    for position_int in range(1, len(common_idx)):
        previous_date_ts = common_idx[position_int - 1]
        current_date_ts = common_idx[position_int]
        if current_date_ts.year != previous_date_ts.year:
            prior_total_float = core5_notional_float + tactical_fi_notional_float
            target_core5_float = (
                prior_total_float * TARGET_COMPONENT_WEIGHT_FLOAT
            )
            transfer_to_core5_float = target_core5_float - core5_notional_float
            rebalance_row_list.append(
                {
                    "rebalance_effective_date_str": (
                        current_date_ts.date().isoformat()
                    ),
                    "decision_close_date_str": previous_date_ts.date().isoformat(),
                    "prior_core5_weight_float": (
                        core5_notional_float / prior_total_float
                    ),
                    "prior_tactical_fi_weight_float": (
                        tactical_fi_notional_float / prior_total_float
                    ),
                    "transfer_to_core5_float": transfer_to_core5_float,
                    "transfer_to_tactical_fi_float": -transfer_to_core5_float,
                    "prior_total_value_float": prior_total_float,
                }
            )
            core5_notional_float = target_core5_float
            tactical_fi_notional_float = (
                prior_total_float - target_core5_float
            )
        core5_notional_float *= 1.0 + float(core5_return_ser.iloc[position_int])
        tactical_fi_notional_float *= 1.0 + float(
            tactical_fi_return_ser.iloc[position_int]
        )
        row_list.append(
            {
                "date": current_date_ts,
                "core5_notional_float": core5_notional_float,
                "tactical_fi_notional_float": tactical_fi_notional_float,
                "total_value_float": (
                    core5_notional_float + tactical_fi_notional_float
                ),
            }
        )
    annual_path_df = pd.DataFrame(row_list).set_index("date")
    annual_path_df.index = pd.DatetimeIndex(annual_path_df.index, name="date")
    rebalance_event_df = pd.DataFrame(rebalance_row_list)
    return annual_path_df, rebalance_event_df


def build_product_value_df(
    source_path_by_id_dict: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    core5_half_value_ser = source_path_by_id_dict["core5_375000"][
        "total_value_float"
    ]
    tactical_fi_value_ser = source_path_by_id_dict["tactical_fi_375000"][
        "total_value_float"
    ]
    annual_path_df, rebalance_event_df = build_annual_rebalanced_path_tuple(
        core5_half_value_ser,
        tactical_fi_value_ser,
    )
    common_idx = annual_path_df.index
    no_rebalance_value_ser = core5_half_value_ser.reindex(common_idx).add(
        tactical_fi_value_ser.reindex(common_idx)
    )
    core5_full_value_ser = source_path_by_id_dict["core5_750000"][
        "total_value_float"
    ].reindex(common_idx)
    product_value_df = pd.DataFrame(
        {
            "CORE5": core5_full_value_ser,
            "CORE5_50_TFI_50_NO_REBALANCE": no_rebalance_value_ser,
            "CORE5_50_TFI_50_ANNUAL_REBALANCE_DIAGNOSTIC": (
                annual_path_df["total_value_float"]
            ),
        },
        index=common_idx,
    )
    if product_value_df.isna().any().any():
        raise RuntimeError("Product paths contain missing values.")
    return product_value_df, annual_path_df, rebalance_event_df


def headline_metric_df(product_value_df: pd.DataFrame) -> pd.DataFrame:
    row_list: list[dict[str, Any]] = []
    for product_id_str in product_value_df.columns:
        metric_value_dict = phase0_runner.metric_dict(
            product_value_df[product_id_str]
        )
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
                "ending_value_float": float(
                    product_value_df[product_id_str].iloc[-1]
                ),
            }
        )
    return pd.DataFrame(row_list)


def write_report(
    headline_df: pd.DataFrame,
    rebalance_event_df: pd.DataFrame,
    output_dir_path: Path,
) -> Path:
    report_metric_df = headline_df.copy()
    for column_str in (
        "cagr_float",
        "annualized_volatility_float",
        "max_drawdown_float",
        "worst_252d_return_float",
    ):
        report_metric_df[column_str] = report_metric_df[column_str].map(
            lambda value_float: f"{100.0 * float(value_float):.2f}%"
        )
    report_metric_df["sharpe_float"] = report_metric_df["sharpe_float"].map(
        lambda value_float: f"{float(value_float):.3f}"
    )
    report_metric_df["mar_float"] = report_metric_df["mar_float"].map(
        lambda value_float: f"{float(value_float):.3f}"
    )
    report_metric_df["ending_value_float"] = report_metric_df[
        "ending_value_float"
    ].map(lambda value_float: f"${float(value_float):,.0f}")
    report_path = output_dir_path / "REPORT.md"
    report_text_str = f"""# CORE5 and Tactical FI Annual Rebalance Diagnostic

## Authority

Research-only accounting diagnostic. It does not authorize allocation, PAPER,
or LIVE.

## Contract

- Source cash anchor: {CAPITAL_ANCHOR_DATE_STR}
- First source return: {FIRST_RETURN_DATE_STR}
- End: {END_DATE_STR}
- Initial capital: ${TOTAL_CAPITAL_FLOAT:,.0f}
- Annual decision: final common close of each calendar year
- Annual effective date: first common result session of the next year
- Target after each reset: 50% CORE5 and 50% Tactical FI
- Rebalance events: {len(rebalance_event_df)}
- Transfer cost and tax: 0% diagnostic assumption
- Underlying share path after a transfer: not rerun
- Untouched holdout: no
- Inherited historical trial floor: at least {INHERITED_HISTORICAL_TRIAL_COUNT_FLOOR_INT}

## Headline metrics

{phase0_runner.markdown_table_str(report_metric_df)}

## Interpretation limit

The annual path compounds the execution-aligned historical daily returns of
the frozen $375,000 source runs and resets only their notional account values
at year boundaries. Tactical FI uses current-vintage, non-ALFRED FRED history,
so this preserves Close_T to Open_(T+1) execution timing but does not establish
point-in-time macro-vintage causality.
It does not simulate actual inter-account transfers, transfer timing risk,
fractional-share constraints, liquidation costs, taxes, or the different
target shares that changed account capital could cause at a later strategy
rebalance. It is suitable for deciding whether annual weight maintenance is
worth an exact offered-account implementation, not for deployment approval.
"""
    report_path.write_text(report_text_str, encoding="utf-8", newline="\n")
    return report_path


def run_study(
    source_artifact_dir_path: Path = SOURCE_ARTIFACT_DIR_PATH,
    output_dir_path: Path = DEFAULT_OUTPUT_DIR_PATH,
) -> Path:
    if output_dir_path.exists() and any(output_dir_path.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir_path}")
    output_dir_path.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(Path(__file__).resolve(), output_dir_path / "executed_runner.py")
    shutil.copyfile(
        Path(phase0_runner.__file__).resolve(),
        output_dir_path / "metric_report_dependency_frozen.py",
    )
    shutil.copyfile(
        Path(source_runner.__file__).resolve(),
        output_dir_path / "source_io_dependency_frozen.py",
    )
    source_path_by_id_dict = load_validated_source_path_by_id_dict(
        source_artifact_dir_path
    )
    product_value_df, annual_component_df, rebalance_event_df = (
        build_product_value_df(source_path_by_id_dict)
    )
    headline_df = headline_metric_df(product_value_df)
    source_runner.write_csv_gzip(
        product_value_df,
        output_dir_path / "product_paths.csv.gz",
        index_bool=True,
        index_label_str="date",
    )
    source_runner.write_csv_gzip(
        annual_component_df,
        output_dir_path / "annual_component_paths.csv.gz",
        index_bool=True,
        index_label_str="date",
    )
    headline_df.to_csv(output_dir_path / "headline_metrics.csv", index=False)
    rebalance_event_df.to_csv(
        output_dir_path / "annual_rebalance_events.csv", index=False
    )
    report_path = write_report(headline_df, rebalance_event_df, output_dir_path)
    consumed_source_hash_dict = consumed_source_sha256_by_relative_path_dict(
        source_artifact_dir_path
    )
    source_runner.write_json(
        output_dir_path / "run_manifest.json",
        {
            "study_id_str": "core5_tfi_annual_rebalance_diagnostic",
            "authority_str": "research_accounting_diagnostic_only",
            "source_artifact_dir_str": str(source_artifact_dir_path),
            "consumed_source_sha256_by_relative_path_dict": (
                consumed_source_hash_dict
            ),
            "runner_sha256_str": sha256_file_str(Path(__file__).resolve()),
            "frozen_runner_sha256_str": sha256_file_str(
                output_dir_path / "executed_runner.py"
            ),
            "analytical_dependency_sha256_by_name_dict": {
                "run_defensive_sleeve_v3_phase0.py": sha256_file_str(
                    Path(phase0_runner.__file__).resolve()
                ),
                "run_ladder4_candidate_value_add_study.py": sha256_file_str(
                    Path(source_runner.__file__).resolve()
                ),
            },
            "frozen_analytical_dependency_sha256_by_name_dict": {
                "metric_report_dependency_frozen.py": sha256_file_str(
                    output_dir_path / "metric_report_dependency_frozen.py"
                ),
                "source_io_dependency_frozen.py": sha256_file_str(
                    output_dir_path / "source_io_dependency_frozen.py"
                ),
            },
            "annual_rebalance_event_count_int": int(len(rebalance_event_df)),
            "inherited_historical_trial_count_floor_int": (
                INHERITED_HISTORICAL_TRIAL_COUNT_FLOOR_INT
            ),
            "untouched_holdout_bool": False,
        },
    )
    return report_path


def parse_args(arg_list: Iterable[str] | None = None) -> argparse.Namespace:
    parser_obj = argparse.ArgumentParser(
        description="Diagnose annual rebalancing of CORE5 and Tactical FI."
    )
    parser_obj.add_argument(
        "--source-artifact-dir", type=Path, default=SOURCE_ARTIFACT_DIR_PATH
    )
    parser_obj.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR_PATH
    )
    return parser_obj.parse_args(list(arg_list) if arg_list is not None else None)


def main(arg_list: Iterable[str] | None = None) -> int:
    args_obj = parse_args(arg_list)
    report_path = run_study(
        args_obj.source_artifact_dir,
        args_obj.output_dir,
    )
    print(f"study completed: {report_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
