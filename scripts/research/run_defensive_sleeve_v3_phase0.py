"""Run the frozen V3 defensive-sleeve Phase-0 comparison.

This runner executes each required strategy/capital pair once, anchors every
source in cash on 2012-09-28, and sums independently compounded subaccounts.
It is research-only and deliberately cannot authorize allocation, PAPER, or
LIVE use because the offered-account accounting adapter is not yet applied.
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
import yaml

from scripts.research import run_ladder4_candidate_value_add_study as ladder_runner


REPO_ROOT_PATH = Path(__file__).resolve().parents[2]
DEFAULT_SPEC_PATH = (
    REPO_ROOT_PATH
    / "scripts"
    / "research"
    / "specs"
    / "defensive_sleeve_certification_v3.yaml"
)
DEFAULT_FREEZE_PATH = DEFAULT_SPEC_PATH.with_suffix(".freeze.json")
DEFAULT_OUTPUT_DIR_PATH = (
    REPO_ROOT_PATH
    / "results"
    / "research"
    / "portfolio"
    / "defensive_sleeve_certification"
    / "2026-08-30_v3_phase0"
)
SOURCE_IMPORT_BY_ALIAS_DICT = {
    "core5": "strategies.taa_beyond_6040.strategy_taa_adaptive_macro_core5",
    "tactical_fi": (
        "strategies.taa_beyond_6040."
        "strategy_taa_tactical_fixed_income_ief_lqd"
    ),
    "inflation_compass": "strategies.taa_df.strategy_taa_inflation_compass",
    "bil": "strategies.portfolio_controls.strategy_passive_bil",
}
NATIVE_HISTORY_START_BY_IMPORT_DICT = {
    SOURCE_IMPORT_BY_ALIAS_DICT["core5"]: "1990-01-01",
    SOURCE_IMPORT_BY_ALIAS_DICT["tactical_fi"]: "2002-07-26",
    SOURCE_IMPORT_BY_ALIAS_DICT["inflation_compass"]: "2002-01-01",
    SOURCE_IMPORT_BY_ALIAS_DICT["bil"]: "2004-01-01",
}
CAPITAL_ANCHOR_DATE_STR = "2012-09-28"
EFFECTIVE_EXECUTION_START_DATE_STR = "2012-10-01"
FROZEN_END_DATE_STR = "2026-08-19"
TOTAL_CLIENT_CAPITAL_FLOAT = 1_000_000.0
DEFENSIVE_SLEEVE_CAPITAL_FLOAT = 750_000.0
TRADING_DAY_COUNT_PER_YEAR_INT = 252
INFLATION_WINDOW_START_DATE_STR = "2022-01-03"
INFLATION_WINDOW_END_DATE_STR = "2022-10-12"
MINIMUM_NET_CAGR_FLOAT = 0.04
MINIMUM_SHARPE_IMPROVEMENT_FLOAT = 0.02
MAXIMUM_PARENT_RISK_WORSENING_FLOAT = 0.005
MINIMUM_INFLATION_RETURN_IMPROVEMENT_FLOAT = 0.005


def sha256_bytes_str(payload_bytes: bytes) -> str:
    return hashlib.sha256(payload_bytes).hexdigest()


def load_and_validate_frozen_contract(
    spec_path: Path,
    freeze_path: Path,
) -> dict[str, Any]:
    freeze_dict = json.loads(freeze_path.read_text(encoding="utf-8"))
    actual_spec_sha256_str = ladder_runner.sha256_file_str(spec_path)
    if actual_spec_sha256_str != str(freeze_dict["spec_sha256_str"]):
        raise RuntimeError("V3 spec hash does not match its freeze record.")
    spec_dict = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    if str(spec_dict["study_id_str"]) != "defensive_sleeve_certification_v3":
        raise ValueError("Unexpected V3 study id.")
    if str(spec_dict["authority_str"]) != "research_bench_only":
        raise ValueError("V3 must remain research-only.")
    if not bool(
        spec_dict["decision_rule"][
            "no_paper_live_release_or_allocation_authority_bool"
        ]
    ):
        raise ValueError("V3 authority guard is missing.")
    relative_gate_dict = spec_dict["v3_relative_gates"]
    expected_float_by_field_dict = {
        "minimum_defensive_sleeve_net_cagr_float": MINIMUM_NET_CAGR_FLOAT,
        "minimum_primary_sharpe_improvement_float": (
            MINIMUM_SHARPE_IMPROVEMENT_FLOAT
        ),
        "maximum_max_drawdown_worsening_vs_named_parent_float": (
            MAXIMUM_PARENT_RISK_WORSENING_FLOAT
        ),
        "maximum_worst_252d_worsening_vs_named_parent_float": (
            MAXIMUM_PARENT_RISK_WORSENING_FLOAT
        ),
    }
    for field_str, expected_float in expected_float_by_field_dict.items():
        if not math.isclose(
            float(relative_gate_dict[field_str]),
            expected_float,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(f"Runner gate differs from frozen {field_str}.")
    inflation_window_dict = relative_gate_dict["inflation_mechanism_window"]
    if (
        str(inflation_window_dict["start_date_str"])
        != INFLATION_WINDOW_START_DATE_STR
        or str(inflation_window_dict["end_date_str"])
        != INFLATION_WINDOW_END_DATE_STR
        or not math.isclose(
            float(
                inflation_window_dict[
                    "minimum_total_return_improvement_vs_named_parent_float"
                ]
            ),
            MINIMUM_INFLATION_RETURN_IMPROVEMENT_FLOAT,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or not math.isclose(
            float(
                inflation_window_dict[
                    "maximum_max_drawdown_worsening_vs_named_parent_float"
                ]
            ),
            MAXIMUM_PARENT_RISK_WORSENING_FLOAT,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise ValueError("Runner inflation gate differs from the frozen V3 contract.")
    expected_primary_comparator_by_candidate_dict = {
        "DR3_core5_inflation_compass_05": str(
            relative_gate_dict["DR3_primary_comparator_str"]
        ),
        "DR4_core5_tactical_fi_05": str(
            relative_gate_dict["DR4_primary_comparator_str"]
        ),
        "DR5_core5_tactical_fi_equal_inflation_compass_05": str(
            relative_gate_dict["DR5_primary_comparator_str"]
        ),
    }
    expected_primary_comparator_dict = {
        "DR3_core5_inflation_compass_05": "MC1_core5_bil_05",
        "DR4_core5_tactical_fi_05": "B1_core5",
        "DR5_core5_tactical_fi_equal_inflation_compass_05": (
            "MC2_core5_tactical_fi_equal_bil_05"
        ),
    }
    if expected_primary_comparator_by_candidate_dict != (
        expected_primary_comparator_dict
    ):
        raise ValueError("Runner comparators differ from the frozen V3 contract.")
    expected_parent_dict = {
        "DR3_core5_inflation_compass_05": "B1_core5",
        "DR4_core5_tactical_fi_05": "B1_core5",
        "DR5_core5_tactical_fi_equal_inflation_compass_05": (
            "DR2_core5_tfi_equal"
        ),
    }
    actual_parent_dict = {
        str(candidate_id_str): str(parent_id_str)
        for candidate_id_str, parent_id_str in spec_dict[
            "comparison_parent_by_candidate"
        ].items()
    }
    if actual_parent_dict != expected_parent_dict:
        raise ValueError("Runner parents differ from the frozen V3 contract.")
    return spec_dict


def source_id_str(alias_str: str, capital_float: float) -> str:
    rounded_capital_int = int(round(capital_float))
    if not math.isclose(
        float(rounded_capital_int), capital_float, rel_tol=0.0, abs_tol=1e-9
    ):
        raise ValueError("Source capital must be a whole dollar.")
    return f"{alias_str}_{rounded_capital_int}"


def product_weight_by_alias_dict(spec_dict: dict[str, Any]) -> dict[str, dict[str, float]]:
    product_weight_dict: dict[str, dict[str, float]] = {
        "B1_core5": {"core5": 1.0},
        "DR2_core5_tfi_equal": {"core5": 0.50, "tactical_fi": 0.50},
    }
    for candidate_id_str, candidate_dict in spec_dict[
        "v3_candidate_addition"
    ].items():
        product_weight_dict[str(candidate_id_str)] = {
            str(alias_str): float(weight_obj)
            for alias_str, weight_obj in candidate_dict[
                "defensive_sleeve_weight_by_alias"
            ].items()
        }
    for control_id_str, control_dict in spec_dict["matched_controls"].items():
        product_weight_dict[str(control_id_str)] = {
            str(alias_str): float(weight_obj)
            for alias_str, weight_obj in control_dict[
                "defensive_sleeve_weight_by_alias"
            ].items()
        }
    for product_id_str, weight_by_alias_dict in product_weight_dict.items():
        weight_sum_float = sum(weight_by_alias_dict.values())
        if not math.isclose(weight_sum_float, 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(
                f"{product_id_str} defensive weights sum to {weight_sum_float:.12f}."
            )
    return product_weight_dict


def expanded_source_contract_tuple(
    spec_dict: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, dict[str, str]]]:
    product_weight_dict = product_weight_by_alias_dict(spec_dict)
    source_run_by_id_dict: dict[str, dict[str, Any]] = {}
    source_id_by_product_alias_dict: dict[str, dict[str, str]] = {}
    for product_id_str, weight_by_alias_dict in product_weight_dict.items():
        source_id_by_product_alias_dict[product_id_str] = {}
        for alias_str, weight_float in weight_by_alias_dict.items():
            capital_float = DEFENSIVE_SLEEVE_CAPITAL_FLOAT * weight_float
            current_source_id_str = source_id_str(alias_str, capital_float)
            source_id_by_product_alias_dict[product_id_str][alias_str] = (
                current_source_id_str
            )
            run_variant_kwargs_dict: dict[str, Any] = {}
            if alias_str == "bil":
                run_variant_kwargs_dict["accounting_profile_str"] = (
                    "core5_net25_zero_cash"
                )
            source_spec_dict = {
                "strategy_import_str": SOURCE_IMPORT_BY_ALIAS_DICT[alias_str],
                "allocated_capital_float": capital_float,
                "run_variant_kwargs_dict": run_variant_kwargs_dict,
            }
            existing_source_spec_dict = source_run_by_id_dict.get(
                current_source_id_str
            )
            if (
                existing_source_spec_dict is not None
                and existing_source_spec_dict != source_spec_dict
            ):
                raise RuntimeError(f"Conflicting source {current_source_id_str}.")
            source_run_by_id_dict[current_source_id_str] = source_spec_dict

    expanded_spec_dict = {
        "portfolio_contract": {
            "requested_start_date_str": "2004-01-01",
            "capital_anchor_date_str": CAPITAL_ANCHOR_DATE_STR,
            "effective_execution_start_date_str": (
                EFFECTIVE_EXECUTION_START_DATE_STR
            ),
            "end_date_str": FROZEN_END_DATE_STR,
        },
        "lineage_contract": {
            "native_history_request_start_by_strategy_import": (
                NATIVE_HISTORY_START_BY_IMPORT_DICT
            )
        },
        "source_runs": dict(sorted(source_run_by_id_dict.items())),
    }
    return expanded_spec_dict, source_id_by_product_alias_dict


def build_product_total_value_df(
    expanded_spec_dict: dict[str, Any],
    source_id_by_product_alias_dict: dict[str, dict[str, str]],
    output_dir_path: Path,
) -> tuple[pd.DataFrame, str]:
    source_path_by_id_dict = ladder_runner.load_all_source_path_dict(
        expanded_spec_dict,
        output_dir_path,
    )
    global_idx: pd.DatetimeIndex | None = None
    for source_path_df in source_path_by_id_dict.values():
        source_idx = pd.DatetimeIndex(source_path_df.index)
        global_idx = source_idx if global_idx is None else global_idx.intersection(source_idx)
    if global_idx is None or len(global_idx) < 2:
        raise RuntimeError("No usable exact source intersection.")
    global_idx = global_idx.sort_values()
    if global_idx[0] != pd.Timestamp(CAPITAL_ANCHOR_DATE_STR):
        raise RuntimeError("Global source intersection has the wrong cash anchor.")
    if global_idx[-1] != pd.Timestamp(FROZEN_END_DATE_STR):
        raise RuntimeError("Global source intersection has the wrong endpoint.")

    total_value_by_product_dict: dict[str, pd.Series] = {}
    for product_id_str, source_id_by_alias_dict in (
        source_id_by_product_alias_dict.items()
    ):
        product_total_value_ser = pd.Series(0.0, index=global_idx)
        for current_source_id_str in source_id_by_alias_dict.values():
            source_path_df = source_path_by_id_dict[current_source_id_str].reindex(
                global_idx
            )
            if source_path_df.isna().any().any():
                raise RuntimeError(f"{product_id_str} has missing source values.")
            product_total_value_ser = product_total_value_ser.add(
                source_path_df["total_value_float"].astype(float),
                fill_value=0.0,
            )
        if not math.isclose(
            float(product_total_value_ser.iloc[0]),
            DEFENSIVE_SLEEVE_CAPITAL_FLOAT,
            rel_tol=0.0,
            abs_tol=1e-6,
        ):
            raise RuntimeError(f"{product_id_str} has the wrong anchor capital.")
        total_value_by_product_dict[product_id_str] = product_total_value_ser
    total_value_df = pd.DataFrame(total_value_by_product_dict, index=global_idx)
    total_value_df.index.name = "date"
    global_index_sha256_str = sha256_bytes_str(
        "\n".join(date_ts.date().isoformat() for date_ts in global_idx).encode(
            "utf-8"
        )
    )
    return total_value_df, global_index_sha256_str


def validate_uniform_source_execution_hashes(
    expanded_spec_dict: dict[str, Any],
    output_dir_path: Path,
) -> str:
    """Require one identical executed Python tree across every source run."""

    canonical_hash_payload_str_set: set[str] = set()
    for current_source_id_str in expanded_spec_dict["source_runs"]:
        metadata_path = (
            output_dir_path / "source_metadata" / f"{current_source_id_str}.json"
        )
        metadata_dict = json.loads(metadata_path.read_text(encoding="utf-8"))
        hash_dict = metadata_dict["shared_execution_dependency_hash_dict"]
        canonical_hash_payload_str_set.add(
            json.dumps(hash_dict, sort_keys=True, separators=(",", ":"))
        )
    if len(canonical_hash_payload_str_set) != 1:
        raise RuntimeError("Executed Python tree changed between Phase-0 sources.")
    canonical_hash_payload_str = next(iter(canonical_hash_payload_str_set))
    current_hash_payload_str = json.dumps(
        ladder_runner.shared_execution_dependency_hash_dict(),
        sort_keys=True,
        separators=(",", ":"),
    )
    if current_hash_payload_str != canonical_hash_payload_str:
        raise RuntimeError("Executed Python tree changed before Phase-0 analysis.")
    return sha256_bytes_str(canonical_hash_payload_str.encode("utf-8"))


def metric_dict(total_value_ser: pd.Series) -> dict[str, float]:
    clean_total_value_ser = total_value_ser.astype(float).dropna()
    # *** CRITICAL*** retrospective metric only: pct_change at row T uses NAV_T
    # and NAV_(T-1). It never feeds a decision or an execution in this runner.
    return_ser = clean_total_value_ser.pct_change(fill_method=None).iloc[1:]
    observation_count_int = len(return_ser)
    cagr_float = (
        float(clean_total_value_ser.iloc[-1] / clean_total_value_ser.iloc[0])
        ** (TRADING_DAY_COUNT_PER_YEAR_INT / observation_count_int)
        - 1.0
    )
    annualized_volatility_float = float(
        return_ser.std(ddof=1) * np.sqrt(TRADING_DAY_COUNT_PER_YEAR_INT)
    )
    sharpe_float = (
        float(return_ser.mean() / return_ser.std(ddof=1))
        * np.sqrt(TRADING_DAY_COUNT_PER_YEAR_INT)
        if float(return_ser.std(ddof=1)) > 0.0
        else float("nan")
    )
    drawdown_ser = clean_total_value_ser.div(clean_total_value_ser.cummax()).sub(1.0)
    # *** CRITICAL*** backward-only robustness metric: the value at T is divided
    # by the value exactly 252 result sessions earlier. No future row is used.
    rolling_252_return_ser = clean_total_value_ser.div(
        clean_total_value_ser.shift(TRADING_DAY_COUNT_PER_YEAR_INT)
    ).sub(1.0)
    return {
        "cagr_float": cagr_float,
        "annualized_volatility_float": annualized_volatility_float,
        "sharpe_float": sharpe_float,
        "max_drawdown_float": float(drawdown_ser.min()),
        "worst_252d_return_float": float(rolling_252_return_ser.min()),
        "observation_count_int": float(observation_count_int),
    }


def headline_metric_df(total_value_df: pd.DataFrame) -> pd.DataFrame:
    row_list: list[dict[str, Any]] = []
    for product_id_str in total_value_df.columns:
        row_list.append(
            {
                "product_id_str": product_id_str,
                **metric_dict(total_value_df[product_id_str]),
            }
        )
    return pd.DataFrame(row_list)


def inflation_window_metric_df(total_value_df: pd.DataFrame) -> pd.DataFrame:
    result_date_idx = total_value_df.loc[
        INFLATION_WINDOW_START_DATE_STR:INFLATION_WINDOW_END_DATE_STR
    ].index
    if len(result_date_idx) < 2:
        raise RuntimeError("Frozen 2022 inflation window is unavailable.")
    first_result_position_int = int(total_value_df.index.get_loc(result_date_idx[0]))
    if first_result_position_int < 1:
        raise RuntimeError("Frozen 2022 window has no preceding NAV anchor.")
    window_total_value_df = total_value_df.iloc[
        first_result_position_int - 1 : int(total_value_df.index.get_loc(result_date_idx[-1])) + 1
    ]
    row_list: list[dict[str, Any]] = []
    for product_id_str in total_value_df.columns:
        product_value_ser = window_total_value_df[product_id_str].astype(float)
        drawdown_ser = product_value_ser.div(product_value_ser.cummax()).sub(1.0)
        row_list.append(
            {
                "product_id_str": product_id_str,
                "anchor_date_str": product_value_ser.index[0].date().isoformat(),
                "start_date_str": product_value_ser.index[1].date().isoformat(),
                "end_date_str": product_value_ser.index[-1].date().isoformat(),
                "total_return_float": float(
                    product_value_ser.iloc[-1] / product_value_ser.iloc[0] - 1.0
                ),
                "max_drawdown_float": float(drawdown_ser.min()),
            }
        )
    return pd.DataFrame(row_list)


def candidate_gate_df(
    headline_df: pd.DataFrame,
    inflation_df: pd.DataFrame,
) -> pd.DataFrame:
    headline_by_id_df = headline_df.set_index("product_id_str")
    inflation_by_id_df = inflation_df.set_index("product_id_str")
    gate_contract_by_candidate_dict = {
        "DR3_core5_inflation_compass_05": {
            "primary_comparator_str": "MC1_core5_bil_05",
            "parent_str": "B1_core5",
            "inflation_gate_bool": True,
        },
        "DR4_core5_tactical_fi_05": {
            "primary_comparator_str": "B1_core5",
            "parent_str": "B1_core5",
            "inflation_gate_bool": False,
        },
        "DR5_core5_tactical_fi_equal_inflation_compass_05": {
            "primary_comparator_str": "MC2_core5_tactical_fi_equal_bil_05",
            "parent_str": "DR2_core5_tfi_equal",
            "inflation_gate_bool": True,
        },
    }
    gate_row_list: list[dict[str, Any]] = []
    for candidate_id_str, gate_contract_dict in gate_contract_by_candidate_dict.items():
        primary_comparator_str = str(gate_contract_dict["primary_comparator_str"])
        parent_str = str(gate_contract_dict["parent_str"])
        candidate_headline_ser = headline_by_id_df.loc[candidate_id_str]
        comparator_headline_ser = headline_by_id_df.loc[primary_comparator_str]
        parent_headline_ser = headline_by_id_df.loc[parent_str]
        cagr_gate_bool = (
            float(candidate_headline_ser["cagr_float"])
            >= MINIMUM_NET_CAGR_FLOAT
        )
        sharpe_delta_float = float(
            candidate_headline_ser["sharpe_float"]
            - comparator_headline_ser["sharpe_float"]
        )
        sharpe_gate_bool = (
            sharpe_delta_float >= MINIMUM_SHARPE_IMPROVEMENT_FLOAT
        )
        max_drawdown_delta_vs_parent_float = float(
            candidate_headline_ser["max_drawdown_float"]
            - parent_headline_ser["max_drawdown_float"]
        )
        max_drawdown_gate_bool = (
            max_drawdown_delta_vs_parent_float
            >= -MAXIMUM_PARENT_RISK_WORSENING_FLOAT
        )
        worst_252d_delta_vs_parent_float = float(
            candidate_headline_ser["worst_252d_return_float"]
            - parent_headline_ser["worst_252d_return_float"]
        )
        worst_252d_gate_bool = (
            worst_252d_delta_vs_parent_float
            >= -MAXIMUM_PARENT_RISK_WORSENING_FLOAT
        )

        inflation_gate_required_bool = bool(
            gate_contract_dict["inflation_gate_bool"]
        )
        inflation_return_delta_float = float("nan")
        inflation_drawdown_delta_float = float("nan")
        inflation_return_gate_bool = True
        inflation_drawdown_gate_bool = True
        if inflation_gate_required_bool:
            candidate_inflation_ser = inflation_by_id_df.loc[candidate_id_str]
            parent_inflation_ser = inflation_by_id_df.loc[parent_str]
            inflation_return_delta_float = float(
                candidate_inflation_ser["total_return_float"]
                - parent_inflation_ser["total_return_float"]
            )
            inflation_drawdown_delta_float = float(
                candidate_inflation_ser["max_drawdown_float"]
                - parent_inflation_ser["max_drawdown_float"]
            )
            inflation_return_gate_bool = (
                inflation_return_delta_float
                >= MINIMUM_INFLATION_RETURN_IMPROVEMENT_FLOAT
            )
            inflation_drawdown_gate_bool = (
                inflation_drawdown_delta_float
                >= -MAXIMUM_PARENT_RISK_WORSENING_FLOAT
            )

        numeric_gate_bool = all(
            [
                cagr_gate_bool,
                sharpe_gate_bool,
                max_drawdown_gate_bool,
                worst_252d_gate_bool,
                inflation_return_gate_bool,
                inflation_drawdown_gate_bool,
            ]
        )
        gate_row_list.append(
            {
                "candidate_id_str": candidate_id_str,
                "primary_comparator_str": primary_comparator_str,
                "parent_str": parent_str,
                "cagr_gate_bool": cagr_gate_bool,
                "sharpe_delta_float": sharpe_delta_float,
                "sharpe_gate_bool": sharpe_gate_bool,
                "max_drawdown_delta_vs_parent_float": (
                    max_drawdown_delta_vs_parent_float
                ),
                "max_drawdown_gate_bool": max_drawdown_gate_bool,
                "worst_252d_delta_vs_parent_float": (
                    worst_252d_delta_vs_parent_float
                ),
                "worst_252d_gate_bool": worst_252d_gate_bool,
                "inflation_gate_required_bool": inflation_gate_required_bool,
                "inflation_return_delta_float": inflation_return_delta_float,
                "inflation_return_gate_bool": inflation_return_gate_bool,
                "inflation_drawdown_delta_float": inflation_drawdown_delta_float,
                "inflation_drawdown_gate_bool": inflation_drawdown_gate_bool,
                "phase0_numeric_gate_bool": numeric_gate_bool,
                "formal_accounting_gate_bool": False,
                "decision_str": (
                    "phase0_numeric_pass_accounting_blocked"
                    if numeric_gate_bool
                    else "phase0_rejected"
                ),
            }
        )
    return pd.DataFrame(gate_row_list)


def markdown_table_str(frame_df: pd.DataFrame) -> str:
    """Render a small deterministic Markdown table without optional packages."""

    def cell_str(value_obj: Any) -> str:
        if pd.isna(value_obj):
            return ""
        if isinstance(value_obj, (bool, np.bool_)):
            return "true" if bool(value_obj) else "false"
        if isinstance(value_obj, (float, np.floating)):
            return f"{float(value_obj):.6g}"
        return str(value_obj).replace("|", "\\|").replace("\n", " ")

    column_list = [str(column_obj) for column_obj in frame_df.columns]
    row_str_list = [
        "| " + " | ".join(column_list) + " |",
        "| " + " | ".join("---" for _ in column_list) + " |",
    ]
    for row_tuple in frame_df.itertuples(index=False, name=None):
        row_str_list.append(
            "| " + " | ".join(cell_str(value_obj) for value_obj in row_tuple) + " |"
        )
    return "\n".join(row_str_list)


def write_report(
    headline_df: pd.DataFrame,
    inflation_df: pd.DataFrame,
    gate_df: pd.DataFrame,
    output_dir_path: Path,
) -> Path:
    headline_report_df = headline_df.copy()
    for column_str in [
        "cagr_float",
        "annualized_volatility_float",
        "max_drawdown_float",
        "worst_252d_return_float",
    ]:
        headline_report_df[column_str] = headline_report_df[column_str].map(
            lambda value_float: f"{100.0 * float(value_float):.2f}%"
        )
    headline_report_df["sharpe_float"] = headline_report_df["sharpe_float"].map(
        lambda value_float: f"{float(value_float):.3f}"
    )
    report_path = output_dir_path / "REPORT.md"
    report_text_str = f"""# Defensive Sleeve V3 Phase 0

## Authority

Research-only diagnostic. The source-native accounting profiles are not the
final offered-account adapter, so no candidate can be approved for allocation,
PAPER, or LIVE use from this run.

## Window

- Cash anchor: {CAPITAL_ANCHOR_DATE_STR}
- First executable session: {EFFECTIVE_EXECUTION_START_DATE_STR}
- Frozen end: {FROZEN_END_DATE_STR}
- Defensive sleeve capital: ${DEFENSIVE_SLEEVE_CAPITAL_FLOAT:,.0f}
- Outer rebalance: none; subaccounts compound independently

## Headline metrics

{markdown_table_str(headline_report_df)}

## Frozen 2022 inflation window

{markdown_table_str(inflation_df)}

## Candidate gates

{markdown_table_str(gate_df)}

## Interpretation limit

Passing the numeric Phase-0 gates means only that the path deserves the exact
offered-account rerun. Current-vintage FRED history, source-native cash and
financing differences, whole-share path dependence, and one realized inflation
bear remain material limitations.
"""
    report_path.write_text(report_text_str, encoding="utf-8", newline="\n")
    return report_path


def prepare_output_dir(
    output_dir_path: Path,
    spec_path: Path,
    freeze_path: Path,
    *,
    resume_bool: bool,
) -> None:
    frozen_spec_path = output_dir_path / "research_spec_frozen.yaml"
    frozen_record_path = output_dir_path / "research_freeze_record.json"
    if resume_bool:
        if not frozen_spec_path.is_file() or not frozen_record_path.is_file():
            raise FileNotFoundError("Resume requires the frozen spec and freeze record.")
        if frozen_spec_path.read_bytes() != spec_path.read_bytes():
            raise RuntimeError("Resume spec differs from the saved frozen spec.")
        if frozen_record_path.read_bytes() != freeze_path.read_bytes():
            raise RuntimeError("Resume freeze record differs from the saved record.")
        if (
            output_dir_path / "executed_phase0_runner_frozen.py"
        ).read_bytes() != Path(__file__).resolve().read_bytes():
            raise RuntimeError("Current Phase-0 runner differs from the executed copy.")
        return
    if output_dir_path.exists() and any(output_dir_path.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir_path}")
    output_dir_path.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(spec_path, frozen_spec_path)
    shutil.copyfile(freeze_path, frozen_record_path)
    shutil.copyfile(
        Path(__file__).resolve(),
        output_dir_path / "executed_phase0_runner_frozen.py",
    )
    shutil.copyfile(
        Path(ladder_runner.__file__).resolve(),
        output_dir_path / "source_execution_runner_frozen.py",
    )


def run_study(
    spec_path: Path = DEFAULT_SPEC_PATH,
    freeze_path: Path = DEFAULT_FREEZE_PATH,
    output_dir_path: Path = DEFAULT_OUTPUT_DIR_PATH,
    *,
    resume_bool: bool = False,
) -> Path:
    spec_dict = load_and_validate_frozen_contract(spec_path, freeze_path)
    expanded_spec_dict, source_id_by_product_alias_dict = (
        expanded_source_contract_tuple(spec_dict)
    )
    prepare_output_dir(
        output_dir_path,
        spec_path,
        freeze_path,
        resume_bool=resume_bool,
    )
    runner_start_sha256_str = ladder_runner.sha256_file_str(Path(__file__).resolve())
    spec_sha256_str = ladder_runner.sha256_file_str(spec_path)
    ladder_runner.write_json(
        output_dir_path / "expanded_source_contract.json",
        {
            "source_runs": expanded_spec_dict["source_runs"],
            "source_id_by_product_alias_dict": source_id_by_product_alias_dict,
            "source_count_int": len(expanded_spec_dict["source_runs"]),
        },
    )
    norgate_start_path = output_dir_path / "norgate_database_vintage_start.json"
    current_norgate_dict = ladder_runner.norgate_database_vintage_dict()
    if resume_bool:
        norgate_start_dict = json.loads(
            norgate_start_path.read_text(encoding="utf-8")
        )
        if current_norgate_dict != norgate_start_dict:
            raise RuntimeError("Norgate vintage changed before Phase-0 resume.")
    else:
        norgate_start_dict = current_norgate_dict
        ladder_runner.write_json(norgate_start_path, norgate_start_dict)
    source_run_summary_df = ladder_runner.execute_source_runs(
        expanded_spec_dict,
        output_dir_path,
        resume_bool=resume_bool,
    )
    norgate_end_dict = ladder_runner.norgate_database_vintage_dict()
    ladder_runner.write_json(
        output_dir_path / "norgate_database_vintage_end.json",
        norgate_end_dict,
    )
    if norgate_end_dict != norgate_start_dict:
        raise RuntimeError("Norgate database vintage changed during Phase 0.")
    if ladder_runner.sha256_file_str(Path(__file__).resolve()) != runner_start_sha256_str:
        raise RuntimeError("Phase-0 runner changed during execution.")
    if ladder_runner.sha256_file_str(spec_path) != spec_sha256_str:
        raise RuntimeError("Frozen V3 spec changed during execution.")
    source_execution_hash_sha256_str = validate_uniform_source_execution_hashes(
        expanded_spec_dict,
        output_dir_path,
    )

    total_value_df, global_index_sha256_str = build_product_total_value_df(
        expanded_spec_dict,
        source_id_by_product_alias_dict,
        output_dir_path,
    )
    headline_df = headline_metric_df(total_value_df)
    inflation_df = inflation_window_metric_df(total_value_df)
    gate_df = candidate_gate_df(headline_df, inflation_df)
    ladder_runner.write_csv_gzip(
        total_value_df,
        output_dir_path / "global_defensive_paths.csv.gz",
        index_bool=True,
        index_label_str="date",
    )
    source_run_summary_df.to_csv(
        output_dir_path / "source_run_summary.csv",
        index=False,
        float_format="%.12g",
        lineterminator="\n",
    )
    headline_df.to_csv(
        output_dir_path / "headline_metrics.csv",
        index=False,
        float_format="%.12g",
        lineterminator="\n",
    )
    inflation_df.to_csv(
        output_dir_path / "inflation_2022_metrics.csv",
        index=False,
        float_format="%.12g",
        lineterminator="\n",
    )
    gate_df.to_csv(
        output_dir_path / "candidate_gates.csv",
        index=False,
        float_format="%.12g",
        lineterminator="\n",
    )
    report_path = write_report(
        headline_df,
        inflation_df,
        gate_df,
        output_dir_path,
    )
    manifest_dict = {
        "study_id_str": "defensive_sleeve_certification_v3_phase0",
        "authority_str": "research_only_diagnostic",
        "formal_accounting_adapter_applied_bool": False,
        "spec_sha256_str": spec_sha256_str,
        "freeze_sha256_str": ladder_runner.sha256_file_str(freeze_path),
        "runner_sha256_str": runner_start_sha256_str,
        "executed_runner_copy_sha256_str": ladder_runner.sha256_file_str(
            output_dir_path / "executed_phase0_runner_frozen.py"
        ),
        "source_execution_runner_copy_sha256_str": ladder_runner.sha256_file_str(
            output_dir_path / "source_execution_runner_frozen.py"
        ),
        "global_index_sha256_str": global_index_sha256_str,
        "source_execution_hash_sha256_str": source_execution_hash_sha256_str,
        "norgate_start_dict": norgate_start_dict,
        "norgate_end_dict": norgate_end_dict,
        "source_count_int": len(expanded_spec_dict["source_runs"]),
        "phase0_decision_by_candidate_dict": {
            str(row_ser["candidate_id_str"]): str(row_ser["decision_str"])
            for _, row_ser in gate_df.iterrows()
        },
    }
    ladder_runner.write_json(output_dir_path / "run_manifest.json", manifest_dict)
    return report_path


def parse_args(arg_list: Iterable[str] | None = None) -> argparse.Namespace:
    parser_obj = argparse.ArgumentParser(
        description="Run the frozen defensive-sleeve V3 Phase-0 study."
    )
    parser_obj.add_argument("--spec", type=Path, default=DEFAULT_SPEC_PATH)
    parser_obj.add_argument("--freeze", type=Path, default=DEFAULT_FREEZE_PATH)
    parser_obj.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR_PATH)
    parser_obj.add_argument("--resume", action="store_true")
    return parser_obj.parse_args(arg_list)


def main(arg_list: Iterable[str] | None = None) -> int:
    args_obj = parse_args(arg_list)
    report_path = run_study(
        spec_path=args_obj.spec,
        freeze_path=args_obj.freeze,
        output_dir_path=args_obj.output_dir,
        resume_bool=bool(args_obj.resume),
    )
    print(report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
