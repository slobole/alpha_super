"""Frozen H0-H8 Ladder4 stabilizer value-add study.

The runner executes 19 unique strategy/capital paths exactly once, discards
heavy strategy objects after writing compact lineage, builds nine independent
no-rebalance books, and performs all comparisons on one exact global date
intersection. It has research authority only.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import gzip
import hashlib
import importlib
from importlib import metadata as importlib_metadata
import importlib.util
import inspect
import io
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, norm, skew
from statsmodels.stats.multitest import multipletests
import yaml

from alpha.engine.strategy import Strategy
from data.norgate_loader import TOTALRETURN_ADJUSTMENT_STR, load_price_timeseries


REPO_ROOT_PATH = Path(__file__).resolve().parents[2]
DEFAULT_SPEC_PATH = (
    REPO_ROOT_PATH
    / "scripts"
    / "research"
    / "specs"
    / "ladder4_candidate_value_add_h0_h8.yaml"
)
DEFAULT_OUTPUT_DIR_PATH = (
    REPO_ROOT_PATH
    / "results"
    / "research"
    / "portfolio"
    / "ladder4_candidate_value_add_study"
    / "2026-08-29_h0_h8"
)
SELECTABLE_HYPOTHESIS_ID_TUPLE = ("H1", "H2", "H3", "H4")
ALL_HYPOTHESIS_ID_TUPLE = (
    "H0",
    "H1",
    "H2",
    "H3",
    "H4",
    "H5",
    "H6",
    "H7",
    "H8",
)
SHARED_EXECUTION_DEPENDENCY_RELATIVE_PATH_TUPLE = (
    "alpha/engine/backtest.py",
    "alpha/engine/backtester.py",
    "alpha/engine/strategy.py",
    "alpha/engine/order.py",
    "alpha/engine/metrics.py",
    "alpha/engine/risk_analysis.py",
    "alpha/engine/portfolio.py",
    "data/norgate_loader.py",
    "scripts/research/run_ladder4_candidate_value_add_study.py",
)
EXECUTED_PYTHON_TREE_RELATIVE_PATH_TUPLE = (
    "alpha",
    "data",
    "strategies",
)
FROZEN_SPEC_SHA256_STR = (
    "9d3613a92535a9ed5510c46d71ae5316553dcb1d20d6a9ca4235fdba7211bf0c"
)


def utc_now_str() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file_str(file_path: Path) -> str:
    digest_obj = hashlib.sha256()
    with file_path.open("rb") as file_obj:
        for chunk_bytes in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest_obj.update(chunk_bytes)
    return digest_obj.hexdigest()


def shared_execution_dependency_hash_dict() -> dict[str, str]:
    dependency_hash_by_path_dict: dict[str, str] = {}
    for relative_path_str in SHARED_EXECUTION_DEPENDENCY_RELATIVE_PATH_TUPLE:
        dependency_path = REPO_ROOT_PATH / relative_path_str
        if not dependency_path.is_file():
            raise FileNotFoundError(
                f"Shared execution dependency is missing: {relative_path_str}."
            )
        dependency_hash_by_path_dict[relative_path_str] = sha256_file_str(
            dependency_path
        )
    for relative_tree_path_str in EXECUTED_PYTHON_TREE_RELATIVE_PATH_TUPLE:
        tree_root_path = REPO_ROOT_PATH / relative_tree_path_str
        tree_digest_obj = hashlib.sha256()
        for python_path in sorted(tree_root_path.rglob("*.py")):
            normalized_relative_path_str = python_path.relative_to(
                REPO_ROOT_PATH
            ).as_posix()
            tree_digest_obj.update(normalized_relative_path_str.encode("utf-8"))
            tree_digest_obj.update(b"\0")
            tree_digest_obj.update(sha256_file_str(python_path).encode("ascii"))
            tree_digest_obj.update(b"\n")
        dependency_hash_by_path_dict[
            f"python_tree::{relative_tree_path_str}"
        ] = tree_digest_obj.hexdigest()
    return dependency_hash_by_path_dict


def norgate_database_vintage_dict() -> dict[str, Any]:
    """Fingerprint the local Norgate database update state for one fresh run."""

    norgatedata_module_obj = importlib.import_module("norgatedata")
    database_name_list = sorted(
        str(database_name_obj)
        for database_name_obj in norgatedata_module_obj.databases()
    )
    return {
        "norgatedata_package_version_str": importlib_metadata.version(
            "norgatedata"
        ),
        "database_last_update_by_name_dict": {
            database_name_str: str(
                norgatedata_module_obj.last_database_update_time(
                    database_name_str
                )
            )
            for database_name_str in database_name_list
        },
    }


def json_safe_obj(value_obj: Any) -> Any:
    if isinstance(value_obj, dict):
        return {
            str(key_obj): json_safe_obj(item_obj)
            for key_obj, item_obj in value_obj.items()
        }
    if isinstance(value_obj, (list, tuple, set)):
        return [json_safe_obj(item_obj) for item_obj in value_obj]
    if isinstance(value_obj, Path):
        return str(value_obj)
    if isinstance(value_obj, (pd.Timestamp, datetime)):
        return value_obj.isoformat()
    if isinstance(value_obj, np.generic):
        return value_obj.item()
    if isinstance(value_obj, float) and not np.isfinite(value_obj):
        return None
    return value_obj


def write_json(file_path: Path, payload_dict: dict[str, Any]) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = file_path.with_name(f"{file_path.name}.tmp")
    temporary_path.write_text(
        json.dumps(
            json_safe_obj(payload_dict),
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(file_path)


def append_jsonl(file_path: Path, payload_dict: dict[str, Any]) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("a", encoding="utf-8", newline="\n") as file_obj:
        file_obj.write(
            json.dumps(
                json_safe_obj(payload_dict),
                sort_keys=True,
                ensure_ascii=False,
            )
            + "\n"
        )


def write_csv_gzip(
    data_df: pd.DataFrame,
    file_path: Path,
    *,
    index_bool: bool,
    index_label_str: str | None = None,
) -> None:
    """Write deterministic gzip bytes by fixing the gzip mtime to zero."""

    file_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = file_path.with_name(f"{file_path.name}.tmp")
    with temporary_path.open("wb") as raw_file_obj:
        with gzip.GzipFile(
            filename="",
            mode="wb",
            fileobj=raw_file_obj,
            mtime=0,
        ) as gzip_file_obj:
            with io.TextIOWrapper(
                gzip_file_obj,
                encoding="utf-8",
                newline="",
            ) as text_file_obj:
                data_df.to_csv(
                    text_file_obj,
                    index=index_bool,
                    index_label=index_label_str,
                    float_format="%.12g",
                    lineterminator="\n",
                )
    temporary_path.replace(file_path)


def read_source_path_df(file_path: Path) -> pd.DataFrame:
    source_path_df = pd.read_csv(file_path, index_col="date", parse_dates=["date"])
    source_path_df.index = pd.to_datetime(source_path_df.index).normalize()
    source_path_df.index.name = "date"
    return source_path_df.sort_index()


def load_spec_dict(
    spec_path: Path,
    *,
    enforce_frozen_contract_bool: bool = True,
) -> dict[str, Any]:
    if enforce_frozen_contract_bool:
        actual_spec_sha256_str = sha256_file_str(spec_path)
        if actual_spec_sha256_str != FROZEN_SPEC_SHA256_STR:
            raise ValueError(
                "Study spec differs from the approved frozen contract: "
                f"{actual_spec_sha256_str}."
            )
    with spec_path.open(encoding="utf-8") as file_obj:
        spec_dict = yaml.safe_load(file_obj)
    validate_spec_dict(
        spec_dict,
        enforce_frozen_contract_bool=enforce_frozen_contract_bool,
    )
    validate_baseline_reference(spec_dict)
    return spec_dict


def validate_spec_dict(
    spec_dict: dict[str, Any],
    *,
    enforce_frozen_contract_bool: bool = True,
) -> None:
    if str(spec_dict.get("authority_str")) != "research_bench_only":
        raise ValueError("Study authority must remain research_bench_only.")
    hypothesis_dict = spec_dict.get("hypotheses")
    if tuple(hypothesis_dict or {}) != ALL_HYPOTHESIS_ID_TUPLE:
        raise ValueError("Frozen hypotheses must be ordered exactly H0 through H8.")
    source_run_dict = spec_dict.get("source_runs")
    if not isinstance(source_run_dict, dict) or len(source_run_dict) != 19:
        raise ValueError("Frozen study must contain exactly 19 unique source runs.")

    portfolio_contract_dict = spec_dict["portfolio_contract"]
    exact_portfolio_contract_dict = {
        "capital_base_float": 1_000_000.0,
        "requested_start_date_str": "2004-01-01",
        "effective_execution_start_date_str": "2012-10-01",
        "capital_anchor_date_str": "2012-09-28",
        "end_date_str": "2026-08-14",
        "outer_rebalance": None,
        "allocation_semantics_str": "fixed_initial_capital_then_independent_drift",
        "decision_execution_timing_str": "Close_T_to_Open_T_plus_1",
        "warmup_policy_str": (
            "strategy_native_pre_start_history_signal_only_no_positions"
        ),
        "first_result_must_equal_effective_execution_start_bool": True,
        "first_fill_return_policy_str": (
            "include_cash_anchor_to_first_post_fill_close"
        ),
        "global_comparison_index_str": (
            "exact_intersection_of_all_19_sources_before_any_hypothesis_is_built"
        ),
        "missing_return_policy_str": "reject_no_zero_cash_or_forward_fill",
        "resume_policy_str": (
            "forbidden_fresh_output_only_to_prevent_mixed_data_vintages"
        ),
    }
    if enforce_frozen_contract_bool:
        for contract_key_str, expected_value_obj in exact_portfolio_contract_dict.items():
            if portfolio_contract_dict.get(contract_key_str) != expected_value_obj:
                raise ValueError(
                    f"Frozen portfolio contract changed at {contract_key_str}."
                )
    requested_start_ts = pd.Timestamp(
        portfolio_contract_dict["requested_start_date_str"]
    )
    capital_anchor_ts = pd.Timestamp(
        portfolio_contract_dict["capital_anchor_date_str"]
    )
    effective_execution_start_ts = pd.Timestamp(
        portfolio_contract_dict["effective_execution_start_date_str"]
    )
    frozen_end_ts = pd.Timestamp(portfolio_contract_dict["end_date_str"])
    if not (
        requested_start_ts <= capital_anchor_ts
        < effective_execution_start_ts
        <= frozen_end_ts
    ):
        raise ValueError(
            "Portfolio dates must satisfy requested <= cash anchor < effective "
            "execution start <= end."
        )

    lineage_contract_dict = spec_dict["lineage_contract"]
    native_history_start_by_import_dict = lineage_contract_dict.get(
        "native_history_request_start_by_strategy_import",
        {},
    )
    expected_strategy_import_set = {
        str(source_spec_dict["strategy_import_str"])
        for source_spec_dict in source_run_dict.values()
    }
    if set(native_history_start_by_import_dict) != expected_strategy_import_set:
        raise ValueError(
            "Native history request mapping must cover every unique strategy import."
        )
    if bool(lineage_contract_dict.get("requested_start_must_be_covered_bool")):
        late_native_history_import_list = sorted(
            strategy_import_str
            for strategy_import_str, native_start_date_obj in (
                native_history_start_by_import_dict.items()
            )
            if pd.Timestamp(native_start_date_obj) > requested_start_ts
        )
        if late_native_history_import_list:
            raise ValueError(
                "Native loader history requests begin after the requested study "
                f"start for {late_native_history_import_list}."
            )

    capital_base_float = float(
        spec_dict["portfolio_contract"]["capital_base_float"]
    )
    for hypothesis_id_str, hypothesis_dict_obj in hypothesis_dict.items():
        source_weight_list = hypothesis_dict_obj["source_weight_list"]
        weight_sum_float = float(
            math.fsum(float(weight_obj) for _, weight_obj in source_weight_list)
        )
        if not math.isclose(weight_sum_float, 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(
                f"{hypothesis_id_str} weights must sum exactly to one; "
                f"found {weight_sum_float:.17g}."
            )
        source_id_list = [str(source_id_obj) for source_id_obj, _ in source_weight_list]
        if len(source_id_list) != len(set(source_id_list)):
            raise ValueError(f"{hypothesis_id_str} repeats a source run.")
        for source_id_str, weight_obj in source_weight_list:
            if source_id_str not in source_run_dict:
                raise ValueError(
                    f"{hypothesis_id_str} references unknown source {source_id_str}."
                )
            expected_capital_float = capital_base_float * float(weight_obj)
            actual_capital_float = float(
                source_run_dict[source_id_str]["allocated_capital_float"]
            )
            if not math.isclose(
                actual_capital_float,
                expected_capital_float,
                rel_tol=0.0,
                abs_tol=1e-6,
            ):
                raise ValueError(
                    f"{source_id_str} capital {actual_capital_float:.12f} does not "
                    f"match {hypothesis_id_str} weight capital "
                    f"{expected_capital_float:.12f}."
                )

    selectable_id_tuple = tuple(
        spec_dict["statistical_contract"]["selectable_hypothesis_id_list"]
    )
    if selectable_id_tuple != SELECTABLE_HYPOTHESIS_ID_TUPLE:
        raise ValueError("Selectable family must remain H1-H4.")
    expected_control_by_candidate_dict = {
        "H1": "H5",
        "H2": "H6",
        "H3": "H7",
        "H4": "H8",
    }
    for candidate_id_str, expected_control_id_str in (
        expected_control_by_candidate_dict.items()
    ):
        if (
            hypothesis_dict[candidate_id_str]["matched_control_id_str"]
            != expected_control_id_str
        ):
            raise ValueError(
                f"{candidate_id_str} must be matched to {expected_control_id_str}."
            )

    capacity_contract_dict = spec_dict["capacity_contract"]
    if set(
        capacity_contract_dict.get("exact_artifact_path_by_candidate", {})
    ) != set(SELECTABLE_HYPOTHESIS_ID_TUPLE):
        raise ValueError("Exact capacity paths must be frozen separately for H1-H4.")
    if not bool(
        capacity_contract_dict.get(
            "current_study_capacity_gate_must_remain_false_bool"
        )
    ):
        raise ValueError("Phase 1 Capacity gate must remain fail-closed.")
    if bool(
        capacity_contract_dict.get("external_artifact_can_clear_current_study_bool")
    ):
        raise ValueError("External Capacity artifacts cannot authorize Phase 1.")
    if not bool(
        capacity_contract_dict.get(
            "phase_2_contract_must_be_frozen_before_capacity_data_review_bool"
        )
    ):
        raise ValueError("A separate pre-registered Capacity phase is required.")

    stats_contract_dict = spec_dict["statistical_contract"]
    exact_statistical_contract_dict = {
        "bootstrap_type_str": "synchronized_paired_stationary",
        "simulation_count_int": 10_000,
        "random_seed_int": 20_260_829,
        "primary_mean_block_length_int": 63,
        "sensitivity_mean_block_length_int_list": [21, 126],
        "annualization_day_int": 252,
        "risk_free_rate_float": 0.0,
        "es_quantile_float": 0.05,
        "family_correction_str": "Holm",
        "family_alpha_float": 0.05,
        "cumulative_related_path_lower_bound_int": 35,
    }
    if enforce_frozen_contract_bool:
        for contract_key_str, expected_value_obj in exact_statistical_contract_dict.items():
            if stats_contract_dict.get(contract_key_str) != expected_value_obj:
                raise ValueError(
                    f"Frozen statistical contract changed at {contract_key_str}."
                )
        if spec_dict["stop_rule"].get("if_no_candidate_passes_str") != (
            "retain_H0_and_stop_without_tuning"
        ):
            raise ValueError("Frozen no-pass stop rule changed.")


def validate_baseline_reference(spec_dict: dict[str, Any]) -> None:
    portfolio_contract_dict = spec_dict["portfolio_contract"]
    baseline_config_path = (
        REPO_ROOT_PATH
        / str(portfolio_contract_dict["baseline_reference_config_path_str"])
    )
    with baseline_config_path.open(encoding="utf-8") as file_obj:
        baseline_config_dict = yaml.safe_load(file_obj)
    if baseline_config_dict.get("rebalance") is not None:
        raise ValueError("Frozen Ladder4 baseline must have rebalance: null.")
    if float(baseline_config_dict["capital_base_float"]) != float(
        portfolio_contract_dict["capital_base_float"]
    ):
        raise ValueError("Baseline reference capital differs from the frozen study.")
    if str(baseline_config_dict["backtest_start_date_str"]) != str(
        portfolio_contract_dict["requested_start_date_str"]
    ):
        raise ValueError("Baseline reference requested start differs from the study.")

    source_run_dict = spec_dict["source_runs"]
    expected_pair_list = [
        (
            str(source_run_dict[source_id_str]["strategy_import_str"]),
            float(weight_obj),
        )
        for source_id_str, weight_obj in spec_dict["hypotheses"]["H0"][
            "source_weight_list"
        ]
    ]
    actual_pair_list = [
        (str(pod_dict["strategy_import_str"]), float(pod_dict["weight_float"]))
        for pod_dict in baseline_config_dict["pods"]
    ]
    if actual_pair_list != expected_pair_list:
        raise ValueError("H0 does not reproduce the Ladder4 reference config exactly.")


def validate_run_variant_signature(
    run_variant_fn: Any,
    strategy_import_str: str,
    run_variant_kwargs_dict: dict[str, Any] | None = None,
) -> None:
    required_parameter_set = {
        "show_display_bool",
        "save_results_bool",
        "output_dir_str",
        "backtest_start_date_str",
        "capital_base_float",
        "end_date_str",
    }
    missing_parameter_set = required_parameter_set - set(
        inspect.signature(run_variant_fn).parameters
    )
    if missing_parameter_set:
        raise TypeError(
            f"{strategy_import_str} run_variant is missing "
            f"{sorted(missing_parameter_set)}."
        )
    parameter_name_set = set(inspect.signature(run_variant_fn).parameters)
    unsupported_parameter_set = set(run_variant_kwargs_dict or {}) - parameter_name_set
    if unsupported_parameter_set:
        raise TypeError(
            f"{strategy_import_str} run_variant does not support frozen kwargs "
            f"{sorted(unsupported_parameter_set)}."
        )


def extract_source_result_df(strategy_obj: Strategy) -> pd.DataFrame:
    required_column_list = ["total_value", "portfolio_value", "cash"]
    missing_column_list = [
        column_str
        for column_str in required_column_list
        if column_str not in strategy_obj.results.columns
    ]
    if missing_column_list:
        raise RuntimeError(
            f"Strategy {strategy_obj.name} is missing result columns {missing_column_list}."
        )
    source_result_df = strategy_obj.results.loc[:, required_column_list].copy()
    source_result_df.columns = [
        "total_value_float",
        "portfolio_value_float",
        "cash_float",
    ]
    source_result_df.index = pd.to_datetime(source_result_df.index).normalize()
    source_result_df.index.name = "date"
    if source_result_df.index.has_duplicates:
        raise RuntimeError(f"Strategy {strategy_obj.name} produced duplicate dates.")
    if not source_result_df.index.is_monotonic_increasing:
        raise RuntimeError(f"Strategy {strategy_obj.name} dates are not increasing.")
    if not np.isfinite(source_result_df.to_numpy(dtype=float)).all():
        raise RuntimeError(f"Strategy {strategy_obj.name} produced non-finite results.")
    if source_result_df["total_value_float"].le(0.0).any():
        raise RuntimeError(f"Strategy {strategy_obj.name} produced non-positive NAV.")
    return source_result_df


def build_anchored_source_result_df(
    strategy_source_result_df: pd.DataFrame,
    *,
    strategy_name_str: str,
    allocated_capital_float: float,
    capital_anchor_date_str: str,
    effective_execution_start_date_str: str,
) -> pd.DataFrame:
    """Prepend the common cash state that exists before the first liveable fill.

    Every strategy still receives its full native pre-start price history for
    signal warmup, but its engine calendar starts at the frozen effective date.
    The added row is therefore an actual initial cash state, not a rescaled NAV
    from an earlier independently drifted strategy path.
    """

    capital_anchor_ts = pd.Timestamp(capital_anchor_date_str).normalize()
    effective_execution_start_ts = pd.Timestamp(
        effective_execution_start_date_str
    ).normalize()
    strategy_result_start_ts = pd.Timestamp(
        strategy_source_result_df.index[0]
    ).normalize()
    if strategy_result_start_ts != effective_execution_start_ts:
        raise RuntimeError(
            f"{strategy_name_str} first engine result is "
            f"{strategy_result_start_ts.date()}, not frozen effective execution "
            f"start {effective_execution_start_ts.date()}."
        )
    if capital_anchor_ts >= effective_execution_start_ts:
        raise ValueError("The cash anchor must precede the effective execution start.")
    if capital_anchor_ts in strategy_source_result_df.index:
        raise RuntimeError(
            f"{strategy_name_str} unexpectedly contains the dedicated cash anchor."
        )
    if not np.isfinite(allocated_capital_float) or allocated_capital_float <= 0.0:
        raise ValueError("allocated_capital_float must be finite and positive.")

    cash_anchor_df = pd.DataFrame(
        {
            "total_value_float": [float(allocated_capital_float)],
            "portfolio_value_float": [0.0],
            "cash_float": [float(allocated_capital_float)],
        },
        index=pd.DatetimeIndex([capital_anchor_ts], name="date"),
    )
    # *** CRITICAL*** The first realized return is measured from known cash at
    # Close_T on the anchor date to the first post-fill Close_(T+1). This keeps
    # first-fill slippage, commissions, and open-to-close P&L in the study.
    anchored_source_result_df = pd.concat(
        [cash_anchor_df, strategy_source_result_df],
        axis=0,
    )
    anchored_source_result_df.index.name = "date"
    return anchored_source_result_df


def extract_source_transaction_df(
    strategy_obj: Strategy,
    source_id_str: str,
) -> pd.DataFrame:
    transaction_df = strategy_obj.get_transactions().copy()
    required_column_list = [
        "bar",
        "asset",
        "amount",
        "price",
        "total_value",
        "commission",
    ]
    if len(transaction_df) == 0:
        return pd.DataFrame(
            columns=[
                "source_id_str",
                "date",
                "asset_str",
                "amount_float",
                "fill_price_float",
                "signed_notional_float",
                "commission_float",
            ]
        )
    missing_column_list = [
        column_str
        for column_str in required_column_list
        if column_str not in transaction_df.columns
    ]
    if missing_column_list:
        raise RuntimeError(
            f"Strategy {strategy_obj.name} transaction ledger is missing "
            f"{missing_column_list}."
        )
    compact_transaction_df = pd.DataFrame(
        {
            "source_id_str": source_id_str,
            "date": pd.to_datetime(transaction_df["bar"]).dt.normalize(),
            "asset_str": transaction_df["asset"].astype(str),
            "amount_float": pd.to_numeric(transaction_df["amount"]),
            "fill_price_float": pd.to_numeric(transaction_df["price"]),
            "signed_notional_float": pd.to_numeric(transaction_df["total_value"]),
            "commission_float": pd.to_numeric(transaction_df["commission"]),
        }
    )
    return compact_transaction_df


def source_metadata_dict(
    strategy_obj: Strategy,
    *,
    source_result_df: pd.DataFrame,
    strategy_result_start_ts: pd.Timestamp,
    source_id_str: str,
    strategy_import_str: str,
    allocated_capital_float: float,
    requested_start_date_str: str,
    native_history_request_start_date_str: str,
    capital_anchor_date_str: str,
    effective_execution_start_date_str: str,
    engine_request_start_date_str: str,
    module_path: Path,
    run_variant_kwargs_dict: dict[str, Any],
) -> dict[str, Any]:
    transaction_df = strategy_obj.get_transactions()
    accounting_policy_dict = dict(
        getattr(strategy_obj, "_accounting_policy_dict", {})
    )
    data_adjustment_policy_dict = dict(
        getattr(strategy_obj, "_data_adjustment_policy_dict", {})
    )
    return {
        "source_id_str": source_id_str,
        "strategy_name_str": str(strategy_obj.name),
        "strategy_import_str": strategy_import_str,
        "allocated_capital_float": allocated_capital_float,
        "requested_history_start_date_str": requested_start_date_str,
        "native_history_request_start_date_str": (
            native_history_request_start_date_str
        ),
        "capital_anchor_date_str": capital_anchor_date_str,
        "effective_execution_start_date_str": (
            effective_execution_start_date_str
        ),
        "engine_request_start_date_str": engine_request_start_date_str,
        "actual_start_date_str": source_result_df.index[0].date().isoformat(),
        "strategy_result_start_date_str": (
            pd.Timestamp(strategy_result_start_ts).date().isoformat()
        ),
        "actual_end_date_str": source_result_df.index[-1].date().isoformat(),
        "observation_count_int": int(len(source_result_df)),
        "transaction_count_int": int(len(transaction_df)),
        "gross_transaction_notional_float": float(
            pd.to_numeric(transaction_df.get("total_value", pd.Series(dtype=float)))
            .abs()
            .sum()
        ),
        "total_commission_float": float(
            pd.to_numeric(transaction_df.get("commission", pd.Series(dtype=float))).sum()
        ),
        "negative_cash_day_count_int": int(
            (source_result_df["cash_float"] < 0.0).sum()
        ),
        "minimum_cash_float": float(source_result_df["cash_float"].min()),
        "slippage_per_side_float": float(strategy_obj._slippage),
        "commission_per_share_float": float(strategy_obj._commission_per_share),
        "commission_minimum_float": float(strategy_obj._commission_minimum),
        "dividend_withholding_rate_float": float(
            np.nan
            if accounting_policy_dict.get("dividend_withholding_rate_float") is None
            else accounting_policy_dict.get("dividend_withholding_rate_float", np.nan)
        ),
        "positive_cash_rate_policy_str": str(
            accounting_policy_dict.get("positive_cash_rate_policy_str", "missing")
        ),
        "negative_cash_financing_policy_str": str(
            accounting_policy_dict.get(
                "negative_cash_financing_policy_str",
                "missing",
            )
        ),
        "execution_adjustment_str": str(
            data_adjustment_policy_dict.get(
                "execution_and_marks_adjustment_str",
                "missing",
            )
        ),
        "run_variant_kwargs_dict": dict(run_variant_kwargs_dict),
        "accounting_policy_dict": accounting_policy_dict,
        "data_adjustment_policy_dict": data_adjustment_policy_dict,
        "dividend_cash_gross_total_float": float(
            getattr(strategy_obj, "dividend_cash_gross_total_float", 0.0)
        ),
        "dividend_withholding_total_float": float(
            getattr(strategy_obj, "dividend_withholding_total_float", 0.0)
        ),
        "dividend_cash_net_total_float": float(
            getattr(strategy_obj, "dividend_cash_net_total_float", 0.0)
        ),
        "module_path_str": str(module_path),
        "module_sha256_str": sha256_file_str(module_path),
        "shared_execution_dependency_hash_dict": (
            shared_execution_dependency_hash_dict()
        ),
    }


def execute_source_runs(
    spec_dict: dict[str, Any],
    output_dir_path: Path,
    *,
    resume_bool: bool = False,
) -> pd.DataFrame:
    portfolio_contract_dict = spec_dict["portfolio_contract"]
    requested_start_date_str = str(
        portfolio_contract_dict["requested_start_date_str"]
    )
    native_history_start_by_import_dict = spec_dict["lineage_contract"][
        "native_history_request_start_by_strategy_import"
    ]
    effective_execution_start_date_str = str(
        portfolio_contract_dict["effective_execution_start_date_str"]
    )
    capital_anchor_date_str = str(
        portfolio_contract_dict["capital_anchor_date_str"]
    )
    end_date_str = str(portfolio_contract_dict["end_date_str"])
    source_path_dir_path = output_dir_path / "source_paths"
    source_transaction_dir_path = output_dir_path / "source_transactions"
    source_metadata_dir_path = output_dir_path / "source_metadata"
    execution_ledger_path = output_dir_path / "experiment_ledger.jsonl"
    source_summary_row_list: list[dict[str, Any]] = []

    for source_position_int, (source_id_str, source_spec_dict) in enumerate(
        spec_dict["source_runs"].items(),
        start=1,
    ):
        strategy_import_str = str(source_spec_dict["strategy_import_str"])
        native_history_request_start_date_str = str(
            native_history_start_by_import_dict[strategy_import_str]
        )
        module_import_str = strategy_import_str.split(":", maxsplit=1)[0]
        allocated_capital_float = float(
            source_spec_dict["allocated_capital_float"]
        )
        run_variant_kwargs_dict = dict(
            source_spec_dict.get("run_variant_kwargs_dict", {})
        )
        engine_request_start_date_str = str(
            source_spec_dict.get(
                "engine_request_start_date_str",
                effective_execution_start_date_str,
            )
        )
        reserved_parameter_set = {
            "show_display_bool",
            "save_results_bool",
            "output_dir_str",
            "backtest_start_date_str",
            "capital_base_float",
            "end_date_str",
        }
        if reserved_parameter_set.intersection(run_variant_kwargs_dict):
            raise ValueError(
                f"{source_id_str} run_variant_kwargs_dict overrides a runner contract."
            )
        source_path_file_path = source_path_dir_path / f"{source_id_str}.csv.gz"
        transaction_file_path = (
            source_transaction_dir_path / f"{source_id_str}.csv.gz"
        )
        metadata_file_path = source_metadata_dir_path / f"{source_id_str}.json"
        checkpoint_exists_bool_list = [
            source_path_file_path.is_file(),
            transaction_file_path.is_file(),
            metadata_file_path.is_file(),
        ]
        if resume_bool and all(checkpoint_exists_bool_list):
            metadata_dict = json.loads(metadata_file_path.read_text(encoding="utf-8"))
            checkpoint_source_df = read_source_path_df(source_path_file_path)
            if str(metadata_dict["strategy_import_str"]) != strategy_import_str:
                raise RuntimeError(f"{source_id_str} checkpoint strategy import changed.")
            if metadata_dict.get("run_variant_kwargs_dict", {}) != run_variant_kwargs_dict:
                raise RuntimeError(f"{source_id_str} checkpoint kwargs changed.")
            if metadata_dict.get(
                "engine_request_start_date_str"
            ) != engine_request_start_date_str:
                raise RuntimeError(
                    f"{source_id_str} checkpoint engine request start changed."
                )
            if not math.isclose(
                float(metadata_dict["allocated_capital_float"]),
                allocated_capital_float,
                rel_tol=0.0,
                abs_tol=1e-9,
            ):
                raise RuntimeError(f"{source_id_str} checkpoint capital changed.")
            if checkpoint_source_df.index[-1] != pd.Timestamp(end_date_str):
                raise RuntimeError(f"{source_id_str} checkpoint endpoint changed.")
            if (
                checkpoint_source_df.index.has_duplicates
                or not checkpoint_source_df.index.is_monotonic_increasing
                or not np.isfinite(checkpoint_source_df.to_numpy(dtype=float)).all()
                or checkpoint_source_df["total_value_float"].le(0.0).any()
            ):
                raise RuntimeError(f"{source_id_str} checkpoint path is invalid.")
            if metadata_dict.get("source_path_sha256_str") != sha256_file_str(
                source_path_file_path
            ):
                raise RuntimeError(f"{source_id_str} checkpoint path hash changed.")
            if metadata_dict.get(
                "transaction_path_sha256_str"
            ) != sha256_file_str(transaction_file_path):
                raise RuntimeError(
                    f"{source_id_str} checkpoint transaction hash changed."
                )
            module_spec_obj = importlib.util.find_spec(module_import_str)
            if module_spec_obj is None or module_spec_obj.origin is None:
                raise RuntimeError(f"Cannot resolve module for {source_id_str} checkpoint.")
            current_module_path = Path(module_spec_obj.origin).resolve()
            if metadata_dict.get("module_sha256_str") != sha256_file_str(
                current_module_path
            ):
                raise RuntimeError(f"{source_id_str} strategy code changed since checkpoint.")
            if metadata_dict.get(
                "shared_execution_dependency_hash_dict"
            ) != shared_execution_dependency_hash_dict():
                raise RuntimeError(
                    f"{source_id_str} shared execution code changed since checkpoint."
                )
            source_summary_row_list.append(
                {
                    **{
                        key_str: value_obj
                        for key_str, value_obj in metadata_dict.items()
                        if not key_str.endswith("_dict")
                    },
                    "source_path_sha256_str": sha256_file_str(source_path_file_path),
                    "transaction_path_sha256_str": sha256_file_str(
                        transaction_file_path
                    ),
                    "metadata_sha256_str": sha256_file_str(metadata_file_path),
                }
            )
            append_jsonl(
                execution_ledger_path,
                {
                    "event_str": "source_run_reused_from_valid_checkpoint",
                    "recorded_at_utc_str": utc_now_str(),
                    "source_id_str": source_id_str,
                    "source_position_int": source_position_int,
                },
            )
            print(
                f"[{source_position_int:02d}/{len(spec_dict['source_runs']):02d}] "
                f"{source_id_str} | valid checkpoint",
                flush=True,
            )
            continue
        if resume_bool and any(checkpoint_exists_bool_list):
            append_jsonl(
                execution_ledger_path,
                {
                    "event_str": "partial_source_checkpoint_restarted",
                    "recorded_at_utc_str": utc_now_str(),
                    "source_id_str": source_id_str,
                    "existing_component_bool_list": checkpoint_exists_bool_list,
                },
            )
            print(
                f"[{source_position_int:02d}/{len(spec_dict['source_runs']):02d}] "
                f"{source_id_str} | incomplete checkpoint rerun",
                flush=True,
            )
        append_jsonl(
            execution_ledger_path,
            {
                "event_str": "source_run_started",
                "recorded_at_utc_str": utc_now_str(),
                "source_id_str": source_id_str,
                "source_position_int": source_position_int,
                "source_count_int": len(spec_dict["source_runs"]),
                "strategy_import_str": strategy_import_str,
                "allocated_capital_float": allocated_capital_float,
                "run_variant_kwargs_dict": run_variant_kwargs_dict,
            },
        )
        print(
            f"[{source_position_int:02d}/{len(spec_dict['source_runs']):02d}] "
            f"{source_id_str} | {allocated_capital_float:,.2f}",
            flush=True,
        )
        strategy_module_obj = importlib.import_module(module_import_str)
        run_variant_fn = getattr(strategy_module_obj, "run_variant", None)
        if not callable(run_variant_fn):
            raise AttributeError(f"{module_import_str} has no run_variant function.")
        validate_run_variant_signature(
            run_variant_fn,
            strategy_import_str,
            run_variant_kwargs_dict,
        )
        strategy_obj = run_variant_fn(
            show_display_bool=False,
            save_results_bool=False,
            output_dir_str=str(output_dir_path),
            backtest_start_date_str=engine_request_start_date_str,
            capital_base_float=allocated_capital_float,
            end_date_str=end_date_str,
            **run_variant_kwargs_dict,
        )
        if not isinstance(strategy_obj, Strategy):
            raise TypeError(f"{strategy_import_str} did not return Strategy.")
        if ":" in strategy_import_str:
            expected_class_name_str = strategy_import_str.split(":", maxsplit=1)[1]
            expected_strategy_class_obj = getattr(
                strategy_module_obj,
                expected_class_name_str,
                None,
            )
            if not isinstance(expected_strategy_class_obj, type) or not isinstance(
                strategy_obj,
                expected_strategy_class_obj,
            ):
                raise TypeError(
                    f"{strategy_import_str} run_variant returned the wrong class."
                )
        strategy_source_result_df = extract_source_result_df(strategy_obj)
        if strategy_source_result_df.index[-1] != pd.Timestamp(end_date_str):
            raise RuntimeError(
                f"{source_id_str} ended {strategy_source_result_df.index[-1].date()}, "
                f"not frozen endpoint {end_date_str}."
            )
        strategy_result_start_ts = pd.Timestamp(strategy_source_result_df.index[0])
        source_result_df = build_anchored_source_result_df(
            strategy_source_result_df,
            strategy_name_str=str(strategy_obj.name),
            allocated_capital_float=allocated_capital_float,
            capital_anchor_date_str=capital_anchor_date_str,
            effective_execution_start_date_str=effective_execution_start_date_str,
        )
        compact_transaction_df = extract_source_transaction_df(
            strategy_obj,
            source_id_str,
        )
        module_path = Path(strategy_module_obj.__file__).resolve()
        metadata_dict = source_metadata_dict(
            strategy_obj,
            source_result_df=source_result_df,
            strategy_result_start_ts=strategy_result_start_ts,
            source_id_str=source_id_str,
            strategy_import_str=strategy_import_str,
            allocated_capital_float=allocated_capital_float,
            requested_start_date_str=requested_start_date_str,
            native_history_request_start_date_str=(
                native_history_request_start_date_str
            ),
            capital_anchor_date_str=capital_anchor_date_str,
            effective_execution_start_date_str=effective_execution_start_date_str,
            engine_request_start_date_str=engine_request_start_date_str,
            module_path=module_path,
            run_variant_kwargs_dict=run_variant_kwargs_dict,
        )

        write_csv_gzip(
            source_result_df,
            source_path_file_path,
            index_bool=True,
            index_label_str="date",
        )
        write_csv_gzip(
            compact_transaction_df,
            transaction_file_path,
            index_bool=False,
        )
        metadata_dict["source_path_sha256_str"] = sha256_file_str(
            source_path_file_path
        )
        metadata_dict["transaction_path_sha256_str"] = sha256_file_str(
            transaction_file_path
        )
        write_json(metadata_file_path, metadata_dict)
        source_summary_row_list.append(
            {
                **{
                    key_str: value_obj
                    for key_str, value_obj in metadata_dict.items()
                    if not key_str.endswith("_dict")
                },
                "source_path_sha256_str": sha256_file_str(source_path_file_path),
                "transaction_path_sha256_str": sha256_file_str(
                    transaction_file_path
                ),
                "metadata_sha256_str": sha256_file_str(metadata_file_path),
            }
        )
        append_jsonl(
            execution_ledger_path,
            {
                "event_str": "source_run_completed",
                "recorded_at_utc_str": utc_now_str(),
                "source_id_str": source_id_str,
                "actual_start_date_str": metadata_dict["actual_start_date_str"],
                "actual_end_date_str": metadata_dict["actual_end_date_str"],
                "observation_count_int": metadata_dict["observation_count_int"],
                "transaction_count_int": metadata_dict["transaction_count_int"],
                "source_path_sha256_str": sha256_file_str(source_path_file_path),
            },
        )
        del strategy_obj
        del strategy_source_result_df
        del source_result_df
        del compact_transaction_df
        gc.collect()

    source_summary_df = pd.DataFrame(source_summary_row_list)
    source_summary_df.to_csv(
        output_dir_path / "source_run_summary.csv",
        index=False,
        float_format="%.12g",
        lineterminator="\n",
    )
    return source_summary_df


def load_all_source_path_dict(
    spec_dict: dict[str, Any],
    output_dir_path: Path,
) -> dict[str, pd.DataFrame]:
    return {
        source_id_str: read_source_path_df(
            output_dir_path / "source_paths" / f"{source_id_str}.csv.gz"
        )
        for source_id_str in spec_dict["source_runs"]
    }


def build_global_source_index(
    spec_dict: dict[str, Any],
    source_path_by_id_dict: dict[str, pd.DataFrame],
) -> pd.DatetimeIndex:
    expected_source_id_set = set(spec_dict["source_runs"])
    if set(source_path_by_id_dict) != expected_source_id_set:
        raise ValueError("Loaded source IDs differ from the frozen 19-source contract.")
    global_idx: pd.DatetimeIndex | None = None
    for source_id_str in spec_dict["source_runs"]:
        source_path_df = source_path_by_id_dict[source_id_str]
        if (
            source_path_df.index.has_duplicates
            or not source_path_df.index.is_monotonic_increasing
        ):
            raise RuntimeError(f"{source_id_str} has an invalid date index.")
        source_idx = pd.DatetimeIndex(source_path_df.index)
        global_idx = (
            source_idx
            if global_idx is None
            else global_idx.intersection(source_idx)
        )
    if global_idx is None:
        raise RuntimeError("No source paths were loaded.")
    portfolio_contract_dict = spec_dict["portfolio_contract"]
    capital_anchor_ts = pd.Timestamp(
        portfolio_contract_dict["capital_anchor_date_str"]
    )
    effective_execution_start_ts = pd.Timestamp(
        portfolio_contract_dict["effective_execution_start_date_str"]
    )
    frozen_end_ts = pd.Timestamp(portfolio_contract_dict["end_date_str"])
    global_idx = global_idx.sort_values()
    global_idx = global_idx[
        (global_idx >= capital_anchor_ts) & (global_idx <= frozen_end_ts)
    ]
    if len(global_idx) < 252:
        raise RuntimeError("Global 19-source comparison index is too short.")
    if global_idx[-1] != frozen_end_ts:
        raise RuntimeError(
            f"Global index ends {global_idx[-1].date()}, not {frozen_end_ts.date()}."
        )
    if global_idx[0] != capital_anchor_ts:
        raise RuntimeError(
            f"Global index starts {global_idx[0].date()}, not the frozen cash "
            f"anchor {capital_anchor_ts.date()}."
        )
    if len(global_idx) < 2 or global_idx[1] != effective_execution_start_ts:
        observed_second_date_str = (
            global_idx[1].date().isoformat() if len(global_idx) >= 2 else "missing"
        )
        raise RuntimeError(
            "The first common post-anchor result must be the frozen effective "
            f"execution start {effective_execution_start_ts.date()}; found "
            f"{observed_second_date_str}."
        )
    return pd.DatetimeIndex(global_idx)


def build_no_rebalance_hypothesis(
    source_path_by_id_dict: dict[str, pd.DataFrame],
    source_weight_list: list[list[Any]],
    capital_base_float: float,
    common_idx: pd.DatetimeIndex | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Mirror ``Portfolio._build`` for a compact, no-rebalance source set."""

    resolved_common_idx = None if common_idx is None else pd.DatetimeIndex(common_idx)
    if resolved_common_idx is None:
        for source_id_obj, _weight_obj in source_weight_list:
            source_idx = source_path_by_id_dict[str(source_id_obj)].index
            resolved_common_idx = (
                pd.DatetimeIndex(source_idx)
                if resolved_common_idx is None
                else resolved_common_idx.intersection(source_idx)
            )
    if resolved_common_idx is None or len(resolved_common_idx) < 2:
        raise ValueError("Hypothesis has fewer than two exact common observations.")
    resolved_common_idx = resolved_common_idx.sort_values()

    sleeve_equity_df = pd.DataFrame(index=resolved_common_idx)
    for source_id_obj, weight_obj in source_weight_list:
        source_id_str = str(source_id_obj)
        missing_date_idx = resolved_common_idx.difference(
            source_path_by_id_dict[source_id_str].index
        )
        if len(missing_date_idx) > 0:
            raise RuntimeError(
                f"{source_id_str} is missing frozen global dates: "
                f"{missing_date_idx[:5].tolist()}."
            )
        source_total_value_ser = source_path_by_id_dict[source_id_str].loc[
            resolved_common_idx,
            "total_value_float",
        ].astype(float)
        expected_anchor_capital_float = capital_base_float * float(weight_obj)
        if not math.isclose(
            float(source_total_value_ser.iloc[0]),
            expected_anchor_capital_float,
            rel_tol=0.0,
            abs_tol=1e-6,
        ):
            raise RuntimeError(
                f"{source_id_str} starts at {source_total_value_ser.iloc[0]:.12f}, "
                f"not its frozen cash allocation {expected_anchor_capital_float:.12f}."
            )
        # *** CRITICAL*** The first row is the frozen pre-fill cash anchor shared
        # by every source. The next row therefore retains first-fill costs and
        # Open_(T+1)-to-Close_(T+1) P&L. Any later missing return is rejected.
        source_return_ser = source_total_value_ser.pct_change(fill_method=None)
        if source_return_ser.iloc[1:].isna().any():
            raise RuntimeError(f"{source_id_str} has a missing common-index return.")
        source_return_ser.iloc[0] = 0.0
        sleeve_equity_df[source_id_str] = (
            capital_base_float
            * float(weight_obj)
            * (1.0 + source_return_ser).cumprod()
        )

    total_value_ser = sleeve_equity_df.sum(axis=1).rename("total_value_float")
    return_ser = total_value_ser.pct_change(fill_method=None)
    return_ser.iloc[0] = 0.0
    hypothesis_path_df = pd.DataFrame(
        {
            "total_value_float": total_value_ser,
            "return_float": return_ser,
        },
        index=resolved_common_idx,
    )
    hypothesis_path_df.index.name = "date"
    return hypothesis_path_df, sleeve_equity_df


def build_all_hypothesis_path_dict(
    spec_dict: dict[str, Any],
    source_path_by_id_dict: dict[str, pd.DataFrame],
    global_idx: pd.DatetimeIndex,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    capital_base_float = float(
        spec_dict["portfolio_contract"]["capital_base_float"]
    )
    hypothesis_path_by_id_dict: dict[str, pd.DataFrame] = {}
    sleeve_equity_by_hypothesis_dict: dict[str, pd.DataFrame] = {}
    for hypothesis_id_str, hypothesis_dict in spec_dict["hypotheses"].items():
        hypothesis_path_df, sleeve_equity_df = build_no_rebalance_hypothesis(
            source_path_by_id_dict=source_path_by_id_dict,
            source_weight_list=hypothesis_dict["source_weight_list"],
            capital_base_float=capital_base_float,
            common_idx=global_idx,
        )
        hypothesis_path_by_id_dict[hypothesis_id_str] = hypothesis_path_df
        sleeve_equity_by_hypothesis_dict[hypothesis_id_str] = sleeve_equity_df
    return hypothesis_path_by_id_dict, sleeve_equity_by_hypothesis_dict


def build_global_path_frames(
    spec_dict: dict[str, Any],
    hypothesis_path_by_id_dict: dict[str, pd.DataFrame],
    global_idx: pd.DatetimeIndex,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, str]:
    global_idx = pd.DatetimeIndex(global_idx).sort_values()
    if len(global_idx) < 252:
        raise RuntimeError("Global H0-H8 index is too short.")
    frozen_end_ts = pd.Timestamp(spec_dict["portfolio_contract"]["end_date_str"])
    if global_idx[-1] != frozen_end_ts:
        raise RuntimeError(
            f"Global index ends {global_idx[-1].date()}, not {frozen_end_ts.date()}."
        )

    capital_base_float = float(
        spec_dict["portfolio_contract"]["capital_base_float"]
    )
    total_value_df = pd.DataFrame(index=global_idx)
    return_df = pd.DataFrame(index=global_idx)
    for hypothesis_id_str in ALL_HYPOTHESIS_ID_TUPLE:
        hypothesis_path_df = hypothesis_path_by_id_dict[hypothesis_id_str]
        if not hypothesis_path_df.index.equals(global_idx):
            raise RuntimeError(
                f"{hypothesis_id_str} was not built on the frozen global index."
            )
        anchored_total_value_ser = hypothesis_path_df[
            "total_value_float"
        ].astype(float)
        if not math.isclose(
            float(anchored_total_value_ser.iloc[0]),
            capital_base_float,
            rel_tol=0.0,
            abs_tol=1e-6,
        ):
            raise RuntimeError(
                f"{hypothesis_id_str} does not start at frozen capital weights."
            )
        hypothesis_return_ser = hypothesis_path_df["return_float"].astype(float)
        if hypothesis_return_ser.iloc[1:].isna().any():
            raise RuntimeError(
                f"{hypothesis_id_str} has a missing return on the global index."
            )
        hypothesis_return_ser.iloc[0] = 0.0
        total_value_df[hypothesis_id_str] = anchored_total_value_ser
        return_df[hypothesis_id_str] = hypothesis_return_ser

    benchmark_price_df = load_price_timeseries(
        "$SPXTR",
        adjustment_str=TOTALRETURN_ADJUSTMENT_STR,
        start_date_str=global_idx[0].date().isoformat(),
        end_date_str=frozen_end_ts.date().isoformat(),
    )
    benchmark_close_ser = benchmark_price_df["Close"].astype(float).reindex(global_idx)
    if benchmark_close_ser.isna().any():
        missing_date_list = benchmark_close_ser[benchmark_close_ser.isna()].index.tolist()
        raise RuntimeError(
            f"$SPXTR is missing global comparison dates: {missing_date_list[:5]}."
        )
    benchmark_total_value_ser = (
        benchmark_close_ser / float(benchmark_close_ser.iloc[0]) * capital_base_float
    ).rename("SPXTR")
    benchmark_return_ser = benchmark_total_value_ser.pct_change(fill_method=None)
    benchmark_return_ser.iloc[0] = 0.0

    global_index_bytes = "\n".join(
        date_ts.date().isoformat() for date_ts in global_idx
    ).encode("utf-8")
    global_index_sha256_str = hashlib.sha256(global_index_bytes).hexdigest()
    total_value_df.index.name = "date"
    return_df.index.name = "date"
    return total_value_df, return_df, benchmark_return_ser, global_index_sha256_str


def expected_shortfall_loss_float(
    return_arr: np.ndarray,
    quantile_float: float = 0.05,
) -> float:
    clean_return_arr = np.asarray(return_arr, dtype=float)
    clean_return_arr = clean_return_arr[np.isfinite(clean_return_arr)]
    if clean_return_arr.size == 0:
        return float("nan")
    tail_cutoff_float = float(np.quantile(clean_return_arr, quantile_float))
    tail_return_arr = clean_return_arr[clean_return_arr <= tail_cutoff_float]
    return float(max(0.0, -float(np.mean(tail_return_arr))))


def calculate_path_metric_dict(
    total_value_ser: pd.Series,
    benchmark_return_ser: pd.Series | None = None,
    annualization_day_int: int = 252,
    es_quantile_float: float = 0.05,
) -> dict[str, float | int]:
    clean_total_value_ser = total_value_ser.astype(float).dropna()
    if len(clean_total_value_ser) < 2:
        raise ValueError("At least two total-value observations are required.")
    return_ser = clean_total_value_ser.pct_change(fill_method=None).iloc[1:]
    observation_count_int = int(len(return_ser))
    terminal_multiple_float = float(
        clean_total_value_ser.iloc[-1] / clean_total_value_ser.iloc[0]
    )
    cagr_float = float(
        terminal_multiple_float
        ** (float(annualization_day_int) / observation_count_int)
        - 1.0
    )
    daily_std_float = float(return_ser.std(ddof=1))
    annualized_volatility_float = daily_std_float * math.sqrt(annualization_day_int)
    sharpe_float = (
        float(return_ser.mean() / daily_std_float * math.sqrt(annualization_day_int))
        if daily_std_float > 0.0
        else float("nan")
    )
    drawdown_ser = clean_total_value_ser.div(clean_total_value_ser.cummax()).sub(1.0)
    max_drawdown_float = float(drawdown_ser.min())
    metric_dict: dict[str, float | int] = {
        "observation_count_int": observation_count_int,
        "cagr_float": cagr_float,
        "annualized_volatility_float": annualized_volatility_float,
        "sharpe_float": sharpe_float,
        "max_drawdown_float": max_drawdown_float,
        "calmar_float": (
            cagr_float / abs(max_drawdown_float)
            if max_drawdown_float < 0.0
            else float("nan")
        ),
        "es5_loss_float": expected_shortfall_loss_float(
            return_ser.to_numpy(dtype=float),
            quantile_float=es_quantile_float,
        ),
        "terminal_multiple_float": terminal_multiple_float,
    }
    if benchmark_return_ser is not None:
        aligned_return_df = pd.concat(
            [
                return_ser.rename("path"),
                benchmark_return_ser.astype(float).rename("market"),
            ],
            axis=1,
        ).dropna()
        market_variance_float = float(aligned_return_df["market"].var(ddof=1))
        path_variance_float = float(aligned_return_df["path"].var(ddof=1))
        metric_dict["daily_market_correlation_float"] = (
            float(aligned_return_df["path"].corr(aligned_return_df["market"]))
            if len(aligned_return_df) >= 2
            and path_variance_float > 0.0
            and market_variance_float > 0.0
            else float("nan")
        )
        metric_dict["market_beta_float"] = (
            float(
                aligned_return_df[["path", "market"]].cov().loc["path", "market"]
                / market_variance_float
            )
            if market_variance_float > 0.0
            else float("nan")
        )
        monthly_return_df = (
            (1.0 + aligned_return_df).resample("ME").prod().sub(1.0)
        )
        monthly_metric_ready_bool = bool(
            len(monthly_return_df) >= 2
            and float(monthly_return_df["path"].var(ddof=1)) > 0.0
            and float(monthly_return_df["market"].var(ddof=1)) > 0.0
        )
        metric_dict["monthly_market_correlation_float"] = (
            float(monthly_return_df["path"].corr(monthly_return_df["market"]))
            if monthly_metric_ready_bool
            else float("nan")
        )
    return metric_dict


def calculate_headline_metric_df(
    total_value_df: pd.DataFrame,
    benchmark_total_value_ser: pd.Series,
    benchmark_return_ser: pd.Series,
    spec_dict: dict[str, Any],
) -> pd.DataFrame:
    stats_contract_dict = spec_dict["statistical_contract"]
    row_list: list[dict[str, Any]] = []
    for hypothesis_id_str in ALL_HYPOTHESIS_ID_TUPLE:
        row_list.append(
            {
                "hypothesis_id_str": hypothesis_id_str,
                "label_str": spec_dict["hypotheses"][hypothesis_id_str][
                    "label_str"
                ],
                **calculate_path_metric_dict(
                    total_value_df[hypothesis_id_str],
                    benchmark_return_ser=benchmark_return_ser,
                    annualization_day_int=int(
                        stats_contract_dict["annualization_day_int"]
                    ),
                    es_quantile_float=float(stats_contract_dict["es_quantile_float"]),
                ),
            }
        )
    row_list.append(
        {
            "hypothesis_id_str": "SPXTR",
            "label_str": "$SPXTR total return benchmark",
            **calculate_path_metric_dict(
                benchmark_total_value_ser,
                benchmark_return_ser=benchmark_return_ser,
                annualization_day_int=int(
                    stats_contract_dict["annualization_day_int"]
                ),
                es_quantile_float=float(stats_contract_dict["es_quantile_float"]),
            ),
        }
    )
    return pd.DataFrame(row_list)


def stationary_bootstrap_index_chunk_mat(
    *,
    sample_size_int: int,
    simulation_count_int: int,
    mean_block_length_int: int,
    random_seed_int: int,
    simulation_start_int: int,
) -> np.ndarray:
    """Generate bootstrap rows keyed only by their global simulation IDs."""

    if sample_size_int <= 0 or simulation_count_int <= 0:
        raise ValueError("Bootstrap sample and simulation counts must be positive.")
    if mean_block_length_int <= 0:
        raise ValueError("mean_block_length_int must be positive.")
    restart_probability_float = 1.0 / float(mean_block_length_int)
    index_mat = np.empty(
        (int(simulation_count_int), int(sample_size_int)),
        dtype=np.int64,
    )
    step_position_vec = np.arange(sample_size_int, dtype=np.int64)
    for local_simulation_idx_int in range(int(simulation_count_int)):
        global_simulation_idx_int = (
            int(simulation_start_int) + local_simulation_idx_int
        )
        simulation_seed_int = int(
            np.random.SeedSequence(
                [
                    int(random_seed_int),
                    int(mean_block_length_int),
                    int(global_simulation_idx_int),
                ]
            ).generate_state(1, dtype=np.uint32)[0]
        )
        simulation_rng_obj = np.random.default_rng(simulation_seed_int)
        restart_bool_vec = (
            simulation_rng_obj.random(sample_size_int)
            < restart_probability_float
        )
        restart_bool_vec[0] = True
        restart_draw_vec = simulation_rng_obj.integers(
            0,
            sample_size_int,
            size=sample_size_int,
            dtype=np.int64,
        )
        last_restart_position_vec = np.maximum.accumulate(
            np.where(restart_bool_vec, step_position_vec, 0)
        )
        within_block_offset_vec = (
            step_position_vec - last_restart_position_vec
        )
        index_mat[local_simulation_idx_int] = (
            restart_draw_vec[last_restart_position_vec]
            + within_block_offset_vec
        ) % sample_size_int
    return index_mat


def bootstrap_path_metric_array_dict(
    return_df: pd.DataFrame,
    *,
    simulation_count_int: int,
    mean_block_length_int: int,
    random_seed_int: int,
    annualization_day_int: int,
    es_quantile_float: float,
    chunk_size_int: int = 250,
) -> dict[str, dict[str, np.ndarray]]:
    """Synchronized stationary bootstrap with chunked path calculations."""

    clean_return_df = return_df.loc[:, ALL_HYPOTHESIS_ID_TUPLE].iloc[1:].astype(float)
    if clean_return_df.isna().any().any():
        raise ValueError("Bootstrap returns must be complete on the global index.")
    observation_count_int = int(len(clean_return_df))
    metric_array_by_hypothesis_dict: dict[str, dict[str, np.ndarray]] = {
        hypothesis_id_str: {
            "cagr_float": np.empty(simulation_count_int, dtype=float),
            "sharpe_float": np.empty(simulation_count_int, dtype=float),
            "max_drawdown_float": np.empty(simulation_count_int, dtype=float),
            "es5_loss_float": np.empty(simulation_count_int, dtype=float),
        }
        for hypothesis_id_str in ALL_HYPOTHESIS_ID_TUPLE
    }
    return_arr_by_hypothesis_dict = {
        hypothesis_id_str: clean_return_df[hypothesis_id_str].to_numpy(dtype=float)
        for hypothesis_id_str in ALL_HYPOTHESIS_ID_TUPLE
    }
    for chunk_start_int in range(0, simulation_count_int, chunk_size_int):
        chunk_end_int = min(chunk_start_int + chunk_size_int, simulation_count_int)
        chunk_simulation_count_int = chunk_end_int - chunk_start_int
        # *** CRITICAL*** One sampled position matrix is applied to every H0-H8
        # path in this chunk. Each global simulation ID owns its deterministic
        # seed, so changing chunk size cannot change any sampled path. No
        # bootstrap result feeds signal or sizing logic.
        chunk_index_mat = stationary_bootstrap_index_chunk_mat(
            sample_size_int=observation_count_int,
            simulation_count_int=chunk_simulation_count_int,
            mean_block_length_int=mean_block_length_int,
            random_seed_int=random_seed_int,
            simulation_start_int=chunk_start_int,
        )
        for hypothesis_id_str in ALL_HYPOTHESIS_ID_TUPLE:
            sampled_return_mat = return_arr_by_hypothesis_dict[hypothesis_id_str][
                chunk_index_mat
            ]
            log_terminal_wealth_vec = np.log1p(sampled_return_mat).sum(axis=1)
            cagr_vec = np.expm1(
                log_terminal_wealth_vec
                * float(annualization_day_int)
                / observation_count_int
            )
            daily_mean_vec = sampled_return_mat.mean(axis=1)
            daily_std_vec = sampled_return_mat.std(axis=1, ddof=1)
            sharpe_vec = np.divide(
                daily_mean_vec * math.sqrt(annualization_day_int),
                daily_std_vec,
                out=np.full_like(daily_mean_vec, np.nan),
                where=daily_std_vec > 0.0,
            )
            equity_mat = np.cumprod(1.0 + sampled_return_mat, axis=1)
            # Include the unit starting anchor so a loss on the first sampled
            # return is a drawdown instead of becoming the path's false peak.
            running_peak_mat = np.maximum(
                1.0,
                np.maximum.accumulate(equity_mat, axis=1),
            )
            drawdown_mat = equity_mat / running_peak_mat - 1.0
            max_drawdown_vec = drawdown_mat.min(axis=1)
            tail_cutoff_vec = np.quantile(
                sampled_return_mat,
                es_quantile_float,
                axis=1,
            )
            tail_mask_mat = sampled_return_mat <= tail_cutoff_vec[:, None]
            tail_sum_vec = np.where(tail_mask_mat, sampled_return_mat, 0.0).sum(
                axis=1
            )
            tail_count_vec = tail_mask_mat.sum(axis=1)
            es5_loss_vec = np.maximum(0.0, -tail_sum_vec / tail_count_vec)
            target_metric_dict = metric_array_by_hypothesis_dict[hypothesis_id_str]
            target_metric_dict["cagr_float"][chunk_start_int:chunk_end_int] = cagr_vec
            target_metric_dict["sharpe_float"][chunk_start_int:chunk_end_int] = sharpe_vec
            target_metric_dict["max_drawdown_float"][chunk_start_int:chunk_end_int] = (
                max_drawdown_vec
            )
            target_metric_dict["es5_loss_float"][chunk_start_int:chunk_end_int] = (
                es5_loss_vec
            )
    gc.collect()
    return metric_array_by_hypothesis_dict


def centered_one_sided_p_value_float(
    bootstrap_estimate_arr: np.ndarray,
    observed_estimate_float: float,
    null_boundary_float: float = 0.0,
) -> float:
    """Centered-bootstrap p-value for the one-sided positive-effect test."""

    clean_bootstrap_arr = np.asarray(bootstrap_estimate_arr, dtype=float)
    clean_bootstrap_arr = clean_bootstrap_arr[np.isfinite(clean_bootstrap_arr)]
    if clean_bootstrap_arr.size == 0:
        return float("nan")
    distance_from_null_float = observed_estimate_float - null_boundary_float
    if distance_from_null_float <= 0.0:
        return 1.0
    centered_error_arr = clean_bootstrap_arr - observed_estimate_float
    extreme_count_int = int(
        np.count_nonzero(centered_error_arr >= distance_from_null_float)
    )
    return float((extreme_count_int + 1.0) / (clean_bootstrap_arr.size + 1.0))


def calculate_bootstrap_evidence(
    return_df: pd.DataFrame,
    headline_metric_df: pd.DataFrame,
    spec_dict: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    stats_contract_dict = spec_dict["statistical_contract"]
    observed_metric_by_id_dict = headline_metric_df.set_index(
        "hypothesis_id_str"
    ).to_dict(orient="index")
    block_length_int_list = [
        int(stats_contract_dict["primary_mean_block_length_int"]),
        *[
            int(value_int)
            for value_int in stats_contract_dict[
                "sensitivity_mean_block_length_int_list"
            ]
        ],
    ]
    summary_row_list: list[dict[str, Any]] = []
    candidate_delta_frame_list: list[pd.DataFrame] = []
    p_value_by_block_and_candidate_dict: dict[
        tuple[int, str], dict[str, float]
    ] = {}
    for block_length_int in block_length_int_list:
        print(
            f"stationary bootstrap | mean block {block_length_int} | "
            f"B={int(stats_contract_dict['simulation_count_int'])}",
            flush=True,
        )
        metric_array_by_id_dict = bootstrap_path_metric_array_dict(
            return_df,
            simulation_count_int=int(stats_contract_dict["simulation_count_int"]),
            mean_block_length_int=block_length_int,
            random_seed_int=int(stats_contract_dict["random_seed_int"]),
            annualization_day_int=int(stats_contract_dict["annualization_day_int"]),
            es_quantile_float=float(stats_contract_dict["es_quantile_float"]),
        )
        for candidate_id_str in SELECTABLE_HYPOTHESIS_ID_TUPLE:
            control_id_str = str(
                spec_dict["hypotheses"][candidate_id_str]["matched_control_id_str"]
            )
            tail_improvement_arr = (
                metric_array_by_id_dict["H0"]["es5_loss_float"]
                - metric_array_by_id_dict[candidate_id_str]["es5_loss_float"]
            )
            carry_improvement_arr = (
                metric_array_by_id_dict[candidate_id_str]["cagr_float"]
                - metric_array_by_id_dict[control_id_str]["cagr_float"]
            )
            observed_tail_improvement_float = float(
                observed_metric_by_id_dict["H0"]["es5_loss_float"]
                - observed_metric_by_id_dict[candidate_id_str]["es5_loss_float"]
            )
            observed_carry_improvement_float = float(
                observed_metric_by_id_dict[candidate_id_str]["cagr_float"]
                - observed_metric_by_id_dict[control_id_str]["cagr_float"]
            )
            tail_p_value_float = centered_one_sided_p_value_float(
                tail_improvement_arr,
                observed_tail_improvement_float,
            )
            carry_p_value_float = centered_one_sided_p_value_float(
                carry_improvement_arr,
                observed_carry_improvement_float,
            )
            candidate_p_value_float = max(
                tail_p_value_float,
                carry_p_value_float,
            )
            p_value_by_block_and_candidate_dict[
                (block_length_int, candidate_id_str)
            ] = {
                "tail_p_value_float": tail_p_value_float,
                "carry_p_value_float": carry_p_value_float,
                "candidate_p_value_float": candidate_p_value_float,
            }
            summary_row_list.append(
                {
                    "mean_block_length_int": block_length_int,
                    "candidate_id_str": candidate_id_str,
                    "matched_control_id_str": control_id_str,
                    "observed_tail_improvement_float": (
                        observed_tail_improvement_float
                    ),
                    "tail_improvement_p025_float": float(
                        np.quantile(tail_improvement_arr, 0.025)
                    ),
                    "tail_improvement_p500_float": float(
                        np.quantile(tail_improvement_arr, 0.500)
                    ),
                    "tail_improvement_p975_float": float(
                        np.quantile(tail_improvement_arr, 0.975)
                    ),
                    "tail_p_value_float": tail_p_value_float,
                    "observed_carry_improvement_float": (
                        observed_carry_improvement_float
                    ),
                    "carry_improvement_p025_float": float(
                        np.quantile(carry_improvement_arr, 0.025)
                    ),
                    "carry_improvement_p500_float": float(
                        np.quantile(carry_improvement_arr, 0.500)
                    ),
                    "carry_improvement_p975_float": float(
                        np.quantile(carry_improvement_arr, 0.975)
                    ),
                    "carry_p_value_float": carry_p_value_float,
                    "candidate_p_value_float": candidate_p_value_float,
                }
            )
            candidate_delta_frame_list.append(
                pd.DataFrame(
                    {
                        "mean_block_length_int": block_length_int,
                        "candidate_id_str": candidate_id_str,
                        "bootstrap_iteration_int": np.arange(
                            len(tail_improvement_arr),
                            dtype=int,
                        ),
                        "tail_improvement_float": tail_improvement_arr,
                        "carry_improvement_float": carry_improvement_arr,
                    }
                )
            )
        del metric_array_by_id_dict
        gc.collect()

    primary_block_length_int = int(
        stats_contract_dict["primary_mean_block_length_int"]
    )
    primary_p_value_arr = np.asarray(
        [
            p_value_by_block_and_candidate_dict[
                (primary_block_length_int, candidate_id_str)
            ]["candidate_p_value_float"]
            for candidate_id_str in SELECTABLE_HYPOTHESIS_ID_TUPLE
        ],
        dtype=float,
    )
    reject_bool_arr, adjusted_p_value_arr, _, _ = multipletests(
        primary_p_value_arr,
        alpha=float(stats_contract_dict["family_alpha_float"]),
        method="holm",
    )
    holm_df = pd.DataFrame(
        {
            "candidate_id_str": SELECTABLE_HYPOTHESIS_ID_TUPLE,
            "primary_candidate_p_value_float": primary_p_value_arr,
            "holm_adjusted_p_value_float": adjusted_p_value_arr,
            "holm_reject_bool": reject_bool_arr,
        }
    )
    bootstrap_summary_df = pd.DataFrame(summary_row_list)
    bootstrap_delta_df = pd.concat(candidate_delta_frame_list, ignore_index=True)
    return bootstrap_summary_df, bootstrap_delta_df, holm_df


def equal_observation_third_slice_tuple(
    observation_count_int: int,
) -> tuple[slice, slice, slice]:
    if observation_count_int < 6:
        raise ValueError("At least six observations are required for three thirds.")
    position_arr = np.arange(observation_count_int)
    split_position_list = np.array_split(position_arr, 3)
    return tuple(
        slice(int(position_vec[0]), int(position_vec[-1]) + 1)
        for position_vec in split_position_list
    )  # type: ignore[return-value]


def calculate_subperiod_metric_df(
    total_value_df: pd.DataFrame,
    benchmark_return_ser: pd.Series,
    spec_dict: dict[str, Any],
) -> pd.DataFrame:
    row_list: list[dict[str, Any]] = []
    return_observation_count_int = len(total_value_df) - 1
    covered_return_date_list: list[pd.Timestamp] = []
    for third_position_int, return_slice_obj in enumerate(
        equal_observation_third_slice_tuple(return_observation_count_int),
        start=1,
    ):
        if return_slice_obj.start is None or return_slice_obj.stop is None:
            raise RuntimeError("Subperiod return slices must have finite bounds.")
        # *** CRITICAL*** Slices are defined over realized return rows, not NAV
        # rows. Prepend the immediately preceding NAV as the local cash/equity
        # anchor so the boundary return enters exactly one chronological third.
        nav_start_position_int = int(return_slice_obj.start)
        nav_stop_position_int = int(return_slice_obj.stop) + 1
        subperiod_total_value_df = total_value_df.iloc[
            nav_start_position_int:nav_stop_position_int
        ]
        subperiod_return_date_idx = subperiod_total_value_df.index[1:]
        covered_return_date_list.extend(pd.DatetimeIndex(subperiod_return_date_idx))
        subperiod_benchmark_return_ser = benchmark_return_ser.loc[
            subperiod_total_value_df.index
        ]
        for hypothesis_id_str in ALL_HYPOTHESIS_ID_TUPLE:
            row_list.append(
                {
                    "subperiod_id_str": f"third_{third_position_int}",
                    "anchor_date_str": (
                        subperiod_total_value_df.index[0].date().isoformat()
                    ),
                    "start_date_str": subperiod_return_date_idx[0].date().isoformat(),
                    "end_date_str": (
                        subperiod_total_value_df.index[-1].date().isoformat()
                    ),
                    "hypothesis_id_str": hypothesis_id_str,
                    **calculate_path_metric_dict(
                        subperiod_total_value_df[hypothesis_id_str],
                        benchmark_return_ser=subperiod_benchmark_return_ser,
                        annualization_day_int=int(
                            spec_dict["statistical_contract"][
                                "annualization_day_int"
                            ]
                        ),
                        es_quantile_float=float(
                            spec_dict["statistical_contract"]["es_quantile_float"]
                        ),
                    ),
                }
            )
    expected_return_date_idx = pd.DatetimeIndex(total_value_df.index[1:])
    covered_return_date_idx = pd.DatetimeIndex(covered_return_date_list)
    if (
        len(covered_return_date_idx) != len(expected_return_date_idx)
        or covered_return_date_idx.has_duplicates
        or not covered_return_date_idx.equals(expected_return_date_idx)
    ):
        raise RuntimeError(
            "Chronological thirds must cover every non-anchor return exactly once."
        )
    return pd.DataFrame(row_list)


def calculate_crisis_metric_df(
    total_value_df: pd.DataFrame,
    spec_dict: dict[str, Any],
) -> pd.DataFrame:
    row_list: list[dict[str, Any]] = []
    for crisis_name_str, start_date_str, end_date_str in spec_dict[
        "crisis_diagnostic_list"
    ]:
        crisis_result_date_idx = total_value_df.loc[
            start_date_str:end_date_str
        ].index
        if len(crisis_result_date_idx) == 0:
            row_list.append(
                {
                    "crisis_name_str": crisis_name_str,
                    "status_str": "N/A",
                    "hypothesis_id_str": None,
                    "observation_count_int": 0,
                }
            )
            continue
        first_result_position_int = int(
            total_value_df.index.get_loc(crisis_result_date_idx[0])
        )
        if first_result_position_int == 0:
            crisis_total_value_df = total_value_df.loc[
                crisis_result_date_idx[0]:crisis_result_date_idx[-1]
            ]
        else:
            # *** CRITICAL*** The prior NAV is a measurement anchor only. It
            # keeps the return into the crisis window's first session instead
            # of silently dropping that boundary loss or gain.
            crisis_total_value_df = total_value_df.iloc[
                first_result_position_int - 1:
                int(total_value_df.index.get_loc(crisis_result_date_idx[-1])) + 1
            ]
        if len(crisis_total_value_df) < 2:
            row_list.append(
                {
                    "crisis_name_str": crisis_name_str,
                    "status_str": "N/A",
                    "hypothesis_id_str": None,
                    "observation_count_int": 0,
                }
            )
            continue
        for hypothesis_id_str in ALL_HYPOTHESIS_ID_TUPLE:
            metric_dict = calculate_path_metric_dict(
                crisis_total_value_df[hypothesis_id_str]
            )
            row_list.append(
                {
                    "crisis_name_str": crisis_name_str,
                    "status_str": "available_diagnostic_only",
                    "hypothesis_id_str": hypothesis_id_str,
                    "anchor_date_str": crisis_total_value_df.index[0].date().isoformat(),
                    "start_date_str": crisis_total_value_df.index[1].date().isoformat(),
                    "end_date_str": crisis_total_value_df.index[-1].date().isoformat(),
                    "observation_count_int": metric_dict["observation_count_int"],
                    "cumulative_return_float": float(
                        crisis_total_value_df[hypothesis_id_str].iloc[-1]
                        / crisis_total_value_df[hypothesis_id_str].iloc[0]
                        - 1.0
                    ),
                    "max_drawdown_float": metric_dict["max_drawdown_float"],
                    "es5_loss_float": metric_dict["es5_loss_float"],
                }
            )
    return pd.DataFrame(row_list)


def calculate_shared_tail_event_frames(
    return_df: pd.DataFrame,
    benchmark_return_ser: pd.Series,
    total_value_df: pd.DataFrame,
    sleeve_equity_by_hypothesis_dict: dict[str, pd.DataFrame],
    quantile_float: float,
    capital_base_float: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    analysis_return_df = return_df.iloc[1:].copy()
    benchmark_analysis_return_ser = benchmark_return_ser.loc[
        analysis_return_df.index
    ].astype(float)
    event_date_by_set_dict = {
        "H0_worst_5pct": analysis_return_df.index[
            analysis_return_df["H0"]
            <= float(analysis_return_df["H0"].quantile(quantile_float))
        ],
        "SPXTR_worst_5pct": benchmark_analysis_return_ser.index[
            benchmark_analysis_return_ser
            <= float(benchmark_analysis_return_ser.quantile(quantile_float))
        ],
    }
    event_row_list: list[dict[str, Any]] = []
    contribution_row_list: list[dict[str, Any]] = []
    for event_set_str, event_date_idx in event_date_by_set_dict.items():
        for event_date_ts in event_date_idx:
            event_row_list.append(
                {
                    "event_set_str": event_set_str,
                    "date": event_date_ts,
                    "SPXTR_return_float": float(
                        benchmark_analysis_return_ser.loc[event_date_ts]
                    ),
                    **{
                        f"{hypothesis_id_str}_return_float": float(
                            analysis_return_df.loc[
                                event_date_ts,
                                hypothesis_id_str,
                            ]
                        )
                        for hypothesis_id_str in ALL_HYPOTHESIS_ID_TUPLE
                    },
                }
            )

        for hypothesis_id_str in ALL_HYPOTHESIS_ID_TUPLE:
            raw_sleeve_equity_df = sleeve_equity_by_hypothesis_dict[
                hypothesis_id_str
            ].loc[total_value_df.index]
            scale_float = capital_base_float / float(
                raw_sleeve_equity_df.iloc[0].sum()
            )
            sleeve_equity_df = raw_sleeve_equity_df * scale_float
            portfolio_previous_value_ser = sleeve_equity_df.sum(axis=1).shift(1)
            # *** CRITICAL*** Tail contribution at t uses sleeve value at t-1
            # through the realized dollar change at t. It is report-only and
            # selected from the frozen H0/SPX event sets after the run.
            contribution_df = sleeve_equity_df.diff().div(
                portfolio_previous_value_ser,
                axis=0,
            )
            for event_date_ts in event_date_idx:
                if event_date_ts not in contribution_df.index:
                    continue
                for source_id_str in contribution_df.columns:
                    contribution_row_list.append(
                        {
                            "event_set_str": event_set_str,
                            "date": event_date_ts,
                            "hypothesis_id_str": hypothesis_id_str,
                            "source_id_str": source_id_str,
                            "portfolio_return_contribution_float": float(
                                contribution_df.loc[event_date_ts, source_id_str]
                            ),
                        }
                    )
    return pd.DataFrame(event_row_list), pd.DataFrame(contribution_row_list)


def calculate_rolling_market_correlation_df(
    return_df: pd.DataFrame,
    benchmark_return_ser: pd.Series,
    window_day_int: int = 126,
) -> pd.DataFrame:
    rolling_correlation_df = pd.DataFrame(index=return_df.index)
    # *** CRITICAL*** This trailing correlation includes only returns through t
    # and is a post-run diagnostic. It cannot influence any H0-H8 weight.
    for hypothesis_id_str in ALL_HYPOTHESIS_ID_TUPLE:
        rolling_correlation_df[hypothesis_id_str] = (
            return_df[hypothesis_id_str]
            .rolling(window_day_int, min_periods=window_day_int)
            .corr(benchmark_return_ser)
        )
    rolling_correlation_df.index.name = "date"
    return rolling_correlation_df


def selected_active_sharpe_diagnostic_dict(
    active_return_ser: pd.Series,
    *,
    trial_count_int: int,
    annualization_day_int: int,
) -> dict[str, float | int]:
    clean_return_arr = (
        active_return_ser.astype(float)
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .to_numpy(dtype=float)
    )
    observation_count_int = int(clean_return_arr.size)
    if observation_count_int < 3 or float(np.std(clean_return_arr, ddof=1)) == 0.0:
        return {
            "observation_count_int": observation_count_int,
            "trial_count_int": trial_count_int,
            "active_sharpe_float": float("nan"),
            "expected_max_sharpe_float": float("nan"),
            "approximate_selection_probability_float": float("nan"),
        }
    daily_sharpe_float = float(
        np.mean(clean_return_arr) / np.std(clean_return_arr, ddof=1)
    )
    skew_float = float(skew(clean_return_arr, bias=False))
    pearson_kurtosis_float = float(
        kurtosis(clean_return_arr, fisher=False, bias=False)
    )
    sharpe_variance_float = float(
        (
            1.0
            - skew_float * daily_sharpe_float
            + (pearson_kurtosis_float - 1.0)
            / 4.0
            * daily_sharpe_float**2
        )
        / (observation_count_int - 1)
    )
    if not np.isfinite(sharpe_variance_float) or sharpe_variance_float <= 0.0:
        return {
            "observation_count_int": observation_count_int,
            "trial_count_int": trial_count_int,
            "active_sharpe_float": daily_sharpe_float
            * math.sqrt(annualization_day_int),
            "expected_max_sharpe_float": float("nan"),
            "approximate_selection_probability_float": float("nan"),
        }
    euler_gamma_float = 0.5772156649015329
    expected_max_daily_sharpe_float = math.sqrt(sharpe_variance_float) * (
        (1.0 - euler_gamma_float) * norm.ppf(1.0 - 1.0 / trial_count_int)
        + euler_gamma_float
        * norm.ppf(1.0 - 1.0 / (trial_count_int * math.e))
    )
    z_score_float = float(
        (daily_sharpe_float - expected_max_daily_sharpe_float)
        / math.sqrt(sharpe_variance_float)
    )
    return {
        "observation_count_int": observation_count_int,
        "trial_count_int": trial_count_int,
        "active_sharpe_float": daily_sharpe_float
        * math.sqrt(annualization_day_int),
        "active_return_skew_float": skew_float,
        "active_return_pearson_kurtosis_float": pearson_kurtosis_float,
        "expected_max_sharpe_float": expected_max_daily_sharpe_float
        * math.sqrt(annualization_day_int),
        "approximate_selection_probability_float": float(norm.cdf(z_score_float)),
    }


def calculate_selected_active_sharpe_df(
    return_df: pd.DataFrame,
    spec_dict: dict[str, Any],
) -> pd.DataFrame:
    stats_contract_dict = spec_dict["statistical_contract"]
    row_list: list[dict[str, Any]] = []
    for candidate_id_str in SELECTABLE_HYPOTHESIS_ID_TUPLE:
        row_list.append(
            {
                "candidate_id_str": candidate_id_str,
                "comparison_str": f"{candidate_id_str}_minus_H0_daily_active_return",
                **selected_active_sharpe_diagnostic_dict(
                    return_df[candidate_id_str].iloc[1:]
                    - return_df["H0"].iloc[1:],
                    trial_count_int=int(
                        stats_contract_dict[
                            "cumulative_related_path_lower_bound_int"
                        ]
                    ),
                    annualization_day_int=int(
                        stats_contract_dict["annualization_day_int"]
                    ),
                ),
            }
        )
    return pd.DataFrame(row_list)


def aggregate_book_transaction_evidence(
    spec_dict: dict[str, Any],
    output_dir_path: Path,
    global_idx: pd.DatetimeIndex,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    detail_frame_list: list[pd.DataFrame] = []
    summary_row_list: list[dict[str, Any]] = []
    for hypothesis_id_str, hypothesis_dict in spec_dict["hypotheses"].items():
        transaction_frame_list: list[pd.DataFrame] = []
        for source_id_str, _weight_obj in hypothesis_dict["source_weight_list"]:
            transaction_file_path = (
                output_dir_path
                / "source_transactions"
                / f"{source_id_str}.csv.gz"
            )
            transaction_df = pd.read_csv(transaction_file_path)
            if len(transaction_df) == 0:
                continue
            transaction_df["date"] = pd.to_datetime(transaction_df["date"]).dt.normalize()
            # *** CRITICAL*** Order evidence must describe the same exact book
            # window as the measured H0-H8 paths. Pre-global transactions are
            # lineage only and cannot enter overlap, cost, or capacity claims.
            transaction_df = transaction_df.loc[
                transaction_df["date"].isin(global_idx)
            ].copy()
            if len(transaction_df) == 0:
                continue
            transaction_frame_list.append(transaction_df)
        if len(transaction_frame_list) == 0:
            continue
        book_transaction_df = pd.concat(transaction_frame_list, ignore_index=True)
        book_transaction_df["gross_notional_float"] = book_transaction_df[
            "signed_notional_float"
        ].astype(float).abs()
        grouped_df = (
            book_transaction_df.groupby(["date", "asset_str"], as_index=False)
            .agg(
                gross_notional_float=("gross_notional_float", "sum"),
                net_notional_float=("signed_notional_float", "sum"),
                commission_float=("commission_float", "sum"),
                order_count_int=("asset_str", "size"),
                source_count_int=("source_id_str", "nunique"),
                source_id_list_str=(
                    "source_id_str",
                    lambda value_ser: "|".join(sorted(set(value_ser.astype(str)))),
                ),
            )
        )
        grouped_df.insert(0, "hypothesis_id_str", hypothesis_id_str)
        grouped_df["same_day_same_symbol_overlap_bool"] = (
            grouped_df["source_count_int"] > 1
        )
        detail_frame_list.append(grouped_df)
        overlap_df = grouped_df.loc[
            grouped_df["same_day_same_symbol_overlap_bool"]
        ]
        summary_row_list.append(
            {
                "hypothesis_id_str": hypothesis_id_str,
                "transaction_count_int": int(len(book_transaction_df)),
                "same_day_symbol_row_count_int": int(len(grouped_df)),
                "overlap_row_count_int": int(len(overlap_df)),
                "gross_notional_float": float(
                    book_transaction_df["gross_notional_float"].sum()
                ),
                "maximum_same_day_symbol_gross_notional_float": float(
                    grouped_df["gross_notional_float"].max()
                ),
                "maximum_overlap_gross_notional_float": (
                    float(overlap_df["gross_notional_float"].max())
                    if len(overlap_df) > 0
                    else 0.0
                ),
                "adv_capacity_status_str": "not_computed_book_level",
            }
        )
    detail_df = (
        pd.concat(detail_frame_list, ignore_index=True)
        if detail_frame_list
        else pd.DataFrame()
    )
    return detail_df, pd.DataFrame(summary_row_list)


def load_source_metadata_by_id_dict(
    spec_dict: dict[str, Any],
    output_dir_path: Path,
) -> dict[str, dict[str, Any]]:
    return {
        source_id_str: json.loads(
            (
                output_dir_path
                / "source_metadata"
                / f"{source_id_str}.json"
            ).read_text(encoding="utf-8")
        )
        for source_id_str in spec_dict["source_runs"]
    }


def validate_source_lineage_dict(
    spec_dict: dict[str, Any],
    output_dir_path: Path,
    source_run_summary_df: pd.DataFrame,
    global_idx: pd.DatetimeIndex,
) -> dict[str, Any]:
    reason_list: list[str] = []
    expected_source_id_list = list(spec_dict["source_runs"])
    actual_source_id_list = source_run_summary_df["source_id_str"].astype(str).tolist()
    if actual_source_id_list != expected_source_id_list:
        reason_list.append("source_id_order_or_set_mismatch")
    if source_run_summary_df["source_id_str"].astype(str).duplicated().any():
        reason_list.append("duplicate_source_id")
    current_shared_hash_dict = shared_execution_dependency_hash_dict()
    portfolio_contract_dict = spec_dict["portfolio_contract"]
    frozen_end_date_str = str(portfolio_contract_dict["end_date_str"])
    capital_anchor_date_str = str(
        portfolio_contract_dict["capital_anchor_date_str"]
    )
    effective_execution_start_date_str = str(
        portfolio_contract_dict["effective_execution_start_date_str"]
    )
    requested_start_date_str = str(
        portfolio_contract_dict["requested_start_date_str"]
    )
    native_history_start_by_import_dict = spec_dict["lineage_contract"][
        "native_history_request_start_by_strategy_import"
    ]
    global_start_ts = pd.Timestamp(global_idx[0])

    for source_id_str, source_spec_dict in spec_dict["source_runs"].items():
        source_path = output_dir_path / "source_paths" / f"{source_id_str}.csv.gz"
        transaction_path = (
            output_dir_path / "source_transactions" / f"{source_id_str}.csv.gz"
        )
        metadata_path = output_dir_path / "source_metadata" / f"{source_id_str}.json"
        if not all(path_obj.is_file() for path_obj in (source_path, transaction_path, metadata_path)):
            reason_list.append(f"{source_id_str}:missing_checkpoint_file")
            continue
        metadata_dict = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata_dict.get("source_id_str") != source_id_str:
            reason_list.append(f"{source_id_str}:metadata_source_id_mismatch")
        if metadata_dict.get("strategy_import_str") != str(
            source_spec_dict["strategy_import_str"]
        ):
            reason_list.append(f"{source_id_str}:strategy_import_mismatch")
        expected_engine_request_start_date_str = str(
            source_spec_dict.get(
                "engine_request_start_date_str",
                effective_execution_start_date_str,
            )
        )
        if (
            metadata_dict.get("engine_request_start_date_str")
            != expected_engine_request_start_date_str
        ):
            reason_list.append(f"{source_id_str}:engine_request_start_mismatch")
        expected_native_history_start_date_str = str(
            native_history_start_by_import_dict[
                str(source_spec_dict["strategy_import_str"])
            ]
        )
        if (
            metadata_dict.get("native_history_request_start_date_str")
            != expected_native_history_start_date_str
        ):
            reason_list.append(f"{source_id_str}:native_history_request_mismatch")
        if (
            bool(
                spec_dict["lineage_contract"][
                    "requested_start_must_be_covered_bool"
                ]
            )
            and pd.Timestamp(expected_native_history_start_date_str)
            > pd.Timestamp(requested_start_date_str)
        ):
            reason_list.append(f"{source_id_str}:requested_start_not_covered")
        if not math.isclose(
            float(metadata_dict.get("allocated_capital_float", np.nan)),
            float(source_spec_dict["allocated_capital_float"]),
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            reason_list.append(f"{source_id_str}:capital_mismatch")
        if metadata_dict.get("run_variant_kwargs_dict", {}) != dict(
            source_spec_dict.get("run_variant_kwargs_dict", {})
        ):
            reason_list.append(f"{source_id_str}:run_kwargs_mismatch")
        if metadata_dict.get("actual_end_date_str") != frozen_end_date_str:
            reason_list.append(f"{source_id_str}:end_date_mismatch")
        if metadata_dict.get("actual_start_date_str") != capital_anchor_date_str:
            reason_list.append(f"{source_id_str}:cash_anchor_mismatch")
        if (
            metadata_dict.get("strategy_result_start_date_str")
            != effective_execution_start_date_str
        ):
            reason_list.append(f"{source_id_str}:effective_start_mismatch")
        if pd.Timestamp(metadata_dict.get("actual_start_date_str")) != global_start_ts:
            reason_list.append(f"{source_id_str}:does_not_match_global_start")
        if metadata_dict.get("source_path_sha256_str") != sha256_file_str(source_path):
            reason_list.append(f"{source_id_str}:source_path_hash_mismatch")
        if metadata_dict.get("transaction_path_sha256_str") != sha256_file_str(
            transaction_path
        ):
            reason_list.append(f"{source_id_str}:transaction_hash_mismatch")
        module_path = Path(str(metadata_dict.get("module_path_str", "")))
        if not module_path.is_file() or metadata_dict.get(
            "module_sha256_str"
        ) != sha256_file_str(module_path):
            reason_list.append(f"{source_id_str}:module_hash_mismatch")
        if metadata_dict.get(
            "shared_execution_dependency_hash_dict"
        ) != current_shared_hash_dict:
            reason_list.append(f"{source_id_str}:shared_dependency_hash_mismatch")
        source_path_df = read_source_path_df(source_path)
        expected_anchor_capital_float = float(
            source_spec_dict["allocated_capital_float"]
        )
        if (
            source_path_df.index[0] != pd.Timestamp(capital_anchor_date_str)
            or not math.isclose(
                float(source_path_df.iloc[0]["total_value_float"]),
                expected_anchor_capital_float,
                rel_tol=0.0,
                abs_tol=1e-6,
            )
            or not math.isclose(
                float(source_path_df.iloc[0]["cash_float"]),
                expected_anchor_capital_float,
                rel_tol=0.0,
                abs_tol=1e-6,
            )
            or not math.isclose(
                float(source_path_df.iloc[0]["portfolio_value_float"]),
                0.0,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
        ):
            reason_list.append(f"{source_id_str}:cash_anchor_state_mismatch")
        if (
            len(source_path_df) < 2
            or source_path_df.index[1]
            != pd.Timestamp(effective_execution_start_date_str)
        ):
            reason_list.append(f"{source_id_str}:first_result_date_mismatch")
        if len(global_idx.difference(source_path_df.index)) > 0:
            reason_list.append(f"{source_id_str}:global_index_not_covered")

        matching_summary_df = source_run_summary_df.loc[
            source_run_summary_df["source_id_str"].astype(str) == source_id_str
        ]
        if len(matching_summary_df) != 1:
            reason_list.append(f"{source_id_str}:summary_row_count_invalid")
        else:
            summary_row_ser = matching_summary_df.iloc[0]
            if str(summary_row_ser.get("metadata_sha256_str")) != sha256_file_str(
                metadata_path
            ):
                reason_list.append(f"{source_id_str}:metadata_hash_mismatch")

    unique_reason_list = sorted(set(reason_list))
    return {
        "source_lineage_gate_bool": len(unique_reason_list) == 0,
        "reason_list": unique_reason_list,
        "validated_source_count_int": len(expected_source_id_list),
        "global_start_date_str": global_idx[0].date().isoformat(),
        "global_end_date_str": global_idx[-1].date().isoformat(),
        "shared_execution_dependency_hash_dict": current_shared_hash_dict,
    }


def build_candidate_accounting_cost_gate_df(
    spec_dict: dict[str, Any],
    source_metadata_by_id_dict: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    expected_family_by_candidate_dict = {
        "H1": "tactical_fixed_income",
        "H2": "tactical_fixed_income",
        "H3": "adaptive_macro_core5",
        "H4": "adaptive_macro_core5",
    }
    cost_contract_dict = spec_dict["cost_financing_contract"]
    profile_by_family_dict = spec_dict["passive_bil_contract"][
        "accounting_profile_by_candidate_family"
    ]
    row_list: list[dict[str, Any]] = []
    for candidate_id_str in SELECTABLE_HYPOTHESIS_ID_TUPLE:
        candidate_source_id_str = str(
            spec_dict["hypotheses"][candidate_id_str]["source_weight_list"][-1][0]
        )
        control_id_str = str(
            spec_dict["hypotheses"][candidate_id_str]["matched_control_id_str"]
        )
        control_source_id_str = str(
            spec_dict["hypotheses"][control_id_str]["source_weight_list"][-1][0]
        )
        family_str = expected_family_by_candidate_dict[candidate_id_str]
        profile_contract_dict = profile_by_family_dict[family_str]
        candidate_metadata_dict = source_metadata_by_id_dict[candidate_source_id_str]
        control_metadata_dict = source_metadata_by_id_dict[control_source_id_str]
        expected_withholding_float = float(
            profile_contract_dict["dividend_withholding_rate_float"]
        )
        expected_cash_policy_str = str(
            profile_contract_dict["positive_cash_rate_policy_str"]
        )
        accounting_gate_bool = bool(
            math.isclose(
                float(candidate_metadata_dict["dividend_withholding_rate_float"]),
                expected_withholding_float,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            and math.isclose(
                float(control_metadata_dict["dividend_withholding_rate_float"]),
                expected_withholding_float,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            and str(candidate_metadata_dict["positive_cash_rate_policy_str"])
            == expected_cash_policy_str
            and str(control_metadata_dict["positive_cash_rate_policy_str"])
            == expected_cash_policy_str
            and str(candidate_metadata_dict["negative_cash_financing_policy_str"])
            == "not_modeled"
            and str(control_metadata_dict["negative_cash_financing_policy_str"])
            == "not_modeled"
            and str(candidate_metadata_dict["execution_adjustment_str"])
            == "CAPITALSPECIAL"
            and str(control_metadata_dict["execution_adjustment_str"])
            == "CAPITALSPECIAL"
            and str(control_metadata_dict["run_variant_kwargs_dict"]["accounting_profile_str"])
            == str(profile_contract_dict["accounting_profile_str"])
        )
        cost_gate_bool = True
        for metadata_dict in (candidate_metadata_dict, control_metadata_dict):
            cost_gate_bool = bool(
                cost_gate_bool
                and math.isclose(
                    float(metadata_dict["slippage_per_side_float"]),
                    float(profile_contract_dict["slippage_per_side_float"]),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                and math.isclose(
                    float(metadata_dict["commission_per_share_float"]),
                    float(profile_contract_dict["commission_per_share_float"]),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
                and math.isclose(
                    float(metadata_dict["commission_minimum_float"]),
                    float(profile_contract_dict["commission_minimum_float"]),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
            )
        negative_cash_day_count_int = int(
            candidate_metadata_dict["negative_cash_day_count_int"]
        ) + int(control_metadata_dict["negative_cash_day_count_int"])
        financing_gate_bool = bool(
            negative_cash_day_count_int
            <= int(
                cost_contract_dict[
                    "maximum_candidate_or_control_negative_cash_day_count_int"
                ]
            )
        )
        row_list.append(
            {
                "candidate_id_str": candidate_id_str,
                "candidate_source_id_str": candidate_source_id_str,
                "matched_control_id_str": control_id_str,
                "control_source_id_str": control_source_id_str,
                "accounting_profile_str": profile_contract_dict[
                    "accounting_profile_str"
                ],
                "accounting_compatibility_gate_bool": accounting_gate_bool,
                "matched_cost_gate_bool": cost_gate_bool,
                "negative_cash_day_count_int": negative_cash_day_count_int,
                "unmodeled_financing_gate_bool": financing_gate_bool,
                "accounting_cost_financing_gate_bool": bool(
                    accounting_gate_bool and cost_gate_bool and financing_gate_bool
                ),
            }
        )
    return pd.DataFrame(row_list)


def load_candidate_capacity_evidence_df(
    spec_dict: dict[str, Any],
    global_idx: pd.DatetimeIndex,
    *,
    global_index_sha256_str: str,
    spec_sha256_str: str,
    source_metadata_by_id_dict: dict[str, dict[str, Any]],
    norgate_database_vintage_dict: dict[str, Any],
) -> pd.DataFrame:
    """Record Capacity context while keeping Phase 1 authorization closed."""

    capacity_contract_dict = spec_dict["capacity_contract"]
    portfolio_contract_dict = spec_dict["portfolio_contract"]
    if global_idx[0] != pd.Timestamp(portfolio_contract_dict["capital_anchor_date_str"]):
        raise RuntimeError("Capacity context received the wrong global cash anchor.")
    if not bool(
        capacity_contract_dict["current_study_capacity_gate_must_remain_false_bool"]
    ):
        raise RuntimeError("Phase 1 Capacity gate is not frozen closed.")
    if bool(capacity_contract_dict["external_artifact_can_clear_current_study_bool"]):
        raise RuntimeError("External Capacity evidence cannot authorize Phase 1.")
    del global_index_sha256_str, spec_sha256_str, norgate_database_vintage_dict

    candidate_family_by_id_dict = {
        "H1": "tactical_fixed_income",
        "H2": "tactical_fixed_income",
        "H3": "adaptive_macro_core5",
        "H4": "adaptive_macro_core5",
    }
    row_list: list[dict[str, Any]] = []
    for candidate_id_str, family_str in candidate_family_by_id_dict.items():
        candidate_source_id_str = str(
            spec_dict["hypotheses"][candidate_id_str]["source_weight_list"][-1][0]
        )
        candidate_metadata_dict = source_metadata_by_id_dict[
            candidate_source_id_str
        ]
        relative_path_str = str(
            capacity_contract_dict["exact_artifact_path_by_candidate"][
                candidate_id_str
            ]
        )
        capacity_path = REPO_ROOT_PATH / relative_path_str
        prior_context_relative_path_str = str(
            capacity_contract_dict["prior_artifact_by_strategy"][family_str]
        )
        prior_context_path = REPO_ROOT_PATH / prior_context_relative_path_str
        row_list.append(
            {
                "candidate_id_str": candidate_id_str,
                "candidate_source_id_str": candidate_source_id_str,
                "allocated_capital_float": float(
                    candidate_metadata_dict["allocated_capital_float"]
                ),
                "capacity_artifact_path_str": relative_path_str,
                "artifact_available_bool": capacity_path.is_file(),
                "capacity_artifact_sha256_str": (
                    sha256_file_str(capacity_path)
                    if capacity_path.is_file()
                    else None
                ),
                "prior_context_artifact_path_str": prior_context_relative_path_str,
                "prior_context_artifact_available_bool": prior_context_path.is_file(),
                "prior_context_artifact_sha256_str": (
                    sha256_file_str(prior_context_path)
                    if prior_context_path.is_file()
                    else None
                ),
                "period_matched_bool": False,
                "exact_lineage_matched_bool": False,
                "recommended_capacity_float": np.nan,
                "capacity_gate_bool": False,
                "reason_str": "separate_preregistered_capacity_phase_required",
            }
        )
    return pd.DataFrame(row_list)


def calculate_baseline_context_comparison_df(
    headline_metric_df: pd.DataFrame,
    spec_dict: dict[str, Any],
) -> pd.DataFrame:
    context_path = REPO_ROOT_PATH / str(
        spec_dict["portfolio_contract"][
            "prior_baseline_context_summary_path_str"
        ]
    )
    h0_metric_ser = headline_metric_df.set_index("hypothesis_id_str").loc["H0"]
    if not context_path.is_file():
        return pd.DataFrame(
            [
                {
                    "status_str": "prior_context_missing",
                    "context_path_str": str(context_path.relative_to(REPO_ROOT_PATH)),
                }
            ]
        )
    context_dict = json.loads(context_path.read_text(encoding="utf-8"))
    return pd.DataFrame(
        [
            {
                "status_str": "historical_context_only_not_inference",
                "context_path_str": str(context_path.relative_to(REPO_ROOT_PATH)),
                "context_sha256_str": sha256_file_str(context_path),
                "fresh_h0_cagr_float": float(h0_metric_ser["cagr_float"]),
                "prior_h0_cagr_float": float(context_dict["ann_return_pct"]) / 100.0,
                "cagr_delta_float": float(h0_metric_ser["cagr_float"])
                - float(context_dict["ann_return_pct"]) / 100.0,
                "fresh_h0_sharpe_float": float(h0_metric_ser["sharpe_float"]),
                "prior_h0_sharpe_float": float(context_dict["sharpe"]),
                "sharpe_delta_float": float(h0_metric_ser["sharpe_float"])
                - float(context_dict["sharpe"]),
                "fresh_h0_max_drawdown_float": float(
                    h0_metric_ser["max_drawdown_float"]
                ),
                "prior_h0_max_drawdown_float": float(
                    context_dict["max_drawdown_pct"]
                )
                / 100.0,
                "max_drawdown_delta_float": float(
                    h0_metric_ser["max_drawdown_float"]
                )
                - float(context_dict["max_drawdown_pct"]) / 100.0,
            }
        ]
    )


def evaluate_promotion_gate_df(
    headline_metric_df: pd.DataFrame,
    subperiod_metric_df: pd.DataFrame,
    bootstrap_summary_df: pd.DataFrame,
    holm_df: pd.DataFrame,
    candidate_capacity_evidence_df: pd.DataFrame,
    candidate_accounting_cost_gate_df: pd.DataFrame,
    source_lineage_gate_bool: bool,
    spec_dict: dict[str, Any],
) -> pd.DataFrame:
    metric_by_id_dict = headline_metric_df.set_index("hypothesis_id_str").to_dict(
        orient="index"
    )
    subperiod_metric_by_key_dict = {
        (str(row_ser["subperiod_id_str"]), str(row_ser["hypothesis_id_str"])): row_ser
        for _, row_ser in subperiod_metric_df.iterrows()
    }
    holm_by_candidate_dict = holm_df.set_index("candidate_id_str").to_dict(
        orient="index"
    )
    capacity_by_candidate_dict = candidate_capacity_evidence_df.set_index(
        "candidate_id_str"
    ).to_dict(orient="index")
    integrity_by_candidate_dict = candidate_accounting_cost_gate_df.set_index(
        "candidate_id_str"
    ).to_dict(orient="index")
    versus_h0_gate_dict = spec_dict["economic_gates"]["versus_h0"]
    versus_control_gate_dict = spec_dict["economic_gates"][
        "versus_matched_bil"
    ]
    subperiod_gate_dict = spec_dict["economic_gates"]["formal_subperiods"]
    stats_contract_dict = spec_dict["statistical_contract"]
    row_list: list[dict[str, Any]] = []
    for candidate_id_str in SELECTABLE_HYPOTHESIS_ID_TUPLE:
        control_id_str = str(
            spec_dict["hypotheses"][candidate_id_str]["matched_control_id_str"]
        )
        h0_metric_dict = metric_by_id_dict["H0"]
        candidate_metric_dict = metric_by_id_dict[candidate_id_str]
        control_metric_dict = metric_by_id_dict[control_id_str]
        max_drawdown_improvement_float = float(
            candidate_metric_dict["max_drawdown_float"]
            - h0_metric_dict["max_drawdown_float"]
        )
        es_reduction_fraction_float = float(
            (
                h0_metric_dict["es5_loss_float"]
                - candidate_metric_dict["es5_loss_float"]
            )
            / h0_metric_dict["es5_loss_float"]
        )
        cagr_delta_vs_h0_float = float(
            candidate_metric_dict["cagr_float"] - h0_metric_dict["cagr_float"]
        )
        sharpe_delta_vs_h0_float = float(
            candidate_metric_dict["sharpe_float"]
            - h0_metric_dict["sharpe_float"]
        )
        cagr_delta_vs_control_float = float(
            candidate_metric_dict["cagr_float"]
            - control_metric_dict["cagr_float"]
        )
        sharpe_delta_vs_control_float = float(
            candidate_metric_dict["sharpe_float"]
            - control_metric_dict["sharpe_float"]
        )
        max_drawdown_delta_vs_control_float = float(
            candidate_metric_dict["max_drawdown_float"]
            - control_metric_dict["max_drawdown_float"]
        )
        es_ratio_vs_control_float = float(
            candidate_metric_dict["es5_loss_float"]
            / control_metric_dict["es5_loss_float"]
        )
        h0_economic_gate_bool = bool(
            max_drawdown_improvement_float
            >= float(
                versus_h0_gate_dict[
                    "minimum_max_drawdown_improvement_float"
                ]
            )
            and es_reduction_fraction_float
            >= float(
                versus_h0_gate_dict[
                    "minimum_es5_loss_reduction_fraction_float"
                ]
            )
            and cagr_delta_vs_h0_float
            >= float(versus_h0_gate_dict["minimum_cagr_delta_float"])
            and sharpe_delta_vs_h0_float
            >= float(versus_h0_gate_dict["minimum_sharpe_delta_float"])
        )
        matched_control_gate_bool = bool(
            cagr_delta_vs_control_float
            >= float(versus_control_gate_dict["minimum_cagr_delta_float"])
            and sharpe_delta_vs_control_float
            >= float(versus_control_gate_dict["minimum_sharpe_delta_float"])
            and max_drawdown_delta_vs_control_float
            >= float(
                versus_control_gate_dict[
                    "minimum_max_drawdown_delta_float"
                ]
            )
            and es_ratio_vs_control_float
            <= float(versus_control_gate_dict["maximum_es5_loss_ratio_float"])
        )

        third_id_list = ["third_1", "third_2", "third_3"]
        es_improvement_third_count_int = 0
        positive_carry_third_count_int = 0
        maxdd_stability_bool = True
        cagr_stability_bool = True
        for third_id_str in third_id_list:
            h0_subperiod_ser = subperiod_metric_by_key_dict[(third_id_str, "H0")]
            candidate_subperiod_ser = subperiod_metric_by_key_dict[
                (third_id_str, candidate_id_str)
            ]
            control_subperiod_ser = subperiod_metric_by_key_dict[
                (third_id_str, control_id_str)
            ]
            es_improvement_third_count_int += int(
                float(candidate_subperiod_ser["es5_loss_float"])
                < float(h0_subperiod_ser["es5_loss_float"])
            )
            positive_carry_third_count_int += int(
                float(candidate_subperiod_ser["cagr_float"])
                > float(control_subperiod_ser["cagr_float"])
            )
            maxdd_stability_bool = bool(
                maxdd_stability_bool
                and float(candidate_subperiod_ser["max_drawdown_float"])
                - float(h0_subperiod_ser["max_drawdown_float"])
                >= -float(
                    subperiod_gate_dict[
                        "maximum_single_third_max_drawdown_worsening_vs_h0_float"
                    ]
                )
            )
            cagr_stability_bool = bool(
                cagr_stability_bool
                and float(candidate_subperiod_ser["cagr_float"])
                - float(h0_subperiod_ser["cagr_float"])
                >= -float(
                    subperiod_gate_dict[
                        "maximum_single_third_cagr_shortfall_vs_h0_float"
                    ]
                )
            )
        subperiod_gate_bool = bool(
            es_improvement_third_count_int
            >= int(
                subperiod_gate_dict[
                    "minimum_thirds_with_es5_improvement_vs_h0_int"
                ]
            )
            and positive_carry_third_count_int
            >= int(
                subperiod_gate_dict[
                    "minimum_thirds_with_positive_cagr_vs_matched_bil_int"
                ]
            )
            and maxdd_stability_bool
            and cagr_stability_bool
        )
        sensitivity_row_df = bootstrap_summary_df.loc[
            (bootstrap_summary_df["candidate_id_str"] == candidate_id_str)
            & bootstrap_summary_df["mean_block_length_int"].isin(
                stats_contract_dict["sensitivity_mean_block_length_int_list"]
            )
        ]
        sensitivity_gate_bool = bool(
            len(sensitivity_row_df)
            == len(stats_contract_dict["sensitivity_mean_block_length_int_list"])
            and (sensitivity_row_df["tail_p_value_float"] < 0.05).all()
            and (sensitivity_row_df["carry_p_value_float"] < 0.05).all()
        )
        holm_gate_bool = bool(
            holm_by_candidate_dict[candidate_id_str]["holm_reject_bool"]
        )
        statistical_gate_bool = bool(holm_gate_bool and sensitivity_gate_bool)
        capacity_gate_bool = bool(
            capacity_by_candidate_dict[candidate_id_str]["capacity_gate_bool"]
        )
        integrity_dict = integrity_by_candidate_dict[candidate_id_str]
        accounting_compatibility_gate_bool = bool(
            integrity_dict["accounting_compatibility_gate_bool"]
        )
        matched_cost_gate_bool = bool(integrity_dict["matched_cost_gate_bool"])
        unmodeled_financing_gate_bool = bool(
            integrity_dict["unmodeled_financing_gate_bool"]
        )
        accounting_cost_financing_gate_bool = bool(
            integrity_dict["accounting_cost_financing_gate_bool"]
        )
        non_capacity_promotion_gate_bool = bool(
            h0_economic_gate_bool
            and matched_control_gate_bool
            and subperiod_gate_bool
            and statistical_gate_bool
            and source_lineage_gate_bool
            and accounting_cost_financing_gate_bool
        )
        promotion_gate_bool = bool(
            non_capacity_promotion_gate_bool and capacity_gate_bool
        )
        gate_bool_by_name_dict = {
            "h0_economic": h0_economic_gate_bool,
            "matched_bil": matched_control_gate_bool,
            "formal_subperiod": subperiod_gate_bool,
            "holm_and_sensitivity": statistical_gate_bool,
            "source_lineage": source_lineage_gate_bool,
            "accounting_compatibility": accounting_compatibility_gate_bool,
            "matched_costs": matched_cost_gate_bool,
            "unmodeled_financing": unmodeled_financing_gate_bool,
            "capacity": capacity_gate_bool,
        }
        failed_gate_list = [
            gate_name_str
            for gate_name_str, gate_bool in gate_bool_by_name_dict.items()
            if not gate_bool
        ]
        verdict_str = (
            "forward_shadow_only"
            if promotion_gate_bool
            else (
                "exact_capacity_rerun_required_before_forward_shadow"
                if non_capacity_promotion_gate_bool
                else "reject_retain_H0"
            )
        )
        row_list.append(
            {
                "candidate_id_str": candidate_id_str,
                "matched_control_id_str": control_id_str,
                "max_drawdown_improvement_vs_h0_float": (
                    max_drawdown_improvement_float
                ),
                "es5_reduction_vs_h0_fraction_float": es_reduction_fraction_float,
                "cagr_delta_vs_h0_float": cagr_delta_vs_h0_float,
                "sharpe_delta_vs_h0_float": sharpe_delta_vs_h0_float,
                "cagr_delta_vs_matched_bil_float": cagr_delta_vs_control_float,
                "sharpe_delta_vs_matched_bil_float": sharpe_delta_vs_control_float,
                "max_drawdown_delta_vs_matched_bil_float": (
                    max_drawdown_delta_vs_control_float
                ),
                "es5_ratio_vs_matched_bil_float": es_ratio_vs_control_float,
                "es_improvement_third_count_int": es_improvement_third_count_int,
                "positive_carry_third_count_int": positive_carry_third_count_int,
                "maxdd_subperiod_stability_bool": maxdd_stability_bool,
                "cagr_subperiod_stability_bool": cagr_stability_bool,
                "h0_economic_gate_bool": h0_economic_gate_bool,
                "matched_control_gate_bool": matched_control_gate_bool,
                "subperiod_gate_bool": subperiod_gate_bool,
                "holm_gate_bool": holm_gate_bool,
                "bootstrap_sensitivity_gate_bool": sensitivity_gate_bool,
                "statistical_gate_bool": statistical_gate_bool,
                "source_lineage_gate_bool": source_lineage_gate_bool,
                "accounting_compatibility_gate_bool": (
                    accounting_compatibility_gate_bool
                ),
                "matched_cost_gate_bool": matched_cost_gate_bool,
                "unmodeled_financing_gate_bool": unmodeled_financing_gate_bool,
                "accounting_cost_financing_gate_bool": (
                    accounting_cost_financing_gate_bool
                ),
                "capacity_gate_bool": capacity_gate_bool,
                "non_capacity_promotion_gate_bool": (
                    non_capacity_promotion_gate_bool
                ),
                "promotion_gate_bool": promotion_gate_bool,
                "failed_gate_list_str": "|".join(failed_gate_list),
                "verdict_str": verdict_str,
            }
        )
    return pd.DataFrame(row_list)


def create_charts(
    total_value_df: pd.DataFrame,
    rolling_correlation_df: pd.DataFrame,
    promotion_gate_df: pd.DataFrame,
    output_dir_path: Path,
) -> None:
    chart_dir_path = output_dir_path / "charts"
    chart_dir_path.mkdir(parents=True, exist_ok=True)
    color_by_id_dict = {
        "H0": "#111827",
        "H1": "#2563eb",
        "H2": "#1d4ed8",
        "H3": "#d97706",
        "H4": "#b45309",
        "H5": "#64748b",
        "H6": "#94a3b8",
        "H7": "#475569",
        "H8": "#cbd5e1",
    }
    figure_obj, axis_arr = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    normalized_wealth_df = total_value_df.div(total_value_df.iloc[0])
    drawdown_df = total_value_df.div(total_value_df.cummax()).sub(1.0)
    for hypothesis_id_str in ALL_HYPOTHESIS_ID_TUPLE:
        line_width_float = 2.2 if hypothesis_id_str == "H0" else 1.15
        axis_arr[0].plot(
            normalized_wealth_df.index,
            normalized_wealth_df[hypothesis_id_str],
            label=hypothesis_id_str,
            linewidth=line_width_float,
            color=color_by_id_dict[hypothesis_id_str],
        )
        axis_arr[1].plot(
            drawdown_df.index,
            drawdown_df[hypothesis_id_str],
            linewidth=line_width_float,
            color=color_by_id_dict[hypothesis_id_str],
        )
    axis_arr[0].set_yscale("log")
    axis_arr[0].set_title("H0-H8 wealth on one exact common index")
    axis_arr[0].set_ylabel("Growth of $1")
    axis_arr[0].grid(alpha=0.2)
    axis_arr[0].legend(ncol=9, fontsize=8)
    axis_arr[1].set_title("Drawdown")
    axis_arr[1].set_ylabel("Drawdown")
    axis_arr[1].grid(alpha=0.2)
    figure_obj.tight_layout()
    figure_obj.savefig(chart_dir_path / "equity_and_drawdown.png", dpi=170)
    plt.close(figure_obj)

    figure_obj, axis_obj = plt.subplots(figsize=(12, 4.8))
    for hypothesis_id_str in ("H0", "H1", "H2", "H3", "H4"):
        axis_obj.plot(
            rolling_correlation_df.index,
            rolling_correlation_df[hypothesis_id_str],
            label=hypothesis_id_str,
            linewidth=2.0 if hypothesis_id_str == "H0" else 1.1,
            color=color_by_id_dict[hypothesis_id_str],
        )
    axis_obj.axhline(0.0, color="#111827", linewidth=0.7)
    axis_obj.set_title("Trailing 126-session correlation with $SPXTR")
    axis_obj.set_ylabel("Correlation")
    axis_obj.grid(alpha=0.2)
    axis_obj.legend(ncol=5)
    figure_obj.tight_layout()
    figure_obj.savefig(chart_dir_path / "rolling_market_correlation.png", dpi=170)
    plt.close(figure_obj)

    figure_obj, axis_arr = plt.subplots(1, 3, figsize=(12, 4))
    candidate_label_list = promotion_gate_df["candidate_id_str"].tolist()
    axis_arr[0].bar(
        candidate_label_list,
        promotion_gate_df["cagr_delta_vs_matched_bil_float"] * 100.0,
        color="#2563eb",
    )
    axis_arr[0].axhline(0.5, color="#b91c1c", linestyle="--", linewidth=1.0)
    axis_arr[0].set_title("CAGR minus matched BIL")
    axis_arr[0].set_ylabel("Percentage points")
    axis_arr[1].bar(
        candidate_label_list,
        promotion_gate_df["max_drawdown_improvement_vs_h0_float"] * 100.0,
        color="#0f766e",
    )
    axis_arr[1].axhline(2.0, color="#b91c1c", linestyle="--", linewidth=1.0)
    axis_arr[1].set_title("MaxDD improvement vs H0")
    axis_arr[2].bar(
        candidate_label_list,
        promotion_gate_df["es5_reduction_vs_h0_fraction_float"] * 100.0,
        color="#d97706",
    )
    axis_arr[2].axhline(10.0, color="#b91c1c", linestyle="--", linewidth=1.0)
    axis_arr[2].set_title("ES5 reduction vs H0")
    for axis_obj in axis_arr:
        axis_obj.grid(axis="y", alpha=0.2)
    figure_obj.tight_layout()
    figure_obj.savefig(chart_dir_path / "candidate_gate_deltas.png", dpi=170)
    plt.close(figure_obj)


def format_pct_str(value_obj: Any, digit_int: int = 2) -> str:
    if value_obj is None or not np.isfinite(float(value_obj)):
        return "N/A"
    return f"{float(value_obj) * 100.0:.{digit_int}f}%"


def format_float_str(value_obj: Any, digit_int: int = 3) -> str:
    if value_obj is None or not np.isfinite(float(value_obj)):
        return "N/A"
    return f"{float(value_obj):.{digit_int}f}"


def markdown_metric_table_str(headline_metric_df: pd.DataFrame) -> str:
    line_list = [
        "| מסלול | CAGR | תנודתיות | Sharpe | MaxDD | ES5 יומי | Beta | קורלציה יומית |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row_ser in headline_metric_df.iterrows():
        line_list.append(
            "| {id} | {cagr} | {vol} | {sharpe} | {maxdd} | {es} | {beta} | {corr} |".format(
                id=row_ser["hypothesis_id_str"],
                cagr=format_pct_str(row_ser["cagr_float"]),
                vol=format_pct_str(row_ser["annualized_volatility_float"]),
                sharpe=format_float_str(row_ser["sharpe_float"]),
                maxdd=format_pct_str(row_ser["max_drawdown_float"]),
                es=format_pct_str(row_ser["es5_loss_float"]),
                beta=format_float_str(row_ser.get("market_beta_float")),
                corr=format_float_str(
                    row_ser.get("daily_market_correlation_float")
                ),
            )
        )
    return "\n".join(line_list)


def markdown_gate_table_str(promotion_gate_df: pd.DataFrame) -> str:
    line_list = [
        "| מועמד | מול BIL: CAGR | שיפור MaxDD מול H0 | הפחתת ES5 | שער H0 | שער BIL | תתי תקופות | סטטיסטיקה | חשבונאות ועלויות | Capacity | החלטה |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for _, row_ser in promotion_gate_df.iterrows():
        pass_label_fn = lambda value_obj: "עבר" if bool(value_obj) else "נכשל"
        line_list.append(
            "| {candidate} | {carry} | {maxdd} | {es} | {h0} | {bil} | {sub} | {stats} | {integrity} | {capacity} | {verdict} |".format(
                candidate=row_ser["candidate_id_str"],
                carry=format_pct_str(
                    row_ser["cagr_delta_vs_matched_bil_float"]
                ),
                maxdd=format_pct_str(
                    row_ser["max_drawdown_improvement_vs_h0_float"]
                ),
                es=format_pct_str(
                    row_ser["es5_reduction_vs_h0_fraction_float"]
                ),
                h0=pass_label_fn(row_ser["h0_economic_gate_bool"]),
                bil=pass_label_fn(row_ser["matched_control_gate_bool"]),
                sub=pass_label_fn(row_ser["subperiod_gate_bool"]),
                stats=pass_label_fn(row_ser["statistical_gate_bool"]),
                integrity=pass_label_fn(
                    row_ser["accounting_cost_financing_gate_bool"]
                ),
                capacity=pass_label_fn(row_ser["capacity_gate_bool"]),
                verdict=str(row_ser["verdict_str"]),
            )
        )
    return "\n".join(line_list)


def write_hebrew_report(
    output_dir_path: Path,
    spec_dict: dict[str, Any],
    total_value_df: pd.DataFrame,
    headline_metric_df: pd.DataFrame,
    promotion_gate_df: pd.DataFrame,
    bootstrap_summary_df: pd.DataFrame,
    holm_df: pd.DataFrame,
    selected_active_sharpe_df: pd.DataFrame,
    candidate_capacity_evidence_df: pd.DataFrame,
    baseline_context_df: pd.DataFrame,
) -> Path:
    h0_metric_ser = headline_metric_df.set_index("hypothesis_id_str").loc["H0"]
    passed_candidate_list = promotion_gate_df.loc[
        promotion_gate_df["promotion_gate_bool"],
        "candidate_id_str",
    ].tolist()
    non_capacity_candidate_list = promotion_gate_df.loc[
        promotion_gate_df["non_capacity_promotion_gate_bool"],
        "candidate_id_str",
    ].tolist()
    if passed_candidate_list:
        verdict_he_str = (
            "המועמדים "
            + ", ".join(passed_candidate_list)
            + " עברו את החוזה המלא. הסמכות המקסימלית היא forward shadow בלבד."
        )
    elif non_capacity_candidate_list:
        verdict_he_str = (
            "אין אישור לקידום. "
            + ", ".join(non_capacity_candidate_list)
            + " עברו את השערים שאינם Capacity, אך נדרשת בדיקת Capacity תואמת תקופה לפני forward shadow."
        )
    else:
        verdict_he_str = (
            "אף אחד מ־H1–H4 לא עבר את החוזה הקפוא. ההחלטה היא להשאיר את H0, "
            "לעצור את הסבב ולא לכוונן משקלים על המדגם שכבר נצפה."
        )
    baseline_note_str = ""
    if len(baseline_context_df) > 0 and "cagr_delta_float" in baseline_context_df:
        baseline_note_str = (
            "ההשוואה לארטיפקט H0 הקודם היא הקשר היסטורי בלבד ואינה נחשבת "
            "apples-to-apples: המחקר הטרי מאתחל את כל המקורות במזומן באותו מועד "
            "לפני העסקה הראשונה. פערי CAGR/Sharpe/MaxDD "
            f"היו {format_pct_str(baseline_context_df.iloc[0]['cagr_delta_float'], 4)}, "
            f"{format_float_str(baseline_context_df.iloc[0]['sharpe_delta_float'], 5)} ו־"
            f"{format_pct_str(baseline_context_df.iloc[0]['max_drawdown_delta_float'], 4)} בהתאמה."
        )
    primary_block_int = int(
        spec_dict["statistical_contract"]["primary_mean_block_length_int"]
    )
    primary_bootstrap_df = bootstrap_summary_df.loc[
        bootstrap_summary_df["mean_block_length_int"] == primary_block_int
    ]
    holm_display_df = holm_df.set_index("candidate_id_str")
    bootstrap_line_list = []
    for _, row_ser in primary_bootstrap_df.iterrows():
        candidate_id_str = str(row_ser["candidate_id_str"])
        bootstrap_line_list.append(
            f"- **{candidate_id_str}:** p-tail={float(row_ser['tail_p_value_float']):.4f}, "
            f"p-carry={float(row_ser['carry_p_value_float']):.4f}, "
            f"Holm-adjusted={float(holm_display_df.loc[candidate_id_str, 'holm_adjusted_p_value_float']):.4f}."
        )
    capacity_line_list = []
    for _, row_ser in candidate_capacity_evidence_df.iterrows():
        capacity_line_list.append(
            f"- **{row_ser['candidate_id_str']}:** הארטיפקט מסתיים ב־"
            f"{row_ser.get('artifact_actual_end_date_str', 'N/A')}, recommended capacity="
            f"{row_ser.get('recommended_capacity_float', np.nan)}; מצב: {row_ser['reason_str']}."
        )
    selection_sharpe_line_list = []
    for _, row_ser in selected_active_sharpe_df.iterrows():
        selection_sharpe_line_list.append(
            f"- **{row_ser['candidate_id_str']}:** active Sharpe מול H0 "
            f"{format_float_str(row_ser['active_sharpe_float'])}, הסתברות בחירה מקורבת "
            f"{format_pct_str(row_ser['approximate_selection_probability_float'], 1)}."
        )

    report_str = f"""# בדיקת ערך מוסף למייצבי Ladder 4 — H0–H8

## סיכום מנהלים ופסק דין

זהו ניסוי מחקרי קפוא שבדק אם Tactical Fixed Income או Adaptive Macro CORE5 מוסיפים ערך ל־Ladder 4 מעבר להפחתת סיכון פסיבית באמצעות BIL. כל 19 צירופי האסטרטגיה וההון הורצו מחדש בסכום המדויק שלהם, H0–H8 נבנו ללא rebalance חיצוני, וכל המסלולים חולקים עוגן מזומן ב־**{total_value_df.index[0].date()}**. התשואה הממומשת הראשונה כוללת את העסקה והעלות של **{total_value_df.index[1].date()}**, והמדגם מסתיים ב־**{total_value_df.index[-1].date()}** עם {len(total_value_df) - 1:,} תשואות.

H0 הטרי הניב CAGR של **{format_pct_str(h0_metric_ser['cagr_float'])}**, Sharpe של **{format_float_str(h0_metric_ser['sharpe_float'])}**, MaxDD של **{format_pct_str(h0_metric_ser['max_drawdown_float'])}** ו־ES5 יומי של **{format_pct_str(h0_metric_ser['es5_loss_float'])}**. {verdict_he_str}

החולשה המרכזית אינה רק סטטיסטית: לשני השרוולים הפעילים אין כרגע Capacity תקופתי מאושר, ול־TFI ול־CORE5 נשארו פערי נתונים, מימון או borrow שמונעים סמכות הקצאה. גם מעבר מלא של הניסוי היה מאפשר רק forward shadow.

## מה נבדק ולמה

H0 הוא Ladder 4 הקיים: 16% DV2, ‏17% HPI, ‏25% NDX/VXN, ‏8% MOSAIC ו־34% TAA. בכל מועמד הושארו DV2 ו־HPI ללא שינוי. משקל המייצב, 5% או 10%, נלקח באופן יחסי משלושת התורמים NDX, MOSAIC ו־TAA:

$$
k = \\frac{{0.67-s}}{{0.67}}
$$

$$
w_{{NDX}} = 0.25k,\\quad w_{{MOSAIC}} = 0.08k,\\quad w_{{TAA}} = 0.34k
$$

H1/H2 משתמשים ב־Tactical Fixed Income ב־5%/10%; H3/H4 משתמשים ב־CORE5. ‏H5/H6 הם בקרות BIL התואמות ל־TFI בחוזה המס, ריבית המזומן ועלויות המסחר; H7/H8 תואמות ל־CORE5. לכן ההשוואות המבודדות ערך אקטיבי הן H1 מול H5, H2 מול H6, H3 מול H7 ו־H4 מול H8. השוואה מול H0 בלבד מערבבת alpha אפשרי עם עצם הורדת התקציב מהשרוולים המסוכנים.

## כללים מדויקים ותזמון

```text
היסטוריה עד Close_T לצורך warmup וסיגנל בלבד
            |
            v
  עוגן מזומן משותף: 2012-09-28
            |
            v
     החלטה וגודל מניות
            |
            |  גבול קריטי: אין שימוש ב-Open_(T+1) לצורך החלטה
            v
       מילוי ב-Open_(T+1)
            |
            v
  compounding עצמאי; אין outer rebalance
```

כל אסטרטגיה שמרה על חוזה הביצוע שלה. BIL נטען ב־CAPITALSPECIAL, נקנה פעם אחת במניות שלמות לפי Close קודם ומתמלא בפתיחה הבאה. בקרת TFI משתמשת ב־5bps לכל צד, אפס עמלה, דיבידנד ברוטו וריבית DGS3MO סיבתית על מזומן. בקרת CORE5 משתמשת ב־2.5bps, ‏‎$0.005 למניה ומינימום ‎$1, ניכוי דיבידנד של 25% וריבית מזומן של 0%. בשתיהן אין reinvestment ומימון מזומן שלילי אינו ממודל; הופעה שלו חוסמת קידום.

## נתונים ותכנון הבדיקה

- הון כולל: ‎$1,000,000. כל pod הורץ בהון המדויק שנגזר מהמשקל; לא נעשה scaling של stream ישן.
- היסטוריה מבוקשת לטעינת נתונים ו־warmup: 2004-01-01 עד 2026-08-14. הון ומסחר מתחילים מחדש לכל 19 המקורות ב־2012-10-01; 2012-09-28 הוא מצב מזומן משותף. שום position או NAV מלפני מועד זה אינו נכנס למסלול.
- חסרים: חיתוך index מדויק בלבד. אין `ffill`, ‏`bfill`, תשואה אפס או מזומן שמחליפים יום חסר. האפס היחיד הוא שורת עוגן המזומן; המעבר ממנה לתוצאה הראשונה שומר slippage, עמלה ו־P&L של יום המילוי.
- יקום מניות: אסטרטגיות DV2, HPI ו־MOSAIC משתמשות במנגנוני point-in-time הקיימים שלהן. המחקר לא החליף אותם ברשימת חברות נוכחית.
- מחירים: CAPITALSPECIAL לנכסים נסחרים; $SPXTR כמדד total-return. דיבידנדים מטופלים לפי חוזה כל strategy.
- vintage: הריצה היא fresh-only ללא resume. גרסת חבילת Norgate וזמני העדכון של כל בסיסי הנתונים נרשמים לפני ואחרי 19 המקורות; שינוי באמצע פוסל את הריצה. זו עדיין אינה תמונת raw-data מלאה.
- מרחב חיפוש מצטבר: לפחות **{int(spec_dict['statistical_contract']['cumulative_related_path_lower_bound_int'])}** מסלולים קשורים. H0 המלא הוא הקשר בלבד ואינו משתתף בבחירה או ב־Holm.

## תוצאות על המדגם המשותף

{markdown_metric_table_str(headline_metric_df)}

{baseline_note_str}

![עושר וירידות](charts/equity_and_drawdown.png)

## שערי ערך מוסף

מול H0 נדרש שיפור MaxDD של לפחות 2 נקודות אחוז, הפחתת ES5 של לפחות 10%, ויתור CAGR שאינו גדול מנקודת אחוז ושינוי Sharpe שאינו שלילי. מול BIL תואם נדרש יתרון CAGR של 0.5 נקודת אחוז, Sharpe שאינו גרוע ביותר מ־0.02, MaxDD שאינו גרוע ביותר מנקודת אחוז ו־ES5 שאינו גדול ביותר מ־5%.

{markdown_gate_table_str(promotion_gate_df)}

![פערים מול השערים](charts/candidate_gate_deltas.png)

## ראיה סטטיסטית ותתי תקופות

בוצע stationary bootstrap מסונכרן עם 10,000 חזרות, seed ‏20260829 ובלוק ממוצע ראשי של 63 ימי מסחר. אותם מיקומי זמן נדגמו לכל H0–H8. מבחן הזנב בודק שיפור ES5 מול H0; מבחן carry בודק CAGR מעל BIL התואם. p-value המועמד הוא המקסימום ביניהם, ולאחר מכן מופעל Holm על H1–H4. בלוקים 21 ו־126 הם sensitivities קפואים; הם אינם hypotheses חדשים.

{chr(10).join(bootstrap_line_list)}

המדגם חולק גם לשלושה שלישים כרונולוגיים שווי תצפיות. מועמד נדרש לשפר ES5 בשניים משלושה שלישים, לנצח את BIL ב־CAGR בשניים, ולא להחמיר באופן קיצוני את MaxDD או CAGR באף שליש. חלונות המשבר הקפואים הם diagnostics בלבד; חלון שאינו במדגם מסומן N/A ואינו נחשב הצלחה.

אבחון Sharpe נבחר על התשואה האקטיבית מול H0 משתמש ב־floor של 35 מסלולים קשורים. זהו קירוב לבחירה מרובה, לא DSR פורמלי ולא שער קידום:

{chr(10).join(selection_sharpe_line_list)}

## תלות בשוק וזנב

הטבלה הראשית מציגה beta וקורלציה יומית מול $SPXTR. קורלציה נמוכה אינה מפורשת אוטומטית כ־diversification מבני, משום שחלק מהאסטרטגיות עוברות למזומן או ל־BIL. הקורלציה המתגלגלת ל־126 sessions מציגה אם היחסים יציבים לאורך זמן. בנוסף נשמרו במפורש תשואות כל המסלולים באותם 5% ימים גרועים של H0 ובאותם 5% ימים גרועים של $SPXTR, במקום להשוות זנבות שנבחרו בנפרד.

![קורלציה מתגלגלת](charts/rolling_market_correlation.png)

## Capacity, עלויות ומה עדיין לא הוכח

העסקאות האמיתיות מכל source run, ורק בתוך ה־index הגלובלי, אוחדו לפי תאריך וסימבול כדי לחשוף חפיפת orders בין pods. זהו diagnostic של book overlap בלבד; אין ברפו כיום analyzer שמאחד ADV ו־impact לכל הספר. ארטיפקטים קודמים הם הקשר בלבד. שלב H0–H8 אינו רשאי לאשר Capacity גם אם קיים JSON חיצוני שטוען אחרת. אם מועמד עובר את כל השערים האחרים, יש להקפיא לפני עיון נוסף שלב Capacity נפרד: builder דטרמיניסטי, panel מחיר/Turnover שמור ומגובה hash, חישוב מחדש של כל שער מספרי, diagnostics לכל AUM, ו־baseline slippage של 5bps ל־TFI לעומת 2.5bps ל־CORE5.

{chr(10).join(capacity_line_list)}

מגבלות נוספות:

- תחילת ה־history המבוקשת בכל loader נאכפת מול 2004-01-01, אך זמינות בפועל עשויה להשתנות לפי נכס; אין כאן טענה של panel מלא ואחיד לכל סימבול מאז 2004.
- TFI משתמש ב־FRED current-vintage קפוא ולא ב־ALFRED point-in-time מלא; היסטוריית המימון היא proxy.
- CORE5 כולל short ב־DBC עם baseline borrow של 1% בשנה, אך ללא זמינות locate, recall ושיעור borrow ספציפי לחשבון.
- אין הוכחת partial fills, auction imbalance או TCA חי. whole shares, slippage ועמלות כן נמדדו מחדש בהון המדויק.
- H1–H4 נולדו לאחר צפייה ב־Ladder, MOSAIC, TFI ו־CORE5. bootstrap ו־Holm מצמצמים אופטימיות אך אינם יוצרים holdout היסטורי חדש.
- `rebalance: null` פירושו שהמשקלים הם הקצאה התחלתית בלבד ואז drift. אין לקרוא ל־5% או 10% משקל נשמר.

## המלצה סופית

**{verdict_he_str}**

אם אף מועמד לא עבר, אין לפתוח sweep חדש על 2.5%, ‏7.5%, תמהילי TFI/CORE5 או donors אחרים. פעולה כזאת תהיה tuning על אותו מדגם. כיוון מחקר עתידי אפשרי רק כחוזה חדש, מתועד מראש, או כ־forward observation. אין בדוח הזה אישור ל־allocation, ‏PAPER, ‏LIVE, broker, scheduler או release.

## אינדקס ארטיפקטים

- [המפרט הקפוא](research_spec_frozen.yaml)
- [מסלולים יומיים](global_daily_paths.csv.gz)
- [טבלת מדדים](headline_metrics.csv)
- [שערי קידום](promotion_gates.csv)
- [Bootstrap](bootstrap_summary.csv)
- [תתי תקופות](subperiod_metrics.csv)
- [משברים](crisis_metrics.csv)
- [אירועי זנב משותפים](shared_tail_events.csv.gz)
- [Capacity של המועמדים](candidate_capacity_evidence.csv)
- [בדיקת lineage](source_lineage_validation.json)
- [חשבונאות, עלויות ומימון](candidate_accounting_cost_gates.csv)
- [Manifest](run_manifest.json)
"""
    report_path = output_dir_path / "LADDER4_STABILIZER_VALUE_ADD_REPORT_HE.md"
    report_path.write_text(report_str, encoding="utf-8")
    return report_path


def git_context_dict() -> dict[str, Any]:
    def run_git_arg_list(arg_list: list[str]) -> str:
        process_obj = subprocess.run(
            ["git", *arg_list],
            cwd=REPO_ROOT_PATH,
            capture_output=True,
            text=True,
            check=False,
        )
        return process_obj.stdout.strip()

    status_str = run_git_arg_list(["status", "--porcelain=v1"])
    diff_process_obj = subprocess.run(
        ["git", "diff", "--binary", "HEAD"],
        cwd=REPO_ROOT_PATH,
        capture_output=True,
        check=False,
    )
    untracked_path_str = run_git_arg_list(
        ["ls-files", "--others", "--exclude-standard"]
    )
    untracked_code_hash_by_path_dict: dict[str, str] = {}
    allowed_prefix_tuple = ("alpha/", "strategies/", "data/", "scripts/research/")
    for relative_path_str in untracked_path_str.splitlines():
        normalized_relative_path_str = relative_path_str.replace("\\", "/")
        if not normalized_relative_path_str.startswith(allowed_prefix_tuple):
            continue
        if Path(normalized_relative_path_str).suffix.lower() not in {
            ".py",
            ".yaml",
            ".yml",
        }:
            continue
        untracked_path = REPO_ROOT_PATH / normalized_relative_path_str
        if untracked_path.is_file():
            untracked_code_hash_by_path_dict[normalized_relative_path_str] = (
                sha256_file_str(untracked_path)
            )
    return {
        "commit_sha_str": run_git_arg_list(["rev-parse", "HEAD"]),
        "dirty_bool": bool(status_str),
        "status_sha256_str": hashlib.sha256(status_str.encode("utf-8")).hexdigest(),
        "status_line_count_int": len(status_str.splitlines()) if status_str else 0,
        "tracked_binary_diff_sha256_str": hashlib.sha256(
            diff_process_obj.stdout
        ).hexdigest(),
        "tracked_binary_diff_size_byte_int": len(diff_process_obj.stdout),
        "untracked_code_hash_by_path_dict": untracked_code_hash_by_path_dict,
        "shared_execution_dependency_hash_dict": (
            shared_execution_dependency_hash_dict()
        ),
    }


def write_run_manifest(
    output_dir_path: Path,
    spec_path: Path,
    global_index_sha256_str: str,
    total_value_df: pd.DataFrame,
    source_run_summary_df: pd.DataFrame,
    spec_dict: dict[str, Any],
) -> Path:
    artifact_row_list = []
    manifest_path = output_dir_path / "run_manifest.json"
    for artifact_path in sorted(output_dir_path.rglob("*")):
        if not artifact_path.is_file() or artifact_path == manifest_path:
            continue
        artifact_row_list.append(
            {
                "relative_path_str": artifact_path.relative_to(
                    output_dir_path
                ).as_posix(),
                "size_byte_int": int(artifact_path.stat().st_size),
                "sha256_str": sha256_file_str(artifact_path),
            }
        )
    capacity_artifact_row_list = []
    capacity_contract_dict = spec_dict["capacity_contract"]
    capacity_path_item_list = [
        *(
            ("prior_context", str(relative_path_str))
            for relative_path_str in capacity_contract_dict[
                "prior_artifact_by_strategy"
            ].values()
        ),
        *(
            ("exact_candidate", str(relative_path_str))
            for relative_path_str in capacity_contract_dict[
                "exact_artifact_path_by_candidate"
            ].values()
        ),
    ]
    for artifact_role_str, relative_path_str in capacity_path_item_list:
        capacity_path = REPO_ROOT_PATH / relative_path_str
        capacity_artifact_row_list.append(
            {
                "artifact_role_str": artifact_role_str,
                "relative_path_str": relative_path_str,
                "available_bool": capacity_path.is_file(),
                "sha256_str": (
                    sha256_file_str(capacity_path)
                    if capacity_path.is_file()
                    else None
                ),
            }
        )
    manifest_dict = {
        "artifact_type_str": "frozen_portfolio_candidate_value_add_study",
        "study_id_str": spec_dict["study_id_str"],
        "completed_at_utc_str": utc_now_str(),
        "authority_str": spec_dict["authority_str"],
        "spec_path_str": str(spec_path),
        "spec_sha256_str": sha256_file_str(spec_path),
        "runner_path_str": str(Path(__file__).resolve()),
        "runner_sha256_str": sha256_file_str(Path(__file__).resolve()),
        "git_context_dict": git_context_dict(),
        "source_run_count_int": int(len(source_run_summary_df)),
        "source_run_row_list": source_run_summary_df.to_dict(orient="records"),
        "global_start_date_str": total_value_df.index[0].date().isoformat(),
        "global_end_date_str": total_value_df.index[-1].date().isoformat(),
        "global_observation_count_int": int(len(total_value_df)),
        "global_index_sha256_str": global_index_sha256_str,
        "bootstrap_contract_dict": spec_dict["statistical_contract"],
        "weight_contract_dict": spec_dict["hypotheses"],
        "capacity_input_artifact_row_list": capacity_artifact_row_list,
        "data_lineage_caveat_str": (
            "FRED inputs for Tactical Fixed Income are frozen and hashed by the "
            "strategy. Norgate database update timestamps and package version must "
            "remain identical from the first source through the final benchmark read; module, "
            "compact realized path, transaction, metadata, benchmark, and global "
            "index hashes are recorded. This is still not a full raw-data snapshot."
        ),
        "artifact_row_list": artifact_row_list,
    }
    write_json(manifest_path, manifest_dict)
    return manifest_path


def run_study(
    spec_path: Path,
    output_dir_path: Path,
    *,
    resume_bool: bool = False,
    enforce_frozen_contract_bool: bool = True,
) -> Path:
    spec_path = spec_path.resolve()
    output_dir_path = output_dir_path.resolve()
    if resume_bool:
        raise ValueError(
            "Frozen artifact runs forbid --resume so one study cannot mix code or "
            "mutable Norgate data vintages. Start a fresh output directory."
        )
    output_nonempty_bool = bool(
        output_dir_path.exists() and any(output_dir_path.iterdir())
    )
    if output_nonempty_bool:
        raise FileExistsError(
            f"Refusing to overwrite non-empty frozen output directory: {output_dir_path}"
        )
    output_dir_path.mkdir(parents=True, exist_ok=True)
    spec_dict = load_spec_dict(
        spec_path,
        enforce_frozen_contract_bool=enforce_frozen_contract_bool,
    )
    frozen_spec_copy_path = output_dir_path / "research_spec_frozen.yaml"
    shutil.copyfile(spec_path, frozen_spec_copy_path)
    write_json(
        output_dir_path / "hypothesis_registry.json",
        {
            "study_id_str": spec_dict["study_id_str"],
            "frozen_on_date_str": spec_dict["frozen_on_date_str"],
            "hypotheses": spec_dict["hypotheses"],
            "economic_gates": spec_dict["economic_gates"],
            "statistical_contract": spec_dict["statistical_contract"],
            "stop_rule": spec_dict["stop_rule"],
        },
    )
    append_jsonl(
        output_dir_path / "experiment_ledger.jsonl",
        {
            "event_str": "study_contract_frozen",
            "recorded_at_utc_str": utc_now_str(),
            "study_id_str": spec_dict["study_id_str"],
            "spec_sha256_str": sha256_file_str(frozen_spec_copy_path),
            "source_run_count_int": len(spec_dict["source_runs"]),
            "candidate_count_int": len(SELECTABLE_HYPOTHESIS_ID_TUPLE),
        },
    )
    norgate_vintage_start_dict = norgate_database_vintage_dict()
    write_json(
        output_dir_path / "norgate_database_vintage_start.json",
        norgate_vintage_start_dict,
    )
    source_run_summary_df = execute_source_runs(
        spec_dict,
        output_dir_path,
        resume_bool=False,
    )
    norgate_vintage_source_end_dict = norgate_database_vintage_dict()
    write_json(
        output_dir_path / "norgate_database_vintage_source_end.json",
        norgate_vintage_source_end_dict,
    )
    if norgate_vintage_source_end_dict != norgate_vintage_start_dict:
        raise RuntimeError(
            "Norgate database vintage changed during the 19-source run. Discard "
            "this output and rerun in a fresh directory."
        )
    append_jsonl(
        output_dir_path / "experiment_ledger.jsonl",
        {
            "event_str": "all_source_runs_completed_result_review_unlocked",
            "recorded_at_utc_str": utc_now_str(),
            "source_run_count_int": int(len(source_run_summary_df)),
        },
    )

    source_path_by_id_dict = load_all_source_path_dict(spec_dict, output_dir_path)
    # *** CRITICAL*** Freeze one exact 19-source index before any sleeve is
    # initialized. Every H0-H8 book starts from the same known cash anchor at
    # its approved weights; no pre-anchor performance can drift those weights.
    global_idx = build_global_source_index(spec_dict, source_path_by_id_dict)
    source_lineage_dict = validate_source_lineage_dict(
        spec_dict,
        output_dir_path,
        source_run_summary_df,
        global_idx,
    )
    write_json(
        output_dir_path / "source_lineage_validation.json",
        source_lineage_dict,
    )
    if not bool(source_lineage_dict["source_lineage_gate_bool"]):
        raise RuntimeError(
            "Frozen source lineage validation failed before result analysis: "
            f"{source_lineage_dict['reason_list']}."
        )
    (
        hypothesis_path_by_id_dict,
        sleeve_equity_by_hypothesis_dict,
    ) = build_all_hypothesis_path_dict(
        spec_dict,
        source_path_by_id_dict,
        global_idx,
    )
    (
        total_value_df,
        return_df,
        benchmark_return_ser,
        global_index_sha256_str,
    ) = build_global_path_frames(
        spec_dict,
        hypothesis_path_by_id_dict,
        global_idx,
    )
    norgate_vintage_final_dict = norgate_database_vintage_dict()
    write_json(
        output_dir_path / "norgate_database_vintage_final.json",
        norgate_vintage_final_dict,
    )
    if norgate_vintage_final_dict != norgate_vintage_start_dict:
        raise RuntimeError(
            "Norgate database vintage changed before all source and benchmark "
            "reads completed. Discard this output and rerun in a fresh directory."
        )
    capital_base_float = float(spec_dict["portfolio_contract"]["capital_base_float"])
    benchmark_total_value_ser = (
        capital_base_float * (1.0 + benchmark_return_ser).cumprod()
    ).rename("SPXTR")
    global_path_df = total_value_df.copy()
    global_path_df["SPXTR"] = benchmark_total_value_ser
    global_return_df = return_df.copy()
    global_return_df["SPXTR"] = benchmark_return_ser
    write_csv_gzip(
        global_path_df,
        output_dir_path / "global_daily_paths.csv.gz",
        index_bool=True,
        index_label_str="date",
    )
    write_csv_gzip(
        global_return_df,
        output_dir_path / "global_daily_returns.csv.gz",
        index_bool=True,
        index_label_str="date",
    )

    headline_metric_df = calculate_headline_metric_df(
        total_value_df,
        benchmark_total_value_ser,
        benchmark_return_ser,
        spec_dict,
    )
    headline_metric_df.to_csv(
        output_dir_path / "headline_metrics.csv",
        index=False,
        float_format="%.12g",
        lineterminator="\n",
    )
    baseline_context_df = calculate_baseline_context_comparison_df(
        headline_metric_df,
        spec_dict,
    )
    baseline_context_df.to_csv(
        output_dir_path / "baseline_context_comparison.csv",
        index=False,
        float_format="%.12g",
        lineterminator="\n",
    )
    subperiod_metric_df = calculate_subperiod_metric_df(
        total_value_df,
        benchmark_return_ser,
        spec_dict,
    )
    subperiod_metric_df.to_csv(
        output_dir_path / "subperiod_metrics.csv",
        index=False,
        float_format="%.12g",
        lineterminator="\n",
    )
    crisis_metric_df = calculate_crisis_metric_df(total_value_df, spec_dict)
    crisis_metric_df.to_csv(
        output_dir_path / "crisis_metrics.csv",
        index=False,
        float_format="%.12g",
        lineterminator="\n",
    )
    (
        bootstrap_summary_df,
        bootstrap_delta_df,
        holm_df,
    ) = calculate_bootstrap_evidence(return_df, headline_metric_df, spec_dict)
    bootstrap_summary_df.to_csv(
        output_dir_path / "bootstrap_summary.csv",
        index=False,
        float_format="%.12g",
        lineterminator="\n",
    )
    write_csv_gzip(
        bootstrap_delta_df,
        output_dir_path / "bootstrap_candidate_deltas.csv.gz",
        index_bool=False,
    )
    holm_df.to_csv(
        output_dir_path / "holm_results.csv",
        index=False,
        float_format="%.12g",
        lineterminator="\n",
    )
    selected_active_sharpe_df = calculate_selected_active_sharpe_df(
        return_df,
        spec_dict,
    )
    selected_active_sharpe_df.to_csv(
        output_dir_path / "selected_active_sharpe_diagnostic.csv",
        index=False,
        float_format="%.12g",
        lineterminator="\n",
    )
    shared_tail_event_df, tail_contribution_df = calculate_shared_tail_event_frames(
        return_df,
        benchmark_return_ser,
        total_value_df,
        sleeve_equity_by_hypothesis_dict,
        quantile_float=float(spec_dict["statistical_contract"]["es_quantile_float"]),
        capital_base_float=capital_base_float,
    )
    write_csv_gzip(
        shared_tail_event_df,
        output_dir_path / "shared_tail_events.csv.gz",
        index_bool=False,
    )
    write_csv_gzip(
        tail_contribution_df,
        output_dir_path / "tail_pod_contributions.csv.gz",
        index_bool=False,
    )
    rolling_correlation_df = calculate_rolling_market_correlation_df(
        return_df,
        benchmark_return_ser,
    )
    write_csv_gzip(
        rolling_correlation_df,
        output_dir_path / "rolling_126d_market_correlation.csv.gz",
        index_bool=True,
        index_label_str="date",
    )
    book_transaction_detail_df, book_transaction_summary_df = (
        aggregate_book_transaction_evidence(
            spec_dict,
            output_dir_path,
            global_idx,
        )
    )
    write_csv_gzip(
        book_transaction_detail_df,
        output_dir_path / "book_same_day_symbol_orders.csv.gz",
        index_bool=False,
    )
    book_transaction_summary_df.to_csv(
        output_dir_path / "book_order_overlap_summary.csv",
        index=False,
        float_format="%.12g",
        lineterminator="\n",
    )
    source_metadata_by_id_dict = load_source_metadata_by_id_dict(
        spec_dict,
        output_dir_path,
    )
    candidate_capacity_evidence_df = load_candidate_capacity_evidence_df(
        spec_dict,
        global_idx,
        global_index_sha256_str=global_index_sha256_str,
        spec_sha256_str=sha256_file_str(spec_path),
        source_metadata_by_id_dict=source_metadata_by_id_dict,
        norgate_database_vintage_dict=norgate_vintage_final_dict,
    )
    candidate_capacity_evidence_df.to_csv(
        output_dir_path / "candidate_capacity_evidence.csv",
        index=False,
        float_format="%.12g",
        lineterminator="\n",
    )
    candidate_accounting_cost_gate_df = build_candidate_accounting_cost_gate_df(
        spec_dict,
        source_metadata_by_id_dict,
    )
    candidate_accounting_cost_gate_df.to_csv(
        output_dir_path / "candidate_accounting_cost_gates.csv",
        index=False,
        float_format="%.12g",
        lineterminator="\n",
    )
    promotion_gate_df = evaluate_promotion_gate_df(
        headline_metric_df,
        subperiod_metric_df,
        bootstrap_summary_df,
        holm_df,
        candidate_capacity_evidence_df,
        candidate_accounting_cost_gate_df,
        bool(source_lineage_dict["source_lineage_gate_bool"]),
        spec_dict,
    )
    promotion_gate_df.to_csv(
        output_dir_path / "promotion_gates.csv",
        index=False,
        float_format="%.12g",
        lineterminator="\n",
    )
    decision_log_path = output_dir_path / "decision_log.jsonl"
    for _, gate_row_ser in promotion_gate_df.iterrows():
        append_jsonl(
            decision_log_path,
            {
                "recorded_at_utc_str": utc_now_str(),
                "study_id_str": spec_dict["study_id_str"],
                **gate_row_ser.to_dict(),
            },
        )
    any_pass_bool = bool(promotion_gate_df["promotion_gate_bool"].any())
    any_non_capacity_pass_bool = bool(
        promotion_gate_df["non_capacity_promotion_gate_bool"].any()
    )
    append_jsonl(
        decision_log_path,
        {
            "recorded_at_utc_str": utc_now_str(),
            "study_id_str": spec_dict["study_id_str"],
            "study_verdict_str": (
                "forward_shadow_only_for_passing_candidates"
                if any_pass_bool
                else (
                    "exact_capacity_rerun_required_before_forward_shadow"
                    if any_non_capacity_pass_bool
                    else "retain_H0_stop_no_tuning"
                )
            ),
            "passing_candidate_id_list": promotion_gate_df.loc[
                promotion_gate_df["promotion_gate_bool"],
                "candidate_id_str",
            ].tolist(),
        },
    )
    create_charts(
        total_value_df,
        rolling_correlation_df,
        promotion_gate_df,
        output_dir_path,
    )
    report_path = write_hebrew_report(
        output_dir_path,
        spec_dict,
        total_value_df,
        headline_metric_df,
        promotion_gate_df,
        bootstrap_summary_df,
        holm_df,
        selected_active_sharpe_df,
        candidate_capacity_evidence_df,
        baseline_context_df,
    )
    append_jsonl(
        output_dir_path / "experiment_ledger.jsonl",
        {
            "event_str": "study_completed",
            "recorded_at_utc_str": utc_now_str(),
            "report_path_str": str(report_path),
            "passing_candidate_id_list": promotion_gate_df.loc[
                promotion_gate_df["promotion_gate_bool"],
                "candidate_id_str",
            ].tolist(),
        },
    )
    write_run_manifest(
        output_dir_path,
        frozen_spec_copy_path,
        global_index_sha256_str,
        total_value_df,
        source_run_summary_df,
        spec_dict,
    )
    return report_path


def parse_args(arg_list: Iterable[str] | None = None) -> argparse.Namespace:
    parser_obj = argparse.ArgumentParser(
        description="Run the frozen Ladder4 H0-H8 stabilizer value-add study."
    )
    parser_obj.add_argument(
        "--spec",
        type=Path,
        default=DEFAULT_SPEC_PATH,
        help="Frozen YAML study specification.",
    )
    parser_obj.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR_PATH,
        help="New, empty output directory.",
    )
    parser_obj.add_argument(
        "--resume",
        action="store_true",
        help="Forbidden safety flag; supplying it exits before any artifact write.",
    )
    return parser_obj.parse_args(list(arg_list) if arg_list is not None else None)


def main(arg_list: Iterable[str] | None = None) -> int:
    args_obj = parse_args(arg_list)
    report_path = run_study(
        args_obj.spec,
        args_obj.output_dir,
        resume_bool=bool(args_obj.resume),
    )
    print(f"study completed: {report_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
