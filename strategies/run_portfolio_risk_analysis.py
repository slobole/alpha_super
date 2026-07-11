"""
Run RiskAnalysis on one explicit, trusted portfolio pickle artifact.

Usage:
    uv run python strategies/run_portfolio_risk_analysis.py \
        results/research/portfolio/multipod_monthly/vanilla_backtest/2026-07-10_004316/multipod_monthly.pkl

The runner never searches for a "latest" artifact and never reruns or rebuilds
the portfolio. Python pickle files can execute code while loading; only pass a
trusted local artifact produced by this repository.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


REPO_ROOT_PATH = Path(__file__).resolve().parents[1]
if str(REPO_ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_PATH))

from alpha.engine.portfolio import Portfolio
from alpha.engine.risk_analysis import (
    DEFAULT_CONFIDENCE_LEVEL_FLOAT,
    DEFAULT_PRIMARY_MEAN_BLOCK_LENGTH_INT,
    DEFAULT_RANDOM_SEED_INT,
    DEFAULT_SENSITIVITY_BLOCK_LENGTH_TUPLE,
    DEFAULT_SIMULATION_COUNT_INT,
    RiskAnalysis,
)


def _read_json_dict(json_path: Path, *, required_bool: bool) -> dict[str, object]:
    if not json_path.exists():
        if required_bool:
            raise FileNotFoundError(f"Required source metadata is missing: {json_path}")
        return {}
    loaded_obj = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(loaded_obj, dict):
        raise ValueError(f"Expected a JSON object in {json_path}")
    return loaded_obj


def _sha256_str(file_path: Path) -> str:
    hash_obj = hashlib.sha256()
    with file_path.open("rb") as file_obj:
        while chunk_bytes := file_obj.read(1024 * 1024):
            hash_obj.update(chunk_bytes)
    return hash_obj.hexdigest()


def _validate_portfolio_artifact(
    portfolio_obj: Portfolio,
    portfolio_pickle_path: Path,
    source_metadata_dict: dict[str, object],
) -> None:
    if not isinstance(portfolio_obj, Portfolio):
        raise TypeError(
            f"Expected Portfolio pickle, got {type(portfolio_obj).__name__}."
        )
    result_df = getattr(portfolio_obj, "results", None)
    if not isinstance(result_df, pd.DataFrame) or len(result_df) == 0:
        raise ValueError("Portfolio results are empty.")
    if "daily_returns" not in result_df.columns and "total_value" not in result_df.columns:
        raise ValueError("Portfolio results must contain daily_returns or total_value.")
    if not isinstance(result_df.index, pd.DatetimeIndex):
        raise ValueError("Portfolio results must use a DatetimeIndex.")
    if not result_df.index.is_monotonic_increasing:
        raise ValueError("Portfolio result dates must be monotonic increasing.")
    if not result_df.index.is_unique:
        raise ValueError("Portfolio result dates must be unique.")
    if "total_value" in result_df.columns:
        total_value_ser = result_df["total_value"].astype(float)
        if not bool(np.isfinite(total_value_ser.to_numpy(dtype=float)).all()):
            raise ValueError("Portfolio total_value contains non-finite values.")
        if bool((total_value_ser <= 0.0).any()):
            raise ValueError("Portfolio total_value must stay positive.")
    if "daily_returns" in result_df.columns:
        daily_return_ser = result_df["daily_returns"].astype(float)
        if not bool(np.isfinite(daily_return_ser.to_numpy(dtype=float)).all()):
            raise ValueError("Portfolio daily_returns contains non-finite values.")
        if bool((daily_return_ser <= -1.0).any()):
            raise ValueError("Portfolio daily_returns contains a return <= -100%.")
        if not np.isclose(float(daily_return_ser.iloc[0]), 0.0, rtol=0.0, atol=1e-12):
            raise ValueError(
                "Portfolio first daily_returns row must be the zero initial-state placeholder."
            )
    if "daily_returns" in result_df.columns and "total_value" in result_df.columns:
        reconstructed_return_ser = result_df["total_value"].astype(float).pct_change(
            fill_method=None
        )
        if not np.allclose(
            result_df["daily_returns"].astype(float).iloc[1:].to_numpy(dtype=float),
            reconstructed_return_ser.iloc[1:].to_numpy(dtype=float),
            rtol=1e-10,
            atol=1e-12,
        ):
            raise ValueError("Portfolio daily_returns disagrees with total_value.pct_change().")
    if source_metadata_dict.get("artifact_type") != "portfolio":
        raise ValueError("Sibling metadata.json is not a portfolio artifact.")
    if str(source_metadata_dict.get("portfolio_name")) != str(portfolio_obj.name):
        raise ValueError("Portfolio name does not match sibling metadata.json.")

    metadata_pickle_path_obj = source_metadata_dict.get("pickle_path")
    if metadata_pickle_path_obj:
        metadata_pickle_path = Path(str(metadata_pickle_path_obj))
        if not metadata_pickle_path.is_absolute():
            metadata_pickle_path = portfolio_pickle_path.parent / metadata_pickle_path
        metadata_pickle_path = metadata_pickle_path.resolve()
        if metadata_pickle_path != portfolio_pickle_path.resolve():
            raise ValueError("Explicit pickle path does not match metadata.json pickle_path.")

    metadata_pod_list = source_metadata_dict.get("pods")
    object_pod_list = getattr(portfolio_obj, "pod_info_list", None)
    object_weight_list = getattr(portfolio_obj, "weights", None)
    if not isinstance(metadata_pod_list, list) or not isinstance(object_pod_list, list):
        raise ValueError("Portfolio pod metadata is missing or invalid.")
    if not isinstance(object_weight_list, list):
        raise ValueError("Portfolio weights are missing or invalid.")
    metadata_weight_vec = np.asarray(
        [float(pod_dict["weight"]) for pod_dict in metadata_pod_list],
        dtype=float,
    )
    object_weight_vec = np.asarray(object_weight_list, dtype=float)
    if metadata_weight_vec.shape != object_weight_vec.shape or not np.allclose(
        metadata_weight_vec,
        object_weight_vec,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("Portfolio weights do not match metadata.json pods.")
    metadata_pod_identity_tuple = tuple(
        (
            str(pod_dict.get("pod_id_str")),
            str(pod_dict.get("strategy_name")),
            str(pod_dict.get("strategy_import_str")),
        )
        for pod_dict in metadata_pod_list
    )
    object_pod_identity_tuple = tuple(
        (
            str(pod_dict.get("pod_id_str")),
            str(pod_dict.get("strategy_name")),
            str(pod_dict.get("strategy_import_str")),
        )
        for pod_dict in object_pod_list
    )
    if metadata_pod_identity_tuple != object_pod_identity_tuple:
        raise ValueError("Portfolio pod identities do not match metadata.json.")

    metadata_capital_float = float(source_metadata_dict.get("capital_base"))
    object_capital_float = float(getattr(portfolio_obj, "_capital_base"))
    if not np.isclose(metadata_capital_float, object_capital_float, rtol=0.0, atol=1e-8):
        raise ValueError("Portfolio capital base does not match metadata.json.")
    if "total_value" in result_df.columns and not np.isclose(
        float(result_df["total_value"].iloc[0]),
        object_capital_float,
        rtol=1e-10,
        atol=1e-8,
    ):
        raise ValueError(
            "Portfolio first total_value row must equal the initial capital base."
        )
    if source_metadata_dict.get("rebalance") != getattr(portfolio_obj, "_rebalance", None):
        raise ValueError("Portfolio rebalance frequency does not match metadata.json.")
    if str(source_metadata_dict.get("rebalance_policy")) != str(
        getattr(portfolio_obj, "_rebalance_policy", None)
    ):
        raise ValueError("Portfolio rebalance policy does not match metadata.json.")

    realized_start_str = pd.Timestamp(result_df.index.min()).isoformat()
    realized_end_str = pd.Timestamp(result_df.index.max()).isoformat()
    metadata_start_obj = source_metadata_dict.get("common_start")
    metadata_end_obj = source_metadata_dict.get("common_end")
    if metadata_start_obj and pd.Timestamp(metadata_start_obj).isoformat() != realized_start_str:
        raise ValueError("Portfolio realized start does not match metadata.json common_start.")
    if metadata_end_obj and pd.Timestamp(metadata_end_obj).isoformat() != realized_end_str:
        raise ValueError("Portfolio realized end does not match metadata.json common_end.")


def _normalized_pod_context_list(
    source_metadata_dict: dict[str, object],
) -> list[dict[str, object]]:
    source_pod_list = source_metadata_dict.get("pods", [])
    if not isinstance(source_pod_list, list):
        return []
    pod_context_list: list[dict[str, object]] = []
    for source_pod_obj in source_pod_list:
        if not isinstance(source_pod_obj, dict):
            continue
        pod_context_list.append(
            {
                "pod_id_str": source_pod_obj.get("pod_id_str"),
                "strategy_name_str": source_pod_obj.get("strategy_name"),
                "strategy_import_str": source_pod_obj.get("strategy_import_str"),
                "weight_float": source_pod_obj.get("weight"),
                "allocated_capital_float": source_pod_obj.get("allocated_capital"),
                "source_type_str": source_pod_obj.get("source_type_str"),
            }
        )
    return pod_context_list


def _build_portfolio_analysis_context_dict(
    *,
    portfolio_pickle_path: Path,
    source_metadata_dict: dict[str, object],
    manager_metadata_dict: dict[str, object],
    portfolio_obj: Portfolio | None = None,
) -> dict[str, object]:
    source_config_path_obj = (
        source_metadata_dict.get("source_config_path")
        or manager_metadata_dict.get("source_config_path_str")
    )
    source_config_path = (
        Path(str(source_config_path_obj)).resolve()
        if source_config_path_obj
        else None
    )
    configured_end_obj = manager_metadata_dict.get("end_date_str")
    rebalance_obj = source_metadata_dict.get("rebalance")
    realized_end_weight_list: list[dict[str, object]] = []
    max_absolute_weight_drift_float: float | None = None
    if portfolio_obj is not None:
        pod_equity_df = getattr(portfolio_obj, "_pod_equities", None)
        if isinstance(pod_equity_df, pd.DataFrame) and len(pod_equity_df) > 0:
            end_pod_equity_ser = pod_equity_df.iloc[-1].astype(float)
            end_total_equity_float = float(end_pod_equity_ser.sum())
            configured_weight_vec = np.asarray(portfolio_obj.weights, dtype=float)
            realized_end_weight_vec = (
                end_pod_equity_ser.to_numpy(dtype=float) / end_total_equity_float
            )
            realized_end_weight_list = [
                {
                    "strategy_name_str": str(strategy_name_str),
                    "realized_end_weight_float": float(weight_float),
                }
                for strategy_name_str, weight_float in zip(
                    end_pod_equity_ser.index,
                    realized_end_weight_vec,
                    strict=True,
                )
            ]
            max_absolute_weight_drift_float = float(
                np.max(np.abs(realized_end_weight_vec - configured_weight_vec))
            )
    return {
        "analysis_status_str": "provisional_research_only",
        "investor_use_approved_bool": False,
        "offered_portfolio_frozen_bool": False,
        "source_artifact_path_str": str(portfolio_pickle_path.resolve()),
        "source_artifact_sha256_str": _sha256_str(portfolio_pickle_path),
        "source_artifact_saved_at_str": source_metadata_dict.get("saved_at"),
        "source_config_path_str": str(source_config_path) if source_config_path else None,
        "source_config_exists_bool": bool(source_config_path and source_config_path.exists()),
        "analysis_time_source_config_sha256_str": (
            _sha256_str(source_config_path)
            if source_config_path is not None and source_config_path.exists()
            else None
        ),
        "configured_backtest_start_date_str": manager_metadata_dict.get(
            "backtest_start_date_str"
        ),
        "configured_end_date_str": configured_end_obj,
        "realized_common_start_date_str": source_metadata_dict.get("common_start"),
        "realized_common_end_date_str": source_metadata_dict.get("common_end"),
        "capital_base_float": source_metadata_dict.get("capital_base"),
        "allocation_policy_str": (
            manager_metadata_dict.get("allocation_policy_str")
            or source_metadata_dict.get("rebalance_policy")
        ),
        "rebalance_frequency_str": str(rebalance_obj) if rebalance_obj else "none",
        "rebalance_policy_str": source_metadata_dict.get("rebalance_policy"),
        "realized_end_weight_list": realized_end_weight_list,
        "max_absolute_weight_drift_float": max_absolute_weight_drift_float,
        "source_manager_config_validation_status_str": manager_metadata_dict.get(
            "validation_status_str"
        ),
        "artifact_structure_validation_status_str": "passed_by_risk_runner",
        "point_in_time_universe_validation_status_str": "not_revalidated_by_risk_runner",
        "corporate_action_and_adjustment_validation_status_str": "not_revalidated_by_risk_runner",
        "strategy_source_revision_validation_status_str": "not_captured_by_source_artifact",
        "cost_assumption_status_str": (
            "inherited_from_source_backtest_not_revalidated"
        ),
        "pod_list": _normalized_pod_context_list(source_metadata_dict),
    }


def run_portfolio_risk_analysis(
    *,
    portfolio_pickle_path: Path,
    output_dir_str: str,
    save_results_bool: bool,
    simulation_count_int: int,
    primary_mean_block_length_int: int,
    mean_block_length_tuple: tuple[int, ...],
    confidence_level_float: float,
    random_seed_int: int,
):
    portfolio_pickle_path = portfolio_pickle_path.resolve()
    if portfolio_pickle_path.suffix.lower() != ".pkl":
        raise ValueError("portfolio_pickle_path must point to a .pkl file.")
    if not portfolio_pickle_path.is_file():
        raise FileNotFoundError(portfolio_pickle_path)

    source_dir_path = portfolio_pickle_path.parent
    source_metadata_dict = _read_json_dict(
        source_dir_path / "metadata.json",
        required_bool=True,
    )
    manager_metadata_dict = _read_json_dict(
        source_dir_path / "manager_metadata.json",
        required_bool=False,
    )
    portfolio_obj = Portfolio.read_pickle(portfolio_pickle_path)
    _validate_portfolio_artifact(
        portfolio_obj,
        portfolio_pickle_path,
        source_metadata_dict,
    )
    analysis_context_dict = _build_portfolio_analysis_context_dict(
        portfolio_pickle_path=portfolio_pickle_path,
        source_metadata_dict=source_metadata_dict,
        manager_metadata_dict=manager_metadata_dict,
        portfolio_obj=portfolio_obj,
    )

    risk_result_obj = RiskAnalysis(
        portfolio_obj,
        source_strategy_ref_str=str(portfolio_pickle_path),
        source_entity_type_str="portfolio",
        analysis_context_dict=analysis_context_dict,
        output_dir_str=output_dir_str,
        save_output_bool=save_results_bool,
        primary_mean_block_length_int=primary_mean_block_length_int,
        mean_block_length_tuple=mean_block_length_tuple,
        simulation_count_int=simulation_count_int,
        random_seed_int=random_seed_int,
        confidence_level_float=confidence_level_float,
    ).run()
    print(f"Ran portfolio RiskAnalysis: {portfolio_obj.name}")
    print(f"  Source: {portfolio_pickle_path}")
    print(f"  Realized common start: {source_metadata_dict.get('common_start')}")
    print(
        "  Configured start: "
        f"{manager_metadata_dict.get('backtest_start_date_str', 'N/A')}"
    )
    print(f"  Rebalance frequency: {analysis_context_dict['rebalance_frequency_str']}")
    if risk_result_obj.output_dir_path is not None:
        print(f"  Report folder: {risk_result_obj.output_dir_path.resolve()}")
    return risk_result_obj


def main() -> None:
    parser_obj = argparse.ArgumentParser()
    parser_obj.add_argument(
        "portfolio_pickle_path",
        type=Path,
        help="Explicit trusted portfolio .pkl artifact. No latest lookup is performed.",
    )
    parser_obj.add_argument("--output-dir", default="results")
    parser_obj.add_argument("--no-save", action="store_true")
    parser_obj.add_argument(
        "--simulation-count",
        type=int,
        default=DEFAULT_SIMULATION_COUNT_INT,
    )
    parser_obj.add_argument(
        "--primary-block-length",
        type=int,
        default=DEFAULT_PRIMARY_MEAN_BLOCK_LENGTH_INT,
    )
    parser_obj.add_argument(
        "--block-length",
        action="append",
        type=int,
        default=[],
        help="Repeat for block-length sensitivity.",
    )
    parser_obj.add_argument(
        "--confidence-level",
        type=float,
        default=DEFAULT_CONFIDENCE_LEVEL_FLOAT,
    )
    parser_obj.add_argument(
        "--random-seed",
        type=int,
        default=DEFAULT_RANDOM_SEED_INT,
    )
    arg_namespace = parser_obj.parse_args()
    mean_block_length_tuple = tuple(
        arg_namespace.block_length or DEFAULT_SENSITIVITY_BLOCK_LENGTH_TUPLE
    )
    run_portfolio_risk_analysis(
        portfolio_pickle_path=arg_namespace.portfolio_pickle_path,
        output_dir_str=arg_namespace.output_dir,
        save_results_bool=not arg_namespace.no_save,
        simulation_count_int=arg_namespace.simulation_count,
        primary_mean_block_length_int=arg_namespace.primary_block_length,
        mean_block_length_tuple=mean_block_length_tuple,
        confidence_level_float=arg_namespace.confidence_level,
        random_seed_int=arg_namespace.random_seed,
    )


if __name__ == "__main__":
    main()
