"""Fresh-run multi-pod portfolio manager.

PortfolioManager V1 runs configured strategy pods from scratch with explicit
capital allocations, then hands the completed pod strategies to Portfolio for
read-only aggregation and reporting.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
import importlib
import inspect
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from alpha.engine.portfolio import Portfolio
from alpha.engine.report import (
    build_research_output_path,
    save_portfolio_results,
    save_results as save_strategy_results,
)
from alpha.engine.strategy import Strategy


SUPPORTED_STRATEGY_IMPORT_TUPLE: tuple[str, ...] = (
    "strategies.dv2.strategy_mr_dv2:DVO2Strategy",
    "strategies.qpi.strategy_mr_qpi_ibs_rsi_exit:QPIIbsRsiExitStrategy",
    "strategies.taa_df.strategy_taa_df_btal_fallback_tqqq_vix_cash",
    "strategies.taa_df.strategy_taa_df_btal_1n_fallback_tqqq_vix_cash",
    "strategies.taa_df.strategy_taa_df_btal_linearity_1n_fallback_qqq_vix_cash",
    "strategies.momentum.strategy_mo_atr_normalized_ndx:AtrNormalizedNdxStrategy",
    "strategies.momentum.strategy_mo_atr_normalized_ndx_vxn_scaled:VxnScaledAtrNormalizedNdxStrategy",
)
POD_MINIMUM_ALLOCATED_CAPITAL_FLOAT_DICT: dict[str, float] = {
    "strategies.dv2.strategy_mr_dv2:DVO2Strategy": 25_000.0,
    "strategies.qpi.strategy_mr_qpi_ibs_rsi_exit:QPIIbsRsiExitStrategy": 25_000.0,
}
SUPPORTED_ALLOCATION_POLICY_TUPLE: tuple[str, ...] = ("fixed", "equal")
TOP_LEVEL_FIELD_SET: frozenset[str] = frozenset(
    {
        "name_str",
        "capital_base_float",
        "backtest_start_date_str",
        "end_date_str",
        "allocation_policy_str",
        "max_workers_int",
        "rebalance",
        "save_pod_artifacts_bool",
        "pods",
    }
)
POD_FIELD_SET: frozenset[str] = frozenset(
    {
        "pod_id_str",
        "strategy_import_str",
        "weight_float",
    }
)


@dataclass(frozen=True)
class PortfolioPodConfig:
    pod_id_str: str
    strategy_import_str: str
    weight_float: float
    allocated_capital_float: float


@dataclass(frozen=True)
class PortfolioManagerConfig:
    name_str: str
    capital_base_float: float
    backtest_start_date_str: str
    end_date_str: str | None
    allocation_policy_str: str
    max_workers_int: int | None
    rebalance: None
    save_pod_artifacts_bool: bool
    pod_config_list: list[PortfolioPodConfig]

    @property
    def weight_list(self) -> list[float]:
        return [pod_config.weight_float for pod_config in self.pod_config_list]


@dataclass(frozen=True)
class PortfolioPodRunResult:
    pod_config: PortfolioPodConfig
    strategy: Strategy
    pod_artifact_dir_path: Path | None


@dataclass(frozen=True)
class PortfolioManagerRunResult:
    portfolio: Portfolio
    config: PortfolioManagerConfig
    pod_run_result_list: list[PortfolioPodRunResult]
    manager_run_dir_path: Path | None
    portfolio_output_dir_path: Path | None
    manager_metadata_path: Path | None


def _json_default(value_obj: Any) -> Any:
    if isinstance(value_obj, Path):
        return str(value_obj)
    if isinstance(value_obj, pd.Timestamp):
        return value_obj.isoformat()
    return str(value_obj)


def _module_import_str(strategy_import_str: str) -> str:
    return strategy_import_str.split(":", 1)[0]


def _validate_no_unknown_fields(
    payload_dict: dict[str, Any],
    allowed_field_set: frozenset[str],
    context_str: str,
) -> None:
    unknown_field_list = sorted(set(payload_dict) - set(allowed_field_set))
    if len(unknown_field_list) > 0:
        raise ValueError(
            f"{context_str} contains unsupported field(s): {', '.join(unknown_field_list)}."
        )


def _coerce_required_str(payload_dict: dict[str, Any], field_name_str: str) -> str:
    value_obj = payload_dict.get(field_name_str)
    if not isinstance(value_obj, str) or len(value_obj.strip()) == 0:
        raise ValueError(f"Field '{field_name_str}' must be a non-empty string.")
    return value_obj.strip()


def _coerce_optional_str(payload_dict: dict[str, Any], field_name_str: str) -> str | None:
    value_obj = payload_dict.get(field_name_str)
    if value_obj is None:
        return None
    if not isinstance(value_obj, str) or len(value_obj.strip()) == 0:
        raise ValueError(f"Field '{field_name_str}' must be null or a non-empty string.")
    return value_obj.strip()


def _coerce_positive_float(payload_dict: dict[str, Any], field_name_str: str) -> float:
    if field_name_str not in payload_dict:
        raise ValueError(f"Field '{field_name_str}' is required.")
    value_float = float(payload_dict[field_name_str])
    if value_float <= 0.0:
        raise ValueError(f"Field '{field_name_str}' must be positive.")
    return value_float


def _coerce_optional_positive_int(payload_dict: dict[str, Any], field_name_str: str) -> int | None:
    value_obj = payload_dict.get(field_name_str)
    if value_obj is None:
        return None
    value_int = int(value_obj)
    if value_int <= 0:
        raise ValueError(f"Field '{field_name_str}' must be null or a positive integer.")
    return value_int


def _validate_backtest_dates(
    backtest_start_date_str: str,
    end_date_str: str | None,
) -> None:
    start_ts = pd.Timestamp(backtest_start_date_str)
    if end_date_str is None:
        return
    end_ts = pd.Timestamp(end_date_str)
    if end_ts < start_ts:
        raise ValueError("Field 'end_date_str' must be greater than or equal to backtest_start_date_str.")


def _build_pod_config_list(
    raw_pod_list: list[dict[str, Any]],
    allocation_policy_str: str,
    capital_base_float: float,
) -> list[PortfolioPodConfig]:
    if len(raw_pod_list) == 0:
        raise ValueError("Field 'pods' must contain at least one pod.")

    pod_id_set: set[str] = set()
    fixed_weight_sum_float = 0.0
    pod_config_list: list[PortfolioPodConfig] = []

    for pod_idx_int, pod_payload_dict in enumerate(raw_pod_list, start=1):
        if not isinstance(pod_payload_dict, dict):
            raise ValueError(f"Pod #{pod_idx_int} must be a mapping.")
        if "params" in pod_payload_dict:
            raise ValueError("Per-pod params are not supported in PortfolioManager V1.")
        _validate_no_unknown_fields(
            pod_payload_dict,
            POD_FIELD_SET,
            context_str=f"Pod #{pod_idx_int}",
        )

        pod_id_str = _coerce_required_str(pod_payload_dict, "pod_id_str")
        if pod_id_str in pod_id_set:
            raise ValueError(f"Duplicate pod_id_str '{pod_id_str}'.")
        pod_id_set.add(pod_id_str)

        strategy_import_str = _coerce_required_str(pod_payload_dict, "strategy_import_str")
        if strategy_import_str not in SUPPORTED_STRATEGY_IMPORT_TUPLE:
            raise ValueError(
                "Unsupported strategy_import_str "
                f"'{strategy_import_str}'. Expected one of {SUPPORTED_STRATEGY_IMPORT_TUPLE}."
            )

        if allocation_policy_str == "fixed":
            if "weight_float" not in pod_payload_dict:
                raise ValueError(f"Pod '{pod_id_str}' is missing required weight_float.")
            weight_float = float(pod_payload_dict["weight_float"])
            if weight_float <= 0.0:
                raise ValueError(f"Pod '{pod_id_str}' weight_float must be positive.")
            fixed_weight_sum_float += weight_float
        else:
            if "weight_float" in pod_payload_dict:
                raise ValueError("Equal allocation policy does not allow pod weight_float.")
            weight_float = 1.0 / float(len(raw_pod_list))

        allocated_capital_float = capital_base_float * weight_float
        pod_config_list.append(
            PortfolioPodConfig(
                pod_id_str=pod_id_str,
                strategy_import_str=strategy_import_str,
                weight_float=weight_float,
                allocated_capital_float=allocated_capital_float,
            )
        )

    if allocation_policy_str == "fixed" and abs(fixed_weight_sum_float - 1.0) > 1e-6:
        raise ValueError(f"Fixed pod weights must sum to 1.0, got {fixed_weight_sum_float:.6f}.")

    return pod_config_list


def _validate_allocated_capital_floor(pod_config_list: list[PortfolioPodConfig]) -> None:
    underfunded_pod_message_list: list[str] = []

    for pod_config in pod_config_list:
        minimum_allocated_capital_float = POD_MINIMUM_ALLOCATED_CAPITAL_FLOAT_DICT.get(
            pod_config.strategy_import_str
        )
        if minimum_allocated_capital_float is None:
            continue
        if pod_config.allocated_capital_float >= minimum_allocated_capital_float:
            continue

        underfunded_pod_message_list.append(
            f"{pod_config.pod_id_str} ({pod_config.strategy_import_str}) "
            f"allocated_capital_float={pod_config.allocated_capital_float:.2f} "
            f"< minimum_allocated_capital_float={minimum_allocated_capital_float:.2f}"
        )

    if len(underfunded_pod_message_list) == 0:
        return

    raise ValueError(
        "PortfolioManager allocated capital is below the practical minimum for "
        "live-supported stock pod(s): "
        + "; ".join(underfunded_pod_message_list)
        + ". Increase capital_base_float, increase the pod weight, or remove the "
        "underfunded stock pod."
    )


def build_portfolio_manager_config(config_dict: dict[str, Any]) -> PortfolioManagerConfig:
    if not isinstance(config_dict, dict):
        raise ValueError("PortfolioManager config must be a mapping.")
    _validate_no_unknown_fields(config_dict, TOP_LEVEL_FIELD_SET, context_str="PortfolioManager config")

    rebalance_obj = config_dict.get("rebalance")
    if rebalance_obj is not None:
        raise ValueError("PortfolioManager V1 rejects non-null rebalance.")

    name_str = _coerce_required_str(config_dict, "name_str")
    capital_base_float = _coerce_positive_float(config_dict, "capital_base_float")
    backtest_start_date_str = _coerce_required_str(config_dict, "backtest_start_date_str")
    end_date_str = _coerce_optional_str(config_dict, "end_date_str")
    _validate_backtest_dates(backtest_start_date_str, end_date_str)

    allocation_policy_str = str(config_dict.get("allocation_policy_str", "fixed")).strip().lower()
    if allocation_policy_str not in SUPPORTED_ALLOCATION_POLICY_TUPLE:
        raise ValueError(
            "allocation_policy_str must be one of "
            f"{SUPPORTED_ALLOCATION_POLICY_TUPLE}, got '{allocation_policy_str}'."
        )

    max_workers_int = _coerce_optional_positive_int(config_dict, "max_workers_int")
    save_pod_artifacts_obj = config_dict.get("save_pod_artifacts_bool", True)
    if not isinstance(save_pod_artifacts_obj, bool):
        raise ValueError("Field 'save_pod_artifacts_bool' must be a boolean.")

    raw_pod_list = config_dict.get("pods")
    if not isinstance(raw_pod_list, list):
        raise ValueError("Field 'pods' must be a list.")
    pod_config_list = _build_pod_config_list(
        raw_pod_list=raw_pod_list,
        allocation_policy_str=allocation_policy_str,
        capital_base_float=capital_base_float,
    )
    _validate_allocated_capital_floor(pod_config_list)

    return PortfolioManagerConfig(
        name_str=name_str,
        capital_base_float=capital_base_float,
        backtest_start_date_str=backtest_start_date_str,
        end_date_str=end_date_str,
        allocation_policy_str=allocation_policy_str,
        max_workers_int=max_workers_int,
        rebalance=None,
        save_pod_artifacts_bool=save_pod_artifacts_obj,
        pod_config_list=pod_config_list,
    )


def load_portfolio_manager_config(config_path: Path) -> PortfolioManagerConfig:
    with config_path.open(encoding="utf-8") as file_obj:
        config_dict = yaml.safe_load(file_obj)
    return build_portfolio_manager_config(config_dict)


def _resolve_worker_count_int(config_obj: PortfolioManagerConfig) -> int:
    if config_obj.max_workers_int is not None:
        return min(config_obj.max_workers_int, len(config_obj.pod_config_list))
    return min(len(config_obj.pod_config_list), os.cpu_count() or 1)


def _coerce_run_worker_count_int(
    config_obj: PortfolioManagerConfig,
    max_workers_int: int | None,
) -> int:
    if max_workers_int is None:
        return _resolve_worker_count_int(config_obj)
    if int(max_workers_int) <= 0:
        raise ValueError("max_workers_int override must be a positive integer.")
    return min(int(max_workers_int), len(config_obj.pod_config_list))


def _validate_run_variant_signature(run_variant_fn, strategy_import_str: str) -> None:
    signature_obj = inspect.signature(run_variant_fn)
    parameter_set = set(signature_obj.parameters)
    required_parameter_tuple = (
        "show_display_bool",
        "save_results_bool",
        "output_dir_str",
        "backtest_start_date_str",
        "capital_base_float",
        "end_date_str",
    )
    missing_parameter_list = [
        parameter_str
        for parameter_str in required_parameter_tuple
        if parameter_str not in parameter_set
    ]
    if len(missing_parameter_list) > 0:
        raise TypeError(
            f"Strategy '{strategy_import_str}' run_variant is missing common parameter(s): "
            f"{', '.join(missing_parameter_list)}."
        )


def _pod_failure_message_str(pod_config: PortfolioPodConfig) -> str:
    return (
        "PortfolioManager pod failed: "
        f"pod_id_str='{pod_config.pod_id_str}', "
        f"strategy_import_str='{pod_config.strategy_import_str}'."
    )


def _run_pod_worker(worker_payload_dict: dict[str, Any]) -> dict[str, Any]:
    pod_config = worker_payload_dict["pod_config"]
    try:
        return _run_pod_worker_unchecked(worker_payload_dict)
    except Exception as exc:
        raise RuntimeError(_pod_failure_message_str(pod_config)) from exc


def _run_pod_worker_unchecked(worker_payload_dict: dict[str, Any]) -> dict[str, Any]:
    pod_config = worker_payload_dict["pod_config"]
    module_import_str = _module_import_str(pod_config.strategy_import_str)
    strategy_module = importlib.import_module(module_import_str)
    run_variant_fn = getattr(strategy_module, "run_variant", None)
    if run_variant_fn is None:
        raise AttributeError(f"Module '{module_import_str}' does not expose run_variant(...).")
    _validate_run_variant_signature(run_variant_fn, pod_config.strategy_import_str)

    # *** CRITICAL*** Every fresh-run pod receives the same executable
    # portfolio date range. Individual strategy loaders may still keep earlier
    # history for causal indicators, but fills must start from this shared date.
    strategy_obj = run_variant_fn(
        show_display_bool=bool(worker_payload_dict["show_display_bool"]),
        save_results_bool=False,
        output_dir_str=str(worker_payload_dict["output_dir_str"]),
        backtest_start_date_str=str(worker_payload_dict["backtest_start_date_str"]),
        capital_base_float=float(pod_config.allocated_capital_float),
        end_date_str=worker_payload_dict["end_date_str"],
    )
    if not isinstance(strategy_obj, Strategy):
        raise TypeError(
            f"Strategy '{pod_config.strategy_import_str}' did not return a Strategy instance."
        )

    pod_artifact_dir_path = None
    if bool(worker_payload_dict["save_pod_artifacts_bool"]):
        pod_output_dir_path = Path(worker_payload_dict["pod_output_root_path_str"]) / pod_config.pod_id_str
        pod_artifact_dir_path = save_strategy_results(
            strategy_obj,
            output_path=pod_output_dir_path,
        )

    return {
        "position_int": int(worker_payload_dict["position_int"]),
        "pod_config": pod_config,
        "strategy": strategy_obj,
        "pod_artifact_dir_path": pod_artifact_dir_path,
    }


class PortfolioManager:
    def __init__(
        self,
        config: PortfolioManagerConfig,
        source_config_path_str: str | None = None,
    ):
        self.config = config
        self.source_config_path_str = source_config_path_str

    @classmethod
    def from_yaml(cls, config_path: Path) -> "PortfolioManager":
        config_path = config_path.resolve()
        config_obj = load_portfolio_manager_config(config_path)
        return cls(config=config_obj, source_config_path_str=str(config_path))

    def run(
        self,
        output_dir_str: str = "results",
        save_results_bool: bool = True,
        show_display_bool: bool = False,
        max_workers_int: int | None = None,
    ) -> PortfolioManagerRunResult:
        worker_count_int = _coerce_run_worker_count_int(self.config, max_workers_int)
        manager_run_dir_path = None
        pod_output_root_path = None
        if save_results_bool:
            manager_run_dir_path = build_research_output_path(
                output_dir_str,
                "portfolio",
                self.config.name_str,
                "vanilla_backtest",
            )
            pod_output_root_path = manager_run_dir_path / "pods"
            pod_output_root_path.mkdir(parents=True, exist_ok=True)

        worker_payload_list = []
        for position_int, pod_config in enumerate(self.config.pod_config_list):
            worker_payload_list.append(
                {
                    "position_int": position_int,
                    "pod_config": pod_config,
                    "output_dir_str": output_dir_str,
                    "backtest_start_date_str": self.config.backtest_start_date_str,
                    "end_date_str": self.config.end_date_str,
                    "show_display_bool": show_display_bool,
                    "save_pod_artifacts_bool": bool(
                        save_results_bool and self.config.save_pod_artifacts_bool
                    ),
                    "pod_output_root_path_str": (
                        str(pod_output_root_path) if pod_output_root_path is not None else ""
                    ),
                }
            )

        if worker_count_int == 1:
            raw_result_list = [_run_pod_worker(payload_dict) for payload_dict in worker_payload_list]
        else:
            raw_result_by_position_dict: dict[int, dict[str, Any]] = {}
            with ProcessPoolExecutor(max_workers=worker_count_int) as executor_obj:
                future_map = {
                    executor_obj.submit(_run_pod_worker, payload_dict): int(payload_dict["position_int"])
                    for payload_dict in worker_payload_list
                }
                for future_obj in as_completed(future_map):
                    position_int = future_map[future_obj]
                    try:
                        raw_result_by_position_dict[position_int] = future_obj.result()
                    except Exception as exc:
                        pod_config = self.config.pod_config_list[position_int]
                        if isinstance(exc, RuntimeError) and str(exc) == _pod_failure_message_str(pod_config):
                            raise
                        raise RuntimeError(
                            _pod_failure_message_str(pod_config)
                        ) from exc
            raw_result_list = [
                raw_result_by_position_dict[position_int]
                for position_int in range(len(worker_payload_list))
            ]

        pod_run_result_list = [
            PortfolioPodRunResult(
                pod_config=raw_result_dict["pod_config"],
                strategy=raw_result_dict["strategy"],
                pod_artifact_dir_path=raw_result_dict["pod_artifact_dir_path"],
            )
            for raw_result_dict in raw_result_list
        ]

        pod_info_list = []
        for pod_run_result in pod_run_result_list:
            pod_config = pod_run_result.pod_config
            pod_info_list.append(
                {
                    "pod_id_str": pod_config.pod_id_str,
                    "strategy_name": pod_run_result.strategy.name,
                    "strategy_import_str": pod_config.strategy_import_str,
                    "source_type_str": "fresh_run",
                    "pod_artifact_dir": (
                        str(pod_run_result.pod_artifact_dir_path)
                        if pod_run_result.pod_artifact_dir_path is not None
                        else None
                    ),
                    "backtest_start_date_str": self.config.backtest_start_date_str,
                    "end_date_str": self.config.end_date_str,
                }
            )

        portfolio = Portfolio(
            strategies=[pod_run_result.strategy for pod_run_result in pod_run_result_list],
            weights=self.config.weight_list,
            name=self.config.name_str,
            capital_base=self.config.capital_base_float,
            rebalance=None,
            pod_info_list=pod_info_list,
        )
        portfolio.source_config_path = self.source_config_path_str

        portfolio_output_dir_path = None
        manager_metadata_path = None
        if save_results_bool:
            assert manager_run_dir_path is not None
            portfolio_output_dir_path = save_portfolio_results(
                portfolio,
                output_path=manager_run_dir_path,
            )
            manager_metadata_path = manager_run_dir_path / "manager_metadata.json"
            self._write_manager_metadata(
                manager_metadata_path=manager_metadata_path,
                worker_count_int=worker_count_int,
                pod_run_result_list=pod_run_result_list,
                portfolio_output_dir_path=portfolio_output_dir_path,
            )

        return PortfolioManagerRunResult(
            portfolio=portfolio,
            config=self.config,
            pod_run_result_list=pod_run_result_list,
            manager_run_dir_path=manager_run_dir_path,
            portfolio_output_dir_path=portfolio_output_dir_path,
            manager_metadata_path=manager_metadata_path,
        )

    def _write_manager_metadata(
        self,
        manager_metadata_path: Path,
        worker_count_int: int,
        pod_run_result_list: list[PortfolioPodRunResult],
        portfolio_output_dir_path: Path,
    ) -> None:
        metadata_dict = {
            "artifact_type": "portfolio_manager",
            "saved_at": datetime.now().isoformat(timespec="seconds"),
            "portfolio_name_str": self.config.name_str,
            "source_config_path_str": self.source_config_path_str,
            "capital_base_float": float(self.config.capital_base_float),
            "backtest_start_date_str": self.config.backtest_start_date_str,
            "end_date_str": self.config.end_date_str,
            "allocation_policy_str": self.config.allocation_policy_str,
            "max_workers_int": worker_count_int,
            "rebalance": None,
            "validation_status_str": "passed",
            "portfolio_output_dir_path": str(portfolio_output_dir_path),
            "pods": [
                {
                    "pod_id_str": pod_run_result.pod_config.pod_id_str,
                    "strategy_name": pod_run_result.strategy.name,
                    "strategy_import_str": pod_run_result.pod_config.strategy_import_str,
                    "weight_float": float(pod_run_result.pod_config.weight_float),
                    "allocated_capital_float": float(
                        pod_run_result.pod_config.allocated_capital_float
                    ),
                    "pod_artifact_dir_path": (
                        str(pod_run_result.pod_artifact_dir_path)
                        if pod_run_result.pod_artifact_dir_path is not None
                        else None
                    ),
                }
                for pod_run_result in pod_run_result_list
            ],
        }
        manager_metadata_path.write_text(
            json.dumps(metadata_dict, indent=2, sort_keys=True, default=_json_default),
            encoding="utf-8",
        )
