"""Frozen clean-sheet portfolio study across promoted strategy families.

The study starts from the complete WIRED/PM_READY registry, selects one or two
representatives per economic mechanism before results are reviewed, runs every
required strategy/capital pair once, and builds independently compounded pod
books on one exact date intersection. It has research authority only.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import shutil
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests
import yaml

from alpha import strategy_registry
from data.norgate_loader import TOTALRETURN_ADJUSTMENT_STR, load_price_timeseries
from scripts.research import run_ladder4_candidate_value_add_study as ladder_runner


REPO_ROOT_PATH = Path(__file__).resolve().parents[2]
DEFAULT_SPEC_PATH = (
    REPO_ROOT_PATH
    / "scripts"
    / "research"
    / "specs"
    / "clean_sheet_portfolio_foundry_v4.yaml"
)
DEFAULT_OUTPUT_DIR_PATH = (
    REPO_ROOT_PATH
    / "results"
    / "research"
    / "portfolio"
    / "clean_sheet_portfolio_foundry_study"
    / "2026-08-30_v4"
)
FROZEN_SPEC_SHA256_STR = (
    "b99434ba6d03cad2ec2c99137aa45d6b1ff538c27c379d4c5d0245bd64ac7fab"
)


def source_id_str(strategy_alias_str: str, allocated_capital_float: float) -> str:
    rounded_capital_int = int(round(float(allocated_capital_float)))
    if not math.isclose(
        float(rounded_capital_int),
        float(allocated_capital_float),
        rel_tol=0.0,
        abs_tol=1e-8,
    ):
        raise ValueError("Portfolio Foundry source capital must be a whole dollar.")
    return f"{strategy_alias_str}_{rounded_capital_int}"


def all_product_spec_dict(spec_dict: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        **dict(spec_dict["reference_portfolios"]),
        **dict(spec_dict["candidate_portfolios"]),
    }


def validate_spec_dict(spec_dict: dict[str, Any]) -> None:
    authority_field_list = [
        "allocation_authority_bool",
        "paper_authority_bool",
        "live_authority_bool",
    ]
    if any(bool(spec_dict[field_str]) for field_str in authority_field_list):
        raise ValueError("Portfolio Foundry v1 must remain research-only.")
    if bool(spec_dict["construction_doctrine"]["full_sample_optimizer_allowed_bool"]):
        raise ValueError("Full-sample optimization is forbidden.")
    if bool(spec_dict["construction_doctrine"]["markowitz_allowed_bool"]):
        raise ValueError("Markowitz is forbidden as selection authority.")
    if spec_dict["construction_doctrine"]["outer_rebalance_str"] != (
        "none_independently_compounded_pods"
    ):
        raise ValueError("The v1 aggregation contract must not outer-rebalance pods.")

    registry_contract_dict = spec_dict["registry_contract"]
    selected_by_alias_dict = registry_contract_dict["selected_strategy_by_alias"]
    excluded_strategy_list = registry_contract_dict["excluded_promoted_strategy_list"]
    selected_import_set = {
        str(strategy_spec_dict["strategy_import_str"])
        for strategy_spec_dict in selected_by_alias_dict.values()
    }
    excluded_import_list = [
        str(strategy_spec_dict["strategy_import_str"])
        for strategy_spec_dict in excluded_strategy_list
    ]
    excluded_import_set = set(excluded_import_list)
    if len(excluded_import_list) != len(excluded_import_set):
        raise ValueError("Excluded promoted strategies must be unique.")
    if selected_import_set.intersection(excluded_import_set):
        raise ValueError("A promoted strategy cannot be selected and excluded.")
    registry_import_set = set(strategy_registry.STRATEGY_TIER_DICT)
    if selected_import_set | excluded_import_set != registry_import_set:
        raise RuntimeError("Frozen inventory no longer exactly covers the promoted registry.")
    expected_promoted_count_int = int(
        registry_contract_dict["expected_promoted_count_int"]
    )
    if len(registry_import_set) != expected_promoted_count_int:
        raise RuntimeError("Promoted registry count changed after the spec was frozen.")
    wired_count_int = sum(
        tier_obj == strategy_registry.MaturityTier.WIRED
        for tier_obj in strategy_registry.STRATEGY_TIER_DICT.values()
    )
    pm_ready_count_int = sum(
        tier_obj == strategy_registry.MaturityTier.PM_READY
        for tier_obj in strategy_registry.STRATEGY_TIER_DICT.values()
    )
    if wired_count_int != int(registry_contract_dict["expected_wired_count_int"]):
        raise RuntimeError("WIRED registry count changed after the spec was frozen.")
    if pm_ready_count_int != int(
        registry_contract_dict["expected_pm_ready_count_int"]
    ):
        raise RuntimeError("PM_READY registry count changed after the spec was frozen.")
    for strategy_alias_str, strategy_spec_dict in selected_by_alias_dict.items():
        strategy_import_str = str(strategy_spec_dict["strategy_import_str"])
        actual_tier_str = strategy_registry.STRATEGY_TIER_DICT[
            strategy_import_str
        ].name
        if actual_tier_str != str(strategy_spec_dict["tier_str"]):
            raise RuntimeError(
                f"{strategy_alias_str} maturity tier changed from the frozen inventory."
            )

    product_spec_by_id_dict = all_product_spec_dict(spec_dict)
    if set(spec_dict["economic_gate_by_candidate"]) != set(
        spec_dict["candidate_portfolios"]
    ):
        raise ValueError("Every candidate must have exactly one economic gate.")
    capital_base_float = float(spec_dict["portfolio_contract"]["capital_base_float"])
    for product_id_str, product_spec_dict in product_spec_by_id_dict.items():
        weight_by_alias_dict = product_spec_dict["weight_by_strategy_alias"]
        if not set(weight_by_alias_dict).issubset(selected_by_alias_dict):
            raise ValueError(f"{product_id_str} uses a non-selected strategy alias.")
        weight_sum_float = float(sum(float(value_obj) for value_obj in weight_by_alias_dict.values()))
        if not math.isclose(weight_sum_float, 1.0, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"{product_id_str} weights sum to {weight_sum_float:.12f}.")
        if any(float(weight_obj) <= 0.0 for weight_obj in weight_by_alias_dict.values()):
            raise ValueError(f"{product_id_str} weights must be positive.")
        for strategy_alias_str, weight_obj in weight_by_alias_dict.items():
            allocated_capital_float = capital_base_float * float(weight_obj)
            if strategy_alias_str in {"dv2", "hpi"} and allocated_capital_float < 25_000.0:
                raise ValueError(f"{product_id_str} violates the single-stock pod floor.")

    for candidate_id_str, candidate_spec_dict in spec_dict[
        "candidate_portfolios"
    ].items():
        reference_id_str = str(candidate_spec_dict["reference_id_str"])
        if reference_id_str not in spec_dict["reference_portfolios"]:
            raise ValueError(f"{candidate_id_str} has an unknown reference.")
    low_touch_alias_set = set(
        spec_dict["candidate_portfolios"]["C4_foundry_low_touch"][
            "weight_by_strategy_alias"
        ]
    )
    daily_alias_set = {
        alias_str
        for alias_str, strategy_spec_dict in selected_by_alias_dict.items()
        if str(strategy_spec_dict["cadence_str"]) == "daily_signal"
    }
    if low_touch_alias_set.intersection(daily_alias_set):
        raise ValueError("The low-touch product cannot contain a daily-signal pod.")
    if bool(spec_dict["capacity_contract"]["phase1_capacity_gate_bool"]):
        raise ValueError("Phase 1 cannot clear Capacity.")
    stats_contract_dict = spec_dict["statistical_contract"]
    if set(stats_contract_dict["holm_family_candidate_id_list"]) != set(
        spec_dict["candidate_portfolios"]
    ):
        raise ValueError("Holm family must include every frozen candidate exactly once.")
    if len(stats_contract_dict["holm_family_candidate_id_list"]) != len(
        spec_dict["candidate_portfolios"]
    ):
        raise ValueError("Holm family cannot contain duplicate candidates.")
    if int(stats_contract_dict["historical_trial_count_floor_int"]) < 60:
        raise ValueError("Historical strategy-search correction cannot fall below 60.")
    if not bool(
        stats_contract_dict["current_candidate_count_added_to_historical_floor_bool"]
    ):
        raise ValueError("The four current candidates must be added to historical trials.")
    if stats_contract_dict["historical_trial_adjustment_method_str"] != (
        "bonferroni_floor_then_holm"
    ):
        raise ValueError("Historical-trial adjustment method changed after freeze.")


def load_spec_dict(spec_path: Path = DEFAULT_SPEC_PATH) -> dict[str, Any]:
    actual_sha256_str = ladder_runner.sha256_file_str(spec_path)
    if actual_sha256_str != FROZEN_SPEC_SHA256_STR:
        raise RuntimeError(
            "Frozen Portfolio Foundry spec hash changed: "
            f"expected {FROZEN_SPEC_SHA256_STR}, got {actual_sha256_str}."
        )
    spec_dict = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    validate_spec_dict(spec_dict)
    return spec_dict


def expanded_source_spec_dict(
    spec_dict: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, dict[str, str]]]:
    selected_by_alias_dict = spec_dict["registry_contract"][
        "selected_strategy_by_alias"
    ]
    capital_base_float = float(spec_dict["portfolio_contract"]["capital_base_float"])
    source_run_by_id_dict: dict[str, dict[str, Any]] = {}
    source_id_by_product_alias_dict: dict[str, dict[str, str]] = {}
    for product_id_str, product_spec_dict in all_product_spec_dict(spec_dict).items():
        source_id_by_product_alias_dict[product_id_str] = {}
        for strategy_alias_str, weight_obj in product_spec_dict[
            "weight_by_strategy_alias"
        ].items():
            allocated_capital_float = capital_base_float * float(weight_obj)
            current_source_id_str = source_id_str(
                strategy_alias_str,
                allocated_capital_float,
            )
            source_id_by_product_alias_dict[product_id_str][
                strategy_alias_str
            ] = current_source_id_str
            strategy_inventory_dict = selected_by_alias_dict[strategy_alias_str]
            source_spec_dict = {
                "strategy_import_str": str(
                    strategy_inventory_dict["strategy_import_str"]
                ),
                "allocated_capital_float": allocated_capital_float,
                "run_variant_kwargs_dict": {},
            }
            if "engine_request_start_date_str" in strategy_inventory_dict:
                source_spec_dict["engine_request_start_date_str"] = str(
                    strategy_inventory_dict["engine_request_start_date_str"]
                )
            existing_source_spec_dict = source_run_by_id_dict.get(current_source_id_str)
            if existing_source_spec_dict is not None and existing_source_spec_dict != source_spec_dict:
                raise RuntimeError(f"Conflicting derived source {current_source_id_str}.")
            source_run_by_id_dict[current_source_id_str] = source_spec_dict

    native_history_by_import_dict = {
        str(strategy_spec_dict["strategy_import_str"]): str(
            strategy_spec_dict["native_history_request_start_date_str"]
        )
        for strategy_spec_dict in selected_by_alias_dict.values()
    }
    expanded_spec_dict = dict(spec_dict)
    expanded_spec_dict["source_runs"] = dict(sorted(source_run_by_id_dict.items()))
    expanded_spec_dict["lineage_contract"] = {
        **dict(spec_dict["lineage_contract"]),
        "native_history_request_start_by_strategy_import": (
            native_history_by_import_dict
        ),
    }
    return expanded_spec_dict, source_id_by_product_alias_dict


def build_global_product_frames(
    expanded_spec_dict: dict[str, Any],
    source_id_by_product_alias_dict: dict[str, dict[str, str]],
    output_dir_path: Path,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
    str,
]:
    source_path_by_id_dict = ladder_runner.load_all_source_path_dict(
        expanded_spec_dict,
        output_dir_path,
    )
    global_idx: pd.DatetimeIndex | None = None
    for source_result_df in source_path_by_id_dict.values():
        source_idx = pd.DatetimeIndex(source_result_df.index)
        global_idx = source_idx if global_idx is None else global_idx.intersection(source_idx)
    if global_idx is None or len(global_idx) < 2:
        raise RuntimeError("No usable global source intersection.")
    global_idx = global_idx.sort_values()
    portfolio_contract_dict = expanded_spec_dict["portfolio_contract"]
    expected_anchor_ts = pd.Timestamp(portfolio_contract_dict["capital_anchor_date_str"])
    expected_end_ts = pd.Timestamp(portfolio_contract_dict["end_date_str"])
    if global_idx[0] != expected_anchor_ts or global_idx[-1] != expected_end_ts:
        raise RuntimeError(
            f"Global intersection is {global_idx[0].date()}..{global_idx[-1].date()}, "
            "not the frozen anchor and endpoint."
        )

    capital_base_float = float(portfolio_contract_dict["capital_base_float"])
    total_value_by_product_dict: dict[str, pd.Series] = {}
    portfolio_value_by_product_dict: dict[str, pd.Series] = {}
    cash_by_product_dict: dict[str, pd.Series] = {}
    for product_id_str, alias_source_id_dict in source_id_by_product_alias_dict.items():
        source_frame_list = [
            source_path_by_id_dict[current_source_id_str].reindex(global_idx)
            for current_source_id_str in alias_source_id_dict.values()
        ]
        if any(source_frame_df.isna().any().any() for source_frame_df in source_frame_list):
            raise RuntimeError(f"{product_id_str} has missing source values.")
        total_value_by_product_dict[product_id_str] = sum(
            (source_frame_df["total_value_float"] for source_frame_df in source_frame_list),
            start=pd.Series(0.0, index=global_idx),
        )
        portfolio_value_by_product_dict[product_id_str] = sum(
            (source_frame_df["portfolio_value_float"] for source_frame_df in source_frame_list),
            start=pd.Series(0.0, index=global_idx),
        )
        cash_by_product_dict[product_id_str] = sum(
            (source_frame_df["cash_float"] for source_frame_df in source_frame_list),
            start=pd.Series(0.0, index=global_idx),
        )
        if not math.isclose(
            float(total_value_by_product_dict[product_id_str].iloc[0]),
            capital_base_float,
            rel_tol=0.0,
            abs_tol=1e-6,
        ):
            raise RuntimeError(f"{product_id_str} anchor capital is wrong.")
        if not math.isclose(
            float(portfolio_value_by_product_dict[product_id_str].iloc[0]),
            0.0,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise RuntimeError(f"{product_id_str} anchor is not all cash.")

    total_value_df = pd.DataFrame(total_value_by_product_dict, index=global_idx)
    portfolio_value_df = pd.DataFrame(portfolio_value_by_product_dict, index=global_idx)
    cash_df = pd.DataFrame(cash_by_product_dict, index=global_idx)
    return_df = total_value_df.pct_change(fill_method=None)
    return_df.iloc[0] = 0.0

    benchmark_price_df = load_price_timeseries(
        str(portfolio_contract_dict["benchmark_symbol_str"]),
        adjustment_str=TOTALRETURN_ADJUSTMENT_STR,
        start_date_str=expected_anchor_ts.date().isoformat(),
        end_date_str=expected_end_ts.date().isoformat(),
    )
    benchmark_close_ser = benchmark_price_df["Close"].astype(float).reindex(global_idx)
    if benchmark_close_ser.isna().any():
        raise RuntimeError("$SPXTR is incomplete on the exact global source index.")
    benchmark_total_value_ser = (
        benchmark_close_ser / float(benchmark_close_ser.iloc[0]) * capital_base_float
    ).rename("SPXTR")
    benchmark_return_ser = benchmark_total_value_ser.pct_change(fill_method=None)
    benchmark_return_ser.iloc[0] = 0.0
    global_index_sha256_str = hashlib.sha256(
        "\n".join(date_ts.date().isoformat() for date_ts in global_idx).encode("utf-8")
    ).hexdigest()
    for frame_df in (total_value_df, portfolio_value_df, cash_df, return_df):
        frame_df.index.name = "date"
    return (
        total_value_df,
        portfolio_value_df,
        cash_df,
        benchmark_total_value_ser,
        benchmark_return_ser,
        global_index_sha256_str,
    )


def calculate_headline_metric_df(
    total_value_df: pd.DataFrame,
    cash_df: pd.DataFrame,
    benchmark_total_value_ser: pd.Series,
    benchmark_return_ser: pd.Series,
    spec_dict: dict[str, Any],
) -> pd.DataFrame:
    stats_contract_dict = spec_dict["statistical_contract"]
    row_list: list[dict[str, Any]] = []
    for product_id_str in total_value_df.columns:
        row_list.append(
            {
                "product_id_str": product_id_str,
                "product_type_str": (
                    "candidate"
                    if product_id_str in spec_dict["candidate_portfolios"]
                    else "reference"
                ),
                "objective_str": all_product_spec_dict(spec_dict)[product_id_str][
                    "objective_str"
                ],
                **ladder_runner.calculate_path_metric_dict(
                    total_value_df[product_id_str],
                    benchmark_return_ser=benchmark_return_ser,
                    annualization_day_int=int(stats_contract_dict["annualization_day_int"]),
                    es_quantile_float=float(stats_contract_dict["es_quantile_float"]),
                ),
                "negative_cash_day_count_int": int(
                    (cash_df[product_id_str].iloc[1:] < 0.0).sum()
                ),
                "minimum_cash_float": float(cash_df[product_id_str].min()),
            }
        )
    row_list.append(
        {
            "product_id_str": "SPXTR",
            "product_type_str": "market_benchmark",
            "objective_str": "market",
            **ladder_runner.calculate_path_metric_dict(
                benchmark_total_value_ser,
                benchmark_return_ser=benchmark_return_ser,
                annualization_day_int=int(stats_contract_dict["annualization_day_int"]),
                es_quantile_float=float(stats_contract_dict["es_quantile_float"]),
            ),
            "negative_cash_day_count_int": 0,
            "minimum_cash_float": float("nan"),
        }
    )
    return pd.DataFrame(row_list)


def calculate_subperiod_metric_df(
    total_value_df: pd.DataFrame,
    benchmark_return_ser: pd.Series,
    spec_dict: dict[str, Any],
) -> pd.DataFrame:
    row_list: list[dict[str, Any]] = []
    return_observation_count_int = len(total_value_df) - 1
    covered_return_date_list: list[pd.Timestamp] = []
    for third_position_int, return_slice_obj in enumerate(
        ladder_runner.equal_observation_third_slice_tuple(
            return_observation_count_int
        ),
        start=1,
    ):
        if return_slice_obj.start is None or return_slice_obj.stop is None:
            raise RuntimeError("Subperiod return slices require finite bounds.")
        # *** CRITICAL*** The split is over realized return rows. The prior NAV
        # is prepended only as a measurement anchor, so every boundary return
        # enters one and only one chronological third.
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
        for product_id_str in total_value_df.columns:
            row_list.append(
                {
                    "subperiod_id_str": f"third_{third_position_int}",
                    "anchor_date_str": subperiod_total_value_df.index[
                        0
                    ].date().isoformat(),
                    "start_date_str": subperiod_return_date_idx[0].date().isoformat(),
                    "end_date_str": subperiod_total_value_df.index[
                        -1
                    ].date().isoformat(),
                    "product_id_str": product_id_str,
                    **ladder_runner.calculate_path_metric_dict(
                        subperiod_total_value_df[product_id_str],
                        benchmark_return_ser=subperiod_benchmark_return_ser,
                        annualization_day_int=int(
                            spec_dict["statistical_contract"]["annualization_day_int"]
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
        crisis_date_idx = total_value_df.loc[start_date_str:end_date_str].index
        if len(crisis_date_idx) == 0:
            row_list.append(
                {
                    "crisis_name_str": crisis_name_str,
                    "status_str": "N/A",
                    "product_id_str": None,
                    "observation_count_int": 0,
                }
            )
            continue
        first_result_position_int = int(total_value_df.index.get_loc(crisis_date_idx[0]))
        if first_result_position_int == 0:
            crisis_total_value_df = total_value_df.loc[
                crisis_date_idx[0]:crisis_date_idx[-1]
            ]
        else:
            # *** CRITICAL*** Include the prior NAV measurement anchor so the
            # return into the first crisis session is not silently discarded.
            crisis_total_value_df = total_value_df.iloc[
                first_result_position_int - 1:
                int(total_value_df.index.get_loc(crisis_date_idx[-1])) + 1
            ]
        if len(crisis_total_value_df) < 2:
            continue
        for product_id_str in total_value_df.columns:
            metric_dict = ladder_runner.calculate_path_metric_dict(
                crisis_total_value_df[product_id_str]
            )
            row_list.append(
                {
                    "crisis_name_str": crisis_name_str,
                    "status_str": "available_diagnostic_only",
                    "product_id_str": product_id_str,
                    "anchor_date_str": crisis_total_value_df.index[
                        0
                    ].date().isoformat(),
                    "start_date_str": crisis_total_value_df.index[
                        1
                    ].date().isoformat(),
                    "end_date_str": crisis_total_value_df.index[
                        -1
                    ].date().isoformat(),
                    "observation_count_int": metric_dict["observation_count_int"],
                    "cumulative_return_float": float(
                        crisis_total_value_df[product_id_str].iloc[-1]
                        / crisis_total_value_df[product_id_str].iloc[0]
                        - 1.0
                    ),
                    "max_drawdown_float": metric_dict["max_drawdown_float"],
                    "es5_loss_float": metric_dict["es5_loss_float"],
                }
            )
    return pd.DataFrame(row_list)


def calculate_source_correlation_df(
    source_id_by_product_alias_dict: dict[str, dict[str, str]],
    output_dir_path: Path,
    global_idx: pd.DatetimeIndex,
    candidate_id_set: set[str],
) -> pd.DataFrame:
    row_list: list[dict[str, Any]] = []
    for candidate_id_str in sorted(candidate_id_set):
        alias_source_id_dict = source_id_by_product_alias_dict[candidate_id_str]
        source_return_by_alias_dict: dict[str, pd.Series] = {}
        for strategy_alias_str, current_source_id_str in alias_source_id_dict.items():
            source_result_df = ladder_runner.read_source_path_df(
                output_dir_path / "source_paths" / f"{current_source_id_str}.csv.gz"
            ).reindex(global_idx)
            source_return_by_alias_dict[strategy_alias_str] = source_result_df[
                "total_value_float"
            ].pct_change(fill_method=None).iloc[1:]
        correlation_df = pd.DataFrame(source_return_by_alias_dict).corr()
        for left_alias_str in correlation_df.index:
            for right_alias_str in correlation_df.columns:
                row_list.append(
                    {
                        "candidate_id_str": candidate_id_str,
                        "left_strategy_alias_str": left_alias_str,
                        "right_strategy_alias_str": right_alias_str,
                        "daily_return_correlation_float": float(
                            correlation_df.loc[left_alias_str, right_alias_str]
                        ),
                    }
                )
    return pd.DataFrame(row_list)


def calculate_transaction_overlap_df(
    source_id_by_product_alias_dict: dict[str, dict[str, str]],
    output_dir_path: Path,
) -> pd.DataFrame:
    row_list: list[dict[str, Any]] = []
    for product_id_str, alias_source_id_dict in source_id_by_product_alias_dict.items():
        transaction_frame_list = [
            pd.read_csv(
                output_dir_path
                / "source_transactions"
                / f"{current_source_id_str}.csv.gz"
            )
            for current_source_id_str in alias_source_id_dict.values()
        ]
        transaction_df = pd.concat(transaction_frame_list, ignore_index=True)
        if len(transaction_df) == 0:
            row_list.append(
                {
                    "product_id_str": product_id_str,
                    "transaction_count_int": 0,
                    "same_day_symbol_row_count_int": 0,
                    "overlap_row_count_int": 0,
                    "gross_notional_float": 0.0,
                    "maximum_overlap_gross_notional_float": 0.0,
                }
            )
            continue
        transaction_df["date"] = pd.to_datetime(transaction_df["date"]).dt.normalize()
        transaction_df["gross_notional_float"] = pd.to_numeric(
            transaction_df["signed_notional_float"]
        ).abs()
        grouped_df = transaction_df.groupby(["date", "asset_str"], sort=True).agg(
            row_count_int=("source_id_str", "size"),
            source_count_int=("source_id_str", "nunique"),
            gross_notional_float=("gross_notional_float", "sum"),
        )
        overlap_df = grouped_df.loc[grouped_df["source_count_int"] > 1]
        row_list.append(
            {
                "product_id_str": product_id_str,
                "transaction_count_int": int(len(transaction_df)),
                "same_day_symbol_row_count_int": int(
                    grouped_df.loc[grouped_df["row_count_int"] > 1, "row_count_int"].sum()
                ),
                "overlap_row_count_int": int(overlap_df["row_count_int"].sum()),
                "gross_notional_float": float(
                    transaction_df["gross_notional_float"].sum()
                ),
                "maximum_overlap_gross_notional_float": float(
                    overlap_df["gross_notional_float"].max()
                    if len(overlap_df) > 0
                    else 0.0
                ),
            }
        )
    return pd.DataFrame(row_list)


def bootstrap_metric_array_by_product_dict(
    return_df: pd.DataFrame,
    spec_dict: dict[str, Any],
    mean_block_length_int: int,
) -> dict[str, dict[str, np.ndarray]]:
    stats_contract_dict = spec_dict["statistical_contract"]
    simulation_count_int = int(stats_contract_dict["bootstrap_iteration_count_int"])
    chunk_size_int = int(stats_contract_dict["bootstrap_chunk_size_int"])
    annualization_day_int = int(stats_contract_dict["annualization_day_int"])
    es_quantile_float = float(stats_contract_dict["es_quantile_float"])
    clean_return_df = return_df.iloc[1:].astype(float)
    if clean_return_df.isna().any().any():
        raise ValueError("Bootstrap returns must be complete.")
    product_id_list = list(clean_return_df.columns)
    return_arr = clean_return_df.to_numpy(dtype=float)
    observation_count_int = int(len(clean_return_df))
    metric_array_by_product_dict = {
        product_id_str: {
            metric_name_str: np.empty(simulation_count_int, dtype=float)
            for metric_name_str in (
                "cagr_float",
                "sharpe_float",
                "max_drawdown_float",
                "es5_loss_float",
            )
        }
        for product_id_str in product_id_list
    }
    for chunk_start_int in range(0, simulation_count_int, chunk_size_int):
        chunk_end_int = min(chunk_start_int + chunk_size_int, simulation_count_int)
        chunk_index_mat = ladder_runner.stationary_bootstrap_index_chunk_mat(
            sample_size_int=observation_count_int,
            simulation_count_int=chunk_end_int - chunk_start_int,
            mean_block_length_int=mean_block_length_int,
            random_seed_int=int(stats_contract_dict["random_seed_int"]),
            simulation_start_int=chunk_start_int,
        )
        sampled_return_arr = return_arr[chunk_index_mat]
        gross_return_arr = 1.0 + sampled_return_arr
        terminal_multiple_mat = gross_return_arr.prod(axis=1)
        cagr_mat = terminal_multiple_mat ** (
            float(annualization_day_int) / observation_count_int
        ) - 1.0
        daily_mean_mat = sampled_return_arr.mean(axis=1)
        daily_std_mat = sampled_return_arr.std(axis=1, ddof=1)
        sharpe_mat = np.divide(
            daily_mean_mat * math.sqrt(annualization_day_int),
            daily_std_mat,
            out=np.full_like(daily_mean_mat, np.nan),
            where=daily_std_mat > 0.0,
        )
        equity_arr = gross_return_arr.cumprod(axis=1)
        # *** CRITICAL *** drawdown-sensitive: every bootstrap path starts at
        # unit NAV before its first sampled return. Omitting that anchor hides
        # a first-observation loss and biases MaxDD toward zero.
        running_peak_arr = np.maximum(
            1.0,
            np.maximum.accumulate(equity_arr, axis=1),
        )
        max_drawdown_mat = (equity_arr / running_peak_arr - 1.0).min(axis=1)
        tail_cutoff_mat = np.quantile(
            sampled_return_arr,
            es_quantile_float,
            axis=1,
        )
        tail_mask_arr = sampled_return_arr <= tail_cutoff_mat[:, None, :]
        tail_sum_mat = np.where(tail_mask_arr, sampled_return_arr, 0.0).sum(axis=1)
        tail_count_mat = tail_mask_arr.sum(axis=1)
        es5_loss_mat = np.maximum(0.0, -tail_sum_mat / tail_count_mat)
        for product_position_int, product_id_str in enumerate(product_id_list):
            target_slice_obj = slice(chunk_start_int, chunk_end_int)
            metric_array_by_product_dict[product_id_str]["cagr_float"][
                target_slice_obj
            ] = cagr_mat[:, product_position_int]
            metric_array_by_product_dict[product_id_str]["sharpe_float"][
                target_slice_obj
            ] = sharpe_mat[:, product_position_int]
            metric_array_by_product_dict[product_id_str]["max_drawdown_float"][
                target_slice_obj
            ] = max_drawdown_mat[:, product_position_int]
            metric_array_by_product_dict[product_id_str]["es5_loss_float"][
                target_slice_obj
            ] = es5_loss_mat[:, product_position_int]
    return metric_array_by_product_dict


def calculate_bootstrap_evidence(
    return_df: pd.DataFrame,
    headline_metric_df: pd.DataFrame,
    spec_dict: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    stats_contract_dict = spec_dict["statistical_contract"]
    observed_metric_by_id_dict = headline_metric_df.set_index(
        "product_id_str"
    ).to_dict(orient="index")
    block_length_int_list = [
        int(stats_contract_dict["primary_mean_block_length_int"]),
        *[
            int(value_obj)
            for value_obj in stats_contract_dict[
                "sensitivity_mean_block_length_list"
            ]
        ],
    ]
    summary_row_list: list[dict[str, Any]] = []
    for mean_block_length_int in block_length_int_list:
        metric_array_by_product_dict = bootstrap_metric_array_by_product_dict(
            return_df,
            spec_dict,
            mean_block_length_int,
        )
        for candidate_id_str, candidate_spec_dict in spec_dict[
            "candidate_portfolios"
        ].items():
            reference_id_str = str(candidate_spec_dict["reference_id_str"])
            economic_gate_dict = spec_dict["economic_gate_by_candidate"][
                candidate_id_str
            ]
            candidate_metric_array_dict = metric_array_by_product_dict[
                candidate_id_str
            ]
            reference_metric_array_dict = metric_array_by_product_dict[
                reference_id_str
            ]
            cagr_delta_arr = (
                candidate_metric_array_dict["cagr_float"]
                - reference_metric_array_dict["cagr_float"]
            )
            sharpe_delta_arr = (
                candidate_metric_array_dict["sharpe_float"]
                - reference_metric_array_dict["sharpe_float"]
            )
            max_drawdown_improvement_arr = (
                candidate_metric_array_dict["max_drawdown_float"]
                - reference_metric_array_dict["max_drawdown_float"]
            )
            es5_reduction_arr = 1.0 - np.divide(
                candidate_metric_array_dict["es5_loss_float"],
                reference_metric_array_dict["es5_loss_float"],
                out=np.full_like(candidate_metric_array_dict["es5_loss_float"], np.nan),
                where=reference_metric_array_dict["es5_loss_float"] > 0.0,
            )
            observed_candidate_metric_dict = observed_metric_by_id_dict[
                candidate_id_str
            ]
            observed_reference_metric_dict = observed_metric_by_id_dict[
                reference_id_str
            ]
            observed_cagr_delta_float = float(
                observed_candidate_metric_dict["cagr_float"]
                - observed_reference_metric_dict["cagr_float"]
            )
            observed_sharpe_delta_float = float(
                observed_candidate_metric_dict["sharpe_float"]
                - observed_reference_metric_dict["sharpe_float"]
            )
            observed_max_drawdown_improvement_float = float(
                observed_candidate_metric_dict["max_drawdown_float"]
                - observed_reference_metric_dict["max_drawdown_float"]
            )
            observed_es5_reduction_float = float(
                1.0
                - observed_candidate_metric_dict["es5_loss_float"]
                / observed_reference_metric_dict["es5_loss_float"]
            )
            minimum_cagr_delta_float = float(
                economic_gate_dict["minimum_cagr_delta_vs_reference_float"]
            )
            minimum_sharpe_delta_float = float(
                economic_gate_dict["minimum_sharpe_delta_vs_reference_float"]
            )
            minimum_max_drawdown_improvement_float = float(
                economic_gate_dict["minimum_max_drawdown_improvement_float"]
            )
            minimum_es5_reduction_float = float(
                economic_gate_dict["minimum_es5_reduction_fraction_float"]
            )
            composite_success_arr = (
                (cagr_delta_arr >= minimum_cagr_delta_float)
                & (sharpe_delta_arr >= minimum_sharpe_delta_float)
                & (
                    max_drawdown_improvement_arr
                    >= minimum_max_drawdown_improvement_float
                )
                & (es5_reduction_arr >= minimum_es5_reduction_float)
            )
            p_value_list = [
                ladder_runner.centered_one_sided_p_value_float(
                    cagr_delta_arr,
                    observed_cagr_delta_float,
                    minimum_cagr_delta_float,
                ),
                ladder_runner.centered_one_sided_p_value_float(
                    sharpe_delta_arr,
                    observed_sharpe_delta_float,
                    minimum_sharpe_delta_float,
                ),
                ladder_runner.centered_one_sided_p_value_float(
                    max_drawdown_improvement_arr,
                    observed_max_drawdown_improvement_float,
                    minimum_max_drawdown_improvement_float,
                ),
                ladder_runner.centered_one_sided_p_value_float(
                    es5_reduction_arr,
                    observed_es5_reduction_float,
                    minimum_es5_reduction_float,
                ),
            ]
            summary_row_list.append(
                {
                    "candidate_id_str": candidate_id_str,
                    "reference_id_str": reference_id_str,
                    "mean_block_length_int": mean_block_length_int,
                    "observed_cagr_delta_float": observed_cagr_delta_float,
                    "observed_sharpe_delta_float": observed_sharpe_delta_float,
                    "observed_max_drawdown_improvement_float": (
                        observed_max_drawdown_improvement_float
                    ),
                    "observed_es5_reduction_fraction_float": (
                        observed_es5_reduction_float
                    ),
                    "composite_success_probability_float": float(
                        np.mean(composite_success_arr)
                    ),
                    "cagr_p_value_float": p_value_list[0],
                    "sharpe_p_value_float": p_value_list[1],
                    "max_drawdown_p_value_float": p_value_list[2],
                    "es5_p_value_float": p_value_list[3],
                    "joint_raw_p_value_float": float(max(p_value_list)),
                }
            )
    bootstrap_summary_df = pd.DataFrame(summary_row_list)
    primary_block_length_int = int(stats_contract_dict["primary_mean_block_length_int"])
    primary_mask_ser = (
        bootstrap_summary_df["mean_block_length_int"] == primary_block_length_int
    )
    primary_df = bootstrap_summary_df.loc[primary_mask_ser].copy()
    historical_trial_count_int = int(
        stats_contract_dict["historical_trial_count_floor_int"]
    )
    cumulative_trial_family_count_int = historical_trial_count_int + len(primary_df)
    raw_p_value_arr = primary_df["joint_raw_p_value_float"].to_numpy(dtype=float)
    historical_trial_adjusted_p_value_arr = np.minimum(
        1.0,
        raw_p_value_arr * cumulative_trial_family_count_int,
    )
    reject_bool_arr, adjusted_p_value_arr, _, _ = multipletests(
        historical_trial_adjusted_p_value_arr,
        alpha=float(stats_contract_dict["maximum_holm_adjusted_p_value_float"]),
        method="holm",
    )
    holm_df = pd.DataFrame(
        {
            "candidate_id_str": primary_df["candidate_id_str"].tolist(),
            "raw_p_value_float": raw_p_value_arr,
            "historical_trial_count_int": historical_trial_count_int,
            "cumulative_trial_family_count_int": cumulative_trial_family_count_int,
            "historical_trial_adjusted_p_value_float": (
                historical_trial_adjusted_p_value_arr
            ),
            "holm_adjusted_p_value_float": adjusted_p_value_arr,
            "holm_reject_bool": reject_bool_arr,
        }
    )
    return bootstrap_summary_df, holm_df


def evaluate_candidate_gate_df(
    headline_metric_df: pd.DataFrame,
    subperiod_metric_df: pd.DataFrame,
    bootstrap_summary_df: pd.DataFrame,
    holm_df: pd.DataFrame,
    source_run_summary_df: pd.DataFrame,
    source_id_by_product_alias_dict: dict[str, dict[str, str]],
    spec_dict: dict[str, Any],
    source_lineage_gate_bool: bool,
) -> pd.DataFrame:
    stats_contract_dict = spec_dict["statistical_contract"]
    robustness_contract_dict = spec_dict["robustness_contract"]
    headline_by_id_dict = headline_metric_df.set_index("product_id_str").to_dict(
        orient="index"
    )
    subperiod_by_key_dict = {
        (str(row_ser["subperiod_id_str"]), str(row_ser["product_id_str"])): row_ser
        for _, row_ser in subperiod_metric_df.iterrows()
    }
    source_summary_by_id_dict = source_run_summary_df.set_index(
        "source_id_str"
    ).to_dict(orient="index")
    holm_by_id_dict = holm_df.set_index("candidate_id_str").to_dict(orient="index")
    primary_block_length_int = int(stats_contract_dict["primary_mean_block_length_int"])
    bootstrap_by_key_dict = {
        (str(row_ser["candidate_id_str"]), int(row_ser["mean_block_length_int"])): row_ser
        for _, row_ser in bootstrap_summary_df.iterrows()
    }
    row_list: list[dict[str, Any]] = []
    for candidate_id_str, candidate_spec_dict in spec_dict[
        "candidate_portfolios"
    ].items():
        reference_id_str = str(candidate_spec_dict["reference_id_str"])
        candidate_metric_dict = headline_by_id_dict[candidate_id_str]
        reference_metric_dict = headline_by_id_dict[reference_id_str]
        economic_gate_dict = spec_dict["economic_gate_by_candidate"][candidate_id_str]
        cagr_delta_float = float(
            candidate_metric_dict["cagr_float"] - reference_metric_dict["cagr_float"]
        )
        sharpe_delta_float = float(
            candidate_metric_dict["sharpe_float"]
            - reference_metric_dict["sharpe_float"]
        )
        max_drawdown_improvement_float = float(
            candidate_metric_dict["max_drawdown_float"]
            - reference_metric_dict["max_drawdown_float"]
        )
        es5_reduction_fraction_float = float(
            1.0
            - candidate_metric_dict["es5_loss_float"]
            / reference_metric_dict["es5_loss_float"]
        )
        beta_delta_float = float(
            candidate_metric_dict["market_beta_float"]
            - reference_metric_dict["market_beta_float"]
        )
        economic_gate_bool = bool(
            cagr_delta_float
            >= float(economic_gate_dict["minimum_cagr_delta_vs_reference_float"])
            and sharpe_delta_float
            >= float(economic_gate_dict["minimum_sharpe_delta_vs_reference_float"])
            and max_drawdown_improvement_float
            >= float(economic_gate_dict["minimum_max_drawdown_improvement_float"])
            and es5_reduction_fraction_float
            >= float(economic_gate_dict["minimum_es5_reduction_fraction_float"])
            and beta_delta_float
            <= float(economic_gate_dict["maximum_beta_delta_vs_reference_float"])
            and abs(float(candidate_metric_dict["max_drawdown_float"]))
            <= float(economic_gate_dict["maximum_absolute_max_drawdown_float"])
        )

        passing_third_count_int = 0
        third_gate_bool_list: list[bool] = []
        for third_position_int in range(1, 4):
            third_id_str = f"third_{third_position_int}"
            candidate_third_ser = subperiod_by_key_dict[
                (third_id_str, candidate_id_str)
            ]
            reference_third_ser = subperiod_by_key_dict[
                (third_id_str, reference_id_str)
            ]
            third_gate_bool = bool(
                float(candidate_third_ser["cagr_float"])
                - float(reference_third_ser["cagr_float"])
                >= -float(
                    robustness_contract_dict[
                        "maximum_third_cagr_shortfall_vs_reference_float"
                    ]
                )
                and float(candidate_third_ser["max_drawdown_float"])
                - float(reference_third_ser["max_drawdown_float"])
                >= -float(
                    robustness_contract_dict[
                        "maximum_third_max_drawdown_worsening_float"
                    ]
                )
            )
            third_gate_bool_list.append(third_gate_bool)
            passing_third_count_int += int(third_gate_bool)
        subperiod_gate_bool = bool(
            passing_third_count_int
            >= int(robustness_contract_dict["minimum_passing_third_count_int"])
        )

        primary_bootstrap_ser = bootstrap_by_key_dict[
            (candidate_id_str, primary_block_length_int)
        ]
        primary_success_gate_bool = bool(
            float(primary_bootstrap_ser["composite_success_probability_float"])
            >= float(
                stats_contract_dict[
                    "minimum_primary_composite_success_probability_float"
                ]
            )
        )
        sensitivity_success_gate_bool = all(
            float(
                bootstrap_by_key_dict[(candidate_id_str, int(block_length_obj))][
                    "composite_success_probability_float"
                ]
            )
            >= float(
                stats_contract_dict[
                    "minimum_sensitivity_composite_success_probability_float"
                ]
            )
            for block_length_obj in stats_contract_dict[
                "sensitivity_mean_block_length_list"
            ]
        )
        holm_adjusted_p_value_float = float(
            holm_by_id_dict[candidate_id_str]["holm_adjusted_p_value_float"]
        )
        statistical_gate_bool = bool(
            primary_success_gate_bool
            and sensitivity_success_gate_bool
            and holm_adjusted_p_value_float
            <= float(stats_contract_dict["maximum_holm_adjusted_p_value_float"])
        )

        candidate_source_id_list = list(
            source_id_by_product_alias_dict[candidate_id_str].values()
        )
        reference_source_id_list = list(
            source_id_by_product_alias_dict[reference_id_str].values()
        )
        candidate_negative_cash_day_count_int = int(
            sum(
                int(source_summary_by_id_dict[current_source_id_str][
                    "negative_cash_day_count_int"
                ])
                for current_source_id_str in candidate_source_id_list
            )
        )
        reference_negative_cash_day_count_int = int(
            sum(
                int(source_summary_by_id_dict[current_source_id_str][
                    "negative_cash_day_count_int"
                ])
                for current_source_id_str in reference_source_id_list
            )
        )
        comparison_source_id_set = set(candidate_source_id_list) | set(
            reference_source_id_list
        )
        negative_cash_day_count_int = int(
            sum(
                int(source_summary_by_id_dict[current_source_id_str][
                    "negative_cash_day_count_int"
                ])
                for current_source_id_str in comparison_source_id_set
            )
        )
        financing_gate_bool = negative_cash_day_count_int == 0
        non_capacity_gate_bool = bool(
            economic_gate_bool
            and subperiod_gate_bool
            and statistical_gate_bool
            and financing_gate_bool
            and source_lineage_gate_bool
        )
        capacity_gate_bool = bool(
            spec_dict["capacity_contract"]["phase1_capacity_gate_bool"]
        )
        promotion_gate_bool = bool(non_capacity_gate_bool and capacity_gate_bool)
        row_list.append(
            {
                "candidate_id_str": candidate_id_str,
                "reference_id_str": reference_id_str,
                "cagr_delta_vs_reference_float": cagr_delta_float,
                "sharpe_delta_vs_reference_float": sharpe_delta_float,
                "max_drawdown_improvement_vs_reference_float": (
                    max_drawdown_improvement_float
                ),
                "es5_reduction_vs_reference_fraction_float": (
                    es5_reduction_fraction_float
                ),
                "beta_delta_vs_reference_float": beta_delta_float,
                "economic_gate_bool": economic_gate_bool,
                "passing_third_count_int": passing_third_count_int,
                "third_gate_bool_list_str": json.dumps(third_gate_bool_list),
                "subperiod_gate_bool": subperiod_gate_bool,
                "primary_bootstrap_success_probability_float": float(
                    primary_bootstrap_ser["composite_success_probability_float"]
                ),
                "holm_adjusted_p_value_float": holm_adjusted_p_value_float,
                "statistical_gate_bool": statistical_gate_bool,
                "candidate_negative_cash_day_count_int": (
                    candidate_negative_cash_day_count_int
                ),
                "reference_negative_cash_day_count_int": (
                    reference_negative_cash_day_count_int
                ),
                "negative_cash_day_count_int": negative_cash_day_count_int,
                "unmodeled_financing_gate_bool": financing_gate_bool,
                "source_lineage_gate_bool": source_lineage_gate_bool,
                "non_capacity_gate_bool": non_capacity_gate_bool,
                "capacity_gate_bool": capacity_gate_bool,
                "promotion_gate_bool": promotion_gate_bool,
                "decision_str": (
                    "advance_to_separately_preregistered_capacity_phase"
                    if non_capacity_gate_bool
                    else "reject_without_weight_tuning"
                ),
            }
        )
    return pd.DataFrame(row_list)


def validate_source_lineage_bool(
    expanded_spec_dict: dict[str, Any],
    source_run_summary_df: pd.DataFrame,
    output_dir_path: Path,
) -> bool:
    expected_source_id_list = list(expanded_spec_dict["source_runs"])
    actual_source_id_list = source_run_summary_df["source_id_str"].astype(str).tolist()
    if actual_source_id_list != expected_source_id_list:
        raise RuntimeError(
            "Completed source order or set differs from the frozen source contract."
        )
    if source_run_summary_df["source_id_str"].astype(str).duplicated().any():
        raise RuntimeError("Completed source summary contains duplicate source IDs.")
    portfolio_contract_dict = expanded_spec_dict["portfolio_contract"]
    lineage_contract_dict = expanded_spec_dict["lineage_contract"]
    expected_requested_start_date_str = str(
        portfolio_contract_dict["requested_start_date_str"]
    )
    expected_anchor_date_str = str(portfolio_contract_dict["capital_anchor_date_str"])
    expected_execution_start_date_str = str(
        portfolio_contract_dict["effective_execution_start_date_str"]
    )
    expected_end_date_str = str(portfolio_contract_dict["end_date_str"])
    native_history_by_import_dict = lineage_contract_dict[
        "native_history_request_start_by_strategy_import"
    ]
    current_shared_hash_dict = ladder_runner.shared_execution_dependency_hash_dict()
    summary_by_source_id_df = source_run_summary_df.set_index("source_id_str")
    for current_source_id_str, source_spec_dict in expanded_spec_dict[
        "source_runs"
    ].items():
        source_row_ser = summary_by_source_id_df.loc[current_source_id_str]
        source_path = output_dir_path / "source_paths" / f"{current_source_id_str}.csv.gz"
        transaction_path = (
            output_dir_path
            / "source_transactions"
            / f"{current_source_id_str}.csv.gz"
        )
        metadata_path = (
            output_dir_path / "source_metadata" / f"{current_source_id_str}.json"
        )
        if not all(
            artifact_path.is_file()
            for artifact_path in (source_path, transaction_path, metadata_path)
        ):
            raise RuntimeError(f"{current_source_id_str} is missing a source artifact.")
        metadata_dict = json.loads(metadata_path.read_text(encoding="utf-8"))
        expected_strategy_import_str = str(source_spec_dict["strategy_import_str"])
        expected_native_history_start_date_str = str(
            native_history_by_import_dict[expected_strategy_import_str]
        )
        if metadata_dict.get("source_id_str") != current_source_id_str:
            raise RuntimeError(f"{current_source_id_str} metadata source ID changed.")
        if metadata_dict.get("strategy_import_str") != expected_strategy_import_str:
            raise RuntimeError(f"{current_source_id_str} strategy import changed.")
        if (
            metadata_dict.get("requested_history_start_date_str")
            != expected_requested_start_date_str
        ):
            raise RuntimeError(f"{current_source_id_str} requested history changed.")
        if (
            metadata_dict.get("native_history_request_start_date_str")
            != expected_native_history_start_date_str
        ):
            raise RuntimeError(f"{current_source_id_str} native history changed.")
        if not math.isclose(
            float(metadata_dict.get("allocated_capital_float", np.nan)),
            float(source_spec_dict["allocated_capital_float"]),
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise RuntimeError(f"{current_source_id_str} allocated capital changed.")
        if metadata_dict.get("run_variant_kwargs_dict", {}) != dict(
            source_spec_dict.get("run_variant_kwargs_dict", {})
        ):
            raise RuntimeError(f"{current_source_id_str} run kwargs changed.")
        expected_engine_request_start_date_str = str(
            source_spec_dict.get(
                "engine_request_start_date_str",
                expected_execution_start_date_str,
            )
        )
        if (
            metadata_dict.get("engine_request_start_date_str")
            != expected_engine_request_start_date_str
        ):
            raise RuntimeError(
                f"{current_source_id_str} engine request start changed."
            )
        if metadata_dict.get("actual_start_date_str") != expected_anchor_date_str:
            raise RuntimeError(f"{current_source_id_str} has the wrong cash anchor.")
        if (
            metadata_dict.get("strategy_result_start_date_str")
            != expected_execution_start_date_str
        ):
            raise RuntimeError(
                f"{current_source_id_str} has the wrong first result date."
            )
        if metadata_dict.get("actual_end_date_str") != expected_end_date_str:
            raise RuntimeError(f"{current_source_id_str} has the wrong endpoint.")
        if str(source_row_ser["actual_start_date_str"]) != expected_anchor_date_str:
            raise RuntimeError(f"{current_source_id_str} summary anchor changed.")
        if str(source_row_ser["actual_end_date_str"]) != expected_end_date_str:
            raise RuntimeError(f"{current_source_id_str} summary endpoint changed.")
        if ladder_runner.sha256_file_str(source_path) != str(
            source_row_ser["source_path_sha256_str"]
        ):
            raise RuntimeError(f"{current_source_id_str} source path hash changed.")
        if ladder_runner.sha256_file_str(transaction_path) != str(
            source_row_ser["transaction_path_sha256_str"]
        ):
            raise RuntimeError(f"{current_source_id_str} transaction hash changed.")
        if metadata_dict.get("source_path_sha256_str") != ladder_runner.sha256_file_str(
            source_path
        ):
            raise RuntimeError(f"{current_source_id_str} metadata path hash changed.")
        if metadata_dict.get(
            "transaction_path_sha256_str"
        ) != ladder_runner.sha256_file_str(transaction_path):
            raise RuntimeError(
                f"{current_source_id_str} metadata transaction hash changed."
            )
        if str(source_row_ser["metadata_sha256_str"]) != ladder_runner.sha256_file_str(
            metadata_path
        ):
            raise RuntimeError(f"{current_source_id_str} metadata file hash changed.")
        module_path = Path(str(metadata_dict.get("module_path_str", "")))
        if not module_path.is_file() or metadata_dict.get(
            "module_sha256_str"
        ) != ladder_runner.sha256_file_str(module_path):
            raise RuntimeError(f"{current_source_id_str} strategy module changed.")
        if metadata_dict.get(
            "shared_execution_dependency_hash_dict"
        ) != current_shared_hash_dict:
            raise RuntimeError(
                f"{current_source_id_str} shared execution code changed."
            )
        source_path_df = ladder_runner.read_source_path_df(source_path)
        expected_capital_float = float(source_spec_dict["allocated_capital_float"])
        if (
            source_path_df.index[0] != pd.Timestamp(expected_anchor_date_str)
            or not math.isclose(
                float(source_path_df.iloc[0]["total_value_float"]),
                expected_capital_float,
                rel_tol=0.0,
                abs_tol=1e-6,
            )
            or not math.isclose(
                float(source_path_df.iloc[0]["cash_float"]),
                expected_capital_float,
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
            raise RuntimeError(f"{current_source_id_str} cash anchor state changed.")
        if (
            len(source_path_df) < 2
            or source_path_df.index[1] != pd.Timestamp(expected_execution_start_date_str)
        ):
            raise RuntimeError(
                f"{current_source_id_str} first realized result date changed."
            )
    return True


def create_charts(
    total_value_df: pd.DataFrame,
    gate_df: pd.DataFrame,
    spec_dict: dict[str, Any],
    output_dir_path: Path,
) -> None:
    chart_dir_path = output_dir_path / "charts"
    chart_dir_path.mkdir(parents=True, exist_ok=True)
    candidate_spec_by_id_dict = spec_dict["candidate_portfolios"]
    figure_obj, axis_arr = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
    for axis_obj, (candidate_id_str, candidate_spec_dict) in zip(
        axis_arr.ravel(),
        candidate_spec_by_id_dict.items(),
    ):
        reference_id_str = str(candidate_spec_dict["reference_id_str"])
        candidate_equity_ser = (
            total_value_df[candidate_id_str]
            / float(total_value_df[candidate_id_str].iloc[0])
        )
        reference_equity_ser = (
            total_value_df[reference_id_str]
            / float(total_value_df[reference_id_str].iloc[0])
        )
        axis_obj.plot(candidate_equity_ser.index, candidate_equity_ser, label=candidate_id_str)
        axis_obj.plot(
            reference_equity_ser.index,
            reference_equity_ser,
            linestyle="--",
            label=reference_id_str,
        )
        axis_obj.set_yscale("log")
        axis_obj.set_title(str(candidate_spec_dict["objective_str"]).replace("_", " ").title())
        axis_obj.grid(alpha=0.25)
        axis_obj.legend(fontsize=8)
    figure_obj.suptitle("Clean-sheet candidates versus frozen references")
    figure_obj.tight_layout()
    figure_obj.savefig(chart_dir_path / "candidate_vs_reference_equity.png", dpi=160)
    plt.close(figure_obj)

    plot_gate_df = gate_df.set_index("candidate_id_str")
    figure_obj, axis_arr = plt.subplots(2, 2, figsize=(13, 8))
    delta_column_tuple = (
        ("cagr_delta_vs_reference_float", "CAGR delta", 100.0),
        ("sharpe_delta_vs_reference_float", "Sharpe delta", 1.0),
        (
            "max_drawdown_improvement_vs_reference_float",
            "MaxDD improvement",
            100.0,
        ),
        (
            "es5_reduction_vs_reference_fraction_float",
            "ES5 reduction",
            100.0,
        ),
    )
    color_list = [
        "#2a9d8f" if bool(value_obj) else "#e76f51"
        for value_obj in plot_gate_df["economic_gate_bool"]
    ]
    for axis_obj, (column_str, title_str, scale_float) in zip(
        axis_arr.ravel(),
        delta_column_tuple,
    ):
        value_ser = plot_gate_df[column_str].astype(float) * scale_float
        axis_obj.bar(value_ser.index, value_ser.values, color=color_list)
        axis_obj.axhline(0.0, color="black", linewidth=0.8)
        axis_obj.set_title(title_str)
        axis_obj.tick_params(axis="x", rotation=20)
        axis_obj.grid(axis="y", alpha=0.25)
    figure_obj.tight_layout()
    figure_obj.savefig(chart_dir_path / "candidate_gate_deltas.png", dpi=160)
    plt.close(figure_obj)


def format_pct_str(value_obj: Any, digit_int: int = 2) -> str:
    return f"{float(value_obj) * 100.0:.{digit_int}f}%"


def write_hebrew_report(
    headline_metric_df: pd.DataFrame,
    gate_df: pd.DataFrame,
    source_run_summary_df: pd.DataFrame,
    spec_dict: dict[str, Any],
    output_dir_path: Path,
) -> Path:
    portfolio_contract_dict = spec_dict["portfolio_contract"]
    capital_anchor_date_str = str(portfolio_contract_dict["capital_anchor_date_str"])
    execution_start_date_str = str(
        portfolio_contract_dict["effective_execution_start_date_str"]
    )
    end_date_str = str(portfolio_contract_dict["end_date_str"])
    headline_by_id_dict = headline_metric_df.set_index("product_id_str").to_dict(
        orient="index"
    )
    metric_line_list = [
        "| פורטפוליו | מטרה | CAGR | Sharpe | MaxDD | ES5 | Beta |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for product_id_str in [
        *spec_dict["reference_portfolios"],
        *spec_dict["candidate_portfolios"],
    ]:
        metric_dict = headline_by_id_dict[product_id_str]
        metric_line_list.append(
            "| {product} | {objective} | {cagr} | {sharpe:.3f} | {maxdd} | "
            "{es5} | {beta:.3f} |".format(
                product=product_id_str,
                objective=metric_dict["objective_str"],
                cagr=format_pct_str(metric_dict["cagr_float"]),
                sharpe=float(metric_dict["sharpe_float"]),
                maxdd=format_pct_str(metric_dict["max_drawdown_float"]),
                es5=format_pct_str(metric_dict["es5_loss_float"]),
                beta=float(metric_dict["market_beta_float"]),
            )
        )
    gate_line_list = [
        "| מועמד | ייחוס | כלכלי | תתי תקופות | סטטיסטי | מימון | החלטה |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    construction_line_list = [
        "| פורטפוליו | מטרה | תקציבי sleeve קפואים |",
        "|---|---|---|",
    ]
    for product_id_str, product_spec_dict in all_product_spec_dict(spec_dict).items():
        sleeve_budget_str = ", ".join(
            f"{strategy_alias_str} {strategy_weight_float:.1%}"
            for strategy_alias_str, strategy_weight_float in product_spec_dict[
                "weight_by_strategy_alias"
            ].items()
        )
        construction_line_list.append(
            f"| {product_id_str} | {product_spec_dict['objective_str']} | "
            f"{sleeve_budget_str} |"
        )
    for _, gate_row_ser in gate_df.iterrows():
        gate_line_list.append(
            "| {candidate} | {reference} | {economic} | {subperiod} | {stats} | "
            "{financing} | {decision} |".format(
                candidate=gate_row_ser["candidate_id_str"],
                reference=gate_row_ser["reference_id_str"],
                economic="עבר" if bool(gate_row_ser["economic_gate_bool"]) else "נכשל",
                subperiod="עבר" if bool(gate_row_ser["subperiod_gate_bool"]) else "נכשל",
                stats="עבר" if bool(gate_row_ser["statistical_gate_bool"]) else "נכשל",
                financing=(
                    "עבר"
                    if bool(gate_row_ser["unmodeled_financing_gate_bool"])
                    else "נכשל"
                ),
                decision=gate_row_ser["decision_str"],
            )
        )
    passing_candidate_list = gate_df.loc[
        gate_df["non_capacity_gate_bool"], "candidate_id_str"
    ].astype(str).tolist()
    verdict_str = (
        "המועמדים שעברו את כל שערי שלב 1 הם: " + ", ".join(passing_candidate_list)
        if passing_candidate_list
        else "אף מועמד לא עבר את כל שערי שלב 1. אין לשנות משקלים או לכוונן את המבנים על סמך התוצאות."
    )
    selected_strategy_count_int = len(
        spec_dict["registry_contract"]["selected_strategy_by_alias"]
    )
    source_count_int = len(source_run_summary_df)
    report_str = f"""# מחקר Portfolio Foundry — פורטפוליואים נקיים מ־WIRED ו־PM_READY

## סיכום

{verdict_str}

זהו מחקר השוואתי בלבד. גם מעבר מלא של שלב 1 מאפשר לכל היותר מעבר למחקר Capacity נפרד ול־forward shadow. אין כאן אישור הקצאה, PAPER או LIVE.

## השאלה והגישה

הרישום המלא כלל 9 אסטרטגיות WIRED ו־13 אסטרטגיות PM_READY. הן קובצו לפי מנגנון כלכלי, תזמון ונכסים. נבחרו {selected_strategy_count_int} נציגים ונפסלו מראש וריאציות כפולות. המשקלים נקבעו כתקציבי סליב לפני צפייה בתוצאות; לא בוצעו Markowitz, HRP, risk parity או חיפוש משקלים.

כל Pod קיבל הון אמיתי מתוך 1,000,000 דולר, רץ מחדש במניות שלמות ובעלויות המקוריות שלו, והמשיך להתרכב באופן עצמאי. אין outer rebalance. הפורטפוליו הוא סכום חשבונות ה־Pod ולא קיצור דרך של תשואות מאוזנות מדי יום.

## מבנה מדויק ותזמון

{chr(10).join(construction_line_list)}

שווי הפורטפוליו בכל יום הוא סכום שווי חשבונות ה־Pod העצמאיים:

$$
V_{{portfolio,t}} = \\sum_{{k=1}}^{{K}} V_{{k,t}}
$$

התשואה היומית נגזרת רק לאחר הסכימה:

$$
r_{{portfolio,t}} = \\frac{{V_{{portfolio,t}}}}{{V_{{portfolio,t-1}}}} - 1
$$

כל אסטרטגיה שמרה על כללי ההחלטה והביצוע המקוריים שלה. מנוע המחקר מעביר פקודה שנוצרה לאחר המידע הזמין ב־T לפתיחה הזמינה הבאה; סכימת ה־Pods אינה יוצרת עסקה או איזון מחדש נוסף.

```text
[נתונים זמינים עד T] -> [אות native של sleeve]
                           |
                           | *** CRITICAL *** אין מידע אחרי T
                           v
                    [פקודה ל־open הבא]
                           |
                           v
                 [חשבון Pod מתרכב עצמאית]
                           |
                           v
                 [סכימת כל חשבונות ה־Pod]
```

## התוצאות

{chr(10).join(metric_line_list)}

![עקומות מועמד מול ייחוס](charts/candidate_vs_reference_equity.png)

## שערי ההחלטה

{chr(10).join(gate_line_list)}

![שינוי במדדי השערים](charts/candidate_gate_deltas.png)

## חוזה המחקר

- חלון משותף: {capital_anchor_date_str} כעוגן מזומן, ביצוע ראשון ב־{execution_start_date_str}, ועד {end_date_str}.
- {source_count_int} צירופי אסטרטגיה והון הורצו פעם אחת ונשמרו עם קוד, עלויות ו־lineage.
- כל המוצרים השתמשו באותו intersection מדויק. לא בוצעו fill או forward-fill.
- בוצעו שלושה שלישים כרונולוגיים שווי תצפיות, משברי שוק ו־stationary bootstrap מסונכרן עם 10,000 חזרות ובלוקים 21, 63 ו־126.
- ה־p-values תוקנו תחילה ב־Bonferroni למשפחה מצטברת של 64 ניסויים: 60 ניסויי עבר ועוד ארבעת המועמדים הנוכחיים. לאחר מכן הוחל Holm על ארבעת המועמדים, כחסם שמרני נוסף.
- אין טענה ל־holdout היסטורי נקי: האסטרטגיות והיסטוריית 2012–2026 נצפו במחקרים קודמים.

## מלכודות וסיכונים

- משקל נמוך או קורלציה נמוכה אינם מספיקים. שערי המוצר דורשים יחד תשואה, Sharpe, drawdown, ES ובטא מול נקודת הייחוס המתאימה.
- מזומן שלילי נבדק בכל מקורות המועמד והייחוס. מימון שלילי אינו ממודל ולכן הוא חוסם את ההשוואה כולה.
- Capacity, partial fills, TCA, borrow משתנה ו־corporate-action replay מדויק אינם מוכחים בשלב זה.
- CORE5 משתמש בהנחת השאלת DBC קבועה; Tactical Fixed Income משתמש בנתוני FRED current-vintage; יציאות lifecycle עשויות להשתמש ב־close האחרון הזמין.
- משברי 2008 ו־2011 קודמים לעוגן המשותף ולכן מסומנים N/A; בדיקות המשבר הזמינות מתחילות ב־2015 והן diagnostic בלבד.
- פורטפוליו low-touch אינו מכיל אסטרטגיות signal יומיות, אך CORE5 ו־Trinity יכולים לפעול בשינוי מצב או רצועת תנודתיות ולא רק בסוף חודש.

## ארטיפקטים

- [המפרט הקפוא](research_spec_frozen.yaml)
- [מדדי הכותרת](headline_metrics.csv)
- [שערי המועמדים](candidate_gates.csv)
- [Bootstrap](bootstrap_summary.csv)
- [תתי תקופות](subperiod_metrics.csv)
- [משברים](crisis_metrics.csv)
- [קורלציות מקורות](candidate_source_correlations.csv)
- [חפיפת עסקאות](book_order_overlap_summary.csv)
- [Manifest](run_manifest.json)
"""
    report_path = output_dir_path / "PORTFOLIO_FOUNDRY_REPORT_HE.md"
    report_path.write_text(report_str, encoding="utf-8")
    return report_path


def write_run_manifest(
    output_dir_path: Path,
    spec_dict: dict[str, Any],
    global_index_sha256_str: str,
    norgate_start_dict: dict[str, Any],
    norgate_source_end_dict: dict[str, Any],
    norgate_final_dict: dict[str, Any],
) -> Path:
    manifest_path = output_dir_path / "run_manifest.json"
    artifact_row_list: list[dict[str, Any]] = []
    for artifact_path in sorted(output_dir_path.rglob("*")):
        if not artifact_path.is_file() or artifact_path == manifest_path:
            continue
        artifact_row_list.append(
            {
                "relative_path_str": artifact_path.relative_to(
                    output_dir_path
                ).as_posix(),
                "size_byte_int": int(artifact_path.stat().st_size),
                "sha256_str": ladder_runner.sha256_file_str(artifact_path),
            }
        )
    manifest_dict = {
        "study_id_str": spec_dict["study_id_str"],
        "study_version_str": spec_dict["study_version_str"],
        "completed_at_utc_str": ladder_runner.utc_now_str(),
        "research_authority_str": spec_dict["research_authority_str"],
        "spec_sha256_str": ladder_runner.sha256_file_str(
            output_dir_path / "research_spec_frozen.yaml"
        ),
        "runner_sha256_str": ladder_runner.sha256_file_str(Path(__file__).resolve()),
        "registry_sha256_str": ladder_runner.sha256_file_str(
            REPO_ROOT_PATH / "alpha" / "strategy_registry.py"
        ),
        "global_index_sha256_str": global_index_sha256_str,
        "norgate_start_dict": norgate_start_dict,
        "norgate_source_end_dict": norgate_source_end_dict,
        "norgate_final_dict": norgate_final_dict,
        "git_context_dict": ladder_runner.git_context_dict(),
        "statistical_contract_dict": spec_dict["statistical_contract"],
        "artifact_count_int": len(artifact_row_list),
        "artifact_row_list": artifact_row_list,
    }
    ladder_runner.write_json(manifest_path, manifest_dict)
    return manifest_path


def prepare_output_dir(
    output_dir_path: Path,
    spec_path: Path,
) -> None:
    if output_dir_path.exists() and any(output_dir_path.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {output_dir_path}. Use a fresh directory."
        )
    output_dir_path.mkdir(parents=True, exist_ok=True)
    frozen_spec_path = output_dir_path / "research_spec_frozen.yaml"
    if frozen_spec_path.exists():
        if frozen_spec_path.read_bytes() != spec_path.read_bytes():
            raise RuntimeError("Output directory contains a different frozen spec.")
    else:
        shutil.copyfile(spec_path, frozen_spec_path)


def run_study(
    spec_path: Path = DEFAULT_SPEC_PATH,
    output_dir_path: Path = DEFAULT_OUTPUT_DIR_PATH,
    *,
    resume_bool: bool = False,
) -> Path:
    if resume_bool:
        raise ValueError(
            "Frozen Portfolio Foundry runs forbid --resume so one study cannot "
            "mix code or Norgate database vintages."
        )
    spec_dict = load_spec_dict(spec_path)
    expanded_spec_dict, source_id_by_product_alias_dict = expanded_source_spec_dict(
        spec_dict
    )
    prepare_output_dir(output_dir_path, spec_path)
    execution_ledger_path = output_dir_path / "experiment_ledger.jsonl"
    runner_start_sha256_str = ladder_runner.sha256_file_str(Path(__file__).resolve())
    registry_path = REPO_ROOT_PATH / "alpha" / "strategy_registry.py"
    registry_start_sha256_str = ladder_runner.sha256_file_str(registry_path)
    ladder_runner.write_json(
        output_dir_path / "expanded_source_contract.json",
        {
            "source_run_by_id_dict": expanded_spec_dict["source_runs"],
            "source_id_by_product_alias_dict": source_id_by_product_alias_dict,
            "source_count_int": len(expanded_spec_dict["source_runs"]),
        },
    )
    ladder_runner.append_jsonl(
        execution_ledger_path,
        {
            "event_str": "study_started",
            "recorded_at_utc_str": ladder_runner.utc_now_str(),
            "study_id_str": spec_dict["study_id_str"],
            "spec_sha256_str": ladder_runner.sha256_file_str(spec_path),
            "runner_sha256_str": runner_start_sha256_str,
            "registry_sha256_str": registry_start_sha256_str,
            "source_count_int": len(expanded_spec_dict["source_runs"]),
        },
    )

    norgate_start_dict = ladder_runner.norgate_database_vintage_dict()
    ladder_runner.write_json(
        output_dir_path / "norgate_database_vintage_start.json",
        norgate_start_dict,
    )
    source_run_summary_df = ladder_runner.execute_source_runs(
        expanded_spec_dict,
        output_dir_path,
        resume_bool=False,
    )
    norgate_source_end_dict = ladder_runner.norgate_database_vintage_dict()
    ladder_runner.write_json(
        output_dir_path / "norgate_database_vintage_source_end.json",
        norgate_source_end_dict,
    )
    if norgate_source_end_dict != norgate_start_dict:
        raise RuntimeError("Norgate database vintage changed during source execution.")
    source_lineage_gate_bool = validate_source_lineage_bool(
        expanded_spec_dict,
        source_run_summary_df,
        output_dir_path,
    )

    (
        total_value_df,
        portfolio_value_df,
        cash_df,
        benchmark_total_value_ser,
        benchmark_return_ser,
        global_index_sha256_str,
    ) = build_global_product_frames(
        expanded_spec_dict,
        source_id_by_product_alias_dict,
        output_dir_path,
    )
    norgate_final_dict = ladder_runner.norgate_database_vintage_dict()
    ladder_runner.write_json(
        output_dir_path / "norgate_database_vintage_final.json",
        norgate_final_dict,
    )
    if norgate_final_dict != norgate_start_dict:
        raise RuntimeError("Norgate database vintage changed before analysis freeze.")
    if ladder_runner.sha256_file_str(Path(__file__).resolve()) != runner_start_sha256_str:
        raise RuntimeError("Portfolio Foundry runner changed during execution.")
    if ladder_runner.sha256_file_str(registry_path) != registry_start_sha256_str:
        raise RuntimeError("Strategy registry changed during execution.")

    return_df = total_value_df.pct_change(fill_method=None)
    return_df.iloc[0] = 0.0
    path_output_df = pd.concat(
        [
            total_value_df.add_suffix("__total_value_float"),
            portfolio_value_df.add_suffix("__portfolio_value_float"),
            cash_df.add_suffix("__cash_float"),
            benchmark_total_value_ser,
        ],
        axis=1,
    )
    ladder_runner.write_csv_gzip(
        path_output_df,
        output_dir_path / "global_product_paths.csv.gz",
        index_bool=True,
        index_label_str="date",
    )
    return_output_df = pd.concat(
        [return_df, benchmark_return_ser.rename("SPXTR")],
        axis=1,
    )
    ladder_runner.write_csv_gzip(
        return_output_df,
        output_dir_path / "global_product_returns.csv.gz",
        index_bool=True,
        index_label_str="date",
    )

    headline_metric_df = calculate_headline_metric_df(
        total_value_df,
        cash_df,
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
    source_correlation_df = calculate_source_correlation_df(
        source_id_by_product_alias_dict,
        output_dir_path,
        pd.DatetimeIndex(total_value_df.index),
        set(spec_dict["candidate_portfolios"]),
    )
    source_correlation_df.to_csv(
        output_dir_path / "candidate_source_correlations.csv",
        index=False,
        float_format="%.12g",
        lineterminator="\n",
    )
    transaction_overlap_df = calculate_transaction_overlap_df(
        source_id_by_product_alias_dict,
        output_dir_path,
    )
    transaction_overlap_df.to_csv(
        output_dir_path / "book_order_overlap_summary.csv",
        index=False,
        float_format="%.12g",
        lineterminator="\n",
    )
    bootstrap_summary_df, holm_df = calculate_bootstrap_evidence(
        return_df,
        headline_metric_df,
        spec_dict,
    )
    bootstrap_summary_df.to_csv(
        output_dir_path / "bootstrap_summary.csv",
        index=False,
        float_format="%.12g",
        lineterminator="\n",
    )
    holm_df.to_csv(
        output_dir_path / "holm_results.csv",
        index=False,
        float_format="%.12g",
        lineterminator="\n",
    )
    gate_df = evaluate_candidate_gate_df(
        headline_metric_df,
        subperiod_metric_df,
        bootstrap_summary_df,
        holm_df,
        source_run_summary_df,
        source_id_by_product_alias_dict,
        spec_dict,
        source_lineage_gate_bool,
    )
    gate_df.to_csv(
        output_dir_path / "candidate_gates.csv",
        index=False,
        float_format="%.12g",
        lineterminator="\n",
    )
    create_charts(total_value_df, gate_df, spec_dict, output_dir_path)
    report_path = write_hebrew_report(
        headline_metric_df,
        gate_df,
        source_run_summary_df,
        spec_dict,
        output_dir_path,
    )
    validate_source_lineage_bool(
        expanded_spec_dict,
        source_run_summary_df,
        output_dir_path,
    )
    if ladder_runner.sha256_file_str(Path(__file__).resolve()) != runner_start_sha256_str:
        raise RuntimeError("Portfolio Foundry runner changed before completion.")
    if ladder_runner.sha256_file_str(registry_path) != registry_start_sha256_str:
        raise RuntimeError("Strategy registry changed before completion.")
    ladder_runner.append_jsonl(
        execution_ledger_path,
        {
            "event_str": "study_completed",
            "recorded_at_utc_str": ladder_runner.utc_now_str(),
            "candidate_decision_by_id_dict": {
                str(row_ser["candidate_id_str"]): str(row_ser["decision_str"])
                for _, row_ser in gate_df.iterrows()
            },
            "passing_non_capacity_candidate_id_list": gate_df.loc[
                gate_df["non_capacity_gate_bool"], "candidate_id_str"
            ].astype(str).tolist(),
            "passing_promotion_candidate_id_list": [],
        },
    )
    write_run_manifest(
        output_dir_path,
        spec_dict,
        global_index_sha256_str,
        norgate_start_dict,
        norgate_source_end_dict,
        norgate_final_dict,
    )
    return report_path


def parse_args(arg_list: Iterable[str] | None = None) -> argparse.Namespace:
    parser_obj = argparse.ArgumentParser(
        description="Run the frozen clean-sheet Portfolio Foundry study."
    )
    parser_obj.add_argument(
        "--spec",
        type=Path,
        default=DEFAULT_SPEC_PATH,
    )
    parser_obj.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR_PATH,
    )
    parser_obj.add_argument(
        "--resume",
        action="store_true",
        help="Rejected for frozen runs; use a fresh output directory.",
    )
    return parser_obj.parse_args(list(arg_list) if arg_list is not None else None)


def main(arg_list: Iterable[str] | None = None) -> int:
    args_obj = parse_args(arg_list)
    report_path = run_study(
        spec_path=args_obj.spec.resolve(),
        output_dir_path=args_obj.output_dir.resolve(),
        resume_bool=bool(args_obj.resume),
    )
    print(report_path, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
