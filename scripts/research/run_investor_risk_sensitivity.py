"""
Run a predeclared investor-risk sensitivity sweep on one saved portfolio path.

This is research/report-only. It does not rebuild the portfolio, select new
weights, or change any signal, sizing, execution, or live behavior.

The three evidence layers remain separate:

1. observed history;
2. stationary-bootstrap dependent-block resampling diagnostics;
3. explicit hypothetical drag and volatility overlays.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


REPO_ROOT_PATH = Path(__file__).resolve().parents[2]
if str(REPO_ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_PATH))

from alpha.engine.crisis import CRISIS_PERIODS_LIST
from alpha.engine.portfolio import Portfolio
from alpha.engine.report import build_research_output_path
from alpha.engine.risk_analysis import (
    DEFAULT_RANDOM_SEED_INT,
    DEFAULT_SIMULATION_COUNT_INT,
    TRADING_DAYS_PER_YEAR_INT,
    _path_horizon_metric_dict,
    build_horizon_probability_df,
    build_investor_scenario_df,
    build_observed_calendar_month_df,
    compute_path_metric_dict,
    extract_realized_return_ser,
    stationary_bootstrap_index_mat,
)
from strategies.run_portfolio_risk_analysis import (
    _build_portfolio_analysis_context_dict,
    _read_json_dict,
    _sha256_str,
    _validate_portfolio_artifact,
)


INVESTOR_RISK_SENSITIVITY_TYPE_STR = "investor_risk_sensitivity"
INVESTOR_RISK_NUMERICAL_STABILITY_TYPE_STR = "investor_risk_numerical_stability"
INVESTOR_RISK_SENSITIVITY_SCHEMA_VERSION_INT = 1
PRIMARY_BLOCK_LENGTH_INT = 21
BLOCK_SENSITIVITY_TUPLE = (5, 10, 21, 63)
INVESTOR_HORIZON_YEAR_TUPLE = (1, 3, 5)
STABILITY_ABSOLUTE_TOLERANCE_FLOAT = 0.01
MONTHLY_STABILITY_SIMULATION_COUNT_INT = 100000
MONTHLY_STABILITY_ABSOLUTE_TOLERANCE_FLOAT = 0.005
STABILITY_METRIC_TUPLE = (
    "terminal_return_p05_float",
    "terminal_loss_probability_float",
    "max_drawdown_p05_float",
    "underwater_ge_12m_probability_float",
    "terminal_underwater_probability_float",
)
OBSERVED_CRISIS_WINDOW_COLUMN_TUPLE = (
    "crisis_name_str",
    "configured_start_date_str",
    "configured_end_date_str",
    "effective_start_date_str",
    "effective_end_date_str",
    "trading_day_count_int",
    "observed_period_return_float",
    "observed_max_drawdown_float",
    "observed_longest_underwater_days_float",
    "observed_worst_day_return_float",
    "evidence_kind_str",
)
OBSERVED_CALENDAR_YEAR_COLUMN_TUPLE = (
    "calendar_year_int",
    "trading_day_count_int",
    "calendar_year_return_float",
    "partial_year_bool",
)
SEED_STABILITY_COLUMN_TUPLE = (
    "scenario_key_str",
    "metric_name_str",
    "primary_value_float",
    "alternate_value_float",
    "difference_float",
    "absolute_difference_float",
    "absolute_tolerance_float",
    "within_tolerance_bool",
)


def apply_additional_annual_drag_ser(
    realized_return_ser: pd.Series,
    annual_drag_float: float,
) -> pd.Series:
    """Apply a transparent additional annual drag to each realized day."""
    annual_drag_float = float(annual_drag_float)
    if not 0.0 <= annual_drag_float < 1.0:
        raise ValueError("annual_drag_float must be in [0, 1).")
    daily_drag_multiplier_float = (1.0 - annual_drag_float) ** (
        1.0 / float(TRADING_DAYS_PER_YEAR_INT)
    )
    # *** CRITICAL*** report-only hypothetical friction overlay:
    # (1 + r_stressed,t) = (1 + r_t) * (1 - annual_drag)^(1/252).
    # This is additional drag, not a replacement execution-cost model.
    stressed_return_ser = (
        (1.0 + realized_return_ser.astype(float)) * daily_drag_multiplier_float
        - 1.0
    )
    stressed_return_ser.name = "stressed_return_float"
    return stressed_return_ser


def amplify_daily_volatility_ser(
    realized_return_ser: pd.Series,
    volatility_multiplier_float: float,
) -> pd.Series:
    """Amplify deviations around the realized arithmetic daily mean."""
    volatility_multiplier_float = float(volatility_multiplier_float)
    if volatility_multiplier_float <= 0.0:
        raise ValueError("volatility_multiplier_float must be positive.")
    mean_return_float = float(realized_return_ser.astype(float).mean())
    # *** CRITICAL*** report-only severity overlay: this preserves the sample's
    # arithmetic daily mean and scales observed deviations. It is not a new
    # regime model, correlation shock, or trading input.
    stressed_return_ser = mean_return_float + volatility_multiplier_float * (
        realized_return_ser.astype(float) - mean_return_float
    )
    if bool((stressed_return_ser <= -1.0).any()):
        raise ValueError("Volatility amplification produced a daily return <= -100%.")
    stressed_return_ser.name = "stressed_return_float"
    return stressed_return_ser


def build_observed_calendar_year_df(realized_return_ser: pd.Series) -> pd.DataFrame:
    clean_return_ser = (
        realized_return_ser.astype(float)
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .sort_index()
    )
    if not isinstance(clean_return_ser.index, pd.DatetimeIndex):
        raise ValueError("realized_return_ser must use a DatetimeIndex.")
    row_list: list[dict[str, object]] = []
    # *** CRITICAL*** report-only calendar aggregation: each row uses only
    # returns stamped inside that calendar year. Boundary years may be partial.
    for calendar_year_int, calendar_year_return_ser in clean_return_ser.groupby(
        clean_return_ser.index.year
    ):
        row_list.append(
            {
                "calendar_year_int": int(calendar_year_int),
                "trading_day_count_int": int(len(calendar_year_return_ser)),
                "calendar_year_return_float": float(
                    np.prod(1.0 + calendar_year_return_ser.to_numpy(dtype=float))
                    - 1.0
                ),
                "partial_year_bool": bool(
                    len(calendar_year_return_ser) < TRADING_DAYS_PER_YEAR_INT - 5
                ),
            }
        )
    return pd.DataFrame(row_list, columns=OBSERVED_CALENDAR_YEAR_COLUMN_TUPLE)


def build_observed_crisis_window_df(realized_return_ser: pd.Series) -> pd.DataFrame:
    clean_return_ser = realized_return_ser.astype(float).sort_index()
    sample_start_ts = pd.Timestamp(clean_return_ser.index.min())
    sample_end_ts = pd.Timestamp(clean_return_ser.index.max())
    row_list: list[dict[str, object]] = []
    for crisis_period_obj in CRISIS_PERIODS_LIST:
        configured_start_ts = pd.Timestamp(crisis_period_obj.start_date_str)
        configured_end_ts = pd.Timestamp(crisis_period_obj.end_date_str)
        if configured_start_ts < sample_start_ts or configured_end_ts > sample_end_ts:
            continue
        crisis_return_ser = clean_return_ser.loc[configured_start_ts:configured_end_ts]
        if len(crisis_return_ser) == 0:
            continue
        crisis_metric_dict = _path_horizon_metric_dict(
            crisis_return_ser.to_numpy(dtype=float),
            int(len(crisis_return_ser)),
        )
        row_list.append(
            {
                "crisis_name_str": crisis_period_obj.crisis_name_str,
                "configured_start_date_str": crisis_period_obj.start_date_str,
                "configured_end_date_str": crisis_period_obj.end_date_str,
                "effective_start_date_str": crisis_return_ser.index.min().date().isoformat(),
                "effective_end_date_str": crisis_return_ser.index.max().date().isoformat(),
                "trading_day_count_int": int(len(crisis_return_ser)),
                "observed_period_return_float": float(
                    np.prod(1.0 + crisis_return_ser.to_numpy(dtype=float)) - 1.0
                ),
                "observed_max_drawdown_float": crisis_metric_dict[
                    "max_drawdown_float"
                ],
                "observed_longest_underwater_days_float": crisis_metric_dict[
                    "longest_underwater_days_float"
                ],
                "observed_worst_day_return_float": float(crisis_return_ser.min()),
                "evidence_kind_str": "observed_slice_not_fresh_crisis_replay",
            }
        )
    return pd.DataFrame(row_list, columns=OBSERVED_CRISIS_WINDOW_COLUMN_TUPLE)


def build_rolling_12m_return_df(realized_return_ser: pd.Series) -> pd.DataFrame:
    clean_return_ser = realized_return_ser.astype(float).sort_index()
    # *** CRITICAL*** trailing-only report metric: the value stamped at T uses
    # the 252 realized returns ending at T and no future observation. Windows
    # overlap heavily and must not be treated as independent probabilities.
    rolling_12m_return_ser = (
        (1.0 + clean_return_ser)
        .rolling(TRADING_DAYS_PER_YEAR_INT)
        .apply(np.prod, raw=True)
        .sub(1.0)
        .dropna()
    )
    return pd.DataFrame(
        {
            "window_end_date_str": [
                timestamp_obj.date().isoformat()
                for timestamp_obj in rolling_12m_return_ser.index
            ],
            "rolling_12m_return_float": rolling_12m_return_ser.to_numpy(dtype=float),
            "overlapping_observation_bool": True,
        }
    )


def _build_sensitivity_variant_list(
    realized_return_ser: pd.Series,
    random_seed_int: int,
) -> list[dict[str, object]]:
    variant_list: list[dict[str, object]] = []
    for block_length_int in BLOCK_SENSITIVITY_TUPLE:
        variant_list.append(
            {
                "variant_key_str": f"block_{block_length_int}",
                "variant_category_str": "bootstrap_block_sensitivity",
                "variant_label_str": f"Full sample, mean block {block_length_int}d",
                "transformation_detail_str": "none",
                "mean_block_length_int": int(block_length_int),
                "random_seed_int": int(random_seed_int),
                "return_ser": realized_return_ser,
            }
        )

    pre_2020_return_ser = realized_return_ser.loc[:"2019-12-31"]
    post_2020_return_ser = realized_return_ser.loc["2020-01-01":]
    for variant_key_str, variant_label_str, regime_return_ser in (
        (
            "regime_pre_2020",
            "Observed-return regime: sample start through 2019",
            pre_2020_return_ser,
        ),
        (
            "regime_2020_onward",
            "Observed-return regime: 2020 onward",
            post_2020_return_ser,
        ),
    ):
        variant_list.append(
            {
                "variant_key_str": variant_key_str,
                "variant_category_str": "historical_regime_conditioning",
                "variant_label_str": variant_label_str,
                "transformation_detail_str": "subset_of_realized_return_history",
                "mean_block_length_int": PRIMARY_BLOCK_LENGTH_INT,
                "random_seed_int": int(random_seed_int),
                "return_ser": regime_return_ser,
            }
        )

    for annual_drag_bps_int in (200, 400):
        annual_drag_float = float(annual_drag_bps_int) / 10000.0
        variant_list.append(
            {
                "variant_key_str": f"additional_drag_{annual_drag_bps_int}bps",
                "variant_category_str": "hypothetical_additional_drag",
                "variant_label_str": f"Full sample plus {annual_drag_bps_int} bps annual drag",
                "transformation_detail_str": (
                    "(1+r_t)*(1-annual_drag)^(1/252)-1"
                ),
                "mean_block_length_int": PRIMARY_BLOCK_LENGTH_INT,
                "random_seed_int": int(random_seed_int),
                "return_ser": apply_additional_annual_drag_ser(
                    realized_return_ser,
                    annual_drag_float,
                ),
            }
        )

    for volatility_multiplier_float in (1.25, 1.50):
        multiplier_label_str = str(volatility_multiplier_float).replace(".", "p")
        variant_list.append(
            {
                "variant_key_str": f"volatility_{multiplier_label_str}x",
                "variant_category_str": "hypothetical_volatility_severity",
                "variant_label_str": (
                    f"Full sample deviations amplified {volatility_multiplier_float:.2f}x"
                ),
                "transformation_detail_str": (
                    "mean_daily_return + multiplier*(r_t-mean_daily_return)"
                ),
                "mean_block_length_int": PRIMARY_BLOCK_LENGTH_INT,
                "random_seed_int": int(random_seed_int),
                "return_ser": amplify_daily_volatility_ser(
                    realized_return_ser,
                    volatility_multiplier_float,
                ),
            }
        )
    for variant_dict in variant_list:
        variant_return_ser = variant_dict["return_ser"]
        variant_dict["variant_status_str"] = (
            "available"
            if isinstance(variant_return_ser, pd.Series) and len(variant_return_ser) > 0
            else "skipped_empty_return_history"
        )
    return variant_list


def _run_sensitivity_variant_df(
    variant_dict: dict[str, object],
    simulation_count_int: int,
) -> pd.DataFrame:
    variant_return_ser = variant_dict["return_ser"]
    if not isinstance(variant_return_ser, pd.Series):
        raise TypeError("Sensitivity variant return_ser must be a pandas Series.")
    mean_block_length_int = int(variant_dict["mean_block_length_int"])
    random_seed_int = int(variant_dict["random_seed_int"])
    if len(variant_return_ser) == 0:
        return pd.DataFrame(
            [
                {
                    "variant_key_str": variant_dict["variant_key_str"],
                    "variant_category_str": variant_dict["variant_category_str"],
                    "variant_label_str": variant_dict["variant_label_str"],
                    "transformation_detail_str": variant_dict[
                        "transformation_detail_str"
                    ],
                    "variant_status_str": "skipped_empty_return_history",
                    "mean_block_length_int": mean_block_length_int,
                    "random_seed_int": random_seed_int,
                    "return_count_int": 0,
                    "return_start_date_str": None,
                    "return_end_date_str": None,
                    "scenario_key_str": scenario_key_str,
                    "scenario_label_str": scenario_label_str,
                    "evidence_kind_str": "bootstrap_implied",
                    "simulation_path_count_int": 0,
                    "scenario_status_str": "skipped_empty_return_history",
                }
                for scenario_key_str, scenario_label_str in (
                    ("modeled_21d", "Modeled 21-trading-day period"),
                    ("modeled_1y", "Modeled 1-year horizon"),
                    ("modeled_3y", "Modeled 3-year horizon"),
                    ("modeled_5y", "Modeled 5-year horizon"),
                )
            ]
        )
    horizon_probability_df = build_horizon_probability_df(
        realized_return_ser=variant_return_ser,
        mean_block_length_int=mean_block_length_int,
        simulation_count_int=int(simulation_count_int),
        random_seed_int=random_seed_int,
        horizon_year_tuple=INVESTOR_HORIZON_YEAR_TUPLE,
    )
    investor_scenario_df = build_investor_scenario_df(
        realized_return_ser=variant_return_ser,
        horizon_probability_df=horizon_probability_df,
        mean_block_length_int=mean_block_length_int,
        simulation_count_int=int(simulation_count_int),
        random_seed_int=random_seed_int,
        investor_horizon_year_tuple=INVESTOR_HORIZON_YEAR_TUPLE,
    )
    modeled_scenario_df = investor_scenario_df[
        investor_scenario_df["evidence_kind_str"] == "bootstrap_implied"
    ].copy()
    modeled_scenario_df["scenario_status_str"] = np.where(
        modeled_scenario_df["simulation_path_count_int"].astype("Int64").fillna(0)
        > 0,
        "available",
        "unavailable_insufficient_history",
    )
    modeled_scenario_df.insert(0, "variant_key_str", variant_dict["variant_key_str"])
    modeled_scenario_df.insert(
        1,
        "variant_category_str",
        variant_dict["variant_category_str"],
    )
    modeled_scenario_df.insert(2, "variant_label_str", variant_dict["variant_label_str"])
    modeled_scenario_df.insert(
        3,
        "transformation_detail_str",
        variant_dict["transformation_detail_str"],
    )
    modeled_scenario_df.insert(4, "variant_status_str", "available")
    modeled_scenario_df.insert(5, "mean_block_length_int", mean_block_length_int)
    modeled_scenario_df.insert(6, "random_seed_int", random_seed_int)
    modeled_scenario_df.insert(7, "return_count_int", int(len(variant_return_ser)))
    modeled_scenario_df.insert(
        8,
        "return_start_date_str",
        pd.Timestamp(variant_return_ser.index.min()).date().isoformat(),
    )
    modeled_scenario_df.insert(
        9,
        "return_end_date_str",
        pd.Timestamp(variant_return_ser.index.max()).date().isoformat(),
    )
    return modeled_scenario_df


def _build_seed_stability_df(
    primary_scenario_df: pd.DataFrame,
    alternate_seed_scenario_df: pd.DataFrame,
) -> pd.DataFrame:
    merged_df = primary_scenario_df.merge(
        alternate_seed_scenario_df,
        on="scenario_key_str",
        how="inner",
        suffixes=("_primary", "_alternate"),
    )
    row_list: list[dict[str, object]] = []
    for _, scenario_ser in merged_df.iterrows():
        for metric_name_str in STABILITY_METRIC_TUPLE:
            primary_value_obj = scenario_ser.get(f"{metric_name_str}_primary")
            alternate_value_obj = scenario_ser.get(f"{metric_name_str}_alternate")
            if pd.isna(primary_value_obj) or pd.isna(alternate_value_obj):
                continue
            difference_float = float(alternate_value_obj) - float(primary_value_obj)
            row_list.append(
                {
                    "scenario_key_str": scenario_ser["scenario_key_str"],
                    "metric_name_str": metric_name_str,
                    "primary_value_float": float(primary_value_obj),
                    "alternate_value_float": float(alternate_value_obj),
                    "difference_float": difference_float,
                    "absolute_difference_float": abs(difference_float),
                    "absolute_tolerance_float": STABILITY_ABSOLUTE_TOLERANCE_FLOAT,
                    "within_tolerance_bool": bool(
                        abs(difference_float) <= STABILITY_ABSOLUTE_TOLERANCE_FLOAT
                    ),
                }
            )
    return pd.DataFrame(row_list, columns=SEED_STABILITY_COLUMN_TUPLE)


def build_modeled_21d_terminal_summary_dict(
    realized_return_ser: pd.Series,
    *,
    simulation_count_int: int,
    random_seed_int: int,
    mean_block_length_int: int = PRIMARY_BLOCK_LENGTH_INT,
) -> dict[str, object]:
    clean_return_vec = (
        realized_return_ser.astype(float)
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .to_numpy(dtype=float)
    )
    if clean_return_vec.size < 21:
        raise ValueError("At least 21 realized returns are required.")
    # *** CRITICAL*** report-only numerical convergence check: every simulated
    # path contains 21 stationary-bootstrap draws from the realized returns and
    # has no calendar dates. It cannot feed trading or be called a forecast.
    month_index_mat = stationary_bootstrap_index_mat(
        sample_size_int=int(clean_return_vec.size),
        simulation_count_int=int(simulation_count_int),
        mean_block_length_int=int(mean_block_length_int),
        random_seed_int=int(random_seed_int),
        path_length_int=21,
    )
    month_terminal_return_vec = (
        np.prod(1.0 + clean_return_vec[month_index_mat], axis=1) - 1.0
    )
    return {
        "simulation_count_int": int(simulation_count_int),
        "random_seed_int": int(random_seed_int),
        "mean_block_length_int": int(mean_block_length_int),
        "terminal_return_p05_float": float(
            np.quantile(month_terminal_return_vec, 0.05)
        ),
        "terminal_return_p25_float": float(
            np.quantile(month_terminal_return_vec, 0.25)
        ),
        "terminal_return_p50_float": float(
            np.quantile(month_terminal_return_vec, 0.50)
        ),
        "terminal_return_p75_float": float(
            np.quantile(month_terminal_return_vec, 0.75)
        ),
        "terminal_return_p95_float": float(
            np.quantile(month_terminal_return_vec, 0.95)
        ),
        "terminal_loss_probability_float": float(
            (month_terminal_return_vec < 0.0).mean()
        ),
    }


def run_monthly_numerical_stability_followup(
    *,
    portfolio_pickle_path: Path,
    output_dir_str: str,
    simulation_count_int: int,
    random_seed_int: int,
    parent_sensitivity_dir_path: Path | None = None,
) -> Path:
    portfolio_pickle_path = portfolio_pickle_path.resolve()
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
    source_context_dict = _build_portfolio_analysis_context_dict(
        portfolio_pickle_path=portfolio_pickle_path,
        source_metadata_dict=source_metadata_dict,
        manager_metadata_dict=manager_metadata_dict,
        portfolio_obj=portfolio_obj,
    )
    realized_return_ser = extract_realized_return_ser(portfolio_obj)
    alternate_seed_int = int(random_seed_int) + 1_000_003
    primary_summary_dict = build_modeled_21d_terminal_summary_dict(
        realized_return_ser,
        simulation_count_int=simulation_count_int,
        random_seed_int=random_seed_int,
    )
    alternate_summary_dict = build_modeled_21d_terminal_summary_dict(
        realized_return_ser,
        simulation_count_int=simulation_count_int,
        random_seed_int=alternate_seed_int,
    )
    scenario_df = pd.DataFrame(
        [
            {"seed_role_str": "primary", **primary_summary_dict},
            {"seed_role_str": "alternate", **alternate_summary_dict},
        ]
    )
    comparison_row_list: list[dict[str, object]] = []
    for metric_name_str in (
        "terminal_return_p05_float",
        "terminal_return_p25_float",
        "terminal_return_p50_float",
        "terminal_return_p75_float",
        "terminal_return_p95_float",
        "terminal_loss_probability_float",
    ):
        primary_value_float = float(primary_summary_dict[metric_name_str])
        alternate_value_float = float(alternate_summary_dict[metric_name_str])
        difference_float = alternate_value_float - primary_value_float
        comparison_row_list.append(
            {
                "metric_name_str": metric_name_str,
                "primary_value_float": primary_value_float,
                "alternate_value_float": alternate_value_float,
                "difference_float": difference_float,
                "absolute_difference_float": abs(difference_float),
                "absolute_tolerance_float": (
                    MONTHLY_STABILITY_ABSOLUTE_TOLERANCE_FLOAT
                ),
                "within_tolerance_bool": bool(
                    abs(difference_float)
                    <= MONTHLY_STABILITY_ABSOLUTE_TOLERANCE_FLOAT
                ),
            }
        )
    comparison_df = pd.DataFrame(comparison_row_list)
    max_absolute_difference_float = float(
        comparison_df["absolute_difference_float"].max()
    )
    status_str = (
        "stable_within_0p5_percentage_point"
        if bool(comparison_df["within_tolerance_bool"].all())
        else "requires_more_simulations_or_review"
    )
    stability_resolved_bool = status_str.startswith("stable_")
    checked_metric_list = [
        "modeled_21d_terminal_return_quantiles",
        "modeled_21d_terminal_loss_probability",
    ]

    output_dir_path = build_research_output_path(
        output_dir_str,
        "portfolio",
        str(portfolio_obj.name),
        INVESTOR_RISK_NUMERICAL_STABILITY_TYPE_STR,
    )
    output_dir_path.mkdir(parents=True, exist_ok=True)
    scenario_df.to_csv(output_dir_path / "monthly_seed_scenarios.csv", index=False)
    comparison_df.to_csv(output_dir_path / "monthly_seed_comparison.csv", index=False)
    parent_link_dict: dict[str, object] = {}
    if parent_sensitivity_dir_path is not None:
        parent_sensitivity_dir_path = parent_sensitivity_dir_path.resolve()
        parent_summary_path = parent_sensitivity_dir_path / "summary.json"
        parent_summary_dict = _read_json_dict(parent_summary_path, required_bool=True)
        parent_source_context_dict = parent_summary_dict.get("source_context_dict", {})
        if not isinstance(parent_source_context_dict, dict) or (
            parent_source_context_dict.get("source_artifact_sha256_str")
            != source_context_dict.get("source_artifact_sha256_str")
        ):
            raise ValueError(
                "Parent sensitivity artifact does not use the same source portfolio hash."
            )
        parent_link_dict = {
            "parent_sensitivity_dir_path_str": str(parent_sensitivity_dir_path),
            "parent_summary_sha256_str": _sha256_str(parent_summary_path),
            "parent_original_seed_stability_status_str": (
                parent_summary_dict.get("seed_stability_dict", {}).get("status_str")
                if isinstance(parent_summary_dict.get("seed_stability_dict"), dict)
                else None
            ),
            "checked_metric_list": checked_metric_list,
            "numerical_convergence_resolved_bool": stability_resolved_bool,
        }
        if stability_resolved_bool:
            parent_link_dict["resolved_metric_list"] = checked_metric_list
    summary_dict = {
        "schema_version_int": INVESTOR_RISK_SENSITIVITY_SCHEMA_VERSION_INT,
        "analysis_status_str": "provisional_research_only",
        "check_type_str": "post_sweep_monthly_numerical_convergence",
        "status_str": status_str,
        "simulation_count_per_seed_int": int(simulation_count_int),
        "primary_seed_int": int(random_seed_int),
        "alternate_seed_int": alternate_seed_int,
        "absolute_tolerance_float": MONTHLY_STABILITY_ABSOLUTE_TOLERANCE_FLOAT,
        "max_absolute_difference_float": max_absolute_difference_float,
        "source_context_dict": source_context_dict,
        "parent_sensitivity_link_dict": parent_link_dict,
        "interpretation_str": (
            (
                "This resolves Monte Carlo convergence of the modeled 21-trading-day terminal-return distribution under the stated tolerance. "
                if stability_resolved_bool
                else "This did not resolve Monte Carlo convergence of the modeled 21-trading-day terminal-return distribution under the stated tolerance. "
            )
            + "It does not validate the economic model or convert it into a forecast."
        ),
    }
    _write_json(output_dir_path / "summary.json", summary_dict)
    _write_json(
        output_dir_path / "run_info.json",
        {
            "entity_type": "portfolio",
            "entity_id": str(portfolio_obj.name),
            "analysis_type": INVESTOR_RISK_NUMERICAL_STABILITY_TYPE_STR,
            "parameters": {
                "source_artifact_path": str(portfolio_pickle_path),
                "simulation_count_per_seed_int": int(simulation_count_int),
                "primary_seed_int": int(random_seed_int),
                "alternate_seed_int": alternate_seed_int,
            },
        },
    )
    _write_json(
        output_dir_path / "metadata.json",
        {
            "artifact_type": INVESTOR_RISK_NUMERICAL_STABILITY_TYPE_STR,
            "saved_at": datetime.now().isoformat(timespec="seconds"),
            "portfolio_name": str(portfolio_obj.name),
            "analysis_status": "provisional_research_only",
        },
    )
    if parent_sensitivity_dir_path is not None:
        followup_summary_path = output_dir_path / "summary.json"
        resolution_dict = {
            "resolution_status_str": status_str,
            "numerical_convergence_resolved_bool": stability_resolved_bool,
            "followup_dir_path_str": str(output_dir_path.resolve()),
            "followup_summary_sha256_str": _sha256_str(followup_summary_path),
            "simulation_count_per_seed_int": int(simulation_count_int),
            "checked_metric_list": checked_metric_list,
            "interpretation_str": (
                (
                    "This follow-up resolves numerical convergence for the modeled 21-trading-day terminal distribution only; "
                    if stability_resolved_bool
                    else "This follow-up did not resolve numerical convergence for the modeled 21-trading-day terminal distribution; "
                )
                + "it does not validate the economic model."
            ),
        }
        if stability_resolved_bool:
            resolution_dict["resolved_metric_list"] = checked_metric_list
        _write_json(
            parent_sensitivity_dir_path / "numerical_stability_followup.json",
            resolution_dict,
        )
        (parent_sensitivity_dir_path / "NUMERICAL_STABILITY_FOLLOWUP.md").write_text(
            "\n".join(
                [
                    "# Numerical Stability Follow-up",
                    "",
                    f"Status: **{status_str}**",
                    "",
                    f"Follow-up artifact: `{output_dir_path.resolve()}`",
                    "",
                    (
                        "This resolves the modeled 21-trading-day terminal-return convergence check only. "
                        if stability_resolved_bool
                        else "This does not resolve the modeled 21-trading-day terminal-return convergence check. "
                    )
                    + "It does not turn the bootstrap into a forecast or approve investor use.",
                ]
            ),
            encoding="utf-8",
        )
        parent_report_path = parent_sensitivity_dir_path / "report.md"
        if parent_report_path.exists():
            followup_heading_str = "## Numerical stability follow-up"
            parent_report_str = parent_report_path.read_text(encoding="utf-8")
            parent_report_prefix_str = parent_report_str.split(
                followup_heading_str,
                maxsplit=1,
            )[0].rstrip()
            parent_report_path.write_text(
                parent_report_prefix_str
                + "\n\n"
                + followup_heading_str
                + "\n\n"
                + f"The focused 21-trading-day convergence status is **{status_str}**. "
                + f"See `{output_dir_path.resolve()}`. This is the latest monthly numerical-convergence check; it does not approve investor use.\n",
                encoding="utf-8",
            )
    return output_dir_path


def _json_safe_obj(value_obj):
    if isinstance(value_obj, dict):
        return {
            str(key_obj): _json_safe_obj(child_value_obj)
            for key_obj, child_value_obj in value_obj.items()
        }
    if isinstance(value_obj, list | tuple):
        return [_json_safe_obj(child_value_obj) for child_value_obj in value_obj]
    if isinstance(value_obj, np.integer):
        return int(value_obj)
    if isinstance(value_obj, np.floating):
        value_float = float(value_obj)
        return value_float if np.isfinite(value_float) else None
    if isinstance(value_obj, float) and not np.isfinite(value_obj):
        return None
    if isinstance(value_obj, Path):
        return str(value_obj)
    if isinstance(value_obj, pd.Timestamp):
        return value_obj.isoformat()
    return value_obj


def _write_json(json_path: Path, data_dict: dict[str, object]) -> None:
    json_path.write_text(
        json.dumps(_json_safe_obj(data_dict), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _records_from_df(record_df: pd.DataFrame) -> list[dict[str, object]]:
    record_list: list[dict[str, object]] = []
    for _, record_ser in record_df.iterrows():
        record_dict: dict[str, object] = {}
        for field_str, value_obj in record_ser.items():
            if field_str.endswith("_int") and not pd.isna(value_obj):
                record_dict[str(field_str)] = int(value_obj)
            elif pd.isna(value_obj):
                record_dict[str(field_str)] = None
            else:
                record_dict[str(field_str)] = value_obj
        record_list.append(record_dict)
    return record_list


def _observed_month_summary_dict(
    observed_calendar_month_df: pd.DataFrame,
) -> dict[str, object]:
    month_return_ser = observed_calendar_month_df["calendar_month_return_float"].astype(float)
    return {
        "calendar_month_count_int": int(len(month_return_ser)),
        "negative_calendar_month_frequency_float": float((month_return_ser < 0.0).mean()),
        "worst_calendar_month_return_float": float(month_return_ser.min()),
        "p05_calendar_month_return_float": float(month_return_ser.quantile(0.05)),
        "p25_calendar_month_return_float": float(month_return_ser.quantile(0.25)),
        "p50_calendar_month_return_float": float(month_return_ser.quantile(0.50)),
        "p75_calendar_month_return_float": float(month_return_ser.quantile(0.75)),
        "p95_calendar_month_return_float": float(month_return_ser.quantile(0.95)),
    }


def _presentation_readiness_dict(
    source_context_dict: dict[str, object],
) -> dict[str, object]:
    blocker_list = [
        "The source is a provisional backtest, not verified live performance.",
        "The offered portfolio weights and rebalance policy are not frozen.",
        "Cost assumptions are inherited from the source artifact and were not revalidated here.",
        "Point-in-time universe, corporate-action adjustment, and exact strategy source revisions were not revalidated by this risk runner.",
        "Legal structure, fees, custody, liquidity, and investor reporting are outside this analysis.",
    ]
    if source_context_dict.get("rebalance_frequency_str") == "none":
        blocker_list.append(
            "The source portfolio does not rebalance; the name multipod_monthly must not be read as monthly rebalancing."
        )
    if float(source_context_dict.get("max_absolute_weight_drift_float") or 0.0) > 0.01:
        blocker_list.append(
            "Without rebalancing, realized sleeve weights drift materially from the configured weights."
        )
    return {
        "status_str": "blocked_for_investor_deck_numbers",
        "blocker_list": blocker_list,
    }


def _build_report_markdown_str(summary_dict: dict[str, object]) -> str:
    source_context_dict = summary_dict["source_context_dict"]
    observed_month_dict = summary_dict["observed_calendar_month_summary_dict"]
    stability_dict = summary_dict["seed_stability_dict"]
    return "\n".join(
        [
            "# Investor Risk Sensitivity",
            "",
            "**Status: provisional research only; not approved for investor deck numbers.**",
            "",
            "## Source",
            "",
            f"- Artifact: `{source_context_dict.get('source_artifact_path_str')}`",
            f"- Realized common window: {source_context_dict.get('realized_common_start_date_str')} to {source_context_dict.get('realized_common_end_date_str')}",
            f"- Configured start: {source_context_dict.get('configured_backtest_start_date_str')}",
            f"- Rebalance frequency: {source_context_dict.get('rebalance_frequency_str')}",
            "",
            "## Observed calendar months",
            "",
            f"- Count: {observed_month_dict.get('calendar_month_count_int')}",
            f"- Negative frequency: {float(observed_month_dict.get('negative_calendar_month_frequency_float')):.1%}",
            f"- p25 to p75: {float(observed_month_dict.get('p25_calendar_month_return_float')):.1%} to {float(observed_month_dict.get('p75_calendar_month_return_float')):.1%}",
            f"- p05: {float(observed_month_dict.get('p05_calendar_month_return_float')):.1%}",
            "",
            "## Method",
            "",
            "The sweep contains ten predeclared variants: four block lengths, two historical regime subsets, two additional-drag overlays, and two volatility-severity overlays. It does not search portfolio weights or choose a winner.",
            "",
            f"Second-seed stability: {stability_dict.get('status_str')} (max absolute headline movement {float(stability_dict.get('max_absolute_difference_float', 0.0)):.2%}).",
            "",
            "## Important limits",
            "",
            "- Bootstrap paths resample dependent blocks with replacement; they can duplicate or omit observations and cannot invent an unseen crisis.",
            "- Crisis rows are observed slices, not fresh strategy replays.",
            "- Drag and volatility variants are hypothetical overlays, not execution or correlation models.",
            "- Recovery percentiles are conditional on recovery inside the stated horizon; unrecovered probability is separate.",
            "- Rolling 12-month rows overlap and are not independent probability observations.",
            "",
            "See the CSV files and summary.json for the full results.",
        ]
    )


def run_investor_risk_sensitivity(
    *,
    portfolio_pickle_path: Path,
    output_dir_str: str,
    simulation_count_int: int,
    random_seed_int: int,
) -> Path:
    portfolio_pickle_path = portfolio_pickle_path.resolve()
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
    source_context_dict = _build_portfolio_analysis_context_dict(
        portfolio_pickle_path=portfolio_pickle_path,
        source_metadata_dict=source_metadata_dict,
        manager_metadata_dict=manager_metadata_dict,
        portfolio_obj=portfolio_obj,
    )
    realized_return_ser = extract_realized_return_ser(portfolio_obj)

    observed_calendar_month_df = build_observed_calendar_month_df(realized_return_ser)
    observed_calendar_year_df = build_observed_calendar_year_df(realized_return_ser)
    observed_crisis_window_df = build_observed_crisis_window_df(realized_return_ser)
    rolling_12m_return_df = build_rolling_12m_return_df(realized_return_ser)
    observed_path_metric_dict = compute_path_metric_dict(
        realized_return_ser.to_numpy(dtype=float)
    )

    variant_list = _build_sensitivity_variant_list(realized_return_ser, random_seed_int)
    sensitivity_scenario_df = pd.concat(
        [
            _run_sensitivity_variant_df(variant_dict, simulation_count_int)
            for variant_dict in variant_list
        ],
        ignore_index=True,
    )

    primary_variant_dict = next(
        variant_dict
        for variant_dict in variant_list
        if variant_dict["variant_key_str"] == "block_21"
    )
    alternate_seed_variant_dict = dict(primary_variant_dict)
    alternate_seed_variant_dict["variant_key_str"] = "block_21_alternate_seed"
    alternate_seed_variant_dict["variant_label_str"] = "Full sample, block 21d, alternate seed"
    alternate_seed_variant_dict["random_seed_int"] = int(random_seed_int) + 1_000_003
    alternate_seed_scenario_df = _run_sensitivity_variant_df(
        alternate_seed_variant_dict,
        simulation_count_int,
    )
    primary_scenario_df = sensitivity_scenario_df[
        sensitivity_scenario_df["variant_key_str"] == "block_21"
    ]
    seed_stability_df = _build_seed_stability_df(
        primary_scenario_df,
        alternate_seed_scenario_df,
    )
    max_stability_difference_float = (
        float(seed_stability_df["absolute_difference_float"].max())
        if len(seed_stability_df)
        else np.nan
    )
    seed_stability_status_str = (
        "stable_within_1_percentage_point"
        if np.isfinite(max_stability_difference_float)
        and max_stability_difference_float <= STABILITY_ABSOLUTE_TOLERANCE_FLOAT
        else "requires_more_simulations_or_review"
    )

    output_dir_path = build_research_output_path(
        output_dir_str,
        "portfolio",
        str(portfolio_obj.name),
        INVESTOR_RISK_SENSITIVITY_TYPE_STR,
    )
    output_dir_path.mkdir(parents=True, exist_ok=True)
    observed_calendar_month_df.to_csv(
        output_dir_path / "observed_calendar_months.csv",
        index=False,
    )
    observed_calendar_year_df.to_csv(
        output_dir_path / "observed_calendar_years.csv",
        index=False,
    )
    observed_crisis_window_df.to_csv(
        output_dir_path / "observed_crisis_windows.csv",
        index=False,
    )
    rolling_12m_return_df.to_csv(
        output_dir_path / "rolling_12m_returns.csv",
        index=False,
    )
    sensitivity_scenario_df.to_csv(
        output_dir_path / "sensitivity_scenarios.csv",
        index=False,
    )
    alternate_seed_scenario_df.to_csv(
        output_dir_path / "alternate_seed_scenarios.csv",
        index=False,
    )
    seed_stability_df.to_csv(
        output_dir_path / "seed_stability.csv",
        index=False,
    )

    variant_definition_list = [
        {
            key_str: value_obj
            for key_str, value_obj in variant_dict.items()
            if key_str != "return_ser"
        }
        for variant_dict in variant_list
    ]
    summary_dict = {
        "schema_version_int": INVESTOR_RISK_SENSITIVITY_SCHEMA_VERSION_INT,
        "analysis_status_str": "provisional_research_only",
        "investor_use_approved_bool": False,
        "source_context_dict": source_context_dict,
        "simulation_count_int": int(simulation_count_int),
        "primary_random_seed_int": int(random_seed_int),
        "primary_mean_block_length_int": PRIMARY_BLOCK_LENGTH_INT,
        "horizon_year_list": list(INVESTOR_HORIZON_YEAR_TUPLE),
        "variant_definition_list": variant_definition_list,
        "observed_path_metric_dict": observed_path_metric_dict,
        "observed_calendar_month_summary_dict": _observed_month_summary_dict(
            observed_calendar_month_df
        ),
        "primary_scenario_list": _records_from_df(primary_scenario_df),
        "seed_stability_dict": {
            "status_str": seed_stability_status_str,
            "alternate_seed_int": int(alternate_seed_variant_dict["random_seed_int"]),
            "absolute_tolerance_float": STABILITY_ABSOLUTE_TOLERANCE_FLOAT,
            "max_absolute_difference_float": max_stability_difference_float,
        },
        "presentation_readiness_dict": _presentation_readiness_dict(
            source_context_dict
        ),
        "documented_gap_list": [
            "No correlation-breakdown or sleeve-level joint-tail stress.",
            "No exact fresh crisis replay for this combined portfolio.",
            "No unseen-regime or structural-break generator.",
            "No liquidity, capacity, auction-impact, or live-fill divergence model.",
            "No correction for strategy-selection or portfolio-selection data mining.",
            "No independent revalidation of point-in-time universes, corporate actions, or adjustment modes inherited from the source backtest.",
            "The source artifact did not capture exact strategy source revisions or a save-time config hash.",
            "No investor fee, high-water-mark, tax, or cash-flow NAV model.",
            "No verified live performance layer.",
        ],
    }
    run_info_dict = {
        "entity_type": "portfolio",
        "entity_id": str(portfolio_obj.name),
        "analysis_type": INVESTOR_RISK_SENSITIVITY_TYPE_STR,
        "schema_version_int": INVESTOR_RISK_SENSITIVITY_SCHEMA_VERSION_INT,
        "parameters": {
            "source_artifact_path": str(portfolio_pickle_path),
            "simulation_count_int": int(simulation_count_int),
            "random_seed_int": int(random_seed_int),
            "variant_count_int": int(len(variant_list)),
        },
    }
    metadata_dict = {
        "artifact_type": INVESTOR_RISK_SENSITIVITY_TYPE_STR,
        "schema_version": INVESTOR_RISK_SENSITIVITY_SCHEMA_VERSION_INT,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "portfolio_name": str(portfolio_obj.name),
        "source_artifact_sha256": source_context_dict.get(
            "source_artifact_sha256_str"
        ),
        "analysis_status": "provisional_research_only",
    }
    _write_json(output_dir_path / "summary.json", summary_dict)
    _write_json(output_dir_path / "run_info.json", run_info_dict)
    _write_json(output_dir_path / "metadata.json", metadata_dict)
    (output_dir_path / "report.md").write_text(
        _build_report_markdown_str(summary_dict),
        encoding="utf-8",
    )
    return output_dir_path


def main() -> None:
    parser_obj = argparse.ArgumentParser()
    parser_obj.add_argument(
        "portfolio_pickle_path",
        type=Path,
        help="Explicit trusted portfolio pickle. No latest lookup is performed.",
    )
    parser_obj.add_argument("--output-dir", default="results")
    parser_obj.add_argument(
        "--simulation-count",
        type=int,
        default=DEFAULT_SIMULATION_COUNT_INT,
    )
    parser_obj.add_argument(
        "--random-seed",
        type=int,
        default=DEFAULT_RANDOM_SEED_INT,
    )
    parser_obj.add_argument(
        "--monthly-stability-only",
        action="store_true",
        help="Run only the focused two-seed 21-trading-day convergence check.",
    )
    parser_obj.add_argument(
        "--monthly-stability-count",
        type=int,
        default=MONTHLY_STABILITY_SIMULATION_COUNT_INT,
        help="Simulation count per seed for --monthly-stability-only.",
    )
    parser_obj.add_argument(
        "--parent-sensitivity-dir",
        type=Path,
        default=None,
        help="Optional parent sensitivity artifact to link and resolve.",
    )
    arg_namespace = parser_obj.parse_args()
    if arg_namespace.monthly_stability_only:
        output_dir_path = run_monthly_numerical_stability_followup(
            portfolio_pickle_path=arg_namespace.portfolio_pickle_path,
            output_dir_str=arg_namespace.output_dir,
            simulation_count_int=arg_namespace.monthly_stability_count,
            random_seed_int=arg_namespace.random_seed,
            parent_sensitivity_dir_path=arg_namespace.parent_sensitivity_dir,
        )
        print(f"Saved monthly numerical stability: {output_dir_path.resolve()}")
    else:
        output_dir_path = run_investor_risk_sensitivity(
            portfolio_pickle_path=arg_namespace.portfolio_pickle_path,
            output_dir_str=arg_namespace.output_dir,
            simulation_count_int=arg_namespace.simulation_count,
            random_seed_int=arg_namespace.random_seed,
        )
        print(f"Saved investor-risk sensitivity: {output_dir_path.resolve()}")


if __name__ == "__main__":
    main()
