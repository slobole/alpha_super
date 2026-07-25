"""
Post-run risk diagnostics for completed strategy or portfolio runs.

RiskAnalysis is report-only. It reads realized returns after a completed
backtest and never changes strategy, order, fill, sizing, or live execution
semantics.

Core return path:

    r_t = V_t / V_{t-1} - 1

Stationary block bootstrap:

    p = 1 / L

where L is the expected block length. At each simulated step, the path either
starts a new block with probability p or continues to the next historical
return observation, wrapping around at the sample end.
"""

from __future__ import annotations

import html
import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from alpha.engine.report import _ACTIVE_REPORT_VARIANT_STR, build_research_output_path
from alpha.engine.theme import (
    SIGNATURE_PALETTE_DICT,
    build_analyzer_report_css,
    build_report_font_head_html,
    signature_variant_context,
)


RISK_ANALYSIS_TYPE_STR = "risk_analysis"
RISK_ANALYSIS_SCHEMA_VERSION_INT = 2
RETURN_HISTOGRAM_CSV_FILENAME_STR = "return_histogram.csv"
BOOTSTRAP_EQUITY_PATH_CSV_FILENAME_STR = "bootstrap_equity_paths.csv"
BOOTSTRAP_PATH_METRIC_CSV_FILENAME_STR = "bootstrap_path_metrics.csv"
BOOTSTRAP_INTERVAL_CSV_FILENAME_STR = "bootstrap_metric_intervals.csv"
HORIZON_PROBABILITY_CSV_FILENAME_STR = "horizon_probabilities.csv"
OBSERVED_CALENDAR_MONTH_CSV_FILENAME_STR = "observed_calendar_months.csv"
INVESTOR_SCENARIO_CSV_FILENAME_STR = "investor_scenarios.csv"
INVESTOR_SUMMARY_FILENAME_STR = "investor_summary.json"
SUMMARY_FILENAME_STR = "summary.json"
RUN_INFO_FILENAME_STR = "run_info.json"
METADATA_FILENAME_STR = "metadata.json"
REPORT_FILENAME_STR = "report.html"

DEFAULT_PRIMARY_MEAN_BLOCK_LENGTH_INT = 21
DEFAULT_SENSITIVITY_BLOCK_LENGTH_TUPLE = (5, 10, 21, 63)
DEFAULT_SIMULATION_COUNT_INT = 10000
DEFAULT_RANDOM_SEED_INT = 42
DEFAULT_CONFIDENCE_LEVEL_FLOAT = 0.95
DEFAULT_DRAWDOWN_THRESHOLD_TUPLE = (-0.10, -0.20, -0.30, -0.40, -0.50)
DEFAULT_UPSIDE_THRESHOLD_TUPLE = (0.10, 0.20, 0.30, 0.40, 0.50)
DEFAULT_HORIZON_YEAR_TUPLE = (1, 2, 3, 4, 5)
DEFAULT_INVESTOR_HORIZON_YEAR_TUPLE = (1, 3, 5)
DEFAULT_ROLLING_LOSS_WINDOW_TUPLE = (1, 5, 21, 63, 126, 252)
DEFAULT_TIME_UNDERWATER_BREACH_MONTH_TUPLE = (3, 6, 12, 24)
TRADING_DAYS_PER_MONTH_INT = 21
TRADING_DAYS_PER_YEAR_INT = 252
DEFAULT_RETURN_HISTOGRAM_BIN_COUNT_INT = 80
DEFAULT_EQUITY_PATH_SAMPLE_COUNT_INT = 100
TRADING_DAYS_PER_YEAR_FLOAT = 252.0
OBSERVED_CALENDAR_MONTH_COLUMN_TUPLE = (
    "calendar_month_str",
    "calendar_month_end_str",
    "scheduled_calendar_month_end_str",
    "effective_start_date_str",
    "effective_end_date_str",
    "sample_boundary_month_bool",
    "trading_day_count_int",
    "calendar_month_return_float",
)
CVAR_99_TAIL_FRACTION_FLOAT = 0.01
CVAR_99_MIN_TAIL_SAMPLE_INT = 50
CVAR_99_SMALL_SAMPLE_FOOTNOTE_STR = (
    "Tail sample is small for this metric; CI is wide and the point estimate is noisy."
)
TAIL_SENSITIVE_METRIC_TUPLE = (
    "var_99_daily_return_float",
    "cvar_99_daily_return_float",
    "monthly_var_99_return_float",
    "monthly_cvar_99_return_float",
)

# *** CRITICAL*** verdict bands are generic reference heuristics for fast
# reading only. They are NOT calibrated to any account or risk tolerance and do
# NOT constitute trade advice. See ASSUMPTIONS_AND_GAPS RiskAnalysis note.
VERDICT_STATUS_GREEN_STR = "green"
VERDICT_STATUS_AMBER_STR = "amber"
VERDICT_STATUS_RED_STR = "red"
VERDICT_STATUS_NA_STR = "na"
# Edge: probability the bootstrap path ended below where it started.
EDGE_TERMINAL_LOSS_GREEN_MAX_FLOAT = 0.10
EDGE_TERMINAL_LOSS_AMBER_MAX_FLOAT = 0.30
# Drawdown depth: |bootstrap p05 max drawdown|.
DRAWDOWN_DEPTH_GREEN_MAX_FLOAT = 0.20
DRAWDOWN_DEPTH_AMBER_MAX_FLOAT = 0.35
# Time underwater: P(longest underwater stretch >= 12 trading months).
UNDERWATER_12M_GREEN_MAX_FLOAT = 0.20
UNDERWATER_12M_AMBER_MAX_FLOAT = 0.50
# Worst rolling 12-month return: |bootstrap p05 worst-12m|.
WORST_YEAR_GREEN_MAX_FLOAT = 0.15
WORST_YEAR_AMBER_MAX_FLOAT = 0.30


@dataclass
class RiskAnalysisResult:
    strategy_name_str: str
    source_strategy_ref_str: str
    realized_return_ser: pd.Series
    return_histogram_df: pd.DataFrame
    bootstrap_equity_path_df: pd.DataFrame
    bootstrap_path_metric_df: pd.DataFrame
    bootstrap_interval_df: pd.DataFrame
    summary_dict: dict[str, object]
    horizon_probability_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    output_dir_path: Path | None = None
    observed_calendar_month_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    investor_scenario_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    investor_summary_dict: dict[str, object] = field(default_factory=dict)
    source_entity_type_str: str = "strategy"
    analysis_context_dict: dict[str, object] = field(default_factory=dict)


def extract_realized_return_ser(strategy_obj: object) -> pd.Series:
    """
    Return the realized post-run daily return series.

    The first stored row is excluded because it is a bootstrap/initial-state row
    from the strategy reporting lifecycle, not a realized one-day return.
    """
    result_df = getattr(strategy_obj, "results", None)
    if result_df is None or len(result_df) == 0:
        raise ValueError("strategy.results is empty; run the strategy before RiskAnalysis.")
    if "daily_returns" not in result_df.columns:
        if "total_value" not in result_df.columns:
            raise ValueError("strategy.results must include daily_returns or total_value.")
        # *** CRITICAL*** report-only return reconstruction: this uses realized
        # post-run equity and must never feed signal, sizing, or order logic.
        raw_return_ser = result_df["total_value"].astype(float).pct_change(fill_method=None)
    else:
        raw_return_ser = result_df["daily_returns"].astype(float)

    # *** CRITICAL*** post-run diagnostics boundary: exclude the initial
    # placeholder row so bootstrap paths contain only realized daily returns.
    realized_return_ser = raw_return_ser.iloc[1:].replace([np.inf, -np.inf], np.nan).dropna()
    realized_return_ser.name = "realized_return_float"
    return realized_return_ser.astype(float)


def stationary_bootstrap_index_mat(
    sample_size_int: int,
    simulation_count_int: int,
    mean_block_length_int: int,
    random_seed_int: int,
    path_length_int: int | None = None,
) -> np.ndarray:
    if sample_size_int <= 0:
        raise ValueError("sample_size_int must be positive.")
    if simulation_count_int <= 0:
        raise ValueError("simulation_count_int must be positive.")
    if mean_block_length_int <= 0:
        raise ValueError("mean_block_length_int must be positive.")
    if path_length_int is None:
        path_length_int = int(sample_size_int)
    if path_length_int <= 0:
        raise ValueError("path_length_int must be positive.")

    rng_obj = np.random.default_rng(int(random_seed_int))
    restart_probability_float = 1.0 / float(mean_block_length_int)
    index_mat = np.empty((int(simulation_count_int), int(path_length_int)), dtype=np.int64)

    for simulation_idx_int in range(int(simulation_count_int)):
        current_index_int = 0
        for step_idx_int in range(int(path_length_int)):
            if step_idx_int == 0 or rng_obj.random() < restart_probability_float:
                current_index_int = int(rng_obj.integers(0, sample_size_int))
            else:
                current_index_int = (current_index_int + 1) % sample_size_int
            index_mat[simulation_idx_int, step_idx_int] = current_index_int

    return index_mat


def build_return_histogram_df(
    realized_return_ser: pd.Series,
    bin_count_int: int = DEFAULT_RETURN_HISTOGRAM_BIN_COUNT_INT,
) -> pd.DataFrame:
    return_ser = realized_return_ser.astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    return_vec = return_ser.to_numpy(dtype=float)
    if return_vec.size == 0:
        return pd.DataFrame(
            columns=[
                "bin_left_float",
                "bin_right_float",
                "bin_mid_float",
                "count_int",
                "probability_float",
            ]
        )
    if bin_count_int <= 0:
        raise ValueError("bin_count_int must be positive.")
    count_vec, edge_vec = np.histogram(return_vec, bins=int(bin_count_int))
    row_list: list[dict[str, object]] = []
    total_count_float = float(count_vec.sum())
    for bin_idx_int, count_int in enumerate(count_vec):
        bin_left_float = float(edge_vec[bin_idx_int])
        bin_right_float = float(edge_vec[bin_idx_int + 1])
        row_list.append(
            {
                "bin_left_float": bin_left_float,
                "bin_right_float": bin_right_float,
                "bin_mid_float": (bin_left_float + bin_right_float) / 2.0,
                "count_int": int(count_int),
                "probability_float": float(count_int) / total_count_float if total_count_float > 0 else np.nan,
            }
        )
    return pd.DataFrame(row_list)


def build_bootstrap_equity_path_df(
    realized_return_ser: pd.Series,
    mean_block_length_int: int,
    simulation_count_int: int = DEFAULT_SIMULATION_COUNT_INT,
    random_seed_int: int = DEFAULT_RANDOM_SEED_INT,
    path_sample_count_int: int = DEFAULT_EQUITY_PATH_SAMPLE_COUNT_INT,
) -> pd.DataFrame:
    return_vec = realized_return_ser.astype(float).replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
    if return_vec.size == 0:
        raise ValueError("realized_return_ser must contain at least one return.")
    if path_sample_count_int <= 0:
        raise ValueError("path_sample_count_int must be positive.")

    sampled_path_count_int = min(int(simulation_count_int), int(path_sample_count_int))
    index_mat = stationary_bootstrap_index_mat(
        sample_size_int=int(return_vec.size),
        simulation_count_int=sampled_path_count_int,
        mean_block_length_int=int(mean_block_length_int),
        random_seed_int=int(random_seed_int),
    )
    simulated_return_mat = return_vec[index_mat]
    simulated_equity_mat = np.cumprod(1.0 + simulated_return_mat, axis=1)
    simulated_equity_mat = np.concatenate(
        [np.ones((sampled_path_count_int, 1), dtype=float), simulated_equity_mat],
        axis=1,
    )
    observed_equity_vec = np.concatenate(([1.0], np.cumprod(1.0 + return_vec)))
    percentile_map = {
        "p05": np.quantile(simulated_equity_mat, 0.05, axis=0),
        "p50": np.quantile(simulated_equity_mat, 0.50, axis=0),
        "p95": np.quantile(simulated_equity_mat, 0.95, axis=0),
        "observed": observed_equity_vec,
    }

    row_list: list[dict[str, object]] = []
    for path_id_int, equity_vec in enumerate(simulated_equity_mat):
        for step_int, equity_float in enumerate(equity_vec):
            row_list.append(
                {
                    "mean_block_length_int": int(mean_block_length_int),
                    "path_kind_str": "bootstrap",
                    "path_id_int": int(path_id_int),
                    "step_int": int(step_int),
                    "equity_float": float(equity_float),
                }
            )
    for path_kind_str, equity_vec in percentile_map.items():
        path_id_int = -1 if path_kind_str == "observed" else -int(path_kind_str[1:])
        for step_int, equity_float in enumerate(equity_vec):
            row_list.append(
                {
                    "mean_block_length_int": int(mean_block_length_int),
                    "path_kind_str": path_kind_str,
                    "path_id_int": int(path_id_int),
                    "step_int": int(step_int),
                    "equity_float": float(equity_float),
                }
            )

    return pd.DataFrame(row_list)


def build_bootstrap_path_metric_df(
    realized_return_ser: pd.Series,
    mean_block_length_tuple: Sequence[int],
    simulation_count_int: int = DEFAULT_SIMULATION_COUNT_INT,
    random_seed_int: int = DEFAULT_RANDOM_SEED_INT,
    rolling_loss_window_tuple: Sequence[int] = DEFAULT_ROLLING_LOSS_WINDOW_TUPLE,
) -> pd.DataFrame:
    return_vec = realized_return_ser.astype(float).to_numpy(dtype=float)
    if return_vec.size == 0:
        raise ValueError("realized_return_ser must contain at least one return.")

    block_length_tuple = tuple(dict.fromkeys(int(value_int) for value_int in mean_block_length_tuple))
    if len(block_length_tuple) == 0:
        raise ValueError("At least one block length is required.")

    row_list: list[dict[str, object]] = []
    for block_position_int, mean_block_length_int in enumerate(block_length_tuple):
        index_mat = stationary_bootstrap_index_mat(
            sample_size_int=int(return_vec.size),
            simulation_count_int=int(simulation_count_int),
            mean_block_length_int=int(mean_block_length_int),
            random_seed_int=int(random_seed_int) + block_position_int,
        )
        for simulation_idx_int, index_vec in enumerate(index_mat):
            simulated_return_vec = return_vec[index_vec]
            metric_dict = compute_path_metric_dict(
                simulated_return_vec,
                rolling_loss_window_tuple=rolling_loss_window_tuple,
            )
            metric_dict["mean_block_length_int"] = int(mean_block_length_int)
            metric_dict["simulation_int"] = int(simulation_idx_int)
            row_list.append(metric_dict)

    return pd.DataFrame(row_list)


def compute_path_metric_dict(
    return_vec: np.ndarray,
    rolling_loss_window_tuple: Sequence[int] = DEFAULT_ROLLING_LOSS_WINDOW_TUPLE,
) -> dict[str, object]:
    clean_return_vec = np.asarray(return_vec, dtype=float)
    clean_return_vec = clean_return_vec[np.isfinite(clean_return_vec)]
    if clean_return_vec.size == 0:
        raise ValueError("return_vec must contain at least one finite return.")

    equity_vec = np.cumprod(1.0 + clean_return_vec)
    terminal_return_float = float(equity_vec[-1] - 1.0)
    sample_year_float = float(clean_return_vec.size) / TRADING_DAYS_PER_YEAR_FLOAT
    if equity_vec[-1] > 0.0 and sample_year_float > 0.0:
        cagr_float = float(equity_vec[-1] ** (1.0 / sample_year_float) - 1.0)
    else:
        cagr_float = np.nan
    expected_daily_return_float = float(clean_return_vec.mean())
    annualized_ev_float = expected_daily_return_float * TRADING_DAYS_PER_YEAR_FLOAT

    annual_volatility_float = (
        float(clean_return_vec.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR_FLOAT))
        if clean_return_vec.size >= 2
        else np.nan
    )
    sharpe_float = (
        float(clean_return_vec.mean() / clean_return_vec.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR_FLOAT))
        if clean_return_vec.size >= 2 and clean_return_vec.std(ddof=1) > 0.0
        else np.nan
    )

    # *** CRITICAL*** drawdown uses only the simulated path's running peak:
    # drawdown_t = V_t / max(V_0, ..., V_t) - 1.
    running_peak_vec = np.maximum.accumulate(np.concatenate(([1.0], equity_vec)))[1:]
    drawdown_vec = equity_vec / running_peak_vec - 1.0
    max_drawdown_float = float(drawdown_vec.min())
    mar_float = (
        float(cagr_float / abs(max_drawdown_float))
        if np.isfinite(cagr_float) and max_drawdown_float < 0.0
        else np.nan
    )
    longest_underwater_days_float = float(_longest_underwater_days_int(drawdown_vec))

    # *** CRITICAL*** monthly = non-overlapping 21-trading-day chunks of the
    # same simulated daily returns; this keeps observed and bootstrap monthly
    # metrics directly comparable. See ASSUMPTIONS_AND_GAPS G-011 detailed note.
    monthly_return_vec = _monthly_return_vec_from_daily_float(clean_return_vec)
    monthly_count_int = int(monthly_return_vec.size)
    if monthly_count_int >= 1:
        monthly_expected_return_float = float(monthly_return_vec.mean())
    else:
        monthly_expected_return_float = np.nan
    if monthly_count_int >= 2:
        monthly_volatility_float = float(monthly_return_vec.std(ddof=1))
        monthly_sharpe_float = (
            float(monthly_return_vec.mean() / monthly_return_vec.std(ddof=1) * np.sqrt(12.0))
            if monthly_return_vec.std(ddof=1) > 0.0
            else np.nan
        )
    else:
        monthly_volatility_float = np.nan
        monthly_sharpe_float = np.nan

    metric_dict: dict[str, object] = {
        "expected_daily_return_float": expected_daily_return_float,
        "annualized_ev_float": annualized_ev_float,
        "terminal_return_float": terminal_return_float,
        "cagr_float": cagr_float,
        "annual_volatility_float": annual_volatility_float,
        "sharpe_float": sharpe_float,
        "max_drawdown_float": max_drawdown_float,
        "mar_float": mar_float,
        "longest_underwater_days_float": longest_underwater_days_float,
        "var_95_daily_return_float": _var_float(clean_return_vec, 0.05),
        "cvar_95_daily_return_float": _tail_mean_float(clean_return_vec, 0.05),
        "var_99_daily_return_float": _var_float(clean_return_vec, 0.01),
        "cvar_99_daily_return_float": _tail_mean_float(clean_return_vec, 0.01),
        "monthly_expected_return_float": monthly_expected_return_float,
        "monthly_volatility_float": monthly_volatility_float,
        "monthly_sharpe_float": monthly_sharpe_float,
        "monthly_var_95_return_float": _var_float(monthly_return_vec, 0.05),
        "monthly_cvar_95_return_float": _tail_mean_float(monthly_return_vec, 0.05),
        "monthly_var_99_return_float": _var_float(monthly_return_vec, 0.01),
        "monthly_cvar_99_return_float": _tail_mean_float(monthly_return_vec, 0.01),
    }
    for window_int in rolling_loss_window_tuple:
        normalized_window_int = int(window_int)
        metric_dict[f"worst_{normalized_window_int}d_return_float"] = _worst_rolling_return_float(
            clean_return_vec,
            normalized_window_int,
        )
    return metric_dict


def build_bootstrap_interval_df(
    bootstrap_path_metric_df: pd.DataFrame,
    observed_metric_dict: dict[str, object],
    confidence_level_float: float = DEFAULT_CONFIDENCE_LEVEL_FLOAT,
) -> pd.DataFrame:
    confidence_level_float = float(confidence_level_float)
    if not 0.0 < confidence_level_float < 1.0:
        raise ValueError("confidence_level_float must be between 0 and 1.")

    metric_column_list = [
        column_name_str
        for column_name_str in bootstrap_path_metric_df.columns
        if column_name_str.endswith("_float")
    ]
    lower_quantile_float = (1.0 - confidence_level_float) / 2.0
    upper_quantile_float = 1.0 - lower_quantile_float
    row_list: list[dict[str, object]] = []
    for block_length_int, block_metric_df in bootstrap_path_metric_df.groupby("mean_block_length_int"):
        for metric_name_str in metric_column_list:
            metric_ser = block_metric_df[metric_name_str].astype(float).replace([np.inf, -np.inf], np.nan).dropna()
            if len(metric_ser) == 0:
                continue
            observed_value_float = _json_float(observed_metric_dict.get(metric_name_str))
            row_list.append(
                {
                    "mean_block_length_int": int(block_length_int),
                    "metric_name_str": metric_name_str,
                    "observed_value_float": observed_value_float,
                    "observed_percentile_float": _observed_percentile_float(
                        metric_ser, observed_value_float
                    ),
                    "bootstrap_mean_float": float(metric_ser.mean()),
                    "ci_half_width_float": float(
                        (
                            metric_ser.quantile(upper_quantile_float)
                            - metric_ser.quantile(lower_quantile_float)
                        )
                        / 2.0
                    ),
                    "ci_lower_float": float(metric_ser.quantile(lower_quantile_float)),
                    "ci_upper_float": float(metric_ser.quantile(upper_quantile_float)),
                    "p05_float": float(metric_ser.quantile(0.05)),
                    "p50_float": float(metric_ser.quantile(0.50)),
                    "p95_float": float(metric_ser.quantile(0.95)),
                    "confidence_level_float": float(confidence_level_float),
                }
            )
    return pd.DataFrame(row_list)


def build_observed_calendar_month_df(
    realized_return_ser: pd.Series,
) -> pd.DataFrame:
    """Compound dated realized returns into observed calendar-month rows."""
    return_ser = realized_return_ser.astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    if not isinstance(return_ser.index, pd.DatetimeIndex):
        raise ValueError("realized_return_ser must use a DatetimeIndex for calendar-month analysis.")
    if len(return_ser) == 0:
        return pd.DataFrame(columns=OBSERVED_CALENDAR_MONTH_COLUMN_TUPLE)

    sorted_return_ser = return_ser.sort_index()
    # *** CRITICAL*** report-only calendar aggregation: each calendar-month
    # return compounds only realized daily returns whose timestamps fall inside
    # that month. It must never feed signal, sizing, or execution logic.
    month_group_obj = sorted_return_ser.groupby(sorted_return_ser.index.to_period("M"))
    row_list: list[dict[str, object]] = []
    calendar_month_obj_list = list(month_group_obj.groups)
    for calendar_month_obj, calendar_month_return_ser in month_group_obj:
        calendar_month_return_float = float(
            np.prod(1.0 + calendar_month_return_ser.to_numpy(dtype=float)) - 1.0
        )
        effective_start_date_str = (
            pd.Timestamp(calendar_month_return_ser.index.min()).date().isoformat()
        )
        effective_end_date_str = (
            pd.Timestamp(calendar_month_return_ser.index.max()).date().isoformat()
        )
        row_list.append(
            {
                "calendar_month_str": str(calendar_month_obj),
                "calendar_month_end_str": effective_end_date_str,
                "scheduled_calendar_month_end_str": (
                    calendar_month_obj.end_time.date().isoformat()
                ),
                "effective_start_date_str": effective_start_date_str,
                "effective_end_date_str": effective_end_date_str,
                "sample_boundary_month_bool": bool(
                    calendar_month_obj == calendar_month_obj_list[0]
                    or calendar_month_obj == calendar_month_obj_list[-1]
                ),
                "trading_day_count_int": int(len(calendar_month_return_ser)),
                "calendar_month_return_float": calendar_month_return_float,
            }
        )
    return pd.DataFrame(row_list)


def build_horizon_probability_df(
    realized_return_ser: pd.Series,
    mean_block_length_int: int,
    simulation_count_int: int = DEFAULT_SIMULATION_COUNT_INT,
    random_seed_int: int = DEFAULT_RANDOM_SEED_INT,
    horizon_year_tuple: Sequence[int] = DEFAULT_HORIZON_YEAR_TUPLE,
    drawdown_threshold_tuple: Sequence[float] = DEFAULT_DRAWDOWN_THRESHOLD_TUPLE,
    upside_threshold_tuple: Sequence[float] = DEFAULT_UPSIDE_THRESHOLD_TUPLE,
    time_underwater_breach_month_tuple: Sequence[int] = DEFAULT_TIME_UNDERWATER_BREACH_MONTH_TUPLE,
) -> pd.DataFrame:
    """
    Build horizon-level downside and upside probability rows.

    For horizon H and bootstrap path s:

        equity_s,t = product(1 + return_s,i), i=1..t
        drawdown_s,t = equity_s,t / max(1, equity_s,1, ..., equity_s,t) - 1
        max_gain_s,H = max(1, equity_s,1, ..., equity_s,H) - 1

    The downside cells count paths where min(drawdown_s,1..H) breaches a
    threshold. The upside cells count paths where max_gain_s,H reaches a
    threshold at least once. This is report-only and uses only resampled
    realized returns.
    """
    return_vec = realized_return_ser.astype(float).replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
    if return_vec.size == 0:
        raise ValueError("realized_return_ser must contain at least one return.")

    normalized_horizon_year_tuple = _normalized_positive_int_tuple(horizon_year_tuple, "horizon years")
    normalized_drawdown_threshold_tuple = tuple(float(value_float) for value_float in drawdown_threshold_tuple)
    normalized_upside_threshold_tuple = tuple(float(value_float) for value_float in upside_threshold_tuple)
    normalized_underwater_month_tuple = _normalized_positive_int_tuple(
        time_underwater_breach_month_tuple,
        "time-underwater months",
    )
    if len(normalized_drawdown_threshold_tuple) == 0:
        raise ValueError("At least one drawdown threshold is required.")
    if len(normalized_upside_threshold_tuple) == 0:
        raise ValueError("At least one upside threshold is required.")

    max_requested_day_int = max(
        int(year_int) * TRADING_DAYS_PER_YEAR_INT
        for year_int in normalized_horizon_year_tuple
    )
    bootstrap_path_length_int = min(int(return_vec.size), int(max_requested_day_int))
    index_mat = stationary_bootstrap_index_mat(
        sample_size_int=int(return_vec.size),
        simulation_count_int=int(simulation_count_int),
        mean_block_length_int=int(mean_block_length_int),
        random_seed_int=int(random_seed_int),
        path_length_int=bootstrap_path_length_int,
    )

    horizon_metric_dict: dict[int, dict[str, list[object]]] = {
        int(year_int): {
            "max_drawdown_float": [],
            "max_gain_float": [],
            "terminal_return_float": [],
            "longest_underwater_days_float": [],
            "max_drawdown_recovery_days_float": [],
            "max_drawdown_unrecovered_bool": [],
            "terminal_underwater_bool": [],
        }
        for year_int in normalized_horizon_year_tuple
    }
    for index_vec in index_mat:
        simulated_return_vec = return_vec[index_vec]
        for horizon_year_int in normalized_horizon_year_tuple:
            horizon_day_int = int(horizon_year_int) * TRADING_DAYS_PER_YEAR_INT
            if simulated_return_vec.size < horizon_day_int:
                continue
            path_horizon_metric_dict = _path_horizon_metric_dict(
                simulated_return_vec,
                horizon_day_int,
            )
            horizon_metric_dict[int(horizon_year_int)]["max_drawdown_float"].append(
                float(path_horizon_metric_dict["max_drawdown_float"])
            )
            horizon_metric_dict[int(horizon_year_int)]["max_gain_float"].append(
                float(path_horizon_metric_dict["max_gain_float"])
            )
            horizon_metric_dict[int(horizon_year_int)]["terminal_return_float"].append(
                float(path_horizon_metric_dict["terminal_return_float"])
            )
            horizon_metric_dict[int(horizon_year_int)]["longest_underwater_days_float"].append(
                float(path_horizon_metric_dict["longest_underwater_days_float"])
            )
            recovery_days_obj = path_horizon_metric_dict["max_drawdown_recovery_days_float"]
            if recovery_days_obj is not None:
                horizon_metric_dict[int(horizon_year_int)]["max_drawdown_recovery_days_float"].append(
                    float(recovery_days_obj)
                )
            horizon_metric_dict[int(horizon_year_int)]["max_drawdown_unrecovered_bool"].append(
                bool(path_horizon_metric_dict["max_drawdown_unrecovered_bool"])
            )
            horizon_metric_dict[int(horizon_year_int)]["terminal_underwater_bool"].append(
                bool(path_horizon_metric_dict["terminal_underwater_bool"])
            )

    row_list: list[dict[str, object]] = []
    for horizon_year_int in normalized_horizon_year_tuple:
        horizon_day_int = int(horizon_year_int) * TRADING_DAYS_PER_YEAR_INT
        row_dict: dict[str, object] = {
            "horizon_year_int": int(horizon_year_int),
            "horizon_day_int": int(horizon_day_int),
        }
        max_drawdown_vec = np.asarray(
            horizon_metric_dict[int(horizon_year_int)]["max_drawdown_float"],
            dtype=float,
        )
        max_gain_vec = np.asarray(
            horizon_metric_dict[int(horizon_year_int)]["max_gain_float"],
            dtype=float,
        )
        terminal_return_vec = np.asarray(
            horizon_metric_dict[int(horizon_year_int)]["terminal_return_float"],
            dtype=float,
        )
        longest_underwater_days_vec = np.asarray(
            horizon_metric_dict[int(horizon_year_int)]["longest_underwater_days_float"],
            dtype=float,
        )
        recovery_days_vec = np.asarray(
            horizon_metric_dict[int(horizon_year_int)]["max_drawdown_recovery_days_float"],
            dtype=float,
        )
        unrecovered_bool_vec = np.asarray(
            horizon_metric_dict[int(horizon_year_int)]["max_drawdown_unrecovered_bool"],
            dtype=bool,
        )
        terminal_underwater_bool_vec = np.asarray(
            horizon_metric_dict[int(horizon_year_int)]["terminal_underwater_bool"],
            dtype=bool,
        )
        row_dict["simulation_path_count_int"] = int(max_drawdown_vec.size)
        if max_drawdown_vec.size == 0:
            row_dict["max_drawdown_p05_float"] = None
            row_dict["max_drawdown_p50_float"] = None
            row_dict["max_gain_p50_float"] = None
            row_dict["max_gain_p95_float"] = None
            row_dict["terminal_return_p05_float"] = None
            row_dict["terminal_return_p25_float"] = None
            row_dict["terminal_return_p50_float"] = None
            row_dict["terminal_return_p75_float"] = None
            row_dict["terminal_return_p95_float"] = None
            row_dict["terminal_loss_probability_float"] = None
            row_dict["longest_underwater_days_p50_float"] = None
            row_dict["longest_underwater_days_p95_float"] = None
            row_dict["max_drawdown_recovery_days_p50_float"] = None
            row_dict["max_drawdown_recovery_days_p95_float"] = None
            row_dict["max_drawdown_recovered_path_count_int"] = 0
            row_dict["max_drawdown_unrecovered_probability_float"] = None
            row_dict["deepest_drawdown_unrecovered_probability_float"] = None
            row_dict["terminal_underwater_probability_float"] = None
            for threshold_float in normalized_drawdown_threshold_tuple:
                row_dict[_threshold_column_name_str("drawdown_lte", abs(float(threshold_float)))] = None
            for threshold_float in normalized_upside_threshold_tuple:
                row_dict[_threshold_column_name_str("gain_gte", abs(float(threshold_float)))] = None
            for month_int in normalized_underwater_month_tuple:
                row_dict[f"underwater_ge_{int(month_int)}m_probability_float"] = None
            row_list.append(row_dict)
            continue

        row_dict["max_drawdown_p05_float"] = float(np.quantile(max_drawdown_vec, 0.05))
        row_dict["max_drawdown_p50_float"] = float(np.quantile(max_drawdown_vec, 0.50))
        row_dict["max_gain_p50_float"] = float(np.quantile(max_gain_vec, 0.50))
        row_dict["max_gain_p95_float"] = float(np.quantile(max_gain_vec, 0.95))
        row_dict["terminal_return_p05_float"] = float(np.quantile(terminal_return_vec, 0.05))
        row_dict["terminal_return_p25_float"] = float(np.quantile(terminal_return_vec, 0.25))
        row_dict["terminal_return_p50_float"] = float(np.quantile(terminal_return_vec, 0.50))
        row_dict["terminal_return_p75_float"] = float(np.quantile(terminal_return_vec, 0.75))
        row_dict["terminal_return_p95_float"] = float(np.quantile(terminal_return_vec, 0.95))
        row_dict["terminal_loss_probability_float"] = float((terminal_return_vec < 0.0).mean())
        row_dict["longest_underwater_days_p50_float"] = float(
            np.quantile(longest_underwater_days_vec, 0.50)
        )
        row_dict["longest_underwater_days_p95_float"] = float(
            np.quantile(longest_underwater_days_vec, 0.95)
        )
        row_dict["max_drawdown_recovery_days_p50_float"] = (
            float(np.quantile(recovery_days_vec, 0.50)) if recovery_days_vec.size else None
        )
        row_dict["max_drawdown_recovery_days_p95_float"] = (
            float(np.quantile(recovery_days_vec, 0.95)) if recovery_days_vec.size else None
        )
        row_dict["max_drawdown_recovered_path_count_int"] = int(recovery_days_vec.size)
        row_dict["max_drawdown_unrecovered_probability_float"] = float(
            unrecovered_bool_vec.mean()
        )
        row_dict["deepest_drawdown_unrecovered_probability_float"] = row_dict[
            "max_drawdown_unrecovered_probability_float"
        ]
        row_dict["terminal_underwater_probability_float"] = float(
            terminal_underwater_bool_vec.mean()
        )
        for threshold_float in normalized_drawdown_threshold_tuple:
            drawdown_threshold_float = -abs(float(threshold_float))
            row_dict[_threshold_column_name_str("drawdown_lte", abs(drawdown_threshold_float))] = float(
                (max_drawdown_vec <= drawdown_threshold_float).mean()
            )
        for threshold_float in normalized_upside_threshold_tuple:
            upside_threshold_float = abs(float(threshold_float))
            row_dict[_threshold_column_name_str("gain_gte", upside_threshold_float)] = float(
                (max_gain_vec >= upside_threshold_float).mean()
            )
        for month_int in normalized_underwater_month_tuple:
            underwater_day_int = int(month_int) * TRADING_DAYS_PER_MONTH_INT
            row_dict[f"underwater_ge_{int(month_int)}m_probability_float"] = float(
                (longest_underwater_days_vec >= float(underwater_day_int)).mean()
            )
        row_list.append(row_dict)

    return pd.DataFrame(row_list)


def build_investor_scenario_df(
    realized_return_ser: pd.Series,
    horizon_probability_df: pd.DataFrame,
    mean_block_length_int: int,
    simulation_count_int: int = DEFAULT_SIMULATION_COUNT_INT,
    random_seed_int: int = DEFAULT_RANDOM_SEED_INT,
    investor_horizon_year_tuple: Sequence[int] = DEFAULT_INVESTOR_HORIZON_YEAR_TUPLE,
) -> pd.DataFrame:
    """
    Build a small, investor-readable scenario table from realized returns.

    Observed calendar months and bootstrap-implied 21-trading-day periods are
    intentionally separate. The modeled rows are historically conditioned
    diagnostics, not forecasts or promised ranges.
    """
    clean_return_ser = (
        realized_return_ser.astype(float)
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .sort_index()
    )
    if len(clean_return_ser) == 0:
        raise ValueError("realized_return_ser must contain at least one return.")
    normalized_horizon_year_tuple = _normalized_positive_int_tuple(
        investor_horizon_year_tuple,
        "investor horizon years",
    )

    row_list: list[dict[str, object]] = []
    daily_return_vec = clean_return_ser.to_numpy(dtype=float)
    observed_daily_row_dict = _empty_investor_scenario_row_dict(
        scenario_key_str="observed_daily",
        scenario_label_str="Observed trading day",
        evidence_kind_str="observed",
        period_definition_str="realized_trading_day",
        horizon_day_int=1,
        sample_count_int=int(daily_return_vec.size),
    )
    observed_daily_row_dict.update(_terminal_distribution_dict(daily_return_vec))
    row_list.append(observed_daily_row_dict)

    observed_calendar_month_df = (
        build_observed_calendar_month_df(clean_return_ser)
        if isinstance(clean_return_ser.index, pd.DatetimeIndex)
        else pd.DataFrame(columns=OBSERVED_CALENDAR_MONTH_COLUMN_TUPLE)
    )
    calendar_month_return_vec = observed_calendar_month_df[
        "calendar_month_return_float"
    ].to_numpy(dtype=float)
    observed_calendar_month_row_dict = _empty_investor_scenario_row_dict(
        scenario_key_str="observed_calendar_month",
        scenario_label_str="Observed calendar month",
        evidence_kind_str="observed",
        period_definition_str="calendar_month_including_boundary_months",
        horizon_day_int=None,
        sample_count_int=int(calendar_month_return_vec.size),
    )
    observed_calendar_month_row_dict.update(
        _terminal_distribution_dict(calendar_month_return_vec)
    )
    row_list.append(observed_calendar_month_row_dict)

    modeled_month_row_dict = _empty_investor_scenario_row_dict(
        scenario_key_str="modeled_21d",
        scenario_label_str="Modeled 21-trading-day period",
        evidence_kind_str="bootstrap_implied",
        period_definition_str="21_consecutive_trading_days",
        horizon_day_int=TRADING_DAYS_PER_MONTH_INT,
        sample_count_int=int(daily_return_vec.size),
    )
    if daily_return_vec.size >= TRADING_DAYS_PER_MONTH_INT:
        # *** CRITICAL*** report-only bootstrap: these 21-day paths resample
        # realized returns and have no calendar timestamps. They must not be
        # labeled as observed calendar months or used by trading logic.
        month_index_mat = stationary_bootstrap_index_mat(
            sample_size_int=int(daily_return_vec.size),
            simulation_count_int=int(simulation_count_int),
            mean_block_length_int=int(mean_block_length_int),
            random_seed_int=int(random_seed_int),
            path_length_int=TRADING_DAYS_PER_MONTH_INT,
        )
        month_path_return_mat = daily_return_vec[month_index_mat]
        modeled_month_row_dict.update(
            _path_distribution_dict(month_path_return_mat)
        )
        modeled_month_row_dict["simulation_path_count_int"] = int(
            month_path_return_mat.shape[0]
        )
    row_list.append(modeled_month_row_dict)

    for horizon_year_int in normalized_horizon_year_tuple:
        horizon_day_int = int(horizon_year_int) * TRADING_DAYS_PER_YEAR_INT
        horizon_row_dict = _empty_investor_scenario_row_dict(
            scenario_key_str=f"modeled_{int(horizon_year_int)}y",
            scenario_label_str=f"Modeled {int(horizon_year_int)}-year horizon",
            evidence_kind_str="bootstrap_implied",
            period_definition_str=f"{int(horizon_day_int)}_consecutive_trading_days",
            horizon_day_int=int(horizon_day_int),
            sample_count_int=int(daily_return_vec.size),
        )
        matching_horizon_df = horizon_probability_df[
            horizon_probability_df["horizon_year_int"] == int(horizon_year_int)
        ]
        if len(matching_horizon_df) > 0:
            source_horizon_ser = matching_horizon_df.iloc[0]
            for field_str in (
                "simulation_path_count_int",
                "terminal_return_p05_float",
                "terminal_return_p25_float",
                "terminal_return_p50_float",
                "terminal_return_p75_float",
                "terminal_return_p95_float",
                "terminal_loss_probability_float",
                "max_drawdown_p05_float",
                "max_drawdown_p50_float",
                "longest_underwater_days_p50_float",
                "longest_underwater_days_p95_float",
                "max_drawdown_recovery_days_p50_float",
                "max_drawdown_recovery_days_p95_float",
                "max_drawdown_recovered_path_count_int",
                "max_drawdown_unrecovered_probability_float",
                "deepest_drawdown_unrecovered_probability_float",
                "terminal_underwater_probability_float",
                "underwater_ge_12m_probability_float",
            ):
                if field_str in source_horizon_ser.index:
                    horizon_row_dict[field_str] = source_horizon_ser[field_str]
        row_list.append(horizon_row_dict)

    investor_scenario_df = pd.DataFrame(row_list)
    for integer_column_str in (
        "horizon_day_int",
        "sample_count_int",
        "simulation_path_count_int",
        "max_drawdown_recovered_path_count_int",
    ):
        investor_scenario_df[integer_column_str] = investor_scenario_df[
            integer_column_str
        ].astype("Int64")
    return investor_scenario_df


def _empty_investor_scenario_row_dict(
    *,
    scenario_key_str: str,
    scenario_label_str: str,
    evidence_kind_str: str,
    period_definition_str: str,
    horizon_day_int: int | None,
    sample_count_int: int,
) -> dict[str, object]:
    return {
        "scenario_key_str": str(scenario_key_str),
        "scenario_label_str": str(scenario_label_str),
        "evidence_kind_str": str(evidence_kind_str),
        "period_definition_str": str(period_definition_str),
        "horizon_day_int": horizon_day_int,
        "sample_count_int": int(sample_count_int),
        "simulation_path_count_int": 0,
        "terminal_return_p05_float": None,
        "terminal_return_p25_float": None,
        "terminal_return_p50_float": None,
        "terminal_return_p75_float": None,
        "terminal_return_p95_float": None,
        "terminal_loss_probability_float": None,
        "max_drawdown_p05_float": None,
        "max_drawdown_p50_float": None,
        "longest_underwater_days_p50_float": None,
        "longest_underwater_days_p95_float": None,
        "max_drawdown_recovery_days_p50_float": None,
        "max_drawdown_recovery_days_p95_float": None,
        "max_drawdown_recovered_path_count_int": 0,
        "max_drawdown_unrecovered_probability_float": None,
        "deepest_drawdown_unrecovered_probability_float": None,
        "terminal_underwater_probability_float": None,
        "underwater_ge_12m_probability_float": None,
    }


def _terminal_distribution_dict(period_return_vec: np.ndarray) -> dict[str, object]:
    clean_period_return_vec = np.asarray(period_return_vec, dtype=float)
    clean_period_return_vec = clean_period_return_vec[np.isfinite(clean_period_return_vec)]
    if clean_period_return_vec.size == 0:
        return {}
    return {
        "terminal_return_p05_float": float(np.quantile(clean_period_return_vec, 0.05)),
        "terminal_return_p25_float": float(np.quantile(clean_period_return_vec, 0.25)),
        "terminal_return_p50_float": float(np.quantile(clean_period_return_vec, 0.50)),
        "terminal_return_p75_float": float(np.quantile(clean_period_return_vec, 0.75)),
        "terminal_return_p95_float": float(np.quantile(clean_period_return_vec, 0.95)),
        "terminal_loss_probability_float": float((clean_period_return_vec < 0.0).mean()),
    }


def _path_distribution_dict(path_return_mat: np.ndarray) -> dict[str, object]:
    clean_path_return_mat = np.asarray(path_return_mat, dtype=float)
    if clean_path_return_mat.ndim != 2 or clean_path_return_mat.shape[0] == 0:
        return {}

    horizon_day_int = int(clean_path_return_mat.shape[1])
    terminal_return_vec = np.prod(1.0 + clean_path_return_mat, axis=1) - 1.0
    path_metric_dict_list = [
        _path_horizon_metric_dict(path_return_vec, horizon_day_int)
        for path_return_vec in clean_path_return_mat
    ]
    max_drawdown_vec = np.asarray(
        [metric_dict["max_drawdown_float"] for metric_dict in path_metric_dict_list],
        dtype=float,
    )
    longest_underwater_days_vec = np.asarray(
        [metric_dict["longest_underwater_days_float"] for metric_dict in path_metric_dict_list],
        dtype=float,
    )
    recovery_days_vec = np.asarray(
        [
            metric_dict["max_drawdown_recovery_days_float"]
            for metric_dict in path_metric_dict_list
            if metric_dict["max_drawdown_recovery_days_float"] is not None
        ],
        dtype=float,
    )
    unrecovered_bool_vec = np.asarray(
        [metric_dict["max_drawdown_unrecovered_bool"] for metric_dict in path_metric_dict_list],
        dtype=bool,
    )
    terminal_underwater_bool_vec = np.asarray(
        [metric_dict["terminal_underwater_bool"] for metric_dict in path_metric_dict_list],
        dtype=bool,
    )
    metric_summary_dict = _terminal_distribution_dict(terminal_return_vec)
    metric_summary_dict.update(
        {
            "max_drawdown_p05_float": float(np.quantile(max_drawdown_vec, 0.05)),
            "max_drawdown_p50_float": float(np.quantile(max_drawdown_vec, 0.50)),
            "longest_underwater_days_p50_float": float(
                np.quantile(longest_underwater_days_vec, 0.50)
            ),
            "longest_underwater_days_p95_float": float(
                np.quantile(longest_underwater_days_vec, 0.95)
            ),
            "max_drawdown_recovery_days_p50_float": (
                float(np.quantile(recovery_days_vec, 0.50))
                if recovery_days_vec.size
                else None
            ),
            "max_drawdown_recovery_days_p95_float": (
                float(np.quantile(recovery_days_vec, 0.95))
                if recovery_days_vec.size
                else None
            ),
            "max_drawdown_recovered_path_count_int": int(recovery_days_vec.size),
            "max_drawdown_unrecovered_probability_float": float(
                unrecovered_bool_vec.mean()
            ),
            "deepest_drawdown_unrecovered_probability_float": float(
                unrecovered_bool_vec.mean()
            ),
            "terminal_underwater_probability_float": float(
                terminal_underwater_bool_vec.mean()
            ),
            "underwater_ge_12m_probability_float": float(
                (
                    longest_underwater_days_vec
                    >= float(12 * TRADING_DAYS_PER_MONTH_INT)
                ).mean()
            ),
        }
    )
    return metric_summary_dict


def _build_investor_summary_dict(
    *,
    realized_return_ser: pd.Series,
    investor_scenario_df: pd.DataFrame,
    source_entity_type_str: str,
    analysis_context_dict: dict[str, object],
) -> dict[str, object]:
    scenario_record_list = _records_from_df(investor_scenario_df)
    scenario_by_key_dict = {
        str(scenario_dict.get("scenario_key_str")): scenario_dict
        for scenario_dict in scenario_record_list
    }

    def scenario_value_obj(scenario_key_str: str, field_str: str):
        return scenario_by_key_dict.get(scenario_key_str, {}).get(field_str)

    return {
        "schema_version_int": RISK_ANALYSIS_SCHEMA_VERSION_INT,
        "status_str": "historically_conditioned_not_forecast",
        "source_entity_type_str": str(source_entity_type_str),
        "month_definition_dict": {
            "modeled_month_str": "21_trading_days",
            "observed_month_str": "calendar_month",
        },
        "sample_window_dict": {
            "start_date_str": (
                _date_or_none_str(realized_return_ser.index.min())
                if isinstance(realized_return_ser.index, pd.DatetimeIndex)
                else None
            ),
            "end_date_str": (
                _date_or_none_str(realized_return_ser.index.max())
                if isinstance(realized_return_ser.index, pd.DatetimeIndex)
                else None
            ),
            "realized_trading_day_count_int": int(len(realized_return_ser)),
        },
        "analysis_context_dict": dict(analysis_context_dict),
        "headline_metric_dict": {
            "modeled_21d_typical_low_p25_float": scenario_value_obj(
                "modeled_21d", "terminal_return_p25_float"
            ),
            "modeled_21d_typical_high_p75_float": scenario_value_obj(
                "modeled_21d", "terminal_return_p75_float"
            ),
            "modeled_21d_bad_case_p05_float": scenario_value_obj(
                "modeled_21d", "terminal_return_p05_float"
            ),
            "modeled_21d_loss_probability_float": scenario_value_obj(
                "modeled_21d", "terminal_loss_probability_float"
            ),
            "observed_calendar_month_loss_probability_float": scenario_value_obj(
                "observed_calendar_month", "terminal_loss_probability_float"
            ),
            "modeled_1y_terminal_p05_block_specific_float": scenario_value_obj(
                "modeled_1y", "terminal_return_p05_float"
            ),
            "modeled_3y_bad_drawdown_p05_float": scenario_value_obj(
                "modeled_3y", "max_drawdown_p05_float"
            ),
            "modeled_3y_underwater_ge_12m_probability_float": scenario_value_obj(
                "modeled_3y", "underwater_ge_12m_probability_float"
            ),
            "modeled_3y_recovery_days_p50_conditional_float": scenario_value_obj(
                "modeled_3y", "max_drawdown_recovery_days_p50_float"
            ),
            "modeled_3y_recovery_days_p95_conditional_float": scenario_value_obj(
                "modeled_3y", "max_drawdown_recovery_days_p95_float"
            ),
            "modeled_3y_deepest_drawdown_unrecovered_probability_float": scenario_value_obj(
                "modeled_3y", "deepest_drawdown_unrecovered_probability_float"
            ),
            "modeled_3y_terminal_underwater_probability_float": scenario_value_obj(
                "modeled_3y", "terminal_underwater_probability_float"
            ),
        },
        "scenario_list": scenario_record_list,
        "limitations_list": [
            "Bootstrap rows resample dependent blocks from realized daily returns with replacement; they can duplicate or omit observations and do not forecast unseen regimes.",
            "Bootstrapping does not correct source-backtest lookahead, survivorship, corporate-action, price-adjustment, strategy-selection, or data-mining defects.",
            "Observed calendar-month rows include partial boundary months when the sample starts or ends mid-month.",
            "Modeled 21-day periods have no simulated calendar dates and are not observed calendar months.",
            "Recovery-day percentiles are conditional on recovery of the last deepest-drawdown episode inside the stated horizon; its unrecovered path share is reported separately.",
            "No claim should be used in investor materials until the offered portfolio, costs, and legal structure are approved.",
            "Each modeled horizon row uses one mean block length and is model-specific; block-length sensitivity must accompany any horizon claim.",
        ],
    }


def _observed_percentile_float(
    bootstrap_metric_ser: pd.Series,
    observed_value_float: float | None,
) -> float | None:
    if observed_value_float is None:
        return None
    finite_metric_vec = (
        bootstrap_metric_ser.astype(float)
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .to_numpy(dtype=float)
    )
    if finite_metric_vec.size == 0:
        return None
    observed_value_float = float(observed_value_float)
    less_or_tied_bool_vec = (finite_metric_vec < observed_value_float) | np.isclose(
        finite_metric_vec,
        observed_value_float,
        rtol=1e-12,
        atol=1e-14,
    )
    return float(less_or_tied_bool_vec.mean())


def save_risk_analysis_results(
    risk_result_obj: RiskAnalysisResult,
    output_dir_str: str = "results",
) -> Path:
    output_dir_path = build_research_output_path(
        output_dir_str,
        risk_result_obj.source_entity_type_str,
        risk_result_obj.strategy_name_str,
        RISK_ANALYSIS_TYPE_STR,
    )
    output_dir_path.mkdir(parents=True, exist_ok=True)

    risk_result_obj.return_histogram_df.to_csv(
        output_dir_path / RETURN_HISTOGRAM_CSV_FILENAME_STR,
        index=False,
    )
    risk_result_obj.bootstrap_equity_path_df.to_csv(
        output_dir_path / BOOTSTRAP_EQUITY_PATH_CSV_FILENAME_STR,
        index=False,
    )
    risk_result_obj.bootstrap_path_metric_df.to_csv(
        output_dir_path / BOOTSTRAP_PATH_METRIC_CSV_FILENAME_STR,
        index=False,
    )
    risk_result_obj.bootstrap_interval_df.to_csv(
        output_dir_path / BOOTSTRAP_INTERVAL_CSV_FILENAME_STR,
        index=False,
    )
    risk_result_obj.horizon_probability_df.to_csv(
        output_dir_path / HORIZON_PROBABILITY_CSV_FILENAME_STR,
        index=False,
    )
    risk_result_obj.observed_calendar_month_df.to_csv(
        output_dir_path / OBSERVED_CALENDAR_MONTH_CSV_FILENAME_STR,
        index=False,
    )
    risk_result_obj.investor_scenario_df.to_csv(
        output_dir_path / INVESTOR_SCENARIO_CSV_FILENAME_STR,
        index=False,
    )

    _write_json_file(output_dir_path / SUMMARY_FILENAME_STR, risk_result_obj.summary_dict)
    _write_json_file(
        output_dir_path / INVESTOR_SUMMARY_FILENAME_STR,
        risk_result_obj.investor_summary_dict,
    )
    _write_json_file(output_dir_path / RUN_INFO_FILENAME_STR, _build_run_info_dict(risk_result_obj))
    _write_json_file(output_dir_path / METADATA_FILENAME_STR, _build_metadata_dict(risk_result_obj))
    # Render inside the active signature variant so the analyzer page matches
    # the reports it sits beside.
    with signature_variant_context(_ACTIVE_REPORT_VARIANT_STR):
        report_html_str = _build_report_html_str(risk_result_obj)
    (output_dir_path / REPORT_FILENAME_STR).write_text(report_html_str, encoding="utf-8")

    risk_result_obj.output_dir_path = output_dir_path
    return output_dir_path


class RiskAnalysis:
    def __init__(
        self,
        strategy_obj: object,
        *,
        source_strategy_ref_str: str = "",
        source_entity_type_str: str = "strategy",
        analysis_context_dict: dict[str, object] | None = None,
        output_dir_str: str = "results",
        save_output_bool: bool = True,
        primary_mean_block_length_int: int = DEFAULT_PRIMARY_MEAN_BLOCK_LENGTH_INT,
        mean_block_length_tuple: Sequence[int] = DEFAULT_SENSITIVITY_BLOCK_LENGTH_TUPLE,
        simulation_count_int: int = DEFAULT_SIMULATION_COUNT_INT,
        random_seed_int: int = DEFAULT_RANDOM_SEED_INT,
        confidence_level_float: float = DEFAULT_CONFIDENCE_LEVEL_FLOAT,
        drawdown_threshold_tuple: Sequence[float] = DEFAULT_DRAWDOWN_THRESHOLD_TUPLE,
        upside_threshold_tuple: Sequence[float] = DEFAULT_UPSIDE_THRESHOLD_TUPLE,
        horizon_year_tuple: Sequence[int] = DEFAULT_HORIZON_YEAR_TUPLE,
        rolling_loss_window_tuple: Sequence[int] = DEFAULT_ROLLING_LOSS_WINDOW_TUPLE,
        time_underwater_breach_month_tuple: Sequence[int] = DEFAULT_TIME_UNDERWATER_BREACH_MONTH_TUPLE,
        investor_horizon_year_tuple: Sequence[int] = DEFAULT_INVESTOR_HORIZON_YEAR_TUPLE,
    ):
        self.strategy_obj = strategy_obj
        self.source_strategy_ref_str = str(source_strategy_ref_str)
        self.source_entity_type_str = str(source_entity_type_str).strip().lower()
        if self.source_entity_type_str not in {"strategy", "portfolio"}:
            raise ValueError("source_entity_type_str must be 'strategy' or 'portfolio'.")
        self.analysis_context_dict = dict(analysis_context_dict or {})
        self.output_dir_str = str(output_dir_str)
        self.save_output_bool = bool(save_output_bool)
        self.primary_mean_block_length_int = int(primary_mean_block_length_int)
        self.mean_block_length_tuple = _normalized_block_length_tuple(
            mean_block_length_tuple,
            self.primary_mean_block_length_int,
        )
        self.simulation_count_int = int(simulation_count_int)
        self.random_seed_int = int(random_seed_int)
        self.confidence_level_float = float(confidence_level_float)
        self.drawdown_threshold_tuple = tuple(float(value_float) for value_float in drawdown_threshold_tuple)
        self.upside_threshold_tuple = tuple(float(value_float) for value_float in upside_threshold_tuple)
        self.horizon_year_tuple = _normalized_positive_int_tuple(horizon_year_tuple, "horizon years")
        self.rolling_loss_window_tuple = tuple(int(value_int) for value_int in rolling_loss_window_tuple)
        self.time_underwater_breach_month_tuple = _normalized_positive_int_tuple(
            time_underwater_breach_month_tuple,
            "time-underwater months",
        )
        self.investor_horizon_year_tuple = _normalized_positive_int_tuple(
            investor_horizon_year_tuple,
            "investor horizon years",
        )

    def run(self) -> RiskAnalysisResult:
        realized_return_ser = extract_realized_return_ser(self.strategy_obj)
        return_histogram_df = build_return_histogram_df(
            realized_return_ser,
        )
        bootstrap_equity_path_df = build_bootstrap_equity_path_df(
            realized_return_ser=realized_return_ser,
            mean_block_length_int=self.primary_mean_block_length_int,
            simulation_count_int=self.simulation_count_int,
            random_seed_int=self.random_seed_int,
        )
        bootstrap_path_metric_df = build_bootstrap_path_metric_df(
            realized_return_ser=realized_return_ser,
            mean_block_length_tuple=self.mean_block_length_tuple,
            simulation_count_int=self.simulation_count_int,
            random_seed_int=self.random_seed_int,
            rolling_loss_window_tuple=self.rolling_loss_window_tuple,
        )
        observed_metric_dict = compute_path_metric_dict(
            realized_return_ser.to_numpy(dtype=float),
            rolling_loss_window_tuple=self.rolling_loss_window_tuple,
        )
        bootstrap_interval_df = build_bootstrap_interval_df(
            bootstrap_path_metric_df=bootstrap_path_metric_df,
            observed_metric_dict=observed_metric_dict,
            confidence_level_float=self.confidence_level_float,
        )
        horizon_probability_df = build_horizon_probability_df(
            realized_return_ser=realized_return_ser,
            mean_block_length_int=self.primary_mean_block_length_int,
            simulation_count_int=self.simulation_count_int,
            random_seed_int=self.random_seed_int,
            horizon_year_tuple=self.horizon_year_tuple,
            drawdown_threshold_tuple=self.drawdown_threshold_tuple,
            upside_threshold_tuple=self.upside_threshold_tuple,
            time_underwater_breach_month_tuple=self.time_underwater_breach_month_tuple,
        )
        observed_calendar_month_df = (
            build_observed_calendar_month_df(realized_return_ser)
            if isinstance(realized_return_ser.index, pd.DatetimeIndex)
            else pd.DataFrame(columns=OBSERVED_CALENDAR_MONTH_COLUMN_TUPLE)
        )
        investor_scenario_df = build_investor_scenario_df(
            realized_return_ser=realized_return_ser,
            horizon_probability_df=horizon_probability_df,
            mean_block_length_int=self.primary_mean_block_length_int,
            simulation_count_int=self.simulation_count_int,
            random_seed_int=self.random_seed_int,
            investor_horizon_year_tuple=self.investor_horizon_year_tuple,
        )
        investor_summary_dict = _build_investor_summary_dict(
            realized_return_ser=realized_return_ser,
            investor_scenario_df=investor_scenario_df,
            source_entity_type_str=self.source_entity_type_str,
            analysis_context_dict=self.analysis_context_dict,
        )
        summary_dict = _build_summary_dict(
            strategy_obj=self.strategy_obj,
            realized_return_ser=realized_return_ser,
            bootstrap_path_metric_df=bootstrap_path_metric_df,
            bootstrap_interval_df=bootstrap_interval_df,
            horizon_probability_df=horizon_probability_df,
            observed_metric_dict=observed_metric_dict,
            primary_mean_block_length_int=self.primary_mean_block_length_int,
            mean_block_length_tuple=self.mean_block_length_tuple,
            simulation_count_int=self.simulation_count_int,
            random_seed_int=self.random_seed_int,
            confidence_level_float=self.confidence_level_float,
            drawdown_threshold_tuple=self.drawdown_threshold_tuple,
            upside_threshold_tuple=self.upside_threshold_tuple,
            horizon_year_tuple=self.horizon_year_tuple,
            time_underwater_breach_month_tuple=self.time_underwater_breach_month_tuple,
        )
        summary_dict["schema_version_int"] = RISK_ANALYSIS_SCHEMA_VERSION_INT
        summary_dict["investor_horizon_year_list"] = [
            int(value_int) for value_int in self.investor_horizon_year_tuple
        ]
        summary_dict["time_underwater_breach_month_list"] = [
            int(value_int) for value_int in self.time_underwater_breach_month_tuple
        ]
        summary_dict["investor_summary"] = investor_summary_dict

        risk_result_obj = RiskAnalysisResult(
            strategy_name_str=str(self.strategy_obj.name),
            source_strategy_ref_str=self.source_strategy_ref_str,
            realized_return_ser=realized_return_ser,
            return_histogram_df=return_histogram_df,
            bootstrap_equity_path_df=bootstrap_equity_path_df,
            bootstrap_path_metric_df=bootstrap_path_metric_df,
            bootstrap_interval_df=bootstrap_interval_df,
            horizon_probability_df=horizon_probability_df,
            observed_calendar_month_df=observed_calendar_month_df,
            investor_scenario_df=investor_scenario_df,
            investor_summary_dict=investor_summary_dict,
            source_entity_type_str=self.source_entity_type_str,
            analysis_context_dict=self.analysis_context_dict,
            summary_dict=summary_dict,
        )
        if self.save_output_bool:
            save_risk_analysis_results(risk_result_obj, output_dir_str=self.output_dir_str)
        return risk_result_obj


def _normalized_block_length_tuple(
    raw_block_length_tuple: Sequence[int],
    primary_mean_block_length_int: int,
) -> tuple[int, ...]:
    block_length_list = [int(primary_mean_block_length_int)]
    block_length_list.extend(int(value_int) for value_int in raw_block_length_tuple)
    normalized_list = []
    for block_length_int in block_length_list:
        if block_length_int <= 0:
            raise ValueError("Block lengths must be positive.")
        if block_length_int not in normalized_list:
            normalized_list.append(block_length_int)
    return tuple(normalized_list)


def _normalized_positive_int_tuple(
    raw_value_tuple: Sequence[int],
    label_str: str,
) -> tuple[int, ...]:
    normalized_list = []
    for value_int in raw_value_tuple:
        normalized_value_int = int(value_int)
        if normalized_value_int <= 0:
            raise ValueError(f"{label_str} must be positive.")
        if normalized_value_int not in normalized_list:
            normalized_list.append(normalized_value_int)
    if len(normalized_list) == 0:
        raise ValueError(f"At least one {label_str} value is required.")
    return tuple(normalized_list)


def _build_summary_dict(
    *,
    strategy_obj: object,
    realized_return_ser: pd.Series,
    bootstrap_path_metric_df: pd.DataFrame,
    bootstrap_interval_df: pd.DataFrame,
    horizon_probability_df: pd.DataFrame,
    observed_metric_dict: dict[str, object],
    primary_mean_block_length_int: int,
    mean_block_length_tuple: Sequence[int],
    simulation_count_int: int,
    random_seed_int: int,
    confidence_level_float: float,
    drawdown_threshold_tuple: Sequence[float],
    upside_threshold_tuple: Sequence[float],
    horizon_year_tuple: Sequence[int],
    time_underwater_breach_month_tuple: Sequence[int],
) -> dict[str, object]:
    primary_metric_df = bootstrap_path_metric_df[
        bootstrap_path_metric_df["mean_block_length_int"] == int(primary_mean_block_length_int)
    ]
    primary_interval_df = bootstrap_interval_df[
        bootstrap_interval_df["mean_block_length_int"] == int(primary_mean_block_length_int)
    ]
    summary_dict: dict[str, object] = {
        "analysis_type": RISK_ANALYSIS_TYPE_STR,
        "strategy_name_str": str(strategy_obj.name),
        "return_count_int": int(len(realized_return_ser)),
        "start_date_str": _date_or_none_str(realized_return_ser.index.min()),
        "end_date_str": _date_or_none_str(realized_return_ser.index.max()),
        "primary_mean_block_length_int": int(primary_mean_block_length_int),
        "mean_block_length_list": [int(value_int) for value_int in mean_block_length_tuple],
        "simulation_count_int": int(simulation_count_int),
        "random_seed_int": int(random_seed_int),
        "confidence_level_float": float(confidence_level_float),
        "horizon_year_list": [int(value_int) for value_int in horizon_year_tuple],
        "drawdown_threshold_list": [float(value_float) for value_float in drawdown_threshold_tuple],
        "upside_threshold_list": [float(value_float) for value_float in upside_threshold_tuple],
        "observed_metrics": _compact_dict(observed_metric_dict),
        "primary_intervals": _primary_interval_dict(primary_interval_df),
        "primary_horizon_probabilities": _records_from_df(horizon_probability_df),
        "primary_drawdown_breach_probabilities": _drawdown_breach_probability_dict(
            primary_metric_df,
            drawdown_threshold_tuple,
        ),
        "primary_time_underwater_breach_probabilities": _time_underwater_breach_probability_dict(
            primary_metric_df,
            time_underwater_breach_month_tuple,
        ),
        "primary_terminal_loss_probability_float": _terminal_loss_probability_float(primary_metric_df),
    }
    summary_dict["verdict"] = _build_verdict_row_list(summary_dict)
    return summary_dict


def _path_horizon_metric_dict(
    return_vec: np.ndarray,
    horizon_day_int: int,
) -> dict[str, object]:
    clean_return_vec = np.asarray(return_vec, dtype=float)
    clean_return_vec = clean_return_vec[np.isfinite(clean_return_vec)]
    if horizon_day_int <= 0:
        raise ValueError("horizon_day_int must be positive.")
    if clean_return_vec.size < horizon_day_int:
        return {
            "max_drawdown_float": np.nan,
            "max_gain_float": np.nan,
            "terminal_return_float": np.nan,
            "longest_underwater_days_float": np.nan,
            "max_drawdown_recovery_days_float": None,
            "max_drawdown_unrecovered_bool": True,
            "terminal_underwater_bool": True,
        }

    horizon_return_vec = clean_return_vec[: int(horizon_day_int)]
    horizon_equity_vec = np.cumprod(1.0 + horizon_return_vec)
    equity_with_start_vec = np.concatenate(([1.0], horizon_equity_vec))
    terminal_return_float = float(horizon_equity_vec[-1] - 1.0)

    # *** CRITICAL*** horizon drawdown is path-local and starts from fresh
    # equity 1.0: drawdown_t = E_t / max(1, E_1, ..., E_t) - 1. It is a
    # report-only bootstrap diagnostic and must never feed order or sizing.
    running_peak_with_start_vec = np.maximum.accumulate(equity_with_start_vec)
    drawdown_with_start_vec = equity_with_start_vec / running_peak_with_start_vec - 1.0
    drawdown_vec = drawdown_with_start_vec[1:]
    max_drawdown_float = float(np.min(drawdown_vec))
    longest_underwater_days_float = float(_longest_underwater_days_int(drawdown_vec))

    # *** CRITICAL*** report-only recovery measurement: recovery is measured
    # from the peak that precedes the path's deepest drawdown to the first later
    # day at or above that peak. If the path ends first, the duration is
    # right-censored and remains None; it must not be counted as a fast recovery.
    if np.isclose(max_drawdown_float, 0.0):
        max_drawdown_recovery_days_float: float | None = 0.0
        max_drawdown_unrecovered_bool = False
    else:
        deepest_drawdown_position_vec = np.flatnonzero(
            np.isclose(
                drawdown_with_start_vec,
                max_drawdown_float,
                rtol=1e-12,
                atol=1e-12,
            )
        )
        # When the same deepest drawdown recurs, use its last occurrence so a
        # final unrecovered repeat is not hidden by an earlier recovered episode.
        trough_position_int = int(deepest_drawdown_position_vec[-1])
        preceding_peak_float = float(running_peak_with_start_vec[trough_position_int])
        preceding_peak_position_vec = np.flatnonzero(
            np.isclose(
                equity_with_start_vec[: trough_position_int + 1],
                preceding_peak_float,
                rtol=1e-12,
                atol=1e-12,
            )
        )
        preceding_peak_position_int = int(preceding_peak_position_vec[-1])
        recovery_position_vec = np.flatnonzero(
            equity_with_start_vec[trough_position_int + 1 :]
            >= preceding_peak_float * (1.0 - 1e-12)
        )
        if recovery_position_vec.size == 0:
            max_drawdown_recovery_days_float = None
            max_drawdown_unrecovered_bool = True
        else:
            recovery_position_int = (
                int(trough_position_int) + 1 + int(recovery_position_vec[0])
            )
            max_drawdown_recovery_days_float = float(
                recovery_position_int - preceding_peak_position_int
            )
            max_drawdown_unrecovered_bool = False

    return {
        "max_drawdown_float": max_drawdown_float,
        "max_gain_float": float(np.max(equity_with_start_vec) - 1.0),
        "terminal_return_float": terminal_return_float,
        "longest_underwater_days_float": longest_underwater_days_float,
        "max_drawdown_recovery_days_float": max_drawdown_recovery_days_float,
        "max_drawdown_unrecovered_bool": max_drawdown_unrecovered_bool,
        "terminal_underwater_bool": bool(drawdown_vec[-1] < -1e-12),
    }


def _band_status_str(
    value_float: object,
    green_max_float: float,
    amber_max_float: float,
) -> str:
    """
    Map a non-negative magnitude to a reference-band status.

    value <= green_max -> green; <= amber_max -> amber; else red. None/NaN -> na.
    Callers pass magnitudes where larger is worse (e.g. loss probability,
    |drawdown|), so the bands read left-to-right from best to worst.
    """
    band_value_float = _json_float(value_float)
    if band_value_float is None:
        return VERDICT_STATUS_NA_STR
    if band_value_float <= green_max_float:
        return VERDICT_STATUS_GREEN_STR
    if band_value_float <= amber_max_float:
        return VERDICT_STATUS_AMBER_STR
    return VERDICT_STATUS_RED_STR


def _interval_value_float(summary_dict: dict[str, object], metric_name_str: str, field_str: str):
    primary_interval_dict = summary_dict.get("primary_intervals", {})
    if not isinstance(primary_interval_dict, dict):
        return None
    metric_dict = primary_interval_dict.get(metric_name_str, {})
    if not isinstance(metric_dict, dict):
        return None
    return _json_float(metric_dict.get(field_str))


def _build_verdict_row_list(summary_dict: dict[str, object]) -> list[dict[str, str]]:
    row_list: list[dict[str, str]] = []

    # --- Historical resampling diagnostic. This is not independent edge proof. ---
    terminal_loss_float = _json_float(summary_dict.get("primary_terminal_loss_probability_float"))
    sharpe_ci_lower_float = _interval_value_float(summary_dict, "sharpe_float", "ci_lower_float")
    edge_status_str = _band_status_str(
        terminal_loss_float,
        EDGE_TERMINAL_LOSS_GREEN_MAX_FLOAT,
        EDGE_TERMINAL_LOSS_AMBER_MAX_FLOAT,
    )
    if (
        edge_status_str == VERDICT_STATUS_GREEN_STR
        and sharpe_ci_lower_float is not None
        and sharpe_ci_lower_float <= 0.0
    ):
        # Profitable most bootstrap paths, but Sharpe CI still includes zero.
        edge_status_str = VERDICT_STATUS_AMBER_STR
    if terminal_loss_float is None:
        edge_value_str = "N/A"
        edge_conclusion_str = "Not enough data for the historical resampling diagnostic."
    else:
        profit_probability_float = 1.0 - terminal_loss_float
        edge_value_str = f"{profit_probability_float:.0%} profitable bootstrap paths"
        if edge_status_str == VERDICT_STATUS_GREEN_STR:
            edge_conclusion_str = (
                f"Ended profitable in {profit_probability_float:.0%} of historically conditioned bootstrap paths."
            )
        elif edge_status_str == VERDICT_STATUS_AMBER_STR:
            edge_conclusion_str = (
                f"Profitable in {profit_probability_float:.0%} of historically conditioned bootstrap paths; "
                "this does not independently validate edge."
            )
        else:
            edge_conclusion_str = (
                f"Profitable in {profit_probability_float:.0%} of historically conditioned bootstrap paths."
            )
    row_list.append(
        {
            "label_str": "Historical resampling",
            "status_str": edge_status_str,
            "value_str": edge_value_str,
            "conclusion_str": edge_conclusion_str,
        }
    )

    # --- Drawdown depth: p05 of the bootstrap max-drawdown distribution. ---
    drawdown_bad_float = _interval_value_float(summary_dict, "max_drawdown_float", "p05_float")
    drawdown_magnitude_float = abs(drawdown_bad_float) if drawdown_bad_float is not None else None
    drawdown_status_str = _band_status_str(
        drawdown_magnitude_float,
        DRAWDOWN_DEPTH_GREEN_MAX_FLOAT,
        DRAWDOWN_DEPTH_AMBER_MAX_FLOAT,
    )
    if drawdown_bad_float is None:
        drawdown_value_str = "N/A"
        drawdown_conclusion_str = "Not enough data to assess drawdown."
    else:
        drawdown_value_str = f"{drawdown_bad_float:.0%} (bootstrap p05)"
        drawdown_conclusion_str = (
            f"Historically conditioned bootstrap p05 max drawdown was {drawdown_bad_float:.0%}."
        )
    row_list.append(
        {
            "label_str": "Drawdown depth",
            "status_str": drawdown_status_str,
            "value_str": drawdown_value_str,
            "conclusion_str": drawdown_conclusion_str,
        }
    )

    # --- Time underwater: probability of a 12-month+ underwater stretch. ---
    underwater_dict = summary_dict.get("primary_time_underwater_breach_probabilities", {})
    underwater_12m_float = (
        _json_float(underwater_dict.get("underwater_ge_12m"))
        if isinstance(underwater_dict, dict)
        else None
    )
    underwater_status_str = _band_status_str(
        underwater_12m_float,
        UNDERWATER_12M_GREEN_MAX_FLOAT,
        UNDERWATER_12M_AMBER_MAX_FLOAT,
    )
    if underwater_12m_float is None:
        underwater_value_str = "N/A"
        underwater_conclusion_str = "Not enough data to assess time underwater."
    else:
        underwater_value_str = (
            f"{underwater_12m_float:.0%} of bootstrap paths had >= 12m underwater"
        )
        underwater_conclusion_str = (
            f"A 12-month+ underwater stretch occurred in {underwater_12m_float:.0%} "
            "of historically conditioned bootstrap paths."
        )
    row_list.append(
        {
            "label_str": "Time underwater",
            "status_str": underwater_status_str,
            "value_str": underwater_value_str,
            "conclusion_str": underwater_conclusion_str,
        }
    )

    # --- Worst year: p05 worst rolling 12-month return across bootstrap paths. ---
    worst_year_bad_float = _interval_value_float(summary_dict, "worst_252d_return_float", "p05_float")
    worst_year_magnitude_float = (
        abs(worst_year_bad_float) if worst_year_bad_float is not None else None
    )
    worst_year_status_str = _band_status_str(
        worst_year_magnitude_float,
        WORST_YEAR_GREEN_MAX_FLOAT,
        WORST_YEAR_AMBER_MAX_FLOAT,
    )
    if worst_year_bad_float is None:
        worst_year_value_str = "N/A"
        worst_year_conclusion_str = "Not enough history for a 12-month window."
    else:
        worst_year_value_str = f"{worst_year_bad_float:.0%} (bootstrap p05)"
        worst_year_conclusion_str = (
            f"Bootstrap p05 worst rolling 12-month return was {worst_year_bad_float:.0%}."
        )
    row_list.append(
        {
            "label_str": "Worst year",
            "status_str": worst_year_status_str,
            "value_str": worst_year_value_str,
            "conclusion_str": worst_year_conclusion_str,
        }
    )

    return row_list


def _primary_interval_dict(primary_interval_df: pd.DataFrame) -> dict[str, dict[str, float]]:
    interval_dict: dict[str, dict[str, float]] = {}
    for _, row_ser in primary_interval_df.iterrows():
        metric_name_str = str(row_ser["metric_name_str"])
        interval_dict[metric_name_str] = {
            "observed_value_float": _json_float(row_ser.get("observed_value_float")),
            "bootstrap_mean_float": _json_float(row_ser.get("bootstrap_mean_float")),
            "ci_half_width_float": _json_float(row_ser.get("ci_half_width_float")),
            "ci_lower_float": _json_float(row_ser.get("ci_lower_float")),
            "ci_upper_float": _json_float(row_ser.get("ci_upper_float")),
            "p05_float": _json_float(row_ser.get("p05_float")),
            "p50_float": _json_float(row_ser.get("p50_float")),
            "p95_float": _json_float(row_ser.get("p95_float")),
        }
    return interval_dict


def _drawdown_breach_probability_dict(
    metric_df: pd.DataFrame,
    drawdown_threshold_tuple: Sequence[float],
) -> dict[str, float | None]:
    if len(metric_df) == 0 or "max_drawdown_float" not in metric_df.columns:
        return {}
    max_drawdown_ser = metric_df["max_drawdown_float"].astype(float)
    return {
        f"max_drawdown_lte_{abs(float(threshold_float)):.0%}": _json_float(
            (max_drawdown_ser <= float(threshold_float)).mean()
        )
        for threshold_float in drawdown_threshold_tuple
    }


def _time_underwater_breach_probability_dict(
    metric_df: pd.DataFrame,
    breach_month_tuple: Sequence[int],
) -> dict[str, float | None]:
    if len(metric_df) == 0 or "longest_underwater_days_float" not in metric_df.columns:
        return {}
    underwater_day_ser = metric_df["longest_underwater_days_float"].astype(float)
    breach_dict: dict[str, float | None] = {}
    for month_int in breach_month_tuple:
        threshold_day_int = int(month_int) * TRADING_DAYS_PER_MONTH_INT
        breach_dict[f"underwater_ge_{int(month_int)}m"] = _json_float(
            (underwater_day_ser >= float(threshold_day_int)).mean()
        )
    return breach_dict


def _terminal_loss_probability_float(metric_df: pd.DataFrame) -> float | None:
    if len(metric_df) == 0 or "terminal_return_float" not in metric_df.columns:
        return None
    terminal_return_ser = metric_df["terminal_return_float"].astype(float)
    return _json_float((terminal_return_ser < 0.0).mean())


def _threshold_column_name_str(prefix_str: str, threshold_float: float) -> str:
    threshold_pct_float = abs(float(threshold_float)) * 100.0
    if np.isclose(threshold_pct_float, round(threshold_pct_float)):
        threshold_label_str = f"{int(round(threshold_pct_float))}pct"
    else:
        threshold_label_str = f"{threshold_pct_float:.2f}".rstrip("0").rstrip(".").replace(".", "p") + "pct"
    return f"{prefix_str}_{threshold_label_str}_probability_float"


def _monthly_return_vec_from_daily_float(daily_return_vec: np.ndarray) -> np.ndarray:
    """
    Compound non-overlapping 21-trading-day chunks into a monthly return vec.

    Trailing days that do not fill a full 21-day chunk are dropped. This keeps
    observed and bootstrap-simulated monthly returns constructed identically.
    """
    clean_daily_vec = np.asarray(daily_return_vec, dtype=float)
    clean_daily_vec = clean_daily_vec[np.isfinite(clean_daily_vec)]
    full_month_count_int = int(clean_daily_vec.size // TRADING_DAYS_PER_MONTH_INT)
    if full_month_count_int == 0:
        return np.empty((0,), dtype=float)
    usable_day_count_int = full_month_count_int * TRADING_DAYS_PER_MONTH_INT
    chunk_mat = clean_daily_vec[:usable_day_count_int].reshape(
        full_month_count_int, TRADING_DAYS_PER_MONTH_INT
    )
    monthly_return_vec = np.prod(1.0 + chunk_mat, axis=1) - 1.0
    return monthly_return_vec.astype(float)


def _longest_underwater_days_int(drawdown_vec: np.ndarray) -> int:
    """
    Return the longest consecutive run of days where drawdown < 0.

    Treats a flat-at-peak day (drawdown == 0) as a recovery point, ending the
    current underwater run. Robust to non-finite values (treated as recovered).
    """
    drawdown_arr = np.asarray(drawdown_vec, dtype=float)
    longest_run_int = 0
    current_run_int = 0
    for value_float in drawdown_arr:
        if np.isfinite(value_float) and value_float < 0.0:
            current_run_int += 1
            if current_run_int > longest_run_int:
                longest_run_int = current_run_int
        else:
            current_run_int = 0
    return int(longest_run_int)


def _var_float(value_vec: np.ndarray, alpha_float: float) -> float:
    clean_value_vec = np.asarray(value_vec, dtype=float)
    clean_value_vec = clean_value_vec[np.isfinite(clean_value_vec)]
    if clean_value_vec.size == 0:
        return np.nan
    return float(np.quantile(clean_value_vec, alpha_float))


def _tail_mean_float(value_vec: np.ndarray, alpha_float: float) -> float:
    clean_value_vec = np.asarray(value_vec, dtype=float)
    clean_value_vec = clean_value_vec[np.isfinite(clean_value_vec)]
    if clean_value_vec.size == 0:
        return np.nan
    quantile_float = float(np.quantile(clean_value_vec, alpha_float))
    tail_value_vec = clean_value_vec[clean_value_vec <= quantile_float]
    if tail_value_vec.size == 0:
        return np.nan
    return float(tail_value_vec.mean())


def _worst_rolling_return_float(return_vec: np.ndarray, window_int: int) -> float:
    clean_return_vec = np.asarray(return_vec, dtype=float)
    clean_return_vec = clean_return_vec[np.isfinite(clean_return_vec)]
    if window_int <= 0 or clean_return_vec.size < window_int:
        return np.nan
    growth_vec = 1.0 + clean_return_vec
    # *** CRITICAL*** trailing report metric: cumulative_log[t+w] -
    # cumulative_log[t] is exactly the log growth over returns t..t+w-1.
    # Every window is backward/within-path only; no observation after the
    # window end enters the result. This replaces an equivalent product loop.
    if bool((growth_vec > 0.0).all()):
        cumulative_log_vec = np.concatenate(
            ([0.0], np.cumsum(np.log(growth_vec), dtype=float))
        )
        rolling_log_return_vec = (
            cumulative_log_vec[int(window_int) :]
            - cumulative_log_vec[: -int(window_int)]
        )
        rolling_return_vec = np.expm1(rolling_log_return_vec)
    else:
        # Defensive fallback for non-financial inputs containing returns <= -100%.
        rolling_growth_mat = np.lib.stride_tricks.sliding_window_view(
            growth_vec,
            int(window_int),
        )
        rolling_return_vec = np.prod(rolling_growth_mat, axis=1) - 1.0
    return float(np.min(rolling_return_vec))


def _build_run_info_dict(risk_result_obj: RiskAnalysisResult) -> dict[str, object]:
    summary_dict = risk_result_obj.summary_dict
    return {
        "entity_type": risk_result_obj.source_entity_type_str,
        "entity_id": risk_result_obj.strategy_name_str,
        "analysis_type": RISK_ANALYSIS_TYPE_STR,
        "schema_version_int": RISK_ANALYSIS_SCHEMA_VERSION_INT,
        "analysis_context": risk_result_obj.analysis_context_dict,
        "parameters": {
            "source_entity_ref": risk_result_obj.source_strategy_ref_str,
            "source_strategy_ref": risk_result_obj.source_strategy_ref_str,
            "primary_mean_block_length_int": summary_dict.get("primary_mean_block_length_int"),
            "mean_block_length_list": summary_dict.get("mean_block_length_list"),
            "simulation_count_int": summary_dict.get("simulation_count_int"),
            "random_seed_int": summary_dict.get("random_seed_int"),
            "confidence_level_float": summary_dict.get("confidence_level_float"),
            "horizon_year_list": summary_dict.get("horizon_year_list"),
            "investor_horizon_year_list": summary_dict.get(
                "investor_horizon_year_list"
            ),
            "time_underwater_breach_month_list": summary_dict.get(
                "time_underwater_breach_month_list"
            ),
            "drawdown_threshold_list": summary_dict.get("drawdown_threshold_list"),
            "upside_threshold_list": summary_dict.get("upside_threshold_list"),
        },
    }


def _build_metadata_dict(risk_result_obj: RiskAnalysisResult) -> dict[str, object]:
    return {
        "artifact_type": RISK_ANALYSIS_TYPE_STR,
        "schema_version": RISK_ANALYSIS_SCHEMA_VERSION_INT,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "entity_type": risk_result_obj.source_entity_type_str,
        "entity_name": risk_result_obj.strategy_name_str,
        "strategy_name": risk_result_obj.strategy_name_str,
        "source_entity_ref": risk_result_obj.source_strategy_ref_str,
        "source_strategy_ref": risk_result_obj.source_strategy_ref_str,
        "return_count": int(len(risk_result_obj.realized_return_ser)),
        "analysis_context": risk_result_obj.analysis_context_dict,
    }


def _full_sample_drawdown_row_dict(risk_result_obj: RiskAnalysisResult) -> dict[str, object]:
    """Assemble the full-sample-length row of the downside horizon table.

    The breach probabilities and the max-drawdown percentiles come from the
    same primary-block bootstrap as the fixed horizons, just measured over the
    whole realized sample rather than a truncated window.
    """
    summary_dict = risk_result_obj.summary_dict
    row_dict: dict[str, object] = dict(
        summary_dict.get("primary_drawdown_breach_probabilities", {}) or {}
    )
    row_dict["simulation_path_count_int"] = int(
        summary_dict.get("simulation_count_int", DEFAULT_SIMULATION_COUNT_INT)
    )

    interval_df = risk_result_obj.bootstrap_interval_df
    if interval_df is not None and len(interval_df) > 0:
        primary_block_length_int = int(summary_dict["primary_mean_block_length_int"])
        drawdown_row_df = interval_df[
            (interval_df["mean_block_length_int"] == primary_block_length_int)
            & (interval_df["metric_name_str"] == "max_drawdown_float")
        ]
        if len(drawdown_row_df) > 0:
            row_dict["max_drawdown_p50_float"] = drawdown_row_df.iloc[0].get("p50_float")
            row_dict["max_drawdown_p05_float"] = drawdown_row_df.iloc[0].get("p05_float")
    return row_dict


def _build_report_html_str(risk_result_obj: RiskAnalysisResult) -> str:
    summary_dict = risk_result_obj.summary_dict
    strategy_name_html = html.escape(risk_result_obj.strategy_name_str)
    entity_type_html = html.escape(risk_result_obj.source_entity_type_str)
    confidence_level_float = float(summary_dict.get("confidence_level_float", DEFAULT_CONFIDENCE_LEVEL_FLOAT))
    simulation_count_int = int(summary_dict.get("simulation_count_int", DEFAULT_SIMULATION_COUNT_INT))
    realized_return_ser = risk_result_obj.realized_return_ser.astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    return_mean_float = float(realized_return_ser.mean()) if len(realized_return_ser) else None
    return_median_float = float(realized_return_ser.median()) if len(realized_return_ser) else None
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{strategy_name_html} RiskAnalysis</title>
{build_report_font_head_html()}
<style>{build_analyzer_report_css()}
</style>
</head>
<body>
<div class="wrap">
<h1>{strategy_name_html} RiskAnalysis</h1>
<div class="meta">
Return window: {html.escape(str(summary_dict.get("start_date_str")))} to {html.escape(str(summary_dict.get("end_date_str")))}
| Returns: {summary_dict.get("return_count_int")}
| Primary block length: {summary_dict.get("primary_mean_block_length_int")}
| Simulations: {summary_dict.get("simulation_count_int")}
</div>
{_analysis_status_banner_html(risk_result_obj.analysis_context_dict)}
<div class="caveat">
<strong>What this report is.</strong> This stationary-bootstrap report resamples dependent blocks from the {entity_type_html}'s <em>realized</em> daily returns with replacement. Paths can duplicate or omit observations; they measure historically conditioned sampling and sequence sensitivity. <strong>What it is not.</strong> It does not validate edge independently, correct defects in the source backtest, simulate regimes outside the sample, or calibrate forward event odds. Use it as a diagnostic, not as forward stress testing or a promise.
</div>
<div class="section">
<h2>Investor Scenario Summary</h2>
<div class="subtitle">Simple historical and historically conditioned ranges under the primary block assumption. “Observed calendar month” and “modeled 21 trading days” are different definitions. Typical range = p25 to p75; p05 is a bootstrap percentile, not a calibrated forward 1-in-20 event. Horizon rows are model-specific and must be read with the separate block-length sensitivity sweep. Recovery percentiles include only paths whose last deepest-drawdown episode recovered inside the horizon; terminal-underwater path share is separate.</div>
<div class="scroll">{_investor_scenario_table_html(risk_result_obj.investor_scenario_df)}</div>
</div>
{_verdict_panel_html(summary_dict.get("verdict"))}
{_summary_tiles_html(summary_dict)}
<div class="section">
<h2>Sample Distribution</h2>
<div class="subtitle">Each bootstrap path produces its own CAGR, drawdown and Sharpe, so these are {summary_dict.get("simulation_count_int")} draws of how this {entity_type_html} could have gone under the primary block assumption. The shaded band is the 5th to 95th percentile reported in the interval table. The vertical rule is the value the realized backtest actually produced: near the middle of a broad distribution means the result is ordinary for this return series, out at an edge means it depended on the particular sequence that happened.</div>
{_sample_distribution_svg(risk_result_obj.bootstrap_path_metric_df, risk_result_obj.bootstrap_interval_df, int(summary_dict["primary_mean_block_length_int"]))}
</div>
<div class="section">
<h2>Returns Histogram</h2>
<div class="subtitle">Distribution of realized daily returns over the sample window. Mean left of median signals negative skew (occasional larger losses). Bar height is share of days, on a square-root scale so the tail stays visible against the central mass.</div>
{_return_histogram_svg(risk_result_obj.return_histogram_df, mean_float=return_mean_float, median_float=return_median_float)}
</div>
<div class="section">
<h2>Monte Carlo Equity Paths</h2>
<div class="subtitle">Simulated paths have the same length as the realized history; this shows historically conditioned resampling sensitivity, not a forward horizon.</div>
{_bootstrap_equity_svg(risk_result_obj.bootstrap_equity_path_df)}
<div class="legend">Primary block length {summary_dict.get("primary_mean_block_length_int")}; sampled bootstrap paths plus observed, p05, p50, and p95 curves.</div>
</div>
<div class="section">
<h2>Monte Carlo Metric Estimates</h2>
<div class="subtitle">Daily-horizon view. All metrics computed per bootstrap path; columns are observed value, observed percentile inside the bootstrap distribution, and the bootstrap confidence interval.</div>
<div class="scroll">{_interval_table_html(risk_result_obj.bootstrap_interval_df, int(summary_dict["primary_mean_block_length_int"]), confidence_level_float, simulation_count_int)}</div>
</div>
<div class="section">
<h2>Monthly Risk Metrics</h2>
<div class="subtitle">Monthly = 21 consecutive trading days. Computed from the same bootstrap paths as the daily metrics for direct comparability.</div>
<div class="scroll">{_interval_table_html(risk_result_obj.bootstrap_interval_df, int(summary_dict["primary_mean_block_length_int"]), confidence_level_float, simulation_count_int, metric_order_list=MONTHLY_METRIC_ORDER_LIST, label_dict=MONTHLY_METRIC_LABEL_DICT)}</div>
</div>
<div class="section">
<h2>Horizon Probability Tables</h2>
<div class="subtitle">Bootstrap-implied horizon probabilities from realized returns. Trading-year horizons use {TRADING_DAYS_PER_YEAR_INT} days. Downside is max drawdown touched inside the horizon; upside is max gain touched inside the horizon. Rows beyond the realized sample length render as N/A. The final downside row is the bootstrap at its full realized sample length, so it is a longer exposure than the fixed horizons above it rather than a higher rate.</div>
{_horizon_probability_tables_html(risk_result_obj.horizon_probability_df, summary_dict.get("drawdown_threshold_list", DEFAULT_DRAWDOWN_THRESHOLD_TUPLE), summary_dict.get("upside_threshold_list", DEFAULT_UPSIDE_THRESHOLD_TUPLE), _full_sample_drawdown_row_dict(risk_result_obj))}
</div>
<div class="section">
<h2>Time Underwater</h2>
<div class="subtitle">Probability across bootstrap paths that the event occurred at least once over the full realized sample length. Thresholds are in trading months of {TRADING_DAYS_PER_MONTH_INT} days. Drawdown-depth probabilities are the full-sample row of the downside table above.</div>
<div class="scroll">{_breach_table_html(summary_dict.get("primary_terminal_loss_probability_float"), summary_dict.get("primary_time_underwater_breach_probabilities", {}))}</div>
</div>
</div>
</body>
</html>"""


TILE_SUBTITLE_BY_METRIC_DICT = {
    "cagr_float": "annualized compound rate of the realized path",
    "sharpe_float": "return per unit of volatility (rf=0)",
    "max_drawdown_float": "deepest peak-to-trough loss observed",
    "var_95_daily_return_float": "5% of days lose more than this",
    "cvar_95_daily_return_float": "average loss on the worst 5% of days",
}


def _analysis_status_banner_html(analysis_context_dict: dict[str, object]) -> str:
    if not analysis_context_dict:
        return ""
    analysis_status_str = str(
        analysis_context_dict.get("analysis_status_str", "unspecified")
    )
    investor_use_approved_bool = bool(
        analysis_context_dict.get("investor_use_approved_bool", False)
    )
    if investor_use_approved_bool:
        approval_label_str = "INVESTOR USE APPROVED IN SOURCE CONTEXT"
    else:
        approval_label_str = "NOT APPROVED FOR INVESTOR USE"
    return (
        "<div class=\"status-banner\">"
        f"ANALYSIS STATUS: {html.escape(analysis_status_str.replace('_', ' ').upper())}"
        f" &nbsp;|&nbsp; {html.escape(approval_label_str)}"
        "</div>"
    )


def _investor_scenario_table_html(investor_scenario_df: pd.DataFrame) -> str:
    if investor_scenario_df is None or len(investor_scenario_df) == 0:
        return "<p>N/A</p>"
    header_html_str = (
        "<tr>"
        "<th>Scenario</th>"
        "<th>Evidence</th>"
        "<th>Typical p25-p75</th>"
        "<th>Bootstrap p05</th>"
        "<th>Terminal-loss path share</th>"
        "<th>Bad max DD p05</th>"
        "<th>Underwater p95</th>"
        "<th>Recovery p50*</th>"
        "<th>Ends below peak</th>"
        "<th>Last deepest DD unrecovered</th>"
        "</tr>"
    )
    row_html_list = []
    for _, scenario_ser in investor_scenario_df.iterrows():
        typical_low_str = _format_percent(scenario_ser.get("terminal_return_p25_float"))
        typical_high_str = _format_percent(scenario_ser.get("terminal_return_p75_float"))
        typical_range_str = (
            "N/A"
            if typical_low_str == "N/A" or typical_high_str == "N/A"
            else f"{typical_low_str} to {typical_high_str}"
        )
        evidence_label_str = (
            "Observed"
            if str(scenario_ser.get("evidence_kind_str")) == "observed"
            else "Bootstrap implied"
        )
        row_html_list.append(
            "<tr>"
            f"<td>{html.escape(str(scenario_ser.get('scenario_label_str', '')))}</td>"
            f"<td>{html.escape(evidence_label_str)}</td>"
            f"<td>{html.escape(typical_range_str)}</td>"
            f"<td>{html.escape(_format_percent(scenario_ser.get('terminal_return_p05_float')))}</td>"
            f"<td>{html.escape(_format_percent(scenario_ser.get('terminal_loss_probability_float')))}</td>"
            f"<td>{html.escape(_format_percent(scenario_ser.get('max_drawdown_p05_float')))}</td>"
            f"<td>{html.escape(_format_days_str(scenario_ser.get('longest_underwater_days_p95_float')))}</td>"
            f"<td>{html.escape(_format_days_str(scenario_ser.get('max_drawdown_recovery_days_p50_float')))}</td>"
            f"<td>{html.escape(_format_percent(scenario_ser.get('terminal_underwater_probability_float')))}</td>"
            f"<td>{html.escape(_format_percent(scenario_ser.get('deepest_drawdown_unrecovered_probability_float')))}</td>"
            "</tr>"
        )
    return (
        f"<table><thead>{header_html_str}</thead><tbody>{''.join(row_html_list)}</tbody></table>"
        "<div class=\"footnote\">* Recovery p50 is conditional on recovery of the last occurrence of the deepest drawdown inside the stated horizon. "
        "Observed daily and calendar-month rows report return distributions only.</div>"
    )


def _verdict_panel_html(verdict_row_list: object) -> str:
    if not isinstance(verdict_row_list, list) or len(verdict_row_list) == 0:
        return ""
    status_class_dict = {
        VERDICT_STATUS_GREEN_STR: "v-green",
        VERDICT_STATUS_AMBER_STR: "v-amber",
        VERDICT_STATUS_RED_STR: "v-red",
        VERDICT_STATUS_NA_STR: "v-na",
    }
    row_html_list = []
    for row_obj in verdict_row_list:
        if not isinstance(row_obj, dict):
            continue
        status_str = str(row_obj.get("status_str", VERDICT_STATUS_NA_STR))
        dot_class_str = status_class_dict.get(status_str, "v-na")
        row_html_list.append(
            "<div class=\"verdict-row\">"
            f"<span class=\"verdict-dot {dot_class_str}\"></span>"
            f"<span class=\"verdict-label\">{html.escape(str(row_obj.get('label_str', '')))}</span>"
            f"<span class=\"verdict-value\">{html.escape(str(row_obj.get('value_str', '')))}</span>"
            f"<span class=\"verdict-text\">{html.escape(str(row_obj.get('conclusion_str', '')))}</span>"
            "</div>"
        )
    return (
        "<div class=\"verdict-panel\">"
        "<div class=\"verdict-title\">Read-first diagnostic bands</div>"
        + "".join(row_html_list)
        + "<div class=\"verdict-disclaimer\">Bands classify historically conditioned bootstrap diagnostics only. "
        "They do not validate edge, calibrate forward odds, or define account limits. See ASSUMPTIONS_AND_GAPS.</div>"
        "</div>"
    )


def _summary_tiles_html(summary_dict: dict[str, object]) -> str:
    observed_metric_dict = summary_dict.get("observed_metrics", {})
    if not isinstance(observed_metric_dict, dict):
        observed_metric_dict = {}
    tile_tuple = (
        ("CAGR", "cagr_float"),
        ("Sharpe", "sharpe_float"),
        ("Max DD", "max_drawdown_float"),
        ("Daily VaR 95", "var_95_daily_return_float"),
        ("Daily CVaR 95", "cvar_95_daily_return_float"),
    )
    tile_html_list = []
    for label_str, metric_name_str in tile_tuple:
        value_obj = observed_metric_dict.get(metric_name_str)
        subtitle_str = TILE_SUBTITLE_BY_METRIC_DICT.get(metric_name_str, "")
        tile_html_list.append(
            "<div class=\"tile\">"
            f"<div class=\"tile-label\">{html.escape(label_str)}</div>"
            f"<div class=\"tile-value\">{_format_metric_value(value_obj, metric_name_str)}</div>"
            f"<div class=\"tile-sub\">{html.escape(subtitle_str)}</div>"
            "</div>"
        )
    return "<div class=\"tile-grid\">" + "".join(tile_html_list) + "</div>"


def _return_histogram_svg(
    histogram_df: pd.DataFrame,
    mean_float: float | None = None,
    median_float: float | None = None,
) -> str:
    if histogram_df is None or len(histogram_df) == 0:
        return "<p>No return histogram data available.</p>"
    width_float = 960.0
    height_float = 330.0
    left_float = 70.0
    right_float = 20.0
    top_float = 18.0
    bottom_float = 56.0
    plot_width_float = width_float - left_float - right_float
    plot_height_float = height_float - top_float - bottom_float
    x_min_float = float(histogram_df["bin_left_float"].min())
    x_max_float = float(histogram_df["bin_right_float"].max())
    axis_y_float = top_float + plot_height_float

    # *** CRITICAL*** Bar height is the square root of the share of days, not
    # the share itself. On a linear count axis this strategy's peak bin holds
    # 343 days while thirty of eighty bins hold two or fewer, so the entire
    # left tail -- the only part a risk report exists to show -- draws under
    # one pixel. The square root is a deliberate distortion of area for
    # legibility, which is why the axis is labelled with its true shares and
    # the tick spacing is visibly uneven.
    total_count_float = float(max(1.0, float(histogram_df["count_int"].sum())))
    max_share_float = float(max(1e-9, float(histogram_df["count_int"].max()) / total_count_float))

    def share_height_float(share_float: float) -> float:
        return plot_height_float * float(np.sqrt(max(0.0, share_float) / max_share_float))

    gridline_count_int = 4
    gridline_html_list = []
    for tick_idx_int in range(gridline_count_int + 1):
        position_fraction_float = tick_idx_int / gridline_count_int
        tick_share_float = max_share_float * position_fraction_float**2
        tick_y_float = axis_y_float - plot_height_float * position_fraction_float
        if tick_idx_int > 0:
            gridline_html_list.append(
                f"<line x1=\"{left_float:.1f}\" y1=\"{tick_y_float:.1f}\" "
                f"x2=\"{left_float + plot_width_float:.1f}\" y2=\"{tick_y_float:.1f}\" "
                f"stroke=\"{SIGNATURE_PALETTE_DICT['grid']}\" stroke-width=\"1\" />"
            )
        gridline_html_list.append(
            f"<text x=\"{left_float - 6:.1f}\" y=\"{tick_y_float + 4:.1f}\" "
            f"fill=\"{SIGNATURE_PALETTE_DICT['muted']}\" font-size=\"11\" text-anchor=\"end\">{tick_share_float:.1%}</text>"
        )

    # Evenly spaced x-axis ticks across the full return range.
    x_tick_count_int = 6
    x_tick_html_list = []
    for tick_idx_int in range(x_tick_count_int + 1):
        tick_return_float = x_min_float + (x_max_float - x_min_float) * tick_idx_int / x_tick_count_int
        tick_x_float = _scale_float(
            tick_return_float, x_min_float, x_max_float, left_float, left_float + plot_width_float
        )
        anchor_str = "middle"
        if tick_idx_int == 0:
            anchor_str = "start"
        elif tick_idx_int == x_tick_count_int:
            anchor_str = "end"
        x_tick_html_list.append(
            f"<text x=\"{tick_x_float:.1f}\" y=\"{axis_y_float + 16:.1f}\" "
            f"fill=\"{SIGNATURE_PALETTE_DICT['muted']}\" font-size=\"11\" text-anchor=\"{anchor_str}\">{_format_percent(tick_return_float)}</text>"
        )

    bar_html_list = []
    for _, row_ser in histogram_df.iterrows():
        bin_left_float = float(row_ser["bin_left_float"])
        bin_right_float = float(row_ser["bin_right_float"])
        count_float = float(row_ser["count_int"])
        x_float = _scale_float(bin_left_float, x_min_float, x_max_float, left_float, left_float + plot_width_float)
        x2_float = _scale_float(bin_right_float, x_min_float, x_max_float, left_float, left_float + plot_width_float)
        bar_width_float = max(1.0, x2_float - x_float - 1.0)
        bar_height_float = share_height_float(count_float / total_count_float)
        y_float = axis_y_float - bar_height_float
        # Colouring by sign repeats what the position on the axis already says
        # and spends the strongest channel on nothing. One fill; the zero rule
        # carries the split.
        fill_str = str(SIGNATURE_PALETTE_DICT['muted'])
        bar_html_list.append(
            f"<rect x=\"{x_float:.2f}\" y=\"{y_float:.2f}\" width=\"{bar_width_float:.2f}\" height=\"{bar_height_float:.2f}\" fill=\"{fill_str}\" opacity=\"0.55\" />"
        )
    zero_x_float = _scale_float(0.0, x_min_float, x_max_float, left_float, left_float + plot_width_float)

    # Mean (solid blue) and median (dashed purple) reference lines.
    reference_line_html_list = []
    mean_value_float = _json_float(mean_float)
    if mean_value_float is not None:
        mean_x_float = _scale_float(
            mean_value_float, x_min_float, x_max_float, left_float, left_float + plot_width_float
        )
        reference_line_html_list.append(
            f"<line x1=\"{mean_x_float:.1f}\" y1=\"{top_float:.1f}\" x2=\"{mean_x_float:.1f}\" "
            f"y2=\"{axis_y_float:.1f}\" stroke=\"{SIGNATURE_PALETTE_DICT['overlay_cycle'][0]}\" stroke-width=\"2\" />"
        )
    median_value_float = _json_float(median_float)
    if median_value_float is not None:
        median_x_float = _scale_float(
            median_value_float, x_min_float, x_max_float, left_float, left_float + plot_width_float
        )
        reference_line_html_list.append(
            f"<line x1=\"{median_x_float:.1f}\" y1=\"{top_float:.1f}\" x2=\"{median_x_float:.1f}\" "
            f"y2=\"{axis_y_float:.1f}\" stroke=\"{SIGNATURE_PALETTE_DICT['overlay_cycle'][5]}\" stroke-width=\"2\" stroke-dasharray=\"5 3\" />"
        )

    legend_y_float = top_float + 6.0
    legend_x_float = left_float + 10.0
    legend_html_list = [
        f"<line x1=\"{legend_x_float:.1f}\" y1=\"{legend_y_float:.1f}\" x2=\"{legend_x_float + 18:.1f}\" y2=\"{legend_y_float:.1f}\" stroke=\"{SIGNATURE_PALETTE_DICT['overlay_cycle'][0]}\" stroke-width=\"2\" />",
        f"<text x=\"{legend_x_float + 24:.1f}\" y=\"{legend_y_float + 4:.1f}\" fill=\"{SIGNATURE_PALETTE_DICT['muted']}\" font-size=\"11\">mean</text>",
        f"<line x1=\"{legend_x_float + 70:.1f}\" y1=\"{legend_y_float:.1f}\" x2=\"{legend_x_float + 88:.1f}\" y2=\"{legend_y_float:.1f}\" stroke=\"{SIGNATURE_PALETTE_DICT['overlay_cycle'][5]}\" stroke-width=\"2\" stroke-dasharray=\"5 3\" />",
        f"<text x=\"{legend_x_float + 94:.1f}\" y=\"{legend_y_float + 4:.1f}\" fill=\"{SIGNATURE_PALETTE_DICT['muted']}\" font-size=\"11\">median</text>",
        f"<line x1=\"{legend_x_float + 150:.1f}\" y1=\"{legend_y_float:.1f}\" x2=\"{legend_x_float + 168:.1f}\" y2=\"{legend_y_float:.1f}\" stroke=\"{SIGNATURE_PALETTE_DICT['ink']}\" stroke-width=\"1\" stroke-dasharray=\"4 4\" />",
        f"<text x=\"{legend_x_float + 174:.1f}\" y=\"{legend_y_float + 4:.1f}\" fill=\"{SIGNATURE_PALETTE_DICT['muted']}\" font-size=\"11\">zero</text>",
    ]

    y_axis_title_x_float = 18.0
    y_axis_title_y_float = top_float + plot_height_float / 2.0
    return (
        f"<svg class=\"chart\" viewBox=\"0 0 {width_float:.0f} {height_float:.0f}\" role=\"img\" aria-label=\"Returns histogram\">"
        f"<rect x=\"0\" y=\"0\" width=\"{width_float:.0f}\" height=\"{height_float:.0f}\" fill=\"{SIGNATURE_PALETTE_DICT['page']}\" />"
        + "".join(gridline_html_list)
        + f"<line x1=\"{left_float:.1f}\" y1=\"{axis_y_float:.1f}\" x2=\"{left_float + plot_width_float:.1f}\" y2=\"{axis_y_float:.1f}\" stroke=\"{SIGNATURE_PALETTE_DICT['muted']}\" stroke-width=\"1\" />"
        + f"<line x1=\"{left_float:.1f}\" y1=\"{top_float:.1f}\" x2=\"{left_float:.1f}\" y2=\"{axis_y_float:.1f}\" stroke=\"{SIGNATURE_PALETTE_DICT['muted']}\" stroke-width=\"1\" />"
        + "".join(bar_html_list)
        + f"<line x1=\"{zero_x_float:.1f}\" y1=\"{top_float:.1f}\" x2=\"{zero_x_float:.1f}\" y2=\"{axis_y_float:.1f}\" stroke=\"{SIGNATURE_PALETTE_DICT['ink']}\" stroke-width=\"1\" stroke-dasharray=\"4 4\" />"
        + "".join(reference_line_html_list)
        + "".join(x_tick_html_list)
        + "".join(legend_html_list)
        + f"<text x=\"{left_float + plot_width_float / 2.0:.1f}\" y=\"{height_float - 8:.1f}\" fill=\"{SIGNATURE_PALETTE_DICT['ink']}\" font-size=\"12\" text-anchor=\"middle\">Daily return</text>"
        + f"<text x=\"{y_axis_title_x_float:.1f}\" y=\"{y_axis_title_y_float:.1f}\" fill=\"{SIGNATURE_PALETTE_DICT['ink']}\" font-size=\"12\" text-anchor=\"middle\" transform=\"rotate(-90 {y_axis_title_x_float:.1f} {y_axis_title_y_float:.1f})\">Share of days (sqrt scale)</text>"
        + "</svg>"
    )


SAMPLE_DISTRIBUTION_METRIC_TUPLE = (
    "cagr_float",
    "max_drawdown_float",
    "sharpe_float",
    "worst_21d_return_float",
)


def _sample_distribution_panel_html_str(
    path_value_vec: np.ndarray,
    observed_value_float: float | None,
    observed_percentile_float: float | None,
    p05_float: float | None,
    p95_float: float | None,
    title_str: str,
    metric_name_str: str,
    origin_x_float: float,
    origin_y_float: float,
    panel_width_float: float,
    panel_height_float: float,
) -> str:
    """Draw one metric's bootstrap distribution with the observed value on it.

    This is the sampling distribution the report is built on: one draw per
    simulated path, so it answers "was the backtest lucky" directly rather than
    through a percentile in a table. The observed value is the whole point of
    the panel, so it is the only mark drawn in full-strength ink.
    """
    left_float = origin_x_float + 8.0
    plot_width_float = panel_width_float - 24.0
    # The panel title and the observed-value label are both anchored near the
    # top edge, and the label follows the observed value horizontally. Leave a
    # full line between them so they cannot collide when a metric's observed
    # value happens to sit at the left of its distribution.
    top_float = origin_y_float + 36.0
    plot_height_float = panel_height_float - 68.0
    axis_y_float = top_float + plot_height_float

    finite_value_vec = path_value_vec[np.isfinite(path_value_vec)]
    if len(finite_value_vec) == 0:
        return ""

    # *** CRITICAL*** The observed value must be inside the drawn range or the
    # panel would show a distribution and silently clip the one mark that gives
    # it meaning.
    low_float = float(finite_value_vec.min())
    high_float = float(finite_value_vec.max())
    if observed_value_float is not None and np.isfinite(observed_value_float):
        low_float = min(low_float, float(observed_value_float))
        high_float = max(high_float, float(observed_value_float))
    if high_float <= low_float:
        high_float = low_float + 1e-9

    bin_count_int = 44
    count_vec, edge_vec = np.histogram(
        finite_value_vec, bins=bin_count_int, range=(low_float, high_float)
    )
    peak_count_float = float(max(1, int(count_vec.max())))

    def x_of(value_float: float) -> float:
        return _scale_float(
            float(value_float), low_float, high_float, left_float, left_float + plot_width_float
        )

    # The 5th-95th band behind the bars carries the interval the table reports,
    # so the two sections cannot disagree about what "likely range" means.
    band_html_str = ""
    if p05_float is not None and p95_float is not None:
        band_x_float = x_of(p05_float)
        band_x2_float = x_of(p95_float)
        band_html_str = (
            f'<rect x="{band_x_float:.1f}" y="{top_float:.1f}" '
            f'width="{max(0.0, band_x2_float - band_x_float):.1f}" height="{plot_height_float:.1f}" '
            f'fill="{SIGNATURE_PALETTE_DICT["neutral"]}" opacity="0.45" />'
        )

    bar_html_list = []
    for bin_idx_int in range(len(count_vec)):
        if count_vec[bin_idx_int] == 0:
            continue
        bar_x_float = x_of(edge_vec[bin_idx_int])
        bar_x2_float = x_of(edge_vec[bin_idx_int + 1])
        bar_height_float = plot_height_float * float(count_vec[bin_idx_int]) / peak_count_float
        bar_html_list.append(
            f'<rect x="{bar_x_float:.2f}" y="{axis_y_float - bar_height_float:.2f}" '
            f'width="{max(0.6, bar_x2_float - bar_x_float - 0.6):.2f}" height="{bar_height_float:.2f}" '
            f'fill="{SIGNATURE_PALETTE_DICT["muted"]}" opacity="0.5" />'
        )

    observed_html_str = ""
    if observed_value_float is not None and np.isfinite(observed_value_float):
        observed_x_float = x_of(observed_value_float)
        percentile_str = ""
        if observed_percentile_float is not None and np.isfinite(observed_percentile_float):
            percentile_str = f" (p{float(observed_percentile_float) * 100.0:.0f})"
        label_anchor_str = "start" if observed_x_float < left_float + plot_width_float * 0.6 else "end"
        label_offset_float = 5.0 if label_anchor_str == "start" else -5.0
        observed_html_str = (
            f'<line x1="{observed_x_float:.1f}" y1="{top_float - 6:.1f}" '
            f'x2="{observed_x_float:.1f}" y2="{axis_y_float:.1f}" '
            f'stroke="{SIGNATURE_PALETTE_DICT["ink"]}" stroke-width="1.5" />'
            f'<text x="{observed_x_float + label_offset_float:.1f}" y="{top_float - 10:.1f}" '
            f'fill="{SIGNATURE_PALETTE_DICT["ink"]}" font-size="10.5" text-anchor="{label_anchor_str}">'
            f"observed {html.escape(_format_metric_value(observed_value_float, metric_name_str))}"
            f"{html.escape(percentile_str)}</text>"
        )

    axis_label_html_str = "".join(
        f'<text x="{x_of(edge_value_float):.1f}" y="{axis_y_float + 14:.1f}" '
        f'fill="{SIGNATURE_PALETTE_DICT["muted"]}" font-size="10" '
        f'text-anchor="{anchor_str}">'
        f"{html.escape(_format_metric_value(edge_value_float, metric_name_str))}</text>"
        for edge_value_float, anchor_str in (
            (low_float, "start"),
            (high_float, "end"),
        )
    )

    return (
        band_html_str
        + "".join(bar_html_list)
        + f'<line x1="{left_float:.1f}" y1="{axis_y_float:.1f}" '
        f'x2="{left_float + plot_width_float:.1f}" y2="{axis_y_float:.1f}" '
        f'stroke="{SIGNATURE_PALETTE_DICT["muted"]}" stroke-width="1" />'
        + observed_html_str
        + axis_label_html_str
        + f'<text x="{left_float:.1f}" y="{origin_y_float + 12:.1f}" '
        f'fill="{SIGNATURE_PALETTE_DICT["ink"]}" font-size="12" font-weight="600">'
        f"{html.escape(title_str)}</text>"
    )


def _sample_distribution_svg(
    path_metric_df: pd.DataFrame,
    interval_df: pd.DataFrame,
    primary_mean_block_length_int: int,
) -> str:
    """Small multiples of the bootstrap distribution for the headline metrics.

    Every path in the bootstrap produces its own CAGR, drawdown and Sharpe, so
    these are ten thousand draws of "how this strategy could have gone". The
    interval table reduces each one to three percentiles; the shape is what
    says whether the observed result sits on a plateau or on a cliff edge.
    """
    if path_metric_df is None or len(path_metric_df) == 0:
        return "<p>No bootstrap path metric data available.</p>"

    primary_path_df = path_metric_df[
        path_metric_df["mean_block_length_int"] == primary_mean_block_length_int
    ]
    if len(primary_path_df) == 0:
        return "<p>No bootstrap path metric data available.</p>"

    interval_lookup_df = None
    if interval_df is not None and len(interval_df) > 0:
        interval_lookup_df = interval_df[
            interval_df["mean_block_length_int"] == primary_mean_block_length_int
        ]

    width_float = 960.0
    panel_width_float = width_float / 2.0
    panel_height_float = 150.0
    height_float = panel_height_float * 2.0 + 10.0

    panel_html_list = []
    for panel_idx_int, metric_name_str in enumerate(SAMPLE_DISTRIBUTION_METRIC_TUPLE):
        if metric_name_str not in primary_path_df.columns:
            continue
        observed_value_float = None
        observed_percentile_float = None
        p05_float = None
        p95_float = None
        if interval_lookup_df is not None:
            metric_row_df = interval_lookup_df[
                interval_lookup_df["metric_name_str"] == metric_name_str
            ]
            if len(metric_row_df) > 0:
                metric_row_ser = metric_row_df.iloc[0]
                observed_value_float = _json_float(metric_row_ser.get("observed_value_float"))
                observed_percentile_float = _json_float(
                    metric_row_ser.get("observed_percentile_float")
                )
                p05_float = _json_float(metric_row_ser.get("p05_float"))
                p95_float = _json_float(metric_row_ser.get("p95_float"))

        panel_html_list.append(
            _sample_distribution_panel_html_str(
                path_value_vec=primary_path_df[metric_name_str].to_numpy(dtype=float),
                observed_value_float=observed_value_float,
                observed_percentile_float=observed_percentile_float,
                p05_float=p05_float,
                p95_float=p95_float,
                title_str=_metric_label_str(metric_name_str),
                metric_name_str=metric_name_str,
                origin_x_float=(panel_idx_int % 2) * panel_width_float,
                origin_y_float=(panel_idx_int // 2) * panel_height_float + 8.0,
                panel_width_float=panel_width_float,
                panel_height_float=panel_height_float,
            )
        )

    if not any(panel_html_list):
        return "<p>No bootstrap path metric data available.</p>"

    return (
        f'<svg class="chart" viewBox="0 0 {width_float:.0f} {height_float:.0f}" role="img" '
        'aria-label="Bootstrap sample distributions for the headline metrics">'
        f'<rect x="0" y="0" width="{width_float:.0f}" height="{height_float:.0f}" '
        f'fill="{SIGNATURE_PALETTE_DICT["page"]}" />'
        + "".join(panel_html_list)
        + "</svg>"
    )


def _bootstrap_equity_svg(equity_path_df: pd.DataFrame) -> str:
    if equity_path_df is None or len(equity_path_df) == 0:
        return "<p>No Monte Carlo equity path data available.</p>"
    width_float = 960.0
    height_float = 360.0
    left_float = 70.0
    right_float = 20.0
    top_float = 18.0
    bottom_float = 56.0
    plot_width_float = width_float - left_float - right_float
    plot_height_float = height_float - top_float - bottom_float
    equity_float_ser = equity_path_df["equity_float"].astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    if len(equity_float_ser) == 0:
        return "<p>No finite Monte Carlo equity path data available.</p>"
    x_min_float = float(equity_path_df["step_int"].min())
    x_max_float = float(equity_path_df["step_int"].max())

    # *** CRITICAL*** equity is plotted on a log y-axis (the repo convention for
    # compounding curves): equal vertical distance = equal percent change. This
    # also avoids the earlier clamp-flattening of high paths against a linear
    # ceiling. Bounds use 0.5/99.5 percentile in log space so one runaway path
    # does not squash the bulk.
    log_equity_float_ser = np.log(equity_float_ser.clip(lower=1e-6))
    y_min_float = float(log_equity_float_ser.quantile(0.005))
    y_max_float = float(log_equity_float_ser.quantile(0.995))
    if np.isclose(y_min_float, y_max_float):
        y_min_float -= 0.01
        y_max_float += 0.01
    axis_y_float = top_float + plot_height_float

    equity_path_df = equity_path_df.copy()
    equity_path_df["equity_float"] = np.log(
        equity_path_df["equity_float"].astype(float).clip(lower=1e-6)
    )

    gridline_count_int = 4
    gridline_html_list = []
    for tick_idx_int in range(gridline_count_int + 1):
        tick_log_float = y_min_float + (y_max_float - y_min_float) * tick_idx_int / gridline_count_int
        tick_y_float = _scale_float(
            tick_log_float, y_min_float, y_max_float, axis_y_float, top_float
        )
        if 0 < tick_idx_int < gridline_count_int:
            gridline_html_list.append(
                f"<line x1=\"{left_float:.1f}\" y1=\"{tick_y_float:.1f}\" "
                f"x2=\"{left_float + plot_width_float:.1f}\" y2=\"{tick_y_float:.1f}\" "
                f"stroke=\"{SIGNATURE_PALETTE_DICT['grid']}\" stroke-width=\"1\" />"
            )
        gridline_html_list.append(
            f"<text x=\"{left_float - 6:.1f}\" y=\"{tick_y_float + 4:.1f}\" "
            f"fill=\"{SIGNATURE_PALETTE_DICT['muted']}\" font-size=\"11\" text-anchor=\"end\">{np.exp(tick_log_float):.2f}x</text>"
        )

    path_html_list = []
    bootstrap_df = equity_path_df[equity_path_df["path_kind_str"] == "bootstrap"]
    for path_id_int in list(dict.fromkeys(bootstrap_df["path_id_int"].astype(int).tolist()))[:60]:
        path_df = bootstrap_df[bootstrap_df["path_id_int"] == path_id_int]
        path_html_list.append(
            _polyline_svg(
                path_df,
                x_min_float,
                x_max_float,
                y_min_float,
                y_max_float,
                left_float,
                top_float,
                plot_width_float,
                plot_height_float,
                str(SIGNATURE_PALETTE_DICT['muted']),
                0.18,
                1.0,
            )
        )
    overlay_tuple = (
        ("p05", str(SIGNATURE_PALETTE_DICT['loss_dark']), 0.95, 2.2, "p05"),
        ("p50", str(SIGNATURE_PALETTE_DICT['ink']), 0.95, 2.2, "p50"),
        ("p95", str(SIGNATURE_PALETTE_DICT['profit_dark']), 0.95, 2.2, "p95"),
        ("observed", str(SIGNATURE_PALETTE_DICT['overlay_cycle'][0]), 1.0, 2.4, "observed"),
    )
    for path_kind_str, stroke_str, opacity_float, stroke_width_float, _label_str in overlay_tuple:
        path_df = equity_path_df[equity_path_df["path_kind_str"] == path_kind_str]
        if len(path_df) == 0:
            continue
        path_html_list.append(
            _polyline_svg(
                path_df,
                x_min_float,
                x_max_float,
                y_min_float,
                y_max_float,
                left_float,
                top_float,
                plot_width_float,
                plot_height_float,
                stroke_str,
                opacity_float,
                stroke_width_float,
            )
        )
    legend_box_width_float = 110.0
    legend_box_height_float = 16.0 * len(overlay_tuple) + 12.0
    legend_x_float = left_float + plot_width_float - legend_box_width_float - 8.0
    legend_y_float = top_float + 8.0
    legend_html_list = [
        f"<rect x=\"{legend_x_float:.1f}\" y=\"{legend_y_float:.1f}\" "
        f"width=\"{legend_box_width_float:.1f}\" height=\"{legend_box_height_float:.1f}\" "
        f"fill=\"{SIGNATURE_PALETTE_DICT['page']}\" fill-opacity=\"0.92\" stroke=\"{SIGNATURE_PALETTE_DICT['border']}\" stroke-width=\"1\" rx=\"4\" />"
    ]
    for legend_idx_int, (_path_kind_str, stroke_str, _opacity_float, _stroke_width_float, label_str) in enumerate(overlay_tuple):
        row_y_float = legend_y_float + 14.0 + legend_idx_int * 16.0
        legend_html_list.append(
            f"<line x1=\"{legend_x_float + 10.0:.1f}\" y1=\"{row_y_float - 4.0:.1f}\" "
            f"x2=\"{legend_x_float + 32.0:.1f}\" y2=\"{row_y_float - 4.0:.1f}\" "
            f"stroke=\"{stroke_str}\" stroke-width=\"2.4\" />"
        )
        legend_html_list.append(
            f"<text x=\"{legend_x_float + 40.0:.1f}\" y=\"{row_y_float:.1f}\" "
            f"fill=\"{SIGNATURE_PALETTE_DICT['ink']}\" font-size=\"11\">{html.escape(label_str)}</text>"
        )

    y_axis_title_x_float = 18.0
    y_axis_title_y_float = top_float + plot_height_float / 2.0
    return (
        f"<svg class=\"chart\" viewBox=\"0 0 {width_float:.0f} {height_float:.0f}\" role=\"img\" aria-label=\"Monte Carlo equity paths\">"
        f"<rect x=\"0\" y=\"0\" width=\"{width_float:.0f}\" height=\"{height_float:.0f}\" fill=\"{SIGNATURE_PALETTE_DICT['page']}\" />"
        + "".join(gridline_html_list)
        + f"<line x1=\"{left_float:.1f}\" y1=\"{axis_y_float:.1f}\" x2=\"{left_float + plot_width_float:.1f}\" y2=\"{axis_y_float:.1f}\" stroke=\"{SIGNATURE_PALETTE_DICT['muted']}\" stroke-width=\"1\" />"
        + f"<line x1=\"{left_float:.1f}\" y1=\"{top_float:.1f}\" x2=\"{left_float:.1f}\" y2=\"{axis_y_float:.1f}\" stroke=\"{SIGNATURE_PALETTE_DICT['muted']}\" stroke-width=\"1\" />"
        + "".join(path_html_list)
        + f"<text x=\"{left_float:.1f}\" y=\"{axis_y_float + 16:.1f}\" fill=\"{SIGNATURE_PALETTE_DICT['muted']}\" font-size=\"11\">0</text>"
        + f"<text x=\"{left_float + plot_width_float:.1f}\" y=\"{axis_y_float + 16:.1f}\" fill=\"{SIGNATURE_PALETTE_DICT['muted']}\" font-size=\"11\" text-anchor=\"end\">{int(x_max_float)} days</text>"
        + f"<text x=\"{left_float + plot_width_float / 2.0:.1f}\" y=\"{height_float - 8:.1f}\" fill=\"{SIGNATURE_PALETTE_DICT['ink']}\" font-size=\"12\" text-anchor=\"middle\">Days from start</text>"
        + f"<text x=\"{y_axis_title_x_float:.1f}\" y=\"{y_axis_title_y_float:.1f}\" fill=\"{SIGNATURE_PALETTE_DICT['ink']}\" font-size=\"12\" text-anchor=\"middle\" transform=\"rotate(-90 {y_axis_title_x_float:.1f} {y_axis_title_y_float:.1f})\">Equity multiple (log)</text>"
        + "".join(legend_html_list)
        + "</svg>"
    )


DAILY_METRIC_ORDER_LIST = [
    "expected_daily_return_float",
    "annualized_ev_float",
    "terminal_return_float",
    "cagr_float",
    "annual_volatility_float",
    "sharpe_float",
    "max_drawdown_float",
    "mar_float",
    "var_95_daily_return_float",
    "cvar_95_daily_return_float",
    "var_99_daily_return_float",
    "cvar_99_daily_return_float",
    "worst_1d_return_float",
    "worst_5d_return_float",
    "worst_21d_return_float",
    "worst_63d_return_float",
]

MONTHLY_METRIC_ORDER_LIST = [
    "monthly_expected_return_float",
    "monthly_volatility_float",
    "monthly_sharpe_float",
    "monthly_var_95_return_float",
    "monthly_cvar_95_return_float",
    "monthly_var_99_return_float",
    "monthly_cvar_99_return_float",
    "worst_21d_return_float",
    "worst_63d_return_float",
    "worst_126d_return_float",
    "worst_252d_return_float",
]


def _horizon_probability_tables_html(
    horizon_probability_df: pd.DataFrame,
    drawdown_threshold_tuple: Sequence[float],
    upside_threshold_tuple: Sequence[float],
    full_sample_drawdown_row_dict: dict[str, object] | None = None,
) -> str:
    """Lay out downside and upside horizon probabilities.

    ``full_sample_drawdown_row_dict`` appends the whole-sample-length bootstrap
    as one more horizon row. Those probabilities used to live in a separate
    breach table over the *same* five thresholds, which made the reader compare
    a row of columns against a column of rows to answer one question.
    """
    if horizon_probability_df is None or len(horizon_probability_df) == 0:
        return "<p>No horizon probability data available.</p>"

    normalized_drawdown_threshold_tuple = tuple(float(value_float) for value_float in drawdown_threshold_tuple)
    normalized_upside_threshold_tuple = tuple(float(value_float) for value_float in upside_threshold_tuple)

    downside_header_html_list = [
        "<th>Horizon</th>",
        "<th>Paths</th>",
    ]
    for threshold_float in normalized_drawdown_threshold_tuple:
        downside_header_html_list.append(
            f"<th>{html.escape(f'DD <= {_signed_percent_str(-abs(float(threshold_float)))}')}</th>"
        )
    downside_header_html_list.extend(
        [
            "<th>Median max DD</th>",
            "<th>Bootstrap p05 max DD</th>",
        ]
    )

    downside_row_html_list = []
    upside_row_html_list = []
    for _, row_ser in horizon_probability_df.iterrows():
        horizon_label_str = f"{int(row_ser['horizon_year_int'])}y"
        path_count_int = int(row_ser.get("simulation_path_count_int", 0))
        downside_cell_html_list = [
            f"<td>{html.escape(horizon_label_str)}</td>",
            f"<td>{path_count_int}</td>",
        ]
        for threshold_float in normalized_drawdown_threshold_tuple:
            column_name_str = _threshold_column_name_str("drawdown_lte", abs(float(threshold_float)))
            downside_cell_html_list.append(
                f"<td>{_format_percent(row_ser.get(column_name_str))}</td>"
            )
        downside_cell_html_list.append(
            _metric_cell_html(row_ser.get("max_drawdown_p50_float"), "max_drawdown_float")
        )
        downside_cell_html_list.append(
            _metric_cell_html(row_ser.get("max_drawdown_p05_float"), "max_drawdown_float")
        )
        downside_row_html_list.append("<tr>" + "".join(downside_cell_html_list) + "</tr>")

        upside_cell_html_list = [
            f"<td>{html.escape(horizon_label_str)}</td>",
            f"<td>{path_count_int}</td>",
        ]
        for threshold_float in normalized_upside_threshold_tuple:
            column_name_str = _threshold_column_name_str("gain_gte", abs(float(threshold_float)))
            upside_cell_html_list.append(
                f"<td>{_format_percent(row_ser.get(column_name_str))}</td>"
            )
        upside_cell_html_list.append(
            _metric_cell_html(row_ser.get("max_gain_p50_float"), "max_gain_float")
        )
        upside_cell_html_list.append(
            _metric_cell_html(row_ser.get("max_gain_p95_float"), "max_gain_float")
        )
        upside_cell_html_list.append(
            _metric_cell_html(row_ser.get("terminal_return_p50_float"), "terminal_return_float")
        )
        upside_row_html_list.append("<tr>" + "".join(upside_cell_html_list) + "</tr>")

    if full_sample_drawdown_row_dict:
        # *** CRITICAL*** This row is the full realized sample length, not a
        # fixed horizon, so it is not comparable to the 1y-5y rows as a rate.
        # It is the same bootstrap at its native length; the label must say so.
        full_sample_cell_html_list = [
            '<td class="metric">Full sample</td>',
            f"<td>{int(full_sample_drawdown_row_dict.get('simulation_path_count_int', 0))}</td>",
        ]
        for threshold_float in normalized_drawdown_threshold_tuple:
            breach_key_str = f"max_drawdown_lte_{abs(float(threshold_float)):.0%}"
            full_sample_cell_html_list.append(
                f"<td>{_format_percent(full_sample_drawdown_row_dict.get(breach_key_str))}</td>"
            )
        full_sample_cell_html_list.append(
            _metric_cell_html(
                full_sample_drawdown_row_dict.get("max_drawdown_p50_float"), "max_drawdown_float"
            )
        )
        full_sample_cell_html_list.append(
            _metric_cell_html(
                full_sample_drawdown_row_dict.get("max_drawdown_p05_float"), "max_drawdown_float"
            )
        )
        downside_row_html_list.append(
            '<tr class="full-sample-row">' + "".join(full_sample_cell_html_list) + "</tr>"
        )

    upside_header_html_list = [
        "<th>Horizon</th>",
        "<th>Paths</th>",
    ]
    for threshold_float in normalized_upside_threshold_tuple:
        upside_header_html_list.append(
            f"<th>{html.escape(f'Gain >= {_signed_percent_str(abs(float(threshold_float)))}')}</th>"
        )
    upside_header_html_list.extend(
        [
            "<th>Median max gain</th>",
            "<th>Bootstrap p95 max gain</th>",
            "<th>Median terminal</th>",
        ]
    )

    downside_table_html = (
        "<h3>Downside drawdown path shares</h3>"
        "<div class=\"scroll\"><table><thead><tr>"
        + "".join(downside_header_html_list)
        + "</tr></thead><tbody>"
        + "".join(downside_row_html_list)
        + "</tbody></table></div>"
    )
    upside_table_html = (
        "<h3>Upside reach path shares</h3>"
        "<div class=\"scroll\"><table><thead><tr>"
        + "".join(upside_header_html_list)
        + "</tr></thead><tbody>"
        + "".join(upside_row_html_list)
        + "</tbody></table></div>"
    )
    return downside_table_html + upside_table_html


def _interval_table_html(
    interval_df: pd.DataFrame,
    primary_mean_block_length_int: int,
    confidence_level_float: float,
    simulation_count_int: int,
    metric_order_list: list[str] | None = None,
    label_dict: dict[str, str] | None = None,
) -> str:
    if interval_df is None or len(interval_df) == 0:
        return "<p>No bootstrap interval data available.</p>"
    display_df = interval_df[interval_df["mean_block_length_int"] == primary_mean_block_length_int]
    if metric_order_list is None:
        metric_order_list = DAILY_METRIC_ORDER_LIST
    tail_sample_count_int = int(simulation_count_int * CVAR_99_TAIL_FRACTION_FLOAT)
    show_small_sample_footnote_bool = tail_sample_count_int < CVAR_99_MIN_TAIL_SAMPLE_INT
    row_html_list = []
    for metric_name_str in metric_order_list:
        row_df = display_df[display_df["metric_name_str"] == metric_name_str]
        if len(row_df) == 0:
            continue
        row_ser = row_df.iloc[0]
        ci_str = (
            "["
            + _format_metric_value(row_ser.get("ci_lower_float"), metric_name_str)
            + ", "
            + _format_metric_value(row_ser.get("ci_upper_float"), metric_name_str)
            + "]"
        )
        label_html_str = html.escape(_metric_label_str(metric_name_str, label_dict))
        if (
            show_small_sample_footnote_bool
            and metric_name_str in TAIL_SENSITIVE_METRIC_TUPLE
        ):
            label_html_str += "<sup>1</sup>"
        row_html_list.append(
            "<tr>"
            f"<td>{label_html_str}</td>"
            f"{_metric_cell_html(row_ser.get('observed_value_float'), metric_name_str)}"
            f"<td>{_format_observed_percentile_str(row_ser.get('observed_percentile_float'))}</td>"
            f"<td>{html.escape(ci_str)}</td>"
            f"{_metric_cell_html(row_ser.get('p05_float'), metric_name_str)}"
            f"{_metric_cell_html(row_ser.get('p50_float'), metric_name_str)}"
            f"{_metric_cell_html(row_ser.get('p95_float'), metric_name_str)}"
            "</tr>"
        )
    confidence_header_str = html.escape(f"Confidence interval ({confidence_level_float:.0%})")
    table_html_str = (
        "<table><thead><tr>"
        f"<th>Metric</th><th>Observed</th><th>Observed percentile</th><th>{confidence_header_str}</th>"
        "<th>P5</th><th>P50</th><th>P95</th>"
        "</tr></thead><tbody>"
        + "".join(row_html_list)
        + "</tbody></table>"
    )
    if show_small_sample_footnote_bool:
        relevant_tail_metric_in_table_bool = any(
            metric_name_str in TAIL_SENSITIVE_METRIC_TUPLE
            for metric_name_str in metric_order_list
        )
        if relevant_tail_metric_in_table_bool:
            table_html_str += (
                "<div class=\"footnote\"><sup>1</sup> "
                + html.escape(CVAR_99_SMALL_SAMPLE_FOOTNOTE_STR)
                + f" (effective tail sample = {tail_sample_count_int} per simulation; "
                f"raise --simulation-count above {int(CVAR_99_MIN_TAIL_SAMPLE_INT / CVAR_99_TAIL_FRACTION_FLOAT)} to suppress this notice.)"
                + "</div>"
            )
    return table_html_str


def _format_observed_percentile_str(value_obj) -> str:
    value_float = _json_float(value_obj)
    if value_float is None:
        return "N/A"
    return f"p{value_float * 100.0:.0f}"


_UNDERWATER_BREACH_KEY_PATTERN = re.compile(r"^underwater_ge_(\d+)m$")


def _breach_event_label_str(key_str: str) -> str:
    """Turn a breach dictionary key into something a reader can read.

    These keys are storage identifiers. Printing them raw put
    ``underwater_ge_3m`` in the middle of a report whose every other label is
    prose, so the reader had to decode the schema to read the number.
    """
    underwater_match = _UNDERWATER_BREACH_KEY_PATTERN.match(key_str)
    if underwater_match is not None:
        month_int = int(underwater_match.group(1))
        return f"Underwater for {month_int} months or more"
    return key_str.replace("_", " ")


def _breach_event_sort_key_tuple(key_str: str) -> tuple[int, int, str]:
    """Sort underwater keys by their month, keeping unknown keys last and stable."""
    underwater_match = _UNDERWATER_BREACH_KEY_PATTERN.match(key_str)
    if underwater_match is not None:
        return (0, int(underwater_match.group(1)), key_str)
    return (1, 0, key_str)


def _breach_table_html(
    terminal_loss_probability_float: object,
    time_underwater_breach_probability_dict: dict[str, object] | None = None,
) -> str:
    """Report only what the horizon table cannot say.

    The drawdown-threshold rows moved into the downside horizon table as its
    full-sample row; repeating them here asked the same question twice in two
    different shapes.
    """
    row_html_list = [
        '<tr><td class="metric">Ends below its starting value</td>'
        f"<td>{_format_percent(terminal_loss_probability_float)}</td></tr>"
    ]
    if time_underwater_breach_probability_dict:
        # *** CRITICAL*** Order by the month the key encodes, not by the key.
        # The summary is persisted with sorted keys, so relying on dict order
        # printed 12m, 24m, 3m, 6m whenever a saved result was re-rendered --
        # a monotonic series shown out of order reads as a broken model.
        ordered_key_list = sorted(
            time_underwater_breach_probability_dict,
            key=lambda key_str: _breach_event_sort_key_tuple(str(key_str)),
        )
        for key_str in ordered_key_list:
            row_html_list.append(
                f'<tr><td class="metric">{html.escape(_breach_event_label_str(str(key_str)))}</td>'
                f"<td>{_format_percent(time_underwater_breach_probability_dict[key_str])}</td></tr>"
            )
    return "<table><thead><tr><th>Event</th><th>Probability</th></tr></thead><tbody>" + "".join(row_html_list) + "</tbody></table>"


def _polyline_svg(
    path_df: pd.DataFrame,
    x_min_float: float,
    x_max_float: float,
    y_min_float: float,
    y_max_float: float,
    left_float: float,
    top_float: float,
    plot_width_float: float,
    plot_height_float: float,
    stroke_str: str,
    opacity_float: float,
    stroke_width_float: float,
) -> str:
    if len(path_df) == 0:
        return ""
    sorted_path_df = path_df.sort_values("step_int", kind="mergesort")
    if len(sorted_path_df) > 180:
        take_idx = np.linspace(0, len(sorted_path_df) - 1, 180).astype(int)
        sorted_path_df = sorted_path_df.iloc[take_idx]
    point_list = []
    for _, row_ser in sorted_path_df.iterrows():
        x_float = _scale_float(
            float(row_ser["step_int"]),
            x_min_float,
            x_max_float,
            left_float,
            left_float + plot_width_float,
        )
        y_float = _scale_float(
            float(row_ser["equity_float"]),
            y_min_float,
            y_max_float,
            top_float + plot_height_float,
            top_float,
        )
        point_list.append(f"{x_float:.2f},{y_float:.2f}")
    return (
        f"<polyline points=\"{' '.join(point_list)}\" fill=\"none\" "
        f"stroke=\"{html.escape(stroke_str)}\" stroke-width=\"{stroke_width_float:.2f}\" "
        f"opacity=\"{opacity_float:.2f}\" />"
    )


def _scale_float(
    value_float: float,
    source_min_float: float,
    source_max_float: float,
    target_min_float: float,
    target_max_float: float,
) -> float:
    if np.isclose(source_min_float, source_max_float):
        return (target_min_float + target_max_float) / 2.0
    clipped_value_float = min(max(value_float, source_min_float), source_max_float)
    ratio_float = (clipped_value_float - source_min_float) / (source_max_float - source_min_float)
    return target_min_float + ratio_float * (target_max_float - target_min_float)


def _metric_cell_html(value_obj, metric_name_str: str) -> str:
    class_str = ""
    value_float = _json_float(value_obj)
    if value_float is not None and any(
        token_str in metric_name_str
        for token_str in ["return", "cagr", "drawdown", "var", "cvar", "ev", "gain"]
    ):
        class_str = "pos" if value_float >= 0.0 else "neg"
    class_attr_str = f' class="{class_str}"' if class_str else ""
    return f"<td{class_attr_str}>{_format_metric_value(value_obj, metric_name_str)}</td>"


DAILY_METRIC_LABEL_DICT = {
    "expected_daily_return_float": "Expected daily return",
    "annualized_ev_float": "Annualized EV",
    "terminal_return_float": "Terminal return",
    "cagr_float": "CAGR",
    "annual_volatility_float": "Annual volatility",
    "sharpe_float": "Sharpe",
    "max_drawdown_float": "Max drawdown",
    "mar_float": "MAR",
    "var_95_daily_return_float": "Daily VaR 95",
    "cvar_95_daily_return_float": "Daily CVaR 95",
    "var_99_daily_return_float": "Daily VaR 99",
    "cvar_99_daily_return_float": "Daily CVaR 99",
    "worst_1d_return_float": "Worst 1d return",
    "worst_5d_return_float": "Worst 5d return",
    "worst_21d_return_float": "Worst 21d return",
    "worst_63d_return_float": "Worst 63d return",
}

MONTHLY_METRIC_LABEL_DICT = {
    "monthly_expected_return_float": "Expected monthly return",
    "monthly_volatility_float": "Monthly volatility",
    "monthly_sharpe_float": "Monthly Sharpe",
    "monthly_var_95_return_float": "Monthly VaR 95",
    "monthly_cvar_95_return_float": "Monthly CVaR 95",
    "monthly_var_99_return_float": "Monthly VaR 99",
    "monthly_cvar_99_return_float": "Monthly CVaR 99",
    "worst_21d_return_float": "Worst 1m return",
    "worst_63d_return_float": "Worst 3m return",
    "worst_126d_return_float": "Worst 6m return",
    "worst_252d_return_float": "Worst 12m return",
}


def _metric_label_str(metric_name_str: str, label_dict: dict[str, str] | None = None) -> str:
    if label_dict is None:
        label_dict = DAILY_METRIC_LABEL_DICT
    return label_dict.get(metric_name_str, metric_name_str)


def _format_metric_value(value_obj, metric_name_str: str) -> str:
    if metric_name_str in {"sharpe_float", "monthly_sharpe_float", "mar_float"}:
        return _format_float(value_obj, 2)
    return _format_percent(value_obj)


def _format_percent(value_obj) -> str:
    value_float = _json_float(value_obj)
    if value_float is None:
        return "N/A"
    return f"{value_float:.2%}"


def _format_days_str(value_obj) -> str:
    value_float = _json_float(value_obj)
    if value_float is None:
        return "N/A"
    return f"{value_float:.0f} days"


def _signed_percent_str(value_float: float) -> str:
    return f"{float(value_float):+.0%}"


def _format_float(value_obj, digits_int: int) -> str:
    value_float = _json_float(value_obj)
    if value_float is None:
        return "N/A"
    return f"{value_float:.{digits_int}f}"


def _write_json_file(json_path: Path, data_dict: dict[str, object]) -> None:
    json_path.write_text(
        json.dumps(
            _sanitize_json_obj(data_dict),
            indent=2,
            sort_keys=True,
            default=_json_default_obj,
        ),
        encoding="utf-8",
    )


def _sanitize_json_obj(value_obj):
    if isinstance(value_obj, dict):
        return {
            str(key_obj): _sanitize_json_obj(child_value_obj)
            for key_obj, child_value_obj in value_obj.items()
        }
    if isinstance(value_obj, list | tuple):
        return [_sanitize_json_obj(child_value_obj) for child_value_obj in value_obj]
    if isinstance(value_obj, float) and not np.isfinite(value_obj):
        return None
    if isinstance(value_obj, np.floating):
        value_float = float(value_obj)
        return value_float if np.isfinite(value_float) else None
    if isinstance(value_obj, np.integer):
        return int(value_obj)
    if isinstance(value_obj, pd.Timestamp):
        return value_obj.isoformat()
    if isinstance(value_obj, Path):
        return str(value_obj)
    return value_obj


def _json_default_obj(value_obj):
    return _sanitize_json_obj(value_obj)


def _json_float(value_obj):
    try:
        value_float = float(value_obj)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(value_float):
        return None
    return value_float


def _compact_dict(raw_dict: dict[str, object]) -> dict[str, object]:
    return {
        key_str: value_obj
        for key_str, value_obj in raw_dict.items()
        if value_obj is not None
    }


def _records_from_df(record_df: pd.DataFrame) -> list[dict[str, object]]:
    if record_df is None or len(record_df) == 0:
        return []
    record_list: list[dict[str, object]] = []
    for _, row_ser in record_df.iterrows():
        row_dict: dict[str, object] = {}
        for field_str, value_obj in row_ser.items():
            if field_str.endswith("_int") and not pd.isna(value_obj):
                row_dict[str(field_str)] = int(value_obj)
            elif pd.isna(value_obj):
                row_dict[str(field_str)] = None
            else:
                row_dict[str(field_str)] = value_obj
        record_list.append(row_dict)
    return record_list


def _date_or_none_str(value_obj) -> str | None:
    if value_obj is None or pd.isna(value_obj):
        return None
    return pd.Timestamp(value_obj).date().isoformat()


__all__ = [
    "DEFAULT_CONFIDENCE_LEVEL_FLOAT",
    "DEFAULT_PRIMARY_MEAN_BLOCK_LENGTH_INT",
    "DEFAULT_RANDOM_SEED_INT",
    "DEFAULT_SENSITIVITY_BLOCK_LENGTH_TUPLE",
    "DEFAULT_SIMULATION_COUNT_INT",
    "DEFAULT_HORIZON_YEAR_TUPLE",
    "DEFAULT_INVESTOR_HORIZON_YEAR_TUPLE",
    "DEFAULT_UPSIDE_THRESHOLD_TUPLE",
    "RISK_ANALYSIS_TYPE_STR",
    "RISK_ANALYSIS_SCHEMA_VERSION_INT",
    "RiskAnalysis",
    "RiskAnalysisResult",
    "build_bootstrap_equity_path_df",
    "build_bootstrap_interval_df",
    "build_bootstrap_path_metric_df",
    "build_horizon_probability_df",
    "build_investor_scenario_df",
    "build_observed_calendar_month_df",
    "build_return_histogram_df",
    "compute_path_metric_dict",
    "extract_realized_return_ser",
    "save_risk_analysis_results",
    "stationary_bootstrap_index_mat",
]
