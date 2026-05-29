"""
Post-run risk diagnostics for completed strategy runs.

RiskAnalysis is report-only. It reads realized strategy returns after a
completed vanilla backtest and never changes strategy, order, fill, sizing, or
live execution semantics.

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
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from alpha.engine.report import build_research_output_path
from alpha.engine.strategy import Strategy


RISK_ANALYSIS_TYPE_STR = "risk_analysis"
RETURN_HISTOGRAM_CSV_FILENAME_STR = "return_histogram.csv"
BOOTSTRAP_EQUITY_PATH_CSV_FILENAME_STR = "bootstrap_equity_paths.csv"
BOOTSTRAP_PATH_METRIC_CSV_FILENAME_STR = "bootstrap_path_metrics.csv"
BOOTSTRAP_INTERVAL_CSV_FILENAME_STR = "bootstrap_metric_intervals.csv"
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
DEFAULT_ROLLING_LOSS_WINDOW_TUPLE = (1, 5, 21, 63, 126, 252)
DEFAULT_TIME_UNDERWATER_BREACH_MONTH_TUPLE = (3, 6, 12, 24)
TRADING_DAYS_PER_MONTH_INT = 21
DEFAULT_RETURN_HISTOGRAM_BIN_COUNT_INT = 80
DEFAULT_EQUITY_PATH_SAMPLE_COUNT_INT = 100
TRADING_DAYS_PER_YEAR_FLOAT = 252.0
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
# Drawdown depth: |1-in-20 bad-case max drawdown| (p05 of max_drawdown dist).
DRAWDOWN_DEPTH_GREEN_MAX_FLOAT = 0.20
DRAWDOWN_DEPTH_AMBER_MAX_FLOAT = 0.35
# Time underwater: P(longest underwater stretch >= 12 trading months).
UNDERWATER_12M_GREEN_MAX_FLOAT = 0.20
UNDERWATER_12M_AMBER_MAX_FLOAT = 0.50
# Worst rolling 12-month return: |1-in-20 bad-case worst-12m| (p05 of worst_252d).
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
    output_dir_path: Path | None = None


def extract_realized_return_ser(strategy_obj: Strategy) -> pd.Series:
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
) -> np.ndarray:
    if sample_size_int <= 0:
        raise ValueError("sample_size_int must be positive.")
    if simulation_count_int <= 0:
        raise ValueError("simulation_count_int must be positive.")
    if mean_block_length_int <= 0:
        raise ValueError("mean_block_length_int must be positive.")

    rng_obj = np.random.default_rng(int(random_seed_int))
    restart_probability_float = 1.0 / float(mean_block_length_int)
    index_mat = np.empty((int(simulation_count_int), int(sample_size_int)), dtype=np.int64)

    for simulation_idx_int in range(int(simulation_count_int)):
        current_index_int = 0
        for step_idx_int in range(int(sample_size_int)):
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
    return float((finite_metric_vec <= float(observed_value_float)).mean())


def save_risk_analysis_results(
    risk_result_obj: RiskAnalysisResult,
    output_dir_str: str = "results",
) -> Path:
    output_dir_path = build_research_output_path(
        output_dir_str,
        "strategy",
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

    _write_json_file(output_dir_path / SUMMARY_FILENAME_STR, risk_result_obj.summary_dict)
    _write_json_file(output_dir_path / RUN_INFO_FILENAME_STR, _build_run_info_dict(risk_result_obj))
    _write_json_file(output_dir_path / METADATA_FILENAME_STR, _build_metadata_dict(risk_result_obj))
    (output_dir_path / REPORT_FILENAME_STR).write_text(
        _build_report_html_str(risk_result_obj),
        encoding="utf-8",
    )

    risk_result_obj.output_dir_path = output_dir_path
    return output_dir_path


class RiskAnalysis:
    def __init__(
        self,
        strategy_obj: Strategy,
        *,
        source_strategy_ref_str: str = "",
        output_dir_str: str = "results",
        save_output_bool: bool = True,
        primary_mean_block_length_int: int = DEFAULT_PRIMARY_MEAN_BLOCK_LENGTH_INT,
        mean_block_length_tuple: Sequence[int] = DEFAULT_SENSITIVITY_BLOCK_LENGTH_TUPLE,
        simulation_count_int: int = DEFAULT_SIMULATION_COUNT_INT,
        random_seed_int: int = DEFAULT_RANDOM_SEED_INT,
        confidence_level_float: float = DEFAULT_CONFIDENCE_LEVEL_FLOAT,
        drawdown_threshold_tuple: Sequence[float] = DEFAULT_DRAWDOWN_THRESHOLD_TUPLE,
        rolling_loss_window_tuple: Sequence[int] = DEFAULT_ROLLING_LOSS_WINDOW_TUPLE,
        time_underwater_breach_month_tuple: Sequence[int] = DEFAULT_TIME_UNDERWATER_BREACH_MONTH_TUPLE,
    ):
        self.strategy_obj = strategy_obj
        self.source_strategy_ref_str = str(source_strategy_ref_str)
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
        self.rolling_loss_window_tuple = tuple(int(value_int) for value_int in rolling_loss_window_tuple)
        self.time_underwater_breach_month_tuple = tuple(
            int(value_int) for value_int in time_underwater_breach_month_tuple
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
        summary_dict = _build_summary_dict(
            strategy_obj=self.strategy_obj,
            realized_return_ser=realized_return_ser,
            bootstrap_path_metric_df=bootstrap_path_metric_df,
            bootstrap_interval_df=bootstrap_interval_df,
            observed_metric_dict=observed_metric_dict,
            primary_mean_block_length_int=self.primary_mean_block_length_int,
            mean_block_length_tuple=self.mean_block_length_tuple,
            simulation_count_int=self.simulation_count_int,
            random_seed_int=self.random_seed_int,
            confidence_level_float=self.confidence_level_float,
            drawdown_threshold_tuple=self.drawdown_threshold_tuple,
            time_underwater_breach_month_tuple=self.time_underwater_breach_month_tuple,
        )

        risk_result_obj = RiskAnalysisResult(
            strategy_name_str=str(self.strategy_obj.name),
            source_strategy_ref_str=self.source_strategy_ref_str,
            realized_return_ser=realized_return_ser,
            return_histogram_df=return_histogram_df,
            bootstrap_equity_path_df=bootstrap_equity_path_df,
            bootstrap_path_metric_df=bootstrap_path_metric_df,
            bootstrap_interval_df=bootstrap_interval_df,
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


def _build_summary_dict(
    *,
    strategy_obj: Strategy,
    realized_return_ser: pd.Series,
    bootstrap_path_metric_df: pd.DataFrame,
    bootstrap_interval_df: pd.DataFrame,
    observed_metric_dict: dict[str, object],
    primary_mean_block_length_int: int,
    mean_block_length_tuple: Sequence[int],
    simulation_count_int: int,
    random_seed_int: int,
    confidence_level_float: float,
    drawdown_threshold_tuple: Sequence[float],
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
        "observed_metrics": _compact_dict(observed_metric_dict),
        "primary_intervals": _primary_interval_dict(primary_interval_df),
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

    # --- Edge: did the strategy reliably end in profit, and is the edge
    # statistically separated from zero? ---
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
        # Profitable most resamples, but Sharpe CI still includes zero: do not
        # call the edge proven.
        edge_status_str = VERDICT_STATUS_AMBER_STR
    if terminal_loss_float is None:
        edge_value_str = "N/A"
        edge_conclusion_str = "Not enough data to assess edge."
    else:
        profit_probability_float = 1.0 - terminal_loss_float
        edge_value_str = f"{profit_probability_float:.0%} profitable resamples"
        if edge_status_str == VERDICT_STATUS_GREEN_STR:
            edge_conclusion_str = (
                f"Ended profitable in {profit_probability_float:.0%} of resamples — edge looks real."
            )
        elif edge_status_str == VERDICT_STATUS_AMBER_STR:
            edge_conclusion_str = (
                f"Profitable in {profit_probability_float:.0%} of resamples, "
                "but not clearly separated from zero — treat edge as tentative."
            )
        else:
            edge_conclusion_str = (
                f"Profitable in only {profit_probability_float:.0%} of resamples — close to a coin flip."
            )
    row_list.append(
        {
            "label_str": "Edge",
            "status_str": edge_status_str,
            "value_str": edge_value_str,
            "conclusion_str": edge_conclusion_str,
        }
    )

    # --- Drawdown depth: 1-in-20 bad-case max drawdown (p05 of distribution). ---
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
        drawdown_value_str = f"{drawdown_bad_float:.0%} (1-in-20 bad case)"
        word_str = {
            VERDICT_STATUS_GREEN_STR: "tolerable",
            VERDICT_STATUS_AMBER_STR: "painful",
            VERDICT_STATUS_RED_STR: "severe",
        }.get(drawdown_status_str, "")
        drawdown_conclusion_str = (
            f"Bad-case drawdown around {drawdown_bad_float:.0%} — {word_str}."
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
        underwater_value_str = f"{underwater_12m_float:.0%} chance >= 12m underwater"
        if underwater_status_str == VERDICT_STATUS_GREEN_STR:
            underwater_conclusion_str = (
                f"{underwater_12m_float:.0%} chance of a 12-month+ underwater stretch — manageable."
            )
        elif underwater_status_str == VERDICT_STATUS_AMBER_STR:
            underwater_conclusion_str = (
                f"{underwater_12m_float:.0%} chance of a 12-month+ underwater stretch — be prepared."
            )
        else:
            underwater_conclusion_str = (
                f"{underwater_12m_float:.0%} chance of a 12-month+ underwater stretch — expect long pain."
            )
    row_list.append(
        {
            "label_str": "Time underwater",
            "status_str": underwater_status_str,
            "value_str": underwater_value_str,
            "conclusion_str": underwater_conclusion_str,
        }
    )

    # --- Worst year: 1-in-20 bad-case worst rolling 12-month return. ---
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
        worst_year_value_str = f"{worst_year_bad_float:.0%} (1-in-20 bad case)"
        word_str = {
            VERDICT_STATUS_GREEN_STR: "mild",
            VERDICT_STATUS_AMBER_STR: "significant",
            VERDICT_STATUS_RED_STR: "severe",
        }.get(worst_year_status_str, "")
        worst_year_conclusion_str = (
            f"Worst rolling year around {worst_year_bad_float:.0%} — {word_str}."
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
    rolling_return_list = [
        float(np.prod(1.0 + clean_return_vec[start_int : start_int + window_int]) - 1.0)
        for start_int in range(0, clean_return_vec.size - window_int + 1)
    ]
    return float(np.min(rolling_return_list))


def _build_run_info_dict(risk_result_obj: RiskAnalysisResult) -> dict[str, object]:
    summary_dict = risk_result_obj.summary_dict
    return {
        "entity_type": "strategy",
        "entity_id": risk_result_obj.strategy_name_str,
        "analysis_type": RISK_ANALYSIS_TYPE_STR,
        "parameters": {
            "source_strategy_ref": risk_result_obj.source_strategy_ref_str,
            "primary_mean_block_length_int": summary_dict.get("primary_mean_block_length_int"),
            "mean_block_length_list": summary_dict.get("mean_block_length_list"),
            "simulation_count_int": summary_dict.get("simulation_count_int"),
            "random_seed_int": summary_dict.get("random_seed_int"),
            "confidence_level_float": summary_dict.get("confidence_level_float"),
        },
    }


def _build_metadata_dict(risk_result_obj: RiskAnalysisResult) -> dict[str, object]:
    return {
        "artifact_type": RISK_ANALYSIS_TYPE_STR,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "strategy_name": risk_result_obj.strategy_name_str,
        "source_strategy_ref": risk_result_obj.source_strategy_ref_str,
        "return_count": int(len(risk_result_obj.realized_return_ser)),
    }


def _build_report_html_str(risk_result_obj: RiskAnalysisResult) -> str:
    summary_dict = risk_result_obj.summary_dict
    strategy_name_html = html.escape(risk_result_obj.strategy_name_str)
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
<style>
body {{ font-family: Arial, sans-serif; color: #1d2733; margin: 0; background: #f3f5f7; }}
.wrap {{ max-width: 1180px; margin: 0 auto; padding: 30px 28px 42px; }}
h1 {{ margin: 0 0 8px; font-size: 30px; letter-spacing: 0; }}
h2 {{ margin: 0 0 12px; font-size: 19px; letter-spacing: 0; }}
p {{ color: #536170; line-height: 1.45; }}
.meta {{ color: #536170; margin-bottom: 18px; }}
.tile-grid {{ display: grid; grid-template-columns: repeat(5, minmax(140px, 1fr)); gap: 10px; margin: 18px 0 22px; }}
.tile {{ background: #fff; border: 1px solid #d9e0e7; border-radius: 8px; padding: 13px 14px; }}
.tile-label {{ color: #637184; font-size: 12px; margin-bottom: 6px; }}
.tile-value {{ font-size: 20px; font-weight: 700; }}
.tile-sub {{ color: #637184; font-size: 11px; margin-top: 4px; line-height: 1.35; }}
.caveat {{ background: #fff8e1; border: 1px solid #f0d68c; border-radius: 8px; padding: 12px 14px; margin: 14px 0 18px; color: #5c4a14; font-size: 13px; line-height: 1.45; }}
.verdict-panel {{ background: #fff; border: 1px solid #d9e0e7; border-left: 5px solid #8590a2; border-radius: 8px; padding: 14px 16px; margin: 14px 0 18px; }}
.verdict-title {{ font-size: 15px; font-weight: 700; margin: 0 0 10px; color: #1d2733; }}
.verdict-row {{ display: flex; align-items: baseline; gap: 10px; padding: 7px 0; border-bottom: 1px solid #eef2f6; }}
.verdict-row:last-of-type {{ border-bottom: none; }}
.verdict-dot {{ flex: 0 0 auto; width: 11px; height: 11px; border-radius: 50%; margin-top: 3px; }}
.verdict-label {{ flex: 0 0 130px; font-weight: 700; font-size: 13px; }}
.verdict-value {{ flex: 0 0 200px; font-size: 13px; color: #334155; }}
.verdict-text {{ flex: 1 1 auto; font-size: 13px; color: #536170; }}
.verdict-disclaimer {{ color: #8590a2; font-size: 11px; margin-top: 10px; }}
.v-green {{ background: #22a06b; }}
.v-amber {{ background: #f5a524; }}
.v-red {{ background: #c9372c; }}
.v-na {{ background: #8590a2; }}
.subtitle {{ color: #637184; font-size: 12px; margin: -4px 0 12px; }}
.footnote {{ color: #637184; font-size: 11px; margin-top: 8px; }}
.footnote sup {{ color: #b42318; font-weight: 700; }}
.section {{ background: #fff; border: 1px solid #d9e0e7; border-radius: 8px; padding: 20px; margin: 16px 0; }}
.chart {{ width: 100%; max-width: 1080px; height: auto; display: block; }}
.legend {{ color: #536170; font-size: 12px; margin-top: 8px; }}
table {{ border-collapse: collapse; width: 100%; background: #fff; font-size: 13px; }}
th, td {{ border-bottom: 1px solid #e4e8ee; padding: 9px 10px; text-align: right; vertical-align: top; }}
th:first-child, td:first-child {{ text-align: left; }}
th {{ background: #eef2f6; color: #334155; font-weight: 700; }}
.neg {{ color: #b42318; }}
.pos {{ color: #067647; }}
.scroll {{ overflow-x: auto; }}
@media (max-width: 850px) {{ .tile-grid {{ grid-template-columns: repeat(2, minmax(140px, 1fr)); }} .wrap {{ padding: 20px 14px; }} }}
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
<div class="caveat">
<strong>What this report is.</strong> This block-bootstraps the strategy's <em>realized</em> daily returns to quantify how much of the realized path was ordering luck for a history of this length. The intervals, equity fan, and breach probabilities describe alternative orderings of the returns we already observed. <strong>What it is not.</strong> It does not simulate the strategy in regimes outside the sample, does not model regime-conditional dependence, and does not extend the horizon beyond {summary_dict.get("return_count_int")} days. Use it to calibrate confidence in the realized record, not as forward stress testing.
</div>
{_verdict_panel_html(summary_dict.get("verdict"))}
{_summary_tiles_html(summary_dict)}
<div class="section">
<h2>Returns Histogram</h2>
<div class="subtitle">Distribution of realized daily returns over the sample window. Mean left of median signals negative skew (occasional larger losses).</div>
{_return_histogram_svg(risk_result_obj.return_histogram_df, mean_float=return_mean_float, median_float=return_median_float)}
</div>
<div class="section">
<h2>Monte Carlo Equity Paths</h2>
<div class="subtitle">Simulated paths have the same length as the realized history; this shows ordering risk over the historical horizon, not forward horizon.</div>
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
<h2>Drawdown and Time-Underwater Breach Probabilities</h2>
<div class="subtitle">Probability across bootstrap paths that the realized event occurred at least once. Time-underwater thresholds are in trading months of {TRADING_DAYS_PER_MONTH_INT} days.</div>
<div class="scroll">{_breach_table_html(summary_dict.get("primary_drawdown_breach_probabilities", {}), summary_dict.get("primary_terminal_loss_probability_float"), summary_dict.get("primary_time_underwater_breach_probabilities", {}))}</div>
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
        "<div class=\"verdict-title\">Read-first verdict</div>"
        + "".join(row_html_list)
        + "<div class=\"verdict-disclaimer\">Bands are generic reference heuristics for fast reading, "
        "not trade advice or account-calibrated limits. See ASSUMPTIONS_AND_GAPS.</div>"
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
    y_max_float = float(max(1, int(histogram_df["count_int"].max())))
    axis_y_float = top_float + plot_height_float

    gridline_count_int = 4
    gridline_html_list = []
    for tick_idx_int in range(gridline_count_int + 1):
        tick_count_float = y_max_float * tick_idx_int / gridline_count_int
        tick_y_float = _scale_float(
            tick_count_float, 0.0, y_max_float, axis_y_float, top_float
        )
        if tick_idx_int > 0:
            gridline_html_list.append(
                f"<line x1=\"{left_float:.1f}\" y1=\"{tick_y_float:.1f}\" "
                f"x2=\"{left_float + plot_width_float:.1f}\" y2=\"{tick_y_float:.1f}\" "
                "stroke=\"#e4e8ee\" stroke-width=\"1\" />"
            )
        gridline_html_list.append(
            f"<text x=\"{left_float - 6:.1f}\" y=\"{tick_y_float + 4:.1f}\" "
            f"fill=\"#536170\" font-size=\"11\" text-anchor=\"end\">{int(round(tick_count_float))}</text>"
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
            f"fill=\"#536170\" font-size=\"11\" text-anchor=\"{anchor_str}\">{_format_percent(tick_return_float)}</text>"
        )

    bar_html_list = []
    for _, row_ser in histogram_df.iterrows():
        bin_left_float = float(row_ser["bin_left_float"])
        bin_right_float = float(row_ser["bin_right_float"])
        count_float = float(row_ser["count_int"])
        x_float = _scale_float(bin_left_float, x_min_float, x_max_float, left_float, left_float + plot_width_float)
        x2_float = _scale_float(bin_right_float, x_min_float, x_max_float, left_float, left_float + plot_width_float)
        y_float = _scale_float(count_float, 0.0, y_max_float, axis_y_float, top_float)
        bar_width_float = max(1.0, x2_float - x_float - 1.0)
        bar_height_float = axis_y_float - y_float
        fill_str = "#1f7a8c" if float(row_ser["bin_mid_float"]) >= 0.0 else "#b84a4a"
        bar_html_list.append(
            f"<rect x=\"{x_float:.2f}\" y=\"{y_float:.2f}\" width=\"{bar_width_float:.2f}\" height=\"{bar_height_float:.2f}\" fill=\"{fill_str}\" opacity=\"0.82\" />"
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
            f"y2=\"{axis_y_float:.1f}\" stroke=\"#1d4ed8\" stroke-width=\"2\" />"
        )
    median_value_float = _json_float(median_float)
    if median_value_float is not None:
        median_x_float = _scale_float(
            median_value_float, x_min_float, x_max_float, left_float, left_float + plot_width_float
        )
        reference_line_html_list.append(
            f"<line x1=\"{median_x_float:.1f}\" y1=\"{top_float:.1f}\" x2=\"{median_x_float:.1f}\" "
            f"y2=\"{axis_y_float:.1f}\" stroke=\"#7c3aed\" stroke-width=\"2\" stroke-dasharray=\"5 3\" />"
        )

    legend_y_float = top_float + 6.0
    legend_x_float = left_float + 10.0
    legend_html_list = [
        f"<line x1=\"{legend_x_float:.1f}\" y1=\"{legend_y_float:.1f}\" x2=\"{legend_x_float + 18:.1f}\" y2=\"{legend_y_float:.1f}\" stroke=\"#1d4ed8\" stroke-width=\"2\" />",
        f"<text x=\"{legend_x_float + 24:.1f}\" y=\"{legend_y_float + 4:.1f}\" fill=\"#536170\" font-size=\"11\">mean</text>",
        f"<line x1=\"{legend_x_float + 70:.1f}\" y1=\"{legend_y_float:.1f}\" x2=\"{legend_x_float + 88:.1f}\" y2=\"{legend_y_float:.1f}\" stroke=\"#7c3aed\" stroke-width=\"2\" stroke-dasharray=\"5 3\" />",
        f"<text x=\"{legend_x_float + 94:.1f}\" y=\"{legend_y_float + 4:.1f}\" fill=\"#536170\" font-size=\"11\">median</text>",
        f"<line x1=\"{legend_x_float + 150:.1f}\" y1=\"{legend_y_float:.1f}\" x2=\"{legend_x_float + 168:.1f}\" y2=\"{legend_y_float:.1f}\" stroke=\"#111827\" stroke-width=\"1\" stroke-dasharray=\"4 4\" />",
        f"<text x=\"{legend_x_float + 174:.1f}\" y=\"{legend_y_float + 4:.1f}\" fill=\"#536170\" font-size=\"11\">zero</text>",
    ]

    y_axis_title_x_float = 18.0
    y_axis_title_y_float = top_float + plot_height_float / 2.0
    return (
        f"<svg class=\"chart\" viewBox=\"0 0 {width_float:.0f} {height_float:.0f}\" role=\"img\" aria-label=\"Returns histogram\">"
        f"<rect x=\"0\" y=\"0\" width=\"{width_float:.0f}\" height=\"{height_float:.0f}\" fill=\"#ffffff\" />"
        + "".join(gridline_html_list)
        + f"<line x1=\"{left_float:.1f}\" y1=\"{axis_y_float:.1f}\" x2=\"{left_float + plot_width_float:.1f}\" y2=\"{axis_y_float:.1f}\" stroke=\"#8291a3\" stroke-width=\"1\" />"
        + f"<line x1=\"{left_float:.1f}\" y1=\"{top_float:.1f}\" x2=\"{left_float:.1f}\" y2=\"{axis_y_float:.1f}\" stroke=\"#8291a3\" stroke-width=\"1\" />"
        + "".join(bar_html_list)
        + f"<line x1=\"{zero_x_float:.1f}\" y1=\"{top_float:.1f}\" x2=\"{zero_x_float:.1f}\" y2=\"{axis_y_float:.1f}\" stroke=\"#111827\" stroke-width=\"1\" stroke-dasharray=\"4 4\" />"
        + "".join(reference_line_html_list)
        + "".join(x_tick_html_list)
        + "".join(legend_html_list)
        + f"<text x=\"{left_float + plot_width_float / 2.0:.1f}\" y=\"{height_float - 8:.1f}\" fill=\"#334155\" font-size=\"12\" text-anchor=\"middle\">Daily return</text>"
        + f"<text x=\"{y_axis_title_x_float:.1f}\" y=\"{y_axis_title_y_float:.1f}\" fill=\"#334155\" font-size=\"12\" text-anchor=\"middle\" transform=\"rotate(-90 {y_axis_title_x_float:.1f} {y_axis_title_y_float:.1f})\">Count of days</text>"
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
                "stroke=\"#e4e8ee\" stroke-width=\"1\" />"
            )
        gridline_html_list.append(
            f"<text x=\"{left_float - 6:.1f}\" y=\"{tick_y_float + 4:.1f}\" "
            f"fill=\"#536170\" font-size=\"11\" text-anchor=\"end\">{np.exp(tick_log_float):.2f}x</text>"
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
                "#7b8794",
                0.18,
                1.0,
            )
        )
    overlay_tuple = (
        ("p05", "#b42318", 0.95, 2.2, "p05"),
        ("p50", "#111827", 0.95, 2.2, "p50"),
        ("p95", "#067647", 0.95, 2.2, "p95"),
        ("observed", "#1d4ed8", 1.0, 2.4, "observed"),
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
        "fill=\"#ffffff\" fill-opacity=\"0.92\" stroke=\"#d9e0e7\" stroke-width=\"1\" rx=\"4\" />"
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
            f"fill=\"#334155\" font-size=\"11\">{html.escape(label_str)}</text>"
        )

    y_axis_title_x_float = 18.0
    y_axis_title_y_float = top_float + plot_height_float / 2.0
    return (
        f"<svg class=\"chart\" viewBox=\"0 0 {width_float:.0f} {height_float:.0f}\" role=\"img\" aria-label=\"Monte Carlo equity paths\">"
        f"<rect x=\"0\" y=\"0\" width=\"{width_float:.0f}\" height=\"{height_float:.0f}\" fill=\"#ffffff\" />"
        + "".join(gridline_html_list)
        + f"<line x1=\"{left_float:.1f}\" y1=\"{axis_y_float:.1f}\" x2=\"{left_float + plot_width_float:.1f}\" y2=\"{axis_y_float:.1f}\" stroke=\"#8291a3\" stroke-width=\"1\" />"
        + f"<line x1=\"{left_float:.1f}\" y1=\"{top_float:.1f}\" x2=\"{left_float:.1f}\" y2=\"{axis_y_float:.1f}\" stroke=\"#8291a3\" stroke-width=\"1\" />"
        + "".join(path_html_list)
        + f"<text x=\"{left_float:.1f}\" y=\"{axis_y_float + 16:.1f}\" fill=\"#536170\" font-size=\"11\">0</text>"
        + f"<text x=\"{left_float + plot_width_float:.1f}\" y=\"{axis_y_float + 16:.1f}\" fill=\"#536170\" font-size=\"11\" text-anchor=\"end\">{int(x_max_float)} days</text>"
        + f"<text x=\"{left_float + plot_width_float / 2.0:.1f}\" y=\"{height_float - 8:.1f}\" fill=\"#334155\" font-size=\"12\" text-anchor=\"middle\">Days from start</text>"
        + f"<text x=\"{y_axis_title_x_float:.1f}\" y=\"{y_axis_title_y_float:.1f}\" fill=\"#334155\" font-size=\"12\" text-anchor=\"middle\" transform=\"rotate(-90 {y_axis_title_x_float:.1f} {y_axis_title_y_float:.1f})\">Equity multiple (log)</text>"
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


def _breach_table_html(
    breach_probability_dict: dict[str, object],
    terminal_loss_probability_float: object,
    time_underwater_breach_probability_dict: dict[str, object] | None = None,
) -> str:
    row_html_list = [
        f"<tr><td>{html.escape(str(key_str))}</td><td>{_format_percent(value_obj)}</td></tr>"
        for key_str, value_obj in breach_probability_dict.items()
    ]
    row_html_list.append(
        "<tr><td>terminal_return_lt_0</td>"
        f"<td>{_format_percent(terminal_loss_probability_float)}</td></tr>"
    )
    if time_underwater_breach_probability_dict:
        for key_str, value_obj in time_underwater_breach_probability_dict.items():
            row_html_list.append(
                f"<tr><td>{html.escape(str(key_str))}</td>"
                f"<td>{_format_percent(value_obj)}</td></tr>"
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
        for token_str in ["return", "cagr", "drawdown", "var", "cvar", "ev"]
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
    if metric_name_str in {"sharpe_float", "mar_float"}:
        return _format_float(value_obj, 2)
    return _format_percent(value_obj)


def _format_percent(value_obj) -> str:
    value_float = _json_float(value_obj)
    if value_float is None:
        return "N/A"
    return f"{value_float:.2%}"


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
    "RISK_ANALYSIS_TYPE_STR",
    "RiskAnalysis",
    "RiskAnalysisResult",
    "build_bootstrap_equity_path_df",
    "build_bootstrap_interval_df",
    "build_bootstrap_path_metric_df",
    "build_return_histogram_df",
    "compute_path_metric_dict",
    "extract_realized_return_ser",
    "save_risk_analysis_results",
    "stationary_bootstrap_index_mat",
]
