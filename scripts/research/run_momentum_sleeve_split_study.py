"""Analyze an NDX/VXN versus Russell 1000 MOSAIC momentum sleeve split.

The study reuses completed, net-of-modeled-cost return streams from a saved
Portfolio artifact. It does not rerun either strategy or change live wiring.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm


REPO_ROOT_PATH = Path(__file__).resolve().parents[2]
if str(REPO_ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_PATH))


DAYS_IN_YEAR_INT = 252
HAC_MAX_LAG_INT = 9
ROLLING_CORRELATION_WINDOW_INT = 126
BOOTSTRAP_BLOCK_DAY_TUPLE = (21, 63, 126)
BOOTSTRAP_ITERATION_INT = 1_000
BOOTSTRAP_SEED_INT = 20260817

NDX_STRATEGY_NAME_STR = "strategy_mo_atr_normalized_ndx_vxn_scaled"
MOSAIC_STRATEGY_NAME_STR = "strategy_mo_mosaic_russell1000"


def parse_arguments() -> argparse.Namespace:
    parser_obj = argparse.ArgumentParser(
        description="Analyze a fixed allocation between NDX/VXN and MOSAIC momentum."
    )
    parser_obj.add_argument("source_portfolio_pickle_path", type=Path)
    parser_obj.add_argument("output_dir_path", type=Path)
    parser_obj.add_argument(
        "--bootstrap-iterations",
        type=int,
        default=BOOTSTRAP_ITERATION_INT,
    )
    return parser_obj.parse_args()


def load_pickle(pickle_path: Path) -> object:
    with pickle_path.open("rb") as pickle_file_obj:
        return pickle.load(pickle_file_obj)


def find_strategy_obj(portfolio_obj: object, strategy_name_str: str) -> object:
    matching_strategy_list = [
        strategy_obj
        for strategy_obj in portfolio_obj.strategies
        if strategy_obj.name == strategy_name_str
    ]
    if len(matching_strategy_list) != 1:
        raise RuntimeError(
            f"Expected one strategy named {strategy_name_str}, "
            f"found {len(matching_strategy_list)}."
        )
    return matching_strategy_list[0]


def rebalance_date_index(
    date_index: pd.DatetimeIndex,
    rebalance_frequency_str: str | None,
) -> pd.DatetimeIndex:
    if rebalance_frequency_str is None:
        return pd.DatetimeIndex([])
    frequency_by_policy_dict = {
        "monthly": "MS",
        "quarterly": "QS",
        "annually": "YS",
    }
    if rebalance_frequency_str not in frequency_by_policy_dict:
        raise ValueError(f"Unsupported rebalance frequency: {rebalance_frequency_str}")

    calendar_rebalance_date_index = pd.date_range(
        start=date_index[0],
        end=date_index[-1],
        freq=frequency_by_policy_dict[rebalance_frequency_str],
    )
    # *** CRITICAL *** timing-sensitive: each calendar marker maps to the first
    # actual session on or after it. The reset uses only prior-close book value.
    rebalance_position_arr = np.searchsorted(
        date_index,
        calendar_rebalance_date_index,
        side="left",
    )
    rebalance_position_arr = rebalance_position_arr[
        rebalance_position_arr < len(date_index)
    ]
    mapped_rebalance_date_index = date_index[np.unique(rebalance_position_arr)]
    return pd.DatetimeIndex(
        mapped_rebalance_date_index[mapped_rebalance_date_index > date_index[0]]
    )


def simulate_split_book(
    strategy_return_df: pd.DataFrame,
    mosaic_weight_float: float,
    rebalance_frequency_str: str | None,
) -> tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    if not 0.0 <= mosaic_weight_float <= 1.0:
        raise ValueError("mosaic_weight_float must be between 0 and 1.")
    if list(strategy_return_df.columns) != ["ndx", "mosaic"]:
        raise ValueError("strategy_return_df columns must be ['ndx', 'mosaic'].")
    if strategy_return_df.isna().any().any():
        raise ValueError("strategy_return_df must not contain missing returns.")

    target_weight_ser = pd.Series(
        {
            "ndx": 1.0 - float(mosaic_weight_float),
            "mosaic": float(mosaic_weight_float),
        },
        dtype=float,
    )
    sleeve_equity_df = (
        (1.0 + strategy_return_df).cumprod().mul(target_weight_ser, axis=1)
    )

    for rebalance_date_ts in rebalance_date_index(
        pd.DatetimeIndex(strategy_return_df.index),
        rebalance_frequency_str,
    ):
        rebalance_position_int = int(strategy_return_df.index.get_loc(rebalance_date_ts))
        previous_date_ts = strategy_return_df.index[rebalance_position_int - 1]
        previous_book_value_float = float(sleeve_equity_df.loc[previous_date_ts].sum())
        future_return_df = strategy_return_df.iloc[rebalance_position_int:]
        sleeve_equity_df.loc[rebalance_date_ts:] = (
            (1.0 + future_return_df)
            .cumprod()
            .mul(previous_book_value_float * target_weight_ser, axis=1)
        )

    book_equity_ser = sleeve_equity_df.sum(axis=1).rename("book_equity")
    book_return_ser = (
        book_equity_ser.pct_change(fill_method=None).fillna(0.0).rename("book_return")
    )
    realized_sleeve_weight_df = sleeve_equity_df.div(book_equity_ser, axis=0)
    return book_return_ser, sleeve_equity_df, realized_sleeve_weight_df


def calculate_metrics(
    return_ser: pd.Series,
    benchmark_return_ser: pd.Series,
) -> dict[str, float]:
    aligned_return_df = pd.concat(
        [return_ser.rename("strategy"), benchmark_return_ser.rename("benchmark")],
        axis=1,
    ).dropna()
    strategy_return_ser = aligned_return_df["strategy"].astype(float)
    observation_count_int = int(len(strategy_return_ser))
    equity_ser = (1.0 + strategy_return_ser).cumprod()
    drawdown_ser = equity_ser.div(equity_ser.cummax()).sub(1.0)
    annualized_return_float = float(
        equity_ser.iloc[-1] ** (DAYS_IN_YEAR_INT / observation_count_int) - 1.0
    )
    annualized_volatility_float = float(
        strategy_return_ser.std(ddof=1) * np.sqrt(DAYS_IN_YEAR_INT)
    )
    sharpe_float = float(
        strategy_return_ser.mean()
        / strategy_return_ser.std(ddof=1)
        * np.sqrt(DAYS_IN_YEAR_INT)
    )
    max_drawdown_float = float(drawdown_ser.min())
    mar_float = (
        annualized_return_float / abs(max_drawdown_float)
        if max_drawdown_float < 0.0
        else np.nan
    )

    regression_input_df = sm.add_constant(
        aligned_return_df[["benchmark"]],
        has_constant="add",
    )
    regression_result_obj = sm.OLS(
        aligned_return_df["strategy"],
        regression_input_df,
    ).fit(cov_type="HAC", cov_kwds={"maxlags": HAC_MAX_LAG_INT})

    monthly_return_df = (1.0 + aligned_return_df).resample("ME").prod().sub(1.0)
    return {
        "observation_count_int": observation_count_int,
        "annualized_return_float": annualized_return_float,
        "annualized_volatility_float": annualized_volatility_float,
        "sharpe_float": sharpe_float,
        "max_drawdown_float": max_drawdown_float,
        "mar_float": float(mar_float),
        "daily_market_correlation_float": float(
            aligned_return_df["strategy"].corr(aligned_return_df["benchmark"])
        ),
        "monthly_market_correlation_float": float(
            monthly_return_df["strategy"].corr(monthly_return_df["benchmark"])
        ),
        "market_beta_float": float(regression_result_obj.params["benchmark"]),
        "market_alpha_annualized_float": float(
            regression_result_obj.params["const"] * DAYS_IN_YEAR_INT
        ),
        "market_alpha_hac_t_stat_float": float(regression_result_obj.tvalues["const"]),
        "market_r_squared_float": float(regression_result_obj.rsquared),
    }


def calculate_strategy_exposure_metrics(strategy_obj: object) -> dict[str, float | int]:
    realized_weight_df = strategy_obj.realized_weight_df.fillna(0.0)
    cash_weight_ser = realized_weight_df.get(
        "Cash",
        pd.Series(0.0, index=realized_weight_df.index),
    ).astype(float)
    invested_weight_df = realized_weight_df.drop(columns=["Cash"], errors="ignore").clip(
        lower=0.0
    )
    gross_exposure_ser = invested_weight_df.sum(axis=1)
    normalized_invested_weight_df = invested_weight_df.div(
        gross_exposure_ser.replace(0.0, np.nan),
        axis=0,
    )
    effective_position_count_ser = (
        normalized_invested_weight_df.pow(2)
        .sum(axis=1, min_count=1)
        .pow(-1)
        .replace([np.inf, -np.inf], np.nan)
    )
    negative_cash_weight_ser = cash_weight_ser[cash_weight_ser < 0.0]
    return {
        "average_gross_exposure_float": float(gross_exposure_ser.mean()),
        "median_gross_exposure_float": float(gross_exposure_ser.median()),
        "average_cash_weight_float": float(cash_weight_ser.mean()),
        "negative_cash_day_count_int": int((cash_weight_ser < 0.0).sum()),
        "minimum_cash_weight_float": float(cash_weight_ser.min()),
        "average_negative_cash_weight_float": float(negative_cash_weight_ser.mean()),
        "average_effective_position_count_float": float(
            effective_position_count_ser.dropna().mean()
        ),
    }


def calculate_hac_regression_row(
    model_name_str: str,
    dependent_return_ser: pd.Series,
    explanatory_return_df: pd.DataFrame,
) -> dict[str, float | int | str]:
    regression_df = pd.concat(
        [dependent_return_ser.rename("dependent"), explanatory_return_df],
        axis=1,
    ).dropna()
    regression_result_obj = sm.OLS(
        regression_df["dependent"],
        sm.add_constant(regression_df.drop(columns="dependent"), has_constant="add"),
    ).fit(cov_type="HAC", cov_kwds={"maxlags": HAC_MAX_LAG_INT})
    result_dict: dict[str, float | int | str] = {
        "model_name_str": model_name_str,
        "observation_count_int": int(len(regression_df)),
        "annualized_alpha_float": float(
            regression_result_obj.params["const"] * DAYS_IN_YEAR_INT
        ),
        "alpha_hac_t_stat_float": float(regression_result_obj.tvalues["const"]),
        "r_squared_float": float(regression_result_obj.rsquared),
    }
    for factor_name_str in explanatory_return_df.columns:
        result_dict[f"beta_{factor_name_str}_float"] = float(
            regression_result_obj.params[factor_name_str]
        )
    return result_dict


def calculate_subperiod_metrics_df(
    strategy_return_df: pd.DataFrame,
    blend_return_ser: pd.Series,
    benchmark_return_ser: pd.Series,
) -> pd.DataFrame:
    subperiod_tuple = (
        ("2004-2012", 2004, 2012),
        ("2013-2020", 2013, 2020),
        ("2021-2026 YTD", 2021, 2026),
    )
    row_list: list[dict[str, object]] = []
    combined_return_df = strategy_return_df.assign(blend_50_50=blend_return_ser)
    for subperiod_name_str, start_year_int, end_year_int in subperiod_tuple:
        subperiod_mask_ser = (
            (combined_return_df.index.year >= start_year_int)
            & (combined_return_df.index.year <= end_year_int)
        )
        for series_name_str in combined_return_df.columns:
            metric_dict = calculate_metrics(
                return_ser=combined_return_df.loc[subperiod_mask_ser, series_name_str],
                benchmark_return_ser=benchmark_return_ser.loc[subperiod_mask_ser],
            )
            row_list.append(
                {
                    "subperiod_name_str": subperiod_name_str,
                    "series_name_str": series_name_str,
                    **metric_dict,
                }
            )
    return pd.DataFrame(row_list)


def calculate_conditional_monthly_df(
    daily_analysis_df: pd.DataFrame,
) -> pd.DataFrame:
    monthly_return_df = (
        (1.0 + daily_analysis_df[["ndx", "mosaic", "blend_50_50", "spx"]])
        .resample("ME")
        .prod()
        .sub(1.0)
    )
    monthly_vxn_ser = daily_analysis_df["vxn_close"].resample("ME").mean()
    condition_by_name_dict = {
        "NDX down month": monthly_return_df["ndx"] < 0.0,
        "NDX up month": monthly_return_df["ndx"] >= 0.0,
        "SPX down month": monthly_return_df["spx"] < 0.0,
        "SPX up month": monthly_return_df["spx"] >= 0.0,
        "Average VXN >= 30": monthly_vxn_ser >= 30.0,
        "Average VXN < 20": monthly_vxn_ser < 20.0,
    }
    row_list: list[dict[str, object]] = []
    for condition_name_str, condition_mask_ser in condition_by_name_dict.items():
        conditional_return_df = monthly_return_df.loc[condition_mask_ser]
        row_list.append(
            {
                "condition_name_str": condition_name_str,
                "month_count_int": int(len(conditional_return_df)),
                "average_ndx_return_float": float(conditional_return_df["ndx"].mean()),
                "average_mosaic_return_float": float(
                    conditional_return_df["mosaic"].mean()
                ),
                "average_blend_return_float": float(
                    conditional_return_df["blend_50_50"].mean()
                ),
                "average_blend_minus_ndx_float": float(
                    (
                        conditional_return_df["blend_50_50"]
                        - conditional_return_df["ndx"]
                    ).mean()
                ),
                "mosaic_win_rate_vs_ndx_float": float(
                    (
                        conditional_return_df["mosaic"]
                        > conditional_return_df["ndx"]
                    ).mean()
                ),
            }
        )
    return pd.DataFrame(row_list)


def calculate_relative_log_contribution_df(
    daily_analysis_df: pd.DataFrame,
) -> pd.DataFrame:
    # Exact additive decomposition of relative terminal wealth:
    # sum(log(1 + blend_t) - log(1 + ndx_t)).
    relative_log_return_ser = np.log1p(daily_analysis_df["blend_50_50"]).sub(
        np.log1p(daily_analysis_df["ndx"])
    )
    condition_by_name_dict = {
        "NDX up days": daily_analysis_df["ndx"] > 0.0,
        "NDX down days": daily_analysis_df["ndx"] < 0.0,
        "NDX flat days": daily_analysis_df["ndx"] == 0.0,
        "VXN < 20": daily_analysis_df["vxn_close"] < 20.0,
        "20 <= VXN < 30": daily_analysis_df["vxn_close"].between(
            20.0,
            30.0,
            inclusive="left",
        ),
        "VXN >= 30": daily_analysis_df["vxn_close"] >= 30.0,
    }
    return pd.DataFrame(
        [
            {
                "condition_name_str": condition_name_str,
                "day_count_int": int(condition_mask_ser.sum()),
                "relative_log_wealth_contribution_float": float(
                    relative_log_return_ser.loc[condition_mask_ser].sum()
                ),
            }
            for condition_name_str, condition_mask_ser in condition_by_name_dict.items()
        ]
    )


def sample_moving_block_index_arr(
    observation_count_int: int,
    block_day_int: int,
    random_generator_obj: np.random.Generator,
) -> np.ndarray:
    sampled_index_list: list[int] = []
    while len(sampled_index_list) < observation_count_int:
        block_start_int = int(random_generator_obj.integers(0, observation_count_int))
        sampled_index_list.extend(
            (block_start_int + offset_int) % observation_count_int
            for offset_int in range(block_day_int)
        )
    return np.asarray(sampled_index_list[:observation_count_int], dtype=int)


def calculate_path_metrics(return_arr: np.ndarray) -> tuple[float, float, float]:
    equity_arr = np.cumprod(1.0 + return_arr)
    annualized_return_float = float(
        equity_arr[-1] ** (DAYS_IN_YEAR_INT / len(return_arr)) - 1.0
    )
    volatility_float = float(np.std(return_arr, ddof=1))
    sharpe_float = float(
        np.mean(return_arr) / volatility_float * np.sqrt(DAYS_IN_YEAR_INT)
    )
    drawdown_arr = equity_arr / np.maximum.accumulate(equity_arr) - 1.0
    return annualized_return_float, sharpe_float, float(np.min(drawdown_arr))


def simulate_observation_annual_split_return_arr(
    paired_return_arr: np.ndarray,
) -> np.ndarray:
    sleeve_value_arr = np.array([0.5, 0.5], dtype=float)
    blend_return_arr = np.zeros(len(paired_return_arr), dtype=float)
    for observation_int, strategy_return_arr in enumerate(paired_return_arr):
        if observation_int > 0 and observation_int % DAYS_IN_YEAR_INT == 0:
            sleeve_value_arr[:] = float(sleeve_value_arr.sum()) * 0.5
        previous_total_float = float(sleeve_value_arr.sum())
        sleeve_value_arr *= 1.0 + strategy_return_arr
        blend_return_arr[observation_int] = (
            float(sleeve_value_arr.sum()) / previous_total_float - 1.0
        )
    return blend_return_arr


def calculate_bootstrap_df(
    strategy_return_df: pd.DataFrame,
    bootstrap_iteration_int: int,
    block_day_int: int,
) -> pd.DataFrame:
    if bootstrap_iteration_int <= 0:
        raise ValueError("bootstrap_iteration_int must be positive.")
    if block_day_int <= 0:
        raise ValueError("block_day_int must be positive.")
    paired_return_arr = strategy_return_df[["ndx", "mosaic"]].to_numpy(dtype=float)
    random_generator_obj = np.random.default_rng(BOOTSTRAP_SEED_INT)
    row_list: list[dict[str, float | int]] = []
    for bootstrap_iteration_index_int in range(bootstrap_iteration_int):
        sampled_index_arr = sample_moving_block_index_arr(
            observation_count_int=len(paired_return_arr),
            block_day_int=block_day_int,
            random_generator_obj=random_generator_obj,
        )
        sampled_return_arr = paired_return_arr[sampled_index_arr]
        ndx_metric_tuple = calculate_path_metrics(sampled_return_arr[:, 0])
        blend_metric_tuple = calculate_path_metrics(
            simulate_observation_annual_split_return_arr(sampled_return_arr)
        )
        row_list.append(
            {
                "bootstrap_iteration_int": bootstrap_iteration_index_int,
                "block_day_int": block_day_int,
                "cagr_delta_float": blend_metric_tuple[0] - ndx_metric_tuple[0],
                "sharpe_delta_float": blend_metric_tuple[1] - ndx_metric_tuple[1],
                "max_drawdown_delta_float": blend_metric_tuple[2] - ndx_metric_tuple[2],
            }
        )
    return pd.DataFrame(row_list)


def summarize_bootstrap_df(bootstrap_df: pd.DataFrame) -> pd.DataFrame:
    row_list: list[dict[str, float | str]] = []
    for block_day_int, block_bootstrap_df in bootstrap_df.groupby("block_day_int"):
        for metric_name_str in (
            "cagr_delta_float",
            "sharpe_delta_float",
            "max_drawdown_delta_float",
        ):
            metric_ser = block_bootstrap_df[metric_name_str]
            row_list.append(
                {
                    "block_day_int": int(block_day_int),
                    "metric_name_str": metric_name_str,
                    "observed_bootstrap_mean_float": float(metric_ser.mean()),
                    "p025_float": float(metric_ser.quantile(0.025)),
                    "p500_float": float(metric_ser.quantile(0.5)),
                    "p975_float": float(metric_ser.quantile(0.975)),
                    "positive_resample_fraction_float": float((metric_ser > 0.0).mean()),
                }
            )
    return pd.DataFrame(row_list)


def save_charts(
    output_dir_path: Path,
    daily_analysis_df: pd.DataFrame,
    allocation_sweep_df: pd.DataFrame,
    relative_log_contribution_df: pd.DataFrame,
) -> None:
    chart_dir_path = output_dir_path / "charts"
    chart_dir_path.mkdir(parents=True, exist_ok=True)
    color_by_series_dict = {
        "NDX/VXN": "#305F72",
        "MOSAIC": "#B85C38",
        "50/50 annual": "#2F855A",
        "Risk-matched NDX": "#7667A6",
    }

    equity_df = (
        100.0
        * (1.0 + daily_analysis_df[
            ["ndx", "mosaic", "blend_50_50", "risk_matched_ndx"]
        ]).cumprod()
    )
    equity_df.columns = list(color_by_series_dict)
    fig_obj, axis_obj = plt.subplots(figsize=(11, 6))
    for series_name_str in equity_df.columns:
        axis_obj.plot(
            equity_df.index,
            equity_df[series_name_str],
            label=series_name_str,
            color=color_by_series_dict[series_name_str],
            linewidth=1.6,
        )
    axis_obj.set_yscale("log")
    axis_obj.set_title("Momentum sleeve equity, common start = 100 (log scale)")
    axis_obj.set_ylabel("Equity")
    axis_obj.grid(alpha=0.2)
    axis_obj.legend(frameon=False)
    fig_obj.tight_layout()
    fig_obj.savefig(chart_dir_path / "equity_comparison.png", dpi=170)
    plt.close(fig_obj)

    drawdown_df = equity_df.div(equity_df.cummax()).sub(1.0)
    fig_obj, axis_obj = plt.subplots(figsize=(11, 5))
    for series_name_str in ("NDX/VXN", "MOSAIC", "50/50 annual"):
        axis_obj.plot(
            drawdown_df.index,
            100.0 * drawdown_df[series_name_str],
            label=series_name_str,
            color=color_by_series_dict[series_name_str],
            linewidth=1.3,
        )
    axis_obj.set_title("Drawdown")
    axis_obj.set_ylabel("Percent")
    axis_obj.grid(alpha=0.2)
    axis_obj.legend(frameon=False)
    fig_obj.tight_layout()
    fig_obj.savefig(chart_dir_path / "drawdown_comparison.png", dpi=170)
    plt.close(fig_obj)

    # *** CRITICAL *** report-only causal window: correlation at date t uses
    # returns dated no later than t. It never feeds a strategy decision.
    rolling_correlation_ser = daily_analysis_df["ndx"].rolling(
        window=ROLLING_CORRELATION_WINDOW_INT,
        min_periods=ROLLING_CORRELATION_WINDOW_INT,
    ).corr(daily_analysis_df["mosaic"])
    fig_obj, axis_obj = plt.subplots(figsize=(11, 4.5))
    axis_obj.plot(
        rolling_correlation_ser.index,
        rolling_correlation_ser,
        color="#305F72",
        linewidth=1.2,
    )
    axis_obj.axhline(
        float(daily_analysis_df["ndx"].corr(daily_analysis_df["mosaic"])),
        color="#B85C38",
        linestyle="--",
        linewidth=1.2,
        label="Full-sample correlation",
    )
    axis_obj.set_ylim(-0.2, 1.02)
    axis_obj.set_title("NDX/VXN vs MOSAIC rolling 126-session correlation")
    axis_obj.grid(alpha=0.2)
    axis_obj.legend(frameon=False)
    fig_obj.tight_layout()
    fig_obj.savefig(chart_dir_path / "rolling_126d_correlation.png", dpi=170)
    plt.close(fig_obj)

    fig_obj, axis_arr = plt.subplots(1, 2, figsize=(11, 4.5))
    axis_arr[0].plot(
        100.0 * allocation_sweep_df["mosaic_weight_float"],
        100.0 * allocation_sweep_df["annualized_return_float"],
        marker="o",
        color="#305F72",
        label="CAGR",
    )
    axis_arr[0].plot(
        100.0 * allocation_sweep_df["mosaic_weight_float"],
        100.0 * allocation_sweep_df["annualized_volatility_float"],
        marker="o",
        color="#B85C38",
        label="Volatility",
    )
    axis_arr[0].set_xlabel("MOSAIC weight (%)")
    axis_arr[0].set_ylabel("Percent")
    axis_arr[0].set_title("Return and risk by annual-reset weight")
    axis_arr[0].grid(alpha=0.2)
    axis_arr[0].legend(frameon=False)

    contribution_plot_df = relative_log_contribution_df.iloc[:3].copy()
    axis_arr[1].bar(
        contribution_plot_df["condition_name_str"],
        contribution_plot_df["relative_log_wealth_contribution_float"],
        color=["#B85C38", "#2F855A", "#8A8A8A"],
    )
    axis_arr[1].axhline(0.0, color="#333333", linewidth=0.8)
    axis_arr[1].set_title("50/50 relative log-wealth contribution vs NDX")
    axis_arr[1].tick_params(axis="x", rotation=20)
    axis_arr[1].grid(axis="y", alpha=0.2)
    fig_obj.tight_layout()
    fig_obj.savefig(chart_dir_path / "allocation_and_relative_contribution.png", dpi=170)
    plt.close(fig_obj)


def main() -> None:
    argument_namespace = parse_arguments()
    source_portfolio_pickle_path = argument_namespace.source_portfolio_pickle_path.resolve()
    output_dir_path = argument_namespace.output_dir_path.resolve()
    output_dir_path.mkdir(parents=True, exist_ok=True)

    portfolio_obj = load_pickle(source_portfolio_pickle_path)
    ndx_strategy_obj = find_strategy_obj(portfolio_obj, NDX_STRATEGY_NAME_STR)
    mosaic_strategy_obj = find_strategy_obj(portfolio_obj, MOSAIC_STRATEGY_NAME_STR)

    strategy_return_df = portfolio_obj._daily_rets.rename(
        columns={
            NDX_STRATEGY_NAME_STR: "ndx",
            MOSAIC_STRATEGY_NAME_STR: "mosaic",
        }
    )[["ndx", "mosaic"]].astype(float)
    if strategy_return_df.isna().any().any():
        raise RuntimeError("Saved strategy return streams contain missing values.")

    benchmark_value_ser = portfolio_obj.regression_benchmark_value_ser.reindex(
        strategy_return_df.index
    ).astype(float)
    if benchmark_value_ser.isna().any():
        raise RuntimeError("Saved benchmark value series is incomplete on strategy dates.")
    benchmark_return_ser = benchmark_value_ser.pct_change(fill_method=None).fillna(0.0)

    blend_return_ser, _, blend_sleeve_weight_df = simulate_split_book(
        strategy_return_df=strategy_return_df,
        mosaic_weight_float=0.5,
        rebalance_frequency_str="annually",
    )
    saved_blend_return_ser = portfolio_obj.results["daily_returns"].astype(float)
    maximum_reconstruction_error_float = float(
        (blend_return_ser - saved_blend_return_ser).abs().max()
    )
    if maximum_reconstruction_error_float > 1e-12:
        raise RuntimeError(
            "50/50 annual reconstruction does not match the saved portfolio: "
            f"max_error={maximum_reconstruction_error_float:.3e}"
        )

    allocation_row_list: list[dict[str, object]] = []
    blend_return_by_weight_dict: dict[float, pd.Series] = {}
    for mosaic_weight_float in (0.0, 0.25, 0.5, 0.75, 1.0):
        allocation_return_ser, _, _ = simulate_split_book(
            strategy_return_df=strategy_return_df,
            mosaic_weight_float=mosaic_weight_float,
            rebalance_frequency_str="annually",
        )
        blend_return_by_weight_dict[mosaic_weight_float] = allocation_return_ser
        allocation_row_list.append(
            {
                "mosaic_weight_float": mosaic_weight_float,
                "ndx_weight_float": 1.0 - mosaic_weight_float,
                **calculate_metrics(allocation_return_ser, benchmark_return_ser),
            }
        )
    allocation_sweep_df = pd.DataFrame(allocation_row_list)

    rebalance_row_list: list[dict[str, object]] = []
    for rebalance_frequency_str in (None, "annually", "quarterly", "monthly"):
        rebalance_return_ser, _, _ = simulate_split_book(
            strategy_return_df=strategy_return_df,
            mosaic_weight_float=0.5,
            rebalance_frequency_str=rebalance_frequency_str,
        )
        rebalance_row_list.append(
            {
                "rebalance_frequency_str": "none" if rebalance_frequency_str is None else rebalance_frequency_str,
                **calculate_metrics(rebalance_return_ser, benchmark_return_ser),
            }
        )
    rebalance_comparison_df = pd.DataFrame(rebalance_row_list)

    ndx_metric_dict = calculate_metrics(strategy_return_df["ndx"], benchmark_return_ser)
    blend_metric_dict = calculate_metrics(blend_return_ser, benchmark_return_ser)
    risk_match_scale_float = (
        blend_metric_dict["annualized_volatility_float"]
        / ndx_metric_dict["annualized_volatility_float"]
    )
    risk_matched_ndx_return_ser = (
        strategy_return_df["ndx"] * risk_match_scale_float
    ).rename("risk_matched_ndx")

    # *** CRITICAL *** diagnostic as-of mapping: only the latest VXN close on
    # or before each return date is used. This classification never feeds orders.
    vxn_close_ser = ndx_strategy_obj.vxn_scale_signal_df["vxn_close"].reindex(
        strategy_return_df.index,
        method="ffill",
    )
    daily_analysis_df = strategy_return_df.assign(
        blend_50_50=blend_return_ser,
        risk_matched_ndx=risk_matched_ndx_return_ser,
        spx=benchmark_return_ser,
        vxn_close=vxn_close_ser,
        ndx_sleeve_weight=blend_sleeve_weight_df["ndx"],
        mosaic_sleeve_weight=blend_sleeve_weight_df["mosaic"],
    )

    regression_row_list = [
        calculate_hac_regression_row(
            "MOSAIC on NDX strategy",
            strategy_return_df["mosaic"],
            strategy_return_df[["ndx"]],
        ),
        calculate_hac_regression_row(
            "MOSAIC on NDX strategy and SPX",
            strategy_return_df["mosaic"],
            pd.concat(
                [strategy_return_df[["ndx"]], benchmark_return_ser.rename("spx")],
                axis=1,
            ),
        ),
        calculate_hac_regression_row(
            "50/50 annual on NDX strategy",
            blend_return_ser,
            strategy_return_df[["ndx"]],
        ),
        calculate_hac_regression_row(
            "50/50 annual on NDX strategy and SPX",
            blend_return_ser,
            pd.concat(
                [strategy_return_df[["ndx"]], benchmark_return_ser.rename("spx")],
                axis=1,
            ),
        ),
    ]
    regression_df = pd.DataFrame(regression_row_list)

    strategy_characteristic_row_list: list[dict[str, object]] = []
    for series_name_str, strategy_obj in (
        ("ndx", ndx_strategy_obj),
        ("mosaic", mosaic_strategy_obj),
    ):
        strategy_summary_ser = strategy_obj.summary["Strategy"]
        strategy_characteristic_row_list.append(
            {
                "series_name_str": series_name_str,
                **calculate_strategy_exposure_metrics(strategy_obj),
                "annual_turnover_float": float(strategy_summary_ser["Turnover (Ann.) [%]"]) / 100.0,
                "annual_cost_drag_float": float(strategy_summary_ser["Cost Drag (Ann.) [%]"]) / 100.0,
                "total_trading_cost_float": float(strategy_summary_ser["Total Trading Costs [$]"]),
                "trade_count_int": int(len(strategy_obj._trades)),
            }
        )
    strategy_characteristics_df = pd.DataFrame(strategy_characteristic_row_list)

    ndx_holding_bool_df = (
        ndx_strategy_obj.realized_weight_df.reindex(strategy_return_df.index)
        .drop(columns=["Cash"], errors="ignore")
        .fillna(0.0)
        .gt(0.0)
    )
    mosaic_holding_bool_df = (
        mosaic_strategy_obj.realized_weight_df.reindex(strategy_return_df.index)
        .drop(columns=["Cash"], errors="ignore")
        .fillna(0.0)
        .gt(0.0)
    )
    shared_symbol_index = ndx_holding_bool_df.columns.intersection(
        mosaic_holding_bool_df.columns
    )
    overlap_count_ser = (
        ndx_holding_bool_df[shared_symbol_index]
        & mosaic_holding_bool_df[shared_symbol_index]
    ).sum(axis=1)
    ndx_position_count_ser = ndx_holding_bool_df.sum(axis=1).replace(0, np.nan)

    selection_audit_df = mosaic_strategy_obj.get_selection_audit_df()
    selection_summary_dict = {
        "average_holding_overlap_count_float": float(overlap_count_ser.mean()),
        "average_ndx_holding_overlap_share_float": float(
            overlap_count_ser.div(ndx_position_count_ser).mean()
        ),
        "average_mosaic_selected_pairwise_correlation_float": float(
            selection_audit_df["avg_selected_pairwise_corr_float"].mean()
        ),
        "median_mosaic_candidate_count_float": float(
            selection_audit_df["candidate_count_int"].median()
        ),
        "average_mosaic_adv_excluded_count_float": float(
            selection_audit_df["adv_excluded_count_int"].mean()
        ),
    }

    subperiod_df = calculate_subperiod_metrics_df(
        strategy_return_df=strategy_return_df,
        blend_return_ser=blend_return_ser,
        benchmark_return_ser=benchmark_return_ser,
    )
    conditional_monthly_df = calculate_conditional_monthly_df(daily_analysis_df)
    relative_log_contribution_df = calculate_relative_log_contribution_df(daily_analysis_df)

    annual_return_df = (
        (1.0 + daily_analysis_df[["ndx", "mosaic", "blend_50_50", "spx"]])
        .groupby(daily_analysis_df.index.year)
        .prod()
        .sub(1.0)
    )
    annual_return_df.index.name = "calendar_year_int"

    # *** CRITICAL *** report-only causal window; values at t include no return
    # dated after t and do not feed either strategy.
    rolling_correlation_ser = daily_analysis_df["ndx"].rolling(
        window=ROLLING_CORRELATION_WINDOW_INT,
        min_periods=ROLLING_CORRELATION_WINDOW_INT,
    ).corr(daily_analysis_df["mosaic"])
    rolling_correlation_df = rolling_correlation_ser.rename(
        "rolling_126d_ndx_mosaic_correlation_float"
    ).to_frame()

    monthly_strategy_return_df = (
        (1.0 + strategy_return_df).resample("ME").prod().sub(1.0)
    )
    valid_rolling_correlation_ser = rolling_correlation_ser.dropna()
    dependence_summary_df = pd.DataFrame(
        [
            {
                "daily_ndx_mosaic_correlation_float": float(
                    strategy_return_df["ndx"].corr(strategy_return_df["mosaic"])
                ),
                "monthly_ndx_mosaic_correlation_float": float(
                    monthly_strategy_return_df["ndx"].corr(
                        monthly_strategy_return_df["mosaic"]
                    )
                ),
                "portfolio_tail_ndx_mosaic_correlation_float": float(
                    portfolio_obj.tail_correlation_matrix.loc[
                        NDX_STRATEGY_NAME_STR,
                        MOSAIC_STRATEGY_NAME_STR,
                    ]
                ),
                "rolling_126d_correlation_p10_float": float(
                    valid_rolling_correlation_ser.quantile(0.10)
                ),
                "rolling_126d_correlation_median_float": float(
                    valid_rolling_correlation_ser.median()
                ),
                "rolling_126d_correlation_p90_float": float(
                    valid_rolling_correlation_ser.quantile(0.90)
                ),
                "rolling_126d_correlation_min_float": float(
                    valid_rolling_correlation_ser.min()
                ),
                "rolling_126d_correlation_max_float": float(
                    valid_rolling_correlation_ser.max()
                ),
                "realized_diversification_ratio_float": float(
                    portfolio_obj.realized_diversification_ratio
                ),
                "average_rolling_diversification_ratio_float": float(
                    portfolio_obj.average_rolling_diversification_ratio
                ),
                "mosaic_win_year_count_int": int(
                    (annual_return_df["mosaic"] > annual_return_df["ndx"]).sum()
                ),
                "calendar_year_count_int": int(len(annual_return_df)),
            }
        ]
    )

    bootstrap_df = pd.concat(
        [
            calculate_bootstrap_df(
                strategy_return_df=strategy_return_df,
                bootstrap_iteration_int=argument_namespace.bootstrap_iterations,
                block_day_int=block_day_int,
            )
            for block_day_int in BOOTSTRAP_BLOCK_DAY_TUPLE
        ],
        ignore_index=True,
    )
    bootstrap_summary_df = summarize_bootstrap_df(bootstrap_df)

    metric_comparison_row_list = []
    for series_name_str, return_ser in (
        ("NDX/VXN 100%", strategy_return_df["ndx"]),
        ("MOSAIC 100%", strategy_return_df["mosaic"]),
        ("50/50 annual", blend_return_ser),
        ("Risk-matched NDX", risk_matched_ndx_return_ser),
    ):
        metric_comparison_row_list.append(
            {
                "series_name_str": series_name_str,
                **calculate_metrics(return_ser, benchmark_return_ser),
            }
        )
    metric_comparison_df = pd.DataFrame(metric_comparison_row_list)

    allocation_sweep_df.to_csv(output_dir_path / "allocation_sweep.csv", index=False)
    rebalance_comparison_df.to_csv(output_dir_path / "rebalance_comparison.csv", index=False)
    metric_comparison_df.to_csv(output_dir_path / "metric_comparison.csv", index=False)
    regression_df.to_csv(output_dir_path / "regression_results.csv", index=False)
    strategy_characteristics_df.to_csv(
        output_dir_path / "strategy_characteristics.csv",
        index=False,
    )
    subperiod_df.to_csv(output_dir_path / "subperiod_metrics.csv", index=False)
    conditional_monthly_df.to_csv(
        output_dir_path / "conditional_monthly_results.csv",
        index=False,
    )
    relative_log_contribution_df.to_csv(
        output_dir_path / "relative_log_contribution.csv",
        index=False,
    )
    annual_return_df.to_csv(output_dir_path / "annual_returns.csv")
    rolling_correlation_df.to_csv(output_dir_path / "rolling_126d_correlation.csv")
    dependence_summary_df.to_csv(
        output_dir_path / "dependence_summary.csv",
        index=False,
    )
    bootstrap_df.to_csv(output_dir_path / "bootstrap_draws.csv", index=False)
    bootstrap_summary_df.to_csv(
        output_dir_path / "bootstrap_summary.csv",
        index=False,
    )
    daily_analysis_df.to_csv(output_dir_path / "daily_series.csv.gz", compression="gzip")

    save_charts(
        output_dir_path=output_dir_path,
        daily_analysis_df=daily_analysis_df,
        allocation_sweep_df=allocation_sweep_df,
        relative_log_contribution_df=relative_log_contribution_df,
    )

    metadata_dict = {
        "study_name_str": "momentum_sleeve_split_study",
        "research_only_bool": True,
        "source_portfolio_pickle_path_str": str(source_portfolio_pickle_path),
        "source_portfolio_name_str": str(portfolio_obj.name),
        "sample_start_date_str": strategy_return_df.index.min().date().isoformat(),
        "sample_end_date_str": strategy_return_df.index.max().date().isoformat(),
        "observation_count_int": int(len(strategy_return_df)),
        "benchmark_symbol_str": "$SPX",
        "benchmark_adjustment_str": "TOTALRETURN",
        "risk_free_rate_float": 0.0,
        "cash_return_float": 0.0,
        "cost_policy_str": "Saved strategy returns include configured commissions and slippage; no cost assumptions changed.",
        "allocation_weight_list": [0.0, 0.25, 0.5, 0.75, 1.0],
        "rebalance_frequency_list": ["none", "annually", "quarterly", "monthly"],
        "primary_comparison_str": "100% NDX/VXN versus 50/50 annual reset",
        "allocation_sweep_reuses_saved_return_streams_bool": True,
        "allocation_scale_caveat_str": "Weights reuse returns realized at the saved pod capital; they are not fresh per-weight executions, so minimum commissions and integer-share effects are not rescaled.",
        "bootstrap_iteration_int": int(argument_namespace.bootstrap_iterations),
        "bootstrap_block_day_list": list(BOOTSTRAP_BLOCK_DAY_TUPLE),
        "bootstrap_seed_int": BOOTSTRAP_SEED_INT,
        "rolling_correlation_window_int": ROLLING_CORRELATION_WINDOW_INT,
        "maximum_saved_portfolio_reconstruction_error_float": maximum_reconstruction_error_float,
        "risk_match_scale_float": risk_match_scale_float,
        "outer_pod_rebalance_cost_policy_str": "No separate cost is charged for redistributing capital between pods.",
        "negative_cash_financing_policy_str": "Not modeled in the saved source artifacts.",
        "in_sample_inference_caveat_str": "Regressions, allocation weights, conditional tables, and bootstrap diagnostics all reuse the 2004-2026 realized sample and do not prove out-of-sample alpha.",
        "selection_summary": selection_summary_dict,
    }
    (output_dir_path / "metadata.json").write_text(
        json.dumps(metadata_dict, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print(f"Saved momentum sleeve split study to {output_dir_path}")


if __name__ == "__main__":
    main()
