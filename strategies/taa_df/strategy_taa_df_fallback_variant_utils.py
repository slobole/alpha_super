"""
Utilities for the requested Defense First fallback-asset variants.

The quantitative rule for every generated fallback variant is:

    fallback_asset^{variant} = requested_fallback_asset

while preserving the base strategy semantics:

    signal_t = signal_t^{base}
    execution_t = next_open_t

The only extra guard is a start-date clip so the fallback ETF exists in the
execution dataset:

    start_date^{variant}
        = max(start_date^{base}, first_fallback_date)

This keeps the information set and execution contract unchanged while avoiding
pre-inception fills.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Callable
import sys

import pandas as pd
from IPython.display import display

repo_root_path = Path(__file__).resolve().parents[2]
repo_root_str = str(repo_root_path)
if repo_root_str not in sys.path:
    sys.path.insert(0, repo_root_str)

from alpha.engine.backtest import run_daily
from alpha.engine.friction_analysis import FrictionAnalysis
from alpha.engine.report import save_results

try:
    from strategies.taa_df.strategy_taa_df import DefenseFirstConfig, DefenseFirstStrategy
except ModuleNotFoundError:
    from strategy_taa_df import DefenseFirstConfig, DefenseFirstStrategy


FALLBACK_INCEPTION_DATE_MAP: dict[str, str] = {
    "SPY": "1993-01-29",
    "SSO": "2006-06-21",
    "UPRO": "2009-06-25",
    "QQQ": "1999-03-10",
    "QLD": "2006-06-21",
    "TQQQ": "2010-02-11",
}

REQUESTED_FALLBACK_ASSET_TUPLE: tuple[str, ...] = tuple(FALLBACK_INCEPTION_DATE_MAP.keys())


def build_fallback_variant_config(
    base_config: DefenseFirstConfig,
    fallback_asset_str: str,
) -> DefenseFirstConfig:
    """
    Build a fallback-asset variant config from a base Defense First config.

    Formula:

        start_date^{variant}
            = max(start_date^{base}, first_fallback_date)
    """
    if fallback_asset_str not in FALLBACK_INCEPTION_DATE_MAP:
        raise ValueError(
            f"Unsupported fallback asset {fallback_asset_str}. "
            f"Expected one of {REQUESTED_FALLBACK_ASSET_TUPLE}."
        )

    base_start_timestamp = pd.Timestamp(base_config.start_date_str)
    fallback_inception_timestamp = pd.Timestamp(FALLBACK_INCEPTION_DATE_MAP[fallback_asset_str])
    effective_start_timestamp = max(base_start_timestamp, fallback_inception_timestamp)

    return replace(
        base_config,
        fallback_asset=fallback_asset_str,
        start_date_str=effective_start_timestamp.strftime("%Y-%m-%d"),
    )


def _build_defense_first_strategy(
    strategy_name_str: str,
    config: DefenseFirstConfig,
    rebalance_weight_df: pd.DataFrame,
    capital_base_float: float = 100_000.0,
) -> DefenseFirstStrategy:
    strategy = DefenseFirstStrategy(
        name=strategy_name_str,
        benchmarks=config.benchmark_list,
        rebalance_weight_df=rebalance_weight_df,
        tradeable_asset_list=config.tradeable_asset_list,
        capital_base=capital_base_float,
        slippage=0.00025,
        commission_per_share=0.005,
        commission_minimum=1.0,
    )
    return strategy


def _run_strategy_from_weight_df(
    strategy: DefenseFirstStrategy,
    execution_price_df: pd.DataFrame,
    rebalance_weight_df: pd.DataFrame,
    backtest_start_date_str: str | None = None,
    show_progress_bool: bool = False,
) -> None:
    strategy.show_taa_weights_report = True

    # *** CRITICAL*** This forward fill is for post-run weight diagnostics only.
    # Execution still uses the discrete month-to-open rebalance dates stored in
    # `rebalance_weight_df` inside `iterate()`.
    strategy.daily_target_weights = rebalance_weight_df.reindex(execution_price_df.index).ffill().dropna()

    calendar_start_ts = pd.Timestamp(rebalance_weight_df.index[0])
    if backtest_start_date_str is not None:
        calendar_start_ts = max(calendar_start_ts, pd.Timestamp(backtest_start_date_str))
    # *** CRITICAL*** Deployment-reference runs keep full pre-start data for
    # month-end signal formation, but the executable calendar starts at the
    # first deployment fill session.
    calendar_idx = execution_price_df.index[execution_price_df.index >= calendar_start_ts]
    run_daily(
        strategy,
        execution_price_df,
        calendar_idx,
        show_progress=show_progress_bool,
        show_signal_progress_bool=show_progress_bool,
    )


def _map_month_end_weight_to_decision_close_df(
    month_end_weight_df: pd.DataFrame,
    execution_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    decision_weight_map_dict: dict[pd.Timestamp, pd.Series] = {}

    for month_end_ts, target_weight_ser in month_end_weight_df.iterrows():
        month_period_obj = pd.Timestamp(month_end_ts).to_period("M")
        trading_day_idx = execution_index[execution_index.to_period("M") == month_period_obj]
        if len(trading_day_idx) == 0:
            continue

        # *** CRITICAL*** Calendar month-end can fall on a non-trading day.
        # T Close means the last tradable close in that signal month, not the
        # calendar timestamp stored by resample("ME").
        decision_close_ts = pd.Timestamp(trading_day_idx[-1])
        decision_weight_map_dict[decision_close_ts] = target_weight_ser.copy()

    if len(decision_weight_map_dict) == 0:
        raise RuntimeError("No tradable decision-close dates were generated for execution timing analysis.")

    decision_close_weight_df = pd.DataFrame.from_dict(
        decision_weight_map_dict,
        orient="index",
    ).sort_index()
    decision_close_weight_df.index.name = "decision_close_date"
    return decision_close_weight_df


def run_standard_fallback_variant(
    strategy_name_str: str,
    config: DefenseFirstConfig,
    data_loader_fn: Callable[[DefenseFirstConfig], tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]],
    show_display_bool: bool = True,
    save_results_bool: bool = True,
    output_dir_str: str = "results",
    backtest_start_date_str: str | None = None,
    capital_base_float: float = 100_000.0,
) -> DefenseFirstStrategy:
    """
    Run a standard return-momentum Defense First fallback variant.
    """
    execution_price_df, momentum_score_df, month_end_weight_df, rebalance_weight_df = data_loader_fn(config)

    strategy = _build_defense_first_strategy(
        strategy_name_str=strategy_name_str,
        config=config,
        rebalance_weight_df=rebalance_weight_df,
        capital_base_float=capital_base_float,
    )
    _run_strategy_from_weight_df(
        strategy=strategy,
        execution_price_df=execution_price_df,
        rebalance_weight_df=rebalance_weight_df,
        backtest_start_date_str=backtest_start_date_str,
    )

    if show_display_bool:
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 1000)

        print("First momentum scores:")
        display(momentum_score_df.dropna().head())

        print("First month-end decisions:")
        display(month_end_weight_df.head())

        print("First rebalance opens:")
        display(rebalance_weight_df.head())

        display(strategy.summary)
        display(strategy.summary_trades)

    if save_results_bool:
        save_results(strategy, output_dir=output_dir_str)

    return strategy


def build_standard_fallback_friction_analysis_inputs(
    strategy_name_str: str,
    config: DefenseFirstConfig,
    data_loader_fn: Callable[[DefenseFirstConfig], tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]],
    show_display_bool: bool = False,
    backtest_start_date_str: str | None = None,
    capital_base_float: float = 100_000.0,
) -> dict[str, object]:
    """
    Build FrictionAnalysis inputs for a standard fallback variant.
    """
    execution_price_df, _momentum_score_df, _month_end_weight_df, rebalance_weight_df = data_loader_fn(config)
    strategy_obj = _build_defense_first_strategy(
        strategy_name_str=strategy_name_str,
        config=config,
        rebalance_weight_df=rebalance_weight_df,
        capital_base_float=capital_base_float,
    )

    # *** CRITICAL *** FrictionAnalysis must reuse the deployment-reference TAA
    # next-open completed ledger. The month-end signal maps to the next
    # tradable open before auction capacity is assessed.
    _run_strategy_from_weight_df(
        strategy=strategy_obj,
        execution_price_df=execution_price_df,
        rebalance_weight_df=rebalance_weight_df,
        backtest_start_date_str=backtest_start_date_str,
    )

    if show_display_bool:
        display(strategy_obj.summary)

    return {
        "strategy_obj": strategy_obj,
        "pricing_data_df": execution_price_df,
        "execution_policy_str": "MOO",
    }


def run_standard_fallback_friction_analysis(
    strategy_name_str: str,
    config: DefenseFirstConfig,
    data_loader_fn: Callable[[DefenseFirstConfig], tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]],
    save_results_bool: bool = True,
    output_dir_str: str = "results",
    show_display_bool: bool = False,
    backtest_start_date_str: str | None = None,
    capital_base_float: float = 100_000.0,
):
    friction_input_dict = build_standard_fallback_friction_analysis_inputs(
        strategy_name_str=strategy_name_str,
        config=config,
        data_loader_fn=data_loader_fn,
        show_display_bool=show_display_bool,
        backtest_start_date_str=backtest_start_date_str,
        capital_base_float=capital_base_float,
    )
    friction_analysis_obj = FrictionAnalysis(
        strategy_obj=friction_input_dict["strategy_obj"],
        pricing_data_df=friction_input_dict["pricing_data_df"],
        execution_policy_str=friction_input_dict["execution_policy_str"],
        output_dir_str=output_dir_str,
        save_output_bool=save_results_bool,
    )
    return friction_analysis_obj.run()


def build_standard_fallback_execution_timing_analysis_inputs(
    strategy_name_str: str,
    config: DefenseFirstConfig,
    data_loader_fn: Callable[[DefenseFirstConfig], tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]],
) -> dict[str, object]:
    """
    Build ExecutionTimingAnalyzer inputs for a standard fallback variant.

    Formula:

        decision_t = month_end_close_t

        entry_fill = decision_t + entry_lag at entry_price_field
        exit_fill  = decision_t + exit_lag  at exit_price_field
    """
    execution_price_df, _momentum_score_df, month_end_weight_df, rebalance_weight_df = data_loader_fn(config)
    decision_close_weight_df = _map_month_end_weight_to_decision_close_df(
        month_end_weight_df=month_end_weight_df,
        execution_index=pd.DatetimeIndex(execution_price_df.index),
    )

    def strategy_factory_fn():
        strategy_obj = _build_defense_first_strategy(
            strategy_name_str=strategy_name_str,
            config=config,
            rebalance_weight_df=decision_close_weight_df,
        )
        strategy_obj.show_taa_weights_report = True

        # *** CRITICAL*** This forward fill is for post-run weight diagnostics
        # only. ExecutionTimingAnalyzer uses month-end decision dates for order
        # intent and then applies the tested fill timing to those orders.
        strategy_obj.daily_target_weights = (
            rebalance_weight_df.reindex(execution_price_df.index).ffill().dropna()
        )
        return strategy_obj

    calendar_idx = execution_price_df.index[
        execution_price_df.index >= rebalance_weight_df.index[0]
    ]

    return {
        "strategy_factory_fn": strategy_factory_fn,
        "pricing_data_df": execution_price_df,
        "calendar_idx": pd.DatetimeIndex(calendar_idx),
        "order_generation_mode_str": "signal_bar",
        "risk_model_str": "taa_rebalance",
        "entry_timing_str_tuple": ("same_close_moc", "next_open", "next_close"),
        "exit_timing_str_tuple": ("same_close_moc", "next_open", "next_close"),
        "default_entry_timing_str": "next_open",
        "default_exit_timing_str": "next_open",
    }


def run_linearity_1n_fallback_variant(
    strategy_name_str: str,
    config: DefenseFirstConfig,
    data_loader_fn: Callable[[DefenseFirstConfig], tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]],
    show_display_bool: bool = True,
    save_results_bool: bool = True,
    output_dir_str: str = "results",
) -> DefenseFirstStrategy:
    """
    Run a linearity-1/n Defense First fallback variant.
    """
    (
        execution_price_df,
        daily_linearity_score_df,
        month_end_score_df,
        month_end_weight_df,
        rebalance_weight_df,
    ) = data_loader_fn(config)

    strategy = _build_defense_first_strategy(
        strategy_name_str=strategy_name_str,
        config=config,
        rebalance_weight_df=rebalance_weight_df,
    )
    _run_strategy_from_weight_df(
        strategy=strategy,
        execution_price_df=execution_price_df,
        rebalance_weight_df=rebalance_weight_df,
    )

    if show_display_bool:
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 1000)

        print("First daily linearity scores:")
        display(daily_linearity_score_df.dropna().head())

        print("First month-end linearity scores:")
        display(month_end_score_df.head())

        print("First month-end decisions:")
        display(month_end_weight_df.head())

        print("First rebalance opens:")
        display(rebalance_weight_df.head())

        display(strategy.summary)
        display(strategy.summary_trades)

    if save_results_bool:
        save_results(strategy, output_dir=output_dir_str)

    return strategy


def build_metric_row_from_strategy(
    strategy: DefenseFirstStrategy,
    correlation_benchmark_str: str | None = None,
) -> dict[str, float | str]:
    """
    Extract the requested metric row from a completed strategy.

    Let:

        r_t = V_t / V_{t-1} - 1

    Then the correlation field is defined as:

        corr = Corr(r_strategy,t, r_benchmark,t)

    rather than the built-in strategy self-correlation, which is always 1.
    """
    summary_df = strategy.summary
    benchmark_str = correlation_benchmark_str or strategy._benchmarks[0]

    return {
        "strategy_name": strategy.name,
        "sharpe": float(summary_df.loc["Sharpe Ratio", "Strategy"]),
        "ann_ret": float(summary_df.loc["Return (Ann.) [%]", "Strategy"]),
        "volatility": float(summary_df.loc["Volatility (Ann.) [%]", "Strategy"]),
        "max_dd": float(summary_df.loc["Max. Drawdown [%]", "Strategy"]),
        "avg_dd": float(summary_df.loc["Avg. Drawdown [%]", "Strategy"]),
        "corr": float(summary_df.loc["Correlation", benchmark_str]),
    }


def save_variant_metric_table(
    variant_metric_df: pd.DataFrame,
    output_dir_path: Path,
) -> Path:
    """
    Save the fallback-variant metric table as CSV and plain text.
    """
    output_dir_path.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir_path / "taa_df_fallback_variant_metrics.csv"
    txt_path = output_dir_path / "taa_df_fallback_variant_metrics.txt"

    variant_metric_df.to_csv(csv_path, index=False)
    txt_path.write_text(variant_metric_df.to_string(index=False), encoding="utf-8")
    return csv_path
