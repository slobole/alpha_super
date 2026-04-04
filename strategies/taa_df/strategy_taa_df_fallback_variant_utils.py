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
) -> DefenseFirstStrategy:
    strategy = DefenseFirstStrategy(
        name=strategy_name_str,
        benchmarks=config.benchmark_list,
        rebalance_weight_df=rebalance_weight_df,
        tradeable_asset_list=config.tradeable_asset_list,
        capital_base=100_000,
        slippage=0.0001,
        commission_per_share=0.005,
        commission_minimum=1.0,
    )
    return strategy


def _run_strategy_from_weight_df(
    strategy: DefenseFirstStrategy,
    execution_price_df: pd.DataFrame,
    rebalance_weight_df: pd.DataFrame,
) -> None:
    strategy.show_taa_weights_report = True

    # *** CRITICAL*** This forward fill is for post-run weight diagnostics only.
    # Execution still uses the discrete month-to-open rebalance dates stored in
    # `rebalance_weight_df` inside `iterate()`.
    strategy.daily_target_weights = rebalance_weight_df.reindex(execution_price_df.index).ffill().dropna()

    calendar_idx = execution_price_df.index[execution_price_df.index >= rebalance_weight_df.index[0]]
    run_daily(
        strategy,
        execution_price_df,
        calendar_idx,
        show_progress=False,
        show_signal_progress_bool=False,
    )


def run_standard_fallback_variant(
    strategy_name_str: str,
    config: DefenseFirstConfig,
    data_loader_fn: Callable[[DefenseFirstConfig], tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]],
    show_display_bool: bool = True,
    save_results_bool: bool = True,
    output_dir_str: str = "results",
) -> DefenseFirstStrategy:
    """
    Run a standard return-momentum Defense First fallback variant.
    """
    execution_price_df, momentum_score_df, month_end_weight_df, rebalance_weight_df = data_loader_fn(config)

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
