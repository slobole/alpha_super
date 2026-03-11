"""
Defense First tactical allocation strategy.

Practical baseline implementation:
- Fallback asset = SPY
- Paper/article momentum formula = annualized daily lookback returns
- Monthly rebalance with next-open execution

Core formulas
-------------
For each defensive asset i at rebalance decision date t:

    annualized_return_{i,t}^{(L)}
        = (close_{i,t} / close_{i,t-L})^(252 / L) - 1

    momentum_score_{i,t}
        = mean(
            annualized_return_{i,t}^{(21)},
            annualized_return_{i,t}^{(63)},
            annualized_return_{i,t}^{(126)},
            annualized_return_{i,t}^{(252)}
          )

Rank the defensive assets by momentum_score descending and assign slot weights:

    rank_weight_vec = [0.40, 0.30, 0.20, 0.10]

Absolute momentum filter versus cash:

    if momentum_score_{i,t} > cash_return_t:
        keep the asset's original rank slot weight
    else:
        redirect that slot weight to SPY

This file keeps:
1. Signal data: TOTALRETURN closes for ranking.
2. Execution data: CAPITALSPECIAL OHLC for fills and valuation.

That preserves total-return-aware ranking while keeping fills realistic.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from IPython.display import display

from alpha.engine.backtest import run_daily
from alpha.engine.report import save_results
from strategies.strategy_taa_defense_first import (
    DefenseFirstConfig,
    DefenseFirstStrategy,
    load_cash_return_ser,
    load_execution_price_df,
    load_signal_close_df,
    map_month_end_weights_to_rebalance_open_df,
)


PRACTICAL_CONFIG = DefenseFirstConfig(fallback_asset="SPY")


def compute_paper_month_end_weight_df(
    signal_close_df: pd.DataFrame,
    cash_return_ser: pd.Series,
    config: DefenseFirstConfig,
    lookback_day_vec: Sequence[int] = (21, 63, 126, 252),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute month-end target weights using the paper/article daily momentum math.

    Returns:
    - momentum_score_df: month-end momentum scores for diagnostics.
    - month_end_weight_df: weights decided at month-end t, executed at the
      first trading day of month t+1.
    """
    if len(lookback_day_vec) == 0:
        raise ValueError("lookback_day_vec must not be empty.")

    max_lookback_day_int = int(max(lookback_day_vec))
    if len(signal_close_df) <= max_lookback_day_int:
        raise ValueError("signal_close_df does not contain enough rows for the requested lookbacks.")

    annualized_component_df_list: list[pd.DataFrame] = []
    for lookback_day_int in lookback_day_vec:
        # *** CRITICAL*** This shift uses only information available on or
        # before decision date t. No future bars are referenced.
        lagged_close_df = signal_close_df.shift(lookback_day_int)
        annualized_component_df = (signal_close_df / lagged_close_df) ** (252.0 / float(lookback_day_int)) - 1.0
        annualized_component_df_list.append(annualized_component_df)

    momentum_score_daily_df = sum(annualized_component_df_list) / float(len(annualized_component_df_list))

    # *** CRITICAL*** Month-end signals are sampled from already-computed
    # daily signals and are traded only at the first open of the next month.
    momentum_score_df = momentum_score_daily_df.resample("ME").last()
    monthly_cash_return_ser = cash_return_ser.resample("ME").last()
    combined_df = pd.concat([momentum_score_df, monthly_cash_return_ser], axis=1).dropna()

    tradeable_asset_list = list(config.tradeable_asset_list)
    month_end_weight_df = pd.DataFrame(0.0, index=combined_df.index, columns=tradeable_asset_list)

    for decision_date, row_ser in combined_df.iterrows():
        defensive_score_ser = row_ser[list(config.defensive_asset_list)].astype(float)
        ranked_asset_list = defensive_score_ser.sort_values(ascending=False).index.tolist()
        cash_return_float = float(row_ser["cash_return"])

        target_weight_ser = pd.Series(0.0, index=tradeable_asset_list, dtype=float)
        for rank_idx_int, asset_str in enumerate(ranked_asset_list):
            slot_weight_float = float(config.rank_weight_vec[rank_idx_int])
            asset_score_float = float(defensive_score_ser.loc[asset_str])

            if asset_score_float > cash_return_float:
                target_weight_ser.loc[asset_str] = slot_weight_float
            else:
                target_weight_ser.loc[config.fallback_asset] += slot_weight_float

        if not np.isclose(target_weight_ser.sum(), 1.0, atol=1e-12):
            raise ValueError(
                f"Target weights must sum to 1.0. Found {target_weight_ser.sum():.12f} on {decision_date}."
            )

        month_end_weight_df.loc[decision_date] = target_weight_ser

    return momentum_score_df, month_end_weight_df


def get_defense_first_practical_data(
    config: DefenseFirstConfig = PRACTICAL_CONFIG,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load signal data, execution data, momentum scores, and rebalance weights
    for the recommended practical baseline.
    """
    signal_close_df = load_signal_close_df(
        symbol_list=config.defensive_asset_list,
        start_date_str=config.start_date_str,
        end_date_str=config.end_date_str,
    )
    execution_price_df = load_execution_price_df(
        tradeable_asset_list=config.tradeable_asset_list,
        benchmark_list=config.benchmark_list,
        start_date_str=config.start_date_str,
        end_date_str=config.end_date_str,
    )
    cash_return_ser = load_cash_return_ser(config.dtb3_csv_path_str)
    momentum_score_df, month_end_weight_df = compute_paper_month_end_weight_df(
        signal_close_df=signal_close_df,
        cash_return_ser=cash_return_ser,
        config=config,
    )
    rebalance_weight_df = map_month_end_weights_to_rebalance_open_df(
        month_end_weight_df=month_end_weight_df,
        execution_index=execution_price_df.index,
    )

    return execution_price_df, momentum_score_df, month_end_weight_df, rebalance_weight_df


if __name__ == "__main__":
    config = PRACTICAL_CONFIG

    execution_price_df, momentum_score_df, month_end_weight_df, rebalance_weight_df = (
        get_defense_first_practical_data(config)
    )

    strategy = DefenseFirstStrategy(
        name="DefenseFirstPracticalStrategy",
        benchmarks=config.benchmark_list,
        rebalance_weight_df=rebalance_weight_df,
        tradeable_asset_list=config.tradeable_asset_list,
        capital_base=100_000,
        slippage=0.0001,
        commission_per_share=0.005,
        commission_minimum=1.0,
    )
    strategy.show_taa_weights_report = True
    strategy.daily_target_weights = rebalance_weight_df.reindex(execution_price_df.index).ffill().dropna()

    calendar_idx = execution_price_df.index[execution_price_df.index >= rebalance_weight_df.index[0]]
    run_daily(strategy, execution_price_df, calendar_idx)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)

    print(f"Fallback asset: {config.fallback_asset}")
    print("Signal formula: annualized daily 21/63/126/252-day momentum")

    print("First momentum scores:")
    display(momentum_score_df.dropna().head())

    print("First month-end decisions:")
    display(month_end_weight_df.head())

    print("First rebalance opens:")
    display(rebalance_weight_df.head())

    display(strategy.summary)
    display(strategy.summary_trades)

    save_results(strategy)
