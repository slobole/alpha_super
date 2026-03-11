"""
Defense First tactical allocation strategy with BTAL added to the defensive sleeve.

This variant keeps the existing monthly signal construction from
`strategy_taa_defense_first.py` and changes only the investable universe and
rank weights.

Core formulas remain unchanged:

    momentum_score_{i,t}
        = (r_{1m,i,t} + r_{3m,i,t} + r_{6m,i,t} + r_{12m,i,t}) / 4

    r_{k,i,t} = close_{i,t} / close_{i,t-k} - 1

With BTAL added, the defensive sleeve becomes:

    defensive_asset_list = [GLD, UUP, TLT, DBC, BTAL]

Rank weights expand from 4 slots to 5 slots:

    rank_weight_vec = [5/15, 4/15, 3/15, 2/15, 1/15]

Absolute momentum filter versus cash is unchanged:

    if momentum_score_{i,t} > cash_return_t:
        keep the asset's original rank slot weight
    else:
        redirect that slot weight to the fallback asset

This script uses SPY as the fallback asset.
"""

from __future__ import annotations

import pandas as pd
from IPython.display import display

from alpha.engine.backtest import run_daily
from alpha.engine.report import save_results
from strategies.strategy_taa_defense_first import (
    DefenseFirstConfig,
    DefenseFirstStrategy,
    get_defense_first_data,
)


BTAL_SPY_CONFIG = DefenseFirstConfig(
    defensive_asset_list=("GLD", "UUP", "TLT", "DBC", "BTAL"),
    fallback_asset="SPY",
    rank_weight_vec=(5.0 / 15.0, 4.0 / 15.0, 3.0 / 15.0, 2.0 / 15.0, 1.0 / 15.0),
)


if __name__ == "__main__":
    config = BTAL_SPY_CONFIG

    execution_price_df, momentum_score_df, month_end_weight_df, rebalance_weight_df = get_defense_first_data(config)

    strategy = DefenseFirstStrategy(
        name="DefenseFirstBTALStrategy",
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
    print(f"Defensive asset list: {config.defensive_asset_list}")
    print(f"Rank weight vector: {config.rank_weight_vec}")

    print("First momentum scores:")
    display(momentum_score_df.dropna().head())

    print("First month-end decisions:")
    display(month_end_weight_df.head())

    print("First rebalance opens:")
    display(rebalance_weight_df.head())

    display(strategy.summary)
    display(strategy.summary_trades)

    save_results(strategy)
