"""
Defense First tactical allocation with STIP replacing DBC in the defensive
sleeve.

This variant preserves the exact signal and execution logic from
`strategy_taa_df.py` and changes only the configuration:

    defensive_asset_list = ("GLD", "UUP", "TLT", "STIP")

The momentum formula is unchanged:

    momentum_score_{i,t}
        = (r_{1m,i,t} + r_{3m,i,t} + r_{6m,i,t} + r_{12m,i,t}) / 4

    r_{k,i,t} = close_{i,t} / close_{i,t-k} - 1

Quantitative consequence:

    first_usable_decision_t >= first_STIP_month_end + 12 months

because the 12-month lookback must exist for all defensive assets before the
monthly rank can be formed.
"""

from __future__ import annotations

import pandas as pd
from IPython.display import display

from alpha.engine.backtest import run_daily
from alpha.engine.report import save_results

try:
    from strategies.taa_df.strategy_taa_df import (
        DefenseFirstConfig,
        DefenseFirstStrategy,
        get_defense_first_data,
    )
except ModuleNotFoundError:
    from strategy_taa_df import (
        DefenseFirstConfig,
        DefenseFirstStrategy,
        get_defense_first_data,
    )


stip_defensive_asset_list = ("GLD", "UUP", "TLT", "STIP")

DEFAULT_CONFIG = DefenseFirstConfig(
    defensive_asset_list=stip_defensive_asset_list,
)


if __name__ == "__main__":
    taa_config = DEFAULT_CONFIG

    execution_price_df, momentum_score_df, month_end_weight_df, rebalance_weight_df = (
        get_defense_first_data(taa_config)
    )

    strategy = DefenseFirstStrategy(
        name="strategy_taa_df_stip",
        benchmarks=taa_config.benchmark_list,
        rebalance_weight_df=rebalance_weight_df,
        tradeable_asset_list=taa_config.tradeable_asset_list,
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

    print("First momentum scores:")
    display(momentum_score_df.dropna().head())

    print("First month-end decisions:")
    display(month_end_weight_df.head())

    print("First rebalance opens:")
    display(rebalance_weight_df.head())

    display(strategy.summary)
    display(strategy.summary_trades)

    save_results(strategy)
