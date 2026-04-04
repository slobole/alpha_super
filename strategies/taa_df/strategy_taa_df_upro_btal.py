"""
Defense First tactical allocation with BTAL added to the defensive sleeve and
UPRO as the fallback sleeve.

This variant preserves the exact signal and execution logic from
`strategy_taa_df.py` and changes only the configuration:

    defensive_asset_list = ("GLD", "UUP", "TLT", "DBC", "BTAL")
    fallback_asset = "UPRO"

The defensive rank weights switch from fixed four-slot weights to five-slot
rank weights:

    rank_score_vec = [5, 4, 3, 2, 1]
    rank_weight_vec = rank_score_vec / sum(rank_score_vec)
                    = [5, 4, 3, 2, 1] / 15

Because BTAL starts trading on 2011-09-13, the evaluation window is shorter
than the earlier UPRO-only variant and the execution dataset must satisfy:

    start_date >= 2011-09-13

Otherwise the backtest would request signal formation or fills on pre-inception
bars with missing BTAL data.
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


btal_inception_date_str = "2011-09-13"
btal_defensive_asset_list = ("GLD", "UUP", "TLT", "DBC", "BTAL")
btal_rank_weight_vec = (
    5.0 / 15.0,
    4.0 / 15.0,
    3.0 / 15.0,
    2.0 / 15.0,
    1.0 / 15.0,
)

DEFAULT_CONFIG = DefenseFirstConfig(
    defensive_asset_list=btal_defensive_asset_list,
    fallback_asset="UPRO",
    rank_weight_vec=btal_rank_weight_vec,
    start_date_str=btal_inception_date_str,
)


if __name__ == "__main__":
    taa_config = DEFAULT_CONFIG

    execution_price_df, momentum_score_df, month_end_weight_df, rebalance_weight_df = (
        get_defense_first_data(taa_config)
    )

    strategy = DefenseFirstStrategy(
        name="strategy_taa_df_upro_btal",
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
