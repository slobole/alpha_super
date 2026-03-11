"""
Defense First tactical allocation strategy with BTAL added to the defensive sleeve
and UPRO as the fallback asset.

Practical baseline variant:
- Fallback asset = UPRO
- Paper/article momentum formula = annualized daily 21/63/126/252-day returns
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

With BTAL added, the defensive sleeve becomes:

    defensive_asset_list = [GLD, UUP, TLT, DBC, BTAL]

Rank weights expand from 4 slots to 5 slots:

    rank_weight_vec = [5/15, 4/15, 3/15, 2/15, 1/15]

Absolute momentum filter versus cash is unchanged:

    if momentum_score_{i,t} > cash_return_t:
        keep the asset's original rank slot weight
    else:
        redirect that slot weight to UPRO

Because UPRO starts later than SPY, this variant filters out rebalance dates
where any positively-weighted asset lacks a valid opening price.
"""

from __future__ import annotations

import pandas as pd
from IPython.display import display

from alpha.engine.backtest import run_daily
from alpha.engine.report import save_results
from strategies.strategy_taa_defense_first import DefenseFirstConfig, DefenseFirstStrategy
from strategies.strategy_taa_defense_first_practical import get_defense_first_practical_data
from strategies.strategy_taa_defense_first_upro import filter_valid_rebalance_weight_df


BTAL_PRACTICAL_UPRO_CONFIG = DefenseFirstConfig(
    defensive_asset_list=("GLD", "UUP", "TLT", "DBC", "BTAL"),
    fallback_asset="UPRO",
    rank_weight_vec=(5.0 / 15.0, 4.0 / 15.0, 3.0 / 15.0, 2.0 / 15.0, 1.0 / 15.0),
)


if __name__ == "__main__":
    config = BTAL_PRACTICAL_UPRO_CONFIG

    execution_price_df, momentum_score_df, month_end_weight_df, rebalance_weight_df = get_defense_first_practical_data(config)

    if config.fallback_asset not in month_end_weight_df.columns:
        raise RuntimeError(
            f"Expected fallback asset {config.fallback_asset} in month_end_weight_df columns, found {list(month_end_weight_df.columns)}"
        )

    rebalance_weight_df = filter_valid_rebalance_weight_df(rebalance_weight_df, execution_price_df)

    strategy = DefenseFirstStrategy(
        name="DefenseFirstPracticalBTALUPROStrategy",
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
    print("Signal formula: annualized daily 21/63/126/252-day momentum")

    print("First momentum scores:")
    display(momentum_score_df.dropna().head())

    print("First month-end decisions:")
    display(month_end_weight_df.head())

    print("First valid rebalance opens:")
    display(rebalance_weight_df.head())

    display(strategy.summary)
    display(strategy.summary_trades)

    save_results(strategy)
