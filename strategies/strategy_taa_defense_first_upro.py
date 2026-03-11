"""
Defense First tactical allocation strategy with UPRO as the fallback asset.

This reuses the paper-faithful base implementation from
`strategy_taa_defense_first.py` and changes only the fallback sleeve:

    fallback_asset = 'UPRO'

Core formulas remain unchanged:

    momentum_score_{i,t}
        = (r_{1m,i,t} + r_{3m,i,t} + r_{6m,i,t} + r_{12m,i,t}) / 4

    r_{k,i,t} = close_{i,t} / close_{i,t-k} - 1

Rank first, then mask against cash, then redirect rejected slot weights to UPRO.

Because UPRO starts later than SPY, this variant filters out rebalance dates
where any positively-weighted asset lacks a valid opening price.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from IPython.display import display

from alpha.engine.backtest import run_daily
from alpha.engine.report import save_results
from strategies.strategy_taa_defense_first import (
    DefenseFirstConfig,
    DefenseFirstStrategy,
    get_defense_first_data,
)


UPRO_CONFIG = DefenseFirstConfig(fallback_asset="UPRO")


def filter_valid_rebalance_weight_df(
    rebalance_weight_df: pd.DataFrame,
    execution_price_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Keep only rebalance dates where every asset with positive target weight has a
    valid opening price.

    For rebalance date t and asset i, a trade is allowed only if:

        w_{i,t} > 0  =>  open_{i,t} is finite and open_{i,t} > 0
    """
    valid_rebalance_date_list: list[pd.Timestamp] = []

    for rebalance_date, target_weight_ser in rebalance_weight_df.iterrows():
        is_valid_bool = True

        for asset_str, target_weight_float in target_weight_ser.items():
            if float(target_weight_float) <= 0.0:
                continue

            open_col = (asset_str, "Open")
            if open_col not in execution_price_df.columns or rebalance_date not in execution_price_df.index:
                is_valid_bool = False
                break

            open_price_float = float(execution_price_df.loc[rebalance_date, open_col])
            if not np.isfinite(open_price_float) or open_price_float <= 0.0:
                is_valid_bool = False
                break

        if is_valid_bool:
            valid_rebalance_date_list.append(rebalance_date)

    filtered_rebalance_weight_df = rebalance_weight_df.loc[valid_rebalance_date_list].copy()
    if filtered_rebalance_weight_df.empty:
        raise RuntimeError("No valid rebalance dates remain after filtering for available opening prices.")

    return filtered_rebalance_weight_df


if __name__ == "__main__":
    config = UPRO_CONFIG

    execution_price_df, momentum_score_df, month_end_weight_df, rebalance_weight_df = get_defense_first_data(config)

    if config.fallback_asset not in month_end_weight_df.columns:
        raise RuntimeError(
            f"Expected fallback asset {config.fallback_asset} in month_end_weight_df columns, found {list(month_end_weight_df.columns)}"
        )

    rebalance_weight_df = filter_valid_rebalance_weight_df(rebalance_weight_df, execution_price_df)

    strategy = DefenseFirstStrategy(
        name="DefenseFirstUPROStrategy",
        benchmarks=config.benchmark_list,
        rebalance_weight_df=rebalance_weight_df,
        tradeable_asset_list=config.tradeable_asset_list,
        capital_base=100_000,
        slippage=0.0001,
        commission_per_share=0.005,
        commission_minimum=1.0,
    )
    strategy.show_taa_weights_report = True
    # strategy.daily_target_weights = rebalance_weight_df.copy()
    strategy.daily_target_weights = rebalance_weight_df.reindex(execution_price_df.index).ffill().dropna()


    calendar_idx = execution_price_df.index[execution_price_df.index >= rebalance_weight_df.index[0]]
    run_daily(strategy, execution_price_df, calendar_idx)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)

    print(f"Fallback asset: {config.fallback_asset}")

    print("First momentum scores:")
    display(momentum_score_df.dropna().head())

    print("First month-end decisions:")
    display(month_end_weight_df.head())

    print("First valid rebalance opens:")
    display(rebalance_weight_df.head())

    display(strategy.summary)
    display(strategy.summary_trades)

    save_results(strategy)
