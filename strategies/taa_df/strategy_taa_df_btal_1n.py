"""
Defense First tactical allocation with BTAL added to the defensive sleeve and
equal defensive slots instead of rank-weighted slots.

This variant preserves the exact signal and execution logic from
`strategy_taa_df.py` and changes only the configuration:

    defensive_asset_list = ("GLD", "UUP", "TLT", "DBC", "BTAL")
    fallback_asset = "SPY"

The momentum formula is unchanged:

    momentum_score_{i,t}
        = (r_{1m,i,t} + r_{3m,i,t} + r_{6m,i,t} + r_{12m,i,t}) / 4

    r_{k,i,t} = close_{i,t} / close_{i,t-k} - 1

The defensive slot sizing changes from rank-weighted to equal slot weights:

    pass_{i,t} = 1 if momentum_score_{i,t} > cash_return_t else 0

    w_{i,t} = pass_{i,t} / N_def
    w_{SPY,t} = 1 - sum_i(w_{i,t})

where:

    N_def = 5

Therefore each defensive slot weight is:

    rank_weight_vec = (0.20, 0.20, 0.20, 0.20, 0.20)

Because every slot weight is equal, the rank ordering remains an implementation
detail but no longer changes the final defensive allocation.

Quantitative consequence:

    start_date_str = max(first_BTAL_date, first_SPY_date) = first_BTAL_date

Because BTAL starts later than DBC and SPY, the evaluation window is
intentionally clipped to the BTAL inception date to avoid requesting
pre-inception signal history or fills.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from IPython.display import display

repo_root_path = Path(__file__).resolve().parents[2]
repo_root_str = str(repo_root_path)
if repo_root_str not in sys.path:
    sys.path.insert(0, repo_root_str)

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
effective_start_date_str = btal_inception_date_str
btal_defensive_asset_list = ("GLD", "UUP", "TLT", "DBC", "BTAL")
equal_slot_weight_float = 1.0 / float(len(btal_defensive_asset_list))
btal_1n_rank_weight_vec = (
    equal_slot_weight_float,
    equal_slot_weight_float,
    equal_slot_weight_float,
    equal_slot_weight_float,
    equal_slot_weight_float,
)

DEFAULT_CONFIG = DefenseFirstConfig(
    defensive_asset_list=btal_defensive_asset_list,
    fallback_asset="SPY",
    rank_weight_vec=btal_1n_rank_weight_vec,
    start_date_str=effective_start_date_str,
)


if __name__ == "__main__":
    taa_config = DEFAULT_CONFIG

    execution_price_df, momentum_score_df, month_end_weight_df, rebalance_weight_df = (
        get_defense_first_data(taa_config)
    )

    strategy = DefenseFirstStrategy(
        name="strategy_taa_df_btal_1n",
        benchmarks=taa_config.benchmark_list,
        rebalance_weight_df=rebalance_weight_df,
        tradeable_asset_list=taa_config.tradeable_asset_list,
        capital_base=100_000,
        slippage=0.00025,
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
