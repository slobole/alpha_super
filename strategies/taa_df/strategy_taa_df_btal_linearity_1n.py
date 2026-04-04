"""
Defense First tactical allocation with BTAL added to the defensive sleeve,
equal defensive slots, and a simple regression-based linearity signal.

This variant preserves the monthly execution contract from `strategy_taa_df.py`
and the equal-slot sizing convention from `strategy_taa_df_btal_1n.py`, while
replacing the simple-return ranking signal with a non-annualized linearity
signal computed from daily log prices.

For asset i and lookback L in {21, 63, 126, 252} trading days:

    x_n = n,  n = 0, 1, ..., L - 1
    y_n = log(P_{i,t-L+1+n})

    slope_{i,t}^{(L)}
        = sum_n[(x_n - x_bar) * (y_n - y_bar)] / sum_n[(x_n - x_bar)^2]

    R2_{i,t}^{(L)}
        = corr(x, y)^2

    R2_adj_{i,t}^{(L)}
        = 1 - (1 - R2_{i,t}^{(L)}) * (L - 1) / (L - 2)

    linearity_score_{i,t}^{(L)}
        = R2_adj_{i,t}^{(L)} * slope_{i,t}^{(L)}

Composite daily score:

    linearity_score_{i,t}
        = average(
            linearity_score_{i,t}^{(21)},
            linearity_score_{i,t}^{(63)},
            linearity_score_{i,t}^{(126)},
            linearity_score_{i,t}^{(252)},
          )

Month-end qualification uses a zero hurdle:

    pass_{i,m} = 1 if month_end_score_{i,m} > 0 else 0

Equal defensive slot sizing:

    w_{i,m} = pass_{i,m} / N_def
    w_{SPY,m} = 1 - sum_i(w_{i,m})

where:

    N_def = 5

Quantitative consequence versus the simple-return `1/n` variant:
1. Old behavior: monthly simple return ranking over 1/3/6/12 months versus cash.
2. New behavior: daily regression-based linearity score over 21/63/126/252
   trading days with a zero qualification threshold.
3. Execution consequence: signal timing remains month-end decision to next-month
   first tradable open, so only signal semantics change here.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
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
        load_execution_price_df,
        load_signal_close_df,
        map_month_end_weights_to_rebalance_open_df,
    )
    from strategies.taa_df.strategy_taa_df_btal_1n import DEFAULT_CONFIG as BTAL_1N_DEFAULT_CONFIG
    from strategies.taa_df.strategy_taa_df_btal_linearity import (
        compute_daily_linearity_score_df,
        compute_linearity_signal_from_log_price_window_vec,
        linearity_lookback_day_vec,
        linearity_signal_threshold_float,
    )
except ModuleNotFoundError:
    from strategy_taa_df import (
        DefenseFirstConfig,
        DefenseFirstStrategy,
        load_execution_price_df,
        load_signal_close_df,
        map_month_end_weights_to_rebalance_open_df,
    )
    from strategy_taa_df_btal_1n import DEFAULT_CONFIG as BTAL_1N_DEFAULT_CONFIG
    from strategy_taa_df_btal_linearity import (
        compute_daily_linearity_score_df,
        compute_linearity_signal_from_log_price_window_vec,
        linearity_lookback_day_vec,
        linearity_signal_threshold_float,
    )


DEFAULT_CONFIG = DefenseFirstConfig(
    defensive_asset_list=tuple(BTAL_1N_DEFAULT_CONFIG.defensive_asset_list),
    fallback_asset=BTAL_1N_DEFAULT_CONFIG.fallback_asset,
    benchmark_list=tuple(BTAL_1N_DEFAULT_CONFIG.benchmark_list),
    rank_weight_vec=tuple(BTAL_1N_DEFAULT_CONFIG.rank_weight_vec),
    start_date_str=BTAL_1N_DEFAULT_CONFIG.start_date_str,
    end_date_str=BTAL_1N_DEFAULT_CONFIG.end_date_str,
    dtb3_csv_path_str=BTAL_1N_DEFAULT_CONFIG.dtb3_csv_path_str,
)


def compute_month_end_weight_df_from_daily_linearity_score_df(
    daily_linearity_score_df: pd.DataFrame,
    config: DefenseFirstConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Sample daily linearity scores at month-end and build `1/n` defensive weights.
    """
    rank_weight_vec = np.asarray(config.rank_weight_vec, dtype=float)
    if not np.allclose(rank_weight_vec, rank_weight_vec[0], atol=1e-12):
        raise ValueError("Linearity 1/n variant requires equal defensive slot weights.")

    slot_weight_float = float(rank_weight_vec[0])

    # *** CRITICAL*** Month-end sampling must use the last available trading day
    # in each month. Using any intra-month value would change the information set
    # available at the rebalance decision point.
    month_end_score_df = daily_linearity_score_df.resample("ME").last().dropna(how="any")

    tradeable_asset_list = list(config.tradeable_asset_list)
    month_end_weight_df = pd.DataFrame(0.0, index=month_end_score_df.index, columns=tradeable_asset_list, dtype=float)

    for decision_date, score_row_ser in month_end_score_df.iterrows():
        defensive_score_ser = score_row_ser[list(config.defensive_asset_list)].astype(float)
        target_weight_ser = pd.Series(0.0, index=tradeable_asset_list, dtype=float)

        for asset_str in config.defensive_asset_list:
            asset_score_float = float(defensive_score_ser.loc[asset_str])

            if asset_score_float > linearity_signal_threshold_float:
                target_weight_ser.loc[asset_str] = slot_weight_float
            else:
                target_weight_ser.loc[config.fallback_asset] += slot_weight_float

        if not np.isclose(float(target_weight_ser.sum()), 1.0, atol=1e-12):
            raise ValueError(
                f"Target weights must sum to 1.0. Found {float(target_weight_ser.sum()):.12f} on {decision_date}."
            )

        month_end_weight_df.loc[decision_date] = target_weight_ser

    return month_end_score_df, month_end_weight_df


def get_defense_first_linearity_1n_data(
    config: DefenseFirstConfig = DEFAULT_CONFIG,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load signal data, execution data, daily linearity scores, and rebalance weights.
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
    daily_linearity_score_df = compute_daily_linearity_score_df(
        signal_close_df=signal_close_df,
        lookback_day_vec=linearity_lookback_day_vec,
    )
    month_end_score_df, month_end_weight_df = compute_month_end_weight_df_from_daily_linearity_score_df(
        daily_linearity_score_df=daily_linearity_score_df,
        config=config,
    )

    # *** CRITICAL*** Month-end decisions must map to the first tradable open in
    # the following month. Same-bar execution would create look-ahead bias.
    rebalance_weight_df = map_month_end_weights_to_rebalance_open_df(month_end_weight_df, execution_price_df.index)

    return execution_price_df, daily_linearity_score_df, month_end_score_df, month_end_weight_df, rebalance_weight_df


if __name__ == "__main__":
    taa_config = DEFAULT_CONFIG

    (
        execution_price_df,
        daily_linearity_score_df,
        month_end_score_df,
        month_end_weight_df,
        rebalance_weight_df,
    ) = get_defense_first_linearity_1n_data(taa_config)

    strategy = DefenseFirstStrategy(
        name="strategy_taa_df_btal_linearity_1n",
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

    save_results(strategy)
