"""
Defense First tactical allocation with BTAL added to the defensive sleeve,
equal defensive slots, and a simple FIP-penalized linearity signal.

This variant preserves the monthly execution contract from `strategy_taa_df.py`
and the equal-slot sizing convention from `strategy_taa_df_btal_1n.py`, while
replacing the simple-return ranking signal with a daily log-price regression
signal penalized by the fraction of negative days inside each trailing window.

For asset i and lookback L in {21, 63, 126, 252} trading days:

    x_n = n,  n = 0, 1, ..., L - 1
    y_n = log(P_{i,t-L+1+n})

    slope_{i,t}^{(L)}
        = sum_n[(x_n - x_bar) * (y_n - y_bar)] / sum_n[(x_n - x_bar)^2]

    R2_{i,t}^{(L)}
        = corr(x, y)^2

    R2_adj_{i,t}^{(L)}
        = 1 - (1 - R2_{i,t}^{(L)}) * (L - 1) / (L - 2)

    frac_neg_{i,t}^{(L)}
        = mean(Delta y_n < 0)

    fip_score_{i,t}^{(L)}
        = R2_adj_{i,t}^{(L)} * slope_{i,t}^{(L)} * (1 - frac_neg_{i,t}^{(L)})

Composite daily score:

    fip_score_{i,t}
        = average(
            fip_score_{i,t}^{(21)},
            fip_score_{i,t}^{(63)},
            fip_score_{i,t}^{(126)},
            fip_score_{i,t}^{(252)},
          )

Month-end qualification uses a zero hurdle:

    pass_{i,m} = 1 if month_end_score_{i,m} > 0 else 0

Equal defensive slot sizing:

    w_{i,m} = pass_{i,m} / N_def
    w_{SPY,m} = 1 - sum_i(w_{i,m})

where:

    N_def = 5

Quantitative consequence versus the raw-linearity `1/n` variant:
1. Old behavior: score = R2_adj * slope.
2. New behavior: score = R2_adj * slope * (1 - frac_neg_days).
3. Execution consequence: signal timing remains month-end decision to next-month
   first tradable open, so only signal semantics change here.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from IPython.display import display

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
except ModuleNotFoundError:
    from strategy_taa_df import (
        DefenseFirstConfig,
        DefenseFirstStrategy,
        load_execution_price_df,
        load_signal_close_df,
        map_month_end_weights_to_rebalance_open_df,
    )
    from strategy_taa_df_btal_1n import DEFAULT_CONFIG as BTAL_1N_DEFAULT_CONFIG


fip_lookback_day_vec = (21, 63, 126, 252)
fip_signal_threshold_float = 0.0

DEFAULT_CONFIG = DefenseFirstConfig(
    defensive_asset_list=tuple(BTAL_1N_DEFAULT_CONFIG.defensive_asset_list),
    fallback_asset=BTAL_1N_DEFAULT_CONFIG.fallback_asset,
    benchmark_list=tuple(BTAL_1N_DEFAULT_CONFIG.benchmark_list),
    rank_weight_vec=tuple(BTAL_1N_DEFAULT_CONFIG.rank_weight_vec),
    start_date_str=BTAL_1N_DEFAULT_CONFIG.start_date_str,
    end_date_str=BTAL_1N_DEFAULT_CONFIG.end_date_str,
    dtb3_csv_path_str=BTAL_1N_DEFAULT_CONFIG.dtb3_csv_path_str,
)


def compute_fip_score_from_log_price_window_vec(
    log_price_window_vec: np.ndarray,
    x_centered_vec: np.ndarray,
    sxx_float: float,
) -> float:
    """
    Compute a simple FIP-penalized linearity score from one trailing log-price
    window.

    The score is:

        score = R2_adj * slope * (1 - frac_neg_days)
    """
    if np.isnan(log_price_window_vec).any():
        return np.nan

    window_length_int = int(len(log_price_window_vec))
    if window_length_int < 3:
        return np.nan

    y_mean_float = float(np.mean(log_price_window_vec))
    y_centered_vec = log_price_window_vec - y_mean_float
    syy_float = float(np.dot(y_centered_vec, y_centered_vec))

    if not np.isfinite(syy_float) or syy_float <= 0.0:
        return 0.0

    sxy_float = float(np.dot(x_centered_vec, y_centered_vec))
    slope_float = sxy_float / sxx_float
    corr_float = sxy_float / float(np.sqrt(sxx_float * syy_float))
    r_squared_float = float(corr_float * corr_float)
    adjusted_r_squared_float = float(1.0 - (1.0 - r_squared_float) * (window_length_int - 1.0) / (window_length_int - 2.0))

    # *** CRITICAL*** The negative-day penalty must be computed strictly inside
    # the trailing window. Using any return outside the current window would
    # leak information across decision timestamps.
    daily_log_return_vec = np.diff(log_price_window_vec)
    fraction_negative_days_float = float(np.mean(daily_log_return_vec < 0.0))

    return adjusted_r_squared_float * slope_float * (1.0 - fraction_negative_days_float)


def compute_daily_fip_score_df(
    signal_close_df: pd.DataFrame,
    lookback_day_vec: Sequence[int] = fip_lookback_day_vec,
) -> pd.DataFrame:
    """
    Compute daily composite FIP scores from TOTALRETURN closes.
    """
    log_signal_close_df = np.log(signal_close_df.astype(float))
    lookback_score_df_list: list[pd.DataFrame] = []

    for lookback_day_int in lookback_day_vec:
        time_index_vec = np.arange(lookback_day_int, dtype=float)
        time_centered_vec = time_index_vec - float(np.mean(time_index_vec))
        sxx_float = float(np.dot(time_centered_vec, time_centered_vec))
        asset_score_map: dict[str, pd.Series] = {}

        for asset_str in log_signal_close_df.columns:
            log_price_ser = log_signal_close_df[asset_str].astype(float)

            # *** CRITICAL*** The rolling regression window must be strictly
            # backward-looking. Any centered or forward window would leak future
            # path information into the current score.
            asset_score_ser = log_price_ser.rolling(lookback_day_int).apply(
                lambda log_price_window_vec: compute_fip_score_from_log_price_window_vec(
                    log_price_window_vec=log_price_window_vec,
                    x_centered_vec=time_centered_vec,
                    sxx_float=sxx_float,
                ),
                raw=True,
            )
            asset_score_map[asset_str] = asset_score_ser

        lookback_score_df = pd.DataFrame(asset_score_map, index=log_signal_close_df.index, dtype=float)
        lookback_score_df_list.append(lookback_score_df)

    daily_fip_score_df = sum(lookback_score_df_list) / float(len(lookback_score_df_list))
    return daily_fip_score_df


def compute_month_end_weight_df_from_daily_fip_score_df(
    daily_fip_score_df: pd.DataFrame,
    config: DefenseFirstConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Sample daily FIP scores at month-end and build `1/n` defensive weights.
    """
    rank_weight_vec = np.asarray(config.rank_weight_vec, dtype=float)
    if not np.allclose(rank_weight_vec, rank_weight_vec[0], atol=1e-12):
        raise ValueError("FIP 1/n variant requires equal defensive slot weights.")

    slot_weight_float = float(rank_weight_vec[0])

    # *** CRITICAL*** Month-end sampling must use the last available trading day
    # in each month. Using any intra-month value would change the information set
    # available at the rebalance decision point.
    month_end_score_df = daily_fip_score_df.resample("ME").last().dropna(how="any")

    tradeable_asset_list = list(config.tradeable_asset_list)
    month_end_weight_df = pd.DataFrame(0.0, index=month_end_score_df.index, columns=tradeable_asset_list, dtype=float)

    for decision_date_ts, score_row_ser in month_end_score_df.iterrows():
        defensive_score_ser = score_row_ser[list(config.defensive_asset_list)].astype(float)
        target_weight_ser = pd.Series(0.0, index=tradeable_asset_list, dtype=float)

        for asset_str in config.defensive_asset_list:
            asset_score_float = float(defensive_score_ser.loc[asset_str])

            if asset_score_float > fip_signal_threshold_float:
                target_weight_ser.loc[asset_str] = slot_weight_float
            else:
                target_weight_ser.loc[config.fallback_asset] += slot_weight_float

        if not np.isclose(float(target_weight_ser.sum()), 1.0, atol=1e-12):
            raise ValueError(
                f"Target weights must sum to 1.0. Found {float(target_weight_ser.sum()):.12f} on {decision_date_ts}."
            )

        month_end_weight_df.loc[decision_date_ts] = target_weight_ser

    return month_end_score_df, month_end_weight_df


def get_defense_first_fip_1n_data(
    config: DefenseFirstConfig = DEFAULT_CONFIG,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load signal data, execution data, daily FIP scores, and rebalance weights.
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
    daily_fip_score_df = compute_daily_fip_score_df(
        signal_close_df=signal_close_df,
        lookback_day_vec=fip_lookback_day_vec,
    )
    month_end_score_df, month_end_weight_df = compute_month_end_weight_df_from_daily_fip_score_df(
        daily_fip_score_df=daily_fip_score_df,
        config=config,
    )

    # *** CRITICAL*** Month-end decisions must map to the first tradable open in
    # the following month. Same-bar execution would create look-ahead bias.
    rebalance_weight_df = map_month_end_weights_to_rebalance_open_df(month_end_weight_df, execution_price_df.index)

    return execution_price_df, daily_fip_score_df, month_end_score_df, month_end_weight_df, rebalance_weight_df


if __name__ == "__main__":
    taa_config = DEFAULT_CONFIG

    (
        execution_price_df,
        daily_fip_score_df,
        month_end_score_df,
        month_end_weight_df,
        rebalance_weight_df,
    ) = get_defense_first_fip_1n_data(taa_config)

    strategy = DefenseFirstStrategy(
        name="strategy_taa_df_btal_fip_1n",
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

    print("First daily FIP scores:")
    display(daily_fip_score_df.dropna().head())

    print("First month-end FIP scores:")
    display(month_end_score_df.head())

    print("First month-end decisions:")
    display(month_end_weight_df.head())

    print("First rebalance opens:")
    display(rebalance_weight_df.head())

    display(strategy.summary)
    display(strategy.summary_trades)

    save_results(strategy)
