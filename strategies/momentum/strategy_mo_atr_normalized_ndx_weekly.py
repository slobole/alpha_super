"""
Weekly ATR-adjusted momentum rotation for point-in-time Nasdaq 100 members.

This is the weekly-cadence research variant of
``strategy_mo_atr_normalized_ndx``. It keeps the selection, sizing, PIT
universe, regime filter, cost, and next-open execution assumptions unchanged.
Only the rebalance decision grid changes from month-end closes to completed
week closes.

Core formulas
-------------
For stock i on completed week decision date t:

    weekly_roc_{i,t}^{(L)}
        = Close_W_{i,t} / Close_W_{i,t-L} - 1

    prior_close_{i,d}
        = Close_{i,d-1}

    TR_{i,d}
        = max(
            High_{i,d} - Low_{i,d},
            abs(High_{i,d} - prior_close_{i,d}),
            abs(Low_{i,d} - prior_close_{i,d})
        )

    ATR20_{i,t}
        = mean(TR_{i,t-19:t})

    stock_trend_pass_{i,t}
        = 1[Close_{i,t} > SMA100_{i,t}]

    regime_pass_t
        = 1[SPY_t > SMA200_t]

    risk_adj_score_{i,t}
        = weekly_roc_{i,t}^{(L)} / ATR20_{i,t}

Selection and sizing are inherited from the monthly model:

    eligible_{i,t}
        = 1[PIT_NDX_{i,t} = 1 and stock_trend_pass_{i,t} = 1]

    selected_t
        = top max_positions eligible symbols by risk_adj_score_{i,t}

    target_weight_{i,t}
        = 1 / max_positions    if i in selected_t
        = 0                    otherwise

Execution mapping:

    decision_date_t
        = actual last tradable close of a completed week

    execution_date_t
        = next tradable open after decision_date_t
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

import numpy as np
import pandas as pd
from IPython.display import display

from alpha.engine.backtest import run_daily
from alpha.engine.report import save_results
from data.norgate_loader import build_index_constituent_matrix, load_raw_prices
from strategies.momentum.strategy_mo_atr_normalized_ndx import (
    ATR_WINDOW_INT,
    AtrNormalizedNdxConfig,
    AtrNormalizedNdxStrategy,
    audit_pit_universe_df,
)


@dataclass(frozen=True)
class WeeklyAtrNormalizedNdxConfig(AtrNormalizedNdxConfig):
    lookback_week_int: int = 52

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.lookback_week_int <= 0:
            raise ValueError("lookback_week_int must be positive.")


DEFAULT_CONFIG = WeeklyAtrNormalizedNdxConfig()

__all__ = [
    "ATR_WINDOW_INT",
    "DEFAULT_CONFIG",
    "WeeklyAtrNormalizedNdxConfig",
    "WeeklyAtrNormalizedNdxStrategy",
    "audit_pit_universe_df",
    "compute_weekly_atr_normalized_signal_tables",
    "get_weekly_atr_normalized_ndx_data",
    "get_weekly_decision_close_df",
    "map_week_end_decision_dates_to_rebalance_schedule_df",
    "run_variant",
]


def get_weekly_decision_close_df(price_close_df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse daily closes to the actual last tradable close of each completed week.
    """
    if len(price_close_df.index) == 0:
        raise ValueError("price_close_df must not be empty.")

    sorted_price_close_df = price_close_df.sort_index()
    # *** CRITICAL*** Weekly decisions must use the actual last tradable
    # close in each completed W-FRI week, not a synthetic calendar timestamp.
    week_period_idx = sorted_price_close_df.index.to_period("W-FRI")
    decision_date_ser = pd.Series(
        sorted_price_close_df.index,
        index=week_period_idx,
    ).groupby(level=0).max()

    last_available_ts = pd.Timestamp(sorted_price_close_df.index[-1])
    last_week_end_ts = pd.Timestamp(last_available_ts.to_period("W-FRI").end_time).normalize()
    if (
        len(decision_date_ser) > 0
        and pd.Timestamp(decision_date_ser.iloc[-1]) == last_available_ts
        and last_available_ts.normalize() != last_week_end_ts
    ):
        decision_date_ser = decision_date_ser.iloc[:-1]

    decision_date_idx = pd.DatetimeIndex(decision_date_ser.to_numpy(), name="decision_date_ts")
    weekly_decision_close_df = sorted_price_close_df.loc[decision_date_idx].copy()
    weekly_decision_close_df.index = decision_date_idx
    return weekly_decision_close_df


def compute_weekly_atr_normalized_signal_tables(
    price_close_df: pd.DataFrame,
    price_high_df: pd.DataFrame,
    price_low_df: pd.DataFrame,
    regime_close_ser: pd.Series,
    config: WeeklyAtrNormalizedNdxConfig = DEFAULT_CONFIG,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
    pd.DataFrame,
]:
    weekly_decision_close_df = get_weekly_decision_close_df(price_close_df=price_close_df)

    # *** CRITICAL*** Weekly ROC must use only completed-week decision closes
    # and trailing completed-week history.
    weekly_roc_df = (
        weekly_decision_close_df / weekly_decision_close_df.shift(config.lookback_week_int)
    ) - 1.0

    # *** CRITICAL*** prior close alignment for true range must use shift(1)
    # so ATR is strictly trailing.
    prior_close_df = price_close_df.shift(1)
    true_range_df = (price_high_df - price_low_df).combine(
        (price_high_df - prior_close_df).abs(),
        np.maximum,
    )
    true_range_df = true_range_df.combine(
        (price_low_df - prior_close_df).abs(),
        np.maximum,
    )

    # *** CRITICAL*** ATR20 must remain a trailing rolling mean of past true
    # range values only.
    atr_value_df = true_range_df.rolling(
        window=ATR_WINDOW_INT,
        min_periods=ATR_WINDOW_INT,
    ).mean()
    atr_decision_df = atr_value_df.reindex(weekly_decision_close_df.index)

    # *** CRITICAL*** The stock trend filter must remain a trailing rolling
    # average on past closes only.
    stock_trend_sma_df = price_close_df.rolling(
        window=config.stock_trend_window_int,
        min_periods=config.stock_trend_window_int,
    ).mean()
    stock_trend_pass_df = (price_close_df > stock_trend_sma_df).reindex(
        weekly_decision_close_df.index
    )

    # *** CRITICAL*** The regime SMA filter must remain a trailing rolling
    # average on past SPY closes only.
    regime_close_decision_ser = regime_close_ser.reindex(weekly_decision_close_df.index)
    regime_sma_ser = regime_close_ser.rolling(
        window=config.index_trend_window_int,
        min_periods=config.index_trend_window_int,
    ).mean().reindex(weekly_decision_close_df.index)
    regime_pass_ser = regime_close_decision_ser > regime_sma_ser

    risk_adj_score_df = weekly_roc_df / atr_decision_df
    risk_adj_score_df = risk_adj_score_df.replace([np.inf, -np.inf], np.nan)

    valid_weekly_roc_bool_ser = weekly_roc_df.notna().any(axis=1)
    valid_atr_bool_ser = atr_decision_df.notna().any(axis=1)
    valid_stock_trend_bool_ser = stock_trend_pass_df.notna().any(axis=1)
    valid_regime_bool_ser = regime_close_decision_ser.notna() & regime_sma_ser.notna()
    valid_decision_index = weekly_decision_close_df.index[
        valid_weekly_roc_bool_ser
        & valid_atr_bool_ser
        & valid_stock_trend_bool_ser
        & valid_regime_bool_ser
    ]

    weekly_decision_close_df = weekly_decision_close_df.reindex(valid_decision_index)
    weekly_roc_df = weekly_roc_df.reindex(valid_decision_index)
    atr_decision_df = atr_decision_df.reindex(valid_decision_index)
    stock_trend_pass_df = stock_trend_pass_df.reindex(valid_decision_index)
    regime_sma_ser = regime_sma_ser.reindex(valid_decision_index)
    regime_pass_ser = regime_pass_ser.reindex(valid_decision_index)
    risk_adj_score_df = risk_adj_score_df.reindex(valid_decision_index)
    return (
        weekly_decision_close_df,
        weekly_roc_df,
        atr_decision_df,
        stock_trend_pass_df,
        regime_sma_ser,
        regime_pass_ser,
        risk_adj_score_df,
    )


def map_week_end_decision_dates_to_rebalance_schedule_df(
    decision_date_index: pd.DatetimeIndex,
    execution_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    """
    Map each completed-week decision close to the next tradable open.
    """
    if len(execution_index) < 2:
        raise ValueError("execution_index must contain at least two trading dates.")
    if len(decision_date_index) == 0:
        raise ValueError("decision_date_index must not be empty.")

    execution_index = pd.DatetimeIndex(execution_index).sort_values()
    decision_date_index = pd.DatetimeIndex(decision_date_index).sort_values()

    rebalance_schedule_map: dict[pd.Timestamp, pd.Timestamp] = {}
    for decision_date_ts in decision_date_index:
        execution_insert_int = int(
            execution_index.searchsorted(pd.Timestamp(decision_date_ts), side="right")
        )
        if execution_insert_int >= len(execution_index):
            continue

        # *** CRITICAL*** Week-end decisions must execute strictly on the
        # next tradable open after the decision close, never on the same bar.
        execution_date_ts = pd.Timestamp(execution_index[execution_insert_int])
        rebalance_schedule_map[execution_date_ts] = pd.Timestamp(decision_date_ts)

    if len(rebalance_schedule_map) == 0:
        raise RuntimeError("No weekly rebalance dates were generated.")

    rebalance_schedule_df = pd.DataFrame.from_dict(
        rebalance_schedule_map,
        orient="index",
        columns=["decision_date_ts"],
    ).sort_index()
    rebalance_schedule_df.index.name = "execution_date_ts"
    return rebalance_schedule_df


def _extract_signal_inputs_from_pricing_data(
    pricing_data_df: pd.DataFrame,
    regime_symbol_str: str,
) -> tuple[list[str], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series]:
    tradeable_symbol_list = [
        str(symbol_str)
        for symbol_str in pricing_data_df.columns.get_level_values(0).unique()
        if str(symbol_str) != regime_symbol_str
    ]
    if len(tradeable_symbol_list) == 0:
        raise RuntimeError("No tradeable stock symbols were found in pricing_data_df.")

    price_close_df = pd.DataFrame(
        {symbol_str: pricing_data_df[(symbol_str, "Close")] for symbol_str in tradeable_symbol_list},
        index=pricing_data_df.index,
    ).astype(float)
    price_high_df = pd.DataFrame(
        {symbol_str: pricing_data_df[(symbol_str, "High")] for symbol_str in tradeable_symbol_list},
        index=pricing_data_df.index,
    ).astype(float)
    price_low_df = pd.DataFrame(
        {symbol_str: pricing_data_df[(symbol_str, "Low")] for symbol_str in tradeable_symbol_list},
        index=pricing_data_df.index,
    ).astype(float)

    regime_close_key = (regime_symbol_str, "Close")
    if regime_close_key not in pricing_data_df.columns:
        raise RuntimeError(f"Missing regime close data for {regime_symbol_str}.")
    regime_close_ser = pricing_data_df[regime_close_key].astype(float)
    return tradeable_symbol_list, price_close_df, price_high_df, price_low_df, regime_close_ser


def _make_weekly_rebalance_schedule_df(
    pricing_data_df: pd.DataFrame,
    config: WeeklyAtrNormalizedNdxConfig,
) -> pd.DataFrame:
    (
        _tradeable_symbol_list,
        price_close_df,
        price_high_df,
        price_low_df,
        regime_close_ser,
    ) = _extract_signal_inputs_from_pricing_data(
        pricing_data_df=pricing_data_df,
        regime_symbol_str=config.regime_symbol_str,
    )
    (
        weekly_decision_close_df,
        _weekly_roc_df,
        _atr_decision_df,
        _stock_trend_pass_df,
        _regime_sma_ser,
        _regime_pass_ser,
        _risk_adj_score_df,
    ) = compute_weekly_atr_normalized_signal_tables(
        price_close_df=price_close_df,
        price_high_df=price_high_df,
        price_low_df=price_low_df,
        regime_close_ser=regime_close_ser,
        config=config,
    )
    return map_week_end_decision_dates_to_rebalance_schedule_df(
        decision_date_index=pd.DatetimeIndex(weekly_decision_close_df.index),
        execution_index=pricing_data_df.index,
    )


def get_weekly_atr_normalized_ndx_data(
    config: WeeklyAtrNormalizedNdxConfig = DEFAULT_CONFIG,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    _, raw_universe_df = build_index_constituent_matrix(indexname=config.indexname_str)

    history_start_ts = pd.Timestamp(config.history_start_date_str)
    backtest_start_ts = pd.Timestamp(config.backtest_start_date_str)
    filtered_universe_df = raw_universe_df.loc[raw_universe_df.index >= history_start_ts].copy()
    active_universe_df = filtered_universe_df.loc[filtered_universe_df.index >= backtest_start_ts].copy()
    if config.end_date_str is not None:
        end_date_ts = pd.Timestamp(config.end_date_str)
        active_universe_df = active_universe_df.loc[active_universe_df.index <= end_date_ts]

    active_symbol_list = active_universe_df.columns[active_universe_df.sum(axis=0) > 0].tolist()
    if len(active_symbol_list) == 0:
        raise RuntimeError(
            "No active Nasdaq-100 universe symbols were found for the requested backtest window."
        )

    price_symbol_list = list(dict.fromkeys(active_symbol_list + [config.regime_symbol_str]))
    pricing_data_df = load_raw_prices(
        symbols=price_symbol_list,
        benchmarks=[],
        start_date=config.history_start_date_str,
        end_date=config.end_date_str,
    )
    loaded_symbol_list = [
        symbol_str
        for symbol_str in active_symbol_list
        if symbol_str in pricing_data_df.columns.get_level_values(0)
    ]
    universe_df = audit_pit_universe_df(
        universe_df=filtered_universe_df,
        execution_index=pricing_data_df.index,
        tradeable_symbol_list=loaded_symbol_list,
    )

    keep_symbol_set = set(universe_df.columns.tolist() + [config.regime_symbol_str])
    pricing_data_df = pricing_data_df.loc[
        :,
        pricing_data_df.columns.get_level_values(0).isin(keep_symbol_set),
    ].sort_index()

    rebalance_schedule_df = _make_weekly_rebalance_schedule_df(
        pricing_data_df=pricing_data_df,
        config=config,
    )
    return pricing_data_df, universe_df, rebalance_schedule_df


class WeeklyAtrNormalizedNdxStrategy(AtrNormalizedNdxStrategy):
    """
    Long-only weekly Nasdaq-100 momentum rotation with fixed slot sizing.
    """

    def __init__(
        self,
        name: str,
        benchmarks: Sequence[str],
        rebalance_schedule_df: pd.DataFrame,
        regime_symbol_str: str = "SPY",
        capital_base: float = 100_000.0,
        slippage: float = 0.00025,
        commission_per_share: float = 0.005,
        commission_minimum: float = 1.0,
        lookback_week_int: int = 52,
        index_trend_window_int: int = 200,
        stock_trend_window_int: int = 100,
        max_positions_int: int = 10,
    ):
        if lookback_week_int <= 0:
            raise ValueError("lookback_week_int must be positive.")

        super().__init__(
            name=name,
            benchmarks=benchmarks,
            rebalance_schedule_df=rebalance_schedule_df,
            regime_symbol_str=regime_symbol_str,
            capital_base=capital_base,
            slippage=slippage,
            commission_per_share=commission_per_share,
            commission_minimum=commission_minimum,
            lookback_month_int=12,
            index_trend_window_int=index_trend_window_int,
            stock_trend_window_int=stock_trend_window_int,
            max_positions_int=max_positions_int,
        )
        self.lookback_week_int = int(lookback_week_int)

    def compute_signals(self, pricing_data: pd.DataFrame) -> pd.DataFrame:
        signal_data_df = pricing_data.copy()
        (
            _tradeable_symbol_list,
            price_close_df,
            price_high_df,
            price_low_df,
            regime_close_ser,
        ) = _extract_signal_inputs_from_pricing_data(
            pricing_data_df=signal_data_df,
            regime_symbol_str=self.regime_symbol_str,
        )

        helper_config = WeeklyAtrNormalizedNdxConfig(
            regime_symbol_str=self.regime_symbol_str,
            lookback_week_int=self.lookback_week_int,
            index_trend_window_int=self.index_trend_window_int,
            stock_trend_window_int=self.stock_trend_window_int,
            max_positions_int=self.max_positions_int,
        )
        (
            _weekly_decision_close_df,
            weekly_roc_df,
            atr_decision_df,
            stock_trend_pass_df,
            regime_sma_ser,
            regime_pass_ser,
            risk_adj_score_df,
        ) = compute_weekly_atr_normalized_signal_tables(
            price_close_df=price_close_df,
            price_high_df=price_high_df,
            price_low_df=price_low_df,
            regime_close_ser=regime_close_ser,
            config=helper_config,
        )

        weekly_roc_aligned_df = weekly_roc_df.reindex(signal_data_df.index)
        atr_aligned_df = atr_decision_df.reindex(signal_data_df.index)
        stock_trend_pass_aligned_df = stock_trend_pass_df.reindex(signal_data_df.index)
        risk_adj_score_aligned_df = risk_adj_score_df.reindex(signal_data_df.index)
        regime_sma_aligned_ser = regime_sma_ser.reindex(signal_data_df.index)
        regime_pass_aligned_ser = regime_pass_ser.reindex(signal_data_df.index)

        feature_frame_list: list[pd.DataFrame] = []
        feature_map: dict[str, pd.DataFrame] = {
            f"weekly_roc_{self.lookback_week_int}_ser": weekly_roc_aligned_df,
            f"atr_{ATR_WINDOW_INT}_ser": atr_aligned_df,
            "stock_trend_pass_bool": stock_trend_pass_aligned_df,
            "risk_adj_score_ser": risk_adj_score_aligned_df,
        }

        for field_str, field_df in feature_map.items():
            feature_df = field_df.copy()
            feature_df.columns = pd.MultiIndex.from_tuples(
                [(symbol_str, field_str) for symbol_str in feature_df.columns]
            )
            feature_frame_list.append(feature_df)

        regime_feature_df = pd.DataFrame(
            {
                (self.regime_symbol_str, f"regime_sma_{self.index_trend_window_int}_ser"): regime_sma_aligned_ser,
                (self.regime_symbol_str, "regime_pass_bool"): regime_pass_aligned_ser,
            },
            index=signal_data_df.index,
        )
        regime_feature_df.columns = pd.MultiIndex.from_tuples(regime_feature_df.columns)
        feature_frame_list.append(regime_feature_df)

        return pd.concat([signal_data_df] + feature_frame_list, axis=1)


def run_variant(
    show_display_bool: bool = True,
    save_results_bool: bool = True,
    output_dir_str: str = "results",
    backtest_start_date_str: str | None = None,
    capital_base_float: float | None = None,
    end_date_str: str | None = None,
) -> WeeklyAtrNormalizedNdxStrategy:
    config_obj = DEFAULT_CONFIG
    if (
        backtest_start_date_str is not None
        or capital_base_float is not None
        or end_date_str is not None
    ):
        config_obj = replace(
            DEFAULT_CONFIG,
            backtest_start_date_str=(
                DEFAULT_CONFIG.backtest_start_date_str
                if backtest_start_date_str is None
                else backtest_start_date_str
            ),
            capital_base_float=(
                DEFAULT_CONFIG.capital_base_float
                if capital_base_float is None
                else float(capital_base_float)
            ),
            end_date_str=end_date_str,
        )
    pricing_data_df, universe_df, rebalance_schedule_df = get_weekly_atr_normalized_ndx_data(
        config_obj
    )

    strategy_obj = WeeklyAtrNormalizedNdxStrategy(
        name="strategy_mo_atr_normalized_ndx_weekly",
        benchmarks=[config_obj.regime_symbol_str],
        rebalance_schedule_df=rebalance_schedule_df,
        regime_symbol_str=config_obj.regime_symbol_str,
        capital_base=config_obj.capital_base_float,
        slippage=config_obj.slippage_float,
        commission_per_share=config_obj.commission_per_share_float,
        commission_minimum=config_obj.commission_minimum_float,
        lookback_week_int=config_obj.lookback_week_int,
        index_trend_window_int=config_obj.index_trend_window_int,
        stock_trend_window_int=config_obj.stock_trend_window_int,
        max_positions_int=config_obj.max_positions_int,
    )
    strategy_obj.universe_df = universe_df

    # *** CRITICAL*** Research backtests keep full pre-start history for
    # weekly momentum, daily ATR, and trend features, but the executable
    # calendar starts at the first requested fill session.
    calendar_idx = pricing_data_df.index[
        pricing_data_df.index >= pd.Timestamp(config_obj.backtest_start_date_str)
    ]
    run_daily(
        strategy_obj,
        pricing_data_df,
        calendar=calendar_idx,
        show_progress=show_display_bool,
        show_signal_progress_bool=show_display_bool,
        audit_override_bool=None,
    )

    if show_display_bool:
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 1000)
        display(strategy_obj.summary)
        display(strategy_obj.summary_trades)

    if save_results_bool:
        save_results(strategy_obj, output_dir=output_dir_str)

    return strategy_obj


if __name__ == "__main__":
    run_variant()
