"""
Weekly ATR-adjusted Nasdaq-100 momentum rotation with a KAMA trend filter.

This research variant keeps the weekly ATR-normalized NDX model intact and
adds one stock-level trend condition:

    kama_pass_{i,t}
        = 1[Close_{i,t} > KAMA_{i,t}]

KAMA uses the standard Kaufman defaults in this module:

    er_window = 10
    fast_period = 2
    slow_period = 30

Core formulas
-------------
For stock i on daily bar d:

    change_{i,d}
        = abs(Close_{i,d} - Close_{i,d-er_window})

    volatility_{i,d}
        = sum(abs(Close_{i,j} - Close_{i,j-1}), j=d-er_window+1..d)

    ER_{i,d}
        = change_{i,d} / volatility_{i,d}

    fast_sc
        = 2 / (fast_period + 1)

    slow_sc
        = 2 / (slow_period + 1)

    SC_{i,d}
        = (ER_{i,d} * (fast_sc - slow_sc) + slow_sc)^2

    KAMA_{i,d}
        = KAMA_{i,d-1} + SC_{i,d} * (Close_{i,d} - KAMA_{i,d-1})

Selection then requires every inherited weekly condition plus:

    Close_{i,t} > KAMA_{i,t}

Execution mapping is unchanged: completed weekly decision close t to the next
tradable open.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

import numpy as np
import pandas as pd
from IPython.display import display

from alpha.engine.backtest import run_daily
from alpha.engine.report import save_results
from strategies.momentum.strategy_mo_atr_normalized_ndx_weekly import (
    ATR_WINDOW_INT,
    WeeklyAtrNormalizedNdxConfig,
    WeeklyAtrNormalizedNdxStrategy,
    _extract_signal_inputs_from_pricing_data,
    audit_pit_universe_df,
    compute_weekly_atr_normalized_signal_tables,
    get_weekly_atr_normalized_ndx_data,
    map_week_end_decision_dates_to_rebalance_schedule_df,
)


DEFAULT_KAMA_ER_WINDOW_INT = 10
DEFAULT_KAMA_FAST_PERIOD_INT = 2
DEFAULT_KAMA_SLOW_PERIOD_INT = 30


@dataclass(frozen=True)
class WeeklyKamaAtrNormalizedNdxConfig(WeeklyAtrNormalizedNdxConfig):
    kama_er_window_int: int = DEFAULT_KAMA_ER_WINDOW_INT
    kama_fast_period_int: int = DEFAULT_KAMA_FAST_PERIOD_INT
    kama_slow_period_int: int = DEFAULT_KAMA_SLOW_PERIOD_INT

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.kama_er_window_int <= 0:
            raise ValueError("kama_er_window_int must be positive.")
        if self.kama_fast_period_int <= 0:
            raise ValueError("kama_fast_period_int must be positive.")
        if self.kama_slow_period_int <= 0:
            raise ValueError("kama_slow_period_int must be positive.")
        if self.kama_fast_period_int >= self.kama_slow_period_int:
            raise ValueError("kama_fast_period_int must be less than kama_slow_period_int.")


DEFAULT_CONFIG = WeeklyKamaAtrNormalizedNdxConfig()

__all__ = [
    "ATR_WINDOW_INT",
    "DEFAULT_CONFIG",
    "DEFAULT_KAMA_ER_WINDOW_INT",
    "DEFAULT_KAMA_FAST_PERIOD_INT",
    "DEFAULT_KAMA_SLOW_PERIOD_INT",
    "WeeklyKamaAtrNormalizedNdxConfig",
    "WeeklyKamaAtrNormalizedNdxStrategy",
    "audit_pit_universe_df",
    "compute_kama_df",
    "compute_weekly_kama_atr_normalized_signal_tables",
    "get_weekly_atr_normalized_ndx_data",
    "map_week_end_decision_dates_to_rebalance_schedule_df",
    "run_variant",
]


def compute_kama_df(
    price_close_df: pd.DataFrame,
    er_window_int: int = DEFAULT_KAMA_ER_WINDOW_INT,
    fast_period_int: int = DEFAULT_KAMA_FAST_PERIOD_INT,
    slow_period_int: int = DEFAULT_KAMA_SLOW_PERIOD_INT,
) -> pd.DataFrame:
    """
    Compute daily Kaufman Adaptive Moving Average values for each column.
    """
    if er_window_int <= 0:
        raise ValueError("er_window_int must be positive.")
    if fast_period_int <= 0:
        raise ValueError("fast_period_int must be positive.")
    if slow_period_int <= 0:
        raise ValueError("slow_period_int must be positive.")
    if fast_period_int >= slow_period_int:
        raise ValueError("fast_period_int must be less than slow_period_int.")

    clean_price_close_df = price_close_df.astype(float).sort_index()
    if len(clean_price_close_df) == 0:
        raise ValueError("price_close_df must not be empty.")

    # *** CRITICAL*** KAMA efficiency ratio uses shift(er_window_int) and a
    # trailing rolling sum of absolute close-to-close moves. It may use the
    # decision-date close only because orders execute on the next tradable open.
    change_df = (clean_price_close_df - clean_price_close_df.shift(er_window_int)).abs()
    absolute_move_df = clean_price_close_df.diff().abs()
    volatility_df = absolute_move_df.rolling(
        window=er_window_int,
        min_periods=er_window_int,
    ).sum()
    efficiency_ratio_df = change_df / volatility_df.replace(0.0, np.nan)
    efficiency_ratio_df = efficiency_ratio_df.clip(lower=0.0, upper=1.0).fillna(0.0)

    fast_sc_float = 2.0 / (float(fast_period_int) + 1.0)
    slow_sc_float = 2.0 / (float(slow_period_int) + 1.0)
    smoothing_constant_df = (
        efficiency_ratio_df * (fast_sc_float - slow_sc_float) + slow_sc_float
    ) ** 2.0

    kama_df = pd.DataFrame(np.nan, index=clean_price_close_df.index, columns=clean_price_close_df.columns)
    if len(clean_price_close_df.index) <= er_window_int:
        return kama_df.astype(float)

    seed_row_int = int(er_window_int)
    kama_df.iloc[seed_row_int] = clean_price_close_df.iloc[seed_row_int]
    for row_int in range(seed_row_int + 1, len(clean_price_close_df.index)):
        price_ser = clean_price_close_df.iloc[row_int]
        prior_kama_ser = kama_df.iloc[row_int - 1]
        smoothing_constant_ser = smoothing_constant_df.iloc[row_int]
        # *** CRITICAL*** KAMA recursion is causal: KAMA_d depends only on
        # KAMA_{d-1}, Close_d, and trailing efficiency data through d.
        kama_df.iloc[row_int] = prior_kama_ser + smoothing_constant_ser * (
            price_ser - prior_kama_ser
        )

    return kama_df.astype(float)


def compute_weekly_kama_atr_normalized_signal_tables(
    price_close_df: pd.DataFrame,
    price_high_df: pd.DataFrame,
    price_low_df: pd.DataFrame,
    regime_close_ser: pd.Series,
    config: WeeklyKamaAtrNormalizedNdxConfig = DEFAULT_CONFIG,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
    pd.DataFrame,
]:
    (
        weekly_decision_close_df,
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
        config=config,
    )

    kama_value_df = compute_kama_df(
        price_close_df=price_close_df,
        er_window_int=config.kama_er_window_int,
        fast_period_int=config.kama_fast_period_int,
        slow_period_int=config.kama_slow_period_int,
    )
    kama_decision_df = kama_value_df.reindex(weekly_decision_close_df.index)
    # *** CRITICAL*** KAMA filter is sampled on the completed weekly decision
    # close. It must not use any post-decision price before next-open execution.
    kama_pass_df = weekly_decision_close_df > kama_decision_df

    valid_kama_bool_ser = kama_decision_df.notna().any(axis=1)
    valid_decision_index = weekly_decision_close_df.index[valid_kama_bool_ser]

    return (
        weekly_decision_close_df.reindex(valid_decision_index),
        weekly_roc_df.reindex(valid_decision_index),
        atr_decision_df.reindex(valid_decision_index),
        stock_trend_pass_df.reindex(valid_decision_index),
        kama_decision_df.reindex(valid_decision_index),
        kama_pass_df.reindex(valid_decision_index),
        regime_sma_ser.reindex(valid_decision_index),
        regime_pass_ser.reindex(valid_decision_index),
        risk_adj_score_df.reindex(valid_decision_index),
    )


class WeeklyKamaAtrNormalizedNdxStrategy(WeeklyAtrNormalizedNdxStrategy):
    """
    Weekly ATR-normalized NDX momentum with an added Close > KAMA stock filter.
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
        kama_er_window_int: int = DEFAULT_KAMA_ER_WINDOW_INT,
        kama_fast_period_int: int = DEFAULT_KAMA_FAST_PERIOD_INT,
        kama_slow_period_int: int = DEFAULT_KAMA_SLOW_PERIOD_INT,
    ):
        super().__init__(
            name=name,
            benchmarks=benchmarks,
            rebalance_schedule_df=rebalance_schedule_df,
            regime_symbol_str=regime_symbol_str,
            capital_base=capital_base,
            slippage=slippage,
            commission_per_share=commission_per_share,
            commission_minimum=commission_minimum,
            lookback_week_int=lookback_week_int,
            index_trend_window_int=index_trend_window_int,
            stock_trend_window_int=stock_trend_window_int,
            max_positions_int=max_positions_int,
        )
        if kama_er_window_int <= 0:
            raise ValueError("kama_er_window_int must be positive.")
        if kama_fast_period_int <= 0:
            raise ValueError("kama_fast_period_int must be positive.")
        if kama_slow_period_int <= 0:
            raise ValueError("kama_slow_period_int must be positive.")
        if kama_fast_period_int >= kama_slow_period_int:
            raise ValueError("kama_fast_period_int must be less than kama_slow_period_int.")

        self.kama_er_window_int = int(kama_er_window_int)
        self.kama_fast_period_int = int(kama_fast_period_int)
        self.kama_slow_period_int = int(kama_slow_period_int)

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

        helper_config = WeeklyKamaAtrNormalizedNdxConfig(
            regime_symbol_str=self.regime_symbol_str,
            lookback_week_int=self.lookback_week_int,
            index_trend_window_int=self.index_trend_window_int,
            stock_trend_window_int=self.stock_trend_window_int,
            max_positions_int=self.max_positions_int,
            kama_er_window_int=self.kama_er_window_int,
            kama_fast_period_int=self.kama_fast_period_int,
            kama_slow_period_int=self.kama_slow_period_int,
        )
        (
            _weekly_decision_close_df,
            weekly_roc_df,
            atr_decision_df,
            stock_trend_pass_df,
            kama_decision_df,
            kama_pass_df,
            regime_sma_ser,
            regime_pass_ser,
            risk_adj_score_df,
        ) = compute_weekly_kama_atr_normalized_signal_tables(
            price_close_df=price_close_df,
            price_high_df=price_high_df,
            price_low_df=price_low_df,
            regime_close_ser=regime_close_ser,
            config=helper_config,
        )

        feature_frame_list: list[pd.DataFrame] = []
        feature_map: dict[str, pd.DataFrame] = {
            f"weekly_roc_{self.lookback_week_int}_ser": weekly_roc_df.reindex(signal_data_df.index),
            f"atr_{ATR_WINDOW_INT}_ser": atr_decision_df.reindex(signal_data_df.index),
            "stock_trend_pass_bool": stock_trend_pass_df.reindex(signal_data_df.index),
            (
                f"kama_{self.kama_er_window_int}_{self.kama_fast_period_int}_"
                f"{self.kama_slow_period_int}_ser"
            ): kama_decision_df.reindex(signal_data_df.index),
            "kama_pass_bool": kama_pass_df.reindex(signal_data_df.index),
            "risk_adj_score_ser": risk_adj_score_df.reindex(signal_data_df.index),
        }

        for field_str, field_df in feature_map.items():
            feature_df = field_df.copy()
            feature_df.columns = pd.MultiIndex.from_tuples(
                [(symbol_str, field_str) for symbol_str in feature_df.columns]
            )
            feature_frame_list.append(feature_df)

        regime_feature_df = pd.DataFrame(
            {
                (
                    self.regime_symbol_str,
                    f"regime_sma_{self.index_trend_window_int}_ser",
                ): regime_sma_ser.reindex(signal_data_df.index),
                (self.regime_symbol_str, "regime_pass_bool"): regime_pass_ser.reindex(signal_data_df.index),
            },
            index=signal_data_df.index,
        )
        regime_feature_df.columns = pd.MultiIndex.from_tuples(regime_feature_df.columns)
        feature_frame_list.append(regime_feature_df)

        return pd.concat([signal_data_df] + feature_frame_list, axis=1)

    def get_target_weight_ser(self, close_row_ser: pd.Series) -> pd.Series:
        candidate_feature_df = close_row_ser.unstack()
        if "kama_pass_bool" not in candidate_feature_df.columns:
            return pd.Series(dtype=float)

        kama_pass_raw_ser = candidate_feature_df["kama_pass_bool"]
        kama_pass_ser = kama_pass_raw_ser.where(kama_pass_raw_ser.notna(), False).astype(bool)
        filtered_close_row_ser = close_row_ser.copy()
        for symbol_str, pass_bool in kama_pass_ser.items():
            if bool(pass_bool):
                continue
            if (symbol_str, "stock_trend_pass_bool") in filtered_close_row_ser.index:
                filtered_close_row_ser.loc[(symbol_str, "stock_trend_pass_bool")] = False

        return super().get_target_weight_ser(close_row_ser=filtered_close_row_ser)


def _make_weekly_kama_rebalance_schedule_df(
    pricing_data_df: pd.DataFrame,
    config: WeeklyKamaAtrNormalizedNdxConfig,
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
        _kama_decision_df,
        _kama_pass_df,
        _regime_sma_ser,
        _regime_pass_ser,
        _risk_adj_score_df,
    ) = compute_weekly_kama_atr_normalized_signal_tables(
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


def get_weekly_kama_atr_normalized_ndx_data(
    config: WeeklyKamaAtrNormalizedNdxConfig = DEFAULT_CONFIG,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pricing_data_df, universe_df, _weekly_rebalance_schedule_df = get_weekly_atr_normalized_ndx_data(
        config=config,
    )
    rebalance_schedule_df = _make_weekly_kama_rebalance_schedule_df(
        pricing_data_df=pricing_data_df,
        config=config,
    )
    return pricing_data_df, universe_df, rebalance_schedule_df


def run_variant(
    show_display_bool: bool = True,
    save_results_bool: bool = True,
    output_dir_str: str = "results",
    backtest_start_date_str: str | None = None,
    capital_base_float: float | None = None,
    end_date_str: str | None = None,
) -> WeeklyKamaAtrNormalizedNdxStrategy:
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
    pricing_data_df, universe_df, rebalance_schedule_df = get_weekly_kama_atr_normalized_ndx_data(
        config_obj
    )

    strategy_obj = WeeklyKamaAtrNormalizedNdxStrategy(
        name="strategy_mo_atr_normalized_ndx_weekly_kama",
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
        kama_er_window_int=config_obj.kama_er_window_int,
        kama_fast_period_int=config_obj.kama_fast_period_int,
        kama_slow_period_int=config_obj.kama_slow_period_int,
    )
    strategy_obj.universe_df = universe_df

    # *** CRITICAL*** Research backtests keep full pre-start history for
    # weekly momentum, daily KAMA/ATR, and trend features, but the executable
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
