"""
Monthly PTA-conditioned winner-continuation research strategy.

Core formula
------------
For stock i on month-end decision date t:

    PTH_{i,t}
        = Close_{i,t} / max(Close_{i,t-251}, ..., Close_{i,t})

    PTL_{i,t}
        = Close_{i,t} / min(Close_{i,t-251}, ..., Close_{i,t})

    PTA_{i,t}
        = (zscore(winsor_1pct(PTH_{i,t}))
           + zscore(winsor_1pct(PTL_{i,t}))) / 2

Selection for the default long-short 10x10 research config:

    return_bucket_{i,t}
        = decile rank of month-t close-to-close return

    pta_bucket_{i,t}
        = decile rank of PTA_{i,t-1}

    long_t
        = stocks where return_bucket_{i,t} = 10 and pta_bucket_{i,t} = 10

    short_t
        = stocks where return_bucket_{i,t} = 10 and pta_bucket_{i,t} = 1

Execution:

    decision_date_t
        = actual last tradable close of month t

    execution_date_t
        = next tradable open after decision_date_t under the Vanilla engine
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, replace
from typing import Sequence

import numpy as np
import pandas as pd
from IPython.display import display

from alpha.engine.backtest import run_daily
from alpha.engine.report import save_results
from alpha.engine.strategy import Strategy
from data.norgate_loader import build_index_constituent_matrix, load_raw_prices
from strategies.momentum.strategy_mo_atr_normalized_ndx import (
    audit_pit_universe_df,
    get_asof_universe_membership_ser,
    get_monthly_decision_close_df,
    map_month_end_decision_dates_to_rebalance_schedule_df,
)


PTA_LOOKBACK_TRADING_DAY_INT = 252
BUCKET_COUNT_INT = 10
LONG_GROSS_EXPOSURE_FLOAT = 1.0
SHORT_GROSS_EXPOSURE_FLOAT = 1.0
WINSOR_TAIL_PROB_FLOAT = 0.01
STOCK_SMA_WINDOW_INT = 100
INVERSE_VOL_WINDOW_INT = 63


def default_trade_id_int() -> int:
    return -1


@dataclass(frozen=True)
class PtaWinnerContinuationConfig:
    variant_key_str: str
    indexname_str: str
    benchmark_symbol_str: str
    regime_symbol_str: str | None = None
    history_start_date_str: str = "1998-01-01"
    backtest_start_date_str: str = "2000-01-01"
    end_date_str: str | None = None
    pta_lookback_trading_day_int: int = PTA_LOOKBACK_TRADING_DAY_INT
    bucket_count_int: int = BUCKET_COUNT_INT
    winner_return_bucket_int: int = BUCKET_COUNT_INT
    long_pta_bucket_int: int = BUCKET_COUNT_INT
    short_pta_bucket_int: int = 1
    long_gross_exposure_float: float = LONG_GROSS_EXPOSURE_FLOAT
    short_gross_exposure_float: float = SHORT_GROSS_EXPOSURE_FLOAT
    winsor_tail_prob_float: float = WINSOR_TAIL_PROB_FLOAT
    index_sma_filter_enabled_bool: bool = False
    index_sma_window_int: int = 200
    stock_sma_filter_enabled_bool: bool = False
    stock_sma_window_int: int = STOCK_SMA_WINDOW_INT
    inverse_vol_weighting_enabled_bool: bool = False
    inverse_vol_window_int: int = INVERSE_VOL_WINDOW_INT
    max_long_positions_int: int | None = None
    capital_base_float: float = 100_000.0
    slippage_float: float = 0.00025
    commission_per_share_float: float = 0.005
    commission_minimum_float: float = 1.0

    def __post_init__(self) -> None:
        if not self.variant_key_str:
            raise ValueError("variant_key_str must not be empty.")
        if not self.indexname_str:
            raise ValueError("indexname_str must not be empty.")
        if not self.benchmark_symbol_str:
            raise ValueError("benchmark_symbol_str must not be empty.")
        if pd.Timestamp(self.history_start_date_str) >= pd.Timestamp(self.backtest_start_date_str):
            raise ValueError("history_start_date_str must be earlier than backtest_start_date_str.")
        if self.pta_lookback_trading_day_int <= 1:
            raise ValueError("pta_lookback_trading_day_int must be greater than 1.")
        if self.bucket_count_int <= 1:
            raise ValueError("bucket_count_int must be greater than 1.")
        for bucket_int in (
            self.winner_return_bucket_int,
            self.long_pta_bucket_int,
            self.short_pta_bucket_int,
        ):
            if bucket_int < 1 or bucket_int > self.bucket_count_int:
                raise ValueError("bucket ids must be between 1 and bucket_count_int.")
        if self.long_pta_bucket_int == self.short_pta_bucket_int:
            raise ValueError("long_pta_bucket_int and short_pta_bucket_int must differ.")
        if self.long_gross_exposure_float < 0.0:
            raise ValueError("long_gross_exposure_float must be non-negative.")
        if self.short_gross_exposure_float < 0.0:
            raise ValueError("short_gross_exposure_float must be non-negative.")
        if not 0.0 <= self.winsor_tail_prob_float < 0.5:
            raise ValueError("winsor_tail_prob_float must be in [0.0, 0.5).")
        if self.index_sma_filter_enabled_bool and not self.regime_symbol_str:
            raise ValueError("regime_symbol_str must be set when index_sma_filter_enabled_bool is True.")
        if self.index_sma_window_int <= 1:
            raise ValueError("index_sma_window_int must be greater than 1.")
        if self.stock_sma_window_int <= 1:
            raise ValueError("stock_sma_window_int must be greater than 1.")
        if self.inverse_vol_window_int <= 1:
            raise ValueError("inverse_vol_window_int must be greater than 1.")
        if self.max_long_positions_int is not None and self.max_long_positions_int <= 0:
            raise ValueError("max_long_positions_int must be positive when set.")
        if self.capital_base_float <= 0.0:
            raise ValueError("capital_base_float must be positive.")
        if self.slippage_float < 0.0:
            raise ValueError("slippage_float must be non-negative.")
        if self.commission_per_share_float < 0.0:
            raise ValueError("commission_per_share_float must be non-negative.")
        if self.commission_minimum_float < 0.0:
            raise ValueError("commission_minimum_float must be non-negative.")


RUSSELL3000_LONG_SHORT_10X10_CONFIG = PtaWinnerContinuationConfig(
    variant_key_str="russell3000_long_short_10x10",
    indexname_str="Russell 3000",
    benchmark_symbol_str="$SPX",
)
RUSSELL3000_LONG_ONLY_10X10_RUA_SMA200_CONFIG = replace(
    RUSSELL3000_LONG_SHORT_10X10_CONFIG,
    variant_key_str="russell3000_long_only_10x10_rua_sma200",
    benchmark_symbol_str="$RUA",
    regime_symbol_str="$RUA",
    short_gross_exposure_float=0.0,
    index_sma_filter_enabled_bool=True,
)
RUSSELL3000_LONG_ONLY_10X10_RUA_SMA200_STOCK_SMA100_IVOL63_CONFIG = replace(
    RUSSELL3000_LONG_ONLY_10X10_RUA_SMA200_CONFIG,
    variant_key_str="russell3000_long_only_10x10_rua_sma200_stock_sma100_ivol63",
    stock_sma_filter_enabled_bool=True,
    stock_sma_window_int=STOCK_SMA_WINDOW_INT,
    inverse_vol_weighting_enabled_bool=True,
    inverse_vol_window_int=INVERSE_VOL_WINDOW_INT,
)
RUSSELL3000_LONG_ONLY_10X10_RUA_SMA200_TOP10_CONFIG = replace(
    RUSSELL3000_LONG_ONLY_10X10_RUA_SMA200_CONFIG,
    variant_key_str="russell3000_long_only_10x10_rua_sma200_top10",
    max_long_positions_int=10,
)
DEFAULT_CONFIG = RUSSELL3000_LONG_SHORT_10X10_CONFIG
CONFIG_BY_VARIANT_KEY_DICT = {
    RUSSELL3000_LONG_SHORT_10X10_CONFIG.variant_key_str: RUSSELL3000_LONG_SHORT_10X10_CONFIG,
    RUSSELL3000_LONG_ONLY_10X10_RUA_SMA200_CONFIG.variant_key_str: RUSSELL3000_LONG_ONLY_10X10_RUA_SMA200_CONFIG,
    RUSSELL3000_LONG_ONLY_10X10_RUA_SMA200_TOP10_CONFIG.variant_key_str: (
        RUSSELL3000_LONG_ONLY_10X10_RUA_SMA200_TOP10_CONFIG
    ),
    (
        RUSSELL3000_LONG_ONLY_10X10_RUA_SMA200_STOCK_SMA100_IVOL63_CONFIG.variant_key_str
    ): RUSSELL3000_LONG_ONLY_10X10_RUA_SMA200_STOCK_SMA100_IVOL63_CONFIG,
}


__all__ = [
    "BUCKET_COUNT_INT",
    "CONFIG_BY_VARIANT_KEY_DICT",
    "DEFAULT_CONFIG",
    "LONG_GROSS_EXPOSURE_FLOAT",
    "PTA_LOOKBACK_TRADING_DAY_INT",
    "PtaWinnerContinuationConfig",
    "PtaWinnerContinuationStrategy",
    "RUSSELL3000_LONG_ONLY_10X10_RUA_SMA200_CONFIG",
    "RUSSELL3000_LONG_ONLY_10X10_RUA_SMA200_STOCK_SMA100_IVOL63_CONFIG",
    "RUSSELL3000_LONG_ONLY_10X10_RUA_SMA200_TOP10_CONFIG",
    "RUSSELL3000_LONG_SHORT_10X10_CONFIG",
    "SHORT_GROSS_EXPOSURE_FLOAT",
    "STOCK_SMA_WINDOW_INT",
    "WINSOR_TAIL_PROB_FLOAT",
    "assign_quantile_bucket_ser",
    "compute_pta_winner_continuation_signal_tables",
    "get_pta_winner_continuation_data",
    "run_all_variants",
    "run_variant",
]


def assign_quantile_bucket_ser(
    value_ser: pd.Series,
    bucket_count_int: int,
) -> pd.Series:
    if bucket_count_int <= 1:
        raise ValueError("bucket_count_int must be greater than 1.")

    clean_value_ser = pd.to_numeric(value_ser, errors="coerce").dropna().astype(float)
    clean_value_ser = clean_value_ser[np.isfinite(clean_value_ser.to_numpy(dtype=float))]
    if len(clean_value_ser) == 0:
        return pd.Series(dtype="Int64")

    ordered_value_df = pd.DataFrame(
        {
            "value_float": clean_value_ser,
            "symbol_str": clean_value_ser.index.astype(str),
        },
        index=clean_value_ser.index,
    ).sort_values(
        by=["value_float", "symbol_str"],
        ascending=[True, True],
        kind="mergesort",
    )
    rank_vec = np.arange(1, len(ordered_value_df) + 1, dtype=float)
    bucket_vec = np.ceil(rank_vec * float(bucket_count_int) / float(len(ordered_value_df))).astype(int)
    bucket_vec = np.clip(bucket_vec, 1, bucket_count_int)
    return pd.Series(bucket_vec, index=ordered_value_df.index, dtype="Int64").sort_index()


def _winsorize_cross_section_df(
    value_df: pd.DataFrame,
    tail_prob_float: float,
) -> pd.DataFrame:
    if tail_prob_float <= 0.0:
        return value_df.copy()

    lower_bound_ser = value_df.quantile(tail_prob_float, axis=1)
    upper_bound_ser = value_df.quantile(1.0 - tail_prob_float, axis=1)
    return value_df.clip(lower=lower_bound_ser, upper=upper_bound_ser, axis=0)


def _cross_section_zscore_df(value_df: pd.DataFrame) -> pd.DataFrame:
    mean_ser = value_df.mean(axis=1, skipna=True)
    std_ser = value_df.std(axis=1, skipna=True, ddof=0)
    valid_std_ser = std_ser.where(std_ser > 0.0)
    return value_df.sub(mean_ser, axis=0).div(valid_std_ser, axis=0)


def compute_pta_winner_continuation_signal_tables(
    price_close_df: pd.DataFrame,
    config: PtaWinnerContinuationConfig = DEFAULT_CONFIG,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if len(price_close_df.index) == 0:
        raise ValueError("price_close_df must not be empty.")
    if not price_close_df.index.is_monotonic_increasing:
        raise ValueError("price_close_df index must be sorted.")
    if price_close_df.index.has_duplicates:
        raise ValueError("price_close_df index must not contain duplicates.")

    monthly_decision_close_df = get_monthly_decision_close_df(price_close_df=price_close_df)

    # *** CRITICAL*** PTA anchor uses only the trailing 252 trading-day window
    # ending at each daily close. A value sampled at month-end t is known only
    # after that close and is not allowed to rank the month-t winner return.
    rolling_high_df = price_close_df.rolling(
        window=config.pta_lookback_trading_day_int,
        min_periods=config.pta_lookback_trading_day_int,
    ).max()
    rolling_low_df = price_close_df.rolling(
        window=config.pta_lookback_trading_day_int,
        min_periods=config.pta_lookback_trading_day_int,
    ).min()
    pth_df = (price_close_df / rolling_high_df).replace([np.inf, -np.inf], np.nan)
    ptl_df = (price_close_df / rolling_low_df).replace([np.inf, -np.inf], np.nan)

    pth_decision_df = pth_df.reindex(monthly_decision_close_df.index)
    ptl_decision_df = ptl_df.reindex(monthly_decision_close_df.index)
    winsorized_pth_df = _winsorize_cross_section_df(
        value_df=pth_decision_df,
        tail_prob_float=config.winsor_tail_prob_float,
    )
    winsorized_ptl_df = _winsorize_cross_section_df(
        value_df=ptl_decision_df,
        tail_prob_float=config.winsor_tail_prob_float,
    )
    pta_df = (
        _cross_section_zscore_df(winsorized_pth_df)
        + _cross_section_zscore_df(winsorized_ptl_df)
    ) / 2.0

    # *** CRITICAL*** The paper signal is conditioned on prior PTA:
    # PTA_{t-1} ranks the stocks after month-t winners are identified.
    # Using PTA_t here would let the same month-t move influence both legs.
    prior_pta_df = pta_df.shift(1)

    # *** CRITICAL*** Month-t winner return uses actual month-end closes only.
    # The Vanilla trade generated from this value fills on the next tradable
    # open, not at the same month-end close.
    monthly_return_df = monthly_decision_close_df.pct_change(fill_method=None)

    valid_decision_mask_ser = monthly_return_df.notna().any(axis=1) & prior_pta_df.notna().any(axis=1)
    valid_decision_index = monthly_decision_close_df.index[valid_decision_mask_ser]
    return (
        monthly_decision_close_df.reindex(valid_decision_index),
        monthly_return_df.reindex(valid_decision_index),
        prior_pta_df.reindex(valid_decision_index),
    )


def compute_index_above_sma_filter_ser(
    index_close_ser: pd.Series,
    window_int: int,
) -> pd.Series:
    if len(index_close_ser.index) == 0:
        raise ValueError("index_close_ser must not be empty.")
    if not index_close_ser.index.is_monotonic_increasing:
        raise ValueError("index_close_ser index must be sorted.")
    if index_close_ser.index.has_duplicates:
        raise ValueError("index_close_ser index must not contain duplicates.")
    if window_int <= 1:
        raise ValueError("window_int must be greater than 1.")

    clean_index_close_ser = pd.Series(index_close_ser, copy=True).astype(float)
    # *** CRITICAL*** Regime gating uses only the trailing index SMA ending
    # at decision close T. Orders still fill at the next Vanilla open.
    index_sma_ser = clean_index_close_ser.rolling(
        window=window_int,
        min_periods=window_int,
    ).mean()
    return clean_index_close_ser.gt(index_sma_ser) & index_sma_ser.notna()


def compute_stock_above_sma_filter_df(
    price_close_df: pd.DataFrame,
    window_int: int,
) -> pd.DataFrame:
    if len(price_close_df.index) == 0:
        raise ValueError("price_close_df must not be empty.")
    if not price_close_df.index.is_monotonic_increasing:
        raise ValueError("price_close_df index must be sorted.")
    if price_close_df.index.has_duplicates:
        raise ValueError("price_close_df index must not contain duplicates.")
    if window_int <= 1:
        raise ValueError("window_int must be greater than 1.")

    # *** CRITICAL*** Stock eligibility uses only each stock's trailing SMA
    # ending at decision close T. Names below SMA100 are removed before the
    # winner/PTA buckets are formed for next-open Vanilla execution.
    stock_sma_df = price_close_df.rolling(
        window=window_int,
        min_periods=window_int,
    ).mean()
    return price_close_df.gt(stock_sma_df) & stock_sma_df.notna()


def compute_trailing_vol_df(
    price_close_df: pd.DataFrame,
    window_int: int,
) -> pd.DataFrame:
    if len(price_close_df.index) == 0:
        raise ValueError("price_close_df must not be empty.")
    if not price_close_df.index.is_monotonic_increasing:
        raise ValueError("price_close_df index must be sorted.")
    if price_close_df.index.has_duplicates:
        raise ValueError("price_close_df index must not contain duplicates.")
    if window_int <= 1:
        raise ValueError("window_int must be greater than 1.")

    # *** CRITICAL*** Vol63 sizing uses trailing close-to-close returns only.
    # No current open or future returns can influence the inverse-vol weights.
    daily_return_df = price_close_df.pct_change(fill_method=None)
    trailing_vol_df = daily_return_df.rolling(
        window=window_int,
        min_periods=window_int,
    ).std()
    return trailing_vol_df.replace([np.inf, -np.inf], np.nan)


def get_pta_winner_continuation_data(
    config: PtaWinnerContinuationConfig = DEFAULT_CONFIG,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    _, raw_universe_df = build_index_constituent_matrix(indexname=config.indexname_str)

    history_start_ts = pd.Timestamp(config.history_start_date_str)
    backtest_start_ts = pd.Timestamp(config.backtest_start_date_str)
    filtered_universe_df = raw_universe_df.loc[raw_universe_df.index >= history_start_ts].copy()
    active_universe_df = filtered_universe_df.loc[filtered_universe_df.index >= backtest_start_ts].copy()
    if config.end_date_str is not None:
        active_universe_df = active_universe_df.loc[active_universe_df.index <= pd.Timestamp(config.end_date_str)]

    active_symbol_list = active_universe_df.columns[
        active_universe_df.sum(axis=0) > 0
    ].astype(str).tolist()
    if len(active_symbol_list) == 0:
        raise RuntimeError(f"No active {config.indexname_str} symbols were found for the requested window.")

    benchmark_symbol_list = list(
        dict.fromkeys(
            [
                symbol_str
                for symbol_str in [config.benchmark_symbol_str, config.regime_symbol_str]
                if symbol_str is not None
            ]
        )
    )
    pricing_data_df = load_raw_prices(
        symbols=active_symbol_list,
        benchmarks=benchmark_symbol_list,
        start_date=config.history_start_date_str,
        end_date=config.end_date_str,
    )
    loaded_symbol_list = [
        symbol_str
        for symbol_str in active_symbol_list
        if symbol_str in pricing_data_df.columns.get_level_values(0)
    ]
    audited_universe_df = audit_pit_universe_df(
        universe_df=filtered_universe_df,
        execution_index=pricing_data_df.index,
        tradeable_symbol_list=loaded_symbol_list,
    )

    keep_symbol_set = set(audited_universe_df.columns.astype(str).tolist() + benchmark_symbol_list)
    pricing_data_df = pricing_data_df.loc[
        :,
        pricing_data_df.columns.get_level_values(0).isin(keep_symbol_set),
    ].sort_index()

    close_symbol_list = audited_universe_df.columns.astype(str).tolist()
    price_close_df = pd.DataFrame(
        {symbol_str: pricing_data_df[(symbol_str, "Close")] for symbol_str in close_symbol_list},
        index=pricing_data_df.index,
    ).astype(float)
    monthly_decision_close_df, _monthly_return_df, _prior_pta_df = (
        compute_pta_winner_continuation_signal_tables(
            price_close_df=price_close_df,
            config=config,
        )
    )
    rebalance_schedule_df = map_month_end_decision_dates_to_rebalance_schedule_df(
        decision_date_index=pd.DatetimeIndex(monthly_decision_close_df.index),
        execution_index=pricing_data_df.index,
    )
    return pricing_data_df, audited_universe_df, rebalance_schedule_df


class PtaWinnerContinuationStrategy(Strategy):
    """Research-only long-short monthly PTA-conditioned winner strategy."""

    enable_signal_audit = True
    signal_audit_sample_size = 10

    def __init__(
        self,
        name: str,
        benchmarks: Sequence[str],
        rebalance_schedule_df: pd.DataFrame,
        config: PtaWinnerContinuationConfig,
    ):
        super().__init__(
            name=name,
            benchmarks=list(benchmarks),
            capital_base=config.capital_base_float,
            slippage=config.slippage_float,
            commission_per_share=config.commission_per_share_float,
            commission_minimum=config.commission_minimum_float,
        )

        if len(rebalance_schedule_df) == 0:
            raise ValueError("rebalance_schedule_df must not be empty.")
        if "decision_date_ts" not in rebalance_schedule_df.columns:
            raise ValueError("rebalance_schedule_df must contain decision_date_ts.")
        if config.benchmark_symbol_str not in benchmarks:
            raise ValueError("benchmarks must include config.benchmark_symbol_str.")

        self.rebalance_schedule_df = rebalance_schedule_df.copy().sort_index()
        self.config = config
        self.trade_id_int = 0
        self.current_trade_map: defaultdict[str, int] = defaultdict(default_trade_id_int)
        self.universe_df: pd.DataFrame | None = None

    @property
    def monthly_return_field_str(self) -> str:
        return "pta_monthly_return_float"

    @property
    def prior_pta_field_str(self) -> str:
        return f"pta_prior_{self.config.pta_lookback_trading_day_int}_float"

    @property
    def index_sma_pass_field_str(self) -> str:
        return f"index_close_above_sma_{self.config.index_sma_window_int}_bool"

    @property
    def stock_sma_pass_field_str(self) -> str:
        return f"stock_close_above_sma_{self.config.stock_sma_window_int}_bool"

    @property
    def inverse_vol_field_str(self) -> str:
        return f"trailing_vol_{self.config.inverse_vol_window_int}_float"

    def compute_signals(self, pricing_data: pd.DataFrame) -> pd.DataFrame:
        signal_data_df = pricing_data.copy()
        benchmark_symbol_set = {str(symbol_str) for symbol_str in self._benchmarks}
        tradeable_symbol_list = [
            str(symbol_str)
            for symbol_str in signal_data_df.columns.get_level_values(0).unique()
            if str(symbol_str) not in benchmark_symbol_set
        ]
        if len(tradeable_symbol_list) == 0:
            raise RuntimeError("No tradeable stock symbols were found in pricing_data.")

        price_close_df = pd.DataFrame(
            {symbol_str: signal_data_df[(symbol_str, "Close")] for symbol_str in tradeable_symbol_list},
            index=signal_data_df.index,
        ).astype(float)
        _monthly_decision_close_df, monthly_return_df, prior_pta_df = (
            compute_pta_winner_continuation_signal_tables(
                price_close_df=price_close_df,
                config=self.config,
            )
        )

        feature_frame_list: list[pd.DataFrame] = []
        feature_map: dict[str, pd.DataFrame] = {
            self.monthly_return_field_str: monthly_return_df.reindex(signal_data_df.index),
            self.prior_pta_field_str: prior_pta_df.reindex(signal_data_df.index),
        }
        if self.config.stock_sma_filter_enabled_bool:
            stock_sma_pass_df = compute_stock_above_sma_filter_df(
                price_close_df=price_close_df,
                window_int=self.config.stock_sma_window_int,
            )
            feature_map[self.stock_sma_pass_field_str] = stock_sma_pass_df.reindex(signal_data_df.index)
        if self.config.inverse_vol_weighting_enabled_bool:
            trailing_vol_df = compute_trailing_vol_df(
                price_close_df=price_close_df,
                window_int=self.config.inverse_vol_window_int,
            )
            feature_map[self.inverse_vol_field_str] = trailing_vol_df.reindex(signal_data_df.index)
        for field_str, field_df in feature_map.items():
            feature_df = field_df.copy()
            feature_df.columns = pd.MultiIndex.from_tuples(
                [(symbol_str, field_str) for symbol_str in feature_df.columns]
            )
            feature_frame_list.append(feature_df)

        if self.config.index_sma_filter_enabled_bool:
            regime_symbol_str = str(self.config.regime_symbol_str)
            regime_close_key = (regime_symbol_str, "Close")
            if regime_close_key not in signal_data_df.columns:
                raise RuntimeError(f"Missing regime close data for {regime_symbol_str}.")
            index_sma_pass_ser = compute_index_above_sma_filter_ser(
                index_close_ser=signal_data_df[regime_close_key].astype(float),
                window_int=self.config.index_sma_window_int,
            ).reindex(signal_data_df.index)
            regime_feature_df = pd.DataFrame(
                {
                    (regime_symbol_str, self.index_sma_pass_field_str): index_sma_pass_ser,
                },
                index=signal_data_df.index,
            )
            regime_feature_df.columns = pd.MultiIndex.from_tuples(regime_feature_df.columns)
            feature_frame_list.append(regime_feature_df)

        return pd.concat([signal_data_df] + feature_frame_list, axis=1)

    def is_index_sma_filter_pass_bool(self, close_row_ser: pd.Series) -> bool:
        if not self.config.index_sma_filter_enabled_bool:
            return True

        regime_symbol_str = str(self.config.regime_symbol_str)
        field_tuple = (regime_symbol_str, self.index_sma_pass_field_str)
        if field_tuple not in close_row_ser.index:
            return False
        value_obj = close_row_ser.loc[field_tuple]
        if pd.isna(value_obj):
            return False
        return bool(value_obj)

    def get_target_weight_ser(self, close_row_ser: pd.Series) -> pd.Series:
        if self.universe_df is None:
            raise RuntimeError("universe_df must be set before rebalances.")

        if not self.is_index_sma_filter_pass_bool(close_row_ser=close_row_ser):
            return pd.Series(dtype=float)

        candidate_feature_df = close_row_ser.unstack()
        required_field_list = [self.monthly_return_field_str, self.prior_pta_field_str]
        if self.config.stock_sma_filter_enabled_bool:
            required_field_list.append(self.stock_sma_pass_field_str)
        if self.config.inverse_vol_weighting_enabled_bool:
            required_field_list.append(self.inverse_vol_field_str)
        if any(field_str not in candidate_feature_df.columns for field_str in required_field_list):
            return pd.Series(dtype=float)

        universe_member_ser = get_asof_universe_membership_ser(
            self.universe_df,
            pd.Timestamp(self.previous_bar),
        )
        active_symbol_list = universe_member_ser[universe_member_ser == 1].index.astype(str).tolist()
        candidate_feature_df = candidate_feature_df[candidate_feature_df.index.isin(active_symbol_list)].copy()
        if len(candidate_feature_df) == 0:
            return pd.Series(dtype=float)

        candidate_feature_df = candidate_feature_df.assign(
            monthly_return_float=pd.to_numeric(
                candidate_feature_df[self.monthly_return_field_str],
                errors="coerce",
            ),
            prior_pta_float=pd.to_numeric(
                candidate_feature_df[self.prior_pta_field_str],
                errors="coerce",
            ),
        )
        finite_mask_vec = (
            np.isfinite(candidate_feature_df["monthly_return_float"].to_numpy(dtype=float))
            & np.isfinite(candidate_feature_df["prior_pta_float"].to_numpy(dtype=float))
        )
        candidate_feature_df = candidate_feature_df.loc[finite_mask_vec].copy()
        if len(candidate_feature_df) == 0:
            return pd.Series(dtype=float)

        if self.config.stock_sma_filter_enabled_bool:
            stock_sma_pass_ser = pd.Series(
                candidate_feature_df[self.stock_sma_pass_field_str],
                index=candidate_feature_df.index,
                dtype="boolean",
            ).fillna(False).astype(bool)
            candidate_feature_df = candidate_feature_df.loc[stock_sma_pass_ser].copy()
            if len(candidate_feature_df) == 0:
                return pd.Series(dtype=float)

        return_bucket_ser = assign_quantile_bucket_ser(
            value_ser=candidate_feature_df["monthly_return_float"],
            bucket_count_int=self.config.bucket_count_int,
        )
        pta_bucket_ser = assign_quantile_bucket_ser(
            value_ser=candidate_feature_df["prior_pta_float"],
            bucket_count_int=self.config.bucket_count_int,
        )
        candidate_feature_df = candidate_feature_df.assign(
            return_bucket_int=return_bucket_ser,
            pta_bucket_int=pta_bucket_ser,
        )
        long_symbol_list = candidate_feature_df.index[
            (candidate_feature_df["return_bucket_int"] == self.config.winner_return_bucket_int)
            & (candidate_feature_df["pta_bucket_int"] == self.config.long_pta_bucket_int)
        ].astype(str).tolist()
        short_symbol_list = candidate_feature_df.index[
            (candidate_feature_df["return_bucket_int"] == self.config.winner_return_bucket_int)
            & (candidate_feature_df["pta_bucket_int"] == self.config.short_pta_bucket_int)
        ].astype(str).tolist()
        if (
            self.config.max_long_positions_int is not None
            and len(long_symbol_list) > self.config.max_long_positions_int
        ):
            capped_long_feature_df = candidate_feature_df.loc[long_symbol_list].copy()
            capped_long_feature_df = capped_long_feature_df.assign(
                symbol_str=capped_long_feature_df.index.astype(str)
            ).sort_values(
                by=["prior_pta_float", "monthly_return_float", "symbol_str"],
                ascending=[False, False, True],
                kind="mergesort",
            )
            long_symbol_list = (
                capped_long_feature_df.head(self.config.max_long_positions_int).index.astype(str).tolist()
            )

        target_weight_map: dict[str, float] = {}
        if len(long_symbol_list) > 0 and self.config.long_gross_exposure_float > 0.0:
            if self.config.inverse_vol_weighting_enabled_bool:
                long_vol_ser = pd.to_numeric(
                    candidate_feature_df.loc[long_symbol_list, self.inverse_vol_field_str],
                    errors="coerce",
                ).astype(float)
                finite_positive_vol_ser = long_vol_ser[
                    np.isfinite(long_vol_ser.to_numpy(dtype=float)) & (long_vol_ser > 0.0)
                ]
                inverse_vol_ser = 1.0 / finite_positive_vol_ser
                inverse_vol_sum_float = float(inverse_vol_ser.sum())
                if inverse_vol_sum_float > 0.0:
                    long_weight_ser = (
                        inverse_vol_ser / inverse_vol_sum_float
                    ) * self.config.long_gross_exposure_float
                    for symbol_str, long_weight_float in long_weight_ser.items():
                        target_weight_map[str(symbol_str)] = float(long_weight_float)
            else:
                long_weight_float = self.config.long_gross_exposure_float / float(len(long_symbol_list))
                for symbol_str in long_symbol_list:
                    target_weight_map[symbol_str] = long_weight_float
        if len(short_symbol_list) > 0 and self.config.short_gross_exposure_float > 0.0:
            short_weight_float = -self.config.short_gross_exposure_float / float(len(short_symbol_list))
            for symbol_str in short_symbol_list:
                target_weight_map[symbol_str] = short_weight_float

        return pd.Series(target_weight_map, dtype=float).sort_index()

    def iterate(self, data: pd.DataFrame, close: pd.Series, open_prices: pd.Series):
        if close is None or data is None:
            return
        if self.current_bar not in self.rebalance_schedule_df.index:
            return

        decision_date_ts = pd.Timestamp(self.rebalance_schedule_df.loc[self.current_bar, "decision_date_ts"])
        # *** CRITICAL*** Monthly PTA/winner intent must be formed from the
        # previous completed month-end close. Vanilla then fills at current_bar
        # open; any mismatch here changes signal and execution timing.
        if pd.Timestamp(self.previous_bar) != decision_date_ts:
            raise RuntimeError(
                f"Schedule misalignment on {self.current_bar}: "
                f"decision_date_ts={decision_date_ts}, previous_bar={self.previous_bar}."
            )

        target_weight_ser = self.get_target_weight_ser(close_row_ser=close)
        target_symbol_set = set(target_weight_ser.index.astype(str).tolist())
        current_position_ser = self.get_positions()
        active_position_ser = current_position_ser[current_position_ser != 0]

        for symbol_str in active_position_ser.index.astype(str):
            if symbol_str in target_symbol_set:
                continue
            self.order_target_value(
                symbol_str,
                0.0,
                trade_id=self.current_trade_map[symbol_str],
            )

        for symbol_str, target_weight_float in target_weight_ser.items():
            current_share_float = float(current_position_ser.get(symbol_str, 0.0))
            if current_share_float == 0.0:
                self.trade_id_int += 1
                self.current_trade_map[symbol_str] = self.trade_id_int

            self.order_target_percent(
                symbol_str,
                float(target_weight_float),
                trade_id=self.current_trade_map[symbol_str],
            )


def _with_run_overrides(
    config: PtaWinnerContinuationConfig,
    backtest_start_date_str: str | None = None,
    capital_base_float: float | None = None,
    end_date_str: str | None = None,
) -> PtaWinnerContinuationConfig:
    if backtest_start_date_str is None and capital_base_float is None and end_date_str is None:
        return config

    return replace(
        config,
        backtest_start_date_str=(
            config.backtest_start_date_str if backtest_start_date_str is None else backtest_start_date_str
        ),
        capital_base_float=(
            config.capital_base_float if capital_base_float is None else float(capital_base_float)
        ),
        end_date_str=end_date_str,
    )


def run_variant(
    variant_key_str: str = DEFAULT_CONFIG.variant_key_str,
    show_display_bool: bool = True,
    save_results_bool: bool = True,
    output_dir_str: str = "results",
    backtest_start_date_str: str | None = None,
    capital_base_float: float | None = None,
    end_date_str: str | None = None,
    audit_override_bool: bool | None = None,
) -> PtaWinnerContinuationStrategy:
    if variant_key_str not in CONFIG_BY_VARIANT_KEY_DICT:
        raise ValueError(f"Unknown variant_key_str: {variant_key_str}")

    config_obj = _with_run_overrides(
        config=CONFIG_BY_VARIANT_KEY_DICT[variant_key_str],
        backtest_start_date_str=backtest_start_date_str,
        capital_base_float=capital_base_float,
        end_date_str=end_date_str,
    )
    pricing_data_df, universe_df, rebalance_schedule_df = get_pta_winner_continuation_data(config=config_obj)
    benchmark_symbol_list = list(
        dict.fromkeys(
            [
                symbol_str
                for symbol_str in [config_obj.benchmark_symbol_str, config_obj.regime_symbol_str]
                if symbol_str is not None
            ]
        )
    )
    strategy_obj = PtaWinnerContinuationStrategy(
        name=f"strategy_mo_pta_winner_continuation_{config_obj.variant_key_str}",
        benchmarks=benchmark_symbol_list,
        rebalance_schedule_df=rebalance_schedule_df,
        config=config_obj,
    )
    strategy_obj.universe_df = universe_df

    # *** CRITICAL*** Keep pre-start price history for 252-day PTA and prior
    # month anchor formation, but execute only from backtest_start_date_str.
    calendar_idx = pricing_data_df.index[
        pricing_data_df.index >= pd.Timestamp(config_obj.backtest_start_date_str)
    ]
    run_daily(
        strategy_obj,
        pricing_data_df,
        calendar=calendar_idx,
        show_progress=show_display_bool,
        show_signal_progress_bool=show_display_bool,
        audit_override_bool=audit_override_bool,
    )

    if show_display_bool:
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 1000)
        display(strategy_obj.summary)
        display(strategy_obj.summary_trades)

    if save_results_bool:
        save_results(strategy_obj, output_dir=output_dir_str)

    return strategy_obj


def run_all_variants(
    show_display_bool: bool = True,
    save_results_bool: bool = True,
    output_dir_str: str = "results",
    backtest_start_date_str: str | None = None,
    capital_base_float: float | None = None,
    end_date_str: str | None = None,
    audit_override_bool: bool | None = None,
) -> dict[str, PtaWinnerContinuationStrategy]:
    strategy_dict: dict[str, PtaWinnerContinuationStrategy] = {}
    for variant_key_str in CONFIG_BY_VARIANT_KEY_DICT:
        strategy_dict[variant_key_str] = run_variant(
            variant_key_str=variant_key_str,
            show_display_bool=show_display_bool,
            save_results_bool=save_results_bool,
            output_dir_str=output_dir_str,
            backtest_start_date_str=backtest_start_date_str,
            capital_base_float=capital_base_float,
            end_date_str=end_date_str,
            audit_override_bool=audit_override_bool,
        )
    return strategy_dict
