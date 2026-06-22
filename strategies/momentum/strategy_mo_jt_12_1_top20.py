"""
Monthly 12-1 cross-sectional momentum selector for point-in-time index members.

Core formula
------------
For stock i on month-end decision date t:

    momentum_12_1_{i,t}
        = Close_{i,t-21} / Close_{i,t-252} - 1

Selection on decision date t:

    eligible_{i,t}
        = 1[PIT_INDEX_{i,t} = 1 and momentum_12_1_{i,t} is finite]

Optional stock-level SMA100 filter:

    sma100_{i,t}
        = mean(Close_{i,t-99}, ..., Close_{i,t})

    eligible_{i,t}
        = eligible_{i,t} and 1[Close_{i,t} > sma100_{i,t}]

Optional benchmark-index SMA200 regime filter:

    index_sma200_t
        = mean(IndexClose_{t-199}, ..., IndexClose_t)

    selected_t
        = empty set if IndexClose_t <= index_sma200_t

    selected_t
        = top N eligible symbols by momentum_12_1_{i,t}

    target_weight_{i,t}
        = 1 / |selected_t| if i in selected_t
        = 0                otherwise

Optional volatility-targeted overlay:

    realized_vol_t
        = stdev(strategy_return_{t-125:t}) * sqrt(252)

    gross_exposure_t
        = min(1.0, target_vol / realized_vol_t)

    target_weight_{i,t}
        = gross_exposure_t / |selected_t| if i in selected_t
        = 0                              otherwise

Execution mapping:

    decision_date_t
        = actual last tradable close of month t

    execution_date_t
        = next tradable open after decision_date_t
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


LOOKBACK_TRADING_DAY_INT = 252
SKIP_TRADING_DAY_INT = 21
MAX_POSITIONS_INT = 20
REALIZED_VOL_WINDOW_INT = 126
TARGET_ANNUAL_VOLATILITY_FLOAT = 0.12
MAX_GROSS_EXPOSURE_FLOAT = 1.0
STOCK_SMA_FILTER_WINDOW_INT = 100
INDEX_SMA_FILTER_WINDOW_INT = 200
RAW_12_1_RANKING_METHOD_STR = "raw_12_1"
VOL_NORMALIZED_12_1_RANKING_METHOD_STR = "vol_normalized_12_1"
TREND_QUALITY_RANKING_METHOD_STR = "trend_quality"
MULTI_HORIZON_Z_RANKING_METHOD_STR = "multi_horizon_z"
RESIDUAL_12_1_RANKING_METHOD_STR = "residual_12_1"
PN_EV_MULTI_WINDOW_Z_RANKING_METHOD_STR = "pn_ev_multi_window_z"
PN_LRB_MULTI_WINDOW_Z_RANKING_METHOD_STR = "pn_lrb_multi_window_z"
RANKING_METHOD_SET = {
    RAW_12_1_RANKING_METHOD_STR,
    VOL_NORMALIZED_12_1_RANKING_METHOD_STR,
    TREND_QUALITY_RANKING_METHOD_STR,
    MULTI_HORIZON_Z_RANKING_METHOD_STR,
    RESIDUAL_12_1_RANKING_METHOD_STR,
    PN_EV_MULTI_WINDOW_Z_RANKING_METHOD_STR,
    PN_LRB_MULTI_WINDOW_Z_RANKING_METHOD_STR,
}
TREND_QUALITY_WINDOW_INT = 126
VOL_NORMALIZATION_WINDOW_INT = LOOKBACK_TRADING_DAY_INT - SKIP_TRADING_DAY_INT
PN_RANKING_WINDOW_TUPLE = (21, 63, 128, 252)


def default_trade_id_int() -> int:
    return -1


@dataclass(frozen=True)
class Jt121Top20Config:
    variant_key_str: str
    indexname_str: str
    benchmark_symbol_str: str
    history_start_date_str: str = "1998-01-01"
    backtest_start_date_str: str = "2000-01-01"
    end_date_str: str | None = None
    lookback_trading_day_int: int = LOOKBACK_TRADING_DAY_INT
    skip_trading_day_int: int = SKIP_TRADING_DAY_INT
    max_positions_int: int = MAX_POSITIONS_INT
    volatility_target_enabled_bool: bool = False
    target_annual_volatility_float: float = TARGET_ANNUAL_VOLATILITY_FLOAT
    realized_vol_window_int: int = REALIZED_VOL_WINDOW_INT
    max_gross_exposure_float: float = MAX_GROSS_EXPOSURE_FLOAT
    stock_sma_filter_enabled_bool: bool = False
    stock_sma_window_int: int = STOCK_SMA_FILTER_WINDOW_INT
    index_sma_filter_enabled_bool: bool = False
    index_sma_window_int: int = INDEX_SMA_FILTER_WINDOW_INT
    ranking_method_str: str = RAW_12_1_RANKING_METHOD_STR
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
        if self.lookback_trading_day_int <= 1:
            raise ValueError("lookback_trading_day_int must be greater than 1.")
        if self.skip_trading_day_int <= 0:
            raise ValueError("skip_trading_day_int must be positive.")
        if self.skip_trading_day_int >= self.lookback_trading_day_int:
            raise ValueError("skip_trading_day_int must be less than lookback_trading_day_int.")
        if self.max_positions_int <= 0:
            raise ValueError("max_positions_int must be positive.")
        if self.target_annual_volatility_float <= 0.0:
            raise ValueError("target_annual_volatility_float must be positive.")
        if self.realized_vol_window_int <= 1:
            raise ValueError("realized_vol_window_int must be greater than 1.")
        if self.max_gross_exposure_float <= 0.0:
            raise ValueError("max_gross_exposure_float must be positive.")
        if self.stock_sma_window_int <= 1:
            raise ValueError("stock_sma_window_int must be greater than 1.")
        if self.index_sma_window_int <= 1:
            raise ValueError("index_sma_window_int must be greater than 1.")
        if self.ranking_method_str not in RANKING_METHOD_SET:
            raise ValueError(f"Unknown ranking_method_str: {self.ranking_method_str}")
        if self.capital_base_float <= 0.0:
            raise ValueError("capital_base_float must be positive.")
        if self.slippage_float < 0.0:
            raise ValueError("slippage_float must be non-negative.")
        if self.commission_per_share_float < 0.0:
            raise ValueError("commission_per_share_float must be non-negative.")
        if self.commission_minimum_float < 0.0:
            raise ValueError("commission_minimum_float must be non-negative.")


SP500_CONFIG = Jt121Top20Config(
    variant_key_str="sp500",
    indexname_str="S&P 500",
    benchmark_symbol_str="$SPX",
)
NASDAQ100_CONFIG = Jt121Top20Config(
    variant_key_str="nasdaq100",
    indexname_str="Nasdaq 100",
    benchmark_symbol_str="$NDX",
)
RUSSELL1000_CONFIG = Jt121Top20Config(
    variant_key_str="russell1000",
    indexname_str="Russell 1000",
    benchmark_symbol_str="$RUI",
)
SP500_VOL_TARGET_CONFIG = replace(
    SP500_CONFIG,
    variant_key_str="sp500_vol_target",
    volatility_target_enabled_bool=True,
)
NASDAQ100_VOL_TARGET_CONFIG = replace(
    NASDAQ100_CONFIG,
    variant_key_str="nasdaq100_vol_target",
    volatility_target_enabled_bool=True,
)
RUSSELL1000_VOL_TARGET_CONFIG = replace(
    RUSSELL1000_CONFIG,
    variant_key_str="russell1000_vol_target",
    volatility_target_enabled_bool=True,
)
SP500_SMA100_CONFIG = replace(
    SP500_CONFIG,
    variant_key_str="sp500_sma100",
    stock_sma_filter_enabled_bool=True,
)
NASDAQ100_SMA100_CONFIG = replace(
    NASDAQ100_CONFIG,
    variant_key_str="nasdaq100_sma100",
    stock_sma_filter_enabled_bool=True,
)
RUSSELL1000_SMA100_CONFIG = replace(
    RUSSELL1000_CONFIG,
    variant_key_str="russell1000_sma100",
    stock_sma_filter_enabled_bool=True,
)
SP500_SMA100_VOL_TARGET_CONFIG = replace(
    SP500_VOL_TARGET_CONFIG,
    variant_key_str="sp500_sma100_vol_target",
    stock_sma_filter_enabled_bool=True,
)
NASDAQ100_SMA100_VOL_TARGET_CONFIG = replace(
    NASDAQ100_VOL_TARGET_CONFIG,
    variant_key_str="nasdaq100_sma100_vol_target",
    stock_sma_filter_enabled_bool=True,
)
RUSSELL1000_SMA100_VOL_TARGET_CONFIG = replace(
    RUSSELL1000_VOL_TARGET_CONFIG,
    variant_key_str="russell1000_sma100_vol_target",
    stock_sma_filter_enabled_bool=True,
)
SP500_SMA100_INDEX_SMA200_CONFIG = replace(
    SP500_SMA100_CONFIG,
    variant_key_str="sp500_sma100_index_sma200",
    index_sma_filter_enabled_bool=True,
)
NASDAQ100_SMA100_INDEX_SMA200_CONFIG = replace(
    NASDAQ100_SMA100_CONFIG,
    variant_key_str="nasdaq100_sma100_index_sma200",
    index_sma_filter_enabled_bool=True,
)
RUSSELL1000_SMA100_INDEX_SMA200_CONFIG = replace(
    RUSSELL1000_SMA100_CONFIG,
    variant_key_str="russell1000_sma100_index_sma200",
    index_sma_filter_enabled_bool=True,
)
SP500_SMA100_INDEX_SMA200_VOL_TARGET_CONFIG = replace(
    SP500_SMA100_VOL_TARGET_CONFIG,
    variant_key_str="sp500_sma100_index_sma200_vol_target",
    index_sma_filter_enabled_bool=True,
)
NASDAQ100_SMA100_INDEX_SMA200_VOL_TARGET_CONFIG = replace(
    NASDAQ100_SMA100_VOL_TARGET_CONFIG,
    variant_key_str="nasdaq100_sma100_index_sma200_vol_target",
    index_sma_filter_enabled_bool=True,
)
RUSSELL1000_SMA100_INDEX_SMA200_VOL_TARGET_CONFIG = replace(
    RUSSELL1000_SMA100_VOL_TARGET_CONFIG,
    variant_key_str="russell1000_sma100_index_sma200_vol_target",
    index_sma_filter_enabled_bool=True,
)
SP500_SMA100_INDEX_SMA200_VOLNORM_CONFIG = replace(
    SP500_SMA100_INDEX_SMA200_CONFIG,
    variant_key_str="sp500_sma100_index_sma200_volnorm",
    ranking_method_str=VOL_NORMALIZED_12_1_RANKING_METHOD_STR,
)
NASDAQ100_SMA100_INDEX_SMA200_VOLNORM_CONFIG = replace(
    NASDAQ100_SMA100_INDEX_SMA200_CONFIG,
    variant_key_str="nasdaq100_sma100_index_sma200_volnorm",
    ranking_method_str=VOL_NORMALIZED_12_1_RANKING_METHOD_STR,
)
RUSSELL1000_SMA100_INDEX_SMA200_VOLNORM_CONFIG = replace(
    RUSSELL1000_SMA100_INDEX_SMA200_CONFIG,
    variant_key_str="russell1000_sma100_index_sma200_volnorm",
    ranking_method_str=VOL_NORMALIZED_12_1_RANKING_METHOD_STR,
)
SP500_SMA100_INDEX_SMA200_TREND_QUALITY_CONFIG = replace(
    SP500_SMA100_INDEX_SMA200_CONFIG,
    variant_key_str="sp500_sma100_index_sma200_trend_quality",
    ranking_method_str=TREND_QUALITY_RANKING_METHOD_STR,
)
NASDAQ100_SMA100_INDEX_SMA200_TREND_QUALITY_CONFIG = replace(
    NASDAQ100_SMA100_INDEX_SMA200_CONFIG,
    variant_key_str="nasdaq100_sma100_index_sma200_trend_quality",
    ranking_method_str=TREND_QUALITY_RANKING_METHOD_STR,
)
RUSSELL1000_SMA100_INDEX_SMA200_TREND_QUALITY_CONFIG = replace(
    RUSSELL1000_SMA100_INDEX_SMA200_CONFIG,
    variant_key_str="russell1000_sma100_index_sma200_trend_quality",
    ranking_method_str=TREND_QUALITY_RANKING_METHOD_STR,
)
SP500_SMA100_INDEX_SMA200_MULTI_HORIZON_CONFIG = replace(
    SP500_SMA100_INDEX_SMA200_CONFIG,
    variant_key_str="sp500_sma100_index_sma200_multi_horizon",
    ranking_method_str=MULTI_HORIZON_Z_RANKING_METHOD_STR,
)
NASDAQ100_SMA100_INDEX_SMA200_MULTI_HORIZON_CONFIG = replace(
    NASDAQ100_SMA100_INDEX_SMA200_CONFIG,
    variant_key_str="nasdaq100_sma100_index_sma200_multi_horizon",
    ranking_method_str=MULTI_HORIZON_Z_RANKING_METHOD_STR,
)
RUSSELL1000_SMA100_INDEX_SMA200_MULTI_HORIZON_CONFIG = replace(
    RUSSELL1000_SMA100_INDEX_SMA200_CONFIG,
    variant_key_str="russell1000_sma100_index_sma200_multi_horizon",
    ranking_method_str=MULTI_HORIZON_Z_RANKING_METHOD_STR,
)
SP500_SMA100_INDEX_SMA200_RESIDUAL_CONFIG = replace(
    SP500_SMA100_INDEX_SMA200_CONFIG,
    variant_key_str="sp500_sma100_index_sma200_residual",
    ranking_method_str=RESIDUAL_12_1_RANKING_METHOD_STR,
)
NASDAQ100_SMA100_INDEX_SMA200_RESIDUAL_CONFIG = replace(
    NASDAQ100_SMA100_INDEX_SMA200_CONFIG,
    variant_key_str="nasdaq100_sma100_index_sma200_residual",
    ranking_method_str=RESIDUAL_12_1_RANKING_METHOD_STR,
)
RUSSELL1000_SMA100_INDEX_SMA200_RESIDUAL_CONFIG = replace(
    RUSSELL1000_SMA100_INDEX_SMA200_CONFIG,
    variant_key_str="russell1000_sma100_index_sma200_residual",
    ranking_method_str=RESIDUAL_12_1_RANKING_METHOD_STR,
)
SP500_SMA100_INDEX_SMA200_PN_EV_CONFIG = replace(
    SP500_SMA100_INDEX_SMA200_CONFIG,
    variant_key_str="sp500_sma100_index_sma200_pn_ev",
    ranking_method_str=PN_EV_MULTI_WINDOW_Z_RANKING_METHOD_STR,
)
NASDAQ100_SMA100_INDEX_SMA200_PN_EV_CONFIG = replace(
    NASDAQ100_SMA100_INDEX_SMA200_CONFIG,
    variant_key_str="nasdaq100_sma100_index_sma200_pn_ev",
    ranking_method_str=PN_EV_MULTI_WINDOW_Z_RANKING_METHOD_STR,
)
RUSSELL1000_SMA100_INDEX_SMA200_PN_EV_CONFIG = replace(
    RUSSELL1000_SMA100_INDEX_SMA200_CONFIG,
    variant_key_str="russell1000_sma100_index_sma200_pn_ev",
    ranking_method_str=PN_EV_MULTI_WINDOW_Z_RANKING_METHOD_STR,
)
SP500_SMA100_INDEX_SMA200_PN_LRB_CONFIG = replace(
    SP500_SMA100_INDEX_SMA200_CONFIG,
    variant_key_str="sp500_sma100_index_sma200_pn_lrb",
    ranking_method_str=PN_LRB_MULTI_WINDOW_Z_RANKING_METHOD_STR,
)
NASDAQ100_SMA100_INDEX_SMA200_PN_LRB_CONFIG = replace(
    NASDAQ100_SMA100_INDEX_SMA200_CONFIG,
    variant_key_str="nasdaq100_sma100_index_sma200_pn_lrb",
    ranking_method_str=PN_LRB_MULTI_WINDOW_Z_RANKING_METHOD_STR,
)
RUSSELL1000_SMA100_INDEX_SMA200_PN_LRB_CONFIG = replace(
    RUSSELL1000_SMA100_INDEX_SMA200_CONFIG,
    variant_key_str="russell1000_sma100_index_sma200_pn_lrb",
    ranking_method_str=PN_LRB_MULTI_WINDOW_Z_RANKING_METHOD_STR,
)
DEFAULT_CONFIG = SP500_CONFIG
CONFIG_BY_VARIANT_KEY_DICT = {
    SP500_CONFIG.variant_key_str: SP500_CONFIG,
    NASDAQ100_CONFIG.variant_key_str: NASDAQ100_CONFIG,
    RUSSELL1000_CONFIG.variant_key_str: RUSSELL1000_CONFIG,
}
VOL_TARGET_CONFIG_BY_VARIANT_KEY_DICT = {
    SP500_VOL_TARGET_CONFIG.variant_key_str: SP500_VOL_TARGET_CONFIG,
    NASDAQ100_VOL_TARGET_CONFIG.variant_key_str: NASDAQ100_VOL_TARGET_CONFIG,
    RUSSELL1000_VOL_TARGET_CONFIG.variant_key_str: RUSSELL1000_VOL_TARGET_CONFIG,
}
SMA100_CONFIG_BY_VARIANT_KEY_DICT = {
    SP500_SMA100_CONFIG.variant_key_str: SP500_SMA100_CONFIG,
    NASDAQ100_SMA100_CONFIG.variant_key_str: NASDAQ100_SMA100_CONFIG,
    RUSSELL1000_SMA100_CONFIG.variant_key_str: RUSSELL1000_SMA100_CONFIG,
}
SMA100_VOL_TARGET_CONFIG_BY_VARIANT_KEY_DICT = {
    SP500_SMA100_VOL_TARGET_CONFIG.variant_key_str: SP500_SMA100_VOL_TARGET_CONFIG,
    NASDAQ100_SMA100_VOL_TARGET_CONFIG.variant_key_str: NASDAQ100_SMA100_VOL_TARGET_CONFIG,
    RUSSELL1000_SMA100_VOL_TARGET_CONFIG.variant_key_str: RUSSELL1000_SMA100_VOL_TARGET_CONFIG,
}
SMA100_INDEX_SMA200_CONFIG_BY_VARIANT_KEY_DICT = {
    SP500_SMA100_INDEX_SMA200_CONFIG.variant_key_str: SP500_SMA100_INDEX_SMA200_CONFIG,
    NASDAQ100_SMA100_INDEX_SMA200_CONFIG.variant_key_str: NASDAQ100_SMA100_INDEX_SMA200_CONFIG,
    RUSSELL1000_SMA100_INDEX_SMA200_CONFIG.variant_key_str: RUSSELL1000_SMA100_INDEX_SMA200_CONFIG,
}
SMA100_INDEX_SMA200_VOL_TARGET_CONFIG_BY_VARIANT_KEY_DICT = {
    SP500_SMA100_INDEX_SMA200_VOL_TARGET_CONFIG.variant_key_str: SP500_SMA100_INDEX_SMA200_VOL_TARGET_CONFIG,
    NASDAQ100_SMA100_INDEX_SMA200_VOL_TARGET_CONFIG.variant_key_str: NASDAQ100_SMA100_INDEX_SMA200_VOL_TARGET_CONFIG,
    RUSSELL1000_SMA100_INDEX_SMA200_VOL_TARGET_CONFIG.variant_key_str: RUSSELL1000_SMA100_INDEX_SMA200_VOL_TARGET_CONFIG,
}
SMA100_INDEX_SMA200_RANKING_SWEEP_CONFIG_BY_VARIANT_KEY_DICT = {
    SP500_SMA100_INDEX_SMA200_CONFIG.variant_key_str: SP500_SMA100_INDEX_SMA200_CONFIG,
    NASDAQ100_SMA100_INDEX_SMA200_CONFIG.variant_key_str: NASDAQ100_SMA100_INDEX_SMA200_CONFIG,
    RUSSELL1000_SMA100_INDEX_SMA200_CONFIG.variant_key_str: RUSSELL1000_SMA100_INDEX_SMA200_CONFIG,
    SP500_SMA100_INDEX_SMA200_VOLNORM_CONFIG.variant_key_str: SP500_SMA100_INDEX_SMA200_VOLNORM_CONFIG,
    NASDAQ100_SMA100_INDEX_SMA200_VOLNORM_CONFIG.variant_key_str: NASDAQ100_SMA100_INDEX_SMA200_VOLNORM_CONFIG,
    RUSSELL1000_SMA100_INDEX_SMA200_VOLNORM_CONFIG.variant_key_str: RUSSELL1000_SMA100_INDEX_SMA200_VOLNORM_CONFIG,
    SP500_SMA100_INDEX_SMA200_TREND_QUALITY_CONFIG.variant_key_str: SP500_SMA100_INDEX_SMA200_TREND_QUALITY_CONFIG,
    NASDAQ100_SMA100_INDEX_SMA200_TREND_QUALITY_CONFIG.variant_key_str: NASDAQ100_SMA100_INDEX_SMA200_TREND_QUALITY_CONFIG,
    RUSSELL1000_SMA100_INDEX_SMA200_TREND_QUALITY_CONFIG.variant_key_str: RUSSELL1000_SMA100_INDEX_SMA200_TREND_QUALITY_CONFIG,
    SP500_SMA100_INDEX_SMA200_MULTI_HORIZON_CONFIG.variant_key_str: SP500_SMA100_INDEX_SMA200_MULTI_HORIZON_CONFIG,
    NASDAQ100_SMA100_INDEX_SMA200_MULTI_HORIZON_CONFIG.variant_key_str: NASDAQ100_SMA100_INDEX_SMA200_MULTI_HORIZON_CONFIG,
    RUSSELL1000_SMA100_INDEX_SMA200_MULTI_HORIZON_CONFIG.variant_key_str: RUSSELL1000_SMA100_INDEX_SMA200_MULTI_HORIZON_CONFIG,
    SP500_SMA100_INDEX_SMA200_RESIDUAL_CONFIG.variant_key_str: SP500_SMA100_INDEX_SMA200_RESIDUAL_CONFIG,
    NASDAQ100_SMA100_INDEX_SMA200_RESIDUAL_CONFIG.variant_key_str: NASDAQ100_SMA100_INDEX_SMA200_RESIDUAL_CONFIG,
    RUSSELL1000_SMA100_INDEX_SMA200_RESIDUAL_CONFIG.variant_key_str: RUSSELL1000_SMA100_INDEX_SMA200_RESIDUAL_CONFIG,
}
SMA100_INDEX_SMA200_PN_RANKING_CONFIG_BY_VARIANT_KEY_DICT = {
    SP500_SMA100_INDEX_SMA200_CONFIG.variant_key_str: SP500_SMA100_INDEX_SMA200_CONFIG,
    NASDAQ100_SMA100_INDEX_SMA200_CONFIG.variant_key_str: NASDAQ100_SMA100_INDEX_SMA200_CONFIG,
    RUSSELL1000_SMA100_INDEX_SMA200_CONFIG.variant_key_str: RUSSELL1000_SMA100_INDEX_SMA200_CONFIG,
    SP500_SMA100_INDEX_SMA200_PN_EV_CONFIG.variant_key_str: SP500_SMA100_INDEX_SMA200_PN_EV_CONFIG,
    NASDAQ100_SMA100_INDEX_SMA200_PN_EV_CONFIG.variant_key_str: NASDAQ100_SMA100_INDEX_SMA200_PN_EV_CONFIG,
    RUSSELL1000_SMA100_INDEX_SMA200_PN_EV_CONFIG.variant_key_str: RUSSELL1000_SMA100_INDEX_SMA200_PN_EV_CONFIG,
    SP500_SMA100_INDEX_SMA200_PN_LRB_CONFIG.variant_key_str: SP500_SMA100_INDEX_SMA200_PN_LRB_CONFIG,
    NASDAQ100_SMA100_INDEX_SMA200_PN_LRB_CONFIG.variant_key_str: NASDAQ100_SMA100_INDEX_SMA200_PN_LRB_CONFIG,
    RUSSELL1000_SMA100_INDEX_SMA200_PN_LRB_CONFIG.variant_key_str: RUSSELL1000_SMA100_INDEX_SMA200_PN_LRB_CONFIG,
}
ALL_CONFIG_BY_VARIANT_KEY_DICT = {
    **CONFIG_BY_VARIANT_KEY_DICT,
    **VOL_TARGET_CONFIG_BY_VARIANT_KEY_DICT,
    **SMA100_CONFIG_BY_VARIANT_KEY_DICT,
    **SMA100_VOL_TARGET_CONFIG_BY_VARIANT_KEY_DICT,
    **SMA100_INDEX_SMA200_CONFIG_BY_VARIANT_KEY_DICT,
    **SMA100_INDEX_SMA200_VOL_TARGET_CONFIG_BY_VARIANT_KEY_DICT,
    **SMA100_INDEX_SMA200_RANKING_SWEEP_CONFIG_BY_VARIANT_KEY_DICT,
    **SMA100_INDEX_SMA200_PN_RANKING_CONFIG_BY_VARIANT_KEY_DICT,
}


__all__ = [
    "CONFIG_BY_VARIANT_KEY_DICT",
    "DEFAULT_CONFIG",
    "ALL_CONFIG_BY_VARIANT_KEY_DICT",
    "INDEX_SMA_FILTER_WINDOW_INT",
    "Jt121Top20Config",
    "Jt121Top20Strategy",
    "LOOKBACK_TRADING_DAY_INT",
    "MAX_GROSS_EXPOSURE_FLOAT",
    "MAX_POSITIONS_INT",
    "MULTI_HORIZON_Z_RANKING_METHOD_STR",
    "NASDAQ100_CONFIG",
    "NASDAQ100_SMA100_INDEX_SMA200_CONFIG",
    "NASDAQ100_SMA100_INDEX_SMA200_MULTI_HORIZON_CONFIG",
    "NASDAQ100_SMA100_INDEX_SMA200_PN_EV_CONFIG",
    "NASDAQ100_SMA100_INDEX_SMA200_PN_LRB_CONFIG",
    "NASDAQ100_SMA100_INDEX_SMA200_RESIDUAL_CONFIG",
    "NASDAQ100_SMA100_INDEX_SMA200_TREND_QUALITY_CONFIG",
    "NASDAQ100_SMA100_INDEX_SMA200_VOLNORM_CONFIG",
    "NASDAQ100_SMA100_INDEX_SMA200_VOL_TARGET_CONFIG",
    "NASDAQ100_SMA100_CONFIG",
    "NASDAQ100_SMA100_VOL_TARGET_CONFIG",
    "NASDAQ100_VOL_TARGET_CONFIG",
    "REALIZED_VOL_WINDOW_INT",
    "PN_EV_MULTI_WINDOW_Z_RANKING_METHOD_STR",
    "PN_LRB_MULTI_WINDOW_Z_RANKING_METHOD_STR",
    "PN_RANKING_WINDOW_TUPLE",
    "RESIDUAL_12_1_RANKING_METHOD_STR",
    "RUSSELL1000_CONFIG",
    "RUSSELL1000_SMA100_INDEX_SMA200_CONFIG",
    "RUSSELL1000_SMA100_INDEX_SMA200_MULTI_HORIZON_CONFIG",
    "RUSSELL1000_SMA100_INDEX_SMA200_PN_EV_CONFIG",
    "RUSSELL1000_SMA100_INDEX_SMA200_PN_LRB_CONFIG",
    "RUSSELL1000_SMA100_INDEX_SMA200_RESIDUAL_CONFIG",
    "RUSSELL1000_SMA100_INDEX_SMA200_TREND_QUALITY_CONFIG",
    "RUSSELL1000_SMA100_INDEX_SMA200_VOLNORM_CONFIG",
    "RUSSELL1000_SMA100_INDEX_SMA200_VOL_TARGET_CONFIG",
    "RUSSELL1000_SMA100_CONFIG",
    "RUSSELL1000_SMA100_VOL_TARGET_CONFIG",
    "RUSSELL1000_VOL_TARGET_CONFIG",
    "SMA100_INDEX_SMA200_CONFIG_BY_VARIANT_KEY_DICT",
    "SMA100_INDEX_SMA200_PN_RANKING_CONFIG_BY_VARIANT_KEY_DICT",
    "SMA100_INDEX_SMA200_RANKING_SWEEP_CONFIG_BY_VARIANT_KEY_DICT",
    "SMA100_INDEX_SMA200_VOL_TARGET_CONFIG_BY_VARIANT_KEY_DICT",
    "SMA100_CONFIG_BY_VARIANT_KEY_DICT",
    "SMA100_VOL_TARGET_CONFIG_BY_VARIANT_KEY_DICT",
    "SKIP_TRADING_DAY_INT",
    "SP500_CONFIG",
    "SP500_SMA100_INDEX_SMA200_CONFIG",
    "SP500_SMA100_INDEX_SMA200_MULTI_HORIZON_CONFIG",
    "SP500_SMA100_INDEX_SMA200_PN_EV_CONFIG",
    "SP500_SMA100_INDEX_SMA200_PN_LRB_CONFIG",
    "SP500_SMA100_INDEX_SMA200_RESIDUAL_CONFIG",
    "SP500_SMA100_INDEX_SMA200_TREND_QUALITY_CONFIG",
    "SP500_SMA100_INDEX_SMA200_VOLNORM_CONFIG",
    "SP500_SMA100_INDEX_SMA200_VOL_TARGET_CONFIG",
    "SP500_SMA100_CONFIG",
    "SP500_SMA100_VOL_TARGET_CONFIG",
    "SP500_VOL_TARGET_CONFIG",
    "STOCK_SMA_FILTER_WINDOW_INT",
    "TARGET_ANNUAL_VOLATILITY_FLOAT",
    "TREND_QUALITY_RANKING_METHOD_STR",
    "VOL_TARGET_CONFIG_BY_VARIANT_KEY_DICT",
    "VOL_NORMALIZED_12_1_RANKING_METHOD_STR",
    "compute_index_above_sma_filter_ser",
    "compute_jt_12_1_momentum_score_df",
    "compute_pn_indicator_score_df",
    "compute_ranking_score_df",
    "compute_stock_above_sma_filter_df",
    "get_jt_12_1_top20_data",
    "run_all_variants",
    "run_variant",
]


def compute_jt_12_1_momentum_score_df(
    price_close_df: pd.DataFrame,
    config_obj: Jt121Top20Config = DEFAULT_CONFIG,
) -> pd.DataFrame:
    if len(price_close_df.index) == 0:
        raise ValueError("price_close_df must not be empty.")
    if not price_close_df.index.is_monotonic_increasing:
        raise ValueError("price_close_df index must be sorted.")
    if price_close_df.index.has_duplicates:
        raise ValueError("price_close_df index must not contain duplicates.")

    monthly_decision_close_df = get_monthly_decision_close_df(price_close_df=price_close_df)

    # *** CRITICAL*** lookahead-sensitive 12-1 construction: at decision
    # close T, the numerator is close T-21 and the denominator is close T-252.
    # The most recent 21 trading sessions and every future return are excluded.
    daily_score_df = (
        price_close_df.shift(config_obj.skip_trading_day_int)
        / price_close_df.shift(config_obj.lookback_trading_day_int)
    ) - 1.0
    daily_score_df = daily_score_df.replace([np.inf, -np.inf], np.nan)

    momentum_score_df = daily_score_df.reindex(monthly_decision_close_df.index)
    valid_score_bool_ser = momentum_score_df.notna().any(axis=1)
    return momentum_score_df.loc[valid_score_bool_ser].copy()


def row_zscore_df(value_df: pd.DataFrame) -> pd.DataFrame:
    row_mean_ser = value_df.mean(axis=1, skipna=True)
    row_std_ser = value_df.std(axis=1, skipna=True)
    return value_df.sub(row_mean_ser, axis=0).div(row_std_ser.replace(0.0, np.nan), axis=0)


def compute_vol_normalized_12_1_score_df(
    price_close_df: pd.DataFrame,
    config_obj: Jt121Top20Config = DEFAULT_CONFIG,
) -> pd.DataFrame:
    raw_momentum_df = compute_jt_12_1_momentum_score_df(
        price_close_df=price_close_df,
        config_obj=config_obj,
    )
    vol_normalization_window_int = config_obj.lookback_trading_day_int - config_obj.skip_trading_day_int
    daily_return_df = price_close_df.pct_change(fill_method=None)
    # *** CRITICAL*** lookahead-sensitive volatility normalization: at
    # decision close T, the volatility window is shifted by the same 21-day
    # skip as the 12-1 momentum numerator, so returns after T-21 are excluded.
    realized_volatility_df = (
        daily_return_df.shift(config_obj.skip_trading_day_int)
        .rolling(
            window=vol_normalization_window_int,
            min_periods=vol_normalization_window_int,
        )
        .std(ddof=1)
        * np.sqrt(252.0)
    )
    score_df = raw_momentum_df.div(realized_volatility_df.reindex(raw_momentum_df.index))
    return score_df.replace([np.inf, -np.inf], np.nan)


def compute_multi_horizon_z_score_df(
    price_close_df: pd.DataFrame,
    config_obj: Jt121Top20Config = DEFAULT_CONFIG,
) -> pd.DataFrame:
    monthly_decision_close_df = get_monthly_decision_close_df(price_close_df=price_close_df)
    # *** CRITICAL*** lookahead-sensitive multi-horizon ranking: each horizon
    # uses the same T-21 skip before row-wise normalization on the decision
    # close cross-section.
    momentum_12_1_df = (
        price_close_df.shift(config_obj.skip_trading_day_int)
        / price_close_df.shift(252)
    ) - 1.0
    momentum_6_1_df = (
        price_close_df.shift(config_obj.skip_trading_day_int)
        / price_close_df.shift(126)
    ) - 1.0
    momentum_3_1_df = (
        price_close_df.shift(config_obj.skip_trading_day_int)
        / price_close_df.shift(63)
    ) - 1.0
    monthly_12_1_df = momentum_12_1_df.reindex(monthly_decision_close_df.index)
    monthly_6_1_df = momentum_6_1_df.reindex(monthly_decision_close_df.index)
    monthly_3_1_df = momentum_3_1_df.reindex(monthly_decision_close_df.index)
    score_df = (
        0.50 * row_zscore_df(monthly_12_1_df)
        + 0.30 * row_zscore_df(monthly_6_1_df)
        + 0.20 * row_zscore_df(monthly_3_1_df)
    )
    return score_df.replace([np.inf, -np.inf], np.nan)


def compute_trend_quality_score_df(
    price_close_df: pd.DataFrame,
    config_obj: Jt121Top20Config = DEFAULT_CONFIG,
) -> pd.DataFrame:
    if len(price_close_df.index) == 0:
        raise ValueError("price_close_df must not be empty.")
    if not price_close_df.index.is_monotonic_increasing:
        raise ValueError("price_close_df index must be sorted.")
    if price_close_df.index.has_duplicates:
        raise ValueError("price_close_df index must not contain duplicates.")

    monthly_decision_close_df = get_monthly_decision_close_df(price_close_df=price_close_df)
    log_price_df = np.log(price_close_df.where(price_close_df > 0.0))
    log_price_mat = log_price_df.to_numpy(dtype=float)
    symbol_list = log_price_df.columns.astype(str).tolist()
    x_vec = np.arange(TREND_QUALITY_WINDOW_INT, dtype=float)
    centered_x_vec = x_vec - float(x_vec.mean())
    x_sum_sq_float = float(np.square(centered_x_vec).sum())
    score_row_list: list[np.ndarray] = []
    score_index_list: list[pd.Timestamp] = []

    for decision_ts in monthly_decision_close_df.index:
        row_position_int = int(log_price_df.index.get_loc(decision_ts))
        if row_position_int + 1 < TREND_QUALITY_WINDOW_INT:
            continue
        window_mat = log_price_mat[
            row_position_int + 1 - TREND_QUALITY_WINDOW_INT : row_position_int + 1,
            :,
        ]
        finite_col_mask_vec = np.isfinite(window_mat).all(axis=0)
        score_vec = np.full(len(symbol_list), np.nan, dtype=float)
        if finite_col_mask_vec.any():
            clean_window_mat = window_mat[:, finite_col_mask_vec]
            y_mean_vec = clean_window_mat.mean(axis=0)
            centered_y_mat = clean_window_mat - y_mean_vec
            # *** CRITICAL*** lookahead-sensitive trend-quality ranking:
            # regression uses only the trailing 126 closes ending at decision
            # close T, then any resulting order fills at the next open.
            slope_vec = centered_x_vec @ centered_y_mat / x_sum_sq_float
            fitted_centered_y_mat = np.outer(centered_x_vec, slope_vec)
            residual_mat = centered_y_mat - fitted_centered_y_mat
            total_sum_sq_vec = np.square(centered_y_mat).sum(axis=0)
            residual_sum_sq_vec = np.square(residual_mat).sum(axis=0)
            r2_vec = 1.0 - (residual_sum_sq_vec / total_sum_sq_vec)
            annualized_slope_vec = np.exp(slope_vec * 252.0) - 1.0
            clean_score_vec = annualized_slope_vec * np.clip(r2_vec, 0.0, 1.0)
            score_vec[finite_col_mask_vec] = clean_score_vec
        score_row_list.append(score_vec)
        score_index_list.append(pd.Timestamp(decision_ts))

    score_df = pd.DataFrame(score_row_list, index=score_index_list, columns=symbol_list)
    return score_df.replace([np.inf, -np.inf], np.nan)


def compute_residual_12_1_score_df(
    price_close_df: pd.DataFrame,
    index_close_ser: pd.Series,
    config_obj: Jt121Top20Config = DEFAULT_CONFIG,
) -> pd.DataFrame:
    raw_momentum_df = compute_jt_12_1_momentum_score_df(
        price_close_df=price_close_df,
        config_obj=config_obj,
    )
    clean_index_close_ser = pd.Series(index_close_ser, copy=True).astype(float)
    stock_return_df = price_close_df.pct_change(fill_method=None)
    index_return_ser = clean_index_close_ser.pct_change(fill_method=None)
    vol_normalization_window_int = config_obj.lookback_trading_day_int - config_obj.skip_trading_day_int
    # *** CRITICAL*** lookahead-sensitive residual momentum: beta is estimated
    # from daily returns ending at T-21, matching the 12-1 skip window.
    skipped_stock_return_df = stock_return_df.shift(config_obj.skip_trading_day_int)
    skipped_index_return_ser = index_return_ser.shift(config_obj.skip_trading_day_int)
    rolling_cov_df = skipped_stock_return_df.rolling(
        window=vol_normalization_window_int,
        min_periods=vol_normalization_window_int,
    ).cov(skipped_index_return_ser)
    rolling_index_var_ser = skipped_index_return_ser.rolling(
        window=vol_normalization_window_int,
        min_periods=vol_normalization_window_int,
    ).var(ddof=1)
    beta_df = rolling_cov_df.div(rolling_index_var_ser.replace(0.0, np.nan), axis=0)
    index_momentum_ser = (
        clean_index_close_ser.shift(config_obj.skip_trading_day_int)
        / clean_index_close_ser.shift(config_obj.lookback_trading_day_int)
    ) - 1.0
    residual_score_df = raw_momentum_df.sub(
        beta_df.reindex(raw_momentum_df.index).mul(index_momentum_ser.reindex(raw_momentum_df.index), axis=0)
    )
    return residual_score_df.replace([np.inf, -np.inf], np.nan)


def compute_pn_indicator_score_df(
    price_close_df: pd.DataFrame,
    score_name_str: str,
) -> pd.DataFrame:
    if len(price_close_df.index) == 0:
        raise ValueError("price_close_df must not be empty.")
    if not price_close_df.index.is_monotonic_increasing:
        raise ValueError("price_close_df index must be sorted.")
    if price_close_df.index.has_duplicates:
        raise ValueError("price_close_df index must not contain duplicates.")
    if score_name_str not in {"ev", "lrb"}:
        raise ValueError("score_name_str must be 'ev' or 'lrb'.")

    monthly_decision_close_df = get_monthly_decision_close_df(price_close_df=price_close_df)
    log_price_df = np.log(price_close_df.where(price_close_df > 0.0))
    # *** CRITICAL *** lookahead-sensitive P/N ranking: log returns and rolling
    # P/N windows end at decision close T, and orders still fill only at the
    # next tradable open. No data after T is used here.
    log_return_df = log_price_df.diff()
    positive_return_df = log_return_df.clip(lower=0.0)
    negative_return_df = (-log_return_df).clip(lower=0.0)

    window_score_df_list: list[pd.DataFrame] = []
    for window_int in PN_RANKING_WINDOW_TUPLE:
        positive_sum_df = positive_return_df.rolling(window=window_int, min_periods=window_int).sum()
        negative_sum_df = negative_return_df.rolling(window=window_int, min_periods=window_int).sum()
        if score_name_str == "ev":
            # EV = (P - N) / (P + N), where P is positive-return sum and N is negative-return sum.
            lr_df = positive_sum_df - negative_sum_df
            v_df = positive_sum_df + negative_sum_df
            window_score_df = lr_df.div(v_df.replace(0.0, np.nan))
        else:
            # LRB = P / N. The ranking uses log(LRB), which preserves single-window order.
            lrb_df = positive_sum_df.div(negative_sum_df.replace(0.0, np.nan))
            window_score_df = np.log(lrb_df.where(lrb_df > 0.0))
        window_score_df = window_score_df.reindex(monthly_decision_close_df.index)
        window_score_df_list.append(row_zscore_df(window_score_df))

    score_panel_df = pd.concat(window_score_df_list, keys=range(len(window_score_df_list)), names=["window_idx", "date"])
    score_df = score_panel_df.groupby(level="date").mean()
    return score_df.replace([np.inf, -np.inf], np.nan)


def compute_ranking_score_df(
    price_close_df: pd.DataFrame,
    index_close_ser: pd.Series,
    config_obj: Jt121Top20Config = DEFAULT_CONFIG,
) -> pd.DataFrame:
    if config_obj.ranking_method_str == RAW_12_1_RANKING_METHOD_STR:
        return compute_jt_12_1_momentum_score_df(price_close_df=price_close_df, config_obj=config_obj)
    if config_obj.ranking_method_str == VOL_NORMALIZED_12_1_RANKING_METHOD_STR:
        return compute_vol_normalized_12_1_score_df(price_close_df=price_close_df, config_obj=config_obj)
    if config_obj.ranking_method_str == TREND_QUALITY_RANKING_METHOD_STR:
        return compute_trend_quality_score_df(price_close_df=price_close_df, config_obj=config_obj)
    if config_obj.ranking_method_str == MULTI_HORIZON_Z_RANKING_METHOD_STR:
        return compute_multi_horizon_z_score_df(price_close_df=price_close_df, config_obj=config_obj)
    if config_obj.ranking_method_str == RESIDUAL_12_1_RANKING_METHOD_STR:
        return compute_residual_12_1_score_df(
            price_close_df=price_close_df,
            index_close_ser=index_close_ser,
            config_obj=config_obj,
        )
    if config_obj.ranking_method_str == PN_EV_MULTI_WINDOW_Z_RANKING_METHOD_STR:
        return compute_pn_indicator_score_df(price_close_df=price_close_df, score_name_str="ev")
    if config_obj.ranking_method_str == PN_LRB_MULTI_WINDOW_Z_RANKING_METHOD_STR:
        return compute_pn_indicator_score_df(price_close_df=price_close_df, score_name_str="lrb")
    raise ValueError(f"Unknown ranking_method_str: {config_obj.ranking_method_str}")


def compute_stock_above_sma_filter_df(
    price_close_df: pd.DataFrame,
    config_obj: Jt121Top20Config = DEFAULT_CONFIG,
) -> pd.DataFrame:
    if len(price_close_df.index) == 0:
        raise ValueError("price_close_df must not be empty.")
    if not price_close_df.index.is_monotonic_increasing:
        raise ValueError("price_close_df index must be sorted.")
    if price_close_df.index.has_duplicates:
        raise ValueError("price_close_df index must not contain duplicates.")

    # *** CRITICAL*** lookahead-sensitive stock trend filter: at decision
    # close T, Close_T is compared only to the trailing SMA ending at T.
    # Orders are still generated for the next tradable open after T.
    stock_sma_df = price_close_df.rolling(
        window=config_obj.stock_sma_window_int,
        min_periods=config_obj.stock_sma_window_int,
    ).mean()
    return price_close_df.gt(stock_sma_df) & stock_sma_df.notna()


def compute_index_above_sma_filter_ser(
    index_close_ser: pd.Series,
    config_obj: Jt121Top20Config = DEFAULT_CONFIG,
) -> pd.Series:
    if len(index_close_ser.index) == 0:
        raise ValueError("index_close_ser must not be empty.")
    if not index_close_ser.index.is_monotonic_increasing:
        raise ValueError("index_close_ser index must be sorted.")
    if index_close_ser.index.has_duplicates:
        raise ValueError("index_close_ser index must not contain duplicates.")

    clean_index_close_ser = pd.Series(index_close_ser, copy=True).astype(float)
    # *** CRITICAL*** lookahead-sensitive benchmark regime filter: at decision
    # close T, IndexClose_T is compared only to the trailing SMA ending at T.
    # The resulting monthly order still fills at the next tradable open.
    index_sma_ser = clean_index_close_ser.rolling(
        window=config_obj.index_sma_window_int,
        min_periods=config_obj.index_sma_window_int,
    ).mean()
    return clean_index_close_ser.gt(index_sma_ser) & index_sma_ser.notna()


def get_jt_12_1_top20_data(
    config_obj: Jt121Top20Config = DEFAULT_CONFIG,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    _, raw_universe_df = build_index_constituent_matrix(indexname=config_obj.indexname_str)

    history_start_ts = pd.Timestamp(config_obj.history_start_date_str)
    backtest_start_ts = pd.Timestamp(config_obj.backtest_start_date_str)
    filtered_universe_df = raw_universe_df.loc[raw_universe_df.index >= history_start_ts].copy()
    active_universe_df = filtered_universe_df.loc[filtered_universe_df.index >= backtest_start_ts].copy()
    if config_obj.end_date_str is not None:
        end_date_ts = pd.Timestamp(config_obj.end_date_str)
        active_universe_df = active_universe_df.loc[active_universe_df.index <= end_date_ts]

    active_symbol_list = active_universe_df.columns[
        active_universe_df.sum(axis=0) > 0
    ].astype(str).tolist()
    if len(active_symbol_list) == 0:
        raise RuntimeError(
            f"No active {config_obj.indexname_str} symbols were found for the requested window."
        )

    pricing_data_df = load_raw_prices(
        symbols=active_symbol_list,
        benchmarks=[config_obj.benchmark_symbol_str],
        start_date=config_obj.history_start_date_str,
        end_date=config_obj.end_date_str,
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

    keep_symbol_set = set(audited_universe_df.columns.astype(str).tolist() + [config_obj.benchmark_symbol_str])
    pricing_data_df = pricing_data_df.loc[
        :,
        pricing_data_df.columns.get_level_values(0).isin(keep_symbol_set),
    ].sort_index()

    close_symbol_list = audited_universe_df.columns.astype(str).tolist()
    price_close_df = pd.DataFrame(
        {symbol_str: pricing_data_df[(symbol_str, "Close")] for symbol_str in close_symbol_list},
        index=pricing_data_df.index,
    ).astype(float)
    momentum_score_df = compute_jt_12_1_momentum_score_df(
        price_close_df=price_close_df,
        config_obj=config_obj,
    )
    rebalance_schedule_df = map_month_end_decision_dates_to_rebalance_schedule_df(
        decision_date_index=pd.DatetimeIndex(momentum_score_df.index),
        execution_index=pricing_data_df.index,
    )
    return pricing_data_df, audited_universe_df, rebalance_schedule_df


class Jt121Top20Strategy(Strategy):
    """
    Long-only monthly 12-1 cross-sectional momentum selector.

    This strategy has no regime filter and no positive-momentum floor. It ranks
    the PIT universe every month and owns the top N finite scores.
    """

    enable_signal_audit = True
    signal_audit_sample_size = 10

    def __init__(
        self,
        name: str,
        benchmarks: Sequence[str],
        rebalance_schedule_df: pd.DataFrame,
        config_obj: Jt121Top20Config,
    ):
        super().__init__(
            name=name,
            benchmarks=list(benchmarks),
            capital_base=config_obj.capital_base_float,
            slippage=config_obj.slippage_float,
            commission_per_share=config_obj.commission_per_share_float,
            commission_minimum=config_obj.commission_minimum_float,
        )

        if len(rebalance_schedule_df) == 0:
            raise ValueError("rebalance_schedule_df must not be empty.")
        if "decision_date_ts" not in rebalance_schedule_df.columns:
            raise ValueError("rebalance_schedule_df must contain decision_date_ts.")
        if config_obj.benchmark_symbol_str not in benchmarks:
            raise ValueError("benchmarks must include config_obj.benchmark_symbol_str.")

        self.rebalance_schedule_df = rebalance_schedule_df.copy().sort_index()
        self.config_obj = config_obj
        self.trade_id_int = 0
        self.current_trade_map: defaultdict[str, int] = defaultdict(default_trade_id_int)
        self.universe_df: pd.DataFrame | None = None

    @property
    def momentum_score_field_str(self) -> str:
        return f"ranking_{self.config_obj.ranking_method_str}_float"

    @property
    def stock_sma_pass_field_str(self) -> str:
        return f"close_above_sma_{self.config_obj.stock_sma_window_int}_bool"

    @property
    def index_sma_pass_field_str(self) -> str:
        return f"index_close_above_sma_{self.config_obj.index_sma_window_int}_bool"

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
        benchmark_close_ser = signal_data_df[(self.config_obj.benchmark_symbol_str, "Close")].astype(float)
        momentum_score_df = compute_ranking_score_df(
            price_close_df=price_close_df,
            index_close_ser=benchmark_close_ser,
            config_obj=self.config_obj,
        ).reindex(signal_data_df.index)

        feature_df = momentum_score_df.copy()
        feature_df.columns = pd.MultiIndex.from_tuples(
            [(symbol_str, self.momentum_score_field_str) for symbol_str in feature_df.columns]
        )
        if self.config_obj.stock_sma_filter_enabled_bool:
            stock_sma_pass_df = compute_stock_above_sma_filter_df(
                price_close_df=price_close_df,
                config_obj=self.config_obj,
            ).reindex(signal_data_df.index)
            stock_sma_feature_df = stock_sma_pass_df.copy()
            stock_sma_feature_df.columns = pd.MultiIndex.from_tuples(
                [(symbol_str, self.stock_sma_pass_field_str) for symbol_str in stock_sma_feature_df.columns]
            )
            feature_df = pd.concat([feature_df, stock_sma_feature_df], axis=1)
        if self.config_obj.index_sma_filter_enabled_bool:
            index_sma_pass_ser = compute_index_above_sma_filter_ser(
                index_close_ser=benchmark_close_ser,
                config_obj=self.config_obj,
            ).reindex(signal_data_df.index)
            index_sma_feature_df = pd.DataFrame(
                index=signal_data_df.index,
            )
            index_sma_feature_df[
                (self.config_obj.benchmark_symbol_str, self.index_sma_pass_field_str)
            ] = index_sma_pass_ser
            index_sma_feature_df.columns = pd.MultiIndex.from_tuples(index_sma_feature_df.columns)
            feature_df = pd.concat([feature_df, index_sma_feature_df], axis=1)
        return pd.concat([signal_data_df, feature_df], axis=1)

    def is_index_sma_filter_pass_bool(self, close_row_ser: pd.Series) -> bool:
        if not self.config_obj.index_sma_filter_enabled_bool:
            return True
        field_tuple = (self.config_obj.benchmark_symbol_str, self.index_sma_pass_field_str)
        if field_tuple not in close_row_ser.index:
            return False
        value_obj = close_row_ser.loc[field_tuple]
        if pd.isna(value_obj):
            return False
        return bool(value_obj)

    def get_selected_symbol_list(self, close_row_ser: pd.Series) -> list[str]:
        if self.universe_df is None:
            raise RuntimeError("universe_df must be set before monthly rebalances.")

        candidate_feature_df = close_row_ser.unstack()
        if self.momentum_score_field_str not in candidate_feature_df.columns:
            return []

        universe_member_ser = get_asof_universe_membership_ser(
            self.universe_df,
            pd.Timestamp(self.previous_bar),
        )
        active_symbol_list = universe_member_ser[universe_member_ser == 1].index.astype(str).tolist()
        candidate_feature_df = candidate_feature_df[candidate_feature_df.index.isin(active_symbol_list)].copy()
        if len(candidate_feature_df) == 0:
            return []

        candidate_feature_df = candidate_feature_df.assign(
            momentum_score_float=pd.to_numeric(
                candidate_feature_df[self.momentum_score_field_str],
                errors="coerce",
            ),
            symbol_str=candidate_feature_df.index.astype(str),
        )
        finite_score_mask_vec = np.isfinite(
            candidate_feature_df["momentum_score_float"].to_numpy(dtype=float)
        )
        candidate_feature_df = candidate_feature_df.loc[finite_score_mask_vec].copy()
        if len(candidate_feature_df) == 0:
            return []

        if self.config_obj.stock_sma_filter_enabled_bool:
            if self.stock_sma_pass_field_str not in candidate_feature_df.columns:
                return []
            stock_sma_pass_ser = pd.Series(
                candidate_feature_df[self.stock_sma_pass_field_str],
                index=candidate_feature_df.index,
                dtype="boolean",
            ).fillna(False).astype(bool)
            candidate_feature_df = candidate_feature_df.loc[stock_sma_pass_ser].copy()
            if len(candidate_feature_df) == 0:
                return []

        selected_feature_df = candidate_feature_df.sort_values(
            by=["momentum_score_float", "symbol_str"],
            ascending=[False, True],
            kind="mergesort",
        ).iloc[: self.config_obj.max_positions_int]
        return selected_feature_df.index.astype(str).tolist()

    def get_gross_exposure_target_float(self) -> float:
        if not self.config_obj.volatility_target_enabled_bool:
            return 1.0

        if len(self._daily_return_history_list) < self.config_obj.realized_vol_window_int:
            return 1.0

        # *** CRITICAL*** volatility-target sizing uses only realized strategy
        # returns already recorded through previous_bar. The current rebalance
        # open and all future returns are excluded.
        trailing_return_ser = pd.Series(
            self._daily_return_history_list[-self.config_obj.realized_vol_window_int :],
            dtype=float,
        )
        realized_volatility_float = float(
            trailing_return_ser.std(ddof=1) * np.sqrt(252.0)
        )
        if not np.isfinite(realized_volatility_float):
            return 0.0
        if realized_volatility_float <= 0.0:
            return self.config_obj.max_gross_exposure_float

        raw_scale_float = float(
            self.config_obj.target_annual_volatility_float / realized_volatility_float
        )
        capped_scale_float = min(raw_scale_float, self.config_obj.max_gross_exposure_float)
        return max(0.0, capped_scale_float)

    def iterate(self, data: pd.DataFrame, close: pd.Series, open_prices: pd.Series):
        if close is None or data is None:
            return
        if self.current_bar not in self.rebalance_schedule_df.index:
            return

        decision_date_ts = pd.Timestamp(self.rebalance_schedule_df.loc[self.current_bar, "decision_date_ts"])
        # *** CRITICAL*** The scheduled month-end decision close must equal
        # previous_bar exactly. If not, this 12-1 rank could be paired with
        # the wrong next-open fill.
        if pd.Timestamp(self.previous_bar) != decision_date_ts:
            raise RuntimeError(
                f"Schedule misalignment on {self.current_bar}: "
                f"decision_date_ts={decision_date_ts}, previous_bar={self.previous_bar}."
            )

        selected_symbol_list: list[str] = []
        if self.is_index_sma_filter_pass_bool(close_row_ser=close):
            selected_symbol_list = self.get_selected_symbol_list(close_row_ser=close)
        gross_exposure_target_float = 0.0
        if len(selected_symbol_list) > 0:
            gross_exposure_target_float = self.get_gross_exposure_target_float()
        selected_symbol_set = set(selected_symbol_list) if gross_exposure_target_float > 0.0 else set()
        current_position_ser = self.get_positions()
        long_position_ser = current_position_ser[current_position_ser > 0]

        for symbol_str in long_position_ser.index.astype(str):
            if symbol_str in selected_symbol_set:
                continue
            self.order_target_value(
                symbol_str,
                0.0,
                trade_id=self.current_trade_map[symbol_str],
            )

        if len(selected_symbol_list) == 0:
            return

        if gross_exposure_target_float <= 0.0:
            return

        target_weight_float = gross_exposure_target_float / float(len(selected_symbol_list))
        for symbol_str in selected_symbol_list:
            current_share_float = float(current_position_ser.get(symbol_str, 0.0))
            if current_share_float == 0.0:
                self.trade_id_int += 1
                self.current_trade_map[symbol_str] = self.trade_id_int

            self.order_target_percent(
                symbol_str,
                target_weight_float,
                trade_id=self.current_trade_map[symbol_str],
            )


def _with_run_overrides(
    config_obj: Jt121Top20Config,
    backtest_start_date_str: str | None = None,
    capital_base_float: float | None = None,
    end_date_str: str | None = None,
) -> Jt121Top20Config:
    if backtest_start_date_str is None and capital_base_float is None and end_date_str is None:
        return config_obj

    return replace(
        config_obj,
        backtest_start_date_str=(
            config_obj.backtest_start_date_str if backtest_start_date_str is None else backtest_start_date_str
        ),
        capital_base_float=(
            config_obj.capital_base_float if capital_base_float is None else float(capital_base_float)
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
) -> Jt121Top20Strategy:
    if variant_key_str not in ALL_CONFIG_BY_VARIANT_KEY_DICT:
        raise ValueError(f"Unknown variant_key_str: {variant_key_str}")

    config_obj = _with_run_overrides(
        config_obj=ALL_CONFIG_BY_VARIANT_KEY_DICT[variant_key_str],
        backtest_start_date_str=backtest_start_date_str,
        capital_base_float=capital_base_float,
        end_date_str=end_date_str,
    )
    pricing_data_df, universe_df, rebalance_schedule_df = get_jt_12_1_top20_data(config_obj=config_obj)
    strategy_obj = Jt121Top20Strategy(
        name=f"strategy_mo_jt_12_1_top20_monthly_{config_obj.variant_key_str}",
        benchmarks=[config_obj.benchmark_symbol_str],
        rebalance_schedule_df=rebalance_schedule_df,
        config_obj=config_obj,
    )
    strategy_obj.universe_df = universe_df

    # *** CRITICAL*** Keep pre-start history for 12-1 score formation, but
    # execute only from the configured backtest start.
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
) -> dict[str, Jt121Top20Strategy]:
    strategy_dict: dict[str, Jt121Top20Strategy] = {}
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


if __name__ == "__main__":
    run_all_variants()
