"""
PDP timing strategy driven by the Kenneth French daily momentum factor.

Core formulas
-------------
Let r^{factor}_t be the Kenneth French daily momentum factor return in decimal
form and define a synthetic cumulative factor index:

    factor_index_t
        = 100 * prod_{u <= t}(1 + r^{factor}_u)

The timing oscillator then matches the article logic:

    proxy_return_{10d,t}
        = factor_index_t / factor_index_{t-10} - 1

    proxy_smooth_ser_t
        = mean(proxy_return_{10d,t-4:t})

    proxy_rank_pct_t
        = (1 / N_t) * sum_{s <= t} 1[proxy_smooth_ser_s <= proxy_smooth_ser_t]

The traded asset is `PDP` and the binary target is:

    target_weight_t
        = 1[proxy_rank_pct_t < 0.50]

Execution uses the next daily open:

    q^{target}_{t+1}
        = floor(V_t * target_weight_t / O^{PDP}_{t+1})

Research-only caveat
--------------------
This file uses the current Kenneth French Data Library daily momentum history
as the signal input. The library reconstructs full history when it updates, so
this is useful for research extension but not a point-in-time live signal feed.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Sequence

import norgatedata
import numpy as np
import pandas as pd
from IPython.display import display

from alpha.data import load_daily_kenneth_french_momentum_snapshot
from alpha.engine.backtest import run_daily
from alpha.engine.report import save_results
from alpha.engine.strategy import Strategy


SUPPORTED_SHARED_START_DATE_STR = "2007-03-01"
DEFAULT_KENNETH_FRENCH_CACHE_ZIP_PATH_STR = str(
    Path(__file__).resolve().parents[2]
    / "data_cache"
    / "kenneth_french"
    / "F-F_Momentum_Factor_daily_CSV.zip"
)


def default_trade_id_int() -> int:
    return -1


@dataclass(frozen=True)
class PdpTimedByKfMomConfig:
    trade_symbol_str: str = "PDP"
    signal_symbol_str: str = "KF_MOM"
    benchmark_list: tuple[str, ...] = ("$SPX",)
    start_date_str: str = SUPPORTED_SHARED_START_DATE_STR
    end_date_str: str | None = None
    signal_return_lookback_day_int: int = 10
    signal_smoothing_day_int: int = 5
    oversold_rank_threshold_float: float = 0.50
    kenneth_french_cache_zip_path_str: str = DEFAULT_KENNETH_FRENCH_CACHE_ZIP_PATH_STR
    capital_base_float: float = 100_000.0
    slippage_float: float = 0.00025
    commission_per_share_float: float = 0.005
    commission_minimum_float: float = 1.0

    def __post_init__(self) -> None:
        if not self.trade_symbol_str:
            raise ValueError("trade_symbol_str must not be empty.")
        if not self.signal_symbol_str:
            raise ValueError("signal_symbol_str must not be empty.")
        if self.trade_symbol_str == self.signal_symbol_str:
            raise ValueError("trade_symbol_str must differ from signal_symbol_str.")
        if len(set(self.benchmark_list)) != len(self.benchmark_list):
            raise ValueError("benchmark_list contains duplicate symbols.")
        if self.signal_return_lookback_day_int <= 0:
            raise ValueError("signal_return_lookback_day_int must be positive.")
        if self.signal_smoothing_day_int <= 0:
            raise ValueError("signal_smoothing_day_int must be positive.")
        if (
            not np.isfinite(self.oversold_rank_threshold_float)
            or self.oversold_rank_threshold_float <= 0.0
            or self.oversold_rank_threshold_float >= 1.0
        ):
            raise ValueError("oversold_rank_threshold_float must lie in the open interval (0, 1).")
        if not self.kenneth_french_cache_zip_path_str:
            raise ValueError("kenneth_french_cache_zip_path_str must not be empty.")
        if self.capital_base_float <= 0.0:
            raise ValueError("capital_base_float must be positive.")
        if self.slippage_float < 0.0:
            raise ValueError("slippage_float must be non-negative.")
        if self.commission_per_share_float < 0.0:
            raise ValueError("commission_per_share_float must be non-negative.")
        if self.commission_minimum_float < 0.0:
            raise ValueError("commission_minimum_float must be non-negative.")


DEFAULT_CONFIG = PdpTimedByKfMomConfig()

__all__ = [
    "DEFAULT_CONFIG",
    "PdpTimedByKfMomConfig",
    "PdpTimedByKfMomStrategy",
    "SUPPORTED_SHARED_START_DATE_STR",
    "compute_kf_mom_proxy_signal_df",
    "get_pdp_timed_by_kf_mom_data",
]


def _resolve_requested_end_date_ts(config: PdpTimedByKfMomConfig) -> pd.Timestamp | None:
    if config.end_date_str is None:
        return None
    return pd.Timestamp(config.end_date_str)


def validate_requested_start_date_ts(config: PdpTimedByKfMomConfig) -> pd.Timestamp:
    requested_start_date_ts = pd.Timestamp(config.start_date_str)
    supported_start_date_ts = pd.Timestamp(SUPPORTED_SHARED_START_DATE_STR)

    if requested_start_date_ts < supported_start_date_ts:
        raise ValueError(
            "Requested start_date_str lies outside the supported PDP/KF_MOM shared window. "
            f"Expected start_date_str >= {SUPPORTED_SHARED_START_DATE_STR}."
        )
    return requested_start_date_ts


def load_execution_price_df(
    trade_symbol_str: str,
    benchmark_list: Sequence[str],
    start_date_str: str,
    end_date_str: str | None,
) -> pd.DataFrame:
    execution_frame_list: list[pd.DataFrame] = []
    symbol_list = [trade_symbol_str] + list(benchmark_list)

    for symbol_str in symbol_list:
        adjustment_type = (
            norgatedata.StockPriceAdjustmentType.CAPITALSPECIAL
            if symbol_str == trade_symbol_str
            else norgatedata.StockPriceAdjustmentType.TOTALRETURN
        )
        price_timeseries_kwargs = dict(
            stock_price_adjustment_setting=adjustment_type,
            padding_setting=norgatedata.PaddingType.ALLMARKETDAYS,
            start_date=start_date_str,
            timeseriesformat="pandas-dataframe",
        )
        if end_date_str is not None:
            price_timeseries_kwargs["end_date"] = end_date_str

        price_df = norgatedata.price_timeseries(symbol_str, **price_timeseries_kwargs)
        if len(price_df) == 0:
            continue

        price_df.columns = pd.MultiIndex.from_tuples([(symbol_str, field_str) for field_str in price_df.columns])
        execution_frame_list.append(price_df)

    if len(execution_frame_list) == 0:
        raise RuntimeError("No execution price data was loaded.")

    execution_price_df = pd.concat(execution_frame_list, axis=1).sort_index()
    required_trade_key_list = [
        (trade_symbol_str, "Open"),
        (trade_symbol_str, "High"),
        (trade_symbol_str, "Low"),
        (trade_symbol_str, "Close"),
    ]
    missing_trade_key_list = [key_tup for key_tup in required_trade_key_list if key_tup not in execution_price_df.columns]
    if len(missing_trade_key_list) > 0:
        raise RuntimeError(f"Missing execution OHLC fields: {missing_trade_key_list}")

    return execution_price_df


def load_signal_factor_return_ser(
    config: PdpTimedByKfMomConfig,
    as_of_ts: datetime | None,
) -> pd.Series:
    signal_snapshot_obj = load_daily_kenneth_french_momentum_snapshot(
        cache_zip_path_str=config.kenneth_french_cache_zip_path_str,
        as_of_ts=as_of_ts,
        mode_str="backtest",
    )
    factor_return_ser = signal_snapshot_obj.value_ser.astype(float).sort_index()
    factor_return_ser.name = config.signal_symbol_str
    return factor_return_ser


def merge_signal_return_into_pricing_data_df(
    execution_price_df: pd.DataFrame,
    signal_symbol_str: str,
    factor_return_ser: pd.Series,
) -> pd.DataFrame:
    pricing_data_df = execution_price_df.copy()
    aligned_factor_return_ser = factor_return_ser.reindex(pricing_data_df.index)
    if aligned_factor_return_ser.isna().any():
        missing_signal_index = aligned_factor_return_ser.index[aligned_factor_return_ser.isna()]
        missing_signal_preview_list = [pd.Timestamp(bar_ts).strftime("%Y-%m-%d") for bar_ts in missing_signal_index[:5]]
        raise RuntimeError(
            f"SignalReturn alignment failed for {signal_symbol_str}. "
            f"First missing dates: {missing_signal_preview_list}"
        )

    pricing_data_df[(signal_symbol_str, "SignalReturn")] = aligned_factor_return_ser.astype(float)
    pricing_data_df = pricing_data_df.sort_index(axis=1)
    return pricing_data_df


def get_pdp_timed_by_kf_mom_data(
    config: PdpTimedByKfMomConfig = DEFAULT_CONFIG,
) -> pd.DataFrame:
    requested_start_date_ts = validate_requested_start_date_ts(config)
    requested_end_date_ts = _resolve_requested_end_date_ts(config)
    requested_end_datetime_ts = None
    if requested_end_date_ts is not None:
        if requested_start_date_ts > requested_end_date_ts:
            raise ValueError("start_date_str must be earlier than or equal to end_date_str.")
        requested_end_datetime_ts = datetime.combine(
            requested_end_date_ts.date(),
            datetime.min.time(),
            tzinfo=UTC,
        )

    execution_price_df = load_execution_price_df(
        trade_symbol_str=config.trade_symbol_str,
        benchmark_list=config.benchmark_list,
        start_date_str=requested_start_date_ts.strftime("%Y-%m-%d"),
        end_date_str=None if requested_end_date_ts is None else requested_end_date_ts.strftime("%Y-%m-%d"),
    )
    factor_return_ser = load_signal_factor_return_ser(
        config=config,
        as_of_ts=requested_end_datetime_ts,
    )

    available_start_date_ts = max(
        pd.Timestamp(execution_price_df.index.min()),
        pd.Timestamp(factor_return_ser.index.min()),
        pd.Timestamp(SUPPORTED_SHARED_START_DATE_STR),
    )
    available_end_date_ts = min(
        pd.Timestamp(execution_price_df.index.max()),
        pd.Timestamp(factor_return_ser.index.max()),
    )

    if requested_start_date_ts < available_start_date_ts:
        raise ValueError(
            "Requested start_date_str lies outside the supported PDP/KF_MOM shared window. "
            f"Expected start_date_str >= {available_start_date_ts.strftime('%Y-%m-%d')}."
        )
    if requested_end_date_ts is not None and requested_end_date_ts > available_end_date_ts:
        raise ValueError(
            "Requested end_date_str lies outside the supported PDP/KF_MOM shared window. "
            f"Expected end_date_str <= {available_end_date_ts.strftime('%Y-%m-%d')}."
        )

    effective_end_date_ts = available_end_date_ts if requested_end_date_ts is None else requested_end_date_ts
    execution_price_df = execution_price_df.loc[requested_start_date_ts:effective_end_date_ts]
    factor_return_ser = factor_return_ser.loc[requested_start_date_ts:effective_end_date_ts]
    if len(execution_price_df.index) == 0:
        raise RuntimeError("No execution price rows remain after applying the requested PDP/KF_MOM window.")
    if len(factor_return_ser.index) == 0:
        raise RuntimeError("No Kenneth French signal rows remain after applying the requested PDP/KF_MOM window.")

    pricing_data_df = merge_signal_return_into_pricing_data_df(
        execution_price_df=execution_price_df,
        signal_symbol_str=config.signal_symbol_str,
        factor_return_ser=factor_return_ser,
    )
    return pricing_data_df


def compute_kf_mom_proxy_signal_df(
    factor_return_ser: pd.Series,
    config: PdpTimedByKfMomConfig = DEFAULT_CONFIG,
) -> pd.DataFrame:
    factor_return_ser = pd.Series(factor_return_ser, copy=True).astype(float)

    # *** CRITICAL*** The synthetic factor index must be compounded only from
    # returns observed up to each bar; it is used solely to express the 10-day
    # factor performance in ETF-like return space without peeking ahead.
    factor_index_ser = (1.0 + factor_return_ser).cumprod() * 100.0

    # *** CRITICAL*** The 10-day proxy return must use a trailing shift so the
    # current signal never references future factor performance.
    proxy_return_10d_ser = (
        factor_index_ser
        / factor_index_ser.shift(config.signal_return_lookback_day_int)
    ) - 1.0

    # *** CRITICAL*** The 5-day smoothing must remain a trailing rolling mean
    # of already-observed 10-day proxy returns only.
    proxy_return_10d_sma5_ser = proxy_return_10d_ser.rolling(
        window=config.signal_smoothing_day_int,
        min_periods=config.signal_smoothing_day_int,
    ).mean()

    proxy_rank_pct_ser = pd.Series(np.nan, index=factor_return_ser.index, dtype=float)
    historical_proxy_smooth_value_list: list[float] = []

    for bar_ts, proxy_smooth_value in proxy_return_10d_sma5_ser.items():
        proxy_smooth_float = float(proxy_smooth_value)
        if not np.isfinite(proxy_smooth_float):
            continue

        # *** CRITICAL*** The expanding percentile rank must use only values
        # available up to the current bar, including the current observation.
        historical_proxy_smooth_value_list.append(proxy_smooth_float)
        historical_proxy_smooth_vec = np.asarray(historical_proxy_smooth_value_list, dtype=float)
        proxy_rank_pct_ser.loc[bar_ts] = float(np.mean(historical_proxy_smooth_vec <= proxy_smooth_float))

    target_weight_ser = pd.Series(np.nan, index=factor_return_ser.index, dtype=float)
    valid_rank_bool_ser = proxy_rank_pct_ser.notna()
    target_weight_ser.loc[valid_rank_bool_ser] = (
        proxy_rank_pct_ser.loc[valid_rank_bool_ser] < config.oversold_rank_threshold_float
    ).astype(float)

    signal_feature_df = pd.DataFrame(
        {
            "factor_index_ser": factor_index_ser,
            "proxy_return_10d_ser": proxy_return_10d_ser,
            "proxy_return_10d_sma5_ser": proxy_return_10d_sma5_ser,
            "proxy_rank_pct_ser": proxy_rank_pct_ser,
            "target_weight_ser": target_weight_ser,
        },
        index=factor_return_ser.index,
    )
    return signal_feature_df


class PdpTimedByKfMomStrategy(Strategy):
    """
    Long/flat PDP strategy gated by mean-reversion in Kenneth French daily Mom.

    At close t:

        target_weight_t = 1[proxy_rank_pct_t < 0.50]

    At the next open t + 1:

        q^{target}_{t+1}
            = floor(V_t * target_weight_t / O^{PDP}_{t+1})
    """

    enable_signal_audit = True
    signal_audit_sample_size = 10

    def __init__(
        self,
        name: str,
        benchmarks: Sequence[str],
        trade_symbol_str: str = DEFAULT_CONFIG.trade_symbol_str,
        signal_symbol_str: str = DEFAULT_CONFIG.signal_symbol_str,
        signal_return_lookback_day_int: int = DEFAULT_CONFIG.signal_return_lookback_day_int,
        signal_smoothing_day_int: int = DEFAULT_CONFIG.signal_smoothing_day_int,
        oversold_rank_threshold_float: float = DEFAULT_CONFIG.oversold_rank_threshold_float,
        capital_base: float = DEFAULT_CONFIG.capital_base_float,
        slippage: float = DEFAULT_CONFIG.slippage_float,
        commission_per_share: float = DEFAULT_CONFIG.commission_per_share_float,
        commission_minimum: float = DEFAULT_CONFIG.commission_minimum_float,
    ) -> None:
        super().__init__(
            name=name,
            benchmarks=list(benchmarks),
            capital_base=capital_base,
            slippage=slippage,
            commission_per_share=commission_per_share,
            commission_minimum=commission_minimum,
        )
        if not trade_symbol_str:
            raise ValueError("trade_symbol_str must not be empty.")
        if not signal_symbol_str:
            raise ValueError("signal_symbol_str must not be empty.")
        if trade_symbol_str == signal_symbol_str:
            raise ValueError("trade_symbol_str must differ from signal_symbol_str.")
        if signal_return_lookback_day_int <= 0:
            raise ValueError("signal_return_lookback_day_int must be positive.")
        if signal_smoothing_day_int <= 0:
            raise ValueError("signal_smoothing_day_int must be positive.")
        if (
            not np.isfinite(oversold_rank_threshold_float)
            or oversold_rank_threshold_float <= 0.0
            or oversold_rank_threshold_float >= 1.0
        ):
            raise ValueError("oversold_rank_threshold_float must lie in the open interval (0, 1).")

        self.trade_symbol_str = str(trade_symbol_str)
        self.signal_symbol_str = str(signal_symbol_str)
        self.signal_return_lookback_day_int = int(signal_return_lookback_day_int)
        self.signal_smoothing_day_int = int(signal_smoothing_day_int)
        self.oversold_rank_threshold_float = float(oversold_rank_threshold_float)
        self.trade_id_int = 0
        self.current_trade_id_int = default_trade_id_int()

    def _signal_config(self) -> PdpTimedByKfMomConfig:
        return PdpTimedByKfMomConfig(
            trade_symbol_str=self.trade_symbol_str,
            signal_symbol_str=self.signal_symbol_str,
            benchmark_list=tuple(self._benchmarks),
            signal_return_lookback_day_int=self.signal_return_lookback_day_int,
            signal_smoothing_day_int=self.signal_smoothing_day_int,
            oversold_rank_threshold_float=self.oversold_rank_threshold_float,
            capital_base_float=self._capital_base,
            slippage_float=self._slippage,
            commission_per_share_float=self._commission_per_share,
            commission_minimum_float=self._commission_minimum,
        )

    def compute_signals(self, pricing_data_df: pd.DataFrame) -> pd.DataFrame:
        signal_return_key = (self.signal_symbol_str, "SignalReturn")
        trade_close_key = (self.trade_symbol_str, "Close")

        if signal_return_key not in pricing_data_df.columns:
            raise RuntimeError(f"Missing SignalReturn history for signal symbol {self.signal_symbol_str}.")
        if trade_close_key not in pricing_data_df.columns:
            raise RuntimeError(f"Missing trade close data for {self.trade_symbol_str}.")

        factor_return_ser = pricing_data_df.loc[:, signal_return_key].astype(float)
        signal_data_df = pricing_data_df.copy()
        signal_feature_df = compute_kf_mom_proxy_signal_df(
            factor_return_ser=factor_return_ser,
            config=self._signal_config(),
        )

        signal_feature_frame_list: list[pd.DataFrame] = []
        proxy_feature_name_list = [
            "factor_index_ser",
            "proxy_return_10d_ser",
            "proxy_return_10d_sma5_ser",
            "proxy_rank_pct_ser",
        ]
        proxy_feature_df = signal_feature_df.loc[:, proxy_feature_name_list].copy()
        proxy_feature_df.columns = pd.MultiIndex.from_tuples(
            [(self.signal_symbol_str, field_str) for field_str in proxy_feature_df.columns]
        )
        signal_feature_frame_list.append(proxy_feature_df)

        target_weight_df = signal_feature_df.loc[:, ["target_weight_ser"]].copy()
        target_weight_df.columns = pd.MultiIndex.from_tuples(
            [(self.trade_symbol_str, field_str) for field_str in target_weight_df.columns]
        )
        signal_feature_frame_list.append(target_weight_df)

        return pd.concat([signal_data_df] + signal_feature_frame_list, axis=1).sort_index(axis=1)

    def iterate(self, data_df: pd.DataFrame, close_row_ser: pd.Series, open_price_ser: pd.Series) -> None:
        if close_row_ser is None or data_df is None:
            return

        target_weight_key = (self.trade_symbol_str, "target_weight_ser")
        if target_weight_key not in close_row_ser.index:
            return

        target_weight_float = float(close_row_ser.loc[target_weight_key])
        if not np.isfinite(target_weight_float):
            return

        trade_open_price_float = float(open_price_ser.get(self.trade_symbol_str, np.nan))
        if not np.isfinite(trade_open_price_float) or trade_open_price_float <= 0.0:
            raise RuntimeError(f"Invalid open price for {self.trade_symbol_str} on {self.current_bar}.")

        budget_value_float = float(self.previous_total_value)

        # *** CRITICAL*** The PDP target share count must be sized from the
        # prior portfolio value and the realized current-bar open so the
        # execution math matches q_{t+1}^{target} exactly.
        target_share_int = int(np.floor(budget_value_float * target_weight_float / trade_open_price_float))
        current_share_int = int(self.get_position(self.trade_symbol_str))

        if target_share_int == current_share_int:
            return

        if target_share_int <= 0:
            if current_share_int <= 0:
                return
            self.order_target(
                self.trade_symbol_str,
                0,
                trade_id=self.current_trade_id_int,
            )
            self.current_trade_id_int = default_trade_id_int()
            return

        if current_share_int <= 0 or self.current_trade_id_int == default_trade_id_int():
            self.trade_id_int += 1
            self.current_trade_id_int = self.trade_id_int

        self.order_target(
            self.trade_symbol_str,
            target_share_int,
            trade_id=self.current_trade_id_int,
        )


if __name__ == "__main__":
    config = DEFAULT_CONFIG
    pricing_data_df = get_pdp_timed_by_kf_mom_data(config=config)

    strategy = PdpTimedByKfMomStrategy(
        name="strategy_mo_pdp_timed_by_kf_mom",
        benchmarks=config.benchmark_list,
        trade_symbol_str=config.trade_symbol_str,
        signal_symbol_str=config.signal_symbol_str,
        signal_return_lookback_day_int=config.signal_return_lookback_day_int,
        signal_smoothing_day_int=config.signal_smoothing_day_int,
        oversold_rank_threshold_float=config.oversold_rank_threshold_float,
        capital_base=config.capital_base_float,
        slippage=config.slippage_float,
        commission_per_share=config.commission_per_share_float,
        commission_minimum=config.commission_minimum_float,
    )

    run_daily(
        strategy,
        pricing_data_df,
        calendar=pricing_data_df.index,
        audit_override_bool=None,
    )

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)
    display(strategy.summary)
    display(strategy.summary_trades)
    save_results(strategy)
