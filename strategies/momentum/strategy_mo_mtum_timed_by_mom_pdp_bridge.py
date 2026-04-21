"""
MTUM timing strategy driven by a bridged MOM -> PDP momentum proxy.

Core formulas
-------------
Let `T_switch = 2021-03-12`, the last available `MOM-202103` bar.

Define the component proxy returns:

    r_t^{mom}
        = Close_t^{MOM} / Close_{t-1}^{MOM} - 1

    r_t^{pdp}
        = Close_t^{PDP} / Close_{t-1}^{PDP} - 1

The bridged daily proxy return is:

    r_t^{bridge}
        = r_t^{mom} * 1[t <= T_switch]
        + r_t^{pdp} * 1[t > T_switch]

To avoid splicing unrelated ETF price levels directly, the strategy converts
the bridged return path into a synthetic continuous proxy index:

    proxy_close_equiv_t
        = 100 * prod_{u <= t}(1 + r_u^{bridge})

The timing oscillator is then:

    proxy_return_{10d,t}
        = proxy_close_equiv_t / proxy_close_equiv_{t-10} - 1

    proxy_smooth_ser_t
        = mean(proxy_return_{10d,t-4:t})

    proxy_rank_pct_t
        = (1 / N_t) * sum_{s <= t} 1[proxy_smooth_ser_s <= proxy_smooth_ser_t]

The traded asset is `MTUM` and the binary target is:

    target_weight_t
        = 1[proxy_rank_pct_t < 0.50]

Execution uses the next daily open:

    q^{target}_{t+1}
        = floor(V_t * target_weight_t / O^{MTUM}_{t+1})
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import norgatedata
import numpy as np
import pandas as pd
from IPython.display import display

from alpha.engine.backtest import run_daily
from alpha.engine.report import save_results
from alpha.engine.strategy import Strategy


SUPPORTED_SHARED_START_DATE_STR = "2013-04-18"
DEFAULT_SIGNAL_SWITCH_DATE_STR = "2021-03-12"


def default_trade_id_int() -> int:
    return -1


@dataclass(frozen=True)
class MtumTimedByMomPdpBridgeConfig:
    trade_symbol_str: str = "MTUM"
    signal_symbol_str: str = "MOM_PDP_BRIDGE"
    primary_signal_symbol_str: str = "MOM-202103"
    fallback_signal_symbol_str: str = "PDP"
    benchmark_list: tuple[str, ...] = ("$SPX",)
    start_date_str: str = SUPPORTED_SHARED_START_DATE_STR
    end_date_str: str | None = None
    signal_switch_date_str: str = DEFAULT_SIGNAL_SWITCH_DATE_STR
    signal_return_lookback_day_int: int = 10
    signal_smoothing_day_int: int = 5
    oversold_rank_threshold_float: float = 0.50
    capital_base_float: float = 100_000.0
    slippage_float: float = 0.00025
    commission_per_share_float: float = 0.005
    commission_minimum_float: float = 1.0

    def __post_init__(self) -> None:
        if not self.trade_symbol_str:
            raise ValueError("trade_symbol_str must not be empty.")
        if not self.signal_symbol_str:
            raise ValueError("signal_symbol_str must not be empty.")
        if not self.primary_signal_symbol_str:
            raise ValueError("primary_signal_symbol_str must not be empty.")
        if not self.fallback_signal_symbol_str:
            raise ValueError("fallback_signal_symbol_str must not be empty.")
        if len(
            {
                self.trade_symbol_str,
                self.signal_symbol_str,
                self.primary_signal_symbol_str,
                self.fallback_signal_symbol_str,
            }
        ) != 4:
            raise ValueError(
                "trade_symbol_str, signal_symbol_str, primary_signal_symbol_str, and "
                "fallback_signal_symbol_str must all be distinct."
            )
        benchmark_symbol_set = set(self.benchmark_list)
        if len(benchmark_symbol_set) != len(self.benchmark_list):
            raise ValueError("benchmark_list contains duplicate symbols.")
        if len(
            benchmark_symbol_set.intersection(
                {
                    self.trade_symbol_str,
                    self.primary_signal_symbol_str,
                    self.fallback_signal_symbol_str,
                }
            )
        ) > 0:
            raise ValueError(
                "benchmark_list must not overlap the trade symbol or raw signal proxy symbols."
            )
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
        if self.capital_base_float <= 0.0:
            raise ValueError("capital_base_float must be positive.")
        if self.slippage_float < 0.0:
            raise ValueError("slippage_float must be non-negative.")
        if self.commission_per_share_float < 0.0:
            raise ValueError("commission_per_share_float must be non-negative.")
        if self.commission_minimum_float < 0.0:
            raise ValueError("commission_minimum_float must be non-negative.")


DEFAULT_CONFIG = MtumTimedByMomPdpBridgeConfig()

__all__ = [
    "DEFAULT_CONFIG",
    "DEFAULT_SIGNAL_SWITCH_DATE_STR",
    "MtumTimedByMomPdpBridgeConfig",
    "MtumTimedByMomPdpBridgeStrategy",
    "SUPPORTED_SHARED_START_DATE_STR",
    "compute_mom_pdp_bridge_signal_df",
    "get_mtum_timed_by_mom_pdp_bridge_data",
]


def _resolve_requested_end_date_ts(config: MtumTimedByMomPdpBridgeConfig) -> pd.Timestamp | None:
    if config.end_date_str is None:
        return None
    return pd.Timestamp(config.end_date_str)


def validate_requested_window(
    config: MtumTimedByMomPdpBridgeConfig,
) -> tuple[pd.Timestamp, pd.Timestamp | None]:
    requested_start_date_ts = pd.Timestamp(config.start_date_str)
    requested_end_date_ts = _resolve_requested_end_date_ts(config)
    supported_start_date_ts = pd.Timestamp(SUPPORTED_SHARED_START_DATE_STR)

    if requested_start_date_ts < supported_start_date_ts:
        raise ValueError(
            "Requested start_date_str lies outside the supported MTUM bridge window. "
            f"Expected start_date_str >= {SUPPORTED_SHARED_START_DATE_STR}."
        )
    if requested_end_date_ts is not None and requested_start_date_ts > requested_end_date_ts:
        raise ValueError("start_date_str must be earlier than or equal to end_date_str.")

    return requested_start_date_ts, requested_end_date_ts


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


def load_signal_close_ser(
    signal_symbol_str: str,
    start_date_str: str,
    end_date_str: str | None,
) -> pd.Series:
    price_timeseries_kwargs = dict(
        stock_price_adjustment_setting=norgatedata.StockPriceAdjustmentType.CAPITALSPECIAL,
        padding_setting=norgatedata.PaddingType.ALLMARKETDAYS,
        start_date=start_date_str,
        timeseriesformat="pandas-dataframe",
    )
    if end_date_str is not None:
        price_timeseries_kwargs["end_date"] = end_date_str

    signal_price_df = norgatedata.price_timeseries(signal_symbol_str, **price_timeseries_kwargs)
    if len(signal_price_df) == 0:
        return pd.Series(dtype=float, name=signal_symbol_str)
    if "Close" not in signal_price_df.columns:
        raise RuntimeError(f"Signal data for {signal_symbol_str} is missing Close.")

    signal_close_ser = signal_price_df["Close"].astype(float).sort_index()
    signal_close_ser.name = signal_symbol_str
    return signal_close_ser


def merge_signal_close_into_pricing_data_df(
    execution_price_df: pd.DataFrame,
    signal_symbol_str: str,
    signal_close_ser: pd.Series,
    missing_ok_bool: bool,
) -> pd.DataFrame:
    pricing_data_df = execution_price_df.copy()
    aligned_signal_close_ser = signal_close_ser.reindex(pricing_data_df.index)
    if not missing_ok_bool and aligned_signal_close_ser.isna().any():
        missing_signal_index = aligned_signal_close_ser.index[aligned_signal_close_ser.isna()]
        missing_signal_preview_list = [pd.Timestamp(bar_ts).strftime("%Y-%m-%d") for bar_ts in missing_signal_index[:5]]
        raise RuntimeError(
            f"SignalClose alignment failed for {signal_symbol_str}. "
            f"First missing dates: {missing_signal_preview_list}"
        )

    pricing_data_df[(signal_symbol_str, "SignalClose")] = aligned_signal_close_ser.astype(float)
    return pricing_data_df.sort_index(axis=1)


def get_mtum_timed_by_mom_pdp_bridge_data(
    config: MtumTimedByMomPdpBridgeConfig = DEFAULT_CONFIG,
) -> pd.DataFrame:
    requested_start_date_ts, requested_end_date_ts = validate_requested_window(config)
    requested_end_date_str = None if requested_end_date_ts is None else requested_end_date_ts.strftime("%Y-%m-%d")

    execution_price_df = load_execution_price_df(
        trade_symbol_str=config.trade_symbol_str,
        benchmark_list=config.benchmark_list,
        start_date_str=requested_start_date_ts.strftime("%Y-%m-%d"),
        end_date_str=requested_end_date_str,
    )
    primary_signal_close_ser = load_signal_close_ser(
        signal_symbol_str=config.primary_signal_symbol_str,
        start_date_str=requested_start_date_ts.strftime("%Y-%m-%d"),
        end_date_str=requested_end_date_str,
    )
    fallback_signal_close_ser = load_signal_close_ser(
        signal_symbol_str=config.fallback_signal_symbol_str,
        start_date_str=requested_start_date_ts.strftime("%Y-%m-%d"),
        end_date_str=requested_end_date_str,
    )

    available_end_candidate_list = [
        pd.Timestamp(execution_price_df.index.max()),
        pd.Timestamp(fallback_signal_close_ser.index.max()),
    ]
    if requested_start_date_ts <= pd.Timestamp(config.signal_switch_date_str) and len(primary_signal_close_ser) == 0:
        raise RuntimeError(
            f"No primary signal data was loaded for {config.primary_signal_symbol_str} in the requested pre-switch window."
        )
    if len(fallback_signal_close_ser) == 0:
        raise RuntimeError(f"No fallback signal data was loaded for {config.fallback_signal_symbol_str}.")

    available_end_date_ts = min(available_end_candidate_list)
    if requested_end_date_ts is not None and requested_end_date_ts > available_end_date_ts:
        raise ValueError(
            "Requested end_date_str lies outside the supported MTUM bridge window. "
            f"Expected end_date_str <= {available_end_date_ts.strftime('%Y-%m-%d')}."
        )

    effective_end_date_ts = available_end_date_ts if requested_end_date_ts is None else requested_end_date_ts
    execution_price_df = execution_price_df.loc[requested_start_date_ts:effective_end_date_ts]
    if len(execution_price_df) == 0:
        raise RuntimeError("No execution price rows remain after applying the requested MTUM bridge window.")

    pricing_data_df = merge_signal_close_into_pricing_data_df(
        execution_price_df=execution_price_df,
        signal_symbol_str=config.primary_signal_symbol_str,
        signal_close_ser=primary_signal_close_ser.loc[:effective_end_date_ts],
        missing_ok_bool=True,
    )
    pricing_data_df = merge_signal_close_into_pricing_data_df(
        execution_price_df=pricing_data_df,
        signal_symbol_str=config.fallback_signal_symbol_str,
        signal_close_ser=fallback_signal_close_ser.loc[:effective_end_date_ts],
        missing_ok_bool=False,
    )
    return pricing_data_df


def compute_mom_pdp_bridge_signal_df(
    primary_signal_close_ser: pd.Series,
    fallback_signal_close_ser: pd.Series,
    config: MtumTimedByMomPdpBridgeConfig = DEFAULT_CONFIG,
) -> pd.DataFrame:
    primary_signal_close_ser = pd.Series(primary_signal_close_ser, copy=True).astype(float)
    fallback_signal_close_ser = pd.Series(fallback_signal_close_ser, copy=True).astype(float)
    if not primary_signal_close_ser.index.equals(fallback_signal_close_ser.index):
        raise ValueError("primary_signal_close_ser and fallback_signal_close_ser must share the same index.")

    signal_index = pd.DatetimeIndex(primary_signal_close_ser.index)
    signal_switch_date_ts = pd.Timestamp(config.signal_switch_date_str)
    use_fallback_bool_ser = pd.Series(signal_index > signal_switch_date_ts, index=signal_index, dtype=bool)

    primary_component_return_ser = primary_signal_close_ser.pct_change(fill_method=None)
    fallback_component_return_ser = fallback_signal_close_ser.pct_change(fill_method=None)
    proxy_component_return_ser = pd.Series(np.nan, index=signal_index, dtype=float)
    proxy_component_return_ser.loc[~use_fallback_bool_ser] = primary_component_return_ser.loc[~use_fallback_bool_ser]
    proxy_component_return_ser.loc[use_fallback_bool_ser] = fallback_component_return_ser.loc[use_fallback_bool_ser]

    pre_switch_missing_mask_ser = (~use_fallback_bool_ser) & primary_signal_close_ser.isna()
    if bool(pre_switch_missing_mask_ser.any()):
        first_missing_bar_ts = pd.Timestamp(pre_switch_missing_mask_ser.index[pre_switch_missing_mask_ser.argmax()])
        raise RuntimeError(
            f"Primary proxy {config.primary_signal_symbol_str} is missing pre-switch close data at {first_missing_bar_ts.date()}."
        )
    post_switch_missing_mask_ser = use_fallback_bool_ser & fallback_signal_close_ser.isna()
    if bool(post_switch_missing_mask_ser.any()):
        first_missing_bar_ts = pd.Timestamp(post_switch_missing_mask_ser.index[post_switch_missing_mask_ser.argmax()])
        raise RuntimeError(
            f"Fallback proxy {config.fallback_signal_symbol_str} is missing post-switch close data at {first_missing_bar_ts.date()}."
        )

    # *** CRITICAL*** The bridged proxy index must be compounded from the
    # active proxy's daily return path. Directly splicing unrelated ETF price
    # levels would create artificial jumps and distort the 10-day oscillator.
    proxy_close_equiv_ser = (1.0 + proxy_component_return_ser.fillna(0.0)).cumprod() * 100.0

    # *** CRITICAL*** The 10-day proxy return must use a trailing shift so the
    # current signal never references future bridged proxy levels.
    proxy_return_10d_ser = (
        proxy_close_equiv_ser
        / proxy_close_equiv_ser.shift(config.signal_return_lookback_day_int)
    ) - 1.0

    # *** CRITICAL*** The 5-day smoothing must remain a trailing rolling mean
    # of already-observed 10-day proxy returns only.
    proxy_return_10d_sma5_ser = proxy_return_10d_ser.rolling(
        window=config.signal_smoothing_day_int,
        min_periods=config.signal_smoothing_day_int,
    ).mean()

    proxy_rank_pct_ser = pd.Series(np.nan, index=signal_index, dtype=float)
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

    target_weight_ser = pd.Series(np.nan, index=signal_index, dtype=float)
    valid_rank_bool_ser = proxy_rank_pct_ser.notna()
    target_weight_ser.loc[valid_rank_bool_ser] = (
        proxy_rank_pct_ser.loc[valid_rank_bool_ser] < config.oversold_rank_threshold_float
    ).astype(float)

    signal_feature_df = pd.DataFrame(
        {
            "use_fallback_bool_ser": use_fallback_bool_ser.astype(float),
            "proxy_component_return_ser": proxy_component_return_ser,
            "proxy_close_equiv_ser": proxy_close_equiv_ser,
            "proxy_return_10d_ser": proxy_return_10d_ser,
            "proxy_return_10d_sma5_ser": proxy_return_10d_sma5_ser,
            "proxy_rank_pct_ser": proxy_rank_pct_ser,
            "target_weight_ser": target_weight_ser,
        },
        index=signal_index,
    )
    return signal_feature_df


class MtumTimedByMomPdpBridgeStrategy(Strategy):
    """
    Long/flat MTUM strategy gated by a bridged MOM -> PDP proxy oscillator.

    At close t:

        target_weight_t = 1[proxy_rank_pct_t < 0.50]

    At the next open t + 1:

        q^{target}_{t+1}
            = floor(V_t * target_weight_t / O^{MTUM}_{t+1})
    """

    enable_signal_audit = True
    signal_audit_sample_size = 10

    def __init__(
        self,
        name: str,
        benchmarks: Sequence[str],
        trade_symbol_str: str = DEFAULT_CONFIG.trade_symbol_str,
        signal_symbol_str: str = DEFAULT_CONFIG.signal_symbol_str,
        primary_signal_symbol_str: str = DEFAULT_CONFIG.primary_signal_symbol_str,
        fallback_signal_symbol_str: str = DEFAULT_CONFIG.fallback_signal_symbol_str,
        signal_switch_date_str: str = DEFAULT_CONFIG.signal_switch_date_str,
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
        if not primary_signal_symbol_str:
            raise ValueError("primary_signal_symbol_str must not be empty.")
        if not fallback_signal_symbol_str:
            raise ValueError("fallback_signal_symbol_str must not be empty.")
        if len({trade_symbol_str, signal_symbol_str, primary_signal_symbol_str, fallback_signal_symbol_str}) != 4:
            raise ValueError(
                "trade_symbol_str, signal_symbol_str, primary_signal_symbol_str, and "
                "fallback_signal_symbol_str must all be distinct."
            )
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
        self.primary_signal_symbol_str = str(primary_signal_symbol_str)
        self.fallback_signal_symbol_str = str(fallback_signal_symbol_str)
        self.signal_switch_date_str = str(signal_switch_date_str)
        self.signal_return_lookback_day_int = int(signal_return_lookback_day_int)
        self.signal_smoothing_day_int = int(signal_smoothing_day_int)
        self.oversold_rank_threshold_float = float(oversold_rank_threshold_float)
        self.trade_id_int = 0
        self.current_trade_id_int = default_trade_id_int()

    def _signal_config(self) -> MtumTimedByMomPdpBridgeConfig:
        return MtumTimedByMomPdpBridgeConfig(
            trade_symbol_str=self.trade_symbol_str,
            signal_symbol_str=self.signal_symbol_str,
            primary_signal_symbol_str=self.primary_signal_symbol_str,
            fallback_signal_symbol_str=self.fallback_signal_symbol_str,
            benchmark_list=tuple(self._benchmarks),
            signal_switch_date_str=self.signal_switch_date_str,
            signal_return_lookback_day_int=self.signal_return_lookback_day_int,
            signal_smoothing_day_int=self.signal_smoothing_day_int,
            oversold_rank_threshold_float=self.oversold_rank_threshold_float,
            capital_base_float=self._capital_base,
            slippage_float=self._slippage,
            commission_per_share_float=self._commission_per_share,
            commission_minimum_float=self._commission_minimum,
        )

    def compute_signals(self, pricing_data_df: pd.DataFrame) -> pd.DataFrame:
        primary_signal_close_key = (self.primary_signal_symbol_str, "SignalClose")
        fallback_signal_close_key = (self.fallback_signal_symbol_str, "SignalClose")
        trade_close_key = (self.trade_symbol_str, "Close")

        if primary_signal_close_key not in pricing_data_df.columns:
            raise RuntimeError(f"Missing SignalClose history for {self.primary_signal_symbol_str}.")
        if fallback_signal_close_key not in pricing_data_df.columns:
            raise RuntimeError(f"Missing SignalClose history for {self.fallback_signal_symbol_str}.")
        if trade_close_key not in pricing_data_df.columns:
            raise RuntimeError(f"Missing trade close data for {self.trade_symbol_str}.")

        primary_signal_close_ser = pricing_data_df.loc[:, primary_signal_close_key].astype(float)
        fallback_signal_close_ser = pricing_data_df.loc[:, fallback_signal_close_key].astype(float)

        signal_data_df = pricing_data_df.copy()
        signal_feature_df = compute_mom_pdp_bridge_signal_df(
            primary_signal_close_ser=primary_signal_close_ser,
            fallback_signal_close_ser=fallback_signal_close_ser,
            config=self._signal_config(),
        )

        signal_feature_name_list = [
            "use_fallback_bool_ser",
            "proxy_component_return_ser",
            "proxy_close_equiv_ser",
            "proxy_return_10d_ser",
            "proxy_return_10d_sma5_ser",
            "proxy_rank_pct_ser",
        ]
        proxy_feature_df = signal_feature_df.loc[:, signal_feature_name_list].copy()
        proxy_feature_df.columns = pd.MultiIndex.from_tuples(
            [(self.signal_symbol_str, field_str) for field_str in proxy_feature_df.columns]
        )

        target_weight_df = signal_feature_df.loc[:, ["target_weight_ser"]].copy()
        target_weight_df.columns = pd.MultiIndex.from_tuples(
            [(self.trade_symbol_str, field_str) for field_str in target_weight_df.columns]
        )

        return pd.concat([signal_data_df, proxy_feature_df, target_weight_df], axis=1).sort_index(axis=1)

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

        # *** CRITICAL*** The MTUM target share count must be sized from the
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
    pricing_data_df = get_mtum_timed_by_mom_pdp_bridge_data(config=config)

    strategy = MtumTimedByMomPdpBridgeStrategy(
        name="strategy_mo_mtum_timed_by_mom_pdp_bridge",
        benchmarks=config.benchmark_list,
        trade_symbol_str=config.trade_symbol_str,
        signal_symbol_str=config.signal_symbol_str,
        primary_signal_symbol_str=config.primary_signal_symbol_str,
        fallback_signal_symbol_str=config.fallback_signal_symbol_str,
        signal_switch_date_str=config.signal_switch_date_str,
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
