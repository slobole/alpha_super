"""
Multi-asset ARP Core9 pod.

Core formulas
-------------
For each risk asset i at close t:

    r_{i,t}
        = P^{signal}_{i,t} / P^{signal}_{i,t-1} - 1

    sigma^{signal 2}_{i,t}
        = (1 - eta) * sigma^{signal 2}_{i,t-1} + eta * r_{i,t}^2

    phi_{i,t}
        = (1 - eta) * phi_{i,t-1} + sqrt(eta) * (r_{i,t} / sigma^{signal}_{i,t-1})

The ARP exposure before long-only clipping is:

    u^{raw}_t
        = Sigma_t^{-1} * C_t^{-1/2} * phi_t

and the smoothed exposure is:

    u_t
        = (1 - rho) * u_{t-1} + rho * u^{raw}_t

Long-only risk weights use:

    u^+_t = max(u_t, 0)

    sigma^{port}_t
        = sqrt(u_t^{+T} * Sigma_t * C_t * Sigma_t * u^+_t) * sqrt(252)

    lambda_t
        = min(target_vol / sigma^{port}_t, max_gross / sum(u^+_t))

    w^{risk}_t
        = lambda_t * u^+_t

    w^{cash}_t
        = 1 - sum(w^{risk}_t)

Execution at open t + 1 uses previous-close portfolio value:

    q^{target}_{i,t+1}
        = floor(V_t * w_{i,t} / O_{i,t+1})

SignalClose uses TOTALRETURN for signal formation.
Execution OHLC uses CAPITALSPECIAL for tradable ETFs.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
import norgatedata
from IPython.display import display
from tqdm.auto import tqdm

from alpha.engine.backtest import run_daily
from alpha.engine.report import save_results
from alpha.engine.strategy import Strategy


TRADING_DAYS_PER_YEAR_INT = 252


def default_trade_id_int() -> int:
    return -1


@dataclass(frozen=True)
class ArpCore9Config:
    risk_asset_list: tuple[str, ...] = ("SPY", "VEA", "VWO", "IEF", "TLT", "GLD", "DBC", "VNQ")
    cash_proxy_str: str = "BIL"
    benchmark_list: tuple[str, ...] = ("$SPX",)
    eta_float: float = 1.0 / 112.0
    portfolio_smoothing_float: float = 1.0 / 20.0
    vol_lookback_day_int: int = 40
    corr_lookback_week_int: int = 156
    corr_min_period_week_int: int = 52
    target_volatility_float: float = 0.10
    max_gross_weight_float: float = 1.0
    eigenvalue_floor_float: float = 1e-6
    start_date_str: str = "2004-01-01"
    end_date_str: str | None = None
    capital_base_float: float = 100_000.0
    slippage_float: float = 0.0001
    commission_per_share_float: float = 0.005
    commission_minimum_float: float = 1.0

    def __post_init__(self):
        if len(self.risk_asset_list) == 0:
            raise ValueError("risk_asset_list must not be empty.")
        if len(set(self.risk_asset_list)) != len(self.risk_asset_list):
            raise ValueError("risk_asset_list contains duplicate symbols.")
        if self.cash_proxy_str in self.risk_asset_list:
            raise ValueError("cash_proxy_str must not appear inside risk_asset_list.")
        if len(set(self.benchmark_list)) != len(self.benchmark_list):
            raise ValueError("benchmark_list contains duplicate symbols.")
        if not np.isfinite(self.eta_float) or self.eta_float <= 0.0 or self.eta_float > 1.0:
            raise ValueError("eta_float must be in the interval (0, 1].")
        if (
            not np.isfinite(self.portfolio_smoothing_float)
            or self.portfolio_smoothing_float <= 0.0
            or self.portfolio_smoothing_float > 1.0
        ):
            raise ValueError("portfolio_smoothing_float must be in the interval (0, 1].")
        if self.vol_lookback_day_int <= 0:
            raise ValueError("vol_lookback_day_int must be positive.")
        if self.corr_lookback_week_int <= 0:
            raise ValueError("corr_lookback_week_int must be positive.")
        if self.corr_min_period_week_int <= 0:
            raise ValueError("corr_min_period_week_int must be positive.")
        if self.corr_min_period_week_int > self.corr_lookback_week_int:
            raise ValueError("corr_min_period_week_int must be <= corr_lookback_week_int.")
        if not np.isfinite(self.target_volatility_float) or self.target_volatility_float <= 0.0:
            raise ValueError("target_volatility_float must be positive.")
        if (
            not np.isfinite(self.max_gross_weight_float)
            or self.max_gross_weight_float <= 0.0
            or self.max_gross_weight_float > 1.0
        ):
            raise ValueError("max_gross_weight_float must be in the interval (0, 1].")
        if not np.isfinite(self.eigenvalue_floor_float) or self.eigenvalue_floor_float <= 0.0:
            raise ValueError("eigenvalue_floor_float must be positive.")
        if self.capital_base_float <= 0.0:
            raise ValueError("capital_base_float must be positive.")
        if self.slippage_float < 0.0:
            raise ValueError("slippage_float must be non-negative.")
        if self.commission_per_share_float < 0.0:
            raise ValueError("commission_per_share_float must be non-negative.")
        if self.commission_minimum_float < 0.0:
            raise ValueError("commission_minimum_float must be non-negative.")

    @property
    def tradeable_asset_list(self) -> tuple[str, ...]:
        return self.risk_asset_list + (self.cash_proxy_str,)


DEFAULT_CONFIG = ArpCore9Config()


def load_signal_close_df(
    symbol_list: Sequence[str],
    start_date_str: str,
    end_date_str: str | None = None,
) -> pd.DataFrame:
    signal_close_map: dict[str, pd.Series] = {}

    for symbol_str in tqdm(symbol_list, desc="loading signal closes"):
        price_df = norgatedata.price_timeseries(
            symbol_str,
            stock_price_adjustment_setting=norgatedata.StockPriceAdjustmentType.TOTALRETURN,
            padding_setting=norgatedata.PaddingType.ALLMARKETDAYS,
            start_date=start_date_str,
            end_date=end_date_str,
            timeseriesformat="pandas-dataframe",
        )
        if len(price_df) == 0:
            continue
        signal_close_map[symbol_str] = price_df["Close"]

    if len(signal_close_map) == 0:
        raise RuntimeError("No signal close data was loaded.")

    signal_close_df = pd.DataFrame(signal_close_map).sort_index()
    missing_symbol_list = [symbol_str for symbol_str in symbol_list if symbol_str not in signal_close_df.columns]
    if len(missing_symbol_list) > 0:
        raise RuntimeError(f"Missing signal close data for symbols: {missing_symbol_list}")
    return signal_close_df


def load_execution_price_df(
    tradeable_asset_list: Sequence[str],
    benchmark_list: Sequence[str],
    start_date_str: str,
    end_date_str: str | None = None,
) -> pd.DataFrame:
    execution_frame_list: list[pd.DataFrame] = []
    symbol_list = list(dict.fromkeys(list(tradeable_asset_list) + list(benchmark_list)))

    for symbol_str in tqdm(symbol_list, desc="loading execution prices"):
        adjustment_type = (
            norgatedata.StockPriceAdjustmentType.TOTALRETURN
            if symbol_str in benchmark_list
            else norgatedata.StockPriceAdjustmentType.CAPITALSPECIAL
        )
        price_df = norgatedata.price_timeseries(
            symbol_str,
            stock_price_adjustment_setting=adjustment_type,
            padding_setting=norgatedata.PaddingType.ALLMARKETDAYS,
            start_date=start_date_str,
            end_date=end_date_str,
            timeseriesformat="pandas-dataframe",
        )
        if len(price_df) == 0:
            continue

        price_df.columns = pd.MultiIndex.from_tuples([(symbol_str, field_str) for field_str in price_df.columns])
        execution_frame_list.append(price_df)

    if len(execution_frame_list) == 0:
        raise RuntimeError("No execution price data was loaded.")

    return pd.concat(execution_frame_list, axis=1).sort_index()


def get_arp_core9_data(
    config: ArpCore9Config = DEFAULT_CONFIG,
) -> pd.DataFrame:
    signal_close_df = load_signal_close_df(
        symbol_list=config.tradeable_asset_list,
        start_date_str=config.start_date_str,
        end_date_str=config.end_date_str,
    )
    execution_price_df = load_execution_price_df(
        tradeable_asset_list=config.tradeable_asset_list,
        benchmark_list=config.benchmark_list,
        start_date_str=config.start_date_str,
        end_date_str=config.end_date_str,
    )

    signal_feature_df = signal_close_df.copy()
    signal_feature_df.columns = pd.MultiIndex.from_tuples(
        [(symbol_str, "SignalClose") for symbol_str in signal_feature_df.columns]
    )
    pricing_data_df = pd.concat([execution_price_df, signal_feature_df], axis=1).sort_index()
    pricing_data_df = pricing_data_df.sort_index(axis=1)
    return pricing_data_df


def _compute_single_asset_trend_signal_df(
    signal_close_ser: pd.Series,
    eta_float: float,
) -> pd.DataFrame:
    signal_close_ser = pd.Series(signal_close_ser, copy=True).astype(float)

    # *** CRITICAL*** Daily returns must use trailing closes only.
    return_ser = signal_close_ser.pct_change(fill_method=None)
    return_vec = return_ser.to_numpy(dtype=float)

    signal_vol_ser = pd.Series(np.nan, index=signal_close_ser.index, dtype=float)
    normalized_return_ser = pd.Series(np.nan, index=signal_close_ser.index, dtype=float)
    trend_signal_ser = pd.Series(np.nan, index=signal_close_ser.index, dtype=float)

    prior_signal_vol_sq_float = np.nan
    prior_signal_vol_float = np.nan
    prior_trend_signal_float = np.nan
    sqrt_eta_float = float(np.sqrt(eta_float))

    for bar_idx_int, bar_ts in enumerate(signal_close_ser.index):
        return_float = float(return_vec[bar_idx_int])
        if not np.isfinite(return_float):
            continue

        return_sq_float = return_float * return_float
        if np.isfinite(prior_signal_vol_sq_float):
            signal_vol_sq_float = (
                (1.0 - eta_float) * prior_signal_vol_sq_float
                + eta_float * return_sq_float
            )
        else:
            signal_vol_sq_float = return_sq_float

        signal_vol_float = float(np.sqrt(signal_vol_sq_float))
        signal_vol_ser.loc[bar_ts] = signal_vol_float

        # *** CRITICAL*** Trend normalization must use lagged volatility only.
        if np.isfinite(prior_signal_vol_float) and prior_signal_vol_float > 0.0:
            normalized_return_float = return_float / prior_signal_vol_float
            normalized_return_ser.loc[bar_ts] = normalized_return_float

            if np.isfinite(prior_trend_signal_float):
                trend_signal_float = (
                    (1.0 - eta_float) * prior_trend_signal_float
                    + sqrt_eta_float * normalized_return_float
                )
            else:
                trend_signal_float = sqrt_eta_float * normalized_return_float

            trend_signal_ser.loc[bar_ts] = trend_signal_float
            prior_trend_signal_float = trend_signal_float

        prior_signal_vol_sq_float = signal_vol_sq_float
        prior_signal_vol_float = signal_vol_float

    return pd.DataFrame(
        {
            "return_ser": return_ser,
            "signal_vol_ser": signal_vol_ser,
            "normalized_return_ser": normalized_return_ser,
            "trend_signal_ser": trend_signal_ser,
        },
        index=signal_close_ser.index,
    )


def _compute_ewma_risk_vol_df(
    return_df: pd.DataFrame,
    vol_lookback_day_int: int,
) -> pd.DataFrame:
    alpha_vol_float = 1.0 / float(vol_lookback_day_int)
    risk_vol_df = pd.DataFrame(np.nan, index=return_df.index, columns=return_df.columns, dtype=float)

    for asset_str in return_df.columns:
        return_vec = return_df[asset_str].to_numpy(dtype=float)
        prior_vol_sq_float = np.nan

        for bar_idx_int, bar_ts in enumerate(return_df.index):
            return_float = float(return_vec[bar_idx_int])
            if not np.isfinite(return_float):
                continue

            return_sq_float = return_float * return_float
            if np.isfinite(prior_vol_sq_float):
                vol_sq_float = ((1.0 - alpha_vol_float) * prior_vol_sq_float) + (alpha_vol_float * return_sq_float)
            else:
                vol_sq_float = return_sq_float

            risk_vol_df.loc[bar_ts, asset_str] = float(np.sqrt(vol_sq_float))
            prior_vol_sq_float = vol_sq_float

    return risk_vol_df


def compute_matrix_inverse_sqrt_df(
    matrix_df: pd.DataFrame,
    eigenvalue_floor_float: float,
) -> pd.DataFrame:
    matrix_arr = matrix_df.to_numpy(dtype=float)
    symmetric_arr = 0.5 * (matrix_arr + matrix_arr.T)
    eigenvalue_vec, eigenvector_arr = np.linalg.eigh(symmetric_arr)
    clipped_eigenvalue_vec = np.clip(eigenvalue_vec, eigenvalue_floor_float, None)
    inverse_sqrt_diag_arr = np.diag(1.0 / np.sqrt(clipped_eigenvalue_vec))
    inverse_sqrt_arr = eigenvector_arr @ inverse_sqrt_diag_arr @ eigenvector_arr.T
    inverse_sqrt_arr = 0.5 * (inverse_sqrt_arr + inverse_sqrt_arr.T)
    return pd.DataFrame(inverse_sqrt_arr, index=matrix_df.index, columns=matrix_df.columns, dtype=float)


def _sanitize_correlation_df(
    correlation_df: pd.DataFrame,
    asset_list: Sequence[str],
) -> pd.DataFrame:
    correlation_df = correlation_df.reindex(index=asset_list, columns=asset_list)
    correlation_df = correlation_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    correlation_arr = correlation_df.to_numpy(dtype=float)
    correlation_arr = 0.5 * (correlation_arr + correlation_arr.T)
    np.fill_diagonal(correlation_arr, 1.0)
    return pd.DataFrame(correlation_arr, index=asset_list, columns=asset_list, dtype=float)


def _get_weekly_decision_close_df(signal_close_df: pd.DataFrame) -> pd.DataFrame:
    week_period_ser = pd.Series(signal_close_df.index.to_period("W-FRI"), index=signal_close_df.index)
    # *** CRITICAL*** Only completed weeks may enter the weekly correlation
    # estimator. The trailing partial week is excluded to avoid future leakage.
    # A Friday close is allowed immediately because the weekly bar is complete
    # at that close, even when the next week is not yet present in the data.
    next_week_period_ser = week_period_ser.shift(-1)
    friday_close_mask_ser = pd.Series(signal_close_df.index.weekday == 4, index=signal_close_df.index)
    completed_week_close_mask_ser = (
        (next_week_period_ser.notna() & (next_week_period_ser != week_period_ser))
        | friday_close_mask_ser
    )
    weekly_decision_close_df = signal_close_df.loc[completed_week_close_mask_ser.to_numpy()].copy()
    return weekly_decision_close_df


def _build_weekly_correlation_maps(
    weekly_return_df: pd.DataFrame,
    asset_list: Sequence[str],
    corr_lookback_week_int: int,
    corr_min_period_week_int: int,
    eigenvalue_floor_float: float,
) -> tuple[dict[pd.Timestamp, pd.DataFrame], dict[pd.Timestamp, pd.DataFrame]]:
    correlation_map: dict[pd.Timestamp, pd.DataFrame] = {}
    inverse_sqrt_map: dict[pd.Timestamp, pd.DataFrame] = {}
    identity_df = pd.DataFrame(
        np.eye(len(asset_list), dtype=float),
        index=list(asset_list),
        columns=list(asset_list),
        dtype=float,
    )

    for weekly_date_ts in weekly_return_df.index:
        trailing_weekly_return_df = weekly_return_df.loc[:weekly_date_ts].tail(corr_lookback_week_int)
        clean_weekly_return_df = trailing_weekly_return_df.dropna(how="any")

        if len(clean_weekly_return_df) < corr_min_period_week_int:
            correlation_df = identity_df.copy()
            inverse_sqrt_df = identity_df.copy()
        else:
            correlation_df = clean_weekly_return_df.corr()
            correlation_df = _sanitize_correlation_df(correlation_df, asset_list)
            inverse_sqrt_df = compute_matrix_inverse_sqrt_df(correlation_df, eigenvalue_floor_float)

        correlation_map[pd.Timestamp(weekly_date_ts)] = correlation_df
        inverse_sqrt_map[pd.Timestamp(weekly_date_ts)] = inverse_sqrt_df

    return correlation_map, inverse_sqrt_map


def _latest_weekly_matrix_pair(
    current_bar_ts: pd.Timestamp,
    correlation_map: dict[pd.Timestamp, pd.DataFrame],
    inverse_sqrt_map: dict[pd.Timestamp, pd.DataFrame],
    asset_list: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    eligible_weekly_date_list = [
        weekly_date_ts for weekly_date_ts in correlation_map.keys() if weekly_date_ts <= current_bar_ts
    ]
    if len(eligible_weekly_date_list) == 0:
        identity_df = pd.DataFrame(
            np.eye(len(asset_list), dtype=float),
            index=list(asset_list),
            columns=list(asset_list),
            dtype=float,
        )
        return identity_df, identity_df.copy()

    latest_weekly_date_ts = max(eligible_weekly_date_list)
    return correlation_map[latest_weekly_date_ts], inverse_sqrt_map[latest_weekly_date_ts]


def compute_arp_core9_signal_tables(
    signal_close_df: pd.DataFrame,
    config: ArpCore9Config,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    missing_symbol_list = [
        symbol_str for symbol_str in config.tradeable_asset_list if symbol_str not in signal_close_df.columns
    ]
    if len(missing_symbol_list) > 0:
        raise RuntimeError(f"Missing signal_close_df columns for symbols: {missing_symbol_list}")

    risk_signal_close_df = signal_close_df.loc[:, list(config.risk_asset_list)].astype(float)

    return_df = pd.DataFrame(index=risk_signal_close_df.index, columns=risk_signal_close_df.columns, dtype=float)
    signal_vol_df = pd.DataFrame(index=risk_signal_close_df.index, columns=risk_signal_close_df.columns, dtype=float)
    normalized_return_df = pd.DataFrame(index=risk_signal_close_df.index, columns=risk_signal_close_df.columns, dtype=float)
    trend_signal_df = pd.DataFrame(index=risk_signal_close_df.index, columns=risk_signal_close_df.columns, dtype=float)

    for asset_str in config.risk_asset_list:
        asset_signal_df = _compute_single_asset_trend_signal_df(
            signal_close_ser=risk_signal_close_df[asset_str],
            eta_float=config.eta_float,
        )
        return_df[asset_str] = asset_signal_df["return_ser"]
        signal_vol_df[asset_str] = asset_signal_df["signal_vol_ser"]
        normalized_return_df[asset_str] = asset_signal_df["normalized_return_ser"]
        trend_signal_df[asset_str] = asset_signal_df["trend_signal_ser"]

    risk_vol_df = _compute_ewma_risk_vol_df(
        return_df=return_df,
        vol_lookback_day_int=config.vol_lookback_day_int,
    )

    weekly_decision_close_df = _get_weekly_decision_close_df(risk_signal_close_df)
    # *** CRITICAL*** Weekly returns must remain trailing-only and are later
    # carried forward from completed weekly closes to daily decision dates.
    weekly_return_df = weekly_decision_close_df.pct_change(fill_method=None)
    correlation_map, inverse_sqrt_map = _build_weekly_correlation_maps(
        weekly_return_df=weekly_return_df,
        asset_list=config.risk_asset_list,
        corr_lookback_week_int=config.corr_lookback_week_int,
        corr_min_period_week_int=config.corr_min_period_week_int,
        eigenvalue_floor_float=config.eigenvalue_floor_float,
    )

    raw_arp_exposure_df = pd.DataFrame(0.0, index=signal_close_df.index, columns=config.risk_asset_list, dtype=float)
    smoothed_arp_exposure_df = pd.DataFrame(0.0, index=signal_close_df.index, columns=config.risk_asset_list, dtype=float)
    target_weight_df = pd.DataFrame(0.0, index=signal_close_df.index, columns=config.tradeable_asset_list, dtype=float)

    prior_smoothed_exposure_vec = np.zeros(len(config.risk_asset_list), dtype=float)

    for bar_ts in signal_close_df.index:
        correlation_df, inverse_sqrt_df = _latest_weekly_matrix_pair(
            current_bar_ts=pd.Timestamp(bar_ts),
            correlation_map=correlation_map,
            inverse_sqrt_map=inverse_sqrt_map,
            asset_list=config.risk_asset_list,
        )

        trend_signal_vec = trend_signal_df.loc[bar_ts, list(config.risk_asset_list)].to_numpy(dtype=float)
        risk_vol_vec = risk_vol_df.loc[bar_ts, list(config.risk_asset_list)].to_numpy(dtype=float)

        valid_vec = np.isfinite(trend_signal_vec) & np.isfinite(risk_vol_vec) & (risk_vol_vec > 0.0)
        clean_trend_signal_vec = np.where(valid_vec, trend_signal_vec, 0.0)
        clean_risk_vol_vec = np.where(valid_vec, risk_vol_vec, np.nan)
        inverse_risk_vol_vec = np.where(valid_vec, 1.0 / clean_risk_vol_vec, 0.0)

        raw_arp_exposure_vec = (
            np.diag(inverse_risk_vol_vec)
            @ inverse_sqrt_df.to_numpy(dtype=float)
            @ clean_trend_signal_vec
        )
        smoothed_arp_exposure_vec = (
            (1.0 - config.portfolio_smoothing_float) * prior_smoothed_exposure_vec
            + config.portfolio_smoothing_float * raw_arp_exposure_vec
        )

        positive_exposure_vec = np.clip(smoothed_arp_exposure_vec, 0.0, None)
        positive_gross_float = float(np.sum(positive_exposure_vec))
        risk_weight_vec = np.zeros(len(config.risk_asset_list), dtype=float)

        if positive_gross_float > 0.0:
            volatility_diag_arr = np.diag(np.where(np.isfinite(clean_risk_vol_vec), clean_risk_vol_vec, 0.0))
            covariance_arr = volatility_diag_arr @ correlation_df.to_numpy(dtype=float) @ volatility_diag_arr
            annualized_portfolio_vol_float = float(
                np.sqrt(max(float(positive_exposure_vec.T @ covariance_arr @ positive_exposure_vec), 0.0))
                * np.sqrt(TRADING_DAYS_PER_YEAR_INT)
            )

            if np.isfinite(annualized_portfolio_vol_float) and annualized_portfolio_vol_float > 0.0:
                scale_by_vol_float = config.target_volatility_float / annualized_portfolio_vol_float
                scale_by_gross_float = config.max_gross_weight_float / positive_gross_float
                portfolio_scale_float = float(min(scale_by_vol_float, scale_by_gross_float))
                if np.isfinite(portfolio_scale_float) and portfolio_scale_float > 0.0:
                    risk_weight_vec = positive_exposure_vec * portfolio_scale_float
                    risk_weight_vec = np.clip(risk_weight_vec, 0.0, None)
                    risk_weight_sum_float = float(np.sum(risk_weight_vec))
                    if risk_weight_sum_float > config.max_gross_weight_float:
                        risk_weight_vec *= config.max_gross_weight_float / risk_weight_sum_float

        risk_weight_sum_float = float(np.sum(risk_weight_vec))
        cash_weight_float = max(0.0, 1.0 - risk_weight_sum_float)

        raw_arp_exposure_df.loc[bar_ts, list(config.risk_asset_list)] = raw_arp_exposure_vec
        smoothed_arp_exposure_df.loc[bar_ts, list(config.risk_asset_list)] = smoothed_arp_exposure_vec
        target_weight_df.loc[bar_ts, list(config.risk_asset_list)] = risk_weight_vec
        target_weight_df.loc[bar_ts, config.cash_proxy_str] = cash_weight_float

        prior_smoothed_exposure_vec = smoothed_arp_exposure_vec

    return (
        return_df,
        signal_vol_df,
        normalized_return_df,
        trend_signal_df,
        raw_arp_exposure_df,
        smoothed_arp_exposure_df,
        target_weight_df,
    )


class ArpCore9Strategy(Strategy):
    """
    Long-only multi-asset ARP pod with `BIL` as the residual cash proxy.

    The close-to-next-open mapping is:

        signal_t -> target_weight_t -> execute_at_open_{t+1}

    and share sizing uses:

        q^{target}_{i,t+1}
            = floor(V_t * w_{i,t} / O_{i,t+1})
    """

    enable_signal_audit = True
    signal_audit_sample_size = 10

    def __init__(
        self,
        name: str,
        benchmarks: Sequence[str],
        risk_asset_list: Sequence[str] = DEFAULT_CONFIG.risk_asset_list,
        cash_proxy_str: str = DEFAULT_CONFIG.cash_proxy_str,
        eta_float: float = DEFAULT_CONFIG.eta_float,
        portfolio_smoothing_float: float = DEFAULT_CONFIG.portfolio_smoothing_float,
        vol_lookback_day_int: int = DEFAULT_CONFIG.vol_lookback_day_int,
        corr_lookback_week_int: int = DEFAULT_CONFIG.corr_lookback_week_int,
        corr_min_period_week_int: int = DEFAULT_CONFIG.corr_min_period_week_int,
        target_volatility_float: float = DEFAULT_CONFIG.target_volatility_float,
        max_gross_weight_float: float = DEFAULT_CONFIG.max_gross_weight_float,
        eigenvalue_floor_float: float = DEFAULT_CONFIG.eigenvalue_floor_float,
        capital_base: float = DEFAULT_CONFIG.capital_base_float,
        slippage: float = DEFAULT_CONFIG.slippage_float,
        commission_per_share: float = DEFAULT_CONFIG.commission_per_share_float,
        commission_minimum: float = DEFAULT_CONFIG.commission_minimum_float,
    ):
        super().__init__(
            name=name,
            benchmarks=list(benchmarks),
            capital_base=capital_base,
            slippage=slippage,
            commission_per_share=commission_per_share,
            commission_minimum=commission_minimum,
        )
        self.risk_asset_list = list(risk_asset_list)
        self.cash_proxy_str = cash_proxy_str
        self.tradeable_asset_list = self.risk_asset_list + [self.cash_proxy_str]
        self.eta_float = float(eta_float)
        self.portfolio_smoothing_float = float(portfolio_smoothing_float)
        self.vol_lookback_day_int = int(vol_lookback_day_int)
        self.corr_lookback_week_int = int(corr_lookback_week_int)
        self.corr_min_period_week_int = int(corr_min_period_week_int)
        self.target_volatility_float = float(target_volatility_float)
        self.max_gross_weight_float = float(max_gross_weight_float)
        self.eigenvalue_floor_float = float(eigenvalue_floor_float)
        self.trade_id_int = 0
        self.current_trade_map: defaultdict[str, int] = defaultdict(default_trade_id_int)

    def _signal_config(self) -> ArpCore9Config:
        return ArpCore9Config(
            risk_asset_list=tuple(self.risk_asset_list),
            cash_proxy_str=self.cash_proxy_str,
            benchmark_list=tuple(self._benchmarks),
            eta_float=self.eta_float,
            portfolio_smoothing_float=self.portfolio_smoothing_float,
            vol_lookback_day_int=self.vol_lookback_day_int,
            corr_lookback_week_int=self.corr_lookback_week_int,
            corr_min_period_week_int=self.corr_min_period_week_int,
            target_volatility_float=self.target_volatility_float,
            max_gross_weight_float=self.max_gross_weight_float,
            eigenvalue_floor_float=self.eigenvalue_floor_float,
            capital_base_float=self._capital_base,
            slippage_float=self._slippage,
            commission_per_share_float=self._commission_per_share,
            commission_minimum_float=self._commission_minimum,
        )

    def compute_signals(self, pricing_data_df: pd.DataFrame) -> pd.DataFrame:
        signal_close_key_list = [(asset_str, "SignalClose") for asset_str in self.tradeable_asset_list]
        missing_signal_key_list = [key_tup for key_tup in signal_close_key_list if key_tup not in pricing_data_df.columns]
        if len(missing_signal_key_list) > 0:
            raise RuntimeError(f"Missing SignalClose columns: {missing_signal_key_list}")

        signal_close_df = pricing_data_df.xs("SignalClose", axis=1, level=1)[self.tradeable_asset_list].astype(float)
        (
            return_df,
            signal_vol_df,
            normalized_return_df,
            trend_signal_df,
            raw_arp_exposure_df,
            smoothed_arp_exposure_df,
            target_weight_df,
        ) = compute_arp_core9_signal_tables(signal_close_df=signal_close_df, config=self._signal_config())

        feature_frame_list = [pricing_data_df.copy()]
        feature_name_df_map = {
            "return_ser": return_df,
            "signal_vol_ser": signal_vol_df,
            "normalized_return_ser": normalized_return_df,
            "trend_signal_ser": trend_signal_df,
            "raw_arp_exposure_ser": raw_arp_exposure_df,
            "smoothed_arp_exposure_ser": smoothed_arp_exposure_df,
            "target_weight_ser": target_weight_df,
        }

        for field_str, feature_df in feature_name_df_map.items():
            asset_feature_df = feature_df.copy()
            asset_feature_df.columns = pd.MultiIndex.from_tuples(
                [(asset_str, field_str) for asset_str in asset_feature_df.columns]
            )
            feature_frame_list.append(asset_feature_df)

        signal_data_df = pd.concat(feature_frame_list, axis=1).sort_index(axis=1)
        return signal_data_df

    def get_target_weight_ser(self, close_row_ser: pd.Series) -> pd.Series:
        target_weight_map: dict[str, float] = {}
        for asset_str in self.tradeable_asset_list:
            key_tup = (asset_str, "target_weight_ser")
            target_weight_float = float(close_row_ser.get(key_tup, 0.0))
            if not np.isfinite(target_weight_float):
                target_weight_float = 0.0
            target_weight_map[asset_str] = target_weight_float
        return pd.Series(target_weight_map, dtype=float)

    def get_target_share_int_map(
        self,
        target_weight_ser: pd.Series,
        open_price_ser: pd.Series,
    ) -> dict[str, int]:
        budget_value_float = float(self.previous_total_value)
        target_share_int_map: dict[str, int] = {}

        for asset_str in self.tradeable_asset_list:
            target_weight_float = float(target_weight_ser.get(asset_str, 0.0))
            if not np.isfinite(target_weight_float) or target_weight_float <= 0.0:
                target_share_int_map[asset_str] = 0
                continue

            open_price_float = float(open_price_ser.get(asset_str, np.nan))
            if not np.isfinite(open_price_float) or open_price_float <= 0.0:
                raise RuntimeError(f"Invalid open price for {asset_str} on {self.current_bar}.")

            target_share_int_map[asset_str] = int(np.floor(budget_value_float * target_weight_float / open_price_float))

        return target_share_int_map

    def iterate(self, data_df: pd.DataFrame, close_row_ser: pd.Series, open_price_ser: pd.Series):
        if close_row_ser is None or data_df is None:
            return

        target_weight_ser = self.get_target_weight_ser(close_row_ser)
        target_share_int_map = self.get_target_share_int_map(target_weight_ser, open_price_ser)
        current_position_ser = self.get_positions().reindex(self.tradeable_asset_list, fill_value=0.0).astype(int)

        # *** CRITICAL*** Reductions are submitted before increases so the
        # rebalance path does not rely on temporary same-open leverage.
        for asset_str in self.tradeable_asset_list:
            current_share_int = int(current_position_ser.loc[asset_str])
            target_share_int = int(target_share_int_map[asset_str])

            if target_share_int >= current_share_int:
                continue

            self.order_target(
                asset_str,
                target_share_int,
                trade_id=self.current_trade_map[asset_str],
            )

            if target_share_int == 0:
                self.current_trade_map[asset_str] = default_trade_id_int()

        for asset_str in self.tradeable_asset_list:
            current_share_int = int(current_position_ser.loc[asset_str])
            target_share_int = int(target_share_int_map[asset_str])

            if target_share_int <= current_share_int:
                continue

            if current_share_int <= 0 or self.current_trade_map[asset_str] == default_trade_id_int():
                self.trade_id_int += 1
                self.current_trade_map[asset_str] = self.trade_id_int

            self.order_target(
                asset_str,
                target_share_int,
                trade_id=self.current_trade_map[asset_str],
            )


if __name__ == "__main__":
    config = DEFAULT_CONFIG
    pricing_data_df = get_arp_core9_data(config=config)

    strategy = ArpCore9Strategy(
        name="strategy_mo_arp_core9",
        benchmarks=config.benchmark_list,
        risk_asset_list=config.risk_asset_list,
        cash_proxy_str=config.cash_proxy_str,
        eta_float=config.eta_float,
        portfolio_smoothing_float=config.portfolio_smoothing_float,
        vol_lookback_day_int=config.vol_lookback_day_int,
        corr_lookback_week_int=config.corr_lookback_week_int,
        corr_min_period_week_int=config.corr_min_period_week_int,
        target_volatility_float=config.target_volatility_float,
        max_gross_weight_float=config.max_gross_weight_float,
        eigenvalue_floor_float=config.eigenvalue_floor_float,
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
