"""
Research-only leveraged all-weather risk-budget portfolio.

The source article specifies the universe, risk budgets, volatility target,
leverage caps, financing assumption, and quarterly cadence, but it does not
publish the covariance-estimation window. This BENCH variant makes one explicit
house assumption: a trailing 63-trading-day sample covariance.

Portfolio rule
--------------
For asset return covariance matrix Sigma_t and long-only base weights x_t:

    risk_share_{i,t}
        = x_{i,t} * (Sigma_t x_t)_i / (x_t' Sigma_t x_t)

The solver targets:

    SPY = 30%, TLT = 30%, DBC = 20%, GLD = 20%

Leverage uses a conservative covariance matrix Sigma_t^+ whose negative
correlations are floored at zero:

    gross_t
        = min(
            15% / annualized_volatility(x_t, Sigma_t^+),
            2.0,
            80% / max_i(x_{i,t}),
        )

    target_weight_{i,t} = gross_t * x_{i,t}

Timing and realism
------------------
1. Risk estimates use CAPITALSPECIAL ETF closes through quarter-end T.
2. Target shares are sized from portfolio value and closes known at T.
3. Orders fill through the Vanilla engine at Open T+1.
4. A fixed 2.4% annual financing rate is charged daily on actual negative cash.
5. No pre-inception proxies extend SPY/TLT/DBC/GLD history backward.
6. This module is research-only and is not wired to live releases or pod config.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, replace
from typing import Sequence

import numpy as np
import pandas as pd
from IPython.display import display
from scipy.optimize import minimize

from alpha.engine.backtest import run_daily
from alpha.engine.report import save_results
from alpha.engine.strategy import Strategy
from data.norgate_loader import load_raw_prices


TRADING_DAYS_PER_YEAR_INT = 252
STRATEGY_NAME_STR = "strategy_taa_levered_all_weather"


def default_trade_id_int() -> int:
    return -1


@dataclass(frozen=True)
class LeveredAllWeatherConfig:
    asset_tuple: tuple[str, ...] = ("SPY", "TLT", "DBC", "GLD")
    risk_budget_tuple: tuple[float, ...] = (0.30, 0.30, 0.20, 0.20)
    benchmark_tuple: tuple[str, ...] = ("$SPX",)
    covariance_lookback_day_int: int = 63
    target_annualized_volatility_float: float = 0.15
    max_gross_exposure_float: float = 2.0
    max_asset_weight_float: float = 0.80
    annual_financing_rate_float: float = 0.024
    start_date_str: str = "2005-01-01"
    end_date_str: str | None = None
    capital_base_float: float = 100_000.0
    slippage_float: float = 0.0001
    commission_per_share_float: float = 0.005
    commission_minimum_float: float = 1.0

    def __post_init__(self):
        if len(self.asset_tuple) < 2:
            raise ValueError("asset_tuple must contain at least two assets.")
        if len(set(self.asset_tuple)) != len(self.asset_tuple):
            raise ValueError("asset_tuple contains duplicate symbols.")
        if len(self.risk_budget_tuple) != len(self.asset_tuple):
            raise ValueError("risk_budget_tuple must align one-for-one with asset_tuple.")
        risk_budget_vec = np.asarray(self.risk_budget_tuple, dtype=float)
        if not np.isfinite(risk_budget_vec).all() or np.any(risk_budget_vec <= 0.0):
            raise ValueError("risk_budget_tuple must contain finite positive values.")
        if not np.isclose(float(risk_budget_vec.sum()), 1.0, atol=1e-12):
            raise ValueError("risk_budget_tuple must sum to 1.0.")
        if len(set(self.benchmark_tuple)) != len(self.benchmark_tuple):
            raise ValueError("benchmark_tuple contains duplicate symbols.")
        if set(self.asset_tuple).intersection(self.benchmark_tuple):
            raise ValueError("benchmark_tuple must not overlap asset_tuple.")
        if self.covariance_lookback_day_int < 2:
            raise ValueError("covariance_lookback_day_int must be at least 2.")
        if (
            not np.isfinite(self.target_annualized_volatility_float)
            or self.target_annualized_volatility_float <= 0.0
        ):
            raise ValueError("target_annualized_volatility_float must be positive.")
        if not np.isfinite(self.max_gross_exposure_float) or self.max_gross_exposure_float <= 0.0:
            raise ValueError("max_gross_exposure_float must be positive.")
        if not np.isfinite(self.max_asset_weight_float) or self.max_asset_weight_float <= 0.0:
            raise ValueError("max_asset_weight_float must be positive.")
        if not np.isfinite(self.annual_financing_rate_float) or self.annual_financing_rate_float < 0.0:
            raise ValueError("annual_financing_rate_float must be non-negative.")
        if self.capital_base_float <= 0.0:
            raise ValueError("capital_base_float must be positive.")
        if self.slippage_float < 0.0:
            raise ValueError("slippage_float must be non-negative.")
        if self.commission_per_share_float < 0.0:
            raise ValueError("commission_per_share_float must be non-negative.")
        if self.commission_minimum_float < 0.0:
            raise ValueError("commission_minimum_float must be non-negative.")

    @property
    def risk_budget_ser(self) -> pd.Series:
        return pd.Series(
            self.risk_budget_tuple,
            index=self.asset_tuple,
            dtype=float,
        )


DEFAULT_CONFIG = LeveredAllWeatherConfig()


def get_levered_all_weather_data(
    config: LeveredAllWeatherConfig = DEFAULT_CONFIG,
) -> pd.DataFrame:
    """
    Load CAPITALSPECIAL OHLC for the four ETFs and TOTALRETURN benchmark data.
    """
    return load_raw_prices(
        symbols=list(config.asset_tuple),
        benchmarks=list(config.benchmark_tuple),
        start_date=config.start_date_str,
        end_date=config.end_date_str,
    )


def _flatten_close_df(price_close_df: pd.DataFrame) -> pd.DataFrame:
    flat_close_df = price_close_df.copy()
    if isinstance(flat_close_df.columns, pd.MultiIndex):
        flat_close_df.columns = flat_close_df.columns.get_level_values(0)
    return flat_close_df.astype(float)


def compute_risk_share_ser(
    base_weight_ser: pd.Series,
    covariance_df: pd.DataFrame,
) -> pd.Series:
    """
    Return each asset's percentage contribution to portfolio variance.

        risk_share_i = w_i * (Sigma w)_i / (w' Sigma w)
    """
    aligned_weight_ser = base_weight_ser.reindex(covariance_df.index).astype(float)
    weight_vec = aligned_weight_ser.to_numpy(dtype=float)
    covariance_mat = covariance_df.loc[covariance_df.index, covariance_df.index].to_numpy(dtype=float)
    marginal_variance_vec = covariance_mat @ weight_vec
    portfolio_variance_float = float(weight_vec @ marginal_variance_vec)
    if not np.isfinite(portfolio_variance_float) or portfolio_variance_float <= 0.0:
        raise ValueError("Portfolio variance must be finite and positive.")
    risk_share_vec = weight_vec * marginal_variance_vec / portfolio_variance_float
    return pd.Series(risk_share_vec, index=covariance_df.index, dtype=float)


def solve_risk_budget_weight_ser(
    covariance_df: pd.DataFrame,
    risk_budget_ser: pd.Series,
    convergence_tolerance_float: float = 1e-5,
    max_iteration_int: int = 1_000,
) -> pd.Series:
    """
    Solve long-only risk-budget weights with a bounded convex optimizer.

    The convex objective is:

        min_x 0.5 * x' Sigma x - sum_i(b_i * log(x_i))

    Its first-order condition gives:

        x_i * (Sigma x)_i = b_i

    Normalizing x to sum to one preserves percentage risk contributions.
    Sigma is divided by its mean diagonal variance before optimization. This
    numerical conditioning does not change the normalized solution.
    """
    asset_index = pd.Index(covariance_df.index)
    if list(covariance_df.columns) != list(asset_index):
        raise ValueError("covariance_df index and columns must match in the same order.")

    aligned_budget_ser = risk_budget_ser.reindex(asset_index).astype(float)
    if aligned_budget_ser.isna().any():
        raise ValueError("risk_budget_ser is missing covariance assets.")
    if np.any(aligned_budget_ser.to_numpy(dtype=float) <= 0.0):
        raise ValueError("risk_budget_ser must contain positive values.")
    if not np.isclose(float(aligned_budget_ser.sum()), 1.0, atol=1e-12):
        raise ValueError("risk_budget_ser must sum to 1.0.")

    covariance_mat = covariance_df.to_numpy(dtype=float)
    if not np.isfinite(covariance_mat).all():
        raise ValueError("covariance_df must contain only finite values.")
    if not np.allclose(covariance_mat, covariance_mat.T, atol=1e-12):
        raise ValueError("covariance_df must be symmetric.")

    variance_vec = np.diag(covariance_mat)
    if np.any(variance_vec <= 0.0):
        raise ValueError("covariance_df diagonal variances must be positive.")

    covariance_scale_float = float(variance_vec.mean())
    conditioned_covariance_mat = covariance_mat / covariance_scale_float
    conditioned_variance_vec = np.diag(conditioned_covariance_mat)
    risk_budget_vec = aligned_budget_ser.to_numpy(dtype=float)
    initial_positive_weight_vec = np.sqrt(
        risk_budget_vec / conditioned_variance_vec
    )

    def objective_and_gradient_fn(
        positive_weight_vec: np.ndarray,
    ) -> tuple[float, np.ndarray]:
        objective_float = float(
            0.5
            * positive_weight_vec
            @ conditioned_covariance_mat
            @ positive_weight_vec
            - np.sum(risk_budget_vec * np.log(positive_weight_vec))
        )
        gradient_vec = (
            conditioned_covariance_mat @ positive_weight_vec
            - risk_budget_vec / positive_weight_vec
        )
        return objective_float, gradient_vec

    optimization_result_obj = minimize(
        objective_and_gradient_fn,
        initial_positive_weight_vec,
        method="L-BFGS-B",
        jac=True,
        bounds=[(1e-12, None)] * len(asset_index),
        options={
            "ftol": 1e-15,
            "gtol": 1e-12,
            "maxiter": max_iteration_int,
            "maxls": 50,
        },
    )
    normalized_weight_vec = (
        optimization_result_obj.x / optimization_result_obj.x.sum()
    )
    normalized_weight_ser = pd.Series(
        normalized_weight_vec,
        index=asset_index,
        dtype=float,
    )
    risk_share_ser = compute_risk_share_ser(normalized_weight_ser, covariance_df)
    max_error_float = float(
        np.max(
            np.abs(
                risk_share_ser.to_numpy(dtype=float)
                - risk_budget_vec
            )
        )
    )
    if max_error_float > convergence_tolerance_float:
        raise RuntimeError(
            "Risk-budget optimizer exceeded contribution tolerance: "
            f"{max_error_float:.3e} > {convergence_tolerance_float:.3e}; "
            f"optimizer status={optimization_result_obj.message}."
        )
    return normalized_weight_ser


def build_conservative_covariance_df(covariance_df: pd.DataFrame) -> pd.DataFrame:
    """
    Rebuild covariance after flooring every negative off-diagonal correlation at zero.
    """
    asset_index = pd.Index(covariance_df.index)
    covariance_mat = covariance_df.loc[asset_index, asset_index].to_numpy(dtype=float)
    volatility_vec = np.sqrt(np.diag(covariance_mat))
    if not np.isfinite(volatility_vec).all() or np.any(volatility_vec <= 0.0):
        raise ValueError("Covariance diagonal must imply finite positive volatility.")

    correlation_mat = covariance_mat / np.outer(volatility_vec, volatility_vec)
    conservative_correlation_mat = np.maximum(correlation_mat, 0.0)
    np.fill_diagonal(conservative_correlation_mat, 1.0)
    conservative_covariance_mat = (
        np.diag(volatility_vec)
        @ conservative_correlation_mat
        @ np.diag(volatility_vec)
    )
    return pd.DataFrame(
        conservative_covariance_mat,
        index=asset_index,
        columns=asset_index,
        dtype=float,
    )


def compute_target_weight_ser(
    covariance_df: pd.DataFrame,
    risk_budget_ser: pd.Series,
    target_annualized_volatility_float: float,
    max_gross_exposure_float: float,
    max_asset_weight_float: float,
) -> tuple[pd.Series, pd.Series, float, float]:
    """
    Solve risk-budget base weights and scale them under the article's guardrails.
    """
    base_weight_ser = solve_risk_budget_weight_ser(
        covariance_df=covariance_df,
        risk_budget_ser=risk_budget_ser,
    )
    conservative_covariance_df = build_conservative_covariance_df(covariance_df)
    base_weight_vec = base_weight_ser.to_numpy(dtype=float)
    conservative_covariance_mat = conservative_covariance_df.to_numpy(dtype=float)
    base_annualized_volatility_float = float(
        np.sqrt(base_weight_vec @ conservative_covariance_mat @ base_weight_vec)
        * np.sqrt(TRADING_DAYS_PER_YEAR_INT)
    )
    if (
        not np.isfinite(base_annualized_volatility_float)
        or base_annualized_volatility_float <= 0.0
    ):
        raise ValueError("Base annualized portfolio volatility must be positive.")

    volatility_scale_float = (
        target_annualized_volatility_float / base_annualized_volatility_float
    )
    asset_cap_scale_float = max_asset_weight_float / float(base_weight_ser.max())
    gross_exposure_float = float(
        min(
            volatility_scale_float,
            max_gross_exposure_float,
            asset_cap_scale_float,
        )
    )
    target_weight_ser = (base_weight_ser * gross_exposure_float).astype(float)

    if float(target_weight_ser.sum()) > max_gross_exposure_float + 1e-12:
        raise ValueError("Target weights exceed max_gross_exposure_float.")
    if float(target_weight_ser.max()) > max_asset_weight_float + 1e-12:
        raise ValueError("Target weights exceed max_asset_weight_float.")

    return (
        target_weight_ser,
        base_weight_ser,
        base_annualized_volatility_float,
        gross_exposure_float,
    )


def compute_daily_target_weight_df(
    price_close_df: pd.DataFrame,
    risk_budget_ser: pd.Series,
    covariance_lookback_day_int: int,
    target_annualized_volatility_float: float,
    max_gross_exposure_float: float,
    max_asset_weight_float: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute causal daily returns and daily target weights from trailing covariance.
    """
    flat_close_df = _flatten_close_df(price_close_df)
    asset_list = list(risk_budget_ser.index)
    flat_close_df = flat_close_df.reindex(columns=asset_list)

    # *** CRITICAL*** Daily returns at T use closes no later than T. The Vanilla
    # engine passes only the row through previous_bar into the T+1 order decision.
    asset_return_df = flat_close_df.pct_change(fill_method=None)
    target_weight_df = pd.DataFrame(
        np.nan,
        index=flat_close_df.index,
        columns=asset_list,
        dtype=float,
    )

    for row_idx_int in range(covariance_lookback_day_int, len(asset_return_df)):
        # *** CRITICAL*** The covariance window ends at row_idx_int and contains
        # only backward-looking returns. Centered or forward windows are forbidden.
        trailing_return_df = asset_return_df.iloc[
            row_idx_int - covariance_lookback_day_int + 1 : row_idx_int + 1
        ]
        if len(trailing_return_df) != covariance_lookback_day_int:
            continue
        if trailing_return_df.isna().any(axis=None):
            continue

        covariance_df = trailing_return_df.cov()
        try:
            target_weight_ser, _base_weight_ser, _base_volatility_float, _gross_exposure_float = (
                compute_target_weight_ser(
                    covariance_df=covariance_df,
                    risk_budget_ser=risk_budget_ser,
                    target_annualized_volatility_float=target_annualized_volatility_float,
                    max_gross_exposure_float=max_gross_exposure_float,
                    max_asset_weight_float=max_asset_weight_float,
                )
            )
        except (ValueError, RuntimeError) as exc:
            decision_date_ts = pd.Timestamp(asset_return_df.index[row_idx_int])
            raise RuntimeError(
                f"Risk-budget sizing failed on {decision_date_ts.date()}: {exc}"
            ) from exc
        target_weight_df.loc[asset_return_df.index[row_idx_int], asset_list] = (
            target_weight_ser.reindex(asset_list)
        )

    return asset_return_df, target_weight_df


def get_first_actionable_rebalance_ts(
    pricing_data_df: pd.DataFrame,
    config: LeveredAllWeatherConfig = DEFAULT_CONFIG,
) -> pd.Timestamp:
    """
    Return the first quarter-open whose previous close has valid target weights.
    """
    asset_close_key_list = [(asset_str, "Close") for asset_str in config.asset_tuple]
    missing_key_list = [
        close_key_tuple
        for close_key_tuple in asset_close_key_list
        if close_key_tuple not in pricing_data_df.columns
    ]
    if missing_key_list:
        raise RuntimeError(f"Missing close data for {missing_key_list}.")

    _asset_return_df, daily_target_weight_df = compute_daily_target_weight_df(
        price_close_df=pricing_data_df.loc[:, asset_close_key_list],
        risk_budget_ser=config.risk_budget_ser,
        covariance_lookback_day_int=config.covariance_lookback_day_int,
        target_annualized_volatility_float=config.target_annualized_volatility_float,
        max_gross_exposure_float=config.max_gross_exposure_float,
        max_asset_weight_float=config.max_asset_weight_float,
    )
    execution_index = pd.DatetimeIndex(pricing_data_df.index)

    for row_idx_int in range(1, len(execution_index)):
        previous_bar_ts = pd.Timestamp(execution_index[row_idx_int - 1])
        current_bar_ts = pd.Timestamp(execution_index[row_idx_int])
        is_quarter_turn_bool = (
            current_bar_ts.to_period("Q") != previous_bar_ts.to_period("Q")
        )
        if not is_quarter_turn_bool:
            continue
        previous_target_weight_ser = daily_target_weight_df.loc[previous_bar_ts]
        if previous_target_weight_ser.notna().all():
            return current_bar_ts

    raise RuntimeError("No actionable quarterly all-weather rebalance was generated.")


def build_backtest_calendar_idx(
    pricing_data_df: pd.DataFrame,
    first_rebalance_ts: pd.Timestamp,
    backtest_start_date_str: str | None = None,
) -> pd.DatetimeIndex:
    """
    Include one cash-only anchor bar before the first executable bar.

    Vanilla measures its first executable day's return against starting
    capital. ExecutionTimingAnalyzer uses pct_change(), so both paths need the
    same prior cash observation for exact default-cell metric parity.
    """
    pricing_index = pd.DatetimeIndex(pricing_data_df.index)
    requested_start_ts = pd.Timestamp(first_rebalance_ts)
    if backtest_start_date_str is not None:
        requested_start_ts = max(
            requested_start_ts,
            pd.Timestamp(backtest_start_date_str),
        )

    executable_position_int = int(pricing_index.searchsorted(requested_start_ts))
    if executable_position_int >= len(pricing_index):
        raise RuntimeError(
            f"No pricing bar exists on or after requested start {requested_start_ts.date()}."
        )
    anchor_position_int = max(executable_position_int - 1, 0)

    # *** CRITICAL*** The anchor is performance bookkeeping only. It is the
    # immediately preceding observed bar, never a future or backfilled price.
    return pd.DatetimeIndex(pricing_index[anchor_position_int:])


class LeveredAllWeatherStrategy(Strategy):
    """
    Quarterly long-only risk-budget allocator with explicit financing drag.
    """

    enable_signal_audit = True
    signal_audit_sample_size = 5

    def __init__(
        self,
        name: str,
        benchmarks: Sequence[str] | None = None,
        asset_tuple: Sequence[str] = DEFAULT_CONFIG.asset_tuple,
        risk_budget_tuple: Sequence[float] = DEFAULT_CONFIG.risk_budget_tuple,
        covariance_lookback_day_int: int = DEFAULT_CONFIG.covariance_lookback_day_int,
        target_annualized_volatility_float: float = DEFAULT_CONFIG.target_annualized_volatility_float,
        max_gross_exposure_float: float = DEFAULT_CONFIG.max_gross_exposure_float,
        max_asset_weight_float: float = DEFAULT_CONFIG.max_asset_weight_float,
        annual_financing_rate_float: float = DEFAULT_CONFIG.annual_financing_rate_float,
        capital_base: float = DEFAULT_CONFIG.capital_base_float,
        slippage: float = DEFAULT_CONFIG.slippage_float,
        commission_per_share: float = DEFAULT_CONFIG.commission_per_share_float,
        commission_minimum: float = DEFAULT_CONFIG.commission_minimum_float,
    ):
        benchmark_list = [] if benchmarks is None else list(benchmarks)
        super().__init__(
            name=name,
            benchmarks=benchmark_list,
            capital_base=capital_base,
            slippage=slippage,
            commission_per_share=commission_per_share,
            commission_minimum=commission_minimum,
        )
        self.asset_list = list(asset_tuple)
        self.risk_budget_ser = pd.Series(
            tuple(risk_budget_tuple),
            index=self.asset_list,
            dtype=float,
        )
        if len(self.asset_list) != len(set(self.asset_list)):
            raise ValueError("asset_tuple contains duplicate symbols.")
        if self.risk_budget_ser.isna().any() or not np.isclose(
            float(self.risk_budget_ser.sum()),
            1.0,
            atol=1e-12,
        ):
            raise ValueError("risk_budget_tuple must align with assets and sum to 1.0.")

        self.covariance_lookback_day_int = int(covariance_lookback_day_int)
        self.target_annualized_volatility_float = float(target_annualized_volatility_float)
        self.max_gross_exposure_float = float(max_gross_exposure_float)
        self.max_asset_weight_float = float(max_asset_weight_float)
        self.annual_financing_rate_float = float(annual_financing_rate_float)

        self.trade_id_int = 0
        self.current_trade_id_map: defaultdict[str, int] = defaultdict(default_trade_id_int)
        self.current_target_weight_ser = pd.Series(
            [0.0] * len(self.asset_list) + [1.0],
            index=self.asset_list + ["Cash"],
            dtype=float,
        )
        self.daily_target_weight_map: dict[pd.Timestamp, pd.Series] = {}
        self.daily_target_weights = pd.DataFrame(
            columns=self.asset_list + ["Cash"],
            dtype=float,
        )
        self.financing_cost_map: dict[pd.Timestamp, float] = {}
        self.financing_cost_ser = pd.Series(dtype=float)
        self.show_taa_weights_report = True

    def compute_signals(self, pricing_data_df: pd.DataFrame) -> pd.DataFrame:
        signal_data_df = pricing_data_df.copy()
        asset_close_key_list = [(asset_str, "Close") for asset_str in self.asset_list]
        missing_key_list = [
            close_key_tuple
            for close_key_tuple in asset_close_key_list
            if close_key_tuple not in signal_data_df.columns
        ]
        if missing_key_list:
            raise RuntimeError(f"Missing close data for {missing_key_list}.")

        asset_return_df, daily_target_weight_df = compute_daily_target_weight_df(
            price_close_df=signal_data_df.loc[:, asset_close_key_list],
            risk_budget_ser=self.risk_budget_ser,
            covariance_lookback_day_int=self.covariance_lookback_day_int,
            target_annualized_volatility_float=self.target_annualized_volatility_float,
            max_gross_exposure_float=self.max_gross_exposure_float,
            max_asset_weight_float=self.max_asset_weight_float,
        )

        feature_df = pd.DataFrame(index=signal_data_df.index)
        for asset_str in self.asset_list:
            feature_df[(asset_str, "return_ser")] = asset_return_df[asset_str]
            feature_df[(asset_str, "target_weight_ser")] = daily_target_weight_df[asset_str]
        feature_df.columns = pd.MultiIndex.from_tuples(feature_df.columns)
        return pd.concat([signal_data_df, feature_df], axis=1)

    def _current_signal_target_weight_ser(self, close_row_ser: pd.Series) -> pd.Series:
        target_weight_dict: dict[str, float] = {}
        for asset_str in self.asset_list:
            target_weight_dict[asset_str] = float(
                close_row_ser.get((asset_str, "target_weight_ser"), np.nan)
            )
        return pd.Series(target_weight_dict, dtype=float)

    def _record_daily_target_weight_ser(self):
        target_weight_ser = self.current_target_weight_ser.reindex(
            self.asset_list + ["Cash"]
        ).astype(float)
        self.daily_target_weight_map[pd.Timestamp(self.current_bar)] = target_weight_ser
        self.daily_target_weights.loc[
            pd.Timestamp(self.current_bar),
            target_weight_ser.index,
        ] = target_weight_ser

    def _apply_daily_financing_cost(self):
        current_bar_ts = pd.Timestamp(self.current_bar)
        if current_bar_ts in self.financing_cost_map:
            return

        borrowed_cash_float = max(-float(self.cash), 0.0)
        financing_cost_float = (
            borrowed_cash_float
            * self.annual_financing_rate_float
            / TRADING_DAYS_PER_YEAR_INT
        )
        self.cash -= financing_cost_float
        self.total_value -= financing_cost_float
        self.financing_cost_map[current_bar_ts] = financing_cost_float

    def iterate(
        self,
        pricing_history_df: pd.DataFrame,
        close_row_ser: pd.Series,
        open_price_ser: pd.Series,
    ):
        if close_row_ser is None or pricing_history_df is None or self.previous_bar is None:
            return

        self._apply_daily_financing_cost()
        is_quarter_turn_bool = (
            pd.Timestamp(self.current_bar).to_period("Q")
            != pd.Timestamp(self.previous_bar).to_period("Q")
        )
        if not is_quarter_turn_bool:
            self._record_daily_target_weight_ser()
            return

        signal_target_weight_ser = self._current_signal_target_weight_ser(close_row_ser)
        if signal_target_weight_ser.isna().any():
            raise RuntimeError(
                "Quarterly all-weather rebalance has missing target weights on "
                f"decision bar {pd.Timestamp(self.previous_bar).date()}."
            )
        if np.any(signal_target_weight_ser.to_numpy(dtype=float) < 0.0):
            raise ValueError("All-weather target weights must remain long-only.")
        if float(signal_target_weight_ser.sum()) > self.max_gross_exposure_float + 1e-12:
            raise ValueError("All-weather target weights exceed the gross-exposure cap.")
        if float(signal_target_weight_ser.max()) > self.max_asset_weight_float + 1e-12:
            raise ValueError("All-weather target weights exceed the single-asset cap.")

        cash_weight_float = float(1.0 - signal_target_weight_ser.sum())
        self.current_target_weight_ser = pd.concat(
            [
                signal_target_weight_ser,
                pd.Series({"Cash": cash_weight_float}, dtype=float),
            ]
        )
        self._record_daily_target_weight_ser()

        current_position_ser = self.get_positions().reindex(
            self.asset_list,
            fill_value=0.0,
        ).astype(int)
        budget_value_float = float(self.total_value)

        for asset_str in self.asset_list:
            previous_close_float = float(close_row_ser.get((asset_str, "Close"), np.nan))
            if not np.isfinite(previous_close_float) or previous_close_float <= 0.0:
                raise RuntimeError(
                    f"Invalid prior close for target asset {asset_str} on {self.previous_bar}."
                )

            target_weight_float = float(signal_target_weight_ser.loc[asset_str])
            current_share_int = int(current_position_ser.loc[asset_str])

            # *** CRITICAL*** Target shares use portfolio value and ETF closes
            # known at quarter-end T. The engine fills these fixed share targets
            # at Open T+1; current-open prices must not influence sizing.
            target_share_int = int(
                np.floor(
                    budget_value_float
                    * target_weight_float
                    / previous_close_float
                )
            )
            if target_share_int == current_share_int:
                continue

            if target_share_int <= 0:
                if current_share_int <= 0:
                    continue
                self.order_target(
                    asset_str,
                    0,
                    trade_id=self.current_trade_id_map[asset_str],
                )
                self.current_trade_id_map[asset_str] = default_trade_id_int()
                continue

            if (
                current_share_int <= 0
                or self.current_trade_id_map[asset_str] == default_trade_id_int()
            ):
                self.trade_id_int += 1
                self.current_trade_id_map[asset_str] = self.trade_id_int

            self.order_target(
                asset_str,
                target_share_int,
                trade_id=self.current_trade_id_map[asset_str],
            )

    def finalize(self, current_data_df: pd.DataFrame):
        if len(self.daily_target_weight_map) > 0:
            self.daily_target_weights = pd.DataFrame.from_dict(
                self.daily_target_weight_map,
                orient="index",
            ).sort_index()
            self.daily_target_weights.index = pd.to_datetime(
                self.daily_target_weights.index
            )
            self.daily_target_weights = self.daily_target_weights.reindex(
                columns=self.asset_list + ["Cash"]
            )

        self.financing_cost_ser = pd.Series(
            self.financing_cost_map,
            dtype=float,
        ).sort_index()
        self.financing_cost_ser.index = pd.to_datetime(self.financing_cost_ser.index)
        self.financing_cost_ser.name = "financing_cost"

    def summarize(self, include_benchmarks: bool = True):
        super().summarize(include_benchmarks=include_benchmarks)

        # The generic exposure metric is closed-trade based and reports zero
        # for a TAA book whose final positions remain open. Report the actual
        # fraction of daily snapshots with invested capital for this strategy.
        invested_day_ser = self.results["portfolio_value"].astype(float).abs() > 1e-12
        exposure_percent_float = float(
            invested_day_ser.mean() * 100.0
        )
        self.summary.loc["Exposure Time [%]", "Strategy"] = exposure_percent_float
        if exposure_percent_float > 0.0:
            annualized_return_percent_float = float(
                self.summary.loc["Return (Ann.) [%]", "Strategy"]
            )
            self.summary.loc[
                "Exposure-Adjusted Return (Ann.) [%]",
                "Strategy",
            ] = annualized_return_percent_float / (exposure_percent_float / 100.0)
        self.summary.loc["Modeled Financing Cost [$]", "Strategy"] = float(
            sum(self.financing_cost_map.values())
        )


def _build_strategy_obj(
    config: LeveredAllWeatherConfig,
    capital_base_float: float,
) -> LeveredAllWeatherStrategy:
    return LeveredAllWeatherStrategy(
        name=STRATEGY_NAME_STR,
        benchmarks=config.benchmark_tuple,
        asset_tuple=config.asset_tuple,
        risk_budget_tuple=config.risk_budget_tuple,
        covariance_lookback_day_int=config.covariance_lookback_day_int,
        target_annualized_volatility_float=config.target_annualized_volatility_float,
        max_gross_exposure_float=config.max_gross_exposure_float,
        max_asset_weight_float=config.max_asset_weight_float,
        annual_financing_rate_float=config.annual_financing_rate_float,
        capital_base=capital_base_float,
        slippage=config.slippage_float,
        commission_per_share=config.commission_per_share_float,
        commission_minimum=config.commission_minimum_float,
    )


def _run_strategy_obj(
    strategy_obj: LeveredAllWeatherStrategy,
    pricing_data_df: pd.DataFrame,
    config: LeveredAllWeatherConfig,
    backtest_start_date_str: str | None,
    show_progress_bool: bool,
):
    first_rebalance_ts = get_first_actionable_rebalance_ts(
        pricing_data_df=pricing_data_df,
        config=config,
    )

    # *** CRITICAL*** Keep the full earlier ETF history for the 63-day
    # covariance warm-up; clip only the executable calendar, retaining one
    # cash-only anchor bar for return measurement.
    calendar_idx = build_backtest_calendar_idx(
        pricing_data_df=pricing_data_df,
        first_rebalance_ts=first_rebalance_ts,
        backtest_start_date_str=backtest_start_date_str,
    )
    run_daily(
        strategy_obj,
        pricing_data_df,
        calendar=calendar_idx,
        show_progress=show_progress_bool,
        show_signal_progress_bool=show_progress_bool,
        audit_override_bool=None,
    )


def run_variant(
    show_display_bool: bool = True,
    save_results_bool: bool = True,
    output_dir_str: str = "results",
    backtest_start_date_str: str | None = None,
    capital_base_float: float = DEFAULT_CONFIG.capital_base_float,
    end_date_str: str | None = None,
) -> LeveredAllWeatherStrategy:
    config = (
        DEFAULT_CONFIG
        if end_date_str is None
        else replace(DEFAULT_CONFIG, end_date_str=end_date_str)
    )
    pricing_data_df = get_levered_all_weather_data(config=config)
    strategy_obj = _build_strategy_obj(
        config=config,
        capital_base_float=capital_base_float,
    )
    _run_strategy_obj(
        strategy_obj=strategy_obj,
        pricing_data_df=pricing_data_df,
        config=config,
        backtest_start_date_str=backtest_start_date_str,
        show_progress_bool=show_display_bool,
    )

    if show_display_bool:
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 1000)
        display(strategy_obj.summary)
        display(strategy_obj.summary_trades)
        if len(strategy_obj.financing_cost_ser) > 0:
            print(
                "Modeled financing cost:",
                f"${float(strategy_obj.financing_cost_ser.sum()):,.2f}",
            )

    if save_results_bool:
        save_results(strategy_obj, output_dir=output_dir_str)

    return strategy_obj


def build_capacity_analysis_inputs(
    show_display_bool: bool = False,
    backtest_start_date_str: str | None = None,
    capital_base_float: float = DEFAULT_CONFIG.capital_base_float,
    end_date_str: str | None = None,
) -> dict[str, object]:
    config = (
        DEFAULT_CONFIG
        if end_date_str is None
        else replace(DEFAULT_CONFIG, end_date_str=end_date_str)
    )
    pricing_data_df = get_levered_all_weather_data(config=config)
    strategy_obj = _build_strategy_obj(
        config=config,
        capital_base_float=capital_base_float,
    )

    # *** CRITICAL*** CapacityAnalysis uses the same completed next-open ledger,
    # including the same covariance warm-up, quarterly cadence, and financing.
    _run_strategy_obj(
        strategy_obj=strategy_obj,
        pricing_data_df=pricing_data_df,
        config=config,
        backtest_start_date_str=backtest_start_date_str,
        show_progress_bool=show_display_bool,
    )
    if show_display_bool:
        display(strategy_obj.summary)

    strategy_obj._performance_benchmark_symbol_str = str(config.benchmark_tuple[0])
    strategy_obj._performance_benchmark_adjustment_str = "TOTALRETURN"
    return {
        "strategy_obj": strategy_obj,
        "pricing_data_df": pricing_data_df,
        "execution_policy_str": "MOO",
        "impact_profile_str": "MOO_ETF_PROXY",
    }


def build_execution_timing_analysis_inputs() -> dict[str, object]:
    config = DEFAULT_CONFIG
    pricing_data_df = get_levered_all_weather_data(config=config)
    first_rebalance_ts = get_first_actionable_rebalance_ts(
        pricing_data_df=pricing_data_df,
        config=config,
    )

    def strategy_factory_fn() -> LeveredAllWeatherStrategy:
        return _build_strategy_obj(
            config=config,
            capital_base_float=config.capital_base_float,
        )

    calendar_idx = build_backtest_calendar_idx(
        pricing_data_df=pricing_data_df,
        first_rebalance_ts=first_rebalance_ts,
    )
    return {
        "strategy_factory_fn": strategy_factory_fn,
        "pricing_data_df": pricing_data_df,
        "calendar_idx": pd.DatetimeIndex(calendar_idx),
        "order_generation_mode_str": "vanilla_current_bar",
        "risk_model_str": "taa_rebalance",
        "entry_timing_str_tuple": ("same_open", "same_close_moc"),
        "exit_timing_str_tuple": ("same_open", "same_close_moc"),
        "default_entry_timing_str": "same_open",
        "default_exit_timing_str": "same_open",
    }


if __name__ == "__main__":
    run_variant()
