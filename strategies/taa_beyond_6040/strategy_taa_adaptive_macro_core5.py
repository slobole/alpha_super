"""Adaptive Macro CORE5 with independent sleeves and a DBC short overlay.

Five fixed 20% macro sleeves are evaluated independently:

    SPY, IEF, GLD, DBC, UUP

For each asset, Total Return Close_T drives a drawdown-adaptive moving average.
The sleeve owns its ETF when SMA10_T > AMA_T and otherwise moves to BIL. There
is no relative ranking and inactive sleeve capital is not redistributed among
the active risk assets.

DBC alone may carry an additional volatility-scaled short when SMA10_T < AMA_T:

    short_weight_T = -min(10%, 2.5% / annualized_volatility_63_T)

The long/BIL book always targets 100% of NAV. Short-sale proceeds remain cash
and do not finance larger long sleeves. A fixed 1% annual DBC borrow baseline
is accrued from cash and NAV while the short is held. Signals are decided after
Close_T and queued market orders fill at Open_(T+1). Rebalancing occurs only on
a change in one of the five long states, at month-end, or on first initialization.

PM_READY certifies the engine and portfolio-manager contract only. This module
contains no LIVE, broker, scheduler, release, or account-route wiring.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

import numpy as np
import pandas as pd
from IPython.display import display

from alpha.engine.backtest import run_daily
from alpha.engine.report import save_results
from alpha.engine.strategy import Strategy
from data.norgate_loader import load_raw_prices


STRATEGY_NAME_STR = "strategy_taa_adaptive_macro_core5"
STRATEGY_DISPLAY_NAME_STR = "Adaptive Macro CORE5"
RISK_ASSET_TUPLE = ("SPY", "IEF", "GLD", "DBC", "UUP")
RESERVE_ASSET_STR = "BIL"
COMMODITY_ASSET_STR = "DBC"
TRADEABLE_ASSET_TUPLE = RISK_ASSET_TUPLE + (RESERVE_ASSET_STR,)
SIGNAL_NAMESPACE_PREFIX_STR = "ADAPTIVE_TR_"
PORTFOLIO_NAMESPACE_STR = "Portfolio"
MONTH_END_REBALANCE_FIELD_STR = "month_end_rebalance_bool"
LONG_STATE_CHANGED_FIELD_STR = "long_state_changed_bool"
DEFAULT_ANNUAL_DBC_BORROW_RATE_FLOAT = 0.01
BORROW_DAY_COUNT_DENOMINATOR_INT = 360
BORROW_COLLATERAL_MULTIPLIER_FLOAT = 1.02


def default_trade_id_int() -> int:
    return -1


def signal_namespace_str(asset_str: str) -> str:
    return f"{SIGNAL_NAMESPACE_PREFIX_STR}{asset_str}"


@dataclass(frozen=True)
class AdaptiveMacroCore5Config:
    strategy_name_str: str = STRATEGY_NAME_STR
    risk_asset_tuple: tuple[str, ...] = RISK_ASSET_TUPLE
    reserve_asset_str: str = RESERVE_ASSET_STR
    commodity_asset_str: str = COMMODITY_ASSET_STR
    benchmark_list: tuple[str, ...] = ("$SPX",)
    history_start_date_str: str = "1990-01-01"
    backtest_start_date_str: str = "2007-09-01"
    end_date_str: str | None = None
    percentile_lookback_int: int = 126
    fast_lookback_int: int = 50
    slow_lookback_int: int = 200
    percentile_power_float: float = 2.0
    price_filter_lookback_int: int = 10
    commodity_vol_lookback_int: int = 63
    sleeve_weight_float: float = 0.20
    commodity_short_vol_target_float: float = 0.025
    commodity_short_cap_float: float = 0.10
    annual_dbc_borrow_rate_float: float = DEFAULT_ANNUAL_DBC_BORROW_RATE_FLOAT
    capital_base_float: float = 100_000.0
    slippage_float: float = 0.00025
    commission_per_share_float: float = 0.005
    commission_minimum_float: float = 1.0

    def __post_init__(self) -> None:
        if len(self.risk_asset_tuple) != 5 or len(set(self.risk_asset_tuple)) != 5:
            raise ValueError("risk_asset_tuple must contain five unique assets.")
        if self.reserve_asset_str in self.risk_asset_tuple:
            raise ValueError("reserve_asset_str must not be a risk asset.")
        if self.commodity_asset_str not in self.risk_asset_tuple:
            raise ValueError("commodity_asset_str must be one of the risk assets.")
        if len(self.benchmark_list) == 0:
            raise ValueError("benchmark_list must not be empty.")
        if self.percentile_lookback_int <= 1:
            raise ValueError("percentile_lookback_int must be greater than one.")
        if self.fast_lookback_int <= 1:
            raise ValueError("fast_lookback_int must be greater than one.")
        if self.slow_lookback_int <= self.fast_lookback_int:
            raise ValueError("slow_lookback_int must exceed fast_lookback_int.")
        if self.price_filter_lookback_int <= 1:
            raise ValueError("price_filter_lookback_int must be greater than one.")
        if self.commodity_vol_lookback_int <= 1:
            raise ValueError("commodity_vol_lookback_int must be greater than one.")
        if not np.isfinite(self.percentile_power_float) or self.percentile_power_float <= 0.0:
            raise ValueError("percentile_power_float must be positive.")
        if not np.isclose(
            self.sleeve_weight_float * len(self.risk_asset_tuple),
            1.0,
            atol=1e-12,
        ):
            raise ValueError("Five fixed sleeve weights must sum to 1.0.")
        if self.commodity_short_vol_target_float <= 0.0:
            raise ValueError("commodity_short_vol_target_float must be positive.")
        if not 0.0 < self.commodity_short_cap_float <= 1.0:
            raise ValueError("commodity_short_cap_float must be in (0, 1].")
        if (
            not np.isfinite(self.annual_dbc_borrow_rate_float)
            or self.annual_dbc_borrow_rate_float < 0.0
        ):
            raise ValueError("annual_dbc_borrow_rate_float must be finite and non-negative.")
        if self.capital_base_float <= 0.0:
            raise ValueError("capital_base_float must be positive.")
        if min(
            self.slippage_float,
            self.commission_per_share_float,
            self.commission_minimum_float,
        ) < 0.0:
            raise ValueError("Trading costs must be non-negative.")


DEFAULT_CONFIG = AdaptiveMacroCore5Config()


def compute_midrank_trailing_percentile_ser(
    severity_ser: pd.Series,
    lookback_int: int,
) -> pd.Series:
    """Return the inclusive trailing mid-rank percentile at each Close_T.

    For a complete window of length n:

        rank_T = (N_less + (N_equal + 1) / 2) / n

    ``N_equal`` includes the current observation. No full-sample normalization
    or future observation enters the rank.
    """

    if lookback_int <= 1:
        raise ValueError("lookback_int must be greater than one.")

    severity_ser = pd.Series(severity_ser, copy=True).astype(float)
    percentile_ser = pd.Series(np.nan, index=severity_ser.index, dtype=float)
    severity_vec = severity_ser.to_numpy(dtype=float)

    # *** CRITICAL*** lookahead-sensitive rolling boundary: every window ends
    # at Close_T and contains exactly the latest lookback_int observations.
    for bar_idx_int in range(lookback_int - 1, len(severity_vec)):
        window_start_idx_int = bar_idx_int - lookback_int + 1
        trailing_severity_vec = severity_vec[
            window_start_idx_int : bar_idx_int + 1
        ]
        if not np.isfinite(trailing_severity_vec).all():
            continue
        current_severity_float = float(trailing_severity_vec[-1])
        lower_count_int = int(
            np.count_nonzero(trailing_severity_vec < current_severity_float)
        )
        equal_count_int = int(
            np.count_nonzero(trailing_severity_vec == current_severity_float)
        )
        percentile_ser.iloc[bar_idx_int] = float(
            lower_count_int + (equal_count_int + 1.0) / 2.0
        ) / float(lookback_int)

    return percentile_ser


def compute_adaptive_asset_signal_df(
    signal_price_close_ser: pd.Series,
    config_obj: AdaptiveMacroCore5Config = DEFAULT_CONFIG,
) -> pd.DataFrame:
    """Compute one asset's literal drawdown-adaptive signal path."""

    signal_price_close_ser = pd.Series(
        signal_price_close_ser,
        copy=True,
    ).astype(float)
    if signal_price_close_ser.index.has_duplicates:
        raise ValueError("signal_price_close_ser index must be unique.")
    if not signal_price_close_ser.index.is_monotonic_increasing:
        raise ValueError("signal_price_close_ser index must be increasing.")

    observed_price_ser = signal_price_close_ser.dropna()
    if len(observed_price_ser) == 0:
        # A full Norgate panel legitimately contains an unavailable prefix for
        # ETFs before inception. Preserve that state as unavailable; the
        # execution calendar cannot start until every required signal is valid.
        return pd.DataFrame(
            np.nan,
            index=signal_price_close_ser.index,
            columns=[
                "signal_price_close_ser",
                "reference_high_ser",
                "drawdown_ser",
                "drawdown_severity_ser",
                "drawdown_percentile_ser",
                "adaptive_weight_ser",
                "adaptive_alpha_ser",
                "adaptive_moving_average_ser",
                "filtered_price_ser",
                "daily_return_ser",
                "annualized_volatility_ser",
                "long_state_ser",
                "short_state_ser",
            ],
            dtype=float,
        )
    if (
        not np.isfinite(observed_price_ser.to_numpy(dtype=float)).all()
        or observed_price_ser.le(0.0).any()
    ):
        raise ValueError("Observed signal prices must be finite and positive.")

    # *** CRITICAL*** expanding boundary: the high at Close_T contains only
    # prices observed on or before T and never resets at the backtest start.
    reference_high_ser = observed_price_ser.cummax()
    drawdown_ser = observed_price_ser.divide(reference_high_ser).sub(1.0)
    drawdown_severity_ser = drawdown_ser.mul(-1.0)
    drawdown_percentile_ser = compute_midrank_trailing_percentile_ser(
        severity_ser=drawdown_severity_ser,
        lookback_int=config_obj.percentile_lookback_int,
    )
    adaptive_weight_ser = drawdown_percentile_ser.pow(
        config_obj.percentile_power_float
    )

    fast_alpha_float = 2.0 / float(config_obj.fast_lookback_int + 1)
    slow_alpha_float = 2.0 / float(config_obj.slow_lookback_int + 1)
    adaptive_alpha_ser = adaptive_weight_ser.mul(fast_alpha_float).add(
        (1.0 - adaptive_weight_ser).mul(slow_alpha_float)
    )

    adaptive_moving_average_ser = pd.Series(
        np.nan,
        index=observed_price_ser.index,
        dtype=float,
    )
    prior_adaptive_average_float = np.nan
    for bar_idx_int, bar_ts in enumerate(observed_price_ser.index):
        alpha_float = float(adaptive_alpha_ser.iloc[bar_idx_int])
        if not np.isfinite(alpha_float):
            continue
        price_float = float(observed_price_ser.iloc[bar_idx_int])
        # *** CRITICAL*** recursive boundary: AMA_T uses only Close_T, alpha_T,
        # and AMA_(T-1). The first available alpha initializes AMA_T to Close_T.
        adaptive_average_float = (
            price_float
            if not np.isfinite(prior_adaptive_average_float)
            else alpha_float * price_float
            + (1.0 - alpha_float) * prior_adaptive_average_float
        )
        adaptive_moving_average_ser.loc[bar_ts] = adaptive_average_float
        prior_adaptive_average_float = adaptive_average_float

    # *** CRITICAL*** rolling boundary: SMA10_T contains the ten closes ending
    # at Close_T. DBC volatility likewise contains 63 returns ending at T.
    filtered_price_ser = observed_price_ser.rolling(
        window=config_obj.price_filter_lookback_int,
        min_periods=config_obj.price_filter_lookback_int,
    ).mean()
    daily_return_ser = observed_price_ser.pct_change(fill_method=None)
    annualized_volatility_ser = daily_return_ser.rolling(
        window=config_obj.commodity_vol_lookback_int,
        min_periods=config_obj.commodity_vol_lookback_int,
    ).std(ddof=1).mul(np.sqrt(252.0))

    valid_signal_bool_ser = (
        filtered_price_ser.notna() & adaptive_moving_average_ser.notna()
    )
    long_state_ser = pd.Series(
        np.nan,
        index=observed_price_ser.index,
        dtype=float,
    )
    short_state_ser = pd.Series(
        np.nan,
        index=observed_price_ser.index,
        dtype=float,
    )
    long_state_ser.loc[valid_signal_bool_ser] = (
        filtered_price_ser.loc[valid_signal_bool_ser]
        .gt(adaptive_moving_average_ser.loc[valid_signal_bool_ser])
        .astype(float)
    )
    short_state_ser.loc[valid_signal_bool_ser] = (
        filtered_price_ser.loc[valid_signal_bool_ser]
        .lt(adaptive_moving_average_ser.loc[valid_signal_bool_ser])
        .astype(float)
    )

    observed_signal_df = pd.DataFrame(
        {
            "signal_price_close_ser": observed_price_ser,
            "reference_high_ser": reference_high_ser,
            "drawdown_ser": drawdown_ser,
            "drawdown_severity_ser": drawdown_severity_ser,
            "drawdown_percentile_ser": drawdown_percentile_ser,
            "adaptive_weight_ser": adaptive_weight_ser,
            "adaptive_alpha_ser": adaptive_alpha_ser,
            "adaptive_moving_average_ser": adaptive_moving_average_ser,
            "filtered_price_ser": filtered_price_ser,
            "daily_return_ser": daily_return_ser,
            "annualized_volatility_ser": annualized_volatility_ser,
            "long_state_ser": long_state_ser,
            "short_state_ser": short_state_ser,
        }
    )
    return observed_signal_df.reindex(signal_price_close_ser.index)


def build_target_weight_ser(
    long_state_ser: pd.Series,
    commodity_short_state_bool: bool,
    commodity_annualized_volatility_float: float,
    config_obj: AdaptiveMacroCore5Config = DEFAULT_CONFIG,
) -> pd.Series:
    """Build the 100% long/BIL book plus the independent DBC short layer."""

    long_state_ser = long_state_ser.reindex(config_obj.risk_asset_tuple).astype(float)
    if long_state_ser.isna().any() or not long_state_ser.isin([0.0, 1.0]).all():
        raise ValueError("Every long state must be exactly 0 or 1.")

    risk_target_weight_ser = long_state_ser.mul(config_obj.sleeve_weight_float)
    reserve_weight_float = float(1.0 - risk_target_weight_ser.sum())
    commodity_short_weight_float = 0.0
    if commodity_short_state_bool:
        if (
            not np.isfinite(commodity_annualized_volatility_float)
            or commodity_annualized_volatility_float <= 0.0
        ):
            raise ValueError("DBC short sizing requires positive finite volatility.")
        commodity_short_weight_float = -min(
            config_obj.commodity_short_cap_float,
            config_obj.commodity_short_vol_target_float
            / commodity_annualized_volatility_float,
        )
        if risk_target_weight_ser.loc[config_obj.commodity_asset_str] != 0.0:
            raise ValueError("DBC cannot be long and short in the same target.")
        risk_target_weight_ser.loc[
            config_obj.commodity_asset_str
        ] = commodity_short_weight_float

    restricted_short_proceeds_weight_float = abs(commodity_short_weight_float)
    target_weight_ser = pd.concat(
        [
            risk_target_weight_ser,
            pd.Series(
                {config_obj.reserve_asset_str: reserve_weight_float},
                dtype=float,
            ),
            pd.Series(
                {"Cash": restricted_short_proceeds_weight_float},
                dtype=float,
            ),
        ]
    )
    long_book_weight_float = float(
        target_weight_ser.loc[
            list(config_obj.risk_asset_tuple) + [config_obj.reserve_asset_str]
        ]
        .clip(lower=0.0)
        .sum()
    )
    if not np.isclose(long_book_weight_float, 1.0, atol=1e-12):
        raise ValueError("CORE5 long/BIL targets must sum to 100% of NAV.")
    if not np.isclose(float(target_weight_ser.sum()), 1.0, atol=1e-12):
        raise ValueError("Net positions plus restricted short proceeds must sum to NAV.")
    return target_weight_ser


def get_adaptive_macro_core5_data(
    config_obj: AdaptiveMacroCore5Config = DEFAULT_CONFIG,
) -> pd.DataFrame:
    """Load CAPITALSPECIAL execution data and TOTALRETURN signal closes."""

    execution_price_df = load_raw_prices(
        symbols=list(config_obj.risk_asset_tuple) + [config_obj.reserve_asset_str],
        benchmarks=list(config_obj.benchmark_list),
        start_date=config_obj.history_start_date_str,
        end_date=config_obj.end_date_str,
    )
    total_return_signal_df = load_raw_prices(
        symbols=[],
        benchmarks=list(config_obj.risk_asset_tuple),
        start_date=config_obj.history_start_date_str,
        end_date=config_obj.end_date_str,
    )
    signal_close_df = total_return_signal_df.loc[
        :,
        [(asset_str, "Close") for asset_str in config_obj.risk_asset_tuple],
    ].copy()
    signal_close_df.columns = pd.MultiIndex.from_tuples(
        [
            (signal_namespace_str(asset_str), "Close")
            for asset_str in config_obj.risk_asset_tuple
        ]
    )
    pricing_data_df = pd.concat(
        [execution_price_df, signal_close_df],
        axis=1,
    ).sort_index()
    pricing_data_df.attrs.update(execution_price_df.attrs)
    pricing_data_df.attrs["signal_adjustment_by_symbol_dict"] = {
        signal_namespace_str(asset_str): "TOTALRETURN"
        for asset_str in config_obj.risk_asset_tuple
    }
    return pricing_data_df


def _month_end_rebalance_ser(execution_index: pd.DatetimeIndex) -> pd.Series:
    month_period_idx = execution_index.to_period("M")
    # *** CRITICAL*** This shift uses only the known exchange-session calendar,
    # never a future price or signal. It identifies the final session of month T.
    next_month_period_ser = pd.Series(month_period_idx, index=execution_index).shift(-1)
    return pd.Series(
        next_month_period_ser.isna().to_numpy()
        | (month_period_idx != next_month_period_ser.to_numpy()),
        index=execution_index,
        dtype=bool,
    )


def build_execution_calendar_idx(
    pricing_data_df: pd.DataFrame,
    config_obj: AdaptiveMacroCore5Config = DEFAULT_CONFIG,
    backtest_start_date_str: str | None = None,
) -> pd.DatetimeIndex:
    """Start at the first open after every Close_T signal is actionable."""

    valid_decision_bool_ser = pd.Series(True, index=pricing_data_df.index, dtype=bool)
    for asset_str in config_obj.risk_asset_tuple:
        asset_signal_df = compute_adaptive_asset_signal_df(
            pricing_data_df[(signal_namespace_str(asset_str), "Close")],
            config_obj=config_obj,
        )
        valid_decision_bool_ser &= asset_signal_df["long_state_ser"].notna()
    commodity_signal_df = compute_adaptive_asset_signal_df(
        pricing_data_df[(signal_namespace_str(config_obj.commodity_asset_str), "Close")],
        config_obj=config_obj,
    )
    valid_decision_bool_ser &= commodity_signal_df[
        "annualized_volatility_ser"
    ].notna()

    valid_decision_position_vec = np.flatnonzero(
        valid_decision_bool_ser.to_numpy(dtype=bool)
    )
    valid_execution_position_vec = valid_decision_position_vec + 1
    valid_execution_position_vec = valid_execution_position_vec[
        valid_execution_position_vec < len(pricing_data_df.index)
    ]
    if len(valid_execution_position_vec) == 0:
        raise RuntimeError("No actionable Adaptive Macro CORE5 execution date exists.")

    tradeable_open_df = pricing_data_df.loc[
        :,
        [(asset_str, "Open") for asset_str in TRADEABLE_ASSET_TUPLE],
    ].astype(float)
    valid_open_bool_ser = np.isfinite(tradeable_open_df).all(axis=1) & (
        tradeable_open_df > 0.0
    ).all(axis=1)
    valid_execution_date_index = pricing_data_df.index[
        valid_execution_position_vec[
            valid_open_bool_ser.iloc[valid_execution_position_vec].to_numpy(dtype=bool)
        ]
    ]
    if len(valid_execution_date_index) == 0:
        raise RuntimeError("No actionable CORE5 date has complete next-open prices.")

    requested_start_ts = pd.Timestamp(
        config_obj.backtest_start_date_str
        if backtest_start_date_str is None
        else backtest_start_date_str
    )
    calendar_start_ts = max(pd.Timestamp(valid_execution_date_index[0]), requested_start_ts)
    calendar_idx = pd.DatetimeIndex(
        pricing_data_df.index[pricing_data_df.index >= calendar_start_ts]
    )
    if len(calendar_idx) == 0:
        raise RuntimeError("Adaptive Macro CORE5 execution calendar is empty.")
    return calendar_idx


class AdaptiveMacroCore5Strategy(Strategy):
    """Five independent adaptive macro sleeves with BIL and a DBC short."""

    enable_signal_audit = True
    signal_audit_sample_size = 10

    def __init__(
        self,
        name: str = STRATEGY_NAME_STR,
        benchmarks: Sequence[str] | None = None,
        config_obj: AdaptiveMacroCore5Config = DEFAULT_CONFIG,
    ) -> None:
        benchmark_list = list(config_obj.benchmark_list if benchmarks is None else benchmarks)
        super().__init__(
            name=name,
            benchmarks=benchmark_list,
            capital_base=config_obj.capital_base_float,
            slippage=config_obj.slippage_float,
            commission_per_share=config_obj.commission_per_share_float,
            commission_minimum=config_obj.commission_minimum_float,
            performance_benchmark_symbol_str=benchmark_list[0],
            performance_benchmark_adjustment_str="TOTALRETURN",
        )
        self.config_obj = config_obj
        self.asset_list = list(config_obj.risk_asset_tuple) + [
            config_obj.reserve_asset_str
        ]
        self.trade_id_int = 0
        self.current_trade_id_map = {
            asset_str: default_trade_id_int() for asset_str in self.asset_list
        }
        self.initialized_bool = False
        self.last_target_weight_ser = pd.Series(dtype=float)
        self.rebalance_target_weight_row_dict_list: list[dict[str, object]] = []
        self.daily_target_weight_row_dict_list: list[dict[str, object]] = []
        self.signal_diagnostic_df = pd.DataFrame()
        self.borrow_calendar_idx = pd.DatetimeIndex([])
        self.borrow_fee_row_dict_list: list[dict[str, object]] = []
        self.borrow_fee_total_float = 0.0
        self._data_adjustment_policy_dict.update(
            {
                "signal_adjustment_str": "TOTALRETURN",
                "execution_and_marks_adjustment_str": "CAPITALSPECIAL",
                "performance_benchmark_adjustment_str": "TOTALRETURN",
            }
        )
        self._accounting_policy_dict.update(
            {
                "short_proceeds_policy_str": "restricted_cash_not_reinvested",
                "short_borrow_cost_policy_str": "fixed_annual_dbc_research_baseline",
                "annual_dbc_borrow_rate_float": config_obj.annual_dbc_borrow_rate_float,
                "borrow_day_count_denominator_int": BORROW_DAY_COUNT_DENOMINATOR_INT,
                "borrow_collateral_multiplier_float": BORROW_COLLATERAL_MULTIPLIER_FLOAT,
                "borrow_collateral_rounding_str": "ceil_per_share",
                "short_proceeds_interest_float": 0.0,
                "borrow_accrual_start_str": "trade_date_proxy_settlement_mismatch",
                "borrow_accrual_row_count_int": 0,
                "borrow_fee_total_float": 0.0,
                "target_long_book_gross_limit_float": 1.0,
                "target_total_gross_limit_float": (
                    1.0 + config_obj.commodity_short_cap_float
                ),
            }
        )

    def _next_borrow_session_ts(self) -> pd.Timestamp | None:
        if len(self.borrow_calendar_idx) == 0:
            raise RuntimeError("borrow_calendar_idx must be assigned before the run.")
        current_position_int = int(self.borrow_calendar_idx.get_loc(self.current_bar))
        if current_position_int + 1 >= len(self.borrow_calendar_idx):
            return None
        return pd.Timestamp(self.borrow_calendar_idx[current_position_int + 1])

    def configure_run_calendar(self, calendar_idx: pd.DatetimeIndex) -> None:
        resolved_calendar_idx = pd.DatetimeIndex(calendar_idx)
        if len(resolved_calendar_idx) == 0:
            raise ValueError("CORE5 run calendar must not be empty.")
        self.borrow_calendar_idx = resolved_calendar_idx.copy()

    def apply_post_mark_accounting(self, prices: pd.DataFrame) -> None:
        """Debit the fixed DBC borrow baseline after fills and close marking."""
        annual_borrow_rate_float = float(
            self.config_obj.annual_dbc_borrow_rate_float
        )
        if annual_borrow_rate_float == 0.0:
            return

        commodity_asset_str = self.config_obj.commodity_asset_str
        held_share_float = float(self.get_position(commodity_asset_str))
        if held_share_float >= 0.0:
            return
        next_session_ts = self._next_borrow_session_ts()
        if next_session_ts is None:
            return

        close_price_float = float(
            prices.loc[self.current_bar, (commodity_asset_str, "Close")]
        )
        if not np.isfinite(close_price_float) or close_price_float <= 0.0:
            raise RuntimeError("DBC borrow accounting requires a valid current close.")
        calendar_day_count_int = int(
            (next_session_ts.normalize() - pd.Timestamp(self.current_bar).normalize()).days
        )
        if calendar_day_count_int <= 0:
            raise RuntimeError("Borrow accrual requires a positive calendar-day interval.")

        # *** CRITICAL *** post-fill accounting boundary: the current open order
        # has already filled and the current close has already marked the held
        # position. This fee changes cash/NAV only; it cannot alter the prior
        # Close_T signal or the already executed Open_(T+1) order.
        collateral_price_float = float(
            np.ceil(BORROW_COLLATERAL_MULTIPLIER_FLOAT * close_price_float)
        )
        collateral_value_float = abs(held_share_float) * collateral_price_float
        borrow_fee_float = float(
            collateral_value_float
            * annual_borrow_rate_float
            * calendar_day_count_int
            / BORROW_DAY_COUNT_DENOMINATOR_INT
        )
        self.cash -= borrow_fee_float
        self.total_value -= borrow_fee_float
        self.borrow_fee_row_dict_list.append(
            {
                "accrual_start_date_ts": pd.Timestamp(self.current_bar).normalize(),
                "next_session_date_ts": next_session_ts.normalize(),
                "calendar_day_count_int": calendar_day_count_int,
                "dbc_share_float": held_share_float,
                "dbc_close_float": close_price_float,
                "collateral_price_float": collateral_price_float,
                "collateral_value_float": collateral_value_float,
                "annual_borrow_rate_float": annual_borrow_rate_float,
                "borrow_fee_float": borrow_fee_float,
                "cash_after_fee_float": float(self.cash),
                "total_value_after_fee_float": float(self.total_value),
            }
        )
        self.borrow_fee_total_float += borrow_fee_float
        self._accounting_policy_dict.update(
            {
                "borrow_accrual_row_count_int": len(
                    self.borrow_fee_row_dict_list
                ),
                "borrow_fee_total_float": self.borrow_fee_total_float,
            }
        )
        self.borrow_fee_df = pd.DataFrame(self.borrow_fee_row_dict_list)

    def process_orders(self, prices: pd.DataFrame) -> None:
        super().process_orders(prices)
        self.apply_post_mark_accounting(prices)

    def compute_signals(self, pricing_data_df: pd.DataFrame) -> pd.DataFrame:
        feature_df = pd.DataFrame(index=pricing_data_df.index)
        long_state_column_list: list[tuple[str, str]] = []
        diagnostic_frame_list: list[pd.DataFrame] = []

        for asset_str in self.config_obj.risk_asset_tuple:
            signal_close_key_tuple = (signal_namespace_str(asset_str), "Close")
            if signal_close_key_tuple not in pricing_data_df.columns:
                raise RuntimeError(
                    f"Missing TOTALRETURN signal close for {asset_str}."
                )
            asset_signal_df = compute_adaptive_asset_signal_df(
                pricing_data_df[signal_close_key_tuple],
                config_obj=self.config_obj,
            )
            namespace_str = signal_namespace_str(asset_str)
            for field_str in asset_signal_df.columns:
                feature_df[(namespace_str, field_str)] = asset_signal_df[field_str]
            long_state_column_list.append((namespace_str, "long_state_ser"))
            diagnostic_frame_list.append(
                asset_signal_df.add_prefix(f"{asset_str}_")
            )

        long_state_df = feature_df.loc[:, long_state_column_list].astype(float)
        # *** CRITICAL*** diff compares Close_T only with Close_(T-1); no future
        # state enters the rebalance decision.
        long_state_changed_ser = long_state_df.diff().abs().fillna(0.0).gt(0.0).any(axis=1)
        feature_df[(PORTFOLIO_NAMESPACE_STR, LONG_STATE_CHANGED_FIELD_STR)] = (
            long_state_changed_ser
        )
        feature_df[(PORTFOLIO_NAMESPACE_STR, MONTH_END_REBALANCE_FIELD_STR)] = (
            _month_end_rebalance_ser(pricing_data_df.index)
        )
        feature_df.columns = pd.MultiIndex.from_tuples(feature_df.columns)
        diagnostic_df = pd.concat(diagnostic_frame_list, axis=1)
        # Signal audit recomputes shorter causal prefixes. Preserve the full
        # initial diagnostic frame instead of replacing it with the last audit
        # sample as an incidental side effect.
        if len(diagnostic_df) >= len(self.signal_diagnostic_df):
            self.signal_diagnostic_df = diagnostic_df
        return pd.concat([pricing_data_df, feature_df], axis=1)

    def signal_audit_fields(
        self,
        pricing_data: pd.DataFrame,
        signal_data: pd.DataFrame,
    ):
        audit_column_list = super().signal_audit_fields(pricing_data, signal_data)
        # Month-end is derived from the full known session calendar. A truncated
        # audit prefix makes its final row look like a month-end even when the
        # full calendar says otherwise; price-derived fields remain audited.
        return [
            column_tuple
            for column_tuple in audit_column_list
            if column_tuple
            != (PORTFOLIO_NAMESPACE_STR, MONTH_END_REBALANCE_FIELD_STR)
        ]

    def _long_state_ser(self, close_row_ser: pd.Series) -> pd.Series:
        long_state_ser = pd.Series(
            {
                asset_str: float(
                    close_row_ser.get(
                        (signal_namespace_str(asset_str), "long_state_ser"),
                        np.nan,
                    )
                )
                for asset_str in self.config_obj.risk_asset_tuple
            },
            dtype=float,
        )
        if long_state_ser.isna().any() or not long_state_ser.isin([0.0, 1.0]).all():
            raise RuntimeError(
                f"Incomplete CORE5 long-state snapshot at Close_{self.previous_bar}."
            )
        return long_state_ser

    def _validate_required_close_prices(self, close_row_ser: pd.Series) -> None:
        invalid_asset_list = []
        for asset_str in self.asset_list:
            close_price_float = float(
                close_row_ser.get((asset_str, "Close"), np.nan)
            )
            if not np.isfinite(close_price_float) or close_price_float <= 0.0:
                invalid_asset_list.append(asset_str)
        if invalid_asset_list:
            raise RuntimeError(
                "Incomplete CORE5 execution-price snapshot at "
                f"Close_{self.previous_bar}: {invalid_asset_list}."
            )

    def _target_weight_ser(
        self,
        close_row_ser: pd.Series,
        long_state_ser: pd.Series,
    ) -> pd.Series:
        commodity_namespace_str = signal_namespace_str(
            self.config_obj.commodity_asset_str
        )
        commodity_short_state_float = float(
            close_row_ser.get(
                (commodity_namespace_str, "short_state_ser"),
                np.nan,
            )
        )
        commodity_volatility_float = float(
            close_row_ser.get(
                (commodity_namespace_str, "annualized_volatility_ser"),
                np.nan,
            )
        )
        if commodity_short_state_float not in (0.0, 1.0):
            raise RuntimeError(
                f"Invalid DBC short state at Close_{self.previous_bar}."
            )
        return build_target_weight_ser(
            long_state_ser=long_state_ser,
            commodity_short_state_bool=bool(commodity_short_state_float),
            commodity_annualized_volatility_float=commodity_volatility_float,
            config_obj=self.config_obj,
        )

    def _new_trade_id_int(self, asset_str: str) -> int:
        self.trade_id_int += 1
        self.current_trade_id_map[asset_str] = self.trade_id_int
        return self.trade_id_int

    def _filled_position_trade_id_int(
        self,
        asset_str: str,
        current_share_int: int,
    ) -> int:
        """Return the trade ID of the position that is filled now.

        Timing diagnostics can delay an order after ``iterate()`` creates it.
        The intent map may therefore describe a pending leg rather than the
        position actually held. The filled transaction ledger is authoritative
        whenever a non-zero position must be resized or closed.
        """

        if current_share_int == 0:
            return default_trade_id_int()
        asset_transaction_df = self.get_transactions()
        asset_transaction_df = asset_transaction_df.loc[
            asset_transaction_df["asset"] == asset_str
        ]
        if len(asset_transaction_df) == 0:
            raise RuntimeError(f"Open {asset_str} position has no filled transaction.")
        trade_id_int = int(asset_transaction_df.iloc[-1]["trade_id"])
        if trade_id_int == default_trade_id_int():
            raise RuntimeError(f"Open {asset_str} position has no trade ID.")
        self.current_trade_id_map[asset_str] = trade_id_int
        return trade_id_int

    def _queue_close_order(self, asset_str: str, current_share_int: int) -> None:
        if current_share_int == 0:
            return
        trade_id_int = self._filled_position_trade_id_int(
            asset_str=asset_str,
            current_share_int=current_share_int,
        )
        self.order_target(asset_str, 0, trade_id=trade_id_int)
        self.current_trade_id_map[asset_str] = default_trade_id_int()

    def _submit_target_orders(
        self,
        target_weight_ser: pd.Series,
        close_row_ser: pd.Series,
    ) -> None:
        current_position_ser = self.get_positions().reindex(
            self.asset_list,
            fill_value=0.0,
        ).astype(int)
        budget_value_float = float(self.previous_total_value)
        if not np.isfinite(budget_value_float) or budget_value_float <= 0.0:
            raise RuntimeError("Previous portfolio value must be positive.")

        for asset_str in self.asset_list:
            target_weight_float = float(target_weight_ser.loc[asset_str])
            current_share_int = int(current_position_ser.loc[asset_str])
            sizing_close_float = float(
                close_row_ser.get((asset_str, "Close"), np.nan)
            )
            if not np.isfinite(sizing_close_float) or sizing_close_float <= 0.0:
                raise RuntimeError(
                    f"Invalid execution close for {asset_str} on {self.previous_bar}."
                )

            # *** CRITICAL*** Target shares are fixed from NAV and Close_T. The
            # engine fills the queued order at Open_(T+1); that open cannot size
            # the order or decide whether the rebalance happens.
            target_share_int = int(
                budget_value_float * target_weight_float / sizing_close_float
            )
            if target_share_int == current_share_int:
                continue

            sign_flip_bool = (
                current_share_int != 0
                and target_share_int != 0
                and np.sign(current_share_int) != np.sign(target_share_int)
            )
            if sign_flip_bool:
                # Long and short DBC are distinct lifecycle legs. Closing the old
                # leg before opening the new one keeps trade P&L auditable and is
                # conservatively charged as two orders at the same next open.
                self._queue_close_order(asset_str, current_share_int)
                new_trade_id_int = self._new_trade_id_int(asset_str)
                self.order_target_percent(
                    asset_str,
                    target_weight_float,
                    trade_id=new_trade_id_int,
                )
                continue

            if target_share_int == 0:
                self._queue_close_order(asset_str, current_share_int)
                continue

            if current_share_int == 0:
                trade_id_int = self._new_trade_id_int(asset_str)
            else:
                trade_id_int = self._filled_position_trade_id_int(
                    asset_str=asset_str,
                    current_share_int=current_share_int,
                )
            self.order_target_percent(
                asset_str,
                target_weight_float,
                trade_id=trade_id_int,
            )

    def _record_target_weight(
        self,
        target_weight_ser: pd.Series,
        rebalance_bool: bool,
    ) -> None:
        record_dict = {
            "decision_date_ts": pd.Timestamp(self.previous_bar),
            **{
                str(asset_str): float(weight_float)
                for asset_str, weight_float in target_weight_ser.items()
            },
        }
        self.daily_target_weight_row_dict_list.append(record_dict)
        if rebalance_bool:
            self.rebalance_target_weight_row_dict_list.append(record_dict.copy())

    def iterate(
        self,
        data_df: pd.DataFrame,
        close_row_ser: pd.Series,
        _open_price_ser: pd.Series,
    ) -> None:
        if close_row_ser is None or data_df is None:
            return

        self._validate_required_close_prices(close_row_ser)
        long_state_ser = self._long_state_ser(close_row_ser)
        long_state_changed_bool = bool(
            close_row_ser.get(
                (PORTFOLIO_NAMESPACE_STR, LONG_STATE_CHANGED_FIELD_STR),
                False,
            )
        )
        month_end_rebalance_bool = bool(
            close_row_ser.get(
                (PORTFOLIO_NAMESPACE_STR, MONTH_END_REBALANCE_FIELD_STR),
                False,
            )
        )
        rebalance_bool = bool(
            not self.initialized_bool
            or long_state_changed_bool
            or month_end_rebalance_bool
        )

        if not rebalance_bool:
            if len(self.last_target_weight_ser) == 0:
                raise RuntimeError("Initialized CORE5 strategy has no prior target.")
            self._record_target_weight(
                self.last_target_weight_ser,
                rebalance_bool=False,
            )
            return

        target_weight_ser = self._target_weight_ser(
            close_row_ser=close_row_ser,
            long_state_ser=long_state_ser,
        )
        self._submit_target_orders(
            target_weight_ser=target_weight_ser,
            close_row_ser=close_row_ser,
        )
        self.last_target_weight_ser = target_weight_ser.copy()
        self.initialized_bool = True
        self._record_target_weight(target_weight_ser, rebalance_bool=True)

    def finalize(self, current_data_df: pd.DataFrame) -> None:
        if len(self.daily_target_weight_row_dict_list) > 0:
            daily_target_weight_df = pd.DataFrame(
                self.daily_target_weight_row_dict_list
            ).set_index("decision_date_ts")
            daily_target_weight_df.index = pd.to_datetime(
                daily_target_weight_df.index
            )
            self.daily_target_weights = daily_target_weight_df.sort_index()
        if len(self.rebalance_target_weight_row_dict_list) > 0:
            rebalance_target_weight_df = pd.DataFrame(
                self.rebalance_target_weight_row_dict_list
            ).set_index("decision_date_ts")
            rebalance_target_weight_df.index = pd.to_datetime(
                rebalance_target_weight_df.index
            )
            self.rebalance_target_weight_df = rebalance_target_weight_df.sort_index()
        self.borrow_fee_df = pd.DataFrame(self.borrow_fee_row_dict_list)


def _build_strategy_obj(
    config_obj: AdaptiveMacroCore5Config,
    calendar_idx: pd.DatetimeIndex,
) -> AdaptiveMacroCore5Strategy:
    strategy_obj = AdaptiveMacroCore5Strategy(
        name=config_obj.strategy_name_str,
        benchmarks=config_obj.benchmark_list,
        config_obj=config_obj,
    )
    strategy_obj.configure_run_calendar(calendar_idx)
    return strategy_obj


def _run_strategy(
    config_obj: AdaptiveMacroCore5Config,
    pricing_data_df: pd.DataFrame,
    backtest_start_date_str: str | None,
    show_progress_bool: bool,
) -> AdaptiveMacroCore5Strategy:
    calendar_idx = build_execution_calendar_idx(
        pricing_data_df=pricing_data_df,
        config_obj=config_obj,
        backtest_start_date_str=backtest_start_date_str,
    )
    strategy_obj = _build_strategy_obj(config_obj, calendar_idx)
    run_daily(
        strategy_obj,
        pricing_data_df,
        calendar=calendar_idx,
        show_progress=show_progress_bool,
        show_signal_progress_bool=show_progress_bool,
        audit_override_bool=None,
    )
    return strategy_obj


def run_variant(
    show_display_bool: bool = True,
    save_results_bool: bool = True,
    output_dir_str: str = "results",
    backtest_start_date_str: str | None = DEFAULT_CONFIG.backtest_start_date_str,
    capital_base_float: float = DEFAULT_CONFIG.capital_base_float,
    end_date_str: str | None = None,
    pricing_data_df: pd.DataFrame | None = None,
) -> AdaptiveMacroCore5Strategy:
    """Run the PM_READY Adaptive Macro CORE5 Vanilla strategy."""

    config_obj = replace(
        DEFAULT_CONFIG,
        capital_base_float=float(capital_base_float),
        end_date_str=end_date_str,
    )
    if pricing_data_df is None:
        pricing_data_df = get_adaptive_macro_core5_data(config_obj=config_obj)
    strategy_obj = _run_strategy(
        config_obj=config_obj,
        pricing_data_df=pricing_data_df,
        backtest_start_date_str=backtest_start_date_str,
        show_progress_bool=show_display_bool,
    )

    if show_display_bool:
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 1000)
        display(strategy_obj.summary)
        display(strategy_obj.summary_trades)
    if save_results_bool:
        save_results(strategy_obj, output_dir=output_dir_str)
    return strategy_obj


def _build_analysis_context_dict() -> dict[str, object]:
    config_obj = DEFAULT_CONFIG
    pricing_data_df = get_adaptive_macro_core5_data(config_obj=config_obj)
    calendar_idx = build_execution_calendar_idx(
        pricing_data_df=pricing_data_df,
        config_obj=config_obj,
        backtest_start_date_str=config_obj.backtest_start_date_str,
    )
    return {
        "strategy_name_str": STRATEGY_NAME_STR,
        "capital_base_float": float(config_obj.capital_base_float),
        "config_obj": config_obj,
        "pricing_data_df": pricing_data_df,
        "calendar_idx": calendar_idx,
    }


def build_execution_timing_analysis_inputs() -> dict[str, object]:
    """Build BENCH Timing inputs with exact Vanilla factory/calendar parity."""

    context_dict = _build_analysis_context_dict()

    def strategy_factory_fn() -> AdaptiveMacroCore5Strategy:
        return _build_strategy_obj(
            context_dict["config_obj"],
            context_dict["calendar_idx"],
        )

    return {
        "strategy_factory_fn": strategy_factory_fn,
        "pricing_data_df": context_dict["pricing_data_df"],
        "calendar_idx": context_dict["calendar_idx"],
        "order_generation_mode_str": "vanilla_current_bar",
        "risk_model_str": "taa_rebalance",
        "entry_timing_str_tuple": (
            "same_open",
            "same_close_moc",
            "next_open",
            "next_close",
        ),
        "exit_timing_str_tuple": (
            "same_open",
            "same_close_moc",
            "next_open",
            "next_close",
        ),
        "default_entry_timing_str": "same_open",
        "default_exit_timing_str": "same_open",
    }


def build_stress_test_context_dict() -> dict[str, object]:
    """Build BENCH Stress inputs from the full causal Vanilla history."""

    return _build_analysis_context_dict()


def build_stress_test_strategy_obj(
    context_dict: dict[str, object],
) -> AdaptiveMacroCore5Strategy:
    config_obj = context_dict["config_obj"]
    if not isinstance(config_obj, AdaptiveMacroCore5Config):
        raise TypeError("context config_obj must be AdaptiveMacroCore5Config.")
    return _build_strategy_obj(config_obj, context_dict["calendar_idx"])


def build_capacity_analysis_inputs(
    show_display_bool: bool = False,
    backtest_start_date_str: str | None = DEFAULT_CONFIG.backtest_start_date_str,
    capital_base_float: float = DEFAULT_CONFIG.capital_base_float,
    end_date_str: str | None = None,
) -> dict[str, object]:
    """Run one full-engine BENCH Capacity point with default trading costs."""

    config_obj = replace(
        DEFAULT_CONFIG,
        capital_base_float=float(capital_base_float),
        end_date_str=end_date_str,
    )
    pricing_data_df = get_adaptive_macro_core5_data(config_obj=config_obj)
    strategy_obj = _run_strategy(
        config_obj=config_obj,
        pricing_data_df=pricing_data_df,
        backtest_start_date_str=backtest_start_date_str,
        show_progress_bool=show_display_bool,
    )
    return {
        "strategy_obj": strategy_obj,
        "pricing_data_df": pricing_data_df,
        "execution_policy_str": "MOO",
        "impact_profile_str": "MOO_ETF_PROXY",
    }


if __name__ == "__main__":
    run_variant()
