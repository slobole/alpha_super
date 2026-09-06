"""Frozen Crisis Trend Core hedge strategy.

The strategy is an asymmetric monthly trend sleeve over 17 ETFs. Risk assets
may only be short, safe havens may only be long, and residual long capital is
held in SHY. The signal is decided after ``Close_T`` and queued orders fill at
``Open_(T+1)``.

For asset j and lookback L in {63, 126, 252}:

    excess_{j,L,T} = (TR_j,T / TR_j,T-L) / (TR_SHY,T / TR_SHY,T-L) - 1
    signal_{j,T} = mean_L(sign(excess_{j,L,T}))

After the direction constraint:

    raw_weight_{j,T} = signal_{j,T} / (vol_{j,T} * eligible_count_{class,T} * 4)
    raw_pod_return_T = sum_j(raw_weight_{j,T-1} * return_{j,T})
    weight_{j,T} = raw_weight_{j,T} * min(10% / exante_vol_T, 10)

Gross exposure is capped at 1.5. Targets refresh only at month-end. The Alpha
translation evaluates the 2 percentage-point rebalance band at ``Close_T`` so
the decision is causal; the Pakal research ledger evaluated drift at the next
open. This module is research/BENCH only and has no LIVE, broker, scheduler,
release, or allocation wiring.

The frozen ledger also treats residual SHY as a return-bearing cash proxy and
reports Open-to-Open returns. This Alpha path holds SHY as a traded asset,
charges every executed ETF leg, and marks NAV at each close. Signal/target
parity is tested separately; performance metrics are not labeled exact parity.
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


STRATEGY_NAME_STR = "strategy_crisis_trend_core"
STRATEGY_DISPLAY_NAME_STR = "Crisis Trend Core"

EQUITY_ASSET_TUPLE = ("SPY", "QQQ", "IWM", "EFA", "EEM")
BOND_ASSET_TUPLE = ("TLT", "IEF", "LQD", "HYG")
COMMODITY_ASSET_TUPLE = ("GLD", "SLV", "DBC", "USO")
FX_ASSET_TUPLE = ("UUP", "FXE", "FXY", "FXF")
UNIVERSE_ASSET_TUPLE = (
    EQUITY_ASSET_TUPLE
    + BOND_ASSET_TUPLE
    + COMMODITY_ASSET_TUPLE
    + FX_ASSET_TUPLE
)
LONG_ONLY_ASSET_TUPLE = ("TLT", "IEF", "GLD", "UUP", "FXY", "FXF")
SHORT_ONLY_ASSET_TUPLE = (
    "SPY",
    "QQQ",
    "IWM",
    "EFA",
    "EEM",
    "LQD",
    "HYG",
    "SLV",
    "DBC",
    "USO",
    "FXE",
)
RESERVE_ASSET_STR = "SHY"
TRADEABLE_ASSET_TUPLE = UNIVERSE_ASSET_TUPLE + (RESERVE_ASSET_STR,)
BENCHMARK_TUPLE = ("$SPX",)

ASSET_CLASS_BY_ASSET_DICT = {
    **{asset_str: "EQ" for asset_str in EQUITY_ASSET_TUPLE},
    **{asset_str: "BND" for asset_str in BOND_ASSET_TUPLE},
    **{asset_str: "CMD" for asset_str in COMMODITY_ASSET_TUPLE},
    **{asset_str: "FX" for asset_str in FX_ASSET_TUPLE},
}
LOOKBACK_TUPLE = (63, 126, 252)
REALIZED_VOLATILITY_WINDOW_INT = 63
EXANTE_VOLATILITY_WINDOW_INT = 252
MINIMUM_HISTORY_SESSIONS_INT = 260
POD_VOLATILITY_TARGET_FLOAT = 0.10
GROSS_EXPOSURE_CAP_FLOAT = 1.50
REBALANCE_BAND_FLOAT = 0.02
ONE_WAY_COST_FLOAT = 0.001
ANNUAL_SHORT_BORROW_RATE_FLOAT = 0.005
TRADING_SESSIONS_PER_YEAR_FLOAT = 252.0

SIGNAL_NAMESPACE_PREFIX_STR = "CRISIS_TR_"
PORTFOLIO_NAMESPACE_STR = "CrisisTrendPortfolio"
MONTH_END_FIELD_STR = "month_end_decision_bool"
EXANTE_VOLATILITY_FIELD_STR = "exante_volatility_float"
RAW_POD_RETURN_FIELD_STR = "raw_pod_return_float"
DESIRED_WEIGHT_FIELD_STR = "desired_weight_float"
SIGNAL_FIELD_STR = "signal_float"
ELIGIBLE_FIELD_STR = "eligible_bool"
REALIZED_VOLATILITY_FIELD_STR = "realized_volatility_float"


def signal_namespace_str(asset_str: str) -> str:
    return f"{SIGNAL_NAMESPACE_PREFIX_STR}{asset_str}"


def default_trade_id_int() -> int:
    return -1


@dataclass(frozen=True)
class CrisisTrendCoreConfig:
    strategy_name_str: str = STRATEGY_NAME_STR
    history_start_date_str: str = "2002-01-01"
    backtest_start_date_str: str = "2004-01-01"
    end_date_str: str | None = None
    capital_base_float: float = 100_000.0

    def __post_init__(self) -> None:
        if not self.strategy_name_str:
            raise ValueError("strategy_name_str must not be empty.")
        if self.capital_base_float <= 0.0 or not np.isfinite(self.capital_base_float):
            raise ValueError("capital_base_float must be positive and finite.")


DEFAULT_CONFIG = CrisisTrendCoreConfig()


@dataclass(frozen=True)
class CrisisTrendSignalBundle:
    signal_df: pd.DataFrame
    eligible_df: pd.DataFrame
    realized_volatility_df: pd.DataFrame
    raw_weight_df: pd.DataFrame
    raw_pod_return_ser: pd.Series
    exante_volatility_ser: pd.Series
    desired_weight_df: pd.DataFrame
    month_end_target_weight_df: pd.DataFrame


def _validate_total_return_close_df(total_return_close_df: pd.DataFrame) -> None:
    missing_asset_list = [
        asset_str
        for asset_str in TRADEABLE_ASSET_TUPLE
        if asset_str not in total_return_close_df.columns
    ]
    if missing_asset_list:
        raise ValueError(f"Missing TOTALRETURN close columns: {missing_asset_list}")
    if not isinstance(total_return_close_df.index, pd.DatetimeIndex):
        raise TypeError("total_return_close_df must use a DatetimeIndex.")
    if not total_return_close_df.index.is_monotonic_increasing:
        raise ValueError("total_return_close_df index must be sorted ascending.")
    if not total_return_close_df.index.is_unique:
        raise ValueError("total_return_close_df index must be unique.")


def month_end_decision_bool_ser(session_idx: pd.DatetimeIndex) -> pd.Series:
    month_period_idx = pd.DatetimeIndex(session_idx).to_period("M")
    # *** CRITICAL*** The next-row lookup uses only the known exchange-session
    # calendar. It does not read a future price or signal value.
    next_month_period_ser = pd.Series(month_period_idx, index=session_idx).shift(-1)
    return pd.Series(
        next_month_period_ser.isna().to_numpy()
        | (month_period_idx != next_month_period_ser.to_numpy()),
        index=session_idx,
        dtype=bool,
        name=MONTH_END_FIELD_STR,
    )


def compute_crisis_trend_signal_bundle(
    total_return_close_df: pd.DataFrame,
) -> CrisisTrendSignalBundle:
    """Compute the frozen causal signal and target-weight frames."""

    _validate_total_return_close_df(total_return_close_df)
    price_df = total_return_close_df.loc[:, list(UNIVERSE_ASSET_TUPLE)].astype(float)
    reserve_close_ser = total_return_close_df[RESERVE_ASSET_STR].astype(float)
    reserve_endpoint_valid_bool_ser = (
        np.isfinite(reserve_close_ser) & reserve_close_ser.gt(0.0)
    )

    # *** CRITICAL*** Explicit fill_method=None prevents a missing close from
    # being forward-filled across a decision boundary.
    daily_return_df = price_df.pct_change(fill_method=None)
    sign_frame_list: list[pd.DataFrame] = []
    for lookback_int in LOOKBACK_TUPLE:
        # *** CRITICAL*** Positive shifts use only Close_{T-L} and Close_T.
        asset_trailing_return_df = price_df / price_df.shift(lookback_int)
        reserve_lag_close_ser = reserve_close_ser.shift(lookback_int)
        reserve_endpoint_valid_bool_ser &= (
            np.isfinite(reserve_lag_close_ser) & reserve_lag_close_ser.gt(0.0)
        )
        reserve_trailing_return_ser = reserve_close_ser / reserve_lag_close_ser
        excess_return_df = asset_trailing_return_df.div(
            reserve_trailing_return_ser,
            axis=0,
        ) - 1.0
        sign_frame_list.append(np.sign(excess_return_df))

    signal_df = sum(sign_frame_list) / float(len(LOOKBACK_TUPLE))
    # *** CRITICAL*** Eligibility is 260 consecutive SPY-calendar rows with an
    # observed close. No missing value is forward-filled.
    eligible_df = (
        price_df.notna()
        .rolling(
            MINIMUM_HISTORY_SESSIONS_INT,
            min_periods=MINIMUM_HISTORY_SESSIONS_INT,
        )
        .sum()
        .eq(MINIMUM_HISTORY_SESSIONS_INT)
    )
    # *** CRITICAL*** Each eligible day's raw weights feed the trailing risk
    # estimate. Require SHY endpoints at T and T-{63,126,252} even outside
    # month-end; unavailable evidence must never become a zero-risk signal.
    invalid_reserve_bool_ser = (
        eligible_df.any(axis=1) & ~reserve_endpoint_valid_bool_ser
    )
    if invalid_reserve_bool_ser.any():
        invalid_reserve_ts = invalid_reserve_bool_ser[
            invalid_reserve_bool_ser
        ].index[0]
        raise ValueError(
            "Missing or invalid SHY TOTALRETURN signal endpoints "
            f"at Close_{invalid_reserve_ts}."
        )
    signal_df = signal_df.where(eligible_df)
    signal_df.loc[:, list(LONG_ONLY_ASSET_TUPLE)] = signal_df.loc[
        :,
        list(LONG_ONLY_ASSET_TUPLE),
    ].clip(lower=0.0)
    signal_df.loc[:, list(SHORT_ONLY_ASSET_TUPLE)] = signal_df.loc[
        :,
        list(SHORT_ONLY_ASSET_TUPLE),
    ].clip(upper=0.0)

    # *** CRITICAL*** The rolling volatility window ends at Close_T.
    realized_volatility_df = daily_return_df.rolling(
        REALIZED_VOLATILITY_WINDOW_INT,
        min_periods=REALIZED_VOLATILITY_WINDOW_INT,
    ).std(ddof=1) * np.sqrt(TRADING_SESSIONS_PER_YEAR_FLOAT)
    raw_weight_df = signal_df / realized_volatility_df.replace(0.0, np.nan)

    for asset_class_str in sorted(set(ASSET_CLASS_BY_ASSET_DICT.values())):
        class_asset_list = [
            asset_str
            for asset_str in UNIVERSE_ASSET_TUPLE
            if ASSET_CLASS_BY_ASSET_DICT[asset_str] == asset_class_str
        ]
        eligible_count_ser = eligible_df.loc[:, class_asset_list].sum(axis=1).replace(
            0,
            np.nan,
        )
        raw_weight_df.loc[:, class_asset_list] = raw_weight_df.loc[
            :,
            class_asset_list,
        ].div(eligible_count_ser, axis=0)
    raw_weight_df = (raw_weight_df / 4.0).fillna(0.0)

    # *** CRITICAL*** Yesterday's raw weights earn today's close-to-close
    # return. A missing return on a non-zero lagged weight invalidates the path
    # rather than being silently treated as zero.
    lagged_raw_weight_df = raw_weight_df.shift(1)
    missing_held_return_bool_ser = (
        lagged_raw_weight_df.ne(0.0) & daily_return_df.isna()
    ).any(axis=1)
    raw_pod_return_ser = (lagged_raw_weight_df * daily_return_df).sum(axis=1)
    raw_pod_return_ser = raw_pod_return_ser.mask(missing_held_return_bool_ser)
    raw_pod_return_ser.name = RAW_POD_RETURN_FIELD_STR

    # *** CRITICAL*** The ex-ante scale uses only the raw path through T.
    exante_volatility_ser = raw_pod_return_ser.rolling(
        EXANTE_VOLATILITY_WINDOW_INT,
        min_periods=EXANTE_VOLATILITY_WINDOW_INT,
    ).std(ddof=1) * np.sqrt(TRADING_SESSIONS_PER_YEAR_FLOAT)
    exante_volatility_ser.name = EXANTE_VOLATILITY_FIELD_STR
    volatility_scale_ser = (
        POD_VOLATILITY_TARGET_FLOAT / exante_volatility_ser.replace(0.0, np.nan)
    ).clip(upper=10.0)
    desired_weight_df = raw_weight_df.mul(volatility_scale_ser, axis=0).fillna(0.0)
    gross_exposure_ser = desired_weight_df.abs().sum(axis=1)
    gross_scale_ser = (
        GROSS_EXPOSURE_CAP_FLOAT / gross_exposure_ser.replace(0.0, np.nan)
    ).clip(upper=1.0).fillna(1.0)
    desired_weight_df = desired_weight_df.mul(gross_scale_ser, axis=0)

    # *** CRITICAL*** Targets are sampled only at the final session of each
    # calendar month, then carried forward without reading later prices.
    month_end_bool_ser = month_end_decision_bool_ser(desired_weight_df.index)
    month_end_target_weight_df = (
        desired_weight_df.where(month_end_bool_ser, axis=0).ffill().fillna(0.0)
    )

    return CrisisTrendSignalBundle(
        signal_df=signal_df,
        eligible_df=eligible_df,
        realized_volatility_df=realized_volatility_df,
        raw_weight_df=raw_weight_df,
        raw_pod_return_ser=raw_pod_return_ser,
        exante_volatility_ser=exante_volatility_ser,
        desired_weight_df=desired_weight_df,
        month_end_target_weight_df=month_end_target_weight_df,
    )


def build_tradeable_target_weight_ser(
    risk_target_weight_ser: pd.Series,
) -> pd.Series:
    ordered_risk_target_ser = risk_target_weight_ser.reindex(
        UNIVERSE_ASSET_TUPLE,
        fill_value=0.0,
    ).astype(float)
    long_gross_float = float(ordered_risk_target_ser.clip(lower=0.0).sum())
    reserve_weight_float = max(0.0, 1.0 - long_gross_float)
    return pd.Series(
        {
            **ordered_risk_target_ser.to_dict(),
            RESERVE_ASSET_STR: reserve_weight_float,
        },
        dtype=float,
        name="target_weight_ser",
    )


def should_rebalance_bool(
    risk_target_weight_ser: pd.Series,
    held_risk_weight_ser: pd.Series,
    initialized_bool: bool,
) -> bool:
    if not initialized_bool:
        return True
    target_ser = risk_target_weight_ser.reindex(
        UNIVERSE_ASSET_TUPLE,
        fill_value=0.0,
    ).astype(float)
    held_ser = held_risk_weight_ser.reindex(
        UNIVERSE_ASSET_TUPLE,
        fill_value=0.0,
    ).astype(float)
    max_gap_float = float((target_ser - held_ser).abs().max())
    force_exit_bool = bool(
        np.isclose(float(target_ser.abs().sum()), 0.0)
        and float(held_ser.abs().sum()) > 0.0
    )
    return bool(
        max_gap_float > REBALANCE_BAND_FLOAT + 1e-12
        or force_exit_bool
    )


def get_crisis_trend_core_data(
    config_obj: CrisisTrendCoreConfig = DEFAULT_CONFIG,
) -> pd.DataFrame:
    """Load CAPITALSPECIAL execution bars and TOTALRETURN signal closes."""

    execution_price_df = load_raw_prices(
        symbols=list(TRADEABLE_ASSET_TUPLE),
        benchmarks=list(BENCHMARK_TUPLE),
        start_date=config_obj.history_start_date_str,
        end_date=config_obj.end_date_str,
    )
    total_return_price_df = load_raw_prices(
        symbols=[],
        benchmarks=list(TRADEABLE_ASSET_TUPLE),
        start_date=config_obj.history_start_date_str,
        end_date=config_obj.end_date_str,
    )
    signal_close_df = total_return_price_df.loc[
        :,
        [(asset_str, "Close") for asset_str in TRADEABLE_ASSET_TUPLE],
    ].copy()
    signal_close_df.columns = pd.MultiIndex.from_tuples(
        [
            (signal_namespace_str(asset_str), "Close")
            for asset_str in TRADEABLE_ASSET_TUPLE
        ]
    )
    pricing_data_df = pd.concat([execution_price_df, signal_close_df], axis=1).sort_index()
    spy_calendar_bool_ser = pricing_data_df[("SPY", "Close")].notna()
    pricing_data_df = pricing_data_df.loc[spy_calendar_bool_ser].copy()
    pricing_data_df.attrs.update(execution_price_df.attrs)
    adjustment_by_symbol_dict = dict(
        pricing_data_df.attrs.get("norgate_adjustment_by_symbol_dict", {})
    )
    adjustment_by_symbol_dict.update(
        {
            signal_namespace_str(asset_str): "TOTALRETURN"
            for asset_str in TRADEABLE_ASSET_TUPLE
        }
    )
    pricing_data_df.attrs["norgate_adjustment_by_symbol_dict"] = (
        adjustment_by_symbol_dict
    )
    pricing_data_df.attrs["signal_adjustment_by_symbol_dict"] = {
        signal_namespace_str(asset_str): "TOTALRETURN"
        for asset_str in TRADEABLE_ASSET_TUPLE
    }
    return pricing_data_df


class CrisisTrendCoreStrategy(Strategy):
    """Causal Alpha-engine translation of the frozen hedge core."""

    enable_signal_audit = True
    signal_audit_sample_size = 10

    def __init__(
        self,
        config_obj: CrisisTrendCoreConfig = DEFAULT_CONFIG,
    ) -> None:
        super().__init__(
            name=config_obj.strategy_name_str,
            benchmarks=list(BENCHMARK_TUPLE),
            capital_base=config_obj.capital_base_float,
            slippage=ONE_WAY_COST_FLOAT,
            commission_per_share=0.0,
            commission_minimum=0.0,
            performance_benchmark_symbol_str=BENCHMARK_TUPLE[0],
            performance_benchmark_adjustment_str="TOTALRETURN",
        )
        self.config_obj = config_obj
        self.asset_list = list(TRADEABLE_ASSET_TUPLE)
        self.initialized_bool = False
        self.last_risk_target_weight_ser = pd.Series(dtype=float)
        self.trade_id_int = 0
        self.current_trade_id_map = {
            asset_str: default_trade_id_int() for asset_str in self.asset_list
        }
        self.daily_target_weight_row_dict_list: list[dict[str, object]] = []
        self.rebalance_target_weight_row_dict_list: list[dict[str, object]] = []
        self.signal_bundle_obj: CrisisTrendSignalBundle | None = None
        self.borrow_fee_row_dict_list: list[dict[str, object]] = []
        self.borrow_fee_total_float = 0.0
        self.configure_dividend_cash_ledger(enabled_bool=True, withholding_rate_float=0.0)
        self._data_adjustment_policy_dict.update(
            {
                "signal_adjustment_str": "TOTALRETURN",
                "execution_and_marks_adjustment_str": "CAPITALSPECIAL",
                "performance_benchmark_adjustment_str": "TOTALRETURN",
            }
        )
        self._accounting_policy_dict.update(
            {
                "reserve_asset_str": RESERVE_ASSET_STR,
                "short_proceeds_policy_str": "restricted_cash_not_reinvested",
                "short_proceeds_interest_float": 0.0,
                "annual_short_borrow_rate_float": ANNUAL_SHORT_BORROW_RATE_FLOAT,
                "short_borrow_day_count_str": "252_trading_sessions",
                "rebalance_band_decision_timing_str": "Close_T",
                "pakal_reference_band_timing_str": "Open_(T+1)",
                "reserve_execution_str": "physical_SHY_traded_with_slippage",
                "reference_reserve_str": "synthetic_SHY_cash_proxy",
                "performance_parity_status_str": "not_exact_open_to_open_parity",
            }
        )

    def compute_signals(self, pricing_data_df: pd.DataFrame) -> pd.DataFrame:
        total_return_close_df = pd.DataFrame(
            {
                asset_str: pricing_data_df[(signal_namespace_str(asset_str), "Close")]
                for asset_str in TRADEABLE_ASSET_TUPLE
            },
            index=pricing_data_df.index,
        )
        signal_bundle_obj = compute_crisis_trend_signal_bundle(total_return_close_df)
        if len(signal_bundle_obj.desired_weight_df) >= (
            0
            if self.signal_bundle_obj is None
            else len(self.signal_bundle_obj.desired_weight_df)
        ):
            self.signal_bundle_obj = signal_bundle_obj

        feature_df = pd.DataFrame(index=pricing_data_df.index)
        for asset_str in UNIVERSE_ASSET_TUPLE:
            namespace_str = signal_namespace_str(asset_str)
            feature_df[(namespace_str, SIGNAL_FIELD_STR)] = signal_bundle_obj.signal_df[
                asset_str
            ]
            feature_df[(namespace_str, ELIGIBLE_FIELD_STR)] = signal_bundle_obj.eligible_df[
                asset_str
            ]
            feature_df[(namespace_str, REALIZED_VOLATILITY_FIELD_STR)] = (
                signal_bundle_obj.realized_volatility_df[asset_str]
            )
            feature_df[(namespace_str, DESIRED_WEIGHT_FIELD_STR)] = (
                signal_bundle_obj.desired_weight_df[asset_str]
            )
        feature_df[(PORTFOLIO_NAMESPACE_STR, RAW_POD_RETURN_FIELD_STR)] = (
            signal_bundle_obj.raw_pod_return_ser
        )
        feature_df[(PORTFOLIO_NAMESPACE_STR, EXANTE_VOLATILITY_FIELD_STR)] = (
            signal_bundle_obj.exante_volatility_ser
        )
        feature_df[(PORTFOLIO_NAMESPACE_STR, MONTH_END_FIELD_STR)] = (
            month_end_decision_bool_ser(pricing_data_df.index)
        )
        feature_df.columns = pd.MultiIndex.from_tuples(feature_df.columns)
        return pd.concat([pricing_data_df, feature_df], axis=1)

    def signal_audit_fields(
        self,
        pricing_data: pd.DataFrame,
        signal_data: pd.DataFrame,
    ) -> list[tuple[str, str]]:
        audit_column_list = super().signal_audit_fields(pricing_data, signal_data)
        # A truncated prefix makes its final row appear to be month-end. The
        # price-derived signal and weight fields remain fully audited.
        return [
            column_tuple
            for column_tuple in audit_column_list
            if column_tuple != (PORTFOLIO_NAMESPACE_STR, MONTH_END_FIELD_STR)
        ]

    def _risk_target_weight_ser_from_row(
        self,
        feature_row_ser: pd.Series,
    ) -> pd.Series:
        target_weight_ser = pd.Series(
            {
                asset_str: float(
                    feature_row_ser.get(
                        (signal_namespace_str(asset_str), DESIRED_WEIGHT_FIELD_STR),
                        np.nan,
                    )
                )
                for asset_str in UNIVERSE_ASSET_TUPLE
            },
            dtype=float,
        )
        if target_weight_ser.isna().any() or not np.isfinite(
            target_weight_ser.to_numpy(dtype=float)
        ).all():
            raise RuntimeError(
                f"Incomplete Crisis Trend target at Close_{self.previous_bar}."
            )
        return target_weight_ser

    def _latest_month_end_target_weight_ser(
        self,
        data_df: pd.DataFrame,
    ) -> pd.Series:
        month_end_bool_ser = data_df[
            (PORTFOLIO_NAMESPACE_STR, MONTH_END_FIELD_STR)
        ].astype(bool)
        completed_month_end_idx = month_end_bool_ser[month_end_bool_ser].index
        if len(completed_month_end_idx) == 0:
            raise RuntimeError("No completed month-end Crisis Trend target is available.")
        latest_month_end_ts = pd.Timestamp(completed_month_end_idx[-1])
        return self._risk_target_weight_ser_from_row(data_df.loc[latest_month_end_ts])

    def _held_risk_weight_ser(self, close_row_ser: pd.Series) -> pd.Series:
        portfolio_value_float = float(self.previous_total_value)
        if not np.isfinite(portfolio_value_float) or portfolio_value_float <= 0.0:
            raise RuntimeError("Previous Crisis Trend portfolio value must be positive.")
        position_ser = self.get_positions().reindex(
            UNIVERSE_ASSET_TUPLE,
            fill_value=0.0,
        ).astype(float)
        held_weight_dict: dict[str, float] = {}
        for asset_str in UNIVERSE_ASSET_TUPLE:
            held_share_float = float(position_ser.loc[asset_str])
            if np.isclose(held_share_float, 0.0):
                held_weight_dict[asset_str] = 0.0
                continue
            close_price_float = float(close_row_ser.get((asset_str, "Close"), np.nan))
            if not np.isfinite(close_price_float) or close_price_float <= 0.0:
                raise RuntimeError(
                    f"Missing held-asset close for {asset_str} on {self.previous_bar}."
                )
            held_weight_dict[asset_str] = (
                held_share_float * close_price_float / portfolio_value_float
            )
        return pd.Series(held_weight_dict, dtype=float)

    def _new_trade_id_int(self, asset_str: str) -> int:
        self.trade_id_int += 1
        self.current_trade_id_map[asset_str] = self.trade_id_int
        return self.trade_id_int

    def _filled_position_trade_id_int(
        self,
        asset_str: str,
        current_share_int: int,
    ) -> int:
        if current_share_int == 0:
            return default_trade_id_int()
        asset_transaction_df = self.get_transactions()
        asset_transaction_df = asset_transaction_df.loc[
            asset_transaction_df["asset"] == asset_str
        ]
        if len(asset_transaction_df) == 0:
            raise RuntimeError(f"Open {asset_str} position has no filled transaction.")
        return int(asset_transaction_df.iloc[-1]["trade_id"])

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
            raise RuntimeError("Previous Crisis Trend portfolio value must be positive.")

        for asset_str in self.asset_list:
            target_weight_float = float(target_weight_ser.loc[asset_str])
            current_share_int = int(current_position_ser.loc[asset_str])
            close_price_float = float(close_row_ser.get((asset_str, "Close"), np.nan))
            if target_weight_float == 0.0 and current_share_int == 0:
                continue
            if not np.isfinite(close_price_float) or close_price_float <= 0.0:
                raise RuntimeError(
                    f"Invalid execution close for {asset_str} on {self.previous_bar}."
                )

            # *** CRITICAL*** Shares are fixed from NAV and Close_T. Open_(T+1)
            # can determine only the fill price, never the order size or band.
            target_share_int = int(
                budget_value_float * target_weight_float / close_price_float
            )
            if target_share_int == current_share_int:
                continue
            if target_share_int == 0:
                trade_id_int = self._filled_position_trade_id_int(
                    asset_str,
                    current_share_int,
                )
                self.order_target(asset_str, 0, trade_id=trade_id_int)
                self.current_trade_id_map[asset_str] = default_trade_id_int()
                continue
            trade_id_int = (
                self._new_trade_id_int(asset_str)
                if current_share_int == 0
                else self._filled_position_trade_id_int(asset_str, current_share_int)
            )
            self.order_target(
                asset_str,
                target_share_int,
                trade_id=trade_id_int,
            )

    def _record_target_weight(
        self,
        target_weight_ser: pd.Series,
        rebalance_bool: bool,
    ) -> None:
        target_record_dict = {
            "decision_date_ts": pd.Timestamp(self.previous_bar),
            **{
                asset_str: float(target_weight_ser.loc[asset_str])
                for asset_str in self.asset_list
            },
        }
        self.daily_target_weight_row_dict_list.append(target_record_dict)
        if rebalance_bool:
            self.rebalance_target_weight_row_dict_list.append(target_record_dict.copy())

    def iterate(
        self,
        data_df: pd.DataFrame,
        close_row_ser: pd.Series,
        _open_price_ser: pd.Series,
    ) -> None:
        if data_df is None or close_row_ser is None:
            return
        month_end_decision_bool = bool(
            close_row_ser.get((PORTFOLIO_NAMESPACE_STR, MONTH_END_FIELD_STR), False)
        )
        if not self.initialized_bool:
            self.last_risk_target_weight_ser = (
                self._latest_month_end_target_weight_ser(data_df)
            )
        elif month_end_decision_bool:
            self.last_risk_target_weight_ser = self._risk_target_weight_ser_from_row(
                close_row_ser
            )

        held_risk_weight_ser = self._held_risk_weight_ser(close_row_ser)
        rebalance_bool = should_rebalance_bool(
            risk_target_weight_ser=self.last_risk_target_weight_ser,
            held_risk_weight_ser=held_risk_weight_ser,
            initialized_bool=self.initialized_bool,
        )
        tradeable_target_weight_ser = build_tradeable_target_weight_ser(
            self.last_risk_target_weight_ser
        )
        if rebalance_bool:
            self._submit_target_orders(tradeable_target_weight_ser, close_row_ser)
        self._record_target_weight(tradeable_target_weight_ser, rebalance_bool)
        self.initialized_bool = True

    def apply_post_mark_accounting(self, prices: pd.DataFrame) -> None:
        short_position_ser = self.get_positions().reindex(
            SHORT_ONLY_ASSET_TUPLE,
            fill_value=0.0,
        ).astype(float)
        short_position_ser = short_position_ser.loc[short_position_ser < 0.0]
        if len(short_position_ser) == 0:
            return

        borrow_fee_total_float = 0.0
        short_notional_total_float = 0.0
        for asset_str, held_share_float in short_position_ser.items():
            close_price_float = float(prices.loc[self.current_bar, (asset_str, "Close")])
            if not np.isfinite(close_price_float) or close_price_float <= 0.0:
                raise RuntimeError(
                    f"Borrow accounting requires a valid {asset_str} close."
                )
            short_notional_float = abs(float(held_share_float)) * close_price_float
            short_notional_total_float += short_notional_float
            borrow_fee_total_float += (
                short_notional_float
                * ANNUAL_SHORT_BORROW_RATE_FLOAT
                / TRADING_SESSIONS_PER_YEAR_FLOAT
            )

        # *** CRITICAL*** The current open orders have filled and the current
        # close has marked the positions. This fee changes cash/NAV only and
        # cannot alter the prior Close_T decision.
        self.cash -= borrow_fee_total_float
        self.total_value -= borrow_fee_total_float
        self.borrow_fee_total_float += borrow_fee_total_float
        self.borrow_fee_row_dict_list.append(
            {
                "date_ts": pd.Timestamp(self.current_bar),
                "short_notional_float": short_notional_total_float,
                "annual_borrow_rate_float": ANNUAL_SHORT_BORROW_RATE_FLOAT,
                "borrow_fee_float": borrow_fee_total_float,
                "cash_after_fee_float": float(self.cash),
                "total_value_after_fee_float": float(self.total_value),
            }
        )
        self._accounting_policy_dict["borrow_fee_total_float"] = (
            self.borrow_fee_total_float
        )

    def process_orders(self, prices: pd.DataFrame) -> None:
        super().process_orders(prices)
        self.apply_post_mark_accounting(prices)

    def finalize(self, current_data_df: pd.DataFrame) -> None:
        if self.daily_target_weight_row_dict_list:
            self.daily_target_weights = pd.DataFrame(
                self.daily_target_weight_row_dict_list
            ).set_index("decision_date_ts").sort_index()
        if self.rebalance_target_weight_row_dict_list:
            self.rebalance_target_weight_df = pd.DataFrame(
                self.rebalance_target_weight_row_dict_list
            ).set_index("decision_date_ts").sort_index()
        self.borrow_fee_df = pd.DataFrame(self.borrow_fee_row_dict_list)


def build_execution_calendar_idx(
    pricing_data_df: pd.DataFrame,
    backtest_start_date_str: str,
) -> pd.DatetimeIndex:
    calendar_idx = pd.DatetimeIndex(
        pricing_data_df.index[
            pricing_data_df.index >= pd.Timestamp(backtest_start_date_str)
        ]
    )
    if len(calendar_idx) < 2:
        raise RuntimeError("Crisis Trend execution calendar requires at least two sessions.")
    return calendar_idx


def run_variant(
    show_display_bool: bool = False,
    save_results_bool: bool = True,
    output_dir_str: str = "results",
    backtest_start_date_str: str | None = None,
    end_date_str: str | None = None,
    capital_base_float: float = DEFAULT_CONFIG.capital_base_float,
) -> CrisisTrendCoreStrategy:
    config_obj = replace(
        DEFAULT_CONFIG,
        capital_base_float=capital_base_float,
        backtest_start_date_str=(
            DEFAULT_CONFIG.backtest_start_date_str
            if backtest_start_date_str is None
            else backtest_start_date_str
        ),
        end_date_str=end_date_str,
    )
    pricing_data_df = get_crisis_trend_core_data(config_obj)
    calendar_idx = build_execution_calendar_idx(
        pricing_data_df,
        config_obj.backtest_start_date_str,
    )
    strategy_obj = CrisisTrendCoreStrategy(config_obj)
    run_daily(
        strategy_obj,
        pricing_data_df,
        calendar=calendar_idx,
        show_progress=show_display_bool,
        show_signal_progress_bool=show_display_bool,
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
    run_variant(show_display_bool=True, save_results_bool=True)
