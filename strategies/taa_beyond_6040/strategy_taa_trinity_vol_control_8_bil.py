"""
Trinity volatility-control strategy with BIL as the cash substitute.

Risk assets:

    VTI, GLD, TLT

Monthly inverse-volatility base weights:

    w_{i,t}^{base}
        = (1 / sigma_{i,t}^{(63)}) / sum_j(1 / sigma_{j,t}^{(63)})

Daily exposure overlay:

    m_t^*
        = 1                                      if sigma_{portfolio,t}^{(63)} <= 0.085
        = min(1, 0.08 / sigma_{portfolio,t}^{(63)}) otherwise

The strategy rebalances when the desired risky exposure differs from the
current close-marked risky exposure by at least five percentage points. The
monthly inverse-volatility rebalance always overrides that no-trade band.

Final target weights:

    W_{i,t} = m_t * w_{i,t}^{base}
    W_{BIL,t} = 1 - m_t

There is no target leverage. Decisions use information available after Close_T
and orders execute through the engine at Open_(T+1).
"""

from __future__ import annotations

from dataclasses import replace
from typing import Sequence

import numpy as np
import pandas as pd
from IPython.display import display

from alpha.engine.backtest import run_daily
from alpha.engine.report import save_results
from strategies.taa_beyond_6040.strategy_taa_beyond_6040 import (
    Beyond6040Strategy,
    DEFAULT_CONFIG as BASE_DEFAULT_CONFIG,
    build_signal_base_weight_df,
    compute_gross_exposure_float,
    compute_month_end_inverse_vol_weight_df,
    default_trade_id_int,
    get_beyond_6040_data,
    map_month_end_weights_to_rebalance_open_df,
)


STRATEGY_NAME_STR = "strategy_taa_trinity_vol_control_8_bil"
RISK_ASSET_TUPLE = ("VTI", "GLD", "TLT")
CASH_SUBSTITUTE_ASSET_STR = "BIL"
TRADEABLE_ASSET_TUPLE = RISK_ASSET_TUPLE + (CASH_SUBSTITUTE_ASSET_STR,)
TARGET_PORTFOLIO_VOL_FLOAT = 0.08
TRIGGER_PORTFOLIO_VOL_FLOAT = 0.085
EXPOSURE_REBALANCE_BAND_FLOAT = 0.05
MONTHLY_REBALANCE_FIELD_TUPLE = ("Portfolio", "monthly_rebalance_bool")
DEFAULT_CONFIG = replace(
    BASE_DEFAULT_CONFIG,
    asset_list=TRADEABLE_ASSET_TUPLE,
    target_portfolio_vol_float=TARGET_PORTFOLIO_VOL_FLOAT,
    trigger_portfolio_vol_float=TRIGGER_PORTFOLIO_VOL_FLOAT,
)


def should_rebalance_exposure_bool(
    desired_exposure_float: float,
    current_exposure_float: float,
    exposure_rebalance_band_float: float = EXPOSURE_REBALANCE_BAND_FLOAT,
    monthly_rebalance_bool: bool = False,
) -> bool:
    """Return whether the exposure or monthly rule requires an order cycle."""
    exposure_gap_float = abs(desired_exposure_float - current_exposure_float)
    return bool(
        monthly_rebalance_bool
        or exposure_gap_float + 1e-12 >= exposure_rebalance_band_float
    )


def build_target_weight_ser(
    base_weight_ser: pd.Series,
    gross_exposure_float: float,
    risk_asset_list: Sequence[str] = RISK_ASSET_TUPLE,
    cash_substitute_asset_str: str = CASH_SUBSTITUTE_ASSET_STR,
) -> pd.Series:
    """Build non-levered risky, BIL, and engine-cash target weights."""
    risk_asset_list = list(risk_asset_list)
    risky_target_weight_ser = (
        base_weight_ser.reindex(risk_asset_list).astype(float) * gross_exposure_float
    )
    cash_substitute_weight_float = float(1.0 - gross_exposure_float)
    target_weight_ser = pd.concat(
        [
            risky_target_weight_ser,
            pd.Series({cash_substitute_asset_str: cash_substitute_weight_float}, dtype=float),
            pd.Series({"Cash": 0.0}, dtype=float),
        ]
    )

    if (target_weight_ser < -1e-12).any() or not np.isclose(
        float(target_weight_ser.sum()), 1.0, atol=1e-12
    ):
        raise ValueError("Trinity target weights must be non-negative and sum to 1.0.")
    return target_weight_ser


def build_monthly_rebalance_signal_ser(
    month_end_weight_df: pd.DataFrame,
    execution_index: pd.DatetimeIndex,
) -> pd.Series:
    """Mark the close immediately before each next-month rebalance open."""
    monthly_rebalance_ser = pd.Series(False, index=execution_index, dtype=bool)
    rebalance_weight_df = map_month_end_weights_to_rebalance_open_df(
        month_end_weight_df=month_end_weight_df,
        execution_index=execution_index,
    )
    if len(rebalance_weight_df) == 0:
        return monthly_rebalance_ser

    rebalance_position_vec = execution_index.get_indexer(rebalance_weight_df.index)
    valid_rebalance_position_vec = rebalance_position_vec[rebalance_position_vec > 0]
    if len(valid_rebalance_position_vec) == 0:
        return monthly_rebalance_ser

    # *** CRITICAL*** The monthly override is attached to Close_T, the bar
    # immediately before the mapped next-month Open_(T+1). Marking the open
    # itself would make the decision depend on same-bar information.
    decision_close_index = execution_index[valid_rebalance_position_vec - 1]
    monthly_rebalance_ser.loc[decision_close_index] = True
    return monthly_rebalance_ser


def get_first_actionable_trinity_rebalance_ts(
    pricing_data_df: pd.DataFrame,
    risk_asset_list: Sequence[str] = RISK_ASSET_TUPLE,
    tradeable_asset_list: Sequence[str] = TRADEABLE_ASSET_TUPLE,
    asset_vol_lookback_int: int = DEFAULT_CONFIG.asset_vol_lookback_int,
) -> pd.Timestamp:
    """Return the first monthly rebalance open where every ETF is tradable."""
    risk_close_key_list = [(asset_str, "Close") for asset_str in risk_asset_list]
    tradeable_open_key_list = [(asset_str, "Open") for asset_str in tradeable_asset_list]
    missing_key_list = [
        key_tuple
        for key_tuple in risk_close_key_list + tradeable_open_key_list
        if key_tuple not in pricing_data_df.columns
    ]
    if missing_key_list:
        raise RuntimeError(f"Missing Trinity price data for {missing_key_list}.")

    risk_close_df = pricing_data_df.loc[:, risk_close_key_list].astype(float)
    _, _, month_end_weight_df = compute_month_end_inverse_vol_weight_df(
        price_close_df=risk_close_df,
        asset_vol_lookback_int=asset_vol_lookback_int,
    )
    rebalance_weight_df = map_month_end_weights_to_rebalance_open_df(
        month_end_weight_df=month_end_weight_df,
        execution_index=pricing_data_df.index,
    )

    tradeable_open_df = pricing_data_df.loc[:, tradeable_open_key_list].astype(float)
    valid_tradeable_open_ser = np.isfinite(tradeable_open_df).all(axis=1) & (
        tradeable_open_df > 0.0
    ).all(axis=1)
    valid_rebalance_index = rebalance_weight_df.index.intersection(
        pricing_data_df.index[valid_tradeable_open_ser]
    )
    if len(valid_rebalance_index) == 0:
        raise RuntimeError("No actionable rebalance date was generated for Trinity Vol Control.")
    return pd.Timestamp(valid_rebalance_index[0])


class TrinityVolControlStrategy(Beyond6040Strategy):
    """Three-risk-asset inverse-volatility allocator with a BIL reserve."""

    def __init__(
        self,
        name: str,
        benchmarks: Sequence[str] | None = None,
        risk_asset_list: Sequence[str] = RISK_ASSET_TUPLE,
        cash_substitute_asset_str: str = CASH_SUBSTITUTE_ASSET_STR,
        asset_vol_lookback_int: int = DEFAULT_CONFIG.asset_vol_lookback_int,
        portfolio_vol_lookback_int: int = DEFAULT_CONFIG.portfolio_vol_lookback_int,
        target_portfolio_vol_float: float = TARGET_PORTFOLIO_VOL_FLOAT,
        trigger_portfolio_vol_float: float = TRIGGER_PORTFOLIO_VOL_FLOAT,
        exposure_rebalance_band_float: float = EXPOSURE_REBALANCE_BAND_FLOAT,
        capital_base: float = DEFAULT_CONFIG.capital_base_float,
        slippage: float = DEFAULT_CONFIG.slippage_float,
        commission_per_share: float = DEFAULT_CONFIG.commission_per_share_float,
        commission_minimum: float = DEFAULT_CONFIG.commission_minimum_float,
    ):
        self.risk_asset_list = list(risk_asset_list)
        self.cash_substitute_asset_str = str(cash_substitute_asset_str)
        if self.cash_substitute_asset_str in self.risk_asset_list:
            raise ValueError("cash_substitute_asset_str must not be a risk asset.")
        if not 0.0 <= exposure_rebalance_band_float <= 1.0:
            raise ValueError("exposure_rebalance_band_float must be between 0 and 1.")
        self.exposure_rebalance_band_float = float(exposure_rebalance_band_float)

        super().__init__(
            name=name,
            benchmarks=benchmarks,
            asset_list=self.risk_asset_list + [self.cash_substitute_asset_str],
            asset_vol_lookback_int=asset_vol_lookback_int,
            portfolio_vol_lookback_int=portfolio_vol_lookback_int,
            target_portfolio_vol_float=target_portfolio_vol_float,
            trigger_portfolio_vol_float=trigger_portfolio_vol_float,
            capital_base=capital_base,
            slippage=slippage,
            commission_per_share=commission_per_share,
            commission_minimum=commission_minimum,
        )

    def compute_signals(self, pricing_data_df: pd.DataFrame) -> pd.DataFrame:
        signal_data_df = pricing_data_df.copy()
        risk_close_key_list = [(asset_str, "Close") for asset_str in self.risk_asset_list]
        missing_key_list = [
            key_tuple for key_tuple in risk_close_key_list if key_tuple not in signal_data_df.columns
        ]
        if missing_key_list:
            raise RuntimeError(f"Missing close data for {missing_key_list}.")

        risk_close_df = signal_data_df.loc[:, risk_close_key_list].astype(float)
        risk_return_df, risk_vol_df, month_end_weight_df = (
            compute_month_end_inverse_vol_weight_df(
                price_close_df=risk_close_df,
                asset_vol_lookback_int=self.asset_vol_lookback_int,
            )
        )
        signal_base_weight_df = build_signal_base_weight_df(
            month_end_weight_df=month_end_weight_df,
            execution_index=signal_data_df.index,
        )
        monthly_rebalance_ser = build_monthly_rebalance_signal_ser(
            month_end_weight_df=month_end_weight_df,
            execution_index=signal_data_df.index,
        )

        feature_df = pd.DataFrame(index=signal_data_df.index)
        for asset_str in self.risk_asset_list:
            feature_df[(asset_str, "return_ser")] = risk_return_df[asset_str]
            feature_df[(asset_str, "volatility_ser")] = risk_vol_df[asset_str]
            feature_df[(asset_str, "base_weight_ser")] = signal_base_weight_df[asset_str]
        feature_df[MONTHLY_REBALANCE_FIELD_TUPLE] = monthly_rebalance_ser
        feature_df.columns = pd.MultiIndex.from_tuples(feature_df.columns)
        return pd.concat([signal_data_df, feature_df], axis=1)

    def signal_audit_fields(self, pricing_data: pd.DataFrame, signal_data: pd.DataFrame):
        audit_col_list = super().signal_audit_fields(pricing_data, signal_data)
        return [
            col_tuple
            for col_tuple in audit_col_list
            if col_tuple != MONTHLY_REBALANCE_FIELD_TUPLE
        ]

    def _current_base_weight_ser(self, close_row_ser: pd.Series) -> pd.Series:
        base_weight_dict = {
            asset_str: float(close_row_ser.get((asset_str, "base_weight_ser"), np.nan))
            for asset_str in self.risk_asset_list
        }
        return pd.Series(base_weight_dict, dtype=float)

    def _current_close_weight_ser(self, close_row_ser: pd.Series) -> pd.Series:
        total_value_float = float(self.previous_total_value)
        if not np.isfinite(total_value_float) or total_value_float <= 0.0:
            raise RuntimeError("Previous portfolio value must be positive.")

        current_position_ser = self.get_positions().reindex(
            self.asset_list, fill_value=0.0
        ).astype(float)
        current_weight_dict: dict[str, float] = {}
        for asset_str in self.asset_list:
            close_price_float = float(close_row_ser.get((asset_str, "Close"), np.nan))
            if not np.isfinite(close_price_float) or close_price_float <= 0.0:
                raise RuntimeError(
                    f"Invalid prior close for target asset {asset_str} on {self.current_bar}."
                )
            current_weight_dict[asset_str] = float(
                current_position_ser.loc[asset_str] * close_price_float / total_value_float
            )

        invested_weight_float = float(sum(current_weight_dict.values()))
        current_weight_dict["Cash"] = float(1.0 - invested_weight_float)
        return pd.Series(current_weight_dict, dtype=float)

    def _submit_target_orders(
        self,
        target_weight_ser: pd.Series,
        close_row_ser: pd.Series,
    ) -> None:
        current_position_ser = self.get_positions().reindex(
            self.asset_list, fill_value=0.0
        ).astype(int)
        budget_value_float = float(self.previous_total_value)

        for asset_str in self.asset_list:
            target_weight_float = float(target_weight_ser.loc[asset_str])
            current_share_int = int(current_position_ser.loc[asset_str])
            sizing_close_float = float(close_row_ser.get((asset_str, "Close"), np.nan))
            if not np.isfinite(sizing_close_float) or sizing_close_float <= 0.0:
                raise RuntimeError(
                    f"Invalid prior close for target asset {asset_str} on {self.current_bar}."
                )

            # *** CRITICAL*** Match the engine's target-percent sizing price:
            # decide the target share count from Close_T, then let the queued
            # order fill at Open_(T+1). Open_(T+1) must not decide whether an
            # order is submitted.
            target_share_int = int(
                np.floor(budget_value_float * target_weight_float / sizing_close_float)
            )
            if target_share_int == current_share_int:
                continue

            if target_share_int <= 0:
                if current_share_int <= 0:
                    continue
                self.order_target_value(
                    asset_str,
                    0.0,
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

            self.order_target_percent(
                asset_str,
                target_weight_float,
                trade_id=self.current_trade_id_map[asset_str],
            )

    def iterate(self, data_df: pd.DataFrame, close_row_ser: pd.Series, _open_price_ser: pd.Series):
        if close_row_ser is None or data_df is None:
            return

        base_weight_ser = self._current_base_weight_ser(close_row_ser)
        if base_weight_ser.isna().any():
            return

        realized_return_ser = self._realized_strategy_return_ser()
        desired_exposure_float = compute_gross_exposure_float(
            realized_return_ser=realized_return_ser,
            portfolio_vol_lookback_int=self.portfolio_vol_lookback_int,
            target_portfolio_vol_float=self.target_portfolio_vol_float,
            trigger_portfolio_vol_float=self.trigger_portfolio_vol_float,
        )

        # *** CRITICAL*** Current exposure is marked using positions and prices
        # known at Close_T. Same-day open prices must not decide whether the
        # five-percentage-point band is breached.
        current_weight_ser = self._current_close_weight_ser(close_row_ser)
        current_exposure_float = float(current_weight_ser.loc[self.risk_asset_list].sum())
        monthly_rebalance_bool = bool(
            close_row_ser.get(MONTHLY_REBALANCE_FIELD_TUPLE, False)
        )
        rebalance_exposure_bool = should_rebalance_exposure_bool(
            desired_exposure_float=desired_exposure_float,
            current_exposure_float=current_exposure_float,
            exposure_rebalance_band_float=self.exposure_rebalance_band_float,
            monthly_rebalance_bool=monthly_rebalance_bool,
        )

        if not rebalance_exposure_bool:
            self._record_daily_target_weight_ser(current_weight_ser)
            return

        target_weight_ser = build_target_weight_ser(
            base_weight_ser=base_weight_ser,
            gross_exposure_float=desired_exposure_float,
            risk_asset_list=self.risk_asset_list,
            cash_substitute_asset_str=self.cash_substitute_asset_str,
        )
        self._record_daily_target_weight_ser(target_weight_ser)
        self._submit_target_orders(
            target_weight_ser=target_weight_ser,
            close_row_ser=close_row_ser,
        )

    def finalize(self, current_data: pd.DataFrame):
        risk_close_key_list = [(asset_str, "Close") for asset_str in self.risk_asset_list]
        if not all(key_tuple in current_data.columns for key_tuple in risk_close_key_list):
            return

        risk_close_df = current_data.loc[:, risk_close_key_list].astype(float)
        _, _, month_end_weight_df = compute_month_end_inverse_vol_weight_df(
            price_close_df=risk_close_df,
            asset_vol_lookback_int=self.asset_vol_lookback_int,
        )
        self.month_end_weight_df = month_end_weight_df
        self.rebalance_weight_df = map_month_end_weights_to_rebalance_open_df(
            month_end_weight_df=month_end_weight_df,
            execution_index=current_data.index,
        )

        if len(self.daily_target_weight_map) > 0:
            self.daily_target_weights = pd.DataFrame.from_dict(
                self.daily_target_weight_map,
                orient="index",
            ).sort_index()
            self.daily_target_weights.index = pd.to_datetime(self.daily_target_weights.index)
            self.daily_target_weights = self.daily_target_weights.reindex(
                columns=self.asset_list + ["Cash"]
            )


class TrinityVolControlTimingStrategy(TrinityVolControlStrategy):
    """Timing-analyzer adapter that preserves the daily volatility state."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._timing_previous_close_total_value_float: float | None = None
        self.timing_pricing_data_df: pd.DataFrame | None = None

    def iterate(self, data_df: pd.DataFrame, close_row_ser: pd.Series, open_price_ser: pd.Series):
        current_close_total_value_float = float(self.total_value)
        previous_close_total_value_float = self._timing_previous_close_total_value_float
        if previous_close_total_value_float is not None:
            # *** CRITICAL*** ExecutionTimingAnalyzer marks total_value at
            # Close_T before calling iterate for the next execution opportunity.
            # This reproduces the Vanilla close-to-close realized return used by
            # the 63-day portfolio-volatility overlay without reading future fills.
            realized_return_float = float(
                current_close_total_value_float / previous_close_total_value_float - 1.0
            )
            self._daily_return_history_list.append(realized_return_float)
        self._timing_previous_close_total_value_float = current_close_total_value_float

        if self.timing_pricing_data_df is None:
            raise RuntimeError("Timing strategy requires the complete pricing_data_df.")
        # *** CRITICAL*** Match Vanilla dividend entitlement exactly: shares
        # held at Close_T receive Dividend_T before Open_(T+1). The engine
        # method also preserves withholding, adjustment, and duplicate-post guards.
        self._credit_dividend_cash_before_open(self.timing_pricing_data_df)
        super().iterate(data_df, close_row_ser, open_price_ser)


def _build_trinity_strategy(
    config_obj,
    capital_base_float: float,
    strategy_class_obj: type[TrinityVolControlStrategy] = TrinityVolControlStrategy,
) -> TrinityVolControlStrategy:
    return strategy_class_obj(
        name=STRATEGY_NAME_STR,
        benchmarks=config_obj.benchmark_list,
        risk_asset_list=RISK_ASSET_TUPLE,
        cash_substitute_asset_str=CASH_SUBSTITUTE_ASSET_STR,
        asset_vol_lookback_int=config_obj.asset_vol_lookback_int,
        portfolio_vol_lookback_int=config_obj.portfolio_vol_lookback_int,
        target_portfolio_vol_float=config_obj.target_portfolio_vol_float,
        trigger_portfolio_vol_float=config_obj.trigger_portfolio_vol_float,
        exposure_rebalance_band_float=EXPOSURE_REBALANCE_BAND_FLOAT,
        capital_base=capital_base_float,
        slippage=config_obj.slippage_float,
        commission_per_share=config_obj.commission_per_share_float,
        commission_minimum=config_obj.commission_minimum_float,
    )


def _execution_calendar_index(
    pricing_data_df: pd.DataFrame,
    config_obj,
    backtest_start_date_str: str | None,
) -> pd.DatetimeIndex:
    relevant_start_ts = get_first_actionable_trinity_rebalance_ts(
        pricing_data_df=pricing_data_df,
        risk_asset_list=RISK_ASSET_TUPLE,
        tradeable_asset_list=TRADEABLE_ASSET_TUPLE,
        asset_vol_lookback_int=config_obj.asset_vol_lookback_int,
    )
    calendar_start_ts = relevant_start_ts
    if backtest_start_date_str is not None:
        calendar_start_ts = max(calendar_start_ts, pd.Timestamp(backtest_start_date_str))

    # *** CRITICAL*** Keep the complete pre-start history for 63-day volatility
    # and month-end signals. Only the executable fill calendar is clipped.
    return pd.DatetimeIndex(pricing_data_df.index[pricing_data_df.index >= calendar_start_ts])


def _run_trinity_strategy(
    config_obj,
    pricing_data_df: pd.DataFrame,
    backtest_start_date_str: str | None,
    show_progress_bool: bool,
) -> TrinityVolControlStrategy:
    calendar_index = _execution_calendar_index(
        pricing_data_df=pricing_data_df,
        config_obj=config_obj,
        backtest_start_date_str=backtest_start_date_str,
    )
    strategy_obj = _build_trinity_strategy(
        config_obj=config_obj,
        capital_base_float=float(config_obj.capital_base_float),
    )
    run_daily(
        strategy_obj,
        pricing_data_df,
        calendar=calendar_index,
        show_progress=show_progress_bool,
        show_signal_progress_bool=show_progress_bool,
        audit_override_bool=None,
    )
    return strategy_obj


def run_variant(
    show_display_bool: bool = True,
    save_results_bool: bool = True,
    output_dir_str: str = "results",
    backtest_start_date_str: str | None = None,
    capital_base_float: float = DEFAULT_CONFIG.capital_base_float,
    end_date_str: str | None = None,
) -> TrinityVolControlStrategy:
    config = replace(
        DEFAULT_CONFIG,
        capital_base_float=capital_base_float,
        end_date_str=end_date_str,
    )
    pricing_data_df = get_beyond_6040_data(config=config)
    strategy_obj = _run_trinity_strategy(
        config_obj=config,
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


def build_capacity_analysis_inputs(
    show_display_bool: bool = False,
    backtest_start_date_str: str | None = None,
    capital_base_float: float = DEFAULT_CONFIG.capital_base_float,
    end_date_str: str | None = None,
) -> dict[str, object]:
    config_obj = replace(
        DEFAULT_CONFIG,
        capital_base_float=capital_base_float,
        end_date_str=end_date_str,
    )
    pricing_data_df = get_beyond_6040_data(config=config_obj)
    strategy_obj = _run_trinity_strategy(
        config_obj=config_obj,
        pricing_data_df=pricing_data_df,
        backtest_start_date_str=backtest_start_date_str,
        show_progress_bool=show_display_bool,
    )
    strategy_obj._performance_benchmark_symbol_str = str(config_obj.benchmark_list[0])
    return {
        "strategy_obj": strategy_obj,
        "pricing_data_df": pricing_data_df,
        "execution_policy_str": "MOO",
        "impact_profile_str": "MOO_ETF_PROXY",
    }


def build_execution_timing_analysis_inputs() -> dict[str, object]:
    config_obj = DEFAULT_CONFIG
    pricing_data_df = get_beyond_6040_data(config=config_obj)
    calendar_index = _execution_calendar_index(
        pricing_data_df=pricing_data_df,
        config_obj=config_obj,
        backtest_start_date_str=None,
    )

    def strategy_factory_fn() -> TrinityVolControlTimingStrategy:
        strategy_obj = _build_trinity_strategy(
            config_obj=config_obj,
            capital_base_float=float(config_obj.capital_base_float),
            strategy_class_obj=TrinityVolControlTimingStrategy,
        )
        strategy_obj.timing_pricing_data_df = pricing_data_df
        return strategy_obj

    return {
        "strategy_factory_fn": strategy_factory_fn,
        "pricing_data_df": pricing_data_df,
        "calendar_idx": calendar_index,
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
    config_obj = DEFAULT_CONFIG
    pricing_data_df = get_beyond_6040_data(config=config_obj)
    calendar_index = _execution_calendar_index(
        pricing_data_df=pricing_data_df,
        config_obj=config_obj,
        backtest_start_date_str=None,
    )
    return {
        "strategy_name_str": STRATEGY_NAME_STR,
        "capital_base_float": float(config_obj.capital_base_float),
        "config_obj": config_obj,
        "pricing_data_df": pricing_data_df,
        "calendar_idx": calendar_index,
    }


def build_stress_test_strategy_obj(context_dict: dict[str, object]) -> TrinityVolControlStrategy:
    return _build_trinity_strategy(
        config_obj=context_dict["config_obj"],
        capital_base_float=float(context_dict["capital_base_float"]),
    )


if __name__ == "__main__":
    run_variant()
