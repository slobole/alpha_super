"""
Beginning-of-month TLT long-short split-month strategy.

TL;DR: Short `TLT` during the first 5 trading days of each calendar month,
stay flat through trading day 15, then hold a long `TLT` position through the
rest of that month.

Core formulas
-------------
Let the ordered trading dates in month m be:

    D_m = {d_{m,1}, d_{m,2}, ..., d_{m,N_m}}

Let:

    k_short = 5
    k_long = 16
    w_short = -1.0
    w_flat = 0.0
    w_long = +1.0

The signed daily target weight is:

    w_t = w_short * 1[t in {d_{m,1}, ..., d_{m,min(k_short, N_m)}}]
        + w_long  * 1[t in {d_{m,k_long}, d_{m,k_long+1}, ..., d_{m,N_m}}]

The open-to-open execution mapping is:

    short_entry_t = 1[w_{t-1} != w_short and w_t = w_short]
    short_exit_t  = 1[w_{t-1} = w_short and w_t = w_flat]
    long_entry_t  = 1[w_{t-1} != w_long  and w_t = w_long]

Target sizing uses the engine's generic overnight approximation:

    q_t^{intent} ~= floor(V_{t-1} * w_t / O_t)

where:

    V_{t-1} = previous total portfolio value
    O_t     = current bar open

This means the strategy:

    starts each month short at the first trading-day open
    stays short through trading day 5
    exits to cash at the open of trading day 6
    enters long at the open of trading day 16
    stays long through month-end

There is no same-bar close execution assumption.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Sequence

import numpy as np
import pandas as pd
from IPython.display import display

WORKSPACE_ROOT_PATH = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT_PATH))

from alpha.engine.backtest import run_daily
from alpha.engine.report import save_results
from alpha.engine.strategy import Strategy
from data.norgate_loader import load_raw_prices


def default_trade_id_int() -> int:
    return -1


@dataclass(frozen=True)
class BomTltLongShortConfig:
    trade_symbol_str: str = "TLT"
    benchmark_list: tuple[str, ...] = ("$SPX",)
    short_hold_day_count_int: int = 5
    long_start_trading_day_int: int = 16
    short_target_weight_float: float = -1.0
    long_target_weight_float: float = 1.0
    start_date_str: str = "2002-08-01"
    end_date_str: str | None = None

    def __post_init__(self):
        if self.short_hold_day_count_int <= 0:
            raise ValueError("short_hold_day_count_int must be positive.")
        if self.long_start_trading_day_int <= 1:
            raise ValueError("long_start_trading_day_int must be greater than 1.")
        if self.long_start_trading_day_int <= self.short_hold_day_count_int:
            raise ValueError("long_start_trading_day_int must be greater than short_hold_day_count_int.")
        if self.short_target_weight_float >= 0.0:
            raise ValueError("short_target_weight_float must be negative.")
        if self.long_target_weight_float <= 0.0:
            raise ValueError("long_target_weight_float must be positive.")


DEFAULT_CONFIG = BomTltLongShortConfig()


def get_prices(
    trade_symbol_str: str,
    benchmark_list: Sequence[str],
    start_date_str: str,
    end_date_str: str | None = None,
) -> pd.DataFrame:
    return load_raw_prices(
        symbols=[trade_symbol_str],
        benchmarks=list(benchmark_list),
        start_date=start_date_str,
        end_date=end_date_str,
    )


def build_bom_long_short_target_weight_ser(
    trading_day_index: pd.DatetimeIndex,
    short_hold_day_count_int: int,
    long_start_trading_day_int: int,
    short_target_weight_float: float,
    long_target_weight_float: float,
) -> pd.Series:
    """
    Build a signed daily target-weight series that is short for the first
    k_short trading days of each month, flat until trading day k_long - 1, and
    long from trading day k_long through the remainder of that month.
    """
    if short_hold_day_count_int <= 0:
        raise ValueError("short_hold_day_count_int must be positive.")
    if long_start_trading_day_int <= 1:
        raise ValueError("long_start_trading_day_int must be greater than 1.")
    if long_start_trading_day_int <= short_hold_day_count_int:
        raise ValueError("long_start_trading_day_int must be greater than short_hold_day_count_int.")
    if short_target_weight_float >= 0.0:
        raise ValueError("short_target_weight_float must be negative.")
    if long_target_weight_float <= 0.0:
        raise ValueError("long_target_weight_float must be positive.")

    ordered_day_index = pd.DatetimeIndex(pd.to_datetime(trading_day_index)).sort_values()
    target_weight_ser = pd.Series(0.0, index=ordered_day_index, dtype=float)

    # *** CRITICAL*** The short/flat/long regime split is built from the
    # realized trading calendar for each month. The first k_short observed
    # trading dates are short, the middle segment is flat, and only dates at
    # or after trading day k_long are long.
    month_period_index = ordered_day_index.to_period("M")
    month_group_map = pd.Series(ordered_day_index, index=ordered_day_index).groupby(month_period_index).groups
    for _, month_date_index in month_group_map.items():
        month_day_index = pd.DatetimeIndex(month_date_index).sort_values()
        short_window_index = month_day_index[:short_hold_day_count_int]
        long_window_index = month_day_index[long_start_trading_day_int - 1:]

        target_weight_ser.loc[short_window_index] = float(short_target_weight_float)
        target_weight_ser.loc[long_window_index] = float(long_target_weight_float)

    return target_weight_ser


class BomTltLongShortStrategy(Strategy):
    """
    Single-asset seasonal pod:

        short TLT through trading day 5 of each month
        flat through trading day 15
        long TLT for the remainder of the month
    """

    enable_signal_audit = False
    signal_audit_sample_size = 10

    def __init__(
        self,
        name: str,
        benchmarks: Sequence[str],
        trade_symbol_str: str,
        daily_target_weight_ser: pd.Series,
        capital_base: float = 100_000,
        slippage: float = 0.0001,
        commission_per_share: float = 0.005,
        commission_minimum: float = 1.0,
    ):
        super().__init__(
            name=name,
            benchmarks=list(benchmarks),
            capital_base=capital_base,
            slippage=slippage,
            commission_per_share=commission_per_share,
            commission_minimum=commission_minimum,
        )
        self.trade_symbol_str = str(trade_symbol_str)
        self.daily_target_weight_ser = daily_target_weight_ser.astype(float).copy().sort_index()
        self.trade_id_int = 0
        self.active_trade_id_int = default_trade_id_int()

    def compute_signals(self, pricing_data_df: pd.DataFrame) -> pd.DataFrame:
        return pricing_data_df

    def iterate(self, data_df: pd.DataFrame, close_row_ser: pd.Series, open_price_ser: pd.Series):
        if close_row_ser is None or self.previous_bar is None:
            return

        previous_target_weight_float = float(self.daily_target_weight_ser.get(pd.Timestamp(self.previous_bar), 0.0))
        current_target_weight_float = float(self.daily_target_weight_ser.get(pd.Timestamp(self.current_bar), 0.0))

        # *** CRITICAL*** The regime switch is detected only from the
        # previous_bar/current_bar target change. That preserves the intended
        # open-to-open timing at the month start, the day-6 exit, and the
        # day-16 long entry.
        if np.isclose(previous_target_weight_float, current_target_weight_float, atol=1e-12):
            return

        open_price_float = float(open_price_ser.loc[self.trade_symbol_str])
        if not np.isfinite(open_price_float) or open_price_float <= 0.0:
            raise RuntimeError(
                f"Invalid open price for {self.trade_symbol_str} on {self.current_bar}: {open_price_float}"
            )

        current_position_float = float(self.get_position(self.trade_symbol_str))
        sign_flip_bool = (
            not np.isclose(current_position_float, 0.0, atol=1e-12)
            and current_position_float * current_target_weight_float < 0.0
        )

        # Submit the close leg first on a sign flip so trade accounting keeps
        # the old direction and the new direction in separate trade_id groups.
        if sign_flip_bool:
            exit_trade_id_int = None if self.active_trade_id_int == default_trade_id_int() else self.active_trade_id_int
            self.order_target_value(
                self.trade_symbol_str,
                0.0,
                trade_id=exit_trade_id_int,
            )
            self.active_trade_id_int = default_trade_id_int()

        if np.isclose(current_target_weight_float, 0.0, atol=1e-12):
            if np.isclose(current_position_float, 0.0, atol=1e-12):
                return

            exit_trade_id_int = None if self.active_trade_id_int == default_trade_id_int() else self.active_trade_id_int
            self.order_target_value(
                self.trade_symbol_str,
                0.0,
                trade_id=exit_trade_id_int,
            )
            self.active_trade_id_int = default_trade_id_int()
            return

        if (
            np.isclose(current_position_float, 0.0, atol=1e-12)
            or sign_flip_bool
            or self.active_trade_id_int == default_trade_id_int()
        ):
            self.trade_id_int += 1
            self.active_trade_id_int = self.trade_id_int

        self.order_target_percent(
            self.trade_symbol_str,
            current_target_weight_float,
            trade_id=self.active_trade_id_int,
        )


if __name__ == "__main__":
    config = DEFAULT_CONFIG

    pricing_data_df = get_prices(
        trade_symbol_str=config.trade_symbol_str,
        benchmark_list=config.benchmark_list,
        start_date_str=config.start_date_str,
        end_date_str=config.end_date_str,
    )
    daily_target_weight_ser = build_bom_long_short_target_weight_ser(
        trading_day_index=pricing_data_df.index,
        short_hold_day_count_int=config.short_hold_day_count_int,
        long_start_trading_day_int=config.long_start_trading_day_int,
        short_target_weight_float=config.short_target_weight_float,
        long_target_weight_float=config.long_target_weight_float,
    )

    strategy = BomTltLongShortStrategy(
        name="strategy_bom_tlt_long_short",
        benchmarks=config.benchmark_list,
        trade_symbol_str=config.trade_symbol_str,
        daily_target_weight_ser=daily_target_weight_ser,
        capital_base=100_000,
        slippage=0.0001,
        commission_per_share=0.005,
        commission_minimum=1.0,
    )
    strategy.daily_target_weights = daily_target_weight_ser.to_frame(name=config.trade_symbol_str)

    calendar_idx = pricing_data_df.index
    run_daily(strategy, pricing_data_df, calendar_idx)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)

    display(strategy.summary)
    display(strategy.summary_trades)

    save_results(strategy)
