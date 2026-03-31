"""
Beginning-of-month short TLT strategy.

TL;DR: Short `TLT` during the first 5 trading days of each calendar month,
then cover at the first trading-day open after that window.

Core formulas
-------------
Let the ordered trading dates in month m be:

    D_m = {d_{m,1}, d_{m,2}, ..., d_{m,N_m}}

For a hold window of n = 5 trading days and a signed target weight of
`w_short = -1.0`, the daily target weight is:

    w_t = w_short * 1[t in {d_{m,1}, d_{m,2}, ..., d_{m,n}}]

The open-to-open execution mapping is:

    entry_t = 1[w_{t-1} = 0 and w_t = w_short]
    exit_t  = 1[w_{t-1} = w_short and w_t = 0]

Target sizing uses the engine's generic overnight approximation:

    q_t^{intent} ~= floor(V_{t-1} * w_t / O_t)

where:

    V_{t-1} = previous total portfolio value
    O_t     = current bar open

This means the strategy:

    enters short at the first trading-day open of each month
    remains short through the first 5 trading sessions
    covers at the first trading-day open after the window

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
class BomTltShortConfig:
    trade_symbol_str: str = "TLT"
    benchmark_list: tuple[str, ...] = ("$SPX",)
    hold_day_count_int: int = 5
    target_weight_float: float = -1.0
    start_date_str: str = "2002-08-01"
    end_date_str: str | None = None

    def __post_init__(self):
        if self.hold_day_count_int <= 0:
            raise ValueError("hold_day_count_int must be positive.")
        if self.target_weight_float >= 0.0:
            raise ValueError("target_weight_float must be negative for the short-only TLT strategy.")


DEFAULT_CONFIG = BomTltShortConfig()


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


def build_bom_target_weight_ser(
    trading_index: pd.DatetimeIndex,
    hold_day_count_int: int,
    target_weight_float: float,
) -> pd.Series:
    """
    Build a signed daily target-weight series for the first n trading days of
    each calendar month using the known trading calendar.
    """
    if hold_day_count_int <= 0:
        raise ValueError("hold_day_count_int must be positive.")
    if target_weight_float >= 0.0:
        raise ValueError("target_weight_float must be negative for the short-only TLT strategy.")

    ordered_index = pd.DatetimeIndex(pd.to_datetime(trading_index)).sort_values()
    target_weight_ser = pd.Series(0.0, index=ordered_index, dtype=float)

    # *** CRITICAL*** The beginning-of-month window is mapped from the trading
    # calendar itself. We mark the first n trading dates in each month and
    # trade at their opens, never by peeking at later bars in the month.
    month_period_index = ordered_index.to_period("M")
    month_group_map = pd.Series(ordered_index, index=ordered_index).groupby(month_period_index).groups
    for _, month_date_index in month_group_map.items():
        month_trading_index = pd.DatetimeIndex(month_date_index).sort_values()
        hold_index = month_trading_index[:hold_day_count_int]
        target_weight_ser.loc[hold_index] = float(target_weight_float)

    return target_weight_ser


class BomTltShortStrategy(Strategy):
    """
    Single-asset seasonal pod:

        short TLT during the first 5 trading days of each month
        flat at the first trading-day open after the window
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

        # *** CRITICAL*** The position state changes only when the target weight
        # flips between previous_bar and current_bar. That preserves the
        # intended open-to-open timing and prevents same-bar month-start
        # leakage.
        if np.isclose(previous_target_weight_float, current_target_weight_float, atol=1e-12):
            return

        open_price_float = float(open_price_ser.loc[self.trade_symbol_str])
        if not np.isfinite(open_price_float) or open_price_float <= 0.0:
            raise RuntimeError(
                f"Invalid open price for {self.trade_symbol_str} on {self.current_bar}: {open_price_float}"
            )

        current_position_float = float(self.get_position(self.trade_symbol_str))

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

        if np.isclose(current_position_float, 0.0, atol=1e-12) or self.active_trade_id_int == default_trade_id_int():
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
    daily_target_weight_ser = build_bom_target_weight_ser(
        trading_index=pricing_data_df.index,
        hold_day_count_int=config.hold_day_count_int,
        target_weight_float=config.target_weight_float,
    )

    strategy = BomTltShortStrategy(
        name="strategy_bom_tlt_short",
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
