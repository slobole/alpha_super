"""
Beginning-of-next-month QQQ trend strategy conditioned on the prior month's
first 15 trading days.

TL;DR: For each month m, compute the cumulative log return of `QQQ` over the
first 15 trading days. If that value is positive, go long `QQQ` at the first
trading-day open of month m+1 and hold through trading day 5 of month m+1.
If the value is non-positive, stay in cash.

Core formulas
-------------
Let the ordered trading dates in month m be:

    D_m = {d_{m,1}, d_{m,2}, ..., d_{m,N_m}}

Define:

    L = 15   # signal window length in trading days
    H = 5    # hold window length in next month

Signal-month cumulative log return:

    g_m^{(L)}
        = log(Close_{d_{m,L}} / Open_{d_{m,1}})

which is equivalent to:

    g_m^{(L)}
        = log(Close_{d_{m,1}} / Open_{d_{m,1}})
        + sum_{j=2}^{L} log(Close_{d_{m,j}} / Close_{d_{m,j-1}})

Eligibility rule:

    eligible_m = 1[g_m^{(L)} > 0]

Next-month target weight:

    w_t = 1[eligible_{m-1} = 1 and t in {d_{m,1}, d_{m,2}, ..., d_{m,H}}]

Open-to-open execution mapping under the engine contract:

    entry_t = 1[w_{t-1} = 0 and w_t = 1]
    exit_t  = 1[w_{t-1} = 1 and w_t = 0]

Target sizing uses the engine's generic overnight approximation:

    q_t^{intent} ~= floor(V_{t-1} * w_t / O_t)

where:

    V_{t-1} = previous total portfolio value
    O_t     = current bar open

There is no same-bar close execution assumption.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Sequence

import numpy as np
import pandas as pd
import yfinance as yf
from IPython.display import display

WORKSPACE_ROOT_PATH = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT_PATH))

from alpha.engine.backtest import run_daily
from alpha.engine.report import save_results
from alpha.engine.strategy import Strategy


def default_trade_id_int() -> int:
    return -1


@dataclass(frozen=True)
class BomQqqFirst15LogTrendConfig:
    trade_symbol_str: str = "QQQ"
    benchmark_list: tuple[str, ...] = ("QQQ",)
    signal_day_count_int: int = 15
    hold_day_count_int: int = 5
    target_weight_float: float = 1.0
    start_date_str: str = "1999-03-10"
    end_date_str: str | None = None

    def __post_init__(self):
        if len(self.trade_symbol_str) == 0:
            raise ValueError("trade_symbol_str must be non-empty.")
        if self.signal_day_count_int <= 0:
            raise ValueError("signal_day_count_int must be positive.")
        if self.hold_day_count_int <= 0:
            raise ValueError("hold_day_count_int must be positive.")
        if self.target_weight_float <= 0.0:
            raise ValueError("target_weight_float must be positive.")


DEFAULT_CONFIG = BomQqqFirst15LogTrendConfig()


def load_yahoo_ohlcv_df(
    symbol_str: str,
    start_date_str: str,
    end_date_str: str | None = None,
) -> pd.DataFrame:
    """
    Load raw Yahoo OHLCV bars for one symbol.

    We deliberately keep raw OHLC fields (`auto_adjust=False`) so fills and
    valuation use tradable price paths rather than dividend-adjusted synthetic
    total-return bars.
    """
    price_df = yf.download(
        symbol_str,
        start=start_date_str,
        end=end_date_str,
        auto_adjust=False,
        progress=False,
        threads=False,
    )

    if len(price_df) == 0:
        return pd.DataFrame()

    if isinstance(price_df.columns, pd.MultiIndex):
        if symbol_str in price_df.columns.get_level_values(-1):
            price_df = price_df.xs(symbol_str, axis=1, level=-1)
        else:
            price_df = price_df.droplevel(-1, axis=1)

    price_df = price_df.copy()
    price_df.index = pd.to_datetime(price_df.index)
    if getattr(price_df.index, "tz", None) is not None:
        price_df.index = price_df.index.tz_localize(None)

    required_field_list = ["Open", "High", "Low", "Close"]
    missing_field_list = [field_str for field_str in required_field_list if field_str not in price_df.columns]
    if len(missing_field_list) > 0:
        raise RuntimeError(f"{symbol_str} is missing required fields: {missing_field_list}")

    field_order_list = [
        field_str
        for field_str in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
        if field_str in price_df.columns
    ]
    price_df = price_df[field_order_list].sort_index().dropna(subset=required_field_list, how="any")
    return price_df


def attach_symbol_level(price_df: pd.DataFrame, symbol_str: str) -> pd.DataFrame:
    field_list = list(price_df.columns)
    labeled_price_df = price_df.copy()
    labeled_price_df.columns = pd.MultiIndex.from_tuples([(symbol_str, field_str) for field_str in field_list])
    return labeled_price_df


def load_pricing_data_df(
    trade_symbol_str: str,
    benchmark_list: Sequence[str],
    start_date_str: str,
    end_date_str: str | None = None,
) -> pd.DataFrame:
    symbol_list = list(dict.fromkeys([trade_symbol_str] + list(benchmark_list)))
    price_frame_list: list[pd.DataFrame] = []

    for symbol_str in symbol_list:
        price_df = load_yahoo_ohlcv_df(
            symbol_str=symbol_str,
            start_date_str=start_date_str,
            end_date_str=end_date_str,
        )
        if len(price_df) == 0:
            raise RuntimeError(f"{symbol_str} returned no data.")
        price_frame_list.append(attach_symbol_level(price_df=price_df, symbol_str=symbol_str))

    pricing_data_df = pd.concat(price_frame_list, axis=1).sort_index()
    return pricing_data_df


def build_month_signal_df(
    open_price_ser: pd.Series,
    close_price_ser: pd.Series,
    signal_day_count_int: int,
    hold_day_count_int: int,
    target_weight_float: float,
) -> pd.DataFrame:
    """
    Build a month-by-month signal table for the prior-month first-15-day rule.
    """
    if signal_day_count_int <= 0:
        raise ValueError("signal_day_count_int must be positive.")
    if hold_day_count_int <= 0:
        raise ValueError("hold_day_count_int must be positive.")
    if target_weight_float <= 0.0:
        raise ValueError("target_weight_float must be positive.")

    ordered_index = pd.DatetimeIndex(pd.to_datetime(close_price_ser.index)).sort_values()
    open_price_ser = open_price_ser.reindex(ordered_index).astype(float)
    close_price_ser = close_price_ser.reindex(ordered_index).astype(float)

    month_group_map = pd.Series(ordered_index, index=ordered_index).groupby(ordered_index.to_period("M")).groups
    month_signal_row_list: list[dict[str, object]] = []

    for month_period, month_date_index in month_group_map.items():
        month_trading_index = pd.DatetimeIndex(month_date_index).sort_values()
        month_observation_count_int = len(month_trading_index)
        if month_observation_count_int < signal_day_count_int:
            continue

        signal_start_bar_ts = pd.Timestamp(month_trading_index[0])
        signal_end_bar_ts = pd.Timestamp(month_trading_index[signal_day_count_int - 1])

        signal_start_open_float = float(open_price_ser.loc[signal_start_bar_ts])
        signal_end_close_float = float(close_price_ser.loc[signal_end_bar_ts])
        if not np.isfinite(signal_start_open_float) or signal_start_open_float <= 0.0:
            raise RuntimeError(f"Invalid signal_start open on {signal_start_bar_ts}: {signal_start_open_float}")
        if not np.isfinite(signal_end_close_float) or signal_end_close_float <= 0.0:
            raise RuntimeError(f"Invalid signal_end close on {signal_end_bar_ts}: {signal_end_close_float}")

        first15_log_return_float = float(np.log(signal_end_close_float / signal_start_open_float))
        eligible_bool = bool(first15_log_return_float > 0.0)

        next_month_period = month_period + 1
        hold_start_bar_ts = pd.NaT
        hold_end_bar_ts = pd.NaT

        # *** CRITICAL*** The trade window is mapped strictly from signal month
        # m to trading days 1..H of month m+1. That prevents same-month overlap
        # between the signal window and the execution window.
        if next_month_period in month_group_map:
            next_month_trading_index = pd.DatetimeIndex(month_group_map[next_month_period]).sort_values()

            # *** CRITICAL*** We only schedule a next-month trade when the
            # backtest sample contains at least H observed trading days in that
            # next month. This avoids partial-window leakage at the sample end.
            if len(next_month_trading_index) >= hold_day_count_int:
                hold_start_bar_ts = pd.Timestamp(next_month_trading_index[0])
                hold_end_bar_ts = pd.Timestamp(next_month_trading_index[hold_day_count_int - 1])

        month_signal_row_list.append(
            {
                "signal_month_period": month_period,
                "month_observation_count_int": int(month_observation_count_int),
                "signal_start_bar_ts": signal_start_bar_ts,
                "signal_end_bar_ts": signal_end_bar_ts,
                "first15_log_return_float": first15_log_return_float,
                "eligible_bool": eligible_bool,
                "target_weight_float": float(target_weight_float) if eligible_bool else 0.0,
                "hold_start_bar_ts": hold_start_bar_ts,
                "hold_end_bar_ts": hold_end_bar_ts,
            }
        )

    month_signal_df = pd.DataFrame(month_signal_row_list)
    if len(month_signal_df) == 0:
        return pd.DataFrame(
            columns=[
                "signal_month_period",
                "month_observation_count_int",
                "signal_start_bar_ts",
                "signal_end_bar_ts",
                "first15_log_return_float",
                "eligible_bool",
                "target_weight_float",
                "hold_start_bar_ts",
                "hold_end_bar_ts",
            ]
        )

    month_signal_df["signal_month_period_str"] = month_signal_df["signal_month_period"].astype(str)
    month_signal_df = month_signal_df.set_index("signal_month_period_str", drop=True)
    return month_signal_df


def build_daily_target_weight_ser(
    trading_index: pd.DatetimeIndex,
    month_signal_df: pd.DataFrame,
) -> pd.Series:
    daily_target_weight_ser = pd.Series(0.0, index=pd.DatetimeIndex(trading_index), dtype=float)
    if len(month_signal_df) == 0:
        return daily_target_weight_ser

    eligible_signal_df = month_signal_df.loc[month_signal_df["eligible_bool"]].copy()
    eligible_signal_df = eligible_signal_df.dropna(subset=["hold_start_bar_ts", "hold_end_bar_ts"])

    for _, signal_row_ser in eligible_signal_df.iterrows():
        hold_start_bar_ts = pd.Timestamp(signal_row_ser["hold_start_bar_ts"])
        hold_end_bar_ts = pd.Timestamp(signal_row_ser["hold_end_bar_ts"])
        target_weight_float = float(signal_row_ser["target_weight_float"])

        daily_target_weight_ser.loc[hold_start_bar_ts:hold_end_bar_ts] = target_weight_float

    return daily_target_weight_ser


class BomQqqFirst15LogTrendStrategy(Strategy):
    """
    Single-asset QQQ monthly trend pod:

        compute month m first-15-day log return
        if positive, long QQQ for trading days 1-5 of month m+1
        otherwise stay flat
    """

    enable_signal_audit = False
    signal_audit_sample_size = 10

    def __init__(
        self,
        name: str,
        benchmarks: Sequence[str],
        trade_symbol_str: str,
        daily_target_weight_ser: pd.Series,
        month_signal_df: pd.DataFrame,
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
        self.month_signal_df = month_signal_df.copy()
        self.trade_id_int = 0
        self.active_trade_id_int = default_trade_id_int()

    def compute_signals(self, pricing_data_df: pd.DataFrame) -> pd.DataFrame:
        return pricing_data_df

    def iterate(self, data_df: pd.DataFrame, close_row_ser: pd.Series, open_price_ser: pd.Series):
        if close_row_ser is None or self.previous_bar is None:
            return

        previous_target_weight_float = float(self.daily_target_weight_ser.get(pd.Timestamp(self.previous_bar), 0.0))
        current_target_weight_float = float(self.daily_target_weight_ser.get(pd.Timestamp(self.current_bar), 0.0))

        # *** CRITICAL*** Position changes are triggered only when the target
        # weight flips between previous_bar and current_bar. That preserves the
        # intended open-to-open timing for next-month day-1 entry and day-6
        # exit after holding days 1-5.
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

    pricing_data_df = load_pricing_data_df(
        trade_symbol_str=config.trade_symbol_str,
        benchmark_list=config.benchmark_list,
        start_date_str=config.start_date_str,
        end_date_str=config.end_date_str,
    )

    open_price_ser = pricing_data_df[(config.trade_symbol_str, "Open")].astype(float)
    close_price_ser = pricing_data_df[(config.trade_symbol_str, "Close")].astype(float)
    month_signal_df = build_month_signal_df(
        open_price_ser=open_price_ser,
        close_price_ser=close_price_ser,
        signal_day_count_int=config.signal_day_count_int,
        hold_day_count_int=config.hold_day_count_int,
        target_weight_float=config.target_weight_float,
    )
    daily_target_weight_ser = build_daily_target_weight_ser(
        trading_index=pricing_data_df.index,
        month_signal_df=month_signal_df,
    )

    strategy = BomQqqFirst15LogTrendStrategy(
        name="strategy_bom_qqq_first15_log_trend",
        benchmarks=config.benchmark_list,
        trade_symbol_str=config.trade_symbol_str,
        daily_target_weight_ser=daily_target_weight_ser,
        month_signal_df=month_signal_df,
        capital_base=100_000,
        slippage=0.0001,
        commission_per_share=0.005,
        commission_minimum=1.0,
    )
    strategy.show_taa_weights_report = True
    strategy.daily_target_weights = daily_target_weight_ser.to_frame(name=config.trade_symbol_str)

    calendar_idx = pricing_data_df.index
    run_daily(strategy, pricing_data_df, calendar_idx)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)

    print("Month signal preview:")
    display(month_signal_df.head(12))
    display(strategy.summary)
    display(strategy.summary_trades)

    save_results(strategy)
