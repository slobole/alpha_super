"""
IBIT end-of-month window strategy.

TL;DR: Hold a Bitcoin ETF during the final 5 trading days of each month and
liquidate at the first trading day of the next month open.

Core formulas
-------------
Let the ordered trading dates in month m be:

    D_m = {d_{m,1}, d_{m,2}, ..., d_{m,N_m}}

For a hold window of n = 5 trading days, the target weight is:

    w_t = 1[t in {d_{m,N_m-n+1}, ..., d_{m,N_m}}]

The open-to-open execution mapping is:

    entry_t = 1[w_{t-1} = 0 and w_t = 1]
    exit_t  = 1[w_{t-1} = 1 and w_t = 0]

Target sizing uses the engine's generic overnight approximation:

    q_t^{intent} ~= floor(V_{t-1} * w_t / O_t)

where:

    V_{t-1} = previous total portfolio value
    O_t     = current bar open

This means the strategy:

    enters at the open of the first trading day inside the final 5-day window
    exits at the open of the first trading day of the next month

This means the strategy:

    enters at the open of the first trading day inside the final 5-day window
    exits at the open of the first trading day of the next month

There is no same-bar month-end close fill assumption.
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
class EomWindowConfig:
    trade_symbol_candidate_list: tuple[str, ...] = ("IBIT", "FBTC", "BITB", "ARKB", "BITO")
    benchmark_list: tuple[str, ...] = ("SPY",)
    hold_day_count_int: int = 5
    start_date_str: str = "2024-01-01"
    end_date_str: str | None = None

    def __post_init__(self):
        if len(self.trade_symbol_candidate_list) == 0:
            raise ValueError("trade_symbol_candidate_list must contain at least one symbol.")
        if self.hold_day_count_int <= 0:
            raise ValueError("hold_day_count_int must be positive.")


DEFAULT_CONFIG = EomWindowConfig()


def load_yahoo_ohlcv_df(
    symbol_str: str,
    start_date_str: str,
    end_date_str: str | None = None,
) -> pd.DataFrame:
    """
    Load adjusted OHLCV bars from Yahoo Finance for one symbol.

    For the chosen Bitcoin ETF candidate set, dividend-adjustment leakage is
    not the main concern because these funds do not distribute meaningful cash
    dividends like a classic equity ETF.
    """
    price_df = yf.download(
        symbol_str,
        start=start_date_str,
        end=end_date_str,
        auto_adjust=True,
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

    field_order_list = [field_str for field_str in ["Open", "High", "Low", "Close", "Volume"] if field_str in price_df.columns]
    price_df = price_df[field_order_list].sort_index().dropna(subset=required_field_list, how="any")
    return price_df


def attach_symbol_level(price_df: pd.DataFrame, symbol_str: str) -> pd.DataFrame:
    field_list = list(price_df.columns)
    price_df = price_df.copy()
    price_df.columns = pd.MultiIndex.from_tuples([(symbol_str, field_str) for field_str in field_list])
    return price_df


def load_pricing_data_df(
    trade_symbol_candidate_list: Sequence[str],
    benchmark_list: Sequence[str],
    start_date_str: str,
    end_date_str: str | None = None,
) -> tuple[pd.DataFrame, str]:
    """
    Select the first tradable Bitcoin ETF candidate with data and load the
    aligned pricing frame used by the engine.
    """
    selection_error_list: list[str] = []
    selected_symbol_str: str | None = None
    trade_price_df: pd.DataFrame | None = None

    for candidate_symbol_str in trade_symbol_candidate_list:
        try:
            candidate_price_df = load_yahoo_ohlcv_df(
                candidate_symbol_str,
                start_date_str=start_date_str,
                end_date_str=end_date_str,
            )
        except Exception as exc:
            selection_error_list.append(f"{candidate_symbol_str}: {exc}")
            continue

        if len(candidate_price_df) == 0:
            selection_error_list.append(f"{candidate_symbol_str}: no rows returned")
            continue

        selected_symbol_str = candidate_symbol_str
        trade_price_df = candidate_price_df
        break

    if selected_symbol_str is None or trade_price_df is None:
        error_message_str = "; ".join(selection_error_list)
        raise RuntimeError(f"Unable to load any trade symbol candidate. Details: {error_message_str}")

    frame_list: list[pd.DataFrame] = [attach_symbol_level(trade_price_df, selected_symbol_str)]
    for benchmark_symbol_str in benchmark_list:
        benchmark_price_df = load_yahoo_ohlcv_df(
            benchmark_symbol_str,
            start_date_str=start_date_str,
            end_date_str=end_date_str,
        )
        if len(benchmark_price_df) == 0:
            raise RuntimeError(f"Benchmark {benchmark_symbol_str} returned no data.")

        benchmark_price_df = benchmark_price_df.reindex(trade_price_df.index)
        frame_list.append(attach_symbol_level(benchmark_price_df, benchmark_symbol_str))

    pricing_data_df = pd.concat(frame_list, axis=1).sort_index()
    pricing_data_df = pricing_data_df.loc[trade_price_df.index]
    return pricing_data_df, selected_symbol_str


def build_eom_target_weight_ser(
    trading_index: pd.DatetimeIndex,
    hold_day_count_int: int,
) -> pd.Series:
    """
    Build a daily target-weight series for the final n trading days of each
    calendar month using the known trading calendar.
    """
    if hold_day_count_int <= 0:
        raise ValueError("hold_day_count_int must be positive.")

    ordered_index = pd.DatetimeIndex(pd.to_datetime(trading_index)).sort_values()
    target_weight_ser = pd.Series(0.0, index=ordered_index, dtype=float)

    # *** CRITICAL*** The end-of-month window is mapped from the trading
    # calendar itself. We mark the last n trading dates in each month and trade
    # at their opens, never at the month-end close.
    month_period_index = ordered_index.to_period("M")
    for month_period, month_date_index in pd.Series(ordered_index, index=ordered_index).groupby(month_period_index).groups.items():
        month_trading_index = pd.DatetimeIndex(month_date_index).sort_values()
        hold_index = month_trading_index[-hold_day_count_int:]
        target_weight_ser.loc[hold_index] = 1.0

    return target_weight_ser


class EomIbitStrategy(Strategy):
    """
    Single-asset seasonal pod:

        long selected Bitcoin ETF during the final 5 trading days of each month
        flat at the first trading day of the next month open
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
        # flips between previous_bar and current_bar. That preserves the intended
        # open-to-open timing and prevents same-bar month-end leakage.
        if np.isclose(previous_target_weight_float, current_target_weight_float, atol=1e-12):
            return

        open_price_float = float(open_price_ser.loc[self.trade_symbol_str])
        if not np.isfinite(open_price_float) or open_price_float <= 0.0:
            raise RuntimeError(
                f"Invalid open price for {self.trade_symbol_str} on {self.current_bar}: {open_price_float}"
            )

        current_position_float = float(self.get_position(self.trade_symbol_str))

        if current_target_weight_float <= 0.0:
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

    pricing_data_df, trade_symbol_str = load_pricing_data_df(
        trade_symbol_candidate_list=config.trade_symbol_candidate_list,
        benchmark_list=config.benchmark_list,
        start_date_str=config.start_date_str,
        end_date_str=config.end_date_str,
    )
    daily_target_weight_ser = build_eom_target_weight_ser(
        trading_index=pricing_data_df.index,
        hold_day_count_int=config.hold_day_count_int,
    )

    strategy = EomIbitStrategy(
        name="strategy_eom_ibit",
        benchmarks=config.benchmark_list,
        trade_symbol_str=trade_symbol_str,
        daily_target_weight_ser=daily_target_weight_ser,
        capital_base=100_000,
        slippage=0.0001,
        commission_per_share=0.005,
        commission_minimum=1.0,
    )
    strategy.show_taa_weights_report = True
    strategy.daily_target_weights = daily_target_weight_ser.to_frame(name=trade_symbol_str)

    calendar_idx = pricing_data_df.index
    run_daily(strategy, pricing_data_df, calendar_idx)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)

    print(f"Selected trade symbol: {trade_symbol_str}")
    display(strategy.summary)
    display(strategy.summary_trades)

    save_results(strategy)
