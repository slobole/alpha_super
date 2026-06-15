"""
Weekly decision-bar variant of the wired DV2 mean-reversion strategy.

Old daily behavior:
    daily signal_t -> next daily open execution

New weekly research behavior:
    completed weekly decision bar_t -> next daily open execution

The backtest still uses daily rows for execution, position marking, and metrics.
Only the signal inputs are sampled from completed weekly OHLC bars.

Core formulas
-------------
For stock i on completed weekly decision date t:

    weekly_return_{i,t}^{26}
        = Close_W{i,t} / Close_W{i,t-26} - 1

    weekly_sma_{i,t}^{40}
        = (1 / 40) * sum_{k=0}^{39} Close_W{i,t-k}

    weekly_exit_{i,t}
        = 1[Close_W{i,t} > High_W{i,t-1}]

    eligible_{i,t}
        = 1[PIT member_t]
        * 1[DV2_W{i,t} < 10]
        * 1[Close_W{i,t} > weekly_sma_{i,t}^{40}]
        * 1[weekly_return_{i,t}^{26} > 0.05]

Execution mapping:
    decision date t is the actual last trading close of a completed W-FRI week.
    Orders generated from t fill at the next tradable daily open.
"""

from __future__ import annotations

from collections import defaultdict
from typing import List

import numpy as np
import pandas as pd
import talib
from IPython.display import display

from alpha.engine.backtest import run_daily
from alpha.engine.report import save_results
from alpha.indicators import dv2_indicator
from data.norgate_loader import build_index_constituent_matrix, load_raw_prices
from strategies.dv2.strategy_mr_dv2 import (
    DVO2Strategy,
    default_trade_id_int,
    get_asof_universe_symbol_list,
)
from strategies.weekly_bar_utils import build_completed_week_ohlcv_df


weekly_decision_marker_field_str = "weekly_decision_bool"
weekly_return_field_str = "weekly_return_26w_ser"
weekly_natr_field_str = "weekly_natr_14w_ser"
weekly_dv2_field_str = "weekly_dv2_26w_ser"
weekly_sma_field_str = "weekly_sma_40w_ser"
weekly_previous_high_field_str = "weekly_previous_high_ser"


def get_prices(
    symbol_list: List[str],
    benchmark_list: List[str],
    start_date_str: str = "1998-01-01",
    end_date_str: str | None = None,
) -> pd.DataFrame:
    return load_raw_prices(symbol_list, benchmark_list, start_date_str, end_date_str)


def _weekly_decision_row_bool(close_row_ser: pd.Series) -> bool:
    if not isinstance(close_row_ser.index, pd.MultiIndex):
        return False
    try:
        weekly_marker_ser = close_row_ser.xs(weekly_decision_marker_field_str, level=1)
    except KeyError:
        return False
    return bool(weekly_marker_ser.dropna().astype(bool).any())


class WeeklyDVO2Strategy(DVO2Strategy):
    """
    DV2 entry/exit logic evaluated only on completed weekly decision bars.

    The default weekly windows preserve the original daily strategy's rough
    calendar horizon for the 126-day return/DV2 and 200-day trend filters:
    126 trading days ~= 26 weeks, and 200 trading days ~= 40 weeks.
    """

    def __init__(
        self,
        name: str,
        benchmarks: list | tuple,
        capital_base=100_000,
        slippage: float = 0.00025,
        commission_per_share: float = 0.005,
        commission_minimum: float = 1.0,
        max_positions_int: int = 10,
        return_lookback_week_int: int = 26,
        dv2_window_week_int: int = 26,
        sma_window_week_int: int = 40,
        natr_window_week_int: int = 14,
    ):
        super().__init__(
            name=name,
            benchmarks=benchmarks,
            capital_base=capital_base,
            slippage=slippage,
            commission_per_share=commission_per_share,
            commission_minimum=commission_minimum,
        )
        if max_positions_int <= 0:
            raise ValueError("max_positions_int must be positive.")
        if return_lookback_week_int <= 0:
            raise ValueError("return_lookback_week_int must be positive.")
        if dv2_window_week_int <= 0:
            raise ValueError("dv2_window_week_int must be positive.")
        if sma_window_week_int <= 0:
            raise ValueError("sma_window_week_int must be positive.")
        if natr_window_week_int <= 0:
            raise ValueError("natr_window_week_int must be positive.")

        self.max_positions = int(max_positions_int)
        self.return_lookback_week_int = int(return_lookback_week_int)
        self.dv2_window_week_int = int(dv2_window_week_int)
        self.sma_window_week_int = int(sma_window_week_int)
        self.natr_window_week_int = int(natr_window_week_int)
        self.trade_id = 0
        self.current_trade = defaultdict(default_trade_id_int)
        self.universe_df = None

    def compute_signals(self, pricing_data_df: pd.DataFrame) -> pd.DataFrame:
        signal_data_df = pricing_data_df.copy()
        weekly_bar_df = build_completed_week_ohlcv_df(signal_data_df)
        if len(weekly_bar_df.index) == 0:
            return signal_data_df

        symbol_list = weekly_bar_df.columns.get_level_values(0).unique()
        weekly_feature_map: dict[tuple[str, str], pd.Series] = {}

        for symbol_str in symbol_list:
            if str(symbol_str).startswith("$"):
                continue
            required_column_tuple = (
                (symbol_str, "Close"),
                (symbol_str, "High"),
                (symbol_str, "Low"),
            )
            if any(column_key not in weekly_bar_df.columns for column_key in required_column_tuple):
                continue

            close_price_ser = weekly_bar_df[(symbol_str, "Close")].astype(float)
            high_price_ser = weekly_bar_df[(symbol_str, "High")].astype(float)
            low_price_ser = weekly_bar_df[(symbol_str, "Low")].astype(float)

            # *** CRITICAL *** Weekly return uses only trailing completed-week closes.
            weekly_feature_map[(symbol_str, weekly_return_field_str)] = (
                close_price_ser / close_price_ser.shift(self.return_lookback_week_int) - 1.0
            )
            # *** CRITICAL *** NATR is computed on completed weekly high/low/close bars only.
            weekly_feature_map[(symbol_str, weekly_natr_field_str)] = pd.Series(
                talib.NATR(
                    high_price_ser.to_numpy(dtype=float),
                    low_price_ser.to_numpy(dtype=float),
                    close_price_ser.to_numpy(dtype=float),
                    timeperiod=self.natr_window_week_int,
                ),
                index=close_price_ser.index,
            )
            weekly_feature_map[(symbol_str, weekly_dv2_field_str)] = dv2_indicator(
                close_price_ser,
                high_price_ser,
                low_price_ser,
                length_int=self.dv2_window_week_int,
            )
            # *** CRITICAL *** Weekly SMA is a trailing mean over completed weekly closes only.
            weekly_feature_map[(symbol_str, weekly_sma_field_str)] = close_price_ser.rolling(
                window=self.sma_window_week_int,
                min_periods=self.sma_window_week_int,
            ).mean()
            # *** CRITICAL *** Exit compares this completed weekly close to the
            # previous completed weekly high, never to an in-progress week.
            weekly_feature_map[(symbol_str, weekly_previous_high_field_str)] = high_price_ser.shift(1)
            weekly_feature_map[(symbol_str, weekly_decision_marker_field_str)] = pd.Series(
                True,
                index=close_price_ser.index,
                dtype=bool,
            )

        if len(weekly_feature_map) == 0:
            return signal_data_df

        weekly_feature_df = pd.DataFrame(weekly_feature_map, index=weekly_bar_df.index)
        weekly_feature_df = weekly_feature_df.reindex(signal_data_df.index)
        return pd.concat([signal_data_df, weekly_feature_df], axis=1)

    def iterate(self, data_df: pd.DataFrame, close_row_ser: pd.Series, open_price_ser: pd.Series):
        if close_row_ser is None or data_df is None or not _weekly_decision_row_bool(close_row_ser):
            return

        position_ser = self.get_positions()
        long_position_ser = position_ser[position_ser > 0]
        long_slots_int = self.max_positions - len(long_position_ser)

        for symbol_str in long_position_ser.index:
            close_price_float = close_row_ser.get((symbol_str, "Close"), np.nan)
            previous_high_float = close_row_ser.get((symbol_str, weekly_previous_high_field_str), np.nan)
            if pd.notna(close_price_float) and pd.notna(previous_high_float) and float(close_price_float) > float(previous_high_float):
                self.order_target_value(symbol_str, 0, trade_id=self.current_trade[symbol_str])
                long_slots_int += 1

        capital_per_trade_float = self.previous_total_value / float(self.max_positions)
        opportunity_symbol_list = self.get_opportunities(close_row_ser)

        while long_slots_int > 0 and len(opportunity_symbol_list) > 0:
            symbol_str = opportunity_symbol_list.pop(0)
            if self.get_position(symbol_str) != 0:
                continue

            self.trade_id += 1
            self.current_trade[symbol_str] = self.trade_id
            self.order_value(symbol_str, capital_per_trade_float, trade_id=self.trade_id)
            long_slots_int -= 1

    def get_opportunities(self, close_row_ser: pd.Series) -> list:
        candidate_df = close_row_ser.unstack()
        required_field_list = [
            "Close",
            weekly_return_field_str,
            weekly_natr_field_str,
            weekly_dv2_field_str,
            weekly_sma_field_str,
        ]
        missing_field_list = [
            field_str for field_str in required_field_list if field_str not in candidate_df.columns
        ]
        if len(missing_field_list) > 0:
            return []

        candidate_df = candidate_df.dropna(subset=required_field_list, how="any")
        candidate_df = candidate_df[~candidate_df.index.astype(str).str.startswith("$")]
        candidate_df = candidate_df[
            (candidate_df[weekly_dv2_field_str].astype(float) < 10.0)
            & (candidate_df["Close"].astype(float) > candidate_df[weekly_sma_field_str].astype(float))
            & (candidate_df[weekly_return_field_str].astype(float) > 0.05)
        ].sort_values(weekly_natr_field_str, ascending=False)

        universe_symbol_list = get_asof_universe_symbol_list(
            self.universe_df,
            pd.Timestamp(self.previous_bar),
        )
        return candidate_df[candidate_df.index.isin(universe_symbol_list)].index.tolist()


def run_variant(
    show_display_bool: bool = True,
    save_results_bool: bool = True,
    output_dir_str: str = "results",
    backtest_start_date_str: str = "2004-01-01",
    capital_base_float: float = 100_000.0,
    end_date_str: str | None = None,
):
    benchmark_list = ["$SPX"]
    index_symbol_list, universe_df = build_index_constituent_matrix(indexname="S&P 500")
    pricing_data_df = get_prices(
        index_symbol_list,
        benchmark_list,
        start_date_str="1998-01-01",
        end_date_str=end_date_str,
    )

    strategy_obj = WeeklyDVO2Strategy(
        name="strategy_mr_dv2_weekly",
        benchmarks=benchmark_list,
        capital_base=capital_base_float,
        slippage=0.00025,
        commission_per_share=0.005,
        commission_minimum=1.0,
    )
    strategy_obj.universe_df = universe_df
    strategy_obj.trade_id = 0
    strategy_obj.current_trade = defaultdict(default_trade_id_int)

    # *** CRITICAL *** Weekly signals use full pre-start history for weekly
    # indicator warmup, but executable fills begin only on the requested daily
    # calendar. Orders still fill at the next daily open after a week-close signal.
    calendar_idx = pricing_data_df.index[pricing_data_df.index >= pd.Timestamp(backtest_start_date_str)]
    run_daily(
        strategy_obj,
        pricing_data_df,
        calendar_idx,
        show_progress=show_display_bool,
        show_signal_progress_bool=show_display_bool,
    )

    strategy_obj.universe_df = None

    if show_display_bool:
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 1000)
        display(strategy_obj.summary)
        display(strategy_obj.summary_trades)

    if save_results_bool:
        save_results(strategy_obj, output_dir=output_dir_str)

    return strategy_obj


if __name__ == "__main__":
    run_variant()
