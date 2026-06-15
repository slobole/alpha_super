"""
Weekly decision-bar variant of the wired QPI + IBS + RSI-exit strategy.

The backtest keeps daily execution and valuation rows. Signal features are
formed from completed weekly OHLCV bars and sampled only at week-close
decision dates.

Core formulas
-------------
For stock i on completed weekly decision date t:

    r^{(3w)}_{i,t} = Close_W{i,t} / Close_W{i,t-3} - 1

    SMA40W_{i,t} = (1 / 40) * sum_{k=0}^{39} Close_W{i,t-k}

    IBS_W{i,t} = (Close_W{i,t} - Low_W{i,t}) / (High_W{i,t} - Low_W{i,t})

    RSI2W_{i,t} = RSI(Close_W{i,t}, length = 2)

    eligible_{i,t}
        = 1[PIT member_t]
        * 1[QPI_W{i,t} < 30]
        * 1[Close_W{i,t} > SMA40W_{i,t}]
        * 1[r^{(3w)}_{i,t} < 0]
        * 1[IBS_W{i,t} < 0.1]

    exit_{i,t} = 1[IBS_W{i,t} > 0.90] or 1[RSI2W_{i,t} > 90]
"""

from __future__ import annotations

from collections import defaultdict
from typing import List

import numpy as np
import pandas as pd
import talib
from IPython.display import display

from alpha.engine.backtest import run_daily
from alpha.engine.qp_indicator_fast import (
    _compute_qpi_value_arr,
    _pct_change_periods,
    _rolling_rank_and_down_count,
)
from alpha.engine.report import save_results
from alpha.engine.strategy import Strategy
from alpha.indicators import ibs_indicator
from data.norgate_loader import build_index_constituent_matrix, load_raw_prices
from strategies.qpi.strategy_mr_qpi_ibs_rsi_exit import (
    default_trade_id_int,
    get_asof_universe_symbol_list,
)
from strategies.weekly_bar_utils import build_completed_week_ohlcv_df


weekly_decision_marker_field_str = "weekly_decision_bool"
three_week_return_field_str = "three_week_return_ser"
weekly_qpi_field_str = "weekly_qpi_value_ser"
weekly_sma_field_str = "weekly_sma_40w_price_ser"
weekly_ibs_field_str = "weekly_ibs_value_ser"
weekly_rsi2_field_str = "weekly_rsi2_value_ser"
weekly_turnover_field_str = "weekly_turnover_ser"


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


def _weekly_qpi_indicator_ser(
    close_price_ser: pd.Series,
    window_week_int: int,
    lookback_years_int: int,
) -> pd.Series:
    if window_week_int <= 0:
        raise ValueError("window_week_int must be positive.")
    if lookback_years_int <= 0:
        raise ValueError("lookback_years_int must be positive.")

    lookback_window_week_int = int(lookback_years_int) * 52
    close_price_arr = close_price_ser.to_numpy(dtype=np.float64, copy=False)

    # *** CRITICAL *** QPI weekly return uses only C_W,t and C_W,t-window.
    return_value_arr = _pct_change_periods(close_price_arr, int(window_week_int))
    percent_rank_arr, down_count_arr = _rolling_rank_and_down_count(
        return_value_arr,
        lookback_window_week_int,
    )
    qpi_value_arr = _compute_qpi_value_arr(
        return_value_arr,
        percent_rank_arr,
        down_count_arr,
        lookback_window_week_int,
    )
    return pd.Series(qpi_value_arr, index=close_price_ser.index, name=close_price_ser.name, dtype=float)


class WeeklyQPIIbsRsiExitStrategy(Strategy):
    """QPI/IBS/RSI exits evaluated only on completed weekly decision bars."""

    enable_signal_audit = False
    signal_audit_sample_size = 10

    def __init__(
        self,
        name: str,
        benchmarks: list[str] | tuple[str, ...],
        capital_base: float = 100_000,
        slippage: float = 0.00025,
        commission_per_share: float = 0.005,
        commission_minimum: float = 1.0,
        max_positions_int: int = 10,
        qpi_threshold_float: float = 30.0,
        sma_window_week_int: int = 40,
        qpi_window_week_int: int = 3,
        qpi_lookback_years_int: int = 5,
        return_lookback_week_int: int = 3,
        max_entry_weekly_ibs_float: float = 0.1,
        exit_weekly_ibs_threshold_float: float = 0.90,
        rsi_window_week_int: int = 2,
        exit_weekly_rsi2_threshold_float: float = 90.0,
    ):
        super().__init__(
            name=name,
            benchmarks=list(benchmarks),
            capital_base=capital_base,
            slippage=slippage,
            commission_per_share=commission_per_share,
            commission_minimum=commission_minimum,
        )

        if max_positions_int <= 0:
            raise ValueError("max_positions_int must be positive.")
        if sma_window_week_int <= 0:
            raise ValueError("sma_window_week_int must be positive.")
        if qpi_window_week_int <= 0:
            raise ValueError("qpi_window_week_int must be positive.")
        if qpi_lookback_years_int <= 0:
            raise ValueError("qpi_lookback_years_int must be positive.")
        if return_lookback_week_int <= 0:
            raise ValueError("return_lookback_week_int must be positive.")
        if not 0.0 <= max_entry_weekly_ibs_float <= 1.0:
            raise ValueError("max_entry_weekly_ibs_float must lie in [0, 1].")
        if not 0.0 <= exit_weekly_ibs_threshold_float <= 1.0:
            raise ValueError("exit_weekly_ibs_threshold_float must lie in [0, 1].")
        if rsi_window_week_int <= 0:
            raise ValueError("rsi_window_week_int must be positive.")
        if not 0.0 <= exit_weekly_rsi2_threshold_float <= 100.0:
            raise ValueError("exit_weekly_rsi2_threshold_float must lie in [0, 100].")

        self.trade_id_int = 0
        self.current_trade_map: defaultdict[str, int] = defaultdict(default_trade_id_int)
        self.universe_df: pd.DataFrame | None = None

        self.max_positions_int = int(max_positions_int)
        self.qpi_threshold_float = float(qpi_threshold_float)
        self.sma_window_week_int = int(sma_window_week_int)
        self.qpi_window_week_int = int(qpi_window_week_int)
        self.qpi_lookback_years_int = int(qpi_lookback_years_int)
        self.return_lookback_week_int = int(return_lookback_week_int)
        self.max_entry_weekly_ibs_float = float(max_entry_weekly_ibs_float)
        self.exit_weekly_ibs_threshold_float = float(exit_weekly_ibs_threshold_float)
        self.rsi_window_week_int = int(rsi_window_week_int)
        self.exit_weekly_rsi2_threshold_float = float(exit_weekly_rsi2_threshold_float)

    def compute_signals(self, pricing_data_df: pd.DataFrame) -> pd.DataFrame:
        signal_data_df = pricing_data_df.copy()
        weekly_bar_df = build_completed_week_ohlcv_df(signal_data_df)
        if len(weekly_bar_df.index) == 0:
            return signal_data_df

        symbol_list = weekly_bar_df.columns.get_level_values(0).unique()
        tradeable_symbol_list = [
            str(symbol_str)
            for symbol_str in symbol_list
            if not str(symbol_str).startswith("$")
            and (symbol_str, "Close") in weekly_bar_df.columns
            and (symbol_str, "High") in weekly_bar_df.columns
            and (symbol_str, "Low") in weekly_bar_df.columns
            and (symbol_str, "Turnover") in weekly_bar_df.columns
        ]
        if len(tradeable_symbol_list) == 0:
            return signal_data_df

        close_price_df = pd.DataFrame(
            {symbol_str: weekly_bar_df[(symbol_str, "Close")] for symbol_str in tradeable_symbol_list},
            index=weekly_bar_df.index,
        )
        high_price_df = pd.DataFrame(
            {symbol_str: weekly_bar_df[(symbol_str, "High")] for symbol_str in tradeable_symbol_list},
            index=weekly_bar_df.index,
        )
        low_price_df = pd.DataFrame(
            {symbol_str: weekly_bar_df[(symbol_str, "Low")] for symbol_str in tradeable_symbol_list},
            index=weekly_bar_df.index,
        )
        turnover_df = pd.DataFrame(
            {symbol_str: weekly_bar_df[(symbol_str, "Turnover")] for symbol_str in tradeable_symbol_list},
            index=weekly_bar_df.index,
        )

        # *** CRITICAL *** Three-week return uses only trailing completed-week closes.
        three_week_return_df = close_price_df.pct_change(
            periods=self.return_lookback_week_int,
            fill_method=None,
        )
        # *** CRITICAL *** Weekly SMA is a trailing average over completed weekly closes only.
        weekly_sma_price_df = close_price_df.rolling(
            window=self.sma_window_week_int,
            min_periods=self.sma_window_week_int,
        ).mean()
        weekly_ibs_value_df = ibs_indicator(close_price_df, high_price_df, low_price_df)

        qpi_value_map: dict[str, pd.Series] = {}
        rsi2_value_map: dict[str, pd.Series] = {}
        for symbol_str in tradeable_symbol_list:
            close_price_ser = close_price_df[symbol_str].astype(float)
            qpi_value_map[symbol_str] = _weekly_qpi_indicator_ser(
                close_price_ser,
                window_week_int=self.qpi_window_week_int,
                lookback_years_int=self.qpi_lookback_years_int,
            )
            # *** CRITICAL *** RSI2W is computed from trailing weekly closes only.
            rsi2_value_map[symbol_str] = pd.Series(
                talib.RSI(close_price_ser.to_numpy(dtype=float), timeperiod=self.rsi_window_week_int),
                index=close_price_ser.index,
            )

        weekly_feature_map = {
            three_week_return_field_str: three_week_return_df,
            weekly_qpi_field_str: pd.DataFrame(qpi_value_map, index=weekly_bar_df.index),
            weekly_sma_field_str: weekly_sma_price_df,
            weekly_ibs_field_str: weekly_ibs_value_df,
            weekly_rsi2_field_str: pd.DataFrame(rsi2_value_map, index=weekly_bar_df.index),
            weekly_turnover_field_str: turnover_df,
            weekly_decision_marker_field_str: pd.DataFrame(True, index=weekly_bar_df.index, columns=tradeable_symbol_list),
        }

        feature_frame_list: list[pd.DataFrame] = []
        for field_str, field_df in weekly_feature_map.items():
            feature_df = field_df.reindex(signal_data_df.index)
            feature_df.columns = pd.MultiIndex.from_tuples(
                [(symbol_str, field_str) for symbol_str in feature_df.columns]
            )
            feature_frame_list.append(feature_df)

        return pd.concat([signal_data_df] + feature_frame_list, axis=1)

    def iterate(self, data_df: pd.DataFrame, close_row_ser: pd.Series, open_price_ser: pd.Series):
        if close_row_ser is None or data_df is None or not _weekly_decision_row_bool(close_row_ser):
            return

        position_ser = self.get_positions()
        long_position_ser = position_ser[position_ser > 0]
        long_slots_int = self.max_positions_int - len(long_position_ser)

        for symbol_str in long_position_ser.index:
            weekly_ibs_float = close_row_ser.get((symbol_str, weekly_ibs_field_str), np.nan)
            weekly_rsi2_float = close_row_ser.get((symbol_str, weekly_rsi2_field_str), np.nan)

            exit_for_ibs_bool = (
                pd.notna(weekly_ibs_float)
                and float(weekly_ibs_float) > self.exit_weekly_ibs_threshold_float
            )
            exit_for_rsi2_bool = (
                pd.notna(weekly_rsi2_float)
                and float(weekly_rsi2_float) > self.exit_weekly_rsi2_threshold_float
            )

            if exit_for_ibs_bool or exit_for_rsi2_bool:
                self.order_target_value(
                    symbol_str,
                    0.0,
                    trade_id=self.current_trade_map[symbol_str],
                )
                long_slots_int += 1

        capital_per_trade_float = self.previous_total_value / float(self.max_positions_int)
        opportunity_symbol_list = self.get_opportunity_list(close_row_ser)

        while long_slots_int > 0 and len(opportunity_symbol_list) > 0:
            symbol_str = opportunity_symbol_list.pop(0)
            if self.get_position(symbol_str) != 0:
                continue

            self.trade_id_int += 1
            self.current_trade_map[symbol_str] = self.trade_id_int
            self.order_value(symbol_str, capital_per_trade_float, trade_id=self.trade_id_int)
            long_slots_int -= 1

    def get_opportunity_list(self, close_row_ser: pd.Series) -> list[str]:
        candidate_df = close_row_ser.unstack()
        candidate_df = candidate_df[~candidate_df.index.astype(str).str.startswith("$")]

        required_field_list = [
            "Close",
            weekly_turnover_field_str,
            weekly_qpi_field_str,
            weekly_sma_field_str,
            three_week_return_field_str,
            weekly_ibs_field_str,
        ]
        missing_field_list = [
            field_str for field_str in required_field_list if field_str not in candidate_df.columns
        ]
        if len(missing_field_list) > 0:
            return []

        candidate_df = candidate_df.dropna(subset=required_field_list)
        candidate_df = candidate_df[
            candidate_df[weekly_qpi_field_str].astype(float) < self.qpi_threshold_float
        ]
        candidate_df = candidate_df[
            candidate_df["Close"].astype(float) > candidate_df[weekly_sma_field_str].astype(float)
        ]
        candidate_df = candidate_df[
            candidate_df[three_week_return_field_str].astype(float) < 0.0
        ]
        candidate_df = candidate_df[
            candidate_df[weekly_ibs_field_str].astype(float) < self.max_entry_weekly_ibs_float
        ]

        if self.universe_df is not None:
            universe_symbol_list = get_asof_universe_symbol_list(
                self.universe_df,
                pd.Timestamp(self.previous_bar),
            )
            candidate_df = candidate_df[candidate_df.index.isin(universe_symbol_list)]

        candidate_df = candidate_df.assign(symbol_str=candidate_df.index.astype(str))
        candidate_df = candidate_df.sort_values(
            by=[weekly_turnover_field_str, "symbol_str"],
            ascending=[False, True],
            kind="mergesort",
        )
        return candidate_df.index.tolist()


def run_variant(
    show_display_bool: bool = True,
    save_results_bool: bool = True,
    output_dir_str: str = "results",
    backtest_start_date_str: str = "2004-01-01",
    capital_base_float: float = 100_000.0,
    end_date_str: str | None = None,
):
    benchmark_list = ["$SPX"]
    symbol_list, universe_df = build_index_constituent_matrix(indexname="S&P 500")
    pricing_data_df = get_prices(
        symbol_list,
        benchmark_list,
        start_date_str="1998-01-01",
        end_date_str=end_date_str,
    )

    strategy_obj = WeeklyQPIIbsRsiExitStrategy(
        name="strategy_mr_qpi_ibs_rsi_exit_weekly",
        benchmarks=benchmark_list,
        capital_base=capital_base_float,
        slippage=0.00025,
        commission_per_share=0.005,
        commission_minimum=1.0,
    )
    strategy_obj.universe_df = universe_df
    strategy_obj.trade_id_int = 0
    strategy_obj.current_trade_map = defaultdict(default_trade_id_int)

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
