"""
QPI long-only mean-reversion strategy with IBS entry and IBS / RSI2 exits.

Core formulas
-------------
For stock i on decision date t:

    r^{(3)}_{i,t} = C_{i,t} / C_{i,t-3} - 1

    SMA200_{i,t} = (1 / 200) * sum_{k=0}^{199} C_{i,t-k}

    IBS_{i,t} = (C_{i,t} - L_{i,t}) / (H_{i,t} - L_{i,t})

    RSI2_{i,t} = RSI(C_{i,t}, length = 2)

    eligible_{i,t}
        = 1[PIT member_t]
        * 1[QPI_{i,t} < 30]
        * 1[C_{i,t} > SMA200_{i,t}]
        * 1[r^{(3)}_{i,t} < 0]
        * 1[IBS_{i,t} < 0.1]

Exit rules
----------

    exit_{i,t} = 1[IBS_{i,t} > 0.90] or 1[RSI2_{i,t} > 90]

Execution philosophy
--------------------
Equal-slot sizing with next-open execution:

    capital_per_trade_t = V_{t-1} / N_max

The decision is made using information through previous_bar = t and orders
fill at the next bar open t+1 under the engine contract.
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
from alpha.engine.strategy import Strategy
from alpha.indicators import ibs_indicator, qp_indicator
from data.norgate_loader import build_index_constituent_matrix, load_raw_prices


def default_trade_id_int() -> int:
    return -1


def get_prices(
    symbol_list: List[str],
    benchmark_list: List[str],
    start_date_str: str = "1998-01-01",
    end_date_str: str | None = None,
) -> pd.DataFrame:
    return load_raw_prices(symbol_list, benchmark_list, start_date_str, end_date_str)


class QPIIbsRsiExitStrategy(Strategy):
    """
    QPI stock pod with IBS oversold entries and overbought IBS / RSI2 exits.

    Entry rules:

        eligible_{i,t}
            = 1[PIT member_t]
            * 1[QPI_t < 30]
            * 1[Close_t > SMA200_t]
            * 1[Return^{(3)}_t < 0]
            * 1[IBS_t < 0.1]

    Exit rules:

        RSI2_{i,t} = RSI(C_{i,t}, length = 2)

        exit_{i,t} = 1[IBS_{i,t} > 0.90] or 1[RSI2_{i,t} > 90]

    Names are ranked by:
    1. Turnover descending
    2. symbol ascending
    """

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
        sma_window_int: int = 200,
        qpi_window_int: int = 3,
        qpi_lookback_years_int: int = 5,
        return_lookback_days_int: int = 3,
        max_entry_ibs_float: float = 0.1,
        exit_ibs_threshold_float: float = 0.90,
        rsi_window_int: int = 2,
        exit_rsi2_threshold_float: float = 90.0,
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
        if sma_window_int <= 0:
            raise ValueError("sma_window_int must be positive.")
        if qpi_window_int <= 0:
            raise ValueError("qpi_window_int must be positive.")
        if qpi_lookback_years_int <= 0:
            raise ValueError("qpi_lookback_years_int must be positive.")
        if return_lookback_days_int <= 0:
            raise ValueError("return_lookback_days_int must be positive.")
        if not 0.0 <= max_entry_ibs_float <= 1.0:
            raise ValueError("max_entry_ibs_float must lie in [0, 1].")
        if not 0.0 <= exit_ibs_threshold_float <= 1.0:
            raise ValueError("exit_ibs_threshold_float must lie in [0, 1].")
        if rsi_window_int <= 0:
            raise ValueError("rsi_window_int must be positive.")
        if not 0.0 <= exit_rsi2_threshold_float <= 100.0:
            raise ValueError("exit_rsi2_threshold_float must lie in [0, 100].")

        self.trade_id_int = 0
        self.current_trade_map: defaultdict[str, int] = defaultdict(default_trade_id_int)
        self.universe_df: pd.DataFrame | None = None

        self.max_positions_int = max_positions_int
        self.qpi_threshold_float = qpi_threshold_float
        self.sma_window_int = sma_window_int
        self.qpi_window_int = qpi_window_int
        self.qpi_lookback_years_int = qpi_lookback_years_int
        self.return_lookback_days_int = return_lookback_days_int
        self.max_entry_ibs_float = max_entry_ibs_float
        self.exit_ibs_threshold_float = exit_ibs_threshold_float
        self.rsi_window_int = rsi_window_int
        self.exit_rsi2_threshold_float = exit_rsi2_threshold_float

    def compute_signals(self, pricing_data_df: pd.DataFrame) -> pd.DataFrame:
        signal_data_df = pricing_data_df.copy()
        symbol_list = signal_data_df.columns.get_level_values(0).unique()
        tradeable_symbol_list = [
            str(symbol_str)
            for symbol_str in symbol_list
            if not str(symbol_str).startswith("$")
            and (symbol_str, "Close") in signal_data_df.columns
            and (symbol_str, "High") in signal_data_df.columns
            and (symbol_str, "Low") in signal_data_df.columns
            and (symbol_str, "Turnover") in signal_data_df.columns
        ]

        if len(tradeable_symbol_list) == 0:
            return signal_data_df

        close_df = pd.DataFrame(
            {symbol_str: signal_data_df[(symbol_str, "Close")] for symbol_str in tradeable_symbol_list},
            index=signal_data_df.index,
        )
        high_df = pd.DataFrame(
            {symbol_str: signal_data_df[(symbol_str, "High")] for symbol_str in tradeable_symbol_list},
            index=signal_data_df.index,
        )
        low_df = pd.DataFrame(
            {symbol_str: signal_data_df[(symbol_str, "Low")] for symbol_str in tradeable_symbol_list},
            index=signal_data_df.index,
        )

        # *** CRITICAL*** The 3-day return must use only trailing closes.
        three_day_return_df = close_df.pct_change(
            periods=self.return_lookback_days_int,
            fill_method=None,
        )
        # *** CRITICAL*** SMA200 must remain a trailing average.
        sma_200_price_df = close_df.rolling(
            window=self.sma_window_int,
            min_periods=self.sma_window_int,
        ).mean()
        ibs_value_df = ibs_indicator(close_df, high_df, low_df)

        qpi_value_map: dict[str, pd.Series] = {}
        rsi2_value_map: dict[str, pd.Series] = {}
        for symbol_str in tradeable_symbol_list:
            close_ser = close_df[symbol_str].astype(float)

            # *** CRITICAL*** QPI must be computed from trailing close history only.
            qpi_value_map[symbol_str] = qp_indicator(
                close_ser,
                window_int=self.qpi_window_int,
                lookback_years_int=self.qpi_lookback_years_int,
            )

            # *** CRITICAL*** RSI2 must be computed from trailing close history
            # only. The library call is causal as long as the input series is.
            rsi2_value_map[symbol_str] = pd.Series(
                talib.RSI(close_ser.to_numpy(dtype=float), timeperiod=self.rsi_window_int),
                index=close_ser.index,
            )

        qpi_value_df = pd.DataFrame(qpi_value_map, index=signal_data_df.index)
        rsi2_value_df = pd.DataFrame(rsi2_value_map, index=signal_data_df.index)

        feature_frame_list: list[pd.DataFrame] = []
        feature_map = {
            "three_day_return_ser": three_day_return_df,
            "qpi_value_ser": qpi_value_df,
            "sma_200_price_ser": sma_200_price_df,
            "ibs_value_ser": ibs_value_df,
            "rsi2_value_ser": rsi2_value_df,
        }

        for field_str, field_df in feature_map.items():
            feature_df = field_df.copy()
            feature_df.columns = pd.MultiIndex.from_tuples(
                [(symbol_str, field_str) for symbol_str in feature_df.columns]
            )
            feature_frame_list.append(feature_df)

        return pd.concat([signal_data_df] + feature_frame_list, axis=1)

    def iterate(self, data_df: pd.DataFrame, close_row_ser: pd.Series, open_price_ser: pd.Series):
        if close_row_ser is None or data_df is None:
            return

        position_ser = self.get_positions()
        long_position_ser = position_ser[position_ser > 0]
        long_slots_int = self.max_positions_int - len(long_position_ser)

        for symbol_str in long_position_ser.index:
            ibs_value_float = close_row_ser.get((symbol_str, "ibs_value_ser"), np.nan)
            rsi2_value_float = close_row_ser.get((symbol_str, "rsi2_value_ser"), np.nan)

            exit_for_ibs_bool = (
                pd.notna(ibs_value_float)
                and float(ibs_value_float) > self.exit_ibs_threshold_float
            )
            exit_for_rsi2_bool = (
                pd.notna(rsi2_value_float)
                and float(rsi2_value_float) > self.exit_rsi2_threshold_float
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
            "Turnover",
            "qpi_value_ser",
            "sma_200_price_ser",
            "three_day_return_ser",
            "ibs_value_ser",
        ]
        missing_field_list = [
            field_str for field_str in required_field_list if field_str not in candidate_df.columns
        ]
        if len(missing_field_list) > 0:
            return []

        candidate_df = candidate_df.dropna(subset=required_field_list)
        candidate_df = candidate_df[
            candidate_df["qpi_value_ser"].astype(float) < self.qpi_threshold_float
        ]
        candidate_df = candidate_df[
            candidate_df["Close"].astype(float) > candidate_df["sma_200_price_ser"].astype(float)
        ]
        candidate_df = candidate_df[
            candidate_df["three_day_return_ser"].astype(float) < 0.0
        ]
        candidate_df = candidate_df[
            candidate_df["ibs_value_ser"].astype(float) < self.max_entry_ibs_float
        ]

        if self.universe_df is not None and self.previous_bar in self.universe_df.index:
            universe_membership_ser = self.universe_df.loc[self.previous_bar]
            universe_symbol_list = universe_membership_ser[universe_membership_ser == 1].index.tolist()
            candidate_df = candidate_df[candidate_df.index.isin(universe_symbol_list)]

        candidate_df = candidate_df.assign(symbol_str=candidate_df.index.astype(str))
        candidate_df = candidate_df.sort_values(
            by=["Turnover", "symbol_str"],
            ascending=[False, True],
            kind="mergesort",
        )
        return candidate_df.index.tolist()


if __name__ == "__main__":
    benchmark_list = ["$SPX"]
    symbol_list, universe_df = build_index_constituent_matrix(indexname="S&P 500")
    pricing_data_df = get_prices(
        symbol_list,
        benchmark_list,
        start_date_str="1998-01-01",
        end_date_str=None,
    )

    strategy = QPIIbsRsiExitStrategy(
        name="strategy_mr_qpi_ibs_rsi_exit",
        benchmarks=benchmark_list,
        capital_base=100_000,
        slippage=0.00025,
        commission_per_share=0.005,
        commission_minimum=1.0,
    )
    strategy.universe_df = universe_df

    calendar_idx = pricing_data_df.index[pricing_data_df.index.year >= 2004]
    run_daily(strategy, pricing_data_df, calendar_idx)

    strategy.universe_df = None

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)
    display(strategy.summary)
    display(strategy.summary_trades)
    save_results(strategy)
