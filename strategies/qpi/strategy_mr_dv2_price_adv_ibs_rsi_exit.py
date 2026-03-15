"""
DV2 long-only mean-reversion strategy with price / ADV filters and IBS / RSI2 exits.

Core formulas
-------------
For stock i on decision date t:

    r^{(126)}_{i,t} = C_{i,t} / C_{i,t-126} - 1

    SMA200_{i,t} = (1 / 200) * sum_{k=0}^{199} C_{i,t-k}

    ADV20_{i,t} = (1 / 20) * sum_{k=0}^{19} Turnover_{i,t-k}

    IBS_{i,t} = (C_{i,t} - L_{i,t}) / (H_{i,t} - L_{i,t})

    RSI2_{i,t} = RSI(C_{i,t}, length = 2)

    eligible_{i,t}
        = 1[PIT member_t]
        * 1[C_{i,t} > 10]
        * 1[ADV20_{i,t} >= 20,000,000]
        * 1[DV2_{i,t} < 10]
        * 1[C_{i,t} > SMA200_{i,t}]
        * 1[r^{(126)}_{i,t} > 0]

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
from alpha.indicators import (
    adv_dollar_indicator,
    dv2_indicator,
    ibs_indicator,
)
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


class DV2PriceAdvIbsRsiExitStrategy(Strategy):
    """
    DV2 stock pod with price / ADV20 entries and overbought IBS / RSI2 exits.

    Entry rules:

        eligible_{i,t}
            = 1[PIT member_t]
            * 1[Close_t > 10]
            * 1[ADV20_t >= 20e6]
            * 1[DV2_t < 10]
            * 1[Close_t > SMA200_t]
            * 1[Return^{(126)}_t > 0]

    Exit rules:

        IBS_{i,t} = (C_{i,t} - L_{i,t}) / (H_{i,t} - L_{i,t})

        RSI2_{i,t} = RSI(C_{i,t}, length = 2)

        exit_{i,t} = 1[IBS_{i,t} > 0.90] or 1[RSI2_{i,t} > 90]

    Names are ranked by:
    1. NATR descending
    2. symbol ascending
    """

    enable_signal_audit = True
    signal_audit_sample_size = 10

    def __init__(
        self,
        name: str,
        benchmarks: list[str] | tuple[str, ...],
        capital_base: float = 100_000,
        slippage: float = 0.0001,
        commission_per_share: float = 0.005,
        commission_minimum: float = 1.0,
        max_positions_int: int = 10,
        min_price_float: float = 10.0,
        min_adv20_dollar_float: float = 20_000_000.0,
        adv_window_int: int = 20,
        sma_window_int: int = 200,
        return_lookback_days_int: int = 126,
        dv2_threshold_float: float = 10.0,
        natr_window_int: int = 14,
        dv2_lookback_days_int: int = 126,
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
        if min_price_float <= 0.0:
            raise ValueError("min_price_float must be positive.")
        if min_adv20_dollar_float < 0.0:
            raise ValueError("min_adv20_dollar_float must be non-negative.")
        if adv_window_int <= 0:
            raise ValueError("adv_window_int must be positive.")
        if sma_window_int <= 0:
            raise ValueError("sma_window_int must be positive.")
        if return_lookback_days_int <= 0:
            raise ValueError("return_lookback_days_int must be positive.")
        if natr_window_int <= 0:
            raise ValueError("natr_window_int must be positive.")
        if dv2_lookback_days_int <= 0:
            raise ValueError("dv2_lookback_days_int must be positive.")
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
        self.min_price_float = min_price_float
        self.min_adv20_dollar_float = min_adv20_dollar_float
        self.adv_window_int = adv_window_int
        self.sma_window_int = sma_window_int
        self.return_lookback_days_int = return_lookback_days_int
        self.dv2_threshold_float = dv2_threshold_float
        self.natr_window_int = natr_window_int
        self.dv2_lookback_days_int = dv2_lookback_days_int
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

        close_price_df = pd.DataFrame(
            {symbol_str: signal_data_df[(symbol_str, "Close")] for symbol_str in tradeable_symbol_list},
            index=signal_data_df.index,
        )
        high_price_df = pd.DataFrame(
            {symbol_str: signal_data_df[(symbol_str, "High")] for symbol_str in tradeable_symbol_list},
            index=signal_data_df.index,
        )
        low_price_df = pd.DataFrame(
            {symbol_str: signal_data_df[(symbol_str, "Low")] for symbol_str in tradeable_symbol_list},
            index=signal_data_df.index,
        )
        turnover_dollar_df = pd.DataFrame(
            {symbol_str: signal_data_df[(symbol_str, "Turnover")] for symbol_str in tradeable_symbol_list},
            index=signal_data_df.index,
        )

        # *** CRITICAL*** The 126-day return must use only prior close history.
        prior_close_126_df = close_price_df.shift(self.return_lookback_days_int)
        return_126_day_df = (close_price_df / prior_close_126_df) - 1.0
        # *** CRITICAL*** SMA200 must remain a trailing average.
        sma_200_price_df = close_price_df.rolling(
            window=self.sma_window_int,
            min_periods=self.sma_window_int,
        ).mean()
        # *** CRITICAL*** ADV20 must remain a trailing average on past turnover only.
        adv20_dollar_df = adv_dollar_indicator(
            turnover_dollar_df,
            window_int=self.adv_window_int,
        )
        ibs_value_df = ibs_indicator(close_price_df, high_price_df, low_price_df)

        natr_value_map: dict[str, pd.Series] = {}
        dv2_value_map: dict[str, pd.Series] = {}
        rsi2_value_map: dict[str, pd.Series] = {}
        for symbol_str in tradeable_symbol_list:
            high_price_ser = high_price_df[symbol_str].astype(float)
            low_price_ser = low_price_df[symbol_str].astype(float)
            close_price_ser = close_price_df[symbol_str].astype(float)

            # *** CRITICAL*** NATR must be computed from trailing OHLC history only.
            natr_value_map[symbol_str] = pd.Series(
                talib.NATR(
                    high_price_ser.to_numpy(dtype=float),
                    low_price_ser.to_numpy(dtype=float),
                    close_price_ser.to_numpy(dtype=float),
                    timeperiod=self.natr_window_int,
                ),
                index=close_price_ser.index,
            )
            # *** CRITICAL*** DV2 must be computed from trailing OHLC history only.
            dv2_value_map[symbol_str] = dv2_indicator(
                close_price_ser,
                high_price_ser,
                low_price_ser,
                length_int=self.dv2_lookback_days_int,
            )
            # *** CRITICAL*** RSI2 must be computed from trailing close history
            # only. The library call is causal as long as the input series is.
            rsi2_value_map[symbol_str] = pd.Series(
                talib.RSI(close_price_ser.to_numpy(dtype=float), timeperiod=self.rsi_window_int),
                index=close_price_ser.index,
            )

        natr_value_df = pd.DataFrame(natr_value_map, index=signal_data_df.index)
        dv2_value_df = pd.DataFrame(dv2_value_map, index=signal_data_df.index)
        rsi2_value_df = pd.DataFrame(rsi2_value_map, index=signal_data_df.index)

        feature_frame_list: list[pd.DataFrame] = []
        feature_map = {
            "return_126_day_ser": return_126_day_df,
            "natr_value_ser": natr_value_df,
            "dv2_value_ser": dv2_value_df,
            "sma_200_price_ser": sma_200_price_df,
            "adv20_dollar_ser": adv20_dollar_df,
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

        position_qty_ser = self.get_positions()
        long_position_qty_ser = position_qty_ser[position_qty_ser > 0]
        long_slots_int = self.max_positions_int - len(long_position_qty_ser)

        for symbol_str in long_position_qty_ser.index:
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
        candidate_signal_df = close_row_ser.unstack()
        candidate_signal_df = candidate_signal_df[
            ~candidate_signal_df.index.astype(str).str.startswith("$")
        ]

        required_field_list = [
            "Close",
            "dv2_value_ser",
            "sma_200_price_ser",
            "return_126_day_ser",
            "natr_value_ser",
            "adv20_dollar_ser",
        ]
        missing_field_list = [
            field_str for field_str in required_field_list if field_str not in candidate_signal_df.columns
        ]
        if len(missing_field_list) > 0:
            return []

        candidate_signal_df = candidate_signal_df.dropna(subset=required_field_list)
        candidate_signal_df = candidate_signal_df[
            candidate_signal_df["Close"].astype(float) > self.min_price_float
        ]
        candidate_signal_df = candidate_signal_df[
            candidate_signal_df["adv20_dollar_ser"].astype(float) >= self.min_adv20_dollar_float
        ]
        candidate_signal_df = candidate_signal_df[
            candidate_signal_df["dv2_value_ser"].astype(float) < self.dv2_threshold_float
        ]
        candidate_signal_df = candidate_signal_df[
            candidate_signal_df["Close"].astype(float)
            > candidate_signal_df["sma_200_price_ser"].astype(float)
        ]
        candidate_signal_df = candidate_signal_df[
            candidate_signal_df["return_126_day_ser"].astype(float) > 0.0
        ]

        if self.universe_df is not None and self.previous_bar in self.universe_df.index:
            universe_membership_ser = self.universe_df.loc[self.previous_bar]
            universe_symbol_list = universe_membership_ser[universe_membership_ser == 1].index.tolist()
            candidate_signal_df = candidate_signal_df[
                candidate_signal_df.index.isin(universe_symbol_list)
            ]

        candidate_signal_df = candidate_signal_df.assign(
            symbol_str=candidate_signal_df.index.astype(str)
        )
        candidate_signal_df = candidate_signal_df.sort_values(
            by=["natr_value_ser", "symbol_str"],
            ascending=[False, True],
            kind="mergesort",
        )
        return candidate_signal_df.index.tolist()


if __name__ == "__main__":
    benchmark_list = ["$SPX"]
    symbol_list, universe_df = build_index_constituent_matrix(indexname="S&P 500")
    pricing_data_df = get_prices(
        symbol_list,
        benchmark_list,
        start_date_str="1998-01-01",
        end_date_str=None,
    )

    strategy = DV2PriceAdvIbsRsiExitStrategy(
        name="strategy_mr_dv2_price_adv_ibs_rsi_exit",
        benchmarks=benchmark_list,
        capital_base=100_000,
        slippage=0.0001,
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
