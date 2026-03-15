"""
DV2 long-only mean-reversion strategy on the Russell 3000 point-in-time universe.

Core formulas
-------------
For stock i on decision date t:

    r^{(126)}_{i,t} = C_{i,t} / C_{i,t-126} - 1

    NATR_{i,t} = NATR(H_{i,t}, L_{i,t}, C_{i,t}; 14)

    DV2_{i,t} = dv2_indicator(C_{i,t}, H_{i,t}, L_{i,t}; 126)

    SMA200_{i,t} = (1 / 200) * sum_{k=0}^{199} C_{i,t-k}

    eligible_{i,t}
        = 1[PIT Russell3000 member_t]
        * 1[DV2_{i,t} < 10]
        * 1[C_{i,t} > SMA200_{i,t}]
        * 1[r^{(126)}_{i,t} > 0]

Exit rule
---------

    exit_{i,t} = 1[C_{i,t} > H_{i,t-1}]

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

import pandas as pd
import talib
from IPython.display import display

from alpha.engine.backtest import run_daily
from alpha.engine.report import save_results
from alpha.engine.strategy import Strategy
from alpha.indicators import dv2_indicator
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


class DV2R3000Strategy(Strategy):
    """
    DV2 stock pod on Russell 3000 constituents with the same rules as strategy_mr_dv2.

    Entry rules:

        eligible_{i,t}
            = 1[PIT Russell3000 member_t]
            * 1[DV2_t < 10]
            * 1[Close_t > SMA200_t]
            * 1[Return^{(126)}_t > 0]

    Exit rule:

        exit_{i,t} = 1[Close_t > High_{t-1}]

    Names are ranked by NATR descending.
    """

    enable_signal_audit = False
    signal_audit_sample_size = 10

    def __init__(
        self,
        name: str,
        benchmarks: list[str] | tuple[str, ...],
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

        self.max_positions_int = 10
        self.trade_id_int = 0
        self.current_trade_map: defaultdict[str, int] = defaultdict(default_trade_id_int)
        self.universe_df: pd.DataFrame | None = None

    def compute_signals(self, pricing_data: pd.DataFrame) -> pd.DataFrame:
        signal_df = pricing_data.copy()
        symbol_idx = signal_df.columns.get_level_values(0).unique()
        feature_series_map: dict[tuple[str, str], pd.Series] = {}

        for symbol_str in symbol_idx:
            if str(symbol_str).startswith("$"):
                continue
            if (symbol_str, "Close") not in signal_df.columns:
                continue
            if (symbol_str, "High") not in signal_df.columns:
                continue
            if (symbol_str, "Low") not in signal_df.columns:
                continue

            close_ser = signal_df[(symbol_str, "Close")].astype(float)
            high_ser = signal_df[(symbol_str, "High")].astype(float)
            low_ser = signal_df[(symbol_str, "Low")].astype(float)

            # *** CRITICAL*** The 126-day return must use only trailing close history.
            prior_close_126_ser = close_ser.shift(126)
            feature_series_map[(symbol_str, "return_126_day_ser")] = (
                close_ser / prior_close_126_ser
            ) - 1.0
            # *** CRITICAL*** NATR must be computed from trailing OHLC history only.
            feature_series_map[(symbol_str, "natr_value_ser")] = pd.Series(
                talib.NATR(
                    high_ser.to_numpy(dtype=float),
                    low_ser.to_numpy(dtype=float),
                    close_ser.to_numpy(dtype=float),
                    timeperiod=14,
                ),
                index=close_ser.index,
            )
            # *** CRITICAL*** DV2 must be computed from trailing OHLC history only.
            feature_series_map[(symbol_str, "dv2_value_ser")] = dv2_indicator(
                close_ser,
                high_ser,
                low_ser,
                length_int=126,
            )
            # *** CRITICAL*** SMA200 must remain a trailing average.
            feature_series_map[(symbol_str, "sma_200_price_ser")] = close_ser.rolling(
                window=200,
            ).mean()

        if not feature_series_map:
            return signal_df

        feature_df = pd.DataFrame(feature_series_map, index=signal_df.index)
        return pd.concat([signal_df, feature_df], axis=1).copy()

    def iterate(self, data: pd.DataFrame, close: pd.Series, open_prices: pd.Series):
        position_ser = self.get_positions()
        long_position_ser = position_ser[position_ser > 0]
        long_slots_int = self.max_positions_int - len(long_position_ser)

        for symbol_str in long_position_ser.index:
            close_price_float = close[(symbol_str, "Close")]
            # *** CRITICAL*** The exit rule must compare to yesterday's high only.
            prior_high_float = data[(symbol_str, "High")].iloc[-2]
            if close_price_float > prior_high_float:
                self.order_target_value(
                    symbol_str,
                    0,
                    trade_id=self.current_trade_map[symbol_str],
                )
                long_slots_int += 1

        capital_per_trade_float = self.previous_total_value / self.max_positions_int
        opportunity_symbol_list = self.get_opportunity_list(close)

        while long_slots_int > 0 and len(opportunity_symbol_list) > 0:
            symbol_str = opportunity_symbol_list.pop(0)

            if self.get_position(symbol_str) != 0:
                continue

            self.trade_id_int += 1
            self.current_trade_map[symbol_str] = self.trade_id_int
            self.order_value(
                symbol_str,
                capital_per_trade_float,
                trade_id=self.trade_id_int,
            )
            long_slots_int -= 1

    def get_opportunity_list(self, close_row_ser: pd.Series) -> list[str]:
        candidate_df = close_row_ser.unstack().dropna()
        candidate_df = candidate_df[~candidate_df.index.astype(str).str.startswith("$")]

        candidate_df = candidate_df[
            (candidate_df["dv2_value_ser"] < 10)
            & (candidate_df["Close"] > candidate_df["sma_200_price_ser"])
            & (candidate_df["return_126_day_ser"] > 0)
        ].sort_values("natr_value_ser", ascending=False)

        # *** CRITICAL*** Universe membership must be point-in-time at previous_bar.
        universe_membership_ser = self.universe_df.loc[self.previous_bar]
        universe_symbol_list = universe_membership_ser[universe_membership_ser == 1].index.tolist()
        return candidate_df[candidate_df.index.isin(universe_symbol_list)].index.tolist()


if __name__ == "__main__":
    benchmark_list = ["$SPX"]
    symbol_list, universe_df = build_index_constituent_matrix(indexname="Russell 3000")
    pricing_data = get_prices(
        symbol_list,
        benchmark_list,
        start_date_str="1998-01-01",
        end_date_str=None,
    )

    strategy = DV2R3000Strategy(
        name="strategy_mr_dv2_r3000",
        benchmarks=benchmark_list,
        capital_base=100_000,
        slippage=0.0001,
        commission_per_share=0.005,
        commission_minimum=1.0,
    )
    strategy.universe_df = universe_df

    calendar_idx = pricing_data.index
    calendar_idx = calendar_idx[calendar_idx.year >= 2004]

    run_daily(strategy, pricing_data, calendar_idx)

    strategy.universe_df = None

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)
    display(strategy.summary)
    display(strategy.summary_trades)
    save_results(strategy)
