"""
TL;DR: this is the original `DVO2Strategy` plus one extra entry filter:

    Close_t < 100

Strategy rule:

    entry_i,t =
        1[DV2_i,t < 10]
        * 1[Close_i,t > SMA200_i,t]
        * 1[Return126_i,t > 0.05]
        * 1[Close_i,t < 100]

    exit_i,t = 1[Close_i,t > High_i,t-1]

Sizing rule:

    capital_per_trade_t = PortfolioValue_t-1 / max_positions

This file intentionally preserves the original DV2 semantics and only adds the
price-cap entry filter.
"""

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


def get_prices(
    symbols: List[str],
    benchmarks: List[str],
    start_date: str = "1998-01-01",
    end_date: str | None = None,
) -> pd.DataFrame:
    return load_raw_prices(symbols, benchmarks, start_date, end_date)


class DVO2PriceLt100Strategy(Strategy):
    max_positions = 10
    max_price_float = 100.0
    trade_id = 0
    current_trade = defaultdict(lambda: -1)
    universe_df = None

    def compute_signals(self, pricing_data: pd.DataFrame) -> pd.DataFrame:
        signal_df = pricing_data.copy()
        symbol_list = signal_df.columns.get_level_values(0).unique()
        feature_map_dict: dict[tuple[str, str], pd.Series] = {}

        for symbol_str in symbol_list:
            if str(symbol_str).startswith("$") or (symbol_str, "Close") not in signal_df.columns:
                continue

            close_ser = signal_df[(symbol_str, "Close")]
            high_ser = signal_df[(symbol_str, "High")]
            low_ser = signal_df[(symbol_str, "Low")]

            # *** CRITICAL*** This 126-day return must stay backward-looking only.
            feature_map_dict[(symbol_str, "p126d_return")] = close_ser / close_ser.shift(126) - 1
            feature_map_dict[(symbol_str, "natr")] = talib.NATR(high_ser, low_ser, close_ser, 14)
            feature_map_dict[(symbol_str, "dv2")] = dv2_indicator(close_ser, high_ser, low_ser, length_int=126)
            # *** CRITICAL*** SMA200 must remain a trailing window to avoid look-ahead bias.
            feature_map_dict[(symbol_str, "sma_200")] = close_ser.rolling(200).mean()

        if not feature_map_dict:
            return signal_df

        feature_df = pd.DataFrame(feature_map_dict, index=signal_df.index)
        return pd.concat([signal_df, feature_df], axis=1).copy()

    def iterate(self, data: pd.DataFrame, close: pd.DataFrame, open_prices: pd.Series):
        positions_ser = self.get_positions()
        long_positions_ser = positions_ser[positions_ser > 0]
        long_slots_int = self.max_positions - len(long_positions_ser)

        for symbol_str in long_positions_ser.index:
            close_float = close[(symbol_str, "Close")]
            # *** CRITICAL*** Yesterday high must reference only the prior completed bar.
            yesterday_high_float = data[(symbol_str, "High")].iloc[-2]
            if close_float > yesterday_high_float:
                self.order_target_value(symbol_str, 0, trade_id=self.current_trade[symbol_str])
                long_slots_int += 1

        capital_per_trade_float = self.previous_total_value / self.max_positions
        opportunity_list = self.get_opportunities(close)

        while long_slots_int > 0 and len(opportunity_list) > 0:
            symbol_str = opportunity_list.pop(0)

            if self.get_position(symbol_str) != 0:
                continue

            self.trade_id += 1
            self.current_trade[symbol_str] = self.trade_id
            self.order_value(symbol_str, capital_per_trade_float, trade_id=self.trade_id)
            long_slots_int -= 1

    def get_opportunities(self, close) -> list:
        candidate_df = close.unstack().dropna()
        candidate_df = candidate_df[~candidate_df.index.astype(str).str.startswith("$")]

        candidate_df = candidate_df[
            (candidate_df["dv2"] < 10)
            & (candidate_df["Close"] > candidate_df["sma_200"])
            & (candidate_df["p126d_return"] > 0.05)
            & (candidate_df["Close"] < self.max_price_float)
        ].sort_values("natr", ascending=False)

        universe_membership_ser = self.universe_df.loc[self.previous_bar]
        tradeable_symbol_list = universe_membership_ser[universe_membership_ser == 1].index.tolist()
        return candidate_df[candidate_df.index.isin(tradeable_symbol_list)].index.tolist()


if __name__ == "__main__":
    benchmark_list = ["$SPX"]
    index_symbols, universe_df = build_index_constituent_matrix(indexname="S&P 500")
    pricing_data = get_prices(index_symbols, benchmark_list, start_date="1998-01-01", end_date=None)

    strategy_obj = DVO2PriceLt100Strategy(
        name="strategy_mr_dv2_price_lt_100",
        benchmarks=benchmark_list,
        capital_base=100_000,
        slippage=0.00025,
        commission_per_share=0.005,
        commission_minimum=1.0,
    )
    strategy_obj.universe_df = universe_df
    calendar = pricing_data.index
    calendar = calendar[calendar.year >= 2004]

    run_daily(strategy_obj, pricing_data, calendar)

    strategy_obj.universe_df = None

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)
    display(strategy_obj.summary)
    display(strategy_obj.summary_trades)
    save_results(strategy_obj)
