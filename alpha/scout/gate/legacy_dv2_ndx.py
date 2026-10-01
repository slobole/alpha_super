"""Engine reference for the DV2 Nasdaq-100 variant (identity gate `dv2_ndx`), restored from git.

*** CRITICAL*** `strategies/dv2/strategy_mr_dv2_nasdaq100.py` has been an empty stub since bf1a334 (it star-imports
itself, so it defines nothing and has no `run_variant`). The class below is its last committed implementation
(300ca70), copied verbatim except for comments and lint formatting, run on the real engine (`alpha.engine.Strategy`,
`run_daily`). Scout uses it only as the gate's engine side; it is not a strategy of record. When the strategies/
module is restored, point `GATED_SPEC_DICT["dv2_ndx"]` back at it and delete this file.

Rule (300ca70): DV2(126) < 20, Close > SMA200, 126-session return > 25%, ranked by NATR(14) descending, members of the
universe row dated exactly T (`universe_df.loc[previous_bar]`), 10 slots of V_T / 10, exit when Close_T > High_(T-1);
fills at the next open with 1 bp slippage, $0.005 a share, $1 minimum; calendar from 2004.
"""

from __future__ import annotations

from collections import defaultdict

import pandas as pd
import talib

from alpha.engine.backtest import run_daily
from alpha.engine.strategy import Strategy
from alpha.indicators import dv2_indicator


def _default_trade_id_int() -> int:
    return -1


class LegacyDvo2NasdaqStrategy(Strategy):
    max_positions = 10
    trade_id = 0
    current_trade = defaultdict(_default_trade_id_int)  # noqa: RUF012 - verbatim class attribute (reset in run_variant)
    universe_df = None

    def compute_signals(self, pricing_data: pd.DataFrame) -> pd.DataFrame:
        signal_data = pricing_data.copy()
        symbols = signal_data.columns.get_level_values(0).unique()
        feature_cols = {}
        for symbol in symbols:
            if str(symbol).startswith('$') or (symbol, 'Close') not in signal_data.columns:
                continue
            close = signal_data[(symbol, 'Close')]
            high = signal_data[(symbol, 'High')]
            low = signal_data[(symbol, 'Low')]
            feature_cols[(symbol, 'p126d_return')] = close / close.shift(126) - 1
            feature_cols[(symbol, 'natr')] = talib.NATR(high, low, close, 14)
            feature_cols[(symbol, 'dv2')] = dv2_indicator(close, high, low, length_int=126)
            feature_cols[(symbol, 'sma_200')] = close.rolling(200).mean()
        if not feature_cols:
            return signal_data
        features = pd.DataFrame(feature_cols, index=signal_data.index)
        return pd.concat([signal_data, features], axis=1).copy()

    def iterate(self, data: pd.DataFrame, close: pd.DataFrame, open_prices: pd.Series):
        positions = self.get_positions()
        long_positions = positions[positions > 0]
        long_slots = self.max_positions - len(long_positions)
        for symbol in long_positions.index:  # exit: sell if price > yesterday's high
            c = close[(symbol, 'Close')]
            yh = data[(symbol, 'High')].iloc[-2]
            if c > yh:
                self.order_target_value(symbol, 0, trade_id=self.current_trade[symbol])
                long_slots += 1
        capital_to_allocate_per_trade = self.previous_total_value / self.max_positions
        long_opportunities = self.get_opportunities(close)
        while long_slots > 0 and len(long_opportunities) > 0:
            symbol = long_opportunities.pop(0)
            if self.get_position(symbol) != 0:
                continue
            self.trade_id += 1
            self.current_trade[symbol] = self.trade_id
            self.order_value(symbol, capital_to_allocate_per_trade, trade_id=self.trade_id)
            long_slots -= 1

    def get_opportunities(self, close) -> list:
        df = close.unstack().dropna()
        df = df[~df.index.astype(str).str.startswith('$')]
        df = df[
            (df['dv2'] < 20) &
            (df['Close'] > df['sma_200']) &
            (df['p126d_return'] > 0.25)
        ].sort_values('natr', ascending=False)
        u = self.universe_df.loc[self.previous_bar]
        u = u[u == 1].index.tolist()
        return df[df.index.isin(u)].index.tolist()


def run_variant(show_display_bool: bool = True, save_results_bool: bool = False, end_date_str: str | None = None) -> Strategy:
    """The 300ca70 `__main__` block as a function (results are never saved: this is a gate reference)."""
    from data.norgate_loader import build_index_constituent_matrix, load_raw_prices

    benchmarks = ['$SPX']
    index_symbols, universe_df = build_index_constituent_matrix(indexname='Nasdaq 100')
    pricing_data = load_raw_prices(index_symbols, benchmarks, start_date='1998-01-01', end_date=end_date_str)
    strategy = LegacyDvo2NasdaqStrategy(name='strategy_mr_dv2_nasdaq100', benchmarks=benchmarks, capital_base=100_000, slippage=0.0001,
                                        commission_per_share=0.005, commission_minimum=1.0)
    strategy.universe_df = universe_df
    strategy.trade_id = 0
    strategy.current_trade = defaultdict(_default_trade_id_int)
    calendar = pricing_data.index
    calendar = calendar[calendar.year >= 2004]
    run_daily(strategy, pricing_data, calendar, show_progress=show_display_bool, show_signal_progress_bool=show_display_bool)
    strategy.universe_df = None
    return strategy
