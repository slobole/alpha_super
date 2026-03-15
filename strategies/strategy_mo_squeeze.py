"""
Histogram-confirmed momentum strategy for point-in-time S&P 500 constituents.

Core formulas
-------------
LazyBear-style histogram:

    HH_t = rolling_max(H_t, 20)
    LL_t = rolling_min(L_t, 20)
    momentum_base_t = C_t - (HH_t + LL_t + 2 * SMA_20(C_t)) / 4
    h_t = LINEARREG_20(momentum_base_t)

Entry rule:

    benchmark_sma200_t = (1 / 200) * sum_{k=0}^{199} C^{SPX}_{t-k}
    benchmark_bullish_t = 1[C^{SPX}_t > benchmark_sma200_t]
    long_breakout_t = 1[h_{t-1} < 0 and h_t > 0 and C_t > O_t and benchmark_bullish_t = 1]

Exit rule:

    holding_days_{i,t} = count_trading_days(entry_bar_i, t - 1)
    exit_t = 1[holding_days_{i,t} >= 5]

Execution philosophy
--------------------
This is an approximate equal-slot strategy with next-open execution:

    capital_per_trade_t = V_{t-1} / N_max

    q^{intent}_{i,t} ~= floor(capital_per_trade_t / P^{fill}_{i,t})

Tradable equities are loaded with CAPITALSPECIAL to avoid synthetic
forward-looking dividend adjustments. Benchmark comparison and the
benchmark regime filter use TOTALRETURN.
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


class SqueezeMomentumBreakoutStrategy(Strategy):
    """
    Long-only momentum pod built on point-in-time S&P 500 membership.

    Entry:

        benchmark_bullish_t = 1[C^{SPX}_t > SMA_200(C^{SPX})_t]
        long_breakout_t = 1[h_{t-1} < 0 and h_t > 0 and C_t > O_t and benchmark_bullish_t = 1]

    Exit:

        holding_days_{i,t} = count_trading_days(entry_bar_i, t - 1)
        exit_t = 1[holding_days_{i,t} >= 5]

    Ranking:

        rank_t = Turnover_t

    This strategy follows the engine's causal contract:

        decision at t uses information through previous_bar
        execution happens at the next bar open
    """

    max_positions_int = 10
    max_holding_days_int = 5
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
    ):
        super().__init__(
            name=name,
            benchmarks=list(benchmarks),
            capital_base=capital_base,
            slippage=slippage,
            commission_per_share=commission_per_share,
            commission_minimum=commission_minimum,
        )
        self.trade_id_int = 0
        self.current_trade_map: defaultdict[str, int] = defaultdict(default_trade_id_int)
        self.universe_df: pd.DataFrame | None = None

    def compute_signals(self, pricing_data: pd.DataFrame) -> pd.DataFrame:
        signal_data = pricing_data.copy()
        symbol_list = signal_data.columns.get_level_values(0).unique()
        feature_map: dict[tuple[str, str], pd.Series] = {}
        primary_benchmark_symbol_str = str(self._benchmarks[0]) if len(self._benchmarks) > 0 else None
        benchmark_entry_gate_ser = pd.Series(False, index=signal_data.index, dtype=bool)

        if (
            primary_benchmark_symbol_str is not None
            and (primary_benchmark_symbol_str, "Close") in signal_data.columns
        ):
            benchmark_close_ser = signal_data[(primary_benchmark_symbol_str, "Close")]
            # *** CRITICAL*** The benchmark 200-day moving average must remain a
            # trailing window so the market regime filter stays causal.
            benchmark_sma_200_ser = benchmark_close_ser.rolling(window=200).mean()
            benchmark_bullish_ser = benchmark_close_ser.gt(benchmark_sma_200_ser)
            benchmark_entry_gate_ser = benchmark_bullish_ser.fillna(False)

            feature_map[(primary_benchmark_symbol_str, "benchmark_sma_200")] = benchmark_sma_200_ser
            feature_map[(primary_benchmark_symbol_str, "benchmark_bullish")] = benchmark_bullish_ser

        for symbol_str in symbol_list:
            if str(symbol_str).startswith("$") or (symbol_str, "Close") not in signal_data.columns:
                continue

            open_ser = signal_data[(symbol_str, "Open")]
            high_ser = signal_data[(symbol_str, "High")]
            low_ser = signal_data[(symbol_str, "Low")]
            close_ser = signal_data[(symbol_str, "Close")]

            green_candle_ser = close_ser > open_ser

            # *** CRITICAL*** Rolling extrema must remain purely trailing.
            rolling_high_ser = high_ser.rolling(window=20).max()
            # *** CRITICAL*** Rolling extrema must remain purely trailing.
            rolling_low_ser = low_ser.rolling(window=20).min()
            # *** CRITICAL*** The rolling mean inside the histogram must remain
            # a trailing average to avoid look-ahead bias.
            close_mean_ser = close_ser.rolling(window=20).mean()
            momentum_base_ser = close_ser - (rolling_high_ser + rolling_low_ser + 2.0 * close_mean_ser) / 4.0
            squeeze_momentum_ser = pd.Series(
                talib.LINEARREG(momentum_base_ser.to_numpy(dtype=float), timeperiod=20),
                index=momentum_base_ser.index,
            )
            # *** CRITICAL*** Entry must use the prior histogram state only,
            # so the zero-line cross stays causal and does not leak future bars.
            prev_squeeze_momentum_ser = squeeze_momentum_ser.shift(1)
            breakout_core_ser = (
                prev_squeeze_momentum_ser.lt(0.0)
                & squeeze_momentum_ser.gt(0.0)
                & green_candle_ser.eq(True)
            )
            long_breakout_ser = breakout_core_ser & benchmark_entry_gate_ser

            feature_map[(symbol_str, "long_breakout")] = long_breakout_ser
            feature_map[(symbol_str, "squeeze_momentum")] = squeeze_momentum_ser

        if len(feature_map) == 0:
            return signal_data

        feature_df = pd.DataFrame(feature_map, index=signal_data.index)
        return pd.concat([signal_data, feature_df], axis=1).copy()

    def iterate(self, data: pd.DataFrame, close: pd.Series, open_prices: pd.Series):
        if close is None or data is None:
            return

        position_ser = self.get_positions()
        long_position_ser = position_ser[position_ser > 0]
        long_slots_int = self.max_positions_int - len(long_position_ser)

        for symbol_str in long_position_ser.index:
            entry_transaction_ser = self.get_latest_transaction(symbol_str)
            entry_bar_ts = pd.Timestamp(entry_transaction_ser["bar"])
            if entry_bar_ts not in data.index or self.previous_bar not in data.index:
                continue

            # *** CRITICAL*** Count the holding period from the actual filled
            # entry bar through previous_bar only. Including current_bar here
            # would shorten the hold and leak the live execution bar.
            entry_bar_loc_int = int(data.index.get_loc(entry_bar_ts))
            previous_bar_loc_int = int(data.index.get_loc(self.previous_bar))
            holding_days_int = previous_bar_loc_int - entry_bar_loc_int + 1
            if holding_days_int >= self.max_holding_days_int:
                self.order_target_value(symbol_str, 0, trade_id=self.current_trade_map[symbol_str])
                long_slots_int += 1

        capital_per_trade_float = self.previous_total_value / float(self.max_positions_int)
        opportunity_list = self.get_opportunities(close)

        while long_slots_int > 0 and len(opportunity_list) > 0:
            symbol_str = opportunity_list.pop(0)
            if self.get_position(symbol_str) != 0:
                continue

            self.trade_id_int += 1
            self.current_trade_map[symbol_str] = self.trade_id_int
            self.order_value(symbol_str, capital_per_trade_float, trade_id=self.trade_id_int)
            long_slots_int -= 1

    def get_opportunities(self, close_ser: pd.Series) -> list[str]:
        candidate_df = close_ser.unstack()
        candidate_df = candidate_df[~candidate_df.index.astype(str).str.startswith("$")]

        required_field_list = ["long_breakout", "Turnover"]
        candidate_df = candidate_df.dropna(subset=required_field_list)
        candidate_df = candidate_df[candidate_df["long_breakout"].astype(bool)]

        if self.universe_df is not None and self.previous_bar in self.universe_df.index:
            universe_membership_ser = self.universe_df.loc[self.previous_bar]
            universe_symbol_list = universe_membership_ser[universe_membership_ser == 1].index.tolist()
            candidate_df = candidate_df[candidate_df.index.isin(universe_symbol_list)]

        candidate_df = candidate_df.sort_values("Turnover", ascending=False)
        return candidate_df.index.tolist()


if __name__ == "__main__":
    benchmark_list = ["$SPX"]
    symbol_list, universe_df = build_index_constituent_matrix(indexname="S&P 500")
    pricing_data = get_prices(symbol_list, benchmark_list, start_date_str="1998-01-01", end_date_str=None)

    strategy = SqueezeMomentumBreakoutStrategy(
        name="strategy_mo_squeeze",
        benchmarks=benchmark_list,
        capital_base=100_000,
        slippage=0.0001,
        commission_per_share=0.005,
        commission_minimum=1.0,
    )
    strategy.universe_df = universe_df

    calendar_idx = pricing_data.index[pricing_data.index.year >= 2018]
    run_daily(strategy, pricing_data, calendar_idx)

    strategy.universe_df = None

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)
    display(strategy.summary)
    display(strategy.summary_trades)
    save_results(strategy)
