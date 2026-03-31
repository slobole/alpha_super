"""
Alpha #23-inspired breakout momentum strategy for point-in-time S&P 500 members.

Core formulas
-------------
For stock i on decision date t:

    breakout_gate_{i,t}
        = 1[H_{i,t} > (1 / L) * sum_{k=1}^{L} H_{i,t-k}]

    high_return^{(2)}_{i,t}
        = H_{i,t} / H_{i,t-2} - 1

    alpha23_score_{i,t}
        = breakout_gate_{i,t} * (-high_return^{(2)}_{i,t})

This keeps the Alpha #23 intuition:

    breakout filter + anti-chasing penalty

while avoiding two issues in the raw literal expression:
1. the self-including average-high quirk
2. cross-sectional dollar-price bias from raw high deltas

Execution philosophy
--------------------
This is an approximate equal-slot strategy with next-open execution:

    capital_per_trade_t = V_{t-1} / N_max

    q^{intent}_{i,t} ~= floor(capital_per_trade_t / P^{fill}_{i,t})

The decision is made using information through previous_bar = t and orders
fill at the next bar open t+1 under the engine contract.
"""

from __future__ import annotations

from collections import defaultdict
from typing import List

import numpy as np
import pandas as pd
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


class Alpha23BreakoutStrategy(Strategy):
    """
    Long-only Alpha #23 breakout pod for controlled continuation entries.

    Decision-date signal:

        breakout_gate_{i,t}
            = 1[H_{i,t} > mean(H_{i,t-1}, ..., H_{i,t-L})]

        high_return^{(2)}_{i,t}
            = H_{i,t} / H_{i,t-2} - 1

        alpha23_score_{i,t}
            = breakout_gate_{i,t} * (-high_return^{(2)}_{i,t})

    Exit rule:

        exit_{i,t} = 1[holding_days_{i,t} >= max_holding_days]

    Entry rules:

        eligible_{i,t}
            = 1[PIT member_t]
            * 1[breakout_gate_{i,t} = 1]
            * 1[alpha23_score_{i,t} is finite]

    Names are ranked by:
    1. alpha23_score descending
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
        breakout_window_int: int = 20,
        score_lookback_days_int: int = 2,
        max_holding_days_int: int = 5,
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
        if breakout_window_int <= 0:
            raise ValueError("breakout_window_int must be positive.")
        if score_lookback_days_int <= 0:
            raise ValueError("score_lookback_days_int must be positive.")
        if max_holding_days_int <= 0:
            raise ValueError("max_holding_days_int must be positive.")

        self.trade_id_int = 0
        self.current_trade_map: defaultdict[str, int] = defaultdict(default_trade_id_int)
        self.universe_df: pd.DataFrame | None = None

        self.max_positions_int = max_positions_int
        self.breakout_window_int = breakout_window_int
        self.score_lookback_days_int = score_lookback_days_int
        self.max_holding_days_int = max_holding_days_int

    def compute_signals(self, pricing_data: pd.DataFrame) -> pd.DataFrame:
        signal_data = pricing_data.copy()
        symbol_list = signal_data.columns.get_level_values(0).unique()
        tradeable_symbol_list = [
            str(symbol_str)
            for symbol_str in symbol_list
            if not str(symbol_str).startswith("$")
            and (symbol_str, "Open") in signal_data.columns
            and (symbol_str, "High") in signal_data.columns
            and (symbol_str, "Low") in signal_data.columns
            and (symbol_str, "Close") in signal_data.columns
        ]

        if len(tradeable_symbol_list) == 0:
            return signal_data

        high_price_df = pd.DataFrame(
            {symbol_str: signal_data[(symbol_str, "High")] for symbol_str in tradeable_symbol_list},
            index=signal_data.index,
        )

        # *** CRITICAL*** The breakout average must use only prior highs.
        prior_high_price_df = high_price_df.shift(1)
        # *** CRITICAL*** rolling(window) must remain a trailing average on prior highs only.
        prior_avg_high_df = prior_high_price_df.rolling(
            window=self.breakout_window_int,
            min_periods=self.breakout_window_int,
        ).mean()

        # *** CRITICAL*** The anti-chasing term must use only lagged highs.
        lag_high_price_df = high_price_df.shift(self.score_lookback_days_int)
        high_return_2_day_df = (high_price_df / lag_high_price_df) - 1.0
        alpha23_breakout_bool_df = high_price_df.gt(prior_avg_high_df).fillna(False)
        anti_chasing_score_df = -high_return_2_day_df
        alpha23_score_df = anti_chasing_score_df.where(alpha23_breakout_bool_df, 0.0)
        alpha23_score_df = alpha23_score_df.replace([np.inf, -np.inf], np.nan)

        feature_frame_list: list[pd.DataFrame] = []
        feature_map = {
            "prior_avg_high_ser": prior_avg_high_df,
            "high_return_2_day_ser": high_return_2_day_df,
            "alpha23_breakout_bool": alpha23_breakout_bool_df,
            "alpha23_score_ser": alpha23_score_df,
        }

        for field_str, field_df in feature_map.items():
            feature_df = field_df.copy()
            feature_df.columns = pd.MultiIndex.from_tuples(
                [(symbol_str, field_str) for symbol_str in feature_df.columns]
            )
            feature_frame_list.append(feature_df)

        return pd.concat([signal_data] + feature_frame_list, axis=1)

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

            # *** CRITICAL*** Holding days must be counted through previous_bar
            # only. Including current_bar would leak the execution bar.
            entry_bar_loc_int = int(data.index.get_loc(entry_bar_ts))
            previous_bar_loc_int = int(data.index.get_loc(self.previous_bar))
            holding_days_int = previous_bar_loc_int - entry_bar_loc_int + 1
            if holding_days_int >= self.max_holding_days_int:
                self.order_target_value(symbol_str, 0.0, trade_id=self.current_trade_map[symbol_str])
                long_slots_int += 1

        capital_per_trade_float = self.previous_total_value / float(self.max_positions_int)
        opportunity_symbol_list = self.get_opportunity_list(close)

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
            "alpha23_breakout_bool",
            "alpha23_score_ser",
        ]
        missing_field_list = [
            field_str for field_str in required_field_list if field_str not in candidate_df.columns
        ]
        if len(missing_field_list) > 0:
            return []

        candidate_df = candidate_df.dropna(subset=required_field_list)
        candidate_df = candidate_df[candidate_df["alpha23_breakout_bool"].astype(bool)]

        if self.universe_df is not None and self.previous_bar in self.universe_df.index:
            universe_membership_ser = self.universe_df.loc[self.previous_bar]
            universe_symbol_list = universe_membership_ser[universe_membership_ser == 1].index.tolist()
            candidate_df = candidate_df[candidate_df.index.isin(universe_symbol_list)]

        candidate_df = candidate_df.assign(symbol_str=candidate_df.index.astype(str))
        candidate_df = candidate_df.sort_values(
            by=["alpha23_score_ser", "symbol_str"],
            ascending=[False, True],
            kind="mergesort",
        )
        return candidate_df.index.tolist()


if __name__ == "__main__":
    benchmark_list = ["$SPX"]
    symbol_list, universe_df = build_index_constituent_matrix(indexname="S&P 500")
    pricing_data = get_prices(
        symbol_list,
        benchmark_list,
        start_date_str="1998-01-01",
        end_date_str=None,
    )

    strategy = Alpha23BreakoutStrategy(
        name="strategy_mo_alpha23_breakout",
        benchmarks=benchmark_list,
        capital_base=100_000,
        slippage=0.0001,
        commission_per_share=0.005,
        commission_minimum=1.0,
    )
    strategy.universe_df = universe_df

    calendar_idx = pricing_data.index[pricing_data.index.year >= 2004]
    run_daily(strategy, pricing_data, calendar_idx)

    strategy.universe_df = None

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)
    display(strategy.summary)
    display(strategy.summary_trades)
    save_results(strategy)
