"""
Alpha 19 long-only pullback strategy with IBS / RSI2 exit rules.

Core formulas
-------------
For stock i on decision date t:

    r_{i,t} = C_{i,t} / C_{i,t-1} - 1

    alpha19_{i,t}
        = -sign(C_{i,t} - C_{i,t-7}) * (1 + rank_t(sum_{k=0}^{249} r_{i,t-k}))

    ADV20_{i,t} = (1 / 20) * sum_{k=0}^{19} Turnover_{i,t-k}

    winner_rank_{i,t} = rank_t(sum_{k=0}^{249} r_{i,t-k})

    IBS_{i,t} = (C_{i,t} - L_{i,t}) / (H_{i,t} - L_{i,t})

    RSI2_{i,t} = RSI(C_{i,t}, length = 2)

Entry rules
-----------
Trade only if:
    - point-in-time S&P 500 member
    - Close_t > 10
    - ADV20_t >= 20,000,000
    - winner_rank_t >= 0.70
    - alpha19_t > 0
    - IBS_t < 0.20

Exit rules
----------

    exit_{i,t} = 1[IBS_{i,t} > 0.90] or 1[RSI2_{i,t} > 90]

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
import talib
from IPython.display import display

from alpha.engine.backtest import run_daily
from alpha.engine.report import save_results
from alpha.engine.strategy import Strategy
from alpha.indicators import adv_dollar_indicator, ibs_indicator
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


class Alpha19IbsRsiExitStrategy(Strategy):
    """
    Long-only Alpha 19 stock pod with short-term overbought exits.

    Decision-date signal:

        alpha19_{i,t}
            = -sign(C_{i,t} - C_{i,t-7}) * (1 + rank_t(sum_{k=0}^{249} r_{i,t-k}))

    Exit rules:

        IBS_{i,t} = (C_{i,t} - L_{i,t}) / (H_{i,t} - L_{i,t})

        RSI2_{i,t} = RSI(C_{i,t}, length = 2)

        exit_{i,t} = 1[IBS_{i,t} > 0.90] or 1[RSI2_{i,t} > 90]

    Entry rules:

        eligible_{i,t}
            = 1[PIT member_t]
            * 1[Close_t > 10]
            * 1[ADV20_t >= 20e6]
            * 1[winner_rank_t >= 0.70]
            * 1[alpha19_t > 0]
            * 1[IBS_t < 0.20]

    Names are ranked by:
    1. alpha19 descending
    2. ADV20 descending
    3. symbol ascending
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
        max_positions_int: int = 20,
        winner_rank_threshold_float: float = 0.70,
        min_price_float: float = 10.0,
        min_adv20_dollar_float: float = 20_000_000.0,
        alpha_delay_days_int: int = 7,
        long_return_window_int: int = 252,
        adv_window_int: int = 20,
        max_entry_ibs_float: float = 0.20,
        exit_ibs_threshold_float: float = 0.90,
        exit_rsi2_threshold_float: float = 90.0,
        rsi_window_int: int = 2,
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
        if not 0.0 <= winner_rank_threshold_float <= 1.0:
            raise ValueError("winner_rank_threshold_float must lie in [0, 1].")
        if min_price_float <= 0.0:
            raise ValueError("min_price_float must be positive.")
        if min_adv20_dollar_float < 0.0:
            raise ValueError("min_adv20_dollar_float must be non-negative.")
        if alpha_delay_days_int <= 0:
            raise ValueError("alpha_delay_days_int must be positive.")
        if long_return_window_int <= 0:
            raise ValueError("long_return_window_int must be positive.")
        if adv_window_int <= 0:
            raise ValueError("adv_window_int must be positive.")
        if not 0.0 <= max_entry_ibs_float <= 1.0:
            raise ValueError("max_entry_ibs_float must lie in [0, 1].")
        if not 0.0 <= exit_ibs_threshold_float <= 1.0:
            raise ValueError("exit_ibs_threshold_float must lie in [0, 1].")
        if not 0.0 <= exit_rsi2_threshold_float <= 100.0:
            raise ValueError("exit_rsi2_threshold_float must lie in [0, 100].")
        if rsi_window_int <= 0:
            raise ValueError("rsi_window_int must be positive.")

        self.trade_id_int = 0
        self.current_trade_map: defaultdict[str, int] = defaultdict(default_trade_id_int)
        self.universe_df: pd.DataFrame | None = None

        self.max_positions_int = max_positions_int
        self.winner_rank_threshold_float = winner_rank_threshold_float
        self.min_price_float = min_price_float
        self.min_adv20_dollar_float = min_adv20_dollar_float
        self.alpha_delay_days_int = alpha_delay_days_int
        self.long_return_window_int = long_return_window_int
        self.adv_window_int = adv_window_int
        self.max_entry_ibs_float = max_entry_ibs_float
        self.exit_ibs_threshold_float = exit_ibs_threshold_float
        self.exit_rsi2_threshold_float = exit_rsi2_threshold_float
        self.rsi_window_int = rsi_window_int

    def compute_signals(self, pricing_data: pd.DataFrame) -> pd.DataFrame:
        signal_data = pricing_data.copy()
        symbol_list = signal_data.columns.get_level_values(0).unique()
        tradeable_symbol_list = [
            str(symbol_str)
            for symbol_str in symbol_list
            if not str(symbol_str).startswith("$")
            and (symbol_str, "Close") in signal_data.columns
            and (symbol_str, "High") in signal_data.columns
            and (symbol_str, "Low") in signal_data.columns
            and (symbol_str, "Turnover") in signal_data.columns
        ]

        if len(tradeable_symbol_list) == 0:
            return signal_data

        close_df = pd.DataFrame(
            {symbol_str: signal_data[(symbol_str, "Close")] for symbol_str in tradeable_symbol_list},
            index=signal_data.index,
        )
        high_df = pd.DataFrame(
            {symbol_str: signal_data[(symbol_str, "High")] for symbol_str in tradeable_symbol_list},
            index=signal_data.index,
        )
        low_df = pd.DataFrame(
            {symbol_str: signal_data[(symbol_str, "Low")] for symbol_str in tradeable_symbol_list},
            index=signal_data.index,
        )
        turnover_df = pd.DataFrame(
            {symbol_str: signal_data[(symbol_str, "Turnover")] for symbol_str in tradeable_symbol_list},
            index=signal_data.index,
        )

        # *** CRITICAL*** Close-to-close returns must remain purely backward-looking.
        return_df = close_df.pct_change(fill_method=None)
        # *** CRITICAL*** The 7-bar delay must stay backward-looking so the
        # pullback sign uses only information available on decision date t.
        delayed_close_df = close_df.shift(self.alpha_delay_days_int)
        short_pullback_dir_df = -np.sign(close_df - delayed_close_df)

        # *** CRITICAL*** The trailing return sum must remain a trailing window.
        trailing_250_return_sum_df = return_df.rolling(
            window=self.long_return_window_int,
            min_periods=self.long_return_window_int,
        ).sum()
        trailing_return_signal_df = 1.0 + trailing_250_return_sum_df
        winner_rank_pct_df = trailing_return_signal_df.rank(axis=1, pct=True)

        # *** CRITICAL*** ADV20 must remain a trailing average on past turnover only.
        adv20_dollar_df = adv_dollar_indicator(turnover_df, window_int=self.adv_window_int)
        ibs_value_df = ibs_indicator(close_df, high_df, low_df)

        rsi2_value_map: dict[str, pd.Series] = {}
        for symbol_str in tradeable_symbol_list:
            close_ser = close_df[symbol_str].astype(float)
            # *** CRITICAL*** RSI2 must be computed from trailing close history
            # only. The library call is causal as long as the input series is.
            rsi2_value_map[symbol_str] = pd.Series(
                talib.RSI(close_ser.to_numpy(dtype=float), timeperiod=self.rsi_window_int),
                index=close_ser.index,
            )
        rsi2_value_df = pd.DataFrame(rsi2_value_map, index=signal_data.index)

        alpha19_signal_df = short_pullback_dir_df * (1.0 + winner_rank_pct_df)
        alpha19_signal_df = alpha19_signal_df.replace([np.inf, -np.inf], np.nan)

        feature_frame_list: list[pd.DataFrame] = []
        feature_map = {
            "return_ser": return_df,
            "alpha19_signal_ser": alpha19_signal_df,
            "trailing_250_return_sum_ser": trailing_250_return_sum_df,
            "winner_rank_pct_ser": winner_rank_pct_df,
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

        return pd.concat([signal_data] + feature_frame_list, axis=1)

    def iterate(self, data: pd.DataFrame, close: pd.Series, open_prices: pd.Series):
        if close is None or data is None:
            return

        position_ser = self.get_positions()
        long_position_ser = position_ser[position_ser > 0]
        long_slots_int = self.max_positions_int - len(long_position_ser)

        for symbol_str in long_position_ser.index:
            ibs_value_float = close.get((symbol_str, "ibs_value_ser"), np.nan)
            rsi2_value_float = close.get((symbol_str, "rsi2_value_ser"), np.nan)

            exit_for_ibs_bool = (
                pd.notna(ibs_value_float)
                and float(ibs_value_float) > self.exit_ibs_threshold_float
            )
            exit_for_rsi2_bool = (
                pd.notna(rsi2_value_float)
                and float(rsi2_value_float) > self.exit_rsi2_threshold_float
            )

            if exit_for_ibs_bool or exit_for_rsi2_bool:
                self.order_target_value(symbol_str, 0, trade_id=self.current_trade_map[symbol_str])
                long_slots_int += 1

        capital_per_trade_float = self.previous_total_value / float(self.max_positions_int)
        opportunity_list = self.get_opportunity_list(close)

        while long_slots_int > 0 and len(opportunity_list) > 0:
            symbol_str = opportunity_list.pop(0)

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
            "alpha19_signal_ser",
            "winner_rank_pct_ser",
            "adv20_dollar_ser",
            "ibs_value_ser",
        ]

        missing_field_list = [field_str for field_str in required_field_list if field_str not in candidate_df.columns]
        if len(missing_field_list) > 0:
            return []

        candidate_df = candidate_df.dropna(subset=required_field_list)
        candidate_df = candidate_df[candidate_df["Close"].astype(float) > self.min_price_float]
        candidate_df = candidate_df[
            candidate_df["adv20_dollar_ser"].astype(float) >= self.min_adv20_dollar_float
        ]
        candidate_df = candidate_df[
            candidate_df["winner_rank_pct_ser"].astype(float) >= self.winner_rank_threshold_float
        ]
        candidate_df = candidate_df[candidate_df["alpha19_signal_ser"].astype(float) > 0.0]
        candidate_df = candidate_df[candidate_df["ibs_value_ser"].astype(float) < self.max_entry_ibs_float]

        if self.universe_df is not None and self.previous_bar in self.universe_df.index:
            universe_membership_ser = self.universe_df.loc[self.previous_bar]
            universe_symbol_list = universe_membership_ser[universe_membership_ser == 1].index.tolist()
            candidate_df = candidate_df[candidate_df.index.isin(universe_symbol_list)]

        candidate_df = candidate_df.assign(symbol_str=candidate_df.index.astype(str))
        candidate_df = candidate_df.sort_values(
            by=["alpha19_signal_ser", "adv20_dollar_ser", "symbol_str"],
            ascending=[False, False, True],
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

    strategy = Alpha19IbsRsiExitStrategy(
        name="strategy_mr_alpha19_ibs_rsi_exit",
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
