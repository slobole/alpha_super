"""
QPI long-only mean-reversion strategy with price and ADV filters.

Core formulas
-------------
For stock i on decision date t:

    r^{(3)}_{i,t} = C_{i,t} / C_{i,t-3} - 1

    SMA200_{i,t} = (1 / 200) * sum_{k=0}^{199} C_{i,t-k}

    ADV20_{i,t} = (1 / 20) * sum_{k=0}^{19} Turnover_{i,t-k}

    eligible_{i,t}
        = 1[PIT member_t]
        * 1[C_{i,t} > 10]
        * 1[ADV20_{i,t} >= 20,000,000]
        * 1[QPI_{i,t} < 15]
        * 1[C_{i,t} > SMA200_{i,t}]
        * 1[r^{(3)}_{i,t} < 0]

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
from IPython.display import display

from alpha.engine.backtest import run_daily
from alpha.engine.report import save_results
from alpha.engine.strategy import Strategy
from alpha.indicators import adv_dollar_indicator, qp_indicator
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


class QPIPriceAdvStrategy(Strategy):
    """
    QPI stock pod with additional price and liquidity eligibility filters.

    Entry rules:

        eligible_{i,t}
            = 1[PIT member_t]
            * 1[Close_t > 10]
            * 1[ADV20_t >= 20e6]
            * 1[QPI_t < 15]
            * 1[Close_t > SMA200_t]
            * 1[Return^{(3)}_t < 0]

    Exit rule:

        exit_{i,t} = 1[Close_t > High_{t-1}]

    Names are ranked by:
    1. Turnover descending
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
        max_positions_int: int = 4,
        min_price_float: float = 10.0,
        min_adv20_dollar_float: float = 20_000_000.0,
        adv_window_int: int = 20,
        sma_window_int: int = 200,
        qpi_threshold_float: float = 15.0,
        qpi_window_int: int = 3,
        qpi_lookback_years_int: int = 5,
        return_lookback_days_int: int = 3,
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
        if qpi_window_int <= 0:
            raise ValueError("qpi_window_int must be positive.")
        if qpi_lookback_years_int <= 0:
            raise ValueError("qpi_lookback_years_int must be positive.")
        if return_lookback_days_int <= 0:
            raise ValueError("return_lookback_days_int must be positive.")

        self.trade_id_int = 0
        self.current_trade_map: defaultdict[str, int] = defaultdict(default_trade_id_int)
        self.universe_df: pd.DataFrame | None = None

        self.max_positions_int = max_positions_int
        self.min_price_float = min_price_float
        self.min_adv20_dollar_float = min_adv20_dollar_float
        self.adv_window_int = adv_window_int
        self.sma_window_int = sma_window_int
        self.qpi_threshold_float = qpi_threshold_float
        self.qpi_window_int = qpi_window_int
        self.qpi_lookback_years_int = qpi_lookback_years_int
        self.return_lookback_days_int = return_lookback_days_int

    def compute_signals(self, pricing_data: pd.DataFrame) -> pd.DataFrame:
        signal_data = pricing_data.copy()
        symbol_list = signal_data.columns.get_level_values(0).unique()
        tradeable_symbol_list = [
            str(symbol_str)
            for symbol_str in symbol_list
            if not str(symbol_str).startswith("$")
            and (symbol_str, "Close") in signal_data.columns
            and (symbol_str, "Turnover") in signal_data.columns
        ]

        if len(tradeable_symbol_list) == 0:
            return signal_data

        close_df = pd.DataFrame(
            {symbol_str: signal_data[(symbol_str, "Close")] for symbol_str in tradeable_symbol_list},
            index=signal_data.index,
        )
        turnover_df = pd.DataFrame(
            {symbol_str: signal_data[(symbol_str, "Turnover")] for symbol_str in tradeable_symbol_list},
            index=signal_data.index,
        )

        # *** CRITICAL*** The 3-day return must use only trailing closes.
        three_day_return_df = close_df.pct_change(
            periods=self.return_lookback_days_int,
            fill_method=None,
        )
        # *** CRITICAL*** SMA200 must remain a trailing average.
        sma_200_df = close_df.rolling(
            window=self.sma_window_int,
            min_periods=self.sma_window_int,
        ).mean()
        # *** CRITICAL*** ADV20 must remain a trailing average on past turnover only.
        adv20_dollar_df = adv_dollar_indicator(turnover_df, window_int=self.adv_window_int)

        qpi_value_map: dict[str, pd.Series] = {}
        for symbol_str in self.signal_progress(
            tradeable_symbol_list,
            desc_str="qpi signal features",
            total_int=len(tradeable_symbol_list),
        ):
            close_ser = close_df[symbol_str].astype(float)
            # *** CRITICAL*** QPI must be computed from trailing close history only.
            qpi_value_map[symbol_str] = qp_indicator(
                close_ser,
                window_int=self.qpi_window_int,
                lookback_years_int=self.qpi_lookback_years_int,
            )
        qpi_value_df = pd.DataFrame(qpi_value_map, index=signal_data.index)

        feature_frame_list: list[pd.DataFrame] = []
        feature_map = {
            "three_day_return_ser": three_day_return_df,
            "qpi_value_ser": qpi_value_df,
            "sma_200_price_ser": sma_200_df,
            "adv20_dollar_ser": adv20_dollar_df,
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
            close_price_float = close.get((symbol_str, "Close"), pd.NA)
            if len(data.index) < 2 or pd.isna(close_price_float):
                continue

            prior_high_float = data[(symbol_str, "High")].iloc[-2]
            if float(close_price_float) > float(prior_high_float):
                self.order_target_value(
                    symbol_str,
                    0.0,
                    trade_id=self.current_trade_map[symbol_str],
                )
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
            "Close",
            "Turnover",
            "qpi_value_ser",
            "sma_200_price_ser",
            "three_day_return_ser",
            "adv20_dollar_ser",
        ]
        missing_field_list = [
            field_str for field_str in required_field_list if field_str not in candidate_df.columns
        ]
        if len(missing_field_list) > 0:
            return []

        candidate_df = candidate_df.dropna(subset=required_field_list)
        candidate_df = candidate_df[candidate_df["Close"].astype(float) > self.min_price_float]
        candidate_df = candidate_df[
            candidate_df["adv20_dollar_ser"].astype(float) >= self.min_adv20_dollar_float
        ]
        candidate_df = candidate_df[
            candidate_df["qpi_value_ser"].astype(float) < self.qpi_threshold_float
        ]
        candidate_df = candidate_df[
            candidate_df["Close"].astype(float) > candidate_df["sma_200_price_ser"].astype(float)
        ]
        candidate_df = candidate_df[
            candidate_df["three_day_return_ser"].astype(float) < 0.0
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
    pricing_data = get_prices(
        symbol_list,
        benchmark_list,
        start_date_str="1998-01-01",
        end_date_str=None,
    )

    strategy = QPIPriceAdvStrategy(
        name="strategy_mr_qpi_price_adv",
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
