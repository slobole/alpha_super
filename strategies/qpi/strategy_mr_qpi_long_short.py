"""
QPI long/short mean-reversion strategy for point-in-time S&P 500 members.

Core formulas
-------------
For stock i on decision date t:

    r^{(3)}_{i,t} = C_{i,t} / C_{i,t-3} - 1

    rank_{i,t}
        = (1 / L) * sum_{j=0}^{L-1} 1[r^{(3)}_{i,t-j} <= r^{(3)}_{i,t}]

    p^{-}_{i,t}
        = (1 / L) * sum_{j=0}^{L-1} 1[r^{(3)}_{i,t-j} <= 0]

    p^{+}_{i,t} = 1 - p^{-}_{i,t}

    QPI_{i,t}
        = 100 * rank_{i,t} / p^{-}_{i,t},          if r^{(3)}_{i,t} <= 0
        = 100 * (1 - rank_{i,t}) / p^{+}_{i,t},    if r^{(3)}_{i,t} > 0

    SMA200_{i,t} = (1 / 200) * sum_{k=0}^{199} C_{i,t-k}

Long eligibility:

    eligible^{long}_{i,t}
        = 1[PIT member_{i,t}]
        * 1[QPI_{i,t} < 15]
        * 1[r^{(3)}_{i,t} < 0]
        * 1[C_{i,t} > SMA200_{i,t}]

Short eligibility:

    eligible^{short}_{i,t}
        = 1[PIT member_{i,t}]
        * 1[QPI_{i,t} < 15]
        * 1[r^{(3)}_{i,t} > 0]
        * 1[C_{i,t} < SMA200_{i,t}]

Exit rules:

    exit^{long}_{i,t} = 1[C_{i,t} > H_{i,t-1}]

    exit^{short}_{i,t} = 1[C_{i,t} < L_{i,t-1}]

Sizing:

    long_notional_per_trade_t
        = G^{long} * V_{t-1} / N^{long}_{max}

    short_notional_per_trade_t
        = G^{short} * V_{t-1} / N^{short}_{max}

Default gross settings are:

    G^{long} = 1.0
    G^{short} = 1.0

so the pod can reach 200% gross exposure when both sleeves are full.
"""

from __future__ import annotations

from collections import defaultdict
from typing import List

import pandas as pd
from IPython.display import display

from alpha.engine.backtest import run_daily
from alpha.engine.report import save_results
from alpha.engine.strategy import Strategy
from alpha.indicators import qp_indicator
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


class QPILongShortStrategy(Strategy):
    """
    QPI stock pod with separate long and short sleeves.

    Long sleeve:

        eligible^{long}_{i,t}
            = 1[PIT member_t]
            * 1[QPI_t < threshold]
            * 1[Return^{(3)}_t < 0]
            * 1[Close_t > SMA200_t]

    Short sleeve:

        eligible^{short}_{i,t}
            = 1[PIT member_t]
            * 1[QPI_t < threshold]
            * 1[Return^{(3)}_t > 0]
            * 1[Close_t < SMA200_t]

    Exit rules:

        exit^{long}_{i,t} = 1[Close_t > High_{t-1}]

        exit^{short}_{i,t} = 1[Close_t < Low_{t-1}]

    Ranking on both sides:
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
        slippage: float = 0.0001,
        commission_per_share: float = 0.005,
        commission_minimum: float = 1.0,
        long_max_positions_int: int = 4,
        short_max_positions_int: int = 12,
        long_gross_target_float: float = 1.0,
        short_gross_target_float: float = 1.0,
        qpi_threshold_float: float = 15.0,
        sma_window_int: int = 200,
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

        if long_max_positions_int <= 0:
            raise ValueError("long_max_positions_int must be positive.")
        if short_max_positions_int <= 0:
            raise ValueError("short_max_positions_int must be positive.")
        if long_gross_target_float < 0.0:
            raise ValueError("long_gross_target_float must be non-negative.")
        if short_gross_target_float < 0.0:
            raise ValueError("short_gross_target_float must be non-negative.")
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

        self.long_max_positions_int = long_max_positions_int
        self.short_max_positions_int = short_max_positions_int
        self.long_gross_target_float = long_gross_target_float
        self.short_gross_target_float = short_gross_target_float
        self.qpi_threshold_float = qpi_threshold_float
        self.sma_window_int = sma_window_int
        self.qpi_window_int = qpi_window_int
        self.qpi_lookback_years_int = qpi_lookback_years_int
        self.return_lookback_days_int = return_lookback_days_int

    def compute_signals(self, pricing_data_df: pd.DataFrame) -> pd.DataFrame:
        signal_data_df = pricing_data_df.copy()
        symbol_list = signal_data_df.columns.get_level_values(0).unique()
        tradeable_symbol_list = [
            str(symbol_str)
            for symbol_str in symbol_list
            if not str(symbol_str).startswith("$")
            and (symbol_str, "Close") in signal_data_df.columns
            and (symbol_str, "Turnover") in signal_data_df.columns
        ]

        if len(tradeable_symbol_list) == 0:
            return signal_data_df

        close_df = pd.DataFrame(
            {symbol_str: signal_data_df[(symbol_str, "Close")] for symbol_str in tradeable_symbol_list},
            index=signal_data_df.index,
        )

        # *** CRITICAL*** The 3-day return must use only trailing closes that
        # were available on decision date t.
        three_day_return_df = close_df.pct_change(
            periods=self.return_lookback_days_int,
            fill_method=None,
        )
        # *** CRITICAL*** SMA200 must remain a trailing average.
        sma_200_price_df = close_df.rolling(
            window=self.sma_window_int,
            min_periods=self.sma_window_int,
        ).mean()

        qpi_value_map: dict[str, pd.Series] = {}
        for symbol_str in tradeable_symbol_list:
            close_ser = close_df[symbol_str].astype(float)
            # *** CRITICAL*** QPI must be computed from trailing close history
            # only. The indicator must not inspect bars after t.
            qpi_value_map[symbol_str] = qp_indicator(
                close_ser,
                window_int=self.qpi_window_int,
                lookback_years_int=self.qpi_lookback_years_int,
            )
        qpi_value_df = pd.DataFrame(qpi_value_map, index=signal_data_df.index)

        feature_frame_list: list[pd.DataFrame] = []
        feature_map = {
            "three_day_return_ser": three_day_return_df,
            "qpi_value_ser": qpi_value_df,
            "sma_200_price_ser": sma_200_price_df,
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
        short_position_ser = position_ser[position_ser < 0]
        long_slots_int = self.long_max_positions_int - len(long_position_ser)
        short_slots_int = self.short_max_positions_int - len(short_position_ser)

        if len(data_df.index) >= 2:
            for symbol_str in long_position_ser.index:
                close_price_float = close_row_ser.get((symbol_str, "Close"), pd.NA)
                if pd.isna(close_price_float):
                    continue

                # *** CRITICAL*** `data_df` ends at previous_bar = t, so `iloc[-2]`
                # is High_{t-1}. Never use current-bar highs in the decision step.
                prior_high_float = data_df[(symbol_str, "High")].iloc[-2]
                if pd.notna(prior_high_float) and float(close_price_float) > float(prior_high_float):
                    self.order_target_value(
                        symbol_str,
                        0.0,
                        trade_id=self.current_trade_map[symbol_str],
                    )
                    long_slots_int += 1

            for symbol_str in short_position_ser.index:
                close_price_float = close_row_ser.get((symbol_str, "Close"), pd.NA)
                if pd.isna(close_price_float):
                    continue

                # *** CRITICAL*** `data_df` ends at previous_bar = t, so `iloc[-2]`
                # is Low_{t-1}. Never use current-bar lows in the decision step.
                prior_low_float = data_df[(symbol_str, "Low")].iloc[-2]
                if pd.notna(prior_low_float) and float(close_price_float) < float(prior_low_float):
                    self.order_target_value(
                        symbol_str,
                        0.0,
                        trade_id=self.current_trade_map[symbol_str],
                    )
                    short_slots_int += 1

        long_notional_per_trade_float = (
            self.previous_total_value
            * self.long_gross_target_float
            / float(self.long_max_positions_int)
        )
        short_notional_per_trade_float = (
            self.previous_total_value
            * self.short_gross_target_float
            / float(self.short_max_positions_int)
        )

        if long_notional_per_trade_float <= 0.0:
            long_opportunity_symbol_list: list[str] = []
        else:
            long_opportunity_symbol_list = self.get_long_opportunity_list(close_row_ser)

        if short_notional_per_trade_float <= 0.0:
            short_opportunity_symbol_list: list[str] = []
        else:
            short_opportunity_symbol_list = self.get_short_opportunity_list(close_row_ser)

        while long_slots_int > 0 and len(long_opportunity_symbol_list) > 0:
            symbol_str = long_opportunity_symbol_list.pop(0)

            if self.get_position(symbol_str) != 0:
                continue

            self.trade_id_int += 1
            self.current_trade_map[symbol_str] = self.trade_id_int
            self.order_value(
                symbol_str,
                long_notional_per_trade_float,
                trade_id=self.trade_id_int,
            )
            long_slots_int -= 1

        while short_slots_int > 0 and len(short_opportunity_symbol_list) > 0:
            symbol_str = short_opportunity_symbol_list.pop(0)

            if self.get_position(symbol_str) != 0:
                continue

            self.trade_id_int += 1
            self.current_trade_map[symbol_str] = self.trade_id_int
            self.order_value(
                symbol_str,
                -short_notional_per_trade_float,
                trade_id=self.trade_id_int,
            )
            short_slots_int -= 1

    def get_candidate_df(self, close_row_ser: pd.Series) -> pd.DataFrame:
        candidate_df = close_row_ser.unstack()
        candidate_df = candidate_df[~candidate_df.index.astype(str).str.startswith("$")]

        required_field_list = [
            "Close",
            "Turnover",
            "qpi_value_ser",
            "sma_200_price_ser",
            "three_day_return_ser",
        ]
        missing_field_list = [
            field_str for field_str in required_field_list if field_str not in candidate_df.columns
        ]
        if len(missing_field_list) > 0:
            return pd.DataFrame(columns=required_field_list + ["symbol_str"])

        candidate_df = candidate_df.dropna(subset=required_field_list).copy()

        if self.universe_df is not None and self.previous_bar in self.universe_df.index:
            universe_membership_ser = self.universe_df.loc[self.previous_bar]
            universe_symbol_list = universe_membership_ser[universe_membership_ser == 1].index.tolist()
            candidate_df = candidate_df[candidate_df.index.isin(universe_symbol_list)]

        candidate_df = candidate_df.assign(symbol_str=candidate_df.index.astype(str))
        return candidate_df

    def get_long_opportunity_list(self, close_row_ser: pd.Series) -> list[str]:
        candidate_df = self.get_candidate_df(close_row_ser)
        if len(candidate_df) == 0:
            return []

        long_candidate_df = candidate_df[
            (candidate_df["qpi_value_ser"].astype(float) < self.qpi_threshold_float)
            & (candidate_df["three_day_return_ser"].astype(float) < 0.0)
            & (
                candidate_df["Close"].astype(float)
                > candidate_df["sma_200_price_ser"].astype(float)
            )
        ]

        long_candidate_df = long_candidate_df.sort_values(
            by=["Turnover", "symbol_str"],
            ascending=[False, True],
            kind="mergesort",
        )
        return long_candidate_df.index.tolist()

    def get_short_opportunity_list(self, close_row_ser: pd.Series) -> list[str]:
        candidate_df = self.get_candidate_df(close_row_ser)
        if len(candidate_df) == 0:
            return []

        short_candidate_df = candidate_df[
            (candidate_df["qpi_value_ser"].astype(float) < self.qpi_threshold_float)
            & (candidate_df["three_day_return_ser"].astype(float) > 0.0)
            & (
                candidate_df["Close"].astype(float)
                < candidate_df["sma_200_price_ser"].astype(float)
            )
        ]

        short_candidate_df = short_candidate_df.sort_values(
            by=["Turnover", "symbol_str"],
            ascending=[False, True],
            kind="mergesort",
        )
        return short_candidate_df.index.tolist()


if __name__ == "__main__":
    benchmark_list = ["$SPX"]
    symbol_list, universe_df = build_index_constituent_matrix(indexname="S&P 500")
    pricing_data = get_prices(
        symbol_list,
        benchmark_list,
        start_date_str="1998-01-01",
        end_date_str=None,
    )

    strategy = QPILongShortStrategy(
        name="strategy_mr_qpi_long_short",
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
