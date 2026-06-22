"""
Research-only Russell 3000 gapper close-to-open strategy.

TL;DR: This module implements the requested GAPPERS rule as a Bench/Vanilla
runnable research strategy with a top-level `run_variant()`. It intentionally
uses a custom execution loop because the standard engine fills orders at the
next open, while this rule enters at the same day's close and exits at the next
trading day's open.

Core formulas
-------------
For stock i on trading day t:

    overnight_gap_{i,t}
        = Open_{i,t} / Close_{i,t-1} - 1

    trailing_gap_vol_{i,t}^{252}
        = std(overnight_gap_{i,t-252}, ..., overnight_gap_{i,t-1})

    gap_z_{i,t}
        = overnight_gap_{i,t} / trailing_gap_vol_{i,t}^{252}

There is no mean subtraction. This is a volatility-normalized gap, not a
classic z-score.

Selection at close t:

    eligible_{i,t}
        = 1[PIT_Russell3000_member_{i,t} = 1]
        * 1[2 <= Close_{i,t} <= 10]
        * 1[gap_z_{i,t} > 2]

`Close_t` is the repo's tradable stock OHLC basis from Norgate
`CAPITALSPECIAL`, not total-return signal data. It is also not a separate
unadjusted raw exchange-price feed.

    selected_t = top 10 eligible names by gap_z_{i,t}

Execution mapping:

    buy selected_t at Close_t
    sell all selected_t at Open_{t+1}

Execution caveat
----------------
This is research-only. Daily bars do not prove that a real MOC order could have
been placed after observing the final close. The module is useful for measuring
the close-to-next-open leg, but it must not be described as live-clean.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from IPython.display import display

from alpha.engine.report import save_results
from alpha.engine.strategy import Strategy
from data.norgate_loader import build_index_constituent_matrix, load_raw_prices
from strategies.eom_tlt_vs_spy.strategy_eom_trend_ibit import (
    build_results_df,
    compute_commission_float,
)
from strategies.momentum.strategy_mo_atr_normalized_ndx import audit_pit_universe_df


GAP_VOL_LOOKBACK_DAY_INT = 252
GAP_Z_MIN_FLOAT = 2.0
MIN_ENTRY_PRICE_FLOAT = 2.0
MAX_ENTRY_PRICE_FLOAT = 10.0
MAX_POSITIONS_INT = 10


@dataclass(frozen=True)
class GappersRussell3000Config:
    indexname_str: str = "Russell 3000"
    benchmark_symbol_str: str = "$RUA"
    history_start_date_str: str = "1998-01-01"
    backtest_start_date_str: str = "2000-01-01"
    end_date_str: str | None = None
    gap_vol_lookback_day_int: int = GAP_VOL_LOOKBACK_DAY_INT
    gap_z_min_float: float = GAP_Z_MIN_FLOAT
    min_entry_price_float: float = MIN_ENTRY_PRICE_FLOAT
    max_entry_price_float: float = MAX_ENTRY_PRICE_FLOAT
    max_positions_int: int = MAX_POSITIONS_INT
    capital_base_float: float = 100_000.0
    slippage_float: float = 0.00025
    commission_per_share_float: float = 0.005
    commission_minimum_float: float = 1.0

    def __post_init__(self) -> None:
        if not self.indexname_str:
            raise ValueError("indexname_str must not be empty.")
        if not self.benchmark_symbol_str:
            raise ValueError("benchmark_symbol_str must not be empty.")
        if pd.Timestamp(self.history_start_date_str) >= pd.Timestamp(self.backtest_start_date_str):
            raise ValueError("history_start_date_str must be earlier than backtest_start_date_str.")
        if self.gap_vol_lookback_day_int <= 1:
            raise ValueError("gap_vol_lookback_day_int must be greater than 1.")
        if self.gap_z_min_float <= 0.0:
            raise ValueError("gap_z_min_float must be positive.")
        if self.min_entry_price_float <= 0.0:
            raise ValueError("min_entry_price_float must be positive.")
        if self.max_entry_price_float <= self.min_entry_price_float:
            raise ValueError("max_entry_price_float must exceed min_entry_price_float.")
        if self.max_positions_int <= 0:
            raise ValueError("max_positions_int must be positive.")
        if self.capital_base_float <= 0.0:
            raise ValueError("capital_base_float must be positive.")
        if self.slippage_float < 0.0:
            raise ValueError("slippage_float must be non-negative.")
        if self.commission_per_share_float < 0.0:
            raise ValueError("commission_per_share_float must be non-negative.")
        if self.commission_minimum_float < 0.0:
            raise ValueError("commission_minimum_float must be non-negative.")


DEFAULT_CONFIG = GappersRussell3000Config()


__all__ = [
    "DEFAULT_CONFIG",
    "GAP_VOL_LOOKBACK_DAY_INT",
    "GAP_Z_MIN_FLOAT",
    "GappersRussell3000CloseToOpenStrategy",
    "GappersRussell3000Config",
    "MAX_ENTRY_PRICE_FLOAT",
    "MAX_POSITIONS_INT",
    "MIN_ENTRY_PRICE_FLOAT",
    "build_gappers_signal_data_df",
    "get_gappers_russell3000_data",
    "get_gappers_selection_df",
    "run_gappers_close_to_open_backtest",
    "run_variant",
]


def _tradeable_symbol_list_from_pricing_data(
    pricing_data_df: pd.DataFrame,
    benchmark_list: Sequence[str],
) -> list[str]:
    benchmark_set = set(str(symbol_str) for symbol_str in benchmark_list)
    return [
        str(symbol_str)
        for symbol_str in pricing_data_df.columns.get_level_values(0).unique()
        if str(symbol_str) not in benchmark_set and not str(symbol_str).startswith("$")
    ]


def build_gappers_signal_data_df(
    pricing_data_df: pd.DataFrame,
    benchmark_list: Sequence[str] = (DEFAULT_CONFIG.benchmark_symbol_str,),
    gap_vol_lookback_day_int: int = DEFAULT_CONFIG.gap_vol_lookback_day_int,
    min_entry_price_float: float = DEFAULT_CONFIG.min_entry_price_float,
    max_entry_price_float: float = DEFAULT_CONFIG.max_entry_price_float,
) -> pd.DataFrame:
    if gap_vol_lookback_day_int <= 1:
        raise ValueError("gap_vol_lookback_day_int must be greater than 1.")

    signal_data_df = pricing_data_df.copy()
    tradeable_symbol_list = _tradeable_symbol_list_from_pricing_data(
        pricing_data_df=signal_data_df,
        benchmark_list=benchmark_list,
    )
    if len(tradeable_symbol_list) == 0:
        raise RuntimeError("No tradeable stock symbols were found in pricing_data_df.")

    price_open_df = pd.DataFrame(
        {symbol_str: signal_data_df[(symbol_str, "Open")] for symbol_str in tradeable_symbol_list},
        index=signal_data_df.index,
    ).astype(float)
    price_close_df = pd.DataFrame(
        {symbol_str: signal_data_df[(symbol_str, "Close")] for symbol_str in tradeable_symbol_list},
        index=signal_data_df.index,
    ).astype(float)

    # *** CRITICAL *** lookahead-sensitive: overnight_gap_{i,t} uses only
    # Open_{i,t} and Close_{i,t-1}. Do not replace this with Close_t or any
    # same-day high/low field.
    previous_close_df = price_close_df.shift(1)
    overnight_gap_df = price_open_df / previous_close_df.replace(0.0, np.nan) - 1.0

    # *** CRITICAL *** rolling-window timing: gap_z_{i,t} scales today's gap
    # by the 252 overnight gaps ending at t-1. Including overnight_gap_t in
    # this volatility would let the event being ranked dampen its own score.
    trailing_gap_vol_df = (
        overnight_gap_df.rolling(
            window=int(gap_vol_lookback_day_int),
            min_periods=int(gap_vol_lookback_day_int),
        )
        .std(ddof=1)
        .shift(1)
    )
    gap_z_df = overnight_gap_df / trailing_gap_vol_df.replace(0.0, np.nan)
    gap_z_df = gap_z_df.replace([np.inf, -np.inf], np.nan)
    # Price filter basis: repo tradable stock OHLC, loaded as Norgate
    # CAPITALSPECIAL through load_raw_prices(), not total-return signal data.
    price_filter_pass_df = (
        price_close_df.ge(float(min_entry_price_float))
        & price_close_df.le(float(max_entry_price_float))
    )

    feature_frame_list: list[pd.DataFrame] = []
    feature_map: dict[str, pd.DataFrame] = {
        "overnight_gap_ser": overnight_gap_df,
        f"trailing_gap_vol_{int(gap_vol_lookback_day_int)}_ser": trailing_gap_vol_df,
        "gap_z_ser": gap_z_df,
        "price_filter_pass_bool": price_filter_pass_df.astype(bool),
    }
    for field_str, feature_df in feature_map.items():
        labeled_feature_df = feature_df.copy()
        labeled_feature_df.columns = pd.MultiIndex.from_tuples(
            [(symbol_str, field_str) for symbol_str in labeled_feature_df.columns.astype(str)]
        )
        feature_frame_list.append(labeled_feature_df)

    return pd.concat([signal_data_df] + feature_frame_list, axis=1)


def get_gappers_selection_df(
    close_row_ser: pd.Series,
    universe_row_ser: pd.Series,
    gap_z_min_float: float = DEFAULT_CONFIG.gap_z_min_float,
    max_positions_int: int = DEFAULT_CONFIG.max_positions_int,
) -> pd.DataFrame:
    if max_positions_int <= 0:
        raise ValueError("max_positions_int must be positive.")

    candidate_feature_df = close_row_ser.unstack()
    required_field_list = [
        "Close",
        "overnight_gap_ser",
        "gap_z_ser",
        "price_filter_pass_bool",
    ]
    if any(field_str not in candidate_feature_df.columns for field_str in required_field_list):
        return pd.DataFrame(
            columns=[
                "symbol_str",
                "gap_z_float",
                "overnight_gap_float",
                "close_price_float",
            ]
        )

    active_symbol_set = set(
        universe_row_ser[universe_row_ser.astype(float) == 1.0].index.astype(str).tolist()
    )
    candidate_feature_df = candidate_feature_df[
        candidate_feature_df.index.astype(str).isin(active_symbol_set)
    ].copy()
    if len(candidate_feature_df) == 0:
        return pd.DataFrame(
            columns=[
                "symbol_str",
                "gap_z_float",
                "overnight_gap_float",
                "close_price_float",
            ]
        )

    candidate_feature_df = candidate_feature_df.assign(
        symbol_str=candidate_feature_df.index.astype(str),
        gap_z_float=pd.to_numeric(candidate_feature_df["gap_z_ser"], errors="coerce"),
        overnight_gap_float=pd.to_numeric(candidate_feature_df["overnight_gap_ser"], errors="coerce"),
        close_price_float=pd.to_numeric(candidate_feature_df["Close"], errors="coerce"),
        price_filter_pass_bool=candidate_feature_df["price_filter_pass_bool"].where(
            candidate_feature_df["price_filter_pass_bool"].notna(),
            False,
        ).astype(bool),
    )
    finite_gap_mask_ser = pd.Series(
        np.isfinite(candidate_feature_df["gap_z_float"].to_numpy(dtype=float)),
        index=candidate_feature_df.index,
    )
    finite_close_mask_ser = pd.Series(
        np.isfinite(candidate_feature_df["close_price_float"].to_numpy(dtype=float)),
        index=candidate_feature_df.index,
    )
    eligible_feature_df = candidate_feature_df.loc[
        finite_gap_mask_ser
        & finite_close_mask_ser
        & candidate_feature_df["price_filter_pass_bool"]
        & (candidate_feature_df["gap_z_float"] > float(gap_z_min_float))
    ].copy()
    if len(eligible_feature_df) == 0:
        return eligible_feature_df[
            [
                "symbol_str",
                "gap_z_float",
                "overnight_gap_float",
                "close_price_float",
            ]
        ]

    selected_feature_df = (
        eligible_feature_df.sort_values(
            by=["gap_z_float", "symbol_str"],
            ascending=[False, True],
            kind="mergesort",
        )
        .iloc[:max_positions_int]
        .loc[:, ["symbol_str", "gap_z_float", "overnight_gap_float", "close_price_float"]]
        .reset_index(drop=True)
    )
    selected_feature_df.insert(0, "rank_int", np.arange(1, len(selected_feature_df) + 1))
    return selected_feature_df


class GappersRussell3000CloseToOpenStrategy(Strategy):
    """
    Research-only close-to-next-open GAPPERS strategy container.

    `iterate()` is intentionally unused because the standard engine cannot
    express same-day close entry followed by next-open exit.
    """

    enable_signal_audit = True
    signal_audit_sample_size = 10

    def __init__(
        self,
        name: str,
        benchmarks: Sequence[str],
        config: GappersRussell3000Config = DEFAULT_CONFIG,
    ):
        super().__init__(
            name=name,
            benchmarks=list(benchmarks),
            capital_base=config.capital_base_float,
            slippage=config.slippage_float,
            commission_per_share=config.commission_per_share_float,
            commission_minimum=config.commission_minimum_float,
        )
        self.config = config
        self.universe_df: pd.DataFrame | None = None
        self.signal_data_df = pd.DataFrame(dtype=float)
        self.daily_target_weights = pd.DataFrame(dtype=float)
        self.daily_selection_df = pd.DataFrame()

    def compute_signals(self, pricing_data_df: pd.DataFrame) -> pd.DataFrame:
        return build_gappers_signal_data_df(
            pricing_data_df=pricing_data_df,
            benchmark_list=list(self._benchmarks),
            gap_vol_lookback_day_int=self.config.gap_vol_lookback_day_int,
            min_entry_price_float=self.config.min_entry_price_float,
            max_entry_price_float=self.config.max_entry_price_float,
        )

    def iterate(
        self,
        data_df: pd.DataFrame,
        close_row_ser: pd.Series,
        open_price_ser: pd.Series,
    ):
        return


def _fit_entry_share_count_int(
    slot_notional_float: float,
    cash_value_float: float,
    entry_fill_price_float: float,
    commission_per_share_float: float,
    commission_minimum_float: float,
) -> int:
    if not np.isfinite(entry_fill_price_float) or entry_fill_price_float <= 0.0:
        return 0

    target_share_count_int = int(slot_notional_float / entry_fill_price_float)
    max_affordable_share_count_int = int(cash_value_float / entry_fill_price_float)
    entry_share_count_int = min(target_share_count_int, max_affordable_share_count_int)

    while entry_share_count_int > 0:
        entry_commission_float = compute_commission_float(
            share_count_int=entry_share_count_int,
            commission_per_share_float=commission_per_share_float,
            commission_minimum_float=commission_minimum_float,
        )
        total_cash_needed_float = (
            float(entry_share_count_int) * entry_fill_price_float
            + entry_commission_float
        )
        if total_cash_needed_float <= cash_value_float + 1e-12:
            return int(entry_share_count_int)
        entry_share_count_int -= 1

    return 0


def _exit_fill_price_float(
    signal_data_df: pd.DataFrame,
    symbol_str: str,
    bar_ts: pd.Timestamp,
    slippage_float: float,
) -> float:
    open_price_float = float(signal_data_df.loc[bar_ts, (symbol_str, "Open")])
    if np.isfinite(open_price_float) and open_price_float > 0.0:
        return float(open_price_float * (1.0 - slippage_float))

    raise RuntimeError(
        f"Missing valid Open_{{T+1}} exit price for {symbol_str} on {bar_ts.date()}. "
        "The requested rule is sell Open_{T+1}; do not silently substitute a close price "
        "or corporate-action fallback in this research module."
    )


def run_gappers_close_to_open_backtest(
    strategy: GappersRussell3000CloseToOpenStrategy,
    pricing_data_df: pd.DataFrame,
    signal_data_df: pd.DataFrame | None = None,
    backtest_start_date_str: str | None = DEFAULT_CONFIG.backtest_start_date_str,
) -> GappersRussell3000CloseToOpenStrategy:
    if strategy.universe_df is None:
        raise RuntimeError("universe_df must be assigned before run_gappers_close_to_open_backtest().")

    if signal_data_df is None:
        signal_data_df = strategy.compute_signals(pricing_data_df)

    signal_data_df = signal_data_df.sort_index().copy()
    if backtest_start_date_str is not None:
        signal_data_df = signal_data_df.loc[
            signal_data_df.index >= pd.Timestamp(backtest_start_date_str)
        ].copy()

    trading_index = pd.DatetimeIndex(signal_data_df.index)
    if len(trading_index) < 2:
        raise RuntimeError("signal_data_df must contain at least two backtest bars.")

    universe_df = strategy.universe_df.reindex(trading_index).ffill().fillna(0).astype(int)
    close_price_df = signal_data_df.xs("Close", axis=1, level=1).astype(float)

    benchmark_equity_map: dict[str, pd.Series] = {}
    for benchmark_str in strategy._benchmarks:
        if benchmark_str not in close_price_df.columns:
            raise RuntimeError(f"Missing benchmark close data for {benchmark_str}.")
        benchmark_close_ser = close_price_df.loc[trading_index, benchmark_str].astype(float)
        benchmark_start_close_float = float(benchmark_close_ser.iloc[0])
        if not np.isfinite(benchmark_start_close_float) or benchmark_start_close_float <= 0.0:
            raise RuntimeError(f"Invalid benchmark start close for {benchmark_str}.")
        benchmark_equity_map[benchmark_str] = (
            benchmark_close_ser / benchmark_start_close_float * float(strategy._capital_base)
        )

    active_share_count_map: dict[str, int] = {}
    active_trade_id_map: dict[str, int] = {}
    pending_exit_symbol_set: set[str] = set()
    transaction_row_list: list[dict[str, object]] = []
    daily_target_weight_row_list: list[dict[str, float | pd.Timestamp]] = []
    selection_row_list: list[dict[str, object]] = []

    cash_value_float = float(strategy._capital_base)
    next_trade_id_int = 0

    portfolio_value_map: dict[pd.Timestamp, float] = {}
    cash_value_map: dict[pd.Timestamp, float] = {}
    total_value_map: dict[pd.Timestamp, float] = {}

    for bar_position_int, bar_ts in enumerate(trading_index):
        bar_ts = pd.Timestamp(bar_ts)
        close_row_ser = signal_data_df.loc[bar_ts]

        for symbol_str in sorted(pending_exit_symbol_set):
            if symbol_str not in active_share_count_map:
                continue

            exit_share_count_int = int(active_share_count_map.pop(symbol_str))
            trade_id_int = int(active_trade_id_map.pop(symbol_str))
            exit_fill_price_float = _exit_fill_price_float(
                signal_data_df=signal_data_df,
                symbol_str=symbol_str,
                bar_ts=bar_ts,
                slippage_float=float(strategy._slippage),
            )
            exit_commission_float = compute_commission_float(
                share_count_int=exit_share_count_int,
                commission_per_share_float=float(strategy._commission_per_share),
                commission_minimum_float=float(strategy._commission_minimum),
            )

            cash_value_float += float(exit_share_count_int) * exit_fill_price_float
            cash_value_float -= exit_commission_float
            transaction_row_list.append(
                {
                    "trade_id": trade_id_int,
                    "bar": bar_ts,
                    "asset": symbol_str,
                    "amount": -int(exit_share_count_int),
                    "price": float(exit_fill_price_float),
                    "total_value": float(-exit_share_count_int * exit_fill_price_float),
                    "order_id": len(transaction_row_list) + 1,
                    "commission": float(exit_commission_float),
                }
            )

        pending_exit_symbol_set = set()

        is_last_bar_bool = bar_position_int == len(trading_index) - 1
        selected_feature_df = pd.DataFrame()
        if not is_last_bar_bool:
            universe_row_ser = universe_df.loc[bar_ts]
            selected_feature_df = get_gappers_selection_df(
                close_row_ser=close_row_ser,
                universe_row_ser=universe_row_ser,
                gap_z_min_float=strategy.config.gap_z_min_float,
                max_positions_int=strategy.config.max_positions_int,
            )

        pre_entry_close_equity_float = float(
            cash_value_float
            + sum(
                float(active_share_count_map[symbol_str])
                * float(close_row_ser.loc[(symbol_str, "Close")])
                for symbol_str in active_share_count_map
            )
        )
        slot_notional_float = pre_entry_close_equity_float / float(strategy.config.max_positions_int)

        for _, selection_row_ser in selected_feature_df.iterrows():
            symbol_str = str(selection_row_ser["symbol_str"])
            if symbol_str in active_share_count_map:
                continue

            close_price_float = float(selection_row_ser["close_price_float"])
            entry_fill_price_float = float(close_price_float * (1.0 + float(strategy._slippage)))
            entry_share_count_int = _fit_entry_share_count_int(
                slot_notional_float=slot_notional_float,
                cash_value_float=cash_value_float,
                entry_fill_price_float=entry_fill_price_float,
                commission_per_share_float=float(strategy._commission_per_share),
                commission_minimum_float=float(strategy._commission_minimum),
            )
            if entry_share_count_int <= 0:
                continue

            entry_commission_float = compute_commission_float(
                share_count_int=entry_share_count_int,
                commission_per_share_float=float(strategy._commission_per_share),
                commission_minimum_float=float(strategy._commission_minimum),
            )
            next_trade_id_int += 1
            active_share_count_map[symbol_str] = int(entry_share_count_int)
            active_trade_id_map[symbol_str] = int(next_trade_id_int)
            pending_exit_symbol_set.add(symbol_str)

            cash_value_float -= float(entry_share_count_int) * entry_fill_price_float
            cash_value_float -= entry_commission_float
            transaction_row_list.append(
                {
                    "trade_id": int(next_trade_id_int),
                    "bar": bar_ts,
                    "asset": symbol_str,
                    "amount": int(entry_share_count_int),
                    "price": float(entry_fill_price_float),
                    "total_value": float(entry_share_count_int * entry_fill_price_float),
                    "order_id": len(transaction_row_list) + 1,
                    "commission": float(entry_commission_float),
                }
            )
            selection_row_list.append(
                {
                    "bar_ts": bar_ts,
                    "rank_int": int(selection_row_ser["rank_int"]),
                    "symbol_str": symbol_str,
                    "gap_z_float": float(selection_row_ser["gap_z_float"]),
                    "overnight_gap_float": float(selection_row_ser["overnight_gap_float"]),
                    "close_price_float": close_price_float,
                    "entry_fill_price_float": entry_fill_price_float,
                    "entry_share_count_int": int(entry_share_count_int),
                }
            )

        portfolio_value_float = float(
            sum(
                float(active_share_count_map[symbol_str])
                * float(close_row_ser.loc[(symbol_str, "Close")])
                for symbol_str in active_share_count_map
            )
        )
        total_value_float = float(cash_value_float + portfolio_value_float)

        portfolio_value_map[bar_ts] = portfolio_value_float
        cash_value_map[bar_ts] = float(cash_value_float)
        total_value_map[bar_ts] = total_value_float

        target_weight_row_map: dict[str, float | pd.Timestamp] = {"bar_ts": bar_ts}
        total_value_for_weights_float = total_value_float if total_value_float > 0.0 else np.nan
        for symbol_str in sorted(active_share_count_map.keys()):
            target_weight_row_map[symbol_str] = (
                float(active_share_count_map[symbol_str])
                * float(close_row_ser.loc[(symbol_str, "Close")])
            ) / total_value_for_weights_float
        daily_target_weight_row_list.append(target_weight_row_map)

    portfolio_value_ser = pd.Series(portfolio_value_map, dtype=float).sort_index()
    cash_ser = pd.Series(cash_value_map, dtype=float).sort_index()
    total_value_ser = pd.Series(total_value_map, dtype=float).sort_index()

    strategy.results = build_results_df(
        total_value_ser=total_value_ser,
        portfolio_value_ser=portfolio_value_ser,
        cash_ser=cash_ser,
        benchmark_equity_map=benchmark_equity_map,
    )
    strategy._transactions = pd.DataFrame(
        transaction_row_list,
        columns=["trade_id", "bar", "asset", "amount", "price", "total_value", "order_id", "commission"],
    )
    strategy._position_amount_map = {
        symbol_str: float(share_count_int)
        for symbol_str, share_count_int in active_share_count_map.items()
    }
    strategy._latest_close_price_ser = close_price_df.iloc[-1].astype(float)
    strategy.current_bar = pd.Timestamp(trading_index[-1])
    strategy.cash = float(cash_ser.iloc[-1])
    strategy.portfolio_value = float(portfolio_value_ser.iloc[-1])
    strategy.total_value = float(total_value_ser.iloc[-1])
    strategy.signal_data_df = signal_data_df.copy()
    strategy.daily_target_weights = (
        pd.DataFrame(daily_target_weight_row_list)
        .set_index("bar_ts")
        .sort_index()
        .fillna(0.0)
    )
    strategy.daily_selection_df = pd.DataFrame(selection_row_list)
    strategy.summarize()
    return strategy


def get_gappers_russell3000_data(
    config: GappersRussell3000Config = DEFAULT_CONFIG,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    _, raw_universe_df = build_index_constituent_matrix(indexname=config.indexname_str)

    history_start_ts = pd.Timestamp(config.history_start_date_str)
    backtest_start_ts = pd.Timestamp(config.backtest_start_date_str)
    filtered_universe_df = raw_universe_df.loc[raw_universe_df.index >= history_start_ts].copy()
    active_universe_df = filtered_universe_df.loc[filtered_universe_df.index >= backtest_start_ts].copy()
    if config.end_date_str is not None:
        active_universe_df = active_universe_df.loc[
            active_universe_df.index <= pd.Timestamp(config.end_date_str)
        ]

    active_symbol_list = active_universe_df.columns[
        active_universe_df.sum(axis=0) > 0
    ].astype(str).tolist()
    if len(active_symbol_list) == 0:
        raise RuntimeError(f"No active {config.indexname_str} symbols were found for the requested window.")

    pricing_data_df = load_raw_prices(
        symbols=active_symbol_list,
        benchmarks=[config.benchmark_symbol_str],
        start_date=config.history_start_date_str,
        end_date=config.end_date_str,
    )
    loaded_symbol_list = [
        symbol_str
        for symbol_str in active_symbol_list
        if symbol_str in pricing_data_df.columns.get_level_values(0)
    ]
    audited_universe_df = audit_pit_universe_df(
        universe_df=filtered_universe_df,
        execution_index=pricing_data_df.index,
        tradeable_symbol_list=loaded_symbol_list,
    )

    keep_symbol_set = set(audited_universe_df.columns.astype(str).tolist() + [config.benchmark_symbol_str])
    pricing_data_df = pricing_data_df.loc[
        :,
        pricing_data_df.columns.get_level_values(0).isin(keep_symbol_set),
    ].sort_index()
    return pricing_data_df, audited_universe_df


def _write_assumptions_md(output_path: Path, strategy: GappersRussell3000CloseToOpenStrategy) -> None:
    config = strategy.config
    assumption_md_str = f"""# GAPPERS Russell 3000 Assumptions

- Research-only strategy; no live/release wiring.
- Universe: `{config.indexname_str}` point-in-time membership through Norgate.
- Benchmark: `{config.benchmark_symbol_str}`.
- Price filter basis: `Close_t` from repo tradable stock OHLC via Norgate `CAPITALSPECIAL`; this is not total-return data and not an unadjusted raw exchange-price feed.
- Signal: `gap_z = (Open_t / Close_(t-1) - 1) / std_252(overnight_gap ending at t-1)`.
- No mean subtraction is used in `gap_z`.
- Entry filter: `gap_z > {config.gap_z_min_float:.4f}` and `{config.min_entry_price_float:.2f} <= Close_t <= {config.max_entry_price_float:.2f}`.
- Selection: top `{config.max_positions_int}` eligible stocks by `gap_z`, equal slot notional.
- Entry fill: `Close_t * (1 + slippage)`.
- Exit fill: `Open_(t+1) * (1 - slippage)`.
- Slippage: `{config.slippage_float:.6f}` per side.
- Commission: `{config.commission_per_share_float:.6f}` per share, minimum `{config.commission_minimum_float:.2f}`.
- Missing next-open exits are fatal because the rule explicitly requires `Open_(t+1)`.
- Same-close entry is a daily-data MOC approximation and is not live-clean.
"""
    (output_path / "gappers_assumptions.md").write_text(assumption_md_str, encoding="utf-8")


def run_variant(
    show_display_bool: bool = True,
    save_results_bool: bool = True,
    output_dir_str: str = "results",
    backtest_start_date_str: str | None = None,
    capital_base_float: float | None = None,
    end_date_str: str | None = None,
    pricing_data_df: pd.DataFrame | None = None,
    universe_df: pd.DataFrame | None = None,
    audit_override_bool: bool | None = None,
) -> GappersRussell3000CloseToOpenStrategy:
    config_obj = DEFAULT_CONFIG
    if (
        backtest_start_date_str is not None
        or capital_base_float is not None
        or end_date_str is not None
    ):
        config_obj = replace(
            DEFAULT_CONFIG,
            backtest_start_date_str=(
                DEFAULT_CONFIG.backtest_start_date_str
                if backtest_start_date_str is None
                else backtest_start_date_str
            ),
            capital_base_float=(
                DEFAULT_CONFIG.capital_base_float
                if capital_base_float is None
                else float(capital_base_float)
            ),
            end_date_str=end_date_str,
        )

    if pricing_data_df is None or universe_df is None:
        pricing_data_df, universe_df = get_gappers_russell3000_data(config=config_obj)

    strategy_obj = GappersRussell3000CloseToOpenStrategy(
        name="strategy_mo_gappers_russell3000_close_to_open",
        benchmarks=[config_obj.benchmark_symbol_str],
        config=config_obj,
    )
    strategy_obj.universe_df = universe_df

    signal_data_df = strategy_obj.compute_signals(pricing_data_df)
    if audit_override_bool is None or audit_override_bool:
        strategy_obj.audit_signals(pricing_data_df, signal_data_df)
    run_gappers_close_to_open_backtest(
        strategy=strategy_obj,
        pricing_data_df=pricing_data_df,
        signal_data_df=signal_data_df,
        backtest_start_date_str=config_obj.backtest_start_date_str,
    )

    if show_display_bool:
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 1000)
        display(strategy_obj.summary)
        display(strategy_obj.summary_trades)
        display(strategy_obj.daily_selection_df.tail(20))

    if save_results_bool:
        output_path = save_results(strategy_obj, output_dir=output_dir_str)
        strategy_obj.daily_selection_df.to_csv(output_path / "daily_selection.csv", index=False)
        strategy_obj.daily_target_weights.to_csv(output_path / "daily_target_weights.csv")
        _write_assumptions_md(output_path=output_path, strategy=strategy_obj)

    return strategy_obj


if __name__ == "__main__":
    run_variant()
