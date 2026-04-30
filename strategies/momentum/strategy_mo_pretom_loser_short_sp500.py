"""
S&P 500 PreTOM loser market-hedged short strategy.

TL;DR: Short the 10 worst-scoring point-in-time S&P 500 losers during the
six-trading-day pre-turn-of-month window documented by Nathan, Suominen, and
Tasa (2026), and hold an offsetting long SPY hedge during the same window.

Core formulas
-------------
For stock i traded during calendar month m:

    momentum_score_{i,m}
        = Close_ME_{i,m-2} / Close_ME_{i,m-12} - 1

    selected_m
        = 10 PIT S&P 500 members with the lowest momentum_score_{i,m}

Let T = 0 be the final trading day of month m. The active short window is:

    PreTOM_m = {T-9, T-8, T-7, T-6, T-5, T-4}

The target weight is:

    target_weight_{i,d}
        = -1 / N_m    if i in selected_m and d in PreTOM_m
        = 0           otherwise

    hedge_weight_{d}
        = +1.0         if d in PreTOM_m and selected_m is non-empty
        = 0            otherwise

Execution mapping
-----------------
The engine calls iterate() with data through previous_bar and fills orders at
the current_bar open:

    basket decision at T-10 close -> short losers and buy SPY at T-9 open
    flat decision at T-4 close    -> cover losers and sell SPY at T-3 open

Short realism note
------------------
Borrow availability, borrow fees, recalls, and locate constraints are not
modeled. Treat short-side results as optimistic under G-007.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Sequence

import numpy as np
import pandas as pd
from IPython.display import display

WORKSPACE_ROOT_PATH = Path(__file__).resolve().parents[2]
if str(WORKSPACE_ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT_PATH))

from alpha.engine.backtest import run_daily
from alpha.engine.report import save_results
from alpha.engine.strategy import Strategy
from data.norgate_loader import build_index_constituent_matrix, load_raw_prices
from strategies.momentum.strategy_mo_radge_ndx import audit_pit_universe_df


def default_trade_id_int() -> int:
    return -1


@dataclass(frozen=True)
class PretomLoserShortSp500Config:
    indexname_str: str = "S&P 500"
    benchmark_list: tuple[str, ...] = ("$SPX",)
    market_hedge_symbol_str: str = "SPY"
    history_start_date_str: str = "1998-01-01"
    backtest_start_date_str: str = "2000-01-01"
    end_date_str: str | None = None
    skip_month_int: int = 2
    lookback_month_int: int = 12
    max_positions_int: int = 10
    gross_short_exposure_float: float = -1.0
    entry_from_month_end_int: int = -9
    exit_from_month_end_int: int = -3
    capital_base_float: float = 100_000.0
    slippage_float: float = 0.00025
    commission_per_share_float: float = 0.005
    commission_minimum_float: float = 1.0

    def __post_init__(self) -> None:
        if not self.indexname_str:
            raise ValueError("indexname_str must not be empty.")
        if len(self.benchmark_list) == 0:
            raise ValueError("benchmark_list must not be empty.")
        if not self.market_hedge_symbol_str:
            raise ValueError("market_hedge_symbol_str must not be empty.")
        if self.market_hedge_symbol_str in self.benchmark_list:
            raise ValueError("market_hedge_symbol_str must be a tradable symbol, not a benchmark.")
        if pd.Timestamp(self.history_start_date_str) >= pd.Timestamp(self.backtest_start_date_str):
            raise ValueError("history_start_date_str must be earlier than backtest_start_date_str.")
        if self.skip_month_int <= 0:
            raise ValueError("skip_month_int must be positive.")
        if self.lookback_month_int <= self.skip_month_int:
            raise ValueError("lookback_month_int must be greater than skip_month_int.")
        if self.max_positions_int <= 0:
            raise ValueError("max_positions_int must be positive.")
        if self.gross_short_exposure_float >= 0.0:
            raise ValueError("gross_short_exposure_float must be negative.")
        if self.entry_from_month_end_int >= self.exit_from_month_end_int:
            raise ValueError("entry_from_month_end_int must be earlier than exit_from_month_end_int.")
        if self.exit_from_month_end_int >= 0:
            raise ValueError("exit_from_month_end_int must be before month end for this strategy.")
        if self.capital_base_float <= 0.0:
            raise ValueError("capital_base_float must be positive.")
        if self.slippage_float < 0.0:
            raise ValueError("slippage_float must be non-negative.")
        if self.commission_per_share_float < 0.0:
            raise ValueError("commission_per_share_float must be non-negative.")
        if self.commission_minimum_float < 0.0:
            raise ValueError("commission_minimum_float must be non-negative.")


DEFAULT_CONFIG = PretomLoserShortSp500Config()

__all__ = [
    "DEFAULT_CONFIG",
    "PretomLoserShortSp500Config",
    "PretomLoserShortSp500Strategy",
    "build_pretom_target_weight_df",
    "compute_loser_momentum_score_df",
    "get_monthly_decision_close_df",
    "get_pretom_loser_short_sp500_data",
]


def get_monthly_decision_close_df(price_close_df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse daily closes to the actual last tradable close of each month.
    """
    if len(price_close_df.index) == 0:
        raise ValueError("price_close_df must not be empty.")

    # *** CRITICAL*** Monthly close sampling must use actual observed trading
    # dates, not synthetic calendar month-end timestamps.
    decision_date_ser = pd.Series(
        price_close_df.index,
        index=price_close_df.index.to_period("M"),
    ).groupby(level=0).max()

    last_available_ts = pd.Timestamp(price_close_df.index[-1])
    expected_business_month_end_ts = pd.Timestamp(last_available_ts + pd.offsets.BMonthEnd(0))
    if (
        len(decision_date_ser) > 0
        and pd.Timestamp(decision_date_ser.iloc[-1]) == last_available_ts
        and expected_business_month_end_ts.normalize() != last_available_ts.normalize()
    ):
        decision_date_ser = decision_date_ser.iloc[:-1]

    decision_date_idx = pd.DatetimeIndex(decision_date_ser.to_numpy(), name="decision_date_ts")
    monthly_decision_close_df = price_close_df.loc[decision_date_idx].copy()
    monthly_decision_close_df.index = decision_date_idx
    return monthly_decision_close_df


def compute_loser_momentum_score_df(
    price_close_df: pd.DataFrame,
    config: PretomLoserShortSp500Config = DEFAULT_CONFIG,
) -> pd.DataFrame:
    """
    Compute article-faithful loser momentum scores for each trading month.
    """
    monthly_decision_close_df = get_monthly_decision_close_df(price_close_df=price_close_df)

    # *** CRITICAL*** The numerator intentionally skips the most recent month:
    # month m uses Close_ME(m-2), not Close_ME(m) or Close_ME(m-1), preventing
    # short-term reversal leakage into the loser rank.
    skipped_month_close_df = monthly_decision_close_df.shift(config.skip_month_int)

    # *** CRITICAL*** The denominator is the trailing Close_ME(m-12), so the
    # score is a causal t-12 to t-2 cumulative return known before PreTOM.
    lookback_month_close_df = monthly_decision_close_df.shift(config.lookback_month_int)

    momentum_score_df = (skipped_month_close_df / lookback_month_close_df) - 1.0
    momentum_score_df = momentum_score_df.replace([np.inf, -np.inf], np.nan)
    return momentum_score_df


def _get_month_period_to_trading_index_map(trading_index: pd.DatetimeIndex) -> dict[pd.Period, pd.DatetimeIndex]:
    ordered_index = pd.DatetimeIndex(pd.to_datetime(trading_index)).sort_values()
    month_period_ser = pd.Series(ordered_index.to_period("M"), index=ordered_index)
    month_map: dict[pd.Period, pd.DatetimeIndex] = {}
    for month_period, month_index in month_period_ser.groupby(month_period_ser).groups.items():
        month_map[month_period] = pd.DatetimeIndex(month_index).sort_values()
    return month_map


def build_pretom_target_weight_df(
    price_close_df: pd.DataFrame,
    universe_df: pd.DataFrame,
    config: PretomLoserShortSp500Config = DEFAULT_CONFIG,
) -> pd.DataFrame:
    """
    Build daily signed target weights for the article-faithful PreTOM window.
    """
    if len(price_close_df.index) == 0:
        raise ValueError("price_close_df must not be empty.")
    if len(universe_df.index) == 0:
        raise ValueError("universe_df must not be empty.")

    trading_index = pd.DatetimeIndex(price_close_df.index).sort_values()
    symbol_list = [str(symbol_str) for symbol_str in price_close_df.columns]
    target_weight_df = pd.DataFrame(0.0, index=trading_index, columns=symbol_list, dtype=float)
    momentum_score_df = compute_loser_momentum_score_df(price_close_df=price_close_df, config=config)
    month_period_map = _get_month_period_to_trading_index_map(trading_index)

    for month_end_ts, score_ser in momentum_score_df.iterrows():
        month_period = pd.Timestamp(month_end_ts).to_period("M")
        if month_period not in month_period_map:
            continue

        month_trading_index = month_period_map[month_period]
        month_day_count_int = len(month_trading_index)
        entry_pos_int = month_day_count_int - 1 + int(config.entry_from_month_end_int)
        exit_flat_pos_int = month_day_count_int - 1 + int(config.exit_from_month_end_int)
        entry_decision_pos_int = entry_pos_int - 1

        if entry_decision_pos_int < 0 or exit_flat_pos_int <= entry_pos_int:
            continue

        hold_index = month_trading_index[entry_pos_int:exit_flat_pos_int]
        if len(hold_index) == 0:
            continue

        entry_decision_ts = pd.Timestamp(month_trading_index[entry_decision_pos_int])
        if entry_decision_ts not in universe_df.index:
            continue

        # *** CRITICAL*** PreTOM basket selection uses PIT membership as of
        # T-10 close, then enters at T-9 open. Do not use month-end membership
        # to choose stocks for an earlier entry.
        universe_member_ser = universe_df.loc[entry_decision_ts].reindex(symbol_list).fillna(0).astype(int)
        active_symbol_list = universe_member_ser[universe_member_ser == 1].index.astype(str).tolist()
        if len(active_symbol_list) == 0:
            continue

        candidate_score_ser = pd.to_numeric(score_ser.reindex(active_symbol_list), errors="coerce").dropna()
        if len(candidate_score_ser) == 0:
            continue

        selected_count_int = min(int(config.max_positions_int), len(candidate_score_ser))
        candidate_rank_df = pd.DataFrame(
            {
                "momentum_score_float": candidate_score_ser.astype(float),
                "symbol_str": candidate_score_ser.index.astype(str),
            },
            index=candidate_score_ser.index.astype(str),
        )
        candidate_rank_df = candidate_rank_df.sort_values(
            by=["momentum_score_float", "symbol_str"],
            ascending=[True, True],
            kind="mergesort",
        )
        selected_symbol_list = candidate_rank_df.index[:selected_count_int].astype(str).tolist()
        if len(selected_symbol_list) == 0:
            continue

        target_weight_float = float(config.gross_short_exposure_float) / float(len(selected_symbol_list))

        # *** CRITICAL*** The active target window is T-9 through T-4. The
        # target is already flat at T-3 so the engine covers at the T-3 open.
        target_weight_df.loc[hold_index, selected_symbol_list] = target_weight_float

    return target_weight_df


def get_pretom_loser_short_sp500_data(
    config: PretomLoserShortSp500Config = DEFAULT_CONFIG,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    _, raw_universe_df = build_index_constituent_matrix(indexname=config.indexname_str)

    history_start_ts = pd.Timestamp(config.history_start_date_str)
    backtest_start_ts = pd.Timestamp(config.backtest_start_date_str)
    filtered_universe_df = raw_universe_df.loc[raw_universe_df.index >= history_start_ts].copy()
    active_universe_df = filtered_universe_df.loc[filtered_universe_df.index >= backtest_start_ts].copy()
    if config.end_date_str is not None:
        end_date_ts = pd.Timestamp(config.end_date_str)
        active_universe_df = active_universe_df.loc[active_universe_df.index <= end_date_ts]

    active_symbol_list = active_universe_df.columns[active_universe_df.sum(axis=0) > 0].astype(str).tolist()
    if len(active_symbol_list) == 0:
        raise RuntimeError("No active S&P 500 universe symbols were found for the requested backtest window.")

    price_symbol_list = list(dict.fromkeys(active_symbol_list + [config.market_hedge_symbol_str]))
    pricing_data_df = load_raw_prices(
        symbols=price_symbol_list,
        benchmarks=list(config.benchmark_list),
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

    keep_symbol_set = set(
        audited_universe_df.columns.astype(str).tolist()
        + [config.market_hedge_symbol_str]
        + list(config.benchmark_list)
    )
    pricing_data_df = pricing_data_df.loc[
        :,
        pricing_data_df.columns.get_level_values(0).isin(keep_symbol_set),
    ].sort_index()

    close_symbol_list = audited_universe_df.columns.astype(str).tolist()
    price_close_df = pd.DataFrame(
        {symbol_str: pricing_data_df[(symbol_str, "Close")] for symbol_str in close_symbol_list},
        index=pricing_data_df.index,
    ).astype(float)
    target_weight_df = build_pretom_target_weight_df(
        price_close_df=price_close_df,
        universe_df=audited_universe_df,
        config=config,
    )
    return pricing_data_df, audited_universe_df, target_weight_df


class PretomLoserShortSp500Strategy(Strategy):
    """
    Multi-stock short-only PreTOM seasonal momentum pod.
    """

    enable_signal_audit = False
    signal_audit_sample_size = 10

    def __init__(
        self,
        name: str,
        benchmarks: Sequence[str],
        daily_target_weight_df: pd.DataFrame,
        market_hedge_symbol_str: str = "SPY",
        capital_base: float = 100_000.0,
        slippage: float = 0.00025,
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
        if len(daily_target_weight_df.index) == 0:
            raise ValueError("daily_target_weight_df must not be empty.")
        if not market_hedge_symbol_str:
            raise ValueError("market_hedge_symbol_str must not be empty.")

        self.daily_target_weight_df = daily_target_weight_df.astype(float).copy().sort_index()
        self.market_hedge_symbol_str = str(market_hedge_symbol_str)
        self.trade_id_int = 0
        self.current_trade_id_map: dict[str, int] = {}

    def compute_signals(self, pricing_data_df: pd.DataFrame) -> pd.DataFrame:
        return pricing_data_df

    def _get_target_row_ser(self, bar_ts: pd.Timestamp | None) -> pd.Series:
        if bar_ts is None or pd.Timestamp(bar_ts) not in self.daily_target_weight_df.index:
            return pd.Series(0.0, index=self.daily_target_weight_df.columns, dtype=float)
        return self.daily_target_weight_df.loc[pd.Timestamp(bar_ts)].astype(float)

    def iterate(
        self,
        data_df: pd.DataFrame,
        close_row_ser: pd.Series,
        open_price_ser: pd.Series,
    ):
        if data_df is None or close_row_ser is None or self.previous_bar is None:
            return

        previous_target_weight_ser = self._get_target_row_ser(pd.Timestamp(self.previous_bar))
        current_target_weight_ser = self._get_target_row_ser(pd.Timestamp(self.current_bar))
        previous_short_gross_float = float(previous_target_weight_ser[previous_target_weight_ser < 0.0].abs().sum())
        current_short_gross_float = float(current_target_weight_ser[current_target_weight_ser < 0.0].abs().sum())

        changed_symbol_set = set(
            current_target_weight_ser.index[
                ~np.isclose(
                    current_target_weight_ser.to_numpy(dtype=float),
                    previous_target_weight_ser.reindex(current_target_weight_ser.index).to_numpy(dtype=float),
                    atol=1e-12,
                )
            ].astype(str)
        )
        current_position_ser = self.get_positions()
        active_position_symbol_set = set(current_position_ser[current_position_ser != 0].index.astype(str))
        active_position_symbol_set.discard(self.market_hedge_symbol_str)
        order_symbol_list = sorted(changed_symbol_set | active_position_symbol_set)

        for symbol_str in order_symbol_list:
            target_weight_float = float(current_target_weight_ser.get(symbol_str, 0.0))
            current_position_float = float(current_position_ser.get(symbol_str, 0.0))
            if symbol_str not in changed_symbol_set and not np.isclose(target_weight_float, 0.0, atol=1e-12):
                continue

            if np.isclose(target_weight_float, 0.0, atol=1e-12):
                if np.isclose(current_position_float, 0.0, atol=1e-12):
                    continue
                trade_id_int = self.current_trade_id_map.get(symbol_str, default_trade_id_int())
                self.order_target_value(
                    symbol_str,
                    0.0,
                    trade_id=None if trade_id_int == default_trade_id_int() else trade_id_int,
                )
                self.current_trade_id_map.pop(symbol_str, None)
                continue

            if target_weight_float > 0.0:
                raise RuntimeError(f"Unexpected long target for {symbol_str} on {self.current_bar}.")

            if current_position_float >= 0.0 or symbol_str not in self.current_trade_id_map:
                self.trade_id_int += 1
                self.current_trade_id_map[symbol_str] = self.trade_id_int

            # *** CRITICAL*** Target flips are read from previous_bar/current_bar
            # target rows. Orders placed here fill at current_bar open, which
            # preserves T-9 entry and T-3 cover timing.
            self.order_target_percent(
                symbol_str,
                target_weight_float,
                trade_id=self.current_trade_id_map[symbol_str],
            )

        # *** CRITICAL*** The SPY hedge flips on the same open-to-open target
        # schedule as the loser basket. This converts the test from raw
        # short-only returns toward R_market - R_loser.
        if not np.isclose(previous_short_gross_float, current_short_gross_float, atol=1e-12):
            hedge_current_position_float = float(current_position_ser.get(self.market_hedge_symbol_str, 0.0))
            hedge_target_weight_float = current_short_gross_float
            if np.isclose(hedge_target_weight_float, 0.0, atol=1e-12):
                if not np.isclose(hedge_current_position_float, 0.0, atol=1e-12):
                    trade_id_int = self.current_trade_id_map.get(
                        self.market_hedge_symbol_str,
                        default_trade_id_int(),
                    )
                    self.order_target_value(
                        self.market_hedge_symbol_str,
                        0.0,
                        trade_id=None if trade_id_int == default_trade_id_int() else trade_id_int,
                    )
                    self.current_trade_id_map.pop(self.market_hedge_symbol_str, None)
                return

            if self.market_hedge_symbol_str not in self.current_trade_id_map:
                self.trade_id_int += 1
                self.current_trade_id_map[self.market_hedge_symbol_str] = self.trade_id_int
            self.order_target_percent(
                self.market_hedge_symbol_str,
                hedge_target_weight_float,
                trade_id=self.current_trade_id_map[self.market_hedge_symbol_str],
            )

    def finalize(self, current_data: pd.DataFrame):
        position_ser = self.get_positions()
        long_position_ser = position_ser[position_ser > 0]
        disallowed_long_ser = long_position_ser.drop(labels=[self.market_hedge_symbol_str], errors="ignore")
        if len(disallowed_long_ser) > 0:
            raise RuntimeError(f"Loser-short invariant violated. Long positions: {disallowed_long_ser.to_dict()}")


if __name__ == "__main__":
    config = DEFAULT_CONFIG
    pricing_data_df, universe_df, daily_target_weight_df = get_pretom_loser_short_sp500_data(config=config)

    strategy = PretomLoserShortSp500Strategy(
        name="strategy_mo_pretom_loser_short_sp500",
        benchmarks=config.benchmark_list,
        daily_target_weight_df=daily_target_weight_df,
        market_hedge_symbol_str=config.market_hedge_symbol_str,
        capital_base=config.capital_base_float,
        slippage=config.slippage_float,
        commission_per_share=config.commission_per_share_float,
        commission_minimum=config.commission_minimum_float,
    )
    strategy.universe_df = universe_df
    strategy.daily_target_weights = daily_target_weight_df

    calendar_idx = pricing_data_df.index[pricing_data_df.index >= pd.Timestamp(config.backtest_start_date_str)]
    run_daily(
        strategy,
        pricing_data_df,
        calendar=calendar_idx,
        show_progress=False,
        audit_override_bool=None,
    )

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)
    display(strategy.summary)
    display(strategy.summary_trades)
    save_results(strategy)
