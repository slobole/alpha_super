"""
Research-only QPI + IBS + RSI2 market-on-close entry POC.

TL;DR: This file isolates a paper-style close-entry variant of
`strategy_mr_qpi_ibs_rsi_exit`:

1. Evaluate the standard QPI + IBS + RSI2 rules on trading day t.
2. Approximate entry with the same day's close.
3. Keep the legacy exit timing: signal on close_t, fill on open_{t+1}.
4. Leave the repository's live-first `run_daily()` contract unchanged.

Core formulas
-------------
For stock i on trading day t:

    r3_{i,t}
        = Close_{i,t} / Close_{i,t-3} - 1

    sma200_{i,t}
        = (1 / 200) * sum_{k=0}^{199} Close_{i,t-k}

    ibs_{i,t}
        = (Close_{i,t} - Low_{i,t}) / (High_{i,t} - Low_{i,t})

    entry_signal_{i,t}
        = 1[QPI_{i,t} < 30]
        * 1[Close_{i,t} > sma200_{i,t}]
        * 1[r3_{i,t} < 0]
        * 1[ibs_{i,t} < 0.10]

    exit_signal_{i,t}
        = 1[ibs_{i,t} > 0.90] or 1[RSI2_{i,t} > 90]

Paper-style execution mapping:

    entry_fill_price_{i,t}^{paper}
        = Close_{i,t}

    exit_fill_price_{i,t+1}^{legacy}
        = Open_{i,t+1}

Per-slot sizing:

    slot_notional_t
        = Equity_t^{close, pre-entry} / max_positions

Execution caveat
----------------
This file is intentionally research-only. With daily OHLC alone, the final
same-day close is not a live-clean approximation of a real close-auction
workflow:

    invalid_daily_shortcut_t
        = f(High_t, Low_t, Close_t) -> fill at Close_t

because the exact closing auction print is only known after the session ends.

Quantitative consequence
------------------------
This POC estimates how much edge appears when the strategy captures the
same-day close to next-open overnight leg:

    overnight_return_{i,t->t+1}
        = Open_{i,t+1} / Close_{i,t} - 1

It must not be presented as executable live performance.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd
from IPython.display import display

from alpha.engine.report import save_results
from strategies.eom_tlt_vs_spy.strategy_eom_trend_ibit import build_results_df
from strategies.qpi.strategy_mr_qpi_ibs_rsi_exit import (
    QPIIbsRsiExitStrategy,
    default_trade_id_int,
    get_prices,
)
from data.norgate_loader import build_index_constituent_matrix


BENCHMARK_SYMBOL_STR = "$SPX"


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
        commission_float = 0.0
        if commission_per_share_float != 0.0:
            commission_float = float(
                max(commission_minimum_float, commission_per_share_float * abs(entry_share_count_int))
            )
        total_cash_needed_float = float(entry_share_count_int) * entry_fill_price_float + commission_float
        if total_cash_needed_float <= cash_value_float + 1e-12:
            return int(entry_share_count_int)
        entry_share_count_int -= 1

    return 0


def run_qpi_ibs_rsi_exit_moc_paper_backtest(
    strategy: QPIIbsRsiExitStrategy,
    pricing_data_df: pd.DataFrame,
    signal_data_df: pd.DataFrame | None = None,
    calendar_idx: pd.DatetimeIndex | None = None,
) -> QPIIbsRsiExitStrategy:
    if strategy.universe_df is None:
        raise RuntimeError("universe_df must be assigned before run_qpi_ibs_rsi_exit_moc_paper_backtest().")

    if signal_data_df is None:
        signal_data_df = strategy.compute_signals(pricing_data_df)

    if calendar_idx is None:
        calendar_idx = pd.DatetimeIndex(signal_data_df.index)
    else:
        calendar_idx = pd.DatetimeIndex(calendar_idx)

    signal_data_df = signal_data_df.loc[calendar_idx].copy()
    pricing_data_df = pricing_data_df.loc[calendar_idx].copy()
    universe_df = strategy.universe_df.reindex(calendar_idx).fillna(0).astype(int)

    active_share_count_map: dict[str, int] = {}
    active_trade_id_map: dict[str, int] = {}
    pending_exit_symbol_set: set[str] = set()
    transaction_row_list: list[dict[str, object]] = []

    cash_value_float = float(strategy._capital_base)
    next_trade_id_int = 0

    portfolio_value_map: dict[pd.Timestamp, float] = {}
    cash_value_map: dict[pd.Timestamp, float] = {}
    total_value_map: dict[pd.Timestamp, float] = {}

    for bar_ts in calendar_idx:
        bar_ts = pd.Timestamp(bar_ts)
        close_row_ser = signal_data_df.loc[bar_ts]

        for symbol_str in sorted(pending_exit_symbol_set):
            if symbol_str not in active_share_count_map:
                continue

            exit_share_count_int = int(active_share_count_map.pop(symbol_str))
            trade_id_int = int(active_trade_id_map.pop(symbol_str))
            strategy.current_trade_map.pop(symbol_str, None)

            open_price_float = float(signal_data_df.loc[bar_ts, (symbol_str, "Open")])
            exit_fill_price_float = float(open_price_float * (1.0 - float(strategy._slippage)))
            exit_commission_float = float(strategy._compute_commission(exit_share_count_int))

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

        held_symbol_list = sorted(active_share_count_map.keys())
        for symbol_str in held_symbol_list:
            ibs_value_float = close_row_ser.get((symbol_str, "ibs_value_ser"), np.nan)
            rsi2_value_float = close_row_ser.get((symbol_str, "rsi2_value_ser"), np.nan)

            exit_for_ibs_bool = (
                pd.notna(ibs_value_float)
                and float(ibs_value_float) > strategy.exit_ibs_threshold_float
            )
            exit_for_rsi2_bool = (
                pd.notna(rsi2_value_float)
                and float(rsi2_value_float) > strategy.exit_rsi2_threshold_float
            )

            if exit_for_ibs_bool or exit_for_rsi2_bool:
                pending_exit_symbol_set.add(symbol_str)

        pre_entry_close_equity_float = float(
            cash_value_float
            + sum(
                float(active_share_count_map[symbol_str]) * float(close_row_ser.loc[(symbol_str, "Close")])
                for symbol_str in active_share_count_map
            )
        )

        # *** CRITICAL*** Pending next-open exits do not free slots on the same
        # close. Reusing those slots at close_t would add implicit overnight
        # leverage relative to the live position count between close_t and
        # open_{t+1}.
        long_slot_int = strategy.max_positions_int - len(active_share_count_map)
        strategy.previous_bar = bar_ts
        strategy.universe_df = universe_df
        opportunity_symbol_list = strategy.get_opportunity_list(close_row_ser)
        slot_notional_float = pre_entry_close_equity_float / float(strategy.max_positions_int)

        while long_slot_int > 0 and len(opportunity_symbol_list) > 0:
            symbol_str = opportunity_symbol_list.pop(0)
            if symbol_str in active_share_count_map:
                continue

            # *** CRITICAL*** This research path intentionally enters on the
            # same bar's final close. That is a paper-style assumption, not a
            # live-clean daily-bar execution model.
            close_price_float = float(close_row_ser.loc[(symbol_str, "Close")])
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

            entry_commission_float = float(strategy._compute_commission(entry_share_count_int))
            next_trade_id_int += 1
            strategy.trade_id_int = int(next_trade_id_int)
            strategy.current_trade_map[symbol_str] = int(next_trade_id_int)
            active_trade_id_map[symbol_str] = int(next_trade_id_int)
            active_share_count_map[symbol_str] = int(entry_share_count_int)

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
            long_slot_int -= 1

        portfolio_value_float = float(
            sum(
                float(active_share_count_map[symbol_str]) * float(close_row_ser.loc[(symbol_str, "Close")])
                for symbol_str in active_share_count_map
            )
        )
        total_value_float = float(cash_value_float + portfolio_value_float)

        portfolio_value_map[bar_ts] = portfolio_value_float
        cash_value_map[bar_ts] = float(cash_value_float)
        total_value_map[bar_ts] = total_value_float

    portfolio_value_ser = pd.Series(portfolio_value_map, dtype=float).sort_index()
    cash_ser = pd.Series(cash_value_map, dtype=float).sort_index()
    total_value_ser = pd.Series(total_value_map, dtype=float).sort_index()

    benchmark_equity_map: dict[str, pd.Series] = {}
    for benchmark_str in strategy._benchmarks:
        benchmark_close_ser = pricing_data_df[(benchmark_str, "Close")].astype(float)
        benchmark_start_close_float = float(benchmark_close_ser.iloc[0])
        benchmark_equity_map[benchmark_str] = (
            benchmark_close_ser / benchmark_start_close_float * float(strategy._capital_base)
        )

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
    strategy._latest_close_price_ser = pricing_data_df.loc[calendar_idx[-1]].xs("Close", level=1).astype(float)
    strategy.current_bar = pd.Timestamp(calendar_idx[-1])
    strategy.cash = float(cash_ser.iloc[-1])
    strategy.portfolio_value = float(portfolio_value_ser.iloc[-1])
    strategy.total_value = float(total_value_ser.iloc[-1])
    strategy.summarize()
    return strategy


if __name__ == "__main__":
    benchmark_list = [BENCHMARK_SYMBOL_STR]
    symbol_list, universe_df = build_index_constituent_matrix(indexname="S&P 500")
    pricing_data_df = get_prices(
        symbol_list,
        benchmark_list,
        start_date_str="1998-01-01",
        end_date_str=None,
    )

    strategy = QPIIbsRsiExitStrategy(
        name="strategy_mr_qpi_ibs_rsi_exit_moc_paper",
        benchmarks=benchmark_list,
        capital_base=100_000.0,
        slippage=0.0001,
        commission_per_share=0.005,
        commission_minimum=1.0,
    )
    strategy.current_trade_map = defaultdict(default_trade_id_int)
    strategy.universe_df = universe_df

    signal_data_df = strategy.compute_signals(pricing_data_df)
    calendar_idx = pricing_data_df.index[pricing_data_df.index.year >= 2004]

    run_qpi_ibs_rsi_exit_moc_paper_backtest(
        strategy=strategy,
        pricing_data_df=pricing_data_df,
        signal_data_df=signal_data_df,
        calendar_idx=calendar_idx,
    )

    strategy.universe_df = None

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)
    display(strategy.summary)
    display(strategy.summary_trades)
    save_results(strategy)
