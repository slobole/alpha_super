"""
Research-only DV2 market-on-close entry POC.

TL;DR: This file isolates a Phase-0 paper-style DV2 variant that:

1. Uses the legacy DV2 daily filters on day t.
2. Approximates MOC entry with the same day's close.
3. Keeps the legacy exit timing: signal on close_t, fill on open_{t+1}.
4. Does not modify the repository's live-first `run_daily()` contract.

Core formulas
-------------
For symbol i on trading day t:

    p126d_return_{i,t}
        = Close_{i,t} / Close_{i,t-126} - 1

    sma200_{i,t}
        = (1 / 200) * sum_{k=0}^{199} Close_{i,t-k}

    entry_signal_{i,t}
        = 1[DV2_{i,t} < 10]
        * 1[Close_{i,t} > sma200_{i,t}]
        * 1[p126d_return_{i,t} > 0.05]

    exit_signal_{i,t}
        = 1[Close_{i,t} > High_{i,t-1}]

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
same-day close is not a live-clean approximation of a real MOC workflow:

    invalid_daily_shortcut_t
        = f(High_t, Low_t, Close_t) -> fill at Close_t

because the exact final close is only known after the session ends.

Quantitative consequence
------------------------
This POC is useful to estimate how much edge appears when the strategy
captures the close-to-next-open overnight leg:

    overnight_return_{i,t->t+1}
        = Open_{i,t+1} / Close_{i,t} - 1

It must not be presented as executable live performance.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from IPython.display import display

from alpha.engine.report import save_results
from alpha.engine.strategy import Strategy
from alpha.indicators import dv2_indicator
from data.norgate_loader import build_index_constituent_matrix, load_raw_prices
from strategies.eom_tlt_vs_spy.strategy_eom_trend_ibit import (
    build_results_df,
    compute_commission_float,
)


BENCHMARK_SYMBOL_STR = "$SPX"
DV2_LOOKBACK_DAY_INT = 126
NATR_LOOKBACK_DAY_INT = 14
SMA_WINDOW_DAY_INT = 200
P126_RETURN_MIN_FLOAT = 0.05
DV2_ENTRY_MAX_FLOAT = 10.0
DEFAULT_MAX_POSITIONS_INT = 10


def get_dv2_moc_paper_prices(
    symbol_iter: Iterable[str],
    benchmark_list: list[str],
    start_date_str: str = "1998-01-01",
    end_date_str: str | None = None,
) -> pd.DataFrame:
    return load_raw_prices(list(symbol_iter), benchmark_list, start_date_str, end_date_str)


def compute_natr_value_ser(
    high_price_ser: pd.Series,
    low_price_ser: pd.Series,
    close_price_ser: pd.Series,
    window_int: int = NATR_LOOKBACK_DAY_INT,
) -> pd.Series:
    """
    Compute normalized ATR with Wilder-style exponential smoothing.

    Formulas:

        true_range_t
            = max(
                High_t - Low_t,
                |High_t - Close_{t-1}|,
                |Low_t - Close_{t-1}|
            )

        atr_t
            = WilderMean(true_range_t, window_int)

        natr_t
            = 100 * atr_t / Close_t
    """
    high_price_ser = pd.Series(high_price_ser, copy=True).astype(float)
    low_price_ser = pd.Series(low_price_ser, copy=True).astype(float)
    close_price_ser = pd.Series(close_price_ser, copy=True).astype(float)

    # *** CRITICAL*** Previous close must be lagged by exactly one bar.
    # Using the current or future close would leak unavailable information
    # into the true-range calculation.
    previous_close_ser = close_price_ser.shift(1)
    true_range_df = pd.concat(
        [
            high_price_ser - low_price_ser,
            (high_price_ser - previous_close_ser).abs(),
            (low_price_ser - previous_close_ser).abs(),
        ],
        axis=1,
    )
    true_range_ser = true_range_df.max(axis=1)

    atr_value_ser = true_range_ser.ewm(
        alpha=1.0 / float(window_int),
        adjust=False,
        min_periods=window_int,
    ).mean()
    natr_value_ser = 100.0 * atr_value_ser / close_price_ser.replace(0.0, np.nan)
    return natr_value_ser.astype(float)


def build_dv2_moc_paper_signal_data_df(pricing_data_df: pd.DataFrame) -> pd.DataFrame:
    signal_data_df = pricing_data_df.copy()
    feature_col_map: dict[tuple[str, str], pd.Series] = {}
    symbol_list = signal_data_df.columns.get_level_values(0).unique().tolist()

    for symbol_str in symbol_list:
        if str(symbol_str).startswith("$") or (symbol_str, "Close") not in signal_data_df.columns:
            continue

        close_price_ser = signal_data_df[(symbol_str, "Close")].astype(float)
        high_price_ser = signal_data_df[(symbol_str, "High")].astype(float)
        low_price_ser = signal_data_df[(symbol_str, "Low")].astype(float)

        # *** CRITICAL*** The 126-day return must remain strictly trailing.
        # Any forward shift here would leak future prices into the current
        # entry filter.
        p126d_return_ser = close_price_ser / close_price_ser.shift(DV2_LOOKBACK_DAY_INT) - 1.0
        natr_value_ser = compute_natr_value_ser(
            high_price_ser=high_price_ser,
            low_price_ser=low_price_ser,
            close_price_ser=close_price_ser,
            window_int=NATR_LOOKBACK_DAY_INT,
        )
        dv2_value_ser = dv2_indicator(
            close_price_ser,
            high_price_ser,
            low_price_ser,
            length_int=DV2_LOOKBACK_DAY_INT,
        )

        # *** CRITICAL*** The 200-day SMA must stay strictly backward-looking.
        # A centered or forward window would leak future trend information.
        sma_200_price_ser = close_price_ser.rolling(SMA_WINDOW_DAY_INT).mean()

        feature_col_map[(symbol_str, "p126d_return_ser")] = p126d_return_ser
        feature_col_map[(symbol_str, "natr_value_ser")] = natr_value_ser
        feature_col_map[(symbol_str, "dv2_value_ser")] = dv2_value_ser
        feature_col_map[(symbol_str, "sma_200_price_ser")] = sma_200_price_ser

    if len(feature_col_map) == 0:
        return signal_data_df

    feature_data_df = pd.DataFrame(feature_col_map, index=signal_data_df.index)
    return pd.concat([signal_data_df, feature_data_df], axis=1)


def get_dv2_opportunity_list(
    close_row_ser: pd.Series,
    universe_row_ser: pd.Series,
) -> list[str]:
    stock_mask_vec = ~close_row_ser.index.get_level_values(0).astype(str).str.startswith("$")
    stock_row_ser = close_row_ser.loc[stock_mask_vec]
    if len(stock_row_ser) == 0:
        return []

    stock_feature_df = stock_row_ser.unstack()
    required_col_list = [
        "Close",
        "p126d_return_ser",
        "natr_value_ser",
        "dv2_value_ser",
        "sma_200_price_ser",
    ]
    if any(field_str not in stock_feature_df.columns for field_str in required_col_list):
        return []

    stock_feature_df = stock_feature_df.dropna(subset=required_col_list)
    if len(stock_feature_df) == 0:
        return []

    eligible_feature_df = stock_feature_df[
        (stock_feature_df["dv2_value_ser"] < DV2_ENTRY_MAX_FLOAT)
        & (stock_feature_df["Close"] > stock_feature_df["sma_200_price_ser"])
        & (stock_feature_df["p126d_return_ser"] > P126_RETURN_MIN_FLOAT)
    ].sort_values("natr_value_ser", ascending=False)

    tradable_symbol_list = universe_row_ser[universe_row_ser == 1].index.tolist()
    return eligible_feature_df[eligible_feature_df.index.isin(tradable_symbol_list)].index.tolist()


class DV2MocPaperResearchStrategy(Strategy):
    """
    Research-only DV2 same-close-entry container.

    `iterate()` is intentionally unused because this file bypasses the
    repository's next-open engine contract and runs a custom execution loop.
    """

    enable_signal_audit = True
    signal_audit_sample_size = 10

    def __init__(
        self,
        name: str,
        benchmarks: list[str] | tuple[str, ...],
        capital_base: float = 100_000.0,
        slippage: float = 0.0001,
        commission_per_share: float = 0.005,
        commission_minimum: float = 1.0,
        max_positions_int: int = DEFAULT_MAX_POSITIONS_INT,
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

        self.max_positions_int = int(max_positions_int)
        self.universe_df: pd.DataFrame | None = None
        self.daily_target_weights = pd.DataFrame(dtype=float)

    def compute_signals(self, pricing_data_df: pd.DataFrame) -> pd.DataFrame:
        return build_dv2_moc_paper_signal_data_df(pricing_data_df)

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


def run_dv2_moc_paper_backtest(
    strategy: DV2MocPaperResearchStrategy,
    pricing_data_df: pd.DataFrame,
    signal_data_df: pd.DataFrame | None = None,
) -> DV2MocPaperResearchStrategy:
    if strategy.universe_df is None:
        raise RuntimeError("universe_df must be assigned before run_dv2_moc_paper_backtest().")

    if signal_data_df is None:
        signal_data_df = strategy.compute_signals(pricing_data_df)

    signal_data_df = signal_data_df.sort_index().copy()
    trading_index = pd.DatetimeIndex(signal_data_df.index)
    if len(trading_index) == 0:
        raise RuntimeError("signal_data_df must contain at least one bar.")

    universe_df = strategy.universe_df.reindex(trading_index).fillna(0).astype(int)

    active_share_count_map: dict[str, int] = {}
    active_trade_id_map: dict[str, int] = {}
    pending_exit_symbol_set: set[str] = set()
    transaction_row_list: list[dict[str, object]] = []
    daily_target_weight_row_list: list[dict[str, float]] = []

    cash_value_float = float(strategy._capital_base)
    next_trade_id_int = 0

    portfolio_value_map: dict[pd.Timestamp, float] = {}
    cash_value_map: dict[pd.Timestamp, float] = {}
    total_value_map: dict[pd.Timestamp, float] = {}

    close_price_df = signal_data_df.xs("Close", axis=1, level=1).astype(float)
    benchmark_equity_map: dict[str, pd.Series] = {}
    for benchmark_str in strategy._benchmarks:
        benchmark_close_ser = close_price_df[benchmark_str].astype(float)
        benchmark_equity_map[benchmark_str] = (
            benchmark_close_ser / float(benchmark_close_ser.iloc[0]) * float(strategy._capital_base)
        )

    for bar_position_int, bar_ts in enumerate(trading_index):
        bar_ts = pd.Timestamp(bar_ts)
        close_row_ser = signal_data_df.loc[bar_ts]

        for symbol_str in sorted(pending_exit_symbol_set):
            if symbol_str not in active_share_count_map:
                continue

            exit_share_count_int = int(active_share_count_map.pop(symbol_str))
            trade_id_int = int(active_trade_id_map.pop(symbol_str))
            open_price_float = float(signal_data_df.loc[bar_ts, (symbol_str, "Open")])
            exit_fill_price_float = float(open_price_float * (1.0 - float(strategy._slippage)))
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

        held_symbol_list_before_entry = sorted(active_share_count_map.keys())
        for symbol_str in held_symbol_list_before_entry:
            if bar_position_int == 0:
                continue

            previous_bar_ts = pd.Timestamp(trading_index[bar_position_int - 1])
            close_price_float = float(close_row_ser.loc[(symbol_str, "Close")])

            # *** CRITICAL*** The exit trigger uses High_{t-1}, not High_t.
            # Replacing this with same-day high changes both timing and meaning.
            previous_high_float = float(signal_data_df.loc[previous_bar_ts, (symbol_str, "High")])
            if close_price_float > previous_high_float:
                pending_exit_symbol_set.add(symbol_str)

        pre_entry_close_equity_float = float(
            cash_value_float
            + sum(
                float(active_share_count_map[symbol_str]) * float(close_row_ser.loc[(symbol_str, "Close")])
                for symbol_str in active_share_count_map
            )
        )

        universe_row_ser = universe_df.loc[bar_ts]
        long_slot_int = strategy.max_positions_int - len(held_symbol_list_before_entry)
        opportunity_list = get_dv2_opportunity_list(close_row_ser, universe_row_ser)
        slot_notional_float = pre_entry_close_equity_float / float(strategy.max_positions_int)

        while long_slot_int > 0 and len(opportunity_list) > 0:
            symbol_str = opportunity_list.pop(0)
            if symbol_str in active_share_count_map:
                continue

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

            entry_commission_float = compute_commission_float(
                share_count_int=entry_share_count_int,
                commission_per_share_float=float(strategy._commission_per_share),
                commission_minimum_float=float(strategy._commission_minimum),
            )

            next_trade_id_int += 1
            active_share_count_map[symbol_str] = int(entry_share_count_int)
            active_trade_id_map[symbol_str] = int(next_trade_id_int)

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

        total_value_for_weights_float = total_value_float if total_value_float > 0.0 else np.nan
        target_weight_row_map = {
            symbol_str: (
                float(active_share_count_map[symbol_str]) * float(close_row_ser.loc[(symbol_str, "Close")])
            ) / total_value_for_weights_float
            for symbol_str in sorted(active_share_count_map.keys())
        }
        target_weight_row_map["bar_ts"] = bar_ts
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
    strategy.daily_target_weights = (
        pd.DataFrame(daily_target_weight_row_list)
        .set_index("bar_ts")
        .sort_index()
        .fillna(0.0)
    )
    strategy.summarize()
    return strategy


if __name__ == "__main__":
    benchmark_list = [BENCHMARK_SYMBOL_STR]
    index_symbol_list, universe_df = build_index_constituent_matrix(indexname="S&P 500")
    pricing_data_df = get_dv2_moc_paper_prices(
        index_symbol_list,
        benchmark_list,
        start_date_str="1998-01-01",
        end_date_str=None,
    )

    strategy = DV2MocPaperResearchStrategy(
        name="strategy_mr_dv2_moc_paper",
        benchmarks=benchmark_list,
        capital_base=100_000.0,
        slippage=0.0001,
        commission_per_share=0.005,
        commission_minimum=1.0,
    )
    strategy.universe_df = universe_df

    signal_data_df = strategy.compute_signals(pricing_data_df)
    signal_data_df = signal_data_df.loc[signal_data_df.index.year >= 2004].copy()
    strategy.universe_df = strategy.universe_df.loc[signal_data_df.index].copy()

    run_dv2_moc_paper_backtest(
        strategy=strategy,
        pricing_data_df=pricing_data_df.loc[signal_data_df.index].copy(),
        signal_data_df=signal_data_df,
    )

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)
    display(strategy.summary)
    display(strategy.summary_trades)

    save_results(strategy)
