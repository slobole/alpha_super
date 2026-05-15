"""
BTC residual to UPRO same-close lookahead research strategy.

TL;DR: This module implements the Robot James BTC residual idea as an isolated
research-only strategy. It computes a BTC-vs-QQQ residual z-score from daily
close data and deliberately enters/exits UPRO at that same day's close.

Core formulas
-------------
For trading day t:

    btc_return_t = BTC_close_t / BTC_close_{t-1} - 1
    qqq_return_t = QQQ_close_t / QQQ_close_{t-1} - 1

    beta_t
        = Cov_40(btc_return, qqq_return)_{t-1}
          / Var_40(qqq_return)_{t-1}

    expected_btc_return_t = beta_t * qqq_return_t
    residual_t = btc_return_t - expected_btc_return_t
    residual_zscore_t = residual_t / Std_20(residual)_{t-1}

Lookahead execution rule
------------------------
If residual_zscore_t >= 1.5, hold a fixed-notional long UPRO position. If the
signal is below threshold or missing, hold cash. Fills are modeled at UPRO's
same-day close.

Execution caveat
----------------
This is intentionally not live-clean:

    signal_t = f(Close_t)
    fill_t = Close_t

With daily bars, the final close is only known after the close. This file is a
biased research diagnostic and must not be treated as a live-tradable backtest.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
import yfinance as yf
from IPython.display import display

from alpha.engine.report import save_results
from alpha.engine.strategy import Strategy
from strategies.eom_tlt_vs_spy.strategy_eom_trend_ibit import (
    build_results_df,
    compute_commission_float,
)


def default_trade_id_int() -> int:
    return -1


@dataclass(frozen=True)
class BtcResidualUproLookaheadConfig:
    btc_symbol_str: str = "BTC-USD"
    risk_symbol_str: str = "QQQ"
    trade_symbol_str: str = "UPRO"
    benchmark_list: tuple[str, ...] = ("SPY", "UPRO")
    beta_lookback_int: int = 40
    zscore_lookback_int: int = 20
    entry_zscore_float: float = 1.5
    target_notional_float: float = 10_000.0
    download_start_date_str: str = "2014-09-17"
    backtest_start_date_str: str = "2025-01-01"
    end_date_str: str | None = None
    capital_base_float: float = 100_000.0
    slippage_float: float = 0.0001
    commission_per_share_float: float = 0.005
    commission_minimum_float: float = 1.0

    def __post_init__(self):
        if self.beta_lookback_int <= 1:
            raise ValueError("beta_lookback_int must be greater than 1.")
        if self.zscore_lookback_int <= 1:
            raise ValueError("zscore_lookback_int must be greater than 1.")
        if self.target_notional_float <= 0.0:
            raise ValueError("target_notional_float must be positive.")
        if self.capital_base_float <= 0.0:
            raise ValueError("capital_base_float must be positive.")
        if self.slippage_float < 0.0:
            raise ValueError("slippage_float must be non-negative.")
        if self.commission_per_share_float < 0.0:
            raise ValueError("commission_per_share_float must be non-negative.")
        if self.commission_minimum_float < 0.0:
            raise ValueError("commission_minimum_float must be non-negative.")


DEFAULT_CONFIG = BtcResidualUproLookaheadConfig()


def load_yahoo_ohlcv_df(
    symbol_str: str,
    start_date_str: str,
    end_date_str: str | None = None,
) -> pd.DataFrame:
    price_df = yf.download(
        symbol_str,
        start=start_date_str,
        end=end_date_str,
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    if len(price_df) == 0:
        return pd.DataFrame()

    if isinstance(price_df.columns, pd.MultiIndex):
        if symbol_str in price_df.columns.get_level_values(-1):
            price_df = price_df.xs(symbol_str, axis=1, level=-1)
        elif symbol_str in price_df.columns.get_level_values(0):
            price_df = price_df.xs(symbol_str, axis=1, level=0)
        else:
            price_df = price_df.droplevel(-1, axis=1)

    price_df = price_df.copy()
    price_df.index = pd.to_datetime(price_df.index)
    if getattr(price_df.index, "tz", None) is not None:
        price_df.index = price_df.index.tz_localize(None)

    required_field_list = ["Open", "High", "Low", "Close"]
    missing_field_list = [
        field_str for field_str in required_field_list if field_str not in price_df.columns
    ]
    if len(missing_field_list) > 0:
        raise RuntimeError(f"{symbol_str} is missing required fields: {missing_field_list}")

    field_order_list = [
        field_str for field_str in ["Open", "High", "Low", "Close", "Volume"]
        if field_str in price_df.columns
    ]
    price_df = price_df[field_order_list].sort_index().dropna(subset=required_field_list, how="any")
    return price_df


def attach_symbol_level(price_df: pd.DataFrame, symbol_str: str) -> pd.DataFrame:
    labeled_price_df = price_df.copy()
    labeled_price_df.columns = pd.MultiIndex.from_tuples(
        [(symbol_str, field_str) for field_str in labeled_price_df.columns]
    )
    return labeled_price_df


def load_pricing_data_df(
    btc_symbol_str: str,
    risk_symbol_str: str,
    trade_symbol_str: str,
    benchmark_list: Sequence[str],
    start_date_str: str,
    end_date_str: str | None = None,
) -> pd.DataFrame:
    symbol_list = list(dict.fromkeys([btc_symbol_str, risk_symbol_str, trade_symbol_str, *benchmark_list]))
    price_frame_list: list[pd.DataFrame] = []

    for symbol_str in symbol_list:
        symbol_price_df = load_yahoo_ohlcv_df(
            symbol_str=symbol_str,
            start_date_str=start_date_str,
            end_date_str=end_date_str,
        )
        if len(symbol_price_df) == 0:
            raise RuntimeError(f"{symbol_str} returned no Yahoo rows.")
        price_frame_list.append(attach_symbol_level(symbol_price_df, symbol_str))

    pricing_data_df = pd.concat(price_frame_list, axis=1, join="inner").sort_index()
    required_col_list = [
        (symbol_str, field_str)
        for symbol_str in symbol_list
        for field_str in ("Open", "High", "Low", "Close")
    ]
    pricing_data_df = pricing_data_df.dropna(subset=required_col_list, how="any")
    if len(pricing_data_df) == 0:
        raise RuntimeError("No common aligned Yahoo rows remained after joining symbols.")
    return pricing_data_df


def build_btc_residual_signal_data_df(
    pricing_data_df: pd.DataFrame,
    btc_symbol_str: str = DEFAULT_CONFIG.btc_symbol_str,
    risk_symbol_str: str = DEFAULT_CONFIG.risk_symbol_str,
    beta_lookback_int: int = DEFAULT_CONFIG.beta_lookback_int,
    zscore_lookback_int: int = DEFAULT_CONFIG.zscore_lookback_int,
) -> pd.DataFrame:
    signal_data_df = pricing_data_df.copy()
    close_price_df = signal_data_df.xs("Close", axis=1, level=1).astype(float)

    btc_close_price_ser = close_price_df[btc_symbol_str].astype(float)
    risk_close_price_ser = close_price_df[risk_symbol_str].astype(float)

    # *** CRITICAL*** lookahead-sensitive: this v0 intentionally uses Close_t
    # as signal information. The shift is exactly one bar so returns never use
    # Close_{t+1} or later.
    btc_return_ser = btc_close_price_ser / btc_close_price_ser.shift(1) - 1.0
    risk_return_ser = risk_close_price_ser / risk_close_price_ser.shift(1) - 1.0

    # *** CRITICAL*** beta_t must be fit on returns ending at t-1. Removing the
    # shift would let the current BTC move influence the hedge estimate used to
    # judge that same BTC move.
    rolling_cov_ser = btc_return_ser.rolling(beta_lookback_int).cov(risk_return_ser).shift(1)
    rolling_var_ser = risk_return_ser.rolling(beta_lookback_int).var().shift(1)
    beta_ser = rolling_cov_ser / rolling_var_ser.replace(0.0, np.nan)

    expected_btc_return_ser = beta_ser * risk_return_ser
    residual_ser = btc_return_ser - expected_btc_return_ser

    # *** CRITICAL*** residual_zscore_t must scale by residual volatility known
    # before t. Using the unshifted rolling std would use residual_t to judge
    # whether residual_t is unusual.
    residual_std_ser = residual_ser.rolling(zscore_lookback_int).std().shift(1)
    residual_zscore_ser = residual_ser / residual_std_ser.replace(0.0, np.nan)

    feature_data_df = pd.DataFrame(
        {
            (btc_symbol_str, "btc_return_ser"): btc_return_ser,
            (risk_symbol_str, "risk_return_ser"): risk_return_ser,
            (btc_symbol_str, "beta_ser"): beta_ser,
            (btc_symbol_str, "expected_btc_return_ser"): expected_btc_return_ser,
            (btc_symbol_str, "residual_ser"): residual_ser,
            (btc_symbol_str, "residual_zscore_ser"): residual_zscore_ser,
        },
        index=signal_data_df.index,
    )
    feature_data_df.columns = pd.MultiIndex.from_tuples(feature_data_df.columns)
    return pd.concat([signal_data_df, feature_data_df], axis=1)


class BtcResidualUproLookaheadResearchStrategy(Strategy):
    """
    Research-only same-close BTC residual strategy container.

    `iterate()` is intentionally unused because this module bypasses the
    repository's next-open engine contract and uses a custom same-close fill
    loop.
    """

    enable_signal_audit = True
    signal_audit_sample_size = 10

    def __init__(
        self,
        name: str,
        benchmarks: Sequence[str],
        btc_symbol_str: str = DEFAULT_CONFIG.btc_symbol_str,
        risk_symbol_str: str = DEFAULT_CONFIG.risk_symbol_str,
        trade_symbol_str: str = DEFAULT_CONFIG.trade_symbol_str,
        beta_lookback_int: int = DEFAULT_CONFIG.beta_lookback_int,
        zscore_lookback_int: int = DEFAULT_CONFIG.zscore_lookback_int,
        entry_zscore_float: float = DEFAULT_CONFIG.entry_zscore_float,
        target_notional_float: float = DEFAULT_CONFIG.target_notional_float,
        capital_base: float = DEFAULT_CONFIG.capital_base_float,
        slippage: float = DEFAULT_CONFIG.slippage_float,
        commission_per_share: float = DEFAULT_CONFIG.commission_per_share_float,
        commission_minimum: float = DEFAULT_CONFIG.commission_minimum_float,
    ):
        super().__init__(
            name=name,
            benchmarks=list(benchmarks),
            capital_base=capital_base,
            slippage=slippage,
            commission_per_share=commission_per_share,
            commission_minimum=commission_minimum,
        )
        if beta_lookback_int <= 1:
            raise ValueError("beta_lookback_int must be greater than 1.")
        if zscore_lookback_int <= 1:
            raise ValueError("zscore_lookback_int must be greater than 1.")
        if target_notional_float <= 0.0:
            raise ValueError("target_notional_float must be positive.")

        self.btc_symbol_str = str(btc_symbol_str)
        self.risk_symbol_str = str(risk_symbol_str)
        self.trade_symbol_str = str(trade_symbol_str)
        self.beta_lookback_int = int(beta_lookback_int)
        self.zscore_lookback_int = int(zscore_lookback_int)
        self.entry_zscore_float = float(entry_zscore_float)
        self.target_notional_float = float(target_notional_float)
        self.signal_data_df = pd.DataFrame(dtype=float)
        self.daily_target_weights = pd.DataFrame(dtype=float)

    def compute_signals(self, pricing_data_df: pd.DataFrame) -> pd.DataFrame:
        return build_btc_residual_signal_data_df(
            pricing_data_df=pricing_data_df,
            btc_symbol_str=self.btc_symbol_str,
            risk_symbol_str=self.risk_symbol_str,
            beta_lookback_int=self.beta_lookback_int,
            zscore_lookback_int=self.zscore_lookback_int,
        )

    def iterate(
        self,
        data_df: pd.DataFrame,
        close_row_ser: pd.Series,
        open_price_ser: pd.Series,
    ):
        return


def _fit_entry_share_count_int(
    target_notional_float: float,
    cash_value_float: float,
    entry_fill_price_float: float,
    commission_per_share_float: float,
    commission_minimum_float: float,
) -> int:
    if not np.isfinite(entry_fill_price_float) or entry_fill_price_float <= 0.0:
        return 0

    target_share_count_int = int(target_notional_float / entry_fill_price_float)
    max_affordable_share_count_int = int(cash_value_float / entry_fill_price_float)
    entry_share_count_int = min(target_share_count_int, max_affordable_share_count_int)

    while entry_share_count_int > 0:
        entry_commission_float = compute_commission_float(
            share_count_int=entry_share_count_int,
            commission_per_share_float=commission_per_share_float,
            commission_minimum_float=commission_minimum_float,
        )
        total_cash_needed_float = float(entry_share_count_int) * entry_fill_price_float + entry_commission_float
        if total_cash_needed_float <= cash_value_float + 1e-12:
            return int(entry_share_count_int)
        entry_share_count_int -= 1

    return 0


def run_btc_residual_upro_lookahead_backtest(
    strategy: BtcResidualUproLookaheadResearchStrategy,
    pricing_data_df: pd.DataFrame,
    signal_data_df: pd.DataFrame | None = None,
    backtest_start_date_str: str | None = DEFAULT_CONFIG.backtest_start_date_str,
) -> BtcResidualUproLookaheadResearchStrategy:
    if signal_data_df is None:
        signal_data_df = strategy.compute_signals(pricing_data_df)

    signal_data_df = signal_data_df.sort_index().copy()
    if backtest_start_date_str is not None:
        signal_data_df = signal_data_df.loc[signal_data_df.index >= pd.Timestamp(backtest_start_date_str)].copy()

    trading_index = pd.DatetimeIndex(signal_data_df.index)
    if len(trading_index) == 0:
        raise RuntimeError("signal_data_df must contain at least one backtest bar.")

    close_price_df = signal_data_df.xs("Close", axis=1, level=1).astype(float)
    trade_close_price_ser = close_price_df[strategy.trade_symbol_str].astype(float)
    residual_zscore_ser = signal_data_df[
        (strategy.btc_symbol_str, "residual_zscore_ser")
    ].astype(float)

    transaction_row_list: list[dict[str, object]] = []
    daily_target_weight_row_list: list[dict[str, float | pd.Timestamp]] = []
    cash_value_float = float(strategy._capital_base)
    share_count_int = 0
    active_trade_id_int = default_trade_id_int()
    next_trade_id_int = 0

    portfolio_value_map: dict[pd.Timestamp, float] = {}
    cash_value_map: dict[pd.Timestamp, float] = {}
    total_value_map: dict[pd.Timestamp, float] = {}

    for bar_ts in trading_index:
        bar_ts = pd.Timestamp(bar_ts)
        close_price_float = float(trade_close_price_ser.loc[bar_ts])
        if not np.isfinite(close_price_float) or close_price_float <= 0.0:
            raise RuntimeError(f"Invalid close price for {strategy.trade_symbol_str} on {bar_ts}: {close_price_float}")

        zscore_float = float(residual_zscore_ser.loc[bar_ts])
        hold_signal_bool = bool(np.isfinite(zscore_float) and zscore_float >= strategy.entry_zscore_float)

        if share_count_int > 0 and not hold_signal_bool:
            # *** CRITICAL*** intentional lookahead diagnostic: the exit signal
            # uses Close_t-derived zscore_t and fills at that same Close_t.
            exit_fill_price_float = float(close_price_float * (1.0 - float(strategy._slippage)))
            exit_commission_float = compute_commission_float(
                share_count_int=share_count_int,
                commission_per_share_float=float(strategy._commission_per_share),
                commission_minimum_float=float(strategy._commission_minimum),
            )
            cash_value_float += float(share_count_int) * exit_fill_price_float
            cash_value_float -= exit_commission_float
            transaction_row_list.append(
                {
                    "trade_id": int(active_trade_id_int),
                    "bar": bar_ts,
                    "asset": strategy.trade_symbol_str,
                    "amount": -int(share_count_int),
                    "price": float(exit_fill_price_float),
                    "total_value": float(-share_count_int * exit_fill_price_float),
                    "order_id": len(transaction_row_list) + 1,
                    "commission": float(exit_commission_float),
                }
            )
            share_count_int = 0
            active_trade_id_int = default_trade_id_int()

        if share_count_int == 0 and hold_signal_bool:
            # *** CRITICAL*** intentional lookahead diagnostic: the entry signal
            # uses Close_t-derived zscore_t and fills at that same Close_t.
            entry_fill_price_float = float(close_price_float * (1.0 + float(strategy._slippage)))
            entry_share_count_int = _fit_entry_share_count_int(
                target_notional_float=float(strategy.target_notional_float),
                cash_value_float=float(cash_value_float),
                entry_fill_price_float=entry_fill_price_float,
                commission_per_share_float=float(strategy._commission_per_share),
                commission_minimum_float=float(strategy._commission_minimum),
            )
            if entry_share_count_int > 0:
                entry_commission_float = compute_commission_float(
                    share_count_int=entry_share_count_int,
                    commission_per_share_float=float(strategy._commission_per_share),
                    commission_minimum_float=float(strategy._commission_minimum),
                )
                next_trade_id_int += 1
                active_trade_id_int = int(next_trade_id_int)
                cash_value_float -= float(entry_share_count_int) * entry_fill_price_float
                cash_value_float -= entry_commission_float
                share_count_int = int(entry_share_count_int)
                transaction_row_list.append(
                    {
                        "trade_id": int(active_trade_id_int),
                        "bar": bar_ts,
                        "asset": strategy.trade_symbol_str,
                        "amount": int(entry_share_count_int),
                        "price": float(entry_fill_price_float),
                        "total_value": float(entry_share_count_int * entry_fill_price_float),
                        "order_id": len(transaction_row_list) + 1,
                        "commission": float(entry_commission_float),
                    }
                )

        portfolio_value_float = float(share_count_int * close_price_float)
        total_value_float = float(cash_value_float + portfolio_value_float)

        portfolio_value_map[bar_ts] = portfolio_value_float
        cash_value_map[bar_ts] = cash_value_float
        total_value_map[bar_ts] = total_value_float

        target_weight_row_map: dict[str, float | pd.Timestamp] = {"bar_ts": bar_ts}
        if total_value_float > 0.0:
            target_weight_row_map[strategy.trade_symbol_str] = portfolio_value_float / total_value_float
        else:
            target_weight_row_map[strategy.trade_symbol_str] = 0.0
        daily_target_weight_row_list.append(target_weight_row_map)

    portfolio_value_ser = pd.Series(portfolio_value_map, dtype=float).sort_index()
    cash_ser = pd.Series(cash_value_map, dtype=float).sort_index()
    total_value_ser = pd.Series(total_value_map, dtype=float).sort_index()

    benchmark_equity_map: dict[str, pd.Series] = {}
    for benchmark_str in strategy._benchmarks:
        if benchmark_str not in close_price_df.columns:
            continue
        benchmark_close_ser = close_price_df.loc[trading_index, benchmark_str].astype(float)
        benchmark_start_close_float = float(benchmark_close_ser.iloc[0])
        if not np.isfinite(benchmark_start_close_float) or benchmark_start_close_float <= 0.0:
            continue
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
    strategy._position_amount_map = (
        {strategy.trade_symbol_str: float(share_count_int)}
        if share_count_int != 0
        else {}
    )
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
    strategy.summarize()
    return strategy


def run_variant(
    show_display_bool: bool = True,
    save_results_bool: bool = True,
    output_dir_str: str = "results",
    pricing_data_df: pd.DataFrame | None = None,
    btc_symbol_str: str = DEFAULT_CONFIG.btc_symbol_str,
    risk_symbol_str: str = DEFAULT_CONFIG.risk_symbol_str,
    trade_symbol_str: str = DEFAULT_CONFIG.trade_symbol_str,
    benchmark_list: Sequence[str] = DEFAULT_CONFIG.benchmark_list,
    beta_lookback_int: int = DEFAULT_CONFIG.beta_lookback_int,
    zscore_lookback_int: int = DEFAULT_CONFIG.zscore_lookback_int,
    entry_zscore_float: float = DEFAULT_CONFIG.entry_zscore_float,
    target_notional_float: float = DEFAULT_CONFIG.target_notional_float,
    download_start_date_str: str = DEFAULT_CONFIG.download_start_date_str,
    backtest_start_date_str: str = DEFAULT_CONFIG.backtest_start_date_str,
    end_date_str: str | None = DEFAULT_CONFIG.end_date_str,
    capital_base_float: float = DEFAULT_CONFIG.capital_base_float,
    slippage_float: float = DEFAULT_CONFIG.slippage_float,
    commission_per_share_float: float = DEFAULT_CONFIG.commission_per_share_float,
    commission_minimum_float: float = DEFAULT_CONFIG.commission_minimum_float,
) -> BtcResidualUproLookaheadResearchStrategy:
    if pricing_data_df is None:
        pricing_data_df = load_pricing_data_df(
            btc_symbol_str=btc_symbol_str,
            risk_symbol_str=risk_symbol_str,
            trade_symbol_str=trade_symbol_str,
            benchmark_list=benchmark_list,
            start_date_str=download_start_date_str,
            end_date_str=end_date_str,
        )

    strategy = BtcResidualUproLookaheadResearchStrategy(
        name="strategy_btc_residual_upro_lookahead",
        benchmarks=list(benchmark_list),
        btc_symbol_str=btc_symbol_str,
        risk_symbol_str=risk_symbol_str,
        trade_symbol_str=trade_symbol_str,
        beta_lookback_int=beta_lookback_int,
        zscore_lookback_int=zscore_lookback_int,
        entry_zscore_float=entry_zscore_float,
        target_notional_float=target_notional_float,
        capital_base=capital_base_float,
        slippage=slippage_float,
        commission_per_share=commission_per_share_float,
        commission_minimum=commission_minimum_float,
    )

    signal_data_df = strategy.compute_signals(pricing_data_df)
    run_btc_residual_upro_lookahead_backtest(
        strategy=strategy,
        pricing_data_df=pricing_data_df,
        signal_data_df=signal_data_df,
        backtest_start_date_str=backtest_start_date_str,
    )

    if show_display_bool:
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 1000)
        display(strategy.summary)
        display(strategy.summary_trades)

    if save_results_bool:
        save_results(strategy, output_dir=output_dir_str)

    return strategy


if __name__ == "__main__":
    run_variant()
