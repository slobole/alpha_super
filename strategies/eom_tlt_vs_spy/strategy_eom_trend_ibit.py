"""
IBIT EOM trend strategy with last-day-close exit.

TL;DR: For each completed month, compute the return over the first 15 trading
days. If that return is positive, buy a Bitcoin ETF at the open of the first
trading day inside the final 5-trading-day window and sell at the close of the
last trading day of the month.

Core formulas
-------------
Let the ordered trading dates in completed month m be:

    D_m = {d_{m,1}, d_{m,2}, ..., d_{m,N_m}}

Define:

    L = 15
    H = 5

Early-month trend signal:

    first_15_return_m
        = Close_{d_{m,L}} / Open_{d_{m,1}} - 1

Entry / exit bars:

    entry_bar_m = d_{m, N_m - H + 1}
    exit_bar_m  = d_{m, N_m}

Eligibility rule:

    eligible_m = 1[first_15_return_m > 0]

Trade return for eligible months:

    trade_return_m
        = Close_{exit_bar_m} / Open_{entry_bar_m} - 1

Causality guard
---------------
To avoid signal/entry overlap, the month must satisfy:

    N_m >= L + H

Otherwise the month is skipped.

Execution-model note
--------------------
Old engine contract:

    orders from iterate() fill at the next tradable open

New behavior in this file:

    entry fill = Open_{entry_bar_m}
    exit fill  = Close_{exit_bar_m}

Quantitative consequence:

    trade_return_m^{close-exit}
        = Close_{exit_bar_m} / Open_{entry_bar_m} - 1

instead of:

    trade_return_m^{next-open-exit}
        = Open_{next_month_start_m} / Open_{entry_bar_m} - 1
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Sequence

import numpy as np
import pandas as pd
import yfinance as yf
from IPython.display import display

WORKSPACE_ROOT_PATH = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT_PATH))

from alpha.engine.report import save_results
from alpha.engine.strategy import Strategy


def default_trade_id_int() -> int:
    return -1


@dataclass(frozen=True)
class EomTrendIbitConfig:
    trade_symbol_candidate_list: tuple[str, ...] = ("IBIT", "FBTC", "BITB", "ARKB", "BITO")
    benchmark_list: tuple[str, ...] = ("SPY",)
    signal_day_count_int: int = 15
    hold_day_count_int: int = 5
    start_date_str: str = "2024-01-01"
    end_date_str: str | None = None
    capital_base_float: float = 100_000.0
    slippage_float: float = 0.0001
    commission_per_share_float: float = 0.005
    commission_minimum_float: float = 1.0

    def __post_init__(self):
        if len(self.trade_symbol_candidate_list) == 0:
            raise ValueError("trade_symbol_candidate_list must contain at least one symbol.")
        if self.signal_day_count_int <= 0:
            raise ValueError("signal_day_count_int must be positive.")
        if self.hold_day_count_int <= 0:
            raise ValueError("hold_day_count_int must be positive.")
        if self.capital_base_float <= 0.0:
            raise ValueError("capital_base_float must be positive.")
        if self.slippage_float < 0.0:
            raise ValueError("slippage_float must be non-negative.")
        if self.commission_per_share_float < 0.0:
            raise ValueError("commission_per_share_float must be non-negative.")
        if self.commission_minimum_float < 0.0:
            raise ValueError("commission_minimum_float must be non-negative.")


DEFAULT_CONFIG = EomTrendIbitConfig()


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
        else:
            price_df = price_df.droplevel(-1, axis=1)

    price_df = price_df.copy()
    price_df.index = pd.to_datetime(price_df.index)
    if getattr(price_df.index, "tz", None) is not None:
        price_df.index = price_df.index.tz_localize(None)

    required_field_list = ["Open", "High", "Low", "Close"]
    missing_field_list = [field_str for field_str in required_field_list if field_str not in price_df.columns]
    if len(missing_field_list) > 0:
        raise RuntimeError(f"{symbol_str} is missing required fields: {missing_field_list}")

    field_order_list = [field_str for field_str in ["Open", "High", "Low", "Close", "Volume"] if field_str in price_df.columns]
    price_df = price_df[field_order_list].sort_index().dropna(subset=required_field_list, how="any")
    return price_df


def attach_symbol_level(price_df: pd.DataFrame, symbol_str: str) -> pd.DataFrame:
    field_list = list(price_df.columns)
    labeled_price_df = price_df.copy()
    labeled_price_df.columns = pd.MultiIndex.from_tuples([(symbol_str, field_str) for field_str in field_list])
    return labeled_price_df


def load_pricing_data_df(
    trade_symbol_candidate_list: Sequence[str],
    benchmark_list: Sequence[str],
    start_date_str: str,
    end_date_str: str | None = None,
) -> tuple[pd.DataFrame, str]:
    selection_error_list: list[str] = []
    selected_symbol_str: str | None = None
    trade_price_df: pd.DataFrame | None = None

    for candidate_symbol_str in trade_symbol_candidate_list:
        try:
            candidate_price_df = load_yahoo_ohlcv_df(
                symbol_str=candidate_symbol_str,
                start_date_str=start_date_str,
                end_date_str=end_date_str,
            )
        except Exception as exc:
            selection_error_list.append(f"{candidate_symbol_str}: {exc}")
            continue

        if len(candidate_price_df) == 0:
            selection_error_list.append(f"{candidate_symbol_str}: no rows returned")
            continue

        selected_symbol_str = candidate_symbol_str
        trade_price_df = candidate_price_df
        break

    if selected_symbol_str is None or trade_price_df is None:
        error_message_str = "; ".join(selection_error_list)
        raise RuntimeError(f"Unable to load any trade symbol candidate. Details: {error_message_str}")

    price_frame_list: list[pd.DataFrame] = [attach_symbol_level(trade_price_df, selected_symbol_str)]
    for benchmark_symbol_str in benchmark_list:
        benchmark_price_df = load_yahoo_ohlcv_df(
            symbol_str=benchmark_symbol_str,
            start_date_str=start_date_str,
            end_date_str=end_date_str,
        )
        if len(benchmark_price_df) == 0:
            raise RuntimeError(f"Benchmark {benchmark_symbol_str} returned no data.")
        benchmark_price_df = benchmark_price_df.reindex(trade_price_df.index)
        price_frame_list.append(attach_symbol_level(benchmark_price_df, benchmark_symbol_str))

    pricing_data_df = pd.concat(price_frame_list, axis=1).sort_index()
    pricing_data_df = pricing_data_df.loc[trade_price_df.index]
    return pricing_data_df, selected_symbol_str


def build_eom_trend_trade_plan_df(
    open_price_ser: pd.Series,
    close_price_ser: pd.Series,
    signal_day_count_int: int,
    hold_day_count_int: int,
) -> pd.DataFrame:
    """
    Build a month-by-month trade plan for the EOM trend rule.
    """
    if signal_day_count_int <= 0:
        raise ValueError("signal_day_count_int must be positive.")
    if hold_day_count_int <= 0:
        raise ValueError("hold_day_count_int must be positive.")

    ordered_index = pd.DatetimeIndex(pd.to_datetime(close_price_ser.index)).sort_values()
    open_price_ser = open_price_ser.reindex(ordered_index).astype(float)
    close_price_ser = close_price_ser.reindex(ordered_index).astype(float)

    trade_plan_row_list: list[dict[str, object]] = []
    month_period_index = ordered_index.to_period("M")
    month_group_map = pd.Series(ordered_index, index=ordered_index).groupby(month_period_index).groups

    last_month_period = ordered_index[-1].to_period("M")
    for month_period, month_date_index in month_group_map.items():
        # *** CRITICAL*** The final observed month is dropped unless later data
        # exist, so a partial month cannot be misclassified as completed.
        if month_period == last_month_period:
            continue

        month_trading_index = pd.DatetimeIndex(month_date_index).sort_values()
        month_observation_count_int = len(month_trading_index)

        # *** CRITICAL*** If:
        #   month_observation_count_int < signal_day_count_int + hold_day_count_int
        # then the signal window would overlap the entry bar, creating
        # same-bar look-ahead bias. Such months are skipped entirely.
        if month_observation_count_int < signal_day_count_int + hold_day_count_int:
            continue

        month_start_bar_ts = pd.Timestamp(month_trading_index[0])
        signal_end_bar_ts = pd.Timestamp(month_trading_index[signal_day_count_int - 1])
        entry_bar_ts = pd.Timestamp(month_trading_index[month_observation_count_int - hold_day_count_int])
        exit_bar_ts = pd.Timestamp(month_trading_index[-1])

        if entry_bar_ts <= signal_end_bar_ts:
            continue

        signal_start_open_float = float(open_price_ser.loc[month_start_bar_ts])
        signal_end_close_float = float(close_price_ser.loc[signal_end_bar_ts])

        if not np.isfinite(signal_start_open_float) or signal_start_open_float <= 0.0:
            raise RuntimeError(f"Invalid month_start open on {month_start_bar_ts}: {signal_start_open_float}")
        if not np.isfinite(signal_end_close_float) or signal_end_close_float <= 0.0:
            raise RuntimeError(f"Invalid signal_end close on {signal_end_bar_ts}: {signal_end_close_float}")

        first_15_return_float = float(signal_end_close_float / signal_start_open_float - 1.0)
        eligible_bool = bool(first_15_return_float > 0.0)

        trade_plan_row_list.append(
            {
                "month_period": month_period,
                "month_observation_count_int": int(month_observation_count_int),
                "month_start_bar_ts": month_start_bar_ts,
                "signal_end_bar_ts": signal_end_bar_ts,
                "entry_bar_ts": entry_bar_ts,
                "exit_bar_ts": exit_bar_ts,
                "first_15_return_float": first_15_return_float,
                "eligible_bool": eligible_bool,
            }
        )

    trade_plan_df = pd.DataFrame(trade_plan_row_list)
    if len(trade_plan_df) == 0:
        return pd.DataFrame(
            columns=[
                "month_period",
                "month_observation_count_int",
                "month_start_bar_ts",
                "signal_end_bar_ts",
                "entry_bar_ts",
                "exit_bar_ts",
                "first_15_return_float",
                "eligible_bool",
            ]
        )

    trade_plan_df["exit_bar_ts"] = pd.to_datetime(trade_plan_df["exit_bar_ts"])
    trade_plan_df = trade_plan_df.sort_values("exit_bar_ts").set_index("exit_bar_ts")
    trade_plan_df.index.name = "exit_bar_ts"
    return trade_plan_df


def build_daily_target_weight_ser(
    trading_index: pd.DatetimeIndex,
    trade_plan_df: pd.DataFrame,
) -> pd.Series:
    target_weight_ser = pd.Series(0.0, index=pd.DatetimeIndex(trading_index), dtype=float)
    if len(trade_plan_df) == 0:
        return target_weight_ser

    eligible_trade_plan_df = trade_plan_df.loc[trade_plan_df["eligible_bool"]].copy()
    for _, trade_row_ser in eligible_trade_plan_df.iterrows():
        entry_bar_ts = pd.Timestamp(trade_row_ser["entry_bar_ts"])
        exit_bar_ts = pd.Timestamp(trade_row_ser.name)
        target_weight_ser.loc[entry_bar_ts:exit_bar_ts] = 1.0

    return target_weight_ser


class EomTrendIbitCloseResearchStrategy(Strategy):
    """
    Research-only monthly close-exit strategy container.

    `iterate()` is intentionally unused because the repository engine executes
    at the next open, while this study exits at the month's last close.
    """

    enable_signal_audit = False
    signal_audit_sample_size = 10

    def __init__(
        self,
        name: str,
        benchmarks: Sequence[str],
        trade_symbol_str: str,
        trade_plan_df: pd.DataFrame,
        daily_target_weight_ser: pd.Series,
        capital_base: float = 100_000.0,
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
        self.trade_symbol_str = str(trade_symbol_str)
        self.trade_plan_df = trade_plan_df.copy()
        self.daily_target_weight_ser = daily_target_weight_ser.astype(float).copy().sort_index()

    def compute_signals(self, pricing_data_df: pd.DataFrame) -> pd.DataFrame:
        return pricing_data_df

    def iterate(self, data_df: pd.DataFrame, close_row_ser: pd.Series, open_price_ser: pd.Series):
        return


def compute_commission_float(
    share_count_int: int,
    commission_per_share_float: float,
    commission_minimum_float: float,
) -> float:
    if share_count_int == 0 or commission_per_share_float == 0.0:
        return 0.0
    return float(max(commission_minimum_float, commission_per_share_float * abs(share_count_int)))


def build_results_df(
    total_value_ser: pd.Series,
    portfolio_value_ser: pd.Series,
    cash_ser: pd.Series,
    benchmark_equity_map: dict[str, pd.Series],
) -> pd.DataFrame:
    results_df = pd.DataFrame(index=total_value_ser.index)
    results_df["portfolio_value"] = portfolio_value_ser.astype(float)
    results_df["cash"] = cash_ser.astype(float)
    results_df["total_value"] = total_value_ser.astype(float)

    daily_return_ser = total_value_ser.pct_change(fill_method=None).fillna(0.0).astype(float)
    results_df["daily_returns"] = daily_return_ser
    results_df["total_returns"] = total_value_ser.astype(float) / float(total_value_ser.iloc[0]) - 1.0

    elapsed_day_count_ser = pd.Series(
        np.arange(1, len(total_value_ser.index) + 1, dtype=float),
        index=total_value_ser.index,
        dtype=float,
    )
    results_df["annualized_returns"] = (
        (total_value_ser.astype(float) / float(total_value_ser.iloc[0])) ** (252.0 / elapsed_day_count_ser) - 1.0
    )

    annualized_volatility_ser = daily_return_ser.expanding(min_periods=2).std(ddof=1) * np.sqrt(252.0)
    results_df["annualized_volatility"] = annualized_volatility_ser.astype(float)

    active_return_mask_ser = (portfolio_value_ser.astype(float) != 0.0) | (daily_return_ser != 0.0)
    sharpe_value_list: list[float] = []
    for end_idx_int in range(len(daily_return_ser)):
        sample_return_ser = daily_return_ser.iloc[:end_idx_int + 1].loc[active_return_mask_ser.iloc[:end_idx_int + 1]]
        if len(sample_return_ser) < 2:
            sharpe_value_list.append(np.nan)
            continue

        sample_std_float = float(sample_return_ser.std(ddof=1))
        if sample_std_float == 0.0 or not np.isfinite(sample_std_float):
            sharpe_value_list.append(np.nan)
            continue

        sample_mean_float = float(sample_return_ser.mean())
        sharpe_value_list.append(float(sample_mean_float / sample_std_float * np.sqrt(252.0)))

    results_df["sharpe_ratio"] = pd.Series(sharpe_value_list, index=total_value_ser.index, dtype=float)

    drawdown_ser = total_value_ser.astype(float) / total_value_ser.astype(float).cummax() - 1.0
    results_df["drawdown"] = drawdown_ser.astype(float)
    results_df["max_drawdown"] = drawdown_ser.cummin().astype(float)

    for benchmark_str, benchmark_equity_ser in benchmark_equity_map.items():
        benchmark_equity_ser = benchmark_equity_ser.astype(float)
        benchmark_drawdown_ser = benchmark_equity_ser / benchmark_equity_ser.cummax() - 1.0
        results_df[benchmark_str] = benchmark_equity_ser
        results_df[f"{benchmark_str}_drawdown"] = benchmark_drawdown_ser.astype(float)
        results_df[f"{benchmark_str}_max_drawdown"] = benchmark_drawdown_ser.cummin().astype(float)

    return results_df


def run_eom_trend_close_research_backtest(
    strategy: EomTrendIbitCloseResearchStrategy,
    pricing_data_df: pd.DataFrame,
) -> EomTrendIbitCloseResearchStrategy:
    """
    Run the monthly close-exit research backtest and populate the strategy
    object with transactions, daily results, and summary tables.
    """
    open_price_ser = pricing_data_df[(strategy.trade_symbol_str, "Open")].astype(float)
    close_price_ser = pricing_data_df[(strategy.trade_symbol_str, "Close")].astype(float)
    trading_index = pd.DatetimeIndex(pricing_data_df.index)

    eligible_trade_plan_df = strategy.trade_plan_df.loc[strategy.trade_plan_df["eligible_bool"]].copy()
    entry_trade_id_map = {
        pd.Timestamp(trade_row_ser["entry_bar_ts"]): trade_id_int
        for trade_id_int, (_, trade_row_ser) in enumerate(eligible_trade_plan_df.iterrows(), start=1)
    }
    exit_trade_id_map = {
        pd.Timestamp(exit_bar_ts): trade_id_int
        for trade_id_int, (exit_bar_ts, _) in enumerate(eligible_trade_plan_df.iterrows(), start=1)
    }

    transaction_row_list: list[dict[str, object]] = []
    cash_value_float = float(strategy._capital_base)
    share_count_int = 0
    active_trade_id_int = default_trade_id_int()

    portfolio_value_map: dict[pd.Timestamp, float] = {}
    cash_value_map: dict[pd.Timestamp, float] = {}
    total_value_map: dict[pd.Timestamp, float] = {}

    for bar_ts in trading_index:
        bar_ts = pd.Timestamp(bar_ts)
        open_price_float = float(open_price_ser.loc[bar_ts])
        close_price_float = float(close_price_ser.loc[bar_ts])

        if bar_ts in entry_trade_id_map:
            if share_count_int != 0:
                raise RuntimeError(f"Expected flat state before entry on {bar_ts}, found {share_count_int} shares.")

            active_trade_id_int = int(entry_trade_id_map[bar_ts])
            entry_fill_price_float = float(open_price_float * (1.0 + float(strategy._slippage)))
            entry_share_count_int = int(cash_value_float / entry_fill_price_float)
            entry_commission_float = compute_commission_float(
                share_count_int=entry_share_count_int,
                commission_per_share_float=float(strategy._commission_per_share),
                commission_minimum_float=float(strategy._commission_minimum),
            )

            if entry_share_count_int > 0:
                cash_value_float -= float(entry_share_count_int * entry_fill_price_float)
                cash_value_float -= float(entry_commission_float)
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
            else:
                active_trade_id_int = default_trade_id_int()

        if bar_ts in exit_trade_id_map:
            expected_trade_id_int = int(exit_trade_id_map[bar_ts])
            if share_count_int == 0:
                raise RuntimeError(f"Expected open position before close exit on {bar_ts}, found zero shares.")
            if active_trade_id_int != expected_trade_id_int:
                raise RuntimeError(
                    f"Active trade id mismatch on {bar_ts}. "
                    f"Expected {expected_trade_id_int}, found {active_trade_id_int}."
                )

            exit_fill_price_float = float(close_price_float * (1.0 - float(strategy._slippage)))
            exit_commission_float = compute_commission_float(
                share_count_int=share_count_int,
                commission_per_share_float=float(strategy._commission_per_share),
                commission_minimum_float=float(strategy._commission_minimum),
            )
            exit_share_count_int = int(share_count_int)

            cash_value_float += float(exit_share_count_int * exit_fill_price_float)
            cash_value_float -= float(exit_commission_float)

            transaction_row_list.append(
                {
                    "trade_id": int(active_trade_id_int),
                    "bar": bar_ts,
                    "asset": strategy.trade_symbol_str,
                    "amount": -int(exit_share_count_int),
                    "price": float(exit_fill_price_float),
                    "total_value": float(-exit_share_count_int * exit_fill_price_float),
                    "order_id": len(transaction_row_list) + 1,
                    "commission": float(exit_commission_float),
                }
            )

            share_count_int = 0
            active_trade_id_int = default_trade_id_int()

        portfolio_value_float = float(share_count_int * close_price_float)
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
        benchmark_equity_map[benchmark_str] = benchmark_close_ser / benchmark_start_close_float * float(strategy._capital_base)

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
    strategy._position_amount_map = {}
    strategy.cash = float(cash_ser.iloc[-1])
    strategy.portfolio_value = float(portfolio_value_ser.iloc[-1])
    strategy.total_value = float(total_value_ser.iloc[-1])
    strategy.summarize()
    return strategy


if __name__ == "__main__":
    config = DEFAULT_CONFIG

    pricing_data_df, trade_symbol_str = load_pricing_data_df(
        trade_symbol_candidate_list=config.trade_symbol_candidate_list,
        benchmark_list=config.benchmark_list,
        start_date_str=config.start_date_str,
        end_date_str=config.end_date_str,
    )

    open_price_ser = pricing_data_df[(trade_symbol_str, "Open")].astype(float)
    close_price_ser = pricing_data_df[(trade_symbol_str, "Close")].astype(float)

    trade_plan_df = build_eom_trend_trade_plan_df(
        open_price_ser=open_price_ser,
        close_price_ser=close_price_ser,
        signal_day_count_int=config.signal_day_count_int,
        hold_day_count_int=config.hold_day_count_int,
    )
    daily_target_weight_ser = build_daily_target_weight_ser(
        trading_index=pricing_data_df.index,
        trade_plan_df=trade_plan_df,
    )

    strategy = EomTrendIbitCloseResearchStrategy(
        name="strategy_eom_trend_ibit_research",
        benchmarks=config.benchmark_list,
        trade_symbol_str=trade_symbol_str,
        trade_plan_df=trade_plan_df,
        daily_target_weight_ser=daily_target_weight_ser,
        capital_base=config.capital_base_float,
        slippage=config.slippage_float,
        commission_per_share=config.commission_per_share_float,
        commission_minimum=config.commission_minimum_float,
    )
    strategy.show_taa_weights_report = True
    strategy.daily_target_weights = daily_target_weight_ser.to_frame(name=trade_symbol_str)

    run_eom_trend_close_research_backtest(
        strategy=strategy,
        pricing_data_df=pricing_data_df,
    )

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)

    print(f"Selected trade symbol: {trade_symbol_str}")
    print("Trade plan preview:")
    display(trade_plan_df.head())
    display(strategy.summary)
    display(strategy.summary_trades)

    save_results(strategy)
