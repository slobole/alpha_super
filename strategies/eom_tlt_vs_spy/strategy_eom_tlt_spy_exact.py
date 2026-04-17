"""
SPY / TLT exact end-of-month relative-strength strategy.

TL;DR: This is the exact single-leg interpretation of the article's first
SPY / TLT idea:

1. On trading day 15 of a completed month, compare true month-to-date returns.
2. If SPY outperformed TLT, buy TLT at the next trading day's open.
3. If TLT outperformed SPY, buy SPY at the next trading day's open.
4. Hold that laggard through the last trading day's close of the same month.

Core formulas
-------------
Let the ordered trading dates in month m be:

    D_m = {d_{m,1}, d_{m,2}, ..., d_{m,N_m}}

Let d_{m,0} be the last trading day before month m.

Define:

    L = 15

True month-to-date returns observed after trading day L:

    r_spy_m^{mtd}
        = Close_SPY_{d_{m,L}} / Close_SPY_{d_{m,0}} - 1

    r_tlt_m^{mtd}
        = Close_TLT_{d_{m,L}} / Close_TLT_{d_{m,0}} - 1

Relative spread:

    rel_m = r_spy_m^{mtd} - r_tlt_m^{mtd}

Trade rule:

    if rel_m > 0:
        long TLT from Open_{d_{m,L+1}} to Close_{d_{m,N_m}}

    if rel_m < 0:
        long SPY from Open_{d_{m,L+1}} to Close_{d_{m,N_m}}

No trade when rel_m = 0.

Data note
---------
Tradeable ETFs use Norgate `CAPITALSPECIAL` prices for both signal formation
and execution. Benchmarks use Norgate benchmark-series settings through the
shared loader.

Execution-model note
--------------------
This file is intentionally research-only because the repository's standard
`run_daily()` engine exits at the next open, while this article-faithful path
exits at:

    Close_{d_{m,N_m}}

Quantitative consequence:

    trade_return_m^{close-exit}
        = Close_{selected_asset, d_{m,N_m}} / Open_{selected_asset, d_{m,L+1}} - 1

instead of:

    trade_return_m^{next-open-exit}
        = Open_{selected_asset, d_{m+1,1}} / Open_{selected_asset, d_{m,L+1}} - 1
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

from alpha.engine.report import save_results
from alpha.engine.strategy import Strategy
from data.norgate_loader import load_raw_prices
from strategies.eom_tlt_vs_spy.strategy_eom_trend_ibit import (
    build_results_df,
    compute_commission_float,
)


def default_trade_id_int() -> int:
    return -1


@dataclass(frozen=True)
class EomTltSpyExactConfig:
    trade_symbol_list: tuple[str, ...] = ("SPY", "TLT")
    benchmark_list: tuple[str, ...] = ("$SPX",)
    signal_day_count_int: int = 15
    position_weight_float: float = 1.0
    start_date_str: str = "2002-07-01"
    end_date_str: str | None = None
    capital_base_float: float = 100_000.0
    slippage_float: float = 0.0001
    commission_per_share_float: float = 0.005
    commission_minimum_float: float = 1.0

    def __post_init__(self):
        if set(self.trade_symbol_list) != {"SPY", "TLT"}:
            raise ValueError("trade_symbol_list must contain exactly SPY and TLT.")
        if len(set(self.trade_symbol_list).intersection(set(self.benchmark_list))) > 0:
            raise ValueError(
                "benchmark_list must not overlap trade_symbol_list because "
                "tradeable assets use CAPITALSPECIAL while benchmarks use benchmark settings."
            )
        if self.signal_day_count_int <= 0:
            raise ValueError("signal_day_count_int must be positive.")
        if self.position_weight_float <= 0.0:
            raise ValueError("position_weight_float must be positive.")
        if self.capital_base_float <= 0.0:
            raise ValueError("capital_base_float must be positive.")
        if self.slippage_float < 0.0:
            raise ValueError("slippage_float must be non-negative.")
        if self.commission_per_share_float < 0.0:
            raise ValueError("commission_per_share_float must be non-negative.")
        if self.commission_minimum_float < 0.0:
            raise ValueError("commission_minimum_float must be non-negative.")


DEFAULT_CONFIG = EomTltSpyExactConfig()


def load_pricing_data_df(
    trade_symbol_list: Sequence[str],
    benchmark_list: Sequence[str],
    start_date_str: str,
    end_date_str: str | None = None,
) -> pd.DataFrame:
    pricing_data_df = load_raw_prices(
        symbols=list(trade_symbol_list),
        benchmarks=list(benchmark_list),
        start_date=start_date_str,
        end_date=end_date_str,
    )
    if len(pricing_data_df) == 0:
        raise RuntimeError("Norgate returned no rows for the requested symbols.")
    pricing_data_df = pricing_data_df.sort_index()

    required_field_list = ["Open", "High", "Low", "Close"]
    trade_required_column_list = [
        (symbol_str, field_str)
        for symbol_str in trade_symbol_list
        for field_str in required_field_list
    ]

    # *** CRITICAL*** This strategy requires a synchronized SPY/TLT calendar.
    # Any row where either tradable ETF lacks OHLC data would create invalid
    # month-to-date comparisons and can also leak NaNs into portfolio value
    # paths. Such rows are removed from the research sample.
    valid_trade_bar_mask_ser = pricing_data_df.loc[:, trade_required_column_list].notna().all(axis=1)
    pricing_data_df = pricing_data_df.loc[valid_trade_bar_mask_ser].copy()
    if len(pricing_data_df) == 0:
        raise RuntimeError("No synchronized SPY/TLT bars remain after dropping incomplete trade rows.")

    return pricing_data_df


def get_completed_month_period_set(trading_index: pd.DatetimeIndex) -> set[pd.Period]:
    if len(trading_index) == 0:
        return set()

    completed_month_period_set = set(trading_index.to_period("M").unique().tolist())
    last_available_bar_ts = pd.Timestamp(trading_index[-1])
    expected_business_month_end_ts = pd.Timestamp(last_available_bar_ts + pd.offsets.BMonthEnd(0))
    if expected_business_month_end_ts.normalize() != last_available_bar_ts.normalize():
        completed_month_period_set.discard(last_available_bar_ts.to_period("M"))
    return completed_month_period_set


def build_month_signal_df(
    close_price_df: pd.DataFrame,
    config: EomTltSpyExactConfig = DEFAULT_CONFIG,
) -> pd.DataFrame:
    ordered_trading_index = pd.DatetimeIndex(close_price_df.index).sort_values()
    month_period_index = ordered_trading_index.to_period("M")
    month_group_map = pd.Series(ordered_trading_index, index=ordered_trading_index).groupby(month_period_index).groups
    completed_month_period_set = get_completed_month_period_set(ordered_trading_index)

    month_signal_row_list: list[dict[str, object]] = []
    for month_period, month_date_index in month_group_map.items():
        month_trading_index = pd.DatetimeIndex(month_date_index).sort_values()
        month_observation_count_int = len(month_trading_index)
        month_start_bar_ts = pd.Timestamp(month_trading_index[0])

        # *** CRITICAL*** The article signal is evaluated only on completed
        # months. Using an incomplete last month would silently mix partial and
        # completed month semantics.
        if month_period not in completed_month_period_set:
            continue

        # *** CRITICAL*** The rule needs trading day L for the signal and
        # trading day L+1 for the entry. If N_m <= L, the signal would have no
        # strictly later entry bar and the month must be skipped.
        if month_observation_count_int <= config.signal_day_count_int:
            continue

        month_start_position_int = int(ordered_trading_index.get_loc(month_start_bar_ts))
        if month_start_position_int == 0:
            continue

        previous_month_end_bar_ts = pd.Timestamp(ordered_trading_index[month_start_position_int - 1])
        signal_end_bar_ts = pd.Timestamp(month_trading_index[config.signal_day_count_int - 1])
        entry_bar_ts = pd.Timestamp(month_trading_index[config.signal_day_count_int])
        exit_bar_ts = pd.Timestamp(month_trading_index[-1])

        if entry_bar_ts <= signal_end_bar_ts:
            raise RuntimeError(
                f"Invalid signal/entry ordering in {month_period}. "
                f"signal_end_bar_ts={signal_end_bar_ts}, entry_bar_ts={entry_bar_ts}"
            )

        spy_previous_close_float = float(close_price_df.loc[previous_month_end_bar_ts, "SPY"])
        tlt_previous_close_float = float(close_price_df.loc[previous_month_end_bar_ts, "TLT"])
        spy_signal_close_float = float(close_price_df.loc[signal_end_bar_ts, "SPY"])
        tlt_signal_close_float = float(close_price_df.loc[signal_end_bar_ts, "TLT"])

        if not np.isfinite(spy_previous_close_float) or spy_previous_close_float <= 0.0:
            raise RuntimeError(
                f"Invalid SPY previous-month-end close on {previous_month_end_bar_ts}: {spy_previous_close_float}"
            )
        if not np.isfinite(tlt_previous_close_float) or tlt_previous_close_float <= 0.0:
            raise RuntimeError(
                f"Invalid TLT previous-month-end close on {previous_month_end_bar_ts}: {tlt_previous_close_float}"
            )
        if not np.isfinite(spy_signal_close_float) or spy_signal_close_float <= 0.0:
            raise RuntimeError(f"Invalid SPY signal close on {signal_end_bar_ts}: {spy_signal_close_float}")
        if not np.isfinite(tlt_signal_close_float) or tlt_signal_close_float <= 0.0:
            raise RuntimeError(f"Invalid TLT signal close on {signal_end_bar_ts}: {tlt_signal_close_float}")

        # *** CRITICAL*** "Month-to-date" is implemented as:
        #   Close_{d_{m,L}} / Close_{d_{m,0}} - 1
        # where d_{m,0} is the last bar before month m starts. Replacing this
        # denominator with Open_{d_{m,1}} or Close_{d_{m,1}} defines a
        # different signal.
        spy_mtd_return_float = float(spy_signal_close_float / spy_previous_close_float - 1.0)
        tlt_mtd_return_float = float(tlt_signal_close_float / tlt_previous_close_float - 1.0)
        signal_spread_float = float(spy_mtd_return_float - tlt_mtd_return_float)

        selected_asset_str: str | None
        if signal_spread_float > 0.0:
            selected_asset_str = "TLT"
        elif signal_spread_float < 0.0:
            selected_asset_str = "SPY"
        else:
            selected_asset_str = None

        month_signal_row_list.append(
            {
                "signal_month_period": month_period,
                "month_observation_count_int": int(month_observation_count_int),
                "previous_month_end_bar_ts": previous_month_end_bar_ts,
                "month_start_bar_ts": month_start_bar_ts,
                "signal_end_bar_ts": signal_end_bar_ts,
                "entry_bar_ts": entry_bar_ts,
                "exit_bar_ts": exit_bar_ts,
                "spy_mtd_return_float": spy_mtd_return_float,
                "tlt_mtd_return_float": tlt_mtd_return_float,
                "signal_spread_float": signal_spread_float,
                "selected_asset_str": selected_asset_str,
            }
        )

    month_signal_df = pd.DataFrame(month_signal_row_list)
    if len(month_signal_df) == 0:
        return pd.DataFrame(
            columns=[
                "signal_month_period",
                "month_observation_count_int",
                "previous_month_end_bar_ts",
                "month_start_bar_ts",
                "signal_end_bar_ts",
                "entry_bar_ts",
                "exit_bar_ts",
                "spy_mtd_return_float",
                "tlt_mtd_return_float",
                "signal_spread_float",
                "selected_asset_str",
            ]
        )

    month_signal_df["signal_month_period_str"] = month_signal_df["signal_month_period"].astype(str)
    month_signal_df = month_signal_df.set_index("signal_month_period_str", drop=True)
    return month_signal_df


def build_trade_plan_df(
    month_signal_df: pd.DataFrame,
    config: EomTltSpyExactConfig = DEFAULT_CONFIG,
) -> pd.DataFrame:
    trade_plan_row_list: list[dict[str, object]] = []
    trade_id_int = 0

    for signal_month_period_str, signal_row_ser in month_signal_df.iterrows():
        selected_asset_str = signal_row_ser["selected_asset_str"]
        if selected_asset_str is None or pd.isna(selected_asset_str):
            continue

        trade_id_int += 1
        trade_plan_row_list.append(
            {
                "trade_id_int": int(trade_id_int),
                "signal_month_period_str": signal_month_period_str,
                "asset_str": str(selected_asset_str),
                "signed_weight_float": float(config.position_weight_float),
                "entry_bar_ts": pd.Timestamp(signal_row_ser["entry_bar_ts"]),
                "exit_bar_ts": pd.Timestamp(signal_row_ser["exit_bar_ts"]),
                "spy_mtd_return_float": float(signal_row_ser["spy_mtd_return_float"]),
                "tlt_mtd_return_float": float(signal_row_ser["tlt_mtd_return_float"]),
                "signal_spread_float": float(signal_row_ser["signal_spread_float"]),
            }
        )

    trade_plan_df = pd.DataFrame(trade_plan_row_list)
    if len(trade_plan_df) == 0:
        return pd.DataFrame(
            columns=[
                "trade_id_int",
                "signal_month_period_str",
                "asset_str",
                "signed_weight_float",
                "entry_bar_ts",
                "exit_bar_ts",
                "spy_mtd_return_float",
                "tlt_mtd_return_float",
                "signal_spread_float",
            ]
        )

    trade_plan_df = trade_plan_df.sort_values(["entry_bar_ts", "trade_id_int"]).set_index("trade_id_int", drop=True)
    trade_plan_df.index.name = "trade_id_int"
    return trade_plan_df


def build_daily_target_weight_df(
    trading_index: pd.DatetimeIndex,
    trade_plan_df: pd.DataFrame,
    asset_list: Sequence[str],
) -> pd.DataFrame:
    target_weight_df = pd.DataFrame(0.0, index=pd.DatetimeIndex(trading_index), columns=list(asset_list), dtype=float)
    for _, trade_row_ser in trade_plan_df.iterrows():
        entry_bar_ts = pd.Timestamp(trade_row_ser["entry_bar_ts"])
        exit_bar_ts = pd.Timestamp(trade_row_ser["exit_bar_ts"])
        asset_str = str(trade_row_ser["asset_str"])

        # *** CRITICAL*** This inclusive date slice encodes the article's
        # exact hold window:
        #   Open_{d_{m,L+1}} through Close_{d_{m,N_m}}
        target_weight_df.loc[entry_bar_ts:exit_bar_ts, asset_str] = float(trade_row_ser["signed_weight_float"])

    return target_weight_df


class EomTltSpyExactResearchStrategy(Strategy):
    """
    Research-only container for the exact SPY / TLT Strategy 1 semantics.
    """

    enable_signal_audit = False
    signal_audit_sample_size = 10

    def __init__(
        self,
        name: str,
        benchmarks: Sequence[str],
        trade_leg_plan_df: pd.DataFrame,
        daily_target_weight_df: pd.DataFrame,
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
        self.trade_leg_plan_df = trade_leg_plan_df.copy()
        self.daily_target_weight_df = daily_target_weight_df.copy()
        self.tradeable_asset_list = ["SPY", "TLT"]

    def compute_signals(self, pricing_data_df: pd.DataFrame) -> pd.DataFrame:
        return pricing_data_df

    def iterate(self, data_df: pd.DataFrame, close_row_ser: pd.Series, open_price_ser: pd.Series):
        return


def run_eom_tlt_spy_exact_research_backtest(
    strategy: EomTltSpyExactResearchStrategy,
    pricing_data_df: pd.DataFrame,
) -> EomTltSpyExactResearchStrategy:
    open_price_df = pricing_data_df.xs("Open", axis=1, level=1)[strategy.tradeable_asset_list].astype(float)
    close_price_df = pricing_data_df.xs("Close", axis=1, level=1)[strategy.tradeable_asset_list].astype(float)
    trading_index = pd.DatetimeIndex(pricing_data_df.index)

    entry_plan_map = {
        pd.Timestamp(entry_bar_ts): trade_leg_sub_df.copy()
        for entry_bar_ts, trade_leg_sub_df in strategy.trade_leg_plan_df.groupby("entry_bar_ts", sort=True)
    }
    exit_plan_map = {
        pd.Timestamp(exit_bar_ts): trade_leg_sub_df.copy()
        for exit_bar_ts, trade_leg_sub_df in strategy.trade_leg_plan_df.groupby("exit_bar_ts", sort=True)
    }

    cash_value_float = float(strategy._capital_base)
    position_share_map: dict[str, int] = {asset_str: 0 for asset_str in strategy.tradeable_asset_list}
    active_trade_share_map: dict[int, int] = {}
    transaction_row_list: list[dict[str, object]] = []
    portfolio_value_map: dict[pd.Timestamp, float] = {}
    cash_value_map: dict[pd.Timestamp, float] = {}
    total_value_map: dict[pd.Timestamp, float] = {}

    for bar_ts in trading_index:
        bar_ts = pd.Timestamp(bar_ts)

        if bar_ts in entry_plan_map:
            pre_entry_equity_float = float(
                cash_value_float
                + sum(
                    position_share_map[asset_str] * float(open_price_df.loc[bar_ts, asset_str])
                    for asset_str in strategy.tradeable_asset_list
                    if position_share_map[asset_str] != 0
                )
            )

            for trade_id_int, trade_row_ser in entry_plan_map[bar_ts].iterrows():
                asset_str = str(trade_row_ser["asset_str"])
                signed_weight_float = float(trade_row_ser["signed_weight_float"])
                open_price_float = float(open_price_df.loc[bar_ts, asset_str])
                entry_fill_price_float = float(open_price_float * (1.0 + float(strategy._slippage)))
                notional_value_float = float(pre_entry_equity_float * abs(signed_weight_float))
                entry_share_count_int = int(notional_value_float / entry_fill_price_float)
                entry_commission_float = compute_commission_float(
                    share_count_int=entry_share_count_int,
                    commission_per_share_float=float(strategy._commission_per_share),
                    commission_minimum_float=float(strategy._commission_minimum),
                )

                if entry_share_count_int == 0:
                    continue

                cash_value_float -= float(entry_share_count_int * entry_fill_price_float)
                cash_value_float -= float(entry_commission_float)
                position_share_map[asset_str] += int(entry_share_count_int)
                active_trade_share_map[int(trade_id_int)] = int(entry_share_count_int)

                transaction_row_list.append(
                    {
                        "trade_id": int(trade_id_int),
                        "bar": bar_ts,
                        "asset": asset_str,
                        "amount": int(entry_share_count_int),
                        "price": float(entry_fill_price_float),
                        "total_value": float(entry_share_count_int * entry_fill_price_float),
                        "order_id": len(transaction_row_list) + 1,
                        "commission": float(entry_commission_float),
                    }
                )

        if bar_ts in exit_plan_map:
            for trade_id_int, trade_row_ser in exit_plan_map[bar_ts].iterrows():
                asset_str = str(trade_row_ser["asset_str"])
                if int(trade_id_int) not in active_trade_share_map:
                    raise RuntimeError(f"Missing active trade shares for trade_id {trade_id_int} on {bar_ts}.")

                exit_share_count_int = int(active_trade_share_map.pop(int(trade_id_int)))
                close_price_float = float(close_price_df.loc[bar_ts, asset_str])
                exit_fill_price_float = float(close_price_float * (1.0 - float(strategy._slippage)))
                exit_commission_float = compute_commission_float(
                    share_count_int=exit_share_count_int,
                    commission_per_share_float=float(strategy._commission_per_share),
                    commission_minimum_float=float(strategy._commission_minimum),
                )

                cash_value_float += float(exit_share_count_int * exit_fill_price_float)
                cash_value_float -= float(exit_commission_float)
                position_share_map[asset_str] -= int(exit_share_count_int)

                transaction_row_list.append(
                    {
                        "trade_id": int(trade_id_int),
                        "bar": bar_ts,
                        "asset": asset_str,
                        "amount": -int(exit_share_count_int),
                        "price": float(exit_fill_price_float),
                        "total_value": float(-exit_share_count_int * exit_fill_price_float),
                        "order_id": len(transaction_row_list) + 1,
                        "commission": float(exit_commission_float),
                    }
                )

        portfolio_value_float = float(
            sum(
                position_share_map[asset_str] * float(close_price_df.loc[bar_ts, asset_str])
                for asset_str in strategy.tradeable_asset_list
                if position_share_map[asset_str] != 0
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

    pricing_data_df = load_pricing_data_df(
        trade_symbol_list=config.trade_symbol_list,
        benchmark_list=config.benchmark_list,
        start_date_str=config.start_date_str,
        end_date_str=config.end_date_str,
    )
    close_price_df = pricing_data_df.xs("Close", axis=1, level=1)[list(config.trade_symbol_list)].astype(float)

    month_signal_df = build_month_signal_df(
        close_price_df=close_price_df,
        config=config,
    )
    trade_plan_df = build_trade_plan_df(
        month_signal_df=month_signal_df,
        config=config,
    )
    daily_target_weight_df = build_daily_target_weight_df(
        trading_index=pricing_data_df.index,
        trade_plan_df=trade_plan_df,
        asset_list=config.trade_symbol_list,
    )

    strategy = EomTltSpyExactResearchStrategy(
        name="strategy_eom_tlt_spy_exact_research",
        benchmarks=config.benchmark_list,
        trade_leg_plan_df=trade_plan_df,
        daily_target_weight_df=daily_target_weight_df,
        capital_base=config.capital_base_float,
        slippage=config.slippage_float,
        commission_per_share=config.commission_per_share_float,
        commission_minimum=config.commission_minimum_float,
    )
    strategy.daily_target_weights = daily_target_weight_df.copy()

    run_eom_tlt_spy_exact_research_backtest(
        strategy=strategy,
        pricing_data_df=pricing_data_df,
    )

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)

    print("Month signal preview:")
    display(month_signal_df.head())
    print("Trade plan preview:")
    display(trade_plan_df.head())
    display(strategy.summary)
    display(strategy.summary_trades)

    save_results(strategy)
