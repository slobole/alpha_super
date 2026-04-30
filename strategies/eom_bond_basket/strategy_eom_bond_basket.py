"""
Monthly close-to-close bond ETF basket strategy.

TL;DR: By default, hold an equal-weight basket of TLT, EDV, and ZROZ from the
entry close of each completed month through the close of the last trading day
of that same month.

Core formulas
-------------
Let the ordered trading dates in completed month m be:

    D_m = {d_{m,1}, d_{m,2}, ..., d_{m,N_m}}

Define the reverse trading-day count:

    b(d_{m,k}) = N_m - k + 1

So:

    b = 3 -> entry_bar_m = d_{m,N_m-2}
    b = 1 -> exit_bar_m  = d_{m,N_m}

For M assets and total basket weight W:

    w_i = W / M

At the entry close:

    q_{i,m} = floor(E^{pre}_{m,entry} * w_i / (Close_{i,entry} * (1 + slippage)))

Daily close-marked equity is:

    E_t = cash_t + sum_i q_{i,t} * Close_{i,t}

Gross month trade return before integer-share cash drag and costs is:

    r_m = sum_i w_i * (Close_{i,exit} / Close_{i,entry} - 1)

Execution-model note
--------------------
This file is intentionally research-only because the repository's standard
`run_daily()` engine fills orders at the next open. The requested rule fills at:

    entry fill = Close_{entry_bar_m}
    exit fill  = Close_{exit_bar_m}
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


@dataclass(frozen=True)
class EomBondBasketConfig:
    trade_symbol_list: tuple[str, ...] = ("TLT", "EDV", "ZROZ")
    benchmark_list: tuple[str, ...] = ("$SPX",)
    entry_bday_count_int: int = 3
    exit_bday_count_int: int = 1
    basket_weight_float: float = 1.0
    start_date_str: str = "2010-01-01"
    end_date_str: str | None = None
    capital_base_float: float = 100_000.0
    slippage_float: float = 0.00025
    commission_per_share_float: float = 0.005
    commission_minimum_float: float = 1.0

    def __post_init__(self):
        if len(self.trade_symbol_list) == 0:
            raise ValueError("trade_symbol_list must contain at least one symbol.")
        if len(set(self.trade_symbol_list)) != len(self.trade_symbol_list):
            raise ValueError("trade_symbol_list must not contain duplicates.")
        if len(set(self.trade_symbol_list).intersection(set(self.benchmark_list))) > 0:
            raise ValueError("benchmark_list must not overlap trade_symbol_list.")
        if self.exit_bday_count_int <= 0:
            raise ValueError("exit_bday_count_int must be positive.")
        if self.entry_bday_count_int <= self.exit_bday_count_int:
            raise ValueError("entry_bday_count_int must be greater than exit_bday_count_int.")
        if self.basket_weight_float <= 0.0:
            raise ValueError("basket_weight_float must be positive.")
        if self.capital_base_float <= 0.0:
            raise ValueError("capital_base_float must be positive.")
        if self.slippage_float < 0.0:
            raise ValueError("slippage_float must be non-negative.")
        if self.commission_per_share_float < 0.0:
            raise ValueError("commission_per_share_float must be non-negative.")
        if self.commission_minimum_float < 0.0:
            raise ValueError("commission_minimum_float must be non-negative.")


DEFAULT_CONFIG = EomBondBasketConfig()


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
    missing_column_list = [
        column_tuple
        for column_tuple in trade_required_column_list
        if column_tuple not in pricing_data_df.columns
    ]
    if len(missing_column_list) > 0:
        raise RuntimeError(f"Missing required trade columns: {missing_column_list}")

    # *** CRITICAL*** The b=3/b=1 month-end window must be computed on a
    # synchronized tradable-ETF calendar. Dropping incomplete rows prevents a
    # missing price in one ETF from silently moving that ETF's effective entry
    # or exit date.
    valid_trade_bar_mask_ser = pricing_data_df.loc[:, trade_required_column_list].notna().all(axis=1)
    pricing_data_df = pricing_data_df.loc[valid_trade_bar_mask_ser].copy()
    if len(pricing_data_df) == 0:
        raise RuntimeError("No synchronized trade bars remain after dropping incomplete ETF rows.")

    return pricing_data_df


def get_completed_month_period_set(trading_index: pd.DatetimeIndex) -> set[pd.Period]:
    if len(trading_index) == 0:
        return set()

    completed_month_period_set = set(trading_index.to_period("M").unique().tolist())
    last_available_bar_ts = pd.Timestamp(trading_index[-1])
    expected_business_month_end_ts = pd.Timestamp(last_available_bar_ts + pd.offsets.BMonthEnd(0))

    # *** CRITICAL*** The last observed month may be partial in live/downloaded
    # data. Excluding it unless the last available bar is business-month-end
    # prevents an incomplete month from being treated as if b=1 were known.
    if expected_business_month_end_ts.normalize() != last_available_bar_ts.normalize():
        completed_month_period_set.discard(last_available_bar_ts.to_period("M"))

    return completed_month_period_set


def build_month_trade_plan_df(
    close_price_df: pd.DataFrame,
    config: EomBondBasketConfig = DEFAULT_CONFIG,
) -> pd.DataFrame:
    ordered_trading_index = pd.DatetimeIndex(close_price_df.index).sort_values()
    month_period_index = ordered_trading_index.to_period("M")
    month_group_map = pd.Series(ordered_trading_index, index=ordered_trading_index).groupby(month_period_index).groups
    completed_month_period_set = get_completed_month_period_set(ordered_trading_index)

    month_trade_row_list: list[dict[str, object]] = []
    trade_id_int = 0
    for month_period, month_date_index in month_group_map.items():
        if month_period not in completed_month_period_set:
            continue

        month_trading_index = pd.DatetimeIndex(month_date_index).sort_values()
        month_observation_count_int = len(month_trading_index)

        # *** CRITICAL*** b=3 requires at least three synchronized trading bars
        # in the completed month. Otherwise the third-to-last close is not
        # defined.
        if month_observation_count_int < config.entry_bday_count_int:
            continue

        entry_bar_ts = pd.Timestamp(month_trading_index[-config.entry_bday_count_int])
        exit_bar_ts = pd.Timestamp(month_trading_index[-config.exit_bday_count_int])
        if entry_bar_ts >= exit_bar_ts:
            raise RuntimeError(
                f"Invalid b-day ordering in {month_period}: "
                f"entry_bar_ts={entry_bar_ts}, exit_bar_ts={exit_bar_ts}"
            )

        entry_close_price_ser = close_price_df.loc[entry_bar_ts, list(config.trade_symbol_list)].astype(float)
        exit_close_price_ser = close_price_df.loc[exit_bar_ts, list(config.trade_symbol_list)].astype(float)
        if (entry_close_price_ser <= 0.0).any() or not np.isfinite(entry_close_price_ser).all():
            raise RuntimeError(f"Invalid entry close prices on {entry_bar_ts}.")
        if (exit_close_price_ser <= 0.0).any() or not np.isfinite(exit_close_price_ser).all():
            raise RuntimeError(f"Invalid exit close prices on {exit_bar_ts}.")

        trade_id_int += 1
        month_trade_row_list.append(
            {
                "trade_id_int": int(trade_id_int),
                "signal_month_period": month_period,
                "signal_month_period_str": str(month_period),
                "month_observation_count_int": int(month_observation_count_int),
                "entry_bday_count_int": int(config.entry_bday_count_int),
                "exit_bday_count_int": int(config.exit_bday_count_int),
                "entry_bar_ts": entry_bar_ts,
                "exit_bar_ts": exit_bar_ts,
            }
        )

    month_trade_plan_df = pd.DataFrame(month_trade_row_list)
    if len(month_trade_plan_df) == 0:
        return pd.DataFrame(
            columns=[
                "trade_id_int",
                "signal_month_period",
                "signal_month_period_str",
                "month_observation_count_int",
                "entry_bday_count_int",
                "exit_bday_count_int",
                "entry_bar_ts",
                "exit_bar_ts",
            ]
        )

    month_trade_plan_df = month_trade_plan_df.sort_values(["entry_bar_ts", "trade_id_int"])
    month_trade_plan_df = month_trade_plan_df.set_index("trade_id_int", drop=True)
    month_trade_plan_df.index.name = "trade_id_int"
    return month_trade_plan_df


def build_trade_leg_plan_df(
    month_trade_plan_df: pd.DataFrame,
    config: EomBondBasketConfig = DEFAULT_CONFIG,
) -> pd.DataFrame:
    trade_leg_row_list: list[dict[str, object]] = []
    leg_trade_id_int = 0
    asset_weight_float = float(config.basket_weight_float) / float(len(config.trade_symbol_list))

    for month_trade_id_int, month_trade_row_ser in month_trade_plan_df.iterrows():
        for asset_str in config.trade_symbol_list:
            leg_trade_id_int += 1
            trade_leg_row_list.append(
                {
                    "trade_id_int": int(leg_trade_id_int),
                    "basket_trade_id_int": int(month_trade_id_int),
                    "signal_month_period_str": str(month_trade_row_ser["signal_month_period_str"]),
                    "asset_str": str(asset_str),
                    "signed_weight_float": float(asset_weight_float),
                    "entry_bar_ts": pd.Timestamp(month_trade_row_ser["entry_bar_ts"]),
                    "exit_bar_ts": pd.Timestamp(month_trade_row_ser["exit_bar_ts"]),
                }
            )

    trade_leg_plan_df = pd.DataFrame(trade_leg_row_list)
    if len(trade_leg_plan_df) == 0:
        return pd.DataFrame(
            columns=[
                "trade_id_int",
                "basket_trade_id_int",
                "signal_month_period_str",
                "asset_str",
                "signed_weight_float",
                "entry_bar_ts",
                "exit_bar_ts",
            ]
        )

    trade_leg_plan_df = trade_leg_plan_df.sort_values(["entry_bar_ts", "trade_id_int"])
    trade_leg_plan_df = trade_leg_plan_df.set_index("trade_id_int", drop=True)
    trade_leg_plan_df.index.name = "trade_id_int"
    return trade_leg_plan_df


def build_daily_target_weight_df(
    trading_index: pd.DatetimeIndex,
    trade_leg_plan_df: pd.DataFrame,
    asset_list: Sequence[str],
) -> pd.DataFrame:
    target_weight_df = pd.DataFrame(0.0, index=pd.DatetimeIndex(trading_index), columns=list(asset_list), dtype=float)
    for _, trade_leg_row_ser in trade_leg_plan_df.iterrows():
        entry_bar_ts = pd.Timestamp(trade_leg_row_ser["entry_bar_ts"])
        exit_bar_ts = pd.Timestamp(trade_leg_row_ser["exit_bar_ts"])
        asset_str = str(trade_leg_row_ser["asset_str"])

        # *** CRITICAL*** This inclusive close-to-close slice encodes:
        #   Close_{b=3} through Close_{b=1}
        # It is a target-weight report for the research path, not an
        # instruction to the standard next-open engine.
        target_weight_df.loc[entry_bar_ts:exit_bar_ts, asset_str] += float(trade_leg_row_ser["signed_weight_float"])

    return target_weight_df


class EomBondBasketCloseResearchStrategy(Strategy):
    """
    Research-only container for the b=3 to b=1 monthly bond ETF basket.
    """

    enable_signal_audit = False
    signal_audit_sample_size = 10

    def __init__(
        self,
        name: str,
        benchmarks: Sequence[str],
        tradeable_asset_list: Sequence[str],
        trade_leg_plan_df: pd.DataFrame,
        daily_target_weight_df: pd.DataFrame,
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
        self.tradeable_asset_list = [str(asset_str) for asset_str in tradeable_asset_list]
        self.trade_leg_plan_df = trade_leg_plan_df.copy()
        self.daily_target_weight_df = daily_target_weight_df.copy()

    def compute_signals(self, pricing_data_df: pd.DataFrame) -> pd.DataFrame:
        return pricing_data_df

    def iterate(self, data_df: pd.DataFrame, close_row_ser: pd.Series, open_price_ser: pd.Series):
        return


def run_eom_bond_basket_close_research_backtest(
    strategy: EomBondBasketCloseResearchStrategy,
    pricing_data_df: pd.DataFrame,
) -> EomBondBasketCloseResearchStrategy:
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
    skipped_zero_share_trade_id_set: set[int] = set()
    transaction_row_list: list[dict[str, object]] = []
    portfolio_value_map: dict[pd.Timestamp, float] = {}
    cash_value_map: dict[pd.Timestamp, float] = {}
    total_value_map: dict[pd.Timestamp, float] = {}

    for bar_ts in trading_index:
        bar_ts = pd.Timestamp(bar_ts)
        close_price_ser = close_price_df.loc[bar_ts, strategy.tradeable_asset_list].astype(float)

        if bar_ts in entry_plan_map:
            pre_entry_equity_float = float(
                cash_value_float
                + sum(
                    position_share_map[asset_str] * float(close_price_ser.loc[asset_str])
                    for asset_str in strategy.tradeable_asset_list
                )
            )

            for trade_id_int, trade_leg_row_ser in entry_plan_map[bar_ts].iterrows():
                asset_str = str(trade_leg_row_ser["asset_str"])
                signed_weight_float = float(trade_leg_row_ser["signed_weight_float"])
                if signed_weight_float <= 0.0:
                    raise RuntimeError(f"Only long basket legs are supported. Found {signed_weight_float}.")

                close_price_float = float(close_price_ser.loc[asset_str])
                entry_fill_price_float = float(close_price_float * (1.0 + float(strategy._slippage)))
                notional_value_float = float(pre_entry_equity_float * signed_weight_float)
                entry_share_count_int = int(notional_value_float / entry_fill_price_float)
                entry_commission_float = compute_commission_float(
                    share_count_int=entry_share_count_int,
                    commission_per_share_float=float(strategy._commission_per_share),
                    commission_minimum_float=float(strategy._commission_minimum),
                )

                if entry_share_count_int == 0:
                    skipped_zero_share_trade_id_set.add(int(trade_id_int))
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
            for trade_id_int, trade_leg_row_ser in exit_plan_map[bar_ts].iterrows():
                asset_str = str(trade_leg_row_ser["asset_str"])
                if int(trade_id_int) not in active_trade_share_map:
                    if int(trade_id_int) in skipped_zero_share_trade_id_set:
                        continue
                    raise RuntimeError(f"Missing active trade shares for trade_id {trade_id_int} on {bar_ts}.")

                exit_share_count_int = int(active_trade_share_map.pop(int(trade_id_int)))
                close_price_float = float(close_price_ser.loc[asset_str])
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
                position_share_map[asset_str] * float(close_price_ser.loc[asset_str])
                for asset_str in strategy.tradeable_asset_list
            )
        )
        total_value_float = float(cash_value_float + portfolio_value_float)

        portfolio_value_map[bar_ts] = portfolio_value_float
        cash_value_map[bar_ts] = float(cash_value_float)
        total_value_map[bar_ts] = total_value_float

    if len(active_trade_share_map) > 0:
        raise RuntimeError(f"Open trade legs remain after backtest: {sorted(active_trade_share_map)}")

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
    strategy._latest_close_price_ser = close_price_df.iloc[-1].astype(float)
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

    month_trade_plan_df = build_month_trade_plan_df(
        close_price_df=close_price_df,
        config=config,
    )
    trade_leg_plan_df = build_trade_leg_plan_df(
        month_trade_plan_df=month_trade_plan_df,
        config=config,
    )
    daily_target_weight_df = build_daily_target_weight_df(
        trading_index=pricing_data_df.index,
        trade_leg_plan_df=trade_leg_plan_df,
        asset_list=config.trade_symbol_list,
    )

    strategy = EomBondBasketCloseResearchStrategy(
        name="strategy_eom_bond_basket_close_research",
        benchmarks=config.benchmark_list,
        tradeable_asset_list=config.trade_symbol_list,
        trade_leg_plan_df=trade_leg_plan_df,
        daily_target_weight_df=daily_target_weight_df,
        capital_base=config.capital_base_float,
        slippage=config.slippage_float,
        commission_per_share=config.commission_per_share_float,
        commission_minimum=config.commission_minimum_float,
    )
    strategy.daily_target_weights = daily_target_weight_df.copy()

    run_eom_bond_basket_close_research_backtest(
        strategy=strategy,
        pricing_data_df=pricing_data_df,
    )

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)

    print("Month trade plan preview:")
    display(month_trade_plan_df.head())
    print("Trade leg plan preview:")
    display(trade_leg_plan_df.head())
    display(strategy.summary)
    display(strategy.summary_trades)

    save_results(strategy)
