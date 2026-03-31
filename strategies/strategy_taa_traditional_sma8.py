"""
Traditional equal-sleeve SMA(8) monthly trend strategy.

TL;DR: Hold each asset at a fixed sleeve weight of 1 / N when its month-end
signal close is above its 8-month SMA. Otherwise hold that sleeve in cash.

Core formulas
-------------
For each asset i and completed month m:

    monthly_close_{i,m}
        = last_trading_close_in_month(i, m)

    sma8_{i,m}
        = (1 / 8) * sum_{k=0}^{7} monthly_close_{i,m-k}

    signal_{i,m}
        = 1[monthly_close_{i,m} > sma8_{i,m}]

With N assets and fixed equal sleeves:

    base_weight_i
        = 1 / N

    target_weight_{i,m}
        = base_weight_i * signal_{i,m}

    cash_weight_m
        = 1 - sum_i target_weight_{i,m}

Execution-model note
--------------------
The user-specified paper rule is:

    month-end close decision -> month-end close execution

The repository engine executes orders placed in `iterate()` at the next bar's
open, so this implementation uses the causal engine-native mapping:

    month-end close decision -> first tradable open of next month

This preserves causality and avoids same-bar fill fantasy.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Sequence

import numpy as np
import pandas as pd
import norgatedata
from IPython.display import display
from tqdm.auto import tqdm

WORKSPACE_ROOT_PATH = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT_PATH))

from alpha.engine.backtest import run_daily
from alpha.engine.report import save_results
from alpha.engine.strategy import Strategy
from data.norgate_loader import load_raw_prices


def default_trade_id_int() -> int:
    return -1


@dataclass(frozen=True)
class AssetSignalSpec:
    asset_class_str: str
    trade_symbol_str: str
    signal_symbol_candidate_tuple: tuple[str, ...]


@dataclass(frozen=True)
class TraditionalSma8Config:
    asset_spec_tuple: tuple[AssetSignalSpec, ...] = (
        AssetSignalSpec("U.S. Stocks", "VTI", ("$SPXTR", "VTI")),
        AssetSignalSpec("Developed Intl.", "VEA", ("$MSCI_EAFE_TR", "VEA")),
        AssetSignalSpec("Emerging Markets", "VWO", ("$MSCI_EM_TR", "VWO")),
        AssetSignalSpec("Real Estate", "VNQ", ("$DJ_REIT_TR", "VNQ")),
        AssetSignalSpec("LT Treasuries", "TLT", ("$UST20Y_TR", "TLT")),
        AssetSignalSpec("Gold", "GLD", ("$GOLD", "GLD")),
        AssetSignalSpec("Commodities", "DBC", ("$BCOMTR", "DBC")),
    )
    benchmark_list: tuple[str, ...] = ("$SPXTR",)
    sma_month_lookback_int: int = 8
    start_date_str: str = "1995-01-01"
    end_date_str: str | None = None
    capital_base_float: float = 100_000.0
    slippage_float: float = 0.0001
    commission_per_share_float: float = 0.005
    commission_minimum_float: float = 1.0

    def __post_init__(self):
        if len(self.asset_spec_tuple) == 0:
            raise ValueError("asset_spec_tuple must contain at least one asset.")
        if len({asset_spec.trade_symbol_str for asset_spec in self.asset_spec_tuple}) != len(self.asset_spec_tuple):
            raise ValueError("trade_symbol_str values must be unique.")
        for asset_spec in self.asset_spec_tuple:
            if len(asset_spec.signal_symbol_candidate_tuple) == 0:
                raise ValueError("Each asset_spec must contain at least one signal symbol candidate.")
        if self.sma_month_lookback_int <= 0:
            raise ValueError("sma_month_lookback_int must be positive.")
        if self.capital_base_float <= 0.0:
            raise ValueError("capital_base_float must be positive.")
        if self.slippage_float < 0.0:
            raise ValueError("slippage_float must be non-negative.")
        if self.commission_per_share_float < 0.0:
            raise ValueError("commission_per_share_float must be non-negative.")
        if self.commission_minimum_float < 0.0:
            raise ValueError("commission_minimum_float must be non-negative.")

    @property
    def trade_symbol_list(self) -> list[str]:
        return [asset_spec.trade_symbol_str for asset_spec in self.asset_spec_tuple]


DEFAULT_CONFIG = TraditionalSma8Config()


def load_available_signal_symbol_set() -> set[str]:
    """
    Load the local Norgate symbol catalog used by signal-series fallbacks.
    """
    database_name_tuple = (
        "US Equities",
        "US Indices",
        "World Indices",
        "Economic",
        "Cash Commodities",
    )
    available_signal_symbol_set: set[str] = set()
    for database_name_str in database_name_tuple:
        try:
            symbol_list = norgatedata.database_symbols(database_name_str)
        except Exception:
            continue
        available_signal_symbol_set.update(symbol_list)
    return available_signal_symbol_set


def load_signal_close_df(
    asset_spec_tuple: Sequence[AssetSignalSpec],
    start_date_str: str,
    end_date_str: str | None = None,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """
    Load TOTALRETURN-like monthly signal closes keyed by trade symbol.

    For each sleeve i, try candidate symbols in order until one loads:

        selected_signal_symbol_i
            = first_available(signal_symbol_candidate_tuple_i)
    """
    signal_close_map: dict[str, pd.Series] = {}
    selected_signal_symbol_map: dict[str, str] = {}
    available_signal_symbol_set = load_available_signal_symbol_set()

    for asset_spec in tqdm(asset_spec_tuple, desc="loading signal closes"):
        last_exception_obj: Exception | None = None
        for signal_symbol_str in asset_spec.signal_symbol_candidate_tuple:
            if signal_symbol_str not in available_signal_symbol_set:
                continue
            try:
                signal_price_df = norgatedata.price_timeseries(
                    signal_symbol_str,
                    stock_price_adjustment_setting=norgatedata.StockPriceAdjustmentType.TOTALRETURN,
                    padding_setting=norgatedata.PaddingType.ALLMARKETDAYS,
                    start_date=start_date_str,
                    end_date=end_date_str,
                    timeseriesformat="pandas-dataframe",
                )
            except Exception as exception_obj:
                last_exception_obj = exception_obj
                continue

            if len(signal_price_df) == 0:
                continue

            signal_close_map[asset_spec.trade_symbol_str] = signal_price_df["Close"].astype(float)
            selected_signal_symbol_map[asset_spec.trade_symbol_str] = signal_symbol_str
            break

        if asset_spec.trade_symbol_str not in signal_close_map:
            raise RuntimeError(
                "Unable to load signal data for "
                f"{asset_spec.trade_symbol_str}. Tried {list(asset_spec.signal_symbol_candidate_tuple)}. "
                f"Last error type: {type(last_exception_obj).__name__ if last_exception_obj is not None else 'None'}."
            )

    if len(signal_close_map) == 0:
        raise RuntimeError("No signal close data was loaded.")

    signal_close_df = pd.DataFrame(signal_close_map).sort_index()
    missing_trade_symbol_list = [
        asset_spec.trade_symbol_str
        for asset_spec in asset_spec_tuple
        if asset_spec.trade_symbol_str not in signal_close_df.columns
    ]
    if len(missing_trade_symbol_list) > 0:
        raise RuntimeError(f"Missing signal data for trade symbols: {missing_trade_symbol_list}")

    return signal_close_df, selected_signal_symbol_map


def compute_month_end_trend_weight_df(
    signal_close_df: pd.DataFrame,
    sma_month_lookback_int: int = 8,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Compute month-end close, SMA, binary trend state, and fixed-sleeve weights.
    """
    if sma_month_lookback_int <= 0:
        raise ValueError("sma_month_lookback_int must be positive.")

    # *** CRITICAL*** Only completed month-end closes may enter the monthly
    # trend rule. Mid-month observations would create signal timing leakage.
    monthly_close_df = signal_close_df.resample("ME").last().astype(float)

    # *** CRITICAL*** The SMA must be a backward-only rolling mean over
    # completed month-end closes. Any forward-looking window would leak future data.
    monthly_sma_df = (
        monthly_close_df.rolling(
            window=sma_month_lookback_int,
            min_periods=sma_month_lookback_int,
        ).mean()
    )
    signal_state_df = (monthly_close_df > monthly_sma_df).astype(float)
    signal_state_df = signal_state_df.where(monthly_sma_df.notna())

    valid_month_end_mask_ser = monthly_sma_df.notna().all(axis=1)
    monthly_close_df = monthly_close_df.loc[valid_month_end_mask_ser]
    monthly_sma_df = monthly_sma_df.loc[valid_month_end_mask_ser]
    signal_state_df = signal_state_df.loc[valid_month_end_mask_ser]

    asset_count_int = int(len(signal_close_df.columns))
    base_weight_float = 1.0 / float(asset_count_int)
    month_end_weight_df = signal_state_df.fillna(0.0) * base_weight_float

    weight_sum_ser = month_end_weight_df.sum(axis=1)
    if not np.all(weight_sum_ser.to_numpy(dtype=float) <= 1.0 + 1e-12):
        raise ValueError("Month-end target weights must not exceed 100% gross exposure.")

    return monthly_close_df, monthly_sma_df, signal_state_df, month_end_weight_df


def map_month_end_weights_to_rebalance_open_df(
    month_end_weight_df: pd.DataFrame,
    execution_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    """
    Map each month-end decision to the first trading day of the next month.
    """
    execution_month_ser = pd.Series(execution_index, index=execution_index.to_period("M"))
    first_trading_day_ser = execution_month_ser.groupby(level=0).min()

    rebalance_weight_map: dict[pd.Timestamp, pd.Series] = {}
    for decision_ts, target_weight_ser in month_end_weight_df.iterrows():
        # *** CRITICAL*** Month-end decision t becomes tradable only at the
        # first tradable open in month t + 1 under the engine's next-open contract.
        next_month_period = (decision_ts + pd.offsets.MonthBegin(1)).to_period("M")
        if next_month_period not in first_trading_day_ser.index:
            continue

        rebalance_ts = pd.Timestamp(first_trading_day_ser.loc[next_month_period])
        rebalance_weight_map[rebalance_ts] = target_weight_ser.copy()

    if len(rebalance_weight_map) == 0:
        return pd.DataFrame(columns=month_end_weight_df.columns, dtype=float)

    rebalance_weight_df = pd.DataFrame.from_dict(rebalance_weight_map, orient="index").sort_index()
    rebalance_weight_df.index.name = "rebalance_date"
    return rebalance_weight_df


def build_daily_target_weight_df(
    rebalance_weight_df: pd.DataFrame,
    execution_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    """
    Expand rebalance targets into a daily schedule for reporting.
    """
    if len(rebalance_weight_df) == 0:
        return pd.DataFrame(columns=list(rebalance_weight_df.columns) + ["Cash"], dtype=float)

    daily_target_weight_df = rebalance_weight_df.reindex(pd.DatetimeIndex(execution_index)).ffill()
    daily_target_weight_df = daily_target_weight_df.loc[
        daily_target_weight_df.index >= pd.Timestamp(rebalance_weight_df.index[0])
    ]
    daily_target_weight_df = daily_target_weight_df.fillna(0.0)
    daily_target_weight_df["Cash"] = 1.0 - daily_target_weight_df.sum(axis=1)
    weight_sum_ser = daily_target_weight_df.sum(axis=1)
    if not np.allclose(weight_sum_ser.to_numpy(dtype=float), 1.0, atol=1e-12):
        raise ValueError("Daily target weights, including cash, must sum to 1.0.")
    return daily_target_weight_df


def get_traditional_sma8_data(
    config: TraditionalSma8Config = DEFAULT_CONFIG,
) -> tuple[pd.DataFrame, dict[str, str], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load signal data, execution data, monthly diagnostics, and rebalance targets.
    """
    signal_close_df, selected_signal_symbol_map = load_signal_close_df(
        asset_spec_tuple=config.asset_spec_tuple,
        start_date_str=config.start_date_str,
        end_date_str=config.end_date_str,
    )
    execution_price_df = load_raw_prices(
        symbols=config.trade_symbol_list,
        benchmarks=list(config.benchmark_list),
        start_date=config.start_date_str,
        end_date=config.end_date_str,
    )
    loaded_symbol_set = set(execution_price_df.columns.get_level_values(0))
    missing_trade_symbol_list = [
        trade_symbol_str
        for trade_symbol_str in config.trade_symbol_list
        if trade_symbol_str not in loaded_symbol_set
    ]
    if len(missing_trade_symbol_list) > 0:
        raise RuntimeError(f"Missing execution data for trade symbols: {missing_trade_symbol_list}")

    missing_benchmark_list = [
        benchmark_str for benchmark_str in config.benchmark_list if benchmark_str not in loaded_symbol_set
    ]
    if len(missing_benchmark_list) > 0:
        raise RuntimeError(f"Missing execution data for benchmarks: {missing_benchmark_list}")

    open_key_list = [(trade_symbol_str, "Open") for trade_symbol_str in config.trade_symbol_list]
    valid_execution_bar_mask_ser = execution_price_df.loc[:, open_key_list].notna().all(axis=1)
    actionable_execution_index = execution_price_df.index[valid_execution_bar_mask_ser]
    if len(actionable_execution_index) == 0:
        raise RuntimeError("No execution bars contain valid opens for every tradeable asset.")

    monthly_close_df, monthly_sma_df, signal_state_df, month_end_weight_df = compute_month_end_trend_weight_df(
        signal_close_df=signal_close_df,
        sma_month_lookback_int=config.sma_month_lookback_int,
    )
    rebalance_weight_df = map_month_end_weights_to_rebalance_open_df(
        month_end_weight_df=month_end_weight_df,
        execution_index=actionable_execution_index,
    )

    return (
        execution_price_df,
        selected_signal_symbol_map,
        monthly_close_df,
        monthly_sma_df,
        signal_state_df,
        month_end_weight_df,
        rebalance_weight_df,
    )


class TraditionalSma8TrendStrategy(Strategy):
    """
    Fixed-sleeve monthly trend allocator.

    Each asset owns a constant sleeve:

        base_weight_i = 1 / N

    At each actionable rebalance open:

        target_weight_{i,t} = base_weight_i * signal_{i,t-1}

    Residual capital stays in cash.
    """

    enable_signal_audit = False
    signal_audit_sample_size = 10

    def __init__(
        self,
        name: str,
        benchmarks: Sequence[str],
        trade_symbol_list: Sequence[str],
        rebalance_weight_df: pd.DataFrame,
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
        self.trade_symbol_list = list(trade_symbol_list)
        self.rebalance_weight_df = rebalance_weight_df.copy()
        self.trade_id_int = 0
        self.current_trade_id_map: defaultdict[str, int] = defaultdict(default_trade_id_int)
        self.show_taa_weights_report = True
        self.daily_target_weights = pd.DataFrame(columns=self.trade_symbol_list + ["Cash"], dtype=float)

    def compute_signals(self, pricing_data_df: pd.DataFrame) -> pd.DataFrame:
        return pricing_data_df

    def _ensure_trade_id_int(self, trade_symbol_str: str) -> int:
        if self.current_trade_id_map[trade_symbol_str] == default_trade_id_int():
            self.trade_id_int += 1
            self.current_trade_id_map[trade_symbol_str] = self.trade_id_int
        return int(self.current_trade_id_map[trade_symbol_str])

    def iterate(self, data_df: pd.DataFrame, close_row_ser: pd.Series, open_price_ser: pd.Series):
        if close_row_ser is None or data_df is None:
            return
        if self.current_bar not in self.rebalance_weight_df.index:
            return

        target_weight_ser = self.rebalance_weight_df.loc[self.current_bar].reindex(self.trade_symbol_list).fillna(0.0)
        current_position_ser = self.get_positions().reindex(self.trade_symbol_list, fill_value=0.0).astype(int)
        budget_value_float = float(self.previous_total_value)

        for trade_symbol_str in self.trade_symbol_list:
            target_weight_float = float(target_weight_ser.loc[trade_symbol_str])
            current_share_int = int(current_position_ser.loc[trade_symbol_str])
            open_price_float = float(open_price_ser.get(trade_symbol_str, np.nan))

            if not np.isfinite(open_price_float) or open_price_float <= 0.0:
                raise RuntimeError(
                    f"Invalid open price for target asset {trade_symbol_str} on {self.current_bar}."
                )

            # *** CRITICAL*** Target shares use previous-bar portfolio value and
            # the current rebalance open. Same-bar close sizing would leak information.
            target_share_int = int(np.floor(budget_value_float * target_weight_float / open_price_float))
            if target_share_int == current_share_int:
                continue

            if target_share_int <= 0:
                if current_share_int <= 0:
                    continue
                self.order_target_value(
                    trade_symbol_str,
                    0.0,
                    trade_id=self._ensure_trade_id_int(trade_symbol_str),
                )
                self.current_trade_id_map[trade_symbol_str] = default_trade_id_int()
                continue

            if (
                current_share_int <= 0
                or self.current_trade_id_map[trade_symbol_str] == default_trade_id_int()
            ):
                self._ensure_trade_id_int(trade_symbol_str)

            self.order_target_percent(
                trade_symbol_str,
                target_weight_float,
                trade_id=self.current_trade_id_map[trade_symbol_str],
            )

    def finalize(self, current_data: pd.DataFrame):
        if len(self.rebalance_weight_df) == 0:
            return
        self.daily_target_weights = build_daily_target_weight_df(
            rebalance_weight_df=self.rebalance_weight_df,
            execution_index=current_data.index,
        )


if __name__ == "__main__":
    config = DEFAULT_CONFIG
    (
        execution_price_df,
        selected_signal_symbol_map,
        monthly_close_df,
        monthly_sma_df,
        signal_state_df,
        month_end_weight_df,
        rebalance_weight_df,
    ) = get_traditional_sma8_data(config=config)

    strategy = TraditionalSma8TrendStrategy(
        name="strategy_taa_traditional_sma8",
        benchmarks=config.benchmark_list,
        trade_symbol_list=config.trade_symbol_list,
        rebalance_weight_df=rebalance_weight_df,
        capital_base=config.capital_base_float,
        slippage=config.slippage_float,
        commission_per_share=config.commission_per_share_float,
        commission_minimum=config.commission_minimum_float,
    )
    strategy.daily_target_weights = build_daily_target_weight_df(
        rebalance_weight_df=rebalance_weight_df,
        execution_index=execution_price_df.index,
    )

    if len(rebalance_weight_df) == 0:
        raise RuntimeError("No actionable rebalance dates were generated for strategy_taa_traditional_sma8.")

    calendar_index = execution_price_df.index[execution_price_df.index >= rebalance_weight_df.index[0]]
    run_daily(
        strategy,
        execution_price_df,
        calendar=calendar_index,
        audit_override_bool=None,
    )

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)

    print("Monthly close preview:")
    display(monthly_close_df.tail())

    print("Selected signal symbols:")
    display(pd.Series(selected_signal_symbol_map).rename("signal_symbol_str"))

    print("Monthly SMA preview:")
    display(monthly_sma_df.tail())

    print("Monthly signal state preview:")
    display(signal_state_df.tail())

    print("Rebalance weights preview:")
    display(rebalance_weight_df.tail())

    display(strategy.summary)
    display(strategy.summary_trades)
    save_results(strategy)
