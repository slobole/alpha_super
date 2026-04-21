"""
Defense First tactical allocation strategy.

Paper-faithful monthly signal math with an intentional phase-1 live execution
approximation.

Core formulas
-------------
For each defensive asset i at month-end t:

    momentum_score_{i,t}
        = (r_{1m,i,t} + r_{3m,i,t} + r_{6m,i,t} + r_{12m,i,t}) / 4

    r_{k,i,t} = close_{i,t} / close_{i,t-k} - 1

Rank the defensive assets by momentum_score descending and assign slot weights:

    rank_weight_vec = [0.40, 0.30, 0.20, 0.10]

Absolute momentum filter versus cash:

    if momentum_score_{i,t} > cash_return_t:
        keep the asset's original rank slot weight
    else:
        redirect that slot weight to the fallback asset

Execution philosophy at rebalance open t:

    V^{budget}_t = V^{close}_{t-1}

    q^{target}_{i,t} = floor(V^{budget}_t * w_{i,t} / close_{i,t-1})

    fill_price_{i,t} = open_{i,t}

This is a deliberate overnight OPG-style approximation. Realized weights are
therefore not exact open marks:

    w^{realized}_{i,t} != w^{target}_{i,t}

in general, because of gaps, slippage, and rounding.

This file deliberately separates:
1. Signal data: TOTALRETURN closes for ranking.
2. Execution data: CAPITALSPECIAL OHLC for fills and valuation.

That keeps the signal logic total-return aware while avoiding synthetic
total-return execution prices.
"""

from __future__ import annotations

from datetime import UTC, datetime
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import norgatedata
from IPython.display import display
from tqdm.auto import tqdm

repo_root_path = Path(__file__).resolve().parents[2]
repo_root_str = str(repo_root_path)
if repo_root_str not in sys.path:
    sys.path.insert(0, repo_root_str)

workspace_root_path = repo_root_path.parent
default_macro_data_dir_path = workspace_root_path / "1_data"
default_dtb3_csv_path = default_macro_data_dir_path / "DTB3.csv"

from alpha.data import FredSeriesSnapshot, load_daily_fred_series_snapshot
from alpha.engine.backtest import run_daily
from alpha.engine.report import save_results
from alpha.engine.strategy import Strategy


@dataclass(frozen=True)
class DefenseFirstConfig:
    defensive_asset_list: tuple[str, ...] = ("GLD", "UUP", "TLT", "DBC")
    fallback_asset: str = "SPY"
    benchmark_list: tuple[str, ...] = ("$SPX",)
    rank_weight_vec: tuple[float, ...] = (0.40, 0.30, 0.20, 0.10)
    momentum_lookback_month_vec: tuple[int, ...] = (1, 3, 6, 12)
    start_date_str: str = "2012-01-01"
    end_date_str: str | None = None
    dtb3_csv_path_str: str = str(default_dtb3_csv_path)
    dtb3_series_id_str: str = "DTB3"
    dtb3_mode_str: str = "backtest"
    dtb3_as_of_timestamp_ts: datetime | None = None

    def __post_init__(self):
        if len(self.rank_weight_vec) != len(self.defensive_asset_list):
            raise ValueError("rank_weight_vec length must match defensive_asset_list length.")
        if not np.isclose(sum(self.rank_weight_vec), 1.0, atol=1e-12):
            raise ValueError("rank_weight_vec must sum to 1.0.")
        if len(set(self.defensive_asset_list)) != len(self.defensive_asset_list):
            raise ValueError("defensive_asset_list contains duplicate symbols.")
        if self.fallback_asset in self.defensive_asset_list:
            raise ValueError("fallback_asset must not also appear in defensive_asset_list.")

    @property
    def tradeable_asset_list(self) -> tuple[str, ...]:
        return self.defensive_asset_list + (self.fallback_asset,)


DEFAULT_CONFIG = DefenseFirstConfig()


def default_trade_id_int() -> int:
    return -1


def _resolve_dtb3_as_of_timestamp_ts(config: DefenseFirstConfig) -> datetime:
    if config.dtb3_as_of_timestamp_ts is not None:
        return config.dtb3_as_of_timestamp_ts
    if config.end_date_str is not None:
        return pd.Timestamp(config.end_date_str).to_pydatetime()
    return datetime.now(tz=UTC)


def load_dtb3_snapshot(config: DefenseFirstConfig) -> FredSeriesSnapshot:
    """
    Load the DTB3 daily FRED series with audit metadata and freshness checks.

    Daily series semantics:

        y_t = DTB3_t / 100

    where `DTB3_t` is the annualized 3-month T-bill yield in percent.
    """
    dtb3_as_of_timestamp_ts = _resolve_dtb3_as_of_timestamp_ts(config)
    return load_daily_fred_series_snapshot(
        series_id_str=config.dtb3_series_id_str,
        cache_csv_path_str=config.dtb3_csv_path_str,
        as_of_ts=dtb3_as_of_timestamp_ts,
        mode_str=config.dtb3_mode_str,
    )


def load_cash_return_ser(
    dtb3_csv_path_str: str,
    dtb3_series_id_str: str = "DTB3",
    as_of_ts: datetime | None = None,
    mode_str: str = "backtest",
) -> pd.Series:
    """
    Load 3M T-bill annualized yield and convert it to an approximate 1-month
    cash return threshold.

    If y_t is the annualized yield in decimal form, the monthly return is:

        cash_return_t = (1 + y_t)^(1/12) - 1
    """
    dtb3_snapshot_obj = load_daily_fred_series_snapshot(
        series_id_str=dtb3_series_id_str,
        cache_csv_path_str=dtb3_csv_path_str,
        as_of_ts=as_of_ts,
        mode_str=mode_str,
    )
    dtb3_value_ser = dtb3_snapshot_obj.value_ser.astype(float)
    cash_return_ser = (1.0 + dtb3_value_ser / 100.0) ** (1.0 / 12.0) - 1.0
    cash_return_ser.name = "cash_return"
    return cash_return_ser.sort_index()


def load_cash_return_ser_and_snapshot(
    config: DefenseFirstConfig,
) -> tuple[pd.Series, FredSeriesSnapshot]:
    dtb3_snapshot_obj = load_dtb3_snapshot(config)
    dtb3_value_ser = dtb3_snapshot_obj.value_ser.astype(float)
    cash_return_ser = (1.0 + dtb3_value_ser / 100.0) ** (1.0 / 12.0) - 1.0
    cash_return_ser.name = "cash_return"
    return cash_return_ser.sort_index(), dtb3_snapshot_obj


def load_signal_close_df(
    symbol_list: Sequence[str],
    start_date_str: str,
    end_date_str: str | None = None,
) -> pd.DataFrame:
    """
    Load TOTALRETURN closes for signal formation.
    """
    signal_close_map: dict[str, pd.Series] = {}

    for symbol_str in tqdm(symbol_list, desc="loading signal closes"):
        price_df = norgatedata.price_timeseries(
            symbol_str,
            stock_price_adjustment_setting=norgatedata.StockPriceAdjustmentType.TOTALRETURN,
            padding_setting=norgatedata.PaddingType.ALLMARKETDAYS,
            start_date=start_date_str,
            end_date=end_date_str,
            timeseriesformat="pandas-dataframe",
        )
        if len(price_df) == 0:
            continue
        signal_close_map[symbol_str] = price_df["Close"]

    if len(signal_close_map) == 0:
        raise RuntimeError("No signal close data was loaded.")

    signal_close_df = pd.DataFrame(signal_close_map).sort_index()
    missing_symbol_list = [symbol_str for symbol_str in symbol_list if symbol_str not in signal_close_df.columns]
    if len(missing_symbol_list) > 0:
        raise RuntimeError(f"Missing signal data for symbols: {missing_symbol_list}")

    return signal_close_df


def load_execution_price_df(
    tradeable_asset_list: Sequence[str],
    benchmark_list: Sequence[str],
    start_date_str: str,
    end_date_str: str | None = None,
) -> pd.DataFrame:
    """
    Load OHLC bars used for fills and account valuation.

    Tradeable ETFs use CAPITALSPECIAL to keep fills realistic.
    Benchmarks use TOTALRETURN for comparison only.
    """
    execution_frame_list: list[pd.DataFrame] = []
    symbol_list = list(dict.fromkeys(list(tradeable_asset_list) + list(benchmark_list)))

    for symbol_str in tqdm(symbol_list, desc="loading execution prices"):
        if symbol_str in benchmark_list:
            adjustment_type = norgatedata.StockPriceAdjustmentType.TOTALRETURN
        else:
            adjustment_type = norgatedata.StockPriceAdjustmentType.CAPITALSPECIAL

        price_df = norgatedata.price_timeseries(
            symbol_str,
            stock_price_adjustment_setting=adjustment_type,
            padding_setting=norgatedata.PaddingType.ALLMARKETDAYS,
            start_date=start_date_str,
            end_date=end_date_str,
            timeseriesformat="pandas-dataframe",
        )
        if len(price_df) == 0:
            continue

        price_df.columns = pd.MultiIndex.from_tuples([(symbol_str, field_str) for field_str in price_df.columns])
        execution_frame_list.append(price_df)

    if len(execution_frame_list) == 0:
        raise RuntimeError("No execution price data was loaded.")

    execution_price_df = pd.concat(execution_frame_list, axis=1).sort_index()
    return execution_price_df


def compute_month_end_weight_df(
    signal_close_df: pd.DataFrame,
    cash_return_ser: pd.Series,
    config: DefenseFirstConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute paper-faithful month-end target weights.

    Returns:
    - momentum_score_df: monthly momentum scores for diagnostics.
    - month_end_weight_df: weights decided at month-end t, to be executed at the
      first trading day of month t+1.
    """
    # *** CRITICAL*** Signals are formed from month-end closes only. The
    # resulting decision at month-end t is not traded until the next month.
    monthly_close_df = signal_close_df.resample("ME").last()
    # *** CRITICAL*** The daily published DTB3 hurdle is sampled at the last
    # available observation within each month. This preserves the causal
    # "latest published value by decision time" rule.
    monthly_cash_return_ser = cash_return_ser.resample("ME").last()

    momentum_component_list: list[pd.DataFrame] = []
    for lookback_month_int in config.momentum_lookback_month_vec:
        component_df = monthly_close_df.pct_change(lookback_month_int, fill_method=None)
        momentum_component_list.append(component_df)

    momentum_score_df = sum(momentum_component_list) / float(len(momentum_component_list))
    combined_df = pd.concat([momentum_score_df, monthly_cash_return_ser], axis=1).dropna()

    tradeable_asset_list = list(config.tradeable_asset_list)
    month_end_weight_df = pd.DataFrame(0.0, index=combined_df.index, columns=tradeable_asset_list)

    for decision_date, row_ser in tqdm(combined_df.iterrows(), total=len(combined_df), desc="computing month-end weights"):
        defensive_score_ser = row_ser[list(config.defensive_asset_list)].astype(float)
        ranked_asset_list = defensive_score_ser.sort_values(ascending=False).index.tolist()
        cash_return_float = float(row_ser["cash_return"])

        target_weight_ser = pd.Series(0.0, index=tradeable_asset_list, dtype=float)
        for rank_idx_int, asset_str in enumerate(ranked_asset_list):
            slot_weight_float = float(config.rank_weight_vec[rank_idx_int])
            asset_score_float = float(defensive_score_ser.loc[asset_str])

            if asset_score_float > cash_return_float:
                target_weight_ser.loc[asset_str] = slot_weight_float
            else:
                target_weight_ser.loc[config.fallback_asset] += slot_weight_float

        if not np.isclose(target_weight_ser.sum(), 1.0, atol=1e-12):
            raise ValueError(
                f"Target weights must sum to 1.0. Found {target_weight_ser.sum():.12f} on {decision_date}."
            )

        month_end_weight_df.loc[decision_date] = target_weight_ser

    return momentum_score_df, month_end_weight_df


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
    for decision_date, target_weight_ser in month_end_weight_df.iterrows():
        # *** CRITICAL*** Decision date t is shifted to the first tradable open
        # in month t+1. This prevents same-bar look-ahead bias.
        next_month_period = (decision_date + pd.offsets.MonthBegin(1)).to_period("M")
        if next_month_period not in first_trading_day_ser.index:
            continue

        rebalance_date = pd.Timestamp(first_trading_day_ser.loc[next_month_period])
        rebalance_weight_map[rebalance_date] = target_weight_ser.copy()

    if len(rebalance_weight_map) == 0:
        raise RuntimeError("No rebalance dates were generated from the month-end decisions.")

    rebalance_weight_df = pd.DataFrame.from_dict(rebalance_weight_map, orient="index").sort_index()
    rebalance_weight_df.index.name = "rebalance_date"
    return rebalance_weight_df


def get_defense_first_data_with_snapshot(
    config: DefenseFirstConfig = DEFAULT_CONFIG,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, FredSeriesSnapshot]:
    """
    Load signal data, execution data, momentum scores, rebalance weights, and
    DTB3 snapshot metadata.
    """
    signal_close_df = load_signal_close_df(
        symbol_list=config.defensive_asset_list,
        start_date_str=config.start_date_str,
        end_date_str=config.end_date_str,
    )
    execution_price_df = load_execution_price_df(
        tradeable_asset_list=config.tradeable_asset_list,
        benchmark_list=config.benchmark_list,
        start_date_str=config.start_date_str,
        end_date_str=config.end_date_str,
    )
    cash_return_ser, dtb3_snapshot_obj = load_cash_return_ser_and_snapshot(config)
    momentum_score_df, month_end_weight_df = compute_month_end_weight_df(signal_close_df, cash_return_ser, config)
    rebalance_weight_df = map_month_end_weights_to_rebalance_open_df(month_end_weight_df, execution_price_df.index)

    return execution_price_df, momentum_score_df, month_end_weight_df, rebalance_weight_df, dtb3_snapshot_obj


def get_defense_first_data(
    config: DefenseFirstConfig = DEFAULT_CONFIG,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load signal data, execution data, momentum scores, and rebalance weights.
    """
    execution_price_df, momentum_score_df, month_end_weight_df, rebalance_weight_df, _ = (
        get_defense_first_data_with_snapshot(config)
    )
    return execution_price_df, momentum_score_df, month_end_weight_df, rebalance_weight_df


class DefenseFirstStrategy(Strategy):
    """
    Monthly tactical allocator for later TAA variants.

    Monthly decisions are formed at month-end and executed at the first
    tradable open of the next month. The chosen execution model is a simple
    pre-open OPG-style rebalance with prior-close share sizing.

        V^{budget}_t = V^{close}_{t-1}

        q^{target}_{i,t} = floor(V^{budget}_t * w_{i,t} / close_{i,t-1})

        fill_price_{i,t} = open_{i,t}

    The strategy intentionally accepts small cash drag and weight drift in
    exchange for simpler live alignment.
    """

    def __init__(
        self,
        name: str,
        benchmarks: Sequence[str],
        rebalance_weight_df: pd.DataFrame,
        tradeable_asset_list: Sequence[str],
        capital_base: float = 100_000,
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
        self.rebalance_weight_df = rebalance_weight_df.copy()
        self.tradeable_asset_list = list(tradeable_asset_list)
        self.trade_id_int = 0
        self.current_trade_map: defaultdict[str, int] = defaultdict(default_trade_id_int)

    def compute_signals(self, pricing_data: pd.DataFrame) -> pd.DataFrame:
        return pricing_data

    def iterate(self, data: pd.DataFrame, close: pd.DataFrame, open_prices: pd.Series):
        if close is None:
            return

        if self.current_bar not in self.rebalance_weight_df.index:
            return

        target_weight_ser = self.rebalance_weight_df.loc[self.current_bar].fillna(0.0)
        budget_value_float = float(self.previous_total_value)
        current_position_ser = self.get_positions().reindex(self.tradeable_asset_list, fill_value=0).astype(int)

        # Submit liquidations first so the rebalance funding path stays explicit.
        for asset_str in self.tradeable_asset_list:
            target_weight_float = float(target_weight_ser.get(asset_str, 0.0))
            current_share_int = int(current_position_ser.loc[asset_str])
            if target_weight_float != 0.0 or current_share_int == 0:
                continue

            self.order_target_value(asset_str, 0, trade_id=self.current_trade_map[asset_str])

        for asset_str in self.tradeable_asset_list:
            target_weight_float = float(target_weight_ser.get(asset_str, 0.0))
            if target_weight_float <= 0.0:
                continue

            current_share_int = int(current_position_ser.loc[asset_str])
            close_price_float = float(close[(asset_str, "Close")])
            if not np.isfinite(close_price_float) or close_price_float <= 0.0:
                raise RuntimeError(f"Invalid prior close for target asset {asset_str} on {self.previous_bar}.")

            # *** CRITICAL*** Rebalance share targets are anchored to the
            # previous_bar close and then filled at the current open.
            target_share_int = int(budget_value_float * target_weight_float / close_price_float)
            delta_share_int = target_share_int - current_share_int
            if delta_share_int == 0:
                continue

            if current_share_int == 0:
                self.trade_id_int += 1
                self.current_trade_map[asset_str] = self.trade_id_int

            self.order_target_percent(asset_str, target_weight_float, trade_id=self.current_trade_map[asset_str])


if __name__ == "__main__":
    config = DEFAULT_CONFIG

    execution_price_df, momentum_score_df, month_end_weight_df, rebalance_weight_df = get_defense_first_data(config)

    strategy = DefenseFirstStrategy(
        name="strategy_taa_df",
        benchmarks=config.benchmark_list,
        rebalance_weight_df=rebalance_weight_df,
        tradeable_asset_list=config.tradeable_asset_list,
        capital_base=100_000,
        slippage=0.00025,
        commission_per_share=0.005,
        commission_minimum=1.0,
    )
    strategy.show_taa_weights_report = True
    # strategy.daily_target_weights = rebalance_weight_df.copy()
    strategy.daily_target_weights = rebalance_weight_df.reindex(execution_price_df.index).ffill().dropna()


    calendar_idx = execution_price_df.index[execution_price_df.index >= rebalance_weight_df.index[0]]
    run_daily(strategy, execution_price_df, calendar_idx)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)

    print("First momentum scores:")
    display(momentum_score_df.dropna().head())

    print("First month-end decisions:")
    display(month_end_weight_df.head())

    print("First rebalance opens:")
    display(rebalance_weight_df.head())

    display(strategy.summary)
    display(strategy.summary_trades)

    save_results(strategy)







