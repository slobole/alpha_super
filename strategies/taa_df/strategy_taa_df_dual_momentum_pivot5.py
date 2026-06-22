"""
Dual Momentum Pivot5 tactical allocation strategy.

Core formulas
-------------
For each asset i at month-end decision close t:

    r_{L,i,t} = P^{signal}_{i,t} / P^{signal}_{i,t-L} - 1

    momentum_score_{i,t}
        = (r_{12,i,t} + r_{9,i,t} + r_{6,i,t} + r_{3,i,t} + r_{1,i,t}) / 5

Relative momentum filter:

    select the five assets with the highest momentum_score.

Absolute momentum filter:

    if momentum_score_{i,t} > 0:
        w_{i,t} = 20%
    else:
        w_{i,t} = 0 and the 20% slot remains in cash

Execution at the first tradable open of the next month:

    q^{target}_{i,t+1}
        = floor(V^{close}_{t} * w_{i,t} / P^{close}_{i,t})

SignalClose uses TOTALRETURN prices for ranking.
Execution OHLC uses CAPITALSPECIAL prices for fills and valuation.

This is an ETF-only Norgate implementation. It does not recreate the article's
pre-ETF proxy history.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
import sys
from typing import Sequence

import numpy as np
import pandas as pd
from IPython.display import display
from tqdm.auto import tqdm

repo_root_path = Path(__file__).resolve().parents[2]
repo_root_str = str(repo_root_path)
if repo_root_str not in sys.path:
    sys.path.insert(0, repo_root_str)

from alpha.engine.backtest import run_daily
from alpha.engine.friction_analysis import FrictionAnalysis
from alpha.engine.report import save_results
from alpha.engine.strategy import Strategy
from data.norgate_loader import (
    CAPITALSPECIAL_ADJUSTMENT_STR,
    TOTALRETURN_ADJUSTMENT_STR,
    load_price_timeseries,
)


STRATEGY_NAME_STR = "strategy_taa_df_dual_momentum_pivot5"


def default_trade_id_int() -> int:
    return -1


@dataclass(frozen=True)
class DualMomentumPivot5Config:
    asset_list: tuple[str, ...] = (
        "SPY",
        "VEA",
        "EEM",
        "TLT",
        "IEF",
        "BNDX",
        "VNQ",
        "DBC",
        "GLD",
    )
    benchmark_list: tuple[str, ...] = ("VT",)
    momentum_lookback_month_tuple: tuple[int, ...] = (12, 9, 6, 3, 1)
    selected_asset_count_int: int = 5
    absolute_momentum_threshold_float: float = 0.0
    start_date_str: str = "2004-01-01"
    end_date_str: str | None = None
    capital_base_float: float = 100_000.0
    slippage_float: float = 0.00025
    commission_per_share_float: float = 0.005
    commission_minimum_float: float = 1.0

    def __post_init__(self):
        if len(self.asset_list) == 0:
            raise ValueError("asset_list must not be empty.")
        if len(set(self.asset_list)) != len(self.asset_list):
            raise ValueError("asset_list contains duplicate symbols.")
        if len(set(self.benchmark_list)) != len(self.benchmark_list):
            raise ValueError("benchmark_list contains duplicate symbols.")
        if self.selected_asset_count_int <= 0:
            raise ValueError("selected_asset_count_int must be positive.")
        if self.selected_asset_count_int > len(self.asset_list):
            raise ValueError("selected_asset_count_int must be <= len(asset_list).")
        if len(self.momentum_lookback_month_tuple) == 0:
            raise ValueError("momentum_lookback_month_tuple must not be empty.")
        if any(int(lookback_month_int) <= 0 for lookback_month_int in self.momentum_lookback_month_tuple):
            raise ValueError("All momentum lookbacks must be positive.")
        if not np.isfinite(self.absolute_momentum_threshold_float):
            raise ValueError("absolute_momentum_threshold_float must be finite.")
        if self.capital_base_float <= 0.0:
            raise ValueError("capital_base_float must be positive.")
        if self.slippage_float < 0.0:
            raise ValueError("slippage_float must be non-negative.")
        if self.commission_per_share_float < 0.0:
            raise ValueError("commission_per_share_float must be non-negative.")
        if self.commission_minimum_float < 0.0:
            raise ValueError("commission_minimum_float must be non-negative.")

    @property
    def slot_weight_float(self) -> float:
        return 1.0 / float(self.selected_asset_count_int)


DEFAULT_CONFIG = DualMomentumPivot5Config()


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
        price_df = load_price_timeseries(
            symbol_str,
            adjustment_str=TOTALRETURN_ADJUSTMENT_STR,
            start_date_str=start_date_str,
            end_date_str=end_date_str,
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

    Tradeable ETFs use CAPITALSPECIAL. Benchmarks use TOTALRETURN.
    """
    execution_frame_list: list[pd.DataFrame] = []
    symbol_list = list(dict.fromkeys(list(tradeable_asset_list) + list(benchmark_list)))

    for symbol_str in tqdm(symbol_list, desc="loading execution prices"):
        adjustment_str = (
            TOTALRETURN_ADJUSTMENT_STR
            if symbol_str in benchmark_list
            else CAPITALSPECIAL_ADJUSTMENT_STR
        )
        price_df = load_price_timeseries(
            symbol_str,
            adjustment_str=adjustment_str,
            start_date_str=start_date_str,
            end_date_str=end_date_str,
        )
        if len(price_df) == 0:
            continue

        price_df.columns = pd.MultiIndex.from_tuples([(symbol_str, field_str) for field_str in price_df.columns])
        execution_frame_list.append(price_df)

    if len(execution_frame_list) == 0:
        raise RuntimeError("No execution price data was loaded.")

    return pd.concat(execution_frame_list, axis=1).sort_index()


def compute_month_end_pivot5_weight_df(
    signal_close_df: pd.DataFrame,
    config: DualMomentumPivot5Config = DEFAULT_CONFIG,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute Pivot5 month-end momentum scores and target weights.

    Returns:
    - momentum_score_df: monthly blended momentum scores.
    - month_end_weight_df: tradeable asset weights decided at month-end t.
      Residual weight is implicit cash.
    """
    missing_symbol_list = [
        symbol_str for symbol_str in config.asset_list if symbol_str not in signal_close_df.columns
    ]
    if len(missing_symbol_list) > 0:
        raise RuntimeError(f"Missing signal close columns for symbols: {missing_symbol_list}")

    # *** CRITICAL*** Signals are formed from completed month-end signal
    # closes only. These weights are not executable until the next month.
    monthly_close_df = signal_close_df.loc[:, list(config.asset_list)].astype(float).resample("ME").last()

    momentum_component_list: list[pd.DataFrame] = []
    for lookback_month_int in config.momentum_lookback_month_tuple:
        # *** CRITICAL*** pct_change uses only trailing month-end closes.
        # fill_method=None prevents hidden forward fill across missing ETF data.
        component_df = monthly_close_df.pct_change(int(lookback_month_int), fill_method=None)
        momentum_component_list.append(component_df)

    momentum_score_df = sum(momentum_component_list) / float(len(momentum_component_list))
    # *** CRITICAL*** Require every ETF to have every lookback before ranking.
    # This avoids comparing partial-history assets against full-history assets.
    eligible_score_df = momentum_score_df.dropna(how="any")

    month_end_weight_df = pd.DataFrame(0.0, index=eligible_score_df.index, columns=list(config.asset_list))

    for decision_date_ts, score_ser in tqdm(
        eligible_score_df.iterrows(),
        total=len(eligible_score_df),
        desc="computing Pivot5 month-end weights",
    ):
        ranked_score_ser = score_ser.astype(float).sort_values(ascending=False)
        selected_asset_list = ranked_score_ser.head(config.selected_asset_count_int).index.tolist()

        target_weight_ser = pd.Series(0.0, index=list(config.asset_list), dtype=float)
        for asset_str in selected_asset_list:
            score_float = float(ranked_score_ser.loc[asset_str])
            if score_float > config.absolute_momentum_threshold_float:
                target_weight_ser.loc[asset_str] = config.slot_weight_float

        target_weight_sum_float = float(target_weight_ser.sum())
        if target_weight_sum_float > 1.0 + 1e-12:
            raise ValueError(
                f"Target weights exceed 1.0 on {decision_date_ts}: {target_weight_sum_float:.12f}."
            )

        month_end_weight_df.loc[decision_date_ts] = target_weight_ser

    if len(month_end_weight_df) == 0:
        raise RuntimeError("No month-end Pivot5 weights were generated.")

    return momentum_score_df, month_end_weight_df


def map_month_end_pivot5_weights_to_rebalance_open_df(
    month_end_weight_df: pd.DataFrame,
    execution_index: pd.DatetimeIndex,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Map each month-end decision to the first tradable open of the next month.

    The schedule stores the actual tradable decision close, not the calendar
    month-end label.
    """
    execution_index = pd.DatetimeIndex(execution_index).sort_values()
    rebalance_weight_map: dict[pd.Timestamp, pd.Series] = {}
    rebalance_schedule_map: dict[pd.Timestamp, pd.Timestamp] = {}

    for month_end_ts, target_weight_ser in month_end_weight_df.iterrows():
        month_period_obj = pd.Timestamp(month_end_ts).to_period("M")
        decision_month_idx = execution_index[execution_index.to_period("M") == month_period_obj]
        if len(decision_month_idx) == 0:
            continue

        next_month_period_obj = (pd.Timestamp(month_end_ts) + pd.offsets.MonthBegin(1)).to_period("M")
        next_month_idx = execution_index[execution_index.to_period("M") == next_month_period_obj]
        if len(next_month_idx) == 0:
            continue

        # *** CRITICAL*** Calendar month-end may be a weekend or holiday.
        # The signal is the last tradable close in the decision month.
        decision_close_ts = pd.Timestamp(decision_month_idx[-1])
        # *** CRITICAL*** Month-end decision t is shifted to the first
        # tradable open in month t+1. This prevents same-bar lookahead.
        rebalance_open_ts = pd.Timestamp(next_month_idx[0])
        rebalance_weight_map[rebalance_open_ts] = target_weight_ser.copy()
        rebalance_schedule_map[rebalance_open_ts] = decision_close_ts

    if len(rebalance_weight_map) == 0:
        raise RuntimeError("No Pivot5 rebalance dates were generated.")

    rebalance_weight_df = pd.DataFrame.from_dict(rebalance_weight_map, orient="index").sort_index()
    rebalance_weight_df.index.name = "rebalance_date"
    rebalance_schedule_df = pd.DataFrame(
        {"decision_date_ts": pd.Series(rebalance_schedule_map, dtype="datetime64[ns]")}
    ).sort_index()
    rebalance_schedule_df.index.name = "rebalance_date"
    return rebalance_weight_df, rebalance_schedule_df


def map_month_end_pivot5_weight_to_decision_close_df(
    month_end_weight_df: pd.DataFrame,
    execution_index: pd.DatetimeIndex,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build order-intent weights indexed by the actual decision close.
    """
    execution_index = pd.DatetimeIndex(execution_index).sort_values()
    decision_weight_map: dict[pd.Timestamp, pd.Series] = {}

    for month_end_ts, target_weight_ser in month_end_weight_df.iterrows():
        month_period_obj = pd.Timestamp(month_end_ts).to_period("M")
        decision_month_idx = execution_index[execution_index.to_period("M") == month_period_obj]
        if len(decision_month_idx) == 0:
            continue

        # *** CRITICAL*** ExecutionTiming uses this date as the signal bar.
        # It must be the last tradable close in the decision month.
        decision_close_ts = pd.Timestamp(decision_month_idx[-1])
        decision_weight_map[decision_close_ts] = target_weight_ser.copy()

    if len(decision_weight_map) == 0:
        raise RuntimeError("No Pivot5 decision-close dates were generated.")

    decision_weight_df = pd.DataFrame.from_dict(decision_weight_map, orient="index").sort_index()
    decision_weight_df.index.name = "decision_close_date"
    decision_schedule_df = pd.DataFrame(
        {"decision_date_ts": pd.Series(decision_weight_df.index, index=decision_weight_df.index)}
    )
    decision_schedule_df.index.name = "decision_close_date"
    return decision_weight_df, decision_schedule_df


def get_dual_momentum_pivot5_data(
    config: DualMomentumPivot5Config = DEFAULT_CONFIG,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    signal_close_df = load_signal_close_df(
        symbol_list=config.asset_list,
        start_date_str=config.start_date_str,
        end_date_str=config.end_date_str,
    )
    execution_price_df = load_execution_price_df(
        tradeable_asset_list=config.asset_list,
        benchmark_list=config.benchmark_list,
        start_date_str=config.start_date_str,
        end_date_str=config.end_date_str,
    )
    momentum_score_df, month_end_weight_df = compute_month_end_pivot5_weight_df(
        signal_close_df=signal_close_df,
        config=config,
    )
    rebalance_weight_df, rebalance_schedule_df = map_month_end_pivot5_weights_to_rebalance_open_df(
        month_end_weight_df=month_end_weight_df,
        execution_index=pd.DatetimeIndex(execution_price_df.index),
    )
    return (
        execution_price_df,
        momentum_score_df,
        month_end_weight_df,
        rebalance_weight_df,
        rebalance_schedule_df,
    )


class DualMomentumPivot5Strategy(Strategy):
    """
    Monthly Pivot5 allocator with implicit residual cash.

    For selected asset i at rebalance open t+1:

        q^{target}_{i,t+1}
            = floor(V^{close}_{t} * w_{i,t} / P^{close}_{i,t})

    The engine fills market orders at the current open.
    """

    def __init__(
        self,
        name: str,
        benchmarks: Sequence[str],
        rebalance_weight_df: pd.DataFrame,
        rebalance_schedule_df: pd.DataFrame,
        asset_list: Sequence[str],
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
        if len(rebalance_weight_df) == 0:
            raise ValueError("rebalance_weight_df must not be empty.")
        if len(rebalance_schedule_df) == 0:
            raise ValueError("rebalance_schedule_df must not be empty.")
        if "decision_date_ts" not in rebalance_schedule_df.columns:
            raise ValueError("rebalance_schedule_df must contain decision_date_ts.")

        self.asset_list = list(asset_list)
        self.rebalance_weight_df = rebalance_weight_df.copy().sort_index()
        self.rebalance_schedule_df = rebalance_schedule_df.copy().sort_index()
        self.trade_id_int = 0
        self.current_trade_map: defaultdict[str, int] = defaultdict(default_trade_id_int)

    def compute_signals(self, pricing_data: pd.DataFrame) -> pd.DataFrame:
        return pricing_data

    def iterate(self, data: pd.DataFrame, close: pd.Series, open_prices: pd.Series):
        if close is None:
            return

        if self.current_bar not in self.rebalance_weight_df.index:
            return

        decision_date_ts = pd.Timestamp(self.rebalance_schedule_df.loc[self.current_bar, "decision_date_ts"])
        if self.previous_bar is None or pd.Timestamp(self.previous_bar) != decision_date_ts:
            raise RuntimeError(
                "Schedule misalignment: Pivot5 rebalance must use the stored "
                f"decision close {decision_date_ts.date()}, found previous_bar={self.previous_bar}."
            )

        # Missing rebalance columns mean explicit zero target, not a carried
        # prior target. The previous_bar check above anchors the timing.
        target_weight_ser = self.rebalance_weight_df.loc[self.current_bar].reindex(self.asset_list).fillna(0.0)
        if (target_weight_ser < -1e-12).any():
            raise RuntimeError(f"Negative target weight found on {self.current_bar}.")
        target_weight_sum_float = float(target_weight_ser.sum())
        if target_weight_sum_float > 1.0 + 1e-12:
            raise RuntimeError(
                f"Target weights exceed 1.0 on {self.current_bar}: {target_weight_sum_float:.12f}."
            )

        budget_value_float = float(self.previous_total_value)
        current_position_ser = self.get_positions().reindex(self.asset_list, fill_value=0).astype(int)

        # Submit liquidations first so the implicit cash sleeve is explicit.
        for asset_str in self.asset_list:
            target_weight_float = float(target_weight_ser.loc[asset_str])
            current_share_int = int(current_position_ser.loc[asset_str])
            if target_weight_float > 0.0 or current_share_int == 0:
                continue

            self.order_target_value(asset_str, 0, trade_id=self.current_trade_map[asset_str])
            self.current_trade_map[asset_str] = default_trade_id_int()

        for asset_str in self.asset_list:
            target_weight_float = float(target_weight_ser.loc[asset_str])
            if target_weight_float <= 0.0:
                continue

            close_key_tuple = (asset_str, "Close")
            if close_key_tuple not in close.index:
                raise RuntimeError(f"Missing prior close for target asset {asset_str} on {self.previous_bar}.")

            close_price_float = float(close[close_key_tuple])
            if not np.isfinite(close_price_float) or close_price_float <= 0.0:
                raise RuntimeError(f"Invalid prior close for target asset {asset_str} on {self.previous_bar}.")

            current_share_int = int(current_position_ser.loc[asset_str])
            # *** CRITICAL*** Rebalance share targets are sized from the
            # previous_bar decision close and then filled at the current open.
            target_share_int = int(budget_value_float * target_weight_float / close_price_float)
            delta_share_int = target_share_int - current_share_int
            if delta_share_int == 0:
                continue

            if current_share_int == 0 or self.current_trade_map[asset_str] == default_trade_id_int():
                self.trade_id_int += 1
                self.current_trade_map[asset_str] = self.trade_id_int

            self.order_target_percent(
                asset_str,
                target_weight_float,
                trade_id=self.current_trade_map[asset_str],
            )


def _build_pivot5_strategy(
    strategy_name_str: str,
    config: DualMomentumPivot5Config,
    rebalance_weight_df: pd.DataFrame,
    rebalance_schedule_df: pd.DataFrame,
    capital_base_float: float,
) -> DualMomentumPivot5Strategy:
    return DualMomentumPivot5Strategy(
        name=strategy_name_str,
        benchmarks=config.benchmark_list,
        rebalance_weight_df=rebalance_weight_df,
        rebalance_schedule_df=rebalance_schedule_df,
        asset_list=config.asset_list,
        capital_base=capital_base_float,
        slippage=config.slippage_float,
        commission_per_share=config.commission_per_share_float,
        commission_minimum=config.commission_minimum_float,
    )


def _run_strategy_from_weight_df(
    strategy_obj: DualMomentumPivot5Strategy,
    execution_price_df: pd.DataFrame,
    rebalance_weight_df: pd.DataFrame,
    backtest_start_date_str: str | None = None,
    show_progress_bool: bool = False,
) -> None:
    strategy_obj.show_taa_weights_report = True

    # *** CRITICAL*** This forward fill is for post-run weight diagnostics
    # only. Execution still uses discrete rebalance dates inside iterate().
    strategy_obj.daily_target_weights = rebalance_weight_df.reindex(execution_price_df.index).ffill().dropna()

    calendar_start_ts = pd.Timestamp(rebalance_weight_df.index[0])
    if backtest_start_date_str is not None:
        calendar_start_ts = max(calendar_start_ts, pd.Timestamp(backtest_start_date_str))

    # *** CRITICAL*** Full pre-start data remains loaded for signal warmup,
    # while executable backtest bars start at the requested deployment date.
    calendar_idx = execution_price_df.index[execution_price_df.index >= calendar_start_ts]
    run_daily(
        strategy_obj,
        execution_price_df,
        calendar_idx,
        show_progress=show_progress_bool,
        show_signal_progress_bool=show_progress_bool,
    )


def run_variant(
    show_display_bool: bool = True,
    save_results_bool: bool = True,
    output_dir_str: str = "results",
    backtest_start_date_str: str | None = None,
    capital_base_float: float = DEFAULT_CONFIG.capital_base_float,
    end_date_str: str | None = None,
    config: DualMomentumPivot5Config = DEFAULT_CONFIG,
    strategy_name_str: str = STRATEGY_NAME_STR,
) -> DualMomentumPivot5Strategy:
    config = config if end_date_str is None else replace(config, end_date_str=end_date_str)
    (
        execution_price_df,
        momentum_score_df,
        month_end_weight_df,
        rebalance_weight_df,
        rebalance_schedule_df,
    ) = get_dual_momentum_pivot5_data(config=config)

    strategy_obj = _build_pivot5_strategy(
        strategy_name_str=strategy_name_str,
        config=config,
        rebalance_weight_df=rebalance_weight_df,
        rebalance_schedule_df=rebalance_schedule_df,
        capital_base_float=capital_base_float,
    )
    _run_strategy_from_weight_df(
        strategy_obj=strategy_obj,
        execution_price_df=execution_price_df,
        rebalance_weight_df=rebalance_weight_df,
        backtest_start_date_str=backtest_start_date_str,
    )

    if show_display_bool:
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 1000)
        print("First Pivot5 momentum scores:")
        display(momentum_score_df.dropna().head())
        print("First Pivot5 month-end decisions:")
        display(month_end_weight_df.head())
        print("First Pivot5 rebalance opens:")
        display(rebalance_weight_df.head())
        display(strategy_obj.summary)
        display(strategy_obj.summary_trades)

    if save_results_bool:
        save_results(strategy_obj, output_dir=output_dir_str)

    return strategy_obj


def build_friction_analysis_inputs(
    show_display_bool: bool = False,
    backtest_start_date_str: str | None = None,
    capital_base_float: float = DEFAULT_CONFIG.capital_base_float,
    end_date_str: str | None = None,
    config: DualMomentumPivot5Config = DEFAULT_CONFIG,
    strategy_name_str: str = STRATEGY_NAME_STR,
) -> dict[str, object]:
    config = config if end_date_str is None else replace(config, end_date_str=end_date_str)
    (
        execution_price_df,
        _momentum_score_df,
        _month_end_weight_df,
        rebalance_weight_df,
        rebalance_schedule_df,
    ) = get_dual_momentum_pivot5_data(config=config)
    strategy_obj = _build_pivot5_strategy(
        strategy_name_str=strategy_name_str,
        config=config,
        rebalance_weight_df=rebalance_weight_df,
        rebalance_schedule_df=rebalance_schedule_df,
        capital_base_float=capital_base_float,
    )

    # *** CRITICAL*** FrictionAnalysis reuses the default next-open completed
    # ledger. Month-end signal timing is not changed by capacity diagnostics.
    _run_strategy_from_weight_df(
        strategy_obj=strategy_obj,
        execution_price_df=execution_price_df,
        rebalance_weight_df=rebalance_weight_df,
        backtest_start_date_str=backtest_start_date_str,
    )

    if show_display_bool:
        display(strategy_obj.summary)

    return {
        "strategy_obj": strategy_obj,
        "pricing_data_df": execution_price_df,
        "execution_policy_str": "MOO",
    }


def run_friction_analysis(
    save_results_bool: bool = True,
    output_dir_str: str = "results",
    show_display_bool: bool = False,
    backtest_start_date_str: str | None = None,
    capital_base_float: float = DEFAULT_CONFIG.capital_base_float,
    end_date_str: str | None = None,
    config: DualMomentumPivot5Config = DEFAULT_CONFIG,
    strategy_name_str: str = STRATEGY_NAME_STR,
):
    friction_input_dict = build_friction_analysis_inputs(
        show_display_bool=show_display_bool,
        backtest_start_date_str=backtest_start_date_str,
        capital_base_float=capital_base_float,
        end_date_str=end_date_str,
        config=config,
        strategy_name_str=strategy_name_str,
    )
    friction_analysis_obj = FrictionAnalysis(
        strategy_obj=friction_input_dict["strategy_obj"],
        pricing_data_df=friction_input_dict["pricing_data_df"],
        execution_policy_str=friction_input_dict["execution_policy_str"],
        output_dir_str=output_dir_str,
        save_output_bool=save_results_bool,
    )
    return friction_analysis_obj.run()


def build_execution_timing_analysis_inputs(
    config: DualMomentumPivot5Config = DEFAULT_CONFIG,
    strategy_name_str: str = STRATEGY_NAME_STR,
) -> dict[str, object]:
    (
        execution_price_df,
        _momentum_score_df,
        month_end_weight_df,
        rebalance_weight_df,
        _rebalance_schedule_df,
    ) = get_dual_momentum_pivot5_data(config=config)
    decision_weight_df, decision_schedule_df = map_month_end_pivot5_weight_to_decision_close_df(
        month_end_weight_df=month_end_weight_df,
        execution_index=pd.DatetimeIndex(execution_price_df.index),
    )

    def strategy_factory_fn():
        strategy_obj = _build_pivot5_strategy(
            strategy_name_str=strategy_name_str,
            config=config,
            rebalance_weight_df=decision_weight_df,
            rebalance_schedule_df=decision_schedule_df,
            capital_base_float=config.capital_base_float,
        )
        strategy_obj.show_taa_weights_report = True
        # *** CRITICAL*** Reporting-only target weights stay aligned to the
        # default next-open ledger, not the diagnostic signal-bar schedule.
        strategy_obj.daily_target_weights = rebalance_weight_df.reindex(execution_price_df.index).ffill().dropna()
        return strategy_obj

    calendar_idx = execution_price_df.index[
        execution_price_df.index >= pd.Timestamp(rebalance_weight_df.index[0])
    ]

    return {
        "strategy_factory_fn": strategy_factory_fn,
        "pricing_data_df": execution_price_df,
        "calendar_idx": pd.DatetimeIndex(calendar_idx),
        "order_generation_mode_str": "signal_bar",
        "risk_model_str": "taa_rebalance",
        "entry_timing_str_tuple": ("same_close_moc", "next_open", "next_close"),
        "exit_timing_str_tuple": ("same_close_moc", "next_open", "next_close"),
        "default_entry_timing_str": "next_open",
        "default_exit_timing_str": "next_open",
    }


if __name__ == "__main__":
    run_variant()
