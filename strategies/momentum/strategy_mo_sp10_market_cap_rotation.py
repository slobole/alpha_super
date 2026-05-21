"""
Research-only monthly S&P 500 top-10 market-cap rotation.

Core rule
---------
For each month-end decision date t:

    eligible_{i,t}
        = 1[PIT_SP500_member_{i,t} = 1 and market_cap_{i,t} is finite]

    selected_t
        = 10 largest eligible names by point-in-time market_cap_{i,t}

    target_weight_{i,t}
        = 1 / 10    if i in selected_t
        = 0         otherwise

Execution mapping:

    decision_date_t
        = actual month-end trading close where the PIT market-cap snapshot is known

    execution_date_t
        = next tradable open after decision_date_t

This module is intentionally research-only. The local Norgate Python API exposes
current `sharesoutstanding`, not a historical point-in-time share-count series,
so this strategy requires an explicit point-in-time market-cap matrix. It must
not silently derive historical market caps from current shares outstanding.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from IPython.display import display

from alpha.engine.backtest import run_daily
from alpha.engine.report import save_results
from alpha.engine.strategy import Strategy
from data.norgate_loader import build_index_constituent_matrix, load_raw_prices
from strategies.momentum.strategy_mo_atr_normalized_ndx import (
    audit_pit_universe_df,
    get_asof_universe_membership_ser,
    get_monthly_decision_close_df,
    map_month_end_decision_dates_to_rebalance_schedule_df,
)


def default_trade_id_int() -> int:
    return -1


@dataclass(frozen=True)
class TrueSp10RotationConfig:
    indexname_str: str = "S&P 500"
    benchmark_list: tuple[str, ...] = ("$SPX",)
    history_start_date_str: str = "1998-01-01"
    backtest_start_date_str: str = "2000-01-01"
    end_date_str: str | None = None
    max_positions_int: int = 10
    capital_base_float: float = 100_000.0
    slippage_float: float = 0.00025
    commission_per_share_float: float = 0.005
    commission_minimum_float: float = 1.0
    market_cap_csv_path_str: str | None = None

    def __post_init__(self) -> None:
        if not self.indexname_str:
            raise ValueError("indexname_str must not be empty.")
        if len(self.benchmark_list) == 0:
            raise ValueError("benchmark_list must not be empty.")
        if pd.Timestamp(self.history_start_date_str) >= pd.Timestamp(self.backtest_start_date_str):
            raise ValueError("history_start_date_str must be earlier than backtest_start_date_str.")
        if self.max_positions_int <= 0:
            raise ValueError("max_positions_int must be positive.")
        if self.capital_base_float <= 0.0:
            raise ValueError("capital_base_float must be positive.")
        if self.slippage_float < 0.0:
            raise ValueError("slippage_float must be non-negative.")
        if self.commission_per_share_float < 0.0:
            raise ValueError("commission_per_share_float must be non-negative.")
        if self.commission_minimum_float < 0.0:
            raise ValueError("commission_minimum_float must be non-negative.")


DEFAULT_CONFIG = TrueSp10RotationConfig()


__all__ = [
    "DEFAULT_CONFIG",
    "TrueSp10RotationConfig",
    "TrueSp10RotationStrategy",
    "compute_month_end_sp10_weight_df",
    "get_true_sp10_rotation_data",
    "load_point_in_time_market_cap_df",
    "map_month_end_sp10_weights_to_rebalance_open_df",
    "run_variant",
]


def load_point_in_time_market_cap_df(market_cap_csv_path_str: str) -> pd.DataFrame:
    """
    Load a point-in-time market-cap matrix.

    Supported CSV shapes:
    - wide: first column is date, remaining columns are symbols.
    - long: columns include date, symbol, market_cap.
    """
    market_cap_csv_path = Path(market_cap_csv_path_str)
    if not market_cap_csv_path.exists():
        raise FileNotFoundError(f"Point-in-time market-cap CSV not found: {market_cap_csv_path}")

    raw_market_cap_df = pd.read_csv(market_cap_csv_path)
    if len(raw_market_cap_df.columns) == 0:
        raise ValueError("market-cap CSV must contain columns.")

    normalized_column_map = {column_str.lower().strip(): column_str for column_str in raw_market_cap_df.columns}
    long_required_column_set = {"date", "symbol", "market_cap"}
    if long_required_column_set.issubset(normalized_column_map):
        date_column_str = normalized_column_map["date"]
        symbol_column_str = normalized_column_map["symbol"]
        market_cap_column_str = normalized_column_map["market_cap"]
        market_cap_df = raw_market_cap_df.pivot(
            index=date_column_str,
            columns=symbol_column_str,
            values=market_cap_column_str,
        )
    else:
        date_column_str = raw_market_cap_df.columns[0]
        market_cap_df = raw_market_cap_df.set_index(date_column_str)

    market_cap_df.index = pd.to_datetime(market_cap_df.index).normalize()
    market_cap_df = market_cap_df.sort_index()
    if market_cap_df.index.has_duplicates:
        raise ValueError("market_cap_df index must not contain duplicate dates.")

    market_cap_df.columns = market_cap_df.columns.astype(str)
    return market_cap_df.apply(pd.to_numeric, errors="coerce")


def _validate_market_cap_and_universe_inputs(
    market_cap_df: pd.DataFrame,
    universe_df: pd.DataFrame,
    max_positions_int: int,
) -> None:
    if len(market_cap_df) == 0:
        raise ValueError("market_cap_df must not be empty.")
    if len(universe_df) == 0:
        raise ValueError("universe_df must not be empty.")
    if max_positions_int <= 0:
        raise ValueError("max_positions_int must be positive.")
    if not market_cap_df.index.is_monotonic_increasing:
        raise ValueError("market_cap_df index must be sorted.")
    if market_cap_df.index.has_duplicates:
        raise ValueError("market_cap_df index must not contain duplicates.")
    if not universe_df.index.is_monotonic_increasing:
        raise ValueError("universe_df index must be sorted.")
    if universe_df.index.has_duplicates:
        raise ValueError("universe_df index must not contain duplicates.")


def compute_month_end_sp10_weight_df(
    market_cap_df: pd.DataFrame,
    universe_df: pd.DataFrame,
    max_positions_int: int = DEFAULT_CONFIG.max_positions_int,
) -> pd.DataFrame:
    """
    Compute month-end equal-weight targets for the true SP10 rotation.

    The selection formula is:

        selected_t = top_n({i | PIT_SP500_member_{i,t}=1}, market_cap_{i,t})

        weight_{i,t} = 1 / n if i in selected_t, else 0
    """
    market_cap_df = market_cap_df.copy().sort_index()
    market_cap_df.columns = market_cap_df.columns.astype(str)
    universe_df = universe_df.copy().sort_index()
    universe_df.columns = universe_df.columns.astype(str)
    _validate_market_cap_and_universe_inputs(
        market_cap_df=market_cap_df,
        universe_df=universe_df,
        max_positions_int=max_positions_int,
    )

    # *** CRITICAL*** Market-cap decisions are sampled only from the actual
    # month-end date present in the PIT market-cap matrix. Do not resample to a
    # synthetic calendar date or fill from future rows.
    monthly_market_cap_df = get_monthly_decision_close_df(price_close_df=market_cap_df)
    target_weight_df = pd.DataFrame(
        0.0,
        index=monthly_market_cap_df.index,
        columns=market_cap_df.columns,
        dtype=float,
    )

    for decision_date_ts, market_cap_row_ser in monthly_market_cap_df.iterrows():
        universe_member_ser = get_asof_universe_membership_ser(
            universe_df=universe_df,
            decision_date_ts=pd.Timestamp(decision_date_ts),
        )
        active_symbol_list = universe_member_ser[universe_member_ser == 1].index.astype(str).tolist()
        candidate_market_cap_ser = market_cap_row_ser.reindex(active_symbol_list)
        candidate_market_cap_ser = pd.to_numeric(candidate_market_cap_ser, errors="coerce")
        candidate_market_cap_ser = candidate_market_cap_ser.replace([np.inf, -np.inf], np.nan).dropna()
        candidate_market_cap_ser = candidate_market_cap_ser[candidate_market_cap_ser > 0.0]

        if len(candidate_market_cap_ser) < max_positions_int:
            raise RuntimeError(
                f"Only {len(candidate_market_cap_ser)} active PIT market-cap values are available "
                f"on {pd.Timestamp(decision_date_ts).date()}, but max_positions_int={max_positions_int}."
            )

        candidate_rank_df = pd.DataFrame(
            {
                "market_cap_float": candidate_market_cap_ser.astype(float),
                "symbol_str": candidate_market_cap_ser.index.astype(str),
            },
            index=candidate_market_cap_ser.index.astype(str),
        )
        selected_symbol_list = (
            candidate_rank_df.sort_values(
                by=["market_cap_float", "symbol_str"],
                ascending=[False, True],
                kind="mergesort",
            )
            .iloc[:max_positions_int]
            .index.astype(str)
            .tolist()
        )

        target_weight_float = 1.0 / float(max_positions_int)
        target_weight_df.loc[decision_date_ts, selected_symbol_list] = target_weight_float

    target_weight_df.index.name = "decision_date_ts"
    return target_weight_df


def map_month_end_sp10_weights_to_rebalance_open_df(
    month_end_weight_df: pd.DataFrame,
    execution_index: pd.DatetimeIndex,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Map month-end SP10 decisions to the next tradable open.
    """
    rebalance_schedule_df = map_month_end_decision_dates_to_rebalance_schedule_df(
        decision_date_index=pd.DatetimeIndex(month_end_weight_df.index),
        execution_index=pd.DatetimeIndex(execution_index),
    )
    decision_date_index = pd.DatetimeIndex(rebalance_schedule_df["decision_date_ts"])
    rebalance_weight_df = month_end_weight_df.reindex(decision_date_index).copy()
    rebalance_weight_df.index = pd.DatetimeIndex(rebalance_schedule_df.index)
    rebalance_weight_df.index.name = "rebalance_date"
    return rebalance_weight_df, rebalance_schedule_df


def get_true_sp10_rotation_data(
    config: TrueSp10RotationConfig = DEFAULT_CONFIG,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if config.market_cap_csv_path_str is None:
        raise RuntimeError(
            "True SP10 rotation requires a point-in-time historical market-cap CSV. "
            "Do not use norgatedata.sharesoutstanding here; that API returns current fundamentals only."
        )

    market_cap_df = load_point_in_time_market_cap_df(config.market_cap_csv_path_str)
    _, raw_universe_df = build_index_constituent_matrix(indexname=config.indexname_str)

    history_start_ts = pd.Timestamp(config.history_start_date_str)
    backtest_start_ts = pd.Timestamp(config.backtest_start_date_str)
    filtered_universe_df = raw_universe_df.loc[raw_universe_df.index >= history_start_ts].copy()
    filtered_market_cap_df = market_cap_df.loc[market_cap_df.index >= history_start_ts].copy()
    if config.end_date_str is not None:
        end_date_ts = pd.Timestamp(config.end_date_str)
        filtered_market_cap_df = filtered_market_cap_df.loc[filtered_market_cap_df.index <= end_date_ts]

    month_end_weight_df = compute_month_end_sp10_weight_df(
        market_cap_df=filtered_market_cap_df,
        universe_df=filtered_universe_df,
        max_positions_int=config.max_positions_int,
    )
    month_end_weight_df = month_end_weight_df.loc[
        month_end_weight_df.index >= pd.Timestamp(config.backtest_start_date_str)
    ]
    selected_symbol_list = month_end_weight_df.columns[
        month_end_weight_df.sum(axis=0) > 0.0
    ].astype(str).tolist()
    if len(selected_symbol_list) == 0:
        raise RuntimeError("No SP10 selected symbols were produced for the requested backtest window.")

    pricing_data_df = load_raw_prices(
        symbols=selected_symbol_list,
        benchmarks=list(config.benchmark_list),
        start_date=config.history_start_date_str,
        end_date=config.end_date_str,
    )
    loaded_symbol_list = [
        symbol_str
        for symbol_str in selected_symbol_list
        if symbol_str in pricing_data_df.columns.get_level_values(0)
    ]
    audited_universe_df = audit_pit_universe_df(
        universe_df=filtered_universe_df,
        execution_index=pricing_data_df.index,
        tradeable_symbol_list=loaded_symbol_list,
    )

    missing_decision_date_index = pd.DatetimeIndex(month_end_weight_df.index).difference(pricing_data_df.index)
    if len(missing_decision_date_index) > 0:
        missing_date_preview_list = [
            pd.Timestamp(date_ts).strftime("%Y-%m-%d")
            for date_ts in missing_decision_date_index[:5]
        ]
        raise RuntimeError(
            "SP10 market-cap decision dates must be trading dates present in pricing_data_df. "
            f"First missing dates: {missing_date_preview_list}"
        )

    rebalance_weight_df, rebalance_schedule_df = map_month_end_sp10_weights_to_rebalance_open_df(
        month_end_weight_df=month_end_weight_df,
        execution_index=pricing_data_df.index,
    )

    keep_symbol_set = set(audited_universe_df.columns.astype(str).tolist() + list(config.benchmark_list))
    pricing_data_df = pricing_data_df.loc[
        :,
        pricing_data_df.columns.get_level_values(0).isin(keep_symbol_set),
    ].sort_index()
    return pricing_data_df, audited_universe_df, month_end_weight_df, rebalance_weight_df, rebalance_schedule_df


class TrueSp10RotationStrategy(Strategy):
    """
    Monthly equal-weight rotation into the top 10 PIT S&P 500 market-cap names.

    For selected stock i at rebalance open t:

        q^{intent}_{i,t}
            = floor(V_{t-1} * (1 / 10) / Close_{i,t-1})
    """

    def __init__(
        self,
        name: str,
        benchmarks: Sequence[str],
        rebalance_weight_df: pd.DataFrame,
        rebalance_schedule_df: pd.DataFrame,
        capital_base: float = 100_000.0,
        slippage: float = 0.00025,
        commission_per_share: float = 0.005,
        commission_minimum: float = 1.0,
        max_positions_int: int = DEFAULT_CONFIG.max_positions_int,
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
        if max_positions_int <= 0:
            raise ValueError("max_positions_int must be positive.")

        self.rebalance_weight_df = rebalance_weight_df.copy().sort_index()
        self.rebalance_schedule_df = rebalance_schedule_df.copy().sort_index()
        self.max_positions_int = int(max_positions_int)
        self.trade_id_int = 0
        self.current_trade_map: defaultdict[str, int] = defaultdict(default_trade_id_int)

    def compute_signals(self, pricing_data: pd.DataFrame) -> pd.DataFrame:
        return pricing_data

    def _get_target_share_int_map(
        self,
        target_weight_ser: pd.Series,
        close_row_ser: pd.Series,
    ) -> dict[str, int]:
        target_share_int_map: dict[str, int] = {}
        positive_target_weight_ser = target_weight_ser[target_weight_ser > 0.0]
        budget_value_float = float(self.previous_total_value)

        for symbol_str, target_weight_float in positive_target_weight_ser.items():
            close_price_float = float(close_row_ser[(symbol_str, "Close")])
            if not np.isfinite(close_price_float) or close_price_float <= 0.0:
                raise RuntimeError(f"Invalid prior close for target asset {symbol_str} on {self.previous_bar}.")

            # *** CRITICAL*** Monthly target shares are anchored to the
            # previous_bar close. The rebalance cannot adapt to the current
            # open without changing the tested next-open timing semantics.
            target_share_int = int(budget_value_float * float(target_weight_float) / close_price_float)
            if target_share_int > 0:
                target_share_int_map[str(symbol_str)] = target_share_int

        return target_share_int_map

    def iterate(self, data: pd.DataFrame, close: pd.Series, open_prices: pd.Series):
        if close is None or data is None:
            return
        if self.current_bar not in self.rebalance_weight_df.index:
            return

        decision_date_ts = pd.Timestamp(self.rebalance_schedule_df.loc[self.current_bar, "decision_date_ts"])
        # *** CRITICAL*** The selected SP10 basket is decided at the prior
        # month-end close and must execute at the next tradable open. If
        # previous_bar differs, the backtest is using the wrong timing boundary.
        if pd.Timestamp(self.previous_bar) != decision_date_ts:
            raise RuntimeError(
                f"Schedule misalignment on {self.current_bar}: "
                f"decision_date_ts={decision_date_ts}, previous_bar={self.previous_bar}."
            )

        # Row-local missing weights mean "not selected" on this rebalance date.
        # This is not a temporal fill and does not cross a decision boundary.
        target_weight_ser = self.rebalance_weight_df.loc[self.current_bar].fillna(0.0)
        target_share_int_map = self._get_target_share_int_map(
            target_weight_ser=target_weight_ser,
            close_row_ser=close,
        )
        target_symbol_set = set(target_share_int_map)

        current_position_ser = self.get_positions()
        long_position_ser = current_position_ser[current_position_ser > 0.0]
        for symbol_str in long_position_ser.index.astype(str):
            if symbol_str in target_symbol_set:
                continue
            self.order_target_value(
                symbol_str,
                0.0,
                trade_id=self.current_trade_map[symbol_str],
            )

        for symbol_str, target_share_int in target_share_int_map.items():
            current_share_int = int(current_position_ser.get(symbol_str, 0.0))
            if current_share_int == target_share_int:
                continue

            if current_share_int == 0:
                self.trade_id_int += 1
                self.current_trade_map[symbol_str] = self.trade_id_int

            self.order_target_percent(
                symbol_str,
                float(target_weight_ser.loc[symbol_str]),
                trade_id=self.current_trade_map[symbol_str],
            )


def run_variant(
    show_display_bool: bool = True,
    save_results_bool: bool = True,
    output_dir_str: str = "results",
    backtest_start_date_str: str | None = None,
    capital_base_float: float | None = None,
    end_date_str: str | None = None,
    market_cap_csv_path_str: str | None = None,
) -> TrueSp10RotationStrategy:
    config_obj = DEFAULT_CONFIG
    if (
        backtest_start_date_str is not None
        or capital_base_float is not None
        or end_date_str is not None
        or market_cap_csv_path_str is not None
    ):
        config_obj = replace(
            DEFAULT_CONFIG,
            backtest_start_date_str=(
                DEFAULT_CONFIG.backtest_start_date_str
                if backtest_start_date_str is None
                else backtest_start_date_str
            ),
            capital_base_float=(
                DEFAULT_CONFIG.capital_base_float
                if capital_base_float is None
                else float(capital_base_float)
            ),
            end_date_str=end_date_str,
            market_cap_csv_path_str=market_cap_csv_path_str,
        )

    (
        pricing_data_df,
        _universe_df,
        _month_end_weight_df,
        rebalance_weight_df,
        rebalance_schedule_df,
    ) = get_true_sp10_rotation_data(config=config_obj)

    strategy_obj = TrueSp10RotationStrategy(
        name="strategy_mo_sp10_market_cap_rotation",
        benchmarks=list(config_obj.benchmark_list),
        rebalance_weight_df=rebalance_weight_df,
        rebalance_schedule_df=rebalance_schedule_df,
        capital_base=config_obj.capital_base_float,
        slippage=config_obj.slippage_float,
        commission_per_share=config_obj.commission_per_share_float,
        commission_minimum=config_obj.commission_minimum_float,
        max_positions_int=config_obj.max_positions_int,
    )
    strategy_obj.show_taa_weights_report = True
    # *** CRITICAL*** This forward-fill is reporting-only target-weight display.
    # It must not feed signal selection or order generation; orders only read
    # rebalance_weight_df on explicit rebalance dates.
    strategy_obj.daily_target_weights = rebalance_weight_df.reindex(pricing_data_df.index).ffill().dropna()

    # *** CRITICAL*** Keep full pre-start history in pricing_data_df, but only
    # execute from the configured backtest start date.
    calendar_idx = pricing_data_df.index[
        pricing_data_df.index >= pd.Timestamp(config_obj.backtest_start_date_str)
    ]
    run_daily(
        strategy_obj,
        pricing_data_df,
        calendar=calendar_idx,
        show_progress=show_display_bool,
        show_signal_progress_bool=show_display_bool,
        audit_override_bool=None,
    )

    if show_display_bool:
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 1000)
        display(strategy_obj.summary)
        display(strategy_obj.summary_trades)

    if save_results_bool:
        save_results(strategy_obj, output_dir=output_dir_str)

    return strategy_obj


if __name__ == "__main__":
    run_variant()
