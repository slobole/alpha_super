"""
Weekly DV2 oversold selector with matched index regime filters.

This is a research-only implementation of the simple "Long DV2 oversold"
rule described by the user.

Core formulas
-------------
For stock i on weekly decision date t:

    DV1_{i,t} = Close_{i,t} / ((High_{i,t} + Low_{i,t}) / 2) - 1

    DV_{i,t} = (DV1_{i,t} + DV1_{i,t-1}) / 2

    DV2_{i,t} = 100 * PctRank(DV_{i,t-L+1}, ..., DV_{i,t})

    oversold_score_{i,t} = 100 - DV2_{i,t}

For the matching regime index r:

    regime_pass_t = 1[Close_{r,t} > SMA200_{r,t}]

Selection:

    selected_t = floor(0.20 * count(valid PIT members_t)) stocks
                 with the lowest DV2 values.

    target_weight_{i,t} = 1 / count(selected_t), if regime_pass_t and i in selected_t
                        = 0, otherwise

Execution:

    The decision is formed after the completed week close t.
    Orders execute at the next tradable daily open.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, replace
from typing import Sequence

import numpy as np
import pandas as pd
from IPython.display import display

from alpha.engine.backtest import run_daily
from alpha.engine.report import save_results
from alpha.engine.strategy import Strategy
from alpha.indicators import dv2_indicator
from data.norgate_loader import build_index_constituent_matrix, load_raw_prices
from strategies.momentum.strategy_mo_atr_normalized_ndx import (
    audit_pit_universe_df,
    get_asof_universe_membership_ser,
)


DV2_WINDOW_INT = 126
REGIME_SMA_WINDOW_INT = 200
SELECTION_FRACTION_FLOAT = 0.20


def default_trade_id_int() -> int:
    return -1


@dataclass(frozen=True)
class WeeklyDv2OversoldRegimeConfig:
    variant_key_str: str
    indexname_str: str
    regime_symbol_str: str
    history_start_date_str: str = "1998-01-01"
    backtest_start_date_str: str = "2004-01-01"
    end_date_str: str | None = None
    dv2_window_int: int = DV2_WINDOW_INT
    regime_sma_window_int: int = REGIME_SMA_WINDOW_INT
    selection_fraction_float: float = SELECTION_FRACTION_FLOAT
    capital_base_float: float = 100_000.0
    slippage_float: float = 0.00025
    commission_per_share_float: float = 0.005
    commission_minimum_float: float = 1.0

    def __post_init__(self) -> None:
        if not self.variant_key_str:
            raise ValueError("variant_key_str must not be empty.")
        if not self.indexname_str:
            raise ValueError("indexname_str must not be empty.")
        if not self.regime_symbol_str:
            raise ValueError("regime_symbol_str must not be empty.")
        if pd.Timestamp(self.history_start_date_str) >= pd.Timestamp(self.backtest_start_date_str):
            raise ValueError("history_start_date_str must be earlier than backtest_start_date_str.")
        if self.dv2_window_int <= 0:
            raise ValueError("dv2_window_int must be positive.")
        if self.regime_sma_window_int <= 0:
            raise ValueError("regime_sma_window_int must be positive.")
        if not 0.0 < self.selection_fraction_float <= 1.0:
            raise ValueError("selection_fraction_float must be in the interval (0, 1].")
        if self.capital_base_float <= 0.0:
            raise ValueError("capital_base_float must be positive.")
        if self.slippage_float < 0.0:
            raise ValueError("slippage_float must be non-negative.")
        if self.commission_per_share_float < 0.0:
            raise ValueError("commission_per_share_float must be non-negative.")
        if self.commission_minimum_float < 0.0:
            raise ValueError("commission_minimum_float must be non-negative.")


SP500_CONFIG = WeeklyDv2OversoldRegimeConfig(
    variant_key_str="sp500",
    indexname_str="S&P 500",
    regime_symbol_str="$SPX",
)
NASDAQ100_CONFIG = WeeklyDv2OversoldRegimeConfig(
    variant_key_str="nasdaq100",
    indexname_str="Nasdaq 100",
    regime_symbol_str="$NDX",
)
RUSSELL1000_CONFIG = WeeklyDv2OversoldRegimeConfig(
    variant_key_str="russell1000",
    indexname_str="Russell 1000",
    regime_symbol_str="$RUI",
)
DEFAULT_CONFIG = SP500_CONFIG
CONFIG_BY_VARIANT_KEY_DICT = {
    SP500_CONFIG.variant_key_str: SP500_CONFIG,
    NASDAQ100_CONFIG.variant_key_str: NASDAQ100_CONFIG,
    RUSSELL1000_CONFIG.variant_key_str: RUSSELL1000_CONFIG,
}


__all__ = [
    "CONFIG_BY_VARIANT_KEY_DICT",
    "DEFAULT_CONFIG",
    "DV2_WINDOW_INT",
    "NASDAQ100_CONFIG",
    "REGIME_SMA_WINDOW_INT",
    "RUSSELL1000_CONFIG",
    "SELECTION_FRACTION_FLOAT",
    "SP500_CONFIG",
    "WeeklyDv2OversoldRegimeConfig",
    "WeeklyDv2OversoldRegimeStrategy",
    "get_weekly_dv2_oversold_regime_data",
    "map_weekly_decision_dates_to_rebalance_schedule_df",
    "run_all_variants",
    "run_variant",
]


def map_weekly_decision_dates_to_rebalance_schedule_df(
    decision_date_index: pd.DatetimeIndex,
    execution_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    if len(execution_index) < 2:
        raise ValueError("execution_index must contain at least two trading dates.")
    if len(decision_date_index) == 0:
        raise ValueError("decision_date_index must not be empty.")

    execution_idx = pd.DatetimeIndex(execution_index).sort_values()
    decision_idx = pd.DatetimeIndex(decision_date_index).sort_values()
    weekly_decision_ser = pd.Series(decision_idx, index=decision_idx.to_period("W-FRI")).groupby(level=0).max()

    rebalance_schedule_map: dict[pd.Timestamp, pd.Timestamp] = {}
    for decision_date_ts in pd.DatetimeIndex(weekly_decision_ser.to_numpy()):
        execution_insert_int = int(execution_idx.searchsorted(pd.Timestamp(decision_date_ts), side="right"))
        if execution_insert_int >= len(execution_idx):
            continue

        execution_date_ts = pd.Timestamp(execution_idx[execution_insert_int])
        # *** CRITICAL *** Weekly rebalance intent is formed after the actual
        # last tradable close of the W-FRI week and executed only at the next
        # tradable open. Same-bar execution would be a lookahead bug.
        rebalance_schedule_map[execution_date_ts] = pd.Timestamp(decision_date_ts)

    if len(rebalance_schedule_map) == 0:
        raise RuntimeError("No weekly rebalance dates were generated.")

    rebalance_schedule_df = pd.DataFrame.from_dict(
        rebalance_schedule_map,
        orient="index",
        columns=["decision_date_ts"],
    ).sort_index()
    rebalance_schedule_df.index.name = "execution_date_ts"
    return rebalance_schedule_df


def get_weekly_dv2_oversold_regime_data(
    config: WeeklyDv2OversoldRegimeConfig = DEFAULT_CONFIG,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    _symbol_list, raw_universe_df = build_index_constituent_matrix(indexname=config.indexname_str)

    history_start_ts = pd.Timestamp(config.history_start_date_str)
    backtest_start_ts = pd.Timestamp(config.backtest_start_date_str)
    filtered_universe_df = raw_universe_df.loc[raw_universe_df.index >= history_start_ts].copy()
    active_universe_df = filtered_universe_df.loc[filtered_universe_df.index >= backtest_start_ts].copy()
    if config.end_date_str is not None:
        active_universe_df = active_universe_df.loc[active_universe_df.index <= pd.Timestamp(config.end_date_str)]

    active_symbol_list = active_universe_df.columns[active_universe_df.sum(axis=0) > 0].astype(str).tolist()
    if len(active_symbol_list) == 0:
        raise RuntimeError(f"No active {config.indexname_str} symbols were found for the requested window.")

    pricing_data_df = load_raw_prices(
        symbols=active_symbol_list,
        benchmarks=[config.regime_symbol_str],
        start_date=config.history_start_date_str,
        end_date=config.end_date_str,
    )
    loaded_symbol_list = [
        symbol_str
        for symbol_str in active_symbol_list
        if symbol_str in pricing_data_df.columns.get_level_values(0)
    ]
    universe_df = audit_pit_universe_df(
        universe_df=filtered_universe_df,
        execution_index=pricing_data_df.index,
        tradeable_symbol_list=loaded_symbol_list,
    )

    keep_symbol_set = set(universe_df.columns.astype(str).tolist() + [config.regime_symbol_str])
    pricing_data_df = pricing_data_df.loc[
        :,
        pricing_data_df.columns.get_level_values(0).isin(keep_symbol_set),
    ].sort_index()

    warmup_start_ts = pd.Timestamp(config.backtest_start_date_str)
    decision_date_index = pricing_data_df.index[pricing_data_df.index >= warmup_start_ts]
    rebalance_schedule_df = map_weekly_decision_dates_to_rebalance_schedule_df(
        decision_date_index=pd.DatetimeIndex(decision_date_index),
        execution_index=pricing_data_df.index,
    )
    return pricing_data_df, universe_df, rebalance_schedule_df


class WeeklyDv2OversoldRegimeStrategy(Strategy):
    """Long-only weekly top-20%-DV2-oversold selector with full cash regime-off exit."""

    enable_signal_audit = True
    signal_audit_sample_size = 10

    def __init__(
        self,
        name: str,
        benchmarks: Sequence[str],
        rebalance_schedule_df: pd.DataFrame,
        config: WeeklyDv2OversoldRegimeConfig,
    ):
        super().__init__(
            name=name,
            benchmarks=list(benchmarks),
            capital_base=config.capital_base_float,
            slippage=config.slippage_float,
            commission_per_share=config.commission_per_share_float,
            commission_minimum=config.commission_minimum_float,
        )

        if len(rebalance_schedule_df) == 0:
            raise ValueError("rebalance_schedule_df must not be empty.")
        if "decision_date_ts" not in rebalance_schedule_df.columns:
            raise ValueError("rebalance_schedule_df must contain decision_date_ts.")
        if config.regime_symbol_str not in benchmarks:
            raise ValueError("benchmarks must include config.regime_symbol_str.")

        self.rebalance_schedule_df = rebalance_schedule_df.copy().sort_index()
        self.config = config
        self.trade_id_int = 0
        self.current_trade_map: defaultdict[str, int] = defaultdict(default_trade_id_int)
        self.universe_df: pd.DataFrame | None = None

    @property
    def dv2_field_str(self) -> str:
        return f"dv2_{self.config.dv2_window_int}_float"

    @property
    def oversold_score_field_str(self) -> str:
        return f"dv2_oversold_{self.config.dv2_window_int}_float"

    @property
    def regime_sma_field_str(self) -> str:
        return f"regime_sma_{self.config.regime_sma_window_int}_float"

    def compute_signals(self, pricing_data: pd.DataFrame) -> pd.DataFrame:
        signal_data_df = pricing_data.copy()
        benchmark_symbol_set = {str(symbol_str) for symbol_str in self._benchmarks}
        tradeable_symbol_list = [
            str(symbol_str)
            for symbol_str in signal_data_df.columns.get_level_values(0).unique()
            if str(symbol_str) not in benchmark_symbol_set
        ]
        if len(tradeable_symbol_list) == 0:
            raise RuntimeError("No tradeable stock symbols were found in pricing_data.")

        feature_frame_list: list[pd.DataFrame] = []
        dv2_feature_map: dict[str, pd.Series] = {}
        for symbol_str in self.signal_progress(
            tradeable_symbol_list,
            desc_str="DV2 features",
            total_int=len(tradeable_symbol_list),
        ):
            required_column_tuple = (
                (symbol_str, "Close"),
                (symbol_str, "High"),
                (symbol_str, "Low"),
            )
            if any(column_key not in signal_data_df.columns for column_key in required_column_tuple):
                continue

            close_price_ser = signal_data_df[(symbol_str, "Close")].astype(float)
            high_price_ser = signal_data_df[(symbol_str, "High")].astype(float)
            low_price_ser = signal_data_df[(symbol_str, "Low")].astype(float)
            # *** CRITICAL *** DV2 is computed from trailing daily OHLC values
            # through decision close T. It may not use the next open where the
            # rebalance order will execute.
            dv2_feature_map[symbol_str] = dv2_indicator(
                close_price_ser,
                high_price_ser,
                low_price_ser,
                length_int=self.config.dv2_window_int,
            )

        if len(dv2_feature_map) > 0:
            dv2_feature_df = pd.DataFrame(dv2_feature_map, index=signal_data_df.index)
            dv2_output_df = dv2_feature_df.copy()
            dv2_output_df.columns = pd.MultiIndex.from_tuples(
                [(symbol_str, self.dv2_field_str) for symbol_str in dv2_output_df.columns]
            )
            feature_frame_list.append(dv2_output_df)

            oversold_output_df = 100.0 - dv2_feature_df
            oversold_output_df.columns = pd.MultiIndex.from_tuples(
                [(symbol_str, self.oversold_score_field_str) for symbol_str in oversold_output_df.columns]
            )
            feature_frame_list.append(oversold_output_df)

        regime_close_key = (self.config.regime_symbol_str, "Close")
        if regime_close_key not in signal_data_df.columns:
            raise RuntimeError(f"Missing regime close data for {self.config.regime_symbol_str}.")

        regime_close_ser = signal_data_df[regime_close_key].astype(float)
        # *** CRITICAL *** Regime SMA is trailing daily close-only information
        # known at decision close T. Missing warmup is treated as regime off.
        regime_sma_ser = regime_close_ser.rolling(
            window=self.config.regime_sma_window_int,
            min_periods=self.config.regime_sma_window_int,
        ).mean()
        regime_pass_ser = regime_close_ser > regime_sma_ser
        regime_feature_df = pd.DataFrame(
            {
                (self.config.regime_symbol_str, self.regime_sma_field_str): regime_sma_ser,
                (self.config.regime_symbol_str, "regime_pass_bool"): regime_pass_ser,
            },
            index=signal_data_df.index,
        )
        regime_feature_df.columns = pd.MultiIndex.from_tuples(regime_feature_df.columns)
        feature_frame_list.append(regime_feature_df)

        return pd.concat([signal_data_df] + feature_frame_list, axis=1)

    def is_regime_pass_bool(self, close_row_ser: pd.Series) -> bool:
        candidate_feature_df = close_row_ser.unstack()
        if self.config.regime_symbol_str not in candidate_feature_df.index:
            raise RuntimeError(f"Missing regime feature row for {self.config.regime_symbol_str}.")

        regime_pass_value = candidate_feature_df.loc[self.config.regime_symbol_str].get("regime_pass_bool", np.nan)
        return bool(False if pd.isna(regime_pass_value) else regime_pass_value)

    def get_selected_symbol_list(self, close_row_ser: pd.Series) -> list[str]:
        if self.universe_df is None:
            raise RuntimeError("universe_df must be set before rebalances.")

        candidate_feature_df = close_row_ser.unstack()
        if self.dv2_field_str not in candidate_feature_df.columns:
            return []

        universe_member_ser = get_asof_universe_membership_ser(
            self.universe_df,
            pd.Timestamp(self.previous_bar),
        )
        active_symbol_list = universe_member_ser[universe_member_ser == 1].index.astype(str).tolist()
        candidate_feature_df = candidate_feature_df[candidate_feature_df.index.isin(active_symbol_list)].copy()
        if len(candidate_feature_df) == 0:
            return []

        candidate_feature_df = candidate_feature_df.assign(
            dv2_float=pd.to_numeric(candidate_feature_df[self.dv2_field_str], errors="coerce"),
            symbol_str=candidate_feature_df.index.astype(str),
        )
        finite_score_mask_vec = np.isfinite(candidate_feature_df["dv2_float"].to_numpy(dtype=float))
        candidate_feature_df = candidate_feature_df.loc[finite_score_mask_vec].copy()
        if len(candidate_feature_df) == 0:
            return []

        selected_count_int = int(np.floor(len(candidate_feature_df) * float(self.config.selection_fraction_float)))
        selected_count_int = max(1, selected_count_int)
        selected_feature_df = candidate_feature_df.sort_values(
            by=["dv2_float", "symbol_str"],
            ascending=[True, True],
            kind="mergesort",
        ).iloc[:selected_count_int]
        return selected_feature_df.index.astype(str).tolist()

    def iterate(self, data: pd.DataFrame, close: pd.Series, open_prices: pd.Series):
        if close is None or data is None:
            return
        if self.current_bar not in self.rebalance_schedule_df.index:
            return

        decision_date_ts = pd.Timestamp(self.rebalance_schedule_df.loc[self.current_bar, "decision_date_ts"])
        # *** CRITICAL *** The engine passes previous_bar as the last close
        # available before current_bar. This must equal the scheduled weekly
        # decision close, or next-open execution semantics are broken.
        if pd.Timestamp(self.previous_bar) != decision_date_ts:
            raise RuntimeError(
                f"Schedule misalignment on {self.current_bar}: "
                f"decision_date_ts={decision_date_ts}, previous_bar={self.previous_bar}."
            )

        selected_symbol_list = self.get_selected_symbol_list(close_row_ser=close)
        selected_symbol_set = set(selected_symbol_list)
        regime_pass_bool = self.is_regime_pass_bool(close_row_ser=close)
        current_position_ser = self.get_positions()
        long_position_ser = current_position_ser[current_position_ser > 0]

        if not regime_pass_bool:
            for symbol_str in long_position_ser.index.astype(str):
                self.order_target_value(
                    symbol_str,
                    0.0,
                    trade_id=self.current_trade_map[symbol_str],
                )
            return

        for symbol_str in long_position_ser.index.astype(str):
            if symbol_str in selected_symbol_set:
                continue
            self.order_target_value(
                symbol_str,
                0.0,
                trade_id=self.current_trade_map[symbol_str],
            )

        if len(selected_symbol_list) == 0:
            return

        target_weight_float = 1.0 / float(len(selected_symbol_list))
        for symbol_str in selected_symbol_list:
            if symbol_str not in long_position_ser.index.astype(str):
                self.trade_id_int += 1
                self.current_trade_map[symbol_str] = self.trade_id_int

            self.order_target_percent(
                symbol_str,
                target_weight_float,
                trade_id=self.current_trade_map[symbol_str],
            )


def _with_run_overrides(
    config: WeeklyDv2OversoldRegimeConfig,
    backtest_start_date_str: str | None = None,
    capital_base_float: float | None = None,
    end_date_str: str | None = None,
) -> WeeklyDv2OversoldRegimeConfig:
    if backtest_start_date_str is None and capital_base_float is None and end_date_str is None:
        return config

    return replace(
        config,
        backtest_start_date_str=(
            config.backtest_start_date_str if backtest_start_date_str is None else backtest_start_date_str
        ),
        capital_base_float=(
            config.capital_base_float if capital_base_float is None else float(capital_base_float)
        ),
        end_date_str=end_date_str,
    )


def run_variant(
    variant_key_str: str = DEFAULT_CONFIG.variant_key_str,
    show_display_bool: bool = True,
    save_results_bool: bool = True,
    output_dir_str: str = "results",
    backtest_start_date_str: str | None = None,
    capital_base_float: float | None = None,
    end_date_str: str | None = None,
    audit_override_bool: bool | None = False,
) -> WeeklyDv2OversoldRegimeStrategy:
    if variant_key_str not in CONFIG_BY_VARIANT_KEY_DICT:
        raise ValueError(f"Unknown variant_key_str: {variant_key_str}")

    config_obj = _with_run_overrides(
        config=CONFIG_BY_VARIANT_KEY_DICT[variant_key_str],
        backtest_start_date_str=backtest_start_date_str,
        capital_base_float=capital_base_float,
        end_date_str=end_date_str,
    )
    pricing_data_df, universe_df, rebalance_schedule_df = get_weekly_dv2_oversold_regime_data(config=config_obj)
    strategy_obj = WeeklyDv2OversoldRegimeStrategy(
        name=f"strategy_mr_dv2_weekly_oversold_regime_{config_obj.variant_key_str}",
        benchmarks=[config_obj.regime_symbol_str],
        rebalance_schedule_df=rebalance_schedule_df,
        config=config_obj,
    )
    strategy_obj.universe_df = universe_df

    # *** CRITICAL *** Keep full pre-start history for DV2 and SMA200 warmup,
    # but execute only from the configured backtest start. Weekly decisions
    # still fill at the next daily open.
    calendar_idx = pricing_data_df.index[
        pricing_data_df.index >= pd.Timestamp(config_obj.backtest_start_date_str)
    ]
    run_daily(
        strategy_obj,
        pricing_data_df,
        calendar=calendar_idx,
        show_progress=show_display_bool,
        show_signal_progress_bool=show_display_bool,
        audit_override_bool=audit_override_bool,
    )

    if show_display_bool:
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 1000)
        display(strategy_obj.summary)
        display(strategy_obj.summary_trades)

    if save_results_bool:
        save_results(strategy_obj, output_dir=output_dir_str)

    return strategy_obj


def run_all_variants(
    show_display_bool: bool = True,
    save_results_bool: bool = True,
    output_dir_str: str = "results",
    backtest_start_date_str: str | None = None,
    capital_base_float: float | None = None,
    end_date_str: str | None = None,
    audit_override_bool: bool | None = False,
) -> dict[str, WeeklyDv2OversoldRegimeStrategy]:
    strategy_dict: dict[str, WeeklyDv2OversoldRegimeStrategy] = {}
    for variant_key_str in CONFIG_BY_VARIANT_KEY_DICT:
        strategy_dict[variant_key_str] = run_variant(
            variant_key_str=variant_key_str,
            show_display_bool=show_display_bool,
            save_results_bool=save_results_bool,
            output_dir_str=output_dir_str,
            backtest_start_date_str=backtest_start_date_str,
            capital_base_float=capital_base_float,
            end_date_str=end_date_str,
            audit_override_bool=audit_override_bool,
        )
    return strategy_dict


if __name__ == "__main__":
    run_all_variants()
