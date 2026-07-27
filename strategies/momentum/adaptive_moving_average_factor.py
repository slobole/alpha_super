"""
Research-only Adaptive Moving-Average Factor (AMAF).

For stock i at the final market close of month T, the eleven features are:

    x_{i,T,L} = SMA_{i,T,L} / Close_{i,T}

    L in {3, 5, 10, 20, 50, 100, 200, 400, 600, 800, 1000}

Each month fits a separate cross-sectional regression inside one PIT universe:

    r_{i,T} = beta_{0,T} + sum_L beta_{T,L} * x_{i,T-1,L} + epsilon_{i,T}

The coefficients used for the T forecast are the trailing 12-month mean of
beta estimates known through Close_T. The highest forecast quintile is held
equal weight from the first Open_(T+1) through the first Open_(T+2).

This module contains no LIVE, release, scheduler, or broker wiring.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from IPython.display import display
from tqdm.auto import tqdm

from alpha.engine.backtest import run_daily
from alpha.engine.report import save_results
from alpha.engine.strategy import Strategy
from data.norgate_loader import (
    CAPITALSPECIAL_ADJUSTMENT_STR,
    TOTALRETURN_ADJUSTMENT_STR,
    build_index_constituent_matrix,
    load_raw_prices,
    norgatedata,
)
from data.norgate_snapshot_store import is_snapshot_mode_enabled_bool
from strategies.momentum.strategy_mo_atr_normalized_ndx import (
    audit_pit_universe_df,
    get_monthly_decision_close_df,
    map_month_end_decision_dates_to_rebalance_schedule_df,
)


PAPER_SMA_LOOKBACK_TUPLE = (3, 5, 10, 20, 50, 100, 200, 400, 600, 800, 1000)
TARGET_WEIGHT_FIELD_STR = "amaf_target_weight_ser"


def default_trade_id_int() -> int:
    return -1


@dataclass(frozen=True)
class AdaptiveMovingAverageFactorConfig:
    strategy_name_str: str
    variant_key_str: str
    indexname_str: str
    source_panel_indexname_str: str
    benchmark_list: tuple[str, ...]
    min_eligible_count_int: int
    history_start_date_str: str = "1990-01-01"
    backtest_start_date_str: str = "2000-01-01"
    end_date_str: str | None = None
    sma_lookback_tuple: tuple[int, ...] = PAPER_SMA_LOOKBACK_TUPLE
    smoothing_month_int: int = 12
    quintile_count_int: int = 5
    minimum_raw_close_float: float = 5.0
    capital_base_float: float = 100_000.0
    slippage_float: float = 0.00025
    commission_per_share_float: float = 0.005
    commission_minimum_float: float = 1.0

    def __post_init__(self) -> None:
        if not self.strategy_name_str:
            raise ValueError("strategy_name_str must not be empty.")
        if not self.variant_key_str:
            raise ValueError("variant_key_str must not be empty.")
        if not self.indexname_str:
            raise ValueError("indexname_str must not be empty.")
        if not self.source_panel_indexname_str:
            raise ValueError("source_panel_indexname_str must not be empty.")
        if len(self.benchmark_list) == 0:
            raise ValueError("benchmark_list must not be empty.")
        if pd.Timestamp(self.history_start_date_str) >= pd.Timestamp(
            self.backtest_start_date_str
        ):
            raise ValueError(
                "history_start_date_str must be earlier than backtest_start_date_str."
            )
        if self.min_eligible_count_int <= 0:
            raise ValueError("min_eligible_count_int must be positive.")
        if len(self.sma_lookback_tuple) == 0:
            raise ValueError("sma_lookback_tuple must not be empty.")
        if any(lookback_int <= 0 for lookback_int in self.sma_lookback_tuple):
            raise ValueError("Every SMA lookback must be positive.")
        if tuple(sorted(set(self.sma_lookback_tuple))) != self.sma_lookback_tuple:
            raise ValueError("sma_lookback_tuple must be sorted and unique.")
        if self.smoothing_month_int <= 0:
            raise ValueError("smoothing_month_int must be positive.")
        if self.quintile_count_int <= 1:
            raise ValueError("quintile_count_int must be greater than one.")
        if self.minimum_raw_close_float < 0.0:
            raise ValueError("minimum_raw_close_float must be non-negative.")
        if self.capital_base_float <= 0.0:
            raise ValueError("capital_base_float must be positive.")
        if self.slippage_float < 0.0:
            raise ValueError("slippage_float must be non-negative.")
        if self.commission_per_share_float < 0.0:
            raise ValueError("commission_per_share_float must be non-negative.")
        if self.commission_minimum_float < 0.0:
            raise ValueError("commission_minimum_float must be non-negative.")


@dataclass(frozen=True)
class AdaptiveMovingAverageFactorSignalBundle:
    target_weight_df: pd.DataFrame
    forecast_df: pd.DataFrame
    coefficient_df: pd.DataFrame
    coverage_df: pd.DataFrame


def assign_stable_quintile_ser(
    forecast_ser: pd.Series,
    quintile_count_int: int,
) -> pd.Series:
    clean_forecast_ser = forecast_ser.dropna().astype(float)
    quintile_ser = pd.Series(
        pd.NA,
        index=forecast_ser.index,
        dtype="Int64",
    )
    if len(clean_forecast_ser) < quintile_count_int:
        return quintile_ser

    stable_order_df = pd.DataFrame(
        {
            "symbol_str": clean_forecast_ser.index.astype(str),
            "forecast_float": clean_forecast_ser.to_numpy(dtype=float),
            "original_index_value": clean_forecast_ser.index.to_numpy(),
        }
    ).sort_values(
        ["forecast_float", "symbol_str"],
        kind="mergesort",
    )
    stable_order_index = pd.Index(
        stable_order_df["original_index_value"].to_numpy()
    )
    eligible_count_int = len(stable_order_index)
    quintile_vec = (
        np.arange(eligible_count_int, dtype=int)
        * quintile_count_int
        // eligible_count_int
        + 1
    )
    quintile_ser.loc[stable_order_index] = quintile_vec
    return quintile_ser


def build_monthly_sma_ratio_by_lookback_dict(
    price_close_df: pd.DataFrame,
    decision_date_index: pd.DatetimeIndex,
    sma_lookback_tuple: tuple[int, ...],
) -> dict[int, pd.DataFrame]:
    feature_series_by_lookback_dict: dict[int, dict[str, pd.Series]] = {
        lookback_int: {} for lookback_int in sma_lookback_tuple
    }
    for symbol_str in price_close_df.columns.astype(str):
        observed_close_ser = (
            pd.to_numeric(price_close_df[symbol_str], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
            .astype(float)
        )
        for lookback_int in sma_lookback_tuple:
            # *** CRITICAL *** lookahead-sensitive rolling boundary:
            # SMA_{i,T,L} contains the latest L observed closes through
            # Close_T, including Close_T, and never uses a later observation.
            sma_ser = observed_close_ser.rolling(
                window=lookback_int,
                min_periods=lookback_int,
            ).mean()
            sma_ratio_ser = sma_ser.divide(observed_close_ser)
            feature_series_by_lookback_dict[lookback_int][symbol_str] = (
                sma_ratio_ser.reindex(decision_date_index)
            )

    return {
        lookback_int: pd.DataFrame(
            feature_series_by_symbol_dict,
            index=decision_date_index,
            dtype=float,
        )
        for lookback_int, feature_series_by_symbol_dict in (
            feature_series_by_lookback_dict.items()
        )
    }


def _fit_cross_section_beta_arr(
    prior_feature_df: pd.DataFrame,
    current_return_ser: pd.Series,
    min_eligible_count_int: int,
) -> tuple[np.ndarray | None, int]:
    regression_df = prior_feature_df.join(
        current_return_ser.rename("close_monthly_return_float"),
        how="inner",
    )
    regression_df = regression_df.replace([np.inf, -np.inf], np.nan).dropna()
    observation_count_int = len(regression_df)
    feature_count_int = prior_feature_df.shape[1]
    if observation_count_int < max(
        min_eligible_count_int,
        feature_count_int + 2,
    ):
        return None, observation_count_int

    feature_mat = regression_df[prior_feature_df.columns].to_numpy(dtype=float)
    design_mat = np.column_stack(
        [np.ones(observation_count_int, dtype=float), feature_mat]
    )
    return_vec = regression_df["close_monthly_return_float"].to_numpy(
        dtype=float
    )
    beta_vec, _, _, _ = np.linalg.lstsq(
        design_mat,
        return_vec,
        rcond=None,
    )
    return beta_vec, observation_count_int


def build_adaptive_moving_average_factor_signal_bundle(
    price_close_df: pd.DataFrame,
    raw_close_df: pd.DataFrame,
    universe_df: pd.DataFrame,
    decision_date_index: pd.DatetimeIndex,
    config_obj: AdaptiveMovingAverageFactorConfig,
) -> AdaptiveMovingAverageFactorSignalBundle:
    symbol_list = [
        symbol_str
        for symbol_str in price_close_df.columns.astype(str)
        if symbol_str in raw_close_df.columns and symbol_str in universe_df.columns
    ]
    if len(symbol_list) == 0:
        raise RuntimeError(
            "No symbols overlap between adjusted prices, raw prices, and PIT membership."
        )

    decision_date_index = pd.DatetimeIndex(decision_date_index).sort_values()
    monthly_close_df = (
        price_close_df.loc[:, symbol_list]
        .reindex(decision_date_index)
        .apply(pd.to_numeric, errors="coerce")
    )
    monthly_raw_close_df = (
        raw_close_df.loc[:, symbol_list]
        .reindex(decision_date_index)
        .apply(pd.to_numeric, errors="coerce")
    )
    # *** CRITICAL *** PIT alignment uses only the latest membership row on or
    # before Close_T. A future universe row must never be backfilled.
    monthly_membership_df = (
        universe_df.loc[:, symbol_list]
        .sort_index()
        .reindex(decision_date_index)
        .ffill()
    )
    if monthly_membership_df.isna().any(axis=None):
        missing_date_list = [
            pd.Timestamp(date_ts).date().isoformat()
            for date_ts in monthly_membership_df.index[
                monthly_membership_df.isna().any(axis=1)
            ][:5]
        ]
        raise RuntimeError(
            "PIT membership is unavailable on early decision dates: "
            f"{missing_date_list}"
        )
    monthly_membership_df = monthly_membership_df.astype(bool)

    feature_by_lookback_dict = build_monthly_sma_ratio_by_lookback_dict(
        price_close_df=price_close_df.loc[:, symbol_list],
        decision_date_index=decision_date_index,
        sma_lookback_tuple=config_obj.sma_lookback_tuple,
    )
    feature_column_list = [
        f"sma_{lookback_int}_ratio_float"
        for lookback_int in config_obj.sma_lookback_tuple
    ]

    # *** CRITICAL *** label timing: return_T uses adjusted Close_(T-1) and
    # adjusted Close_T on the two common market month-ends. It is used only to
    # estimate beta_T after Close_T, never to create a forecast before Close_T.
    close_monthly_return_df = monthly_close_df.divide(
        monthly_close_df.shift(1)
    ).sub(1.0)

    eligible_by_date_dict: dict[pd.Timestamp, pd.Series] = {}
    feature_panel_by_date_dict: dict[pd.Timestamp, pd.DataFrame] = {}
    for decision_date_ts in decision_date_index:
        feature_df = pd.DataFrame(
            {
                f"sma_{lookback_int}_ratio_float": (
                    feature_by_lookback_dict[lookback_int].loc[decision_date_ts]
                )
                for lookback_int in config_obj.sma_lookback_tuple
            },
            index=symbol_list,
            dtype=float,
        )
        complete_feature_ser = (
            feature_df.replace([np.inf, -np.inf], np.nan).notna().all(axis=1)
        )
        adjusted_close_ser = monthly_close_df.loc[decision_date_ts]
        raw_close_ser = monthly_raw_close_df.loc[decision_date_ts]
        eligible_ser = (
            monthly_membership_df.loc[decision_date_ts]
            & adjusted_close_ser.replace([np.inf, -np.inf], np.nan).notna()
            & raw_close_ser.replace([np.inf, -np.inf], np.nan).notna()
            & raw_close_ser.ge(config_obj.minimum_raw_close_float)
            & complete_feature_ser
        )
        eligible_by_date_dict[pd.Timestamp(decision_date_ts)] = eligible_ser
        feature_panel_by_date_dict[pd.Timestamp(decision_date_ts)] = feature_df

    beta_column_list = ["intercept_float", *feature_column_list]
    beta_df = pd.DataFrame(
        np.nan,
        index=decision_date_index,
        columns=beta_column_list,
        dtype=float,
    )
    regression_count_by_date_dict: dict[pd.Timestamp, int] = {}
    for month_index_int in range(1, len(decision_date_index)):
        current_date_ts = pd.Timestamp(decision_date_index[month_index_int])
        prior_date_ts = pd.Timestamp(decision_date_index[month_index_int - 1])
        if (
            current_date_ts.to_period("M")
            != prior_date_ts.to_period("M") + 1
        ):
            regression_count_by_date_dict[current_date_ts] = 0
            continue

        prior_eligible_ser = eligible_by_date_dict[prior_date_ts]
        prior_feature_df = feature_panel_by_date_dict[prior_date_ts].loc[
            prior_eligible_ser
        ]
        current_return_ser = close_monthly_return_df.loc[current_date_ts]
        # *** CRITICAL *** regression boundary: beta_T joins features and PIT
        # eligibility known at Close_(T-1) with returns ending at Close_T.
        beta_vec, regression_count_int = _fit_cross_section_beta_arr(
            prior_feature_df=prior_feature_df,
            current_return_ser=current_return_ser,
            min_eligible_count_int=config_obj.min_eligible_count_int,
        )
        regression_count_by_date_dict[current_date_ts] = regression_count_int
        if beta_vec is not None:
            beta_df.loc[current_date_ts] = beta_vec

    # *** CRITICAL *** coefficient smoothing: the row at Close_T contains only
    # beta estimates through T. Reindexing on every calendar month means a
    # missing estimate cannot be skipped or replaced by an older/future beta.
    smoothed_beta_df = beta_df.rolling(
        window=config_obj.smoothing_month_int,
        min_periods=config_obj.smoothing_month_int,
    ).mean()

    target_weight_row_dict: dict[pd.Timestamp, pd.Series] = {}
    forecast_record_list: list[dict[str, object]] = []
    coverage_record_list: list[dict[str, object]] = []
    valid_signal_started_bool = False
    for decision_date_ts in decision_date_index:
        decision_date_ts = pd.Timestamp(decision_date_ts)
        eligible_ser = eligible_by_date_dict[decision_date_ts]
        eligible_symbol_list = eligible_ser.index[eligible_ser].astype(str).tolist()
        eligible_count_int = len(eligible_symbol_list)
        smoothed_beta_ser = smoothed_beta_df.loc[decision_date_ts]
        regression_count_int = regression_count_by_date_dict.get(
            decision_date_ts,
            0,
        )

        if smoothed_beta_ser.isna().any():
            if valid_signal_started_bool:
                raise RuntimeError(
                    "AMAF lost its required consecutive coefficient history "
                    f"after activation: decision_date={decision_date_ts.date()} "
                    f"regression_count={regression_count_int}"
                )
            coverage_record_list.append(
                {
                    "decision_date_ts": decision_date_ts,
                    "regression_count_int": regression_count_int,
                    "eligible_count_int": eligible_count_int,
                    "selected_count_int": 0,
                    "status_str": "coefficient_warmup",
                }
            )
            continue

        if eligible_count_int < config_obj.min_eligible_count_int:
            raise RuntimeError(
                "AMAF cannot form a valid monthly ranking: "
                f"decision_date={decision_date_ts.date()} "
                f"eligible_count={eligible_count_int} "
                f"required={config_obj.min_eligible_count_int}"
            )

        current_feature_df = feature_panel_by_date_dict[decision_date_ts].loc[
            eligible_symbol_list,
            feature_column_list,
        ]
        # *** CRITICAL *** forecast boundary: x_(i,T) is known only after
        # Close_T and is multiplied only by the beta mean known through T.
        forecast_vec = (
            current_feature_df.to_numpy(dtype=float)
            @ smoothed_beta_ser[feature_column_list].to_numpy(dtype=float)
        )
        forecast_ser = pd.Series(
            forecast_vec,
            index=pd.Index(eligible_symbol_list, name="symbol_str"),
            name="forecast_float",
            dtype=float,
        )
        quintile_ser = assign_stable_quintile_ser(
            forecast_ser=forecast_ser,
            quintile_count_int=config_obj.quintile_count_int,
        )
        selected_symbol_list = sorted(
            quintile_ser.loc[
                quintile_ser.eq(config_obj.quintile_count_int)
            ].index.astype(str)
        )
        if len(selected_symbol_list) == 0:
            raise RuntimeError(
                "AMAF produced no top-quintile symbols on "
                f"{decision_date_ts.date()}."
            )

        target_weight_float = 1.0 / float(len(selected_symbol_list))
        target_weight_row_dict[decision_date_ts] = pd.Series(
            target_weight_float,
            index=selected_symbol_list,
            dtype=float,
        )
        selected_symbol_set = set(selected_symbol_list)
        for symbol_str in eligible_symbol_list:
            forecast_record_list.append(
                {
                    "decision_date_ts": decision_date_ts,
                    "symbol_str": symbol_str,
                    "forecast_float": float(forecast_ser.loc[symbol_str]),
                    "quintile_int": int(quintile_ser.loc[symbol_str]),
                    "selected_bool": symbol_str in selected_symbol_set,
                    "target_weight_float": (
                        target_weight_float
                        if symbol_str in selected_symbol_set
                        else 0.0
                    ),
                }
            )
        coverage_record_list.append(
            {
                "decision_date_ts": decision_date_ts,
                "regression_count_int": regression_count_int,
                "eligible_count_int": eligible_count_int,
                "selected_count_int": len(selected_symbol_list),
                "status_str": "valid_target",
            }
        )
        valid_signal_started_bool = True

    target_weight_df = pd.DataFrame.from_dict(
        target_weight_row_dict,
        orient="index",
        dtype=float,
    ).sort_index()
    target_weight_df.index.name = "decision_date_ts"
    coefficient_df = pd.concat(
        [
            beta_df.add_prefix("monthly_"),
            smoothed_beta_df.add_prefix("smoothed_"),
        ],
        axis=1,
    )
    coefficient_df.index.name = "decision_date_ts"
    return AdaptiveMovingAverageFactorSignalBundle(
        target_weight_df=target_weight_df,
        forecast_df=pd.DataFrame(forecast_record_list),
        coefficient_df=coefficient_df,
        coverage_df=pd.DataFrame(coverage_record_list),
    )


def align_pit_universe_with_unavailable_prefix_df(
    universe_df: pd.DataFrame,
    execution_index: pd.DatetimeIndex,
    tradeable_symbol_list: Sequence[str],
) -> pd.DataFrame:
    if len(universe_df) == 0:
        raise RuntimeError("PIT universe is empty.")
    first_membership_date_ts = pd.Timestamp(universe_df.index.min())
    execution_index = pd.DatetimeIndex(execution_index).sort_values()
    available_execution_index = execution_index[
        execution_index >= first_membership_date_ts
    ]
    if len(available_execution_index) == 0:
        raise RuntimeError(
            "Pricing history ends before PIT membership history begins."
        )
    audited_available_universe_df = audit_pit_universe_df(
        universe_df=universe_df,
        execution_index=available_execution_index,
        tradeable_symbol_list=tradeable_symbol_list,
    )
    unavailable_execution_index = execution_index[
        execution_index < first_membership_date_ts
    ]
    # *** CRITICAL *** PIT availability boundary: price history may predate
    # Norgate's first observed membership row. Those earlier dates are marked
    # ineligible for every stock; membership is never backfilled from the
    # first future row.
    unavailable_universe_df = pd.DataFrame(
        0,
        index=unavailable_execution_index,
        columns=audited_available_universe_df.columns,
        dtype=int,
    )
    return pd.concat(
        [unavailable_universe_df, audited_available_universe_df],
        axis=0,
    ).reindex(execution_index)


def build_pit_intersection_universe_df(
    target_universe_df: pd.DataFrame,
    source_universe_df: pd.DataFrame,
) -> pd.DataFrame:
    common_symbol_list = [
        symbol_str
        for symbol_str in target_universe_df.columns.astype(str)
        if symbol_str in source_universe_df.columns
    ]
    if len(common_symbol_list) == 0:
        raise RuntimeError("Target and source PIT universes have no overlap.")
    target_index = pd.DatetimeIndex(target_universe_df.index).sort_values()
    target_membership_df = align_pit_universe_with_unavailable_prefix_df(
        universe_df=target_universe_df,
        execution_index=target_index,
        tradeable_symbol_list=common_symbol_list,
    )
    source_membership_df = align_pit_universe_with_unavailable_prefix_df(
        universe_df=source_universe_df,
        execution_index=target_index,
        tradeable_symbol_list=common_symbol_list,
    )
    # *** CRITICAL *** The source eligibility is resolved as of each target
    # universe date. An ever-observed source-symbol set would admit companies
    # before they actually joined the domestic source panel.
    intersection_universe_df = (
        target_membership_df.astype(bool)
        & source_membership_df.astype(bool)
    ).astype(int)
    return intersection_universe_df.sort_index()


def build_source_panel_membership_df(
    source_panel_indexname_str: str,
    target_symbol_list: Sequence[str],
) -> pd.DataFrame:
    unique_target_symbol_list = list(
        dict.fromkeys(str(symbol_str) for symbol_str in target_symbol_list)
    )
    if is_snapshot_mode_enabled_bool():
        _, source_universe_df = build_index_constituent_matrix(
            indexname=source_panel_indexname_str
        )
        available_symbol_list = [
            symbol_str
            for symbol_str in unique_target_symbol_list
            if symbol_str in source_universe_df.columns
        ]
        if not available_symbol_list:
            raise RuntimeError("No target symbols overlap the source snapshot.")
        return source_universe_df.loc[:, available_symbol_list].copy()

    market_calendar_index = norgatedata.price_timeseries(
        "$SPX",
        timeseriesformat="pandas-dataframe",
    ).index
    last_trading_day_ts = pd.Timestamp(market_calendar_index[-1])
    membership_df_list: list[pd.DataFrame] = []
    for symbol_str in tqdm(
        unique_target_symbol_list,
        desc="building target source membership",
    ):
        symbol_membership_df = norgatedata.index_constituent_timeseries(
            symbol_str,
            source_panel_indexname_str,
            timeseriesformat="pandas-dataframe",
        )
        if int(symbol_membership_df["Index Constituent"].sum()) <= 0:
            continue
        symbol_membership_df = symbol_membership_df.rename(
            columns={"Index Constituent": symbol_str}
        )
        symbol_membership_df = symbol_membership_df.loc[
            symbol_membership_df[symbol_str].eq(1)
        ]
        if (
            len(symbol_membership_df) > 5
            and last_trading_day_ts
            != pd.Timestamp(symbol_membership_df.index[-1])
        ):
            symbol_membership_df = symbol_membership_df.iloc[:-5]
        if len(symbol_membership_df) > 0:
            membership_df_list.append(symbol_membership_df)
    if not membership_df_list:
        raise RuntimeError("No target symbols were source-panel members.")
    return (
        pd.concat(membership_df_list, axis=1)
        .fillna(0)
        .astype(int)
        .sort_index()
    )


def source_panel_membership_is_structurally_implied_bool(
    target_indexname_str: str,
    source_panel_indexname_str: str,
) -> bool:
    return (
        target_indexname_str == "Russell 1000"
        and source_panel_indexname_str == "Russell 3000"
    )


def get_adaptive_moving_average_factor_data(
    config_obj: AdaptiveMovingAverageFactorConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    _, raw_universe_df = build_index_constituent_matrix(
        indexname=config_obj.indexname_str
    )
    russell_hierarchy_redundant_bool = (
        source_panel_membership_is_structurally_implied_bool(
            target_indexname_str=config_obj.indexname_str,
            source_panel_indexname_str=(
                config_obj.source_panel_indexname_str
            ),
        )
    )
    if russell_hierarchy_redundant_bool:
        # Russell 1000 membership implies same-date Russell 3000 membership by
        # index construction. Requerying the superset cannot remove a valid
        # Russell 1000 member and only adds thousands of redundant API calls.
        pit_intersection_universe_df = raw_universe_df.copy()
    else:
        raw_source_universe_df = build_source_panel_membership_df(
            source_panel_indexname_str=(
                config_obj.source_panel_indexname_str
            ),
            target_symbol_list=raw_universe_df.columns.astype(str).tolist(),
        )
        pit_intersection_universe_df = build_pit_intersection_universe_df(
            target_universe_df=raw_universe_df,
            source_universe_df=raw_source_universe_df,
        )

    history_start_ts = pd.Timestamp(config_obj.history_start_date_str)
    filtered_universe_df = pit_intersection_universe_df.loc[
        pit_intersection_universe_df.index >= history_start_ts,
    ].copy()
    if config_obj.end_date_str is not None:
        filtered_universe_df = filtered_universe_df.loc[
            filtered_universe_df.index <= pd.Timestamp(config_obj.end_date_str)
        ]
    active_symbol_list = (
        filtered_universe_df.columns[
            filtered_universe_df.sum(axis=0) > 0
        ]
        .astype(str)
        .tolist()
    )
    if len(active_symbol_list) == 0:
        raise RuntimeError(
            f"No active {config_obj.indexname_str} symbols were found."
        )

    pricing_data_df = load_raw_prices(
        symbols=active_symbol_list,
        benchmarks=list(config_obj.benchmark_list),
        start_date=config_obj.history_start_date_str,
        end_date=config_obj.end_date_str,
    )
    loaded_symbol_list = [
        symbol_str
        for symbol_str in active_symbol_list
        if symbol_str in pricing_data_df.columns.get_level_values(0)
    ]
    audited_universe_df = align_pit_universe_with_unavailable_prefix_df(
        universe_df=filtered_universe_df,
        execution_index=pricing_data_df.index,
        tradeable_symbol_list=loaded_symbol_list,
    )
    keep_symbol_set = set(
        audited_universe_df.columns.astype(str).tolist()
        + list(config_obj.benchmark_list)
    )
    pricing_data_df = pricing_data_df.loc[
        :,
        pricing_data_df.columns.get_level_values(0).isin(keep_symbol_set),
    ].sort_index()

    price_close_df = pd.DataFrame(
        {
            symbol_str: pricing_data_df[(symbol_str, "Close")]
            for symbol_str in audited_universe_df.columns.astype(str)
        },
        index=pricing_data_df.index,
    ).astype(float)
    monthly_decision_close_df = get_monthly_decision_close_df(
        price_close_df=price_close_df
    )
    rebalance_schedule_df = (
        map_month_end_decision_dates_to_rebalance_schedule_df(
            decision_date_index=pd.DatetimeIndex(
                monthly_decision_close_df.index
            ),
            execution_index=pricing_data_df.index,
        )
    )
    return pricing_data_df, audited_universe_df, rebalance_schedule_df


class AdaptiveMovingAverageFactorStrategy(Strategy):
    """Monthly, PIT, equal-weight top-quintile AMAF strategy."""

    enable_signal_audit = True
    signal_audit_sample_size = 3

    def __init__(
        self,
        name: str,
        benchmarks: Sequence[str],
        universe_df: pd.DataFrame,
        rebalance_schedule_df: pd.DataFrame,
        config_obj: AdaptiveMovingAverageFactorConfig,
    ):
        super().__init__(
            name=name,
            benchmarks=list(benchmarks),
            capital_base=config_obj.capital_base_float,
            slippage=config_obj.slippage_float,
            commission_per_share=config_obj.commission_per_share_float,
            commission_minimum=config_obj.commission_minimum_float,
            performance_benchmark_adjustment_str=TOTALRETURN_ADJUSTMENT_STR,
        )
        self._data_adjustment_policy_dict.update(
            {
                "stock_signal_adjustment_str": CAPITALSPECIAL_ADJUSTMENT_STR,
                "execution_and_marks_adjustment_str": (
                    CAPITALSPECIAL_ADJUSTMENT_STR
                ),
                "performance_benchmark_adjustment_str": (
                    TOTALRETURN_ADJUSTMENT_STR
                ),
            }
        )
        if len(universe_df) == 0:
            raise ValueError("universe_df must not be empty.")
        if len(rebalance_schedule_df) == 0:
            raise ValueError("rebalance_schedule_df must not be empty.")
        if "decision_date_ts" not in rebalance_schedule_df.columns:
            raise ValueError(
                "rebalance_schedule_df must contain decision_date_ts."
            )

        self.config_obj = config_obj
        self.universe_df = universe_df.copy().sort_index()
        self.rebalance_schedule_df = rebalance_schedule_df.copy().sort_index()
        self.trade_id_int = 0
        self.current_trade_map: defaultdict[str, int] = defaultdict(
            default_trade_id_int
        )
        self.signal_bundle_obj: (
            AdaptiveMovingAverageFactorSignalBundle | None
        ) = None
        self.rebalance_selection_row_list: list[dict[str, object]] = []
        self.rebalance_selection_df = pd.DataFrame()

    def compute_signals(self, pricing_data: pd.DataFrame) -> pd.DataFrame:
        benchmark_symbol_set = {
            str(symbol_str) for symbol_str in self._benchmarks
        }
        tradeable_symbol_list = [
            str(symbol_str)
            for symbol_str in pricing_data.columns.get_level_values(0).unique()
            if (
                str(symbol_str) not in benchmark_symbol_set
                and str(symbol_str) in self.universe_df.columns
            )
        ]
        required_field_list = ["Close", "Unadjusted Close"]
        for field_str in required_field_list:
            missing_symbol_list = [
                symbol_str
                for symbol_str in tradeable_symbol_list
                if (symbol_str, field_str) not in pricing_data.columns
            ]
            if missing_symbol_list:
                raise RuntimeError(
                    f"AMAF requires {field_str} for every stock. "
                    f"Missing sample: {missing_symbol_list[:5]}"
                )

        price_close_df = pd.DataFrame(
            {
                symbol_str: pricing_data[(symbol_str, "Close")]
                for symbol_str in tradeable_symbol_list
            },
            index=pricing_data.index,
        ).astype(float)
        raw_close_df = pd.DataFrame(
            {
                symbol_str: pricing_data[(symbol_str, "Unadjusted Close")]
                for symbol_str in tradeable_symbol_list
            },
            index=pricing_data.index,
        ).astype(float)
        monthly_decision_close_df = get_monthly_decision_close_df(
            price_close_df=price_close_df
        )
        signal_bundle_obj = (
            build_adaptive_moving_average_factor_signal_bundle(
                price_close_df=price_close_df,
                raw_close_df=raw_close_df,
                universe_df=self.universe_df,
                decision_date_index=pd.DatetimeIndex(
                    monthly_decision_close_df.index
                ),
                config_obj=self.config_obj,
            )
        )
        if (
            len(signal_bundle_obj.target_weight_df) == 0
            and len(pricing_data.index) > 0
            and pd.Timestamp(pricing_data.index[-1])
            >= pd.Timestamp(self.config_obj.backtest_start_date_str)
        ):
            raise RuntimeError(
                "AMAF produced no valid monthly target after coefficient "
                "warmup by the requested backtest start."
            )
        if self.signal_bundle_obj is None:
            self.signal_bundle_obj = signal_bundle_obj

        target_weight_daily_df = signal_bundle_obj.target_weight_df.reindex(
            pricing_data.index
        )
        target_weight_daily_df = target_weight_daily_df.reindex(
            columns=tradeable_symbol_list
        )
        target_weight_daily_df.columns = pd.MultiIndex.from_tuples(
            [
                (symbol_str, TARGET_WEIGHT_FIELD_STR)
                for symbol_str in tradeable_symbol_list
            ]
        )
        signal_data_df = pd.concat(
            [pricing_data.copy(), target_weight_daily_df],
            axis=1,
        )
        signal_data_df.attrs = dict(pricing_data.attrs)
        return signal_data_df

    def _target_weight_ser(self, close_row_ser: pd.Series) -> pd.Series:
        close_field_df = close_row_ser.unstack()
        if TARGET_WEIGHT_FIELD_STR not in close_field_df.columns:
            return pd.Series(dtype=float)
        target_weight_ser = pd.to_numeric(
            close_field_df[TARGET_WEIGHT_FIELD_STR],
            errors="coerce",
        ).dropna()
        target_weight_ser.index = target_weight_ser.index.astype(str)
        return target_weight_ser.astype(float).sort_index()

    def iterate(
        self,
        data: pd.DataFrame,
        close: pd.Series,
        open_prices: pd.Series,
    ):
        if data is None or close is None:
            return
        if self.current_bar not in self.rebalance_schedule_df.index:
            return

        decision_date_ts = pd.Timestamp(
            self.rebalance_schedule_df.loc[
                self.current_bar,
                "decision_date_ts",
            ]
        )
        # *** CRITICAL *** execution boundary: the scheduled Close_T must be
        # previous_bar exactly. Target orders then fill at current_bar Open_T+1
        # under the unchanged Vanilla engine lifecycle.
        if pd.Timestamp(self.previous_bar) != decision_date_ts:
            raise RuntimeError(
                f"AMAF schedule misalignment on {self.current_bar}: "
                f"decision_date={decision_date_ts}, "
                f"previous_bar={self.previous_bar}."
            )

        target_weight_ser = self._target_weight_ser(close_row_ser=close)
        if len(target_weight_ser) == 0:
            if self.signal_bundle_obj is None:
                raise RuntimeError("AMAF signal bundle was not initialized.")
            first_signal_date_ts = pd.Timestamp(
                self.signal_bundle_obj.target_weight_df.index[0]
            )
            if decision_date_ts < first_signal_date_ts:
                return
            raise RuntimeError(
                f"AMAF has no valid target on {decision_date_ts.date()}."
            )
        if not np.isclose(float(target_weight_ser.sum()), 1.0):
            raise RuntimeError(
                f"AMAF target weights do not sum to one on "
                f"{decision_date_ts.date()}: "
                f"{float(target_weight_ser.sum()):.12f}"
            )

        current_position_ser = self.get_positions()
        target_symbol_set = set(target_weight_ser.index)
        active_position_ser = current_position_ser[
            current_position_ser != 0
        ]
        for symbol_str in active_position_ser.index.astype(str):
            if symbol_str in target_symbol_set:
                continue
            self.order_target_value(
                symbol_str,
                0.0,
                trade_id=self.current_trade_map[symbol_str],
            )

        for symbol_str, target_weight_float in target_weight_ser.items():
            if float(current_position_ser.get(symbol_str, 0.0)) == 0.0:
                self.trade_id_int += 1
                self.current_trade_map[symbol_str] = self.trade_id_int
            self.order_target_percent(
                symbol_str,
                float(target_weight_float),
                trade_id=self.current_trade_map[symbol_str],
            )

        if self.signal_bundle_obj is not None:
            selection_df = self.signal_bundle_obj.forecast_df.loc[
                self.signal_bundle_obj.forecast_df["decision_date_ts"].eq(
                    decision_date_ts
                )
                & self.signal_bundle_obj.forecast_df["selected_bool"].astype(
                    bool
                )
            ].copy()
            if len(selection_df) > 0:
                selection_df.insert(
                    0,
                    "execution_date_ts",
                    pd.Timestamp(self.current_bar),
                )
                self.rebalance_selection_row_list.extend(
                    selection_df.to_dict("records")
                )

    def finalize(self, current_data: pd.DataFrame):
        self.rebalance_selection_df = pd.DataFrame(
            self.rebalance_selection_row_list
        )


def _write_assumptions_md(
    output_path: Path,
    strategy_obj: AdaptiveMovingAverageFactorStrategy,
) -> None:
    config_obj = strategy_obj.config_obj
    assumption_md_str = f"""# Adaptive Moving-Average Factor Assumptions

- Research-only BENCH strategy; no LIVE, release, scheduler, or broker wiring.
- Variant: `{config_obj.variant_key_str}`.
- PIT universe: `{config_obj.indexname_str} Current & Past`.
- Domestic source-panel intersection: `{config_obj.source_panel_indexname_str} Current & Past`.
- Stock signal, execution, and mark basis: Norgate `CAPITALSPECIAL`.
- Raw eligibility filter: `Unadjusted Close_T >= {config_obj.minimum_raw_close_float:.2f}`.
- Decision: actual final market close of month `T`.
- Execution: first tradable open of month `T+1` through Vanilla target-percent orders.
- Feature formula: `x_(i,T,L) = SMA_(i,T,L) / Close_(i,T)`.
- SMA lookbacks: `{list(config_obj.sma_lookback_tuple)}` observed stock sessions.
- Regression formula: `r_(i,T) = beta_(0,T) + sum_L beta_(T,L) * x_(i,T-1,L) + epsilon_(i,T)`.
- Coefficient rule: trailing `{config_obj.smoothing_month_int}` consecutive monthly beta estimates through `T`.
- Minimum usable regression and ranking count: `{config_obj.min_eligible_count_int}`.
- Selection: stable forecast quintiles, forecast ascending then symbol ascending; hold quintile `{config_obj.quintile_count_int}`.
- Sizing: equal weight, 100% long gross and net exposure, no leverage.
- No volatility sizing, regime overlay, stop-loss, or intramonth exit.
- Slippage: `{config_obj.slippage_float:.6f}` per side.
- Commission: `{config_obj.commission_per_share_float:.6f}` per share, minimum `{config_obj.commission_minimum_float:.2f}`.
- Positive cash earns 0% under the engine policy.
- Target-percent sizing uses prior-close portfolio value; overnight gaps, rounding, and missing opens can create realized-open weight drift.
"""
    (output_path / "adaptive_moving_average_factor_assumptions.md").write_text(
        assumption_md_str,
        encoding="utf-8",
    )


def run_amaf_variant(
    config_obj: AdaptiveMovingAverageFactorConfig,
    show_display_bool: bool = True,
    save_results_bool: bool = True,
    output_dir_str: str = "results",
    backtest_start_date_str: str | None = None,
    capital_base_float: float | None = None,
    end_date_str: str | None = None,
    audit_override_bool: bool | None = False,
) -> AdaptiveMovingAverageFactorStrategy:
    if (
        backtest_start_date_str is not None
        or capital_base_float is not None
        or end_date_str is not None
    ):
        config_obj = replace(
            config_obj,
            backtest_start_date_str=(
                config_obj.backtest_start_date_str
                if backtest_start_date_str is None
                else backtest_start_date_str
            ),
            capital_base_float=(
                config_obj.capital_base_float
                if capital_base_float is None
                else float(capital_base_float)
            ),
            end_date_str=end_date_str,
        )

    (
        pricing_data_df,
        universe_df,
        rebalance_schedule_df,
    ) = get_adaptive_moving_average_factor_data(config_obj=config_obj)
    strategy_obj = AdaptiveMovingAverageFactorStrategy(
        name=config_obj.strategy_name_str,
        benchmarks=list(config_obj.benchmark_list),
        universe_df=universe_df,
        rebalance_schedule_df=rebalance_schedule_df,
        config_obj=config_obj,
    )
    # *** CRITICAL *** Full pre-start history is retained for the 1000-session
    # SMA and 12-month beta mean, while orders/reporting begin only at the
    # requested backtest start.
    calendar_index = pricing_data_df.index[
        pricing_data_df.index
        >= pd.Timestamp(config_obj.backtest_start_date_str)
    ]
    run_daily(
        strategy_obj,
        pricing_data_df,
        calendar=calendar_index,
        show_progress=show_display_bool,
        show_signal_progress_bool=show_display_bool,
        audit_override_bool=audit_override_bool,
    )

    if show_display_bool:
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 1000)
        display(strategy_obj.summary)
        display(strategy_obj.summary_trades)
        display(strategy_obj.rebalance_selection_df.tail(40))
        if strategy_obj.signal_bundle_obj is not None:
            display(strategy_obj.signal_bundle_obj.coverage_df.tail(24))

    if save_results_bool:
        output_path = save_results(
            strategy_obj,
            output_dir=output_dir_str,
        )
        strategy_obj.rebalance_selection_df.to_csv(
            output_path / "rebalance_selection.csv",
            index=False,
        )
        if strategy_obj.signal_bundle_obj is not None:
            strategy_obj.signal_bundle_obj.forecast_df.to_csv(
                output_path / "amaf_forecasts.csv",
                index=False,
            )
            strategy_obj.signal_bundle_obj.coefficient_df.to_csv(
                output_path / "amaf_coefficients.csv",
            )
            strategy_obj.signal_bundle_obj.coverage_df.to_csv(
                output_path / "amaf_coverage.csv",
                index=False,
            )
        _write_assumptions_md(
            output_path=output_path,
            strategy_obj=strategy_obj,
        )

    return strategy_obj


__all__ = [
    "AdaptiveMovingAverageFactorConfig",
    "AdaptiveMovingAverageFactorSignalBundle",
    "AdaptiveMovingAverageFactorStrategy",
    "PAPER_SMA_LOOKBACK_TUPLE",
    "TARGET_WEIGHT_FIELD_STR",
    "assign_stable_quintile_ser",
    "align_pit_universe_with_unavailable_prefix_df",
    "build_adaptive_moving_average_factor_signal_bundle",
    "build_monthly_sma_ratio_by_lookback_dict",
    "build_pit_intersection_universe_df",
    "build_source_panel_membership_df",
    "get_adaptive_moving_average_factor_data",
    "run_amaf_variant",
    "source_panel_membership_is_structurally_implied_bool",
]
