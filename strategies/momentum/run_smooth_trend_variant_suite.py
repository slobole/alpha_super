"""
Run practical smooth-trend long-only variant comparisons.

This runner is research-only. It keeps the same default execution cost model as
the base smooth-trend strategy and compares small, predeclared changes:

1. Fixed position counts.
2. Log-price trend scoring.
3. NATR-normalized trend scoring.
4. Liquidity filters.
5. VIX exposure scaling.
6. Different point-in-time constituent universes.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import norgatedata
import numpy as np
import pandas as pd

REPO_ROOT_PATH = Path(__file__).resolve().parents[2]
if str(REPO_ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_PATH))

from alpha.engine.backtest import run_daily
from alpha.engine.report import build_research_output_path
from alpha.engine.strategy import Strategy
from data.norgate_loader import build_index_constituent_matrix, load_raw_prices
from strategies.momentum.strategy_mo_atr_normalized_ndx import (
    audit_pit_universe_df,
    get_asof_universe_membership_ser,
    get_monthly_decision_close_df,
    map_month_end_decision_dates_to_rebalance_schedule_df,
)
from strategies.momentum.strategy_mo_atr_normalized_ndx_vxn_scaled import (
    compute_vxn_scale_signal_df,
    get_asof_vxn_scale_float,
)
from strategies.momentum.strategy_mo_smooth_trend_long_sp500 import DEFAULT_CONFIG


DEFAULT_BACKTEST_START_DATE_STR = "2012-01-01"
DEFAULT_CAPITAL_BASE_FLOAT = 100_000.0
SUITE_ENTITY_ID_STR = "smooth_trend_long"
SUITE_ANALYSIS_TYPE_STR = "variant_comparison"

FIELD_PAPER_SLOPE_STR = "paper_slope_float"
FIELD_PAPER_R2_STR = "paper_r2_float"
FIELD_LOG_BETA_STR = "log_beta_float"
FIELD_LOG_ADJR2_STR = "log_adjr2_float"
FIELD_LOG_SCORE_STR = "log_score_float"
FIELD_LOG_NATR20_SCORE_STR = "log_natr20_score_float"
FIELD_LOG_NATR63_SCORE_STR = "log_natr63_score_float"
FIELD_NATR20_STR = "natr20_float"
FIELD_NATR63_STR = "natr63_float"
FIELD_ADV20_STR = "adv20_float"
FIELD_PRICE_STR = "decision_close_float"
FIELD_VOLUME_DAY_COUNT_20_STR = "volume_day_count_20_int"

SCORE_MODE_PAPER_VARIABLE_STR = "paper_variable"
SCORE_MODE_PAPER_FIXED_STR = "paper_fixed"
SCORE_MODE_LOG_SCORE_STR = "log_score"
SCORE_MODE_LOG_NATR20_STR = "log_natr20"
SCORE_MODE_LOG_NATR63_STR = "log_natr63"

UNIVERSE_PRESET_DICT = {
    "sp500": ("S&P 500", "$SPX"),
    "nasdaq100": ("Nasdaq 100", "$NDX"),
    "mid400": ("S&P MidCap 400", "$MID"),
    "small600": ("S&P SmallCap 600", "$SML"),
    "russell1000": ("Russell 1000", "$RUI"),
    "russell2000": ("Russell 2000", "$RUT"),
}


@dataclass(frozen=True)
class SmoothTrendVariantSpec:
    label_str: str
    universe_key_str: str
    score_mode_str: str
    max_positions_int: int | None = None
    price_floor_float: float | None = None
    adv20_floor_float: float | None = None
    vix_scale_bool: bool = False
    target_vix_pct_float: float = 20.0
    min_exposure_scale_float: float = 0.25
    max_exposure_scale_float: float = 1.0

    def __post_init__(self) -> None:
        if self.universe_key_str not in UNIVERSE_PRESET_DICT:
            raise ValueError(f"Unknown universe_key_str: {self.universe_key_str}")
        if self.score_mode_str not in {
            SCORE_MODE_PAPER_VARIABLE_STR,
            SCORE_MODE_PAPER_FIXED_STR,
            SCORE_MODE_LOG_SCORE_STR,
            SCORE_MODE_LOG_NATR20_STR,
            SCORE_MODE_LOG_NATR63_STR,
        }:
            raise ValueError(f"Unknown score_mode_str: {self.score_mode_str}")
        if self.score_mode_str != SCORE_MODE_PAPER_VARIABLE_STR:
            if self.max_positions_int is None or self.max_positions_int <= 0:
                raise ValueError("max_positions_int must be positive for fixed-position variants.")
        if self.price_floor_float is not None and self.price_floor_float < 0.0:
            raise ValueError("price_floor_float must be non-negative.")
        if self.adv20_floor_float is not None and self.adv20_floor_float < 0.0:
            raise ValueError("adv20_floor_float must be non-negative.")
        if self.target_vix_pct_float <= 0.0:
            raise ValueError("target_vix_pct_float must be positive.")
        if self.min_exposure_scale_float < 0.0:
            raise ValueError("min_exposure_scale_float must be non-negative.")
        if self.max_exposure_scale_float > 1.0:
            raise ValueError("max_exposure_scale_float must be <= 1.0.")


@dataclass(frozen=True)
class SmoothTrendUniverseContext:
    universe_key_str: str
    indexname_str: str
    benchmark_symbol_str: str
    pricing_data_df: pd.DataFrame
    universe_df: pd.DataFrame
    rebalance_schedule_df: pd.DataFrame
    feature_data_df: pd.DataFrame
    vix_scale_signal_df: pd.DataFrame | None


def _top_ranked_df(
    candidate_feature_df: pd.DataFrame,
    score_column_str: str,
    count_int: int,
) -> pd.DataFrame:
    if len(candidate_feature_df) == 0 or count_int <= 0:
        return candidate_feature_df.iloc[:0].copy()
    ranked_feature_df = candidate_feature_df.sort_values(
        by=[score_column_str, "symbol_str"],
        ascending=[False, True],
        kind="mergesort",
    )
    return ranked_feature_df.iloc[: int(count_int)].copy()


def _top_ranked_fraction_df(
    candidate_feature_df: pd.DataFrame,
    score_column_str: str,
    quintile_count_int: int,
) -> pd.DataFrame:
    if len(candidate_feature_df) == 0:
        return candidate_feature_df.copy()
    selected_count_int = max(1, int(np.ceil(len(candidate_feature_df) / float(quintile_count_int))))
    return _top_ranked_df(
        candidate_feature_df=candidate_feature_df,
        score_column_str=score_column_str,
        count_int=selected_count_int,
    )


def _compute_natr_df(
    price_close_df: pd.DataFrame,
    price_high_df: pd.DataFrame,
    price_low_df: pd.DataFrame,
    window_int: int,
) -> pd.DataFrame:
    # *** CRITICAL*** True range uses only prior close via shift(1). It must
    # never use a future close when computing the volatility denominator.
    prior_close_df = price_close_df.shift(1)
    true_range_df = (price_high_df - price_low_df).combine(
        (price_high_df - prior_close_df).abs(),
        np.maximum,
    ).combine(
        (price_low_df - prior_close_df).abs(),
        np.maximum,
    )
    atr_df = true_range_df.rolling(window=window_int, min_periods=window_int).mean()
    return atr_df / price_close_df.replace(0.0, np.nan)


def compute_smooth_trend_research_feature_data_df(
    price_close_df: pd.DataFrame,
    price_high_df: pd.DataFrame,
    price_low_df: pd.DataFrame,
    volume_df: pd.DataFrame,
    turnover_df: pd.DataFrame,
    lookback_trading_day_int: int,
    skip_trading_day_int: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if len(price_close_df.index) == 0:
        raise ValueError("price_close_df must not be empty.")
    if not price_close_df.index.is_monotonic_increasing:
        raise ValueError("price_close_df index must be sorted.")
    if price_close_df.index.has_duplicates:
        raise ValueError("price_close_df index must not contain duplicates.")

    monthly_decision_close_df = get_monthly_decision_close_df(price_close_df=price_close_df)
    feature_field_list = [
        FIELD_PAPER_SLOPE_STR,
        FIELD_PAPER_R2_STR,
        FIELD_LOG_BETA_STR,
        FIELD_LOG_ADJR2_STR,
        FIELD_LOG_SCORE_STR,
        FIELD_LOG_NATR20_SCORE_STR,
        FIELD_LOG_NATR63_SCORE_STR,
        FIELD_NATR20_STR,
        FIELD_NATR63_STR,
        FIELD_ADV20_STR,
        FIELD_PRICE_STR,
        FIELD_VOLUME_DAY_COUNT_20_STR,
    ]
    feature_map = {
        field_str: pd.DataFrame(
            np.nan,
            index=monthly_decision_close_df.index,
            columns=price_close_df.columns,
            dtype=float,
        )
        for field_str in feature_field_list
    }

    natr20_df = _compute_natr_df(
        price_close_df=price_close_df,
        price_high_df=price_high_df,
        price_low_df=price_low_df,
        window_int=20,
    )
    natr63_df = _compute_natr_df(
        price_close_df=price_close_df,
        price_high_df=price_high_df,
        price_low_df=price_low_df,
        window_int=63,
    )
    adv20_df = turnover_df.rolling(window=20, min_periods=20).mean()
    volume_day_count_20_df = (volume_df > 0.0).rolling(window=20, min_periods=20).sum()

    for decision_date_ts in monthly_decision_close_df.index:
        decision_pos_int = int(price_close_df.index.get_loc(decision_date_ts))
        start_pos_int = decision_pos_int - int(lookback_trading_day_int)
        end_pos_int = decision_pos_int - int(skip_trading_day_int)
        if start_pos_int < 0 or end_pos_int <= start_pos_int:
            continue

        # *** CRITICAL*** The trend formation window ends at t-skip. It must
        # not include the newest skipped days or the decision-date close.
        formation_price_df = price_close_df.iloc[start_pos_int : end_pos_int + 1]
        finite_price_mask_ser = formation_price_df.notna().all(axis=0)
        positive_price_mask_ser = (formation_price_df > 0.0).all(axis=0)
        valid_symbol_list = (
            finite_price_mask_ser[finite_price_mask_ser & positive_price_mask_ser]
            .index.astype(str)
            .tolist()
        )
        if len(valid_symbol_list) == 0:
            continue

        # Paper-style cumulative simple-return OLS.
        # *** CRITICAL*** Returns are computed only inside the skipped
        # formation window, so no return after t-skip can enter the signal.
        return_window_df = formation_price_df.loc[:, valid_symbol_list].pct_change().iloc[1:]
        return_window_df = return_window_df.replace([np.inf, -np.inf], np.nan)
        valid_return_symbol_list = (
            return_window_df.notna().all(axis=0)
            .loc[lambda valid_ser: valid_ser]
            .index.astype(str)
            .tolist()
        )
        if len(valid_return_symbol_list) == 0:
            continue

        cumulative_return_df = return_window_df.loc[:, valid_return_symbol_list].cumsum()
        paper_time_vec = np.arange(1, len(cumulative_return_df.index) + 1, dtype=float)
        centered_paper_time_vec = paper_time_vec - float(paper_time_vec.mean())
        paper_time_ss_float = float(np.dot(centered_paper_time_vec, centered_paper_time_vec))
        cumulative_return_mat = cumulative_return_df.to_numpy(dtype=float)
        paper_slope_vec = centered_paper_time_vec @ cumulative_return_mat / paper_time_ss_float
        centered_cumulative_return_mat = cumulative_return_mat - cumulative_return_mat.mean(axis=0)
        paper_total_ss_vec = np.sum(centered_cumulative_return_mat * centered_cumulative_return_mat, axis=0)
        paper_regression_ss_vec = (paper_slope_vec * paper_slope_vec) * paper_time_ss_float
        paper_r2_vec = np.divide(
            paper_regression_ss_vec,
            paper_total_ss_vec,
            out=np.full_like(paper_regression_ss_vec, np.nan, dtype=float),
            where=paper_total_ss_vec > 0.0,
        )
        feature_map[FIELD_PAPER_SLOPE_STR].loc[decision_date_ts, valid_return_symbol_list] = paper_slope_vec
        feature_map[FIELD_PAPER_R2_STR].loc[decision_date_ts, valid_return_symbol_list] = np.clip(
            paper_r2_vec,
            0.0,
            1.0,
        )

        # Log-price OLS.
        log_price_df = np.log(
            formation_price_df.loc[:, valid_symbol_list].divide(
                formation_price_df.loc[:, valid_symbol_list].iloc[0],
                axis=1,
            )
        )
        log_price_df = log_price_df.replace([np.inf, -np.inf], np.nan)
        valid_log_symbol_list = (
            log_price_df.notna().all(axis=0)
            .loc[lambda valid_ser: valid_ser]
            .index.astype(str)
            .tolist()
        )
        if len(valid_log_symbol_list) == 0:
            continue

        log_price_mat = log_price_df.loc[:, valid_log_symbol_list].to_numpy(dtype=float)
        log_time_vec = np.arange(0, len(log_price_df.index), dtype=float)
        centered_log_time_vec = log_time_vec - float(log_time_vec.mean())
        log_time_ss_float = float(np.dot(centered_log_time_vec, centered_log_time_vec))
        log_beta_vec = centered_log_time_vec @ log_price_mat / log_time_ss_float
        centered_log_price_mat = log_price_mat - log_price_mat.mean(axis=0)
        log_total_ss_vec = np.sum(centered_log_price_mat * centered_log_price_mat, axis=0)
        log_regression_ss_vec = (log_beta_vec * log_beta_vec) * log_time_ss_float
        log_r2_vec = np.divide(
            log_regression_ss_vec,
            log_total_ss_vec,
            out=np.full_like(log_regression_ss_vec, np.nan, dtype=float),
            where=log_total_ss_vec > 0.0,
        )
        log_r2_vec = np.clip(log_r2_vec, 0.0, 1.0)
        log_observation_count_int = int(len(log_price_df.index))
        log_adjr2_vec = 1.0 - (1.0 - log_r2_vec) * (
            float(log_observation_count_int - 1) / float(log_observation_count_int - 2)
        )
        log_adjr2_vec = np.clip(log_adjr2_vec, 0.0, 1.0)
        log_score_vec = np.maximum(log_beta_vec, 0.0) * log_adjr2_vec

        natr20_ser = natr20_df.iloc[end_pos_int].reindex(valid_log_symbol_list).astype(float)
        natr63_ser = natr63_df.iloc[end_pos_int].reindex(valid_log_symbol_list).astype(float)
        natr20_score_vec = np.divide(
            log_score_vec,
            natr20_ser.to_numpy(dtype=float),
            out=np.full_like(log_score_vec, np.nan, dtype=float),
            where=natr20_ser.to_numpy(dtype=float) > 0.0,
        )
        natr63_score_vec = np.divide(
            log_score_vec,
            natr63_ser.to_numpy(dtype=float),
            out=np.full_like(log_score_vec, np.nan, dtype=float),
            where=natr63_ser.to_numpy(dtype=float) > 0.0,
        )

        feature_map[FIELD_LOG_BETA_STR].loc[decision_date_ts, valid_log_symbol_list] = log_beta_vec
        feature_map[FIELD_LOG_ADJR2_STR].loc[decision_date_ts, valid_log_symbol_list] = log_adjr2_vec
        feature_map[FIELD_LOG_SCORE_STR].loc[decision_date_ts, valid_log_symbol_list] = log_score_vec
        feature_map[FIELD_LOG_NATR20_SCORE_STR].loc[decision_date_ts, valid_log_symbol_list] = natr20_score_vec
        feature_map[FIELD_LOG_NATR63_SCORE_STR].loc[decision_date_ts, valid_log_symbol_list] = natr63_score_vec
        feature_map[FIELD_NATR20_STR].loc[decision_date_ts, valid_log_symbol_list] = natr20_ser.to_numpy(dtype=float)
        feature_map[FIELD_NATR63_STR].loc[decision_date_ts, valid_log_symbol_list] = natr63_ser.to_numpy(dtype=float)

        feature_map[FIELD_ADV20_STR].loc[decision_date_ts] = adv20_df.loc[decision_date_ts]
        feature_map[FIELD_PRICE_STR].loc[decision_date_ts] = price_close_df.loc[decision_date_ts]
        feature_map[FIELD_VOLUME_DAY_COUNT_20_STR].loc[decision_date_ts] = volume_day_count_20_df.loc[
            decision_date_ts
        ]

    valid_decision_mask_ser = feature_map[FIELD_PAPER_SLOPE_STR].notna().any(axis=1)
    valid_decision_index = feature_map[FIELD_PAPER_SLOPE_STR].index[valid_decision_mask_ser]
    feature_frame_list: list[pd.DataFrame] = []
    for field_str, field_df in feature_map.items():
        aligned_field_df = field_df.reindex(valid_decision_index)
        aligned_field_df.columns = pd.MultiIndex.from_tuples(
            [(str(symbol_str), field_str) for symbol_str in aligned_field_df.columns]
        )
        feature_frame_list.append(aligned_field_df)

    feature_data_df = pd.concat(feature_frame_list, axis=1).sort_index(axis=1)
    return monthly_decision_close_df.reindex(valid_decision_index), feature_data_df


def load_vix_scale_signal_df(
    start_date_str: str,
    end_date_str: str | None,
) -> pd.DataFrame:
    vix_price_df = norgatedata.price_timeseries(
        "$VIX",
        stock_price_adjustment_setting=norgatedata.StockPriceAdjustmentType.CAPITALSPECIAL,
        padding_setting=norgatedata.PaddingType.ALLMARKETDAYS,
        start_date=start_date_str,
        end_date=end_date_str,
        timeseriesformat="pandas-dataframe",
    )
    if len(vix_price_df) == 0:
        raise RuntimeError("$VIX returned no data.")
    return compute_vxn_scale_signal_df(
        vxn_close_ser=vix_price_df["Close"].astype(float).sort_index(),
        target_vxn_pct_float=20.0,
        min_exposure_scale_float=0.25,
        max_exposure_scale_float=1.0,
    )


def load_universe_context(
    universe_key_str: str,
    backtest_start_date_str: str,
    end_date_str: str | None,
) -> SmoothTrendUniverseContext:
    indexname_str, benchmark_symbol_str = UNIVERSE_PRESET_DICT[universe_key_str]
    _, raw_universe_df = build_index_constituent_matrix(indexname=indexname_str)

    history_start_ts = pd.Timestamp(DEFAULT_CONFIG.history_start_date_str)
    backtest_start_ts = pd.Timestamp(backtest_start_date_str)
    filtered_universe_df = raw_universe_df.loc[raw_universe_df.index >= history_start_ts].copy()
    active_universe_df = filtered_universe_df.loc[filtered_universe_df.index >= backtest_start_ts].copy()
    if end_date_str is not None:
        active_universe_df = active_universe_df.loc[active_universe_df.index <= pd.Timestamp(end_date_str)]

    active_symbol_list = active_universe_df.columns[active_universe_df.sum(axis=0) > 0].astype(str).tolist()
    if len(active_symbol_list) == 0:
        raise RuntimeError(f"No active symbols found for {indexname_str}.")

    pricing_data_df = load_raw_prices(
        symbols=active_symbol_list,
        benchmarks=[benchmark_symbol_str],
        start_date=DEFAULT_CONFIG.history_start_date_str,
        end_date=end_date_str,
    )
    loaded_symbol_list = [
        symbol_str
        for symbol_str in active_symbol_list
        if symbol_str in pricing_data_df.columns.get_level_values(0)
    ]
    audited_universe_df = audit_pit_universe_df(
        universe_df=filtered_universe_df,
        execution_index=pricing_data_df.index,
        tradeable_symbol_list=loaded_symbol_list,
    )

    keep_symbol_set = set(audited_universe_df.columns.astype(str).tolist() + [benchmark_symbol_str])
    pricing_data_df = pricing_data_df.loc[
        :,
        pricing_data_df.columns.get_level_values(0).isin(keep_symbol_set),
    ].sort_index()

    tradeable_symbol_list = audited_universe_df.columns.astype(str).tolist()
    price_close_df = pd.DataFrame(
        {symbol_str: pricing_data_df[(symbol_str, "Close")] for symbol_str in tradeable_symbol_list},
        index=pricing_data_df.index,
    ).astype(float)
    price_high_df = pd.DataFrame(
        {symbol_str: pricing_data_df[(symbol_str, "High")] for symbol_str in tradeable_symbol_list},
        index=pricing_data_df.index,
    ).astype(float)
    price_low_df = pd.DataFrame(
        {symbol_str: pricing_data_df[(symbol_str, "Low")] for symbol_str in tradeable_symbol_list},
        index=pricing_data_df.index,
    ).astype(float)
    volume_df = pd.DataFrame(
        {symbol_str: pricing_data_df[(symbol_str, "Volume")] for symbol_str in tradeable_symbol_list},
        index=pricing_data_df.index,
    ).astype(float)
    turnover_df = pd.DataFrame(
        {symbol_str: pricing_data_df[(symbol_str, "Turnover")] for symbol_str in tradeable_symbol_list},
        index=pricing_data_df.index,
    ).astype(float)

    monthly_decision_close_df, feature_data_df = compute_smooth_trend_research_feature_data_df(
        price_close_df=price_close_df,
        price_high_df=price_high_df,
        price_low_df=price_low_df,
        volume_df=volume_df,
        turnover_df=turnover_df,
        lookback_trading_day_int=DEFAULT_CONFIG.lookback_trading_day_int,
        skip_trading_day_int=DEFAULT_CONFIG.skip_trading_day_int,
    )
    rebalance_schedule_df = map_month_end_decision_dates_to_rebalance_schedule_df(
        decision_date_index=pd.DatetimeIndex(monthly_decision_close_df.index),
        execution_index=pricing_data_df.index,
    )
    vix_scale_signal_df = load_vix_scale_signal_df(
        start_date_str=DEFAULT_CONFIG.history_start_date_str,
        end_date_str=end_date_str,
    )
    return SmoothTrendUniverseContext(
        universe_key_str=universe_key_str,
        indexname_str=indexname_str,
        benchmark_symbol_str=benchmark_symbol_str,
        pricing_data_df=pricing_data_df,
        universe_df=audited_universe_df,
        rebalance_schedule_df=rebalance_schedule_df,
        feature_data_df=feature_data_df,
        vix_scale_signal_df=vix_scale_signal_df,
    )


class SmoothTrendResearchStrategy(Strategy):
    enable_signal_audit = False

    def __init__(
        self,
        spec: SmoothTrendVariantSpec,
        benchmarks: Sequence[str],
        rebalance_schedule_df: pd.DataFrame,
        feature_data_df: pd.DataFrame,
        universe_df: pd.DataFrame,
        vix_scale_signal_df: pd.DataFrame | None,
        capital_base: float,
    ):
        super().__init__(
            name=spec.label_str,
            benchmarks=list(benchmarks),
            capital_base=capital_base,
            slippage=DEFAULT_CONFIG.slippage_float,
            commission_per_share=DEFAULT_CONFIG.commission_per_share_float,
            commission_minimum=DEFAULT_CONFIG.commission_minimum_float,
        )
        self.spec = spec
        self.rebalance_schedule_df = rebalance_schedule_df.copy().sort_index()
        self.feature_data_df = feature_data_df.copy().sort_index()
        self.universe_df = universe_df.copy().sort_index()
        self.vix_scale_signal_df = None if vix_scale_signal_df is None else vix_scale_signal_df.copy().sort_index()
        self.trade_id_int = 0
        self.current_trade_map: dict[str, int] = {}

    def compute_signals(self, pricing_data: pd.DataFrame) -> pd.DataFrame:
        aligned_feature_data_df = self.feature_data_df.reindex(pricing_data.index)
        return pd.concat([pricing_data, aligned_feature_data_df], axis=1)

    def _candidate_feature_df(self, close_row_ser: pd.Series) -> pd.DataFrame:
        candidate_feature_df = close_row_ser.unstack()
        universe_member_ser = get_asof_universe_membership_ser(
            self.universe_df,
            pd.Timestamp(self.previous_bar),
        )
        active_symbol_list = universe_member_ser[universe_member_ser == 1].index.astype(str).tolist()
        candidate_feature_df = candidate_feature_df[candidate_feature_df.index.isin(active_symbol_list)].copy()
        if len(candidate_feature_df) == 0:
            return candidate_feature_df

        candidate_feature_df = candidate_feature_df.assign(symbol_str=candidate_feature_df.index.astype(str))
        required_field_list = [
            FIELD_PAPER_SLOPE_STR,
            FIELD_PAPER_R2_STR,
            FIELD_LOG_SCORE_STR,
            FIELD_LOG_NATR20_SCORE_STR,
            FIELD_LOG_NATR63_SCORE_STR,
            FIELD_PRICE_STR,
            FIELD_ADV20_STR,
            FIELD_VOLUME_DAY_COUNT_20_STR,
        ]
        for field_str in required_field_list:
            if field_str not in candidate_feature_df.columns:
                candidate_feature_df[field_str] = np.nan
            candidate_feature_df[field_str] = pd.to_numeric(candidate_feature_df[field_str], errors="coerce")

        if self.spec.price_floor_float is not None:
            candidate_feature_df = candidate_feature_df[
                candidate_feature_df[FIELD_PRICE_STR] >= float(self.spec.price_floor_float)
            ].copy()
        if self.spec.adv20_floor_float is not None:
            candidate_feature_df = candidate_feature_df[
                candidate_feature_df[FIELD_ADV20_STR] >= float(self.spec.adv20_floor_float)
            ].copy()
        if self.spec.price_floor_float is not None or self.spec.adv20_floor_float is not None:
            candidate_feature_df = candidate_feature_df[
                candidate_feature_df[FIELD_VOLUME_DAY_COUNT_20_STR] >= 15.0
            ].copy()
        return candidate_feature_df

    def get_target_weight_ser(self, close_row_ser: pd.Series) -> pd.Series:
        candidate_feature_df = self._candidate_feature_df(close_row_ser=close_row_ser)
        if len(candidate_feature_df) == 0:
            return pd.Series(dtype=float)

        if self.spec.score_mode_str == SCORE_MODE_PAPER_VARIABLE_STR:
            finite_mask_vec = np.isfinite(
                candidate_feature_df[[FIELD_PAPER_SLOPE_STR, FIELD_PAPER_R2_STR]].to_numpy(dtype=float)
            ).all(axis=1)
            candidate_feature_df = candidate_feature_df.loc[finite_mask_vec].copy()
            r2_bucket_df = _top_ranked_fraction_df(
                candidate_feature_df=candidate_feature_df,
                score_column_str=FIELD_PAPER_R2_STR,
                quintile_count_int=DEFAULT_CONFIG.quintile_count_int,
            )
            slope_bucket_df = _top_ranked_fraction_df(
                candidate_feature_df=r2_bucket_df,
                score_column_str=FIELD_PAPER_SLOPE_STR,
                quintile_count_int=DEFAULT_CONFIG.quintile_count_int,
            )
            selected_feature_df = slope_bucket_df.loc[slope_bucket_df[FIELD_PAPER_SLOPE_STR] > 0.0].copy()
            if len(selected_feature_df) == 0:
                return pd.Series(dtype=float)
            target_weight_ser = pd.Series(
                1.0 / float(len(selected_feature_df)),
                index=selected_feature_df.index.astype(str),
                dtype=float,
            )
        elif self.spec.score_mode_str == SCORE_MODE_PAPER_FIXED_STR:
            finite_mask_vec = np.isfinite(
                candidate_feature_df[[FIELD_PAPER_SLOPE_STR, FIELD_PAPER_R2_STR]].to_numpy(dtype=float)
            ).all(axis=1)
            candidate_feature_df = candidate_feature_df.loc[finite_mask_vec].copy()
            r2_bucket_df = _top_ranked_fraction_df(
                candidate_feature_df=candidate_feature_df,
                score_column_str=FIELD_PAPER_R2_STR,
                quintile_count_int=DEFAULT_CONFIG.quintile_count_int,
            )
            positive_slope_df = r2_bucket_df.loc[r2_bucket_df[FIELD_PAPER_SLOPE_STR] > 0.0].copy()
            selected_feature_df = _top_ranked_df(
                candidate_feature_df=positive_slope_df,
                score_column_str=FIELD_PAPER_SLOPE_STR,
                count_int=int(self.spec.max_positions_int),
            )
            if len(selected_feature_df) == 0:
                return pd.Series(dtype=float)
            target_weight_ser = pd.Series(
                1.0 / float(self.spec.max_positions_int),
                index=selected_feature_df.index.astype(str),
                dtype=float,
            )
        else:
            score_column_str = {
                SCORE_MODE_LOG_SCORE_STR: FIELD_LOG_SCORE_STR,
                SCORE_MODE_LOG_NATR20_STR: FIELD_LOG_NATR20_SCORE_STR,
                SCORE_MODE_LOG_NATR63_STR: FIELD_LOG_NATR63_SCORE_STR,
            }[self.spec.score_mode_str]
            score_ser = pd.to_numeric(candidate_feature_df[score_column_str], errors="coerce")
            candidate_feature_df = candidate_feature_df.loc[np.isfinite(score_ser.to_numpy(dtype=float))].copy()
            candidate_feature_df = candidate_feature_df.loc[candidate_feature_df[score_column_str] > 0.0].copy()
            selected_feature_df = _top_ranked_df(
                candidate_feature_df=candidate_feature_df,
                score_column_str=score_column_str,
                count_int=int(self.spec.max_positions_int),
            )
            if len(selected_feature_df) == 0:
                return pd.Series(dtype=float)
            target_weight_ser = pd.Series(
                1.0 / float(self.spec.max_positions_int),
                index=selected_feature_df.index.astype(str),
                dtype=float,
            )

        if self.spec.vix_scale_bool:
            if self.vix_scale_signal_df is None:
                raise RuntimeError("vix_scale_signal_df must be set for VIX-scaled variants.")
            exposure_scale_float = get_asof_vxn_scale_float(
                vxn_scale_signal_df=self.vix_scale_signal_df,
                decision_date_ts=pd.Timestamp(self.previous_bar),
            )
            target_weight_ser = target_weight_ser * exposure_scale_float
        return target_weight_ser

    def iterate(self, data: pd.DataFrame, close: pd.Series, open_prices: pd.Series):
        if close is None or data is None:
            return
        if self.current_bar not in self.rebalance_schedule_df.index:
            return

        decision_date_ts = pd.Timestamp(self.rebalance_schedule_df.loc[self.current_bar, "decision_date_ts"])
        # *** CRITICAL*** Rebalance orders must be based on the exact prior
        # decision close, then executed at the next tradable open.
        if pd.Timestamp(self.previous_bar) != decision_date_ts:
            raise RuntimeError(
                f"Schedule misalignment on {self.current_bar}: "
                f"decision_date_ts={decision_date_ts}, previous_bar={self.previous_bar}."
            )

        target_weight_ser = self.get_target_weight_ser(close_row_ser=close)
        target_symbol_set = set(target_weight_ser.index.astype(str).tolist())

        current_position_ser = self.get_positions()
        long_position_ser = current_position_ser[current_position_ser > 0]
        for symbol_str in long_position_ser.index.astype(str):
            if symbol_str in target_symbol_set:
                continue
            self.order_target_value(
                symbol_str,
                0.0,
                trade_id=self.current_trade_map.get(symbol_str),
            )

        for symbol_str, target_weight_float in target_weight_ser.items():
            current_share_float = float(current_position_ser.get(symbol_str, 0.0))
            if current_share_float == 0.0:
                self.trade_id_int += 1
                self.current_trade_map[symbol_str] = self.trade_id_int
            self.order_target_percent(
                symbol_str,
                float(target_weight_float),
                trade_id=self.current_trade_map[symbol_str],
            )


def _summary_value_obj(strategy_obj: Strategy, metric_name_str: str):
    summary_df = getattr(strategy_obj, "summary", None)
    if summary_df is None or metric_name_str not in summary_df.index:
        return None
    return summary_df.loc[metric_name_str, "Strategy"]


def _summary_value_float(strategy_obj: Strategy, metric_name_str: str) -> float | None:
    value_obj = _summary_value_obj(strategy_obj=strategy_obj, metric_name_str=metric_name_str)
    if value_obj is None or pd.isna(value_obj):
        return None
    return float(value_obj)


def _summary_value_date_str(strategy_obj: Strategy, metric_name_str: str) -> str | None:
    value_obj = _summary_value_obj(strategy_obj=strategy_obj, metric_name_str=metric_name_str)
    if value_obj is None or pd.isna(value_obj):
        return None
    return pd.Timestamp(value_obj).date().isoformat()


def _position_diagnostics_dict(strategy_obj: Strategy) -> dict[str, float | int | None]:
    realized_weight_df = getattr(strategy_obj, "realized_weight_df", pd.DataFrame()).copy()
    if len(realized_weight_df) == 0:
        return {
            "avg_positions": None,
            "median_positions": None,
            "max_positions": None,
            "avg_gross_exposure_pct": None,
            "avg_cash_weight_pct": None,
        }
    asset_weight_df = realized_weight_df.drop(columns=["Cash"], errors="ignore")
    position_count_ser = asset_weight_df.notna().sum(axis=1)
    gross_exposure_ser = asset_weight_df.fillna(0.0).abs().sum(axis=1)
    cash_ser = realized_weight_df.get("Cash", pd.Series(index=realized_weight_df.index, dtype=float)).astype(float)
    return {
        "avg_positions": float(position_count_ser.mean()),
        "median_positions": float(position_count_ser.median()),
        "max_positions": int(position_count_ser.max()),
        "avg_gross_exposure_pct": float(gross_exposure_ser.mean() * 100.0),
        "avg_cash_weight_pct": float(cash_ser.mean() * 100.0),
    }


def _comparison_row_dict(
    spec: SmoothTrendVariantSpec,
    context: SmoothTrendUniverseContext,
    strategy_obj: Strategy,
) -> dict[str, object]:
    transaction_df = strategy_obj.get_transactions()
    missing_liquidation_count_int = 0
    if transaction_df is not None and len(transaction_df) > 0 and "order_id" in transaction_df.columns:
        missing_liquidation_count_int = int((transaction_df["order_id"] == -1).sum())
    position_diag_dict = _position_diagnostics_dict(strategy_obj)
    return {
        "variant": spec.label_str,
        "universe": context.indexname_str,
        "score_mode": spec.score_mode_str,
        "max_positions": spec.max_positions_int,
        "price_floor": spec.price_floor_float,
        "adv20_floor_m": None if spec.adv20_floor_float is None else spec.adv20_floor_float / 1_000_000.0,
        "vix_scaled": spec.vix_scale_bool,
        "start": _summary_value_date_str(strategy_obj, "Start"),
        "end": _summary_value_date_str(strategy_obj, "End"),
        "final_equity": _summary_value_float(strategy_obj, "Final [$]"),
        "total_return_pct": _summary_value_float(strategy_obj, "Return [%]"),
        "ann_return_pct": _summary_value_float(strategy_obj, "Return (Ann.) [%]"),
        "ann_vol_pct": _summary_value_float(strategy_obj, "Volatility (Ann.) [%]"),
        "sharpe": _summary_value_float(strategy_obj, "Sharpe Ratio"),
        "max_drawdown_pct": _summary_value_float(strategy_obj, "Max. Drawdown [%]"),
        "mar": _summary_value_float(strategy_obj, "MAR Ratio"),
        "exposure_pct": _summary_value_float(strategy_obj, "Exposure Time [%]"),
        "turnover_ann_pct": _summary_value_float(strategy_obj, "Turnover (Ann.) [%]"),
        "cost_drag_ann_pct": _summary_value_float(strategy_obj, "Cost Drag (Ann.) [%]"),
        "transactions": int(len(transaction_df)) if transaction_df is not None else None,
        "missing_liquidations": missing_liquidation_count_int,
        **position_diag_dict,
    }


def _run_spec(
    spec: SmoothTrendVariantSpec,
    context: SmoothTrendUniverseContext,
    backtest_start_date_str: str,
    capital_base_float: float,
) -> Strategy:
    strategy_obj = SmoothTrendResearchStrategy(
        spec=spec,
        benchmarks=[context.benchmark_symbol_str],
        rebalance_schedule_df=context.rebalance_schedule_df,
        feature_data_df=context.feature_data_df,
        universe_df=context.universe_df,
        vix_scale_signal_df=context.vix_scale_signal_df,
        capital_base=capital_base_float,
    )
    # *** CRITICAL*** Keep full pre-start history for the skipped OLS window,
    # but execute/report only from the configured comparison start date.
    calendar_idx = context.pricing_data_df.index[
        context.pricing_data_df.index >= pd.Timestamp(backtest_start_date_str)
    ]
    with open(os.devnull, "w", encoding="utf-8") as null_file, contextlib.redirect_stdout(null_file):
        run_daily(
            strategy_obj,
            context.pricing_data_df,
            calendar=calendar_idx,
            show_progress=False,
            show_signal_progress_bool=False,
            audit_override_bool=False,
        )
    return strategy_obj


def _default_spec_list() -> list[SmoothTrendVariantSpec]:
    return [
        SmoothTrendVariantSpec(
            label_str="sp500_paper_variable",
            universe_key_str="sp500",
            score_mode_str=SCORE_MODE_PAPER_VARIABLE_STR,
        ),
        SmoothTrendVariantSpec(
            label_str="sp500_paper_n20",
            universe_key_str="sp500",
            score_mode_str=SCORE_MODE_PAPER_FIXED_STR,
            max_positions_int=20,
        ),
        SmoothTrendVariantSpec(
            label_str="sp500_paper_n30",
            universe_key_str="sp500",
            score_mode_str=SCORE_MODE_PAPER_FIXED_STR,
            max_positions_int=30,
        ),
        SmoothTrendVariantSpec(
            label_str="sp500_log_n20",
            universe_key_str="sp500",
            score_mode_str=SCORE_MODE_LOG_SCORE_STR,
            max_positions_int=20,
        ),
        SmoothTrendVariantSpec(
            label_str="sp500_log_n30",
            universe_key_str="sp500",
            score_mode_str=SCORE_MODE_LOG_SCORE_STR,
            max_positions_int=30,
        ),
        SmoothTrendVariantSpec(
            label_str="sp500_log_natr20_n20",
            universe_key_str="sp500",
            score_mode_str=SCORE_MODE_LOG_NATR20_STR,
            max_positions_int=20,
        ),
        SmoothTrendVariantSpec(
            label_str="sp500_log_natr20_n30",
            universe_key_str="sp500",
            score_mode_str=SCORE_MODE_LOG_NATR20_STR,
            max_positions_int=30,
        ),
        SmoothTrendVariantSpec(
            label_str="sp500_log_natr63_n20",
            universe_key_str="sp500",
            score_mode_str=SCORE_MODE_LOG_NATR63_STR,
            max_positions_int=20,
        ),
        SmoothTrendVariantSpec(
            label_str="sp500_log_natr20_n20_adv20m",
            universe_key_str="sp500",
            score_mode_str=SCORE_MODE_LOG_NATR20_STR,
            max_positions_int=20,
            price_floor_float=10.0,
            adv20_floor_float=20_000_000.0,
        ),
        SmoothTrendVariantSpec(
            label_str="sp500_log_natr20_n20_adv50m",
            universe_key_str="sp500",
            score_mode_str=SCORE_MODE_LOG_NATR20_STR,
            max_positions_int=20,
            price_floor_float=10.0,
            adv20_floor_float=50_000_000.0,
        ),
        SmoothTrendVariantSpec(
            label_str="sp500_log_natr20_n20_vix",
            universe_key_str="sp500",
            score_mode_str=SCORE_MODE_LOG_NATR20_STR,
            max_positions_int=20,
            vix_scale_bool=True,
        ),
        SmoothTrendVariantSpec(
            label_str="nasdaq100_log_natr20_n20",
            universe_key_str="nasdaq100",
            score_mode_str=SCORE_MODE_LOG_NATR20_STR,
            max_positions_int=20,
        ),
        SmoothTrendVariantSpec(
            label_str="mid400_log_natr20_n20",
            universe_key_str="mid400",
            score_mode_str=SCORE_MODE_LOG_NATR20_STR,
            max_positions_int=20,
        ),
        SmoothTrendVariantSpec(
            label_str="small600_log_natr20_n20",
            universe_key_str="small600",
            score_mode_str=SCORE_MODE_LOG_NATR20_STR,
            max_positions_int=20,
        ),
    ]


def _format_value_str(value_obj) -> str:
    if value_obj is None or pd.isna(value_obj):
        return ""
    if isinstance(value_obj, bool):
        return str(value_obj)
    if isinstance(value_obj, (int, np.integer)):
        return str(int(value_obj))
    if isinstance(value_obj, (float, np.floating)):
        return f"{float(value_obj):.2f}"
    return str(value_obj)


def _markdown_table_str(markdown_df: pd.DataFrame) -> str:
    column_list = [str(column_obj) for column_obj in markdown_df.columns]
    line_list = [
        "| " + " | ".join(column_list) + " |",
        "| " + " | ".join(["---"] * len(column_list)) + " |",
    ]
    for _row_index, row_ser in markdown_df.iterrows():
        line_list.append(
            "| "
            + " | ".join(
                str(row_ser[column_str]).replace("|", "\\|")
                for column_str in markdown_df.columns
            )
            + " |"
        )
    return "\n".join(line_list) + "\n"


def _write_outputs(
    comparison_df: pd.DataFrame,
    output_dir_str: str,
    timestamp_str: str | None,
    metadata_dict: dict[str, object],
) -> Path:
    output_path = build_research_output_path(
        output_dir=output_dir_str,
        entity_type_str="strategy",
        entity_id_str=SUITE_ENTITY_ID_STR,
        analysis_type_str=SUITE_ANALYSIS_TYPE_STR,
        timestamp_str=timestamp_str,
    )
    output_path.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(output_path / "comparison_table.csv", index=False)

    display_column_list = [
        "variant",
        "universe",
        "ann_return_pct",
        "sharpe",
        "max_drawdown_pct",
        "mar",
        "turnover_ann_pct",
        "cost_drag_ann_pct",
        "avg_positions",
        "missing_liquidations",
    ]
    display_df = comparison_df.reindex(columns=display_column_list).copy()
    markdown_df = display_df.copy()
    for column_str in markdown_df.columns:
        markdown_df[column_str] = markdown_df[column_str].map(_format_value_str)
    (output_path / "comparison_table.md").write_text(
        _markdown_table_str(markdown_df),
        encoding="utf-8",
    )
    (output_path / "metadata.json").write_text(
        json.dumps(metadata_dict, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    return output_path


def run_suite(
    backtest_start_date_str: str,
    capital_base_float: float,
    end_date_str: str | None,
    output_dir_str: str,
    timestamp_str: str | None,
    include_russell_bool: bool,
    universe_filter_list: Sequence[str] | None = None,
    label_filter_list: Sequence[str] | None = None,
) -> pd.DataFrame:
    spec_list = _default_spec_list()
    if include_russell_bool:
        spec_list.extend(
            [
                SmoothTrendVariantSpec(
                    label_str="russell1000_log_natr20_n20",
                    universe_key_str="russell1000",
                    score_mode_str=SCORE_MODE_LOG_NATR20_STR,
                    max_positions_int=20,
                ),
                SmoothTrendVariantSpec(
                    label_str="russell2000_log_natr20_n20",
                    universe_key_str="russell2000",
                    score_mode_str=SCORE_MODE_LOG_NATR20_STR,
                    max_positions_int=20,
                    price_floor_float=5.0,
                    adv20_floor_float=5_000_000.0,
                ),
            ]
        )
    if universe_filter_list is not None:
        universe_filter_set = {str(universe_key_str) for universe_key_str in universe_filter_list}
        unknown_universe_set = universe_filter_set.difference(UNIVERSE_PRESET_DICT)
        if unknown_universe_set:
            raise ValueError(f"Unknown universe filter(s): {sorted(unknown_universe_set)}")
        spec_list = [spec for spec in spec_list if spec.universe_key_str in universe_filter_set]
        if len(spec_list) == 0:
            raise RuntimeError("No variant specs remain after applying universe filters.")
    if label_filter_list is not None:
        label_filter_set = {str(label_str) for label_str in label_filter_list}
        spec_list = [spec for spec in spec_list if spec.label_str in label_filter_set]
        if len(spec_list) == 0:
            raise RuntimeError("No variant specs remain after applying label filters.")

    universe_key_list = list(dict.fromkeys(spec.universe_key_str for spec in spec_list))
    context_map: dict[str, SmoothTrendUniverseContext] = {}
    for universe_key_str in universe_key_list:
        print(f"loading context: {universe_key_str}")
        context_map[universe_key_str] = load_universe_context(
            universe_key_str=universe_key_str,
            backtest_start_date_str=backtest_start_date_str,
            end_date_str=end_date_str,
        )

    row_dict_list: list[dict[str, object]] = []
    for spec in spec_list:
        print(f"running variant: {spec.label_str}")
        context = context_map[spec.universe_key_str]
        strategy_obj = _run_spec(
            spec=spec,
            context=context,
            backtest_start_date_str=backtest_start_date_str,
            capital_base_float=capital_base_float,
        )
        row_dict_list.append(
            _comparison_row_dict(
                spec=spec,
                context=context,
                strategy_obj=strategy_obj,
            )
        )

    comparison_df = pd.DataFrame(row_dict_list)
    comparison_df = comparison_df.sort_values(
        by=["sharpe", "mar", "ann_return_pct"],
        ascending=[False, False, False],
        kind="mergesort",
    ).reset_index(drop=True)
    metadata_dict = {
        "backtest_start_date_str": backtest_start_date_str,
        "capital_base_float": capital_base_float,
        "end_date_str": end_date_str,
        "slippage_float": DEFAULT_CONFIG.slippage_float,
        "commission_per_share_float": DEFAULT_CONFIG.commission_per_share_float,
        "commission_minimum_float": DEFAULT_CONFIG.commission_minimum_float,
        "lookback_trading_day_int": DEFAULT_CONFIG.lookback_trading_day_int,
        "skip_trading_day_int": DEFAULT_CONFIG.skip_trading_day_int,
        "variant_count_int": int(len(comparison_df)),
    }
    output_path = _write_outputs(
        comparison_df=comparison_df,
        output_dir_str=output_dir_str,
        timestamp_str=timestamp_str,
        metadata_dict=metadata_dict,
    )
    print(f"wrote results: {output_path}")
    display_column_list = [
        "variant",
        "universe",
        "ann_return_pct",
        "sharpe",
        "max_drawdown_pct",
        "mar",
        "turnover_ann_pct",
        "cost_drag_ann_pct",
        "avg_positions",
        "missing_liquidations",
    ]
    print(comparison_df.reindex(columns=display_column_list).to_string(index=False))
    return comparison_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backtest-start", default=DEFAULT_BACKTEST_START_DATE_STR)
    parser.add_argument("--capital-base", type=float, default=DEFAULT_CAPITAL_BASE_FLOAT)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--timestamp", default=None)
    parser.add_argument(
        "--universes",
        default=None,
        help="Comma-separated universe keys to run, e.g. sp500 or sp500,mid400.",
    )
    parser.add_argument(
        "--labels",
        default=None,
        help="Comma-separated variant labels to run.",
    )
    parser.add_argument(
        "--include-russell",
        action="store_true",
        help="Also run Russell 1000/2000 variants. These are slower.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    universe_filter_list = None
    if args.universes:
        universe_filter_list = [
            universe_key_str.strip()
            for universe_key_str in str(args.universes).split(",")
            if universe_key_str.strip()
        ]
    label_filter_list = None
    if args.labels:
        label_filter_list = [
            label_str.strip()
            for label_str in str(args.labels).split(",")
            if label_str.strip()
        ]
    run_suite(
        backtest_start_date_str=args.backtest_start,
        capital_base_float=float(args.capital_base),
        end_date_str=args.end_date,
        output_dir_str=args.output_dir,
        timestamp_str=args.timestamp,
        include_russell_bool=bool(args.include_russell),
        universe_filter_list=universe_filter_list,
        label_filter_list=label_filter_list,
    )
