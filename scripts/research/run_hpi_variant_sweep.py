"""Controlled research sweep for the stateful HPI strategy family.

The sweep excludes the fixed S&P/Nasdaq portfolio combinations. Every executed
row keeps the Vanilla next-open engine, PIT membership, costs, and dividend
accounting unchanged.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import talib

REPO_ROOT_PATH = Path(__file__).resolve().parents[2]
if str(REPO_ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_PATH))

from alpha.engine.backtest import run_daily
from strategies.hpi.stateful_long import (
    EXIT_IBS_THRESHOLD_FLOAT,
    EXIT_RSI2_THRESHOLD_FLOAT,
    HPIStatefulLongStrategy,
    HPI_THRESHOLD_FLOAT,
    MAX_ENTRY_IBS_FLOAT,
    NATR_FIELD_STR,
    TURNOVER_FIELD_STR,
    compute_strict_hpi,
    get_asof_universe_symbol_set,
    load_exact_hpi_inputs,
)


RETURN_2D_FIELD_STR = "return_2d_ser"
RETURN_5D_FIELD_STR = "return_5d_ser"
HPI_2D_FIELD_STR = "hpi_2d_ser"
HPI_5D_FIELD_STR = "hpi_5d_ser"
NATR_10_FIELD_STR = "natr_10_ser"
NATR_20_FIELD_STR = "natr_20_ser"
NATR_RANK_FIELD_STR = "natr_rank_ser"
NATR_ENSEMBLE_FIELD_STR = "natr_ensemble_rank_ser"
RAW_PRICE_FIELD_STR = "raw_price_ser"
ADV_63_FIELD_STR = "adv_63_ser"

ENTRY_BASELINE_STR = "baseline"
ENTRY_HORIZON_VOTE_STR = "hpi_2_3_5_vote"
LIQUIDITY_NONE_STR = "none"
LIQUIDITY_FIXED_STR = "raw_price_5_adv63_5m"
LIQUIDITY_RELATIVE_STR = "raw_price_5_adv63_above_median"
EXIT_OR_STR = "ibs_or_rsi"
EXIT_IBS_STR = "ibs_only"
EXIT_RSI_STR = "rsi_only"
RANK_TURNOVER_STR = "turnover"
RANK_NATR14_STR = "natr14"
RANK_NATR_ENSEMBLE_STR = "natr_10_14_20_ensemble"
SIZING_EQUAL_STR = "equal_slots"
SIZING_INVERSE_NATR_STR = "capped_inverse_natr14"
SIZING_NATR_STRENGTH_STR = "capped_natr14_rank"

MAX_POSITION_WEIGHT_FLOAT = 0.15
ADV_63_MIN_FLOAT = 5_000_000.0
RAW_PRICE_MIN_FLOAT = 5.0
DEFAULT_CAPITAL_FLOAT = 100_000.0
DEFAULT_START_DATE_STR = "2004-01-01"
DEFAULT_FEATURE_START_DATE_STR = "1998-01-01"


@dataclass(frozen=True)
class UniverseSpec:
    key_str: str
    indexname_str: str
    benchmark_symbol_str: str


@dataclass(frozen=True)
class VariantSpec:
    key_str: str
    universe_key_str: str
    entry_mode_str: str = ENTRY_BASELINE_STR
    ranking_mode_str: str = RANK_TURNOVER_STR
    liquidity_mode_str: str = LIQUIDITY_NONE_STR
    exit_mode_str: str = EXIT_OR_STR
    sizing_mode_str: str = SIZING_EQUAL_STR


UNIVERSE_SPEC_DICT = {
    "sp500": UniverseSpec(
        key_str="sp500",
        indexname_str="S&P 500",
        benchmark_symbol_str="$SPX",
    ),
    "nasdaq100": UniverseSpec(
        key_str="nasdaq100",
        indexname_str="Nasdaq 100",
        benchmark_symbol_str="$NDX",
    ),
}


VARIANT_SPEC_TUPLE = (
    VariantSpec("sp500_baseline", "sp500"),
    VariantSpec(
        "sp500_hpi_2_3_5_vote",
        "sp500",
        entry_mode_str=ENTRY_HORIZON_VOTE_STR,
    ),
    VariantSpec(
        "sp500_hpi_2_3_5_vote_liquidity_relative",
        "sp500",
        entry_mode_str=ENTRY_HORIZON_VOTE_STR,
        liquidity_mode_str=LIQUIDITY_RELATIVE_STR,
    ),
    VariantSpec(
        "sp500_liquidity_fixed",
        "sp500",
        liquidity_mode_str=LIQUIDITY_FIXED_STR,
    ),
    VariantSpec(
        "sp500_liquidity_relative",
        "sp500",
        liquidity_mode_str=LIQUIDITY_RELATIVE_STR,
    ),
    VariantSpec("sp500_exit_ibs_only", "sp500", exit_mode_str=EXIT_IBS_STR),
    VariantSpec("sp500_exit_rsi_only", "sp500", exit_mode_str=EXIT_RSI_STR),
    VariantSpec(
        "sp500_rank_natr14_control",
        "sp500",
        ranking_mode_str=RANK_NATR14_STR,
    ),
    VariantSpec(
        "sp500_size_inverse_natr14",
        "sp500",
        sizing_mode_str=SIZING_INVERSE_NATR_STR,
    ),
    VariantSpec(
        "sp500_size_natr14_strength",
        "sp500",
        sizing_mode_str=SIZING_NATR_STRENGTH_STR,
    ),
    VariantSpec(
        "nasdaq100_baseline",
        "nasdaq100",
        ranking_mode_str=RANK_NATR14_STR,
    ),
    VariantSpec(
        "nasdaq100_hpi_2_3_5_vote",
        "nasdaq100",
        entry_mode_str=ENTRY_HORIZON_VOTE_STR,
        ranking_mode_str=RANK_NATR14_STR,
    ),
    VariantSpec(
        "nasdaq100_rank_natr_10_14_20",
        "nasdaq100",
        ranking_mode_str=RANK_NATR_ENSEMBLE_STR,
    ),
    VariantSpec(
        "nasdaq100_liquidity_fixed",
        "nasdaq100",
        ranking_mode_str=RANK_NATR14_STR,
        liquidity_mode_str=LIQUIDITY_FIXED_STR,
    ),
    VariantSpec(
        "nasdaq100_liquidity_relative",
        "nasdaq100",
        ranking_mode_str=RANK_NATR14_STR,
        liquidity_mode_str=LIQUIDITY_RELATIVE_STR,
    ),
    VariantSpec(
        "nasdaq100_exit_ibs_only",
        "nasdaq100",
        ranking_mode_str=RANK_NATR14_STR,
        exit_mode_str=EXIT_IBS_STR,
    ),
    VariantSpec(
        "nasdaq100_exit_rsi_only",
        "nasdaq100",
        ranking_mode_str=RANK_NATR14_STR,
        exit_mode_str=EXIT_RSI_STR,
    ),
    VariantSpec(
        "nasdaq100_rank_turnover_control",
        "nasdaq100",
        ranking_mode_str=RANK_TURNOVER_STR,
    ),
    VariantSpec(
        "nasdaq100_size_inverse_natr14",
        "nasdaq100",
        ranking_mode_str=RANK_NATR14_STR,
        sizing_mode_str=SIZING_INVERSE_NATR_STR,
    ),
    VariantSpec(
        "nasdaq100_size_natr14_strength",
        "nasdaq100",
        ranking_mode_str=RANK_NATR14_STR,
        sizing_mode_str=SIZING_NATR_STRENGTH_STR,
    ),
)
DECLARED_VARIANT_COUNT_INT = len(VARIANT_SPEC_TUPLE)


STATUS_ROW_DICT_LIST = [
    {
        "variant_str": "hpi_threshold_24_30_36_vote",
        "status_str": "not_run_algebraic_duplicate",
        "reason_str": "vote >= 2 is exactly equivalent to HPI < 30",
    },
    {
        "variant_str": "market_stress_breadth",
        "status_str": "diagnostic_only",
        "reason_str": "saved by fixed breadth buckets; no in-sample filter direction chosen",
    },
    {
        "variant_str": "exit_rule_ensemble",
        "status_str": "reported_as_controlled_components",
        "reason_str": "IBS-only, RSI2-only, and OR exits run separately at equal capital",
    },
    {
        "variant_str": "sector_cap_2_or_3",
        "status_str": "rejected",
        "reason_str": "local Norgate sector classification has no point-in-time date argument",
    },
    {
        "variant_str": "market_regime_exposure_scaling",
        "status_str": "deferred",
        "reason_str": "existing positions require an explicit rebalance contract; evidence is unstable",
    },
    {
        "variant_str": "sp500_nasdaq_fixed_portfolios",
        "status_str": "excluded_by_user",
        "reason_str": "50/50 and 75/25 portfolio variants were explicitly excluded",
    },
]


def validate_variant_feature_contract(
    signal_data_df: pd.DataFrame,
    selected_variant_spec_list: list[VariantSpec],
    start_date_str: str,
) -> None:
    """Fail before backtesting when a selected variant lacks its input feature."""

    required_field_set: set[str] = set()
    for variant_spec_obj in selected_variant_spec_list:
        if variant_spec_obj.ranking_mode_str == RANK_TURNOVER_STR:
            required_field_set.add(TURNOVER_FIELD_STR)
        if variant_spec_obj.entry_mode_str == ENTRY_HORIZON_VOTE_STR:
            required_field_set.update(
                {
                    RETURN_2D_FIELD_STR,
                    RETURN_5D_FIELD_STR,
                    HPI_2D_FIELD_STR,
                    HPI_5D_FIELD_STR,
                }
            )
        if variant_spec_obj.ranking_mode_str == RANK_NATR14_STR:
            required_field_set.add(NATR_FIELD_STR)
        if variant_spec_obj.ranking_mode_str == RANK_NATR_ENSEMBLE_STR:
            required_field_set.update(
                {NATR_10_FIELD_STR, NATR_FIELD_STR, NATR_20_FIELD_STR}
            )
        if variant_spec_obj.liquidity_mode_str != LIQUIDITY_NONE_STR:
            required_field_set.update({RAW_PRICE_FIELD_STR, ADV_63_FIELD_STR})
        if variant_spec_obj.sizing_mode_str != SIZING_EQUAL_STR:
            required_field_set.add(NATR_FIELD_STR)

    execution_signal_df = signal_data_df.loc[
        signal_data_df.index >= pd.Timestamp(start_date_str)
    ]
    missing_field_list: list[str] = []
    for field_str in sorted(required_field_set):
        field_column_list = [
            column_tuple
            for column_tuple in execution_signal_df.columns
            if column_tuple[1] == field_str
            and not str(column_tuple[0]).startswith("$")
        ]
        if not field_column_list:
            missing_field_list.append(field_str)
            continue
        field_value_arr = execution_signal_df[field_column_list].to_numpy(
            dtype=float,
        )
        if not np.isfinite(field_value_arr).any():
            missing_field_list.append(field_str)

    if missing_field_list:
        raise RuntimeError(
            "Selected HPI variants lack usable feature data after "
            f"{start_date_str}: {missing_field_list}"
        )


def _feature_symbol_list(signal_data_df: pd.DataFrame) -> list[str]:
    return [
        str(symbol_obj)
        for symbol_obj in signal_data_df.columns.get_level_values(0).unique()
        if not str(symbol_obj).startswith("$")
    ]


def compute_hpi_sweep_signal_data_df(
    pricing_data_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute the feature superset once for every controlled HPI variant."""

    feature_strategy_obj = HPIStatefulLongStrategy(
        name="hpi_sweep_feature_builder",
        benchmarks=[],
        ranking_field_str=NATR_FIELD_STR,
    )
    signal_data_df = feature_strategy_obj.compute_signals(pricing_data_df)
    extra_feature_ser_dict: dict[tuple[str, str], pd.Series] = {}

    for symbol_str in _feature_symbol_list(signal_data_df):
        required_price_key_list = [
            (symbol_str, "Close"),
            (symbol_str, "High"),
            (symbol_str, "Low"),
        ]
        if any(key_tuple not in pricing_data_df.columns for key_tuple in required_price_key_list):
            continue

        symbol_price_df = pd.DataFrame(
            {
                "Close": pricing_data_df[(symbol_str, "Close")],
                "High": pricing_data_df[(symbol_str, "High")],
                "Low": pricing_data_df[(symbol_str, "Low")],
            }
        ).dropna()
        if symbol_price_df.empty:
            continue

        close_price_ser = symbol_price_df["Close"].astype(float)
        high_price_ser = symbol_price_df["High"].astype(float)
        low_price_ser = symbol_price_df["Low"].astype(float)

        for horizon_int, return_field_str, hpi_field_str in (
            (2, RETURN_2D_FIELD_STR, HPI_2D_FIELD_STR),
            (5, RETURN_5D_FIELD_STR, HPI_5D_FIELD_STR),
        ):
            # *** CRITICAL*** Return_w,T uses only Close_T and the w-th prior
            # valid close. HPI excludes T from its 1,260-observation reference.
            return_ser = close_price_ser / close_price_ser.shift(horizon_int) - 1.0
            extra_feature_ser_dict[(symbol_str, return_field_str)] = return_ser
            extra_feature_ser_dict[(symbol_str, hpi_field_str)] = compute_strict_hpi(
                return_ser
            )

        for natr_window_int, natr_field_str in (
            (10, NATR_10_FIELD_STR),
            (20, NATR_20_FIELD_STR),
        ):
            # *** CRITICAL*** NATR uses OHLC through T only and can rank an
            # order only for Open_(T+1).
            extra_feature_ser_dict[(symbol_str, natr_field_str)] = pd.Series(
                talib.NATR(
                    high_price_ser.to_numpy(dtype=float),
                    low_price_ser.to_numpy(dtype=float),
                    close_price_ser.to_numpy(dtype=float),
                    timeperiod=natr_window_int,
                ),
                index=symbol_price_df.index,
                dtype=float,
            )

        raw_close_key = (symbol_str, "Unadjusted Close")
        volume_key = (symbol_str, "Volume")
        if raw_close_key in pricing_data_df.columns and volume_key in pricing_data_df.columns:
            raw_liquidity_df = pd.DataFrame(
                {
                    "raw_close": pricing_data_df[raw_close_key],
                    "volume": pricing_data_df[volume_key],
                }
            ).dropna()
            raw_price_ser = raw_liquidity_df["raw_close"].astype(float)
            dollar_volume_ser = (
                raw_price_ser * raw_liquidity_df["volume"].astype(float)
            )
            # *** CRITICAL*** ADV63_T includes only raw close and volume
            # observations from [T-62, T], known after Close_T.
            adv_63_ser = dollar_volume_ser.rolling(
                window=63,
                min_periods=63,
            ).mean()
            extra_feature_ser_dict[(symbol_str, RAW_PRICE_FIELD_STR)] = raw_price_ser
            extra_feature_ser_dict[(symbol_str, ADV_63_FIELD_STR)] = adv_63_ser

    extra_feature_df = pd.DataFrame(
        {
            key_tuple: feature_ser.reindex(signal_data_df.index)
            for key_tuple, feature_ser in extra_feature_ser_dict.items()
        },
        index=signal_data_df.index,
    )
    return pd.concat([signal_data_df, extra_feature_df], axis=1)


def capped_entry_order_value_ser(
    raw_weight_ser: pd.Series,
    entry_budget_float: float,
    previous_total_value_float: float,
    max_position_weight_float: float = MAX_POSITION_WEIGHT_FLOAT,
) -> pd.Series:
    """Normalize entrant weights and redistribute above-cap residual budget."""

    valid_raw_weight_ser = raw_weight_ser[
        np.isfinite(raw_weight_ser) & raw_weight_ser.gt(0.0)
    ].astype(float)
    if valid_raw_weight_ser.empty or entry_budget_float <= 0.0:
        return pd.Series(dtype=float)

    max_order_value_float = (
        float(previous_total_value_float) * float(max_position_weight_float)
    )
    remaining_budget_float = float(entry_budget_float)
    remaining_raw_weight_ser = valid_raw_weight_ser.copy()
    order_value_ser = pd.Series(0.0, index=valid_raw_weight_ser.index, dtype=float)

    while len(remaining_raw_weight_ser) > 0 and remaining_budget_float > 1e-9:
        proposed_order_value_ser = (
            remaining_budget_float
            * remaining_raw_weight_ser
            / remaining_raw_weight_ser.sum()
        )
        capped_symbol_list = proposed_order_value_ser[
            proposed_order_value_ser.gt(max_order_value_float + 1e-9)
        ].index.tolist()
        if not capped_symbol_list:
            order_value_ser.loc[remaining_raw_weight_ser.index] = (
                proposed_order_value_ser
            )
            break

        for symbol_str in capped_symbol_list:
            order_value_ser.loc[symbol_str] = max_order_value_float
            remaining_budget_float -= max_order_value_float
            remaining_raw_weight_ser = remaining_raw_weight_ser.drop(symbol_str)

    return order_value_ser


class HPIResearchSweepStrategy(HPIStatefulLongStrategy):
    """One controlled research strategy driven by a frozen VariantSpec."""

    def __init__(
        self,
        *,
        variant_spec_obj: VariantSpec,
        benchmark_symbol_str: str,
        signal_data_df: pd.DataFrame,
        capital_base_float: float,
        slippage_float: float = 0.00025,
    ) -> None:
        ranking_field_str = (
            TURNOVER_FIELD_STR
            if variant_spec_obj.ranking_mode_str == RANK_TURNOVER_STR
            else NATR_FIELD_STR
        )
        super().__init__(
            name=f"strategy_mr_hpi_sweep_{variant_spec_obj.key_str}",
            benchmarks=[benchmark_symbol_str],
            ranking_field_str=ranking_field_str,
            capital_base=capital_base_float,
            slippage=slippage_float,
        )
        self.variant_spec_obj = variant_spec_obj
        self._precomputed_signal_data_df = signal_data_df

    def compute_signals(self, pricing_data_df: pd.DataFrame) -> pd.DataFrame:
        return self._precomputed_signal_data_df

    def restrict_data(self, full_data_df: pd.DataFrame) -> tuple:
        """Return only the rows HPI actually consumes, preserving Vanilla timing."""

        open_price_ser = full_data_df.loc[
            self.current_bar,
            (slice(None), "Open"),
        ]
        open_price_ser.index = open_price_ser.index.get_level_values(0)
        if self.previous_bar is None:
            return None, None, open_price_ser

        # *** CRITICAL*** HPI iterate() consumes only the completed previous-bar
        # row. Returning a one-row frame removes an unused O(N^2) history slice
        # without exposing current_bar close or any future observation.
        close_row_ser = full_data_df.loc[self.previous_bar]
        previous_data_df = full_data_df.loc[[self.previous_bar]]
        return previous_data_df, close_row_ser, open_price_ser

    def _member_candidate_df(
        self,
        close_row_ser: pd.Series,
        member_symbol_set: set[str],
    ) -> pd.DataFrame:
        symbol_index = close_row_ser.index.get_level_values(0).astype(str)
        member_field_mask_arr = symbol_index.isin(member_symbol_set)
        if not member_field_mask_arr.any():
            return pd.DataFrame()
        return close_row_ser.loc[member_field_mask_arr].unstack()

    @staticmethod
    def _natr_rank_feature_ser(candidate_df: pd.DataFrame) -> pd.Series:
        if NATR_FIELD_STR not in candidate_df.columns:
            return pd.Series(np.nan, index=candidate_df.index, dtype=float)
        return candidate_df[NATR_FIELD_STR].astype(float).rank(
            method="average",
            pct=True,
        )

    @staticmethod
    def _natr_ensemble_feature_ser(candidate_df: pd.DataFrame) -> pd.Series:
        natr_field_list = [
            NATR_10_FIELD_STR,
            NATR_FIELD_STR,
            NATR_20_FIELD_STR,
        ]
        if any(field_str not in candidate_df.columns for field_str in natr_field_list):
            return pd.Series(np.nan, index=candidate_df.index, dtype=float)
        rank_df = candidate_df[natr_field_list].astype(float).rank(
            method="average",
            pct=True,
        )
        return rank_df.mean(axis=1, skipna=False)

    def _apply_liquidity_filter(
        self,
        candidate_df: pd.DataFrame,
    ) -> pd.DataFrame:
        liquidity_mode_str = self.variant_spec_obj.liquidity_mode_str
        if liquidity_mode_str == LIQUIDITY_NONE_STR:
            return candidate_df
        required_field_list = [RAW_PRICE_FIELD_STR, ADV_63_FIELD_STR]
        if any(field_str not in candidate_df.columns for field_str in required_field_list):
            return candidate_df.iloc[0:0]

        filtered_candidate_df = candidate_df.dropna(subset=required_field_list)
        filtered_candidate_df = filtered_candidate_df[
            filtered_candidate_df[RAW_PRICE_FIELD_STR].astype(float)
            > RAW_PRICE_MIN_FLOAT
        ]
        if liquidity_mode_str == LIQUIDITY_FIXED_STR:
            return filtered_candidate_df[
                filtered_candidate_df[ADV_63_FIELD_STR].astype(float)
                > ADV_63_MIN_FLOAT
            ]

        median_adv_float = float(
            candidate_df[ADV_63_FIELD_STR].dropna().astype(float).median()
        )
        if not np.isfinite(median_adv_float):
            return filtered_candidate_df.iloc[0:0]
        return filtered_candidate_df[
            filtered_candidate_df[ADV_63_FIELD_STR].astype(float)
            > median_adv_float
        ]

    def _opportunity_df(
        self,
        close_row_ser: pd.Series,
        member_symbol_set: set[str],
    ) -> pd.DataFrame:
        member_candidate_df = self._member_candidate_df(
            close_row_ser,
            member_symbol_set,
        )
        member_candidate_df[NATR_RANK_FIELD_STR] = self._natr_rank_feature_ser(
            member_candidate_df
        )
        member_candidate_df[NATR_ENSEMBLE_FIELD_STR] = (
            self._natr_ensemble_feature_ser(member_candidate_df)
        )
        candidate_df = self._apply_liquidity_filter(member_candidate_df)

        common_required_field_list = [
            "Close",
            "sma_200_price_ser",
            "ibs_value_ser",
        ]
        if self.variant_spec_obj.entry_mode_str == ENTRY_HORIZON_VOTE_STR:
            entry_required_field_list = [
                RETURN_2D_FIELD_STR,
                "return_3d_ser",
                RETURN_5D_FIELD_STR,
                HPI_2D_FIELD_STR,
                "hpi_value_ser",
                HPI_5D_FIELD_STR,
            ]
        else:
            entry_required_field_list = ["return_3d_ser", "hpi_value_ser"]

        ranking_field_str = {
            RANK_TURNOVER_STR: TURNOVER_FIELD_STR,
            RANK_NATR14_STR: NATR_FIELD_STR,
            RANK_NATR_ENSEMBLE_STR: NATR_ENSEMBLE_FIELD_STR,
        }[self.variant_spec_obj.ranking_mode_str]
        required_field_list = (
            common_required_field_list
            + entry_required_field_list
            + [ranking_field_str]
        )
        if any(field_str not in candidate_df.columns for field_str in required_field_list):
            return candidate_df.iloc[0:0]
        candidate_df = candidate_df.dropna(subset=required_field_list)

        if self.variant_spec_obj.entry_mode_str == ENTRY_HORIZON_VOTE_STR:
            hpi_vote_ser = (
                (
                    candidate_df[RETURN_2D_FIELD_STR].astype(float).lt(0.0)
                    & candidate_df[HPI_2D_FIELD_STR]
                    .astype(float)
                    .lt(HPI_THRESHOLD_FLOAT)
                ).astype(int)
                + (
                    candidate_df["return_3d_ser"].astype(float).lt(0.0)
                    & candidate_df["hpi_value_ser"]
                    .astype(float)
                    .lt(HPI_THRESHOLD_FLOAT)
                ).astype(int)
                + (
                    candidate_df[RETURN_5D_FIELD_STR].astype(float).lt(0.0)
                    & candidate_df[HPI_5D_FIELD_STR]
                    .astype(float)
                    .lt(HPI_THRESHOLD_FLOAT)
                ).astype(int)
            )
            candidate_df = candidate_df[hpi_vote_ser.ge(2)]
        else:
            candidate_df = candidate_df[
                candidate_df["hpi_value_ser"].astype(float)
                < HPI_THRESHOLD_FLOAT
            ]
            candidate_df = candidate_df[
                candidate_df["return_3d_ser"].astype(float) < 0.0
            ]

        candidate_df = candidate_df[
            candidate_df["ibs_value_ser"].astype(float) < MAX_ENTRY_IBS_FLOAT
        ]
        candidate_df = candidate_df[
            candidate_df["Close"].astype(float)
            > candidate_df["sma_200_price_ser"].astype(float)
        ]
        candidate_df = candidate_df.assign(
            symbol_str=candidate_df.index.astype(str)
        ).sort_values(
            by=[ranking_field_str, "symbol_str"],
            ascending=[False, True],
            kind="mergesort",
        )
        return candidate_df

    def get_opportunity_list(
        self,
        close_row_ser: pd.Series,
        member_symbol_set: set[str] | None = None,
    ) -> list[str]:
        if self.universe_df is None:
            raise RuntimeError("HPI sweep requires a point-in-time universe.")
        if member_symbol_set is None:
            member_symbol_set = get_asof_universe_symbol_set(
                self.universe_df,
                pd.Timestamp(self.previous_bar),
            )
        return self._opportunity_df(
            close_row_ser,
            member_symbol_set,
        ).index.astype(str).tolist()

    def _exit_signal_bool(
        self,
        ibs_value_float: float,
        rsi2_value_float: float,
    ) -> bool:
        exit_for_ibs_bool = (
            pd.notna(ibs_value_float)
            and float(ibs_value_float) > EXIT_IBS_THRESHOLD_FLOAT
        )
        exit_for_rsi_bool = (
            pd.notna(rsi2_value_float)
            and float(rsi2_value_float) > EXIT_RSI2_THRESHOLD_FLOAT
        )
        if self.variant_spec_obj.exit_mode_str == EXIT_IBS_STR:
            return exit_for_ibs_bool
        if self.variant_spec_obj.exit_mode_str == EXIT_RSI_STR:
            return exit_for_rsi_bool
        return exit_for_ibs_bool or exit_for_rsi_bool

    def _entry_order_value_ser(
        self,
        selected_candidate_df: pd.DataFrame,
    ) -> pd.Series:
        selected_count_int = len(selected_candidate_df)
        entry_budget_float = (
            self.previous_total_value
            * float(selected_count_int)
            / float(self.max_positions_int)
        )
        sizing_mode_str = self.variant_spec_obj.sizing_mode_str
        if sizing_mode_str == SIZING_EQUAL_STR:
            return pd.Series(
                self.previous_total_value / float(self.max_positions_int),
                index=selected_candidate_df.index,
                dtype=float,
            )
        if sizing_mode_str == SIZING_INVERSE_NATR_STR:
            raw_weight_ser = 1.0 / selected_candidate_df[
                NATR_FIELD_STR
            ].astype(float)
        else:
            raw_weight_ser = selected_candidate_df[
                NATR_RANK_FIELD_STR
            ].astype(float)
        return capped_entry_order_value_ser(
            raw_weight_ser,
            entry_budget_float=entry_budget_float,
            previous_total_value_float=self.previous_total_value,
        )

    def iterate(
        self,
        data_df: pd.DataFrame,
        close_row_ser: pd.Series,
        open_price_ser: pd.Series,
    ) -> None:
        if data_df is None or close_row_ser is None:
            return
        if self.universe_df is None:
            raise RuntimeError("HPI sweep requires a point-in-time universe.")

        decision_date_ts = pd.Timestamp(self.previous_bar)
        member_symbol_set = get_asof_universe_symbol_set(
            self.universe_df,
            decision_date_ts,
        )
        long_position_ser = self.get_positions()
        long_position_ser = long_position_ser[long_position_ser > 0]
        self.pending_exit_symbol_set.intersection_update(
            set(long_position_ser.index.astype(str))
        )
        long_slots_int = self.max_positions_int - len(long_position_ser)

        for symbol_str in long_position_ser.index.astype(str):
            exit_signal_bool = self._exit_signal_bool(
                close_row_ser.get((symbol_str, "ibs_value_ser"), np.nan),
                close_row_ser.get((symbol_str, "rsi2_value_ser"), np.nan),
            )
            if exit_signal_bool or symbol_str not in member_symbol_set:
                self.pending_exit_symbol_set.add(symbol_str)

            current_open_float = open_price_ser.get(symbol_str, np.nan)
            has_tradable_open_bool = (
                pd.notna(current_open_float)
                and np.isfinite(float(current_open_float))
            )
            if symbol_str in self.pending_exit_symbol_set and has_tradable_open_bool:
                self.order_target_value(
                    symbol_str,
                    0.0,
                    trade_id=self.current_trade_map[symbol_str],
                )
                long_slots_int += 1

        opportunity_df = self._opportunity_df(
            close_row_ser,
            member_symbol_set,
        )
        selected_symbol_list: list[str] = []
        for symbol_str in opportunity_df.index.astype(str):
            if len(selected_symbol_list) >= long_slots_int:
                break
            if self.get_position(symbol_str) == 0:
                selected_symbol_list.append(symbol_str)
        if not selected_symbol_list:
            return

        selected_candidate_df = opportunity_df.loc[selected_symbol_list]
        order_value_ser = self._entry_order_value_ser(selected_candidate_df)
        for symbol_str, order_value_float in order_value_ser.items():
            self.trade_id_int += 1
            self.current_trade_map[symbol_str] = self.trade_id_int
            self.order_value(
                symbol_str,
                float(order_value_float),
                trade_id=self.trade_id_int,
            )


def _metric_float(strategy_obj: HPIResearchSweepStrategy, metric_name_str: str) -> float:
    return float(strategy_obj.summary.loc[metric_name_str, "Strategy"])


def _normal_two_sided_p_float(t_stat_float: float) -> float:
    return math.erfc(abs(float(t_stat_float)) / math.sqrt(2.0))


def build_variant_summary_dict(
    strategy_obj: HPIResearchSweepStrategy,
    variant_spec_obj: VariantSpec,
    multiple_test_count_int: int,
) -> dict[str, object]:
    transaction_df = strategy_obj.get_transactions()
    alpha_t_float = _metric_float(strategy_obj, "Alpha HAC t-stat")
    raw_p_float = _normal_two_sided_p_float(alpha_t_float)
    accounting_policy_dict = dict(strategy_obj._accounting_policy_dict)
    return {
        **asdict(variant_spec_obj),
        "start_str": str(pd.Timestamp(strategy_obj.results.index.min()).date()),
        "end_str": str(pd.Timestamp(strategy_obj.results.index.max()).date()),
        "annual_return_pct": _metric_float(strategy_obj, "Return (Ann.) [%]"),
        "annual_volatility_pct": _metric_float(
            strategy_obj,
            "Volatility (Ann.) [%]",
        ),
        "sharpe_float": _metric_float(strategy_obj, "Sharpe Ratio"),
        "max_drawdown_pct": _metric_float(strategy_obj, "Max. Drawdown [%]"),
        "mar_float": _metric_float(strategy_obj, "MAR Ratio"),
        "exposure_pct": _metric_float(strategy_obj, "Exposure Time [%]"),
        "turnover_annual_pct": _metric_float(strategy_obj, "Turnover (Ann.) [%]"),
        "cost_drag_annual_pct": _metric_float(strategy_obj, "Cost Drag (Ann.) [%]"),
        "final_equity_float": _metric_float(strategy_obj, "Final [$]"),
        "trade_count_int": int(len(strategy_obj._trades)),
        "transaction_count_int": int(len(transaction_df)),
        "synthetic_forced_liquidation_count_int": int(
            transaction_df["order_id"].eq(-1).sum()
        ),
        "dividend_cash_net_float": float(
            accounting_policy_dict.get("dividend_cash_net_total_float", 0.0)
        ),
        "negative_cash_day_count_int": int(
            accounting_policy_dict.get("negative_cash_day_count_int", 0)
        ),
        "alpha_annual_pct": _metric_float(strategy_obj, "Alpha (Ann.) [%]"),
        "alpha_hac_t_float": alpha_t_float,
        "alpha_p_normal_float": raw_p_float,
        "alpha_p_bonferroni_float": min(
            raw_p_float * float(multiple_test_count_int),
            1.0,
        ),
    }


def _daily_result_df(
    run_dir_obj: Path,
    universe_key_str: str,
    variant_key_str: str,
) -> pd.DataFrame:
    result_path_obj = (
        run_dir_obj
        / universe_key_str
        / "daily_results"
        / f"{variant_key_str}.csv"
    )
    return pd.read_csv(result_path_obj, index_col=0, parse_dates=True)


def _paired_hac_stat_dict(
    variant_return_ser: pd.Series,
    baseline_return_ser: pd.Series,
) -> dict[str, float | int]:
    # *** CRITICAL*** This is report-only paired realized-return inference.
    # Inner alignment uses no fills and cannot feed signal or execution logic.
    paired_return_df = pd.concat(
        [
            variant_return_ser.astype(float).rename("variant_return"),
            baseline_return_ser.astype(float).rename("baseline_return"),
        ],
        axis=1,
        join="inner",
    ).replace([np.inf, -np.inf], np.nan).dropna()
    return_delta_ser = (
        paired_return_df["variant_return"]
        - paired_return_df["baseline_return"]
    )
    observation_count_int = len(return_delta_ser)
    result_dict: dict[str, float | int] = {
        "paired_observation_count_int": observation_count_int,
        "mean_return_delta_annual_pct": np.nan,
        "paired_hac_lag_int": 0,
        "paired_hac_t_float": np.nan,
        "paired_hac_p_float": np.nan,
    }
    if observation_count_int < 252:
        return result_dict
    return_delta_arr = return_delta_ser.to_numpy(dtype=float)
    if float(np.var(return_delta_arr, ddof=1)) <= np.finfo(float).eps:
        return result_dict

    hac_lag_int = int(
        np.floor(4.0 * (observation_count_int / 100.0) ** (2.0 / 9.0))
    )
    regression_result_obj = sm.OLS(
        return_delta_arr,
        np.ones((observation_count_int, 1), dtype=float),
    ).fit(
        cov_type="HAC",
        cov_kwds={"maxlags": hac_lag_int},
    )
    result_dict.update(
        {
            "mean_return_delta_annual_pct": (
                float(return_delta_ser.mean()) * 252.0 * 100.0
            ),
            "paired_hac_lag_int": hac_lag_int,
            "paired_hac_t_float": float(regression_result_obj.tvalues[0]),
            "paired_hac_p_float": float(regression_result_obj.pvalues[0]),
        }
    )
    return result_dict


def build_baseline_comparison_df(
    run_dir_obj: Path,
    variant_summary_df: pd.DataFrame,
) -> pd.DataFrame:
    comparison_dict_list: list[dict[str, object]] = []
    for universe_key_str, universe_summary_df in variant_summary_df.groupby(
        "universe_key_str",
        sort=False,
    ):
        baseline_row_df = universe_summary_df[
            universe_summary_df["key_str"].str.endswith("_baseline")
        ]
        if len(baseline_row_df) != 1:
            raise RuntimeError(
                f"Expected one {universe_key_str} baseline, found "
                f"{len(baseline_row_df)}."
            )
        baseline_row_ser = baseline_row_df.iloc[0]
        baseline_key_str = str(baseline_row_ser["key_str"])
        baseline_result_df = _daily_result_df(
            run_dir_obj,
            universe_key_str,
            baseline_key_str,
        )
        for _, variant_row_ser in universe_summary_df.iterrows():
            variant_key_str = str(variant_row_ser["key_str"])
            variant_result_df = _daily_result_df(
                run_dir_obj,
                universe_key_str,
                variant_key_str,
            )
            paired_stat_dict = _paired_hac_stat_dict(
                variant_result_df["daily_returns"],
                baseline_result_df["daily_returns"],
            )
            raw_p_float = float(paired_stat_dict["paired_hac_p_float"])
            comparison_dict_list.append(
                {
                    "key_str": variant_key_str,
                    "universe_key_str": universe_key_str,
                    "baseline_key_str": baseline_key_str,
                    "cagr_delta_vs_baseline_pct": (
                        float(variant_row_ser["annual_return_pct"])
                        - float(baseline_row_ser["annual_return_pct"])
                    ),
                    **paired_stat_dict,
                    "paired_hac_p_bonferroni_float": (
                        min(
                            raw_p_float * float(DECLARED_VARIANT_COUNT_INT),
                            1.0,
                        )
                        if np.isfinite(raw_p_float)
                        else np.nan
                    ),
                }
            )
    return pd.DataFrame(comparison_dict_list)


SUBPERIOD_TUPLE = (
    ("2004_2011", "2004-01-01", "2011-12-31"),
    ("2012_2019", "2012-01-01", "2019-12-31"),
    ("2020_present", "2020-01-01", None),
)


def build_subperiod_summary_df(
    run_dir_obj: Path,
    variant_summary_df: pd.DataFrame,
) -> pd.DataFrame:
    subperiod_dict_list: list[dict[str, object]] = []
    for _, variant_row_ser in variant_summary_df.iterrows():
        variant_key_str = str(variant_row_ser["key_str"])
        universe_key_str = str(variant_row_ser["universe_key_str"])
        result_df = _daily_result_df(
            run_dir_obj,
            universe_key_str,
            variant_key_str,
        )
        daily_return_ser = result_df["daily_returns"].astype(float)
        for (
            subperiod_key_str,
            subperiod_start_str,
            subperiod_end_str,
        ) in SUBPERIOD_TUPLE:
            period_return_ser = daily_return_ser.loc[
                pd.Timestamp(subperiod_start_str) :
                (
                    pd.Timestamp(subperiod_end_str)
                    if subperiod_end_str is not None
                    else daily_return_ser.index.max()
                )
            ].replace([np.inf, -np.inf], np.nan).dropna()
            observation_count_int = len(period_return_ser)
            if observation_count_int < 2:
                continue
            growth_float = float((1.0 + period_return_ser).prod())
            annual_return_pct = (
                growth_float ** (252.0 / observation_count_int) - 1.0
            ) * 100.0
            annual_volatility_pct = (
                float(period_return_ser.std(ddof=1)) * math.sqrt(252.0) * 100.0
            )
            sharpe_float = (
                float(period_return_ser.mean())
                / float(period_return_ser.std(ddof=1))
                * math.sqrt(252.0)
            )
            wealth_arr = (1.0 + period_return_ser).cumprod().to_numpy(dtype=float)
            wealth_with_start_arr = np.concatenate(([1.0], wealth_arr))
            drawdown_arr = (
                wealth_with_start_arr
                / np.maximum.accumulate(wealth_with_start_arr)
                - 1.0
            )
            subperiod_dict_list.append(
                {
                    "key_str": variant_key_str,
                    "universe_key_str": universe_key_str,
                    "subperiod_key_str": subperiod_key_str,
                    "start_str": str(period_return_ser.index.min().date()),
                    "end_str": str(period_return_ser.index.max().date()),
                    "observation_count_int": observation_count_int,
                    "annual_return_pct": annual_return_pct,
                    "annual_volatility_pct": annual_volatility_pct,
                    "sharpe_float": sharpe_float,
                    "max_drawdown_pct": float(drawdown_arr.min() * 100.0),
                }
            )

    subperiod_summary_df = pd.DataFrame(subperiod_dict_list)
    baseline_return_df = (
        subperiod_summary_df[
            subperiod_summary_df["key_str"].str.endswith("_baseline")
        ][
            [
                "universe_key_str",
                "subperiod_key_str",
                "annual_return_pct",
            ]
        ]
        .rename(columns={"annual_return_pct": "baseline_annual_return_pct"})
    )
    subperiod_summary_df = subperiod_summary_df.merge(
        baseline_return_df,
        on=["universe_key_str", "subperiod_key_str"],
        how="left",
        validate="many_to_one",
    )
    subperiod_summary_df["cagr_delta_vs_baseline_pct"] = (
        subperiod_summary_df["annual_return_pct"]
        - subperiod_summary_df["baseline_annual_return_pct"]
    )
    return subperiod_summary_df


def compute_hpi_breadth_ser(
    signal_data_df: pd.DataFrame,
    universe_df: pd.DataFrame,
) -> pd.Series:
    """Return the daily share of PIT members with a baseline HPI event."""

    symbol_list = [
        symbol_str
        for symbol_str in universe_df.columns.astype(str)
        if (symbol_str, "hpi_value_ser") in signal_data_df.columns
        and (symbol_str, "return_3d_ser") in signal_data_df.columns
    ]
    event_df = pd.DataFrame(
        {
            symbol_str: (
                signal_data_df[(symbol_str, "hpi_value_ser")]
                .astype(float)
                .lt(HPI_THRESHOLD_FLOAT)
                & signal_data_df[(symbol_str, "return_3d_ser")]
                .astype(float)
                .lt(0.0)
            )
            for symbol_str in symbol_list
        },
        index=signal_data_df.index,
    )
    # *** CRITICAL*** PIT membership is aligned to the same decision date.
    # Missing membership rows become non-members; no future row is backfilled.
    membership_df = (
        universe_df.reindex(signal_data_df.index)
        .reindex(columns=symbol_list)
        .fillna(0)
        .astype(int)
    )
    member_count_ser = membership_df.eq(1).sum(axis=1)
    event_count_ser = (event_df & membership_df.eq(1)).sum(axis=1)
    return event_count_ser / member_count_ser.replace(0, np.nan)


def build_breadth_trade_summary_df(
    strategy_obj: HPIResearchSweepStrategy,
    breadth_ser: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    trade_df = strategy_obj._trades.reset_index()
    signal_index = pd.DatetimeIndex(breadth_ser.index)
    decision_date_list: list[pd.Timestamp | pd.NaT] = []
    breadth_value_list: list[float] = []
    for start_date_obj in trade_df["start"]:
        start_date_ts = pd.Timestamp(start_date_obj)
        start_location_int = int(signal_index.searchsorted(start_date_ts))
        if (
            start_location_int >= len(signal_index)
            or signal_index[start_location_int] != start_date_ts
            or start_location_int == 0
        ):
            decision_date_list.append(pd.NaT)
            breadth_value_list.append(np.nan)
            continue
        decision_date_ts = signal_index[start_location_int - 1]
        decision_date_list.append(decision_date_ts)
        breadth_value_list.append(float(breadth_ser.loc[decision_date_ts]))

    trade_df["decision_date"] = decision_date_list
    trade_df["hpi_breadth"] = breadth_value_list
    trade_df["breadth_bucket"] = pd.cut(
        trade_df["hpi_breadth"],
        bins=[-np.inf, 0.01, 0.05, np.inf],
        labels=["low_lt_1pct", "medium_1_to_5pct", "high_gt_5pct"],
        right=False,
    )
    valid_trade_df = trade_df.dropna(subset=["breadth_bucket", "return"])
    summary_df = (
        valid_trade_df.groupby("breadth_bucket", observed=True)
        .agg(
            trade_count_int=("return", "size"),
            average_trade_return_float=("return", "mean"),
            median_trade_return_float=("return", "median"),
            win_rate_float=("return", lambda return_ser: float(return_ser.gt(0).mean())),
            average_breadth_float=("hpi_breadth", "mean"),
        )
        .reset_index()
    )
    return trade_df, summary_df


def _write_json(path_obj: Path, payload_dict: dict[str, object]) -> None:
    path_obj.write_text(
        json.dumps(payload_dict, indent=2, default=str),
        encoding="utf-8",
    )


def run_universe_sweep(
    *,
    universe_spec_obj: UniverseSpec,
    selected_variant_spec_list: list[VariantSpec],
    run_dir_obj: Path,
    start_date_str: str,
    end_date_str: str | None,
    capital_base_float: float,
    slippage_float: float,
    show_progress_bool: bool,
    multiple_test_count_int: int,
) -> list[dict[str, object]]:
    _, universe_df, pricing_data_df = load_exact_hpi_inputs(
        indexname_str=universe_spec_obj.indexname_str,
        benchmark_symbol_str=universe_spec_obj.benchmark_symbol_str,
        start_date_str=DEFAULT_FEATURE_START_DATE_STR,
        end_date_str=end_date_str,
    )
    symbol_list = pricing_data_df.columns.get_level_values(0).unique().astype(str)
    pricing_data_df.attrs["norgate_adjustment_by_symbol_dict"] = {
        symbol_str: (
            "TOTALRETURN"
            if symbol_str == universe_spec_obj.benchmark_symbol_str
            else "CAPITALSPECIAL"
        )
        for symbol_str in symbol_list
    }
    signal_data_df = compute_hpi_sweep_signal_data_df(pricing_data_df)
    validate_variant_feature_contract(
        signal_data_df,
        selected_variant_spec_list,
        start_date_str,
    )
    calendar_idx = pricing_data_df.index[
        pricing_data_df.index >= pd.Timestamp(start_date_str)
    ]
    universe_run_dir_obj = run_dir_obj / universe_spec_obj.key_str
    transaction_dir_obj = universe_run_dir_obj / "transactions"
    result_dir_obj = universe_run_dir_obj / "daily_results"
    transaction_dir_obj.mkdir(parents=True, exist_ok=True)
    result_dir_obj.mkdir(parents=True, exist_ok=True)
    summary_dict_list: list[dict[str, object]] = []
    baseline_strategy_obj: HPIResearchSweepStrategy | None = None

    for variant_spec_obj in selected_variant_spec_list:
        print(f"running {variant_spec_obj.key_str}...", flush=True)
        strategy_obj = HPIResearchSweepStrategy(
            variant_spec_obj=variant_spec_obj,
            benchmark_symbol_str=universe_spec_obj.benchmark_symbol_str,
            signal_data_df=signal_data_df,
            capital_base_float=capital_base_float,
            slippage_float=slippage_float,
        )
        strategy_obj.universe_df = universe_df
        run_daily(
            strategy_obj,
            pricing_data_df,
            calendar_idx,
            show_progress=show_progress_bool,
            show_signal_progress_bool=False,
        )
        strategy_obj.get_transactions().to_csv(
            transaction_dir_obj / f"{variant_spec_obj.key_str}.csv",
            index=False,
        )
        strategy_obj.results.to_csv(
            result_dir_obj / f"{variant_spec_obj.key_str}.csv",
        )
        summary_dict_list.append(
            build_variant_summary_dict(
                strategy_obj,
                variant_spec_obj,
                multiple_test_count_int,
            )
        )
        if variant_spec_obj.key_str.endswith("_baseline"):
            baseline_strategy_obj = strategy_obj
        strategy_obj._precomputed_signal_data_df = None

    breadth_ser = compute_hpi_breadth_ser(signal_data_df, universe_df)
    breadth_ser.rename("hpi_breadth").to_csv(
        universe_run_dir_obj / "hpi_breadth_by_date.csv"
    )
    if baseline_strategy_obj is not None:
        breadth_trade_df, breadth_summary_df = build_breadth_trade_summary_df(
            baseline_strategy_obj,
            breadth_ser,
        )
        breadth_trade_df.to_csv(
            universe_run_dir_obj / "hpi_breadth_trades.csv",
            index=False,
        )
        breadth_summary_df.to_csv(
            universe_run_dir_obj / "hpi_breadth_trade_summary.csv",
            index=False,
        )
    return summary_dict_list


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", default=DEFAULT_START_DATE_STR)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--capital", type=float, default=DEFAULT_CAPITAL_FLOAT)
    parser.add_argument("--slippage-bps", type=float, default=2.5)
    parser.add_argument(
        "--output-dir",
        default="results/research/hpi_variant_sweep",
    )
    parser.add_argument(
        "--variant",
        action="append",
        default=[],
        help=(
            "Run a named variant; repeat to select more than one. "
            "Its universe baseline is included automatically."
        ),
    )
    parser.add_argument("--show-progress", action="store_true")
    return parser.parse_args()


def slippage_bps_to_float(slippage_bps_float: float) -> float:
    if (
        not np.isfinite(slippage_bps_float)
        or slippage_bps_float < 0.0
        or slippage_bps_float >= 10_000.0
    ):
        raise ValueError("--slippage-bps must be finite and in [0, 10000).")
    return float(slippage_bps_float) / 10_000.0


def resolve_selected_variant_spec_list(
    selected_variant_key_set: set[str],
) -> list[VariantSpec]:
    known_variant_key_set = {
        variant_spec_obj.key_str for variant_spec_obj in VARIANT_SPEC_TUPLE
    }
    unknown_variant_key_set = selected_variant_key_set - known_variant_key_set
    if unknown_variant_key_set:
        raise ValueError(
            f"Unknown variants: {sorted(unknown_variant_key_set)}"
        )
    if not selected_variant_key_set:
        return list(VARIANT_SPEC_TUPLE)

    selected_universe_key_set = {
        variant_spec_obj.universe_key_str
        for variant_spec_obj in VARIANT_SPEC_TUPLE
        if variant_spec_obj.key_str in selected_variant_key_set
    }
    resolved_variant_key_set = set(selected_variant_key_set)
    resolved_variant_key_set.update(
        variant_spec_obj.key_str
        for variant_spec_obj in VARIANT_SPEC_TUPLE
        if variant_spec_obj.universe_key_str in selected_universe_key_set
        and variant_spec_obj.key_str.endswith("_baseline")
    )
    return [
        variant_spec_obj
        for variant_spec_obj in VARIANT_SPEC_TUPLE
        if variant_spec_obj.key_str in resolved_variant_key_set
    ]


def main() -> None:
    args_obj = parse_args()
    slippage_float = slippage_bps_to_float(args_obj.slippage_bps)
    selected_variant_spec_list = resolve_selected_variant_spec_list(
        set(args_obj.variant)
    )
    executed_variant_count_int = len(selected_variant_spec_list)
    timestamp_str = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    run_dir_obj = Path(args_obj.output_dir) / timestamp_str
    run_dir_obj.mkdir(parents=True, exist_ok=False)

    summary_dict_list: list[dict[str, object]] = []
    for universe_key_str, universe_spec_obj in UNIVERSE_SPEC_DICT.items():
        universe_variant_spec_list = [
            variant_spec_obj
            for variant_spec_obj in selected_variant_spec_list
            if variant_spec_obj.universe_key_str == universe_key_str
        ]
        if not universe_variant_spec_list:
            continue
        summary_dict_list.extend(
            run_universe_sweep(
                universe_spec_obj=universe_spec_obj,
                selected_variant_spec_list=universe_variant_spec_list,
                run_dir_obj=run_dir_obj,
                start_date_str=args_obj.start_date,
                end_date_str=args_obj.end_date,
                capital_base_float=float(args_obj.capital),
                slippage_float=slippage_float,
                show_progress_bool=bool(args_obj.show_progress),
                multiple_test_count_int=DECLARED_VARIANT_COUNT_INT,
            )
        )

    summary_df = pd.DataFrame(summary_dict_list)
    summary_df.to_csv(run_dir_obj / "variant_summary.csv", index=False)
    build_baseline_comparison_df(
        run_dir_obj,
        summary_df,
    ).to_csv(
        run_dir_obj / "baseline_comparison.csv",
        index=False,
    )
    build_subperiod_summary_df(
        run_dir_obj,
        summary_df,
    ).to_csv(
        run_dir_obj / "subperiod_summary.csv",
        index=False,
    )
    pd.DataFrame(STATUS_ROW_DICT_LIST).to_csv(
        run_dir_obj / "variant_status.csv",
        index=False,
    )
    _write_json(
        run_dir_obj / "metadata.json",
        {
            "artifact_type_str": "hpi_variant_sweep",
            "created_at_str": datetime.now().isoformat(),
            "start_date_str": args_obj.start_date,
            "end_date_str": args_obj.end_date,
            "capital_base_float": float(args_obj.capital),
            "slippage_float": slippage_float,
            "slippage_bps_float": float(args_obj.slippage_bps),
            "commission_per_share_float": 0.005,
            "commission_minimum_float": 1.0,
            "dividend_withholding_rate_float": 0.25,
            "execution_timing_str": "decision_after_close_T_fill_open_T_plus_1",
            "sizing_timing_str": "new_entries_only_no_incumbent_rebalance",
            "declared_variant_count_int": DECLARED_VARIANT_COUNT_INT,
            "executed_variant_count_int": executed_variant_count_int,
            "executed_variant_key_list": [
                variant_spec_obj.key_str
                for variant_spec_obj in selected_variant_spec_list
            ],
            "breadth_bucket_definition_dict": {
                "low_lt_1pct": "breadth < 1%",
                "medium_1_to_5pct": "1% <= breadth < 5%",
                "high_gt_5pct": "breadth >= 5%",
            },
            "status_row_dict_list": STATUS_ROW_DICT_LIST,
        },
    )
    print(summary_df.to_string(index=False))
    print(f"saved sweep to {run_dir_obj.resolve()}")


if __name__ == "__main__":
    main()
