"""
Run the frozen AMAF turnover/stability improvement sweep.

This is research-only. It does not add BENCH variants and does not touch LIVE,
release, scheduler, allocation, or broker configuration.

Declared candidate search space:

1. ``amaf_buffered_20_30`` keeps the original learned AMAF forecast, enters
   enough names to fill the highest 20%, and retains incumbents while they
   remain in the highest 30%.
2. ``static_amaf_composite`` replaces the monthly OLS forecast with the
   negative mean cross-sectional z-score of the same eleven SMA/Close ratios.

The original AMAF, eligible-universe equal weight, and classic 12-1 momentum
are controls rather than additional promotion candidates.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import statsmodels.api as sm

REPO_ROOT_PATH = Path(__file__).resolve().parents[2]
if str(REPO_ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_PATH))

from alpha.engine.backtest import run_daily
from alpha.engine.order import MarketOrder
from alpha.engine.report import build_research_output_path, save_results
from strategies.momentum.adaptive_moving_average_factor import (
    AdaptiveMovingAverageFactorConfig,
    AdaptiveMovingAverageFactorSignalBundle,
    AdaptiveMovingAverageFactorStrategy,
    build_adaptive_moving_average_factor_signal_bundle,
    build_monthly_sma_ratio_by_lookback_dict,
    get_adaptive_moving_average_factor_data,
)
from strategies.momentum.strategy_mo_amaf_nasdaq100 import (
    DEFAULT_CONFIG as NASDAQ100_DEFAULT_CONFIG,
)
from strategies.momentum.strategy_mo_amaf_russell1000 import (
    DEFAULT_CONFIG as RUSSELL1000_DEFAULT_CONFIG,
)
from strategies.momentum.strategy_mo_atr_normalized_ndx import (
    get_monthly_decision_close_df,
)


BASELINE_VARIANT_KEY_STR = "amaf_baseline"
BUFFERED_VARIANT_KEY_STR = "amaf_buffered_20_30"
STATIC_VARIANT_KEY_STR = "static_amaf_composite"
EQUAL_WEIGHT_CONTROL_KEY_STR = "eligible_equal_weight_control"
MOMENTUM_CONTROL_KEY_STR = "classic_12_1_momentum_control"

VARIANT_KEY_TUPLE = (
    BASELINE_VARIANT_KEY_STR,
    BUFFERED_VARIANT_KEY_STR,
    STATIC_VARIANT_KEY_STR,
    EQUAL_WEIGHT_CONTROL_KEY_STR,
    MOMENTUM_CONTROL_KEY_STR,
)
CANDIDATE_VARIANT_KEY_TUPLE = (
    BUFFERED_VARIANT_KEY_STR,
    STATIC_VARIANT_KEY_STR,
)

UNIVERSE_CONFIG_BY_KEY_DICT = {
    "russell1000": RUSSELL1000_DEFAULT_CONFIG,
    "nasdaq100": NASDAQ100_DEFAULT_CONFIG,
}

# ``slippage_float`` is a per-side penalty. The platform row is an exact engine
# run. The two higher-cost rows are report-only incremental turnover stresses
# derived from that run; shares and fills are not rerun.
COST_TIER_SLIPPAGE_BY_KEY_DICT = {
    "platform": 0.00025,
    "round_trip_20bps": 0.00100,
    "round_trip_50bps": 0.00250,
}

SUBPERIOD_TUPLE = (
    ("2000_2009", "2000-01-01", "2009-12-31"),
    ("2010_2017", "2010-01-01", "2017-12-31"),
    ("2018_latest", "2018-01-01", None),
)

DECLARED_CANDIDATE_COUNT_INT = len(CANDIDATE_VARIANT_KEY_TUPLE)
DECLARED_PRIMARY_HYPOTHESIS_COUNT_INT = (
    DECLARED_CANDIDATE_COUNT_INT * len(UNIVERSE_CONFIG_BY_KEY_DICT)
)
RESEARCH_CAPITAL_BASE_FLOAT = 100_000_000.0

RESEARCH_SPEC_DICT = {
    "status_str": "frozen_before_results",
    "research_only_bool": True,
    "candidate_search_count_int": DECLARED_CANDIDATE_COUNT_INT,
    "candidate_variant_key_list": list(CANDIDATE_VARIANT_KEY_TUPLE),
    "control_variant_key_list": [
        BASELINE_VARIANT_KEY_STR,
        EQUAL_WEIGHT_CONTROL_KEY_STR,
        MOMENTUM_CONTROL_KEY_STR,
    ],
    "universe_key_list": list(UNIVERSE_CONFIG_BY_KEY_DICT),
    "decision_timing_str": "final common market Close_T",
    "execution_timing_str": "first tradable Open_T+1",
    "stock_adjustment_str": "CAPITALSPECIAL",
    "raw_price_floor_float": 5.0,
    "research_capital_base_float": RESEARCH_CAPITAL_BASE_FLOAT,
    "research_capital_reason_str": (
        "large research notional limits whole-share distortion in the "
        "eligible-universe equal-weight factor control"
    ),
    "buffer_entry_fraction_float": 0.20,
    "buffer_retention_fraction_float": 0.30,
    "static_composite_formula_str": (
        "-mean_L(zscore_cross_section_T(SMA_L_T / Close_T))"
    ),
    "momentum_control_formula_str": "Close_(T-21 observed) / Close_(T-252 observed) - 1",
    "cost_tier_slippage_by_key_dict": COST_TIER_SLIPPAGE_BY_KEY_DICT,
    "cost_stress_method_str": (
        "platform is an exact engine run at 2.5 bps per side plus configured "
        "commissions; 20/50 bps round-trip rows subtract the incremental "
        "reference-notional slippage from realized daily returns using the "
        "platform transaction path; shares and fills are not rerun"
    ),
    "commission_per_share_float": RUSSELL1000_DEFAULT_CONFIG.commission_per_share_float,
    "commission_minimum_float": RUSSELL1000_DEFAULT_CONFIG.commission_minimum_float,
    "backtest_start_date_str": RUSSELL1000_DEFAULT_CONFIG.backtest_start_date_str,
    "subperiod_list": [
        {
            "subperiod_key_str": subperiod_key_str,
            "start_date_str": start_date_str,
            "end_date_str": end_date_str,
        }
        for subperiod_key_str, start_date_str, end_date_str in SUBPERIOD_TUPLE
    ],
    "buffer_gate_dict": {
        "minimum_turnover_reduction_fraction_float": 0.25,
        "minimum_cagr_retention_fraction_float": 0.95,
        "minimum_sharpe_delta_float": -0.05,
        "maximum_drawdown_worsening_pct_float": 2.0,
        "positive_paired_mean_at_20bps_bool": True,
        "positive_paired_mean_at_50bps_bool": True,
        "missing_liquidation_count_not_worse_bool": True,
    },
    "static_gate_dict": {
        "beat_baseline_and_equal_weight_at_all_cost_tiers_bool": True,
        "platform_sharpe_above_baseline_bool": True,
        "missing_liquidation_count_not_worse_bool": True,
    },
    "inference_str": (
        "paired monthly return HAC; Bonferroni across two candidates and two universes"
    ),
    "promotion_boundary_str": (
        "post-hoc historical results and approximate cost stresses cannot "
        "authorize promotion; exact cost-tier reruns and frozen forward-shadow "
        "evidence are required"
    ),
}


def _stable_ranked_symbol_list(
    score_ser: pd.Series,
) -> list[str]:
    clean_score_ser = (
        pd.to_numeric(score_ser, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .astype(float)
    )
    ranking_df = pd.DataFrame(
        {
            "symbol_str": clean_score_ser.index.astype(str),
            "score_float": clean_score_ser.to_numpy(dtype=float),
        }
    )
    ranking_df = ranking_df.sort_values(
        ["score_float", "symbol_str"],
        ascending=[True, True],
        kind="mergesort",
    )
    return ranking_df["symbol_str"].astype(str).tolist()


def _bundle_from_score_df(
    baseline_bundle_obj: AdaptiveMovingAverageFactorSignalBundle,
    score_df: pd.DataFrame,
    variant_key_str: str,
) -> AdaptiveMovingAverageFactorSignalBundle:
    target_weight_row_dict: dict[pd.Timestamp, pd.Series] = {}
    forecast_record_list: list[dict[str, object]] = []
    coverage_record_list: list[dict[str, object]] = []

    for decision_date_obj, baseline_forecast_df in (
        baseline_bundle_obj.forecast_df.groupby("decision_date_ts", sort=True)
    ):
        decision_date_ts = pd.Timestamp(decision_date_obj)
        eligible_symbol_list = (
            baseline_forecast_df["symbol_str"].astype(str).tolist()
        )
        selected_count_int = int(
            baseline_forecast_df["selected_bool"].astype(bool).sum()
        )
        score_ser = score_df.reindex(
            index=[decision_date_ts],
            columns=eligible_symbol_list,
        ).iloc[0]
        if score_ser.isna().any():
            missing_symbol_list = score_ser.index[score_ser.isna()].astype(str).tolist()
            raise RuntimeError(
                f"{variant_key_str} lacks a score on {decision_date_ts.date()}: "
                f"{missing_symbol_list[:5]}"
            )

        ranked_symbol_list = _stable_ranked_symbol_list(score_ser)
        if len(ranked_symbol_list) != len(eligible_symbol_list):
            raise RuntimeError(
                f"{variant_key_str} lost eligible names on "
                f"{decision_date_ts.date()}."
            )
        selected_symbol_list = ranked_symbol_list[-selected_count_int:]
        selected_symbol_set = set(selected_symbol_list)
        target_weight_float = 1.0 / float(selected_count_int)
        target_weight_row_dict[decision_date_ts] = pd.Series(
            target_weight_float,
            index=sorted(selected_symbol_list),
            dtype=float,
        )
        for symbol_str in eligible_symbol_list:
            forecast_record_list.append(
                {
                    "decision_date_ts": decision_date_ts,
                    "symbol_str": symbol_str,
                    "forecast_float": float(score_ser.loc[symbol_str]),
                    "quintile_int": np.nan,
                    "selected_bool": symbol_str in selected_symbol_set,
                    "target_weight_float": (
                        target_weight_float
                        if symbol_str in selected_symbol_set
                        else 0.0
                    ),
                    "variant_key_str": variant_key_str,
                }
            )
        coverage_record_list.append(
            {
                "decision_date_ts": decision_date_ts,
                "eligible_count_int": len(eligible_symbol_list),
                "selected_count_int": selected_count_int,
                "status_str": "valid_target",
                "variant_key_str": variant_key_str,
            }
        )

    target_weight_df = pd.DataFrame.from_dict(
        target_weight_row_dict,
        orient="index",
        dtype=float,
    ).sort_index()
    target_weight_df.index.name = "decision_date_ts"
    return AdaptiveMovingAverageFactorSignalBundle(
        target_weight_df=target_weight_df,
        forecast_df=pd.DataFrame(forecast_record_list),
        coefficient_df=pd.DataFrame(),
        coverage_df=pd.DataFrame(coverage_record_list),
    )


def build_buffered_signal_bundle(
    baseline_bundle_obj: AdaptiveMovingAverageFactorSignalBundle,
    retention_fraction_float: float = 0.30,
) -> AdaptiveMovingAverageFactorSignalBundle:
    if not 0.20 < retention_fraction_float < 1.0:
        raise ValueError("retention_fraction_float must be between 0.20 and 1.0.")

    target_weight_row_dict: dict[pd.Timestamp, pd.Series] = {}
    forecast_record_list: list[dict[str, object]] = []
    coverage_record_list: list[dict[str, object]] = []
    prior_target_symbol_set: set[str] = set()

    for decision_date_obj, baseline_forecast_df in (
        baseline_bundle_obj.forecast_df.groupby("decision_date_ts", sort=True)
    ):
        decision_date_ts = pd.Timestamp(decision_date_obj)
        score_ser = baseline_forecast_df.set_index("symbol_str")[
            "forecast_float"
        ].astype(float)
        ranked_symbol_list = _stable_ranked_symbol_list(score_ser)
        eligible_count_int = len(ranked_symbol_list)
        target_count_int = int(
            baseline_forecast_df["selected_bool"].astype(bool).sum()
        )
        retention_count_int = max(
            target_count_int,
            int(np.ceil(eligible_count_int * retention_fraction_float)),
        )
        retention_symbol_set = set(ranked_symbol_list[-retention_count_int:])
        rank_by_symbol_dict = {
            symbol_str: rank_int
            for rank_int, symbol_str in enumerate(ranked_symbol_list)
        }
        retained_symbol_list = sorted(
            prior_target_symbol_set.intersection(retention_symbol_set),
            key=lambda symbol_str: rank_by_symbol_dict[symbol_str],
            reverse=True,
        )[:target_count_int]
        selected_symbol_list = list(retained_symbol_list)
        if len(selected_symbol_list) < target_count_int:
            for symbol_str in reversed(ranked_symbol_list):
                if symbol_str in selected_symbol_list:
                    continue
                selected_symbol_list.append(symbol_str)
                if len(selected_symbol_list) == target_count_int:
                    break
        selected_symbol_set = set(selected_symbol_list)
        if len(selected_symbol_set) != target_count_int:
            raise RuntimeError(
                f"Buffered AMAF could not fill {target_count_int} slots on "
                f"{decision_date_ts.date()}."
            )

        target_weight_float = 1.0 / float(target_count_int)
        target_weight_row_dict[decision_date_ts] = pd.Series(
            target_weight_float,
            index=sorted(selected_symbol_set),
            dtype=float,
        )
        for symbol_str in score_ser.index.astype(str):
            forecast_record_list.append(
                {
                    "decision_date_ts": decision_date_ts,
                    "symbol_str": symbol_str,
                    "forecast_float": float(score_ser.loc[symbol_str]),
                    "quintile_int": np.nan,
                    "selected_bool": symbol_str in selected_symbol_set,
                    "target_weight_float": (
                        target_weight_float
                        if symbol_str in selected_symbol_set
                        else 0.0
                    ),
                    "variant_key_str": BUFFERED_VARIANT_KEY_STR,
                }
            )
        entry_count_int = len(selected_symbol_set - prior_target_symbol_set)
        exit_count_int = len(prior_target_symbol_set - selected_symbol_set)
        coverage_record_list.append(
            {
                "decision_date_ts": decision_date_ts,
                "eligible_count_int": eligible_count_int,
                "selected_count_int": target_count_int,
                "retained_count_int": len(
                    selected_symbol_set.intersection(prior_target_symbol_set)
                ),
                "entry_count_int": entry_count_int,
                "exit_count_int": exit_count_int,
                "status_str": "valid_target",
                "variant_key_str": BUFFERED_VARIANT_KEY_STR,
            }
        )
        prior_target_symbol_set = selected_symbol_set

    target_weight_df = pd.DataFrame.from_dict(
        target_weight_row_dict,
        orient="index",
        dtype=float,
    ).sort_index()
    target_weight_df.index.name = "decision_date_ts"
    return AdaptiveMovingAverageFactorSignalBundle(
        target_weight_df=target_weight_df,
        forecast_df=pd.DataFrame(forecast_record_list),
        coefficient_df=baseline_bundle_obj.coefficient_df.copy(),
        coverage_df=pd.DataFrame(coverage_record_list),
    )


def build_static_composite_score_df(
    price_close_df: pd.DataFrame,
    baseline_bundle_obj: AdaptiveMovingAverageFactorSignalBundle,
    sma_lookback_tuple: Sequence[int],
) -> pd.DataFrame:
    decision_date_index = pd.DatetimeIndex(
        baseline_bundle_obj.target_weight_df.index
    )
    feature_by_lookback_dict = build_monthly_sma_ratio_by_lookback_dict(
        price_close_df=price_close_df,
        decision_date_index=decision_date_index,
        sma_lookback_tuple=tuple(sma_lookback_tuple),
    )
    score_row_dict: dict[pd.Timestamp, pd.Series] = {}
    for decision_date_obj, baseline_forecast_df in (
        baseline_bundle_obj.forecast_df.groupby("decision_date_ts", sort=True)
    ):
        decision_date_ts = pd.Timestamp(decision_date_obj)
        eligible_symbol_list = (
            baseline_forecast_df["symbol_str"].astype(str).tolist()
        )
        feature_df = pd.DataFrame(
            {
                f"sma_{lookback_int}_ratio_float": (
                    feature_by_lookback_dict[int(lookback_int)]
                    .loc[decision_date_ts, eligible_symbol_list]
                    .astype(float)
                )
                for lookback_int in sma_lookback_tuple
            },
            index=eligible_symbol_list,
            dtype=float,
        )
        # *** CRITICAL *** Cross-sectional normalization is fit independently
        # at Close_T using only names eligible at that same Close_T. No future
        # month and no full-sample statistic enters the score.
        feature_mean_ser = feature_df.mean(axis=0)
        feature_std_ser = feature_df.std(axis=0, ddof=0)
        if feature_std_ser.le(0.0).any() or feature_std_ser.isna().any():
            raise RuntimeError(
                "Static AMAF encountered a constant feature on "
                f"{decision_date_ts.date()}."
            )
        standardized_feature_df = (
            feature_df - feature_mean_ser
        ).divide(feature_std_ser)
        score_row_dict[decision_date_ts] = -standardized_feature_df.mean(axis=1)

    score_df = pd.DataFrame.from_dict(
        score_row_dict,
        orient="index",
        dtype=float,
    ).sort_index()
    score_df.index.name = "decision_date_ts"
    return score_df


def build_classic_momentum_score_df(
    price_close_df: pd.DataFrame,
    decision_date_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    score_by_symbol_dict: dict[str, pd.Series] = {}
    for symbol_str in price_close_df.columns.astype(str):
        observed_close_ser = (
            pd.to_numeric(price_close_df[symbol_str], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
            .astype(float)
        )
        # *** CRITICAL *** Classic 12-1 momentum at Close_T skips the latest
        # 21 observed sessions and uses only closes from T-21 and T-252.
        momentum_ser = (
            observed_close_ser.shift(21)
            .divide(observed_close_ser.shift(252))
            .sub(1.0)
        )
        score_by_symbol_dict[symbol_str] = momentum_ser.reindex(
            decision_date_index
        )
    score_df = pd.DataFrame(
        score_by_symbol_dict,
        index=decision_date_index,
        dtype=float,
    )
    score_df.index.name = "decision_date_ts"
    return score_df


def build_equal_weight_control_bundle(
    baseline_bundle_obj: AdaptiveMovingAverageFactorSignalBundle,
) -> AdaptiveMovingAverageFactorSignalBundle:
    target_weight_row_dict: dict[pd.Timestamp, pd.Series] = {}
    forecast_record_list: list[dict[str, object]] = []
    coverage_record_list: list[dict[str, object]] = []
    for decision_date_obj, baseline_forecast_df in (
        baseline_bundle_obj.forecast_df.groupby("decision_date_ts", sort=True)
    ):
        decision_date_ts = pd.Timestamp(decision_date_obj)
        eligible_symbol_list = sorted(
            baseline_forecast_df["symbol_str"].astype(str).tolist()
        )
        target_weight_float = 1.0 / float(len(eligible_symbol_list))
        target_weight_row_dict[decision_date_ts] = pd.Series(
            target_weight_float,
            index=eligible_symbol_list,
            dtype=float,
        )
        for symbol_str in eligible_symbol_list:
            forecast_record_list.append(
                {
                    "decision_date_ts": decision_date_ts,
                    "symbol_str": symbol_str,
                    "forecast_float": 0.0,
                    "quintile_int": np.nan,
                    "selected_bool": True,
                    "target_weight_float": target_weight_float,
                    "variant_key_str": EQUAL_WEIGHT_CONTROL_KEY_STR,
                }
            )
        coverage_record_list.append(
            {
                "decision_date_ts": decision_date_ts,
                "eligible_count_int": len(eligible_symbol_list),
                "selected_count_int": len(eligible_symbol_list),
                "status_str": "valid_target",
                "variant_key_str": EQUAL_WEIGHT_CONTROL_KEY_STR,
            }
        )
    target_weight_df = pd.DataFrame.from_dict(
        target_weight_row_dict,
        orient="index",
        dtype=float,
    ).sort_index()
    target_weight_df.index.name = "decision_date_ts"
    return AdaptiveMovingAverageFactorSignalBundle(
        target_weight_df=target_weight_df,
        forecast_df=pd.DataFrame(forecast_record_list),
        coefficient_df=pd.DataFrame(),
        coverage_df=pd.DataFrame(coverage_record_list),
    )


class PrecomputedAmafResearchStrategy(AdaptiveMovingAverageFactorStrategy):
    """Run one causal precomputed monthly target path through Vanilla."""

    enable_signal_audit = False

    def __init__(
        self,
        name: str,
        benchmarks: Sequence[str],
        universe_df: pd.DataFrame,
        rebalance_schedule_df: pd.DataFrame,
        config_obj: AdaptiveMovingAverageFactorConfig,
        signal_bundle_obj: AdaptiveMovingAverageFactorSignalBundle,
    ):
        super().__init__(
            name=name,
            benchmarks=benchmarks,
            universe_df=universe_df,
            rebalance_schedule_df=rebalance_schedule_df,
            config_obj=config_obj,
        )
        self.precomputed_signal_bundle_obj = signal_bundle_obj
        self.missing_price_liquidation_count_int = 0
        self.research_transaction_record_list: list[dict[str, object]] = []
        self.research_transaction_cache_df: pd.DataFrame | None = None

    def compute_signals(self, pricing_data_df: pd.DataFrame) -> pd.DataFrame:
        last_available_date_ts = pd.Timestamp(pricing_data_df.index[-1])
        target_weight_df = self.precomputed_signal_bundle_obj.target_weight_df
        if (
            len(
                target_weight_df.loc[
                    target_weight_df.index <= last_available_date_ts
                ]
            )
            == 0
            and last_available_date_ts
            >= pd.Timestamp(self.config_obj.backtest_start_date_str)
        ):
            raise RuntimeError(
                f"{self.name} has no precomputed target by the backtest start."
            )
        self.signal_bundle_obj = self.precomputed_signal_bundle_obj
        return pricing_data_df

    def _target_weight_ser(self, close_row_ser: pd.Series) -> pd.Series:
        del close_row_ser
        # *** CRITICAL *** The lookup key is previous_bar, which the inherited
        # iterate() verifies equals the scheduled Close_T decision date before
        # placing orders that execute at current_bar Open_T+1.
        decision_date_ts = pd.Timestamp(self.previous_bar)
        target_weight_df = self.precomputed_signal_bundle_obj.target_weight_df
        if decision_date_ts not in target_weight_df.index:
            return pd.Series(dtype=float)
        target_weight_ser = pd.to_numeric(
            target_weight_df.loc[decision_date_ts],
            errors="coerce",
        ).dropna()
        target_weight_ser.index = target_weight_ser.index.astype(str)
        return target_weight_ser.astype(float).sort_index()

    def restrict_data(
        self,
        full_data_df: pd.DataFrame,
    ) -> tuple[pd.DataFrame | None, pd.Series | None, pd.Series]:
        del full_data_df
        if self.previous_bar is None:
            return None, None, pd.Series(dtype=float)
        # *** CRITICAL *** This research adapter's inherited iterate() reads no
        # price history or passed open vector. It looks up only the frozen target
        # keyed by previous_bar, while Vanilla process_orders() and valuation
        # still receive the untouched full pricing panel independently.
        return (
            pd.DataFrame(),
            pd.Series(dtype=float),
            pd.Series(dtype=float),
        )

    def add_transaction(
        self,
        trade_id,
        bar,
        asset,
        amount,
        price,
        total_value,
        order_id,
        commission=0.0,
    ) -> None:
        trade_id_obj = trade_id
        bar_ts = pd.Timestamp(bar)
        asset_str = str(asset)
        amount_float = float(amount)
        price_float = float(price)
        total_value_float = float(total_value)
        order_id_obj = order_id
        commission_float = float(commission)
        self.research_transaction_record_list.append(
            {
                "trade_id": trade_id_obj,
                "bar": bar_ts,
                "asset": asset_str,
                "amount": amount_float,
                "price": price_float,
                "total_value": total_value_float,
                "order_id": order_id_obj,
                "commission": commission_float,
            }
        )
        self.research_transaction_cache_df = None
        position_amount_float = float(
            self._position_amount_map.get(asset_str, 0.0)
        ) + float(amount_float)
        self._position_amount_map[asset_str] = position_amount_float
        self.log_audit_event(
            "engine.order.executed",
            {
                "asset_str": str(asset_str),
                "trade_id": trade_id_obj,
                "order_id_int": int(order_id_obj),
                "execution_bar_timestamp_str": pd.Timestamp(
                    bar_ts
                ).isoformat(),
                "amount_float": float(amount_float),
                "price_float": float(price_float),
                "total_value_float": float(total_value_float),
                "commission_float": float(commission_float),
                "position_after_float": float(position_amount_float),
            },
        )

    def get_transactions(self, bar=None) -> pd.DataFrame:
        if self.research_transaction_cache_df is None:
            self.research_transaction_cache_df = pd.DataFrame.from_records(
                self.research_transaction_record_list,
                columns=[
                    "trade_id",
                    "bar",
                    "asset",
                    "amount",
                    "price",
                    "total_value",
                    "order_id",
                    "commission",
                ],
            )
        if bar is None:
            return self.research_transaction_cache_df
        return self.research_transaction_cache_df.loc[
            self.research_transaction_cache_df["bar"].eq(bar)
        ]

    def _get_open_trade_amount_ser(
        self,
        asset_str: str,
    ) -> pd.Series:
        trade_amount_by_id_dict: dict[object, float] = {}
        for transaction_record_dict in self.research_transaction_record_list:
            if str(transaction_record_dict["asset"]) != asset_str:
                continue
            trade_id_obj = transaction_record_dict["trade_id"]
            trade_amount_by_id_dict[trade_id_obj] = (
                trade_amount_by_id_dict.get(trade_id_obj, 0.0)
                + float(transaction_record_dict["amount"])
            )
        open_trade_amount_ser = pd.Series(
            trade_amount_by_id_dict,
            dtype=float,
            name="open_trade_amount_ser",
        )
        if len(open_trade_amount_ser) == 0:
            return open_trade_amount_ser
        open_trade_mask_arr = ~np.isclose(
            open_trade_amount_ser.to_numpy(dtype=float),
            0.0,
            atol=1e-12,
        )
        return open_trade_amount_ser.loc[open_trade_mask_arr]

    def _credit_dividend_cash_before_open(
        self,
        pricing_data_df: pd.DataFrame,
    ) -> float:
        self._ensure_dividend_accounting_state()
        if not self._dividend_cash_ledger_active_bool(pricing_data_df):
            return 0.0
        position_or_order_asset_set = {
            str(asset_str)
            for asset_str, position_share_float in self.get_positions().items()
            if not np.isclose(float(position_share_float), 0.0)
        }
        position_or_order_asset_set.update(
            str(order_obj.asset) for order_obj in self.get_orders()
        )
        benchmark_data_symbol_set = {
            str(benchmark_str) for benchmark_str in self._benchmarks
        }
        benchmark_data_symbol_set.update(
            str(benchmark_data_symbol_str)
            for benchmark_data_symbol_str in (
                self._benchmark_data_symbol_map_dict.values()
            )
        )
        benchmark_trade_overlap_set = (
            position_or_order_asset_set & benchmark_data_symbol_set
        )
        if benchmark_trade_overlap_set:
            raise RuntimeError(
                "Dividend cash ledger cannot trade a declared benchmark: "
                f"{sorted(benchmark_trade_overlap_set)}"
            )
        adjustment_by_symbol_dict = dict(
            pricing_data_df.attrs.get(
                "norgate_adjustment_by_symbol_dict",
                {},
            )
        )
        if adjustment_by_symbol_dict and position_or_order_asset_set:
            missing_adjustment_asset_list = sorted(
                position_or_order_asset_set
                - {
                    str(asset_str)
                    for asset_str in adjustment_by_symbol_dict
                }
            )
            if missing_adjustment_asset_list:
                raise RuntimeError(
                    "Dividend cash ledger lacks adjustment provenance for "
                    f"{missing_adjustment_asset_list}."
                )
            invalid_adjustment_by_symbol_dict = {
                asset_str: str(adjustment_by_symbol_dict[asset_str])
                for asset_str in sorted(position_or_order_asset_set)
                if str(adjustment_by_symbol_dict[asset_str]).upper()
                != "CAPITALSPECIAL"
            }
            if invalid_adjustment_by_symbol_dict:
                raise RuntimeError(
                    "Dividend cash ledger requires CAPITALSPECIAL: "
                    f"{invalid_adjustment_by_symbol_dict}"
                )
            self._data_adjustment_policy_dict.update(
                {
                    "execution_and_marks_adjustment_str": "CAPITALSPECIAL",
                    "dividend_ledger_execution_basis_validation_str": (
                        "verified_from_norgate_source_metadata"
                    ),
                }
            )
        elif position_or_order_asset_set:
            self._data_adjustment_policy_dict.setdefault(
                "dividend_ledger_execution_basis_validation_str",
                "unverified_input_without_adjustment_metadata",
            )
        if self.current_bar is None or self.previous_bar is None:
            return 0.0

        ex_date_ts = pd.Timestamp(self.current_bar)
        entitlement_date_ts = pd.Timestamp(self.previous_bar)
        if ex_date_ts in self._dividend_processed_ex_date_set:
            return 0.0
        previous_bar_location_int = int(
            pricing_data_df.index.get_loc(entitlement_date_ts)
        )
        current_bar_location_int = int(
            pricing_data_df.index.get_loc(ex_date_ts)
        )
        if current_bar_location_int != previous_bar_location_int + 1:
            raise RuntimeError(
                "Dividend cash ledger requires consecutive sessions."
            )
        preopen_position_ser = self.get_positions().astype(float)
        active_position_ser = preopen_position_ser.loc[
            ~np.isclose(preopen_position_ser, 0.0)
        ]
        active_symbol_list = active_position_ser.index.astype(str).tolist()
        dividend_column_list = [
            (asset_str, "Dividend")
            for asset_str in active_symbol_list
        ]
        missing_dividend_column_list = [
            column_tuple
            for column_tuple in dividend_column_list
            if column_tuple not in pricing_data_df.columns
        ]
        if missing_dividend_column_list:
            raise RuntimeError(
                "Missing Dividend fields for active assets: "
                f"{missing_dividend_column_list}"
            )
        dividend_per_share_ser = pd.Series(dtype=float)
        if dividend_column_list:
            dividend_per_share_ser = pd.to_numeric(
                pricing_data_df.loc[
                    entitlement_date_ts,
                    dividend_column_list,
                ].droplevel(-1),
                errors="coerce",
            ).astype(float)
            dividend_per_share_ser = dividend_per_share_ser.reindex(
                active_symbol_list
            )
            if (
                dividend_per_share_ser.isna().any()
                or not np.isfinite(dividend_per_share_ser).all()
            ):
                raise RuntimeError(
                    "Invalid Dividend value on "
                    f"{entitlement_date_ts.date()}."
                )

        gross_dividend_cash_ser = (
            active_position_ser.reindex(active_symbol_list).astype(float)
            * dividend_per_share_ser
        )
        nonzero_dividend_mask_ser = ~np.isclose(
            gross_dividend_cash_ser,
            0.0,
        )
        gross_dividend_cash_ser = gross_dividend_cash_ser.loc[
            nonzero_dividend_mask_ser
        ]
        withholding_cash_ser = (
            gross_dividend_cash_ser.clip(lower=0.0)
            * float(self.dividend_withholding_rate_float)
        )
        net_dividend_cash_ser = (
            gross_dividend_cash_ser - withholding_cash_ser
        )
        pending_ledger_row_dict_list = [
            {
                "entitlement_date": entitlement_date_ts,
                "ex_date": ex_date_ts,
                "asset_str": str(asset_str),
                "position_share_float": float(
                    active_position_ser.loc[asset_str]
                ),
                "dividend_per_share_float": float(
                    dividend_per_share_ser.loc[asset_str]
                ),
                "gross_dividend_cash_float": float(
                    gross_dividend_cash_ser.loc[asset_str]
                ),
                "withholding_cash_float": float(
                    withholding_cash_ser.loc[asset_str]
                ),
                "net_dividend_cash_float": float(
                    net_dividend_cash_ser.loc[asset_str]
                ),
            }
            for asset_str in gross_dividend_cash_ser.index.astype(str)
        ]
        gross_dividend_cash_sum_float = float(
            gross_dividend_cash_ser.sum()
        )
        withholding_cash_sum_float = float(withholding_cash_ser.sum())
        net_dividend_cash_sum_float = float(net_dividend_cash_ser.sum())
        self.cash += net_dividend_cash_sum_float
        self.dividend_cash_gross_total_float += (
            gross_dividend_cash_sum_float
        )
        self.dividend_withholding_total_float += withholding_cash_sum_float
        self.dividend_cash_net_total_float += net_dividend_cash_sum_float
        self._dividend_ledger_row_dict_list.extend(
            pending_ledger_row_dict_list
        )
        self._dividend_processed_ex_date_set.add(ex_date_ts)
        self._accounting_policy_dict.update(
            {
                "dividend_cash_gross_total_float": float(
                    self.dividend_cash_gross_total_float
                ),
                "dividend_withholding_total_float": float(
                    self.dividend_withholding_total_float
                ),
                "dividend_cash_net_total_float": float(
                    self.dividend_cash_net_total_float
                ),
                "dividend_event_count_int": int(
                    len(self._dividend_ledger_row_dict_list)
                ),
            }
        )
        if pending_ledger_row_dict_list:
            self.log_audit_event(
                "engine.dividend_cash.posted",
                {
                    "entitlement_date_timestamp_str": (
                        entitlement_date_ts.isoformat()
                    ),
                    "ex_date_timestamp_str": ex_date_ts.isoformat(),
                    "event_count_int": int(
                        len(pending_ledger_row_dict_list)
                    ),
                    "gross_dividend_cash_float": (
                        gross_dividend_cash_sum_float
                    ),
                    "withholding_cash_float": withholding_cash_sum_float,
                    "net_dividend_cash_float": net_dividend_cash_sum_float,
                },
            )
        return net_dividend_cash_sum_float

    def _liquidate_missing_price_positions(
        self,
        prices: pd.DataFrame,
    ) -> tuple[float, float]:
        transaction_value_sum_float = 0.0
        commission_sum_float = 0.0
        position_amount_ser = self.get_positions()
        active_position_ser = position_amount_ser[position_amount_ser != 0]
        if len(active_position_ser) == 0:
            return transaction_value_sum_float, commission_sum_float

        active_symbol_list = active_position_ser.index.astype(str).tolist()
        required_column_list = [
            (symbol_str, field_str)
            for symbol_str in active_symbol_list
            for field_str in ("Open", "Close")
        ]
        missing_column_list = [
            column_tuple
            for column_tuple in required_column_list
            if column_tuple not in prices.columns
        ]
        if missing_column_list:
            raise RuntimeError(
                f"Active assets lack price fields: {missing_column_list}"
            )
        current_active_price_df = (
            prices.loc[self.current_bar, required_column_list]
            .unstack()
            .reindex(active_symbol_list)
        )
        finite_price_mask_ser = (
            np.isfinite(
                pd.to_numeric(
                    current_active_price_df["Open"],
                    errors="coerce",
                )
            )
            & np.isfinite(
                pd.to_numeric(
                    current_active_price_df["Close"],
                    errors="coerce",
                )
            )
        )
        missing_price_symbol_list = finite_price_mask_ser.index[
            ~finite_price_mask_ser
        ].astype(str).tolist()
        for asset_str in missing_price_symbol_list:
            liquidation_bar_ts, liquidation_price_float = (
                self._get_last_available_close_before_current_bar(
                    prices=prices,
                    asset_str=asset_str,
                )
            )
            open_trade_amount_ser = self._get_open_trade_amount_ser(
                asset_str=asset_str
            )
            if len(open_trade_amount_ser) == 0:
                raise RuntimeError(
                    f"Found a live position in {asset_str} without open trades."
                )
            self.log_audit_event(
                "engine.missing_price_position_liquidated",
                {
                    "asset_str": asset_str,
                    "liquidation_bar_timestamp_str": (
                        liquidation_bar_ts.isoformat()
                    ),
                    "liquidation_price_float": float(
                        liquidation_price_float
                    ),
                    "open_trade_count_int": int(len(open_trade_amount_ser)),
                },
            )
            self.clear_orders(asset=asset_str)
            for trade_id_obj, open_amount_float in (
                open_trade_amount_ser.items()
            ):
                liquidation_amount_float = -float(open_amount_float)
                commission_float = float(
                    self._compute_commission(liquidation_amount_float)
                )
                liquidation_value_float = (
                    liquidation_amount_float * liquidation_price_float
                )
                self.add_transaction(
                    trade_id_obj,
                    self.current_bar,
                    asset_str,
                    liquidation_amount_float,
                    liquidation_price_float,
                    liquidation_value_float,
                    order_id=-1,
                    commission=commission_float,
                )
                transaction_value_sum_float += liquidation_value_float
                commission_sum_float += commission_float
        return transaction_value_sum_float, commission_sum_float

    def process_orders(self, prices: pd.DataFrame) -> None:
        if self.current_bar not in prices.index:
            return
        self._credit_dividend_cash_before_open(prices)

        total_value_sum_float = 0.0
        commission_sum_float = 0.0
        portfolio_value_float = float(self.previous_total_value)
        (
            stale_transaction_value_float,
            stale_commission_float,
        ) = self._liquidate_missing_price_positions(prices=prices)
        total_value_sum_float += stale_transaction_value_float
        commission_sum_float += stale_commission_float

        for order_obj in list(self.get_orders()):
            if not isinstance(order_obj, MarketOrder):
                raise RuntimeError(
                    "AMAF research adapter accepts market target orders only."
                )
            asset_str = str(order_obj.asset)
            current_open_key_tuple = (asset_str, "Open")
            if current_open_key_tuple not in prices.columns:
                raise RuntimeError(f"{asset_str} not in available prices.")
            current_open_float = float(
                prices.at[self.current_bar, current_open_key_tuple]
            )
            position_float = float(self.get_position(asset_str))
            previous_close_key_tuple = (asset_str, "Close")
            previous_close_float = (
                float(prices.at[self.previous_bar, previous_close_key_tuple])
                if (
                    self.previous_bar is not None
                    and previous_close_key_tuple in prices.columns
                )
                else np.nan
            )
            # *** CRITICAL *** Match Vanilla's causal target-percent sizing:
            # previous Close_T fixes shares; Open_T+1 is used only for the fill.
            sizing_price_float = (
                previous_close_float
                if (
                    np.isfinite(previous_close_float)
                    and previous_close_float > 0.0
                )
                else current_open_float
            )
            if not np.isfinite(current_open_float):
                if not np.isclose(position_float, 0.0):
                    raise RuntimeError(
                        f"Asset {asset_str} still has an open position after "
                        "missing-price liquidation."
                    )
                self.log_audit_event(
                    "engine.order.canceled",
                    self._build_order_log_payload_dict(
                        order_obj,
                        {"reason_code_str": "missing_current_open"},
                    ),
                )
                self.remove_order(order_obj)
                continue
            amount_float = float(
                order_obj.amount_in_shares(
                    sizing_price_float,
                    portfolio_value_float,
                    position_float,
                )
            )
            if self._cancel_zero_share_fill_bool(order_obj, amount_float):
                continue
            penalty_float = (
                1.0 + np.sign(amount_float) * float(self._slippage)
            )
            execution_price_float = current_open_float * penalty_float
            commission_float = float(
                self._compute_commission(amount_float)
            )
            transaction_value_float = (
                execution_price_float * amount_float
            )
            self.add_transaction(
                order_obj.trade_id,
                self.current_bar,
                asset_str,
                amount_float,
                execution_price_float,
                transaction_value_float,
                order_obj.id,
                commission_float,
            )
            total_value_sum_float += transaction_value_float
            commission_sum_float += commission_float
            self.remove_order(order_obj)

        self.cash -= total_value_sum_float
        self.cash -= commission_sum_float
        position_ser = self.get_positions()
        active_position_ser = position_ser[position_ser != 0]
        if len(active_position_ser) == 0:
            self.portfolio_value = 0.0
            self._latest_close_price_ser = pd.Series(dtype=float)
        else:
            active_close_ser = pd.Series(
                {
                    str(asset_obj): float(
                        prices.at[
                            self.current_bar,
                            (str(asset_obj), "Close"),
                        ]
                    )
                    for asset_obj in active_position_ser.index
                },
                dtype=float,
            )
            if active_close_ser.isna().any():
                missing_asset_list = (
                    active_close_ser.loc[active_close_ser.isna()]
                    .index.astype(str)
                    .tolist()
                )
                raise RuntimeError(
                    "Active positions still contain missing close prices after "
                    f"missing-price liquidation: {missing_asset_list}"
                )
            self.portfolio_value = float(
                (active_position_ser * active_close_ser).sum()
            )
            self._latest_close_price_ser = active_close_ser
        self.total_value = self.cash + self.portfolio_value

    def _record_realized_weight_snapshot(
        self,
        price_df: pd.DataFrame,
    ) -> None:
        if self.current_bar is None:
            return
        total_value_float = float(self.total_value)
        if (
            not np.isfinite(total_value_float)
            or np.isclose(total_value_float, 0.0, atol=1e-12)
        ):
            return
        current_date_ts = pd.Timestamp(self.current_bar).normalize()
        realized_weight_ser = pd.Series(dtype=float, name=current_date_ts)
        position_share_ser = self.get_positions()
        active_position_share_ser = position_share_ser[
            position_share_ser != 0
        ]
        if len(active_position_share_ser) > 0:
            # *** CRITICAL *** Close_T marks are post-valuation reporting only
            # and cannot feed the target path or the next-open execution.
            active_close_price_ser = pd.Series(
                {
                    str(asset_obj): float(
                        price_df.at[
                            self.current_bar,
                            (str(asset_obj), "Close"),
                        ]
                    )
                    for asset_obj in active_position_share_ser.index
                },
                dtype=float,
            )
            if active_close_price_ser.isna().any():
                missing_asset_list = (
                    active_close_price_ser.loc[
                        active_close_price_ser.isna()
                    ]
                    .index.astype(str)
                    .tolist()
                )
                raise RuntimeError(
                    f"Cannot compute realized weights on "
                    f"{current_date_ts.date()}; missing close prices for "
                    f"{missing_asset_list}."
                )
            position_value_ser = (
                active_position_share_ser.astype(float)
                * active_close_price_ser
            )
            realized_weight_ser = position_value_ser / total_value_float
        realized_weight_ser.loc["Cash"] = (
            float(self.cash) / total_value_float
        )
        realized_weight_row_dict = {
            str(asset_obj): float(realized_weight_float)
            for asset_obj, realized_weight_float in realized_weight_ser.items()
        }
        realized_weight_row_dict["snapshot_date_ts"] = current_date_ts
        self._realized_weight_snapshot_row_dict_list.append(
            realized_weight_row_dict
        )

    def log_audit_event(
        self,
        event_type_str: str,
        payload_dict: dict[str, object] | None = None,
    ) -> None:
        if event_type_str == "engine.missing_price_position_liquidated":
            self.missing_price_liquidation_count_int += 1
        super().log_audit_event(event_type_str, payload_dict)


def _summary_metric_float(
    strategy_obj: PrecomputedAmafResearchStrategy,
    metric_name_str: str,
) -> float:
    if strategy_obj.summary is None or metric_name_str not in strategy_obj.summary.index:
        return float("nan")
    value_obj = strategy_obj.summary.loc[metric_name_str, "Strategy"]
    return float(value_obj) if pd.notna(value_obj) else float("nan")


def _performance_metric_dict(
    daily_return_ser: pd.Series,
) -> dict[str, float | int]:
    clean_return_ser = (
        pd.to_numeric(daily_return_ser, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .astype(float)
    )
    observation_count_int = len(clean_return_ser)
    if observation_count_int < 2:
        return {
            "observation_count_int": observation_count_int,
            "annual_return_pct_float": np.nan,
            "annual_volatility_pct_float": np.nan,
            "sharpe_float": np.nan,
            "max_drawdown_pct_float": np.nan,
            "mar_float": np.nan,
        }
    growth_float = float((1.0 + clean_return_ser).prod())
    annual_return_float = growth_float ** (252.0 / observation_count_int) - 1.0
    annual_volatility_float = float(
        clean_return_ser.std(ddof=1) * np.sqrt(252.0)
    )
    sharpe_float = (
        float(clean_return_ser.mean() / clean_return_ser.std(ddof=1) * np.sqrt(252.0))
        if clean_return_ser.std(ddof=1) > 0.0
        else np.nan
    )
    equity_ser = (1.0 + clean_return_ser).cumprod()
    max_drawdown_float = float((equity_ser / equity_ser.cummax() - 1.0).min())
    mar_float = (
        annual_return_float / abs(max_drawdown_float)
        if max_drawdown_float < 0.0
        else np.nan
    )
    return {
        "observation_count_int": observation_count_int,
        "annual_return_pct_float": annual_return_float * 100.0,
        "annual_volatility_pct_float": annual_volatility_float * 100.0,
        "sharpe_float": sharpe_float,
        "max_drawdown_pct_float": max_drawdown_float * 100.0,
        "mar_float": mar_float,
    }


def build_cost_stress_daily_return(
    strategy_obj: PrecomputedAmafResearchStrategy,
    target_slippage_per_side_float: float,
) -> tuple[pd.Series, pd.Series]:
    """Apply incremental slippage to the realized platform transaction path."""

    platform_slippage_per_side_float = float(strategy_obj._slippage)
    if target_slippage_per_side_float < platform_slippage_per_side_float:
        raise ValueError(
            "target_slippage_per_side_float cannot be below the platform run."
        )
    platform_daily_return_ser = (
        strategy_obj.results["daily_returns"].astype(float).copy()
    )
    incremental_cost_fraction_ser = pd.Series(
        0.0,
        index=platform_daily_return_ser.index,
        dtype=float,
    )
    incremental_slippage_float = (
        float(target_slippage_per_side_float)
        - platform_slippage_per_side_float
    )
    if incremental_slippage_float == 0.0:
        return platform_daily_return_ser, incremental_cost_fraction_ser

    transaction_df = strategy_obj.get_transactions().copy()
    if len(transaction_df) == 0:
        return platform_daily_return_ser, incremental_cost_fraction_ser
    required_column_set = {"bar", "amount", "total_value"}
    missing_column_set = required_column_set.difference(transaction_df.columns)
    if missing_column_set:
        raise RuntimeError(
            "Transactions lack cost-stress fields: "
            f"{sorted(missing_column_set)}"
        )

    amount_ser = pd.to_numeric(transaction_df["amount"], errors="raise").astype(
        float
    )
    execution_notional_ser = pd.to_numeric(
        transaction_df["total_value"],
        errors="raise",
    ).abs().astype(float)
    execution_denominator_ser = (
        1.0
        + np.sign(amount_ser) * platform_slippage_per_side_float
    )
    reference_notional_ser = (
        execution_notional_ser / execution_denominator_ser
    )
    transaction_date_index = pd.DatetimeIndex(
        pd.to_datetime(transaction_df["bar"])
    )
    incremental_cost_by_date_ser = pd.Series(
        reference_notional_ser.to_numpy(dtype=float)
        * incremental_slippage_float,
        index=transaction_date_index,
        dtype=float,
    ).groupby(level=0).sum()

    total_value_ser = strategy_obj.results["total_value"].astype(float)
    # *** CRITICAL *** This is report-only. Equity immediately before each
    # day's realized return is recovered from that same day's ending equity and
    # return. It changes no signal, order, fill, or future transaction notional.
    starting_equity_ser = total_value_ser.divide(
        1.0 + platform_daily_return_ser
    )
    aligned_starting_equity_ser = starting_equity_ser.reindex(
        incremental_cost_by_date_ser.index
    )
    if (
        aligned_starting_equity_ser.isna().any()
        or aligned_starting_equity_ser.le(0.0).any()
    ):
        raise RuntimeError(
            "Cost stress could not align positive starting equity to trades."
        )
    incremental_cost_fraction_ser.loc[
        incremental_cost_by_date_ser.index
    ] = (
        incremental_cost_by_date_ser
        / aligned_starting_equity_ser
    ).to_numpy(dtype=float)
    stressed_daily_return_ser = (
        platform_daily_return_ser - incremental_cost_fraction_ser
    )
    if stressed_daily_return_ser.le(-1.0).any():
        raise RuntimeError("Cost stress produced a daily return at or below -100%.")
    return stressed_daily_return_ser, incremental_cost_fraction_ser


def _paired_monthly_hac_dict(
    candidate_monthly_return_ser: pd.Series,
    baseline_monthly_return_ser: pd.Series,
) -> dict[str, float | int]:
    # *** CRITICAL *** Report-only inner alignment uses no fill. These realized
    # paired returns cannot feed any signal, selection, or execution decision.
    paired_return_df = pd.concat(
        [
            candidate_monthly_return_ser.rename("candidate_return_float"),
            baseline_monthly_return_ser.rename("baseline_return_float"),
        ],
        axis=1,
        join="inner",
    ).replace([np.inf, -np.inf], np.nan).dropna()
    return_delta_ser = (
        paired_return_df["candidate_return_float"]
        - paired_return_df["baseline_return_float"]
    )
    observation_count_int = len(return_delta_ser)
    result_dict: dict[str, float | int] = {
        "paired_month_count_int": observation_count_int,
        "mean_return_delta_annual_pct_float": np.nan,
        "hac_lag_int": 0,
        "hac_t_float": np.nan,
        "hac_p_float": np.nan,
    }
    if observation_count_int < 24:
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
            "mean_return_delta_annual_pct_float": (
                float(return_delta_ser.mean()) * 12.0 * 100.0
            ),
            "hac_lag_int": hac_lag_int,
            "hac_t_float": float(regression_result_obj.tvalues[0]),
            "hac_p_float": float(regression_result_obj.pvalues[0]),
        }
    )
    return result_dict


def _comparison_row_dict(
    strategy_obj: PrecomputedAmafResearchStrategy,
    universe_key_str: str,
    variant_key_str: str,
    cost_tier_key_str: str,
    signal_bundle_obj: AdaptiveMovingAverageFactorSignalBundle,
    daily_return_ser: pd.Series,
    incremental_cost_fraction_ser: pd.Series,
) -> dict[str, object]:
    result_df = strategy_obj.results
    average_target_position_count_float = float(
        signal_bundle_obj.target_weight_df.notna().sum(axis=1).mean()
    )
    realized_weight_df = strategy_obj.realized_weight_df.drop(
        columns=["Cash"],
        errors="ignore",
    )
    average_realized_position_count_float = float(
        realized_weight_df.fillna(0.0).abs().gt(1e-12).sum(axis=1).mean()
    )
    average_cash_weight_pct_float = float(
        strategy_obj.realized_weight_df.get(
            "Cash",
            pd.Series(dtype=float),
        ).mean()
        * 100.0
    )
    performance_metric_dict = _performance_metric_dict(
        daily_return_ser=daily_return_ser
    )
    platform_cost_drag_float = _summary_metric_float(
        strategy_obj,
        "Cost Drag (Ann.) [%]",
    )
    incremental_cost_drag_float = float(
        incremental_cost_fraction_ser.mean() * 252.0 * 100.0
    )
    return {
        "universe_key_str": universe_key_str,
        "variant_key_str": variant_key_str,
        "candidate_bool": variant_key_str in CANDIDATE_VARIANT_KEY_TUPLE,
        "cost_tier_key_str": cost_tier_key_str,
        "start_date_str": pd.Timestamp(result_df.index[0]).date().isoformat(),
        "end_date_str": pd.Timestamp(result_df.index[-1]).date().isoformat(),
        **performance_metric_dict,
        "turnover_ann_pct_float": _summary_metric_float(
            strategy_obj, "Turnover (Ann.) [%]"
        ),
        "cost_drag_ann_pct_float": (
            platform_cost_drag_float + incremental_cost_drag_float
        ),
        "exposure_time_pct_float": _summary_metric_float(
            strategy_obj, "Exposure Time [%]"
        ),
        "average_target_position_count_float": (
            average_target_position_count_float
        ),
        "average_realized_position_count_float": (
            average_realized_position_count_float
        ),
        "average_cash_weight_pct_float": average_cash_weight_pct_float,
        "missing_price_liquidation_count_int": (
            strategy_obj.missing_price_liquidation_count_int
        ),
        "slippage_per_side_float": float(
            COST_TIER_SLIPPAGE_BY_KEY_DICT[cost_tier_key_str]
        ),
        "commission_per_share_float": float(
            strategy_obj.config_obj.commission_per_share_float
        ),
        "commission_minimum_float": float(
            strategy_obj.config_obj.commission_minimum_float
        ),
    }


def _build_inference_df(
    monthly_return_df: pd.DataFrame,
) -> pd.DataFrame:
    inference_row_list: list[dict[str, object]] = []
    for universe_key_str in UNIVERSE_CONFIG_BY_KEY_DICT:
        for cost_tier_key_str in COST_TIER_SLIPPAGE_BY_KEY_DICT:
            baseline_mask_ser = (
                monthly_return_df["universe_key_str"].eq(universe_key_str)
                & monthly_return_df["cost_tier_key_str"].eq(cost_tier_key_str)
                & monthly_return_df["variant_key_str"].eq(
                    BASELINE_VARIANT_KEY_STR
                )
            )
            baseline_monthly_return_ser = monthly_return_df.loc[
                baseline_mask_ser
            ].set_index("month_end_ts")["monthly_return_float"]
            for candidate_variant_key_str in CANDIDATE_VARIANT_KEY_TUPLE:
                candidate_mask_ser = (
                    monthly_return_df["universe_key_str"].eq(universe_key_str)
                    & monthly_return_df["cost_tier_key_str"].eq(
                        cost_tier_key_str
                    )
                    & monthly_return_df["variant_key_str"].eq(
                        candidate_variant_key_str
                    )
                )
                candidate_monthly_return_ser = monthly_return_df.loc[
                    candidate_mask_ser
                ].set_index("month_end_ts")["monthly_return_float"]
                hac_metric_dict = _paired_monthly_hac_dict(
                    candidate_monthly_return_ser=candidate_monthly_return_ser,
                    baseline_monthly_return_ser=baseline_monthly_return_ser,
                )
                raw_p_float = float(hac_metric_dict["hac_p_float"])
                inference_row_list.append(
                    {
                        "universe_key_str": universe_key_str,
                        "candidate_variant_key_str": candidate_variant_key_str,
                        "cost_tier_key_str": cost_tier_key_str,
                        **hac_metric_dict,
                        "bonferroni_p_float": (
                            min(
                                raw_p_float
                                * float(DECLARED_PRIMARY_HYPOTHESIS_COUNT_INT),
                                1.0,
                            )
                            if np.isfinite(raw_p_float)
                            else np.nan
                        ),
                    }
                )
    return pd.DataFrame(inference_row_list)


def _comparison_value_float(
    comparison_df: pd.DataFrame,
    universe_key_str: str,
    variant_key_str: str,
    cost_tier_key_str: str,
    column_name_str: str,
) -> float:
    row_df = comparison_df.loc[
        comparison_df["universe_key_str"].eq(universe_key_str)
        & comparison_df["variant_key_str"].eq(variant_key_str)
        & comparison_df["cost_tier_key_str"].eq(cost_tier_key_str)
    ]
    if len(row_df) != 1:
        raise RuntimeError(
            "Expected one comparison row for "
            f"{universe_key_str}/{variant_key_str}/{cost_tier_key_str}."
        )
    return float(row_df.iloc[0][column_name_str])


def _inference_delta_float(
    inference_df: pd.DataFrame,
    universe_key_str: str,
    candidate_variant_key_str: str,
    cost_tier_key_str: str,
) -> float:
    row_df = inference_df.loc[
        inference_df["universe_key_str"].eq(universe_key_str)
        & inference_df["candidate_variant_key_str"].eq(
            candidate_variant_key_str
        )
        & inference_df["cost_tier_key_str"].eq(cost_tier_key_str)
    ]
    if len(row_df) != 1:
        raise RuntimeError(
            "Expected one inference row for "
            f"{universe_key_str}/{candidate_variant_key_str}/"
            f"{cost_tier_key_str}."
        )
    return float(row_df.iloc[0]["mean_return_delta_annual_pct_float"])


def build_promotion_gate_df(
    comparison_df: pd.DataFrame,
    inference_df: pd.DataFrame,
) -> pd.DataFrame:
    gate_row_list: list[dict[str, object]] = []
    for universe_key_str in UNIVERSE_CONFIG_BY_KEY_DICT:
        baseline_return_float = _comparison_value_float(
            comparison_df,
            universe_key_str,
            BASELINE_VARIANT_KEY_STR,
            "platform",
            "annual_return_pct_float",
        )
        baseline_sharpe_float = _comparison_value_float(
            comparison_df,
            universe_key_str,
            BASELINE_VARIANT_KEY_STR,
            "platform",
            "sharpe_float",
        )
        baseline_drawdown_float = _comparison_value_float(
            comparison_df,
            universe_key_str,
            BASELINE_VARIANT_KEY_STR,
            "platform",
            "max_drawdown_pct_float",
        )
        baseline_turnover_float = _comparison_value_float(
            comparison_df,
            universe_key_str,
            BASELINE_VARIANT_KEY_STR,
            "platform",
            "turnover_ann_pct_float",
        )
        baseline_missing_float = _comparison_value_float(
            comparison_df,
            universe_key_str,
            BASELINE_VARIANT_KEY_STR,
            "platform",
            "missing_price_liquidation_count_int",
        )

        buffered_return_float = _comparison_value_float(
            comparison_df,
            universe_key_str,
            BUFFERED_VARIANT_KEY_STR,
            "platform",
            "annual_return_pct_float",
        )
        buffered_sharpe_float = _comparison_value_float(
            comparison_df,
            universe_key_str,
            BUFFERED_VARIANT_KEY_STR,
            "platform",
            "sharpe_float",
        )
        buffered_drawdown_float = _comparison_value_float(
            comparison_df,
            universe_key_str,
            BUFFERED_VARIANT_KEY_STR,
            "platform",
            "max_drawdown_pct_float",
        )
        buffered_turnover_float = _comparison_value_float(
            comparison_df,
            universe_key_str,
            BUFFERED_VARIANT_KEY_STR,
            "platform",
            "turnover_ann_pct_float",
        )
        buffered_missing_float = _comparison_value_float(
            comparison_df,
            universe_key_str,
            BUFFERED_VARIANT_KEY_STR,
            "platform",
            "missing_price_liquidation_count_int",
        )
        turnover_reduction_float = (
            (baseline_turnover_float - buffered_turnover_float)
            / baseline_turnover_float
        )
        cagr_retention_float = (
            buffered_return_float / baseline_return_float
            if baseline_return_float > 0.0
            else np.nan
        )
        buffered_gate_dict = {
            "turnover_reduction_pass_bool": turnover_reduction_float >= 0.25,
            "cagr_retention_pass_bool": cagr_retention_float >= 0.95,
            "sharpe_delta_pass_bool": (
                buffered_sharpe_float - baseline_sharpe_float >= -0.05
            ),
            "drawdown_pass_bool": (
                buffered_drawdown_float >= baseline_drawdown_float - 2.0
            ),
            "approx_round_trip_20bps_pass_bool": (
                _inference_delta_float(
                    inference_df,
                    universe_key_str,
                    BUFFERED_VARIANT_KEY_STR,
                    "round_trip_20bps",
                )
                > 0.0
            ),
            "approx_round_trip_50bps_pass_bool": (
                _inference_delta_float(
                    inference_df,
                    universe_key_str,
                    BUFFERED_VARIANT_KEY_STR,
                    "round_trip_50bps",
                )
                > 0.0
            ),
            "missing_liquidation_pass_bool": (
                buffered_missing_float <= baseline_missing_float
            ),
        }
        gate_row_list.append(
            {
                "universe_key_str": universe_key_str,
                "candidate_variant_key_str": BUFFERED_VARIANT_KEY_STR,
                "turnover_reduction_fraction_float": turnover_reduction_float,
                "cagr_retention_fraction_float": cagr_retention_float,
                "sharpe_delta_float": (
                    buffered_sharpe_float - baseline_sharpe_float
                ),
                "drawdown_delta_pct_float": (
                    buffered_drawdown_float - baseline_drawdown_float
                ),
                **buffered_gate_dict,
                "historical_screen_pass_bool": all(
                    buffered_gate_dict.values()
                ),
                "mechanical_gate_pass_bool": False,
                "mechanical_gate_block_reason_str": (
                    "20/50 bps rows are approximate; exact engine reruns and "
                    "forward evidence are required"
                ),
                "research_status_str": (
                    "post_hoc_approximate_cost_screen_only"
                ),
            }
        )

        static_gate_dict: dict[str, bool] = {}
        for cost_tier_key_str in COST_TIER_SLIPPAGE_BY_KEY_DICT:
            static_return_float = _comparison_value_float(
                comparison_df,
                universe_key_str,
                STATIC_VARIANT_KEY_STR,
                cost_tier_key_str,
                "annual_return_pct_float",
            )
            cost_baseline_return_float = _comparison_value_float(
                comparison_df,
                universe_key_str,
                BASELINE_VARIANT_KEY_STR,
                cost_tier_key_str,
                "annual_return_pct_float",
            )
            equal_weight_return_float = _comparison_value_float(
                comparison_df,
                universe_key_str,
                EQUAL_WEIGHT_CONTROL_KEY_STR,
                cost_tier_key_str,
                "annual_return_pct_float",
            )
            static_gate_dict[
                f"{cost_tier_key_str}_return_pass_bool"
            ] = (
                static_return_float > cost_baseline_return_float
                and static_return_float > equal_weight_return_float
            )
        static_sharpe_float = _comparison_value_float(
            comparison_df,
            universe_key_str,
            STATIC_VARIANT_KEY_STR,
            "platform",
            "sharpe_float",
        )
        static_missing_float = _comparison_value_float(
            comparison_df,
            universe_key_str,
            STATIC_VARIANT_KEY_STR,
            "platform",
            "missing_price_liquidation_count_int",
        )
        static_gate_dict["sharpe_pass_bool"] = (
            static_sharpe_float > baseline_sharpe_float
        )
        static_gate_dict["missing_liquidation_pass_bool"] = (
            static_missing_float <= baseline_missing_float
        )
        gate_row_list.append(
            {
                "universe_key_str": universe_key_str,
                "candidate_variant_key_str": STATIC_VARIANT_KEY_STR,
                **static_gate_dict,
                "historical_screen_pass_bool": all(
                    static_gate_dict.values()
                ),
                "mechanical_gate_pass_bool": False,
                "mechanical_gate_block_reason_str": (
                    "20/50 bps rows are approximate; exact engine reruns and "
                    "forward evidence are required"
                ),
                "research_status_str": (
                    "post_hoc_approximate_cost_screen_only"
                ),
            }
        )
    return pd.DataFrame(gate_row_list)


def _markdown_table_str(table_df: pd.DataFrame) -> str:
    if len(table_df) == 0:
        return "_No rows._\n"
    column_list = table_df.columns.astype(str).tolist()
    line_list = [
        "| " + " | ".join(column_list) + " |",
        "| " + " | ".join(["---"] * len(column_list)) + " |",
    ]
    for _, row_ser in table_df.iterrows():
        value_str_list: list[str] = []
        for value_obj in row_ser.tolist():
            if pd.isna(value_obj):
                value_str_list.append("")
            elif isinstance(value_obj, float):
                value_str_list.append(f"{value_obj:.4f}")
            else:
                value_str_list.append(str(value_obj))
        line_list.append("| " + " | ".join(value_str_list) + " |")
    return "\n".join(line_list) + "\n"


def _write_report(
    run_output_path: Path,
    comparison_df: pd.DataFrame,
    promotion_gate_df: pd.DataFrame,
) -> None:
    platform_df = comparison_df.loc[
        comparison_df["cost_tier_key_str"].eq("platform"),
        [
            "universe_key_str",
            "variant_key_str",
            "annual_return_pct_float",
            "annual_volatility_pct_float",
            "sharpe_float",
            "max_drawdown_pct_float",
            "mar_float",
            "turnover_ann_pct_float",
            "cost_drag_ann_pct_float",
            "average_target_position_count_float",
            "average_realized_position_count_float",
            "average_cash_weight_pct_float",
            "missing_price_liquidation_count_int",
        ],
    ].copy()
    passing_screen_candidate_list = sorted(
        promotion_gate_df.groupby("candidate_variant_key_str")[
            "historical_screen_pass_bool"
        ]
        .all()
        .loc[lambda pass_ser: pass_ser]
        .index.astype(str)
        .tolist()
    )
    if passing_screen_candidate_list:
        verdict_str = (
            "Historical approximate-cost screens passed: "
            + ", ".join(passing_screen_candidate_list)
            + ". This is not a mechanical promotion pass: exact cost-tier "
            "engine reruns and frozen forward evidence are still required."
        )
    else:
        verdict_str = (
            "No candidate passed the frozen historical screen in both universes. "
            "Keep the original AMAF unchanged."
        )
    report_md_str = f"""# AMAF Improvement Sweep

## TL;DR

{verdict_str}

## Frozen design

- Candidate search count: exactly {DECLARED_CANDIDATE_COUNT_INT}.
- Universes: point-in-time Russell 1000 and Nasdaq-100 domestic-panel intersection.
- Decision: final common market `Close_T`.
- Execution: first tradable `Open_T+1`.
- Research capital: `${RESEARCH_CAPITAL_BASE_FLOAT:,.0f}` to limit whole-share
  distortion in the broad equal-weight control.
- Central costs: unchanged platform 2.5 bps per side plus `$0.005/share`,
  `$1` minimum.
- Stress costs: approximate 20 bps and 50 bps round trip using the realized
  platform transaction path; shares and future notionals are not rerun.
- Historical status: post-hoc diagnostic; it cannot authorize LIVE or release wiring.

## Platform-cost result

{_markdown_table_str(platform_df)}

## Frozen historical screens

{_markdown_table_str(promotion_gate_df)}

## Interpretation

The equal-weight and 12-1 rows are controls, not searched promotion candidates.
HAC inference and subperiod tables are saved separately. The approximate
cost-sensitivity rows cannot satisfy a mechanical promotion gate. No strategy,
LIVE configuration, scheduler, allocation, or broker route was changed.
"""
    (run_output_path / "REPORT.md").write_text(report_md_str, encoding="utf-8")


def _build_variant_bundle_by_key_dict(
    price_close_df: pd.DataFrame,
    baseline_bundle_obj: AdaptiveMovingAverageFactorSignalBundle,
    config_obj: AdaptiveMovingAverageFactorConfig,
) -> dict[str, AdaptiveMovingAverageFactorSignalBundle]:
    static_score_df = build_static_composite_score_df(
        price_close_df=price_close_df,
        baseline_bundle_obj=baseline_bundle_obj,
        sma_lookback_tuple=config_obj.sma_lookback_tuple,
    )
    momentum_score_df = build_classic_momentum_score_df(
        price_close_df=price_close_df,
        decision_date_index=pd.DatetimeIndex(
            baseline_bundle_obj.target_weight_df.index
        ),
    )
    return {
        BASELINE_VARIANT_KEY_STR: baseline_bundle_obj,
        BUFFERED_VARIANT_KEY_STR: build_buffered_signal_bundle(
            baseline_bundle_obj=baseline_bundle_obj,
        ),
        STATIC_VARIANT_KEY_STR: _bundle_from_score_df(
            baseline_bundle_obj=baseline_bundle_obj,
            score_df=static_score_df,
            variant_key_str=STATIC_VARIANT_KEY_STR,
        ),
        EQUAL_WEIGHT_CONTROL_KEY_STR: build_equal_weight_control_bundle(
            baseline_bundle_obj=baseline_bundle_obj,
        ),
        MOMENTUM_CONTROL_KEY_STR: _bundle_from_score_df(
            baseline_bundle_obj=baseline_bundle_obj,
            score_df=momentum_score_df,
            variant_key_str=MOMENTUM_CONTROL_KEY_STR,
        ),
    }


def run_sweep(
    output_dir_str: str = "results",
    sweep_output_dir_str: str | None = None,
    universe_key_list: Sequence[str] | None = None,
    backtest_start_date_str: str | None = None,
    end_date_str: str | None = None,
    show_progress_bool: bool = False,
    save_platform_reports_bool: bool = True,
) -> Path:
    selected_universe_key_list = (
        list(UNIVERSE_CONFIG_BY_KEY_DICT)
        if universe_key_list is None
        else list(universe_key_list)
    )
    unknown_universe_key_list = [
        universe_key_str
        for universe_key_str in selected_universe_key_list
        if universe_key_str not in UNIVERSE_CONFIG_BY_KEY_DICT
    ]
    if unknown_universe_key_list:
        raise ValueError(f"Unknown universe keys: {unknown_universe_key_list}")
    if set(selected_universe_key_list) != set(UNIVERSE_CONFIG_BY_KEY_DICT):
        raise ValueError(
            "The frozen sweep requires both Russell 1000 and Nasdaq-100."
        )

    if sweep_output_dir_str is None:
        timestamp_str = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        run_output_path = build_research_output_path(
            output_dir=output_dir_str,
            entity_type_str="strategy",
            entity_id_str="amaf_improvement_sweep",
            analysis_type_str="controlled_sweep",
            timestamp_str=timestamp_str,
        ).resolve()
    else:
        run_output_path = Path(sweep_output_dir_str).resolve()
    run_output_path.mkdir(parents=True, exist_ok=True)

    frozen_spec_dict = dict(RESEARCH_SPEC_DICT)
    frozen_spec_dict["frozen_at_utc_str"] = datetime.now(
        timezone.utc
    ).isoformat()
    frozen_spec_dict["requested_backtest_start_date_str"] = (
        backtest_start_date_str
    )
    frozen_spec_dict["requested_end_date_str"] = end_date_str
    (run_output_path / "research_spec.json").write_text(
        json.dumps(frozen_spec_dict, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    comparison_row_list: list[dict[str, object]] = []
    daily_return_row_list: list[dict[str, object]] = []
    monthly_return_row_list: list[dict[str, object]] = []
    subperiod_row_list: list[dict[str, object]] = []

    for universe_key_str in selected_universe_key_list:
        base_config_obj = UNIVERSE_CONFIG_BY_KEY_DICT[universe_key_str]
        config_obj = replace(
            base_config_obj,
            capital_base_float=RESEARCH_CAPITAL_BASE_FLOAT,
            backtest_start_date_str=(
                base_config_obj.backtest_start_date_str
                if backtest_start_date_str is None
                else backtest_start_date_str
            ),
            end_date_str=end_date_str,
        )
        print(f"LOAD {universe_key_str}", flush=True)
        (
            pricing_data_df,
            universe_df,
            rebalance_schedule_df,
        ) = get_adaptive_moving_average_factor_data(config_obj=config_obj)
        benchmark_symbol_set = set(config_obj.benchmark_list)
        tradeable_symbol_list = [
            str(symbol_str)
            for symbol_str in pricing_data_df.columns.get_level_values(0).unique()
            if (
                str(symbol_str) not in benchmark_symbol_set
                and str(symbol_str) in universe_df.columns
            )
        ]
        price_close_df = pd.DataFrame(
            {
                symbol_str: pricing_data_df[(symbol_str, "Close")]
                for symbol_str in tradeable_symbol_list
            },
            index=pricing_data_df.index,
            dtype=float,
        )
        raw_close_df = pd.DataFrame(
            {
                symbol_str: pricing_data_df[(symbol_str, "Unadjusted Close")]
                for symbol_str in tradeable_symbol_list
            },
            index=pricing_data_df.index,
            dtype=float,
        )
        monthly_decision_close_df = get_monthly_decision_close_df(
            price_close_df=price_close_df
        )
        print(f"BUILD TARGETS {universe_key_str}", flush=True)
        baseline_bundle_obj = build_adaptive_moving_average_factor_signal_bundle(
            price_close_df=price_close_df,
            raw_close_df=raw_close_df,
            universe_df=universe_df,
            decision_date_index=pd.DatetimeIndex(
                monthly_decision_close_df.index
            ),
            config_obj=config_obj,
        )
        variant_bundle_by_key_dict = _build_variant_bundle_by_key_dict(
            price_close_df=price_close_df,
            baseline_bundle_obj=baseline_bundle_obj,
            config_obj=config_obj,
        )
        universe_output_path = run_output_path / universe_key_str
        universe_output_path.mkdir(parents=True, exist_ok=True)
        for variant_key_str, signal_bundle_obj in (
            variant_bundle_by_key_dict.items()
        ):
            selection_df = signal_bundle_obj.forecast_df.loc[
                signal_bundle_obj.forecast_df["selected_bool"].astype(bool)
            ].copy()
            variant_output_path = universe_output_path / variant_key_str
            variant_output_path.mkdir(parents=True, exist_ok=True)
            selection_df.to_csv(
                variant_output_path / "selected_targets.csv",
                index=False,
            )
            signal_bundle_obj.coverage_df.to_csv(
                variant_output_path / "coverage.csv",
                index=False,
            )

        calendar_index = pricing_data_df.index[
            pricing_data_df.index
            >= pd.Timestamp(config_obj.backtest_start_date_str)
        ]
        for variant_key_str in VARIANT_KEY_TUPLE:
            signal_bundle_obj = variant_bundle_by_key_dict[variant_key_str]
            strategy_name_str = (
                f"strategy_mo_{universe_key_str}_{variant_key_str}"
            )
            strategy_config_obj = replace(
                config_obj,
                slippage_float=COST_TIER_SLIPPAGE_BY_KEY_DICT["platform"],
                strategy_name_str=strategy_name_str,
                variant_key_str=variant_key_str,
            )
            strategy_obj = PrecomputedAmafResearchStrategy(
                name=strategy_name_str,
                benchmarks=list(strategy_config_obj.benchmark_list),
                universe_df=universe_df,
                rebalance_schedule_df=rebalance_schedule_df,
                config_obj=strategy_config_obj,
                signal_bundle_obj=signal_bundle_obj,
            )
            print(
                f"RUN {universe_key_str} {variant_key_str} platform",
                flush=True,
            )
            # *** CRITICAL *** Full pre-start history remains available to the
            # frozen target builders; orders start only on the explicit
            # backtest calendar and execute at next-open through Vanilla.
            run_daily(
                strategy_obj,
                pricing_data_df,
                calendar=calendar_index,
                show_progress=show_progress_bool,
                show_signal_progress_bool=False,
                audit_override_bool=False,
            )
            for cost_tier_key_str, slippage_float in (
                COST_TIER_SLIPPAGE_BY_KEY_DICT.items()
            ):
                (
                    strategy_daily_return_ser,
                    incremental_cost_fraction_ser,
                ) = build_cost_stress_daily_return(
                    strategy_obj=strategy_obj,
                    target_slippage_per_side_float=float(slippage_float),
                )
                comparison_row_list.append(
                    _comparison_row_dict(
                        strategy_obj=strategy_obj,
                        universe_key_str=universe_key_str,
                        variant_key_str=variant_key_str,
                        cost_tier_key_str=cost_tier_key_str,
                        signal_bundle_obj=signal_bundle_obj,
                        daily_return_ser=strategy_daily_return_ser,
                        incremental_cost_fraction_ser=(
                            incremental_cost_fraction_ser
                        ),
                    )
                )
                for date_ts, daily_return_float in (
                    strategy_daily_return_ser.items()
                ):
                    daily_return_row_list.append(
                        {
                            "date_ts": pd.Timestamp(date_ts),
                            "universe_key_str": universe_key_str,
                            "variant_key_str": variant_key_str,
                            "cost_tier_key_str": cost_tier_key_str,
                            "daily_return_float": float(daily_return_float),
                        }
                    )
                # *** CRITICAL *** Report-only monthly compounding uses only
                # completed realized daily returns and cannot affect decisions.
                monthly_return_ser = (
                    strategy_daily_return_ser.add(1.0)
                    .resample("ME")
                    .prod(min_count=1)
                    .sub(1.0)
                    .dropna()
                )
                for month_end_ts, monthly_return_float in (
                    monthly_return_ser.items()
                ):
                    monthly_return_row_list.append(
                        {
                            "month_end_ts": pd.Timestamp(month_end_ts),
                            "universe_key_str": universe_key_str,
                            "variant_key_str": variant_key_str,
                            "cost_tier_key_str": cost_tier_key_str,
                            "monthly_return_float": float(
                                monthly_return_float
                            ),
                        }
                    )
                for (
                    subperiod_key_str,
                    subperiod_start_str,
                    subperiod_end_str,
                ) in SUBPERIOD_TUPLE:
                    subperiod_return_ser = strategy_daily_return_ser.loc[
                        pd.Timestamp(subperiod_start_str) : (
                            pd.Timestamp(subperiod_end_str)
                            if subperiod_end_str is not None
                            else None
                        )
                    ]
                    subperiod_row_list.append(
                        {
                            "universe_key_str": universe_key_str,
                            "variant_key_str": variant_key_str,
                            "cost_tier_key_str": cost_tier_key_str,
                            "subperiod_key_str": subperiod_key_str,
                            **_performance_metric_dict(
                                daily_return_ser=subperiod_return_ser
                            ),
                        }
                    )
            if save_platform_reports_bool:
                report_output_path = (
                    universe_output_path
                    / variant_key_str
                    / "platform_report"
                )
                save_results(
                    strategy_obj,
                    output_dir=output_dir_str,
                    output_path=report_output_path,
                )
            pd.DataFrame(comparison_row_list).to_csv(
                run_output_path / "comparison_partial.csv",
                index=False,
            )

    comparison_df = pd.DataFrame(comparison_row_list)
    daily_return_df = pd.DataFrame(daily_return_row_list)
    monthly_return_df = pd.DataFrame(monthly_return_row_list)
    subperiod_df = pd.DataFrame(subperiod_row_list)
    inference_df = _build_inference_df(monthly_return_df=monthly_return_df)
    promotion_gate_df = build_promotion_gate_df(
        comparison_df=comparison_df,
        inference_df=inference_df,
    )

    comparison_df.to_csv(run_output_path / "comparison.csv", index=False)
    daily_return_df.to_csv(run_output_path / "daily_returns.csv", index=False)
    monthly_return_df.to_csv(
        run_output_path / "monthly_returns.csv",
        index=False,
    )
    subperiod_df.to_csv(
        run_output_path / "subperiod_metrics.csv",
        index=False,
    )
    inference_df.to_csv(
        run_output_path / "paired_monthly_inference.csv",
        index=False,
    )
    promotion_gate_df.to_csv(
        run_output_path / "promotion_gate.csv",
        index=False,
    )
    _write_report(
        run_output_path=run_output_path,
        comparison_df=comparison_df,
        promotion_gate_df=promotion_gate_df,
    )
    metadata_dict = {
        "artifact_type_str": "amaf_improvement_controlled_sweep",
        "completed_at_utc_str": datetime.now(timezone.utc).isoformat(),
        "candidate_search_count_int": DECLARED_CANDIDATE_COUNT_INT,
        "primary_hypothesis_count_int": (
            DECLARED_PRIMARY_HYPOTHESIS_COUNT_INT
        ),
        "comparison_row_count_int": len(comparison_df),
        "research_only_bool": True,
        "live_wiring_changed_bool": False,
        "release_wiring_changed_bool": False,
    }
    (run_output_path / "metadata.json").write_text(
        json.dumps(metadata_dict, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    partial_path = run_output_path / "comparison_partial.csv"
    if partial_path.exists():
        partial_path.unlink()
    print(f"SWEEP SAVED {run_output_path}", flush=True)
    return run_output_path


def parse_args() -> argparse.Namespace:
    parser_obj = argparse.ArgumentParser()
    parser_obj.add_argument("--output-dir", default="results")
    parser_obj.add_argument("--sweep-output-dir", default=None)
    parser_obj.add_argument("--start", default=None)
    parser_obj.add_argument("--end", default=None)
    parser_obj.add_argument("--show-progress", action="store_true")
    parser_obj.add_argument("--no-platform-reports", action="store_true")
    return parser_obj.parse_args()


def main() -> None:
    arg_namespace = parse_args()
    run_sweep(
        output_dir_str=arg_namespace.output_dir,
        sweep_output_dir_str=arg_namespace.sweep_output_dir,
        backtest_start_date_str=arg_namespace.start,
        end_date_str=arg_namespace.end,
        show_progress_bool=arg_namespace.show_progress,
        save_platform_reports_bool=not arg_namespace.no_platform_reports,
    )


if __name__ == "__main__":
    main()
