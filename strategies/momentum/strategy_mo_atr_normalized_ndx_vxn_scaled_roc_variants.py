"""
Research ROC-window variants of the VXN-scaled ATR-normalized Nasdaq-100 model.

This module keeps the production-reference VXN-scaled ATR model intact and
changes only the momentum numerator and, for focused research grids, the ATR
denominator window.

Core formulas
-------------
For stock i on month-end decision date t:

    roc_last_12m_{i,t}
        = Close_ME_{i,t} / Close_ME_{i,t-12} - 1

    roc_last_1m_{i,t}
        = Close_ME_{i,t} / Close_ME_{i,t-1} - 1

    roc_prior_1m_{i,t}
        = Close_ME_{i,t-1} / Close_ME_{i,t-2} - 1

    roc_last_3m_{i,t}
        = Close_ME_{i,t} / Close_ME_{i,t-3} - 1

    roc_skip_12_1_{i,t}
        = Close_ME_{i,t-1} / Close_ME_{i,t-12} - 1

    roc_skip_6_1_{i,t}
        = Close_ME_{i,t-1} / Close_ME_{i,t-6} - 1

    roc_skip_3_1_{i,t}
        = Close_ME_{i,t-1} / Close_ME_{i,t-3} - 1

    risk_adj_score_{i,t}
        = chosen_roc_{i,t} / ATR_N_{i,t}

Everything else is inherited from the VXN-scaled ATR model:

    base_weight_{i,t}
        = 1 / max_positions    if i is selected by the score
        = 0                    otherwise

    vxn_scale_t
        = clip(target_vxn_pct / VXN_t, min_scale, max_scale)

    target_weight_{i,t}
        = base_weight_{i,t} * vxn_scale_t

Execution remains month-end decision close t -> next tradable open.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

import numpy as np
import pandas as pd
from IPython.display import display

from alpha.engine.backtest import run_daily
from alpha.engine.report import save_results
from strategies.momentum.strategy_mo_atr_normalized_ndx import (
    ATR_WINDOW_INT,
    audit_pit_universe_df,
    get_atr_normalized_ndx_data,
    get_monthly_decision_close_df,
    map_month_end_decision_dates_to_rebalance_schedule_df,
)
from strategies.momentum.strategy_mo_atr_normalized_ndx_vxn_scaled import (
    VxnScaledAtrNormalizedNdxConfig,
    VxnScaledAtrNormalizedNdxStrategy,
    compute_vxn_scale_signal_df,
    load_vxn_close_ser,
)


DEFAULT_ATR_WINDOW_INT = ATR_WINDOW_INT
ROC_MODE_LAST_12M_STR = "last_12m"
ROC_MODE_LAST_1M_STR = "last_1m"
ROC_MODE_PRIOR_1M_STR = "prior_1m"
ROC_MODE_LAST_3M_STR = "last_3m"
ROC_MODE_SKIP_12_1_STR = "skip_12_1"
ROC_MODE_SKIP_6_1_STR = "skip_6_1"
ROC_MODE_SKIP_3_1_STR = "skip_3_1"
ROC_MODE_EQUAL_SKIP_BLEND_STR = "equal_skip_3_6_12"
ROC_MODE_WEIGHTED_SKIP_BLEND_STR = "weighted_skip_3_6_12"
ROC_MODE_CONSISTENCY_SKIP_BLEND_STR = "consistency_skip_3_6_12"
ROC_MODE_ANTI_REVERSAL_SKIP_BLEND_STR = "anti_reversal_skip_3_6_12"
VALID_ROC_MODE_SET = frozenset(
    {
        ROC_MODE_LAST_12M_STR,
        ROC_MODE_LAST_1M_STR,
        ROC_MODE_PRIOR_1M_STR,
        ROC_MODE_LAST_3M_STR,
        ROC_MODE_SKIP_12_1_STR,
        ROC_MODE_SKIP_6_1_STR,
        ROC_MODE_SKIP_3_1_STR,
        ROC_MODE_EQUAL_SKIP_BLEND_STR,
        ROC_MODE_WEIGHTED_SKIP_BLEND_STR,
        ROC_MODE_CONSISTENCY_SKIP_BLEND_STR,
        ROC_MODE_ANTI_REVERSAL_SKIP_BLEND_STR,
    }
)


def get_required_roc_history_month_int(roc_mode_str: str) -> int:
    if roc_mode_str == ROC_MODE_LAST_12M_STR:
        return 12
    if roc_mode_str == ROC_MODE_LAST_1M_STR:
        return 1
    if roc_mode_str == ROC_MODE_PRIOR_1M_STR:
        return 2
    if roc_mode_str == ROC_MODE_LAST_3M_STR:
        return 3
    if roc_mode_str == ROC_MODE_SKIP_3_1_STR:
        return 3
    if roc_mode_str == ROC_MODE_SKIP_6_1_STR:
        return 6
    if roc_mode_str in {
        ROC_MODE_SKIP_12_1_STR,
        ROC_MODE_EQUAL_SKIP_BLEND_STR,
        ROC_MODE_WEIGHTED_SKIP_BLEND_STR,
        ROC_MODE_CONSISTENCY_SKIP_BLEND_STR,
        ROC_MODE_ANTI_REVERSAL_SKIP_BLEND_STR,
    }:
        return 12
    raise ValueError(
        f"Unsupported roc_mode_str={roc_mode_str!r}. "
        f"Expected one of {sorted(VALID_ROC_MODE_SET)}."
    )


@dataclass(frozen=True)
class VxnScaledAtrNormalizedNdxRocVariantConfig(VxnScaledAtrNormalizedNdxConfig):
    lookback_month_int: int = 1
    roc_mode_str: str = ROC_MODE_LAST_1M_STR
    atr_window_int: int = DEFAULT_ATR_WINDOW_INT

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.roc_mode_str not in VALID_ROC_MODE_SET:
            raise ValueError(
                f"roc_mode_str must be one of {sorted(VALID_ROC_MODE_SET)}, "
                f"got {self.roc_mode_str!r}."
            )
        if self.atr_window_int <= 0:
            raise ValueError("atr_window_int must be positive.")


DEFAULT_CONFIG = VxnScaledAtrNormalizedNdxRocVariantConfig()

__all__ = [
    "DEFAULT_CONFIG",
    "DEFAULT_ATR_WINDOW_INT",
    "ROC_MODE_LAST_12M_STR",
    "ROC_MODE_LAST_1M_STR",
    "ROC_MODE_LAST_3M_STR",
    "ROC_MODE_PRIOR_1M_STR",
    "ROC_MODE_ANTI_REVERSAL_SKIP_BLEND_STR",
    "ROC_MODE_CONSISTENCY_SKIP_BLEND_STR",
    "ROC_MODE_EQUAL_SKIP_BLEND_STR",
    "ROC_MODE_SKIP_12_1_STR",
    "ROC_MODE_SKIP_3_1_STR",
    "ROC_MODE_SKIP_6_1_STR",
    "ROC_MODE_WEIGHTED_SKIP_BLEND_STR",
    "VALID_ROC_MODE_SET",
    "VxnScaledAtrNormalizedNdxRocVariantConfig",
    "VxnScaledAtrNormalizedNdxRocVariantStrategy",
    "audit_pit_universe_df",
    "build_roc_variant_config",
    "compute_monthly_roc_variant_df",
    "compute_roc_variant_signal_tables",
    "get_monthly_decision_close_df",
    "get_required_roc_history_month_int",
    "get_vxn_scaled_atr_normalized_ndx_roc_variant_data",
    "make_roc_variant_rebalance_schedule_df",
    "map_month_end_decision_dates_to_rebalance_schedule_df",
    "run_last_1m_variant",
    "run_last_3m_variant",
    "run_prior_1m_variant",
    "run_variant",
]


def build_roc_variant_config(
    roc_mode_str: str,
    atr_window_int: int = DEFAULT_ATR_WINDOW_INT,
    backtest_start_date_str: str | None = None,
    capital_base_float: float | None = None,
    end_date_str: str | None = None,
) -> VxnScaledAtrNormalizedNdxRocVariantConfig:
    required_history_month_int = get_required_roc_history_month_int(roc_mode_str=roc_mode_str)
    return replace(
        DEFAULT_CONFIG,
        roc_mode_str=roc_mode_str,
        lookback_month_int=required_history_month_int,
        atr_window_int=int(atr_window_int),
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
    )


def compute_monthly_roc_variant_df(
    monthly_decision_close_df: pd.DataFrame,
    roc_mode_str: str,
) -> pd.DataFrame:
    if roc_mode_str not in VALID_ROC_MODE_SET:
        raise ValueError(
            f"Unsupported roc_mode_str={roc_mode_str!r}. "
            f"Expected one of {sorted(VALID_ROC_MODE_SET)}."
        )
    if len(monthly_decision_close_df) == 0:
        return monthly_decision_close_df.astype(float).copy()

    close_df = monthly_decision_close_df.astype(float)
    if roc_mode_str == ROC_MODE_LAST_12M_STR:
        # *** CRITICAL*** last_12m ROC uses the decision month-end close and
        # the trailing 12-month-old close only. This is the legacy base
        # momentum numerator, exposed here for ATR-window research.
        monthly_roc_df = (close_df / close_df.shift(12)) - 1.0
    elif roc_mode_str == ROC_MODE_LAST_1M_STR:
        # *** CRITICAL*** last_1m ROC uses only the decision month-end close
        # and the previous month-end close, both known at decision_t.
        monthly_roc_df = (close_df / close_df.shift(1)) - 1.0
    elif roc_mode_str == ROC_MODE_PRIOR_1M_STR:
        # *** CRITICAL*** prior_1m ROC intentionally skips the newest month.
        # It uses t-1 divided by t-2, never a value after decision_t.
        monthly_roc_df = (close_df.shift(1) / close_df.shift(2)) - 1.0
    elif roc_mode_str == ROC_MODE_LAST_3M_STR:
        # *** CRITICAL*** last_3m ROC uses the decision month-end close and
        # the trailing three-month-old close only.
        monthly_roc_df = (close_df / close_df.shift(3)) - 1.0
    else:
        skip_3_1_roc_df = _compute_skip_month_endpoint_roc_df(
            close_df=close_df,
            endpoint_month_int=3,
        )
        skip_6_1_roc_df = _compute_skip_month_endpoint_roc_df(
            close_df=close_df,
            endpoint_month_int=6,
        )
        skip_12_1_roc_df = _compute_skip_month_endpoint_roc_df(
            close_df=close_df,
            endpoint_month_int=12,
        )

        if roc_mode_str == ROC_MODE_SKIP_3_1_STR:
            monthly_roc_df = skip_3_1_roc_df
        elif roc_mode_str == ROC_MODE_SKIP_6_1_STR:
            monthly_roc_df = skip_6_1_roc_df
        elif roc_mode_str == ROC_MODE_SKIP_12_1_STR:
            monthly_roc_df = skip_12_1_roc_df
        elif roc_mode_str == ROC_MODE_EQUAL_SKIP_BLEND_STR:
            monthly_roc_df = (
                skip_3_1_roc_df
                + skip_6_1_roc_df
                + skip_12_1_roc_df
            ) / 3.0
        elif roc_mode_str == ROC_MODE_WEIGHTED_SKIP_BLEND_STR:
            monthly_roc_df = (
                0.20 * skip_3_1_roc_df
                + 0.30 * skip_6_1_roc_df
                + 0.50 * skip_12_1_roc_df
            )
        elif roc_mode_str == ROC_MODE_CONSISTENCY_SKIP_BLEND_STR:
            raw_blend_df = (
                skip_3_1_roc_df
                + skip_6_1_roc_df
                + skip_12_1_roc_df
            ) / 3.0
            positive_horizon_count_df = (
                (skip_3_1_roc_df > 0.0).astype(float)
                + (skip_6_1_roc_df > 0.0).astype(float)
                + (skip_12_1_roc_df > 0.0).astype(float)
            )
            positive_horizon_share_df = positive_horizon_count_df / 3.0
            monthly_roc_df = raw_blend_df * positive_horizon_share_df
        elif roc_mode_str == ROC_MODE_ANTI_REVERSAL_SKIP_BLEND_STR:
            # *** CRITICAL*** The one-month term is known at decision_t and is
            # subtracted as an explicit short-term reversal penalty. It must
            # never be replaced by a future post-decision month.
            last_1m_roc_df = (close_df / close_df.shift(1)) - 1.0
            weighted_skip_blend_df = (
                0.20 * skip_3_1_roc_df
                + 0.30 * skip_6_1_roc_df
                + 0.50 * skip_12_1_roc_df
            )
            monthly_roc_df = weighted_skip_blend_df - (0.25 * last_1m_roc_df)
        else:
            raise ValueError(
                f"Unsupported roc_mode_str={roc_mode_str!r}. "
                f"Expected one of {sorted(VALID_ROC_MODE_SET)}."
            )

    return monthly_roc_df.replace([np.inf, -np.inf], np.nan)


def _compute_skip_month_endpoint_roc_df(
    close_df: pd.DataFrame,
    endpoint_month_int: int,
) -> pd.DataFrame:
    if endpoint_month_int <= 1:
        raise ValueError("endpoint_month_int must be greater than one.")

    # *** CRITICAL*** skip-month endpoint momentum uses the close one month
    # before decision_t divided by the older endpoint close. For example,
    # 12-1 is Close_ME_{t-1} / Close_ME_{t-12} - 1. This excludes the newest
    # completed month while staying fully causal at the month-end decision.
    return (close_df.shift(1) / close_df.shift(endpoint_month_int)) - 1.0


def compute_roc_variant_signal_tables(
    price_close_df: pd.DataFrame,
    price_high_df: pd.DataFrame,
    price_low_df: pd.DataFrame,
    regime_close_ser: pd.Series,
    config: VxnScaledAtrNormalizedNdxRocVariantConfig = DEFAULT_CONFIG,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
    pd.DataFrame,
]:
    monthly_decision_close_df = get_monthly_decision_close_df(price_close_df=price_close_df)
    monthly_roc_df = compute_monthly_roc_variant_df(
        monthly_decision_close_df=monthly_decision_close_df,
        roc_mode_str=config.roc_mode_str,
    )

    # *** CRITICAL*** prior close alignment for true range must use shift(1)
    # so ATR is strictly trailing.
    prior_close_df = price_close_df.shift(1)
    true_range_df = (price_high_df - price_low_df).combine(
        (price_high_df - prior_close_df).abs(),
        np.maximum,
    )
    true_range_df = true_range_df.combine(
        (price_low_df - prior_close_df).abs(),
        np.maximum,
    )

    atr_window_int = int(config.atr_window_int)
    if atr_window_int <= 0:
        raise ValueError("atr_window_int must be positive.")

    # *** CRITICAL*** ATR_N must remain a trailing rolling mean of past true
    # range values only. N is a research parameter and must never use future
    # bars relative to the month-end decision.
    atr_value_df = true_range_df.rolling(
        window=atr_window_int,
        min_periods=atr_window_int,
    ).mean()
    atr_decision_df = atr_value_df.reindex(monthly_decision_close_df.index)

    # *** CRITICAL*** The stock trend filter must remain a trailing rolling
    # average on past closes only.
    stock_trend_sma_df = price_close_df.rolling(
        window=config.stock_trend_window_int,
        min_periods=config.stock_trend_window_int,
    ).mean()
    stock_trend_pass_df = (price_close_df > stock_trend_sma_df).reindex(
        monthly_decision_close_df.index
    )

    # *** CRITICAL*** The regime SMA filter must remain a trailing rolling
    # average on past SPY closes only.
    regime_close_decision_ser = regime_close_ser.reindex(monthly_decision_close_df.index)
    regime_sma_ser = regime_close_ser.rolling(
        window=config.index_trend_window_int,
        min_periods=config.index_trend_window_int,
    ).mean().reindex(monthly_decision_close_df.index)
    regime_pass_ser = regime_close_decision_ser > regime_sma_ser

    risk_adj_score_df = monthly_roc_df / atr_decision_df
    risk_adj_score_df = risk_adj_score_df.replace([np.inf, -np.inf], np.nan)

    valid_monthly_roc_bool_ser = monthly_roc_df.notna().any(axis=1)
    valid_atr_bool_ser = atr_decision_df.notna().any(axis=1)
    valid_stock_trend_bool_ser = stock_trend_pass_df.notna().any(axis=1)
    valid_regime_bool_ser = regime_close_decision_ser.notna() & regime_sma_ser.notna()
    valid_decision_index = monthly_decision_close_df.index[
        valid_monthly_roc_bool_ser
        & valid_atr_bool_ser
        & valid_stock_trend_bool_ser
        & valid_regime_bool_ser
    ]

    monthly_decision_close_df = monthly_decision_close_df.reindex(valid_decision_index)
    monthly_roc_df = monthly_roc_df.reindex(valid_decision_index)
    atr_decision_df = atr_decision_df.reindex(valid_decision_index)
    stock_trend_pass_df = stock_trend_pass_df.reindex(valid_decision_index)
    regime_sma_ser = regime_sma_ser.reindex(valid_decision_index)
    regime_pass_ser = regime_pass_ser.reindex(valid_decision_index)
    risk_adj_score_df = risk_adj_score_df.reindex(valid_decision_index)
    return (
        monthly_decision_close_df,
        monthly_roc_df,
        atr_decision_df,
        stock_trend_pass_df,
        regime_sma_ser,
        regime_pass_ser,
        risk_adj_score_df,
    )


def _extract_signal_inputs_from_pricing_data(
    pricing_data_df: pd.DataFrame,
    regime_symbol_str: str,
) -> tuple[list[str], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series]:
    tradeable_symbol_list = [
        str(symbol_str)
        for symbol_str in pricing_data_df.columns.get_level_values(0).unique()
        if str(symbol_str) != regime_symbol_str
    ]
    if len(tradeable_symbol_list) == 0:
        raise RuntimeError("No tradeable stock symbols were found in pricing_data_df.")

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

    regime_close_key = (regime_symbol_str, "Close")
    if regime_close_key not in pricing_data_df.columns:
        raise RuntimeError(f"Missing regime close data for {regime_symbol_str}.")
    regime_close_ser = pricing_data_df[regime_close_key].astype(float)
    return tradeable_symbol_list, price_close_df, price_high_df, price_low_df, regime_close_ser


def make_roc_variant_rebalance_schedule_df(
    pricing_data_df: pd.DataFrame,
    config: VxnScaledAtrNormalizedNdxRocVariantConfig,
) -> pd.DataFrame:
    (
        _tradeable_symbol_list,
        price_close_df,
        price_high_df,
        price_low_df,
        regime_close_ser,
    ) = _extract_signal_inputs_from_pricing_data(
        pricing_data_df=pricing_data_df,
        regime_symbol_str=config.regime_symbol_str,
    )
    (
        monthly_decision_close_df,
        _monthly_roc_df,
        _atr_decision_df,
        _stock_trend_pass_df,
        _regime_sma_ser,
        _regime_pass_ser,
        _risk_adj_score_df,
    ) = compute_roc_variant_signal_tables(
        price_close_df=price_close_df,
        price_high_df=price_high_df,
        price_low_df=price_low_df,
        regime_close_ser=regime_close_ser,
        config=config,
    )
    return map_month_end_decision_dates_to_rebalance_schedule_df(
        decision_date_index=pd.DatetimeIndex(monthly_decision_close_df.index),
        execution_index=pricing_data_df.index,
    )


class VxnScaledAtrNormalizedNdxRocVariantStrategy(VxnScaledAtrNormalizedNdxStrategy):
    """
    VXN-scaled ATR-normalized NDX model with a configurable ROC score window.
    """

    def __init__(
        self,
        name: str,
        benchmarks: Sequence[str],
        rebalance_schedule_df: pd.DataFrame,
        vxn_scale_signal_df: pd.DataFrame,
        roc_mode_str: str = ROC_MODE_LAST_1M_STR,
        regime_symbol_str: str = "SPY",
        capital_base: float = 100_000.0,
        slippage: float = 0.00025,
        commission_per_share: float = 0.005,
        commission_minimum: float = 1.0,
        lookback_month_int: int | None = None,
        index_trend_window_int: int = 200,
        stock_trend_window_int: int = 100,
        max_positions_int: int = 10,
        atr_window_int: int = DEFAULT_ATR_WINDOW_INT,
    ):
        required_history_month_int = get_required_roc_history_month_int(roc_mode_str=roc_mode_str)
        if lookback_month_int is not None and int(lookback_month_int) != required_history_month_int:
            raise ValueError(
                "lookback_month_int must match the required ROC history for "
                f"{roc_mode_str}: expected {required_history_month_int}, got {lookback_month_int}."
            )
        if atr_window_int <= 0:
            raise ValueError("atr_window_int must be positive.")

        super().__init__(
            name=name,
            benchmarks=benchmarks,
            rebalance_schedule_df=rebalance_schedule_df,
            vxn_scale_signal_df=vxn_scale_signal_df,
            regime_symbol_str=regime_symbol_str,
            capital_base=capital_base,
            slippage=slippage,
            commission_per_share=commission_per_share,
            commission_minimum=commission_minimum,
            lookback_month_int=required_history_month_int,
            index_trend_window_int=index_trend_window_int,
            stock_trend_window_int=stock_trend_window_int,
            max_positions_int=max_positions_int,
        )
        self.roc_mode_str = str(roc_mode_str)
        self.atr_window_int = int(atr_window_int)

    def compute_signals(self, pricing_data: pd.DataFrame) -> pd.DataFrame:
        signal_data_df = pricing_data.copy()
        (
            _tradeable_symbol_list,
            price_close_df,
            price_high_df,
            price_low_df,
            regime_close_ser,
        ) = _extract_signal_inputs_from_pricing_data(
            pricing_data_df=signal_data_df,
            regime_symbol_str=self.regime_symbol_str,
        )

        helper_config = VxnScaledAtrNormalizedNdxRocVariantConfig(
            regime_symbol_str=self.regime_symbol_str,
            lookback_month_int=self.lookback_month_int,
            index_trend_window_int=self.index_trend_window_int,
            stock_trend_window_int=self.stock_trend_window_int,
            max_positions_int=self.max_positions_int,
            roc_mode_str=self.roc_mode_str,
            atr_window_int=self.atr_window_int,
        )
        (
            _monthly_decision_close_df,
            monthly_roc_df,
            atr_decision_df,
            stock_trend_pass_df,
            regime_sma_ser,
            regime_pass_ser,
            risk_adj_score_df,
        ) = compute_roc_variant_signal_tables(
            price_close_df=price_close_df,
            price_high_df=price_high_df,
            price_low_df=price_low_df,
            regime_close_ser=regime_close_ser,
            config=helper_config,
        )

        monthly_roc_aligned_df = monthly_roc_df.reindex(signal_data_df.index)
        atr_aligned_df = atr_decision_df.reindex(signal_data_df.index)
        stock_trend_pass_aligned_df = stock_trend_pass_df.reindex(signal_data_df.index)
        risk_adj_score_aligned_df = risk_adj_score_df.reindex(signal_data_df.index)
        regime_sma_aligned_ser = regime_sma_ser.reindex(signal_data_df.index)
        regime_pass_aligned_ser = regime_pass_ser.reindex(signal_data_df.index)

        feature_frame_list: list[pd.DataFrame] = []
        feature_map: dict[str, pd.DataFrame] = {
            f"monthly_roc_{self.roc_mode_str}_ser": monthly_roc_aligned_df,
            f"atr_{self.atr_window_int}_ser": atr_aligned_df,
            "stock_trend_pass_bool": stock_trend_pass_aligned_df,
            "risk_adj_score_ser": risk_adj_score_aligned_df,
        }

        for field_str, field_df in feature_map.items():
            feature_df = field_df.copy()
            feature_df.columns = pd.MultiIndex.from_tuples(
                [(symbol_str, field_str) for symbol_str in feature_df.columns]
            )
            feature_frame_list.append(feature_df)

        regime_feature_df = pd.DataFrame(
            {
                (self.regime_symbol_str, f"regime_sma_{self.index_trend_window_int}_ser"): regime_sma_aligned_ser,
                (self.regime_symbol_str, "regime_pass_bool"): regime_pass_aligned_ser,
            },
            index=signal_data_df.index,
        )
        regime_feature_df.columns = pd.MultiIndex.from_tuples(regime_feature_df.columns)
        feature_frame_list.append(regime_feature_df)

        return pd.concat([signal_data_df] + feature_frame_list, axis=1)


def get_vxn_scaled_atr_normalized_ndx_roc_variant_data(
    config: VxnScaledAtrNormalizedNdxRocVariantConfig = DEFAULT_CONFIG,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    schedule_config_obj = replace(
        config,
        lookback_month_int=get_required_roc_history_month_int(roc_mode_str=config.roc_mode_str),
    )
    pricing_data_df, universe_df, _base_rebalance_schedule_df = get_atr_normalized_ndx_data(
        config=schedule_config_obj,
    )
    rebalance_schedule_df = make_roc_variant_rebalance_schedule_df(
        pricing_data_df=pricing_data_df,
        config=schedule_config_obj,
    )
    vxn_close_ser = load_vxn_close_ser(
        symbol_str=config.vxn_symbol_str,
        start_date_str=config.history_start_date_str,
        end_date_str=config.end_date_str,
    )
    vxn_scale_signal_df = compute_vxn_scale_signal_df(
        vxn_close_ser=vxn_close_ser,
        target_vxn_pct_float=config.target_vxn_pct_float,
        min_exposure_scale_float=config.min_exposure_scale_float,
        max_exposure_scale_float=config.max_exposure_scale_float,
    )
    return pricing_data_df, universe_df, rebalance_schedule_df, vxn_scale_signal_df


def run_variant(
    roc_mode_str: str = ROC_MODE_LAST_1M_STR,
    atr_window_int: int = DEFAULT_ATR_WINDOW_INT,
    show_display_bool: bool = True,
    save_results_bool: bool = True,
    output_dir_str: str = "results",
    backtest_start_date_str: str | None = None,
    capital_base_float: float | None = None,
    end_date_str: str | None = None,
) -> VxnScaledAtrNormalizedNdxRocVariantStrategy:
    config_obj = build_roc_variant_config(
        roc_mode_str=roc_mode_str,
        atr_window_int=atr_window_int,
        backtest_start_date_str=backtest_start_date_str,
        capital_base_float=capital_base_float,
        end_date_str=end_date_str,
    )
    pricing_data_df, universe_df, rebalance_schedule_df, vxn_scale_signal_df = (
        get_vxn_scaled_atr_normalized_ndx_roc_variant_data(config_obj)
    )

    strategy_obj = VxnScaledAtrNormalizedNdxRocVariantStrategy(
        name=f"strategy_mo_atr_normalized_ndx_vxn_scaled_{config_obj.roc_mode_str}",
        benchmarks=[config_obj.regime_symbol_str],
        rebalance_schedule_df=rebalance_schedule_df,
        vxn_scale_signal_df=vxn_scale_signal_df,
        roc_mode_str=config_obj.roc_mode_str,
        regime_symbol_str=config_obj.regime_symbol_str,
        capital_base=config_obj.capital_base_float,
        slippage=config_obj.slippage_float,
        commission_per_share=config_obj.commission_per_share_float,
        commission_minimum=config_obj.commission_minimum_float,
        lookback_month_int=config_obj.lookback_month_int,
        index_trend_window_int=config_obj.index_trend_window_int,
        stock_trend_window_int=config_obj.stock_trend_window_int,
        max_positions_int=config_obj.max_positions_int,
        atr_window_int=config_obj.atr_window_int,
    )
    strategy_obj.universe_df = universe_df

    # *** CRITICAL*** Research backtests keep full pre-start history for
    # trailing ATR, ROC, and trend features, while executable simulation starts
    # at the requested deployment comparison date.
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


def run_last_1m_variant(**run_kwarg_dict) -> VxnScaledAtrNormalizedNdxRocVariantStrategy:
    return run_variant(roc_mode_str=ROC_MODE_LAST_1M_STR, **run_kwarg_dict)


def run_prior_1m_variant(**run_kwarg_dict) -> VxnScaledAtrNormalizedNdxRocVariantStrategy:
    return run_variant(roc_mode_str=ROC_MODE_PRIOR_1M_STR, **run_kwarg_dict)


def run_last_3m_variant(**run_kwarg_dict) -> VxnScaledAtrNormalizedNdxRocVariantStrategy:
    return run_variant(roc_mode_str=ROC_MODE_LAST_3M_STR, **run_kwarg_dict)


if __name__ == "__main__":
    run_variant()
