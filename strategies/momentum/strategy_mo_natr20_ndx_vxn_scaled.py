"""
Independent NATR20 Nasdaq-100 momentum with monthly VXN position scaling.

Full strategy implementation: no imports or inheritance from sibling strategies.
Research/BENCH only. One requested variant; no parameter search. History through
2026 has already been examined and is not an untouched holdout.
Cash earns 0%; negative cash financing is unmodeled. Ordinary cash dividends
use the shared 25% withholding contract. Historical-equivalent whole-share
sizing and commissions are enabled; CAPITALSPECIAL is not a physical replay
of special-distribution shares. Costs: 2.5 bps/side, $0.005/share, $1 minimum.
Unused stock slots and VXN-reduced exposure remain in cash; no ROC>0 filter.

Core formulas
-------------
For stock i on month-end decision date t:

    monthly_roc_{i,t}^{(L)}
        = Close_ME_{i,t} / Close_ME_{i,t-L} - 1

    prior_close_{i,d}
        = Close_{i,d-1}

    TR_{i,d}
        = max(
            High_{i,d} - Low_{i,d},
            abs(High_{i,d} - prior_close_{i,d}),
            abs(Low_{i,d} - prior_close_{i,d})
        )

    ATR20_{i,t}
        = mean(TR_adjusted_{i,t-19:t}) * UnadjustedClose_{i,t} / Close_{i,t}

    Rebase the entire trailing ATR with the decision-date factor. This
    removes future corporate actions while retaining continuity across
    actions already known at t. Missing adjustment anchors are an error.

    stock_trend_pass_{i,t}
        = 1[Close_{i,t} > SMA100_{i,t}]

    regime_pass_t
        = 1[SPY_t > SMA200_t]

    risk_adj_score_{i,t}
        = monthly_roc_{i,t}^{(L)} / (ATR20_{i,t} / UnadjustedClose_{i,t})

    NATR20_pct_{i,t} = 100 * ATR20_{i,t} / UnadjustedClose_{i,t}

Selection on decision date t:

    eligible_{i,t}
        = 1[PIT_NDX_{i,t} = 1 and stock_trend_pass_{i,t} = 1]

    selected_t
        = top max_positions eligible symbols by risk_adj_score_{i,t}

    target_weight_{i,t}
        = clip(22 / VXN_asof_t, 0.25, 1.0) / max_positions    if i in selected_t
        = 0                    otherwise

Execution mapping:

    decision_date_t
        = actual last tradable close of month t

    execution_date_t
        = next tradable open after decision_date_t
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, replace
from typing import Sequence

import numpy as np
import pandas as pd
import exchange_calendars as exchange_calendar_module
from IPython.display import display

from alpha.engine.backtest import run_daily
from alpha.engine.report import save_results
from alpha.engine.strategy import Strategy
from data.norgate_loader import build_index_constituent_matrix, load_raw_prices, load_price_timeseries
from data.norgate_snapshot_store import (
    CAPITALSPECIAL_ADJUSTMENT_STR,
    TOTALRETURN_ADJUSTMENT_STR,
)


STRATEGY_NAME_STR = "strategy_mo_natr20_ndx_vxn_scaled"
ATR_WINDOW_INT = 20


def default_trade_id_int() -> int:
    return -1


def get_asof_universe_membership_ser(
    universe_df: pd.DataFrame,
    decision_date_ts: pd.Timestamp,
) -> pd.Series:
    if len(universe_df) == 0:
        raise RuntimeError("universe_df is empty.")

    sorted_universe_df = universe_df.sort_index()
    # *** CRITICAL*** PIT universe membership may lag the newest price date.
    # Use only the latest universe row available on or before decision_t; never
    # use a later row, because that would leak future index membership.
    universe_row_int = int(
        sorted_universe_df.index.searchsorted(pd.Timestamp(decision_date_ts), side="right")
    ) - 1
    if universe_row_int < 0:
        raise RuntimeError(f"universe_df has no row on or before decision date {decision_date_ts}.")
    return sorted_universe_df.iloc[universe_row_int]


@dataclass(frozen=True)
class Natr20VxnScaledNdxConfig:
    indexname_str: str = "Nasdaq 100"
    regime_symbol_str: str = "SPY"
    performance_benchmark_symbol_str: str = "$SPX"
    performance_benchmark_data_symbol_str: str = "$SPXTR"
    history_start_date_str: str = "1999-01-01"
    backtest_start_date_str: str = "2000-01-01"
    end_date_str: str | None = None
    lookback_month_int: int = 12
    index_trend_window_int: int = 200
    stock_trend_window_int: int = 100
    max_positions_int: int = 10
    capital_base_float: float = 100_000.0
    slippage_float: float = 0.00025
    commission_per_share_float: float = 0.005
    commission_minimum_float: float = 1.0
    vxn_symbol_str: str = "$VXN"
    target_vxn_pct_float: float = 22.0
    min_exposure_scale_float: float = 0.25
    max_exposure_scale_float: float = 1.0

    def __post_init__(self) -> None:
        if not self.vxn_symbol_str:
            raise ValueError("vxn_symbol_str must not be empty.")
        if not np.isfinite(self.target_vxn_pct_float) or self.target_vxn_pct_float <= 0.0:
            raise ValueError("target_vxn_pct_float must be finite and positive.")
        if not (0.0 <= self.min_exposure_scale_float <= self.max_exposure_scale_float <= 1.0):
            raise ValueError("VXN exposure bounds must satisfy 0 <= min <= max <= 1.")
        if not self.indexname_str:
            raise ValueError("indexname_str must not be empty.")
        if not self.regime_symbol_str:
            raise ValueError("regime_symbol_str must not be empty.")
        if not self.performance_benchmark_symbol_str:
            raise ValueError("performance_benchmark_symbol_str must not be empty.")
        if not self.performance_benchmark_data_symbol_str:
            raise ValueError("performance_benchmark_data_symbol_str must not be empty.")
        if pd.Timestamp(self.history_start_date_str) >= pd.Timestamp(self.backtest_start_date_str):
            raise ValueError("history_start_date_str must be earlier than backtest_start_date_str.")
        if self.lookback_month_int <= 0:
            raise ValueError("lookback_month_int must be positive.")
        if self.index_trend_window_int <= 0:
            raise ValueError("index_trend_window_int must be positive.")
        if self.stock_trend_window_int <= 0:
            raise ValueError("stock_trend_window_int must be positive.")
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


DEFAULT_CONFIG = Natr20VxnScaledNdxConfig()

__all__ = [
    "ATR_WINDOW_INT",
    "Natr20VxnScaledNdxConfig",
    "Natr20VxnScaledNdxStrategy",
    "DEFAULT_CONFIG",
    "append_total_return_benchmark_data_df",
    "audit_pit_universe_df",
    "build_execution_timing_analysis_inputs",
    "compute_natr20_signal_tables",
    "configure_total_return_benchmark_provenance",
    "get_natr20_vxn_scaled_ndx_data",
    "get_monthly_decision_close_df",
    "map_month_end_decision_dates_to_rebalance_schedule_df",
    "run_variant",
]


def load_vxn_close_ser(
    symbol_str: str,
    start_date_str: str,
    end_date_str: str | None,
) -> pd.Series:
    """
    Load the VXN close series from Norgate.
    """
    vxn_price_df = load_price_timeseries(
        symbol_str,
        adjustment_str=CAPITALSPECIAL_ADJUSTMENT_STR,
        start_date_str=start_date_str,
        end_date_str=end_date_str,
    )
    if len(vxn_price_df) == 0:
        raise RuntimeError(f"{symbol_str} returned no VXN helper data.")

    vxn_close_ser = vxn_price_df["Close"].astype(float).sort_index()
    vxn_close_ser.name = symbol_str
    return vxn_close_ser


def compute_vxn_scale_signal_df(
    vxn_close_ser: pd.Series,
    target_vxn_pct_float: float = DEFAULT_CONFIG.target_vxn_pct_float,
    min_exposure_scale_float: float = DEFAULT_CONFIG.min_exposure_scale_float,
    max_exposure_scale_float: float = DEFAULT_CONFIG.max_exposure_scale_float,
) -> pd.DataFrame:
    """
    Compute the daily VXN exposure scale.

    Formula:

        vxn_scale_t = clip(target_vxn_pct / VXN_t, min_scale, max_scale)
    """
    if target_vxn_pct_float <= 0.0:
        raise ValueError("target_vxn_pct_float must be positive.")
    if min_exposure_scale_float < 0.0:
        raise ValueError("min_exposure_scale_float must be non-negative.")
    if min_exposure_scale_float > max_exposure_scale_float:
        raise ValueError("min_exposure_scale_float must be <= max_exposure_scale_float.")
    if max_exposure_scale_float > 1.0:
        raise ValueError("max_exposure_scale_float must be <= 1.0 for this no-leverage variant.")

    if not isinstance(vxn_close_ser.index, pd.DatetimeIndex) or not vxn_close_ser.index.is_unique:
        raise ValueError("VXN must have a unique DatetimeIndex.")
    if not np.isfinite(target_vxn_pct_float):
        raise ValueError("target_vxn_pct_float must be finite.")
    clean_vxn_close_ser = vxn_close_ser.astype(float).sort_index().dropna()
    if (~np.isfinite(clean_vxn_close_ser) | clean_vxn_close_ser.le(0.0)).any():
        raise ValueError("Observed VXN closes must be finite and positive.")
    if len(clean_vxn_close_ser) == 0:
        raise ValueError("vxn_close_ser must contain at least one non-null close.")

    vxn_scale_signal_df = pd.DataFrame({"vxn_close": clean_vxn_close_ser})

    # *** CRITICAL*** VXN scaling uses only the VXN close observed on or before
    # the month-end decision close. No future VXN value may enter this series.
    raw_exposure_scale_ser = float(target_vxn_pct_float) / vxn_scale_signal_df["vxn_close"]
    exposure_scale_ser = raw_exposure_scale_ser.replace([np.inf, -np.inf], np.nan).clip(
        lower=float(min_exposure_scale_float),
        upper=float(max_exposure_scale_float),
    )
    vxn_scale_signal_df["vxn_exposure_scale_float"] = exposure_scale_ser
    return vxn_scale_signal_df.dropna(subset=["vxn_exposure_scale_float"])


def get_asof_vxn_scale_float(
    vxn_scale_signal_df: pd.DataFrame,
    decision_date_ts: pd.Timestamp,
) -> float:
    """
    Return the latest VXN exposure scale known on or before decision_date_ts.
    """
    if len(vxn_scale_signal_df) == 0:
        raise RuntimeError("vxn_scale_signal_df must not be empty.")
    if "vxn_exposure_scale_float" not in vxn_scale_signal_df.columns:
        raise RuntimeError("vxn_scale_signal_df must contain vxn_exposure_scale_float.")

    sorted_vxn_scale_signal_df = vxn_scale_signal_df.sort_index()
    if sorted_vxn_scale_signal_df.index.has_duplicates:
        raise RuntimeError("vxn_scale_signal_df index must not contain duplicates.")

    # *** CRITICAL*** This is an as-of lookup. If VXN has no row on the exact
    # stock decision date, use only the latest prior VXN close, never a later
    # row that would leak future volatility information into the rebalance.
    vxn_row_int = int(
        sorted_vxn_scale_signal_df.index.searchsorted(pd.Timestamp(decision_date_ts), side="right")
    ) - 1
    if vxn_row_int < 0:
        raise RuntimeError(f"No VXN scale exists on or before decision date {decision_date_ts}.")

    exposure_scale_float = float(
        sorted_vxn_scale_signal_df.iloc[vxn_row_int]["vxn_exposure_scale_float"]
    )
    if not np.isfinite(exposure_scale_float) or not 0.0 <= exposure_scale_float <= 1.0:
        raise RuntimeError(f"Invalid VXN exposure scale for decision date {decision_date_ts}.")
    return exposure_scale_float


def append_total_return_benchmark_data_df(
    pricing_data_df: pd.DataFrame,
    config_obj: Natr20VxnScaledNdxConfig,
) -> pd.DataFrame:
    total_return_benchmark_df = load_raw_prices(
        symbols=[],
        benchmarks=[config_obj.performance_benchmark_data_symbol_str],
        start_date=config_obj.history_start_date_str,
        end_date=config_obj.end_date_str,
    )
    benchmark_data_symbol_str = config_obj.performance_benchmark_data_symbol_str
    # *** CRITICAL*** Benchmark provenance must not expand or remap the
    # strategy calendar. Align the TR benchmark to the already-established
    # CAPITALSPECIAL price index; it is reporting-only and cannot create a
    # signal or execution session.
    total_return_benchmark_df = total_return_benchmark_df.reindex(
        pricing_data_df.index
    )
    benchmark_close_ser = total_return_benchmark_df[
        (benchmark_data_symbol_str, "Close")
    ]
    if benchmark_close_ser.isna().any():
        missing_benchmark_date_list = [
            pd.Timestamp(date_obj).date().isoformat()
            for date_obj in benchmark_close_ser.index[benchmark_close_ser.isna()][:5]
        ]
        raise RuntimeError(
            "TOTALRETURN benchmark is missing CAPITALSPECIAL calendar dates: "
            f"symbol={config_obj.performance_benchmark_data_symbol_str} "
            f"sample_dates={missing_benchmark_date_list}"
        )
    adjustment_by_symbol_dict = {
        **dict(
            pricing_data_df.attrs.get(
                "norgate_adjustment_by_symbol_dict",
                {},
            )
        ),
        **dict(
            total_return_benchmark_df.attrs.get(
                "norgate_adjustment_by_symbol_dict",
                {},
            )
        ),
    }
    combined_price_df = pd.concat(
        [pricing_data_df, total_return_benchmark_df],
        axis=1,
    ).sort_index()
    combined_price_df.attrs["norgate_adjustment_by_symbol_dict"] = (
        adjustment_by_symbol_dict
    )
    return combined_price_df


def configure_total_return_benchmark_provenance(
    strategy_obj: Strategy,
    config_obj: Natr20VxnScaledNdxConfig,
) -> None:
    benchmark_data_symbol_str = config_obj.performance_benchmark_data_symbol_str
    strategy_obj._benchmark_data_symbol_map_dict = {
        config_obj.performance_benchmark_symbol_str: benchmark_data_symbol_str
    }
    strategy_obj._performance_benchmark_symbol_str = (
        config_obj.performance_benchmark_symbol_str
    )
    strategy_obj._performance_benchmark_adjustment_str = TOTALRETURN_ADJUSTMENT_STR
    strategy_obj._data_adjustment_policy_dict = {
        "execution_and_marks_adjustment_str": CAPITALSPECIAL_ADJUSTMENT_STR,
        "regime_signal_adjustment_str": CAPITALSPECIAL_ADJUSTMENT_STR,
        "performance_benchmark_adjustment_str": TOTALRETURN_ADJUSTMENT_STR,
        "performance_benchmark_data_symbol_str": benchmark_data_symbol_str,
    }


def audit_pit_universe_df(
    universe_df: pd.DataFrame,
    execution_idx: pd.DatetimeIndex,
    tradeable_symbol_list: Sequence[str],
) -> pd.DataFrame:
    if not universe_df.index.is_monotonic_increasing:
        raise ValueError("universe_df index must be sorted.")
    if universe_df.index.has_duplicates:
        raise ValueError("universe_df index must not contain duplicates.")

    aligned_symbol_list = [
        symbol_str for symbol_str in tradeable_symbol_list if symbol_str in universe_df.columns
    ]
    if len(aligned_symbol_list) == 0:
        raise RuntimeError("No tradeable symbols overlap between pricing data and universe_df.")

    # *** CRITICAL*** Align PIT membership to every price date by causal
    # forward-fill only. This supports Norgate universe matrices that lag the
    # latest price date while preserving member_{i,t} = member_{i,max(s <= t)}.
    aligned_universe_df = universe_df.loc[:, aligned_symbol_list].reindex(execution_idx).ffill()
    missing_execution_index = aligned_universe_df.index[aligned_universe_df.isna().any(axis=1)]
    if len(missing_execution_index) > 0:
        missing_date_preview_list = [pd.Timestamp(date_ts).strftime("%Y-%m-%d") for date_ts in missing_execution_index[:5]]
        raise RuntimeError(
            "PIT universe is missing execution dates after loader alignment. "
            f"First missing dates: {missing_date_preview_list}"
        )

    return aligned_universe_df.astype(int).sort_index()


def get_monthly_decision_close_df(price_close_df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse daily closes to the actual last tradable close of each month.
    """
    if len(price_close_df.index) == 0:
        raise ValueError("price_close_df must not be empty.")

    # *** CRITICAL*** Monthly decisions must use the actual last tradable
    # close in each month, not a synthetic calendar month-end timestamp.
    decision_date_ser = pd.Series(
        price_close_df.index,
        index=price_close_df.index.to_period("M"),
    ).groupby(level=0).max()

    last_available_ts = pd.Timestamp(price_close_df.index[-1])
    # *** CRITICAL *** Calendar business days include exchange holidays. An
    # as-of prefix ending on Good Friday eve must recognize the same actual
    # month-end as the full backtest. Use the established XNYS schedule.
    month_start_ts = last_available_ts.to_period("M").start_time.normalize()
    month_end_ts = last_available_ts.to_period("M").end_time.normalize()
    exchange_calendar_obj = exchange_calendar_module.get_calendar(
        "XNYS", start=f"{last_available_ts.year - 1}-12-01",
        end=f"{last_available_ts.year + 1}-01-31",
    )
    month_session_idx = exchange_calendar_obj.sessions_in_range(month_start_ts, month_end_ts)
    expected_business_month_end_ts = pd.Timestamp(month_session_idx[-1]).tz_localize(None)
    if (
        len(decision_date_ser) > 0
        and pd.Timestamp(decision_date_ser.iloc[-1]) == last_available_ts
        and expected_business_month_end_ts.normalize() != last_available_ts.normalize()
    ):
        decision_date_ser = decision_date_ser.iloc[:-1]

    decision_date_idx = pd.DatetimeIndex(decision_date_ser.to_numpy(), name="decision_date_ts")
    monthly_decision_close_df = price_close_df.loc[decision_date_idx].copy()
    monthly_decision_close_df.index = decision_date_idx
    return monthly_decision_close_df


def get_unadjusted_close_df(
    pricing_data_df: pd.DataFrame,
    symbol_list: list[str],
) -> pd.DataFrame:
    missing_symbol_list = [
        symbol_str for symbol_str in symbol_list
        if (symbol_str, "Unadjusted Close") not in pricing_data_df.columns
    ]
    if missing_symbol_list:
        raise ValueError(
            "Historical dollar ATR requires Unadjusted Close; missing: "
            + ", ".join(missing_symbol_list[:10])
        )
    return pd.DataFrame(
        {symbol_str: pricing_data_df[(symbol_str, "Unadjusted Close")]
         for symbol_str in symbol_list}, index=pricing_data_df.index,
    ).astype(float)


def compute_natr20_signal_tables(
    price_close_df: pd.DataFrame,
    price_high_df: pd.DataFrame,
    price_low_df: pd.DataFrame,
    regime_close_ser: pd.Series,
    config_obj: Natr20VxnScaledNdxConfig = DEFAULT_CONFIG,
    *,
    price_unadjusted_close_df: pd.DataFrame,
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

    # *** CRITICAL*** Monthly ROC must use only actual month-end decision
    # closes and trailing month-end history.
    monthly_roc_df = (
        monthly_decision_close_df / monthly_decision_close_df.shift(config_obj.lookback_month_int)
    ) - 1.0

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

    # *** CRITICAL*** ATR20 must remain a trailing rolling mean of past true
    # range values only.
    atr_value_df = true_range_df.rolling(
        window=ATR_WINDOW_INT,
        min_periods=ATR_WINDOW_INT,
    ).mean()
    atr_decision_df = atr_value_df.reindex(monthly_decision_close_df.index)
    # *** CRITICAL *** Future corporate actions rescale historical OHLC and
    # ATR. Rebase the whole trailing window into Close_T nominal units using
    # k_T = UnadjustedClose_T / AdjustedClose_T. Never use k from T+1, or raw
    # OHLC bar by bar: the latter introduces artificial split gaps into ATR.
    unadjusted_decision_close_df = price_unadjusted_close_df.reindex(
        index=monthly_decision_close_df.index, columns=monthly_decision_close_df.columns,
    )
    observed_close_mask_df = monthly_decision_close_df.notna()
    invalid_anchor_mask_df = observed_close_mask_df & (
        ~np.isfinite(unadjusted_decision_close_df)
        | (unadjusted_decision_close_df <= 0.0)
        | ~np.isfinite(monthly_decision_close_df)
        | (monthly_decision_close_df <= 0.0)
    )
    if invalid_anchor_mask_df.any().any():
        raise ValueError("Invalid or missing decision-date Unadjusted Close/Close anchor.")
    atr_decision_df = atr_decision_df * (
        unadjusted_decision_close_df / monthly_decision_close_df
    )

    # *** CRITICAL*** The stock trend filter must remain a trailing rolling
    # average on past closes only.
    stock_trend_sma_df = price_close_df.rolling(
        window=config_obj.stock_trend_window_int,
        min_periods=config_obj.stock_trend_window_int,
    ).mean()
    stock_trend_pass_df = (price_close_df > stock_trend_sma_df).reindex(monthly_decision_close_df.index)

    # *** CRITICAL*** The regime SMA filter must remain a trailing rolling
    # average on past SPY closes only.
    regime_close_decision_ser = regime_close_ser.reindex(monthly_decision_close_df.index)
    regime_sma_ser = regime_close_ser.rolling(
        window=config_obj.index_trend_window_int,
        min_periods=config_obj.index_trend_window_int,
    ).mean().reindex(monthly_decision_close_df.index)
    regime_pass_ser = regime_close_decision_ser > regime_sma_ser

    # *** CRITICAL *** Both ATR and raw close are in decision-date units.
    # Score_T = (ROC12_T / ATR_nominal_T) * RawClose_T
    #         = ROC12_T / (ATR_adjusted_T / AdjustedClose_T).
    # A later split rescales both adjusted terms and cancels out.
    risk_adj_score_df = (monthly_roc_df / atr_decision_df) * unadjusted_decision_close_df
    risk_adj_score_df = risk_adj_score_df.replace([np.inf, -np.inf], np.nan)

    valid_monthly_roc_bool_ser = monthly_roc_df.notna().any(axis=1)
    valid_atr_bool_ser = atr_decision_df.notna().any(axis=1)
    valid_stock_trend_bool_ser = stock_trend_pass_df.notna().any(axis=1)
    valid_regime_bool_ser = regime_close_decision_ser.notna() & regime_sma_ser.notna()
    valid_decision_idx = monthly_decision_close_df.index[
        valid_monthly_roc_bool_ser
        & valid_atr_bool_ser
        & valid_stock_trend_bool_ser
        & valid_regime_bool_ser
    ]

    monthly_decision_close_df = monthly_decision_close_df.reindex(valid_decision_idx)
    monthly_roc_df = monthly_roc_df.reindex(valid_decision_idx)
    atr_decision_df = atr_decision_df.reindex(valid_decision_idx)
    stock_trend_pass_df = stock_trend_pass_df.reindex(valid_decision_idx)
    regime_sma_ser = regime_sma_ser.reindex(valid_decision_idx)
    regime_pass_ser = regime_pass_ser.reindex(valid_decision_idx)
    risk_adj_score_df = risk_adj_score_df.reindex(valid_decision_idx)
    return (
        monthly_decision_close_df,
        monthly_roc_df,
        atr_decision_df,
        stock_trend_pass_df,
        regime_sma_ser,
        regime_pass_ser,
        risk_adj_score_df,
    )


def map_month_end_decision_dates_to_rebalance_schedule_df(
    decision_date_idx: pd.DatetimeIndex,
    execution_idx: pd.DatetimeIndex,
) -> pd.DataFrame:
    """
    Map each month-end decision close to the next tradable open.
    """
    if len(execution_idx) < 2:
        raise ValueError("execution_idx must contain at least two trading dates.")
    if len(decision_date_idx) == 0:
        raise ValueError("decision_date_idx must not be empty.")

    execution_idx = pd.DatetimeIndex(execution_idx).sort_values()
    decision_date_idx = pd.DatetimeIndex(decision_date_idx).sort_values()

    rebalance_schedule_dict: dict[pd.Timestamp, pd.Timestamp] = {}
    for decision_date_ts in decision_date_idx:
        execution_insert_int = int(execution_idx.searchsorted(pd.Timestamp(decision_date_ts), side="right"))
        if execution_insert_int >= len(execution_idx):
            continue

        # *** CRITICAL*** Month-end decisions must execute strictly on the
        # next tradable open after the decision close, never on the same bar.
        execution_date_ts = pd.Timestamp(execution_idx[execution_insert_int])
        rebalance_schedule_dict[execution_date_ts] = pd.Timestamp(decision_date_ts)

    if len(rebalance_schedule_dict) == 0:
        raise RuntimeError("No month-end rebalance dates were generated.")

    rebalance_schedule_df = pd.DataFrame.from_dict(
        rebalance_schedule_dict,
        orient="index",
        columns=["decision_date_ts"],
    ).sort_index()
    rebalance_schedule_df.index.name = "execution_date_ts"
    return rebalance_schedule_df


def get_natr20_vxn_scaled_ndx_data(
    config_obj: Natr20VxnScaledNdxConfig = DEFAULT_CONFIG,
    *,
    include_total_return_benchmark_bool: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    _, raw_universe_df = build_index_constituent_matrix(indexname=config_obj.indexname_str)

    history_start_ts = pd.Timestamp(config_obj.history_start_date_str)
    backtest_start_ts = pd.Timestamp(config_obj.backtest_start_date_str)
    filtered_universe_df = raw_universe_df.loc[raw_universe_df.index >= history_start_ts].copy()
    active_universe_df = filtered_universe_df.loc[filtered_universe_df.index >= backtest_start_ts].copy()
    if config_obj.end_date_str is not None:
        end_date_ts = pd.Timestamp(config_obj.end_date_str)
        active_universe_df = active_universe_df.loc[active_universe_df.index <= end_date_ts]

    active_symbol_list = active_universe_df.columns[active_universe_df.sum(axis=0) > 0].tolist()
    if len(active_symbol_list) == 0:
        raise RuntimeError("No active Nasdaq-100 universe symbols were found for the requested backtest window.")

    price_symbol_list = list(dict.fromkeys(active_symbol_list + [config_obj.regime_symbol_str]))
    pricing_data_df = load_raw_prices(
        symbols=price_symbol_list,
        benchmarks=[],
        start_date=config_obj.history_start_date_str,
        end_date=config_obj.end_date_str,
    )
    loaded_symbol_list = [
        symbol_str
        for symbol_str in active_symbol_list
        if symbol_str in pricing_data_df.columns.get_level_values(0)
    ]
    audited_universe_df = audit_pit_universe_df(
        universe_df=filtered_universe_df,
        execution_idx=pricing_data_df.index,
        tradeable_symbol_list=loaded_symbol_list,
    )

    keep_symbol_set = set(audited_universe_df.columns.tolist() + [config_obj.regime_symbol_str])
    pricing_data_df = pricing_data_df.loc[
        :,
        pricing_data_df.columns.get_level_values(0).isin(keep_symbol_set),
    ].sort_index()
    if include_total_return_benchmark_bool:
        pricing_data_df = append_total_return_benchmark_data_df(
            pricing_data_df=pricing_data_df,
            config_obj=config_obj,
        )

    close_symbol_list = audited_universe_df.columns.tolist()
    price_close_df = pd.DataFrame(
        {symbol_str: pricing_data_df[(symbol_str, "Close")] for symbol_str in close_symbol_list},
        index=pricing_data_df.index,
    ).astype(float)
    price_high_df = pd.DataFrame(
        {symbol_str: pricing_data_df[(symbol_str, "High")] for symbol_str in close_symbol_list},
        index=pricing_data_df.index,
    ).astype(float)
    price_low_df = pd.DataFrame(
        {symbol_str: pricing_data_df[(symbol_str, "Low")] for symbol_str in close_symbol_list},
        index=pricing_data_df.index,
    ).astype(float)
    regime_close_ser = pricing_data_df[(config_obj.regime_symbol_str, "Close")].astype(float)

    (
        monthly_decision_close_df,
        _monthly_roc_df,
        _atr_decision_df,
        _stock_trend_pass_df,
        _regime_sma_ser,
        _regime_pass_ser,
        _risk_adj_score_df,
    ) = compute_natr20_signal_tables(
        price_close_df=price_close_df,
        price_high_df=price_high_df,
        price_low_df=price_low_df,
        regime_close_ser=regime_close_ser,
        config_obj=config_obj,
        price_unadjusted_close_df=get_unadjusted_close_df(pricing_data_df, loaded_symbol_list),
    )
    rebalance_schedule_df = map_month_end_decision_dates_to_rebalance_schedule_df(
        decision_date_idx=pd.DatetimeIndex(monthly_decision_close_df.index),
        execution_idx=pricing_data_df.index,
    )
    vxn_close_ser = load_vxn_close_ser(config_obj.vxn_symbol_str, config_obj.history_start_date_str, config_obj.end_date_str)
    vxn_scale_signal_df = compute_vxn_scale_signal_df(
        vxn_close_ser, config_obj.target_vxn_pct_float,
        config_obj.min_exposure_scale_float, config_obj.max_exposure_scale_float,
    )
    return pricing_data_df, audited_universe_df, rebalance_schedule_df, vxn_scale_signal_df


def _map_rebalance_schedule_to_decision_close_schedule_df(
    rebalance_schedule_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convert next-open rebalance schedule to decision-close order-intent dates.
    """
    decision_schedule_map: dict[pd.Timestamp, pd.Timestamp] = {}
    for _execution_date_ts, schedule_row_ser in rebalance_schedule_df.iterrows():
        decision_date_ts = pd.Timestamp(schedule_row_ser["decision_date_ts"])
        # *** CRITICAL*** ExecutionTimingAnalysis emits order intent on the
        # actual month-end decision close, then the matrix applies T close,
        # T+1 open, or T+1 close fills. Keeping the original execution-date
        # schedule here would hide the biased T close diagnostic.
        decision_schedule_map[decision_date_ts] = decision_date_ts

    if len(decision_schedule_map) == 0:
        raise RuntimeError("No decision-close dates were available for execution timing analysis.")

    decision_close_schedule_df = pd.DataFrame.from_dict(
        decision_schedule_map,
        orient="index",
        columns=["decision_date_ts"],
    ).sort_index()
    decision_close_schedule_df.index.name = "decision_close_date_ts"
    return decision_close_schedule_df


def build_execution_timing_analysis_inputs() -> dict[str, object]:
    """
    Build inputs for ExecutionTimingAnalysis.

    Formula:

        decision_t = monthly ATR-normalized signal known after T close

        entry_fill = decision_t + entry_lag at entry_price_field
        exit_fill  = decision_t + exit_lag  at exit_price_field
    """
    config_obj = DEFAULT_CONFIG
    pricing_data_df, universe_df, rebalance_schedule_df, vxn_scale_signal_df = get_natr20_vxn_scaled_ndx_data(
        config_obj,
        include_total_return_benchmark_bool=True,
    )
    decision_close_schedule_df = _map_rebalance_schedule_to_decision_close_schedule_df(
        rebalance_schedule_df=rebalance_schedule_df,
    )

    def strategy_factory_fn():
        strategy_obj = Natr20VxnScaledNdxStrategy(
            name="strategy_mo_natr20_ndx_vxn_scaled",
            benchmarks=[config_obj.performance_benchmark_symbol_str],
            rebalance_schedule_df=decision_close_schedule_df,
            vxn_scale_signal_df=vxn_scale_signal_df,
            regime_symbol_str=config_obj.regime_symbol_str,
            capital_base=config_obj.capital_base_float,
            slippage=config_obj.slippage_float,
            commission_per_share=config_obj.commission_per_share_float,
            commission_minimum=config_obj.commission_minimum_float,
            lookback_month_int=config_obj.lookback_month_int,
            index_trend_window_int=config_obj.index_trend_window_int,
            stock_trend_window_int=config_obj.stock_trend_window_int,
            max_positions_int=config_obj.max_positions_int,
        )
        strategy_obj.universe_df = universe_df
        configure_total_return_benchmark_provenance(
            strategy_obj=strategy_obj,
            config_obj=config_obj,
        )
        strategy_obj.trade_id_int = 0
        strategy_obj.current_trade_map = defaultdict(default_trade_id_int)
        return strategy_obj

    calendar_idx = pricing_data_df.index[
        pricing_data_df.index >= pd.Timestamp(config_obj.backtest_start_date_str)
    ]

    return {
        "strategy_factory_fn": strategy_factory_fn,
        "pricing_data_df": pricing_data_df,
        "calendar_idx": pd.DatetimeIndex(calendar_idx),
        "order_generation_mode_str": "signal_bar",
        "risk_model_str": "taa_rebalance",
        "entry_timing_str_tuple": ("same_close_moc", "next_open", "next_close"),
        "exit_timing_str_tuple": ("same_close_moc", "next_open", "next_close"),
        "default_entry_timing_str": "next_open",
        "default_exit_timing_str": "next_open",
    }


class Natr20VxnScaledNdxStrategy(Strategy):
    """
    Long-only monthly Nasdaq-100 momentum rotation with fixed slot sizing.

    For selected stock i at rebalance open t:

        q^{intent}_{i,t}
            = floor(V_{t-1} * vxn_scale_T / max_positions / RawClose_{i,t-1})

    The engine maps these nominal shares to adjusted economic share units.
    """

    enable_signal_audit = True
    signal_audit_sample_size = 10

    def __init__(
        self,
        name: str,
        benchmarks: Sequence[str],
        rebalance_schedule_df: pd.DataFrame,
        vxn_scale_signal_df: pd.DataFrame,
        regime_symbol_str: str = "SPY",
        capital_base: float = 100_000.0,
        slippage: float = 0.00025,
        commission_per_share: float = 0.005,
        commission_minimum: float = 1.0,
        lookback_month_int: int = 12,
        index_trend_window_int: int = 200,
        stock_trend_window_int: int = 100,
        max_positions_int: int = 10,
    ):
        super().__init__(
            name=name,
            benchmarks=list(benchmarks),
            capital_base=capital_base,
            slippage=slippage,
            commission_per_share=commission_per_share,
            commission_minimum=commission_minimum,
        )

        if len(rebalance_schedule_df) == 0:
            raise ValueError("rebalance_schedule_df must not be empty.")
        if "decision_date_ts" not in rebalance_schedule_df.columns:
            raise ValueError("rebalance_schedule_df must contain decision_date_ts.")
        if not regime_symbol_str:
            raise ValueError("regime_symbol_str must not be empty.")
        if lookback_month_int <= 0:
            raise ValueError("lookback_month_int must be positive.")
        if index_trend_window_int <= 0:
            raise ValueError("index_trend_window_int must be positive.")
        if stock_trend_window_int <= 0:
            raise ValueError("stock_trend_window_int must be positive.")
        if max_positions_int <= 0:
            raise ValueError("max_positions_int must be positive.")

        if vxn_scale_signal_df.empty or "vxn_exposure_scale_float" not in vxn_scale_signal_df:
            raise ValueError("VXN exposure scale data is required.")
        if not isinstance(vxn_scale_signal_df.index, pd.DatetimeIndex) or not vxn_scale_signal_df.index.is_unique:
            raise ValueError("VXN exposure scales require unique dated observations.")
        exposure_scale_ser = vxn_scale_signal_df["vxn_exposure_scale_float"].astype(float)
        if (~np.isfinite(exposure_scale_ser) | ~exposure_scale_ser.between(0.0, 1.0)).any():
            raise ValueError("VXN exposure scales must be finite and within [0, 1].")
        self.vxn_scale_signal_df = vxn_scale_signal_df.copy().sort_index()
        self.rebalance_schedule_df = rebalance_schedule_df.copy().sort_index()
        self.regime_symbol_str = str(regime_symbol_str)
        self.lookback_month_int = int(lookback_month_int)
        self.index_trend_window_int = int(index_trend_window_int)
        self.stock_trend_window_int = int(stock_trend_window_int)
        self.max_positions_int = int(max_positions_int)
        # Research execution must round and charge fees in historical nominal
        # share units. The live host consumes weights, not this backtest ledger.
        self.historical_share_units_bool = True
        self.trade_id_int = 0
        self.current_trade_map: defaultdict[str, int] = defaultdict(default_trade_id_int)
        self.universe_df: pd.DataFrame | None = None

    def get_tradeable_symbol_list(self, pricing_data_df: pd.DataFrame) -> list[str]:
        benchmark_data_symbol_set = set(
            self._benchmark_data_symbol_map_dict.values()
        )
        tradeable_symbol_list = [
            str(symbol_str)
            for symbol_str in pricing_data_df.columns.get_level_values(0).unique()
            if (
                str(symbol_str) not in self._benchmarks
                and str(symbol_str) != self.regime_symbol_str
                and str(symbol_str) not in benchmark_data_symbol_set
            )
        ]
        if len(tradeable_symbol_list) == 0:
            raise RuntimeError("No tradeable stock symbols were found in pricing_data_df.")
        return tradeable_symbol_list

    def compute_signals(self, pricing_data_df: pd.DataFrame) -> pd.DataFrame:
        signal_data_df = pricing_data_df.copy()
        tradeable_symbol_list = self.get_tradeable_symbol_list(signal_data_df)

        price_close_df = pd.DataFrame(
            {symbol_str: signal_data_df[(symbol_str, "Close")] for symbol_str in tradeable_symbol_list},
            index=signal_data_df.index,
        ).astype(float)
        price_high_df = pd.DataFrame(
            {symbol_str: signal_data_df[(symbol_str, "High")] for symbol_str in tradeable_symbol_list},
            index=signal_data_df.index,
        ).astype(float)
        price_low_df = pd.DataFrame(
            {symbol_str: signal_data_df[(symbol_str, "Low")] for symbol_str in tradeable_symbol_list},
            index=signal_data_df.index,
        ).astype(float)

        regime_close_key_tuple = (self.regime_symbol_str, "Close")
        if regime_close_key_tuple not in signal_data_df.columns:
            raise RuntimeError(f"Missing regime close data for {self.regime_symbol_str}.")
        regime_close_ser = signal_data_df[regime_close_key_tuple].astype(float)

        helper_config_obj = Natr20VxnScaledNdxConfig(
            regime_symbol_str=self.regime_symbol_str,
            lookback_month_int=self.lookback_month_int,
            index_trend_window_int=self.index_trend_window_int,
            stock_trend_window_int=self.stock_trend_window_int,
            max_positions_int=self.max_positions_int,
        )
        (
            _monthly_decision_close_df,
            monthly_roc_df,
            atr_decision_df,
            stock_trend_pass_df,
            regime_sma_ser,
            regime_pass_ser,
            risk_adj_score_df,
        ) = compute_natr20_signal_tables(
            price_close_df=price_close_df,
            price_high_df=price_high_df,
            price_low_df=price_low_df,
            regime_close_ser=regime_close_ser,
            config_obj=helper_config_obj,
            price_unadjusted_close_df=get_unadjusted_close_df(signal_data_df, tradeable_symbol_list),
        )

        monthly_roc_aligned_df = monthly_roc_df.reindex(signal_data_df.index)
        atr_aligned_df = atr_decision_df.reindex(signal_data_df.index)
        stock_trend_pass_aligned_df = stock_trend_pass_df.reindex(signal_data_df.index)
        risk_adj_score_aligned_df = risk_adj_score_df.reindex(signal_data_df.index)
        regime_sma_aligned_ser = regime_sma_ser.reindex(signal_data_df.index)
        regime_pass_aligned_ser = regime_pass_ser.reindex(signal_data_df.index)

        feature_frame_list: list[pd.DataFrame] = []
        feature_map_dict: dict[str, pd.DataFrame] = {
            f"monthly_roc_{self.lookback_month_int}_ser": monthly_roc_aligned_df,
            f"atr_{ATR_WINDOW_INT}_ser": atr_aligned_df,
            "stock_trend_pass_bool": stock_trend_pass_aligned_df,
            "risk_adj_score_ser": risk_adj_score_aligned_df,
        }

        for field_str, field_df in feature_map_dict.items():
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

        # *** CRITICAL *** NATR uses matching decision-date units only.
        # NATR20_pct_T = 100 * ATR_nominal_T / RawClose_T.
        raw_close_df = get_unadjusted_close_df(signal_data_df, tradeable_symbol_list)
        natr_feature_df = 100.0 * atr_aligned_df / raw_close_df
        natr_feature_df.columns = pd.MultiIndex.from_tuples(
            [(symbol_str, "natr_20_pct_ser") for symbol_str in natr_feature_df.columns]
        )
        feature_frame_list.append(natr_feature_df)
        self._data_adjustment_policy_dict.update({
            "ranking_formula_str": "ROC12 / (SMA20_TRUE_RANGE / Close)",
            "atr_smoothing_str": "simple_20_session_mean",
            "exposure_overlay_str": "clip(target_vxn_pct / asof_VXN_close, min_scale, max_scale)",
            "research_history_status_str": "previously_seen_2000_2026_not_untouched_holdout",
        })
        return pd.concat([signal_data_df] + feature_frame_list, axis=1)

    def get_ranked_candidate_feature_df(self, close_row_ser: pd.Series) -> pd.DataFrame:
        """
        Return all eligible candidates sorted by risk-adjusted score.

        Ordering is deterministic: risk_adj_score_float descending, then
        symbol_str ascending on ties. Returns an empty DataFrame when the
        regime gate fails or no candidate survives the filters.
        """
        if self.universe_df is None:
            raise RuntimeError("universe_df must be set before monthly rebalances.")
        candidate_feature_df = close_row_ser.unstack()
        if self.regime_symbol_str not in candidate_feature_df.index:
            raise RuntimeError(f"Missing regime feature row for {self.regime_symbol_str}.")

        empty_candidate_feature_df = pd.DataFrame(
            columns=["risk_adj_score_float", "stock_trend_pass_bool", "symbol_str"]
        )
        regime_pass_value_obj = candidate_feature_df.loc[self.regime_symbol_str].get("regime_pass_bool", np.nan)
        if pd.isna(regime_pass_value_obj) or not bool(regime_pass_value_obj):
            return empty_candidate_feature_df

        required_field_list = ["stock_trend_pass_bool", "risk_adj_score_ser"]
        if any(field_str not in candidate_feature_df.columns for field_str in required_field_list):
            return empty_candidate_feature_df

        universe_member_ser = get_asof_universe_membership_ser(
            self.universe_df,
            pd.Timestamp(self.previous_bar),
        )
        active_symbol_list = universe_member_ser[universe_member_ser == 1].index.astype(str).tolist()
        candidate_feature_df = candidate_feature_df[candidate_feature_df.index.isin(active_symbol_list)].copy()
        if len(candidate_feature_df) == 0:
            return empty_candidate_feature_df

        stock_trend_raw_ser = candidate_feature_df["stock_trend_pass_bool"]
        stock_trend_pass_ser = stock_trend_raw_ser.where(stock_trend_raw_ser.notna(), False).astype(bool)
        candidate_feature_df = candidate_feature_df.assign(
            risk_adj_score_float=pd.to_numeric(candidate_feature_df["risk_adj_score_ser"], errors="coerce"),
            stock_trend_pass_bool=stock_trend_pass_ser,
            symbol_str=candidate_feature_df.index.astype(str),
        )
        finite_risk_adj_mask_vec = np.isfinite(
            candidate_feature_df["risk_adj_score_float"].to_numpy(dtype=float)
        )
        stock_trend_pass_mask_vec = candidate_feature_df["stock_trend_pass_bool"].to_numpy(dtype=bool)
        candidate_feature_df = candidate_feature_df.loc[
            finite_risk_adj_mask_vec & stock_trend_pass_mask_vec
        ]
        if len(candidate_feature_df) == 0:
            return empty_candidate_feature_df

        return candidate_feature_df.sort_values(
            by=["risk_adj_score_float", "symbol_str"],
            ascending=[False, True],
            kind="mergesort",
        )

    def get_target_weight_ser(self, close_row_ser: pd.Series) -> pd.Series:
        ranked_candidate_feature_df = self.get_ranked_candidate_feature_df(close_row_ser=close_row_ser)
        if len(ranked_candidate_feature_df) == 0:
            return pd.Series(dtype=float)

        selected_feature_df = ranked_candidate_feature_df.iloc[: self.max_positions_int].copy()

        target_weight_float = 1.0 / float(self.max_positions_int)
        target_weight_ser = pd.Series(
            target_weight_float,
            index=selected_feature_df.index,
            dtype=float,
        )
        # *** CRITICAL *** At Close_T select the latest VXN observation <= T;
        # do not read the execution-day or terminal dataset VXN value.
        exposure_scale_float = get_asof_vxn_scale_float(
            self.vxn_scale_signal_df, pd.Timestamp(self.previous_bar),
        )
        return target_weight_ser * exposure_scale_float

    def get_target_share_int_map(
        self,
        target_weight_ser: pd.Series,
        close_row_ser: pd.Series,
    ) -> dict[str, float]:
        target_share_amount_dict: dict[str, float] = {}
        if len(target_weight_ser) == 0:
            return target_share_amount_dict

        budget_value_float = float(self.previous_total_value)
        for symbol_str, target_weight_float in target_weight_ser.items():
            close_price_float = float(close_row_ser[(symbol_str, "Close")])
            if not np.isfinite(close_price_float) or close_price_float <= 0.0:
                raise RuntimeError(f"Invalid prior close for target asset {symbol_str} on {self.previous_bar}.")

            # *** CRITICAL*** Monthly target shares must be fixed from the
            # previous_bar close so the rebalance does not adapt to the
            # realized current-bar open.
            if self.historical_share_units_bool:
                target_share_amount_float = self.historical_share_amount_float(
                    budget_value_float * float(target_weight_float),
                    close_price_float,
                    float(close_row_ser[(symbol_str, "Unadjusted Close")]),
                )
            else:
                target_share_amount_float = int(budget_value_float * float(target_weight_float) / close_price_float)
            if target_share_amount_float > 0:
                target_share_amount_dict[str(symbol_str)] = target_share_amount_float

        return target_share_amount_dict

    def iterate(self, data_df: pd.DataFrame, close_ser: pd.Series, open_price_ser: pd.Series):
        if close_ser is None or data_df is None:
            return
        if self.current_bar not in self.rebalance_schedule_df.index:
            return

        decision_date_ts = pd.Timestamp(self.rebalance_schedule_df.loc[self.current_bar, "decision_date_ts"])
        # *** CRITICAL*** The scheduled month-end decision close_ser must equal
        # previous_bar exactly, otherwise signals and next-open execution drift.
        if pd.Timestamp(self.previous_bar) != decision_date_ts:
            raise RuntimeError(
                f"Schedule misalignment on {self.current_bar}: "
                f"decision_date_ts={decision_date_ts}, previous_bar={self.previous_bar}."
            )

        target_weight_ser = self.get_target_weight_ser(close_row_ser=close_ser)
        target_share_amount_dict = self.get_target_share_int_map(
            target_weight_ser=target_weight_ser,
            close_row_ser=close_ser,
        )
        target_symbol_set = set(target_share_amount_dict)

        current_position_ser = self.get_positions()
        long_position_ser = current_position_ser[current_position_ser > 0]
        for symbol_str in long_position_ser.index:
            if symbol_str in target_symbol_set:
                continue
            self.order_target_value(
                symbol_str,
                0.0,
                trade_id=self.current_trade_map[symbol_str],
            )

        for symbol_str, target_share_amount_float in target_share_amount_dict.items():
            current_share_amount_float = float(current_position_ser.get(symbol_str, 0.0))
            if current_share_amount_float == target_share_amount_float:
                continue

            if current_share_amount_float == 0:
                self.trade_id_int += 1
                self.current_trade_map[symbol_str] = self.trade_id_int

            target_weight_float = float(target_weight_ser.loc[symbol_str])
            self.order_target_percent(
                symbol_str,
                target_weight_float,
                trade_id=self.current_trade_map[symbol_str],
            )


def run_variant(
    show_display_bool: bool = True,
    save_results_bool: bool = True,
    output_dir_str: str = "results",
    backtest_start_date_str: str | None = None,
    capital_base_float: float | None = None,
    end_date_str: str | None = None,
) -> Natr20VxnScaledNdxStrategy:
    config_obj = DEFAULT_CONFIG
    if (
        backtest_start_date_str is not None
        or capital_base_float is not None
        or end_date_str is not None
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
        )
    pricing_data_df, universe_df, rebalance_schedule_df, vxn_scale_signal_df = get_natr20_vxn_scaled_ndx_data(
        config_obj,
        include_total_return_benchmark_bool=True,
    )

    strategy_obj = Natr20VxnScaledNdxStrategy(
        name="strategy_mo_natr20_ndx_vxn_scaled",
        benchmarks=[config_obj.performance_benchmark_symbol_str],
        rebalance_schedule_df=rebalance_schedule_df,
        vxn_scale_signal_df=vxn_scale_signal_df,
        regime_symbol_str=config_obj.regime_symbol_str,
        capital_base=config_obj.capital_base_float,
        slippage=config_obj.slippage_float,
        commission_per_share=config_obj.commission_per_share_float,
        commission_minimum=config_obj.commission_minimum_float,
        lookback_month_int=config_obj.lookback_month_int,
        index_trend_window_int=config_obj.index_trend_window_int,
        stock_trend_window_int=config_obj.stock_trend_window_int,
        max_positions_int=config_obj.max_positions_int,
    )
    strategy_obj.universe_df = universe_df
    configure_total_return_benchmark_provenance(
        strategy_obj=strategy_obj,
        config_obj=config_obj,
    )

    # *** CRITICAL*** Deployment-reference backtests keep full pre-start
    # history for monthly ATR and trend features, but the executable calendar
    # starts at the first deployment fill session.
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


def build_capacity_analysis_inputs(
    show_display_bool: bool = False,
    backtest_start_date_str: str | None = None,
    capital_base_float: float | None = None,
    end_date_str: str | None = None,
) -> dict[str, object]:
    config_obj = DEFAULT_CONFIG
    if (
        backtest_start_date_str is not None
        or capital_base_float is not None
        or end_date_str is not None
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
        )
    pricing_data_df, universe_df, rebalance_schedule_df, vxn_scale_signal_df = get_natr20_vxn_scaled_ndx_data(
        config_obj,
        include_total_return_benchmark_bool=True,
    )

    strategy_obj = Natr20VxnScaledNdxStrategy(
        name="strategy_mo_natr20_ndx_vxn_scaled",
        benchmarks=[config_obj.performance_benchmark_symbol_str],
        rebalance_schedule_df=rebalance_schedule_df,
        vxn_scale_signal_df=vxn_scale_signal_df,
        regime_symbol_str=config_obj.regime_symbol_str,
        capital_base=config_obj.capital_base_float,
        slippage=config_obj.slippage_float,
        commission_per_share=config_obj.commission_per_share_float,
        commission_minimum=config_obj.commission_minimum_float,
        lookback_month_int=config_obj.lookback_month_int,
        index_trend_window_int=config_obj.index_trend_window_int,
        stock_trend_window_int=config_obj.stock_trend_window_int,
        max_positions_int=config_obj.max_positions_int,
    )
    strategy_obj.universe_df = universe_df
    configure_total_return_benchmark_provenance(
        strategy_obj=strategy_obj,
        config_obj=config_obj,
    )

    # *** CRITICAL *** CapacityAnalysis must assess the same completed order
    # ledger as the deployment-reference NDX momentum backtest. Keep pre-start
    # history for monthly features, but execute only on the configured calendar.
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

    strategy_obj.universe_df = None
    return {
        "strategy_obj": strategy_obj,
        "pricing_data_df": pricing_data_df,
        "execution_policy_str": "MOO",
        "impact_profile_str": "MOO_NASDAQ_LARGE",
    }


if __name__ == "__main__":
    run_variant()
