"""Inflation Compass monthly tactical allocation strategy.

The strategy classifies each month-end into one of four regimes:

    growth_up_T = 1[SPY_T > SMA200_T]

    inflation_on_T
        = 1[T5YIE_T > 2.0]
          AND (
              1[T5YIE_T > T5YIE_(T-60 sessions)]
              OR 1[OLS_slope_60(asset_ratio)_T > 0]
          )

where ``asset_ratio`` is the cumulative wealth of the inflation-positive
basket divided by the cumulative wealth of the inflation-negative basket:

    positive_return_t = 0.50*XLE_t + (XLI_t + XLF_t + XLB_t)/6
    negative_return_t = (XLU_t + XLV_t + XLP_t)/3

The regime map is literal:

    growth up, inflation on   -> 100% XLE
    growth up, inflation off  -> 100% XLK
    growth down, inflation on -> 100% XLU
    growth down, inflation off -> 50% XLP + 50% IEF

Every decision uses final month-end Close_T information. The existing TAA
engine sizes from that close and fills at the first Open_(T+1). The old sleeve
remains invested through the overnight interval; there is no month-end cash
gap and no target-weight leverage.

The share-level Vanilla engine does not cash-constrain opening fills. A gap
against a 100% target can therefore create realized negative cash even though
the target book itself never exceeds 100%. Financing is not modeled; this is
an explicit research gap and blocks PAPER/LIVE interpretation.

Data roles are deliberately separate:

1. Signal closes use TOTALRETURN adjustment.
2. ETF fills and marks use CAPITALSPECIAL adjustment.
3. The benchmark uses the explicit total-return index series.
4. T5YIE uses the shared FRED loader and a current-vintage local cache.

The FRED series is not an ALFRED vintage archive. This module is PM_READY
research plumbing only and is deliberately absent from LIVE release wiring.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from IPython.display import display

from alpha.data import FredSeriesSnapshot, load_daily_fred_series_snapshot
from alpha.engine.backtest import run_daily
from alpha.engine.report import save_results
from strategies.taa_df.strategy_taa_df import (
    DefenseFirstStrategy,
    load_execution_price_df,
    load_signal_close_df,
    map_month_end_weights_to_rebalance_open_df,
)


STRATEGY_NAME_STR = "strategy_taa_inflation_compass"
SIGNAL_ASSET_TUPLE = ("SPY", "XLE", "XLI", "XLF", "XLB", "XLU", "XLV", "XLP")
TRADEABLE_ASSET_TUPLE = ("XLE", "XLK", "XLU", "XLP", "IEF")
BENCHMARK_TUPLE = ("$SPX",)

POSITIVE_BASKET_WEIGHT_DICT = {
    "XLE": 0.50,
    "XLI": 1.0 / 6.0,
    "XLF": 1.0 / 6.0,
    "XLB": 1.0 / 6.0,
}
NEGATIVE_BASKET_WEIGHT_DICT = {
    "XLU": 1.0 / 3.0,
    "XLV": 1.0 / 3.0,
    "XLP": 1.0 / 3.0,
}

GROWTH_SMA_SESSION_INT = 200
BREAKEVEN_LOOKBACK_SESSION_INT = 60
ASSET_SLOPE_LOOKBACK_SESSION_INT = 60
INFLATION_THRESHOLD_FLOAT = 2.0
FRED_ALIGNMENT_TOLERANCE_DAY_INT = 7

# Five basis points on each executed side approximates the source's declared
# ten-basis-point full switch. Per-share commission is disabled so the cost
# contract does not change with nominal ETF share price.
SLIPPAGE_PER_SIDE_FLOAT = 0.0005
COMMISSION_PER_SHARE_FLOAT = 0.0
COMMISSION_MINIMUM_FLOAT = 0.0

REPO_ROOT_PATH = Path(__file__).resolve().parents[2]
DEFAULT_T5YIE_CSV_PATH = REPO_ROOT_PATH.parent / "1_data" / "T5YIE.csv"


@dataclass(frozen=True)
class InflationCompassConfig:
    signal_asset_tuple: tuple[str, ...] = SIGNAL_ASSET_TUPLE
    tradeable_asset_tuple: tuple[str, ...] = TRADEABLE_ASSET_TUPLE
    benchmark_tuple: tuple[str, ...] = BENCHMARK_TUPLE
    growth_sma_session_int: int = GROWTH_SMA_SESSION_INT
    breakeven_lookback_session_int: int = BREAKEVEN_LOOKBACK_SESSION_INT
    asset_slope_lookback_session_int: int = ASSET_SLOPE_LOOKBACK_SESSION_INT
    inflation_threshold_float: float = INFLATION_THRESHOLD_FLOAT
    start_date_str: str = "2002-01-01"
    end_date_str: str | None = None
    t5yie_csv_path_str: str = str(DEFAULT_T5YIE_CSV_PATH)
    t5yie_series_id_str: str = "T5YIE"
    t5yie_mode_str: str = "backtest"
    t5yie_as_of_timestamp_ts: datetime | None = None
    capital_base_float: float = 100_000.0
    slippage_per_side_float: float = SLIPPAGE_PER_SIDE_FLOAT
    commission_per_share_float: float = COMMISSION_PER_SHARE_FLOAT
    commission_minimum_float: float = COMMISSION_MINIMUM_FLOAT

    def __post_init__(self) -> None:
        if self.growth_sma_session_int < 2:
            raise ValueError("growth_sma_session_int must be at least two sessions.")
        if self.breakeven_lookback_session_int < 1:
            raise ValueError("breakeven_lookback_session_int must be positive.")
        if self.asset_slope_lookback_session_int < 2:
            raise ValueError("asset_slope_lookback_session_int must be at least two sessions.")
        if len(set(self.signal_asset_tuple)) != len(self.signal_asset_tuple):
            raise ValueError("signal_asset_tuple contains duplicate symbols.")
        if len(set(self.tradeable_asset_tuple)) != len(self.tradeable_asset_tuple):
            raise ValueError("tradeable_asset_tuple contains duplicate symbols.")
        if not np.isclose(sum(POSITIVE_BASKET_WEIGHT_DICT.values()), 1.0, atol=1e-12):
            raise ValueError("Positive basket weights must sum to one.")
        if not np.isclose(sum(NEGATIVE_BASKET_WEIGHT_DICT.values()), 1.0, atol=1e-12):
            raise ValueError("Negative basket weights must sum to one.")


DEFAULT_CONFIG = InflationCompassConfig()
InflationCompassDataTuple = tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    FredSeriesSnapshot,
]


def _resolve_t5yie_as_of_timestamp_ts(config_obj: InflationCompassConfig) -> datetime:
    if config_obj.t5yie_as_of_timestamp_ts is not None:
        return config_obj.t5yie_as_of_timestamp_ts
    if config_obj.end_date_str is not None:
        return pd.Timestamp(config_obj.end_date_str).to_pydatetime()
    return datetime.now(tz=UTC)


def load_t5yie_snapshot(config_obj: InflationCompassConfig) -> FredSeriesSnapshot:
    """Load current-vintage daily T5YIE with the shared FRED audit metadata."""
    return load_daily_fred_series_snapshot(
        series_id_str=config_obj.t5yie_series_id_str,
        cache_csv_path_str=config_obj.t5yie_csv_path_str,
        as_of_ts=_resolve_t5yie_as_of_timestamp_ts(config_obj),
        mode_str=config_obj.t5yie_mode_str,
    )


def align_fred_to_session_ser(
    fred_value_ser: pd.Series,
    session_date_index: pd.DatetimeIndex,
    tolerance_day_int: int = FRED_ALIGNMENT_TOLERANCE_DAY_INT,
) -> tuple[pd.Series, pd.Series]:
    """Backward as-of alignment; a session never sees a later FRED date."""
    session_date_index = pd.DatetimeIndex(session_date_index).tz_localize(None).normalize()
    session_df = pd.DataFrame({"session_date": session_date_index}).sort_values("session_date")
    fred_df = fred_value_ser.dropna().rename("fred_value_float").reset_index()
    fred_df.columns = ["observation_date", "fred_value_float"]
    fred_df["observation_date"] = pd.to_datetime(fred_df["observation_date"]).dt.normalize()
    fred_df = fred_df.sort_values("observation_date")

    # *** CRITICAL*** lookahead-sensitive: direction='backward' and exact
    # matches are mandatory. Date-T T5YIE is treated as known after Close_T,
    # while any observation dated after T is forbidden.
    aligned_df = pd.merge_asof(
        session_df,
        fred_df,
        left_on="session_date",
        right_on="observation_date",
        direction="backward",
        tolerance=pd.Timedelta(days=tolerance_day_int),
        allow_exact_matches=True,
    ).set_index("session_date")

    future_observation_bool_ser = aligned_df["observation_date"].notna() & aligned_df[
        "observation_date"
    ].gt(aligned_df.index.to_series())
    if future_observation_bool_ser.any():
        raise AssertionError("FRED as-of alignment selected a future observation.")

    aligned_value_ser = aligned_df["fred_value_float"].astype(float)
    aligned_value_ser.name = str(fred_value_ser.name or "T5YIE")
    observation_age_day_ser = (
        aligned_df.index.to_series().sub(aligned_df["observation_date"]).dt.days.astype("Float64")
    )
    observation_age_day_ser.name = "t5yie_observation_age_day_float"
    return aligned_value_ser, observation_age_day_ser


def compute_rolling_ols_slope_ser(
    value_ser: pd.Series,
    lookback_session_int: int,
) -> pd.Series:
    """OLS slope over trailing values ``T-L+1, ..., T``."""
    if lookback_session_int < 2:
        raise ValueError("OLS slope lookback must be at least two sessions.")
    time_vec = np.arange(lookback_session_int, dtype=float)

    def slope_float(window_vec: np.ndarray) -> float:
        if not np.isfinite(window_vec).all():
            return float("nan")
        return float(np.polyfit(time_vec, window_vec, deg=1)[0])

    # *** CRITICAL*** lookahead-sensitive: the rolling window ends at T and
    # is never centered. Every value in the regression is known by Close_T.
    slope_ser = value_ser.rolling(
        window=lookback_session_int,
        min_periods=lookback_session_int,
    ).apply(slope_float, raw=True)
    slope_ser.name = f"asset_ratio_ols_slope_{lookback_session_int}_float"
    return slope_ser


def get_month_end_session_index(session_date_index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    normalized_date_index = pd.DatetimeIndex(session_date_index).tz_localize(None).normalize()
    session_date_ser = pd.Series(normalized_date_index, index=normalized_date_index)
    # *** CRITICAL*** Month-end sampling selects the last observed trading
    # session in each month, never the calendar month-end or a future session.
    month_end_session_ser = session_date_ser.groupby(normalized_date_index.to_period("M")).max()
    return pd.DatetimeIndex(month_end_session_ser.to_numpy()).sort_values()


def _regime_target_weight_ser(
    growth_on_bool: bool,
    inflation_on_bool: bool,
) -> tuple[str, pd.Series]:
    target_weight_ser = pd.Series(0.0, index=list(TRADEABLE_ASSET_TUPLE), dtype=float)
    if growth_on_bool and inflation_on_bool:
        regime_label_str = "growth_up__inflation_on"
        target_weight_ser.loc["XLE"] = 1.0
    elif growth_on_bool and not inflation_on_bool:
        regime_label_str = "growth_up__inflation_off"
        target_weight_ser.loc["XLK"] = 1.0
    elif not growth_on_bool and inflation_on_bool:
        regime_label_str = "growth_down__inflation_on"
        target_weight_ser.loc["XLU"] = 1.0
    else:
        regime_label_str = "growth_down__inflation_off"
        target_weight_ser.loc[["XLP", "IEF"]] = 0.5
    return regime_label_str, target_weight_ser


def compute_month_end_signal_and_weight_df(
    signal_close_df: pd.DataFrame,
    t5yie_value_ser: pd.Series,
    config_obj: InflationCompassConfig = DEFAULT_CONFIG,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute the literal daily features and month-end target weights."""
    missing_signal_asset_list = [
        asset_str for asset_str in config_obj.signal_asset_tuple if asset_str not in signal_close_df
    ]
    if missing_signal_asset_list:
        raise RuntimeError(f"Missing signal closes for {missing_signal_asset_list}.")

    signal_close_df = signal_close_df.loc[:, list(config_obj.signal_asset_tuple)].astype(float)
    session_date_index = pd.DatetimeIndex(signal_close_df.index)
    aligned_t5yie_ser, t5yie_age_day_ser = align_fred_to_session_ser(
        fred_value_ser=t5yie_value_ser,
        session_date_index=session_date_index,
    )

    signal_return_df = signal_close_df.pct_change(fill_method=None)
    positive_basket_return_ser = signal_return_df[
        list(POSITIVE_BASKET_WEIGHT_DICT)
    ].mul(pd.Series(POSITIVE_BASKET_WEIGHT_DICT)).sum(
        axis=1,
        min_count=len(POSITIVE_BASKET_WEIGHT_DICT),
    )
    negative_basket_return_ser = signal_return_df[
        list(NEGATIVE_BASKET_WEIGHT_DICT)
    ].mul(pd.Series(NEGATIVE_BASKET_WEIGHT_DICT)).sum(
        axis=1,
        min_count=len(NEGATIVE_BASKET_WEIGHT_DICT),
    )
    positive_basket_wealth_ser = positive_basket_return_ser.add(1.0).cumprod()
    negative_basket_wealth_ser = negative_basket_return_ser.add(1.0).cumprod()
    asset_ratio_ser = positive_basket_wealth_ser.div(negative_basket_wealth_ser)
    asset_slope_ser = compute_rolling_ols_slope_ser(
        value_ser=asset_ratio_ser,
        lookback_session_int=config_obj.asset_slope_lookback_session_int,
    )

    # *** CRITICAL*** lookahead-sensitive: the SMA window includes Close_T and
    # ends at Close_T. Its decision is not executable before Open_(T+1).
    growth_sma_ser = signal_close_df["SPY"].rolling(
        window=config_obj.growth_sma_session_int,
        min_periods=config_obj.growth_sma_session_int,
    ).mean()
    growth_on_ser = signal_close_df["SPY"].gt(growth_sma_ser)
    inflation_level_on_ser = aligned_t5yie_ser.gt(config_obj.inflation_threshold_float)

    # *** CRITICAL*** lookahead-sensitive: shift(+60) reads T-60 sessions.
    # A negative shift would leak future T5YIE observations into Close_T.
    prior_t5yie_ser = aligned_t5yie_ser.shift(config_obj.breakeven_lookback_session_int)
    breakeven_up_ser = aligned_t5yie_ser.gt(prior_t5yie_ser)
    asset_up_ser = asset_slope_ser.gt(0.0)
    inflation_on_ser = inflation_level_on_ser & (breakeven_up_ser | asset_up_ser)

    daily_feature_df = pd.DataFrame(
        {
            "spy_close_float": signal_close_df["SPY"],
            "growth_sma_float": growth_sma_ser,
            "growth_on_bool": growth_on_ser,
            "t5yie_float": aligned_t5yie_ser,
            "t5yie_observation_age_day_float": t5yie_age_day_ser,
            "t5yie_prior_float": prior_t5yie_ser,
            "inflation_level_on_bool": inflation_level_on_ser,
            "breakeven_up_bool": breakeven_up_ser,
            "positive_basket_return_float": positive_basket_return_ser,
            "negative_basket_return_float": negative_basket_return_ser,
            "asset_ratio_float": asset_ratio_ser,
            "asset_slope_float": asset_slope_ser,
            "asset_up_bool": asset_up_ser,
            "inflation_on_bool": inflation_on_ser,
        },
        index=session_date_index,
    )
    month_end_session_index = get_month_end_session_index(session_date_index)
    required_feature_column_list = [
        "growth_sma_float",
        "t5yie_float",
        "t5yie_prior_float",
        "asset_ratio_float",
        "asset_slope_float",
    ]
    sampled_month_end_feature_df = daily_feature_df.reindex(month_end_session_index)
    complete_month_end_bool_ser = sampled_month_end_feature_df[
        required_feature_column_list
    ].notna().all(axis=1)
    if complete_month_end_bool_ser.any():
        first_complete_month_end_ts = pd.Timestamp(
            complete_month_end_bool_ser[complete_month_end_bool_ser].index[0]
        )
        incomplete_after_warmup_index = sampled_month_end_feature_df.loc[
            first_complete_month_end_ts:
        ].index[
            ~complete_month_end_bool_ser.loc[first_complete_month_end_ts:]
        ]
        if len(incomplete_after_warmup_index) > 0:
            missing_month_str = ", ".join(
                pd.Timestamp(month_end_ts).strftime("%Y-%m-%d")
                for month_end_ts in incomplete_after_warmup_index[:5]
            )
            raise RuntimeError(
                "Incomplete Inflation Compass month-end signal after warmup; "
                "refusing to silently hold the prior sleeve. Missing dates: "
                f"{missing_month_str}."
            )

    month_end_feature_df = sampled_month_end_feature_df.loc[
        complete_month_end_bool_ser
    ].copy()
    if len(month_end_feature_df) == 0:
        raise RuntimeError("No complete Inflation Compass month-end signals were generated.")

    target_record_list: list[dict[str, object]] = []
    for decision_date_ts, feature_row_ser in month_end_feature_df.iterrows():
        regime_label_str, target_weight_ser = _regime_target_weight_ser(
            growth_on_bool=bool(feature_row_ser["growth_on_bool"]),
            inflation_on_bool=bool(feature_row_ser["inflation_on_bool"]),
        )
        target_record_dict: dict[str, object] = {
            "decision_date": pd.Timestamp(decision_date_ts),
            "regime_label_str": regime_label_str,
        }
        target_record_dict.update(target_weight_ser.to_dict())
        target_record_list.append(target_record_dict)

    target_df = pd.DataFrame(target_record_list).set_index("decision_date").sort_index()
    month_end_weight_df = target_df.loc[:, list(config_obj.tradeable_asset_tuple)].astype(float)
    target_weight_sum_ser = month_end_weight_df.sum(axis=1)
    if not np.allclose(target_weight_sum_ser.to_numpy(dtype=float), 1.0, atol=1e-12):
        raise ValueError("Every Inflation Compass target must sum to one.")
    month_end_feature_df = month_end_feature_df.join(target_df[["regime_label_str"]])
    return month_end_feature_df, month_end_weight_df


def get_inflation_compass_data(
    config_obj: InflationCompassConfig = DEFAULT_CONFIG,
) -> InflationCompassDataTuple:
    signal_close_df = load_signal_close_df(
        symbol_list=config_obj.signal_asset_tuple,
        start_date_str=config_obj.start_date_str,
        end_date_str=config_obj.end_date_str,
    )
    execution_price_df = load_execution_price_df(
        tradeable_asset_list=config_obj.tradeable_asset_tuple,
        benchmark_list=config_obj.benchmark_tuple,
        start_date_str=config_obj.start_date_str,
        end_date_str=config_obj.end_date_str,
    )
    t5yie_snapshot_obj = load_t5yie_snapshot(config_obj)
    month_end_feature_df, month_end_weight_df = compute_month_end_signal_and_weight_df(
        signal_close_df=signal_close_df,
        t5yie_value_ser=t5yie_snapshot_obj.value_ser,
        config_obj=config_obj,
    )
    rebalance_weight_df = map_month_end_weights_to_rebalance_open_df(
        month_end_weight_df=month_end_weight_df,
        execution_index=execution_price_df.index,
    )
    return (
        execution_price_df,
        month_end_feature_df,
        month_end_weight_df,
        rebalance_weight_df,
        t5yie_snapshot_obj,
    )


class InflationCompassStrategy(DefenseFirstStrategy):
    """Monthly four-regime allocator using the existing TAA order contract."""


class InflationCompassTimingStrategy(InflationCompassStrategy):
    """Timing adapter that preserves Vanilla dividend and sizing semantics."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.timing_pricing_data_df: pd.DataFrame | None = None

    def iterate(
        self,
        data_df: pd.DataFrame,
        close_row_ser: pd.Series,
        open_price_ser: pd.Series,
    ) -> None:
        if self.timing_pricing_data_df is None:
            raise RuntimeError("Timing strategy requires the complete pricing_data_df.")
        # *** CRITICAL*** Match Vanilla dividend entitlement exactly: shares
        # held at Close_T receive Dividend_T before Open_(T+1). The engine
        # method preserves withholding and duplicate-post guards.
        self._credit_dividend_cash_before_open(self.timing_pricing_data_df)

        if self.current_bar in self.rebalance_weight_df.index:
            current_ex_date_ts = pd.Timestamp(self.current_bar)
            current_dividend_cash_float = float(
                sum(
                    float(dividend_row_dict["net_dividend_cash_float"])
                    for dividend_row_dict in self._dividend_ledger_row_dict_list
                    if pd.Timestamp(dividend_row_dict["ex_date"])
                    == current_ex_date_ts
                )
            )
            # *** CRITICAL*** Vanilla credits Close_T dividend entitlement to
            # cash before Open_(T+1), but target-percent sizing remains anchored
            # to V_close_T recorded before that cash posting. The timing engine
            # otherwise sizes from cash-plus-positions after the posting and can
            # buy extra shares on a dividend/rebalance coincidence.
            vanilla_budget_value_float = float(
                self.total_value - current_dividend_cash_float
            )
            self.total_value = vanilla_budget_value_float
            self._total_value_history_list = [vanilla_budget_value_float]

        super().iterate(data_df, close_row_ser, open_price_ser)


def _attach_t5yie_provenance(
    strategy_obj: InflationCompassStrategy,
    t5yie_snapshot_obj: FredSeriesSnapshot,
) -> None:
    strategy_obj.t5yie_snapshot_obj = t5yie_snapshot_obj
    strategy_obj._data_adjustment_policy_dict["fred_series_provenance_dict"] = {
        "source_name_str": str(t5yie_snapshot_obj.source_name_str),
        "series_id_str": str(t5yie_snapshot_obj.series_id_str),
        "download_attempt_timestamp_str": (
            t5yie_snapshot_obj.download_attempt_timestamp_ts.isoformat()
        ),
        "download_status_str": str(t5yie_snapshot_obj.download_status_str),
        "latest_observation_date_str": pd.Timestamp(
            t5yie_snapshot_obj.latest_observation_date_ts
        ).date().isoformat(),
        "used_cache_bool": bool(t5yie_snapshot_obj.used_cache_bool),
        "freshness_business_days_int": int(
            t5yie_snapshot_obj.freshness_business_days_int
        ),
        "vintage_policy_str": "current_vintage_not_alfred",
    }


def _build_strategy_obj(
    config_obj: InflationCompassConfig,
    rebalance_weight_df: pd.DataFrame,
    strategy_class_obj: type[InflationCompassStrategy] = InflationCompassStrategy,
) -> InflationCompassStrategy:
    return strategy_class_obj(
        name=STRATEGY_NAME_STR,
        benchmarks=config_obj.benchmark_tuple,
        rebalance_weight_df=rebalance_weight_df,
        tradeable_asset_list=config_obj.tradeable_asset_tuple,
        capital_base=config_obj.capital_base_float,
        slippage=config_obj.slippage_per_side_float,
        commission_per_share=config_obj.commission_per_share_float,
        commission_minimum=config_obj.commission_minimum_float,
    )


def _execution_calendar_index(
    execution_price_df: pd.DataFrame,
    rebalance_weight_df: pd.DataFrame,
    backtest_start_date_str: str | None,
) -> pd.DatetimeIndex:
    calendar_start_ts = pd.Timestamp(rebalance_weight_df.index[0])
    if backtest_start_date_str is not None:
        calendar_start_ts = max(calendar_start_ts, pd.Timestamp(backtest_start_date_str))
    # *** CRITICAL*** Signal warmup remains loaded before this boundary. Only
    # executable bars are clipped, so the first order still uses Close_T data.
    return pd.DatetimeIndex(execution_price_df.index[execution_price_df.index >= calendar_start_ts])


def _run_inflation_compass_strategy(
    config_obj: InflationCompassConfig,
    execution_price_df: pd.DataFrame,
    month_end_feature_df: pd.DataFrame,
    month_end_weight_df: pd.DataFrame,
    rebalance_weight_df: pd.DataFrame,
    t5yie_snapshot_obj: FredSeriesSnapshot,
    backtest_start_date_str: str | None,
    show_progress_bool: bool,
) -> InflationCompassStrategy:
    strategy_obj = _build_strategy_obj(
        config_obj=config_obj,
        rebalance_weight_df=rebalance_weight_df,
    )
    strategy_obj.show_taa_weights_report = True
    strategy_obj.month_end_feature_df = month_end_feature_df.copy()
    strategy_obj.month_end_weight_df = month_end_weight_df.copy()
    _attach_t5yie_provenance(strategy_obj, t5yie_snapshot_obj)

    # *** CRITICAL*** Forward fill is report-only. Orders still use only the
    # discrete next-month open rows in rebalance_weight_df inside iterate().
    strategy_obj.daily_target_weights = (
        rebalance_weight_df.reindex(execution_price_df.index).ffill().dropna()
    )
    calendar_index = _execution_calendar_index(
        execution_price_df=execution_price_df,
        rebalance_weight_df=rebalance_weight_df,
        backtest_start_date_str=backtest_start_date_str,
    )
    run_daily(
        strategy_obj,
        execution_price_df,
        calendar=calendar_index,
        show_progress=show_progress_bool,
        show_signal_progress_bool=show_progress_bool,
        audit_override_bool=None,
    )
    return strategy_obj


def run_variant(
    show_display_bool: bool = True,
    save_results_bool: bool = True,
    output_dir_str: str = "results",
    backtest_start_date_str: str | None = None,
    capital_base_float: float = DEFAULT_CONFIG.capital_base_float,
    end_date_str: str | None = None,
    config_obj: InflationCompassConfig = DEFAULT_CONFIG,
) -> InflationCompassStrategy:
    config_obj = replace(
        config_obj,
        capital_base_float=capital_base_float,
        end_date_str=end_date_str,
    )
    (
        execution_price_df,
        month_end_feature_df,
        month_end_weight_df,
        rebalance_weight_df,
        t5yie_snapshot_obj,
    ) = get_inflation_compass_data(config_obj)
    strategy_obj = _run_inflation_compass_strategy(
        config_obj=config_obj,
        execution_price_df=execution_price_df,
        month_end_feature_df=month_end_feature_df,
        month_end_weight_df=month_end_weight_df,
        rebalance_weight_df=rebalance_weight_df,
        t5yie_snapshot_obj=t5yie_snapshot_obj,
        backtest_start_date_str=backtest_start_date_str,
        show_progress_bool=show_display_bool,
    )

    if show_display_bool:
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 1000)
        display(month_end_feature_df.head())
        display(strategy_obj.summary)
        display(strategy_obj.summary_trades)

    if save_results_bool:
        save_results(strategy_obj, output_dir=output_dir_str)
    return strategy_obj


def build_capacity_analysis_inputs(
    show_display_bool: bool = False,
    backtest_start_date_str: str | None = None,
    capital_base_float: float = DEFAULT_CONFIG.capital_base_float,
    end_date_str: str | None = None,
) -> dict[str, object]:
    config_obj = replace(
        DEFAULT_CONFIG,
        capital_base_float=capital_base_float,
        end_date_str=end_date_str,
    )
    (
        execution_price_df,
        month_end_feature_df,
        month_end_weight_df,
        rebalance_weight_df,
        t5yie_snapshot_obj,
    ) = get_inflation_compass_data(config_obj)
    strategy_obj = _run_inflation_compass_strategy(
        config_obj=config_obj,
        execution_price_df=execution_price_df,
        month_end_feature_df=month_end_feature_df,
        month_end_weight_df=month_end_weight_df,
        rebalance_weight_df=rebalance_weight_df,
        t5yie_snapshot_obj=t5yie_snapshot_obj,
        backtest_start_date_str=backtest_start_date_str,
        show_progress_bool=show_display_bool,
    )
    strategy_obj._performance_benchmark_symbol_str = str(config_obj.benchmark_tuple[0])
    return {
        "strategy_obj": strategy_obj,
        "pricing_data_df": execution_price_df,
        "execution_policy_str": "MOO",
        "impact_profile_str": "MOO_ETF_PROXY",
    }


def build_execution_timing_analysis_inputs() -> dict[str, object]:
    config_obj = DEFAULT_CONFIG
    (
        execution_price_df,
        month_end_feature_df,
        month_end_weight_df,
        rebalance_weight_df,
        t5yie_snapshot_obj,
    ) = get_inflation_compass_data(config_obj)
    calendar_index = _execution_calendar_index(
        execution_price_df=execution_price_df,
        rebalance_weight_df=rebalance_weight_df,
        backtest_start_date_str=None,
    )

    def strategy_factory_fn() -> InflationCompassTimingStrategy:
        strategy_obj = _build_strategy_obj(
            config_obj=config_obj,
            rebalance_weight_df=rebalance_weight_df,
            strategy_class_obj=InflationCompassTimingStrategy,
        )
        strategy_obj.month_end_feature_df = month_end_feature_df.copy()
        strategy_obj.month_end_weight_df = month_end_weight_df.copy()
        _attach_t5yie_provenance(strategy_obj, t5yie_snapshot_obj)
        strategy_obj.timing_pricing_data_df = execution_price_df
        return strategy_obj

    return {
        "strategy_factory_fn": strategy_factory_fn,
        "pricing_data_df": execution_price_df,
        "calendar_idx": calendar_index,
        "order_generation_mode_str": "vanilla_current_bar",
        "risk_model_str": "taa_rebalance",
        "entry_timing_str_tuple": (
            "same_open",
            "same_close_moc",
            "next_open",
            "next_close",
        ),
        "exit_timing_str_tuple": (
            "same_open",
            "same_close_moc",
            "next_open",
            "next_close",
        ),
        # The order-generation bar is already the mapped T+1 session. Thus the
        # analyzer's same_open cell is the strategy's causal Open_(T+1).
        "default_entry_timing_str": "same_open",
        "default_exit_timing_str": "same_open",
    }


def build_stress_test_context_dict() -> dict[str, object]:
    config_obj = DEFAULT_CONFIG
    (
        execution_price_df,
        month_end_feature_df,
        month_end_weight_df,
        rebalance_weight_df,
        t5yie_snapshot_obj,
    ) = get_inflation_compass_data(config_obj)
    calendar_index = _execution_calendar_index(
        execution_price_df=execution_price_df,
        rebalance_weight_df=rebalance_weight_df,
        backtest_start_date_str=None,
    )
    return {
        "strategy_name_str": STRATEGY_NAME_STR,
        "capital_base_float": float(config_obj.capital_base_float),
        "config_obj": config_obj,
        "pricing_data_df": execution_price_df,
        "calendar_idx": calendar_index,
        "month_end_feature_df": month_end_feature_df,
        "month_end_weight_df": month_end_weight_df,
        "rebalance_weight_df": rebalance_weight_df,
        "t5yie_snapshot_obj": t5yie_snapshot_obj,
    }


def build_stress_test_strategy_obj(
    context_dict: dict[str, object],
) -> InflationCompassStrategy:
    strategy_obj = _build_strategy_obj(
        config_obj=context_dict["config_obj"],
        rebalance_weight_df=context_dict["rebalance_weight_df"],
    )
    strategy_obj.month_end_feature_df = context_dict["month_end_feature_df"].copy()
    strategy_obj.month_end_weight_df = context_dict["month_end_weight_df"].copy()
    _attach_t5yie_provenance(
        strategy_obj,
        context_dict["t5yie_snapshot_obj"],
    )
    return strategy_obj


if __name__ == "__main__":
    run_variant()
