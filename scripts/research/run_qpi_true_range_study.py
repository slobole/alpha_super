"""Research-only True Range percentile study for the active QPI strategy."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT_PATH = Path(__file__).resolve().parents[2]
if str(REPO_ROOT_PATH) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_PATH))

from alpha.engine.backtest import run_daily
from alpha.engine.report import build_research_output_path
from data.norgate_loader import build_index_constituent_matrix
from scripts.research.run_sector_dispersion_marginal_universe_study import (
    _json_default_obj,
    _summary_value_float,
    compute_period_metric_dict,
)
from scripts.research.run_sector_dispersion_short_sleeve_study import (
    _benchmark_return_ser,
    _markdown_table_str,
    _market_metric_dict,
    _normalize_equity_ser,
    _performance_metric_dict,
)
from strategies.qpi.strategy_mr_qpi_ibs_rsi_exit import (
    QPIIbsRsiExitStrategy,
    default_trade_id_int,
    get_asof_universe_symbol_list,
    get_prices,
)


VARIANT_BASELINE_STR = "baseline_turnover_rank"
VARIANT_RANGE_CONFIRMATION_STR = "range_pct_ge_80"
VARIANT_EXTREME_GUARD_STR = "range_pct_lt_95"
VARIANT_MODE_TUPLE = (
    VARIANT_BASELINE_STR,
    VARIANT_RANGE_CONFIRMATION_STR,
    VARIANT_EXTREME_GUARD_STR,
)
RANGE_PERCENTILE_LOOKBACK_INT = 252
RANGE_CONFIRMATION_THRESHOLD_FLOAT = 80.0
EXTREME_RANGE_THRESHOLD_FLOAT = 95.0
BACKTEST_START_DATE_STR = "2004-01-01"
VALIDATION_START_DATE_STR = "2016-01-01"
MARKET_TAIL_QUANTILE_FLOAT = 0.10
EVENT_HORIZON_DAY_TUPLE = (1, 2, 3, 5)
RANGE_BUCKET_LABEL_TUPLE = ("00_20", "20_50", "50_80", "80_95", "95_100")


def compute_true_range_percentile_df(
    close_price_df: pd.DataFrame,
    high_price_df: pd.DataFrame,
    low_price_df: pd.DataFrame,
    lookback_day_int: int = RANGE_PERCENTILE_LOOKBACK_INT,
) -> pd.DataFrame:
    """Return the trailing percentile of TrueRange_T / Close_T-1."""
    if lookback_day_int <= 1:
        raise ValueError("lookback_day_int must be greater than 1.")

    numeric_close_price_df = close_price_df.apply(pd.to_numeric, errors="coerce")
    numeric_high_price_df = high_price_df.apply(pd.to_numeric, errors="coerce")
    numeric_low_price_df = low_price_df.apply(pd.to_numeric, errors="coerce")
    # *** CRITICAL*** previous_close_price_df is shifted by one session. True
    # Range_T may use completed OHLC_T, but no Close after T may affect an order
    # generated at Close T and filled at Open T+1.
    previous_close_price_df = numeric_close_price_df.shift(1)
    intraday_range_df = numeric_high_price_df - numeric_low_price_df
    high_gap_range_df = (numeric_high_price_df - previous_close_price_df).abs()
    low_gap_range_df = (numeric_low_price_df - previous_close_price_df).abs()
    true_range_df = intraday_range_df.combine(high_gap_range_df, np.maximum)
    true_range_df = true_range_df.combine(low_gap_range_df, np.maximum)
    normalized_true_range_df = true_range_df / previous_close_price_df.where(
        previous_close_price_df > 0.0
    )

    # *** CRITICAL*** rolling rank contains only observations through T. The
    # current completed True Range is intentionally included in its own
    # trailing distribution; future observations cannot revise RangePct_T.
    return (
        normalized_true_range_df.rolling(
            window=int(lookback_day_int),
            min_periods=int(lookback_day_int),
        ).rank(pct=True)
        * 100.0
    )


class QPITrueRangeResearchStrategy(QPIIbsRsiExitStrategy):
    """Active QPI/IBS/RSI2 strategy with one locked RangePct interpretation."""

    signal_audit_sample_size = 3

    def __init__(
        self,
        name: str,
        benchmarks: list[str],
        variant_mode_str: str,
        capital_base_float: float = 100_000.0,
    ) -> None:
        if variant_mode_str not in VARIANT_MODE_TUPLE:
            raise ValueError(f"variant_mode_str must be one of {VARIANT_MODE_TUPLE}.")
        super().__init__(
            name=name,
            benchmarks=benchmarks,
            capital_base=capital_base_float,
            slippage=0.00025,
            commission_per_share=0.005,
            commission_minimum=1.0,
        )
        self.variant_mode_str = variant_mode_str
        self.candidate_day_count_int = 0
        self.baseline_candidate_count_int = 0
        self.confirmation_candidate_count_int = 0
        self.guard_candidate_count_int = 0
        self.candidate_range_percentile_list: list[float] = []

    def compute_signals(self, pricing_data_df: pd.DataFrame) -> pd.DataFrame:
        signal_data_df = super().compute_signals(pricing_data_df)
        tradable_symbol_list = [
            str(symbol_obj)
            for symbol_obj in signal_data_df.columns.get_level_values(0).unique()
            if not str(symbol_obj).startswith("$")
            and (symbol_obj, "Close") in signal_data_df.columns
            and (symbol_obj, "High") in signal_data_df.columns
            and (symbol_obj, "Low") in signal_data_df.columns
        ]
        close_price_df = pd.DataFrame(
            {
                symbol_str: signal_data_df[(symbol_str, "Close")]
                for symbol_str in tradable_symbol_list
            },
            index=signal_data_df.index,
        )
        high_price_df = pd.DataFrame(
            {
                symbol_str: signal_data_df[(symbol_str, "High")]
                for symbol_str in tradable_symbol_list
            },
            index=signal_data_df.index,
        )
        low_price_df = pd.DataFrame(
            {
                symbol_str: signal_data_df[(symbol_str, "Low")]
                for symbol_str in tradable_symbol_list
            },
            index=signal_data_df.index,
        )
        range_percentile_df = compute_true_range_percentile_df(
            close_price_df=close_price_df,
            high_price_df=high_price_df,
            low_price_df=low_price_df,
        )
        range_feature_df = range_percentile_df.copy()
        range_feature_df.columns = pd.MultiIndex.from_tuples(
            [
                (str(symbol_str), "true_range_percentile_ser")
                for symbol_str in range_feature_df.columns
            ]
        )
        return pd.concat([signal_data_df, range_feature_df], axis=1)

    def _baseline_candidate_df(self, close_row_ser: pd.Series) -> pd.DataFrame:
        candidate_df = close_row_ser.unstack()
        candidate_df = candidate_df[~candidate_df.index.astype(str).str.startswith("$")]
        required_field_list = [
            "Close",
            "Turnover",
            "qpi_value_ser",
            "sma_200_price_ser",
            "three_day_return_ser",
            "ibs_value_ser",
        ]
        if any(field_str not in candidate_df.columns for field_str in required_field_list):
            return candidate_df.iloc[0:0]

        candidate_df = candidate_df.dropna(subset=required_field_list)
        candidate_df = candidate_df[
            candidate_df["qpi_value_ser"].astype(float) < self.qpi_threshold_float
        ]
        candidate_df = candidate_df[
            candidate_df["Close"].astype(float)
            > candidate_df["sma_200_price_ser"].astype(float)
        ]
        candidate_df = candidate_df[
            candidate_df["three_day_return_ser"].astype(float) < 0.0
        ]
        candidate_df = candidate_df[
            candidate_df["ibs_value_ser"].astype(float) < self.max_entry_ibs_float
        ]
        if self.universe_df is not None:
            universe_symbol_list = get_asof_universe_symbol_list(
                self.universe_df,
                pd.Timestamp(self.previous_bar),
            )
            candidate_df = candidate_df[candidate_df.index.isin(universe_symbol_list)]
        return candidate_df

    @staticmethod
    def _turnover_ranked_symbol_list(candidate_df: pd.DataFrame) -> list[str]:
        ranked_candidate_df = candidate_df.assign(
            symbol_str=candidate_df.index.astype(str)
        ).sort_values(
            by=["Turnover", "symbol_str"],
            ascending=[False, True],
            kind="mergesort",
        )
        return ranked_candidate_df.index.astype(str).tolist()

    def get_opportunity_list(self, close_row_ser: pd.Series) -> list[str]:
        candidate_df = self._baseline_candidate_df(close_row_ser)
        range_percentile_ser = pd.to_numeric(
            candidate_df.get("true_range_percentile_ser"),
            errors="coerce",
        ).replace([np.inf, -np.inf], np.nan)
        confirmation_bool_ser = range_percentile_ser.ge(
            RANGE_CONFIRMATION_THRESHOLD_FLOAT
        )
        guard_bool_ser = range_percentile_ser.lt(EXTREME_RANGE_THRESHOLD_FLOAT)

        self.candidate_day_count_int += 1
        self.baseline_candidate_count_int += int(len(candidate_df))
        self.confirmation_candidate_count_int += int(confirmation_bool_ser.sum())
        self.guard_candidate_count_int += int(guard_bool_ser.sum())
        self.candidate_range_percentile_list.extend(
            range_percentile_ser.dropna().astype(float).tolist()
        )

        if self.variant_mode_str == VARIANT_RANGE_CONFIRMATION_STR:
            candidate_df = candidate_df.loc[confirmation_bool_ser]
        elif self.variant_mode_str == VARIANT_EXTREME_GUARD_STR:
            candidate_df = candidate_df.loc[guard_bool_ser]
        return self._turnover_ranked_symbol_list(candidate_df)


def compute_event_target_dict(
    decision_idx_int: int,
    open_price_arr: np.ndarray,
    high_price_arr: np.ndarray,
    low_price_arr: np.ndarray,
    close_price_arr: np.ndarray,
    ibs_value_arr: np.ndarray,
    rsi2_value_arr: np.ndarray,
    exit_ibs_threshold_float: float = 0.90,
    exit_rsi2_threshold_float: float = 90.0,
) -> dict[str, float]:
    """Compute gross event outcomes from entry Open T+1 to the QPI exit Open."""
    observation_count_int = len(open_price_arr)
    entry_idx_int = int(decision_idx_int) + 1
    output_dict = {
        "entry_open_price_float": np.nan,
        "exit_open_return_pct_float": np.nan,
        "holding_session_count_float": np.nan,
        "mae_to_exit_pct_float": np.nan,
        "mfe_to_exit_pct_float": np.nan,
        "exit_observed_bool": False,
    }
    for horizon_day_int in EVENT_HORIZON_DAY_TUPLE:
        output_dict[f"forward_{horizon_day_int}d_return_pct_float"] = np.nan
    output_dict["forward_5d_mae_pct_float"] = np.nan
    output_dict["forward_5d_mfe_pct_float"] = np.nan

    if entry_idx_int >= observation_count_int:
        return output_dict
    entry_open_price_float = float(open_price_arr[entry_idx_int])
    if not np.isfinite(entry_open_price_float) or entry_open_price_float <= 0.0:
        return output_dict
    output_dict["entry_open_price_float"] = entry_open_price_float

    # *** CRITICAL*** all targets start from Open T+1. They are labels used only
    # after candidate generation and never flow back into the trading signal.
    for horizon_day_int in EVENT_HORIZON_DAY_TUPLE:
        target_idx_int = entry_idx_int + horizon_day_int - 1
        if target_idx_int < observation_count_int:
            target_close_float = float(close_price_arr[target_idx_int])
            if np.isfinite(target_close_float):
                output_dict[f"forward_{horizon_day_int}d_return_pct_float"] = (
                    target_close_float / entry_open_price_float - 1.0
                ) * 100.0

    five_day_end_idx_int = min(entry_idx_int + 5, observation_count_int)
    five_day_low_arr = low_price_arr[entry_idx_int:five_day_end_idx_int]
    five_day_high_arr = high_price_arr[entry_idx_int:five_day_end_idx_int]
    if np.isfinite(five_day_low_arr).any():
        output_dict["forward_5d_mae_pct_float"] = (
            float(np.nanmin(five_day_low_arr)) / entry_open_price_float - 1.0
        ) * 100.0
    if np.isfinite(five_day_high_arr).any():
        output_dict["forward_5d_mfe_pct_float"] = (
            float(np.nanmax(five_day_high_arr)) / entry_open_price_float - 1.0
        ) * 100.0

    exit_signal_idx_int = -1
    for signal_idx_int in range(entry_idx_int, observation_count_int - 1):
        ibs_value_float = float(ibs_value_arr[signal_idx_int])
        rsi2_value_float = float(rsi2_value_arr[signal_idx_int])
        exit_for_ibs_bool = np.isfinite(ibs_value_float) and (
            ibs_value_float > exit_ibs_threshold_float
        )
        exit_for_rsi_bool = np.isfinite(rsi2_value_float) and (
            rsi2_value_float > exit_rsi2_threshold_float
        )
        if exit_for_ibs_bool or exit_for_rsi_bool:
            candidate_exit_open_float = float(open_price_arr[signal_idx_int + 1])
            if np.isfinite(candidate_exit_open_float) and candidate_exit_open_float > 0.0:
                exit_signal_idx_int = signal_idx_int
                break

    if exit_signal_idx_int < 0:
        return output_dict
    exit_idx_int = exit_signal_idx_int + 1
    exit_open_price_float = float(open_price_arr[exit_idx_int])
    output_dict["exit_open_return_pct_float"] = (
        exit_open_price_float / entry_open_price_float - 1.0
    ) * 100.0
    output_dict["holding_session_count_float"] = float(exit_idx_int - entry_idx_int)
    output_dict["exit_observed_bool"] = True

    exposure_low_arr = low_price_arr[entry_idx_int:exit_idx_int]
    exposure_high_arr = high_price_arr[entry_idx_int:exit_idx_int]
    if np.isfinite(exposure_low_arr).any():
        output_dict["mae_to_exit_pct_float"] = (
            float(np.nanmin(exposure_low_arr)) / entry_open_price_float - 1.0
        ) * 100.0
    if np.isfinite(exposure_high_arr).any():
        output_dict["mfe_to_exit_pct_float"] = (
            float(np.nanmax(exposure_high_arr)) / entry_open_price_float - 1.0
        ) * 100.0
    return output_dict


def build_candidate_event_df(
    strategy_obj: QPITrueRangeResearchStrategy,
    signal_data_df: pd.DataFrame,
    calendar_idx: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Build candidate-level outcomes without portfolio slot interactions."""
    candidate_frame_list: list[pd.DataFrame] = []
    for decision_date_ts in calendar_idx:
        strategy_obj.previous_bar = pd.Timestamp(decision_date_ts)
        close_row_ser = signal_data_df.loc[decision_date_ts]
        candidate_df = strategy_obj._baseline_candidate_df(close_row_ser)
        if len(candidate_df) == 0:
            continue
        event_df = candidate_df[
            [
                "qpi_value_ser",
                "ibs_value_ser",
                "three_day_return_ser",
                "Turnover",
                "true_range_percentile_ser",
            ]
        ].copy()
        event_df["decision_date"] = pd.Timestamp(decision_date_ts)
        event_df["symbol_str"] = event_df.index.astype(str)
        candidate_frame_list.append(event_df.reset_index(drop=True))

    if len(candidate_frame_list) == 0:
        return pd.DataFrame()
    candidate_event_df = pd.concat(candidate_frame_list, ignore_index=True)
    candidate_event_df["true_range_percentile_ser"] = pd.to_numeric(
        candidate_event_df["true_range_percentile_ser"],
        errors="coerce",
    )
    candidate_event_df = candidate_event_df.dropna(
        subset=["true_range_percentile_ser"]
    ).copy()

    date_position_dict = {
        pd.Timestamp(date_obj): idx_int
        for idx_int, date_obj in enumerate(signal_data_df.index)
    }
    target_row_list: list[dict[str, object]] = []
    for symbol_str, symbol_event_df in candidate_event_df.groupby("symbol_str", sort=False):
        required_column_tuple = (
            (symbol_str, "Open"),
            (symbol_str, "High"),
            (symbol_str, "Low"),
            (symbol_str, "Close"),
            (symbol_str, "ibs_value_ser"),
            (symbol_str, "rsi2_value_ser"),
        )
        if any(column_tuple not in signal_data_df.columns for column_tuple in required_column_tuple):
            continue
        open_price_arr = signal_data_df[(symbol_str, "Open")].to_numpy(dtype=float)
        high_price_arr = signal_data_df[(symbol_str, "High")].to_numpy(dtype=float)
        low_price_arr = signal_data_df[(symbol_str, "Low")].to_numpy(dtype=float)
        close_price_arr = signal_data_df[(symbol_str, "Close")].to_numpy(dtype=float)
        ibs_value_arr = signal_data_df[(symbol_str, "ibs_value_ser")].to_numpy(dtype=float)
        rsi2_value_arr = signal_data_df[(symbol_str, "rsi2_value_ser")].to_numpy(dtype=float)

        for event_row_obj in symbol_event_df.itertuples(index=False):
            decision_date_ts = pd.Timestamp(event_row_obj.decision_date)
            target_dict = compute_event_target_dict(
                decision_idx_int=date_position_dict[decision_date_ts],
                open_price_arr=open_price_arr,
                high_price_arr=high_price_arr,
                low_price_arr=low_price_arr,
                close_price_arr=close_price_arr,
                ibs_value_arr=ibs_value_arr,
                rsi2_value_arr=rsi2_value_arr,
                exit_ibs_threshold_float=strategy_obj.exit_ibs_threshold_float,
                exit_rsi2_threshold_float=strategy_obj.exit_rsi2_threshold_float,
            )
            target_row_list.append(
                {
                    "decision_date": decision_date_ts,
                    "symbol_str": symbol_str,
                    "qpi_value_float": float(event_row_obj.qpi_value_ser),
                    "ibs_value_float": float(event_row_obj.ibs_value_ser),
                    "three_day_return_pct_float": float(
                        event_row_obj.three_day_return_ser
                    )
                    * 100.0,
                    "turnover_dollar_float": float(event_row_obj.Turnover),
                    "true_range_percentile_float": float(
                        event_row_obj.true_range_percentile_ser
                    ),
                    **target_dict,
                }
            )

    event_df = pd.DataFrame(target_row_list)
    if len(event_df) == 0:
        return event_df
    event_df["sample_period_str"] = np.where(
        event_df["decision_date"] < pd.Timestamp(VALIDATION_START_DATE_STR),
        "discovery_2004_2015",
        "validation_2016_plus",
    )
    event_df["range_bucket_str"] = pd.cut(
        event_df["true_range_percentile_float"],
        bins=[0.0, 20.0, 50.0, 80.0, 95.0, 100.0],
        labels=RANGE_BUCKET_LABEL_TUPLE,
        include_lowest=True,
        right=True,
    ).astype("string")
    return event_df.sort_values(["decision_date", "symbol_str"]).reset_index(drop=True)


def summarize_candidate_event_df(event_df: pd.DataFrame) -> pd.DataFrame:
    summary_row_list: list[dict[str, object]] = []
    sample_frame_list = [
        ("full_sample", event_df),
        ("discovery_2004_2015", event_df[event_df["sample_period_str"] == "discovery_2004_2015"]),
        ("validation_2016_plus", event_df[event_df["sample_period_str"] == "validation_2016_plus"]),
    ]
    for sample_period_str, sample_df in sample_frame_list:
        for range_bucket_str in RANGE_BUCKET_LABEL_TUPLE:
            bucket_df = sample_df[sample_df["range_bucket_str"] == range_bucket_str]
            row_dict: dict[str, object] = {
                "sample_period_str": sample_period_str,
                "range_bucket_str": range_bucket_str,
                "candidate_count_int": int(len(bucket_df)),
                "entry_label_count_int": int(
                    pd.to_numeric(
                        bucket_df["entry_open_price_float"],
                        errors="coerce",
                    ).notna().sum()
                ),
                "exit_observed_pct_float": float(bucket_df["exit_observed_bool"].mean() * 100.0)
                if len(bucket_df) > 0
                else np.nan,
            }
            for horizon_day_int in EVENT_HORIZON_DAY_TUPLE:
                return_column_str = f"forward_{horizon_day_int}d_return_pct_float"
                return_ser = pd.to_numeric(
                    bucket_df[return_column_str],
                    errors="coerce",
                ).dropna()
                row_dict[f"forward_{horizon_day_int}d_label_count_int"] = int(
                    len(return_ser)
                )
                row_dict[f"forward_{horizon_day_int}d_mean_pct_float"] = float(return_ser.mean())
                row_dict[f"forward_{horizon_day_int}d_hit_pct_float"] = float(
                    return_ser.gt(0.0).mean() * 100.0
                )
            for source_column_str, output_column_str in (
                ("exit_open_return_pct_float", "exit_return_mean_pct_float"),
                ("holding_session_count_float", "holding_session_median_float"),
                ("mae_to_exit_pct_float", "mae_to_exit_mean_pct_float"),
                ("mfe_to_exit_pct_float", "mfe_to_exit_mean_pct_float"),
                ("forward_5d_mae_pct_float", "forward_5d_mae_mean_pct_float"),
                ("forward_5d_mfe_pct_float", "forward_5d_mfe_mean_pct_float"),
            ):
                value_ser = pd.to_numeric(bucket_df[source_column_str], errors="coerce")
                aggregation_func = value_ser.median if "median" in output_column_str else value_ser.mean
                row_dict[output_column_str] = float(aggregation_func())
            exit_return_ser = pd.to_numeric(
                bucket_df["exit_open_return_pct_float"],
                errors="coerce",
            ).dropna()
            row_dict["exit_return_label_count_int"] = int(len(exit_return_ser))
            row_dict["exit_return_hit_pct_float"] = float(
                exit_return_ser.gt(0.0).mean() * 100.0
            )
            summary_row_list.append(row_dict)
    return pd.DataFrame(summary_row_list)


def _exposure_metric_dict(strategy_obj: QPITrueRangeResearchStrategy) -> dict[str, float]:
    realized_weight_df = getattr(strategy_obj, "realized_weight_df", pd.DataFrame())
    if realized_weight_df is None or len(realized_weight_df) == 0:
        return {
            "avg_position_count_float": np.nan,
            "avg_gross_exposure_pct_float": np.nan,
            "active_day_pct_float": np.nan,
        }
    weight_df = realized_weight_df.copy()
    weight_df.columns = [str(column_obj) for column_obj in weight_df.columns]
    tradable_column_list = [
        column_str
        for column_str in weight_df.columns
        if not column_str.startswith("$") and column_str.lower() != "cash"
    ]
    tradable_weight_df = weight_df[tradable_column_list].apply(
        pd.to_numeric,
        errors="coerce",
    ).fillna(0.0)
    gross_exposure_ser = tradable_weight_df.abs().sum(axis=1)
    position_count_ser = tradable_weight_df.abs().gt(1e-9).sum(axis=1)
    return {
        "avg_position_count_float": float(position_count_ser.mean()),
        "avg_gross_exposure_pct_float": float(gross_exposure_ser.mean() * 100.0),
        "active_day_pct_float": float(gross_exposure_ser.gt(1e-9).mean() * 100.0),
    }


def _candidate_diagnostic_dict(
    strategy_obj: QPITrueRangeResearchStrategy,
) -> dict[str, float | int]:
    candidate_count_int = int(strategy_obj.baseline_candidate_count_int)
    range_percentile_arr = np.asarray(
        strategy_obj.candidate_range_percentile_list,
        dtype=float,
    )
    range_percentile_arr = range_percentile_arr[np.isfinite(range_percentile_arr)]
    return {
        "candidate_day_count_int": int(strategy_obj.candidate_day_count_int),
        "baseline_candidate_count_int": candidate_count_int,
        "confirmation_candidate_count_int": int(strategy_obj.confirmation_candidate_count_int),
        "guard_candidate_count_int": int(strategy_obj.guard_candidate_count_int),
        "confirmation_candidate_pct_float": (
            np.nan
            if candidate_count_int == 0
            else float(strategy_obj.confirmation_candidate_count_int / candidate_count_int * 100.0)
        ),
        "guard_candidate_pct_float": (
            np.nan
            if candidate_count_int == 0
            else float(strategy_obj.guard_candidate_count_int / candidate_count_int * 100.0)
        ),
        "candidate_range_pct_median_float": (
            np.nan if len(range_percentile_arr) == 0 else float(np.median(range_percentile_arr))
        ),
    }


def _portfolio_summary_row_dict(
    strategy_obj: QPITrueRangeResearchStrategy,
    benchmark_return_ser: pd.Series,
) -> dict[str, object]:
    total_value_ser = strategy_obj.results["total_value"]
    row_dict: dict[str, object] = {
        "variant_mode_str": strategy_obj.variant_mode_str,
        "strategy_name_str": strategy_obj.name,
        "turnover_ann_pct_float": _summary_value_float(
            strategy_obj.summary,
            "Turnover (Ann.) [%]",
        ),
        "cost_drag_ann_pct_float": _summary_value_float(
            strategy_obj.summary,
            "Cost Drag (Ann.) [%]",
        ),
        "exposure_time_pct_float": _summary_value_float(
            strategy_obj.summary,
            "Exposure Time [%]",
        ),
        "transaction_count_int": int(len(strategy_obj.get_transactions())),
    }
    row_dict.update(_performance_metric_dict(total_value_ser))
    ann_return_float = float(row_dict["ann_return_pct_float"])
    max_drawdown_float = float(row_dict["max_drawdown_pct_float"])
    row_dict["mar_float"] = (
        np.nan
        if max_drawdown_float == 0.0
        else ann_return_float / abs(max_drawdown_float)
    )
    row_dict.update(
        compute_period_metric_dict(
            total_value_ser=total_value_ser,
            start_ts=pd.Timestamp(BACKTEST_START_DATE_STR),
            end_ts=pd.Timestamp("2015-12-31"),
            prefix_str="discovery_2004_2015",
        )
    )
    row_dict.update(
        compute_period_metric_dict(
            total_value_ser=total_value_ser,
            start_ts=pd.Timestamp(VALIDATION_START_DATE_STR),
            end_ts=None,
            prefix_str="validation_2016_plus",
        )
    )
    row_dict.update(
        _market_metric_dict(
            total_value_ser,
            benchmark_return_ser,
            MARKET_TAIL_QUANTILE_FLOAT,
        )
    )
    row_dict.update(_exposure_metric_dict(strategy_obj))
    row_dict.update(_candidate_diagnostic_dict(strategy_obj))
    return row_dict


def _save_equity_chart(output_path: Path, equity_df: pd.DataFrame) -> None:
    fig_obj, axis_obj = plt.subplots(figsize=(14, 8))
    for column_str in equity_df.columns:
        normalized_equity_ser = _normalize_equity_ser(equity_df[column_str])
        axis_obj.plot(
            normalized_equity_ser.index,
            normalized_equity_ser.values,
            label=column_str,
            linewidth=1.6,
        )
    axis_obj.axvline(pd.Timestamp(VALIDATION_START_DATE_STR), color="black", linestyle="--", alpha=0.6)
    axis_obj.set_title("QPI True Range Percentile Variants")
    axis_obj.set_xlabel("Date")
    axis_obj.set_ylabel("Growth of 1.0")
    axis_obj.grid(True, alpha=0.25)
    axis_obj.legend(loc="best")
    fig_obj.tight_layout()
    fig_obj.savefig(output_path / "equity_curves.png", dpi=170)
    plt.close(fig_obj)


def _save_event_chart(output_path: Path, event_summary_df: pd.DataFrame) -> None:
    validation_df = event_summary_df[
        event_summary_df["sample_period_str"] == "validation_2016_plus"
    ].set_index("range_bucket_str")
    validation_df = validation_df.reindex(RANGE_BUCKET_LABEL_TUPLE)
    fig_obj, axis_obj = plt.subplots(figsize=(11, 6))
    axis_obj.bar(
        validation_df.index,
        validation_df["exit_return_mean_pct_float"],
        color=["#4c78a8", "#72b7b2", "#54a24b", "#f2cf5b", "#e45756"],
    )
    axis_obj.axhline(0.0, color="black", linewidth=1.0)
    axis_obj.set_title("QPI Validation Event Return by True Range Percentile")
    axis_obj.set_xlabel("True Range percentile bucket")
    axis_obj.set_ylabel("Mean gross return to QPI exit [%]")
    axis_obj.grid(True, axis="y", alpha=0.25)
    fig_obj.tight_layout()
    fig_obj.savefig(output_path / "validation_event_returns.png", dpi=170)
    plt.close(fig_obj)


def _write_recommendations_md(
    output_path: Path,
    portfolio_summary_df: pd.DataFrame,
    event_summary_df: pd.DataFrame,
) -> None:
    portfolio_column_list = [
        "variant_mode_str",
        "ann_return_pct_float",
        "volatility_ann_pct_float",
        "sharpe_float",
        "max_drawdown_pct_float",
        "mar_float",
        "validation_2016_plus_ann_return_pct_float",
        "validation_2016_plus_sharpe_float",
        "validation_2016_plus_max_drawdown_pct_float",
        "turnover_ann_pct_float",
        "cost_drag_ann_pct_float",
        "avg_gross_exposure_pct_float",
        "market_tail_beta_to_spx_float",
    ]
    event_column_list = [
        "sample_period_str",
        "range_bucket_str",
        "candidate_count_int",
        "forward_5d_label_count_int",
        "forward_1d_mean_pct_float",
        "forward_5d_mean_pct_float",
        "exit_return_label_count_int",
        "exit_return_mean_pct_float",
        "exit_return_hit_pct_float",
        "mae_to_exit_mean_pct_float",
        "mfe_to_exit_mean_pct_float",
        "holding_session_median_float",
    ]
    validation_event_df = event_summary_df[
        event_summary_df["sample_period_str"] == "validation_2016_plus"
    ]
    recommendations_md_str = f"""# QPI True Range Percentile Study

## Scope

- Research-only; active QPI, Portfolio Manager, Bench, and live wiring are unchanged.
- Three predeclared portfolio variants; validation starts `2016-01-01`.
- Universe: Norgate point-in-time S&P 500 Current & Past.
- Signal: completed Close `T`; market fill at Open `T+1`.
- Costs: 2.5 bps slippage per side, `$0.005/share`, `$1` minimum.
- `RangePct_T` is the trailing 252-session percentile of `TrueRange_T / Close_T-1`.

## Portfolio Results

{_markdown_table_str(portfolio_summary_df[portfolio_column_list])}

## Validation Candidate Events

{_markdown_table_str(validation_event_df[event_column_list])}

## Interpretation Limits

Candidate events overlap and are not independent trades. Event returns are gross diagnostics without portfolio capacity or costs; the portfolio rows are the costed strategy test. Label-count columns disclose end-of-sample observations that lack enough future data. The broader QPI family has prior timing, sizing, exit, short, and weekly experiments, so these three variants are not the complete multiple-testing family. Study Sharpe uses all calendar returns and therefore differs slightly from the engine report's active-time Sharpe; comparisons within this table use one consistent definition.
"""
    (output_path / "recommendations.md").write_text(
        recommendations_md_str,
        encoding="utf-8",
    )


def run_qpi_true_range_study(
    output_dir_str: str = "results",
    end_date_str: str | None = None,
    show_progress_bool: bool = False,
) -> Path:
    timestamp_str = pd.Timestamp.now().strftime("%Y-%m-%d_%H%M%S")
    output_path = build_research_output_path(
        output_dir=output_dir_str,
        entity_type_str="strategy",
        entity_id_str="strategy_mr_qpi_ibs_rsi_exit",
        analysis_type_str="true_range_percentile_study",
        timestamp_str=timestamp_str,
    )
    output_path.mkdir(parents=True, exist_ok=False)

    benchmark_list = ["$SPX"]
    symbol_list, universe_df = build_index_constituent_matrix(indexname="S&P 500")
    pricing_data_df = get_prices(
        symbol_list,
        benchmark_list,
        start_date_str="1998-01-01",
        end_date_str=end_date_str,
    )
    calendar_idx = pricing_data_df.index[
        pricing_data_df.index >= pd.Timestamp(BACKTEST_START_DATE_STR)
    ]
    benchmark_return_ser = _benchmark_return_ser(pricing_data_df, "$SPX")

    print("Building candidate event study...", flush=True)
    event_strategy_obj = QPITrueRangeResearchStrategy(
        name="strategy_mr_qpi_true_range_event_study",
        benchmarks=benchmark_list,
        variant_mode_str=VARIANT_BASELINE_STR,
    )
    event_strategy_obj.universe_df = universe_df
    signal_data_df = event_strategy_obj.compute_signals(pricing_data_df)
    candidate_event_df = build_candidate_event_df(
        strategy_obj=event_strategy_obj,
        signal_data_df=signal_data_df,
        calendar_idx=calendar_idx,
    )
    event_summary_df = summarize_candidate_event_df(candidate_event_df)
    candidate_event_df.to_csv(output_path / "candidate_events.csv", index=False)
    event_summary_df.to_csv(output_path / "event_summary.csv", index=False)
    del signal_data_df

    portfolio_summary_row_list: list[dict[str, object]] = []
    equity_dict: dict[str, pd.Series] = {}
    for variant_mode_str in VARIANT_MODE_TUPLE:
        print(f"Running {variant_mode_str}...", flush=True)
        strategy_obj = QPITrueRangeResearchStrategy(
            name=f"strategy_mr_qpi_{variant_mode_str}",
            benchmarks=benchmark_list,
            variant_mode_str=variant_mode_str,
        )
        strategy_obj.universe_df = universe_df
        strategy_obj.trade_id_int = 0
        strategy_obj.current_trade_map = defaultdict(default_trade_id_int)
        run_daily(
            strategy_obj,
            pricing_data_df,
            calendar=calendar_idx,
            show_progress=show_progress_bool,
            show_signal_progress_bool=show_progress_bool,
            audit_override_bool=True,
        )
        portfolio_summary_row_list.append(
            _portfolio_summary_row_dict(
                strategy_obj=strategy_obj,
                benchmark_return_ser=benchmark_return_ser,
            )
        )
        equity_dict[variant_mode_str] = strategy_obj.results["total_value"]
        pd.DataFrame(strategy_obj.get_transactions()).to_csv(
            output_path / f"transactions_{variant_mode_str}.csv",
            index=False,
        )

    portfolio_summary_df = pd.DataFrame(portfolio_summary_row_list)
    equity_df = pd.DataFrame(equity_dict).sort_index()
    portfolio_summary_df.to_csv(output_path / "portfolio_summary.csv", index=False)
    equity_df.to_csv(output_path / "equity_curves.csv", index_label="date")
    _save_equity_chart(output_path=output_path, equity_df=equity_df)
    _save_event_chart(output_path=output_path, event_summary_df=event_summary_df)
    _write_recommendations_md(
        output_path=output_path,
        portfolio_summary_df=portfolio_summary_df,
        event_summary_df=event_summary_df,
    )

    metadata_dict = {
        "analysis_type_str": "true_range_percentile_study",
        "generated_at_str": pd.Timestamp.now().isoformat(),
        "output_path_str": str(output_path.resolve()),
        "portfolio_run_script_sha256_str": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "local_variant_count_int": len(VARIANT_MODE_TUPLE),
        "variant_mode_tuple": VARIANT_MODE_TUPLE,
        "backtest_start_date_str": BACKTEST_START_DATE_STR,
        "validation_start_date_str": VALIDATION_START_DATE_STR,
        "end_date_str": end_date_str,
        "candidate_event_count_int": int(len(candidate_event_df)),
        "universe_str": "Norgate point-in-time S&P 500 Current & Past",
        "stock_adjustment_str": "CAPITALSPECIAL",
        "benchmark_symbol_str": "$SPX",
        "benchmark_adjustment_str": "TOTALRETURN",
        "range_percentile_lookback_int": RANGE_PERCENTILE_LOOKBACK_INT,
        "range_confirmation_threshold_float": RANGE_CONFIRMATION_THRESHOLD_FLOAT,
        "extreme_range_threshold_float": EXTREME_RANGE_THRESHOLD_FLOAT,
        "slippage_float": 0.00025,
        "commission_per_share_float": 0.005,
        "commission_minimum_float": 1.0,
        "execution_timing_note_str": "Completed bar T signal; market fill at Open T+1.",
        "event_study_note_str": (
            "Candidate events overlap and are gross diagnostics; portfolio variants are costed."
        ),
    }
    (output_path / "metadata.json").write_text(
        json.dumps(metadata_dict, indent=2, default=_json_default_obj),
        encoding="utf-8",
    )
    print(f"Saved QPI True Range study to {output_path}", flush=True)
    return output_path


def _parse_args() -> argparse.Namespace:
    parser_obj = argparse.ArgumentParser(
        description="Run the research-only QPI True Range percentile study."
    )
    parser_obj.add_argument("--output-dir", default="results")
    parser_obj.add_argument("--end-date", default=None)
    parser_obj.add_argument("--show-progress", action="store_true")
    return parser_obj.parse_args()


def main() -> int:
    args_obj = _parse_args()
    run_qpi_true_range_study(
        output_dir_str=str(args_obj.output_dir),
        end_date_str=args_obj.end_date,
        show_progress_bool=bool(args_obj.show_progress),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
