"""
Historical pre-crisis stress testing for strategy research.

StressTestAnalyzer answers a different question from CrisisAnalyzer:

    "If this strategy was funded before a known crisis and traded normally
    into the event, what happened?"

For each crisis window and launch offset:

    L = launch date, offset trading bars before the crisis start
    S = effective crisis start
    P = previous tradable bar before S
    E = effective crisis end

The engine runs normally from L through E with a fresh strategy instance.
Returns are then measured as:

    launch_return = V_E / V_L - 1
    event_return = V_E / V_P - 1

where V_L is fresh capital at launch and V_P is close-marked equity on the
bar before the event starts.
"""

from __future__ import annotations

import base64
import html
import io
import json
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from alpha.engine.backtest import run_daily
from alpha.engine.crisis import (
    CRISIS_PERIODS_LIST,
    SUPPORTED_CRISIS_STRATEGY_KEY_TUPLE,
    SUPPORTED_CRISIS_STRATEGY_SPEC_MAP,
    CrisisPeriodConfig,
    CrisisStrategySpec,
    _coerce_crisis_period_config_list,
    resolve_crisis_window,
)
from alpha.engine.report import build_research_output_path
from alpha.engine.strategy import Strategy
from alpha.engine.theme import (
    SEABORN_DEEP_COLOR_LIST,
    SIGNATURE_PALETTE_DICT,
    blend_hex_color_str,
    build_report_css,
    build_report_font_head_html,
    build_signature_rcparams,
)


STRESS_TEST_ANALYSIS_TYPE_STR = "stress_test"
DEFAULT_LAUNCH_OFFSET_TUPLE: tuple[int, ...] = (5, 21, 42, 63)
STRESS_METRICS_CSV_FILENAME_STR = "stress_metrics.csv"
STRESS_PATHS_CSV_FILENAME_STR = "stress_paths.csv"
STRESS_ENTRY_POSITIONS_CSV_FILENAME_STR = "stress_entry_positions.csv"
STRESS_TRANSACTIONS_CSV_FILENAME_STR = "stress_transactions.csv"
METADATA_FILENAME_STR = "metadata.json"
RUN_INFO_FILENAME_STR = "run_info.json"
SUMMARY_FILENAME_STR = "summary.json"
REPORT_FILENAME_STR = "report.html"


@dataclass
class StressTestResult:
    strategy_key_str: str
    strategy_name_str: str
    capital_base_float: float
    crisis_period_config_list: list[CrisisPeriodConfig]
    launch_offset_tuple: tuple[int, ...]
    stress_metric_df: pd.DataFrame
    stress_path_df: pd.DataFrame
    stress_entry_position_df: pd.DataFrame
    stress_transaction_df: pd.DataFrame
    stress_strategy_map: dict[str, Strategy] = field(default_factory=dict)
    output_dir_path: Path | None = None


class StressTestAnalyzer:
    """
    Run historical launch-before-crisis stress tests.

    Each scenario runs a normal Vanilla backtest from a pre-crisis launch date
    through the crisis end. The analyzer is research/report-only and does not
    change strategy, order, fill, or live execution semantics.
    """

    def __init__(
        self,
        strategy_key_str: str | None = None,
        *,
        strategy_spec_obj: CrisisStrategySpec | None = None,
        crisis_period_list: Sequence[CrisisPeriodConfig | dict[str, str]] | None = None,
        launch_offset_tuple: Sequence[int] = DEFAULT_LAUNCH_OFFSET_TUPLE,
        output_dir_str: str = "results",
        save_output_bool: bool = True,
        show_progress_bool: bool = False,
        show_signal_progress_bool: bool = False,
    ):
        if strategy_spec_obj is None:
            if strategy_key_str is None:
                raise ValueError(
                    "strategy_key_str is required when strategy_spec_obj is not provided."
                )
            if strategy_key_str not in SUPPORTED_CRISIS_STRATEGY_SPEC_MAP:
                raise ValueError(
                    f"Unsupported stress-test strategy '{strategy_key_str}'. "
                    f"Available keys: {sorted(SUPPORTED_CRISIS_STRATEGY_SPEC_MAP)}"
                )
            strategy_spec_obj = SUPPORTED_CRISIS_STRATEGY_SPEC_MAP[strategy_key_str]
        elif strategy_key_str is None:
            strategy_key_str = strategy_spec_obj.strategy_key_str

        self.strategy_key_str = str(strategy_key_str)
        self.strategy_spec_obj = strategy_spec_obj
        if crisis_period_list is None:
            crisis_period_list = CRISIS_PERIODS_LIST
        self.crisis_period_config_list = _coerce_crisis_period_config_list(crisis_period_list)
        self.launch_offset_tuple = _coerce_launch_offset_tuple(launch_offset_tuple)
        self.output_dir_str = str(output_dir_str)
        self.save_output_bool = bool(save_output_bool)
        self.show_progress_bool = bool(show_progress_bool)
        self.show_signal_progress_bool = bool(show_signal_progress_bool)
        self.latest_result_obj: StressTestResult | None = None

    @classmethod
    def from_strategy_key(
        cls,
        strategy_key_str: str,
        **kwargs,
    ) -> "StressTestAnalyzer":
        return cls(strategy_key_str=strategy_key_str, **kwargs)

    @classmethod
    def supported_strategy_key_tuple(cls) -> tuple[str, ...]:
        return supported_stress_test_strategy_key_list()

    def run(
        self,
        *,
        crisis_period_list: Sequence[CrisisPeriodConfig | dict[str, str]] | None = None,
        launch_offset_tuple: Sequence[int] | None = None,
        output_dir_str: str | None = None,
        save_output_bool: bool | None = None,
        show_progress_bool: bool | None = None,
        show_signal_progress_bool: bool | None = None,
    ) -> StressTestResult:
        normalized_crisis_period_list = (
            self.crisis_period_config_list
            if crisis_period_list is None
            else _coerce_crisis_period_config_list(crisis_period_list)
        )
        normalized_launch_offset_tuple = (
            self.launch_offset_tuple
            if launch_offset_tuple is None
            else _coerce_launch_offset_tuple(launch_offset_tuple)
        )
        resolved_output_dir_str = self.output_dir_str if output_dir_str is None else str(output_dir_str)
        resolved_save_output_bool = (
            self.save_output_bool if save_output_bool is None else bool(save_output_bool)
        )
        resolved_show_progress_bool = (
            self.show_progress_bool if show_progress_bool is None else bool(show_progress_bool)
        )
        resolved_show_signal_progress_bool = (
            self.show_signal_progress_bool
            if show_signal_progress_bool is None
            else bool(show_signal_progress_bool)
        )

        context_dict = self.strategy_spec_obj.load_context_fn()
        pricing_data_df = context_dict["pricing_data_df"]
        supported_calendar_idx = pd.DatetimeIndex(
            context_dict.get("calendar_idx", pricing_data_df.index)
        ).sort_values().unique()
        capital_base_float = float(context_dict["capital_base_float"])
        strategy_name_str = str(context_dict["strategy_name_str"])

        metric_row_list: list[dict[str, object]] = []
        path_frame_list: list[pd.DataFrame] = []
        entry_position_frame_list: list[pd.DataFrame] = []
        transaction_frame_list: list[pd.DataFrame] = []
        stress_strategy_map: dict[str, Strategy] = {}

        for crisis_period_config in normalized_crisis_period_list:
            event_start_ts, event_end_ts, skip_reason_str = resolve_crisis_window(
                crisis_period_config=crisis_period_config,
                calendar_idx=supported_calendar_idx,
            )
            if skip_reason_str:
                continue

            for launch_offset_int in normalized_launch_offset_tuple:
                launch_ts, event_entry_ts, launch_skip_reason_str = resolve_stress_launch_window(
                    event_start_ts=event_start_ts,
                    calendar_idx=supported_calendar_idx,
                    launch_offset_int=launch_offset_int,
                )
                if launch_skip_reason_str:
                    continue

                stress_calendar_idx = supported_calendar_idx[
                    (supported_calendar_idx >= launch_ts)
                    & (supported_calendar_idx <= event_end_ts)
                ]
                if len(stress_calendar_idx) == 0:
                    continue

                # *** CRITICAL*** The stress run starts before the event and
                # uses the normal run_daily lifecycle. Its first decision uses
                # data through the bar before launch_ts and fills at launch_ts
                # open. No crisis-window data is available before its bar.
                strategy_obj = self.strategy_spec_obj.build_strategy_fn(context_dict)
                run_daily(
                    strategy=strategy_obj,
                    pricing_data=pricing_data_df,
                    calendar=stress_calendar_idx,
                    show_progress=resolved_show_progress_bool,
                    show_signal_progress_bool=resolved_show_signal_progress_bool,
                )

                scenario_key_str = _scenario_key_str(
                    crisis_name_str=crisis_period_config.crisis_name_str,
                    launch_offset_int=launch_offset_int,
                )
                entry_position_df = _build_entry_position_df(
                    strategy_obj=strategy_obj,
                    pricing_data_df=pricing_data_df,
                    crisis_name_str=crisis_period_config.crisis_name_str,
                    launch_offset_int=launch_offset_int,
                    launch_ts=launch_ts,
                    event_start_ts=event_start_ts,
                    event_entry_ts=event_entry_ts,
                    event_end_ts=event_end_ts,
                    scenario_key_str=scenario_key_str,
                )
                stress_transaction_df = _build_stress_transaction_df(
                    strategy_obj=strategy_obj,
                    crisis_name_str=crisis_period_config.crisis_name_str,
                    launch_offset_int=launch_offset_int,
                    launch_ts=launch_ts,
                    event_start_ts=event_start_ts,
                    event_entry_ts=event_entry_ts,
                    event_end_ts=event_end_ts,
                    scenario_key_str=scenario_key_str,
                )
                metric_row_list.append(
                    _build_stress_metric_row_dict(
                        strategy_obj=strategy_obj,
                        crisis_name_str=crisis_period_config.crisis_name_str,
                        launch_offset_int=launch_offset_int,
                        launch_ts=launch_ts,
                        event_start_ts=event_start_ts,
                        event_entry_ts=event_entry_ts,
                        event_end_ts=event_end_ts,
                        scenario_key_str=scenario_key_str,
                        entry_position_df=entry_position_df,
                        stress_transaction_df=stress_transaction_df,
                    )
                )
                path_frame_list.append(
                    _build_stress_path_df(
                        strategy_obj=strategy_obj,
                        crisis_name_str=crisis_period_config.crisis_name_str,
                        launch_offset_int=launch_offset_int,
                        launch_ts=launch_ts,
                        event_start_ts=event_start_ts,
                        event_entry_ts=event_entry_ts,
                        event_end_ts=event_end_ts,
                        scenario_key_str=scenario_key_str,
                    )
                )
                entry_position_frame_list.append(entry_position_df)
                transaction_frame_list.append(stress_transaction_df)
                stress_strategy_map[scenario_key_str] = strategy_obj

        stress_metric_df = _concat_or_empty_df(metric_row_list, _stress_metric_column_list())
        if len(stress_metric_df) > 0:
            stress_metric_df = stress_metric_df.sort_values(
                by=["event_start_ts", "crisis_name_str", "launch_offset_int"],
                kind="mergesort",
            ).reset_index(drop=True)

        stress_path_df = _concat_frame_or_empty_df(path_frame_list, _stress_path_column_list())
        stress_entry_position_df = _concat_frame_or_empty_df(
            entry_position_frame_list,
            _stress_entry_position_column_list(),
        )
        stress_transaction_df = _concat_frame_or_empty_df(
            transaction_frame_list,
            _stress_transaction_column_list(),
        )

        stress_result_obj = StressTestResult(
            strategy_key_str=self.strategy_key_str,
            strategy_name_str=strategy_name_str,
            capital_base_float=capital_base_float,
            crisis_period_config_list=normalized_crisis_period_list,
            launch_offset_tuple=normalized_launch_offset_tuple,
            stress_metric_df=stress_metric_df,
            stress_path_df=stress_path_df,
            stress_entry_position_df=stress_entry_position_df,
            stress_transaction_df=stress_transaction_df,
            stress_strategy_map=stress_strategy_map,
        )

        if resolved_save_output_bool:
            stress_result_obj.output_dir_path = save_stress_test_results(
                stress_result_obj,
                output_dir_str=resolved_output_dir_str,
            )

        self.latest_result_obj = stress_result_obj
        return stress_result_obj


def supported_stress_test_strategy_key_list() -> tuple[str, ...]:
    return SUPPORTED_CRISIS_STRATEGY_KEY_TUPLE


def resolve_stress_launch_window(
    event_start_ts: pd.Timestamp,
    calendar_idx: pd.DatetimeIndex,
    launch_offset_int: int,
) -> tuple[pd.Timestamp | None, pd.Timestamp | None, str]:
    if int(launch_offset_int) <= 0:
        return None, None, "launch_offset_int must be positive."

    supported_calendar_idx = pd.DatetimeIndex(calendar_idx).sort_values().unique()
    if len(supported_calendar_idx) == 0:
        return None, None, "strategy calendar is empty."

    event_start_ts = pd.Timestamp(event_start_ts)
    event_start_pos_int = int(supported_calendar_idx.searchsorted(event_start_ts, side="left"))
    if event_start_pos_int >= len(supported_calendar_idx):
        return None, None, "event_start_ts is after supported history."
    if pd.Timestamp(supported_calendar_idx[event_start_pos_int]) != event_start_ts:
        return None, None, "event_start_ts is not a supported trading bar."

    event_entry_pos_int = event_start_pos_int - 1
    launch_pos_int = event_start_pos_int - int(launch_offset_int)
    if event_entry_pos_int < 0:
        return None, None, "no tradable bar exists before event_start_ts."
    if launch_pos_int <= 0:
        return None, None, "insufficient pre-launch history for launch offset."

    # *** CRITICAL*** launch_ts is selected strictly before event_start_ts by
    # trading-bar offset. event_entry_ts is the close-marked bar before the
    # event and is the latest point allowed for entering-event exposure.
    launch_ts = pd.Timestamp(supported_calendar_idx[launch_pos_int])
    event_entry_ts = pd.Timestamp(supported_calendar_idx[event_entry_pos_int])
    return launch_ts, event_entry_ts, ""


def run_stress_test_suite(
    strategy_key_str: str | None = None,
    crisis_period_list: Sequence[CrisisPeriodConfig | dict[str, str]] = CRISIS_PERIODS_LIST,
    launch_offset_tuple: Sequence[int] = DEFAULT_LAUNCH_OFFSET_TUPLE,
    strategy_spec_obj: CrisisStrategySpec | None = None,
    output_dir: str = "results",
    save_output_bool: bool = True,
    show_progress_bool: bool = False,
    show_signal_progress_bool: bool = False,
) -> StressTestResult:
    stress_test_analyzer = StressTestAnalyzer(
        strategy_key_str=strategy_key_str,
        strategy_spec_obj=strategy_spec_obj,
        crisis_period_list=crisis_period_list,
        launch_offset_tuple=launch_offset_tuple,
        output_dir_str=output_dir,
        save_output_bool=save_output_bool,
        show_progress_bool=show_progress_bool,
        show_signal_progress_bool=show_signal_progress_bool,
    )
    return stress_test_analyzer.run()


def save_stress_test_results(
    stress_result_obj: StressTestResult,
    output_dir_str: str | Path = "results",
) -> Path:
    output_path = build_research_output_path(
        output_dir_str,
        "strategy",
        stress_result_obj.strategy_key_str,
        STRESS_TEST_ANALYSIS_TYPE_STR,
    )
    output_path.mkdir(parents=True, exist_ok=True)

    _write_json(output_path / METADATA_FILENAME_STR, _build_metadata_dict(stress_result_obj))
    _write_json(output_path / RUN_INFO_FILENAME_STR, _build_run_info_dict(stress_result_obj))
    _write_json(output_path / SUMMARY_FILENAME_STR, _build_summary_dict(stress_result_obj))
    stress_result_obj.stress_metric_df.to_csv(
        output_path / STRESS_METRICS_CSV_FILENAME_STR,
        index=False,
        date_format="%Y-%m-%d",
    )
    stress_result_obj.stress_path_df.to_csv(
        output_path / STRESS_PATHS_CSV_FILENAME_STR,
        index=False,
        date_format="%Y-%m-%d",
    )
    stress_result_obj.stress_entry_position_df.to_csv(
        output_path / STRESS_ENTRY_POSITIONS_CSV_FILENAME_STR,
        index=False,
        date_format="%Y-%m-%d",
    )
    stress_result_obj.stress_transaction_df.to_csv(
        output_path / STRESS_TRANSACTIONS_CSV_FILENAME_STR,
        index=False,
        date_format="%Y-%m-%d",
    )
    (output_path / REPORT_FILENAME_STR).write_text(
        _build_report_html_str(stress_result_obj),
        encoding="utf-8",
    )

    print(f"Results saved to: {output_path.resolve()}")
    return output_path


def _coerce_launch_offset_tuple(raw_launch_offset_tuple: Sequence[int]) -> tuple[int, ...]:
    launch_offset_list: list[int] = []
    for raw_launch_offset_obj in raw_launch_offset_tuple:
        launch_offset_int = int(raw_launch_offset_obj)
        if launch_offset_int <= 0:
            raise ValueError("launch offsets must be positive trading-day counts.")
        if launch_offset_int not in launch_offset_list:
            launch_offset_list.append(launch_offset_int)
    if len(launch_offset_list) == 0:
        raise ValueError("at least one launch offset is required.")
    return tuple(launch_offset_list)


def _scenario_key_str(crisis_name_str: str, launch_offset_int: int) -> str:
    return f"{crisis_name_str}__launch_offset_{int(launch_offset_int)}"


def _normalize_result_index_df(result_df: pd.DataFrame) -> pd.DataFrame:
    normalized_result_df = result_df.copy()
    normalized_result_df.index = pd.to_datetime(normalized_result_df.index).normalize()
    return normalized_result_df


def _primary_benchmark_name_str(strategy_obj: Strategy) -> str | None:
    if not hasattr(strategy_obj, "_benchmarks") or len(strategy_obj._benchmarks) == 0:
        return None
    benchmark_name_str = str(strategy_obj._benchmarks[0])
    if strategy_obj.results is None or benchmark_name_str not in strategy_obj.results.columns:
        return None
    return benchmark_name_str


def _value_at_ts_float(value_ser: pd.Series, bar_ts: pd.Timestamp) -> float:
    normalized_ts = pd.Timestamp(bar_ts).normalize()
    return float(value_ser.loc[normalized_ts])


def _window_drawdown_pct_float(value_list: list[float]) -> float:
    if len(value_list) == 0:
        return np.nan
    value_vec = np.asarray(value_list, dtype=float)
    running_peak_vec = np.maximum.accumulate(value_vec)
    drawdown_vec = value_vec / running_peak_vec - 1.0
    return float(np.nanmin(drawdown_vec) * 100.0)


def _safe_ratio_float(numerator_float: float, denominator_float: float) -> float:
    if not np.isfinite(numerator_float) or not np.isfinite(denominator_float):
        return np.nan
    if np.isclose(denominator_float, 0.0):
        return np.nan
    return float(numerator_float / denominator_float)


def _build_stress_metric_row_dict(
    strategy_obj: Strategy,
    crisis_name_str: str,
    launch_offset_int: int,
    launch_ts: pd.Timestamp,
    event_start_ts: pd.Timestamp,
    event_entry_ts: pd.Timestamp,
    event_end_ts: pd.Timestamp,
    scenario_key_str: str,
    entry_position_df: pd.DataFrame,
    stress_transaction_df: pd.DataFrame,
) -> dict[str, object]:
    result_df = _normalize_result_index_df(strategy_obj.results)
    total_value_ser = result_df["total_value"].astype(float)
    daily_return_ser = result_df["daily_returns"].astype(float)
    launch_ts = pd.Timestamp(launch_ts).normalize()
    event_start_ts = pd.Timestamp(event_start_ts).normalize()
    event_entry_ts = pd.Timestamp(event_entry_ts).normalize()
    event_end_ts = pd.Timestamp(event_end_ts).normalize()

    capital_base_float = float(strategy_obj._capital_base)
    event_entry_value_float = _value_at_ts_float(total_value_ser, event_entry_ts)
    event_end_value_float = _value_at_ts_float(total_value_ser, event_end_ts)
    launch_window_value_list = [capital_base_float] + [
        float(value_obj)
        for value_obj in total_value_ser.loc[launch_ts:event_end_ts].to_list()
    ]
    event_window_value_list = [event_entry_value_float] + [
        float(value_obj)
        for value_obj in total_value_ser.loc[event_start_ts:event_end_ts].to_list()
    ]
    event_total_value_ser = total_value_ser.loc[event_start_ts:event_end_ts].astype(float)

    benchmark_name_str = _primary_benchmark_name_str(strategy_obj)
    benchmark_event_return_pct_float = np.nan
    benchmark_daily_return_ser = pd.Series(np.nan, index=result_df.index)
    if benchmark_name_str is not None:
        benchmark_value_ser = result_df[benchmark_name_str].astype(float)
        benchmark_entry_value_float = _value_at_ts_float(benchmark_value_ser, event_entry_ts)
        benchmark_end_value_float = _value_at_ts_float(benchmark_value_ser, event_end_ts)
        benchmark_event_return_pct_float = float(
            (benchmark_end_value_float / benchmark_entry_value_float - 1.0) * 100.0
        )
        benchmark_daily_return_ser = benchmark_value_ser.pct_change(fill_method=None)

    # *** CRITICAL*** Entering-event exposure uses event_entry_ts, the final
    # close before event_start_ts. Same-day event prices and later crisis data
    # must not influence these exposure fields.
    non_cash_position_df = entry_position_df[
        entry_position_df["cash_position_bool"] == False  # noqa: E712
    ]
    position_weight_ser = non_cash_position_df["position_weight_float"].astype(float)
    long_gross_exposure_float = float(position_weight_ser[position_weight_ser > 0.0].sum())
    short_gross_exposure_float = float(abs(position_weight_ser[position_weight_ser < 0.0].sum()))
    net_exposure_float = float(position_weight_ser.sum())
    gross_exposure_float = float(long_gross_exposure_float + short_gross_exposure_float)
    cash_weight_float = _cash_weight_float(entry_position_df)
    abs_position_weight_ser = non_cash_position_df["abs_position_weight_float"].astype(float)
    sorted_abs_position_weight_ser = abs_position_weight_ser.sort_values(ascending=False)
    entry_top1_weight_float = (
        float(sorted_abs_position_weight_ser.iloc[0])
        if len(sorted_abs_position_weight_ser) > 0
        else 0.0
    )
    entry_top5_weight_float = float(sorted_abs_position_weight_ser.head(5).sum())
    open_trade_count_entering_int = _open_trade_count_entering_int(
        transaction_df=strategy_obj.get_transactions(),
        event_entry_ts=event_entry_ts,
    )

    event_transaction_df = stress_transaction_df[
        stress_transaction_df["during_event_bool"] == True  # noqa: E712
    ]
    event_daily_return_ser = daily_return_ser.loc[event_start_ts:event_end_ts]
    worst_event_day_pct_float = (
        float(event_daily_return_ser.min() * 100.0)
        if len(event_daily_return_ser) > 0
        else np.nan
    )
    first_event_day_return_pct_float = (
        float(event_daily_return_ser.iloc[0] * 100.0)
        if len(event_daily_return_ser) > 0
        else np.nan
    )
    # *** CRITICAL*** Event risk metrics start at event_start_ts but benchmark
    # all event losses/recovery against event_entry_value_float, the P close
    # before event_start_ts. They must not use same-day or future values when
    # measuring entering-event exposure.
    if len(event_total_value_ser) > 0:
        event_relative_value_ser = event_total_value_ser / event_entry_value_float - 1.0
        entry_to_trough_pct_float = float(event_relative_value_ser.min() * 100.0)
        event_trough_pos_int = int(np.argmin(event_relative_value_ser.to_numpy(dtype=float)))
        recovered_mask_ser = event_total_value_ser >= event_entry_value_float
        recovered_by_event_end_bool = bool(event_end_value_float >= event_entry_value_float)
        if bool(recovered_mask_ser.any()):
            time_to_recover_day_count_int = int(
                np.flatnonzero(recovered_mask_ser.to_numpy(dtype=bool))[0]
            )
        else:
            time_to_recover_day_count_int = np.nan
    else:
        entry_to_trough_pct_float = np.nan
        event_trough_pos_int = np.nan
        recovered_by_event_end_bool = False
        time_to_recover_day_count_int = np.nan

    event_turnover_float = _safe_ratio_float(
        float(event_transaction_df["total_value"].astype(float).abs().sum())
        if "total_value" in event_transaction_df.columns
        else np.nan,
        event_entry_value_float,
    )
    event_commission_pct_float = _safe_ratio_float(
        float(event_transaction_df["commission"].astype(float).sum())
        if "commission" in event_transaction_df.columns
        else np.nan,
        event_entry_value_float,
    )
    event_volatility_ann_pct_float = (
        float(event_daily_return_ser.astype(float).std(ddof=1) * np.sqrt(252.0) * 100.0)
        if len(event_daily_return_ser) >= 2
        else np.nan
    )
    benchmark_event_daily_return_ser = benchmark_daily_return_ser.loc[event_start_ts:event_end_ts]
    benchmark_down_mask_ser = benchmark_event_daily_return_ser.astype(float) < 0.0
    if bool(benchmark_down_mask_ser.any()):
        strategy_down_return_ser = event_daily_return_ser.loc[benchmark_down_mask_ser.index][
            benchmark_down_mask_ser
        ].astype(float)
        benchmark_down_return_ser = benchmark_event_daily_return_ser[benchmark_down_mask_ser].astype(float)
        strategy_down_return_float = float((1.0 + strategy_down_return_ser).prod() - 1.0)
        benchmark_down_return_float = float((1.0 + benchmark_down_return_ser).prod() - 1.0)
        event_down_capture_float = _safe_ratio_float(
            strategy_down_return_float,
            benchmark_down_return_float,
        )
    else:
        event_down_capture_float = np.nan

    event_return_pct_float = float(
        (event_end_value_float / event_entry_value_float - 1.0) * 100.0
    )
    return {
        "scenario_key_str": scenario_key_str,
        "crisis_name_str": crisis_name_str,
        "launch_offset_int": int(launch_offset_int),
        "launch_ts": pd.Timestamp(launch_ts),
        "event_start_ts": pd.Timestamp(event_start_ts),
        "event_entry_ts": pd.Timestamp(event_entry_ts),
        "event_end_ts": pd.Timestamp(event_end_ts),
        "benchmark_name_str": benchmark_name_str or "",
        "launch_return_pct_float": float(
            (event_end_value_float / capital_base_float - 1.0) * 100.0
        ),
        "event_return_pct_float": event_return_pct_float,
        "benchmark_event_return_pct_float": benchmark_event_return_pct_float,
        "relative_event_return_pct_float": (
            event_return_pct_float - benchmark_event_return_pct_float
            if np.isfinite(benchmark_event_return_pct_float)
            else np.nan
        ),
        "launch_max_drawdown_pct_float": _window_drawdown_pct_float(launch_window_value_list),
        "event_max_drawdown_pct_float": _window_drawdown_pct_float(event_window_value_list),
        "worst_event_day_pct_float": worst_event_day_pct_float,
        "first_event_day_return_pct_float": first_event_day_return_pct_float,
        "entry_to_trough_pct_float": entry_to_trough_pct_float,
        "time_to_trough_day_count_int": event_trough_pos_int,
        "recovered_by_event_end_bool": recovered_by_event_end_bool,
        "time_to_recover_day_count_int": time_to_recover_day_count_int,
        "cash_weight_float": cash_weight_float,
        "long_gross_exposure_float": long_gross_exposure_float,
        "short_gross_exposure_float": short_gross_exposure_float,
        "net_exposure_float": net_exposure_float,
        "gross_exposure_float": gross_exposure_float,
        "entry_top1_weight_float": entry_top1_weight_float,
        "entry_top5_weight_float": entry_top5_weight_float,
        "event_turnover_float": event_turnover_float,
        "event_commission_pct_float": event_commission_pct_float,
        "event_volatility_ann_pct_float": event_volatility_ann_pct_float,
        "event_down_capture_float": event_down_capture_float,
        "trade_count_int": int(len(stress_transaction_df)),
        "event_trade_count_int": int(len(event_transaction_df)),
        "open_trade_count_entering_int": open_trade_count_entering_int,
    }


def _build_stress_path_df(
    strategy_obj: Strategy,
    crisis_name_str: str,
    launch_offset_int: int,
    launch_ts: pd.Timestamp,
    event_start_ts: pd.Timestamp,
    event_entry_ts: pd.Timestamp,
    event_end_ts: pd.Timestamp,
    scenario_key_str: str,
) -> pd.DataFrame:
    result_df = _normalize_result_index_df(strategy_obj.results)
    total_value_ser = result_df["total_value"].astype(float)
    capital_base_float = float(strategy_obj._capital_base)
    event_entry_value_float = _value_at_ts_float(total_value_ser, event_entry_ts)
    benchmark_name_str = _primary_benchmark_name_str(strategy_obj)
    benchmark_value_ser = (
        result_df[benchmark_name_str].astype(float)
        if benchmark_name_str is not None
        else pd.Series(np.nan, index=result_df.index)
    )
    benchmark_entry_value_float = (
        _value_at_ts_float(benchmark_value_ser, event_entry_ts)
        if benchmark_name_str is not None
        else np.nan
    )

    path_row_list: list[dict[str, object]] = [
        {
            "scenario_key_str": scenario_key_str,
            "crisis_name_str": crisis_name_str,
            "launch_offset_int": int(launch_offset_int),
            "bar_offset_int": 0,
            "bar_ts": pd.Timestamp(launch_ts),
            "launch_ts": pd.Timestamp(launch_ts),
            "event_start_ts": pd.Timestamp(event_start_ts),
            "event_entry_ts": pd.Timestamp(event_entry_ts),
            "event_end_ts": pd.Timestamp(event_end_ts),
            "strategy_name_str": strategy_obj.name,
            "benchmark_name_str": benchmark_name_str or "",
            "normalized_strategy_from_launch_float": 1.0,
            "normalized_benchmark_from_launch_float": 1.0 if benchmark_name_str else np.nan,
            "normalized_strategy_from_event_entry_float": np.nan,
            "normalized_benchmark_from_event_entry_float": np.nan,
            "drawdown_from_launch_pct_float": 0.0,
            "event_window_bool": False,
        }
    ]

    running_peak_float = capital_base_float
    for bar_offset_int, bar_ts in enumerate(total_value_ser.loc[launch_ts:event_end_ts].index, start=1):
        strategy_value_float = float(total_value_ser.loc[bar_ts])
        benchmark_value_float = float(benchmark_value_ser.loc[bar_ts])
        running_peak_float = max(running_peak_float, strategy_value_float)
        event_window_bool = pd.Timestamp(bar_ts) >= pd.Timestamp(event_start_ts)
        path_row_list.append(
            {
                "scenario_key_str": scenario_key_str,
                "crisis_name_str": crisis_name_str,
                "launch_offset_int": int(launch_offset_int),
                "bar_offset_int": int(bar_offset_int),
                "bar_ts": pd.Timestamp(bar_ts),
                "launch_ts": pd.Timestamp(launch_ts),
                "event_start_ts": pd.Timestamp(event_start_ts),
                "event_entry_ts": pd.Timestamp(event_entry_ts),
                "event_end_ts": pd.Timestamp(event_end_ts),
                "strategy_name_str": strategy_obj.name,
                "benchmark_name_str": benchmark_name_str or "",
                "normalized_strategy_from_launch_float": float(
                    strategy_value_float / capital_base_float
                ),
                "normalized_benchmark_from_launch_float": (
                    float(benchmark_value_float / capital_base_float)
                    if benchmark_name_str
                    else np.nan
                ),
                "normalized_strategy_from_event_entry_float": (
                    float(strategy_value_float / event_entry_value_float)
                    if event_window_bool
                    else np.nan
                ),
                "normalized_benchmark_from_event_entry_float": (
                    float(benchmark_value_float / benchmark_entry_value_float)
                    if benchmark_name_str and event_window_bool
                    else np.nan
                ),
                "drawdown_from_launch_pct_float": float(
                    (strategy_value_float / running_peak_float - 1.0) * 100.0
                ),
                "event_window_bool": bool(event_window_bool),
            }
        )
    return pd.DataFrame(path_row_list, columns=_stress_path_column_list())


def _build_entry_position_df(
    strategy_obj: Strategy,
    pricing_data_df: pd.DataFrame,
    crisis_name_str: str,
    launch_offset_int: int,
    launch_ts: pd.Timestamp,
    event_start_ts: pd.Timestamp,
    event_entry_ts: pd.Timestamp,
    event_end_ts: pd.Timestamp,
    scenario_key_str: str,
) -> pd.DataFrame:
    result_df = _normalize_result_index_df(strategy_obj.results)
    event_entry_ts = pd.Timestamp(event_entry_ts).normalize()
    total_value_float = float(result_df.loc[event_entry_ts, "total_value"])
    cash_float = float(result_df.loc[event_entry_ts, "cash"])
    close_price_ser = pricing_data_df.loc[event_entry_ts, (slice(None), "Close")]
    close_price_ser.index = close_price_ser.index.get_level_values(0)
    transaction_df = strategy_obj.get_transactions().copy()
    if len(transaction_df) == 0:
        active_position_ser = pd.Series(dtype=float)
    else:
        transaction_df["bar"] = pd.to_datetime(transaction_df["bar"]).dt.normalize()
        # *** CRITICAL*** Entry positions are reconstructed only from
        # transactions with bar <= event_entry_ts and marked on event_entry_ts
        # close, the last close before the crisis window starts.
        pre_event_transaction_df = transaction_df[transaction_df["bar"] <= event_entry_ts]
        if len(pre_event_transaction_df) == 0:
            active_position_ser = pd.Series(dtype=float)
        else:
            active_position_ser = (
                pre_event_transaction_df.groupby("asset")["amount"].sum().astype(float)
            )
            active_position_ser = active_position_ser[~np.isclose(active_position_ser, 0.0)]

    row_list: list[dict[str, object]] = []
    for asset_rank_int, asset_str in enumerate(
        active_position_ser.abs().sort_values(ascending=False).index,
        start=1,
    ):
        position_share_float = float(active_position_ser.loc[asset_str])
        close_price_float = float(close_price_ser.loc[asset_str])
        position_value_float = float(position_share_float * close_price_float)
        position_weight_float = float(position_value_float / total_value_float)
        row_list.append(
            _entry_position_row_dict(
                scenario_key_str=scenario_key_str,
                crisis_name_str=crisis_name_str,
                launch_offset_int=launch_offset_int,
                launch_ts=launch_ts,
                event_start_ts=event_start_ts,
                event_entry_ts=event_entry_ts,
                event_end_ts=event_end_ts,
                asset_rank_int=asset_rank_int,
                asset_str=str(asset_str),
                position_share_float=position_share_float,
                entry_close_price_float=close_price_float,
                position_value_float=position_value_float,
                position_weight_float=position_weight_float,
                cash_position_bool=False,
            )
        )

    row_list.append(
        _entry_position_row_dict(
            scenario_key_str=scenario_key_str,
            crisis_name_str=crisis_name_str,
            launch_offset_int=launch_offset_int,
            launch_ts=launch_ts,
            event_start_ts=event_start_ts,
            event_entry_ts=event_entry_ts,
            event_end_ts=event_end_ts,
            asset_rank_int=0,
            asset_str="Cash",
            position_share_float=np.nan,
            entry_close_price_float=np.nan,
            position_value_float=cash_float,
            position_weight_float=float(cash_float / total_value_float),
            cash_position_bool=True,
        )
    )
    return pd.DataFrame(row_list, columns=_stress_entry_position_column_list())


def _entry_position_row_dict(
    scenario_key_str: str,
    crisis_name_str: str,
    launch_offset_int: int,
    launch_ts: pd.Timestamp,
    event_start_ts: pd.Timestamp,
    event_entry_ts: pd.Timestamp,
    event_end_ts: pd.Timestamp,
    asset_rank_int: int,
    asset_str: str,
    position_share_float: float,
    entry_close_price_float: float,
    position_value_float: float,
    position_weight_float: float,
    cash_position_bool: bool,
) -> dict[str, object]:
    return {
        "scenario_key_str": scenario_key_str,
        "crisis_name_str": crisis_name_str,
        "launch_offset_int": int(launch_offset_int),
        "launch_ts": pd.Timestamp(launch_ts),
        "event_start_ts": pd.Timestamp(event_start_ts),
        "event_entry_ts": pd.Timestamp(event_entry_ts),
        "event_end_ts": pd.Timestamp(event_end_ts),
        "asset_rank_int": int(asset_rank_int),
        "asset_str": asset_str,
        "position_share_float": position_share_float,
        "entry_close_price_float": entry_close_price_float,
        "position_value_float": position_value_float,
        "position_weight_float": position_weight_float,
        "abs_position_weight_float": abs(float(position_weight_float)),
        "cash_position_bool": bool(cash_position_bool),
    }


def _build_stress_transaction_df(
    strategy_obj: Strategy,
    crisis_name_str: str,
    launch_offset_int: int,
    launch_ts: pd.Timestamp,
    event_start_ts: pd.Timestamp,
    event_entry_ts: pd.Timestamp,
    event_end_ts: pd.Timestamp,
    scenario_key_str: str,
) -> pd.DataFrame:
    transaction_df = strategy_obj.get_transactions().copy()
    if len(transaction_df) == 0:
        return pd.DataFrame(columns=_stress_transaction_column_list())

    transaction_df["bar"] = pd.to_datetime(transaction_df["bar"]).dt.normalize()
    transaction_df = transaction_df[
        (transaction_df["bar"] >= pd.Timestamp(launch_ts).normalize())
        & (transaction_df["bar"] <= pd.Timestamp(event_end_ts).normalize())
    ].copy()
    if len(transaction_df) == 0:
        return pd.DataFrame(columns=_stress_transaction_column_list())

    transaction_df.insert(0, "scenario_key_str", scenario_key_str)
    transaction_df.insert(1, "crisis_name_str", crisis_name_str)
    transaction_df.insert(2, "launch_offset_int", int(launch_offset_int))
    transaction_df.insert(3, "launch_ts", pd.Timestamp(launch_ts))
    transaction_df.insert(4, "event_start_ts", pd.Timestamp(event_start_ts))
    transaction_df.insert(5, "event_entry_ts", pd.Timestamp(event_entry_ts))
    transaction_df.insert(6, "event_end_ts", pd.Timestamp(event_end_ts))
    transaction_df["during_event_bool"] = (
        transaction_df["bar"] >= pd.Timestamp(event_start_ts).normalize()
    ) & (
        transaction_df["bar"] <= pd.Timestamp(event_end_ts).normalize()
    )
    return transaction_df.reindex(columns=_stress_transaction_column_list())


def _cash_weight_float(entry_position_df: pd.DataFrame) -> float:
    cash_position_df = entry_position_df[
        entry_position_df["cash_position_bool"] == True  # noqa: E712
    ]
    if len(cash_position_df) == 0:
        return np.nan
    return float(cash_position_df.iloc[0]["position_weight_float"])


def _open_trade_count_entering_int(
    transaction_df: pd.DataFrame,
    event_entry_ts: pd.Timestamp,
) -> int:
    if transaction_df is None or len(transaction_df) == 0:
        return 0
    pre_event_transaction_df = transaction_df.copy()
    pre_event_transaction_df["bar"] = pd.to_datetime(pre_event_transaction_df["bar"]).dt.normalize()
    pre_event_transaction_df = pre_event_transaction_df[
        pre_event_transaction_df["bar"] <= pd.Timestamp(event_entry_ts).normalize()
    ]
    if len(pre_event_transaction_df) == 0:
        return 0
    open_trade_count_int = 0
    for _trade_id_obj, trade_group_df in pre_event_transaction_df.groupby("trade_id"):
        position_by_asset_ser = trade_group_df.groupby("asset")["amount"].sum().astype(float)
        if bool((~np.isclose(position_by_asset_ser, 0.0)).any()):
            open_trade_count_int += 1
    return int(open_trade_count_int)


def _concat_or_empty_df(
    row_list: list[dict[str, object]],
    column_list: list[str],
) -> pd.DataFrame:
    if len(row_list) == 0:
        return pd.DataFrame(columns=column_list)
    return pd.DataFrame(row_list, columns=column_list)


def _concat_frame_or_empty_df(
    frame_list: list[pd.DataFrame],
    column_list: list[str],
) -> pd.DataFrame:
    active_frame_list = [frame_df for frame_df in frame_list if frame_df is not None and len(frame_df) > 0]
    if len(active_frame_list) == 0:
        return pd.DataFrame(columns=column_list)
    return pd.concat(active_frame_list, ignore_index=True).reindex(columns=column_list)


def _stress_metric_column_list() -> list[str]:
    return [
        "scenario_key_str",
        "crisis_name_str",
        "launch_offset_int",
        "launch_ts",
        "event_start_ts",
        "event_entry_ts",
        "event_end_ts",
        "benchmark_name_str",
        "launch_return_pct_float",
        "event_return_pct_float",
        "benchmark_event_return_pct_float",
        "relative_event_return_pct_float",
        "launch_max_drawdown_pct_float",
        "event_max_drawdown_pct_float",
        "worst_event_day_pct_float",
        "first_event_day_return_pct_float",
        "entry_to_trough_pct_float",
        "time_to_trough_day_count_int",
        "recovered_by_event_end_bool",
        "time_to_recover_day_count_int",
        "cash_weight_float",
        "long_gross_exposure_float",
        "short_gross_exposure_float",
        "net_exposure_float",
        "gross_exposure_float",
        "entry_top1_weight_float",
        "entry_top5_weight_float",
        "event_turnover_float",
        "event_commission_pct_float",
        "event_volatility_ann_pct_float",
        "event_down_capture_float",
        "trade_count_int",
        "event_trade_count_int",
        "open_trade_count_entering_int",
    ]


def _stress_path_column_list() -> list[str]:
    return [
        "scenario_key_str",
        "crisis_name_str",
        "launch_offset_int",
        "bar_offset_int",
        "bar_ts",
        "launch_ts",
        "event_start_ts",
        "event_entry_ts",
        "event_end_ts",
        "strategy_name_str",
        "benchmark_name_str",
        "normalized_strategy_from_launch_float",
        "normalized_benchmark_from_launch_float",
        "normalized_strategy_from_event_entry_float",
        "normalized_benchmark_from_event_entry_float",
        "drawdown_from_launch_pct_float",
        "event_window_bool",
    ]


def _stress_entry_position_column_list() -> list[str]:
    return [
        "scenario_key_str",
        "crisis_name_str",
        "launch_offset_int",
        "launch_ts",
        "event_start_ts",
        "event_entry_ts",
        "event_end_ts",
        "asset_rank_int",
        "asset_str",
        "position_share_float",
        "entry_close_price_float",
        "position_value_float",
        "position_weight_float",
        "abs_position_weight_float",
        "cash_position_bool",
    ]


def _stress_transaction_column_list() -> list[str]:
    return [
        "scenario_key_str",
        "crisis_name_str",
        "launch_offset_int",
        "launch_ts",
        "event_start_ts",
        "event_entry_ts",
        "event_end_ts",
        "during_event_bool",
        "trade_id",
        "bar",
        "asset",
        "amount",
        "price",
        "total_value",
        "order_id",
        "commission",
    ]


def _json_default(value_obj):
    if isinstance(value_obj, Path):
        return str(value_obj)
    if isinstance(value_obj, pd.Timestamp):
        return value_obj.isoformat()
    if isinstance(value_obj, np.integer):
        return int(value_obj)
    if isinstance(value_obj, np.floating):
        return float(value_obj)
    return value_obj


def _write_json(json_path: Path, payload_dict: dict[str, object]) -> None:
    json_path.write_text(
        json.dumps(payload_dict, indent=2, sort_keys=True, default=_json_default),
        encoding="utf-8",
    )


def _build_run_info_dict(stress_result_obj: StressTestResult) -> dict[str, object]:
    return {
        "entity_type": "strategy",
        "entity_id": stress_result_obj.strategy_key_str,
        "analysis_type": STRESS_TEST_ANALYSIS_TYPE_STR,
        "parameters": {
            "stress_type": "historical_pre_crisis_launch",
            "capital": float(stress_result_obj.capital_base_float),
            "launch_offsets": list(stress_result_obj.launch_offset_tuple),
            "crisis_windows": [
                {
                    "name": crisis_period_config.crisis_name_str,
                    "start_date": crisis_period_config.start_date_str,
                    "end_date": crisis_period_config.end_date_str,
                }
                for crisis_period_config in stress_result_obj.crisis_period_config_list
            ],
        },
    }


def _build_metadata_dict(stress_result_obj: StressTestResult) -> dict[str, object]:
    return {
        "artifact_type": STRESS_TEST_ANALYSIS_TYPE_STR,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "strategy_key": stress_result_obj.strategy_key_str,
        "strategy_name": stress_result_obj.strategy_name_str,
        "capital_base": float(stress_result_obj.capital_base_float),
        "configured_crisis_count": int(len(stress_result_obj.crisis_period_config_list)),
        "launch_offsets": list(stress_result_obj.launch_offset_tuple),
        "evaluated_scenario_count": int(len(stress_result_obj.stress_metric_df)),
    }


def _metric_extreme_row_ser(
    metric_df: pd.DataFrame,
    metric_column_str: str,
    *,
    largest_bool: bool,
) -> pd.Series | None:
    if metric_column_str not in metric_df.columns or len(metric_df) == 0:
        return None
    clean_metric_df = metric_df[pd.to_numeric(metric_df[metric_column_str], errors="coerce").notna()]
    if len(clean_metric_df) == 0:
        return None
    sorted_metric_df = clean_metric_df.sort_values(
        by=[metric_column_str, "event_start_ts", "crisis_name_str", "launch_offset_int"],
        ascending=[not largest_bool, True, True, True],
        kind="mergesort",
    )
    return sorted_metric_df.iloc[0]


def _add_extreme_summary_fields(
    summary_dict: dict[str, object],
    metric_df: pd.DataFrame,
    metric_column_str: str,
    prefix_str: str,
    *,
    largest_bool: bool,
) -> None:
    metric_row_ser = _metric_extreme_row_ser(
        metric_df,
        metric_column_str,
        largest_bool=largest_bool,
    )
    if metric_row_ser is None:
        return
    summary_dict[f"{prefix_str}_float"] = float(metric_row_ser[metric_column_str])
    summary_dict[f"{prefix_str}_crisis_name_str"] = str(metric_row_ser["crisis_name_str"])
    summary_dict[f"{prefix_str}_launch_offset_int"] = int(metric_row_ser["launch_offset_int"])
    summary_dict[f"{prefix_str}_scenario_key_str"] = str(metric_row_ser["scenario_key_str"])


def _build_summary_dict(stress_result_obj: StressTestResult) -> dict[str, object]:
    metric_df = stress_result_obj.stress_metric_df
    summary_dict: dict[str, object] = {
        "analysis_type_str": STRESS_TEST_ANALYSIS_TYPE_STR,
        "scenario_count_int": int(len(metric_df)),
        "launch_offsets": list(stress_result_obj.launch_offset_tuple),
    }
    if len(metric_df) == 0:
        return summary_dict
    _add_extreme_summary_fields(
        summary_dict,
        metric_df,
        "event_return_pct_float",
        "worst_event_return_pct",
        largest_bool=False,
    )
    _add_extreme_summary_fields(
        summary_dict,
        metric_df,
        "relative_event_return_pct_float",
        "worst_relative_event_return_pct",
        largest_bool=False,
    )
    _add_extreme_summary_fields(
        summary_dict,
        metric_df,
        "event_max_drawdown_pct_float",
        "worst_event_max_drawdown_pct",
        largest_bool=False,
    )
    _add_extreme_summary_fields(
        summary_dict,
        metric_df,
        "first_event_day_return_pct_float",
        "worst_first_event_day_return_pct",
        largest_bool=False,
    )
    _add_extreme_summary_fields(
        summary_dict,
        metric_df,
        "gross_exposure_float",
        "max_entering_gross_exposure",
        largest_bool=True,
    )
    _add_extreme_summary_fields(
        summary_dict,
        metric_df,
        "entry_top1_weight_float",
        "max_entry_top1_weight",
        largest_bool=True,
    )
    summary_dict["unrecovered_scenario_count_int"] = int(
        (metric_df["recovered_by_event_end_bool"] == False).sum()  # noqa: E712
    )
    return summary_dict


def _fmt_pct(value_obj) -> str:
    try:
        value_float = float(value_obj)
    except (TypeError, ValueError):
        return ""
    if not np.isfinite(value_float):
        return ""
    return f"{value_float:+,.2f}%"


def _fmt_ratio_pct(value_obj) -> str:
    try:
        value_float = float(value_obj)
    except (TypeError, ValueError):
        return ""
    if not np.isfinite(value_float):
        return ""
    return f"{value_float * 100.0:+,.2f}%"


def _fmt_unsigned_ratio_pct(value_obj) -> str:
    try:
        value_float = float(value_obj)
    except (TypeError, ValueError):
        return ""
    if not np.isfinite(value_float):
        return ""
    return f"{value_float * 100.0:,.2f}%"


def _fmt_multiple(value_obj) -> str:
    try:
        value_float = float(value_obj)
    except (TypeError, ValueError):
        return ""
    if not np.isfinite(value_float):
        return ""
    return f"{value_float:,.2f}x"


def _fmt_float(value_obj, decimals_int: int = 2) -> str:
    try:
        value_float = float(value_obj)
    except (TypeError, ValueError):
        return ""
    if not np.isfinite(value_float):
        return ""
    return f"{value_float:,.{decimals_int}f}"


def _fmt_date(value_obj) -> str:
    if value_obj is None or pd.isna(value_obj):
        return ""
    return str(pd.Timestamp(value_obj).date())


def _fmt_dollar(value_obj) -> str:
    try:
        value_float = float(value_obj)
    except (TypeError, ValueError):
        return ""
    if not np.isfinite(value_float):
        return ""
    return f"${value_float:,.2f}"


def _fmt_bool(value_obj) -> str:
    if pd.isna(value_obj):
        return ""
    return "Yes" if bool(value_obj) else "No"


def _fmt_int(value_obj) -> str:
    if pd.isna(value_obj):
        return ""
    return str(int(value_obj))


def _class_for_signed_float(value_obj) -> str:
    try:
        value_float = float(value_obj)
    except (TypeError, ValueError):
        return ""
    if not np.isfinite(value_float):
        return ""
    return "pos" if value_float >= 0.0 else "neg"


def _wrap_card_html(card_body_html_str: str, card_class_str: str = "") -> str:
    card_class_attr_str = "card"
    if card_class_str:
        card_class_attr_str += f" {card_class_str}"
    return f'<section class="{card_class_attr_str}">{card_body_html_str}</section>'


def _build_kpi_card_html(
    title_str: str,
    value_str: str,
    meaning_str: str,
    context_str: str,
    class_str: str = "",
) -> str:
    value_class_str = "kpi-value"
    if class_str:
        value_class_str += f" {class_str}"
    return (
        '<div class="kpi-card">'
        f"<div class=\"kpi-label\">{html.escape(title_str)}</div>"
        f"<div class=\"{value_class_str}\">{html.escape(value_str)}</div>"
        f"<div class=\"kpi-meaning\">{html.escape(meaning_str)}</div>"
        f"<div class=\"kpi-note\">{html.escape(context_str)}</div>"
        "</div>"
    )


def _scenario_note_str(summary_dict: dict[str, object], prefix_str: str) -> str:
    crisis_name_str = str(summary_dict.get(f"{prefix_str}_crisis_name_str", ""))
    launch_offset_obj = summary_dict.get(f"{prefix_str}_launch_offset_int")
    if crisis_name_str == "" or pd.isna(launch_offset_obj):
        return ""
    return f"{crisis_name_str}, offset {int(launch_offset_obj)}"


def _median_metric_float(metric_df: pd.DataFrame, metric_column_str: str) -> float:
    if metric_df is None or metric_column_str not in metric_df.columns:
        return np.nan
    value_vec = pd.to_numeric(metric_df[metric_column_str], errors="coerce").dropna()
    if len(value_vec) == 0:
        return np.nan
    return float(value_vec.median())


def _build_verdict_html(stress_result_obj: StressTestResult) -> str:
    """One plain-English summary line so the report leads with the conclusion."""
    metric_df = stress_result_obj.stress_metric_df
    if metric_df is None or len(metric_df) == 0:
        return (
            '<section class="stress-verdict stress-verdict-empty">'
            "<p>No stress scenarios were evaluated, so there is no verdict to report. "
            "This usually means there was not enough pre-crisis history for the chosen "
            "launch offsets.</p>"
            "</section>"
        )

    summary_dict = _build_summary_dict(stress_result_obj)
    scenario_count_int = int(len(metric_df))
    crisis_count_int = int(metric_df["crisis_name_str"].nunique())
    offset_list = sorted(int(offset_obj) for offset_obj in stress_result_obj.launch_offset_tuple)
    offset_str = ", ".join(f"{offset_int}d" for offset_int in offset_list)

    worst_event_return_float = summary_dict.get("worst_event_return_pct_float")
    worst_event_note_str = _scenario_note_str(summary_dict, "worst_event_return_pct")
    median_event_return_float = _median_metric_float(metric_df, "event_return_pct_float")
    worst_drawdown_float = summary_dict.get("worst_event_max_drawdown_pct_float")
    worst_drawdown_note_str = _scenario_note_str(summary_dict, "worst_event_max_drawdown_pct")
    median_entry_gross_float = _median_metric_float(metric_df, "gross_exposure_float")
    worst_top1_float = summary_dict.get("max_entry_top1_weight_float")
    worst_top1_note_str = _scenario_note_str(summary_dict, "max_entry_top1_weight")
    unrecovered_count_int = int(summary_dict.get("unrecovered_scenario_count_int", 0))

    headline_str = (
        f"Across {scenario_count_int} scenarios "
        f"({crisis_count_int} crises x offsets {offset_str}): "
        f"worst entering-event return was {_fmt_pct(worst_event_return_float)}"
    )
    if worst_event_note_str:
        headline_str += f" ({worst_event_note_str})"
    headline_str += f", median {_fmt_pct(median_event_return_float)}."

    detail_str = (
        f"Worst intra-event drawdown {_fmt_pct(worst_drawdown_float)}"
    )
    if worst_drawdown_note_str:
        detail_str += f" ({worst_drawdown_note_str})"
    detail_str += (
        f". The book typically entered crises with "
        f"{_fmt_unsigned_ratio_pct(median_entry_gross_float)} median gross exposure"
    )
    if worst_top1_float is not None and np.isfinite(float(worst_top1_float)):
        detail_str += f" and a worst single-name weight of {_fmt_unsigned_ratio_pct(worst_top1_float)}"
        if worst_top1_note_str:
            detail_str += f" ({worst_top1_note_str})"
    detail_str += "."

    if unrecovered_count_int > 0:
        recovery_str = (
            f"{unrecovered_count_int} of {scenario_count_int} scenarios had not recovered "
            "to their pre-event mark by event end."
        )
    else:
        recovery_str = "All scenarios recovered to their pre-event mark by event end."

    return (
        '<section class="stress-verdict">'
        '<div class="report-eyebrow">Verdict</div>'
        f"<p class=\"stress-verdict-headline\">{html.escape(headline_str)}</p>"
        f"<p class=\"stress-verdict-detail\">{html.escape(detail_str)}</p>"
        f"<p class=\"stress-verdict-detail\">{html.escape(recovery_str)}</p>"
        "</section>"
    )


def _build_stress_kpi_grid_html(stress_result_obj: StressTestResult) -> str:
    summary_dict = _build_summary_dict(stress_result_obj)
    kpi_card_list = [
        _build_kpi_card_html(
            "Worst Event Return",
            _fmt_pct(summary_dict.get("worst_event_return_pct_float")),
            "From entering the crisis to event end.",
            _scenario_note_str(summary_dict, "worst_event_return_pct"),
            _class_for_signed_float(summary_dict.get("worst_event_return_pct_float")),
        ),
        _build_kpi_card_html(
            "Worst Relative Return",
            _fmt_pct(summary_dict.get("worst_relative_event_return_pct_float")),
            "Worst underperformance versus benchmark.",
            _scenario_note_str(summary_dict, "worst_relative_event_return_pct"),
            _class_for_signed_float(summary_dict.get("worst_relative_event_return_pct_float")),
        ),
        _build_kpi_card_html(
            "Worst Event Drawdown",
            _fmt_pct(summary_dict.get("worst_event_max_drawdown_pct_float")),
            "How bad the intra-event pain got.",
            _scenario_note_str(summary_dict, "worst_event_max_drawdown_pct"),
            "neg",
        ),
        _build_kpi_card_html(
            "Worst First Event Day",
            _fmt_pct(summary_dict.get("worst_first_event_day_return_pct_float")),
            "What happened when the event began.",
            _scenario_note_str(summary_dict, "worst_first_event_day_return_pct"),
            _class_for_signed_float(summary_dict.get("worst_first_event_day_return_pct_float")),
        ),
        _build_kpi_card_html(
            "Max Entry Gross",
            _fmt_ratio_pct(summary_dict.get("max_entering_gross_exposure_float")),
            "How exposed the strategy was before the event.",
            _scenario_note_str(summary_dict, "max_entering_gross_exposure"),
            "",
        ),
        _build_kpi_card_html(
            "Max Top-1 Weight",
            _fmt_ratio_pct(summary_dict.get("max_entry_top1_weight_float")),
            "Single-position concentration before the event.",
            _scenario_note_str(summary_dict, "max_entry_top1_weight"),
            "",
        ),
        _build_kpi_card_html(
            "Unrecovered",
            _fmt_int(summary_dict.get("unrecovered_scenario_count_int")),
            "Scenarios that did not recover by event end.",
            "Scenarios ending below P close",
            "neg" if int(summary_dict.get("unrecovered_scenario_count_int", 0)) > 0 else "",
        ),
        _build_kpi_card_html(
            "Scenarios",
            _fmt_int(summary_dict.get("scenario_count_int")),
            "Number of evaluated crisis and offset rows.",
            "Crisis x launch offset rows",
            "",
        ),
    ]
    return f'<div class="kpi-grid">{"".join(kpi_card_list)}</div>'


def _stress_full_column_spec_list() -> list[tuple[str, str]]:
    return [
        ("crisis_name_str", "Crisis"),
        ("launch_offset_int", "Launch Offset"),
        ("launch_ts", "Launch"),
        ("event_start_ts", "Event Start"),
        ("event_entry_ts", "Entry Mark"),
        ("event_end_ts", "Event End"),
        ("launch_return_pct_float", "Launch Return"),
        ("event_return_pct_float", "Event Return"),
        ("benchmark_event_return_pct_float", "Benchmark Event"),
        ("relative_event_return_pct_float", "Relative Event"),
        ("event_max_drawdown_pct_float", "Event Max DD"),
        ("worst_event_day_pct_float", "Worst Event Day"),
        ("first_event_day_return_pct_float", "First Event Day"),
        ("entry_to_trough_pct_float", "Entry To Trough"),
        ("recovered_by_event_end_bool", "Recovered"),
        ("time_to_recover_day_count_int", "Recovery Bars"),
        ("gross_exposure_float", "Entry Gross"),
        ("cash_weight_float", "Entry Cash"),
        ("entry_top1_weight_float", "Top-1 Weight"),
        ("entry_top5_weight_float", "Top-5 Weight"),
        ("event_turnover_float", "Event Turnover"),
        ("event_commission_pct_float", "Event Commission"),
        ("event_volatility_ann_pct_float", "Event Vol Ann"),
        ("event_down_capture_float", "Down Capture"),
        ("event_trade_count_int", "Event Trades"),
    ]


# *** Stress Matrix groups ***
# The full matrix is 25 columns wide; split it into themed sub-tables so the
# appendix is readable. Crisis + Offset repeat as the identity key in each
# group so any sub-table can be read on its own.
_STRESS_MATRIX_IDENTITY_SPEC_LIST: list[tuple[str, str]] = [
    ("crisis_name_str", "Crisis"),
    ("launch_offset_int", "Offset"),
]


def _stress_matrix_group_spec_list() -> list[tuple[str, list[tuple[str, str]]]]:
    return [
        (
            "Timing",
            [
                ("launch_ts", "Launch"),
                ("event_start_ts", "Event Start"),
                ("event_entry_ts", "Entry Mark"),
                ("event_end_ts", "Event End"),
            ],
        ),
        (
            "Returns",
            [
                ("launch_return_pct_float", "Launch Return"),
                ("event_return_pct_float", "Event Return"),
                ("benchmark_event_return_pct_float", "Benchmark Event"),
                ("relative_event_return_pct_float", "Relative Event"),
            ],
        ),
        (
            "Risk",
            [
                ("event_max_drawdown_pct_float", "Event Max DD"),
                ("worst_event_day_pct_float", "Worst Event Day"),
                ("first_event_day_return_pct_float", "First Event Day"),
                ("entry_to_trough_pct_float", "Entry To Trough"),
                ("event_volatility_ann_pct_float", "Event Vol Ann"),
                ("event_down_capture_float", "Down Capture"),
                ("recovered_by_event_end_bool", "Recovered"),
                ("time_to_recover_day_count_int", "Recovery Bars"),
            ],
        ),
        (
            "Entry Exposure",
            [
                ("gross_exposure_float", "Entry Gross"),
                ("net_exposure_float", "Entry Net"),
                ("cash_weight_float", "Entry Cash"),
                ("entry_top1_weight_float", "Top-1 Weight"),
                ("entry_top5_weight_float", "Top-5 Weight"),
            ],
        ),
        (
            "Trading",
            [
                ("event_turnover_float", "Event Turnover"),
                ("event_commission_pct_float", "Event Commission"),
                ("trade_count_int", "Trades"),
                ("event_trade_count_int", "Event Trades"),
                ("open_trade_count_entering_int", "Open Trades Entering"),
            ],
        ),
    ]


def _stress_metric_cell_html(column_name_str: str, value_obj) -> str:
    class_str = ""
    if column_name_str.endswith("_ts"):
        cell_text_str = _fmt_date(value_obj)
    elif column_name_str.endswith("_bool"):
        cell_text_str = _fmt_bool(value_obj)
    elif column_name_str.endswith("_pct_float"):
        if column_name_str == "event_commission_pct_float":
            cell_text_str = _fmt_unsigned_ratio_pct(value_obj)
        else:
            cell_text_str = _fmt_pct(value_obj)
        if column_name_str in {
            "event_return_pct_float",
            "relative_event_return_pct_float",
            "first_event_day_return_pct_float",
        }:
            class_str = _class_for_signed_float(value_obj)
        elif column_name_str in {
            "event_max_drawdown_pct_float",
            "worst_event_day_pct_float",
            "entry_to_trough_pct_float",
        }:
            class_str = "neg"
    elif column_name_str.endswith("_weight_float") or column_name_str.endswith("_exposure_float"):
        cell_text_str = _fmt_ratio_pct(value_obj)
    elif column_name_str == "event_turnover_float":
        cell_text_str = _fmt_unsigned_ratio_pct(value_obj)
    elif column_name_str == "event_down_capture_float":
        cell_text_str = _fmt_multiple(value_obj)
    elif column_name_str.endswith("_int"):
        cell_text_str = "" if pd.isna(value_obj) else str(int(value_obj))
    else:
        cell_text_str = "" if pd.isna(value_obj) else str(value_obj)
    class_attr_str = f' class="{class_str}"' if class_str else ""
    return f"<td{class_attr_str}>{html.escape(cell_text_str)}</td>"


def _format_metric_table_html(
    metric_df: pd.DataFrame,
    column_spec_list: list[tuple[str, str]],
) -> str:
    if metric_df is None or len(metric_df) == 0:
        return "<p>No stress scenarios were evaluated.</p>"
    header_html_str = "".join(f"<th>{html.escape(label_str)}</th>" for _, label_str in column_spec_list)
    row_html_list: list[str] = []
    for _, row_ser in metric_df.iterrows():
        cell_html_list = [
            _stress_metric_cell_html(column_name_str, row_ser[column_name_str])
            for column_name_str, _label_str in column_spec_list
            if column_name_str in row_ser.index
        ]
        row_html_list.append("<tr>" + "".join(cell_html_list) + "</tr>")
    return (
        f"<table><thead><tr>{header_html_str}</tr></thead>"
        f"<tbody>{''.join(row_html_list)}</tbody></table>"
    )


def _format_stress_metric_table_html(stress_metric_df: pd.DataFrame) -> str:
    return _format_metric_table_html(stress_metric_df, _stress_full_column_spec_list())


def _build_stress_matrix_grouped_html(stress_metric_df: pd.DataFrame) -> str:
    if stress_metric_df is None or len(stress_metric_df) == 0:
        return "<p>No stress scenarios were evaluated.</p>"
    ordered_metric_df = stress_metric_df.sort_values(
        by=["event_start_ts", "crisis_name_str", "launch_offset_int"],
        kind="mergesort",
    )
    group_html_list: list[str] = []
    for group_title_str, group_column_spec_list in _stress_matrix_group_spec_list():
        present_column_spec_list = [
            (column_name_str, label_str)
            for column_name_str, label_str in group_column_spec_list
            if column_name_str in ordered_metric_df.columns
        ]
        if len(present_column_spec_list) == 0:
            continue
        full_column_spec_list = _STRESS_MATRIX_IDENTITY_SPEC_LIST + present_column_spec_list
        table_html_str = _format_metric_table_html(ordered_metric_df, full_column_spec_list)
        group_html_list.append(
            f'<div class="stress-matrix-group">'
            f"<h4>{html.escape(group_title_str)}</h4>"
            f'<div class="scroll">{table_html_str}</div>'
            "</div>"
        )
    return f'<div class="stress-matrix-groups">{"".join(group_html_list)}</div>'


def _format_worst_case_table_html(stress_metric_df: pd.DataFrame) -> str:
    if stress_metric_df is None or len(stress_metric_df) == 0:
        return "<p>No worst cases are available.</p>"
    worst_case_df = stress_metric_df.sort_values(
        by=["event_return_pct_float", "event_max_drawdown_pct_float"],
        ascending=[True, True],
        kind="mergesort",
    ).head(10)
    return _format_stress_metric_table_html(worst_case_df)


def _format_metric_value_str(metric_column_str: str, value_obj) -> str:
    if metric_column_str.endswith("_ts"):
        return _fmt_date(value_obj)
    if metric_column_str.endswith("_bool"):
        return _fmt_bool(value_obj)
    if metric_column_str == "event_commission_pct_float":
        return _fmt_unsigned_ratio_pct(value_obj)
    if metric_column_str.endswith("_pct_float"):
        return _fmt_pct(value_obj)
    if metric_column_str in {
        "cash_weight_float",
        "long_gross_exposure_float",
        "short_gross_exposure_float",
        "gross_exposure_float",
        "entry_top1_weight_float",
        "entry_top5_weight_float",
        "event_turnover_float",
    }:
        return _fmt_unsigned_ratio_pct(value_obj)
    if metric_column_str == "net_exposure_float":
        return _fmt_ratio_pct(value_obj)
    if metric_column_str == "event_down_capture_float":
        return _fmt_multiple(value_obj)
    if metric_column_str.endswith("_int"):
        return _fmt_int(value_obj)
    if metric_column_str.endswith("_float"):
        return _fmt_float(value_obj)
    return "" if pd.isna(value_obj) else str(value_obj)


# *** Heatmap color-scale floors ***
# Minimum magnitude that maps to full color intensity for each heatmap metric.
# The actual scale is max(floor, observed max-abs in the column) so a calm
# sample does not saturate every cell, while a severe crisis is not clipped: a
# -35% and a -60% drawdown stay visually distinct instead of both maxing out.
_HEATMAP_SCALE_FLOOR_DICT: dict[str, float] = {
    "event_return_pct_float": 10.0,
    "relative_event_return_pct_float": 10.0,
    "event_max_drawdown_pct_float": 10.0,
    "entry_top1_weight_float": 0.25,
    "gross_exposure_float": 1.0,
}


def _heatmap_scale_max_float(stress_metric_df: pd.DataFrame, metric_column_str: str) -> float:
    """Data-driven full-intensity magnitude for one heatmap metric.

    Returns max(per-metric floor, largest finite |value| in the column) so the
    color ramp adapts to the data instead of clipping at a fixed cap.
    """
    floor_float = float(_HEATMAP_SCALE_FLOOR_DICT.get(metric_column_str, 1.0))
    if stress_metric_df is None or metric_column_str not in stress_metric_df.columns:
        return floor_float
    value_vec = pd.to_numeric(stress_metric_df[metric_column_str], errors="coerce").abs()
    observed_max_float = float(value_vec.max()) if value_vec.notna().any() else 0.0
    if not np.isfinite(observed_max_float):
        observed_max_float = 0.0
    return max(floor_float, observed_max_float)


def _heatmap_cell_style_str(
    metric_column_str: str,
    value_obj,
    scale_max_float: float = 1.0,
) -> str:
    try:
        value_float = float(value_obj)
    except (TypeError, ValueError):
        value_float = np.nan
    if not np.isfinite(value_float):
        return (
            f'background-color: {SIGNATURE_PALETTE_DICT["neutral"]}; '
            f'color: {SIGNATURE_PALETTE_DICT["ink"]};'
        )

    # *** CRITICAL*** Intensity is scaled by the data-driven scale_max so worse
    # values read as darker. Color *choice* still uses fixed risk thresholds
    # (top-1 > 25%, gross > 100%) so semantic limits keep their meaning.
    safe_scale_float = scale_max_float if (np.isfinite(scale_max_float) and scale_max_float > 0.0) else 1.0

    if metric_column_str in {
        "event_return_pct_float",
        "relative_event_return_pct_float",
    }:
        intensity_float = min(abs(value_float) / safe_scale_float, 1.0)
        fill_weight_float = 0.12 + 0.45 * intensity_float
        end_color_str = (
            SIGNATURE_PALETTE_DICT["profit"]
            if value_float >= 0.0
            else SIGNATURE_PALETTE_DICT["loss"]
        )
        text_color_str = (
            SIGNATURE_PALETTE_DICT["profit_dark"]
            if value_float >= 0.0
            else SIGNATURE_PALETTE_DICT["loss_dark"]
        )
    elif metric_column_str == "event_max_drawdown_pct_float":
        intensity_float = min(abs(value_float) / safe_scale_float, 1.0)
        fill_weight_float = 0.12 + 0.48 * intensity_float
        end_color_str = SIGNATURE_PALETTE_DICT["loss"]
        text_color_str = SIGNATURE_PALETTE_DICT["loss_dark"]
    elif metric_column_str == "entry_top1_weight_float":
        intensity_float = min(abs(value_float) / safe_scale_float, 1.0)
        fill_weight_float = 0.10 + 0.42 * intensity_float
        end_color_str = SIGNATURE_PALETTE_DICT["loss" if value_float > 0.25 else "benchmark"]
        text_color_str = SIGNATURE_PALETTE_DICT["loss_dark" if value_float > 0.25 else "benchmark_dark"]
    else:
        intensity_float = min(abs(value_float) / safe_scale_float, 1.0)
        fill_weight_float = 0.10 + 0.42 * intensity_float
        end_color_str = SIGNATURE_PALETTE_DICT["loss" if value_float > 1.0 else "benchmark"]
        text_color_str = SIGNATURE_PALETTE_DICT["loss_dark" if value_float > 1.0 else "benchmark_dark"]

    background_color_str = blend_hex_color_str(
        SIGNATURE_PALETTE_DICT["page"],
        end_color_str,
        fill_weight_float,
    )
    if intensity_float >= 0.72:
        text_color_str = SIGNATURE_PALETTE_DICT["page"]
    return f"background-color: {background_color_str}; color: {text_color_str};"


def _heatmap_legend_html(metric_column_str: str, scale_max_float: float) -> str:
    """Color-scale key so a reader can map a shade back to a number.

    Diverging metrics (event/relative return) show -max..0..+max; one-sided
    metrics show 0..max. Endpoints are the actual data-driven scale bounds.
    """
    if not np.isfinite(scale_max_float) or scale_max_float <= 0.0:
        return ""
    diverging_bool = metric_column_str in {
        "event_return_pct_float",
        "relative_event_return_pct_float",
    }
    if diverging_bool:
        sample_value_list = [
            -scale_max_float,
            -scale_max_float / 2.0,
            0.0,
            scale_max_float / 2.0,
            scale_max_float,
        ]
    else:
        sample_value_list = [
            0.0,
            scale_max_float / 4.0,
            scale_max_float / 2.0,
            scale_max_float * 3.0 / 4.0,
            scale_max_float,
        ]
    chip_html_list: list[str] = []
    for sample_value_float in sample_value_list:
        swatch_style_str = _heatmap_cell_style_str(
            metric_column_str,
            sample_value_float,
            scale_max_float,
        )
        label_str = _format_metric_value_str(metric_column_str, sample_value_float)
        chip_html_list.append(
            '<span class="legend-chip">'
            f'<span class="legend-swatch" style="{swatch_style_str}"></span>'
            f"<span>{html.escape(label_str)}</span>"
            "</span>"
        )
    return (
        '<div class="heatmap-legend">'
        '<span class="legend-label">Color scale</span>'
        + "".join(chip_html_list)
        + "</div>"
    )


def _heatmap_crisis_label_str(crisis_name_str: str) -> str:
    return str(crisis_name_str).replace("_", " ")


def _format_metric_heatmap_html(
    stress_metric_df: pd.DataFrame,
    metric_column_str: str,
    title_str: str,
    meaning_str: str,
) -> str:
    if stress_metric_df is None or len(stress_metric_df) == 0:
        return _wrap_card_html(
            f"<h3>{html.escape(title_str)}</h3>"
            f"<p class=\"heatmap-meaning\">{html.escape(meaning_str)}</p>"
            "<p>No scenarios.</p>"
        )
    ordered_metric_df = stress_metric_df.sort_values(
        by=["event_start_ts", "crisis_name_str", "launch_offset_int"],
        kind="mergesort",
    )
    crisis_name_list = list(dict.fromkeys(ordered_metric_df["crisis_name_str"].astype(str)))
    launch_offset_list = sorted(
        int(offset_obj) for offset_obj in ordered_metric_df["launch_offset_int"].dropna().unique()
    )
    scale_max_float = _heatmap_scale_max_float(ordered_metric_df, metric_column_str)
    colgroup_html_str = '<colgroup><col class="heatmap-label-col">' + "".join(
        '<col class="heatmap-value-col">' for _offset_int in launch_offset_list
    ) + "</colgroup>"
    header_html_str = "<th>Crisis</th>" + "".join(
        f"<th>{offset_int}d</th>" for offset_int in launch_offset_list
    )
    row_html_list: list[str] = []
    for crisis_name_str in crisis_name_list:
        crisis_metric_df = ordered_metric_df[
            ordered_metric_df["crisis_name_str"].astype(str) == crisis_name_str
        ]
        display_crisis_name_str = _heatmap_crisis_label_str(crisis_name_str)
        cell_html_list = [
            (
                f'<td class="metric heatmap-row-label" title="{html.escape(crisis_name_str)}">'
                f"{html.escape(display_crisis_name_str)}</td>"
            )
        ]
        for launch_offset_int in launch_offset_list:
            scenario_metric_df = crisis_metric_df[
                crisis_metric_df["launch_offset_int"].astype(int) == launch_offset_int
            ]
            if len(scenario_metric_df) == 0:
                cell_html_list.append('<td class="heatmap-value-cell"></td>')
                continue
            value_obj = scenario_metric_df.iloc[0][metric_column_str]
            cell_style_str = _heatmap_cell_style_str(metric_column_str, value_obj, scale_max_float)
            cell_text_str = _format_metric_value_str(metric_column_str, value_obj)
            cell_html_list.append(
                f'<td class="heatmap-value-cell" style="{cell_style_str}">{html.escape(cell_text_str)}</td>'
            )
        row_html_list.append("<tr>" + "".join(cell_html_list) + "</tr>")
    table_html_str = (
        f'<table class="heatmap stress-heatmap">{colgroup_html_str}<thead><tr>{header_html_str}</tr></thead>'
        f"<tbody>{''.join(row_html_list)}</tbody></table>"
    )
    legend_html_str = _heatmap_legend_html(metric_column_str, scale_max_float)
    return _wrap_card_html(
        f"<h3>{html.escape(title_str)}</h3>"
        f"<p class=\"heatmap-meaning\">{html.escape(meaning_str)}</p>"
        f'<div class="heatmap-matrix-wrap">{table_html_str}</div>'
        f"{legend_html_str}"
    )


def _build_heatmap_dashboard_html(stress_metric_df: pd.DataFrame) -> str:
    heatmap_card_html_list = [
        _format_metric_heatmap_html(
            stress_metric_df,
            "event_return_pct_float",
            "Event Return Heatmap",
            "Rows are crises, columns are launch offsets; green is better P-to-E return.",
        ),
        _format_metric_heatmap_html(
            stress_metric_df,
            "event_max_drawdown_pct_float",
            "Event Max Drawdown Heatmap",
            "More negative red means deeper peak-to-trough pain during the event.",
        ),
        _format_metric_heatmap_html(
            stress_metric_df,
            "relative_event_return_pct_float",
            "Relative Return Heatmap",
            "Green means the strategy beat the benchmark during the event.",
        ),
        _format_metric_heatmap_html(
            stress_metric_df,
            "gross_exposure_float",
            "Entry Gross Exposure Heatmap",
            "Shows how much capital was exposed at P close before the event.",
        ),
        _format_metric_heatmap_html(
            stress_metric_df,
            "entry_top1_weight_float",
            "Entry Top-1 Concentration Heatmap",
            "Higher values mean more single-position concentration risk at P close.",
        ),
    ]
    return (
        '<section class="stress-section">'
        "<h2>Heatmap Dashboard</h2>"
        f'<div class="card-grid">{"".join(heatmap_card_html_list)}</div>'
        "</section>"
    )


def _risk_flag_card_html(
    severity_str: str,
    title_str: str,
    value_str: str,
    note_str: str,
) -> str:
    return (
        f'<div class="risk-flag risk-{html.escape(severity_str)}">'
        f"<div class=\"risk-flag-title\">{html.escape(title_str)}</div>"
        f"<div class=\"risk-flag-value\">{html.escape(value_str)}</div>"
        f"<div class=\"risk-flag-note\">{html.escape(note_str)}</div>"
        "</div>"
    )


def _build_risk_flag_grid_html(stress_metric_df: pd.DataFrame) -> str:
    if stress_metric_df is None or len(stress_metric_df) == 0:
        return _wrap_card_html("<h2>Risk Flags</h2><p>No scenarios were evaluated.</p>")

    flag_card_html_list: list[str] = []
    drawdown_df = stress_metric_df[stress_metric_df["event_max_drawdown_pct_float"] <= -20.0]
    if len(drawdown_df) > 0:
        metric_row_ser = drawdown_df.sort_values("event_max_drawdown_pct_float", kind="mergesort").iloc[0]
        flag_card_html_list.append(
            _risk_flag_card_html(
                "red",
                "Event drawdown <= -20%",
                _fmt_pct(metric_row_ser["event_max_drawdown_pct_float"]),
                f"{metric_row_ser['crisis_name_str']}, offset {int(metric_row_ser['launch_offset_int'])}",
            )
        )

    gross_exposure_df = stress_metric_df[stress_metric_df["gross_exposure_float"] > 1.0]
    if len(gross_exposure_df) > 0:
        metric_row_ser = gross_exposure_df.sort_values(
            "gross_exposure_float",
            ascending=False,
            kind="mergesort",
        ).iloc[0]
        flag_card_html_list.append(
            _risk_flag_card_html(
                "red",
                "Entry gross exposure > 100%",
                _fmt_unsigned_ratio_pct(metric_row_ser["gross_exposure_float"]),
                f"{metric_row_ser['crisis_name_str']}, offset {int(metric_row_ser['launch_offset_int'])}",
            )
        )

    concentration_df = stress_metric_df[stress_metric_df["entry_top1_weight_float"] > 0.25]
    if len(concentration_df) > 0:
        metric_row_ser = concentration_df.sort_values(
            "entry_top1_weight_float",
            ascending=False,
            kind="mergesort",
        ).iloc[0]
        flag_card_html_list.append(
            _risk_flag_card_html(
                "amber",
                "Top-1 entry weight > 25%",
                _fmt_unsigned_ratio_pct(metric_row_ser["entry_top1_weight_float"]),
                f"{metric_row_ser['crisis_name_str']}, offset {int(metric_row_ser['launch_offset_int'])}",
            )
        )

    unrecovered_df = stress_metric_df[
        stress_metric_df["recovered_by_event_end_bool"] == False  # noqa: E712
    ]
    if len(unrecovered_df) > 0:
        metric_row_ser = unrecovered_df.sort_values(
            "event_return_pct_float",
            kind="mergesort",
        ).iloc[0]
        flag_card_html_list.append(
            _risk_flag_card_html(
                "amber",
                "Not recovered by event end",
                str(int(len(unrecovered_df))),
                f"Worst: {metric_row_ser['crisis_name_str']}, offset {int(metric_row_ser['launch_offset_int'])}",
            )
        )

    if len(flag_card_html_list) == 0:
        flag_card_html_list.append(
            _risk_flag_card_html(
                "clean",
                "No threshold flags",
                "Clean",
                "No V2 report flag threshold was breached.",
            )
        )
    return _wrap_card_html(
        "<h2>Risk Flags</h2>"
        f'<div class="risk-flag-grid">{"".join(flag_card_html_list)}</div>'
    )


def _event_chapter_chart_b64(
    path_df: pd.DataFrame,
    crisis_metric_df: pd.DataFrame,
    chart_kind_str: str,
) -> str | None:
    if path_df is None or len(path_df) == 0 or len(crisis_metric_df) == 0:
        return None
    sorted_metric_df = crisis_metric_df.sort_values("launch_offset_int", kind="mergesort")
    event_start_ts = pd.Timestamp(sorted_metric_df.iloc[0]["event_start_ts"])
    event_end_ts = pd.Timestamp(sorted_metric_df.iloc[0]["event_end_ts"])

    with plt.rc_context(build_signature_rcparams(to_web_bool=True)):
        figure_obj, axis_obj = plt.subplots(figsize=(13.5, 5.4))
        for color_idx_int, (_, metric_row_ser) in enumerate(sorted_metric_df.iterrows()):
            scenario_key_str = str(metric_row_ser["scenario_key_str"])
            scenario_path_df = path_df[path_df["scenario_key_str"] == scenario_key_str].copy()
            if len(scenario_path_df) == 0:
                continue
            scenario_path_df = scenario_path_df.sort_values("bar_offset_int", kind="mergesort")
            bar_ts_ser = pd.to_datetime(scenario_path_df["bar_ts"])
            color_str = SEABORN_DEEP_COLOR_LIST[color_idx_int % len(SEABORN_DEEP_COLOR_LIST)]
            launch_offset_int = int(metric_row_ser["launch_offset_int"])

            if chart_kind_str == "event_equity":
                event_entry_ts = pd.Timestamp(metric_row_ser["event_entry_ts"])
                chart_path_df = scenario_path_df[
                    (bar_ts_ser >= event_entry_ts) & (bar_ts_ser <= event_end_ts)
                ].copy()
                if len(chart_path_df) == 0:
                    continue
                chart_bar_ts_ser = pd.to_datetime(chart_path_df["bar_ts"])
                strategy_value_ser = chart_path_df[
                    "normalized_strategy_from_event_entry_float"
                ].astype(float)
                benchmark_value_ser = chart_path_df[
                    "normalized_benchmark_from_event_entry_float"
                ].astype(float)
                strategy_value_ser.loc[chart_bar_ts_ser == event_entry_ts] = 1.0
                if benchmark_value_ser.notna().any():
                    benchmark_value_ser.loc[chart_bar_ts_ser == event_entry_ts] = 1.0
            elif chart_kind_str == "drawdown":
                chart_bar_ts_ser = bar_ts_ser
                strategy_value_ser = scenario_path_df["drawdown_from_launch_pct_float"].astype(float)
                benchmark_value_ser = pd.Series(np.nan, index=scenario_path_df.index)
            else:
                chart_bar_ts_ser = bar_ts_ser
                strategy_value_ser = scenario_path_df[
                    "normalized_strategy_from_launch_float"
                ].astype(float)
                benchmark_value_ser = scenario_path_df[
                    "normalized_benchmark_from_launch_float"
                ].astype(float)

            axis_obj.plot(
                chart_bar_ts_ser,
                strategy_value_ser,
                label=f"{launch_offset_int}d strategy",
                color=color_str,
                linewidth=1.6,
            )
            if chart_kind_str != "drawdown" and benchmark_value_ser.notna().any():
                axis_obj.plot(
                    chart_bar_ts_ser,
                    benchmark_value_ser,
                    label=f"{launch_offset_int}d benchmark",
                    color=color_str,
                    linewidth=1.0,
                    linestyle="--",
                    alpha=0.50,
                )

        axis_obj.axvspan(
            event_start_ts,
            event_end_ts,
            color=SIGNATURE_PALETTE_DICT["loss"],
            alpha=0.08,
            label="Event window",
        )
        axis_obj.axvline(
            event_start_ts,
            color=SIGNATURE_PALETTE_DICT["loss_dark"],
            linestyle="--",
            linewidth=1.0,
        )
        axis_obj.grid(True)
        if chart_kind_str == "event_equity":
            axis_obj.set_title("Event-only equity, normalized to P close")
            axis_obj.set_ylabel("Equity / V_P")
        elif chart_kind_str == "drawdown":
            axis_obj.set_title("Drawdown by launch offset")
            axis_obj.set_ylabel("Drawdown [%]")
        else:
            axis_obj.set_title("Launch-to-end equity by launch offset")
            axis_obj.set_ylabel("Equity / V_L")
        axis_obj.set_xlabel("Date")
        axis_obj.legend(loc="best", fontsize=7, ncol=2)
        figure_obj.tight_layout()

        buffer_obj = io.BytesIO()
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="FigureCanvasAgg is non-interactive, and thus cannot be shown",
                category=UserWarning,
            )
            figure_obj.savefig(buffer_obj, format="png", dpi=160, bbox_inches="tight")
        plt.close(figure_obj)
        buffer_obj.seek(0)
        return base64.b64encode(buffer_obj.read()).decode("ascii")


def _format_event_chapter_summary_html(crisis_metric_df: pd.DataFrame) -> str:
    if len(crisis_metric_df) == 0:
        return ""
    worst_row_ser = crisis_metric_df.sort_values("event_return_pct_float", kind="mergesort").iloc[0]
    best_row_ser = crisis_metric_df.sort_values(
        "event_return_pct_float",
        ascending=False,
        kind="mergesort",
    ).iloc[0]
    max_exposure_row_ser = crisis_metric_df.sort_values(
        "gross_exposure_float",
        ascending=False,
        kind="mergesort",
    ).iloc[0]
    max_concentration_row_ser = crisis_metric_df.sort_values(
        "entry_top1_weight_float",
        ascending=False,
        kind="mergesort",
    ).iloc[0]
    unrecovered_count_int = int(
        (crisis_metric_df["recovered_by_event_end_bool"] == False).sum()  # noqa: E712
    )
    summary_row_list = [
        ("Worst offset", f"{int(worst_row_ser['launch_offset_int'])}d", _fmt_pct(worst_row_ser["event_return_pct_float"])),
        ("Best offset", f"{int(best_row_ser['launch_offset_int'])}d", _fmt_pct(best_row_ser["event_return_pct_float"])),
        ("Max entry gross", f"{int(max_exposure_row_ser['launch_offset_int'])}d", _fmt_unsigned_ratio_pct(max_exposure_row_ser["gross_exposure_float"])),
        ("Max top-1 weight", f"{int(max_concentration_row_ser['launch_offset_int'])}d", _fmt_unsigned_ratio_pct(max_concentration_row_ser["entry_top1_weight_float"])),
        ("Unrecovered offsets", str(unrecovered_count_int), "ending below P close"),
    ]
    row_html_list = [
        "<tr>"
        f"<td class=\"metric\">{html.escape(label_str)}</td>"
        f"<td>{html.escape(value_str)}</td>"
        f"<td>{html.escape(note_str)}</td>"
        "</tr>"
        for label_str, value_str, note_str in summary_row_list
    ]
    return (
        '<div class="scroll">'
        "<table><thead><tr><th>Metric</th><th>Value</th><th>Context</th></tr></thead>"
        f"<tbody>{''.join(row_html_list)}</tbody></table>"
        "</div>"
    )


def _format_event_exposure_table_html(crisis_metric_df: pd.DataFrame) -> str:
    column_spec_list = [
        ("launch_offset_int", "Offset"),
        ("cash_weight_float", "Cash"),
        ("long_gross_exposure_float", "Long Gross"),
        ("short_gross_exposure_float", "Short Gross"),
        ("net_exposure_float", "Net"),
        ("gross_exposure_float", "Gross"),
        ("entry_top1_weight_float", "Top-1"),
        ("entry_top5_weight_float", "Top-5"),
        ("recovered_by_event_end_bool", "Recovered"),
    ]
    header_html_str = "".join(f"<th>{html.escape(label_str)}</th>" for _, label_str in column_spec_list)
    row_html_list: list[str] = []
    sorted_metric_df = crisis_metric_df.sort_values("launch_offset_int", kind="mergesort")
    for _, metric_row_ser in sorted_metric_df.iterrows():
        cell_html_list = []
        for column_name_str, _label_str in column_spec_list:
            cell_text_str = _format_metric_value_str(column_name_str, metric_row_ser[column_name_str])
            cell_html_list.append(f"<td>{html.escape(cell_text_str)}</td>")
        row_html_list.append("<tr>" + "".join(cell_html_list) + "</tr>")
    return (
        f"<table><thead><tr>{header_html_str}</tr></thead>"
        f"<tbody>{''.join(row_html_list)}</tbody></table>"
    )


def _format_crisis_entry_positions_html(
    entry_position_df: pd.DataFrame,
    crisis_name_str: str,
    max_position_count_int: int = 10,
) -> str:
    crisis_position_df = entry_position_df[
        (entry_position_df["crisis_name_str"].astype(str) == crisis_name_str)
        & (entry_position_df["cash_position_bool"] == False)  # noqa: E712
    ].copy()
    if len(crisis_position_df) == 0:
        return "<p>No non-cash entering positions are available.</p>"
    sorted_position_df = crisis_position_df.sort_values(
        by=["launch_offset_int", "abs_position_weight_float"],
        ascending=[True, False],
        kind="mergesort",
    )
    top_position_df = (
        sorted_position_df.groupby("launch_offset_int", group_keys=False)
        .head(max_position_count_int)
        .reset_index(drop=True)
    )
    header_html_str = (
        "<th>Offset</th><th>Asset</th><th>Shares</th><th>Entry Close</th>"
        "<th>Value</th><th>Weight</th>"
    )
    row_html_list: list[str] = []
    for _, position_row_ser in top_position_df.iterrows():
        row_html_list.append(
            "<tr>"
            f"<td>{int(position_row_ser['launch_offset_int'])}d</td>"
            f"<td>{html.escape(str(position_row_ser['asset_str']))}</td>"
            f"<td>{html.escape(_fmt_float(position_row_ser['position_share_float'], 4))}</td>"
            f"<td>{html.escape(_fmt_float(position_row_ser['entry_close_price_float'], 2))}</td>"
            f"<td>{html.escape(_fmt_dollar(position_row_ser['position_value_float']))}</td>"
            f"<td>{html.escape(_fmt_ratio_pct(position_row_ser['position_weight_float']))}</td>"
            "</tr>"
        )
    return (
        f"<table><thead><tr>{header_html_str}</tr></thead>"
        f"<tbody>{''.join(row_html_list)}</tbody></table>"
    )


def _build_event_chapters_html(stress_result_obj: StressTestResult) -> str:
    metric_df = stress_result_obj.stress_metric_df
    if metric_df is None or len(metric_df) == 0:
        return ""
    ordered_metric_df = metric_df.sort_values(
        by=["event_start_ts", "crisis_name_str", "launch_offset_int"],
        kind="mergesort",
    )
    chapter_html_list: list[str] = []
    for crisis_name_str in dict.fromkeys(ordered_metric_df["crisis_name_str"].astype(str)):
        crisis_metric_df = ordered_metric_df[
            ordered_metric_df["crisis_name_str"].astype(str) == crisis_name_str
        ]
        event_start_ts = pd.Timestamp(crisis_metric_df.iloc[0]["event_start_ts"])
        event_end_ts = pd.Timestamp(crisis_metric_df.iloc[0]["event_end_ts"])
        launch_chart_b64_str = _event_chapter_chart_b64(
            stress_result_obj.stress_path_df,
            crisis_metric_df,
            "launch_equity",
        )
        event_chart_b64_str = _event_chapter_chart_b64(
            stress_result_obj.stress_path_df,
            crisis_metric_df,
            "event_equity",
        )
        drawdown_chart_b64_str = _event_chapter_chart_b64(
            stress_result_obj.stress_path_df,
            crisis_metric_df,
            "drawdown",
        )
        chart_panel_html_list: list[str] = []
        for title_str, chart_b64_str in [
            ("Combined Launch-to-End Chart", launch_chart_b64_str),
            ("Event-Only Chart", event_chart_b64_str),
            ("Drawdown Chart", drawdown_chart_b64_str),
        ]:
            if chart_b64_str is None:
                continue
            chart_panel_html_list.append(
                '<div class="chart-panel event-chart-panel">'
                f"<h3>{html.escape(title_str)}</h3>"
                f'<img src="data:image/png;base64,{chart_b64_str}" alt="{html.escape(title_str)}">'
                "</div>"
            )
        chapter_summary_card_html_str = _wrap_card_html(
            "<h3>Chapter Summary</h3>" + _format_event_chapter_summary_html(crisis_metric_df)
        )
        exposure_card_html_str = _wrap_card_html(
            '<h3>Exposure By Offset</h3><div class="scroll">'
            + _format_event_exposure_table_html(crisis_metric_df)
            + "</div>"
        )
        entering_position_card_html_str = _wrap_card_html(
            '<h3>Entering Positions By Offset</h3><div class="scroll">'
            + _format_crisis_entry_positions_html(
                stress_result_obj.stress_entry_position_df,
                crisis_name_str,
            )
            + "</div>"
        )
        chapter_html_list.append(
            f"""
<section class="event-chapter">
  <div class="chapter-divider">
    <div class="report-eyebrow">Event Chapter</div>
    <h2>{html.escape(crisis_name_str)}</h2>
    <div class="meta">
      Crisis start marker: {_fmt_date(event_start_ts)} &nbsp;|&nbsp;
      End: {_fmt_date(event_end_ts)}
    </div>
  </div>
  <div class="card-grid">
    {chapter_summary_card_html_str}
    {exposure_card_html_str}
  </div>
  <div class="event-chart-stack">{"".join(chart_panel_html_list)}</div>
  {entering_position_card_html_str}
</section>
"""
        )
    return '<section class="stress-section"><h2>Event Chapters</h2>' + "".join(chapter_html_list) + "</section>"


def _format_transaction_summary_table_html(stress_metric_df: pd.DataFrame) -> str:
    if stress_metric_df is None or len(stress_metric_df) == 0:
        return "<p>No transaction summary is available.</p>"
    column_spec_list = [
        ("crisis_name_str", "Crisis"),
        ("launch_offset_int", "Offset"),
        ("trade_count_int", "Trades"),
        ("event_trade_count_int", "Event Trades"),
        ("event_turnover_float", "Event Turnover"),
        ("event_commission_pct_float", "Event Commission"),
        ("open_trade_count_entering_int", "Open Trades Entering"),
    ]
    header_html_str = "".join(f"<th>{html.escape(label_str)}</th>" for _, label_str in column_spec_list)
    row_html_list: list[str] = []
    sorted_metric_df = stress_metric_df.sort_values(
        by=["event_start_ts", "crisis_name_str", "launch_offset_int"],
        kind="mergesort",
    )
    for _, metric_row_ser in sorted_metric_df.iterrows():
        cell_html_list = []
        for column_name_str, _label_str in column_spec_list:
            cell_text_str = _format_metric_value_str(column_name_str, metric_row_ser[column_name_str])
            cell_html_list.append(f"<td>{html.escape(cell_text_str)}</td>")
        row_html_list.append("<tr>" + "".join(cell_html_list) + "</tr>")
    return (
        f"<table><thead><tr>{header_html_str}</tr></thead>"
        f"<tbody>{''.join(row_html_list)}</tbody></table>"
    )


def _build_appendix_html(
    metric_matrix_html_str: str,
    transaction_summary_html_str: str,
) -> str:
    stress_matrix_card_html_str = _wrap_card_html(
        '<details class="appendix-details" open>'
        '<summary class="appendix-summary">Stress Matrix</summary>'
        '<p class="muted">Full per-scenario metrics, split into themed groups. '
        "Crisis and offset repeat in each group as the row key.</p>"
        + metric_matrix_html_str
        + "</details>"
    )
    transaction_summary_card_html_str = _wrap_card_html(
        '<details class="appendix-details" open>'
        '<summary class="appendix-summary">Transaction Summary</summary>'
        '<div class="scroll">'
        + transaction_summary_html_str
        + "</div></details>"
    )
    assumption_card_html_str = _wrap_card_html(
        "<h3>Assumption Boundary</h3>"
        "<p><strong>What this report is.</strong> Historical pre-crisis launch stress: "
        "the strategy starts flat before each known event, trades through the normal Vanilla engine, "
        "and enters the event with whatever positions that run produced.</p>"
        "<p><strong>What it is not.</strong> It is not synthetic shock generation, not Monte Carlo, "
        "and not the existing fresh-start Crisis Replay.</p>"
    )
    return f"""
<section class="stress-section appendix-section">
  <h2>Raw Detail Appendix</h2>
  {stress_matrix_card_html_str}
  {transaction_summary_card_html_str}
  {assumption_card_html_str}
</section>
"""


def _stress_report_css_str() -> str:
    # *** UI*** The stress CSS below references var(--color-*) tokens. The shared
    # base CSS interpolates raw hex and does not define a :root palette, so these
    # tokens must be declared here or every var() reference silently falls back to
    # an unstyled default (broken borders/colors across the whole report).
    root_var_css_str = (
        ":root {\n"
        f"    --color-page: {SIGNATURE_PALETTE_DICT['page']};\n"
        f"    --color-panel: {SIGNATURE_PALETTE_DICT['panel']};\n"
        f"    --color-ink: {SIGNATURE_PALETTE_DICT['ink']};\n"
        f"    --color-muted: {SIGNATURE_PALETTE_DICT['muted']};\n"
        f"    --color-grid: {SIGNATURE_PALETTE_DICT['grid']};\n"
        f"    --color-border: {SIGNATURE_PALETTE_DICT['grid']};\n"
        f"    --color-profit: {SIGNATURE_PALETTE_DICT['profit']};\n"
        f"    --color-profit-dark: {SIGNATURE_PALETTE_DICT['profit_dark']};\n"
        f"    --color-loss: {SIGNATURE_PALETTE_DICT['loss']};\n"
        f"    --color-loss-dark: {SIGNATURE_PALETTE_DICT['loss_dark']};\n"
        f"    --color-benchmark: {SIGNATURE_PALETTE_DICT['benchmark']};\n"
        f"    --color-benchmark-dark: {SIGNATURE_PALETTE_DICT['benchmark_dark']};\n"
        f"    --color-neutral: {SIGNATURE_PALETTE_DICT['neutral']};\n"
        "}\n"
    )
    return root_var_css_str + """
.stress-section {
    margin: 0 0 18px;
}
.risk-flag-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 10px;
}
.risk-flag {
    border: 1px solid var(--color-border);
    border-left: 4px solid var(--color-muted);
    border-radius: 4px;
    padding: 10px 12px;
    background: var(--color-panel);
}
.risk-red {
    border-left-color: var(--color-loss-dark);
}
.risk-amber {
    border-left-color: var(--color-benchmark-dark);
}
.risk-clean {
    border-left-color: var(--color-profit-dark);
}
.risk-flag-title {
    font-weight: 700;
    font-size: 0.88rem;
}
.risk-flag-value {
    margin-top: 5px;
    font-weight: 700;
    font-size: 1.18rem;
}
.risk-flag-note {
    margin-top: 3px;
    color: var(--color-muted);
    font-size: 0.82rem;
}
.kpi-meaning {
    margin-top: 6px;
    color: var(--color-ink);
    font-size: 0.84rem;
    line-height: 1.25;
}
.heatmap-meaning {
    margin: -4px 0 10px;
    color: var(--color-muted);
    font-size: 0.83rem;
    line-height: 1.25;
}
.heatmap-matrix-wrap {
    width: 100%;
    overflow: visible;
}
.stress-heatmap {
    table-layout: fixed;
    width: 100%;
    min-width: 0;
    border-collapse: collapse;
}
.stress-heatmap .heatmap-label-col {
    width: 36%;
}
.stress-heatmap .heatmap-value-col {
    width: auto;
}
.stress-heatmap th:first-child,
.stress-heatmap td.metric {
    text-align: left;
    min-width: 0;
    max-width: none;
    width: auto;
    padding-left: 8px;
    padding-right: 8px;
}
.stress-heatmap th:not(:first-child),
.stress-heatmap .heatmap-value-cell {
    min-width: 0;
    width: auto;
    max-width: none;
    text-align: center;
    white-space: nowrap;
    padding-left: 6px;
    padding-right: 6px;
    font-variant-numeric: tabular-nums;
}
.stress-heatmap .heatmap-row-label {
    white-space: normal;
    overflow-wrap: anywhere;
    line-height: 1.2;
}
.stress-heatmap td,
.stress-heatmap th {
    height: 34px;
    vertical-align: middle;
}
@media (max-width: 760px) {
    .stress-heatmap {
        font-size: 0.76em;
    }
    .stress-heatmap .heatmap-label-col {
        width: 42%;
    }
    .stress-heatmap th:first-child,
    .stress-heatmap td.metric {
        padding-left: 6px;
        padding-right: 6px;
    }
    .stress-heatmap th:not(:first-child),
    .stress-heatmap .heatmap-value-cell {
        padding-left: 4px;
        padding-right: 4px;
    }
}
.event-chapter {
    border-top: 2px solid var(--color-border);
    padding-top: 18px;
    margin-top: 22px;
}
.chapter-divider {
    margin-bottom: 12px;
}
.event-chart-stack {
    display: flex;
    flex-direction: column;
    gap: 12px;
    margin: 12px 0;
}
.event-chart-panel {
    width: 100%;
}
.event-chart-panel img {
    width: 100%;
}
.appendix-section {
    border-top: 2px solid var(--color-border);
    padding-top: 18px;
}
.stress-verdict {
    background: var(--color-panel);
    border: 1px solid var(--color-border);
    border-left: 4px solid var(--color-benchmark-dark);
    border-radius: 8px;
    padding: 14px 18px;
    margin: 0 0 22px;
}
.stress-verdict-headline {
    margin: 4px 0 6px;
    font-size: 1.05rem;
    font-weight: 700;
    line-height: 1.35;
}
.stress-verdict-detail {
    margin: 4px 0 0;
    color: var(--color-muted);
    font-size: 0.9rem;
    line-height: 1.4;
}
.heatmap-legend {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 8px;
    margin-top: 10px;
    font-size: 0.72rem;
    color: var(--color-muted);
}
.heatmap-legend .legend-label {
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
.heatmap-legend .legend-chip {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    font-variant-numeric: tabular-nums;
}
.heatmap-legend .legend-swatch {
    display: inline-block;
    width: 18px;
    height: 12px;
    border: 1px solid var(--color-border);
    border-radius: 2px;
}
.appendix-details > .appendix-summary {
    cursor: pointer;
    font-size: 1.05rem;
    font-weight: 700;
    margin-bottom: 8px;
    list-style: revert;
}
.stress-matrix-groups {
    display: flex;
    flex-direction: column;
    gap: 14px;
}
.stress-matrix-group h4 {
    margin: 6px 0 6px;
    font-size: 0.95rem;
    border-left: 3px solid var(--color-benchmark);
    padding-left: 8px;
}
"""


def _build_report_html_str(stress_result_obj: StressTestResult) -> str:
    run_date_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    verdict_html_str = _build_verdict_html(stress_result_obj)
    metric_matrix_html_str = _build_stress_matrix_grouped_html(stress_result_obj.stress_metric_df)
    worst_case_table_html_str = _format_worst_case_table_html(stress_result_obj.stress_metric_df)
    transaction_summary_html_str = _format_transaction_summary_table_html(
        stress_result_obj.stress_metric_df
    )
    heatmap_dashboard_html_str = _build_heatmap_dashboard_html(stress_result_obj.stress_metric_df)
    risk_flag_html_str = _build_risk_flag_grid_html(stress_result_obj.stress_metric_df)
    event_chapters_html_str = _build_event_chapters_html(stress_result_obj)
    appendix_html_str = _build_appendix_html(
        metric_matrix_html_str=metric_matrix_html_str,
        transaction_summary_html_str=transaction_summary_html_str,
    )
    launch_offset_str = ", ".join(str(offset_int) for offset_int in stress_result_obj.launch_offset_tuple)

    worst_case_card_html_str = _wrap_card_html(
        f"""
<h2>Worst Cases</h2>
<div class="scroll">{worst_case_table_html_str}</div>
"""
    )

    body_html_str = f"""<div class="report-shell">
<header class="report-header">
  <div class="report-eyebrow">StressTestAnalyzer Report</div>
  <h1>{html.escape(stress_result_obj.strategy_name_str)}</h1>
  <div class="meta">
    Run: {run_date_str} &nbsp;|&nbsp;
    Strategy Key: {html.escape(stress_result_obj.strategy_key_str)} &nbsp;|&nbsp;
    Capital: {_fmt_dollar(stress_result_obj.capital_base_float)} &nbsp;|&nbsp;
    Launch offsets: {html.escape(launch_offset_str)}
  </div>
</header>
{verdict_html_str}
{_build_stress_kpi_grid_html(stress_result_obj)}
{risk_flag_html_str}
{heatmap_dashboard_html_str}
{worst_case_card_html_str}
{event_chapters_html_str}
{appendix_html_str}
</div>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{html.escape(stress_result_obj.strategy_name_str)} - StressTestAnalyzer Report</title>
{build_report_font_head_html()}
<style>{build_report_css()}{_stress_report_css_str()}</style>
</head>
<body>
{body_html_str}
</body>
</html>"""


__all__ = [
    "DEFAULT_LAUNCH_OFFSET_TUPLE",
    "STRESS_TEST_ANALYSIS_TYPE_STR",
    "StressTestAnalyzer",
    "StressTestResult",
    "resolve_stress_launch_window",
    "run_stress_test_suite",
    "save_stress_test_results",
    "supported_stress_test_strategy_key_list",
]
