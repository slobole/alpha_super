"""
Crisis-period replay suite for strategy stress evaluation.

For each configured crisis window c:

    C_c = { t : start_c <= t <= end_c }

The replay runs a fresh Vanilla backtest on the restricted crisis calendar:

    effective_start_c = first tradable bar on or after start_c
    effective_end_c = last tradable bar on or before end_c

Per-crisis return is measured from fresh crisis capital:

    R_c = V_end / V_0 - 1

with:

    V_0 = capital_base

Normalized crisis paths are:

    normalized_equity_t = V_t / V_0

This module keeps the engine contract unchanged. It only restricts the
execution calendar while preserving full pre-crisis history for causal signals.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from alpha.engine.backtest import run_daily
from alpha.engine.strategy import Strategy


@dataclass(frozen=True)
class CrisisPeriodConfig:
    crisis_name_str: str
    start_date_str: str
    end_date_str: str


@dataclass(frozen=True)
class CrisisStrategySpec:
    strategy_key_str: str
    load_context_fn: Callable[[], dict[str, object]]
    build_strategy_fn: Callable[[dict[str, object]], Strategy]


@dataclass
class CrisisReplayResult:
    strategy_key_str: str
    strategy_name_str: str
    capital_base_float: float
    crisis_period_config_list: list[CrisisPeriodConfig]
    crisis_metric_df: pd.DataFrame
    crisis_path_df: pd.DataFrame
    crisis_strategy_map: dict[str, Strategy] = field(default_factory=dict)
    output_dir_path: Path | None = None

    @property
    def supported_crisis_df(self) -> pd.DataFrame:
        return self.crisis_metric_df.copy()

    @property
    def unsupported_crisis_df(self) -> pd.DataFrame:
        return pd.DataFrame(columns=self.crisis_metric_df.columns)


class CrisisAnalyzer:
    """
    Class-based wrapper for crisis-period replay analysis.

    For each crisis c:

        R_c = V_end / V_0 - 1

    where each crisis run uses a fresh strategy instance:

        strategy_state^{(c)}_0 = fresh_instance()
    """

    def __init__(
        self,
        strategy_key_str: str | None = None,
        *,
        strategy_spec_obj: CrisisStrategySpec | None = None,
        crisis_period_list: Sequence[CrisisPeriodConfig | dict[str, str]] | None = None,
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
                    f"Unsupported crisis replay strategy '{strategy_key_str}'. "
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
        self.output_dir_str = str(output_dir_str)
        self.save_output_bool = bool(save_output_bool)
        self.show_progress_bool = bool(show_progress_bool)
        self.show_signal_progress_bool = bool(show_signal_progress_bool)
        self.latest_result_obj: CrisisReplayResult | None = None

    @classmethod
    def from_strategy_key(
        cls,
        strategy_key_str: str,
        **kwargs,
    ) -> "CrisisAnalyzer":
        return cls(strategy_key_str=strategy_key_str, **kwargs)

    @classmethod
    def supported_strategy_key_tuple(cls) -> tuple[str, ...]:
        return supported_crisis_strategy_key_list()

    def run(
        self,
        *,
        crisis_period_list: Sequence[CrisisPeriodConfig | dict[str, str]] | None = None,
        output_dir_str: str | None = None,
        save_output_bool: bool | None = None,
        show_progress_bool: bool | None = None,
        show_signal_progress_bool: bool | None = None,
    ) -> CrisisReplayResult:
        normalized_crisis_period_list = (
            self.crisis_period_config_list
            if crisis_period_list is None
            else _coerce_crisis_period_config_list(crisis_period_list)
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
        )
        capital_base_float = float(context_dict["capital_base_float"])
        strategy_name_str = str(context_dict["strategy_name_str"])

        metric_row_list: list[dict[str, object]] = []
        crisis_path_frame_list: list[pd.DataFrame] = []
        crisis_strategy_map: dict[str, Strategy] = {}

        for crisis_period_config in normalized_crisis_period_list:
            effective_start_ts, effective_end_ts, skip_reason_str = resolve_crisis_window(
                crisis_period_config=crisis_period_config,
                calendar_idx=supported_calendar_idx,
            )
            if skip_reason_str:
                continue

            crisis_calendar_idx = supported_calendar_idx[
                (supported_calendar_idx >= effective_start_ts)
                & (supported_calendar_idx <= effective_end_ts)
            ]
            if len(crisis_calendar_idx) == 0:
                continue

            # *** CRITICAL*** Restrict the execution calendar to crisis bars only
            # while leaving the full pre-crisis price history available for causal
            # signals through previous_bar.
            strategy_obj = self.strategy_spec_obj.build_strategy_fn(context_dict)
            run_daily(
                strategy=strategy_obj,
                pricing_data=pricing_data_df,
                calendar=crisis_calendar_idx,
                show_progress=resolved_show_progress_bool,
                show_signal_progress_bool=resolved_show_signal_progress_bool,
            )

            metric_row_list.append(
                _build_supported_metric_row_dict(
                    strategy_obj=strategy_obj,
                    crisis_period_config=crisis_period_config,
                    effective_start_ts=effective_start_ts,
                    effective_end_ts=effective_end_ts,
                )
            )
            crisis_path_frame_list.append(
                _build_crisis_path_df(
                    strategy_obj=strategy_obj,
                    crisis_name_str=crisis_period_config.crisis_name_str,
                    effective_start_ts=effective_start_ts,
                )
            )
            crisis_strategy_map[crisis_period_config.crisis_name_str] = strategy_obj

        crisis_metric_df = pd.DataFrame(metric_row_list)
        if len(crisis_metric_df) == 0:
            crisis_metric_df = pd.DataFrame(
                columns=[
                    "crisis_name_str",
                    "effective_start_ts",
                    "effective_end_ts",
                    "strategy_return_pct_float",
                    "benchmark_return_pct_float",
                    "relative_return_pct_float",
                    "max_drawdown_pct_float",
                    "volatility_ann_pct_float",
                    "sharpe_ratio_float",
                    "trade_count_int",
                ]
            )
        else:
            crisis_metric_df = crisis_metric_df.sort_values(
                by=["effective_start_ts", "crisis_name_str"],
                kind="mergesort",
            ).reset_index(drop=True)

        if len(crisis_path_frame_list) > 0:
            crisis_path_df = pd.concat(crisis_path_frame_list, ignore_index=True)
        else:
            crisis_path_df = pd.DataFrame(
                columns=[
                    "crisis_name_str",
                    "bar_offset_int",
                    "bar_ts",
                    "strategy_name_str",
                    "benchmark_name_str",
                    "normalized_strategy_equity_float",
                    "normalized_benchmark_equity_float",
                ]
            )

        crisis_replay_result = CrisisReplayResult(
            strategy_key_str=self.strategy_key_str,
            strategy_name_str=strategy_name_str,
            capital_base_float=capital_base_float,
            crisis_period_config_list=normalized_crisis_period_list,
            crisis_metric_df=crisis_metric_df,
            crisis_path_df=crisis_path_df,
            crisis_strategy_map=crisis_strategy_map,
        )

        if resolved_save_output_bool:
            from alpha.engine.report import save_crisis_replay_results

            crisis_replay_result.output_dir_path = save_crisis_replay_results(
                crisis_replay_result,
                output_dir=resolved_output_dir_str,
            )

        self.latest_result_obj = crisis_replay_result
        return crisis_replay_result


CRISIS_PERIODS_LIST: list[CrisisPeriodConfig] = [
    CrisisPeriodConfig(
        crisis_name_str="dot_com_bubble",
        start_date_str="2000-03-10",
        end_date_str="2002-10-09",
    ),
    CrisisPeriodConfig(
        crisis_name_str="911_aftermath",
        start_date_str="2001-09-11",
        end_date_str="2001-10-11",
    ),
    CrisisPeriodConfig(
        crisis_name_str="gfc_2008",
        start_date_str="2008-09-15",
        end_date_str="2009-03-09",
    ),
    CrisisPeriodConfig(
        crisis_name_str="european_debt_2011",
        start_date_str="2011-08-01",
        end_date_str="2011-10-04",
    ),
    CrisisPeriodConfig(
        crisis_name_str="china_oil_shock_2015",
        start_date_str="2015-08-18",
        end_date_str="2016-02-11",
    ),
    CrisisPeriodConfig(
        crisis_name_str="volmageddon",
        start_date_str="2018-02-02",
        end_date_str="2018-02-09",
    ),
    CrisisPeriodConfig(
        crisis_name_str="covid_crash",
        start_date_str="2020-02-20",
        end_date_str="2020-04-07",
    ),
    CrisisPeriodConfig(
        crisis_name_str="inflation_bear_2022",
        start_date_str="2022-01-03",
        end_date_str="2022-10-12",
    ),
    CrisisPeriodConfig(
        crisis_name_str="trump_tariffs_2025",
        start_date_str="2025-02-01",
        end_date_str="2025-04-30",
    ),
]


def _load_qpi_ibs_rsi_exit_context_dict() -> dict[str, object]:
    from data.norgate_loader import build_index_constituent_matrix
    from strategies.qpi.strategy_mr_qpi_ibs_rsi_exit import get_prices

    benchmark_list = ["$SPX"]
    symbol_list, universe_df = build_index_constituent_matrix(indexname="S&P 500")
    pricing_data_df = get_prices(
        symbol_list,
        benchmark_list,
        start_date_str="1998-01-01",
        end_date_str=None,
    )
    calendar_idx = pricing_data_df.index[pricing_data_df.index.year >= 2004]
    return {
        "strategy_name_str": "strategy_mr_qpi_ibs_rsi_exit",
        "capital_base_float": 100_000.0,
        "benchmark_list": benchmark_list,
        "pricing_data_df": pricing_data_df,
        "calendar_idx": pd.DatetimeIndex(calendar_idx),
        "universe_df": universe_df,
    }


def _build_qpi_ibs_rsi_exit_strategy_obj(context_dict: dict[str, object]) -> Strategy:
    from strategies.qpi.strategy_mr_qpi_ibs_rsi_exit import QPIIbsRsiExitStrategy

    strategy_obj = QPIIbsRsiExitStrategy(
        name=str(context_dict["strategy_name_str"]),
        benchmarks=list(context_dict["benchmark_list"]),
        capital_base=float(context_dict["capital_base_float"]),
        slippage=0.00025,
        commission_per_share=0.005,
        commission_minimum=1.0,
    )
    strategy_obj.universe_df = context_dict["universe_df"]
    return strategy_obj


def _load_dv2_context_dict() -> dict[str, object]:
    from data.norgate_loader import build_index_constituent_matrix
    from strategies.dv2.strategy_mr_dv2 import get_prices

    benchmark_list = ["$SPX"]
    symbol_list, universe_df = build_index_constituent_matrix(indexname="S&P 500")
    pricing_data_df = get_prices(
        symbol_list,
        benchmark_list,
        start_date="1998-01-01",
        end_date=None,
    )
    calendar_idx = pricing_data_df.index[pricing_data_df.index.year >= 2004]
    return {
        "strategy_name_str": "strategy_mr_dv2",
        "capital_base_float": 100_000.0,
        "benchmark_list": benchmark_list,
        "pricing_data_df": pricing_data_df,
        "calendar_idx": pd.DatetimeIndex(calendar_idx),
        "universe_df": universe_df,
    }


def _build_dv2_strategy_obj(context_dict: dict[str, object]) -> Strategy:
    from strategies.dv2.strategy_mr_dv2 import DVO2Strategy

    strategy_obj = DVO2Strategy(
        name=str(context_dict["strategy_name_str"]),
        benchmarks=list(context_dict["benchmark_list"]),
        capital_base=float(context_dict["capital_base_float"]),
        slippage=0.00025,
        commission_per_share=0.005,
        commission_minimum=1.0,
    )
    strategy_obj.universe_df = context_dict["universe_df"]
    return strategy_obj


def _load_dv2_price_adv_ibs_rsi_exit_context_dict() -> dict[str, object]:
    from data.norgate_loader import build_index_constituent_matrix
    from strategies.qpi.strategy_mr_dv2_price_adv_ibs_rsi_exit import get_prices

    benchmark_list = ["$SPX"]
    symbol_list, universe_df = build_index_constituent_matrix(indexname="S&P 500")
    pricing_data_df = get_prices(
        symbol_list,
        benchmark_list,
        start_date_str="1998-01-01",
        end_date_str=None,
    )
    calendar_idx = pricing_data_df.index[pricing_data_df.index.year >= 2004]
    return {
        "strategy_name_str": "strategy_mr_dv2_price_adv_ibs_rsi_exit",
        "capital_base_float": 100_000.0,
        "benchmark_list": benchmark_list,
        "pricing_data_df": pricing_data_df,
        "calendar_idx": pd.DatetimeIndex(calendar_idx),
        "universe_df": universe_df,
    }


def _build_dv2_price_adv_ibs_rsi_exit_strategy_obj(
    context_dict: dict[str, object],
) -> Strategy:
    from strategies.qpi.strategy_mr_dv2_price_adv_ibs_rsi_exit import (
        DV2PriceAdvIbsRsiExitStrategy,
    )

    strategy_obj = DV2PriceAdvIbsRsiExitStrategy(
        name=str(context_dict["strategy_name_str"]),
        benchmarks=list(context_dict["benchmark_list"]),
        capital_base=float(context_dict["capital_base_float"]),
        slippage=0.00025,
        commission_per_share=0.005,
        commission_minimum=1.0,
    )
    strategy_obj.universe_df = context_dict["universe_df"]
    return strategy_obj


def _load_taa_df_context_dict() -> dict[str, object]:
    from strategies.taa_df.strategy_taa_df import (
        DEFAULT_CONFIG,
        get_defense_first_data,
    )

    config = DEFAULT_CONFIG
    execution_price_df, _momentum_score_df, _month_end_weight_df, rebalance_weight_df = (
        get_defense_first_data(config)
    )
    calendar_idx = execution_price_df.index[
        execution_price_df.index >= rebalance_weight_df.index[0]
    ]
    return {
        "strategy_name_str": "strategy_taa_df",
        "capital_base_float": 100_000.0,
        "config_obj": config,
        "pricing_data_df": execution_price_df,
        "calendar_idx": pd.DatetimeIndex(calendar_idx),
        "rebalance_weight_df": rebalance_weight_df,
    }


def _build_taa_df_strategy_obj(context_dict: dict[str, object]) -> Strategy:
    from strategies.taa_df.strategy_taa_df import DefenseFirstStrategy

    config_obj = context_dict["config_obj"]
    strategy_obj = DefenseFirstStrategy(
        name=str(context_dict["strategy_name_str"]),
        benchmarks=config_obj.benchmark_list,
        rebalance_weight_df=context_dict["rebalance_weight_df"],
        tradeable_asset_list=config_obj.tradeable_asset_list,
        capital_base=float(context_dict["capital_base_float"]),
        slippage=0.0001,
        commission_per_share=0.005,
        commission_minimum=1.0,
    )
    pricing_data_df = context_dict["pricing_data_df"]
    rebalance_weight_df = context_dict["rebalance_weight_df"]
    # *** CRITICAL*** This forward fill is for daily weight diagnostics only.
    # Execution still uses the discrete rebalance dates stored in
    # `rebalance_weight_df` inside the strategy iterate path.
    strategy_obj.daily_target_weights = (
        rebalance_weight_df.reindex(pricing_data_df.index).ffill().dropna()
    )
    return strategy_obj


def _load_taa_df_btal_fallback_tqqq_vix_cash_context_dict() -> dict[str, object]:
    from strategies.taa_df.strategy_taa_df import get_defense_first_data
    from strategies.taa_df.strategy_taa_df_btal_fallback_tqqq_vix_cash import (
        DEFAULT_CONFIG,
    )
    from strategies.taa_df.strategy_taa_df_fallback_vix_cash_variant_utils import (
        get_standard_fallback_vix_cash_data,
    )

    config_obj = DEFAULT_CONFIG
    (
        execution_price_df,
        _momentum_score_df,
        daily_vrp_signal_df,
        _month_end_weight_df,
        rebalance_weight_df,
        month_end_vrp_diagnostic_df,
    ) = get_standard_fallback_vix_cash_data(
        config=config_obj,
        base_data_loader_fn=get_defense_first_data,
    )
    calendar_idx = execution_price_df.index[
        execution_price_df.index >= rebalance_weight_df.index[0]
    ]
    return {
        "strategy_name_str": "strategy_taa_df_btal_fallback_tqqq_vix_cash",
        "capital_base_float": 100_000.0,
        "config_obj": config_obj,
        "pricing_data_df": execution_price_df,
        "calendar_idx": pd.DatetimeIndex(calendar_idx),
        "rebalance_weight_df": rebalance_weight_df,
        "daily_vrp_signal_df": daily_vrp_signal_df,
        "month_end_vrp_diagnostic_df": month_end_vrp_diagnostic_df,
    }


def _build_taa_df_btal_fallback_tqqq_vix_cash_strategy_obj(
    context_dict: dict[str, object],
) -> Strategy:
    strategy_obj = _build_taa_df_strategy_obj(context_dict)
    strategy_obj.daily_vrp_signal_df = context_dict["daily_vrp_signal_df"].copy()
    strategy_obj.month_end_vrp_diagnostic_df = context_dict[
        "month_end_vrp_diagnostic_df"
    ].copy()
    return strategy_obj


def _load_mo_atr_normalized_ndx_context_dict() -> dict[str, object]:
    from strategies.momentum.strategy_mo_atr_normalized_ndx import (
        DEFAULT_CONFIG,
        get_atr_normalized_ndx_data,
    )

    config_obj = DEFAULT_CONFIG
    pricing_data_df, universe_df, rebalance_schedule_df = get_atr_normalized_ndx_data(
        config=config_obj,
    )
    calendar_idx = pricing_data_df.index[
        pricing_data_df.index >= pd.Timestamp(config_obj.backtest_start_date_str)
    ]
    return {
        "strategy_name_str": "strategy_mo_atr_normalized_ndx",
        "capital_base_float": float(config_obj.capital_base_float),
        "config_obj": config_obj,
        "pricing_data_df": pricing_data_df,
        "calendar_idx": pd.DatetimeIndex(calendar_idx),
        "rebalance_schedule_df": rebalance_schedule_df,
        "universe_df": universe_df,
    }


def _build_mo_atr_normalized_ndx_strategy_obj(
    context_dict: dict[str, object],
) -> Strategy:
    from strategies.momentum.strategy_mo_atr_normalized_ndx import AtrNormalizedNdxStrategy

    config_obj = context_dict["config_obj"]
    strategy_obj = AtrNormalizedNdxStrategy(
        name=str(context_dict["strategy_name_str"]),
        benchmarks=[str(config_obj.regime_symbol_str)],
        rebalance_schedule_df=context_dict["rebalance_schedule_df"],
        regime_symbol_str=str(config_obj.regime_symbol_str),
        capital_base=float(context_dict["capital_base_float"]),
        slippage=float(config_obj.slippage_float),
        commission_per_share=float(config_obj.commission_per_share_float),
        commission_minimum=float(config_obj.commission_minimum_float),
        lookback_month_int=int(config_obj.lookback_month_int),
        index_trend_window_int=int(config_obj.index_trend_window_int),
        stock_trend_window_int=int(config_obj.stock_trend_window_int),
        max_positions_int=int(config_obj.max_positions_int),
    )
    strategy_obj.universe_df = context_dict["universe_df"]
    return strategy_obj


SUPPORTED_CRISIS_STRATEGY_SPEC_MAP: dict[str, CrisisStrategySpec] = {
    "strategy_mr_qpi_ibs_rsi_exit": CrisisStrategySpec(
        strategy_key_str="strategy_mr_qpi_ibs_rsi_exit",
        load_context_fn=_load_qpi_ibs_rsi_exit_context_dict,
        build_strategy_fn=_build_qpi_ibs_rsi_exit_strategy_obj,
    ),
    "strategy_mr_dv2": CrisisStrategySpec(
        strategy_key_str="strategy_mr_dv2",
        load_context_fn=_load_dv2_context_dict,
        build_strategy_fn=_build_dv2_strategy_obj,
    ),
    "strategy_mr_dv2_price_adv_ibs_rsi_exit": CrisisStrategySpec(
        strategy_key_str="strategy_mr_dv2_price_adv_ibs_rsi_exit",
        load_context_fn=_load_dv2_price_adv_ibs_rsi_exit_context_dict,
        build_strategy_fn=_build_dv2_price_adv_ibs_rsi_exit_strategy_obj,
    ),
    "strategy_taa_df": CrisisStrategySpec(
        strategy_key_str="strategy_taa_df",
        load_context_fn=_load_taa_df_context_dict,
        build_strategy_fn=_build_taa_df_strategy_obj,
    ),
    "strategy_taa_df_btal_fallback_tqqq_vix_cash": CrisisStrategySpec(
        strategy_key_str="strategy_taa_df_btal_fallback_tqqq_vix_cash",
        load_context_fn=_load_taa_df_btal_fallback_tqqq_vix_cash_context_dict,
        build_strategy_fn=_build_taa_df_btal_fallback_tqqq_vix_cash_strategy_obj,
    ),
    "strategy_mo_atr_normalized_ndx": CrisisStrategySpec(
        strategy_key_str="strategy_mo_atr_normalized_ndx",
        load_context_fn=_load_mo_atr_normalized_ndx_context_dict,
        build_strategy_fn=_build_mo_atr_normalized_ndx_strategy_obj,
    ),
}

SUPPORTED_CRISIS_STRATEGY_KEY_TUPLE: tuple[str, ...] = tuple(
    SUPPORTED_CRISIS_STRATEGY_SPEC_MAP.keys()
)


def supported_crisis_strategy_key_list() -> tuple[str, ...]:
    return SUPPORTED_CRISIS_STRATEGY_KEY_TUPLE


def _coerce_crisis_period_config_list(
    crisis_period_list: Sequence[CrisisPeriodConfig | dict[str, str]],
) -> list[CrisisPeriodConfig]:
    normalized_crisis_period_list: list[CrisisPeriodConfig] = []
    for crisis_period_obj in crisis_period_list:
        if isinstance(crisis_period_obj, CrisisPeriodConfig):
            normalized_crisis_period_list.append(crisis_period_obj)
            continue

        normalized_crisis_period_list.append(
            CrisisPeriodConfig(
                crisis_name_str=str(crisis_period_obj["name"]),
                start_date_str=str(crisis_period_obj["start"]),
                end_date_str=str(crisis_period_obj["end"]),
            )
        )
    return normalized_crisis_period_list


def resolve_crisis_window(
    crisis_period_config: CrisisPeriodConfig,
    calendar_idx: pd.DatetimeIndex,
) -> tuple[pd.Timestamp | None, pd.Timestamp | None, str]:
    requested_start_ts = pd.Timestamp(crisis_period_config.start_date_str)
    requested_end_ts = pd.Timestamp(crisis_period_config.end_date_str)
    if requested_end_ts < requested_start_ts:
        return None, None, "requested_end_ts is earlier than requested_start_ts."

    supported_calendar_idx = pd.DatetimeIndex(calendar_idx).sort_values().unique()
    if len(supported_calendar_idx) == 0:
        return None, None, "strategy calendar is empty."

    supported_start_ts = pd.Timestamp(supported_calendar_idx[0])
    supported_end_ts = pd.Timestamp(supported_calendar_idx[-1])
    if requested_end_ts < supported_start_ts:
        return (
            None,
            None,
            f"requested window ends before supported history begins on {supported_start_ts.date()}.",
        )
    if requested_start_ts > supported_end_ts:
        return (
            None,
            None,
            f"requested window starts after supported history ends on {supported_end_ts.date()}.",
        )

    # *** CRITICAL*** Snap requested crisis anchors to the actual tradable bars
    # in the supported backtest calendar so entries never use non-trading dates.
    start_pos_int = int(supported_calendar_idx.searchsorted(requested_start_ts, side="left"))
    end_pos_int = int(supported_calendar_idx.searchsorted(requested_end_ts, side="right") - 1)

    if start_pos_int >= len(supported_calendar_idx):
        return None, None, "no tradable bars exist on or after requested_start_ts."
    if end_pos_int < 0:
        return None, None, "no tradable bars exist on or before requested_end_ts."

    effective_start_ts = pd.Timestamp(supported_calendar_idx[start_pos_int])
    effective_end_ts = pd.Timestamp(supported_calendar_idx[end_pos_int])
    if effective_start_ts > requested_end_ts:
        return None, None, "no tradable bars fall inside the requested crisis window."
    if effective_end_ts < requested_start_ts:
        return None, None, "no tradable bars fall inside the requested crisis window."
    if effective_end_ts < effective_start_ts:
        return None, None, "effective_end_ts falls before effective_start_ts."

    return effective_start_ts, effective_end_ts, ""


def _primary_benchmark_name_str(strategy_obj: Strategy) -> str | None:
    if not hasattr(strategy_obj, "_benchmarks") or len(strategy_obj._benchmarks) == 0:
        return None
    benchmark_name_str = str(strategy_obj._benchmarks[0])
    if strategy_obj.summary is None or benchmark_name_str not in strategy_obj.summary.columns:
        return None
    if benchmark_name_str not in strategy_obj.results.columns:
        return None
    return benchmark_name_str


def _build_supported_metric_row_dict(
    strategy_obj: Strategy,
    crisis_period_config: CrisisPeriodConfig,
    effective_start_ts: pd.Timestamp,
    effective_end_ts: pd.Timestamp,
) -> dict[str, object]:
    summary_ser = strategy_obj.summary["Strategy"]
    benchmark_name_str = _primary_benchmark_name_str(strategy_obj)

    benchmark_return_pct_float = np.nan
    if benchmark_name_str is not None:
        benchmark_return_pct_float = float(
            strategy_obj.summary.loc["Return [%]", benchmark_name_str]
        )

    open_trade_count_int = (
        0 if strategy_obj._open_trades is None else int(len(strategy_obj._open_trades))
    )
    closed_trade_count_int = 0 if strategy_obj._trades is None else int(len(strategy_obj._trades))

    return {
        "crisis_name_str": crisis_period_config.crisis_name_str,
        "effective_start_ts": pd.Timestamp(effective_start_ts),
        "effective_end_ts": pd.Timestamp(effective_end_ts),
        "strategy_return_pct_float": float(summary_ser.loc["Return [%]"]),
        "benchmark_return_pct_float": benchmark_return_pct_float,
        "relative_return_pct_float": (
            float(summary_ser.loc["Return [%]"]) - benchmark_return_pct_float
            if np.isfinite(benchmark_return_pct_float)
            else np.nan
        ),
        "max_drawdown_pct_float": float(summary_ser.loc["Max. Drawdown [%]"]),
        "volatility_ann_pct_float": float(summary_ser.loc["Volatility (Ann.) [%]"]),
        "sharpe_ratio_float": float(summary_ser.loc["Sharpe Ratio"]),
        "trade_count_int": closed_trade_count_int + open_trade_count_int,
    }

def _build_crisis_path_df(
    strategy_obj: Strategy,
    crisis_name_str: str,
    effective_start_ts: pd.Timestamp,
) -> pd.DataFrame:
    strategy_total_value_ser = strategy_obj.results["total_value"].astype(float)
    capital_base_float = float(strategy_obj._capital_base)
    benchmark_name_str = _primary_benchmark_name_str(strategy_obj)
    if benchmark_name_str is None:
        benchmark_total_value_ser = pd.Series(np.nan, index=strategy_total_value_ser.index)
    else:
        benchmark_total_value_ser = strategy_obj.results[benchmark_name_str].astype(float)

    # *** CRITICAL*** Normalize each crisis path to fresh crisis capital so
    # windows with different lengths remain directly comparable.
    path_row_list: list[dict[str, object]] = [
        {
            "crisis_name_str": crisis_name_str,
            "bar_offset_int": 0,
            "bar_ts": pd.Timestamp(effective_start_ts),
            "strategy_name_str": strategy_obj.name,
            "benchmark_name_str": benchmark_name_str or "",
            "normalized_strategy_equity_float": 1.0,
            "normalized_benchmark_equity_float": 1.0 if benchmark_name_str else np.nan,
        }
    ]
    for bar_offset_int, bar_ts in enumerate(strategy_total_value_ser.index, start=1):
        path_row_list.append(
            {
                "crisis_name_str": crisis_name_str,
                "bar_offset_int": int(bar_offset_int),
                "bar_ts": pd.Timestamp(bar_ts),
                "strategy_name_str": strategy_obj.name,
                "benchmark_name_str": benchmark_name_str or "",
                "normalized_strategy_equity_float": float(
                    strategy_total_value_ser.loc[bar_ts] / capital_base_float
                ),
                "normalized_benchmark_equity_float": (
                    float(benchmark_total_value_ser.loc[bar_ts] / capital_base_float)
                    if benchmark_name_str
                    else np.nan
                ),
            }
        )

    return pd.DataFrame(path_row_list)


def run_crisis_replay_suite(
    strategy_key_str: str | None = None,
    crisis_period_list: Sequence[CrisisPeriodConfig | dict[str, str]] = CRISIS_PERIODS_LIST,
    strategy_spec_obj: CrisisStrategySpec | None = None,
    output_dir: str = "results",
    save_output_bool: bool = True,
    show_progress_bool: bool = False,
    show_signal_progress_bool: bool = False,
) -> CrisisReplayResult:
    crisis_analyzer = CrisisAnalyzer(
        strategy_key_str=strategy_key_str,
        strategy_spec_obj=strategy_spec_obj,
        crisis_period_list=crisis_period_list,
        output_dir_str=output_dir,
        save_output_bool=save_output_bool,
        show_progress_bool=show_progress_bool,
        show_signal_progress_bool=show_signal_progress_bool,
    )
    return crisis_analyzer.run()


__all__ = [
    "CRISIS_PERIODS_LIST",
    "CrisisAnalyzer",
    "CrisisPeriodConfig",
    "CrisisReplayResult",
    "CrisisStrategySpec",
    "SUPPORTED_CRISIS_STRATEGY_KEY_TUPLE",
    "SUPPORTED_CRISIS_STRATEGY_SPEC_MAP",
    "resolve_crisis_window",
    "run_crisis_replay_suite",
    "supported_crisis_strategy_key_list",
]
