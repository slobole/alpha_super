"""Research-only QQQ Adaptive Momentum market-regime strategy.

The frozen Adaptive Momentum signal is computed from QQQ total-return closes.
A risk-on state buys QQQ at Open_(T+1); a risk-off state exits QQQ at
Open_(T+1).

This module contains no LIVE, broker, scheduler, release, allocation, or
strategy-registry wiring.
"""

from __future__ import annotations

from dataclasses import replace

import pandas as pd
from IPython.display import display

from alpha.engine.backtest import run_daily
from alpha.engine.report import save_results
from strategies.momentum.strategy_mo_spy_adaptive_momentum_regime import (
    SpyAdaptiveMomentumRegimeConfig,
    SpyAdaptiveMomentumRegimeStrategy,
    build_adaptive_momentum_calendar_idx,
    get_spy_adaptive_momentum_regime_data,
)


QQQ_CONFIG = SpyAdaptiveMomentumRegimeConfig(
    trade_symbol_str="QQQ",
    signal_symbol_str="QQQ_TR_SIGNAL",
    benchmark_symbol_str="$SPX",
)


def _validate_qqq_config(config_obj: SpyAdaptiveMomentumRegimeConfig) -> None:
    if config_obj.trade_symbol_str != "QQQ":
        raise ValueError("QQQ strategy requires trade_symbol_str='QQQ'.")
    if config_obj.signal_symbol_str != "QQQ_TR_SIGNAL":
        raise ValueError("QQQ strategy requires signal_symbol_str='QQQ_TR_SIGNAL'.")


def get_qqq_adaptive_momentum_regime_data(
    config_obj: SpyAdaptiveMomentumRegimeConfig = QQQ_CONFIG,
) -> pd.DataFrame:
    """Load QQQ execution prices and QQQ total-return signal prices."""

    _validate_qqq_config(config_obj)
    pricing_data_df = get_spy_adaptive_momentum_regime_data(config=config_obj)
    pricing_data_df.attrs["signal_data_symbol_by_alias_dict"] = {
        config_obj.signal_symbol_str: config_obj.trade_symbol_str
    }
    return pricing_data_df


class QqqAdaptiveMomentumRegimeStrategy(SpyAdaptiveMomentumRegimeStrategy):
    """Trade QQQ using the frozen Adaptive Momentum state computed on QQQ."""

    def __init__(
        self,
        name: str,
        benchmarks: list[str] | tuple[str, ...],
        config: SpyAdaptiveMomentumRegimeConfig = QQQ_CONFIG,
    ) -> None:
        _validate_qqq_config(config)
        super().__init__(name=name, benchmarks=benchmarks, config=config)


def run_variant(
    show_display_bool: bool = True,
    save_results_bool: bool = True,
    output_dir_str: str = "results",
    pricing_data_df: pd.DataFrame | None = None,
    config_obj: SpyAdaptiveMomentumRegimeConfig = QQQ_CONFIG,
) -> QqqAdaptiveMomentumRegimeStrategy:
    """Run the research-only QQQ self-signal variant."""

    _validate_qqq_config(config_obj)
    if pricing_data_df is None:
        pricing_data_df = get_qqq_adaptive_momentum_regime_data(config_obj)
    strategy_obj = QqqAdaptiveMomentumRegimeStrategy(
        name="strategy_mo_qqq_adaptive_momentum_regime_research",
        benchmarks=[config_obj.benchmark_symbol_str],
        config=config_obj,
    )
    calendar_idx = build_adaptive_momentum_calendar_idx(
        pricing_data_df=pricing_data_df,
        config_obj=config_obj,
    )
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


def _build_analysis_context_dict() -> dict[str, object]:
    config_obj = QQQ_CONFIG
    pricing_data_df = get_qqq_adaptive_momentum_regime_data(config_obj)
    calendar_idx = build_adaptive_momentum_calendar_idx(
        pricing_data_df=pricing_data_df,
        config_obj=config_obj,
    )
    return {
        "strategy_name_str": "strategy_mo_qqq_adaptive_momentum_regime_research",
        "capital_base_float": float(config_obj.capital_base_float),
        "config_obj": config_obj,
        "pricing_data_df": pricing_data_df,
        "calendar_idx": calendar_idx,
    }


def _build_strategy_obj(context_dict: dict[str, object]) -> QqqAdaptiveMomentumRegimeStrategy:
    config_obj = context_dict["config_obj"]
    if not isinstance(config_obj, SpyAdaptiveMomentumRegimeConfig):
        raise TypeError("context config_obj must be SpyAdaptiveMomentumRegimeConfig.")
    _validate_qqq_config(config_obj)
    return QqqAdaptiveMomentumRegimeStrategy(
        name=str(context_dict["strategy_name_str"]),
        benchmarks=[config_obj.benchmark_symbol_str],
        config=config_obj,
    )


def build_execution_timing_analysis_inputs() -> dict[str, object]:
    context_dict = _build_analysis_context_dict()

    def strategy_factory_fn() -> QqqAdaptiveMomentumRegimeStrategy:
        return _build_strategy_obj(context_dict)

    return {
        "strategy_factory_fn": strategy_factory_fn,
        "pricing_data_df": context_dict["pricing_data_df"],
        "calendar_idx": context_dict["calendar_idx"],
        "order_generation_mode_str": "vanilla_current_bar",
        "risk_model_str": "daily_ohlc_signal",
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
        "default_entry_timing_str": "same_open",
        "default_exit_timing_str": "same_open",
    }


def build_stress_test_context_dict() -> dict[str, object]:
    return _build_analysis_context_dict()


def build_stress_test_strategy_obj(
    context_dict: dict[str, object],
) -> QqqAdaptiveMomentumRegimeStrategy:
    return _build_strategy_obj(context_dict)


def build_capacity_analysis_inputs(
    capital_base_float: float,
    show_display_bool: bool = False,
    backtest_start_date_str: str | None = None,
    end_date_str: str | None = None,
) -> dict[str, object]:
    config_obj = replace(
        QQQ_CONFIG,
        capital_base_float=float(capital_base_float),
        backtest_start_date_str=(
            QQQ_CONFIG.backtest_start_date_str
            if backtest_start_date_str is None
            else str(backtest_start_date_str)
        ),
        end_date_str=end_date_str,
    )
    pricing_data_df = get_qqq_adaptive_momentum_regime_data(config_obj)
    strategy_obj = run_variant(
        show_display_bool=show_display_bool,
        save_results_bool=False,
        pricing_data_df=pricing_data_df,
        config_obj=config_obj,
    )
    return {
        "strategy_obj": strategy_obj,
        "pricing_data_df": pricing_data_df,
        "execution_policy_str": "MOO",
        "impact_profile_str": "MOO_ETF_PROXY",
    }


if __name__ == "__main__":
    run_variant()
