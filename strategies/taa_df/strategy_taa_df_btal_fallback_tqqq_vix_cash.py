"""
Defense First BTAL fallback-TQQQ variant with a month-end VRP cash gate.

Fallback overlay:
    if rv20_m < VIX_m:
        keep TQQQ fallback weight
    else:
        set TQQQ fallback weight to 0 and leave the residual as cash
"""

from __future__ import annotations

from dataclasses import replace

import pandas as pd

try:
    from strategies.taa_df.strategy_taa_df import get_defense_first_data
    from strategies.taa_df.strategy_taa_df_btal_fallback_tqqq import DEFAULT_CONFIG as BASE_CONFIG
    from strategies.taa_df.strategy_taa_df_fallback_variant_utils import _build_defense_first_strategy
    from strategies.taa_df.strategy_taa_df_fallback_vix_cash_variant_utils import (
        build_vix_cash_variant_config,
        get_standard_fallback_vix_cash_data,
        run_standard_fallback_vix_cash_variant,
    )
except ModuleNotFoundError:
    from strategy_taa_df import get_defense_first_data
    from strategy_taa_df_btal_fallback_tqqq import DEFAULT_CONFIG as BASE_CONFIG
    from strategy_taa_df_fallback_variant_utils import _build_defense_first_strategy
    from strategy_taa_df_fallback_vix_cash_variant_utils import (
        build_vix_cash_variant_config,
        get_standard_fallback_vix_cash_data,
        run_standard_fallback_vix_cash_variant,
    )


STRATEGY_NAME_STR = "strategy_taa_df_btal_fallback_tqqq_vix_cash"
DEFAULT_CONFIG = build_vix_cash_variant_config(BASE_CONFIG)


def run_variant(
    show_display_bool: bool = True,
    save_results_bool: bool = True,
    output_dir_str: str = "results",
    backtest_start_date_str: str | None = None,
    capital_base_float: float = 100_000.0,
    end_date_str: str | None = None,
):
    config = DEFAULT_CONFIG if end_date_str is None else replace(DEFAULT_CONFIG, end_date_str=end_date_str)
    return run_standard_fallback_vix_cash_variant(
        strategy_name_str=STRATEGY_NAME_STR,
        config=config,
        base_data_loader_fn=get_defense_first_data,
        show_display_bool=show_display_bool,
        save_results_bool=save_results_bool,
        output_dir_str=output_dir_str,
        backtest_start_date_str=backtest_start_date_str,
        capital_base_float=capital_base_float,
    )


def _map_month_end_weight_to_decision_close_df(
    month_end_weight_df: pd.DataFrame,
    execution_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    decision_weight_map: dict[pd.Timestamp, pd.Series] = {}

    for month_end_ts, target_weight_ser in month_end_weight_df.iterrows():
        month_period = pd.Timestamp(month_end_ts).to_period("M")
        trading_day_idx = execution_index[execution_index.to_period("M") == month_period]
        if len(trading_day_idx) == 0:
            continue

        # *** CRITICAL*** Calendar month-end can fall on a non-trading day.
        # T Close means the last tradable close in that signal month, not the
        # calendar timestamp stored by resample("ME").
        decision_close_ts = pd.Timestamp(trading_day_idx[-1])
        decision_weight_map[decision_close_ts] = target_weight_ser.copy()

    if len(decision_weight_map) == 0:
        raise RuntimeError("No tradable decision-close dates were generated for execution timing analysis.")

    decision_close_weight_df = pd.DataFrame.from_dict(decision_weight_map, orient="index").sort_index()
    decision_close_weight_df.index.name = "decision_close_date"
    return decision_close_weight_df


def build_execution_timing_analysis_inputs() -> dict[str, object]:
    """
    Build inputs for ExecutionTimingAnalysis.

    Formula:

        decision_t = month_end_close_t

        entry_fill = decision_t + entry_lag at entry_price_field
        exit_fill  = decision_t + exit_lag  at exit_price_field
    """
    (
        execution_price_df,
        _momentum_score_df,
        daily_vrp_signal_df,
        month_end_weight_df,
        rebalance_weight_df,
        month_end_vrp_diagnostic_df,
    ) = get_standard_fallback_vix_cash_data(
        config=DEFAULT_CONFIG,
        base_data_loader_fn=get_defense_first_data,
    )
    decision_close_weight_df = _map_month_end_weight_to_decision_close_df(
        month_end_weight_df=month_end_weight_df,
        execution_index=pd.DatetimeIndex(execution_price_df.index),
    )

    def strategy_factory_fn():
        strategy_obj = _build_defense_first_strategy(
            strategy_name_str=STRATEGY_NAME_STR,
            config=DEFAULT_CONFIG,
            rebalance_weight_df=decision_close_weight_df,
        )
        strategy_obj.daily_vrp_signal_df = daily_vrp_signal_df.copy()
        strategy_obj.month_end_vrp_diagnostic_df = month_end_vrp_diagnostic_df.copy()
        strategy_obj.show_taa_weights_report = True

        # *** CRITICAL*** This forward fill is for post-run weight diagnostics
        # only. ExecutionTimingAnalysis uses month-end decision dates for order
        # intent and then applies the tested fill timing to those orders.
        strategy_obj.daily_target_weights = (
            rebalance_weight_df.reindex(execution_price_df.index).ffill().dropna()
        )
        return strategy_obj

    calendar_idx = execution_price_df.index[
        execution_price_df.index >= rebalance_weight_df.index[0]
    ]

    return {
        "strategy_factory_fn": strategy_factory_fn,
        "pricing_data_df": execution_price_df,
        "calendar_idx": pd.DatetimeIndex(calendar_idx),
        "order_generation_mode_str": "signal_bar",
        "risk_model_str": "taa_rebalance",
        "entry_timing_str_tuple": ("same_close_moc", "next_open", "next_close"),
        "exit_timing_str_tuple": ("same_close_moc", "next_open", "next_close"),
        "default_entry_timing_str": "next_open",
        "default_exit_timing_str": "next_open",
    }


def run_execution_timing_analysis(
    show_display_bool: bool = True,
    save_results_bool: bool = True,
    output_dir_str: str = "results",
):
    from alpha.engine.execution_timing import ExecutionTimingAnalysis

    strategy_input_dict = build_execution_timing_analysis_inputs()
    timing_result_obj = ExecutionTimingAnalysis(
        strategy_factory_fn=strategy_input_dict["strategy_factory_fn"],
        pricing_data_df=strategy_input_dict["pricing_data_df"],
        calendar_idx=strategy_input_dict["calendar_idx"],
        output_dir_str=output_dir_str,
        save_output_bool=save_results_bool,
        entry_timing_str_tuple=strategy_input_dict["entry_timing_str_tuple"],
        exit_timing_str_tuple=strategy_input_dict["exit_timing_str_tuple"],
        order_generation_mode_str=strategy_input_dict["order_generation_mode_str"],
        risk_model_str=strategy_input_dict["risk_model_str"],
        default_entry_timing_str=strategy_input_dict["default_entry_timing_str"],
        default_exit_timing_str=strategy_input_dict["default_exit_timing_str"],
    ).run()

    if show_display_bool:
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 1000)
        print(timing_result_obj.metric_df.to_string(index=False))
        if timing_result_obj.output_dir_path is not None:
            print(f"\nSaved execution timing artifacts to: {timing_result_obj.output_dir_path.resolve()}")

    return timing_result_obj


if __name__ == "__main__":
    run_variant()
