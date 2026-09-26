"""Monthly Nasdaq-100 momentum ranked by fractional ATR20.

Research-only NATR20 variant of the corrected nominal-ATR strategy.
Score_T = ROC12_T / (ATR20_adjusted_T / Close_adjusted_T).
ATR20 is the simple mean of 20 true ranges, not Wilder's ATR14.

Preserved rules: PIT Nasdaq 100; stock Close > SMA100; SPY > SMA200;
top ten names, 10% each, unused slots in cash; no positive-ROC requirement.
Decision uses the last exchange close of the month; fills at next open.
CAPITALSPECIAL signals/fills, SPXTR benchmark, $100,000 initial capital,
2.5 bps slippage/side, $0.005 per historical-equivalent share, $1 minimum.
The shared ordinary-dividend cash contract and its limitations are inherited.
No VXN overlay, parameter search, live release or allocation is introduced.
The 2000-2026 history has already been examined; it is not a fresh holdout.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import replace

import pandas as pd

from alpha.engine.backtest import run_daily
from alpha.engine.report import save_results
from strategies.momentum.strategy_mo_atr_normalized_ndx import (
    ATR_WINDOW_INT,
    DEFAULT_CONFIG,
    AtrNormalizedNdxConfig,
    AtrNormalizedNdxStrategy,
    _map_rebalance_schedule_to_decision_close_schedule_df,
    configure_total_return_benchmark_provenance,
    default_trade_id_int,
    get_atr_normalized_ndx_data,
)

STRATEGY_NAME_STR = "strategy_mo_natr20_ndx"


class Natr20NdxStrategy(AtrNormalizedNdxStrategy):
    """Keep the original filters and sizing; replace only the ranking score."""

    def compute_signals(self, pricing_data_df: pd.DataFrame) -> pd.DataFrame:
        signal_df = super().compute_signals(pricing_data_df)
        natr_feature_dict = {}
        for symbol_str in self.get_tradeable_symbol_list(pricing_data_df):
            raw_close_ser = signal_df[(symbol_str, "Unadjusted Close")]
            nominal_atr_ser = signal_df[(symbol_str, f"atr_{ATR_WINDOW_INT}_ser")]
            # *** CRITICAL *** All quantities end at decision Close_T. The base
            # has already validated raw anchors and produced ATR in T's units.
            # NATR20_T = 100 * ATR_nominal_T / RawClose_T.
            # Score_T = (ROC12_T / ATR_nominal_T) * RawClose_T
            #         = ROC12_T / (ATR_adjusted_T / AdjustedClose_T).
            natr_feature_dict[(symbol_str, "natr_20_pct_ser")] = 100.0 * nominal_atr_ser / raw_close_ser
            signal_df[(symbol_str, "risk_adj_score_ser")] *= raw_close_ser
        self._data_adjustment_policy_dict.update({
            "ranking_formula_str": "ROC12 / (SMA20_TRUE_RANGE / Close)",
            "atr_smoothing_str": "simple_20_session_mean",
            "research_history_status_str": "previously_seen_2000_2026_not_untouched_holdout",
        })
        return pd.concat([signal_df, pd.DataFrame(natr_feature_dict, index=signal_df.index)], axis=1)


def _config_obj(
    backtest_start_date_str: str | None,
    capital_base_float: float | None,
    end_date_str: str | None,
) -> AtrNormalizedNdxConfig:
    return replace(DEFAULT_CONFIG,
        backtest_start_date_str=backtest_start_date_str or DEFAULT_CONFIG.backtest_start_date_str,
        capital_base_float=DEFAULT_CONFIG.capital_base_float if capital_base_float is None else capital_base_float,
        end_date_str=end_date_str)


def _new_strategy_obj(
    config_obj: AtrNormalizedNdxConfig,
    universe_df: pd.DataFrame,
    rebalance_schedule_df: pd.DataFrame,
) -> Natr20NdxStrategy:
    strategy_obj = Natr20NdxStrategy(
        name=STRATEGY_NAME_STR,
        benchmarks=[config_obj.performance_benchmark_symbol_str],
        rebalance_schedule_df=rebalance_schedule_df,
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
    strategy_obj.trade_id_int = 0
    strategy_obj.current_trade_map = defaultdict(default_trade_id_int)
    configure_total_return_benchmark_provenance(strategy_obj, config_obj)
    return strategy_obj


def _run_with_prices_tuple(
    config_obj: AtrNormalizedNdxConfig, show_display_bool: bool,
) -> tuple[Natr20NdxStrategy, pd.DataFrame]:
    pricing_df, universe_df, schedule_df = get_atr_normalized_ndx_data(
        config_obj, include_total_return_benchmark_bool=True)
    strategy_obj = _new_strategy_obj(config_obj, universe_df, schedule_df)
    # *** CRITICAL *** Keep warmup before the requested start, but execute
    # only at Open_(T+1) on the requested calendar; never trade warmup rows.
    calendar_idx = pricing_df.index[pricing_df.index >= pd.Timestamp(config_obj.backtest_start_date_str)]
    run_daily(strategy_obj, pricing_df, calendar=calendar_idx,
        show_progress=show_display_bool, show_signal_progress_bool=show_display_bool,
        audit_override_bool=None)
    return strategy_obj, pricing_df


def run_variant(
    show_display_bool: bool = True,
    save_results_bool: bool = True,
    output_dir_str: str = "results",
    backtest_start_date_str: str | None = None,
    capital_base_float: float | None = None,
    end_date_str: str | None = None,
) -> Natr20NdxStrategy:
    config_obj = _config_obj(backtest_start_date_str, capital_base_float, end_date_str)
    strategy_obj, _pricing_df = _run_with_prices_tuple(config_obj, show_display_bool)
    if show_display_bool:
        print(strategy_obj.summary.to_string())
    if save_results_bool:
        save_results(strategy_obj, output_dir=output_dir_str)
    return strategy_obj


def build_capacity_analysis_inputs(
    show_display_bool: bool = False,
    backtest_start_date_str: str | None = None,
    capital_base_float: float | None = None,
    end_date_str: str | None = None,
) -> dict[str, object]:
    config_obj = _config_obj(backtest_start_date_str, capital_base_float, end_date_str)
    strategy_obj, pricing_df = _run_with_prices_tuple(config_obj, show_display_bool)
    strategy_obj.universe_df = None
    return dict(strategy_obj=strategy_obj, pricing_data_df=pricing_df,
        execution_policy_str="MOO", impact_profile_str="MOO_NASDAQ_LARGE")


def build_execution_timing_analysis_inputs() -> dict[str, object]:
    pricing_df, universe_df, schedule_df = get_atr_normalized_ndx_data(
        DEFAULT_CONFIG, include_total_return_benchmark_bool=True)
    # *** CRITICAL *** Timing analysis generates intent at Close_T; only
    # next_open is the causal baseline. Same-close fills remain diagnostic.
    decision_schedule_df = _map_rebalance_schedule_to_decision_close_schedule_df(schedule_df)

    def strategy_factory_fn() -> Natr20NdxStrategy:
        return _new_strategy_obj(DEFAULT_CONFIG, universe_df, decision_schedule_df)

    calendar_idx = pricing_df.index[pricing_df.index >= pd.Timestamp(DEFAULT_CONFIG.backtest_start_date_str)]
    return dict(strategy_factory_fn=strategy_factory_fn, pricing_data_df=pricing_df,
        calendar_idx=pd.DatetimeIndex(calendar_idx), order_generation_mode_str="signal_bar",
        risk_model_str="taa_rebalance",
        entry_timing_str_tuple=("same_close_moc", "next_open", "next_close"),
        exit_timing_str_tuple=("same_close_moc", "next_open", "next_close"),
        default_entry_timing_str="next_open", default_exit_timing_str="next_open")


if __name__ == "__main__":
    run_variant()
