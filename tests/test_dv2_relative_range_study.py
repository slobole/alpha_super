from collections import defaultdict

import numpy as np
import pandas as pd
import pytest

from scripts.research.run_dv2_relative_range_study import (
    VARIANT_BASELINE_STR,
    VARIANT_RANGE_FILTER_STR,
    VARIANT_RANGE_RANK_STR,
    DV2RelativeRangeResearchStrategy,
    compute_relative_range_feature_df,
)
from strategies.dv2.strategy_mr_dv2 import DVO2Strategy, default_trade_id_int


def _close_row_ser() -> pd.Series:
    row_dict: dict[tuple[str, str], float] = {}
    value_dict = {
        "AAA": {"dv2": 5.0, "Close": 110.0, "sma_200": 100.0, "p126d_return": 0.10, "natr": 4.0, "relative_range_ser": 1.2},
        "BBB": {"dv2": 6.0, "Close": 120.0, "sma_200": 100.0, "p126d_return": 0.12, "natr": 8.0, "relative_range_ser": 0.8},
        "CCC": {"dv2": 7.0, "Close": 115.0, "sma_200": 100.0, "p126d_return": 0.08, "natr": 6.0, "relative_range_ser": 2.0},
    }
    for symbol_str, field_value_dict in value_dict.items():
        for field_str, value_float in field_value_dict.items():
            row_dict[(symbol_str, field_str)] = value_float
    return pd.Series(row_dict, dtype=float)


def _strategy(variant_mode_str: str) -> DV2RelativeRangeResearchStrategy:
    strategy_obj = DV2RelativeRangeResearchStrategy(
        name="dv2_range_test",
        benchmarks=[],
        variant_mode_str=variant_mode_str,
    )
    strategy_obj.previous_bar = pd.Timestamp("2024-01-05")
    strategy_obj.universe_df = pd.DataFrame(
        {"AAA": [1], "BBB": [1], "CCC": [1]},
        index=[pd.Timestamp("2024-01-05")],
    )
    strategy_obj.current_trade = defaultdict(default_trade_id_int)
    return strategy_obj


def test_relative_range_denominator_uses_only_prior_ranges():
    date_index = pd.bdate_range("2024-01-02", periods=6)
    low_price_df = pd.DataFrame({"AAA": [100.0] * 6}, index=date_index)
    log_range_ser = pd.Series([0.01, 0.02, 0.04, 0.03, 0.08, 0.06], index=date_index)
    high_price_df = pd.DataFrame({"AAA": 100.0 * np.exp(log_range_ser)}, index=date_index)

    relative_range_df = compute_relative_range_feature_df(
        high_price_df,
        low_price_df,
        lookback_day_int=3,
    )
    expected_denominator_float = float(log_range_ser.iloc[:3].std())

    assert float(relative_range_df.loc[date_index[3], "AAA"]) == pytest.approx(
        float(log_range_ser.iloc[3] / expected_denominator_float)
    )


def test_relative_range_feature_is_unchanged_by_future_prices():
    date_index = pd.bdate_range("2024-01-02", periods=7)
    low_price_df = pd.DataFrame({"AAA": [100.0] * 7}, index=date_index)
    high_price_df = pd.DataFrame(
        {"AAA": 100.0 * np.exp([0.01, 0.02, 0.04, 0.03, 0.08, 0.06, 0.05])},
        index=date_index,
    )
    original_feature_df = compute_relative_range_feature_df(
        high_price_df,
        low_price_df,
        lookback_day_int=3,
    )
    changed_high_price_df = high_price_df.copy()
    changed_high_price_df.loc[date_index[5]:, "AAA"] *= 1.5
    changed_feature_df = compute_relative_range_feature_df(
        changed_high_price_df,
        low_price_df,
        lookback_day_int=3,
    )

    assert float(changed_feature_df.loc[date_index[4], "AAA"]) == pytest.approx(
        float(original_feature_df.loc[date_index[4], "AAA"])
    )


def test_dv2_declares_signal_execution_and_benchmark_adjustment_provenance():
    date_index = pd.bdate_range("2024-01-02", periods=5)
    close_price_vec = np.linspace(100.0, 104.0, len(date_index))
    pricing_data_df = pd.DataFrame(
        {
            ("AAA", "High"): close_price_vec + 1.0,
            ("AAA", "Low"): close_price_vec - 1.0,
            ("AAA", "Close"): close_price_vec,
        },
        index=date_index,
    )
    pricing_data_df.columns = pd.MultiIndex.from_tuples(pricing_data_df.columns)
    strategy_obj = DVO2Strategy(name="dv2_adjustment_test", benchmarks=[])

    strategy_obj.compute_signals(pricing_data_df)

    assert strategy_obj._data_adjustment_policy_dict == {
        "stock_signal_adjustment_str": "CAPITALSPECIAL",
        "execution_and_marks_adjustment_str": "CAPITALSPECIAL",
        "performance_benchmark_adjustment_str": "TOTALRETURN",
    }


def test_dv2_production_paths_declare_total_return_benchmark_before_run(
    monkeypatch,
):
    date_index = pd.bdate_range("2024-01-02", periods=2)
    pricing_data_df = pd.DataFrame(
        {
            ("AAA", "Open"): [100.0, 101.0],
            ("AAA", "High"): [101.0, 102.0],
            ("AAA", "Low"): [99.0, 100.0],
            ("AAA", "Close"): [100.0, 101.0],
            ("AAA", "Dividend"): [0.0, 0.0],
            ("$SPX", "Open"): [4_700.0, 4_710.0],
            ("$SPX", "High"): [4_710.0, 4_720.0],
            ("$SPX", "Low"): [4_690.0, 4_700.0],
            ("$SPX", "Close"): [4_700.0, 4_710.0],
            ("$SPX", "Dividend"): [0.0, 0.0],
        },
        index=date_index,
    )
    pricing_data_df.columns = pd.MultiIndex.from_tuples(pricing_data_df.columns)
    pricing_data_df.attrs["norgate_adjustment_by_symbol_dict"] = {
        "AAA": "CAPITALSPECIAL",
        "$SPX": "TOTALRETURN",
    }
    universe_df = pd.DataFrame({"AAA": [1, 1]}, index=date_index)
    captured_adjustment_list: list[str] = []

    monkeypatch.setattr(
        "strategies.dv2.strategy_mr_dv2.build_index_constituent_matrix",
        lambda indexname: (["AAA"], universe_df),
    )
    monkeypatch.setattr(
        "strategies.dv2.strategy_mr_dv2.get_prices",
        lambda *args, **kwargs: pricing_data_df,
    )

    def run_daily_stub(strategy_obj, *args, **kwargs):
        captured_adjustment_list.append(
            strategy_obj._performance_benchmark_adjustment_str
        )

    monkeypatch.setattr(
        "strategies.dv2.strategy_mr_dv2.run_daily",
        run_daily_stub,
    )

    from strategies.dv2 import strategy_mr_dv2 as dv2_module

    dv2_module.build_capacity_analysis_inputs(
        backtest_start_date_str="2024-01-02",
        end_date_str="2024-01-03",
    )
    strategy_obj = dv2_module.run_variant(
        show_display_bool=False,
        save_results_bool=False,
        backtest_start_date_str="2024-01-02",
        end_date_str="2024-01-03",
    )

    assert captured_adjustment_list == ["TOTALRETURN", "TOTALRETURN"]
    assert strategy_obj._performance_benchmark_adjustment_str == "TOTALRETURN"


def test_baseline_opportunities_match_current_dv2_natr_ranking():
    close_row_ser = _close_row_ser()
    research_strategy_obj = _strategy(VARIANT_BASELINE_STR)
    baseline_strategy_obj = DVO2Strategy(
        name="baseline",
        benchmarks=[],
        capital_base=100_000.0,
        slippage=0.00025,
        commission_per_share=0.005,
        commission_minimum=1.0,
    )
    baseline_strategy_obj.previous_bar = research_strategy_obj.previous_bar
    baseline_strategy_obj.universe_df = research_strategy_obj.universe_df
    baseline_close_row_ser = close_row_ser[
        close_row_ser.index.get_level_values(1) != "relative_range_ser"
    ]

    assert research_strategy_obj.get_opportunities(close_row_ser) == baseline_strategy_obj.get_opportunities(baseline_close_row_ser)
    assert research_strategy_obj.get_opportunities(close_row_ser) == ["BBB", "CCC", "AAA"]


def test_missing_relative_range_does_not_change_baseline_opportunities():
    close_row_ser = _close_row_ser()
    close_row_ser.loc[("BBB", "relative_range_ser")] = np.nan
    research_strategy_obj = _strategy(VARIANT_BASELINE_STR)
    baseline_strategy_obj = DVO2Strategy(
        name="baseline",
        benchmarks=[],
        capital_base=100_000.0,
        slippage=0.00025,
        commission_per_share=0.005,
        commission_minimum=1.0,
    )
    baseline_strategy_obj.previous_bar = research_strategy_obj.previous_bar
    baseline_strategy_obj.universe_df = research_strategy_obj.universe_df
    baseline_close_row_ser = close_row_ser[
        close_row_ser.index.get_level_values(1) != "relative_range_ser"
    ]

    assert research_strategy_obj.get_opportunities(close_row_ser) == baseline_strategy_obj.get_opportunities(baseline_close_row_ser)
    assert research_strategy_obj.get_opportunities(close_row_ser) == ["BBB", "CCC", "AAA"]


def test_range_filter_preserves_natr_ranking_after_filtering():
    strategy_obj = _strategy(VARIANT_RANGE_FILTER_STR)

    assert strategy_obj.get_opportunities(_close_row_ser()) == ["CCC", "AAA"]


def test_range_rank_replaces_natr_order_without_filtering():
    strategy_obj = _strategy(VARIANT_RANGE_RANK_STR)

    assert strategy_obj.get_opportunities(_close_row_ser()) == ["CCC", "AAA", "BBB"]


def test_research_strategy_rejects_unknown_variant():
    with pytest.raises(ValueError, match="variant_mode_str"):
        _strategy("unknown")
