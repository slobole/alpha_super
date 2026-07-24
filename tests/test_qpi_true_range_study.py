from collections import defaultdict

import numpy as np
import pandas as pd
import pytest

from scripts.research.run_qpi_true_range_study import (
    VARIANT_BASELINE_STR,
    VARIANT_EXTREME_GUARD_STR,
    VARIANT_RANGE_CONFIRMATION_STR,
    QPITrueRangeResearchStrategy,
    compute_event_target_dict,
    compute_true_range_percentile_df,
    summarize_candidate_event_df,
)
from strategies.qpi.strategy_mr_qpi_ibs_rsi_exit import (
    QPIIbsRsiExitStrategy,
    default_trade_id_int,
)


def _close_row_ser() -> pd.Series:
    row_dict: dict[tuple[str, str], float] = {}
    value_dict = {
        "AAA": {
            "Close": 110.0,
            "Turnover": 4_000_000.0,
            "qpi_value_ser": 10.0,
            "sma_200_price_ser": 100.0,
            "three_day_return_ser": -0.05,
            "ibs_value_ser": 0.05,
            "true_range_percentile_ser": 85.0,
        },
        "BBB": {
            "Close": 120.0,
            "Turnover": 8_000_000.0,
            "qpi_value_ser": 12.0,
            "sma_200_price_ser": 100.0,
            "three_day_return_ser": -0.04,
            "ibs_value_ser": 0.04,
            "true_range_percentile_ser": 97.0,
        },
        "CCC": {
            "Close": 115.0,
            "Turnover": 6_000_000.0,
            "qpi_value_ser": 8.0,
            "sma_200_price_ser": 100.0,
            "three_day_return_ser": -0.03,
            "ibs_value_ser": 0.03,
            "true_range_percentile_ser": 60.0,
        },
    }
    for symbol_str, field_value_dict in value_dict.items():
        for field_str, value_float in field_value_dict.items():
            row_dict[(symbol_str, field_str)] = value_float
    return pd.Series(row_dict, dtype=float)


def _strategy(variant_mode_str: str) -> QPITrueRangeResearchStrategy:
    strategy_obj = QPITrueRangeResearchStrategy(
        name="qpi_range_test",
        benchmarks=[],
        variant_mode_str=variant_mode_str,
    )
    strategy_obj.previous_bar = pd.Timestamp("2024-01-05")
    strategy_obj.universe_df = pd.DataFrame(
        {"AAA": [1], "BBB": [1], "CCC": [1]},
        index=[pd.Timestamp("2024-01-05")],
    )
    strategy_obj.current_trade_map = defaultdict(default_trade_id_int)
    return strategy_obj


def test_true_range_percentile_includes_gap_and_current_observation():
    date_index = pd.bdate_range("2024-01-02", periods=4)
    close_price_df = pd.DataFrame({"AAA": [100.0, 100.0, 100.0, 100.0]}, index=date_index)
    high_price_df = pd.DataFrame({"AAA": [101.0, 102.0, 103.0, 110.0]}, index=date_index)
    low_price_df = pd.DataFrame({"AAA": [99.0, 99.0, 99.0, 109.0]}, index=date_index)

    range_percentile_df = compute_true_range_percentile_df(
        close_price_df,
        high_price_df,
        low_price_df,
        lookback_day_int=3,
    )

    assert float(range_percentile_df.loc[date_index[3], "AAA"]) == pytest.approx(100.0)


def test_true_range_percentile_is_unchanged_by_future_prices():
    date_index = pd.bdate_range("2024-01-02", periods=7)
    close_price_df = pd.DataFrame({"AAA": [100.0] * 7}, index=date_index)
    high_price_df = pd.DataFrame({"AAA": [101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0]}, index=date_index)
    low_price_df = pd.DataFrame({"AAA": [99.0] * 7}, index=date_index)
    original_feature_df = compute_true_range_percentile_df(
        close_price_df,
        high_price_df,
        low_price_df,
        lookback_day_int=3,
    )
    changed_high_price_df = high_price_df.copy()
    changed_high_price_df.loc[date_index[5]:, "AAA"] *= 2.0
    changed_feature_df = compute_true_range_percentile_df(
        close_price_df,
        changed_high_price_df,
        low_price_df,
        lookback_day_int=3,
    )

    assert float(changed_feature_df.loc[date_index[4], "AAA"]) == pytest.approx(
        float(original_feature_df.loc[date_index[4], "AAA"])
    )


def test_baseline_opportunities_match_active_qpi_with_missing_range_value():
    close_row_ser = _close_row_ser()
    close_row_ser.loc[("BBB", "true_range_percentile_ser")] = np.nan
    research_strategy_obj = _strategy(VARIANT_BASELINE_STR)
    baseline_strategy_obj = QPIIbsRsiExitStrategy(name="baseline", benchmarks=[])
    baseline_strategy_obj.previous_bar = research_strategy_obj.previous_bar
    baseline_strategy_obj.universe_df = research_strategy_obj.universe_df
    baseline_close_row_ser = close_row_ser[
        close_row_ser.index.get_level_values(1) != "true_range_percentile_ser"
    ]

    assert research_strategy_obj.get_opportunity_list(
        close_row_ser
    ) == baseline_strategy_obj.get_opportunity_list(baseline_close_row_ser)
    assert research_strategy_obj.get_opportunity_list(close_row_ser) == ["BBB", "CCC", "AAA"]


def test_range_confirmation_requires_eightieth_percentile_and_keeps_turnover_rank():
    strategy_obj = _strategy(VARIANT_RANGE_CONFIRMATION_STR)

    assert strategy_obj.get_opportunity_list(_close_row_ser()) == ["BBB", "AAA"]


def test_extreme_range_guard_excludes_top_five_percent_and_keeps_turnover_rank():
    strategy_obj = _strategy(VARIANT_EXTREME_GUARD_STR)

    assert strategy_obj.get_opportunity_list(_close_row_ser()) == ["CCC", "AAA"]


def test_event_targets_enter_next_open_and_exit_open_after_close_signal():
    target_dict = compute_event_target_dict(
        decision_idx_int=0,
        open_price_arr=np.array([99.0, 100.0, 101.0, 105.0]),
        high_price_arr=np.array([100.0, 103.0, 104.0, 106.0]),
        low_price_arr=np.array([98.0, 97.0, 100.0, 104.0]),
        close_price_arr=np.array([99.0, 102.0, 103.0, 105.0]),
        ibs_value_arr=np.array([0.05, 0.20, 0.95, 0.50]),
        rsi2_value_arr=np.array([20.0, 30.0, 40.0, 50.0]),
    )

    assert target_dict["entry_open_price_float"] == pytest.approx(100.0)
    assert target_dict["forward_1d_return_pct_float"] == pytest.approx(2.0)
    assert target_dict["exit_open_return_pct_float"] == pytest.approx(5.0)
    assert target_dict["holding_session_count_float"] == pytest.approx(2.0)


def test_event_hit_rate_excludes_missing_target_returns():
    event_df = pd.DataFrame(
        {
            "sample_period_str": ["validation_2016_plus"] * 2,
            "range_bucket_str": ["20_50"] * 2,
            "exit_observed_bool": [True, False],
            "entry_open_price_float": [100.0, 100.0],
            "forward_1d_return_pct_float": [1.0, np.nan],
            "forward_2d_return_pct_float": [1.0, np.nan],
            "forward_3d_return_pct_float": [1.0, np.nan],
            "forward_5d_return_pct_float": [1.0, np.nan],
            "exit_open_return_pct_float": [1.0, np.nan],
            "holding_session_count_float": [2.0, np.nan],
            "mae_to_exit_pct_float": [-1.0, np.nan],
            "mfe_to_exit_pct_float": [2.0, np.nan],
            "forward_5d_mae_pct_float": [-1.0, np.nan],
            "forward_5d_mfe_pct_float": [2.0, np.nan],
        }
    )

    summary_df = summarize_candidate_event_df(event_df)
    summary_row_ser = summary_df[
        (summary_df["sample_period_str"] == "validation_2016_plus")
        & (summary_df["range_bucket_str"] == "20_50")
    ].iloc[0]

    assert float(summary_row_ser["forward_1d_hit_pct_float"]) == pytest.approx(100.0)
    assert float(summary_row_ser["exit_return_hit_pct_float"]) == pytest.approx(100.0)
    assert int(summary_row_ser["entry_label_count_int"]) == 2
    assert int(summary_row_ser["forward_1d_label_count_int"]) == 1
    assert int(summary_row_ser["exit_return_label_count_int"]) == 1


def test_research_strategy_rejects_unknown_variant():
    with pytest.raises(ValueError, match="variant_mode_str"):
        _strategy("unknown")
