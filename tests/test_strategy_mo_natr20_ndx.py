"""NATR20 formula, independent symbol units, causal prefixes and entry points."""
import numpy as np
import pandas as pd
import pytest

from strategies.momentum import strategy_mo_natr20_ndx as natr_module


@pytest.fixture
def pricing_df():
    session_idx = pd.bdate_range("2011-01-03", "2013-08-30")
    elapsed_vec = np.arange(len(session_idx), dtype=float)
    field_dict = {}
    for symbol_str, level_float, drift_float in [("AAA", 10.0, 0.0008), ("BBB", 50.0, 0.0005), ("SPY", 100.0, 0.0003)]:
        close_vec = level_float * np.exp(elapsed_vec * drift_float)
        for field_str, factor_float in [("Open", 1.0), ("Close", 1.0), ("High", 1.01), ("Low", 0.99), ("Unadjusted Close", 1.0)]:
            field_dict[(symbol_str, field_str)] = close_vec * factor_float
    return pd.DataFrame(field_dict, index=session_idx)


def make_strategy_obj(pricing_df):
    universe_df = pd.DataFrame(1, index=pricing_df.index, columns=["AAA", "BBB"])
    schedule_df = pd.DataFrame({"decision_date_ts": [pd.Timestamp("2013-06-28")]}, index=pd.to_datetime(["2013-07-01"]))
    return natr_module._new_strategy_obj(natr_module.DEFAULT_CONFIG, universe_df, schedule_df)


def test_score_matches_direct_fractional_true_range_mean(pricing_df):
    strategy_obj = make_strategy_obj(pricing_df)
    signal_df = strategy_obj.compute_signals(pricing_df)
    decision_ts = pd.Timestamp("2013-06-28")
    for symbol_str in ["AAA", "BBB"]:
        close_series = pricing_df[(symbol_str, "Close")]
        roc_float = close_series.loc[decision_ts] / close_series.loc["2012-06-29"] - 1.0
        # These deterministic bars have H-L greater than either overnight gap.
        # The independent reference uses exactly the twenty bars ending at T.
        true_range_mean_float = (0.02 * close_series.loc[:decision_ts].iloc[-20:]).mean()
        fractional_atr_float = true_range_mean_float / close_series.loc[decision_ts]
        assert signal_df.loc[decision_ts, (symbol_str, "risk_adj_score_ser")] == pytest.approx(roc_float / fractional_atr_float)
        assert signal_df.loc[decision_ts, (symbol_str, "natr_20_pct_ser")] == pytest.approx(100 * fractional_atr_float)
    assert strategy_obj.historical_share_units_bool is True


@pytest.mark.parametrize("scale_raw_bool", [False, True])
@pytest.mark.parametrize("factor_float", [40.0, 0.1])
def test_independent_symbol_units_cannot_change_score_or_selection(pricing_df, scale_raw_bool, factor_float):
    strategy_obj = make_strategy_obj(pricing_df)
    reference_df = strategy_obj.compute_signals(pricing_df)
    adjusted_df = pricing_df.copy()
    field_list = ["Open", "High", "Low", "Close"] + (["Unadjusted Close"] if scale_raw_bool else [])
    for field_str in field_list:
        adjusted_df[("AAA", field_str)] /= factor_float
    actual_df = make_strategy_obj(pricing_df).compute_signals(adjusted_df)
    for field_str in ["risk_adj_score_ser", "natr_20_pct_ser"]:
        pd.testing.assert_series_equal(actual_df[("AAA", field_str)], reference_df[("AAA", field_str)])
    decision_ts = pd.Timestamp("2013-06-28")
    strategy_obj.previous_bar = decision_ts
    pd.testing.assert_frame_equal(
        strategy_obj.get_ranked_candidate_feature_df(actual_df.loc[decision_ts])[["risk_adj_score_float"]],
        strategy_obj.get_ranked_candidate_feature_df(reference_df.loc[decision_ts])[["risk_adj_score_float"]])


@pytest.mark.parametrize("cutoff_str", ["2012-12-31", "2013-06-20"])
def test_prefix_and_future_perturbations_leave_decision_scores_unchanged(pricing_df, cutoff_str):
    strategy_obj = make_strategy_obj(pricing_df)
    reference_df = strategy_obj.compute_signals(pricing_df)
    prefix_df = strategy_obj.compute_signals(pricing_df.loc[:cutoff_str])
    perturbed_df = pricing_df.copy()
    perturbed_df.loc[perturbed_df.index > cutoff_str] *= 17.0
    future_df = strategy_obj.compute_signals(perturbed_df)
    for symbol_str in ["AAA", "BBB"]:
        column_tuple = (symbol_str, "risk_adj_score_ser")
        pd.testing.assert_series_equal(prefix_df[column_tuple], reference_df.loc[:cutoff_str, column_tuple])
        pd.testing.assert_series_equal(future_df.loc[:cutoff_str, column_tuple], reference_df.loc[:cutoff_str, column_tuple])


def test_entrypoints_build_the_natr_strategy_and_preserve_timing(monkeypatch, pricing_df):
    reference_obj = make_strategy_obj(pricing_df)
    schedule_df = reference_obj.rebalance_schedule_df
    monkeypatch.setattr(natr_module, "get_atr_normalized_ndx_data", lambda *args, **kwargs: (pricing_df, reference_obj.universe_df, schedule_df))
    call_list = []
    monkeypatch.setattr(natr_module, "run_daily", lambda strategy_obj, pricing_data_df, **kwargs: call_list.append((strategy_obj, kwargs)))
    strategy_obj = natr_module.run_variant(show_display_bool=False, save_results_bool=False, backtest_start_date_str="2012-01-01", capital_base_float=250000.0)
    assert type(strategy_obj) is natr_module.Natr20NdxStrategy
    assert strategy_obj.name == natr_module.STRATEGY_NAME_STR
    assert call_list[0][1]["calendar"].min() >= pd.Timestamp("2012-01-01")
    capacity_dict = natr_module.build_capacity_analysis_inputs()
    assert type(capacity_dict["strategy_obj"]) is natr_module.Natr20NdxStrategy
    assert capacity_dict["execution_policy_str"] == "MOO"
    timing_dict = natr_module.build_execution_timing_analysis_inputs()
    assert timing_dict["default_entry_timing_str"] == "next_open"
    assert timing_dict["default_exit_timing_str"] == "next_open"
    assert type(timing_dict["strategy_factory_fn"]()) is natr_module.Natr20NdxStrategy


def test_bench_discovers_separate_research_module():
    from alpha.bench import catalog
    entry_obj = catalog.get_strategy_by_module("strategies.momentum.strategy_mo_natr20_ndx")
    assert entry_obj is not None
    assert entry_obj.stem_str == natr_module.STRATEGY_NAME_STR
