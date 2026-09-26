"""NATR20 formula, independent symbol units, causal prefixes and entry points."""
import numpy as np
import pandas as pd
import pytest
import ast
from pathlib import Path
from alpha.engine.backtest import run_daily

from strategies.momentum import strategy_mo_natr20_ndx_vxn_scaled as natr_module


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
    vxn_close_ser = pd.Series([44.0, 11.0], index=pd.to_datetime(["2013-06-28", "2013-07-01"]))
    strategy_obj = natr_module.Natr20VxnScaledNdxStrategy(
        name=natr_module.STRATEGY_NAME_STR, benchmarks=[], rebalance_schedule_df=schedule_df,
        vxn_scale_signal_df=natr_module.compute_vxn_scale_signal_df(vxn_close_ser),
    )
    strategy_obj.universe_df = universe_df
    return strategy_obj


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




def test_independent_source_has_no_strategy_imports_or_sibling_inheritance():
    module_ast = ast.parse(Path(natr_module.__file__).read_text(encoding="utf-8"))
    for node_obj in ast.walk(module_ast):
        if isinstance(node_obj, ast.ImportFrom):
            assert not (node_obj.module or "").startswith("strategies")
        if isinstance(node_obj, ast.Import):
            assert all(not alias_obj.name.startswith("strategies") for alias_obj in node_obj.names)
    assert natr_module.Natr20VxnScaledNdxStrategy.__bases__ == (natr_module.Strategy,)


@pytest.mark.parametrize("close_float, expected_float", [(11.0, 1.0), (22.0, 1.0), (44.0, 0.5), (88.0, 0.25), (150.0, 0.25)])
def test_vxn_formula_bounds(close_float, expected_float):
    vxn_ser = pd.Series([close_float], index=pd.to_datetime(["2013-06-28"]))
    assert natr_module.compute_vxn_scale_signal_df(vxn_ser).iloc[0].vxn_exposure_scale_float == expected_float


@pytest.mark.parametrize("invalid_float", [0.0, -1.0, np.inf, -np.inf])
def test_invalid_observed_vxn_is_rejected(invalid_float):
    vxn_ser = pd.Series([invalid_float], index=pd.to_datetime(["2013-06-28"]))
    with pytest.raises(ValueError, match="finite and positive"):
        natr_module.compute_vxn_scale_signal_df(vxn_ser)


def test_vxn_asof_never_uses_future_and_fails_without_prior_observation(pricing_df):
    strategy_obj = make_strategy_obj(pricing_df)
    assert natr_module.get_asof_vxn_scale_float(strategy_obj.vxn_scale_signal_df, pd.Timestamp("2013-06-28")) == 0.5
    assert natr_module.get_asof_vxn_scale_float(strategy_obj.vxn_scale_signal_df, pd.Timestamp("2013-06-30")) == 0.5
    with pytest.raises(RuntimeError, match="No VXN scale"):
        natr_module.get_asof_vxn_scale_float(strategy_obj.vxn_scale_signal_df, pd.Timestamp("2013-06-27"))


def test_vxn_changes_only_position_weight_and_unfilled_slots_remain_cash(pricing_df):
    strategy_obj = make_strategy_obj(pricing_df)
    signal_df = strategy_obj.compute_signals(pricing_df)
    decision_ts = pd.Timestamp("2013-06-28")
    strategy_obj.previous_bar = decision_ts
    weight_ser = strategy_obj.get_target_weight_ser(signal_df.loc[decision_ts])
    assert weight_ser.to_dict() == {"AAA": 0.05, "BBB": 0.05}
    strategy_obj.vxn_scale_signal_df.loc["2013-07-01", "vxn_exposure_scale_float"] = 0.25
    pd.testing.assert_series_equal(strategy_obj.get_target_weight_ser(signal_df.loc[decision_ts]), weight_ser)
    strategy_obj.vxn_scale_signal_df = strategy_obj.vxn_scale_signal_df.loc[:decision_ts]
    pd.testing.assert_series_equal(strategy_obj.get_target_weight_ser(signal_df.loc[decision_ts]), weight_ser)
    signal_df.loc[decision_ts, ("SPY", "regime_pass_bool")] = False
    assert strategy_obj.get_target_weight_ser(signal_df.loc[decision_ts]).empty


def test_month_end_partial_cutoff_and_exchange_holiday():
    price_df = pd.DataFrame({"AAA": 10.0}, index=pd.bdate_range("2024-02-01", "2024-03-28"))
    assert natr_module.get_monthly_decision_close_df(price_df).index[-1] == pd.Timestamp("2024-03-28")
    assert natr_module.get_monthly_decision_close_df(price_df.loc[:"2024-03-20"]).index[-1] == pd.Timestamp("2024-02-29")
    execution_idx = pd.DatetimeIndex(["2024-03-28", "2024-04-01", "2024-04-02"])
    schedule_df = natr_module.map_month_end_decision_dates_to_rebalance_schedule_df(pd.DatetimeIndex(["2024-03-28"]), execution_idx)
    assert schedule_df.index.tolist() == [pd.Timestamp("2024-04-01")]


def test_pit_membership_uses_decision_day_not_future(pricing_df):
    strategy_obj = make_strategy_obj(pricing_df)
    strategy_obj.universe_df.loc[:"2013-06-28", "AAA"] = 0
    strategy_obj.previous_bar = pd.Timestamp("2013-06-28")
    signal_df = strategy_obj.compute_signals(pricing_df)
    assert strategy_obj.get_target_weight_ser(signal_df.loc["2013-06-28"]).to_dict() == {"BBB": 0.05}


def test_real_engine_prior_close_vxn_rounding_commissions_and_split_units(pricing_df):
    strategy_list = []
    for split_float in [1.0, 40.0]:
        adjusted_df = pricing_df.copy()
        for field_str in ["Open", "High", "Low", "Close"]:
            adjusted_df[("AAA", field_str)] /= split_float
        # Only execution-day open jumps; order intent must remain Close_T based.
        adjusted_df.loc["2013-07-01", ("AAA", "Open")] *= 1.12
        strategy_obj = make_strategy_obj(adjusted_df)
        run_daily(strategy_obj, adjusted_df, calendar=pd.to_datetime(["2013-07-01", "2013-07-02"]),
                  show_progress=False, show_signal_progress_bool=False)
        transaction_df = strategy_obj.get_transactions()
        assert len(transaction_df) == 2
        for symbol_str in ["AAA", "BBB"]:
            row_ser = transaction_df.loc[transaction_df.asset == symbol_str].iloc[0]
            factor_float = split_float if symbol_str == "AAA" else 1.0
            nominal_share_int = int(100000.0 * 0.05 / pricing_df.loc["2013-06-28", (symbol_str, "Unadjusted Close")])
            assert row_ser.amount / factor_float == pytest.approx(nominal_share_int)
            assert row_ser.commission == pytest.approx(max(1.0, nominal_share_int * 0.005))
            assert row_ser.price == pytest.approx(adjusted_df.loc["2013-07-01", (symbol_str, "Open")] * 1.00025)
        strategy_list.append(strategy_obj)
    np.testing.assert_allclose(strategy_list[0].results.total_value, strategy_list[1].results.total_value, rtol=0, atol=1e-9)
    np.testing.assert_allclose(strategy_list[0].results.cash, strategy_list[1].results.cash, rtol=0, atol=1e-9)


def test_entrypoints_preserve_independent_class_and_timing(monkeypatch, pricing_df):
    reference_obj = make_strategy_obj(pricing_df)
    input_tuple = (pricing_df, reference_obj.universe_df, reference_obj.rebalance_schedule_df, reference_obj.vxn_scale_signal_df)
    monkeypatch.setattr(natr_module, "get_natr20_vxn_scaled_ndx_data", lambda *args, **kwargs: input_tuple)
    call_list = []
    monkeypatch.setattr(natr_module, "run_daily", lambda strategy_obj, pricing_data_df, **kwargs: call_list.append((strategy_obj, kwargs)))
    strategy_obj = natr_module.run_variant(show_display_bool=False, save_results_bool=False, backtest_start_date_str="2012-01-01", capital_base_float=250000.0)
    assert type(strategy_obj) is natr_module.Natr20VxnScaledNdxStrategy
    assert strategy_obj.name == natr_module.STRATEGY_NAME_STR
    assert call_list[0][1]["calendar"].min() >= pd.Timestamp("2012-01-01")
    capacity_dict = natr_module.build_capacity_analysis_inputs()
    assert type(capacity_dict["strategy_obj"]) is natr_module.Natr20VxnScaledNdxStrategy
    timing_dict = natr_module.build_execution_timing_analysis_inputs()
    assert timing_dict["default_entry_timing_str"] == "next_open"
    assert timing_dict["default_exit_timing_str"] == "next_open"
    assert type(timing_dict["strategy_factory_fn"]()) is natr_module.Natr20VxnScaledNdxStrategy


def test_bench_discovers_separate_research_module():
    from alpha.bench import catalog
    entry_obj = catalog.get_strategy_by_module("strategies.momentum.strategy_mo_natr20_ndx_vxn_scaled")
    assert entry_obj is not None
    assert entry_obj.stem_str == natr_module.STRATEGY_NAME_STR
