"""Offline evidence comparison tests; never load Norgate or contact a broker."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from alpha.engine.order import MarketOrder
from scripts.research.mr_capsule_build_20261004.qualify_wiring_offline import (
    _canonical_research_intents, _host_fixture, compare_csv_frames,
    compare_gate_switches, parking_price_coverage,
)


def test_complete_gate_sessions_keep_old_and_new_switches_identical():
    pricing_index = pd.bdate_range("2024-01-08", periods=4)
    gate_open_ser = pd.Series([False, True, True, False], index=pricing_index)
    result_dict = compare_gate_switches(gate_open_ser, pricing_index, pricing_index[1:])
    assert result_dict["passed_bool"]
    assert result_dict["compared_history_session_count_int"] == 4
    assert result_dict["full_history_mismatch_count_int"] == 0


def test_missing_vix_row_exposes_old_switch_replay_on_an_executed_decision():
    pricing_index = pd.bdate_range("2024-01-08", periods=4)
    gate_open_ser = pd.Series([False, True, True], index=pricing_index[[0, 1, 3]])
    result_dict = compare_gate_switches(gate_open_ser, pricing_index, pricing_index[1:])
    assert not result_dict["passed_bool"]
    assert result_dict["mismatch_list"] == [{"decision_date_str": "2024-01-10", "old_switch_bool": True,
        "new_switch_bool": False, "used_by_baseline_bool": True}]


def test_first_input_boundary_difference_is_visible_but_not_a_baseline_trade():
    gate_index = pd.bdate_range("2024-01-08", periods=3)
    gate_open_ser = pd.Series([False, True, True], index=gate_index)
    result_dict = compare_gate_switches(gate_open_ser, gate_index[1:], gate_index[2:])
    assert result_dict["passed_bool"]
    assert result_dict["full_history_mismatch_count_int"] == 1
    assert result_dict["baseline_mismatch_count_int"] == 0


def _parking_inputs():
    pricing_index = pd.bdate_range("2024-01-08", periods=4)
    pricing_df = pd.DataFrame({(symbol_str, field_str): [np.nan, 100., 100., 100.]
        for symbol_str in ("BIL", "SPMO") for field_str in ("Open", "Close")}, index=pricing_index)
    transaction_df = pd.DataFrame({"asset": ["BIL", "BIL"], "bar": pricing_index[[1, 3]], "amount": [10., -10.]})
    return pricing_df, transaction_df, pricing_index


def test_unheld_preinception_gaps_are_not_held_parking_failures():
    pricing_df, transaction_df, pricing_index = _parking_inputs()
    result_dict = parking_price_coverage(pricing_df, transaction_df, pricing_index)
    assert result_dict["passed_bool"]
    assert result_dict["symbol_coverage_dict"]["BIL"]["held_before_session_count_int"] == 2


def test_empty_or_reordered_baseline_calendar_cannot_prove_parking_coverage():
    pricing_df, transaction_df, pricing_index = _parking_inputs()
    for invalid_index in (pricing_index[:0], pricing_index[::-1], pricing_index.append(pricing_index[:1])):
        assert not parking_price_coverage(pricing_df, transaction_df, invalid_index)["passed_bool"]


@pytest.mark.parametrize("field_str,row_int", [("Open", 3), ("Close", 2), ("Open", 1)])
def test_buy_hold_and_liquidation_sessions_all_require_observed_prices(field_str, row_int):
    pricing_df, transaction_df, pricing_index = _parking_inputs()
    pricing_df.loc[pricing_index[row_int], ("BIL", field_str)] = np.nan
    result_dict = parking_price_coverage(pricing_df, transaction_df, pricing_index)
    assert not result_dict["passed_bool"]
    assert result_dict["symbol_coverage_dict"]["BIL"]["failure_list"][0]["date_str"] == str(pricing_index[row_int].date())


def _transactions():
    return pd.DataFrame({"trade_id": [1, 2], "bar": ["2024-01-08", "2024-01-09"], "asset": ["AAA", "BBB"],
        "amount": [10., 20.], "price": [100., 50.], "order_id": [1, 2], "commission": [1., 1.]})


def test_constant_global_order_id_offset_is_reported_without_hiding_economic_fields():
    original_df = _transactions()
    current_df = original_df.assign(order_id=original_df["order_id"] + 11070)
    result_dict = compare_csv_frames(current_df, original_df)
    assert result_dict["passed_bool"]
    assert not result_dict["strict_equal_bool"]
    assert result_dict["economic_columns_equal_bool"]
    assert result_dict["differing_column_list"] == ["order_id"]
    assert result_dict["order_id_comparison_dict"]["offset_int"] == 11070


def test_integer_precision_loss_cannot_create_a_false_constant_order_id_offset():
    original_df = _transactions().assign(order_id=[0, 0])
    current_df = original_df.assign(order_id=[2**53, 2**53 + 1])
    result_dict = compare_csv_frames(current_df, original_df)
    assert not result_dict["passed_bool"]
    assert not result_dict["order_id_comparison_dict"]["constant_offset_allowed_bool"]


@pytest.mark.parametrize("column_str,value_obj", [("order_id", 999), ("trade_id", 999), ("price", 100. + 1e-12), ("asset", "CCC")])
def test_variable_order_offsets_and_any_other_difference_fail(column_str, value_obj):
    original_df = _transactions()
    current_df = original_df.assign(order_id=original_df["order_id"] + 11070)
    current_df.loc[0, column_str] = value_obj
    result_dict = compare_csv_frames(current_df, original_df)
    assert not result_dict["passed_bool"]
    assert column_str in result_dict["differing_column_list"]


def test_transaction_sequence_and_empty_artifacts_are_compared_explicitly():
    original_df = _transactions()
    assert not compare_csv_frames(original_df.iloc[::-1].reset_index(drop=True), original_df)["passed_bool"]
    assert compare_csv_frames(original_df.iloc[:0], original_df.iloc[:0])["strict_equal_bool"]


def test_independent_research_canonicalization_preserves_dollars_shares_and_priority():
    order_list = [MarketOrder("BBB", 10000., unit="value"), MarketOrder("AAA", 10000., unit="value"),
        MarketOrder("EXIT", 0., unit="value", target=True), MarketOrder("BIL", 123., unit="shares", target=True)]
    intent_dict = _canonical_research_intents(SimpleNamespace(get_orders=lambda: order_list), 100000.)
    assert intent_dict["entry_weight_dict"] == {"BBB": .1, "AAA": .1}
    assert intent_dict["entry_priority_list"] == ["BBB", "AAA"]
    assert intent_dict["exit_symbol_list"] == ["EXIT"]
    assert intent_dict["share_target_dict"] == {"BIL": 123.}


def _fixture_inputs():
    pricing_index = pd.bdate_range(end="2024-01-12", periods=300)
    pricing_df = pd.DataFrame({(symbol_str, field_str): value_float
        for symbol_str in ("AAA", "BBB", "BIL", "SPMO", "$SPX")
        for field_str, value_float in {"Open": 100., "High": 101., "Low": 99., "Close": 100., "Volume": 10000., "Dividend": 0.}.items()}, index=pricing_index)
    universe_df = pd.DataFrame(1, columns=["AAA", "BBB"], index=pricing_index)
    vix_close_ser = pd.Series(10., index=pd.bdate_range("1990-01-02", "2024-01-12"))
    vix_close_ser.loc[pd.Timestamp("2024-01-12")] = 50.
    return pricing_df, universe_df, vix_close_ser


@pytest.mark.parametrize("mode_str", ["cash", "bil", "spmo"])
def test_synthetic_fixture_uses_real_dv2_indicators_and_independent_state_without_norgate(tmp_path, mode_str):
    pricing_df, universe_df, vix_close_ser = _fixture_inputs()
    result_dict = _host_fixture("dv2", mode_str, pricing_df, universe_df, vix_close_ser, tmp_path, {"direct_dv2_pricing": "synthetic-input-hash"})
    assert result_dict["passed_bool"], result_dict
    assert result_dict["fixture_only_bool"] and not result_dict["broker_truth_bool"]
    assert result_dict["execution_date_str"] == "2024-01-16"
    assert result_dict["position_dict"]["AAA"] > 0.


def test_reference_cache_reuses_indicators_but_host_and_account_states_remain_independent(tmp_path, monkeypatch):
    from strategies.mr_capsule.dv2_vix_gated import DV2VixGatedStrategy

    pricing_df, universe_df, vix_close_ser = _fixture_inputs()
    original_compute_fn = DV2VixGatedStrategy.compute_signals
    computation_list = []

    def count_real_computation(strategy_obj, input_df):
        computation_list.append(strategy_obj.name)
        return original_compute_fn(strategy_obj, input_df)

    monkeypatch.setattr(DV2VixGatedStrategy, "compute_signals", count_real_computation)
    signal_cache_dict = {}
    result_list = [_host_fixture("dv2", mode_str, pricing_df, universe_df, vix_close_ser, tmp_path,
        {"direct_dv2_pricing": "synthetic-input-hash"}, signal_cache_dict) for mode_str in ("cash", "bil")]
    assert all(result_dict["passed_bool"] for result_dict in result_list)
    assert len(computation_list) == 3  # Two unmocked host computations; one shared reference computation.
    assert not result_list[0]["reference_indicator_cache_reused_bool"]
    assert result_list[1]["reference_indicator_cache_reused_bool"]
    assert "BIL" not in result_list[0]["position_dict"] and "BIL" in result_list[1]["position_dict"]
