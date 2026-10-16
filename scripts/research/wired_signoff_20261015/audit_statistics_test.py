"""Boundary tests for saved-ledger diagnostics, independent of trading code."""
import gzip
import json
import pickle
import shutil

import numpy as np
import pandas as pd
import pytest

from audit_statistics import (common_close_comparison, equity_metrics, ledger_stress,
    load_selected_prices, order_diagnostics, process_baseline, select_benchmarks, subperiod_dict,
    hash_dict, verify_input_manifest, liquidity_period_details, update_index)
from audit_statistics import resolve_input_sources, normalize_trace, verify_repaired_hpi_trace, REPO_PATH


def test_first_session_loss_and_peak_are_not_erased():
    equity_ser = pd.Series([90.0, 99.0, 89.0], index=pd.date_range("2025-01-02", periods=3))
    report_dict = equity_metrics(equity_ser, 100.0)
    assert report_dict["first_return"] == pytest.approx(-.1)
    assert report_dict["max_drawdown"] == pytest.approx(-.11)


def test_subperiod_retains_first_return_from_preceding_close():
    equity_ser = pd.Series([100.0, 90.0, 99.0], index=pd.to_datetime(["2021-12-31", "2022-01-03", "2022-01-04"]))
    report_dict = subperiod_dict(equity_ser, 100.0, "2022-01-01")
    assert report_dict["first_return"] == pytest.approx(-.1)
    assert report_dict["total_return"] == pytest.approx(-.01)
    assert report_dict["initial_equity"] == 100


def test_comparison_anchors_both_at_common_close_and_never_bridges_missing_price():
    equity_ser = pd.Series([90.0, 99.0, 89.1, 98.01], index=pd.date_range("2025-01-02", periods=4))
    benchmark_ser = equity_ser * 10
    report_dict, chart_df = common_close_comparison(equity_ser, benchmark_ser, window_int=2)
    assert report_dict["strategy"]["first_return"] == pytest.approx(.1)
    assert report_dict["strategy"]["total_return"] == pytest.approx(report_dict["benchmark"]["total_return"])
    assert report_dict["daily_beta"] == pytest.approx(1)
    assert chart_df.iloc[0]["strategy_common_close_equity"] == 1_000_000
    benchmark_ser.iloc[1] = np.nan
    unavailable_dict, _ = common_close_comparison(equity_ser, benchmark_ser)
    assert unavailable_dict["status"] == "unavailable"
    assert unavailable_dict["coverage"]["missing_dates"] == ["2025-01-03"]


def test_rolling_estimates_do_not_see_future_tail():
    date_idx = pd.bdate_range("2024-01-02", periods=145)
    return_vec = .001 + .005 * np.sin(np.arange(145))
    benchmark_ser = pd.Series(100 * np.cumprod(1 + return_vec), index=date_idx)
    equity_ser = benchmark_ser * 2
    original_dict, original_df = common_close_comparison(equity_ser, benchmark_ser)
    assert original_dict["monthly_beta"] == pytest.approx(1)
    changed_ser = benchmark_ser.copy()
    changed_ser.iloc[135:] *= 1.5
    _, changed_df = common_close_comparison(equity_ser, changed_ser)
    pd.testing.assert_frame_equal(original_df.iloc[:135], changed_df.iloc[:135])
    assert original_df["rolling126_beta"].iloc[:126].isna().all()
    assert original_df["rolling126_beta"].iloc[126] == pytest.approx(1)


def _order_case():
    date_idx = pd.bdate_range("2025-01-02", periods=25)
    pricing_df = pd.DataFrame({("AAA", "Turnover"): [100.0] * 20 + [1e9] * 5,
        ("AAA", "Unadjusted Close"): [200.0] * 20 + [1000.0] * 5}, index=date_idx)
    equity_ser = pd.Series(2_000_000.0, index=date_idx)
    transaction_df = pd.DataFrame({"bar": [date_idx[20], date_idx[19], date_idx[20]],
        "asset": ["AAA", "AAA", "MISSING"], "amount": [1, 1, 1], "price": [10, 10, 10]})
    trace_list = [{"execution_date": date_idx[20], "decision_date": date_idx[19], "nav": 2_000_000.0}]
    return pricing_df, equity_ser, transaction_df, trace_list


def test_adv_is_prior_only_normalized_and_missing_orders_remain_explicit():
    pricing_df, equity_ser, transaction_df, trace_list = _order_case()
    report_dict, order_df = order_diagnostics(transaction_df, pricing_df, equity_ser, 1_000_000, trace_list)
    assert len(order_df) == report_dict["order_count"] == 3
    assert report_dict["missing_adv_order_count"] == 2
    assert report_dict["missing_adv_notional"] == 20
    assert order_df.iloc[0]["native_trailing20_median_turnover"] == 100
    assert order_df.iloc[0]["constant_1m_notional"] == 5
    assert order_df.iloc[0]["constant_1m_participation"] == .05
    assert order_df.iloc[0]["one_raw_share_1m_weight_bound"] == .0002
    assert report_dict["missing_nominal_price_order_count"] == 1


def test_recent_capacity_separates_old_gaps_from_current_endpoint_information():
    date_idx = pd.bdate_range("2026-01-02", periods=25)
    pricing_df = pd.DataFrame({("AAA", "Turnover"): [100.0] * 24 + [900.0]}, index=date_idx)
    order_df = pd.DataFrame({"execution_date": [pd.Timestamp("2020-01-02"), date_idx[-1]],
        "asset": ["AAA", "AAA"], "adv_status": ["missing_or_insufficient_native_turnover", "available"],
        "actual_ledger_notional": [1000.0, 200.0], "constant_1m_notional": [500.0, 100.0],
        "constant_1m_participation": [np.nan, 1.0], "actual_participation": [np.nan, 2.0]})
    report_dict = liquidity_period_details(order_df, pricing_df, date_idx[-1])
    assert report_dict["full"]["missing_adv_orders"] == 1
    assert report_dict["last3y"]["missing_adv_orders"] == 0
    endpoint_dict = report_dict["asset_list"][0]["current_endpoint"]
    assert endpoint_dict["native_20session_median_turnover"] == 100
    assert endpoint_dict["last3y_order_weights_at_fixed_1m_max_of_endpoint_ADV"] == 1
    assert endpoint_dict["last3y_actual_compounded_notionals_max_of_endpoint_ADV"] == 2
    pricing_df.loc[date_idx[-1], ("AAA", "Turnover")] = np.nan
    missing_dict = liquidity_period_details(order_df, pricing_df, date_idx[-1])
    assert missing_dict["asset_list"][0]["current_endpoint"]["status"] == "missing_or_insufficient_endpoint_native_turnover"


def test_financing_uses_previous_negative_cash_and_calendar_weekend():
    date_idx = pd.to_datetime(["2025-01-03", "2025-01-06"])
    equity_ser = pd.Series([1_000_000.0, 1_000_000.0], index=date_idx)
    cash_ser = pd.Series([-36_000.0, 900_000.0], index=date_idx)
    transaction_df = pd.DataFrame(columns=["bar", "amount", "price"])
    report_dict, stressed_ser = ledger_stress(equity_ser, cash_ser, transaction_df, 1_000_000, 0, .04)
    assert report_dict["financing_total"] == pytest.approx(12)
    assert stressed_ser.iloc[0] == 1_000_000
    assert stressed_ser.iloc[1] == 999_988


def test_extra_slippage_is_fixed_ledger_dollar_debit_without_resizing():
    date_idx = pd.to_datetime(["2025-01-02", "2025-01-03"])
    equity_ser = pd.Series([990.0, 1000.0], index=date_idx)
    cash_ser = pd.Series([0.0, 0.0], index=date_idx)
    transaction_df = pd.DataFrame({"bar": date_idx, "amount": [400.0, -800.0], "price": [10.0, 10.0]})
    report_dict, stressed_ser = ledger_stress(equity_ser, cash_ser, transaction_df, 1000, 2.5, 0)
    assert report_dict["extra_slippage_total"] == 3
    assert stressed_ser.tolist() == [989, 997]
    assert report_dict["metrics"]["first_return"] == pytest.approx(-.011)


def test_zero_stress_retains_standalone_cagr_anchor():
    date_idx = pd.to_datetime(["2025-01-02", "2025-01-03"])
    equity_ser = pd.Series([990.0, 1010.0], index=date_idx)
    cash_ser = pd.Series([0.0, 0.0], index=date_idx)
    transaction_df = pd.DataFrame(columns=["bar", "amount", "price"])
    anchor_ts = pd.Timestamp("2024-12-31")
    report_dict, _ = ledger_stress(equity_ser, cash_ser, transaction_df, 1000, 0, 0, anchor_ts)
    assert report_dict["metrics"] == equity_metrics(equity_ser, 1000, anchor_ts)


def test_capitalspecial_spy_cannot_be_presented_as_total_return():
    pricing_df = pd.DataFrame({("SPY", "Close"): [100.0, 101.0]})
    pricing_df.attrs["norgate_adjustment_by_symbol_dict"] = {"SPY": "CAPITALSPECIAL"}
    benchmark_dict = select_benchmarks(pricing_df)
    assert benchmark_dict["SPY_TR"]["status"] == "unavailable"


@pytest.mark.parametrize("mapping_dict", [{}, {"$SPX": "$SPX"}])
def test_unproven_or_wrong_spx_index_cannot_be_presented_as_total_return(mapping_dict):
    pricing_df = pd.DataFrame({("$SPX", "Close"): [100.0, 101.0]})
    pricing_df.attrs["benchmark_data_symbol_dict"] = mapping_dict
    assert select_benchmarks(pricing_df)["SPX_TR"]["status"] == "unavailable"


def test_saved_frame_projection_preserves_metadata_and_end_to_end_hashes(tmp_path):
    baseline_path = tmp_path / "baselines" / "synthetic_strategy"
    baseline_path.mkdir(parents=True)
    date_idx = pd.bdate_range("2021-12-01", periods=30)
    results_df = pd.DataFrame({"total_value": np.linspace(990_000.0, 1_010_000.0, 30), "cash": -100.0}, index=date_idx)
    pricing_df = pd.DataFrame({("$SPX", "Close"): np.linspace(100.0, 110.0, 30),
        ("AAA", "Turnover"): 1_000_000.0, ("AAA", "Unadjusted Close"): 101.0,
        ("AAA", "Close"): 50.0, ("SPY", "Close"): 100.0}, index=date_idx)
    pricing_df.attrs["benchmark_data_symbol_dict"] = {"$SPX": "$SPXTR"}
    pricing_df.attrs["norgate_adjustment_by_symbol_dict"] = {"SPY": "CAPITALSPECIAL"}
    results_df.to_parquet(baseline_path / "results.parquet")
    pricing_df.to_parquet(baseline_path / "pricing.parquet")
    transaction_df = pd.DataFrame({"bar": [date_idx[-1]], "asset": ["AAA"], "amount": [100.0], "price": [50.0]})
    transaction_df.to_csv(baseline_path / "transactions.csv.gz", index=False)
    (baseline_path / "summary.json").write_text(json.dumps({"status": "pass", "capital": 1_000_000,
        "deterministic_equity": True, "deterministic_economic_transactions": True, "capital_honored": True,
        "actual_capital_base": 1_000_000, "actual_double_capital_base": 2_000_000, "seconds": 1}), encoding="utf-8")
    (baseline_path / "input_manifest.json").write_text(json.dumps({"supplemental_saved_inputs": [
        {"path": "prepared_hpi.parquet", "sha256": "frozen_hpi_input"}]}), encoding="utf-8")
    with gzip.open(baseline_path / "trace.pkl.gz", "wb") as handle_obj:
        pickle.dump([{"execution_date": date_idx[-1], "decision_date": date_idx[-2], "nav": 1_000_000.0}], handle_obj)
    selected_df = load_selected_prices(baseline_path / "pricing.parquet", {"AAA"})
    assert selected_df.attrs == pricing_df.attrs
    assert ("AAA", "Close") not in selected_df.columns
    result_dict = process_baseline(baseline_path, tmp_path / "statistics")
    assert result_dict["status"] == "pass"
    report_dict = json.loads((tmp_path / "statistics/synthetic_strategy/statistics.json").read_text(encoding="utf-8"))
    assert report_dict["standalone_metrics"]["first_return"] == pytest.approx(-.01)
    assert report_dict["benchmarks"]["SPX_TR"]["status"] == "available"
    assert report_dict["benchmarks"]["SPY_TR"]["status"] == "unavailable"
    assert len(report_dict["input_manifest"]) == 6
    assert report_dict["baseline_input_evidence"]["supplemental_saved_inputs"][0]["sha256"] == "frozen_hpi_input"
    assert all(len(row_dict["sha256"]) == 64 for row_dict in report_dict["input_manifest"])
    assert report_dict["input_source_hashes_unchanged_at_completion"]
    assert (tmp_path / "statistics/synthetic_strategy/daily_chart.csv.gz").exists()


def test_interim_pass_without_final_receipts_is_not_processed(tmp_path):
    baseline_path = tmp_path / "interim_strategy"
    baseline_path.mkdir()
    (baseline_path / "summary.json").write_text(json.dumps({"status": "pass", "capital": 1_000_000}), encoding="utf-8")
    result_dict = process_baseline(baseline_path, tmp_path / "statistics")
    assert result_dict["status"] == "not_processed"
    assert result_dict["reason"] == "baseline_not_final_repeat_and_capital_pass"


def test_qualified_hpi_source_map_preserves_original_results_and_transactions(tmp_path):
    key_str = "strategy_mr_hpi_sp500_2_3_5_vote"
    baseline_path = tmp_path / "baselines" / key_str
    canonical_path = tmp_path / "input_preparation" / key_str
    repaired_path = tmp_path / "mr_hpi235_trace_only"
    for path_obj in (baseline_path, canonical_path, repaired_path):
        path_obj.mkdir(parents=True)
    for name_str in ("summary.json", "results.parquet", "transactions.csv.gz"):
        (baseline_path / name_str).write_text("original", encoding="utf-8")
    pricing_df = pd.DataFrame({("AAA", "Close"): [50.0]}, index=pd.to_datetime(["2026-10-14"]))
    pricing_df.to_parquet(canonical_path / "pricing.parquet")
    shutil.copyfile(canonical_path / "pricing.parquet", repaired_path / "pricing.parquet")
    auxiliary_dict = {"universe_df": pd.DataFrame({"AAA": [1]}, index=pricing_df.index)}
    for path_obj in (canonical_path, repaired_path):
        pd.to_pickle(auxiliary_dict, path_obj / "auxiliary.pkl.gz", compression="gzip")
    (repaired_path / "trace.pkl.gz").write_bytes(b"qualified trace")
    (repaired_path / "results.parquet").write_bytes(b"qualified equity")
    source_row_list = [hash_dict(canonical_path / name_str) for name_str in ("pricing.parquet", "auxiliary.pkl.gz")]
    (baseline_path / "input_manifest.json").write_text(json.dumps({"supplemental_saved_inputs": source_row_list}), encoding="utf-8")
    checker_path = REPO_PATH / "scripts/research/wired_signoff_20261015/mr_hpi_cached_trace.py"
    qualification_dict = {"status_str": "pass", "exact_original_baseline_equity_bool": True,
        "source_unchanged_bool": True, "new_strategy_variants_int": 0,
        "checker_sha256_str": hash_dict(checker_path)["sha256"],
        "canonical_input_sha256_str": hash_dict(canonical_path / "pricing.parquet")["sha256"],
        "source_hash_dict": {str(checker_path.relative_to(REPO_PATH)): hash_dict(checker_path)["sha256"]}}
    (repaired_path / "qualification.json").write_text(json.dumps(qualification_dict), encoding="utf-8")
    source_dict, extra_list, _ = resolve_input_sources(baseline_path)
    assert source_dict["results.parquet"] == baseline_path / "results.parquet"
    assert source_dict["transactions.csv.gz"] == baseline_path / "transactions.csv.gz"
    assert source_dict["pricing.parquet"] == repaired_path / "pricing.parquet"
    assert source_dict["trace.pkl.gz"] == repaired_path / "trace.pkl.gz"
    assert repaired_path / "qualification.json" in extra_list
    qualification_dict["source_hash_dict"]["scripts/research/wired_signoff_20261015/mr_native_replay.py"] = "historical_recorder_hash"
    (repaired_path / "qualification.json").write_text(json.dumps(qualification_dict), encoding="utf-8")
    _, _, observed_dict = resolve_input_sources(baseline_path)
    assert observed_dict["statistics_observed_historical_recorder_mismatch_list"][0]["qualified_historical_sha256"] == "historical_recorder_hash"
    qualification_dict["source_hash_dict"]["strategies/hpi/stateful_long.py"] = "changed_production_hash"
    (repaired_path / "qualification.json").write_text(json.dumps(qualification_dict), encoding="utf-8")
    with pytest.raises(ValueError, match="production source changed"):
        resolve_input_sources(baseline_path)
    qualification_dict["source_hash_dict"].pop("strategies/hpi/stateful_long.py")
    qualification_dict["exact_original_baseline_equity_bool"] = False
    (repaired_path / "qualification.json").write_text(json.dumps(qualification_dict), encoding="utf-8")
    with pytest.raises(ValueError, match="qualification"):
        resolve_input_sources(baseline_path)


def test_repaired_trace_requires_original_equity_cash_and_prefill_boundary():
    date_idx = pd.to_datetime(["2025-01-02", "2025-01-03"])
    original_df = pd.DataFrame({"total_value": [990_000.0, 1_010_000.0], "cash": [400_000.0, 500_000.0]}, index=date_idx)
    trace_dict = {pd.Timestamp("2024-12-31"): {"execution_date": date_idx[0], "nav_float": 1_000_000.0, "cash_float": 1_000_000.0},
        date_idx[0]: {"execution_date": date_idx[1], "nav_float": 990_000.0, "cash_float": 400_000.0}}
    trace_list = normalize_trace(trace_dict)
    report_dict = verify_repaired_hpi_trace(trace_list, {"trace_session_count_int": 2}, original_df, original_df.copy(), 1_000_000)
    assert report_dict["exact_original_equity_and_cash"]
    assert report_dict["original_close_nav_cash_boundary_checked_sessions"] == 2
    trace_list[-1]["cash"] = 500_000.0
    with pytest.raises(ValueError, match="NAV/cash"):
        verify_repaired_hpi_trace(trace_list, {"trace_session_count_int": 2}, original_df, original_df.copy(), 1_000_000)
    wrong_cash_df = original_df.copy()
    wrong_cash_df.iloc[0, 1] += 1
    with pytest.raises(AssertionError):
        verify_repaired_hpi_trace(normalize_trace(trace_dict), {"trace_session_count_int": 2}, original_df, wrong_cash_df, 1_000_000)


def test_input_change_cannot_leave_a_passing_frozen_evidence_claim(tmp_path):
    input_path = tmp_path / "saved_ledger.csv"
    input_path.write_text("date,equity\n2025-01-02,100\n", encoding="utf-8")
    manifest_list = [hash_dict(input_path)]
    verify_input_manifest(manifest_list)
    input_path.write_text("date,equity\n2025-01-02,999\n", encoding="utf-8")
    with pytest.raises(ValueError, match="changed during statistics"):
        verify_input_manifest(manifest_list)


def test_incremental_statistics_index_preserves_other_completed_families(tmp_path):
    index_path = tmp_path / "index.json"
    update_index(index_path, [{"key": "taa", "status": "pass"}])
    update_index(index_path, [{"key": "ndx", "status": "pass"}])
    update_index(index_path, [{"key": "ndx", "status": "error", "error": "changed input"}])
    row_list = json.loads(index_path.read_text(encoding="utf-8"))
    assert row_list == [{"key": "ndx", "status": "error", "error": "changed input"}, {"key": "taa", "status": "pass"}]
