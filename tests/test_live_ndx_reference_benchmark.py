"""The NDX reference borrows only its reporting benchmark from the ETF snapshot."""
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from dataclasses import replace
from datetime import UTC, datetime

import pandas as pd
import pytest

from alpha.live import reference_compare
from data import norgate_snapshot_store as snapshot_module
from data.norgate_loader import use_norgate_data_profile
from strategies.momentum import strategy_mo_atr_normalized_ndx as ndx_module
from strategies.momentum import strategy_mo_atr_normalized_ndx_vxn_scaled as vxn_module
from test_live_reference_compare import AutoReferenceAccountingTestStrategy, _build_release

NDX_PROFILE_STR = "norgate_eod_ndx_pit_plus_vxn_helper"
ETF_PROFILE_STR = "norgate_eod_etf_plus_vix_helper"
NDX_IMPORT_STR = "strategies.momentum.strategy_mo_atr_normalized_ndx_vxn_scaled:VxnScaledAtrNormalizedNdxStrategy"


def write_profile(tmp_path, profile_str, date_list, symbol_str, adjustment_str, base_price_float=100.0):
    price_df = pd.DataFrame([
        {"date": date_str, "symbol_str": symbol_str, "adjustment_str": adjustment_str,
         "Open": base_price_float + index_int, "High": base_price_float + 2.0 + index_int, "Low": base_price_float - 1.0 + index_int,
         "Close": base_price_float + 1.0 + index_int, "Dividend": 0.0}
        for index_int, date_str in enumerate(date_list)
    ])
    return snapshot_module.write_snapshot_files(
        snapshot_root_str=str(tmp_path), profile_str=profile_str, snapshot_date_str=date_list[-1],
        price_df=price_df, required_symbol_list=[symbol_str],
        adjustment_mode_map_dict={symbol_str: adjustment_str},
        generated_timestamp_ts=datetime(2024, 1, 4, 22, tzinfo=UTC),
        schema_version_int=snapshot_module.SNAPSHOT_SCHEMA_VERSION_INT,
    )


@pytest.fixture
def separate_snapshots(tmp_path, monkeypatch):
    monkeypatch.setenv("NORGATE_SNAPSHOT_ROOT", str(tmp_path))
    monkeypatch.setenv("ALPHA_USE_NORGATE_SNAPSHOT_BOOL", "true")
    write_profile(tmp_path, NDX_PROFILE_STR, ["2024-01-02", "2024-01-03"], "AAA", "CAPITALSPECIAL")
    write_profile(tmp_path, ETF_PROFILE_STR, ["2024-01-02", "2024-01-03", "2024-01-04"], "$SPXTR", "TOTALRETURN")
    return tmp_path


def stock_prices():
    return pd.DataFrame({("AAA", "Close"): [101.0, 102.0]}, index=pd.to_datetime(["2024-01-02", "2024-01-03"]))


def append_benchmark():
    return ndx_module.append_total_return_benchmark_data_df(
        stock_prices(), replace(ndx_module.DEFAULT_CONFIG, history_start_date_str="2024-01-02", backtest_start_date_str="2024-01-03", end_date_str="2024-01-03"),
    )


def run_reference(tmp_path):
    release_obj = replace(_build_release(strategy_import_str=NDX_IMPORT_STR), data_profile_str=NDX_PROFILE_STR)
    return reference_compare.run_auto_reference_strategy(
        release_obj=release_obj, deployment_start_date_str="2024-01-02", reference_end_date_str="2024-01-03",
        deployment_initial_cash_float=10000.0, output_dir_path_obj=tmp_path / "reference",
    )


def test_separate_benchmark_preserves_stocks_calendar_and_live_snapshot(separate_snapshots):
    before_dict = {file_path: file_path.read_bytes() for file_path in separate_snapshots.rglob('*') if file_path.is_file()}
    with use_norgate_data_profile(NDX_PROFILE_STR), ndx_module.use_reference_benchmark_profile(ETF_PROFILE_STR):
        combined_df = append_benchmark()
        assert snapshot_module.get_active_data_profile_str() == NDX_PROFILE_STR
        pd.testing.assert_frame_equal(combined_df.loc[:, [("AAA", "Close")]], stock_prices())
        assert combined_df[("$SPXTR", "Close")].tolist() == [101.0, 102.0]
        assert len(combined_df) == 2  # A later benchmark bar cannot extend the execution calendar.
    assert before_dict == {file_path: file_path.read_bytes() for file_path in separate_snapshots.rglob('*') if file_path.is_file()}
    with use_norgate_data_profile(NDX_PROFILE_STR), pytest.raises(snapshot_module.NorgateSnapshotValidationError):
        append_benchmark()  # Ordinary LIVE/research profile reads retain the original contract.


def test_benchmark_context_is_nested_and_thread_local(separate_snapshots):
    with use_norgate_data_profile(NDX_PROFILE_STR), ndx_module.use_reference_benchmark_profile(ETF_PROFILE_STR):
        assert append_benchmark()[("$SPXTR", "Close")].tolist() == [101.0, 102.0]
        with ndx_module.use_reference_benchmark_profile(NDX_PROFILE_STR), pytest.raises(snapshot_module.NorgateSnapshotValidationError):
            append_benchmark()
        assert append_benchmark()[("$SPXTR", "Close")].tolist() == [101.0, 102.0]
        with ThreadPoolExecutor(max_workers=1) as executor_obj:
            assert executor_obj.submit(ndx_module._REFERENCE_BENCHMARK_PROFILE_CONTEXT_VAR.get).result() is None


@pytest.mark.parametrize("failure_str", ["missing_symbol", "missing_date"])
def test_missing_benchmark_evidence_fails_without_substitution(tmp_path, monkeypatch, failure_str):
    monkeypatch.setenv("NORGATE_SNAPSHOT_ROOT", str(tmp_path))
    monkeypatch.setenv("ALPHA_USE_NORGATE_SNAPSHOT_BOOL", "true")
    write_profile(tmp_path, NDX_PROFILE_STR, ["2024-01-02", "2024-01-03"], "AAA", "CAPITALSPECIAL")
    write_profile(tmp_path, ETF_PROFILE_STR, ["2024-01-02"], "SPY" if failure_str == "missing_symbol" else "$SPXTR", "TOTALRETURN")
    with use_norgate_data_profile(NDX_PROFILE_STR), ndx_module.use_reference_benchmark_profile(ETF_PROFILE_STR):
        with pytest.raises((snapshot_module.NorgateSnapshotValidationError, RuntimeError), match="missing"):
            append_benchmark()
        assert snapshot_module.get_active_data_profile_str() == NDX_PROFILE_STR


def test_reference_scopes_benchmark_and_records_validated_source(separate_snapshots, monkeypatch):
    def run_variant(backtest_start_date_str, capital_base_float, end_date_str, **kwargs_dict):
        assert snapshot_module.get_active_data_profile_str() == NDX_PROFILE_STR
        assert append_benchmark()[("$SPXTR", "Close")].tolist() == [101.0, 102.0]
        assert snapshot_module.get_active_data_profile_str() == NDX_PROFILE_STR
        return AutoReferenceAccountingTestStrategy(name="reference", benchmarks=[])
    monkeypatch.setattr(vxn_module, "run_variant", run_variant)
    strategy_obj = run_reference(separate_snapshots)
    source_dict = strategy_obj.reference_benchmark_source_dict
    assert source_dict['symbol_str'] == '$SPXTR'
    assert source_dict['adjustment_str'] == 'TOTALRETURN'
    assert source_dict['data_profile_str'] == ETF_PROFILE_STR
    assert source_dict['manifest_hash_str'] == snapshot_module.load_valid_snapshot_manifest(ETF_PROFILE_STR).manifest_hash_str
    assert reference_compare.build_reference_maps_dict(strategy_obj)['reference_benchmark_source_dict'] == source_dict
    assert ndx_module._REFERENCE_BENCHMARK_PROFILE_CONTEXT_VAR.get() is None
    assert snapshot_module.get_active_data_profile_str() is None


def test_reference_rejects_stale_benchmark_before_running(tmp_path, monkeypatch):
    monkeypatch.setenv("NORGATE_SNAPSHOT_ROOT", str(tmp_path))
    monkeypatch.setenv("ALPHA_USE_NORGATE_SNAPSHOT_BOOL", "true")
    write_profile(tmp_path, ETF_PROFILE_STR, ["2024-01-02"], "$SPXTR", "TOTALRETURN")
    monkeypatch.setattr(vxn_module, "run_variant", lambda backtest_start_date_str, capital_base_float, end_date_str: pytest.fail('must not run'))
    with pytest.raises(snapshot_module.NorgateSnapshotValidationError, match="stale"):
        run_reference(tmp_path)


def test_failed_reference_restores_profiles(separate_snapshots, monkeypatch):
    def run_variant(backtest_start_date_str, capital_base_float, end_date_str, **kwargs_dict):
        append_benchmark()
        raise RuntimeError('reference failed')
    monkeypatch.setattr(vxn_module, "run_variant", run_variant)
    with pytest.raises(RuntimeError, match='reference failed'):
        run_reference(separate_snapshots)
    assert ndx_module._REFERENCE_BENCHMARK_PROFILE_CONTEXT_VAR.get() is None
    assert snapshot_module.get_active_data_profile_str() is None


def test_reference_keeps_frozen_benchmark_when_new_snapshot_is_published(separate_snapshots, monkeypatch):
    original_hash_str = snapshot_module.load_valid_snapshot_manifest(ETF_PROFILE_STR).manifest_hash_str
    def run_variant(backtest_start_date_str, capital_base_float, end_date_str, **kwargs_dict):
        append_benchmark()
        write_profile(separate_snapshots, ETF_PROFILE_STR, ['2024-01-05'], '$SPXTR', 'TOTALRETURN')
        return AutoReferenceAccountingTestStrategy(name='reference', benchmarks=[])
    monkeypatch.setattr(vxn_module, "run_variant", run_variant)
    strategy_obj = run_reference(separate_snapshots)
    assert strategy_obj.reference_benchmark_source_dict['manifest_hash_str'] == original_hash_str

def test_overlapping_benchmark_loads_restore_each_callers_profiles(separate_snapshots):
    write_profile(separate_snapshots, 'other_benchmark', ['2024-01-02', '2024-01-03'], '$SPXTR', 'TOTALRETURN', base_price_float=500.0)
    barrier_obj = Barrier(2)
    def load_in_thread(profile_str):
        with use_norgate_data_profile(NDX_PROFILE_STR):
            with ndx_module.use_reference_benchmark_profile(profile_str):
                barrier_obj.wait(timeout=10)
                close_list = append_benchmark()[('$SPXTR', 'Close')].tolist()
                assert snapshot_module.get_active_data_profile_str() == NDX_PROFILE_STR
            assert ndx_module._REFERENCE_BENCHMARK_PROFILE_CONTEXT_VAR.get() is None
        assert snapshot_module.get_active_data_profile_str() is None
        return close_list
    with ThreadPoolExecutor(max_workers=2) as executor_obj:
        future_list = [executor_obj.submit(load_in_thread, profile_str) for profile_str in (ETF_PROFILE_STR, 'other_benchmark')]
        assert [future_obj.result() for future_obj in future_list] == [[101.0, 102.0], [501.0, 502.0]]


def test_real_ndx_loaders_keep_stock_universe_vxn_and_calendar_unchanged(tmp_path, monkeypatch):
    import exchange_calendars
    monkeypatch.setenv('ALPHA_USE_NORGATE_SNAPSHOT_BOOL', 'true')
    monkeypatch.setenv('NORGATE_SNAPSHOT_ROOT', str(tmp_path))
    date_idx = exchange_calendars.get_calendar('XNYS').sessions_in_range('2022-01-03', '2024-01-04').tz_localize(None)
    write_profile(tmp_path, ETF_PROFILE_STR, list(date_idx.strftime('%Y-%m-%d')), '$SPXTR', 'TOTALRETURN')
    pricing_df = pd.DataFrame({
        (symbol_str, field_str): [100.0 + index_int / 10 for index_int in range(len(date_idx))]
        for symbol_str in ('AAA', 'SPY') for field_str in ('Open', 'High', 'Low', 'Close', 'Unadjusted Close')
    }, index=date_idx)
    universe_df = pd.DataFrame({'AAA': 1}, index=date_idx)
    original_load_fn = ndx_module.load_raw_prices
    call_list = []
    def universe_fn(indexname):
        call_list.append(('universe', snapshot_module.get_active_data_profile_str()))
        return ['AAA'], universe_df.copy()
    def stock_fn(symbols, benchmarks, start_date, end_date):
        if benchmarks:
            call_list.append(('benchmark', snapshot_module.get_active_data_profile_str()))
            return original_load_fn(symbols, benchmarks, start_date, end_date)
        call_list.append(('stocks', snapshot_module.get_active_data_profile_str()))
        assert symbols == ['AAA', 'SPY']
        return pricing_df.copy()
    def vxn_fn(**kwargs_dict):
        call_list.append(('vxn', snapshot_module.get_active_data_profile_str()))
        return pd.Series(22.0, index=date_idx)
    monkeypatch.setattr(ndx_module, 'build_index_constituent_matrix', universe_fn)
    monkeypatch.setattr(ndx_module, 'load_raw_prices', stock_fn)
    monkeypatch.setattr(vxn_module, 'load_vxn_close_ser', vxn_fn)
    config_obj = replace(vxn_module.DEFAULT_CONFIG, history_start_date_str='2022-01-03', backtest_start_date_str='2023-02-01', end_date_str='2024-01-04')
    with use_norgate_data_profile(NDX_PROFILE_STR):
        baseline_tuple = vxn_module.get_vxn_scaled_atr_normalized_ndx_data(config_obj)
        call_list.clear()
        with ndx_module.use_reference_benchmark_profile(ETF_PROFILE_STR):
            benchmark_tuple = vxn_module.get_vxn_scaled_atr_normalized_ndx_data(config_obj, include_total_return_benchmark_bool=True)
    assert call_list == [('universe', NDX_PROFILE_STR), ('stocks', NDX_PROFILE_STR), ('benchmark', ETF_PROFILE_STR), ('vxn', NDX_PROFILE_STR)]
    pd.testing.assert_frame_equal(benchmark_tuple[0].loc[:, pricing_df.columns], baseline_tuple[0])
    for baseline_df, benchmark_df in zip(baseline_tuple[1:], benchmark_tuple[1:]):
        pd.testing.assert_frame_equal(baseline_df, benchmark_df)


def test_direct_ndx_reference_does_not_require_etf_snapshot(tmp_path, monkeypatch):
    monkeypatch.setenv('ALPHA_USE_NORGATE_SNAPSHOT_BOOL', 'false')
    def run_variant(backtest_start_date_str, capital_base_float, end_date_str, **kwargs_dict):
        assert ndx_module._REFERENCE_BENCHMARK_PROFILE_CONTEXT_VAR.get() is None
        return AutoReferenceAccountingTestStrategy(name='reference', benchmarks=[])
    monkeypatch.setattr(vxn_module, 'run_variant', run_variant)
    assert not hasattr(run_reference(tmp_path), 'reference_benchmark_source_dict')

def test_comparison_summary_preserves_separate_benchmark_lineage(tmp_path):
    strategy_obj = AutoReferenceAccountingTestStrategy(name='reference', benchmarks=[])
    source_dict = {'symbol_str': '$SPXTR', 'adjustment_str': 'TOTALRETURN', 'data_profile_str': ETF_PROFILE_STR,
                   'snapshot_date_str': '2024-01-03', 'manifest_hash_str': 'benchmark_hash'}
    strategy_obj.reference_benchmark_source_dict = source_dict
    maps_dict = reference_compare.build_reference_maps_dict(strategy_obj)
    artifact_dict = reference_compare.write_reference_compare_artifacts(
        output_dir_path_obj=tmp_path / 'artifacts',
        release_obj=replace(_build_release(strategy_import_str=NDX_IMPORT_STR), data_profile_str=NDX_PROFILE_STR),
        compare_report_dict={'target_session_date_str': '2024-01-03'}, reference_maps_dict=maps_dict,
        live_history_row_dict_list=[], reference_strategy_pickle_path_str=None,
    )
    assert json.loads(Path(artifact_dict['summary_json_path_str']).read_text())['reference_benchmark_source_dict'] == source_dict
