"""Checks for audit evidence calculations, independent of production trading."""
import numpy as np
import pandas as pd
import pytest

from audit_runner import AuditCacheMiss, build_saved_hpi_loader, calculate_metrics, frame_hash_str, save_pickle, write_json


def test_first_session_loss_is_in_drawdown():
    equity_ser = pd.Series([90.0, 99.0], index=pd.to_datetime(["2025-01-02", "2025-01-03"]))
    metric_dict = calculate_metrics(equity_ser, 100.0)
    assert metric_dict["max_drawdown"] == pytest.approx(-0.1)
    assert metric_dict["terminal_equity"] == 99.0


@pytest.mark.parametrize("value_list", [[100.0, np.nan], [100.0, 0.0], [100.0, -10.0]])
def test_invalid_equity_cannot_be_reported_as_valid(value_list):
    equity_ser = pd.Series(value_list, index=pd.to_datetime(["2025-01-02", "2025-01-03"]))
    with pytest.raises(ValueError):
        calculate_metrics(equity_ser)


def test_hash_includes_empty_frame_schema_and_all_column_names():
    empty_df = pd.DataFrame(columns=["Open", "Close"])
    assert frame_hash_str(empty_df) != frame_hash_str(empty_df.rename(columns={"Close": "Other"}))


def test_reversed_or_duplicate_dates_are_rejected():
    for date_list in (["2025-01-03", "2025-01-02"], ["2025-01-02", "2025-01-02"]):
        equity_ser = pd.Series([100.0, 101.0], index=pd.to_datetime(date_list))
        with pytest.raises(ValueError):
            calculate_metrics(equity_ser)


def test_offline_missing_classification_cannot_be_swallowed_as_unknown(monkeypatch):
    import sys
    from types import SimpleNamespace
    from strategies.momentum.strategy_mo_atr_normalized_sector_cap import build_current_gics_sector_map

    def missing_cache_fn(*argument_tuple):
        raise AuditCacheMiss("missing frozen classification")

    monkeypatch.setitem(sys.modules, "norgatedata", SimpleNamespace(classification_at_level=missing_cache_fn))
    with pytest.raises(AuditCacheMiss):
        build_current_gics_sector_map(["AAA"])


def test_saved_hpi_inputs_are_exact_private_and_fail_closed(tmp_path):
    prepared_path = tmp_path / "input_preparation/strategy_mr_hpi_sp500_2_3_5_vote"
    prepared_path.mkdir(parents=True)
    date_idx = pd.to_datetime(["2026-10-13", "2026-10-14"])
    pricing_df = pd.DataFrame({("AAA", "Close"): [50., 51.]}, index=date_idx)
    pricing_df.attrs["norgate_adjustment_by_symbol_dict"] = {"AAA": "CAPITALSPECIAL"}
    pricing_df.to_parquet(prepared_path / "pricing.parquet")
    universe_df = pd.DataFrame({"AAA": [1, 0]}, index=date_idx)
    save_pickle(prepared_path / "auxiliary.pkl.gz", {"universe_df": universe_df})
    write_json(prepared_path / "summary.json", {"status": "inputs_prepared_not_run", "end_requested": "2026-10-14"})
    evidence_list = []
    loader_fn = build_saved_hpi_loader(tmp_path, "2026-10-14", evidence_list)
    request_dict = dict(indexname_str="S&P 500", benchmark_symbol_str="$SPXTR", start_date_str="1998-01-01", end_date_str="2026-10-14")
    symbol_list, first_universe_df, first_pricing_df = loader_fn(**request_dict)
    assert symbol_list == ["AAA"] and len(evidence_list) == 2
    first_universe_df.iloc[-1, 0] = 1
    first_pricing_df.iloc[-1, 0] = 0
    _, second_universe_df, second_pricing_df = loader_fn(**request_dict)
    pd.testing.assert_frame_equal(universe_df, second_universe_df)
    pd.testing.assert_frame_equal(pricing_df, second_pricing_df)
    assert second_pricing_df.attrs == pricing_df.attrs
    with pytest.raises(AuditCacheMiss):
        loader_fn(**{**request_dict, "end_date_str": "2026-10-15"})
    pricing_df.to_parquet(prepared_path / "pricing.parquet", compression="gzip")
    with pytest.raises(AuditCacheMiss):
        loader_fn(**request_dict)
