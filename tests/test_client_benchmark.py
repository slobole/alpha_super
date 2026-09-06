from dataclasses import replace
from datetime import UTC, datetime
import hashlib
import json

import pandas as pd
import pytest

from alpha.live.client_benchmark import BenchmarkSnapshot, account_benchmark_dict, load_benchmark_snapshot, validate_benchmark_config
from alpha.live.client_reporting import build_client_report_dict
from alpha.live.dashboard_v3.demo import build_demo_benchmark_snapshot, build_demo_fixture_tuple
from data.norgate_snapshot_store import write_snapshot_files


def strategy_dict(from_str="2026-09-01", to_str="2026-09-02"):
    return {"from_date_str": from_str, "to_date_str": to_str, "coverage_complete_bool": True, "twr_float": .05}


def benchmark_obj():
    return BenchmarkSnapshot("SPY", (("2026-08-31", 100), ("2026-09-01", 102), ("2026-09-02", 110)), "a" * 64, "b" * 64, "2026-09-02", "test_profile")


def test_exact_interval_uses_previous_close_and_pp_not_dollars():
    result_dict = account_benchmark_dict(strategy_dict(), benchmark_obj())
    assert result_dict["status_str"] == "ready"
    assert result_dict["baseline_date_str"] == "2026-08-31"
    assert result_dict["return_float"] == pytest.approx(.10)
    assert result_dict["difference_pp_float"] == pytest.approx(-5)
    assert "pnl_float" not in result_dict


@pytest.mark.parametrize("close_tuple", [(("2026-09-01", 102), ("2026-09-02", 110)), (("2026-08-31", 100), ("2026-09-02", 110)), (("2026-08-31", 100), ("2026-09-01", None), ("2026-09-02", 110)), (("2026-08-31", 100), ("2026-09-01", 102), ("2026-09-02", float("nan")))])
def test_missing_baseline_or_middle_session_never_shortens_or_fills(close_tuple):
    result_dict = account_benchmark_dict(strategy_dict(), replace(benchmark_obj(), close_tuple=close_tuple))
    assert result_dict["status_str"] == "unavailable"
    assert result_dict["return_float"] is result_dict["difference_pp_float"] is None


def test_weekend_holiday_account_returns_are_preserved():
    # Saturday through Labor Day: the market benchmark is unchanged at Friday's
    # close, while a real account fee can still make its official return -1%.
    account_dict = dict(strategy_dict("2026-09-05", "2026-09-07"), twr_float=-.01)
    source_obj = replace(benchmark_obj(), close_tuple=(("2026-09-04", 100),), snapshot_date_str="2026-09-04")
    result_dict = account_benchmark_dict(account_dict, source_obj)
    assert result_dict["status_str"] == "ready"
    assert result_dict["baseline_date_str"] == result_dict["end_price_date_str"] == "2026-09-04"
    assert result_dict["return_float"] == 0
    assert result_dict["difference_pp_float"] == -1


def test_extreme_finite_prices_never_poison_account_report_with_infinity():
    source_obj = replace(benchmark_obj(), close_tuple=(("2026-08-31", 1e-300), ("2026-09-01", 1), ("2026-09-02", 1e300)))
    result_dict = account_benchmark_dict(strategy_dict(), source_obj)
    assert result_dict["status_str"] == "unavailable"
    assert result_dict["return_float"] is result_dict["difference_pp_float"] is None
    json.dumps(result_dict, allow_nan=False)


def test_retired_account_compares_its_own_interval_and_incomplete_account_withholds():
    result_dict = account_benchmark_dict(strategy_dict("2026-09-02", "2026-09-02"), benchmark_obj())
    assert result_dict["baseline_date_str"] == "2026-09-01"
    assert result_dict["return_float"] == pytest.approx(110 / 102 - 1)
    result_dict = account_benchmark_dict(dict(strategy_dict(), coverage_complete_bool=False), benchmark_obj())
    assert result_dict["status_str"] == "unavailable"


def test_future_dated_snapshot_cannot_support_report():
    result_dict = account_benchmark_dict(strategy_dict(), replace(benchmark_obj(), snapshot_date_str="2026-10-01"), as_of_date_str="2026-09-05")
    assert result_dict["status_str"] == "unavailable"
    assert result_dict["return_float"] is None


def write_source_dict(tmp_path, *, adjustment_str="TOTALRETURN", symbol_str="SPY"):
    price_df = pd.DataFrame([{"date": date_str, "symbol_str": symbol_str, "adjustment_str": adjustment_str, "Close": price_float} for date_str, price_float in benchmark_obj().close_tuple])
    directory_obj = write_snapshot_files(snapshot_root_str=str(tmp_path), profile_str="test_profile", snapshot_date_str="2026-09-02", price_df=price_df)
    return {"symbol": symbol_str, "snapshot_directory": str(directory_obj)}


def test_reads_pinned_manifest_and_prices_without_alternate_source(tmp_path):
    config_dict = write_source_dict(tmp_path)
    source_obj = load_benchmark_snapshot(config_dict)
    assert source_obj.unavailable_reason_str is None
    assert source_obj.price_hash_str == hashlib.sha256((tmp_path / "test_profile/2026-09-02/prices.parquet").read_bytes()).hexdigest()
    assert account_benchmark_dict(strategy_dict(), source_obj)["return_float"] == pytest.approx(.1)
    # An unrelated newer snapshot must not alter the explicitly selected source.
    write_snapshot_files(snapshot_root_str=str(tmp_path), profile_str="test_profile", snapshot_date_str="2026-09-03", price_df=pd.DataFrame([{"date": "2026-09-03", "symbol_str": "SPY", "adjustment_str": "TOTALRETURN", "Close": 500}]))
    assert load_benchmark_snapshot(config_dict) == source_obj


def test_capital_price_cannot_substitute_for_total_return(tmp_path):
    assert load_benchmark_snapshot(write_source_dict(tmp_path, adjustment_str="CAPITALSPECIAL")).unavailable_reason_str


@pytest.mark.parametrize("snapshot_date_str", ["20260902", "2026-09-01"])
def test_manifest_date_must_be_canonical_and_match_directory(tmp_path, snapshot_date_str):
    config_dict = write_source_dict(tmp_path)
    manifest_path_obj = tmp_path / "test_profile/2026-09-02/manifest.json"
    manifest_dict = json.loads(manifest_path_obj.read_text())
    manifest_dict["snapshot_market_session_date_str"] = snapshot_date_str
    manifest_path_obj.write_text(json.dumps(manifest_dict))
    assert load_benchmark_snapshot(config_dict).unavailable_reason_str


def test_spxtr_is_named_as_index_not_spy(tmp_path):
    result_dict = account_benchmark_dict(strategy_dict(), load_benchmark_snapshot(write_source_dict(tmp_path, symbol_str="$SPXTR")))
    assert result_dict["label_str"] == "S&P 500 total return"
    assert result_dict["status_str"] == "ready"


def test_wrong_hash_and_same_date_replacement_change_evidence(tmp_path):
    config_dict = write_source_dict(tmp_path)
    source_obj = load_benchmark_snapshot(config_dict)
    price_path_obj = tmp_path / "test_profile/2026-09-02/prices.parquet"
    price_df = pd.read_parquet(price_path_obj)
    price_df.loc[price_df["date"] == "2026-09-02", "Close"] = 120
    price_df.to_parquet(price_path_obj, index=False)
    assert load_benchmark_snapshot(config_dict).unavailable_reason_str
    manifest_path_obj = price_path_obj.with_name("manifest.json")
    manifest_dict = json.loads(manifest_path_obj.read_text())
    manifest_dict["files"]["prices.parquet"]["sha256"] = hashlib.sha256(price_path_obj.read_bytes()).hexdigest()
    manifest_path_obj.write_text(json.dumps(manifest_dict))
    revised_obj = load_benchmark_snapshot(config_dict)
    assert revised_obj.unavailable_reason_str is None
    assert revised_obj.price_hash_str != source_obj.price_hash_str
    assert revised_obj.manifest_hash_str != source_obj.manifest_hash_str


@pytest.mark.parametrize("config_dict", [{}, {"symbol": [], "snapshot_directory": "x"}, {"symbol": "QQQ", "snapshot_directory": "x"}, {"symbol": "SPY", "snapshot_directory": ""}])
def test_invalid_selection_is_rejected(config_dict):
    with pytest.raises(ValueError):
        validate_benchmark_config(config_dict)


def test_benchmark_shared_result_hash_changes_but_account_money_does_not():
    registry_dict, source_dict = build_demo_fixture_tuple()
    client_dict = registry_dict["clients"][0]
    argument_dict = dict(from_date_str="2026-06-01", to_date_str="2026-09-04", as_of_ts=datetime(2026, 9, 5, 12, tzinfo=UTC))
    plain_dict = build_client_report_dict(client_dict, source_dict["demo-owner"], **argument_dict)
    compared_dict = build_client_report_dict(client_dict, source_dict["demo-owner"], benchmark_snapshot_obj=build_demo_benchmark_snapshot(), **argument_dict)
    assert plain_dict["report_hash_str"] != compared_dict["report_hash_str"]
    assert plain_dict["pnl_float"] == compared_dict["pnl_float"]
    assert plain_dict["twr_float"] is not None
    assert plain_dict["twr_float"] == compared_dict["twr_float"]
    assert plain_dict["return_path_list"] == compared_dict["return_path_list"]
    assert all(row_dict["benchmark_dict"]["status_str"] == "ready" for row_dict in compared_dict["strategy_list"])
