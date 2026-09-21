"""Synthetic local sources only; no installed NDU, broker or account access."""

from copy import deepcopy
from datetime import UTC, datetime
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import requests

from alpha.live.dashboard_v4 import pod_close_prices
from data import norgate_snapshot_store


PROFILE_STR = "norgate_eod_etf_plus_vix_helper"
CLOSE_STR = "2026-08-07"
AS_OF_TS = datetime(2026, 8, 10, 12, tzinfo=UTC)


@pytest.fixture(autouse=True)
def isolated_sources(monkeypatch, tmp_path):
    monkeypatch.setenv("ALPHA_USE_NORGATE_SNAPSHOT_BOOL", "true")
    monkeypatch.setenv("NORGATE_SNAPSHOT_ROOT", str(tmp_path))
    pod_close_prices._cache_dict.clear()
    norgate_snapshot_store.clear_snapshot_manifest_cache()
    monkeypatch.setattr(pod_close_prices, "_local_response_tuple", lambda *args, **kwargs: pytest.fail("Unexpected local NDU call"))
    yield
    pod_close_prices._cache_dict.clear()
    norgate_snapshot_store.clear_snapshot_manifest_cache()


def _snapshot_path(tmp_path, *, close_str=CLOSE_STR, override_dict=None, contract_dict=None, rows_list=None, universe_df=None):
    row_dict = {"date": pd.Timestamp(close_str), "symbol_str": "SPY", "adjustment_str": "CAPITALSPECIAL",
        "Open": 90.0, "High": 110.0, "Low": 80.0, "Close": 25.0, "Unadjusted Close": 100.0,
        "Volume": 1000.0, "Dividend": 0.0}
    row_dict.update(override_dict or {})
    return norgate_snapshot_store.write_snapshot_files(snapshot_root_str=str(tmp_path), profile_str=PROFILE_STR,
        snapshot_date_str=close_str, price_df=pd.DataFrame(rows_list if rows_list is not None else [row_dict]),
        required_symbol_list=["SPY"], adjustment_mode_map_dict={"SPY": "CAPITALSPECIAL"},
        data_contract_dict=contract_dict or {}, universe_df=universe_df, generated_timestamp_ts=datetime(2026, 8, 7, 22, tzinfo=UTC))


def _read_dict(symbol_list=None, **override_dict):
    parameter_dict = {"profile_str": PROFILE_STR, "close_date_str": CLOSE_STR, "as_of_ts": AS_OF_TS}
    parameter_dict.update(override_dict)
    return pod_close_prices.load_close_prices_dict(["SPY"] if symbol_list is None else symbol_list, **parameter_dict)


def _refresh_hash(snapshot_path):
    manifest_path = snapshot_path / "manifest.json"
    manifest_dict = json.loads(manifest_path.read_text())
    manifest_dict["files"]["prices.parquet"]["sha256"] = hashlib.sha256((snapshot_path / "prices.parquet").read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest_dict))


def test_snapshot_uses_raw_exact_close_and_changes_neither_bytes_nor_timestamps(tmp_path):
    snapshot_path = _snapshot_path(tmp_path)
    source_path_list = [snapshot_path / "manifest.json", snapshot_path / "prices.parquet"]
    before_list = [(source_path.read_bytes(), source_path.stat().st_mtime_ns) for source_path in source_path_list]
    result_dict = _read_dict()
    assert result_dict == {"available_bool": True, "reason_str": "", "price_map_dict": {"SPY": 100},
        "source_str": "Norgate snapshot", "close_date_str": CLOSE_STR}
    result_dict["price_map_dict"]["SPY"] = -999
    assert _read_dict()["price_map_dict"] == {"SPY": 100}
    assert [(source_path.read_bytes(), source_path.stat().st_mtime_ns) for source_path in source_path_list] == before_list


@pytest.mark.parametrize("field_str,value_obj", [("Unadjusted Close", None), ("Unadjusted Close", 0),
    ("Unadjusted Close", -10), ("Unadjusted Close", float("inf")), ("Volume", 0), ("Volume", None),
    ("Volume", float("nan")), ("Volume", -1), ("date", pd.Timestamp("2026-08-06")),
    ("adjustment_str", "TOTALRETURN"), ("symbol_str", "OTHER")])
def test_missing_invalid_or_padded_close_never_falls_back_to_adjusted_values(tmp_path, field_str, value_obj):
    _snapshot_path(tmp_path, override_dict={field_str: value_obj})
    result_dict = _read_dict()
    assert not result_dict["available_bool"] and result_dict["price_map_dict"] == {}


@pytest.mark.parametrize("proof_dict", [{"price_padding_setting_str": "NONE"},
    {"observed_endpoint_date_by_symbol_dict": {"SPY": CLOSE_STR}}])
def test_manifest_unpadded_observation_allows_close_without_positive_volume(tmp_path, proof_dict):
    _snapshot_path(tmp_path, override_dict={"Volume": 0}, contract_dict=proof_dict)
    assert _read_dict()["price_map_dict"] == {"SPY": 100}


def test_wrong_manifest_endpoint_does_not_prove_observation(tmp_path):
    _snapshot_path(tmp_path, override_dict={"Volume": 0}, contract_dict={"observed_endpoint_date_by_symbol_dict": {"SPY": "2026-08-06"}})
    assert not _read_dict()["available_bool"]


def test_missing_exact_artifact_uses_local_prices_not_another_snapshot(tmp_path, monkeypatch):
    call_list = _local_fixture(monkeypatch)
    monkeypatch.setenv("ALPHA_USE_NORGATE_SNAPSHOT_BOOL", "true")
    _snapshot_path(tmp_path, close_str="2026-08-10", override_dict={"date": pd.Timestamp(CLOSE_STR)})
    result_dict = _read_dict()
    assert result_dict["price_map_dict"] == {"SPY": 123.5}
    assert result_dict["source_str"] == "Norgate local" and call_list
    assert norgate_snapshot_store.is_snapshot_mode_enabled_bool()


def test_monthly_snapshot_fallback_is_cached_and_new_exact_snapshot_takes_priority(tmp_path, monkeypatch):
    call_list = _local_fixture(monkeypatch)
    monkeypatch.setenv("ALPHA_USE_NORGATE_SNAPSHOT_BOOL", "true")
    _snapshot_path(tmp_path, close_str="2026-07-31")
    before_list = sorted(str(path_obj) for path_obj in tmp_path.rglob("*"))
    result_dict = _read_dict()
    assert result_dict["available_bool"] and result_dict["source_str"] == "Norgate local"
    count_int = len(call_list)
    result_dict["price_map_dict"]["SPY"] = -1
    assert _read_dict()["price_map_dict"] == {"SPY": 123.5} and len(call_list) == count_int
    assert sorted(str(path_obj) for path_obj in tmp_path.rglob("*")) == before_list
    _snapshot_path(tmp_path)
    assert _read_dict()["source_str"] == "Norgate snapshot"
    assert _read_dict()["price_map_dict"] == {"SPY": 100}
    assert len(call_list) == count_int


@pytest.mark.parametrize("missing_file_str", ["manifest.json", "prices.parquet"])
def test_partial_existing_snapshot_does_not_probe_local_source(tmp_path, missing_file_str):
    snapshot_path = _snapshot_path(tmp_path)
    (snapshot_path / missing_file_str).unlink()
    result_dict = _read_dict()
    assert not result_dict["available_bool"]
    assert result_dict["reason_str"] == "Closing price snapshot could not be read"


def test_missing_snapshot_and_local_failure_give_reason_and_recover(monkeypatch):
    monkeypatch.setattr(pod_close_prices, "_local_response_tuple", lambda *args, **kwargs: (_ for _ in ()).throw(requests.ConnectionError()))
    result_dict = _read_dict()
    assert not result_dict["available_bool"] and result_dict["price_map_dict"] == {}
    assert result_dict["reason_str"] == "No saved closing prices; local prices unavailable"
    _local_fixture(monkeypatch)
    monkeypatch.setenv("ALPHA_USE_NORGATE_SNAPSHOT_BOOL", "true")
    assert _read_dict()["price_map_dict"] == {"SPY": 123.5}


@pytest.mark.parametrize("price_override_dict", [{"date_str": "2026-08-06"}, {"count_str": "0"}])
def test_missing_snapshot_cannot_use_stale_or_missing_local_record(monkeypatch, price_override_dict):
    _local_fixture(monkeypatch, price_override_dict=price_override_dict)
    monkeypatch.setenv("ALPHA_USE_NORGATE_SNAPSHOT_BOOL", "true")
    result_dict = _read_dict()
    assert not result_dict["available_bool"] and result_dict["price_map_dict"] == {}


def test_no_partial_map_when_one_symbol_missing_or_source_contains_duplicate(tmp_path):
    snapshot_path = _snapshot_path(tmp_path)
    assert not _read_dict(["SPY", "QQQ"])["available_bool"]
    price_df = pd.read_parquet(snapshot_path / "prices.parquet")
    pd.concat([price_df, price_df]).to_parquet(snapshot_path / "prices.parquet", index=False)
    _refresh_hash(snapshot_path)
    assert not _read_dict()["available_bool"]


def test_raw_field_absence_cannot_use_adjusted_close(tmp_path):
    snapshot_path = _snapshot_path(tmp_path)
    price_df = pd.read_parquet(snapshot_path / "prices.parquet").drop(columns=["Unadjusted Close"])
    price_df.to_parquet(snapshot_path / "prices.parquet", index=False)
    _refresh_hash(snapshot_path)
    assert not _read_dict()["available_bool"]


@pytest.mark.parametrize("target_str", ["prices.parquet", "manifest.json"])
def test_modified_file_invalidates_success_cache_and_previous_canonical_hash_validation(tmp_path, target_str):
    snapshot_path = _snapshot_path(tmp_path)
    assert _read_dict()["available_bool"]
    (snapshot_path / target_str).write_bytes(b"broken")
    assert not _read_dict()["available_bool"]


def test_cache_is_scoped_to_configured_root_and_generated_time(tmp_path, monkeypatch):
    snapshot_path = _snapshot_path(tmp_path)
    assert _read_dict()["available_bool"]
    assert not _read_dict(as_of_ts=datetime(2026, 8, 7, 21, tzinfo=UTC))["available_bool"]
    other_path = tmp_path / "other"
    _snapshot_path(other_path, override_dict={"Unadjusted Close": 250})
    monkeypatch.setenv("NORGATE_SNAPSHOT_ROOT", str(other_path))
    assert _read_dict()["price_map_dict"] == {"SPY": 250}
    assert snapshot_path.exists()


def test_declared_universe_is_part_of_current_byte_identity_and_hash_validation(tmp_path):
    snapshot_path = _snapshot_path(tmp_path, universe_df=pd.DataFrame({"SPY": [True]}, index=[pd.Timestamp(CLOSE_STR)]))
    assert _read_dict()["available_bool"]
    (snapshot_path / "universe.parquet").write_bytes(b"broken")
    assert not _read_dict()["available_bool"]


@pytest.mark.parametrize("override_dict", [{"close_date_str": "20260807"}, {"close_date_str": "2026-08-08"},
    {"close_date_str": "2026-08-11"}, {"profile_str": "../../foreign"}, {"profile_str": "unapproved_profile"},
    {"as_of_ts": datetime(2026, 8, 10)}, {"as_of_ts": datetime(2026, 8, 7, 19, 59, tzinfo=UTC)}])
def test_invalid_scope_and_unclosed_session_fail_before_source_read(tmp_path, override_dict):
    _snapshot_path(tmp_path)
    assert not _read_dict(**override_dict)["available_bool"]


@pytest.mark.parametrize("symbol_list", [["$SPX"], ["SPY", "SPY"], ["spy"], ["/SPY"], ["BRK B"], [None],
    [["SPY"]], [f"S{index_int}" for index_int in range(129)]])
def test_unsupported_duplicate_and_oversized_symbol_requests_fail_closed(symbol_list):
    assert not _read_dict(symbol_list)["available_bool"]


def test_cash_only_has_no_price_io_and_missing_snapshot_root_cannot_switch_to_local(monkeypatch):
    monkeypatch.delenv("NORGATE_SNAPSHOT_ROOT")
    assert _read_dict([])["available_bool"]
    assert not _read_dict()["available_bool"]


def _local_fixture(monkeypatch, *, metadata_override_dict=None, price_override_dict=None):
    monkeypatch.setenv("ALPHA_USE_NORGATE_SNAPSHOT_BOOL", "false")
    call_list = []
    metadata_dict = {"currency": "USD", "basetype": "Stock Market", "subtype1": "Equity", "subtype2": "Operating/Holding Company"}
    metadata_dict.update(metadata_override_dict or {})
    price_dict = {"date_str": CLOSE_STR, "value_float": 123.5, "count_str": "1", "format_str": "<M8[D],f4"}
    price_dict.update(price_override_dict or {})

    def local_response_tuple(session_obj, endpoint_str, deadline_float, *, parameter_dict=None):
        assert session_obj.trust_env is False
        call_list.append((endpoint_str, deepcopy(parameter_dict)))
        if endpoint_str.startswith("security/"):
            return {}, metadata_dict[endpoint_str.rsplit("/", 1)[1]].encode()
        assert parameter_dict["start_date"] == parameter_dict["end_date"] == CLOSE_STR
        assert parameter_dict["stock_price_adjustment_setting"] == parameter_dict["padding_setting"] == "NONE"
        assert parameter_dict["fields"] == "Close"
        record_arr = np.array([(np.datetime64(price_dict["date_str"]), price_dict["value_float"])],
            dtype=[("Date", "<M8[D]"), ("Close", "f4")])
        return {"X-Norgate-Data-Record-Count": price_dict["count_str"], "X-Norgate-Data-Field-Count": "2",
            "X-Norgate-Data-Field-Names": "Date,Close", "X-Norgate-Data-Field-Formats": price_dict["format_str"]}, record_arr.tobytes()

    monkeypatch.setattr(pod_close_prices, "_local_response_tuple", local_response_tuple)
    return call_list


@pytest.mark.parametrize("etf_bool", [True, False])
def test_local_raw_close_uses_exact_day_no_padding_and_verified_usd_instrument(monkeypatch, etf_bool):
    call_list = _local_fixture(monkeypatch, metadata_override_dict={"subtype1": "Exchange Traded Product",
        "subtype2": "Exchange Traded Fund (ETF)"} if etf_bool else {})
    result_dict = _read_dict()
    assert result_dict["available_bool"] and result_dict["source_str"] == "Norgate local"
    assert result_dict["price_map_dict"] == {"SPY": 123.5}
    count_int = len(call_list)
    result_dict["price_map_dict"]["SPY"] = -1
    assert _read_dict()["price_map_dict"] == {"SPY": 123.5} and len(call_list) == count_int


@pytest.mark.parametrize("override_dict", [{"currency": "EUR"}, {"currency": ""}, {"basetype": "Futures"},
    {"subtype1": "Index"}, {"subtype1": "Exchange Traded Product", "subtype2": "Exchange Traded Note (ETN)"}])
def test_local_unsupported_metadata_never_requests_prices(monkeypatch, override_dict):
    call_list = _local_fixture(monkeypatch, metadata_override_dict=override_dict)
    assert not _read_dict()["available_bool"]
    assert all(not endpoint_str.startswith("prices/") for endpoint_str, _ in call_list)


@pytest.mark.parametrize("override_dict", [{"date_str": "2026-08-06"}, {"value_float": float("nan")},
    {"value_float": 0}, {"value_float": -1}, {"count_str": "0"}, {"count_str": "2"}, {"format_str": "<M8[D],O"}])
def test_invalid_local_price_record_does_not_return_partial_or_previous_prices(monkeypatch, override_dict):
    _local_fixture(monkeypatch, price_override_dict=override_dict)
    result_dict = _read_dict()
    assert not result_dict["available_bool"] and result_dict["price_map_dict"] == {}


def test_local_error_never_loads_snapshot_as_fallback(monkeypatch, tmp_path):
    _snapshot_path(tmp_path)
    monkeypatch.setenv("ALPHA_USE_NORGATE_SNAPSHOT_BOOL", "false")
    monkeypatch.setattr(pod_close_prices, "_local_response_tuple", lambda *args, **kwargs: (_ for _ in ()).throw(requests.Timeout()))
    assert not _read_dict()["available_bool"]


def test_local_cache_is_bounded_and_rechecks_after_expiry(monkeypatch):
    call_list = _local_fixture(monkeypatch)
    clock_list = [10.0]
    monkeypatch.setattr(pod_close_prices.time, "monotonic", lambda: clock_list[0])
    assert _read_dict()["available_bool"]
    first_count_int = len(call_list)
    clock_list[0] += pod_close_prices.CACHE_SECONDS_FLOAT + 1
    assert _read_dict()["available_bool"]
    assert len(call_list) == first_count_int * 2
    for index_int in range(pod_close_prices.CACHE_LIMIT_INT + 1):
        assert _read_dict([f"S{index_int}"])["available_bool"]
    assert len(pod_close_prices._cache_dict) == pod_close_prices.CACHE_LIMIT_INT


def test_local_transport_has_fixed_loopback_no_redirects_and_bounded_body(monkeypatch):
    response_obj = SimpleNamespace(status_code=200, headers={}, iter_content=lambda chunk_int: [b"USD"])

    class ResponseContext:
        def __enter__(self):
            return response_obj

        def __exit__(self, *argument_tuple):
            return False

    call_list = []
    session_obj = SimpleNamespace(get=lambda url_str, **option_dict: call_list.append((url_str, option_dict)) or ResponseContext())
    monkeypatch.undo()
    header_dict, body_bytes = pod_close_prices._local_response_tuple(session_obj, "security/SPY/currency", pod_close_prices.time.monotonic() + 1)
    assert body_bytes == b"USD"
    url_str, option_dict = call_list[0]
    assert url_str == "http://127.0.0.1:38889/api/v1/security/SPY/currency"
    assert option_dict["allow_redirects"] is False and option_dict["stream"] is True
    assert all(0 < timeout_float <= .5 for timeout_float in option_dict["timeout"])
    response_obj.iter_content = lambda chunk_int: [b"x" * (pod_close_prices.RESPONSE_BYTE_LIMIT_INT + 1)]
    with pytest.raises(ValueError):
        pod_close_prices._local_response_tuple(session_obj, "prices/SPY", pod_close_prices.time.monotonic() + 1)
