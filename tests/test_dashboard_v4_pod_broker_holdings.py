"""Saved portfolio values keep their own time, owner, shares and cash."""

from dataclasses import replace
import json
from pathlib import Path
import sqlite3

import pytest

from alpha.live.dashboard_v4 import pod_broker_holdings
from alpha.live.dashboard_v4.pod_broker_holdings import load_broker_holdings_dict
from alpha.live.state_store_v2 import LiveStateStore
from test_dashboard_v4_evidence import NOW_TS, build_fixture_tuple, update_db


OBSERVED_STR = "2026-09-18T13:55:00+00:00"


def _payload_dict(target_obj):
    return {"schema_version_int": 1, "owner_dict": {
        field_str: getattr(target_obj.release_obj, field_str)
        for field_str in ("release_id_str", "user_id_str", "pod_id_str", "account_route_str", "mode_str")},
        "available_bool": True, "reason_str": "", "account_route_str": "U111",
        "observed_timestamp_str": OBSERVED_STR, "source_str": "IBKR portfolio",
        "currency_str": "USD", "cash_float": 125.25, "broker_nav_float": 1234.5,
        "position_list": [
            {"symbol_str": "SPY", "conid_int": 1, "currency_str": "USD", "shares_float": 3.125,
                "market_price_float": 100.0, "value_float": 312.5},
            {"symbol_str": "SHORT", "conid_int": 2, "currency_str": "USD", "shares_float": -1.0,
                "market_price_float": 20.0, "value_float": -20.0}]}


def _save_payload(target_obj, payload_obj):
    payload_str = json.dumps(payload_obj) if isinstance(payload_obj, dict) else payload_obj
    update_db(target_obj, """INSERT OR REPLACE INTO broker_snapshot_cache
        (account_route_str,snapshot_timestamp_str,cash_float,total_value_float,net_liq_float,
         position_json_str,open_order_id_json_str,updated_timestamp_str,portfolio_valuation_json_str)
        VALUES ('U111',?,999,999,999,'{"WRONG":999}','[]',?,?)""",
        (NOW_TS.isoformat(), NOW_TS.isoformat(), payload_str))


@pytest.fixture
def broker_source_tuple(tmp_path):
    target_obj, _ = build_fixture_tuple(tmp_path)
    payload_dict = _payload_dict(target_obj)
    _save_payload(target_obj, payload_dict)
    return target_obj, payload_dict


def _read_dict(source_tuple):
    return load_broker_holdings_dict(source_tuple[0], as_of_ts=NOW_TS)


def test_read_uses_only_original_payload_and_does_not_write(broker_source_tuple):
    target_obj, payload_dict = broker_source_tuple
    path_obj = Path(target_obj.db_path_str)
    original_bytes, original_mtime_int = path_obj.read_bytes(), path_obj.stat().st_mtime_ns
    result_dict = _read_dict(broker_source_tuple)
    assert result_dict["available_bool"]
    assert result_dict["position_list"] == payload_dict["position_list"]
    assert result_dict["cash_float"] == 125.25 and result_dict["broker_nav_float"] == 1234.5
    assert result_dict["observed_timestamp_str"] == OBSERVED_STR
    assert path_obj.read_bytes() == original_bytes and path_obj.stat().st_mtime_ns == original_mtime_int


@pytest.mark.parametrize("legacy_str", ["column", "null", "table"])
def test_legacy_missing_values_are_expected_not_fabricated(broker_source_tuple, legacy_str):
    target_obj, _ = broker_source_tuple
    if legacy_str == "column":
        update_db(target_obj, "ALTER TABLE broker_snapshot_cache DROP COLUMN portfolio_valuation_json_str")
    elif legacy_str == "table":
        update_db(target_obj, "DROP TABLE broker_snapshot_cache")
    else:
        _save_payload(target_obj, None)
    result_dict = _read_dict(broker_source_tuple)
    assert not result_dict["available_bool"]
    assert result_dict["reason_str"] == "IBKR position values not saved yet"


def test_failed_capture_replaces_values_without_using_old_evidence(broker_source_tuple):
    target_obj, payload_dict = broker_source_tuple
    payload_dict.update(available_bool=False, reason_str="internal detail", position_list=[])
    _save_payload(target_obj, payload_dict)
    result_dict = _read_dict(broker_source_tuple)
    assert not result_dict["available_bool"] and not result_dict["position_list"]
    assert result_dict["reason_str"] == "IBKR position values unavailable at the last capture"


@pytest.mark.parametrize("field_str,value_obj", [
    ("schema_version_int", 2), ("schema_version_int", True), ("owner_dict", {}),
    ("account_route_str", "FOREIGN"), ("source_str", "Norgate"),
    ("currency_str", "EUR"), ("available_bool", 1), ("cash_float", True),
    ("cash_float", float("nan")), ("broker_nav_float", float("inf")),
    ("observed_timestamp_str", "garbage"), ("observed_timestamp_str", "2026-09-18T13:55:00"),
    ("observed_timestamp_str", "2026-09-18T09:55:00-04:00"),
    ("observed_timestamp_str", "2026-09-18T14:00:01+00:00"), ("position_list", None)])
def test_invalid_header_is_unavailable(broker_source_tuple, field_str, value_obj):
    target_obj, payload_dict = broker_source_tuple
    payload_dict[field_str] = value_obj
    _save_payload(target_obj, payload_dict)
    assert not _read_dict(broker_source_tuple)["available_bool"]


@pytest.mark.parametrize("field_str,value_obj", [
    ("symbol_str", " SPY"), ("symbol_str", ""), ("symbol_str", "X" * 101),
    ("conid_int", 0), ("conid_int", True), ("conid_int", "1"),
    ("currency_str", "EUR"), ("shares_float", "3.125"), ("shares_float", True),
    ("shares_float", float("inf")), ("shares_float", 0), ("market_price_float", 0),
    ("market_price_float", -1), ("market_price_float", float("nan")),
    ("value_float", -312.5), ("value_float", 313.0), ("value_float", float("inf"))])
def test_invalid_or_inconsistent_row_withholds_whole_portfolio(broker_source_tuple, field_str, value_obj):
    target_obj, payload_dict = broker_source_tuple
    payload_dict["position_list"][0][field_str] = value_obj
    _save_payload(target_obj, payload_dict)
    result_dict = _read_dict(broker_source_tuple)
    assert not result_dict["available_bool"] and not result_dict["position_list"]


@pytest.mark.parametrize("duplicate_str", ["symbol", "conid"])
def test_contract_or_symbol_ambiguity_is_rejected(broker_source_tuple, duplicate_str):
    target_obj, payload_dict = broker_source_tuple
    field_str = "symbol_str" if duplicate_str == "symbol" else "conid_int"
    payload_dict["position_list"][1][field_str] = payload_dict["position_list"][0][field_str]
    _save_payload(target_obj, payload_dict)
    assert not _read_dict(broker_source_tuple)["available_bool"]


@pytest.mark.parametrize("field_str,value_obj", [("release_id_str", "new"), ("user_id_str", "other"),
    ("pod_id_str", "other"), ("account_route_str", "other"), ("mode_str", "paper"), ("enabled_bool", False)])
def test_configured_target_and_saved_owner_must_match(broker_source_tuple, field_str, value_obj):
    target_obj, _ = broker_source_tuple
    target_obj = replace(target_obj, release_obj=replace(target_obj.release_obj, **{field_str: value_obj}))
    assert not load_broker_holdings_dict(target_obj, as_of_ts=NOW_TS)["available_bool"]


def test_release_change_requires_new_observation(broker_source_tuple):
    target_obj, _ = broker_source_tuple
    new_release_obj = replace(target_obj.release_obj, release_id_str="new-release")
    LiveStateStore(target_obj.db_path_str).upsert_release(new_release_obj)
    target_obj = replace(target_obj, release_obj=new_release_obj)
    assert not load_broker_holdings_dict(target_obj, as_of_ts=NOW_TS)["available_bool"]


def test_foreign_history_does_not_override_explicit_payload_owner(broker_source_tuple):
    target_obj, _ = broker_source_tuple
    LiveStateStore(target_obj.db_path_str).upsert_release(replace(target_obj.release_obj,
        release_id_str="legacy-other", pod_id_str="other-pod"))
    assert _read_dict(broker_source_tuple)["available_bool"]


@pytest.mark.parametrize("payload_str", ["null", "[]", "", '{"schema_version_int":1,"schema_version_int":1}', "[" * 1100 + "]" * 1100])
def test_malformed_json_fails_closed(broker_source_tuple, payload_str):
    _save_payload(broker_source_tuple[0], payload_str)
    assert not _read_dict(broker_source_tuple)["available_bool"]


def test_json_bytes_and_rows_are_bounded(broker_source_tuple, monkeypatch):
    monkeypatch.setattr(pod_broker_holdings, "VALUATION_BYTE_LIMIT_INT", 32)
    assert not _read_dict(broker_source_tuple)["available_bool"]
    monkeypatch.setattr(pod_broker_holdings, "VALUATION_BYTE_LIMIT_INT", 1048576)
    monkeypatch.setattr(pod_broker_holdings, "POSITION_LIMIT_INT", 1)
    assert not _read_dict(broker_source_tuple)["available_bool"]


def test_cash_only_and_zero_rows_are_supported(broker_source_tuple):
    target_obj, payload_dict = broker_source_tuple
    payload_dict["position_list"] = []
    _save_payload(target_obj, payload_dict)
    assert _read_dict(broker_source_tuple)["available_bool"]
    payload_dict["position_list"] = [{"symbol_str": "FLAT", "conid_int": 5, "currency_str": "USD",
        "shares_float": 0, "market_price_float": 0, "value_float": 0}]
    _save_payload(target_obj, payload_dict)
    assert _read_dict(broker_source_tuple)["position_list"] == []


def test_broker_rounding_tolerance_accepts_two_cents_not_more(broker_source_tuple):
    target_obj, payload_dict = broker_source_tuple
    payload_dict["position_list"][0]["value_float"] += .02
    _save_payload(target_obj, payload_dict)
    assert _read_dict(broker_source_tuple)["available_bool"]
    payload_dict["position_list"][0]["value_float"] += .01
    _save_payload(target_obj, payload_dict)
    assert not _read_dict(broker_source_tuple)["available_bool"]


def test_tiny_signed_value_cannot_pass_through_float_underflow(broker_source_tuple):
    target_obj, payload_dict = broker_source_tuple
    payload_dict["position_list"][0].update(shares_float=1e-300, market_price_float=1.0, value_float=-1e-300)
    _save_payload(target_obj, payload_dict)
    assert not _read_dict(broker_source_tuple)["available_bool"]


def test_exclusive_writer_lock_fails_closed_and_recovers(broker_source_tuple):
    target_obj, _ = broker_source_tuple
    connection_obj = sqlite3.connect(target_obj.db_path_str)
    try:
        connection_obj.execute("BEGIN EXCLUSIVE")
        assert not _read_dict(broker_source_tuple)["available_bool"]
    finally:
        connection_obj.rollback()
        connection_obj.close()
    assert _read_dict(broker_source_tuple)["available_bool"]


def test_missing_database_is_not_created(tmp_path, broker_source_tuple):
    target_obj = replace(broker_source_tuple[0], db_path_str=str(tmp_path / "missing.sqlite3"))
    assert not load_broker_holdings_dict(target_obj, as_of_ts=NOW_TS)["available_bool"]
    assert not Path(target_obj.db_path_str).exists()
