"""Saved EOD facts stay scoped, retrospective and in one broker observation."""

from dataclasses import replace
from datetime import UTC, datetime
import json

import pytest

from alpha.live.dashboard_v4 import pod_eod_holdings
from alpha.live.dashboard_v4.pod_eod_holdings import load_eod_holdings_dict
from alpha.live.state_store_v2 import LiveStateStore
from test_dashboard_v4_evidence import NOW_TS, build_fixture_tuple, update_db


CLOSE_STR = "2026-09-17"
OBSERVED_STR = "2026-09-17T20:10:00.000001+00:00"


def _eod(target_obj, *, timestamp_str=OBSERVED_STR, positions_str='{"SPY":3.125,"SHORT":-4,"FLAT":0}',
         cash_float=123.45, nav_float=1000.25, pod_str="pod", owner_str="owner", account_str="U111",
         stage_str="eod", source_str="broker"):
    update_db(target_obj, """INSERT INTO pod_state_history
        (pod_id_str,user_id_str,account_route_str,position_json_str,cash_float,total_value_float,
         strategy_state_json_str,snapshot_stage_str,snapshot_source_str,updated_timestamp_str,recorded_timestamp_str)
        VALUES (?,?,?,?,?,?,'{}',?,?,?,?)""", (pod_str, owner_str, account_str, positions_str, cash_float,
            nav_float, stage_str, source_str, timestamp_str, NOW_TS.isoformat()))


def test_exact_close_returns_one_broker_observation_without_writes(tmp_path):
    target_obj, _ = build_fixture_tuple(tmp_path)
    _eod(target_obj)
    path_obj = tmp_path / "pod.sqlite3"
    before_bytes, before_mtime_int = path_obj.read_bytes(), path_obj.stat().st_mtime_ns
    result_dict = load_eod_holdings_dict(target_obj, close_date_str=CLOSE_STR, as_of_ts=NOW_TS)
    assert result_dict["available_bool"] is True
    assert result_dict["position_map_dict"] == {"SPY": 3.125, "SHORT": -4, "FLAT": 0}
    assert result_dict["cash_float"] == 123.45
    assert result_dict["broker_nav_float"] == 1000.25
    assert result_dict["close_date_str"] == CLOSE_STR
    assert result_dict["observed_timestamp_str"] == OBSERVED_STR
    assert result_dict["source_str"] == "Saved broker EOD"
    assert result_dict["release_id_str"] == "release"
    assert result_dict["user_id_str"] == "owner"
    assert result_dict["account_route_str"] == "U111"
    assert path_obj.read_bytes() == before_bytes
    assert path_obj.stat().st_mtime_ns == before_mtime_int


def test_latest_close_uses_observation_time_and_ignores_current_day(tmp_path):
    target_obj, _ = build_fixture_tuple(tmp_path)
    _eod(target_obj, timestamp_str="2026-09-18T20:10:00+00:00", positions_str='{"SPY":999}')
    _eod(target_obj, positions_str='{"SPY":7}')
    _eod(target_obj, timestamp_str="2026-09-16T20:10:00+00:00", positions_str='{"SPY":2}')
    result_dict = load_eod_holdings_dict(target_obj, as_of_ts=datetime(2026, 9, 18, 22, tzinfo=UTC))
    assert result_dict["available_bool"] is True
    assert result_dict["position_map_dict"] == {"SPY": 7}
    assert result_dict["close_date_str"] == CLOSE_STR


def test_latest_same_date_is_not_first_inserted_row(tmp_path):
    target_obj, _ = build_fixture_tuple(tmp_path)
    _eod(target_obj, timestamp_str="2026-09-17T20:11:00+00:00", positions_str='{"SPY":7}', cash_float=17, nav_float=77)
    _eod(target_obj, positions_str='{"SPY":2}', cash_float=12, nav_float=22)
    result_dict = load_eod_holdings_dict(target_obj, close_date_str=CLOSE_STR, as_of_ts=NOW_TS)
    assert result_dict["position_map_dict"] == {"SPY": 7}
    assert (result_dict["cash_float"], result_dict["broker_nav_float"]) == (17, 77)


def test_exact_date_never_borrows_other_days(tmp_path):
    target_obj, _ = build_fixture_tuple(tmp_path)
    _eod(target_obj)
    for date_str in ("2026-09-16", "2026-09-18", "2026-09-19"):
        assert load_eod_holdings_dict(target_obj, close_date_str=date_str, as_of_ts=NOW_TS)["available_bool"] is False


def test_explicit_empty_map_proves_flat_missing_does_not(tmp_path):
    target_obj, _ = build_fixture_tuple(tmp_path)
    assert load_eod_holdings_dict(target_obj, as_of_ts=NOW_TS)["available_bool"] is False
    _eod(target_obj, positions_str="{}")
    result_dict = load_eod_holdings_dict(target_obj, as_of_ts=NOW_TS)
    assert result_dict["available_bool"] is True
    assert result_dict["position_map_dict"] == {}


@pytest.mark.parametrize("positions_str", ["null", "[]", "", '{"SPY":1,"SPY":2}', '{"SPY":NaN}',
    '{"SPY":1e999}', '{"SPY":true}', '{"SPY":"2"}', '{" SPY":2}', '{"SPY":{}}'])
@pytest.mark.parametrize("close_date_str", [None, CLOSE_STR])
def test_bad_latest_map_never_revives_older_healthy_close(tmp_path, positions_str, close_date_str):
    target_obj, _ = build_fixture_tuple(tmp_path)
    _eod(target_obj, timestamp_str="2026-09-17T20:10:00+00:00")
    _eod(target_obj, timestamp_str="2026-09-17T20:11:00+00:00", positions_str=positions_str)
    result_dict = load_eod_holdings_dict(target_obj, close_date_str=close_date_str, as_of_ts=NOW_TS)
    assert result_dict["available_bool"] is False
    assert result_dict["position_map_dict"] == {}
    assert result_dict["cash_float"] is None


@pytest.mark.parametrize("timestamp_str", ["garbage", "2026-09-17T21:00:00", "2026-09-18T23:00:00+00:00"])
def test_bad_or_future_observation_cannot_restore_older_data(tmp_path, timestamp_str):
    target_obj, _ = build_fixture_tuple(tmp_path)
    _eod(target_obj)
    _eod(target_obj, timestamp_str=timestamp_str)
    assert load_eod_holdings_dict(target_obj, as_of_ts=NOW_TS)["available_bool"] is False


def test_duplicate_latest_instant_even_with_different_offsets_is_ambiguous(tmp_path):
    target_obj, _ = build_fixture_tuple(tmp_path)
    _eod(target_obj, timestamp_str="2026-09-17T20:10:00+00:00")
    _eod(target_obj, timestamp_str="2026-09-17T16:10:00-04:00")
    assert load_eod_holdings_dict(target_obj, as_of_ts=NOW_TS)["available_bool"] is False


def test_sqlite_submillisecond_order_cannot_hide_duplicate_latest(tmp_path):
    target_obj, _ = build_fixture_tuple(tmp_path)
    _eod(target_obj, timestamp_str="2026-09-17T20:10:00.000100+00:00")
    _eod(target_obj, timestamp_str="2026-09-17T20:10:00.000001+00:00")
    _eod(target_obj, timestamp_str="2026-09-17T20:10:00.000100+00:00")
    assert load_eod_holdings_dict(target_obj, as_of_ts=NOW_TS)["available_bool"] is False


def test_latest_submillisecond_observation_wins_despite_recording_order(tmp_path):
    target_obj, _ = build_fixture_tuple(tmp_path)
    _eod(target_obj, timestamp_str="2026-09-17T20:10:00.000100+00:00", positions_str='{"SPY":7}')
    _eod(target_obj, timestamp_str="2026-09-17T20:10:00.000001+00:00", positions_str='{"SPY":2}')
    assert load_eod_holdings_dict(target_obj, as_of_ts=NOW_TS)["position_map_dict"] == {"SPY": 7}


def test_bounded_read_must_not_split_ambiguous_latest_time_group(tmp_path, monkeypatch):
    target_obj, _ = build_fixture_tuple(tmp_path)
    _eod(target_obj)
    _eod(target_obj)
    monkeypatch.setattr(pod_eod_holdings, "HISTORY_LIMIT_INT", 1)
    assert load_eod_holdings_dict(target_obj, as_of_ts=NOW_TS)["available_bool"] is False


@pytest.mark.parametrize("field_str,value_str", [("mode_str", "paper"), ("enabled_bool", False),
    ("release_id_str", "not-saved"), ("user_id_str", "wrong"), ("account_route_str", "wrong"), ("pod_id_str", "wrong")])
def test_target_must_be_enabled_live_and_exact_saved_release(tmp_path, field_str, value_str):
    target_obj, _ = build_fixture_tuple(tmp_path)
    _eod(target_obj)
    target_obj = replace(target_obj, release_obj=replace(target_obj.release_obj, **{field_str: value_str}))
    assert load_eod_holdings_dict(target_obj, as_of_ts=NOW_TS)["available_bool"] is False


@pytest.mark.parametrize("field_str,value_str", [("user_id_str", "wrong"), ("account_route_str", "wrong"), ("mode_str", "paper")])
def test_conflicting_saved_same_pod_ownership_is_unknown(tmp_path, field_str, value_str):
    target_obj, _ = build_fixture_tuple(tmp_path)
    _eod(target_obj)
    LiveStateStore(target_obj.db_path_str).upsert_release(replace(target_obj.release_obj,
        release_id_str="old-release", **{field_str: value_str}))
    assert load_eod_holdings_dict(target_obj, as_of_ts=NOW_TS)["available_bool"] is False


@pytest.mark.parametrize("owner_str,account_str", [("wrong", "U111"), ("owner", "wrong")])
def test_latest_eod_row_must_have_exact_owner_and_account(tmp_path, owner_str, account_str):
    target_obj, _ = build_fixture_tuple(tmp_path)
    _eod(target_obj)
    _eod(target_obj, timestamp_str="2026-09-17T20:11:00+00:00", owner_str=owner_str, account_str=account_str)
    assert load_eod_holdings_dict(target_obj, as_of_ts=NOW_TS)["available_bool"] is False


def test_other_pod_cannot_replace_explicitly_owned_eod(tmp_path):
    target_obj, _ = build_fixture_tuple(tmp_path)
    _eod(target_obj)
    LiveStateStore(target_obj.db_path_str).upsert_release(replace(target_obj.release_obj,
        release_id_str="other-release", pod_id_str="other-pod"))
    _eod(target_obj, timestamp_str="2026-09-17T20:11:00+00:00", pod_str="other-pod", positions_str='{"OTHER":999}')
    assert load_eod_holdings_dict(target_obj, as_of_ts=NOW_TS)["position_map_dict"]["SPY"] == 3.125


@pytest.mark.parametrize("stage_str,source_str", [("post_execution", "broker"), ("eod", "pod_state"), ("unknown", "broker")])
def test_model_or_non_eod_state_is_not_a_closing_broker_source(tmp_path, stage_str, source_str):
    target_obj, _ = build_fixture_tuple(tmp_path)
    _eod(target_obj, stage_str=stage_str, source_str=source_str)
    assert load_eod_holdings_dict(target_obj, as_of_ts=NOW_TS)["available_bool"] is False


@pytest.mark.parametrize("date_str,timestamp_str,asof_str,expected_bool", [
    ("2026-09-17", "2026-09-17T20:09:59+00:00", "2026-09-18T14:00:00+00:00", False),
    ("2026-09-17", "2026-09-17T20:10:00+00:00", "2026-09-18T14:00:00+00:00", True),
    ("2026-11-27", "2026-11-27T18:09:59+00:00", "2026-11-28T14:00:00+00:00", False),
    ("2026-11-27", "2026-11-27T18:10:00+00:00", "2026-11-28T14:00:00+00:00", True),
    ("2026-11-30", "2026-11-30T21:10:00+00:00", "2026-12-01T14:00:00+00:00", True),
    ("2026-09-19", "2026-09-19T20:10:00+00:00", "2026-09-20T14:00:00+00:00", False),
])
def test_eod_uses_canonical_session_close_plus_ten_minutes(tmp_path, date_str, timestamp_str, asof_str, expected_bool):
    target_obj, _ = build_fixture_tuple(tmp_path)
    _eod(target_obj, timestamp_str=timestamp_str)
    result_dict = load_eod_holdings_dict(target_obj, close_date_str=date_str, as_of_ts=datetime.fromisoformat(asof_str))
    assert result_dict["available_bool"] is expected_bool


@pytest.mark.parametrize("positions_str", ['{"SPY":1}' + "\x00" * 262144,
    json.dumps({"SPY": 1}) + "\u00e9" * 150000], ids=["nul", "unicode"])
def test_sql_byte_limit_rejects_nul_and_unicode_before_json_loading(tmp_path, monkeypatch, positions_str):
    target_obj, _ = build_fixture_tuple(tmp_path)
    _eod(target_obj, positions_str=positions_str)
    seen_list = []
    original_func = pod_eod_holdings._position_map_dict

    def bounded_parser(position_str):
        seen_list.append(position_str)
        return original_func(position_str)

    monkeypatch.setattr(pod_eod_holdings, "_position_map_dict", bounded_parser)
    assert load_eod_holdings_dict(target_obj, as_of_ts=NOW_TS)["available_bool"] is False
    assert seen_list == [None]


@pytest.mark.parametrize("column_str", ["cash_float", "total_value_float"])
@pytest.mark.parametrize("value_obj", [float("inf"), "bad"])
def test_nonfinite_money_cannot_make_an_available_snapshot(tmp_path, column_str, value_obj):
    target_obj, _ = build_fixture_tuple(tmp_path)
    _eod(target_obj)
    update_db(target_obj, f"UPDATE pod_state_history SET {column_str}=?", (value_obj,))
    assert load_eod_holdings_dict(target_obj, as_of_ts=NOW_TS)["available_bool"] is False


def test_negative_money_is_preserved_for_truthful_downstream_handling(tmp_path):
    target_obj, _ = build_fixture_tuple(tmp_path)
    _eod(target_obj, cash_float=-123, nav_float=-5)
    result_dict = load_eod_holdings_dict(target_obj, as_of_ts=NOW_TS)
    assert result_dict["available_bool"] is True
    assert (result_dict["cash_float"], result_dict["broker_nav_float"]) == (-123, -5)


def test_missing_database_is_not_created(tmp_path):
    target_obj, _ = build_fixture_tuple(tmp_path)
    missing_path_obj = tmp_path / "absent.sqlite3"
    target_obj = replace(target_obj, db_path_str=str(missing_path_obj))
    assert load_eod_holdings_dict(target_obj, as_of_ts=NOW_TS)["available_bool"] is False
    assert not missing_path_obj.exists()
