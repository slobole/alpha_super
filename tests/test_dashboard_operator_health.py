"""Operator evidence checks: idle calendars never manufacture fresh evidence."""

from datetime import UTC, datetime
import sqlite3
from types import SimpleNamespace

import pytest

from alpha.live.dashboard import _build_eod_snapshot_dict


def _eod_evidence_dict(as_of_str, snapshot_str_list, *, source_str="broker", account_str="TEST_ACCOUNT"):
    connection_obj = sqlite3.connect(":memory:")
    connection_obj.row_factory = sqlite3.Row
    connection_obj.execute(
        "CREATE TABLE pod_state_history (pod_state_history_id_int INTEGER PRIMARY KEY, "
        "pod_id_str TEXT, snapshot_stage_str TEXT, snapshot_source_str TEXT, "
        "updated_timestamp_str TEXT, recorded_timestamp_str TEXT, position_json_str TEXT, "
        "total_value_float REAL, cash_float REAL, account_route_str TEXT)"
    )
    connection_obj.executemany(
        "INSERT INTO pod_state_history VALUES (?, 'pod_test', 'eod', ?, ?, ?, '{}', 100, 100, ?)",
        [(row_index_int, source_str, timestamp_str, timestamp_str, account_str)
         for row_index_int, timestamp_str in enumerate(snapshot_str_list)],
    )
    try:
        return _build_eod_snapshot_dict(
            connection_obj,
            SimpleNamespace(pod_id_str="pod_test", account_route_str="TEST_ACCOUNT", mode_str="live", session_calendar_id_str="XNYS"),
            None, datetime.fromisoformat(as_of_str).astimezone(UTC),
        )
    finally:
        connection_obj.close()


@pytest.mark.parametrize("as_of_str", [
    "2026-09-05T13:00:00+00:00",  # Saturday
    "2026-09-06T13:00:00+00:00",  # Sunday
    "2026-09-07T13:00:00+00:00",  # Labor Day
])
def test_closed_market_uses_last_required_eod(as_of_str):
    evidence_dict = _eod_evidence_dict(as_of_str, ["2026-09-04T20:10:00+00:00"])
    assert evidence_dict["status_str"] == "not_applicable"
    assert evidence_dict["severity_str"] == "green"
    assert evidence_dict["last_required_market_date_str"] == "2026-09-04"
    assert evidence_dict["last_required_eod_present_bool"] is True
    assert evidence_dict["expected_due_timestamp_str"] is None


@pytest.mark.parametrize("snapshot_str_list", [[], ["2026-09-03T20:10:00+00:00"]])
def test_weekend_does_not_hide_missing_friday_eod(snapshot_str_list):
    evidence_dict = _eod_evidence_dict("2026-09-05T13:00:00+00:00", snapshot_str_list)
    assert evidence_dict["status_str"] == "due_missing"
    assert evidence_dict["severity_str"] == "yellow"
    assert evidence_dict["last_required_eod_present_bool"] is False
    assert "2026-09-04" in evidence_dict["detail_str"]


def test_preclose_is_healthy_only_with_last_required_eod():
    evidence_dict = _eod_evidence_dict(
        "2026-09-08T15:00:00+00:00", ["2026-09-04T20:10:00+00:00"]
    )
    assert evidence_dict["status_str"] == "waiting"
    assert evidence_dict["severity_str"] == "green"
    assert evidence_dict["expected_market_date_str"] == "2026-09-08"
    assert evidence_dict["last_required_market_date_str"] == "2026-09-04"
    missing_evidence_dict = _eod_evidence_dict(
        "2026-09-08T15:00:00+00:00", ["2026-09-03T20:10:00+00:00"]
    )
    assert missing_evidence_dict["severity_str"] == "yellow"


def test_close_buffer_and_early_close_use_exchange_calendar():
    before_due_dict = _eod_evidence_dict(
        "2026-11-27T18:05:00+00:00", ["2026-11-25T21:10:00+00:00"]
    )
    assert before_due_dict["last_required_market_date_str"] == "2026-11-25"
    assert before_due_dict["severity_str"] == "green"
    after_due_dict = _eod_evidence_dict(
        "2026-11-27T18:11:00+00:00", ["2026-11-25T21:10:00+00:00"]
    )
    assert after_due_dict["last_required_market_date_str"] == "2026-11-27"
    assert after_due_dict["status_str"] == "due_missing"


def test_future_and_nonbroker_eod_do_not_prove_live_health():
    future_dict = _eod_evidence_dict(
        "2026-09-05T13:00:00+00:00", ["2026-09-08T20:10:00+00:00"]
    )
    assert future_dict["severity_str"] != "green"
    virtual_dict = _eod_evidence_dict(
        "2026-09-05T13:00:00+00:00", ["2026-09-04T20:10:00+00:00"], source_str="virtual_broker"
    )
    assert virtual_dict["severity_str"] != "green"


def test_wrong_account_or_preclose_eod_label_cannot_verify_weekend():
    wrong_account_dict = _eod_evidence_dict("2026-09-05T13:00:00+00:00", ["2026-09-04T20:10:00+00:00"], account_str="OTHER_CLIENT")
    assert wrong_account_dict["severity_str"] != "green"
    for timestamp_str in ("2026-09-04T17:00:00+00:00", "2026-09-04T20:09:00+00:00"):
        preclose_dict = _eod_evidence_dict("2026-09-05T13:00:00+00:00", [timestamp_str])
        assert preclose_dict["severity_str"] != "green"
