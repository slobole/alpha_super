"""Daily cycle reader: real writers, exact keys, no free text, read-only, and watchdog parity."""

from dataclasses import replace
from datetime import UTC, timedelta
import json
import sqlite3

import pytest

from alpha.live.daily_notifications import enqueue_daily_cycle_overdue
from alpha.live.daily_reconcile import claim_daily_completion_request
from alpha.live.dashboard import DashboardPodTarget
from alpha.live.dashboard_v4.cycle import build_cycle_view_dict
from alpha.live.dashboard_v4.daily_data import load_daily_cycle_dict
from alpha.live.dashboard_v4.daily_demo import DEMO_NOW_TS, DemoDailyStore
from alpha.live.dashboard_v4.pod_data import load_pod_cycles_dict
from alpha.live.dispatch_state import record_dispatch_failure
from alpha.live.execution_engine import build_broker_order_request_list_from_vplan
from alpha.live.models import BrokerOrderRequest
from test_live_daily_reconcile import CLOSE_TS, INTRADAY_TS, OPEN_TS, daily_case, reconcile_case  # noqa: F401
from test_live_daily_watchdog import _alerts


TOP_KEY_SET = {"status_str", "reason_str", "as_of_timestamp_str", "current_bool", "decision_dict", "vplan_dict",
    "opening_dispatch_dict", "execution_result_dict", "completion_request_list",
    "pod_alert_list", "execution_alert_list", "post_close_report_dict", "open_previous_list",
    "open_previous_truncated_bool", "last_closed_dict", "signal_session_dict", "session_close_timestamp_str",
    "next_open_timestamp_str"}
WATCHDOG_KIND_TO_STATUS_DICT = {"decision_incomplete": "no_decision", "opening_not_dispatched": "not_sent",
    "opening_dispatch_unconfirmed": "dispatch_unconfirmed", "cycle_open_after_close": "open_after_close"}
# Writers stamp some rows with wall time; tests read at fixed 2026-10 instants.
PINNED_CREATED_TS = OPEN_TS - timedelta(days=3) + timedelta(hours=8)
PINNED_UPDATED_TS = OPEN_TS - timedelta(minutes=10)


def _utc_str(timestamp_ts):
    return timestamp_ts.astimezone(UTC).isoformat()


def _pin(store_obj):
    with sqlite3.connect(store_obj.db_path_str) as connection_obj:
        for table_str in ("decision_plan", "vplan"):
            connection_obj.execute(f"UPDATE {table_str} SET created_timestamp_str=?", (PINNED_CREATED_TS.isoformat(),))
            connection_obj.execute(f"""UPDATE {table_str} SET updated_timestamp_str=?
                WHERE julianday(updated_timestamp_str)>julianday(?)""",
                (PINNED_UPDATED_TS.isoformat(), (CLOSE_TS + timedelta(days=2)).isoformat()))


def _target(store_obj, release_obj):
    return DashboardPodTarget(release_obj, store_obj.db_path_str, False)


def _read(daily_case, as_of_ts):
    store_obj, release_obj = daily_case[:2]
    _pin(store_obj)
    return load_daily_cycle_dict(_target(store_obj, release_obj), as_of_ts=as_of_ts)


def _status_key(daily_case, as_of_ts, *, data_ready_bool=True):
    daily_dict = _read(daily_case, as_of_ts)
    row_dict = {"mode_str": "paper", "pod_id_str": "daily_pod", "db_status_str": "ok", "as_of_timestamp_str": as_of_ts.isoformat(),
        "norgate_snapshot_status_dict": {"status_str": "ready", "snapshot_fresh_for_cycle_bool": True, "snapshot_date_str": "2026-10-02"}
            if data_ready_bool else {"status_str": "waiting", "snapshot_fresh_for_cycle_bool": False,
                "snapshot_stale_past_alert_deadline_bool": False}, "daily_cycle_dict": daily_dict}
    cycle_dict = build_cycle_view_dict(row_dict, now_ts=as_of_ts, allowed_mode_set=frozenset({"paper"}))
    return cycle_dict["daily_status_dict"]


def _dispatch_attempted(daily_case):
    store_obj, release_obj, decision_obj, plan_obj, _ = daily_case
    store_obj.mark_vplan_status(plan_obj.vplan_id_int, "ready")
    store_obj.mark_decision_plan_status(decision_obj.decision_plan_id_int, "vplan_ready")
    request_list = build_broker_order_request_list_from_vplan(plan_obj)
    assert record_dispatch_failure(store_obj, release_obj, plan_obj, request_list, ValueError("SECRET dispatch text"),
        OPEN_TS - timedelta(minutes=3)) == "opening_dispatch_incomplete"
    return request_list


def test_exact_keys_counts_and_no_free_text(daily_case):
    store_obj, release_obj, decision_obj, plan_obj, _ = daily_case
    request_list = _dispatch_attempted(daily_case)
    completion_obj = BrokerOrderRequest(release_id_str=release_obj.release_id_str, pod_id_str=release_obj.pod_id_str,
        account_route_str=release_obj.account_route_str, submission_key_str="daily:1",
        order_request_key_str="daily:1:daily-completion:MSFT", asset_str="MSFT", broker_order_type_str="MKT",
        order_class_str="MarketOrder", unit_str="shares", amount_float=-5.0, target_bool=False, trade_id_int=None,
        sizing_reference_price_float=100.0, portfolio_value_float=100000.0,
        decision_plan_id_int=decision_obj.decision_plan_id_int, vplan_id_int=plan_obj.vplan_id_int)
    assert claim_daily_completion_request(store_obj, release_obj, decision_obj, plan_obj, completion_obj, INTRADAY_TS)
    with store_obj._connect() as connection_obj:
        connection_obj.execute("UPDATE daily_completion_request SET send_error_str='SECRET send text'")
    assert enqueue_daily_cycle_overdue(store_obj, release_obj, decision_obj, CLOSE_TS + timedelta(hours=1),
        error_str="SECRET overdue text", owned_order_row_list=[{"account_route_str": "DU_TEST", "asset_str": "MSFT",
            "side_str": "SELL", "remaining_amount_float": 5.0, "client_id_int": 4242}])
    daily_dict = _read(daily_case, CLOSE_TS + timedelta(hours=1, minutes=1))
    assert set(daily_dict) == TOP_KEY_SET and daily_dict["status_str"] == "ok"
    assert set(daily_dict["decision_dict"]) == {"decision_plan_id_int", "release_id_str", "status_str", "signal_timestamp_str",
        "submission_timestamp_str", "target_execution_timestamp_str", "signal_session_date_str", "target_session_date_str",
        "updated_timestamp_str"}
    assert set(daily_dict["vplan_dict"]) == {"vplan_id_int", "status_str", "submission_key_str", "submission_timestamp_str",
        "target_execution_timestamp_str", "missing_ack_count_int", "submit_ack_status_str", "sent_order_count_int",
        "ack_count_int", "fill_count_int"}
    assert daily_dict["opening_dispatch_dict"] == {"reason_code_str": "opening_dispatch_incomplete", "severity_str": "critical",
        "observed_timestamp_str": _utc_str(OPEN_TS - timedelta(minutes=3)),
        "dispatch_deadline_timestamp_str": _utc_str(OPEN_TS - timedelta(minutes=2)),
        "attempted_count_int": len(request_list), "never_dispatched_count_int": 0}
    assert daily_dict["completion_request_list"] == [{"asset_str": "MSFT", "vplan_id_int": plan_obj.vplan_id_int,
        "claimed_timestamp_str": _utc_str(INTRADAY_TS), "send_failed_bool": True}]
    alert_dict, = daily_dict["pod_alert_list"]
    assert alert_dict == {"alert_kind_str": "cycle_overdue", "decision_plan_id_int": decision_obj.decision_plan_id_int,
        "created_timestamp_str": _utc_str(CLOSE_TS + timedelta(hours=1)), "delivered_bool": False, "session_date_str": None,
        "session_close_timestamp_str": _utc_str(CLOSE_TS), "order_details_available_bool": True,
        "owned_order_row_list": [{"asset_str": "MSFT", "side_str": "SELL", "quantity_float": 5.0}]}
    assert daily_dict["session_close_timestamp_str"] == _utc_str(CLOSE_TS)
    assert daily_dict["signal_session_dict"]["session_date_str"] == "2026-10-05"
    text_str = json.dumps(daily_dict)
    for withheld_str in ("SECRET", "4242", "client_id", "error", "daily:1:", "ValueError"):
        assert withheld_str not in text_str
    assert _status_key(daily_case, CLOSE_TS + timedelta(hours=1, minutes=1))["key_str"] == "open_after_close"


def test_exception_close_previous_open_and_last_close(daily_case):
    store_obj, release_obj, decision_obj, plan_obj, broker_obj = daily_case
    broker_obj.as_of_ts = CLOSE_TS
    result_obj = reconcile_case(daily_case)
    assert store_obj.complete_daily_cycle(decision_obj.decision_plan_id_int, plan_obj.vplan_id_int, result_obj.broker_snapshot_obj,
        result_obj.status_str, result_obj.exception_list, CLOSE_TS)
    daily_dict = _read(daily_case, CLOSE_TS + timedelta(minutes=5))
    result_dict = daily_dict["execution_result_dict"]
    assert result_dict["status_str"] == "completed_with_exceptions"
    assert result_dict["exception_count_int"] == len(result_obj.exception_list) == len(result_dict["exception_list"])
    assert all(set(row_dict) == {"asset_str", "side_str", "quantity_float", "reason_str", "expected_share_float",
        "actual_share_float", "target_weight_float", "intent_str"} for row_dict in result_dict["exception_list"])
    assert {row_dict["asset_str"] for row_dict in result_dict["exception_list"]} >= {"AAPL", "MSFT"}
    assert daily_dict["execution_alert_list"][0]["alert_kind_str"] == "daily_exception"
    # A newer decision: the closed one becomes the last close, an older open one is listed.
    older_obj = store_obj.insert_decision_plan(replace(decision_obj, decision_plan_id_int=None,
        signal_timestamp_ts=decision_obj.signal_timestamp_ts - timedelta(days=1),
        target_execution_timestamp_ts=OPEN_TS - timedelta(days=3)))
    store_obj.mark_decision_plan_status(older_obj.decision_plan_id_int, "submitted")
    newer_obj = store_obj.insert_decision_plan(replace(decision_obj, decision_plan_id_int=None,
        signal_timestamp_ts=CLOSE_TS, submission_timestamp_ts=OPEN_TS + timedelta(days=1, minutes=-8),
        target_execution_timestamp_ts=OPEN_TS + timedelta(days=1)))
    daily_dict = _read(daily_case, CLOSE_TS + timedelta(minutes=50))
    assert daily_dict["decision_dict"]["decision_plan_id_int"] == newer_obj.decision_plan_id_int
    assert daily_dict["last_closed_dict"] == {"decision_plan_id_int": decision_obj.decision_plan_id_int,
        "status_str": "completed_with_exceptions", "target_session_date_str": "2026-10-05",
        "completed_timestamp_str": _utc_str(CLOSE_TS), "exception_count_int": len(result_obj.exception_list)}
    assert [item_dict["decision_plan_id_int"] for item_dict in daily_dict["open_previous_list"]] == [older_obj.decision_plan_id_int]
    assert daily_dict["open_previous_list"][0]["session_close_timestamp_str"] == "2026-10-02T20:00:00+00:00"
    status_dict = _status_key(daily_case, CLOSE_TS + timedelta(minutes=50))
    # The older cycle is still open long after its close: red, never hidden.
    assert status_dict["key_str"] == "open_after_close"


def test_read_leaves_the_database_byte_unchanged(daily_case, tmp_path):
    store_obj, release_obj = daily_case[:2]
    _pin(store_obj)
    file_dict = {path_obj.name: (path_obj.read_bytes(), path_obj.stat().st_mtime_ns) for path_obj in tmp_path.iterdir()}
    for decision_id_obj in (None, 1):
        assert load_daily_cycle_dict(_target(store_obj, release_obj), as_of_ts=INTRADAY_TS,
            decision_plan_id_int=decision_id_obj)["status_str"] == "ok"
    assert {path_obj.name: (path_obj.read_bytes(), path_obj.stat().st_mtime_ns) for path_obj in tmp_path.iterdir()} == file_dict


def test_older_database_without_optional_tables(daily_case):
    store_obj = daily_case[0]
    with sqlite3.connect(store_obj.db_path_str) as connection_obj:
        for table_str in ("daily_completion_request", "daily_pod_alert", "mr_capsule_execution_alert", "daily_post_close_report"):
            connection_obj.execute(f"DROP TABLE IF EXISTS {table_str}")
    daily_dict = _read(daily_case, INTRADAY_TS)
    assert daily_dict["status_str"] == "ok"
    assert daily_dict["completion_request_list"] == daily_dict["pod_alert_list"] == daily_dict["execution_alert_list"] == []
    assert daily_dict["post_close_report_dict"] == {} and daily_dict["signal_session_dict"]["capsule_halt_bool"] is False
    assert _status_key(daily_case, INTRADAY_TS)["key_str"] == "trading"


def test_non_daily_release_missing_db_and_bad_evidence_stay_unknown(daily_case, tmp_path):
    store_obj, release_obj, decision_obj = daily_case[:3]
    _pin(store_obj)
    monthly_obj = replace(release_obj, strategy_import_str="strategies.example:Monthly")
    assert load_daily_cycle_dict(_target(store_obj, monthly_obj), as_of_ts=INTRADAY_TS) == {"status_str": "unknown"}
    missing_obj = DashboardPodTarget(release_obj, str(tmp_path / "missing.sqlite3"), False)
    assert load_daily_cycle_dict(missing_obj, as_of_ts=INTRADAY_TS)["status_str"] == "unknown"
    assert not (tmp_path / "missing.sqlite3").exists()
    # Saved timestamps after the read time are future evidence, never partly used.
    assert load_daily_cycle_dict(_target(store_obj, release_obj), as_of_ts=PINNED_CREATED_TS - timedelta(seconds=1)) == {
        "status_str": "unknown", "reason_str": "Saved daily cycle could not be verified"}
    with sqlite3.connect(store_obj.db_path_str) as connection_obj:
        connection_obj.execute("UPDATE decision_plan SET snapshot_metadata_json_str='[1]'")
    assert load_daily_cycle_dict(_target(store_obj, release_obj), as_of_ts=INTRADAY_TS)["status_str"] == "unknown"


def test_demo_store_seeds_each_state_with_real_writers_and_pod_reader_attaches_facts():
    store_obj = DemoDailyStore()
    try:
        assert all(target_obj.release_obj.mode_str == "live" and target_obj.release_obj.account_route_str.startswith("DEMO_DU")
            and target_obj.release_obj.pod_id_str.startswith("demo_daily_") for target_obj in store_obj.get_target_list())
        for pod_id_str, expected_str in store_obj.expected_status_dict.items():
            target_obj = store_obj.get_target_for_pod(pod_id_str)
            daily_dict = store_obj.get_daily_cycle_dict({"pod_id_str": pod_id_str,
                "account_route_str": target_obj.release_obj.account_route_str}, as_of_ts=DEMO_NOW_TS)
            row_dict = {"mode_str": "live", "pod_id_str": pod_id_str, "db_status_str": "ok", "as_of_timestamp_str": DEMO_NOW_TS.isoformat(),
                "norgate_snapshot_status_dict": {"status_str": "ready", "snapshot_fresh_for_cycle_bool": True,
                    "snapshot_date_str": "2026-09-04"}, "daily_cycle_dict": daily_dict}
            status_dict = build_cycle_view_dict(row_dict, now_ts=DEMO_NOW_TS)["daily_status_dict"]
            assert status_dict["key_str"] == expected_str, pod_id_str
            source_dict = load_pod_cycles_dict(target_obj, as_of_ts=DEMO_NOW_TS)
            assert source_dict["status_str"] == "ok"
            assert source_dict["pod_row_dict"]["daily_cycle_dict"] == daily_dict
        exceptions_dict = store_obj.get_daily_cycle_dict({"pod_id_str": "demo_daily_exceptions",
            "account_route_str": "DEMO_DU1"}, as_of_ts=DEMO_NOW_TS)
        assert exceptions_dict["last_closed_dict"]["exception_count_int"] == 2
        trading_dict = store_obj.get_daily_cycle_dict({"pod_id_str": "demo_daily_trading", "account_route_str": "DEMO_DU0"},
            as_of_ts=DEMO_NOW_TS)
        assert [row_dict["asset_str"] for row_dict in trading_dict["completion_request_list"]] == ["DBC"]
    finally:
        store_obj.close()
    assert not store_obj.directory_path_obj.exists()


# Watchdog parity: each red heartbeat kind is the same red projection state,
# and one side of each boundary raises neither.
def _assert_parity(monkeypatch, daily_case, tmp_path, as_of_ts, expected_kind_str, *, data_ready_bool=True):
    alert_list = _alerts(monkeypatch, daily_case, tmp_path, as_of_ts)
    status_dict = _status_key(daily_case, as_of_ts, data_ready_bool=data_ready_bool)
    if expected_kind_str is None:
        assert alert_list == [] and not status_dict["red_bool"], (status_dict, alert_list)
    else:
        assert [alert_dict["kind_str"] for alert_dict in alert_list] == [expected_kind_str]
        assert status_dict["key_str"] == WATCHDOG_KIND_TO_STATUS_DICT[expected_kind_str] and status_dict["red_bool"]
    return status_dict


def test_parity_cycle_open_after_close(daily_case, monkeypatch, tmp_path):
    assert _assert_parity(monkeypatch, daily_case, tmp_path, CLOSE_TS + timedelta(minutes=59), None)["key_str"] == "closing"
    _assert_parity(monkeypatch, daily_case, tmp_path, CLOSE_TS + timedelta(minutes=61), "cycle_open_after_close")


def test_parity_cycle_open_after_early_close(daily_case, monkeypatch, tmp_path):
    store_obj, _, decision_obj = daily_case[:3]
    target_ts = OPEN_TS.replace(month=11, day=27)
    with store_obj._connect() as connection_obj:
        connection_obj.execute("UPDATE decision_plan SET target_execution_timestamp_str=? WHERE decision_plan_id_int=?",
            (target_ts.isoformat(), decision_obj.decision_plan_id_int))
    early_close_ts = target_ts.replace(hour=13, minute=0)
    for as_of_ts, open_after_bool in ((early_close_ts + timedelta(minutes=59), False), (early_close_ts + timedelta(minutes=61), True)):
        alert_kind_set = {alert_dict["kind_str"] for alert_dict in _alerts(monkeypatch, daily_case, tmp_path, as_of_ts)}
        assert ("cycle_open_after_close" in alert_kind_set) is open_after_bool
        assert (_status_key(daily_case, as_of_ts)["key_str"] == "open_after_close") is open_after_bool


def test_parity_opening_not_dispatched(daily_case, monkeypatch, tmp_path):
    store_obj, _, decision_obj, plan_obj, _ = daily_case
    store_obj.mark_vplan_status(plan_obj.vplan_id_int, "ready")
    store_obj.mark_decision_plan_status(decision_obj.decision_plan_id_int, "vplan_ready")
    assert _assert_parity(monkeypatch, daily_case, tmp_path, OPEN_TS + timedelta(minutes=4), None)["key_str"] == "ready"
    _assert_parity(monkeypatch, daily_case, tmp_path, OPEN_TS + timedelta(minutes=6), "opening_not_dispatched")


def test_parity_opening_not_dispatched_without_plan(daily_case, monkeypatch, tmp_path):
    store_obj, _, decision_obj, plan_obj, _ = daily_case
    with store_obj._connect() as connection_obj:
        connection_obj.execute("DELETE FROM vplan_row WHERE vplan_id_int=?", (plan_obj.vplan_id_int,))
        connection_obj.execute("DELETE FROM vplan WHERE vplan_id_int=?", (plan_obj.vplan_id_int,))
    store_obj.mark_decision_plan_status(decision_obj.decision_plan_id_int, "planned")
    assert _assert_parity(monkeypatch, daily_case, tmp_path, OPEN_TS + timedelta(minutes=4), None)["key_str"] == "ready"
    _assert_parity(monkeypatch, daily_case, tmp_path, OPEN_TS + timedelta(minutes=6), "opening_not_dispatched")


def test_parity_opening_dispatch_unconfirmed(daily_case, monkeypatch, tmp_path):
    store_obj, _, decision_obj, plan_obj, _ = daily_case
    store_obj.mark_vplan_status(plan_obj.vplan_id_int, "submitting")
    store_obj.mark_decision_plan_status(decision_obj.decision_plan_id_int, "vplan_ready")
    assert _assert_parity(monkeypatch, daily_case, tmp_path, OPEN_TS + timedelta(minutes=4), None)["key_str"] == "sending"
    _assert_parity(monkeypatch, daily_case, tmp_path, OPEN_TS + timedelta(minutes=6), "opening_dispatch_unconfirmed")


def test_parity_decision_incomplete_for_blocked_decision(daily_case, monkeypatch, tmp_path):
    store_obj, _, decision_obj = daily_case[:3]
    store_obj.mark_decision_plan_status(decision_obj.decision_plan_id_int, "blocked")
    _assert_parity(monkeypatch, daily_case, tmp_path, OPEN_TS - timedelta(minutes=2, seconds=1), None)
    status_dict = _assert_parity(monkeypatch, daily_case, tmp_path, OPEN_TS - timedelta(minutes=2), "decision_incomplete")
    assert status_dict["label_str"] == "No decision for 2026-10-02"


def test_parity_decision_incomplete_for_missing_next_decision(daily_case, monkeypatch, tmp_path):
    store_obj, _, decision_obj, plan_obj, broker_obj = daily_case
    broker_obj.as_of_ts = CLOSE_TS
    broker_obj.position_dict.update(plan_obj.target_share_map)
    result_obj = reconcile_case(daily_case)
    assert store_obj.complete_daily_cycle(decision_obj.decision_plan_id_int, plan_obj.vplan_id_int, result_obj.broker_snapshot_obj,
        result_obj.status_str, result_obj.exception_list, CLOSE_TS)
    next_cutoff_ts = OPEN_TS + timedelta(days=1, minutes=-2)
    assert _assert_parity(monkeypatch, daily_case, tmp_path, next_cutoff_ts - timedelta(seconds=1), None)["key_str"] == "deciding"
    assert _status_key(daily_case, next_cutoff_ts - timedelta(seconds=1), data_ready_bool=False)["key_str"] == "closed"
    status_dict = _assert_parity(monkeypatch, daily_case, tmp_path, next_cutoff_ts, "decision_incomplete", data_ready_bool=False)
    assert status_dict["label_str"] == "No decision for 2026-10-05"
