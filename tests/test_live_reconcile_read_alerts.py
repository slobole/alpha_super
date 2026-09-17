from __future__ import annotations

from datetime import UTC, datetime, timedelta
from dataclasses import replace
import json
from pathlib import Path
import socket
from types import SimpleNamespace

import pandas as pd
import pytest

from alpha.live import dashboard, logging_utils, ops_report, runner, scheduler_service, scheduler_utils
from alpha.live import state_store, state_store_v2
from alpha.live.dashboard_v3 import notifications
from alpha.live.ibkr_socket_client import IBKRSocketClient
from alpha.live.models import PodState
from alpha.live.order_clerk import StubBrokerAdapter
from alpha.live.release_manifest import load_release_list
from alpha.live.state_store_v2 import LiveStateStore
from test_live_runner import _insert_ready_vplan_for_release, _write_guardrail_manifest


@pytest.fixture(autouse=True)
def offline_only(monkeypatch):
    def deny_connection(*argument_tuple, **argument_dict):
        raise AssertionError("Reconcile alert tests must never contact a broker or network")

    monkeypatch.setattr(socket.socket, "connect", deny_connection)
    monkeypatch.setattr(socket.socket, "connect_ex", deny_connection)
    monkeypatch.setattr(socket, "create_connection", deny_connection)
    monkeypatch.setattr(IBKRSocketClient, "connect", deny_connection)
    monkeypatch.setenv("ALPHA_USE_NORGATE_SNAPSHOT_BOOL", "0")
    monkeypatch.setattr(scheduler_utils, "load_latest_norgate_heartbeat_session_label_ts",
                        lambda profile_str: pd.Timestamp("2024-01-31"))


@pytest.fixture
def live_case(tmp_path, monkeypatch):
    clock_obj = SimpleNamespace(now_ts=datetime(2024, 2, 1, 14, 22, tzinfo=UTC))

    class ControlledDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return clock_obj.now_ts.astimezone(tz) if tz else clock_obj.now_ts.replace(tzinfo=None)

    monkeypatch.setattr(logging_utils, "datetime", ControlledDateTime)
    monkeypatch.setattr(scheduler_service, "datetime", ControlledDateTime)
    monkeypatch.setattr(state_store, "_utc_now_ts", lambda: clock_obj.now_ts)
    monkeypatch.setattr(state_store_v2, "_utc_now_ts", lambda: clock_obj.now_ts)
    monkeypatch.setattr(scheduler_utils, "utc_now_ts", lambda: clock_obj.now_ts)
    monkeypatch.setattr(ops_report, "utc_now_ts", lambda: clock_obj.now_ts)
    monkeypatch.setattr(logging_utils, "DEFAULT_CRITICAL_LOG_PATH_STR", str(tmp_path / "critical.jsonl"))
    _write_guardrail_manifest(tmp_path, user_id_str="user_001", pod_id_str="pod_test_01",
        release_id_str="alert_test.v1", mode_str="live", enabled_bool=True, account_route_str="U_TEST")
    releases_path = tmp_path / "releases"
    release_obj = load_release_list(str(releases_path))[0]
    db_path = tmp_path / "state.sqlite3"
    store_obj = LiveStateStore(str(db_path))
    store_obj.upsert_release(release_obj)
    store_obj.upsert_pod_state(PodState(
        pod_id_str=release_obj.pod_id_str, user_id_str=release_obj.user_id_str,
        account_route_str=release_obj.account_route_str, position_amount_map={},
        cash_float=10000.0, total_value_float=10000.0, strategy_state_dict={},
        updated_timestamp_ts=datetime(2024, 1, 31, 21, 10, tzinfo=UTC),
        snapshot_stage_str="eod", snapshot_source_str="broker",
    ))
    vplan_obj = _insert_ready_vplan_for_release(store_obj, release_obj)
    broker_obj = StubBrokerAdapter()
    broker_obj.seed_account_snapshot(account_route_str="U_TEST", cash_float=10000.0,
        total_value_float=10000.0, net_liq_float=10000.0, available_funds_float=10000.0,
        excess_liquidity_float=10000.0, position_amount_map={},
        snapshot_timestamp_ts=clock_obj.now_ts, session_mode_str="live")
    broker_obj.seed_live_price_snapshot("U_TEST", {"AAPL": 100.0}, clock_obj.now_ts)
    event_path = tmp_path / "events.jsonl"
    submit_dict = runner.submit_ready_vplans(store_obj, broker_obj, clock_obj.now_ts, "live", False,
        vplan_id_int=vplan_obj.vplan_id_int, log_path_str=str(event_path), trace_enabled_bool=False)
    assert submit_dict["submitted_vplan_count_int"] == 1
    config_path = tmp_path / "dashboard.yaml"
    config_path.write_text(json.dumps({"db_overrides": {"pod_test_01": {"db_path": str(db_path)}}}), encoding="utf-8")
    app_obj = dashboard.DashboardApp(releases_root_path_str=str(releases_path),
        config_path_str=str(config_path), results_root_path_str=str(tmp_path / "results"), event_log_path_str=str(event_path))
    return SimpleNamespace(clock_obj=clock_obj, store_obj=store_obj, broker_obj=broker_obj,
        app_obj=app_obj, event_path=event_path, release_obj=release_obj, vplan_obj=vplan_obj,
        releases_path=releases_path, root_path=tmp_path)


def _summary_dict(case_obj):
    return dashboard.build_dashboard_summary_dict(case_obj.app_obj, as_of_ts=case_obj.clock_obj.now_ts)


def _notify_list(case_obj, summary_dict, delivered_bool, payload_list):
    def capture_delivery(url_str, payload_dict):
        payload_list.append(payload_dict)
        return delivered_bool

    # Recreate the state store each time, as the watchdog does across processes.
    return notifications.check_and_notify_for_red_transitions(summary_dict,
        state_store_obj=notifications.NotificationStateStore(str(case_obj.root_path / "notifications.json")),
        webhook_url_str="https://offline.invalid", webhook_poster_fn=capture_delivery)


def test_real_scheduler_read_failures_alert_retry_restart_and_recover(live_case, monkeypatch):
    case_obj = live_case
    case_obj.clock_obj.now_ts = datetime(2024, 2, 1, 14, 34, 59, tzinfo=UTC)
    decision_obj = scheduler_service.get_scheduler_decision(case_obj.store_obj, case_obj.clock_obj.now_ts,
        str(case_obj.releases_path), "live")
    assert not decision_obj.due_now_bool
    payload_list = []
    assert not _notify_list(case_obj, _summary_dict(case_obj), True, payload_list)
    original_read_fn = case_obj.broker_obj.get_account_snapshot
    failure_obj = RuntimeError("synthetic account read unavailable")

    def fail_read(account_route_str):
        raise failure_obj

    class StopServe(BaseException):
        pass

    sleep_list = []

    def controlled_sleep(seconds_float):
        sleep_list.append(seconds_float)
        case_obj.clock_obj.now_ts += timedelta(seconds=seconds_float)
        if len(sleep_list) == 3:
            raise StopServe()

    monkeypatch.setattr(case_obj.broker_obj, "get_account_snapshot", fail_read)
    monkeypatch.setattr(scheduler_service.time, "sleep", controlled_sleep)
    case_obj.clock_obj.now_ts = datetime(2024, 2, 1, 14, 40, tzinfo=UTC)
    with pytest.raises(StopServe):
        scheduler_service.serve(case_obj.store_obj, case_obj.broker_obj, str(case_obj.releases_path), "live",
            None, None, None, log_path_str=str(case_obj.event_path), trace_enabled_bool=False)
    assert sleep_list == [60.0, 60.0, 60.0]
    event_list = [json.loads(line_str) for line_str in case_obj.event_path.read_text().splitlines()]
    assert sum(event_dict["event_name_str"] == "scheduler_error_retry" for event_dict in event_list) == 3
    assert sum(event_dict["event_name_str"] == "post_execution_reconcile_failed" for event_dict in event_list) == 3
    summary_dict = _summary_dict(case_obj)
    row_dict = summary_dict["pod_row_dict_list"][0]
    assert row_dict["latest_vplan_status_str"] == "submitted"
    assert row_dict["exception_count_int"] == 0  # No invented holdings mismatch.
    assert row_dict["health_str"] == "red"
    assert row_dict["required_action_dict"]["label_str"] == "Review reconcile read"
    assert row_dict["lifecycle_step_dict_list"][5]["status_str"] == "read_failed"
    assert row_dict["debug_summary_dict"]["verdict_label_str"] == "Reconcile read failed"
    assert summary_dict["inspector_report_dict"]["overall_severity_str"] == "red"
    assert {record_obj.pod_id_str for record_obj in _notify_list(case_obj, summary_dict, False, payload_list)} == {"pod_test_01", "__inspector__"}
    # A new dashboard instance still sees persisted failure evidence.
    case_obj.app_obj = dashboard.DashboardApp(**{
        field_str: getattr(case_obj.app_obj, field_str) for field_str in
        ("releases_root_path_str", "config_path_str", "results_root_path_str", "event_log_path_str")
    })
    assert len(_notify_list(case_obj, _summary_dict(case_obj), True, payload_list)) == 2
    assert not _notify_list(case_obj, _summary_dict(case_obj), True, payload_list)
    monkeypatch.setattr(case_obj.broker_obj, "get_account_snapshot", original_read_fn)
    case_obj.clock_obj.now_ts += timedelta(minutes=1)
    reconcile_dict = runner.post_execution_reconcile(case_obj.store_obj, case_obj.broker_obj,
        case_obj.clock_obj.now_ts, "live", log_path_str=str(case_obj.event_path), trace_enabled_bool=False)
    assert reconcile_dict["completed_vplan_count_int"] == 1
    recovered_dict = _summary_dict(case_obj)
    assert recovered_dict["pod_row_dict_list"][0]["reconcile_read_failure_dict"] is None
    assert recovered_dict["pod_row_dict_list"][0]["health_str"] == "green"
    assert not _notify_list(case_obj, recovered_dict, True, payload_list)


@pytest.mark.parametrize("method_str", ["get_adapter", "get_account_snapshot", "get_recent_order_state_snapshot", "get_session_open_price_list"])
def test_each_broker_read_preserves_original_exception_and_plan(live_case, monkeypatch, method_str):
    case_obj = live_case
    case_obj.clock_obj.now_ts = datetime(2024, 2, 1, 14, 40, tzinfo=UTC)
    failure_obj = RuntimeError(method_str)

    def fail_read(*argument_tuple, **argument_dict):
        raise failure_obj

    target_obj = runner.BrokerAdapterResolver if method_str == "get_adapter" else case_obj.broker_obj
    monkeypatch.setattr(target_obj, method_str, fail_read)
    with pytest.raises(RuntimeError) as exception_info:
        runner.post_execution_reconcile(case_obj.store_obj, case_obj.broker_obj, case_obj.clock_obj.now_ts,
            "live", log_path_str=str(case_obj.event_path), trace_enabled_bool=False)
    assert exception_info.value is failure_obj
    assert case_obj.store_obj.get_latest_vplan_for_pod("pod_test_01").status_str == "submitted"
    assert _summary_dict(case_obj)["pod_row_dict_list"][0]["reconcile_read_failure_dict"]["error_str"] == method_str


def test_failed_log_write_does_not_replace_broker_exception(live_case, monkeypatch):
    failure_obj = RuntimeError("original broker failure")

    def fail_read(*argument_tuple, **argument_dict):
        raise failure_obj

    def fail_log(*argument_tuple, **argument_dict):
        raise OSError("disk unavailable")

    monkeypatch.setattr(live_case.broker_obj, "get_account_snapshot", fail_read)
    monkeypatch.setattr(runner, "log_event", fail_log)
    with pytest.raises(RuntimeError) as exception_info:
        runner.post_execution_reconcile(live_case.store_obj, live_case.broker_obj,
            datetime(2024, 2, 1, 14, 40, tzinfo=UTC), "live",
            log_path_str=str(live_case.event_path), trace_enabled_bool=False)
    assert exception_info.value is failure_obj


def test_read_recovery_with_position_mismatch_still_blocks_and_alerts(live_case, monkeypatch):
    case_obj = live_case
    case_obj.clock_obj.now_ts = datetime(2024, 2, 1, 14, 40, tzinfo=UTC)
    snapshot_obj = case_obj.broker_obj.get_account_snapshot("U_TEST")

    def fail_read(account_route_str):
        raise RuntimeError("temporary read failure")

    monkeypatch.setattr(case_obj.broker_obj, "get_account_snapshot", fail_read)
    with pytest.raises(RuntimeError, match="temporary read failure"):
        runner.post_execution_reconcile(case_obj.store_obj, case_obj.broker_obj, case_obj.clock_obj.now_ts,
            "live", log_path_str=str(case_obj.event_path), trace_enabled_bool=False)
    assert _summary_dict(case_obj)["pod_row_dict_list"][0]["reconcile_read_failure_dict"]
    case_obj.clock_obj.now_ts += timedelta(minutes=1)
    # The account can now be read, but the planned ten shares are still absent.
    monkeypatch.setattr(case_obj.broker_obj, "get_account_snapshot", lambda account_route_str:
        replace(snapshot_obj, position_amount_map={}, snapshot_timestamp_ts=case_obj.clock_obj.now_ts))
    result_dict = runner.post_execution_reconcile(case_obj.store_obj, case_obj.broker_obj, case_obj.clock_obj.now_ts,
        "live", log_path_str=str(case_obj.event_path), trace_enabled_bool=False)
    assert result_dict["completed_vplan_count_int"] == 0
    summary_dict = _summary_dict(case_obj)
    row_dict = summary_dict["pod_row_dict_list"][0]
    assert row_dict["reconcile_read_failure_dict"] is None
    assert row_dict["latest_vplan_status_str"] == "submitted"
    assert row_dict["exception_count_int"] > 0
    assert row_dict["health_str"] == "red"
    assert row_dict["debug_summary_dict"]["verdict_label_str"] == "Reconcile blocked"
    assert summary_dict["inspector_report_dict"]["overall_severity_str"] == "red"
    assert len(_notify_list(case_obj, summary_dict, True, [])) == 2


def _failure_input_tuple():
    row_dict = {
        "mode_str": "live", "pod_id_str": "pod1", "account_route_str": "U_TEST",
        "latest_decision_release_id_str": "release1", "latest_decision_plan_id_int": 1,
        "latest_vplan_id_int": 2, "latest_vplan_decision_plan_id_int": 1,
        "latest_vplan_status_str": "submitted", "latest_vplan_submission_timestamp_str": "2024-02-01T14:22:00+00:00",
        "latest_vplan_target_execution_timestamp_str": "2024-02-01T14:30:00+00:00",
    }
    event_dict = {
        "event_name_str": "post_execution_reconcile_failed", "mode_str": "live", "pod_id_str": "pod1",
        "account_route_str": "U_TEST", "release_id_str": "release1", "decision_plan_id_int": 1, "vplan_id_int": 2,
        "submission_timestamp_str": row_dict["latest_vplan_submission_timestamp_str"],
        "target_execution_timestamp_str": row_dict["latest_vplan_target_execution_timestamp_str"],
        "event_timestamp_str": "2024-02-01T14:40:00+00:00", "error_str": "read unavailable",
    }
    return row_dict, event_dict


@pytest.mark.parametrize("key_str,value_obj", [
    ("pod_id_str", "other"), ("account_route_str", "U_OTHER"), ("release_id_str", "other"),
    ("decision_plan_id_int", 9), ("vplan_id_int", 9), ("mode_str", "paper"),
    ("submission_timestamp_str", "2024-01-31T14:22:00+00:00"),
    ("target_execution_timestamp_str", "2024-01-31T14:30:00+00:00"),
    ("event_timestamp_str", "2024-02-01T15:00:00+00:00"),
    ("event_timestamp_str", "2024-02-01T14:20:00+00:00"),
    ("event_timestamp_str", "not a timestamp"),
    ("event_name_str", "scheduler_error_retry"),
])
def test_unrelated_stale_or_future_failure_does_not_alert(tmp_path, key_str, value_obj):
    row_dict, event_dict = _failure_input_tuple()
    event_dict[key_str] = value_obj
    log_path = tmp_path / "events.jsonl"
    log_path.write_text(json.dumps(event_dict) + "\n", encoding="utf-8")
    assert dashboard._active_reconcile_read_failure_dict(row_dict, None, str(log_path),
        datetime(2024, 2, 1, 14, 45, tzinfo=UTC)) is None


@pytest.mark.parametrize("row_change_dict", [
    {"mode_str": "incubation"}, {"mode_str": "paper"}, {"latest_vplan_status_str": "completed"},
    {"latest_vplan_status_str": "ready"}, {"latest_decision_plan_id_int": 3},
])
def test_finished_obsolete_and_non_live_cycles_do_not_inherit_failure(tmp_path, row_change_dict):
    row_dict, event_dict = _failure_input_tuple()
    row_dict.update(row_change_dict)
    log_path = tmp_path / "events.jsonl"
    log_path.write_text(json.dumps(event_dict) + "\n", encoding="utf-8")
    assert dashboard._active_reconcile_read_failure_dict(row_dict, None, str(log_path),
        datetime(2024, 2, 1, 14, 45, tzinfo=UTC)) is None


@pytest.mark.parametrize("stage_str,vplan_id_int,created_str,active_bool", [
    ("post_execution", 2, "2024-02-01T09:41:00-05:00", False),
    ("post_execution", 2, "2024-02-01T14:39:00+00:00", True),
    ("pre_vplan", 2, "2024-02-01T14:41:00+00:00", True),
    ("post_execution", 99, "2024-02-01T14:41:00+00:00", True),
])
def test_only_newer_matching_observation_clears_failure_in_rotated_log(
    tmp_path, stage_str, vplan_id_int, created_str, active_bool,
):
    row_dict, event_dict = _failure_input_tuple()
    log_path = tmp_path / "events.jsonl"
    log_path.with_name("events.jsonl.1").write_text(json.dumps(event_dict) + "\n", encoding="utf-8")
    # Noise, restarts and malformed unrelated lines cannot clear active evidence.
    log_path.write_text('broken JSON\n' + '{"event_name_str":"scheduler_started"}\n' * 300, encoding="utf-8")
    reconciliation_dict = {"stage_str": stage_str, "vplan_id_int": vplan_id_int, "created_timestamp_str": created_str}
    assert bool(dashboard._active_reconcile_read_failure_dict(row_dict, reconciliation_dict, str(log_path),
        datetime(2024, 2, 1, 14, 45, tzinfo=UTC))) is active_bool


@pytest.mark.parametrize("marker_str", ["build_vplan_created", "submit_vplan_completed"])
def test_healthy_cycle_scan_stops_at_its_own_creation_or_submission(tmp_path, monkeypatch, marker_str):
    row_dict, event_dict = _failure_input_tuple()
    event_dict.update(event_name_str=marker_str, event_timestamp_str="2024-02-01T14:22:00+00:00")

    def reverse_events(path_obj):
        yield {**event_dict, "pod_id_str": "other"}
        yield event_dict
        raise AssertionError("Do not parse unrelated retained history before this cycle")

    monkeypatch.setattr(dashboard, "_iter_event_dict_reverse", reverse_events)
    assert dashboard._active_reconcile_read_failure_dict(row_dict, None, str(tmp_path / "events.jsonl"),
        datetime(2024, 2, 1, 14, 45, tzinfo=UTC)) is None


def test_before_execution_does_not_scan_logs(tmp_path, monkeypatch):
    row_dict, event_dict = _failure_input_tuple()

    def reject_scan(path_obj):
        raise AssertionError("No reconciliation read is due before execution")

    monkeypatch.setattr(dashboard, "_event_log_path_obj_list", reject_scan)
    assert dashboard._active_reconcile_read_failure_dict(row_dict, None, str(tmp_path / "events.jsonl"),
        datetime(2024, 2, 1, 14, 25, tzinfo=UTC)) is None
