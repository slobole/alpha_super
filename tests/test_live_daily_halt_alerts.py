"""Daily-only durable overdue and pre-decision holding-halt notifications."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from dataclasses import replace
from datetime import timedelta
import json
import sqlite3
from threading import Event
from types import SimpleNamespace

import pandas as pd
import pytest

from alpha.live import daily_notifications as alert_module, mr_capsule_adapter, scheduler_service
from alpha.live import runner as runner_module
from alpha.live.execution_engine import build_broker_order_request_list_from_vplan
from alpha.live.models import BrokerSnapshot, DecisionPlan
from alpha.live.mr_capsule_notifications import enqueue_daily_exception_alert
from alpha.live.state_store_v2 import LiveStateStore
from scripts import live_ops_watchdog as watchdog_module
from test_live_daily_reconcile import CLOSE_TS, MARKET_ZONE_OBJ, daily_case, open_row, reconcile_case
from test_live_mr_capsule_host import AS_OF_TS, SIGNAL_DATE_TS, _release, _state


def _row_list(store_obj):
    with store_obj._connect() as connection_obj:
        return [dict(row_obj) for row_obj in connection_obj.execute("SELECT * FROM daily_pod_alert")]


def _summary_dict(store_obj, release_obj):
    return {"pod_row_dict_list": [{"db_path_str": store_obj.db_path_str,
        "pod_id_str": release_obj.pod_id_str, "account_route_str": release_obj.account_route_str,
        "mode_str": release_obj.mode_str, "strategy_import_str": release_obj.strategy_import_str}]}


def test_new_alert_schema_does_not_modify_existing_tables_and_joins_transaction(tmp_path):
    with closing(sqlite3.connect(tmp_path / "schema.sqlite3")) as connection_obj:
        connection_obj.execute("CREATE TABLE legacy_state (pod TEXT PRIMARY KEY, value TEXT)")
        connection_obj.execute("INSERT INTO legacy_state VALUES ('NDX', 'unchanged')")
        connection_obj.commit()
        original_schema_str = connection_obj.execute("SELECT sql FROM sqlite_master WHERE name='legacy_state'").fetchone()[0]
        connection_obj.execute("BEGIN IMMEDIATE")
        alert_module.ensure_daily_alert_schema(connection_obj)
        connection_obj.rollback()
        assert not connection_obj.execute("SELECT 1 FROM sqlite_master WHERE name='daily_pod_alert'").fetchone()
        alert_module.ensure_daily_alert_schema(connection_obj)
        alert_module.ensure_daily_alert_schema(connection_obj)
        assert connection_obj.execute("SELECT sql FROM sqlite_master WHERE name='legacy_state'").fetchone()[0] == original_schema_str
        assert connection_obj.execute("SELECT * FROM legacy_state").fetchall() == [("NDX", "unchanged")]


def test_owned_cancel_failure_alerts_once_at_close_plus_one_hour_with_full_order_details(daily_case):
    store_obj, release_obj, decision_obj, plan_obj, broker_obj = daily_case
    request_obj = build_broker_order_request_list_from_vplan(plan_obj)[0]
    broker_obj.open_row_list = [open_row(request_obj.asset_str, request_obj.order_request_key_str, 77),
        open_row("FOREIGN", "manual-order", 99)]
    broker_obj.cancel_confirmed_bool = False
    broker_obj.as_of_ts = CLOSE_TS + timedelta(hours=1, seconds=-1)
    with pytest.raises(RuntimeError, match="waiting for confirmation"):
        reconcile_case(daily_case)
    assert _row_list(store_obj) == []
    broker_obj.as_of_ts += timedelta(seconds=1)
    for _attempt_int in range(2):
        with pytest.raises(RuntimeError, match="waiting for confirmation"):
            reconcile_case(daily_case)
    assert not alert_module.enqueue_daily_cycle_overdue(LiveStateStore(store_obj.db_path_str), release_obj,
        decision_obj, broker_obj.as_of_ts, error_str="fallback without details")
    row_dict, = _row_list(store_obj)
    payload_dict = json.loads(row_dict["payload_json_str"])
    assert payload_dict["owned_order_row_list"] == [{"asset_str": request_obj.asset_str,
        "quantity_float": 100.0, "side_str": "SELL", "client_id_int": 77}]
    assert "FOREIGN" not in row_dict["payload_json_str"]
    delivery_list = alert_module.deliver_daily_alerts(_summary_dict(store_obj, release_obj), webhook_url_str="fake",
        webhook_poster_fn=lambda _url_str, message_dict: all(value_str in message_dict["content"] for value_str in
            ("CRITICAL", request_obj.asset_str, "quantity=100", "side=SELL", "client_id=77", "Required action")),
        now_ts=broker_obj.as_of_ts)
    assert len(delivery_list) == 1 and delivery_list[0].delivered_bool
    assert alert_module.deliver_daily_alerts(_summary_dict(store_obj, release_obj), webhook_url_str="fake",
        webhook_poster_fn=lambda *_args: pytest.fail("duplicate delivery"), now_ts=broker_obj.as_of_ts) == []


def test_overdue_threshold_uses_early_exchange_close(daily_case):
    store_obj, release_obj, decision_obj, _, _ = daily_case
    target_ts = pd.Timestamp("2024-11-29 09:30", tz=MARKET_ZONE_OBJ).to_pydatetime()
    early_obj = store_obj.insert_decision_plan(replace(decision_obj, decision_plan_id_int=None,
        signal_timestamp_ts=target_ts - timedelta(days=2), target_execution_timestamp_ts=target_ts))
    alert_ts = target_ts.replace(hour=14, minute=0)
    assert not alert_module.enqueue_daily_cycle_overdue(store_obj, release_obj, early_obj,
        alert_ts - timedelta(seconds=1), error_str="unreachable")
    assert alert_module.enqueue_daily_cycle_overdue(store_obj, release_obj, early_obj, alert_ts, error_str="unreachable")


@pytest.mark.parametrize("terminal_str", ["completed", "completed_with_exceptions", "superseded"])
def test_completed_or_superseded_cycle_cannot_enqueue_an_overdue_alert(daily_case, terminal_str):
    store_obj, release_obj, decision_obj, _, _ = daily_case
    store_obj.mark_decision_plan_status(decision_obj.decision_plan_id_int, terminal_str)
    assert not alert_module.enqueue_daily_cycle_overdue(store_obj, release_obj, decision_obj,
        CLOSE_TS + timedelta(hours=1), error_str="stale worker")
    assert _row_list(store_obj) == []


def test_missing_broker_details_stay_unknown_and_resolved_cycle_is_labeled_historical(daily_case):
    store_obj, release_obj, decision_obj, _, _ = daily_case
    alert_ts = CLOSE_TS + timedelta(hours=1)
    assert alert_module.enqueue_daily_cycle_overdue(store_obj, release_obj, decision_obj, alert_ts,
        error_str="connection failed")
    summary_dict = _summary_dict(store_obj, release_obj)
    assert alert_module.pending_daily_alert_count_int(summary_dict) == 1
    assert not alert_module.deliver_daily_alerts(summary_dict, webhook_url_str="fake",
        webhook_poster_fn=lambda *_args: False, now_ts=alert_ts)[0].delivered_bool
    store_obj.mark_decision_plan_status(decision_obj.decision_plan_id_int, "completed")
    message_list = []
    alert_module.deliver_daily_alerts(summary_dict, webhook_url_str="fake",
        webhook_poster_fn=lambda _url_str, payload_dict: message_list.append(payload_dict) or True, now_ts=alert_ts)
    assert "HISTORICAL" in message_list[0]["content"]
    assert "details unavailable" in message_list[0]["content"]
    assert alert_module.pending_daily_alert_count_int(summary_dict) == 0
    assert _row_list(store_obj)[0]["attempt_count_int"] == 2


def test_parallel_watchdogs_claim_once_and_failed_http_releases_claim(daily_case):
    store_obj, release_obj, decision_obj, _, _ = daily_case
    alert_ts = CLOSE_TS + timedelta(hours=1)
    alert_module.enqueue_daily_cycle_overdue(store_obj, release_obj, decision_obj, alert_ts, error_str="unreachable")
    summary_dict = _summary_dict(store_obj, release_obj)
    sending_event_obj, release_event_obj = Event(), Event()
    def blocking_send(_url_str, _payload_dict):
        sending_event_obj.set()
        assert release_event_obj.wait(timeout=10)
        return True
    with ThreadPoolExecutor(max_workers=2) as executor_obj:
        future_obj = executor_obj.submit(alert_module.deliver_daily_alerts, summary_dict, webhook_url_str="fake",
            webhook_poster_fn=blocking_send, now_ts=alert_ts)
        try:
            assert sending_event_obj.wait(timeout=10)
            assert alert_module.deliver_daily_alerts(summary_dict, webhook_url_str="fake",
                webhook_poster_fn=lambda *_args: pytest.fail("parallel duplicate"), now_ts=alert_ts) == []
        finally:
            release_event_obj.set()
        assert future_obj.result()[0].delivered_bool


def test_crashed_delivery_claim_survives_restart_and_retries_only_after_lease_expiry(daily_case):
    store_obj, release_obj, decision_obj, _, _ = daily_case
    alert_ts = CLOSE_TS + timedelta(hours=1)
    alert_module.enqueue_daily_cycle_overdue(store_obj, release_obj, decision_obj, alert_ts, error_str="unreachable")
    with store_obj._connect() as connection_obj:
        connection_obj.execute("UPDATE daily_pod_alert SET delivery_claim_str='crashed-worker', "
            "delivery_claimed_timestamp_str=?,attempt_count_int=1", (alert_module._timestamp_str(alert_ts),))
    restarted_store_obj = LiveStateStore(store_obj.db_path_str)
    summary_dict = _summary_dict(restarted_store_obj, release_obj)
    assert alert_module.deliver_daily_alerts(summary_dict, webhook_url_str="fake",
        webhook_poster_fn=lambda *_args: pytest.fail("unexpired lease"), now_ts=alert_ts + timedelta(seconds=299)) == []
    assert alert_module.deliver_daily_alerts(summary_dict, webhook_url_str="fake",
        webhook_poster_fn=lambda *_args: True, now_ts=alert_ts + timedelta(seconds=300))[0].delivered_bool
    row_dict, = _row_list(restarted_store_obj)
    assert row_dict["attempt_count_int"] == 2 and row_dict["delivery_claim_str"] is None


@pytest.mark.parametrize("parking_str,position_dict,identity_bool,expected_asset_set", [
    ("bil", {"AAA": 7.0}, False, {"AAA"}),
    ("cash", {"BIL": 4.0, "SPMO": 2.0}, True, {"BIL", "SPMO"}),
    ("bil", {"SPMO": 9.0, "AAA": 3.0}, True, {"SPMO"}),
])
def test_capsule_holding_halt_has_details_and_one_alert_per_signal_session(tmp_path,
        parking_str, position_dict, identity_bool, expected_asset_set):
    release_obj = _release(mode_str=parking_str)
    state_obj = _state(release_obj, position_dict)
    if not identity_bool:
        state_obj = replace(state_obj, strategy_state_dict={})
    with pytest.raises(mr_capsule_adapter.CapsuleHoldingMismatchError) as error_info_obj:
        mr_capsule_adapter._validate_account_state(release_obj, state_obj, SIGNAL_DATE_TS, AS_OF_TS)
    error_obj = error_info_obj.value
    assert {row_dict["asset_str"] for row_dict in error_obj.holding_row_list} == expected_asset_set
    store_obj = LiveStateStore(str(tmp_path / "halt.sqlite3"))
    store_obj.upsert_release(release_obj)
    for _attempt_int in range(2):
        alert_module.enqueue_capsule_holding_halt(store_obj, release_obj, SIGNAL_DATE_TS, error_obj, AS_OF_TS)
    assert not alert_module.enqueue_capsule_holding_halt(store_obj,
        replace(release_obj, release_id_str="same_pod_new_release"), SIGNAL_DATE_TS, error_obj, AS_OF_TS)
    assert len(_row_list(store_obj)) == 1
    assert store_obj.get_latest_decision_plan_for_pod(release_obj.pod_id_str) is None
    row_dict, = _row_list(store_obj)
    message_str = alert_module._payload_list(row_dict, None)[0]["content"]
    assert "HALTED" in message_str and "Required action" in message_str
    assert all(asset_str in message_str for asset_str in expected_asset_set)
    assert alert_module.enqueue_capsule_holding_halt(LiveStateStore(store_obj.db_path_str), release_obj,
        SIGNAL_DATE_TS + pd.Timedelta(days=1), error_obj, AS_OF_TS + timedelta(days=1))


def test_watchdog_delivers_daily_halt_and_existing_exception_together(daily_case, monkeypatch):
    store_obj, release_obj, decision_obj, plan_obj, _ = daily_case
    alert_module.enqueue_daily_cycle_overdue(store_obj, release_obj, decision_obj, CLOSE_TS + timedelta(hours=1), error_str="unreachable")
    with store_obj._connect() as connection_obj:
        enqueue_daily_exception_alert(connection_obj, decision_plan_id_int=decision_obj.decision_plan_id_int,
            vplan_id_int=plan_obj.vplan_id_int, pod_id_str=release_obj.pod_id_str, account_route_str=release_obj.account_route_str,
            mode_str=release_obj.mode_str, exception_list=[{"asset_str": "AAPL", "quantity_float": 2.0,
                "side_str": "BUY", "reason_str": "missed"}], created_timestamp_ts=CLOSE_TS)
    message_list = []
    monkeypatch.setattr(watchdog_module.notifications_module, "post_discord_webhook_bool",
        lambda _url_str, payload_dict: message_list.append(payload_dict) or True)
    result_dict, invalid_list = watchdog_module._run_capsule_notifications_tuple(_summary_dict(store_obj, release_obj), "paper", "fake")
    assert invalid_list == []
    assert result_dict["capsule_notification_attempt_count_int"] == 2
    assert result_dict["capsule_notification_pending_count_int"] == 0
    assert len(message_list) == 2


def test_legacy_watchdog_scope_does_not_call_daily_alerts(daily_case, monkeypatch):
    store_obj, release_obj, _, _, _ = daily_case
    legacy_obj = replace(release_obj, strategy_import_str="strategies.taa.strategy_taa")
    monkeypatch.setattr(alert_module, "deliver_daily_alerts", lambda *_args, **_kwargs: pytest.fail("legacy delivery changed"))
    assert watchdog_module._run_capsule_notifications_tuple(_summary_dict(store_obj, legacy_obj), "paper", "fake") == ({}, [])


def test_scheduler_keeps_daily_polling_quiet_and_legacy_reconcile_alert_unchanged(daily_case):
    _, release_obj, _, plan_obj, _ = daily_case
    submitted_obj = replace(plan_obj, status_str="submitted")
    store_obj = SimpleNamespace(get_enabled_release_list=lambda: [release_obj],
        get_latest_vplan_for_pod=lambda _pod_str: submitted_obj,
        has_post_execution_reconciliation_snapshot=lambda _plan_int: False)
    for as_of_ts in [CLOSE_TS - timedelta(hours=4), CLOSE_TS - timedelta(seconds=1)]:
        assert scheduler_service._build_stuck_operator_message_spec_list(state_store_obj=store_obj,
            as_of_ts=as_of_ts, reconcile_grace_seconds_int=300) == []
    for as_of_ts in [CLOSE_TS, CLOSE_TS + timedelta(hours=2)]:
        daily_message_dict, = scheduler_service._build_stuck_operator_message_spec_list(state_store_obj=store_obj,
            as_of_ts=as_of_ts, reconcile_grace_seconds_int=300)
        assert daily_message_dict["phase_action_str"] == "reconcile.stuck"
        assert daily_message_dict["level_str"] == "CRITICAL"
    legacy_obj = replace(release_obj, strategy_import_str="strategies.ndx.strategy_ndx")
    store_obj.get_enabled_release_list = lambda: [legacy_obj]
    message_dict, = scheduler_service._build_stuck_operator_message_spec_list(state_store_obj=store_obj,
        as_of_ts=CLOSE_TS, reconcile_grace_seconds_int=300)
    assert message_dict["phase_action_str"] == "reconcile.stuck"
    assert message_dict["level_str"] == "CRITICAL"
    assert message_dict["field_map_dict"]["reason"] == "no post-execution reconcile snapshot recorded"


@pytest.mark.parametrize("mismatch_str", ["foreign", "wrong_parking", "unrelated_invalid_snapshot"])
def test_runner_decision_holding_halt_is_durable_without_creating_a_plan(tmp_path, monkeypatch, mismatch_str):
    release_obj = replace(_release(), enabled_bool=True, params_dict={"margin_account_confirmed_bool": True})
    state_obj = _state(release_obj, {"SPMO": 4.0} if mismatch_str == "wrong_parking" else {"AAA": 7.0})
    if mismatch_str == "foreign":
        state_obj = replace(state_obj, strategy_state_dict={})
    elif mismatch_str == "unrelated_invalid_snapshot":
        state_obj = replace(state_obj, updated_timestamp_ts=AS_OF_TS - timedelta(days=1))
    store_obj = LiveStateStore(str(tmp_path / "decision.sqlite3"))
    store_obj.upsert_release(release_obj)
    store_obj.upsert_pod_state(state_obj, snapshot_stage_str="eod", snapshot_source_str="broker")
    monkeypatch.setattr(runner_module, "_load_release_list_validate_and_sync", lambda *_args, **_kwargs: [release_obj])
    monkeypatch.setattr(runner_module.scheduler_utils, "select_due_release_list", lambda release_list, _as_of_ts: release_list)
    for _attempt_int in range(2):
        result_dict = runner_module.build_decision_plans(store_obj, AS_OF_TS, str(tmp_path),
            auto_sync_norgate_snapshots_bool=False, trace_enabled_bool=False, log_path_str=str(tmp_path / "decision.log"))
        assert result_dict["created_decision_plan_count_int"] == 0
    assert len(_row_list(store_obj)) == int(mismatch_str != "unrelated_invalid_snapshot")
    assert store_obj.get_latest_decision_plan_for_pod(release_obj.pod_id_str) is None


@pytest.mark.parametrize("failure_str", ["resolution", "refresh"])
def test_runner_overdue_fallback_waits_one_hour_and_retries_missing_broker_data(daily_case, tmp_path, failure_str):
    store_obj, release_obj, decision_obj, _, broker_obj = daily_case
    broker_obj.error_obj = ConnectionError("synthetic disconnected broker")
    def resolve_fn(_release_obj):
        if failure_str == "resolution":
            raise ConnectionError("synthetic adapter resolution failed")
        return broker_obj
    resolver_obj = SimpleNamespace(get_adapter=resolve_fn)
    for elapsed_seconds_int in [-3600, 3599, 3600, 3601]:
        as_of_ts = CLOSE_TS + timedelta(seconds=elapsed_seconds_int)
        assert runner_module._reconcile_daily_cycles(store_obj, resolver_obj, as_of_ts, release_obj.mode_str,
            release_obj.pod_id_str, str(tmp_path / "reconcile.log"), False, str(tmp_path / "trace")) == 0
        assert len(_row_list(store_obj)) == int(elapsed_seconds_int >= 3600)
        assert store_obj.get_decision_plan_by_id(decision_obj.decision_plan_id_int).status_str == "submitted"
    assert broker_obj.sent_request_list == []


@pytest.mark.parametrize("legacy_strategy_str", [
    "strategies.momentum.strategy_mo_atr_normalized_ndx:AtrNormalizedNdxStrategy",
    "strategies.taa_df.strategy_taa_df_btal_fallback_tqqq_vix_cash",
])
def test_holding_alert_sqlite_failure_keeps_capsule_blocked_and_builds_next_legacy_pod(
        tmp_path, monkeypatch, legacy_strategy_str):
    capsule_release_obj = replace(_release(), enabled_bool=True,
        params_dict={"margin_account_confirmed_bool": True})
    legacy_release_obj = replace(capsule_release_obj, release_id_str="legacy.v1", pod_id_str="legacy_pod",
        account_route_str="DU_LEGACY", strategy_import_str=legacy_strategy_str)
    release_list = [capsule_release_obj, legacy_release_obj]
    store_obj = LiveStateStore(str(tmp_path / "holding-alert-failure.sqlite3"))
    for release_obj in release_list:
        store_obj.upsert_release(release_obj)
    store_obj.upsert_pod_state(replace(_state(capsule_release_obj, {"AAA": 7.0}), strategy_state_dict={}),
        snapshot_stage_str="eod", snapshot_source_str="broker")
    with store_obj._connect() as connection_obj:
        connection_obj.execute("CREATE TRIGGER fail_alert BEFORE INSERT ON daily_pod_alert "
            "BEGIN SELECT RAISE(ABORT,'synthetic alert database failure'); END")
    original_build_fn = runner_module.strategy_host.build_decision_plan_for_release
    visited_pod_list = []
    def build_fn(*, release_obj, as_of_ts, pod_state_obj):
        visited_pod_list.append(release_obj.pod_id_str)
        if release_obj.pod_id_str == capsule_release_obj.pod_id_str:
            return original_build_fn(release_obj=release_obj, as_of_ts=as_of_ts, pod_state_obj=pod_state_obj)
        return DecisionPlan(release_obj.release_id_str, release_obj.user_id_str, release_obj.pod_id_str,
            release_obj.account_route_str, AS_OF_TS, AS_OF_TS + timedelta(days=4), AS_OF_TS + timedelta(days=4),
            release_obj.execution_policy_str, {}, {}, {})
    monkeypatch.setattr(runner_module.strategy_host, "build_decision_plan_for_release", build_fn)
    monkeypatch.setattr(runner_module, "_load_release_list_validate_and_sync", lambda *_args, **_kwargs: release_list)
    monkeypatch.setattr(runner_module.scheduler_utils, "select_due_release_list", lambda release_list, _as_of_ts: release_list)
    log_path_obj = tmp_path / "decision.log"
    result_dict = runner_module.build_decision_plans(store_obj, AS_OF_TS, str(tmp_path),
        auto_sync_norgate_snapshots_bool=False, trace_enabled_bool=False, log_path_str=str(log_path_obj))
    assert result_dict["created_decision_plan_count_int"] == 1
    assert result_dict["skipped_decision_plan_count_int"] == 1
    assert result_dict["reason_count_map_dict"] == {"mr_capsule_decision_blocked": 1}
    assert visited_pod_list == [release_obj.pod_id_str for release_obj in release_list]
    assert store_obj.get_latest_decision_plan_for_pod(capsule_release_obj.pod_id_str) is None
    assert store_obj.get_latest_decision_plan_for_pod(legacy_release_obj.pod_id_str).status_str == "planned"
    assert _row_list(store_obj) == []
    log_str = log_path_obj.read_text(encoding="utf-8")
    assert "daily_alert_enqueue_failed" in log_str and "synthetic alert database failure" in log_str
    assert "mr_capsule_decision_blocked" in log_str


@pytest.mark.parametrize("failure_str", ["resolution", "refresh"])
def test_overdue_alert_sqlite_failure_leaves_daily_retryable_and_completes_legacy_pod(
        daily_case, tmp_path, monkeypatch, failure_str):
    store_obj, release_obj, decision_obj, plan_obj, broker_obj = daily_case
    legacy_release_obj = replace(release_obj, release_id_str="legacy.v1", pod_id_str="legacy_pod",
        account_route_str="DU_LEGACY",
        strategy_import_str="strategies.momentum.strategy_mo_atr_normalized_ndx:AtrNormalizedNdxStrategy")
    store_obj.upsert_release(legacy_release_obj)
    legacy_identity_dict = {field_str: getattr(legacy_release_obj, field_str) for field_str in
        ("release_id_str", "pod_id_str", "user_id_str", "account_route_str")}
    legacy_decision_obj = store_obj.insert_decision_plan(replace(decision_obj, **legacy_identity_dict,
        decision_plan_id_int=None, snapshot_metadata_dict={}, target_share_map_dict={}))
    legacy_plan_obj = store_obj.insert_vplan(replace(plan_obj, **legacy_identity_dict,
        decision_plan_id_int=legacy_decision_obj.decision_plan_id_int, vplan_id_int=None, submission_key_str="legacy:1"))
    store_obj.mark_vplan_status(legacy_plan_obj.vplan_id_int, "submitted")
    store_obj.mark_decision_plan_status(legacy_decision_obj.decision_plan_id_int, "submitted")
    with store_obj._connect() as connection_obj:
        connection_obj.execute("CREATE TRIGGER fail_alert BEFORE INSERT ON daily_pod_alert "
            "BEGIN SELECT RAISE(ABORT,'synthetic alert database failure'); END")
    as_of_ts = CLOSE_TS + timedelta(hours=1)
    broker_obj.error_obj = ConnectionError("synthetic disconnected broker")
    actual_position_dict = {**legacy_plan_obj.current_broker_position_map, **legacy_plan_obj.target_share_map}
    actual_position_dict = {asset_str: amount_float for asset_str, amount_float in actual_position_dict.items() if amount_float}
    legacy_broker_obj = SimpleNamespace(
        get_account_snapshot=lambda account_str: BrokerSnapshot(account_str, as_of_ts, 1000.0, 100000.0,
            actual_position_dict, net_liq_float=100000.0),
        get_recent_order_state_snapshot=lambda **_kwargs: ([], [], []),
        get_session_open_price_list=lambda **_kwargs: [])
    visited_pod_list = []
    def resolve_fn(current_release_obj):
        visited_pod_list.append(current_release_obj.pod_id_str)
        if current_release_obj.pod_id_str == legacy_release_obj.pod_id_str:
            return legacy_broker_obj
        if failure_str == "resolution":
            raise ConnectionError("synthetic adapter resolution failed")
        return broker_obj
    resolver_obj = SimpleNamespace(get_adapter=resolve_fn, set_incubation_context=lambda **_kwargs: None)
    monkeypatch.setattr(runner_module, "_load_release_list_validate_and_sync",
        lambda *_args, **_kwargs: [release_obj, legacy_release_obj])
    log_path_obj = tmp_path / "reconcile.log"
    result_dict = runner_module.post_execution_reconcile(store_obj, None, as_of_ts,
        env_mode_str=release_obj.mode_str, releases_root_path_str=str(tmp_path),
        broker_adapter_resolver_obj=resolver_obj, log_path_str=str(log_path_obj), trace_enabled_bool=False)
    assert result_dict["completed_vplan_count_int"] == 1
    assert visited_pod_list == [release_obj.pod_id_str, legacy_release_obj.pod_id_str]
    assert store_obj.get_decision_plan_by_id(decision_obj.decision_plan_id_int).status_str == "submitted"
    assert store_obj.get_vplan_by_id(plan_obj.vplan_id_int).status_str == "submitted"
    assert store_obj.get_decision_plan_by_id(legacy_decision_obj.decision_plan_id_int).status_str == "completed"
    assert store_obj.get_vplan_by_id(legacy_plan_obj.vplan_id_int).status_str == "completed"
    assert broker_obj.sent_request_list == [] and _row_list(store_obj) == []
    log_str = log_path_obj.read_text(encoding="utf-8")
    assert "daily_alert_enqueue_failed" in log_str and "synthetic alert database failure" in log_str
    assert "daily_cycle_reconcile_retry" in log_str and "synthetic" in log_str
