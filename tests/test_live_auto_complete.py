"""Offline cross-cancel recovery: real SQLite state, fake broker, no network."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

from alpha.live import auto_complete, runner
from alpha.live.models import BrokerOrderRecord, BrokerOrderEvent, SubmitBatchResult
from alpha.live.state_store_v2 import LiveStateStore
from test_live_daily_reconcile import daily_case, OPEN_TS, CLOSE_TS, DailyBroker, open_row, reconcile_case
from test_live_daily_shared_isolation import _monthly_case
from test_live_daily_scheduler_fairness import MONTHLY_STRATEGY_LIST


def seed_cancel(case_tuple, asset_str="AAPL", message_str="Order would cross related resting order", status_str="Cancelled"):
    store_obj, release_obj, decision_obj, plan_obj, broker_obj = case_tuple
    amount_float = plan_obj.order_delta_map[asset_str]
    order_id_str = f"original-{asset_str}"
    store_obj.upsert_vplan_broker_order_record_list([BrokerOrderRecord(order_id_str, decision_obj.decision_plan_id_int,
        plan_obj.vplan_id_int, release_obj.account_route_str, asset_str, f"daily:1:original:{asset_str}", "MOO", "shares",
        amount_float, 0, status_str, OPEN_TS - timedelta(minutes=7), submission_key_str=plan_obj.submission_key_str)])
    store_obj.insert_vplan_broker_order_event_list([BrokerOrderEvent(order_id_str, decision_obj.decision_plan_id_int,
        plan_obj.vplan_id_int, release_obj.account_route_str, asset_str, f"daily:1:original:{asset_str}", status_str,
        0, abs(amount_float), None, OPEN_TS - timedelta(minutes=7), message_str=message_str)])
    broker_obj.as_of_ts = OPEN_TS + timedelta(minutes=5)


def attempt_rows(case_tuple):
    with case_tuple[0]._connect() as connection_obj:
        return auto_complete.load_attempt_row_list(connection_obj, case_tuple[3].vplan_id_int)


def run_auto(case_tuple):
    store_obj, release_obj, _, plan_obj, broker_obj = case_tuple
    return auto_complete.reconcile_auto_completion(store_obj, broker_obj, release_obj, plan_obj, broker_obj.as_of_ts)


@pytest.mark.parametrize("seconds_int,expected_int", [(299, 0), (300, 1), (359, 1), (360, 0), (1800, 0)])
def test_one_send_only_in_the_minute_with_separate_dispatch_and_broker_deadlines(daily_case, seconds_int, expected_int):
    seed_cancel(daily_case)
    broker_obj = daily_case[4]
    broker_obj.as_of_ts = OPEN_TS + timedelta(seconds=seconds_int)
    run_auto(daily_case)
    run_auto(daily_case)
    assert len(broker_obj.sent_request_list) == expected_int
    if expected_int:
        request_obj = broker_obj.sent_request_list[0]
        assert request_obj.amount_float == 10
        assert request_obj.order_request_key_str.endswith(":auto-complete:AAPL")
        assert request_obj.submission_deadline_timestamp_str == (OPEN_TS + timedelta(minutes=6)).isoformat()
        assert request_obj.execution_deadline_timestamp_str == (OPEN_TS + timedelta(minutes=30)).isoformat()


@pytest.mark.parametrize("message_str,status_str", [("not enough buying power", "Cancelled"),
    ("Order would cross related resting order", "Rejected")])
def test_unrelated_reason_and_rejection_do_not_create_table_or_broker_reads(daily_case, message_str, status_str):
    seed_cancel(daily_case, message_str=message_str, status_str=status_str)
    assert not run_auto(daily_case)["affected_asset_set"]
    assert not attempt_rows(daily_case)
    assert daily_case[4].refresh_count_int == 0
    with daily_case[0]._connect() as connection_obj:
        assert not connection_obj.execute("SELECT 1 FROM sqlite_master WHERE name='auto_completion_attempt'").fetchone()


@pytest.mark.parametrize("actual_float,expected_float", [(6, 4), (-10, 10), (10, 0), (12, 0)])
def test_remainder_and_original_quantity_cap(daily_case, actual_float, expected_float):
    seed_cancel(daily_case)
    daily_case[4].position_dict["AAPL"] = actual_float
    run_auto(daily_case)
    assert sum(request_obj.amount_float for request_obj in daily_case[4].sent_request_list) == expected_float


@pytest.mark.parametrize("fault_str", ["open", "unreachable", "late_snapshot", "shared"])
def test_failed_check_consumes_opportunity(daily_case, fault_str):
    seed_cancel(daily_case)
    store_obj, release_obj, _, _, broker_obj = daily_case
    if fault_str == "open":
        broker_obj.open_row_list = [open_row("AAPL")]
    elif fault_str == "unreachable":
        broker_obj.error_obj = ConnectionError("offline")
    elif fault_str == "late_snapshot":
        broker_obj.snapshot_transform_fn = lambda snapshot_obj: replace(snapshot_obj,
            refreshed_timestamp_ts=OPEN_TS + timedelta(minutes=6))
    else:
        store_obj.upsert_release(replace(release_obj, release_id_str="other", pod_id_str="other"))
    run_auto(daily_case)
    broker_obj.open_row_list = []
    broker_obj.error_obj = None
    broker_obj.snapshot_transform_fn = lambda snapshot_obj: snapshot_obj
    run_auto(daily_case)
    assert not broker_obj.sent_request_list
    assert attempt_rows(daily_case)[0]["ended_timestamp_str"]


def test_uncertain_send_is_not_replayed_after_restart_and_final_partial_quantity_is_exact(daily_case):
    seed_cancel(daily_case)
    broker_obj = daily_case[4]
    broker_obj.send_error_obj = ConnectionError("ACK lost")
    run_auto(daily_case)
    assert attempt_rows(daily_case)[0]["ended_timestamp_str"] is None
    restarted_tuple = (LiveStateStore(daily_case[0].db_path_str), *daily_case[1:])
    run_auto(restarted_tuple)
    assert len(broker_obj.sent_request_list) == 1
    broker_obj.send_error_obj = None
    broker_obj.position_dict["AAPL"] = 4
    broker_obj.as_of_ts = OPEN_TS + timedelta(minutes=31)
    run_auto(restarted_tuple)
    attempt_dict = attempt_rows(restarted_tuple)[0]
    assert attempt_dict["status_str"] == "not_completed"
    assert attempt_dict["remaining_quantity_float"] == 6


def test_local_to_permanent_id_refresh_keeps_exact_request_cancel_link(daily_case):
    seed_cancel(daily_case)
    store_obj, _, _, plan_obj, _ = daily_case
    with store_obj._connect() as connection_obj:
        connection_obj.execute("UPDATE vplan_broker_order SET broker_order_id_str='permanent-123'")
    assert auto_complete.load_leg_dict(store_obj, plan_obj)["AAPL"]["original_broker_order_id_str"] == "permanent-123"
    run_auto(daily_case)
    assert len(daily_case[4].sent_request_list) == 1
    # A different request must not borrow the reason merely because its asset matches.
    with store_obj._connect() as connection_obj:
        connection_obj.execute("UPDATE vplan_broker_order_event SET order_request_key_str='different-request'")
    assert not auto_complete.load_leg_dict(store_obj, plan_obj)


def test_sparse_original_refresh_cannot_strand_a_working_attempt(daily_case):
    seed_cancel(daily_case)
    run_auto(daily_case)
    store_obj, _, _, plan_obj, broker_obj = daily_case
    original_obj = store_obj.get_broker_order_row_dict_list_for_vplan(plan_obj.vplan_id_int)[0]
    store_obj.upsert_vplan_broker_order_record_list([BrokerOrderRecord(
        original_obj["broker_order_id_str"], plan_obj.decision_plan_id_int, plan_obj.vplan_id_int,
        plan_obj.account_route_str, "AAPL", original_obj["order_request_key_str"], "MOO", "shares",
        0, 0, "Cancelled", OPEN_TS + timedelta(minutes=7), submission_key_str=plan_obj.submission_key_str)])
    assert not auto_complete.load_leg_dict(store_obj, plan_obj)
    broker_obj.as_of_ts = OPEN_TS + timedelta(minutes=31)
    result_dict = run_auto(daily_case)
    assert "AAPL" in result_dict["affected_asset_set"] and not result_dict["active_bool"]
    assert attempt_rows(daily_case)[0]["status_str"] == "completed"
    assert len(broker_obj.sent_request_list) == 1


def test_mark_before_send_crash_does_not_dispatch_on_restart(daily_case):
    seed_cancel(daily_case)
    store_obj, _, _, plan_obj, broker_obj = daily_case
    leg_dict = auto_complete.load_leg_dict(store_obj, plan_obj)["AAPL"]
    assert auto_complete._claim(store_obj, plan_obj, "AAPL", leg_dict, broker_obj.as_of_ts)
    run_auto(daily_case)
    assert not broker_obj.sent_request_list
    broker_obj.as_of_ts = OPEN_TS + timedelta(minutes=31)
    run_auto(daily_case)
    assert attempt_rows(daily_case)[0]["remaining_quantity_float"] == 10


def test_claim_is_atomic_for_concurrent_workers(daily_case):
    seed_cancel(daily_case)
    store_obj, _, _, plan_obj, broker_obj = daily_case
    leg_dict = auto_complete.load_leg_dict(store_obj, plan_obj)["AAPL"]
    with ThreadPoolExecutor(max_workers=2) as executor_obj:
        result_list = list(executor_obj.map(lambda worker_int: auto_complete._claim(
            store_obj, plan_obj, "AAPL", leg_dict, broker_obj.as_of_ts), range(2)))
    assert sorted(result_list) == [False, True]


@pytest.mark.parametrize("fault_str", ["offline", "open"])
def test_1001_cannot_unlock_without_complete_read_and_no_open_order_even_if_target_matches(daily_case, fault_str):
    seed_cancel(daily_case)
    run_auto(daily_case)
    broker_obj = daily_case[4]
    broker_obj.as_of_ts = OPEN_TS + timedelta(minutes=31)
    if fault_str == "offline":
        broker_obj.error_obj = ConnectionError("offline")
    else:
        broker_obj.open_row_list = [open_row("AAPL")]
    assert run_auto(daily_case)["active_bool"]
    assert attempt_rows(daily_case)[0]["ended_timestamp_str"] is None
    broker_obj.error_obj = None
    broker_obj.open_row_list = []
    assert not run_auto(daily_case)["active_bool"]
    assert attempt_rows(daily_case)[0]["status_str"] == "completed"


@pytest.mark.parametrize("status_str,filled_float,minutes_int", [
    ("Filled", 10.0, 5), ("Cancelled", 0.0, 5), ("Expired", 4.0, 30),
])
def test_terminal_recovery_finishes_early_only_after_fresh_broker_check(daily_case, status_str, filled_float, minutes_int):
    seed_cancel(daily_case)
    store_obj, release_obj, decision_obj, plan_obj, broker_obj = daily_case
    run_auto(daily_case)
    request_obj = broker_obj.sent_request_list[0]
    broker_obj.as_of_ts = OPEN_TS + timedelta(minutes=minutes_int, seconds=3)
    broker_obj.position_dict["AAPL"] = filled_float
    store_obj.upsert_vplan_broker_order_record_list([BrokerOrderRecord(
        "recovery-AAPL", decision_obj.decision_plan_id_int, plan_obj.vplan_id_int,
        release_obj.account_route_str, "AAPL", request_obj.order_request_key_str, "MKT", "shares",
        10, filled_float, status_str, OPEN_TS + timedelta(minutes=5),
        submission_key_str=plan_obj.submission_key_str)])
    broker_obj.error_obj = ConnectionError("observation failed")
    assert run_auto(daily_case)["active_bool"]
    assert attempt_rows(daily_case)[0]["ended_timestamp_str"] is None
    broker_obj.error_obj = None
    assert not run_auto(daily_case)["active_bool"]
    attempt_dict = attempt_rows(daily_case)[0]
    assert attempt_dict["status_str"] == ("completed" if filled_float == 10 else "not_completed")
    assert attempt_dict["remaining_quantity_float"] == 10 - filled_float
    assert attempt_dict["ended_timestamp_str"]
    run_auto(daily_case)
    assert len(broker_obj.sent_request_list) == 1


def test_daily_exit_cross_is_only_completed_once_by_new_rule(daily_case):
    seed_cancel(daily_case, "MSFT")
    reconcile_case(daily_case)
    request_list = [request_obj for request_obj in daily_case[4].sent_request_list if request_obj.asset_str == "MSFT"]
    assert len(request_list) == 1
    assert ":auto-complete:" in request_list[0].order_request_key_str
    reconcile_case(daily_case)
    assert len([request_obj for request_obj in daily_case[4].sent_request_list if request_obj.asset_str == "MSFT"]) == 1


def test_late_original_cancel_does_not_override_replacement():
    original_dict = {"asset_str": "DBC", "order_request_key_str": "batch:original", "last_status_timestamp_str": "2026-11-02T16:00:00Z"}
    replacement_dict = {"asset_str": "DBC", "order_request_key_str": "batch:auto-complete:DBC", "last_status_timestamp_str": "2026-11-02T14:35:00Z"}
    assert runner._build_latest_broker_order_row_by_asset_map_dict([replacement_dict, original_dict])["DBC"] == replacement_dict


def test_short_open_and_moc_are_out_of_scope(daily_case):
    store_obj, release_obj, _, plan_obj, broker_obj = daily_case
    if "DBC" in plan_obj.target_share_map:
        seed_cancel(daily_case, "DBC")
        run_auto(daily_case)
        assert not broker_obj.sent_request_list
    seed_cancel(daily_case)
    moc_obj = replace(plan_obj, execution_policy_str="next_close_moc")
    assert not auto_complete.reconcile_auto_completion(store_obj, broker_obj, release_obj, moc_obj,
        broker_obj.as_of_ts)["affected_asset_set"]


@pytest.mark.parametrize("strategy_str", MONTHLY_STRATEGY_LIST)
def test_standard_route_dispatch_tracks_until_verified_and_then_completes(tmp_path, strategy_str):
    store_obj, release_obj, plan_obj, adapter_obj = _monthly_case(tmp_path, strategy_str)
    execution_obj = DailyBroker({}, OPEN_TS + timedelta(minutes=5))
    case_tuple = (store_obj, release_obj, store_obj.get_decision_plan_by_id(plan_obj.decision_plan_id_int), plan_obj, execution_obj)
    seed_cancel(case_tuple)
    adapter_obj.get_daily_execution_snapshot = execution_obj.get_daily_execution_snapshot
    adapter_obj.get_account_snapshot = lambda account_str: execution_obj.get_daily_execution_snapshot(account_str).broker_snapshot_obj
    adapter_obj.submit_order_request_list = execution_obj.submit_order_request_list
    result_dict = runner.post_execution_reconcile(store_obj, adapter_obj, execution_obj.as_of_ts, "paper",
        pod_id_str=release_obj.pod_id_str, trace_enabled_bool=False, log_path_str=str(tmp_path / "events.jsonl"))
    assert result_dict["completed_vplan_count_int"] == 0
    assert len(execution_obj.sent_request_list) == 1
    assert not runner.is_vplan_execution_exception_parked_with_auto(store_obj, plan_obj, as_of_ts=execution_obj.as_of_ts)
    execution_obj.as_of_ts = OPEN_TS + timedelta(minutes=31)
    result_dict = runner.post_execution_reconcile(store_obj, adapter_obj, execution_obj.as_of_ts, "paper",
        pod_id_str=release_obj.pod_id_str, trace_enabled_bool=False, log_path_str=str(tmp_path / "events.jsonl"))
    assert result_dict["completed_vplan_count_int"] == 1
    assert attempt_rows(case_tuple)[0]["status_str"] == "completed"


def test_parked_standard_pod_rechecks_at_eod_and_manual_completion_unparks(tmp_path):
    store_obj, release_obj, plan_obj, adapter_obj = _monthly_case(tmp_path, MONTHLY_STRATEGY_LIST[0])
    execution_obj = DailyBroker({}, OPEN_TS + timedelta(minutes=31))
    case_tuple = (store_obj, release_obj, store_obj.get_decision_plan_by_id(plan_obj.decision_plan_id_int), plan_obj, execution_obj)
    seed_cancel(case_tuple)
    execution_obj.as_of_ts = OPEN_TS + timedelta(minutes=31)
    adapter_obj.get_account_snapshot = lambda account_str: execution_obj.get_daily_execution_snapshot(account_str).broker_snapshot_obj
    adapter_obj.get_daily_execution_snapshot = execution_obj.get_daily_execution_snapshot
    runner.post_execution_reconcile(store_obj, adapter_obj, execution_obj.as_of_ts, "paper", pod_id_str=release_obj.pod_id_str,
        trace_enabled_bool=False, log_path_str=str(tmp_path / "events.jsonl"))
    assert runner.is_vplan_execution_exception_parked(store_obj, plan_obj)
    assert not auto_complete.parked_recheck_due_bool(store_obj, plan_obj, release_obj, CLOSE_TS)
    execution_obj.as_of_ts = CLOSE_TS + timedelta(minutes=10)
    assert auto_complete.parked_recheck_due_bool(store_obj, plan_obj, release_obj, execution_obj.as_of_ts)
    execution_obj.position_dict["AAPL"] = 10
    result_dict = runner.post_execution_reconcile(store_obj, adapter_obj, execution_obj.as_of_ts, "paper",
        pod_id_str=release_obj.pod_id_str, trace_enabled_bool=False, log_path_str=str(tmp_path / "events.jsonl"))
    assert result_dict["completed_vplan_count_int"] == 1
    assert not auto_complete.parked_recheck_due_bool(store_obj, plan_obj, release_obj, execution_obj.as_of_ts)


def test_daily_close_owns_uncertain_recovery_ref_without_broker_record(daily_case):
    from alpha.live.daily_reconcile import load_daily_owned_order_ref_set
    seed_cancel(daily_case)
    daily_case[4].send_error_obj = ConnectionError("accepted but no result")
    run_auto(daily_case)
    reference_str = attempt_rows(daily_case)[0]["order_request_key_str"]
    store_obj, release_obj, decision_obj, _, _ = daily_case
    assert reference_str in load_daily_owned_order_ref_set(store_obj, release_obj, decision_obj, [])


def test_new_decision_during_broker_refresh_prevents_stale_send(daily_case):
    seed_cancel(daily_case)
    store_obj, _, decision_obj, _, broker_obj = daily_case
    def supersede_fn(snapshot_obj):
        store_obj.insert_decision_plan(replace(decision_obj, decision_plan_id_int=None,
            signal_timestamp_ts=decision_obj.signal_timestamp_ts + timedelta(days=1)))
        return snapshot_obj
    broker_obj.snapshot_transform_fn = supersede_fn
    run_auto(daily_case)
    assert not broker_obj.sent_request_list
    assert "newer decision" in attempt_rows(daily_case)[0]["reason_str"]


def test_dedicated_check_sees_other_enabled_manifest_outside_pod_database(daily_case, monkeypatch):
    seed_cancel(daily_case)
    store_obj, release_obj, _, plan_obj, broker_obj = daily_case
    other_obj = replace(release_obj, release_id_str="other", pod_id_str="other")
    monkeypatch.setattr("alpha.live.release_manifest.load_release_list", lambda root_str: [release_obj, other_obj])
    auto_complete.reconcile_auto_completion(store_obj, broker_obj, release_obj, plan_obj, broker_obj.as_of_ts,
        releases_root_path_str="isolated-manifests")
    assert not broker_obj.sent_request_list
    assert attempt_rows(daily_case)[0]["reason_str"] == "Account shared with another pod"


def test_standard_submit_persists_cross_evidence_if_followup_refresh_raises(tmp_path):
    store_obj, release_obj, plan_obj, adapter_obj = _monthly_case(tmp_path, MONTHLY_STRATEGY_LIST[0])
    store_obj.mark_vplan_status(plan_obj.vplan_id_int, "ready")
    store_obj.mark_decision_plan_status(plan_obj.decision_plan_id_int, "vplan_ready")
    adapter_obj.seed_account_snapshot(release_obj.account_route_str, cash_float=5000, total_value_float=5000,
        position_amount_map={}, snapshot_timestamp_ts=plan_obj.submission_timestamp_ts)
    def fail_submit_fn(*, account_route_str, broker_order_request_list, submitted_timestamp_ts):
        request_obj = broker_order_request_list[0]
        record_obj = BrokerOrderRecord("901", None, None, account_route_str, request_obj.asset_str,
            request_obj.order_request_key_str, "MOO", "shares", request_obj.amount_float, 0, "Cancelled", submitted_timestamp_ts)
        event_obj = BrokerOrderEvent("901", None, None, account_route_str, request_obj.asset_str,
            request_obj.order_request_key_str, "Cancelled", 0, abs(request_obj.amount_float), None, submitted_timestamp_ts,
            message_str="Order would cross related resting order")
        exception_obj = ConnectionError("refresh failed after callback")
        exception_obj.cross_cancel_result_obj = SubmitBatchResult(broker_order_record_list=[record_obj], broker_order_event_list=[event_obj])
        raise exception_obj
    adapter_obj.submit_order_request_list = fail_submit_fn
    with pytest.raises(ConnectionError, match="refresh failed"):
        runner.submit_ready_vplans(store_obj, adapter_obj, plan_obj.submission_timestamp_ts, env_mode_str="paper",
            manual_only_bool=False, vplan_id_int=plan_obj.vplan_id_int, trace_enabled_bool=False,
            log_path_str=str(tmp_path / "events.jsonl"))
    assert auto_complete.load_leg_dict(store_obj, plan_obj)["AAPL"]["original_broker_order_id_str"] == "901"


def test_mixed_failed_and_working_legs_do_not_park_while_one_attempt_is_unresolved(daily_case):
    seed_cancel(daily_case, "AAPL")
    seed_cancel(daily_case, "MSFT")
    store_obj, _, _, plan_obj, broker_obj = daily_case
    broker_obj.open_row_list = [open_row("AAPL")]
    broker_obj.send_error_obj = ConnectionError("MSFT outcome not yet known")
    assert run_auto(daily_case)["active_bool"]
    attempt_map_dict = {row_dict["asset_str"]: row_dict for row_dict in attempt_rows(daily_case)}
    assert attempt_map_dict["AAPL"]["ended_timestamp_str"]
    assert attempt_map_dict["MSFT"]["ended_timestamp_str"] is None
    assert not runner.is_vplan_execution_exception_parked_with_auto(store_obj, plan_obj, as_of_ts=broker_obj.as_of_ts)


@pytest.mark.parametrize("date_str,utc_hour_int,close_hour_int", [
    ("2026-03-06", 14, 16), ("2026-03-09", 13, 16), ("2026-11-27", 14, 13)])
def test_exchange_calendar_anchors_dst_and_early_close(daily_case, date_str, utc_hour_int, close_hour_int):
    plan_obj = replace(daily_case[3], target_execution_timestamp_ts=datetime.fromisoformat(date_str).replace(
        hour=9, minute=30, tzinfo=OPEN_TS.tzinfo))
    window_dict = auto_complete.timing_dict(plan_obj, "XNYS")
    assert window_dict["start_ts"].astimezone(UTC).hour == utc_hour_int
    assert window_dict["start_ts"].minute == 35
    assert window_dict["final_check_ts"].hour == 10
    assert window_dict["close_ts"].hour == close_hour_int


@pytest.mark.parametrize("matches_bool", [False, True])
def test_scheduler_rechecks_unrelated_parked_cancel_once_then_allows_eod_only_if_matched(
        tmp_path, monkeypatch, matches_bool):
    from alpha.live import scheduler_service
    store_obj, release_obj, plan_obj, adapter_obj = _monthly_case(tmp_path, MONTHLY_STRATEGY_LIST[0])
    execution_obj = DailyBroker({}, OPEN_TS + timedelta(minutes=31))
    case_tuple = (store_obj, release_obj, store_obj.get_decision_plan_by_id(plan_obj.decision_plan_id_int),
        plan_obj, execution_obj)
    seed_cancel(case_tuple, message_str="Not enough buying power")
    execution_obj.as_of_ts = OPEN_TS + timedelta(minutes=31)
    adapter_obj.get_account_snapshot = lambda account_str: execution_obj.get_daily_execution_snapshot(account_str).broker_snapshot_obj
    adapter_obj.get_daily_execution_snapshot = execution_obj.get_daily_execution_snapshot
    adapter_obj.submit_order_request_list = execution_obj.submit_order_request_list
    monkeypatch.setattr(scheduler_service, "_load_release_list_and_sync",
        lambda *argument_tuple, **keyword_dict: [release_obj])
    monkeypatch.setattr(scheduler_service.scheduler_utils, "evaluate_build_gate_dict",
        lambda release_obj, as_of_ts: {"due_bool": False})

    def scheduler_at(as_of_ts):
        return scheduler_service.get_scheduler_decision(store_obj, as_of_ts,
            str(tmp_path / "releases"), "paper", pod_id_str=release_obj.pod_id_str)

    runner.post_execution_reconcile(store_obj, adapter_obj, execution_obj.as_of_ts, "paper",
        pod_id_str=release_obj.pod_id_str, trace_enabled_bool=False, log_path_str=str(tmp_path / "events.jsonl"))
    before_obj = scheduler_at(CLOSE_TS + timedelta(minutes=9, seconds=59))
    assert not before_obj.due_now_bool and before_obj.reason_code_str == "execution_exception_parked"
    execution_obj.as_of_ts = CLOSE_TS + timedelta(minutes=10)
    due_obj = scheduler_at(execution_obj.as_of_ts)
    assert due_obj.due_now_bool and due_obj.next_phase_str == "post_execution_reconcile"
    assert due_obj.reason_code_str == "parked_eod_recheck_due"
    if matches_bool:
        execution_obj.position_dict["AAPL"] = 10
    result_dict = runner.post_execution_reconcile(store_obj, adapter_obj, execution_obj.as_of_ts, "paper",
        pod_id_str=release_obj.pod_id_str, trace_enabled_bool=False, log_path_str=str(tmp_path / "events.jsonl"))
    assert result_dict["completed_vplan_count_int"] == int(matches_bool)
    after_obj = scheduler_at(execution_obj.as_of_ts + timedelta(minutes=1))
    if matches_bool:
        assert after_obj.due_now_bool and after_obj.next_phase_str == "eod_snapshot"
    else:
        assert not after_obj.due_now_bool and after_obj.reason_code_str == "execution_exception_parked"
    assert not execution_obj.sent_request_list
    assert not attempt_rows(case_tuple)
