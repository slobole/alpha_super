"""Cycle projections must not turn missing or different-cycle evidence green."""

from copy import deepcopy
from datetime import datetime, timedelta

import pytest

from alpha.live.dashboard_v4.cycle import build_cycle_view_dict


@pytest.fixture
def pod_row_dict():
    return {
        "mode_str": "live", "pod_id_str": "test_daily", "account_route_str": "TEST_ACCOUNT", "db_status_str": "ok",
        "as_of_timestamp_str": "2026-09-18T13:32:00+00:00",
        "norgate_snapshot_status_dict": {
            "status_str": "ready", "snapshot_date_str": "2026-09-17",
            "snapshot_fresh_for_cycle_bool": True,
        },
        "signal_clock_str": "eod_snapshot_ready", "execution_policy_str": "next_open_moo",
        "session_calendar_id_str": "XNYS", "reason_code_str": "waiting_for_post_execution_reconcile",
        "next_action_str": "post_execution_reconcile", "required_action_dict": {"severity_str": "yellow"},
        "latest_decision_plan_id_int": 10, "latest_decision_plan_status_str": "submitted",
        "latest_decision_signal_timestamp_str": "2026-09-17T20:00:00+00:00",
        "latest_decision_plan_submission_timestamp_str": "2026-09-18T13:23:30+00:00",
        "latest_decision_plan_target_execution_timestamp_str": "2026-09-18T13:30:00+00:00",
        "latest_vplan_id_int": 20, "latest_vplan_decision_plan_id_int": 10,
        "latest_vplan_is_for_latest_decision_bool": True, "latest_vplan_cycle_role_str": "current",
        "latest_vplan_status_str": "submitted",
        "latest_vplan_submission_timestamp_str": "2026-09-18T13:23:30+00:00",
        "latest_vplan_target_execution_timestamp_str": "2026-09-18T13:30:00+00:00",
        "latest_submit_ack_status_str": "complete", "broker_order_count_int": 3,
        "broker_ack_count_int": 3, "missing_ack_count_int": 0, "fill_count_int": 4,
        "latest_reconciliation_status_str": "passed",
        "latest_reconciliation_timestamp_str": "2026-09-18T13:23:15+00:00",
        "eod_snapshot_dict": {
            "status_str": "waiting", "expected_market_date_str": "2026-09-18",
            "expected_due_timestamp_str": "2026-09-18T20:10:00+00:00",
            "latest_market_date_str": "2026-09-17", "latest_timestamp_str": "2026-09-17T20:10:00+00:00",
            "same_session_bool": False, "last_required_eod_present_bool": True, "source_str": "broker",
        },
    }


def _view_dict(pod_row_dict, now_str="2026-09-18T13:32:10+00:00"):
    return build_cycle_view_dict(pod_row_dict, now_ts=datetime.fromisoformat(now_str))


def _step_dict(view_dict, label_str):
    return next(step_dict for step_dict in view_dict["step_dict_list"] if step_dict["label_str"] == label_str)


def test_cycle_order_and_partial_fill_not_completed(pod_row_dict):
    view_dict = _view_dict(pod_row_dict)
    assert [step_dict["label_str"] for step_dict in view_dict["step_dict_list"]] == ["Data", "Decide", "Plan", "Submit", "Fill", "Reconcile", "EOD"]
    assert _step_dict(view_dict, "Submit")["fact_str"] == "3 sent · 3 ack"
    assert _step_dict(view_dict, "Fill")["state_str"] == "Now"
    assert _step_dict(view_dict, "Fill")["fact_str"] == "4 fill records"
    assert _step_dict(view_dict, "Reconcile")["state_str"] == "Planned"
    assert _step_dict(view_dict, "Reconcile")["actual_time_str"] == ""
    assert view_dict["pill_str"] == "Working"


def test_scheduled_timestamps_are_not_actual_timestamps(pod_row_dict):
    view_dict = _view_dict(pod_row_dict)
    for label_str in ("Data", "Decide", "Plan", "Submit", "Fill"):
        assert _step_dict(view_dict, label_str)["actual_time_str"] == ""
        assert _step_dict(view_dict, label_str)["delta_str"] == ""
    assert _step_dict(view_dict, "Submit")["planned_time_str"] == "09:23:30"
    assert _step_dict(view_dict, "Fill")["planned_time_str"] == "09:30:00"
    assert _step_dict(view_dict, "Plan")["planned_time_str"] == ""
    assert _step_dict(view_dict, "Reconcile")["planned_time_str"] == "09:35:00"
    assert _step_dict(view_dict, "EOD")["planned_time_str"] == "16:10:00"


@pytest.mark.parametrize("change_dict", [
    {"source_stale_bool": True},
    {"as_of_timestamp_str": "2026-09-18T13:29:00+00:00"},
    {"as_of_timestamp_str": "2026-09-18T13:33:00+00:00"},
    {"as_of_timestamp_str": None},
    {"db_status_str": "missing"},
    {"db_status_str": "error"},
    {"norgate_snapshot_status_dict": {}},
    {"norgate_snapshot_status_dict": {"status_str": "ready", "snapshot_fresh_for_cycle_bool": False}},
])
def test_stale_or_missing_source_never_green(pod_row_dict, change_dict):
    pod_row_dict.update(change_dict)
    view_dict = _view_dict(pod_row_dict)
    assert {step_dict["state_str"] for step_dict in view_dict["step_dict_list"]} == {"Unknown"}
    assert view_dict["pill_str"] == "Unknown"
    assert view_dict["next_time_str"] == ""


@pytest.mark.parametrize("vplan_status_str", ["submitted", "completed"])
@pytest.mark.parametrize("ack_status_str,missing_count_int", [("missing_critical", 0), ("not_checked", 1)])
def test_missing_ack_is_failure_not_filled_or_late(pod_row_dict, vplan_status_str, ack_status_str, missing_count_int):
    pod_row_dict.update(latest_vplan_status_str=vplan_status_str, latest_submit_ack_status_str=ack_status_str,
        missing_ack_count_int=missing_count_int, broker_ack_count_int=2)
    view_dict = _view_dict(pod_row_dict)
    assert _step_dict(view_dict, "Submit")["state_str"] == "Failed"
    assert view_dict["pill_str"] == "Action needed"
    assert view_dict["next_str"] == "Submit"


def test_ready_order_plan_past_due_is_late_not_failed(pod_row_dict):
    pod_row_dict.update(latest_vplan_status_str="ready", latest_submit_ack_status_str="not_checked", broker_order_count_int=0, broker_ack_count_int=0, fill_count_int=0)
    view_dict = _view_dict(pod_row_dict)
    assert _step_dict(view_dict, "Submit")["state_str"] == "Late"
    assert _step_dict(view_dict, "Fill")["state_str"] == "Planned"
    assert view_dict["pill_str"] == "Late"


def test_missing_fill_rows_are_not_a_no_order_day(pod_row_dict):
    pod_row_dict.update(latest_vplan_status_str="completed", fill_count_int=0, broker_order_count_int=0)
    assert _step_dict(_view_dict(pod_row_dict), "Fill")["state_str"] == "Unknown"


@pytest.mark.parametrize("fill_count_int", [1, 3, 5])
def test_completed_vplan_does_not_prove_every_order_filled(pod_row_dict, fill_count_int):
    pod_row_dict.update(latest_vplan_status_str="completed", fill_count_int=fill_count_int)
    assert _step_dict(_view_dict(pod_row_dict), "Fill")["state_str"] == "Unknown"


def test_missing_previous_eod_does_not_make_todays_future_capture_late(pod_row_dict):
    pod_row_dict["eod_snapshot_dict"].update(status_str="due_missing", last_required_eod_present_bool=False,
        last_required_market_date_str="2026-09-17", latest_market_date_str="2026-09-16")
    step_dict = _step_dict(_view_dict(pod_row_dict), "EOD")
    assert step_dict["state_str"] == "Late"
    assert step_dict["fact_str"] == "Missing snapshot · 2026-09-17"
    assert step_dict["planned_time_str"] == ""


def test_missing_counts_are_not_displayed_as_zero(pod_row_dict):
    for key_str in ("broker_order_count_int", "broker_ack_count_int", "fill_count_int"):
        pod_row_dict.pop(key_str)
    view_dict = _view_dict(pod_row_dict)
    assert _step_dict(view_dict, "Submit")["fact_str"] == "— sent · — ack"
    assert _step_dict(view_dict, "Submit")["state_str"] == "Unknown"
    assert _step_dict(view_dict, "Fill")["state_str"] == "Unknown"


def test_previous_unresolved_cycle_does_not_borrow_new_decision(pod_row_dict):
    pod_row_dict.update(
        latest_decision_plan_id_int=11, latest_decision_plan_status_str="completed",
        latest_decision_plan_target_execution_timestamp_str="2026-09-21T13:30:00+00:00",
        latest_decision_plan_submission_timestamp_str="2026-09-21T13:23:30+00:00",
        latest_submit_ack_status_str="missing_critical", missing_ack_count_int=1,
        latest_vplan_is_for_latest_decision_bool=False, latest_vplan_cycle_role_str="previous",
    )
    view_dict = _view_dict(pod_row_dict)
    assert view_dict["cycle_role_str"] == "previous"
    assert _step_dict(view_dict, "Decide")["state_str"] == "Unknown"
    assert _step_dict(view_dict, "Submit")["state_str"] == "Failed"
    assert _step_dict(view_dict, "Fill")["planned_time_str"] == "09:30:00"
    assert view_dict["pill_str"] == "Action needed"


def test_completed_previous_cycle_does_not_green_new_plan(pod_row_dict):
    pod_row_dict.update(
        latest_decision_plan_id_int=11, latest_decision_plan_status_str="planned",
        latest_vplan_status_str="completed", latest_vplan_is_for_latest_decision_bool=False,
        latest_vplan_cycle_role_str="previous", next_action_str="build_vplan",
        latest_decision_plan_submission_timestamp_str="2026-09-21T13:23:30+00:00",
        latest_decision_plan_target_execution_timestamp_str="2026-09-21T13:30:00+00:00",
    )
    view_dict = _view_dict(pod_row_dict)
    assert _step_dict(view_dict, "Decide")["state_str"] == "Done"
    assert _step_dict(view_dict, "Plan")["state_str"] == "Planned"
    assert _step_dict(view_dict, "Fill")["state_str"] == "Planned"
    assert "4 fill" not in _step_dict(view_dict, "Fill")["fact_str"]


def test_missing_cycle_identity_does_not_borrow_vplan(pod_row_dict):
    pod_row_dict.update(latest_vplan_decision_plan_id_int=None, latest_vplan_is_for_latest_decision_bool=None)
    view_dict = _view_dict(pod_row_dict)
    assert _step_dict(view_dict, "Plan")["state_str"] == "Unknown"
    assert _step_dict(view_dict, "Submit")["state_str"] == "Unknown"


def test_completed_reconcile_has_actual_time_and_delta(pod_row_dict):
    pod_row_dict.update(
        latest_vplan_status_str="completed", latest_decision_plan_status_str="completed",
        latest_reconciliation_timestamp_str="2026-09-18T13:35:01.053574+00:00",
        as_of_timestamp_str="2026-09-18T13:36:00+00:00",
    )
    view_dict = _view_dict(pod_row_dict, "2026-09-18T13:36:05+00:00")
    assert _step_dict(view_dict, "Reconcile")["state_str"] == "Done"
    assert _step_dict(view_dict, "Reconcile")["actual_time_str"] == "09:35:01"
    assert _step_dict(view_dict, "Reconcile")["delta_str"] == "+1s"


@pytest.mark.parametrize("reconcile_str", ["2026-09-17T13:35:00+00:00", "2026-09-18T13:23:00+00:00", "2026-09-19T13:35:00+00:00"])
def test_other_session_or_pre_execution_reconcile_cannot_complete(pod_row_dict, reconcile_str):
    pod_row_dict.update(latest_vplan_status_str="completed", latest_reconciliation_timestamp_str=reconcile_str)
    assert _step_dict(_view_dict(pod_row_dict), "Reconcile")["state_str"] != "Done"


def test_reconcile_read_failure_is_visible(pod_row_dict):
    pod_row_dict["reconcile_read_failure_dict"] = {"timestamp_str": "2026-09-18T13:32:00+00:00", "error_str": "Read failed"}
    view_dict = _view_dict(pod_row_dict)
    assert _step_dict(view_dict, "Reconcile")["state_str"] == "Failed"
    assert view_dict["pill_str"] == "Action needed"


def test_waiting_eod_does_not_reuse_yesterday_actual_time(pod_row_dict):
    eod_step_dict = _step_dict(_view_dict(pod_row_dict), "EOD")
    assert eod_step_dict["state_str"] == "Planned"
    assert eod_step_dict["actual_time_str"] == ""


def test_eod_same_session_only_and_seconds(pod_row_dict):
    pod_row_dict["as_of_timestamp_str"] = "2026-09-18T20:11:00+00:00"
    pod_row_dict["eod_snapshot_dict"].update(
        status_str="completed", same_session_bool=True, latest_market_date_str="2026-09-18",
        latest_timestamp_str="2026-09-18T20:10:01.999999+00:00",
    )
    view_dict = _view_dict(pod_row_dict, "2026-09-18T20:11:05+00:00")
    assert _step_dict(view_dict, "EOD")["state_str"] == "Done"
    assert _step_dict(view_dict, "EOD")["actual_time_str"] == "16:10:01"
    assert _step_dict(view_dict, "EOD")["delta_str"] == "+1s"
    pod_row_dict["eod_snapshot_dict"]["latest_market_date_str"] = "2026-09-17"
    assert _step_dict(_view_dict(pod_row_dict, "2026-09-18T20:11:05+00:00"), "EOD")["state_str"] == "Unknown"


def test_weekend_eod_none_requires_last_required_snapshot(pod_row_dict):
    pod_row_dict["as_of_timestamp_str"] = "2026-09-19T13:32:00+00:00"
    pod_row_dict["eod_snapshot_dict"].update(status_str="not_applicable", expected_due_timestamp_str=None)
    assert _step_dict(_view_dict(pod_row_dict, "2026-09-19T13:32:10+00:00"), "EOD")["state_str"] == "None"
    pod_row_dict["eod_snapshot_dict"]["last_required_eod_present_bool"] = False
    assert _step_dict(_view_dict(pod_row_dict, "2026-09-19T13:32:10+00:00"), "EOD")["state_str"] == "Unknown"


def test_idle_monthly_still_has_eod_due(pod_row_dict):
    pod_row_dict.update(
        reason_code_str="not_month_end_session", signal_clock_str="month_end_snapshot_ready",
        next_action_str="wait", required_action_dict={"severity_str": "green"},
        latest_vplan_status_str="completed", latest_decision_plan_status_str="completed",
    )
    view_dict = _view_dict(pod_row_dict)
    assert all(step_dict["state_str"] == "None" for step_dict in view_dict["step_dict_list"][:-1])
    assert _step_dict(view_dict, "EOD")["state_str"] == "Planned"
    assert view_dict["pill_str"] == "Waiting"
    assert view_dict["now_str"] == "Waiting"
    assert view_dict["now_detail_str"] == "No trade scheduled"
    assert view_dict["next_str"] == "EOD"


def test_saved_close_times_are_not_replaced_by_open_constants(pod_row_dict):
    pod_row_dict.update(
        execution_policy_str="same_day_moc", latest_vplan_status_str="ready",
        latest_vplan_submission_timestamp_str="2026-11-27T17:50:00+00:00",
        latest_vplan_target_execution_timestamp_str="2026-11-27T18:00:00+00:00",
        as_of_timestamp_str="2026-11-27T17:40:00+00:00", latest_submit_ack_status_str="not_checked",
    )
    view_dict = _view_dict(pod_row_dict, "2026-11-27T17:40:10+00:00")
    assert _step_dict(view_dict, "Submit")["planned_time_str"] == "12:50:00"
    assert _step_dict(view_dict, "Fill")["planned_time_str"] == "13:00:00"
    assert _step_dict(view_dict, "Reconcile")["planned_time_str"] == "13:05:00"


def test_projection_does_not_mutate_input(pod_row_dict):
    original_dict = deepcopy(pod_row_dict)
    _view_dict(pod_row_dict)
    assert pod_row_dict == original_dict


@pytest.mark.parametrize("mode_str", ["paper", "incubation", ""])
def test_other_modes_rejected(pod_row_dict, mode_str):
    pod_row_dict["mode_str"] = mode_str
    with pytest.raises(ValueError, match="LIVE only"):
        _view_dict(pod_row_dict)


@pytest.mark.parametrize("stage_str,due_str,allowance_int", [
    ("Submit", "2026-09-18T13:23:30+00:00", 60),
    ("Reconcile", "2026-09-18T13:35:00+00:00", 30),
    ("EOD", "2026-09-18T20:10:00+00:00", 30),
])
@pytest.mark.parametrize("point_str", ["before", "eligible", "within", "boundary", "late", "cached"])
def test_stage_eligibility_allows_scheduler_poll(pod_row_dict, stage_str, due_str, allowance_int, point_str):
    due_dt = datetime.fromisoformat(due_str)
    offset_int = {"before": -1, "eligible": 0, "within": 5, "boundary": allowance_int,
                  "late": allowance_int + 1, "cached": allowance_int + 5}[point_str]
    now_dt = due_dt + timedelta(seconds=offset_int)
    pod_row_dict["as_of_timestamp_str"] = (due_dt if point_str == "cached" else now_dt).isoformat()
    if stage_str == "Submit":
        pod_row_dict.update(latest_vplan_status_str="ready", latest_submit_ack_status_str="not_checked", broker_order_count_int=0, broker_ack_count_int=0)
    elif stage_str == "EOD":
        pod_row_dict["eod_snapshot_dict"].update(status_str="due_missing", last_required_market_date_str="2026-09-18", last_required_eod_present_bool=False)
    step_dict = _step_dict(_view_dict(pod_row_dict, now_dt.isoformat()), stage_str)
    assert step_dict["state_str"] == ("Planned" if point_str == "before" else "Late" if point_str == "late" else "Now")


def _complete_evidence_dict(pod_row_dict):
    return {
        "pod_id_str": pod_row_dict["pod_id_str"], "account_route_str": pod_row_dict["account_route_str"],
        "vplan_id_int": 20, "decision_plan_id_int": 10, "vplan_status_str": "completed",
        "state_str": "complete", "order_count_int": 3, "filled_order_count_int": 3,
        "actual_fill_timestamp_str": "2026-09-18T13:30:02+00:00",
    }


def test_completed_cycle_with_per_order_proof_is_on_track(pod_row_dict):
    pod_row_dict.update(latest_vplan_status_str="completed", latest_decision_plan_status_str="completed",
        latest_reconciliation_timestamp_str="2026-09-18T13:35:01+00:00", as_of_timestamp_str="2026-09-18T13:36:00+00:00")
    pod_row_dict["cycle_evidence_dict"] = _complete_evidence_dict(pod_row_dict)
    view_dict = _view_dict(pod_row_dict, "2026-09-18T13:36:00+00:00")
    assert _step_dict(view_dict, "Fill")["state_str"] == "Done"
    assert _step_dict(view_dict, "Fill")["actual_time_str"] == "09:30:02"
    assert view_dict["pill_str"] == "On track"
    assert view_dict["now_str"] == "Reconciled"
    assert view_dict["now_detail_str"] == "3 of 3 filled"


def test_reconciled_cycle_explains_unverified_fills_without_turning_green(pod_row_dict):
    pod_row_dict.update(latest_vplan_status_str="completed", latest_decision_plan_status_str="completed",
        latest_reconciliation_timestamp_str="2026-09-18T13:35:01+00:00", as_of_timestamp_str="2026-09-18T13:36:00+00:00")
    evidence_dict = _complete_evidence_dict(pod_row_dict)
    evidence_dict.update(state_str="unknown", actual_fill_timestamp_str=None,
        reason_str="A fill cannot be matched to its order, account or symbol.")
    pod_row_dict["cycle_evidence_dict"] = evidence_dict
    view_dict = _view_dict(pod_row_dict, "2026-09-18T13:36:00+00:00")
    assert view_dict["pill_str"] == "Not verified"
    assert view_dict["tone_str"] == "gray"
    assert view_dict["now_str"] == "Reconciled · Fill details not verified"
    assert view_dict["now_detail_str"] == evidence_dict["reason_str"]
    assert _step_dict(view_dict, "Fill")["state_str"] == "Unknown"
    assert _step_dict(view_dict, "Fill")["actual_time_str"] == ""
    assert _step_dict(view_dict, "Reconcile")["state_str"] == "Done"
    pod_row_dict["latest_submit_ack_status_str"] = "not_checked"
    assert _view_dict(pod_row_dict, "2026-09-18T13:36:00+00:00")["pill_str"] == "Unknown"


@pytest.mark.parametrize("ack_status_str", ["not_checked", "", None, "unrecognized"])
def test_completed_cycle_without_verified_ack_is_unknown_not_late(pod_row_dict, ack_status_str):
    pod_row_dict.update(latest_vplan_status_str="completed", latest_decision_plan_status_str="completed",
        latest_submit_ack_status_str=ack_status_str, broker_ack_count_int=0,
        latest_reconciliation_timestamp_str="2026-09-18T13:35:01+00:00", as_of_timestamp_str="2026-09-18T13:36:00+00:00")
    pod_row_dict["cycle_evidence_dict"] = _complete_evidence_dict(pod_row_dict)
    view_dict = _view_dict(pod_row_dict, "2026-09-18T13:36:00+00:00")
    submit_step_dict = _step_dict(view_dict, "Submit")
    assert submit_step_dict["state_str"] == "Unknown"
    assert submit_step_dict["fact_str"] == "ACK not verified"
    assert submit_step_dict["actual_time_str"] == ""
    assert _step_dict(view_dict, "Fill")["state_str"] == "Done"
    assert _step_dict(view_dict, "Reconcile")["state_str"] == "Done"
    assert view_dict["pill_str"] == "Unknown"


@pytest.mark.parametrize("field_str,value_obj", [
    ("account_route_str", "FOREIGN"), ("pod_id_str", "other"), ("vplan_id_int", 21),
    ("decision_plan_id_int", 11), ("vplan_status_str", "submitted"),
    ("actual_fill_timestamp_str", "2026-09-18T13:33:00+00:00"),
    ("actual_fill_timestamp_str", "2026-09-18T13:29:00+00:00"),
    ("filled_order_count_int", 2),
])
def test_wrong_cycle_or_incomplete_quantity_proof_cannot_complete_fill(pod_row_dict, field_str, value_obj):
    pod_row_dict["latest_vplan_status_str"] = "completed"
    evidence_dict = _complete_evidence_dict(pod_row_dict)
    evidence_dict[field_str] = value_obj
    pod_row_dict["cycle_evidence_dict"] = evidence_dict
    fill_step_dict = _step_dict(_view_dict(pod_row_dict), "Fill")
    assert fill_step_dict["state_str"] == "Unknown"
    if field_str == "actual_fill_timestamp_str":
        assert fill_step_dict["detail_str"] == "Fill completion time could not be verified for this cycle."


@pytest.mark.parametrize("vplan_status_str", ["submitted", "completed"])
def test_proven_no_orders_still_requires_reconciliation(pod_row_dict, vplan_status_str):
    evidence_dict = _complete_evidence_dict(pod_row_dict)
    evidence_dict.update(state_str="no_orders", vplan_status_str=vplan_status_str, order_count_int=0, filled_order_count_int=0, actual_fill_timestamp_str=None)
    pod_row_dict.update(cycle_evidence_dict=evidence_dict, latest_vplan_status_str=vplan_status_str,
        latest_submit_ack_status_str="not_checked", broker_order_count_int=0, broker_ack_count_int=0, fill_count_int=0)
    view_dict = _view_dict(pod_row_dict)
    assert _step_dict(view_dict, "Submit")["state_str"] == "None"
    assert _step_dict(view_dict, "Fill")["fact_str"] == "No orders"
    assert _step_dict(view_dict, "Reconcile")["state_str"] == "Planned"


def test_completed_decision_without_vplan_is_unknown_not_waiting(pod_row_dict):
    pod_row_dict.update(latest_decision_plan_status_str="completed", latest_vplan_id_int=None)
    view_dict = _view_dict(pod_row_dict)
    assert view_dict["pill_str"] == "Unknown"
    assert _step_dict(view_dict, "Plan")["fact_str"] == "Plan evidence missing"


def test_missing_vplan_after_submit_allowance_is_late(pod_row_dict):
    pod_row_dict.update(latest_decision_plan_status_str="planned", latest_vplan_id_int=None, next_action_str="build_vplan")
    assert _step_dict(_view_dict(pod_row_dict), "Submit")["state_str"] == "Late"


def test_no_order_claim_cannot_erase_ack_failure(pod_row_dict):
    pod_row_dict.update(latest_vplan_status_str="completed", missing_ack_count_int=1, latest_submit_ack_status_str="missing_critical")
    evidence_dict = _complete_evidence_dict(pod_row_dict)
    evidence_dict.update(state_str="no_orders", order_count_int=0)
    pod_row_dict["cycle_evidence_dict"] = evidence_dict
    view_dict = _view_dict(pod_row_dict)
    assert _step_dict(view_dict, "Submit")["state_str"] == "Failed"
    assert view_dict["pill_str"] == "Action needed"
