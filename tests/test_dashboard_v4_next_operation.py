"""A current Pod forecast must not become evidence or hide an obligation."""

from copy import deepcopy
from datetime import datetime
from types import SimpleNamespace

import pytest

from alpha.live.dashboard_v4.cycle import build_cycle_view_dict
from alpha.live.dashboard_v4.next_operation import build_next_operation_dict
from alpha.live.dashboard_v4.overview import build_overview_dict
from alpha.live.dashboard_v4.pod import build_pod_page_dict
from test_dashboard_v4_cycle import pod_row_dict, _complete_evidence_dict


def _monthly_dict(pod_row_dict, now_str="2026-09-20T14:00:00+00:00"):
    row_dict = deepcopy(pod_row_dict)
    row_dict.update(as_of_timestamp_str=now_str, reason_code_str="not_month_end_session",
        signal_clock_str="month_end_snapshot_ready", execution_policy_str="next_month_first_open",
        next_action_str="wait", required_action_dict={"severity_str": "green", "label_str": "No action"},
        latest_decision_plan_status_str="completed", latest_vplan_status_str="completed",
        latest_decision_signal_timestamp_str="2026-08-31T20:00:00+00:00",
        latest_decision_plan_submission_timestamp_str="2026-09-01T13:23:30+00:00",
        latest_decision_plan_target_execution_timestamp_str="2026-09-01T13:30:00+00:00",
        latest_vplan_submission_timestamp_str="2026-09-01T13:23:30+00:00",
        latest_vplan_target_execution_timestamp_str="2026-09-01T13:30:00+00:00",
        latest_reconciliation_timestamp_str="2026-09-01T13:36:00+00:00")
    row_dict["eod_snapshot_dict"].update(status_str="not_applicable", expected_due_timestamp_str=None,
        last_required_market_date_str="2026-09-18", last_required_eod_present_bool=True)
    return row_dict


def _next_dict(row_dict, **option_dict):
    now_ts = datetime.fromisoformat(row_dict["as_of_timestamp_str"])
    cycle_dict = build_cycle_view_dict(row_dict, now_ts=now_ts)
    return build_next_operation_dict(row_dict, cycle_dict, now_ts=now_ts, **option_dict)


def test_weekend_wait_has_next_session_eod_without_changing_cycle_evidence(pod_row_dict, monkeypatch):
    row_dict = _monthly_dict(pod_row_dict)
    before_dict = deepcopy(row_dict)
    def forbidden_fn(*args, **kwargs):
        raise AssertionError("A display forecast must not run or query the scheduler")
    for name_str in ("next_due", "get_scheduler_decision", "run_once"):
        monkeypatch.setattr("alpha.live.scheduler_service." + name_str, forbidden_fn)
    result_dict = _next_dict(row_dict)
    assert result_dict["next_str"] == "EOD"
    assert result_dict["next_time_str"] == "09-21 16:10:00"
    assert datetime.fromisoformat(result_dict["next_timestamp_str"]) == datetime.fromisoformat("2026-09-21T20:10:00+00:00")
    assert result_dict["next_detail_str"] == "Scheduled" and result_dict["next_forecast_bool"]
    assert row_dict == before_dict
    cycle_dict = build_cycle_view_dict(row_dict, now_ts=datetime.fromisoformat(row_dict["as_of_timestamp_str"]))
    assert cycle_dict["pill_str"] == "Waiting"
    assert [step_dict["state_str"] for step_dict in cycle_dict["step_dict_list"]] == ["None"] * 7


@pytest.mark.parametrize("now_str,signal_str,expected_str,utc_str", [
    ("2026-11-26T15:00:00+00:00", "2026-10-30T20:00:00+00:00", "11-27 13:10:00", "2026-11-27T18:10:00+00:00"),
    ("2026-11-08T15:00:00+00:00", "2026-10-30T20:00:00+00:00", "11-09 16:10:00", "2026-11-09T21:10:00+00:00"),
])
def test_eod_forecast_uses_holidays_early_close_and_et_dst(pod_row_dict, now_str, signal_str, expected_str, utc_str):
    row_dict = _monthly_dict(pod_row_dict, now_str)
    row_dict["latest_decision_signal_timestamp_str"] = signal_str
    result_dict = _next_dict(row_dict)
    assert result_dict["next_str"] == "EOD"
    assert result_dict["next_time_str"] == expected_str
    assert datetime.fromisoformat(result_dict["next_timestamp_str"]) == datetime.fromisoformat(utc_str)


def test_month_end_decision_precedes_eod_but_is_not_a_data_ready_deadline(pod_row_dict):
    row_dict = _monthly_dict(pod_row_dict, "2026-09-30T19:00:00+00:00")
    row_dict["eod_snapshot_dict"].update(status_str="waiting", expected_market_date_str="2026-09-30",
        expected_due_timestamp_str="2026-09-30T20:10:00+00:00")
    result_dict = _next_dict(row_dict)
    assert result_dict["next_str"] == "Decide"
    assert result_dict["next_time_str"] == "after 16:00:00"
    assert result_dict["next_timestamp_str"] == ""  # Cannot count down to vendor readiness.
    assert result_dict["next_detail_str"] == "Scheduled · when data is ready"
    assert result_dict["next_forecast_bool"] is True


def test_saved_monday_eod_time_is_retained_as_scheduled(pod_row_dict):
    row_dict = _monthly_dict(pod_row_dict, "2026-09-21T14:00:00+00:00")
    row_dict["eod_snapshot_dict"].update(status_str="waiting", expected_market_date_str="2026-09-21",
        expected_due_timestamp_str="2026-09-21T20:10:00+00:00")
    result_dict = _next_dict(row_dict)
    assert (result_dict["next_str"], result_dict["next_time_str"]) == ("EOD", "16:10:00")
    assert result_dict["next_detail_str"] == "Scheduled" and not result_dict["next_forecast_bool"]


def test_completed_friday_eod_forecasts_monday_without_relabeling_snapshot(pod_row_dict):
    row_dict = _monthly_dict(pod_row_dict, "2026-09-18T21:00:00+00:00")
    row_dict["eod_snapshot_dict"].update(status_str="completed", expected_market_date_str="2026-09-18",
        latest_market_date_str="2026-09-18", latest_timestamp_str="2026-09-18T20:10:04+00:00",
        expected_due_timestamp_str="2026-09-18T20:10:00+00:00", same_session_bool=True)
    assert _next_dict(row_dict)["next_time_str"] == "09-21 16:10:00"
    assert row_dict["eod_snapshot_dict"]["latest_market_date_str"] == "2026-09-18"


@pytest.mark.parametrize("failure_str", ["stale", "database", "data", "missing_eod", "failed_ack", "action"])
def test_unverified_or_actionable_state_never_gets_a_forecast(pod_row_dict, failure_str):
    row_dict = _monthly_dict(pod_row_dict)
    if failure_str == "stale":
        row_dict["source_stale_bool"] = True
    elif failure_str == "database":
        row_dict["db_status_str"] = "missing"
    elif failure_str == "data":
        row_dict["norgate_snapshot_status_dict"]["snapshot_fresh_for_cycle_bool"] = False
    elif failure_str == "missing_eod":
        row_dict["eod_snapshot_dict"].update(status_str="due_missing", last_required_eod_present_bool=False)
    elif failure_str == "failed_ack":
        row_dict["missing_ack_count_int"] = 1
    result_dict = _next_dict(row_dict, action_required_bool=failure_str == "action")
    assert result_dict["next_forecast_bool"] is False
    assert result_dict["next_str"] != "Decide"
    if failure_str in {"stale", "database", "data"}:
        assert result_dict["next_str"] == "Time unknown"


def test_existing_pending_plan_wins_over_calendar_forecasts(pod_row_dict):
    pod_row_dict.update(as_of_timestamp_str="2026-09-18T13:20:00+00:00", latest_vplan_status_str="ready",
        latest_decision_plan_status_str="vplan_ready", latest_submit_ack_status_str="not_checked",
        next_action_str="submit_vplan", required_action_dict={"severity_str": "yellow", "label_str": "VPlan ready"})
    result_dict = _next_dict(pod_row_dict)
    assert result_dict["next_str"] == "Submit"
    assert result_dict["next_time_str"] == "09:23:30" and not result_dict["next_forecast_bool"]


def test_missing_calendar_is_time_unknown_without_guessing_xnys(pod_row_dict):
    row_dict = _monthly_dict(pod_row_dict)
    row_dict["session_calendar_id_str"] = ""
    result_dict = _next_dict(row_dict)
    assert result_dict["next_str"] == "Time unknown"
    assert result_dict["next_detail_str"] == "Schedule unavailable"
    assert result_dict["next_timestamp_str"] == "" and not result_dict["next_forecast_bool"]


def test_unverified_fills_after_eod_do_not_claim_nothing_is_scheduled(pod_row_dict):
    pod_row_dict.update(as_of_timestamp_str="2026-09-18T21:00:00+00:00", next_action_str="wait",
        latest_vplan_status_str="completed", latest_decision_plan_status_str="completed",
        latest_reconciliation_timestamp_str="2026-09-18T13:36:00+00:00",
        required_action_dict={"severity_str": "green"})
    pod_row_dict["eod_snapshot_dict"].update(status_str="completed", same_session_bool=True,
        latest_market_date_str="2026-09-18", latest_timestamp_str="2026-09-18T20:10:00+00:00")
    cycle_dict = build_cycle_view_dict(pod_row_dict, now_ts=datetime.fromisoformat(pod_row_dict["as_of_timestamp_str"]))
    assert cycle_dict["pill_str"] == "Not verified"
    result_dict = _next_dict(pod_row_dict)
    assert result_dict["next_str"] == "Time unknown"
    assert result_dict["next_timestamp_str"] == "" and not result_dict["next_forecast_bool"]


@pytest.mark.parametrize("no_orders_bool", [False, True])
def test_completed_daily_cycle_forecasts_next_decision_without_changing_fill_facts(pod_row_dict, no_orders_bool):
    pod_row_dict.update(as_of_timestamp_str="2026-09-18T15:00:00+00:00", next_action_str="wait",
        latest_vplan_status_str="completed", latest_decision_plan_status_str="completed",
        latest_reconciliation_timestamp_str="2026-09-18T13:36:00+00:00", required_action_dict={"severity_str": "green"})
    pod_row_dict["cycle_evidence_dict"] = _complete_evidence_dict(pod_row_dict)
    if no_orders_bool:
        pod_row_dict["cycle_evidence_dict"].update(state_str="no_orders", order_count_int=0, filled_order_count_int=0, actual_fill_timestamp_str=None)
    result_dict = _next_dict(pod_row_dict)
    assert result_dict["next_str"] == "Decide"
    assert result_dict["next_time_str"] == "after 16:00:00"
    assert result_dict["next_timestamp_str"] == "" and result_dict["next_forecast_bool"]
    cycle_dict = build_cycle_view_dict(pod_row_dict, now_ts=datetime.fromisoformat(pod_row_dict["as_of_timestamp_str"]))
    assert cycle_dict["now_str"] == "Reconciled"
    assert cycle_dict["step_dict_list"][4]["fact_str"] == ("No orders" if no_orders_bool else "3 of 3 filled")


@pytest.mark.parametrize("policy_str,clock_str", [("same_day_moc", "pre_close_15m"), ("unknown", "unknown")])
def test_no_guessed_decision_forecast_for_unhandled_policy_clock(pod_row_dict, policy_str, clock_str):
    row_dict = _monthly_dict(pod_row_dict)
    row_dict.update(execution_policy_str=policy_str, signal_clock_str=clock_str)
    result_dict = _next_dict(row_dict)
    assert result_dict["next_str"] == "EOD"  # EOD's exchange calendar is still known.
    assert result_dict["next_time_str"] == "09-21 16:10:00"


@pytest.mark.parametrize("previous_bool", [False, True])
def test_unreconciled_no_orders_and_previous_execution_keep_the_current_obligation(pod_row_dict, previous_bool):
    evidence_dict = _complete_evidence_dict(pod_row_dict)
    evidence_dict.update(state_str="no_orders", vplan_status_str="submitted", order_count_int=0,
        filled_order_count_int=0, actual_fill_timestamp_str=None)
    pod_row_dict["cycle_evidence_dict"] = evidence_dict
    if previous_bool:
        pod_row_dict.update(latest_decision_plan_id_int=11, latest_decision_plan_status_str="planned",
            latest_vplan_is_for_latest_decision_bool=False)
    result_dict = _next_dict(pod_row_dict)
    assert result_dict["next_str"] == "Reconcile"
    assert result_dict["next_time_str"] == "09:35:00"
    assert result_dict["next_forecast_bool"] is False


def test_missed_daily_decision_from_canonical_calendar_takes_priority_over_eod(pod_row_dict, monkeypatch):
    pod_row_dict.update(as_of_timestamp_str="2026-09-21T14:00:00+00:00", next_action_str="wait",
        reason_code_str="submission_window_expired", latest_vplan_status_str="completed", latest_decision_plan_status_str="completed",
        latest_reconciliation_timestamp_str="2026-09-18T13:35:01+00:00", required_action_dict={"severity_str": "green", "label_str": "No action"})
    pod_row_dict["cycle_evidence_dict"] = _complete_evidence_dict(pod_row_dict)
    pod_row_dict["eod_snapshot_dict"].update(status_str="waiting", expected_market_date_str="2026-09-21",
        expected_due_timestamp_str="2026-09-21T20:10:00+00:00", last_required_market_date_str="2026-09-18",
        last_required_eod_present_bool=True)
    result_dict = _next_dict(pod_row_dict)
    assert result_dict["next_str"] == "Review saved evidence"
    assert result_dict["schedule_action_dict"]["label_str"] == "Missed DecisionPlan cycle"
    assert result_dict["next_timestamp_str"] == "" and not result_dict["next_forecast_bool"]
    workspace_dict = {"client_dict": {"accounts": []}, "operations_account_list": [
        {"pod_id": pod_row_dict["pod_id_str"], "account_route": pod_row_dict["account_route_str"], "display_name": "Daily"}],
        "summary_dict": {"as_of_timestamp_str": pod_row_dict["as_of_timestamp_str"], "pod_row_dict_list": [pod_row_dict]}}
    monkeypatch.setattr("alpha.live.dashboard_v4.overview.build_health_rollup", lambda *args, **kwargs: SimpleNamespace(severity_str="green"))
    overview_dict = build_overview_dict(workspace_dict, None, SimpleNamespace(),
        as_of_ts=datetime.fromisoformat(pod_row_dict["as_of_timestamp_str"]), include_finance_bool=False)
    assert overview_dict["pod_list"][0]["pill_str"] == "Action needed"
    assert overview_dict["pod_list"][0]["next_str"] == "Review saved evidence"
    assert overview_dict["attention_list"][0]["title_str"] == "Missed DecisionPlan cycle"


def test_overview_and_current_header_keep_next_operation_when_browsing_history(pod_row_dict, monkeypatch):
    row_dict = _monthly_dict(pod_row_dict)
    now_ts = datetime.fromisoformat(row_dict["as_of_timestamp_str"])
    workspace_dict = {"client_dict": {"accounts": []}, "operations_account_list": [
        {"pod_id": row_dict["pod_id_str"], "account_route": row_dict["account_route_str"], "display_name": "Monthly"}],
        "summary_dict": {"as_of_timestamp_str": now_ts.isoformat(), "pod_row_dict_list": [row_dict]}}
    monkeypatch.setattr("alpha.live.dashboard_v4.overview.build_health_rollup", lambda *args, **kwargs: SimpleNamespace(severity_str="green"))
    overview_dict = build_overview_dict(workspace_dict, None, SimpleNamespace(), as_of_ts=now_ts, include_finance_bool=False)
    current_dict = overview_dict["pod_list"][0]
    assert current_dict["pill_str"] == current_dict["now_str"] == "Waiting"
    assert current_dict["now_detail_str"] == "No trade scheduled"
    assert current_dict["next_detail_str"].startswith("Scheduled · in ")
    source_dict = {"status_str": "ok", "pod_row_dict": deepcopy(row_dict), "selected_explicit_bool": True,
        "selected_cycle_dict": {"current_bool": False, "session_date_str": "2026-09-01"}}
    result_dict = build_pod_page_dict(overview_dict, source_dict, {}, pod_id_str=row_dict["pod_id_str"], as_of_ts=now_ts)
    assert result_dict["header_dict"]["next_str"] == "EOD"
    assert result_dict["header_dict"]["next_time_str"] == "09-21 16:10:00"
    assert result_dict["historical_bool"] and result_dict["verdict_str"].startswith("Saved cycle:")
    assert result_dict["verdict_detail_str"] == "Saved status for 2026-09-01."
    unavailable_dict = build_pod_page_dict(overview_dict, {"status_str": "unknown", "selected_current_bool": True}, {},
        pod_id_str=row_dict["pod_id_str"], as_of_ts=now_ts)
    assert unavailable_dict["header_dict"]["next_str"] == "Time unknown"
    assert unavailable_dict["header_dict"]["next_time_str"] == ""
    assert unavailable_dict["header_dict"]["next_forecast_bool"] is False
