"""Waiting for tonight's data before its alert deadline is a wait, not an Unknown cycle."""

from copy import deepcopy
from datetime import datetime

import pytest

from alpha.live.dashboard_v4.cycle import build_cycle_view_dict
from alpha.live.dashboard_v4.next_operation import build_next_operation_dict
from test_dashboard_v4_cycle import pod_row_dict, _complete_evidence_dict


EVENING_STR = "2026-09-18T21:30:00+00:00"  # 17:30 ET, after today's EOD


def _evening_row_dict(pod_row_dict, *, past_deadline_obj=False, status_str="ready"):
    row_dict = deepcopy(pod_row_dict)
    row_dict.update(as_of_timestamp_str=EVENING_STR, release_id_str="release-daily",
        next_action_str="wait", reason_code_str="snapshot_not_ready_for_session",
        required_action_dict={"severity_str": "green", "label_str": "No action"},
        latest_decision_plan_status_str="completed", latest_vplan_status_str="completed",
        latest_reconciliation_timestamp_str="2026-09-18T13:36:00+00:00")
    row_dict["cycle_evidence_dict"] = _complete_evidence_dict(row_dict)
    row_dict["norgate_snapshot_status_dict"] = {
        "status_str": status_str, "snapshot_date_str": "2026-09-17", "snapshot_fresh_for_cycle_bool": False,
        "snapshot_stale_past_alert_deadline_bool": past_deadline_obj,
        "required_snapshot_date_by_release_dict": {"release-daily": "2026-09-18"},
        "stale_alert_deadline_by_release_dict": {"release-daily": "2026-09-21T12:53:30+00:00"},
    }
    row_dict["eod_snapshot_dict"].update(status_str="completed", latest_market_date_str="2026-09-18",
        latest_timestamp_str="2026-09-18T20:10:04+00:00", same_session_bool=True)
    return row_dict


def _view_tuple(row_dict):
    now_ts = datetime.fromisoformat(row_dict["as_of_timestamp_str"])
    cycle_dict = build_cycle_view_dict(row_dict, now_ts=now_ts)
    return cycle_dict, build_next_operation_dict(row_dict, cycle_dict, now_ts=now_ts)


def test_data_wait_before_deadline_keeps_saved_steps_and_names_the_wait(pod_row_dict):
    cycle_dict, next_dict = _view_tuple(_evening_row_dict(pod_row_dict))
    state_list = [step_dict["state_str"] for step_dict in cycle_dict["step_dict_list"]]
    assert state_list[0] == "Now"
    assert "Unknown" not in state_list
    assert cycle_dict["step_dict_list"][0]["fact_str"] == "Waiting for 2026-09-18 data"
    assert cycle_dict["step_dict_list"][-1]["state_str"] == "Done"  # Today's EOD stays visible.
    assert cycle_dict["data_waiting_bool"] is True and not cycle_dict["stale_bool"]
    assert cycle_dict["now_str"] == "Waiting for 2026-09-18 data"
    assert (next_dict["next_str"], next_dict["next_time_str"]) == ("Data", "when ready")
    assert next_dict["next_detail_str"] == "alert 09-21 08:53:30 ET"
    assert next_dict["next_timestamp_str"] == "" and next_dict["next_forecast_bool"] is False


@pytest.mark.parametrize("past_deadline_obj", [True, None, "false"])
def test_past_or_unknown_deadline_keeps_the_unknown_cycle(pod_row_dict, past_deadline_obj):
    cycle_dict, _ = _view_tuple(_evening_row_dict(pod_row_dict, past_deadline_obj=past_deadline_obj))
    assert all(step_dict["state_str"] == "Unknown" for step_dict in cycle_dict["step_dict_list"])
    assert cycle_dict["stale_bool"] is True and not cycle_dict.get("data_waiting_bool")


def test_failed_sync_is_never_shown_as_a_calm_wait(pod_row_dict):
    cycle_dict, _ = _view_tuple(_evening_row_dict(pod_row_dict, status_str="failed"))
    assert all(step_dict["state_str"] == "Unknown" for step_dict in cycle_dict["step_dict_list"])


def test_wait_without_deadline_has_no_alert_claim(pod_row_dict):
    row_dict = _evening_row_dict(pod_row_dict)
    row_dict["norgate_snapshot_status_dict"].pop("stale_alert_deadline_by_release_dict")
    row_dict["norgate_snapshot_status_dict"].pop("required_snapshot_date_by_release_dict")
    cycle_dict, next_dict = _view_tuple(row_dict)
    assert cycle_dict["step_dict_list"][0]["fact_str"] == "Waiting for new data"
    assert next_dict["next_detail_str"] == ""


def test_stale_summary_still_overrides_a_data_wait(pod_row_dict):
    row_dict = _evening_row_dict(pod_row_dict)
    row_dict["source_stale_bool"] = True
    cycle_dict, _ = _view_tuple(row_dict)
    assert all(step_dict["fact_str"] == "Status out of date" for step_dict in cycle_dict["step_dict_list"])
