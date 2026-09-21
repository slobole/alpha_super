"""System health distinguishes saved evidence from current service connectivity."""

from copy import deepcopy
from datetime import datetime, timedelta, timezone

import pytest

from alpha.live.dashboard_v4.system import build_system_page_dict, system_scope_matches_bool


NOW_TS = datetime(2026, 9, 21, 14, 0, tzinfo=timezone.utc)


@pytest.fixture
def source_tuple():
    overview_dict = {"source_fresh_bool": True, "system_dict": {"state_str": "done", "detail_str": "Data 2026-09-18"},
        "health_list": [{"label_str": "Disk", "severity_str": "green", "value_str": "50% used"}],
        "pod_list": [{"pod_id_str": "pod1", "name_str": "First strategy", "scheduler_dict": {
            "state_str": "sleeping", "alive_bool": True, "checked_timestamp_str": NOW_TS.isoformat(),
            "last_seen_timestamp_str": (NOW_TS - timedelta(seconds=30)).isoformat(),
            "promised_wake_timestamp_str": (NOW_TS + timedelta(seconds=3570)).isoformat(), "next_phase_str": "eod_snapshot"}}]}
    workspace_dict = {"operations_error_str": "", "operations_account_list": [{"pod_id": "pod1", "account_route": "U123456", "display_name": "First strategy"}],
        "summary_dict": {"as_of_timestamp_str": NOW_TS.isoformat(), "pod_row_dict_list": [{
            "as_of_timestamp_str": NOW_TS.isoformat(), "pod_id_str": "pod1", "account_route_str": "U123456", "mode_str": "live", "release_id_str": "release1",
            "db_status_str": "ok", "latest_broker_snapshot_timestamp_str": "2026-09-18T20:10:00+00:00",
            "norgate_snapshot_status_dict": {"snapshot_date_str": "2026-09-18", "severity_str": "green", "status_str": "ready",
                "last_sync_utc_str": "2026-09-18T22:00:00+00:00", "required_snapshot_date_by_release_dict": {"release1": "2026-09-18"}},
            "eod_snapshot_dict": {"latest_timestamp_str": "2026-09-18T20:10:00+00:00", "severity_str": "green"},
            "dtb3_latest_observation_date_str": "2026-09-18", "data_freshness_dict": {"item_dict_list": [{"label_str": "DTB3/FRED", "severity_str": "green"}]}}]}}
    source_dict = {"scope_verified_bool": True, "checked_timestamp_str": NOW_TS.isoformat(), "release_list": [{"pod_id_str": "pod1", "name_str": "First strategy", "mode_str": "live", "enabled_bool": True,
        "release_id_str": "release1", "execution_policy_str": "next_month_first_open", "account_str": "U···456", "private_str": "must not export"}]}
    for key_str in ("watchdog", "flex", "event_log", "database"):
        source_dict[key_str + "_dict"] = {"state_str": "ok", "now_str": "Saved evidence available", "last_timestamp_str": NOW_TS.isoformat(), "expected_str": "Saved configuration"}
    return overview_dict, workspace_dict, source_dict


def _view_dict(source_tuple, *, as_of_ts=NOW_TS):
    return build_system_page_dict(*source_tuple, as_of_ts=as_of_ts)


def _rows_dict(view_dict):
    return {row_dict["key_str"]: row_dict for group_dict in view_dict["group_list"] for row_dict in group_dict["row_list"]}


def test_saved_evidence_never_claims_live_gateway_or_missing_receipts(source_tuple):
    original_tuple = deepcopy(source_tuple)
    view_dict = _view_dict(source_tuple)
    rows_dict = _rows_dict(view_dict)
    assert [group_dict["label_str"] for group_dict in view_dict["group_list"]] == ["Runs all the time", "Runs on a schedule", "Data and space"]
    assert rows_dict["schedulers"]["state_str"] == "done"
    assert rows_dict["schedulers"]["now_str"] == "1 of 1 alive"
    assert rows_dict["gateway"]["state_str"] == "skip" and rows_dict["gateway"]["checked_bool"] is False
    assert rows_dict["gateway"]["now_str"] == "Not checked here"
    assert rows_dict["gateway"]["last_str"] == "1 of 1 saved reads · 09-18 16:10:00"
    assert rows_dict["alerts"]["state_str"] == rows_dict["deadman"]["state_str"] == "unk"
    assert rows_dict["dashboard"]["state_str"] == "done" and rows_dict["dashboard"]["expected_str"] == "Refresh every 15 s"
    assert view_dict["state_str"] == "unk" and "unverified" in view_dict["verdict_str"]
    assert "Connected" not in str(view_dict) and "delivered" not in str(view_dict)
    assert view_dict["release_list"][0]["trades_str"] == "First open of month"
    assert "U123456" not in str(view_dict) and "must not export" not in str(view_dict)
    assert source_tuple == original_tuple


@pytest.mark.parametrize("state_str,alive_bool,tone_str,label_str", [
    ("holding", True, "skip", "Holding · waits for your review"),
    ("running", True, "done", "Running · EOD"),
    ("error", True, "fail", "Error · check log"),
    ("late", None, "late", "Wake overdue"),
    ("stopped", False, "fail", "Not responding · check service"),
    ("unknown", None, "unk", "Unknown · recent activity only"),
])
def test_scheduler_states_use_saved_actual_state(source_tuple, state_str, alive_bool, tone_str, label_str):
    source_tuple[0]["pod_list"][0]["scheduler_dict"].update(state_str=state_str, alive_bool=alive_bool)
    view_dict = _view_dict(source_tuple)
    assert view_dict["pod_list"][0]["scheduler_state_str"] == tone_str
    assert view_dict["pod_list"][0]["scheduler_str"] == label_str
    assert view_dict["pod_list"][0]["wake_str"] == "10:59:30"
    assert _rows_dict(view_dict)["schedulers"]["state_str"] == ("done" if state_str == "holding" else tone_str)
    if tone_str in {"late", "fail"}:
        assert view_dict["state_str"] == tone_str


def test_real_wait_for_data_flag_and_error_do_not_derive_from_pod_issue(source_tuple):
    scheduler_dict = source_tuple[0]["pod_list"][0]["scheduler_dict"]
    scheduler_dict["waiting_for_data_bool"] = True
    assert _view_dict(source_tuple)["pod_list"][0]["scheduler_str"] == "Waiting for data"
    scheduler_dict["state_str"] = "error"
    assert _view_dict(source_tuple)["pod_list"][0]["scheduler_str"] == "Error · check log"


@pytest.mark.parametrize("seconds_int,tone_str", [(60, "done"), (61, "late"), (300, "late"), (301, "fail")])
def test_cached_wake_promise_cannot_remain_green_after_deadline(source_tuple, seconds_int, tone_str):
    source_tuple[0]["pod_list"][0]["scheduler_dict"]["promised_wake_timestamp_str"] = (NOW_TS - timedelta(seconds=seconds_int)).isoformat()
    assert _view_dict(source_tuple)["pod_list"][0]["scheduler_state_str"] == tone_str


def test_error_stays_failed_after_wake_grace_and_does_not_count_as_alive(source_tuple):
    source_tuple[0]["pod_list"][0]["scheduler_dict"].update(state_str="error",
        promised_wake_timestamp_str=(NOW_TS - timedelta(seconds=61)).isoformat())
    view_dict = _view_dict(source_tuple)
    assert view_dict["pod_list"][0]["scheduler_state_str"] == "fail"
    assert _rows_dict(view_dict)["schedulers"]["now_str"] == "0 of 1 alive"


@pytest.mark.parametrize("state_str", ["fail", "late"])
def test_existing_system_problem_cannot_be_hidden_by_unknown_auxiliary_checks(source_tuple, state_str):
    source_tuple[0]["system_dict"] = {"state_str": state_str}
    assert _view_dict(source_tuple)["state_str"] == state_str


def test_eod_problem_is_visible_without_exposing_raw_reason(source_tuple):
    source_tuple[1]["summary_dict"]["pod_row_dict_list"][0]["eod_snapshot_dict"].update(
        status_str="due_missing", severity_str="yellow", detail_str="private raw reason")
    view_dict = _view_dict(source_tuple)
    assert view_dict["state_str"] == "late"
    assert view_dict["pod_list"][0]["eod_state_str"] == "late"
    assert view_dict["pod_list"][0]["eod_str"] == "Due snapshot missing · 09-18 16:10:00"
    assert "private" not in str(view_dict)


def test_eod_capture_grace_reuses_overview_cycle_assessment(source_tuple):
    source_tuple[1]["summary_dict"]["pod_row_dict_list"][0]["eod_snapshot_dict"].update(status_str="due_missing", severity_str="yellow")
    source_tuple[0]["pod_list"][0]["step_list"] = [{"name_str": "EOD", "state_str": "now"}]
    view_dict = _view_dict(source_tuple)
    assert view_dict["pod_list"][0]["eod_state_str"] == "now"
    assert view_dict["pod_list"][0]["eod_str"].startswith("Waiting for capture")
    assert view_dict["state_str"] == "unk"


@pytest.mark.parametrize("change_str", ["stale", "future", "foreign_account", "foreign_mode", "duplicate", "missing", "row_stale", "row_future", "operations_error"])
def test_unverified_operations_do_not_green_pod_evidence(source_tuple, change_str):
    overview_dict, workspace_dict, _ = source_tuple
    summary_dict = workspace_dict["summary_dict"]
    saved_dict = summary_dict["pod_row_dict_list"][0]
    if change_str in {"stale", "future"}:
        summary_dict["as_of_timestamp_str"] = (NOW_TS + timedelta(seconds=1 if change_str == "future" else -121)).isoformat()
    elif change_str == "foreign_account":
        saved_dict["account_route_str"] = "OTHER"
    elif change_str == "foreign_mode":
        saved_dict["mode_str"] = "paper"
    elif change_str == "duplicate":
        summary_dict["pod_row_dict_list"].append(deepcopy(saved_dict))
    elif change_str == "missing":
        summary_dict["pod_row_dict_list"] = []
    elif change_str in {"row_stale", "row_future"}:
        saved_dict["as_of_timestamp_str"] = (NOW_TS + timedelta(seconds=1 if change_str == "row_future" else -121)).isoformat()
    else:
        workspace_dict["operations_error_str"] = "Private source failure"
    view_dict = _view_dict(source_tuple)
    pod_dict = view_dict["pod_list"][0]
    assert pod_dict["scheduler_state_str"] == "unk"
    assert pod_dict["broker_str"] == pod_dict["data_str"] == pod_dict["eod_str"] == "—"
    assert "Private" not in str(view_dict)


@pytest.mark.parametrize("field_str", ["last_seen_timestamp_str", "checked_timestamp_str"])
@pytest.mark.parametrize("timestamp_str", ["bad", "2026-09-21T14:00:00", "2026-09-21T14:00:01+00:00"])
def test_invalid_scheduler_timestamps_never_green(source_tuple, field_str, timestamp_str):
    source_tuple[0]["pod_list"][0]["scheduler_dict"][field_str] = timestamp_str
    assert _view_dict(source_tuple)["pod_list"][0]["scheduler_state_str"] == "unk"


def test_no_pods_and_stale_shell_are_not_certified(source_tuple):
    source_tuple[0]["pod_list"] = []
    view_dict = _view_dict(source_tuple)
    assert _rows_dict(view_dict)["schedulers"]["state_str"] == "unk"
    assert view_dict["pod_list"] == [] and view_dict["state_str"] == "unk"
    source_tuple[0]["source_fresh_bool"] = False
    assert _rows_dict(_view_dict(source_tuple))["disk"]["state_str"] == "unk"


def test_missing_required_data_does_not_borrow_current_cycle_green(source_tuple):
    saved_dict = source_tuple[1]["summary_dict"]["pod_row_dict_list"][0]
    saved_dict["data_freshness_dict"]["item_dict_list"].append({"label_str": "Norgate", "severity_str": "green"})
    saved_dict["norgate_snapshot_status_dict"].update(severity_str="red", status_str="failed", snapshot_date_str=None)
    rows_dict = _rows_dict(_view_dict(source_tuple))
    assert rows_dict["norgate"]["state_str"] == rows_dict["market_data"]["state_str"] == "fail"
    assert rows_dict["norgate"]["now_str"] == "Needs action · Data date unknown"


def test_data_requirement_and_fred_status_are_not_invented(source_tuple):
    rows_dict = _rows_dict(_view_dict(source_tuple))
    assert rows_dict["market_data"]["now_str"] == "Have 2026-09-18"
    assert rows_dict["market_data"]["expected_str"] == "Needed 2026-09-18"
    assert rows_dict["fred"]["state_str"] == "skip" and rows_dict["fred"]["checked_bool"] is False
    assert rows_dict["fred"]["now_str"] == "Not checked here"
    assert rows_dict["fred"]["last_str"] == "Last decision used 2026-09-18"
    saved_dict = source_tuple[1]["summary_dict"]["pod_row_dict_list"][0]
    saved_dict["norgate_snapshot_status_dict"]["required_snapshot_date_by_release_dict"] = {}
    saved_dict["dtb3_latest_observation_date_str"] = None
    rows_dict = _rows_dict(_view_dict(source_tuple))
    assert rows_dict["market_data"]["state_str"] == "unk"
    assert rows_dict["fred"]["state_str"] == "skip" and rows_dict["fred"]["last_str"] == "—"


@pytest.mark.parametrize("contradiction_str", ["older_than_required", "explicitly_not_fresh"])
@pytest.mark.parametrize("severity_str,expected_str", [("green", "unk"), ("yellow", "late"), ("red", "fail")])
def test_norgate_date_facts_override_green_but_preserve_existing_problems(source_tuple,
        contradiction_str, severity_str, expected_str):
    saved_dict = source_tuple[1]["summary_dict"]["pod_row_dict_list"][0]
    norgate_dict = saved_dict["norgate_snapshot_status_dict"]
    norgate_dict["severity_str"] = severity_str
    if contradiction_str == "older_than_required":
        norgate_dict["snapshot_date_str"] = "2026-09-17"
    else:
        norgate_dict["snapshot_fresh_for_cycle_bool"] = False
    # Current local snapshot facts win over an older consumed DecisionPlan and
    # V3's green current-cycle continuation allowance.
    saved_dict["latest_decision_norgate_snapshot_date_str"] = "2026-09-18"
    saved_dict["data_freshness_dict"]["item_dict_list"].append({"label_str": "Norgate", "severity_str": "green"})
    view_dict = _view_dict(source_tuple)
    rows_dict = _rows_dict(view_dict)
    assert view_dict["pod_list"][0]["data_state_str"] == expected_str
    assert view_dict["pod_list"][0]["data_str"] == norgate_dict["snapshot_date_str"]
    assert rows_dict["norgate"]["state_str"] == rows_dict["market_data"]["state_str"] == expected_str
    assert rows_dict["market_data"]["expected_str"] == "Needed 2026-09-18"


def test_different_pod_data_dates_use_short_aggregate_wording(source_tuple):
    overview_dict, workspace_dict, source_dict = source_tuple
    pod_dict = deepcopy(overview_dict["pod_list"][0])
    pod_dict.update(pod_id_str="pod2", name_str="Second strategy")
    overview_dict["pod_list"].append(pod_dict)
    workspace_dict["operations_account_list"].append({"pod_id": "pod2", "account_route": "U654321"})
    saved_dict = deepcopy(workspace_dict["summary_dict"]["pod_row_dict_list"][0])
    saved_dict.update(pod_id_str="pod2", account_route_str="U654321", release_id_str="release2")
    saved_dict["norgate_snapshot_status_dict"].update(snapshot_date_str="2026-09-17",
        required_snapshot_date_by_release_dict={"release2": "2026-09-17"})
    workspace_dict["summary_dict"]["pod_row_dict_list"].append(saved_dict)
    release_dict = deepcopy(source_dict["release_list"][0])
    release_dict.update(pod_id_str="pod2", release_id_str="release2", account_str="U···321")
    source_dict["release_list"].append(release_dict)
    rows_dict = _rows_dict(_view_dict(source_tuple))
    assert rows_dict["norgate"]["now_str"] == rows_dict["market_data"]["now_str"] == "Dates vary by Pod"
    assert rows_dict["market_data"]["expected_str"] == "Needed Varies by Pod"
    assert rows_dict["market_data"]["state_str"] == "done"


@pytest.mark.parametrize("scope_obj", [False, None, 1, "true"])
def test_rejected_or_unproven_scope_overrides_cached_green_workspace(source_tuple, scope_obj):
    source_tuple[2]["scope_verified_bool"] = scope_obj
    view_dict = _view_dict(source_tuple)
    pod_dict = view_dict["pod_list"][0]
    rows_dict = _rows_dict(view_dict)
    assert pod_dict["scheduler_state_str"] == pod_dict["data_state_str"] == pod_dict["eod_state_str"] == "unk"
    assert pod_dict["broker_str"] == pod_dict["data_str"] == pod_dict["eod_str"] == "—"
    assert rows_dict["database"]["state_str"] == rows_dict["norgate"]["state_str"] == rows_dict["disk"]["state_str"] == "unk"
    assert view_dict["release_list"] == []
    assert rows_dict["dashboard"]["state_str"] == "done"  # Direct response fact only.


@pytest.mark.parametrize("change_str", ["new_release", "missing", "duplicate", "disabled", "paper", "other_pod"])
def test_saved_rows_require_one_exact_current_enabled_live_release(source_tuple, change_str):
    source_dict = source_tuple[2]
    release_dict = source_dict["release_list"][0]
    if change_str == "new_release":
        release_dict["release_id_str"] = "release2"
    elif change_str == "missing":
        source_dict["release_list"] = []
    elif change_str == "duplicate":
        source_dict["release_list"].append(deepcopy(release_dict))
    elif change_str == "disabled":
        release_dict["enabled_bool"] = False
    elif change_str == "paper":
        release_dict["mode_str"] = "paper"
    else:
        release_dict["pod_id_str"] = "other_pod"
    pod_dict = _view_dict(source_tuple)["pod_list"][0]
    assert pod_dict["scheduler_state_str"] == pod_dict["data_state_str"] == pod_dict["eod_state_str"] == "unk"
    assert pod_dict["data_str"] == pod_dict["broker_str"] == "—"


@pytest.mark.parametrize("severity_str,prefix_str", [
    ("green", "Last decision used"), ("yellow", "Saved warning"), ("red", "Saved failure"),
])
def test_fred_decision_metadata_is_historical_not_current_feed_proof(source_tuple, severity_str, prefix_str):
    saved_dict = source_tuple[1]["summary_dict"]["pod_row_dict_list"][0]
    saved_dict["dtb3_latest_observation_date_str"] = "2026-08-31"
    saved_dict["data_freshness_dict"]["item_dict_list"][0]["severity_str"] = severity_str
    row_dict = _rows_dict(_view_dict(source_tuple))["fred"]
    assert row_dict["state_str"] == "skip" and row_dict["checked_bool"] is False
    assert row_dict["now_str"] == "Not checked here"
    assert row_dict["last_str"].startswith(prefix_str)
    assert "Last decision used 2026-08-31" in row_dict["last_str"]


@pytest.mark.parametrize("state_str,expected_str", [("ok", "done"), ("warning", "late"), ("error", "fail"), ("unknown", "unk"), ("invented", "unk")])
def test_auxiliary_reader_state_contract(source_tuple, state_str, expected_str):
    source_tuple[2]["watchdog_dict"]["state_str"] = state_str
    assert _rows_dict(_view_dict(source_tuple))["watchdog"]["state_str"] == expected_str


@pytest.mark.parametrize("timestamp_str", ["bad", "2026-09-21T14:00:01+00:00", "2026-09-21T14:00:00"])
def test_invalid_auxiliary_observation_is_unknown(source_tuple, timestamp_str):
    source_tuple[2]["watchdog_dict"]["last_timestamp_str"] = timestamp_str
    assert _rows_dict(_view_dict(source_tuple))["watchdog"]["state_str"] == "unk"


def test_stale_auxiliary_collection_never_renews_or_exposes_release_claims(source_tuple):
    source_tuple[2]["checked_timestamp_str"] = (NOW_TS - timedelta(seconds=121)).isoformat()
    view_dict = _view_dict(source_tuple)
    assert _rows_dict(view_dict)["database"]["state_str"] == "unk"
    assert view_dict["release_list"] == []


def test_live_release_projection_and_disk_are_allowlisted(source_tuple):
    source_tuple[2]["release_list"].append({"pod_id_str": "other", "mode_str": "paper", "account_str": "DU123"})
    source_tuple[0]["health_list"][0].update(severity_str="yellow", value_str="76% used", detail_str="private path")
    view_dict = _view_dict(source_tuple)
    assert len(view_dict["release_list"]) == 1
    assert _rows_dict(view_dict)["disk"]["state_str"] == "late"
    assert "private path" not in str(view_dict) and "DU123" not in str(view_dict)
    source_tuple[0]["health_list"][0]["value_str"] = "private path"
    assert _rows_dict(_view_dict(source_tuple))["disk"]["state_str"] == "unk"


def test_naive_assessment_clock_is_rejected(source_tuple):
    with pytest.raises(ValueError, match="aware"):
        _view_dict(source_tuple, as_of_ts=NOW_TS.replace(tzinfo=None))


def _unsupported_receipts(source_tuple):
    for key_str in ("alerts", "deadman"):
        source_tuple[2][key_str + "_dict"] = {"checked_bool": False, "state_str": "unknown"}


@pytest.mark.parametrize("scheduler_state_str", ["sleeping", "holding"])
def test_all_supported_checks_can_be_healthy_with_unsupported_checks_neutral(source_tuple, scheduler_state_str):
    _unsupported_receipts(source_tuple)
    source_tuple[0]["pod_list"][0]["scheduler_dict"]["state_str"] = scheduler_state_str
    view_dict = _view_dict(source_tuple)
    rows_dict = _rows_dict(view_dict)
    assert view_dict["state_str"] == "done"
    assert view_dict["detail_str"] == ""
    assert view_dict["problem_list"] == [] and view_dict["problem_count_int"] == 0
    for key_str in ("gateway", "fred", "alerts", "deadman"):
        assert rows_dict[key_str]["checked_bool"] is False
        assert rows_dict[key_str]["state_str"] == "skip" and rows_dict[key_str]["now_str"] == "Not checked here"
    assert all(row_dict["state_str"] == "done" for row_dict in rows_dict.values() if row_dict["checked_bool"])


@pytest.mark.parametrize("change_str", ["expired", "rejected_scope", "future", "corrupt"])
def test_unsupported_checks_stay_neutral_when_assessment_cannot_be_used(source_tuple, change_str):
    _unsupported_receipts(source_tuple)
    source_dict = source_tuple[2]
    if change_str == "rejected_scope":
        source_dict["scope_verified_bool"] = False
    else:
        source_dict["checked_timestamp_str"] = {"expired": (NOW_TS - timedelta(seconds=120)).isoformat(),
            "future": (NOW_TS + timedelta(seconds=1)).isoformat(), "corrupt": "bad"}[change_str]
    view_dict = _view_dict(source_tuple)
    rows_dict = _rows_dict(view_dict)
    for key_str in ("gateway", "fred", "alerts", "deadman"):
        assert rows_dict[key_str]["state_str"] == "skip" and rows_dict[key_str]["checked_bool"] is False
    assert rows_dict["database"]["state_str"] == "unk" and rows_dict["database"]["checked_bool"] is True
    assert view_dict["state_str"] == "unk"
    assert all(row_dict["key_str"] not in {"gateway", "fred", "alerts", "deadman"} for row_dict in view_dict["problem_list"])


@pytest.mark.parametrize("record_obj", [None, {}, {"checked_bool": True}, {"checked_bool": 0},
    {"checked_bool": "false"}, {"checked_bool": True, "state_str": "error", "last_timestamp_str": "bad"}])
def test_missing_or_corrupt_expected_evidence_stays_checked_and_unknown(source_tuple, record_obj):
    _unsupported_receipts(source_tuple)
    source_tuple[2]["watchdog_dict"] = record_obj
    view_dict = _view_dict(source_tuple)
    row_dict = _rows_dict(view_dict)["watchdog"]
    assert row_dict["state_str"] == "unk" and row_dict["checked_bool"] is True
    assert view_dict["state_str"] == "unk"
    assert view_dict["problem_count_int"] == 1 and view_dict["problem_list"][0]["key_str"] == "watchdog"


@pytest.mark.parametrize("phase_str,label_str", [("eod_snapshot", "Sleeping · next EOD"),
    ("build_decision_plan", "Sleeping · next Decision"), ("invented", "Sleeping"), ("", "Sleeping")])
def test_sleeping_next_phase_is_a_label_not_a_due_time(source_tuple, phase_str, label_str):
    source_tuple[0]["pod_list"][0]["scheduler_dict"]["next_phase_str"] = phase_str
    pod_dict = _view_dict(source_tuple)["pod_list"][0]
    assert pod_dict["scheduler_str"] == label_str
    assert pod_dict["wake_str"] == "10:59:30"


@pytest.mark.parametrize("wake_str,expected_str", [("2026-09-21T23:59:00+00:00", "19:59:00"),
    ("2026-09-22T04:00:00+00:00", "09-22 00:00:00")])
def test_wake_date_is_only_shown_when_et_day_differs(source_tuple, wake_str, expected_str):
    source_tuple[0]["pod_list"][0]["scheduler_dict"]["promised_wake_timestamp_str"] = wake_str
    assert _view_dict(source_tuple)["pod_list"][0]["wake_str"] == expected_str


def test_main_problems_are_worst_first_specific_and_bounded(source_tuple):
    _unsupported_receipts(source_tuple)
    source_tuple[2]["database_dict"].update(state_str="error", now_str="Saved state unavailable")
    source_tuple[0]["health_list"][0].update(severity_str="yellow", value_str="76% used")
    source_tuple[1]["summary_dict"]["pod_row_dict_list"][0]["eod_snapshot_dict"].update(status_str="due_missing", severity_str="yellow")
    source_tuple[2]["watchdog_dict"].update(state_str="unknown", now_str="Report unavailable")
    source_tuple[2]["flex_dict"].update(state_str="unknown", now_str="Report unavailable")
    source_tuple[0]["system_dict"] = {"state_str": "fail", "detail_str": "State DB unavailable · Disk 76% used · EOD Snapshot needs review"}
    view_dict = _view_dict(source_tuple)
    assert view_dict["state_str"] == "fail"
    assert [row_dict["key_str"] for row_dict in view_dict["problem_list"]] == ["database", "disk", "eod:pod1"]
    assert view_dict["problem_count_int"] == 5
    assert view_dict["detail_str"] == "Database Saved state unavailable · Disk 76% used · First strategy EOD Due snapshot missing · 09-18 16:10:00 · and 2 more"
    assert all(row_dict["key_str"] != "system" for row_dict in view_dict["problem_list"])


def test_uncovered_base_system_cause_is_named_and_sorted_before_unknown(source_tuple):
    source_tuple[0]["system_dict"] = {"state_str": "fail", "detail_str": "Pod state needs action"}
    view_dict = _view_dict(source_tuple)
    assert view_dict["problem_list"][0] == {"key_str": "system", "label_str": "System", "state_str": "fail", "detail_str": "Pod state needs action"}
    assert view_dict["detail_str"].startswith("System Pod state needs action")
    assert view_dict["problem_count_int"] == 3


def test_norgate_same_source_problem_is_not_counted_twice(source_tuple):
    _unsupported_receipts(source_tuple)
    source_tuple[1]["summary_dict"]["pod_row_dict_list"][0]["norgate_snapshot_status_dict"]["severity_str"] = "red"
    source_tuple[0]["system_dict"] = {"state_str": "fail", "detail_str": "Norgate needs action"}
    view_dict = _view_dict(source_tuple)
    assert view_dict["problem_count_int"] == 1
    assert view_dict["problem_list"][0]["key_str"] == "norgate"


def test_public_scope_check_is_pure_identity_only(source_tuple):
    original_tuple = deepcopy(source_tuple)
    assert system_scope_matches_bool(*source_tuple) is True
    assert source_tuple == original_tuple
    source_tuple[2]["checked_timestamp_str"] = "bad"
    assert system_scope_matches_bool(*source_tuple) is True  # Caller separately checks age.
    assert _view_dict(source_tuple)["pod_list"][0]["scheduler_state_str"] == "unk"


@pytest.mark.parametrize("change_str", ["release_changed", "release_empty", "release_missing", "release_duplicate", "release_added",
    "overview_removed", "overview_duplicate", "account_empty", "account_changed", "account_missing", "account_duplicate",
    "row_duplicate", "row_missing", "row_paper", "scope_rejected"])
def test_public_scope_check_rejects_changed_or_ambiguous_live_identity(source_tuple, change_str):
    overview_dict, workspace_dict, source_dict = source_tuple
    account_list = workspace_dict["operations_account_list"]
    raw_list = workspace_dict["summary_dict"]["pod_row_dict_list"]
    release_list = source_dict["release_list"]
    if change_str == "release_changed":
        release_list[0]["release_id_str"] = "replacement"
    elif change_str == "release_empty":
        release_list[0]["release_id_str"] = ""
    elif change_str == "release_missing":
        source_dict["release_list"] = []
    elif change_str == "release_duplicate":
        release_list.append(deepcopy(release_list[0]))
    elif change_str == "release_added":
        release_list.append({**release_list[0], "pod_id_str": "new_pod", "release_id_str": "new_release"})
    elif change_str == "overview_removed":
        overview_dict["pod_list"] = []
    elif change_str == "overview_duplicate":
        overview_dict["pod_list"].append(deepcopy(overview_dict["pod_list"][0]))
    elif change_str == "account_empty":
        account_list[0]["account_route"] = raw_list[0]["account_route_str"] = ""
    elif change_str == "account_changed":
        account_list[0]["account_route"] = "OTHER"
    elif change_str == "account_missing":
        workspace_dict["operations_account_list"] = []
    elif change_str == "account_duplicate":
        account_list.append(deepcopy(account_list[0]))
    elif change_str == "row_duplicate":
        raw_list.append(deepcopy(raw_list[0]))
    elif change_str == "row_missing":
        workspace_dict["summary_dict"]["pod_row_dict_list"] = []
    elif change_str == "row_paper":
        raw_list[0]["mode_str"] = "paper"
    else:
        source_dict["scope_verified_bool"] = False
    assert system_scope_matches_bool(*source_tuple) is False
    view_dict = _view_dict(source_tuple)
    assert view_dict["state_str"] == "unk"
    if change_str == "scope_rejected":
        assert view_dict["release_list"] == []
    assert all(pod_dict["scheduler_state_str"] == pod_dict["data_state_str"] == pod_dict["eod_state_str"] == "unk" for pod_dict in view_dict["pod_list"])


def test_public_scope_check_ignores_disabled_and_non_live_release_metadata(source_tuple):
    source_tuple[2]["release_list"].extend([
        {"pod_id_str": "old_pod", "mode_str": "live", "enabled_bool": False, "release_id_str": "old_release"},
        {"pod_id_str": "paper_pod", "mode_str": "paper", "enabled_bool": True, "release_id_str": "paper_release"}])
    assert system_scope_matches_bool(*source_tuple) is True


def test_public_scope_check_allows_verified_disabled_only_metadata_without_runtime_rows(source_tuple):
    source_tuple[0]["pod_list"] = []
    source_tuple[1].clear()
    source_tuple[2]["release_list"][0]["enabled_bool"] = False
    assert system_scope_matches_bool(*source_tuple) is True
    view_dict = _view_dict(source_tuple)
    assert view_dict["pod_list"] == [] and len(view_dict["release_list"]) == 1
    source_tuple[2]["scope_verified_bool"] = False
    assert system_scope_matches_bool(*source_tuple) is False


def test_public_scope_check_rejects_two_pods_sharing_one_account(source_tuple):
    overview_dict, workspace_dict, source_dict = source_tuple
    overview_dict["pod_list"].append({**overview_dict["pod_list"][0], "pod_id_str": "second_pod"})
    workspace_dict["operations_account_list"].append({**workspace_dict["operations_account_list"][0], "pod_id": "second_pod"})
    workspace_dict["summary_dict"]["pod_row_dict_list"].append({**workspace_dict["summary_dict"]["pod_row_dict_list"][0],
        "pod_id_str": "second_pod", "release_id_str": "second_release"})
    source_dict["release_list"].append({**source_dict["release_list"][0], "pod_id_str": "second_pod", "release_id_str": "second_release"})
    assert system_scope_matches_bool(*source_tuple) is False


@pytest.mark.parametrize("change_str", ["changed_release", "stale_operations", "stale_overview", "operations_error"])
def test_auxiliary_results_require_the_same_fresh_operations_scope(source_tuple, change_str):
    _unsupported_receipts(source_tuple)
    overview_dict, workspace_dict, source_dict = source_tuple
    source_dict["watchdog_dict"].update(state_str="error", now_str="Current report failed")
    if change_str == "changed_release":
        source_dict["release_list"][0]["release_id_str"] = "new_release"
    elif change_str == "stale_operations":
        workspace_dict["summary_dict"]["as_of_timestamp_str"] = (NOW_TS - timedelta(seconds=120)).isoformat()
    elif change_str == "stale_overview":
        overview_dict["source_fresh_bool"] = False
    else:
        workspace_dict["operations_error_str"] = "Unavailable"
    original_tuple = deepcopy(source_tuple)
    view_dict = _view_dict(source_tuple)
    rows_dict = _rows_dict(view_dict)
    assert view_dict["state_str"] == "unk"
    for key_str in ("watchdog", "flex", "database", "event_log"):
        assert rows_dict[key_str]["state_str"] == "unk" and rows_dict[key_str]["now_str"] == "Unknown"
    for key_str in ("gateway", "fred", "alerts", "deadman"):
        assert rows_dict[key_str]["state_str"] == "skip" and rows_dict[key_str]["checked_bool"] is False
    assert len(view_dict["release_list"]) == 1  # Verified configuration remains useful when runtime evidence is unavailable.
    assert view_dict["release_list"][0]["release_id_str"] == source_dict["release_list"][0]["release_id_str"]
    assert source_tuple == original_tuple
