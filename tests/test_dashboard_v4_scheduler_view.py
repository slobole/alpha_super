"""Scheduler liveness changes present advice, never saved trading evidence."""

from copy import deepcopy
from datetime import timedelta
from types import SimpleNamespace
import json

import pytest

from alpha.live.dashboard_v4.app import create_app
from alpha.live.dashboard_v4.data import LiveDataProvider
from alpha.live.dashboard_v4.demo import DEMO_NOW_TS, build_demo_workspace_tuple
from alpha.live.dashboard_v4.overview import build_overview_dict
from alpha.live.dashboard_v4.pod import build_pod_page_dict
from alpha.live.dashboard_v4.scheduler_view import apply_scheduler_to_steps, scheduler_note_str
from alpha.live.dashboard_v4.scheduler_status import load_scheduler_status_dict
from alpha.live.logging_utils import build_structured_event_record_dict


@pytest.fixture
def fixture_tuple():
    workspace_dict, snapshot_obj, provider_obj = build_demo_workspace_tuple()
    try:
        yield workspace_dict, snapshot_obj, provider_obj
    finally:
        provider_obj.close()


def _status_dict(state_str):
    return {"state_str": state_str,
        "alive_bool": True if state_str in {"sleeping", "holding", "error"} else False if state_str == "stopped" else None,
        "last_seen_timestamp_str": (DEMO_NOW_TS - timedelta(minutes=8)).isoformat(),
        "promised_wake_timestamp_str": (DEMO_NOW_TS - timedelta(minutes=7)).isoformat(),
        "next_phase_str": "manual_review_pending" if state_str == "holding" else "post_execution_reconcile",
        "reason_code_str": "manual_review_required" if state_str == "holding" else "ready_to_reconcile"}


def _set_status(provider_obj, pod_id_str, state_str):
    original_fn = provider_obj.get_scheduler_status_dict
    provider_obj.get_scheduler_status_dict = lambda requested_str, *, as_of_ts: (
        _status_dict(state_str) if requested_str == pod_id_str else original_fn(requested_str, as_of_ts=as_of_ts))


@pytest.mark.parametrize("state_str,title_str,tone_str", [
    ("stopped", "Scheduler not responding", "fail"),
    ("error", "Scheduler error", "fail"),
    ("late", "Scheduler check overdue", "late"),
])
def test_scheduler_problem_changes_only_affected_current_pod(fixture_tuple, state_str, title_str, tone_str):
    workspace_dict, snapshot_obj, provider_obj = fixture_tuple
    before_dict = build_overview_dict(workspace_dict, snapshot_obj, provider_obj, as_of_ts=DEMO_NOW_TS)
    _set_status(provider_obj, "demo_1_2", state_str)
    view_dict = build_overview_dict(workspace_dict, snapshot_obj, provider_obj, as_of_ts=DEMO_NOW_TS)
    for index_int in (0, 1, 3):
        assert view_dict["pod_list"][index_int] == before_dict["pod_list"][index_int]
    pod_dict = view_dict["pod_list"][2]
    assert pod_dict["state_str"] == tone_str
    assert pod_dict["now_str"] == title_str
    assert pod_dict["next_str"] == "Check scheduler"
    assert pod_dict["next_forecast_bool"] is False and pod_dict["next_timestamp_str"] == ""
    assert pod_dict["step_list"][-1]["state_str"] == "unk"
    assert view_dict["system_dict"]["detail_str"] == "Scheduler"
    warning_list = [item_dict for item_dict in view_dict["attention_list"] if item_dict["pod_id_str"] == "demo_1_2"]
    assert len(warning_list) == 1
    assert "did not run" not in warning_list[0]["detail_str"]


def test_unknown_scheduler_does_not_downgrade_waiting_pod(fixture_tuple, monkeypatch):
    workspace_dict, snapshot_obj, provider_obj = fixture_tuple
    workspace_dict["operations_account_list"] = workspace_dict["operations_account_list"][2:]
    monkeypatch.setattr("alpha.live.dashboard_v4.overview.build_health_rollup", lambda *args, **kwargs: SimpleNamespace(severity_str="green"))
    _set_status(provider_obj, "demo_1_2", "unknown")
    view_dict = build_overview_dict(workspace_dict, snapshot_obj, provider_obj, as_of_ts=DEMO_NOW_TS)
    assert view_dict["system_dict"] == {"state_str": "unk", "label_str": "System unknown", "detail_str": "Scheduler unknown"}
    assert view_dict["pod_list"][0]["pill_str"] == "Waiting"
    assert view_dict["pod_list"][0]["next_str"] == "EOD"
    assert view_dict["attention_list"] == []


def test_stopped_scheduler_preserves_underlying_order_issue_and_one_attention_row(fixture_tuple):
    workspace_dict, snapshot_obj, provider_obj = fixture_tuple
    _set_status(provider_obj, "demo_1_1", "stopped")
    view_dict = build_overview_dict(workspace_dict, snapshot_obj, provider_obj, as_of_ts=DEMO_NOW_TS)
    assert len(view_dict["attention_list"]) == 1
    warning_dict = view_dict["attention_list"][0]
    assert warning_dict["title_str"] == "Scheduler not responding"
    assert "1 of 3 orders has no saved broker ACK" in warning_dict["detail_str"]
    assert view_dict["pod_list"][1]["step_list"][3]["state_str"] == "fail"


@pytest.mark.parametrize("historical_bool", [False, True])
def test_current_scheduler_warning_cannot_rewrite_saved_cycle(fixture_tuple, historical_bool):
    workspace_dict, snapshot_obj, provider_obj = fixture_tuple
    source_dict = provider_obj.get_pod_cycles_dict("demo_1_0", as_of_ts=DEMO_NOW_TS, vplan_id_int=1 if historical_bool else 2)
    before_dict = build_pod_page_dict(build_overview_dict(workspace_dict, snapshot_obj, provider_obj, as_of_ts=DEMO_NOW_TS),
        source_dict, {}, pod_id_str="demo_1_0", as_of_ts=DEMO_NOW_TS)
    _set_status(provider_obj, "demo_1_0", "stopped")
    overview_dict = build_overview_dict(workspace_dict, snapshot_obj, provider_obj, as_of_ts=DEMO_NOW_TS)
    view_dict = build_pod_page_dict(overview_dict, source_dict, {}, pod_id_str="demo_1_0", as_of_ts=DEMO_NOW_TS)
    assert view_dict["header_dict"]["verdict_str"] == "Scheduler not responding."
    assert view_dict["attention_dict"]["title_str"] == "Scheduler not responding"
    if historical_bool:
        assert view_dict["step_list"] == before_dict["step_list"]
        assert view_dict["cycle_state_str"] == before_dict["cycle_state_str"]
        assert view_dict["verdict_str"] == before_dict["verdict_str"]
    else:
        assert view_dict["step_list"][:6] == before_dict["step_list"][:6]
        assert view_dict["step_list"][-1]["state_str"] == "Unknown"


@pytest.mark.parametrize("state_str,expected_str", [
    ("holding", "It waits for operator action."),
    ("sleeping", "It is waiting to check execution."),
    ("unknown", ""),
])
def test_issue_sentence_uses_scheduler_state_not_missing_ack(fixture_tuple, state_str, expected_str):
    workspace_dict, snapshot_obj, provider_obj = fixture_tuple
    _set_status(provider_obj, "demo_1_1", state_str)
    overview_dict = build_overview_dict(workspace_dict, snapshot_obj, provider_obj, as_of_ts=DEMO_NOW_TS)
    warning_dict = overview_dict["attention_list"][0]
    assert warning_dict["title_str"] == "Review broker ACK"
    assert warning_dict.get("scheduler_note_str", "").endswith(expected_str)
    if not expected_str:
        assert warning_dict.get("scheduler_note_str", "") == ""


def test_source_staleness_wins_and_prevents_scheduler_acquisition(fixture_tuple):
    workspace_dict, snapshot_obj, provider_obj = fixture_tuple
    workspace_dict["summary_dict"]["as_of_timestamp_str"] = (DEMO_NOW_TS - timedelta(seconds=121)).isoformat()
    provider_obj.get_scheduler_status_dict = lambda *args, **kwargs: pytest.fail("Do not read unverified Pod sources")
    view_dict = build_overview_dict(workspace_dict, snapshot_obj, provider_obj, as_of_ts=DEMO_NOW_TS)
    assert view_dict["attention_list"] == []
    assert all(pod_dict["pill_str"] == "Unknown" for pod_dict in view_dict["pod_list"])


def test_foreign_account_never_loads_scheduler_evidence(fixture_tuple):
    workspace_dict, snapshot_obj, provider_obj = fixture_tuple
    workspace_dict["operations_account_list"] = workspace_dict["operations_account_list"][:1]
    workspace_dict["summary_dict"]["pod_row_dict_list"][0]["account_route_str"] = "FOREIGN"
    provider_obj.get_scheduler_status_dict = lambda *args, **kwargs: pytest.fail("Foreign identity")
    view_dict = build_overview_dict(workspace_dict, snapshot_obj, provider_obj, as_of_ts=DEMO_NOW_TS)
    assert view_dict["pod_list"][0]["state_str"] == "unk"


def test_healthy_pages_have_no_scheduler_widget_and_issue_has_only_sentence(fixture_tuple):
    workspace_dict, snapshot_obj, provider_obj = fixture_tuple
    app_obj = create_app(provider_obj, workspace_snapshot_fn=lambda: (deepcopy(workspace_dict), snapshot_obj), now_fn=lambda: DEMO_NOW_TS)
    client_obj = app_obj.test_client()
    for url_str in ("/", "/pods/demo_1_0"):
        assert "scheduler" not in client_obj.get(url_str).get_data(as_text=True).lower()
    html_str = client_obj.get("/pods/demo_1_1").get_data(as_text=True)
    assert "The scheduler is alive. It is waiting to check execution." in html_str
    assert "Last check" not in html_str and "Next check" not in html_str


def test_scheduler_command_is_scoped_quoted_and_copy_only(monkeypatch):
    provider_obj = LiveDataProvider(releases_root_path_str="C:/owner's releases", event_log_path_str="C:/logs/events.jsonl")
    target_obj = SimpleNamespace(release_obj=SimpleNamespace(mode_str="live", enabled_bool=True), db_path_str="C:/state/pod.sqlite3")
    monkeypatch.setattr(provider_obj, "get_target_for_pod", lambda pod_id_str: target_obj)
    captured_list = []
    def load_status(event_log_path_str, pod_id_str, *, as_of_ts):
        captured_list.append((event_log_path_str, pod_id_str, as_of_ts))
        return _status_dict("stopped")
    monkeypatch.setattr("alpha.live.dashboard_v4.data.load_scheduler_status_dict", load_status)
    status_dict = provider_obj.get_scheduler_status_dict("pod_one", as_of_ts=DEMO_NOW_TS)
    assert captured_list == [("C:/logs/events.jsonl", "pod_one", DEMO_NOW_TS)]
    assert "'next_due' '--mode' 'live' '--pod-id' 'pod_one'" in status_dict["check_command_str"]
    assert "'C:/owner''s releases'" in status_dict["check_command_str"]
    assert "'--db-path' 'C:/state/pod.sqlite3'" in status_dict["check_command_str"]
    assert "serve" not in status_dict["check_command_str"]


@pytest.mark.parametrize("mode_str,enabled_bool", [("paper", True), ("incubation", True), ("live", False)])
def test_unapproved_modes_and_disabled_targets_do_not_read_logs(monkeypatch, mode_str, enabled_bool):
    provider_obj = LiveDataProvider()
    monkeypatch.setattr(provider_obj, "get_target_for_pod", lambda pod_id_str: SimpleNamespace(release_obj=SimpleNamespace(mode_str=mode_str, enabled_bool=enabled_bool)))
    monkeypatch.setattr("alpha.live.dashboard_v4.data.load_scheduler_status_dict", lambda *args, **kwargs: pytest.fail("Unexpected log read"))
    assert provider_obj.get_scheduler_status_dict("pod_one", as_of_ts=DEMO_NOW_TS)["state_str"] == "unknown"


def test_unavailable_scheduler_only_changes_unperformed_planned_steps():
    step_list = [{"state_str": state_str, "fact_str": "saved"} for state_str in ("Done", "Failed", "Late", "None", "Planned")]
    apply_scheduler_to_steps(step_list, _status_dict("stopped"))
    assert [step_dict["state_str"] for step_dict in step_list] == ["Done", "Failed", "Late", "None", "Unknown"]
    assert all(step_dict["fact_str"] == "saved" for step_dict in step_list[:-1])


def test_waiting_for_data_sentence_uses_verified_state():
    assert scheduler_note_str({"alive_bool": True, "reason_code_str": "snapshot_not_ready"}) == "The scheduler is alive. It waits for data."


@pytest.mark.parametrize("flag_str", ["norgate_snapshot_sync_active_wait_bool", "norgate_snapshot_sync_blocks_decision_plan_bool"])
def test_real_data_wait_flags_flow_from_reader_to_issue_banner(tmp_path, fixture_tuple, flag_str):
    workspace_dict, snapshot_obj, provider_obj = fixture_tuple
    log_path_obj = tmp_path / "events.jsonl"
    record_dict = build_structured_event_record_dict("scheduler_sleeping", {
        "env_mode_str": "live", "related_pod_id_list": ["demo_1_1"], "next_phase_str": "post_execution_reconcile",
        "reason_code_str": "ready_to_reconcile", "sleep_seconds_float": 30, flag_str: True}, timestamp_obj=DEMO_NOW_TS)
    log_path_obj.write_text(json.dumps(record_dict) + "\n", encoding="utf-8")
    status_dict = load_scheduler_status_dict(str(log_path_obj), "demo_1_1", as_of_ts=DEMO_NOW_TS)
    provider_obj.get_scheduler_status_dict = lambda pod_id_str, *, as_of_ts: status_dict
    app_obj = create_app(provider_obj, workspace_snapshot_fn=lambda: (deepcopy(workspace_dict), snapshot_obj), now_fn=lambda: DEMO_NOW_TS)
    html_str = app_obj.test_client().get("/pods/demo_1_1").get_data(as_text=True)
    assert "The scheduler is alive. It waits for data." in html_str
    assert "It is checking execution" not in html_str


def test_detail_only_current_issue_includes_current_scheduler_explanation(fixture_tuple):
    workspace_dict, snapshot_obj, provider_obj = fixture_tuple
    _set_status(provider_obj, "demo_1_2", "holding")
    overview_dict = build_overview_dict(workspace_dict, snapshot_obj, provider_obj, as_of_ts=DEMO_NOW_TS)
    source_dict = provider_obj.get_pod_cycles_dict("demo_1_2", as_of_ts=DEMO_NOW_TS)
    source_dict["ack_list"][0]["ack_status_str"] = "missing_critical"
    view_dict = build_pod_page_dict(overview_dict, source_dict, {}, pod_id_str="demo_1_2", as_of_ts=DEMO_NOW_TS)
    assert view_dict["attention_dict"]["title_str"] == "Review broker ACK"
    assert view_dict["attention_dict"]["scheduler_note_str"] == "The scheduler is alive. It waits for operator action."
