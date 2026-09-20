"""A saved cycle never hides the Pod's current operational warning."""

from copy import deepcopy
from datetime import timedelta

import pytest

from alpha.live.dashboard_v4.app import create_app
from alpha.live.dashboard_v4.demo import DEMO_NOW_TS, build_demo_workspace_tuple, create_demo_app
from alpha.live.dashboard_v4.overview import build_overview_dict
from alpha.live.dashboard_v4.pod import build_pod_page_dict, build_evidence_tables_dict


@pytest.mark.parametrize("query_str", ["", "?cycle=vplan:1", "?cycle=vplan:2&tab=orders"])
@pytest.mark.parametrize("failure_str", ["gate", "database"])
def test_current_warning_survives_entering_pod_and_history(query_str, failure_str):
    workspace_dict, snapshot_obj, provider_obj = build_demo_workspace_tuple()
    row_dict = workspace_dict["summary_dict"]["pod_row_dict_list"][0]
    if failure_str == "gate":
        row_dict["required_action_dict"] = {"severity_str": "red", "label_str": "Review DecisionPlan gate", "reason_str": "Decision approval is missing."}
        expected_str = "Review DecisionPlan gate"
    else:
        row_dict["db_status_str"] = "error"
        expected_str = "State DB unavailable."
    app_obj = create_app(provider_obj, workspace_snapshot_fn=lambda: (deepcopy(workspace_dict), snapshot_obj), now_fn=lambda: DEMO_NOW_TS)
    client_obj = app_obj.test_client()
    assert expected_str in client_obj.get('/').get_data(as_text=True)
    html_str = client_obj.get('/pods/' + row_dict["pod_id_str"] + query_str).get_data(as_text=True)
    assert expected_str in html_str and 'aria-label="Pod needs attention"' in html_str
    assert 'data-pill-label>Action needed</span>' in html_str


def test_old_healthy_qpi_cycle_cannot_replace_current_failed_header():
    workspace_dict, snapshot_obj, provider_obj = build_demo_workspace_tuple()
    pod_id_str = workspace_dict["operations_account_list"][1]["pod_id"]
    overview_dict = build_overview_dict(workspace_dict, snapshot_obj, provider_obj, as_of_ts=DEMO_NOW_TS)
    source_dict = provider_obj.get_pod_cycles_dict(pod_id_str, as_of_ts=DEMO_NOW_TS, vplan_id_int=1)
    view_dict = build_pod_page_dict(overview_dict, source_dict, {}, pod_id_str=pod_id_str, as_of_ts=DEMO_NOW_TS)
    assert view_dict["state_str"] == "fail" and view_dict["pill_str"] == "Action needed"
    assert view_dict["cycle_state_str"] == "done" and view_dict["cycle_pill_str"] == "On track"
    assert view_dict["attention_dict"]["title_str"] == "Review broker ACK"


def test_monthly_idle_header_does_not_become_a_completed_cycle_status():
    workspace_dict, snapshot_obj, provider_obj = build_demo_workspace_tuple()
    pod_id_str = workspace_dict["operations_account_list"][2]["pod_id"]
    overview_dict = build_overview_dict(workspace_dict, snapshot_obj, provider_obj, as_of_ts=DEMO_NOW_TS)
    source_dict = provider_obj.get_pod_cycles_dict(pod_id_str, as_of_ts=DEMO_NOW_TS)
    view_dict = build_pod_page_dict(overview_dict, source_dict, {}, pod_id_str=pod_id_str, as_of_ts=DEMO_NOW_TS)
    assert view_dict["pill_str"] == "Waiting"
    assert view_dict["cycle_pill_str"] == "On track"


def test_current_critical_ack_can_weaken_an_idle_header():
    workspace_dict, snapshot_obj, provider_obj = build_demo_workspace_tuple()
    pod_id_str = workspace_dict["operations_account_list"][2]["pod_id"]
    overview_dict = build_overview_dict(workspace_dict, snapshot_obj, provider_obj, as_of_ts=DEMO_NOW_TS)
    source_dict = provider_obj.get_pod_cycles_dict(pod_id_str, as_of_ts=DEMO_NOW_TS)
    source_dict["ack_list"][0]["ack_status_str"] = "missing_critical"
    view_dict = build_pod_page_dict(overview_dict, source_dict, {}, pod_id_str=pod_id_str, as_of_ts=DEMO_NOW_TS)
    assert view_dict["state_str"] == "fail" and view_dict["pill_str"] == "Action needed"
    assert view_dict["attention_dict"]["title_str"] == "Review broker ACK"
    assert view_dict["header_dict"]["next_str"] == "Review saved evidence"
    assert view_dict["header_dict"]["next_forecast_bool"] is False


def test_submit_time_is_last_verified_broker_ack_not_mutable_order_time():
    workspace_dict, snapshot_obj, provider_obj = build_demo_workspace_tuple()
    pod_id_str = workspace_dict["operations_account_list"][0]["pod_id"]
    overview_dict = build_overview_dict(workspace_dict, snapshot_obj, provider_obj, as_of_ts=DEMO_NOW_TS)
    source_dict = provider_obj.get_pod_cycles_dict(pod_id_str, as_of_ts=DEMO_NOW_TS)
    for order_dict in source_dict["order_list"]:
        order_dict["submitted_timestamp_str"] = DEMO_NOW_TS.isoformat()
    source_dict["ack_list"][-1]["response_timestamp_str"] = (DEMO_NOW_TS - timedelta(minutes=1)).isoformat()
    view_dict = build_pod_page_dict(overview_dict, source_dict, {}, pod_id_str=pod_id_str, as_of_ts=DEMO_NOW_TS)
    assert view_dict["step_list"][3]["actual_time_str"] == "09:40:07"
    assert view_dict["step_list"][3]["fact_str"] == "All orders acknowledged"
    assert view_dict["verdict_detail_str"].startswith("Cycle next: EOD")


def test_event_table_keeps_symbol_and_latest_first():
    source_dict = {"event_list": [
        {"asset_str": "AMD", "status_str": "Submitted", "event_timestamp_str": (DEMO_NOW_TS-timedelta(minutes=4)).isoformat()},
        {"asset_str": "CRM", "status_str": "Filled", "event_timestamp_str": DEMO_NOW_TS.isoformat()}]}
    table_dict = build_evidence_tables_dict(source_dict, as_of_ts=DEMO_NOW_TS, fresh_bool=True)["events"]
    assert table_dict["column_list"] == ["Time", "Symbol", "What"]
    assert table_dict["row_list"][0]["cell_list"] == ["09:41:07", "CRM", "Order filled"]


def test_ack_timestamp_at_planned_boundary_is_not_displayed_as_actual():
    workspace_dict, snapshot_obj, provider_obj = build_demo_workspace_tuple()
    pod_id_str = workspace_dict["operations_account_list"][0]["pod_id"]
    overview_dict = build_overview_dict(workspace_dict, snapshot_obj, provider_obj, as_of_ts=DEMO_NOW_TS)
    source_dict = provider_obj.get_pod_cycles_dict(pod_id_str, as_of_ts=DEMO_NOW_TS)
    for ack_dict in source_dict["ack_list"]:
        ack_dict["response_timestamp_str"] = source_dict["pod_row_dict"]["latest_vplan_submission_timestamp_str"]
    view_dict = build_pod_page_dict(overview_dict, source_dict, {}, pod_id_str=pod_id_str, as_of_ts=DEMO_NOW_TS)
    assert view_dict["step_list"][3]["state_str"] == "Done"
    assert not view_dict["step_list"][3]["actual_time_str"]
    assert {row_dict["cell_list"][4] for row_dict in view_dict["tables_dict"]["orders"]["row_list"]} == {"—"}


def test_unconnected_controls_are_hidden_and_mobile_overview_is_not_selected():
    app_obj = create_demo_app()
    html_str = app_obj.test_client().get('/pods/demo_1_0').get_data(as_text=True)
    for text_str in ("Trade sheet", "Tools for this step", "Live vs backtest unavailable", "Slip bps", '>Files</a>'):
        assert text_str not in html_str
    mobile_str = html_str.split('aria-label="Main on mobile"')[1]
    assert 'aria-current="page"' not in mobile_str


@pytest.mark.parametrize("query_str,current_bool", [("", True), ("?cycle=vplan:2", True), ("?cycle=vplan:1", False)])
def test_reader_failure_weakens_only_current_selection(monkeypatch, query_str, current_bool):
    workspace_dict, snapshot_obj, provider_obj = build_demo_workspace_tuple()
    monkeypatch.setattr(provider_obj, "get_pod_cycles_dict", lambda *args, **kwargs: {"status_str": "unknown"})
    app_obj = create_app(provider_obj, workspace_snapshot_fn=lambda: (deepcopy(workspace_dict), snapshot_obj), now_fn=lambda: DEMO_NOW_TS)
    html_str = app_obj.test_client().get('/pods/demo_1_0' + query_str).get_data(as_text=True)
    heading_str = html_str.split('class="page-head pod-head"')[1].split('aria-label="Selected cycle"')[0]
    assert ('Current cycle unavailable.' in heading_str) is current_bool
    assert ('data-pill-label>On track</span>' in heading_str) is not current_bool
