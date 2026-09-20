"""Current readiness must not replace the selected cycle's saved data session."""

from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path

from flask import template_rendered
import pytest

from alpha.live.dashboard_v4.app import create_app
from alpha.live.dashboard_v4.demo import build_demo_workspace_tuple
from test_dashboard_v4_evidence import update_db


NOW_TS = datetime(2026, 9, 20, 14, tzinfo=UTC)


@pytest.fixture
def monthly_page_tuple():
    # DemoPodStore writes real production-format SQLite in its owned temporary
    # directory. The September 1 monthly cycle is still the latest on Sep 20.
    workspace_dict, snapshot_obj, provider_obj = build_demo_workspace_tuple()
    account_dict = workspace_dict["operations_account_list"][2]
    row_dict = workspace_dict["summary_dict"]["pod_row_dict_list"][2]
    workspace_dict["operations_account_list"] = [account_dict]
    workspace_dict["summary_dict"].update(as_of_timestamp_str=NOW_TS.isoformat(), pod_row_dict_list=[row_dict])
    row_dict["as_of_timestamp_str"] = NOW_TS.isoformat()
    row_dict["norgate_snapshot_status_dict"] = {
        "status_str": "ready", "snapshot_date_str": "2026-09-16", "snapshot_fresh_for_cycle_bool": True}
    row_dict["eod_snapshot_dict"].update(status_str="not_applicable", expected_due_timestamp_str=None,
        last_required_market_date_str="2026-09-18", last_required_eod_present_bool=True)
    app_obj = create_app(provider_obj, demo_bool=True,
        workspace_snapshot_fn=lambda: (deepcopy(workspace_dict), snapshot_obj), now_fn=lambda: NOW_TS)
    context_list = []

    def capture_context(sender_obj, **render_dict):
        context_list.append(render_dict["context"])

    template_rendered.connect(capture_context, app_obj)
    target_obj = provider_obj.get_target_for_pod(account_dict["pod_id"])
    try:
        yield app_obj.test_client(), row_dict, target_obj, context_list
    finally:
        template_rendered.disconnect(capture_context, app_obj)
        provider_obj.close()


def _page_context_dict(monthly_page_tuple, suffix_str="", query_str=""):
    client_obj, row_dict, target_obj, context_list = monthly_page_tuple
    db_path_obj = Path(target_obj.db_path_str)
    before_bytes, before_mtime_int = db_path_obj.read_bytes(), db_path_obj.stat().st_mtime_ns
    response_obj = client_obj.get("/pods/" + row_dict["pod_id_str"] + suffix_str + query_str)
    assert response_obj.status_code == 200
    assert db_path_obj.read_bytes() == before_bytes
    assert db_path_obj.stat().st_mtime_ns == before_mtime_int
    return context_list[-1]


@pytest.mark.parametrize("suffix_str,query_str", [("", ""), ("", "?cycle=vplan:2"), ("/refresh", "?cycle=vplan:2")])
def test_latest_monthly_cycle_keeps_its_saved_data_date(monthly_page_tuple, suffix_str, query_str):
    context_dict = _page_context_dict(monthly_page_tuple, suffix_str, query_str)
    page_dict = context_dict["pod_page_dict"]
    assert page_dict["selected_cycle_dict"]["current_bool"] is True
    assert page_dict["selected_cycle_dict"]["session_date_str"] == "2026-09-01"
    assert page_dict["step_list"][0]["state_str"] == "Done"
    assert page_dict["step_list"][0]["fact_str"] == "2026-08-31"
    assert page_dict["header_dict"]["pill_str"] == "Waiting"
    assert monthly_page_tuple[1]["norgate_snapshot_status_dict"]["snapshot_date_str"] == "2026-09-16"


@pytest.mark.parametrize("failed_bool", [False, True])
def test_current_data_problem_stays_in_header_without_rewriting_saved_cycle(monthly_page_tuple, failed_bool):
    row_dict = monthly_page_tuple[1]
    row_dict["norgate_snapshot_status_dict"].update(
        status_str="failed" if failed_bool else "unknown", snapshot_fresh_for_cycle_bool=False)
    if failed_bool:
        row_dict["required_action_dict"] = {"severity_str": "red", "label_str": "Review data",
            "reason_str": "Current snapshot failed validation."}
    context_dict = _page_context_dict(monthly_page_tuple)
    page_dict = context_dict["pod_page_dict"]
    assert page_dict["step_list"][0]["state_str"] == "Done"
    assert page_dict["step_list"][0]["fact_str"] == "2026-08-31"
    expected_pill_str = "Action needed" if failed_bool else "Unknown"
    assert context_dict["overview_dict"]["pod_list"][0]["pill_str"] == expected_pill_str
    assert page_dict["header_dict"]["pill_str"] == expected_pill_str
    if failed_bool:
        assert page_dict["attention_dict"]["title_str"] == "Review data"
        assert page_dict["attention_dict"]["detail_str"] == "Current snapshot failed validation."


def test_current_ready_snapshot_cannot_fill_missing_saved_decision_metadata(monthly_page_tuple):
    update_db(monthly_page_tuple[2], "UPDATE decision_plan SET snapshot_metadata_json_str='{}' WHERE decision_plan_id_int=2")
    context_dict = _page_context_dict(monthly_page_tuple, query_str="?cycle=vplan:2")
    page_dict = context_dict["pod_page_dict"]
    assert page_dict["step_list"][0]["state_str"] == "Unknown"
    assert page_dict["step_list"][0]["fact_str"] == "Freshness not verified"
    assert page_dict["step_list"][1]["state_str"] == "Done"
    assert page_dict["header_dict"]["pill_str"] == "Waiting"
