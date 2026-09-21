"""Every V4 page shares service health and the same evidence expiry boundary."""

from copy import deepcopy
from datetime import timedelta
from zoneinfo import ZoneInfo

from flask import template_rendered
import pytest

from alpha.live.dashboard_v4.app import create_app
from alpha.live.dashboard_v4.demo import DEMO_NOW_TS, build_demo_workspace_tuple


PATH_LIST = ["/", "/overview/refresh", "/pods/demo_1_0", "/pods/demo_1_0/refresh",
             "/positions", "/positions/refresh", "/performance", "/performance/refresh",
             "/performance/status", "/activity", "/activity/refresh", "/system", "/system/refresh"]


@pytest.fixture
def shared_tuple(monkeypatch):
    workspace_dict, snapshot_obj, provider_obj = build_demo_workspace_tuple(include_holdings_bool=True)
    clock_dict = {"now_ts": DEMO_NOW_TS}
    original_source_fn = provider_obj.get_system_source_dict
    control_dict = {"delay_int": 0, "scope_case_str": "valid"}

    def source_fn(current_workspace_dict, *, as_of_ts):
        source_dict = original_source_fn(current_workspace_dict, as_of_ts=as_of_ts)
        source_dict["watchdog_dict"].update(state_str="error", now_str="Saved watchdog failure")
        if control_dict["scope_case_str"] == "changed_release":
            source_dict["release_list"][0]["release_id_str"] = "different_release"
        elif control_dict["scope_case_str"] == "rejected":
            source_dict["scope_verified_bool"] = False
        clock_dict["now_ts"] += timedelta(seconds=control_dict["delay_int"])
        return source_dict

    def forbidden_probe(*args_tuple, **kwargs_dict):
        pytest.fail("Demo reached a real host/service source")

    provider_obj.get_system_source_dict = source_fn
    monkeypatch.setattr("alpha.live.dashboard_v4.overview.build_health_rollup", forbidden_probe)
    monkeypatch.setattr("alpha.live.dashboard_v4.app.load_system_source_dict", forbidden_probe)
    app_obj = create_app(provider_obj, demo_bool=True,
        workspace_snapshot_fn=lambda: (deepcopy(workspace_dict), snapshot_obj),
        operations_workspace_fn=lambda: deepcopy(workspace_dict), now_fn=lambda: clock_dict["now_ts"])
    context_list = []

    def capture_fn(sender_obj, template, context, **extra_dict):
        context_list.append(context)

    template_rendered.connect(capture_fn, app_obj, weak=False)
    yield app_obj.test_client(), context_list, control_dict
    template_rendered.disconnect(capture_fn, app_obj)
    provider_obj.close()


def test_all_headers_use_the_same_saved_service_assessment(shared_tuple, monkeypatch):
    from alpha.live.dashboard_v4.overview import build_overview_dict

    def healthy_operations_dict(*args_tuple, **kwargs_dict):
        overview_dict = build_overview_dict(*args_tuple, **kwargs_dict)
        # Isolate a service-only failure: the incoming operating assessment
        # is healthy before the shared watchdog evidence is incorporated.
        overview_dict.update(live_state_str="done", attention_list=[],
            system_dict={"state_str": "done", "label_str": "System OK", "detail_str": ""})
        for pod_dict in overview_dict["pod_list"]:
            pod_dict["state_str"] = "done"
        return overview_dict

    monkeypatch.setattr("alpha.live.dashboard_v4.app.build_overview_dict", healthy_operations_dict)
    client_obj, context_list, _ = shared_tuple
    header_list = []
    for path_str in PATH_LIST:
        response_obj = client_obj.get(path_str)
        assert response_obj.status_code == 200
        overview_dict = context_list[-1]["overview_dict"]
        header_list.append(overview_dict["system_dict"])
        assert overview_dict["source_fresh_bool"] is True
        assert overview_dict["source_valid_ms_int"] == 120_000
        assert overview_dict["live_state_str"] == "fail"
        assert "Saved watchdog failure" in overview_dict["system_dict"]["detail_str"]
    assert all(header_dict == header_list[0] for header_dict in header_list)
    assert header_list[0]["label_str"] == "System needs action"


@pytest.mark.parametrize("path_str", PATH_LIST)
@pytest.mark.parametrize("failure_str", ["slow", "changed_release", "rejected"])
def test_shared_service_reads_cannot_keep_current_claims_after_expiry_or_scope_change(shared_tuple, path_str, failure_str):
    client_obj, context_list, control_dict = shared_tuple
    control_dict.update(delay_int=121 if failure_str == "slow" else 0,
        scope_case_str="valid" if failure_str == "slow" else failure_str)
    response_obj = client_obj.get(path_str)
    assert response_obj.status_code == 200
    context_dict = context_list[-1]
    overview_dict = context_dict["overview_dict"]
    assert overview_dict["source_fresh_bool"] is False
    assert overview_dict["source_valid_ms_int"] == 0
    assert overview_dict["system_dict"]["state_str"] == "unk"
    assert overview_dict["live_state_str"] == "unk"
    assert all(pod_dict["state_str"] == "unk" for pod_dict in overview_dict["pod_list"])
    assert all(step_dict["state_str"] == "unk" for pod_dict in overview_dict["pod_list"] for step_dict in pod_dict["step_list"])
    if "positions_page_dict" in context_dict:
        assert context_dict["positions_page_dict"]["source_fresh_bool"] is False
        assert all(row_dict["today_detail_str"] == "Unknown" for row_dict in context_dict["positions_page_dict"]["row_list"]
                   if row_dict["today_pending_bool"])
    if failure_str == "slow":
        assert overview_dict["clock_str"] == (DEMO_NOW_TS + timedelta(seconds=121)).astimezone(
            ZoneInfo("America/New_York")).strftime("%H:%M:%S")


def test_unsupported_checks_have_no_browser_expiry_markers(shared_tuple):
    import re

    client_obj, _, _ = shared_tuple
    html_str = client_obj.get("/system").get_data(as_text=True)
    for key_str in ("gateway", "fred"):
        row_str = re.search(r'<tr data-system-check="' + key_str + r'".*?</tr>', html_str, re.S).group()
        assert "Not checked here" in row_str
        assert "data-evidence-status" not in row_str
        assert "data-observed-state" not in row_str
        assert "system-problem" not in row_str
