"""Tools integration: copy in production, isolated simulation only in demo."""

from copy import deepcopy
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from alpha.live.dashboard_v4.app import create_app
from alpha.live.dashboard_v4.demo import DEMO_NOW_TS, build_demo_workspace_tuple


def _forbidden(*args_tuple, **kwargs_dict):
    pytest.fail("Tools reached an unapproved source")


@pytest.fixture
def workspace_tuple():
    workspace_dict, _, provider_obj = build_demo_workspace_tuple()
    provider_obj.releases_root_path_str = "TEST-ONLY/releases"
    yield workspace_dict, provider_obj
    provider_obj.close()


def _app_obj(workspace_tuple, *, demo_bool=False, demo_tools_bool=False):
    workspace_dict, provider_obj = workspace_tuple
    return create_app(provider_obj, demo_bool=demo_bool, demo_tools_bool=demo_tools_bool,
        operations_workspace_fn=lambda: deepcopy(workspace_dict),
        workspace_snapshot_fn=_forbidden, now_fn=lambda: DEMO_NOW_TS)


@pytest.mark.parametrize("path_str", ["/tools", "/tools/status", "/api/tools/demo_1_0/tick/confirm",
    "/api/demo-tools/demo_1_0/tick/preview", "/api/demo-tools/demo_1_0/tick/confirm",
    "/api/demo-tools/demo_1_0/tick/cancel"])
@pytest.mark.parametrize("method_str", ["POST", "PUT", "PATCH", "DELETE"])
def test_production_cannot_mutate_or_acquire_sources(path_str, method_str):
    app_obj = create_app(SimpleNamespace(), operations_workspace_fn=_forbidden)
    response_obj = app_obj.test_client().open(path_str, method=method_str)
    assert response_obj.status_code == 403
    assert response_obj.json["error"] == "read_only"
    assert app_obj.config["tools_action_service_obj"].action_token_str == ""


@pytest.mark.parametrize("query_str", ["x=1", "mode=paper", "pod=", "pod=a&pod=b", "tool=a&tool=b"])
def test_invalid_query_rejected_before_sources(query_str):
    app_obj = create_app(SimpleNamespace(), operations_workspace_fn=_forbidden)
    assert app_obj.test_client().get("/tools?" + query_str).status_code == 400


def test_production_catalog_and_status_need_no_financial_snapshot(workspace_tuple):
    app_obj = _app_obj(workspace_tuple)
    client_obj = app_obj.test_client()
    response_obj = client_obj.get("/tools?pod=demo_1_0&tool=status")
    assert response_obj.status_code == 200
    html_str = response_obj.get_data(as_text=True)
    assert html_str.count('data-tool="') == 20
    assert 'data-actions-enabled="false"' in html_str
    assert 'data-action-token=""' in html_str
    assert "Execution is not connected" in html_str
    assert 'data-tool-preview>' not in html_str
    assert 'data-tool-copy-button' in html_str
    assert 'hx-get="/tools/status"' in html_str and 'hx-swap="none"' in html_str
    assert 'data-selection-scope="tools:demo_1_0:status"' in html_str
    assert 'href="/tools" aria-current="page"' in html_str
    assert "/static/tools.css" in html_str and "/static/tools.js" in html_str
    assert response_obj.headers["Cache-Control"] == "no-store"
    status_str = client_obj.get("/tools/status").get_data(as_text=True)
    assert 'hx-swap-oob="outerHTML"' in status_str
    assert 'data-tools-page' not in status_str and 'data-tool-form' not in status_str
    assert 'href="/tools" aria-current="page"' in status_str
    assert client_obj.get("/tools/status?pod=demo_1_0").status_code == 400
    assert client_obj.get("/tools?pod=foreign-pod").status_code == 404
    assert client_obj.get("/tools?tool=arbitrary_shell").status_code == 400


def test_simulation_cannot_be_enabled_without_demo():
    with pytest.raises(ValueError, match="isolated demo"):
        create_app(SimpleNamespace(), demo_tools_bool=True)


def test_default_demo_remains_copy_only(workspace_tuple):
    app_obj = _app_obj(workspace_tuple, demo_bool=True)
    assert app_obj.config["tools_action_service_obj"].action_token_str == ""
    assert app_obj.test_client().post("/api/demo-tools/demo_1_0/tick/preview").status_code == 403


@pytest.mark.parametrize("action_str", ["tick", "submit_vplan", "post_execution_reconcile", "eod_snapshot", "compare_reference", "manual_order"])
def test_demo_flow_only_changes_memory_and_activity(workspace_tuple, monkeypatch, action_str):
    app_obj = _app_obj(workspace_tuple, demo_bool=True, demo_tools_bool=True)
    client_obj = app_obj.test_client()
    service_obj = app_obj.config["tools_action_service_obj"]
    provider_obj = workspace_tuple[1]
    path_list = [Path(target_obj.db_path_str) for target_obj in provider_obj.get_target_list()]
    snapshot_dict = {path_obj: path_obj.read_bytes() for path_obj in path_list}
    monkeypatch.setattr("alpha.live.dashboard.DashboardApp.__post_init__", _forbidden)
    html_str = client_obj.get("/tools?pod=demo_1_0&tool=tick").get_data(as_text=True)
    assert 'data-actions-enabled="true"' in html_str
    assert "synthetic results" in html_str
    headers_dict = {"Origin": "http://localhost", "X-Alpha-Action-Token": service_obj.action_token_str}
    preview_body_dict = {"confirmed_bool": True}
    if action_str == "manual_order":
        preview_body_dict["manual_order_dict"] = {"asset_str": "AAPL", "side_str": "BUY",
            "broker_order_type_str": "LMT", "quantity_int": 1, "limit_price_float": 100.0,
            "time_in_force_str": "DAY", "operator_id_str": "demo-reviewer", "reason_str": "Synthetic test",
            "confirmation_text_str": "SUBMIT MANUAL ORDER"}
    action_path_str = "/api/demo-tools/demo_1_0/" + action_str
    preview_obj = client_obj.post(action_path_str + "/preview",
        json=preview_body_dict, headers=headers_dict)
    assert preview_obj.status_code == 200, preview_obj.json
    nonce_str = preview_obj.json["confirmation_nonce_str"]
    confirm_dict = {"confirmed_bool": True, "browser_confirmed_bool": True, "confirmation_nonce_str": nonce_str}
    result_obj = client_obj.post(action_path_str + "/confirm", json=confirm_dict, headers=headers_dict)
    assert result_obj.status_code == 202, result_obj.json
    assert result_obj.json["demo_bool"] is True
    assert "No real command" in result_obj.json["message_str"]
    assert client_obj.get(result_obj.json["poll_url_str"]).json == result_obj.json
    assert client_obj.post(action_path_str + "/confirm", json=confirm_dict, headers=headers_dict).status_code == 409
    assert client_obj.post("/api/tools/demo_1_0/tick/confirm", json=confirm_dict, headers=headers_dict).status_code == 403
    activity_str = client_obj.get("/activity").get_data(as_text=True)
    assert "Simulated tool action. No real command was run." in activity_str
    assert "simulated_succeeded" in activity_str
    assert {path_obj: path_obj.read_bytes() for path_obj in path_list} == snapshot_dict


def test_missing_saved_state_keeps_diagnostic_copy_available(workspace_tuple):
    workspace_tuple[0]["summary_dict"]["pod_row_dict_list"] = []
    html_str = _app_obj(workspace_tuple).test_client().get("/tools?pod=demo_1_0").get_data(as_text=True)
    assert 'data-tool-copy-button' in html_str
    assert 'data-tool-preview>' not in html_str


def test_stale_operations_block_demo_preview(workspace_tuple):
    workspace_tuple[0]["summary_dict"]["as_of_timestamp_str"] = (DEMO_NOW_TS - timedelta(seconds=121)).isoformat()
    app_obj = _app_obj(workspace_tuple, demo_bool=True, demo_tools_bool=True)
    service_obj = app_obj.config["tools_action_service_obj"]
    response_obj = app_obj.test_client().post("/api/demo-tools/demo_1_0/tick/preview",
        json={"confirmed_bool": True}, headers={"Origin": "http://localhost", "X-Alpha-Action-Token": service_obj.action_token_str})
    assert response_obj.status_code == 409
    assert not service_obj.provider_obj.job_dict


def test_system_links_to_selected_pod_tools(workspace_tuple):
    html_str = _app_obj(workspace_tuple).test_client().get("/system").get_data(as_text=True)
    assert 'href="/tools?pod=demo_1_0"' in html_str
    assert 'aria-label="Tools for ' in html_str


def test_cli_rejects_demo_tools_before_any_config_or_provider(monkeypatch):
    from alpha.live.dashboard_v4 import __main__ as entry_obj
    monkeypatch.setattr("sys.argv", ["dashboard_v4", "--demo-tools"])
    monkeypatch.setattr(entry_obj, "LiveDataProvider", _forbidden)
    with pytest.raises(SystemExit) as error_obj:
        entry_obj.main()
    assert error_obj.value.code == 2
