"""Do-next suggestions: fitting, copy-only, safest first, never a guess."""

import re

import pytest

from alpha.live.dashboard_v4 import next_commands as next_commands_module
from alpha.live.dashboard_v4.demo import DEMO_NOW_TS, build_demo_workspace_tuple, create_demo_app
from alpha.live.dashboard_v4.next_commands import (
    attach_next_command_list, next_command_key_list, serve_running_command_str)


@pytest.mark.parametrize("attention_dict,expected_list", [
    ({"kind_str": "scheduler"}, ["serve_running", "next_due"]),
    ({"kind_str": "hold", "reason_code_str": "execution_exception_parked"}, ["show_vplan", "execution_report"]),
    ({"kind_str": "hold", "reason_code_str": "manual_review_required"}, ["show_vplan", "submit_vplan"]),
    ({"kind_str": "hold", "reason_code_str": "unknown_new_reason"}, ["status", "show_vplan"]),
    ({"kind_str": "database", "db_status_str": "error"}, ["status"]),
    ({"kind_str": "database", "db_status_str": "missing"}, ["db_path_check"]),
    ({"kind_str": "cycle", "step_str": "Submit", "scheduler_bool": True},
        ["serve_running", "next_due", "show_vplan", "execution_report"]),
    ({"kind_str": "action", "inspect_str": "show_vplan", "next_action_str": "review_vplan", "step_str": ""},
        ["show_vplan", "submit_vplan"]),
    ({"kind_str": "cycle", "step_str": "Submit"}, ["show_vplan", "execution_report"]),
    ({"kind_str": "cycle", "step_str": "Reconcile"}, ["execution_report", "post_execution_reconcile"]),
    ({"kind_str": "cycle", "step_str": "Data"}, ["status", "doctor_norgate_client"]),
    ({"kind_str": "cycle", "step_str": "Unknown step"}, []),
    ({"title_str": "No kind"}, []),
])
def test_key_list_fits_the_problem(attention_dict, expected_list):
    assert next_command_key_list(attention_dict) == expected_list


def test_serve_running_check_is_exact_and_rejects_unsafe_pod_ids():
    command_str = serve_running_command_str("pod_ndx.v2-live_01")
    # Private bytes are never trimmed like the working set, so an idle scheduler still matches.
    assert "Get-CimInstance Win32_Process" in command_str and "PrivatePageCount -gt 20MB" in command_str
    assert "WorkingSetSize" not in command_str
    assert re.escape("pod_ndx.v2-live_01") in command_str  # pod_ndx.v2 must not match pod_ndxXv2
    assert "(?:^|\\s)--pod-id\\s+" in command_str and "(?:\\s|$)" in command_str
    for unsafe_str in ("", "pod a", "pod';Remove-Item", "x" * 201, None):
        assert serve_running_command_str(unsafe_str) == ""


@pytest.fixture()
def workspace_tuple():
    workspace_dict, snapshot_obj, provider_obj = build_demo_workspace_tuple()
    try:
        yield workspace_dict, provider_obj
    finally:
        provider_obj.close()


def test_commands_come_from_the_tools_catalog_safest_first_and_capped(workspace_tuple, monkeypatch):
    workspace_dict, provider_obj = workspace_tuple
    monkeypatch.setitem(next_commands_module.STEP_KEY_DICT, "Submit",
        ("post_execution_reconcile", "show_vplan", "status", "execution_report"))
    attention_list = [{"pod_id_str": "demo_1_1", "kind_str": "cycle", "step_str": "Submit"}]
    attach_next_command_list(attention_list, workspace_dict, provider_obj, demo_bool=True, as_of_ts=DEMO_NOW_TS)
    command_list = attention_list[0]["next_command_list"]
    assert [command_dict["key_str"] for command_dict in command_list] == ["show_vplan", "status", "execution_report"]
    assert all(command_dict["class_str"] == "INSPECT" for command_dict in command_list)
    assert all(command_dict["command_str"].startswith("& 'uv' 'run' 'python' '-m' 'alpha.live.runner'")
        and "'--pod-id' 'demo_1_1'" in command_dict["command_str"] for command_dict in command_list)


def test_submit_is_offered_only_with_a_verified_ready_plan(workspace_tuple):
    workspace_dict, provider_obj = workspace_tuple
    attention_list = [{"pod_id_str": "demo_1_0", "kind_str": "action", "inspect_str": "show_vplan",
        "next_action_str": "review_vplan", "step_str": ""}]
    attach_next_command_list(attention_list, workspace_dict, provider_obj, demo_bool=True, as_of_ts=DEMO_NOW_TS)
    # The demo plan is completed, not ready: submit_vplan has no verified ID, so it is left out.
    assert [command_dict["key_str"] for command_dict in attention_list[0]["next_command_list"]] == ["show_vplan"]


def test_unverified_scope_suggests_nothing(workspace_tuple):
    workspace_dict, provider_obj = workspace_tuple
    broken_dict = {**workspace_dict, "operations_account_list": []}
    attention_list = [{"pod_id_str": "demo_1_1", "kind_str": "scheduler"}]
    attach_next_command_list(attention_list, broken_dict, provider_obj, demo_bool=True, as_of_ts=DEMO_NOW_TS)
    assert attention_list[0]["next_command_list"] == []


def test_scheduler_item_suggests_the_process_check_first(workspace_tuple):
    workspace_dict, provider_obj = workspace_tuple
    attention_list = [{"pod_id_str": "demo_1_0", "kind_str": "scheduler"}]
    attach_next_command_list(attention_list, workspace_dict, provider_obj, demo_bool=True, as_of_ts=DEMO_NOW_TS)
    command_list = attention_list[0]["next_command_list"]
    assert [command_dict["key_str"] for command_dict in command_list] == ["serve_running", "next_due"]
    assert command_list[0]["class_str"] == "READ" and command_list[0]["orders_bool"] is False


def test_overview_and_pod_page_render_copy_chips():
    client_obj = create_demo_app().test_client()
    overview_str = client_obj.get("/").get_data(as_text=True)
    assert overview_str.count("data-copy-command=") == 2 and "Do next" in overview_str
    pod_str = client_obj.get("/pods/demo_1_1").get_data(as_text=True)
    assert re.findall(r'class="cmd-chip[^"]*"[^>]*>([a-z_]+)', pod_str) == ["show_vplan", "execution_report"]
    assert "Click to copy, then run it in the VPS terminal." in pod_str
    # A historical cycle's issue never suggests commands for the latest cycle.
    history_str = client_obj.get("/pods/demo_1_1?cycle=vplan:1").get_data(as_text=True)
    assert "data-copy-command=" not in history_str


def test_missing_state_db_gets_a_pure_read_path_check(workspace_tuple):
    workspace_dict, provider_obj = workspace_tuple
    attention_list = [{"pod_id_str": "demo_1_0", "kind_str": "database", "db_status_str": "missing"}]
    attach_next_command_list(attention_list, workspace_dict, provider_obj, demo_bool=True, as_of_ts=DEMO_NOW_TS)
    command_dict, = attention_list[0]["next_command_list"]
    # runner commands open (and so create) the DB, which would hide the problem.
    assert command_dict["key_str"] == "db_path_check" and command_dict["class_str"] == "READ"
    assert command_dict["command_str"] == "Test-Path -LiteralPath 'DEMO/state/demo_1_0.sqlite3'"


def test_powershell_quoting_doubles_typographic_quotes():
    from alpha.live.dashboard_v4.tools import powershell_command_str, powershell_literal_str
    assert powershell_literal_str("it's") == "'it''s'"
    assert powershell_literal_str("O\u2019Brien; Write-Output X") == "'O\u2019\u2019Brien; Write-Output X'"
    assert powershell_literal_str("\u2018a\u201ab\u201b") == "'\u2018\u2018a\u201a\u201ab\u201b\u201b'"
    assert powershell_command_str(["uv", "$env:x", "a`b"]) == "& 'uv' '$env:x' 'a`b'"
