"""Catalog identity and CLI contracts; no live state or broker access."""

from copy import deepcopy
import argparse
import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

from alpha.live.dashboard_v4.tools import (
    EXECUTABLE_ACTION_SET, build_tools_page_dict, powershell_command_str,
    resolve_tools_target_obj,
)


@pytest.fixture
def context_tuple():
    row_dict = {"pod_id_str": "pod-one", "account_route_str": "U123456",
        "user_id_str": "owner-one", "release_id_str": "release-one", "mode_str": "live",
        "db_path_str": "state/pod-one.sqlite3"}
    release_obj = SimpleNamespace(**{key_str: value_obj for key_str, value_obj in row_dict.items()
        if key_str != "db_path_str"}, enabled_bool=True)
    target_obj = SimpleNamespace(release_obj=release_obj, db_path_str=row_dict["db_path_str"])
    requested_list = []

    def get_target_fn(pod_id_str):
        requested_list.append(pod_id_str)
        return target_obj

    provider_obj = SimpleNamespace(get_target_for_pod=get_target_fn,
        releases_root_path_str="local/releases", config_path_str="local/dashboard.json",
        event_log_path_str="local/events.jsonl", results_root_path_str="local/results")
    workspace_dict = {"operations_error_str": "", "operations_account_list": [{
        "pod_id": "pod-one", "account_route": "U123456", "display_name": "Long Pod name"}],
        "summary_dict": {"pod_row_dict_list": [row_dict]}}
    return workspace_dict, provider_obj, target_obj, requested_list


def _row_map_dict(page_dict):
    return {row_dict["key_str"]: row_dict for block_dict in page_dict["block_list"]
        for group_dict in block_dict["group_list"] for row_dict in group_dict["row_list"]}


def test_catalog_copies_in_read_only_and_only_existing_allowlist_runs(context_tuple):
    workspace_dict, provider_obj, _, _ = context_tuple
    page_dict = build_tools_page_dict(workspace_dict, provider_obj, selected_pod_str="pod-one")
    row_map_dict = _row_map_dict(page_dict)
    assert len(row_map_dict) == 20
    assert all(not row_dict["runnable_bool"] for row_dict in row_map_dict.values())
    assert sum(row_dict["copy_available_bool"] for row_dict in row_map_dict.values()) == 18
    assert page_dict["selected_pod_dict"]["account_str"] == "···456"
    enabled_dict = _row_map_dict(build_tools_page_dict(workspace_dict, provider_obj,
        selected_pod_str="pod-one", actions_enabled_bool=True))
    assert {key_str for key_str, row_dict in enabled_dict.items() if row_dict["runnable_bool"]} == EXECUTABLE_ACTION_SET
    assert not enabled_dict["manual_order"]["argument_list"]
    assert enabled_dict["saved_watchdog_report"]["href_str"] == "/system"


def test_unknown_selection_never_looks_up_foreign_target(context_tuple):
    workspace_dict, provider_obj, _, requested_list = context_tuple
    with pytest.raises(ValueError, match="Choose an enabled"):
        resolve_tools_target_obj(workspace_dict, provider_obj, "foreign-pod")
    assert requested_list == []
    page_dict = build_tools_page_dict(workspace_dict, provider_obj, selected_pod_str="foreign-pod")
    assert page_dict["error_str"]
    assert not _row_map_dict(page_dict)["tick"]["copy_available_bool"]
    assert requested_list == ["pod-one"]


@pytest.mark.parametrize("field_str,value_obj", [
    ("mode_str", "paper"), ("mode_str", "incubation"), ("enabled_bool", False),
    ("pod_id_str", "other"), ("account_route_str", "U999999"),
    ("user_id_str", "other-owner"), ("release_id_str", "new-release"),
])
def test_resolver_rejects_current_target_drift(context_tuple, field_str, value_obj):
    workspace_dict, provider_obj, target_obj, _ = context_tuple
    setattr(target_obj.release_obj, field_str, value_obj)
    with pytest.raises(ValueError):
        resolve_tools_target_obj(workspace_dict, provider_obj, "pod-one")
    page_dict = build_tools_page_dict(workspace_dict, provider_obj,
        selected_pod_str="pod-one", actions_enabled_bool=True)
    assert not _row_map_dict(page_dict)["tick"]["runnable_bool"]


def test_resolver_rejects_database_path_drift(context_tuple):
    workspace_dict, provider_obj, target_obj, _ = context_tuple
    target_obj.db_path_str = "state/other.sqlite3"
    with pytest.raises(ValueError, match="state path changed"):
        resolve_tools_target_obj(workspace_dict, provider_obj, "pod-one")


@pytest.mark.parametrize("failure_str", ["duplicate_pod", "duplicate_account",
    "duplicate_row", "mixed_owners", "paper_row"])
def test_bad_workspace_fails_closed_before_provider_lookup(context_tuple, failure_str):
    workspace_dict, provider_obj, _, requested_list = context_tuple
    if failure_str == "operations_error":
        workspace_dict["operations_error_str"] = "Configuration unavailable"
    elif failure_str == "duplicate_pod":
        workspace_dict["operations_account_list"] *= 2
    elif failure_str == "duplicate_account":
        workspace_dict["operations_account_list"].append({"pod_id": "other", "account_route": "U123456"})
    elif failure_str == "duplicate_row":
        workspace_dict["summary_dict"]["pod_row_dict_list"] *= 2
    elif failure_str == "missing_row":
        workspace_dict["summary_dict"]["pod_row_dict_list"] = []
    elif failure_str == "missing_owner":
        del workspace_dict["summary_dict"]["pod_row_dict_list"][0]["user_id_str"]
    elif failure_str == "paper_row":
        workspace_dict["summary_dict"]["pod_row_dict_list"][0]["mode_str"] = "paper"
    else:
        workspace_dict["operations_account_list"].append({"pod_id": "other", "account_route": "U999999"})
        workspace_dict["summary_dict"]["pod_row_dict_list"].append({"pod_id_str": "other",
            "account_route_str": "U999999", "mode_str": "live", "release_id_str": "other",
            "user_id_str": "foreign-owner"})
    with pytest.raises(ValueError):
        resolve_tools_target_obj(workspace_dict, provider_obj, "pod-one")
    page_dict = build_tools_page_dict(workspace_dict, provider_obj,
        selected_pod_str="pod-one", actions_enabled_bool=True)
    assert page_dict["error_str"]
    assert not any(row_dict["copy_available_bool"] or row_dict["runnable_bool"]
        for row_dict in _row_map_dict(page_dict).values())
    assert all(pod_id_str in {"pod-one", "other"} for pod_id_str in requested_list)


def test_no_selection_system_scope_is_explicit_and_no_pod_is_picked(context_tuple):
    workspace_dict, provider_obj, _, requested_list = context_tuple
    row_map_dict = _row_map_dict(build_tools_page_dict(workspace_dict, provider_obj))
    assert row_map_dict["ops_report"]["copy_available_bool"]
    assert "all LIVE Pods" in row_map_dict["ops_report"]["scope_str"]
    assert "all configured releases" in row_map_dict["doctor_norgate_client"]["scope_str"]
    assert not row_map_dict["tick"]["copy_available_bool"]
    assert "--pod-id" not in row_map_dict["ops_report"]["argument_list"]
    assert "--db-path" not in row_map_dict["live_ops_watchdog"]["argument_list"]
    assert requested_list == ["pod-one"]


@pytest.mark.parametrize("failure_str", ["operations_error", "missing_row", "missing_owner"])
def test_current_config_allows_copy_diagnostics_when_saved_state_is_missing(context_tuple, failure_str):
    workspace_dict, provider_obj, _, _ = context_tuple
    if failure_str == "operations_error":
        workspace_dict["operations_error_str"] = "Saved operations could not be read."
    elif failure_str == "missing_row":
        workspace_dict["summary_dict"]["pod_row_dict_list"] = []
    else:
        del workspace_dict["summary_dict"]["pod_row_dict_list"][0]["user_id_str"]
    page_dict = build_tools_page_dict(workspace_dict, provider_obj,
        selected_pod_str="pod-one", actions_enabled_bool=True)
    row_map_dict = _row_map_dict(page_dict)
    assert row_map_dict["status"]["copy_available_bool"]
    assert row_map_dict["doctor"]["copy_available_bool"]
    assert not any(row_dict["runnable_bool"] for row_dict in row_map_dict.values())
    with pytest.raises(ValueError):
        resolve_tools_target_obj(workspace_dict, provider_obj, "pod-one")


def test_missing_other_pod_state_keeps_both_known_config_diagnostics(context_tuple):
    workspace_dict, provider_obj, first_target_obj, requested_list = context_tuple
    second_target_obj = deepcopy(first_target_obj)
    second_target_obj.release_obj.pod_id_str = "pod-two"
    second_target_obj.release_obj.account_route_str = "U777777"
    second_target_obj.db_path_str = "state/pod-two.sqlite3"
    target_map_dict = {"pod-one": first_target_obj, "pod-two": second_target_obj}
    provider_obj.get_target_for_pod = target_map_dict.get
    workspace_dict["operations_account_list"].append({"pod_id": "pod-two", "account_route": "U777777"})
    page_dict = build_tools_page_dict(workspace_dict, provider_obj, selected_pod_str="pod-two")
    assert _row_map_dict(page_dict)["status"]["copy_available_bool"]
    assert not page_dict["error_str"]
    second_target_obj.release_obj.user_id_str = "foreign-owner"
    failed_dict = build_tools_page_dict(workspace_dict, provider_obj, selected_pod_str="pod-two")
    assert failed_dict["error_str"]
    assert not any(row_dict["copy_available_bool"] for row_dict in _row_map_dict(failed_dict).values())


def test_generated_commands_parse_against_actual_cli_declarations_without_running_them(context_tuple):
    workspace_dict, provider_obj, _, _ = context_tuple
    row_map_dict = _row_map_dict(build_tools_page_dict(workspace_dict, provider_obj, selected_pod_str="pod-one"))
    parser_map_dict = {}
    for row_dict in row_map_dict.values():
        argument_list = row_dict["argument_list"]
        if not argument_list:
            continue
        module_str = argument_list[4]
        if module_str not in parser_map_dict:
            source_path_obj = Path(__file__).resolve().parents[1] / (module_str.replace(".", "/") + ".py")
            module_obj = ast.parse(source_path_obj.read_text(encoding="utf-8"))
            main_obj = next(node_obj for node_obj in module_obj.body
                if isinstance(node_obj, ast.FunctionDef) and node_obj.name == "main")
            parser_obj = argparse.ArgumentParser()
            for node_obj in ast.walk(main_obj):
                if not (isinstance(node_obj, ast.Call) and isinstance(node_obj.func, ast.Attribute)
                        and isinstance(node_obj.func.value, ast.Name)
                        and node_obj.func.value.id == "parser_obj" and node_obj.func.attr == "add_argument"):
                    continue
                names_list = [ast.literal_eval(name_obj) for name_obj in node_obj.args]
                keyword_dict = {}
                for keyword_obj in node_obj.keywords:
                    if keyword_obj.arg == "type" and isinstance(keyword_obj.value, ast.Name):
                        keyword_dict["type"] = {"str": str, "int": int, "float": float}[keyword_obj.value.id]
                    elif keyword_obj.arg in {"action", "choices", "nargs", "required", "dest"}:
                        keyword_dict[keyword_obj.arg] = ast.literal_eval(keyword_obj.value)
                parser_obj.add_argument(*names_list, **keyword_dict)
            parser_map_dict[module_str] = parser_obj
        parser_map_dict[module_str].parse_args(argument_list[5:])


def test_commands_use_the_real_cli_modules_and_supported_flags(context_tuple):
    workspace_dict, provider_obj, _, _ = context_tuple
    row_map_dict = _row_map_dict(build_tools_page_dict(workspace_dict, provider_obj, selected_pod_str="pod-one"))
    for key_str in ("status", "show_decision_plan", "show_vplan", "execution_report", "export_trade_sheet", "doctor", "tick"):
        argument_list = row_map_dict[key_str]["argument_list"]
        assert argument_list[:6] == ["uv", "run", "python", "-m", "alpha.live.runner", key_str]
        assert argument_list[argument_list.index("--db-path") + 1] == "state/pod-one.sqlite3"
        assert "--dashboard-config" not in argument_list
    assert row_map_dict["next_due"]["argument_list"][4] == "alpha.live.scheduler_service"
    assert row_map_dict["collect_vps_debug_bundle"]["argument_list"][4] == "scripts.live_debug.collect_vps_debug_bundle"
    assert "--include-doctor" not in row_map_dict["collect_vps_debug_bundle"]["argument_list"]
    assert "--dashboard-config" in row_map_dict["ops_report"]["argument_list"]
    norgate_list = row_map_dict["doctor_norgate_client"]["argument_list"]
    assert not {"--json", "--mode", "--pod-id", "--db-path"}.intersection(norgate_list)
    assert "--reference-strategy-pickle" not in row_map_dict["compare_reference"]["argument_list"]


def test_required_client_id_is_declared_never_invented(context_tuple):
    workspace_dict, provider_obj, _, _ = context_tuple
    row_map_dict = _row_map_dict(build_tools_page_dict(workspace_dict, provider_obj, selected_pod_str="pod-one"))
    doctor_dict = row_map_dict["doctor"]
    assert doctor_dict["parameter_list"][0]["required_bool"]
    assert doctor_dict["parameter_list"][0]["flag_str"] == "--broker-client-id"
    assert "--broker-client-id" not in doctor_dict["argument_list"]
    assert row_map_dict["show_vplan"]["parameter_list"][0]["required_bool"] is False
    assert row_map_dict["submit_vplan"]["parameter_list"] == []
    assert row_map_dict["compare_reference"]["parameter_list"] == []


def test_powershell_quoting_preserves_literal_shell_metacharacters(context_tuple):
    workspace_dict, provider_obj, _, _ = context_tuple
    provider_obj.releases_root_path_str = "C:/Owner's files/$(do-not-run);`stuff"
    page_dict = build_tools_page_dict(workspace_dict, provider_obj, selected_pod_str="pod-one")
    command_str = _row_map_dict(page_dict)["tick"]["command_str"]
    assert "'C:/Owner''s files/$(do-not-run);`stuff'" in command_str
    assert powershell_command_str(["uv", "a'b", "$HOME"]) == "& 'uv' 'a''b' '$HOME'"


def test_demo_commands_are_explicitly_synthetic_and_do_not_leak_host_paths(context_tuple):
    workspace_dict, provider_obj, _, _ = context_tuple
    unchanged_dict = deepcopy(workspace_dict)
    page_dict = build_tools_page_dict(workspace_dict, provider_obj,
        selected_pod_str="pod-one", demo_bool=True)
    assert page_dict["demo_bool"]
    for row_dict in _row_map_dict(page_dict).values():
        assert "local/" not in row_dict["command_str"]
        assert "state/pod-one.sqlite3'" not in row_dict["command_str"].replace("DEMO/state/pod-one.sqlite3'", "")
    assert "DEMO/releases" in _row_map_dict(page_dict)["tick"]["command_str"]
    assert workspace_dict == unchanged_dict
