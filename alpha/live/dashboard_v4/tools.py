"""Fixed, display-only operator command catalog for the local LIVE workspace.

Building the page resolves current release identities but never invokes a tool,
opens state for writing, reads environment secrets, or accepts shell text.
"""

from pathlib import Path
from urllib.parse import quote


EXECUTABLE_ACTION_SET = frozenset({"tick", "submit_vplan", "post_execution_reconcile",
    "eod_snapshot", "compare_reference", "manual_order"})

# key, block, group, class, effect, command family, scope
TOOL_CATALOG_TUPLE = (
    ("ops_report", "read", "Daily checks", "READ", "Summarize saved LIVE health across this installation.", "runner", "system"),
    ("status", "read", "Daily checks", "INSPECT", "Show saved Pod state; may record diagnostic metadata.", "runner", "pod"),
    ("next_due", "read", "Daily checks", "INSPECT", "Explain the next scheduled step; may record diagnostic metadata.", "scheduler", "pod"),
    ("show_decision_plan", "read", "Diagnose and review plans", "INSPECT", "Show the saved decision and its target holdings.", "runner", "pod"),
    ("show_vplan", "read", "Diagnose and review plans", "INSPECT", "Show the saved order plan and execution evidence.", "runner", "pod"),
    ("execution_report", "read", "Diagnose and review plans", "INSPECT", "Review saved orders, fills and reconciliation.", "runner", "pod"),
    ("export_trade_sheet", "read", "Reports and support", "INSPECT", "Write a trade sheet from the saved plan; may record metadata.", "runner", "pod"),
    ("compare_reference", "read", "Reports and support", "INSPECT", "Build a deployment comparison; writes reports and may update data caches.", "runner", "pod"),
    ("collect_vps_debug_bundle", "read", "Reports and support", "INSPECT", "Collect saved diagnostics into a local support bundle.", "bundle", "pod"),
    ("saved_watchdog_report", "read", "Reports and support", "READ", "Open the saved watchdog assessment and task evidence.", "saved", "system"),
    ("doctor", "act", "Diagnose", "ACTIVE", "Query the broker and check data; may sync snapshots. Run once with an unused client ID.", "runner", "pod"),
    ("tick", "act", "Run the cycle", "ACTIVE", "Run one lifecycle pass; may sync data, build plans and send orders.", "runner", "pod"),
    ("run_once", "act", "Run the cycle", "ACTIVE", "Run one scheduler pass; may send orders when due.", "scheduler", "pod"),
    ("serve", "act", "Run the cycle", "ACTIVE", "Start the Pod scheduler. Never start a second scheduler for this Pod.", "scheduler", "pod"),
    ("submit_vplan", "act", "Submit and confirm execution", "ACTIVE", "Submit the ready plan using the existing execution guards.", "runner", "pod"),
    ("post_execution_reconcile", "act", "Submit and confirm execution", "ACTIVE", "Query the broker and save the position check.", "runner", "pod"),
    ("eod_snapshot", "act", "Submit and confirm execution", "ACTIVE", "Query the broker and save the end-of-day account observation.", "runner", "pod"),
    ("live_ops_watchdog", "act", "Data and watchdog", "ACTIVE", "Refresh LIVE health across this installation; may send alerts.", "watchdog", "system"),
    ("doctor_norgate_client", "act", "Data and watchdog", "ACTIVE", "Check the configured Norgate client; may download snapshot files.", "norgate", "system"),
    ("manual_order", "act", "Break glass", "ACTIVE", "Create and confirm a manual broker order for this Pod.", "manual", "pod"),
)


def powershell_command_str(argument_list):
    """Quote each literal argument: apostrophes, dollars and backticks stay data."""
    return "& " + " ".join("'" + str(argument_str).replace("'", "''") + "'" for argument_str in argument_list)


def _scope_rows_tuple(workspace_dict):
    if workspace_dict.get("operations_error_str"):
        raise ValueError("LIVE workspace identity could not be verified.")
    account_list = workspace_dict.get("operations_account_list") or []
    source_row_list = (workspace_dict.get("summary_dict") or {}).get("pod_row_dict_list") or []
    if not account_list:
        raise ValueError("No enabled LIVE Pods are available.")
    pod_id_list = [account_dict.get("pod_id") for account_dict in account_list]
    account_route_list = [account_dict.get("account_route") for account_dict in account_list]
    if (not all(pod_id_list) or not all(account_route_list)
            or len(set(pod_id_list)) != len(pod_id_list)
            or len(set(account_route_list)) != len(account_route_list)):
        raise ValueError("LIVE Pod ownership is ambiguous.")
    scoped_row_list = []
    for account_dict in account_list:
        match_list = [row_dict for row_dict in source_row_list if isinstance(row_dict, dict)
            and row_dict.get("pod_id_str") == account_dict["pod_id"]]
        if len(match_list) != 1:
            raise ValueError("LIVE Pod ownership is ambiguous.")
        row_dict = match_list[0]
        if (row_dict.get("mode_str") != "live"
                or row_dict.get("account_route_str") != account_dict["account_route"]
                or not row_dict.get("user_id_str") or not row_dict.get("release_id_str")):
            raise ValueError("Saved LIVE identity does not match this workspace.")
        scoped_row_list.append(row_dict)
    if len({row_dict["user_id_str"] for row_dict in scoped_row_list}) != 1:
        raise ValueError("The LIVE workspace must have one owner.")
    return account_list, scoped_row_list


def _copy_scope_tuple(workspace_dict, provider_obj):
    """Diagnostics remain copyable when saved state is absent, using live config.

    Only target metadata for already-approved local account bindings is read.
    Conflicting saved identities still block; missing state is not a conflict.
    The action resolver above remains stricter and requires saved evidence.
    """
    account_list = workspace_dict.get("operations_account_list") or []
    pod_id_list = [account_dict.get("pod_id") for account_dict in account_list]
    account_route_list = [account_dict.get("account_route") for account_dict in account_list]
    if (not account_list or not all(pod_id_list) or not all(account_route_list)
            or len(set(pod_id_list)) != len(pod_id_list)
            or len(set(account_route_list)) != len(account_route_list)):
        raise ValueError("LIVE Pod ownership is ambiguous.")
    target_dict = {}
    source_row_list = (workspace_dict.get("summary_dict") or {}).get("pod_row_dict_list") or []
    for account_dict in account_list:
        pod_id_str = account_dict["pod_id"]
        target_obj = provider_obj.get_target_for_pod(pod_id_str)
        release_obj = getattr(target_obj, "release_obj", None)
        if (release_obj is None or release_obj.enabled_bool is not True
                or release_obj.mode_str != "live" or release_obj.pod_id_str != pod_id_str
                or release_obj.account_route_str != account_dict["account_route"]
                or not release_obj.user_id_str or not release_obj.release_id_str
                or not getattr(target_obj, "db_path_str", "")):
            raise ValueError("Current LIVE configuration could not be verified.")
        match_list = [row_dict for row_dict in source_row_list if isinstance(row_dict, dict)
            and row_dict.get("pod_id_str") == pod_id_str]
        if len(match_list) > 1:
            raise ValueError("Saved LIVE identity is ambiguous.")
        if match_list:
            row_dict = match_list[0]
            if any(row_dict.get(field_str) and row_dict[field_str] != getattr(release_obj, field_str)
                    for field_str in ("user_id_str", "account_route_str", "release_id_str", "mode_str")):
                raise ValueError("Current and saved LIVE identities differ.")
            if row_dict.get("db_path_str") and Path(row_dict["db_path_str"]).resolve() != Path(target_obj.db_path_str).resolve():
                raise ValueError("Current and saved state paths differ.")
        target_dict[pod_id_str] = target_obj
    if len({target_obj.release_obj.user_id_str for target_obj in target_dict.values()}) != 1:
        raise ValueError("The LIVE workspace must have one owner.")
    return account_list, target_dict


def resolve_tools_target_obj(workspace_dict, provider_obj, pod_id_str):
    """Resolve exactly one current enabled LIVE target without inspecting other modes."""
    account_list, scoped_row_list = _scope_rows_tuple(workspace_dict)
    if not pod_id_str or pod_id_str not in {account_dict["pod_id"] for account_dict in account_list}:
        raise ValueError("Choose an enabled LIVE Pod from this workspace.")
    row_dict = next(row_dict for row_dict in scoped_row_list if row_dict["pod_id_str"] == pod_id_str)
    target_obj = provider_obj.get_target_for_pod(pod_id_str)
    release_obj = getattr(target_obj, "release_obj", None)
    if release_obj is None or release_obj.enabled_bool is not True or release_obj.mode_str != "live":
        raise ValueError("This Pod is not an enabled LIVE target.")
    if any(getattr(release_obj, field_str, None) != row_dict[field_str] for field_str in (
            "user_id_str", "pod_id_str", "account_route_str", "release_id_str", "mode_str")):
        raise ValueError("The current LIVE release changed. Refresh before continuing.")
    if not getattr(target_obj, "db_path_str", ""):
        raise ValueError("The Pod state path could not be verified.")
    if row_dict.get("db_path_str") and Path(row_dict["db_path_str"]).resolve() != Path(target_obj.db_path_str).resolve():
        raise ValueError("The Pod state path changed. Refresh before continuing.")
    return target_obj


def _parameter_dict(name_str, label_str, flag_str, *, required_bool=False):
    return {"name_str": name_str, "label_str": label_str, "flag_str": flag_str,
        "type_str": "number", "required_bool": required_bool, "placeholder_str": "",
        "min_int": 1, "max_int": 2147483647}


def _parameters_list(key_str):
    if key_str == "doctor":
        return [_parameter_dict("broker_client_id_int", "Unused broker client ID", "--broker-client-id", required_bool=True)]
    if key_str == "show_decision_plan":
        return [_parameter_dict("decision_plan_id_int", "Decision ID (optional; latest by default)", "--decision-plan-id")]
    if key_str in {"show_vplan", "execution_report", "export_trade_sheet"}:
        return [_parameter_dict("vplan_id_int", "Plan ID (optional; latest by default)", "--vplan-id")]
    return []


def _arguments_list(key_str, family_str, scope_str, target_obj, path_dict):
    if family_str in {"manual", "saved"}:
        return []
    module_str = {"runner": "alpha.live.runner", "scheduler": "alpha.live.scheduler_service",
        "bundle": "scripts.live_debug.collect_vps_debug_bundle", "watchdog": "scripts.live_ops_watchdog",
        "norgate": "scripts.doctor_norgate_client"}[family_str]
    argument_list = ["uv", "run", "python", "-m", module_str]
    if family_str in {"runner", "scheduler"}:
        argument_list.append(key_str)
    if family_str != "norgate":
        argument_list.extend(["--mode", "live"])
    argument_list.extend(["--releases-root", path_dict["releases_root_str"]])
    if scope_str == "pod":
        argument_list.extend(["--pod-id", target_obj.release_obj.pod_id_str,
            "--db-path", path_dict["db_path_str"]])
    if key_str in {"ops_report", "live_ops_watchdog"} and path_dict["config_path_str"]:
        argument_list.extend(["--dashboard-config", path_dict["config_path_str"]])
    if key_str == "live_ops_watchdog" and path_dict["log_path_str"]:
        argument_list.extend(["--output-path", str(Path(path_dict["log_path_str"]).parent / "ops_report_latest.json")])
    if family_str in {"runner", "scheduler"} and path_dict["log_path_str"]:
        argument_list.extend(["--log-path", path_dict["log_path_str"]])
    if key_str == "compare_reference":
        argument_list.append("--html")
        if path_dict["results_root_str"]:
            argument_list.extend(["--output-dir", path_dict["results_root_str"]])
    # No universal --json: Norgate's diagnostic CLI does not support that flag.
    if family_str != "norgate" and key_str != "serve":
        argument_list.append("--json")
    return argument_list


def build_tools_page_dict(workspace_dict, provider_obj, *, selected_pod_str="",
                          actions_enabled_bool=False, demo_bool=False):
    """Build copyable fixed commands and expose only the existing execution allowlist."""
    error_str, scope_verified_bool, execution_verified_bool, target_obj = "", False, False, None
    account_list = []
    try:
        account_list, target_dict = _copy_scope_tuple(workspace_dict, provider_obj)
        scope_verified_bool = True
        if selected_pod_str:
            if selected_pod_str not in target_dict:
                raise ValueError("Choose an enabled LIVE Pod.")
            target_obj = target_dict[selected_pod_str]
    except (ValueError, OSError, AttributeError, TypeError, KeyError):
        error_str = "LIVE Pod identity is unavailable or changed. Refresh and choose a Pod."
    if target_obj is not None:
        try:
            resolve_tools_target_obj(workspace_dict, provider_obj, selected_pod_str)
            execution_verified_bool = True
        except (ValueError, OSError, AttributeError, TypeError, KeyError):
            pass  # Copy diagnostics can help restore missing saved evidence.
    path_dict = {"releases_root_str": str(getattr(provider_obj, "releases_root_path_str", "") or ""),
        "config_path_str": str(getattr(provider_obj, "config_path_str", "") or ""),
        "results_root_str": str(getattr(provider_obj, "results_root_path_str", "") or ""),
        "log_path_str": str(getattr(provider_obj, "event_log_path_str", "") or ""),
        "db_path_str": str(getattr(target_obj, "db_path_str", "") or "")}
    if demo_bool:
        # Display-only placeholders, explicitly marked as synthetic. Never leak
        # this workstation's real paths into a demonstration command.
        path_dict = {"releases_root_str": "DEMO/releases", "config_path_str": "DEMO/dashboard.json",
            "results_root_str": "DEMO/results", "log_path_str": "DEMO/events.jsonl",
            "db_path_str": "DEMO/state/" + selected_pod_str + ".sqlite3"}
    block_list = [{"key_str": "read", "label_str": "Read · never trades", "group_list": []},
        {"key_str": "act", "label_str": "Act · changes state", "group_list": []}]
    for key_str, block_str, group_str, class_str, effect_str, family_str, scope_str in TOOL_CATALOG_TUPLE:
        available_bool = bool(scope_verified_bool and path_dict["releases_root_str"]
            and (scope_str == "system" or target_obj is not None))
        argument_list = _arguments_list(key_str, family_str, scope_str, target_obj, path_dict) if available_bool else []
        href_str = "/system" if key_str in {"ops_report", "saved_watchdog_report"} else ""
        if target_obj is not None and key_str in {"status", "show_decision_plan", "show_vplan", "execution_report"}:
            tab_str = {"status": "events", "show_decision_plan": "decision", "show_vplan": "plan", "execution_report": "fills"}[key_str]
            href_str = "/pods/" + quote(selected_pod_str, safe="") + "?tab=" + tab_str + "#evidence"
        reason_str = ("Choose a LIVE Pod." if not selected_pod_str and scope_str == "pod" else
            "LIVE scope or command paths could not be verified.") if not available_bool else ""
        if family_str == "manual":
            reason_str = reason_str or "Use the dashboard ticket; no standalone CLI is provided."
        row_dict = {"key_str": key_str, "label_str": "Saved watchdog report" if family_str == "saved" else key_str,
            "class_str": class_str, "effect_str": effect_str,
            "scope_str": "Installation · all configured releases" if family_str == "norgate" else
                "Installation · all LIVE Pods" if scope_str == "system" else "Selected LIVE Pod",
            "command_str": powershell_command_str(argument_list) if argument_list else "",
            "argument_list": argument_list, "parameter_list": _parameters_list(key_str),
            "runnable_bool": bool(available_bool and execution_verified_bool
                and actions_enabled_bool and key_str in EXECUTABLE_ACTION_SET),
            "action_str": key_str if key_str in EXECUTABLE_ACTION_SET else "",
            "copy_available_bool": bool(argument_list), "unavailable_reason_str": reason_str, "href_str": href_str}
        block_dict = next(block_dict for block_dict in block_list if block_dict["key_str"] == block_str)
        if not block_dict["group_list"] or block_dict["group_list"][-1]["label_str"] != group_str:
            block_dict["group_list"].append({"label_str": group_str, "row_list": []})
        block_dict["group_list"][-1]["row_list"].append(row_dict)
    selected_account_dict = next((account_dict for account_dict in account_list
        if account_dict["pod_id"] == selected_pod_str), {})
    account_str = str(selected_account_dict.get("account_route") or "")
    return {"selected_pod_str": selected_pod_str, "selected_pod_dict": {
            "pod_id_str": selected_pod_str, "label_str": selected_account_dict.get("display_name", selected_pod_str),
            "account_str": "···" + account_str[-3:] if account_str else "", "mode_str": "LIVE"},
        "pod_option_list": [{"pod_id_str": account_dict["pod_id"],
            "label_str": account_dict.get("display_name") or account_dict["pod_id"]} for account_dict in account_list],
        "block_list": block_list, "error_str": error_str, "demo_bool": demo_bool,
        "actions_enabled_bool": actions_enabled_bool,
        "manual_confirmation_str": "SUBMIT MANUAL ORDER"}
