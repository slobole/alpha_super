"""Up to three copy-only commands that fit an attention item, safest first.

Commands come from the Tools catalog for the same Pod, so paths, IDs and quoting
are identical to the Tools page. Nothing here runs a command; the operator
copies one into the VPS terminal. No fitting command means no suggestion.
"""

import re

from alpha.live.dashboard_v4.tools import build_tools_page_dict


LIMIT_INT = 3
CLASS_RANK_DICT = {"READ": 0, "INSPECT": 1, "ACTIVE": 2}
POD_ID_RE = re.compile(r"[A-Za-z0-9._-]{1,200}")
STEP_KEY_DICT = {
    "Data": ("doctor_norgate_client",), "Decide": ("show_decision_plan",), "Plan": ("show_vplan",),
    "Submit": ("show_vplan", "execution_report"), "Fill": ("show_vplan", "execution_report"),
    "Reconcile": ("execution_report", "post_execution_reconcile"), "EOD": ("status", "eod_snapshot"),
}
HOLD_KEY_DICT = {
    "execution_exception_parked": ("show_vplan", "execution_report"),
    "manual_review_required": ("show_vplan", "submit_vplan"),
    "mr_capsule_eod_snapshot_untrusted": ("status", "show_vplan"),
}
ACTION_KEY_DICT = {
    "review_vplan": ("show_vplan", "submit_vplan"), "submit_vplan": ("show_vplan",),
    "build_vplan": ("show_decision_plan",), "missed_decision_cycle": ("next_due", "status"),
}


def serve_running_command_str(pod_id_str):
    """READ: list this Pod's running scheduler (one line = running, none = stopped, two = duplicate)."""
    if not isinstance(pod_id_str, str) or not POD_ID_RE.fullmatch(pod_id_str):
        return ""
    return ("Get-CimInstance Win32_Process -Filter \"Name LIKE 'python%.exe'\" | Where-Object { "
        "[int64]$_.WorkingSetSize -gt 20MB -and $_.CommandLine -like '*alpha.live.scheduler_service*' "
        "-and $_.CommandLine -match '(?:^|\\s)serve(?:\\s|$)' "
        "-and $_.CommandLine -match '--pod-id\\s+\"?" + re.escape(pod_id_str) + "\"?(?:\\s|$)' } "
        "| Select-Object ProcessId, CreationDate")


def next_command_key_list(attention_dict):
    """Catalog keys that fit this item, most relevant first, without duplicates."""
    kind_str = attention_dict.get("kind_str")
    if kind_str == "scheduler":
        key_tuple = ("serve_running", "next_due")
    elif kind_str == "hold":
        key_tuple = HOLD_KEY_DICT.get(attention_dict.get("reason_code_str") or "", ("status", "show_vplan"))
    elif kind_str == "database":
        key_tuple = ("status",)
    elif kind_str in {"action", "cycle"}:
        key_tuple = ((attention_dict.get("inspect_str") or "",)
            + ACTION_KEY_DICT.get(attention_dict.get("next_action_str") or "", ())
            + STEP_KEY_DICT.get(attention_dict.get("step_str") or "", ()))
    else:
        return []
    return list(dict.fromkeys(key_str for key_str in key_tuple if key_str))


def attach_next_command_list(attention_list, workspace_dict, provider_obj, *, demo_bool, as_of_ts):
    """Set next_command_list on each item: READ, then INSPECT, then ACTIVE; at most three."""
    row_cache_dict = {}
    for attention_dict in attention_list:
        if not attention_dict or "next_command_list" in attention_dict:
            continue
        pod_id_str, key_list = attention_dict.get("pod_id_str"), next_command_key_list(attention_dict)
        command_list = []
        if pod_id_str and key_list:
            if pod_id_str not in row_cache_dict:
                tools_dict = build_tools_page_dict(workspace_dict, provider_obj, selected_pod_str=pod_id_str,
                    demo_bool=demo_bool, as_of_ts=as_of_ts)
                # An unverified scope suggests nothing rather than a guess.
                row_cache_dict[pod_id_str] = {} if tools_dict["error_str"] else {row_dict["key_str"]: row_dict
                    for block_dict in tools_dict["block_list"] for group_dict in block_dict["group_list"]
                    for row_dict in group_dict["row_list"]}
            row_by_key_dict = row_cache_dict[pod_id_str]
            for key_str in key_list:
                if key_str == "serve_running":
                    command_str = serve_running_command_str(pod_id_str) if row_by_key_dict else ""
                    if command_str:
                        command_list.append({"key_str": key_str, "class_str": "READ", "orders_bool": False,
                            "command_str": command_str, "effect_str": "Show whether this Pod's scheduler process is running."})
                    continue
                row_dict = row_by_key_dict.get(key_str)
                if row_dict and row_dict["command_str"]:
                    command_list.append({key_name_str: row_dict[key_name_str] for key_name_str in
                        ("key_str", "class_str", "orders_bool", "command_str", "effect_str")})
        # Stable sort: within a class, the most relevant command stays first.
        command_list.sort(key=lambda command_dict: CLASS_RANK_DICT.get(command_dict["class_str"], 9))
        attention_dict["next_command_list"] = command_list[:LIMIT_INT]
    return attention_list
