"""Optional execution-bound authorization for operator dashboard commands.

CLI and scheduled runs have no context and retain their existing behavior.
This guards an approved target; it is not a global lock on scheduler writes.
"""

from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from dataclasses import asdict
from datetime import datetime
import hashlib
from functools import wraps
from inspect import signature
import json
from pathlib import Path
import sqlite3
import time

from alpha.live.state_store_v2 import LiveStateStore


_authorization_context_obj = ContextVar("dashboard_authorization", default=None)


def _json_default(value_obj):
    if isinstance(value_obj, datetime):
        return value_obj.isoformat()
    if isinstance(value_obj, set):
        return sorted(value_obj)
    raise TypeError("Unsupported confirmation value")


def confirmation_hash_str(value_obj):
    return hashlib.sha256(json.dumps(value_obj, sort_keys=True, allow_nan=False,
        separators=(",", ":"), default=_json_default).encode("utf-8")).hexdigest()


class _ReadOnlyConfirmationStore(LiveStateStore):
    """Use the existing decoders inside one read-only SQLite snapshot."""

    def __init__(self, db_path_str):
        self.connection_obj = sqlite3.connect(Path(db_path_str).resolve().as_uri() + "?mode=ro", uri=True)
        self.connection_obj.row_factory = sqlite3.Row
        self.connection_obj.execute("BEGIN")

    def _connect(self):
        return nullcontext(self.connection_obj)


def read_confirmation_context_dict(target_obj):
    try:
        return _read_confirmation_context_dict(target_obj)
    except (sqlite3.Error, OSError, ValueError, TypeError, KeyError, IndexError) as exception_obj:
        raise ValueError("Confirmation evidence is missing, invalid or does not match this account. Review saved diagnostics before retrying.") from exception_obj


def _read_confirmation_context_dict(target_obj):
    release_obj = target_obj.release_obj
    state_store_obj = _ReadOnlyConfirmationStore(target_obj.db_path_str)
    try:
        decision_obj = state_store_obj.get_latest_decision_plan_for_pod(release_obj.pod_id_str)
        vplan_obj = state_store_obj.get_latest_vplan_for_pod(release_obj.pod_id_str)
        submitted_list = [item_obj for item_obj in state_store_obj.get_submitted_vplan_list()
                          if item_obj.pod_id_str == release_obj.pod_id_str]
        pod_state_obj = state_store_obj.get_pod_state(release_obj.pod_id_str)
        for plan_obj in [decision_obj, vplan_obj, *submitted_list]:
            if plan_obj is not None:
                _assert_plan_identity(plan_obj, asdict(release_obj),
                    require_release_bool=plan_obj.status_str not in {"completed", "expired", "blocked"})
        if pod_state_obj is not None and pod_state_obj.account_route_str != release_obj.account_route_str:
            raise ValueError("Saved state belongs to a different account.")
        state_dict = {
            "decision_dict": asdict(decision_obj) if decision_obj else None,
            "vplan_dict": asdict(vplan_obj) if vplan_obj else None,
            "submitted_list": [asdict(item_obj) for item_obj in submitted_list],
            "pod_state_dict": asdict(pod_state_obj) if pod_state_obj else None,
        }
    finally:
        state_store_obj.connection_obj.close()
    return {
        "pod_id_str": release_obj.pod_id_str,
        "mode_str": release_obj.mode_str,
        "account_route_str": release_obj.account_route_str,
        "release_id_str": release_obj.release_id_str,
        "release_hash_str": confirmation_hash_str(asdict(release_obj)),
        "db_path_str": str(Path(target_obj.db_path_str).resolve()),
        "state_hash_str": confirmation_hash_str(state_dict),
        "decision_plan_id_int": decision_obj.decision_plan_id_int if decision_obj else None,
        "vplan_id_int": vplan_obj.vplan_id_int if vplan_obj else None,
        "submit_hash_list": [confirmation_hash_str(asdict(vplan_obj))]
            if vplan_obj and vplan_obj.status_str == "ready" and release_obj.auto_submit_enabled_bool else [],
        "submitted_hash_list": sorted(confirmation_hash_str(asdict(item_obj)) for item_obj in submitted_list),
    }


def validate_current_confirmation(target_obj, expected_dict):
    if time.monotonic() >= expected_dict["expires_at_float"]:
        raise ValueError("Confirmation expired. Open a new preview.")
    current_dict = read_confirmation_context_dict(target_obj)
    if any(current_dict[key_str] != expected_dict[key_str] for key_str in current_dict):
        raise ValueError("Target or saved execution state changed. Open a new preview.")


@contextmanager
def operator_execution_context(target_obj, target_loader_fn, action_name_str):
    expected_dict = target_obj.operator_confirmation_dict
    if expected_dict is None:
        yield
        return
    if expected_dict["action_name_str"] != action_name_str:
        raise ValueError("Dispatch action does not match the confirmed action.")
    current_target_obj = target_loader_fn(target_obj.release_obj.pod_id_str)
    if current_target_obj is None:
        raise ValueError("Confirmed target is no longer enabled.")
    validate_current_confirmation(current_target_obj, expected_dict)
    context_token_obj = _authorization_context_obj.set(expected_dict)
    try:
        yield
    finally:
        _authorization_context_obj.reset(context_token_obj)


def assert_authorized_release(release_obj):
    expected_dict = _authorization_context_obj.get()
    if expected_dict is not None and confirmation_hash_str(asdict(release_obj)) != expected_dict["release_hash_str"]:
        raise ValueError("Release changed after confirmation; execution stopped.")


def guard_dashboard_execution(command_fn):
    signature_obj = signature(command_fn)
    @wraps(command_fn)
    def guarded_fn(pod_target_obj, *arg_tuple, **keyword_dict):
        expected_dict = pod_target_obj.operator_confirmation_dict
        if expected_dict is None:
            return command_fn(pod_target_obj, *arg_tuple, **keyword_dict)
        from alpha.live.dashboard import DashboardApp

        app_obj = DashboardApp(releases_root_path_str=expected_dict["releases_root_path_str"],
                               config_path_str=expected_dict["config_path_str"])
        bound_dict = signature_obj.bind(pod_target_obj, *arg_tuple, **keyword_dict).arguments
        action_name_str = bound_dict.get("action_name_str", "compare_reference")
        with operator_execution_context(pod_target_obj, app_obj.get_target_for_pod, action_name_str):
            return command_fn(pod_target_obj, *arg_tuple, **keyword_dict)
    return guarded_fn


def assert_authorized_release_list(release_list):
    if _authorization_context_obj.get() is None:
        return
    enabled_list = [release_obj for release_obj in release_list if release_obj.enabled_bool]
    if len(enabled_list) != 1:
        raise ValueError("Confirmation permits exactly one enabled release.")
    enabled_obj = enabled_list[0]
    if sum(release_obj.release_id_str == enabled_obj.release_id_str for release_obj in release_list) != 1:
        raise ValueError("Ambiguous release ID would overwrite the approved release.")
    assert_authorized_release(enabled_obj)


def assert_authorized_vplans(action_name_str, vplan_list):
    expected_dict = _authorization_context_obj.get()
    if expected_dict is None:
        return
    for plan_obj in vplan_list:
        _assert_plan_identity(plan_obj, expected_dict)
    if expected_dict["action_name_str"] != action_name_str:
        return
    expected_hash_list = expected_dict["submit_hash_list" if action_name_str == "submit_vplan" else "submitted_hash_list"]
    if sorted(confirmation_hash_str(asdict(item_obj)) for item_obj in vplan_list) != expected_hash_list:
        raise ValueError("Confirmed execution plan changed; execution stopped.")


def _assert_plan_identity(plan_obj, expected_dict, require_release_bool=True):
    key_tuple = ("pod_id_str", "account_route_str", "release_id_str") if require_release_bool else ("pod_id_str", "account_route_str")
    for key_str in key_tuple:
        if getattr(plan_obj, key_str) != expected_dict[key_str]:
            raise ValueError("Saved execution plan does not match the confirmed Pod/account/release.")
