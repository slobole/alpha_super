"""Synthetic Tools execution for the local demo; no production executor."""

from dataclasses import asdict
from datetime import UTC, datetime
import hashlib
import json
import uuid


def assert_same_target(target_obj, current_obj):
    if (current_obj is None or not current_obj.release_obj.enabled_bool
            or current_obj.release_obj.mode_str != "live"
            or asdict(current_obj.release_obj) != asdict(target_obj.release_obj)
            or current_obj.db_path_str != target_obj.db_path_str):
        raise ValueError("The enabled LIVE target changed. Open a new preview.")


class SyntheticToolsActionProvider:
    """In-memory demonstration only. Never constructs an executor or broker."""

    synthetic_bool = True

    def __init__(self, target_scope_fn):
        self.target_scope_fn = target_scope_fn
        self.job_dict = {}
        self.journal_list = []

    def get_confirmation_context_dict(self, target_obj):
        assert_same_target(target_obj, self.target_scope_fn(target_obj.release_obj.pod_id_str))
        return {"pod_id_str": target_obj.release_obj.pod_id_str, "mode_str": "live",
            "account_route_str": target_obj.release_obj.account_route_str,
            "release_hash_str": release_hash_str(target_obj),
            "db_path_str": target_obj.db_path_str, "state_hash_str": "synthetic",
            "decision_plan_id_int": None, "vplan_id_int": None}

    def start_action_job(self, action_name_str, target_obj):
        job_id_str = uuid.uuid4().hex
        result_dict = {"job_id_str": job_id_str, "pod_id_str": target_obj.release_obj.pod_id_str,
            "mode_str": "live", "action_name_str": action_name_str, "status_str": "succeeded",
            "completed_timestamp_str": datetime.now(UTC).isoformat()}
        if action_name_str == "submit_vplan":
            result_dict["vplan_id_int"] = target_obj.operator_confirmation_dict["requested_vplan_id_int"]
        if len(self.job_dict) >= 200:
            self.job_dict.pop(next(iter(self.job_dict)))
        self.job_dict[job_id_str] = result_dict
        return result_dict

    def start_diff_job(self, target_obj):
        return self.start_action_job("compare_reference", target_obj)

    def submit_manual_order_dict(self, target_obj, manual_order_dict):
        return {"submit_ack_status_str": "acknowledged", "broker_order_ack_count_int": 1}

    def get_job_dict(self, job_id_str):
        return self.job_dict.get(job_id_str)


def release_hash_str(target_obj):
    return hashlib.sha256(json.dumps(asdict(target_obj.release_obj), sort_keys=True,
        default=str, allow_nan=False).encode()).hexdigest()
