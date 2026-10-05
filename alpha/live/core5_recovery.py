"""CORE5 terminal execution evidence, without advancing strategy memory."""
from dataclasses import replace
from datetime import datetime
import json
import math

from alpha.live.core5_adapter import CORE5_STRATEGY_IMPORT_STR
from alpha.live.execution_engine import build_broker_order_request_list_from_vplan
from alpha.live.execution_resolution import resolve_never_sent_requests
from alpha.live.mr_capsule_notifications import enqueue_execution_alert
from alpha.live.reconcile import reconcile_account_state


def refresh_core5_cycle_evidence(state_store_obj, broker_adapter_obj, release_obj, vplan_obj, as_of_ts):
    if release_obj.strategy_import_str != CORE5_STRATEGY_IMPORT_STR:
        raise ValueError("CORE5 recovery cannot inspect another strategy.")
    order_list = state_store_obj.get_broker_order_row_dict_list_for_vplan(vplan_obj.vplan_id_int)
    record_list, event_list, fill_list = broker_adapter_obj.get_capsule_order_state_snapshot(
        vplan_obj.account_route_str, vplan_obj.submission_timestamp_ts,
        submission_key_str=vplan_obj.submission_key_str,
        allowed_broker_order_id_set={str(order_dict["broker_order_id_str"]) for order_dict in order_list})
    identity_dict = {"decision_plan_id_int": vplan_obj.decision_plan_id_int, "vplan_id_int": vplan_obj.vplan_id_int}
    state_store_obj.upsert_vplan_broker_order_record_list([replace(record_obj, **identity_dict) for record_obj in record_list])
    state_store_obj.insert_vplan_broker_order_event_list([replace(event_obj, **identity_dict) for event_obj in event_list])
    state_store_obj.upsert_vplan_fill_list([replace(fill_obj, **identity_dict) for fill_obj in fill_list])
    request_list = build_broker_order_request_list_from_vplan(vplan_obj)
    resolution_dict = resolve_never_sent_requests(state_store_obj, broker_adapter_obj,
        release_obj, vplan_obj, request_list, as_of_ts)
    snapshot_obj = broker_adapter_obj.get_core5_account_snapshot(vplan_obj.account_route_str)
    if (snapshot_obj.account_route_str != vplan_obj.account_route_str
            or snapshot_obj.snapshot_timestamp_ts.tzinfo is None
            or snapshot_obj.snapshot_timestamp_ts < vplan_obj.target_execution_timestamp_ts
            or not all(math.isfinite(float(value_float)) for value_float in [snapshot_obj.cash_float,
                snapshot_obj.net_liq_float, *snapshot_obj.position_amount_map.values()])
            or snapshot_obj.net_liq_float <= 0):
        raise ValueError("CORE5 recovery account observation is invalid or stale.")
    order_list = state_store_obj.get_broker_order_row_dict_list_for_vplan(vplan_obj.vplan_id_int, include_evidence_bool=True)
    fill_list = state_store_obj.get_fill_row_dict_list_for_vplan(vplan_obj.vplan_id_int,
        include_order_identity_bool=True, include_evidence_bool=True)
    return {**classify_core5_cycle_evidence(vplan_obj, snapshot_obj, order_list, fill_list, resolution_dict),
        "broker_snapshot_obj": snapshot_obj, "resolution_dict": resolution_dict}


def classify_core5_cycle_evidence(vplan_obj, snapshot_obj, order_list, fill_list, resolution_dict):
    """Expected holdings = frozen actual holdings + signed, correlated executions."""
    tolerance_float = 1e-9
    expected_position_dict = dict(vplan_obj.current_broker_position_map)
    terminal_bool = (snapshot_obj.account_route_str == vplan_obj.account_route_str
        and snapshot_obj.snapshot_timestamp_ts >= vplan_obj.target_execution_timestamp_ts
        and not snapshot_obj.open_order_id_list
        and all(math.isfinite(float(value_float)) for value_float in [snapshot_obj.cash_float,
            snapshot_obj.net_liq_float, *snapshot_obj.position_amount_map.values(), *expected_position_dict.values()])
        and snapshot_obj.net_liq_float > 0)
    complete_bool = terminal_bool
    matched_id_set = set()
    for request_obj in build_broker_order_request_list_from_vplan(vplan_obj):
        matching_list = [order_dict for order_dict in order_list
            if order_dict.get("order_request_key_str") == request_obj.order_request_key_str]
        request_resolution_dict = resolution_dict.get(request_obj.order_request_key_str, {})
        if (not matching_list and request_resolution_dict.get("asset_str") == request_obj.asset_str
                and request_resolution_dict.get("resolution_str") in {"never_sent", "never_dispatched"}):
            evidence_dict = request_resolution_dict.get("evidence_dict") or {}
            proof_timestamp_str = evidence_dict.get("refreshed_timestamp_str", request_resolution_dict.get("created_timestamp_str"))
            if proof_timestamp_str and snapshot_obj.snapshot_timestamp_ts >= datetime.fromisoformat(proof_timestamp_str):
                complete_bool = False
                continue
        if len(matching_list) != 1:
            terminal_bool = complete_bool = False
            continue
        order_dict = matching_list[0]
        order_id_str = str(order_dict["broker_order_id_str"])
        duplicate_bool = order_id_str in matched_id_set
        matched_id_set.add(order_id_str)
        matched_fill_list = [fill_dict for fill_dict in fill_list if str(fill_dict["broker_order_id_str"]) == order_id_str]
        filled_float = sum(float(fill_dict["fill_amount_float"]) for fill_dict in matched_fill_list)
        payload_dict = order_dict.get("raw_payload_dict") or {}
        valid_bool = (not duplicate_bool and order_dict.get("account_route_str") == vplan_obj.account_route_str
            and order_dict.get("asset_str") == request_obj.asset_str and order_dict.get("unit_str") == "shares"
            and math.isfinite(filled_float) and math.isfinite(float(order_dict["filled_amount_float"]))
            and abs(float(order_dict["amount_float"]) - request_obj.amount_float) <= tolerance_float
            and abs(abs(filled_float) - abs(float(order_dict["filled_amount_float"]))) <= tolerance_float
            and abs(filled_float) <= abs(request_obj.amount_float) + tolerance_float
            and all(fill_dict.get("account_route_str") == vplan_obj.account_route_str
                and fill_dict["asset_str"] == request_obj.asset_str
                and math.isfinite(float(fill_dict["fill_amount_float"]))
                and float(fill_dict["fill_amount_float"]) * request_obj.amount_float >= 0
                and vplan_obj.submission_timestamp_ts <= datetime.fromisoformat(fill_dict["fill_timestamp_str"])
                    <= snapshot_obj.snapshot_timestamp_ts
                for fill_dict in matched_fill_list)
            and payload_dict.get("open_order_observed_bool") is not True
            and payload_dict.get("snapshot_source_str") != "open_order"
            and payload_dict.get("completed_quantity_verified_bool") is not False)
        full_bool = abs(filled_float - request_obj.amount_float) <= tolerance_float
        terminal_order_bool = (full_bool or (order_dict["status_str"] in {
            "Cancelled", "ApiCancelled", "Rejected", "Expired", "Inactive"}
            and payload_dict.get("snapshot_source_str") == "completed_order"))
        terminal_bool = terminal_bool and valid_bool and terminal_order_bool
        complete_bool = complete_bool and valid_bool and full_bool
        expected_position_dict[request_obj.asset_str] = float(expected_position_dict.get(request_obj.asset_str, 0)) + filled_float
    if len(matched_id_set) != len(order_list) or any(str(fill_dict["broker_order_id_str"]) not in matched_id_set for fill_dict in fill_list):
        terminal_bool = complete_bool = False
    reconciliation_obj = reconcile_account_state(expected_position_dict, snapshot_obj.cash_float, snapshot_obj)
    terminal_bool = terminal_bool and reconciliation_obj.passed_bool
    return {"terminal_bool": bool(terminal_bool), "complete_bool": bool(terminal_bool and complete_bool),
        "expected_position_map_dict": expected_position_dict}


def park_core5_cycle(state_store_obj, release_obj, vplan_obj, as_of_ts, evidence_dict):
    """Close a proved terminal batch; retain prior memory for reviewed resumption."""
    if not evidence_dict["terminal_bool"]:
        raise ValueError("Cannot close an execution cycle with uncertain broker evidence.")
    payload_dict = {key_str: value_obj for key_str, value_obj in evidence_dict.items()
        if key_str != "broker_snapshot_obj"}
    payload_dict.update(severity_str="critical", reason_code_str="core5_reviewed_resume_required")
    with state_store_obj._connect() as connection_obj:
        connection_obj.execute("BEGIN IMMEDIATE")
        row_obj = connection_obj.execute("SELECT snapshot_metadata_json_str FROM decision_plan WHERE decision_plan_id_int=?",
            (vplan_obj.decision_plan_id_int,)).fetchone()
        metadata_dict = json.loads(row_obj["snapshot_metadata_json_str"])
        metadata_dict["core5_terminal_execution_dict"] = payload_dict
        metadata_dict["opening_dispatch_parked_bool"] = True
        cursor_obj = connection_obj.execute("UPDATE vplan SET status_str='parked', updated_timestamp_str=? WHERE vplan_id_int=? AND status_str IN ('submitting','submitted')",
            (as_of_ts.isoformat(), vplan_obj.vplan_id_int))
        if cursor_obj.rowcount != 1:
            raise ValueError("CORE5 execution cycle changed while parking it.")
        connection_obj.execute("UPDATE decision_plan SET status_str='blocked', snapshot_metadata_json_str=?, updated_timestamp_str=? WHERE decision_plan_id_int=?",
            (json.dumps(metadata_dict, sort_keys=True), as_of_ts.isoformat(), vplan_obj.decision_plan_id_int))
        enqueue_execution_alert(connection_obj, vplan_id_int=vplan_obj.vplan_id_int, alert_kind_str="dispatch_failed",
            pod_id_str=vplan_obj.pod_id_str, account_route_str=vplan_obj.account_route_str, mode_str=release_obj.mode_str,
            payload_dict=payload_dict, created_timestamp_ts=as_of_ts)


def record_core5_unresolved_cycle(state_store_obj, release_obj, vplan_obj, as_of_ts, reason_str):
    """A restart with missing broker evidence must never remain silently submitting."""
    if (release_obj.strategy_import_str != CORE5_STRATEGY_IMPORT_STR
            or (release_obj.pod_id_str, release_obj.account_route_str) != (vplan_obj.pod_id_str, vplan_obj.account_route_str)):
        raise ValueError("CORE5 unresolved evidence requires the matching CORE5 account.")
    payload_dict = {"severity_str": "critical", "reason_code_str": "core5_execution_evidence_pending",
        "detail_str": reason_str, "observed_timestamp_str": as_of_ts.isoformat()}
    with state_store_obj._connect() as connection_obj:
        connection_obj.execute("BEGIN IMMEDIATE")
        row_obj = connection_obj.execute("SELECT d.snapshot_metadata_json_str FROM decision_plan d JOIN vplan v ON v.decision_plan_id_int=d.decision_plan_id_int WHERE v.vplan_id_int=? AND v.status_str IN ('submitting','submitted')",
            (vplan_obj.vplan_id_int,)).fetchone()
        if row_obj is None:
            return
        metadata_dict = json.loads(row_obj["snapshot_metadata_json_str"])
        metadata_dict["core5_pending_execution_dict"] = payload_dict
        connection_obj.execute("UPDATE vplan SET status_str='submitted', updated_timestamp_str=? WHERE vplan_id_int=?",
            (as_of_ts.isoformat(), vplan_obj.vplan_id_int))
        connection_obj.execute("UPDATE decision_plan SET status_str='submitted', snapshot_metadata_json_str=?, updated_timestamp_str=? WHERE decision_plan_id_int=?",
            (json.dumps(metadata_dict, sort_keys=True), as_of_ts.isoformat(), vplan_obj.decision_plan_id_int))
        enqueue_execution_alert(connection_obj, vplan_id_int=vplan_obj.vplan_id_int, alert_kind_str="dispatch_failed",
            pod_id_str=vplan_obj.pod_id_str, account_route_str=vplan_obj.account_route_str, mode_str=release_obj.mode_str,
            payload_dict=payload_dict, created_timestamp_ts=as_of_ts)
