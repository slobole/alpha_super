"""Audited terminal evidence for capsule/CORE5 requests that were never sent.

Broker absence is conclusive only after the original session closes and a fresh,
complete account query covers that request. Local dispatch proof is separate:
the sender must establish that placeOrder was never entered for those requests.
Neither path creates fictitious broker orders or executions.
"""
from dataclasses import asdict
from contextlib import nullcontext
from datetime import datetime
import json

from alpha.live import scheduler_utils
from alpha.live.core5_adapter import CORE5_STRATEGY_IMPORT_STR
from alpha.live.execution_engine import build_broker_order_request_list_from_vplan
from alpha.live.mr_capsule_adapter import MR_CAPSULE_STRATEGY_IMPORT_TUPLE


def ensure_execution_resolution_schema(connection_obj):
    connection_obj.execute("""CREATE TABLE IF NOT EXISTS vplan_execution_resolution (
        vplan_id_int INTEGER NOT NULL, order_request_key_str TEXT NOT NULL,
        asset_str TEXT NOT NULL, resolution_str TEXT NOT NULL,
        reason_str TEXT NOT NULL, evidence_json_str TEXT NOT NULL,
        created_timestamp_str TEXT NOT NULL,
        PRIMARY KEY (vplan_id_int, order_request_key_str))""")


def load_request_resolution_dict(state_store_obj, vplan_obj, *, connection_obj=None):
    with (state_store_obj._connect() if connection_obj is None else nullcontext(connection_obj)) as connection_obj:
        ensure_execution_resolution_schema(connection_obj)
        row_list = connection_obj.execute("SELECT * FROM vplan_execution_resolution WHERE vplan_id_int = ?",
            (vplan_obj.vplan_id_int,)).fetchall()
    return {row_obj["order_request_key_str"]: {
        **{key_str: row_obj[key_str] for key_str in row_obj.keys() if key_str != "evidence_json_str"},
        "evidence_dict": json.loads(row_obj["evidence_json_str"])} for row_obj in row_list}


def _aware_timestamp(timestamp_str):
    timestamp_ts = datetime.fromisoformat(timestamp_str)
    if timestamp_ts.tzinfo is None or timestamp_ts.utcoffset() is None:
        raise ValueError("Execution evidence timestamps must be timezone aware.")
    return timestamp_ts


def _validate_scope(release_obj, vplan_obj):
    if release_obj.strategy_import_str not in (*MR_CAPSULE_STRATEGY_IMPORT_TUPLE, CORE5_STRATEGY_IMPORT_STR):
        raise ValueError("Execution resolution is restricted to capsule and CORE5.")
    if (release_obj.release_id_str, release_obj.pod_id_str, release_obj.account_route_str) != (
            vplan_obj.release_id_str, vplan_obj.pod_id_str, vplan_obj.account_route_str):
        raise ValueError("Execution resolution account/pod/release mismatch.")


def _request_identity_tuple(request_obj):
    # Dispatch deadlines are added after the frozen VPlan is built. They do not
    # change the account, quantity, side, order type or durable orderRef identity.
    return tuple((key_str, value_obj) for key_str, value_obj in asdict(request_obj).items()
        if key_str not in {"submission_deadline_timestamp_str", "execution_deadline_timestamp_str"})


def _persist_resolution_list(state_store_obj, release_obj, vplan_obj, request_list,
        as_of_ts, reason_str, resolution_str, evidence_dict, *, connection_obj=None):
    _validate_scope(release_obj, vplan_obj)
    _aware_timestamp(as_of_ts.isoformat())
    if not reason_str.strip():
        raise ValueError("Execution resolution requires an audit reason.")
    with (state_store_obj._connect() if connection_obj is None else nullcontext(connection_obj)) as connection_obj:
        ensure_execution_resolution_schema(connection_obj)
        if not connection_obj.in_transaction:
            connection_obj.execute("BEGIN IMMEDIATE")
        row_obj = connection_obj.execute("SELECT * FROM vplan WHERE vplan_id_int = ?",
            (vplan_obj.vplan_id_int,)).fetchone()
        if row_obj is None or row_obj["status_str"] not in {"ready", "submitting", "submitted"}:
            return
        stored_vplan_obj = state_store_obj._row_to_vplan(row_obj)
        if stored_vplan_obj.submission_key_str != vplan_obj.submission_key_str:
            raise ValueError("Execution resolution VPlan identity changed.")
        expected_request_dict = {request_obj.order_request_key_str: request_obj
            for request_obj in build_broker_order_request_list_from_vplan(stored_vplan_obj)}
        supplemental_row_list = connection_obj.execute(
            "SELECT * FROM mr_capsule_execution_request WHERE vplan_id_int = ?",
            (vplan_obj.vplan_id_int,)).fetchall()
        from alpha.live.models import BrokerOrderRequest
        for supplemental_row_obj in supplemental_row_list:
            if supplemental_row_obj["request_kind_str"] == "recovery":
                request_obj = BrokerOrderRequest(**json.loads(supplemental_row_obj["request_json_str"]))
                expected_request_dict[request_obj.order_request_key_str] = request_obj
        for request_obj in request_list:
            expected_request_obj = expected_request_dict.get(request_obj.order_request_key_str)
            if expected_request_obj is None or _request_identity_tuple(request_obj) != _request_identity_tuple(expected_request_obj):
                raise ValueError("Execution resolution request differs from persisted intent.")
            if connection_obj.execute("SELECT 1 FROM vplan_broker_order WHERE vplan_id_int = ? AND order_request_key_str = ?",
                    (vplan_obj.vplan_id_int, request_obj.order_request_key_str)).fetchone() is not None:
                continue
            # An orphan execution on the same asset can belong to this request.
            if connection_obj.execute("""SELECT 1 FROM vplan_fill f LEFT JOIN vplan_broker_order o
                    ON f.vplan_id_int = o.vplan_id_int AND f.broker_order_id_str = o.broker_order_id_str
                    WHERE f.vplan_id_int = ? AND f.asset_str = ?
                    AND (o.order_request_key_str IS NULL OR o.order_request_key_str = ?)""",
                    (vplan_obj.vplan_id_int, request_obj.asset_str, request_obj.order_request_key_str)).fetchone() is not None:
                continue
            connection_obj.execute("""INSERT OR IGNORE INTO vplan_execution_resolution
                (vplan_id_int, order_request_key_str, asset_str, resolution_str, reason_str,
                 evidence_json_str, created_timestamp_str) VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (vplan_obj.vplan_id_int, request_obj.order_request_key_str, request_obj.asset_str,
                 resolution_str, reason_str, json.dumps(evidence_dict, sort_keys=True, allow_nan=False), as_of_ts.isoformat()))


def record_never_dispatched_requests(state_store_obj, release_obj, vplan_obj, request_list, as_of_ts, reason_str, *, connection_obj=None):
    """Only call with keys proved not to have entered the broker placeOrder call."""
    _persist_resolution_list(state_store_obj, release_obj, vplan_obj, request_list, as_of_ts,
        reason_str, "never_dispatched", {"source_str": "local_dispatch_before_placeOrder",
            "recorded_timestamp_str": as_of_ts.isoformat(),
            "order_request_key_list": [request_obj.order_request_key_str for request_obj in request_list]},
        connection_obj=connection_obj)
    return load_request_resolution_dict(state_store_obj, vplan_obj, connection_obj=connection_obj)


def resolve_never_sent_requests(state_store_obj, broker_adapter_obj, release_obj, vplan_obj, request_list, as_of_ts):
    _validate_scope(release_obj, vplan_obj)
    resolution_dict = load_request_resolution_dict(state_store_obj, vplan_obj)
    target_ts = scheduler_utils.to_market_timestamp_ts(vplan_obj.target_execution_timestamp_ts,
        release_obj.session_calendar_id_str)
    close_ts = scheduler_utils.get_session_close_timestamp_ts(target_ts.date(), release_obj.session_calendar_id_str)
    if as_of_ts < close_ts:
        return resolution_dict
    order_row_list = state_store_obj.get_broker_order_row_dict_list_for_vplan(vplan_obj.vplan_id_int)
    known_key_set = {row_dict["order_request_key_str"] for row_dict in order_row_list}
    known_key_by_id_dict = {str(row_dict["broker_order_id_str"]): row_dict["order_request_key_str"]
        for row_dict in order_row_list if row_dict.get("order_request_key_str")}
    missing_request_list = [request_obj for request_obj in request_list
        if request_obj.order_request_key_str not in known_key_set | set(resolution_dict)
        and not request_obj.order_request_key_str.startswith(f"{vplan_obj.submission_key_str}:manual:")]
    if not missing_request_list:
        return resolution_dict
    evidence_dict = broker_adapter_obj.get_refreshed_order_evidence(vplan_obj.account_route_str, vplan_obj.submission_timestamp_ts)
    if evidence_dict.get("account_route_str") != vplan_obj.account_route_str or not evidence_dict.get("source_str"):
        raise ValueError("Execution absence proof has wrong account or no source.")
    if not all(evidence_dict.get(field_str) is True for field_str in (
            "open_orders_complete_bool", "completed_orders_complete_bool", "executions_complete_bool")):
        raise ValueError("Execution absence proof is incomplete.")
    started_ts = _aware_timestamp(evidence_dict["refresh_started_timestamp_str"])
    refreshed_ts = _aware_timestamp(evidence_dict["refreshed_timestamp_str"])
    coverage_ts = _aware_timestamp(evidence_dict["coverage_since_timestamp_str"])
    if started_ts < max(as_of_ts, close_ts) or refreshed_ts < started_ts or coverage_ts > vplan_obj.submission_timestamp_ts:
        raise ValueError("Execution absence proof is stale or does not cover this submission.")
    with state_store_obj._connect() as connection_obj:
        claimed_timestamp_dict = {row_obj["order_request_key_str"]: _aware_timestamp(row_obj["claimed_timestamp_str"])
            for row_obj in connection_obj.execute("SELECT order_request_key_str, claimed_timestamp_str FROM mr_capsule_execution_request WHERE vplan_id_int = ?",
                (vplan_obj.vplan_id_int,))}
    observed_order_list = evidence_dict["order_row_list"]
    execution_row_list = evidence_dict["execution_row_list"]
    if not isinstance(observed_order_list, list) or not isinstance(execution_row_list, list):
        raise ValueError("Execution absence proof requires complete order and execution lists.")
    if any(not isinstance(row_dict, dict) or not row_dict.get("broker_order_id_str")
            or not row_dict.get("asset_str") for row_dict in [*observed_order_list, *execution_row_list]):
        raise ValueError("Execution absence proof has an unidentified order or execution.")
    ack_row_list = state_store_obj.get_broker_ack_row_dict_list_for_vplan(vplan_obj.vplan_id_int)
    absent_request_list = []
    for request_obj in missing_request_list:
        claimed_ts = claimed_timestamp_dict.get(request_obj.order_request_key_str, vplan_obj.submission_timestamp_ts)
        if not coverage_ts <= claimed_ts <= started_ts:
            continue
        matching_ack_list = [row_dict for row_dict in ack_row_list
            if row_dict.get("order_request_key_str") == request_obj.order_request_key_str]
        if any(row_dict.get("broker_response_ack_bool") is True for row_dict in matching_ack_list):
            continue  # A previously observed broker response contradicts never-sent.
        known_order_id_set = {str(row_dict[field_str]) for row_dict in matching_ack_list
            for field_str in ("broker_order_id_str", "perm_id_int") if row_dict.get(field_str)}
        # Missing orderRef on a same-asset observation is ambiguous, never proof.
        if any(row_dict.get("order_request_key_str") == request_obj.order_request_key_str
                or str(row_dict.get("broker_order_id_str")) in known_order_id_set
                or (not row_dict.get("order_request_key_str") and row_dict.get("asset_str") == request_obj.asset_str
                    and str(row_dict.get("broker_order_id_str")) not in known_key_by_id_dict)
                for row_dict in [*observed_order_list, *execution_row_list]):
            continue
        absent_request_list.append(request_obj)
    _persist_resolution_list(state_store_obj, release_obj, vplan_obj, absent_request_list, as_of_ts,
        "Target session closed; refreshed broker orders and executions prove this orderRef was never sent.",
        "never_sent", evidence_dict)
    return load_request_resolution_dict(state_store_obj, vplan_obj)
