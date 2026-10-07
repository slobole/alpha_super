"""Opening-batch retries and failure reporting for the daily reconciliation path."""
from datetime import UTC, datetime
from dataclasses import replace
import json

from alpha.live.core5_adapter import CORE5_STRATEGY_IMPORT_STR
from alpha.live.guarded_dispatch import DispatchFailure, is_transient_broker_error_bool, moo_dispatch_deadline_ts
from alpha.live.mr_capsule_adapter import MR_CAPSULE_STRATEGY_IMPORT_TUPLE


def is_guarded_release_bool(release_obj):
    return (release_obj.strategy_import_str == CORE5_STRATEGY_IMPORT_STR
        or release_obj.strategy_import_str in MR_CAPSULE_STRATEGY_IMPORT_TUPLE)


def record_dispatch_failure(state_store_obj, release_obj, vplan_obj, request_list,
        exception_obj, as_of_ts, *, before_send_bool=False, claim_owned_bool=False):
    """Retry a pre-send transient error; otherwise leave the cycle for reconciliation."""

    if not is_guarded_release_bool(release_obj):
        raise ValueError("Dispatch recovery is limited to CORE5 and capsules.")
    if isinstance(exception_obj, DispatchFailure):
        result_obj = exception_obj.partial_result_obj
        identity_dict = {"decision_plan_id_int": vplan_obj.decision_plan_id_int, "vplan_id_int": vplan_obj.vplan_id_int}
        state_store_obj.upsert_vplan_broker_order_record_list([replace(record_obj, **identity_dict) for record_obj in result_obj.broker_order_record_list])
        state_store_obj.insert_vplan_broker_order_event_list([replace(event_obj, **identity_dict) for event_obj in result_obj.broker_order_event_list])
        state_store_obj.upsert_vplan_fill_list([replace(fill_obj, **identity_dict) for fill_obj in result_obj.broker_order_fill_list])
        state_store_obj.upsert_vplan_broker_ack_list([replace(ack_obj, **identity_dict) for ack_obj in result_obj.broker_order_ack_list])
        never_list = exception_obj.never_dispatched_request_list
        attempted_key_list = exception_obj.attempted_key_list
        transient_bool = exception_obj.transient_bool
    else:
        never_list = request_list if before_send_bool else []
        attempted_key_list = [] if before_send_bool else [request_obj.order_request_key_str for request_obj in request_list]
        transient_bool = is_transient_broker_error_bool(exception_obj)
    retry_bool = (not attempted_key_list and transient_bool
        and max(as_of_ts, datetime.now(UTC)) < moo_dispatch_deadline_ts(vplan_obj))
    payload_dict = {"severity_str": "warning" if retry_bool else "critical",
        "reason_code_str": "dispatch_retry_pending" if retry_bool else "opening_dispatch_incomplete",
        "error_type_str": getattr(exception_obj, "error_type_str", type(exception_obj).__name__),
        "attempted_request_key_list": attempted_key_list,
        "never_dispatched_request_key_list": [request_obj.order_request_key_str for request_obj in never_list],
        "observed_timestamp_str": as_of_ts.isoformat(),
        "dispatch_deadline_timestamp_str": moo_dispatch_deadline_ts(vplan_obj).isoformat()}
    with state_store_obj._connect() as connection_obj:
        connection_obj.execute("BEGIN IMMEDIATE")
        row_obj = connection_obj.execute("""SELECT v.status_str, d.snapshot_metadata_json_str
            FROM vplan v JOIN decision_plan d ON d.decision_plan_id_int=v.decision_plan_id_int
            WHERE v.vplan_id_int=?""", (vplan_obj.vplan_id_int,)).fetchone()
        expected_status_set = {"submitting"} if claim_owned_bool else {"ready"}
        if row_obj is None or row_obj["status_str"] not in expected_status_set:
            return "submission_claim_failed"
        if not claim_owned_bool and any(connection_obj.execute(
                f"SELECT 1 FROM {table_str} WHERE vplan_id_int=? LIMIT 1", (vplan_obj.vplan_id_int,)).fetchone()
                for table_str in ("vplan_broker_order", "vplan_broker_order_event", "vplan_broker_ack", "vplan_fill")):
            return "submission_claim_failed"
        if retry_bool and connection_obj.execute("SELECT 1 FROM vplan_broker_order WHERE vplan_id_int=? LIMIT 1",
                (vplan_obj.vplan_id_int,)).fetchone() is not None:
            raise ValueError("Cannot release a dispatch claim with recorded orders.")
        metadata_dict = json.loads(row_obj["snapshot_metadata_json_str"])
        metadata_dict["opening_dispatch_result_dict"] = payload_dict
        # No proof ledger or separate recovery phase: both failed claims and
        # unsent batches remain observable until this session's reconciliation.
        connection_obj.execute("UPDATE vplan SET status_str=?, updated_timestamp_str=? WHERE vplan_id_int=?",
            ("ready" if retry_bool else "submitted", as_of_ts.isoformat(), vplan_obj.vplan_id_int))
        connection_obj.execute("UPDATE decision_plan SET status_str=?, snapshot_metadata_json_str=?, updated_timestamp_str=? WHERE decision_plan_id_int=?",
            ("vplan_ready" if retry_bool else "submitted", json.dumps(metadata_dict, sort_keys=True),
             as_of_ts.isoformat(), vplan_obj.decision_plan_id_int))
    return payload_dict["reason_code_str"]
