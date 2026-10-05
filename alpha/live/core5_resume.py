"""Reviewed CORE5 signal-memory resumption. This service never sends orders."""
from dataclasses import asdict, replace
from datetime import datetime, timedelta, timezone
import hashlib
import json

from alpha.live import core5_adapter, scheduler_utils
from alpha.live.core5_recovery import refresh_core5_cycle_evidence
from alpha.live.models import PodState


def _utc_now_ts():
    return datetime.now(timezone.utc)


def _json_default_obj(value_obj):
    if isinstance(value_obj, datetime):
        return value_obj.isoformat()
    if isinstance(value_obj, set):
        return sorted(value_obj)
    raise TypeError(f"Unsupported resume evidence type: {type(value_obj).__name__}")


def _canonical_json_str(payload_obj):
    return json.dumps(payload_obj, sort_keys=True, separators=(",", ":"),
        allow_nan=False, default=_json_default_obj)


def _hash_str(payload_obj):
    return hashlib.sha256(_canonical_json_str(payload_obj).encode("utf-8")).hexdigest()


def _capture_scope_dict(connection_obj, release_obj):
    """Capture immutable intent and current state for the final transaction CAS."""
    account_tuple = (release_obj.account_route_str,)
    pod_tuple = (release_obj.pod_id_str,)
    result_dict = {}
    for key_str, query_str, parameter_tuple in (
        ("release_row_list", "SELECT * FROM live_release WHERE account_route_str=? ORDER BY release_id_str", account_tuple),
        ("pod_state_row_list", "SELECT * FROM pod_state WHERE pod_id_str=?", pod_tuple),
        ("decision_row_list", "SELECT * FROM decision_plan WHERE account_route_str=? ORDER BY decision_plan_id_int", account_tuple),
        ("vplan_row_list", "SELECT * FROM vplan WHERE account_route_str=? ORDER BY vplan_id_int", account_tuple),
        ("history_row_list", "SELECT * FROM pod_state_history WHERE pod_id_str=? AND snapshot_stage_str='eod' ORDER BY pod_state_history_id_int", pod_tuple),
    ):
        result_dict[key_str] = [dict(row_obj) for row_obj in connection_obj.execute(query_str, parameter_tuple)]
    return result_dict


def _cycle_evidence_hash_str(connection_obj, vplan_id_int):
    evidence_dict = {}
    for table_str in ("vplan_broker_order", "vplan_broker_order_event", "vplan_broker_ack",
            "vplan_fill", "vplan_execution_resolution", "mr_capsule_execution_request"):
        evidence_dict[table_str] = sorted([dict(row_obj) for row_obj in connection_obj.execute(
            f"SELECT * FROM {table_str} WHERE vplan_id_int=?", (vplan_id_int,))], key=_canonical_json_str)
    return _hash_str(evidence_dict)


def _trusted_eod_state_obj(scope_dict, release_obj, signal_date_ts, as_of_ts):
    state_row_list = scope_dict["pod_state_row_list"]
    if len(state_row_list) != 1:
        raise ValueError("CORE5 resume requires saved account state.")
    current_row_dict = state_row_list[0]
    candidate_list = []
    for row_dict in [*scope_dict["history_row_list"], current_row_dict]:
        timestamp_ts = datetime.fromisoformat(row_dict["updated_timestamp_str"])
        if timestamp_ts.tzinfo is None:
            continue
        market_timestamp_ts = scheduler_utils.to_market_timestamp_ts(timestamp_ts, "XNYS")
        if (row_dict["snapshot_stage_str"] == "eod" and row_dict["snapshot_source_str"] == "broker"
                and row_dict["account_route_str"] == release_obj.account_route_str
                and row_dict["user_id_str"] == release_obj.user_id_str
                and market_timestamp_ts.date() == signal_date_ts.date()
                and scheduler_utils.get_session_close_timestamp_ts(signal_date_ts, "XNYS") <= timestamp_ts <= as_of_ts):
            candidate_list.append((timestamp_ts, row_dict))
    if not candidate_list:
        raise ValueError("CORE5 resume requires trusted broker EOD evidence at the exact signal session.")
    _, eod_row_dict = max(candidate_list, key=lambda candidate_tuple: candidate_tuple[0])
    if (current_row_dict["account_route_str"], current_row_dict["user_id_str"]) != (
            release_obj.account_route_str, release_obj.user_id_str):
        raise ValueError("CORE5 resume saved account identity changed.")
    return PodState(pod_id_str=release_obj.pod_id_str, user_id_str=release_obj.user_id_str,
        account_route_str=release_obj.account_route_str,
        position_amount_map=json.loads(eod_row_dict["position_json_str"]),
        cash_float=float(eod_row_dict["cash_float"]), total_value_float=float(eod_row_dict["total_value_float"]),
        strategy_state_dict=json.loads(current_row_dict["strategy_state_json_str"]),
        updated_timestamp_ts=datetime.fromisoformat(eod_row_dict["updated_timestamp_str"]),
        snapshot_stage_str="eod", snapshot_source_str="broker")


def _require_fresh_account(snapshot_obj, release_obj, expected_position_dict, as_of_ts, cutoff_ts):
    if (snapshot_obj.account_route_str != release_obj.account_route_str
            or snapshot_obj.snapshot_timestamp_ts.tzinfo is None
            or snapshot_obj.snapshot_timestamp_ts < as_of_ts
            or snapshot_obj.snapshot_timestamp_ts >= cutoff_ts
            or snapshot_obj.open_order_id_list):
        raise ValueError("CORE5 resume requires fresh matching account evidence with no open orders before cutoff.")
    core5_adapter.require_core5_position_match(expected_position_dict, snapshot_obj.position_amount_map)


def _prepare_core5_resume(state_store_obj, broker_adapter_obj, release_obj, as_of_ts):
    if release_obj.strategy_import_str != core5_adapter.CORE5_STRATEGY_IMPORT_STR or not release_obj.enabled_bool:
        raise ValueError("Reviewed resume requires an enabled CORE5 release.")
    core5_adapter.validate_core5_release(release_obj)
    if as_of_ts.tzinfo is None:
        raise ValueError("CORE5 resume time must have a timezone.")
    with state_store_obj._connect() as connection_obj:
        scope_dict = _capture_scope_dict(connection_obj, release_obj)
    matching_release_list = [row_dict for row_dict in scope_dict["release_row_list"]
        if row_dict["release_id_str"] == release_obj.release_id_str]
    if len(matching_release_list) != 1 or state_store_obj._row_to_release(matching_release_list[0]) != release_obj:
        raise ValueError("CORE5 resume release differs from the saved release.")
    if any(row_dict["enabled_bool"] and row_dict["release_id_str"] != release_obj.release_id_str
            for row_dict in scope_dict["release_row_list"]):
        raise ValueError("CORE5 resume requires its dedicated account route.")
    signal_date_ts = scheduler_utils.get_latest_completed_session_label_ts(as_of_ts, "XNYS")
    if signal_date_ts is None:
        raise ValueError("CORE5 resume has no completed signal session.")
    cutoff_ts = scheduler_utils.build_target_execution_timestamp_ts(signal_date_ts, release_obj) - timedelta(minutes=2)
    if max(as_of_ts, _utc_now_ts()) >= cutoff_ts:
        raise ValueError("CORE5 resume missed the next-open submission cutoff.")
    eod_state_obj = _trusted_eod_state_obj(scope_dict, release_obj, signal_date_ts, as_of_ts)
    decision_obj = core5_adapter.build_core5_resync_decision_plan(release_obj, as_of_ts, eod_state_obj)
    prior_cycle_list = []
    execution_hash_dict = {}
    for decision_row_dict in scope_dict["decision_row_list"]:
        if decision_row_dict["pod_id_str"] != release_obj.pod_id_str:
            if decision_row_dict["status_str"] in {"planned", "vplan_ready", "submitted"}:
                raise ValueError("Another pod has an unresolved decision on the CORE5 account.")
            continue
        if decision_row_dict["status_str"] == "superseded":
            continue
        if decision_row_dict["status_str"] == "completed":
            if datetime.fromisoformat(decision_row_dict["signal_timestamp_str"]) >= decision_obj.signal_timestamp_ts:
                raise ValueError("CORE5 already completed this signal session.")
            continue
        if datetime.fromisoformat(decision_row_dict["signal_timestamp_str"]) > decision_obj.signal_timestamp_ts:
            raise ValueError("CORE5 resume cannot supersede a future decision.")
        if json.loads(decision_row_dict["snapshot_metadata_json_str"]).get("sizing_contract_str") != core5_adapter.CORE5_CONTRACT_STR:
            raise ValueError("CORE5 resume cannot supersede another strategy's decision.")
        vplan_row_list = [row_dict for row_dict in scope_dict["vplan_row_list"]
            if row_dict["decision_plan_id_int"] == decision_row_dict["decision_plan_id_int"]]
        evidence_dict = {"terminal_bool": True, "source_str": "never_claimed_decision"}
        if vplan_row_list:
            vplan_row_dict = vplan_row_list[0]
            vplan_obj = state_store_obj.get_vplan_by_id(vplan_row_dict["vplan_id_int"])
            with state_store_obj._connect() as connection_obj:
                connection_obj.execute("BEGIN")
                observed_bool = any(connection_obj.execute(f"SELECT 1 FROM {table_str} WHERE vplan_id_int=? LIMIT 1",
                    (vplan_obj.vplan_id_int,)).fetchone() is not None
                    for table_str in ("vplan_broker_order", "vplan_broker_order_event", "vplan_broker_ack", "vplan_fill"))
                execution_hash_dict[vplan_obj.vplan_id_int] = _cycle_evidence_hash_str(connection_obj, vplan_obj.vplan_id_int)
            unclaimed_bool = (vplan_obj.status_str == "ready" or
                (vplan_obj.status_str in {"blocked", "expired"} and json.loads(
                    decision_row_dict["snapshot_metadata_json_str"]).get("core5_unclaimed_terminal_bool") is True))
            if unclaimed_bool and not observed_bool:
                evidence_dict = {"terminal_bool": True, "source_str": "never_claimed_vplan"}
            else:
                old_release_obj = state_store_obj.get_release_by_id(vplan_obj.release_id_str)
                refreshed_dict = refresh_core5_cycle_evidence(state_store_obj, broker_adapter_obj,
                    old_release_obj, vplan_obj, as_of_ts)
                if not refreshed_dict["terminal_bool"]:
                    raise ValueError("CORE5 prior execution cycle remains uncertain or nonterminal.")
                core5_adapter.require_core5_position_match(eod_state_obj.position_amount_map,
                    refreshed_dict["expected_position_map_dict"])
                evidence_dict = {key_str: value_obj for key_str, value_obj in refreshed_dict.items()
                    if key_str != "broker_snapshot_obj"}
                with state_store_obj._connect() as connection_obj:
                    execution_hash_dict[vplan_obj.vplan_id_int] = _cycle_evidence_hash_str(connection_obj, vplan_obj.vplan_id_int)
        elif decision_row_dict["status_str"] not in {"planned", "blocked", "expired"}:
            raise ValueError("CORE5 prior decision has no verifiable execution plan.")
        prior_cycle_list.append({"decision_plan_id_int": decision_row_dict["decision_plan_id_int"],
            "decision_status_str": decision_row_dict["status_str"],
            "vplan_id_int": vplan_row_list[0]["vplan_id_int"] if vplan_row_list else None,
            "vplan_status_str": vplan_row_list[0]["status_str"] if vplan_row_list else None,
            "evidence_dict": evidence_dict})
    snapshot_obj = broker_adapter_obj.get_core5_account_snapshot(release_obj.account_route_str)
    _require_fresh_account(snapshot_obj, release_obj, eod_state_obj.position_amount_map, as_of_ts, cutoff_ts)
    if _utc_now_ts() >= cutoff_ts:
        raise ValueError("CORE5 resume evidence completed after the next-open cutoff.")
    with state_store_obj._connect() as connection_obj:
        if (_capture_scope_dict(connection_obj, release_obj) != scope_dict
                or any(_cycle_evidence_hash_str(connection_obj, vplan_id_int) != evidence_hash_str
                    for vplan_id_int, evidence_hash_str in execution_hash_dict.items())):
            raise ValueError("CORE5 saved state or execution cycle changed during resume review.")
    review_dict = {"release_id_str": release_obj.release_id_str, "pod_id_str": release_obj.pod_id_str,
        "account_route_str": release_obj.account_route_str, "state_fingerprint_str": _hash_str(scope_dict),
        "decision_plan_dict": json.loads(_canonical_json_str(asdict(decision_obj))),
        "prior_cycle_list": prior_cycle_list, "submission_cutoff_str": cutoff_ts.isoformat()}
    return {**review_dict, "review_hash_str": _hash_str(review_dict),
        "broker_evidence_dict": json.loads(_canonical_json_str(asdict(snapshot_obj)))}, decision_obj, scope_dict, execution_hash_dict


def preview_core5_resume(state_store_obj, broker_adapter_obj, release_obj, as_of_ts):
    """Refresh evidence and return the exact review hash; send no broker orders."""
    return _prepare_core5_resume(state_store_obj, broker_adapter_obj, release_obj, as_of_ts)[0]


def apply_core5_resume(state_store_obj, broker_adapter_obj, release_obj, as_of_ts,
        *, review_hash_str, reason_str, operator_str):
    """Atomically replace proved terminal intent; commit strategy memory only on completion."""
    if not isinstance(reason_str, str) or not reason_str.strip() or not isinstance(operator_str, str) or not operator_str.strip():
        raise ValueError("CORE5 resume requires a nonempty operator and reason.")
    preview_dict, decision_obj, scope_dict, execution_hash_dict = _prepare_core5_resume(
        state_store_obj, broker_adapter_obj, release_obj, as_of_ts)
    if preview_dict["review_hash_str"] != review_hash_str:
        raise ValueError("CORE5 resume review hash changed; inspect a new preview before applying.")
    audit_dict = {"review_hash_str": review_hash_str, "reason_str": reason_str.strip(),
        "operator_str": operator_str.strip(), "applied_timestamp_str": _utc_now_ts().isoformat(),
        "prior_cycle_list": preview_dict["prior_cycle_list"],
        "broker_evidence_dict": preview_dict["broker_evidence_dict"]}
    decision_obj = replace(decision_obj, snapshot_metadata_dict={**decision_obj.snapshot_metadata_dict,
        "core5_resume_audit_dict": audit_dict})
    with state_store_obj._connect() as connection_obj:
        connection_obj.execute("BEGIN IMMEDIATE")
        if (_capture_scope_dict(connection_obj, release_obj) != scope_dict
                or any(_cycle_evidence_hash_str(connection_obj, vplan_id_int) != evidence_hash_str
                    for vplan_id_int, evidence_hash_str in execution_hash_dict.items())):
            raise ValueError("CORE5 resume lost the compare-and-set race; inspect a new preview.")
        if _utc_now_ts() >= datetime.fromisoformat(preview_dict["submission_cutoff_str"]):
            raise ValueError("CORE5 resume reached the next-open cutoff before persistence.")
        revision_int = 1 + int(connection_obj.execute("SELECT COALESCE(MAX(intent_revision_int), 0) FROM decision_plan "
            "WHERE pod_id_str=? AND signal_timestamp_str=? AND execution_policy_str=?",
            (release_obj.pod_id_str, decision_obj.signal_timestamp_ts.isoformat(), decision_obj.execution_policy_str)).fetchone()[0])
        for cycle_dict in preview_dict["prior_cycle_list"]:
            old_id_int = cycle_dict["decision_plan_id_int"]
            old_row_dict = next(row_dict for row_dict in scope_dict["decision_row_list"] if row_dict["decision_plan_id_int"] == old_id_int)
            metadata_dict = json.loads(old_row_dict["snapshot_metadata_json_str"])
            metadata_dict["core5_superseded_audit_dict"] = audit_dict
            connection_obj.execute("UPDATE decision_plan SET status_str='superseded', snapshot_metadata_json_str=?, updated_timestamp_str=? WHERE decision_plan_id_int=?",
                (_canonical_json_str(metadata_dict), audit_dict["applied_timestamp_str"], old_id_int))
            if cycle_dict["vplan_id_int"] is not None:
                connection_obj.execute("UPDATE vplan SET status_str='superseded', updated_timestamp_str=? WHERE vplan_id_int=?",
                    (audit_dict["applied_timestamp_str"], cycle_dict["vplan_id_int"]))
        inserted_obj = state_store_obj.insert_decision_plan(decision_obj, connection_obj=connection_obj,
            intent_revision_int=revision_int)
    return {**preview_dict, "applied_bool": True, "decision_plan_id_int": inserted_obj.decision_plan_id_int,
        "intent_revision_int": revision_int, "core5_resume_audit_dict": audit_dict}
