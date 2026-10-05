"""Capsule-only recovery, manual-fill adoption and atomic settlement.

This is an intentional execution-policy extension: one same-session market sale
may complete an opening-auction exit. It does not change research decisions.
"""
from dataclasses import asdict, replace
import json
import math

from alpha.live import scheduler_utils
from alpha.live.execution_engine import build_broker_order_request_list_from_vplan
from alpha.live.models import BrokerOrderRequest, PodState
from alpha.live.mr_capsule_reconcile import classify_capsule_execution, TERMINAL_SHORTFALL_STATUS_SET


def ensure_capsule_recovery_schema(connection_obj):
    connection_obj.execute("""CREATE TABLE IF NOT EXISTS mr_capsule_execution_request (
        vplan_id_int INTEGER NOT NULL, order_request_key_str TEXT NOT NULL,
        asset_str TEXT NOT NULL, request_kind_str TEXT NOT NULL,
        request_json_str TEXT NOT NULL, claimed_timestamp_str TEXT NOT NULL,
        PRIMARY KEY(vplan_id_int, order_request_key_str))""")
    connection_obj.execute("""CREATE UNIQUE INDEX IF NOT EXISTS mr_capsule_one_recovery_per_asset
        ON mr_capsule_execution_request(vplan_id_int, asset_str)
        WHERE request_kind_str = 'recovery'""")


def load_supplemental_requests(state_store_obj, vplan_obj):
    with state_store_obj._connect() as connection_obj:
        row_list = connection_obj.execute("SELECT request_json_str FROM mr_capsule_execution_request WHERE vplan_id_int = ? ORDER BY rowid",
            (vplan_obj.vplan_id_int,)).fetchall()
    return [BrokerOrderRequest(**json.loads(row_obj["request_json_str"])) for row_obj in row_list]


def claim_capsule_request(state_store_obj, vplan_obj, request_obj, kind_str, as_of_ts):
    """Commit before network I/O. An uncertain attempt is never transmitted again."""
    with state_store_obj._connect() as connection_obj:
        connection_obj.execute("BEGIN IMMEDIATE")
        row_obj = connection_obj.execute("SELECT status_str, account_route_str FROM vplan WHERE vplan_id_int = ?",
            (vplan_obj.vplan_id_int,)).fetchone()
        if row_obj is None or row_obj["status_str"] not in {"submitted", "submitting"}:
            return False
        if request_obj.account_route_str != row_obj["account_route_str"] or kind_str not in {"recovery", "manual"}:
            raise ValueError("Invalid capsule recovery identity.")
        cursor_obj = connection_obj.execute("""INSERT OR IGNORE INTO mr_capsule_execution_request
            (vplan_id_int,order_request_key_str,asset_str,request_kind_str,request_json_str,claimed_timestamp_str)
            VALUES (?,?,?,?,?,?)""", (vplan_obj.vplan_id_int, request_obj.order_request_key_str,
                request_obj.asset_str, kind_str, json.dumps(asdict(request_obj), sort_keys=True, allow_nan=False), as_of_ts.isoformat()))
        return cursor_obj.rowcount == 1


def _manual_request_by_order_id(state_store_obj, vplan_obj):
    prefix_str = f"{vplan_obj.submission_key_str}:manual:"
    return {request_obj.order_request_key_str[len(prefix_str):]: request_obj
        for request_obj in load_supplemental_requests(state_store_obj, vplan_obj)
        if request_obj.order_request_key_str.startswith(prefix_str)}


def _persist_observations(state_store_obj, vplan_obj, record_list, event_list, fill_list, *, late_bool=False):
    supplemental_key_set = {request_obj.order_request_key_str for request_obj in load_supplemental_requests(state_store_obj, vplan_obj)}
    manual_request_dict = _manual_request_by_order_id(state_store_obj, vplan_obj)
    known_key_dict = {row_dict["broker_order_id_str"]: row_dict["order_request_key_str"] for row_dict in
        state_store_obj.get_broker_order_row_dict_list_for_vplan(vplan_obj.vplan_id_int)}
    # The broker does not know our manual-adoption key. Recover it by exact ID,
    # including a restart between committing adoption and persisting observations.
    key_by_id_dict = {record_obj.broker_order_id_str:
        manual_request_dict[record_obj.broker_order_id_str].order_request_key_str
        if record_obj.broker_order_id_str in manual_request_dict else
        (record_obj.order_request_key_str or known_key_dict.get(record_obj.broker_order_id_str))
        for record_obj in record_list}
    late_order_id_set = {order_id_str for order_id_str, key_str in {**known_key_dict, **key_by_id_dict}.items()
        if late_bool or key_str in supplemental_key_set}
    normalized_record_list = [replace(record_obj, decision_plan_id_int=vplan_obj.decision_plan_id_int,
        vplan_id_int=vplan_obj.vplan_id_int, submission_key_str=vplan_obj.submission_key_str,
        order_request_key_str=key_by_id_dict[record_obj.broker_order_id_str],
        raw_payload_dict={**record_obj.raw_payload_dict, **({"execution_phase_str": "late_execution"}
            if record_obj.broker_order_id_str in late_order_id_set else {})}) for record_obj in record_list]
    state_store_obj.upsert_vplan_broker_order_record_list(normalized_record_list)
    state_store_obj.insert_vplan_broker_order_event_list([replace(event_obj,
        decision_plan_id_int=vplan_obj.decision_plan_id_int, vplan_id_int=vplan_obj.vplan_id_int,
        order_request_key_str=key_by_id_dict.get(event_obj.broker_order_id_str, event_obj.order_request_key_str),
        submission_key_str=vplan_obj.submission_key_str) for event_obj in event_list])
    open_price_dict = state_store_obj.get_session_open_price_map_dict(
        account_route_str=vplan_obj.account_route_str,
        session_date_str=vplan_obj.target_execution_timestamp_ts.date().isoformat())
    fill_list = [replace(fill_obj,
        official_open_price_float=open_price_dict[fill_obj.asset_str].official_open_price_float,
        open_price_source_str=open_price_dict[fill_obj.asset_str].open_price_source_str)
        if fill_obj.asset_str in open_price_dict and fill_obj.broker_order_id_str not in late_order_id_set
        else fill_obj for fill_obj in fill_list]
    state_store_obj.upsert_vplan_fill_list([replace(fill_obj,
        decision_plan_id_int=vplan_obj.decision_plan_id_int, vplan_id_int=vplan_obj.vplan_id_int,
        raw_payload_dict={**fill_obj.raw_payload_dict, **({"execution_phase_str": "late_execution"}
            if fill_obj.broker_order_id_str in late_order_id_set else {})},
        official_open_price_float=None if fill_obj.broker_order_id_str in late_order_id_set else fill_obj.official_open_price_float,
        open_price_source_str="late_execution" if fill_obj.broker_order_id_str in late_order_id_set else fill_obj.open_price_source_str)
        for fill_obj in fill_list])


def _usable_snapshot(snapshot_obj, vplan_obj):
    return (snapshot_obj.account_route_str == vplan_obj.account_route_str
        and snapshot_obj.snapshot_timestamp_ts >= vplan_obj.target_execution_timestamp_ts
        and all(math.isfinite(float(value_float)) for value_float in
            [snapshot_obj.cash_float, snapshot_obj.net_liq_float, *snapshot_obj.position_amount_map.values()])
        and snapshot_obj.net_liq_float > 0
        and all(value_float >= 0 for value_float in snapshot_obj.position_amount_map.values()))


def _classify(state_store_obj, vplan_obj, snapshot_obj):
    return classify_capsule_execution(vplan_obj, snapshot_obj,
        state_store_obj.get_broker_order_row_dict_list_for_vplan(vplan_obj.vplan_id_int, include_evidence_bool=True),
        state_store_obj.get_fill_row_dict_list_for_vplan(vplan_obj.vplan_id_int,
            include_order_identity_bool=True, include_evidence_bool=True),
        supplemental_request_list=load_supplemental_requests(state_store_obj, vplan_obj))


def _refresh(state_store_obj, broker_adapter_obj, vplan_obj):
    record_list, event_list, fill_list = broker_adapter_obj.get_capsule_order_state_snapshot(
        account_route_str=vplan_obj.account_route_str, since_timestamp_ts=vplan_obj.submission_timestamp_ts,
        submission_key_str=vplan_obj.submission_key_str or f"vplan:{vplan_obj.decision_plan_id_int}",
        allowed_broker_order_id_set={row_dict["broker_order_id_str"] for row_dict in
            state_store_obj.get_broker_order_row_dict_list_for_vplan(vplan_obj.vplan_id_int)}
            | set(_manual_request_by_order_id(state_store_obj, vplan_obj)))
    _persist_observations(state_store_obj, vplan_obj, record_list, event_list, fill_list)
    # *** CRITICAL *** Holdings must be sampled AFTER observing fills; an earlier
    # snapshot could falsely imply a residual and cause a duplicate sale.
    snapshot_obj = broker_adapter_obj.get_capsule_account_snapshot(vplan_obj.account_route_str)
    if _usable_snapshot(snapshot_obj, vplan_obj):
        state_store_obj.upsert_broker_snapshot_cache(snapshot_obj)
    return snapshot_obj


def _adopt_manual_repairs(state_store_obj, broker_adapter_obj, vplan_obj, snapshot_obj, residual_list, as_of_ts):
    """Only adopt final, account-scoped orders that close a known original residual.

    Holdings alone are never sufficient. Unknown original/recovery orders remain
    unresolved even if a manual trade happens to reach the original target.
    """
    if snapshot_obj.open_order_id_list or not _usable_snapshot(snapshot_obj, vplan_obj):
        return
    order_list, event_list, fill_list = broker_adapter_obj.get_capsule_order_state_snapshot(
        account_route_str=vplan_obj.account_route_str, since_timestamp_ts=vplan_obj.target_execution_timestamp_ts,
        submission_key_str=None, allowed_broker_order_id_set=None)
    known_id_set = {row_dict["broker_order_id_str"] for row_dict in
        state_store_obj.get_broker_order_row_dict_list_for_vplan(vplan_obj.vplan_id_int)}
    request_by_asset_dict = {request_obj.asset_str: request_obj for request_obj in build_broker_order_request_list_from_vplan(vplan_obj)}
    residual_by_asset_dict = {row_dict["asset_str"]: row_dict["residual_amount_float"] for row_dict in residual_list}
    for order_obj in order_list:
        residual_float = residual_by_asset_dict.get(order_obj.asset_str, 0.0)
        if (order_obj.broker_order_id_str in known_id_set or order_obj.account_route_str != vplan_obj.account_route_str
            or order_obj.asset_str not in request_by_asset_dict):
            continue
        if order_obj.submitted_timestamp_ts < vplan_obj.target_execution_timestamp_ts or order_obj.unit_str != "shares":
            continue
        matching_fill_list = [fill_obj for fill_obj in fill_list if fill_obj.broker_order_id_str == order_obj.broker_order_id_str]
        filled_float = sum(fill_obj.fill_amount_float for fill_obj in matching_fill_list)
        if (not math.isfinite(filled_float) or not filled_float or filled_float * residual_float <= 0
            or not math.isfinite(order_obj.amount_float) or order_obj.amount_float * residual_float <= 0
            or not math.isfinite(order_obj.filled_amount_float)
            or abs(order_obj.amount_float) > abs(residual_float) + 1e-9
            or abs(filled_float) > abs(order_obj.amount_float) + 1e-9
            or abs(abs(filled_float) - abs(order_obj.filled_amount_float)) > 1e-9
            or any(fill_obj.account_route_str != vplan_obj.account_route_str or fill_obj.asset_str != order_obj.asset_str
                or fill_obj.fill_timestamp_ts < vplan_obj.target_execution_timestamp_ts
                or fill_obj.fill_timestamp_ts > snapshot_obj.snapshot_timestamp_ts
                or fill_obj.fill_amount_float * residual_float <= 0 for fill_obj in matching_fill_list)
            or order_obj.raw_payload_dict.get("open_order_observed_bool") is True
            or order_obj.raw_payload_dict.get("snapshot_source_str") == "open_order"
            or order_obj.raw_payload_dict.get("completed_quantity_verified_bool") is False):
            continue
        if abs(filled_float) < abs(order_obj.amount_float) - 1e-9 and (
            order_obj.status_str not in TERMINAL_SHORTFALL_STATUS_SET
            or order_obj.raw_payload_dict.get("snapshot_source_str") != "completed_order"):
            continue
        original_request_obj = request_by_asset_dict[order_obj.asset_str]
        manual_request_obj = replace(original_request_obj,
            order_request_key_str=f"{original_request_obj.submission_key_str}:manual:{order_obj.broker_order_id_str}",
            broker_order_type_str=order_obj.broker_order_type_str, amount_float=order_obj.amount_float)
        if claim_capsule_request(state_store_obj, vplan_obj, manual_request_obj, "manual", as_of_ts):
            adopted_order_obj = replace(order_obj, order_request_key_str=manual_request_obj.order_request_key_str,
                raw_payload_dict={**order_obj.raw_payload_dict, "manual_repair_bool": True,
                    "original_order_request_key_str": order_obj.order_request_key_str})
            _persist_observations(state_store_obj, vplan_obj, [adopted_order_obj], [], matching_fill_list, late_bool=True)
            residual_by_asset_dict[order_obj.asset_str] -= filled_float
            known_id_set.add(order_obj.broker_order_id_str)


def _same_session_recovery_allowed(release_obj, vplan_obj, as_of_ts, snapshot_obj):
    local_ts = scheduler_utils.to_market_timestamp_ts(as_of_ts, release_obj.session_calendar_id_str)
    target_ts = scheduler_utils.to_market_timestamp_ts(vplan_obj.target_execution_timestamp_ts, release_obj.session_calendar_id_str)
    close_ts = scheduler_utils.get_session_close_timestamp_ts(target_ts.date(), release_obj.session_calendar_id_str)
    observed_ts = scheduler_utils.to_market_timestamp_ts(snapshot_obj.snapshot_timestamp_ts, release_obj.session_calendar_id_str)
    return (_usable_snapshot(snapshot_obj, vplan_obj) and target_ts.date() == local_ts.date() == observed_ts.date()
        and target_ts < local_ts < close_ts and target_ts < observed_ts < close_ts)


def record_capsule_result(state_store_obj, release_obj, vplan_obj, decision_plan_obj,
        snapshot_obj, reconciliation_obj, outcome_str, residual_list, as_of_ts):
    from alpha.live.mr_capsule_notifications import enqueue_execution_alert
    payload_dict = {"outcome_str": outcome_str, "residual_row_dict_list": residual_list,
        "cash_float": snapshot_obj.cash_float, "net_liq_float": snapshot_obj.net_liq_float,
        "available_funds_float": snapshot_obj.available_funds_float,
        "excess_liquidity_float": snapshot_obj.excess_liquidity_float}
    # Optional margin diagnostics must not strand an otherwise resolved cycle.
    # Invalid core observations are reported, but never saved as trusted state.
    for field_str in ("cash_float", "net_liq_float", "available_funds_float", "excess_liquidity_float"):
        value_float = payload_dict[field_str]
        if value_float is not None and not math.isfinite(float(value_float)):
            payload_dict[field_str] = None
    payload_dict["broker_snapshot_valid_bool"] = _usable_snapshot(snapshot_obj, vplan_obj)
    with state_store_obj._connect() as connection_obj:
        connection_obj.execute("BEGIN IMMEDIATE")
        row_obj = connection_obj.execute("SELECT status_str FROM vplan WHERE vplan_id_int = ?",
            (vplan_obj.vplan_id_int,)).fetchone()
        if row_obj is None or row_obj["status_str"] == "completed":
            return
        if _usable_snapshot(snapshot_obj, vplan_obj):
            state_store_obj.insert_vplan_reconciliation_snapshot(vplan_obj.pod_id_str,
                vplan_obj.decision_plan_id_int, vplan_obj.vplan_id_int, "post_execution", reconciliation_obj,
                connection_obj=connection_obj)
            state_store_obj.upsert_pod_state(PodState(pod_id_str=vplan_obj.pod_id_str,
                user_id_str=release_obj.user_id_str, account_route_str=vplan_obj.account_route_str,
                position_amount_map=snapshot_obj.position_amount_map, cash_float=snapshot_obj.cash_float,
                total_value_float=snapshot_obj.net_liq_float, strategy_state_dict=dict(decision_plan_obj.strategy_state_dict),
                updated_timestamp_ts=as_of_ts), snapshot_stage_str="post_execution",
                snapshot_source_str="virtual_broker" if release_obj.mode_str == "incubation" else "broker",
                connection_obj=connection_obj)
        metadata_dict = dict(decision_plan_obj.snapshot_metadata_dict)
        metadata_dict["mr_capsule_execution_result_dict"] = payload_dict
        connection_obj.execute("UPDATE decision_plan SET snapshot_metadata_json_str = ? WHERE decision_plan_id_int = ?",
            (json.dumps(metadata_dict, sort_keys=True, allow_nan=False), vplan_obj.decision_plan_id_int))
        if residual_list or not reconciliation_obj.passed_bool:
            enqueue_execution_alert(connection_obj, vplan_id_int=vplan_obj.vplan_id_int,
                alert_kind_str="accepted_residual" if reconciliation_obj.passed_bool else "unresolved_execution",
                pod_id_str=vplan_obj.pod_id_str, account_route_str=vplan_obj.account_route_str,
                mode_str=release_obj.mode_str, payload_dict=payload_dict, created_timestamp_ts=as_of_ts)
        late_fill_row_list = connection_obj.execute("SELECT asset_str, fill_amount_float, fill_price_float, fill_timestamp_str FROM vplan_fill WHERE vplan_id_int = ? AND open_price_source_str = 'late_execution'",
            (vplan_obj.vplan_id_int,)).fetchall()
        if reconciliation_obj.passed_bool and late_fill_row_list:
            enqueue_execution_alert(connection_obj, vplan_id_int=vplan_obj.vplan_id_int,
                alert_kind_str="late_execution", pod_id_str=vplan_obj.pod_id_str,
                account_route_str=vplan_obj.account_route_str, mode_str=release_obj.mode_str,
                payload_dict={**payload_dict, "late_fill_row_dict_list": [dict(row_obj) for row_obj in late_fill_row_list]},
                created_timestamp_ts=as_of_ts)
        if reconciliation_obj.passed_bool:
            state_store_obj.complete_mr_capsule_cycle(vplan_obj.decision_plan_id_int, vplan_obj.vplan_id_int,
                connection_obj=connection_obj)


def reconcile_capsule_cycle(state_store_obj, broker_adapter_obj, release_obj, vplan_obj, decision_plan_obj, as_of_ts):
    if release_obj.mode_str == "live":
        raise ValueError("MR capsule LIVE is locked pending owner authorization.")
    # *** CRITICAL *** Open references are diagnostics anchored to the original
    # target session. They never resize the frozen intent or label a late MKT fill.
    asset_list = sorted({request_obj.asset_str for request_obj in build_broker_order_request_list_from_vplan(vplan_obj)})
    if asset_list:
        try:
            open_price_list = broker_adapter_obj.get_session_open_price_list(
                account_route_str=vplan_obj.account_route_str, asset_str_list=asset_list,
                session_open_timestamp_ts=vplan_obj.target_execution_timestamp_ts,
                session_calendar_id_str=release_obj.session_calendar_id_str)
            target_date_str = scheduler_utils.to_market_timestamp_ts(
                vplan_obj.target_execution_timestamp_ts, release_obj.session_calendar_id_str).date().isoformat()
            valid_open_list = []
            for price_obj in open_price_list:
                if price_obj.account_route_str != vplan_obj.account_route_str or price_obj.session_date_str != target_date_str:
                    raise ValueError("Open reference account/session mismatch.")
                if price_obj.official_open_price_float is None:
                    continue
                if not math.isfinite(price_obj.official_open_price_float) or price_obj.official_open_price_float <= 0:
                    raise ValueError("Invalid official open reference.")
                # ticker.open is today's field even when a caller requests an older
                # session. Never overwrite a saved target-session reference with it.
                if price_obj.open_price_source_str == "ibkr.tick_open" and scheduler_utils.to_market_timestamp_ts(
                        price_obj.snapshot_timestamp_ts, release_obj.session_calendar_id_str).date().isoformat() != target_date_str:
                    raise ValueError("Current tick open cannot label a historical capsule session.")
                valid_open_list.append(price_obj)
            state_store_obj.upsert_session_open_price_list(valid_open_list)
        except Exception as exception_obj:
            decision_plan_obj = replace(decision_plan_obj, snapshot_metadata_dict={
                **decision_plan_obj.snapshot_metadata_dict,
                "mr_capsule_open_reference_error_str": str(exception_obj) or type(exception_obj).__name__})
    snapshot_obj = _refresh(state_store_obj, broker_adapter_obj, vplan_obj)
    reconciliation_obj, outcome_str, residual_list = _classify(state_store_obj, vplan_obj, snapshot_obj)
    if outcome_str == "unexplained_positions":
        _adopt_manual_repairs(state_store_obj, broker_adapter_obj, vplan_obj, snapshot_obj, residual_list, as_of_ts)
        snapshot_obj = _refresh(state_store_obj, broker_adapter_obj, vplan_obj)
        reconciliation_obj, outcome_str, residual_list = _classify(state_store_obj, vplan_obj, snapshot_obj)
    if outcome_str == "accepted_residual" and _same_session_recovery_allowed(release_obj, vplan_obj, as_of_ts, snapshot_obj):
        request_by_asset_dict = {request_obj.asset_str: request_obj for request_obj in build_broker_order_request_list_from_vplan(vplan_obj)}
        for residual_dict in residual_list:
            asset_str = residual_dict["asset_str"]
            if residual_dict["residual_amount_float"] >= 0 or (
                asset_str != "BIL" and abs(vplan_obj.target_share_map.get(asset_str, 0)) > 1e-9):
                continue
            # Refresh before EACH new sale: another fill/manual trade may have
            # changed holdings while an earlier network request was in flight.
            snapshot_obj = _refresh(state_store_obj, broker_adapter_obj, vplan_obj)
            reconciliation_obj, outcome_str, current_residual_list = _classify(state_store_obj, vplan_obj, snapshot_obj)
            if outcome_str != "accepted_residual" or not _same_session_recovery_allowed(release_obj, vplan_obj, as_of_ts, snapshot_obj):
                break
            residual_float = next((row_dict["residual_amount_float"] for row_dict in current_residual_list
                if row_dict["asset_str"] == asset_str), 0.0)
            # Q_sell = min(original unfilled sale, max(actual shares - frozen target, 0)).
            remainder_float = min(max(-residual_float, 0.0), max(
                snapshot_obj.position_amount_map.get(asset_str, 0.0) - vplan_obj.target_share_map.get(asset_str, 0.0), 0.0))
            if remainder_float <= 1e-9:
                continue
            original_request_obj = request_by_asset_dict[asset_str]
            recovery_request_obj = replace(original_request_obj,
                order_request_key_str=f"{original_request_obj.submission_key_str}:late:{asset_str}",
                amount_float=-remainder_float, broker_order_type_str="MKT",
                execution_deadline_timestamp_str=scheduler_utils.get_session_close_timestamp_ts(
                    scheduler_utils.to_market_timestamp_ts(vplan_obj.target_execution_timestamp_ts,
                        release_obj.session_calendar_id_str).date(), release_obj.session_calendar_id_str).isoformat())
            if not claim_capsule_request(state_store_obj, vplan_obj, recovery_request_obj, "recovery", as_of_ts):
                continue
            try:
                result_obj = broker_adapter_obj.submit_order_request_list(account_route_str=vplan_obj.account_route_str,
                    broker_order_request_list=[recovery_request_obj], submitted_timestamp_ts=snapshot_obj.snapshot_timestamp_ts)
                _persist_observations(state_store_obj, vplan_obj, result_obj.broker_order_record_list,
                    result_obj.broker_order_event_list, result_obj.broker_order_fill_list, late_bool=True)
            except Exception:
                # Claim remains durable. The next observation must prove what
                # happened; a timeout is not permission to send again.
                pass
        snapshot_obj = _refresh(state_store_obj, broker_adapter_obj, vplan_obj)
        reconciliation_obj, outcome_str, residual_list = _classify(state_store_obj, vplan_obj, snapshot_obj)
    record_capsule_result(state_store_obj, release_obj, vplan_obj, decision_plan_obj,
        snapshot_obj, reconciliation_obj, outcome_str, residual_list, as_of_ts)
    return reconciliation_obj, outcome_str, residual_list, snapshot_obj
