"""One opening cross-cancel completion; no retry and no changes to strategy targets."""
from dataclasses import asdict, replace
from datetime import UTC, datetime, timedelta
import json
import math

from alpha.live import scheduler_utils
from alpha.live.models import BrokerOrderRequest, SubmitBatchResult

TABLE_NAME_STR = "auto_completion_attempt"
CROSS_MESSAGE_STR = "order would cross related resting order"
TERMINAL_STATUS_SET = {"Filled", "Cancelled", "ApiCancelled", "Rejected", "Expired", "Inactive"}


def _utc_now_ts():
    return datetime.now(UTC)


def timing_dict(vplan_obj, calendar_id_str):
    session_date_obj = scheduler_utils.to_market_timestamp_ts(
        vplan_obj.target_execution_timestamp_ts, calendar_id_str).date()
    open_ts = scheduler_utils.get_session_open_timestamp_ts(session_date_obj, calendar_id_str)
    close_ts = scheduler_utils.get_session_close_timestamp_ts(session_date_obj, calendar_id_str)
    return {"open_ts": open_ts, "start_ts": open_ts + timedelta(minutes=5),
        "send_cutoff_ts": close_ts - timedelta(minutes=10), "close_ts": close_ts}


def load_attempt_row_list(connection_obj, vplan_id_int):
    """Read-only, including against a pre-feature database."""
    if not connection_obj.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
            (TABLE_NAME_STR,)).fetchone():
        return []
    return [dict(row_obj) for row_obj in connection_obj.execute(
        "SELECT * FROM auto_completion_attempt WHERE vplan_id_int=? ORDER BY asset_str", (vplan_id_int,))]


def cross_cancel_leg_dict(vplan_obj, order_row_list, event_row_list):
    """Identify the cancelled original MOO, never a replacement or a generic rejection."""
    if vplan_obj.execution_policy_str not in {"next_open_moo", "next_month_first_open"}:
        return {}
    original_order_dict = {str(row_dict["broker_order_id_str"]): row_dict for row_dict in order_row_list
        if row_dict.get("broker_order_type_str") == "MOO"}
    leg_dict = {}
    for event_dict in event_row_list:
        message_str = str(event_dict.get("message_str") or "")
        if CROSS_MESSAGE_STR not in message_str.casefold() or event_dict.get("status_str") not in {"Cancelled", "ApiCancelled"}:
            continue
        order_dict = original_order_dict.get(str(event_dict.get("broker_order_id_str")))
        if order_dict is None and event_dict.get("order_request_key_str"):
            # IBKR can replace its local API ID with a permanent ID on a later refresh.
            matching_list = [row_dict for row_dict in original_order_dict.values()
                if row_dict.get("order_request_key_str") == event_dict["order_request_key_str"]
                and row_dict.get("asset_str") == event_dict.get("asset_str")
                and row_dict.get("account_route_str") == event_dict.get("account_route_str")]
            if len(matching_list) == 1:
                order_dict = matching_list[0]
        if order_dict is None:
            continue
        try:
            event_ts = datetime.fromisoformat(str(event_dict.get("event_timestamp_str")))
            if event_ts.tzinfo is None or event_ts >= vplan_obj.target_execution_timestamp_ts:
                continue
            asset_str = str(order_dict["asset_str"])
            amount_float = float(order_dict["amount_float"])
            target_float = float(vplan_obj.target_share_map[asset_str])
        except (ValueError, TypeError, KeyError):
            continue
        if not math.isfinite(amount_float) or not amount_float or not math.isfinite(target_float):
            continue
        leg_dict[asset_str] = {"original_broker_order_id_str": str(order_dict["broker_order_id_str"]),
            "original_order_request_key_str": order_dict.get("order_request_key_str"),
            "original_message_str": message_str, "cancelled_qty_float": abs(amount_float),
            "direction_float": 1.0 if amount_float > 0 else -1.0, "target_float": target_float,
            "filled_quantity_float": max(abs(float(order_dict.get("filled_amount_float") or 0)),
                abs(float(event_dict.get("filled_amount_float") or 0)))}
    return leg_dict


def load_leg_dict(state_store_obj, vplan_obj):
    return cross_cancel_leg_dict(vplan_obj,
        state_store_obj.get_broker_order_row_dict_list_for_vplan(vplan_obj.vplan_id_int),
        state_store_obj.get_broker_order_event_row_dict_list_for_vplan(vplan_obj.vplan_id_int))


def active_auto_completion_bool(state_store_obj, vplan_obj, as_of_ts):
    with state_store_obj._connect() as connection_obj:
        attempt_list = load_attempt_row_list(connection_obj, vplan_obj.vplan_id_int)
    if any(not row_dict["ended_timestamp_str"] for row_dict in attempt_list):
        return True
    leg_map_dict = load_leg_dict(state_store_obj, vplan_obj)
    if not leg_map_dict:
        return False
    release_obj = state_store_obj.get_release_by_id(vplan_obj.release_id_str)
    if as_of_ts < timing_dict(vplan_obj, release_obj.session_calendar_id_str)["send_cutoff_ts"]:
        marked_set = {row_dict["asset_str"] for row_dict in attempt_list}
        latest_plan_obj = state_store_obj.get_latest_decision_plan_for_pod(vplan_obj.pod_id_str)
        return (latest_plan_obj is not None and latest_plan_obj.decision_plan_id_int == vplan_obj.decision_plan_id_int
            and any(asset_str not in marked_set for asset_str in leg_map_dict))
    return False


def parked_recheck_due_bool(state_store_obj, vplan_obj, release_obj, as_of_ts):
    """One standard-route parked-position observation per exchange session after close + 10m."""
    session_date_obj = scheduler_utils.to_market_timestamp_ts(as_of_ts, release_obj.session_calendar_id_str).date()
    try:
        close_ts = scheduler_utils.get_session_close_timestamp_ts(session_date_obj, release_obj.session_calendar_id_str)
    except ValueError:
        return False
    due_ts = close_ts + timedelta(minutes=10)
    if as_of_ts < due_ts:
        return False
    with state_store_obj._connect() as connection_obj:
        return not connection_obj.execute("""SELECT 1 FROM vplan_reconciliation_snapshot
            WHERE vplan_id_int=? AND stage_str='parked_eod_recheck'
            AND julianday(created_timestamp_str)>=julianday(?) LIMIT 1""",
            (vplan_obj.vplan_id_int, due_ts.isoformat())).fetchone()


def record_parked_recheck(state_store_obj, vplan_obj, as_of_ts):
    with state_store_obj._connect() as connection_obj:
        connection_obj.execute("""INSERT INTO vplan_reconciliation_snapshot
            (pod_id_str,decision_plan_id_int,vplan_id_int,stage_str,status_str,mismatch_json_str,
             model_position_json_str,broker_position_json_str,model_cash_float,broker_cash_float,created_timestamp_str)
            SELECT pod_id_str,decision_plan_id_int,vplan_id_int,'parked_eod_recheck',status_str,mismatch_json_str,
                model_position_json_str,broker_position_json_str,model_cash_float,broker_cash_float,?
            FROM vplan_reconciliation_snapshot WHERE vplan_id_int=? AND stage_str='post_execution'
            ORDER BY vplan_reconciliation_snapshot_id_int DESC LIMIT 1""",
            (as_of_ts.isoformat(), vplan_obj.vplan_id_int))


def _ensure_schema(connection_obj):
    connection_obj.execute("""CREATE TABLE IF NOT EXISTS auto_completion_attempt (
        vplan_id_int INTEGER NOT NULL, asset_str TEXT NOT NULL, decision_plan_id_int INTEGER NOT NULL,
        pod_id_str TEXT NOT NULL, account_route_str TEXT NOT NULL,
        original_broker_order_id_str TEXT NOT NULL, original_message_str TEXT NOT NULL,
        attempted_timestamp_str TEXT NOT NULL, updated_timestamp_str TEXT NOT NULL,
        ended_timestamp_str TEXT, last_checked_timestamp_str TEXT,
        order_request_key_str TEXT NOT NULL, request_json_str TEXT, broker_order_id_str TEXT,
        status_str TEXT NOT NULL, reason_str TEXT NOT NULL,
        quantity_float REAL NOT NULL, remaining_quantity_float REAL,
        PRIMARY KEY(vplan_id_int,asset_str))""")


def _claim(state_store_obj, vplan_obj, asset_str, leg_dict, as_of_ts):
    with state_store_obj._connect() as connection_obj:
        connection_obj.execute("BEGIN IMMEDIATE")
        current_row_obj = connection_obj.execute("SELECT status_str FROM vplan WHERE vplan_id_int=? AND pod_id_str=? AND account_route_str=?",
            (vplan_obj.vplan_id_int, vplan_obj.pod_id_str, vplan_obj.account_route_str)).fetchone()
        latest_row_obj = connection_obj.execute("SELECT decision_plan_id_int FROM decision_plan WHERE pod_id_str=? ORDER BY decision_plan_id_int DESC LIMIT 1",
            (vplan_obj.pod_id_str,)).fetchone()
        if (current_row_obj is None or current_row_obj[0] not in {"submitted", "submitting"}
                or latest_row_obj is None or latest_row_obj[0] != vplan_obj.decision_plan_id_int):
            return False
        _ensure_schema(connection_obj)
        reference_str = f"{vplan_obj.submission_key_str or f'vplan:{vplan_obj.decision_plan_id_int}'}:auto-complete:{asset_str}"
        cursor_obj = connection_obj.execute("""INSERT OR IGNORE INTO auto_completion_attempt
            (vplan_id_int,asset_str,decision_plan_id_int,pod_id_str,account_route_str,
             original_broker_order_id_str,original_message_str,attempted_timestamp_str,updated_timestamp_str,
             order_request_key_str,status_str,reason_str,quantity_float)
            VALUES (?,?,?,?,?,?,?,?,?,?,'working','Attempt started; awaiting broker check',0)""",
            (vplan_obj.vplan_id_int, asset_str, vplan_obj.decision_plan_id_int, vplan_obj.pod_id_str,
                vplan_obj.account_route_str, leg_dict["original_broker_order_id_str"], leg_dict["original_message_str"],
                as_of_ts.isoformat(), as_of_ts.isoformat(), reference_str))
        return cursor_obj.rowcount == 1


def _update(state_store_obj, vplan_obj, asset_str, as_of_ts, **field_dict):
    field_dict["updated_timestamp_str"] = as_of_ts.isoformat()
    with state_store_obj._connect() as connection_obj:
        connection_obj.execute("UPDATE auto_completion_attempt SET " + ",".join(f"{key_str}=?" for key_str in field_dict)
            + " WHERE vplan_id_int=? AND asset_str=?", (*field_dict.values(), vplan_obj.vplan_id_int, asset_str))


def _emit(log_path_str, event_str, vplan_obj, asset_str, as_of_ts, **field_dict):
    if log_path_str is not None:
        from alpha.live.logging_utils import log_event
        log_event(event_str, {"pod_id_str": vplan_obj.pod_id_str, "account_route_str": vplan_obj.account_route_str,
            "vplan_id_int": vplan_obj.vplan_id_int, "decision_plan_id_int": vplan_obj.decision_plan_id_int,
            "asset_str": asset_str, "as_of_timestamp_str": as_of_ts.isoformat(), **field_dict}, log_path_str=log_path_str)


def _persist_result(state_store_obj, vplan_obj, result_obj):
    identity_dict = {"vplan_id_int": vplan_obj.vplan_id_int, "decision_plan_id_int": vplan_obj.decision_plan_id_int}
    state_store_obj.upsert_vplan_broker_order_record_list([replace(record_obj, **identity_dict)
        for record_obj in result_obj.broker_order_record_list])
    state_store_obj.insert_vplan_broker_order_event_list([replace(event_obj, **identity_dict)
        for event_obj in result_obj.broker_order_event_list])
    state_store_obj.upsert_vplan_broker_ack_list([replace(ack_obj, **identity_dict)
        for ack_obj in result_obj.broker_order_ack_list])
    state_store_obj.upsert_vplan_fill_list([replace(fill_obj, **identity_dict)
        for fill_obj in result_obj.broker_order_fill_list])


def finish_daily_closed_attempts(state_store_obj, vplan_obj, snapshot_obj, as_of_ts, *, log_path_str=None):
    """Called only after daily settlement has confirmed all owned orders absent."""
    with state_store_obj._connect() as connection_obj:
        attempt_list = load_attempt_row_list(connection_obj, vplan_obj.vplan_id_int)
    for attempt_dict in attempt_list:
        if attempt_dict["ended_timestamp_str"]:
            continue
        asset_str = attempt_dict["asset_str"]
        remainder_float = abs(float(vplan_obj.target_share_map[asset_str])
            - float(snapshot_obj.position_amount_map.get(asset_str, 0)))
        # Settlement proves the order is no longer working; holdings alone do not prove its fills.
        reason_str = "Daily close checked; no owned order working"
        _update(state_store_obj, vplan_obj, asset_str, as_of_ts, status_str="not_completed", reason_str=reason_str,
            ended_timestamp_str=as_of_ts.isoformat(), remaining_quantity_float=remainder_float,
            last_checked_timestamp_str=snapshot_obj.snapshot_timestamp_ts.isoformat())
        _emit(log_path_str, "auto_complete_not_completed", vplan_obj, asset_str, as_of_ts,
            quantity_float=attempt_dict["quantity_float"], remaining_quantity_float=remainder_float, reason_str=reason_str,
            side_str=("BUY" if json.loads(attempt_dict["request_json_str"])["amount_float"] > 0 else "SELL")
                if attempt_dict["request_json_str"] else None)


def _filled_quantity_float(order_row_list, event_row_list, fill_row_list, reference_str, order_id_str):
    matching_order_list = [row_dict for row_dict in order_row_list
        if (reference_str and row_dict.get("order_request_key_str") == reference_str)
        or (order_id_str and row_dict.get("broker_order_id_str") == order_id_str)]
    order_id_set = {str(row_dict["broker_order_id_str"]) for row_dict in [*matching_order_list, *event_row_list]
        if (reference_str and row_dict.get("order_request_key_str") == reference_str)
        or (order_id_str and row_dict.get("broker_order_id_str") == order_id_str)} | {order_id_str}
    cumulative_list = [abs(float(row_dict.get("filled_amount_float") or 0))
        for row_dict in [*matching_order_list, *event_row_list]
        if (reference_str and row_dict.get("order_request_key_str") == reference_str)
        or str(row_dict.get("broker_order_id_str")) in order_id_set]
    # The same saved execution can retain its local ID and later permanent ID. Count its existing fill tuple once.
    fill_tuple_set = {(row_dict["fill_timestamp_str"], row_dict["fill_amount_float"], row_dict["fill_price_float"])
        for row_dict in fill_row_list if str(row_dict["broker_order_id_str"]) in order_id_set}
    executed_float = sum(abs(float(fill_tuple[1])) for fill_tuple in fill_tuple_set)
    return max([executed_float, *cumulative_list])


def reconcile_auto_completion(state_store_obj, broker_adapter_obj, release_obj, vplan_obj, as_of_ts,
        *, log_path_str=None, releases_root_path_str=None, observed_result_obj=None, now_fn=None):
    """One original-order remainder, then terminal evidence or the unanswered-send rule."""
    from alpha.live.daily_reconcile import _refresh
    clock_fn = now_fn or _utc_now_ts
    leg_map_dict = load_leg_dict(state_store_obj, vplan_obj)
    with state_store_obj._connect() as connection_obj:
        attempt_map_dict = {row_dict["asset_str"]: row_dict for row_dict in load_attempt_row_list(connection_obj, vplan_obj.vplan_id_int)}
    result_dict = {"affected_asset_set": set(leg_map_dict) | set(attempt_map_dict), "active_bool": False,
        "snapshot_obj": None, "reporting_result_list": [], "sent_bool": False, "observed_result_obj": None}
    observed_cross_bool = observed_result_obj is not None and any(
        CROSS_MESSAGE_STR in str(event_obj.message_str or "").casefold()
        for event_obj in observed_result_obj.broker_order_event_list)
    if not leg_map_dict and not attempt_map_dict and not observed_cross_bool:
        return result_dict
    # Daily pods normally read history only at EOD. This affected branch needs original fills and re-send terminal state.
    if observed_result_obj is None and hasattr(broker_adapter_obj, "get_recent_order_state_snapshot"):
        try:
            known_id_set = {str(row_dict["broker_order_id_str"]) for row_dict in
                state_store_obj.get_broker_order_row_dict_list_for_vplan(vplan_obj.vplan_id_int)}
            record_list, event_list, fill_list = broker_adapter_obj.get_recent_order_state_snapshot(
                account_route_str=vplan_obj.account_route_str, since_timestamp_ts=vplan_obj.submission_timestamp_ts,
                submission_key_str=vplan_obj.submission_key_str or f"vplan:{vplan_obj.decision_plan_id_int}",
                allowed_broker_order_id_set=known_id_set)
            observed_result_obj = SubmitBatchResult(broker_order_record_list=record_list,
                broker_order_event_list=event_list, broker_order_fill_list=fill_list)
        except Exception as exception_obj:
            _emit(log_path_str, "auto_complete_observation_unavailable", vplan_obj, None, as_of_ts,
                error_str=str(exception_obj))
    if observed_result_obj is not None:
        normalized_record_list = []
        for record_obj in observed_result_obj.broker_order_record_list:
            leg_dict = leg_map_dict.get(record_obj.asset_str)
            if (leg_dict is not None and record_obj.broker_order_type_str == "MOO" and record_obj.amount_float == 0
                    and record_obj.account_route_str == vplan_obj.account_route_str
                    and (record_obj.broker_order_id_str == leg_dict["original_broker_order_id_str"]
                        or (leg_dict["original_order_request_key_str"]
                            and record_obj.order_request_key_str == leg_dict["original_order_request_key_str"]))):
                record_obj = replace(record_obj, amount_float=leg_dict["cancelled_qty_float"] * leg_dict["direction_float"])
            normalized_record_list.append(record_obj)
        observed_result_obj = replace(observed_result_obj, broker_order_record_list=normalized_record_list)
        result_dict["observed_result_obj"] = observed_result_obj
        _persist_result(state_store_obj, vplan_obj, observed_result_obj)
        result_dict["reporting_result_list"].append(observed_result_obj)
        # Retain known original quantity if a sparse refresh omits it.
        leg_map_dict.update(load_leg_dict(state_store_obj, vplan_obj))
        result_dict["affected_asset_set"].update(leg_map_dict)
    window_dict = timing_dict(vplan_obj, release_obj.session_calendar_id_str)
    order_row_list = state_store_obj.get_broker_order_row_dict_list_for_vplan(vplan_obj.vplan_id_int)
    event_row_list = state_store_obj.get_broker_order_event_row_dict_list_for_vplan(vplan_obj.vplan_id_int)
    fill_row_list = state_store_obj.get_fill_row_dict_list_for_vplan(vplan_obj.vplan_id_int, include_order_identity_bool=True)
    for asset_str, leg_dict in leg_map_dict.items():
        send_ts = clock_fn()
        if asset_str in attempt_map_dict or not window_dict["start_ts"] <= send_ts < window_dict["send_cutoff_ts"]:
            continue
        if not _claim(state_store_obj, vplan_obj, asset_str, leg_dict, send_ts):
            continue
        # *** CRITICAL *** Original execution only: Q = sign(original) * max(0, abs(original) - original fills).
        # Current holdings never resize this re-send; the ordinary reconciliation still checks the frozen target.
        filled_float = max(leg_dict["filled_quantity_float"], _filled_quantity_float(order_row_list,
            event_row_list, fill_row_list, leg_dict["original_order_request_key_str"], leg_dict["original_broker_order_id_str"]))
        quantity_float = max(0.0, leg_dict["cancelled_qty_float"] - filled_float)
        side_str = "BUY" if leg_dict["direction_float"] > 0 else "SELL"
        _emit(log_path_str, "auto_complete_planned", vplan_obj, asset_str, send_ts,
            side_str=side_str, quantity_float=quantity_float)
        if quantity_float <= 1e-9:
            _update(state_store_obj, vplan_obj, asset_str, send_ts, status_str="completed",
                reason_str="Original order already filled", ended_timestamp_str=send_ts.isoformat(), remaining_quantity_float=0.0)
            continue
        submission_key_str = vplan_obj.submission_key_str or f"vplan:{vplan_obj.decision_plan_id_int}"
        request_obj = BrokerOrderRequest(release_obj.release_id_str, release_obj.pod_id_str, release_obj.account_route_str,
            submission_key_str, f"{submission_key_str}:auto-complete:{asset_str}", asset_str, "MKT", "MarketOrder", "shares",
            quantity_float * leg_dict["direction_float"], False, None,
            float(vplan_obj.live_reference_price_map.get(asset_str, 0)), vplan_obj.pod_budget_float,
            decision_plan_id_int=vplan_obj.decision_plan_id_int, vplan_id_int=vplan_obj.vplan_id_int,
            submission_deadline_timestamp_str=window_dict["send_cutoff_ts"].isoformat())
        _update(state_store_obj, vplan_obj, asset_str, send_ts, quantity_float=quantity_float,
            request_json_str=json.dumps(asdict(request_obj), sort_keys=True, allow_nan=False), remaining_quantity_float=quantity_float)
        latest_plan_obj = state_store_obj.get_latest_decision_plan_for_pod(vplan_obj.pod_id_str)
        if latest_plan_obj is None or latest_plan_obj.decision_plan_id_int != vplan_obj.decision_plan_id_int:
            _update(state_store_obj, vplan_obj, asset_str, send_ts, status_str="not_completed",
                reason_str="A newer decision supersedes this plan", ended_timestamp_str=send_ts.isoformat())
            _emit(log_path_str, "auto_complete_not_completed", vplan_obj, asset_str, send_ts,
                side_str=side_str, quantity_float=quantity_float, remaining_quantity_float=quantity_float,
                reason_str="A newer decision supersedes this plan", sent_bool=False)
            continue
        try:
            result_dict["sent_bool"] = True
            submit_obj = broker_adapter_obj.submit_order_request_list(account_route_str=vplan_obj.account_route_str,
                broker_order_request_list=[request_obj], submitted_timestamp_ts=send_ts)
            _persist_result(state_store_obj, vplan_obj, submit_obj)
            result_dict["reporting_result_list"].append(submit_obj)
            order_id_str = next((record_obj.broker_order_id_str for record_obj in submit_obj.broker_order_record_list
                if record_obj.order_request_key_str == request_obj.order_request_key_str), None)
            acknowledged_bool = submit_obj.submit_ack_status_str == "complete"
            returned_ts = clock_fn()
            _update(state_store_obj, vplan_obj, asset_str, returned_ts,
                attempted_timestamp_str=(send_ts if acknowledged_bool else returned_ts).isoformat(),
                broker_order_id_str=order_id_str if acknowledged_bool else None,
                reason_str="Sent; awaiting broker check" if acknowledged_bool else "Attempt started; awaiting broker check")
            _emit(log_path_str, "auto_complete_sent", vplan_obj, asset_str, returned_ts, broker_order_id_str=order_id_str,
                side_str=side_str, quantity_float=quantity_float, broker_acknowledged_bool=acknowledged_bool)
        except Exception as exception_obj:
            # May have reached IBKR. The durable mark forbids replay, including after restart.
            returned_ts = clock_fn()
            _update(state_store_obj, vplan_obj, asset_str, returned_ts, attempted_timestamp_str=returned_ts.isoformat(),
                reason_str="Awaiting broker check after send interruption")
            _emit(log_path_str, "auto_complete_send_unanswered", vplan_obj, asset_str, returned_ts,
                side_str=side_str, quantity_float=quantity_float, error_str=str(exception_obj))
    with state_store_obj._connect() as connection_obj:
        attempt_list = load_attempt_row_list(connection_obj, vplan_obj.vplan_id_int)
    order_row_list = state_store_obj.get_broker_order_row_dict_list_for_vplan(vplan_obj.vplan_id_int)
    event_row_list = state_store_obj.get_broker_order_event_row_dict_list_for_vplan(vplan_obj.vplan_id_int)
    fill_row_list = state_store_obj.get_fill_row_dict_list_for_vplan(vplan_obj.vplan_id_int, include_order_identity_bool=True)
    for attempt_dict in attempt_list:
        if attempt_dict["ended_timestamp_str"]:
            continue
        asset_str = attempt_dict["asset_str"]
        checked_ts = clock_fn()
        linked_list = [row_dict for row_dict in [*order_row_list, *event_row_list]
            if row_dict.get("order_request_key_str") == attempt_dict["order_request_key_str"]]
        terminal_bool = any(row_dict["status_str"] in TERMINAL_STATUS_SET for row_dict in linked_list)
        if terminal_bool:
            filled_float = _filled_quantity_float(order_row_list, event_row_list, fill_row_list,
                attempt_dict["order_request_key_str"], attempt_dict["broker_order_id_str"])
            remainder_float = max(0.0, attempt_dict["quantity_float"] - filled_float)
            complete_bool = remainder_float <= 1e-9
            reason_str = "Re-sent and filled" if complete_bool else "Re-sent, not filled"
        else:
            acknowledged_bool = bool(attempt_dict["broker_order_id_str"]) or any(
                row_dict["status_str"] in {"Submitted", "PreSubmitted", "PartiallyFilled", "PendingCancel"} for row_dict in linked_list)
            if acknowledged_bool or checked_ts < datetime.fromisoformat(attempt_dict["attempted_timestamp_str"]) + timedelta(minutes=2):
                continue
            try:
                final_obj = _refresh(broker_adapter_obj, release_obj, checked_ts)
            except Exception:
                _update(state_store_obj, vplan_obj, asset_str, checked_ts, reason_str="Broker check unavailable; no new order")
                continue
            result_dict["snapshot_obj"] = final_obj.broker_snapshot_obj
            remainder_float = abs(float(vplan_obj.target_share_map[asset_str])
                - float(final_obj.broker_snapshot_obj.position_amount_map.get(asset_str, 0)))
            if any(row_dict["asset_str"] == asset_str for row_dict in final_obj.open_order_row_list):
                _update(state_store_obj, vplan_obj, asset_str, checked_ts, reason_str=f"Order still working in {asset_str}; no new order",
                    remaining_quantity_float=remainder_float, last_checked_timestamp_str=final_obj.refreshed_timestamp_ts.isoformat())
                continue
            # Holdings can close the plan, but only execution evidence may label the re-send Filled.
            complete_bool = False
            reason_str = "No broker answer; checked, no order working"
        _update(state_store_obj, vplan_obj, asset_str, checked_ts, status_str="completed" if complete_bool else "not_completed",
            reason_str=reason_str, ended_timestamp_str=checked_ts.isoformat(), remaining_quantity_float=remainder_float,
            last_checked_timestamp_str=checked_ts.isoformat())
        _emit(log_path_str, "auto_complete_done" if complete_bool else "auto_complete_not_completed", vplan_obj,
            asset_str, checked_ts, remaining_quantity_float=remainder_float, quantity_float=attempt_dict["quantity_float"],
            # Report only: the re-send's own executed shares, so Activity can say "Partly filled" without a price.
            filled_quantity_float=filled_float if terminal_bool else None,
            side_str=("BUY" if json.loads(attempt_dict["request_json_str"])["amount_float"] > 0 else "SELL")
                if attempt_dict["request_json_str"] else None,
            fill_price_float=next((row_dict["avg_fill_price_float"] for row_dict in linked_list
                if row_dict.get("avg_fill_price_float") is not None), None))
    result_dict["active_bool"] = active_auto_completion_bool(state_store_obj, vplan_obj, as_of_ts)
    return result_dict
