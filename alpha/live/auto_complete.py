"""One opening cross-cancel completion; no retry and no changes to strategy targets."""
from dataclasses import asdict, replace
from datetime import datetime, timedelta
import json
import math

from alpha.live import scheduler_utils
from alpha.live.models import BrokerOrderRequest

TABLE_NAME_STR = "auto_completion_attempt"
CROSS_MESSAGE_STR = "order would cross related resting order"
TERMINAL_STATUS_SET = {"Filled", "Cancelled", "ApiCancelled", "Rejected", "Expired", "Inactive"}


def timing_dict(vplan_obj, calendar_id_str):
    session_date_obj = scheduler_utils.to_market_timestamp_ts(
        vplan_obj.target_execution_timestamp_ts, calendar_id_str).date()
    open_ts = scheduler_utils.get_session_open_timestamp_ts(session_date_obj, calendar_id_str)
    return {"open_ts": open_ts, "manual_block_ts": open_ts + timedelta(minutes=4),
        "start_ts": open_ts + timedelta(minutes=5), "send_cutoff_ts": open_ts + timedelta(minutes=6),
        "expiry_ts": open_ts + timedelta(minutes=30), "final_check_ts": open_ts + timedelta(minutes=31),
        "close_ts": scheduler_utils.get_session_close_timestamp_ts(session_date_obj, calendar_id_str)}


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
            "original_message_str": message_str, "cancelled_qty_float": abs(amount_float),
            "direction_float": 1.0 if amount_float > 0 else -1.0, "target_float": target_float,
            "short_bool": target_float < min(float(vplan_obj.current_broker_position_map.get(asset_str, 0)), 0)}
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
    if as_of_ts < vplan_obj.target_execution_timestamp_ts + timedelta(minutes=6):
        marked_set = {row_dict["asset_str"] for row_dict in attempt_list}
        return any(not leg_dict["short_bool"] and asset_str not in marked_set
            for asset_str, leg_dict in load_leg_dict(state_store_obj, vplan_obj).items())
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


def _dedicated_account_bool(state_store_obj, release_obj, releases_root_path_str):
    if releases_root_path_str is None:
        deployment_release_list = state_store_obj.get_enabled_release_list()
    else:
        from alpha.live.release_manifest import load_release_list
        deployment_release_list = load_release_list(releases_root_path_str)
    return not any(other_obj.enabled_bool and other_obj.mode_str == release_obj.mode_str
        and other_obj.account_route_str.strip().upper() == release_obj.account_route_str.strip().upper()
        and other_obj.pod_id_str != release_obj.pod_id_str for other_obj in deployment_release_list)


def reconcile_auto_completion(state_store_obj, broker_adapter_obj, release_obj, vplan_obj, as_of_ts,
        *, log_path_str=None, releases_root_path_str=None):
    """Affected branch only. Failure to observe the broker never unlocks an attempted leg."""
    from alpha.live.daily_reconcile import _refresh
    leg_map_dict = load_leg_dict(state_store_obj, vplan_obj)
    with state_store_obj._connect() as connection_obj:
        attempt_map_dict = {row_dict["asset_str"]: row_dict for row_dict in load_attempt_row_list(connection_obj, vplan_obj.vplan_id_int)}
    result_dict = {"affected_asset_set": set(leg_map_dict) | set(attempt_map_dict), "active_bool": False,
        "snapshot_obj": None, "reporting_result_list": []}
    if not leg_map_dict and not attempt_map_dict:
        return result_dict
    # Once claimed, final observation depends on the durable attempt, not on later sparse original-order refreshes.
    window_dict = timing_dict(vplan_obj, release_obj.session_calendar_id_str)
    for asset_str, leg_dict in leg_map_dict.items():
        if asset_str in attempt_map_dict or leg_dict["short_bool"]:
            continue
        if not window_dict["start_ts"] <= as_of_ts < window_dict["send_cutoff_ts"]:
            continue
        if not _claim(state_store_obj, vplan_obj, asset_str, leg_dict, as_of_ts):
            continue
        _emit(log_path_str, "auto_complete_planned", vplan_obj, asset_str, as_of_ts)
        # *** CRITICAL *** Execution-only recovery: frozen target and direction, fresh broker remainder.
        # Q = min(original cancelled quantity, abs(target - actual)); no signal or target is rebuilt.
        try:
            dedicated_bool = _dedicated_account_bool(state_store_obj, release_obj, releases_root_path_str)
            snapshot_obj = _refresh(broker_adapter_obj, release_obj, as_of_ts)
            result_dict["snapshot_obj"] = snapshot_obj.broker_snapshot_obj
            actual_float = float(snapshot_obj.broker_snapshot_obj.position_amount_map.get(asset_str, 0))
            gap_float = leg_dict["target_float"] - actual_float
            latest_plan_obj = state_store_obj.get_latest_decision_plan_for_pod(vplan_obj.pod_id_str)
            reason_str = ("A newer decision supersedes this plan" if latest_plan_obj is None
                or latest_plan_obj.decision_plan_id_int != vplan_obj.decision_plan_id_int else
                "Account shared with another pod" if not dedicated_bool else
                f"not sent: open order in {asset_str}" if any(row_dict["asset_str"] == asset_str for row_dict in snapshot_obj.open_order_row_list) else
                "nothing missing" if abs(gap_float) < 1e-9 else
                "gap changed direction" if gap_float * leg_dict["direction_float"] <= 0 else
                "opens a short" if leg_dict["target_float"] < min(actual_float, 0) else
                "submission minute missed" if not window_dict["start_ts"] <= snapshot_obj.refreshed_timestamp_ts < window_dict["send_cutoff_ts"] else "")
            quantity_float = float(math.floor(min(leg_dict["cancelled_qty_float"], abs(gap_float)) + 1e-9))
            if not quantity_float and not reason_str:
                reason_str = "no whole shares missing"
            if reason_str:
                _update(state_store_obj, vplan_obj, asset_str, as_of_ts, status_str="not_completed", reason_str=reason_str,
                    ended_timestamp_str=as_of_ts.isoformat(), remaining_quantity_float=abs(gap_float),
                    last_checked_timestamp_str=snapshot_obj.refreshed_timestamp_ts.isoformat())
                _emit(log_path_str, "auto_complete_not_completed", vplan_obj, asset_str, as_of_ts, reason_str=reason_str)
                continue
            submission_key_str = vplan_obj.submission_key_str or f"vplan:{vplan_obj.decision_plan_id_int}"
            request_obj = BrokerOrderRequest(release_obj.release_id_str, release_obj.pod_id_str, release_obj.account_route_str,
                submission_key_str, f"{submission_key_str}:auto-complete:{asset_str}", asset_str, "MKT", "MarketOrder", "shares",
                quantity_float * leg_dict["direction_float"], False, None,
                float(vplan_obj.live_reference_price_map.get(asset_str, 0)), vplan_obj.pod_budget_float,
                decision_plan_id_int=vplan_obj.decision_plan_id_int, vplan_id_int=vplan_obj.vplan_id_int,
                execution_deadline_timestamp_str=window_dict["expiry_ts"].isoformat(),
                submission_deadline_timestamp_str=window_dict["send_cutoff_ts"].isoformat())
            _update(state_store_obj, vplan_obj, asset_str, as_of_ts, quantity_float=quantity_float,
                request_json_str=json.dumps(asdict(request_obj), sort_keys=True, allow_nan=False), remaining_quantity_float=abs(gap_float))
        except Exception as exception_obj:
            # The opportunity is consumed, but no broker send has started.
            _update(state_store_obj, vplan_obj, asset_str, as_of_ts, status_str="not_completed",
                reason_str=f"not sent: {type(exception_obj).__name__}", ended_timestamp_str=as_of_ts.isoformat())
            _emit(log_path_str, "auto_complete_not_completed", vplan_obj, asset_str, as_of_ts, error_str=str(exception_obj))
            continue
        try:
            submit_obj = broker_adapter_obj.submit_order_request_list(account_route_str=vplan_obj.account_route_str,
                broker_order_request_list=[request_obj], submitted_timestamp_ts=snapshot_obj.refreshed_timestamp_ts)
            _persist_result(state_store_obj, vplan_obj, submit_obj)
            result_dict["reporting_result_list"].append(submit_obj)
            order_id_str = next((record_obj.broker_order_id_str for record_obj in submit_obj.broker_order_record_list
                if record_obj.order_request_key_str == request_obj.order_request_key_str), None)
            acknowledged_bool = submit_obj.submit_ack_status_str == "complete"
            _update(state_store_obj, vplan_obj, asset_str, as_of_ts,
                broker_order_id_str=order_id_str if acknowledged_bool else None,
                reason_str="Sent; awaiting broker check" if acknowledged_bool else "Attempt started; awaiting broker check")
            _emit(log_path_str, "auto_complete_sent", vplan_obj, asset_str, as_of_ts, broker_order_id_str=order_id_str,
                quantity_float=quantity_float, broker_acknowledged_bool=acknowledged_bool)
        except Exception as exception_obj:
            # May have reached IBKR. The durable mark forbids replay, including after restart.
            _update(state_store_obj, vplan_obj, asset_str, as_of_ts, reason_str="Awaiting broker check after send interruption")
            _emit(log_path_str, "auto_complete_not_completed", vplan_obj, asset_str, as_of_ts, error_str=str(exception_obj))
    with state_store_obj._connect() as connection_obj:
        attempt_list = load_attempt_row_list(connection_obj, vplan_obj.vplan_id_int)
    order_row_list = state_store_obj.get_broker_order_row_dict_list_for_vplan(vplan_obj.vplan_id_int)
    for attempt_dict in attempt_list:
        if attempt_dict["ended_timestamp_str"]:
            continue
        asset_str = attempt_dict["asset_str"]
        terminal_bool = any(row_dict.get("order_request_key_str") == attempt_dict["order_request_key_str"]
            and row_dict["status_str"] in TERMINAL_STATUS_SET for row_dict in order_row_list)
        if not terminal_bool and as_of_ts < window_dict["final_check_ts"]:
            continue
        try:
            final_obj = _refresh(broker_adapter_obj, release_obj, as_of_ts)
        except Exception as exception_obj:
            _update(state_store_obj, vplan_obj, asset_str, as_of_ts, reason_str="Broker check unavailable; no new order")
            continue
        result_dict["snapshot_obj"] = final_obj.broker_snapshot_obj
        remainder_float = abs(float(vplan_obj.target_share_map[asset_str])
            - float(final_obj.broker_snapshot_obj.position_amount_map.get(asset_str, 0)))
        if any(row_dict["asset_str"] == asset_str for row_dict in final_obj.open_order_row_list):
            _update(state_store_obj, vplan_obj, asset_str, as_of_ts, reason_str=f"Order still working in {asset_str}; no new order",
                remaining_quantity_float=remainder_float, last_checked_timestamp_str=final_obj.refreshed_timestamp_ts.isoformat())
            continue
        complete_bool = remainder_float < 1e-9
        _update(state_store_obj, vplan_obj, asset_str, as_of_ts, status_str="completed" if complete_bool else "not_completed",
            reason_str="Completed" if complete_bool else "Not completed; broker checked, no order working",
            ended_timestamp_str=as_of_ts.isoformat(), remaining_quantity_float=remainder_float,
            last_checked_timestamp_str=final_obj.refreshed_timestamp_ts.isoformat())
        _emit(log_path_str, "auto_complete_done" if complete_bool else "auto_complete_not_completed", vplan_obj,
            asset_str, as_of_ts, remaining_quantity_float=remainder_float)
    result_dict["active_bool"] = active_auto_completion_bool(state_store_obj, vplan_obj, as_of_ts)
    return result_dict
