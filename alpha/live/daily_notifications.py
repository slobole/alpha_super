"""Daily-only durable halt/overdue alerts; the existing watchdog delivers them.

One local outbox row is claimed at a time. HTTP delivery remains at least once:
a crash after Discord accepts a message can repeat it after the lease expires.
"""
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path
import sqlite3
from uuid import uuid4

from alpha.live import scheduler_utils
from alpha.live.mr_capsule_notifications import _configured_scope_list


DELIVERY_LEASE_SECONDS_INT = 300
ALERT_KIND_SET = {"cycle_overdue", "capsule_holding_halt"}


def ensure_daily_alert_schema(connection_obj):
    connection_obj.execute("""CREATE TABLE IF NOT EXISTS daily_pod_alert (
        alert_key_str TEXT NOT NULL, alert_kind_str TEXT NOT NULL, release_id_str TEXT NOT NULL,
        pod_id_str TEXT NOT NULL, account_route_str TEXT NOT NULL, mode_str TEXT NOT NULL,
        decision_plan_id_int INTEGER, payload_json_str TEXT NOT NULL, created_timestamp_str TEXT NOT NULL,
        delivery_claim_str TEXT, delivery_claimed_timestamp_str TEXT, delivered_timestamp_str TEXT,
        attempt_count_int INTEGER NOT NULL DEFAULT 0,
        PRIMARY KEY(alert_key_str,pod_id_str,account_route_str,mode_str))""")


def _timestamp_str(timestamp_ts):
    if timestamp_ts.tzinfo is None or timestamp_ts.utcoffset() is None:
        raise ValueError("Daily alert timestamps must be timezone aware.")
    return timestamp_ts.astimezone(timezone.utc).isoformat()


def _enqueue(connection_obj, release_obj, alert_key_str, alert_kind_str, payload_dict,
        as_of_ts, decision_plan_id_int=None):
    from alpha.live.daily_reconcile import is_daily_reconcile_release_bool

    if (not is_daily_reconcile_release_bool(release_obj) or alert_kind_str not in ALERT_KIND_SET
            or release_obj.mode_str not in {"live", "paper", "incubation"}):
        raise ValueError("Daily alerts require a supported daily pod and alert kind.")
    cursor_obj = connection_obj.execute("""INSERT OR IGNORE INTO daily_pod_alert
        (alert_key_str,alert_kind_str,release_id_str,pod_id_str,account_route_str,mode_str,
         decision_plan_id_int,payload_json_str,created_timestamp_str) VALUES (?,?,?,?,?,?,?,?,?)""",
        (alert_key_str, alert_kind_str, release_obj.release_id_str, release_obj.pod_id_str,
            release_obj.account_route_str, release_obj.mode_str, decision_plan_id_int,
            json.dumps(payload_dict, sort_keys=True, allow_nan=False), _timestamp_str(as_of_ts)))
    return cursor_obj.rowcount == 1


def enqueue_daily_cycle_overdue(state_store_obj, release_obj, decision_plan_obj, as_of_ts,
        *, error_str, owned_order_row_list=None):
    """One CRITICAL per still-open cycle, beginning at its exchange close + 1h."""
    from alpha.live.daily_reconcile import DAILY_TERMINAL_STATUS_SET, _validate_identity

    _validate_identity(release_obj, decision_plan_obj, None)
    session_date_obj = scheduler_utils.to_market_timestamp_ts(decision_plan_obj.target_execution_timestamp_ts,
        release_obj.session_calendar_id_str).date()
    close_ts = scheduler_utils.get_session_close_timestamp_ts(session_date_obj, release_obj.session_calendar_id_str)
    if as_of_ts < close_ts + timedelta(hours=1):
        return False
    detail_list = []
    for row_dict in owned_order_row_list or []:
        quantity_float = row_dict.get("remaining_amount_float", row_dict.get("amount_float"))
        if (row_dict.get("account_route_str") != release_obj.account_route_str or not row_dict.get("asset_str")
                or row_dict.get("side_str") not in {"BUY", "SELL"} or quantity_float is None
                or not math.isfinite(float(quantity_float)) or float(quantity_float) < 0
                or not isinstance(row_dict.get("client_id_int"), int)):
            raise ValueError("Overdue order details require account, symbol, quantity, side and client ID.")
        detail_list.append({"asset_str": row_dict["asset_str"], "quantity_float": float(quantity_float),
            "side_str": row_dict["side_str"], "client_id_int": row_dict["client_id_int"]})
    with state_store_obj._connect() as connection_obj:
        connection_obj.execute("BEGIN IMMEDIATE")
        row_obj = connection_obj.execute("""SELECT status_str FROM decision_plan
            WHERE decision_plan_id_int=? AND release_id_str=? AND pod_id_str=? AND account_route_str=?""",
            (decision_plan_obj.decision_plan_id_int, release_obj.release_id_str,
                release_obj.pod_id_str, release_obj.account_route_str)).fetchone()
        if row_obj is None or row_obj[0] in {*DAILY_TERMINAL_STATUS_SET, "superseded"}:
            return False
        return _enqueue(connection_obj, release_obj, f"cycle_overdue:{decision_plan_obj.decision_plan_id_int}",
            "cycle_overdue", {"session_close_timestamp_str": close_ts.isoformat(), "error_str": str(error_str),
                "owned_order_row_list": detail_list, "order_details_available_bool": owned_order_row_list is not None,
                "required_action_str": "Check the daily-cycle error log, restore broker connectivity if needed, and verify owned orders and owning client IDs; let daily reconciliation retry. Do not send replacement orders from this alert."},
            as_of_ts, decision_plan_obj.decision_plan_id_int)


def enqueue_capsule_holding_halt(state_store_obj, release_obj, signal_date_obj, error_obj, as_of_ts):
    from alpha.live.mr_capsule_adapter import CapsuleHoldingMismatchError, MR_CAPSULE_STRATEGY_IMPORT_TUPLE

    if release_obj.strategy_import_str not in MR_CAPSULE_STRATEGY_IMPORT_TUPLE or not isinstance(error_obj, CapsuleHoldingMismatchError):
        raise ValueError("Capsule holding alerts require a structured capsule holding mismatch.")
    signal_date_str = str(signal_date_obj.date() if hasattr(signal_date_obj, "date") else signal_date_obj)
    with state_store_obj._connect() as connection_obj:
        return _enqueue(connection_obj, release_obj, f"capsule_holding_halt:{signal_date_str}",
            "capsule_holding_halt", {"signal_date_str": signal_date_str, "reason_str": error_obj.reason_str,
                "holding_row_list": error_obj.holding_row_list, "required_action_str": error_obj.required_action_str}, as_of_ts)


def _pending_alert_list(scope_tuple):
    db_path_str, pod_id_str, account_route_str, mode_str = scope_tuple
    db_path_obj = Path(db_path_str)
    if not db_path_obj.is_file():
        return []
    with closing(sqlite3.connect(db_path_obj.as_uri() + "?mode=ro", uri=True)) as connection_obj:
        connection_obj.row_factory = sqlite3.Row
        if not connection_obj.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='daily_pod_alert'").fetchone():
            return []
        return [dict(row_obj) for row_obj in connection_obj.execute("""SELECT * FROM daily_pod_alert
            WHERE pod_id_str=? AND account_route_str=? AND mode_str=? AND delivered_timestamp_str IS NULL
            ORDER BY created_timestamp_str,alert_key_str""", (pod_id_str, account_route_str, mode_str))]


def pending_daily_alert_count_int(summary_dict, *, mode_str=None):
    return sum(len(_pending_alert_list(scope_tuple)) for scope_tuple in _configured_scope_list(summary_dict, mode_str))


def _payload_list(alert_dict, current_status_str):
    payload_dict = json.loads(alert_dict["payload_json_str"])
    historical_bool = current_status_str in {"completed", "completed_with_exceptions", "superseded"}
    line_list = [f"CRITICAL DAILY / {alert_dict['mode_str'].upper()} / {alert_dict['pod_id_str']}",
        f"Account {alert_dict['account_route_str']} | Observed {alert_dict['created_timestamp_str']}"]
    if alert_dict["alert_kind_str"] == "cycle_overdue":
        line_list.append(f"Decision {alert_dict['decision_plan_id_int']} remained open at least one hour after target session close.")
        if historical_bool:
            line_list.append(f"HISTORICAL: this cycle is now {current_status_str}; verify current state before any action.")
        elif current_status_str is None:
            line_list.append("Current cycle state unavailable; verify current broker state before any action.")
        for row_dict in payload_dict["owned_order_row_list"]:
            line_list.append(f"{row_dict['asset_str']}: quantity={row_dict['quantity_float']:g}; side={row_dict['side_str']}; client_id={row_dict['client_id_int']}")
        if not payload_dict["owned_order_row_list"]:
            line_list.append("No owned open orders in the observed snapshot." if payload_dict["order_details_available_bool"]
                else "Owned open-order details unavailable in this alert; verify current broker state.")
    else:
        line_list.extend([f"Capsule decision HALTED for {payload_dict['signal_date_str']}: {payload_dict['reason_str']}",
            "This is the recorded holding mismatch; verify current account holdings before acting."])
        line_list.extend(f"{row_dict['asset_str']}: holding={row_dict['quantity_float']:g} shares"
            for row_dict in payload_dict["holding_row_list"])
    line_list.append("Required action: " + payload_dict["required_action_str"])
    content_str = "\n".join(line_list)
    return [{"content": content_str[offset_int:offset_int + 1900], "allowed_mentions": {"parse": []}}
        for offset_int in range(0, len(content_str), 1900)]


@dataclass(frozen=True)
class DailyAlertDelivery:
    pod_id_str: str
    mode_str: str
    alert_key_str: str
    alert_kind_str: str
    delivered_bool: bool


def deliver_daily_alerts(summary_dict, *, webhook_url_str, webhook_poster_fn, mode_str=None, now_ts=None,
        suppressed_session_key_set=None):
    if not webhook_url_str:
        return []
    delivery_list = []
    for scope_tuple in _configured_scope_list(summary_dict, mode_str):
        db_path_str, pod_id_str, account_route_str, configured_mode_str = scope_tuple
        for alert_dict in _pending_alert_list(scope_tuple):
            if alert_dict["alert_kind_str"] not in ALERT_KIND_SET:
                raise ValueError("Unknown daily alert kind.")
            claim_ts = now_ts or datetime.now(timezone.utc)
            claim_str = uuid4().hex
            identity_tuple = (alert_dict["alert_key_str"], pod_id_str, account_route_str, configured_mode_str)
            identity_sql_str = "alert_key_str=? AND pod_id_str=? AND account_route_str=? AND mode_str=?"
            db_uri_str = Path(db_path_str).as_uri() + "?mode=rw"
            with closing(sqlite3.connect(db_uri_str, uri=True, timeout=5.0)) as connection_obj, connection_obj:
                connection_obj.execute("BEGIN IMMEDIATE")
                cursor_obj = connection_obj.execute(f"""UPDATE daily_pod_alert SET delivery_claim_str=?,
                    delivery_claimed_timestamp_str=?,attempt_count_int=attempt_count_int+1 WHERE {identity_sql_str}
                    AND delivered_timestamp_str IS NULL AND (delivery_claim_str IS NULL OR delivery_claimed_timestamp_str<=?)""",
                    (claim_str, _timestamp_str(claim_ts), *identity_tuple,
                        _timestamp_str(claim_ts - timedelta(seconds=DELIVERY_LEASE_SECONDS_INT))))
                claimed_bool = cursor_obj.rowcount == 1
                current_status_str = None
                if claimed_bool and alert_dict["decision_plan_id_int"] is not None:
                    row_obj = connection_obj.execute("""SELECT status_str FROM decision_plan WHERE decision_plan_id_int=?
                        AND release_id_str=? AND pod_id_str=? AND account_route_str=?""",
                        (alert_dict["decision_plan_id_int"], alert_dict["release_id_str"], pod_id_str, account_route_str)).fetchone()
                    current_status_str = None if row_obj is None else row_obj[0]
            if not claimed_bool:
                continue
            delivered_bool = False
            try:
                payload_body_dict = json.loads(alert_dict["payload_json_str"])
                session_str = (payload_body_dict.get("session_close_timestamp_str", "")[:10]
                    if alert_dict["alert_kind_str"] == "cycle_overdue" else
                    payload_body_dict.get("signal_date_str", "") if alert_dict["alert_kind_str"] == "capsule_holding_halt" else "")
                session_key_str = "|".join((configured_mode_str, pod_id_str, session_str))
                delivered_bool = session_key_str in (suppressed_session_key_set or set())
                if not delivered_bool:
                    delivered_bool = all(bool(webhook_poster_fn(webhook_url_str, payload_dict))
                        for payload_dict in _payload_list(alert_dict, current_status_str))
            finally:
                with closing(sqlite3.connect(db_uri_str, uri=True, timeout=5.0)) as connection_obj, connection_obj:
                    connection_obj.execute(f"""UPDATE daily_pod_alert SET delivered_timestamp_str=?,
                        delivery_claim_str=NULL,delivery_claimed_timestamp_str=NULL WHERE {identity_sql_str} AND delivery_claim_str=?""",
                        (_timestamp_str(now_ts or datetime.now(timezone.utc)) if delivered_bool else None, *identity_tuple, claim_str))
            delivery_list.append(DailyAlertDelivery(pod_id_str, configured_mode_str,
                alert_dict["alert_key_str"], alert_dict["alert_kind_str"], delivered_bool))
    return delivery_list
