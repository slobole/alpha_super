"""Durable capsule execution alerts, delivered by the OPS watchdog.

Enqueue joins the caller's transaction. Delivery is at least once: a crash
between Discord accepting a message and recording success can cause a retry.
"""
from __future__ import annotations

from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sqlite3
from typing import Any, Callable
from uuid import uuid4


ALERT_KIND_SET = frozenset({"accepted_residual", "unresolved_execution", "late_execution", "dispatch_failed"})
DELIVERY_LEASE_SECONDS_INT = 300


def ensure_execution_alert_schema(connection_obj: sqlite3.Connection) -> None:
    # executescript would commit the caller's completion transaction.
    connection_obj.execute("""
        CREATE TABLE IF NOT EXISTS mr_capsule_execution_alert (
            vplan_id_int INTEGER NOT NULL,
            alert_kind_str TEXT NOT NULL,
            pod_id_str TEXT NOT NULL,
            account_route_str TEXT NOT NULL,
            mode_str TEXT NOT NULL,
            payload_json_str TEXT NOT NULL,
            created_timestamp_str TEXT NOT NULL,
            delivery_claim_str TEXT,
            delivery_claimed_timestamp_str TEXT,
            delivered_timestamp_str TEXT,
            attempt_count_int INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (vplan_id_int, alert_kind_str)
        )
    """)


def _utc_timestamp_str(timestamp_ts: datetime) -> str:
    if timestamp_ts.tzinfo is None or timestamp_ts.utcoffset() is None:
        raise ValueError("Capsule alert timestamps must be timezone aware.")
    return timestamp_ts.astimezone(timezone.utc).isoformat()


def enqueue_execution_alert(connection_obj: sqlite3.Connection, *, vplan_id_int: int,
        alert_kind_str: str, pod_id_str: str, account_route_str: str, mode_str: str,
        payload_dict: dict[str, Any], created_timestamp_ts: datetime) -> None:
    if alert_kind_str not in ALERT_KIND_SET or vplan_id_int <= 0:
        raise ValueError("Invalid capsule execution alert identity.")
    if not pod_id_str or not account_route_str or mode_str not in {"live", "paper", "incubation"}:
        raise ValueError("Invalid capsule execution alert route.")
    connection_obj.execute("""
        INSERT OR IGNORE INTO mr_capsule_execution_alert
        (vplan_id_int, alert_kind_str, pod_id_str, account_route_str, mode_str,
         payload_json_str, created_timestamp_str)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (vplan_id_int, alert_kind_str, pod_id_str, account_route_str, mode_str,
          json.dumps(payload_dict, sort_keys=True, allow_nan=False),
          _utc_timestamp_str(created_timestamp_ts)))


@dataclass(frozen=True)
class ExecutionAlertDelivery:
    pod_id_str: str
    mode_str: str
    vplan_id_int: int
    alert_kind_str: str
    delivered_bool: bool


def _configured_scope_list(summary_dict: dict[str, Any], mode_str: str | None) -> list[tuple]:
    # Reuse DashboardApp's configured paths; do not instantiate a mutable store.
    from alpha.live.mr_capsule_adapter import MR_CAPSULE_STRATEGY_IMPORT_TUPLE
    from alpha.live.core5_adapter import CORE5_STRATEGY_IMPORT_STR

    scope_list: list[tuple] = []
    scope_set: set[tuple] = set()
    pod_scope_dict: dict[tuple, tuple] = {}
    account_scope_dict: dict[tuple, tuple] = {}
    for row_dict in summary_dict.get("pod_row_dict_list") or []:
        if row_dict.get("strategy_import_str") not in (*MR_CAPSULE_STRATEGY_IMPORT_TUPLE, CORE5_STRATEGY_IMPORT_STR):
            continue
        if mode_str not in {None, "all"} and row_dict.get("mode_str") != mode_str:
            continue
        identity_tuple = tuple(row_dict.get(field_str) for field_str in
            ("pod_id_str", "account_route_str", "mode_str"))
        db_path_str = row_dict.get("db_path_str")
        if (not isinstance(db_path_str, str) or not db_path_str
                or any(not isinstance(value_str, str) or not value_str for value_str in identity_tuple)
                or identity_tuple[2] not in {"live", "paper", "incubation"}):
            raise ValueError("Invalid configured capsule alert scope.")
        scope_tuple = (str(Path(db_path_str).resolve()), *identity_tuple)
        pod_key_tuple = (identity_tuple[2], identity_tuple[0])
        account_key_tuple = (identity_tuple[2], identity_tuple[1])
        if (pod_scope_dict.get(pod_key_tuple, scope_tuple) != scope_tuple
                or account_scope_dict.get(account_key_tuple, scope_tuple) != scope_tuple):
            raise ValueError("Ambiguous configured capsule alert scope.")
        pod_scope_dict[pod_key_tuple] = scope_tuple
        account_scope_dict[account_key_tuple] = scope_tuple
        if scope_tuple not in scope_set:
            scope_set.add(scope_tuple)
            scope_list.append(scope_tuple)
    return scope_list


def _pending_alert_list(scope_tuple: tuple) -> list[dict[str, Any]]:
    db_path_str, pod_id_str, account_route_str, mode_str = scope_tuple
    db_path_obj = Path(db_path_str)
    if not db_path_obj.is_file():
        return []
    # Old or missing databases stay unchanged; only the state-store migration
    # owns schema creation, never this monitoring reader.
    with closing(sqlite3.connect(db_path_obj.as_uri() + "?mode=ro", uri=True)) as connection_obj:
        connection_obj.row_factory = sqlite3.Row
        if connection_obj.execute("SELECT 1 FROM sqlite_master WHERE type='table' "
                "AND name='mr_capsule_execution_alert'").fetchone() is None:
            return []
        return [dict(row_obj) for row_obj in connection_obj.execute("""
            SELECT * FROM mr_capsule_execution_alert
            WHERE pod_id_str = ? AND account_route_str = ? AND mode_str = ?
              AND delivered_timestamp_str IS NULL
            ORDER BY created_timestamp_str, vplan_id_int, alert_kind_str
        """, (pod_id_str, account_route_str, mode_str))]


def pending_execution_alert_count_int(summary_dict: dict[str, Any], *, mode_str: str | None = None) -> int:
    return sum(len(_pending_alert_list(scope_tuple))
               for scope_tuple in _configured_scope_list(summary_dict, mode_str))


def _current_cycle_context_dict(connection_obj: sqlite3.Connection, alert_dict: dict[str, Any]) -> dict[str, Any] | None:
    # Snapshot this exact cycle while claiming its alert. Preserve the original
    # queued payload; old warnings must not become stale trading instructions.
    required_table_set = {"vplan", "decision_plan", "live_release"}
    table_set = {row_obj[0] for row_obj in connection_obj.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if not required_table_set.issubset(table_set):
        return None
    row_obj = connection_obj.execute("""
        SELECT v.status_str, d.status_str, d.snapshot_metadata_json_str, r.strategy_import_str
        FROM vplan v JOIN decision_plan d ON d.decision_plan_id_int = v.decision_plan_id_int
            AND d.release_id_str = v.release_id_str
        JOIN live_release r ON r.release_id_str = v.release_id_str
        WHERE v.vplan_id_int = ? AND v.pod_id_str = ? AND v.account_route_str = ?
            AND d.pod_id_str = v.pod_id_str AND d.account_route_str = v.account_route_str
            AND r.pod_id_str = v.pod_id_str AND r.account_route_str = v.account_route_str AND r.mode_str = ?
    """, (alert_dict["vplan_id_int"], alert_dict["pod_id_str"], alert_dict["account_route_str"], alert_dict["mode_str"])).fetchone()
    if row_obj is None:
        return None
    from alpha.live.mr_capsule_adapter import MR_CAPSULE_STRATEGY_IMPORT_TUPLE
    from alpha.live.core5_adapter import CORE5_STRATEGY_IMPORT_STR
    if row_obj[3] not in (*MR_CAPSULE_STRATEGY_IMPORT_TUPLE, CORE5_STRATEGY_IMPORT_STR):
        return None
    result_dict = json.loads(row_obj[2]).get("mr_capsule_execution_result_dict")
    return {"completed_bool": row_obj[0] == row_obj[1] == "completed",
        "vplan_status_str": row_obj[0], "decision_status_str": row_obj[1],
        "result_dict": result_dict if isinstance(result_dict, dict) else None,
        "checked_timestamp_str": datetime.now(timezone.utc).isoformat()}


def _build_payload_dict(alert_dict: dict[str, Any]) -> dict[str, Any]:
    payload_dict = json.loads(alert_dict["payload_json_str"])
    description_str = {
        "accepted_residual": "HISTORICAL execution observation: terminal residual accepted. VERIFY current broker holdings before any manual action. Missed buys receive no replacement order.",
        "unresolved_execution": "Execution unresolved; verify broker orders and fills before any manual action.",
        "late_execution": "Late execution recorded separately from the opening auction.",
        "dispatch_failed": "CRITICAL: opening dispatch failed or missed its deadline. Pod parked; review broker orders/fills before resuming.",
    }[alert_dict["alert_kind_str"]]
    current_context_dict = alert_dict.get("current_cycle_context_dict")
    context_line_list = []
    if alert_dict["alert_kind_str"] == "dispatch_failed":
        if current_context_dict and (current_context_dict["completed_bool"]
                or current_context_dict["decision_status_str"] == "superseded"):
            description_str = "HISTORICAL dispatch failure: this cycle is now completed or superseded by reviewed recovery. Verify current state before acting."
        for field_str in ("reason_code_str", "error_type_str", "dispatch_deadline_timestamp_str"):
            if field_str in payload_dict:
                context_line_list.append(f"{field_str}={payload_dict[field_str]}")
    if alert_dict["alert_kind_str"] == "unresolved_execution":
        if current_context_dict and current_context_dict["completed_bool"]:
            description_str = "HISTORICAL alert: this cycle is now recorded completed. No action from this old warning."
            original_asset_list = [row_dict["asset_str"] for row_dict in payload_dict.get("residual_row_dict_list") or []]
            context_line_list.append("Earlier affected symbols: " + ", ".join(original_asset_list))
            payload_dict = {}
        elif current_context_dict and current_context_dict["result_dict"]:
            description_str = "Cycle remains unresolved. Latest recorded details below; verify current broker state before acting."
            payload_dict = current_context_dict["result_dict"]
        else:
            description_str = "HISTORICAL execution observation; current cycle state unavailable. VERIFY the broker before any action."
            payload_dict = {**payload_dict, "residual_row_dict_list": [
                {**row_dict, "required_action_str": "VERIFY", "reason_str": "Historical observation; current state unavailable"}
                for row_dict in payload_dict.get("residual_row_dict_list") or []]}
        if current_context_dict:
            context_line_list.append(f"Stored state checked at {current_context_dict['checked_timestamp_str']}; "
                f"VPlan={current_context_dict['vplan_status_str']}, decision={current_context_dict['decision_status_str']}.")
    line_list = [f"{str(alert_dict['mode_str']).upper()} / {alert_dict['pod_id_str']}: {description_str}",
        f"Account {alert_dict['account_route_str']} | VPlan {alert_dict['vplan_id_int']} | Original observation {alert_dict['created_timestamp_str']}",
        *context_line_list]
    broker_snapshot_dict = payload_dict.get("broker_snapshot_dict") or {}
    for field_str in ("cash_float", "broker_cash_float", "net_liq_float", "available_funds_float",
                      "buying_power_float", "excess_liquidity_float", "cushion_float"):
        value_obj = payload_dict.get(field_str, broker_snapshot_dict.get(field_str))
        if value_obj is not None:
            line_list.append(f"{field_str}={value_obj}")
    for residual_dict in payload_dict.get("residual_row_dict_list") or []:
        residual_float = residual_dict["residual_amount_float"]
        quantity_str = "unknown" if residual_float is None else f"{abs(float(residual_float)):g}"
        action_label_str = "action at observation" if alert_dict["alert_kind_str"] == "accepted_residual" else "action"
        quantity_label_str = "recorded remaining" if alert_dict["alert_kind_str"] == "accepted_residual" else "remaining"
        line_list.append(f"{residual_dict['asset_str']}: {action_label_str}={residual_dict.get('required_action_str', 'VERIFY')}; "
            f"{quantity_label_str}={quantity_str} shares; "
            f"requested={residual_dict['requested_amount_float']}, filled={residual_dict['filled_amount_float']}; "
            f"status={residual_dict['status_str']}; reason={residual_dict.get('reason_str', '')}; "
            f"order={residual_dict.get('broker_order_id_str', '')}; request={residual_dict.get('order_request_key_str', '')}")
    for late_fill_dict in payload_dict.get("late_fill_row_dict_list") or []:
        line_list.append(f"LATE {late_fill_dict['asset_str']}: filled={late_fill_dict['fill_amount_float']} shares; "
            f"price={late_fill_dict['fill_price_float']}; time={late_fill_dict['fill_timestamp_str']}")
    return {"content": "\n".join(line_list), "allowed_mentions": {"parse": []}}


def _build_payload_list(alert_dict: dict[str, Any]) -> list[dict[str, Any]]:
    payload_dict = _build_payload_dict(alert_dict)
    if len(payload_dict["content"]) <= 1900:
        return [payload_dict]
    identity_str = (f"{str(alert_dict['mode_str']).upper()} / {alert_dict['pod_id_str']} | "
        f"Account {alert_dict['account_route_str']} | VPlan {alert_dict['vplan_id_int']} | {alert_dict['alert_kind_str']}")
    body_limit_int = 1900 - len(identity_str) - 40
    if body_limit_int < 100:
        raise ValueError("Capsule alert identity exceeds the message size limit.")
    body_list = []
    current_body_str = ""
    for line_str in payload_dict["content"].splitlines():
        if len(line_str) > body_limit_int:
            if current_body_str:
                body_list.append(current_body_str)
                current_body_str = ""
            # Exceptional oversized rows continue in the next part. Never drop
            # an asset, share quantity, action or explanation to fit the limit.
            while len(line_str) > body_limit_int:
                body_list.append(line_str[:body_limit_int])
                line_str = line_str[body_limit_int:]
        candidate_str = current_body_str + ("\n" if current_body_str else "") + line_str
        if len(candidate_str) > body_limit_int:
            body_list.append(current_body_str)
            current_body_str = line_str
        else:
            current_body_str = candidate_str
    if current_body_str:
        body_list.append(current_body_str)
    return [{"content": f"{identity_str} | part {part_int}/{len(body_list)}\n{body_str}",
             "allowed_mentions": {"parse": []}}
            for part_int, body_str in enumerate(body_list, start=1)]


def deliver_execution_alerts(summary_dict: dict[str, Any], *, webhook_url_str: str,
        webhook_poster_fn: Callable[[str, dict[str, Any]], bool], mode_str: str | None = None,
        now_ts: datetime | None = None) -> list[ExecutionAlertDelivery]:
    if not webhook_url_str:
        return []
    delivery_list: list[ExecutionAlertDelivery] = []
    for scope_tuple in _configured_scope_list(summary_dict, mode_str):
        db_path_str, pod_id_str, account_route_str, configured_mode_str = scope_tuple
        # Freeze candidates: failed delivery gets only one attempt per pass.
        for alert_dict in _pending_alert_list(scope_tuple):
            if alert_dict["alert_kind_str"] not in ALERT_KIND_SET:
                raise ValueError("Unknown capsule execution alert kind.")
            claim_ts = now_ts or datetime.now(timezone.utc)
            claim_timestamp_str = _utc_timestamp_str(claim_ts)
            expired_timestamp_str = _utc_timestamp_str(claim_ts - timedelta(seconds=DELIVERY_LEASE_SECONDS_INT))
            claim_str = uuid4().hex
            identity_tuple = (alert_dict["vplan_id_int"], alert_dict["alert_kind_str"],
                              pod_id_str, account_route_str, configured_mode_str)
            db_uri_str = Path(db_path_str).as_uri() + "?mode=rw"
            with closing(sqlite3.connect(db_uri_str, uri=True, timeout=5.0)) as connection_obj, connection_obj:
                connection_obj.execute("BEGIN IMMEDIATE")
                claim_cursor_obj = connection_obj.execute("""
                    UPDATE mr_capsule_execution_alert
                    SET delivery_claim_str = ?, delivery_claimed_timestamp_str = ?,
                        attempt_count_int = attempt_count_int + 1
                    WHERE vplan_id_int = ? AND alert_kind_str = ? AND pod_id_str = ?
                      AND account_route_str = ? AND mode_str = ? AND delivered_timestamp_str IS NULL
                      AND (delivery_claim_str IS NULL OR delivery_claimed_timestamp_str <= ?)
                """, (claim_str, claim_timestamp_str, *identity_tuple, expired_timestamp_str))
                claimed_bool = claim_cursor_obj.rowcount == 1
                current_context_dict = _current_cycle_context_dict(connection_obj, alert_dict) if claimed_bool else None
            if not claimed_bool:
                continue
            # Existing HTTP transport timeout is 3s, well below the 300s lease.
            delivered_bool = False
            try:
                payload_list = _build_payload_list({**alert_dict, "current_cycle_context_dict": current_context_dict})
                # Success is recorded only after ALL parts succeed. A later retry
                # may repeat earlier parts, consistent with at-least-once delivery.
                delivered_bool = all(bool(webhook_poster_fn(webhook_url_str, payload_dict)) for payload_dict in payload_list)
            finally:
                completed_timestamp_str = _utc_timestamp_str(now_ts or datetime.now(timezone.utc))
                with closing(sqlite3.connect(db_uri_str, uri=True, timeout=5.0)) as connection_obj, connection_obj:
                    connection_obj.execute("""
                        UPDATE mr_capsule_execution_alert
                        SET delivered_timestamp_str = ?, delivery_claim_str = NULL,
                            delivery_claimed_timestamp_str = NULL
                        WHERE vplan_id_int = ? AND alert_kind_str = ? AND pod_id_str = ?
                          AND account_route_str = ? AND mode_str = ? AND delivery_claim_str = ?
                    """, (completed_timestamp_str if delivered_bool else None, *identity_tuple, claim_str))
            delivery_list.append(ExecutionAlertDelivery(pod_id_str, configured_mode_str,
                int(alert_dict["vplan_id_int"]), str(alert_dict["alert_kind_str"]), delivered_bool))
    return delivery_list
