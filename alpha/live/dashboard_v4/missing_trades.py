"""Read-only "Missing trades" for the V4 Pod page: plan target minus saved broker position.

Generic for monthly and daily pods. Reads only the pod's own SQLite state (read-only)
and a bounded tail of the event log. Never contacts the broker: the order ticket's
preview does the fresh broker read (alpha.live.dashboard_v4.orders).
"""

from __future__ import annotations

from contextlib import closing
from datetime import datetime
import json
import math
from pathlib import Path
import sqlite3

import pandas as pd

from alpha.live import scheduler_utils
from alpha.live.dashboard_authorization import _ReadOnlyConfirmationStore


ELIGIBLE_VPLAN_STATUS_SET = frozenset({"submitted", "completed"})
EVENT_TAIL_BYTES_INT = 2 * 1024 * 1024
COMPLETED_EVENT_STR = "manual_order_submit_completed"
CANCELLED_STATUS_SET = frozenset({"cancelled", "apicancelled", "inactive", "rejected"})
TYPE_LABEL_DICT = {"MOC": "MOC", "MKT": "MKT", "LMT": "LMT"}


def _hidden_dict(reason_str=""):
    return {"status_str": "hidden", "reason_str": reason_str, "row_list": []}


def _session_label_ts(timestamp_ts, calendar_id_str):
    calendar_obj = scheduler_utils.get_exchange_calendar_obj(calendar_id_str)
    date_ts = pd.Timestamp(scheduler_utils.to_market_timestamp_ts(timestamp_ts, calendar_id_str).date())
    return calendar_obj.date_to_session(date_ts, direction="previous")


def plan_fresh_bool(target_execution_ts, as_of_ts, calendar_id_str) -> bool:
    """The plan's session is today's (latest) session or the one before it."""
    calendar_obj = scheduler_utils.get_exchange_calendar_obj(calendar_id_str)
    current_label_ts = _session_label_ts(as_of_ts, calendar_id_str)
    previous_label_ts = calendar_obj.previous_session(current_label_ts)
    return _session_label_ts(target_execution_ts, calendar_id_str) >= previous_label_ts


def read_completion_event_list(log_path_str, pod_id_str, *, tail_bytes_int=EVENT_TAIL_BYTES_INT):
    """Recent manual-order completions for this pod, from a bounded log tail."""
    path_obj = Path(log_path_str)
    if not path_obj.is_file():
        return []
    with path_obj.open("rb") as file_obj:
        file_obj.seek(0, 2)
        size_int = file_obj.tell()
        file_obj.seek(max(0, size_int - tail_bytes_int))
        raw_bytes = file_obj.read(tail_bytes_int)
    event_list = []
    for line_bytes in raw_bytes.splitlines():
        if COMPLETED_EVENT_STR.encode() not in line_bytes:
            continue
        try:
            record_dict = json.loads(line_bytes)
        except (ValueError, UnicodeDecodeError):
            continue  # The first line of a tail can be partial.
        if (not isinstance(record_dict, dict) or record_dict.get("event_name_str") != COMPLETED_EVENT_STR
                or record_dict.get("pod_id_str") != pod_id_str):
            continue
        event_list.append(record_dict)
    return event_list


def _why_map(db_path_str, vplan_id_int):
    """One short reason per asset from this plan's saved broker evidence."""
    uri_str = Path(db_path_str).resolve().as_uri() + "?mode=ro"
    with closing(sqlite3.connect(uri_str, uri=True)) as connection_obj:
        connection_obj.row_factory = sqlite3.Row
        table_set = {row_obj[0] for row_obj in connection_obj.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        order_list = list(connection_obj.execute("""SELECT asset_str, amount_float, filled_amount_float, status_str
            FROM vplan_broker_order WHERE vplan_id_int=?""", (vplan_id_int,))) if "vplan_broker_order" in table_set else []
        ack_set = {row_obj["asset_str"] for row_obj in connection_obj.execute("""SELECT asset_str FROM vplan_broker_ack
            WHERE vplan_id_int=? AND broker_response_ack_bool=1""", (vplan_id_int,))} if "vplan_broker_ack" in table_set else set()
    why_dict = {}
    for row_obj in order_list:
        asset_str = row_obj["asset_str"]
        amount_float = abs(float(row_obj["amount_float"] or 0))
        filled_float = abs(float(row_obj["filled_amount_float"] or 0))
        status_str = str(row_obj["status_str"] or "").lower()
        if 0 < filled_float < amount_float:
            why_dict[asset_str] = f"partly filled {_share_str(filled_float)} of {_share_str(amount_float)}"
        elif filled_float == 0 and status_str in CANCELLED_STATUS_SET:
            why_dict[asset_str] = "broker cancelled"
        elif filled_float == 0:
            why_dict[asset_str] = "not filled"
    return why_dict, ack_set


def _share_str(value_float):
    return str(int(value_float)) if float(value_float).is_integer() else f"{value_float:g}"


def load_missing_trades_dict(target_obj, *, as_of_ts: datetime, shared_account_bool: bool,
                             completion_event_list: list[dict]) -> dict:
    """Rows for the Missing trades panel.

    status_str: "ok" (rows may be empty), "hidden" (nothing to show, with a reason
    for tests), or "unknown" (saved evidence could not be read). Rule set: handoff §6.
    """
    release_obj = target_obj.release_obj
    if shared_account_bool:
        return _hidden_dict("Account shared with another pod")
    try:
        store_obj = _ReadOnlyConfirmationStore(target_obj.db_path_str)
        try:
            vplan_obj = store_obj.get_latest_vplan_for_pod(release_obj.pod_id_str)
            decision_obj = store_obj.get_latest_decision_plan_for_pod(release_obj.pod_id_str)
            snapshot_obj = store_obj.get_latest_broker_snapshot_for_account(release_obj.account_route_str)
        finally:
            store_obj.connection_obj.close()
    except (sqlite3.Error, OSError, ValueError, TypeError, KeyError) as exception_obj:
        return {"status_str": "unknown", "reason_str": f"Saved state unreadable: {type(exception_obj).__name__}", "row_list": []}
    if vplan_obj is None:
        return _hidden_dict("No order plan")
    if vplan_obj.account_route_str != release_obj.account_route_str or vplan_obj.pod_id_str != release_obj.pod_id_str:
        return {"status_str": "unknown", "reason_str": "Plan identity does not match the pod", "row_list": []}
    if vplan_obj.status_str not in ELIGIBLE_VPLAN_STATUS_SET:
        return _hidden_dict("Plan not sent")
    if decision_obj is not None and decision_obj.decision_plan_id_int != vplan_obj.decision_plan_id_int:
        return _hidden_dict("A newer decision supersedes this plan")
    if snapshot_obj is None or snapshot_obj.snapshot_timestamp_ts <= vplan_obj.target_execution_timestamp_ts:
        return _hidden_dict("Waiting for broker positions after the plan's execution time")
    fresh_bool = plan_fresh_bool(vplan_obj.target_execution_timestamp_ts, as_of_ts, release_obj.session_calendar_id_str)
    try:
        why_dict, ack_set = _why_map(target_obj.db_path_str, vplan_obj.vplan_id_int)
    except (sqlite3.Error, OSError, ValueError, TypeError) as exception_obj:
        return {"status_str": "unknown", "reason_str": f"Saved orders unreadable: {type(exception_obj).__name__}", "row_list": []}
    today_label_ts = _session_label_ts(as_of_ts, release_obj.session_calendar_id_str)
    event_by_asset_dict = {}
    for event_dict in completion_event_list:
        if event_dict.get("completion_vplan_id_int") != vplan_obj.vplan_id_int:
            continue
        try:
            event_ts = datetime.fromisoformat(str(event_dict.get("event_timestamp_str")))
        except ValueError:
            continue
        event_by_asset_dict[event_dict.get("asset_str")] = {**event_dict, "event_ts": event_ts}
    row_list = []
    for asset_str, target_obj_float in sorted(vplan_obj.target_share_map.items()):
        target_float = float(target_obj_float)
        if not math.isfinite(target_float) or target_float < 0:
            continue  # Sentinels and shorts are never pre-filled.
        saved_float = float(snapshot_obj.position_amount_map.get(asset_str, 0.0))
        gap_float = target_float - saved_float
        quantity_int = int(math.floor(abs(gap_float) + 1e-9))
        event_dict = event_by_asset_dict.get(asset_str)
        base_dict = {"asset_str": asset_str, "target_share_float": target_float, "saved_position_float": saved_float}
        if quantity_int < 1:
            if event_dict is not None and _session_label_ts(event_dict["event_ts"], release_obj.session_calendar_id_str) == today_label_ts:
                row_list.append({**base_dict, "state_str": "closed", "side_str": event_dict.get("side_str") or "",
                    "quantity_int": int(event_dict.get("quantity_int") or 0),
                    "text_str": f"Closed by manual {TYPE_LABEL_DICT.get(event_dict.get('broker_order_type_str'), 'order')}"
                                f" · {int(event_dict.get('quantity_int') or 0)}"})
            continue
        side_str = "BUY" if gap_float > 0 else "SELL"
        row_dict = {**base_dict, "side_str": side_str, "quantity_int": quantity_int}
        if event_dict is not None and event_dict["event_ts"] > snapshot_obj.snapshot_timestamp_ts:
            type_str = event_dict.get("broker_order_type_str")
            order_id_str = ", ".join(str(item_obj) for item_obj in event_dict.get("broker_order_id_list") or []) or "—"
            text_str = ("Sent · MOC · waits for close" if type_str == "MOC" else f"Sent · {TYPE_LABEL_DICT.get(type_str, 'order')}") + f" · order {order_id_str}"
            if event_dict.get("submit_ack_status_str") != "complete":
                text_str = f"Sent · broker ACK not confirmed · check TWS · order {order_id_str}"
            row_list.append({**row_dict, "state_str": "sent", "text_str": text_str})
        elif not fresh_bool:
            row_list.append({**row_dict, "state_str": "off", "text_str": "Plan older than 1 session · no pre-fill"})
        else:
            ordered_bool = abs(float(vplan_obj.order_delta_map.get(asset_str, 0.0))) > 1e-9
            if ordered_bool and asset_str not in ack_set and not why_dict.get(asset_str, "").startswith("partly"):
                why_str = "no broker ACK"
            else:
                why_str = why_dict.get(asset_str) or "differs from plan"
            row_list.append({**row_dict, "state_str": "missing", "text_str": "Missing · " + why_str})
    return {"status_str": "ok", "reason_str": "", "row_list": row_list,
        "vplan_id_int": vplan_obj.vplan_id_int, "fresh_bool": fresh_bool,
        "target_share_map": {asset_str: float(value_obj) for asset_str, value_obj in vplan_obj.target_share_map.items()},
        "positions_timestamp_ts": snapshot_obj.snapshot_timestamp_ts}
