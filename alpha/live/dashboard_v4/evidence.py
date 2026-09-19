"""Read saved LIVE order quantities without opening a state-store writer."""

from __future__ import annotations

import json
import math
import sqlite3
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from alpha.live.execution_engine import build_broker_order_request_list_from_vplan
from alpha.live.ops_report import parse_timestamp_ts


QUANTITY_TOLERANCE_FLOAT = 1e-9  # Same share tolerance as the execution engine.


def _timestamp_ts(value_obj: Any, *, as_of_ts: datetime) -> datetime:
    timestamp_ts = parse_timestamp_ts(value_obj)
    if timestamp_ts is None or timestamp_ts > as_of_ts:
        raise ValueError("Missing or future evidence time")
    return timestamp_ts


def _quantity_float(value_obj: Any) -> float:
    if isinstance(value_obj, bool):
        raise ValueError("Invalid quantity")
    quantity_float = float(value_obj)
    if not math.isfinite(quantity_float):
        raise ValueError("Invalid quantity")
    return quantity_float


def load_cycle_evidence_dict(target_obj, pod_row_dict: dict[str, Any], *, as_of_ts: datetime) -> dict[str, Any]:
    """Prove fills per request, never by net asset amount or record count.

    For each canonical request i, signed filled shares are F_i = sum(fill_j).
    Completion requires abs(F_i - requested_i) <= 1e-9 for EVERY request,
    with each execution on the requested side and one unambiguous order ID.
    A reconciled position may pass a different tolerance; it is not fill proof.
    """
    result_dict: dict[str, Any] = {
        "state_str": "unknown", "reason_str": "Fill evidence unavailable",
        "pod_id_str": pod_row_dict.get("pod_id_str"),
        "account_route_str": pod_row_dict.get("account_route_str"),
        "vplan_id_int": pod_row_dict.get("latest_vplan_id_int"),
        "decision_plan_id_int": pod_row_dict.get("latest_vplan_decision_plan_id_int"),
        "vplan_status_str": None, "order_count_int": 0, "filled_order_count_int": 0,
        "fill_record_count_int": 0, "actual_fill_timestamp_str": None, "order_list": [],
    }
    try:
        release_obj = target_obj.release_obj
        if release_obj.mode_str != "live" or pod_row_dict.get("mode_str") != "live":
            return result_dict
        identity_dict = {
            "release_id_str": release_obj.release_id_str,
            "user_id_str": release_obj.user_id_str,
            "pod_id_str": release_obj.pod_id_str,
            "account_route_str": release_obj.account_route_str,
        }
        if any(pod_row_dict.get(field_str) != identity_dict[field_str] for field_str in ("pod_id_str", "account_route_str")):
            return result_dict
        vplan_id_int = int(result_dict["vplan_id_int"])
        decision_id_int = int(result_dict["decision_plan_id_int"])
        if vplan_id_int <= 0 or decision_id_int <= 0:
            return result_dict
        as_of_ts = as_of_ts.replace(tzinfo=UTC) if as_of_ts.tzinfo is None else as_of_ts.astimezone(UTC)
        db_path_obj = Path(target_obj.db_path_str).resolve()
        # URI mode=ro cannot create/migrate a missing DB. BEGIN pins all evidence
        # to one SQLite read snapshot; closing releases it without a write.
        with closing(sqlite3.connect(f"{db_path_obj.as_uri()}?mode=ro", uri=True, timeout=1.0)) as connection_obj:
            connection_obj.row_factory = sqlite3.Row
            connection_obj.execute("BEGIN")
            vplan_obj = connection_obj.execute("SELECT * FROM vplan WHERE vplan_id_int = ?", (vplan_id_int,)).fetchone()
            decision_obj = connection_obj.execute("SELECT * FROM decision_plan WHERE decision_plan_id_int = ?", (decision_id_int,)).fetchone()
            saved_release_obj = connection_obj.execute("SELECT * FROM live_release WHERE release_id_str = ?", (release_obj.release_id_str,)).fetchone()
            if vplan_obj is None or decision_obj is None or saved_release_obj is None:
                return result_dict
            for saved_obj in (vplan_obj, decision_obj, saved_release_obj):
                if any(saved_obj[field_str] != value_str for field_str, value_str in identity_dict.items()):
                    return result_dict
            if saved_release_obj["mode_str"] != "live" or vplan_obj["decision_plan_id_int"] != decision_id_int:
                return result_dict
            if vplan_obj["status_str"] != pod_row_dict.get("latest_vplan_status_str"):
                return result_dict
            for saved_obj in (vplan_obj, decision_obj):
                _timestamp_ts(saved_obj["created_timestamp_str"], as_of_ts=as_of_ts)
                _timestamp_ts(saved_obj["updated_timestamp_str"], as_of_ts=as_of_ts)
            plan_row_list = connection_obj.execute(
                "SELECT * FROM vplan_row WHERE vplan_id_int = ? ORDER BY asset_str, vplan_row_id_int", (vplan_id_int,),
            ).fetchall()
            order_row_list = connection_obj.execute("SELECT * FROM vplan_broker_order WHERE vplan_id_int = ?", (vplan_id_int,)).fetchall()
            fill_row_list = connection_obj.execute("SELECT * FROM vplan_fill WHERE vplan_id_int = ?", (vplan_id_int,)).fetchall()
            ack_count_int = connection_obj.execute("SELECT COUNT(*) FROM vplan_broker_ack WHERE vplan_id_int = ?", (vplan_id_int,)).fetchone()[0]
        plan_dict = dict(vplan_obj)
        delta_map_dict = json.loads(plan_dict["order_delta_json_str"])
        if not isinstance(delta_map_dict, dict):
            raise ValueError("Invalid order intent")
        delta_map_dict = {asset_str: _quantity_float(value_obj) for asset_str, value_obj in delta_map_dict.items()}
        row_delta_map_dict: dict[str, float] = {}
        for plan_row_obj in plan_row_list:
            asset_str = plan_row_obj["asset_str"]
            row_delta_map_dict[asset_str] = row_delta_map_dict.get(asset_str, 0.0) + _quantity_float(plan_row_obj["order_delta_share_float"])
        if any(abs(row_delta_map_dict.get(asset_str, 0.0) - delta_map_dict.get(asset_str, 0.0)) > QUANTITY_TOLERANCE_FLOAT
               for asset_str in set(row_delta_map_dict) | set(delta_map_dict)):
            raise ValueError("Incomplete or conflicting plan rows")
        for field_str, actual_count_int in (("broker_order_count_int", len(order_row_list)), ("fill_count_int", len(fill_row_list))):
            if pod_row_dict.get(field_str) is not None and int(pod_row_dict[field_str]) != actual_count_int:
                raise ValueError("Evidence changed since summary")
        # The production builder owns request keys, row ordering and zero-share
        # filtering. This snapshot contains only its persisted input fields.
        plan_snapshot_obj = SimpleNamespace(**plan_dict, vplan_row_list=[SimpleNamespace(**dict(plan_row_obj)) for plan_row_obj in plan_row_list])
        request_list = build_broker_order_request_list_from_vplan(plan_snapshot_obj)
        result_dict.update(vplan_status_str=plan_dict["status_str"], order_count_int=len(request_list))
        if not request_list:
            ack_clear_bool = (
                not ack_count_int and not pod_row_dict.get("missing_ack_count_int")
                and pod_row_dict.get("latest_submit_ack_status_str") != "missing_critical"
                and plan_dict["missing_ack_count_int"] == 0
                and plan_dict["submit_ack_status_str"] != "missing_critical"
            )
            if ack_clear_bool and not order_row_list and not fill_row_list and all(abs(delta_float) <= QUANTITY_TOLERANCE_FLOAT for delta_float in delta_map_dict.values()):
                result_dict.update(state_str="no_orders", reason_str="No orders needed")
            return result_dict
        request_map_dict = {request_obj.order_request_key_str: request_obj for request_obj in request_list}
        order_map_dict: dict[str, dict[str, Any]] = {}
        request_key_set: set[str] = set()
        for order_row_obj in order_row_list:
            order_dict = dict(order_row_obj)
            order_id_str = str(order_dict["broker_order_id_str"] or "")
            request_key_str = str(order_dict["order_request_key_str"] or "")
            request_obj = request_map_dict.get(request_key_str)
            if not order_id_str or order_id_str in order_map_dict or request_key_str in request_key_set or request_obj is None:
                raise ValueError("Ambiguous order identity")
            if (order_dict["decision_plan_id_int"] != decision_id_int or order_dict["account_route_str"] != release_obj.account_route_str
                    or order_dict["asset_str"] != request_obj.asset_str or order_dict["unit_str"] != "shares"
                    or order_dict["broker_order_type_str"] != request_obj.broker_order_type_str
                    or order_dict["submission_key_str"] != request_obj.submission_key_str):
                raise ValueError("Order identity mismatch")
            amount_float = _quantity_float(order_dict["amount_float"])
            if abs(amount_float - request_obj.amount_float) > QUANTITY_TOLERANCE_FLOAT:
                raise ValueError("Requested quantity mismatch")
            reported_float = _quantity_float(order_dict["filled_amount_float"])
            if reported_float < 0 or reported_float > abs(amount_float) + QUANTITY_TOLERANCE_FLOAT:
                raise ValueError("Invalid broker filled quantity")
            if order_dict["remaining_amount_float"] is not None:
                remaining_float = _quantity_float(order_dict["remaining_amount_float"])
                if remaining_float < 0 or remaining_float > abs(amount_float) + QUANTITY_TOLERANCE_FLOAT:
                    raise ValueError("Invalid broker remaining quantity")
            order_dict["submitted_ts"] = _timestamp_ts(order_dict["submitted_timestamp_str"], as_of_ts=as_of_ts)
            if order_dict["last_status_timestamp_str"]:
                _timestamp_ts(order_dict["last_status_timestamp_str"], as_of_ts=as_of_ts)
            order_dict.update(filled_share_float=0.0, fill_timestamp_list=[])
            order_map_dict[order_id_str] = order_dict
            request_key_set.add(request_key_str)
        if request_key_set != set(request_map_dict):
            return result_dict
        execution_id_set: set[str] = set()
        fill_key_set: set[tuple] = set()
        for fill_row_obj in fill_row_list:
            fill_dict = dict(fill_row_obj)
            order_dict = order_map_dict.get(str(fill_dict["broker_order_id_str"]))
            if order_dict is None or fill_dict["decision_plan_id_int"] != decision_id_int or fill_dict["account_route_str"] != release_obj.account_route_str or fill_dict["asset_str"] != order_dict["asset_str"]:
                raise ValueError("Fill identity mismatch")
            fill_ts = _timestamp_ts(fill_dict["fill_timestamp_str"], as_of_ts=as_of_ts)
            if fill_ts < order_dict["submitted_ts"]:
                raise ValueError("Fill predates order")
            fill_float = _quantity_float(fill_dict["fill_amount_float"])
            if fill_float * order_dict["amount_float"] <= 0:
                raise ValueError("Wrong fill side")
            payload_dict = json.loads(fill_dict["raw_payload_json_str"])
            execution_id_str = str(payload_dict.get("exec_id_str") or "")
            price_float = _quantity_float(fill_dict["fill_price_float"])
            if price_float <= 0:
                raise ValueError("Invalid execution price")
            fill_key_tuple = (fill_dict["broker_order_id_str"], fill_ts, fill_float, price_float)
            if (execution_id_str and execution_id_str in execution_id_set) or fill_key_tuple in fill_key_set:
                raise ValueError("Repeated execution evidence")
            execution_id_set.add(execution_id_str)
            fill_key_set.add(fill_key_tuple)
            order_dict["filled_share_float"] += fill_float
            order_dict["fill_timestamp_list"].append(fill_ts)
        order_list = []
        fill_timestamp_list = []
        for order_id_str, order_dict in order_map_dict.items():
            requested_float, filled_float = float(order_dict["amount_float"]), order_dict["filled_share_float"]
            if abs(filled_float) > abs(requested_float) + QUANTITY_TOLERANCE_FLOAT:
                raise ValueError("Fill exceeds request")
            complete_bool = abs(filled_float - requested_float) <= QUANTITY_TOLERANCE_FLOAT
            order_list.append({"broker_order_id_str": order_id_str, "asset_str": order_dict["asset_str"], "requested_share_float": requested_float, "filled_share_float": filled_float, "complete_bool": complete_bool})
            fill_timestamp_list.extend(order_dict["fill_timestamp_list"])
        filled_count_int = sum(order_dict["complete_bool"] for order_dict in order_list)
        complete_bool = filled_count_int == len(request_list)
        result_dict.update(
            state_str="complete" if complete_bool else "partial", reason_str="All orders filled" if complete_bool else "Orders not fully filled",
            filled_order_count_int=filled_count_int, fill_record_count_int=len(fill_row_list), order_list=order_list,
            actual_fill_timestamp_str=max(fill_timestamp_list).isoformat() if complete_bool and fill_timestamp_list else None,
        )
    except (AttributeError, IndexError, KeyError, TypeError, ValueError, OverflowError, OSError, sqlite3.Error):
        # Evidence is optional display data. Corrupt, legacy or concurrently
        # changed sources must never turn a cycle green or break Overview.
        result_dict.update(state_str="unknown", reason_str="Fill evidence not verified", order_list=[], filled_order_count_int=0, actual_fill_timestamp_str=None)
    return result_dict
