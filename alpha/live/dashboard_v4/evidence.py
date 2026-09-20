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


class _EvidenceError(ValueError):
    """Static operator-safe reason; never include raw DB values or exceptions."""


def _timestamp_ts(value_obj: Any, *, as_of_ts: datetime) -> datetime:
    timestamp_ts = parse_timestamp_ts(value_obj)
    if timestamp_ts is None or timestamp_ts > as_of_ts:
        raise _EvidenceError("A saved timestamp is missing or in the future.")
    return timestamp_ts


def _quantity_float(value_obj: Any) -> float:
    if isinstance(value_obj, bool):
        raise _EvidenceError("A saved quantity is invalid.")
    quantity_float = float(value_obj)
    if not math.isfinite(quantity_float):
        raise _EvidenceError("A saved quantity is invalid.")
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
            raise _EvidenceError("The saved Pod or account does not match.")
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
                raise _EvidenceError("The saved decision, plan or release is missing.")
            for saved_obj in (vplan_obj, decision_obj, saved_release_obj):
                if any(saved_obj[field_str] != value_str for field_str, value_str in identity_dict.items()):
                    raise _EvidenceError("The saved cycle identity does not match.")
            if saved_release_obj["mode_str"] != "live" or vplan_obj["decision_plan_id_int"] != decision_id_int:
                raise _EvidenceError("The saved plan does not match this LIVE decision.")
            if vplan_obj["status_str"] != pod_row_dict.get("latest_vplan_status_str"):
                raise _EvidenceError("The saved plan changed during this check. Refresh to check again.")
            for saved_obj in (vplan_obj, decision_obj):
                _timestamp_ts(saved_obj["created_timestamp_str"], as_of_ts=as_of_ts)
                _timestamp_ts(saved_obj["updated_timestamp_str"], as_of_ts=as_of_ts)
            plan_row_list = connection_obj.execute(
                "SELECT * FROM vplan_row WHERE vplan_id_int = ? ORDER BY asset_str, vplan_row_id_int", (vplan_id_int,),
            ).fetchall()
            order_row_list = connection_obj.execute("SELECT * FROM vplan_broker_order WHERE vplan_id_int = ?", (vplan_id_int,)).fetchall()
            fill_row_list = connection_obj.execute("SELECT * FROM vplan_fill WHERE vplan_id_int = ?", (vplan_id_int,)).fetchall()
            ack_row_list = connection_obj.execute("SELECT * FROM vplan_broker_ack WHERE vplan_id_int = ?", (vplan_id_int,)).fetchall()
        plan_dict = dict(vplan_obj)
        result_dict.update(vplan_status_str=plan_dict["status_str"], fill_record_count_int=len(fill_row_list))
        delta_map_dict = json.loads(plan_dict["order_delta_json_str"])
        if not isinstance(delta_map_dict, dict):
            raise _EvidenceError("The saved order plan is invalid.")
        delta_map_dict = {asset_str: _quantity_float(value_obj) for asset_str, value_obj in delta_map_dict.items()}
        row_delta_map_dict: dict[str, float] = {}
        for plan_row_obj in plan_row_list:
            asset_str = plan_row_obj["asset_str"]
            row_delta_map_dict[asset_str] = row_delta_map_dict.get(asset_str, 0.0) + _quantity_float(plan_row_obj["order_delta_share_float"])
        if any(abs(row_delta_map_dict.get(asset_str, 0.0) - delta_map_dict.get(asset_str, 0.0)) > QUANTITY_TOLERANCE_FLOAT
               for asset_str in set(row_delta_map_dict) | set(delta_map_dict)):
            raise _EvidenceError("Saved order quantities do not match the plan.")
        for field_str, actual_count_int in (("broker_order_count_int", len(order_row_list)), ("fill_count_int", len(fill_row_list))):
            if pod_row_dict.get(field_str) is not None and int(pod_row_dict[field_str]) != actual_count_int:
                raise _EvidenceError("Saved orders or fills changed during this check. Refresh to check again.")
        # The production builder owns request keys, row ordering and zero-share
        # filtering. This snapshot contains only its persisted input fields.
        plan_snapshot_obj = SimpleNamespace(**plan_dict, vplan_row_list=[SimpleNamespace(**dict(plan_row_obj)) for plan_row_obj in plan_row_list])
        request_list = build_broker_order_request_list_from_vplan(plan_snapshot_obj)
        result_dict.update(vplan_status_str=plan_dict["status_str"], order_count_int=len(request_list))
        if not request_list:
            ack_clear_bool = (
                not ack_row_list and not pod_row_dict.get("missing_ack_count_int")
                and pod_row_dict.get("latest_submit_ack_status_str") != "missing_critical"
                and plan_dict["missing_ack_count_int"] == 0
                and plan_dict["submit_ack_status_str"] != "missing_critical"
            )
            if ack_clear_bool and not order_row_list and not fill_row_list and all(abs(delta_float) <= QUANTITY_TOLERANCE_FLOAT for delta_float in delta_map_dict.values()):
                result_dict.update(state_str="no_orders", reason_str="No orders needed")
                return result_dict
            raise _EvidenceError("Orders or fills were saved for a plan with no orders.")
        request_map_dict = {request_obj.order_request_key_str: request_obj for request_obj in request_list}
        order_map_dict: dict[str, dict[str, Any]] = {}
        request_key_set: set[str] = set()
        for order_row_obj in order_row_list:
            order_dict = dict(order_row_obj)
            order_id_str = str(order_dict["broker_order_id_str"] or "")
            request_key_str = str(order_dict["order_request_key_str"] or "")
            request_obj = request_map_dict.get(request_key_str)
            if not order_id_str or order_id_str in order_map_dict or request_key_str in request_key_set or request_obj is None:
                raise _EvidenceError("A saved order cannot be matched uniquely to the plan.")
            if (order_dict["decision_plan_id_int"] != decision_id_int or order_dict["account_route_str"] != release_obj.account_route_str
                    or order_dict["asset_str"] != request_obj.asset_str or order_dict["unit_str"] != "shares"
                    or order_dict["broker_order_type_str"] != request_obj.broker_order_type_str
                    or order_dict["submission_key_str"] != request_obj.submission_key_str):
                raise _EvidenceError("A saved order does not match its planned account, symbol or type.")
            amount_float = _quantity_float(order_dict["amount_float"])
            reported_float = _quantity_float(order_dict["filled_amount_float"])
            sparse_terminal_bool = (
                plan_dict["status_str"] == "completed" and order_dict["status_str"] == "Filled"
                and amount_float == 0 and reported_float == 0
                and order_dict["remaining_amount_float"] is not None
                and _quantity_float(order_dict["remaining_amount_float"]) == 0
            )
            if abs(amount_float - request_obj.amount_float) > QUANTITY_TOLERANCE_FLOAT and not sparse_terminal_bool:
                raise _EvidenceError("A saved order quantity differs from the planned quantity.")
            if reported_float < 0 or reported_float > abs(amount_float) + QUANTITY_TOLERANCE_FLOAT:
                raise _EvidenceError("The broker's saved filled quantity is invalid.")
            if order_dict["remaining_amount_float"] is not None:
                remaining_float = _quantity_float(order_dict["remaining_amount_float"])
                if remaining_float < 0 or remaining_float > abs(amount_float) + QUANTITY_TOLERANCE_FLOAT:
                    raise _EvidenceError("The broker's saved remaining quantity is invalid.")
            order_dict["submitted_ts"] = _timestamp_ts(order_dict["submitted_timestamp_str"], as_of_ts=as_of_ts)
            if order_dict["last_status_timestamp_str"]:
                _timestamp_ts(order_dict["last_status_timestamp_str"], as_of_ts=as_of_ts)
            order_dict.update(
                requested_share_float=request_obj.amount_float, sparse_terminal_bool=sparse_terminal_bool,
                filled_share_float=0.0, fill_timestamp_list=[],
            )
            order_map_dict[order_id_str] = order_dict
            request_key_set.add(request_key_str)
        if request_key_set != set(request_map_dict):
            raise _EvidenceError("A planned order has no matching saved broker order.")
        if any(order_dict["sparse_terminal_bool"] for order_dict in order_map_dict.values()):
            # Terminal broker refreshes can save zero/zero/zero quantities.
            # Only exact ACK identities plus complete, identifiable executions
            # may prove those orders against the immutable canonical request.
            planned_submission_ts = _timestamp_ts(plan_dict["submission_timestamp_str"], as_of_ts=as_of_ts)
            ack_request_set: set[str] = set()
            ack_order_set: set[str] = set()
            for ack_row_obj in ack_row_list:
                order_id_str = str(ack_row_obj["broker_order_id_str"] or "")
                request_key_str = ack_row_obj["order_request_key_str"]
                order_dict = order_map_dict.get(order_id_str)
                if (order_dict is None or request_key_str in ack_request_set or order_id_str in ack_order_set
                        or request_key_str != order_dict["order_request_key_str"]
                        or ack_row_obj["decision_plan_id_int"] != decision_id_int
                        or ack_row_obj["account_route_str"] != release_obj.account_route_str
                        or ack_row_obj["asset_str"] != order_dict["asset_str"]
                        or ack_row_obj["broker_order_type_str"] != order_dict["broker_order_type_str"]
                        or ack_row_obj["broker_response_ack_bool"] != 1 or ack_row_obj["ack_status_str"] != "broker_acked"):
                    raise _EvidenceError("A saved broker acknowledgement does not match the planned order.")
                if _timestamp_ts(ack_row_obj["response_timestamp_str"], as_of_ts=as_of_ts) < max(order_dict["submitted_ts"], planned_submission_ts):
                    raise _EvidenceError("A saved broker acknowledgement is earlier than the order.")
                ack_request_set.add(request_key_str)
                ack_order_set.add(order_id_str)
            if ack_request_set != request_key_set:
                raise _EvidenceError("The broker order has no quantity and its acknowledgement could not be verified.")
        execution_id_set: set[str] = set()
        fill_key_set: set[tuple] = set()
        for fill_row_obj in fill_row_list:
            fill_dict = dict(fill_row_obj)
            order_dict = order_map_dict.get(str(fill_dict["broker_order_id_str"]))
            if order_dict is None or fill_dict["decision_plan_id_int"] != decision_id_int or fill_dict["account_route_str"] != release_obj.account_route_str or fill_dict["asset_str"] != order_dict["asset_str"]:
                raise _EvidenceError("A fill cannot be matched to its order, account or symbol.")
            fill_ts = _timestamp_ts(fill_dict["fill_timestamp_str"], as_of_ts=as_of_ts)
            if fill_ts < order_dict["submitted_ts"]:
                raise _EvidenceError("A fill time is earlier than its saved order time.")
            fill_float = _quantity_float(fill_dict["fill_amount_float"])
            if fill_float * order_dict["requested_share_float"] <= 0:
                raise _EvidenceError("A fill's buy or sell direction differs from the order.")
            payload_dict = json.loads(fill_dict["raw_payload_json_str"])
            execution_id_str = str(payload_dict.get("exec_id_str") or "")
            if order_dict["sparse_terminal_bool"] and (not isinstance(payload_dict.get("exec_id_str"), str) or not execution_id_str.strip()):
                raise _EvidenceError("A saved fill has no execution ID to verify the broker's empty order summary.")
            price_float = _quantity_float(fill_dict["fill_price_float"])
            if price_float <= 0:
                raise _EvidenceError("A saved fill price is invalid.")
            fill_key_tuple = (fill_dict["broker_order_id_str"], fill_ts, fill_float, price_float)
            if (execution_id_str and execution_id_str in execution_id_set) or fill_key_tuple in fill_key_set:
                raise _EvidenceError("A fill appears more than once in the saved records.")
            execution_id_set.add(execution_id_str)
            fill_key_set.add(fill_key_tuple)
            order_dict["filled_share_float"] += fill_float
            order_dict["fill_timestamp_list"].append(fill_ts)
        order_list = []
        fill_timestamp_list = []
        for order_id_str, order_dict in order_map_dict.items():
            requested_float, filled_float = order_dict["requested_share_float"], order_dict["filled_share_float"]
            if abs(filled_float) > abs(requested_float) + QUANTITY_TOLERANCE_FLOAT:
                raise _EvidenceError("Saved filled shares exceed the ordered quantity.")
            complete_bool = abs(filled_float - requested_float) <= QUANTITY_TOLERANCE_FLOAT
            if order_dict["sparse_terminal_bool"] and not complete_bool:
                raise _EvidenceError("Saved executions do not fully cover the broker's empty order summary.")
            order_list.append({"broker_order_id_str": order_id_str, "asset_str": order_dict["asset_str"], "requested_share_float": requested_float, "filled_share_float": filled_float, "complete_bool": complete_bool})
            fill_timestamp_list.extend(order_dict["fill_timestamp_list"])
        filled_count_int = sum(order_dict["complete_bool"] for order_dict in order_list)
        complete_bool = filled_count_int == len(request_list)
        result_dict.update(
            state_str="complete" if complete_bool else "partial", reason_str="All orders filled" if complete_bool else "Orders not fully filled",
            filled_order_count_int=filled_count_int, fill_record_count_int=len(fill_row_list), order_list=order_list,
            actual_fill_timestamp_str=max(fill_timestamp_list).isoformat() if complete_bool and fill_timestamp_list else None,
        )
    except (AttributeError, IndexError, KeyError, TypeError, ValueError, OverflowError, OSError, sqlite3.Error) as error_obj:
        # Evidence is optional display data. Corrupt, legacy or concurrently
        # changed sources must never turn a cycle green or break Overview.
        reason_str = str(error_obj) if isinstance(error_obj, _EvidenceError) else "Saved fill details could not be read or checked."
        result_dict.update(state_str="unknown", reason_str=reason_str, order_list=[], filled_order_count_int=0, actual_fill_timestamp_str=None)
    return result_dict
