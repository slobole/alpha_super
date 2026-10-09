"""Bounded, read-only execution activity for one LIVE Pod's current ET day."""

from contextlib import closing
from datetime import datetime, time, timedelta
import json
import math
from pathlib import Path
import sqlite3
from types import SimpleNamespace
from zoneinfo import ZoneInfo

from alpha.live.dashboard_v4.evidence import QUANTITY_TOLERANCE_FLOAT, _quantity_float
from alpha.live.dashboard_v4.positions_data import (
    IDENTITY_FIELD_TUPLE, MAP_BYTE_LIMIT_INT, RELEASE_LIMIT_INT, _position_map_dict,
    observed_timestamp_ts,
)
from alpha.live.execution_engine import build_broker_order_request_list_from_vplan


MARKET_TIMEZONE_OBJ = ZoneInfo("America/New_York")
CYCLE_LIMIT_INT = 64
ROW_LIMIT_INT = 2000
PAYLOAD_BYTE_LIMIT_INT = 16384
PENDING_STATUS_SET = {"PendingSubmit", "PreSubmitted", "Submitted", "PendingCancel"}
CANCELLED_STATUS_SET = {"Cancelled", "ApiCancelled"}
ACTIVITY_COLUMN_DICT = {
    "live_release": (*IDENTITY_FIELD_TUPLE, "mode_str"),
    "decision_plan": (*IDENTITY_FIELD_TUPLE, "decision_plan_id_int", "created_timestamp_str", "updated_timestamp_str"),
    "vplan": (*IDENTITY_FIELD_TUPLE, "vplan_id_int", "decision_plan_id_int", "submission_key_str", "submission_timestamp_str",
        "target_execution_timestamp_str", "broker_snapshot_timestamp_str", "pod_budget_float", "order_delta_json_str",
        "current_broker_position_json_str", "status_str", "created_timestamp_str", "updated_timestamp_str"),
    "vplan_row": ("vplan_id_int", "vplan_row_id_int", "asset_str", "order_delta_share_float", "live_reference_price_float", "broker_order_type_str"),
    "vplan_broker_order": ("vplan_id_int", "broker_order_id_str", "decision_plan_id_int", "account_route_str", "asset_str",
        "broker_order_type_str", "unit_str", "amount_float", "filled_amount_float", "remaining_amount_float", "status_str",
        "last_status_timestamp_str", "submitted_timestamp_str", "submission_key_str", "order_request_key_str"),
    "vplan_broker_ack": ("vplan_id_int", "broker_order_id_str", "decision_plan_id_int", "account_route_str", "asset_str",
        "broker_order_type_str", "order_request_key_str", "broker_response_ack_bool", "ack_status_str", "response_timestamp_str"),
    "vplan_fill": ("vplan_id_int", "broker_order_id_str", "decision_plan_id_int", "account_route_str", "asset_str",
        "fill_timestamp_str", "fill_amount_float", "fill_price_float", "raw_payload_json_str"),
    "vplan_reconciliation_snapshot": ("vplan_id_int", "pod_id_str", "decision_plan_id_int", "stage_str",
        "status_str", "broker_position_json_str", "created_timestamp_str"),
}


def _rows_list(connection_obj, table_str, column_tuple, where_str, parameter_tuple, *, limit_int=None, order_str=""):
    # Table/column/predicate strings are code constants, never request input.
    limit_int = ROW_LIMIT_INT if limit_int is None else limit_int
    projection_list = []
    oversized_list = []
    for column_str in column_tuple:
        if column_str.endswith("_str"):
            cap_int = PAYLOAD_BYTE_LIMIT_INT if column_str == "raw_payload_json_str" else MAP_BYTE_LIMIT_INT if column_str.endswith("_json_str") else 512
            projection_list.append(f"CASE WHEN length(CAST({column_str} AS BLOB))<={cap_int} THEN {column_str} END AS {column_str}")
            oversized_list.append(f"coalesce(length(CAST({column_str} AS BLOB))>{cap_int},0)")
        else:
            projection_list.append(column_str)
    projection_list.append("(" + (" OR ".join(oversized_list) or "0") + ") AS oversized_bool")
    row_list = connection_obj.execute("SELECT " + ",".join(projection_list) + " FROM " + table_str
        + " WHERE " + where_str + (" ORDER BY " + order_str if order_str else "") + " LIMIT ?",
        (*parameter_tuple, limit_int + 1)).fetchall()
    if len(row_list) > limit_int:
        raise ValueError("Activity exceeds the bounded read")
    if any(row_obj["oversized_bool"] for row_obj in row_list):
        raise ValueError("Saved activity field exceeds the bounded read")
    return [dict(row_obj) for row_obj in row_list]


def _planned_timestamp_ts(timestamp_str):
    timestamp_ts = datetime.fromisoformat(timestamp_str)
    if timestamp_ts.tzinfo is None:
        raise ValueError("Missing planned timezone")
    return timestamp_ts


def _unique_dict(pair_list):
    result_dict = dict(pair_list)
    if len(result_dict) != len(pair_list):
        raise ValueError("Duplicate saved field")
    return result_dict


def _cycle_dict(connection_obj, plan_id_int, identity_dict, release_id_set, day_start_ts, day_end_ts, as_of_ts, execution_id_set):
    plan_list = _rows_list(connection_obj, "vplan", ACTIVITY_COLUMN_DICT["vplan"],
        "vplan_id_int=?", (plan_id_int,), limit_int=1)
    if len(plan_list) != 1:
        raise ValueError("Orphan execution")
    plan_dict = plan_list[0]
    decision_list = _rows_list(connection_obj, "decision_plan", ACTIVITY_COLUMN_DICT["decision_plan"],
        "decision_plan_id_int=?", (plan_dict["decision_plan_id_int"],), limit_int=1)
    if (len(decision_list) != 1 or plan_dict["release_id_str"] not in release_id_set
            or any(plan_dict[field_str] != identity_dict[field_str] for field_str in IDENTITY_FIELD_TUPLE[1:])
            or any(decision_list[0][field_str] != plan_dict[field_str] for field_str in IDENTITY_FIELD_TUPLE)):
        raise ValueError("Unverified cycle owner")
    for record_dict in (plan_dict, decision_list[0]):
        for field_str in ("created_timestamp_str", "updated_timestamp_str"):
            observed_timestamp_ts(record_dict[field_str], as_of_ts)
    target_ts = _planned_timestamp_ts(plan_dict["target_execution_timestamp_str"])
    submission_ts = _planned_timestamp_ts(plan_dict["submission_timestamp_str"])
    baseline_ts = observed_timestamp_ts(plan_dict["broker_snapshot_timestamp_str"], as_of_ts)
    before_map_dict = _position_map_dict(plan_dict["current_broker_position_json_str"])
    delta_map_dict = _position_map_dict(plan_dict["order_delta_json_str"])
    plan_row_list = _rows_list(connection_obj, "vplan_row", ACTIVITY_COLUMN_DICT["vplan_row"],
        "vplan_id_int=?", (plan_id_int,), order_str="asset_str,vplan_row_id_int")
    row_delta_dict = {}
    for row_dict in plan_row_list:
        symbol_str = row_dict["asset_str"]
        if not isinstance(symbol_str, str) or not symbol_str.strip() or symbol_str != symbol_str.strip():
            raise ValueError("Invalid planned symbol")
        row_delta_dict[symbol_str] = row_delta_dict.get(symbol_str, 0.0) + _quantity_float(row_dict["order_delta_share_float"])
    if any(abs(row_delta_dict.get(symbol_str, 0) - delta_map_dict.get(symbol_str, 0)) > QUANTITY_TOLERANCE_FLOAT
            for symbol_str in set(row_delta_dict) | set(delta_map_dict)):
        raise ValueError("Planned quantities differ")
    request_list = build_broker_order_request_list_from_vplan(SimpleNamespace(**plan_dict,
        vplan_row_list=[SimpleNamespace(**row_dict) for row_dict in plan_row_list]))
    request_map_dict = {request_obj.order_request_key_str: request_obj for request_obj in request_list}
    order_list = _rows_list(connection_obj, "vplan_broker_order", ACTIVITY_COLUMN_DICT["vplan_broker_order"], "vplan_id_int=?", (plan_id_int,))
    ack_list = _rows_list(connection_obj, "vplan_broker_ack", ACTIVITY_COLUMN_DICT["vplan_broker_ack"], "vplan_id_int=?", (plan_id_int,))
    fill_list = _rows_list(connection_obj, "vplan_fill", ACTIVITY_COLUMN_DICT["vplan_fill"], "vplan_id_int=?", (plan_id_int,))
    order_map_dict, request_order_dict = {}, {}
    for order_dict in order_list:
        order_id_str, request_key_str = order_dict["broker_order_id_str"], order_dict["order_request_key_str"]
        request_obj = request_map_dict.get(request_key_str)
        if (not order_id_str or order_id_str in order_map_dict or request_key_str in request_order_dict or request_obj is None
                or order_dict["decision_plan_id_int"] != plan_dict["decision_plan_id_int"]
                or order_dict["account_route_str"] != identity_dict["account_route_str"] or order_dict["asset_str"] != request_obj.asset_str
                or order_dict["unit_str"] != "shares" or order_dict["broker_order_type_str"] != request_obj.broker_order_type_str
                or order_dict["submission_key_str"] != request_obj.submission_key_str):
            raise ValueError("Unverified broker order")
        amount_float = _quantity_float(order_dict["amount_float"])
        filled_float = _quantity_float(order_dict["filled_amount_float"])
        remaining_float = None if order_dict["remaining_amount_float"] is None else _quantity_float(order_dict["remaining_amount_float"])
        sparse_bool = (plan_dict["status_str"] == "completed" and order_dict["status_str"] == "Filled"
            and amount_float == filled_float == 0 and remaining_float == 0)
        if ((not sparse_bool and abs(amount_float - request_obj.amount_float) > QUANTITY_TOLERANCE_FLOAT)
                or not 0 <= filled_float <= abs(amount_float) + QUANTITY_TOLERANCE_FLOAT
                or (remaining_float is not None and not 0 <= remaining_float <= abs(amount_float) + QUANTITY_TOLERANCE_FLOAT)):
            raise ValueError("Unverified order amount")
        order_dict.update(submitted_ts=observed_timestamp_ts(order_dict["submitted_timestamp_str"], as_of_ts),
            sparse_bool=sparse_bool, verified_filled_float=0.0)
        if order_dict["last_status_timestamp_str"]:
            observed_timestamp_ts(order_dict["last_status_timestamp_str"], as_of_ts)
        order_map_dict[order_id_str] = order_dict
        request_order_dict[request_key_str] = order_dict
    sparse_cycle_bool = any(order_dict["sparse_bool"] for order_dict in order_list)
    ack_key_set = set()
    for ack_dict in ack_list:
        request_key_str = ack_dict["order_request_key_str"]
        order_dict = request_order_dict.get(request_key_str)
        if (request_key_str in ack_key_set or order_dict is None or ack_dict["broker_order_id_str"] != order_dict["broker_order_id_str"]
                or any(ack_dict[field_str] != order_dict[field_str] for field_str in
                    ("decision_plan_id_int", "account_route_str", "asset_str", "broker_order_type_str"))):
            raise ValueError("Unverified acknowledgement identity")
        if ack_dict["response_timestamp_str"]:
            ack_ts = observed_timestamp_ts(ack_dict["response_timestamp_str"], as_of_ts)
        else:
            ack_ts = None
        if sparse_cycle_bool and (ack_dict["broker_response_ack_bool"] != 1 or ack_dict["ack_status_str"] != "broker_acked"
                or ack_ts is None or ack_ts < max(order_dict["submitted_ts"], submission_ts)):
            raise ValueError("Sparse order lacks verified acknowledgement")
        ack_key_set.add(request_key_str)
    if sparse_cycle_bool and ack_key_set != set(request_map_dict):
        raise ValueError("Sparse orders lack complete acknowledgements")
    today_fill_list, fill_key_set = [], set()
    for fill_dict in fill_list:
        order_dict = order_map_dict.get(fill_dict["broker_order_id_str"])
        if order_dict is None or any(fill_dict[field_str] != order_dict[field_str] for field_str in ("decision_plan_id_int", "account_route_str", "asset_str")):
            raise ValueError("Orphan or foreign fill")
        fill_ts = observed_timestamp_ts(fill_dict["fill_timestamp_str"], as_of_ts)
        amount_float, price_float = _quantity_float(fill_dict["fill_amount_float"]), _quantity_float(fill_dict["fill_price_float"])
        request_obj = request_map_dict[order_dict["order_request_key_str"]]
        payload_dict = json.loads(fill_dict["raw_payload_json_str"], object_pairs_hook=_unique_dict)
        execution_id_str = payload_dict.get("exec_id_str")
        fill_key_tuple = (fill_dict["broker_order_id_str"], fill_ts, amount_float, price_float)
        if (fill_ts < order_dict["submitted_ts"] or amount_float * request_obj.amount_float <= 0 or price_float <= 0
                or not isinstance(execution_id_str, str) or not execution_id_str.strip()
                or execution_id_str in execution_id_set or fill_key_tuple in fill_key_set):
            raise ValueError("Unverified or duplicate execution")
        execution_id_set.add(execution_id_str)
        fill_key_set.add(fill_key_tuple)
        order_dict["verified_filled_float"] += amount_float
        fill_dict.update(timestamp_ts=fill_ts, amount_float=amount_float)
        if day_start_ts <= fill_ts < day_end_ts:
            today_fill_list.append(fill_dict)
    for request_key_str, order_dict in request_order_dict.items():
        requested_float = request_map_dict[request_key_str].amount_float
        filled_float = order_dict["verified_filled_float"]
        if abs(filled_float) > abs(requested_float) + QUANTITY_TOLERANCE_FLOAT or (order_dict["sparse_bool"]
                and abs(filled_float - requested_float) > QUANTITY_TOLERANCE_FLOAT) or (
                _quantity_float(order_dict["filled_amount_float"]) > abs(filled_float) + QUANTITY_TOLERANCE_FLOAT) or (
                order_dict["status_str"] == "Filled" and abs(filled_float - requested_float) > QUANTITY_TOLERANCE_FLOAT):
            raise ValueError("Execution quantity differs from order")
    # A same-day broker baseline and a passed post-execution observation may
    # prove endpoints. Target shares and a newer unrelated cache are not proof.
    after_map_dict, after_ts = None, None
    if today_fill_list and len(today_fill_list) == len(fill_list) and day_start_ts <= baseline_ts <= min(row_dict["timestamp_ts"] for row_dict in today_fill_list):
        reconcile_list = _rows_list(connection_obj, "vplan_reconciliation_snapshot", ACTIVITY_COLUMN_DICT["vplan_reconciliation_snapshot"],
            "vplan_id_int=? AND stage_str='post_execution'", (plan_id_int,))
        for reconcile_dict in reconcile_list:
            if reconcile_dict["pod_id_str"] != identity_dict["pod_id_str"] or reconcile_dict["decision_plan_id_int"] != plan_dict["decision_plan_id_int"]:
                raise ValueError("Unverified reconciliation owner")
            reconcile_dict["timestamp_ts"] = observed_timestamp_ts(reconcile_dict["created_timestamp_str"], as_of_ts)
        if reconcile_list:
            last_ts = max(row_dict["timestamp_ts"] for row_dict in reconcile_list)
            last_list = [row_dict for row_dict in reconcile_list if row_dict["timestamp_ts"] == last_ts]
            if len(last_list) == 1 and last_list[0]["status_str"] == "passed" and last_ts >= max(row_dict["timestamp_ts"] for row_dict in today_fill_list):
                candidate_dict = _position_map_dict(last_list[0]["broker_position_json_str"])
                total_fill_dict = {}
                for fill_dict in fill_list:
                    symbol_str = fill_dict["asset_str"]
                    total_fill_dict[symbol_str] = total_fill_dict.get(symbol_str, 0.0) + fill_dict["amount_float"]
                if all(abs(before_map_dict.get(symbol_str, 0) + total_fill_dict.get(symbol_str, 0) - candidate_dict.get(symbol_str, 0)) <= QUANTITY_TOLERANCE_FLOAT
                        for symbol_str in set(before_map_dict) | set(candidate_dict) | set(total_fill_dict)):
                    after_map_dict, after_ts = candidate_dict, last_ts
    return {"plan_dict": plan_dict, "request_map_dict": request_map_dict, "request_order_dict": request_order_dict,
        "today_fill_list": today_fill_list, "today_plan_bool": day_start_ts <= target_ts < day_end_ts,
        "before_map_dict": before_map_dict if after_map_dict is not None else None, "after_map_dict": after_map_dict,
        "before_ts": baseline_ts, "after_ts": after_ts}


def load_position_activity_dict(target_obj, *, as_of_ts):
    """Today means execution timestamp in ET, including older-target cycles.

    delta = sum(signed fills); bought/sold are separate positive gross totals.
    Any actual fill marks a symbol changed, including a zero-net round trip.
    Pending intent never proves change. Before/after and New/Closed additionally
    require coherent same-day broker endpoints around all verified executions.
    """
    result_dict = {"available_bool": False, "reason_str": "Today's activity could not be verified",
        "symbol_dict": {}, "assessed_timestamp_str": "", "market_date_str": ""}
    try:
        if as_of_ts.tzinfo is None:
            return result_dict
        day_start_ts = datetime.combine(as_of_ts.astimezone(MARKET_TIMEZONE_OBJ).date(), time(), MARKET_TIMEZONE_OBJ)
        day_end_ts = day_start_ts + timedelta(days=1)
        result_dict.update(assessed_timestamp_str=as_of_ts.isoformat(), market_date_str=day_start_ts.date().isoformat())
        release_obj = target_obj.release_obj
        if release_obj.mode_str != "live" or release_obj.enabled_bool is not True:
            return result_dict
        identity_dict = {field_str: getattr(release_obj, field_str) for field_str in IDENTITY_FIELD_TUPLE}
        if any(not isinstance(value_str, str) or not value_str.strip() for value_str in identity_dict.values()):
            return result_dict
        with closing(sqlite3.connect(Path(target_obj.db_path_str).resolve().as_uri() + "?mode=ro", uri=True, timeout=.2)) as connection_obj:
            connection_obj.row_factory = sqlite3.Row
            connection_obj.execute("PRAGMA query_only=ON")
            progress_list = [0]

            def stop_large_read_bool():
                progress_list[0] += 1
                return progress_list[0] > 3000

            connection_obj.set_progress_handler(stop_large_read_bool, 1000)
            connection_obj.execute("BEGIN")
            # Even an empty day needs the complete evidence schema; a missing
            # order/fill field must not become a positive "No activity" claim.
            for table_str, column_tuple in ACTIVITY_COLUMN_DICT.items():
                connection_obj.execute("SELECT " + ",".join(column_tuple) + " FROM " + table_str + " LIMIT 0")
            release_list = _rows_list(connection_obj, "live_release", ACTIVITY_COLUMN_DICT["live_release"],
                "pod_id_str=?", (release_obj.pod_id_str,), limit_int=RELEASE_LIMIT_INT)
            if not release_list or any(row_dict["mode_str"] != "live" or any(row_dict[field_str] != identity_dict[field_str]
                    for field_str in IDENTITY_FIELD_TUPLE[1:]) for row_dict in release_list):
                raise ValueError("Conflicting saved owner")
            release_id_set = {row_dict["release_id_str"] for row_dict in release_list}
            if release_obj.release_id_str not in release_id_set or len(release_id_set) != len(release_list):
                raise ValueError("Current release is unverified")
            # Include malformed/future records so neither can disappear behind
            # a SQL date filter. The one-second margin protects sub-ms midnight.
            fill_scope_str = "(account_route_str=? OR vplan_id_int IN (SELECT vplan_id_int FROM vplan WHERE pod_id_str=?))"
            recent_fill_list = _rows_list(connection_obj, "vplan_fill", ("vplan_id_int", "fill_timestamp_str"), fill_scope_str
                + " AND (julianday(fill_timestamp_str) IS NULL OR julianday(fill_timestamp_str)>=julianday(?))",
                (release_obj.account_route_str, release_obj.pod_id_str, (day_start_ts - timedelta(seconds=1)).isoformat()))
            candidate_set = set()
            for fill_dict in recent_fill_list:
                fill_ts = observed_timestamp_ts(fill_dict["fill_timestamp_str"], as_of_ts)
                if fill_ts >= day_start_ts:
                    candidate_set.add(fill_dict["vplan_id_int"])
            plan_list = _rows_list(connection_obj, "vplan", ("vplan_id_int", "target_execution_timestamp_str"),
                "pod_id_str=? AND (julianday(target_execution_timestamp_str) IS NULL OR "
                "(julianday(target_execution_timestamp_str)>=julianday(?) AND julianday(target_execution_timestamp_str)<=julianday(?)))",
                (release_obj.pod_id_str, (day_start_ts - timedelta(seconds=1)).isoformat(), (day_end_ts + timedelta(seconds=1)).isoformat()), limit_int=CYCLE_LIMIT_INT)
            for plan_dict in plan_list:
                if day_start_ts <= _planned_timestamp_ts(plan_dict["target_execution_timestamp_str"]) < day_end_ts:
                    candidate_set.add(plan_dict["vplan_id_int"])
            if len(candidate_set) > CYCLE_LIMIT_INT:
                raise ValueError("Too many cycles")
            cycle_list, execution_id_set = [], set()
            for plan_id_int in sorted(candidate_set):
                cycle_list.append(_cycle_dict(connection_obj, plan_id_int, identity_dict, release_id_set, day_start_ts, day_end_ts, as_of_ts, execution_id_set))
        symbol_dict, endpoint_dict = {}, {}
        for cycle_dict in cycle_list:
            plan_dict = cycle_dict["plan_dict"]
            filled_symbol_set = {row_dict["asset_str"] for row_dict in cycle_dict["today_fill_list"]}
            for request_key_str, request_obj in cycle_dict["request_map_dict"].items():
                symbol_str = request_obj.asset_str
                if not cycle_dict["today_plan_bool"] and symbol_str not in filled_symbol_set:
                    continue
                item_dict = symbol_dict.setdefault(symbol_str, {"filled_delta_float": 0.0, "bought_float": 0.0, "sold_float": 0.0,
                    "before_float": None, "after_float": None, "new_bool": False, "closed_bool": False,
                    "status_str": "Filled", "changed_bool": False, "detail_str": "", "cycle_key_str": "",
                    "observed_timestamp_str": "", "status_set": set(), "source_key_tuple": None})
                order_dict = cycle_dict["request_order_dict"].get(request_key_str)
                source_ts = observed_timestamp_ts(plan_dict["updated_timestamp_str"], as_of_ts)
                if order_dict is not None and order_dict["last_status_timestamp_str"]:
                    source_ts = max(source_ts, observed_timestamp_ts(order_dict["last_status_timestamp_str"], as_of_ts))
                source_ts = max([source_ts] + [row_dict["timestamp_ts"] for row_dict in cycle_dict["today_fill_list"] if row_dict["asset_str"] == symbol_str])
                source_key_tuple = (source_ts, plan_dict["vplan_id_int"])
                if item_dict["source_key_tuple"] is None or source_key_tuple > item_dict["source_key_tuple"]:
                    item_dict.update(source_key_tuple=source_key_tuple, cycle_key_str="vplan:" + str(plan_dict["vplan_id_int"]),
                        observed_timestamp_str=source_ts.isoformat())
                if order_dict is None:
                    status_str = ("Buy pending" if request_obj.amount_float > 0 else "Sell pending") if plan_dict["status_str"] == "ready" else "Unknown"
                elif abs(order_dict["verified_filled_float"] - request_obj.amount_float) <= QUANTITY_TOLERANCE_FLOAT:
                    status_str = "Filled"
                elif order_dict["status_str"] in CANCELLED_STATUS_SET:
                    status_str = "Cancelled"
                elif order_dict["status_str"] in PENDING_STATUS_SET:
                    status_str = "Partial" if order_dict["verified_filled_float"] else "Buy pending" if request_obj.amount_float > 0 else "Sell pending"
                else:
                    status_str = "Unknown"
                item_dict["status_set"].add(status_str)
            for fill_dict in cycle_dict["today_fill_list"]:
                item_dict = symbol_dict[fill_dict["asset_str"]]
                amount_float = fill_dict["amount_float"]
                item_dict["filled_delta_float"] += amount_float
                item_dict["bought_float"] += max(amount_float, 0)
                item_dict["sold_float"] += max(-amount_float, 0)
                item_dict["changed_bool"] = True
                timestamp_str = fill_dict["timestamp_ts"].isoformat()
                if not item_dict["observed_timestamp_str"] or fill_dict["timestamp_ts"] > datetime.fromisoformat(item_dict["observed_timestamp_str"]):
                    item_dict["observed_timestamp_str"] = timestamp_str
            for symbol_str in filled_symbol_set:
                endpoint_dict.setdefault(symbol_str, []).append(cycle_dict)
        for symbol_str, item_dict in symbol_dict.items():
            status_set = item_dict.pop("status_set")
            item_dict.pop("source_key_tuple")
            if "Unknown" in status_set or {"Buy pending", "Sell pending"}.issubset(status_set):
                item_dict["status_str"] = "Unknown"
            else:
                item_dict["status_str"] = next((status_str for status_str in ("Partial", "Buy pending", "Sell pending", "Cancelled", "Filled") if status_str in status_set), "Unknown")
            if not all(math.isfinite(item_dict[field_str]) for field_str in ("filled_delta_float", "bought_float", "sold_float")):
                raise ValueError("Activity quantity overflow")
            endpoint_list = sorted(endpoint_dict.get(symbol_str, []), key=lambda cycle_dict: cycle_dict["before_ts"])
            if endpoint_list and all(cycle_dict["after_map_dict"] is not None for cycle_dict in endpoint_list):
                coherent_bool = all(previous_dict["after_ts"] <= next_dict["before_ts"]
                    and abs(previous_dict["after_map_dict"].get(symbol_str, 0) - next_dict["before_map_dict"].get(symbol_str, 0)) <= QUANTITY_TOLERANCE_FLOAT
                    for previous_dict, next_dict in zip(endpoint_list, endpoint_list[1:]))
                if coherent_bool:
                    before_float = endpoint_list[0]["before_map_dict"].get(symbol_str, 0.0)
                    after_float = endpoint_list[-1]["after_map_dict"].get(symbol_str, 0.0)
                    item_dict.update(before_float=before_float, after_float=after_float,
                        new_bool=abs(before_float) <= QUANTITY_TOLERANCE_FLOAT < abs(after_float),
                        closed_bool=abs(after_float) <= QUANTITY_TOLERANCE_FLOAT < abs(before_float))
            item_dict["detail_str"] = "Verified fills today" if item_dict["changed_bool"] else "No verified fill today"
        result_dict.update(available_bool=True, reason_str="", symbol_dict=symbol_dict)
    except (AttributeError, IndexError, KeyError, TypeError, ValueError, OverflowError, ArithmeticError, RecursionError, OSError, sqlite3.Error):
        pass
    return result_dict
