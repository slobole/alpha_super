"""Bounded, identity-scoped saved cycle reads for the LIVE Pod page."""

from __future__ import annotations

import json
import math
import sqlite3
from contextlib import closing
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

from alpha.live import scheduler_utils
from alpha.live.dashboard_v3.filters import MARKET_TIMEZONE_OBJ
from alpha.live.execution_engine import build_broker_order_request_list_from_vplan
from alpha.live.ops_report import parse_timestamp_ts
from alpha.live.runner import DEFAULT_EOD_SNAPSHOT_BUFFER_MINUTES_INT


HISTORY_LIMIT_INT = 60
DETAIL_LIMIT_INT = 2000
IDENTITY_FIELD_TUPLE = ("release_id_str", "user_id_str", "pod_id_str", "account_route_str")
RELEASE_FIELD_TUPLE = IDENTITY_FIELD_TUPLE + (
    "mode_str", "session_calendar_id_str", "data_profile_str", "signal_clock_str", "execution_policy_str",
)
PLAN_FIELD_TUPLE = IDENTITY_FIELD_TUPLE + (
    "decision_plan_id_int", "signal_timestamp_str", "submission_timestamp_str",
    "target_execution_timestamp_str", "execution_policy_str", "status_str",
    "created_timestamp_str", "updated_timestamp_str",
)
CHILD_FIELD_DICT = {
    "vplan_row": ("vplan_row_id_int", "vplan_id_int", "asset_str", "current_share_float", "target_share_float", "order_delta_share_float", "live_reference_price_float", "estimated_target_notional_float", "broker_order_type_str", "live_reference_source_str"),
    "vplan_broker_order": ("broker_order_record_id_int", "broker_order_id_str", "decision_plan_id_int", "vplan_id_int", "account_route_str", "asset_str", "broker_order_type_str", "unit_str", "amount_float", "filled_amount_float", "remaining_amount_float", "avg_fill_price_float", "status_str", "last_status_timestamp_str", "submitted_timestamp_str", "submission_key_str", "order_request_key_str"),
    "vplan_broker_ack": ("broker_order_ack_id_int", "decision_plan_id_int", "vplan_id_int", "account_route_str", "order_request_key_str", "asset_str", "broker_order_type_str", "local_submit_ack_bool", "broker_response_ack_bool", "ack_status_str", "ack_source_str", "broker_order_id_str", "response_timestamp_str"),
    "vplan_fill": ("fill_record_id_int", "broker_order_id_str", "decision_plan_id_int", "vplan_id_int", "account_route_str", "asset_str", "fill_amount_float", "fill_price_float", "official_open_price_float", "open_price_source_str", "fill_timestamp_str"),
    "vplan_broker_order_event": ("broker_order_event_id_int", "broker_order_id_str", "decision_plan_id_int", "vplan_id_int", "account_route_str", "asset_str", "status_str", "filled_amount_float", "remaining_amount_float", "avg_fill_price_float", "event_timestamp_str", "event_source_str", "submission_key_str", "order_request_key_str"),
}


def _allow_dict(row_obj, field_tuple):
    result_dict = {field_str: row_obj[field_str] for field_str in field_tuple if field_str in row_obj.keys()}
    for field_str, value_obj in result_dict.items():
        if field_str.endswith("_float") and value_obj is not None and (
                isinstance(value_obj, bool) or not isinstance(value_obj, (int, float)) or not math.isfinite(value_obj)):
            raise ValueError("Invalid saved amount")
    return result_dict


def _map_dict(value_str):
    value_dict = json.loads(value_str)
    if not isinstance(value_dict, dict):
        raise ValueError("Invalid saved map")
    if any(not isinstance(asset_str, str) or isinstance(amount_obj, bool)
           or not isinstance(amount_obj, (int, float)) or not math.isfinite(amount_obj)
           for asset_str, amount_obj in value_dict.items()):
        raise ValueError("Invalid saved amount")
    return value_dict


def _observed_ts(value_str, as_of_ts):
    value_ts = parse_timestamp_ts(value_str)
    if value_ts is None or value_ts > as_of_ts:
        raise ValueError("Missing or future saved observation")
    return value_ts


def _cycle_dict(decision_obj, vplan_obj, latest_decision_id_int, current_release_id_str):
    timing_obj = vplan_obj if vplan_obj is not None else decision_obj
    target_ts = parse_timestamp_ts(timing_obj["target_execution_timestamp_str"])
    if target_ts is None:
        raise ValueError("Missing execution time")
    decision_id_int = decision_obj["decision_plan_id_int"]
    vplan_id_int = None if vplan_obj is None else vplan_obj["vplan_id_int"]
    return {
        "cycle_key_str": f"vplan:{vplan_id_int}" if vplan_id_int else f"decision:{decision_id_int}",
        "release_id_str": decision_obj["release_id_str"],
        "decision_plan_id_int": decision_id_int, "vplan_id_int": vplan_id_int,
        "signal_timestamp_str": decision_obj["signal_timestamp_str"],
        "submission_timestamp_str": timing_obj["submission_timestamp_str"],
        "target_execution_timestamp_str": timing_obj["target_execution_timestamp_str"],
        "session_date_str": target_ts.astimezone(MARKET_TIMEZONE_OBJ).date().isoformat(),
        "execution_policy_str": timing_obj["execution_policy_str"],
        "decision_status_str": decision_obj["status_str"],
        "vplan_status_str": None if vplan_obj is None else vplan_obj["status_str"],
        "current_bool": decision_id_int == latest_decision_id_int and decision_obj["release_id_str"] == current_release_id_str,
        "unresolved_bool": vplan_obj is not None and (
            vplan_obj["status_str"] in {"ready", "submitting", "submitted", "blocked", "expired"}
            or vplan_obj["missing_ack_count_int"] > 0 or vplan_obj["submit_ack_status_str"] == "missing_critical"),
    }


def _eod_dict(connection_obj, release_obj, target_ts, as_of_ts):
    """An account EOD belongs to its exchange session, never the latest row."""
    session_label_ts = scheduler_utils.session_label_from_timestamp_ts(target_ts, release_obj.session_calendar_id_str)
    if session_label_ts is None:
        return {"status_str": "unknown"}
    close_ts = scheduler_utils.get_session_close_timestamp_ts(session_label_ts, release_obj.session_calendar_id_str)
    due_ts = close_ts + timedelta(minutes=DEFAULT_EOD_SNAPSHOT_BUFFER_MINUTES_INT)
    market_date_str = target_ts.astimezone(MARKET_TIMEZONE_OBJ).date().isoformat()
    start_ts = target_ts.astimezone(MARKET_TIMEZONE_OBJ).replace(hour=0, minute=0, second=0, microsecond=0)
    end_ts = start_ts + timedelta(days=1)
    row_list = connection_obj.execute(
        """SELECT updated_timestamp_str, recorded_timestamp_str FROM pod_state_history
        WHERE pod_id_str=? AND user_id_str=? AND account_route_str=?
          AND snapshot_stage_str='eod' AND snapshot_source_str='broker'
          AND julianday(updated_timestamp_str)>=julianday(?)
          AND julianday(updated_timestamp_str)<julianday(?)
        ORDER BY pod_state_history_id_int DESC LIMIT ?""",
        (release_obj.pod_id_str, release_obj.user_id_str, release_obj.account_route_str,
         start_ts.isoformat(), end_ts.isoformat(), DETAIL_LIMIT_INT + 1),
    ).fetchall()
    if len(row_list) > DETAIL_LIMIT_INT:
        raise ValueError("Too many EOD observations")
    valid_list = []
    for row_obj in row_list:
        observation_ts = _observed_ts(row_obj["updated_timestamp_str"], as_of_ts)
        _observed_ts(row_obj["recorded_timestamp_str"], as_of_ts)
        if observation_ts >= due_ts:
            valid_list.append(observation_ts)
    latest_ts = max(valid_list) if valid_list else None
    return {
        "status_str": "completed" if latest_ts else "waiting" if as_of_ts < due_ts else "due_missing",
        "source_str": "broker", "expected_market_date_str": market_date_str,
        "expected_due_timestamp_str": due_ts.isoformat(), "same_session_bool": latest_ts is not None,
        "latest_timestamp_str": latest_ts.isoformat() if latest_ts else None,
        "latest_market_date_str": market_date_str if latest_ts else None,
    }


def load_pod_cycles_dict(target_obj, *, as_of_ts: datetime, decision_plan_id_int=None, vplan_id_int=None):
    """Read selected evidence in one SQLite snapshot; never call a state writer.

    A cycle is a DecisionPlan and its own optional VPlan. Newer decisions and
    older unresolved plans stay separate. The reader does not use the current
    position cache, Norgate files, unscoped log rows, or arbitrary file paths.
    """
    result_dict = {
        "status_str": "unknown", "reason_str": "Saved cycle evidence unavailable",
        "cycle_list": [], "history_truncated_bool": False, "selected_cycle_dict": None,
        "pod_row_dict": {}, "decision_dict": {}, "vplan_dict": {}, "plan_row_list": [],
        "order_list": [], "ack_list": [], "fill_list": [], "reconciliation_dict": {},
        "event_list": [], "eod_dict": {}, "file_list": [], "selected_release_dict": {},
    }
    try:
        release_obj = target_obj.release_obj
        if release_obj.mode_str != "live":
            return result_dict
        for selector_obj in (decision_plan_id_int, vplan_id_int):
            if selector_obj is not None and (type(selector_obj) is not int or selector_obj <= 0):
                raise ValueError("Invalid cycle selector")
        as_of_ts = as_of_ts.replace(tzinfo=UTC) if as_of_ts.tzinfo is None else as_of_ts.astimezone(UTC)
        identity_tuple = tuple(getattr(release_obj, field_str) for field_str in IDENTITY_FIELD_TUPLE)
        scope_str = " AND ".join(f"{field_str}=?" for field_str in IDENTITY_FIELD_TUPLE)
        owner_tuple = identity_tuple[1:]
        owner_scope_str = " AND ".join(f"plan.{field_str}=?" for field_str in IDENTITY_FIELD_TUPLE[1:])
        release_join_str = " AND ".join(f"release.{field_str}=plan.{field_str}" for field_str in IDENTITY_FIELD_TUPLE)
        select_sql_dict = {table_str: f"SELECT plan.* FROM {table_str} AS plan JOIN live_release AS release ON {release_join_str} WHERE {owner_scope_str} AND release.mode_str='live'"
                           for table_str in ("decision_plan", "vplan")}
        db_path_obj = Path(target_obj.db_path_str).resolve()
        with closing(sqlite3.connect(f"{db_path_obj.as_uri()}?mode=ro", uri=True, timeout=1.0)) as connection_obj:
            connection_obj.row_factory = sqlite3.Row
            connection_obj.execute("BEGIN")
            saved_release_obj = connection_obj.execute(
                f"SELECT * FROM live_release WHERE {scope_str} AND mode_str='live'", identity_tuple,
            ).fetchone()
            if saved_release_obj is None:
                raise ValueError("Release identity mismatch")
            decision_list = connection_obj.execute(
                select_sql_dict["decision_plan"] + " ORDER BY plan.decision_plan_id_int DESC LIMIT ?",
                (*owner_tuple, HISTORY_LIMIT_INT + 1),
            ).fetchall()
            if not decision_list:
                result_dict.update(status_str="not_found" if decision_plan_id_int or vplan_id_int else "empty", reason_str="No saved cycles")
                return result_dict
            latest_decision_id_int = decision_list[0]["decision_plan_id_int"]
            result_dict["history_truncated_bool"] = len(decision_list) > HISTORY_LIMIT_INT
            decision_list = decision_list[:HISTORY_LIMIT_INT]
            # Keep the last unresolved VPlan visible even when its decision is
            # older than the recent window. It remains its own cycle.
            unresolved_obj = connection_obj.execute(
                select_sql_dict["vplan"] + """ AND (plan.status_str IN ('ready','submitting','submitted','blocked','expired')
                     OR plan.missing_ack_count_int>0 OR plan.submit_ack_status_str='missing_critical')
                ORDER BY plan.vplan_id_int DESC LIMIT 1""", owner_tuple,
            ).fetchone()
            if unresolved_obj is not None and not any(row_obj["decision_plan_id_int"] == unresolved_obj["decision_plan_id_int"] for row_obj in decision_list):
                unresolved_decision_obj = connection_obj.execute(
                    select_sql_dict["decision_plan"] + " AND plan.decision_plan_id_int=?",
                    (*owner_tuple, unresolved_obj["decision_plan_id_int"]),
                ).fetchone()
                if unresolved_decision_obj is None:
                    raise ValueError("Unresolved plan has no scoped decision")
                decision_list.append(unresolved_decision_obj)
            if vplan_id_int is not None:
                selected_vplan_obj = connection_obj.execute(
                    select_sql_dict["vplan"] + " AND plan.vplan_id_int=?", (*owner_tuple, vplan_id_int),
                ).fetchone()
                if selected_vplan_obj is None or (decision_plan_id_int is not None and decision_plan_id_int != selected_vplan_obj["decision_plan_id_int"]):
                    result_dict.update(status_str="not_found", reason_str="Saved cycle not found")
                    return result_dict
                decision_plan_id_int = selected_vplan_obj["decision_plan_id_int"]
            decision_plan_id_int = decision_plan_id_int or latest_decision_id_int
            selected_decision_obj = connection_obj.execute(
                select_sql_dict["decision_plan"] + " AND plan.decision_plan_id_int=?", (*owner_tuple, decision_plan_id_int),
            ).fetchone()
            if selected_decision_obj is None:
                result_dict.update(status_str="not_found", reason_str="Saved cycle not found")
                return result_dict
            if not any(row_obj["decision_plan_id_int"] == decision_plan_id_int for row_obj in decision_list):
                decision_list.append(selected_decision_obj)
            cycle_list, vplan_map_dict = [], {}
            for decision_obj in decision_list:
                for field_str in ("created_timestamp_str", "updated_timestamp_str"):
                    _observed_ts(decision_obj[field_str], as_of_ts)
                plan_obj = connection_obj.execute(
                    "SELECT * FROM vplan WHERE decision_plan_id_int=?", (decision_obj["decision_plan_id_int"],),
                ).fetchone()
                if plan_obj is not None:
                    if any(plan_obj[field_str] != decision_obj[field_str] for field_str in IDENTITY_FIELD_TUPLE):
                        raise ValueError("Plan identity mismatch")
                    for field_str in ("created_timestamp_str", "updated_timestamp_str"):
                        _observed_ts(plan_obj[field_str], as_of_ts)
                    vplan_map_dict[decision_obj["decision_plan_id_int"]] = plan_obj
                cycle_list.append(_cycle_dict(decision_obj, plan_obj, latest_decision_id_int, release_obj.release_id_str))
            cycle_list.sort(key=lambda cycle_dict: cycle_dict["decision_plan_id_int"], reverse=True)
            selected_cycle_dict = next(cycle_dict for cycle_dict in cycle_list if cycle_dict["decision_plan_id_int"] == decision_plan_id_int)
            selected_vplan_obj = vplan_map_dict.get(decision_plan_id_int)
            selected_identity_tuple = tuple(selected_decision_obj[field_str] for field_str in IDENTITY_FIELD_TUPLE)
            selected_release_row_obj = connection_obj.execute(
                f"SELECT * FROM live_release WHERE {scope_str} AND mode_str='live'", selected_identity_tuple,
            ).fetchone()
            if selected_release_row_obj is None:
                raise ValueError("Selected release identity mismatch")
            selected_release_dict = _allow_dict(selected_release_row_obj, RELEASE_FIELD_TUPLE)
            selected_release_obj = SimpleNamespace(**selected_release_dict)
            decision_dict = _allow_dict(selected_decision_obj, PLAN_FIELD_TUPLE + ("decision_book_type_str", "cash_reserve_weight_float", "preserve_untouched_positions_bool", "rebalance_omitted_assets_to_zero_bool"))
            for source_str, destination_str in (
                ("decision_base_position_json_str", "decision_base_position_map_dict"),
                ("target_weight_json_str", "target_weight_map_dict"),
                ("entry_target_weight_json_str", "entry_target_weight_map_dict"),
                ("full_target_weight_json_str", "full_target_weight_map_dict"),
            ):
                decision_dict[destination_str] = _map_dict(selected_decision_obj[source_str])
            target_map_key_str = "full_target_weight_map_dict" if decision_dict["decision_book_type_str"] == "full_target_weight_book" else "entry_target_weight_map_dict"
            decision_dict["display_target_weight_map_dict"] = decision_dict[target_map_key_str] or decision_dict["target_weight_map_dict"]
            for source_str, destination_str in (("exit_asset_json_str", "exit_asset_list"), ("entry_priority_json_str", "entry_priority_list")):
                asset_list = json.loads(selected_decision_obj[source_str])
                if not isinstance(asset_list, list) or any(not isinstance(asset_str, str) for asset_str in asset_list):
                    raise ValueError("Invalid saved asset list")
                decision_dict[destination_str] = asset_list
            metadata_dict = json.loads(selected_decision_obj["snapshot_metadata_json_str"])
            decision_dict["snapshot_metadata_dict"] = _allow_dict(metadata_dict, ("norgate_data_profile_str", "norgate_snapshot_date_str", "data_profile_str", "snapshot_date_str"))
            pod_row_dict = dict(zip(IDENTITY_FIELD_TUPLE, selected_identity_tuple))
            pod_row_dict.update(mode_str="live", db_status_str="ok", as_of_timestamp_str=as_of_ts.isoformat(), source_stale_bool=False,
                latest_decision_plan_id_int=decision_plan_id_int, latest_decision_plan_status_str=decision_dict["status_str"],
                latest_decision_plan_submission_timestamp_str=decision_dict["submission_timestamp_str"],
                latest_decision_plan_target_execution_timestamp_str=decision_dict["target_execution_timestamp_str"],
                latest_vplan_id_int=None, latest_vplan_decision_plan_id_int=None,
                norgate_snapshot_status_dict={"status_str": "unknown"}, required_action_dict={})
            # Only the selected decision's saved date/profile can establish its
            # data session. A current Norgate file cannot prove a past cycle.
            snapshot_date_str = metadata_dict.get("norgate_snapshot_date_str")
            signal_ts = _observed_ts(decision_dict["signal_timestamp_str"], as_of_ts)
            if (metadata_dict.get("norgate_data_profile_str") == selected_release_obj.data_profile_str
                    and snapshot_date_str == signal_ts.astimezone(MARKET_TIMEZONE_OBJ).date().isoformat()):
                pod_row_dict["norgate_snapshot_status_dict"] = {"status_str": "ready", "snapshot_date_str": snapshot_date_str, "snapshot_fresh_for_cycle_bool": True}
            detail_dict = {"decision_dict": decision_dict}
            if selected_vplan_obj is not None:
                vplan_id_int = selected_vplan_obj["vplan_id_int"]
                vplan_dict = _allow_dict(selected_vplan_obj, PLAN_FIELD_TUPLE + ("vplan_id_int", "broker_snapshot_timestamp_str", "live_reference_snapshot_timestamp_str", "live_price_source_str", "submit_ack_status_str", "missing_ack_count_int", "submit_ack_checked_timestamp_str", "submission_key_str", "pod_budget_float"))
                for field_str in ("broker_snapshot_timestamp_str", "live_reference_snapshot_timestamp_str", "submit_ack_checked_timestamp_str"):
                    if vplan_dict.get(field_str) is not None:
                        _observed_ts(vplan_dict[field_str], as_of_ts)
                for source_str, destination_str in (("current_broker_position_json_str", "current_broker_position_map_dict"), ("target_share_json_str", "target_share_map_dict"), ("order_delta_json_str", "order_delta_map_dict"), ("live_reference_price_json_str", "live_reference_price_map_dict")):
                    vplan_dict[destination_str] = _map_dict(selected_vplan_obj[source_str])
                detail_dict["vplan_dict"] = vplan_dict
                for table_str, destination_str in (("vplan_row", "plan_row_list"), ("vplan_broker_order", "order_list"), ("vplan_broker_ack", "ack_list"), ("vplan_fill", "fill_list"), ("vplan_broker_order_event", "event_list")):
                    order_str = "asset_str, vplan_row_id_int" if table_str == "vplan_row" else CHILD_FIELD_DICT[table_str][0]
                    row_list = connection_obj.execute(f"SELECT * FROM {table_str} WHERE vplan_id_int=? ORDER BY {order_str} LIMIT ?", (vplan_id_int, DETAIL_LIMIT_INT + 1)).fetchall()
                    if len(row_list) > DETAIL_LIMIT_INT:
                        raise ValueError("Cycle detail exceeds display limit")
                    for child_obj in row_list:
                        if table_str != "vplan_row" and (child_obj["decision_plan_id_int"] != decision_plan_id_int or child_obj["account_route_str"] != release_obj.account_route_str):
                            raise ValueError("Child identity mismatch")
                        for field_str in CHILD_FIELD_DICT[table_str]:
                            if field_str.endswith("timestamp_str") and child_obj[field_str] is not None:
                                _observed_ts(child_obj[field_str], as_of_ts)
                    detail_dict[destination_str] = [_allow_dict(child_obj, CHILD_FIELD_DICT[table_str]) for child_obj in row_list]
                request_list = build_broker_order_request_list_from_vplan(SimpleNamespace(
                    **vplan_dict, vplan_row_list=[SimpleNamespace(**row_dict) for row_dict in detail_dict["plan_row_list"]]))
                request_iter = iter(request_list)
                for row_dict in detail_dict["plan_row_list"]:
                    if abs(row_dict["order_delta_share_float"]) > 1e-9:
                        row_dict["order_request_key_str"] = next(request_iter).order_request_key_str
                reconciliation_obj = connection_obj.execute(
                    """SELECT * FROM vplan_reconciliation_snapshot WHERE pod_id_str=? AND decision_plan_id_int=?
                    AND vplan_id_int=? AND stage_str='post_execution' ORDER BY vplan_reconciliation_snapshot_id_int DESC LIMIT 1""",
                    (release_obj.pod_id_str, decision_plan_id_int, vplan_id_int),
                ).fetchone()
                if reconciliation_obj is not None:
                    reconcile_ts = _observed_ts(reconciliation_obj["created_timestamp_str"], as_of_ts)
                    target_ts = parse_timestamp_ts(vplan_dict["target_execution_timestamp_str"])
                    if (target_ts is None or reconcile_ts < target_ts
                            or reconcile_ts.astimezone(MARKET_TIMEZONE_OBJ).date() != target_ts.astimezone(MARKET_TIMEZONE_OBJ).date()):
                        reconciliation_obj = None
                if reconciliation_obj is not None:
                    reconciliation_dict = _allow_dict(reconciliation_obj, ("vplan_reconciliation_snapshot_id_int", "pod_id_str", "decision_plan_id_int", "vplan_id_int", "stage_str", "status_str", "created_timestamp_str", "model_cash_float", "broker_cash_float"))
                    for source_str, destination_str in (("model_position_json_str", "model_position_map_dict"), ("broker_position_json_str", "broker_position_map_dict")):
                        reconciliation_dict[destination_str] = _map_dict(reconciliation_obj[source_str])
                    detail_dict["reconciliation_dict"] = reconciliation_dict
                    pod_row_dict.update(latest_reconciliation_status_str=reconciliation_dict["status_str"], latest_reconciliation_timestamp_str=reconciliation_dict["created_timestamp_str"])
                pod_row_dict.update(latest_vplan_id_int=vplan_id_int, latest_vplan_decision_plan_id_int=decision_plan_id_int,
                    latest_vplan_is_for_latest_decision_bool=True, latest_vplan_status_str=vplan_dict["status_str"],
                    latest_vplan_submission_timestamp_str=vplan_dict["submission_timestamp_str"],
                    latest_vplan_target_execution_timestamp_str=vplan_dict["target_execution_timestamp_str"],
                    latest_submit_ack_status_str=vplan_dict["submit_ack_status_str"], missing_ack_count_int=vplan_dict["missing_ack_count_int"],
                    broker_order_count_int=len(detail_dict["order_list"]), broker_ack_count_int=len(detail_dict["ack_list"]), fill_count_int=len(detail_dict["fill_list"]))
            target_ts = parse_timestamp_ts(selected_cycle_dict["target_execution_timestamp_str"])
            eod_dict = _eod_dict(connection_obj, selected_release_obj, target_ts, as_of_ts)
            pod_row_dict["eod_snapshot_dict"] = eod_dict
        result_dict.update(detail_dict)
        result_dict.update(status_str="ok", reason_str="", cycle_list=cycle_list, selected_cycle_dict=selected_cycle_dict, selected_release_dict=selected_release_dict, pod_row_dict=pod_row_dict, eod_dict=eod_dict)
    except (AttributeError, IndexError, KeyError, TypeError, ValueError, OverflowError, OSError, sqlite3.Error):
        # Do not keep partially read evidence after an identity/schema/time error.
        result_dict.update(status_str="unknown", reason_str="Saved cycle evidence could not be verified", cycle_list=[], selected_cycle_dict=None)
    return result_dict
