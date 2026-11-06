"""Read-only projection of the one opening-cross recovery attempt."""

from contextlib import closing
from datetime import datetime, timedelta
import math
from pathlib import Path
import sqlite3

from alpha.live import auto_complete
from alpha.live.dashboard_v3.filters import MARKET_TIMEZONE_OBJ


TERMINAL_STATUS_SET = {"cancelled", "apicancelled", "inactive", "rejected"}


def _time_str(timestamp_obj, format_str="%H:%M"):
    if isinstance(timestamp_obj, str):
        timestamp_obj = datetime.fromisoformat(timestamp_obj)
    return timestamp_obj.astimezone(MARKET_TIMEZONE_OBJ).strftime(format_str)


def _quantity_int(value_obj):
    value_float = float(value_obj or 0)
    return int(math.floor(abs(value_float) + 1e-9)) if math.isfinite(value_float) else 0


def _fill_text_str(fill_list, side_str):
    quantity_float = sum(abs(float(row_dict["fill_amount_float"])) for row_dict in fill_list)
    if quantity_float <= 0:
        return "Fill price not available"
    price_float = sum(abs(float(row_dict["fill_amount_float"])) * float(row_dict["fill_price_float"])
                      for row_dict in fill_list) / quantity_float
    price_text_str = f"{_quantity_int(quantity_float)} @ {price_float:.2f}"
    open_list = [row_dict.get("official_open_price_float") for row_dict in fill_list]
    if not open_list or any(value_obj is None or not math.isfinite(float(value_obj)) or float(value_obj) <= 0
                            for value_obj in open_list):
        return price_text_str + " · open price not available"
    if max(open_list) - min(open_list) > 1e-9:
        return price_text_str + " · open price not available"
    # Display only: adverse-positive cost = side * (fill VWAP / official open - 1) * 10,000.
    side_float = 1.0 if side_str == "BUY" else -1.0
    cost_float = side_float * (price_float / float(open_list[0]) - 1.0) * 10000.0
    direction_str = "above" if price_float >= float(open_list[0]) else "below"
    return price_text_str + f" · {cost_float:+.1f} bps {direction_str} the open"


def load_recovery_view_dict(target_obj, vplan_obj, snapshot_obj, *, as_of_ts, newer_plan_bool=False):
    """Saved facts only: one re-send, then the normal missing-position calculation."""
    release_obj = target_obj.release_obj
    uri_str = Path(target_obj.db_path_str).resolve().as_uri() + "?mode=ro"
    with closing(sqlite3.connect(uri_str, uri=True)) as connection_obj:
        connection_obj.row_factory = sqlite3.Row
        table_set = {row_obj[0] for row_obj in connection_obj.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        def scoped_rows_list(table_str):
            if table_str not in table_set:
                return []
            return [dict(row_obj) for row_obj in connection_obj.execute(
                f"SELECT * FROM {table_str} WHERE vplan_id_int=? AND account_route_str=?",
                (vplan_obj.vplan_id_int, release_obj.account_route_str))]
        order_list = scoped_rows_list("vplan_broker_order")
        event_list = scoped_rows_list("vplan_broker_order_event")
        fill_list = scoped_rows_list("vplan_fill")
        ack_list = scoped_rows_list("vplan_broker_ack")
        attempt_list = auto_complete.load_attempt_row_list(connection_obj, vplan_obj.vplan_id_int)
    attempt_map_dict = {row_dict["asset_str"]: dict(row_dict) for row_dict in attempt_list
                       if row_dict["pod_id_str"] == release_obj.pod_id_str
                       and row_dict["account_route_str"] == release_obj.account_route_str}
    cross_map_dict = auto_complete.cross_cancel_leg_dict(vplan_obj, order_list, event_list)
    cancelled_map_dict = {}
    for order_dict in order_list:
        if str(order_dict.get("status_str", "")).lower() not in TERMINAL_STATUS_SET:
            continue
        matching_list = [row_dict for row_dict in event_list
                         if row_dict["broker_order_id_str"] == order_dict["broker_order_id_str"]]
        message_str = next((str(row_dict["message_str"]) for row_dict in reversed(matching_list)
                            if row_dict.get("message_str")), "broker cancelled")
        cancelled_map_dict[order_dict["asset_str"]] = message_str[:240]
    if not cancelled_map_dict and not attempt_map_dict and not cross_map_dict:
        return {"row_list": []}
    timing_dict = auto_complete.timing_dict(vplan_obj, release_obj.session_calendar_id_str)
    position_map_dict = (snapshot_obj.position_amount_map if snapshot_obj is not None
                         else vplan_obj.current_broker_position_map)
    row_list = []
    for asset_str in sorted(set(cancelled_map_dict) | set(attempt_map_dict) | set(cross_map_dict)):
        target_float = float(vplan_obj.target_share_map.get(asset_str, 0.0))
        actual_float = float(position_map_dict.get(asset_str, 0.0))
        gap_float = target_float - actual_float
        original_float = float(vplan_obj.order_delta_map.get(asset_str, 0.0))
        side_str = "BUY" if (gap_float or original_float) > 0 else "SELL"
        quantity_int = _quantity_int(gap_float)
        attempt_dict = attempt_map_dict.get(asset_str)
        cross_dict = cross_map_dict.get(asset_str)
        state_str = "not_automatic"
        text_str = "Rejected by IBKR · " + cancelled_map_dict.get(asset_str, "broker cancelled")
        if attempt_dict:
            state_str = {"working": "sent", "completed": "completed"}.get(attempt_dict["status_str"], "not_completed")
            if state_str in {"sent", "completed"}:
                quantity_int = _quantity_int(attempt_dict.get("quantity_float")) or _quantity_int(original_float)
                side_str = "BUY" if original_float > 0 else "SELL"
            linked_order_list = [row_dict for row_dict in order_list
                if row_dict.get("order_request_key_str") == attempt_dict["order_request_key_str"]]
            linked_id_set = {row_dict["broker_order_id_str"] for row_dict in linked_order_list}
            order_id_str = attempt_dict.get("broker_order_id_str")
            if order_id_str:
                linked_id_set.add(order_id_str)
            elif len(linked_id_set) == 1:
                order_id_str = next(iter(linked_id_set))
            if state_str == "sent":
                acknowledged_bool = any(row_dict.get("broker_order_id_str") == order_id_str
                    and row_dict.get("order_request_key_str") == attempt_dict["order_request_key_str"]
                    and row_dict.get("broker_response_ack_bool") in (True, 1)
                    and row_dict.get("ack_status_str") == "broker_acked" for row_dict in ack_list)
                acknowledged_bool = acknowledged_bool or any(row_dict.get("status_str") in
                    {"Submitted", "PreSubmitted", "PartiallyFilled", "PendingCancel", "Filled"} for row_dict in linked_order_list)
                submitted_str = next((row_dict["submitted_timestamp_str"] for row_dict in linked_order_list
                    if row_dict.get("broker_order_id_str") == order_id_str and row_dict.get("submitted_timestamp_str")),
                    attempt_dict["attempted_timestamp_str"])
                text_str = (f"Re-sent {_time_str(submitted_str, '%H:%M:%S')} · market · order {order_id_str}"
                            " · waiting for fill" if order_id_str and acknowledged_bool else
                            "Re-send started · awaiting broker check")
            elif state_str == "completed":
                if as_of_ts >= timing_dict["close_ts"] + timedelta(minutes=10):
                    continue
                original_filled_bool = attempt_dict.get("reason_str") == "Original order already filled"
                if original_filled_bool:
                    linked_id_set = {attempt_dict["original_broker_order_id_str"]}
                matching_fill_dict = {(row_dict["fill_timestamp_str"], row_dict["fill_amount_float"], row_dict["fill_price_float"]): row_dict
                    for row_dict in fill_list if row_dict["broker_order_id_str"] in linked_id_set}
                text_str = ("Original order filled · " if original_filled_bool else "Re-sent and filled · ") + _fill_text_str(list(matching_fill_dict.values()), side_str)
                ended_ts = datetime.fromisoformat(attempt_dict["ended_timestamp_str"] or attempt_dict["updated_timestamp_str"])
                if snapshot_obj is not None and snapshot_obj.snapshot_timestamp_ts >= ended_ts and _quantity_int(gap_float):
                    state_str, quantity_int = "not_completed", _quantity_int(gap_float)
                    side_str = "BUY" if gap_float > 0 else "SELL"
                    text_str = f"Re-sent and filled · {quantity_int} missing from plan"
            else:
                reason_str = str(attempt_dict.get("reason_str") or "needs operator review")
                sent_bool = (bool(attempt_dict.get("request_json_str") or order_id_str or attempt_dict.get("quantity_float"))
                    and "newer decision" not in reason_str.casefold() and not reason_str.casefold().startswith("not sent"))
                # Execution evidence of the re-send itself, never holdings, decides "partly filled".
                partly_bool = sent_bool and auto_complete._filled_quantity_float(order_list, event_list, fill_list,
                    attempt_dict["order_request_key_str"], order_id_str) > 1e-9
                text_str = (f"Re-sent, {'partly' if partly_bool else 'not'} filled · {quantity_int} missing" if sent_bool else
                            "Not re-sent · " + reason_str)
                if reason_str.startswith("No broker answer"):
                    text_str = f"No broker answer · {quantity_int} missing"
        elif cross_dict:
            if newer_plan_bool or as_of_ts >= timing_dict["send_cutoff_ts"]:
                state_str = "not_completed"
                text_str = "Not re-sent · " + ("a newer plan exists" if newer_plan_bool else
                    "the pod did not re-send before " + _time_str(timing_dict["send_cutoff_ts"]))
            else:
                state_str = "planned"
                side_str = "BUY" if cross_dict["direction_float"] > 0 else "SELL"
                filled_float = max(cross_dict["filled_quantity_float"], auto_complete._filled_quantity_float(
                    order_list, event_list, fill_list, cross_dict["original_order_request_key_str"],
                    cross_dict["original_broker_order_id_str"]))
                quantity_int = _quantity_int(max(0.0, cross_dict["cancelled_qty_float"] - filled_float))
                when_str = "at " + _time_str(timing_dict["start_ts"]) if as_of_ts < timing_dict["start_ts"] else "at the next check"
                text_str = "Cancelled by IBKR · crosses a related order · re-sends " + when_str
        can_complete_bool = state_str in {"not_completed", "not_automatic"} and quantity_int > 0
        if not quantity_int and state_str not in {"sent", "completed"}:
            continue
        if can_complete_bool and as_of_ts < timing_dict["close_ts"] - timedelta(minutes=15):
            text_str += " · MOC until " + _time_str(timing_dict["close_ts"] - timedelta(minutes=15))
        row_list.append({"asset_str": asset_str, "target_share_float": target_float, "saved_position_float": actual_float,
            "side_str": side_str, "quantity_int": quantity_int, "state_str": state_str, "text_str": text_str,
            "can_complete_bool": can_complete_bool, "auto_complete_bool": bool(cross_dict or attempt_dict)})
    unrelated_ack_bool = any(row_dict.get("ack_status_str") == "missing_critical"
                             and row_dict.get("asset_str") not in cross_map_dict for row_dict in ack_list)
    return {"row_list": row_list, "timing_dict": timing_dict, "attempt_list": list(attempt_map_dict.values()),
        "lagging_bool": as_of_ts >= timing_dict["start_ts"],
        "moc_open_bool": as_of_ts < timing_dict["close_ts"] - timedelta(minutes=15),
        "total_order_count_int": sum(abs(float(value_obj)) > 1e-9 for value_obj in vplan_obj.order_delta_map.values()),
        "unrelated_issue_bool": unrelated_ack_bool}


def recovery_summary_dict(missing_dict):
    row_list = missing_dict.get("row_list") or []
    # A saved broker rejection is red with Complete…, so it owns the header like a failed re-send.
    recovery_list = [row_dict for row_dict in row_list if row_dict.get("auto_complete_bool")
                     or (row_dict["state_str"] == "not_automatic" and row_dict.get("can_complete_bool"))]
    if not recovery_list:
        return {}
    planned_list = [row_dict for row_dict in recovery_list if row_dict["state_str"] == "planned"]
    unresolved_list = [row_dict for row_dict in row_list if row_dict["state_str"] in {"missing", "off", "not_completed", "not_automatic"}]
    recovery_dict = missing_dict.get("recovery_dict") or {}
    timing_dict = recovery_dict.get("timing_dict") or {}
    start_str = _time_str(timing_dict["start_ts"]) if timing_dict else "scheduled time"
    recheck_str = _time_str(timing_dict["close_ts"] + timedelta(minutes=10)) if timing_dict else "after close"
    cutoff_str = _time_str(timing_dict["close_ts"] - timedelta(minutes=15)) if timing_dict else "the close"
    detail_str = " · ".join(f"{row_dict['asset_str']} {row_dict['side_str']} {row_dict['quantity_int']}" for row_dict in recovery_list)
    when_str = "at the next check" if recovery_dict.get("lagging_bool") else "at " + start_str
    if unresolved_list:
        complete_str = ", ".join(f"{row_dict['asset_str']} {row_dict['side_str']} {row_dict['quantity_int']}" for row_dict in unresolved_list)
        state_str, pill_str, title_str = "fail", "Action needed", "Complete " + complete_str
        if recovery_dict.get("moc_open_bool"):
            title_str += " by " + cutoff_str
        cause_str = ("Some planned shares are still missing. " if any(row_dict["state_str"] in {"not_automatic", "missing", "off"} for row_dict in unresolved_list)
            else "The re-sent order filled; holdings still differ from the plan. " if any(row_dict["text_str"].startswith("Re-sent and filled") for row_dict in unresolved_list)
            else "The order was not re-sent. " if any(row_dict["text_str"].startswith("Not re-sent") for row_dict in unresolved_list)
            else "The broker reply could not be confirmed. " if any(row_dict["text_str"].startswith("No broker answer") for row_dict in unresolved_list)
            else "The re-sent order was partly filled. " if any(row_dict["text_str"].startswith("Re-sent, partly filled") for row_dict in unresolved_list)
            else "The re-sent order was not filled. ")
        detail_str = cause_str + "Use Complete… below. The pod continues by itself after the " + recheck_str + " re-check."
    elif any(row_dict["state_str"] == "sent" for row_dict in recovery_list):
        state_str, pill_str, title_str = "now", "Completing", "Completing cancelled orders"
    elif any(row_dict["state_str"] == "planned" for row_dict in recovery_list):
        state_str, pill_str, title_str = "late", ("Re-send next check" if recovery_dict.get("lagging_bool") else "Re-send " + start_str), "Re-send " + when_str
    else:
        state_str, pill_str, title_str = "done", "On track", ("Order filled" if all(
            row_dict["text_str"].startswith("Original order filled") for row_dict in recovery_list) else "Re-sent and filled")
    return {"state_str": state_str, "pill_str": pill_str, "title_str": title_str,
        "detail_str": detail_str,
        "banner_detail_str": " and ".join(f"{row_dict['asset_str']} {row_dict['side_str']} {row_dict['quantity_int']}" for row_dict in planned_list)
            + (" crosses an order in a related account. The pod re-sends it " + when_str + " as a market order."
               if len(planned_list) == 1 else
               " cross an order in a related account. The pod re-sends them " + when_str + " as market orders."),
        "planned_bool": bool(planned_list),
        "attention_bool": bool(unresolved_list), "count_int": len(planned_list or recovery_list),
        "total_order_count_int": recovery_dict.get("total_order_count_int", len(recovery_list)),
        "unrelated_issue_bool": bool((missing_dict.get("recovery_dict") or {}).get("unrelated_issue_bool")),
        "row_list": recovery_list}


def apply_recovery_overview(overview_dict, pod_dict, missing_dict):
    """Replace only the affected execution warning, leaving other problems visible."""
    summary_dict = recovery_summary_dict(missing_dict)
    if not summary_dict:
        return
    pod_id_str = pod_dict["pod_id_str"]
    existing_list = [item_dict for item_dict in overview_dict["attention_list"] if item_dict["pod_id_str"] == pod_id_str]
    protected_bool = summary_dict["unrelated_issue_bool"] or any(
        item_dict.get("kind_str") not in {"cycle", "action", "missing_trades", "auto_complete"}
        or item_dict.get("scheduler_bool")
        or item_dict.get("step_str", "") not in {"", "Submit", "Fill", "Reconcile"} for item_dict in existing_list)
    pod_dict["recovery_summary_dict"] = summary_dict
    if protected_bool:
        return
    overview_dict["attention_list"] = [item_dict for item_dict in overview_dict["attention_list"] if item_dict["pod_id_str"] != pod_id_str]
    pod_dict.update(state_str=summary_dict["state_str"], pill_str=summary_dict["pill_str"],
        now_str=summary_dict["title_str"], now_detail_str=summary_dict["detail_str"],
        next_str="Review remaining quantity" if summary_dict["attention_bool"] else
            "Wait for scheduled reconciliation" if summary_dict["state_str"] == "done" else "Automatic completion",
        next_time_str="", next_detail_str="", next_line_list=[], next_in_str="")
    if summary_dict["attention_bool"]:
        overview_dict["attention_list"].append({"state_str": "fail", "pod_id_str": pod_id_str,
            "pod_name_str": pod_dict.get("name_str", pod_id_str), "title_str": summary_dict["title_str"],
            "detail_str": summary_dict["detail_str"], "kind_str": "auto_complete", "age_str": "—", "next_command_list": []})


def apply_recovery_pod(pod_page_dict, missing_dict):
    summary_dict = recovery_summary_dict(missing_dict)
    if not summary_dict or pod_page_dict.get("historical_bool"):
        return
    pod_page_dict["recovery_summary_dict"] = summary_dict
    attention_dict = pod_page_dict.get("attention_dict") or {}
    protected_bool = summary_dict["unrelated_issue_bool"] or (attention_dict and (
        attention_dict.get("kind_str") not in {"cycle", "action", "missing_trades", "auto_complete"}
        or attention_dict.get("scheduler_bool")
        or attention_dict.get("step_str", "") not in {"", "Submit", "Fill", "Reconcile"}))
    if protected_bool:
        return
    pod_page_dict.update(state_str=summary_dict["state_str"], pill_str=summary_dict["pill_str"],
        cycle_state_str=summary_dict["state_str"], cycle_pill_str=summary_dict["pill_str"],
        verdict_str=summary_dict["title_str"] + ".", verdict_detail_str=summary_dict["detail_str"])
    pod_page_dict["header_dict"].update(state_str=summary_dict["state_str"], pill_str=summary_dict["pill_str"],
        verdict_str=summary_dict["title_str"] + ".", detail_str=summary_dict["detail_str"], quiet_bool=False)
    pod_page_dict["attention_dict"] = ({"state_str": "fail", "title_str": summary_dict["title_str"],
        "detail_str": summary_dict["detail_str"], "kind_str": "auto_complete", "pod_id_str": pod_page_dict["pod_id_str"]}
        if summary_dict["attention_bool"] else {})
