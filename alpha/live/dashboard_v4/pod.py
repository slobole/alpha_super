"""Small, allowlisted presentation of one saved LIVE cycle."""

from copy import deepcopy
import math

from alpha.live.dashboard_v3.client_operations import EVENT_LABEL_DICT, SOURCE_MAX_AGE_SECONDS_INT
from alpha.live.dashboard_v3.filters import MARKET_TIMEZONE_OBJ
from alpha.live.dashboard_v3.operator_tools import redact_diagnostic_value
from alpha.live.dashboard_v4.cycle import build_cycle_view_dict
from alpha.live.dashboard_v4.overview import STATE_RANK_DICT, _state_str
from alpha.live.ops_report import parse_timestamp_ts


TAB_TUPLE = (("plan", "Plan vs actual"), ("decision", "Decision"), ("orders", "Orders"),
             ("fills", "Fills"), ("reconcile", "Reconcile"), ("events", "Events"), ("files", "Files"))
STEP_TAB_DICT = {"Data": "decision", "Decide": "decision", "Plan": "plan", "Submit": "orders",
                 "Fill": "fills", "Reconcile": "reconcile", "EOD": "events"}


def _number_float(value_obj):
    if isinstance(value_obj, bool) or not isinstance(value_obj, (int, float)):
        return None
    return float(value_obj) if math.isfinite(value_obj) else None


def _number_str(value_obj, *, price_bool=False):
    value_float = _number_float(value_obj)
    if value_float is None:
        return "—"
    return f"{value_float:,.2f}" if price_bool else f"{value_float:,.6f}".rstrip("0").rstrip(".")


def _text_str(value_obj):
    return str(redact_diagnostic_value(value_obj)) if isinstance(value_obj, str) else "—"


def _time_str(value_obj, as_of_ts):
    timestamp_ts = parse_timestamp_ts(value_obj)
    if timestamp_ts is None or timestamp_ts > as_of_ts:
        return "—"
    market_ts = timestamp_ts.astimezone(MARKET_TIMEZONE_OBJ)
    format_str = "%H:%M:%S" if market_ts.date() == as_of_ts.astimezone(MARKET_TIMEZONE_OBJ).date() else "%m-%d %H:%M:%S"
    return market_ts.strftime(format_str)


def _table_dict(columns_list, rows_list, *, note_str=""):
    return {"column_list": columns_list, "row_list": rows_list, "note_str": note_str,
            "empty_str": "No saved evidence for this cycle."}


def _row_dict(value_list, *, state_str="", match_str=""):
    return {"cell_list": value_list, "state_str": state_str, "match_str": match_str}


def _ack_dict(plan_dict, order_list, ack_list, *, as_of_ts):
    """One acknowledged broker order must identify this exact saved request."""
    request_str = plan_dict.get("order_request_key_str")
    matching_orders = [item_dict for item_dict in order_list if request_str and item_dict.get("order_request_key_str") == request_str]
    matching_acks = [item_dict for item_dict in ack_list if request_str and item_dict.get("order_request_key_str") == request_str]
    order_dict = matching_orders[0] if len(matching_orders) == 1 else {}
    ack_dict = matching_acks[0] if len(matching_acks) == 1 else {}
    order_id_str = str(order_dict.get("broker_order_id_str") or "")
    critical_bool = any(item_dict.get("ack_status_str") == "missing_critical" for item_dict in matching_acks)
    response_ts = parse_timestamp_ts(ack_dict.get("response_timestamp_str"))
    owned_bool = (bool(order_id_str) and order_dict.get("asset_str") == plan_dict.get("asset_str")
        and ack_dict.get("asset_str") == plan_dict.get("asset_str")
        and str(ack_dict.get("broker_order_id_str") or "") == order_id_str
        and sum(str(item_dict.get("broker_order_id_str") or "") == order_id_str for item_dict in order_list) == 1)
    if plan_dict.get("broker_order_type_str"):
        owned_bool = owned_bool and all(item_dict.get("broker_order_type_str") == plan_dict["broker_order_type_str"] for item_dict in (order_dict, ack_dict))
    acked_bool = (owned_bool and ack_dict.get("broker_response_ack_bool") in (True, 1)
        and ack_dict.get("ack_status_str") == "broker_acked" and response_ts is not None and response_ts <= as_of_ts)
    return {"label_str": "No ack" if critical_bool else "Acked" if acked_bool else "Unknown",
        "ack_dict": ack_dict, "order_dict": order_dict, "order_id_str": order_id_str}


def _apply_ack_evidence(row_dict, source_dict, *, as_of_ts):
    """Detailed contradictions can weaken a summary, never promote a stage."""
    order_list, ack_list = (source_dict.get(key_str) or [] for key_str in ("order_list", "ack_list"))
    critical_count_int = sum(item_dict.get("ack_status_str") == "missing_critical" for item_dict in ack_list)
    if critical_count_int:
        row_dict["missing_ack_count_int"] = max(row_dict.get("missing_ack_count_int") or 0, critical_count_int)
        row_dict["latest_submit_ack_status_str"] = "missing_critical"
    elif row_dict.get("latest_submit_ack_status_str") == "complete":
        plan_list = [plan_dict for plan_dict in source_dict.get("plan_row_list") or []
            if _number_float(plan_dict.get("order_delta_share_float")) is not None and abs(plan_dict["order_delta_share_float"]) > 1e-9]
        request_set = {plan_dict.get("order_request_key_str") for plan_dict in plan_list}
        complete_bool = (bool(plan_list) and None not in request_set and "" not in request_set
            and len(request_set) == len(plan_list) == len(order_list) == len(ack_list)
            and row_dict.get("broker_order_count_int") == len(order_list)
            and row_dict.get("broker_ack_count_int") == len(ack_list)
            and all(_ack_dict(plan_dict, order_list, ack_list, as_of_ts=as_of_ts)["label_str"] == "Acked" for plan_dict in plan_list))
        if not complete_bool:
            row_dict["broker_ack_count_int"] = None


def build_evidence_tables_dict(source_dict, *, as_of_ts, fresh_bool):
    """Never join executions by symbol: a sell and a buy can share that symbol."""
    plan_list = source_dict.get("plan_row_list") or []
    order_list, ack_list, fill_list = (source_dict.get(key_str) or [] for key_str in ("order_list", "ack_list", "fill_list"))
    proof_dict = source_dict.get("cycle_evidence_dict") or {}
    proof_order_map = {item_dict["broker_order_id_str"]: item_dict for item_dict in proof_dict.get("order_list") or []}
    reconcile_dict = source_dict.get("reconciliation_dict") or {}
    before_map = (source_dict.get("vplan_dict") or {}).get("current_broker_position_map_dict")
    broker_map = reconcile_dict.get("broker_position_map_dict")
    model_map = reconcile_dict.get("model_position_map_dict")
    plan_rows, order_rows = [], []
    for plan_dict in plan_list:
        symbol_str = _text_str(plan_dict.get("asset_str"))
        ack_evidence_dict = _ack_dict(plan_dict, order_list, ack_list, as_of_ts=as_of_ts)
        order_id_str = ack_evidence_dict["order_id_str"]
        proof_order_dict = proof_order_map.get(order_id_str, {})
        amount_float = _number_float(plan_dict.get("order_delta_share_float"))
        order_str = "—" if amount_float is None or abs(amount_float) <= 1e-9 else f"{'BUY' if amount_float > 0 else 'SELL'} {_number_str(abs(amount_float))}"
        filled_float = proof_order_dict.get("filled_share_float")
        matched_fills = [item_dict for item_dict in fill_list if order_id_str and str(item_dict.get("broker_order_id_str")) == order_id_str]
        price_float = None
        if filled_float and matched_fills:
            quantity_list = [_number_float(item_dict.get("fill_amount_float")) for item_dict in matched_fills]
            price_list = [_number_float(item_dict.get("fill_price_float")) for item_dict in matched_fills]
            if all(value_float is not None for value_float in quantity_list + price_list):
                # Display-only VWAP = sum(abs(fill shares) * fill price) / sum(abs(fill shares)).
                gross_float = sum(abs(value_float) for value_float in quantity_list)
                if gross_float:
                    price_float = sum(abs(quantity_float) * fill_price_float for quantity_float, fill_price_float in zip(quantity_list, price_list)) / gross_float
        after_float, match_str = None, "unk"
        if isinstance(broker_map, dict) and isinstance(model_map, dict):
            after_float = _number_float(broker_map.get(plan_dict.get("asset_str"), 0))
            model_float = _number_float(model_map.get(plan_dict.get("asset_str"), 0))
            if fresh_bool and after_float is not None and model_float is not None:
                match_str = "done" if abs(after_float - model_float) <= 1e-9 else "fail"
        before_float = before_map.get(plan_dict.get("asset_str"), 0) if isinstance(before_map, dict) else None
        plan_rows.append(_row_dict([symbol_str, _number_str(before_float), order_str,
            _number_str(abs(filled_float)) if filled_float is not None else "—", _number_str(price_float, price_bool=True),
            _number_str(after_float)], match_str=match_str))
        if amount_float is not None and abs(amount_float) <= 1e-9:
            continue
        ack_dict = ack_evidence_dict["ack_dict"]
        ack_label_str = ack_evidence_dict["label_str"]
        response_ts = parse_timestamp_ts(ack_dict.get("response_timestamp_str"))
        submit_ts = parse_timestamp_ts((source_dict.get("vplan_dict") or {}).get("submission_timestamp_str"))
        ack_time_str = _time_str(response_ts.isoformat(), as_of_ts) if (ack_label_str == "Acked" and response_ts is not None
            and submit_ts is not None and response_ts > submit_ts) else "—"
        order_rows.append(_row_dict([symbol_str, order_str, ack_label_str if fresh_bool else "Unknown",
            _text_str(ack_dict.get("ack_source_str")), ack_time_str,
            _number_str(abs(filled_float)) if filled_float is not None else "—", _text_str(order_id_str) or "—"],
            state_str="fail" if ack_label_str == "No ack" and fresh_bool else ""))
    decision_dict = source_dict.get("decision_dict") or {}
    target_map = decision_dict.get("display_target_weight_map_dict") or decision_dict.get("target_weight_map_dict") or {}
    decision_rows = [_row_dict([_text_str(symbol_str), _number_str(weight_float * 100) + "%" if _number_float(weight_float) is not None else "—"])
                     for symbol_str, weight_float in target_map.items()]
    decision_rows += [_row_dict([_text_str(symbol_str), "Exit"]) for symbol_str in decision_dict.get("exit_asset_list") or []]
    reconcile_rows = []
    if isinstance(broker_map, dict) and isinstance(model_map, dict):
        for symbol_str in sorted(set(broker_map) | set(model_map)):
            model_float, broker_float = _number_float(model_map.get(symbol_str, 0)), _number_float(broker_map.get(symbol_str, 0))
            difference_float = broker_float - model_float if model_float is not None and broker_float is not None else None
            reconcile_rows.append(_row_dict([_text_str(symbol_str), _number_str(model_float), _number_str(broker_float), _number_str(difference_float)]))
    event_rows = []
    event_list = sorted(source_dict.get("event_list") or [],
        key=lambda item_dict: parse_timestamp_ts(item_dict.get("timestamp_str") or item_dict.get("event_timestamp_str")) or as_of_ts, reverse=True)
    for item_dict in event_list:
        code_str = item_dict.get("event_type_str") or item_dict.get("event_str") or ""
        status_str = item_dict.get("status_str") or ""
        label_str = EVENT_LABEL_DICT.get(code_str, _text_str(code_str).replace("_", " ")) if code_str else {
            "Submitted": "Order sent", "PreSubmitted": "Order accepted", "Filled": "Order filled",
            "Cancelled": "Order cancelled", "Inactive": "Order inactive", "PendingSubmit": "Waiting for broker",
        }.get(status_str, "Order " + _text_str(status_str))
        event_rows.append(_row_dict([_time_str(item_dict.get("timestamp_str") or item_dict.get("event_timestamp_str"), as_of_ts),
            _text_str(item_dict.get("asset_str")), label_str]))
    result_dict = {
        "plan": _table_dict(["Symbol", "Before", "Order", "Filled", "Fill px", "After", "Broker = model"], plan_rows),
        "decision": _table_dict(["Symbol", "Target"], decision_rows, note_str="Entry and exit targets" if decision_dict.get("decision_book_type_str") == "incremental_entry_exit_book" else "Full portfolio targets" if decision_dict.get("decision_book_type_str") == "full_target_weight_book" else "Saved decision targets"),
        "orders": _table_dict(["Symbol", "Order", "Ack", "Source", "Ack time", "Filled", "Broker id"], order_rows),
        "fills": _table_dict(["Symbol", "Shares", "Fill px", "Time", "Broker id"], [
            _row_dict([_text_str(item_dict.get("asset_str")), _number_str(item_dict.get("fill_amount_float")),
                       _number_str(item_dict.get("fill_price_float"), price_bool=True), _time_str(item_dict.get("fill_timestamp_str"), as_of_ts),
                       _text_str(item_dict.get("broker_order_id_str"))]) for item_dict in fill_list]),
        "reconcile": _table_dict(["Symbol", "Model", "Broker", "Difference"], reconcile_rows),
        "events": _table_dict(["Time", "Symbol", "What"], event_rows),
        "files": _table_dict(["File", "Recorded"], [_row_dict([_text_str(item_dict.get("name_str")), _time_str(item_dict.get("timestamp_str"), as_of_ts)])
                                                   for item_dict in source_dict.get("file_list") or []]),
    }
    result_dict["files"]["empty_str"] = "No cycle-linked files available here."
    return result_dict


def build_pod_page_dict(overview_dict, source_dict, finance_dict, *, pod_id_str, as_of_ts, tab_str=""):
    pod_summary_dict = next(item_dict for item_dict in overview_dict["pod_list"] if item_dict["pod_id_str"] == pod_id_str)
    row_dict = deepcopy(source_dict.get("pod_row_dict") or {})
    row_dict.update(mode_str="live", pod_id_str=pod_id_str)
    source_ts = parse_timestamp_ts(row_dict.get("as_of_timestamp_str"))
    fresh_bool = (overview_dict["source_fresh_bool"] and source_dict.get("status_str") == "ok"
        and source_ts is not None and 0 <= (as_of_ts - source_ts).total_seconds() <= SOURCE_MAX_AGE_SECONDS_INT)
    row_dict["source_stale_bool"] = not fresh_bool
    row_dict["cycle_evidence_dict"] = source_dict.get("cycle_evidence_dict") or {}
    _apply_ack_evidence(row_dict, source_dict, as_of_ts=as_of_ts)
    cycle_dict = build_cycle_view_dict(row_dict, now_ts=as_of_ts)
    fresh_bool = fresh_bool and not cycle_dict["stale_bool"]
    step_list = cycle_dict["step_dict_list"]
    for step_dict in step_list:
        step_dict["class_str"] = _state_str(step_dict["state_str"])
        step_dict["tab_str"] = STEP_TAB_DICT[step_dict["label_str"]]
    # DB creation times prove Decide/Plan, not signal observation or submission.
    if fresh_bool:
        for index_int, source_key_str in ((1, "decision_dict"), (2, "vplan_dict")):
            if step_list[index_int]["state_str"] == "Done":
                time_str = _time_str((source_dict.get(source_key_str) or {}).get("created_timestamp_str"), as_of_ts)
                if time_str != "—":
                    step_list[index_int]["actual_time_str"] = time_str
        if step_list[3]["state_str"] == "Done" and source_dict.get("ack_list"):
            # This proves broker acknowledgement completion, not when the
            # request was transmitted. Order submitted timestamps are mutable.
            ack_time_list = [parse_timestamp_ts(item_dict["response_timestamp_str"]) for item_dict in source_dict["ack_list"]]
            submit_ts = parse_timestamp_ts(row_dict.get("latest_vplan_submission_timestamp_str"))
            step_list[3]["fact_str"] = "All orders acknowledged"
            # Legacy broker refresh may use the planned boundary as fallback.
            # Do not display that value as the actual acknowledgement time.
            if submit_ts is not None and all(ack_ts > submit_ts for ack_ts in ack_time_list):
                step_list[3].update(actual_time_str=_time_str(max(ack_time_list).isoformat(), as_of_ts), delta_str="", time_kind_str="ACK")
    failed_step_dict = next((step_dict for step_dict in step_list if step_dict["state_str"] == "Failed"), {})
    tab_str = tab_str or failed_step_dict.get("tab_str") or "plan"
    tables_dict = build_evidence_tables_dict(source_dict, as_of_ts=as_of_ts, fresh_bool=fresh_bool)
    selected_dict = source_dict.get("selected_cycle_dict") or {}
    historical_bool = bool(selected_dict) and not selected_dict.get("current_bool", False)
    saved_wording_bool = historical_bool and (source_dict.get("selected_explicit_bool", True) or not selected_dict.get("unresolved_bool", False))
    account_str = str(row_dict.get("account_route_str") or "")
    issue_bool = fresh_bool and cycle_dict["tone_str"] in {"red", "amber"}
    missing_ack_bool = (row_dict.get("missing_ack_count_int") or 0) > 0 or row_dict.get("latest_submit_ack_status_str") == "missing_critical"
    verdict_str = cycle_dict["now_str"] + "."
    verdict_detail_str = ("Next: " + cycle_dict["next_str"] + " " + cycle_dict["next_time_str"]).strip() if cycle_dict["next_str"] != "—" else ""
    issue_title_str = "Review broker ACK" if missing_ack_bool else cycle_dict["now_str"]
    issue_detail_str = "Check the order plan and the broker connection. Do not resubmit blindly." if missing_ack_bool else "Check the saved evidence before taking action."
    if issue_bool:
        verdict_detail_str = "Next: Review broker ACK" if missing_ack_bool else "Next: Check saved evidence"
    if saved_wording_bool:
        verdict_str = "Saved cycle: " + verdict_str
        verdict_detail_str = "Saved status for " + (selected_dict.get("session_date_str") or "this cycle") + "."
        issue_title_str = "Saved cycle · Missing broker ACK" if missing_ack_bool else "Saved cycle · " + cycle_dict["now_str"]
        issue_detail_str = "Review the saved records for " + (selected_dict.get("session_date_str") or "this cycle") + "."
    header_dict = {"state_str": pod_summary_dict["state_str"], "pill_str": pod_summary_dict["pill_str"],
        "verdict_str": pod_summary_dict["now_str"].rstrip(".") + ".", "detail_str": pod_summary_dict["now_detail_str"]}
    attention_dict = next((deepcopy(item_dict) for item_dict in overview_dict["attention_list"] if item_dict["pod_id_str"] == pod_id_str), {})
    cycle_state_str = _state_str(cycle_dict["tone_str"])
    if (source_dict.get("status_str") != "ok" and source_dict.get("selected_current_bool")
        and header_dict["state_str"] not in {"fail", "late"}):
        header_dict.update(state_str="unk", pill_str="Unknown", verdict_str="Current cycle unavailable.", detail_str="")
    # Selected current-cycle contradictions may weaken current status. A saved
    # historical result must never replace today's Pod warning or Idle state.
    if (selected_dict.get("current_bool") and STATE_RANK_DICT[cycle_state_str] < STATE_RANK_DICT[header_dict["state_str"]]
        and (header_dict["pill_str"] != "Idle" or cycle_state_str == "fail")):
        header_dict.update(state_str=cycle_state_str, pill_str=cycle_dict["pill_str"], verdict_str=verdict_str, detail_str=verdict_detail_str)
        if issue_bool and not attention_dict:
            attention_dict = {"state_str": cycle_state_str, "title_str": issue_title_str, "detail_str": issue_detail_str}
    return {
        **finance_dict, "pod_id_str": pod_id_str, "name_str": pod_summary_dict["name_str"],
        "account_str": (account_str[:1] + "···" + account_str[-3:]) if len(account_str) >= 4 else "—",
        "cadence_str": pod_summary_dict["cadence_str"], "pill_str": header_dict["pill_str"],
        "state_str": header_dict["state_str"], "header_dict": header_dict, "attention_dict": attention_dict,
        "cycle_state_str": cycle_state_str, "cycle_pill_str": cycle_dict["pill_str"], "verdict_str": verdict_str,
        "verdict_detail_str": verdict_detail_str,
        "issue_bool": issue_bool, "issue_title_str": issue_title_str, "issue_detail_str": issue_detail_str,
        "step_list": step_list, "tab_str": tab_str, "table_dict": tables_dict[tab_str], "tables_dict": tables_dict,
        "tab_list": [{"key_str": key_str, "label_str": label_str, "count_int": len(tables_dict[key_str]["row_list"]) if key_str in {"orders", "fills"} else None} for key_str, label_str in TAB_TUPLE if key_str != "files" or tables_dict["files"]["row_list"]],
        "cycle_list": source_dict.get("cycle_list") or [], "selected_cycle_dict": selected_dict,
        "history_truncated_bool": source_dict.get("history_truncated_bool", False),
        "historical_bool": historical_bool,
        "source_reason_str": source_dict.get("reason_str") or "", "fresh_bool": fresh_bool,
    }
