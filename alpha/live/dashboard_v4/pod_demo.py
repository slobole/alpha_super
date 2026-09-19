"""Explicit synthetic Pod evidence for the isolated demo server."""

from copy import deepcopy
from datetime import datetime, timedelta


def build_demo_pod_source_dict(row_list, pod_id_str, *, as_of_ts, decision_plan_id_int=None, vplan_id_int=None):
    matched_list = [row_dict for row_dict in row_list if row_dict["pod_id_str"] == pod_id_str]
    if len(matched_list) != 1 or (decision_plan_id_int or vplan_id_int or 2) not in {1, 2}:
        return {"status_str": "not_found", "reason_str": "Cycle not found"}
    row_dict = deepcopy(matched_list[0])
    selected_int = vplan_id_int or decision_plan_id_int or 2
    current_target_ts = datetime.fromisoformat(row_dict["latest_vplan_target_execution_timestamp_str"])
    previous_target_ts = current_target_ts - timedelta(days=4 if current_target_ts.day == 8 else 29)
    target_ts = current_target_ts if selected_int == 2 else previous_target_ts
    signal_ts = datetime.fromisoformat(row_dict["latest_decision_signal_timestamp_str"]) if selected_int == 2 else (
        target_ts - timedelta(days=3 if target_ts.weekday() == 0 else 1)).replace(hour=20, minute=0, second=0)
    submission_ts = target_ts - timedelta(seconds=390)
    issue_bool = row_dict.get("missing_ack_count_int", 0) > 0 and selected_int == 2
    status_str = "submitted" if issue_bool else "completed"
    row_dict.update(latest_decision_plan_id_int=selected_int, latest_vplan_id_int=selected_int,
        latest_vplan_decision_plan_id_int=selected_int, latest_vplan_is_for_latest_decision_bool=True,
        latest_decision_plan_status_str=status_str, latest_vplan_status_str=status_str,
        latest_vplan_submission_timestamp_str=submission_ts.isoformat(), latest_vplan_target_execution_timestamp_str=target_ts.isoformat(),
        as_of_timestamp_str=as_of_ts.isoformat(), reason_code_str="cycle_completed", next_action_str="wait",
        latest_submit_ack_status_str="missing_critical" if issue_bool else "complete",
        broker_order_count_int=3, broker_ack_count_int=2 if issue_bool else 3, missing_ack_count_int=1 if issue_bool else 0,
        fill_count_int=2 if issue_bool else 3, latest_reconciliation_status_str="" if issue_bool else "passed",
        latest_reconciliation_timestamp_str=None if issue_bool else (target_ts + timedelta(minutes=6, seconds=12)).isoformat())
    row_dict["norgate_snapshot_status_dict"] = {"status_str": "ready", "snapshot_date_str": signal_ts.date().isoformat(), "snapshot_fresh_for_cycle_bool": True}
    if selected_int == 1 or target_ts.date() < as_of_ts.date():
        eod_ts = target_ts.replace(hour=20, minute=10, second=1)
        row_dict["eod_snapshot_dict"].update(status_str="completed", same_session_bool=True,
            expected_market_date_str=target_ts.date().isoformat(), latest_market_date_str=target_ts.date().isoformat(),
            expected_due_timestamp_str=eod_ts.replace(second=0).isoformat(), latest_timestamp_str=eod_ts.isoformat(), source_str="broker")
    plan_list, order_list, ack_list, fill_list, proof_list, event_list = [], [], [], [], [], []
    before_map, after_map = {"AMD": 0, "CRM": 0, "DIS": 44}, {"AMD": 31, "CRM": 17, "DIS": 0}
    for index_int, (symbol_str, amount_float, price_float) in enumerate((("AMD", 31.0, 158.42), ("CRM", 17.0, 291.05), ("DIS", -44.0, 112.80))):
        request_str, order_id_str = f"demo-{selected_int}-{index_int}", f"11842203{selected_int}{index_int}"
        filled_bool = not issue_bool or index_int != 1
        fill_ts = target_ts + timedelta(seconds=index_int)
        plan_list.append({"asset_str": symbol_str, "current_share_float": before_map[symbol_str],
            "target_share_float": after_map[symbol_str], "order_delta_share_float": amount_float, "order_request_key_str": request_str})
        order_list.append({"asset_str": symbol_str, "broker_order_id_str": order_id_str, "order_request_key_str": request_str, "amount_float": amount_float})
        ack_list.append({"asset_str": symbol_str, "order_request_key_str": request_str, "broker_order_id_str": order_id_str,
            "broker_response_ack_bool": filled_bool, "ack_status_str": "broker_acked" if filled_bool else "missing_critical",
            "ack_source_str": "open order" if filled_bool else "—", "response_timestamp_str": (submission_ts + timedelta(seconds=1)).isoformat() if filled_bool else None})
        if filled_bool:
            fill_list.append({"asset_str": symbol_str, "broker_order_id_str": order_id_str, "fill_amount_float": amount_float,
                "fill_price_float": price_float, "fill_timestamp_str": fill_ts.isoformat()})
        proof_list.append({"broker_order_id_str": order_id_str, "asset_str": symbol_str, "requested_share_float": amount_float,
            "filled_share_float": amount_float if filled_bool else 0.0, "complete_bool": filled_bool})
    for event_str, timestamp_ts in (("build_vplan_created", submission_ts - timedelta(minutes=2)),
                                  ("submit_vplan_completed", submission_ts + timedelta(seconds=1)),
                                  ("submit_vplan_missing_broker_ack" if issue_bool else "post_execution_reconcile_completed", target_ts + timedelta(minutes=6, seconds=12))):
        event_list.append({"event_type_str": event_str, "timestamp_str": timestamp_ts.isoformat()})
    cycle_list = [{"cycle_key_str": f"vplan:{index_int}", "decision_plan_id_int": index_int, "vplan_id_int": index_int,
        "session_date_str": timestamp_ts.date().isoformat(), "execution_policy_str": row_dict["execution_policy_str"], "current_bool": index_int == 2,
        "target_execution_timestamp_str": timestamp_ts.isoformat()} for index_int, timestamp_ts in ((2, current_target_ts), (1, previous_target_ts))]
    proof_dict = {"state_str": "partial" if issue_bool else "complete", "pod_id_str": pod_id_str,
        "account_route_str": row_dict["account_route_str"], "vplan_id_int": selected_int, "decision_plan_id_int": selected_int,
        "vplan_status_str": status_str, "order_count_int": 3, "filled_order_count_int": 2 if issue_bool else 3,
        "actual_fill_timestamp_str": None if issue_bool else (target_ts + timedelta(seconds=2)).isoformat(), "order_list": proof_list}
    return {"status_str": "ok", "reason_str": "", "pod_row_dict": row_dict, "cycle_list": cycle_list,
        "selected_cycle_dict": next(item_dict for item_dict in cycle_list if item_dict["vplan_id_int"] == selected_int),
        "cycle_evidence_dict": proof_dict, "history_truncated_bool": False,
        "decision_dict": {"created_timestamp_str": (signal_ts + timedelta(hours=2)).isoformat(),
            "display_target_weight_map_dict": {"AMD": .10, "CRM": .10}, "exit_asset_list": ["DIS"], "decision_book_type_str": "incremental_entry_exit_book"},
        "vplan_dict": {"created_timestamp_str": (submission_ts - timedelta(minutes=2)).isoformat(), "current_broker_position_map_dict": before_map},
        "plan_row_list": plan_list, "order_list": order_list, "ack_list": ack_list, "fill_list": fill_list,
        "reconciliation_dict": {} if issue_bool else {"status_str": "passed", "model_position_map_dict": after_map, "broker_position_map_dict": after_map},
        "event_list": list(reversed(event_list)), "file_list": []}
