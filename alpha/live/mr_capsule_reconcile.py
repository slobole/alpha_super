"""Capsule settlement from broker evidence; original targets remain immutable."""
from dataclasses import replace
from datetime import datetime
import math

from alpha.live.execution_engine import build_broker_order_request_list_from_vplan
from alpha.live.reconcile import reconcile_account_state

TERMINAL_SHORTFALL_STATUS_SET = {"Cancelled", "ApiCancelled", "Rejected", "Expired", "Inactive"}


def classify_capsule_execution(vplan_obj, broker_snapshot_obj, order_row_list, fill_row_list,
        tolerance_float=1e-9, supplemental_request_list=(), resolved_request_dict=None):
    """Expected actual shares = pre-submit shares + correlated signed fills.

    A terminal residual is accepted, including parking sales. Missing/open evidence
    never authorizes a new order. Supplemental requests were durably recorded by
    the recovery path before submission, or adopted with manual-fill evidence.
    """
    original_request_list = build_broker_order_request_list_from_vplan(vplan_obj)
    request_list = [*original_request_list, *supplemental_request_list]
    expected_position_dict = dict(vplan_obj.current_broker_position_map)
    invalid_account_bool = (
        broker_snapshot_obj.account_route_str != vplan_obj.account_route_str
        or broker_snapshot_obj.snapshot_timestamp_ts < vplan_obj.target_execution_timestamp_ts
        or not math.isfinite(float(broker_snapshot_obj.cash_float))
        or not math.isfinite(float(broker_snapshot_obj.net_liq_float))
        or broker_snapshot_obj.net_liq_float <= 0
        or any(not math.isfinite(float(amount_float)) or float(amount_float) < 0
               for amount_float in [*expected_position_dict.values(), *broker_snapshot_obj.position_amount_map.values()])
    )
    # A known pending request blocks its own asset, not an unrelated verified
    # exit. Unknown order identities can affect any holding and block all sales.
    known_order_id_set = {str(row_dict["broker_order_id_str"]) for row_dict in order_row_list}
    unknown_open_order_bool = bool(set(map(str, broker_snapshot_obj.open_order_id_list)) - known_order_id_set)
    global_uncertainty_bool = invalid_account_bool or unknown_open_order_bool
    uncertain_bool = global_uncertainty_bool or bool(broker_snapshot_obj.open_order_id_list)
    matched_order_id_set = set()
    filled_by_asset_dict = {}
    status_by_asset_dict = {}
    uncertain_asset_set = set()
    for request_obj in request_list:
        matching_order_list = [row_dict for row_dict in order_row_list
            if row_dict.get("order_request_key_str") == request_obj.order_request_key_str]
        resolution_dict = (resolved_request_dict or {}).get(request_obj.order_request_key_str, {})
        if (not matching_order_list and resolution_dict.get("asset_str") == request_obj.asset_str
                and resolution_dict.get("resolution_str") in {"never_sent", "never_dispatched"}):
            evidence_dict = resolution_dict.get("evidence_dict") or {}
            proof_timestamp_str = evidence_dict.get("refreshed_timestamp_str", resolution_dict.get("created_timestamp_str"))
            if proof_timestamp_str and broker_snapshot_obj.snapshot_timestamp_ts >= datetime.fromisoformat(proof_timestamp_str):
                status_by_asset_dict[request_obj.asset_str] = (resolution_dict["resolution_str"], "")
                continue
        if len(matching_order_list) != 1:
            uncertain_bool = True
            uncertain_asset_set.add(request_obj.asset_str)
            continue
        order_dict = matching_order_list[0]
        order_id_str = str(order_dict["broker_order_id_str"])
        duplicate_bool = order_id_str in matched_order_id_set
        matched_order_id_set.add(order_id_str)
        payload_dict = order_dict.get("raw_payload_dict") or {}
        matching_fill_list = [row_dict for row_dict in fill_row_list
            if str(row_dict["broker_order_id_str"]) == order_id_str]
        requested_float = float(request_obj.amount_float)
        filled_float = sum(float(row_dict["fill_amount_float"]) for row_dict in matching_fill_list)
        reported_filled_float = float(order_dict["filled_amount_float"])
        invalid_bool = (
            duplicate_bool or order_dict["asset_str"] != request_obj.asset_str
            or order_dict.get("account_route_str") != vplan_obj.account_route_str
            or order_dict.get("unit_str") != "shares"
            or not math.isfinite(float(order_dict["amount_float"]))
            or abs(float(order_dict["amount_float"]) - requested_float) > tolerance_float
            or payload_dict.get("open_order_observed_bool") is True
            or order_id_str in set(map(str, broker_snapshot_obj.open_order_id_list))
            or payload_dict.get("snapshot_source_str") == "open_order"
            or payload_dict.get("completed_quantity_verified_bool") is False
            or not math.isfinite(filled_float) or not math.isfinite(reported_filled_float)
            or any(row_dict["asset_str"] != request_obj.asset_str
                or row_dict.get("account_route_str") != vplan_obj.account_route_str
                or not math.isfinite(float(row_dict["fill_amount_float"]))
                or float(row_dict["fill_amount_float"]) * requested_float < 0
                or datetime.fromisoformat(row_dict["fill_timestamp_str"]) > broker_snapshot_obj.snapshot_timestamp_ts
                for row_dict in matching_fill_list)
            or filled_float * requested_float < 0
            or abs(filled_float) > abs(requested_float) + tolerance_float
            or abs(abs(filled_float) - abs(reported_filled_float)) > tolerance_float
        )
        residual_float = requested_float - filled_float
        if abs(residual_float) > tolerance_float and (
            order_dict["status_str"] not in TERMINAL_SHORTFALL_STATUS_SET
            or payload_dict.get("snapshot_source_str") != "completed_order"
        ):
            invalid_bool = True
        if invalid_bool:
            uncertain_bool = True
            uncertain_asset_set.add(request_obj.asset_str)
        # Keep observed fills visible in alerts, even when another item is unresolved.
        if math.isfinite(filled_float):
            expected_position_dict[request_obj.asset_str] = float(expected_position_dict.get(request_obj.asset_str, 0)) + filled_float
            filled_by_asset_dict[request_obj.asset_str] = filled_by_asset_dict.get(request_obj.asset_str, 0.0) + filled_float
        status_by_asset_dict[request_obj.asset_str] = (order_dict["status_str"], order_id_str)
    if len(matched_order_id_set) != len(order_row_list) or any(
            str(row_dict["broker_order_id_str"]) not in matched_order_id_set for row_dict in fill_row_list):
        uncertain_bool = True
        global_uncertainty_bool = True
    reconciliation_obj = reconcile_account_state(
        model_position_map=expected_position_dict, model_cash_float=broker_snapshot_obj.cash_float,
        broker_snapshot_obj=broker_snapshot_obj, tolerance_float=tolerance_float)
    # Pending fills on a known other asset cannot change this exit's quantity.
    # An unexplained change outside those pending assets remains account-wide
    # uncertainty and requires manual review before any completion sale.
    unexplained_holdings_bool = bool(set(reconciliation_obj.mismatch_dict) - uncertain_asset_set)
    outcome_str = "awaiting_evidence" if uncertain_bool else (
        "unexplained_positions" if not reconciliation_obj.passed_bool else "completed")
    residual_row_list = []
    for request_obj in original_request_list:
        filled_float = filled_by_asset_dict.get(request_obj.asset_str, 0.0)
        residual_float = request_obj.amount_float - filled_float
        uncertain_asset_bool = (global_uncertainty_bool or request_obj.asset_str in uncertain_asset_set
            or request_obj.asset_str in reconciliation_obj.mismatch_dict)
        if abs(residual_float) <= tolerance_float and not uncertain_asset_bool:
            continue
        status_str, order_id_str = status_by_asset_dict.get(request_obj.asset_str, ("Unknown", ""))
        action_str = "VERIFY" if uncertain_asset_bool or unexplained_holdings_bool else (
            "SELL" if residual_float < 0 else "NONE")
        residual_row_list.append({"asset_str": request_obj.asset_str,
            "requested_amount_float": request_obj.amount_float, "filled_amount_float": filled_float,
            "residual_amount_float": residual_float, "status_str": status_str,
            "broker_order_id_str": order_id_str, "order_request_key_str": request_obj.order_request_key_str,
            "required_action_str": action_str,
            "reason_str": "Verify broker evidence before trading" if action_str == "VERIFY" else (
                "Sale remainder requires attention" if action_str == "SELL" else "Missed buy; no replacement required")})
    alerted_asset_set = {row_dict["asset_str"] for row_dict in residual_row_list}
    for asset_str, mismatch_dict in reconciliation_obj.mismatch_dict.items():
        if asset_str in alerted_asset_set:
            continue
        expected_float = mismatch_dict["model_amount_float"]
        actual_float = mismatch_dict["broker_amount_float"]
        residual_row_list.append({"asset_str": asset_str, "requested_amount_float": expected_float,
            "filled_amount_float": actual_float, "residual_amount_float": expected_float - actual_float,
            "status_str": "UnexplainedHoldings", "broker_order_id_str": "", "order_request_key_str": "",
            "required_action_str": "VERIFY", "reason_str": "Holdings differ from accounted fills; verify before trading"})
    invalid_position_asset_set = {asset_str for asset_str in set(expected_position_dict) | set(broker_snapshot_obj.position_amount_map)
        if not math.isfinite(float(expected_position_dict.get(asset_str, 0.0)))
        or not math.isfinite(float(broker_snapshot_obj.position_amount_map.get(asset_str, 0.0)))}
    residual_row_list = [row_dict for row_dict in residual_row_list if row_dict["asset_str"] not in invalid_position_asset_set]
    for asset_str in sorted(invalid_position_asset_set):
        residual_row_list.append({"asset_str": asset_str, "requested_amount_float": None,
            "filled_amount_float": None, "residual_amount_float": None,
            "status_str": "InvalidHoldings", "broker_order_id_str": "", "order_request_key_str": "",
            "required_action_str": "VERIFY", "reason_str": "Broker share quantity unavailable or non-finite; refresh before trading"})
    if outcome_str == "completed" and residual_row_list:
        outcome_str = "accepted_residual"
    if outcome_str not in {"accepted_residual", "completed"}:
        reconciliation_obj = replace(reconciliation_obj, passed_bool=False, status_str="blocked",
            mismatch_dict={**reconciliation_obj.mismatch_dict,
                "mr_capsule_incomplete_order_evidence_bool": uncertain_bool,
                "mr_capsule_execution_outcome_str": outcome_str})
    return reconciliation_obj, outcome_str, residual_row_list
