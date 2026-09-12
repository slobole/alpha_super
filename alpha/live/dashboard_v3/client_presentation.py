"""Display projections of scoped, saved facts. Never used by trading/accounting."""

import math
import json
from datetime import UTC, datetime
from zoneinfo import ZoneInfo

from alpha.live.ops_report import parse_timestamp_ts
from alpha.live.dashboard_v3.operator_tools import redact_diagnostic_value


ALLOCATION_COLOR_LIST = ["#5274a4", "#6c9a91", "#b69b73", "#8c82aa", "#789aaa", "#ac8790"]


def allocation_dict(item_list, *, date_str="", basis_str=""):
    """Nonnegative composition: weight_i=value_i/sum(value). No exclusions."""
    result_dict = {"item_list": [], "date_str": date_str, "basis_str": basis_str,
                   "total_float": None, "reason_str": "Allocation unavailable"}
    if not item_list or any(type(item_dict.get("value_float")) not in {int, float}
            or not math.isfinite(item_dict["value_float"]) or item_dict["value_float"] < 0 for item_dict in item_list):
        return result_dict
    total_float = sum(item_dict["value_float"] for item_dict in item_list)
    if total_float <= 0 or not math.isfinite(total_float):
        return result_dict
    offset_float = 0.0
    for index_int, item_dict in enumerate(item_list):
        weight_float = item_dict["value_float"] / total_float
        result_dict["item_list"].append({**item_dict, "weight_float": weight_float,
            "offset_float": offset_float, "color_str": ALLOCATION_COLOR_LIST[index_int % len(ALLOCATION_COLOR_LIST)]})
        offset_float += weight_float * 100
    result_dict.update(total_float=total_float, reason_str="")
    return result_dict


def portfolio_account_list(report_dict):
    """The same end-date ownership scope for NAV and cash display."""
    date_str = report_dict.get("closing_date_str") or ""
    if "valuation_account_list" in report_dict:
        return report_dict["valuation_account_list"]
    return [{"account_route": item_dict["account_route_str"], "pod_id": item_dict.get("pod_id_str"),
             "display_name": item_dict["display_name_str"]}
        for item_dict in report_dict["strategy_list"] if item_dict["to_date_str"] == date_str]


def portfolio_allocation_dict(report_dict, snapshot_obj, *, cash_snapshot_list=()):
    """Use one valuation source for the whole ring; never change accounting.

    With complete EOD equity: cash_share_i = cash_i / sum(EOD_equity_i),
    invested_share_i = (EOD_equity_i-cash_i) / sum(EOD_equity_i).
    Otherwise retain the entire finalized NAV ring with cash unknown.
    """
    date_str = report_dict.get("closing_date_str") or ""
    account_list = portfolio_account_list(report_dict)
    item_list = []
    for account_dict in account_list:
        # *** CRITICAL *** retrospective same-date display only. Never borrow
        # an earlier NAV, mix accounts, or use performance start as NAV scope.
        row_list = [row_obj for row_obj in snapshot_obj.row_tuple
            if row_obj.account_route_str == account_dict["account_route"] and row_obj.market_date_str == date_str]
        item_list.append({"label_str": account_dict.get("display_name") or account_dict.get("pod_id") or account_dict["account_route"],
            "detail_str": account_dict["account_route"],
            "value_float": float(row_list[0].closing_nav_decimal) if len(row_list) == 1 else None})
    result_dict = allocation_dict(item_list, date_str=date_str, basis_str="Finalized IBKR account value")
    route_list = [item_dict["account_route"] for item_dict in account_list]
    expected_float = report_dict.get("closing_nav_float")
    scope_complete_bool = True if "valuation_account_list" in report_dict else report_dict.get("scope_complete_bool", False)
    if (len(set(route_list)) != len(route_list) or expected_float is None or not scope_complete_bool
            or result_dict["total_float"] is None or abs(result_dict["total_float"] - expected_float) > .01):
        result_dict.update(item_list=[], reason_str="Complete, matching account values are required.")
    result_dict.update(source_str="flex_nav", cash_weight_float=None, cash_complete_bool=False)
    if not result_dict["item_list"]:
        return result_dict
    matched_cash_list = []
    for account_dict in account_list:
        cash_list = [cash_dict for cash_dict in cash_snapshot_list
            if cash_dict.get("account_route_str") == account_dict["account_route"]
            and cash_dict.get("pod_id_str") == account_dict.get("pod_id")
            and cash_dict.get("market_date_str") == date_str]
        matched_cash_list.append(cash_list[0] if len(cash_list) == 1 else {})
    # *** CRITICAL *** retrospective display basis: broker cash and equity must
    # come from the same saved EOD row. Never mix any sleeve with Flex NAV.
    eod_dict = allocation_dict([{**item_dict, "value_float": cash_dict.get("equity_float")}
        for item_dict, cash_dict in zip(item_list, matched_cash_list)],
        date_str=date_str, basis_str="Saved broker end-of-day snapshot; financial totals use finalized Flex")
    if eod_dict["item_list"]:
        result_dict = {**eod_dict, "source_str": "broker_eod", "cash_weight_float": None, "cash_complete_bool": False}
    for item_dict, cash_dict in zip(result_dict["item_list"], matched_cash_list):
        item_dict["show_identity_bool"] = sum(candidate_dict["label_str"] == item_dict["label_str"] for candidate_dict in result_dict["item_list"]) > 1
        cash_float = cash_dict.get("cash_float")
        cash_valid_bool = (result_dict["source_str"] == "broker_eod"
            and type(cash_float) in {int, float} and math.isfinite(cash_float)
            and 0 <= cash_float <= item_dict["value_float"])
        item_dict.update(cash_float=cash_float if cash_valid_bool else None,
            cash_weight_float=cash_float / result_dict["total_float"] if cash_valid_bool else None,
            invested_weight_float=(item_dict["value_float"] - cash_float) / result_dict["total_float"] if cash_valid_bool else None)
    if result_dict["item_list"] and all(item_dict["cash_float"] is not None for item_dict in result_dict["item_list"]):
        result_dict.update(cash_complete_bool=True, cash_weight_float=sum(item_dict["cash_weight_float"] for item_dict in result_dict["item_list"]))
    return result_dict


def holdings_allocation_dict(evidence_dict):
    """Reference composition only: abs(shares*reference price)/gross priced value.

    Cash and unpriced holdings cannot be inferred. Signed values remain visible.
    This deliberately does not claim NAV weights or synchronized current marks.
    """
    item_list, missing_list = [], []
    for position_dict in evidence_dict.get("position_exposure_dict_list") or []:
        share_float = position_dict.get("share_float")
        price_float = position_dict.get("price_float")
        if type(share_float) not in {int, float} or not math.isfinite(share_float):
            missing_list.append(position_dict.get("asset_str") or "Unknown")
            continue
        if share_float == 0:
            continue
        if type(price_float) not in {int, float} or not math.isfinite(price_float) or price_float <= 0 or not math.isfinite(share_float * price_float):
            missing_list.append(position_dict.get("asset_str") or "Unknown")
            continue
        signed_float = share_float * price_float
        item_list.append({"label_str": position_dict.get("asset_str") or "Unknown", "value_float": abs(signed_float),
            "signed_value_float": signed_float, "detail_str": "Short" if signed_float < 0 else "Long"})
    result_dict = allocation_dict(item_list, basis_str="Share of priced holdings · reference values · excludes cash")
    result_dict.update(missing_list=missing_list, position_timestamp_str=evidence_dict.get("latest_pod_state_timestamp_str"),
        price_timestamp_str=evidence_dict.get("latest_live_reference_snapshot_timestamp_str"))
    return result_dict


def flow_dict(evidence_dict, *, source_fresh_bool):
    """Focus follows an explicit saved action or failed stage, never first gray."""
    step_list = [dict(step_dict) for step_dict in evidence_dict.get("lifecycle_step_dict_list") or []
        if step_dict.get("step_key_str") != "diff" and step_dict.get("label_str") != "Live vs Backtest"]
    previous_bool = evidence_dict.get("latest_vplan_is_for_latest_decision_bool") is False or evidence_dict.get("latest_vplan_cycle_role_str") in {"previous", "previous_cycle"}
    action_str = evidence_dict.get("next_action_str") or "unknown"
    focus_str = {"build_decision_plan": "decision", "build_vplan": "vplan", "review_vplan": "vplan",
        "submit_vplan": "ack", "post_execution_reconcile": "reconcile", "no_db": "db",
        "expire_stale": "decision", "missed_decision_cycle": "decision"}.get(action_str)
    failed_list = [step_dict for step_dict in step_list if step_dict.get("severity_str") == "red"]
    if failed_list:
        focus_str = failed_list[0].get("step_key_str")
    required_label_str = (evidence_dict.get("required_action_dict") or {}).get("label_str")
    required_focus_str = {"Review broker ACK": "ack", "Review reconcile": "reconcile",
        "Review Norgate data": "decision", "Wait Norgate data": "decision",
        "Review DecisionPlan gate": "decision", "Wait submission window": "vplan"}.get(required_label_str)
    if required_focus_str:
        focus_str = required_focus_str
    if not source_fresh_bool:
        focus_str = None
    for step_dict in step_list:
        step_dict["previous_bool"] = previous_bool and step_dict.get("step_key_str") in {"vplan", "ack", "fill", "reconcile"}
        step_dict["focus_bool"] = bool(focus_str) and step_dict.get("step_key_str") == focus_str
    return {"step_list": step_list, "previous_bool": previous_bool,
        "action_str": action_str, "source_fresh_bool": source_fresh_bool,
        "headline_str": (evidence_dict.get("required_action_dict") or {}).get("label_str") or action_str.replace("_", " ").capitalize()}


def saved_stage_table_dict(detail_dict, strategy_dict, *, local_workspace_bool=False):
    """Allowlisted detail tables; raw SQL payloads never enter the client view."""
    evidence_dict = strategy_dict["evidence_dict"]
    row_dict = detail_dict.get("pod_row_dict") or {}
    for field_str in ("pod_id_str", "account_route_str", "mode_str", "release_id_str", "latest_pod_state_timestamp_str",
                      "latest_decision_plan_id_int", "latest_vplan_id_int", "latest_decision_plan_status_str",
                      "latest_vplan_status_str", "latest_reconciliation_status_str", "latest_reconciliation_timestamp_str",
                      "latest_submit_ack_status_str", "missing_ack_count_int", "broker_ack_count_int", "fill_count_int"):
        if row_dict.get(field_str) != evidence_dict.get(field_str):
            return {}
    if row_dict.get("mode_str") != "live":
        return {}
    table_dict = {}

    def owned_plan_bool(plan_dict, id_field_str, expected_id_obj):
        if not plan_dict or expected_id_obj is None or plan_dict.get(id_field_str) != expected_id_obj:
            return False
        if any(plan_dict.get(field_str) != evidence_dict.get(field_str) for field_str in ("pod_id_str", "account_route_str", "release_id_str")):
            return False
        signal_ts = parse_timestamp_ts(plan_dict.get("signal_timestamp_str") or "")
        if signal_ts is None or signal_ts > datetime.now(UTC):
            return False
        return local_workspace_bool or signal_ts.astimezone(ZoneInfo("America/New_York")).date().isoformat() >= (strategy_dict.get("effective_from_str") or "9999-12-31")

    def scalar_obj(value_obj):
        if type(value_obj) in {float, int}:
            return value_obj if math.isfinite(value_obj) else None
        return redact_diagnostic_value(value_obj) if isinstance(value_obj, str) else None

    def project_table_dict(label_str, column_tuple, row_list):
        return {"label_str": label_str, "column_list": [label_str for _, label_str in column_tuple],
            "row_list": [[scalar_obj(row_dict.get(field_str)) for field_str, _ in column_tuple] for row_dict in row_list]}

    decision_dict = detail_dict.get("latest_decision_plan_dict") or {}
    if (owned_plan_bool(decision_dict, "decision_plan_id_int", evidence_dict.get("latest_decision_plan_id_int"))
            and decision_dict.get("status_str") == evidence_dict.get("latest_decision_plan_status_str")):
        target_map_dict = decision_dict.get("display_target_weight_map_dict") or {}
        target_list = [{"asset_str": asset_str, "weight_float": weight_obj * 100 if type(weight_obj) in {int, float} else None}
            for asset_str, weight_obj in target_map_dict.items()]
        book_label_str = {"full_target_weight_book": "Full portfolio targets", "incremental_entry_exit_book": "Entry and exit targets"}.get(decision_dict.get("decision_book_type_str"), "Saved targets")
        table_dict["decision"] = [project_table_dict(book_label_str,
            (("asset_str", "Asset"), ("weight_float", "Target · %")), target_list),
            project_table_dict("Planned exits", (("asset_str", "Asset"),), [{"asset_str": asset_str} for asset_str in decision_dict.get("exit_asset_list") or []])]
    vplan_dict = detail_dict.get("latest_vplan_dict") or {}
    vplan_consistent_bool = (vplan_dict.get("status_str") == evidence_dict.get("latest_vplan_status_str")
        and vplan_dict.get("submit_ack_status_str") == evidence_dict.get("latest_submit_ack_status_str")
        and len(vplan_dict.get("fill_row_dict_list") or []) == evidence_dict.get("fill_count_int")
        and len(vplan_dict.get("broker_ack_row_dict_list") or []) == evidence_dict.get("broker_ack_count_int"))
    if owned_plan_bool(vplan_dict, "vplan_id_int", evidence_dict.get("latest_vplan_id_int")) and vplan_consistent_bool:
        order_list = [item_dict for item_dict in vplan_dict.get("vplan_row_dict_list") or []
            if isinstance(item_dict, dict) and item_dict.get("vplan_id_int") == vplan_dict["vplan_id_int"]]
        table_dict["vplan"] = [project_table_dict("Saved order plan", (("asset_str", "Asset"), ("current_share_float", "Current shares"),
            ("target_share_float", "Target shares"), ("order_delta_share_float", "Order shares"), ("live_reference_price_float", "Reference price")), order_list)]
        for step_str, list_str, label_str, column_tuple in (
            ("ack", "broker_ack_row_dict_list", "Broker acknowledgements", (("asset_str", "Asset"), ("ack_status_str", "Status"), ("ack_source_str", "Source"), ("response_timestamp_str", "Recorded time"))),
            ("fill", "fill_row_dict_list", "Recorded fills · does not imply all orders filled", (("asset_str", "Asset"), ("fill_amount_float", "Shares"), ("fill_price_float", "Fill price"), ("fill_timestamp_str", "Fill time"))),
        ):
            child_list = [item_dict for item_dict in vplan_dict.get(list_str) or [] if isinstance(item_dict, dict)
                and item_dict.get("vplan_id_int") == vplan_dict["vplan_id_int"]
                and item_dict.get("account_route_str") == evidence_dict.get("account_route_str")]
            table_dict[step_str] = [project_table_dict(label_str, column_tuple, child_list)]
        reconciliation_dict = detail_dict.get("latest_reconciliation_dict") or {}
        if (reconciliation_dict.get("pod_id_str") == evidence_dict.get("pod_id_str")
                and reconciliation_dict.get("vplan_id_int") == vplan_dict["vplan_id_int"]
                and reconciliation_dict.get("created_timestamp_str") == evidence_dict.get("latest_reconciliation_timestamp_str")
                and reconciliation_dict.get("status_str") == evidence_dict.get("latest_reconciliation_status_str")):
            model_dict = json.loads(reconciliation_dict.get("model_position_json_str") or "null")
            broker_dict = json.loads(reconciliation_dict.get("broker_position_json_str") or "null")
            if isinstance(model_dict, dict) and isinstance(broker_dict, dict):
                # Complete persisted position maps: absent symbol means zero in
                # that recorded snapshot, never a fallback to current cache.
                position_list = [{"asset_str": asset_str, "model_float": model_dict.get(asset_str, 0), "broker_float": broker_dict.get(asset_str, 0)}
                    for asset_str in sorted(set(model_dict) | set(broker_dict))]
                table_dict["reconcile"] = [project_table_dict("Recorded reconciliation · " + str(reconciliation_dict.get("stage_str") or "Unknown stage"),
                    (("asset_str", "Asset"), ("model_float", "Model shares"), ("broker_float", "Broker shares")), position_list)]
    return table_dict
