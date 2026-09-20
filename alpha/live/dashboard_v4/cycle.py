"""Read-only, LIVE-only cycle presentation from Dashboard V3 evidence.

Summary timestamps for submission and execution are plans, not observations.
Fill rows are executions (including partial fills), not filled-order counts.
Missing detail therefore stays unknown rather than manufacturing completion.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from alpha.live.dashboard_v3.client_operations import SOURCE_MAX_AGE_SECONDS_INT
from alpha.live.dashboard_v3.filters import MARKET_TIMEZONE_OBJ
from alpha.live.scheduler_service import (
    DEFAULT_ACTIVE_POLL_SECONDS_INT,
    DEFAULT_RECONCILE_GRACE_SECONDS_INT,
    DEFAULT_SUBMIT_STUCK_SECONDS_INT,
)


STEP_LABEL_TUPLE = ("Data", "Decide", "Plan", "Submit", "Fill", "Reconcile", "EOD")


def _timestamp_dt(value_obj: Any) -> datetime | None:
    if value_obj in (None, ""):
        return None
    try:
        timestamp_dt = datetime.fromisoformat(str(value_obj).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if timestamp_dt.tzinfo is None:
        timestamp_dt = timestamp_dt.replace(tzinfo=UTC)
    return timestamp_dt.astimezone(UTC)


def _clock_str(timestamp_dt: datetime | None, now_dt: datetime) -> str:
    if timestamp_dt is None:
        return ""
    market_dt = timestamp_dt.astimezone(MARKET_TIMEZONE_OBJ)
    same_day_bool = market_dt.date() == now_dt.astimezone(MARKET_TIMEZONE_OBJ).date()
    return market_dt.strftime("%H:%M:%S" if same_day_bool else "%m-%d %H:%M:%S")


def _step_dict(
    label_str: str,
    state_str: str,
    fact_str: str,
    now_dt: datetime,
    *,
    actual_dt: datetime | None = None,
    planned_dt: datetime | None = None,
) -> dict[str, str]:
    delta_str = ""
    if actual_dt is not None and actual_dt > now_dt:
        state_str, fact_str, actual_dt = "Unknown", "Future evidence", None
    if actual_dt is not None and planned_dt is not None:
        delta_str = f"{int((actual_dt - planned_dt).total_seconds()):+d}s"
    return {
        "key_str": label_str.lower(),
        "label_str": label_str,
        "state_str": state_str,
        "fact_str": fact_str,
        "actual_time_str": _clock_str(actual_dt, now_dt),
        "planned_time_str": _clock_str(planned_dt, now_dt),
        "planned_timestamp_str": planned_dt.isoformat() if planned_dt is not None else "",
        "delta_str": delta_str,
    }


def _due_state_str(planned_dt: datetime | None, now_dt: datetime,
                   source_dt: datetime, *, submit_bool: bool = False) -> str:
    if planned_dt is None or now_dt < planned_dt:
        return "Planned"
    allowance_int = max(DEFAULT_ACTIVE_POLL_SECONDS_INT, DEFAULT_SUBMIT_STUCK_SECONDS_INT) if submit_bool else DEFAULT_ACTIVE_POLL_SECONDS_INT
    late_dt = planned_dt + timedelta(seconds=allowance_int)
    # Eligibility is not a completion deadline. A cached pre-deadline summary
    # cannot prove the scheduler missed its next opportunity to do the work.
    return "Late" if min(now_dt, source_dt) > late_dt else "Now"


def _count_int(value_obj: Any) -> int | None:
    try:
        count_int = int(value_obj)
    except (TypeError, ValueError):
        return None
    return count_int if count_int >= 0 else None


def _eod_step_dict(eod_dict: dict[str, Any], now_dt: datetime, source_dt: datetime) -> dict[str, str]:
    status_str = str(eod_dict.get("status_str") or "")
    planned_dt = _timestamp_dt(eod_dict.get("expected_due_timestamp_str"))
    actual_dt = None
    if status_str == "completed" and eod_dict.get("same_session_bool") is True:
        expected_date_str = eod_dict.get("expected_market_date_str")
        actual_dt = _timestamp_dt(eod_dict.get("latest_timestamp_str"))
        same_date_bool = (
            expected_date_str is not None
            and eod_dict.get("latest_market_date_str") == expected_date_str
            and actual_dt is not None
            and actual_dt.astimezone(MARKET_TIMEZONE_OBJ).date().isoformat() == expected_date_str
        )
        valid_bool = same_date_bool and planned_dt is not None and actual_dt >= planned_dt and eod_dict.get("source_str") == "broker"
        state_str, fact_str = ("Done", "Snapshot saved") if valid_bool else ("Unknown", "Check EOD evidence")
        if not valid_bool:
            actual_dt = None
    elif status_str in {"waiting", "due_missing", "blocked_by_execution"}:
        state_str = _due_state_str(planned_dt, now_dt, source_dt) if planned_dt else "Unknown"
        fact_str = "Waiting for reconcile" if status_str == "blocked_by_execution" else "Snapshot due"
        last_required_str = eod_dict.get("last_required_market_date_str")
        if status_str == "due_missing" and eod_dict.get("last_required_eod_present_bool") is False and last_required_str:
            fact_str = f"Missing snapshot · {last_required_str}"
            # Today's future capture is not the deadline for yesterday's gap.
            if last_required_str != eod_dict.get("expected_market_date_str"):
                state_str = "Late"
                planned_dt = None
    elif status_str == "not_applicable" and eod_dict.get("last_required_eod_present_bool") is True:
        state_str, fact_str = "None", "Not due today"
        planned_dt = None
    else:
        state_str, fact_str = "Unknown", "EOD not verified"
    return _step_dict("EOD", state_str, fact_str, now_dt, actual_dt=actual_dt, planned_dt=planned_dt)


def build_cycle_view_dict(
    pod_row_dict: dict[str, Any], *, now_ts: datetime | None = None
) -> dict[str, Any]:
    """Project saved facts without reading files, changing state, or running jobs.

    ``source_stale_bool`` lets the caller carry a failed/stale summary read into
    the component. The existing client-operation freshness limit also applies.
    An earlier unresolved VPlan remains visible; its steps never borrow the
    newer DecisionPlan's evidence. No-order intent is absent from V3 summary
    rows, so zero broker orders alone does not produce a no-trade claim.
    """
    if str(pod_row_dict.get("mode_str") or "").lower() != "live":
        raise ValueError("Dashboard V4 cycle views support LIVE only.")
    now_dt = _timestamp_dt(now_ts or datetime.now(UTC))
    assert now_dt is not None
    source_dt = _timestamp_dt(pod_row_dict.get("as_of_timestamp_str"))
    source_fresh_bool = source_dt is not None and 0 <= (now_dt - source_dt).total_seconds() <= SOURCE_MAX_AGE_SECONDS_INT
    stale_bool = bool(pod_row_dict.get("source_stale_bool")) or not source_fresh_bool
    data_dict = pod_row_dict.get("norgate_snapshot_status_dict") or {}
    data_stale_bool = data_dict.get("snapshot_fresh_for_cycle_bool") is False
    source_missing_bool = pod_row_dict.get("db_status_str") != "ok" or not data_dict
    if stale_bool or data_stale_bool or source_missing_bool:
        fact_str = "Status out of date" if stale_bool else "Data not verified"
        step_dict_list = [_step_dict(label_str, "Unknown", fact_str, now_dt) for label_str in STEP_LABEL_TUPLE]
        return _cycle_dict(step_dict_list, stale_bool or data_stale_bool, "unknown")

    decision_status_str = str(pod_row_dict.get("latest_decision_plan_status_str") or "")
    vplan_status_str = str(pod_row_dict.get("latest_vplan_status_str") or "")
    action_str = str(pod_row_dict.get("next_action_str") or "")
    required_dict = pod_row_dict.get("required_action_dict") or {}
    decision_id_obj = pod_row_dict.get("latest_decision_plan_id_int")
    vplan_id_obj = pod_row_dict.get("latest_vplan_id_int")
    vplan_decision_id_obj = pod_row_dict.get("latest_vplan_decision_plan_id_int")
    match_obj = pod_row_dict.get("latest_vplan_is_for_latest_decision_bool")
    if decision_id_obj is not None and vplan_decision_id_obj is not None:
        match_obj = str(decision_id_obj) == str(vplan_decision_id_obj)
    previous_bool = vplan_id_obj is not None and match_obj is False
    previous_unresolved_bool = previous_bool and (
        vplan_status_str in {"ready", "submitting", "submitted", "blocked", "expired"}
        or (_count_int(pod_row_dict.get("missing_ack_count_int")) or 0) > 0
        or str(required_dict.get("label_str") or "").startswith("Review previous")
    )
    use_vplan_bool = vplan_id_obj is not None and (match_obj is True or previous_unresolved_bool)
    cycle_role_str = "previous" if previous_unresolved_bool else "current"
    prefix_str = "latest_vplan" if use_vplan_bool else "latest_decision_plan"
    submission_dt = _timestamp_dt(pod_row_dict.get(f"{prefix_str}_submission_timestamp_str"))
    target_dt = _timestamp_dt(pod_row_dict.get(f"{prefix_str}_target_execution_timestamp_str"))
    reconcile_due_dt = target_dt + timedelta(seconds=DEFAULT_RECONCILE_GRACE_SECONDS_INT) if target_dt is not None else None

    data_status_str = str(data_dict.get("status_str") or "")
    data_ready_bool = data_status_str == "ready" and bool(data_dict.get("snapshot_date_str")) and data_dict.get("snapshot_fresh_for_cycle_bool") is True
    data_state_str = "Done" if data_ready_bool else "Failed" if data_status_str == "failed" else "Unknown"
    data_fact_str = str(data_dict.get("snapshot_date_str") or "Freshness not verified")
    if previous_unresolved_bool:
        data_state_str, data_fact_str = "Unknown", "Previous cycle data"
    step_dict_list = [_step_dict("Data", data_state_str, data_fact_str, now_dt)]

    if previous_unresolved_bool:
        decision_state_str, decision_fact_str = "Unknown", "Previous decision"
    elif decision_status_str in {"planned", "vplan_ready", "submitted", "completed"} and decision_id_obj is not None:
        decision_state_str, decision_fact_str = "Done", f"Decision #{decision_id_obj}"
    elif decision_status_str in {"blocked", "expired"}:
        decision_state_str = "Failed" if decision_status_str == "blocked" else "Late"
        decision_fact_str = "Decision blocked" if decision_status_str == "blocked" else "Decision expired"
    else:
        decision_state_str = "Planned" if action_str == "build_decision_plan" else "Unknown"
        decision_fact_str = "Waiting for decision"
    step_dict_list.append(_step_dict("Decide", decision_state_str, decision_fact_str, now_dt))

    if use_vplan_bool:
        plan_state_str = "Failed" if vplan_status_str == "blocked" else "Late" if vplan_status_str == "expired" else "Done" if vplan_status_str in {"ready", "submitting", "submitted", "completed"} else "Unknown"
        step_dict_list.append(_step_dict("Plan", plan_state_str, f"Plan #{vplan_id_obj}", now_dt))
        order_count_int = _count_int(pod_row_dict.get("broker_order_count_int"))
        ack_count_int = _count_int(pod_row_dict.get("broker_ack_count_int"))
        missing_count_int = _count_int(pod_row_dict.get("missing_ack_count_int"))
        ack_status_str = str(pod_row_dict.get("latest_submit_ack_status_str") or "")
        order_count_str = "—" if order_count_int is None else str(order_count_int)
        ack_count_str = "—" if ack_count_int is None else str(ack_count_int)
        submit_fact_str = f"{order_count_str} sent · {ack_count_str} ack"
        if (missing_count_int or 0) > 0 or ack_status_str == "missing_critical":
            submit_state_str = "Failed"
        elif ack_status_str == "complete" and vplan_status_str in {"submitting", "submitted", "completed"}:
            complete_bool = order_count_int is not None and order_count_int > 0 and ack_count_int is not None and ack_count_int >= order_count_int and missing_count_int == 0
            submit_state_str = "Done" if complete_bool else "Unknown"
        elif vplan_status_str == "completed":
            # Legacy completed plans default missing ACK history to not_checked.
            submit_state_str, submit_fact_str = "Unknown", "ACK not verified"
        elif vplan_status_str in {"submitting", "submitted"}:
            submit_state_str = "Now"
        else:
            submit_state_str = _due_state_str(submission_dt, now_dt, source_dt, submit_bool=True)
            submit_fact_str = "Waiting to submit"
        step_dict_list.append(_step_dict("Submit", submit_state_str, submit_fact_str, now_dt, planned_dt=submission_dt))

        fill_count_int = _count_int(pod_row_dict.get("fill_count_int"))
        fill_fact_str = "Fill count not available" if fill_count_int is None else f"{fill_count_int} fill records" if fill_count_int else "No fills recorded"
        # Completion requires scoped per-order quantity proof from saved state.
        # Summary counts and reconciled positions alone remain insufficient.
        fill_state_str = "Unknown" if vplan_status_str == "completed" or fill_count_int is None else "Now" if vplan_status_str in {"submitting", "submitted"} and target_dt is not None and now_dt >= target_dt else "Planned"
        fill_dt = None
        evidence_dict = pod_row_dict.get("cycle_evidence_dict") or {}
        evidence_matches_bool = all(
            evidence_dict.get(evidence_key_str) is not None
            and evidence_dict[evidence_key_str] == pod_row_dict.get(row_key_str)
            for evidence_key_str, row_key_str in (
                ("pod_id_str", "pod_id_str"), ("account_route_str", "account_route_str"),
                ("vplan_id_int", "latest_vplan_id_int"),
                ("decision_plan_id_int", "latest_vplan_decision_plan_id_int"),
                ("vplan_status_str", "latest_vplan_status_str"),
            )
        )
        if evidence_matches_bool:
            evidence_state_str = evidence_dict.get("state_str")
            if evidence_state_str in {"complete", "partial"}:
                filled_int = _count_int(evidence_dict.get("filled_order_count_int"))
                requested_int = _count_int(evidence_dict.get("order_count_int"))
                if filled_int is not None and requested_int and 0 <= filled_int <= requested_int:
                    fill_fact_str = f"{filled_int} of {requested_int} filled"
                    fill_dt = _timestamp_dt(evidence_dict.get("actual_fill_timestamp_str"))
                    if evidence_state_str == "complete" and filled_int == requested_int and fill_dt is not None and target_dt is not None and target_dt <= fill_dt <= source_dt:
                        fill_state_str = "Done"
                    else:
                        fill_dt = None
                        fill_state_str = "Unknown" if vplan_status_str == "completed" else fill_state_str
            elif evidence_state_str == "no_orders":
                if submit_state_str != "Failed":
                    step_dict_list[-1] = _step_dict("Submit", "None", "No orders", now_dt)
                fill_state_str, fill_fact_str = "None", "No orders"
        step_dict_list.append(_step_dict("Fill", fill_state_str, fill_fact_str, now_dt, actual_dt=fill_dt, planned_dt=target_dt if fill_state_str != "None" else None))
        if fill_state_str == "Unknown" and evidence_matches_bool:
            step_dict_list[-1]["detail_str"] = (
                "Fill completion time could not be verified for this cycle."
                if evidence_dict.get("state_str") == "complete"
                else evidence_dict.get("reason_str") or "Fill details could not be checked."
            )

        reconcile_status_str = str(pod_row_dict.get("latest_reconciliation_status_str") or "")
        reconcile_dt = _timestamp_dt(pod_row_dict.get("latest_reconciliation_timestamp_str"))
        post_target_bool = (
            target_dt is not None and reconcile_dt is not None
            and target_dt <= reconcile_dt <= now_dt
            and target_dt.astimezone(MARKET_TIMEZONE_OBJ).date() == reconcile_dt.astimezone(MARKET_TIMEZONE_OBJ).date()
        )
        failure_dict = pod_row_dict.get("reconcile_read_failure_dict") or {}
        if failure_dict and not previous_unresolved_bool:
            reconcile_state_str, reconcile_fact_str = "Failed", "Broker read failed"
            reconcile_dt = _timestamp_dt(failure_dict.get("timestamp_str"))
        elif reconcile_status_str == "passed" and vplan_status_str == "completed" and post_target_bool:
            reconcile_state_str, reconcile_fact_str = "Done", "Positions checked"
        elif reconcile_status_str == "blocked" and post_target_bool:
            reconcile_state_str, reconcile_fact_str = "Failed", "Position check blocked"
        else:
            reconcile_state_str = _due_state_str(reconcile_due_dt, now_dt, source_dt)
            reconcile_fact_str, reconcile_dt = "Waiting for check", None
        step_dict_list.append(_step_dict("Reconcile", reconcile_state_str, reconcile_fact_str, now_dt, actual_dt=reconcile_dt, planned_dt=reconcile_due_dt))
    else:
        identity_unknown_bool = (vplan_id_obj is not None and match_obj is None) or decision_status_str == "completed"
        for label_str in STEP_LABEL_TUPLE[2:6]:
            state_str = "Unknown" if identity_unknown_bool else "Planned"
            fact_str = "Plan evidence missing" if decision_status_str == "completed" else "Cycle not verified" if identity_unknown_bool else "Waiting for plan"
            planned_dt = submission_dt if label_str == "Submit" else target_dt if label_str == "Fill" else reconcile_due_dt if label_str == "Reconcile" else None
            if not identity_unknown_bool and label_str in {"Plan", "Submit"}:
                state_str = _due_state_str(submission_dt, now_dt, source_dt, submit_bool=True)
            step_dict_list.append(_step_dict(label_str, state_str, fact_str, now_dt, planned_dt=planned_dt))

    idle_bool = (
        pod_row_dict.get("reason_code_str") == "not_month_end_session"
        and action_str == "wait"
        and required_dict.get("severity_str") not in {"red", "yellow"}
        and decision_status_str == "completed"
        and vplan_status_str == "completed"
        and match_obj is True
        and not previous_unresolved_bool
        and not any(step_dict["state_str"] == "Failed" for step_dict in step_dict_list)
    )
    if idle_bool:
        step_dict_list = [_step_dict(label_str, "None", "No trade scheduled", now_dt) for label_str in STEP_LABEL_TUPLE[:-1]]
    step_dict_list.append(_eod_step_dict(pod_row_dict.get("eod_snapshot_dict") or {}, now_dt, source_dt))
    return _cycle_dict(step_dict_list, False, cycle_role_str)


def _cycle_dict(step_dict_list: list[dict[str, str]], stale_bool: bool, cycle_role_str: str) -> dict[str, Any]:
    state_list = [step_dict["state_str"] for step_dict in step_dict_list]
    if "Failed" in state_list:
        pill_str, tone_str, priority_str = "Action needed", "red", "Failed"
    elif "Late" in state_list:
        pill_str, tone_str, priority_str = "Late", "amber", "Late"
    elif "Unknown" in state_list:
        pill_str, tone_str, priority_str = "Unknown", "gray", "Unknown"
    elif "Now" in state_list:
        pill_str, tone_str, priority_str = "Working", "blue", "Now"
    elif all(state_str == "None" for state_str in state_list[:-1]):
        pill_str, tone_str, priority_str = "Waiting", "gray", "None"
    else:
        pill_str, tone_str, priority_str = "On track", "green", "Done"
    focus_dict = next((step_dict for step_dict in step_dict_list if step_dict["state_str"] == priority_str), step_dict_list[-1])
    next_dict = next((step_dict for step_dict in step_dict_list if step_dict["state_str"] in {"Planned", "Now", "Late", "Failed"}), {})
    now_str = focus_dict["fact_str"]
    now_detail_str = ""
    if pill_str == "Waiting":
        now_str, now_detail_str = "Waiting", "No trade scheduled"
    if focus_dict["label_str"] == "Fill" and priority_str in {"Unknown", "Now"}:
        now_str = "Fill not verified" if priority_str == "Unknown" else "Filling"
        now_detail_str = focus_dict.get("detail_str") or focus_dict["fact_str"]
        if priority_str == "Unknown" and state_list.count("Unknown") == 1 and step_dict_list[5]["state_str"] == "Done":
            # Keep the unverified status gray while retaining the independent
            # fact that broker positions were reconciled for this cycle.
            pill_str, now_str = "Not verified", "Reconciled · Fill details not verified"
    if pill_str == "On track":
        if step_dict_list[5]["state_str"] == "Done":
            now_str, now_detail_str = "Reconciled", step_dict_list[4]["fact_str"]
        else:
            now_str = f"Waiting for {next_dict['label_str']}" if next_dict else "Cycle complete"
    return {
        "step_dict_list": step_dict_list,
        "pill_str": pill_str,
        "tone_str": tone_str,
        "now_str": now_str,
        "now_detail_str": now_detail_str,
        "next_str": next_dict.get("label_str", "—"),
        "next_time_str": next_dict.get("planned_time_str", ""),
        "next_timestamp_str": next_dict.get("planned_timestamp_str", ""),
        "stale_bool": stale_bool,
        "cycle_role_str": cycle_role_str,
    }
