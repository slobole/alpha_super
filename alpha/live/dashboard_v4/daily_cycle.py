"""Daily (CORE5, MR capsule) status projection from saved daily facts.

Pure: no file, DB or clock reads. A daily cycle closes after the session from
actual holdings, so its status is the daily vocabulary below, never the monthly
VPlan/ACK/fill completion rules. Rows are checked in order; the first match
wins. Times come from the exchange calendar (early closes included) through
the daily reader. The time checks use the oldest of now, the summary time and
the daily read time, so a cached early read can never prove a miss.

 #  Label                                         Tone      Predicate
 1  Closed                                        green     decision completed
 2  Closed with exceptions (n)                    amber     completed_with_exceptions; all not dispatched:
                                                            "Not sent · closed with exceptions (n)"
 3  Superseded                                    neutral   decision superseded
 4  Open after close                              red       any non-terminal daily decision, now >= close + 1h
 5  Dispatch unconfirmed: check broker, do not    red       VPlan submitting at target + 5 min; or opening
    resend                                                  dispatch incomplete with attempted keys only
 6  Partly sent (a of a+n)                        red       opening dispatch incomplete, attempted and never sent
 7  Not sent                                      red       opening dispatch incomplete, nothing attempted; or
                                                            VPlan None/ready and decision planned/vplan_ready at
                                                            target + 5 min; or VPlan blocked/expired
 8  Retrying send                                 working   dispatch_retry_pending before its deadline
 9  No decision for <date>                        red       the watchdog's decision_incomplete
10  Closing                                       working   not terminal, close <= now < close + 1h
11  Trading                                       working   VPlan submitted, now < close
12  Ready · sends 09:28                           on track  decision planned/vplan_ready, VPlan None/ready,
                                                            now < target + 5 min
13  Deciding                                      working   data ready, owed decision not built, before cutoff

Rows 1-3 describe the current decision only while no older daily decision is
open past its close + 1h: a newer close never hides row 4.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta

from alpha.live.daily_watchdog import OPENING_DISPATCH_GRACE_MINUTES_INT
from alpha.live.dashboard_v3.filters import MARKET_TIMEZONE_OBJ
from alpha.live.dashboard_v4.cycle import _step_dict, _timestamp_dt


CLOSE_OVERDUE = timedelta(hours=1)
OPENING_GRACE = timedelta(minutes=OPENING_DISPATCH_GRACE_MINUTES_INT)
# The MOO batch must reach the broker by the open minus two minutes.
DISPATCH_LEAD = timedelta(minutes=2)
TERMINAL_STATUS_SET = frozenset({"completed", "completed_with_exceptions", "superseded"})
UNDISPATCHED_DECISION_STATUS_SET = frozenset({"planned", "vplan_ready"})
SENT_VPLAN_STATUS_SET = frozenset({"submitted", "completed", "completed_with_exceptions"})
EXCEPTION_REASON_LABEL_DICT = {
    "decision_intent_not_dispatched": "Not sent",
    "actual_holding_differs_from_decision": "Holding differs from decision",
    "actual_holding_differs_from_vplan": "Not filled / holding differs from plan",
    "funding_buys_dropped": "Buy dropped (funding)",
    "bil_buy_withheld_after_funding_buys_dropped": "BIL buy withheld (funding)",
    "bil_sale_withheld_after_funding_buys_dropped": "BIL sale withheld (funding)",
}
# key: (tone, pill). Tones are the existing V4 state tones.
STATUS_STYLE_DICT = {
    "closed": ("green", "Closed"), "closed_with_exceptions": ("amber", "Needs review"),
    "superseded": ("neutral", "Superseded"), "open_after_close": ("red", "Action needed"),
    "dispatch_unconfirmed": ("red", "Action needed"), "partly_sent": ("red", "Action needed"),
    "not_sent": ("red", "Action needed"), "retrying_send": ("blue", "Working"),
    "no_decision": ("red", "Action needed"), "closing": ("blue", "Working"), "trading": ("blue", "Working"),
    "sending": ("blue", "Working"), "ready": ("green", "On track"), "deciding": ("blue", "Working"),
    "unknown": ("gray", "Unknown"),
}
OPEN_STATUS_SET = frozenset(STATUS_STYLE_DICT) - {"closed", "closed_with_exceptions", "superseded", "unknown"}
DAILY_STEP_LABEL_TUPLE = ("Decide", "Plan", "Submit", "Fill", "Reconcile")


def _hm_str(timestamp_dt: datetime | None, now_dt: datetime) -> str:
    if timestamp_dt is None:
        return "—"
    market_dt = timestamp_dt.astimezone(MARKET_TIMEZONE_OBJ)
    same_day_bool = market_dt.date() == now_dt.astimezone(MARKET_TIMEZONE_OBJ).date()
    return market_dt.strftime("%H:%M" if same_day_bool else "%m-%d %H:%M")


def _int(value_obj) -> int:
    return value_obj if isinstance(value_obj, int) and not isinstance(value_obj, bool) and value_obj >= 0 else 0


def _status_dict(key_str, label_str, detail_str=""):
    tone_str, pill_str = STATUS_STYLE_DICT[key_str]
    return {"key_str": key_str, "label_str": label_str, "tone_str": tone_str, "pill_str": pill_str,
        "detail_str": detail_str, "red_bool": tone_str == "red"}


def _exception_summary_str(exception_list):
    count_dict = Counter(row_dict.get("reason_str") for row_dict in exception_list)
    return " · ".join(f"{EXCEPTION_REASON_LABEL_DICT.get(reason_str, 'Other exception')} ({count_int})"
        for reason_str, count_int in sorted(count_dict.items(), key=lambda item_tuple: (-item_tuple[1], str(item_tuple[0]))))


def _unknown_result_dict(now_dt, fact_str="Daily status not verified"):
    return {"step_dict_list": [_step_dict(label_str, "Unknown", fact_str, now_dt) for label_str in DAILY_STEP_LABEL_TUPLE],
        "status_dict": _status_dict("unknown", "Status unknown", fact_str + ".")}


def build_daily_steps_and_status(pod_row_dict, data_step_dict, *, now_dt: datetime, source_dt: datetime | None):
    """Decide, Plan, Submit, Fill and Reconcile steps plus one daily status.

    Data and EOD stay with the caller. Returns ``{"step_dict_list", "status_dict"}``.
    """
    daily_dict = pod_row_dict.get("daily_cycle_dict") or {}
    if daily_dict.get("status_str") != "ok":
        return _unknown_result_dict(now_dt)
    read_dt = _timestamp_dt(daily_dict.get("as_of_timestamp_str"))
    if read_dt is None or read_dt > now_dt:
        return _unknown_result_dict(now_dt)
    check_dt = min(item_dt for item_dt in (now_dt, source_dt, read_dt) if item_dt is not None)
    decision_dict = daily_dict.get("decision_dict") or {}
    vplan_dict = daily_dict.get("vplan_dict") or {}
    dispatch_dict = daily_dict.get("opening_dispatch_dict") or {}
    result_dict = daily_dict.get("execution_result_dict") or {}
    signal_dict = daily_dict.get("signal_session_dict") or {}
    last_closed_dict = daily_dict.get("last_closed_dict") or {}
    decision_status_str = decision_dict.get("status_str")
    vplan_status_str = vplan_dict.get("status_str")
    target_dt = _timestamp_dt(decision_dict.get("target_execution_timestamp_str"))
    close_dt = _timestamp_dt(daily_dict.get("session_close_timestamp_str"))
    cutoff_dt = _timestamp_dt(signal_dict.get("decision_cutoff_timestamp_str"))
    send_dt = target_dt - DISPATCH_LEAD if target_dt is not None else None
    if decision_dict and (target_dt is None or close_dt is None):
        return _unknown_result_dict(now_dt)
    data_ready_bool = data_step_dict.get("state_str") == "Done"
    terminal_bool = decision_status_str in TERMINAL_STATUS_SET

    # The current cycle is the owed decision once the latest one has closed and
    # the next one is due (cutoff passed) or can be built (data ready).
    owed_missing_bool = (daily_dict.get("current_bool") is True and bool(signal_dict)
        and signal_dict.get("decision_plan_id_int") is None and (not decision_dict or terminal_bool))
    owed_bool = owed_missing_bool and cutoff_dt is not None and (check_dt >= cutoff_dt or data_ready_bool)
    subject_dict = {} if owed_bool else decision_dict

    overdue_list = []
    if subject_dict and not terminal_bool and check_dt >= close_dt + CLOSE_OVERDUE:
        overdue_list.append((decision_dict.get("decision_plan_id_int"), decision_dict.get("target_session_date_str")))
    for previous_dict in daily_dict.get("open_previous_list") or []:
        previous_close_dt = _timestamp_dt(previous_dict.get("session_close_timestamp_str"))
        if previous_close_dt is not None and check_dt >= previous_close_dt + CLOSE_OVERDUE:
            overdue_list.append((previous_dict.get("decision_plan_id_int"), previous_dict.get("target_session_date_str")))

    reason_str = dispatch_dict.get("reason_code_str")
    attempted_int = _int(dispatch_dict.get("attempted_count_int"))
    never_int = _int(dispatch_dict.get("never_dispatched_count_int"))
    incomplete_bool = reason_str == "opening_dispatch_incomplete"
    deadline_dt = _timestamp_dt(dispatch_dict.get("dispatch_deadline_timestamp_str")) or send_dt
    grace_bool = target_dt is not None and check_dt >= target_dt + OPENING_GRACE
    unsent_plan_bool = vplan_status_str in {None, "ready"} and decision_status_str in UNDISPATCHED_DECISION_STATUS_SET
    exception_list = result_dict.get("exception_list") or []
    exception_count_int = _int(result_dict.get("exception_count_int")) or len(exception_list)
    completion_asset_list = [row_dict.get("asset_str") or "—" for row_dict in daily_dict.get("completion_request_list") or []]
    halt_bool = signal_dict.get("capsule_halt_bool") is True

    if subject_dict and not overdue_list and decision_status_str == "completed":
        status_dict = _status_dict("closed", "Closed", "Closed from holdings · " + _hm_str(
            _timestamp_dt(result_dict.get("completed_timestamp_str")), now_dt) + " ET")
    elif subject_dict and not overdue_list and decision_status_str == "completed_with_exceptions":
        not_sent_bool = bool(exception_list) and all(row_dict.get("reason_str") == "decision_intent_not_dispatched"
            for row_dict in exception_list)
        status_dict = _status_dict("closed_with_exceptions",
            ("Not sent · closed with exceptions" if not_sent_bool else "Closed with exceptions") + f" ({exception_count_int})",
            _exception_summary_str(exception_list) or "Exceptions saved without detail.")
    elif subject_dict and not overdue_list and decision_status_str == "superseded":
        status_dict = _status_dict("superseded", "Superseded", "A newer decision replaced this one.")
    elif overdue_list:
        decision_id_obj, date_str = overdue_list[0]
        more_str = f" and {len(overdue_list) - 1} more" if len(overdue_list) > 1 else ""
        status_dict = _status_dict("open_after_close", "Open after close",
            f"Decision #{decision_id_obj} for {date_str}{more_str} is still open 1 h after the close. "
            "Check owned orders at the broker; daily reconciliation keeps retrying.")
    elif subject_dict and ((vplan_status_str == "submitting" and grace_bool)
            or (incomplete_bool and attempted_int and not never_int)):
        status_dict = _status_dict("dispatch_unconfirmed", "Dispatch unconfirmed: check broker, do not resend",
            "Orders may already be live. Check broker orders and fills before any action; do not resend.")
    elif subject_dict and incomplete_bool and attempted_int and never_int:
        status_dict = _status_dict("partly_sent", f"Partly sent ({attempted_int} of {attempted_int + never_int})",
            f"{never_int} opening order{'s' if never_int != 1 else ''} never reached the broker. "
            "Check broker orders; do not resend. The day closes from holdings.")
    elif subject_dict and ((incomplete_bool and not attempted_int) or (unsent_plan_bool and grace_bool)
            or vplan_status_str in {"blocked", "expired"}):
        status_dict = _status_dict("not_sent", "Not sent",
            "The opening batch never reached the broker. Do not resend: the day closes from holdings with exceptions.")
    elif subject_dict and reason_str == "dispatch_retry_pending" and deadline_dt is not None and check_dt < deadline_dt:
        status_dict = _status_dict("retrying_send", "Retrying send",
            "Transient broker error · retry before " + _hm_str(deadline_dt, now_dt) + " ET")
    elif (owed_bool and check_dt >= cutoff_dt) or (subject_dict and decision_status_str in {"blocked", "expired"}
            and send_dt is not None and check_dt >= send_dt):
        date_str = signal_dict.get("session_date_str") if owed_bool else decision_dict.get("signal_session_date_str")
        status_dict = _status_dict("no_decision", f"No decision for {date_str}",
            "No usable decision; nothing was sent for it. Check the daily serve and its event log."
            + (" The capsule halted on a holding mismatch." if halt_bool else ""))
    elif subject_dict and not terminal_bool and close_dt <= check_dt < close_dt + CLOSE_OVERDUE:
        status_dict = _status_dict("closing", "Closing", "Closes from holdings by " + _hm_str(close_dt + CLOSE_OVERDUE, now_dt) + " ET")
    elif subject_dict and vplan_status_str == "submitted" and check_dt < close_dt:
        completing_str = (" · completing exit " + ", ".join(completion_asset_list) + " (MKT)") if completion_asset_list else ""
        status_dict = _status_dict("trading", "Trading" + completing_str,
            "Opening orders sent · closes " + _hm_str(close_dt, now_dt) + " ET")
    elif subject_dict and vplan_status_str == "submitting":
        status_dict = _status_dict("sending", "Sending", "The opening batch is being sent.")
    elif subject_dict and unsent_plan_bool and not grace_bool:
        plan_str = f"Plan #{vplan_dict['vplan_id_int']}" if vplan_dict else f"Decision #{decision_dict.get('decision_plan_id_int')}"
        status_dict = _status_dict("ready", "Ready · sends " + _hm_str(send_dt, now_dt), plan_str + " · trades at the open")
    elif owed_bool and data_ready_bool and check_dt < cutoff_dt:
        status_dict = _status_dict("deciding", "Deciding", "Data ready · decision due " + _hm_str(cutoff_dt, now_dt) + " ET")
    else:
        status_dict = _status_dict("unknown", "Status unknown", "Daily status not verified.")
    if status_dict["key_str"] in OPEN_STATUS_SET and _int(last_closed_dict.get("exception_count_int")):
        count_int = _int(last_closed_dict["exception_count_int"])
        status_dict["last_close_str"] = (f"Last close {last_closed_dict.get('target_session_date_str')}: "
            f"{count_int} exception{'s' if count_int != 1 else ''}")

    key_str = status_dict["key_str"]
    if owed_bool:
        missed_bool = check_dt >= cutoff_dt
        if missed_bool:
            decide_dict = _step_dict("Decide", "Failed", f"No decision for {signal_dict.get('session_date_str')}", now_dt)
        else:
            decide_dict = _step_dict("Decide", "Now" if data_ready_bool else "Planned",
                "Building decision" if data_ready_bool else "Waiting for data", now_dt, planned_dt=cutoff_dt)
        return {"step_dict_list": [decide_dict] + [_step_dict(label_str, "None" if missed_bool else "Planned",
            "No decision" if missed_bool else "Waiting for decision", now_dt) for label_str in DAILY_STEP_LABEL_TUPLE[1:]],
            "status_dict": status_dict}
    if not decision_dict:
        return {"step_dict_list": _unknown_result_dict(now_dt, "No saved decision")["step_dict_list"], "status_dict": status_dict}
    if decision_status_str == "superseded":
        return {"step_dict_list": [_step_dict(label_str, "None", "Superseded", now_dt) for label_str in DAILY_STEP_LABEL_TUPLE],
            "status_dict": status_dict}

    decision_id_obj = decision_dict.get("decision_plan_id_int")
    if decision_status_str in {"blocked", "expired"}:
        decide_dict = _step_dict("Decide", "Failed", f"Decision {decision_status_str}", now_dt)
    else:
        decide_dict = _step_dict("Decide", "Done", f"Decision #{decision_id_obj}", now_dt)

    submission_dt = _timestamp_dt(vplan_dict.get("submission_timestamp_str") or decision_dict.get("submission_timestamp_str"))
    if vplan_status_str in {"blocked", "expired"}:
        plan_dict = _step_dict("Plan", "Late", f"Plan {vplan_status_str}", now_dt)
    elif vplan_dict:
        plan_dict = _step_dict("Plan", "Done", f"Plan #{vplan_dict.get('vplan_id_int')}", now_dt)
    elif terminal_bool or grace_bool or decision_status_str in {"blocked", "expired"}:
        plan_dict = _step_dict("Plan", "Late", "No plan", now_dt)
    else:
        plan_dict = _step_dict("Plan", "Planned", "Builds before the open", now_dt, planned_dt=submission_dt)

    sent_int, ack_int = _int(vplan_dict.get("sent_order_count_int")), _int(vplan_dict.get("ack_count_int"))
    count_str = f"{sent_int} sent · {ack_int} ack"
    # Nothing of the opening batch reached the broker, whether open or closed.
    nothing_sent_bool = (key_str == "not_sent" or (incomplete_bool and not attempted_int)
        or (decision_status_str == "completed_with_exceptions" and status_dict["label_str"].startswith("Not sent")))
    if key_str == "dispatch_unconfirmed":
        submit_dict = _step_dict("Submit", "Failed", "Unconfirmed · do not resend", now_dt, planned_dt=send_dt)
        submit_dict["detail_str"] = "Orders may already be live. Check the broker before any action; do not resend."
    elif (incomplete_bool and attempted_int and never_int) or key_str == "partly_sent":
        submit_dict = _step_dict("Submit", "Failed", f"Partly sent · {attempted_int} of {attempted_int + never_int}", now_dt, planned_dt=send_dt)
    elif nothing_sent_bool:
        submit_dict = _step_dict("Submit", "Failed", "Not sent", now_dt, planned_dt=send_dt)
    elif _int(vplan_dict.get("missing_ack_count_int")):
        submit_dict = _step_dict("Submit", "Failed", count_str, now_dt, planned_dt=send_dt)
    elif key_str == "retrying_send":
        submit_dict = _step_dict("Submit", "Now", "Retrying send", now_dt, planned_dt=send_dt)
    elif vplan_status_str == "submitting":
        submit_dict = _step_dict("Submit", "Now", "Sending", now_dt, planned_dt=send_dt)
    elif vplan_status_str in SENT_VPLAN_STATUS_SET:
        submit_dict = _step_dict("Submit", "Done", count_str, now_dt, planned_dt=send_dt)
    elif terminal_bool:
        submit_dict = _step_dict("Submit", "None", "No orders", now_dt)
    elif decision_status_str in {"blocked", "expired"}:
        submit_dict = _step_dict("Submit", "None", "Nothing to send", now_dt)
    else:
        submit_dict = _step_dict("Submit", "Planned", "Sends " + _hm_str(send_dt, now_dt), now_dt, planned_dt=send_dt)

    # Fills are reporting records: informational, never the close condition.
    fill_int = _int(vplan_dict.get("fill_count_int"))
    fill_fact_str = f"{fill_int} fill record{'s' if fill_int != 1 else ''}" if fill_int else "No fills recorded"
    sent_bool = vplan_status_str in SENT_VPLAN_STATUS_SET | {"submitting"} and not nothing_sent_bool
    if key_str == "dispatch_unconfirmed":
        fill_dict = _step_dict("Fill", "Unknown", "Check broker fills", now_dt, planned_dt=target_dt)
    elif not sent_bool:
        fill_dict = _step_dict("Fill", "None", "Nothing sent" if submit_dict["state_str"] == "Failed" else "No orders yet", now_dt)
    elif terminal_bool:
        fill_dict = _step_dict("Fill", "Done", fill_fact_str, now_dt)
    elif check_dt >= target_dt:
        fill_dict = _step_dict("Fill", "Now", fill_fact_str, now_dt, planned_dt=target_dt)
    else:
        fill_dict = _step_dict("Fill", "Planned", "Fills at the open", now_dt, planned_dt=target_dt)
    if completion_asset_list and not terminal_bool:
        fill_dict["detail_str"] = "Completing exit " + ", ".join(completion_asset_list) + " (MKT)"

    completed_dt = _timestamp_dt(result_dict.get("completed_timestamp_str"))
    if decision_status_str == "completed":
        reconcile_dict = _step_dict("Reconcile", "Done", "Closed from holdings", now_dt, actual_dt=completed_dt, planned_dt=close_dt)
    elif decision_status_str == "completed_with_exceptions":
        reconcile_dict = _step_dict("Reconcile", "Done", f"Closed · {exception_count_int} exception{'s' if exception_count_int != 1 else ''}",
            now_dt, actual_dt=completed_dt, planned_dt=close_dt)
    elif check_dt >= close_dt + CLOSE_OVERDUE:
        reconcile_dict = _step_dict("Reconcile", "Failed", "Open after close", now_dt, planned_dt=close_dt)
    elif check_dt >= close_dt:
        reconcile_dict = _step_dict("Reconcile", "Now", "Closing", now_dt, planned_dt=close_dt)
    else:
        reconcile_dict = _step_dict("Reconcile", "Planned", "Closes " + _hm_str(close_dt, now_dt), now_dt, planned_dt=close_dt)
    return {"step_dict_list": [decide_dict, plan_dict, submit_dict, fill_dict, reconcile_dict], "status_dict": status_dict}
