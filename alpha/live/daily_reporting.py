"""Daily post-close reporting retries failed reads, never a successful report.

Retries stop REPORT_RETRY_WINDOW after the target session close. An attempt
left 'attempted' by a stopped process is retried once its lease expires.
"""
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import json
import math

from alpha.live import scheduler_utils
from alpha.live.daily_reconcile import is_daily_reconcile_release_bool


REPORT_ATTEMPT_LEASE = timedelta(minutes=30)
REPORT_RETRY_WINDOW = timedelta(hours=24)


UNPROVED_SPARSE_SOURCE_STR = "unproved_sparse_refresh"


def _unproved_summary_obj(fallback_record_obj, proof_gap_str):
    """A sparse Filled refresh without proof keeps the saved summary (or itself) and names the gap."""
    return replace(fallback_record_obj, raw_payload_dict={**fallback_record_obj.raw_payload_dict,
        "order_summary_source_str": UNPROVED_SPARSE_SOURCE_STR, "order_summary_proof_gap_str": proof_gap_str})


def normalize_daily_order_summary_list(record_list, fill_list, known_order_list, request_list, ack_list, *, as_of_ts):
    """Repair incomplete Filled summaries from exact order and execution proof.

    Reporting only: filled = sum(abs(execution shares));
    VWAP = sum(abs(execution shares) * execution price) / filled.
    A sparse refresh (empty quantities) is repaired only from complete proof: the
    canonical request, a positive broker ACK and identified executions that add
    up to the request. Without that proof the saved summary is kept as it is and
    the gap is named in its payload (``order_summary_proof_gap_str``). A refresh
    that contradicts saved identity, quantities or executions still fails the
    report. Native nonzero quantities keep their existing reporting path. The
    caller writes the fills either way; daily holdings settlement is independent.
    """
    normalized_record_list = []
    identity_field_tuple = ("broker_order_id_str", "decision_plan_id_int", "vplan_id_int",
        "account_route_str", "asset_str", "order_request_key_str", "broker_order_type_str",
        "unit_str", "submission_key_str")
    for record_obj in record_list:
        if not (record_obj.status_str == "Filled" and record_obj.remaining_amount_float in (0, None)
                and (record_obj.amount_float == 0 or record_obj.filled_amount_float == 0
                    or record_obj.remaining_amount_float is None or record_obj.avg_fill_price_float is None)):
            normalized_record_list.append(record_obj)
            continue
        known_match_list = [row_dict for row_dict in known_order_list
            if row_dict["broker_order_id_str"] == record_obj.broker_order_id_str or (
                record_obj.order_request_key_str and row_dict["order_request_key_str"] == record_obj.order_request_key_str)]
        if len(known_match_list) > 1:
            raise ValueError("Sparse order summary has ambiguous saved order identity.")
        fallback_record_obj = record_obj
        # What an unproved refresh writes back: the saved row whenever there is
        # one (valid or not), never the sparse record over it.
        saved_record_obj = record_obj
        preserve_complete_bool = False
        empty_quantity_bool = record_obj.amount_float == 0 or record_obj.filled_amount_float == 0
        if known_match_list:
            known_dict = known_match_list[0]
            if any(known_dict.get(field_str) != getattr(record_obj, field_str) for field_str in identity_field_tuple):
                raise ValueError("Sparse order summary conflicts with saved order identity.")
            known_amount_float = float(known_dict["amount_float"])
            known_filled_float = float(known_dict["filled_amount_float"])
            known_remaining_float = known_dict["remaining_amount_float"]
            known_price_float = known_dict["avg_fill_price_float"]
            valid_known_bool = (
                math.isfinite(known_amount_float) and known_amount_float != 0
                and math.isfinite(known_filled_float) and abs(known_filled_float) <= abs(known_amount_float)
                and (known_filled_float >= 0 or known_amount_float < 0)
                and (known_remaining_float is None or (math.isfinite(known_remaining_float)
                    and 0 <= known_remaining_float <= abs(known_amount_float)))
                and (known_price_float is None or (math.isfinite(known_price_float) and known_price_float > 0))
                and (known_filled_float == 0 or known_price_float is not None)
                and (known_dict["status_str"] != "Filled" or (
                    abs(abs(known_filled_float) - abs(known_amount_float)) <= 1e-9 and known_remaining_float == 0))
            )
            saved_record_obj = replace(record_obj, amount_float=known_amount_float,
                filled_amount_float=known_filled_float, remaining_amount_float=known_remaining_float,
                avg_fill_price_float=known_price_float, status_str=known_dict["status_str"],
                submitted_timestamp_ts=datetime.fromisoformat(known_dict["submitted_timestamp_str"]),
                last_status_timestamp_ts=(datetime.fromisoformat(known_dict["last_status_timestamp_str"])
                    if known_dict["last_status_timestamp_str"] else None),
                raw_payload_dict=json.loads(known_dict["raw_payload_json_str"]))
            if valid_known_bool:
                fallback_record_obj = saved_record_obj
                preserve_complete_bool = known_dict["status_str"] == "Filled"
        request_match_list = [request_obj for request_obj in request_list
            if request_obj.order_request_key_str == record_obj.order_request_key_str]
        if len(request_match_list) > 1:
            raise ValueError("Sparse order summary has ambiguous canonical request identity.")
        if not request_match_list:
            if preserve_complete_bool or not empty_quantity_bool:
                normalized_record_list.append(fallback_record_obj if preserve_complete_bool else record_obj)
            else:
                normalized_record_list.append(_unproved_summary_obj(saved_record_obj, "no_canonical_request"))
            continue
        request_obj = request_match_list[0]
        if (any(getattr(request_obj, field_str) != getattr(record_obj, field_str)
                for field_str in identity_field_tuple if field_str != "broker_order_id_str")
                or not request_obj.decision_plan_id_int or not request_obj.vplan_id_int):
            raise ValueError("Sparse order summary conflicts with canonical request identity.")
        requested_amount_float = float(request_obj.amount_float)
        if not math.isfinite(requested_amount_float) or requested_amount_float == 0:
            raise ValueError("Sparse order summary has an invalid canonical requested quantity.")
        if (not math.isfinite(record_obj.amount_float) or (record_obj.amount_float != 0
                and abs(record_obj.amount_float - requested_amount_float) > 1e-9)):
            raise ValueError("Sparse order summary conflicts with the canonical requested quantity.")
        if (not math.isfinite(record_obj.filled_amount_float)
                or abs(record_obj.filled_amount_float) > abs(requested_amount_float) + 1e-9):
            raise ValueError("Sparse order summary has an invalid reported filled quantity.")
        if fallback_record_obj is not record_obj and abs(fallback_record_obj.amount_float - requested_amount_float) > 1e-9:
            raise ValueError("Sparse order summary conflicts with the saved requested quantity.")
        ack_match_list = [ack_dict for ack_dict in ack_list
            if ack_dict["broker_order_id_str"] == record_obj.broker_order_id_str
            or ack_dict["order_request_key_str"] == record_obj.order_request_key_str]
        if len(ack_match_list) > 1:
            raise ValueError("Sparse order summary has ambiguous broker ACK identity.")
        if ack_match_list and any(ack_match_list[0].get(field_str) != getattr(record_obj, field_str)
                for field_str in identity_field_tuple if field_str not in {"unit_str", "submission_key_str"}):
            raise ValueError("Sparse order summary conflicts with broker ACK identity.")
        submitted_timestamp_ts = fallback_record_obj.submitted_timestamp_ts
        try:
            ack_timestamp_ts = datetime.fromisoformat(ack_match_list[0]["response_timestamp_str"])
            valid_ack_bool = (ack_timestamp_ts.tzinfo is not None and submitted_timestamp_ts.tzinfo is not None
                and as_of_ts.tzinfo is not None and submitted_timestamp_ts <= ack_timestamp_ts <= as_of_ts
                and ack_match_list[0]["local_submit_ack_bool"] == 1
                and ack_match_list[0]["broker_response_ack_bool"] == 1
                and ack_match_list[0]["ack_status_str"] == "broker_acked")
        except (IndexError, KeyError, TypeError, ValueError):
            valid_ack_bool = False
        if not valid_ack_bool:
            if preserve_complete_bool or not empty_quantity_bool:
                # Native nonzero quantities retain their existing reporting path.
                normalized_record_list.append(fallback_record_obj if preserve_complete_bool else record_obj)
            else:
                # Missing ACKs never turn empty quantities into proved fills.
                normalized_record_list.append(_unproved_summary_obj(saved_record_obj, "no_broker_ack"))
            continue
        execution_id_set = set()
        matched_fill_list = []
        proof_complete_bool = True
        for fill_obj in fill_list:
            execution_id_str = fill_obj.raw_payload_dict.get("exec_id_str")
            if fill_obj.broker_order_id_str != record_obj.broker_order_id_str:
                continue
            if any(getattr(fill_obj, field_str) != getattr(record_obj, field_str) for field_str in (
                    "decision_plan_id_int", "vplan_id_int", "account_route_str", "asset_str")):
                raise ValueError("Sparse order summary conflicts with execution identity.")
            fill_amount_float = float(fill_obj.fill_amount_float)
            fill_price_float = float(fill_obj.fill_price_float)
            # *** CRITICAL *** reporting attribution: executions must occur at or
            # after this exact order's saved submission and no later than this
            # report observation. Never attribute prior-session or future fills.
            if (not math.isfinite(fill_amount_float) or fill_amount_float * requested_amount_float <= 0
                    or not math.isfinite(fill_price_float) or fill_price_float <= 0
                    or fill_obj.fill_timestamp_ts.tzinfo is None or submitted_timestamp_ts.tzinfo is None
                    or as_of_ts.tzinfo is None or not submitted_timestamp_ts <= fill_obj.fill_timestamp_ts <= as_of_ts):
                raise ValueError("Sparse order summary has invalid execution quantities, prices or timestamps.")
            if not isinstance(execution_id_str, str) or not execution_id_str.strip():
                proof_complete_bool = False
            else:
                if execution_id_str in execution_id_set or any(
                        other_fill_obj is not fill_obj and other_fill_obj.broker_order_id_str != record_obj.broker_order_id_str
                        and other_fill_obj.raw_payload_dict.get("exec_id_str") == execution_id_str for other_fill_obj in fill_list):
                    raise ValueError("Sparse order summary has a duplicate or conflicting execution ID.")
                execution_id_set.add(execution_id_str)
            matched_fill_list.append(fill_obj)
        filled_amount_float = sum(abs(float(fill_obj.fill_amount_float)) for fill_obj in matched_fill_list)
        if filled_amount_float > abs(requested_amount_float) + 1e-9:
            raise ValueError("Sparse order summary executions exceed the canonical requested quantity.")
        if not matched_fill_list or not proof_complete_bool or abs(filled_amount_float - abs(requested_amount_float)) > 1e-9:
            if preserve_complete_bool or not empty_quantity_bool:
                normalized_record_list.append(fallback_record_obj if preserve_complete_bool else record_obj)
            else:
                normalized_record_list.append(_unproved_summary_obj(saved_record_obj, "incomplete_executions"))
            continue
        average_price_float = sum(abs(float(fill_obj.fill_amount_float)) * float(fill_obj.fill_price_float)
            for fill_obj in matched_fill_list) / filled_amount_float
        if not math.isfinite(average_price_float):
            raise ValueError("Sparse order summary has an invalid execution weighted price.")
        normalized_record_list.append(replace(record_obj, amount_float=requested_amount_float,
            filled_amount_float=filled_amount_float, remaining_amount_float=0.0, avg_fill_price_float=average_price_float,
            last_status_timestamp_ts=max(fill_obj.fill_timestamp_ts for fill_obj in matched_fill_list),
            raw_payload_dict={**record_obj.raw_payload_dict, "order_summary_source_str": "canonical_request_and_executions"}))
    return normalized_record_list


def _session_close_ts(target_execution_ts, session_calendar_id_str):
    session_date_obj = scheduler_utils.to_market_timestamp_ts(target_execution_ts, session_calendar_id_str).date()
    return scheduler_utils.get_session_close_timestamp_ts(session_date_obj, session_calendar_id_str)


def _retry_due_bool(status_str, attempted_timestamp_str, close_ts, as_of_ts):
    if as_of_ts >= close_ts + REPORT_RETRY_WINDOW:
        return False
    if status_str == "failed":
        return True
    if status_str != "attempted":
        return False
    try:
        return as_of_ts - datetime.fromisoformat(attempted_timestamp_str) >= REPORT_ATTEMPT_LEASE
    except (TypeError, ValueError):
        # An unreadable claim time cannot hold a report forever.
        return True


def claim_post_close_report_attempt(state_store_obj, release_obj, decision_plan_obj, as_of_ts):
    if not is_daily_reconcile_release_bool(release_obj):
        raise ValueError("Post-close reporting is limited to CORE5 and MR capsule.")
    close_ts = _session_close_ts(decision_plan_obj.target_execution_timestamp_ts, release_obj.session_calendar_id_str)
    if as_of_ts < close_ts:
        return False
    with state_store_obj._connect() as connection_obj:
        connection_obj.execute("""CREATE TABLE IF NOT EXISTS daily_post_close_report (
            decision_plan_id_int INTEGER PRIMARY KEY, attempted_timestamp_str TEXT NOT NULL,
            status_str TEXT NOT NULL DEFAULT 'attempted', error_str TEXT)""")
        # A successful report is final. A failed attempt may be retried after
        # the next fresh broker observation. An in-progress attempt is not
        # stolen until its lease expires: broker calls may still be running.
        connection_obj.execute("BEGIN IMMEDIATE")
        cursor_obj = connection_obj.execute("""INSERT OR IGNORE INTO daily_post_close_report
            (decision_plan_id_int,attempted_timestamp_str) VALUES (?,?)""",
            (decision_plan_obj.decision_plan_id_int, as_of_ts.isoformat()))
        if cursor_obj.rowcount == 1:
            return True
        row_obj = connection_obj.execute("""SELECT status_str,attempted_timestamp_str FROM daily_post_close_report
            WHERE decision_plan_id_int=?""", (decision_plan_obj.decision_plan_id_int,)).fetchone()
        if row_obj is None or not _retry_due_bool(row_obj[0], row_obj[1], close_ts, as_of_ts):
            return False
        cursor_obj = connection_obj.execute("""UPDATE daily_post_close_report
            SET attempted_timestamp_str=?,status_str='attempted',error_str=NULL
            WHERE decision_plan_id_int=? AND status_str=? AND attempted_timestamp_str=?""",
            (as_of_ts.isoformat(), decision_plan_obj.decision_plan_id_int, row_obj[0], row_obj[1]))
        return cursor_obj.rowcount == 1


def get_retry_post_close_report_decision_id_list(state_store_obj, as_of_ts, pod_id_str=None, env_mode_str=None):
    with state_store_obj._connect() as connection_obj:
        if not connection_obj.execute("SELECT 1 FROM sqlite_master WHERE name='daily_post_close_report'").fetchone():
            return []
        row_list = connection_obj.execute("""SELECT r.decision_plan_id_int,r.status_str,r.attempted_timestamp_str,
                p.target_execution_timestamp_str,l.session_calendar_id_str
            FROM daily_post_close_report r JOIN decision_plan p
                ON p.decision_plan_id_int=r.decision_plan_id_int
            JOIN live_release l ON l.release_id_str=p.release_id_str
            WHERE (? IS NULL OR p.pod_id_str=?) AND (? IS NULL OR l.mode_str=?)
                AND l.enabled_bool=1 AND r.status_str IN ('failed','attempted')""",
            (pod_id_str, pod_id_str, env_mode_str, env_mode_str)).fetchall()
    retry_id_list = []
    for decision_id_int, status_str, attempted_timestamp_str, target_timestamp_str, calendar_id_str in row_list:
        try:
            target_ts = datetime.fromisoformat(target_timestamp_str)
            if target_ts.tzinfo is None:
                target_ts = target_ts.replace(tzinfo=timezone.utc)
            close_ts = _session_close_ts(target_ts, calendar_id_str)
        except Exception:
            # Optional reporting: an unreadable row is skipped, never allowed
            # to stop the pod's reconcile pass.
            continue
        if _retry_due_bool(status_str, attempted_timestamp_str, close_ts, as_of_ts):
            retry_id_list.append(int(decision_id_int))
    return retry_id_list


def finish_post_close_report_attempt(state_store_obj, decision_plan_obj, error_str=None, *, retryable_bool=True):
    """A non-retryable error is recorded but final: the report is done."""
    with state_store_obj._connect() as connection_obj:
        # Only an open attempt is finished: a worker whose lease was taken over
        # cannot turn another worker's final report back into a failure.
        connection_obj.execute("""UPDATE daily_post_close_report SET status_str=?,error_str=?
            WHERE decision_plan_id_int=? AND status_str='attempted'""",
            ("failed" if error_str and retryable_bool else "reported", error_str,
                decision_plan_obj.decision_plan_id_int))
