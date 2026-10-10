"""Daily post-close reporting retries failed reads, never a successful report.

Retries stop REPORT_RETRY_WINDOW after the target session close. An attempt
left 'attempted' by a stopped process is retried once its lease expires.
"""
from datetime import datetime, timedelta, timezone

from alpha.live import scheduler_utils
from alpha.live.daily_reconcile import is_daily_reconcile_release_bool


REPORT_ATTEMPT_LEASE = timedelta(minutes=30)
REPORT_RETRY_WINDOW = timedelta(hours=24)


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
