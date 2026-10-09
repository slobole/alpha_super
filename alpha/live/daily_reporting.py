"""Daily post-close reporting retries failed reads, never a successful report."""
from alpha.live import scheduler_utils
from alpha.live.daily_reconcile import is_daily_reconcile_release_bool


def claim_post_close_report_attempt(state_store_obj, release_obj, decision_plan_obj, as_of_ts):
    if not is_daily_reconcile_release_bool(release_obj):
        raise ValueError("Post-close reporting is limited to CORE5 and MR capsule.")
    session_date_obj = scheduler_utils.to_market_timestamp_ts(
        decision_plan_obj.target_execution_timestamp_ts, release_obj.session_calendar_id_str).date()
    close_ts = scheduler_utils.get_session_close_timestamp_ts(session_date_obj, release_obj.session_calendar_id_str)
    if as_of_ts < close_ts:
        return False
    with state_store_obj._connect() as connection_obj:
        connection_obj.execute("""CREATE TABLE IF NOT EXISTS daily_post_close_report (
            decision_plan_id_int INTEGER PRIMARY KEY, attempted_timestamp_str TEXT NOT NULL,
            status_str TEXT NOT NULL DEFAULT 'attempted', error_str TEXT)""")
        # A successful report is final. A failed attempt may be retried after
        # the next fresh broker observation. An in-progress attempt is not
        # stolen: broker history/open calls may still be running.
        connection_obj.execute("BEGIN IMMEDIATE")
        cursor_obj = connection_obj.execute("""INSERT OR IGNORE INTO daily_post_close_report
            (decision_plan_id_int,attempted_timestamp_str) VALUES (?,?)""",
            (decision_plan_obj.decision_plan_id_int, as_of_ts.isoformat()))
        if cursor_obj.rowcount == 1:
            return True
        cursor_obj = connection_obj.execute("""UPDATE daily_post_close_report
            SET attempted_timestamp_str=?,status_str='attempted',error_str=NULL
            WHERE decision_plan_id_int=? AND status_str='failed'""",
            (as_of_ts.isoformat(), decision_plan_obj.decision_plan_id_int))
        return cursor_obj.rowcount == 1


def get_retry_post_close_report_decision_id_list(state_store_obj, as_of_ts, pod_id_str=None, env_mode_str=None):
    with state_store_obj._connect() as connection_obj:
        if not connection_obj.execute("SELECT 1 FROM sqlite_master WHERE name='daily_post_close_report'").fetchone():
            return []
        return [int(row_obj[0]) for row_obj in connection_obj.execute("""SELECT r.decision_plan_id_int
            FROM daily_post_close_report r JOIN decision_plan p
                ON p.decision_plan_id_int=r.decision_plan_id_int
            JOIN live_release l ON l.release_id_str=p.release_id_str
            WHERE (? IS NULL OR p.pod_id_str=?) AND (? IS NULL OR l.mode_str=?)
                AND l.enabled_bool=1 AND r.status_str='failed'""",
            (pod_id_str, pod_id_str, env_mode_str, env_mode_str))]


def finish_post_close_report_attempt(state_store_obj, decision_plan_obj, error_str=None):
    with state_store_obj._connect() as connection_obj:
        connection_obj.execute("""UPDATE daily_post_close_report SET status_str=?,error_str=?
            WHERE decision_plan_id_int=?""", ("failed" if error_str else "reported", error_str,
                decision_plan_obj.decision_plan_id_int))
