"""Daily reporting queries run at most once, after the exchange session closes."""
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
        # Commit the attempt before broker I/O. Reporting must not become another
        # repeated history request after a failed call or a process restart.
        cursor_obj = connection_obj.execute("""INSERT OR IGNORE INTO daily_post_close_report
            (decision_plan_id_int,attempted_timestamp_str) VALUES (?,?)""",
            (decision_plan_obj.decision_plan_id_int, as_of_ts.isoformat()))
        return cursor_obj.rowcount == 1


def finish_post_close_report_attempt(state_store_obj, decision_plan_obj, error_str=None):
    with state_store_obj._connect() as connection_obj:
        connection_obj.execute("""UPDATE daily_post_close_report SET status_str=?,error_str=?
            WHERE decision_plan_id_int=?""", ("failed" if error_str else "reported", error_str,
                decision_plan_obj.decision_plan_id_int))
