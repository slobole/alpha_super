"""Read-only daily-pod decision and settlement heartbeat for the OPS watchdog."""
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sqlite3
from uuid import uuid4

from alpha.live import dashboard, scheduler_utils
from alpha.live.daily_reconcile import DAILY_TERMINAL_STATUS_SET, is_daily_reconcile_release_bool
from alpha.live.release_manifest import load_release_list


class DailyHeartbeatDeliveryError(RuntimeError):
    """An explicit daily-mode HTTP failure remains retryable."""

    def __init__(self, failed_count_int):
        self.failed_count_int = failed_count_int
        super().__init__(f"{failed_count_int} daily heartbeat alert delivery attempt(s) failed.")


def _last_error_str(event_log_path_str, pod_id_str, decision_plan_id_int=None):
    log_path_obj = Path(event_log_path_str)
    for candidate_path_obj in reversed(dashboard._event_log_path_obj_list(log_path_obj)):
        for event_dict in dashboard._iter_event_dict_reverse(candidate_path_obj):
            if event_dict.get("pod_id_str") != pod_id_str:
                continue
            if (decision_plan_id_int is not None and event_dict.get("decision_plan_id_int") is not None
                    and event_dict["decision_plan_id_int"] != decision_plan_id_int):
                continue
            error_str = event_dict.get("error_str")
            if error_str:
                return " ".join(str(error_str).split())[:400]
    return "No error recorded; check the serve and daily event log."


def _decision_state_tuple(db_path_str, release_obj):
    db_path_obj = Path(db_path_str)
    if not db_path_obj.is_file():
        raise FileNotFoundError("Daily pod state DB is missing.")
    with closing(sqlite3.connect(db_path_obj.resolve().as_uri() + "?mode=ro", uri=True)) as connection_obj:
        connection_obj.row_factory = sqlite3.Row
        decision_row_list = [dict(row_obj) for row_obj in connection_obj.execute("""SELECT decision_plan_id_int,
            signal_timestamp_str,target_execution_timestamp_str,status_str FROM decision_plan
            WHERE release_id_str=? AND pod_id_str=? AND account_route_str=?
            ORDER BY decision_plan_id_int DESC""", (release_obj.release_id_str,
                release_obj.pod_id_str, release_obj.account_route_str))]
        existing_overdue_id_set, existing_halt_session_set = set(), set()
        if connection_obj.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='daily_pod_alert'").fetchone():
            for row_obj in connection_obj.execute("""SELECT alert_kind_str,decision_plan_id_int,alert_key_str
                    FROM daily_pod_alert WHERE pod_id_str=? AND account_route_str=? AND mode_str=?""",
                    (release_obj.pod_id_str, release_obj.account_route_str, release_obj.mode_str)):
                if row_obj["alert_kind_str"] == "cycle_overdue":
                    existing_overdue_id_set.add(row_obj["decision_plan_id_int"])
                elif row_obj["alert_kind_str"] == "capsule_holding_halt":
                    existing_halt_session_set.add(row_obj["alert_key_str"].removeprefix("capsule_holding_halt:"))
        return decision_row_list, existing_overdue_id_set, existing_halt_session_set


def daily_heartbeat_alert_list(releases_root_path_str, dashboard_config_path_str,
        event_log_path_str, as_of_ts, mode_str=None):
    """Find missed decisions and open cycles without initializing or writing pod DBs."""
    config_obj = dashboard.load_dashboard_config(dashboard_config_path_str)
    alert_list = []
    for release_obj in load_release_list(releases_root_path_str):
        if (not release_obj.enabled_bool or not is_daily_reconcile_release_bool(release_obj)
                or mode_str not in (None, "all", release_obj.mode_str)):
            continue
        calendar_id_str = release_obj.session_calendar_id_str
        signal_label_ts = scheduler_utils.get_latest_completed_session_label_ts(as_of_ts,
            calendar_id_str, snapshot_ready_buffer_minutes_int=0)
        if signal_label_ts is None:
            continue
        try:
            decision_row_list, existing_overdue_id_set, existing_halt_session_set = _decision_state_tuple(
                dashboard.resolve_db_path_for_release_str(release_obj, config_obj), release_obj)
            load_error_str = None
        except (OSError, sqlite3.DatabaseError) as error_obj:
            decision_row_list = []
            existing_overdue_id_set, existing_halt_session_set = set(), set()
            load_error_str = " ".join(str(error_obj).split())[:400]
        # *** CRITICAL *** A signal-session decision is due only at the next
        # session's 09:28 market cutoff; later sessions cannot satisfy it.
        # Keep a bounded lookback so a watchdog outage across the next close
        # cannot erase a previously missed session. Before the first saved
        # decision, inspect only the latest session to avoid pre-launch alerts.
        earliest_decision_date_obj = min((scheduler_utils.to_market_timestamp_ts(
            dashboard._parse_timestamp_ts(row_dict["signal_timestamp_str"]), calendar_id_str).date()
            for row_dict in decision_row_list), default=None)
        candidate_label_list = [signal_label_ts]
        if earliest_decision_date_obj is not None:
            calendar_obj = scheduler_utils.get_exchange_calendar_obj(calendar_id_str)
            for _index_int in range(4):
                prior_label_ts = calendar_obj.previous_session(candidate_label_list[-1])
                if prior_label_ts.date() < earliest_decision_date_obj:
                    break
                candidate_label_list.append(prior_label_ts)
        for candidate_label_ts in candidate_label_list:
            next_candidate_label_ts = scheduler_utils.get_next_session_label_ts(candidate_label_ts, calendar_id_str)
            candidate_cutoff_ts = scheduler_utils.get_session_open_timestamp_ts(
                next_candidate_label_ts, calendar_id_str) - timedelta(minutes=2)
            if as_of_ts < candidate_cutoff_ts:
                continue
            matching_row_dict = next((row_dict for row_dict in decision_row_list
                if scheduler_utils.to_market_timestamp_ts(
                    dashboard._parse_timestamp_ts(row_dict["signal_timestamp_str"]), calendar_id_str).date()
                    == candidate_label_ts.date()), None)
            if ((matching_row_dict is None or matching_row_dict["status_str"] in {"blocked", "expired", "superseded"})
                    and candidate_label_ts.date().isoformat() not in existing_halt_session_set):
                last_error_str = load_error_str or _last_error_str(event_log_path_str, release_obj.pod_id_str,
                    None if matching_row_dict is None else matching_row_dict["decision_plan_id_int"])
                alert_list.append({"mode_str": release_obj.mode_str, "pod_id_str": release_obj.pod_id_str,
                    "session_str": candidate_label_ts.date().isoformat(), "kind_str": "decision_incomplete",
                    "last_error_str": last_error_str})
        # *** CRITICAL *** A target execution session is overdue only after
        # that exchange session's own close plus one hour (including early closes).
        for row_dict in decision_row_list:
            if row_dict["status_str"] in DAILY_TERMINAL_STATUS_SET | {"superseded"}:
                continue
            if row_dict["decision_plan_id_int"] in existing_overdue_id_set:
                continue
            target_ts = dashboard._parse_timestamp_ts(row_dict["target_execution_timestamp_str"])
            target_label_ts = scheduler_utils.session_label_from_timestamp_ts(target_ts, calendar_id_str)
            if target_label_ts is None:
                continue
            close_ts = scheduler_utils.get_session_close_timestamp_ts(target_label_ts, calendar_id_str)
            if as_of_ts < close_ts + timedelta(hours=1):
                continue
            alert_list.append({"mode_str": release_obj.mode_str, "pod_id_str": release_obj.pod_id_str,
                "session_str": target_label_ts.date().isoformat(), "kind_str": "cycle_open_after_close",
                "last_error_str": load_error_str or _last_error_str(event_log_path_str,
                    release_obj.pod_id_str, row_dict["decision_plan_id_int"])})
    return alert_list


def load_delivered_key_set(state_path_str):
    state_path_obj = Path(state_path_str)
    if not state_path_obj.is_file():
        return set()
    with closing(sqlite3.connect(state_path_obj)) as connection_obj:
        return {row_obj[0] for row_obj in connection_obj.execute(
            "SELECT alert_key_str FROM daily_watchdog_alert WHERE delivered_timestamp_str IS NOT NULL")}


def deliver_daily_heartbeat_alerts(alert_list, state_path_str, webhook_url_str, webhook_poster_fn,
        *, raise_on_failure_bool=False):
    """Claim one pod/session; opt-in failures raise after other delivery attempts."""
    if not alert_list or not webhook_url_str:
        return []
    state_path_obj = Path(state_path_str)
    state_path_obj.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(state_path_obj)) as connection_obj, connection_obj:
        connection_obj.execute("""CREATE TABLE IF NOT EXISTS daily_watchdog_alert (
            alert_key_str TEXT PRIMARY KEY, delivery_claim_str TEXT,
            delivery_claimed_timestamp_str TEXT, delivered_timestamp_str TEXT)""")
    delivered_list = []
    failed_count_int = 0
    for alert_dict in alert_list:
        key_str = "|".join((alert_dict["mode_str"], alert_dict["pod_id_str"], alert_dict["session_str"]))
        claim_str = uuid4().hex
        claim_ts = datetime.now(timezone.utc)
        with closing(sqlite3.connect(state_path_obj, timeout=5.0)) as connection_obj, connection_obj:
            connection_obj.execute("BEGIN IMMEDIATE")
            connection_obj.execute("INSERT OR IGNORE INTO daily_watchdog_alert(alert_key_str) VALUES (?)", (key_str,))
            cursor_obj = connection_obj.execute("""UPDATE daily_watchdog_alert
                SET delivery_claim_str=?,delivery_claimed_timestamp_str=?
                WHERE alert_key_str=? AND delivered_timestamp_str IS NULL AND
                    (delivery_claim_str IS NULL OR delivery_claimed_timestamp_str<=?)""",
                (claim_str, claim_ts.isoformat(), key_str,
                    (claim_ts - timedelta(minutes=15)).isoformat()))
            if cursor_obj.rowcount != 1:
                continue
        payload_dict = {"content": (f"CRITICAL DAILY / {alert_dict['mode_str'].upper()} / {alert_dict['pod_id_str']}\n"
            f"Session {alert_dict['session_str']}: {alert_dict['kind_str']}.\n"
            f"Last error: {alert_dict['last_error_str']}")[:1900], "allowed_mentions": {"parse": []}}
        try:
            delivered_bool = bool(webhook_poster_fn(webhook_url_str, payload_dict))
        except Exception:
            delivered_bool = False
        with closing(sqlite3.connect(state_path_obj, timeout=5.0)) as connection_obj, connection_obj:
            connection_obj.execute("""UPDATE daily_watchdog_alert SET delivery_claim_str=NULL,
                delivery_claimed_timestamp_str=NULL,delivered_timestamp_str=?
                WHERE alert_key_str=? AND delivery_claim_str=?""",
                (datetime.now(timezone.utc).isoformat() if delivered_bool else None, key_str, claim_str))
        if delivered_bool:
            delivered_list.append(alert_dict)
        else:
            failed_count_int += 1
    if failed_count_int and raise_on_failure_bool:
        raise DailyHeartbeatDeliveryError(failed_count_int)
    return delivered_list
