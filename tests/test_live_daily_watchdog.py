"""The daily watchdog reads saved state and deduplicates by pod/session."""
from dataclasses import replace
from datetime import timedelta
import json

from alpha.live import daily_watchdog
from alpha.live.daily_notifications import deliver_daily_alerts, enqueue_daily_cycle_overdue
from test_live_daily_reconcile import CLOSE_TS, OPEN_TS, daily_case


def _alerts(monkeypatch, daily_case, tmp_path, as_of_ts, *, release_obj=None):
    store_obj, saved_release_obj, _, _, _ = daily_case
    release_obj = release_obj or saved_release_obj
    monkeypatch.setattr(daily_watchdog, "load_release_list", lambda _root_str: [release_obj])
    monkeypatch.setattr(daily_watchdog.dashboard, "resolve_db_path_for_release_str",
        lambda _release_obj, _config_obj: store_obj.db_path_str)
    return daily_watchdog.daily_heartbeat_alert_list(str(tmp_path), str(tmp_path / "none.yaml"),
        str(tmp_path / "events.jsonl"), as_of_ts, release_obj.mode_str)


def test_decision_heartbeat_waits_until_next_open_minus_two_minutes(daily_case, monkeypatch, tmp_path):
    store_obj, _, decision_obj, _, _ = daily_case
    store_obj.mark_decision_plan_status(decision_obj.decision_plan_id_int, "blocked")
    event_path_obj = tmp_path / "events.jsonl"
    event_path_obj.write_text(json.dumps({"pod_id_str": "daily_pod", "error_str": "unknown foreign symbol"}) + "\n",
        encoding="utf-8")
    assert _alerts(monkeypatch, daily_case, tmp_path, OPEN_TS - timedelta(minutes=2, seconds=1)) == []
    alert_dict, = _alerts(monkeypatch, daily_case, tmp_path, OPEN_TS - timedelta(minutes=2))
    assert alert_dict == {"mode_str": "paper", "pod_id_str": "daily_pod",
        "session_str": "2026-10-02", "kind_str": "decision_incomplete",
        "last_error_str": "unknown foreign symbol"}


def test_saved_planned_decision_counts_as_completed_decision_step(daily_case, monkeypatch, tmp_path):
    store_obj, _, decision_obj, _, _ = daily_case
    store_obj.mark_decision_plan_status(decision_obj.decision_plan_id_int, "planned")
    assert _alerts(monkeypatch, daily_case, tmp_path, OPEN_TS - timedelta(minutes=2)) == []


def test_late_watchdog_run_recovers_prior_session_miss(daily_case, monkeypatch, tmp_path):
    as_of_ts = OPEN_TS + timedelta(days=1, hours=8)
    alert_list = _alerts(monkeypatch, daily_case, tmp_path, as_of_ts)
    assert any(alert_dict["session_str"] == "2026-10-05" and
        alert_dict["kind_str"] == "decision_incomplete" for alert_dict in alert_list)


def test_last_error_never_uses_another_pods_colliding_decision_id(tmp_path):
    log_path_obj = tmp_path / "events.jsonl"
    log_path_obj.write_text("\n".join(json.dumps(row_dict) for row_dict in [
        {"pod_id_str": "core5", "decision_plan_id_int": 1, "error_str": "core5 data missing"},
        {"pod_id_str": "other", "decision_plan_id_int": 1, "error_str": "foreign pod error"}]) + "\n",
        encoding="utf-8")
    assert daily_watchdog._last_error_str(str(log_path_obj), "core5", 1) == "core5 data missing"


def test_missing_db_and_disabled_release_still_alert_at_cutoff(daily_case, monkeypatch, tmp_path):
    _, release_obj, _, _, _ = daily_case
    monkeypatch.setattr(daily_watchdog.dashboard, "resolve_db_path_for_release_str",
        lambda _release_obj, _config_obj: str(tmp_path / "missing.sqlite3"))
    monkeypatch.setattr(daily_watchdog, "load_release_list", lambda _root_str: [replace(release_obj, enabled_bool=False)])
    assert daily_watchdog.daily_heartbeat_alert_list(str(tmp_path), str(tmp_path / "none.yaml"),
        str(tmp_path / "events.jsonl"), OPEN_TS - timedelta(minutes=2), "paper")[0]["kind_str"] == "decision_incomplete"


def test_disabled_release_with_healthy_db_alerts_for_missing_decision(daily_case, monkeypatch, tmp_path):
    _, release_obj, _, _, _ = daily_case
    as_of_ts = OPEN_TS + timedelta(days=1) - timedelta(minutes=2)
    alert_list = _alerts(monkeypatch, daily_case, tmp_path, as_of_ts,
        release_obj=replace(release_obj, enabled_bool=False))
    assert any(alert_dict["session_str"] == "2026-10-05"
        and alert_dict["last_error_str"] == "Release disabled." for alert_dict in alert_list)


def test_serve_outage_with_intact_db_reports_latest_pod_error(daily_case, monkeypatch, tmp_path):
    event_path_obj = tmp_path / "events.jsonl"
    event_path_obj.write_text(json.dumps({"pod_id_str": "daily_pod", "error_str": "serve down"}) + "\n",
        encoding="utf-8")
    as_of_ts = OPEN_TS + timedelta(days=1) - timedelta(minutes=2)
    alert_list = _alerts(monkeypatch, daily_case, tmp_path, as_of_ts)
    assert any(alert_dict["session_str"] == "2026-10-05"
        and alert_dict["last_error_str"] == "serve down" for alert_dict in alert_list)


def test_open_cycle_alert_after_exchange_close_plus_one_hour(daily_case, monkeypatch, tmp_path):
    store_obj, release_obj, decision_obj, _, _ = daily_case
    assert _alerts(monkeypatch, daily_case, tmp_path, CLOSE_TS + timedelta(minutes=59)) == []
    alert_dict, = _alerts(monkeypatch, daily_case, tmp_path, CLOSE_TS + timedelta(hours=1))
    assert alert_dict["session_str"] == "2026-10-05"
    assert alert_dict["kind_str"] == "cycle_open_after_close"
    enqueue_daily_cycle_overdue(store_obj, release_obj, decision_obj, CLOSE_TS + timedelta(hours=1),
        error_str="broker unavailable")
    assert _alerts(monkeypatch, daily_case, tmp_path, CLOSE_TS + timedelta(hours=1)) == []


def test_early_close_cycle_alert_uses_exchange_close(daily_case, monkeypatch, tmp_path):
    store_obj, _, decision_obj, _, _ = daily_case
    target_ts = OPEN_TS.replace(month=11, day=27)
    close_ts = target_ts.replace(hour=13, minute=0)
    with store_obj._connect() as connection_obj:
        connection_obj.execute("""UPDATE decision_plan SET target_execution_timestamp_str=?
            WHERE decision_plan_id_int=?""", (target_ts.isoformat(), decision_obj.decision_plan_id_int))
    before_list = _alerts(monkeypatch, daily_case, tmp_path, close_ts + timedelta(minutes=59))
    after_list = _alerts(monkeypatch, daily_case, tmp_path, close_ts + timedelta(hours=1))
    assert not any(alert_dict["kind_str"] == "cycle_open_after_close" for alert_dict in before_list)
    assert any(alert_dict["kind_str"] == "cycle_open_after_close"
        and alert_dict["session_str"] == "2026-11-27" for alert_dict in after_list)


def test_successful_heartbeat_is_one_per_pod_session_and_failed_send_retries(tmp_path):
    alert_list = [{"mode_str": "live", "pod_id_str": "core5", "session_str": "2026-10-05",
        "kind_str": "decision_incomplete", "last_error_str": "serve down"},
        {"mode_str": "live", "pod_id_str": "core5", "session_str": "2026-10-05",
        "kind_str": "cycle_open_after_close", "last_error_str": "load failure"}]
    state_path_str = str(tmp_path / "watchdog.daily.sqlite3")
    sent_list = []
    def poster_fn(_url_str, payload_dict):
        sent_list.append(payload_dict)
        return len(sent_list) > 1
    assert daily_watchdog.deliver_daily_heartbeat_alerts(alert_list[:1], state_path_str, "webhook", poster_fn) == []
    assert daily_watchdog.deliver_daily_heartbeat_alerts(alert_list, state_path_str, "webhook", poster_fn) != []
    assert len(sent_list) == 2
    assert daily_watchdog.deliver_daily_heartbeat_alerts(alert_list, state_path_str, "webhook", poster_fn) == []
    assert len(sent_list) == 2
    assert daily_watchdog.load_delivered_key_set(state_path_str) == {"live|core5|2026-10-05"}


def test_existing_overdue_outbox_is_suppressed_after_heartbeat_delivery(daily_case, tmp_path):
    store_obj, release_obj, decision_obj, _, _ = daily_case
    state_path_str = str(tmp_path / "watchdog.daily.sqlite3")
    alert_list = [{"mode_str": "paper", "pod_id_str": release_obj.pod_id_str,
        "session_str": "2026-10-05", "kind_str": "cycle_open_after_close",
        "last_error_str": "serve down"}]
    posted_list = []
    def poster_fn(_url_str, payload_dict):
        posted_list.append(payload_dict)
        return True
    daily_watchdog.deliver_daily_heartbeat_alerts(alert_list, state_path_str, "webhook", poster_fn)
    assert len(posted_list) == 1
    enqueue_daily_cycle_overdue(store_obj, release_obj, decision_obj, CLOSE_TS + timedelta(hours=1),
        error_str="serve recovered")
    summary_dict = {"pod_row_dict_list": [{"db_path_str": store_obj.db_path_str,
        "pod_id_str": release_obj.pod_id_str, "account_route_str": release_obj.account_route_str,
        "mode_str": release_obj.mode_str, "strategy_import_str": release_obj.strategy_import_str}]}
    delivery_list = deliver_daily_alerts(summary_dict, webhook_url_str="webhook", webhook_poster_fn=poster_fn,
        mode_str="paper", suppressed_session_key_set=daily_watchdog.load_delivered_key_set(state_path_str))
    assert len(delivery_list) == 1 and delivery_list[0].delivered_bool
    assert len(posted_list) == 1
