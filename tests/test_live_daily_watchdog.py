"""The daily watchdog reads saved state and deduplicates by pod/session."""
from dataclasses import replace
from datetime import timedelta
import json
import sqlite3

import pytest

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


def test_disabled_release_never_checks_heartbeat_or_database(daily_case, monkeypatch, tmp_path):
    _, release_obj, _, _, _ = daily_case
    monkeypatch.setattr(daily_watchdog, "load_release_list", lambda _root_str: [replace(release_obj, enabled_bool=False)])
    monkeypatch.setattr(daily_watchdog.scheduler_utils, "get_latest_completed_session_label_ts",
        lambda *_args, **_kwargs: pytest.fail("Disabled release must not inspect heartbeat timing"))
    monkeypatch.setattr(daily_watchdog, "_decision_state_tuple",
        lambda *_args: pytest.fail("Disabled release must not inspect its database"))
    assert daily_watchdog.daily_heartbeat_alert_list(str(tmp_path), str(tmp_path / "none.yaml"),
        str(tmp_path / "events.jsonl"), CLOSE_TS + timedelta(days=1, hours=1), "paper") == []


def test_disabled_release_with_healthy_db_does_not_alert_for_missing_decision(daily_case, monkeypatch, tmp_path):
    _, release_obj, _, _, _ = daily_case
    as_of_ts = OPEN_TS + timedelta(days=1) - timedelta(minutes=2)
    alert_list = _alerts(monkeypatch, daily_case, tmp_path, as_of_ts,
        release_obj=replace(release_obj, enabled_bool=False))
    assert alert_list == []


def test_enabled_release_with_missing_db_still_alerts_at_cutoff(daily_case, monkeypatch, tmp_path):
    _, release_obj, _, _, _ = daily_case
    monkeypatch.setattr(daily_watchdog.dashboard, "resolve_db_path_for_release_str",
        lambda _release_obj, _config_obj: str(tmp_path / "missing.sqlite3"))
    monkeypatch.setattr(daily_watchdog, "load_release_list", lambda _root_str: [release_obj])
    alert_dict, = daily_watchdog.daily_heartbeat_alert_list(str(tmp_path), str(tmp_path / "none.yaml"),
        str(tmp_path / "events.jsonl"), OPEN_TS - timedelta(minutes=2), "paper")
    assert alert_dict["kind_str"] == "decision_incomplete"
    assert alert_dict["last_error_str"] == "Daily pod state DB is missing."


def test_heartbeat_error_uses_configured_log_not_default(daily_case, monkeypatch, tmp_path):
    store_obj, release_obj, decision_obj, _, _ = daily_case
    store_obj.mark_decision_plan_status(decision_obj.decision_plan_id_int, "blocked")
    configured_path_obj = tmp_path / "daily-configured.jsonl"
    default_path_obj = tmp_path / "legacy-default.jsonl"
    for path_obj, error_str in [(configured_path_obj, "configured daily failure"),
            (default_path_obj, "unrelated default log")]:
        path_obj.write_text(json.dumps({"pod_id_str": release_obj.pod_id_str,
            "decision_plan_id_int": decision_obj.decision_plan_id_int, "error_str": error_str}) + "\n", encoding="utf-8")
    monkeypatch.setattr(daily_watchdog.dashboard, "DEFAULT_EVENT_LOG_PATH_STR", str(default_path_obj))
    monkeypatch.setattr(daily_watchdog, "load_release_list", lambda _root_str: [release_obj])
    monkeypatch.setattr(daily_watchdog.dashboard, "resolve_db_path_for_release_str",
        lambda _release_obj, _config_obj: store_obj.db_path_str)
    alert_dict, = daily_watchdog.daily_heartbeat_alert_list(str(tmp_path), str(tmp_path / "none.yaml"),
        str(configured_path_obj), OPEN_TS - timedelta(minutes=2), "paper")
    assert alert_dict["last_error_str"] == "configured daily failure"


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


@pytest.mark.parametrize("raise_http_bool", [False, True])
def test_explicit_delivery_failure_is_visible_after_other_alerts_and_releases_claim(tmp_path, raise_http_bool):
    alert_list = [{"mode_str": "live", "pod_id_str": pod_id_str, "session_str": "2026-10-05",
        "kind_str": "decision_incomplete", "last_error_str": "serve down"}
        for pod_id_str in ("failed_pod", "healthy_pod")]
    state_path_str = str(tmp_path / "watchdog.daily.sqlite3")
    posted_list = []

    def poster_fn(_url_str, payload_dict):
        posted_list.append(payload_dict)
        if "failed_pod" in payload_dict["content"]:
            if raise_http_bool:
                raise OSError("private transport detail")
            return False
        return True

    with pytest.raises(daily_watchdog.DailyHeartbeatDeliveryError, match="1 daily heartbeat") as error_info:
        daily_watchdog.deliver_daily_heartbeat_alerts(alert_list, state_path_str, "webhook", poster_fn,
            raise_on_failure_bool=True)
    assert error_info.value.failed_count_int == 1
    assert "private transport detail" not in str(error_info.value)
    assert len(posted_list) == 2
    assert daily_watchdog.load_delivered_key_set(state_path_str) == {"live|healthy_pod|2026-10-05"}
    with sqlite3.connect(state_path_str) as connection_obj:
        assert connection_obj.execute("SELECT delivery_claim_str, delivery_claimed_timestamp_str, "
            "delivered_timestamp_str FROM daily_watchdog_alert WHERE alert_key_str=?",
            ("live|failed_pod|2026-10-05",)).fetchone() == (None, None, None)

    retry_list = []
    assert daily_watchdog.deliver_daily_heartbeat_alerts(alert_list, state_path_str, "webhook",
        lambda _url_str, payload_dict: retry_list.append(payload_dict) or True,
        raise_on_failure_bool=True) == alert_list[:1]
    assert len(retry_list) == 1


@pytest.mark.parametrize("operation_str", ["load", "deliver"])
def test_watchdog_owned_sqlite_failure_reaches_caller(tmp_path, operation_str):
    state_path_obj = tmp_path / "watchdog.daily.sqlite3"
    state_path_obj.write_bytes(b"not a SQLite database")
    with pytest.raises(sqlite3.DatabaseError):
        if operation_str == "load":
            daily_watchdog.load_delivered_key_set(str(state_path_obj))
        else:
            alert_list = [{"mode_str": "live", "pod_id_str": "core5", "session_str": "2026-10-05",
                "kind_str": "decision_incomplete", "last_error_str": "serve down"}]
            daily_watchdog.deliver_daily_heartbeat_alerts(alert_list, str(state_path_obj), "webhook",
                lambda *_args: pytest.fail("Invalid state must fail before HTTP"), raise_on_failure_bool=True)
