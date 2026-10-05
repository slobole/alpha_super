"""Durable capsule alerts using temporary SQLite and fake webhook transports."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from datetime import UTC, datetime, timedelta
import hashlib
import json
import sqlite3
from threading import Event
from types import SimpleNamespace

import pytest

from alpha.live import mr_capsule_notifications as alert_module
from test_live_ops_watchdog import _run_watchdog, _summary_dict, AS_OF_TS, HEARTBEAT_URL_STR


NOW_TS = datetime(2026, 10, 5, 15, 0, tzinfo=UTC)
STRATEGY_IMPORT_STR = "strategies.mr_capsule.strategy_mr_dv2_vix_gated_bil"


def _alert_payload_dict():
    return {"residual_row_dict_list": [{
        "asset_str": "MSFT", "requested_amount_float": -100.0,
        "filled_amount_float": -60.0, "residual_amount_float": -40.0,
        "status_str": "Cancelled", "required_action_str": "SELL",
        "reason_str": "Exit remains after the single completion attempt.",
    }], "broker_snapshot_dict": {"cash_float": -200.0, "available_funds_float": 20000.0}}


def _enqueue(connection_obj, *, vplan_id_int=1, alert_kind_str="accepted_residual",
        pod_id_str="capsule_one", account_route_str="DU_ONE", mode_str="paper", payload_dict=None):
    alert_module.enqueue_execution_alert(connection_obj, vplan_id_int=vplan_id_int,
        alert_kind_str=alert_kind_str, pod_id_str=pod_id_str,
        account_route_str=account_route_str, mode_str=mode_str,
        payload_dict=payload_dict if payload_dict is not None else _alert_payload_dict(), created_timestamp_ts=NOW_TS)


def _create_alert_db(tmp_path, **alert_kwarg_dict):
    db_path_obj = tmp_path / "capsule.sqlite3"
    with closing(sqlite3.connect(db_path_obj)) as connection_obj, connection_obj:
        alert_module.ensure_execution_alert_schema(connection_obj)
        _enqueue(connection_obj, **alert_kwarg_dict)
    return db_path_obj


def _capsule_summary_dict(db_path_obj, **row_override_dict):
    summary_dict = _summary_dict()
    summary_dict["pod_row_dict_list"][0].update(
        pod_id_str="capsule_one", account_route_str="DU_ONE", mode_str="paper",
        strategy_import_str=STRATEGY_IMPORT_STR, db_path_str=str(db_path_obj),
    )
    summary_dict["pod_row_dict_list"][0].update(row_override_dict)
    return summary_dict


def _read_alert_dict(db_path_obj, vplan_id_int=1):
    with closing(sqlite3.connect(db_path_obj)) as connection_obj:
        connection_obj.row_factory = sqlite3.Row
        return dict(connection_obj.execute("SELECT * FROM mr_capsule_execution_alert WHERE vplan_id_int = ?", (vplan_id_int,)).fetchone())


def test_schema_and_enqueue_join_existing_transaction_without_committing(tmp_path):
    db_path_obj = tmp_path / "transaction.sqlite3"
    with closing(sqlite3.connect(db_path_obj)) as connection_obj:
        connection_obj.execute("BEGIN IMMEDIATE")
        alert_module.ensure_execution_alert_schema(connection_obj)
        _enqueue(connection_obj)
        connection_obj.rollback()
        assert connection_obj.execute("SELECT name FROM sqlite_master WHERE name='mr_capsule_execution_alert'").fetchone() is None
        alert_module.ensure_execution_alert_schema(connection_obj)
        connection_obj.commit()
        connection_obj.execute("BEGIN IMMEDIATE")
        _enqueue(connection_obj)
        connection_obj.rollback()
        assert connection_obj.execute("SELECT COUNT(*) FROM mr_capsule_execution_alert").fetchone()[0] == 0


def test_same_event_deduplicates_but_new_cycles_and_kinds_are_distinct(tmp_path):
    db_path_obj = _create_alert_db(tmp_path)
    with closing(sqlite3.connect(db_path_obj)) as connection_obj, connection_obj:
        _enqueue(connection_obj, payload_dict={"changed": True})
        _enqueue(connection_obj, vplan_id_int=2)
        _enqueue(connection_obj, alert_kind_str="unresolved_execution")
        _enqueue(connection_obj, alert_kind_str="late_execution")
        assert connection_obj.execute("SELECT COUNT(*) FROM mr_capsule_execution_alert").fetchone()[0] == 4
        payload_str = connection_obj.execute("SELECT payload_json_str FROM mr_capsule_execution_alert WHERE vplan_id_int=1 AND alert_kind_str='accepted_residual'").fetchone()[0]
        assert json.loads(payload_str) == _alert_payload_dict()


def test_failed_delivery_survives_restart_and_completed_new_cycle_then_success_deduplicates(tmp_path):
    db_path_obj = _create_alert_db(tmp_path)
    summary_dict = _capsule_summary_dict(db_path_obj)
    first_delivery_list = alert_module.deliver_execution_alerts(summary_dict, webhook_url_str="test", webhook_poster_fn=lambda *_: False, now_ts=NOW_TS)
    assert not first_delivery_list[0].delivered_bool
    assert alert_module.pending_execution_alert_count_int(summary_dict) == 1
    assert _read_alert_dict(db_path_obj)["delivery_claim_str"] is None
    summary_dict["pod_row_dict_list"][0].update(latest_vplan_id_int=999, latest_vplan_status_str="completed")
    post_payload_list = []
    def post_webhook(url_str, payload_dict):
        post_payload_list.append(payload_dict)
        return True
    second_delivery_list = alert_module.deliver_execution_alerts(summary_dict, webhook_url_str="test", webhook_poster_fn=post_webhook, now_ts=NOW_TS)
    assert second_delivery_list[0].delivered_bool
    assert alert_module.deliver_execution_alerts(summary_dict, webhook_url_str="test", webhook_poster_fn=post_webhook, now_ts=NOW_TS) == []
    assert alert_module.pending_execution_alert_count_int(summary_dict) == 0
    assert len(post_payload_list) == 1
    assert _read_alert_dict(db_path_obj)["attempt_count_int"] == 2
    content_str = post_payload_list[0]["content"]
    assert all(text_str in content_str for text_str in ["MSFT", "SELL", "40", "Exit remains", "cash_float=-200.0"])
    assert post_payload_list[0]["allowed_mentions"] == {"parse": []}


def test_missing_webhook_keeps_pending_without_claiming_or_incrementing(tmp_path):
    db_path_obj = _create_alert_db(tmp_path)
    summary_dict = _capsule_summary_dict(db_path_obj)
    assert alert_module.deliver_execution_alerts(summary_dict, webhook_url_str="", webhook_poster_fn=lambda *_: pytest.fail("Unexpected send")) == []
    assert _read_alert_dict(db_path_obj)["attempt_count_int"] == 0
    assert alert_module.pending_execution_alert_count_int(summary_dict) == 1


def test_concurrent_deliveries_claim_once_and_active_claim_counts_pending(tmp_path):
    db_path_obj = _create_alert_db(tmp_path)
    summary_dict = _capsule_summary_dict(db_path_obj)
    post_started_event, finish_post_event = Event(), Event()
    def blocked_post(url_str, payload_dict):
        post_started_event.set()
        assert finish_post_event.wait(timeout=5)
        return True
    with ThreadPoolExecutor(max_workers=2) as executor_obj:
        first_future_obj = executor_obj.submit(alert_module.deliver_execution_alerts, summary_dict,
            webhook_url_str="test", webhook_poster_fn=blocked_post, now_ts=NOW_TS)
        assert post_started_event.wait(timeout=5)
        try:
            assert alert_module.pending_execution_alert_count_int(summary_dict) == 1
            assert alert_module.deliver_execution_alerts(summary_dict, webhook_url_str="test",
                webhook_poster_fn=lambda *_: pytest.fail("Concurrent duplicate"), now_ts=NOW_TS) == []
        finally:
            finish_post_event.set()
        assert first_future_obj.result(timeout=5)[0].delivered_bool
    assert _read_alert_dict(db_path_obj)["attempt_count_int"] == 1


@pytest.mark.parametrize("lease_age_seconds_int, expected_count_int", [(299, 0), (300, 1)])
def test_abandoned_claim_recovers_only_after_lease_expiry(tmp_path, lease_age_seconds_int, expected_count_int):
    db_path_obj = _create_alert_db(tmp_path)
    with closing(sqlite3.connect(db_path_obj)) as connection_obj, connection_obj:
        connection_obj.execute("UPDATE mr_capsule_execution_alert SET delivery_claim_str='crashed', delivery_claimed_timestamp_str=?",
            ((NOW_TS - timedelta(seconds=lease_age_seconds_int)).isoformat(),))
    delivery_list = alert_module.deliver_execution_alerts(_capsule_summary_dict(db_path_obj), webhook_url_str="test", webhook_poster_fn=lambda *_: True, now_ts=NOW_TS)
    assert len(delivery_list) == expected_count_int


def test_old_sender_cannot_undo_new_claim_delivery(tmp_path):
    db_path_obj = _create_alert_db(tmp_path)
    summary_dict = _capsule_summary_dict(db_path_obj)
    def stale_sender(url_str, payload_dict):
        delivery_list = alert_module.deliver_execution_alerts(summary_dict, webhook_url_str="test",
            webhook_poster_fn=lambda *_: True, now_ts=NOW_TS + timedelta(seconds=301))
        assert delivery_list[0].delivered_bool
        return False
    alert_module.deliver_execution_alerts(summary_dict, webhook_url_str="test", webhook_poster_fn=stale_sender, now_ts=NOW_TS)
    alert_dict = _read_alert_dict(db_path_obj)
    assert alert_dict["delivered_timestamp_str"] == (NOW_TS + timedelta(seconds=301)).isoformat()
    assert alert_dict["attempt_count_int"] == 2


def test_transport_exception_preserves_pending_and_releases_owned_claim(tmp_path):
    db_path_obj = _create_alert_db(tmp_path)
    def failed_post(*_):
        raise RuntimeError("synthetic transport failure")
    with pytest.raises(RuntimeError, match="synthetic transport"):
        alert_module.deliver_execution_alerts(_capsule_summary_dict(db_path_obj), webhook_url_str="test", webhook_poster_fn=failed_post, now_ts=NOW_TS)
    assert _read_alert_dict(db_path_obj)["delivery_claim_str"] is None
    assert _read_alert_dict(db_path_obj)["delivered_timestamp_str"] is None


@pytest.mark.parametrize("row_override_dict", [
    {"pod_id_str": "other_pod"}, {"account_route_str": "DU_OTHER"}, {"mode_str": "live"},
    {"strategy_import_str": "strategies.taa_df.strategy_taa_df"},
])
def test_scope_requires_exact_configured_capsule_identity(tmp_path, row_override_dict):
    db_path_obj = _create_alert_db(tmp_path)
    summary_dict = _capsule_summary_dict(db_path_obj, **row_override_dict)
    assert alert_module.pending_execution_alert_count_int(summary_dict) == 0
    assert alert_module.deliver_execution_alerts(summary_dict, webhook_url_str="test", webhook_poster_fn=lambda *_: pytest.fail("Wrong scope")) == []
    assert _read_alert_dict(db_path_obj)["attempt_count_int"] == 0


def test_old_and_missing_databases_are_unchanged(tmp_path):
    missing_path_obj = tmp_path / "missing.sqlite3"
    old_path_obj = tmp_path / "legacy.sqlite3"
    with closing(sqlite3.connect(old_path_obj)) as connection_obj:
        connection_obj.execute("CREATE TABLE legacy (value_int INTEGER)")
    original_bytes = old_path_obj.read_bytes()
    for db_path_obj in [missing_path_obj, old_path_obj]:
        summary_dict = _capsule_summary_dict(db_path_obj)
        assert alert_module.pending_execution_alert_count_int(summary_dict) == 0
        assert alert_module.deliver_execution_alerts(summary_dict, webhook_url_str="test", webhook_poster_fn=lambda *_: pytest.fail("Unexpected send")) == []
    assert not missing_path_obj.exists()
    assert old_path_obj.read_bytes() == original_bytes


def test_scope_deduplicates_same_row_but_rejects_conflicting_account(tmp_path):
    db_path_obj = _create_alert_db(tmp_path)
    summary_dict = _capsule_summary_dict(db_path_obj)
    summary_dict["pod_row_dict_list"].append(dict(summary_dict["pod_row_dict_list"][0]))
    assert alert_module.pending_execution_alert_count_int(summary_dict) == 1
    summary_dict["pod_row_dict_list"][1]["account_route_str"] = "DU_OTHER"
    with pytest.raises(ValueError, match="Ambiguous"):
        alert_module.pending_execution_alert_count_int(summary_dict)


def test_oversized_row_continues_across_messages_without_dropping_evidence(tmp_path):
    payload_dict = _alert_payload_dict()
    payload_dict["residual_row_dict_list"][0]["reason_str"] = "@everyone " * 500
    db_path_obj = _create_alert_db(tmp_path, payload_dict=payload_dict)
    post_payload_list = []
    def post_webhook(url_str, delivered_payload_dict):
        post_payload_list.append(delivered_payload_dict)
        return True
    alert_module.deliver_execution_alerts(_capsule_summary_dict(db_path_obj), webhook_url_str="test", webhook_poster_fn=post_webhook, now_ts=NOW_TS)
    assert len(post_payload_list) > 1
    assert all(len(message_dict["content"]) <= 1900 for message_dict in post_payload_list)
    assert all(message_dict["allowed_mentions"] == {"parse": []} for message_dict in post_payload_list)
    reconstructed_str = "".join(message_dict["content"].split("\n", 1)[1] for message_dict in post_payload_list)
    assert "@everyone " * 500 in reconstructed_str
    assert "Truncated" not in reconstructed_str
    assert json.loads(_read_alert_dict(db_path_obj)["payload_json_str"]) == payload_dict


def test_watchdog_delivers_saved_warning_even_when_capsule_is_green_and_keeps_legacy_count(monkeypatch, tmp_path, capsys):
    db_path_obj = _create_alert_db(tmp_path)
    summary_dict = _capsule_summary_dict(db_path_obj)
    return_code_int, _, webhook_list, output_path_obj = _run_watchdog(monkeypatch, tmp_path,
        summary_dict=summary_dict, discord_webhook_url_str="test")
    assert return_code_int == 0 and len(webhook_list) == 1
    result_dict = json.loads(capsys.readouterr().out)
    receipt_dict = json.loads(output_path_obj.with_suffix(".run.json").read_text(encoding="utf-8"))
    for result_obj in [result_dict, receipt_dict]:
        assert result_obj["capsule_notification_attempt_count_int"] == 1
        assert result_obj["capsule_notification_pending_count_int"] == 0
        assert result_obj["capsule_notification_pending_live_count_int"] == 0
    assert result_dict["notification_fired_count_int"] == 0
    # CLI --as-of is historical; leases and delivery use the actual wall clock.
    assert datetime.fromisoformat(_read_alert_dict(db_path_obj)["delivered_timestamp_str"]) > AS_OF_TS


def test_watchdog_mode_filter_does_not_send_or_count_other_modes(monkeypatch, tmp_path, capsys):
    db_path_obj = _create_alert_db(tmp_path)
    with closing(sqlite3.connect(db_path_obj)) as connection_obj, connection_obj:
        _enqueue(connection_obj, vplan_id_int=2, pod_id_str="capsule_live", account_route_str="U_TWO", mode_str="live")
    summary_dict = _capsule_summary_dict(db_path_obj)
    summary_dict["pod_row_dict_list"].append(dict(summary_dict["pod_row_dict_list"][0],
        pod_id_str="capsule_live", account_route_str="U_TWO", mode_str="live"))
    return_code_int, _, webhook_list, output_path_obj = _run_watchdog(monkeypatch, tmp_path,
        summary_dict=summary_dict, discord_webhook_url_str="test", discord_delivery_bool=False,
        extra_argv_list=["--mode", "paper"])
    assert return_code_int == 0 and len(webhook_list) == 1
    result_dict = json.loads(capsys.readouterr().out)
    assert result_dict["capsule_notification_pending_count_int"] == 1
    assert result_dict["capsule_notification_pending_live_count_int"] == 0
    assert _read_alert_dict(db_path_obj, 2)["attempt_count_int"] == 0


@pytest.mark.parametrize("problem_str", ["invalid_scope", "ambiguous_scope", "shared_account", "sqlite", "pending_sqlite"])
@pytest.mark.parametrize("legacy_severity_str", ["green", "red"])
def test_watchdog_capsule_failure_preserves_other_pods_receipt_and_heartbeat(
        monkeypatch, tmp_path, capsys, problem_str, legacy_severity_str):
    healthy_path_obj = _create_alert_db(tmp_path)
    healthy_row_dict = _capsule_summary_dict(healthy_path_obj)["pod_row_dict_list"][0]
    broken_path_obj = tmp_path / "broken.sqlite3"
    broken_path_obj.write_bytes(b"not a sqlite database")
    broken_row_dict = dict(healthy_row_dict, pod_id_str="capsule_broken", account_route_str="DU_BAD",
        db_path_str=str(broken_path_obj))
    broken_row_list = [broken_row_dict]
    if problem_str == "invalid_scope":
        broken_row_dict["db_path_str"] = None
    elif problem_str == "ambiguous_scope":
        broken_row_list.append(dict(broken_row_dict, db_path_str=str(tmp_path / "alias.sqlite3")))
    elif problem_str == "shared_account":
        broken_row_list.append(dict(broken_row_dict, pod_id_str="capsule_alias"))
    elif problem_str == "pending_sqlite":
        original_delivery_fn = alert_module.deliver_execution_alerts
        def delivery_fn(summary_dict, **kwarg_dict):
            if summary_dict["pod_row_dict_list"][0]["pod_id_str"] == "capsule_broken":
                return []
            return original_delivery_fn(summary_dict, **kwarg_dict)
        monkeypatch.setattr(alert_module, "deliver_execution_alerts", delivery_fn)
    summary_dict = _summary_dict(severity_str=legacy_severity_str)
    taa_row_dict = summary_dict["pod_row_dict_list"][0]
    ndx_row_dict = dict(taa_row_dict, pod_id_str="pod_ndx_live", account_route_str="U_NDX",
        release_id_str="ndx_release", strategy_import_str="strategies.ndx.strategy_ndx")
    summary_dict["pod_row_dict_list"].extend([ndx_row_dict, *broken_row_list, healthy_row_dict])

    return_code_int, heartbeat_list, webhook_list, output_path_obj = _run_watchdog(
        monkeypatch, tmp_path, summary_dict=summary_dict, discord_webhook_url_str="test",
        heartbeat_env_url_str=HEARTBEAT_URL_STR)

    assert return_code_int == (1 if legacy_severity_str == "red" else 0)
    assert [url_str for url_str, _ in heartbeat_list] == [
        HEARTBEAT_URL_STR + ("/fail" if legacy_severity_str == "red" else "")]
    assert _read_alert_dict(healthy_path_obj)["delivered_timestamp_str"] is not None
    failure_payload_list = [payload_dict for _, payload_dict in webhook_list
        if payload_dict["content"].startswith("Capsule watchdog notification failure.")]
    assert len(failure_payload_list) == 1
    assert failure_payload_list[0]["allowed_mentions"] == {"parse": []}
    assert len(webhook_list) == (4 if legacy_severity_str == "red" else 2)
    report_dict = json.loads(output_path_obj.read_text(encoding="utf-8"))
    receipt_dict = json.loads(output_path_obj.with_suffix(".run.json").read_text(encoding="utf-8"))
    result_dict = json.loads(capsys.readouterr().out)
    phase_str = "pending" if problem_str == "pending_sqlite" else "delivery" if problem_str == "sqlite" else "scope"
    expected_error_type_str = "DatabaseError" if "sqlite" in problem_str else "ValueError"
    for observation_dict in (report_dict, receipt_dict, result_dict):
        error_list = observation_dict["capsule_notification_error_list"]
        assert {error_dict["pod_id_str"] for error_dict in error_list} == (
            {"capsule_broken", "capsule_alias"} if problem_str == "shared_account" else {"capsule_broken"})
        assert all(error_dict["error_type_str"] == expected_error_type_str for error_dict in error_list)
        assert all(error_dict["reason_code_str"] == f"capsule_notification_{phase_str}_failed" for error_dict in error_list)
        assert observation_dict["capsule_notification_pending_count_int"] is None
        assert observation_dict["capsule_notification_pending_live_count_int"] == 0
        assert observation_dict["capsule_notification_failure_alert_status_str"] == "sent"
    assert result_dict["notification_fired_count_int"] == (2 if legacy_severity_str == "red" else 0)
    assert result_dict["run_receipt_status_str"] == "saved"
    assert receipt_dict["heartbeat_status_str"] == "sent"
    assert {row_dict["pod_id_str"] for row_dict in receipt_dict["scope_list"]}.issuperset(
        {"pod_taa_live_01", "pod_ndx_live", "capsule_one"})
    assert receipt_dict["report_sha256_str"] == hashlib.sha256(json.dumps(
        report_dict, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")).hexdigest()
    assert str(broken_path_obj) not in json.dumps(receipt_dict)


@pytest.mark.parametrize("delivery_result_str", ["failed", "raises", "disabled"])
def test_watchdog_capsule_failure_alert_transport_cannot_abort_run(monkeypatch, tmp_path, capsys, delivery_result_str):
    import scripts.live_ops_watchdog as watchdog_module
    summary_dict = _capsule_summary_dict(tmp_path / "unused.sqlite3", db_path_str=None)
    if delivery_result_str == "raises":
        def failure_delivery_fn(summary_dict, **kwarg_dict):
            def failed_post_fn(*arg_list):
                raise OSError("private webhook token")
            monkeypatch.setattr(watchdog_module.notifications_module, "post_discord_webhook_bool", failed_post_fn)
            return []
        monkeypatch.setattr(watchdog_module.notifications_module, "check_and_notify_for_red_transitions", failure_delivery_fn)
    return_code_int, heartbeat_list, webhook_list, output_path_obj = _run_watchdog(monkeypatch, tmp_path,
        summary_dict=summary_dict, heartbeat_env_url_str=HEARTBEAT_URL_STR,
        discord_webhook_url_str=None if delivery_result_str == "disabled" else "test", discord_delivery_bool=False)
    assert return_code_int == 0 and len(heartbeat_list) == 1
    assert len(webhook_list) == (1 if delivery_result_str == "failed" else 0)
    result_dict = json.loads(capsys.readouterr().out)
    receipt_dict = json.loads(output_path_obj.with_suffix(".run.json").read_text(encoding="utf-8"))
    assert result_dict["run_receipt_status_str"] == "saved"
    assert receipt_dict["capsule_notification_failure_alert_status_str"] == (
        "disabled" if delivery_result_str == "disabled" else "failed")
    assert "private webhook token" not in json.dumps(receipt_dict)


@pytest.mark.parametrize("problem_str", ["invalid_scope", "ambiguous_scope", "sqlite"])
def test_live_capsule_alert_failure_retains_dashboard_receipt_scope(monkeypatch, tmp_path, capsys, problem_str):
    from alpha.live.dashboard_v4 import system_data
    import scripts.live_ops_watchdog as watchdog_module
    completed_ts = AS_OF_TS + timedelta(seconds=17)
    monkeypatch.setattr(watchdog_module.ops_report_module, "utc_now_ts", lambda: completed_ts)
    broken_path_obj = tmp_path / "broken.sqlite3"
    broken_path_obj.write_bytes(b"not a sqlite database")
    capsule_row_dict = _capsule_summary_dict(broken_path_obj, mode_str="live",
        account_route_str="U_CAPSULE")["pod_row_dict_list"][0]
    summary_dict = _summary_dict()
    summary_dict["pod_row_dict_list"].append(capsule_row_dict)
    target_list = [SimpleNamespace(release_obj=SimpleNamespace(**{
        field_str: row_dict[field_str] for field_str in system_data.IDENTITY_FIELD_TUPLE}))
        for row_dict in summary_dict["pod_row_dict_list"]]
    if problem_str == "invalid_scope":
        capsule_row_dict["db_path_str"] = None
    elif problem_str == "ambiguous_scope":
        summary_dict["pod_row_dict_list"].append(dict(capsule_row_dict,
            db_path_str=str(tmp_path / "alias.sqlite3")))
    return_code_int, heartbeat_list, webhook_list, output_path_obj = _run_watchdog(monkeypatch, tmp_path,
        summary_dict=summary_dict, heartbeat_env_url_str=HEARTBEAT_URL_STR, discord_webhook_url_str="test")
    assert return_code_int == 0 and len(heartbeat_list) == len(webhook_list) == 1
    report_dict = json.loads(output_path_obj.read_text(encoding="utf-8"))
    receipt_dict = system_data._watchdog_run_dict(output_path_obj, target_list, report_dict, completed_ts)
    assert receipt_dict and len(receipt_dict["scope_list"]) == 2
    assert receipt_dict["capsule_notification_pending_live_count_int"] is None
    assert receipt_dict["capsule_notification_error_list"][0]["pod_id_str"] == "capsule_one"
    assert system_data._deadman_dict(receipt_dict, completed_ts)["now_str"] == "Ping sent"
    assert json.loads(capsys.readouterr().out)["run_receipt_status_str"] == "saved"


def test_late_execution_without_residual_identifies_actual_fill(tmp_path):
    payload_dict = {"residual_row_dict_list": [], "late_fill_row_dict_list": [{
        "asset_str": "BIL", "fill_amount_float": -40.0, "fill_price_float": 91.25,
        "fill_timestamp_str": "2026-10-05T10:05:00-04:00",
    }]}
    db_path_obj = _create_alert_db(tmp_path, alert_kind_str="late_execution", payload_dict=payload_dict)
    post_payload_list = []
    def post_webhook(url_str, delivered_payload_dict):
        post_payload_list.append(delivered_payload_dict)
        return True
    alert_module.deliver_execution_alerts(_capsule_summary_dict(db_path_obj), webhook_url_str="test", webhook_poster_fn=post_webhook, now_ts=NOW_TS)
    content_str = post_payload_list[0]["content"]
    assert all(value_str in content_str for value_str in ["Late execution", "LATE BIL", "-40.0", "91.25", "2026-10-05T10:05:00-04:00"])


def test_distinct_cycles_for_same_pod_are_both_delivered_once(tmp_path):
    db_path_obj = _create_alert_db(tmp_path)
    with closing(sqlite3.connect(db_path_obj)) as connection_obj, connection_obj:
        _enqueue(connection_obj, vplan_id_int=2)
    summary_dict = _capsule_summary_dict(db_path_obj)
    sent_content_list = []
    def post_webhook(url_str, payload_dict):
        sent_content_list.append(payload_dict["content"])
        return True
    delivery_list = alert_module.deliver_execution_alerts(summary_dict, webhook_url_str="test", webhook_poster_fn=post_webhook, now_ts=NOW_TS)
    assert {delivery_obj.vplan_id_int for delivery_obj in delivery_list} == {1, 2}
    assert len(sent_content_list) == 2
    assert alert_module.deliver_execution_alerts(summary_dict, webhook_url_str="test", webhook_poster_fn=post_webhook, now_ts=NOW_TS) == []
    assert len(sent_content_list) == 2


def test_watchdog_live_pending_count_tracks_durable_retry_then_success(monkeypatch, tmp_path, capsys):
    # Historical LIVE alerts can remain in a DB although new capsule LIVE
    # releases are locked. Monitoring must still deliver those saved events.
    db_path_obj = _create_alert_db(tmp_path, account_route_str="U_ONE", mode_str="live")
    summary_dict = _capsule_summary_dict(db_path_obj, account_route_str="U_ONE", mode_str="live")
    for delivered_bool, expected_pending_int in [(False, 1), (True, 0)]:
        return_code_int, _, webhook_list, output_path_obj = _run_watchdog(monkeypatch, tmp_path,
            summary_dict=summary_dict, discord_webhook_url_str="test", discord_delivery_bool=delivered_bool,
            extra_argv_list=["--mode", "live"])
        assert return_code_int == 0 and len(webhook_list) == 1
        result_dict = json.loads(capsys.readouterr().out)
        receipt_dict = json.loads(output_path_obj.with_suffix(".run.json").read_text(encoding="utf-8"))
        for result_obj in [result_dict, receipt_dict]:
            assert result_obj["capsule_notification_pending_live_count_int"] == expected_pending_int
            assert result_obj["capsule_notification_pending_count_int"] == expected_pending_int
            assert result_obj["capsule_notification_attempt_count_int"] == 1
        assert result_dict["notification_fired_count_int"] == 0


def _seed_cycle_context(db_path_obj, *, completed_bool=False, result_dict=None, mismatch_field_str=None):
    with closing(sqlite3.connect(db_path_obj)) as connection_obj, connection_obj:
        connection_obj.execute("CREATE TABLE live_release (release_id_str TEXT, pod_id_str TEXT, account_route_str TEXT, mode_str TEXT, strategy_import_str TEXT)")
        connection_obj.execute("CREATE TABLE decision_plan (decision_plan_id_int INTEGER, release_id_str TEXT, pod_id_str TEXT, account_route_str TEXT, status_str TEXT, snapshot_metadata_json_str TEXT)")
        connection_obj.execute("CREATE TABLE vplan (vplan_id_int INTEGER, decision_plan_id_int INTEGER, release_id_str TEXT, pod_id_str TEXT, account_route_str TEXT, status_str TEXT)")
        connection_obj.execute("INSERT INTO live_release VALUES ('release', 'capsule_one', 'DU_ONE', 'paper', ?)", (STRATEGY_IMPORT_STR,))
        status_str = "completed" if completed_bool else "submitted"
        connection_obj.execute("INSERT INTO decision_plan VALUES (1, 'release', 'capsule_one', 'DU_ONE', ?, ?)",
            (status_str, json.dumps({"mr_capsule_execution_result_dict": result_dict or _alert_payload_dict()})))
        connection_obj.execute("INSERT INTO vplan VALUES (1, 1, 'release', 'capsule_one', 'DU_ONE', ?)", (status_str,))
        if mismatch_field_str == "release_mode":
            connection_obj.execute("UPDATE live_release SET mode_str='live'")
        elif mismatch_field_str == "decision_account":
            connection_obj.execute("UPDATE decision_plan SET account_route_str='DU_OTHER'")
        elif mismatch_field_str == "vplan_pod":
            connection_obj.execute("UPDATE vplan SET pod_id_str='other_pod'")
        elif mismatch_field_str == "release_strategy":
            connection_obj.execute("UPDATE live_release SET strategy_import_str='strategies.taa_df.strategy_taa_df'")


def _capture_delivery(db_path_obj):
    payload_list = []
    def post_webhook(url_str, payload_dict):
        payload_list.append(payload_dict)
        return True
    alert_module.deliver_execution_alerts(_capsule_summary_dict(db_path_obj), webhook_url_str="test", webhook_poster_fn=post_webhook, now_ts=NOW_TS)
    return payload_list


def test_old_unresolved_alert_is_historical_after_cycle_completion(tmp_path):
    db_path_obj = _create_alert_db(tmp_path, alert_kind_str="unresolved_execution")
    original_payload_str = _read_alert_dict(db_path_obj)["payload_json_str"]
    _seed_cycle_context(db_path_obj, completed_bool=True, result_dict={"outcome_str": "completed", "residual_row_dict_list": []})
    content_str = _capture_delivery(db_path_obj)[0]["content"]
    assert "HISTORICAL" in content_str and "now recorded completed" in content_str and "No action" in content_str
    assert "MSFT" in content_str and "action=SELL" not in content_str and "remaining=40" not in content_str
    assert _read_alert_dict(db_path_obj)["payload_json_str"] == original_payload_str


def test_unresolved_alert_uses_latest_recorded_remainder_without_rewriting_audit(tmp_path):
    db_path_obj = _create_alert_db(tmp_path, alert_kind_str="unresolved_execution")
    original_payload_str = _read_alert_dict(db_path_obj)["payload_json_str"]
    latest_result_dict = _alert_payload_dict()
    latest_result_dict["residual_row_dict_list"][0].update(residual_amount_float=-10.0, filled_amount_float=-90.0)
    _seed_cycle_context(db_path_obj, result_dict=latest_result_dict)
    content_str = _capture_delivery(db_path_obj)[0]["content"]
    assert "Latest recorded details" in content_str and "remaining=10 shares" in content_str
    assert "remaining=40 shares" not in content_str and "action=SELL" in content_str
    assert "Stored state checked at" in content_str
    assert _read_alert_dict(db_path_obj)["payload_json_str"] == original_payload_str


@pytest.mark.parametrize("mismatch_field_str", ["release_mode", "decision_account", "vplan_pod", "release_strategy"])
def test_current_alert_context_requires_exact_cycle_identity(tmp_path, mismatch_field_str):
    db_path_obj = _create_alert_db(tmp_path, alert_kind_str="unresolved_execution")
    _seed_cycle_context(db_path_obj, completed_bool=True, mismatch_field_str=mismatch_field_str)
    content_str = _capture_delivery(db_path_obj)[0]["content"]
    assert "current cycle state unavailable" in content_str
    assert "action=VERIFY" in content_str and "action=SELL" not in content_str
    assert "now recorded completed" not in content_str


def test_alert_without_cycle_tables_is_historical_and_unknown_quantity_is_explicit(tmp_path):
    payload_dict = _alert_payload_dict()
    payload_dict["residual_row_dict_list"][0].update(residual_amount_float=None, required_action_str="VERIFY")
    db_path_obj = _create_alert_db(tmp_path, alert_kind_str="unresolved_execution", payload_dict=payload_dict)
    content_str = _capture_delivery(db_path_obj)[0]["content"]
    assert "HISTORICAL" in content_str and "remaining=unknown shares" in content_str and "action=VERIFY" in content_str


def test_delayed_accepted_sale_is_explicitly_historical_not_a_current_instruction(tmp_path):
    db_path_obj = _create_alert_db(tmp_path)
    summary_dict = _capsule_summary_dict(db_path_obj)
    alert_module.deliver_execution_alerts(summary_dict, webhook_url_str="test", webhook_poster_fn=lambda *_: False, now_ts=NOW_TS)
    summary_dict["pod_row_dict_list"][0].update(latest_vplan_id_int=999, latest_vplan_status_str="completed")
    payload_list = []
    def post_webhook(url_str, payload_dict):
        payload_list.append(payload_dict)
        return True
    alert_module.deliver_execution_alerts(summary_dict, webhook_url_str="test", webhook_poster_fn=post_webhook, now_ts=NOW_TS)
    content_str = payload_list[0]["content"]
    assert "HISTORICAL" in content_str and "VERIFY current broker holdings" in content_str
    assert "action at observation=SELL" in content_str and "recorded remaining=40 shares" in content_str
    assert "action=SELL" not in content_str


def test_long_alert_delivers_every_symbol_quantity_and_action_in_whole_rows(tmp_path):
    payload_dict = _alert_payload_dict()
    template_row_dict = payload_dict["residual_row_dict_list"][0]
    payload_dict["residual_row_dict_list"] = [dict(template_row_dict, asset_str=f"SYM{index_int:02}",
        residual_amount_float=-float(index_int + 1)) for index_int in range(30)]
    db_path_obj = _create_alert_db(tmp_path, payload_dict=payload_dict)
    message_list = _capture_delivery(db_path_obj)
    assert len(message_list) > 1
    for part_int, message_dict in enumerate(message_list, start=1):
        assert len(message_dict["content"]) <= 1900
        assert f"part {part_int}/{len(message_list)}" in message_dict["content"]
        assert "Account DU_ONE | VPlan 1 | accepted_residual" in message_dict["content"]
    for index_int in range(30):
        matching_line_list = [line_str for message_dict in message_list for line_str in message_dict["content"].splitlines()
            if line_str.startswith(f"SYM{index_int:02}:")]
        assert len(matching_line_list) == 1
        assert f"recorded remaining={index_int + 1} shares" in matching_line_list[0]
        assert "action at observation=SELL" in matching_line_list[0]
    assert _read_alert_dict(db_path_obj)["delivered_timestamp_str"] is not None


def test_partial_multimessage_failure_remains_pending_and_retries_whole_event(tmp_path):
    payload_dict = _alert_payload_dict()
    payload_dict["residual_row_dict_list"] = [dict(payload_dict["residual_row_dict_list"][0], asset_str=f"SYM{index_int}") for index_int in range(30)]
    db_path_obj = _create_alert_db(tmp_path, payload_dict=payload_dict)
    failed_attempt_list = []
    def fail_second_part(url_str, delivered_payload_dict):
        failed_attempt_list.append(delivered_payload_dict)
        return len(failed_attempt_list) == 1
    result_list = alert_module.deliver_execution_alerts(_capsule_summary_dict(db_path_obj), webhook_url_str="test", webhook_poster_fn=fail_second_part, now_ts=NOW_TS)
    assert len(failed_attempt_list) == 2 and not result_list[0].delivered_bool
    assert _read_alert_dict(db_path_obj)["delivered_timestamp_str"] is None
    assert alert_module.pending_execution_alert_count_int(_capsule_summary_dict(db_path_obj)) == 1
    retry_message_list = _capture_delivery(db_path_obj)
    assert len(retry_message_list) > 2
    assert retry_message_list[0] == failed_attempt_list[0]
    assert _read_alert_dict(db_path_obj)["delivered_timestamp_str"] is not None
    assert _read_alert_dict(db_path_obj)["attempt_count_int"] == 2
