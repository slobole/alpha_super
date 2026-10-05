"""Operator command and critical-alert plumbing; fake broker and webhook only."""
import json
from types import SimpleNamespace

from alpha.live import runner
from alpha.live.core5_adapter import CORE5_STRATEGY_IMPORT_STR
from alpha.live import mr_capsule_notifications as alert_module
from test_live_core5_resume_command import context_obj
from test_live_mr_capsule_notifications import _create_alert_db, _capsule_summary_dict
from test_live_ops_watchdog import _run_watchdog, HEARTBEAT_URL_STR


def test_resume_command_preview_then_apply_preserves_review_hash(context_obj, monkeypatch):
    monkeypatch.setattr(runner, "load_release_list", lambda *_: [context_obj.release_obj])
    monkeypatch.setattr(runner, "_coerce_broker_adapter_resolver_obj", lambda **_: SimpleNamespace(
        get_adapter=lambda _: context_obj.broker_obj))
    argument_obj = SimpleNamespace(command_name_str="resume_core5", pod_id_str=context_obj.release_obj.pod_id_str,
        env_mode_str="paper", releases_root_path_str="fake", broker_host_str=None, broker_port_int=None,
        broker_client_id_int=None, broker_timeout_seconds_float=None, review_hash_str=None,
        reason_str="Reviewed offline fixture", operator_str="test_operator")
    preview_dict = runner._execute_runner_command_detail_dict(argument_obj, context_obj.store_obj,
        context_obj.as_of_ts, context_obj.store_obj.db_path_str)
    argument_obj.review_hash_str = preview_dict["review_hash_str"]
    applied_dict = runner._execute_runner_command_detail_dict(argument_obj, context_obj.store_obj,
        context_obj.as_of_ts, context_obj.store_obj.db_path_str)
    assert applied_dict["applied_bool"]
    assert applied_dict["review_hash_str"] == preview_dict["review_hash_str"]
    assert context_obj.store_obj.get_pod_state(context_obj.release_obj.pod_id_str) == context_obj.state_obj


def test_core5_critical_outbox_is_delivered_by_watchdog_without_losing_heartbeat(monkeypatch, tmp_path, capsys):
    database_path = _create_alert_db(tmp_path, alert_kind_str="dispatch_failed",
        payload_dict={"severity_str": "critical", "reason_code_str": "opening_dispatch_parked"})
    summary_dict = _capsule_summary_dict(database_path, strategy_import_str=CORE5_STRATEGY_IMPORT_STR)
    result_int, heartbeat_list, webhook_list, report_path = _run_watchdog(monkeypatch, tmp_path,
        summary_dict=summary_dict, discord_webhook_url_str="test", heartbeat_env_url_str=HEARTBEAT_URL_STR)
    assert result_int == 0
    assert heartbeat_list
    assert any("CRITICAL: opening dispatch" in payload_dict["content"] for _, payload_dict in webhook_list)
    assert report_path.with_suffix(".run.json").is_file()
    assert json.loads(capsys.readouterr().out)["capsule_notification_attempt_count_int"] == 1
    assert alert_module.deliver_execution_alerts(summary_dict, webhook_url_str="test",
        webhook_poster_fn=lambda *_: (_ for _ in ()).throw(AssertionError("Already delivered"))) == []
