"""Removed manual resume and daily exception delivery; offline fixtures only."""
from contextlib import closing
from datetime import UTC, datetime
import json
import sqlite3

import pytest

from alpha.live import runner
from alpha.live.core5_adapter import CORE5_STRATEGY_IMPORT_STR
from alpha.live import mr_capsule_notifications as alert_module
from test_live_mr_capsule_notifications import _capsule_summary_dict
from test_live_ops_watchdog import _run_watchdog, HEARTBEAT_URL_STR


def test_resume_command_is_removed_before_any_broker_or_state_action(monkeypatch, capsys):
    monkeypatch.setattr(runner, "LiveStateStore", lambda *_arg_tuple, **_kwarg_dict: pytest.fail("Removed command opened state"))
    with pytest.raises(SystemExit) as error_obj:
        runner.main(["resume_core5", "--mode", "paper"])
    assert error_obj.value.code == 2
    assert "invalid choice" in capsys.readouterr().err


def test_core5_daily_exception_is_delivered_by_watchdog_without_losing_heartbeat(monkeypatch, tmp_path, capsys):
    database_path = tmp_path / "daily.sqlite3"
    with closing(sqlite3.connect(database_path)) as connection_obj, connection_obj:
        alert_module.ensure_execution_alert_schema(connection_obj)
        alert_module.enqueue_daily_exception_alert(connection_obj, decision_plan_id_int=12,
            pod_id_str="capsule_one", account_route_str="DU_ONE", mode_str="paper",
            exception_list=[{"asset_str": "DBC", "side_str": "SELL", "quantity_float": 7.0,
                "reason_str": "Missed opening cutoff", "expected_share_float": 0.0, "actual_share_float": 7.0}],
            created_timestamp_ts=datetime(2026, 10, 7, 20, 1, tzinfo=UTC))
    summary_dict = _capsule_summary_dict(database_path, strategy_import_str=CORE5_STRATEGY_IMPORT_STR)
    result_int, heartbeat_list, webhook_list, report_path = _run_watchdog(monkeypatch, tmp_path,
        summary_dict=summary_dict, discord_webhook_url_str="test", heartbeat_env_url_str=HEARTBEAT_URL_STR)
    assert result_int == 0
    assert heartbeat_list
    daily_payload_list = [payload_dict for _, payload_dict in webhook_list if "DAILY cycle exceptions" in payload_dict["content"]]
    assert len(daily_payload_list) == 1
    assert "DBC: side=SELL; quantity=7 shares; reason=Missed opening cutoff" in daily_payload_list[0]["content"]
    assert "Decision 12 | VPlan not built" in daily_payload_list[0]["content"]
    assert report_path.with_suffix(".run.json").is_file()
    assert json.loads(capsys.readouterr().out)["capsule_notification_attempt_count_int"] == 1
    assert alert_module.deliver_execution_alerts(summary_dict, webhook_url_str="test",
        webhook_poster_fn=lambda *_: pytest.fail("Already delivered")) == []
