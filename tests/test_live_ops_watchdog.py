from __future__ import annotations

from copy import deepcopy
from datetime import UTC, datetime, timedelta
import hashlib
import json
from pathlib import Path

import pytest

import alpha.live.dashboard as dashboard_module
import alpha.live.dashboard_v3.notifications as notifications_module
import alpha.live.ops_report as ops_report_module
import scripts.live_ops_watchdog as watchdog_module


AS_OF_TS = datetime(2026, 6, 9, 12, 0, tzinfo=UTC)
HEARTBEAT_URL_STR = "https://hc-ping.example/abc123"


def _summary_dict(*, severity_str: str = "green") -> dict[str, object]:
    return {
        "as_of_timestamp_str": AS_OF_TS.isoformat(),
        "pod_row_dict_list": [
            {
                "pod_id_str": "pod_taa_live_01",
                "user_id_str": "owner_one",
                "release_id_str": "release_one",
                "mode_str": "live",
                "account_route_str": "U1",
                "strategy_import_str": "strategies.taa_df.strategy_taa_df",
                "db_status_str": "ok",
                "health_str": severity_str,
                "next_action_str": "status",
                "required_action_dict": {
                    "label_str": "No action",
                    "severity_str": severity_str,
                    "reason_str": "POD is idle or completed.",
                    "inspect_command_name_str": "status",
                },
                "debug_summary_dict": {
                    "severity_str": severity_str,
                    "verdict_label_str": "healthy",
                    "primary_reason_str": "POD is healthy.",
                },
            }
        ],
    }


class FakeDashboardApp:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def _run_watchdog(
    monkeypatch,
    tmp_path: Path,
    *,
    summary_dict: dict[str, object] | None = None,
    summary_builder_fn=None,
    extra_argv_list: list[str] | None = None,
    heartbeat_env_url_str: str | None = None,
    discord_webhook_url_str: str | None = None,
    discord_delivery_bool: bool = True,
    heartbeat_delivery_bool: bool = True,
    step_list: list[str] | None = None,
) -> tuple[int, list[tuple[str, dict[str, object]]], list[tuple[str, dict[str, object]]], Path]:
    # Without this stub the real config.env would clobber test env vars via
    # override_existing_bool=True.
    monkeypatch.setattr(watchdog_module, "load_config_env_file", lambda **kwargs: {})
    monkeypatch.delenv("ALPHA_DISCORD_WEBHOOK_URL", raising=False)
    monkeypatch.delenv(watchdog_module.HEARTBEAT_URL_ENV_VAR_NAME_STR, raising=False)
    if heartbeat_env_url_str is not None:
        monkeypatch.setenv(watchdog_module.HEARTBEAT_URL_ENV_VAR_NAME_STR, heartbeat_env_url_str)
    if discord_webhook_url_str is not None:
        monkeypatch.setenv("ALPHA_DISCORD_WEBHOOK_URL", discord_webhook_url_str)

    monkeypatch.setattr(dashboard_module, "DashboardApp", FakeDashboardApp)
    if summary_builder_fn is None:
        def summary_builder_fn(app_obj, as_of_ts=None):
            assert isinstance(app_obj, FakeDashboardApp)
            assert as_of_ts == AS_OF_TS
            return summary_dict if summary_dict is not None else _summary_dict()

    monkeypatch.setattr(dashboard_module, "build_dashboard_summary_dict", summary_builder_fn)

    heartbeat_call_list: list[tuple[str, dict[str, object]]] = []

    def fake_post_heartbeat_bool(url_str, payload_dict, *, timeout_seconds_float=3.0):
        if step_list is not None:
            step_list.append("heartbeat")
        heartbeat_call_list.append((url_str, payload_dict))
        return heartbeat_delivery_bool

    monkeypatch.setattr(ops_report_module, "post_heartbeat_bool", fake_post_heartbeat_bool)

    webhook_call_list: list[tuple[str, dict[str, object]]] = []

    def fake_post_discord_webhook_bool(url_str, payload_dict):
        if step_list is not None:
            step_list.append("discord")
        webhook_call_list.append((url_str, payload_dict))
        return discord_delivery_bool

    monkeypatch.setattr(
        notifications_module,
        "post_discord_webhook_bool",
        fake_post_discord_webhook_bool,
    )

    output_path_obj = tmp_path / "ops_report_latest.json"
    state_path_obj = tmp_path / "watchdog_notification_state.json"
    argv_list = [
        "--json",
        "--vps-id",
        "vps_01",
        "--as-of-ts",
        AS_OF_TS.isoformat(),
        "--output-path",
        str(output_path_obj),
        "--notification-state-path",
        str(state_path_obj),
    ]
    argv_list.extend(extra_argv_list or [])
    return_code_int = watchdog_module.main(argv_list)
    return return_code_int, heartbeat_call_list, webhook_call_list, output_path_obj


def test_watchdog_green_writes_report_pings_plain_url_and_exits_zero(
    monkeypatch, tmp_path, capsys
) -> None:
    return_code_int, heartbeat_call_list, _, output_path_obj = _run_watchdog(
        monkeypatch,
        tmp_path,
        heartbeat_env_url_str=HEARTBEAT_URL_STR,
    )

    assert return_code_int == 0
    assert len(heartbeat_call_list) == 1
    assert heartbeat_call_list[0][0] == HEARTBEAT_URL_STR
    report_dict = json.loads(output_path_obj.read_text(encoding="utf-8"))
    assert report_dict["schema_version_str"] == "live_ops_inspector.v1"
    assert report_dict["vps_id_str"] == "vps_01"
    assert report_dict["overall_severity_str"] == "green"
    result_dict = json.loads(capsys.readouterr().out)
    assert result_dict["heartbeat_status_str"] == "sent"
    assert result_dict["heartbeat_fail_signal_bool"] is False


def test_watchdog_red_pings_fail_url_and_exits_one(monkeypatch, tmp_path, capsys) -> None:
    return_code_int, heartbeat_call_list, _, _ = _run_watchdog(
        monkeypatch,
        tmp_path,
        summary_dict=_summary_dict(severity_str="red"),
        heartbeat_env_url_str=HEARTBEAT_URL_STR,
    )

    assert return_code_int == 1
    assert len(heartbeat_call_list) == 1
    assert heartbeat_call_list[0][0] == HEARTBEAT_URL_STR + "/fail"
    result_dict = json.loads(capsys.readouterr().out)
    assert result_dict["status_str"] == "red"
    assert result_dict["heartbeat_fail_signal_bool"] is True


def test_watchdog_no_pods_is_gray_and_pings_plain_success(
    monkeypatch, tmp_path, capsys
) -> None:
    return_code_int, heartbeat_call_list, _, output_path_obj = _run_watchdog(
        monkeypatch,
        tmp_path,
        summary_dict={
            "as_of_timestamp_str": AS_OF_TS.isoformat(),
            "pod_row_dict_list": [],
        },
        heartbeat_env_url_str=HEARTBEAT_URL_STR,
    )

    assert return_code_int == 0
    assert heartbeat_call_list[0][0] == HEARTBEAT_URL_STR
    report_dict = json.loads(output_path_obj.read_text(encoding="utf-8"))
    assert report_dict["overall_severity_str"] == "gray"
    result_dict = json.loads(capsys.readouterr().out)
    assert result_dict["status_str"] == "ok"


def test_watchdog_fatal_summary_error_exits_two_and_skips_heartbeat(
    monkeypatch, tmp_path, capsys
) -> None:
    def failing_summary_builder_fn(app_obj, as_of_ts=None):
        raise RuntimeError("summary build exploded")

    return_code_int, heartbeat_call_list, _, output_path_obj = _run_watchdog(
        monkeypatch,
        tmp_path,
        summary_builder_fn=failing_summary_builder_fn,
        heartbeat_env_url_str=HEARTBEAT_URL_STR,
    )

    assert return_code_int == watchdog_module.FATAL_EXIT_CODE_INT
    assert heartbeat_call_list == []
    assert not output_path_obj.exists()
    result_dict = json.loads(capsys.readouterr().out)
    assert result_dict["status_str"] == "error"
    assert result_dict["reason_code_str"] == "watchdog_fatal_error"
    assert "summary build exploded" in result_dict["error_str"]


def test_watchdog_report_write_is_atomic_and_overwrites_previous(
    monkeypatch, tmp_path
) -> None:
    output_path_obj = tmp_path / "ops_report_latest.json"
    output_path_obj.write_text('{"stale": true}', encoding="utf-8")

    return_code_int, _, _, _ = _run_watchdog(monkeypatch, tmp_path)

    assert return_code_int == 0
    report_dict = json.loads(output_path_obj.read_text(encoding="utf-8"))
    assert report_dict["schema_version_str"] == "live_ops_inspector.v1"
    assert "stale" not in report_dict
    assert list(tmp_path.glob("*.tmp")) == []


def test_watchdog_red_transition_fires_webhook_once_across_runs(
    monkeypatch, tmp_path
) -> None:
    state_path_obj = tmp_path / "watchdog_notification_state.json"
    red_summary_dict = _summary_dict(severity_str="red")

    webhook_total_call_list: list[tuple[str, dict[str, object]]] = []
    for _ in range(2):
        return_code_int, _, webhook_call_list, _ = _run_watchdog(
            monkeypatch,
            tmp_path,
            summary_dict=red_summary_dict,
            discord_webhook_url_str="https://discord.example/webhook",
        )
        assert return_code_int == 1
        webhook_total_call_list.extend(webhook_call_list)

    assert len(webhook_total_call_list) == 1
    state_dict = json.loads(state_path_obj.read_text(encoding="utf-8"))
    assert state_dict["pod_severity_map_dict"]["pod_taa_live_01"] == "red"


def test_watchdog_retries_failed_discord_without_changing_red_heartbeat(
    monkeypatch, tmp_path, capsys
) -> None:
    webhook_total_call_list = []
    for delivery_bool, expected_attempt_count_int in [(False, 1), (True, 1), (True, 0)]:
        return_code_int, heartbeat_call_list, webhook_call_list, output_path_obj = _run_watchdog(
            monkeypatch, tmp_path, summary_dict=_summary_dict(severity_str="red"),
            discord_webhook_url_str="https://discord.example/webhook",
            discord_delivery_bool=delivery_bool, heartbeat_env_url_str=HEARTBEAT_URL_STR,
        )
        assert return_code_int == 1
        assert len(webhook_call_list) == expected_attempt_count_int
        assert [url_str for url_str, payload_dict in heartbeat_call_list] == [HEARTBEAT_URL_STR + "/fail"]
        assert json.loads(output_path_obj.read_text(encoding="utf-8"))["overall_severity_str"] == "red"
        result_dict = json.loads(capsys.readouterr().out)
        assert result_dict["notification_fired_count_int"] == expected_attempt_count_int
        assert result_dict["heartbeat_status_str"] == "sent"
        webhook_total_call_list.extend(webhook_call_list)
    assert len(webhook_total_call_list) == 2


def test_watchdog_yellow_norgate_waiting_does_not_fire_discord(
    monkeypatch,
    tmp_path,
) -> None:
    yellow_summary_dict = _summary_dict(severity_str="yellow")
    yellow_row_dict = yellow_summary_dict["pod_row_dict_list"][0]
    yellow_row_dict["required_action_dict"] = {
        "label_str": "Wait Norgate data",
        "severity_str": "yellow",
        "reason_str": (
            "Waiting: Local Norgate data is too old for the next DecisionPlan. "
            "This is still inside the normal Norgate publish window."
        ),
        "inspect_command_name_str": "status",
    }

    return_code_int, _, webhook_call_list, _ = _run_watchdog(
        monkeypatch,
        tmp_path,
        summary_dict=yellow_summary_dict,
        discord_webhook_url_str="https://discord.example/webhook",
    )

    assert return_code_int == 0
    assert webhook_call_list == []


def test_watchdog_norgate_red_recovery_allows_later_red_alert(
    monkeypatch,
    tmp_path,
) -> None:
    stale_reason_str = (
        "Blocked: local Norgate data is too old for the next DecisionPlan. "
        "Required data date: 2026-06-30. Local data date: 2026-06-18."
    )
    red_summary_dict = _summary_dict(severity_str="red")
    red_row_dict = red_summary_dict["pod_row_dict_list"][0]
    red_row_dict["required_action_dict"] = {
        "label_str": "Review Norgate data",
        "severity_str": "red",
        "reason_str": stale_reason_str,
        "inspect_command_name_str": "status",
    }
    red_row_dict["debug_summary_dict"] = {
        "severity_str": "red",
        "verdict_label_str": "Norgate freshness",
        "primary_reason_str": stale_reason_str,
    }

    webhook_total_call_list: list[tuple[str, dict[str, object]]] = []
    for summary_dict in [red_summary_dict, _summary_dict(severity_str="green"), red_summary_dict]:
        return_code_int, _, webhook_call_list, _ = _run_watchdog(
            monkeypatch,
            tmp_path,
            summary_dict=summary_dict,
            discord_webhook_url_str="https://discord.example/webhook",
        )
        webhook_total_call_list.extend(webhook_call_list)
        assert return_code_int in {0, 1}

    assert len(webhook_total_call_list) == 2
    assert stale_reason_str in str(webhook_total_call_list[0][1]["content"])
    assert stale_reason_str in str(webhook_total_call_list[1][1]["content"])


def test_watchdog_missing_heartbeat_url_is_disabled_but_report_still_written(
    monkeypatch, tmp_path, capsys
) -> None:
    return_code_int, heartbeat_call_list, _, output_path_obj = _run_watchdog(
        monkeypatch, tmp_path
    )

    assert return_code_int == 0
    assert heartbeat_call_list == []
    assert output_path_obj.exists()
    result_dict = json.loads(capsys.readouterr().out)
    assert result_dict["heartbeat_status_str"] == "disabled"


def test_watchdog_heartbeat_url_flag_overrides_env(monkeypatch, tmp_path) -> None:
    flag_url_str = "https://flag.example/y"
    return_code_int, heartbeat_call_list, _, _ = _run_watchdog(
        monkeypatch,
        tmp_path,
        heartbeat_env_url_str="https://env.example/x",
        extra_argv_list=["--heartbeat-url", flag_url_str],
    )

    assert return_code_int == 0
    assert heartbeat_call_list[0][0] == flag_url_str


def test_run_receipt_is_saved_after_report_alert_state_and_heartbeat(monkeypatch, tmp_path, capsys):
    step_list = []
    original_report_fn = watchdog_module.write_report_atomic
    original_state_fn = notifications_module.NotificationStateStore.save_state
    original_receipt_fn = watchdog_module._write_run_receipt_atomic
    completion_ts = AS_OF_TS + timedelta(seconds=17)
    def report_fn(report_dict, output_path_str):
        original_report_fn(report_dict, output_path_str)
        step_list.append("report")
    def state_fn(store_obj, state_obj):
        original_state_fn(store_obj, state_obj)
        step_list.append("state")
    def receipt_fn(receipt_dict, output_path_str):
        assert step_list == ["report", "discord", "state", "heartbeat"]
        original_receipt_fn(receipt_dict, output_path_str)
        step_list.append("receipt")
    monkeypatch.setattr(watchdog_module, "write_report_atomic", report_fn)
    monkeypatch.setattr(notifications_module.NotificationStateStore, "save_state", state_fn)
    monkeypatch.setattr(watchdog_module, "_write_run_receipt_atomic", receipt_fn)
    monkeypatch.setattr(ops_report_module, "utc_now_ts", lambda: completion_ts)
    return_code_int, _heartbeat_list, _webhook_list, output_path_obj = _run_watchdog(
        monkeypatch, tmp_path, summary_dict=_summary_dict(severity_str="red"), step_list=step_list,
        heartbeat_env_url_str=HEARTBEAT_URL_STR, discord_webhook_url_str="https://discord.example/private-webhook")
    assert return_code_int == 1
    assert step_list == ["report", "discord", "state", "heartbeat", "receipt"]
    receipt_dict = json.loads(output_path_obj.with_suffix(".run.json").read_text(encoding="utf-8"))
    report_dict = json.loads(output_path_obj.read_text(encoding="utf-8"))
    assert receipt_dict == {"schema_version_str": "live_ops_watchdog_run.v1",
        "completed_at_utc_str": completion_ts.isoformat(), "report_generated_at_utc_str": AS_OF_TS.isoformat(),
        "report_sha256_str": hashlib.sha256(json.dumps(report_dict, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")).hexdigest(),
        "mode_str": "all", "scope_list": [{"mode_str": "live", "user_id_str": "owner_one",
            "pod_id_str": "pod_taa_live_01", "account_route_str": "U1", "release_id_str": "release_one"}],
        "heartbeat_status_str": "sent", "heartbeat_fail_signal_bool": True,
        "notification_configured_bool": True, "notification_pending_live_count_int": 0}
    assert json.loads(capsys.readouterr().out)["run_receipt_status_str"] == "saved"
    assert list(tmp_path.glob("*.tmp")) == []


@pytest.mark.parametrize("status_str,severity_str,url_str,delivered_bool,exit_int,fail_bool", [
    ("sent", "green", HEARTBEAT_URL_STR, True, 0, False),
    ("sent", "red", HEARTBEAT_URL_STR, True, 1, True),
    ("failed", "green", HEARTBEAT_URL_STR, False, 0, False),
    ("failed", "red", HEARTBEAT_URL_STR, False, 1, True),
    ("disabled", "green", None, True, 0, False),
    ("disabled", "red", None, True, 1, False),
])
def test_run_receipt_heartbeat_results_preserve_existing_exit_contract(monkeypatch, tmp_path, capsys,
        status_str, severity_str, url_str, delivered_bool, exit_int, fail_bool):
    return_code_int, heartbeat_list, _webhook_list, output_path_obj = _run_watchdog(monkeypatch, tmp_path,
        summary_dict=_summary_dict(severity_str=severity_str), heartbeat_env_url_str=url_str,
        heartbeat_delivery_bool=delivered_bool)
    receipt_dict = json.loads(output_path_obj.with_suffix(".run.json").read_text(encoding="utf-8"))
    result_dict = json.loads(capsys.readouterr().out)
    assert return_code_int == exit_int
    assert receipt_dict["heartbeat_status_str"] == result_dict["heartbeat_status_str"] == status_str
    assert receipt_dict["heartbeat_fail_signal_bool"] is result_dict["heartbeat_fail_signal_bool"] is fail_bool
    assert len(heartbeat_list) == (0 if url_str is None else 1)


def test_run_receipt_failed_live_count_tracks_saved_pending_state_excludes_other_modes(monkeypatch, tmp_path, capsys):
    summary_dict = _summary_dict(severity_str="red")
    paper_dict = {**deepcopy(summary_dict["pod_row_dict_list"][0]), "pod_id_str": "paper_one",
        "mode_str": "paper", "account_route_str": "DU1", "release_id_str": "release_paper"}
    summary_dict["pod_row_dict_list"].append(paper_dict)
    for delivery_bool, pending_int in [(False, 1), (True, 0), (True, 0)]:
        _run_watchdog(monkeypatch, tmp_path, summary_dict=summary_dict, discord_delivery_bool=delivery_bool,
            discord_webhook_url_str="https://discord.example/private-webhook", extra_argv_list=["--mode", "live"])
        receipt_dict = json.loads((tmp_path / "ops_report_latest.run.json").read_text(encoding="utf-8"))
        state_dict = json.loads((tmp_path / "watchdog_notification_state.json").read_text(encoding="utf-8"))
        assert receipt_dict["notification_pending_live_count_int"] == pending_int
        assert len([pod_str for pod_str in state_dict["pending_red_previous_severity_map_dict"] if pod_str == "pod_taa_live_01"]) == pending_int
        assert {row_dict["mode_str"] for row_dict in receipt_dict["scope_list"]} == {"live"}
        assert json.loads(capsys.readouterr().out)["run_receipt_status_str"] == "saved"


def test_run_receipt_missing_webhook_keeps_pending_unknown_not_zero(monkeypatch, tmp_path, capsys):
    summary_dict = _summary_dict(severity_str="red")
    _run_watchdog(monkeypatch, tmp_path, summary_dict=summary_dict, discord_delivery_bool=False,
        discord_webhook_url_str="https://discord.example/private-webhook")
    capsys.readouterr()
    _run_watchdog(monkeypatch, tmp_path, summary_dict=summary_dict)
    receipt_dict = json.loads((tmp_path / "ops_report_latest.run.json").read_text(encoding="utf-8"))
    state_dict = json.loads((tmp_path / "watchdog_notification_state.json").read_text(encoding="utf-8"))
    assert receipt_dict["notification_configured_bool"] is False
    assert receipt_dict["notification_pending_live_count_int"] is None
    assert "pod_taa_live_01" in state_dict["pending_red_previous_severity_map_dict"]


@pytest.mark.parametrize("phase_str", ["summary", "report", "notification"])
def test_run_receipt_fatal_earlier_phase_keeps_prior_receipt_without_heartbeat(monkeypatch, tmp_path, capsys, phase_str):
    receipt_path_obj = tmp_path / "ops_report_latest.run.json"
    receipt_path_obj.write_text('{"prior_receipt":true}', encoding="utf-8")
    def failure_fn(*argument_list, **argument_dict):
        raise RuntimeError("Private failure")
    if phase_str == "report":
        monkeypatch.setattr(watchdog_module, "write_report_atomic", failure_fn)
    elif phase_str == "notification":
        monkeypatch.setattr(notifications_module, "check_and_notify_for_red_transitions", failure_fn)
    return_code_int, heartbeat_list, _webhook_list, output_path_obj = _run_watchdog(monkeypatch, tmp_path,
        summary_builder_fn=failure_fn if phase_str == "summary" else None, heartbeat_env_url_str=HEARTBEAT_URL_STR)
    assert return_code_int == 2 and heartbeat_list == []
    assert receipt_path_obj.read_text(encoding="utf-8") == '{"prior_receipt":true}'
    assert output_path_obj.exists() is (phase_str == "notification")
    assert json.loads(capsys.readouterr().out)["reason_code_str"] == "watchdog_fatal_error"


def test_run_receipt_prior_success_hash_cannot_match_report_from_later_fatal_pass(monkeypatch, tmp_path, capsys):
    _run_watchdog(monkeypatch, tmp_path, heartbeat_env_url_str=HEARTBEAT_URL_STR)
    capsys.readouterr()
    receipt_path_obj = tmp_path / "ops_report_latest.run.json"
    before_bytes = receipt_path_obj.read_bytes()
    def failed_notification_fn(*argument_list, **argument_dict):
        raise OSError("state write failed")
    monkeypatch.setattr(notifications_module, "check_and_notify_for_red_transitions", failed_notification_fn)
    return_code_int, heartbeat_list, _webhook_list, output_path_obj = _run_watchdog(monkeypatch, tmp_path,
        summary_dict=_summary_dict(severity_str="red"), heartbeat_env_url_str=HEARTBEAT_URL_STR)
    report_dict = json.loads(output_path_obj.read_text(encoding="utf-8"))
    assert return_code_int == 2 and heartbeat_list == [] and receipt_path_obj.read_bytes() == before_bytes
    assert hashlib.sha256(json.dumps(report_dict, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")).hexdigest() != json.loads(before_bytes)["report_sha256_str"]


@pytest.mark.parametrize("severity_str,expected_exit_int", [("green", 0), ("red", 1)])
def test_run_receipt_atomic_replace_failure_preserves_previous_file_network_and_exit(monkeypatch, tmp_path, capsys, severity_str, expected_exit_int):
    receipt_path_obj = tmp_path / "ops_report_latest.run.json"
    receipt_path_obj.write_text('{"prior_receipt":true}', encoding="utf-8")
    original_replace_fn = watchdog_module.os.replace
    def replacement_fn(source_obj, target_obj):
        if Path(target_obj) == receipt_path_obj:
            raise PermissionError("secret filesystem path and token")
        return original_replace_fn(source_obj, target_obj)
    monkeypatch.setattr(watchdog_module.os, "replace", replacement_fn)
    return_code_int, heartbeat_list, webhook_list, _output_obj = _run_watchdog(monkeypatch, tmp_path,
        summary_dict=_summary_dict(severity_str=severity_str), heartbeat_env_url_str=HEARTBEAT_URL_STR,
        discord_webhook_url_str="https://discord.example/private-webhook")
    result_dict = json.loads(capsys.readouterr().out)
    assert return_code_int == expected_exit_int and len(heartbeat_list) == 1
    assert len(webhook_list) == (1 if severity_str == "red" else 0)
    assert result_dict["heartbeat_status_str"] == "sent"
    assert result_dict["run_receipt_status_str"] == "unavailable"
    assert result_dict["run_receipt_reason_code_str"] == "watchdog_run_receipt_unavailable"
    assert "secret" not in str(result_dict)
    assert receipt_path_obj.read_text(encoding="utf-8") == '{"prior_receipt":true}'
    assert list(tmp_path.glob("*.tmp")) == []


@pytest.mark.parametrize("problem_str", ["missing_owner", "invalid_release", "duplicate", "bad_mode", "oversized"])
def test_run_receipt_invalid_scope_is_withheld_after_existing_actions_complete(monkeypatch, tmp_path, capsys, problem_str):
    summary_dict = _summary_dict()
    row_dict = summary_dict["pod_row_dict_list"][0]
    if problem_str == "missing_owner":
        del row_dict["user_id_str"]
    elif problem_str == "invalid_release":
        row_dict["release_id_str"] = "C:/private/path"
    elif problem_str == "duplicate":
        summary_dict["pod_row_dict_list"].append(deepcopy(row_dict))
    elif problem_str == "bad_mode":
        row_dict["mode_str"] = "other"
    else:
        summary_dict["pod_row_dict_list"] = [deepcopy(row_dict)] * 129
    return_code_int, heartbeat_list, _webhook_list, output_obj = _run_watchdog(monkeypatch, tmp_path,
        summary_dict=summary_dict, heartbeat_env_url_str=HEARTBEAT_URL_STR)
    assert return_code_int == 0 and len(heartbeat_list) == 1 and output_obj.exists()
    assert not output_obj.with_suffix(".run.json").exists()
    result_dict = json.loads(capsys.readouterr().out)
    assert result_dict["run_receipt_reason_code_str"] == "watchdog_run_receipt_unavailable"


def test_run_receipt_contains_only_identity_time_and_result_not_private_payloads(monkeypatch, tmp_path):
    summary_dict = _summary_dict()
    summary_dict["pod_row_dict_list"][0]["private_dict"] = {"token": "TOP-SECRET", "path": "C:/private/state.sqlite"}
    summary_dict["pod_row_dict_list"][0]["debug_summary_dict"]["primary_reason_str"] = "TOP-SECRET"
    _run_watchdog(monkeypatch, tmp_path, summary_dict=summary_dict,
        heartbeat_env_url_str="https://secret.example/private-heartbeat", discord_webhook_url_str="https://secret.example/private-discord")
    receipt_str = (tmp_path / "ops_report_latest.run.json").read_text(encoding="utf-8")
    for private_str in ("TOP-SECRET", "C:/private", "secret.example", "private-heartbeat", "private-discord", "vps_01", "primary_reason_str"):
        assert private_str not in receipt_str
