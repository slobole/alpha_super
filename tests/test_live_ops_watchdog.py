from __future__ import annotations

from datetime import UTC, datetime
import json
from pathlib import Path

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
        heartbeat_call_list.append((url_str, payload_dict))
        return True

    monkeypatch.setattr(ops_report_module, "post_heartbeat_bool", fake_post_heartbeat_bool)

    webhook_call_list: list[tuple[str, dict[str, object]]] = []

    def fake_post_discord_webhook_bool(url_str, payload_dict):
        webhook_call_list.append((url_str, payload_dict))
        return True

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
