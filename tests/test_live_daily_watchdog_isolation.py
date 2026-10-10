"""Watchdog-only isolation: fake dashboard/transports and temporary state files."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
from types import ModuleType

import pytest

from alpha.live.core5_adapter import CORE5_STRATEGY_IMPORT_STR
from alpha.live.mr_capsule_adapter import MR_CAPSULE_STRATEGY_IMPORT_TUPLE
import scripts.live_ops_watchdog as watchdog_module
import test_live_ops_watchdog as watchdog_test_module
from test_live_ops_watchdog import AS_OF_TS, HEARTBEAT_URL_STR, _run_watchdog, _summary_dict


DAILY_HEARTBEAT_URL_STR = "https://hc-ping.example/daily-only"
DAILY_HEARTBEAT_ENV_STR = "ALPHA_DAILY_HEARTBEAT_URL"
REPO_ROOT_PATH = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def isolated_watchdog_io(monkeypatch):
    real_load_delivered_key_set_fn = watchdog_module.daily_watchdog_module.load_delivered_key_set
    monkeypatch.delenv(DAILY_HEARTBEAT_ENV_STR, raising=False)
    monkeypatch.setattr(watchdog_module.ops_report_module, "utc_now_ts", lambda: AS_OF_TS)
    real_refresh_enabled_since_fn = watchdog_module.daily_watchdog_module.refresh_enabled_since_dict
    monkeypatch.setattr(watchdog_module.daily_watchdog_module, "daily_heartbeat_alert_list",
        lambda *args, **kwargs: [])
    monkeypatch.setattr(watchdog_module.daily_watchdog_module, "load_delivered_key_set",
        lambda *args, **kwargs: set())
    # The fake dashboard has no release files: report one enabled daily pod and
    # no enabled-since history unless a test exercises those checks itself.
    monkeypatch.setattr(watchdog_module.daily_watchdog_module, "enabled_daily_release_count_int",
        lambda *args, **kwargs: 1)
    monkeypatch.setattr(watchdog_module.daily_watchdog_module, "refresh_enabled_since_dict",
        lambda *args, **kwargs: {})
    # Durable per-cycle delivery is covered by its own SQLite tests; this file
    # isolates the watchdog routing and failure boundary without opening pod DBs.
    monkeypatch.setattr(watchdog_module, "_run_capsule_notifications_tuple",
        lambda *args, **kwargs: ({}, []))
    return {"load_delivered_key_set": real_load_delivered_key_set_fn,
        "refresh_enabled_since_dict": real_refresh_enabled_since_fn}


def _daily_row_dict(strategy_import_str=CORE5_STRATEGY_IMPORT_STR, *,
        severity_str="green", mode_str="live", pod_id_str="daily_core5"):
    row_dict = deepcopy(_summary_dict(severity_str=severity_str)["pod_row_dict_list"][0])
    row_dict.update(pod_id_str=pod_id_str, release_id_str="release_" + pod_id_str,
        account_route_str="U_" + pod_id_str, strategy_import_str=strategy_import_str,
        mode_str=mode_str)
    return row_dict


def _mixed_summary_dict(*, daily_severity_str="green", legacy_severity_str="green"):
    summary_dict = _summary_dict(severity_str=legacy_severity_str)
    ndx_row_dict = deepcopy(summary_dict["pod_row_dict_list"][0])
    ndx_row_dict.update(pod_id_str="pod_ndx_live_01", release_id_str="release_ndx",
        account_route_str="U_NDX", strategy_import_str="strategies.ndx.strategy_ndx")
    summary_dict["pod_row_dict_list"].extend([ndx_row_dict,
        _daily_row_dict(severity_str=daily_severity_str)])
    # This is the original all-pod report that must not leak into daily alerts.
    summary_dict["inspector_report_dict"] = watchdog_module.ops_report_module.build_ops_report_dict(
        summary_dict, generated_at_ts=AS_OF_TS, vps_id_str="vps_01")
    return summary_dict


@pytest.mark.parametrize("severity_str", ["green", "red"])
@pytest.mark.parametrize("empty_override_bool", [False, True])
def test_default_monthly_watchdog_matches_actual_54b417f(monkeypatch, tmp_path, capsys,
        severity_str, empty_override_bool):
    source_str = subprocess.check_output(["git", "-c", f"safe.directory={REPO_ROOT_PATH.as_posix()}",
        "-c", "core.fsmonitor=false", "show", "54b417f:scripts/live_ops_watchdog.py"],
        cwd=REPO_ROOT_PATH, text=True, encoding="utf-8")
    baseline_module = ModuleType("baseline54_watchdog")
    baseline_module.__file__ = str(REPO_ROOT_PATH / "scripts" / "live_ops_watchdog.py")
    exec(compile(source_str, baseline_module.__file__, "exec"), baseline_module.__dict__)
    observation_list = []
    for name_str, implementation_module in (("baseline", baseline_module), ("current", watchdog_module)):
        monkeypatch.setattr(watchdog_test_module, "watchdog_module", implementation_module)
        return_code_int, heartbeat_list, webhook_list, report_path_obj = _run_watchdog(
            monkeypatch, tmp_path / name_str, summary_dict=_summary_dict(severity_str=severity_str),
            heartbeat_env_url_str=HEARTBEAT_URL_STR, discord_webhook_url_str="mock-discord",
            extra_argv_list=["--heartbeat-url", ""] if empty_override_bool else [])
        result_dict = json.loads(capsys.readouterr().out)
        result_dict.pop("report_output_path_str")
        observation_list.append((return_code_int, heartbeat_list, webhook_list, result_dict,
            json.loads(report_path_obj.read_text(encoding="utf-8")),
            json.loads(report_path_obj.with_suffix(".run.json").read_text(encoding="utf-8")),
            json.loads((report_path_obj.parent / "watchdog_notification_state.json").read_text(encoding="utf-8"))))
    assert observation_list[1] == observation_list[0]


@pytest.mark.parametrize("strategy_import_str", [CORE5_STRATEGY_IMPORT_STR, *MR_CAPSULE_STRATEGY_IMPORT_TUPLE])
def test_explicit_daily_red_never_emits_inspector_all_or_monthly_transition(
        monkeypatch, tmp_path, capsys, strategy_import_str):
    summary_dict = _mixed_summary_dict(daily_severity_str="red")
    summary_dict["pod_row_dict_list"][-1]["strategy_import_str"] = strategy_import_str
    monkeypatch.setenv(DAILY_HEARTBEAT_ENV_STR, DAILY_HEARTBEAT_URL_STR)
    return_code_int, heartbeat_list, webhook_list, report_path_obj = _run_watchdog(monkeypatch,
        tmp_path, summary_dict=summary_dict, heartbeat_env_url_str=HEARTBEAT_URL_STR,
        discord_webhook_url_str="mock-discord", extra_argv_list=["--daily-heartbeat", "--mode", "live"])
    assert return_code_int == 1
    assert [url_str for url_str, _ in heartbeat_list] == [DAILY_HEARTBEAT_URL_STR + "/fail"]
    assert all("INSPECTOR" not in payload_dict["content"] and "ALL /" not in payload_dict["content"]
        and "pod_ndx_live_01" not in payload_dict["content"] and "pod_taa_live_01" not in payload_dict["content"]
        for _, payload_dict in webhook_list)
    assert all("daily_core5" in payload_dict["content"] for _, payload_dict in webhook_list)
    report_dict = json.loads(report_path_obj.read_text(encoding="utf-8"))
    assert report_dict["pod_count_int"] == 1
    assert {row_dict["pod_id_str"] for row_dict in report_dict["pod_report_dict_list"]} == {"daily_core5"}
    state_dict = json.loads((tmp_path / "watchdog_notification_state.json").read_text(encoding="utf-8"))
    assert state_dict["pod_severity_map_dict"] == {"daily_core5": "red"}
    receipt_dict = json.loads(report_path_obj.with_suffix(".run.json").read_text(encoding="utf-8"))
    assert {row_dict["pod_id_str"] for row_dict in receipt_dict["scope_list"]} == {"daily_core5"}
    assert json.loads(capsys.readouterr().out)["heartbeat_fail_signal_bool"] is True


def test_explicit_daily_healthy_ignores_red_monthly_and_other_mode(monkeypatch, tmp_path, capsys):
    summary_dict = _mixed_summary_dict(legacy_severity_str="red")
    summary_dict["pod_row_dict_list"].append(_daily_row_dict(severity_str="red",
        mode_str="paper", pod_id_str="daily_paper"))
    monkeypatch.setenv(DAILY_HEARTBEAT_ENV_STR, DAILY_HEARTBEAT_URL_STR)
    return_code_int, heartbeat_list, webhook_list, report_path_obj = _run_watchdog(monkeypatch,
        tmp_path, summary_dict=summary_dict, heartbeat_env_url_str=HEARTBEAT_URL_STR,
        discord_webhook_url_str="mock-discord", extra_argv_list=["--daily-heartbeat", "--mode", "live"])
    assert return_code_int == 0 and webhook_list == []
    assert [url_str for url_str, _ in heartbeat_list] == [DAILY_HEARTBEAT_URL_STR]
    assert json.loads(report_path_obj.read_text(encoding="utf-8"))["overall_severity_str"] == "green"
    assert json.loads(capsys.readouterr().out)["notification_fired_count_int"] == 0


@pytest.mark.parametrize("daily_env_str,override_str,expected_url_str", [
    (None, None, None), (DAILY_HEARTBEAT_URL_STR, None, DAILY_HEARTBEAT_URL_STR),
    (DAILY_HEARTBEAT_URL_STR, "", None),
    (DAILY_HEARTBEAT_URL_STR, "https://hc-ping.example/explicit", "https://hc-ping.example/explicit")])
def test_daily_heartbeat_url_is_independent_and_empty_override_disables(monkeypatch, tmp_path,
        capsys, daily_env_str, override_str, expected_url_str):
    if daily_env_str is not None:
        monkeypatch.setenv(DAILY_HEARTBEAT_ENV_STR, daily_env_str)
    argv_list = ["--daily-heartbeat"]
    if override_str is not None:
        argv_list.extend(["--heartbeat-url", override_str])
    return_code_int, heartbeat_list, webhook_list, _ = _run_watchdog(monkeypatch, tmp_path,
        summary_dict={"as_of_timestamp_str": AS_OF_TS.isoformat(),
            "pod_row_dict_list": [_daily_row_dict()]},
        heartbeat_env_url_str=HEARTBEAT_URL_STR, discord_webhook_url_str="mock-discord",
        extra_argv_list=argv_list)
    result_dict = json.loads(capsys.readouterr().out)
    assert [url_str for url_str, _ in heartbeat_list] == ([] if expected_url_str is None else [expected_url_str])
    assert result_dict["heartbeat_status_str"] == ("disabled" if expected_url_str is None else "sent")
    if daily_env_str is None and override_str is None:
        # A missing daily URL is a broken monitor, not a silent opt-out.
        assert return_code_int == 1 and len(webhook_list) == 1
        assert result_dict["daily_watchdog_error_list"] == [{
            "reason_code_str": "daily_watchdog_heartbeat_url_missing", "error_type_str": "ConfigurationError"}]
    else:
        assert return_code_int == 0 and webhook_list == []


def test_daily_default_paths_are_distinct_before_any_report_write(monkeypatch, capsys):
    parsed_list = []
    def capture_pipeline(parsed_args_obj, _as_of_ts):
        parsed_list.append(parsed_args_obj)
        raise RuntimeError("stop before any file or network operation")
    monkeypatch.setattr(watchdog_module, "load_config_env_file", lambda **kwargs: {})
    monkeypatch.setattr(watchdog_module, "_run_report_pipeline_tuple", capture_pipeline)
    assert watchdog_module.main(["--daily-heartbeat", "--json", "--as-of-ts", AS_OF_TS.isoformat()]) == 2
    assert Path(parsed_list[0].output_path_str).parent == Path("alpha/live/logs/daily_watchdog")
    assert Path(parsed_list[0].notification_state_path_str).parent == Path("alpha/live/logs/daily_watchdog")
    assert json.loads(capsys.readouterr().out)["reason_code_str"] == "watchdog_fatal_error"


def test_daily_configured_event_log_reaches_dashboard_and_daily_check(monkeypatch, tmp_path):
    observed_app_dict = {}
    observed_check_list = []
    log_path_str = str(tmp_path / "daily events.jsonl")
    def summary_builder_fn(app_obj, as_of_ts=None):
        observed_app_dict.update(app_obj.kwargs)
        return {"as_of_timestamp_str": AS_OF_TS.isoformat(), "pod_row_dict_list": [_daily_row_dict()]}
    def daily_check_fn(*args, **kwargs):
        observed_check_list.append((args, kwargs))
        return []
    monkeypatch.setattr(watchdog_module.daily_watchdog_module, "daily_heartbeat_alert_list", daily_check_fn)
    return_code_int, _, _, _ = _run_watchdog(monkeypatch, tmp_path,
        summary_builder_fn=summary_builder_fn, discord_webhook_url_str="mock-discord",
        extra_argv_list=["--daily-heartbeat", "--event-log-path", log_path_str, "--heartbeat-url", ""])
    assert return_code_int == 0
    assert observed_app_dict["event_log_path_str"] == log_path_str
    assert observed_check_list[0][0][2] == log_path_str


@pytest.mark.parametrize("failure_phase_str", ["check", "enabled_since_state"])
def test_daily_check_failures_are_visible_once_without_monthly_contamination(
        monkeypatch, tmp_path, capsys, failure_phase_str):
    def fail_fn(*args, **kwargs):
        raise RuntimeError("private-token-and-path-must-not-be-printed")
    method_name_str = "daily_heartbeat_alert_list" if failure_phase_str == "check" else "refresh_enabled_since_dict"
    monkeypatch.setattr(watchdog_module.daily_watchdog_module, method_name_str, fail_fn)
    monkeypatch.setenv(DAILY_HEARTBEAT_ENV_STR, DAILY_HEARTBEAT_URL_STR)
    return_code_int, heartbeat_list, webhook_list, report_path_obj = _run_watchdog(monkeypatch,
        tmp_path, summary_dict=_mixed_summary_dict(), heartbeat_env_url_str=HEARTBEAT_URL_STR,
        discord_webhook_url_str="mock-discord", extra_argv_list=["--daily-heartbeat"])
    assert return_code_int == 1
    assert [url_str for url_str, _ in heartbeat_list] == [DAILY_HEARTBEAT_URL_STR + "/fail"]
    assert len(webhook_list) == 1
    assert "INSPECTOR" not in webhook_list[0][1]["content"]
    captured_obj = capsys.readouterr()
    result_dict = json.loads(captured_obj.out)
    report_dict = json.loads(report_path_obj.read_text(encoding="utf-8"))
    receipt_dict = json.loads(report_path_obj.with_suffix(".run.json").read_text(encoding="utf-8"))
    for observation_dict in (result_dict, report_dict, receipt_dict):
        assert observation_dict["daily_watchdog_error_list"] == [{
            "reason_code_str": "daily_watchdog_check_failed", "error_type_str": "RuntimeError"}]
        assert observation_dict["daily_watchdog_failure_alert_status_str"] == "sent"
        assert "private-token-and-path" not in json.dumps(observation_dict)
    assert "daily_watchdog_check_failed" in captured_obj.err
    assert "private-token-and-path" not in captured_obj.err
    assert report_dict["overall_severity_str"] == "red"


def test_daily_heartbeat_failure_tries_own_fail_once_and_stays_failed(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv(DAILY_HEARTBEAT_ENV_STR, DAILY_HEARTBEAT_URL_STR)
    return_code_int, heartbeat_list, webhook_list, report_path_obj = _run_watchdog(monkeypatch,
        tmp_path, summary_dict=_mixed_summary_dict(), heartbeat_env_url_str=HEARTBEAT_URL_STR,
        discord_webhook_url_str="mock-discord", heartbeat_delivery_bool=False,
        extra_argv_list=["--daily-heartbeat"])
    assert return_code_int == 1
    assert [url_str for url_str, _ in heartbeat_list] == [DAILY_HEARTBEAT_URL_STR,
        DAILY_HEARTBEAT_URL_STR + "/fail"]
    assert len(webhook_list) == 1
    captured_obj = capsys.readouterr()
    result_dict = json.loads(captured_obj.out)
    report_dict = json.loads(report_path_obj.read_text(encoding="utf-8"))
    receipt_dict = json.loads(report_path_obj.with_suffix(".run.json").read_text(encoding="utf-8"))
    for observation_dict in (result_dict, report_dict, receipt_dict):
        assert observation_dict["daily_watchdog_error_list"][0]["reason_code_str"] == "daily_watchdog_heartbeat_failed"
        assert observation_dict["daily_watchdog_failure_alert_status_str"] == "sent"
    assert result_dict["heartbeat_status_str"] == receipt_dict["heartbeat_status_str"] == "failed"
    assert report_dict["overall_severity_str"] == "red"
    assert "daily_watchdog_heartbeat_failed" in captured_obj.err

@pytest.mark.parametrize("cache_failure_str", ["corrupt", "locked"])
def test_real_daily_dedup_cache_failure_is_loud_and_daily_only(monkeypatch, tmp_path,
        capsys, cache_failure_str, isolated_watchdog_io):
    from contextlib import ExitStack, closing
    import sqlite3

    state_path_obj = tmp_path / "watchdog_notification_state.daily.sqlite3"
    # Restore the real state writer saved before this file's no-I/O autofixture;
    # no release files are needed to reach the watchdog-owned SQLite state.
    monkeypatch.setattr(watchdog_module.daily_watchdog_module, "refresh_enabled_since_dict",
        isolated_watchdog_io["refresh_enabled_since_dict"])
    monkeypatch.setattr(watchdog_module.daily_watchdog_module, "load_release_list", lambda _root_str: [])
    with ExitStack() as context_obj:
        if cache_failure_str == "corrupt":
            state_path_obj.write_bytes(b"not a SQLite database")
        else:
            connection_obj = context_obj.enter_context(closing(sqlite3.connect(state_path_obj)))
            connection_obj.execute("CREATE TABLE daily_watchdog_alert (alert_key_str TEXT, delivered_timestamp_str TEXT)")
            connection_obj.commit()
            connection_obj.execute("BEGIN EXCLUSIVE")
        monkeypatch.setenv(DAILY_HEARTBEAT_ENV_STR, DAILY_HEARTBEAT_URL_STR)
        return_code_int, heartbeat_list, webhook_list, report_path_obj = _run_watchdog(monkeypatch,
            tmp_path, summary_dict=_mixed_summary_dict(), heartbeat_env_url_str=HEARTBEAT_URL_STR,
            discord_webhook_url_str="mock-discord", extra_argv_list=["--daily-heartbeat"])
    assert return_code_int == 1
    assert [url_str for url_str, _ in heartbeat_list] == [DAILY_HEARTBEAT_URL_STR + "/fail"]
    assert len(webhook_list) == 1
    captured_obj = capsys.readouterr()
    assert "daily_watchdog_check_failed" in captured_obj.err
    result_dict = json.loads(captured_obj.out)
    report_dict = json.loads(report_path_obj.read_text(encoding="utf-8"))
    receipt_dict = json.loads(report_path_obj.with_suffix(".run.json").read_text(encoding="utf-8"))
    for observation_dict in (result_dict, report_dict, receipt_dict):
        assert observation_dict["daily_watchdog_error_list"] == [{
            "reason_code_str": "daily_watchdog_check_failed",
            "error_type_str": "DatabaseError" if cache_failure_str == "corrupt" else "OperationalError"}]
        assert observation_dict["daily_watchdog_failure_alert_status_str"] == "sent"
    legacy_state_dict = json.loads((tmp_path / "watchdog_notification_state.json").read_text(encoding="utf-8"))
    assert legacy_state_dict["pod_severity_map_dict"] == {"daily_core5": "green"}


@pytest.mark.parametrize("failure_mode_str", ["raises_then_succeeds", "raises_twice", "false_then_succeeds"])
def test_daily_heartbeat_transport_exception_is_contained_and_never_leaks_secrets(
        monkeypatch, tmp_path, capsys, failure_mode_str):
    heartbeat_url_list = []
    def heartbeat_poster_fn(url_str, payload_dict, **kwargs):
        heartbeat_url_list.append(url_str)
        if len(heartbeat_url_list) == 2 and failure_mode_str != "raises_twice":
            return True
        if failure_mode_str == "false_then_succeeds":
            return False
        raise OSError("https://private-host/token-secret filesystem-secret")
    def summary_builder_fn(app_obj, as_of_ts=None):
        # _run_watchdog installs its fake transport before building the summary.
        monkeypatch.setattr(watchdog_module.ops_report_module, "post_heartbeat_bool", heartbeat_poster_fn)
        return _mixed_summary_dict()
    monkeypatch.setenv(DAILY_HEARTBEAT_ENV_STR, DAILY_HEARTBEAT_URL_STR)
    return_code_int, _, webhook_list, report_path_obj = _run_watchdog(monkeypatch,
        tmp_path, summary_builder_fn=summary_builder_fn, heartbeat_env_url_str=HEARTBEAT_URL_STR,
        discord_webhook_url_str="mock-discord", extra_argv_list=["--daily-heartbeat"])
    assert return_code_int == 1
    assert heartbeat_url_list == [DAILY_HEARTBEAT_URL_STR, DAILY_HEARTBEAT_URL_STR + "/fail"]
    assert len(webhook_list) == 1
    captured_obj = capsys.readouterr()
    result_dict = json.loads(captured_obj.out)
    report_dict = json.loads(report_path_obj.read_text(encoding="utf-8"))
    receipt_dict = json.loads(report_path_obj.with_suffix(".run.json").read_text(encoding="utf-8"))
    for observation_dict in (result_dict, report_dict, receipt_dict):
        assert observation_dict["daily_watchdog_error_list"] == [{
            "reason_code_str": "daily_watchdog_heartbeat_failed",
            "error_type_str": "DeliveryFailed" if failure_mode_str == "false_then_succeeds" else "OSError"}]
        assert observation_dict["daily_watchdog_failure_alert_status_str"] == "sent"
    assert result_dict["heartbeat_status_str"] == receipt_dict["heartbeat_status_str"] == "failed"
    assert "daily_watchdog_heartbeat_failed" in captured_obj.err
    assert "secret" not in captured_obj.out + captured_obj.err + json.dumps(webhook_list)


def test_failed_daily_delivery_is_visible_and_never_suppresses_detailed_alerts(monkeypatch, tmp_path, capsys):
    received_call_list = []
    monkeypatch.setattr(watchdog_module.daily_watchdog_module, "daily_heartbeat_alert_list",
        lambda *args, **kwargs: [{"pod_id_str": "daily_core5", "kind_str": "decision_incomplete"}])
    def failed_delivery_fn(*args, **kwargs):
        assert kwargs["raise_on_failure_bool"] is True
        raise watchdog_module.daily_watchdog_module.DailyHeartbeatDeliveryError(1)
    def capsule_delivery_fn(summary_dict, mode_str, webhook_url_str):
        # The serve's detailed outbox runs with no heartbeat-based suppression.
        received_call_list.append((mode_str, webhook_url_str))
        return {}, []
    monkeypatch.setattr(watchdog_module.daily_watchdog_module, "deliver_daily_heartbeat_alerts", failed_delivery_fn)
    monkeypatch.setattr(watchdog_module.daily_watchdog_module, "load_delivered_key_set",
        lambda *_args: pytest.fail("Delivered heartbeat keys must not suppress detailed alerts"))
    monkeypatch.setattr(watchdog_module, "_run_capsule_notifications_tuple", capsule_delivery_fn)
    monkeypatch.setenv(DAILY_HEARTBEAT_ENV_STR, DAILY_HEARTBEAT_URL_STR)
    return_code_int, heartbeat_list, webhook_list, _ = _run_watchdog(monkeypatch, tmp_path,
        summary_dict=_mixed_summary_dict(), heartbeat_env_url_str=HEARTBEAT_URL_STR,
        discord_webhook_url_str="mock-discord", extra_argv_list=["--daily-heartbeat"])
    assert return_code_int == 1
    assert received_call_list == [(None, "mock-discord")]
    assert len(webhook_list) == 1
    assert [url_str for url_str, _ in heartbeat_list] == [DAILY_HEARTBEAT_URL_STR + "/fail"]
    assert json.loads(capsys.readouterr().out)["daily_watchdog_error_list"] == [{
        "reason_code_str": "daily_watchdog_check_failed", "error_type_str": "DailyHeartbeatDeliveryError"}]


def test_daily_late_report_failure_is_fatal_visible_and_sanitized(monkeypatch, tmp_path, capsys):
    original_write_fn = watchdog_module.write_report_atomic
    write_count_list = []
    def failing_rewrite_fn(report_dict, output_path_str):
        write_count_list.append(output_path_str)
        if len(write_count_list) > 1:
            raise OSError("secret-report-path")
        return original_write_fn(report_dict, output_path_str)
    def failed_check_fn(*args, **kwargs):
        raise RuntimeError("secret-check-data")
    monkeypatch.setattr(watchdog_module, "write_report_atomic", failing_rewrite_fn)
    monkeypatch.setattr(watchdog_module.daily_watchdog_module, "daily_heartbeat_alert_list", failed_check_fn)
    monkeypatch.setenv(DAILY_HEARTBEAT_ENV_STR, DAILY_HEARTBEAT_URL_STR)
    return_code_int, heartbeat_list, webhook_list, report_path_obj = _run_watchdog(monkeypatch,
        tmp_path, summary_dict=_mixed_summary_dict(), heartbeat_env_url_str=HEARTBEAT_URL_STR,
        discord_webhook_url_str="mock-discord", extra_argv_list=["--daily-heartbeat"])
    assert return_code_int == 2 and len(webhook_list) == 1
    assert [url_str for url_str, _ in heartbeat_list] == [DAILY_HEARTBEAT_URL_STR + "/fail"]
    captured_obj = capsys.readouterr()
    result_dict = json.loads(captured_obj.out)
    assert result_dict["reason_code_str"] == "watchdog_fatal_error"
    assert result_dict["error_str"] == "Daily watchdog finalization failed (OSError)."
    assert "Daily watchdog finalization failed" in captured_obj.err
    assert "secret" not in captured_obj.err + captured_obj.out + json.dumps(webhook_list)
    assert not report_path_obj.with_suffix(".run.json").exists()


@pytest.mark.parametrize("configured_via_str", ["env", "cli"])
@pytest.mark.parametrize("suffix_str", ["/", "/fail", "?rid=x", "#fragment"])
def test_daily_heartbeat_refuses_legacy_url_collision_including_known_endpoint_aliases(
        monkeypatch, tmp_path, capsys, configured_via_str, suffix_str):
    selected_url_str = HEARTBEAT_URL_STR + suffix_str
    argv_list = ["--daily-heartbeat"]
    if configured_via_str == "env":
        monkeypatch.setenv(DAILY_HEARTBEAT_ENV_STR, selected_url_str)
    else:
        monkeypatch.setenv(DAILY_HEARTBEAT_ENV_STR, DAILY_HEARTBEAT_URL_STR)
        argv_list.extend(["--heartbeat-url", selected_url_str])
    return_code_int, heartbeat_list, webhook_list, report_path_obj = _run_watchdog(monkeypatch,
        tmp_path, summary_dict=_mixed_summary_dict(), heartbeat_env_url_str=HEARTBEAT_URL_STR,
        discord_webhook_url_str="mock-discord", extra_argv_list=argv_list)
    assert return_code_int == 1 and heartbeat_list == []
    assert len(webhook_list) == 1
    captured_obj = capsys.readouterr()
    result_dict = json.loads(captured_obj.out)
    report_dict = json.loads(report_path_obj.read_text(encoding="utf-8"))
    receipt_dict = json.loads(report_path_obj.with_suffix(".run.json").read_text(encoding="utf-8"))
    for observation_dict in (result_dict, report_dict, receipt_dict):
        assert observation_dict["daily_watchdog_error_list"] == [{
            "reason_code_str": "daily_watchdog_heartbeat_scope_conflict", "error_type_str": "ConfigurationError"}]
        assert observation_dict["daily_watchdog_failure_alert_status_str"] == "sent"
    assert result_dict["heartbeat_status_str"] == receipt_dict["heartbeat_status_str"] == "failed"
    assert result_dict["heartbeat_fail_signal_bool"] is False
    assert "daily_watchdog_heartbeat_scope_conflict" in captured_obj.err
    assert HEARTBEAT_URL_STR not in captured_obj.out + captured_obj.err + json.dumps(webhook_list)
