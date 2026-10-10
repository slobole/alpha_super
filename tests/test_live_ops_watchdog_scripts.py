"""Execute PowerShell wrappers offline; never register a task or run the watchdog."""
from __future__ import annotations

import base64
import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest


REPOSITORY_PATH_OBJ = Path(__file__).resolve().parents[1]
POWERSHELL_PATH_STR = shutil.which("powershell.exe")
pytestmark = pytest.mark.skipif(os.name != "nt" or not POWERSHELL_PATH_STR, reason="Windows PowerShell contract")


def _run_powershell(command_str, environment_dict):
    encoded_str = base64.b64encode(command_str.encode("utf-16-le")).decode("ascii")
    return subprocess.run([POWERSHELL_PATH_STR, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
        "-EncodedCommand", encoded_str], env=environment_dict, capture_output=True, text=True, timeout=30)


def _script_case(tmp_path, parameter_dict):
    repository_path_obj = tmp_path / "checkout's & spaces"
    scripts_path_obj = repository_path_obj / "scripts"
    scripts_path_obj.mkdir(parents=True)
    for script_str in ("run_live_ops_watchdog.ps1", "setup_live_ops_watchdog_task.ps1"):
        shutil.copyfile(REPOSITORY_PATH_OBJ / "scripts" / script_str, scripts_path_obj / script_str)
    parameters_path_obj = tmp_path / "parameters.json"
    parameters_path_obj.write_text(json.dumps(parameter_dict), encoding="utf-8")
    environment_dict = dict(os.environ, WATCHDOG_TEST_PARAMS=str(parameters_path_obj),
        WATCHDOG_TEST_SCRIPT_DIR=str(scripts_path_obj), WATCHDOG_TEST_CAPTURE=str(tmp_path / "capture.json"))
    return repository_path_obj, environment_dict


PARAMETER_LOAD_STR = """
$ErrorActionPreference = 'Stop'
$parameter_dict = @{}
(Get-Content -LiteralPath $env:WATCHDOG_TEST_PARAMS -Raw | ConvertFrom-Json).PSObject.Properties |
    ForEach-Object { $parameter_dict[$_.Name] = $_.Value }
"""


def _run_wrapper(tmp_path, parameter_dict):
    repository_path_obj, environment_dict = _script_case(tmp_path, parameter_dict)
    fake_uv_path_obj = tmp_path / "fake uv.ps1"
    fake_uv_path_obj.write_text("""
@{arguments=@($args); working_directory=(Get-Location).Path} | ConvertTo-Json -Depth 5 |
    Set-Content -LiteralPath $env:WATCHDOG_TEST_CAPTURE -Encoding UTF8
$global:LASTEXITCODE = 17
""", encoding="utf-8")
    environment_dict["WATCHDOG_TEST_UV"] = str(fake_uv_path_obj)
    result_obj = _run_powershell(PARAMETER_LOAD_STR + """
function Get-Command { param($Name, $ErrorAction)
    if ($Name -ne 'uv') { throw 'Unexpected command lookup' }
    [pscustomobject]@{Source=$env:WATCHDOG_TEST_UV}
}
& (Join-Path $env:WATCHDOG_TEST_SCRIPT_DIR 'run_live_ops_watchdog.ps1') @parameter_dict
exit $LASTEXITCODE
""", environment_dict)
    assert result_obj.returncode == 17, result_obj.stderr
    capture_dict = json.loads(Path(environment_dict["WATCHDOG_TEST_CAPTURE"]).read_text(encoding="utf-8-sig"))
    assert Path(capture_dict["working_directory"]) == repository_path_obj
    return capture_dict["arguments"]


@pytest.mark.parametrize("parameter_dict,expected_list", [({}, []), ({"Mode": "live"}, ["--mode", "live"])])
def test_wrapper_preserves_legacy_mode_only_arguments_and_exit_code(tmp_path, parameter_dict, expected_list):
    assert _run_wrapper(tmp_path, parameter_dict) == ["run", "python", "scripts\\live_ops_watchdog.py", *expected_list]


def test_daily_wrapper_uses_separate_paths_and_preserves_empty_heartbeat_override(tmp_path):
    assert _run_wrapper(tmp_path, {"Mode": "live", "DailyHeartbeat": True, "HeartbeatUrl": ""}) == [
        "run", "python", "scripts\\live_ops_watchdog.py", "--mode", "live", "--daily-heartbeat",
        "--output-path", "alpha/live/logs/daily_watchdog/ops_report_latest.json",
        "--notification-state-path", "alpha/live/logs/daily_watchdog/notification_state.json", "--heartbeat-url=",
    ]


def test_wrapper_forwards_explicit_daily_paths_and_url_without_interpreting_them(tmp_path):
    parameter_dict = {"Mode": "live", "DailyHeartbeat": True, "ReleasesRoot": "C:\\owner's & daily\\",
        "OutputPath": "C:\\daily reports\\report.json", "NotificationStatePath": "C:\\daily state\\state.json",
        "EventLogPath": "C:\\daily serve\\events.jsonl", "DashboardConfig": "C:\\daily mappings\\config.yaml",
        "HeartbeatUrl": "https://heartbeat.invalid/token?a='value'&b=$false", "Json": True}
    # A trailing separator is dropped: Windows PowerShell 5.1 would let it
    # escape the closing quote of a native argument that contains a space.
    assert _run_wrapper(tmp_path, parameter_dict) == [
        "run", "python", "scripts\\live_ops_watchdog.py", "--mode", "live", "--daily-heartbeat",
        "--releases-root", "C:\\owner's & daily", "--output-path", parameter_dict["OutputPath"],
        "--notification-state-path", parameter_dict["NotificationStatePath"],
        "--heartbeat-url=" + parameter_dict["HeartbeatUrl"], "--event-log-path", parameter_dict["EventLogPath"],
        "--dashboard-config", parameter_dict["DashboardConfig"], "--json",
    ]


SCHEDULER_STUB_STR = """
function New-ScheduledTaskAction { param($Execute, $Argument, $WorkingDirectory)
    [pscustomobject]@{Execute=$Execute; Argument=$Argument; WorkingDirectory=$WorkingDirectory}
}
function New-ScheduledTaskTrigger { param([switch]$Once, $At, $RepetitionInterval) [pscustomobject]@{} }
function New-ScheduledTaskSettingsSet { param($MultipleInstances, $ExecutionTimeLimit, [switch]$StartWhenAvailable)
    [pscustomobject]@{MultipleInstances=$MultipleInstances; ExecutionTimeLimitSeconds=$ExecutionTimeLimit.TotalSeconds}
}
function New-ScheduledTaskPrincipal { param($UserId, $LogonType, $RunLevel) [pscustomobject]@{} }
function Register-ScheduledTask { param($TaskName, $Action, $Trigger, $Settings, $Principal, [switch]$Force)
    @{TaskName=$TaskName; Action=$Action; Settings=$Settings} | ConvertTo-Json -Depth 5 |
        Set-Content -LiteralPath $env:WATCHDOG_TEST_CAPTURE -Encoding UTF8
}
function Unregister-ScheduledTask { param($TaskName, $Confirm)
    @{UnregisteredTaskName=$TaskName} | ConvertTo-Json | Set-Content -LiteralPath $env:WATCHDOG_TEST_CAPTURE -Encoding UTF8
}
"""


def _run_setup(tmp_path, parameter_dict):
    repository_path_obj, environment_dict = _script_case(tmp_path, parameter_dict)
    result_obj = _run_powershell(PARAMETER_LOAD_STR + SCHEDULER_STUB_STR + """
& (Join-Path $env:WATCHDOG_TEST_SCRIPT_DIR 'setup_live_ops_watchdog_task.ps1') @parameter_dict
""", environment_dict)
    assert result_obj.returncode == 0, result_obj.stderr
    return repository_path_obj, environment_dict, json.loads(
        Path(environment_dict["WATCHDOG_TEST_CAPTURE"]).read_text(encoding="utf-8-sig"))


@pytest.mark.parametrize("mode_str", ["", "live"])
def test_setup_preserves_exact_legacy_task_action(tmp_path, mode_str):
    repository_path_obj, _, capture_dict = _run_setup(tmp_path, {"Mode": mode_str})
    expected_str = f'-NoProfile -ExecutionPolicy Bypass -File "{repository_path_obj / "scripts" / "run_live_ops_watchdog.ps1"}"'
    if mode_str:
        expected_str += " -Mode live"
    assert capture_dict["TaskName"] == "AlphaLiveOpsWatchdog"
    assert capture_dict["Action"] == {"Execute": "powershell.exe", "Argument": expected_str,
        "WorkingDirectory": str(repository_path_obj)}
    assert capture_dict["Settings"] == {"MultipleInstances": "IgnoreNew", "ExecutionTimeLimitSeconds": 600}


@pytest.mark.parametrize("heartbeat_str", ["", "https://heartbeat.invalid/path?owner='ops'&value=$false"])
def test_daily_scheduled_action_roundtrips_all_parameters_through_windows_powershell(tmp_path, heartbeat_str):
    parameter_dict = {"Mode": "live", "DailyHeartbeat": True, "HeartbeatUrl": heartbeat_str,
        "ReleasesRoot": "C:\\owner's & releases\\", "OutputPath": "C:\\daily reports\\report.json",
        "NotificationStatePath": "C:\\daily state\\notifications.json", "EventLogPath": "C:\\daily serve\\events.jsonl",
        "DashboardConfig": "C:\\daily mapping\\config.yaml", "Json": True}
    repository_path_obj, environment_dict, capture_dict = _run_setup(tmp_path, parameter_dict)
    assert capture_dict["TaskName"] == "AlphaDailyOpsWatchdog"
    assert capture_dict["Action"]["WorkingDirectory"] == str(repository_path_obj)
    # Execute the actual scheduled command against a recording wrapper, never uv.
    wrapper_path_obj = repository_path_obj / "scripts" / "run_live_ops_watchdog.ps1"
    wrapper_path_obj.write_text("""
param([string]$Mode, [switch]$DailyHeartbeat, [string]$ReleasesRoot, [string]$OutputPath,
    [string]$NotificationStatePath, [AllowEmptyString()][string]$HeartbeatUrl, [string]$EventLogPath,
    [string]$DashboardConfig, [switch]$Json)
$capture_dict = @{}
foreach ($key_str in $PSBoundParameters.Keys) {
    $value_obj = $PSBoundParameters[$key_str]
    if ($value_obj -is [System.Management.Automation.SwitchParameter]) { $value_obj = [bool]$value_obj }
    $capture_dict[$key_str] = $value_obj
}
$capture_dict | ConvertTo-Json | Set-Content -LiteralPath $env:WATCHDOG_TEST_CAPTURE -Encoding UTF8
exit 17
""", encoding="utf-8")
    result_obj = subprocess.run(f'"{POWERSHELL_PATH_STR}" {capture_dict["Action"]["Argument"]}',
        env=environment_dict, capture_output=True, text=True, timeout=30)
    assert result_obj.returncode == 17, result_obj.stderr
    assert json.loads(Path(environment_dict["WATCHDOG_TEST_CAPTURE"]).read_text(encoding="utf-8-sig")) == parameter_dict


def test_daily_unregistration_targets_separate_task_by_default(tmp_path):
    _, _, capture_dict = _run_setup(tmp_path, {"DailyHeartbeat": True, "Unregister": True})
    assert capture_dict == {"UnregisteredTaskName": "AlphaDailyOpsWatchdog"}


def test_wrapper_keeps_a_drive_root_separator(tmp_path):
    arguments_list = _run_wrapper(tmp_path, {"Mode": "paper", "DailyHeartbeat": True, "ReleasesRoot": "D:\\"})
    assert arguments_list[arguments_list.index("--releases-root") + 1] == "D:\\"


@pytest.mark.parametrize("parameter_dict,message_str", [
    ({"Mode": "live", "ReleasesRoot": "C:\\alpha\\daily_releases"}, "Refusing to change the NDX/TAA task"),
    ({"TaskName": "AlphaLiveOpsWatchdog", "DailyHeartbeat": True, "ReleasesRoot": "C:\\alpha\\daily_releases"},
        "Refusing to change the NDX/TAA task"),
    ({"TaskName": "AlphaLiveOpsWatchdog", "DailyHeartbeat": True, "Unregister": True}, "Refusing to change the NDX/TAA task"),
    ({"Mode": "live", "HeartbeatUrl": ""}, "Refusing to change the NDX/TAA task"),
    ({"Mode": "paper", "DailyHeartbeat": True}, "requires -ReleasesRoot"),
    ({"Mode": "paper", "DailyHeartbeat": True, "ReleasesRoot": "<repo>\\alpha\\live\\releases\\daily"},
        "must be outside alpha\\live\\releases"),
    # A relative root resolves against the repo root, as the task's wrapper does.
    ({"Mode": "paper", "DailyHeartbeat": True, "ReleasesRoot": "alpha\\live\\releases\\daily"},
        "must be outside alpha\\live\\releases"),
    ({"TaskName": "AlphaLiveOpsWatchdog ", "DailyHeartbeat": True, "ReleasesRoot": "C:\\alpha\\daily_releases"},
        "Refusing to change the NDX/TAA task"),
])
def test_setup_refuses_daily_options_that_would_touch_the_ndx_taa_task(tmp_path, parameter_dict, message_str):
    repository_path_obj, environment_dict = _script_case(tmp_path, {})
    parameter_dict = {key_str: value_obj.replace("<repo>", str(repository_path_obj)) if isinstance(value_obj, str)
        else value_obj for key_str, value_obj in parameter_dict.items()}
    Path(environment_dict["WATCHDOG_TEST_PARAMS"]).write_text(json.dumps(parameter_dict), encoding="utf-8")
    result_obj = _run_powershell(PARAMETER_LOAD_STR + SCHEDULER_STUB_STR + """
& (Join-Path $env:WATCHDOG_TEST_SCRIPT_DIR 'setup_live_ops_watchdog_task.ps1') @parameter_dict
""", environment_dict)
    assert result_obj.returncode != 0
    assert message_str in result_obj.stderr + result_obj.stdout
    # Nothing was registered or unregistered.
    assert not Path(environment_dict["WATCHDOG_TEST_CAPTURE"]).exists()
