"""Run the real setup offline with all Scheduler commands replaced by in-memory stubs."""

import base64
import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest

REPO_ROOT_PATH = Path(__file__).resolve().parents[1]
POWERSHELL_PATH_STR = shutil.which('powershell.exe')
pytestmark = pytest.mark.skipif(os.name != 'nt' or not POWERSHELL_PATH_STR, reason='Windows PowerShell contract')


@pytest.mark.parametrize('failure_name_str', ['', 'AlphaIbkrPerformanceSync', 'AlphaBacktestCheck', 'AlphaBacktestRuler'])
@pytest.mark.parametrize('real_trigger_bool', [False, True])
def test_disabled_tasks_never_become_enabled_even_on_partial_failure(tmp_path, failure_name_str, real_trigger_bool):
    script_dir_path = tmp_path / "checkout's & spaces" / 'scripts'
    script_dir_path.mkdir(parents=True)
    for name_str in ('setup_live_tasks.ps1', 'setup_live_ops_watchdog_task.ps1', 'setup_ibkr_performance_task.ps1',
                     'run_live_ops_watchdog.ps1', 'run_ibkr_performance_sync.ps1', 'run_backtest_check.ps1', 'run_backtest_ruler.ps1'):
        shutil.copyfile(REPO_ROOT_PATH / 'scripts' / name_str, script_dir_path / name_str)
    uv_path = tmp_path / 'uv.ps1'
    uv_path.write_text('$global:LASTEXITCODE = 0', encoding='utf-8')
    capture_path = tmp_path / 'capture.json'
    environment_dict = dict(os.environ, TASK_TEST_SCRIPT_DIR=str(script_dir_path), TASK_TEST_UV=str(uv_path),
                            TASK_TEST_CAPTURE=str(capture_path), TASK_TEST_FAILURE=failure_name_str,
                            TASK_TEST_REAL_TRIGGER='1' if real_trigger_bool else '0')
    command_str = r'''
$ErrorActionPreference = 'Stop'
if ($env:TASK_TEST_REAL_TRIGGER -eq '1') { Import-Module ScheduledTasks }
$global:task_state_dict = @{}
$global:registered_list = [System.Collections.Generic.List[object]]::new()
function Get-TimeZone { [pscustomobject]@{Id='Eastern Standard Time'} }
function Get-ScheduledTask { param($TaskName, $ErrorAction) [pscustomobject]@{State='Disabled'} }
function Get-Command { param($Name, $ErrorAction) [pscustomobject]@{Source=$env:TASK_TEST_UV} }
function New-ScheduledTaskAction { param($Execute, $Argument, $WorkingDirectory) [pscustomobject]@{} }
function New-ScheduledTaskTrigger { param([switch]$Once, [switch]$Weekly, [switch]$Daily, $At, $RepetitionInterval, $DaysOfWeek)
    if ($env:TASK_TEST_REAL_TRIGGER -eq '1') {
        if ($Once) { return ScheduledTasks\New-ScheduledTaskTrigger -Once -At $At -RepetitionInterval $RepetitionInterval }
        if ($Weekly) { return ScheduledTasks\New-ScheduledTaskTrigger -Weekly -At $At -DaysOfWeek $DaysOfWeek }
        return ScheduledTasks\New-ScheduledTaskTrigger -Daily -At $At
    }
    [pscustomobject]@{StartBoundary='initial'}
}
function New-ScheduledTaskSettingsSet { param($MultipleInstances, $ExecutionTimeLimit, [switch]$StartWhenAvailable, [switch]$Disable, $RestartCount, $RestartInterval)
    [pscustomobject]@{Disabled=[bool]$Disable; Catchup=[bool]$StartWhenAvailable}
}
function New-ScheduledTaskPrincipal { param($UserId, $LogonType, $RunLevel) [pscustomobject]@{} }
function Register-ScheduledTask { param($TaskName, $Action, $Trigger, $Settings, $Principal, [switch]$Force)
    if ($TaskName -eq $env:TASK_TEST_FAILURE) { throw 'injected registration failure' }
    $global:registered_list.Add(@{Name=$TaskName; Disabled=$Settings.Disabled; Catchup=$Settings.Catchup; StartBoundary=$Trigger.StartBoundary})
}
function Disable-ScheduledTask { throw 'Disabled state must be supplied at registration, never restored afterwards' }
try { & (Join-Path $env:TASK_TEST_SCRIPT_DIR 'setup_live_tasks.ps1') } catch { Write-Host $_ }
ConvertTo-Json -InputObject @($global:registered_list.ToArray()) -Depth 5 | Set-Content -LiteralPath $env:TASK_TEST_CAPTURE -Encoding UTF8
'''
    encoded_str = base64.b64encode(command_str.encode('utf-16-le')).decode('ascii')
    result_obj = subprocess.run([POWERSHELL_PATH_STR, '-NoProfile', '-NonInteractive', '-ExecutionPolicy', 'Bypass',
                                 '-EncodedCommand', encoded_str], env=environment_dict, capture_output=True, text=True, timeout=30)
    assert result_obj.returncode == 0, result_obj.stderr
    registration_list = json.loads(capture_path.read_text(encoding='utf-8-sig'))
    expected_count_int = {'': 4, 'AlphaIbkrPerformanceSync': 1, 'AlphaBacktestCheck': 2, 'AlphaBacktestRuler': 3}[failure_name_str]
    assert len(registration_list) == expected_count_int, result_obj.stdout + result_obj.stderr
    assert all(row_dict['Disabled'] for row_dict in registration_list)
    assert all(not row_dict['Catchup'] for row_dict in registration_list if row_dict['Name'].startswith('AlphaBacktest'))
    assert all(not row_dict['StartBoundary'].endswith('Z') and '+' not in row_dict['StartBoundary']
               for row_dict in registration_list if row_dict['Name'] != 'AlphaLiveOpsWatchdog')


@pytest.mark.parametrize('ruler_bool', [False, True])
def test_wrappers_use_guarded_runner_and_preserve_its_failure_code(tmp_path, ruler_bool):
    script_dir_path = tmp_path / 'scripts'
    script_dir_path.mkdir()
    for name_str in ('run_backtest_check.ps1', 'run_backtest_ruler.ps1'):
        shutil.copyfile(REPO_ROOT_PATH / 'scripts' / name_str, script_dir_path / name_str)
    uv_path = tmp_path / 'uv.ps1'
    uv_path.write_text(r'''
ConvertTo-Json -InputObject @($args) -Compress | Add-Content -LiteralPath $env:TASK_TEST_CAPTURE -Encoding UTF8
$global:LASTEXITCODE = $(if ($args -contains 'alpha.live.scheduled_backtest') { 17 } else { 0 })
''', encoding='utf-8')
    capture_path = tmp_path / 'calls.jsonl'
    environment_dict = dict(os.environ, TASK_TEST_UV=str(uv_path), TASK_TEST_CAPTURE=str(capture_path))
    name_str = 'run_backtest_ruler.ps1' if ruler_bool else 'run_backtest_check.ps1'
    command_str = "function Get-Command { param($Name, $ErrorAction) [pscustomobject]@{Source=$env:TASK_TEST_UV} }\n" + f"& '{str(script_dir_path / name_str).replace(chr(39), chr(39)*2)}'\nexit $LASTEXITCODE"
    encoded_str = base64.b64encode(command_str.encode('utf-16-le')).decode('ascii')
    result_obj = subprocess.run([POWERSHELL_PATH_STR, '-NoProfile', '-NonInteractive', '-ExecutionPolicy', 'Bypass',
                                 '-EncodedCommand', encoded_str], env=environment_dict, capture_output=True, text=True, timeout=30)
    assert result_obj.returncode == 17, result_obj.stderr
    call_list = [json.loads(line_str) for line_str in capture_path.read_text(encoding='utf-8-sig').splitlines()]
    assert call_list == [['run', 'python', '-m', 'alpha.live.scheduled_backtest'] + (['--rulers-only'] if ruler_bool else []),
                         ['run', 'python', '-m', 'alpha.live.scheduled_tasks', 'check']]
