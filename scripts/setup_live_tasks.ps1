<#
Register every Live Ops scheduled task in one go, then check them (PASS / FAIL, like doctor).
The task list lives in alpha/live/scheduled_tasks.py (TASK_TUPLE). Safe to run again after each deploy.

Usage:
  .\scripts\setup_live_tasks.ps1              # register all, then check
  .\scripts\setup_live_tasks.ps1 -CheckOnly   # only check, change nothing

Tasks:
  AlphaLiveOpsWatchdog      every 5 min, LIVE only   (setup_live_ops_watchdog_task.ps1 -Mode live)
  AlphaIbkrPerformanceSync  daily 06:15 ET           (setup_ibkr_performance_task.ps1)
  AlphaBacktestCheck        Mon-Fri 06:30 ET         (report only; stopped at 09:00 if still running)
  AlphaBacktestRuler        Saturday 10:00 ET        (long backtests for the rare-loss rulers)
#>

[CmdletBinding()]
param([switch]$CheckOnly)

$ErrorActionPreference = "Stop"
$script_dir_path_str = Split-Path -Parent $MyInvocation.MyCommand.Path
$repo_root_path_str = Split-Path -Parent $script_dir_path_str
Set-Location -LiteralPath $repo_root_path_str

function Register-WeeklyTask {
    param([string]$NameStr, [string]$WrapperStr, [string[]]$DayList, [string]$AtStr, [int]$LimitMinutesInt)
    $wrapper_path_str = Join-Path $script_dir_path_str $WrapperStr
    if (-not (Test-Path -LiteralPath $wrapper_path_str)) { throw "Wrapper script not found: $wrapper_path_str" }
    $action_obj = New-ScheduledTaskAction -Execute "powershell.exe" `
        -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$wrapper_path_str`"" -WorkingDirectory $repo_root_path_str
    $trigger_obj = New-ScheduledTaskTrigger -Weekly -DaysOfWeek $DayList -At $AtStr
    # IgnoreNew: never two runs at once. The time limit keeps a slow run away from the open.
    $settings_obj = New-ScheduledTaskSettingsSet -MultipleInstances IgnoreNew `
        -ExecutionTimeLimit (New-TimeSpan -Minutes $LimitMinutesInt) -StartWhenAvailable
    $principal_obj = New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME" -LogonType S4U -RunLevel Limited
    Register-ScheduledTask -TaskName $NameStr -Action $action_obj -Trigger $trigger_obj `
        -Settings $settings_obj -Principal $principal_obj -Force | Out-Null
    Write-Host "[PASS] Registered '$NameStr' $($DayList -join ',') $AtStr ET (stops after $LimitMinutesInt min)."
}

if (-not $CheckOnly) {
    $time_zone_id_str = (Get-TimeZone).Id
    if ($time_zone_id_str -ne "Eastern Standard Time") {
        throw "VPS time zone must be 'Eastern Standard Time' (New York). Current: $time_zone_id_str"
    }
    & (Join-Path $script_dir_path_str "setup_live_ops_watchdog_task.ps1") -Mode live
    & (Join-Path $script_dir_path_str "setup_ibkr_performance_task.ps1")
    Register-WeeklyTask -NameStr "AlphaBacktestCheck" -WrapperStr "run_backtest_check.ps1" `
        -DayList @("Monday", "Tuesday", "Wednesday", "Thursday", "Friday") -AtStr "06:30" -LimitMinutesInt 150
    Register-WeeklyTask -NameStr "AlphaBacktestRuler" -WrapperStr "run_backtest_ruler.ps1" `
        -DayList @("Saturday") -AtStr "10:00" -LimitMinutesInt 480
}

$uv_command_obj = Get-Command uv -ErrorAction SilentlyContinue
$uv_exe_path_str = if ($null -ne $uv_command_obj) { $uv_command_obj.Source } else {
    @((Join-Path $env:USERPROFILE ".local\bin\uv.exe"), (Join-Path $env:USERPROFILE ".cargo\bin\uv.exe")) |
        Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1 }
if ([string]::IsNullOrWhiteSpace($uv_exe_path_str)) { Write-Host "[FAIL] uv.exe not found"; exit 1 }
& $uv_exe_path_str run python -m alpha.live.scheduled_tasks check
exit $LASTEXITCODE
