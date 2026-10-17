<#
Register every Live Ops scheduled task in one go, then check them (PASS / FAIL, like doctor).
The task list lives in alpha/live/scheduled_tasks.py (TASK_TUPLE). Safe to run again after each deploy.

Usage:
  .\scripts\setup_live_tasks.ps1              # register all, then check
  .\scripts\setup_live_tasks.ps1 -CheckOnly   # only check, change nothing
  .\scripts\setup_live_tasks.ps1 -Disable AlphaBacktestCheck   # turn a task off (kept registered); 'All' for every task
  .\scripts\setup_live_tasks.ps1 -Enable AlphaBacktestCheck    # turn it back on
A turned-off task shows as OFF in the check and in System health (amber), never as missing.

Tasks:
  AlphaLiveOpsWatchdog      every 5 min, LIVE only   (setup_live_ops_watchdog_task.ps1 -Mode live)
  AlphaIbkrPerformanceSync  daily 06:15 ET           (setup_ibkr_performance_task.ps1)
  AlphaBacktestCheck        Mon-Fri 06:30 ET         (report only; stopped at 09:00 if still running)
  AlphaBacktestRuler        Saturday 10:00 ET        (long backtests for the rare-loss rulers)
#>

[CmdletBinding()]
param(
    [switch]$CheckOnly,
    [ValidateSet("All", "AlphaLiveOpsWatchdog", "AlphaIbkrPerformanceSync", "AlphaBacktestCheck", "AlphaBacktestRuler")]
    [string[]]$Disable = @(),
    [ValidateSet("All", "AlphaLiveOpsWatchdog", "AlphaIbkrPerformanceSync", "AlphaBacktestCheck", "AlphaBacktestRuler")]
    [string[]]$Enable = @()
)

$ErrorActionPreference = "Stop"
$script_dir_path_str = Split-Path -Parent $MyInvocation.MyCommand.Path
$repo_root_path_str = Split-Path -Parent $script_dir_path_str
Set-Location -LiteralPath $repo_root_path_str

function Register-WeeklyTask {
    param([string]$NameStr, [string]$WrapperStr, [string[]]$DayList, [string]$AtStr, [int]$LimitMinutesInt, [switch]$Disabled)
    $wrapper_path_str = Join-Path $script_dir_path_str $WrapperStr
    if (-not (Test-Path -LiteralPath $wrapper_path_str)) { throw "Wrapper script not found: $wrapper_path_str" }
    $action_obj = New-ScheduledTaskAction -Execute "powershell.exe" `
        -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$wrapper_path_str`"" -WorkingDirectory $repo_root_path_str
    $trigger_obj = New-ScheduledTaskTrigger -Weekly -DaysOfWeek $DayList -At $AtStr
    # Offset-free local wall time follows New York DST; the cmdlet may emit UTC otherwise.
    $trigger_obj.StartBoundary = (Get-Date).Date.Add([TimeSpan]::Parse($AtStr)).ToString("s")
    # The wrapper enforces the absolute NY deadline; do not catch up missed backtests.
    $settings_obj = New-ScheduledTaskSettingsSet -MultipleInstances IgnoreNew `
        -ExecutionTimeLimit (New-TimeSpan -Minutes $LimitMinutesInt) -Disable:$Disabled
    $principal_obj = New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME" -LogonType S4U -RunLevel Limited
    Register-ScheduledTask -TaskName $NameStr -Action $action_obj -Trigger $trigger_obj `
        -Settings $settings_obj -Principal $principal_obj -Force | Out-Null
    Write-Host "[PASS] Registered '$NameStr' $($DayList -join ',') $AtStr ET (stops after $LimitMinutesInt min)."
}

$all_task_list = @("AlphaLiveOpsWatchdog", "AlphaIbkrPerformanceSync", "AlphaBacktestCheck", "AlphaBacktestRuler")
if ($Disable.Count -or $Enable.Count) {
    foreach ($name_str in $(if ($Disable -contains "All") { $all_task_list } else { $Disable })) {
        Disable-ScheduledTask -TaskName $name_str | Out-Null
        Write-Host "[OFF] '$name_str' turned off (still registered; -Enable $name_str turns it back on)."
    }
    foreach ($name_str in $(if ($Enable -contains "All") { $all_task_list } else { $Enable })) {
        Enable-ScheduledTask -TaskName $name_str | Out-Null
        Write-Host "[PASS] '$name_str' turned on."
    }
    $CheckOnly = $true   # a switch change never re-registers anything
}

if (-not $CheckOnly) {
    $time_zone_id_str = (Get-TimeZone).Id
    if ($time_zone_id_str -ne "Eastern Standard Time") {
        throw "VPS time zone must be 'Eastern Standard Time' (New York). Current: $time_zone_id_str"
    }
    # Re-registering replaces a task and would switch it back on: keep turned-off tasks off.
    $kept_off_list = @($all_task_list | Where-Object {
        $task_obj = Get-ScheduledTask -TaskName $_ -ErrorAction SilentlyContinue
        $null -ne $task_obj -and [string]$task_obj.State -eq "Disabled" })
    # Register disabled from the outset: a later failure must never enable an OFF task.
    & (Join-Path $script_dir_path_str "setup_live_ops_watchdog_task.ps1") -Mode live `
        -Disabled:($kept_off_list -contains "AlphaLiveOpsWatchdog")
    & (Join-Path $script_dir_path_str "setup_ibkr_performance_task.ps1") `
        -Disabled:($kept_off_list -contains "AlphaIbkrPerformanceSync")
    Register-WeeklyTask -NameStr "AlphaBacktestCheck" -WrapperStr "run_backtest_check.ps1" `
        -DayList @("Monday", "Tuesday", "Wednesday", "Thursday", "Friday") -AtStr "06:30" -LimitMinutesInt 150 `
        -Disabled:($kept_off_list -contains "AlphaBacktestCheck")
    Register-WeeklyTask -NameStr "AlphaBacktestRuler" -WrapperStr "run_backtest_ruler.ps1" `
        -DayList @("Saturday") -AtStr "10:00" -LimitMinutesInt 480 `
        -Disabled:($kept_off_list -contains "AlphaBacktestRuler")
    foreach ($name_str in $kept_off_list) {
        Write-Host "[OFF] '$name_str' kept off (was turned off before this setup)."
    }
}

$uv_command_obj = Get-Command uv -ErrorAction SilentlyContinue
$uv_exe_path_str = if ($null -ne $uv_command_obj) { $uv_command_obj.Source } else {
    @((Join-Path $env:USERPROFILE ".local\bin\uv.exe"), (Join-Path $env:USERPROFILE ".cargo\bin\uv.exe")) |
        Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1 }
if ([string]::IsNullOrWhiteSpace($uv_exe_path_str)) { Write-Host "[FAIL] uv.exe not found"; exit 1 }
& $uv_exe_path_str run python -m alpha.live.scheduled_tasks check
exit $LASTEXITCODE
