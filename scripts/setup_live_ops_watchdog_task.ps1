<#
Register (or remove) the Windows Task Scheduler job that runs the Live OPS
watchdog every few minutes. Uses Register-ScheduledTask (not schtasks.exe)
because the design needs MultipleInstances=IgnoreNew (overlap guard) and an
ExecutionTimeLimit, which schtasks cannot set.

Usage:
  .\scripts\setup_live_ops_watchdog_task.ps1                  # register, 5 min
  .\scripts\setup_live_ops_watchdog_task.ps1 -IntervalMinutes 10
  .\scripts\setup_live_ops_watchdog_task.ps1 -Unregister      # remove
#>

[CmdletBinding()]
param(
    [string]$TaskName = "AlphaLiveOpsWatchdog",
    [int]$IntervalMinutes = 5,
    [switch]$Unregister
)

$ErrorActionPreference = "Stop"

function Write-Step {
    param([string]$LevelStr, [string]$MessageStr)
    Write-Host "[$LevelStr] $MessageStr"
}

if ($Unregister) {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
    Write-Step "PASS" "Unregistered scheduled task '$TaskName'."
    exit 0
}

$script_dir_path_str = Split-Path -Parent $MyInvocation.MyCommand.Path
$repo_root_path_str = Split-Path -Parent $script_dir_path_str
$wrapper_path_str = Join-Path $script_dir_path_str "run_live_ops_watchdog.ps1"
if (-not (Test-Path -LiteralPath $wrapper_path_str)) {
    throw "Wrapper script not found: $wrapper_path_str"
}

$action_obj = New-ScheduledTaskAction -Execute "powershell.exe" `
    -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$wrapper_path_str`"" `
    -WorkingDirectory $repo_root_path_str

# -Once + -RepetitionInterval without -RepetitionDuration repeats indefinitely
# on Windows 10/11 (older builds needed -RepetitionDuration [TimeSpan]::MaxValue).
$trigger_obj = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(1) `
    -RepetitionInterval (New-TimeSpan -Minutes $IntervalMinutes)

# IgnoreNew: never start a second instance while one is still running.
# ExecutionTimeLimit: kill a hung build (broker socket, locked SQLite); the
# killed run never pings the dead-man switch, so the external watcher alerts.
$settings_obj = New-ScheduledTaskSettingsSet -MultipleInstances IgnoreNew `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 10) -StartWhenAvailable

# S4U: runs whether the user is logged on or not, without storing a password.
# Never run as SYSTEM — wrong USERPROFILE means uv and config.env are missing.
$principal_obj = New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME" `
    -LogonType S4U -RunLevel Limited

Register-ScheduledTask -TaskName $TaskName -Action $action_obj -Trigger $trigger_obj `
    -Settings $settings_obj -Principal $principal_obj -Force | Out-Null

Write-Step "PASS" "Registered scheduled task '$TaskName' every $IntervalMinutes minute(s)."
Write-Step "PASS" "Wrapper: $wrapper_path_str"
Write-Step "INFO" "Verify now:  Start-ScheduledTask -TaskName $TaskName"
Write-Step "INFO" "Inspect:     Get-ScheduledTaskInfo -TaskName $TaskName"
