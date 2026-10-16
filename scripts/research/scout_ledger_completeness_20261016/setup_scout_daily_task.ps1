<#
Register (or remove) the Windows Task Scheduler job that runs Scout's daily research run on this workstation
(design A21): `python -m alpha.scout daily` = ledger chain check, S8 shadows, live pod health (report only), BY-FDR.

Research only. It reads Norgate and the Scout ledger and writes under results/scout/. It never touches a live path,
account, order or the VPS. Health reads the Flex store if present, else the Flex XML files in Downloads
(SCOUT_FLEX_XML_GLOB overrides); with old files it says STALE.

Usage:
  .\scripts\research\scout_ledger_completeness_20261016\setup_scout_daily_task.ps1              # register, 07:30 daily
  .\scripts\research\scout_ledger_completeness_20261016\setup_scout_daily_task.ps1 -At 08:15
  .\scripts\research\scout_ledger_completeness_20261016\setup_scout_daily_task.ps1 -Unregister
  Start-ScheduledTask -TaskName ScoutDailyResearch                                              # run now

07:30 Israel time is after Norgate's overnight update of the previous US session. StartWhenAvailable runs a missed
run when the machine wakes; IgnoreNew prevents overlap; the run is capped at 90 minutes.
#>

[CmdletBinding()]
param(
    [string]$TaskName = "ScoutDailyResearch",
    [string]$At = "07:30",
    [switch]$Unregister
)

$ErrorActionPreference = "Stop"

if ($Unregister) {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
    Write-Host "[PASS] Unregistered scheduled task '$TaskName'."
    exit 0
}

$script_dir_path_str = Split-Path -Parent $MyInvocation.MyCommand.Path
$repo_root_path_str = (Resolve-Path (Join-Path $script_dir_path_str "..\..\..")).Path
$python_path_str = Join-Path $repo_root_path_str ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python_path_str)) {
    throw "Python not found: $python_path_str"
}
$log_dir_path_str = Join-Path $repo_root_path_str "results\scout\daily\logs"
New-Item -ItemType Directory -Force -Path $log_dir_path_str | Out-Null

# One appending log (the dated status files are written by Python); PYTHONUTF8 keeps the markdown readable.
$command_str = "set `"PYTHONUTF8=1`" &&`"$python_path_str`" -m alpha.scout daily >> `"$log_dir_path_str\daily.log`" 2>&1"
$action = New-ScheduledTaskAction -Execute "cmd.exe" -Argument "/c $command_str" -WorkingDirectory $repo_root_path_str
$trigger = New-ScheduledTaskTrigger -Daily -At $At
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -MultipleInstances IgnoreNew `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 90) -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
$principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive -RunLevel Limited

Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger -Settings $settings -Principal $principal `
    -Description "Scout daily research run (alpha_super, research only): ledger, shadows, pod health, FDR." -Force | Out-Null
Write-Host "[PASS] Registered '$TaskName' daily at $At in $repo_root_path_str (logs: $log_dir_path_str)."
