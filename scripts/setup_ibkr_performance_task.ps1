<#
Register the daily 06:15 New York IBKR Flex performance shadow sync.
The VPS must use Windows time zone "Eastern Standard Time" so DST follows
New York automatically.
#>

[CmdletBinding()]
param(
    [string]$TaskName = "AlphaIbkrPerformanceSync",
    [switch]$Unregister
)

$ErrorActionPreference = "Stop"

if ($Unregister) {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
    Write-Host "[PASS] Unregistered scheduled task '$TaskName'."
    exit 0
}

$time_zone_id_str = (Get-TimeZone).Id
if ($time_zone_id_str -ne "Eastern Standard Time") {
    throw "VPS time zone must be 'Eastern Standard Time' before registering this 06:15 ET task. Current: $time_zone_id_str"
}

$script_dir_path_str = Split-Path -Parent $MyInvocation.MyCommand.Path
$repo_root_path_str = Split-Path -Parent $script_dir_path_str
$wrapper_path_str = Join-Path $script_dir_path_str "run_ibkr_performance_sync.ps1"
if (-not (Test-Path -LiteralPath $wrapper_path_str)) {
    throw "Wrapper script not found: $wrapper_path_str"
}

$action_obj = New-ScheduledTaskAction -Execute "powershell.exe" `
    -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$wrapper_path_str`"" `
    -WorkingDirectory $repo_root_path_str
$trigger_obj = New-ScheduledTaskTrigger -Daily -At "06:15"
$settings_obj = New-ScheduledTaskSettingsSet -MultipleInstances IgnoreNew `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 10) -StartWhenAvailable `
    -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 15)
$principal_obj = New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME" `
    -LogonType S4U -RunLevel Limited

Register-ScheduledTask -TaskName $TaskName -Action $action_obj -Trigger $trigger_obj `
    -Settings $settings_obj -Principal $principal_obj -Force | Out-Null

Write-Host "[PASS] Registered '$TaskName' daily at 06:15 ET."
Write-Host "[INFO] Run now: Start-ScheduledTask -TaskName $TaskName"
Write-Host "[INFO] Inspect: Get-ScheduledTaskInfo -TaskName $TaskName"
