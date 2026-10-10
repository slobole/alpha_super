<#
Register (or remove) the Windows Task Scheduler job that runs the Live OPS
watchdog every few minutes. Uses Register-ScheduledTask (not schtasks.exe)
because the design needs MultipleInstances=IgnoreNew (overlap guard) and an
ExecutionTimeLimit, which schtasks cannot set.

Usage:
  .\scripts\setup_live_ops_watchdog_task.ps1                  # register, all modes
  .\scripts\setup_live_ops_watchdog_task.ps1 -Mode live       # scope to live only
  .\scripts\setup_live_ops_watchdog_task.ps1 -IntervalMinutes 10
  .\scripts\setup_live_ops_watchdog_task.ps1 -Unregister      # remove
  .\scripts\setup_live_ops_watchdog_task.ps1 -Mode live -DailyHeartbeat -ReleasesRoot C:\alpha\daily_releases

DailyHeartbeat uses a separate default task name, report and notification state,
and requires -ReleasesRoot outside alpha\live\releases. Daily-only options are
refused for the NDX/TAA task name AlphaLiveOpsWatchdog.
Use EventLogPath and DashboardConfig to match the daily serves' saved evidence.
With -DailyHeartbeat, HeartbeatUrl '' explicitly disables the daily dead-man
ping; otherwise Python requires ALPHA_DAILY_HEARTBEAT_URL from config.env.

Scope the task to -Mode live when incubation/paper rehearsal pods would
otherwise keep the dead-man switch permanently red (rehearsal pods build plans
but never submit, so their execution window always reads as missed).
#>

[CmdletBinding()]
param(
    [string]$TaskName = "AlphaLiveOpsWatchdog",
    [int]$IntervalMinutes = 5,
    [ValidateSet("", "live", "paper", "incubation")]
    [string]$Mode = "",
    [switch]$DailyHeartbeat,
    [string]$ReleasesRoot = "",
    [string]$OutputPath = "",
    [string]$NotificationStatePath = "",
    [AllowEmptyString()][string]$HeartbeatUrl,
    [string]$EventLogPath = "",
    [string]$DashboardConfig = "",
    [switch]$Json,
    [switch]$Unregister
)

$ErrorActionPreference = "Stop"

function Write-Step {
    param([string]$LevelStr, [string]$MessageStr)
    Write-Host "[$LevelStr] $MessageStr"
}

if ($DailyHeartbeat -and -not $PSBoundParameters.ContainsKey("TaskName")) {
    $TaskName = "AlphaDailyOpsWatchdog"
}

$script_dir_path_str = Split-Path -Parent $MyInvocation.MyCommand.Path
$repo_root_path_str = Split-Path -Parent $script_dir_path_str

# Guard the existing NDX/TAA task: daily-only options must never repoint or
# replace it (a forgotten -DailyHeartbeat would watch the daily root while the
# legacy dead-man check stays green).
# Capture first: inside a script block $PSBoundParameters is that block's own.
$bound_parameter_dict = $PSBoundParameters
$daily_option_list = @("DailyHeartbeat", "ReleasesRoot", "OutputPath", "NotificationStatePath", "HeartbeatUrl",
    "EventLogPath", "DashboardConfig", "Json") | Where-Object { $bound_parameter_dict.ContainsKey($_) }
if ($TaskName.Trim() -eq "AlphaLiveOpsWatchdog" -and $daily_option_list) {
    throw ("Refusing to change the NDX/TAA task 'AlphaLiveOpsWatchdog' with daily-only option(s): " +
        ($daily_option_list -join ", ") + ". Use -DailyHeartbeat with its own task name.")
}
if ($DailyHeartbeat -and -not $Unregister) {
    if (-not $ReleasesRoot) {
        throw "-DailyHeartbeat requires -ReleasesRoot: the daily pods' own release root."
    }
    $legacy_root_path_str = [IO.Path]::GetFullPath((Join-Path $repo_root_path_str "alpha\live\releases")).TrimEnd('\')
    # The task runs from the repo root, so resolve a relative root the same way
    # (.NET's own current directory is not the PowerShell location).
    $daily_root_input_str = if ([IO.Path]::IsPathRooted($ReleasesRoot)) { $ReleasesRoot } else { Join-Path $repo_root_path_str $ReleasesRoot }
    $daily_root_path_str = [IO.Path]::GetFullPath($daily_root_input_str).TrimEnd('\')
    if ($daily_root_path_str -eq $legacy_root_path_str -or
            $daily_root_path_str.StartsWith($legacy_root_path_str + "\", [StringComparison]::OrdinalIgnoreCase)) {
        throw "The daily release root must be outside alpha\live\releases (the NDX/TAA root reads every subfolder)."
    }
}

if ($Unregister) {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
    Write-Step "PASS" "Unregistered scheduled task '$TaskName'."
    exit 0
}

$wrapper_path_str = Join-Path $script_dir_path_str "run_live_ops_watchdog.ps1"
if (-not (Test-Path -LiteralPath $wrapper_path_str)) {
    throw "Wrapper script not found: $wrapper_path_str"
}

$wrapper_argument_str = "-NoProfile -ExecutionPolicy Bypass -File `"$wrapper_path_str`""
if ($Mode) { $wrapper_argument_str += " -Mode $Mode" }
if ($DailyHeartbeat -or $ReleasesRoot -or $OutputPath -or $NotificationStatePath -or
        $PSBoundParameters.ContainsKey("HeartbeatUrl") -or $EventLogPath -or $DashboardConfig -or $Json) {
    # Task Scheduler accepts one command-line string. Encode a literal PS call
    # so spaces, apostrophes, trailing backslashes and an empty URL round-trip
    # through powershell.exe without being interpreted as command syntax.
    $wrapper_command_str = "& '" + $wrapper_path_str.Replace("'", "''") + "'"
    foreach ($parameter_name_str in @("Mode", "ReleasesRoot", "OutputPath", "NotificationStatePath", "EventLogPath", "DashboardConfig")) {
        $parameter_value_str = Get-Variable -Name $parameter_name_str -ValueOnly
        if ($parameter_value_str) {
            $wrapper_command_str += " -$parameter_name_str '" + $parameter_value_str.Replace("'", "''") + "'"
        }
    }
    if ($DailyHeartbeat) { $wrapper_command_str += " -DailyHeartbeat" }
    if ($PSBoundParameters.ContainsKey("HeartbeatUrl")) {
        $wrapper_command_str += " -HeartbeatUrl '" + $HeartbeatUrl.Replace("'", "''") + "'"
    }
    if ($Json) { $wrapper_command_str += " -Json" }
    $wrapper_command_str += '; exit $LASTEXITCODE'
    $encoded_command_str = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($wrapper_command_str))
    $wrapper_argument_str = "-NoProfile -ExecutionPolicy Bypass -EncodedCommand $encoded_command_str"
}
$action_obj = New-ScheduledTaskAction -Execute "powershell.exe" `
    -Argument $wrapper_argument_str `
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

$mode_label_str = if ($Mode) { $Mode } else { "all modes" }
Write-Step "PASS" "Registered scheduled task '$TaskName' every $IntervalMinutes minute(s), scope: $mode_label_str."
Write-Step "PASS" "Wrapper: $wrapper_path_str"
Write-Step "INFO" "Verify now:  Start-ScheduledTask -TaskName $TaskName"
Write-Step "INFO" "Inspect:     Get-ScheduledTaskInfo -TaskName $TaskName"
