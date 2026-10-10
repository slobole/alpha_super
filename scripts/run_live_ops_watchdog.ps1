<#
Thin wrapper for the Live OPS watchdog: anchor to the repo root, resolve uv
(Task Scheduler S4U sessions may not carry the user's PATH), run the Python
watchdog, and propagate its exit code. No config.env parsing here — the Python
script loads it itself so there is exactly one parser.
#>

[CmdletBinding()]
param(
    [string]$Mode = "",
    [switch]$DailyHeartbeat,
    [string]$ReleasesRoot = "",
    [string]$OutputPath = "",
    [string]$NotificationStatePath = "",
    [AllowEmptyString()][string]$HeartbeatUrl,
    [string]$EventLogPath = "",
    [string]$DashboardConfig = "",
    [switch]$Json
)

$ErrorActionPreference = "Stop"

$script_dir_path_str = Split-Path -Parent $MyInvocation.MyCommand.Path
$repo_root_path_str = Split-Path -Parent $script_dir_path_str
Set-Location -LiteralPath $repo_root_path_str

$uv_command_obj = Get-Command uv -ErrorAction SilentlyContinue
if ($null -ne $uv_command_obj) {
    $uv_exe_path_str = $uv_command_obj.Source
}
else {
    $uv_candidate_path_list = @(
        (Join-Path $env:USERPROFILE ".local\bin\uv.exe"),
        (Join-Path $env:USERPROFILE ".cargo\bin\uv.exe")
    )
    $uv_exe_path_str = $uv_candidate_path_list | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
    if ([string]::IsNullOrWhiteSpace($uv_exe_path_str)) {
        throw "uv.exe not found on PATH or in known per-user install locations."
    }
}

function Get-NativePathArgument {
    # Windows PowerShell 5.1 quotes an argument containing a space, and a
    # trailing backslash then escapes the closing quote for the native process.
    # Drop trailing separators (a drive root keeps one; it has no space).
    param([string]$PathStr)
    if (-not $PathStr) { return $PathStr }
    $trimmed_path_str = $PathStr.TrimEnd('\', '/')
    if ($trimmed_path_str -match '^[A-Za-z]:$') { return $trimmed_path_str + '\' }
    return $trimmed_path_str
}
$ReleasesRoot = Get-NativePathArgument $ReleasesRoot
$OutputPath = Get-NativePathArgument $OutputPath
$NotificationStatePath = Get-NativePathArgument $NotificationStatePath
$EventLogPath = Get-NativePathArgument $EventLogPath
$DashboardConfig = Get-NativePathArgument $DashboardConfig

# Forward an optional mode scope (live/paper/incubation) to the watchdog.
# Omitting -Mode keeps the all-modes default.
$py_arg_list = @()
if ($Mode) { $py_arg_list += @("--mode", $Mode) }
if ($DailyHeartbeat) {
    $py_arg_list += "--daily-heartbeat"
    if (-not $OutputPath) { $OutputPath = "alpha/live/logs/daily_watchdog/ops_report_latest.json" }
    if (-not $NotificationStatePath) { $NotificationStatePath = "alpha/live/logs/daily_watchdog/notification_state.json" }
}
if ($ReleasesRoot) { $py_arg_list += @("--releases-root", $ReleasesRoot) }
if ($OutputPath) { $py_arg_list += @("--output-path", $OutputPath) }
if ($NotificationStatePath) { $py_arg_list += @("--notification-state-path", $NotificationStatePath) }
if ($PSBoundParameters.ContainsKey("HeartbeatUrl")) {
    # An explicit empty override disables the heartbeat. The equals form also
    # survives Windows PowerShell 5.1 native argument handling of empty strings.
    $py_arg_list += "--heartbeat-url=$HeartbeatUrl"
}
if ($EventLogPath) { $py_arg_list += @("--event-log-path", $EventLogPath) }
if ($DashboardConfig) { $py_arg_list += @("--dashboard-config", $DashboardConfig) }
if ($Json) { $py_arg_list += "--json" }

& $uv_exe_path_str run python scripts\live_ops_watchdog.py @py_arg_list
exit $LASTEXITCODE
