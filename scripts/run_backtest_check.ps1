<#
Task Scheduler wrapper (AlphaBacktestCheck, Mon-Fri 06:30 ET): run the report-only backtest
check, then refresh the scheduled-tasks receipt that System health shows. Exit code = the
backtest check's. The Python commands load config.env themselves.
#>

[CmdletBinding()]
param([switch]$RulersOnly)

$ErrorActionPreference = "Stop"
$script_dir_path_str = Split-Path -Parent $MyInvocation.MyCommand.Path
$repo_root_path_str = Split-Path -Parent $script_dir_path_str
Set-Location -LiteralPath $repo_root_path_str

$uv_command_obj = Get-Command uv -ErrorAction SilentlyContinue
if ($null -ne $uv_command_obj) {
    $uv_exe_path_str = $uv_command_obj.Source
}
else {
    $uv_exe_path_str = @((Join-Path $env:USERPROFILE ".local\bin\uv.exe"), (Join-Path $env:USERPROFILE ".cargo\bin\uv.exe")) |
        Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
    if ([string]::IsNullOrWhiteSpace($uv_exe_path_str)) {
        throw "uv.exe not found on PATH or in known per-user install locations."
    }
}

if ($RulersOnly) {
    & $uv_exe_path_str run python -m alpha.live.scheduled_backtest --rulers-only
}
else {
    & $uv_exe_path_str run python -m alpha.live.scheduled_backtest
}
$check_exit_int = $LASTEXITCODE
& $uv_exe_path_str run python -m alpha.live.scheduled_tasks check | Out-Null
exit $check_exit_int
