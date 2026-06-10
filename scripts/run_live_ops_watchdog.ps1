<#
Thin wrapper for the Live OPS watchdog: anchor to the repo root, resolve uv
(Task Scheduler S4U sessions may not carry the user's PATH), run the Python
watchdog, and propagate its exit code. No config.env parsing here — the Python
script loads it itself so there is exactly one parser.
#>

[CmdletBinding()]
param(
    [string]$Mode = ""
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

# Forward an optional mode scope (live/paper/incubation) to the watchdog.
# Omitting -Mode keeps the all-modes default.
$py_arg_list = @()
if ($Mode) { $py_arg_list += @("--mode", $Mode) }

& $uv_exe_path_str run python scripts\live_ops_watchdog.py @py_arg_list
exit $LASTEXITCODE
