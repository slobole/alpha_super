<#
Run the once-daily IBKR Flex performance shadow sync from Task Scheduler.
The Python command loads ignored config.env and never prints the token.
#>

[CmdletBinding()]
param()

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

& $uv_exe_path_str run python -m alpha.live.ibkr_performance_sync sync
exit $LASTEXITCODE
