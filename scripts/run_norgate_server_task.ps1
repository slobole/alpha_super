<#
Hidden, on-demand Norgate API task action. See NORGATE_SERVER_DEPLOYMENT.md.
Registration, restart and vendor updates are separate operator actions.
#>
[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$repo_root_path_str = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$python_path_str = Join-Path $repo_root_path_str ".venv\Scripts\python.exe"
$bootstrap_path_str = Join-Path $PSScriptRoot "run_norgate_server_task.py"
foreach ($required_path_str in @($python_path_str, $bootstrap_path_str)) {
    if (-not (Test-Path -LiteralPath $required_path_str -PathType Leaf)) {
        throw "Required Norgate task file is missing: $required_path_str"
    }
}

$log_dir_path_str = Join-Path $repo_root_path_str "results\logs\norgate_api"
New-Item -ItemType Directory -Force -Path $log_dir_path_str | Out-Null
$run_id_str = (Get-Date).ToUniversalTime().ToString("yyyyMMddTHHmmssfffffffZ") + "_$PID"
$stdout_path_str = Join-Path $log_dir_path_str "${run_id_str}_stdout.log"
$stderr_path_str = Join-Path $log_dir_path_str "${run_id_str}_stderr.log"

# Module arguments contain no paths; WorkingDirectory/FilePath handle spaces.
# Stop-ScheduledTask alone may leave Python children: follow the runbook.
$process_obj = Start-Process -FilePath $python_path_str `
    -ArgumentList @("-B", "-u", "-m", "scripts.run_norgate_server_task") `
    -WorkingDirectory $repo_root_path_str -WindowStyle Hidden `
    -RedirectStandardOutput $stdout_path_str -RedirectStandardError $stderr_path_str `
    -PassThru -Wait
exit $process_obj.ExitCode
