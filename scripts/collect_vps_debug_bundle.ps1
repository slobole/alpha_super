<#
Collect a read-first VPS debug bundle for alpha_super live ops.

This wrapper anchors to the repo root, resolves uv.exe in common per-user
locations, and delegates all collection logic to the Python script.
#>

[CmdletBinding()]
param(
    [ValidateSet("live", "paper", "incubation")]
    [string]$Mode = "live",
    [string]$PodId = "",
    [string]$ReleaseManifestPath = "",
    [string]$ReleasesRoot = "",
    [string]$DbPath = "",
    [string]$OutputRoot = "results\vps_debug_bundles",
    [int]$TailLines = 300,
    [double]$TimeoutSeconds = 120.0,
    [switch]$IncludeRunnerDetails,
    [switch]$IncludeDoctor,
    [Nullable[int]]$DoctorBrokerClientId = $null,
    [switch]$IncludeNorgateDoctor,
    [Nullable[int]]$IbkrProbeClientId = $null,
    [switch]$NoZip,
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

$python_arg_list = @(
    "run",
    "python",
    "scripts\live_debug\collect_vps_debug_bundle.py",
    "--mode",
    $Mode,
    "--output-root",
    $OutputRoot,
    "--tail-lines",
    "$TailLines",
    "--timeout-seconds",
    "$TimeoutSeconds"
)

if ($PodId) {
    $python_arg_list += @("--pod-id", $PodId)
}
if ($ReleaseManifestPath) {
    $python_arg_list += @("--release-manifest-path", $ReleaseManifestPath)
}
if ($ReleasesRoot) {
    $python_arg_list += @("--releases-root", $ReleasesRoot)
}
if ($DbPath) {
    $python_arg_list += @("--db-path", $DbPath)
}
if ($IncludeRunnerDetails) {
    $python_arg_list += @("--include-runner-details")
}
if ($IncludeDoctor) {
    $python_arg_list += @("--include-doctor")
}
if ($null -ne $DoctorBrokerClientId) {
    $python_arg_list += @("--doctor-broker-client-id", "$DoctorBrokerClientId")
}
if ($IncludeNorgateDoctor) {
    $python_arg_list += @("--include-norgate-doctor")
}
if ($null -ne $IbkrProbeClientId) {
    $python_arg_list += @("--ibkr-probe-client-id", "$IbkrProbeClientId")
}
if ($NoZip) {
    $python_arg_list += @("--no-zip")
}
if ($Json) {
    $python_arg_list += @("--json")
}

& $uv_exe_path_str @python_arg_list
exit $LASTEXITCODE
