<#
Start the private Norgate artifact API in a visible debug window, wait for
/healthz, then run the server doctor. Reads ignored config.env when present.
#>

[CmdletBinding()]
param(
    [string]$ConfigPath = "",
    [string]$ServiceRoot,
    [string]$ApiHost,
    [int]$ApiPort = 0,
    [string]$ApiToken,
    [int]$HealthTimeoutSeconds = 90
)

$ErrorActionPreference = "Stop"

function Write-Step {
    param([string]$LevelStr, [string]$MessageStr)
    Write-Host "[$LevelStr] $MessageStr"
}

function ConvertTo-PsLiteral {
    param([string]$ValueStr)
    return "'" + $ValueStr.Replace("'", "''") + "'"
}

function Load-ConfigEnv {
    param([string]$ConfigPathStr)

    if (-not (Test-Path -LiteralPath $ConfigPathStr)) {
        Write-Step "INFO" "config.env not found: $ConfigPathStr"
        return
    }

    $line_number_int = 0
    foreach ($raw_line_str in Get-Content -LiteralPath $ConfigPathStr) {
        $line_number_int += 1
        $line_str = $raw_line_str.Trim()
        if ([string]::IsNullOrWhiteSpace($line_str) -or $line_str.StartsWith("#")) {
            continue
        }
        if ($line_str.StartsWith("export ")) {
            $line_str = $line_str.Substring(7).Trim()
        }

        $equal_index_int = $line_str.IndexOf("=")
        if ($equal_index_int -lt 1) {
            throw "Invalid config.env line $line_number_int`: expected KEY=value."
        }

        $key_str = $line_str.Substring(0, $equal_index_int).Trim()
        $value_str = $line_str.Substring($equal_index_int + 1).Trim()
        if ([string]::IsNullOrWhiteSpace($key_str)) {
            throw "Invalid config.env line $line_number_int`: empty key."
        }
        if ($value_str.Length -ge 2) {
            $first_char_str = $value_str.Substring(0, 1)
            $last_char_str = $value_str.Substring($value_str.Length - 1, 1)
            if (($first_char_str -eq "`"" -or $first_char_str -eq "'") -and $first_char_str -eq $last_char_str) {
                $value_str = $value_str.Substring(1, $value_str.Length - 2)
            }
        }
        if ([string]::IsNullOrWhiteSpace([Environment]::GetEnvironmentVariable($key_str, "Process"))) {
            [Environment]::SetEnvironmentVariable($key_str, $value_str, "Process")
        }
    }
    Write-Step "INFO" "loaded config.env: $ConfigPathStr"
}

function Test-ApiHealth {
    param([string]$ApiUrlStr)
    try {
        $response_obj = Invoke-WebRequest -Uri "$ApiUrlStr/healthz" -UseBasicParsing -TimeoutSec 5
        return [int]$response_obj.StatusCode -eq 200
    }
    catch {
        return $false
    }
}

$script_dir_path_str = Split-Path -Parent $MyInvocation.MyCommand.Path
$repo_root_path_str = (Resolve-Path -LiteralPath (Join-Path $script_dir_path_str "..")).Path
Set-Location -LiteralPath $repo_root_path_str

if ([string]::IsNullOrWhiteSpace($ConfigPath)) {
    $ConfigPath = Join-Path $repo_root_path_str "config.env"
}
Load-ConfigEnv -ConfigPathStr $ConfigPath

if (-not [string]::IsNullOrWhiteSpace($ApiToken)) { $env:NORGATE_API_TOKEN = $ApiToken }
if (-not [string]::IsNullOrWhiteSpace($ServiceRoot)) { $env:NORGATE_SERVICE_ROOT = $ServiceRoot }
if (-not [string]::IsNullOrWhiteSpace($ApiHost)) { $env:NORGATE_API_HOST = $ApiHost }
if ($ApiPort -gt 0) { $env:NORGATE_API_PORT = [string]$ApiPort }

$token_str = [string]$env:NORGATE_API_TOKEN
$service_root_path_str = [string]$env:NORGATE_SERVICE_ROOT
$api_host_str = [string]$env:NORGATE_API_HOST
$api_port_str = [string]$env:NORGATE_API_PORT
if ([string]::IsNullOrWhiteSpace($api_port_str)) { $api_port_str = "8787" }

if ([string]::IsNullOrWhiteSpace($token_str)) {
    throw "NORGATE_API_TOKEN is required. Put it in ignored config.env or pass -ApiToken."
}
if ($token_str -eq "CHANGE_ME_LONG_RANDOM_SECRET" -or $token_str -eq "test-token-123") {
    Write-Warning "NORGATE_API_TOKEN looks like a placeholder. Replace it before live use."
}
if ([string]::IsNullOrWhiteSpace($service_root_path_str)) {
    throw "NORGATE_SERVICE_ROOT is required. Example: C:\alpha\norgate_service"
}
if ([string]::IsNullOrWhiteSpace($api_host_str)) {
    throw "NORGATE_API_HOST is required. Use the Norgate node Tailscale IP, for example 100.123.13.69."
}

$api_port_int = [int]$api_port_str
$api_url_str = "http://$api_host_str`:$api_port_int"
New-Item -ItemType Directory -Force -Path $service_root_path_str | Out-Null

if ($null -eq (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw "uv was not found on PATH. Run this from a shell where uv is available."
}

Write-Step "INFO" "repo root: $repo_root_path_str"
Write-Step "INFO" "service root: $service_root_path_str"
Write-Step "INFO" "api url: $api_url_str"

$api_process_obj = $null
if (Test-ApiHealth -ApiUrlStr $api_url_str) {
    Write-Warning "API already responds at $api_url_str. Using the existing server process."
}
else {
    $repo_literal_str = ConvertTo-PsLiteral $repo_root_path_str
    $service_root_literal_str = ConvertTo-PsLiteral $service_root_path_str
    $api_host_literal_str = ConvertTo-PsLiteral $api_host_str
    $server_command_str = @"
`$ErrorActionPreference = 'Stop'
`$Host.UI.RawUI.WindowTitle = 'Norgate API debug'
Set-Location -LiteralPath $repo_literal_str
Write-Host '[INFO] Norgate API debug window. Stop with Ctrl+C.'
uv run python scripts\serve_norgate_snapshot_api.py --service-root $service_root_literal_str --host $api_host_literal_str --port $api_port_int
"@

    Write-Step "INFO" "starting Norgate API in a visible debug PowerShell window"
    $api_process_obj = Start-Process `
        -FilePath "powershell.exe" `
        -ArgumentList @("-NoExit", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", $server_command_str) `
        -PassThru

    $deadline_dt = (Get-Date).AddSeconds($HealthTimeoutSeconds)
    while ((Get-Date) -lt $deadline_dt) {
        if ($null -ne $api_process_obj -and $api_process_obj.HasExited) {
            throw "Norgate API process exited before /healthz became ready."
        }
        if (Test-ApiHealth -ApiUrlStr $api_url_str) {
            Write-Step "PASS" "api healthz: $api_url_str/healthz"
            break
        }
        Start-Sleep -Seconds 2
    }
    if (-not (Test-ApiHealth -ApiUrlStr $api_url_str)) {
        throw "Timed out waiting for Norgate API healthz: $api_url_str/healthz"
    }
}

Write-Step "INFO" "running server doctor"
uv run python scripts\doctor_norgate_server.py --service-root "$service_root_path_str" --api-url "$api_url_str"
$doctor_exit_code_int = $LASTEXITCODE
if ($doctor_exit_code_int -ne 0) {
    Write-Step "FAIL" "server doctor failed with exit code $doctor_exit_code_int"
    exit $doctor_exit_code_int
}

Write-Step "PASS" "server doctor passed"
exit 0
