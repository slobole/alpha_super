<#
Task Scheduler wrapper (AlphaBacktestRuler, Saturday 10:00 ET): build missing or old rare-loss
rulers (one long backtest per LIVE pod) away from trading hours. Never touches the report.
#>

[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
& (Join-Path (Split-Path -Parent $MyInvocation.MyCommand.Path) "run_backtest_check.ps1") -RulersOnly
exit $LASTEXITCODE
