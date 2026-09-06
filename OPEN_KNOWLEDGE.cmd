@echo off
title Alpha Super Workspace
cd /d "%~dp0"

powershell.exe -NoProfile -Command "try { $responseObj = Invoke-WebRequest -UseBasicParsing 'http://127.0.0.1:8765/healthz' -TimeoutSec 2; if ($responseObj.StatusCode -eq 200 -and $responseObj.Content -like 'bench ok*') { Start-Process 'http://127.0.0.1:8765/knowledge/'; exit 0 } } catch {}; exit 1"
if not errorlevel 1 exit /b 0

start "" /min powershell.exe -NoProfile -WindowStyle Hidden -Command "$deadlineObj = (Get-Date).AddSeconds(45); do { Start-Sleep -Milliseconds 500; try { $responseObj = Invoke-WebRequest -UseBasicParsing 'http://127.0.0.1:8765/healthz' -TimeoutSec 2; if ($responseObj.StatusCode -eq 200 -and $responseObj.Content -like 'bench ok*') { Start-Process 'http://127.0.0.1:8765/knowledge/'; exit 0 } } catch {} } while ((Get-Date) -lt $deadlineObj)"
uv run python -m alpha.bench

if errorlevel 1 (
  echo.
  echo The Alpha Super workspace could not start.
  pause
)
