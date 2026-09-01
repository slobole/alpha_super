@echo off
title Alpha Super Knowledge
cd /d "%~dp0"

start "" /min powershell.exe -NoProfile -WindowStyle Hidden -Command "Start-Sleep -Seconds 2; Start-Process 'http://127.0.0.1:8765/'"
uv run mkdocs serve --dev-addr 127.0.0.1:8765

if errorlevel 1 (
  echo.
  echo The knowledge site could not start.
  pause
)
