@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0start_norgate_server.ps1" %*
exit /b %ERRORLEVEL%
