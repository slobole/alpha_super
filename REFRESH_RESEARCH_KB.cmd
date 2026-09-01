@echo off
title Refresh Research Knowledge Base
cd /d "%~dp0"

set "PAKAL_ROOT_PATH=%~dp0..\pakal"
set "KB_GENERATOR_PATH=%PAKAL_ROOT_PATH%\pakal-research\research_os\build_research_knowledge_base.py"
set "KB_OUTPUT_PATH=%~dp0docs\research\knowledge-base"

if not exist "%KB_GENERATOR_PATH%" (
  echo Pakal knowledge-base generator was not found:
  echo %KB_GENERATOR_PATH%
  pause
  exit /b 1
)

uv run python "%KB_GENERATOR_PATH%" --repo-root "%PAKAL_ROOT_PATH%" --reports-dir "%PAKAL_ROOT_PATH%\pakal-research\reports" --output-dir "%KB_OUTPUT_PATH%"
if errorlevel 1 (
  echo.
  echo Research Knowledge Base refresh failed. Existing evidence was not published.
  pause
  exit /b 1
)

uv run mkdocs build --strict
if errorlevel 1 (
  echo.
  echo The Knowledge Base was generated, but the strict portal build failed.
  pause
  exit /b 1
)

echo.
echo Research Knowledge Base refreshed and validated.
pause
