@echo off
REM Start the interactive review (http://localhost:8000). Close this window or press Ctrl+C to stop.
cd /d "%~dp0.."
if not exist .venv\Scripts\python.exe (
  echo Run scripts\setup_env.bat first.
  pause
  exit /b 1
)
.venv\Scripts\python.exe scripts\review.py %*
pause
