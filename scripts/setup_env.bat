@echo off
REM Windows (Command Prompt): create .venv and install the requirements.
REM Usage: double-click, or run  scripts\setup_env.bat  from the seed_project folder.
cd /d "%~dp0.."
set "PY=python"
where py >nul 2>nul && set "PY=py -3"
%PY% -c "import sys; assert sys.version_info >= (3, 10), 'Python 3.10 or newer is required'" || goto :fail
%PY% -m venv .venv || goto :fail
.venv\Scripts\python.exe -m pip install --upgrade pip || goto :fail
.venv\Scripts\python.exe -m pip install -r requirements.txt || goto :fail
echo.
echo Done. Run:   .venv\Scripts\python scripts\run_count.py
pause
exit /b 0
:fail
echo Setup failed - see the messages above.
pause
exit /b 1
