@echo off
REM Runs the V2 pipeline: synthetic clips + real recordings mixed in -> train
REM -> convert -> verify -> package -> test against held-out real recordings.
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo.
    echo ERROR: Setup hasn't been run yet in this folder.
    echo Run install.bat first.
    pause
    exit /b 1
)

.venv\Scripts\python.exe scripts\train_with_real_samples.py %*
pause
