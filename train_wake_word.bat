@echo off
REM Runs the standard pipeline: synthetic clips only -> train -> convert ->
REM verify -> package. Prompts for wake phrase, model name, sample count,
REM cutoff (writes config\<model_name>.yaml for you).
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo.
    echo ERROR: Setup hasn't been run yet in this folder.
    echo Run install.bat first.
    pause
    exit /b 1
)

.venv\Scripts\python.exe scripts\train_wake_word.py %*
pause
