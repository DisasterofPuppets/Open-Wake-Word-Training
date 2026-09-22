@echo off
REM Tests a trained .tflite model against one or more real WAV recordings.
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo.
    echo ERROR: Setup hasn't been run yet in this folder.
    echo Run install.bat first.
    pause
    exit /b 1
)

if "%~1"=="" (
    echo.
    echo Usage: eval_real_voice.bat ^<model.tflite^> ^<recording1.wav^> [recording2.wav ...] [--cutoff 0.7]
    echo Example: eval_real_voice.bat model_output\hey_holly.tflite real_samples\eval_holdout\*.wav
    pause
    exit /b 1
)

.venv\Scripts\python.exe scripts\eval_real_voice.py %*
pause
