@echo off
REM Master launcher -- asks how you're training, then hands off to the
REM right script. (Named "Train.bat" -- NOT "Train_Wake_Word.bat": Windows
REM filenames are case-insensitive, so that name previously collided with
REM and silently overwrote train_wake_word.bat below. Keep this name distinct
REM from every other .bat in this folder.)
cd /d "%~dp0"

:MENU
echo.
echo ============================================================
echo  Train Wake Word
echo ============================================================
echo.
echo  1. Standard training
echo     Synthetic TTS clips only. Prompts for wake phrase, model
echo     name, sample count, cutoff. Use this for a brand new wake
echo     word with no real recordings of your own.
echo.
echo  2. Training with real recordings mixed in (bulk)
echo     Opens a picker to choose which config to train, mixes in
echo     real_samples\^<model_name^>\train\*.wav alongside synthetic
echo     clips, and tests the finished model against
echo     real_samples\^<model_name^>\eval_holdout\*.wav.
echo.
set /p CHOICE="Choose 1 or 2: "

if "%CHOICE%"=="1" goto STANDARD
if "%CHOICE%"=="2" goto REALSAMPLES
echo.
echo Please enter 1 or 2.
goto MENU

:STANDARD
call "%~dp0train_wake_word.bat"
goto END

:REALSAMPLES
call "%~dp0train_with_real_samples.bat"
goto END

:END
