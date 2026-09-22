@echo off
REM Pin working directory to this script's own folder, no matter where it was run from.
cd /d "%~dp0"
REM Thin launcher -- all actual logic is in installer_gui.ps1 (PowerShell/WinForms).
REM A PowerShell console stays open behind the GUI window -- that's normal,
REM it shows prerequisite-check output (e.g. git/Python auto-install via winget).
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0installer_gui.ps1"
