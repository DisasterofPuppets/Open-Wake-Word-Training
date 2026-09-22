@echo off
REM Pin working directory to this script's own folder, no matter where it was run from.
cd /d "%~dp0"
REM Thin launcher -- all actual logic is in download_assets.ps1 (PowerShell).
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0download_assets.ps1" %*
