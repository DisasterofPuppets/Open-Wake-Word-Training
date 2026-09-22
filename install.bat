@echo off
REM Thin launcher -- all actual logic is in install.ps1 (PowerShell).
REM Deliberately does NOT cd into this script's own folder -- install.ps1
REM installs into whatever directory you were in when you ran this, not into
REM this toolkit's own location. This .bat exists only because double-clicking
REM a .ps1 in Explorer opens it in a text editor by default rather than running it.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0install.ps1"
