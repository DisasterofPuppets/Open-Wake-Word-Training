#Requires -Version 5.1
# Launch via download_assets.bat (double-click), or:
#   powershell -NoProfile -ExecutionPolicy Bypass -File download_assets.ps1
# Any arguments given are passed straight through to scripts\download_assets.py
# (see that script's --help for flags).

$ErrorActionPreference = 'Stop'
$RepoDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvPy = Join-Path $RepoDir '.venv\Scripts\python.exe'

if (-not (Test-Path $VenvPy)) {
    Write-Host ""
    Write-Host "ERROR: Setup hasn't been run yet in this folder."
    Write-Host "Run install.bat first."
    Read-Host "Press Enter to exit"
    exit 1
}

Push-Location $RepoDir
try {
    & $VenvPy "scripts\download_assets.py" @args
} finally {
    Pop-Location
}
Read-Host "Press Enter to exit"
