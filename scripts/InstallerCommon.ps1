# Shared helpers, dot-sourced by install.ps1, download_assets.ps1,
# train_wake_word.ps1, and installer_gui.ps1. Windows-only.

function Update-SessionPath {
    # Winget-installed tools (git, python) aren't on PATH in the CURRENT process
    # until the shell restarts. Re-read PATH from the registry (Machine + User)
    # so a just-installed tool is usable immediately, no restart needed.
    $machine = [System.Environment]::GetEnvironmentVariable('Path', 'Machine')
    $user = [System.Environment]::GetEnvironmentVariable('Path', 'User')
    $env:Path = "$machine;$user"
}

function Test-CommandExists {
    param([string]$Name)
    return [bool](Get-Command $Name -ErrorAction SilentlyContinue)
}

function Ensure-Winget {
    if (-not (Test-CommandExists 'winget')) {
        throw "winget (App Installer) is not available on this system. Install it from the Microsoft Store ('App Installer'), then re-run this script."
    }
}

function Ensure-Git {
    if (Test-CommandExists 'git') { return }
    Write-Host "Git not found -- installing via winget (Git.Git)..."
    Ensure-Winget
    winget install --id Git.Git -e --source winget --accept-package-agreements --accept-source-agreements
    if ($LASTEXITCODE -ne 0) { throw "winget failed to install Git (exit $LASTEXITCODE)." }
    Update-SessionPath
    if (-not (Test-CommandExists 'git')) {
        throw "Git was installed but isn't on PATH yet -- close this window and re-run the script (a fresh shell will pick it up)."
    }
}

function Ensure-Python {
    if (Test-CommandExists 'python') {
        # The "python" name can be a Microsoft Store alias stub that does nothing
        # but open the Store. Confirm it actually runs.
        $v = & python --version 2>&1
        if ($LASTEXITCODE -eq 0 -and $v -match 'Python 3\.(1[0-9]|[2-9][0-9])') { return }
    }
    Write-Host "Python 3.10+ not found -- installing via winget (Python.Python.3.12)..."
    Ensure-Winget
    winget install --id Python.Python.3.12 -e --source winget --accept-package-agreements --accept-source-agreements
    if ($LASTEXITCODE -ne 0) { throw "winget failed to install Python (exit $LASTEXITCODE)." }
    Update-SessionPath
    if (-not (Test-CommandExists 'python')) {
        throw "Python was installed but isn't on PATH yet -- close this window and re-run the script (a fresh shell will pick it up)."
    }
}

function Test-NvidiaGpuPresent {
    try {
        $gpu = Get-CimInstance Win32_VideoController -ErrorAction Stop | Where-Object { $_.Name -match 'NVIDIA' }
        return [bool]$gpu
    } catch {
        return $false
    }
}

function Confirm-YesNo {
    param([string]$Prompt)
    $answer = Read-Host "$Prompt (Y/N)"
    return $answer -match '^[Yy]'
}
