#Requires -Version 5.1
# One-shot setup for this pipeline. Installs into the directory you were in
# when you RAN this script (not this toolkit's own folder) -- so you keep one
# copy of this toolkit somewhere and run its install.bat by full path from
# inside any new project folder to set a fresh, independent instance up there.
#
# Launch via install.bat (double-click while sitting in your target folder,
# or run it by full path from a terminal that's cd'd into your target folder), or:
#   powershell -NoProfile -ExecutionPolicy Bypass -File install.ps1

$ErrorActionPreference = 'Stop'
$TemplateDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$TargetDir = (Get-Location).Path

function Update-SessionPath {
    $machine = [System.Environment]::GetEnvironmentVariable('Path', 'Machine')
    $user = [System.Environment]::GetEnvironmentVariable('Path', 'User')
    $env:Path = "$machine;$user"
}
function Test-CommandExists {
    param([string]$Name)
    return [bool](Get-Command $Name -ErrorAction SilentlyContinue)
}
function Ensure-GitMinimal {
    if (Test-CommandExists 'git') { return }
    Write-Host "Git not found -- installing via winget..."
    if (-not (Test-CommandExists 'winget')) {
        throw "winget (App Installer) is not available. Install it from the Microsoft Store, then re-run."
    }
    winget install --id Git.Git -e --source winget --accept-package-agreements --accept-source-agreements
    if ($LASTEXITCODE -ne 0) { throw "winget failed to install Git (exit $LASTEXITCODE)." }
    Update-SessionPath
    if (-not (Test-CommandExists 'git')) {
        throw "Git installed but not on PATH yet -- close this window and re-run (a fresh shell will pick it up)."
    }
}

# Files/folders that make up the reusable toolkit -- copied into $TargetDir
# on first run there so it becomes a fully independent, self-contained project.
# install.bat/install.ps1 are deliberately NOT copied -- this folder is the
# toolkit's "repo"; a target folder only needs what it takes to download
# assets and train, not to re-run setup itself.
# 'config' is handled specially below: only the generic example template is
# copied, never a wake-word-specific config (e.g. hey_holly.yaml) -- those
# are personal to this instance and stay out of fresh installs.
$TemplateItems = @(
    'install.py',
    'README.md', 'PIPELINE_README.md',
    'Train.bat',
    'download_assets.bat', 'download_assets.ps1',
    'train_wake_word.bat', 'train_wake_word.ps1',
    'train_with_real_samples.bat',
    'eval_real_voice.bat',
    'installer_gui.bat', 'installer_gui.ps1',
    'requirements.txt', 'requirements-full-freeze.txt',
    'config', 'scripts', 'patches'
)

$NeedSeed = -not (Test-Path (Join-Path $TargetDir 'install.py'))

Write-Host ""
Write-Host "openWakeWord Windows Training Pipeline - Setup"
Write-Host ""
$confirm = Read-Host "Confirm installation into `"$TargetDir`" (Y/N)"
if ($confirm -notmatch '^[Yy]') {
    Write-Host "Cancelled."
    Read-Host "Press Enter to exit"
    exit 0
}

if ($NeedSeed) {
    if ((Test-Path (Join-Path $TemplateDir 'install.py')) -and ($TemplateDir -ne $TargetDir)) {
        Write-Host "Copying pipeline toolkit from `"$TemplateDir`" into `"$TargetDir`"..."
        foreach ($item in $TemplateItems) {
            if ($item -eq 'config') {
                # Only the generic template -- never a personal wake-word config.
                $configSrc = Join-Path $TemplateDir 'config\wake_word.example.yaml'
                if (Test-Path $configSrc) {
                    $configDest = Join-Path $TargetDir 'config'
                    New-Item -ItemType Directory -Path $configDest -Force | Out-Null
                    Copy-Item -Path $configSrc -Destination $configDest -Force
                }
                continue
            }
            $src = Join-Path $TemplateDir $item
            if (Test-Path $src) {
                Copy-Item -Path $src -Destination $TargetDir -Recurse -Force
            }
        }
    } else {
        Ensure-GitMinimal
        $RepoUrl = "https://github.com/DisasterofPuppets/Open-Wake-Word-Training.git"
        Write-Host "Cloning repository into `"$TargetDir`"..."
        git clone $RepoUrl $TargetDir
        if ($LASTEXITCODE -ne 0) { throw "git clone failed (exit $LASTEXITCODE). Check the URL in this script." }
    }
}

$RepoDir = $TargetDir

. (Join-Path $RepoDir 'scripts\InstallerCommon.ps1')
Ensure-Git
Ensure-Python

$UseGpu = $false
if (Test-NvidiaGpuPresent) {
    Write-Host "NVIDIA GPU detected."
    $UseGpu = Confirm-YesNo "Install GPU-accelerated (CUDA) PyTorch?"
} else {
    Write-Host "No NVIDIA GPU detected."
    $UseGpu = Confirm-YesNo "Install GPU-accelerated (CUDA) PyTorch anyway? (answer N for CPU-only)"
}

Push-Location $RepoDir
try {
    if ($UseGpu) { python install.py --gpu } else { python install.py --cpu }
    if ($LASTEXITCODE -ne 0) { throw "install.py failed (exit $LASTEXITCODE)." }
} finally {
    Pop-Location
}

Write-Host ""
Write-Host "============================================================"
Write-Host " Done! Installed to: $RepoDir"
Write-Host " Next: run download_assets.bat, then train_wake_word.bat"
Write-Host " (or installer_gui.bat for the all-in-one GUI) -- from inside"
Write-Host " $RepoDir, since it now has its own copy of everything."
Write-Host "============================================================"
Read-Host "Press Enter to exit"
