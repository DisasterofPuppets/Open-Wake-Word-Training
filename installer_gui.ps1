#Requires -Version 5.1
# openWakeWord Windows Training Pipeline -- unified GUI installer.
# Launch via installer_gui.bat (double-click), or:
#   powershell -NoProfile -ExecutionPolicy Bypass -File installer_gui.ps1
#
# This window stays open alongside a PowerShell console (visible behind it) --
# that console shows output from prerequisite checks (git/Python auto-install
# via winget); each tab's own log box shows that step's actual work.

$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing

$RepoDir = Split-Path -Parent $MyInvocation.MyCommand.Path
. (Join-Path $RepoDir 'scripts\InstallerCommon.ps1')

# ---------------------------------------------------------------------------
# Helpers: run a scriptblock as a background job, streaming its output into
# a log TextBox and updating a ProgressBar, via a polling Timer.
# ---------------------------------------------------------------------------

function Start-StreamedJob {
    param(
        [scriptblock]$ScriptBlock,
        [array]$JobArgs,
        [System.Windows.Forms.TextBox]$LogBox,
        [System.Windows.Forms.ProgressBar]$ProgressBar,
        [System.Windows.Forms.Label]$StatusLabel,
        [System.Windows.Forms.Button]$RunButton,
        [scriptblock]$OnDone = $null
    )

    $LogBox.Clear()
    $ProgressBar.Style = 'Marquee'
    $ProgressBar.MarqueeAnimationSpeed = 30
    $StatusLabel.Text = "Running..."
    $RunButton.Enabled = $false

    $job = Start-Job -ArgumentList $JobArgs -ScriptBlock $ScriptBlock

    $timer = New-Object System.Windows.Forms.Timer
    $timer.Interval = 300
    $state = [PSCustomObject]@{ Job = $job; LastCount = 0 }

    $timer.Add_Tick({
        $j = $state.Job
        $newOutput = Receive-Job -Job $j -Keep | Select-Object -Skip $state.LastCount
        foreach ($line in $newOutput) {
            $text = "$line"
            $LogBox.AppendText("$text`r`n")
            if ($text -match '\((\d{1,3})%\)') {
                $pct = [int]$Matches[1]
                if ($ProgressBar.Style -ne 'Continuous') { $ProgressBar.Style = 'Continuous' }
                $ProgressBar.Value = [Math]::Min([Math]::Max($pct, 0), 100)
            }
            if ($text -match '^=== (.+) ===$') {
                $StatusLabel.Text = $Matches[1]
            }
        }
        $state.LastCount += $newOutput.Count

        if ($j.State -ne 'Running') {
            $timer.Stop()
            $RunButton.Enabled = $true
            if ($j.State -eq 'Completed') {
                $ProgressBar.Style = 'Continuous'
                $ProgressBar.Value = 100
                $StatusLabel.Text = "Done."
            } else {
                $ProgressBar.Style = 'Continuous'
                $ProgressBar.Value = 0
                $StatusLabel.Text = "FAILED -- see log."
                $errs = Receive-Job -Job $j -ErrorAction SilentlyContinue 2>&1 | Out-String
                if ($errs) { $LogBox.AppendText("`r`n$errs`r`n") }
            }
            Remove-Job -Job $j -Force -ErrorAction SilentlyContinue
            if ($OnDone) { & $OnDone $j.State }
        }
    }.GetNewClosure())

    $timer.Start()
}

# ---------------------------------------------------------------------------
# Main form
# ---------------------------------------------------------------------------

$form = New-Object System.Windows.Forms.Form
$form.Text = "openWakeWord Windows Training Pipeline"
$form.Size = New-Object System.Drawing.Size(820, 680)
$form.StartPosition = 'CenterScreen'
$form.MinimumSize = New-Object System.Drawing.Size(700, 550)

# --- GPU/CPU selector (used by Tab 1) ---
$gpuGroup = New-Object System.Windows.Forms.GroupBox
$gpuGroup.Text = "PyTorch mode (used by Step 1)"
$gpuGroup.Location = New-Object System.Drawing.Point(10, 10)
$gpuGroup.Size = New-Object System.Drawing.Size(780, 50)
$gpuGroup.Anchor = 'Top,Left,Right'

$radioGpu = New-Object System.Windows.Forms.RadioButton
$radioGpu.Text = "GPU (NVIDIA CUDA)"
$radioGpu.Location = New-Object System.Drawing.Point(15, 20)
$radioGpu.AutoSize = $true

$radioCpu = New-Object System.Windows.Forms.RadioButton
$radioCpu.Text = "CPU only"
$radioCpu.Location = New-Object System.Drawing.Point(200, 20)
$radioCpu.AutoSize = $true

if (Test-NvidiaGpuPresent) {
    $radioGpu.Checked = $true
    $radioGpu.Text += "  (detected)"
} else {
    $radioCpu.Checked = $true
    $radioCpu.Text += "  (no NVIDIA GPU detected)"
}
$gpuGroup.Controls.AddRange(@($radioGpu, $radioCpu))

# --- Tabs ---
$tabs = New-Object System.Windows.Forms.TabControl
$tabs.Location = New-Object System.Drawing.Point(10, 70)
$tabs.Size = New-Object System.Drawing.Size(780, 570)
$tabs.Anchor = 'Top,Bottom,Left,Right'

$tab1 = New-Object System.Windows.Forms.TabPage; $tab1.Text = "1. Install"
$tab2 = New-Object System.Windows.Forms.TabPage; $tab2.Text = "2. Download Assets"
$tab3 = New-Object System.Windows.Forms.TabPage; $tab3.Text = "3. Train Wake Word"
$tabs.Controls.AddRange(@($tab1, $tab2, $tab3))

function New-LogBox {
    $box = New-Object System.Windows.Forms.TextBox
    $box.Multiline = $true
    $box.ReadOnly = $true
    $box.ScrollBars = 'Vertical'
    $box.Font = New-Object System.Drawing.Font("Consolas", 9)
    $box.Location = New-Object System.Drawing.Point(10, 100)
    $box.Size = New-Object System.Drawing.Size(745, 400)
    $box.Anchor = 'Top,Bottom,Left,Right'
    return $box
}

function New-ProgressBarControl {
    $pb = New-Object System.Windows.Forms.ProgressBar
    $pb.Location = New-Object System.Drawing.Point(10, 70)
    $pb.Size = New-Object System.Drawing.Size(745, 20)
    $pb.Anchor = 'Top,Left,Right'
    return $pb
}

function New-StatusLabel {
    $lbl = New-Object System.Windows.Forms.Label
    $lbl.Location = New-Object System.Drawing.Point(10, 50)
    $lbl.Size = New-Object System.Drawing.Size(745, 18)
    $lbl.Anchor = 'Top,Left,Right'
    $lbl.Text = ""
    return $lbl
}

# ---------------------------------------------------------------------------
# Tab 1: Install
# ---------------------------------------------------------------------------

$lbl1 = New-Object System.Windows.Forms.Label
$lbl1.Text = "Sets up the venv, installs Python packages, clones openWakeWord + applies the Windows patch. No large downloads."
$lbl1.Location = New-Object System.Drawing.Point(10, 10)
$lbl1.Size = New-Object System.Drawing.Size(745, 35)

$btn1 = New-Object System.Windows.Forms.Button
$btn1.Text = "Run Setup"
$btn1.Location = New-Object System.Drawing.Point(650, 10)
$btn1.Size = New-Object System.Drawing.Size(105, 30)
$btn1.Anchor = 'Top,Right'

$status1 = New-StatusLabel
$pb1 = New-ProgressBarControl
$log1 = New-LogBox

$btn1.Add_Click({
    $useGpu = $radioGpu.Checked
    Start-StreamedJob -ScriptBlock {
        param($RepoDir, $UseGpu)
        Set-Location $RepoDir
        . (Join-Path $RepoDir 'scripts\InstallerCommon.ps1')
        Ensure-Git
        Ensure-Python
        if ($UseGpu) { python install.py --gpu } else { python install.py --cpu }
        if ($LASTEXITCODE -ne 0) { throw "install.py exited with code $LASTEXITCODE" }
    } -JobArgs @($RepoDir, $useGpu) -LogBox $log1 -ProgressBar $pb1 -StatusLabel $status1 -RunButton $btn1
})

$tab1.Controls.AddRange(@($lbl1, $btn1, $status1, $pb1, $log1))

# ---------------------------------------------------------------------------
# Tab 2: Download Assets
# ---------------------------------------------------------------------------

$lbl2 = New-Object System.Windows.Forms.Label
$lbl2.Text = "Downloads voice pack (~430MB), RIRs (~10MB), and the ACAV100M feature cache (~16GB + ~180MB validation). No HF account needed; optional token below speeds up the big file. All 'existing copy' fields below are optional -- leave blank to download."
$lbl2.Location = New-Object System.Drawing.Point(10, 10)
$lbl2.Size = New-Object System.Drawing.Size(745, 50)

$panel2 = New-Object System.Windows.Forms.Panel
$panel2.Location = New-Object System.Drawing.Point(10, 60)
$panel2.Size = New-Object System.Drawing.Size(745, 145)
$panel2.Anchor = 'Top,Left,Right'

function New-PathRow {
    param([string]$LabelText, [int]$Y, [bool]$IsFolder)
    $lbl = New-Object System.Windows.Forms.Label
    $lbl.Text = $LabelText
    $lbl.Location = New-Object System.Drawing.Point(0, $Y)
    $lbl.Size = New-Object System.Drawing.Size(230, 20)

    $tb = New-Object System.Windows.Forms.TextBox
    $tb.Location = New-Object System.Drawing.Point(235, $Y - 2)
    $tb.Size = New-Object System.Drawing.Size(410, 20)
    $tb.Anchor = 'Top,Left,Right'

    $btn = New-Object System.Windows.Forms.Button
    $btn.Text = "Browse..."
    $btn.Location = New-Object System.Drawing.Point(655, $Y - 3)
    $btn.Size = New-Object System.Drawing.Size(80, 24)
    $btn.Anchor = 'Top,Right'
    $btn.Add_Click({
        if ($IsFolder) {
            $dlg = New-Object System.Windows.Forms.FolderBrowserDialog
            if ($dlg.ShowDialog() -eq 'OK') { $tb.Text = $dlg.SelectedPath }
        } else {
            $dlg = New-Object System.Windows.Forms.OpenFileDialog
            if ($dlg.ShowDialog() -eq 'OK') { $tb.Text = $dlg.FileName }
        }
    }.GetNewClosure())

    return @($lbl, $tb, $btn)
}

$voiceRow = New-PathRow "Existing voice pack folder:" 5 $true
$rirRow = New-PathRow "Existing RIR folder:" 35 $true
$acavRow = New-PathRow "Existing ACAV100M file:" 65 $false
$valRow = New-PathRow "Existing validation file:" 95 $false
$panel2.Controls.AddRange($voiceRow + $rirRow + $acavRow + $valRow)

$lblToken = New-Object System.Windows.Forms.Label
$lblToken.Text = "HF token (optional):"
$lblToken.Location = New-Object System.Drawing.Point(0, 125)
$lblToken.Size = New-Object System.Drawing.Size(230, 20)
$tbToken = New-Object System.Windows.Forms.TextBox
$tbToken.Location = New-Object System.Drawing.Point(235, 123)
$tbToken.Size = New-Object System.Drawing.Size(300, 20)
$tbToken.PasswordChar = '*'
$panel2.Controls.AddRange(@($lblToken, $tbToken))

$chkSkipFeatures = New-Object System.Windows.Forms.CheckBox
$chkSkipFeatures.Text = "Skip the ACAV100M feature cache for now (training won't work until it's added later)"
$chkSkipFeatures.Location = New-Object System.Drawing.Point(10, 210)
$chkSkipFeatures.Size = New-Object System.Drawing.Size(600, 20)
$chkSkipFeatures.Anchor = 'Top,Left'

$btn2 = New-Object System.Windows.Forms.Button
$btn2.Text = "Download"
$btn2.Location = New-Object System.Drawing.Point(650, 10)
$btn2.Size = New-Object System.Drawing.Size(105, 30)
$btn2.Anchor = 'Top,Right'

$status2 = New-Object System.Windows.Forms.Label
$status2.Location = New-Object System.Drawing.Point(10, 235)
$status2.Size = New-Object System.Drawing.Size(745, 18)
$status2.Anchor = 'Top,Left,Right'

$pb2 = New-Object System.Windows.Forms.ProgressBar
$pb2.Location = New-Object System.Drawing.Point(10, 255)
$pb2.Size = New-Object System.Drawing.Size(745, 20)
$pb2.Anchor = 'Top,Left,Right'

$log2 = New-Object System.Windows.Forms.TextBox
$log2.Multiline = $true
$log2.ReadOnly = $true
$log2.ScrollBars = 'Vertical'
$log2.Font = New-Object System.Drawing.Font("Consolas", 9)
$log2.Location = New-Object System.Drawing.Point(10, 280)
$log2.Size = New-Object System.Drawing.Size(745, 220)
$log2.Anchor = 'Top,Bottom,Left,Right'

$btn2.Add_Click({
    $pyArgs = @('--yes')
    if ($voiceRow[1].Text) { $pyArgs += @('--voice-pack-path', $voiceRow[1].Text) }
    if ($rirRow[1].Text) { $pyArgs += @('--rir-path', $rirRow[1].Text) }
    if ($acavRow[1].Text) { $pyArgs += @('--acav-path', $acavRow[1].Text) }
    if ($valRow[1].Text) { $pyArgs += @('--validation-path', $valRow[1].Text) }
    if ($tbToken.Text) { $pyArgs += @('--hf-token', $tbToken.Text) }
    if ($chkSkipFeatures.Checked) { $pyArgs += '--skip-features' }

    Start-StreamedJob -ScriptBlock {
        param($RepoDir, $PyArgs)
        & (Join-Path $RepoDir '.venv\Scripts\python.exe') (Join-Path $RepoDir 'scripts\download_assets.py') @PyArgs
        if ($LASTEXITCODE -ne 0) { throw "download_assets.py exited with code $LASTEXITCODE" }
    } -JobArgs @($RepoDir, $pyArgs) -LogBox $log2 -ProgressBar $pb2 -StatusLabel $status2 -RunButton $btn2
})

$tab2.Controls.AddRange(@($lbl2, $panel2, $chkSkipFeatures, $btn2, $status2, $pb2, $log2))

# ---------------------------------------------------------------------------
# Tab 3: Train Wake Word
# ---------------------------------------------------------------------------

$panel3 = New-Object System.Windows.Forms.Panel
$panel3.Location = New-Object System.Drawing.Point(10, 10)
$panel3.Size = New-Object System.Drawing.Size(745, 175)
$panel3.Anchor = 'Top,Left,Right'

function New-FieldRow {
    param([string]$LabelText, [int]$Y, [System.Windows.Forms.Control]$Control)
    $lbl = New-Object System.Windows.Forms.Label
    $lbl.Text = $LabelText
    $lbl.Location = New-Object System.Drawing.Point(0, $Y)
    $lbl.Size = New-Object System.Drawing.Size(150, 20)
    $Control.Location = New-Object System.Drawing.Point(155, $Y - 2)
    return @($lbl, $Control)
}

$tbName = New-Object System.Windows.Forms.TextBox; $tbName.Size = New-Object System.Drawing.Size(250, 20)
$tbPhrase = New-Object System.Windows.Forms.TextBox; $tbPhrase.Size = New-Object System.Drawing.Size(250, 20)
$tbDisplay = New-Object System.Windows.Forms.TextBox; $tbDisplay.Size = New-Object System.Drawing.Size(250, 20)
$numSamples = New-Object System.Windows.Forms.NumericUpDown
$numSamples.Minimum = 100; $numSamples.Maximum = 200000; $numSamples.Value = 10000
$numSamples.Size = New-Object System.Drawing.Size(100, 20)
$numCutoff = New-Object System.Windows.Forms.NumericUpDown
$numCutoff.DecimalPlaces = 2; $numCutoff.Increment = 0.05; $numCutoff.Minimum = 0; $numCutoff.Maximum = 1; $numCutoff.Value = 0.7
$numCutoff.Size = New-Object System.Drawing.Size(100, 20)

$panel3.Controls.AddRange((New-FieldRow "Model name:" 5 $tbName))
$panel3.Controls.AddRange((New-FieldRow "Wake phrase:" 35 $tbPhrase))
$panel3.Controls.AddRange((New-FieldRow "Display name (for HA):" 65 $tbDisplay))
$panel3.Controls.AddRange((New-FieldRow "Sample count:" 95 $numSamples))
$panel3.Controls.AddRange((New-FieldRow "Detection cutoff:" 125 $numCutoff))

$lblTest = New-Object System.Windows.Forms.Label
$lblTest.Text = "Real-voice test recordings (optional):"
$lblTest.Location = New-Object System.Drawing.Point(10, 190)
$lblTest.Size = New-Object System.Drawing.Size(300, 20)

$listTest = New-Object System.Windows.Forms.ListBox
$listTest.Location = New-Object System.Drawing.Point(10, 210)
$listTest.Size = New-Object System.Drawing.Size(560, 60)
$listTest.Anchor = 'Top,Left,Right'

$btnAddTest = New-Object System.Windows.Forms.Button
$btnAddTest.Text = "Add..."
$btnAddTest.Location = New-Object System.Drawing.Point(580, 210)
$btnAddTest.Size = New-Object System.Drawing.Size(80, 26)
$btnAddTest.Anchor = 'Top,Right'
$btnAddTest.Add_Click({
    $dlg = New-Object System.Windows.Forms.OpenFileDialog
    $dlg.Filter = "WAV files (*.wav)|*.wav|All files (*.*)|*.*"
    $dlg.Multiselect = $true
    if ($dlg.ShowDialog() -eq 'OK') {
        foreach ($f in $dlg.FileNames) { [void]$listTest.Items.Add($f) }
    }
})

$btnRemoveTest = New-Object System.Windows.Forms.Button
$btnRemoveTest.Text = "Remove"
$btnRemoveTest.Location = New-Object System.Drawing.Point(580, 240)
$btnRemoveTest.Size = New-Object System.Drawing.Size(80, 26)
$btnRemoveTest.Anchor = 'Top,Right'
$btnRemoveTest.Add_Click({
    @($listTest.SelectedItems) | ForEach-Object { $listTest.Items.Remove($_) }
})

$btn3 = New-Object System.Windows.Forms.Button
$btn3.Text = "Train"
$btn3.Location = New-Object System.Drawing.Point(650, 285)
$btn3.Size = New-Object System.Drawing.Size(105, 30)
$btn3.Anchor = 'Top,Right'

$status3 = New-Object System.Windows.Forms.Label
$status3.Location = New-Object System.Drawing.Point(10, 320)
$status3.Size = New-Object System.Drawing.Size(745, 18)
$status3.Anchor = 'Top,Left,Right'

$pb3 = New-Object System.Windows.Forms.ProgressBar
$pb3.Location = New-Object System.Drawing.Point(10, 340)
$pb3.Size = New-Object System.Drawing.Size(745, 20)
$pb3.Anchor = 'Top,Left,Right'

$log3 = New-Object System.Windows.Forms.TextBox
$log3.Multiline = $true
$log3.ReadOnly = $true
$log3.ScrollBars = 'Vertical'
$log3.Font = New-Object System.Drawing.Font("Consolas", 9)
$log3.Location = New-Object System.Drawing.Point(10, 365)
$log3.Size = New-Object System.Drawing.Size(745, 135)
$log3.Anchor = 'Top,Bottom,Left,Right'

$btn3.Add_Click({
    if (-not $tbName.Text -or -not $tbPhrase.Text) {
        [System.Windows.Forms.MessageBox]::Show("Model name and wake phrase are required.", "Missing info") | Out-Null
        return
    }
    $pyArgs = @(
        '--name', $tbName.Text,
        '--phrase', $tbPhrase.Text,
        '--samples', $numSamples.Value.ToString([System.Globalization.CultureInfo]::InvariantCulture),
        '--cutoff', $numCutoff.Value.ToString([System.Globalization.CultureInfo]::InvariantCulture)
    )
    if ($tbDisplay.Text) { $pyArgs += @('--display-name', $tbDisplay.Text) }
    if ($listTest.Items.Count -gt 0) {
        foreach ($f in $listTest.Items) { $pyArgs += @('--test-wav', $f) }
    } else {
        $pyArgs += '--skip-eval'
    }

    Start-StreamedJob -ScriptBlock {
        param($RepoDir, $PyArgs)
        & (Join-Path $RepoDir '.venv\Scripts\python.exe') (Join-Path $RepoDir 'scripts\train_wake_word.py') @PyArgs
        if ($LASTEXITCODE -ne 0) { throw "train_wake_word.py exited with code $LASTEXITCODE" }
    } -JobArgs @($RepoDir, $pyArgs) -LogBox $log3 -ProgressBar $pb3 -StatusLabel $status3 -RunButton $btn3
})

$tab3.Controls.AddRange(@($panel3, $lblTest, $listTest, $btnAddTest, $btnRemoveTest, $btn3, $status3, $pb3, $log3))

# ---------------------------------------------------------------------------

$form.Controls.AddRange(@($gpuGroup, $tabs))
[void]$form.ShowDialog()
