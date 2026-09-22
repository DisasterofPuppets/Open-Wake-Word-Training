Get-ChildItem "Track 1_003_merged_*.wav" | ForEach-Object {
    if ($_.BaseName -match 'merged_(\d+)$') {
        Rename-Item $_.FullName -NewName "Hey_Holly$($matches[1]).wav"
    }
}