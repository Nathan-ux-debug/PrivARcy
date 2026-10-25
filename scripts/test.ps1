# Run all tests:  .\scripts\test.ps1   (add -k name to filter)
Set-Location (Split-Path $PSScriptRoot -Parent)
$env:QT_QPA_PLATFORM = "offscreen"
.\.venv\Scripts\python.exe -m pytest @args
