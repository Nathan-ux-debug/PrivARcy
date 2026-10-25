# One-time setup on Windows (PowerShell):  .\scripts\setup.ps1
$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)
if (-not (Test-Path .venv)) { python -m venv .venv }
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
Write-Host "Done. Put best.pt in .\models\ then run .\scripts\run.ps1"
Write-Host "Optional sensitive-text (OCR):  .\.venv\Scripts\python.exe -m pip install -e '.[ocr]'"
