$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path $PSScriptRoot -Parent
Set-Location -LiteralPath $projectRoot
& '.\.venv\Scripts\python.exe' scripts/check.py
exit $LASTEXITCODE

