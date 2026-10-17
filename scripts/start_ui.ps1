$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path $PSScriptRoot -Parent
Set-Location -LiteralPath $projectRoot
$env:RF_SEARCHLAB_HOME = $projectRoot
if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) { throw 'Run scripts/setup.ps1 first.' }
& '.\.venv\Scripts\python.exe' -m rf_searchlab.cli ui
exit $LASTEXITCODE

