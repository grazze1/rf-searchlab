$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path $PSScriptRoot -Parent
Set-Location -LiteralPath $projectRoot
$env:RF_SEARCHLAB_HOME = $projectRoot
& '.\.venv\Scripts\python.exe' -m rf_searchlab.cli run --environment omni --seed 101 --count 10
if ($LASTEXITCODE -ne 0) { throw 'Omnidirectional demo failed' }
& '.\.venv\Scripts\python.exe' -m rf_searchlab.cli run --environment mixed --seed 101 --count 10
if ($LASTEXITCODE -ne 0) { throw 'Mixed-source demo failed' }

