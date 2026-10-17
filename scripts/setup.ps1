param([string]$Python = 'python')
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path $PSScriptRoot -Parent
Set-Location -LiteralPath $projectRoot
New-Item -ItemType Directory -Force -Path '.work' | Out-Null
$env:PIP_CACHE_DIR = Join-Path $projectRoot '.work\pip-cache'
$env:TEMP = Join-Path $projectRoot '.work'
$env:TMP = $env:TEMP
& $Python -c 'import sys; assert sys.version_info >= (3,12), "Python 3.12+ required"'
if ($LASTEXITCODE -ne 0) { throw 'Pass -Python with a Python 3.12+ executable.' }
if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) {
    & $Python -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'Environment creation failed' }
}
& '.\.venv\Scripts\python.exe' -m pip install -r requirements-lock.txt
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed' }
& '.\.venv\Scripts\python.exe' -m pip install --no-build-isolation -e .
if ($LASTEXITCODE -ne 0) { throw 'Project installation failed' }
& '.\.venv\Scripts\python.exe' -m rf_searchlab.cli doctor
if ($LASTEXITCODE -ne 0) { throw 'Workspace check failed' }

