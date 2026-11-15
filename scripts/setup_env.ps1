# Windows PowerShell: create .venv and install the requirements.
# Usage (from the seed_project folder):  powershell -ExecutionPolicy Bypass -File scripts\setup_env.ps1
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
$usePy = [bool](Get-Command py -ErrorAction SilentlyContinue)
function Invoke-Python {
    if ($usePy) { & py -3 @args } else { & python @args }
    if ($LASTEXITCODE -ne 0) { throw "Python command failed: $args" }
}
Invoke-Python -c "import sys; assert sys.version_info >= (3, 10), 'Python 3.10 or newer is required'"
Invoke-Python -m venv .venv
& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw "Installing the requirements failed" }
Write-Host ""
Write-Host "Done. Run:   .venv\Scripts\python scripts\run_count.py"
