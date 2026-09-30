# One-command launch (Windows PowerShell):  .\start.ps1
# First run installs dependencies and builds the dashboard; later runs start immediately.
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

python -c "import fastapi, uvicorn, numpy, scipy, psutil, multipart" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Installing Python dependencies..." -ForegroundColor Cyan
    python -m pip install -r requirements.txt
}

if (-not (Test-Path "frontend\dist\index.html")) {
    Write-Host "Building the dashboard..." -ForegroundColor Cyan
    Push-Location frontend
    if (-not (Test-Path "node_modules")) { npm install --no-audit --no-fund }
    npm run build
    Pop-Location
}

Write-Host "Dashboard: http://127.0.0.1:8053" -ForegroundColor Green
python -m backend --port 8053 --open
