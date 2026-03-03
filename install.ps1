# XAI Financial Services - One-time install script
# Run from project root: .\install.ps1
# Requires: Python 3.10+, Node.js 18+ in PATH

$ErrorActionPreference = "Stop"
$ProjectRoot = $PSScriptRoot
Set-Location $ProjectRoot

Write-Host "=== 1. Creating Python venv ===" -ForegroundColor Cyan
if (-not (Test-Path "venv")) {
    python -m venv venv
    Write-Host "venv created." -ForegroundColor Green
} else {
    Write-Host "venv already exists." -ForegroundColor Yellow
}

Write-Host "`n=== 2. Activating venv and installing Python deps ===" -ForegroundColor Cyan
& "$ProjectRoot\venv\Scripts\pip.exe" install -r training\requirements.txt
& "$ProjectRoot\venv\Scripts\pip.exe" install -r backend\requirements.txt
Write-Host "Python packages installed." -ForegroundColor Green

Write-Host "`n=== 3. Installing frontend (npm) ===" -ForegroundColor Cyan
Set-Location frontend
npm install
Set-Location $ProjectRoot
Write-Host "Frontend packages installed." -ForegroundColor Green

Write-Host "`n=== Done. ===" -ForegroundColor Green
Write-Host "Next: activate venv, run training (see RUN.md), then start backend and frontend."
