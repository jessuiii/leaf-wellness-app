<#
Start the backend (uvicorn) using the project's .venv python.

Usage (PowerShell):
  .\start-backend.ps1

This script will:
- check for a .venv python at .\.venv\Scripts\python.exe
- if found, run: python -m uvicorn backend.main:app --reload --port 5000 --app-dir <project>
#>
param()

$proj = Split-Path -Parent $MyInvocation.MyCommand.Definition
$python = Join-Path $proj '.venv\Scripts\python.exe'
if (-not (Test-Path $python)) {
    Write-Error "Virtual environment python not found: $python"
    Write-Output "Create the venv first: python -m venv .venv ; .\\.venv\\Scripts\\Activate.ps1 ; pip install -r requirements.txt (or pip install fastapi uvicorn pillow numpy tensorflow)"
    exit 1
}

Write-Output "Starting backend with: $python -m uvicorn backend.main:app --reload --port 5000 --app-dir $proj"
& $python -m uvicorn backend.main:app --reload --port 5000 --app-dir $proj
