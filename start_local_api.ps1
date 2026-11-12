$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

Write-Host "Starting local clickbait inference API at http://127.0.0.1:8000"
Write-Host "Keep this window open while using the website. Press Ctrl+C to stop."
python -m uvicorn local_api:app --host 127.0.0.1 --port 8000 --log-level info
