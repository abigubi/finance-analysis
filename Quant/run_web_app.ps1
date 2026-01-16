# PowerShell script to run the Streamlit web app
Write-Host "Starting Falling Knife Detector Web App..." -ForegroundColor Green
Write-Host ""
Write-Host "The app will open in your browser at http://localhost:8501" -ForegroundColor Yellow
Write-Host ""
Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Yellow
Write-Host ""

# Change to script directory
Set-Location $PSScriptRoot

# Run Streamlit
python -m streamlit run Falling_Knife_Web.py
