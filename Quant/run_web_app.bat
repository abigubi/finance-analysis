@echo off
cd /d "%~dp0"
echo Starting Falling Knife Detector Web App...
echo.
echo The app will open in your browser at http://localhost:8501
echo.
echo Press Ctrl+C to stop the server
echo.
python -m streamlit run Falling_Knife_Web.py
pause
