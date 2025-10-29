@echo off
echo.
echo ========================================
echo   PHASE 15: LAUNCHING STREAMLIT
echo ========================================
echo.
echo Starting Gravitational Lensing Analysis Platform...
echo.
echo New Pages Available:
echo   - Scientific Validation (publication-ready)
echo   - Bayesian UQ (Monte Carlo Dropout)
echo.
echo Server will start on: http://localhost:8501
echo.
echo Press Ctrl+C to stop the server
echo.

cd /d "%~dp0.."
call .venv\Scripts\activate.bat
python -m streamlit run app/main.py

pause
