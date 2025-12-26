@echo off
echo ========================================
echo Highway Guardian - Sign Detection Training
echo ========================================
echo.

REM Activate virtual environment if exists
if exist "venv\Scripts\activate.bat" (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
)

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found! Please install Python first.
    pause
    exit /b 1
)

REM Install required packages if needed
echo Checking dependencies...
pip install ultralytics torch torchvision pyyaml

REM Change to project directory
cd /d "%~dp0"

REM Run training script
echo.
echo Starting sign detection training...
echo.
python src\train_sign_detection_local.py

echo.
echo Training completed! Check the logs for results.
pause