@echo off
REM Quick Setup Script for Highway Guardian (Windows)
REM One-line setup for Windows users

echo ========================================
echo Highway Guardian - Quick Setup
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://python.org
    pause
    exit /b 1
)

echo Python detected: 
python --version
echo.

REM Ask for installation mode
echo Select installation mode:
echo 1. Basic (CPU only)
echo 2. Full (GPU support + all features)
echo 3. GPU only (just GPU dependencies)
echo.
set /p choice="Enter choice (1-3): "

if "%choice%"=="1" (
    set mode=basic
) else if "%choice%"=="2" (
    set mode=full
) else if "%choice%"=="3" (
    set mode=gpu
) else (
    echo Invalid choice, using full mode
    set mode=full
)

echo.
echo Installing in %mode% mode...
echo.

REM Run the Python setup script
python setup_environment.py --mode %mode%

if errorlevel 1 (
    echo.
    echo Setup failed! Check the error messages above.
    pause
    exit /b 1
)

echo.
echo ========================================
echo Setup completed successfully!
echo ========================================
echo.
echo You can now:
echo 1. Train car detection: python src/training/scripts/train_car_detection.py --config src/configs/car_detection_config.yaml
echo 2. Train improved sign detection: python src/training/scripts/train_sign_detection_improved.py --config src/configs/sign_det_improved.yaml
echo 3. Setup Docker: docker-compose up --build
echo.
pause