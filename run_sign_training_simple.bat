@echo off
echo ========================================
echo 🚦 Highway Guardian - Sign Training
echo ========================================
echo.

:: Kiểm tra Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python not found! Please install Python first.
    pause
    exit /b 1
)

echo ✅ Python found
echo.

:: Chuyển đến thư mục src
cd /d "%~dp0src"
echo 📁 Current directory: %CD%
echo.

:: Cài đặt dependencies
echo 📦 Installing dependencies...
pip install ultralytics torch torchvision pyyaml matplotlib --quiet
if errorlevel 1 (
    echo ⚠️ Some packages may already be installed
)
echo ✅ Dependencies ready
echo.

:: Chạy training
echo 🚀 Starting sign detection training...
echo ⏰ This may take 30-60 minutes depending on your hardware
echo.
python train_sign_simple.py

echo.
echo ========================================
echo 🏁 Training completed!
echo ========================================
echo.
echo 📁 Check results in: runs/detect/sign_simple/
echo 🎯 Best model: runs/detect/sign_simple/weights/best.pt
echo.
pause