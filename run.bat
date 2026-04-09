@echo off
title Highway Guardian Orchestrator
color 0A

:: 1. Xác định thư mục gốc
set "PROJECT_ROOT=%~dp0"
if "%PROJECT_ROOT:~-1%"=="\" set "PROJECT_ROOT=%PROJECT_ROOT:~0,-1%"

echo =========================================================
echo       SYSTEM PRE-FLIGHT CHECKS...
echo =========================================================

:: 2. Kiểm tra file cấu hình .env (Cực kỳ quan trọng)
if not exist "%PROJECT_ROOT%\.env" (
    echo [WARNING] .env file not found! 
    if exist "%PROJECT_ROOT%\.env.example" (
        echo Creating .env from .env.example...
        copy "%PROJECT_ROOT%\.env.example" "%PROJECT_ROOT%\.env" >nul
    ) else (
        echo [ERROR] Cannot find .env or .env.example. System might crash!
    )
)

:: 3. Kiểm tra và tạo thư mục uploads
if not exist "%PROJECT_ROOT%\uploads" (
    echo Creating uploads directory...
    mkdir "%PROJECT_ROOT%\uploads"
)

:: 4. Kiểm tra thư viện Frontend
if not exist "%PROJECT_ROOT%\frontend\node_modules" (
    echo [WARNING] node_modules not found! Installing Vue dependencies...
    cd /d "%PROJECT_ROOT%\frontend"
    call npm install
)

:: 5. Setup Python Path
set "PYTHONPATH=%PROJECT_ROOT%"

:: =========================================================
:: TÌM LỆNH KÍCH HOẠT VENV (Tự động phát hiện venv hoặc env)
:: =========================================================
set "ACTIVATE_CMD="
if exist "%PROJECT_ROOT%\venv\Scripts\activate.bat" (
    set "ACTIVATE_CMD=call "%PROJECT_ROOT%\venv\Scripts\activate.bat" && "
) else if exist "%PROJECT_ROOT%\env\Scripts\activate.bat" (
    set "ACTIVATE_CMD=call "%PROJECT_ROOT%\env\Scripts\activate.bat" && "
)

echo.
echo Starting Services...
echo.

:: 6. Khởi chạy 3 luồng (Bọc venv vào từng terminal)
start "Backend (FastAPI)" cmd /k "cd /d "%PROJECT_ROOT%" && %ACTIVATE_CMD% uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload"

start "Streamlit UI" cmd /k "cd /d "%PROJECT_ROOT%\streamlit_app" && %ACTIVATE_CMD% streamlit run app.py --server.port 8501"

start "Vue Dashboard" cmd /k "cd /d "%PROJECT_ROOT%\frontend" && npm run dev"


:: 7. Hiển thị thông tin
cls
color 0B
echo =========================================================
echo             HIGHWAY GUARDIAN SYSTEM RUNNING              
echo =========================================================
echo.
echo  API Docs        : http://localhost:8000/docs
echo  Streamlit UI    : http://localhost:8501
echo  Vue Dashboard   : http://localhost:5173
echo.
echo =========================================================
echo All services launched in separate windows.
echo Keep those windows open to view logs or debug.
echo Press any key to exit this orchestrator window...
pause >nul