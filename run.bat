@echo off
title Highway Guardian Orchestrator
color 0A

:: =============================================================
:: Highway Guardian — One-Click Startup (Windows)
:: Starts FastAPI backend + Vue dashboard concurrently.
:: =============================================================

:: 1. Resolve project root
set "PROJECT_ROOT=%~dp0"
if "%PROJECT_ROOT:~-1%"=="\" set "PROJECT_ROOT=%PROJECT_ROOT:~0,-1%"

echo.
echo =========================================================
echo         HIGHWAY GUARDIAN — SYSTEM STARTUP
echo =========================================================
echo.

:: 2. Pre-flight: .env
if not exist "%PROJECT_ROOT%\.env" (
    echo [WARN] .env not found.
    if exist "%PROJECT_ROOT%\.env.example" (
        echo       Copying .env.example → .env. Fill in your keys!
        copy "%PROJECT_ROOT%\.env.example" "%PROJECT_ROOT%\.env" >nul
    ) else (
        echo [WARN] No .env.example either. Backend may fail to start.
    )
)

:: 3. Pre-flight: uploads dir
if not exist "%PROJECT_ROOT%\uploads" mkdir "%PROJECT_ROOT%\uploads"

:: 4. Pre-flight: frontend node_modules
if not exist "%PROJECT_ROOT%\frontend\node_modules" (
    echo [INFO] node_modules not found — installing Vue dependencies...
    cd /d "%PROJECT_ROOT%\frontend"
    call npm install
    cd /d "%PROJECT_ROOT%"
)

:: 5. Python path
set "PYTHONPATH=%PROJECT_ROOT%"

:: 6. Auto-detect virtualenv (supports .venv, venv, env)
set "ACTIVATE_CMD="
if exist "%PROJECT_ROOT%\.venv\Scripts\activate.bat" (
    set "ACTIVATE_CMD=call "%PROJECT_ROOT%\.venv\Scripts\activate.bat" && "
) else if exist "%PROJECT_ROOT%\venv\Scripts\activate.bat" (
    set "ACTIVATE_CMD=call "%PROJECT_ROOT%\venv\Scripts\activate.bat" && "
) else if exist "%PROJECT_ROOT%\env\Scripts\activate.bat" (
    set "ACTIVATE_CMD=call "%PROJECT_ROOT%\env\Scripts\activate.bat" && "
)

echo [INFO] Starting services...
echo.

:: 7. FastAPI backend (new window)
start "Highway Guardian — Backend (FastAPI :8000)" cmd /k "cd /d "%PROJECT_ROOT%" && %ACTIVATE_CMD% python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload"

:: 8. Vue dashboard (new window)
start "Highway Guardian — Vue Dashboard (:5173)" cmd /k "cd /d "%PROJECT_ROOT%\frontend" && npm run dev"

:: 9. (Optional) Streamlit inference UI — uncomment to enable
:: start "Highway Guardian — Streamlit UI (:8501)" cmd /k "cd /d "%PROJECT_ROOT%\streamlit_app" && %ACTIVATE_CMD% streamlit run app.py --server.port 8501"

:: 10. Summary
cls
color 0B
echo =========================================================
echo          HIGHWAY GUARDIAN — SYSTEM RUNNING
echo =========================================================
echo.
echo   API Docs      : http://localhost:8000/docs
echo   Vue Dashboard : http://localhost:5173
echo   Streamlit UI  : http://localhost:8501 (if enabled)
echo.
echo =========================================================
echo   Services launched in separate windows.
echo   Close those windows (or press Ctrl+C in each) to stop.
echo   Press any key to close this orchestrator window...
echo =========================================================
pause >nul