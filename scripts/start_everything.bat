@echo off
setlocal

echo ========================================================
echo   HIGHWAY GUARDIAN - AUTO STARTER
echo ========================================================

:: 1. Check if Docker is running
docker info >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Docker is not running!
    echo Please start Docker Desktop and try again.
    pause
    exit /b 1
)

echo [OK] Docker is running.

:: 2. Start containers (Build & Up)
echo.
echo [INFO] Starting services... This may take a while for the first time.
echo --------------------------------------------------------
docker-compose up -d --build

if %errorlevel% neq 0 (
    echo [ERROR] Failed to start containers.
    pause
    exit /b 1
)

echo.
echo [SUCCESS] Services are up and running!

:: 3. Open Frontend
echo [INFO] Opening Gateway...
timeout /t 5 >nul
start http://localhost:8080

echo.
echo ========================================================
echo   SYSTEM READY
echo   Backend: http://localhost:8000/docs
echo   Frontend: http://localhost:8080
echo ========================================================
echo Press any key to stop services and exit...
pause >nul

:: Stop services on exit
echo.
echo [INFO] Stopping services...
docker-compose down
echo [INFO] Goodbye!
