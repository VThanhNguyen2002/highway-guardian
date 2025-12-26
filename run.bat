@echo off
title Highway Guardian Launcher
cls
echo ========================================================
echo   HIGHWAY GUARDIAN - PROJECT LAUNCHER
echo ========================================================
echo.
echo 1. Run with Docker (RECOMMENDED)
echo 2. Run Locally (Backend + Frontend)
echo 3. Start Backend Only (Local)
echo 4. Start Frontend Only (Local)
echo 5. Setup Environment (Local/Dev)
echo 6. Exit
echo.
set /p choice="Enter your choice (1-6): "

if "%choice%"=="1" goto run_docker
if "%choice%"=="2" goto start_all_local
if "%choice%"=="3" goto start_backend
if "%choice%"=="4" goto start_frontend
if "%choice%"=="5" goto setup_env
if "%choice%"=="6" goto exit

:run_docker
echo Starting with Docker...
docker-compose up --build
goto exit

:start_all_local
echo Starting Backend and Frontend locally...
start cmd /k "cd scripts && start_backend.bat"
cd frontend
npm run dev
goto exit

:start_backend
echo Starting Backend...
cd scripts
call start_backend.bat
goto exit

:start_frontend
echo Starting Frontend...
cd frontend
npm run dev
goto exit

:setup_env
echo Setting up environment...
python scripts/setup_environment.py
pause
goto exit

:exit
echo Goodbye!
exit