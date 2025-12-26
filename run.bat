@echo off
title Highway Guardian Launcher
cls
echo ========================================================
echo   HIGHWAY GUARDIAN - PROJECT LAUNCHER
echo ========================================================
echo.
echo 1. Start ALL (Backend + Frontend)
echo 2. Start Backend Only
echo 3. Start Frontend Only
echo 4. Setup Environment (Install All Dependencies)
echo 5. Exit
echo.
set /p choice="Enter your choice (1-5): "

if "%choice%"=="1" goto start_all
if "%choice%"=="2" goto start_backend
if "%choice%"=="3" goto start_frontend
if "%choice%"=="4" goto setup_env
if "%choice%"=="5" goto exit

:start_all
echo Starting Backend and Frontend...
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
