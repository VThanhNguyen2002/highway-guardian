@echo off
echo ========================================
echo   Highway Guardian - Docker Startup
echo ========================================
echo.

echo Building and starting containers...
docker-compose up --build -d

echo.
echo ========================================
echo   Containers started successfully!
echo ========================================
echo.
echo Backend:  http://localhost:8000
echo Frontend: http://localhost:8080
echo.
echo To view logs: docker-compose logs -f
echo To stop:      docker-compose down
echo.

pause
