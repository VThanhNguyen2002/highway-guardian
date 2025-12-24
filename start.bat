@echo off
echo.
echo [Highway Guardian] Cleaning up unused Docker objects to save space...
echo.
docker system prune -f

echo.
echo [Highway Guardian] Starting the application... (This may take a moment)
echo.
docker-compose up --build
