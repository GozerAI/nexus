@echo off
:: Nexus Intelligence - Full Startup (Redis + GUI)
:: This starts Redis in the background then launches the GUI

cd /d "%~dp0.."

echo ============================================
echo  Nexus Intelligence Platform
echo ============================================
echo.

:: Check for Redis
if exist "tools\redis\redis-server.exe" (
    echo Checking Redis status...

    :: Check if Redis is already running
    netstat -an | findstr ":6379" | findstr "LISTENING" >nul 2>&1
    if errorlevel 1 (
        echo Starting Redis in background...
        start /B "" "tools\redis\redis-server.exe" "tools\redis\redis-nexus.conf"
        timeout /t 2 /nobreak >nul
        echo Redis started.
    ) else (
        echo Redis is already running.
    )
) else (
    echo Redis not installed - using SQLite for all storage.
    echo To install Redis: powershell -ExecutionPolicy Bypass -File scripts\setup_redis.ps1
)

echo.
echo Starting Nexus Intelligence GUI...
echo.

:: Activate venv and run
if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
)

python -m nexus.gui
