@echo off
:: Start Redis Stack for Sentinel Intelligence

cd /d "%~dp0..\tools\redis-stack"

if not exist "bin\redis-stack-server.exe" (
    echo Redis Stack not installed. Run setup_redis.ps1 first.
    echo.
    echo   powershell -ExecutionPolicy Bypass -File "%~dp0setup_redis.ps1"
    echo.
    pause
    exit /b 1
)

echo Starting Redis Stack...
echo Data stored in: %~dp0..\data\redis
echo.
echo Press Ctrl+C to stop Redis
echo.

bin\redis-stack-server.exe redis-sentinel.conf
