@echo off
:: Nexus Intelligence Platform Launcher (Debug Mode)
:: Shows console output for troubleshooting

cd /d "J:\dev\nexus"

echo ============================================
echo  Nexus Intelligence Platform - Debug Launch
echo ============================================
echo.

:: Activate virtual environment
if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
    echo Virtual environment activated.
) else (
    echo WARNING: Virtual environment not found!
)

echo.
echo Starting Nexus GUI...
echo.

:: Run with console output visible
python -m nexus.gui

echo.
echo ============================================
echo  Nexus has exited.
echo ============================================
pause
