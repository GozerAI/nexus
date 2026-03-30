@echo off
:: Sentinel Intelligence - First Time Setup & Run
:: This script installs dependencies and launches the application

echo ============================================
echo  Sentinel Intelligence Platform Setup
echo ============================================
echo.

cd /d "%~dp0.."

:: Check for Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://python.org
    pause
    exit /b 1
)

:: Create virtual environment if it doesn't exist
if not exist ".venv" (
    echo Creating virtual environment...
    python -m venv .venv
)

:: Activate virtual environment
call .venv\Scripts\activate.bat

:: Install/upgrade pip
echo.
echo Upgrading pip...
python -m pip install --upgrade pip

:: Install the package with full dependencies (GUI + embeddings)
echo.
echo Installing Sentinel Intelligence with all dependencies...
echo This may take a few minutes on first run...
echo.
pip install -e ".[full]"

if errorlevel 1 (
    echo.
    echo WARNING: Full install failed. Trying GUI-only install...
    pip install -e ".[gui]"
)

:: Create desktop shortcut
echo.
echo Creating desktop shortcut...
powershell -ExecutionPolicy Bypass -File "%~dp0create_shortcut.ps1"

:: Launch the application
echo.
echo ============================================
echo  Starting Sentinel Intelligence...
echo ============================================
echo.
python -m sentinel.gui

pause
