@echo off
:: Nexus Intelligence Platform Launcher
:: Double-click this file or create a shortcut to it

cd /d "%~dp0.."

:: Check if virtual environment exists
if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
) else if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
)

:: Launch the GUI
python -m nexus.gui

pause
