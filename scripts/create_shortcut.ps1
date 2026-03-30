# Create Desktop Shortcut for Nexus Intelligence
# Run this script once: Right-click -> Run with PowerShell

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$LauncherPath = Join-Path $ProjectRoot "scripts\launch_nexus.pyw"
$DesktopPath = [Environment]::GetFolderPath("Desktop")
$ShortcutPath = Join-Path $DesktopPath "Nexus Intelligence.lnk"

# Find pythonw.exe (windowless Python)
$PythonW = $null
$VenvPythonW = Join-Path $ProjectRoot ".venv\Scripts\pythonw.exe"
$VenvPythonW2 = Join-Path $ProjectRoot "venv\Scripts\pythonw.exe"

if (Test-Path $VenvPythonW) {
    $PythonW = $VenvPythonW
} elseif (Test-Path $VenvPythonW2) {
    $PythonW = $VenvPythonW2
} else {
    # Find system pythonw
    $PythonW = (Get-Command pythonw -ErrorAction SilentlyContinue).Source
    if (-not $PythonW) {
        $PythonW = (Get-Command python -ErrorAction SilentlyContinue).Source
        if ($PythonW) {
            $PythonW = $PythonW -replace "python.exe$", "pythonw.exe"
        }
    }
}

if (-not $PythonW -or -not (Test-Path $PythonW)) {
    Write-Host "ERROR: Could not find pythonw.exe" -ForegroundColor Red
    Write-Host "Please ensure Python is installed and in your PATH"
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host "Creating desktop shortcut..." -ForegroundColor Cyan
Write-Host "  Python: $PythonW"
Write-Host "  Launcher: $LauncherPath"
Write-Host "  Shortcut: $ShortcutPath"

# Create the shortcut
$WshShell = New-Object -ComObject WScript.Shell
$Shortcut = $WshShell.CreateShortcut($ShortcutPath)
$Shortcut.TargetPath = $PythonW
$Shortcut.Arguments = "`"$LauncherPath`""
$Shortcut.WorkingDirectory = $ProjectRoot
$Shortcut.Description = "Nexus Intelligence Platform"
$Shortcut.WindowStyle = 1  # Normal window

# Try to set an icon (use Python icon as fallback)
$IconPath = Join-Path $ProjectRoot "assets\nexus.ico"
if (Test-Path $IconPath) {
    $Shortcut.IconLocation = $IconPath
} else {
    # Use Python's icon
    $PythonDir = Split-Path -Parent $PythonW
    $PythonIcon = Join-Path (Split-Path -Parent $PythonDir) "Lib\test\imghdrdata\python.ico"
    if (-not (Test-Path $PythonIcon)) {
        $PythonIcon = "$PythonW,0"
    }
    $Shortcut.IconLocation = $PythonIcon
}

$Shortcut.Save()

Write-Host ""
Write-Host "Desktop shortcut created successfully!" -ForegroundColor Green
Write-Host "You can now double-click 'Nexus Intelligence' on your desktop."
Write-Host ""

Write-Host ""
Read-Host "Press Enter to exit"
