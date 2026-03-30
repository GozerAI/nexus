# Update existing Nexus Intelligence desktop shortcut
$ShortcutPath = Join-Path ([Environment]::GetFolderPath("Desktop")) "Nexus Intelligence.lnk"

if (-not (Test-Path $ShortcutPath)) {
    Write-Host "Shortcut not found at: $ShortcutPath" -ForegroundColor Red
    exit 1
}

$ProjectRoot = "J:\dev\nexus"
$LauncherPath = Join-Path $ProjectRoot "scripts\launch_nexus.pyw"

# Find pythonw.exe
$PythonW = Join-Path $ProjectRoot ".venv\Scripts\pythonw.exe"
if (-not (Test-Path $PythonW)) {
    $PythonW = Join-Path $ProjectRoot "venv\Scripts\pythonw.exe"
}
if (-not (Test-Path $PythonW)) {
    $PythonW = (Get-Command pythonw -ErrorAction SilentlyContinue).Source
}

Write-Host "Updating shortcut..." -ForegroundColor Cyan
Write-Host "  Shortcut: $ShortcutPath"
Write-Host "  Python: $PythonW"
Write-Host "  Launcher: $LauncherPath"

$WshShell = New-Object -ComObject WScript.Shell
$Shortcut = $WshShell.CreateShortcut($ShortcutPath)
$Shortcut.TargetPath = $PythonW
$Shortcut.Arguments = "`"$LauncherPath`""
$Shortcut.WorkingDirectory = $ProjectRoot
$Shortcut.Description = "Nexus Intelligence Platform"
$Shortcut.Save()

Write-Host "Shortcut updated successfully!" -ForegroundColor Green
