# Create NEW Desktop Shortcut for Nexus Intelligence
# This creates a fresh shortcut using the batch launcher

$ProjectRoot = "J:\dev\nexus"
$LauncherPath = Join-Path $ProjectRoot "scripts\launch_nexus_debug.bat"
$IconPath = Join-Path $ProjectRoot "assets\nexus.ico"
$DesktopPath = [Environment]::GetFolderPath("Desktop")
$ShortcutPath = Join-Path $DesktopPath "Nexus.lnk"

Write-Host "Creating Nexus desktop shortcut..." -ForegroundColor Cyan
Write-Host "  Launcher: $LauncherPath"
Write-Host "  Icon: $IconPath"
Write-Host "  Shortcut: $ShortcutPath"

# Create the shortcut
$WshShell = New-Object -ComObject WScript.Shell
$Shortcut = $WshShell.CreateShortcut($ShortcutPath)
$Shortcut.TargetPath = $LauncherPath
$Shortcut.WorkingDirectory = $ProjectRoot
$Shortcut.Description = "Nexus Intelligence Platform"
$Shortcut.WindowStyle = 1

# Set icon
if (Test-Path $IconPath) {
    $Shortcut.IconLocation = "$IconPath,0"
    Write-Host "  Using custom Nexus icon" -ForegroundColor Green
} else {
    Write-Host "  Icon not found, using default" -ForegroundColor Yellow
}

$Shortcut.Save()

Write-Host ""
Write-Host "Shortcut created successfully!" -ForegroundColor Green
Write-Host "Look for 'Nexus' on your desktop."
