# Redis Local Setup for Windows
# Run this script once to download and setup Redis
# Note: Uses community Redis Windows build (redis-windows project)

$RedisVersion = "8.0.4"
$RedisUrl = "https://github.com/redis-windows/redis-windows/releases/download/$RedisVersion/Redis-$RedisVersion-Windows-x64-msys2.zip"
$InstallDir = "$PSScriptRoot\..\tools\redis"
$DataDir = "$PSScriptRoot\..\data\redis"

Write-Host "============================================" -ForegroundColor Cyan
Write-Host " Redis Local Setup for Sentinel" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# Create directories
New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null
New-Item -ItemType Directory -Force -Path $DataDir | Out-Null

$ZipPath = "$InstallDir\redis.zip"
$ExtractPath = "$InstallDir\extracted"

# Check if already installed
if (Test-Path "$InstallDir\redis-server.exe") {
    Write-Host "Redis already installed at: $InstallDir" -ForegroundColor Green
} else {
    Write-Host "Downloading Redis $RedisVersion for Windows..." -ForegroundColor Yellow
    Write-Host "URL: $RedisUrl"
    Write-Host ""

    try {
        # Download Redis
        [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
        Invoke-WebRequest -Uri $RedisUrl -OutFile $ZipPath -UseBasicParsing

        Write-Host "Extracting..." -ForegroundColor Yellow
        Expand-Archive -Path $ZipPath -DestinationPath $ExtractPath -Force

        # Move contents to install dir (files are in a subdirectory)
        $ExtractedFolder = Get-ChildItem -Path $ExtractPath -Directory | Select-Object -First 1
        if ($ExtractedFolder) {
            Get-ChildItem -Path $ExtractedFolder.FullName | Move-Item -Destination $InstallDir -Force
        } else {
            # Files might be directly in extract path
            Get-ChildItem -Path $ExtractPath | Move-Item -Destination $InstallDir -Force
        }

        # Cleanup
        Remove-Item -Path $ZipPath -Force -ErrorAction SilentlyContinue
        Remove-Item -Path $ExtractPath -Recurse -Force -ErrorAction SilentlyContinue

        Write-Host "Redis installed successfully!" -ForegroundColor Green
    } catch {
        Write-Host "ERROR: Failed to download/install Redis" -ForegroundColor Red
        Write-Host $_.Exception.Message
        Write-Host ""
        Write-Host "Alternative options:" -ForegroundColor Yellow
        Write-Host "  1. Install via winget: winget install Redis.Redis"
        Write-Host "  2. Download manually from: https://github.com/redis-windows/redis-windows/releases"
        Write-Host ""
        Write-Host "Sentinel will still work without Redis using SQLite fallback."
        Read-Host "Press Enter to exit"
        exit 1
    }
}

# Create Redis config
$ConfigPath = "$InstallDir\redis-sentinel.conf"
$DataDirNorm = ($DataDir -replace '\\', '/')
$ConfigContent = @"
# Redis Configuration for Sentinel Intelligence

# Network
bind 127.0.0.1
port 6379
protected-mode yes

# Persistence - store data in project directory
dir $DataDirNorm
dbfilename sentinel-dump.rdb
appendonly yes
appendfilename "sentinel-appendonly.aof"

# Memory management
maxmemory 1gb
maxmemory-policy allkeys-lru

# Logging
loglevel notice
logfile "$DataDirNorm/redis.log"

# Performance
tcp-keepalive 300
timeout 0
"@

$ConfigContent | Out-File -FilePath $ConfigPath -Encoding utf8 -Force
Write-Host "Configuration saved to: $ConfigPath" -ForegroundColor Green

# Create start script
$StartScript = @"
@echo off
cd /d "%~dp0"
echo Starting Redis for Sentinel Intelligence...
echo.
echo Data directory: $DataDir
echo Config: redis-sentinel.conf
echo.
redis-server.exe redis-sentinel.conf
pause
"@

$StartScript | Out-File -FilePath "$InstallDir\start-redis.bat" -Encoding ascii -Force
Write-Host "Start script created: $InstallDir\start-redis.bat" -ForegroundColor Green

Write-Host ""
Write-Host "============================================" -ForegroundColor Cyan
Write-Host " Setup Complete!" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "To start Redis manually:" -ForegroundColor Yellow
Write-Host "  $InstallDir\start-redis.bat"
Write-Host ""
Write-Host "Redis will store data in: $DataDir" -ForegroundColor Green
Write-Host ""
Write-Host "NOTE: Sentinel uses SQLite+NumPy for vector search (works great locally)." -ForegroundColor Cyan
Write-Host "      Redis is used for caching and fast key-value storage." -ForegroundColor Cyan
Write-Host ""
Read-Host "Press Enter to exit"
