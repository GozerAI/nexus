"""
Ollama Lifecycle Manager - Automatic startup and management of Ollama service.

This service ensures Ollama is always running in the background when the
Nexus Intelligence GUI is active. It handles:
1. Automatic startup of Ollama on GUI launch
2. Health monitoring and auto-restart
3. Graceful shutdown coordination
4. Model preloading for faster response times
"""

import asyncio
import logging
import os
import platform
import subprocess
import sys
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import aiohttp

logger = logging.getLogger(__name__)


class OllamaStatus(Enum):
    """Ollama service status."""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    ERROR = "error"
    UNKNOWN = "unknown"


class OllamaManager:
    """
    Manages the Ollama service lifecycle.

    Ensures Ollama runs in the background automatically when the GUI starts,
    monitors health, and handles graceful shutdown.
    """

    # Default Ollama installation paths by platform
    DEFAULT_PATHS = {
        "Windows": [
            Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Ollama" / "ollama.exe",
            Path("C:/Program Files/Ollama/ollama.exe"),
            Path.home() / "AppData" / "Local" / "Programs" / "Ollama" / "ollama.exe",
        ],
        "Darwin": [  # macOS
            Path("/usr/local/bin/ollama"),
            Path("/opt/homebrew/bin/ollama"),
            Path.home() / ".ollama" / "ollama",
        ],
        "Linux": [
            Path("/usr/local/bin/ollama"),
            Path("/usr/bin/ollama"),
            Path.home() / ".local" / "bin" / "ollama",
        ],
    }

    def __init__(
        self,
        host: str = "http://localhost:11434",
        auto_start: bool = True,
        health_check_interval: float = 30.0,
        startup_timeout: float = 60.0,
        preload_models: Optional[List[str]] = None,
    ):
        """
        Initialize Ollama manager.

        Args:
            host: Ollama API host URL
            auto_start: Whether to automatically start Ollama
            health_check_interval: Seconds between health checks
            startup_timeout: Maximum seconds to wait for Ollama to start
            preload_models: Models to preload on startup for faster responses
        """
        self.host = host
        self.auto_start = auto_start
        self.health_check_interval = health_check_interval
        self.startup_timeout = startup_timeout
        self.preload_models = preload_models or []

        self._status = OllamaStatus.UNKNOWN
        self._process: Optional[subprocess.Popen] = None
        self._health_task: Optional[asyncio.Task] = None
        self._status_callbacks: List[Callable[[OllamaStatus], None]] = []
        self._ollama_path: Optional[Path] = None
        self._started_by_us = False
        self._last_health_check: Optional[datetime] = None
        self._consecutive_failures = 0

        # Find Ollama executable
        self._ollama_path = self._find_ollama()

        logger.info(f"OllamaManager initialized (host={host}, auto_start={auto_start})")
        if self._ollama_path:
            logger.info(f"Found Ollama at: {self._ollama_path}")
        else:
            logger.warning("Ollama executable not found in default locations")

    def _find_ollama(self) -> Optional[Path]:
        """Find Ollama executable on the system."""
        system = platform.system()

        # Check PATH first
        try:
            if system == "Windows":
                result = subprocess.run(
                    ["where", "ollama"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
            else:
                result = subprocess.run(
                    ["which", "ollama"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )

            if result.returncode == 0:
                path = Path(result.stdout.strip().split('\n')[0])
                if path.exists():
                    return path
        except Exception:
            pass

        # Check default locations
        paths = self.DEFAULT_PATHS.get(system, [])
        for path in paths:
            if path.exists():
                return path

        return None

    @property
    def status(self) -> OllamaStatus:
        """Get current Ollama status."""
        return self._status

    @property
    def is_running(self) -> bool:
        """Check if Ollama is running."""
        return self._status == OllamaStatus.RUNNING

    def on_status_change(self, callback: Callable[[OllamaStatus], None]):
        """Register callback for status changes."""
        self._status_callbacks.append(callback)

    def _set_status(self, status: OllamaStatus):
        """Update status and notify callbacks."""
        if status != self._status:
            old_status = self._status
            self._status = status
            logger.info(f"Ollama status changed: {old_status.value} -> {status.value}")

            for callback in self._status_callbacks:
                try:
                    callback(status)
                except Exception as e:
                    logger.error(f"Status callback error: {e}")

    async def check_health(self) -> bool:
        """
        Check if Ollama is healthy and responding.

        Returns:
            True if Ollama is healthy
        """
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.host}/api/tags",
                    timeout=aiohttp.ClientTimeout(total=5),
                ) as response:
                    healthy = response.status == 200
                    self._last_health_check = datetime.now()

                    if healthy:
                        self._consecutive_failures = 0
                        self._set_status(OllamaStatus.RUNNING)
                    else:
                        self._consecutive_failures += 1
                        if self._consecutive_failures >= 3:
                            self._set_status(OllamaStatus.ERROR)

                    return healthy

        except Exception as e:
            self._consecutive_failures += 1
            logger.debug(f"Health check failed: {e}")

            if self._consecutive_failures >= 3:
                self._set_status(OllamaStatus.STOPPED)

            return False

    async def start(self) -> bool:
        """
        Start Ollama service.

        Returns:
            True if Ollama is running (was started or already running)
        """
        # Check if already running
        if await self.check_health():
            logger.info("Ollama is already running")
            return True

        if not self._ollama_path:
            logger.error("Cannot start Ollama: executable not found")
            self._set_status(OllamaStatus.ERROR)
            return False

        self._set_status(OllamaStatus.STARTING)
        logger.info(f"Starting Ollama from {self._ollama_path}")

        try:
            # Start Ollama serve in background
            if platform.system() == "Windows":
                # On Windows, use CREATE_NO_WINDOW to hide console
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                startupinfo.wShowWindow = subprocess.SW_HIDE

                self._process = subprocess.Popen(
                    [str(self._ollama_path), "serve"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    startupinfo=startupinfo,
                    creationflags=subprocess.CREATE_NO_WINDOW | subprocess.DETACHED_PROCESS,
                )
            else:
                # On Unix, use nohup-like behavior
                self._process = subprocess.Popen(
                    [str(self._ollama_path), "serve"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True,
                )

            self._started_by_us = True

            # Wait for Ollama to be ready
            start_time = datetime.now()
            while (datetime.now() - start_time).total_seconds() < self.startup_timeout:
                if await self.check_health():
                    logger.info("Ollama started successfully")

                    # Preload models if configured
                    if self.preload_models:
                        asyncio.create_task(self._preload_models())

                    return True

                await asyncio.sleep(1)

            logger.error(f"Ollama failed to start within {self.startup_timeout}s")
            self._set_status(OllamaStatus.ERROR)
            return False

        except Exception as e:
            logger.error(f"Failed to start Ollama: {e}")
            self._set_status(OllamaStatus.ERROR)
            return False

    async def _preload_models(self):
        """Preload configured models for faster response times."""
        for model in self.preload_models:
            try:
                logger.info(f"Preloading model: {model}")
                async with aiohttp.ClientSession() as session:
                    # Send a minimal generate request to load the model
                    async with session.post(
                        f"{self.host}/api/generate",
                        json={
                            "model": model,
                            "prompt": "Hi",
                            "stream": False,
                            "options": {"num_predict": 1}
                        },
                        timeout=aiohttp.ClientTimeout(total=120),
                    ) as response:
                        if response.status == 200:
                            logger.info(f"Model {model} preloaded successfully")
                        else:
                            logger.warning(f"Failed to preload {model}: {response.status}")
            except Exception as e:
                logger.warning(f"Failed to preload model {model}: {e}")

    async def stop(self, force: bool = False):
        """
        Stop Ollama service.

        Args:
            force: If True, force kill the process
        """
        if not self._started_by_us and not force:
            logger.info("Ollama was not started by us, leaving it running")
            return

        if self._process:
            logger.info("Stopping Ollama")
            try:
                if force:
                    self._process.kill()
                else:
                    self._process.terminate()

                # Wait for process to exit
                try:
                    self._process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self._process.kill()
                    self._process.wait()

            except Exception as e:
                logger.error(f"Error stopping Ollama: {e}")

            self._process = None
            self._started_by_us = False

        self._set_status(OllamaStatus.STOPPED)

    async def restart(self) -> bool:
        """
        Restart Ollama service.

        Returns:
            True if restart successful
        """
        await self.stop(force=True)
        await asyncio.sleep(2)
        return await self.start()

    async def start_health_monitoring(self):
        """Start background health monitoring task."""
        if self._health_task and not self._health_task.done():
            return

        self._health_task = asyncio.create_task(self._health_monitor_loop())
        logger.info("Started Ollama health monitoring")

    async def stop_health_monitoring(self):
        """Stop background health monitoring."""
        if self._health_task:
            self._health_task.cancel()
            try:
                await self._health_task
            except asyncio.CancelledError:
                pass
            self._health_task = None
            logger.info("Stopped Ollama health monitoring")

    async def _health_monitor_loop(self):
        """Background loop for health monitoring."""
        while True:
            try:
                await asyncio.sleep(self.health_check_interval)

                healthy = await self.check_health()

                # Auto-restart if unhealthy and we originally started it
                if not healthy and self._started_by_us and self.auto_start:
                    logger.warning("Ollama unhealthy, attempting restart")
                    await self.restart()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health monitor error: {e}")

    async def get_models(self) -> List[Dict[str, Any]]:
        """
        Get list of available models.

        Returns:
            List of model information dicts
        """
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.host}/api/tags",
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get("models", [])
                    return []
        except Exception as e:
            logger.error(f"Failed to get models: {e}")
            return []

    async def get_running_models(self) -> List[Dict[str, Any]]:
        """
        Get list of currently loaded models.

        Returns:
            List of running model information
        """
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.host}/api/ps",
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get("models", [])
                    return []
        except Exception as e:
            logger.error(f"Failed to get running models: {e}")
            return []

    def get_status_info(self) -> Dict[str, Any]:
        """
        Get detailed status information.

        Returns:
            Status information dict
        """
        return {
            "status": self._status.value,
            "is_running": self.is_running,
            "started_by_us": self._started_by_us,
            "ollama_path": str(self._ollama_path) if self._ollama_path else None,
            "host": self.host,
            "last_health_check": self._last_health_check.isoformat() if self._last_health_check else None,
            "consecutive_failures": self._consecutive_failures,
            "auto_start": self.auto_start,
        }


# Singleton instance for global access
_manager_instance: Optional[OllamaManager] = None


def get_ollama_manager(
    host: str = "http://localhost:11434",
    auto_start: bool = True,
    **kwargs
) -> OllamaManager:
    """
    Get or create the global OllamaManager instance.

    Args:
        host: Ollama API host
        auto_start: Whether to auto-start Ollama
        **kwargs: Additional OllamaManager arguments

    Returns:
        OllamaManager singleton instance
    """
    global _manager_instance

    if _manager_instance is None:
        _manager_instance = OllamaManager(
            host=host,
            auto_start=auto_start,
            **kwargs
        )

    return _manager_instance


async def ensure_ollama_running(
    host: str = "http://localhost:11434",
    preload_models: Optional[List[str]] = None,
) -> OllamaManager:
    """
    Ensure Ollama is running and return the manager.

    Convenience function that gets/creates the manager and ensures
    Ollama is started with health monitoring.

    Args:
        host: Ollama API host
        preload_models: Models to preload

    Returns:
        Running OllamaManager instance
    """
    manager = get_ollama_manager(
        host=host,
        auto_start=True,
        preload_models=preload_models,
    )

    await manager.start()
    await manager.start_health_monitoring()

    return manager
