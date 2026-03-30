"""
Automatic model unloading under memory pressure.

Monitors system memory and automatically unloads least-recently-used
models when memory pressure exceeds configurable thresholds. Prevents
OOM kills in multi-model deployments.
"""

import gc
import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class PressureLevel(str, Enum):
    NORMAL = "normal"
    ELEVATED = "elevated"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class LoadedModel:
    """Tracks a loaded model's memory usage and access pattern."""
    model_name: str
    memory_bytes: int = 0
    loaded_at: float = field(default_factory=time.time)
    last_used_at: float = field(default_factory=time.time)
    use_count: int = 0
    priority: int = 0  # Higher = less likely to be unloaded
    unload_fn: Optional[Callable] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def memory_mb(self) -> float:
        return self.memory_bytes / (1024 * 1024)

    @property
    def idle_seconds(self) -> float:
        return time.time() - self.last_used_at


@dataclass
class UnloadEvent:
    """Record of a model unload event."""
    model_name: str
    memory_freed_bytes: int
    reason: str
    pressure_level: PressureLevel
    timestamp: float = field(default_factory=time.time)


class MemoryPressureMonitor:
    """
    Monitors system memory and determines pressure level.
    """

    def __init__(
        self,
        elevated_threshold: float = 0.70,
        high_threshold: float = 0.85,
        critical_threshold: float = 0.95,
    ):
        self._elevated = elevated_threshold
        self._high = high_threshold
        self._critical = critical_threshold

    def get_memory_usage(self) -> float:
        """Get current memory usage as fraction 0.0-1.0."""
        try:
            import psutil
            return psutil.virtual_memory().percent / 100.0
        except ImportError:
            # Fallback: read from /proc/meminfo on Linux
            try:
                with open("/proc/meminfo") as f:
                    info = {}
                    for line in f:
                        parts = line.split()
                        if len(parts) >= 2:
                            info[parts[0].rstrip(":")] = int(parts[1])
                    total = info.get("MemTotal", 1)
                    available = info.get("MemAvailable", total)
                    return 1.0 - (available / total)
            except (FileNotFoundError, ValueError, ZeroDivisionError):
                return 0.0

    def get_pressure_level(self) -> PressureLevel:
        """Determine current memory pressure level."""
        usage = self.get_memory_usage()
        if usage >= self._critical:
            return PressureLevel.CRITICAL
        elif usage >= self._high:
            return PressureLevel.HIGH
        elif usage >= self._elevated:
            return PressureLevel.ELEVATED
        return PressureLevel.NORMAL

    def get_available_bytes(self) -> int:
        """Get available memory in bytes."""
        try:
            import psutil
            return psutil.virtual_memory().available
        except ImportError:
            return 0


class AutoModelUnloader:
    """
    Automatically unloads models when memory pressure is detected.

    Features:
    - Configurable pressure thresholds
    - LRU-based unloading (least recently used first)
    - Priority-aware (protected models unloaded last)
    - Configurable minimum model count (never unload below this)
    - Periodic monitoring with configurable interval
    - Unload events for observability
    """

    def __init__(
        self,
        monitor: Optional[MemoryPressureMonitor] = None,
        check_interval: float = 30.0,
        min_models: int = 1,
        unload_batch_size: int = 1,
        on_unload: Optional[Callable[[UnloadEvent], None]] = None,
    ):
        """
        Args:
            monitor: Memory pressure monitor
            check_interval: Seconds between pressure checks
            min_models: Minimum models to keep loaded
            unload_batch_size: Models to unload per pressure check
            on_unload: Callback when a model is unloaded
        """
        self._monitor = monitor or MemoryPressureMonitor()
        self._check_interval = check_interval
        self._min_models = min_models
        self._unload_batch_size = unload_batch_size
        self._on_unload = on_unload
        self._models: Dict[str, LoadedModel] = {}
        self._events: List[UnloadEvent] = []
        self._timer: Optional[threading.Timer] = None
        self._running = False
        self._lock = threading.Lock()
        self._stats = {
            "checks": 0,
            "models_unloaded": 0,
            "memory_freed_bytes": 0,
            "pressure_events": 0,
        }

    def register_model(
        self,
        model_name: str,
        memory_bytes: int,
        unload_fn: Optional[Callable] = None,
        priority: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> LoadedModel:
        """
        Register a loaded model for memory management.

        Args:
            model_name: Model identifier
            memory_bytes: Approximate memory usage
            unload_fn: Callable to unload the model
            priority: Higher = less likely to be unloaded
            metadata: Additional info
        """
        model = LoadedModel(
            model_name=model_name,
            memory_bytes=memory_bytes,
            unload_fn=unload_fn,
            priority=priority,
            metadata=metadata or {},
        )
        with self._lock:
            self._models[model_name] = model
        logger.info("Registered model %s (%.1fMB)", model_name, model.memory_mb)
        return model

    def mark_used(self, model_name: str) -> None:
        """Mark a model as recently used."""
        model = self._models.get(model_name)
        if model:
            model.last_used_at = time.time()
            model.use_count += 1

    def unregister_model(self, model_name: str) -> None:
        """Unregister a model (already unloaded externally)."""
        with self._lock:
            self._models.pop(model_name, None)

    def start(self) -> None:
        """Start periodic memory pressure monitoring."""
        self._running = True
        self._schedule_check()
        logger.info("AutoModelUnloader started (interval=%.0fs)", self._check_interval)

    def stop(self) -> None:
        """Stop monitoring."""
        self._running = False
        if self._timer:
            self._timer.cancel()
            self._timer = None

    def _schedule_check(self) -> None:
        if not self._running:
            return
        self._timer = threading.Timer(self._check_interval, self._check_and_unload)
        self._timer.daemon = True
        self._timer.start()

    def _check_and_unload(self) -> None:
        """Check memory pressure and unload if needed."""
        try:
            self._stats["checks"] += 1
            level = self._monitor.get_pressure_level()

            if level in (PressureLevel.HIGH, PressureLevel.CRITICAL):
                self._stats["pressure_events"] += 1
                count = self._unload_batch_size
                if level == PressureLevel.CRITICAL:
                    count = max(count * 2, 2)  # Unload more aggressively
                self._unload_lru(count, level)
        except Exception as e:
            logger.error("Auto-unloader check failed: %s", e)
        finally:
            self._schedule_check()

    def _unload_lru(self, count: int, pressure: PressureLevel) -> int:
        """Unload the least recently used models."""
        with self._lock:
            if len(self._models) <= self._min_models:
                return 0

            # Sort by priority (ascending) then last_used_at (ascending)
            candidates = sorted(
                self._models.values(),
                key=lambda m: (m.priority, m.last_used_at),
            )

            # Don't go below min_models
            max_unload = len(self._models) - self._min_models
            count = min(count, max_unload)

            unloaded = 0
            for model in candidates[:count]:
                try:
                    if model.unload_fn:
                        model.unload_fn()

                    event = UnloadEvent(
                        model_name=model.model_name,
                        memory_freed_bytes=model.memory_bytes,
                        reason=f"memory_pressure_{pressure.value}",
                        pressure_level=pressure,
                    )
                    self._events.append(event)

                    if self._on_unload:
                        self._on_unload(event)

                    self._stats["models_unloaded"] += 1
                    self._stats["memory_freed_bytes"] += model.memory_bytes
                    del self._models[model.model_name]
                    unloaded += 1

                    logger.info(
                        "Unloaded model %s (freed %.1fMB, pressure=%s)",
                        model.model_name, model.memory_mb, pressure.value,
                    )
                except Exception as e:
                    logger.error("Failed to unload model %s: %s", model.model_name, e)

            # Force GC after unloading
            if unloaded > 0:
                gc.collect()

            return unloaded

    def force_check(self) -> int:
        """Force an immediate pressure check and unload."""
        level = self._monitor.get_pressure_level()
        if level in (PressureLevel.HIGH, PressureLevel.CRITICAL):
            return self._unload_lru(self._unload_batch_size, level)
        return 0

    def get_loaded_models(self) -> Dict[str, Dict[str, Any]]:
        """Get information about currently loaded models."""
        return {
            name: {
                "memory_mb": m.memory_mb,
                "idle_seconds": m.idle_seconds,
                "use_count": m.use_count,
                "priority": m.priority,
            }
            for name, m in self._models.items()
        }

    def get_events(self, limit: int = 50) -> List[UnloadEvent]:
        return list(self._events[-limit:])

    def get_stats(self) -> Dict[str, Any]:
        return {
            **self._stats,
            "loaded_models": len(self._models),
            "total_model_memory_mb": sum(
                m.memory_bytes for m in self._models.values()
            ) / (1024 * 1024),
            "current_pressure": self._monitor.get_pressure_level().value,
        }
