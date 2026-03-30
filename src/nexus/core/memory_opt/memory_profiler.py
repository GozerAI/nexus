"""
Memory profiling hooks for runtime monitoring.

Provides lightweight memory profiling that can run in production
without significant overhead. Tracks memory usage over time,
detects leaks, and identifies high-memory components.
"""

import gc
import logging
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class MemorySnapshot:
    """A point-in-time memory measurement."""
    timestamp: float = field(default_factory=time.time)
    rss_bytes: int = 0         # Resident set size
    vms_bytes: int = 0         # Virtual memory size
    heap_bytes: int = 0        # Python heap allocated
    gc_objects: int = 0        # Tracked GC objects
    gc_collections: Tuple[int, int, int] = (0, 0, 0)  # gen0, gen1, gen2 collections
    label: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def rss_mb(self) -> float:
        return self.rss_bytes / (1024 * 1024)

    @property
    def vms_mb(self) -> float:
        return self.vms_bytes / (1024 * 1024)


@dataclass
class MemoryLeak:
    """Detected memory leak."""
    component: str
    growth_bytes_per_second: float
    observation_window_seconds: float
    start_bytes: int
    end_bytes: int
    confidence: float  # 0-1

    @property
    def growth_mb_per_hour(self) -> float:
        return (self.growth_bytes_per_second * 3600) / (1024 * 1024)


class MemoryProfiler:
    """
    Production-safe memory profiler.

    Features:
    - Periodic memory snapshots
    - Per-component memory tracking via context managers
    - Memory leak detection via trend analysis
    - GC statistics tracking
    - Configurable snapshot retention
    - Low-overhead design (<1% CPU impact)
    """

    DEFAULT_INTERVAL = 60.0  # seconds between snapshots
    MAX_SNAPSHOTS = 1440     # 24 hours at 1/minute

    def __init__(
        self,
        snapshot_interval: float = DEFAULT_INTERVAL,
        max_snapshots: int = MAX_SNAPSHOTS,
        leak_detection_window: float = 3600.0,  # 1 hour
        leak_threshold_mb_per_hour: float = 100.0,
        on_leak_detected: Optional[Callable[[MemoryLeak], None]] = None,
    ):
        self._interval = snapshot_interval
        self._max_snapshots = max_snapshots
        self._leak_window = leak_detection_window
        self._leak_threshold = leak_threshold_mb_per_hour
        self._on_leak = on_leak_detected
        self._snapshots: List[MemorySnapshot] = []
        self._component_baselines: Dict[str, int] = {}
        self._component_current: Dict[str, int] = {}
        self._timer: Optional[threading.Timer] = None
        self._running = False
        self._lock = threading.Lock()
        self._detected_leaks: List[MemoryLeak] = []

    def take_snapshot(self, label: str = "") -> MemorySnapshot:
        """Take an immediate memory snapshot."""
        snapshot = MemorySnapshot(
            rss_bytes=self._get_rss(),
            vms_bytes=self._get_vms(),
            heap_bytes=self._get_heap_size(),
            gc_objects=len(gc.get_objects()),
            gc_collections=tuple(gc.get_stats()[i]["collections"] for i in range(3)),
            label=label,
        )

        with self._lock:
            self._snapshots.append(snapshot)
            if len(self._snapshots) > self._max_snapshots:
                self._snapshots.pop(0)

        return snapshot

    def start_periodic(self) -> None:
        """Start periodic memory snapshots."""
        self._running = True
        self._schedule_next()
        logger.info("Memory profiler started (interval=%.0fs)", self._interval)

    def stop_periodic(self) -> None:
        """Stop periodic snapshots."""
        self._running = False
        if self._timer:
            self._timer.cancel()
            self._timer = None

    def _schedule_next(self) -> None:
        if not self._running:
            return
        self._timer = threading.Timer(self._interval, self._periodic_snapshot)
        self._timer.daemon = True
        self._timer.start()

    def _periodic_snapshot(self) -> None:
        try:
            self.take_snapshot(label="periodic")
            self._check_for_leaks()
        except Exception as e:
            logger.error("Memory profiler error: %s", e)
        finally:
            self._schedule_next()

    def track_component(self, component_name: str) -> "ComponentTracker":
        """
        Create a context manager to track memory usage of a component.

        Usage::

            with profiler.track_component("embedding_cache") as tracker:
                # ... do work ...
                pass
            print(f"Used {tracker.delta_mb:.1f}MB")
        """
        return ComponentTracker(self, component_name)

    def record_component_usage(self, component: str, bytes_used: int) -> None:
        """Manually record memory usage for a component."""
        self._component_current[component] = bytes_used

    def _check_for_leaks(self) -> None:
        """Analyze snapshots for memory leaks."""
        with self._lock:
            if len(self._snapshots) < 10:
                return

            cutoff = time.time() - self._leak_window
            recent = [s for s in self._snapshots if s.timestamp >= cutoff]
            if len(recent) < 5:
                return

            # Linear regression on RSS
            first = recent[0]
            last = recent[-1]
            duration = last.timestamp - first.timestamp
            if duration < 60:
                return

            growth = last.rss_bytes - first.rss_bytes
            rate = growth / duration  # bytes per second
            rate_mb_hour = (rate * 3600) / (1024 * 1024)

            if rate_mb_hour > self._leak_threshold:
                # Compute R^2 to check if growth is consistent
                n = len(recent)
                times = [s.timestamp - first.timestamp for s in recent]
                values = [s.rss_bytes for s in recent]
                mean_t = sum(times) / n
                mean_v = sum(values) / n
                ss_tt = sum((t - mean_t) ** 2 for t in times)
                ss_vv = sum((v - mean_v) ** 2 for v in values)
                ss_tv = sum((t - mean_t) * (v - mean_v) for t, v in zip(times, values))
                r_squared = (ss_tv ** 2) / (ss_tt * ss_vv) if ss_tt > 0 and ss_vv > 0 else 0

                if r_squared > 0.7:
                    leak = MemoryLeak(
                        component="system",
                        growth_bytes_per_second=rate,
                        observation_window_seconds=duration,
                        start_bytes=first.rss_bytes,
                        end_bytes=last.rss_bytes,
                        confidence=r_squared,
                    )
                    self._detected_leaks.append(leak)
                    logger.warning(
                        "Memory leak detected: %.1f MB/hour (confidence=%.2f)",
                        rate_mb_hour, r_squared,
                    )
                    if self._on_leak:
                        self._on_leak(leak)

    @staticmethod
    def _get_rss() -> int:
        """Get resident set size in bytes."""
        try:
            import psutil
            return psutil.Process().memory_info().rss
        except ImportError:
            # Fallback: read from /proc on Linux
            try:
                with open(f"/proc/{os.getpid()}/status") as f:
                    for line in f:
                        if line.startswith("VmRSS:"):
                            return int(line.split()[1]) * 1024
            except (FileNotFoundError, ValueError):
                pass
        return 0

    @staticmethod
    def _get_vms() -> int:
        """Get virtual memory size in bytes."""
        try:
            import psutil
            return psutil.Process().memory_info().vms
        except ImportError:
            return 0

    @staticmethod
    def _get_heap_size() -> int:
        """Get approximate Python heap size."""
        gc.collect()
        total = 0
        for obj in gc.get_objects():
            try:
                total += sys.getsizeof(obj)
            except (TypeError, ReferenceError):
                pass
            if total > 100_000_000:  # Cap at ~100MB of scanning
                break
        return total

    def get_snapshots(self, limit: int = 100) -> List[MemorySnapshot]:
        with self._lock:
            return list(self._snapshots[-limit:])

    def get_detected_leaks(self) -> List[MemoryLeak]:
        return list(self._detected_leaks)

    def get_component_usage(self) -> Dict[str, int]:
        return dict(self._component_current)

    def get_stats(self) -> Dict[str, Any]:
        current = self.take_snapshot(label="stats_request")
        return {
            "current_rss_mb": current.rss_mb,
            "current_vms_mb": current.vms_mb,
            "gc_objects": current.gc_objects,
            "snapshots_count": len(self._snapshots),
            "detected_leaks": len(self._detected_leaks),
            "components_tracked": len(self._component_current),
        }


class ComponentTracker:
    """Context manager for tracking component memory usage."""

    def __init__(self, profiler: MemoryProfiler, component_name: str):
        self._profiler = profiler
        self._component = component_name
        self._start_rss = 0
        self._end_rss = 0

    def __enter__(self):
        self._start_rss = MemoryProfiler._get_rss()
        return self

    def __exit__(self, *args):
        self._end_rss = MemoryProfiler._get_rss()
        delta = self._end_rss - self._start_rss
        self._profiler.record_component_usage(self._component, max(delta, 0))

    @property
    def delta_bytes(self) -> int:
        return max(self._end_rss - self._start_rss, 0)

    @property
    def delta_mb(self) -> float:
        return self.delta_bytes / (1024 * 1024)
