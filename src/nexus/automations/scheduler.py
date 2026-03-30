"""
Scheduler - Time-based automation triggers
"""

import asyncio
import inspect
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Callable, Any, Optional
from datetime import datetime, timedelta
from enum import Enum
import threading

logger = logging.getLogger(__name__)


class ScheduleType(Enum):
    """Types of schedules."""
    INTERVAL = "interval"      # Every N minutes/hours
    DAILY = "daily"            # Once per day at specific time
    WEEKLY = "weekly"          # Once per week
    CRON = "cron"              # Cron expression (simplified)


@dataclass
class Schedule:
    """A schedule definition."""
    name: str
    schedule_type: ScheduleType
    callback: Callable
    enabled: bool = True
    
    # For INTERVAL
    interval_minutes: int = 60
    
    # For DAILY
    daily_hour: int = 9
    daily_minute: int = 0
    
    # For WEEKLY
    weekly_day: int = 0  # 0=Monday
    
    # Tracking
    last_run: Optional[datetime] = None
    next_run: Optional[datetime] = None
    run_count: int = 0
    
    def calculate_next_run(self) -> datetime:
        """Calculate next run time."""
        now = datetime.now()
        
        if self.schedule_type == ScheduleType.INTERVAL:
            if self.last_run:
                return self.last_run + timedelta(minutes=self.interval_minutes)
            return now
        
        elif self.schedule_type == ScheduleType.DAILY:
            next_run = now.replace(
                hour=self.daily_hour,
                minute=self.daily_minute,
                second=0,
                microsecond=0
            )
            if next_run <= now:
                next_run += timedelta(days=1)
            return next_run
        
        elif self.schedule_type == ScheduleType.WEEKLY:
            days_ahead = self.weekly_day - now.weekday()
            if days_ahead <= 0:
                days_ahead += 7
            next_run = now + timedelta(days=days_ahead)
            return next_run.replace(
                hour=self.daily_hour,
                minute=self.daily_minute,
                second=0
            )
        
        return now + timedelta(hours=1)


class Scheduler:
    """
    Time-based automation scheduler.
    
    Runs schedules in background thread, executes callbacks when due.
    """
    
    def __init__(self):
        self._schedules: Dict[str, Schedule] = {}
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
    
    def add_schedule(self, schedule: Schedule):
        """Add a schedule."""
        schedule.next_run = schedule.calculate_next_run()
        with self._lock:
            self._schedules[schedule.name] = schedule
    
    def remove_schedule(self, name: str):
        """Remove a schedule."""
        with self._lock:
            self._schedules.pop(name, None)
    
    def enable_schedule(self, name: str, enabled: bool = True):
        """Enable or disable a schedule."""
        with self._lock:
            if name in self._schedules:
                self._schedules[name].enabled = enabled
    
    def get_schedules(self) -> List[Schedule]:
        """Get all schedules."""
        with self._lock:
            return list(self._schedules.values())
    
    def start(self):
        """Start the scheduler background thread."""
        if self._running:
            return

        self._running = True
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop the scheduler."""
        self._running = False
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)

    def _run_loop(self):
        """Main scheduler loop."""
        # Create event loop for this thread
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

        while self._running:
            now = datetime.now()

            with self._lock:
                schedules = list(self._schedules.values())

            for schedule in schedules:
                if not schedule.enabled:
                    continue

                if schedule.next_run and schedule.next_run <= now:
                    self._execute_schedule(schedule)

            # Interruptible wait — returns immediately if stop() is called
            self._stop_event.wait(timeout=10)
            if self._stop_event.is_set():
                break

        self._loop.close()
    
    def _execute_schedule(self, schedule: Schedule):
        """Execute a scheduled callback."""
        try:
            if inspect.iscoroutinefunction(schedule.callback):
                self._loop.run_until_complete(schedule.callback())
            else:
                schedule.callback()
            
            schedule.last_run = datetime.now()
            schedule.run_count += 1
            schedule.next_run = schedule.calculate_next_run()
            
        except Exception as e:
            logger.error("Schedule %s failed: %s: %s", schedule.name, type(e).__name__, e)
    
    def get_status(self) -> Dict[str, Any]:
        """Get scheduler status."""
        with self._lock:
            return {
                "running": self._running,
                "schedules": [
                    {
                        "name": s.name,
                        "type": s.schedule_type.value,
                        "enabled": s.enabled,
                        "next_run": s.next_run.isoformat() if s.next_run else None,
                        "run_count": s.run_count,
                    }
                    for s in self._schedules.values()
                ]
            }
