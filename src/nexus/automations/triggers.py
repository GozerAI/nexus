"""
Triggers - Event-based automation triggers
"""

import asyncio
import inspect
from dataclasses import dataclass, field
from typing import Dict, List, Callable, Any, Optional
from datetime import datetime
from enum import Enum
import threading
import queue


class TriggerType(Enum):
    """Types of triggers."""
    FILE_CREATED = "file_created"
    FILE_MODIFIED = "file_modified"
    EMAIL_RECEIVED = "email_received"
    WEBHOOK = "webhook"
    MANUAL = "manual"
    THRESHOLD = "threshold"
    PATTERN = "pattern"


@dataclass
class TriggerEvent:
    """An event that triggered automation."""
    trigger_name: str
    trigger_type: TriggerType
    timestamp: datetime
    data: Dict[str, Any] = field(default_factory=dict)
    source: str = ""


@dataclass
class Trigger:
    """A trigger definition."""
    name: str
    trigger_type: TriggerType
    callback: Callable[[TriggerEvent], Any]
    enabled: bool = True
    
    # Filter conditions
    filters: Dict[str, Any] = field(default_factory=dict)
    
    # Tracking
    fire_count: int = 0
    last_fired: Optional[datetime] = None


class TriggerManager:
    """
    Manages event-based triggers.
    
    Receives events from integrations and dispatches to registered callbacks.
    """
    
    def __init__(self):
        self._triggers: Dict[str, Trigger] = {}
        self._event_queue: queue.Queue = queue.Queue()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
    
    def register_trigger(self, trigger: Trigger):
        """Register a trigger."""
        with self._lock:
            self._triggers[trigger.name] = trigger
    
    def unregister_trigger(self, name: str):
        """Unregister a trigger."""
        with self._lock:
            self._triggers.pop(name, None)
    
    def enable_trigger(self, name: str, enabled: bool = True):
        """Enable or disable a trigger."""
        with self._lock:
            if name in self._triggers:
                self._triggers[name].enabled = enabled
    
    def fire_event(self, event: TriggerEvent):
        """Fire an event to be processed."""
        self._event_queue.put(event)
    
    def fire(
        self,
        trigger_type: TriggerType,
        data: Dict[str, Any] = None,
        source: str = ""
    ):
        """Convenience method to fire an event."""
        event = TriggerEvent(
            trigger_name="",  # Will match by type
            trigger_type=trigger_type,
            timestamp=datetime.now(),
            data=data or {},
            source=source
        )
        self.fire_event(event)
    
    def start(self):
        """Start the trigger processing thread."""
        if self._running:
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._process_loop, daemon=True)
        self._thread.start()
    
    def stop(self):
        """Stop trigger processing."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
    
    def _process_loop(self):
        """Process events from queue."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        
        while self._running:
            try:
                event = self._event_queue.get(timeout=1)
                self._process_event(event)
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Trigger processing error: {e}")
        
        self._loop.close()
    
    def _process_event(self, event: TriggerEvent):
        """Process a single event."""
        with self._lock:
            triggers = list(self._triggers.values())
        
        for trigger in triggers:
            if not trigger.enabled:
                continue
            
            # Match by type
            if trigger.trigger_type != event.trigger_type:
                continue
            
            # Check filters
            if not self._matches_filters(event, trigger.filters):
                continue
            
            # Execute callback
            try:
                event.trigger_name = trigger.name
                
                if inspect.iscoroutinefunction(trigger.callback):
                    self._loop.run_until_complete(trigger.callback(event))
                else:
                    trigger.callback(event)
                
                trigger.fire_count += 1
                trigger.last_fired = datetime.now()
                
            except Exception as e:
                print(f"Trigger {trigger.name} callback failed: {e}")
    
    def _matches_filters(self, event: TriggerEvent, filters: Dict[str, Any]) -> bool:
        """Check if event matches trigger filters."""
        for key, value in filters.items():
            event_value = event.data.get(key)
            
            if callable(value):
                if not value(event_value):
                    return False
            elif event_value != value:
                return False
        
        return True
    
    def get_triggers(self) -> List[Trigger]:
        """Get all registered triggers."""
        with self._lock:
            return list(self._triggers.values())
    
    def get_status(self) -> Dict[str, Any]:
        """Get trigger manager status."""
        with self._lock:
            return {
                "running": self._running,
                "queue_size": self._event_queue.qsize(),
                "triggers": [
                    {
                        "name": t.name,
                        "type": t.trigger_type.value,
                        "enabled": t.enabled,
                        "fire_count": t.fire_count,
                        "last_fired": t.last_fired.isoformat() if t.last_fired else None,
                    }
                    for t in self._triggers.values()
                ]
            }
