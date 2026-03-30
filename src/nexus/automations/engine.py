"""
Automation Engine - Central orchestrator for all automations
"""

import asyncio
import inspect
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
from enum import Enum
from pathlib import Path

from .scheduler import Scheduler, Schedule, ScheduleType
from .triggers import TriggerManager, Trigger, TriggerType, TriggerEvent
from .integrations import (
    GmailConnector, GmailConfig,
    NotionConnector, NotionConfig,
    FileWatcher, FileWatcherConfig
)


class AutomationStatus(Enum):
    """Status of an automation."""
    ACTIVE = "active"
    PAUSED = "paused"
    DISABLED = "disabled"
    ERROR = "error"


@dataclass
class AutomationRun:
    """Record of an automation execution."""
    id: str
    automation_name: str
    trigger_event: Optional[TriggerEvent]
    started_at: datetime
    completed_at: Optional[datetime] = None
    success: bool = False
    output: Any = None
    error: Optional[str] = None


@dataclass
class Automation:
    """An automation definition."""
    name: str
    description: str
    trigger_type: TriggerType
    action: Callable
    enabled: bool = True
    filters: Dict[str, Any] = field(default_factory=dict)
    
    # Schedule (if time-based)
    schedule_type: Optional[ScheduleType] = None
    schedule_config: Dict[str, Any] = field(default_factory=dict)
    
    # Tracking
    status: AutomationStatus = AutomationStatus.ACTIVE
    run_count: int = 0
    last_run: Optional[datetime] = None
    last_error: Optional[str] = None


class AutomationEngine:
    """
    Central automation engine.
    
    Manages:
    - Scheduled automations
    - Triggered automations
    - Integration connectors
    - Execution history
    
    Usage:
        engine = AutomationEngine(platform)
        await engine.initialize()
        
        engine.register_automation(Automation(
            name="email_triage",
            trigger_type=TriggerType.EMAIL_RECEIVED,
            action=my_email_handler
        ))
        
        engine.start()
    """
    
    def __init__(self, platform=None, config_path: str = None):
        self.platform = platform
        self.config_path = config_path or "automations_config.json"
        
        # Core components
        self.scheduler = Scheduler()
        self.triggers = TriggerManager()
        
        # Integrations
        self.gmail: Optional[GmailConnector] = None
        self.notion: Optional[NotionConnector] = None
        self.file_watcher: Optional[FileWatcher] = None
        
        # Automations
        self._automations: Dict[str, Automation] = {}
        self._runs: List[AutomationRun] = []
        self._running = False
    
    async def initialize(self, config: Dict[str, Any] = None) -> bool:
        """Initialize engine and integrations."""
        config = config or {}
        
        # Initialize Gmail if configured
        if "gmail" in config:
            gmail_config = GmailConfig(**config["gmail"])
            self.gmail = GmailConnector(gmail_config, self.triggers)
            await self.gmail.initialize()
        
        # Initialize Notion if configured
        if "notion" in config:
            notion_config = NotionConfig(**config["notion"])
            self.notion = NotionConnector(notion_config, self.triggers)
            await self.notion.initialize()
        
        # Initialize file watcher if configured
        if "file_watcher" in config:
            watcher_config = FileWatcherConfig(**config["file_watcher"])
            self.file_watcher = FileWatcher(watcher_config, self.triggers)
        
        return True
    
    def register_automation(self, automation: Automation):
        """Register an automation."""
        self._automations[automation.name] = automation
        
        # Set up trigger
        trigger = Trigger(
            name=f"trigger_{automation.name}",
            trigger_type=automation.trigger_type,
            callback=lambda event: self._execute_automation(automation, event),
            enabled=automation.enabled,
            filters=automation.filters
        )
        self.triggers.register_trigger(trigger)
        
        # Set up schedule if time-based
        if automation.schedule_type:
            schedule = Schedule(
                name=f"schedule_{automation.name}",
                schedule_type=automation.schedule_type,
                callback=lambda: self._execute_scheduled(automation),
                enabled=automation.enabled,
                **automation.schedule_config
            )
            self.scheduler.add_schedule(schedule)
    
    def unregister_automation(self, name: str):
        """Unregister an automation."""
        if name in self._automations:
            del self._automations[name]
            self.triggers.unregister_trigger(f"trigger_{name}")
            self.scheduler.remove_schedule(f"schedule_{name}")
    
    def enable_automation(self, name: str, enabled: bool = True):
        """Enable or disable an automation."""
        if name in self._automations:
            self._automations[name].enabled = enabled
            self._automations[name].status = (
                AutomationStatus.ACTIVE if enabled else AutomationStatus.PAUSED
            )
            self.triggers.enable_trigger(f"trigger_{name}", enabled)
            self.scheduler.enable_schedule(f"schedule_{name}", enabled)
    
    def start(self):
        """Start the automation engine."""
        if self._running:
            return
        
        self._running = True
        
        # Start core components
        self.scheduler.start()
        self.triggers.start()
        
        # Start integrations
        if self.gmail and self.gmail.is_configured():
            self.gmail.start_watching()
        
        if self.notion and self.notion.is_configured():
            self.notion.start_watching()
        
        if self.file_watcher:
            self.file_watcher.start_watching()
    
    def stop(self):
        """Stop the automation engine."""
        self._running = False
        
        self.scheduler.stop()
        self.triggers.stop()
        
        if self.gmail:
            self.gmail.stop_watching()
        
        if self.notion:
            self.notion.stop_watching()
        
        if self.file_watcher:
            self.file_watcher.stop_watching()

    def _execute_automation(self, automation: Automation, event: TriggerEvent):
        """Execute an automation from trigger."""
        run = AutomationRun(
            id=str(uuid.uuid4())[:8],
            automation_name=automation.name,
            trigger_event=event,
            started_at=datetime.now()
        )
        
        try:
            # Execute action
            if inspect.iscoroutinefunction(automation.action):
                # Get or create event loop
                try:
                    loop = asyncio.get_running_loop()
                    # If we're in an async context, schedule it
                    future = asyncio.run_coroutine_threadsafe(
                        automation.action(event, self.platform), loop
                    )
                    result = future.result(timeout=300)
                except RuntimeError:
                    # No running loop, create one
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        result = loop.run_until_complete(automation.action(event, self.platform))
                    finally:
                        loop.close()
            else:
                result = automation.action(event, self.platform)
            
            run.success = True
            run.output = result
            automation.run_count += 1
            automation.last_run = datetime.now()
            automation.status = AutomationStatus.ACTIVE
            
        except Exception as e:
            run.success = False
            run.error = str(e)
            automation.last_error = str(e)
            automation.status = AutomationStatus.ERROR
        
        run.completed_at = datetime.now()
        self._runs.append(run)
        
        # Limit history
        if len(self._runs) > 500:
            self._runs = self._runs[-500:]
    
    def _execute_scheduled(self, automation: Automation):
        """Execute a scheduled automation."""
        event = TriggerEvent(
            trigger_name=f"schedule_{automation.name}",
            trigger_type=TriggerType.MANUAL,
            timestamp=datetime.now(),
            source="scheduler"
        )
        self._execute_automation(automation, event)
    
    def fire_manual(self, automation_name: str, data: Dict[str, Any] = None):
        """Manually fire an automation."""
        if automation_name not in self._automations:
            return False
        
        event = TriggerEvent(
            trigger_name=f"manual_{automation_name}",
            trigger_type=TriggerType.MANUAL,
            timestamp=datetime.now(),
            data=data or {},
            source="manual"
        )
        
        self._execute_automation(self._automations[automation_name], event)
        return True
    
    def get_automation(self, name: str) -> Optional[Automation]:
        """Get an automation by name."""
        return self._automations.get(name)
    
    def get_automations(self) -> List[Automation]:
        """Get all automations."""
        return list(self._automations.values())
    
    def get_runs(self, automation_name: str = None, limit: int = 50) -> List[AutomationRun]:
        """Get recent runs, optionally filtered by automation."""
        runs = self._runs
        if automation_name:
            runs = [r for r in runs if r.automation_name == automation_name]
        return runs[-limit:]
    
    def get_status(self) -> Dict[str, Any]:
        """Get engine status."""
        return {
            "running": self._running,
            "automations": {
                name: {
                    "status": a.status.value,
                    "enabled": a.enabled,
                    "trigger": a.trigger_type.value,
                    "run_count": a.run_count,
                    "last_run": a.last_run.isoformat() if a.last_run else None,
                }
                for name, a in self._automations.items()
            },
            "scheduler": self.scheduler.get_status(),
            "triggers": self.triggers.get_status(),
            "integrations": {
                "gmail": self.gmail.is_configured() if self.gmail else False,
                "notion": self.notion.is_configured() if self.notion else False,
                "file_watcher": self.file_watcher.is_running() if self.file_watcher else False,
            },
            "total_runs": len(self._runs),
        }


# ============================================================
# Pre-built Automation Factories
# ============================================================

def create_email_triage_automation(
    email_pipeline,
    user_context: str = "professional"
) -> Automation:
    """Create email triage automation."""
    
    async def handle_emails(event: TriggerEvent, platform):
        from workflows.email_triage import EmailItem
        
        email_data = event.data
        email = EmailItem(
            id=email_data["id"],
            sender=email_data["sender"],
            subject=email_data["subject"],
            body=email_data["body"],
            received_at=email_data["received_at"]
        )
        
        return await email_pipeline.run({
            "emails": [email],
            "user_context": user_context
        })
    
    return Automation(
        name="email_triage",
        description="Automatically triage incoming emails",
        trigger_type=TriggerType.EMAIL_RECEIVED,
        action=handle_emails
    )


def create_daily_email_digest(
    email_pipeline,
    user_context: str = "professional"
) -> Automation:
    """Create daily email digest automation."""
    
    async def generate_digest(event: TriggerEvent, platform):
        # This would fetch all unread emails and create digest
        # For now, placeholder
        return {"status": "digest_generated"}
    
    return Automation(
        name="daily_email_digest",
        description="Generate daily email digest at 9 AM",
        trigger_type=TriggerType.MANUAL,  # Triggered by schedule
        action=generate_digest,
        schedule_type=ScheduleType.DAILY,
        schedule_config={"daily_hour": 9, "daily_minute": 0}
    )


def create_file_processor(
    processor_fn: Callable,
    watch_patterns: List[str] = None
) -> Automation:
    """Create file processing automation."""
    
    async def handle_file(event: TriggerEvent, platform):
        file_path = event.data.get("path")
        return await processor_fn(file_path, platform)
    
    return Automation(
        name="file_processor",
        description="Process new files automatically",
        trigger_type=TriggerType.FILE_CREATED,
        action=handle_file,
        filters={"patterns": watch_patterns or ["*.pdf", "*.docx", "*.txt"]}
    )


def create_content_pipeline_trigger(
    ebook_pipeline,
    input_folder: str,
    output_folder: str
) -> Automation:
    """Create content pipeline automation for new content files."""
    
    async def handle_content_file(event: TriggerEvent, platform):
        file_path = event.data.get("path")
        filename = event.data.get("filename", "")
        
        # Read topic from file
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Parse topic and audience from file
            lines = content.strip().split('\n')
            topic = lines[0] if lines else filename
            audience = lines[1] if len(lines) > 1 else "general readers"
            
            result = await ebook_pipeline.run({
                "topic": topic,
                "audience": audience,
                "tone": "professional"
            })
            
            return result
            
        except Exception as e:
            return {"error": str(e)}
    
    return Automation(
        name="content_pipeline",
        description="Auto-generate ebook from content brief files",
        trigger_type=TriggerType.FILE_CREATED,
        action=handle_content_file,
        filters={"path": lambda p: input_folder in str(p)}
    )
