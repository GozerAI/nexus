"""
Automations Module - Built-in automation engine

Provides:
- Scheduler for time-based automation
- Triggers for event-based automation
- Smart router for folder-based automation
- Direct integrations (Gmail, Notion, Files)
- Pre-built automation factories
"""

from .engine import AutomationEngine, Automation, AutomationStatus
from .scheduler import Scheduler, Schedule, ScheduleType
from .triggers import TriggerManager, Trigger, TriggerType, TriggerEvent
from .integrations import GmailConnector, NotionConnector, FileWatcher
from .router import SmartRouter, WorkflowType, ProcessingJob

# Pre-built automation factories
from .engine import (
    create_email_triage_automation,
    create_daily_email_digest,
    create_file_processor,
    create_content_pipeline_trigger,
)

from .factories import (
    create_research_report_automation,
    create_meeting_notes_automation,
    create_content_repurpose_automation,
    create_weekly_digest_automation,
    create_proposal_automation,
    create_daily_standup_reminder,
)

__all__ = [
    # Core
    "AutomationEngine",
    "Automation",
    "AutomationStatus",
    # Scheduler
    "Scheduler",
    "Schedule",
    "ScheduleType",
    # Triggers
    "TriggerManager",
    "Trigger",
    "TriggerType",
    "TriggerEvent",
    # Router
    "SmartRouter",
    "WorkflowType",
    "ProcessingJob",
    # Integrations
    "GmailConnector",
    "NotionConnector",
    "FileWatcher",
    # Factories
    "create_email_triage_automation",
    "create_daily_email_digest",
    "create_file_processor",
    "create_content_pipeline_trigger",
    "create_research_report_automation",
    "create_meeting_notes_automation",
    "create_content_repurpose_automation",
    "create_weekly_digest_automation",
    "create_proposal_automation",
    "create_daily_standup_reminder",
]
