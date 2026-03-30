"""
Additional Automation Factories - More pre-built automations
"""

from typing import List, Dict, Any, Callable
from pathlib import Path

from .engine import Automation
from .triggers import TriggerType, TriggerEvent
from .scheduler import ScheduleType


def create_research_report_automation(
    research_pipeline,
    watch_folder: str = None
) -> Automation:
    """
    Create research report automation.
    
    Triggers on new .research files in watch folder.
    File format:
        Line 1: Topic
        Line 2: Context
        Lines 3+: Questions (one per line)
    """
    
    async def handle_research_request(event: TriggerEvent, platform):
        file_path = event.data.get("path", "")
        
        if not file_path.endswith(".research"):
            return {"skipped": True, "reason": "Not a .research file"}
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.read().strip().split('\n')
            
            topic = lines[0] if lines else "General Research"
            context = lines[1] if len(lines) > 1 else ""
            questions = lines[2:] if len(lines) > 2 else []
            
            return await research_pipeline.run({
                "topic": topic,
                "context": context,
                "questions": questions
            })
        except Exception as e:
            return {"error": str(e)}
    
    return Automation(
        name="research_report",
        description="Generate research reports from .research files",
        trigger_type=TriggerType.FILE_CREATED,
        action=handle_research_request,
        filters={"path": lambda p: str(p).endswith(".research")} if not watch_folder else {}
    )


def create_meeting_notes_automation(
    meeting_pipeline,
    watch_folder: str = None
) -> Automation:
    """
    Create meeting notes automation.
    
    Triggers on new .transcript or .meeting files.
    Can also process .txt files with "meeting" or "transcript" in name.
    """
    
    async def handle_transcript(event: TriggerEvent, platform):
        file_path = event.data.get("path", "")
        filename = event.data.get("filename", "")
        
        # Check if it's a meeting/transcript file
        valid_extensions = ['.transcript', '.meeting', '.txt']
        valid_keywords = ['meeting', 'transcript', 'notes', 'call']
        
        is_valid = any(file_path.endswith(ext) for ext in valid_extensions)
        has_keyword = any(kw in filename.lower() for kw in valid_keywords)
        
        if not (is_valid and has_keyword):
            return {"skipped": True, "reason": "Not a meeting transcript file"}
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                transcript = f.read()
            
            # Try to extract meeting title from filename
            title = Path(filename).stem.replace('_', ' ').replace('-', ' ').title()
            
            return await meeting_pipeline.run({
                "transcript": transcript,
                "meeting_title": title,
                "attendees": [],  # Would need to parse from transcript
                "meeting_date": "",
                "context": "Auto-processed meeting transcript"
            })
        except Exception as e:
            return {"error": str(e)}
    
    return Automation(
        name="meeting_notes",
        description="Process meeting transcripts into structured notes",
        trigger_type=TriggerType.FILE_CREATED,
        action=handle_transcript
    )


def create_content_repurpose_automation(
    repurpose_pipeline,
    watch_folder: str = None
) -> Automation:
    """
    Create content repurposing automation.
    
    Triggers on new .content files.
    File format:
        Line 1: Content type (article, blog, report, etc.)
        Line 2: Key message (optional)
        Line 3+: Source content
    """
    
    async def handle_content(event: TriggerEvent, platform):
        file_path = event.data.get("path", "")
        
        if not file_path.endswith(".content"):
            return {"skipped": True, "reason": "Not a .content file"}
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.read().strip().split('\n')
            
            content_type = lines[0] if lines else "article"
            key_message = lines[1] if len(lines) > 1 else ""
            source_content = '\n'.join(lines[2:]) if len(lines) > 2 else ""
            
            return await repurpose_pipeline.run({
                "source_content": source_content,
                "content_type": content_type,
                "key_message": key_message
            })
        except Exception as e:
            return {"error": str(e)}
    
    return Automation(
        name="content_repurposer",
        description="Transform content into multiple formats",
        trigger_type=TriggerType.FILE_CREATED,
        action=handle_content,
        filters={"path": lambda p: str(p).endswith(".content")}
    )


def create_weekly_digest_automation(
    digest_pipeline,
    schedule_day: int = 4,  # Friday
    schedule_hour: int = 16  # 4 PM
) -> Automation:
    """
    Create weekly digest automation.
    
    Scheduled to run every Friday at 4 PM.
    Reads accomplishments from a weekly log file.
    """
    
    async def generate_digest(event: TriggerEvent, platform):
        # Look for weekly log file
        log_path = Path.home() / "Documents" / "weekly_log.txt"
        
        accomplishments = []
        metrics = {}
        blockers = []
        next_week = []
        
        if log_path.exists():
            try:
                with open(log_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Simple parsing - each section starts with ##
                current_section = None
                for line in content.split('\n'):
                    line = line.strip()
                    if line.startswith('## '):
                        current_section = line[3:].lower()
                    elif line.startswith('- ') and current_section:
                        item = line[2:]
                        if 'accomplish' in current_section or 'done' in current_section:
                            accomplishments.append(item)
                        elif 'metric' in current_section:
                            if ':' in item:
                                k, v = item.split(':', 1)
                                metrics[k.strip()] = v.strip()
                        elif 'blocker' in current_section or 'risk' in current_section:
                            blockers.append(item)
                        elif 'next' in current_section or 'plan' in current_section:
                            next_week.append(item)
            except (IOError, UnicodeDecodeError) as e:
                import logging
                logging.getLogger(__name__).warning(f"Could not read weekly log: {e}")
        
        return await digest_pipeline.run({
            "accomplishments": accomplishments or ["Review weekly_log.txt for accomplishments"],
            "metrics": metrics,
            "blockers": blockers,
            "next_week": next_week,
            "recipient": "leadership"
        })
    
    return Automation(
        name="weekly_digest",
        description="Generate weekly status digest every Friday",
        trigger_type=TriggerType.MANUAL,
        action=generate_digest,
        schedule_type=ScheduleType.WEEKLY,
        schedule_config={
            "weekly_day": schedule_day,
            "daily_hour": schedule_hour,
            "daily_minute": 0
        }
    )


def create_proposal_automation(
    proposal_pipeline,
    watch_folder: str = None
) -> Automation:
    """
    Create proposal generation automation.
    
    Triggers on new .proposal files.
    File format (YAML-like):
        client: Client Name
        project: Project Name
        problem: Problem description
        solution: Solution brief
        differentiators:
            - Point 1
            - Point 2
        timeline: 3 months
        budget: $50k-$100k
    """
    
    async def handle_proposal_request(event: TriggerEvent, platform):
        file_path = event.data.get("path", "")
        
        if not file_path.endswith(".proposal"):
            return {"skipped": True, "reason": "Not a .proposal file"}
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Simple parsing
            inputs = {
                "client_name": "",
                "project_name": "",
                "problem": "",
                "solution_brief": "",
                "differentiators": [],
                "budget_range": "",
                "timeline": ""
            }
            
            current_key = None
            for line in content.split('\n'):
                line = line.strip()
                
                if ':' in line and not line.startswith('-'):
                    key, value = line.split(':', 1)
                    key = key.strip().lower()
                    value = value.strip()
                    
                    key_map = {
                        'client': 'client_name',
                        'project': 'project_name',
                        'problem': 'problem',
                        'solution': 'solution_brief',
                        'timeline': 'timeline',
                        'budget': 'budget_range'
                    }
                    
                    if key in key_map:
                        inputs[key_map[key]] = value
                        current_key = key_map[key]
                    elif key == 'differentiators':
                        current_key = 'differentiators'
                
                elif line.startswith('- ') and current_key == 'differentiators':
                    inputs['differentiators'].append(line[2:])
            
            return await proposal_pipeline.run(inputs)
        except Exception as e:
            return {"error": str(e)}
    
    return Automation(
        name="proposal_generator",
        description="Generate proposals from .proposal files",
        trigger_type=TriggerType.FILE_CREATED,
        action=handle_proposal_request,
        filters={"path": lambda p: str(p).endswith(".proposal")}
    )


def create_daily_standup_reminder(
    callback: Callable = None
) -> Automation:
    """
    Create daily standup reminder automation.
    
    Runs at 9 AM daily, reminds to log standup notes.
    """
    
    async def standup_reminder(event: TriggerEvent, platform):
        if callback:
            return await callback(event, platform)
        
        return {
            "message": "Time to log your daily standup!",
            "template": """
## Daily Standup - {date}

### Yesterday
- 

### Today
- 

### Blockers
- 
"""
        }
    
    return Automation(
        name="daily_standup",
        description="Daily standup reminder at 9 AM",
        trigger_type=TriggerType.MANUAL,
        action=standup_reminder,
        schedule_type=ScheduleType.DAILY,
        schedule_config={"daily_hour": 9, "daily_minute": 0}
    )
